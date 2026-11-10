// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <stop_token>
#include <thread>
#include "gow3_toggles.h"
#include <ranges>
#include "common/assert.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"

#include <vk_mem_alloc.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include "common/thread.h"

namespace VideoCore {

using namespace Vulkan;

Common::IncrementalIdProvider<u64> Image::global_image_uid{};

static vk::ImageUsageFlags ImageUsageFlags(const Vulkan::Instance& instance,
                                           const ImageInfo& info) {
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc |
                                vk::ImageUsageFlagBits::eTransferDst |
                                vk::ImageUsageFlagBits::eSampled;
    if (!info.props.is_block) {
        if (info.props.is_depth) {
            usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
        } else {
            usage |= vk::ImageUsageFlagBits::eColorAttachment;
            if (instance.IsAttachmentFeedbackLoopLayoutSupported()) {
                usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
            }
            // Always create images with storage flag to avoid needing re-creation in case of e.g
            // compute clears This sacrifices a bit of performance but is less work. ExtendedUsage
            // flag is also used.
            usage |= vk::ImageUsageFlagBits::eStorage;
        }
    } else {
        // Similarly to above, we specify storage usage. This is typically not supported by
        // compressed formats, but may be used for uncompressed views. In order to satisfy this,
        // we will also specify the extended usage bit.
        usage |= vk::ImageUsageFlagBits::eStorage;
    }

    return usage;
}

static vk::ImageType ConvertImageType(AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
        return vk::ImageType::e1D;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DArray:
        return vk::ImageType::e2D;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE();
    }
}

static vk::FormatFeatureFlags2 FormatFeatureFlags(const vk::ImageUsageFlags usage_flags) {
    vk::FormatFeatureFlags2 feature_flags{};
    if (usage_flags & vk::ImageUsageFlagBits::eTransferSrc) {
        feature_flags |= vk::FormatFeatureFlagBits2::eTransferSrc;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eTransferDst) {
        feature_flags |= vk::FormatFeatureFlagBits2::eTransferDst;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eSampled) {
        feature_flags |= vk::FormatFeatureFlagBits2::eSampledImage;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eColorAttachment) {
        feature_flags |= vk::FormatFeatureFlagBits2::eColorAttachment;
    }
    if (usage_flags & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
        feature_flags |= vk::FormatFeatureFlagBits2::eDepthStencilAttachment;
    }
    // Note: StorageImage is intentionally ignored for now since it is always set, and can mess up
    // compatibility checks.
    return feature_flags;
}

// gow3: recent image creations and destructions, for the barrier dump in crash reports.
namespace {
struct ImageEvent {
    std::atomic<VkImage> image;
    std::atomic<bool> destroyed;
};
std::array<ImageEvent, 8192> image_events;
std::atomic<u64> image_event_head{0};

void NoteImage(VkImage image, bool destroyed) {
    ImageEvent& e = image_events[image_event_head.fetch_add(1, std::memory_order_relaxed) %
                                 image_events.size()];
    e.image.store(image, std::memory_order_relaxed);
    e.destroyed.store(destroyed, std::memory_order_relaxed);
}
} // namespace

const char* DescribeImageHandle(VkImage image) {
    if (!image) {
        return " (null)";
    }
    const u64 head = image_event_head.load(std::memory_order_relaxed);
    const u64 first = head > image_events.size() ? head - image_events.size() : 0;
    for (u64 i = head; i-- > first;) {
        const ImageEvent& e = image_events[i % image_events.size()];
        if (e.image.load(std::memory_order_relaxed) == image) {
            return e.destroyed.load(std::memory_order_relaxed) ? " (DESTROYED)" : " (live)";
        }
    }
    return "";
}

namespace {
// gow3: image memory is released on its own thread. Freeing an image can free a whole VMA block
// (vkFreeMemory), a kernel call that takes the GPU driver's lock; on the draw recording thread a
// GPU monitor's polling (MSI Afterburner's NVAPI escapes) held it up to 33 ms (research:
// plan-frame-pacing.md, "CPU oscillation", trace cpu-osc-1). An image is only destroyed once the
// GPU is done with it, so destroying it a little later is safe. GOW3_IMAGE_RELEASE_THREAD=0: inline.
class ImageReleaser {
public:
    static ImageReleaser& Get() {
        static ImageReleaser releaser;
        return releaser;
    }

    bool Enabled() const {
        return enabled;
    }

    void Release(VmaAllocator allocator, VkImage image, VmaAllocation allocation) {
        {
            std::scoped_lock lock{mutex};
            items.push_back({allocator, image, allocation});
            if (!worker.joinable()) {
                worker = std::jthread([this](std::stop_token stop) { Run(stop); });
            }
        }
        work_cv.notify_one();
    }

    /// Returns once every queued image is destroyed.
    void Drain() {
        std::unique_lock lock{mutex};
        idle_cv.wait(lock, [this] { return items.empty() && !busy; });
    }

private:
    struct Item {
        VmaAllocator allocator;
        VkImage image;
        VmaAllocation allocation;
    };

    ImageReleaser() {
        const char* env = std::getenv("GOW3_IMAGE_RELEASE_THREAD");
        enabled = !env || env[0] != '0';
    }

    void Run(std::stop_token stop) {
        Common::SetCurrentThreadName("gow3:ImageRelease");
        std::unique_lock lock{mutex};
        while (true) {
            work_cv.wait(lock, stop, [this] { return !items.empty(); });
            if (items.empty()) {
                return; // stop requested
            }
            const Item item = items.front();
            items.pop_front();
            busy = true;
            lock.unlock();
            vmaDestroyImage(item.allocator, item.image, item.allocation);
            lock.lock();
            busy = false;
            if (items.empty()) {
                idle_cv.notify_all();
            }
        }
    }

    bool enabled = true;
    std::mutex mutex;
    std::condition_variable_any work_cv;
    std::condition_variable_any idle_cv;
    std::deque<Item> items;
    bool busy = false;
    std::jthread worker;
};
} // namespace

void DrainImageReleases() {
    ImageReleaser::Get().Drain();
}

UniqueImage::~UniqueImage() {
    Destroy();
}

void UniqueImage::Destroy() {
    if (image) {
        NoteImage(image, true);
        if (ImageReleaser::Get().Enabled()) {
            ImageReleaser::Get().Release(allocator, image, allocation);
        } else {
            vmaDestroyImage(allocator, image, allocation);
        }
        image = vk::Image{};
        allocation = {};
    }
}

namespace {
/// From the Uncharted port (18 Oct 2026): a driver takes up to ~170 ms to allocate a new block of
/// video memory (256 MiB, AMD Windows), and VMA allocates one the moment an image no longer fits:
/// the GPU thread stalled for 70-240 ms there (hitches). This thread keeps room in reserve
/// instead: when a memory type images use has no free range of 64 MiB left, it allocates a 64 MiB
/// spare there (VMA makes a new block, with room for the next images) and frees the previous
/// spare. GOW3_SPARE_IMAGE_BLOCK=0: off.
class SpareBlockKeeper {
public:
    static void Note(VmaAllocator allocator, u32 memory_type) {
        static SpareBlockKeeper* const keeper = [allocator]() -> SpareBlockKeeper* {
            const char* env = std::getenv("GOW3_SPARE_IMAGE_BLOCK");
            return env && env[0] == '0' ? nullptr : new SpareBlockKeeper(allocator); // process-long
        }();
        if (keeper) {
            keeper->Wake(memory_type);
        }
    }

private:
    static constexpr VkDeviceSize Spare = VkDeviceSize{64} << 20;

    explicit SpareBlockKeeper(VmaAllocator allocator_) : allocator{allocator_} {
        std::thread{[this] { Run(); }}.detach();
    }

    void Wake(u32 memory_type) {
        {
            std::scoped_lock lock{mutex};
            types |= 1u << memory_type;
            pending = true;
        }
        cv.notify_one();
    }

    void Run() {
        Common::SetCurrentThreadName("GPU:SpareImageBlock");
        u32 made = 0;
        for (;;) {
            u32 wanted;
            {
                std::unique_lock lock{mutex};
                cv.wait(lock, [this] { return pending; });
                // Images come in bursts: act once one is over, so that the block is allocated
                // while the GPU thread creates no images (VMA holds the memory type's lock while
                // the driver allocates; an image created meanwhile waits for it).
                do {
                    pending = false;
                    cv.wait_for(lock, std::chrono::milliseconds{150}, [this] { return pending; });
                } while (pending);
                wanted = types;
            }
            VmaTotalStatistics stats{};
            vmaCalculateStatistics(allocator, &stats);
            for (u32 type = 0; type < VK_MAX_MEMORY_TYPES; ++type) {
                if (!(wanted >> type & 1) || stats.memoryType[type].unusedRangeSizeMax >= Spare) {
                    continue;
                }
                const VkMemoryRequirements requirements = {
                    .size = Spare, .alignment = 64 * 1024, .memoryTypeBits = 1u << type};
                const VmaAllocationCreateInfo create = {
                    .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
                    .usage = VMA_MEMORY_USAGE_UNKNOWN,
                    .memoryTypeBits = 1u << type,
                };
                // The spare stays allocated (a block that only held a freed spare would be
                // deleted by VMA at the next free); the previous one is freed once the new one
                // has its block, which opens room for images in an older block.
                VmaAllocation spare{};
                const auto started = std::chrono::steady_clock::now();
                if (vmaAllocateMemory(allocator, &requirements, &create, &spare, nullptr) ==
                    VK_SUCCESS) {
                    if (held[type]) {
                        vmaFreeMemory(allocator, held[type]);
                    }
                    held[type] = spare;
                    if (++made <= 16) {
                        std::printf("GPU: spare image memory block ready (type %u, %.0f ms)\n", type,
                                    std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - started).count());
                    }
                }
            }
            // At most one check per 20 ms: image creation comes in bursts.
            std::this_thread::sleep_for(std::chrono::milliseconds{20});
        }
    }

    VmaAllocator allocator;
    std::mutex mutex;
    std::condition_variable cv;
    u32 types = 0;
    bool pending = false;
    std::array<VmaAllocation, VK_MAX_MEMORY_TYPES> held{};
};
} // namespace

void UniqueImage::Create(const vk::ImageCreateInfo& image_ci) {
    this->image_ci = image_ci;
    ASSERT(!image);
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    const VkImageCreateInfo image_ci_unsafe = static_cast<VkImageCreateInfo>(image_ci);
    VkImage unsafe_image{};
    VmaAllocationInfo alloc_info{};
    // MediEvil: the driver prefers its own memory block for most render targets, and Unreal
    // Engine 4 reuses its memory for other targets every frame, so the texture cache created and
    // freed one per image: 16% of the draw recording thread in kernel allocation calls. Images
    // that only prefer one are placed in VMA's shared blocks. NVIDIA only by default (measured
    // there; an AMD driver crashed in a level load, issue #1); GOW3_IMAGE_SUBALLOC=1/0 forces it.
    static const bool suballocate = [this] {
        if (const char* env = std::getenv("GOW3_IMAGE_SUBALLOC")) {
            return std::atoi(env) != 0;
        }
        VmaAllocatorInfo info{};
        vmaGetAllocatorInfo(allocator, &info);
        VkPhysicalDeviceProperties properties{};
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties(info.physicalDevice, &properties);
        const bool nvidia = properties.vendorID == 0x10DE;
        std::printf("Texture cache: shared image memory %s (%s)\n", nvidia ? "on" : "off",
                    properties.deviceName);
        return nvidia;
    }();
    VkResult result = VK_ERROR_UNKNOWN;
    bool created = false;
    const auto create_started = std::chrono::steady_clock::now();
    if (suballocate) {
        VmaAllocatorInfo allocator_info{};
        vmaGetAllocatorInfo(allocator, &allocator_info);
        const VkDevice device = allocator_info.device;
        if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateImage(device, &image_ci_unsafe, nullptr, &unsafe_image) == VK_SUCCESS) {
            VkMemoryDedicatedRequirements dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
            VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dedicated};
            const VkImageMemoryRequirementsInfo2 info{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                                                      nullptr, unsafe_image};
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetImageMemoryRequirements2(device, &info, &requirements);
            if (!dedicated.requiresDedicatedAllocation) {
                // vmaAllocateMemory does not take the AUTO usages: ask for device-local memory.
                VmaAllocationCreateInfo block_ci = alloc_ci;
                block_ci.usage = VMA_MEMORY_USAGE_UNKNOWN;
                block_ci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                result = vmaAllocateMemory(allocator, &requirements.memoryRequirements, &block_ci,
                                           &allocation, &alloc_info);
                static u32 failures = 0;
                if (result != VK_SUCCESS && failures++ < 4) {
                    std::printf("Texture cache: shared image memory failed (%d), own block used\n",
                                int(result));
                }
                if (result == VK_SUCCESS) {
                    result = vmaBindImageMemory(allocator, allocation, unsafe_image);
                    if (result != VK_SUCCESS) {
                        vmaFreeMemory(allocator, allocation);
                        allocation = {};
                    }
                }
                created = result == VK_SUCCESS;
            }
            if (!created) {
                VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyImage(device, unsafe_image, nullptr);
                unsafe_image = VK_NULL_HANDLE;
            }
        }
    }
    if (!created) {
        result = vmaCreateImage(allocator, &image_ci_unsafe, &alloc_ci, &unsafe_image, &allocation,
                                &alloc_info);
        if (result != VK_SUCCESS && ImageReleaser::Get().Enabled()) {
            // Within budget only: the images still queued for release may hold the memory.
            DrainImageReleases();
            result = vmaCreateImage(allocator, &image_ci_unsafe, &alloc_ci, &unsafe_image, &allocation,
                                    &alloc_info);
        }
    }
    ASSERT_MSG(result == VK_SUCCESS, "Failed allocating image with error {}",
               vk::to_string(vk::Result{result}));
    image = vk::Image{unsafe_image};
    size_bytes = alloc_info.size;
    NoteImage(unsafe_image, false);
    // A slow creation is a hitch on the GPU thread: logged (first 64), as in the Uncharted port.
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                create_started).count();
    static std::atomic<u32> slow_logged{0};
    if (ms >= 4.0 && slow_logged.fetch_add(1, std::memory_order_relaxed) < 64) {
        std::printf("Slow image create: %.1f ms, %ux%u, %.1f MiB\n", ms, image_ci.extent.width,
                    image_ci.extent.height, alloc_info.size / 1048576.0);
    }
    SpareBlockKeeper::Note(allocator, alloc_info.memoryType);
}

Image::Image(const Vulkan::Instance& instance, Vulkan::Runtime& runtime_,
             Common::SlotVector<ImageView>& slot_image_views_, const ImageInfo& info_)
    : guest_begin{info_.guest_address}, guest_end{info_.guest_address + info_.guest_size},
      info{info_}, runtime{&runtime_}, slot_image_views{&slot_image_views_} {
    Gow3Stats::Timer timer{Gow3Stats::t_image_create};
    if (info.pixel_format == vk::Format::eUndefined) {
        return;
    }

    image_uid = global_image_uid.Next();
    mip_hashes.resize(info.resources.levels);
    vk::ImageCreateFlags flags{vk::ImageCreateFlagBits::eMutableFormat |
                               vk::ImageCreateFlagBits::eExtendedUsage};
    if (info.props.is_volume) {
        flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
        if (instance.Is2dViewOf3dSupported()) {
            flags |= vk::ImageCreateFlagBits::e2DViewCompatibleEXT;
        }
    }
    if (info.props.is_block && instance.IsBlockTexelViewSupported()) {
        flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
    }

    usage_flags = ImageUsageFlags(instance, info);
    format_features = FormatFeatureFlags(usage_flags);
    if (info.props.is_depth) {
        aspect_mask = vk::ImageAspectFlagBits::eDepth;
        if (info.props.has_stencil) {
            aspect_mask |= vk::ImageAspectFlagBits::eStencil;
        }
    }

    constexpr auto tiling = vk::ImageTiling::eOptimal;
    const auto supported_format = instance.GetSupportedFormat(info.pixel_format, format_features);
    vk::PhysicalDeviceImageFormatInfo2 format_info{
        .format = supported_format,
        .type = ConvertImageType(info.type),
        .tiling = tiling,
        .usage = usage_flags,
        .flags = flags,
    };
    auto image_format_properties =
        instance.GetPhysicalDevice().getImageFormatProperties2(format_info);
    // gow3: storage usage is speculative (see ImageUsageFlags). Drivers that refuse it for a
    // format (AMD's Windows driver for BC6H) get the image without it: creating the unsupported
    // combination anyway loses the device a few frames later.
    if (image_format_properties.result == vk::Result::eErrorFormatNotSupported &&
        (usage_flags & vk::ImageUsageFlagBits::eStorage)) {
        format_info.usage = usage_flags & ~vk::ImageUsageFlagBits::eStorage;
        image_format_properties = instance.GetPhysicalDevice().getImageFormatProperties2(format_info);
        if (image_format_properties.result == vk::Result::eSuccess) {
            usage_flags = format_info.usage;
            format_features = FormatFeatureFlags(usage_flags);
        } else {
            format_info.usage = usage_flags;
        }
    }
    if (image_format_properties.result == vk::Result::eErrorFormatNotSupported) {
        LOG_ERROR(Render_Vulkan, "image format {} type {} is not supported (flags {}, usage {})",
                  vk::to_string(supported_format), vk::to_string(format_info.type),
                  vk::to_string(format_info.flags), vk::to_string(format_info.usage));
    }
    supported_samples = image_format_properties.result == vk::Result::eSuccess
                            ? image_format_properties.value.imageFormatProperties.sampleCounts
                            : vk::SampleCountFlagBits::e1;

    const vk::ImageCreateInfo image_ci = {
        .flags = flags,
        .imageType = ConvertImageType(info.type),
        .format = supported_format,
        .extent{
            .width = info.size.width,
            .height = info.size.height,
            .depth = info.size.depth,
        },
        .mipLevels = static_cast<u32>(info.resources.levels),
        .arrayLayers = static_cast<u32>(info.resources.layers),
        .samples = LiverpoolToVK::NumSamples(info.num_samples, supported_samples),
        .tiling = tiling,
        .usage = usage_flags,
        .initialLayout = vk::ImageLayout::eUndefined,
    };

    backing = &backing_images.emplace_back();
    backing->num_samples = info.num_samples;
    backing->image = UniqueImage{instance.GetDevice(), instance.GetAllocator()};
    backing->image.Create(image_ci);

    Vulkan::SetObjectName(instance.GetDevice(), GetImage(),
                          "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{}", info.size.width,
                          info.size.height, info.size.depth, AmdGpu::NameOf(info.tile_mode),
                          vk::to_string(info.pixel_format), info.guest_address, info.guest_size,
                          info.resources.layers, info.resources.levels, info.num_samples);
}

Image::~Image() = default;

ImageView& Image::FindView(const ImageViewInfo& view_info, bool ensure_guest_samples) {
    if (ensure_guest_samples && backing->num_samples > 1 != info.num_samples > 1) {
        runtime->SetBackingSamples(this, info.num_samples);
    }
    const auto& view_infos = backing->image_view_infos;
    if (backing->last_view < view_infos.size() && view_infos[backing->last_view] == view_info) {
        return (*slot_image_views)[backing->image_view_ids[backing->last_view]];
    }
    const auto it = std::ranges::find(view_infos, view_info);
    if (it != view_infos.end()) {
        backing->last_view = static_cast<u32>(std::distance(view_infos.begin(), it));
        const auto view_id = backing->image_view_ids[backing->last_view];
        return (*slot_image_views)[view_id];
    }
    const auto view_id = slot_image_views->insert(runtime->GetInstance(), view_info, *this);
    backing->image_view_infos.emplace_back(view_info);
    backing->image_view_ids.emplace_back(view_id);
    return (*slot_image_views)[view_id];
}

// gow3 GOW3_SKIP_READ_BARRIERS=1: a read after a read in the same layout needs no barrier (no
// hazard). The reads are merged into the tracked state instead, so the next real barrier (a
// write or a layout change) waits for all of them. T-F2: the kills' effects put ~36k barriers a
// 5 s window between depth-test reads and shader reads of the depth buffer in one read-only
// layout, each ending and reopening the render pass.
static bool SkipReadBarriers() {
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_SKIP_READ_BARRIERS");
        return !(env && env[0] == '0'); // default on (launcher: perf_lean_barriers)
    }();
    return enabled;
}

// gow3 GOW3_NARROW_IMAGE_STAGES: a sampled image's stored stage is now the shader stages that read
// it, not all commands. A read by another kind of shader (compute after graphics, or back) is
// outside the earlier barrier's scope, so it gets its own read-after-read barrier, which chains
// onto that one. Stored stages that are not only shader stages (depth tests, attachments, all
// commands) skip as before (T-F2).
static bool ReadCovered(vk::PipelineStageFlags2 have, vk::PipelineStageFlags2 want) {
    using S = vk::PipelineStageFlagBits2;
    constexpr vk::PipelineStageFlags2 pre_raster =
        S::eVertexShader | S::eTessellationControlShader | S::eTessellationEvaluationShader |
        S::eGeometryShader;
    constexpr vk::PipelineStageFlags2 shaders =
        pre_raster | S::ePreRasterizationShaders | S::eFragmentShader | S::eComputeShader;
    if (have & ~shaders) {
        return true;
    }
    if (have & S::ePreRasterizationShaders) {
        have |= pre_raster;
    }
    if (want & S::ePreRasterizationShaders) {
        want = (want & ~vk::PipelineStageFlags2{S::ePreRasterizationShaders}) | pre_raster;
    }
    return !(want & ~have);
}

static bool IsReadOnly(vk::AccessFlags2 access) {
    constexpr auto writes = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderStorageWrite |
                            vk::AccessFlagBits2::eColorAttachmentWrite |
                            vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                            vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eHostWrite |
                            vk::AccessFlagBits2::eMemoryWrite;
    return !(access & writes);
}

void Image::GetBarriers(Barriers& barriers, vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                        vk::PipelineStageFlags2 dst_stage,
                        std::optional<SubresourceRange> subres_range) {
    if (info.props.is_volume && subres_range) {
        // gow3: a 3D image has one array layer; its 2D array views (render targets) address depth
        // slices, and a barrier covers the whole image (Vulkan: layer 0, count 1).
        subres_range->base.layer = 0;
        subres_range->extent.layers = info.resources.layers;
    }
    auto& last_state = backing->state;
    auto& subresource_states = backing->subresource_states;
    // gow3: a 3D image has one array layer; the layers of a 2D array view of it are depth
    // slices, which a barrier cannot select (Vulkan: layer 0, count 1), so it covers the volume.
    if (subres_range && info.props.is_volume) {
        subres_range->base.layer = 0;
        subres_range->extent.layers = 1;
    }

    const bool needs_partial_transition =
        subres_range &&
        (subres_range->base != SubresourceBase{} || subres_range->extent != info.resources);
    const bool partially_transited = !subresource_states.empty();

    if (needs_partial_transition || partially_transited) {
        if (!partially_transited) {
            subresource_states.resize(info.resources.levels * info.resources.layers);
            std::fill(subresource_states.begin(), subresource_states.end(), last_state);
        }

        // In case of partial transition, we need to change the specified subresources only.
        // Otherwise all subresources need to be set to the same state so we can use a full
        // resource transition for the next time.
        const auto mips =
            needs_partial_transition
                ? std::ranges::views::iota(subres_range->base.level,
                                           subres_range->base.level + subres_range->extent.levels)
                : std::views::iota(0u, info.resources.levels);
        const auto layers =
            needs_partial_transition
                ? std::ranges::views::iota(subres_range->base.layer,
                                           subres_range->base.layer + subres_range->extent.layers)
                : std::views::iota(0u, info.resources.layers);

        // gow3: per-layer barriers of the same transition are merged into one per layer range. A
        // layered image gave 3696 barriers in one batch, and AMD's older Windows driver (RX 400/500,
        // Vega) crashed recording them (issues #18, #34, #41, #43).
        // GOW3_BARRIER_MERGE=1 merges; OFF by default: with the background pipeline warm-up it
        // broke glowing effects (red outlines on the crypt gargoyle, 29 Oct 2026). To be fixed
        // with the AMD crash work.
        static const bool merge = [] {
            const char* env = std::getenv("GOW3_BARRIER_MERGE");
            return env && env[0] == '1';
        }();
        const size_t first_barrier = merge ? barriers.size() : SIZE_MAX;
        for (u32 mip : mips) {
            for (u32 layer : layers) {
                const auto subres_idx = mip * info.resources.layers + layer;
                ASSERT(subres_idx < subresource_states.size());
                auto& state = subresource_states[subres_idx];

                constexpr auto write_flags = vk::AccessFlagBits2::eTransferWrite |
                                             vk::AccessFlagBits2::eShaderWrite |
                                             vk::AccessFlagBits2::eMemoryWrite;
                const bool is_write = static_cast<bool>(state.access_mask & write_flags);
                if (SkipReadBarriers() && state.layout == dst_layout &&
                    IsReadOnly(state.access_mask) && IsReadOnly(dst_mask) &&
                    ReadCovered(state.pl_stage, dst_stage)) {
                    state.access_mask |= dst_mask;
                    state.pl_stage |= dst_stage;
                    continue;
                }
                if (state.layout != dst_layout || state.access_mask != dst_mask || is_write ||
                    !ReadCovered(state.pl_stage, dst_stage)) {
                    // A read after a read keeps the earlier readers: a later write waits for all.
                    const auto keep_stages =
                        state.layout == dst_layout && IsReadOnly(state.access_mask) && IsReadOnly(dst_mask)
                            ? state.pl_stage
                            : vk::PipelineStageFlags2{};
                    if (merge && barriers.size() > first_barrier) {
                        auto& last = barriers.back();
                        auto& range = last.subresourceRange;
                        if (range.baseMipLevel == mip && range.levelCount == 1 &&
                            range.baseArrayLayer + range.layerCount == layer &&
                            last.oldLayout == state.layout && last.srcStageMask == state.pl_stage &&
                            last.srcAccessMask == state.access_mask) {
                            ++range.layerCount;
                            state.layout = dst_layout;
                            state.access_mask = dst_mask;
                            state.pl_stage = keep_stages | dst_stage;
                            continue;
                        }
                    }
                    barriers.emplace_back(vk::ImageMemoryBarrier2{
                        .srcStageMask = state.pl_stage,
                        .srcAccessMask = state.access_mask,
                        .dstStageMask = dst_stage,
                        .dstAccessMask = dst_mask,
                        .oldLayout = state.layout,
                        .newLayout = dst_layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = GetImage(),
                        .subresourceRange{
                            .aspectMask = aspect_mask,
                            .baseMipLevel = mip,
                            .levelCount = 1,
                            .baseArrayLayer = layer,
                            .layerCount = 1,
                        },
                    });
                    state.layout = dst_layout;
                    state.access_mask = dst_mask;
                    state.pl_stage = keep_stages | dst_stage;
                }
            }
        }

        if (!needs_partial_transition) {
            subresource_states.clear();
        }
    } else { // Full resource transition
        constexpr auto write_flags = vk::AccessFlagBits2::eTransferWrite |
                                     vk::AccessFlagBits2::eShaderWrite |
                                     vk::AccessFlagBits2::eMemoryWrite;
        const bool is_write = static_cast<bool>(last_state.access_mask & write_flags);
        if (last_state.layout == dst_layout && last_state.access_mask == dst_mask && !is_write &&
            ReadCovered(last_state.pl_stage, dst_stage)) {
            return;
        }
        if (SkipReadBarriers() && last_state.layout == dst_layout &&
            IsReadOnly(last_state.access_mask) && IsReadOnly(dst_mask) &&
            ReadCovered(last_state.pl_stage, dst_stage)) {
            last_state.access_mask |= dst_mask;
            last_state.pl_stage |= dst_stage;
            return;
        }
        // A read after a read keeps the earlier readers: a later write waits for all.
        const auto keep_stages = last_state.layout == dst_layout && IsReadOnly(last_state.access_mask) &&
                                         IsReadOnly(dst_mask)
                                     ? last_state.pl_stage
                                     : vk::PipelineStageFlags2{};

        barriers.emplace_back(vk::ImageMemoryBarrier2{
            .srcStageMask = last_state.pl_stage,
            .srcAccessMask = last_state.access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_mask,
            .oldLayout = last_state.layout,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = GetImage(),
            .subresourceRange{
                .aspectMask = aspect_mask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        });
        last_state.layout = dst_layout;
        last_state.access_mask = dst_mask;
        last_state.pl_stage = keep_stages | dst_stage;
        return;
    }

    last_state.layout = dst_layout;
    last_state.access_mask = dst_mask;
    last_state.pl_stage = dst_stage;
}

} // namespace VideoCore
