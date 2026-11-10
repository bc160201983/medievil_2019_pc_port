// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdio>
#include <stop_token>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <array>
#include <map>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <magic_enum/magic_enum.hpp>
#include "gow3_copy.h"
#include "gow3_sections.h"
#include "gow3_toggles.h"
#include "gow3_settings.h"
#include "common/alignment.h"
#include "common/thread.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/renderer_vulkan/vk_wait_diagnostics.h"

#include <vk_mem_alloc.h>

extern "C" unsigned char* runtime_image_base(void);
extern "C" void runtime_memory_mark_gpu_written(uintptr_t address, uint64_t size, int written);

namespace {
// gow3 D3 (ported from the Bloodborne port's GpuWriteTicks / FastReadback, 0.3; GPL-2.0-or-later):
// the submission tick of the newest GPU write per 64 KiB page (1 MiB pages for writes over
// 16 MiB), lock-free sparse page tables (guest addresses below 2^40).
struct GpuWriteTicks {
    static constexpr u64 FineShift = 16, CoarseShift = 20;
    static constexpr u64 FineLimit = 16_MB;
    static constexpr u64 AddressBits = 40, LeafBits = 12, LeafMask = (1ull << LeafBits) - 1;
    struct Table {
        explicit Table(u64 shift_)
            : shift{shift_}, leaves((1ull << (AddressBits - shift_)) >> LeafBits) {}
        u64 shift;
        std::vector<std::atomic<std::atomic<u64>*>> leaves;
        std::atomic<u64>* Leaf(u64 page) const {
            return leaves[page >> LeafBits].load(std::memory_order_acquire);
        }
    };
    Table fine{FineShift};
    Table coarse{CoarseShift};
    std::mutex mutex; ///< leaf creation
    void Note(u64 address, u64 size, u64 tick) {
        if (size == 0 || (address + size - 1) >> AddressBits) {
            return;
        }
        auto& table = size <= FineLimit ? fine : coarse;
        for (u64 page = address >> table.shift; page <= (address + size - 1) >> table.shift;
             ++page) {
            std::atomic<u64>* leaf = table.Leaf(page);
            if (!leaf) {
                std::scoped_lock lk{mutex};
                leaf = table.Leaf(page);
                if (!leaf) {
                    leaf = new std::atomic<u64>[1ull << LeafBits]{};
                    table.leaves[page >> LeafBits].store(leaf, std::memory_order_release);
                }
            }
            leaf[page & LeafMask].store(tick, std::memory_order_release);
        }
    }
    /// The newest write tick in the range, 0 when none is known.
    u64 Newest(u64 address, u64 size) const {
        if (size == 0 || (address + size - 1) >> AddressBits) {
            return 0;
        }
        u64 newest = 0;
        for (const Table* table : {&fine, &coarse}) {
            const u64 last = (address + size - 1) >> table->shift;
            for (u64 page = address >> table->shift; page <= last; ++page) {
                if (const std::atomic<u64>* leaf = table->Leaf(page)) {
                    newest = std::max(newest, leaf[page & LeafMask].load(std::memory_order_acquire));
                } else {
                    page |= LeafMask;
                }
            }
        }
        return newest;
    }
};
GpuWriteTicks& WriteTicks() {
    static GpuWriteTicks write_ticks;
    return write_ticks;
}

// gow3 U2: the newest tick at which the GPU uses each arena page: draws and dispatches binding it
// (reads and writes), copies and fills into it, all noted when recorded (ObtainBuffer), so work
// recorded but not yet submitted counts (its tick is the current one, never free). A direct arena
// write needs every page of its range free. Pages are 4 KiB for uses up to 64 KiB (the game
// sub-allocates per-frame data within 64 KiB pages: a coarser page would refuse the next
// upload beside a range a draw just read), 64 KiB up to 16 MiB, 1 MiB above; a query reads all
// three. A tick is only ever raised: two threads may note the same page.
struct GpuUseTicks {
    static constexpr u64 AddressBits = 40, LeafBits = 12, LeafMask = (1ull << LeafBits) - 1;
    struct Table {
        Table(u64 shift_, u64 limit_)
            : shift{shift_}, limit{limit_}, leaves((1ull << (AddressBits - shift_)) >> LeafBits) {}
        u64 shift, limit;
        std::vector<std::atomic<std::atomic<u64>*>> leaves;
    };
    std::array<Table, 3> tables{Table{12, 64_KB}, Table{16, 16_MB}, Table{20, ~0ull}};
    std::mutex mutex; ///< leaf creation
    ~GpuUseTicks() {
        for (auto& table : tables) {
            for (auto& leaf : table.leaves) {
                delete[] leaf.load(std::memory_order_relaxed);
            }
        }
    }
    void Note(u64 address, u64 size, u64 tick) {
        if (size == 0 || (address + size - 1) >> AddressBits) {
            return;
        }
        auto& table = *std::ranges::find_if(tables, [size](const Table& t) { return size <= t.limit; });
        for (u64 page = address >> table.shift; page <= (address + size - 1) >> table.shift;
             ++page) {
            std::atomic<u64>* leaf = table.leaves[page >> LeafBits].load(std::memory_order_acquire);
            if (!leaf) {
                std::scoped_lock lk{mutex};
                leaf = table.leaves[page >> LeafBits].load(std::memory_order_acquire);
                if (!leaf) {
                    leaf = new std::atomic<u64>[1ull << LeafBits]{};
                    table.leaves[page >> LeafBits].store(leaf, std::memory_order_release);
                }
            }
            auto& entry = leaf[page & LeafMask];
            u64 current = entry.load(std::memory_order_relaxed);
            while (current < tick &&
                   !entry.compare_exchange_weak(current, tick, std::memory_order_release,
                                                std::memory_order_relaxed)) {
            }
        }
    }
    /// The newest use tick in the range, 0 when none is known.
    u64 Newest(u64 address, u64 size) const {
        if (size == 0 || (address + size - 1) >> AddressBits) {
            return size == 0 ? 0 : ~0ull; // outside the tables: treated as in use
        }
        u64 newest = 0;
        for (const Table& table : tables) {
            const u64 last = (address + size - 1) >> table.shift;
            for (u64 page = address >> table.shift; page <= last; ++page) {
                if (const std::atomic<u64>* leaf =
                        table.leaves[page >> LeafBits].load(std::memory_order_acquire)) {
                    newest = std::max(newest, leaf[page & LeafMask].load(std::memory_order_acquire));
                } else {
                    page |= LeafMask;
                }
            }
        }
        return newest;
    }
};
GpuUseTicks& UseTicks() {
    static GpuUseTicks use_ticks;
    return use_ticks;
}
/// The staged copies into the arena alone (stats only: tells "a staged copy is still queued"
/// apart from other GPU use when a direct write falls back).
GpuUseTicks& StagedTicks() {
    static GpuUseTicks staged_ticks;
    return staged_ticks;
}

/// Copies finished GPU data out of an arena on the readback queue, waiting only for the
/// submission of the newest write (the timeline semaphore at its tick), not for the frames queued
/// behind it on the graphics queue.
class FastReadback {
public:
    explicit FastReadback(const Vulkan::Instance& instance)
        : device{instance.GetDevice()}, queue{instance.GetReadbackQueue()},
          allocator{instance.GetAllocator()} {
        if (!queue) {
            return;
        }
        pool = Vulkan::Check(device.createCommandPoolUnique({
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = instance.GetReadbackQueueFamilyIndex(),
        }));
        cmdbuf = Vulkan::Check(device.allocateCommandBuffers({
            .commandPool = *pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        }))[0];
        fence = Vulkan::Check(device.createFenceUnique({}));
    }
    ~FastReadback() {
        if (buffer) {
            vmaDestroyBuffer(allocator, buffer, allocation);
        }
    }
    [[nodiscard]] bool Available() const noexcept {
        static const bool off = [] {
            const char* env = std::getenv("GOW3_READBACK_QUEUE");
            return env && env[0] == '0';
        }();
        return bool(queue) && !off;
    }
    /// Copies `copies` (destination offsets from 0) of `arena` once `semaphore` reaches `tick`;
    /// the host copy, or null on failure (the caller takes the regular path).
    const u8* Copy(const VideoCore::Buffer* arena, std::span<const vk::BufferCopy> copies, u64 total,
                   vk::Semaphore semaphore, u64 tick) {
        if (!Ensure(total)) {
            return nullptr;
        }
        Vulkan::Check(cmdbuf.reset());
        Vulkan::Check(cmdbuf.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
        cmdbuf.copyBuffer(arena->Handle(), vk::Buffer{buffer}, copies);
        const vk::MemoryBarrier2 to_host{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        };
        cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &to_host});
        Vulkan::Check(cmdbuf.end());
        const vk::SemaphoreSubmitInfo wait{
            .semaphore = semaphore,
            .value = tick,
            .stageMask = vk::PipelineStageFlagBits2::eTransfer,
        };
        const vk::CommandBufferSubmitInfo command{.commandBuffer = cmdbuf};
        const vk::SubmitInfo2 submit{
            .waitSemaphoreInfoCount = 1,
            .pWaitSemaphoreInfos = &wait,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &command,
        };
        if (queue.submit2(submit, *fence) != vk::Result::eSuccess) {
            return nullptr;
        }
        const auto result = device.waitForFences(*fence, vk::True, UINT64_MAX);
        Vulkan::Check(device.resetFences(*fence));
        if (result != vk::Result::eSuccess) {
            return nullptr;
        }
        vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
        return mapped;
    }

private:
    bool Ensure(u64 size) {
        if (size <= capacity) {
            return true;
        }
        if (buffer) {
            vmaDestroyBuffer(allocator, buffer, allocation);
            buffer = VK_NULL_HANDLE;
            capacity = 0;
        }
        const u64 new_capacity = std::max<u64>(std::bit_ceil(size), 1_MB);
        const VkBufferCreateInfo buffer_ci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = new_capacity,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo alloc_ci{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        };
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(allocator, &buffer_ci, &alloc_ci, &buffer, &allocation, &info) !=
            VK_SUCCESS) {
            buffer = VK_NULL_HANDLE;
            return false;
        }
        mapped = static_cast<u8*>(info.pMappedData);
        capacity = new_capacity;
        return true;
    }

    vk::Device device;
    vk::Queue queue;
    VmaAllocator allocator;
    vk::UniqueCommandPool pool;
    vk::CommandBuffer cmdbuf;
    vk::UniqueFence fence;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    u8* mapped = nullptr;
    u64 capacity = 0;
};
FastReadback& SharedReadback(const Vulkan::Instance& instance) {
    static FastReadback readback{instance};
    return readback;
}

// gow3 GPU write twins (ported from the Bloodborne port's BB_GPU_WRITE_TWINS, 0.4; GPL-2.0-or-
// later). A guest write into a page holding bytes the GPU is still writing (counters, small
// compute outputs rewritten every frame) does not wait for the GPU: the guest's copy of those
// bytes is kept as a "twin" and the page becomes CPU-modified at once. Uploads leave the GPU's
// bytes in the arena while the guest bytes still match their twin (the guest did not write them);
// downloads do not overwrite guest bytes that no longer match it (the guest wrote them since).
// Only pages with at most GOW3_GPU_WRITE_TWINS_MAX (1024) GPU-written bytes: larger outputs keep
// waiting. Guest bytes are read through the backing view: the page may be protected, and a fault
// with twins_mutex held could wait for a readback that needs it. GOW3_GPU_WRITE_TWINS=0: off.
// T-D3b: these writes (9-20 a second in the light view) each waited 2-5 ms for the frame being
// recorded.
bool TwinsEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_GPU_WRITE_TWINS");
        return (!env || env[0] != '0') &&
               EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Precise;
    }();
    return enabled;
}
u64 TwinsMaxBytes() {
    static const u64 max_bytes = [] {
        const char* env = std::getenv("GOW3_GPU_WRITE_TWINS_MAX");
        // gow3: 16 KiB (was 1 KiB, the Bloodborne port's value): a twin is a snapshot of the
        // page's GPU-written bytes, microseconds even at this size, against a GPU wait otherwise.
        return env ? std::strtoull(env, nullptr, 10) : 16384ull;
    }();
    return max_bytes;
}
std::mutex twins_mutex;
std::map<VAddr, std::vector<u8>> twins; ///< start -> guest bytes when the twin was made

enum class TwinRun { None, Unchanged, Changed };

/// Calls func(start, end, TwinRun) over [begin, end) in order (twins_mutex held).
template <typename Func>
void ForEachTwinRun(VAddr begin, VAddr end, Func&& func) {
    VAddr cursor = begin;
    auto it = twins.upper_bound(begin);
    if (it != twins.begin()) {
        --it;
    }
    for (; it != twins.end() && it->first < end; ++it) {
        const VAddr start = it->first, stop = it->first + it->second.size();
        if (stop <= begin) {
            continue;
        }
        const VAddr from = std::max(start, begin), to = std::min(stop, end);
        if (cursor < from) {
            func(cursor, from, TwinRun::None);
        }
        thread_local std::vector<u8> guest_bytes;
        guest_bytes.resize(to - from);
        Core::Memory::Instance()->ReadBacking(from, guest_bytes.data(), to - from);
        const u8* guest = guest_bytes.data();
        const u8* twin = it->second.data() + (from - start);
        VAddr run = from;
        bool same = guest[0] == twin[0];
        for (VAddr at = from + 1; at <= to; ++at) {
            const bool now_same = at < to && guest[at - from] == twin[at - from];
            if (at == to || now_same != same) {
                func(run, at, same ? TwinRun::Unchanged : TwinRun::Changed);
                run = at;
                same = now_same;
            }
        }
        cursor = to;
    }
    if (cursor < end) {
        func(cursor, end, TwinRun::None);
    }
}

/// Drops the twins' parts in [begin, end), keeping the parts outside it (twins_mutex held).
void EraseTwins(VAddr begin, VAddr end) {
    auto it = twins.upper_bound(begin);
    if (it != twins.begin()) {
        --it;
    }
    while (it != twins.end() && it->first < end) {
        const VAddr start = it->first, stop = it->first + it->second.size();
        if (stop <= begin) {
            ++it;
            continue;
        }
        std::vector<u8> bytes = std::move(it->second);
        it = twins.erase(it);
        if (start < begin) {
            twins[start] = std::vector<u8>(bytes.begin(), bytes.begin() + (begin - start));
        }
        if (stop > end) {
            twins[end] = std::vector<u8>(bytes.begin() + (end - start), bytes.end());
        }
    }
}

/// The pages fully inside [addr, addr + size) (the edge pages may still hold other GPU data).
void ForgetGpuWritten(VAddr addr, u64 size) {
    const VAddr first = (addr + 4095) & ~VAddr{4095}, end = (addr + size) & ~VAddr{4095};
    if (end > first) {
        runtime_memory_mark_gpu_written(first, end - first, 0);
    }
}
} // namespace

#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

// gow3: device memory for resident arena blocks. EnsureResident allocated each new range itself
// (vkAllocateMemory on the draw recording thread), a kernel call that takes the GPU driver's lock:
// a GPU monitor's polling held it up to 33 ms there (MSI Afterburner's NVAPI escapes, research:
// plan-frame-pacing.md, "CPU oscillation"). Resident blocks are never released, so they are taken
// from large chunks in order; a worker allocates the next chunk once the current one is half used.
// GOW3_RESIDENCY_CHUNK_MB (default 32; 0: one allocation per call, as before).
class ResidencyPool {
public:
    struct Piece {
        vk::DeviceMemory memory;
        u64 offset;
        u64 blocks;
        u8* host = nullptr; ///< gow3 U2: the piece's first byte in the mapping (null: unmapped)
    };
    struct Chunk {
        vk::DeviceMemory memory{};
        u8* host = nullptr;
    };

    /// gow3 U2: `map_`: the memory type is host-visible; chunks are mapped persistently.
    ResidencyPool(vk::Device device_, u32 memory_type_, u32 block_shift_, bool map_)
        : device{device_}, memory_type{memory_type_}, block_shift{block_shift_}, map{map_} {
        const char* env = std::getenv("GOW3_RESIDENCY_CHUNK_MB");
        const u64 mb = env ? std::strtoull(env, nullptr, 10) : 32;
        chunk_blocks = (mb << 20) >> block_shift;
        if (chunk_blocks == 0) {
            return;
        }
        current = Allocate("at start");
        worker = std::jthread([this](std::stop_token stop) { Run(stop); });
    }

    bool Enabled() const {
        return chunk_blocks != 0;
    }

    /// gow3 U2: whether the first chunk could be mapped (false: direct writes stay off).
    bool FirstChunkMapped() const {
        return current.host != nullptr;
    }

    /// Up to `blocks` blocks from the current chunk (at least one); the next chunk when it is full.
    Piece Take(u64 blocks) {
        std::unique_lock lock{mutex};
        if (used == chunk_blocks) {
            if (!next.memory && requested) {
                ready_cv.wait(lock, [this] { return next.memory || !requested; });
            }
            if (next.memory) {
                current = std::exchange(next, Chunk{});
            } else {
                current = Allocate("on demand");
            }
            used = 0;
        }
        const u64 count = std::min(blocks, chunk_blocks - used);
        const u64 offset = used << block_shift;
        const Piece piece{current.memory, offset, count,
                          current.host ? current.host + offset : nullptr};
        used += count;
        if (!next.memory && !requested && used * 2 >= chunk_blocks) {
            requested = true;
            request_cv.notify_one();
        }
        return piece;
    }

    /// gow3 U2: maps `memory` whole and persistently; null when mapping fails (the blocks then
    /// take the staging path).
    static u8* Map(vk::Device device, vk::DeviceMemory memory) {
        const auto mapped = device.mapMemory(memory, 0, VK_WHOLE_SIZE);
        if (mapped.result != vk::Result::eSuccess) {
            std::printf("Buffer residency: mapping arena memory failed (%s): staging uploads\n",
                        vk::to_string(mapped.result).c_str());
            return nullptr;
        }
        return static_cast<u8*>(mapped.value);
    }

private:
    Chunk Allocate(const char* when) {
        const vk::MemoryAllocateInfo info{.allocationSize = chunk_blocks << block_shift,
                                          .memoryTypeIndex = memory_type};
        const auto memory = Vulkan::Check(device.allocateMemory(info));
        u8* host = map ? Map(device, memory) : nullptr;
        std::printf("Buffer residency: %llu MB chunk %u allocated %s%s\n",
                    (unsigned long long)((chunk_blocks << block_shift) >> 20), ++chunks, when,
                    host ? ", mapped" : "");
        return {memory, host};
    }

    void Run(std::stop_token stop) {
        Common::SetCurrentThreadName("gow3:Residency");
        std::unique_lock lock{mutex};
        while (true) {
            request_cv.wait(lock, stop, [this] { return requested; });
            if (!requested) {
                return; // stop requested
            }
            lock.unlock();
            const auto chunk = Allocate("ahead");
            lock.lock();
            next = chunk;
            requested = false;
            ready_cv.notify_all();
        }
    }

    vk::Device device;
    u32 memory_type;
    u32 block_shift;
    bool map;
    u64 chunk_blocks = 0;
    std::atomic<u32> chunks = 0;
    std::mutex mutex;
    std::condition_variable_any request_cv;
    std::condition_variable_any ready_cv;
    Chunk current{};
    Chunk next{};
    u64 used = 0;
    bool requested = false;
    std::jthread worker;
};

// gow3 U2: the host address of each resident arena block whose memory is mapped, in a lock-free
// two-level table by block number (guest addresses below 2^40). Resident blocks are never
// released, so an entry once set stays valid.
struct ArenaMappings {
    static constexpr u64 LeafBits = 12, LeafMask = (1ull << LeafBits) - 1;
    explicit ArenaMappings(u32 block_shift)
        : leaves(((1ull << 40) >> block_shift) >> LeafBits) {}
    ~ArenaMappings() {
        for (auto& leaf : leaves) {
            delete[] leaf.load(std::memory_order_relaxed);
        }
    }
    void Set(u64 block, u8* host) {
        if ((block >> LeafBits) >= leaves.size()) {
            return;
        }
        std::atomic<u8*>* leaf = leaves[block >> LeafBits].load(std::memory_order_acquire);
        if (!leaf) {
            std::scoped_lock lk{mutex};
            leaf = leaves[block >> LeafBits].load(std::memory_order_acquire);
            if (!leaf) {
                leaf = new std::atomic<u8*>[1ull << LeafBits]{};
                leaves[block >> LeafBits].store(leaf, std::memory_order_release);
            }
        }
        leaf[block & LeafMask].store(host, std::memory_order_release);
    }
    u8* Get(u64 block) const {
        if ((block >> LeafBits) >= leaves.size()) {
            return nullptr;
        }
        const std::atomic<u8*>* leaf = leaves[block >> LeafBits].load(std::memory_order_acquire);
        return leaf ? leaf[block & LeafMask].load(std::memory_order_acquire) : nullptr;
    }
    std::vector<std::atomic<std::atomic<u8*>*>> leaves;
    std::mutex mutex; ///< leaf creation
};

namespace {
std::string MemoryFlagNames(vk::MemoryPropertyFlags flags) {
    std::string names;
    const std::pair<vk::MemoryPropertyFlagBits, const char*> bits[] = {
        {vk::MemoryPropertyFlagBits::eDeviceLocal, "DEVICE_LOCAL"},
        {vk::MemoryPropertyFlagBits::eHostVisible, "HOST_VISIBLE"},
        {vk::MemoryPropertyFlagBits::eHostCoherent, "HOST_COHERENT"},
        {vk::MemoryPropertyFlagBits::eHostCached, "HOST_CACHED"},
        {vk::MemoryPropertyFlagBits::eLazilyAllocated, "LAZY"},
        {vk::MemoryPropertyFlagBits::eProtected, "PROTECTED"},
        {vk::MemoryPropertyFlagBits::eDeviceCoherentAMD, "DEVICE_COHERENT_AMD"},
        {vk::MemoryPropertyFlagBits::eDeviceUncachedAMD, "DEVICE_UNCACHED_AMD"},
    };
    for (const auto& [bit, name] : bits) {
        if (flags & bit) {
            names += names.empty() ? "" : "|";
            names += name;
        }
    }
    return names.empty() ? "0" : names;
}

// gow3 U2: GOW3_ARENA_HOST_VISIBLE (read at start; the memory type cannot change live): 1 takes
// the arena's residency chunks from a DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT type, HOST_CACHED
// preferred (CPU reads for U3), and maps them; 2 the same with an uncached type preferred; 0 off.
// Default 1 on Linux integrated GPUs (Steam Deck: one LPDDR5 for CPU and GPU), 0 elsewhere (on a
// discrete card the writes cross PCIe; ReBAR desktops can try 1). The type must be one the
// sparse arena accepts, in the heap the arena uses today or in one of 4 GiB or more (a 256 MiB
// BAR window without ReBAR would run full). Without one, U2 stays off: today's type, staging.
std::optional<u32> HostVisibleArenaType(const Vulkan::Instance& instance, u32 memory_type_bits,
                                        u32 device_local_type) {
    int mode = 0;
    if (const char* env = std::getenv("GOW3_ARENA_HOST_VISIBLE"); env && env[0]) {
        mode = std::atoi(env);
    } else {
#ifdef __linux__
        mode = instance.IsIntegrated() ? 1 : 0;
#endif
    }
    if (mode <= 0) {
        return std::nullopt;
    }
    const auto& props = instance.GetMemoryProperties();
    const u32 arena_heap = props.memoryTypes[device_local_type].heapIndex;
    const auto wanted = vk::MemoryPropertyFlagBits::eDeviceLocal |
                        vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent;
    const auto unwanted = vk::MemoryPropertyFlagBits::eLazilyAllocated |
                          vk::MemoryPropertyFlagBits::eProtected |
                          vk::MemoryPropertyFlagBits::eDeviceCoherentAMD |
                          vk::MemoryPropertyFlagBits::eDeviceUncachedAMD;
    std::optional<u32> best;
    bool best_cached = false;
    for (u32 i = 0; i < props.memoryTypeCount; ++i) {
        const auto flags = props.memoryTypes[i].propertyFlags;
        const u32 heap = props.memoryTypes[i].heapIndex;
        if (((memory_type_bits >> i) & 1) == 0 || (flags & wanted) != wanted ||
            (flags & unwanted)) {
            continue;
        }
        if (heap != arena_heap && props.memoryHeaps[heap].size < 4_GB) {
            continue;
        }
        const bool cached = bool(flags & vk::MemoryPropertyFlagBits::eHostCached);
        if (!best || (mode == 1 && cached && !best_cached) ||
            (mode != 1 && !cached && best_cached)) {
            best = i;
            best_cached = cached;
        }
    }
    return best;
}
} // namespace

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    stream_buffer.written_stat = &Gow3Stats::stream_buffer_bytes;
    while (arena_page_bits > MIN_ARENA_PAGE_BITS &&
           (u64{2} << arena_page_bits) > instance.GetMaxBufferSize()) {
        --arena_page_bits;
    }
    arena_page_size = u64{1} << arena_page_bits;
    num_arena_pages = u64{1} << (ADDRESS_SPACE_BITS - arena_page_bits);
    if (arena_page_bits != MAX_ARENA_PAGE_BITS) {
        LOG_INFO(Render_Vulkan, "Buffer arenas of {} MiB (driver buffer limit {} MiB)",
                 arena_page_size >> 20, instance.GetMaxBufferSize() >> 20);
    }
    const std::array<u32, 2> families = {instance.GetGraphicsQueueFamilyIndex(),
                                         instance.GetReadbackQueueFamilyIndex()};
    const bool shared = bool(instance.GetReadbackQueue()); // as Buffer creates the arenas
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = arena_page_size,
        .usage = ARENA_USAGE,
        .sharingMode = shared ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = shared ? 2u : 0u,
        .pQueueFamilyIndices = shared ? families.data() : nullptr,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = arena_page_size / block_size;
    blocks_per_arena_page_shift = arena_page_bits - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();
    {
        // gow3 U1 / U2: the memory types (which heap is which on an integrated GPU) and the
        // arena's choice, in every log.
        const auto& props = instance.GetMemoryProperties();
        std::string types;
        for (u32 i = 0; i < props.memoryTypeCount; ++i) {
            const u32 heap = props.memoryTypes[i].heapIndex;
            char text[192];
            std::snprintf(text, sizeof(text), "%s%u: %s heap %u (%llu MiB)%s", i ? "; " : "", i,
                          MemoryFlagNames(props.memoryTypes[i].propertyFlags).c_str(), heap,
                          (unsigned long long)(props.memoryHeaps[heap].size >> 20),
                          ((reqs.memoryTypeBits >> i) & 1) ? "" : " (not for the arena)");
            types += text;
        }
        std::printf("GPU memory types: %s\n", types.c_str());
    }
    const u32 device_local_type = arena_memory_type_index;
    if (const auto host_visible = HostVisibleArenaType(instance, reqs.memoryTypeBits,
                                                       arena_memory_type_index)) {
        arena_memory_type_index = *host_visible;
        arena_mappings = std::make_unique<ArenaMappings>(block_shift);
    }
    residency_pool = std::make_unique<ResidencyPool>(device, arena_memory_type_index, block_shift,
                                                     arena_mappings != nullptr);
    if (arena_mappings && residency_pool->Enabled() && !residency_pool->FirstChunkMapped()) {
        arena_mappings.reset(); // later chunks would map no better
    }
    {
        const auto& type = instance.GetMemoryProperties().memoryTypes[arena_memory_type_index];
        std::printf("Buffer arena: memory type %u (%s, heap %u), %u KiB blocks; direct arena writes "
                    "(U2) %s\n",
                    arena_memory_type_index, MemoryFlagNames(type.propertyFlags).c_str(),
                    type.heapIndex, block_size >> 10,
                    arena_mappings ? "available (toggle file second word 8: off)"
                    : arena_memory_type_index != device_local_type
                        ? "unavailable: the arena memory could not be mapped"
                        : "off (GOW3_ARENA_HOST_VISIBLE=0 or no suitable memory type)");
    }
    if (arena_mappings) {
        // Device writes made available to the host at the end of every submission: a direct write
        // after a GPU write to the same bytes must not be overwritten by it later (WAW).
        scheduler.SetHostAccessBarrier(true);
        Gow3Stats::arena_mapped.store(true, std::memory_order_relaxed);
    }

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * num_arena_pages) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * num_arena_pages);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
}

BufferCache::~BufferCache() = default;

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

// gow3: the game fills its per-frame buffers (constants, skinning output) sequentially and
// every 4 KiB page cost a protection fault (~110k/s in the test scene, a fifth of each
// render worker's time in the kernel). A fault unprotects the aligned window around it instead;
// pages marked CPU-modified without being written only cost an upload when bound.
// Item: context = BufferCache, source = guest address, destination = host pointer, size,
// extra = the Buffer whose mapping holds the destination (flushed after the copy).
void BufferCache::RunGuestCopy(const Gow3Copy::Item& item) {
    auto* cache = static_cast<BufferCache*>(item.context);
    auto* dst = reinterpret_cast<u8*>(item.destination);
    cache->memory->CopySparseMemory(item.source, dst, item.size);
    auto* buffer = reinterpret_cast<Buffer*>(item.extra);
    buffer->Flush(dst - buffer->mapped_data.data(), item.size);
}

// Small guest copies run on the recording thread (it spins for work: no wakeup, and it is
// idle most of the time); PoolSmallCopies (toggle 524288) batches them for the copy threads.
void BufferCache::SmallGuestCopy(const Gow3Copy::Item& item) {
    // Only the recording thread may Record() or use its copy batch. A draw preparation worker
    // obtaining a buffer copies at once, before the buffer is returned: recording from a worker
    // while the GPU thread handed its chunk to the recorder wrote into a null chunk (crash in
    // SmallGuestCopy, MediEvil), and a worker's thread-local copy batch could stay unflushed.
    if (!scheduler.OwnsRecording() && scheduler.IsRecordingDeferred()) {
        item.run(item);
        return;
    }
    scheduler.NoteHostCopySource(item.source, item.size);
    if (scheduler.IsRecordingDeferred() && !Gow3Toggle::Disabled(Gow3Toggle::PoolSmallCopies)) {
        scheduler.RecordHostCopy([item] { item.run(item); });
        return;
    }
    Gow3Copy::QueueCopy(item);
}

void BufferCache::ExtendWriteFault(VAddr device_addr) {
    static const u64 window = [] {
        const char* env = std::getenv("GOW3_FAULT_WINDOW");
        const u64 kib = env ? std::strtoull(env, nullptr, 10) : 256;
        return std::bit_ceil(std::clamp<u64>(kib, 4, 1024)) * 1024;
    }();
    if (window <= TRACKER_BYTES_PER_PAGE || Gow3Toggle::Disabled(Gow3Toggle::FaultWindow)) {
        return;
    }
    memory_tracker->ExtendWriteFault(Common::AlignDown(device_addr, window), window);
}

namespace {
// gow3 GOW3_PERF_DIAG: write-fault downloads by guest instruction and 64 KiB page: which guest
// code writes into memory a shader wrote (each one a full GPU wait).
struct DownloadSite {
    u64 count = 0, bytes = 0;
};
std::mutex download_sites_mutex;
std::map<std::pair<u64, u64>, DownloadSite> download_sites;
// Who last made each 64 KiB page GPU-written: the ObtainBuffer caller, the binding's shader when
// it was a shader binding (NoteGpuWriterShader), the range, and how often (under the mutex).
struct GpuWriter {
    const char* file = nullptr;
    u32 line = 0;
    u64 shader = 0;
    VAddr address = 0;
    u64 size = 0;
    u64 marks = 0;
};
std::unordered_map<u64, GpuWriter> gpu_writers;
} // namespace

void BufferCache::NoteGpuWriterShader(VAddr device_addr, u64 size, u64 shader_hash) {
    if (!Vulkan::WaitDiagnostics::Enabled() || size == 0) {
        return;
    }
    std::scoped_lock lock{download_sites_mutex};
    for (u64 page = device_addr >> 16; page <= (device_addr + size - 1) >> 16; ++page) {
        if (auto it = gpu_writers.find(page); it != gpu_writers.end()) {
            it->second.shader = shader_hash;
        }
    }
}

void BufferCache::ReportTwins() {
    // gow3: the twin table's size (t-nostats6: panning got slower the longer a session ran; with
    // the 16 KiB twin limit far more pages can be twinned, and every fault and upload searches it).
    u64 count = 0, bytes = 0;
    {
        std::scoped_lock lk{twins_mutex};
        count = twins.size();
        for (const auto& [start, data] : twins) {
            bytes += data.size();
        }
    }
    std::printf("Twins: %llu entries, %.2f MB\n", (unsigned long long)count, bytes / 1e6);
}

void BufferCache::ReportDownloadSites(double seconds) {
    std::vector<std::pair<u64, std::pair<u64, u64>>> top;
    {
        std::scoped_lock lock{download_sites_mutex};
        for (const auto& [key, site] : download_sites) {
            top.push_back({site.count, key});
        }
        download_sites.clear();
    }
    std::sort(top.begin(), top.end(), std::greater<>{});
    const u64 image = reinterpret_cast<u64>(runtime_image_base());
    for (size_t i = 0; i < top.size() && i < 8; ++i) {
        const auto [rip, page] = top[i].second;
        char where[40];
        if (image && rip - image < 0x20000000) {
            std::snprintf(where, sizeof(where), "guest+0x%llx", (unsigned long long)(rip - image));
        } else {
            std::snprintf(where, sizeof(where), "%#llx", (unsigned long long)rip);
        }
        GpuWriter writer;
        {
            std::scoped_lock lock{download_sites_mutex};
            if (const auto it = gpu_writers.find(page); it != gpu_writers.end()) {
                writer = it->second;
            }
        }
        std::string_view file = writer.file ? writer.file : "?";
        file = file.substr(file.find_last_of("/\\") + 1);
        std::printf("Write-fault downloads: %s writing %#llx (64 KiB page): %.1f/s; GPU-written by "
                    "%.*s:%u shader %#llx, range %#llx +%#llx (%llu marks)\n",
                    where, (unsigned long long)(page << 16), top[i].first / seconds,
                    int(file.size()), file.data(), writer.line, (unsigned long long)writer.shader,
                    (unsigned long long)writer.address, (unsigned long long)writer.size,
                    (unsigned long long)writer.marks);
    }
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    const unsigned reason = Gow3Stats::download_reason;
    const u64 rip = reinterpret_cast<u64>(Gow3Stats::fault_rip);
    const auto flush_request = [this, device_addr, size, is_write, reason, rip] {
        if (reason == Gow3Stats::WriteFault && Vulkan::WaitDiagnostics::Enabled() &&
            memory_tracker->IsRegionGpuModified(device_addr, size)) {
            std::scoped_lock lock{download_sites_mutex};
            ++download_sites[{rip, device_addr >> 16}].count;
        }
        // gow3: a guest unmap ends the memory's contents: the GPU's data there is not read back
        // (a download is a full GPU wait, ~4 ms; UE4 decommits pool memory constantly). The
        // next use of the range uploads whatever the guest put there.
        if (reason == Gow3Stats::Unmap && TwinsEnabled()) {
            std::scoped_lock lk{twins_mutex};
            EraseTwins(device_addr, device_addr + size);
        }
        if (reason == Gow3Stats::Unmap && !Gow3Toggle::Disabled(Gow3Toggle::UnmapDiscard)) {
            gpu_modified_ranges.Subtract(device_addr, size);
            ForgetGpuWritten(device_addr, size);
            memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
            Gow3Stats::unmap_discards.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Gow3Stats::DownloadReasonScope scope{reason};
        // Twins: bytes the GPU is still writing on the faulting page are twinned instead of
        // waited for (see TwinsEnabled).
        if (is_write && TwinsEnabled()) {
            const VAddr page_begin = Common::AlignDown(device_addr, TRACKER_BYTES_PER_PAGE);
            const VAddr page_end = Common::AlignUp(device_addr + size, TRACKER_BYTES_PER_PAGE);
            u64 tick = 0, gpu_bytes = 0;
            gpu_modified_ranges.ForEachInRange(page_begin, page_end - page_begin,
                                               [&](VAddr start, VAddr end) {
                                                   gpu_bytes += end - start;
                                                   tick = std::max(tick, WriteTicks().Newest(
                                                                             start, end - start));
                                               });
            if (gpu_bytes != 0 && gpu_bytes <= TwinsMaxBytes() && tick != 0 &&
                !scheduler.IsFree(tick)) {
                {
                    std::scoped_lock lk{twins_mutex};
                    gpu_modified_ranges.ForEachInRange(
                        page_begin, page_end - page_begin, [&](VAddr start, VAddr end) {
                            // Parts with a twin keep their older snapshot: the guest bytes may
                            // have been written since it was taken.
                            ForEachTwinRun(start, end, [&](VAddr from, VAddr to, TwinRun run) {
                                if (run == TwinRun::None) {
                                    std::vector<u8> bytes(to - from);
                                    Core::Memory::Instance()->ReadBacking(from, bytes.data(),
                                                                          to - from);
                                    twins[from] = std::move(bytes);
                                }
                            });
                        });
                }
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                Gow3Stats::twin_faults.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);

        // gow3 GOW3_FAULT_PAGE_DOWNLOAD (default on; 0: always the window below): a write fault
        // on a page whose GPU-written bytes the GPU has finished downloads that page only. The
        // window below takes in other GPU-written ranges, and DownloadMemory waits for the newest
        // write into any of them: t-up5's map oscillation, ~25 faults a second each waiting ~6 ms
        // for an unrelated small range the GPU rewrote every draw (the twin path covers a page
        // whose own bytes are still being written).
        static const bool page_download = [] {
            const char* env = std::getenv("GOW3_FAULT_PAGE_DOWNLOAD");
            return !(env && env[0] == '0');
        }();
        if (is_write && page_download) {
            const VAddr page_begin = Common::AlignDown(device_addr, TRACKER_BYTES_PER_PAGE);
            const VAddr page_end = Common::AlignUp(device_addr + size, TRACKER_BYTES_PER_PAGE);
            u64 tick = 0, gpu_bytes = 0;
            gpu_modified_ranges.ForEachInRange(page_begin, page_end - page_begin,
                                               [&](VAddr start, VAddr end) {
                                                   gpu_bytes += end - start;
                                                   tick = std::max(tick, WriteTicks().Newest(
                                                                             start, end - start));
                                               });
            if (gpu_bytes != 0 && tick != 0 && scheduler.IsFree(tick)) {
                DownloadMemory(arena, page_begin, page_end - page_begin);
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                Gow3Stats::page_downloads.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            // t-map2: the oscillation's faults took none of the paths above. Nothing the GPU wrote
            // on the faulting page: the write needs no GPU data (the page was protected for a
            // neighbour's), so it becomes CPU-modified with no download at all.
            if (gpu_bytes == 0) {
                memory_tracker->UnmarkRegionAsGpuModified(page_begin, page_end - page_begin);
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                Gow3Stats::fault_no_gpu_bytes.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            // Left for the window below (each counted): no write tick known for the page's GPU
            // bytes, or more of them still in flight than a twin takes.
            (tick == 0 ? Gow3Stats::fault_unknown_tick : Gow3Stats::fault_too_big_for_twin)
                .fetch_add(1, std::memory_order_relaxed);
        }

        // GPU-modified ranges come as many small scattered islands,
        // so the download is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
        DownloadMemory(arena, window_start, window_end - window_start);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        flush_request();
    } else {
        liverpool->SendCommand<true>(std::move(flush_request));
    }
}

void BufferCache::WriteBackDownload(VAddr arena_base, std::span<const vk::BufferCopy> copies,
                                    const u8* data, u64 data_offset) {
    std::unique_lock twins_lock{twins_mutex, std::defer_lock};
    if (TwinsEnabled()) {
        twins_lock.lock();
    }
    for (const auto& copy : copies) {
        const VAddr guest = arena_base + copy.srcOffset;
        const u8* src = data + (copy.dstOffset - data_offset);
        if (!TwinsEnabled() || twins.empty()) {
            memory->TryWriteBacking(std::bit_cast<u8*>(guest), src, copy.size);
            continue;
        }
        // Guest bytes written since their twin was made stay; the page is CPU-modified, so its
        // next upload carries them to the arena.
        ForEachTwinRun(guest, guest + copy.size, [&](VAddr from, VAddr to, TwinRun run) {
            if (run != TwinRun::Changed) {
                memory->TryWriteBacking(std::bit_cast<u8*>(from), src + (from - guest), to - from);
            }
        });
        EraseTwins(guest, guest + copy.size);
    }
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    // gow3: queued CPU writes go into the GPU's copy first, and none lands in guest memory while
    // the download writes its data there (the guest's bytes would be overwritten).
    std::scoped_lock cpu_writes_lock{cpu_writes_mutex};
    ProcessCpuWritesLocked();
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        gpu_modified_ranges.Subtract(address, size);
        ForgetGpuWritten(address, size);
    });
    if (total_size_bytes == 0) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    // gow3 D3: the readback queue copies the data once the GPU has done the newest write into it
    // (waiting for that submission only while it is still queued); the graphics queue would
    // first run everything queued (`Finish`: a full GPU wait, up to ~8 ms, whenever the game
    // writes into memory a shader wrote: T-D2c / T-O2d / T-W2b).
    u64 write_tick = 0;
    for (const auto& copy : copies) {
        write_tick = std::max(write_tick, WriteTicks().Newest(arena_base + copy.srcOffset, copy.size));
    }
    auto& readback = SharedReadback(instance);
    if (write_tick != 0 && readback.Available()) {
        if (!scheduler.IsFree(write_tick)) {
            scheduler.Wait(write_tick); // submitted first when it is the current one
        }
        if (const u8* data = readback.Copy(arena, copies, total_size_bytes,
                                           scheduler.GetWorkSemaphore()->Handle(), write_tick)) {
            WriteBackDownload(arena_base, copies, data, 0);
            memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
            Gow3Stats::fast_downloads.fetch_add(1, std::memory_order_relaxed);
            const unsigned reason = Gow3Stats::download_reason;
            Gow3Stats::download_count[reason].fetch_add(1, std::memory_order_relaxed);
            Gow3Stats::download_bytes[reason].fetch_add(total_size_bytes, std::memory_order_relaxed);
            Gow3Stats::download_ns[reason].fetch_add(
                u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count()),
                std::memory_order_relaxed);
            return;
        }
    }
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    scheduler.Finish();
    {
        const unsigned reason = Gow3Stats::download_reason;
        Gow3Stats::download_count[reason].fetch_add(1, std::memory_order_relaxed);
        Gow3Stats::download_bytes[reason].fetch_add(total_size_bytes, std::memory_order_relaxed);
        Gow3Stats::download_ns[reason].fetch_add(
            u64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                     start)
                    .count()),
            std::memory_order_relaxed);
    }

    download.buffer->Invalidate(download.offset, download.size);
    WriteBackDownload(arena_base, copies, download.mapped, download.offset);
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
}

namespace {
// gow3: GOW3_BUFFER_STATS=1 — how buffer bindings reach the GPU, by guest region (256 MiB),
// printed every 5 s: small read-only copies into the stream buffer, arena bindings, and the
// bytes those re-upload after CPU writes. Input for engine-level short paths.
struct BufferStats {
    struct Region {
        u64 stream_count{}, stream_bytes{}, arena_count{}, arena_bytes{}, upload_bytes{};
    };
    std::map<u64, Region> regions;
    /// Hot window: frame of the last binding per 64 KiB block, and the distribution of the
    /// frames between uses (1, 2, 3, 4, 5-8, 9-16, 17+): the ring's reuse distance.
    std::unordered_map<u64, u64> last_use;
    std::array<u64, 7> reuse{};
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
    u64 calls = 0;
};
/// 256 MiB regions; 1 MiB inside the hot 0x104xxxxxxx window (the engine's frame data).
BufferStats& Stats();
void NoteHotUse(VAddr address, u32 size) {
    if ((address >> 28) != 0x104) {
        return;
    }
    auto& st = Stats();
    const u64 frame = Gow3Stats::gpu_frames.load(std::memory_order_relaxed);
    for (u64 block = address >> 16; block <= (address + size - 1) >> 16; ++block) {
        auto [it, inserted] = st.last_use.try_emplace(block, frame);
        if (!inserted && it->second != frame) {
            const u64 d = frame - it->second;
            ++st.reuse[d <= 4 ? d - 1 : d <= 8 ? 4 : d <= 16 ? 5 : 6];
            it->second = frame;
        }
    }
}
u64 RegionKey(VAddr address) {
    return (address >> 28) == 0x104 ? (address >> 20) | (1ull << 40) : address >> 28;
}
bool BufferStatsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("GOW3_BUFFER_STATS");
        return value && value[0] == '1';
    }();
    return enabled;
}
BufferStats& Stats() {
    static BufferStats stats;
    return stats;
}
void PrintBufferStats() {
    auto& st = Stats();
    if ((++st.calls & 4095) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - st.window).count();
    if (seconds < 5.0) {
        return;
    }
    std::printf("Buffer stats (%.1f s, per second): region, stream copies/bytes, arena bindings, "
                "re-uploaded bytes\n", seconds);
    std::printf("  hot window reuse distance in frames (1,2,3,4,5-8,9-16,17+): %llu %llu %llu %llu "
                "%llu %llu %llu\n",
                (unsigned long long)st.reuse[0], (unsigned long long)st.reuse[1],
                (unsigned long long)st.reuse[2], (unsigned long long)st.reuse[3],
                (unsigned long long)st.reuse[4], (unsigned long long)st.reuse[5],
                (unsigned long long)st.reuse[6]);
    st.reuse = {};
    for (const auto& [region, r] : st.regions) {
        const u64 base = (region >> 40) ? (region & ((1ull << 40) - 1)) << 20 : region << 28;
        if ((region >> 40) && r.stream_bytes + r.upload_bytes < 1e6 * seconds) {
            continue; // quiet MiB of the hot window
        }
        std::printf("  %#012llx: %8.0f copies %8.2f MB, %8.0f arena %8.2f MB bound, %8.2f MB uploads\n",
                    static_cast<unsigned long long>(base), r.stream_count / seconds,
                    r.stream_bytes / seconds / 1e6, r.arena_count / seconds,
                    r.arena_bytes / seconds / 1e6,
                    r.upload_bytes / seconds / 1e6);
    }
    st.regions.clear();
    st.window = now;
}
} // namespace

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer,
                                                        std::source_location caller) {
    GOW3_SECTION(ObtainBuffer);
    {
        GOW3_SECTION(ObtReadbacks);
        texture_cache.ResolveReadbacks(device_addr, size, true, ReadbackSource::BufferObtain);
    }
    const bool stats = BufferStatsEnabled();
    if (stats) {
        PrintBufferStats();
        NoteHotUse(device_addr, size);
    }
    // gow3 S2 (study-stage-b-native.md; GOW3_STREAM_CLEAN=1): the stream copy below re-copies a
    // small read-only buffer on every use (~1.6 copies, ~5 KB per draw, ~2 GB/s), whether or not
    // the CPU wrote it. A range the CPU has not written since its last upload is already in the
    // arena: bound there, with no copy and no upload (no render pass break). A CPU-written range
    // is uploaded into the arena outside a render pass (nothing to break) once; written again
    // after that, it is dynamic and keeps the stream copy.
    // The in-game menu switches it live (Performance patches; GOW3_STREAM_CLEAN=0 at start).
    const bool stream_clean =
        Gow3Settings::Get().perf[Gow3Settings::PerfSmallBuffers].load(std::memory_order_relaxed);
    bool small_in_arena = false;
    // gow3: computed once (it was asked twice below), timed for the sections report.
    bool small_gpu_modified = true;
    if (!is_written && size <= STREAM_THRESHOLD) {
        GOW3_SECTION(ObtRegions);
        small_gpu_modified = IsRegionGpuModified(device_addr, size);
    }
    if (stream_clean && !is_written && size <= STREAM_THRESHOLD && !small_gpu_modified) {
        GOW3_SECTION(ObtRegions);
        if (!IsRegionCpuModified(device_addr, size)) {
            small_in_arena = true;
            Gow3Stats::small_arena_binds.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::scoped_lock lk{stream_class_mutex};
            // Direct-mapped: an address in another's slot replaces it (starts again as new).
            auto& entry = stream_class[((device_addr >> 4) * 0x9E3779B97F4A7C15ull >> 48) %
                                       stream_class.size()];
            if (entry.address != device_addr) {
                entry = {device_addr, 0};
            }
            if (entry.state == 1) {
                entry.state = 2; // written again after its upload
                Gow3Stats::small_dynamic.fetch_add(1, std::memory_order_relaxed);
            } else if (entry.state == 0 && !scheduler.IsRendering()) {
                entry.state = 1;
                small_in_arena = true;
                Gow3Stats::small_promotions.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!small_in_arena && !is_written && size <= STREAM_THRESHOLD && !small_gpu_modified) {
        GOW3_SECTION(ObtStream);
        if (stream_clean) Gow3Stats::small_stream_copies.fetch_add(1, std::memory_order_relaxed);
        if (stats) {
            auto& region = Stats().regions[RegionKey(device_addr)];
            ++region.stream_count;
            region.stream_bytes += size;
        }
        // gow3: the guest data is copied on a copy thread, started now; submission and
        // guest-visible fences wait for it (Scheduler::WaitHostCopies).
        if (!stream_buffer.mapped_data.empty() &&
            !Gow3Toggle::Disabled(Gow3Toggle::DeferredStreamCopies)) {
            if (const auto offset = stream_buffer.Reserve(size, instance.UniformMinAlignment())) {
                SmallGuestCopy({
                    .run = &RunGuestCopy,
                    .context = this,
                    .source = device_addr,
                    .destination = reinterpret_cast<u64>(stream_buffer.mapped_data.data() + *offset),
                    .size = size,
                    .extra = reinterpret_cast<u64>(static_cast<Buffer*>(&stream_buffer)),
                });
                return {&stream_buffer, *offset};
            }
        }
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    {
        GOW3_SECTION(ObtResident);
        EnsureResident(arena, first_block, last_block);
    }
    const u64 uploaded_before = Gow3Stats::buffer_upload_bytes.load(std::memory_order_relaxed);
    {
        GOW3_SECTION(ObtSync);
        SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    }
    if (arena_mappings) {
        // gow3 U2: whatever the caller records with this binding uses the range at the current
        // tick (also after a direct write: the next one waits for the GPU to be past it).
        NoteArenaUse(device_addr, size);
    }
    if (stats) {
        auto& region = Stats().regions[RegionKey(device_addr)];
        ++region.arena_count;
        region.arena_bytes += size;
        region.upload_bytes +=
            Gow3Stats::buffer_upload_bytes.load(std::memory_order_relaxed) - uploaded_before;
    }
    if (is_written) {
        if (Vulkan::WaitDiagnostics::Enabled() && size) {
            std::scoped_lock lock{download_sites_mutex};
            for (u64 page = device_addr >> 16; page <= (device_addr + size - 1) >> 16; ++page) {
                auto& writer = gpu_writers[page];
                writer = {caller.file_name(), caller.line(), 0, device_addr, size, writer.marks + 1};
            }
        }
        gpu_modified_ranges.Add(device_addr, size);
        runtime_memory_mark_gpu_written(device_addr, size, 1);
        WriteTicks().Note(device_addr, size, scheduler.CurrentTick());
    }
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    texture_cache.ResolveReadbacks(device_addr, size, true, ReadbackSource::BufferForImage);
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    if (!Gow3Toggle::Disabled(Gow3Toggle::DeferredUploads)) {
        // gow3: texture data is copied on the copy threads (streaming: 100+ MB per frame);
        // the upload reads the staging only after submission, which waits for the copies.
        constexpr u64 Chunk = 1_MB;
        for (u64 offset = 0; offset < staging.size; offset += Chunk) {
            const Gow3Copy::Item item{
                .run = &RunGuestCopy,
                .context = this,
                .source = device_addr + offset,
                .destination = reinterpret_cast<u64>(staging.mapped + offset),
                .size = std::min(Chunk, staging.size - offset),
                .extra = reinterpret_cast<u64>(staging.buffer),
            };
            if (staging.size < Chunk) {
                SmallGuestCopy(item);
            } else {
                scheduler.NoteHostCopySource(item.source, item.size);
                Gow3Copy::Async([item] { item.run(item); });
            }
        }
        return {staging.buffer, staging.offset};
    }
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

// gow3 U2: direct arena writes (research/docs/impl-u2-direct-arena-writes.md). With the arena's
// memory host-visible and mapped (GOW3_ARENA_HOST_VISIBLE), uploads (C1) and precise CPU writes
// (C2) into blocks the GPU is done with are copied by the CPU straight into the arena: no staging,
// no vkCmdCopyBuffer, no barrier after it, no render pass end. The guard: every use of the range
// by the GPU (UseTicks: draws and dispatches binding it, staged copies and fills into it, direct
// writes into it; DMA work for all ranges) was recorded at a tick the GPU has finished. Work
// recorded into the command buffer still open has the current tick, never finished: an earlier
// draw reading the range in it keeps the write on the staging path, whose copy the command
// buffer orders after that draw, and a staged copy still queued keeps a later write behind it.
// Visibility: the mapping is HOST_COHERENT and every direct write ends before the submission that
// may read it (the copy threads like the staging copies: SubmitExecution waits for them), so
// vkQueueSubmit's implicit host write domain operation makes it visible. Device writes are made
// available to the host at the end of each submission (Scheduler::SetHostAccessBarrier), so a
// GPU write the guard saw finished cannot land over a later direct write.
void BufferCache::NoteArenaUse(VAddr addr, u64 size, bool staged) {
    const u64 tick = scheduler.CurrentTick();
    UseTicks().Note(addr, size, tick);
    if (staged && Gow3Stats::enabled) {
        StagedTicks().Note(addr, size, tick);
    }
}

void BufferCache::NoteDmaUse() {
    if (!arena_mappings) {
        return;
    }
    const u64 tick = scheduler.CurrentTick();
    u64 current = dma_use_tick.load(std::memory_order_relaxed);
    while (current < tick && !dma_use_tick.compare_exchange_weak(current, tick,
                                                                 std::memory_order_release,
                                                                 std::memory_order_relaxed)) {
    }
}

unsigned BufferCache::DirectWriteBlocked(VAddr addr, u64 size) {
    if (size == 0) {
        return DirectWriteOk;
    }
    const auto in_flight = [this](u64 tick) {
        // The open command buffer's tick is never finished: no semaphore query for it.
        return tick != 0 && (tick >= scheduler.CurrentTick() || !scheduler.IsFree(tick));
    };
    if (in_flight(std::max(UseTicks().Newest(addr, size),
                           dma_use_tick.load(std::memory_order_acquire)))) {
        if (Gow3Stats::enabled && in_flight(StagedTicks().Newest(addr, size))) {
            return Gow3Stats::ArenaStagedPending;
        }
        return Gow3Stats::ArenaGpuBusy;
    }
    for (u64 block = addr >> block_shift; block <= (addr + size - 1) >> block_shift; ++block) {
        if (!arena_mappings->Get(block)) {
            return Gow3Stats::ArenaNotMapped;
        }
    }
    return DirectWriteOk;
}

template <typename Func>
void BufferCache::ForEachDirectPiece(VAddr addr, u64 size, Func&& func) {
    const VAddr end = addr + size;
    const u64 block_mask = u64{block_size} - 1;
    for (VAddr at = addr; at < end;) {
        u8* host = arena_mappings->Get(at >> block_shift) + (at & block_mask);
        u64 bytes = std::min<u64>(block_size - (at & block_mask), end - at);
        // Blocks taken in order from one chunk are contiguous in its mapping too.
        while (at + bytes < end && arena_mappings->Get((at + bytes) >> block_shift) == host + bytes) {
            bytes += std::min<u64>(block_size, end - (at + bytes));
        }
        func(at, host, bytes);
        at += bytes;
    }
}

void BufferCache::RunDirectCopy(const Gow3Copy::Item& item) {
    auto* cache = static_cast<BufferCache*>(item.context);
    cache->memory->CopySparseMemory(item.source, reinterpret_cast<u8*>(item.destination),
                                    item.size);
}

size_t BufferCache::UploadDirect(boost::container::small_vector<vk::BufferCopy, 4>& copies,
                                 size_t total_size_bytes) {
    // Toggled off, or not in order with the queued draws (Scheduler::DrawRecorderQuiet): every
    // copy takes the staging path, whose Record() waits for them.
    const unsigned all_blocked = Gow3Toggle2::Disabled(Gow3Toggle2::ArenaDirectWrites)
                                     ? unsigned(Gow3Stats::ArenaToggledOff)
                                 : !scheduler.DrawRecorderQuiet() ? unsigned(Gow3Stats::ArenaOffThread)
                                                                  : DirectWriteOk;
    // All decided before any is noted: the copies of one upload never overlap, but may share a
    // page, and a noted direct write would hold back the next copy in it.
    boost::container::small_vector<vk::BufferCopy, 4> direct, staged;
    u64 direct_bytes = 0, staged_bytes = 0;
    for (const auto& copy : copies) {
        const unsigned blocked = all_blocked != DirectWriteOk
                                     ? all_blocked
                                     : DirectWriteBlocked(copy.dstOffset, copy.size);
        if (blocked == DirectWriteOk) {
            direct.push_back(copy);
            direct_bytes += copy.size;
            continue;
        }
        if (Gow3Stats::enabled) {
            Gow3Stats::arena_fallbacks[blocked].fetch_add(1, std::memory_order_relaxed);
        }
        staged.push_back({staged_bytes, copy.dstOffset, copy.size});
        staged_bytes += copy.size;
    }
    if (direct.empty()) {
        return total_size_bytes;
    }
    // Resolve before launching host copies: a copy-thread fault could deadlock submission.
    for (const auto& copy : direct) {
        texture_cache.ResolveReadbacks(copy.dstOffset, copy.size, true, ReadbackSource::BufferUpload);
    }
    Gow3Stats::buffer_upload_bytes.fetch_add(direct_bytes, std::memory_order_relaxed);
    // The guest bytes are copied as the staging path copies them (UploadCopies): on the copy
    // threads in groups of about 1 MiB, small uploads in the calling thread's batch; submission
    // and guest-visible fences wait for them (Scheduler::WaitHostCopies).
    if (Gow3Toggle::Disabled(Gow3Toggle::DeferredUploads)) {
        for (const auto& copy : direct) {
            ForEachDirectPiece(copy.dstOffset, copy.size, [&](VAddr at, u8* host, u64 bytes) {
                memory->CopySparseMemory(at, host, bytes);
            });
        }
    } else if (direct_bytes < 1_MB) {
        for (const auto& copy : direct) {
            ForEachDirectPiece(copy.dstOffset, copy.size, [&](VAddr at, u8* host, u64 bytes) {
                SmallGuestCopy({
                    .run = &RunDirectCopy,
                    .context = this,
                    .source = at,
                    .destination = reinterpret_cast<u64>(host),
                    .size = bytes,
                    .extra = 0,
                });
            });
        }
    } else {
        struct HostCopy {
            VAddr source;
            u8* destination;
            u64 size;
        };
        using Group = boost::container::small_vector<HostCopy, 8>;
        auto group = std::make_shared<Group>();
        u64 group_bytes = 0;
        const auto launch = [&] {
            Gow3Copy::Async([group, memory = memory] {
                for (const auto& copy : *group) {
                    memory->CopySparseMemory(copy.source, copy.destination, copy.size);
                }
            });
        };
        for (const auto& copy : direct) {
            scheduler.NoteHostCopySource(copy.dstOffset, copy.size);
            ForEachDirectPiece(copy.dstOffset, copy.size, [&](VAddr at, u8* host, u64 bytes) {
                group->push_back({at, host, bytes});
                group_bytes += bytes;
                if (group_bytes >= 1_MB) {
                    launch();
                    group = std::make_shared<Group>();
                    group_bytes = 0;
                }
            });
        }
        if (!group->empty()) {
            launch();
        }
    }
    for (const auto& copy : direct) {
        NoteArenaUse(copy.dstOffset, copy.size);
    }
    if (Gow3Stats::enabled) {
        Gow3Stats::arena_direct_calls.fetch_add(1, std::memory_order_relaxed);
        Gow3Stats::arena_direct_bytes.fetch_add(direct_bytes, std::memory_order_relaxed);
        if (staged.empty()) {
            // The staging path's whole copy command, and the pass it would have ended.
            Gow3Stats::arena_copies_saved.fetch_add(1, std::memory_order_relaxed);
            if (scheduler.IsRendering()) {
                Gow3Stats::arena_pass_ends_saved.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    copies.assign(staged.begin(), staged.end());
    return staged_bytes;
}

bool BufferCache::PreciseCpuWrite(VAddr addr, const void* src, int value, u64 size) {
    std::vector<u8> data(size);
    if (src) {
        std::memcpy(data.data(), src, size); // before any lock: src may fault (a read)
    } else {
        std::memset(data.data(), value, size);
    }
    {
        std::scoped_lock lock{cpu_writes_mutex};
        cpu_writes.emplace_back(addr, data);
        cpu_writes_pending.store(true, std::memory_order_release);
    }
    // After queueing: a download from here on has the bytes in the GPU's copy already, and one
    // in progress (it holds the lock) is done before this queued. Outside the lock: the
    // runtime's lock is taken here.
    if (!memory->TryWriteBacking(reinterpret_cast<void*>(addr), data.data(), size)) {
        return false; // part of it has no backing: the normal write faults as before
    }
    Gow3Stats::precise_writes.fetch_add(1, std::memory_order_relaxed);
    Gow3Stats::precise_write_bytes.fetch_add(size, std::memory_order_relaxed);
    return true;
}

void BufferCache::ProcessCpuWritesLocked() {
    cpu_writes_pending.store(false, std::memory_order_relaxed);
    auto writes = std::move(cpu_writes);
    cpu_writes.clear();
    for (const auto& [addr, data] : writes) {
        const u64 size = data.size();
        texture_cache.InvalidateMemory(addr, size);
        if (TwinsEnabled()) {
            std::scoped_lock lk{twins_mutex};
            EraseTwins(addr, addr + size); // the guest wrote them: uploaded below or on next use
        }
        if (!IsRegionGpuModified(addr, size)) {
            // Downloaded meanwhile: guest memory is the current copy, uploaded on the next use.
            memory_tracker->MarkRegionAsCpuModified(addr, size);
            continue;
        }
        const u64 first_block = addr >> block_shift;
        const u64 last_block = (addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);
        EnsureResident(arena, first_block, last_block);
        // gow3 U2: the bytes go straight into the arena mapping when the GPU is done with the
        // blocks (DirectWriteBlocked), else through staging and a GPU copy as before.
        const unsigned blocked =
            !arena_mappings ? DirectWriteOk
            : Gow3Toggle2::Disabled(Gow3Toggle2::ArenaDirectWrites)
                ? unsigned(Gow3Stats::ArenaToggledOff)
            : !scheduler.DrawRecorderQuiet() ? unsigned(Gow3Stats::ArenaOffThread)
                                             : DirectWriteBlocked(addr, size);
        if (arena_mappings && blocked == DirectWriteOk) {
            ForEachDirectPiece(addr, size, [&](VAddr at, u8* host, u64 bytes) {
                std::memcpy(host, data.data() + (at - addr), bytes);
            });
            NoteArenaUse(addr, size);
            if (Gow3Stats::enabled) {
                Gow3Stats::arena_direct_calls.fetch_add(1, std::memory_order_relaxed);
                Gow3Stats::arena_direct_bytes.fetch_add(size, std::memory_order_relaxed);
                Gow3Stats::arena_copies_saved.fetch_add(1, std::memory_order_relaxed);
                if (scheduler.IsRendering()) {
                    Gow3Stats::arena_pass_ends_saved.fetch_add(1, std::memory_order_relaxed);
                }
            }
        } else {
            if (Gow3Stats::enabled) {
                Gow3Stats::arena_staged_calls.fetch_add(1, std::memory_order_relaxed);
                Gow3Stats::arena_staged_bytes.fetch_add(size, std::memory_order_relaxed);
                if (arena_mappings) {
                    Gow3Stats::arena_fallbacks[blocked].fetch_add(1, std::memory_order_relaxed);
                }
                if (scheduler.IsRendering()) {
                    Gow3Stats::arena_staged_pass_ends.fetch_add(1, std::memory_order_relaxed);
                }
            }
            const auto staging = staging_pool.Request(size, MemoryType::HostUncached);
            std::memcpy(staging.mapped, data.data(), size);
            staging.Flush();
            const vk::BufferCopy copy{staging.offset, arena->Offset(addr), size};
            runtime.CopyBuffer(staging.buffer, arena, std::span{&copy, 1});
            if (arena_mappings) {
                NoteArenaUse(addr, size, true);
            }
        }
        WriteTicks().Note(addr, size, scheduler.CurrentTick()); // a download must see it (D3)
        // D2b: the arena and guest memory now agree on these bytes, so they are no longer the
        // GPU's alone. Kept GPU-modified, later direct guest writes there (not libc) faulted
        // and went to a download (T-D2c: 5-10/s). Pages left with no GPU-only bytes are
        // ordinary pages again: CPU writes there mark them CPU-modified without the GPU thread.
        // GOW3_PRECISE_WRITE_RELEASE=0: keep them GPU-modified (D2 as first built). No toggle
        // bit left; bit 63 (D2 off) turns this off too.
        static const bool release = [] {
            const char* env = std::getenv("GOW3_PRECISE_WRITE_RELEASE");
            return !env || env[0] != '0';
        }();
        if (release) {
            gpu_modified_ranges.Subtract(addr, size);
            const VAddr first_page = addr & ~VAddr{4095};
            for (VAddr page = first_page; page < addr + size; page += 4096) {
                bool gpu_only = false;
                gpu_modified_ranges.ForEachInRange(page, 4096, [&](VAddr, VAddr) { gpu_only = true; });
                if (!gpu_only) {
                    memory_tracker->UnmarkRegionAsGpuModified(page, 4096);
                    runtime_memory_mark_gpu_written(page, 4096, 0);
                }
            }
        }
    }
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager->ProcessFaultBuffer();
}

void BufferCache::SynchronizeDmaBuffers() {
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (arena_page_bits - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << arena_page_bits, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << arena_page_bits);
    const u64 first_size = first_arena ? first_arena->size_bytes : arena_page_size;
    const u64 last_size = last_arena ? last_arena->size_bytes : arena_page_size;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = backing.offset + ((start - backing.start) << block_shift),
        });
    });

    u64 base_page = first_addr >> arena_page_bits;
    for (u32 page = 0; page < (first_size >> arena_page_bits); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> arena_page_bits); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    // gow3 R2 (impl-stage-b-per-draw-2.md): residency only grows and every range added below is
    // also set in the bitmap, so all bits set is exactly "no gap": the search below would return
    // with nothing to bind. Second toggle word 65536: the search every time.
    if (!Gow3Toggle2::Disabled(Gow3Toggle2::ResidencyBitmap) &&
        resident_bitmap.All(first_block, last_block + 1)) {
        if (Gow3Stats::enabled) Gow3Stats::Bump(Gow3Stats::residency_checks[0]);
        return;
    }
    if (Gow3Stats::enabled) Gow3Stats::Bump(Gow3Stats::residency_checks[1]);
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }
    Gow3Stats::Timer timer{Gow3Stats::t_resident};

    // gow3: memory from the residency pool (allocated ahead, off this thread); without it, one
    // allocation for all the ranges, as before.
    vk::DeviceMemory device_memory{};
    u8* device_memory_host = nullptr;
    if (!residency_pool->Enabled()) {
        const vk::MemoryAllocateInfo alloc_info = {
            .allocationSize = resident_blocks << block_shift,
            .memoryTypeIndex = arena_memory_type_index,
        };
        device_memory = Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));
        if (arena_mappings) {
            device_memory_host = ResidencyPool::Map(instance.GetDevice(), device_memory);
        }
    }

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        // A range may span two pool chunks: one backing and one bind per piece.
        for (u64 block = range.start; block < range.end;) {
            ResidencyPool::Piece piece{device_memory, memory_offset, range.end - block,
                                       device_memory_host ? device_memory_host + memory_offset
                                                          : nullptr};
            if (residency_pool->Enabled()) {
                piece = residency_pool->Take(range.end - block);
            } else {
                memory_offset += piece.blocks << block_shift;
            }
            if (arena_mappings && piece.host) {
                for (u64 i = 0; i < piece.blocks; ++i) {
                    arena_mappings->Set(block + i, piece.host + (i << block_shift));
                }
            }
            Backing backing;
            backing.start = block;
            backing.end = block + piece.blocks;
            backing.memory = piece.memory;
            backing.offset = piece.offset;
            resident_ranges.Add(backing);
            resident_bitmap.Set(backing.start, backing.end);

            LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

            binds->binds.emplace_back(vk::SparseMemoryBind{
                .resourceOffset = (block << block_shift) - arena->cpu_addr,
                .size = piece.blocks << block_shift,
                .memory = piece.memory,
                .memoryOffset = piece.offset,
            });
            block += piece.blocks;
        }

        const u64 range_offset = (range.start << block_shift) - arena->cpu_addr;
        const u64 range_size = (range.end - range.start) << block_shift;
        for (u64 block = 0; block < range_size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + range_offset + block;
        }
        const u64 copy_size = (range.end - range.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, range.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    const Buffer* src_buffer{};
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 addr, u64 size) {
            copies.emplace_back(total_size_bytes, addr, size);
            total_size_bytes += size;
        },
        [&] {
            if (copies.empty()) {
                return; // nothing CPU-written in the range (UploadCopies would return null)
            }
            GOW3_SECTION(ObtUpload);
            // Twins: bytes whose guest copy still matches its twin keep the GPU's data in the
            // arena; twin bytes the guest wrote since are uploaded and no longer GPU-modified.
            if (TwinsEnabled() && !copies.empty()) {
                std::scoped_lock lk{twins_mutex};
                if (!twins.empty()) {
                    boost::container::small_vector<vk::BufferCopy, 4> kept;
                    boost::container::small_vector<std::pair<VAddr, VAddr>, 4> changed;
                    u64 offset = 0;
                    for (const auto& copy : copies) {
                        ForEachTwinRun(copy.dstOffset, copy.dstOffset + copy.size,
                                       [&](VAddr from, VAddr to, TwinRun run) {
                                           if (run == TwinRun::Unchanged) {
                                               return;
                                           }
                                           if (run == TwinRun::Changed) {
                                               gpu_modified_ranges.Subtract(from, to - from);
                                               changed.emplace_back(from, to);
                                           }
                                           kept.push_back({offset, from, to - from});
                                           offset += to - from;
                                       });
                    }
                    for (const auto& [from, to] : changed) {
                        EraseTwins(from, to);
                    }
                    copies.assign(kept.begin(), kept.end());
                    total_size_bytes = offset;
                }
            }
            // gow3 U2: copies into blocks the GPU is done with go straight into the arena.
            if (arena_mappings && !copies.empty()) {
                total_size_bytes = UploadDirect(copies, total_size_bytes);
            }
            src_buffer = UploadCopies(arena, copies, total_size_bytes);
        });

    if (src_buffer) {
        GOW3_SECTION(ObtUpload);
        if (Gow3Stats::enabled) {
            u64 bytes = 0;
            for (const auto& copy : copies) {
                bytes += copy.size;
            }
            Gow3Stats::arena_staged_calls.fetch_add(1, std::memory_order_relaxed);
            Gow3Stats::arena_staged_bytes.fetch_add(bytes, std::memory_order_relaxed);
            if (scheduler.IsRendering()) {
                Gow3Stats::arena_staged_pass_ends.fetch_add(1, std::memory_order_relaxed);
            }
        }
        runtime.CopyBuffer(src_buffer, arena, copies);
        if (arena_mappings) {
            for (const auto& copy : copies) {
                NoteArenaUse(arena->cpu_addr + copy.dstOffset, copy.size, true);
            }
        }
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

const Buffer* BufferCache::UploadCopies(const Buffer* arena, std::span<vk::BufferCopy> copies,
                                        size_t total_size_bytes) {
    if (copies.empty()) {
        return nullptr;
    }
    // Resolve before launching host copies: a copy-thread fault could deadlock submission.
    for (const auto& copy : copies) {
        texture_cache.ResolveReadbacks(copy.dstOffset, copy.size, true, ReadbackSource::BufferUpload);
    }
    Gow3Stats::buffer_upload_bytes.fetch_add(total_size_bytes, std::memory_order_relaxed);
    const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
    // gow3: the guest memory is copied into staging on the copy threads, started now in
    // groups of about 1 MiB; submission and guest-visible fences wait for them
    // (Scheduler::WaitHostCopies).
    if (!Gow3Toggle::Disabled(Gow3Toggle::DeferredUploads) && total_size_bytes < 1_MB) {
        // Small uploads join the calling thread's batch.
        for (auto& copy : copies) {
            SmallGuestCopy({
                .run = &RunGuestCopy,
                .context = this,
                .source = copy.dstOffset,
                .destination = reinterpret_cast<u64>(staging.mapped + copy.srcOffset),
                .size = copy.size,
                .extra = reinterpret_cast<u64>(staging.buffer),
            });
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        return staging.buffer;
    }
    if (!Gow3Toggle::Disabled(Gow3Toggle::DeferredUploads)) {
        struct HostCopy {
            VAddr source;
            u8* destination;
            u64 size;
        };
        using Group = boost::container::small_vector<HostCopy, 8>;
        auto group = std::make_shared<Group>();
        u64 group_bytes = 0;
        const auto launch = [&] {
            Gow3Copy::Async([group, memory = memory, staging] {
                for (const auto& copy : *group) {
                    memory->CopySparseMemory(copy.source, copy.destination, copy.size);
                }
                staging.Flush();
            });
        };
        for (auto& copy : copies) {
            scheduler.NoteHostCopySource(copy.dstOffset, copy.size);
            group->push_back({copy.dstOffset, staging.mapped + copy.srcOffset, copy.size});
            group_bytes += copy.size;
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
            if (group_bytes >= 1_MB) {
                launch();
                group = std::make_shared<Group>();
                group_bytes = 0;
            }
        }
        if (!group->empty()) {
            launch();
        }
        return staging.buffer;
    }
    const auto copy = [&](std::size_t i) {
        memory->CopySparseMemory(copies[i].dstOffset, staging.mapped + copies[i].srcOffset,
                                 copies[i].size);
    };
    if (total_size_bytes >= 2_MB && copies.size() > 1) {
        Gow3Copy::ParallelFor(copies.size(), copy);
    } else {
        for (std::size_t i = 0; i < copies.size(); ++i) {
            copy(i);
        }
    }
    for (auto& copy : copies) {
        copy.srcOffset += staging.offset;
        copy.dstOffset -= arena->cpu_addr;
    }
    staging.Flush();
    return staging.buffer;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else if (*type == TextureCache::MetaType::CMask && [] {
                       static const bool fill = [] {
                           const char* env = std::getenv("GOW3_CMASK_FILL");
                           return !(env && env[0] == '0');
                       }();
                       return fill;
                   }()) {
            // gow3: color images are always fully expanded on the host, so every 4-bit CMask
            // tile code reads as 0xF (uncompressed: the color in memory is valid). MediEvil reads
            // CMask through buffers; leaving it stale made it treat tiles as fast-cleared.
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, 0xFFFFFFFFu);
            return true;
        } else {
            // Once per type: MediEvil reads CMask through buffers every frame.
            static std::atomic<u32> warned{0};
            const u32 bit = 1u << static_cast<u32>(*type);
            if (!(warned.fetch_or(bit) & bit)) {
                LOG_WARNING(Render_Vulkan, "Unhandled metadata type {} (reported once)",
                            magic_enum::enum_name(*type));
            }
        }
    }
    // gow3: most texel buffers alias no image; remember misses until images change.
    const u64 generation = texture_cache.RegistryGeneration();
    auto& miss = image_miss_cache[((device_addr >> 6) ^ size * 0x9E3779B1u) % image_miss_cache.size()];
    if (miss.address == device_addr && miss.size == size && miss.generation == generation &&
        !Gow3Toggle::Disabled(Gow3Toggle::TextureBindingMemo)) {
        return false;
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        miss = {device_addr, size, generation};
        return false;
    }
    // gow3: the lookups above only read what the texture binding helper leaves alone; the
    // copy below changes image state, so the helper finishes first.
    runtime.BeforeImageAccess();
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    pending_binds.clear();
}

} // namespace VideoCore
