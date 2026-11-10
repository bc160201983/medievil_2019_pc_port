// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <xxhash.h>
#include "video_core/renderer_vulkan/ui_composition.h"
#include "gow3_decode_pacing.h"
#include "gow3_fence_stats.h"
#include "gow3_sections.h"
#include "gow3_settings.h"
#include "gow3_toggles.h"
#include "video_core/renderer_vulkan/vk_frame_capture.h"
#include "common/debug.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_depth_stencil_state.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "gow3_threads.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, page_manager{this},
      buffer_cache{instance, scheduler, runtime, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, runtime, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    // Before the rasterizer is bound: Liverpool enqueues buffers only once it sees it.
    draw_prep = std::make_unique<DrawPreparation>(pipeline_cache);
    // gow3 (impl-texset-index.md): the texture-set memo's compact layout, made here as the table
    // before it was (its pages written, huge pages collapsed: not on stage B).
    MakeCompactTextureSets();
    scene_targets = std::make_unique<SceneTargets>(instance, scheduler, runtime, texture_cache);
    // Object motion first: it fixes the buffer addresses the motion shader variants embed.
    object_motion = std::make_unique<ObjectMotion>(instance, scheduler);
    camera_motion = std::make_unique<CameraMotion>(instance, scheduler, texture_cache, runtime);
    camera_motion->SetObjectMotion(object_motion.get());
    upscaler = std::make_unique<TemporalUpscaler>(instance, scheduler, texture_cache, runtime,
                                                  *camera_motion, *scene_targets);
    if (Ue4FrameMap::Enabled()) {
        frame_map = std::make_unique<Ue4FrameMap>();
    }
    if (Ue4Motion::Enabled() || Ue4Dlss::Enabled()) {
        ue4_motion = std::make_unique<Ue4Motion>(instance, scheduler, texture_cache, runtime);
    }
    if (Ue4Dlss::Enabled()) {
        ue4_dlss = std::make_unique<Ue4Dlss>(instance, scheduler, texture_cache, runtime,
                                             *ue4_motion);
    }
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);

    GpuProfiler::Init(instance, scheduler);
    if (DrawPipeWanted()) {
        constant_ring = std::make_unique<ConstantRing>(instance, scheduler);
        // gow3 (impl-deck-cpu-bundle.md): stage B out of packets wakes a recorder that polled
        // labels left asleep (lean mode), so they are never held back by an idle stage B.
        draw_pipe = std::make_unique<DrawPipe>(&RunDrawPacket, this, [](void* self) {
            static_cast<Rasterizer*>(self)->scheduler.WakeDeferredRecorder();
        });
        // gow3: commands recorded on stage A while stage B still has queued draws land between
        // them: an EndRendering from a buffer copy closed the render pass stage B was drawing
        // in (NVIDIA driver crash in level 3, issue #2/#12). Stage A first waits for stage B,
        // except under the texture cache lock (stage B may need it): those callers drain
        // before taking the lock (TextureCache::UpdateImage / FindImage).
        scheduler.SetDrawPipeSync(
            [](void* self, bool drain) {
                auto* rasterizer = static_cast<Rasterizer*>(self);
                if (!drain) {
                    // gow3 U2 (Scheduler::DrawRecorderQuiet): stage B, or stage A with nothing
                    // queued (only stage A reads its own queue position reliably).
                    return !(DrawPipe::OnStageB() || !rasterizer->draw_pipe ||
                             (rasterizer->OnStageA() && rasterizer->draw_pipe->Idle()));
                }
                if (DrawPipe::OnStageB() || !rasterizer->draw_pipe || rasterizer->draw_pipe->Idle()) {
                    return false;
                }
                if (!rasterizer->OnStageA() || rasterizer->texture_cache.LockHeldByCurrentThread()) {
                    return true;
                }
                rasterizer->DrainDrawPipe(DrawPipe::ReasonForeignRecord);
                return false;
            },
            this);
    }
    // gow3: this thread joins the texture binding helper before it changes image state.
    runtime.SetImageAccessHook(&JoinBindHelper, this);
    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
    });
    scheduler.CountSubmits(); // gow3 GOW3_FRAME_STATS: submits per frame (gow3_gpu_fences.h)
}

Rasterizer::~Rasterizer() = default;

inline const AmdGpu::Regs& Rasterizer::Regs() const {
    return stage_regs ? *stage_regs : liverpool->regs;
}

inline AmdGpu::CbDbExtent Rasterizer::CbExtent(u32 index) const {
    return stage_regs ? pipe_cb_extent[index] : liverpool->last_cb_extent[index];
}

inline const AmdGpu::ComputeProgram& Rasterizer::CsRegs() const {
    return stage_cs ? *stage_cs : liverpool->GetCsRegs();
}

inline AmdGpu::CbDbExtent Rasterizer::DbExtent() const {
    return stage_regs ? pipe_db_extent : liverpool->last_db_extent;
}

namespace {
/// magic_enum::enum_contains, from a table: it scans every enumerator (per texture per draw).
bool IsKnownFormat(AmdGpu::DataFormat data_fmt, AmdGpu::NumberFormat num_fmt) {
    static const auto tables = [] {
        std::pair<std::array<bool, 256>, std::array<bool, 256>> t{};
        for (const auto value : magic_enum::enum_values<AmdGpu::DataFormat>()) {
            if (static_cast<u32>(value) < 256) {
                t.first[static_cast<u32>(value)] = true;
            }
        }
        for (const auto value : magic_enum::enum_values<AmdGpu::NumberFormat>()) {
            if (static_cast<u32>(value) < 256) {
                t.second[static_cast<u32>(value)] = true;
            }
        }
        return t;
    }();
    const u32 d = static_cast<u32>(data_fmt), n = static_cast<u32>(num_fmt);
    return d < 256 && n < 256 && tables.first[d] && tables.second[n];
}
} // namespace

bool Rasterizer::FilterDraw() {
    const auto& regs = Regs();
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

template <typename... Parts>
VideoCore::ImageId Rasterizer::FindTargetMemoized(VideoCore::TextureCache::ImageDesc& desc,
                                                  LastTarget& last, auto&& make_desc,
                                                  const Parts&... parts) {
    static_assert((sizeof(Parts) + ...) <= 256, "target memo key too large");
    std::array<u8, 256> key{};
    u32 key_size = 0;
    const auto append = [&](const auto& part) {
        std::memcpy(key.data() + key_size, &part, sizeof(part));
        key_size += sizeof(part);
    };
    (append(parts), ...);
    const u64 generation = texture_cache.RegistryGeneration();
    const bool memo_enabled = !Gow3Toggle::Disabled(Gow3Toggle::TextureBindingMemo);
    if (memo_enabled && last.generation == generation && last.key_size == key_size &&
        std::memcmp(last.key.data(), key.data(), key_size) == 0) {
        texture_cache.MarkFound(last.image_id);
        return last.image_id;
    }
    const auto remember = [&](VideoCore::ImageId image_id) {
        last.key = key;
        last.key_size = key_size;
        last.generation = generation;
        last.image_id = image_id;
    };
    u64 hash = 0xCBF29CE484222325ull;
    for (u32 i = 0; i < key_size; i += 8) {
        u64 word;
        std::memcpy(&word, key.data() + i, 8);
        hash = (hash ^ word) * 0x100000001B3ull;
        hash ^= hash >> 29;
    }
    auto& memo = target_memo[hash % target_memo.size()];
    if (memo.generation == generation && memo.key_size == key_size &&
        std::memcmp(memo.key.data(), key.data(), key_size) == 0 && memo_enabled) {
        desc = memo.desc;
        texture_cache.MarkFound(memo.image_id);
        remember(memo.image_id);
        return memo.image_id;
    }
    GOW3_SECTION(RsTarget);
    make_desc();
    const VideoCore::ImageId image_id = texture_cache.FindImage(desc);
    if (generation == texture_cache.RegistryGeneration()) {
        memo.key = key;
        memo.key_size = key_size;
        memo.generation = generation;
        memo.image_id = image_id;
        memo.desc = desc;
        remember(image_id);
    } else {
        last.generation = ~0ULL;
    }
    return image_id;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    GOW3_SECTION(PrepareRenderState);
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = Regs();
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = CbExtent(cb);
        const u32 tag = 1;
        image_id = bound_images.emplace_back(FindTargetMemoized(
            desc, last_targets[cb], [&] { std::construct_at(&desc, col_buf, hint); }, tag,
            col_buf, hint));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if (GetEffectiveDepthStencilState(regs).needs_attachment) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = DbExtent();
        auto& [image_id, desc] = db_desc;
        const u32 tag = 2;
        image_id = bound_images.emplace_back(FindTargetMemoized(
            desc, last_targets[AmdGpu::NUM_COLOR_BUFFERS],
            [&] {
                std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                                  htile_address, hint);
            },
            tag, regs.depth_buffer, regs.depth_view, regs.depth_control, htile_address, hint));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }
    // gow3: the G-buffer pass (5+ color targets) holds the scene depth, and its constants the
    // main camera (shadow passes bind the same layout with the light's camera).
    gbuffer_draw = camera_motion->Enabled() && std::popcount(key.mrt_mask) >= 5 && db_desc.first;
    if (gbuffer_draw) {
        camera_motion->OnGBufferPass(db_desc.first);
    }
    if (upscaler->Enabled() && cb_descs[0].first) {
        upscaler->OnColorTarget(cb_descs[0].first);
    }
    // gow3: scene color: a full-size RGBA16F target drawn with the scene depth.
    if (upscaler->Enabled() && db_desc.first && db_desc.first == camera_motion->Depth() &&
        cb_descs[0].first && std::popcount(key.mrt_mask) <= 2) {
        const auto& color = texture_cache.GetImage(cb_descs[0].first);
        const auto& depth = texture_cache.GetImage(db_desc.first);
        if (color.info.pixel_format == vk::Format::eR16G16B16A16Sfloat &&
            color.info.size.width == depth.info.size.width &&
            color.info.size.height == depth.info.size.height) {
            upscaler->OnSceneColor(cb_descs[0].first);
            // Blended geometry without depth writes (not full-screen passes): transparents and
            // effects start. Blended layers writing depth (decals, wet/blood ground) are
            // surfaces and stay in the history.
            if (regs.blend_control[0].enable && !regs.color_buffers[0].info.blend_bypass &&
                !regs.depth_control.depth_write_enable &&
                (regs.num_indices > 6 || regs.num_instances.NumInstances() > 1)) {
                upscaler->OnBlendedSceneDraw();
            } else if (regs.num_indices <= 6 && regs.num_instances.NumInstances() <= 1) {
                // A full-screen pass over the scene (the fog composite): the transparents are in.
                upscaler->OnSceneComposite();
            }
        }
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::Info& info,
    const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.UserData()[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.UserData()[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::DrawClass(const GraphicsPipeline* pipeline, Gow3DrawClasses::Scope& scope) {
    if (!scope.start) {
        return;
    }
    const auto& regs = Regs();
    Gow3DrawClasses::Shape shape;
    const u32 num_color = std::bit_width(pipeline->GetGraphicsKey().mrt_mask);
    for (u32 cb = 0; cb < num_color && cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        if (!cb_descs[cb].first) {
            continue;
        }
        if (shape.colors++ == 0) {
            const auto format = texture_cache.GetImage(cb_descs[cb].first).info.pixel_format;
            scope.target = regs.color_buffers[cb] ? regs.color_buffers[cb].Address() : 0;
            shape.color0_rgba32f = format == vk::Format::eR32G32B32A32Sfloat;
            shape.color0_rgba16f = format == vk::Format::eR16G16B16A16Sfloat;
            shape.color0_rg16 = format == vk::Format::eR16G16Unorm;
            shape.display = regs.color_buffers[cb] &&
                            FrameCapture::IsDisplayBuffer(regs.color_buffers[cb].Address());
        }
    }
    if (db_desc.first) {
        const auto format = texture_cache.GetImage(db_desc.first).info.pixel_format;
        shape.depth = true;
        shape.depth16 = format == vk::Format::eD16Unorm || format == vk::Format::eD16UnormS8Uint;
        shape.depth_write = regs.depth_control.depth_write_enable;
    }
    scope.cls = Gow3DrawClasses::Classify(shape);
    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    scope.ps = ps ? ps->pgm_hash : 0;
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = Regs().color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, CbExtent(0));
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    runtime.ClearImage(&image, desc.view_info.range, clear_value);
    ScopeMarkerEnd();
}

// gow3: two-stage draw pipeline (vk_draw_pipe.h). Packet layout, 8-byte aligned parts:
// DrawPacket, u16 block indices, u32 block words (RegDirty::BlockWords each), then per stage a
// PacketStage with its user data and flattened user data, then (verification) the full file.
namespace {
enum class PacketKind : u32 { Draw, Task, Dispatch, Special, Indirect, IndirectDispatch };
struct TaskPacket {
    PacketKind kind;
    u32 size;
    Rasterizer::OrderedTask task;
};
struct DrawPacket {
    PacketKind kind;
    const Pipeline* pipeline;
    const PreparedDraw* prepared;
    u32 index_offset;
    u8 is_indexed;
    u8 regs_reset;
    u16 num_blocks;
    u32 num_stages;
    u32 verify;
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> cb_extent;
    AmdGpu::CbDbExtent db_extent;
    u64 ring_end; ///< constant ring position after this packet's copies
};
struct PacketStage {
    const Shader::Info* info;
    VAddr pgm_base;
    u32 ud_size;
    u32 flat_size;
    u32 num_ring;
    u32 pad;
};
constexpr u32 AlignPacket(u32 size) {
    return (size + 7) & ~7u;
}
u32 VerifyInterval() {
    static const u32 interval = [] {
        const char* env = std::getenv("GOW3_PIPE_VERIFY");
        return env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    return interval;
}
} // namespace

bool Rasterizer::DrawPipeWanted() {
    const char* env = std::getenv("GOW3_DRAW_PIPE");
    if (env && env[0]) {
        return env[0] == '1';
    }
    // Stage B spins while draws flow. Measured ahead with 16 threads (+19%) and with 4 cores /
    // 8 threads (taskset, Steam Deck-like: +18%).
    return Gow3Threads::Available() >= 8;
}

bool Rasterizer::UseDrawPipe() const {
    return draw_pipe && !host_markers_enabled && !Gow3Toggle::Disabled(Gow3Toggle::DrawPipeline);
}

bool Rasterizer::OnStageA() const {
#ifdef _WIN32
    // gow3 (Windows): libc++ compares std::thread::id through winpthreads' pthread_equal,
    // which asks the kernel (GetThreadId) for both threads: two system calls per check, many
    // per draw. The numeric thread id comes from the TEB.
    return static_cast<u32>(gettid()) == liverpool->GetGpuCommandProcessorThreadId();
#else
    return std::this_thread::get_id() == liverpool->GetGpuCommandProcessorThread();
#endif
}

namespace {
/// Stage A: cycles waited per drain site (GOW3_FRAME_STATS), keyed by function name and line.
std::unordered_map<const char*, std::unordered_map<u32, u64>> drain_sites;
} // namespace

void Rasterizer::DrainDrawPipe(u32 reason, u32 line, const char* function) {
    if (!draw_pipe || !OnStageA()) {
        return;
    }
    static const bool stats = std::getenv("GOW3_FRAME_STATS") != nullptr;
    if (!stats) {
        draw_pipe->Drain(reason);
        return;
    }
    const u64 before = draw_pipe->drain_cycles;
    draw_pipe->Drain(reason);
    if (const u64 waited = draw_pipe->drain_cycles - before) {
        drain_sites[function][line] += waited;
    }
}

bool Rasterizer::IsGpuSideThread() const {
    return OnStageA() || DrawPipe::OnStageB();
}

bool Rasterizer::IsGpuSideThreadId(u32 tid) const {
    // The userfaultfd thread handles the fault while the faulting thread waits: without locks
    // only if that thread owns the caches. The GPU command thread does not while the draw
    // recording thread may be running, so its faults take the locked path of guest threads.
    if (draw_pipe) {
        return tid == draw_pipe->StageBThreadId();
    }
    return tid == liverpool->GetGpuCommandProcessorThreadId();
}

void Rasterizer::NotePendingGpuWrite(VAddr address, u64 size) {
    if (!draw_pipe || !size) {
        return;
    }
    // Draws write the same buffers over and over: an entry for the range only moves on.
    const u64 position = draw_pipe->Head();
    const VAddr end = address + size;
    // A2: the entry grows by [address, end) at most (a merge is the union of two ranges that
    // touch), so marking the new range keeps every byte of every entry marked.
    pending_pages.Mark(address, end);
    for (auto& write : pending_writes) {
        if (address <= write.end && write.begin <= end) {
            write.begin = std::min(write.begin, address);
            write.end = std::max(write.end, end);
            write.position = position;
            pending_min = std::min(pending_min, address);
            pending_max = std::max(pending_max, end);
            return;
        }
    }
    pending_writes.push_back({address, end, position});
    pending_min = std::min(pending_min, address);
    pending_max = std::max(pending_max, end);
}

bool Rasterizer::PendingFilterOn() {
    static const bool wanted = [] {
        const char* env = std::getenv("GOW3_PENDING_FILTER");
        return env && env[0] == '1';
    }();
    return wanted && !Gow3Toggle2::Disabled(Gow3Toggle2::PendingWriteFilter);
}

bool Rasterizer::PendingWriteOverlaps(VAddr address, u64 size) {
    if (Gow3Stats::enabled) {
        Gow3StageA::Add(Gow3StageA::pending_checks);
    }
    if (pending_writes.empty() || address >= pending_max || address + size <= pending_min) {
        if (Gow3Stats::enabled) {
            Gow3StageA::Add(Gow3StageA::pending_bounds);
        }
        return false;
    }
    // Now and then drop entries whose writes the recording thread has done: the bounds shrink.
    if ((++pending_checks & 63) == 0) {
        std::erase_if(pending_writes, [&](const PendingWrite& write) {
            return draw_pipe->Reached(write.position);
        });
        pending_min = ~VAddr{0};
        pending_max = 0;
        pending_pages.Clear();
        for (const auto& write : pending_writes) {
            pending_min = std::min(pending_min, write.begin);
            pending_max = std::max(pending_max, write.end);
            pending_pages.Mark(write.begin, write.end);
        }
    }
    // gow3 A2 (impl-stage-a-decode.md; GOW3_PENDING_FILTER=1, second toggle word 2097152 off):
    // the pages of the range are on no entry, so no entry overlaps it and the scan below would
    // find nothing. After the pruning above, so the entries (and every later answer) are the
    // same as without the filter.
    if (size != 0 && PendingFilterOn() && pending_pages.Untouched(address, address + size)) {
        if (Gow3Stats::enabled) {
            Gow3StageA::Add(Gow3StageA::pending_filtered);
        }
        return false;
    }
    if (Gow3Stats::enabled) {
        Gow3StageA::Add(Gow3StageA::pending_scanned, pending_writes.size());
    }
    const VAddr end = address + size;
    for (const auto& write : pending_writes) {
        if (address < write.end && write.begin < end && !draw_pipe->Reached(write.position)) {
            return true;
        }
    }
    return false;
}

/// gow3 GOW3_FRAME_STATS: guest-draw pipeline binds, and the ones with the previous guest draw's
/// handle ("Binds per frame" line; an estimate: host passes in between are not counted).
static void NoteGuestPipelineBind(vk::Pipeline handle) {
    if (!Gow3Stats::enabled) {
        return;
    }
    thread_local vk::Pipeline last;
    Gow3Stats::pipeline_binds.fetch_add(1, std::memory_order_relaxed);
    if (handle == last) {
        Gow3Stats::pipeline_bind_repeats.fetch_add(1, std::memory_order_relaxed);
    }
    last = handle;
}

void Rasterizer::CollectRingBindings(const Shader::Info& stage, const PreparedDraw* prepared,
                           boost::container::static_vector<RingBinding, Shader::NUM_BUFFERS>& out) {
    const PreparedStage* prepared_stage = nullptr;
    if (prepared && !Gow3Toggle::Disabled(Gow3Toggle::PreparedResources)) {
        for (u32 i = 0; i < prepared->num_stages; ++i) {
            const auto& candidate = prepared->stages[i];
            if (&candidate.program->info == &stage &&
                candidate.num_buffers == stage.buffers.size()) {
                prepared_stage = &candidate;
                break;
            }
        }
    }
    const u64 alignment = std::max<u64>(instance.StorageMinAlignment(),
                                        instance.UniformMinAlignment());
    // gow3 ring reuse (launcher / menu perf_ring_reuse, live; GOW3_RING_REUSE=0 starts off):
    // within one guest command buffer a guest range's bytes do not change (the CPU wrote them
    // before submitting; ranges the GPU writes are skipped above the copy), so a range bound again
    // binds the first copy. Graphics queue only; the table resets at each command buffer and each
    // WAIT_REG_MEM (CloseRingReuse). Off from N-15 (9-10 ms frames while panning, cause not found)
    // until the ring stamp fix (N-16); on again since N-32: heavy-area 0.1% lows ~145 -> ~185, no
    // full ring, no panning spikes. Second toggle word bit 4 switches it off live (A/B): copies made
    // while on keep their hold until the epoch closes, so switching either way mid-epoch is safe.
    const bool reuse = Gow3Settings::PerfOn(Gow3Settings::PerfRingReuse) &&
                       !Gow3Toggle2::Disabled(Gow3Toggle2::RingReuse) && liverpool &&
                       liverpool->OnGfxQueue();
    const auto copy = [&](u32 index, const void* source, VAddr address, u64 size) {
        RingReuse* entry = nullptr;
        if (reuse && address) {
            u64 hash = (address ^ size << 40) * 0x9E3779B97F4A7C15ull;
            entry = &ring_reuse[(hash >> 52) % ring_reuse.size()];
            if (entry->epoch == ring_reuse_epoch && entry->address == address &&
                entry->size == size) {
                Gow3Stats::constant_ring_reused_bytes.fetch_add(size, std::memory_order_relaxed);
                if (Gow3Stats::enabled) {
                    Gow3StageA::Add(Gow3StageA::ring_reused);
                }
                out.push_back({index, static_cast<u32>(size), entry->offset, address});
                return;
            }
        }
        u64 absolute = 0;
        const auto offset = constant_ring->Allocate(size, alignment, &absolute);
        if (!offset) {
            return;
        }
        if (entry) {
            constant_ring->Hold(absolute);
            *entry = {address, static_cast<u32>(size), ring_reuse_epoch, *offset};
        }
        u8* dst = constant_ring->Data(*offset);
        if (source) {
            std::memcpy(dst, source, size);
        } else {
            memory->CopySparseMemory(address, dst, size);
        }
        constant_ring->Flush(*offset, size);
        if (Gow3Stats::enabled) {
            Gow3StageA::Add(Gow3StageA::ring_copied);
        }
        out.push_back({index, static_cast<u32>(size), *offset, address});
    };
    for (u32 index = 0; index < stage.buffers.size(); ++index) {
        const auto& desc = stage.buffers[index];
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::Flatbuf &&
                !stage.flattened_ud_buf.empty()) {
                copy(index, stage.flattened_ud_buf.data(), 0,
                     stage.flattened_ud_buf.size() * sizeof(u32));
            }
            continue;
        }
        const auto vsharp = prepared_stage ? prepared_stage->buffer_sharps[index]
                                           : desc.GetSharp(stage);
        const VAddr address = vsharp.base_address;
        if (address == 0 || vsharp.GetSize() == 0) {
            continue;
        }
        if (Gow3Stats::enabled) {
            Gow3StageA::Add(Gow3StageA::ring_bindings);
        }
        const u64 size = memory->ClampRangeSize(address, vsharp.GetSize());
        if (desc.is_written) {
            NotePendingGpuWrite(address, size);
            continue;
        }
        // The stream path of BufferCache::ObtainBuffer, taken here: small, read-only, not
        // written by the GPU (now or by work still queued for the recording thread).
        if (size == 0 || size > VideoCore::BufferCache::STREAM_THRESHOLD ||
            buffer_cache.IsRegionGpuModified(address, size) || PendingWriteOverlaps(address, size)) {
            continue;
        }
        copy(index, nullptr, address, size);
    }
}

void Rasterizer::PostDraw(const Pipeline* pipeline, const PreparedDraw* used_prepared,
                          bool is_indexed, u32 index_offset, const AmdGpu::ComputeProgram* cs,
                          const IndirectDraw* indirect) {
    auto& dirty = liverpool->pipe_dirty;
    if (!pipe_synced) {
        // Stage B has not run yet: its register copy starts as this one.
        pipe_regs = liverpool->regs;
        dirty.Clear();
        pipe_synced = true;
    }
    std::array<u16, AmdGpu::RegDirty::NumBlocks> blocks;
    u32 num_blocks = 0;
    dirty.blocks.ForEach([&](u32 block) { blocks[num_blocks++] = static_cast<u16>(block); });
    const auto stages =
        pipeline ? pipeline->GetStages() : std::span<const Shader::Info* const>{};
    // Constants: copied here into the ring, the recording thread only binds them.
    thread_local std::array<boost::container::static_vector<RingBinding, Shader::NUM_BUFFERS>,
                            Shader::MaxStageTypes>
        rings;
    u32 num_stages = 0;
    u32 size = sizeof(DrawPacket) + AlignPacket(num_blocks * sizeof(u16)) +
               num_blocks * AmdGpu::RegDirty::BlockWords * sizeof(u32);
    for (const auto* stage : stages) {
        if (stage) {
            auto& ring = rings[num_stages];
            ring.clear();
            if (constant_ring && !Gow3Toggle::Disabled(Gow3Toggle::ConstantRing)) {
                CollectRingBindings(*stage, used_prepared, ring);
            }
            ++num_stages;
            size += sizeof(PacketStage) +
                    AlignPacket(u32(stage->user_data.size() + stage->flattened_ud_buf.size()) *
                                sizeof(u32)) +
                    u32(ring.size() * sizeof(RingBinding));
        }
    }
    if (cs) {
        size += AlignPacket(sizeof(AmdGpu::ComputeProgram));
    }
    if (indirect) {
        size += AlignPacket(sizeof(IndirectDraw));
    }
    const u32 interval = VerifyInterval();
    const bool verify = interval && (draw_pipe->packets % interval) == 0;
    if (verify) {
        size += sizeof(AmdGpu::Regs);
    }

    u8* out = draw_pipe->Begin(size);
    u8* const start = out;
    auto& packet = *reinterpret_cast<DrawPacket*>(out);
    packet.kind = cs         ? (indirect ? PacketKind::IndirectDispatch : PacketKind::Dispatch)
                  : indirect ? PacketKind::Indirect
                  : pipeline ? PacketKind::Draw
                             : PacketKind::Special;
    packet.pipeline = pipeline;
    packet.prepared = used_prepared;
    packet.index_offset = index_offset;
    packet.is_indexed = is_indexed;
    packet.regs_reset = dirty.reset;
    packet.num_blocks = static_cast<u16>(num_blocks);
    packet.num_stages = num_stages;
    packet.verify = verify;
    packet.cb_extent = liverpool->last_cb_extent;
    packet.db_extent = liverpool->last_db_extent;
    packet.ring_end = constant_ring ? constant_ring->Position() : 0;
    out += sizeof(DrawPacket);
    std::memcpy(out, blocks.data(), num_blocks * sizeof(u16));
    out += AlignPacket(num_blocks * sizeof(u16));
    for (u32 i = 0; i < num_blocks; ++i) {
        std::memcpy(out,
                    liverpool->regs.reg_array.data() + blocks[i] * AmdGpu::RegDirty::BlockWords,
                    AmdGpu::RegDirty::BlockWords * sizeof(u32));
        out += AmdGpu::RegDirty::BlockWords * sizeof(u32);
    }
    u32 stage_index = 0;
    for (const auto* stage : stages) {
        if (!stage) {
            continue;
        }
        const auto& ring = rings[stage_index++];
        auto& header = *reinterpret_cast<PacketStage*>(out);
        header = {stage,
                  stage->pgm_base,
                  static_cast<u32>(stage->user_data.size()),
                  static_cast<u32>(stage->flattened_ud_buf.size()),
                  static_cast<u32>(ring.size()),
                  0};
        out += sizeof(PacketStage);
        std::memcpy(out, ring.data(), ring.size() * sizeof(RingBinding));
        out += ring.size() * sizeof(RingBinding);
        std::memcpy(out, stage->user_data.data(), stage->user_data.size_bytes());
        std::memcpy(out + stage->user_data.size_bytes(), stage->flattened_ud_buf.data(),
                    stage->flattened_ud_buf.size() * sizeof(u32));
        out += AlignPacket(u32(stage->user_data.size() + stage->flattened_ud_buf.size()) *
                           sizeof(u32));
    }
    if (cs) {
        std::memcpy(out, cs, sizeof(AmdGpu::ComputeProgram));
        out += AlignPacket(sizeof(AmdGpu::ComputeProgram));
    }
    if (indirect) {
        std::memcpy(out, indirect, sizeof(IndirectDraw));
        out += AlignPacket(sizeof(IndirectDraw));
    }
    if (verify) {
        std::memcpy(out, &liverpool->regs, sizeof(AmdGpu::Regs));
        out += sizeof(AmdGpu::Regs);
    }
    dirty.Clear();
    draw_pipe->Commit(static_cast<u32>(out - start));
    PrintPipeStats();
}

void Rasterizer::RetireSubmission() {
    if (!draw_pipe || !OnStageA()) {
        return;
    }
    while (!pipe_keepalive.empty() && draw_pipe->Reached(pipe_keepalive.front().first)) {
        pipe_keepalive.pop_front();
    }
    if (!draw_pipe->Idle()) {
        pipe_keepalive.emplace_back(draw_pipe->Head(), draw_prep->KeepCurrent());
    }
}

bool Rasterizer::RunInOrder(OrderedTask task, const void* data, u32 size, u64 toggle) {
    if (!UseDrawPipe() || Gow3Toggle::Disabled(toggle)) {
        DrainDrawPipe(DrawPipe::ReasonRasterizer);
        task(*this, static_cast<const u8*>(data));
        return false;
    }
    const u32 total = sizeof(TaskPacket) + AlignPacket(size);
    u8* out = draw_pipe->Begin(total);
    *reinterpret_cast<TaskPacket*>(out) = {PacketKind::Task, size, task};
    std::memcpy(out + sizeof(TaskPacket), data, size);
    draw_pipe->Commit(total);
    return true;
}

void Rasterizer::RunDrawPacket(void* context, const u8* data, u32 size) {
    GOW3_SECTION(Packet);
    auto& self = *static_cast<Rasterizer*>(context);
    if (*reinterpret_cast<const PacketKind*>(data) == PacketKind::Task) {
        const auto& task = *reinterpret_cast<const TaskPacket*>(data);
        GOW3_SECTION(Task);
        const u64 task_start = Gow3Sections::Enabled() ? __rdtsc() : 0;
        task.task(self, data + sizeof(TaskPacket));
        if (task_start) {
            Gow3Sections::NoteTask(reinterpret_cast<const void*>(task.task), __rdtsc() - task_start);
        }
        return;
    }
    const auto& packet = *reinterpret_cast<const DrawPacket*>(data);
    const u8* in = data + sizeof(DrawPacket);
    const auto* blocks = reinterpret_cast<const u16*>(in);
    in += AlignPacket(packet.num_blocks * sizeof(u16));
    auto& regs = self.pipe_regs;
    if (packet.regs_reset) {
        regs.SetDefaults();
    }
    for (u32 i = 0; i < packet.num_blocks; ++i) {
        std::memcpy(regs.reg_array.data() + blocks[i] * AmdGpu::RegDirty::BlockWords, in,
                    AmdGpu::RegDirty::BlockWords * sizeof(u32));
        in += AmdGpu::RegDirty::BlockWords * sizeof(u32);
    }
    ASSERT(packet.num_stages <= Shader::Info::MaxUdSnapshots);
    for (u32 i = 0; i < packet.num_stages; ++i) {
        const auto& stage = *reinterpret_cast<const PacketStage*>(in);
        in += sizeof(PacketStage);
        self.ring_stages[i] = {stage.info, reinterpret_cast<const RingBinding*>(in),
                               stage.num_ring};
        in += stage.num_ring * sizeof(RingBinding);
        const auto* words = reinterpret_cast<const u32*>(in);
        Shader::Info::ud_snapshots[i] = {stage.info, words, stage.ud_size, words + stage.ud_size,
                                         stage.flat_size, stage.pgm_base};
        in += AlignPacket((stage.ud_size + stage.flat_size) * sizeof(u32));
    }
    Shader::Info::num_ud_snapshots = packet.num_stages;
    self.num_ring_stages = packet.num_stages;
    const AmdGpu::ComputeProgram* cs = nullptr;
    if (packet.kind == PacketKind::Dispatch || packet.kind == PacketKind::IndirectDispatch) {
        cs = reinterpret_cast<const AmdGpu::ComputeProgram*>(in);
        in += AlignPacket(sizeof(AmdGpu::ComputeProgram));
    }
    const IndirectDraw* indirect = nullptr;
    if (packet.kind == PacketKind::Indirect || packet.kind == PacketKind::IndirectDispatch) {
        indirect = reinterpret_cast<const IndirectDraw*>(in);
        in += AlignPacket(sizeof(IndirectDraw));
    }
    if (packet.verify) {
        // The resource tables the GPU thread walked must read the same now: else a write
        // still queued here (WriteData, DMA) changed them after that thread read them.
        static u32 flat_reports = 0;
        for (u32 i = 0; i < packet.num_stages && flat_reports < 32; ++i) {
            const auto& snapshot = Shader::Info::ud_snapshots[i];
            const auto& srt = snapshot.info->srt_info;
            // Stages the pipeline selection did not refresh (no user data) keep their tables.
            if (!srt.walker_func || snapshot.flat_size == 0 || snapshot.user_data_size == 0) {
                continue;
            }
            std::vector<u32> flat(snapshot.flat_size);
            std::memcpy(flat.data(), snapshot.user_data,
                        std::min(snapshot.user_data_size, snapshot.flat_size) * sizeof(u32));
            // Pointers in the tables may be stale by now: a fault only skips the check.
            sigjmp_buf recover;
            if (sigsetjmp(recover, 0)) {
                runtime_fault_recover = nullptr;
                continue;
            }
            runtime_fault_recover = &recover;
            srt.walker_func(snapshot.user_data, flat.data());
            runtime_fault_recover = nullptr;
            if (std::memcmp(flat.data(), snapshot.flat, snapshot.flat_size * sizeof(u32)) != 0) {
                u32 first = 0, count = 0;
                for (u32 w = 0; w < snapshot.flat_size; ++w) {
                    if (flat[w] != snapshot.flat[w]) {
                        first = count++ ? first : w;
                    }
                }
                std::printf("Draw pipe: resource tables of shader %016llx changed after the GPU "
                            "thread read them (%u of %u words, first %u: %#x -> %#x, user data "
                            "%u words)\n",
                            static_cast<unsigned long long>(snapshot.info->pgm_hash), count,
                            snapshot.flat_size, first, snapshot.flat[first], flat[first],
                            snapshot.user_data_size);
                ++flat_reports;
            }
        }
        const auto* live = reinterpret_cast<const AmdGpu::Regs*>(in);
        in += sizeof(AmdGpu::Regs);
        static u32 reports = 0;
        for (u32 block = 0; block < AmdGpu::RegDirty::NumBlocks && reports < 32; ++block) {
            const u32 at = block * AmdGpu::RegDirty::BlockWords;
            if (std::memcmp(regs.reg_array.data() + at, live->reg_array.data() + at,
                            AmdGpu::RegDirty::BlockWords * sizeof(u32)) != 0) {
                for (u32 w = at; w < at + AmdGpu::RegDirty::BlockWords; ++w) {
                    if (regs.reg_array[w] != live->reg_array[w]) {
                        std::printf("Draw pipe: register copy differs at word %#x (%#x vs %#x)\n",
                                    w, regs.reg_array[w], live->reg_array[w]);
                        ++reports;
                        break;
                    }
                }
            }
        }
    }
    ASSERT(static_cast<u32>(in - data) == size);
    stage_regs = &regs;
    self.pipe_cb_extent = packet.cb_extent;
    self.pipe_db_extent = packet.db_extent;
    if (packet.kind == PacketKind::Special) {
        // A draw FilterDraw handles (fast clear elimination, resolve, depth copy, skip).
        FrameCapture::Poll();
        self.scheduler.PopPendingOperations();
        self.BeforeRecord();
        self.FilterDraw();
    } else if (cs) {
        stage_cs = cs;
        const auto* pipeline = static_cast<const ComputePipeline*>(packet.pipeline);
        if (indirect) {
            self.DispatchIndirectRecord(pipeline, indirect->args, indirect->stride);
        } else {
            self.DispatchRecord(pipeline);
        }
        stage_cs = nullptr;
    } else if (indirect) {
        self.DrawIndirectRecord(static_cast<const GraphicsPipeline*>(packet.pipeline),
                                packet.is_indexed, *indirect);
    } else {
        self.DrawRecord(static_cast<const GraphicsPipeline*>(packet.pipeline), packet.prepared,
                        packet.is_indexed, packet.index_offset);
    }
    Shader::Info::num_ud_snapshots = 0;
    self.num_ring_stages = 0;
    if (self.constant_ring) {
        self.constant_ring->Stamp(packet.ring_end);
    }
}

void Rasterizer::PrintPipeStats() {
    static const bool stats = std::getenv("GOW3_FRAME_STATS") != nullptr;
    if (!stats || (draw_pipe->packets & 1023) != 0) {
        return;
    }
    static auto window = std::chrono::steady_clock::now();
    static u64 last_packets = 0, last_drains = 0, last_drain_cycles = 0, last_busy = 0;
    static u64 last_tsc = __rdtsc();
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - window).count();
    if (seconds < 5.0) {
        return;
    }
    const u64 tsc = __rdtsc();
    const double cycles = double(tsc - last_tsc);
    const u64 busy = draw_pipe->busy_cycles.load(std::memory_order_relaxed);
    std::printf("Draw pipe: %.0f draws/s pipelined, %.0f drains/s that waited, stage A waited "
                "%.1f%%, stage B busy %.1f%%\n",
                (draw_pipe->packets - last_packets) / seconds,
                (draw_pipe->drains - last_drains) / seconds,
                100.0 * (draw_pipe->drain_cycles - last_drain_cycles) / cycles,
                100.0 * (busy - last_busy) / cycles);
    if (Gow3Sections::Enabled()) {
        // gow3 GOW3_SECTIONS: stage B's time per packet by part (inclusive), in microseconds.
        static std::array<u64, Gow3Sections::Count> last_sections{};
        const double packets = double(draw_pipe->packets - last_packets);
        const double tsc_hz = cycles / seconds;
        std::printf("  stage B us per packet:");
        for (u32 s = 0; s < Gow3Sections::Count; ++s) {
            const u64 now_cycles = Gow3Sections::cycles[s].load(std::memory_order_relaxed);
            if (packets > 0) {
                std::printf(" %s %.3f", Gow3Sections::Names[s],
                            1e6 * double(now_cycles - last_sections[s]) / tsc_hz / packets);
            }
            last_sections[s] = now_cycles;
        }
        static u64 last_record_calls = 0;
        const u64 record_calls = Gow3Sections::record_calls.load(std::memory_order_relaxed);
        if (packets > 0) {
            std::printf(" (%.1f Record calls per packet)", double(record_calls - last_record_calls) / packets);
        }
        last_record_calls = record_calls;
        std::printf("\n  stage B tasks (us per packet, per second): ");
        static std::array<std::pair<u64, u64>, Gow3Sections::task_kinds.size()> last_tasks{};
#ifdef _WIN32
        const auto exe = reinterpret_cast<uintptr_t>(GetModuleHandleA(nullptr));
#else
        const uintptr_t exe = 0;
#endif
        for (size_t k = 0; k < Gow3Sections::task_kinds.size(); ++k) {
            const auto& kind = Gow3Sections::task_kinds[k];
            const void* fn = kind.fn.load(std::memory_order_relaxed);
            if (!fn) {
                break;
            }
            const u64 c = kind.cycles.load(std::memory_order_relaxed),
                      n = kind.count.load(std::memory_order_relaxed);
            if (packets > 0 && n != last_tasks[k].second) {
                std::printf(" exe+%#llx %.3f (%.0f/s)",
                            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(fn) - exe),
                            1e6 * double(c - last_tasks[k].first) / tsc_hz / packets,
                            double(n - last_tasks[k].second) / seconds);
            }
            last_tasks[k] = {c, n};
        }
        std::printf("\n");
    }
    std::printf("  drains by reason/s:");
    for (u32 r = 0; r < DrawPipe::NumReasons; ++r) {
        if (const u64 n = draw_pipe->drains_by_reason[r]) {
            static constexpr const char* names[] = {"ConstRam", "commands", "compute",
                                                    "submission end", "draw", "rasterizer",
                                                    "foreign record"};
            const double share = 100.0 * draw_pipe->cycles_by_reason[r] / cycles;
            draw_pipe->cycles_by_reason[r] = 0;
            if (r < 256) {
                std::printf(" %s=%.0f (%.1f%%)",
                            magic_enum::enum_name(AmdGpu::PM4ItOpcode(r)).data(), n / seconds,
                            share);
            } else {
                std::printf(" %s=%.0f (%.1f%%)", names[r - 256], n / seconds, share);
            }
        }
    }
    std::printf("\n  stage A waits over 0.5%%:");
    for (const auto& [function, lines] : drain_sites) {
        for (const auto& [line, waited] : lines) {
            if (waited * 200 > cycles) {
                std::printf(" %s:%u %.1f%%", function, line, 100.0 * waited / cycles);
            }
        }
    }
    std::printf("\n");
    drain_sites.clear();
    draw_pipe->drains_by_reason = {};
    window = now;
    last_tsc = tsc;
    last_packets = draw_pipe->packets;
    last_drains = draw_pipe->drains;
    last_drain_cycles = draw_pipe->drain_cycles;
    last_busy = busy;
}

bool Rasterizer::FilterDrawPasses() const {
    // FilterDraw's checks without its side effects: false when it would skip the draw or run
    // a pass of its own (fast clear elimination, resolve, depth/stencil copy).
    const auto& regs = Regs();
    using Mode = AmdGpu::ColorControl::OperationMode;
    const auto mode = regs.color_control.mode;
    if (mode == Mode::EliminateFastClear || mode == Mode::FmaskDecompress ||
        mode == Mode::Resolve || regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return false;
    }
    const bool depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const bool stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    return !(mode == Mode::Disable && (depth_copy || stencil_copy));
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset, const PreparedDraw* prepared) {
    RENDERER_TRACE;
    Gow3Stats::draws.fetch_add(1, std::memory_order_relaxed);

    // gow3: with the draw pipeline this thread only selects the pipeline and hands the draw
    // to the recording thread (DrawRecord there); draws FilterDraw handles itself run here.
    const bool pipelined = UseDrawPipe();
    if (pipelined && !FilterDrawPasses() &&
        !Gow3Toggle::Disabled(Gow3Toggle::PipelinedMemoryWrites)) {
        PostDraw(nullptr, nullptr, false, 0);
        return;
    }
    if (!pipelined || !FilterDrawPasses()) {
        DrainDrawPipe(DrawPipe::ReasonDraw);
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
        BeforeRecord();
        if (!FilterDraw()) {
            return;
        }
    }
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline({}, prepared);
    const PreparedDraw* used_prepared = pipeline_cache.UsedPrepared();
    if (prepared) {
        const u32 state = prepared->state.load(std::memory_order_acquire);
        const bool match = prepared->reg_checksum == liverpool->gfx_reg_checksum;
        draw_prep->Count(pipeline != nullptr && state == PreparedDraw::Ready && match);
        if (Gow3Stats::enabled && state != PreparedDraw::Ready) {
            (state == PreparedDraw::Pending ? Gow3Stats::unprep_pending : Gow3Stats::unprep_unavailable)
                .fetch_add(1, std::memory_order_relaxed);
        } else if (Gow3Stats::enabled && !match) {
            Gow3Stats::unprep_checksum.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (!pipeline) {
        return;
    }
    if (pipelined && FilterDrawPasses()) {
        PostDraw(pipeline, used_prepared, is_indexed, index_offset);
        return;
    }
    DrawRecord(pipeline, used_prepared, is_indexed, index_offset);
}

void Rasterizer::DrawRecord(const GraphicsPipeline* pipeline, const PreparedDraw* used_prepared,
                            bool is_indexed, u32 index_offset) {
    GOW3_SECTION(DrawRecord);
    // gow3 N0: this draw's stage B time and class (GOW3_FRAME_STATS only).
    Gow3DrawClasses::Scope draw_class{Gow3Stats::enabled};
    if (DrawPipe::OnStageB()) {
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
        BeforeRecord();
    }
    const auto& regs = Regs();
    // gow3 GOW3_GPU_PROFILE: frames also end at the display pass without the upscaler's motion
    // vectors (MediEvil with upscaler off never reached NoteFrameStart: no profile was printed).
    // Only the first draw of a display pass: the game draws into the display buffer about four
    // times a frame (the image, then the UI), which split each frame into slivers (T-P960 profile 4:
    // 480 "frames" a second, 0.02 ms each).
    if (auto* profiler = GpuProfiler::Get(); profiler && !camera_motion->Enabled()) {
        static bool in_display_pass = false;
        const bool display = regs.color_buffers[0] &&
                             FrameCapture::IsDisplayBuffer(regs.color_buffers[0].Address());
        if (display && !in_display_pass) {
            profiler->BeginFrame();
        }
        in_display_pass = display;
    }
    // gow3: the pass copying a finished frame to a display buffer; the previous draw's
    // target is that frame.
    if (camera_motion->Enabled() && regs.color_buffers[0] &&
        FrameCapture::IsDisplayBuffer(regs.color_buffers[0].Address())) {
        scene_started = false;
        camera_motion->OnDisplayPass(cb_descs[0].first);
        if (upscaler->OnFrameStart()) {
            object_motion->InvalidateHistory();
            camera_motion->InvalidateHistory();
        }
        camera_motion->SetJitter(upscaler->Jitter());
        object_motion->OnFrameStart();
        NoteFrameStart();
        static const char* scene_debug = std::getenv("GOW3_SCENE_DEBUG");
        scene_debug_frame = scene_debug && std::remove(scene_debug) == 0;
        scene_targets->debug = scene_debug_frame;
    } else if (!camera_motion->Enabled()) {
        // Without camera motion (MediEvil's default) a frame starts at the first draw into a
        // display buffer (another one each frame while menus draw only there): the GPU profile
        // and the frame counter need it too (issue #30).
        const VAddr target = regs.color_buffers[0] ? regs.color_buffers[0].Address() : 0;
        const VAddr display = target && FrameCapture::IsDisplayBuffer(target) ? target : 0;
        if (display && display != display_target) {
            NoteFrameStart();
        }
        display_target = display;
    }
    bind_prepared = used_prepared;
    motion_draw = pipeline->GetGraphicsKey().motion_vectors;
    motion_geometry = 0;

    PrepareRenderState(pipeline);
    DrawClass(pipeline, draw_class);
    if (upscaler->Enabled() && std::popcount(pipeline->GetGraphicsKey().mrt_mask) == 1) {
        const auto& viewport = Regs().viewports[0];
        upscaler->OnDraw(pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash,
                         cb_descs[0].first, db_desc.first,
                         UiComposition::NativeViewport(viewport.xscale * 2, viewport.yscale * 2));
    }
    if (frame_map || (ue4_motion && (Ue4Motion::Enabled() || Ue4Dlss::On()))) [[unlikely]] {
        if (ObserveFrameMap(pipeline)) {
            // gow3 U3: DLSS wrote TAA's output; the TAA draw is not recorded.
            bind_prepared = nullptr;
            ResetBindings(false);
            return;
        }
    }
    const PreparedDraw* draw_prepared = bind_prepared;
    // gow3: vertex and index buffers are resolved while the helper binds textures (their
    // commands are recorded after BeginRendering, as before).
    draw_inputs = {pipeline, draw_prepared, index_offset, is_indexed, true, false};
    const bool bound = BindResources(pipeline);
    bind_prepared = nullptr; // indirect draws and dispatches bind without prepared sharps
    const bool inputs_resolved = draw_inputs.resolved;
    draw_inputs.pending = false;
    if (!bound) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    if (!inputs_resolved) {
        ResolveVertexBuffers(pipeline, draw_prepared);
        if (is_indexed) {
            ResolveIndexBuffer(index_offset);
        }
    }
    EmitVertexBuffers();
    if (is_indexed) {
        EmitIndexBuffer();
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    // gow3: screen-space (clip disabled) draws into the upscaler's output-size images.
    push_data.xscale *= target_scale[0];
    push_data.xoffset *= target_scale[0];
    push_data.yscale *= target_scale[1];
    push_data.yoffset *= target_scale[1];
    if (motion_draw && motion_geometry) {
        const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
        const auto [base_vertex, first_instance] =
            GetDrawOffsets(regs, vs, pipeline->GetFetchShader());
        const u32 index_size =
            regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 2u : 4u;
        const VAddr index_address = is_indexed
            ? regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size : 0;
        // Without a character-size skeleton the draw is gated on its bone palette changing
        // since last frame (model constants follow the camera, the palette does not). The
        // check comes first: most gated draws are static world pieces with long index lists.
        static const bool all_motion = [] {
            const char* value = std::getenv("GOW3_OBJECT_MOTION_ALL");
            return value && value[0] == '1';
        }();
        bool gated = !all_motion;
        u64 palette = 0;
        for (const auto& resource : vs.buffers) {
            if (resource.IsSpecial()) continue;
            const auto buffer = resource.GetSharp(vs);
            const u32 size = buffer.GetSize();
            if (!buffer.Valid() || buffer.GetStride() != 16) continue;
            const auto role = Motion::ClassifyBuffer(size);
            if (role == Motion::BufferRole::Skeleton) {
                gated = false;
                break;
            }
            const VAddr address = buffer.base_address;
            if (role == Motion::BufferRole::SmallSkeleton &&
                memory->IsValidGpuMapping(address, 0) &&
                memory->ClampRangeSize(address, size) == size) {
                palette = XXH3_64bits_withSeed(reinterpret_cast<const void*>(address), size,
                                               palette);
            }
        }
        push_data.motion_param = 0;
        if (!gated || object_motion->Moving({.shader = vs.pgm_hash,
                                             .geometry = motion_geometry,
                                             .indices = index_address,
                                             .index_count = regs.num_indices,
                                             .instances = regs.num_instances.NumInstances(),
                                             .first_instance = first_instance},
                                            palette)) {
            Motion::VertexRange range{base_vertex, regs.num_indices};
            u64 topology = 0;
            if (is_indexed) {
                range = {};
                const u64 bytes = u64(regs.num_indices) * index_size;
                if (index_address && memory->IsValidGpuMapping(index_address, 0) &&
                    memory->ClampRangeSize(index_address, bytes) == bytes) {
                    const bool restart = regs.enable_primitive_restart != 0;
                    const u32 count = regs.num_indices;
                    const auto scanned = object_motion->IndexRange(
                        {index_address, count, index_size, s32(base_vertex), restart}, [&] {
                            const auto* data = reinterpret_cast<const void*>(index_address);
                            const auto scan_range =
                                index_size == 2
                                    ? Motion::IndexedRange(
                                          std::span(static_cast<const u16*>(data), count),
                                          s32(base_vertex), restart)
                                    : Motion::IndexedRange(
                                          std::span(static_cast<const u32*>(data), count),
                                          s32(base_vertex), restart);
                            return Motion::IndexRangeCache::Result{scan_range,
                                                                   XXH3_64bits(data, bytes)};
                        });
                    range = scanned.range;
                    topology = scanned.topology;
                }
            }
            push_data.motion_param = object_motion->PrepareDraw({
                .shader = vs.pgm_hash,
                .geometry = motion_geometry,
                .indices = index_address,
                .topology = topology,
                .index_count = regs.num_indices,
                .instances = regs.num_instances.NumInstances(),
                .first_instance = first_instance,
                .vertices = range,
            });
        }
    }
    {
        GOW3_SECTION(PipelineBind);
        pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                                {buffer_infos.data(), buffer_infos.size()});
    }
    // gow3: jitter geometry drawn with the scene depth, not full-screen passes (a shifted
    // full-screen quad leaves an edge column unwritten).
    draw_jitter = {};
    if (upscaler->Enabled() && db_desc.first && db_desc.first == camera_motion->Depth() &&
        (regs.num_indices > 6 || regs.num_instances.NumInstances() > 1)) {
        draw_jitter = upscaler->Jitter();
    }
    UpdateDynamicState(pipeline, is_indexed);
    MarkPass(pipeline, state);
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const vk::Pipeline handle = pipeline->Handle();
    NoteGuestPipelineBind(handle);
    const vk::Pipeline square = SquarePass(*pipeline);
    const u32 num_indices = regs.num_indices;
    const u32 num_instances = regs.num_instances.NumInstances();
    const u32 first_vertex = vertex_offset;
    const u32 first_instance = instance_offset;
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        const auto draw = [&] {
            if (is_indexed) {
                cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(first_vertex),
                                   first_instance);
            } else {
                cmdbuf.draw(num_indices, num_instances, first_vertex, first_instance);
            }
        };
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
        draw();
        if (square) [[unlikely]] {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, square);
            draw();
        }
    });
    if (FrameCapture::Active()) {
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        FrameCapture::Draw(vs_info.pgm_hash, ps ? ps->pgm_hash : 0, num_indices, num_instances);
        if (const auto& bc = regs.blend_control[0]; bc.enable && regs.color_buffers[0]) {
            char note[128];
            std::snprintf(note, sizeof(note), "\n  blend ps %08x idx %u: src %u dst %u func %u z %u%u",
                          u32(ps ? ps->pgm_hash : 0), num_indices, u32(bc.color_src_factor),
                          u32(bc.color_dst_factor), u32(bc.color_func),
                          u32(regs.depth_control.depth_enable),
                          u32(regs.depth_control.depth_write_enable));
            FrameCapture::Note(note);
        }
    }
    DebugState.IncDrawCall();

    ResetBindings(false);
    scheduler.KickRecording();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    RENDERER_TRACE;
    // gow3: like direct draws, handed to the recording thread after the pipeline selection
    // (else the GPU thread waited here for a whole frame of queued draws).
    const bool pipelined = UseDrawPipe() && FilterDrawPasses() &&
                           !Gow3Toggle::Disabled(Gow3Toggle::PipelinedIndirectDraws);
    if (!pipelined) {
        DrainDrawPipe();
        scheduler.PopPendingOperations();
        BeforeRecord();
        if (!FilterDraw()) {
            return;
        }
    }

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
    };
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params);
    if (!pipeline) {
        return;
    }
    const IndirectDraw indirect = {arg_address + offset, count_address, stride, max_count};
    if (pipelined) {
        PostDraw(pipeline, nullptr, is_indexed, 0, nullptr, &indirect);
        return;
    }
    DrawIndirectRecord(pipeline, is_indexed, indirect);
}

void Rasterizer::DrawIndirectRecord(const GraphicsPipeline* pipeline, bool is_indexed,
                                    const IndirectDraw& indirect) {
    Gow3DrawClasses::Scope draw_class{Gow3Stats::enabled};
    if (DrawPipe::OnStageB()) {
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
        BeforeRecord();
    }
    const VAddr count_address = indirect.count;
    const u32 stride = indirect.stride;
    const u32 max_count = indirect.max_count;

    // Indirect arguments may be GPU-written: leave these draws on camera fallback.
    motion_draw = false;
    motion_geometry = 0;
    PrepareRenderState(pipeline);
    DrawClass(pipeline, draw_class);
    if (upscaler->Enabled() && std::popcount(pipeline->GetGraphicsKey().mrt_mask) == 1) {
        const auto& viewport = Regs().viewports[0];
        upscaler->OnDraw(pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash,
                         cb_descs[0].first, db_desc.first,
                         UiComposition::NativeViewport(viewport.xscale * 2, viewport.yscale * 2));
    }
    if (frame_map || (ue4_motion && (Ue4Motion::Enabled() || Ue4Dlss::On()))) [[unlikely]] {
        ObserveFrameMap(pipeline);
    }
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer();
    }

    const auto [buffer, base] =
        buffer_cache.ObtainBuffer(indirect.args, stride * max_count, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, stride * max_count);

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4, false);
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    // gow3: screen-space (clip disabled) draws into the upscaler's output-size images.
    push_data.xscale *= target_scale[0];
    push_data.xoffset *= target_scale[0];
    push_data.yscale *= target_scale[1];
    push_data.yoffset *= target_scale[1];
    {
        GOW3_SECTION(PipelineBind);
        pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                                {buffer_infos.data(), buffer_infos.size()});
    }
    draw_jitter = {};
    if (upscaler->Enabled() && db_desc.first && db_desc.first == camera_motion->Depth()) {
        draw_jitter = upscaler->Jitter();
    }
    UpdateDynamicState(pipeline, is_indexed);
    MarkPass(pipeline, state);
    scheduler.BeginRendering(state);

    ASSERT(stride == (is_indexed ? sizeof(VkDrawIndexedIndirectCommand)
                                 : sizeof(VkDrawIndirectCommand)));
    const vk::Pipeline handle = pipeline->Handle();
    NoteGuestPipelineBind(handle);
    const vk::Pipeline square = SquarePass(*pipeline);
    const vk::Buffer args = buffer->Handle();
    const u64 args_offset = base;
    const vk::Buffer counts = count_address != 0 ? count_buffer->Handle() : vk::Buffer{};
    const u64 counts_offset = count_address != 0 ? count_offset : 0;
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        for (const vk::Pipeline bound : {handle, square}) {
            if (!bound) {
                break;
            }
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, bound);
            if (is_indexed) {
                if (counts) {
                    cmdbuf.drawIndexedIndirectCount(args, args_offset, counts, counts_offset, max_count,
                                                    stride);
                } else {
                    cmdbuf.drawIndexedIndirect(args, args_offset, max_count, stride);
                }
            } else if (counts) {
                cmdbuf.drawIndirectCount(args, args_offset, counts, counts_offset, max_count, stride);
            } else {
                cmdbuf.drawIndirect(args, args_offset, max_count, stride);
            }
        }
    });
    DebugState.IncDrawCall();

    ResetBindings(false);
    scheduler.KickRecording();
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;
    Gow3Stats::dispatches.fetch_add(1, std::memory_order_relaxed);
    // gow3: like draws, handed to the draw recording thread after the pipeline selection.
    const bool pipelined = UseDrawPipe() && !Gow3Toggle::Disabled(Gow3Toggle::PipelinedDispatch);
    if (!pipelined) {
        DrainDrawPipe();
    }
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    if (pipelined) {
        PostDraw(pipeline, nullptr, false, 0, &CsRegs());
        return;
    }
    DispatchRecord(pipeline);
}

void Rasterizer::DispatchRecord(const ComputePipeline* pipeline) {
    GOW3_SECTION(DispatchRecord);
    FrameCapture::Poll();
    gbuffer_draw = false;

    scheduler.PopPendingOperations();
    BeforeRecord();

    const auto& cs_program = CsRegs();

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    // After the resource binding: its transfers (image uploads, downloads) mark themselves.
    const auto mark = [&] {
        if (auto* profiler = GpuProfiler::Get()) {
            profiler->Mark(cs.pgm_hash ^ 0xD15Aull, [&] {
                return fmt::format("dispatch cs {:016x} ({}x{}x{} groups)", cs.pgm_hash,
                                   cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
            });
        }
    };
    if (upscaler->Enabled()) {
        upscaler->OnDispatch(cs.pgm_hash);
    }
    if (ExecuteShaderHLE(cs, Regs(), cs_program, *this)) {
        return;
    }
    if (frame_map || (ue4_motion && (Ue4Motion::Enabled() || Ue4Dlss::On()))) [[unlikely]] {
        if (ObserveFrameMap(pipeline)) {
            // gow3 U4: DLSS wrote the TAAU dispatch's output; the dispatch is not recorded.
            ResetBindings(true);
            return;
        }
    }

    if (!BindResources(pipeline)) {
        return;
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    mark();
    {
        GOW3_SECTION(PipelineBind);
        pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                                {buffer_infos.data(), buffer_infos.size()});
    }

    const vk::Pipeline handle = pipeline->Handle();
    const u32 dim_x = cs_program.dim_x, dim_y = cs_program.dim_y, dim_z = cs_program.dim_z;
    if (FrameCapture::Active()) {
        FrameCapture::Dispatch(cs.pgm_hash, dim_x, dim_y, dim_z);
    }
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatch(dim_x, dim_y, dim_z);
    });
    DebugState.IncDispatch();

    ResetBindings(true);
    scheduler.KickRecording();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;
    // gow3: handed to the recording thread like direct dispatches.
    const bool pipelined = UseDrawPipe() && !Gow3Toggle::Disabled(Gow3Toggle::PipelinedIndirectDraws);
    if (!pipelined) {
        DrainDrawPipe();
    }
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    if (pipelined) {
        const IndirectDraw indirect = {address + offset, 0, size, 1};
        PostDraw(pipeline, nullptr, false, 0, &CsRegs(), &indirect);
        return;
    }
    DispatchIndirectRecord(pipeline, address + offset, size);
}

void Rasterizer::DispatchIndirectRecord(const ComputePipeline* pipeline, VAddr address,
                                        u32 size) {
    scheduler.PopPendingOperations();
    BeforeRecord();
    const u32 offset = 0;

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    {
        GOW3_SECTION(PipelineBind);
        pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                                {buffer_infos.data(), buffer_infos.size()});
    }

    const vk::Pipeline handle = pipeline->Handle();
    const vk::Buffer args = buffer->Handle();
    const u64 args_offset = base;
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatchIndirect(args, args_offset);
    });
    DebugState.IncDispatch();

    ResetBindings(true);
    scheduler.KickRecording();
}

u64 Rasterizer::Flush() {
    DrainDrawPipe();
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    DrainDrawPipe();
    scheduler.Finish();
}

void Rasterizer::CloseRingReuse() {
    ++ring_reuse_epoch;
    if (constant_ring) {
        constant_ring->CloseReuseEpoch();
    }
}

void Rasterizer::OnSubmit() {
    DrainDrawPipe();
    OnSubmitWork();
}

void Rasterizer::OnSubmitWork() {
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    runtime.TickFrame();
}

bool Rasterizer::OnSubmitInOrder() {
    // gow3 GOW3_ONSUBMIT_STAGE_B (impl-decode-pacing.md, "OnSubmit on stage B"): the frame-end
    // work owns the caches, which stage B owns whenever the draw pipe has work, so it runs there
    // at the same stream position the drain gave it (after everything handed over before, before
    // anything decoded after); the Flush after it too (it drained as well). Stage A goes on
    // decoding the next frame. Prepared draws ahead of it hold guest descriptors (sharps, hashes)
    // and pipelines, not cache objects; every image and buffer is resolved on stage B when it
    // binds, after this step, through memos checked against the registry generation and region
    // stamps the collector bumps.
    if (!Gow3DecodePacing::OnSubmitOnStageB()) {
        return false;
    }
    const u64 handed_ns = Gow3DecodePacing::NowNs();
    RunInOrder(
        [](Rasterizer& self, const u8* data) {
            GOW3_SECTION(TaskSubmit);
            u64 handed;
            std::memcpy(&handed, data, sizeof(handed));
            const u64 start = Gow3DecodePacing::NowNs();
            self.OnSubmitWork();
            {
                GOW3_SECTION(SubmitFlush);
                self.Flush();
            }
            Gow3DecodePacing::NoteOnSubmitRun(handed, start);
        },
        &handed_ns, sizeof(handed_ns));
    return true;
}

const PreparedStage* Rasterizer::FindPreparedStage(const Shader::Info& stage) const {
    // Sharps a draw-preparation worker read from the same flattened user data.
    if (!bind_prepared || Gow3Toggle::Disabled(Gow3Toggle::PreparedResources)) {
        return nullptr;
    }
    for (u32 i = 0; i < bind_prepared->num_stages; ++i) {
        const auto& candidate = bind_prepared->stages[i];
        if (&candidate.program->info == &stage &&
            candidate.num_images == stage.images.size() &&
            candidate.num_samplers == stage.samplers.size() &&
            candidate.num_buffers == stage.buffers.size()) {
            return &candidate;
        }
    }
    return nullptr;
}

bool Rasterizer::BindHelperWanted() {
    // A spinning helper pays off only with cores to spare (the recording thread spins too).
    const char* env = std::getenv("GOW3_TEXTURE_HELPER");
    if (env && env[0]) {
        return env[0] == '1';
    }
    // Measured neutral on 16 threads (docs/parallel_gpu.md): opt-in.
    return false;
}

bool Rasterizer::HelperEligible(const Pipeline* pipeline) const {
    if (pipeline->IsCompute() || !bind_helper.Available() || FrameCapture::Active() ||
        Gow3Toggle::Disabled(Gow3Toggle::TextureBindHelper) ||
        Gow3Toggle::Disabled(Gow3Toggle::TextureBindingMemo) ||
        Gow3Toggle::Disabled(Gow3Toggle::UpdateImageFastPath)) {
        return false;
    }
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        // Mip arrays make the descriptor layout depend on the T#s.
        for (const auto& image : stage->images) {
            if (image.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
                return false;
            }
        }
        // Storage writes invalidate cached images from the buffer side.
        for (const auto& buffer : stage->buffers) {
            if (!buffer.IsSpecial() && buffer.is_written) {
                return false;
            }
        }
    }
    return true;
}

bool Rasterizer::TexturesBindableOnHelper(const Shader::Info& stage,
                                          const PreparedStage* prepared) {
    const u64 generation = texture_cache.RegistryGeneration();
    for (u32 image_index = 0; image_index < stage.images.size(); ++image_index) {
        const auto& image_desc = stage.images[image_index];
        if (image_desc.is_written) {
            return false;
        }
        const auto tsharp =
            prepared ? prepared->image_sharps[image_index] : image_desc.GetSharp(stage);
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid ||
            !memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !IsKnownFormat(data_fmt, num_fmt)) {
            continue; // null descriptor
        }
        const auto& entry =
            prepared ? CachedImageDescEntry(tsharp, image_desc, prepared->image_hashes[image_index])
                     : CachedImageDescEntry(tsharp, image_desc);
        if (entry.found_generation != generation) {
            return false;
        }
        VideoCore::ImageId image_id = entry.found_id;
        auto* image = &texture_cache.GetImage(image_id);
        if (const auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
            image_id = depth_image_id;
            image = &texture_cache.GetImage(image_id);
        }
        if (image->binding.needs_rebind || !texture_cache.IsUpToDate(image_id) ||
            scene_targets->Tracks(image->image_uid) ||
            (upscaler->Enabled() && upscaler->RedirectsSampled(image_id))) {
            return false;
        }
    }
    return true;
}

void Rasterizer::RunTextureTask(void* context) {
    auto& self = *static_cast<Rasterizer*>(context);
    auto& task = self.texture_task;
    for (u32 i = 0; i < task.count; ++i) {
        auto& stage = task.stages[i];
        if (!self.TexturesBindableOnHelper(*stage.info, stage.prepared)) {
            return;
        }
        auto binding = stage.binding;
        u32 write_index = stage.write_index;
        self.BindTextures(*stage.info, stage.prepared, binding, write_index, task.barrier, true);
        task.completed = i + 1;
    }
}

void Rasterizer::JoinBindHelper(void* context) {
    if (!BindHelper::OnHelper()) {
        static_cast<Rasterizer*>(context)->bind_helper.Join();
    }
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    GOW3_SECTION(BindResources);
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }
    NoteStencilPlaneWrites(pipeline);

    set_write_index = 0;
    set_writes.clear();
    buffer_infos.clear();
    image_infos.clear();
    // gow3: G-buffer passes sample material textures for a scene rendered below the output
    // size; the temporal upscaler restores the detail of the output's mip level (FSR guide:
    // log2(render / output)). Shadows, post-processing and UI keep the guest's bias.
    sampler_lod_bias = 0.0f;
    if (!pipeline->IsCompute() && !Gow3Toggle::Disabled(Gow3Toggle::SceneMipBias) &&
        std::popcount(static_cast<const GraphicsPipeline*>(pipeline)->GetGraphicsKey().mrt_mask &
                      0xff) >= 5) {
        sampler_lod_bias = upscaler->SceneMipBias();
        static float reported = 0.0f;
        if (sampler_lod_bias != reported) {
            reported = sampler_lod_bias;
            std::printf("Upscaler: scene texture LOD bias %.2f\n", sampler_lod_bias);
        }
    }

    bool uses_dma = false;

    static const bool stats = std::getenv("GOW3_FRAME_STATS") != nullptr;
    if (stats && ((helper_full + helper_partial + helper_serial) & 1023) == 0) {
        static auto window = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        if (now - window >= std::chrono::seconds(5)) {
            window = now;
            const double forks = std::max<double>(1, helper_full + helper_partial);
            std::printf("Texture helper: %llu binds in parallel, %llu partly, %llu serial; "
                        "per fork %.0f cycles helper task, %.0f cycles GPU thread wait\n",
                        static_cast<unsigned long long>(helper_full),
                        static_cast<unsigned long long>(helper_partial),
                        static_cast<unsigned long long>(helper_serial),
                        bind_helper.task_cycles.exchange(0) / forks,
                        bind_helper.wait_cycles / forks);
            bind_helper.wait_cycles = 0;
            helper_full = helper_partial = helper_serial = 0;
        }
    }

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(Regs());
    if (!HelperEligible(pipeline)) {
        ++helper_serial;
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) {
                continue;
            }
            set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                              stage->samplers.size());
            stage->PushUd(binding, push_data);
            const PreparedStage* prepared = FindPreparedStage(*stage);
            BindBuffers(*stage, prepared, binding, push_data, set_write_index);
            BindTextures(*stage, prepared, binding, set_write_index, needs_barrier);
            uses_dma |= stage->uses_dma;
        }
    } else {
        // gow3: textures on the helper, buffers here. Without mip arrays every stage's
        // descriptors are [buffers][images][samplers], one each, so both sides know their
        // binding numbers and descriptor write slots up front.
        struct BufferStage {
            const Shader::Info* info;
            const PreparedStage* prepared;
            Shader::Backend::Bindings binding;
            u32 write_index;
        };
        std::array<BufferStage, Shader::MaxStageTypes> buffer_stages;
        auto& task = texture_task;
        task.count = 0;
        task.completed = 0;
        task.barrier = false;
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) {
                continue;
            }
            const u32 num_buffers = static_cast<u32>(stage->buffers.size());
            const u32 num_descriptors =
                num_buffers + static_cast<u32>(stage->images.size() + stage->samplers.size());
            set_writes.resize(set_writes.size() + num_descriptors);
            stage->PushUd(binding, push_data);
            const PreparedStage* prepared = FindPreparedStage(*stage);
            buffer_stages[task.count] = {stage, prepared, binding, set_write_index};
            auto texture_binding = binding;
            texture_binding.buffer += num_buffers;
            texture_binding.unified += num_buffers;
            task.stages[task.count] = {stage, prepared, texture_binding,
                                       set_write_index + num_buffers};
            ++task.count;
            binding.buffer += num_buffers;
            binding.unified += num_descriptors;
            set_write_index += num_descriptors;
            uses_dma |= stage->uses_dma;
        }
        bind_helper.Fork(&RunTextureTask, this);
        for (u32 i = 0; i < task.count; ++i) {
            auto& stage = buffer_stages[i];
            BindBuffers(*stage.info, stage.prepared, stage.binding, push_data, stage.write_index);
        }
        if (draw_inputs.pending && !Gow3Toggle::Disabled(Gow3Toggle::EarlyDrawInputs)) {
            ResolveVertexBuffers(draw_inputs.pipeline, draw_inputs.prepared);
            if (draw_inputs.is_indexed) {
                ResolveIndexBuffer(draw_inputs.index_offset);
            }
            draw_inputs.resolved = true;
        }
        bind_helper.Join();
        needs_barrier |= task.barrier;
        ++(task.completed == task.count ? helper_full
                                        : task.completed ? helper_partial : helper_serial);
        // Stages the helper left (an image needing work that records commands).
        for (u32 i = task.completed; i < task.count; ++i) {
            auto& stage = task.stages[i];
            u32 write_index = stage.write_index;
            BindTextures(*stage.info, stage.prepared, stage.binding, write_index, needs_barrier);
        }
    }

    if (uses_dma) {
        buffer_cache.SynchronizeDmaBuffers();
        buffer_cache.NoteDmaUse(); // gow3 U2: it may read any resident block
        fault_process_pending = true;
    }

    return true;
}

void Rasterizer::BindVertexBuffers(const GraphicsPipeline* pipeline, const PreparedDraw* prepared) {
    ResolveVertexBuffers(pipeline, prepared);
    EmitVertexBuffers();
}

void Rasterizer::EmitVertexBuffers() {
    GOW3_SECTION(EmitVertexBuffers);
    // gow3: only the used entries go into the recording chunk (the static vectors hold 32 of
    // each: ~3 KiB copied per draw).
    auto& v = vertex_binds;
    const bool dynamic_input = instance.IsVertexInputDynamicState();
    const u32 num_buffers = v.num_buffers;
    scheduler.ReserveRecordData(
        v.bindings.size() * sizeof(v.bindings[0]) + v.attributes.size() * sizeof(v.attributes[0]) +
        num_buffers * (sizeof(vk::Buffer) + 3 * sizeof(vk::DeviceSize)) + 128);
    if (dynamic_input) {
        scheduler.Record([bindings = scheduler.RecordData(std::span<const vk::VertexInputBindingDescription2EXT>{v.bindings.data(), v.bindings.size()}),
                          attributes = scheduler.RecordData(std::span<const vk::VertexInputAttributeDescription2EXT>{
                              v.attributes.data(), v.attributes.size()})](
                             vk::CommandBuffer cmdbuf) {
            cmdbuf.setVertexInputEXT(bindings, attributes);
        });
    }
    if (num_buffers == 0) {
        return;
    }
    const auto buffers = scheduler.RecordData(std::span<const vk::Buffer>{v.host_buffers.data(), num_buffers});
    const auto offsets = scheduler.RecordData(std::span<const vk::DeviceSize>{v.host_offsets.data(), num_buffers});
    if (dynamic_input) {
        scheduler.Record([num_buffers, buffers, offsets](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindVertexBuffers(0, num_buffers, buffers.data(), offsets.data());
        });
        return;
    }
    const auto sizes = scheduler.RecordData(std::span<const vk::DeviceSize>{v.host_sizes.data(), num_buffers});
    const auto strides = scheduler.RecordData(std::span<const vk::DeviceSize>{v.host_strides.data(), num_buffers});
    scheduler.Record([num_buffers, buffers, offsets, sizes, strides](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindVertexBuffers2(0, num_buffers, buffers.data(), offsets.data(), sizes.data(),
                                  strides.data());
    });
}

void Rasterizer::ResolveVertexBuffers(const GraphicsPipeline* pipeline,
                                      const PreparedDraw* prepared) {
    GOW3_SECTION(ResolveVertexBuffers);
    const auto& regs = Regs();
    auto& v = vertex_binds;
    v.num_buffers = 0;
    v.host_buffers.clear();
    v.host_offsets.clear();
    v.host_sizes.clear();
    v.host_strides.clear();
    auto& attributes = v.attributes;
    auto& bindings = v.bindings;
    attributes.clear();
    bindings.clear();
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    // Inputs a draw-preparation worker read from the same registers and (verified) user data.
    const auto& fetch = pipeline->GetFetchShader();
    const PreparedVertexInputs* ready =
        prepared && prepared->vertex.valid && instance.IsVertexInputDynamicState() && fetch &&
                fetch->attributes.size() == prepared->vertex.count && !FrameCapture::Active() &&
                !Gow3Toggle::Disabled(Gow3Toggle::PreparedResources)
            ? &prepared->vertex
            : nullptr;
    if (ready) {
        attributes.assign(ready->attributes, ready->attributes + ready->count);
        bindings.assign(ready->bindings, ready->bindings + ready->count);
        guest_buffers.assign(ready->buffers, ready->buffers + ready->count);
    } else {
        pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                                  regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }
    if (motion_draw && !guest_buffers.empty()) {
        // Include every stream: identical position buffers can be paired with different
        // skinning/instance data. Index topology and base offsets are matched separately.
        motion_geometry = ready ? ready->buffers_hash
                                : XXH3_64bits(guest_buffers.data(),
                                              guest_buffers.size() * sizeof(AmdGpu::Buffer));
    }
    // Object motion research: vertex streams of G-buffer draws.
    if (FrameCapture::Active() && gbuffer_draw) {
        for (const auto& vb : guest_buffers) {
            char note[128];
            std::snprintf(note, sizeof(note), "\n  gb vb at %#llx size %u stride %u",
                          (unsigned long long)vb.base_address, u32(vb.GetSize()),
                          u32(vb.GetStride()));
            FrameCapture::Note(note);
        }
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    VertexInputs<BufferRange> ranges{};
    VertexInputs<BufferRange> ranges_merged{};
    if (ready) {
        for (u32 i = 0; i < ready->num_ranges; ++i) {
            ranges_merged.emplace_back(ready->ranges[i].base, ready->ranges[i].end);
        }
    } else {
        for (const auto& buffer : guest_buffers) {
            if (buffer.base_address != 0 && buffer.GetSize() > 0) {
                ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
            }
        }
    }

    // Merge connecting ranges together
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        std::tie(range.buffer, range.offset) =
            buffer_cache.ObtainBuffer(range.base_address, size, false);
        needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
    }

    // Bind vertex buffers
    auto& host_buffers = v.host_buffers;
    auto& host_offsets = v.host_offsets;
    auto& host_sizes = v.host_sizes;
    auto& host_strides = v.host_strides;
    u32 stream_with_memory = 0;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                ready ? ranges_merged.begin() + ready->range_index[stream_with_memory++]
                      : std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                            return buffer.base_address >= range.base_address &&
                                   buffer.base_address < range.end_address;
                        });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->buffer->Handle());
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    v.num_buffers = static_cast<u32>(guest_buffers.size());
}

void Rasterizer::BindIndexBuffer(u32 index_offset) {
    ResolveIndexBuffer(index_offset);
    EmitIndexBuffer();
}

void Rasterizer::EmitIndexBuffer() {
    scheduler.Record([handle = index_bind.handle, offset = index_bind.offset,
                      type = index_bind.type](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindIndexBuffer(handle, offset, type);
    });
}

void Rasterizer::ResolveIndexBuffer(u32 index_offset) {
    GOW3_SECTION(ResolveIndexBuffer);
    const auto& regs = Regs();

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(index_address, index_buffer_size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
    index_bind = {buffer->Handle(), offset, index_type};
}

void Rasterizer::ResetBindings(bool is_compute) {
    GOW3_SECTION(Kick);
    for (auto& image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    for (const auto [buffer, offset, size, is_written] : bound_buffers) {
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        const auto write_flag =
            is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone;
        runtime.AccessBuffer(buffer, offset, size, dst_stage,
                             vk::AccessFlagBits2::eShaderRead | write_flag);
    }
    bound_images.clear();
    bound_buffers.clear();
    needs_barrier = false;
}

// gow3: GCN keeps the stencil of a depth buffer in a separate plane at its own address; the
// texture cache's depth image covers the depth plane only, so shader writes to the stencil plane
// never reached it. MediEvil (UE4) clears the stencil by filling that plane with a compute shader
// before the base pass and before the lights; the stencil kept the base pass's 0x80 and the shadow
// projections, which test "stencil != 0" inside a shadow volume, darkened their whole box (V1).
// A dispatch writing a whole known plane (starting at it, about its size; not a larger buffer that
// merely spans it) marks it; the next bind copies it into the image.
void Rasterizer::NoteStencilPlaneWrites(const Pipeline* pipeline) {
    if (!pipeline->IsCompute() || Gow3Toggle::Disabled(Gow3Toggle::StencilPlaneWrites)) {
        return;
    }
    std::scoped_lock lock{stencil_plane_mutex};
    if (stencil_planes.empty()) {
        return;
    }
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    for (const auto& desc : info.buffers) {
        if (desc.IsSpecial() || !desc.is_written) {
            continue;
        }
        const auto sharp = desc.GetSharp(info);
        const VAddr begin = sharp.base_address;
        const VAddr end = begin + sharp.GetSize();
        const u64 written = end - begin;
        for (const auto& [plane, size] : stencil_planes) {
            if (written < size - size / 16 || written > size + size / 16) {
                continue;
            }
            if (begin == plane) {
                stencil_plane_writes[plane] = {begin, written, info.pgm_hash};
            } else {
                // Diagnostics: a plane-sized write elsewhere (another plane, or a misread address).
                static std::atomic<u32> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 12) {
                    std::printf("Stencil plane-sized write by shader %#llx at %#llx +%#llx (known plane %#llx)\n",
                                (unsigned long long)info.pgm_hash, (unsigned long long)begin,
                                (unsigned long long)written, (unsigned long long)plane);
                }
            }
        }
    }
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = CsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    runtime.CopyColorAndDepth(&src_image, &dst_image);
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = CsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    runtime.ClearImage(&image1, range, clear);
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, const PreparedStage* prepared,
                             Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data, u32& write_index) {
    GOW3_SECTION(BindBuffers);
    const u64 alignment = instance.StorageMinAlignment();
    for (u32 buffer_index = 0; buffer_index < stage.buffers.size(); ++buffer_index) {
        const auto& desc = stage.buffers[buffer_index];
        const RingBinding* ring = num_ring_stages ? FindRingBinding(stage, buffer_index) : nullptr;
        if (ring) {
            // Copied by the GPU command thread into the constant ring (read-only).
            if (FrameCapture::Active() && ring->size != 0) {
                // A flatbuf has no guest address and is copied with address 0: record the
                // flattened user data it was copied from.
                const void* data = ring->address
                                       ? reinterpret_cast<const void*>(ring->address)
                                       : stage.FlatUserData().data();
                if (data && (ring->address == 0 ||
                             memory->IsValidGpuMapping(ring->address, ring->size))) {
                    FrameCapture::Buffer(stage.pgm_hash, buffer_index, ring->address, data,
                                         ring->size);
                }
            }
            if (!desc.IsSpecial()) {
                if (ring->size == 864 && gbuffer_draw && ring->address != 0 &&
                    memory->IsValidGpuMapping(ring->address, 0)) {
                    camera_motion->OnConstants(reinterpret_cast<const float*>(ring->address));
                }
                push_data.AddOffset(binding.buffer, 0);
            }
            buffer_infos.emplace_back(constant_ring->Handle(), ring->offset, ring->size);
        } else if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                needs_barrier |= runtime.IsBufferAccessed(gds_buf, 0, gds_buf->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetStreamBuffer();
                const u32 ubo_size = stage.FlatUserData().size() * sizeof(u32);
                if (FrameCapture::Active()) {
                    FrameCapture::Buffer(stage.pgm_hash, binding.buffer, 0,
                                         stage.FlatUserData().data(), ubo_size);
                }
                const u64 offset =
                    vk_buffer.Copy(stage.FlatUserData().data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (Regs().clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetStreamBuffer();
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = Regs().clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetStreamBuffer();
                const auto& cs_program = CsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                lds_buffer.Commit();
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
            }
        } else {
            const auto vsharp =
                prepared ? prepared->buffer_sharps[buffer_index] : desc.GetSharp(stage);
            if (vsharp.GetSize() == 864 && gbuffer_draw &&
                memory->IsValidGpuMapping(vsharp.base_address, 0)) {
                camera_motion->OnConstants(reinterpret_cast<const float*>(vsharp.base_address));
            }
            // Object motion research: the vertex shader buffers of G-buffer draws.
            if (FrameCapture::Active() && gbuffer_draw && stage.sw_stage == Shader::SwStage::Vertex &&
                vsharp.base_address != 0 && memory->IsValidGpuMapping(vsharp.base_address, 0)) {
                const auto* f = reinterpret_cast<const float*>(vsharp.base_address);
                char note[192];
                std::snprintf(note, sizeof(note),
                              "\n  gb vs %08x slot %u at %#llx size %u stride %u: %g %g %g %g",
                              u32(stage.pgm_hash), binding.buffer,
                              (unsigned long long)vsharp.base_address, u32(vsharp.GetSize()),
                              u32(vsharp.GetStride()), f[0], f[1], f[2], f[3]);
                FrameCapture::Note(note);
            }
            if (FrameCapture::Active() && vsharp.base_address != 0 && vsharp.GetSize() != 0 &&
                memory->IsValidGpuMapping(vsharp.base_address, 0)) {
                FrameCapture::Buffer(stage.pgm_hash, binding.buffer, vsharp.base_address,
                                     reinterpret_cast<const void*>(vsharp.base_address),
                                     memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize()));
            }
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                if (size != vsharp.GetSize()) {
                    LOG_ERROR(Render, "Clamped size from {} to {} for stage {:#x}",
                              vsharp.GetSize(), size, stage.pgm_hash);
                }
                const auto [buffer, offset] = buffer_cache.ObtainBuffer(
                    vsharp.base_address, size, desc.is_written, desc.is_formatted);
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                bound_buffers.emplace_back(buffer, offset, size, desc.is_written);
                if (desc.is_written) {
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
                    VideoCore::BufferCache::NoteGpuWriterShader(vsharp.base_address, size,
                                                                stage.pgm_hash);
                }
                needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, desc.is_written);
            }
        }

        auto& set_write = set_writes[write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        ++binding.buffer;
    }
}

Rasterizer::ImageDescCacheEntry& Rasterizer::CachedImageDescEntry(const AmdGpu::Image& sharp,
                                                                   const Shader::ImageResource& res,
                                                                   u64 hash) {
    std::array<u64, 4> key;
    std::memcpy(key.data(), &sharp, sizeof(key));
    const u32 flags = u32(res.is_written) | u32(res.is_depth) << 1 | u32(res.is_array) << 2;
    // Two-way set associative: a frame binds a few thousand distinct T#s.
    auto* set = &image_desc_cache[(hash % (image_desc_cache.size() / 2)) * 2];
    const auto matches = [&](const ImageDescCacheEntry& e) {
        return e.flags == flags && e.sharp == key;
    };
    const bool disabled = Gow3Toggle::Disabled(Gow3Toggle::ImageDescCache);
    ImageDescCacheEntry* slot = nullptr;
    if (!disabled && matches(set[0])) {
        slot = &set[0];
    } else if (!disabled && matches(set[1])) {
        slot = &set[1];
    } else {
        // Replace the least recently used way that no binding of this call points at.
        slot = set[0].last_use <= set[1].last_use ? &set[0] : &set[1];
        if (slot->pinned == bind_epoch) {
            slot = slot == &set[0] ? &set[1] : &set[0];
        }
        if (slot->pinned == bind_epoch) {
            slot = &image_desc_overflow.emplace_back();
        }
    }
    slot->last_use = ++desc_use_counter;
    auto& entry = *slot;
    if (entry.flags != flags || entry.sharp != key ||
        Gow3Toggle::Disabled(Gow3Toggle::ImageDescCache)) {
        entry.desc = VideoCore::TextureCache::ImageDesc{sharp, res};
        entry.sharp = key;
        entry.flags = flags;
        entry.found_generation = ~0ULL;
        entry.view_memo = {};
    }
    return entry;
}

static bool TextureSetRegions();

// gow3 GOW3_NARROW_IMAGE_STAGES (default on; 0: eAllCommands as upstream; also live through the
// toggle bit Gow3Toggle::NarrowImageStages, for an A/B on the same frames): the pipeline stages that
// read a sampled or storage image bound for this shader, as the destination of its transition. The
// stage is stored as the image's last use, so the next transition (e.g. back to a render target)
// waits for these shaders only, not for all commands (N-3: nearly every barrier drained the GPU).
// Image::GetBarriers only skips a read-after-read barrier when the new reader's stage is covered.
static vk::PipelineStageFlags2 ImageReadStages(const Shader::Info& stage) {
    static const bool narrow = [] {
        const char* env = std::getenv("GOW3_NARROW_IMAGE_STAGES");
        return !(env && env[0] == '0');
    }();
    if (!narrow || Gow3Toggle::Disabled(Gow3Toggle::NarrowImageStages)) {
        return vk::PipelineStageFlagBits2::eAllCommands;
    }
    return stage.sw_stage == Shader::SwStage::Compute
               ? vk::PipelineStageFlagBits2::eComputeShader
               : vk::PipelineStageFlagBits2::ePreRasterizationShaders |
                     vk::PipelineStageFlagBits2::eFragmentShader;
}

void Rasterizer::BindTextures(const Shader::Info& stage, const PreparedStage* prepared,
                              Shader::Backend::Bindings& binding, u32& write_index,
                              bool& barrier, bool on_helper) {
    GOW3_SECTION(BindTextures);
    const u32 first_image_idx = image_infos.size();
    TextureSetSlot set_slot;
    if (!on_helper && BindTexturesFromSet(stage, prepared, first_image_idx, barrier, set_slot)) {
        // Descriptor writes: one per image (no mip arrays in a memoized set).
        for (u32 i = 0; i < stage.images.size(); ++i) {
            auto& set_write = set_writes[write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = binding.unified++;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = vk::DescriptorType::eSampledImage;
            set_write.pImageInfo = &image_infos[first_image_idx + i];
        }
        BindSamplers(stage, prepared, binding, write_index);
        return;
    }
    // gow3 T1 (impl-stage-b-per-draw-2.md): a stage without images (most vertex shaders: ~970
    // binds a heavy Deck frame, "memo off or no image" at ~0.18 us each) leaves the slow path
    // nothing to do but reset its per-call state and bind the samplers; this is that, without
    // the set-up of the image loops. Second toggle word 131072: through the slow path again.
    if (stage.images.empty() && !Gow3Toggle2::Disabled(Gow3Toggle2::NoImageFastPath)) {
        image_bindings.clear();
        image_binding_entries.clear();
        image_desc_overflow.clear();
        image_desc_storage.clear();
        ++bind_epoch;
        if (Gow3Stats::enabled && !on_helper) {
            Gow3Stats::Bump(Gow3Stats::no_image_binds);
        }
        BindSamplers(stage, prepared, binding, write_index);
        return;
    }
    GOW3_SECTION(TexSlow);
    if (!on_helper && !set_slot && !stage.images.empty()) {
        ++texture_set_unkept[7]; // written images, mip fallback, too many, memo off
    }
    // gow3 (study-stage-b-per-draw.md C3): why this bind took the slow path, and its time
    // (samplers included), for the "slow texture binds" line. Statistics only.
    struct TexslowTimer {
        u64* calls = nullptr;
        u64* ns = nullptr;
        u64 start = 0;
        ~TexslowTimer() {
            if (ns) {
                ++*calls;
                *ns += Semaphore::NowNs() - start;
            }
        }
    } texslow_timer;
    if (Gow3Stats::enabled && !on_helper) {
        u32 cause = TexslowOther;
        if (stage.sw_stage == Shader::SwStage::Compute) {
            cause = TexslowCompute;
        } else if (set_slot) {
            cause = TexslowMemoMiss;
        } else if (!stage.images.empty() && !prepared) {
            cause = TexslowUnprepared;
        } else if (std::ranges::any_of(stage.images, [](const auto& image) {
                       return image.is_written ||
                              image.mip_fallback_mode != Shader::MipStorageFallbackMode::None;
                   })) {
            cause = TexslowWritten;
        } else if (stage.images.size() > TextureSet::MaxImages) {
            cause = TexslowTooMany;
        }
        texslow_timer.calls = &texslow_calls[cause];
        texslow_timer.start = Semaphore::NowNs();
        texslow_timer.ns = &texslow_ns[cause];
    }
    const u64 bind_generation = texture_cache.RegistryGeneration();
    image_bindings.clear();
    image_binding_entries.clear();
    image_desc_overflow.clear();
    image_desc_storage.clear();
    ++bind_epoch;
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;
    // TextureSetMemo: what this call resolves, to remember when the set qualifies. Only a call
    // with a set slot reads or writes it (a slot means one binding per image, up to the slot's
    // size): its images' entries start empty, as the zeroed local array this was gave them.
    auto& resolved = texture_set_resolved;
    if (set_slot) {
        std::fill_n(resolved.begin(), std::min<size_t>(stage.images.size(), resolved.size()),
                    TextureSetEntry{});
    }
    bool set_ok = static_cast<bool>(set_slot);
    u32 unkept = ~0u; // why the set is not kept (texture_set_unkept), the first reason found
    // SampleSceneProxies: bindings that may read a reduced scene proxy instead of the native
    // image (normalized sampling only; see Shader::ImageResource::needs_native).
    boost::container::small_vector<bool, 16> binding_proxy_ok;
    const bool sample_proxies = !Gow3Toggle::Disabled(Gow3Toggle::SampleSceneProxies) &&
                                !on_helper && scene_targets->Reduced();

    for (u32 image_index = 0; image_index < stage.images.size(); ++image_index) {
        const auto& image_desc = stage.images[image_index];
        const auto tsharp =
            prepared ? prepared->image_sharps[image_index] : image_desc.GetSharp(stage);
        // gow3: a hash lookup per texture per draw for a diagnostic only.
        static const bool warn_meta = std::getenv("GOW3_WARN_META_TEXTURE") != nullptr;
        if (warn_meta && texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings.emplace_back(VideoCore::ImageId{}, &image_desc_storage.emplace_back());
            image_binding_entries.push_back(nullptr);
            binding_proxy_ok.push_back(false);
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        if (!memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !IsKnownFormat(data_fmt, num_fmt)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt));
            image_bindings.emplace_back(VideoCore::ImageId{}, &image_desc_storage.emplace_back());
            image_binding_entries.push_back(nullptr);
            binding_proxy_ok.push_back(false);
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        // Same as image_desc.NumBindings(stage), without fetching the T# again.
        const u32 num_bindings =
            mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex
                ? static_cast<u32>(tsharp.last_level - tsharp.base_level + 1)
                : 1u;
        const bool proxy_candidate = sample_proxies && num_bindings == 1 &&
                                     !image_desc.needs_native && !image_desc.is_written &&
                                     mip_fallback_mode == Shader::MipStorageFallbackMode::None;

        auto& desc_entry =
            prepared ? CachedImageDescEntry(tsharp, image_desc, prepared->image_hashes[image_index])
                     : CachedImageDescEntry(tsharp, image_desc);
        // GOW3_SCENE_DEBUG: proxied scene targets this binding reads at the native size, and why.
        const auto debug_native = [&](const VideoCore::Image& image) {
            if (!scene_debug_frame || proxy_candidate || !image.scene_proxy) return;
            std::printf("Scene native read: %s %016llx %s %ux%u needs_native %d written %d "
                        "bindings %u mip_fallback %d\n",
                        stage.sw_stage == Shader::SwStage::Compute ? "cs" : "gfx",
                        (unsigned long long)stage.pgm_hash,
                        vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                        image.info.size.height, int(image_desc.needs_native),
                        int(image_desc.is_written), num_bindings, int(mip_fallback_mode));
        };
        for (auto i = 0; i < num_bindings; i++) {
            // gow3: a plain binding (no mip override) of the same T# resolves to the same image
            // while no image was registered or unregistered.
            if (mip_fallback_mode == Shader::MipStorageFallbackMode::None &&
                desc_entry.found_generation == texture_cache.RegistryGeneration() &&
                !Gow3Toggle::Disabled(Gow3Toggle::TextureBindingMemo)) {
                desc_entry.pinned = bind_epoch;
                auto& [image_id, _] =
                    image_bindings.emplace_back(desc_entry.found_id, &desc_entry.found_desc);
                image_binding_entries.push_back(&desc_entry);
                binding_proxy_ok.push_back(proxy_candidate);
                texture_cache.MarkFound(image_id);
                auto* image = &texture_cache.GetImage(image_id);
                if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                    image_id = depth_image_id;
                    image = &texture_cache.GetImage(image_id);
                }
                debug_native(*image);
                if (image->binding.is_bound) {
                    image->binding.force_general |= image_desc.is_written;
                }
                image->binding.is_bound = 1u;
                continue;
            }
            auto& desc = image_desc_storage.emplace_back(desc_entry.desc);
            auto& [image_id, _] = image_bindings.emplace_back(VideoCore::ImageId{}, &desc);
            image_binding_entries.push_back(nullptr);
            binding_proxy_ok.push_back(proxy_candidate);

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            const u64 generation = texture_cache.RegistryGeneration();
            image_id = texture_cache.FindImage(desc);
            if (mip_fallback_mode == Shader::MipStorageFallbackMode::None &&
                generation == texture_cache.RegistryGeneration() &&
                desc_entry.pinned != bind_epoch) {
                desc_entry.found_generation = generation;
                desc_entry.found_id = image_id;
                desc_entry.found_desc = desc;
                desc_entry.view_memo = {};
            }
            auto* image = &texture_cache.GetImage(image_id);
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            debug_native(*image);
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    // Second pass to re-bind images that were updated after binding
    for (u32 binding_index = 0; binding_index < image_bindings.size(); ++binding_index) {
        auto& [image_id, desc_ptr] = image_bindings[binding_index];
        auto* memo_entry = image_binding_entries[binding_index];
        const auto& desc = *desc_ptr;
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                auto& rebind_desc = image_desc_storage.emplace_back(desc);
                desc_ptr = &rebind_desc;
                image_id = texture_cache.FindImage(rebind_desc);
                memo_entry = nullptr;
                if (set_ok) unkept = 4;
                set_ok = false;
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(
                image_id, desc, memo_entry ? &memo_entry->view_memo : nullptr, !on_helper);
            const auto binding = image.binding;

            // gow3: a proxied scene image sampled with normalized coordinates reads the proxy:
            // no resample to the native size (the native image is not touched).
            if (binding_proxy_ok[binding_index] && !is_storage && !binding.force_general &&
                !binding.is_target &&
                !(upscaler->Enabled() && upscaler->RedirectsSampled(image_id))) {
                if (const auto proxy = scene_targets->SampleProxy(image, desc.view_info)) {
                    image.usage.texture = 1u;
                    image_infos.emplace_back(VK_NULL_HANDLE, proxy->view, proxy->layout);
                    if (set_ok && binding_index < resolved.size()) {
                        resolved[binding_index] = {image_id, proxy->view, image.backing,
                                                   desc.view_info.range, true,
                                                   image.info.guest_address, desc.info.guest_address,
                                                   image.info.guest_size, desc.info.guest_size};
                    }
                    ++proxy_samples;
                    continue;
                }
            }
            // gow3 GOW3_CAPTURE_ON_TARGET (diagnostics): the inputs of the watched pass chain
            // (UE4's sky light integration at a level's load): their texel statistics, read
            // before the layout transition below.
            if (!is_storage && FrameCapture::TargetRecording()) {
                texture_cache.LogTexelStats(image);
            }
            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
                if (instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target) {
                    barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                        vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
                } else {
                    barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, ImageReadStages(stage),
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
                }
            } else {
                if (is_storage) {
                    barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, ImageReadStages(stage),
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else if (!(image.info.props.is_depth &&
                             image_view.view_aspect == vk::ImageAspectFlagBits::eDepth &&
                             runtime.SampleDepthInPlace(&image))) {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    barrier |= runtime.Transit(
                        &image, new_layout, ImageReadStages(stage),
                        vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;
            if (FrameCapture::Active()) {
                FrameCapture::Sampled(image.info, is_storage);
            }

            vk::ImageView view = *image_view.image_view;
            vk::ImageLayout layout = image.backing->state.layout;
            if (upscaler->Enabled()) {
                // gow3: the display pass reads the upscaled frame (scaled presets).
                upscaler->RedirectSampled(image_id, image_view.info, view, layout);
            }
            image_infos.emplace_back(VK_NULL_HANDLE, view, layout);
            if (set_ok && binding_index < resolved.size()) {
                resolved[binding_index] = {image_id, *image_view.image_view, image.backing,
                                           desc.view_info.range, false, image.info.guest_address,
                                           desc.info.guest_address, image.info.guest_size,
                                           desc.info.guest_size};
                resolved[binding_index].depth_aspect =
                    image_view.view_aspect == vk::ImageAspectFlagBits::eDepth;
                set_ok = !is_storage && !binding.force_general && !binding.is_target;
                if (!set_ok) {
                    unkept = is_storage ? 0 : binding.force_general ? 1 : image.info.props.is_depth ? 3 : 2;
                }
            }
        }
    }
    if (set_ok && image_bindings.size() == stage.images.size() &&
        (!TextureSetRegions() || texture_cache.RegistryGeneration() == bind_generation)) {
        // Remember the set: the same T#s resolve the same way while no image is (un)registered
        // (over their own ranges with region stamps; none was during this bind).
        KeepTextureSet(set_slot, {resolved.data(), stage.images.size()});
    } else if (set_slot) {
        ++texture_set_unkept[unkept != ~0u ? unkept : image_bindings.size() != stage.images.size() ? 5 : 6];
        DropTextureSet(set_slot);
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& desc = *image_bindings[image_binding_idx].second;
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    BindSamplers(stage, prepared, binding, write_index);
}

void Rasterizer::BindSamplers(const Shader::Info& stage, const PreparedStage* prepared,
                              Shader::Backend::Bindings& binding, u32& write_index) {
    GOW3_SECTION(Samplers);
    // gow3 (study 2026-10-29 item 3): GetSampler hashes the S# with XXH3 and takes the sampler
    // mutex for every sampler of every draw (0.12 ms a heavy frame, N-28). A handle it returned
    // in the current GC period is reused within that period (per thread: the binding helper
    // binds samplers too); the first use in each period still goes through it (LRU touch).
    struct SamplerMemo {
        AmdGpu::Sampler sharp{};
        u64 border_color = 0;
        float lod_bias = 0.0f;
        bool is_depth = false;
        u64 tick = ~0ull;
        vk::Sampler handle{};
    };
    static thread_local std::array<SamplerMemo, 256> sampler_memo{};
    const bool memo_on = Gow3Settings::PerfOn(Gow3Settings::PerfLeanBinding) &&
                         !Gow3Toggle2::Disabled(Gow3Toggle2::SamplerMemo);
    const u64 gc_tick = texture_cache.GcTick();
    const u64 border_color = Regs().ta_bc_base.base_addr;
    for (u32 sampler_index = 0; sampler_index < stage.samplers.size(); ++sampler_index) {
        const auto& sampler = stage.samplers[sampler_index];
        auto ssharp =
            prepared ? prepared->sampler_sharps[sampler_index] : sampler.GetSharp(stage);
        const float lod_bias = sampler.is_depth ? 0.0f : sampler_lod_bias;
        u64 slot_hash = (ssharp.raw0 ^ ssharp.raw1 * 0x9E3779B97F4A7C15ull ^ border_color) * 0xFF51AFD7ED558CCDull;
        slot_hash ^= slot_hash >> 29;
        auto& memo = sampler_memo[(slot_hash ^ sampler.is_depth) % sampler_memo.size()];
        vk::Sampler vk_sampler;
        if (memo_on && memo.tick == gc_tick && memo.sharp == ssharp && memo.border_color == border_color &&
            memo.is_depth == sampler.is_depth && memo.lod_bias == lod_bias) {
            vk_sampler = memo.handle;
        } else {
            vk_sampler = texture_cache.GetSampler(ssharp, Regs().ta_bc_base, sampler.is_depth, lod_bias);
            memo = {ssharp, border_color, lod_bias, sampler.is_depth, gc_tick, vk_sampler};
        }
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

Rasterizer::BeginSignature Rasterizer::MakeBeginSignature(const GraphicsPipeline* pipeline) const {
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = Regs();
    BeginSignature sig;
    const u32 num_color = std::bit_width(key.mrt_mask);
    for (u32 cb = 0; cb < num_color && cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        sig.ids[cb] = cb_descs[cb].first;
        if (cb_descs[cb].first) {
            sig.views[cb] = cb_descs[cb].second.view_info;
        }
        sig.color_samples[cb] = key.color_samples[cb];
    }
    sig.ids[AmdGpu::NUM_COLOR_BUFFERS] = db_desc.first;
    if (db_desc.first) {
        sig.views[AmdGpu::NUM_COLOR_BUFFERS] = db_desc.second.view_info;
    }
    sig.mrt_mask = key.mrt_mask;
    sig.num_samples = key.num_samples;
    sig.motion = key.motion_vectors;
    sig.scene_started = scene_started;
    sig.raster_scaling = upscaler->RasterScaling();
    sig.upscaler_state = upscaler->RedirectState();
    sig.generation = texture_cache.RegistryGeneration();
    std::memcpy(&sig.depth_control, &regs.depth_control, sizeof(u32));
    sig.depth_valid = regs.depth_buffer.DepthValid();
    sig.stencil_valid = regs.depth_buffer.StencilValid();
    return sig;
}

/// gow3 T-S1a: memoized texture sets are checked against the region stamps of their own images
/// instead of the registry generation, which any image (un)registration anywhere changed.
/// GOW3_TEXSET_REGIONS=0: the generation as before.
static bool TextureSetRegions() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_TEXSET_REGIONS");
        return !env || env[0] != '0';
    }();
    return on;
}

template <class Set>
u64 Rasterizer::TextureSetStamp(const Set& set) const {
    u64 stamp = 0;
    for (u32 i = 0; i < set.count; ++i) {
        const auto& entry = set.entries[i];
        if (!entry.id) continue;
        stamp += texture_cache.RegionStamp(entry.address, entry.size) +
                 texture_cache.RegionStamp(entry.sharp_address, entry.sharp_size);
    }
    return stamp;
}

u64 Rasterizer::TextureSetStamp(u32 compact_slot) const {
    // The same sum, from the compact layout's rows.
    const auto& hot = texture_set_hot[compact_slot];
    const auto& cold = texture_set_cold[compact_slot];
    u64 stamp = 0;
    for (u32 i = 0; i < texture_set_heads[compact_slot].count; ++i) {
        if (!hot[i].id) continue;
        stamp += texture_cache.RegionStamp(cold[i].address, cold[i].size) +
                 texture_cache.RegionStamp(cold[i].sharp_address, cold[i].sharp_size);
    }
    return stamp;
}

/// Two ways (T-S1b: half the misses were another set in the slot): a set lives in either of two
/// slots; the less recently used one is replaced. GOW3_TEXSET_TWO_WAY=0: one slot.
static bool TextureSetTwoWay() {
    static const bool two_way = [] {
        const char* env = std::getenv("GOW3_TEXSET_TWO_WAY");
        return !env || env[0] != '0';
    }();
    return two_way;
}

/// gow3 (impl-texset-index.md): the memo's two layouts as BindTexturesFromSetIn reads them. A
/// slot has a head (key, last use, generation, stamp, stage, scene generation, count), the T#
/// hashes and the images (the fields a hit reads have the same names in both).
struct Rasterizer::LegacyTextureSets {
    static constexpr bool compact = false;
    static constexpr u32 slots = Gow3TexSet::Slots;
    static constexpr int stat = 1; ///< Gow3TexSet::lookups index (the memo line's counters)
    TextureSet* heads;
    TextureSetSlot Slot(u32 index) const {
        return {&heads[index], ~0u};
    }
    u64& Hash(u32 index, u32 i) const {
        return heads[index].hashes[i];
    }
    const TextureSetEntry& Entry(u32 index, u32 i) const {
        return heads[index].entries[i];
    }
    u64 Stamp(const Rasterizer& rasterizer, u32 index) const {
        return rasterizer.TextureSetStamp(heads[index]);
    }
};
struct Rasterizer::CompactTextureSets {
    static constexpr bool compact = true;
    static constexpr u32 slots = Gow3TexSet::Slots;
    static constexpr int stat = 0;
    TextureSetHead* heads;
    TextureSetHotRow* hot;
    TextureSetSlot Slot(u32 index) const {
        return {nullptr, index};
    }
    u64& Hash(u32 index, u32 i) const {
        return hot[index][i].hash;
    }
    const TextureSetHot& Entry(u32 index, u32 i) const {
        return hot[index][i];
    }
    u64 Stamp(const Rasterizer& rasterizer, u32 index) const {
        return rasterizer.TextureSetStamp(index);
    }
};
/// T2: sets of 17-32 images, laid out as the table as before; their own lookup counters.
struct Rasterizer::WideTextureSets {
    static constexpr bool compact = false;
    static constexpr u32 slots = WideTextureSet::Slots;
    static constexpr int stat = -1;
    WideTextureSet* heads;
    TextureSetSlot Slot(u32 index) const {
        return {nullptr, ~0u, &heads[index]};
    }
    u64& Hash(u32 index, u32 i) const {
        return heads[index].hashes[i];
    }
    const TextureSetEntry& Entry(u32 index, u32 i) const {
        return heads[index].entries[i];
    }
    u64 Stamp(const Rasterizer& rasterizer, u32 index) const {
        return rasterizer.TextureSetStamp(heads[index]);
    }
};

/// gow3 T2 (impl-stage-b-per-draw-2.md): the memo also takes sets of 17-32 images and stages
/// without a prepared draw. Opt-in at start (GOW3_TEXSET_WIDE=1); second toggle word 262144
/// switches it off live.
static bool TextureSetWideWanted() {
    static const bool wanted = [] {
        const char* env = std::getenv("GOW3_TEXSET_WIDE");
        return env && env[0] == '1';
    }();
    return wanted;
}
static bool TextureSetWide() {
    return TextureSetWideWanted() && !Gow3Toggle2::Disabled(Gow3Toggle2::TextureSetWide);
}

void Rasterizer::MakeCompactTextureSets() {
    static_assert(sizeof(TextureSetHead) == 64 && sizeof(TextureSetHot) == 48 &&
                  sizeof(TextureSetCold) == 32);
    static_assert(std::is_trivially_destructible_v<TextureSetHead> &&
                  std::is_trivially_destructible_v<TextureSetHotRow> &&
                  std::is_trivially_destructible_v<TextureSetColdRow>);
    constexpr size_t heads_bytes = sizeof(TextureSetHead) * Gow3TexSet::Slots;
    constexpr size_t hot_bytes = sizeof(TextureSetHotRow) * Gow3TexSet::Slots;
    constexpr size_t cold_bytes = sizeof(TextureSetColdRow) * Gow3TexSet::Slots;
    texture_set_memory =
        std::make_unique<Gow3TexSet::TableMemory>(heads_bytes + hot_bytes + cold_bytes);
    auto* base = static_cast<u8*>(texture_set_memory->Data());
    ASSERT_MSG(base, "Texture-set memo: no memory for the compact layout");
    texture_set_heads = reinterpret_cast<TextureSetHead*>(base);
    texture_set_hot = reinterpret_cast<TextureSetHotRow*>(base + heads_bytes);
    texture_set_cold = reinterpret_cast<TextureSetColdRow*>(base + heads_bytes + hot_bytes);
    std::uninitialized_value_construct_n(texture_set_heads, Gow3TexSet::Slots);
    std::uninitialized_value_construct_n(texture_set_hot, Gow3TexSet::Slots);
    std::uninitialized_value_construct_n(texture_set_cold, Gow3TexSet::Slots);
    const int huge = texture_set_memory->Huge();
    Gow3TexSet::compact_bytes.store(heads_bytes + hot_bytes + cold_bytes, std::memory_order_relaxed);
    Gow3TexSet::compact_huge.store(huge, std::memory_order_relaxed);
    std::printf("Texture-set memo: compact layout, %.1f MB (index %.1f, images %.1f, region data "
                "%.1f MB; the table as before: %.1f MB); transparent huge pages: %s (system: %s)\n",
                (heads_bytes + hot_bytes + cold_bytes) / 1e6, heads_bytes / 1e6, hot_bytes / 1e6,
                cold_bytes / 1e6, sizeof(std::array<TextureSet, Gow3TexSet::Slots>) / 1e6,
                huge == 1 ? "yes" : huge == 0 ? "no" : huge == -1 ? "unknown" : "n/a",
                Gow3TexSet::HugePageMode());
}

template <class Table>
bool Rasterizer::BindTexturesFromSetIn(const Table& table, const Shader::Info& stage,
                                       const u64* hashes, u32 count, u64 key, bool& barrier,
                                       TextureSetSlot& slot) {
    const u64 generation = texture_cache.RegistryGeneration();
    const bool regions = TextureSetRegions();
    // The memo line's counters are the compact and legacy tables' (T2's wide table apart).
    const bool counted = Gow3Stats::enabled && Table::stat >= 0;
    if (counted) Gow3TexSet::Bump(Gow3TexSet::lookups[Table::stat < 0 ? 0 : Table::stat]);
    bool found = false;
    const u32 index = Gow3TexSet::Choose(
        table.heads, Gow3TexSet::WaysOf(key, TextureSetTwoWay(), Table::slots), key, found);
    auto& set = table.heads[index];
    slot = table.Slot(index);
    set.last_use = ++texture_set_tick;
    // Region stamps: unchanged registry generation since the set was last confirmed needs no
    // stamp; otherwise the stamps decide and a still-valid set takes the new generation.
    bool current = found && set.generation != ~0ull;
    if (current && regions && set.generation != generation) {
        current = set.stamp == table.Stamp(*this, index);
        if (current) set.generation = generation;
    } else if (current && !regions) {
        current = set.generation == generation;
    }
    bool match = found && set.stage == &stage && set.count == count && current &&
                 set.scene_generation == scene_targets->Generation();
    for (u32 i = 0; match && i < count; ++i) {
        match = table.Hash(index, i) == hashes[i];
    }
    if (!match) {
        const u32 why = !found ? 0 : !current ? 1 : 3;
        ++texture_set_why[why];
        ++texture_set_misses;
        if (Gow3Stats::enabled) {
            Gow3Stats::texset_misses.fetch_add(1, std::memory_order_relaxed);
            if (counted) Gow3TexSet::Bump(Gow3TexSet::misses[why]);
        }
        set.key = key;
        set.stage = &stage;
        set.count = 0;
        set.generation = ~0ull;
        for (u32 i = 0; i < count; ++i) {
            table.Hash(index, i) = hashes[i];
        }
        return false;
    }
    // Every image still has the backing its view belongs to and needs no refresh.
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = table.Entry(index, i);
        if (!entry.id) {
            continue;
        }
        const auto& image = texture_cache.GetImage(entry.id);
        // Proxy entries need a current proxy; native entries of a proxied image go through
        // BindTextures, which may sample the proxy instead.
        const bool proxies_on = !Gow3Toggle::Disabled(Gow3Toggle::SampleSceneProxies);
        if (image.backing != entry.backing || image.binding.needs_rebind ||
            image.binding.is_target || !texture_cache.IsUpToDate(entry.id) ||
            (entry.proxy ? !proxies_on ||
                               !scene_targets->ProxyCurrent(image, entry.range.base.level,
                                                            entry.range.extent.levels)
                         : image.scene_proxy && proxies_on) ||
            (upscaler->Enabled() && upscaler->RedirectsSampled(entry.id))) {
            ++texture_set_why[2];
            ++texture_set_misses;
            if (Gow3Stats::enabled) {
                Gow3Stats::texset_misses.fetch_add(1, std::memory_order_relaxed);
                if (counted) Gow3TexSet::Bump(Gow3TexSet::misses[2]);
            }
            set.generation = ~0ull;
            return false;
        }
    }
    ++texture_set_hits;
    if (counted) Gow3TexSet::Bump(Gow3TexSet::hits);
    slot = {};
    GOW3_SECTION(TexSetEmit);
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = table.Entry(index, i);
        if (!entry.id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            continue;
        }
        texture_cache.MarkFound(entry.id);
        auto& image = texture_cache.GetImage(entry.id);
        image.binding.is_bound = 1u;
        bound_images.emplace_back(entry.id);
        image.usage.texture = 1u;
        if (entry.proxy) {
            image_infos.emplace_back(VK_NULL_HANDLE, entry.view,
                                     scene_targets->PrepareSample(image, entry.range.base.level));
            ++proxy_samples;
            continue;
        }
        if (!(image.info.props.is_depth && entry.depth_aspect && runtime.SampleDepthInPlace(&image))) {
            const auto new_layout = image.info.props.is_depth
                                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                        : vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier |= runtime.Transit(&image, new_layout, ImageReadStages(stage),
                                       vk::AccessFlagBits2::eShaderRead, entry.range);
        }
        image_infos.emplace_back(VK_NULL_HANDLE, entry.view, image.backing->state.layout);
    }
    return true;
}

bool Rasterizer::BindTexturesFromSet(const Shader::Info& stage, const PreparedStage* prepared,
                                     u32 first_image_idx, bool& barrier, TextureSetSlot& slot) {
    GOW3_SECTION(TexSetHit);
    const u32 count = static_cast<u32>(stage.images.size());
    // T2: sets of 17-32 images and stages of draws without a prepared draw too (graphics only:
    // dispatches never have one and keep the slow path, ~34 a heavy frame).
    const bool wide = TextureSetWide();
    if (count == 0 || (!prepared && (!wide || stage.sw_stage == Shader::SwStage::Compute)) ||
        count > (wide ? WideTextureSet::MaxImages : TextureSet::MaxImages) ||
        Gow3Toggle::Disabled(Gow3Toggle::TextureSetMemo) || FrameCapture::Active()) {
        return false;
    }
    for (const auto& image : stage.images) {
        if (image.is_written || image.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
            return false;
        }
    }
    // The T# hashes a draw-preparation worker computed, or (T2, no prepared draw) the same
    // hashes of the sharps the slow path would read.
    std::array<u64, WideTextureSet::MaxImages> bind_hashes;
    const u64* hashes = prepared ? prepared->image_hashes : bind_hashes.data();
    if (!prepared) {
        for (u32 i = 0; i < count; ++i) {
            const auto& image = stage.images[i];
            bind_hashes[i] = ImageDescHash(image.GetSharp(stage), image);
        }
    }
    u64 key = XXH3_64bits_withSeed(hashes, count * sizeof(u64), reinterpret_cast<u64>(&stage));
    key |= 1; // 0 marks an empty slot
    (void)first_image_idx;
    bool hit;
    if (count > TextureSet::MaxImages) {
        if (!wide_texture_sets) {
            wide_texture_sets =
                std::make_unique<std::array<WideTextureSet, WideTextureSet::Slots>>();
            Gow3TexSet::wide_bytes.store(sizeof(*wide_texture_sets), std::memory_order_relaxed);
        }
        hit = BindTexturesFromSetIn(WideTextureSets{wide_texture_sets->data()}, stage, hashes,
                                    count, key, barrier, slot);
        if (Gow3Stats::enabled) {
            Gow3TexSet::Bump(Gow3TexSet::wide_lookups);
            if (hit) Gow3TexSet::Bump(Gow3TexSet::wide_hits);
        }
    } else if (Gow3Toggle2::Disabled(Gow3Toggle2::TextureSetLegacy)) {
        // gow3 (impl-texset-index.md): the compact layout, or with second toggle word 2048 the
        // table as before (made on its first use); each keeps its sets while the other is in use.
        if (!texture_sets) {
            texture_sets = std::make_unique<std::array<TextureSet, Gow3TexSet::Slots>>();
            Gow3TexSet::legacy_bytes.store(sizeof(*texture_sets), std::memory_order_relaxed);
        }
        hit = BindTexturesFromSetIn(LegacyTextureSets{texture_sets->data()}, stage, hashes, count,
                                    key, barrier, slot);
    } else {
        hit = BindTexturesFromSetIn(CompactTextureSets{texture_set_heads, texture_set_hot}, stage,
                                    hashes, count, key, barrier, slot);
    }
    if (!prepared && Gow3Stats::enabled) {
        Gow3TexSet::Bump(Gow3TexSet::unprepared_lookups);
        if (hit) Gow3TexSet::Bump(Gow3TexSet::unprepared_hits);
    }
    return hit;
}

void Rasterizer::KeepTextureSet(const TextureSetSlot& slot,
                                std::span<const TextureSetEntry> resolved) {
    const u32 count = static_cast<u32>(resolved.size());
    if (slot.wide) {
        WideTextureSet* set_slot = slot.wide;
        set_slot->generation = texture_cache.RegistryGeneration();
        set_slot->scene_generation = scene_targets->Generation();
        set_slot->count = count;
        std::copy_n(resolved.begin(), count, set_slot->entries.begin());
        if (TextureSetRegions()) set_slot->stamp = TextureSetStamp(*set_slot);
        return;
    }
    if (Gow3Stats::enabled) Gow3TexSet::Bump(Gow3TexSet::kept);
    if (slot.legacy) {
        TextureSet* set_slot = slot.legacy;
        set_slot->generation = texture_cache.RegistryGeneration();
        set_slot->scene_generation = scene_targets->Generation();
        set_slot->count = count;
        std::copy_n(resolved.begin(), set_slot->count, set_slot->entries.begin());
        if (TextureSetRegions()) set_slot->stamp = TextureSetStamp(*set_slot);
        return;
    }
    auto& head = texture_set_heads[slot.compact];
    auto& hot = texture_set_hot[slot.compact];
    auto& cold = texture_set_cold[slot.compact];
    head.generation = texture_cache.RegistryGeneration();
    head.scene_generation = scene_targets->Generation();
    head.count = count;
    for (u32 i = 0; i < count; ++i) {
        // The hashes stay as the lookup wrote them with the key.
        const auto& entry = resolved[i];
        hot[i].view = entry.view;
        hot[i].backing = entry.backing;
        hot[i].range = entry.range;
        hot[i].id = entry.id;
        hot[i].proxy = entry.proxy;
        hot[i].depth_aspect = entry.depth_aspect;
        cold[i] = {entry.address, entry.sharp_address, entry.size, entry.sharp_size};
    }
    if (TextureSetRegions()) head.stamp = TextureSetStamp(slot.compact);
}

void Rasterizer::DropTextureSet(const TextureSetSlot& slot) {
    if (slot.wide) {
        slot.wide->key = 0;
        return;
    }
    if (Gow3Stats::enabled) Gow3TexSet::Bump(Gow3TexSet::dropped);
    if (slot.legacy) {
        slot.legacy->key = 0;
    } else {
        texture_set_heads[slot.compact].key = 0;
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    GOW3_SECTION(BeginRendering);
    // gow3: RenderStateMemo (see BeginMemo).
    const bool memo_on = !Gow3Toggle::Disabled(Gow3Toggle::RenderStateMemo) && !FrameCapture::Active();
    if (Gow3Stats::enabled && ((begin_memo_hits + begin_memo_misses) & 0x3FFFF) == 0x3FFFF) {
        std::printf("Render state memo: %llu hits, %llu misses; texture sets: %llu hits, %llu "
                    "misses\n",
                    static_cast<unsigned long long>(begin_memo_hits),
                    static_cast<unsigned long long>(begin_memo_misses),
                    static_cast<unsigned long long>(texture_set_hits),
                    static_cast<unsigned long long>(texture_set_misses));
        std::printf("  texture set misses: other key %llu, generation %llu, image check %llu, "
                    "other %llu\n",
                    (unsigned long long)texture_set_why[0], (unsigned long long)texture_set_why[1],
                    (unsigned long long)texture_set_why[2], (unsigned long long)texture_set_why[3]);
        const auto& u = texture_set_unkept;
        std::printf("  texture sets not kept: storage %llu, written %llu, own colour target %llu, "
                    "own depth target %llu, rebind %llu, incomplete %llu, registry changed %llu; "
                    "not memoizable %llu\n",
                    (unsigned long long)u[0], (unsigned long long)u[1], (unsigned long long)u[2],
                    (unsigned long long)u[3], (unsigned long long)u[4], (unsigned long long)u[5],
                    (unsigned long long)u[6], (unsigned long long)u[7]);
        texture_set_unkept = {};
        // gow3 (study-stage-b-per-draw.md C3): the slow binds by cause, per 1000 draws (this
        // window is 262144 draws), and their time each (us; samplers included).
        std::printf("  slow texture binds per 1000 draws (us each): dispatch %.1f (%.2f), "
                    "unprepared draw %.1f (%.2f), written or mip image %.1f (%.2f), over %u "
                    "images %.1f (%.2f), memo off or no image %.1f (%.2f), memo miss %.1f (%.2f)\n",
                    texslow_calls[TexslowCompute] / 262.144,
                    texslow_calls[TexslowCompute] ? texslow_ns[TexslowCompute] / 1e3 / texslow_calls[TexslowCompute] : 0.0,
                    texslow_calls[TexslowUnprepared] / 262.144,
                    texslow_calls[TexslowUnprepared] ? texslow_ns[TexslowUnprepared] / 1e3 / texslow_calls[TexslowUnprepared] : 0.0,
                    texslow_calls[TexslowWritten] / 262.144,
                    texslow_calls[TexslowWritten] ? texslow_ns[TexslowWritten] / 1e3 / texslow_calls[TexslowWritten] : 0.0,
                    TextureSet::MaxImages, texslow_calls[TexslowTooMany] / 262.144,
                    texslow_calls[TexslowTooMany] ? texslow_ns[TexslowTooMany] / 1e3 / texslow_calls[TexslowTooMany] : 0.0,
                    texslow_calls[TexslowOther] / 262.144,
                    texslow_calls[TexslowOther] ? texslow_ns[TexslowOther] / 1e3 / texslow_calls[TexslowOther] : 0.0,
                    texslow_calls[TexslowMemoMiss] / 262.144,
                    texslow_calls[TexslowMemoMiss] ? texslow_ns[TexslowMemoMiss] / 1e3 / texslow_calls[TexslowMemoMiss] : 0.0);
        texslow_calls = {};
        texslow_ns = {};
        std::printf("  scene proxies sampled directly: %llu bindings\n",
                    static_cast<unsigned long long>(proxy_samples));
        proxy_samples = 0;
        texture_set_why = {};
        begin_memo_hits = begin_memo_misses = texture_set_hits = texture_set_misses = 0;
    }
    BeginSignature signature;
    if (memo_on) {
        signature = MakeBeginSignature(pipeline);
        const auto& regs = Regs();
        bool usable = begin_memo.valid && scheduler.IsRenderingWith(begin_memo.state) &&
                      !regs.depth_render_control.depth_clear_enable &&
                      !regs.depth_render_control.stencil_clear_enable &&
                      signature == begin_memo.signature;
        for (const auto id : signature.ids) {
            if (usable && id) {
                const auto& image = texture_cache.GetImage(id);
                usable = !image.binding.is_bound && !image.binding.needs_rebind;
            }
        }
        if (usable) {
            ++begin_memo_hits;
            attachment_feedback_loop = false;
            push_data.scene_size = begin_memo.scene_size;
            target_scale = begin_memo.target_scale;
            return begin_memo.state;
        }
        ++begin_memo_misses;
        if (Gow3Stats::enabled) Gow3Stats::render_memo_misses.fetch_add(1, std::memory_order_relaxed);
    }
    begin_memo.valid = false;
    RenderState state = BeginRenderingFull(pipeline);
    if (memo_on && !attachment_feedback_loop) {
        // (is_clear of the depth attachment shares its bytes with has_depth.)
        bool clears = state.depth_stencil_attachment.depth_clear ||
                      state.depth_stencil_attachment.stencil_clear;
        for (u32 i = 0; i < state.num_color_attachments; ++i) {
            clears |= state.color_attachments[i].is_clear != 0;
        }
        if (!clears) {
            // Recomputed after the full path: it may have started the scene (scene_started).
            begin_memo = {true, MakeBeginSignature(pipeline), state, push_data.scene_size,
                          target_scale};
        }
    }
    return state;
}

RenderState Rasterizer::BeginRenderingFull(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    using VulkanUpscalerTarget = TemporalUpscaler::Target;
    VulkanUpscalerTarget redirect{};
    bool color_redirected = false, depth_redirected = false;
    std::pair<u32, u32> guest_extent{1, 1};
    std::pair<vk::ImageView, vk::ImageLayout> original_color{};
    const auto& regs = Regs();
    const auto& key = pipeline->GetGraphicsKey();
    if (std::popcount(key.mrt_mask & 0x7f) >= 5 && db_desc.first &&
        scene_targets->EligibleScene(texture_cache.GetImage(db_desc.first))) scene_started = true;
    // A proxy attachment cannot represent MSAA or a feedback loop that reads the
    // same image through the guest's native descriptor during this draw.
    bool reduced = scene_started && upscaler->RasterScaling() && key.num_samples == 1;
    // GOW3_SCENE_DEBUG=<file>: once the file exists, why the passes of one frame keep the
    // native size (see NoteFrameStart).
    const bool debug_pass = scene_debug_frame && scene_started;
    std::string why;
    if (debug_pass) {
        why = fmt::format("pass vs {:08x} ps {:08x} mrt {:#x} samples {} raster_scaling {}:",
                          pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash,
                          key.mrt_mask ? pipeline->GetStage(Shader::SwStage::Fragment).pgm_hash
                                       : 0,
                          key.mrt_mask, u32(key.num_samples), upscaler->RasterScaling());
    }
    // Only this pass's attachments: slots past the mask's width keep earlier passes' targets,
    // often the G-buffer images the lighting passes sample (bound), which kept those native.
    const u32 num_attachments = Gow3Toggle::Disabled(Gow3Toggle::SceneAttachmentsOnly)
                                    ? u32(cb_descs.size())
                                    : u32(std::bit_width(key.mrt_mask));
    // All attachments of a reduced pass share one proxy size (scene or half resolution).
    std::optional<SceneResolution::Size> pass_size;
    const auto same_size = [&](const VideoCore::Image& image, u32 level) {
        const auto proxy = scene_targets->ProxySize(image, level);
        if (!pass_size) pass_size = proxy;
        return *pass_size == proxy;
    };
    for (u32 cb = 0; cb < num_attachments; ++cb) {
        const auto& [id, desc] = cb_descs[cb];
        if (id) {
            const auto& image = texture_cache.GetImage(id);
            // Mip levels of eligible chains have their own proxies (SceneTargets::Get).
            if (!scene_targets->Eligible(image) || image.binding.is_bound ||
                image.binding.needs_rebind || desc.view_info.range.base.layer ||
                !same_size(image, desc.view_info.range.base.level)) reduced = false;
            if (debug_pass) {
                why += fmt::format(" [{} {}x{} eligible {} bound {} rebind {} level {} layer {}]",
                                   vk::to_string(image.info.pixel_format), image.info.size.width,
                                   image.info.size.height, scene_targets->Eligible(image),
                                   u32(image.binding.is_bound), u32(image.binding.needs_rebind),
                                   desc.view_info.range.base.level,
                                   desc.view_info.range.base.layer);
            }
        }
    }
    if (db_desc.first && (!scene_targets->Eligible(texture_cache.GetImage(db_desc.first)) ||
                          !same_size(texture_cache.GetImage(db_desc.first), 0))) {
        reduced = false;
    }
    if (debug_pass) {
        if (db_desc.first) {
            const auto& depth = texture_cache.GetImage(db_desc.first);
            why += fmt::format(" depth [{} {}x{} eligible {}]",
                               vk::to_string(depth.info.pixel_format), depth.info.size.width,
                               depth.info.size.height, scene_targets->Eligible(depth));
        }
        std::printf("Scene pass: reduced %d %s\n", reduced, why.c_str());
    }
    push_data.scene_size = reduced ? SceneResolution::Pack(scene_targets->Size()) : 0;
    if (Gow3Stats::enabled) {
        Gow3Stats::reduced_draws.fetch_add(reduced, std::memory_order_relaxed);
        Gow3Stats::scene_draws.fetch_add(scene_started, std::memory_order_relaxed);
    }
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            last_targets[cb].generation = ~0ULL; // FindImage may rewrite the description
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.DiscardUndefinedContents(image_id);
        // gow3 GOW3_CLEAR_SKIPS_UPLOAD (default on; 0: upload as before): the pass clears this
        // whole image (its pending fast clear, below), so its memory is not uploaded first.
        static const bool clear_skips_upload = [] {
            const char* env = std::getenv("GOW3_CLEAR_SKIPS_UPLOAD");
            return !(env && env[0] == '0');
        }();
        if (clear_skips_upload && image->info.resources.levels == 1 &&
            image->info.resources.layers == 1 &&
            texture_cache.IsMetaCleared(regs.color_buffers[cb].CmaskAddress(),
                                        desc.view_info.range.base.layer)) {
            texture_cache.DiscardClearedContents(image_id);
        }
        texture_cache.UpdateImage(image_id);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (reduced) {
            // The native image is resolved lazily when a shader/copy actually reads it.
        } else if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            runtime.FlushBarriers();
            needs_barrier |=
                runtime.Transit(image,
                                instance.IsAttachmentFeedbackLoopLayoutSupported()
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                    : vk::ImageLayout::eGeneral,
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                    vk::AccessFlagBits2::eColorAttachmentRead);
            attachment_feedback_loop = true;
        } else {
            needs_barrier |= runtime.Transit(image, vk::ImageLayout::eColorAttachmentOptimal,
                                             vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                             vk::AccessFlagBits2::eColorAttachmentWrite |
                                                 vk::AccessFlagBits2::eColorAttachmentRead,
                                             desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        if (reduced) {
            const auto target = scene_targets->Attachment(image_id, desc.view_info);
            attachment.image_view = target.view;
            attachment.image_layout = target.layout;
        }
        image->usage.render_target = 1u;
        if (cb == 0 && upscaler->Enabled()) {
            color_redirected = upscaler->RedirectColor(image_id, desc.view_info, redirect);
            if (color_redirected) {
                guest_extent = {image->info.size.width, image->info.size.height};
                original_color = {attachment.image_view, attachment.image_layout};
                attachment.image_view = redirect.view;
                attachment.image_layout = redirect.layout;
            }
        }
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        if (has_stencil && regs.depth_buffer.StencilValid() && !reduced &&
            !Gow3Toggle::Disabled(Gow3Toggle::StencilPlaneWrites)) {
            const VAddr plane = regs.depth_buffer.StencilAddress();
            const u64 plane_size = u64(regs.depth_buffer.depth_slice.tile_max + 1) * 64 *
                                   regs.depth_buffer.NumSamples();
            std::optional<StencilPlaneWrite> rewritten;
            {
                std::scoped_lock lock{stencil_plane_mutex};
                stencil_planes[plane] = plane_size;
                if (const auto it = stencil_plane_writes.find(plane); it != stencil_plane_writes.end()) {
                    rewritten = it->second;
                    stencil_plane_writes.erase(it);
                }
            }
            if (rewritten && !is_stencil_clear && image.info.num_samples == 1) {
                // A uniform fill (a clear) needs no detiling; other contents would.
                const auto [buffer, offset] = buffer_cache.ObtainBufferForImage(plane, u32(plane_size));
                runtime.CopyBufferToStencil(&image, buffer, offset, image.info.size.width,
                                            image.info.size.height);
                static std::atomic<u32> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 24) {
                    std::printf("Stencil plane copy: depth %#llx (image %#llx %ux%u), stencil plane %#llx "
                                "(%llu bytes, format %u), written by shader %#llx at %#llx +%#llx\n",
                                (unsigned long long)regs.depth_buffer.DepthAddress(),
                                (unsigned long long)image.info.guest_address, image.info.size.width,
                                image.info.size.height, (unsigned long long)plane,
                                (unsigned long long)plane_size, u32(regs.depth_buffer.stencil_info.format),
                                (unsigned long long)rewritten->shader, (unsigned long long)rewritten->begin,
                                (unsigned long long)rewritten->size);
                }
            }
        }
        // gow3 GOW3_DEPTH_WRITE_EFFECTIVE (default on; 0: the write bit alone, as upstream): depth
        // is written only with the depth test on (DB_DEPTH_CONTROL Z_ENABLE gates Z_WRITE_ENABLE;
        // Vulkan likewise, and the pipeline uses that: vk_depth_stencil_state.h), or by a pending
        // clear. A draw with the write bit but no depth test made the depth image writable (a
        // layout change) and the next draw sampling it changed it back: N-5's depth ping-pong.
        static const bool effective_writes = [] {
            const char* env = std::getenv("GOW3_DEPTH_WRITE_EFFECTIVE");
            return !(env && env[0] == '0');
        }();
        const bool depth_written =
            desc.view_info.is_storage &&
            (!effective_writes || regs.depth_control.depth_enable || is_depth_clear);
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !depth_written;
        const auto new_layout = depth_written
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        // gow3 GOW3_SKIP_READ_BARRIERS=1: a read-only depth layout cannot be written, so its access
        // is a read (it was recorded as a write, and every later read then got a barrier: T-F3).
        static const bool skip_read_barriers = [] {
            const char* env = std::getenv("GOW3_SKIP_READ_BARRIERS");
            return !(env && env[0] == '0'); // default on (launcher: perf_lean_barriers)
        }();
        const bool read_only_depth = new_layout == vk::ImageLayout::eDepthStencilReadOnlyOptimal ||
                                     new_layout == vk::ImageLayout::eDepthReadOnlyOptimal;
        const auto depth_access = skip_read_barriers && read_only_depth
                                      ? vk::AccessFlagBits2::eDepthStencilAttachmentRead
                                      : vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                            vk::AccessFlagBits2::eDepthStencilAttachmentRead;
        // gow3 GOW3_DEPTH_SAMPLE_IN_PLACE=1: back to DepthReadOnlyStencilAttachment as an attachment
        // after in-place depth-aspect samples needs no barrier: depth is never written in this
        // layout, the draw writes stencil only, and attachment accesses between draws are ordered
        // by rasterization order. Per-image tracking put one there each time (T-F5: ~43k a window).
        static const bool depth_in_place = [] {
            const char* env = std::getenv("GOW3_DEPTH_SAMPLE_IN_PLACE");
            return !(env && env[0] == '0'); // default on (launcher: perf_lean_barriers)
        }();
        if (!reduced && depth_in_place &&
            new_layout == vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal &&
            runtime.KeepStencilAttachment(&image, depth_access)) {
            // no barrier: see above
        } else if (!reduced) needs_barrier |= runtime.Transit(&image, new_layout,
                                         vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                             vk::PipelineStageFlagBits2::eLateFragmentTests,
                                         depth_access, desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};
        attachment.is_clear = 0;

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        if (reduced) {
            const auto target = scene_targets->Attachment(image_id, desc.view_info);
            attachment.image_view = target.view;
            attachment.image_layout = target.layout;
        }
        image.usage.depth_target = true;
        if (upscaler->Enabled()) {
            VulkanUpscalerTarget depth_redirect;
            depth_redirected = upscaler->RedirectDepth(image_id, depth_redirect);
            if (depth_redirected) {
                guest_extent = {image.info.size.width, image.info.size.height};
                attachment.image_view = depth_redirect.view;
                attachment.image_layout = depth_redirect.layout;
                redirect = depth_redirect;
            }
        }
    } else {
        state.depth_stencil_attachment = {};
    }

    // gow3: a pass drawn into the upscaler's output-size images (UI, display pass): every
    // attachment must be redirected, viewports and scissors are scaled.
    target_scale = {1.0f, 1.0f};
    if (color_redirected || depth_redirected) {
        u32 color_targets = 0;
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            color_targets += state.color_attachments[cb].image_view ? 1 : 0;
        }
        const bool has_depth = bool(db_desc.first);
        if ((color_targets > 0 && !color_redirected) || (has_depth && !depth_redirected) ||
            color_targets > 1) {
            static u32 warned = 0;
            if (warned < 8) {
                ++warned;
                const auto describe = [&](VideoCore::ImageId id) {
                    if (!id) {
                        return std::string{"-"};
                    }
                    const auto& info = texture_cache.GetImage(id).info;
                    return fmt::format("{:#x} {}x{} {}", info.guest_address, info.size.width,
                                       info.size.height, vk::to_string(info.pixel_format));
                };
                std::printf("Upscaler: pass not redirected (color %s: %s, depth %s: %s, %u "
                            "targets)\n",
                            color_redirected ? "yes" : "no", describe(cb_descs[0].first).c_str(),
                            depth_redirected ? "yes" : "no", describe(db_desc.first).c_str(),
                            state.num_color_attachments);
            }
            if (color_redirected) {
                state.color_attachments[0].image_view = original_color.first;
                state.color_attachments[0].image_layout = original_color.second;
            }
            if (depth_redirected) {
                auto& depth_image = texture_cache.GetImage(db_desc.first);
                state.depth_stencil_attachment.image_view =
                    *texture_cache.FindDepthTarget(db_desc.first, db_desc.second).image_view;
                state.depth_stencil_attachment.image_layout = depth_image.backing->state.layout;
            }
        } else {
            state.width = redirect.width;
            state.height = redirect.height;
            // The UI movie viewport is now 1920x1080 even when the scene is 960x540.
            // A native viewport (or native window-space UI vertices) must not be doubled
            // again. The final display copy still uses the guest render-size coordinates.
            const auto& viewport = regs.viewports[0];
            const bool native_coordinates = redirect.native_ui &&
                (UiComposition::NativeViewport(viewport.xscale * 2, viewport.yscale * 2) ||
                 (regs.IsClipDisabled() && !regs.viewport_control.xscale_enable &&
                  !regs.viewport_control.yscale_enable));
            target_scale = UiComposition::Scale(guest_extent.first, guest_extent.second,
                                                redirect.width, redirect.height,
                                                native_coordinates);
        }
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    if (reduced) {
        // Half-resolution passes keep the scene's factor (proxy / native = scene / 1920).
        const auto size = scene_targets->Size();
        const auto proxy = pass_size.value_or(size);
        state.width = proxy.width;
        state.height = proxy.height;
        target_scale = {float(size.width) / 1920.0f, float(size.height) / 1080.0f};
    }
    if (FrameCapture::TargetWatch() && !FrameCapture::Active()) {
        // gow3 GOW3_CAPTURE_ON_TARGET: a pass into a target of the watched size starts a capture.
        std::array<const VideoCore::ImageInfo*, AmdGpu::NUM_COLOR_BUFFERS> colors{};
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            if (cb_descs[cb].first) {
                colors[cb] = &texture_cache.GetImage(cb_descs[cb].first).info;
            }
        }
        FrameCapture::StartForTarget(colors.data(), state.num_color_attachments);
    }
    if (FrameCapture::Active()) {
        std::array<const VideoCore::ImageInfo*, AmdGpu::NUM_COLOR_BUFFERS> colors{};
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            if (cb_descs[cb].first) {
                colors[cb] = &texture_cache.GetImage(cb_descs[cb].first).info;
            }
        }
        const auto* depth =
            db_desc.first ? &texture_cache.GetImage(db_desc.first).info : nullptr;
        FrameCapture::BeginPass(colors.data(), state.num_color_attachments, depth);
    }
    // gow3: object motion vector attachment of G-buffer pipelines.
    if (key.motion_vectors) {
        object_motion->Attach(state, state.width, state.height);
    }
    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = CbExtent(0);
    const auto& mrt1_hint = CbExtent(1);
    VideoCore::TextureCache::ImageDesc mrt0_desc{Regs().color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{Regs().color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 Regs().color_buffers[0].Address(),
                                 Regs().color_buffers[1].Address()));
    runtime.ResolveImage(&mrt0_image, &mrt1_image, mrt0_desc.view_info.range,
                         mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = Regs();

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), DbExtent(), false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), DbExtent(), true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = Regs().depth_view.slice_start;
    sub_range.extent.layers = Regs().depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    runtime.CopyDepthStencil(&read_image, &write_image, sub_range);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    DrainDrawPipe();
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!buffer_cache.IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + (num_bytes / sizeof(u32)), value);
            return;
        }
    }
    const auto [buffer, offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (is_gds) {
            return {buffer_cache.GetGdsBuffer(), address};
        }
        return buffer_cache.ObtainBuffer(address, num_bytes, true);
    }();
    runtime.FillBuffer(buffer, offset, num_bytes, value);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    DrainDrawPipe();
    if (!dst_gds && !buffer_cache.IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    const auto* gds_buffer = buffer_cache.GetGdsBuffer();
    const auto [src_buffer, src_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (src_gds) {
            return {gds_buffer, src};
        }
        return buffer_cache.ObtainBuffer(src, num_bytes, false, true);
    }();
    const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (dst_gds) {
            return {gds_buffer, dst};
        }
        return buffer_cache.ObtainBuffer(dst, num_bytes, true, true);
    }();
    const vk::BufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = num_bytes,
    };
    runtime.CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    DrainDrawPipe();
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, bool assume_locks) {
    if (Gow3Stats::enabled) {
        // gow3 gate 2: who invalidates, and what the drain below costs that thread.
        const u64 start = Gow3FenceStats::NowNs();
        DrainDrawPipe();
        Gow3FenceStats::NoteInvalidate(size, Gow3FenceStats::NowNs() - start);
    } else {
        DrainDrawPipe();
    }
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    texture_cache.ResolveReadbacks(addr, size, assume_locks, VideoCore::ReadbackSource::CpuWrite);
    buffer_cache.InvalidateMemory(addr, size, assume_locks);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

namespace {
struct ZpassEvent {
    VAddr address;
    u32 pairs;
};
} // namespace

bool Rasterizer::OcclusionEvent(VAddr address, u32 pairs) {
    if (!OcclusionQueries::Enabled() || !instance.IsHostQueryResetSupported() || pairs == 0) {
        return false;
    }
    if (!occlusion) {
        occlusion = std::make_unique<OcclusionQueries>(instance, scheduler);
    }
    const ZpassEvent event{address, pairs};
    RunInOrder(
        [](Rasterizer& self, const u8* data) {
            GOW3_SECTION(TaskOcclusion);
            const auto& e = *reinterpret_cast<const ZpassEvent*>(data);
            self.occlusion->Event(e.address, e.pairs);
        },
        &event, sizeof(event));
    return true;
}

bool Rasterizer::PreciseCpuWrite(VAddr addr, const void* src, int value, u64 size) {
    if (Gow3Toggle::Disabled(Gow3Toggle::PreciseCpuWrites) || !IsMapped(addr, size)) {
        return false;
    }
    return buffer_cache.PreciseCpuWrite(addr, src, value, size);
}

bool Rasterizer::OnWriteFault(VAddr addr, bool assume_locks) {
    Gow3Stats::DownloadReasonScope reason{Gow3Stats::WriteFault};
    DrainDrawPipe();
    if (!InvalidateMemory(addr, 8, assume_locks)) {
        return false;
    }
    buffer_cache.ExtendWriteFault(addr);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks) {
    Gow3Stats::DownloadReasonScope reason{Gow3Stats::ReadFault};
    DrainDrawPipe();
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    const bool image_readback =
        texture_cache.ResolveReadbacks(addr, size, assume_locks, VideoCore::ReadbackSource::CpuRead);
    if (!image_readback || buffer_cache.IsRegionGpuModified(addr, size)) {
        buffer_cache.ReadMemory(addr, size, false, assume_locks);
    }
    return true;
}

void Rasterizer::ProcessDownloadImages() {
    DrainDrawPipe();
    texture_cache.ProcessDownloadImages();
    // gow3 R1: with precise readbacks the game reads a readback (UE4's HZB occlusion results)
    // about a frame after asking for it; recorded here but submitted at the frame's end, the
    // copy ran only after the rest of the frame and the game waited ~2.6 ms for it (T-HZ2b).
    // Submitted now, the GPU makes it right after the work it reads. GOW3_READBACK_SUBMIT=0: off.
    static const bool submit = [] {
        const char* env = std::getenv("GOW3_READBACK_SUBMIT");
        return !env || env[0] != '0';
    }();
    if (texture_cache.TakeReadbackSubmitWanted() && submit) {
        scheduler.Flush();
    }
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::RegisterMemory(VAddr addr, u64 size) {
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    Gow3Stats::DownloadReasonScope reason{Gow3Stats::Unmap};
    DrainDrawPipe();
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    GOW3_SECTION(UpdateDynamicState);
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.CommitWith(instance.IsDepthBoundsSupported(),
                             instance.IsDynamicColorWriteMaskSupported(),
                             instance.IsAttachmentFeedbackLoopLayoutSupported(),
                             [&](auto&& command) { scheduler.Record(std::move(command)); });
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = Regs();

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            // gow3: sub-pixel jitter of scene geometry for the temporal upscaler; the same
            // shift as jittering the projection.
            viewport.x = (xoffset - xscale) * target_scale[0] + draw_jitter[0];
            viewport.y = (yoffset - yscale) * target_scale[1] + draw_jitter[1];
            viewport.width = xscale * 2.0f * target_scale[0];
            viewport.height = yscale * 2.0f * target_scale[1];
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        if (target_scale[0] != 1.0f || target_scale[1] != 1.0f) {
            const auto scale = [](s32 v, float f) { return s32(std::lround(float(v) * f)); };
            const s32 x0 = scale(vp_scsr.top_left_x, target_scale[0]);
            const s32 y0 = scale(vp_scsr.top_left_y, target_scale[1]);
            const s32 x1 = scale(vp_scsr.top_left_x + s32(vp_scsr.GetWidth()), target_scale[0]);
            const s32 y1 = scale(vp_scsr.top_left_y + s32(vp_scsr.GetHeight()), target_scale[1]);
            scissors.push_back({
                .offset = {x0, y0},
                .extent = {u32(std::max(x1 - x0, 0)), u32(std::max(y1 - y0, 0))},
            });
        } else {
            scissors.push_back({
                .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
                .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
            });
        }
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

vk::Pipeline Rasterizer::SquarePass(const GraphicsPipeline& pipeline) const {
    const vk::Pipeline square = pipeline.SquarePassHandle();
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_SQUARE_PASS");
        return !env || env[0] != '0';
    }();
    if (!square || !enabled) [[likely]] {
        return {};
    }
    // MIN(s * s, d * d) = MIN(s, d)^2: the second draw needs the first one's coverage, which a
    // depth or stencil write would change.
    const auto& regs = Regs();
    const bool depth_write = regs.depth_control.depth_enable &&
                             regs.depth_control.depth_write_enable && regs.depth_buffer.DepthValid();
    if (depth_write || regs.depth_control.stencil_enable) {
        static bool warned = false;
        if (!std::exchange(warned, true)) {
            LOG_WARNING(Render_Vulkan, "Scaled MIN/MAX blend not squared: the draw writes depth "
                                       "or stencil");
        }
        return {};
    }
    return square;
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_stencil = GetEffectiveDepthStencilState(regs);
    const bool depth_test_enabled = depth_stencil.depth_test_enable;
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    dynamic_state.SetDepthWriteEnabled(depth_stencil.depth_write_enable &&
                                       !regs.depth_render_control.depth_clear_enable);
    if (depth_test_enabled) {
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const bool depth_bounds_test_enabled = depth_stencil.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        // gow3: the constant offset is in units of the depth format the game names
        // (PA_SU_POLY_OFFSET_DB_FMT_CNTL: 2^-16 for Z16). When the host depth buffer has another
        // precision (Z16+S8 has no PC equivalent and becomes D32S8, about 2^-24 per unit), the
        // same number was ~256 times too small: shadow acne, shadows flickering while moving.
        float constant_scale = 1.0f;
        if (const auto db_fmt = regs.poly_offset_db_fmt; !db_fmt.IsFloat() && db_fmt.NumDbBits() > 0) {
            static thread_local u32 cached_key = ~0u;
            static thread_local s32 cached_host_bits = 0;
            const u32 key = (u32(regs.depth_buffer.z_info.format) << 8) |
                            u32(regs.depth_buffer.stencil_info.format);
            if (key != cached_key) {
                cached_key = key;
                switch (instance.GetSupportedFormat(
                    LiverpoolToVK::DepthFormat(regs.depth_buffer.z_info.format,
                                               regs.depth_buffer.stencil_info.format),
                    vk::FormatFeatureFlagBits2::eDepthStencilAttachment)) {
                case vk::Format::eD16Unorm:
                case vk::Format::eD16UnormS8Uint:
                    cached_host_bits = 16;
                    break;
                case vk::Format::eX8D24UnormPack32:
                case vk::Format::eD24UnormS8Uint:
                case vk::Format::eD32Sfloat: // float: 2^(e-23), 2^-24 for depths in [0.5, 1)
                case vk::Format::eD32SfloatS8Uint:
                    cached_host_bits = 24;
                    break;
                default:
                    cached_host_bits = 0;
                    break;
                }
            }
            if (cached_host_bits != 0 && cached_host_bits != db_fmt.NumDbBits()) {
                constant_scale = std::ldexp(1.0f, cached_host_bits - db_fmt.NumDbBits());
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true)) {
                    std::printf("GPU: depth bias in %d-bit units scaled x%g for the host depth format\n",
                                db_fmt.NumDbBits(), constant_scale);
                    std::fflush(stdout);
                }
            }
        }
        dynamic_state.SetDepthBias(
            (front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset) * constant_scale,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const bool stencil_test_enabled = depth_stencil.stencil_test_enable;
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
        // include ReplaceOp takes its Vulkan reference from op_val.
        const auto& sc = regs.stencil_control;
        const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                    AmdGpu::StencilFunc zfail) {
            return fail == AmdGpu::StencilFunc::ReplaceOp ||
                   zpass == AmdGpu::StencilFunc::ReplaceOp ||
                   zfail == AmdGpu::StencilFunc::ReplaceOp;
        };
        const bool front_op =
            uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op =
            regs.depth_control.backface_enable
                ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                : front_op;
        const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
            return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
                   ref.stencil_test_val != ref.stencil_op_val;
        };
        if ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
            (back_op && regs.depth_control.backface_enable &&
             ref_conflict(regs.depth_control.stencil_bf_func, back))) {
            LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                       "op_val; the stencil test will use op_val");
        }
        dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                           back_op ? back.stencil_op_val : back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    DrainDrawPipe();
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    DrainDrawPipe();
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    DrainDrawPipe();
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    DrainDrawPipe();
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool->GetGpuCommandProcessorThread();
}

#if defined(__linux__) || defined(_WIN32)
u32 Rasterizer::GetGpuCommandProcessorThreadId() {
    return liverpool->GetGpuCommandProcessorThreadId();
}
#endif

} // namespace Vulkan

namespace Vulkan {

void Rasterizer::MarkPass(const GraphicsPipeline* pipeline, const RenderState& state) {
    auto* profiler = GpuProfiler::Get();
    if (!profiler || !scheduler.WillBeginRendering(state)) {
        return;
    }
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    std::array<u64, AmdGpu::NUM_COLOR_BUFFERS + 6> parts{};
    parts[0] = state.width | u64(state.height) << 32;
    parts[1] = vs.pgm_hash;
    parts[2] = ps ? ps->pgm_hash : 0;
    parts[3] = db_desc.first ? u64(texture_cache.GetImage(db_desc.first).info.pixel_format) : 0;
    for (u32 cb = 0; cb < state.num_color_attachments && cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        parts[4 + cb] = cb_descs[cb].first
                            ? u64(texture_cache.GetImage(cb_descs[cb].first).info.pixel_format) + 1
                            : 0;
    }
    const u64 key = XXH3_64bits(parts.data(), sizeof(parts));
    profiler->Mark(key, [&] {
        std::string targets;
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            targets += cb_descs[cb].first
                           ? vk::to_string(texture_cache.GetImage(cb_descs[cb].first).info.pixel_format)
                           : std::string{"-"};
            targets += ' ';
        }
        return fmt::format("pass {}x{} [{}] depth {} vs {:016x} ps {:016x}", state.width,
                           state.height, targets,
                           db_desc.first ? vk::to_string(
                                               texture_cache.GetImage(db_desc.first).info.pixel_format)
                                         : std::string{"-"},
                           vs.pgm_hash, ps ? ps->pgm_hash : 0);
    });
}

void Rasterizer::NoteFrameStart() {
    if (auto* profiler = GpuProfiler::Get()) {
        profiler->BeginFrame();
    }
    const u64 frame = Gow3Stats::gpu_frames.fetch_add(1, std::memory_order_relaxed) + 1;
    // GOW3_BUFFER_STATS=1: how many earlier frames the GPU has not finished when the GPU thread
    // starts a frame — the lag that decides whether guest memory can be read in place.
    static const bool stats = [] {
        const char* value = std::getenv("GOW3_BUFFER_STATS");
        return value && value[0] == '1';
    }();
    if (!stats) {
        return;
    }
    static std::array<u64, 16> frame_ticks{};
    static std::array<u64, 17> lag_histogram{};
    u32 lag = 0;
    for (u32 back = 1; back < frame_ticks.size() && back < frame; ++back) {
        const u64 tick = frame_ticks[(frame - back) % frame_ticks.size()];
        if (tick && !scheduler.IsFree(tick)) {
            lag = back;
        }
    }
    ++lag_histogram[lag];
    frame_ticks[frame % frame_ticks.size()] = scheduler.CurrentTick();
    if (frame % 600 == 0) {
        std::printf("GPU lag at frame start (frames not finished: count):");
        for (u32 i = 0; i < lag_histogram.size(); ++i) {
            if (lag_histogram[i]) {
                std::printf(" %u:%llu", i, static_cast<unsigned long long>(lag_histogram[i]));
            }
        }
        std::printf("\n");
        lag_histogram = {};
    }
}

bool Rasterizer::ObserveFrameMap(const Pipeline* pipeline) {
    Ue4FrameMap::Pass pass;
    const auto& regs = Regs();
    pass.compute = pipeline->IsCompute();
    if (pass.compute) {
        const auto& cs = CsRegs();
        pass.stage = "cs";
        pass.groups = {cs.dim_x, cs.dim_y, cs.dim_z};
    } else {
        const auto& key = static_cast<const GraphicsPipeline*>(pipeline)->GetGraphicsKey();
        pass.targets = std::popcount(key.mrt_mask);
        if ((key.mrt_mask & 1) && cb_descs[0].first) {
            const auto& info = cb_descs[0].second.info;
            pass.target = {regs.color_buffers[0].Address(), info.size.width, info.size.height};
            pass.target_rg16 = info.pixel_format == vk::Format::eR16G16Unorm;
            pass.target_rgba16f = info.pixel_format == vk::Format::eR16G16B16A16Sfloat;
            pass.display = FrameCapture::IsDisplayBuffer(pass.target.address);
        }
        if (db_desc.first) {
            pass.depth = regs.depth_buffer.DepthAddress();
        }
    }
    // U4: the first storage image a dispatch writes (the TAAU dispatch's output), to find its
    // image only when DLSS will replace the dispatch.
    std::optional<VideoCore::TextureCache::ImageDesc> written_desc;
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        if (stage->sw_stage == Shader::SwStage::Fragment ||
            stage->sw_stage == Shader::SwStage::Compute) {
            pass.hash = stage->pgm_hash;
        } else if (stage->sw_stage == Shader::SwStage::Vertex && !pass.hash) {
            pass.stage = "vs";
            pass.hash = stage->pgm_hash;
        }
        if (stage->sw_stage == Shader::SwStage::Fragment) {
            pass.stage = "fs";
        }
        const PreparedStage* prepared = FindPreparedStage(*stage);
        for (u32 i = 0; i < stage->images.size(); ++i) {
            const auto& desc = stage->images[i];
            const auto tsharp = prepared ? prepared->image_sharps[i] : desc.GetSharp(*stage);
            if (tsharp.Address() == 0) {
                continue;
            }
            if (desc.is_written) {
                if (pass.compute && !written_desc && ue4_dlss && Ue4Dlss::On()) {
                    written_desc.emplace(tsharp, desc);
                }
                if (pass.num_writes < pass.writes.size()) {
                    pass.writes[pass.num_writes++] = {tsharp.Address(), u32(tsharp.width) + 1,
                                                      u32(tsharp.height) + 1};
                }
            } else if (pass.num_reads < pass.reads.size()) {
                pass.read_sizes[pass.num_reads] = {u32(tsharp.width) + 1, u32(tsharp.height) + 1};
                pass.reads[pass.num_reads++] = tsharp.Address();
            }
        }
        for (u32 i = 0; i < stage->buffers.size(); ++i) {
            const auto& desc = stage->buffers[i];
            if (desc.IsSpecial()) {
                continue;
            }
            VAddr address;
            u32 size;
            if (const RingBinding* ring = FindRingBinding(*stage, i)) {
                address = ring->address;
                size = ring->size;
            } else {
                const auto vsharp = prepared ? prepared->buffer_sharps[i] : desc.GetSharp(*stage);
                address = vsharp.base_address;
                size = vsharp.GetSize();
            }
            if (pass.num_buffers < pass.buffer_sizes.size()) {
                pass.buffer_sizes[pass.num_buffers++] = size;
            }
            if (size == Ue4FrameMap::ViewBytes && !pass.view && address &&
                memory->IsValidGpuMapping(address, size)) {
                pass.view = reinterpret_cast<const u8*>(address);
            }
            if (size == Ue4FrameMap::TaaParamBytes && !pass.taa_params && address &&
                memory->IsValidGpuMapping(address, size)) {
                pass.taa_params = reinterpret_cast<const u8*>(address);
            }
        }
    }
    if (frame_map) {
        frame_map->OnPass(pass);
    }
    bool skip = false;
    // U4: the TAAU compute TAA by role, not only by its hash: UE4 compiles a permutation per
    // upscale ratio (`cs b373f39908ae979f` at 80-92%; at lower scales another shader drew and
    // nothing ran, run u4-dlss). A dispatch writing one image while sampling the velocity target;
    // its hash is remembered for frames without velocity draws. (Under TAAU only TAA's dispatch
    // samples velocity: U1's readers list.)
    bool taau = pass.compute && pass.hash == Ue4FrameMap::TaaCompute;
    if (pass.compute && !taau && ue4_motion) {
        static std::vector<u64> taau_hashes;
        const VAddr velocity = ue4_motion->VelocityAddress();
        if (std::ranges::find(taau_hashes, pass.hash) != taau_hashes.end()) {
            taau = true;
        } else if (velocity && pass.num_writes == 1 &&
                   std::find(pass.reads.begin(), pass.reads.begin() + pass.num_reads, velocity) !=
                       pass.reads.begin() + pass.num_reads) {
            taau = true;
            taau_hashes.push_back(pass.hash);
            std::printf("UE4 motion: TAAU compute shader cs %016llx (samples velocity, writes one "
                        "%ux%u image)\n",
                        static_cast<unsigned long long>(pass.hash), pass.writes[0].width,
                        pass.writes[0].height);
        }
    }
    if (ue4_motion) {
        if (!pass.compute && pass.target_rg16 && pass.targets == 1 && pass.depth) {
            ue4_motion->OnVelocityDraw(cb_descs[0].first, pass.target.address, db_desc.first);
        } else if (!pass.compute && pass.targets >= 3 && db_desc.first) {
            ue4_motion->OnSceneDraw(db_desc.first);
        } else if ((!pass.compute && (pass.hash == Ue4FrameMap::TaaNormal ||
                                      pass.hash == Ue4FrameMap::TaaResponsive)) ||
                   taau) {
            ue4_motion->OnTaa(pass.view, std::span(pass.reads.data(), pass.num_reads),
                              pass.compute ? VideoCore::ImageId{} : db_desc.first);
            if (ue4_dlss && pass.compute && written_desc && pass.num_writes) {
                const VideoCore::ImageId output_id = texture_cache.FindImage(*written_desc);
                skip = ue4_dlss->OnTaauDispatch(
                    pass.view, std::span(pass.reads.data(), pass.num_reads), output_id,
                    pass.writes[0].address, pass.writes[0].width, pass.writes[0].height, [&] {
                        texture_cache.FindTexture(output_id, *written_desc);
                    });
            }
            if (ue4_dlss && !pass.compute) {
                const auto& [output_id, output_desc] = cb_descs[0];
                skip = ue4_dlss->OnTaaDraw(pass.view, std::span(pass.reads.data(), pass.num_reads),
                                           output_id, pass.target.address, [&] {
                                               texture_cache.FindRenderTarget(output_id, output_desc);
                                           });
            }
        } else if (ue4_dlss && !pass.compute && pass.targets == 1 && pass.target_rgba16f) {
            ue4_dlss->OnColorTarget(cb_descs[0].first, pass.target.address);
        }
    }
    return skip;
}

} // namespace Vulkan
