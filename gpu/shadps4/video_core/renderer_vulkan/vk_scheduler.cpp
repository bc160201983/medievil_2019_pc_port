// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>
#include <mutex>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif
#include <functional>

#include "gow3_copy.h"
#include "gow3_sampler.h"
#include "gow3_stage_b_wait.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "gow3_toggles.h"
#include "video_core/renderer_vulkan/vk_wait_diagnostics.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include <cstdio>
#include "gow3_threads.h"
#include "gow3_fence_stats.h"
#include "gow3_gpu_fences.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance, bool threaded_recording)
    : instance{instance}, work_semaphore{instance}, command_pool{instance, &work_semaphore} {
    // gow3: GOW3_VK_RECORD_THREAD=0 records on the calling thread.
    const char* env = std::getenv("GOW3_VK_RECORD_THREAD");
    if (threaded_recording && !(env && env[0] == '0')) {
        record_chunk = AcquireChunk();
        recorder_thread = std::jthread(std::bind_front(&Scheduler::RecorderThread, this));
        recording_threaded = true;
    }
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    AllocateWorkerCommandBuffers();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
    if (recorder_thread.joinable()) {
        SyncRecording();
        recorder_thread.request_stop();
        recorder_cv.notify_all();
        recorder_thread.join();
    }
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

// gow3 GOW3_PERF_DIAG: the targets (views) of two render states are the same.
static bool SameTargets(const RenderState& a, const RenderState& b) {
    if (a.num_color_attachments != b.num_color_attachments ||
        a.depth_stencil_attachment.image_view != b.depth_stencil_attachment.image_view) {
        return false;
    }
    for (u32 i = 0; i < a.num_color_attachments; ++i) {
        if (a.color_attachments[i].image_view != b.color_attachments[i].image_view) {
            return false;
        }
    }
    return true;
}

static const char* SwitchReason(const RenderState& a, const RenderState& b) {
    if (!SameTargets(a, b)) {
        return "other targets";
    }
    if (a.width != b.width || a.height != b.height || a.num_layers != b.num_layers) {
        return "same targets, other size or layers";
    }
    bool layout = a.depth_stencil_attachment.image_layout != b.depth_stencil_attachment.image_layout;
    bool clear = b.depth_stencil_attachment.is_clear != 0;
    for (u32 i = 0; i < a.num_color_attachments; ++i) {
        layout |= a.color_attachments[i].image_layout != b.color_attachments[i].image_layout;
        clear |= b.color_attachments[i].is_clear != 0;
    }
    return layout ? "same targets, other layout" : clear ? "same targets, a clear" : "same targets, other";
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    if (WaitDiagnostics::Enabled()) {
        if (is_rendering) {
            WaitDiagnostics::PassSwitches().Add(SwitchReason(render_state, new_state), 0, 0, 0);
        } else if (last_end_file && SameTargets(last_ended, new_state)) {
            WaitDiagnostics::PassReopens().Add(last_end_file, last_end_line, 0, 0);
        }
        last_end_file = nullptr;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;

    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    if (!recording_threaded) {
        current_cmdbuf.beginRendering(rendering_info);
        if (render_pass_hook) {
            render_pass_hook->OnBeginRendering();
        }
        return;
    }
    // The attachment infos live on this stack frame: the recorded closure keeps copies.
    Record([info = rendering_info, color_attachments, depth_attachment,
            stencil_attachment](vk::CommandBuffer cmdbuf) mutable {
        info.pColorAttachments = color_attachments.data();
        if (info.pDepthAttachment) {
            info.pDepthAttachment = &depth_attachment;
        }
        if (info.pStencilAttachment) {
            info.pStencilAttachment = &stencil_attachment;
        }
        cmdbuf.beginRendering(info);
    });
    if (render_pass_hook) {
        render_pass_hook->OnBeginRendering();
    }
}

void Scheduler::EndRendering(std::source_location caller) {
    SyncWithDrawRecorder(__builtin_return_address(0));
    if (!is_rendering) {
        return;
    }
    Gow3Stats::render_pass_ends.fetch_add(1, std::memory_order_relaxed);
    if (WaitDiagnostics::Enabled()) {
        WaitDiagnostics::RenderPassEnds().Add(caller.file_name(), caller.line(), 0, 0);
        last_ended = render_state;
        last_end_file = caller.file_name();
        last_end_line = caller.line();
    }
    if (render_pass_hook) {
        render_pass_hook->OnEndRendering();
    }
    is_rendering = false;
    Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRendering(); });
}

void Scheduler::TraceDirectRecording(void* caller) {
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_RECORDER_TRACE");
        return env && env[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    static std::mutex mutex;
    static std::unordered_map<void*, u64> callers;
    static u64 calls;
    std::scoped_lock lk{mutex};
    ++callers[caller];
    if (++calls % 2000) {
        return;
    }
    std::vector<std::pair<u64, void*>> top;
    for (const auto& [address, count] : callers) {
        top.emplace_back(count, address);
    }
    std::ranges::sort(top, std::greater{});
    for (size_t i = 0; i < std::min<size_t>(top.size(), 8); ++i) {
#ifdef _WIN32
        struct { const char* dli_fname; void* dli_fbase; } info{nullptr, nullptr};
#else
        Dl_info info{};
        dladdr(top[i].second, &info);
#endif
        std::printf("Recorder sync caller: %llu x %s+0x%lx\n",
                    static_cast<unsigned long long>(top[i].first),
                    info.dli_fname ? info.dli_fname : "?",
                    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(top[i].second) -
                                               reinterpret_cast<uintptr_t>(info.dli_fbase)));
    }
    callers.clear();
}

void Scheduler::ReportProducer(void* caller) {
    char name[128] = "?";
#ifdef _WIN32
    using GetDescription = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const auto get_description = reinterpret_cast<GetDescription>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription")));
    PWSTR wide = nullptr;
    if (get_description && SUCCEEDED(get_description(GetCurrentThread(), &wide)) && wide) {
        std::snprintf(name, sizeof(name), "%ls", wide);
        LocalFree(wide);
    }
    const auto base = reinterpret_cast<uintptr_t>(GetModuleHandleA(nullptr));
#else
    const uintptr_t base = 0;
#endif
    std::printf("Scheduler %p: commands recorded from thread '%s' (caller medievil-runtime.exe+%#llx)\n",
                static_cast<void*>(this), name,
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(caller) - base));
}

void Scheduler::ReportStaleRecordData(void* caller) {
    static std::atomic<u32> count{0};
    const u32 n = count.fetch_add(1, std::memory_order_relaxed);
    if (n < 8 || (n & (n - 1)) == 0) {
#ifdef _WIN32
        const auto base = reinterpret_cast<uintptr_t>(GetModuleHandleA(nullptr));
#else
        const uintptr_t base = 0; // medievil-runtime is not position-independent on Linux
#endif
        std::printf("Scheduler: a command landed in another chunk than its data (%u times; caller "
                    "medievil-runtime.exe+%#llx)\n",
                    n + 1, static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(caller) - base));
        std::fflush(stdout);
    }
}

void Scheduler::ReportInterleave(void* caller) {
    static std::mutex mutex;
    static void* reported[16]{};
    {
        std::scoped_lock lk{mutex};
        for (void*& slot : reported) {
            if (slot == caller) {
                return;
            }
            if (!slot) {
                slot = caller;
                goto report;
            }
        }
        return;
    }
report:
    char name[128] = "?";
#ifdef _WIN32
    using GetDescription = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const auto get_description = reinterpret_cast<GetDescription>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription")));
    PWSTR wide = nullptr;
    if (get_description && SUCCEEDED(get_description(GetCurrentThread(), &wide)) && wide) {
        std::snprintf(name, sizeof(name), "%ls", wide);
        LocalFree(wide);
    }
    const auto base = reinterpret_cast<uintptr_t>(GetModuleHandleA(nullptr));
#else
    const uintptr_t base = 0;
#endif
    std::printf("Scheduler: recorded while the draw recorder had queued draws (thread '%s', caller "
                "medievil-runtime.exe+%#llx)\n",
                name, static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(caller) - base));
#ifdef _WIN32
    // The path that got here: offsets in medievil-runtime.exe, other modules as addresses.
    void* frames[12]{};
    const USHORT count = RtlCaptureStackBackTrace(1, 12, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        const auto address = reinterpret_cast<uintptr_t>(frames[i]);
        HMODULE module{};
        const bool own = GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                            reinterpret_cast<LPCSTR>(frames[i]), &module) &&
                         reinterpret_cast<uintptr_t>(module) == base;
        std::printf(own ? "  #%u medievil-runtime.exe+%#llx\n" : "  #%u %#llx\n", i,
                    static_cast<unsigned long long>(own ? address - base : address));
    }
#endif
    std::fflush(stdout);
}

std::unique_ptr<RecordChunk> Scheduler::AcquireChunk() {
    std::scoped_lock lk{recorder_mutex};
    if (free_chunks.empty()) {
        return std::make_unique<RecordChunk>();
    }
    auto chunk = std::move(free_chunks.back());
    free_chunks.pop_back();
    return chunk;
}

void Scheduler::SignalAfterHostCopies(std::function<void()> signal, bool urgent) {
    // gow3 gate 2 (GOW3_FRAME_STATS): the fence's timeline tick and our signal time, for the lag.
    const u64 fence = Gow3FenceStats::BeginFence();
    if (fence != ~0ull) {
        Gow3FenceStats::SetGpuWaiter(this, [](void* scheduler, u64 tick) {
            static_cast<Scheduler*>(scheduler)->GetWorkSemaphore()->Wait(tick);
        });
    }
    if (!IsRecordingDeferred()) {
        WaitHostCopies();
        Gow3FenceStats::FenceRecorded(fence, CurrentTick());
        signal();
        Gow3FenceStats::FenceSignaled(fence);
        return;
    }
    Gow3Copy::FlushBatch();
    deferred_signals_issued.fetch_add(1, std::memory_order_relaxed);
    Record([this, fence, signal = std::move(signal), done = deferred_signals_done](vk::CommandBuffer) mutable {
        Gow3FenceStats::FenceRecorded(fence, CurrentTick());
        Gow3Copy::AfterCopies([fence, signal = std::move(signal), done = std::move(done)] {
            signal();
            Gow3FenceStats::FenceSignaled(fence);
            done->fetch_add(1, std::memory_order_release);
            Gow3SignalWait::Notify(); // a recording thread may sleep in WaitDeferredSignals
        });
    });
    // gow3 (impl-deck-cpu-bundle.md): a polled label recorded on stage B may leave a sleeping
    // recorder asleep (lean mode); the order of the signals does not change, only when the
    // recorder starts on them. Stage B is the only thread whose idle Kick() can count on.
    Kick(!urgent && Gow3Sections::recording_thread ? KickKind::Label : KickKind::Now);
}

void Scheduler::WaitDeferredSignals() {
    const u64 issued = deferred_signals_issued.load(std::memory_order_relaxed);
    if (deferred_signals_done->load(std::memory_order_acquire) >= issued ||
        Gow3Toggle::Disabled(Gow3Toggle::OrderedGuestWrites)) {
        return;
    }
    Gow3Stats::WaitTimer timer{Gow3Stats::host_copies_wait_ns};
    KickRecording(true);
    const auto done = [&] { return deferred_signals_done->load(std::memory_order_acquire) >= issued; };
    if (Gow3SignalWait::Enabled()) {
        // gow3 (gow3_signal_wait.h): helps the copy threads once, then pauses and sleeps.
        Gow3Copy::WaitAsync();
        Gow3SignalWait::Wait(done, Gow3Stats::enabled);
        return;
    }
    const u64 start_ns = Gow3Stats::enabled ? Gow3DrainWait::NowNs() : 0;
    while (!done()) {
        // Helps the copy threads the signals wait for.
        Gow3Copy::WaitAsync();
        std::this_thread::yield();
    }
    if (Gow3Stats::enabled) {
        Gow3SignalWait::waits.fetch_add(1, std::memory_order_relaxed);
        Gow3SignalWait::wait_ns.fetch_add(Gow3DrainWait::NowNs() - start_ns, std::memory_order_relaxed);
    }
}

namespace {
constexpr u32 HostCopyGranuleShift = 16;
constexpr u64 HostCopyMaxGranules = 64; // larger notes and checks are treated as everywhere
u32 HostCopySlotOf(u64 granule, u32 slots) {
    return u32((granule * 0x9e3779b97f4a7c15ull) >> 40) % slots;
}
} // namespace

void Scheduler::NoteHostCopySource(u64 address, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 first = address >> HostCopyGranuleShift;
    const u64 last = (address + size - 1) >> HostCopyGranuleShift;
    std::scoped_lock lk{host_copy_sources_mutex};
    const u64 seq = ++host_copy_source_seq;
    if (last - first >= HostCopyMaxGranules) {
        host_copy_big_seq = seq;
        return;
    }
    for (u64 g = first; g <= last; ++g) {
        auto& slot = host_copy_slots[HostCopySlotOf(g, HostCopySlots)];
        if (slot.seq > host_copy_sources_done && slot.granule != g) {
            slot.granule = HostCopyWildcard; // two pending granules share it
        } else if (slot.granule != HostCopyWildcard || slot.seq <= host_copy_sources_done) {
            slot.granule = g;
        }
        slot.seq = seq;
    }
}

void Scheduler::WaitHostCopiesFor(u64 address, u64 size, std::source_location caller) {
    if (!Gow3Toggle::Disabled(Gow3Toggle::TargetedHostCopyWaits)) {
        std::scoped_lock lk{host_copy_sources_mutex};
        bool overlap = host_copy_big_seq > host_copy_sources_done;
        const u64 first = address >> HostCopyGranuleShift;
        const u64 last = (address + std::max<u64>(size, 1) - 1) >> HostCopyGranuleShift;
        if (last - first >= HostCopyMaxGranules) {
            overlap = true;
        }
        for (u64 g = first; !overlap && g <= last; ++g) {
            const auto& slot = host_copy_slots[HostCopySlotOf(g, HostCopySlots)];
            overlap = slot.seq > host_copy_sources_done &&
                      (slot.granule == g || slot.granule == HostCopyWildcard);
        }
        if (!overlap) {
            Gow3Stats::host_copy_waits_skipped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    WaitHostCopies(caller);
}

void Scheduler::WaitHostCopies(std::source_location caller) {
    u64 sources_seen;
    {
        std::scoped_lock lk{host_copy_sources_mutex};
        sources_seen = host_copy_source_seq;
    }
    // Copies noted after this point may still run when the waits below return.
    struct Forget {
        Scheduler& s;
        u64 seen;
        ~Forget() {
            std::scoped_lock lk{s.host_copy_sources_mutex};
            s.host_copy_sources_done = std::max(s.host_copy_sources_done, seen);
        }
    } forget{*this, sources_seen};
    const bool diagnose = WaitDiagnostics::Enabled();
    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto start = diagnose ? now() : std::chrono::steady_clock::time_point{};
    bool waited = false;
    if (host_copies_done.load(std::memory_order_acquire) < host_copies_issued) {
        waited = true;
        Gow3Stats::WaitTimer timer{Gow3Stats::host_copies_wait_ns};
        Gow3Stats::host_copy_waits.fetch_add(1, std::memory_order_relaxed);
        KickRecording(true);
        const u64 issued = host_copies_issued;
        const auto done = [&] { return host_copies_done.load(std::memory_order_acquire) >= issued; };
        if (Gow3SignalWait::Enabled()) {
            // gow3 (gow3_signal_wait.h): pauses, then sleeps until the recorder ran the copies.
            Gow3SignalWait::Wait(done, Gow3Stats::enabled);
        } else {
            const u64 start_ns = Gow3Stats::enabled ? Gow3DrainWait::NowNs() : 0;
            while (!done()) {
                std::this_thread::yield();
            }
            if (Gow3Stats::enabled) {
                Gow3SignalWait::waits.fetch_add(1, std::memory_order_relaxed);
                Gow3SignalWait::wait_ns.fetch_add(Gow3DrainWait::NowNs() - start_ns,
                                                  std::memory_order_relaxed);
            }
        }
    }
    const auto recorded = diagnose ? now() : start;
    {
        Gow3Stats::WaitTimer timer{Gow3Stats::copy_threads_wait_ns};
        Gow3Copy::WaitAsync();
    }
    if (diagnose) {
        const auto end = now();
        const auto ns = [](auto d) {
            return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
        };
        // Only callers that waited (a wait of the copy threads over 2 us counts as one).
        if (waited || end - recorded > std::chrono::microseconds(2)) {
            WaitDiagnostics::HostCopies().Add(caller.file_name(), caller.line(),
                                              ns(recorded - start), ns(end - recorded));
        }
    }
}

namespace {
// gow3 (impl-deck-cpu-bundle.md; study-deck-other-threads.md 3.1, study-stage-b-per-draw.md C2):
// the recorder's lean mode, for machines with <= 8 hardware threads (the Steam Deck) or
// GOW3_RECORDER_LEAN=1 (0: never). Second toggle word 32 switches it off live.
// - No spin before sleeping (GOW3_RECORDER_SPIN_US overrides). On the Deck the recorder runs a
//   chunk faster than stage B fills the next (>= 100 us for 32 KiB at ~8.5 us a draw), so the
//   20 us spin almost never caught one: ~7.3k spins/s, ~15% of a CPU, and the pauses took issue
//   slots from the sibling hyperthread (often stage B's). The wake it would save stage B was
//   paid anyway.
// - Labels the guest polls (end-of-pipe without interrupt, end-of-shader: ~200 a frame) do not
//   wake a sleeping recorder; their chunk is queued at once and the wake comes with the next one
//   that is due anyway (32 KiB hand-off, a wait, a fence with an interrupt, a submission), at
//   the latest GOW3_RECORDER_WAKE_US (default 200; 0: every label wakes as before) after the
//   first deferred label, checked at every kick stage B makes, or when stage B runs dry.
// - GOW3_RECORDER_HANDOFF_KB: the hand-off size in lean mode (default 32, as without it).
bool RecorderLeanWanted() {
    static const bool wanted = [] {
        if (const char* env = std::getenv("GOW3_RECORDER_LEAN"); env && *env) {
            return env[0] == '1';
        }
        return Gow3Threads::Available() <= 8;
    }();
    return wanted;
}
bool RecorderLean() {
    return RecorderLeanWanted() && !Gow3Toggle2::Disabled(Gow3Toggle2::RecorderLean);
}
u64 EnvU64(const char* name, u64 fallback) {
    const char* env = std::getenv(name);
    return env && *env ? std::strtoull(env, nullptr, 10) : fallback;
}
u64 RecorderWakeDelayNs() {
    static const u64 ns = EnvU64("GOW3_RECORDER_WAKE_US", 200) * 1000;
    return ns;
}
u64 HandoffBytes(bool lean) {
    static const u64 lean_bytes =
        std::clamp<u64>(EnvU64("GOW3_RECORDER_HANDOFF_KB", 32), 4, 96) * 1024;
    return lean ? lean_bytes : 32 * 1024;
}
std::chrono::microseconds RecorderSpin(bool lean) {
    // Spin briefly before sleeping: the next chunk usually follows within microseconds, and a
    // sleeping recorder costs the GPU thread a wake-up syscall per kick. With few hardware
    // threads (Steam Deck: 8) the spin would take time from guest threads.
    static const auto normal =
        std::chrono::microseconds(Gow3Threads::Available() >= 12 ? 200 : 20);
    static const auto lean_spin = std::chrono::microseconds(EnvU64("GOW3_RECORDER_SPIN_US", 0));
    return lean ? lean_spin : normal;
}
/// Lean mode: a sleeping recorder still wakes this often, so a deferred wake nobody made (stage
/// B blocked somewhere that neither kicks nor goes through Wait()) is bounded too.
std::chrono::microseconds RecorderBackstop() {
    return std::chrono::microseconds(std::max<u64>(1000, 2 * RecorderWakeDelayNs() / 1000));
}
void CountRecorderWake(u64 start_ns) {
    // gow3 (study-stage-b-per-draw.md C1): wake-ups by thread kind, with their cost
    // ("Stage B syscalls" line, GOW3_FRAME_STATS).
    const unsigned kind = Gow3Sections::recording_thread ? 0 : 1;
    Gow3Stats::recorder_wakes[kind].fetch_add(1, std::memory_order_relaxed);
    Gow3Stats::recorder_wake_ns[kind].fetch_add(Semaphore::NowNs() - start_ns,
                                                std::memory_order_relaxed);
}
} // namespace

void Scheduler::Kick(KickKind kind) {
    if (!recording_threaded) {
        return;
    }
    recording_owner.store(std::this_thread::get_id(), std::memory_order_relaxed);
    // Callers kick where nobody holds the raw command buffer: deferral resumes.
    direct_mode = false;
    ChunkGuard guard{chunk_lock};
    const bool lean = RecorderLean();
    // gow3 (impl-deck-cpu-bundle.md): a wake a label kick left owed is due at its deadline, at
    // any Now kick, and as soon as lean mode is off. The clock is read only while one is owed.
    const u64 owed = wake_deadline_ns.load(std::memory_order_relaxed);
    const bool due = owed && (kind == KickKind::Now || !lean || Semaphore::NowNs() >= owed);
    // Batches of tens of KiB keep the queue handoff cheap relative to the work it carries.
    if (kind == KickKind::Batched && !due && full_chunks.empty() &&
        record_chunk->Size() < HandoffBytes(lean)) {
        return;
    }
    if (!due && full_chunks.empty() && record_chunk->Empty()) {
        return;
    }
    const bool lazy = kind == KickKind::Label && lean && RecorderWakeDelayNs() != 0;
    bool wake = false;
    {
        std::scoped_lock lk{recorder_mutex};
        u64 handed = full_chunks.size();
        for (auto& chunk : full_chunks) {
            recorder_queue.push_back(std::move(chunk));
        }
        if (!record_chunk->Empty()) {
            recorder_queue.push_back(std::move(record_chunk));
            ++handed;
        }
        if (!recorder_sleeping || recorder_queue.empty()) {
            // An awake recorder runs everything queued before it sleeps: nothing is owed.
            wake_deadline_ns.store(0, std::memory_order_relaxed);
        } else if (lazy && !due) {
            if (!owed) {
                wake_deadline_ns.store(Semaphore::NowNs() + RecorderWakeDelayNs(),
                                       std::memory_order_relaxed);
            }
            if (Gow3Stats::enabled) {
                Gow3Stats::recorder_lazy_kicks.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            wake = true;
            wake_deadline_ns.store(0, std::memory_order_relaxed);
        }
        queued_chunks.store(recorder_queue.size(), std::memory_order_release);
        if (Gow3Stats::enabled && handed) {
            Gow3Stats::recorder_handoffs.fetch_add(handed, std::memory_order_relaxed);
        }
    }
    full_chunks.clear();
    // A busy recorder picks the new chunks up by itself: waking it is a syscall per draw.
    if (wake) {
        const u64 start_ns = Gow3Stats::enabled ? Semaphore::NowNs() : 0;
        recorder_cv.notify_one();
        if (Gow3Stats::enabled) {
            CountRecorderWake(start_ns);
            if (due) {
                // [0] made by a kick after the deadline, [1] by a Now kick or lean mode off.
                const unsigned why = kind == KickKind::Now || !lean ? 1 : 0;
                Gow3Stats::recorder_late_wakes[why].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    if (!record_chunk) {
        record_chunk = AcquireChunk();
    }
}

void Scheduler::WakeDeferredRecorderSlow() {
    bool wake;
    {
        std::scoped_lock lk{recorder_mutex};
        wake = recorder_sleeping && !recorder_queue.empty() &&
               wake_deadline_ns.load(std::memory_order_relaxed);
        wake_deadline_ns.store(0, std::memory_order_relaxed);
    }
    if (wake) {
        const u64 start_ns = Gow3Stats::enabled ? Semaphore::NowNs() : 0;
        recorder_cv.notify_one();
        if (Gow3Stats::enabled) {
            CountRecorderWake(start_ns);
            Gow3Stats::recorder_late_wakes[1].fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void Scheduler::SyncRecording() {
    if (!recording_threaded) {
        return;
    }
    KickRecording(true);
    Gow3Stats::WaitTimer timer{Gow3Stats::sync_recording_ns};
    // gow3 #4 (gow3_stage_b_wait.h): stage B with GOW3_STAGE_B_SPIN=1 watches the recorder
    // (lock-free hints) before it takes the lock; the check under the lock decides as before.
    Gow3StageBWait::Spin([this] {
        return queued_chunks.load(std::memory_order_acquire) == 0 &&
               !recorder_running.load(std::memory_order_acquire);
    });
    std::unique_lock lk{recorder_mutex};
    if (recorder_queue.empty() && !recorder_busy) {
        return;
    }
    Gow3StageBWait::SleepTimer sleep{Gow3StageBWait::Recorder};
    recorder_idle_cv.wait(lk, [this] { return recorder_queue.empty() && !recorder_busy; });
}

void Scheduler::RecorderThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("gow3:VkRecorder");
    Gow3Sampler::SampleCurrentThread("VkRecorder");
    while (true) {
        const bool lean = RecorderLean();
        if (Gow3Stats::recorder_lean.load(std::memory_order_relaxed) != int(lean)) {
            Gow3Stats::recorder_lean.store(int(lean), std::memory_order_relaxed);
        }
        const auto spin_time = RecorderSpin(lean);
        if (spin_time.count() > 0 && queued_chunks.load(std::memory_order_acquire) == 0) {
            const auto spin_start = std::chrono::steady_clock::now();
            const auto spin_until = spin_start + spin_time;
            for (u32 spins = 1; queued_chunks.load(std::memory_order_acquire) == 0; ++spins) {
                __builtin_ia32_pause();
                // The clock is read every 256 pauses, not per iteration.
                if (!(spins & 255) &&
                    (stoken.stop_requested() || std::chrono::steady_clock::now() >= spin_until)) {
                    break;
                }
            }
            if (Gow3Stats::enabled) {
                // gow3 (impl-deck-cpu-bundle.md): spin time and catches ("Recorder" line).
                Gow3Stats::recorder_spins.fetch_add(1, std::memory_order_relaxed);
                Gow3Stats::recorder_spin_ns.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - spin_start)
                        .count(),
                    std::memory_order_relaxed);
                if (queued_chunks.load(std::memory_order_relaxed) != 0) {
                    Gow3Stats::recorder_spin_catches.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        std::unique_ptr<RecordChunk> chunk;
        {
            std::unique_lock lk{recorder_mutex};
            const auto ready = [this] { return !recorder_queue.empty(); };
            if (Gow3Stats::enabled && recorder_queue.empty()) {
                Gow3Stats::recorder_sleeps.fetch_add(1, std::memory_order_relaxed);
            }
            recorder_sleeping = true;
            if (lean) {
                // gow3 (impl-deck-cpu-bundle.md): timed, so a wake a label kick left owed is
                // made here at the latest. Work found while one was still owed: the backstop
                // made it.
                while (!recorder_cv.wait_for(lk, stoken, RecorderBackstop(), ready) &&
                       !stoken.stop_requested()) {
                }
                if (Gow3Stats::enabled && !recorder_queue.empty() &&
                    wake_deadline_ns.load(std::memory_order_relaxed)) {
                    Gow3Stats::recorder_backstop_wakes.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                recorder_cv.wait(lk, stoken, ready);
            }
            recorder_sleeping = false;
            // Awake: everything queued runs before this thread sleeps again.
            wake_deadline_ns.store(0, std::memory_order_relaxed);
            if (recorder_queue.empty()) {
                return; // stop requested
            }
            chunk = std::move(recorder_queue.front());
            recorder_queue.pop_front();
            recorder_running.store(true, std::memory_order_relaxed);
            queued_chunks.store(recorder_queue.size(), std::memory_order_release);
            recorder_busy = true;
        }
        // current_cmdbuf only changes after SyncRecording(), which waits for this thread.
        chunk->Execute(current_cmdbuf);
        {
            std::scoped_lock lk{recorder_mutex};
            free_chunks.push_back(std::move(chunk));
            recorder_busy = false;
            recorder_running.store(false, std::memory_order_release);
            if (recorder_queue.empty()) {
                recorder_idle_cv.notify_all();
            }
        }
    }
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish(std::source_location caller) {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    const bool diagnose = WaitDiagnostics::Enabled();
    const auto start = diagnose ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    SubmitInfo info{};
    SubmitExecution(info);
    const u64 submit_ns = diagnose ? std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count() : 0;
    Wait(presubmit_tick, caller, submit_ns);
}

void Scheduler::WaitSubmitted(std::source_location caller) {
    const u64 tick = CurrentTick();
    if (tick > 1) {
        Wait(tick - 1, caller);
    }
}

namespace {
// gow3 GOW3_LONG_WAITS=1: GPU waits of 1 ms or more by call site and thread, every 5 s. Only
// long waits do any work, so frames keep their timing: GOW3_PERF_DIAG's wait profile hid the
// view-switch spikes it was meant to find (t-hzb5).
struct LongWaits {
    struct Site {
        u64 count = 0, total_ns = 0, max_ns = 0;
    };
    static bool Enabled() {
        static const bool enabled = [] {
            const char* env = std::getenv("GOW3_LONG_WAITS");
            return env && env[0] == '1';
        }();
        return enabled;
    }
    static const std::string& ThreadName() {
        thread_local const std::string name = Common::GetCurrentThreadName();
        return name;
    }
    static void Note(const std::source_location& caller, u64 ns) {
        static std::mutex mutex;
        static std::map<std::string, Site> sites;
        static auto last = std::chrono::steady_clock::now();
        std::string_view file{caller.file_name()};
        file = file.substr(file.find_last_of("/\\") + 1);
        char key[256];
        std::snprintf(key, sizeof(key), "%.*s:%u (%s)", int(file.size()), file.data(), caller.line(),
                      ThreadName().c_str());
        std::scoped_lock lock{mutex};
        auto& site = sites[key];
        ++site.count;
        site.total_ns += ns;
        site.max_ns = std::max(site.max_ns, ns);
        const auto now = std::chrono::steady_clock::now();
        if (now - last < std::chrono::seconds(5)) return;
        last = now;
        std::string out;
        for (const auto& [name, s] : sites) {
            Gow3AsyncLog::Append(out, "Long GPU wait (5 s): %s; %llu waits, %.1f ms, max %.1f ms\n",
                                 name.c_str(), (unsigned long long)s.count, s.total_ns / 1e6,
                                 s.max_ns / 1e6);
        }
        Gow3AsyncLog::Write(std::move(out));
        sites.clear();
    }
};
} // namespace

void Scheduler::Wait(u64 tick, std::source_location caller, u64 submit_ns) {
    const bool diagnose = WaitDiagnostics::Enabled();
    auto start = diagnose ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // gow3 (impl-deck-cpu-bundle.md): labels queued for a sleeping recorder are not left
    // waiting behind a GPU wait.
    WakeDeferredRecorder();
    if (tick >= work_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    Gow3Stats::WaitTimer timer{Gow3Stats::tick_wait_ns};
    const auto submitted = diagnose ? std::chrono::steady_clock::now() : start;
    work_semaphore.Wait(tick);
    if (LongWaits::Enabled()) {
        const u64 ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - timer.start).count();
        if (ns >= 1'000'000) LongWaits::Note(caller, ns);
    }
    if (diagnose) {
        const auto completed = std::chrono::steady_clock::now();
        WaitDiagnostics::Get().Add(caller.file_name(), caller.line(),
            submit_ns + std::chrono::duration_cast<std::chrono::nanoseconds>(submitted - start).count(),
            std::chrono::duration_cast<std::chrono::nanoseconds>(completed - submitted).count());
    }
}

void Scheduler::PopPendingOperations() {
    if (num_pending_ops.load(std::memory_order_acquire) == 0) {
        return; // every draw comes here
    }
    std::unique_lock lk(pending_ops_mutex);
    // gow3: this runs on every draw and dispatch. Querying the timeline semaphore is an
    // ioctl, so it is skipped when nothing waits and done once per 32 calls (~0.3 ms; reading
    // the clock per draw instead was itself a hot spot).
    if (pending_ops.empty()) {
        return;
    }
    if (!work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        if ((++pending_polls & 31) != 0 && !Gow3Toggle::Disabled(Gow3Toggle::PendingPollLimit)) {
            return;
        }
        work_semaphore.Refresh();
    }
    while (!pending_ops.empty() && work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
        num_pending_ops.fetch_sub(1, std::memory_order_release);
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    current_cmdbuf = command_pool.Commit();
    Check(current_cmdbuf.begin(begin_info));

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    std::scoped_lock lk{submit_mutex};
    const u64 signal_value = work_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    if (on_submit) {
        on_submit(info);
    }

    EndRendering();
    if (auto* profiler = GpuProfiler::Get(); profiler && profiler->Records(this)) {
        // Until the next submission's first timestamp: mostly the GPU waiting for it.
        profiler->Mark(0x5B317ull, [] { return std::string{"(between submissions: GPU idle)"}; });
    }
    SyncRecording();
    if (host_access_barrier) {
        // gow3 U2: see SetHostAccessBarrier. Recorded here, after every recorded command and
        // outside any render pass (ended above), while the recording thread is idle.
        const vk::MemoryBarrier2 to_host{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead | vk::AccessFlagBits2::eHostWrite,
        };
        current_cmdbuf.pipelineBarrier2(
            vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &to_host});
    }
    // Guest memory copies into staging read by this submission (copy threads).
    WaitHostCopies();
    Check(current_cmdbuf.end());

    const vk::Semaphore timeline = work_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    static constexpr std::array<vk::PipelineStageFlags, 2> wait_stage_masks = {
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &current_cmdbuf,
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    {
        auto& entry = submit_times[signal_value % submit_times.size()];
        entry.ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count(),
                       std::memory_order_relaxed);
        entry.tick.store(signal_value, std::memory_order_release);
    }
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    if (count_submits && Gow3Stats::enabled) {
        Gow3GpuFences::counters.submits.fetch_add(1, std::memory_order_relaxed);
    }

    work_semaphore.Refresh();
    AllocateWorkerCommandBuffers();

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        work_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    // A null command buffer only updates the dirty flags, exactly as recording would.
    CommitWith(instance.IsDepthBoundsSupported(), instance.IsDynamicColorWriteMaskSupported(),
               instance.IsAttachmentFeedbackLoopLayoutSupported(), [&](auto&& command) {
                   if (cmdbuf) {
                       command(cmdbuf);
                   }
               });
}

} // namespace Vulkan
