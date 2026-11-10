// gow3: optimizations that can be switched off while the game runs (GOW3_TOGGLE_FILE,
// see runtime_memory.c), to find which one changes rendering without restarting.
#pragma once
#include <atomic>
#include <csetjmp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#ifdef _WIN32
// gow3 (Windows): no signals. The loader's exception handler resumes the faulting thread
// in gow3_longjmp (src/compat_win.c: no unwinding through the faulting frames).
typedef unsigned long long sigjmp_buf[32];
extern "C" __attribute__((returns_twice)) int gow3_setjmp(sigjmp_buf buffer);
#define sigsetjmp(buffer, save) gow3_setjmp(buffer)
#endif

extern "C" std::uint64_t runtime_disabled_optimizations;
/// The toggle file's optional second number (Gow3Toggle2).
extern "C" std::uint64_t runtime_disabled_optimizations2;
/// The toggle file's optional third word: a frame cap in FPS replacing the menu's limit live (0: none).
extern "C" std::uint64_t runtime_toggle_frame_cap;
/// scePthreadGetthreadid calls and kernel lookups, with GOW3_FRAME_STATS (runtime_extra.c).
extern "C" std::uint64_t runtime_thread_id_calls, runtime_thread_id_lookups;
/// Recovery point for speculative guest memory reads on this thread (runtime_memory.c).
extern "C" __thread sigjmp_buf* runtime_fault_recover;

namespace Gow3Toggle {
enum : std::uint64_t {
    RegionCache = 1,
    FetchShaderCache = 2,
    PageTrackingEarlyExit = 4,
    PendingPollLimit = 8,
    ThreadedRecording = 16,
    ImageDescCache = 32,
    LockFreeUploadCheck = 64,
    FindImageCache = 128,
    DeferredUploads = 256,
    AccessMemo = 512,
    TextureBindingMemo = 1024,
    CoarseReadTracking = 2048,
    PreparedResources = 4096,
    DrawPreparation = 8192,
    DeferredStreamCopies = 16384,
    HotPages = 32768,
    FaultWindow = 65536,
    ParallelCopies = 131072,
    AsyncFences = 262144,
    PoolSmallCopies = 524288,
    RecordPrefetch = 1ull << 32,
    TextureViewMemo = 1ull << 33,
    TextureBindHelper = 1ull << 34,
    EarlyDrawInputs = 1ull << 35,
    ConstantRing = 1ull << 36,
    DrawPipeline = 1ull << 37,
    PipelinedTasks = 1ull << 38,
    PendingFenceWaits = 1ull << 39,
    PipelinedDispatch = 1ull << 40,
    RecorderFences = 1ull << 41,
    PipelinedMemoryWrites = 1ull << 42,
    MultiCopyShader = 1ull << 43,
    RenderStateMemo = 1ull << 44,
    TextureSetMemo = 1ull << 45,
    PipelinedIndirectDraws = 1ull << 46,
    SceneAttachmentsOnly = 1ull << 47,
    SampleSceneProxies = 1ull << 48,
    OrderedGuestWrites = 1ull << 49,
    SceneHalfRes = 1ull << 50, ///< live scaling also reduces the 960x540 post targets
    UpdateImageFastPath = 1u << 30,
    /// Image transitions wait for the reading shader stages, not all commands (N-4); the file
    /// value 2147483648 switches back to all commands live (also GOW3_NARROW_IMAGE_STAGES=0).
    NarrowImageStages = 1ull << 31,
    // TAA A/B in one run: optional techniques, off by default (no measured gain, 2026-10-02).
    TaaTonemapBlend = 1ull << 51,
    TaaClip = 1ull << 52,
    TaaVariance = 1ull << 53,
    TaaFilter = 1ull << 54,
    // On by default (bit set: off): history of a thin feature this jitter phase missed is kept
    // when nothing moves. Static-camera flicker of railings/window bars p99.9 -45% (2026-10-03).
    TaaKeepNearerHistory = 1ull << 55,
    UnmapDiscard = 1ull << 56, ///< guest unmaps drop GPU-written buffer data, no download
    PreciseCpuWrites = 1ull << 63, ///< guest libc writes into shader-written memory: no download
    SceneMipBias = 1ull << 57, ///< negative LOD bias of G-buffer samplers at reduced scene sizes
    VolumeSliceViews = 1ull << 58, ///< volumes rendered slice by slice through 2D array views
    StencilPlaneWrites = 1ull << 59, ///< shader writes to a stencil plane copied into the image
    IdleEopSkip = 1ull << 60, ///< end-of-pipe host copy wait: idle skip (P2) / targeted (W2) off
    UndefinedTargets = 1ull << 61, ///< targets made where an unmap freed images skip the upload
    TargetedHostCopyWaits = 1ull << 62, ///< WRITE_DATA waits only for copies reading its range
    // Bits 20-29 are used as raw debug toggles by the camera/object motion and the upscaler.
};
inline bool Disabled(std::uint64_t bit) {
    return (__atomic_load_n(&runtime_disabled_optimizations, __ATOMIC_RELAXED) & bit) != 0;
}
} // namespace Gow3Toggle

/// The second word of toggles: the toggle file's second number (all 64 bits of the first are taken).
namespace Gow3Toggle2 {
enum : std::uint64_t {
    BulkDescriptorCopy = 1, ///< a descriptor push copies its writes and info arrays in one piece each
    SamplerMemo = 2,        ///< sampler handles reused within a GC period (Rasterizer::BindSamplers)
    RingReuse = 4,          ///< constant ring copies reused (perf_ring_reuse; switch off live)
    /// U2: buffer uploads and precise CPU writes go straight into the mapped arena when the GPU
    /// is done with the blocks (GOW3_ARENA_HOST_VISIBLE at start; set: staging copies again).
    ArenaDirectWrites = 8,
    /// Stage B's end-of-pipe readback retirement asks the GPU's timeline semaphore at most once
    /// per GOW3_SEMAPHORE_POLL_US (Scheduler::IsFreePolled; set: a query at every event again).
    SemaphorePollInterval = 16,
    /// The Vulkan recorder's lean mode on small machines (impl-deck-cpu-bundle.md; <= 8 hardware
    /// threads or GOW3_RECORDER_LEAN=1): it sleeps without spinning, and end-of-pipe (no
    /// interrupt) / end-of-shader labels hand their chunk over without waking a sleeping
    /// recorder, which is woken within GOW3_RECORDER_WAKE_US. Set: 20 us spin, a wake per label.
    RecorderLean = 32,
    // 64: reserved (particle-sim skip, impl-draw-reduction.md).
    /// C5 (impl-stage-b-bundle-2.md): the buffer cache's ResolveReadbacks calls skip
    /// readback_mutex when the range is on no page of a pending copy (TextureCache::
    /// ReadbackSpansMiss). Set: the lock and the queue scan every time.
    ReadbackPrecheck = 128,
    /// D4 (runtime_memory.c, Linux): pool commits / decommits make their host mmap outside the
    /// region table lock. Set: under the lock again. Read by the runtime, listed here.
    PoolMapsLocked = 256,
    /// D5 (runtime_memory.c): the region table lock is tried for up to 1000 pauses before a
    /// thread blocks (Linux), and an empty GPU hook queue is seen without it. Set: as before.
    TableLockPlain = 512,
    /// #4 (gow3_stage_b_wait.h): with GOW3_STAGE_B_SPIN=1 stage B spins up to
    /// GOW3_STAGE_B_SPIN_US before it blocks. Set: blocks as before.
    StageBSpin = 1024,
    /// The texture-set memo in its compact layout (impl-texset-index.md, gow3_texset_table.h):
    /// the same slots and decisions, a 64-byte index entry per slot and the images apart.
    /// Set: the table as before (32,768 slots of ~1.55 KiB, made on its first use).
    TextureSetLegacy = 2048,
    /// Stage A's hybrid drain wait (gow3_drain_wait.h, impl-cpu-waste.md): with
    /// GOW3_DRAIN_HYBRID=1 DrawPipe::Drain pauses up to GOW3_DRAIN_SPIN_US, then sleeps on a
    /// futex that stage B signals near the end. Set: today's pauses and sched_yield again.
    DrainHybridOff = 4096,
    /// The single GPU particle pass (perf_particle_single_pass, src/runtime_extra.c
    /// ue4_particle_pass_poll): set, r.GPUParticle.FixDeltaSeconds goes back to the game's fixed
    /// 1/30 s step live, whatever the in-game menu says (the menu shows it as forced and keeps
    /// its choice); clear, the menu's / ini's choice again. Only a change of the effective value
    /// sets the cvar ("Particle single pass: on|off (live...)" in the log). Read by the runtime.
    ParticleFixedStep = 8192,
    /// The GPU particle flip guard (src/particle_hook.h, runtime_extra.c), with
    /// GOW3_PARTICLE_FLIP_GUARD=1 at start (opt-in): one state flip per main-phase simulation.
    /// Set: the game's flips as they come, live (the hooks keep counting: "GPU particle flips"
    /// in the log). Read by the runtime.
    ParticleFlipGuardOff = 16384,
    /// #8 (gow3_linux_sched.cpp, Linux), with GOW3_THREAD_NICE at start (opt-in): stage B and the
    /// vblank thread at the start's nice. Set: both back at nice 0, live; clear: the start's nice
    /// again (setpriority or rtkit, from a helper thread). "Thread priority ... (live)" in the log.
    ThreadNiceOff = 32768,
    /// R2 (impl-stage-b-per-draw-2.md): the buffer cache's residency check (EnsureResident,
    /// every ObtainBuffer outside the stream buffer) answers from a bitmap of resident blocks
    /// (gow3_resident_blocks.h; kept either way). Set: the interval search every time.
    ResidencyBitmap = 65536,
    /// T1 (impl-stage-b-per-draw-2.md): BindTextures for a stage without images resets its
    /// per-call state and binds the samplers, without the slow path's set-up (the same effects).
    /// Set: through the slow path again.
    NoImageFastPath = 131072,
    /// T2 (impl-stage-b-per-draw-2.md), with GOW3_TEXSET_WIDE=1 at start (opt-in): the
    /// texture-set memo also keeps sets of 17-32 images (a table of their own) and the sets of
    /// stages without a prepared draw (their T# hashes computed at the bind). Set: off, live.
    TextureSetWide = 262144,
    /// R1 (impl-stage-b-per-draw-2.md; src/runtime_memory.c), with GOW3_REGION_INDEX=1 at start
    /// (opt-in): the region cache in front of runtime_memory_clamp / _region also has an index
    /// of 1,024 entries by address page per thread, filled by the lock-free searches too. Set:
    /// the 8 entries filled on the locked path only, as before (live). Read by the runtime.
    RegionIndex = 524288,
    /// A1 (impl-stage-a-decode.md; src/runtime_memory.c, Linux), with GOW3_BACKING_COPY=1 at
    /// start (opt-in): CopySparseMemory (stage A's constant ring copies, stream copies, uploads)
    /// reads the backing view without the region table lock, checked by the table generation.
    /// Set: the region lookup and the locked read, as before (live). Read by the runtime.
    BackingCopy = 1048576,
    /// A2 (impl-stage-a-decode.md; vk_rasterizer.cpp, gow3_stage_a.h), with GOW3_PENDING_FILTER=1
    /// at start (opt-in): stage A's pending-write check answers "no overlap" from a page filter
    /// of the pending writes before scanning them (the same answers). Set: the scan every time.
    PendingWriteFilter = 2097152,
    /// Former A3 live bit, no longer read: A3 (GOW3_REG_HASH_PAIRED=1) is restart-only, because a
    /// live flip stopped draw preparation for the rest of the launch (liverpool.cpp
    /// PairedRegisterHash). Kept reserved so old toggle states (7340032) mean A1 + A2 only.
    PairedRegisterHash = 4194304,
    /// With GOW3_DRAIN_HYBRID=1 and GOW3_DRAIN_WAKE_AHEAD at start (gow3_drain_wait.h): stage B
    /// wakes a sleeping stage A the default 2 packets before the end again, not the start's count
    /// (a same-launch A/B of the count). Set: 2 (live).
    DrainWakeAheadDefault = 8388608,
    /// With GOW3_SIGNAL_SLEEP=1 at start (gow3_signal_wait.h): the recording thread's waits for
    /// the recorder (deferred signals, host copies) spin briefly, then sleep. Set: sched_yield
    /// until done, as before (live).
    SignalSleepOff = 16777216,
};
inline bool Disabled(std::uint64_t bit) {
    return (__atomic_load_n(&runtime_disabled_optimizations2, __ATOMIC_RELAXED) & bit) != 0;
}
} // namespace Gow3Toggle2

namespace Gow3Stats {
/// Guest writes caught by page protection, and pages currently left unprotected as hot.
inline std::atomic<std::uint64_t> tracker_faults{0};
inline std::atomic<std::int64_t> hot_pages{0};
/// Stall diagnostics (GOW3_FRAME_STATS): per-frame deltas printed for frames over 40 ms.
inline std::atomic<std::uint64_t> images_registered{0};
inline std::atomic<std::uint64_t> image_upload_bytes{0};
inline std::atomic<std::uint64_t> buffer_upload_bytes{0};
/// Small read-only buffers (BufferCache::ObtainBuffer, GOW3_STREAM_CLEAN): bound from the arena
/// (clean), uploaded once outside a render pass (promoted), copied into the stream buffer, and
/// promoted ranges found CPU-written again (dynamic: streamed from then on).
inline std::atomic<std::uint64_t> small_arena_binds{0}, small_promotions{0}, small_stream_copies{0},
    small_dynamic{0};
/// Bytes the CPU writes into GPU-visible mapped memory ("Mapped writes" line): the stream buffer
/// and the constant ring live in VRAM written through the BAR (PCIe writes); staging uploads are
/// system memory the GPU copies from (PCIe reads), staging downloads what it copies back.
inline std::atomic<std::uint64_t> stream_buffer_bytes{0}, constant_ring_bytes{0},
    staging_upload_bytes{0}, staging_download_bytes{0};
/// Constant bytes bound from an earlier ring copy instead of copied again (GOW3_RING_REUSE).
inline std::atomic<std::uint64_t> constant_ring_reused_bytes{0};
/// U2 ("Arena writes" line; buffer_cache.cpp): buffer uploads (C1) and precise CPU writes (C2)
/// written straight into the mapped arena (calls, bytes) or through staging and a GPU copy; why
/// a write took the staging path; copy commands and render pass ends the direct writes avoided;
/// render pass ends the staged buffer copies caused; barrier batches recorded (all causes).
/// arena_mapped: the arena memory is host-visible and mapped (GOW3_ARENA_HOST_VISIBLE).
inline std::atomic<bool> arena_mapped{false};
enum ArenaFallback : unsigned {
    ArenaToggledOff,    ///< toggle second word 8 set
    ArenaGpuBusy,       ///< work the GPU has not finished (draws, dispatches, DMA) uses the blocks
    ArenaStagedPending, ///< a staged copy into the blocks is still queued (ordering; stats only)
    ArenaNotMapped,     ///< a block's memory is not mapped
    ArenaOffThread,     ///< asked off the draw recording thread while it has draws queued
    ArenaFallbacks
};
inline std::atomic<std::uint64_t> arena_direct_calls{0}, arena_direct_bytes{0},
    arena_staged_calls{0}, arena_staged_bytes{0}, arena_copies_saved{0}, arena_pass_ends_saved{0},
    arena_staged_pass_ends{0}, barrier_flushes{0};
inline std::atomic<std::uint64_t> arena_fallbacks[ArenaFallbacks]{};
/// Stage B binds ("Binds per frame" line, GOW3_FRAME_STATS): guest-draw pipeline binds and the
/// ones with the previous guest draw's handle; descriptor set updates (updateDescriptorSets, a
/// device call: more bindings than push descriptors allow) and their time (ns). Identical
/// descriptor pushes were counted once (t-base1: 0; a byte hash per push cost ~0.4 us a draw).
inline std::atomic<std::uint64_t> pipeline_binds{0}, pipeline_bind_repeats{0},
    descriptor_set_updates{0}, t_descriptor_set_updates{0};
/// Descriptor pushes on the deferred path and their writes (study 2026-10-29 item 2: one
/// RecordData() per write plus one for the writes).
inline std::atomic<std::uint64_t> descriptor_pushes{0}, descriptor_writes{0};
/// Stage B's system calls ("Stage B syscalls" line; study-stage-b-per-draw.md C1): timeline
/// semaphore queries (Semaphore::Refresh: an ioctl on Linux) and recorder wake-ups
/// (KickRecording: a futex wake), [0] on the draw recording thread, [1] on other threads, with
/// their time (ns); the queries the poll interval skipped (Scheduler::IsFreePolled).
inline std::atomic<std::uint64_t> semaphore_queries[2]{}, semaphore_query_ns[2]{},
    semaphore_polls_skipped{0}, recorder_wakes[2]{}, recorder_wake_ns[2]{};
/// The Vulkan recorder ("Recorder" line; impl-deck-cpu-bundle.md): chunks handed over; label
/// kicks that left a sleeping recorder asleep (lean mode) and the wakes made for them later:
/// [0] a kick after the deadline, [1] stage B ran dry, a GPU wait or a forced kick; timed
/// sleeps that ended with work queued (the backstop); sleeps; spin time (ns), spins, and the
/// spins that found work.
inline std::atomic<std::uint64_t> recorder_handoffs{0}, recorder_lazy_kicks{0},
    recorder_late_wakes[2]{}, recorder_backstop_wakes{0}, recorder_sleeps{0}, recorder_spin_ns{0},
    recorder_spins{0}, recorder_spin_catches{0};
/// C5 ("Readback pre-check" line): the buffer cache's ResolveReadbacks calls while copies were
/// pending, [0] answered without the lock (no pending copy on the range's pages), [1] locked.
inline std::atomic<std::uint64_t> readback_prechecks[2]{};
/// Stage B per-draw cuts ("Stage B cuts per frame" line; impl-stage-b-per-draw-2.md), written by
/// the thread that owns the caches (one writer: Bump, no locked add). R2: residency checks
/// answered by the bitmap [0] or by the interval search [1].
inline std::atomic<std::uint64_t> residency_checks[2]{};
/// T1: texture binds of stages without images that took the short path.
inline std::atomic<std::uint64_t> no_image_binds{0};
inline void Bump(std::atomic<std::uint64_t>& counter) {
    counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}
/// Whether the recorder runs in lean mode (-1: not started).
inline std::atomic<int> recorder_lean{-1};
inline std::atomic<int> gpu_thread_clock{-1}; ///< clockid_t of the GPU command thread
inline std::atomic<unsigned long> gpu_thread_id{0}; ///< Windows thread id of the GPU command thread
inline std::atomic<std::uint64_t> draws{0}, dispatches{0}, submissions{0};
/// Occlusion query counter dumps (ZPASS_DONE; a query is two), render passes ended, and the
/// draw recording thread's busy TSC cycles (stage B), for the per-frame diagnostics.
inline std::atomic<std::uint64_t> occlusion_dumps{0}, render_pass_ends{0}, stage_b_busy_cycles{0};
/// Submissions made early for occlusion queries (O3).
inline std::atomic<std::uint64_t> occlusion_submits{0};
/// Synchronous GPU -> CPU buffer downloads (BufferCache::DownloadMemory, a full GPU wait each),
/// by what asked for them: 0 guest read fault, 1 guest write fault, 2 invalidation (runtime hook,
/// player frames), 3 guest unmap; and unmaps that dropped the data instead (UnmapDiscard).
enum DownloadReason : unsigned { ReadFault, WriteFault, Invalidation, Unmap, DownloadReasons };
inline thread_local unsigned download_reason = Invalidation;
/// The guest instruction of the write fault being handled on this thread (download sites).
inline thread_local const void* fault_rip = nullptr;
inline std::atomic<std::uint64_t> download_count[DownloadReasons], download_bytes[DownloadReasons],
    download_ns[DownloadReasons], unmap_discards{0};
/// Guest libc writes into shader-written memory taken without a download (PreciseCpuWrites).
inline std::atomic<std::uint64_t> precise_writes{0}, precise_write_bytes{0};
/// Downloads done on the readback queue (D3), of all counted by reason.
inline std::atomic<std::uint64_t> fast_downloads{0};
/// Write faults on GPU-written pages taken with twins instead of a download (no wait).
inline std::atomic<std::uint64_t> twin_faults{0};
/// Write faults on a page whose GPU bytes were finished, downloaded alone (GOW3_FAULT_PAGE_DOWNLOAD).
inline std::atomic<std::uint64_t> page_downloads{0};
/// Write faults on a page with no GPU-written bytes (no download), and the ones left to the
/// window download: no write tick known, or too many bytes still in flight for a twin.
inline std::atomic<std::uint64_t> fault_no_gpu_bytes{0}, fault_unknown_tick{0}, fault_too_big_for_twin{0};
/// Sets the reason for the downloads the current thread asks for, restores it after.
struct DownloadReasonScope {
    unsigned saved = download_reason;
    explicit DownloadReasonScope(unsigned reason) { download_reason = reason; }
    ~DownloadReasonScope() { download_reason = saved; }
};
/// Frames the GPU command thread has started (display pass), for per-frame diagnostics.
inline std::atomic<std::uint64_t> gpu_frames{0};
/// Wall time spent in operations suspected of stalls (ns, all threads).
inline std::atomic<std::uint64_t> t_resident{0}, t_protect{0}, t_image_create{0}, t_refresh{0},
    t_staging{0}, t_host_wait{0}, t_copy{0}, copy_bytes{0}, t_read_faults{0}, read_faults{0},
    t_write_faults{0}, t_copy_cpu{0}, copy_sys_us{0}, copy_minflt{0}, t_gc{0}, t_readback_wait{0};
/// Diagnostics are collected only with GOW3_FRAME_STATS=1.
inline const bool enabled = [] {
    const char* env = std::getenv("GOW3_FRAME_STATS");
    return env && env[0] == '1';
}();
struct Timer {
    std::atomic<std::uint64_t>& total;
    std::chrono::steady_clock::time_point start =
        enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ~Timer() {
        if (!enabled) {
            return;
        }
        total.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count(),
                        std::memory_order_relaxed);
    }
};
/// Wall time the GPU command thread waited for guest submissions (ns).
inline std::atomic<std::uint64_t> gpu_idle_ns{0};
/// Cumulative (never reset): stage A waiting for the game to submit work (ns), and draining
/// the draw pipe for stage B (TSC cycles). For the slow-frame report.
inline std::atomic<std::uint64_t> gpu_idle_total_ns{0}, stage_a_drain_cycles{0};
/// Cumulative, with GOW3_FRAME_STATS: what a new view costs stage B (slow-frame report): texture
/// sets bound without the memo, render-state memo misses, draws without a prepared pipeline, and
/// FindImage lookups that searched the page table.
inline std::atomic<std::uint64_t> texset_misses{0}, render_memo_misses{0}, unprepared_draws{0},
    image_slow_lookups{0};
/// Of those lookups: the same key found the same image again (only the registry generation had
/// changed; study 2026-10-29 item 10).
inline std::atomic<std::uint64_t> image_generation_misses{0};
/// Unprepared draws by cause: the workers had not reached it, could not prepare it (no such
/// pipeline yet), or prepared it from other registers.
inline std::atomic<std::uint64_t> unprep_pending{0}, unprep_unavailable{0}, unprep_checksum{0};
/// Draws recorded into the reduced scene targets, and draws after the scene started.
inline std::atomic<std::uint64_t> reduced_draws{0}, scene_draws{0};
/// Wall time spent blocked in the scheduler (ns): waiting for the recording thread to drain,
/// for host copies before a submission or fence, and for GPU ticks.
inline std::atomic<std::uint64_t> sync_recording_ns{0}, host_copies_wait_ns{0}, tick_wait_ns{0},
    copy_threads_wait_ns{0}, host_copy_waits{0};
/// WRITE_DATA packets that skipped the host copy wait (no pending copy reads their range).
inline std::atomic<std::uint64_t> host_copy_waits_skipped{0};
struct WaitTimer {
    std::atomic<std::uint64_t>& total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~WaitTimer() {
        total.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start).count(),
                        std::memory_order_relaxed);
    }
};
/// GPU thread rusage, refreshed after each graphics submission.
inline std::atomic<std::uint64_t> gpu_sys_us{0}, gpu_user_us{0}, gpu_invol_switches{0},
    gpu_vol_switches{0}, gpu_minor_faults{0};
/// Protection faults (signals) taken by the GPU thread itself.
inline std::atomic<std::uint64_t> gpu_signal_faults{0};
/// Protection changes: calls and pages, those removing write access (TLB shootdowns) apart.
inline std::atomic<std::uint64_t> protect_calls{0}, protect_pages{0}, protect_revoke_calls{0},
    protect_revoke_pages{0};
} // namespace Gow3Stats
