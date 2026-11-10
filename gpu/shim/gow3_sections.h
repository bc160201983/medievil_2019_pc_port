// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 GOW3_SECTIONS=1 (diagnostics): the time the draw recording thread (stage B) spends per
// part of a packet, inclusive (nested parts count in their parent too), printed per packet with
// the draw pipe statistics. Ported from the Bloodborne port's bbport_sections.h (BB_SECTIONS).
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <x86intrin.h>

namespace Gow3Sections {
enum Id : std::uint32_t {
    Packet,          ///< a whole packet of the draw pipe
    Task,            ///< ordered tasks (fences, end of pipe, signals, DMA, flips)
    EndOfPipe,       ///< TextureCache::ProcessDownloadImages (within tasks)
    DrawRecord,
    BeforeRecord,    ///< D2's CPU-write queue, O3's submission
    PrepareRenderState,
    BindResources,
    BindBuffers,
    BindTextures,
    ObtainBuffer,
    BeginRendering,  ///< Rasterizer::BeginRendering (render state, targets)
    ResolveVertexBuffers,
    ResolveIndexBuffer,
    EmitVertexBuffers,
    FlushBarriers,
    PipelineBind,    ///< Pipeline::BindResources (push descriptors, push constants)
    UpdateDynamicState,
    Kick,            ///< ResetBindings and the hand-over to the Vulkan recorder
    DispatchRecord,
    RecordCall,      ///< Scheduler::Record (closure into the chunk)
    TexSetHit,       ///< BindTexturesFromSet: lookup, validation and emit (within textures)
    TexSetEmit,      ///< its emit loop: found marks, layout checks, image infos
    Samplers,        ///< BindSamplers
    TexSlow,         ///< BindTextures without a memoized set
    // Within ObtainBuffer (t-pan1: ObtainBuffer 10-12 ms in panning hitches, 1.9 ms otherwise):
    ObtReadbacks,    ///< TextureCache::ResolveReadbacks
    ObtRegions,      ///< the GPU- / CPU-modified region checks (memory tracker locks)
    ObtStream,       ///< stream buffer reservation and copy (may wait for the GPU)
    ObtResident,     ///< EnsureResident (residency pool, sparse binds)
    ObtSync,         ///< SynchronizeMemory (uploads)
    // Within SignalFlip (t-pan5: 4-5 ms in the 9 ms frames of the heavy area, GPU usage low):
    FlipSignals,     ///< WaitDeferredSignals: the recording thread's fence writes caught up
    FlipPresent,     ///< the flip interrupt: the guest's present (acquire, limiter, Reflex sleep)
    // The render-state spikes (study 2026-10-29: 0.5-2 ms in ~8% of heavy frames, cause unknown):
    RsTarget,        ///< FindTargetMemoized past both memos: the descriptor and FindImage
    FindImage,       ///< TextureCache::FindImage, from any caller on stage B
    CacheLock,       ///< waiting for the texture cache mutex in FindImage
    // Ordered tasks by kind (within tasks; Deck 2026-11-04: ~1.3 ms a heavy frame of tasks was
    // neither flip_present nor end of pipe). Nested sections count in these too.
    TaskEop,         ///< end-of-pipe fence (EVENT_WRITE_EOP), or its tail when signalled early
    TaskEos,         ///< end-of-shader fence (EVENT_WRITE_EOS)
    TaskWriteData,   ///< WRITE_DATA
    WriteDataWait,   ///< its wait for the host copies and the deferred fences (within task_wdata)
    TaskDma,         ///< DMA_DATA (fills and copies)
    TaskFlip,        ///< the flip (flip_signals and flip_present within)
    TaskSubmit,      ///< the frame-end step (OnSubmit on stage B) with its Flush
    SubmitFlush,     ///< that Flush (within task_submit)
    TaskRetire,      ///< decode pacing: frame retirement and GPU idle steps
    TaskSignals,     ///< GPU-ordered fences: the submission for pending signals
    TaskOcclusion,   ///< occlusion query events (ZPASS_DONE)
    // Per-draw parts (impl-stage-b-per-draw-2.md): what obt_sync and vertex are made of.
    ObtUpload,       ///< within obt_sync: the tracker's locked scan of CPU-written pages, their
                     ///< re-protection, staging and the recorded copy (calls with work to do)
    Clamp,           ///< MemoryManager::ClampRangeSize (the region table lookup; vertex, buffers)
    Count,
};
inline constexpr const char* Names[Count] = {
    "packet",   "tasks",  "end of pipe", "draw",         "before record", "render state",
    "bind",     "buffers", "textures",   "ObtainBuffer", "begin render",  "vertex",
    "index",    "emit vertex", "barriers", "descriptors", "dynamic",      "kick",
    "dispatch", "Record calls", "texset", "texset emit", "samplers", "texslow",
    "obt_readbacks", "obt_regions", "obt_stream", "obt_resident", "obt_sync",
    "flip_signals", "flip_present", "rs_target", "find_image", "cache_lock",
    "task_eop", "task_eos", "task_wdata", "wdata_wait", "task_dma", "task_flip",
    "task_submit", "submit_flush", "task_retire", "task_signals", "task_occl",
    "obt_upload", "clamp",
};
static_assert(Names[Count - 1] != nullptr, "a section without a name");
inline std::array<std::atomic<std::uint64_t>, Count> cycles{};
/// Scheduler::Record calls on the draw recording thread.
inline std::atomic<std::uint64_t> record_calls{0};
/// Ordered tasks by function: address, cycles, count (stage B only; read racily for the report).
struct TaskKind {
    std::atomic<const void*> fn{nullptr};
    std::atomic<std::uint64_t> cycles{0}, count{0};
};
inline std::array<TaskKind, 24> task_kinds{};
inline void NoteTask(const void* fn, std::uint64_t task_cycles) {
    for (auto& kind : task_kinds) {
        const void* current = kind.fn.load(std::memory_order_relaxed);
        if (!current) {
            kind.fn.store(fn, std::memory_order_relaxed);
            current = fn;
        }
        if (current == fn) {
            kind.cycles.fetch_add(task_cycles, std::memory_order_relaxed);
            kind.count.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
}

inline bool Enabled() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_SECTIONS");
        return env && env[0] == '1';
    }();
    return on;
}
/// Set on the draw recording thread (DrawPipe::Run): only its time is counted.
inline thread_local bool recording_thread = false;

struct Scope {
    explicit Scope(Id id_) : id{id_}, start{Enabled() && recording_thread ? __rdtsc() : 0} {}
    ~Scope() {
        if (start) {
            cycles[id].fetch_add(__rdtsc() - start, std::memory_order_relaxed);
        }
    }
    Id id;
    std::uint64_t start;
};
} // namespace Gow3Sections

#define GOW3_SECTION_JOIN2(a, b) a##b
#define GOW3_SECTION_JOIN(a, b) GOW3_SECTION_JOIN2(a, b)
#define GOW3_SECTION(id) Gow3Sections::Scope GOW3_SECTION_JOIN(gow3_section_scope_, __LINE__){Gow3Sections::id}
