// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <pthread.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include <time.h>
#include "gow3_copy.h"
#include "gow3_sampler.h"
#include "gow3_decode_pacing.h"
#include "gow3_fence_stats.h"
#include "gow3_gpu_fences.h"
#include "gow3_sections.h"
#include "gow3_toggles.h"
#include "gow3_stage_a.h"
#include <cstdio>
#include <cstring>
#include <boost/preprocessor/stringize.hpp>

#include "common/assert.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "gow3_write_log.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    if (Gow3DecodePacing::Enabled()) {
        // gow3 GOW3_DECODE_PACING: what this thread is at, for the pacing watchdog.
        Gow3DecodePacing::describe_context = this;
        Gow3DecodePacing::describe_gpu = [](void* context) {
            auto& self = *static_cast<Liverpool*>(context);
            std::size_t frame_ends;
            {
                std::scoped_lock lk{self.submit_mutex};
                frame_ends = self.frame_ends.size();
            }
            char line[256];
            std::snprintf(line, sizeof(line),
                          "GPU command thread: %u submissions and %u commands queued, %zu frame "
                          "ends not decoded, oldest undecoded submission %llu of %llu; draw pipe "
                          "%s",
                          self.num_submits.load(), self.num_commands.load(), frame_ends,
                          static_cast<unsigned long long>(self.OldestUndecoded()),
                          static_cast<unsigned long long>(self.submit_counter.load()),
                          !self.rasterizer             ? "-"
                          : self.rasterizer->DrawPipeIdle() ? "idle"
                                                            : "busy");
            return std::string{line};
        };
    }
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
}

void Liverpool::ProcessCommands() {
    // Process incoming commands with high priority
    while (num_commands) {
        // gow3: commands touch the caches and record into the scheduler (readbacks: DownloadMemory
        // copies and Finish()es), so the draw recording thread must be idle. Drained per command:
        // draining only when the first check saw one let a command that arrived between the two
        // checks run beside the recording thread, two threads in the scheduler (MediEvil: hang in
        // WaitHostCopies, a draw split across command buffers crashing in vkCmdDrawIndexed).
        // Same fix as the Bloodborne port's 0.3 (4d4d208).
        if (rasterizer) {
            rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonCommands);
        }
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    Gow3Sampler::SampleCurrentThread("GpuCmd");
#ifndef _WIN32
    if (clockid_t clock; pthread_getcpuclockid(pthread_self(), &clock) == 0) {
        Gow3Stats::gpu_thread_clock.store(static_cast<int>(clock));
    }
#else
    Gow3Stats::gpu_thread_id.store(GetCurrentThreadId());
#endif
    gpu_id = std::this_thread::get_id();
#if defined(__linux__) || defined(_WIN32)
    gpu_tid = gettid();
#endif

    const bool gpu_fences = Gow3GpuFences::Enabled();
    // gow3 GOW3_DECODE_PACING (gow3_decode_pacing.h): submissions numbered and frames retired as
    // under GOW3_GPU_FENCES, but in stream order on stage B instead of on the GPU.
    const bool decode_pacing = Gow3DecodePacing::Enabled();
    const bool track_frames = gpu_fences || decode_pacing;
    while (!stoken.stop_requested()) {
        // gow3 GOW3_GPU_FENCES (Bloodborne 0.7, 0fc2921 gpu/cp/command_processor.cpp WaitForWork):
        // fences wait for work not submitted yet; it goes out once this thread has had nothing
        // to do for GOW3_SIGNAL_IDLE_US (0.2 ms): the guest may be waiting for one.
        if (gpu_fences && rasterizer && rasterizer->HasUnsubmittedSignals()) {
            static const auto idle_wait = std::chrono::microseconds([] {
                const char* env = std::getenv("GOW3_SIGNAL_IDLE_US");
                return env && *env ? std::strtoll(env, nullptr, 10) : 200ll;
            }());
            std::unique_lock lk{submit_mutex};
            if (!submit_cv.wait_for(lk, idle_wait,
                                    [this] { return num_commands || num_submits || submit_done; })) {
                lk.unlock();
                Gow3GpuFences::counters.flush_decoder_idle.fetch_add(1, std::memory_order_relaxed);
                rasterizer->Flush();
            }
        }
        {
            const auto idle_start = std::chrono::steady_clock::now();
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken,
                                [this] { return num_commands || num_submits || submit_done; });
            const u64 idle_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - idle_start).count();
            Gow3Stats::gpu_idle_ns.fetch_add(idle_ns, std::memory_order_relaxed);
            Gow3Stats::gpu_idle_total_ns.fetch_add(idle_ns, std::memory_order_relaxed);
        }
        if (stoken.stop_requested()) {
            break;
        }

        VideoCore::StartCapture();

        curr_qid = -1;

        while (num_submits || num_commands) {
            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            task.resume();

            if (task.done()) {
                task.destroy();

                bool frame_pending = false;
                {
                    std::scoped_lock lock{queue.m_access};
                    queue.submits.pop();
                    if (track_frames) {
                        queue.submit_index.pop();
                    }

                    --num_submits;
                    std::scoped_lock lock2{submit_mutex};
                    frame_pending = !frame_ends.empty();
                    submit_cv.notify_all();
                }
                // gow3 GOW3_GPU_FENCES: a frame's last submission decoded (the next frame's
                // first ones may follow at once): it retires once the GPU has run it.
                // GOW3_DECODE_PACING: once stage B and its label writes have passed it.
                if (track_frames && frame_pending) {
                    SignalDecodedFrames();
                }
            }
        }

        if (submit_done) {
            VideoCore::EndCapture();
            // gow3 GOW3_ONSUBMIT_STAGE_B (impl-decode-pacing.md): on stage B in stream order
            // instead of a drain here, when on.
            if (rasterizer && !rasterizer->OnSubmitInOrder()) {
                rasterizer->OnSubmit();
                rasterizer->Flush();
            }
            submit_done = false;
        }

        if (gpu_fences && rasterizer) {
            // gow3 GOW3_GPU_FENCES (Bloodborne 0.7, 0fc2921 command_processor.cpp AfterBatch):
            // GPU idle once the GPU has run this work and the labels before it are written (the
            // signal thread runs them in order), and only if nothing was submitted meanwhile:
            // a submission plus sceGnmSubmitDone in that time would otherwise release the lock
            // for work not even decoded. Frames retire the same way (the pacing waits for them).
            SignalDecodedFrames();
            rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonSubmissionEnd);
            rasterizer->WaitDeferredSignals(); // copy-ordered ones (opt-in W3 / W3b only)
            const u64 generation = submit_counter.load(std::memory_order_acquire);
            // Urgent only under the console lock (GOW3_SUBMIT_LOCK=gpu): the decode pacing does
            // not wait for it, so its submission is left to the next one.
            rasterizer->SignalAfterGpu([this, generation] { SignalGpuIdle(generation); }, 0, 0,
                                       Gow3GpuFences::LockOnGpu());
            continue;
        }
        if (decode_pacing && !rasterizer) {
            SignalDecodedFrames(); // nothing recorded: they retire now (below: GPU idle now)
        } else if (decode_pacing) {
            // gow3 GOW3_DECODE_PACING (impl-decode-pacing.md): no drain here, so this thread can
            // decode the game's next submission (and signal its frame EOP early, with
            // GOW3_EARLY_FRAME_EOP) while stage B still records this work. The rule below (the
            // guest frees label objects once the GPU is idle, so every label is written first)
            // still holds: GPU idle is a step stage B runs in stream order and defers after the
            // host copies and the signals it deferred before (Scheduler::SignalAfterHostCopies,
            // which runs them in order), i.e. after every label of the work before it. Raised
            // only if nothing was submitted meanwhile (generation, as SignalGpuIdle under
            // GOW3_GPU_FENCES); else the next idle raises it. Frames retire the same way.
            SignalDecodedFrames();
            DeferGpuIdle();
            continue;
        }
        // gow3: the guest takes GPU idle (sceGnmSubmitDone) as its work being done and frees
        // the objects that hold its fence labels. The labels must be written before: the draw
        // recording thread and the fences it deferred to the Vulkan recording thread
        // (RecorderFences). Else a late fence write lands in freed memory (a corrupted guest
        // heap free list after minutes of play).
        // The same holds when sceGnmSubmitDone finds the GPU idle and does not block at all:
        // decoded (num_submits 0) is not done until then (work_retired).
        if (rasterizer) {
            rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonSubmissionEnd);
            rasterizer->WaitDeferredSignals();
        }
        {
            std::scoped_lock lk{submit_mutex};
            if (num_submits == 0) {
                work_retired = true;
            }
        }
        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

u64 Liverpool::OldestUndecoded() {
    u64 oldest = ~0ull;
    for (u32 q = 0; q < NumTotalQueues; ++q) {
        std::scoped_lock lock{mapped_queues[q].m_access};
        if (!mapped_queues[q].submit_index.empty()) {
            oldest = std::min(oldest, mapped_queues[q].submit_index.front());
        }
    }
    return oldest;
}

void Liverpool::SignalDecodedFrames() {
    // gow3 GOW3_GPU_FENCES (Bloodborne 0.7, 0fc2921 command_processor.cpp SignalDecodedFrames):
    // a frame whose submissions are all decoded (on every queue: they interleave) is recorded
    // once stage B drains, then submitted; it retires when the GPU has executed it.
    // The frame ends are read before the queues: a frame's submissions were all queued before
    // sceGnmSubmitDone noted its end, so the scan sees every one of them still undecoded (read
    // the other way round, a submission queued between the two looked decoded). Only this
    // thread removes ends.
    size_t known;
    {
        std::scoped_lock lk{submit_mutex};
        known = frame_ends.size();
    }
    if (known == 0) {
        return;
    }
    const u64 oldest = OldestUndecoded();
    u64 frame = 0;
    {
        std::scoped_lock lk{submit_mutex};
        for (; known && frame_ends.front().first < oldest; --known) {
            frame = frame_ends.front().second;
            frame_ends.pop_front();
        }
    }
    if (frame == 0) {
        return;
    }
    if (Gow3DecodePacing::Enabled()) {
        // gow3 GOW3_DECODE_PACING: the frame retires once stage B has recorded it and the label
        // writes it deferred ran (in order after the host copies); stage A does not wait.
        struct RetireStep {
            u64 frame, decoded_ns;
        };
        const RetireStep step{frame, Gow3DecodePacing::NowNs()};
        if (!rasterizer) {
            Gow3DecodePacing::NoteFramesRetired(step.frame, step.decoded_ns);
            return;
        }
        rasterizer->RunInOrder(
            [](Vulkan::Rasterizer& rasterizer, const u8* data) {
                GOW3_SECTION(TaskRetire);
                RetireStep step;
                std::memcpy(&step, data, sizeof(step));
                rasterizer.SignalAfterHostCopies([step] {
                    Gow3DecodePacing::NoteFramesRetired(step.frame, step.decoded_ns);
                });
            },
            &step, sizeof(step));
        return;
    }
    if (!rasterizer) {
        Gow3GpuFences::NoteFramesRetired(frame);
        return;
    }
    rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonSubmissionEnd);
    rasterizer->SignalAfterGpu([frame] { Gow3GpuFences::NoteFramesRetired(frame); });
    Gow3GpuFences::counters.flush_frame_end.fetch_add(1, std::memory_order_relaxed);
    rasterizer->Flush();
}

bool Liverpool::SignalGpuIdle(u64 generation) {
    bool idle;
    {
        std::scoped_lock lk{submit_mutex};
        idle = num_submits == 0 && submit_counter.load(std::memory_order_relaxed) == generation;
        if (idle) {
            work_retired = true;
        }
    }
    if (idle) {
        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
    return idle;
}

void Liverpool::DeferGpuIdle() {
    // gow3 GOW3_DECODE_PACING: the end of queued work. Stage B reaches this step after every
    // draw and fence handed to it before, then defers it after the host copies and the earlier
    // deferred signals; only then is the work retired (work_retired, the GPU idle interrupt).
    struct IdleStep {
        Liverpool* self;
        u64 generation, handed_ns;
    };
    const IdleStep step{this, submit_counter.load(std::memory_order_acquire),
                        Gow3DecodePacing::NowNs()};
    Gow3DecodePacing::NoteIdleDeferred();
    rasterizer->RunInOrder(
        [](Vulkan::Rasterizer& rasterizer, const u8* data) {
            GOW3_SECTION(TaskRetire);
            IdleStep step;
            std::memcpy(&step, data, sizeof(step));
            rasterizer.SignalAfterHostCopies([step] {
                Gow3DecodePacing::NoteIdleRun(step.self->SignalGpuIdle(step.generation),
                                              step.handed_ns);
            });
        },
        &step, sizeof(step));
}

namespace {
/// gow3: packets the GPU command thread handles while the draw recording thread may still
/// work on earlier draws (register state, draws it hands over, nested buffers). Every other
/// packet first waits for that thread to run dry (Rasterizer::DrainDrawPipe).
bool PipelinedOpcode(PM4ItOpcode opcode) {
    switch (opcode) {
    case PM4ItOpcode::Nop:
    case PM4ItOpcode::ContextControl:
    case PM4ItOpcode::ClearState:
    case PM4ItOpcode::SetConfigReg:
    case PM4ItOpcode::SetContextReg:
    case PM4ItOpcode::SetShReg:
    case PM4ItOpcode::SetUconfigReg:
    case PM4ItOpcode::IndexType:
    case PM4ItOpcode::NumInstances:
    case PM4ItOpcode::IndexBase:
    case PM4ItOpcode::IndexBufferSize:
    case PM4ItOpcode::SetBase:
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
    case PM4ItOpcode::IndirectBuffer:
    case PM4ItOpcode::AcquireMem:
    case PM4ItOpcode::PfpSyncMe:
    case PM4ItOpcode::IncrementDeCounter:
    case PM4ItOpcode::WaitOnCeCounter:
    case PM4ItOpcode::EventWrite:     // occlusion results only, no rasterizer state
    case PM4ItOpcode::EventWriteEop:  // runs in order on the recording thread (RunInOrder)
    case PM4ItOpcode::EventWriteEos:
    case PM4ItOpcode::WaitRegMem:     // waits for the recording thread only while unmet
    case PM4ItOpcode::DispatchDirect: // handed over like draws (PipelinedDispatch)
    case PM4ItOpcode::DrawIndirect:   // likewise (PipelinedIndirectDraws; else drains itself)
    case PM4ItOpcode::DrawIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirect:
    case PM4ItOpcode::DrawIndexIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirectCountMulti:
    case PM4ItOpcode::DispatchIndirect:
    case PM4ItOpcode::WriteData:     // run in order on the recording thread (RunInOrder)
    case PM4ItOpcode::DmaData:
        return true;
    default:
        return false;
    }
}
} // namespace

namespace {
void SignalEop(const PM4CmdEventWriteEop& eop) {
    eop.SignalFence(
        [](void* address, u64 data, u32 num_bytes) {
            auto* memory = Core::Memory::Instance();
            Gow3WriteLog::Note(reinterpret_cast<u64>(address), &data, num_bytes, Gow3WriteLog::Fence);
            if (!memory->TryWriteBacking(address, &data, num_bytes)) {
                memcpy(address, &data, num_bytes);
            }
        },
        [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
}

/// gow3 GOW3_GPU_FENCES: the end-of-pipe fence on the signal thread, after the GPU: its label as
/// one store (Bloodborne 0.7's StoreLabel, 0fc2921 gpu/cp/cp_tasks.cpp), then the interrupt.
void SignalEopAfterGpu(const PM4CmdEventWriteEop& eop) {
    eop.SignalFence(
        [](void* address, u64 data, u32 num_bytes) {
            Gow3WriteLog::Note(reinterpret_cast<u64>(address), &data, num_bytes, Gow3WriteLog::Fence);
            Core::Memory::Instance()->StoreLabel(address, data, num_bytes, true);
        },
        [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
}

// gow3: end-of-pipe/-shader events run in order with the draws (Rasterizer::RunInOrder: on
// the draw recording thread when the draw pipeline is in use).
// gow3 GOW3_EARLY_FRAME_EOP=1 (plan-linux-deck.md, study-eop-wait-site.md): UE4's submit thread
// waits each frame for the previous frame's end-of-pipe label (interrupt + 64-bit frame number)
// before it submits the next frame; signalled when the draw recording reaches it, that wait is
// the game waiting for our stage B. The label only gates submission (UE4 recycles its transient
// pool by frame count, 4 slots), so it is signalled at decode instead while stage B is fewer than
// GOW3_EARLY_FRAME_EOP_LEAD (default 2) such frames behind; else at stage B as before.
bool EarlyFrameEop() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_EARLY_FRAME_EOP");
        return env && env[0] == '1';
    }();
    return on;
}
u64 EarlyFrameEopLead() {
    static const u64 lead = [] {
        const char* env = std::getenv("GOW3_EARLY_FRAME_EOP_LEAD");
        return env && *env ? std::max<u64>(1, std::strtoull(env, nullptr, 10)) : u64{2};
    }();
    return lead;
}
bool IsFrameEop(const PM4CmdEventWriteEop& eop) {
    return eop.int_sel != InterruptSelect::None && eop.data_sel == DataSelect::Data64;
}
/// Frame EOPs decoded (stage A only) and reached by the draw recording (stage B only).
std::atomic<u64> frame_eops_decoded{0}, frame_eops_recorded{0};

/// Stage B: the rest of a frame EOP signalled at decode (its place in the stream).
void RunEarlyEopTail(Vulkan::Rasterizer& rasterizer, const u8*) {
    GOW3_SECTION(TaskEop);
    rasterizer.ProcessDownloadImages();
    frame_eops_recorded.fetch_add(1, std::memory_order_release);
}

void RunEventWriteEop(Vulkan::Rasterizer& rasterizer, const u8* data) {
    GOW3_SECTION(TaskEop);
    const auto eop = *reinterpret_cast<const PM4CmdEventWriteEop*>(data);
    rasterizer.ProcessDownloadImages();
    if (EarlyFrameEop() && IsFrameEop(eop)) {
        frame_eops_recorded.fetch_add(1, std::memory_order_release); // signalled below, as before
    }
    if (Gow3GpuFences::Enabled()) {
        // gow3 GOW3_GPU_FENCES (Bloodborne 0.7, 0fc2921 cp_tasks.cpp EventWriteEopTask): written
        // once the GPU has finished the work before it, as the hardware does, by the signal
        // thread (labels by the GPU itself need U5's guest memory chunks). No wait for the host
        // copies: the submission holding this tick waits for them before it is handed over.
        Gow3FenceStats::pending_fence_kind = eop.int_sel != InterruptSelect::None
                                                 ? Gow3FenceStats::FenceEopIrq
                                                 : Gow3FenceStats::FenceEop; // gate 2: lag ~0
        const bool label =
            eop.data_sel == DataSelect::Data32Low || eop.data_sel == DataSelect::Data64;
        rasterizer.SignalAfterGpu(
            [eop] { SignalEopAfterGpu(eop); },
            label ? reinterpret_cast<VAddr>(eop.Address<u8>()) : 0,
            eop.data_sel == DataSelect::Data64 ? eop.DataQWord() : eop.DataDWord());
        return;
    }
    // Guest memory copies on the copy threads precede the fence the guest sees.
    // GOW3_ASYNC_FENCES=1 writes it once they are done instead of waiting for them; measured
    // slower (79 vs 83 FPS): the guest waits on these fences, and later fences stall it more
    // than the wait costs.
    static const bool async_fences = [] {
        const char* env = std::getenv("GOW3_ASYNC_FENCES");
        return env && env[0] == '1';
    }();
    if (async_fences && !Gow3Toggle::Disabled(Gow3Toggle::AsyncFences)) {
        Gow3Copy::AfterCopies([eop] { SignalEop(eop); });
    } else if (Vulkan::DrawPipe::OnStageB() && !Gow3Toggle::Disabled(Gow3Toggle::RecorderFences)) {
        // The draw recording thread does not wait: the Vulkan recording thread signals the
        // fence after the copies queued before it.
        Gow3FenceStats::pending_fence_kind = eop.int_sel != InterruptSelect::None
                                                 ? Gow3FenceStats::FenceEopIrq
                                                 : Gow3FenceStats::FenceEop; // gow3 gate 2: lag
        // gow3 (impl-deck-cpu-bundle.md): a label with no interrupt is only polled by the guest
        // (UE4's frame wait is the one with an interrupt): in the recorder's lean mode it does
        // not wake a sleeping recorder by itself (bounded by GOW3_RECORDER_WAKE_US).
        rasterizer.SignalAfterHostCopies([eop] { SignalEop(eop); },
                                         eop.int_sel != InterruptSelect::None);
    } else {
        // Labels are written in order: not before the ones deferred to the recording thread.
        rasterizer.WaitHostCopies();
        rasterizer.WaitDeferredSignals();
        SignalEop(eop);
    }
}

void RunDmaData(Vulkan::Rasterizer& rasterizer, const u8* data) {
    GOW3_SECTION(TaskDma);
    const auto* dma_data = reinterpret_cast<const PM4DmaData*>(data);
    ASSERT(dma_data->command.das == 0);
    if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
        rasterizer.FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                               dma_data->data, true);
    } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
               dma_data->dst_sel == DmaDataDst::Gds) {
        rasterizer.CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                               dma_data->NumBytes(), true, false);
    } else if (dma_data->src_sel == DmaDataSrc::Data &&
               (dma_data->dst_sel == DmaDataDst::Memory ||
                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
        rasterizer.FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                               dma_data->data, false);
    } else if (dma_data->src_sel == DmaDataSrc::Gds &&
               (dma_data->dst_sel == DmaDataDst::Memory ||
                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
        rasterizer.CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                               dma_data->NumBytes(), false, true);
    } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
               (dma_data->dst_sel == DmaDataDst::Memory ||
                dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
        rasterizer.CopyBuffer(dma_data->DstAddress<VAddr>(),
                               dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                               false, false);
    } else {
        UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                        u32(dma_data->dst_sel));
    }
}

void SignalFlip(Vulkan::Rasterizer& rasterizer, const u8*) {
    GOW3_SECTION(TaskFlip);
    // gow3 GOW3_GPU_FENCES: the flip stays at recording time (as in Bloodborne 0.7); counted when
    // GPU-ordered labels before it are still pending (it reaches the guest first).
    if (Gow3GpuFences::Enabled() && rasterizer.HasPendingSignals()) {
        Gow3GpuFences::counters.overtake_flip.fetch_add(1, std::memory_order_relaxed);
    }
    // gow3 W3b (T-W3b): with the WRITE_DATA flags deferred (W3), the flip waited for the whole
    // backlog of fences the Vulkan recording thread writes (~1 ms a frame). That thread raises
    // the flip interrupt after them instead: when the wait would have ended, without holding
    // the draw recording thread. DEADLOCKS (T-W3c, boot): the guest's flip handler runs in the
    // interrupt and presents (SubmitFlip -> Presenter::PrepareFrame -> Scheduler::CommandBuffer
    // -> SyncRecording), waiting for the recording thread it runs on. Kept only as a reference,
    // GOW3_FLIP_DEFERRED=1 (do not use); the present needs the recording caught up anyway.
    static const bool deferred = [] {
        const char* env = std::getenv("GOW3_FLIP_DEFERRED");
        return env && env[0] == '1';
    }();
    if (deferred && Vulkan::DrawPipe::OnStageB() &&
        !Gow3Toggle::Disabled(Gow3Toggle::RecorderFences)) {
        rasterizer.SignalAfterHostCopies(
            [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip); });
        return;
    }
    {
        GOW3_SECTION(FlipSignals);
        rasterizer.WaitDeferredSignals();
    }
    GOW3_SECTION(FlipPresent);
    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
}

/// gow3 GOW3_FRAME_STATS (T-S1c: RunWriteData ~260 us twice a frame on stage B, waiting for the
/// deferred fences): what the WRITE_DATA packets write, and how long each target waited.
static void NoteWriteData(VAddr address, u32 bytes, const u32* data, u64 wait_ns) {
    struct Target {
        VAddr address = 0;
        u32 bytes = 0, first = 0, count = 0;
        u64 wait_ns = 0;
    };
    static std::array<Target, 16> targets{};
    static auto window = std::chrono::steady_clock::now();
    auto* slot = std::ranges::find(targets, address, &Target::address);
    if (slot == targets.end()) slot = std::ranges::find(targets, VAddr{0}, &Target::address);
    if (slot != targets.end()) {
        *slot = {address, bytes, bytes ? data[0] : 0, slot->count + 1, slot->wait_ns + wait_ns};
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - window < std::chrono::seconds(5)) return;
    std::printf("WriteData targets (5 s):");
    for (const auto& t : targets) {
        if (!t.address) break;
        std::printf(" %#llx %u B (first word %#x) x%u, waited %.2f ms;",
                    static_cast<unsigned long long>(t.address), t.bytes, t.first, t.count, t.wait_ns / 1e6);
    }
    std::printf("\n");
    targets = {};
    window = now;
}

void RunWriteData(Vulkan::Rasterizer& rasterizer, const u8* data) {
    GOW3_SECTION(TaskWriteData);
    const auto* header = reinterpret_cast<const PM4Header*>(data);
    const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(data);
    // gow3 W3 (T-W3): MediEvil writes two "done" flags a frame (value 1, 4 bytes, user memory at
    // 0x6d2a4c88/90); each waited ~0.45 ms on stage B for the fences deferred to the Vulkan
    // recording thread. Like those fences (E1), a small write outside direct memory (where GPU
    // resources and command-processor labels live, >= 0x10'0000'0000) is made by that thread
    // after them, in order. Opt-in (GOW3_WRITEDATA_DEFERRED=1): alone it only moves the wait to
    // the flip (T-W3b: ordered tasks 0.30 -> 0.68 us per packet).
    static const bool deferred = [] {
        const char* env = std::getenv("GOW3_WRITEDATA_DEFERRED");
        return env && env[0] == '1';
    }();
    const u32 write_bytes = (header->type3.count.Value() - 2) * sizeof(u32);
    if (deferred && Vulkan::DrawPipe::OnStageB() && write_bytes && write_bytes <= 64 &&
        write_data->Address<VAddr>() + write_bytes <= 0x10'0000'0000ull &&
        !Gow3Toggle::Disabled(Gow3Toggle::RecorderFences)) {
        std::array<u8, 64> payload{};
        std::memcpy(payload.data(), write_data->data, write_bytes);
        const VAddr address = write_data->Address<VAddr>();
        if (Gow3Stats::enabled) NoteWriteData(address, write_bytes, write_data->data, 0);
        rasterizer.SignalAfterHostCopies([address, write_bytes, payload] {
            Gow3WriteLog::Note(address, payload.data(), write_bytes, Gow3WriteLog::WriteData);
            std::memcpy(reinterpret_cast<void*>(address), payload.data(), write_bytes);
        });
        return;
    }
    // Copies deferred to the recording thread that read this range precede the write, and
    // fences deferred to it (RecorderFences) precede writes the guest sees.
    // gow3 GOW3_GPU_FENCES: not the GPU-ordered ones: waiting for the GPU here would stall stage
    // B twice a frame (Bloodborne 0.7's CPU WRITE_DATA does not wait either). Counted when such
    // labels are still pending: this write reaches the guest before them.
    if (Gow3GpuFences::Enabled() && rasterizer.HasPendingSignals()) {
        Gow3GpuFences::counters.overtake_write_data.fetch_add(1, std::memory_order_relaxed);
    }
    const auto wait_start = Gow3Stats::enabled ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
    {
        GOW3_SECTION(WriteDataWait);
        rasterizer.WaitHostCopiesFor(write_data->Address<VAddr>(),
                                     (header->type3.count.Value() - 2) * sizeof(u32));
        rasterizer.WaitDeferredSignals();
    }
    if (Gow3Stats::enabled) {
        NoteWriteData(write_data->Address<VAddr>(), (header->type3.count.Value() - 2) * sizeof(u32),
                      write_data->data,
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - wait_start).count());
    }
    Gow3WriteLog::Note(write_data->Address<u64>(), write_data->data,
                     (header->type3.count.Value() - 2) * sizeof(u32), Gow3WriteLog::WriteData);
    std::memcpy(write_data->Address<u64*>(), write_data->data,
                (header->type3.count.Value() - 2) * sizeof(u32));
}

void SignalEos(const PM4CmdEventWriteEos& event_eos) {
    event_eos.SignalFence([](void* address, u64 value, u32 num_bytes) {
        auto* memory = Core::Memory::Instance();
        Gow3WriteLog::Note(reinterpret_cast<u64>(address), &value, num_bytes, Gow3WriteLog::Fence);
        if (!memory->TryWriteBacking(address, &value, num_bytes)) {
            memcpy(address, &value, num_bytes);
        }
    });
}

void RunEventWriteEos(Vulkan::Rasterizer& rasterizer, const u8* data) {
    GOW3_SECTION(TaskEos);
    const auto& event_eos = *reinterpret_cast<const PM4CmdEventWriteEos*>(data);
    if (Gow3GpuFences::Enabled()) {
        // gow3 GOW3_GPU_FENCES (Bloodborne 0.7, 0fc2921 cp_tasks.cpp EventWriteEosTask): the end
        // of the shader work is written once the GPU has finished it.
        const auto eos = event_eos;
        rasterizer.ProcessDownloadImages();
        if (eos.command != PM4CmdEventWriteEos::Command::GdsStore) {
            Gow3FenceStats::pending_fence_kind = Gow3FenceStats::FenceEos; // gate 2: lag ~0
            rasterizer.SignalAfterGpu(
                [eos] {
                    eos.SignalFence([](void* address, u64 value, u32 num_bytes) {
                        Gow3WriteLog::Note(reinterpret_cast<u64>(address), &value, num_bytes,
                                           Gow3WriteLog::Fence);
                        Core::Memory::Instance()->StoreLabel(address, value, num_bytes, true);
                    });
                },
                reinterpret_cast<VAddr>(eos.Address()), eos.DataDWord());
            return;
        }
        // A GDS store reads the GPU's data: after a GPU wait as before, then written in order
        // with the signals before it (its tick is done, so at once).
        ASSERT(eos.size == 1);
        rasterizer.Finish();
        const u32 value = rasterizer.ReadDataFromGds(eos.gds_index);
        rasterizer.SignalAfterGpu(
            [address = eos.Address(), value] {
                Core::Memory::Instance()->StoreLabel(address, value, sizeof(u32), false);
            },
            reinterpret_cast<VAddr>(eos.Address()), value);
        return;
    }
    // gow3 E1: like the end-of-pipe fence, the draw recording thread does not wait for the host
    // copies: the Vulkan recording thread writes the label after the copies queued before it.
    // T-W2 / T-S3: with the end-of-pipe readback check targeted (W2), this synchronous wait took
    // the copies' waiting over. GOW3_EOS_DEFERRED=0: the synchronous wait as before. A GDS
    // store reads the GPU's data, so it keeps the synchronous path.
    static const bool deferred = [] {
        const char* env = std::getenv("GOW3_EOS_DEFERRED");
        return !env || env[0] != '0';
    }();
    if (deferred && event_eos.command != PM4CmdEventWriteEos::Command::GdsStore &&
        Vulkan::DrawPipe::OnStageB() && !Gow3Toggle::Disabled(Gow3Toggle::RecorderFences)) {
        rasterizer.ProcessDownloadImages();
        const auto eos = event_eos;
        Gow3FenceStats::pending_fence_kind = Gow3FenceStats::FenceEos; // gow3 gate 2: lag
        // A polled label (no interrupt): not urgent, as the end-of-pipe ones without one.
        rasterizer.SignalAfterHostCopies([eos] { SignalEos(eos); }, false);
        return;
    }
    // Copies deferred to the recording thread precede writes the guest sees.
    rasterizer.WaitHostCopies();
    rasterizer.WaitDeferredSignals();
    rasterizer.ProcessDownloadImages();
    SignalEos(event_eos);
    if (event_eos.command == PM4CmdEventWriteEos::Command::GdsStore) {
        ASSERT(event_eos.size == 1);
        rasterizer.Finish();
        const u32 value = rasterizer.ReadDataFromGds(event_eos.gds_index);
        *event_eos.Address() = value;
    }
}
/// gow3 GOW3_GPU_FENCES: a compute RELEASE_MEM on the signal thread, after the GPU: the label at
/// its address (as before, not through the backing view) as one store, then the interrupt.
void SignalReleaseAfterGpu(const PM4CmdReleaseMem& release, u32 pipe_id) {
    void* address = release.Address<void*>();
    switch (release.data_sel.Value()) {
    case DataSelect::Data32Low:
        Core::Memory::Instance()->StoreLabel(address, release.DataDWord(), sizeof(u32), false);
        break;
    case DataSelect::Data64:
        Core::Memory::Instance()->StoreLabel(address, release.DataQWord(), sizeof(u64), false);
        break;
    case DataSelect::GpuClock64:
        Core::Memory::Instance()->StoreLabel(address, GetGpuClock64(), sizeof(u64), false);
        break;
    case DataSelect::PerfCounter:
        Core::Memory::Instance()->StoreLabel(address, GetGpuPerfCounter(), sizeof(u64), false);
        break;
    default:
        break; // none, or a GDS store (recorded as a GPU copy)
    }
    if (release.int_sel.Value() != InterruptSelect::None) {
        Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
    }
}

/// gow3 GOW3_GPU_FENCES: a WAIT_REG_MEM not met (yet) on stage A. Per step: submits unsubmitted
/// signals (the value may come from a fence waiting for that work), counts, and after 256 steps
/// also yields the core (the GPU's lag is milliseconds; the Deck has 4 cores). Reports a wait of
/// 500 ms and of 10 s once. Bloodborne 0.7 gives up only in repaired command buffers or after a
/// corrupt end-of-pipe packet (neither exists here): GOW3_WAIT_REG_MEM_GIVEUP_MS (0: never)
/// passes any wait after that long; Step() then returns true.
struct GpuOrderedWait {
    VAddr address;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    u32 steps = 0;
    bool told_short = false, told_long = false;

    ~GpuOrderedWait() {
        if (!steps || !Gow3Stats::enabled) {
            return;
        }
        const u64 ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count());
        Gow3GpuFences::counters.wrm_waits.fetch_add(1, std::memory_order_relaxed);
        Gow3GpuFences::counters.wrm_wait_ns.fetch_add(ns, std::memory_order_relaxed);
        Gow3GpuFences::AtomicMax(Gow3GpuFences::counters.wrm_wait_max_ns, ns);
    }

    bool Step(Vulkan::Rasterizer& rasterizer) {
        static const u64 give_up_ms = [] {
            const char* env = std::getenv("GOW3_WAIT_REG_MEM_GIVEUP_MS");
            return env && *env ? std::strtoull(env, nullptr, 10) : 0ull;
        }();
        if (rasterizer.HasUnsubmittedSignals()) {
            Gow3GpuFences::counters.wrm_flushes.fetch_add(1, std::memory_order_relaxed);
            Gow3GpuFences::counters.flush_wait_reg_mem.fetch_add(1, std::memory_order_relaxed);
            rasterizer.Flush();
        }
        Gow3GpuFences::counters.wrm_yields.fetch_add(1, std::memory_order_relaxed);
        if (++steps > 256) {
            std::this_thread::yield();
        }
        if ((steps & 255) != 0) {
            return false;
        }
        const auto waited = std::chrono::steady_clock::now() - start;
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(waited).count();
        if ((!told_short && ms >= 500) || (!told_long && ms >= 10000)) {
            told_short = true;
            told_long = ms >= 10000;
            char line[128];
            std::snprintf(line, sizeof(line), "GPU fences: WAIT_REG_MEM on %#llx not met after %lld ms\n",
                          static_cast<unsigned long long>(address), static_cast<long long>(ms));
            const std::string out = line + rasterizer.DescribeSignals() + "  " +
                                    Gow3GpuFences::DescribePacing() + "\n";
            std::fputs(out.c_str(), stdout);
            std::fflush(stdout);
        }
        if (give_up_ms && u64(ms) >= give_up_ms) {
            std::printf("GPU fences: WAIT_REG_MEM on %#llx passed after %lld ms "
                        "(GOW3_WAIT_REG_MEM_GIVEUP_MS)\n",
                        static_cast<unsigned long long>(address), static_cast<long long>(ms));
            std::fflush(stdout);
            return true;
        }
        return false;
    }
};
} // namespace

void Liverpool::NotePendingFences(const auto& event) {
    const u64 position = rasterizer->DrawPipeHead();
    while (!pending_fences.empty() && rasterizer->DrawPipeReached(pending_fences.front().position)) {
        pending_fences.pop_front();
    }
    constexpr bool is_eop = requires { event.SignalFence([](void*, u64, u32) {}, [] {}); };
    const auto note = [&](void* address, u64 data, u32 num_bytes) {
        Gow3WriteLog::NoteIntent(reinterpret_cast<u64>(address), &data, num_bytes,
                               is_eop ? Gow3WriteLog::FenceIntent : Gow3WriteLog::EosIntent);
        pending_fences.push_back({reinterpret_cast<VAddr>(address), data, num_bytes, position});
        rasterizer->NotePendingGpuWrite(reinterpret_cast<VAddr>(address), num_bytes);
    };
    if constexpr (requires { event.SignalFence(note, [] {}); }) {
        event.SignalFence(note, [] {});
    } else {
        event.SignalFence(note);
    }
}

void Liverpool::NotePendingWrite(const PM4CmdWriteData& write_data, u32 num_bytes) {
    if (num_bytes > sizeof(u64)) {
        return; // labels are small; larger writes are not waited on
    }
    u64 data = 0;
    std::memcpy(&data, write_data.data, num_bytes);
    pending_fences.push_back(
        {write_data.Address<VAddr>(), data, num_bytes, rasterizer->DrawPipeHead()});
}

bool Liverpool::PendingFenceValue(VAddr address, u32& value) {
    for (auto it = pending_fences.rbegin(); it != pending_fences.rend(); ++it) {
        if (rasterizer->DrawPipeReached(it->position)) {
            break; // written already: memory holds it (or a later write)
        }
        if (address >= it->address && address + sizeof(u32) <= it->address + it->num_bytes) {
            u64 data = it->data;
            value = static_cast<u32>(data >> ((address - it->address) * 8));
            return true;
        }
    }
    // gow3 GOW3_GPU_FENCES (Bloodborne 0.7, 0fc2921 cp_graphics.cpp PendingFenceValue): past the
    // draw recording thread a fence waits for the GPU to finish the work before it; in stream
    // order it is met already.
    if (Gow3GpuFences::Enabled() && rasterizer) {
        u64 data = 0;
        if (rasterizer->PendingSignalValue(address, data)) {
            value = static_cast<u32>(data);
            return true;
        }
        if ((address & 7) == 4 && rasterizer->PendingSignalValue(address - 4, data)) {
            value = static_cast<u32>(data >> 32); // the high half of a 64-bit fence
            return true;
        }
    }
    return false;
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            // gow3: earlier draws on the recording thread may still read constants here.
            if (rasterizer) {
                rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonConstRam);
            }
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            memcpy(dump_const->Address<void*>(),
                   cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, header->type3.NumWords() + 1);
    }

    FIBER_EXIT;
}

// gow3: graphics-register effects of a type-3 packet. The GPU thread and the draw-preparation
// workers both go through this function, and both fold every register-writing packet into a
// running checksum: equal checksums at a draw mean equal register files.
// gow3 A3 (research/docs/impl-stage-a-decode.md; GOW3_REG_HASH_PAIRED=1 at start): two words per
// step. Every step is one multiply and one shift on a chain that depends on the previous step
// (~6 cycles on Zen 2), so pairs halve the chain over the words of a packet on stage A, the
// scanner and the four workers alike. Still a bijection of the state per step (xor, odd multiply,
// xor-shift), 64 bits of state as before; only the values differ, and every thread that compares
// them (stage A's running checksum, the prepared draws' copies) uses this function.
// Restart-only: the scanner carries its own checksum chain, seeded from stage A only at the first
// submission or a rebase (DrawPreparation::BeginSubmission), so a live switch left the two chains
// apart for good and no draw was prepared for the rest of the launch (deck-stagea-1.log,
// deck-launch15.log). The former live bit (second word 4194304) is no longer read.
bool Liverpool::PairedRegisterHash() {
    static const bool wanted = [] {
        const char* env = std::getenv("GOW3_REG_HASH_PAIRED");
        return env && env[0] == '1';
    }();
    return wanted;
}

u64 Liverpool::HashRegisterPacket(u64 checksum, const u32* words, u32 count) {
    return PairedRegisterHash() ? Gow3StageA::HashWordsPaired(checksum, words, count)
                                : Gow3StageA::HashWords(checksum, words, count);
}

void Liverpool::ApplyGraphicsRegisterPacket(Regs& regs, const PM4Header* header, u64& checksum,
                                             RegDirty* dirty) {
    const u32 count = header->type3.NumWords();
    const auto* payload = reinterpret_cast<const u32*>(header + 2);
    const auto mark = [&](u32 word, u32 words) {
        if (dirty) {
            dirty->Mark(word, words);
        }
    };
    const auto mark_field = [&](const auto& field) {
        if (dirty) {
            dirty->MarkField(regs, field);
        }
    };
    switch (header->type3.opcode) {
    case PM4ItOpcode::ClearState:
        regs.SetDefaults();
        if (dirty) {
            dirty->Clear();
            dirty->reset = true;
        }
        break;
    case PM4ItOpcode::SetConfigReg: {
        const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
        std::memcpy(&regs.reg_array[Regs::ConfigRegWordOffset + set_data->reg_offset], payload,
                    (count - 1) * sizeof(u32));
        mark(Regs::ConfigRegWordOffset + set_data->reg_offset, count - 1);
        break;
    }
    case PM4ItOpcode::SetContextReg: {
        const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
        std::memcpy(&regs.reg_array[Regs::ContextRegWordOffset + set_data->reg_offset], payload,
                    (count - 1) * sizeof(u32));
        mark(Regs::ContextRegWordOffset + set_data->reg_offset, count - 1);
        break;
    }
    case PM4ItOpcode::SetShReg: {
        const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
        // The compute program range goes to the queue's cs_state, not to regs.
        if (!(set_data->reg_offset >= 0x200 &&
              set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4))) {
            std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset], payload,
                        (count - 1) * sizeof(u32));
            mark(Regs::ShRegWordOffset + set_data->reg_offset, count - 1);
        }
        break;
    }
    case PM4ItOpcode::SetUconfigReg: {
        const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
        std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset], payload,
                    (count - 1) * sizeof(u32));
        mark(Regs::UconfigRegWordOffset + set_data->reg_offset, count - 1);
        break;
    }
    case PM4ItOpcode::IndexType:
        regs.index_buffer_type.raw = reinterpret_cast<const PM4CmdDrawIndexType*>(header)->raw;
        mark_field(regs.index_buffer_type);
        break;
    case PM4ItOpcode::DrawIndex2: {
        const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
        regs.max_index_size = draw_index->max_size;
        regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
        regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
        regs.num_indices = draw_index->index_count;
        mark_field(regs.max_index_size);
        mark_field(regs.index_base_address);
        mark_field(regs.num_indices);
        regs.draw_initiator = draw_index->draw_initiator;
        mark_field(regs.draw_initiator);
        break;
    }
    case PM4ItOpcode::DrawIndexOffset2: {
        const auto* draw_index_off = reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
        regs.max_index_size = draw_index_off->max_size;
        regs.num_indices = draw_index_off->index_count;
        regs.draw_initiator = draw_index_off->draw_initiator;
        mark_field(regs.max_index_size);
        mark_field(regs.num_indices);
        mark_field(regs.draw_initiator);
        break;
    }
    case PM4ItOpcode::DrawIndexAuto: {
        const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
        mark_field(regs.num_indices);
        mark_field(regs.draw_initiator);
        regs.num_indices = draw_index->index_count;
        regs.draw_initiator = draw_index->draw_initiator;
        break;
    }
    case PM4ItOpcode::NumInstances:
        regs.num_instances.num_instances =
            reinterpret_cast<const PM4CmdDrawNumInstances*>(header)->num_instances;
        mark_field(regs.num_instances);
        break;
    case PM4ItOpcode::IndexBase: {
        const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
        regs.index_base_address.base_addr_lo = index_base->addr_lo;
        regs.index_base_address.base_addr_hi = index_base->addr_hi;
        mark_field(regs.index_base_address);
        break;
    }
    case PM4ItOpcode::IndexBufferSize:
        regs.num_indices = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header)->num_indices;
        mark_field(regs.num_indices);
        break;
    case PM4ItOpcode::EventWrite: {
        const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
        if (event->event_type.Value() != EventType::SoVgtStreamoutFlush) {
            return;
        }
        // TODO: handle proper synchronization, for now signal that update is done immediately
        regs.cp_strmout_cntl.offset_update_done = 1;
        mark_field(regs.cp_strmout_cntl);
        break;
    }
    default:
        return;
    }
    checksum = HashRegisterPacket(checksum, reinterpret_cast<const u32*>(header), count + 1);
}

// gow3: GOW3_DCB_STATS=1 — structure of graphics command buffers (read-only scan, printed every
// 5 s): how many buffers, draws per buffer, state set before the first draw. Input for
// processing command buffers on several threads.
namespace {
struct DcbStats {
    u64 buffers[2]{}, with_draws[2]{}, draws[2]{}, dispatches[2]{}, max_draws[2]{};
    u64 clear_first[2]{}, ctx_before[2]{}, sh_before[2]{}, ibs[2]{}, dwords[2]{};
    u64 hist[2][6]{}; // draws per buffer: 0, 1-9, 10-99, 100-499, 500-1999, 2000+
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
};
DcbStats g_dcb_stats;
int g_dcb_depth;

bool IsDrawOpcode(PM4ItOpcode op) {
    switch (op) {
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
    case PM4ItOpcode::DrawIndirect:
    case PM4ItOpcode::DrawIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirect:
    case PM4ItOpcode::DrawIndexIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirectCountMulti:
        return true;
    default:
        return false;
    }
}

void ScanDcb(std::span<const u32> dcb, int depth) {
    auto& st = g_dcb_stats;
    const int d = depth > 0 ? 1 : 0;
    u64 draws = 0, dispatches = 0, ctx = 0, sh = 0, ibs = 0;
    bool clear_first = false;
    st.dwords[d] += dcb.size();
    for (size_t at = 0; at < dcb.size();) {
        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data() + at);
        if (header->type == 2) {
            ++at;
            continue;
        }
        if (header->type != 3) {
            break;
        }
        const PM4ItOpcode op = header->type3.opcode;
        if (IsDrawOpcode(op)) {
            ++draws;
        } else if (op == PM4ItOpcode::DispatchDirect || op == PM4ItOpcode::DispatchIndirect) {
            ++dispatches;
        } else if (op == PM4ItOpcode::IndirectBuffer) {
            ++ibs;
        } else if (draws == 0) {
            if (op == PM4ItOpcode::ClearState) {
                clear_first = true;
            } else if (op == PM4ItOpcode::SetContextReg) {
                ctx += header->type3.NumWords() - 1;
            } else if (op == PM4ItOpcode::SetShReg) {
                sh += header->type3.NumWords() - 1;
            }
        }
        at += header->type3.NumWords() + 1;
    }
    ++st.buffers[d];
    st.draws[d] += draws;
    st.dispatches[d] += dispatches;
    st.ibs[d] += ibs;
    st.max_draws[d] = std::max(st.max_draws[d], draws);
    if (draws) {
        ++st.with_draws[d];
        st.clear_first[d] += clear_first;
        st.ctx_before[d] += ctx;
        st.sh_before[d] += sh;
    }
    const int bucket = draws == 0 ? 0 : draws < 10 ? 1 : draws < 100 ? 2 : draws < 500 ? 3
                     : draws < 2000 ? 4 : 5;
    ++st.hist[d][bucket];

    const auto now = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(now - st.window).count();
    if (depth == 0 && secs >= 5.0) {
        for (int k = 0; k < 2; ++k) {
            const double with = std::max<u64>(st.with_draws[k], 1);
            std::printf("DCB stats %s: %.0f buffers/s (%.0f with draws), %.0f draws/s, %.0f "
                        "dispatches/s, max %llu draws in one buffer, %.0f IBs/s, %.0f kdw/s; "
                        "draw buffers: %.0f%% start with ClearState, %.0f ctx + %.0f sh reg "
                        "writes before first draw; draws/buffer hist 0:%llu 1-9:%llu 10-99:%llu "
                        "100-499:%llu 500-1999:%llu 2000+:%llu\n",
                        k ? "indirect" : "submitted", st.buffers[k] / secs, st.with_draws[k] / secs,
                        st.draws[k] / secs, st.dispatches[k] / secs,
                        static_cast<unsigned long long>(st.max_draws[k]), st.ibs[k] / secs,
                        st.dwords[k] / secs / 1000.0, 100.0 * st.clear_first[k] / with,
                        st.ctx_before[k] / with, st.sh_before[k] / with,
                        static_cast<unsigned long long>(st.hist[k][0]),
                        static_cast<unsigned long long>(st.hist[k][1]),
                        static_cast<unsigned long long>(st.hist[k][2]),
                        static_cast<unsigned long long>(st.hist[k][3]),
                        static_cast<unsigned long long>(st.hist[k][4]),
                        static_cast<unsigned long long>(st.hist[k][5]));
        }
        st = DcbStats{};
    }
}
} // namespace

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb,
                                           u64 seq) {
    FIBER_ENTER(dcb_task_name);
    // Top-level buffers enqueued for the draw preparation workers (nested IBs are not).
    Vulkan::DrawPreparation* draw_prep =
        seq != NoSeq && rasterizer ? &rasterizer->GetDrawPreparation() : nullptr;
    if (draw_prep) {
        draw_prep->BeginSubmission(seq, regs, gfx_reg_checksum);
    }
    if (seq != NoSeq && rasterizer) {
        rasterizer->CloseRingReuse(); // constants are reused within one command buffer only
    }
    static const bool dcb_stats = EmulatorSettingsImpl::Flag("GOW3_DCB_STATS", false);
    const int dcb_depth = g_dcb_depth;
    if (dcb_stats) {
        ScanDcb(dcb, dcb_depth);
    }

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    while (!dcb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0:
            UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                            header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            ApplyGraphicsRegisterPacket(regs, header, gfx_reg_checksum, &pipe_dirty);
            if (Gow3Stats::enabled &&
                (opcode == PM4ItOpcode::SetContextReg || opcode == PM4ItOpcode::SetShReg ||
                 opcode == PM4ItOpcode::SetConfigReg || opcode == PM4ItOpcode::SetUconfigReg)) {
                Gow3StageA::Add(Gow3StageA::reg_words, count + 1); // A3: the words hashed
            }
            // DmaData to 0x3022C does nothing here (skipped below): no need to wait.
            if (rasterizer && !PipelinedOpcode(opcode) &&
                !(opcode == PM4ItOpcode::DmaData &&
                  reinterpret_cast<const PM4DmaData*>(header)->dst_addr_lo == 0x3022C)) {
                rasterizer->DrainDrawPipe(static_cast<u32>(opcode));
                // They may write guest memory: not before fences deferred earlier.
                rasterizer->WaitDeferredSignals();
            }
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    // gow3: after the writes before it (the buffer label: WriteData, which
                    // may still be queued on the draw recording thread).
                    if (rasterizer) {
                        rasterizer->RunInOrder(&SignalFlip, header, sizeof(u32),
                                               Gow3Toggle::PipelinedMemoryWrites);
                    } else {
                        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState:
            case PM4ItOpcode::SetConfigReg: {
                break; // registers: ApplyGraphicsRegisterPacket
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } // other SH registers: ApplyGraphicsRegisterPacket
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                break; // registers: ApplyGraphicsRegisterPacket
            }
            case PM4ItOpcode::SetPredication: {
                // gow3: once; God of War III sets it around most draws (~1000 lines a frame).
                static std::once_flag predication_logged;
                std::call_once(predication_logged,
                               [] { LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION"); });
                break;
            }
            case PM4ItOpcode::IndexType: {
                break; // registers: ApplyGraphicsRegisterPacket
            }
            case PM4ItOpcode::DrawIndex2: {
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const auto* prepared = draw_prep ? draw_prep->NextDraw() : nullptr;
                rasterizer->ScopeMarker("gfx:{}:DrawIndex2", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(true, 0, prepared); });
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const auto* prepared = draw_prep ? draw_prep->NextDraw() : nullptr;
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexOffset2", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->Draw(true, draw_index_off->index_offset, prepared); });
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                const auto* prepared = draw_prep ? draw_prep->NextDraw() : nullptr;
                rasterizer->ScopeMarker("gfx:{}:DrawIndexAuto", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(false, 0, prepared); });
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_index_indirect->base_vtx_loc,
                                                 draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count, 0, draw_index_indirect->base_vtx_loc,
                            draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectCountMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count,
                            draw_index_indirect->count_indirect_enable.Value()
                                ? draw_index_indirect->count_addr
                                : 0,
                            draw_index_indirect->base_vtx_loc, draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DispatchDirect", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->DispatchDirect(); });
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DispatchIndirect", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->DispatchIndirect(indirect_args_addr, offset, size); });
                break;
            }
            case PM4ItOpcode::NumInstances:
            case PM4ItOpcode::IndexBase:
            case PM4ItOpcode::IndexBufferSize: {
                break; // registers: ApplyGraphicsRegisterPacket
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                if (Gow3Stats::enabled) { // gow3 gate 2: the guest's cache actions
                    ++Gow3FenceStats::event_write[u32(event->event_type.Value()) & 63];
                    if (event->inv_l2) {
                        ++Gow3FenceStats::event_write_inv_l2;
                    }
                }
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // registers: ApplyGraphicsRegisterPacket
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        Gow3Stats::occlusion_dumps.fetch_add(1, std::memory_order_relaxed);
                        // gow3 GOW3_OCCLUSION=1: counted by Vulkan occlusion queries in order
                        // with the draws, written once the GPU has them (vk_occlusion.h).
                        if (rasterizer && rasterizer->OcclusionEvent(event->Address<VAddr>(),
                                                                     u32(num_counter_pairs))) {
                            break;
                        }
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        u64* results = event->Address<u64*>();
                        for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                            *results = pixel_counter | OcclusionCounterValidMask;
                        }
                        pixel_counter += OcclusionCounterStep;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                if (Gow3Stats::enabled) {
                    ++Gow3FenceStats::eos_event[event_eos->event_type & 63];
                }
                if (rasterizer &&
                    rasterizer->RunInOrder(&RunEventWriteEos, event_eos, sizeof(*event_eos))) {
                    NotePendingFences(*event_eos);
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                if (Gow3Stats::enabled) {
                    ++Gow3FenceStats::eop_event[event_eop->event_type & 63];
                    if (event_eop->int_sel != InterruptSelect::None) {
                        ++Gow3FenceStats::eop_int_irq;
                    }
                    if (event_eop->data_sel == DataSelect::None) {
                        ++Gow3FenceStats::eop_data_none;
                    }
                }
                if (rasterizer) {
                    if (EarlyFrameEop() && !Gow3GpuFences::Enabled() && IsFrameEop(*event_eop)) {
                        const u64 decoded = frame_eops_decoded.load(std::memory_order_relaxed);
                        const u64 lead =
                            decoded - frame_eops_recorded.load(std::memory_order_acquire);
                        frame_eops_decoded.store(decoded + 1, std::memory_order_relaxed);
                        if (lead < EarlyFrameEopLead()) {
                            SignalEop(*event_eop); // the label and the interrupt, now
                            rasterizer->RunInOrder(&RunEarlyEopTail, event_eop,
                                                   sizeof(*event_eop));
                            if (Gow3Stats::enabled) {
                                ++Gow3FenceStats::early_eops;
                                Gow3FenceStats::AtomicMax(Gow3FenceStats::early_max_lead, lead + 1);
                            }
                            break;
                        }
                        if (Gow3Stats::enabled) {
                            ++Gow3FenceStats::early_fallbacks;
                        }
                    }
                    if (rasterizer->RunInOrder(&RunEventWriteEop, event_eop, sizeof(*event_eop))) {
                        NotePendingFences(*event_eop);
                    }
                } else {
                    SignalEop(*event_eop);
                }
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                rasterizer->CloseRingReuse(); // memory changes from here on
                if (rasterizer->RunInOrder(&RunDmaData, dma_data, sizeof(PM4DmaData),
                                           Gow3Toggle::PipelinedMemoryWrites) &&
                    (dma_data->dst_sel == DmaDataDst::Memory ||
                     dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->NotePendingGpuWrite(dma_data->DstAddress<VAddr>(),
                                                    dma_data->NumBytes());
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                ASSERT(!write_data->wr_one_addr.Value());
                if (rasterizer) {
                    rasterizer->CloseRingReuse(); // memory changes from here on
                    // In order with the draws (on the draw recording thread when in use).
                    Gow3WriteLog::NoteIntent(write_data->Address<u64>(), write_data->data,
                                           (count - 2) * sizeof(u32),
                                           Gow3WriteLog::WriteDataIntent);
                    if (rasterizer->RunInOrder(&RunWriteData, header, (count + 1) * sizeof(u32),
                                               Gow3Toggle::PipelinedMemoryWrites)) {
                        NotePendingWrite(*write_data, (count - 2) * sizeof(u32));
                        rasterizer->NotePendingGpuWrite(write_data->Address<VAddr>(),
                                                        (count - 2) * sizeof(u32));
                    }
                } else {
                    std::memcpy(write_data->Address<u64*>(), write_data->data, (count - 2) * 4);
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (mem_semaphore->IsSignaling()) {
                    mem_semaphore->Signal();
                } else {
                    if (rasterizer) {
                        rasterizer->CloseRingReuse(); // the CPU may write memory meanwhile
                    }
                    while (!mem_semaphore->Signaled()) {
                        YIELD_GFX();
                    }
                    mem_semaphore->Decrement();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                if (Gow3Stats::enabled) {
                    const auto* acquire_mem = reinterpret_cast<const PM4CmdAcquireMem*>(header);
                    Gow3FenceStats::NoteAcquireMem(
                        acquire_mem->cp_coher_cntl,
                        (u64(acquire_mem->cp_coher_size_hi) << 32) | acquire_mem->cp_coher_size_lo);
                }
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                rasterizer->CloseRingReuse(); // the CPU may write memory meanwhile
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                if (rasterizer) {
                    rasterizer->CloseRingReuse(); // the CPU may write memory meanwhile
                }
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    // gow3 (spacepepperoni/medievil_ps4_Linux 6ef7566, from the Uncharted Linux
                    // fork): a display buffer the presenter has released already needs no drain;
                    // stage B goes on with the previous frame while this one is decoded. Ours
                    // adds one check: the value in memory is the one this point of the stream
                    // sees only if no write to the label is still queued for stage B (the flip's
                    // WRITE_DATA that marks the buffer in use: NotePendingWrite); with one queued
                    // the wait drains as before. An unmet wait drains and sleeps as before.
                    // GOW3_VO_LABEL_DRAIN=1: always drain.
                    static const bool always_drain = [] {
                        const char* env = std::getenv("GOW3_VO_LABEL_DRAIN");
                        return env && env[0] == '1';
                    }();
                    if (!always_drain && wait_reg_mem->Test(regs.reg_array)) {
                        u32 pending;
                        if (!rasterizer ||
                            !PendingFenceValue(wait_reg_mem->Address<VAddr>(), pending) ||
                            wait_reg_mem->TestValue(pending)) {
                            Gow3DecodePacing::NoteVoLabelWait(Gow3DecodePacing::VoWait::MetNoDrain);
                            break;
                        }
                        Gow3DecodePacing::NoteVoLabelWait(Gow3DecodePacing::VoWait::StalePending);
                    } else {
                        Gow3DecodePacing::NoteVoLabelWait(Gow3DecodePacing::VoWait::Drained);
                    }
                    if (rasterizer) {
                        rasterizer->DrainDrawPipe(static_cast<u32>(opcode));
                    }
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    break;
                }
                // gow3: a fence handed to the draw recording thread counts as written: that
                // thread runs everything in stream order.
                if (u32 value; wait_reg_mem->mem_space.Value() ==
                                   PM4CmdWaitRegMem::MemSpace::Memory &&
                               !Gow3Toggle::Disabled(Gow3Toggle::PendingFenceWaits) &&
                               PendingFenceValue(wait_reg_mem->Address<VAddr>(), value) &&
                               wait_reg_mem->TestValue(value)) {
                    if (Gow3GpuFences::Enabled()) {
                        Gow3GpuFences::counters.wrm_pending_met.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    break;
                }
                if (Gow3GpuFences::Enabled() && rasterizer) {
                    // gow3 GOW3_GPU_FENCES (Bloodborne 0.7, 0fc2921 cp_graphics.cpp
                    // WaitRegMemGraphics): the value may come from a fence waiting for work not
                    // submitted yet (another queue's): submitted, then yielded.
                    GpuOrderedWait wait{wait_reg_mem->Address<VAddr>()};
                    while (!wait_reg_mem->Test(regs.reg_array)) {
                        if (!rasterizer->DrawPipeIdle()) {
                            rasterizer->DrainDrawPipe(static_cast<u32>(opcode));
                            continue;
                        }
                        if (wait.Step(*rasterizer)) {
                            break; // GOW3_WAIT_REG_MEM_GIVEUP_MS
                        }
                        YIELD_GFX();
                    }
                    break;
                }
                // Else the value may come from a fence that thread has yet to write: let it
                // catch up before yielding to other queues.
                while (!wait_reg_mem->Test(regs.reg_array)) {
                    if (rasterizer && !rasterizer->DrawPipeIdle()) {
                        rasterizer->DrainDrawPipe(static_cast<u32>(opcode));
                        continue;
                    }
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {});
                // Tasks start suspended: the nested body (and its scan) runs on this resume.
                g_dcb_depth = dcb_depth + 1;
                RESUME_GFX(task);
                g_dcb_depth = dcb_depth;

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_WARNING(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            dcb = NextPacket(dcb, header->type3.NumWords() + 1);
            break;
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    if (rasterizer && seq != NoSeq) {
        rasterizer->RetireSubmission(); // prepared draws live until the recording thread is past
    }
    if (draw_prep) {
        draw_prep->EndSubmission();
    }
    if (rasterizer && seq != NoSeq && Gow3GpuFences::Enabled()) {
        // gow3 GOW3_GPU_FENCES (Bloodborne 0.7, 0fc2921 cp_graphics.cpp EndGraphics): its
        // signals go to the GPU now if it has run dry, else batched.
        rasterizer->SubmitForSignals();
    }
    if (seq != NoSeq && Gow3Stats::enabled) {
        Gow3Stats::submissions.fetch_add(1, std::memory_order_relaxed);
#ifndef _WIN32 // gow3: per-thread rusage statistics are Linux-only
        if (rusage usage{}; getrusage(RUSAGE_THREAD, &usage) == 0) {
            Gow3Stats::gpu_user_us.store(u64(usage.ru_utime.tv_sec) * 1000000 + usage.ru_utime.tv_usec,
                                       std::memory_order_relaxed);
            Gow3Stats::gpu_sys_us.store(u64(usage.ru_stime.tv_sec) * 1000000 + usage.ru_stime.tv_usec,
                                      std::memory_order_relaxed);
            Gow3Stats::gpu_invol_switches.store(usage.ru_nivcsw, std::memory_order_relaxed);
            Gow3Stats::gpu_vol_switches.store(usage.ru_nvcsw, std::memory_order_relaxed);
            Gow3Stats::gpu_minor_faults.store(usage.ru_minflt, std::memory_order_relaxed);
        }
#endif
    }

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    auto& queue = asc_queues[{vqid}];

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        ProcessCommands();
        if (rasterizer) {
            rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonCompute); // gow3: compute work uses the caches
        }

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            ASSERT(dma_data->command.das == 0);
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                u32(dma_data->dst_sel));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            if (Gow3Stats::enabled) {
                const auto* acquire_mem = reinterpret_cast<const PM4CmdAcquireMem*>(header);
                Gow3FenceStats::NoteAcquireMem(
                    acquire_mem->cp_coher_cntl,
                    (u64(acquire_mem->cp_coher_size_hi) << 32) | acquire_mem->cp_coher_size_lo);
            }
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
                // gow3: interleaves with the graphics stream at an unpredictable point, so
                // draw-preparation workers cannot reproduce it: make their checksums differ.
                gfx_reg_checksum = HashRegisterPacket(gfx_reg_checksum ^ 0xA5C0A5C0A5C0ull,
                                                      reinterpret_cast<const u32*>(header),
                                                      header->type3.NumWords() + 1);
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchDirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchDirect(); });
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchIndirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchIndirect(ib_address, 0, size); });
            break;
        }
        case PM4ItOpcode::WriteData: {
            // gow3: copies deferred to the recording thread that read this range precede the
            // write, and graphics fences deferred to it precede writes the guest sees (the pipe
            // was drained before this packet).
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            if (rasterizer) {
                rasterizer->WaitHostCopiesFor(write_data->Address<VAddr>(),
                                              (header->type3.count.Value() - 2) * 4);
                rasterizer->WaitDeferredSignals();
            }
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (!write_data->wr_one_addr.Value()) {
                std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                mem_semaphore->Signal();
            } else {
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                mem_semaphore->Decrement();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            if (Gow3GpuFences::Enabled() && rasterizer) {
                // gow3 GOW3_GPU_FENCES (Bloodborne 0.7, 0fc2921 cp_compute.cpp): a fence handed
                // to the recording thread or the signal thread counts as written (stream order);
                // else the value may come from a fence waiting for work not submitted yet.
                if (u32 value; wait_reg_mem->mem_space.Value() ==
                                   PM4CmdWaitRegMem::MemSpace::Memory &&
                               !Gow3Toggle::Disabled(Gow3Toggle::PendingFenceWaits) &&
                               PendingFenceValue(wait_reg_mem->Address<VAddr>(), value) &&
                               wait_reg_mem->TestValue(value)) {
                    Gow3GpuFences::counters.wrm_pending_met.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
                GpuOrderedWait wait{wait_reg_mem->Address<VAddr>()};
                while (!wait_reg_mem->Test(regs.reg_array)) {
                    if (!rasterizer->DrawPipeIdle()) {
                        rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonCompute);
                        continue;
                    }
                    if (wait.Step(*rasterizer)) {
                        break; // GOW3_WAIT_REG_MEM_GIVEUP_MS
                    }
                    YIELD_ASC(vqid);
                }
                break;
            }
            while (!wait_reg_mem->Test(regs.reg_array)) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            // gow3: copies deferred to the recording thread precede writes the guest sees, and
            // so do graphics fences deferred to it (the pipe was drained before this packet).
            // GOW3_GPU_FENCES: the GPU's order covers both (below).
            if (rasterizer && !Gow3GpuFences::Enabled()) {
                rasterizer->WaitHostCopies();
                rasterizer->WaitDeferredSignals();
            }
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            if (Gow3Stats::enabled) { // gow3 gate 2: compute cache actions
                ++Gow3FenceStats::release_count;
                ++Gow3FenceStats::release_event[release_mem->event_type & 63];
                const u32 bits[] = {release_mem->tcl1_vol_action_ena, release_mem->tc_vol_action_ena,
                                    release_mem->tc_wb_action_ena, release_mem->tcl1__action_ena,
                                    release_mem->tc_action_ena};
                for (unsigned i = 0; i < 5; ++i) {
                    if (bits[i]) {
                        ++Gow3FenceStats::release_bit_counts[i];
                    }
                }
            }
            if (rasterizer) {
                rasterizer->ProcessDownloadImages();
            }
            if (Gow3GpuFences::Enabled() && rasterizer) {
                // gow3 GOW3_GPU_FENCES (Bloodborne 0.7, 0fc2921 cp_compute.cpp ReleaseMemory):
                // the label and the interrupt once the GPU has finished the work before them; a
                // GDS store as a GPU copy recorded here, in stream order.
                const auto release = *release_mem;
                if (release.data_sel == DataSelect::GdsMemStore) {
                    rasterizer->CopyBuffer(release.Address<VAddr>(), release.gds_index,
                                           release.num_dw * sizeof(u32), false, true);
                }
                const bool label = release.data_sel == DataSelect::Data32Low ||
                                   release.data_sel == DataSelect::Data64;
                rasterizer->SignalAfterGpu(
                    [release, pipe_id = queue.pipe_id] { SignalReleaseAfterGpu(release, pipe_id); },
                    label ? release.Address<VAddr>() : 0,
                    release.data_sel == DataSelect::Data64 ? release.DataQWord()
                                                           : release.DataDWord());
                break;
            }
            release_mem->SignalFence(
                [pipe_id = queue.pipe_id] {
                    Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
                },
                [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                    rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
                });
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    // The copy for the draw preparation workers is made before taking the queue lock, which
    // the GPU thread needs to pick up work.
    auto* draw_prep = rasterizer ? &rasterizer->GetDrawPreparation() : nullptr;
    auto prep_submission = draw_prep ? draw_prep->Build(dcb) : nullptr;
    {
        // Numbering, enqueueing and queueing under one lock keep the three orders identical.
        std::scoped_lock lock{queue.m_access};
        u64 seq = NoSeq;
        if (draw_prep) {
            seq = gfx_submit_seq++;
            draw_prep->Enqueue(seq, std::move(prep_submission));
        }
        auto task = ProcessGraphics(dcb, ccb, seq);
        queue.submits.emplace(task.handle);
        if (Gow3GpuFences::Enabled() || Gow3DecodePacing::Enabled()) {
            queue.submit_index.push(submit_counter.fetch_add(1, std::memory_order_acq_rel) + 1);
        }
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    work_retired = false;
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
        if (Gow3GpuFences::Enabled() || Gow3DecodePacing::Enabled()) {
            queue.submit_index.push(submit_counter.fetch_add(1, std::memory_order_acq_rel) + 1);
        }
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    work_retired = false;
    submit_cv.notify_one();
}

} // namespace AmdGpu
