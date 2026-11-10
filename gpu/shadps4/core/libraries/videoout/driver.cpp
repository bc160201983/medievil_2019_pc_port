// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "gow3_async_log.h"
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <dxgi.h>
#endif
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include "common/assert.h"
#include "gow3_toggles.h"
#include "gow3_fence_stats.h"
#include "gow3_decode_pacing.h"
#include "gow3_draw_classes.h"
#include "gow3_drain_wait.h"
#include "gow3_signal_wait.h"
#include "gow3_stage_b_wait.h"
#include "gow3_texset_table.h"
#include "gow3_stage_a.h"
#include "gow3_gpu_fences.h"
extern "C" void runtime_memory_fault_report(char* out, size_t size, unsigned frames, double window_s);
extern "C" void runtime_memory_lock_report(char* out, size_t size, unsigned frames);
#include "gow3_sections.h"
#include "gow3_flip_trace.h"
#include <x86intrin.h>
#include "gow3_settings.h"
#include "gow3_graphics.h"
#include "gow3_linux_sched.h"
#include "gow3_overlay.h"
#include "video_core/renderer_vulkan/vk_frame_capture.h"
#include "common/debug.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/videoout/driver.h"
#include "core/libraries/videoout/videoout_error.h"
#include "imgui/renderer/imgui_core.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_wait_diagnostics.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Vulkan {
extern std::atomic<u64> g_gow3_compile_ns;
extern std::atomic<u32> g_gow3_compiles;
extern std::atomic<u64> g_gow3_capture_ns, g_gow3_async_ns, g_gow3_compile_wait_ns;
extern std::atomic<u64> g_gow3_snapshot_ns, g_gow3_guest_wait_ns;
extern std::atomic<u32> g_gow3_pending_draws;
} // namespace Vulkan

namespace Libraries::VideoOut {

constexpr static bool Is32BppPixelFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::A8R8G8B8Srgb:
    case PixelFormat::A8B8G8R8Srgb:
    case PixelFormat::A2R10G10B10:
    case PixelFormat::A2R10G10B10Srgb:
    case PixelFormat::A2R10G10B10Bt2020Pq:
        return true;
    default:
        return false;
    }
}

constexpr u32 PixelFormatBpp(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::A16R16G16B16Float:
        return 8;
    default:
        return 4;
    }
}

VideoOutDriver::VideoOutDriver(u32 width, u32 height) {
    main_port.resolution.full_width = width;
    main_port.resolution.full_height = height;
    main_port.resolution.pane_width = width;
    main_port.resolution.pane_height = height;
    // A 120 Hz+ vblank is reported as a 119.88 Hz display: engines that pace to the display's
    // refresh (Unreal Engine 4 on PS4) then run up to 120 FPS.
    if (EmulatorSettings.GetVblankFrequency() >= 120) {
        main_port.resolution.refresh_rate = SCE_VIDEO_OUT_REFRESH_RATE_119_88HZ;
    }
    const char* separate = std::getenv("GOW3_PRESENT_THREAD");
    separate_swap = !(separate && separate[0] == '0');
    if (separate_swap) {
        swap_thread = std::jthread([&](std::stop_token token) { SwapThread(token); });
    }
    present_thread = std::jthread([&](std::stop_token token) { PresentThread(token); });
    if (const char* newest = std::getenv("GOW3_NEWEST_FRAME"); newest && newest[0] == '1') {
        pacer_thread = std::jthread([&](std::stop_token token) { PacerThread(token); });
    }
}

// gow3 GOW3_PRESENT_PRIORITY=1: the presenting threads (vblank tick, pacer, swap) run time-critical
// and in the MMCSS "Games" task, high priority. Common::SetCurrentThreadRealtime is a no-op on
// Windows: these threads ran at normal priority, and with a fixed tick grid releases were
// processed up to a tick late (T-F15: 7/9-tick gaps in equal numbers).
static void BoostPresentingThread(const char* name) {
#ifdef _WIN32
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_PRESENT_PRIORITY");
        return env && env[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    const BOOL priority = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    bool mmcss = false;
    if (const HMODULE avrt = LoadLibraryA("avrt.dll")) {
        using SetTask = HANDLE(WINAPI*)(LPCWSTR, LPDWORD);
        using SetPriority = BOOL(WINAPI*)(HANDLE, int);
        const auto set_task = reinterpret_cast<SetTask>(
            reinterpret_cast<void*>(GetProcAddress(avrt, "AvSetMmThreadCharacteristicsW")));
        const auto set_priority = reinterpret_cast<SetPriority>(
            reinterpret_cast<void*>(GetProcAddress(avrt, "AvSetMmThreadPriority")));
        DWORD task_index = 0;
        if (set_task) {
            if (const HANDLE task = set_task(L"Games", &task_index)) {
                mmcss = true;
                if (set_priority) {
                    set_priority(task, 1 /* AVRT_PRIORITY_HIGH */);
                }
            }
        }
    }
    std::printf("VideoOut: %s thread priority %s, MMCSS %s\n", name, priority ? "time-critical" : "unchanged",
                mmcss ? "Games (high)" : "not available");
#else
    (void)name;
#endif
}

void VideoOutDriver::RunPresenter(std::function<void()> work, bool if_idle) {
    if (!separate_swap) {
        work();
        return;
    }
    {
        std::scoped_lock lock{swap_mutex};
        if (if_idle && (swap_busy || !swap_queue.empty())) {
            return; // a redraw of the last frame is pointless while frames are queued
        }
        swap_queue.push_back(std::move(work));
    }
    swap_cv.notify_one();
}

void VideoOutDriver::SwapThread(std::stop_token token) {
    Common::SetCurrentThreadName("gow3:Present");
    BoostPresentingThread("swap");
    while (true) {
        std::function<void()> work;
        {
            std::unique_lock lock{swap_mutex};
            swap_busy = false;
            if (!swap_cv.wait(lock, token, [this] { return !swap_queue.empty(); })) {
                return;
            }
            work = std::move(swap_queue.front());
            swap_queue.pop_front();
            swap_busy = true;
        }
        work();
    }
}

VideoOutDriver::~VideoOutDriver() = default;

int VideoOutDriver::Open(const ServiceThreadParams* params) {
    if (main_port.is_open) {
        return ORBIS_VIDEO_OUT_ERROR_RESOURCE_BUSY;
    }
    main_port.is_open = true;
    liverpool->SetVoPort(&main_port);
    return 1;
}

void VideoOutDriver::Close(s32 handle) {
    std::scoped_lock lock{mutex};

    // Mark as closed
    main_port.is_open = false;
    main_port.flip_rate = 0;
    main_port.prev_index = -1;

    // Clear port information
    std::memset(main_port.buffer_labels.data(), 0, sizeof(main_port.buffer_labels));
    std::memset(main_port.groups.data(), 0, sizeof(main_port.groups));
    std::memset(&main_port.vblank_status, 0, sizeof(main_port.vblank_status));
    main_port.flip_status = FlipStatus{};

    // Re-initialize buffers
    std::memset(main_port.buffer_slots.data(), 0, sizeof(main_port.buffer_slots));
    for (auto& buffer : main_port.buffer_slots) {
        buffer.group_index = -1;
    }

    // Clear events
    for (auto event : main_port.flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.flip_events.clear();
    for (auto event : main_port.vblank_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.vblank_events.clear();
}

VideoOutPort* VideoOutDriver::GetPort(int handle) {
    if (handle != 1) [[unlikely]] {
        return nullptr;
    }
    return &main_port;
}

int VideoOutDriver::RegisterBuffers(VideoOutPort* port, s32 startIndex, void* const* addresses,
                                    s32 bufferNum, const BufferAttribute* attribute) {
    const s32 group_index = port->FindFreeGroup();
    if (group_index >= MaxDisplayBufferGroups) {
        return ORBIS_VIDEO_OUT_ERROR_NO_EMPTY_SLOT;
    }

    if (startIndex + bufferNum > MaxDisplayBuffers || startIndex > MaxDisplayBuffers ||
        bufferNum > MaxDisplayBuffers) {
        LOG_ERROR(Lib_VideoOut,
                  "Attempted to register too many buffers startIndex = {}, bufferNum = {}",
                  startIndex, bufferNum);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    const s32 end_index = startIndex + bufferNum;
    if (bufferNum > 0 &&
        std::any_of(port->buffer_slots.begin() + startIndex, port->buffer_slots.begin() + end_index,
                    [](auto& buffer) { return buffer.group_index != -1; })) {
        return ORBIS_VIDEO_OUT_ERROR_SLOT_OCCUPIED;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "startIndex = {}, bufferNum = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             startIndex, bufferNum, GetPixelFormatString(attribute->pixel_format),
             attribute->aspect_ratio, static_cast<u32>(attribute->tiling_mode), attribute->width,
             attribute->height, attribute->pitch_in_pixel, attribute->option);

    auto& group = port->groups[group_index];
    std::memcpy(&group.attrib, attribute, sizeof(BufferAttribute));
    group.is_occupied = true;

    for (u32 i = 0; i < bufferNum; i++) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        port->buffer_slots[startIndex + i] = VideoOutBuffer{
            .group_index = group_index,
            .address_left = address,
            .address_right = 0,
        };
        Vulkan::FrameCapture::AddDisplayBuffer(address);

        // Reset flip label also when registering buffer
        port->buffer_labels[startIndex + i] = 0;
        port->SignalVoLabel();

        presenter->RegisterVideoOutSurface(group, address);
        LOG_INFO(Lib_VideoOut, "buffers[{}] = {:#x}", i + startIndex, address);
    }

    return group_index;
}

int VideoOutDriver::UnregisterBuffers(VideoOutPort* port, s32 attributeIndex) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    auto& group = port->groups[attributeIndex];
    group.is_occupied = false;

    for (auto& buffer : port->buffer_slots) {
        if (buffer.group_index != attributeIndex) {
            continue;
        }
        buffer.group_index = -1;
    }

    return ORBIS_OK;
}

int VideoOutDriver::ChangeBufferAttribute(VideoOutPort* port, s32 attributeIndex,
                                          const BufferAttribute* attribute) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "attributeIndex = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             attributeIndex, GetPixelFormatString(attribute->pixel_format), attribute->aspect_ratio,
             static_cast<u32>(attribute->tiling_mode), attribute->width, attribute->height,
             attribute->pitch_in_pixel, attribute->option);

    std::unique_lock lock{port->port_mutex};
    std::memcpy(&port->groups[attributeIndex].attrib, attribute, sizeof(BufferAttribute));
    return 0;
}

// gow3: when the game last flipped. The menu's idle redraws count from it, so they only fill gaps
// of 16 ms without a game frame (else a game running under the open menu got ~60 extra
// presents a second on top of its own, and its frame pacing suffered).
static std::atomic<std::chrono::steady_clock::rep> last_game_flip{0};
static std::chrono::steady_clock::time_point LastGameFlip() {
    return std::chrono::steady_clock::time_point{
        std::chrono::steady_clock::duration{last_game_flip.load(std::memory_order_relaxed)}};
}

void VideoOutDriver::Flip(const Request& req) {
    last_game_flip.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                         std::memory_order_relaxed);
    Gow3FlipTrace::Note("release", req.flip_arg);
    // Update HDR status before presenting, then present the frame (gow3: on the swap thread).
    const auto deadline = std::exchange(flip_deadline, {});
    RunPresenter([this, frame = req.frame, hdr = req.port->is_hdr, flip_arg = req.flip_arg, deadline] {
        presenter->SetHDR(hdr);
        presenter->SetPresentTag(flip_arg);
        presenter->SetPresentDeadline(deadline);
        presenter->Present(frame);
        // gow3 Reflex: the driver's sleep before the game's next frame (the flip completes after it).
        if (!Vulkan::Presenter::ReflexSleepBeforePresent() && presenter->LatencySleep()) {
            Gow3FlipTrace::Note("reflex_wake", flip_arg);
        }
        Gow3FlipTrace::Note("presented", flip_arg);
    });
    Vulkan::FrameCapture::OnFlip(req.index >= 0 ? req.port->buffer_slots[req.index].address_left
                                                : 0);

    // gow3: GOW3_FRAME_STATS=1 prints flip rate and frame time spread every 5 seconds.
    static const bool frame_stats = EmulatorSettingsImpl::Flag("GOW3_FRAME_STATS", false);
    if (frame_stats) {
        using Clock = std::chrono::steady_clock;
        static Clock::time_point window_start = Clock::now(), last = window_start;
        static u32 frames;
        static double worst_ms;
        const auto now = Clock::now();
        const double frame_ms = std::chrono::duration<double, std::milli>(now - last).count();
        worst_ms = std::max(worst_ms, frame_ms);
        static std::vector<double> intervals;
        intervals.push_back(frame_ms);
        last = now;
        // Stall diagnostics: what happened during a long frame.
        static u64 last_gpu_ns, last_images, last_image_bytes, last_buffer_bytes;
        static u64 last_t[6], last_minflt, last_sigf, last_pc, last_pp, last_rc, last_rp;
        const u64 pc = Gow3Stats::protect_calls.load(), pp = Gow3Stats::protect_pages.load(),
                  rc = Gow3Stats::protect_revoke_calls.load(), rp = Gow3Stats::protect_revoke_pages.load();
        const u64 minflt = Gow3Stats::gpu_minor_faults.load(), sigf = Gow3Stats::gpu_signal_faults.load();
        static u64 last_copy_cpu, last_copy_sys, last_copy_flt;
        const u64 copy_cpu = Gow3Stats::t_copy_cpu.load(), copy_sys = Gow3Stats::copy_sys_us.load(),
                  copy_flt = Gow3Stats::copy_minflt.load();
        static u64 last_proc_flt, last_copy_ns, last_copy_bytes, last_rf, last_trf, last_twf;
        const u64 rf = Gow3Stats::read_faults.load(), trf = Gow3Stats::t_read_faults.load(),
                  twf = Gow3Stats::t_write_faults.load();
        // gow3: every frame over 40 ms after the first, at most 10 per 5 s window.
        static bool primed = false;
        static u32 stalls_shown = 0;
        const bool stall = primed && frame_ms > 40.0 && stalls_shown < 10;
        stalls_shown += stall;
        primed = true;
        // Waits and work inside the long frame (window counters are reset every 5 s, so the
        // previous values are reset with them).
        static u64 last_wait[7];
        // The window counters are reset every 5 s: the frame after a reset has no valid deltas.
        static bool window_reset = false;
        static double pacing_median = 0; ///< set by the pacing report, read by the slow frames
        const u64 wait_now[7] = {Gow3Stats::t_readback_wait.load(), Gow3Stats::tick_wait_ns.load(),
                                 Gow3Stats::sync_recording_ns.load(), Gow3Stats::host_copies_wait_ns.load(),
                                 u64(Vulkan::g_gow3_compile_ns.load()), u64(Vulkan::g_gow3_compiles.load()),
                                 Gow3Stats::t_gc.load()};
        if (stall) {
            std::printf("Stall: %.1f ms frame; waits: readback %.1f ms, GPU ticks %.1f ms, "
                        "recorder %.1f ms, host copies %.1f ms; %llu compiles %.1f ms; GC %.1f ms\n",
                        frame_ms, (wait_now[0] - last_wait[0]) / 1e6, (wait_now[1] - last_wait[1]) / 1e6,
                        (wait_now[2] - last_wait[2]) / 1e6, (wait_now[3] - last_wait[3]) / 1e6,
                        static_cast<unsigned long long>(wait_now[5] - last_wait[5]),
                        (wait_now[4] - last_wait[4]) / 1e6, (wait_now[6] - last_wait[6]) / 1e6);
        }
        u64 frame_wait[7];
        std::copy(std::begin(last_wait), std::end(last_wait), std::begin(frame_wait));
        std::copy(std::begin(wait_now), std::end(wait_now), std::begin(last_wait));
        if (stall) {
            std::printf("       fault handlers: %llu read faults %.1f ms, write faults %.1f ms\n",
                        static_cast<unsigned long long>(rf - last_rf), (trf - last_trf) / 1e6,
                        (twf - last_twf) / 1e6);
        }
        last_rf = rf;
        last_trf = trf;
        last_twf = twf;
        const u64 copy_ns = Gow3Stats::t_copy.load(), copy_bytes = Gow3Stats::copy_bytes.load();
        u64 proc_flt = 0;
#ifndef _WIN32
        if (rusage usage{}; getrusage(RUSAGE_SELF, &usage) == 0) {
            proc_flt = usage.ru_minflt;
        }
#endif
        const u64 t_now[6] = {Gow3Stats::t_resident.load(), Gow3Stats::t_protect.load(),
                              Gow3Stats::t_image_create.load(), Gow3Stats::t_refresh.load(),
                              Gow3Stats::t_staging.load(), Gow3Stats::t_host_wait.load()};
        static u64 last_draws, last_dispatches, last_subs, last_sys, last_user, last_invol, last_vol;
        const u64 draws = Gow3Stats::draws.load(), dispatches = Gow3Stats::dispatches.load(),
                  subs = Gow3Stats::submissions.load(), sys_us = Gow3Stats::gpu_sys_us.load(),
                  user_us = Gow3Stats::gpu_user_us.load(), invol = Gow3Stats::gpu_invol_switches.load(),
                  vol = Gow3Stats::gpu_vol_switches.load();
        u64 gpu_ns = 0;
#ifdef _WIN32
        // gow3: Windows has no per-thread clock id; the thread's user + kernel time instead.
        static HANDLE gpu_thread = nullptr;
        if (!gpu_thread) {
            if (const unsigned long id = Gow3Stats::gpu_thread_id.load()) {
                gpu_thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, id);
            }
        }
        if (FILETIME created, exited, kernel, user;
            gpu_thread && GetThreadTimes(gpu_thread, &created, &exited, &kernel, &user)) {
            const auto ns = [](const FILETIME& t) {
                return ((u64(t.dwHighDateTime) << 32) | t.dwLowDateTime) * 100;
            };
            gpu_ns = ns(kernel) + ns(user);
        }
#else
        if (const int clock = Gow3Stats::gpu_thread_clock.load(); clock != -1) {
            timespec ts{};
            clock_gettime(static_cast<clockid_t>(clock), &ts);
            gpu_ns = u64(ts.tv_sec) * 1000000000ull + u64(ts.tv_nsec);
        }
#endif
        const u64 images = Gow3Stats::images_registered.load();
        const u64 image_bytes = Gow3Stats::image_upload_bytes.load();
        const u64 buffer_bytes = Gow3Stats::buffer_upload_bytes.load();
        if (stall) {
            std::printf("       GPU thread on CPU %.1f ms; %llu images registered, "
                        "%.1f MB image uploads, %.1f MB buffer uploads\n",
                        (gpu_ns - last_gpu_ns) / 1e6,
                        static_cast<unsigned long long>(images - last_images),
                        (image_bytes - last_image_bytes) / 1e6,
                        (buffer_bytes - last_buffer_bytes) / 1e6);
            std::printf("       %llu draws, %llu dispatches, %llu submissions; GPU thread user %.1f ms, "
                        "kernel %.1f ms, %llu preempted, %llu waits\n",
                        static_cast<unsigned long long>(draws - last_draws),
                        static_cast<unsigned long long>(dispatches - last_dispatches),
                        static_cast<unsigned long long>(subs - last_subs), (user_us - last_user) / 1e3,
                        (sys_us - last_sys) / 1e3, static_cast<unsigned long long>(invol - last_invol),
                        static_cast<unsigned long long>(vol - last_vol));
        }
        if (stall) {
            std::printf("       ms in: resident %.1f, protect %.1f, image create %.1f, "
                        "image refresh %.1f, staging %.1f, waiting for host copies %.1f\n",
                        (t_now[0] - last_t[0]) / 1e6, (t_now[1] - last_t[1]) / 1e6,
                        (t_now[2] - last_t[2]) / 1e6, (t_now[3] - last_t[3]) / 1e6,
                        (t_now[4] - last_t[4]) / 1e6, (t_now[5] - last_t[5]) / 1e6);
            std::printf("       GPU thread page faults %llu (process %llu), protection faults %llu; "
                        "protect calls %llu (%llu pages), of which write-revoking %llu (%llu pages)\n",
                        static_cast<unsigned long long>(minflt - last_minflt),
                        static_cast<unsigned long long>(proc_flt - last_proc_flt),
                        static_cast<unsigned long long>(sigf - last_sigf),
                        static_cast<unsigned long long>(pc - last_pc),
                        static_cast<unsigned long long>(pp - last_pp),
                        static_cast<unsigned long long>(rc - last_rc),
                        static_cast<unsigned long long>(rp - last_rp));
        }
        last_pc = pc;
        last_pp = pp;
        last_rc = rc;
        last_rp = rp;
        last_minflt = minflt;
        if (stall && copy_ns > last_copy_ns) {
            std::printf("       guest copies %.1f MB in %.1f thread-ms, %.1f ms on CPU (kernel %.1f ms, "
                        "%llu page faults in large copies) (%.2f GB/s per thread)\n",
                        (copy_bytes - last_copy_bytes) / 1e6, (copy_ns - last_copy_ns) / 1e6,
                        (copy_cpu - last_copy_cpu) / 1e6, (copy_sys - last_copy_sys) / 1e3,
                        static_cast<unsigned long long>(copy_flt - last_copy_flt),
                        double(copy_bytes - last_copy_bytes) / double(copy_ns - last_copy_ns));
        }
        last_copy_ns = copy_ns;
        last_copy_cpu = copy_cpu;
        last_copy_sys = copy_sys;
        last_copy_flt = copy_flt;
        last_copy_bytes = copy_bytes;
        last_proc_flt = proc_flt;
        const u64 frame_sigf = last_sigf, sigf_now = sigf;
        last_sigf = sigf;
        std::copy(std::begin(t_now), std::end(t_now), std::begin(last_t));
        const u64 frame_draws = draws - last_draws, frame_dispatches = dispatches - last_dispatches,
                  frame_subs = subs - last_subs, frame_gpu_ns = gpu_ns - last_gpu_ns;
        last_draws = draws;
        last_dispatches = dispatches;
        last_subs = subs;
        last_sys = sys_us;
        last_user = user_us;
        last_invol = invol;
        last_vol = vol;
        last_gpu_ns = gpu_ns;
        {
            // gow3 GOW3_SLOW_FRAMES=<ratio>: one line per frame slower than ratio x the previous
            // window's median, and a sample of ordinary frames (every 100th) to compare with:
            // what differs in the frames that make the 1% lows (microstutter).
            static const double slow_ratio = [] {
                const char* env = std::getenv("GOW3_SLOW_FRAMES");
                return env ? std::atof(env) : 0.0;
            }();
            static double median_ms = 0;
            static u64 last_occ, last_ends, last_busy, last_tsc, sample, last_idle_ns, last_drain;
            static u64 last_texset, last_rmemo, last_unprep, last_slowimg, last_genimg, last_upend, last_uunav, last_uchk;
            static std::array<u64, Gow3Sections::Count> last_sections{};
            static u32 slow_shown;
            const u64 occ = Gow3Stats::occlusion_dumps.load(), ends = Gow3Stats::render_pass_ends.load(),
                      busy = Gow3Stats::stage_b_busy_cycles.load(), tsc = __rdtsc(),
                      idle_ns = Gow3Stats::gpu_idle_total_ns.load(),
                      drain = Gow3Stats::stage_a_drain_cycles.load(),
                      texset = Gow3Stats::texset_misses.load(), rmemo = Gow3Stats::render_memo_misses.load(),
                      unprep = Gow3Stats::unprepared_draws.load(),
                      slowimg = Gow3Stats::image_slow_lookups.load(),
                      genimg = Gow3Stats::image_generation_misses.load(),
                      upend = Gow3Stats::unprep_pending.load(), uunav = Gow3Stats::unprep_unavailable.load(),
                      uchk = Gow3Stats::unprep_checksum.load();
            const bool slow = slow_ratio > 0 && median_ms > 0 && frame_ms > slow_ratio * median_ms &&
                              slow_shown < 600;
            const bool sampled = slow_ratio > 0 && median_ms > 0 && ++sample % 100 == 0;
            if ((slow || sampled) && last_tsc && !window_reset) {
                slow_shown += slow;
                // Written by a background thread (gow3_async_log.h): printing here delayed frames.
                std::string text;
                Gow3AsyncLog::Append(text, "%s %.2f ms: %llu draws, %llu dispatches, %llu submits, %llu occlusion "
                            "queries, %llu passes ended; stage B busy %.0f%%, stage A CPU %.2f ms, "
                            "idle for submits %.2f ms, draining %.2f ms; "
                            "waits: readback %.2f, ticks %.2f, recorder %.2f, host copies %.2f ms; "
                            "uploads %.2f MB buffers, %.2f MB images; %llu protection faults; "
                            "misses: %llu texture sets, %llu render states, %llu unprepared draws, "
                            "%llu image lookups (%llu generation only); unprepared: %llu not reached, %llu no pipeline, "
                            "%llu other registers\n",
                            slow ? "Slow frame" : "Frame sample", frame_ms,
                            static_cast<unsigned long long>(frame_draws),
                            static_cast<unsigned long long>(frame_dispatches),
                            static_cast<unsigned long long>(frame_subs),
                            static_cast<unsigned long long>((occ - last_occ) / 2),
                            static_cast<unsigned long long>(ends - last_ends),
                            100.0 * double(busy - last_busy) / double(tsc - last_tsc),
                            frame_gpu_ns / 1e6, (idle_ns - last_idle_ns) / 1e6,
                            frame_ms * double(drain - last_drain) / double(tsc - last_tsc), (wait_now[0] - frame_wait[0]) / 1e6,
                            (wait_now[1] - frame_wait[1]) / 1e6, (wait_now[2] - frame_wait[2]) / 1e6,
                            (wait_now[3] - frame_wait[3]) / 1e6, (buffer_bytes - last_buffer_bytes) / 1e6,
                            (image_bytes - last_image_bytes) / 1e6,
                            static_cast<unsigned long long>(sigf_now - frame_sigf),
                            static_cast<unsigned long long>(texset - last_texset),
                            static_cast<unsigned long long>(rmemo - last_rmemo),
                            static_cast<unsigned long long>(unprep - last_unprep),
                            static_cast<unsigned long long>(slowimg - last_slowimg),
                            static_cast<unsigned long long>(genimg - last_genimg),
                            static_cast<unsigned long long>(upend - last_upend),
                            static_cast<unsigned long long>(uunav - last_uunav),
                            static_cast<unsigned long long>(uchk - last_uchk));
                if (Gow3Sections::Enabled()) {
                    // Stage B's time in this frame by part (inclusive, ms): what grows in slow frames.
                    Gow3AsyncLog::Append(text, "  stage B ms:");
                    for (u32 i = 0; i < Gow3Sections::Count; ++i) {
                        const u64 now_cycles = Gow3Sections::cycles[i].load(std::memory_order_relaxed);
                        Gow3AsyncLog::Append(text, " %s %.2f", Gow3Sections::Names[i],
                                             frame_ms * double(now_cycles - last_sections[i]) /
                                                 double(tsc - last_tsc));
                    }
                    Gow3AsyncLog::Append(text, "\n");
                }
                Gow3AsyncLog::Write(std::move(text));
            }
            if (Gow3Sections::Enabled()) {
                for (u32 i = 0; i < Gow3Sections::Count; ++i) {
                    last_sections[i] = Gow3Sections::cycles[i].load(std::memory_order_relaxed);
                }
            }
            window_reset = false;
            last_occ = occ;
            last_ends = ends;
            last_busy = busy;
            last_tsc = tsc;
            last_idle_ns = idle_ns;
            last_drain = drain;
            last_texset = texset;
            last_rmemo = rmemo;
            last_unprep = unprep;
            last_slowimg = slowimg;
            last_genimg = genimg;
            last_upend = upend;
            last_uunav = uunav;
            last_uchk = uchk;
            if (pacing_median > 0) {
                median_ms = pacing_median;
                pacing_median = 0;
                slow_shown = 0;
            }
        }
        last_images = images;
        last_image_bytes = image_bytes;
        last_buffer_bytes = buffer_bytes;
        ++frames;
        const double window = std::chrono::duration<double>(now - window_start).count();
        if (window >= 5.0) {
            const u32 compiles = Vulkan::g_gow3_compiles.exchange(0);
            const u64 compile_ns = Vulkan::g_gow3_compile_ns.exchange(0);
            const double capture_ms = Vulkan::g_gow3_capture_ns.exchange(0) / 1e6;
            const double async_ms = Vulkan::g_gow3_async_ns.exchange(0) / 1e6;
            const double wait_ms = Vulkan::g_gow3_compile_wait_ns.exchange(0) / 1e6;
            const double snapshot_ms = Vulkan::g_gow3_snapshot_ns.exchange(0) / 1e6;
            const double guest_wait_ms = Vulkan::g_gow3_guest_wait_ns.exchange(0) / 1e6;
            const u32 pending_draws = Vulkan::g_gow3_pending_draws.exchange(0);
            const u64 direct = Vulkan::Scheduler::direct_recordings.exchange(0);
            const u64 faults = Gow3Stats::tracker_faults.exchange(0);
            // GPU command thread CPU time per draw: comparable between builds even when the scene
            // (and so the frame rate) differs a little.
            static u64 window_draws = 0, window_gpu_us = 0;
            const u64 all_draws = Gow3Stats::draws.load();
            const u64 gpu_us = Gow3Stats::gpu_user_us.load() + Gow3Stats::gpu_sys_us.load();
            const double us_per_draw = all_draws > window_draws
                                           ? double(gpu_us - window_gpu_us) / (all_draws - window_draws)
                                           : 0.0;
            const double draws_per_frame = frames ? double(all_draws - window_draws) / frames : 0.0;
            window_draws = all_draws;
            window_gpu_us = gpu_us;
            std::printf("Frame stats: %.1f FPS, worst frame %.1f ms (vblank %u Hz); "
                        "%u shader/pipeline compiles, %.1f ms; async: capture %.1f ms, workers "
                        "%.1f ms, sync wait %.1f ms, snapshots %.1f ms, GPU capture wait %.1f ms, "
                        "%u pending draws; %llu recorder syncs; "
                        "%.0f write faults/s, %lld hot pages; GPU thread %.2f us/draw, "
                        "%.0f draws/frame, idle %.1f%%; blocked: recorder %.1f%%, host copies "
                        "%.1f%% (%.0f/frame, %.0f/frame skipped), copy threads %.1f%%, GPU ticks %.1f%%; "
                        "reduced-size draws %.0f/frame of %.0f in the scene\n",
                        frames / window, worst_ms, EmulatorSettings.GetVblankFrequency(), compiles,
                        compile_ns / 1e6, capture_ms, async_ms, wait_ms, snapshot_ms, guest_wait_ms,
                        pending_draws,
                        static_cast<unsigned long long>(direct),
                        faults / window, static_cast<long long>(Gow3Stats::hot_pages.load()),
                        us_per_draw, draws_per_frame,
                        Gow3Stats::gpu_idle_ns.exchange(0) / (window * 1e7),
                        Gow3Stats::sync_recording_ns.exchange(0) / (window * 1e7),
                        Gow3Stats::host_copies_wait_ns.exchange(0) / (window * 1e7),
                        frames ? double(Gow3Stats::host_copy_waits.exchange(0)) / frames : 0.0,
                        frames ? double(Gow3Stats::host_copy_waits_skipped.exchange(0)) / frames : 0.0,
                        Gow3Stats::copy_threads_wait_ns.exchange(0) / (window * 1e7),
                        Gow3Stats::tick_wait_ns.exchange(0) / (window * 1e7),
                        frames ? double(Gow3Stats::reduced_draws.exchange(0)) / frames : 0.0,
                        frames ? double(Gow3Stats::scene_draws.exchange(0)) / frames : 0.0);
            {
                // Guest memory copied for the GPU: what a PC build would keep in VRAM is the
                // part that is not new data every frame.
                static u64 window_image_bytes = 0, window_buffer_bytes = 0;
                std::printf("Uploads: images %.1f MB/s, buffers %.1f MB/s\n",
                            (image_bytes - window_image_bytes) / 1e6 / window,
                            (buffer_bytes - window_buffer_bytes) / 1e6 / window);
                {
                    // gow3: time inside the port's protection fault handlers (not the OS's signal or
                    // exception delivery around them), per second, summed over the faulting threads.
                    static u64 window_rf = 0, window_trf = 0, window_twf = 0;
                    const u64 rf_now = Gow3Stats::read_faults.load(),
                              trf_now = Gow3Stats::t_read_faults.load(),
                              twf_now = Gow3Stats::t_write_faults.load();
                    std::printf("Fault handlers: read faults %.0f/s %.2f ms/s, write faults %.2f ms/s\n",
                                (rf_now - window_rf) / window, (trf_now - window_trf) / 1e6 / window,
                                (twf_now - window_twf) / 1e6 / window);
                    window_rf = rf_now;
                    window_trf = trf_now;
                    window_twf = twf_now;
                }
                {
                    // gow3 gate 2 (study-fence-waits.md 5): guest fence waits, the submission lock,
                    // cache actions, and the GPU write faults by PS4 memory type (runtime).
                    std::fputs(Gow3FenceStats::Report(window).c_str(), stdout);
                    // gow3 U5-2 (impl-gpu-ordered-fences.md): GOW3_GPU_FENCES' signals, flushes,
                    // pacing waits, and the runtime's guest memory (the leak view), both ways.
                    std::fputs(Gow3GpuFences::Report(window, frames).c_str(), stdout);
                    // gow3 GOW3_DECODE_PACING (impl-decode-pacing.md): pacing waits, frame lead,
                    // deferred GPU idle steps (nothing with the switch off).
                    std::fputs(Gow3DecodePacing::Report(window, frames).c_str(), stdout);
                    char fault_map[1024];
                    runtime_memory_fault_report(fault_map, sizeof(fault_map), frames, window);
                    std::fputs(fault_map, stdout);
                    // gow3 D4 / D5 (impl-stage-b-bundle-2.md): the region table lock (Linux).
                    runtime_memory_lock_report(fault_map, sizeof(fault_map), frames);
                    std::fputs(fault_map, stdout);
                }
                {
                    // gow3: CPU writes into GPU-visible mapped memory, per frame (the PCIe traffic
                    // GPU Trace showed: VRAM through the BAR, and staging in system memory).
                    const double per_frame = frames ? 1e6 * frames : 1e6;
                    std::printf("Mapped writes per frame: stream buffer %.2f MB, constant ring %.2f "
                                "MB (%.2f MB more reused) (VRAM through the BAR); staging uploads "
                                "%.2f MB, downloads %.2f MB (system memory)\n",
                                Gow3Stats::stream_buffer_bytes.exchange(0) / per_frame,
                                Gow3Stats::constant_ring_bytes.exchange(0) / per_frame,
                                Gow3Stats::constant_ring_reused_bytes.exchange(0) / per_frame,
                                Gow3Stats::staging_upload_bytes.exchange(0) / per_frame,
                                Gow3Stats::staging_download_bytes.exchange(0) / per_frame);
                    // gow3: what skipping repeated binds would save (stage B, guest draws).
                    const double frames_d = frames ? double(frames) : 1.0;
                    const u64 set_updates = Gow3Stats::descriptor_set_updates.exchange(0);
                    const u64 set_update_ns = Gow3Stats::t_descriptor_set_updates.exchange(0);
                    const u64 pushes = Gow3Stats::descriptor_pushes.exchange(0);
                    const u64 push_writes = Gow3Stats::descriptor_writes.exchange(0);
                    std::printf("Binds per frame: pipelines %.0f (%.0f same as the previous draw), "
                                "descriptor set updates %.1f (%.1f us each), descriptor pushes %.0f "
                                "(%.1f writes each)\n",
                                Gow3Stats::pipeline_binds.exchange(0) / frames_d,
                                Gow3Stats::pipeline_bind_repeats.exchange(0) / frames_d,
                                set_updates / frames_d,
                                set_updates ? set_update_ns / 1e3 / set_updates : 0.0,
                                pushes / frames_d, pushes ? double(push_writes) / pushes : 0.0);
                    // gow3 (study-stage-b-per-draw.md C1): stage B's system calls (on the Deck
                    // its ordered tasks cost 7-9x the PC's, its draws 4-5x): timeline semaphore
                    // queries and recorder wake-ups, there and elsewhere, and the queries the
                    // end-of-pipe poll interval skipped (second toggle word 16; the saving is
                    // those skips at stage B's own cost per query).
                    u64 queries[2], query_ns[2], wakes[2], wake_ns[2];
                    for (unsigned k = 0; k < 2; ++k) {
                        queries[k] = Gow3Stats::semaphore_queries[k].exchange(0);
                        query_ns[k] = Gow3Stats::semaphore_query_ns[k].exchange(0);
                        wakes[k] = Gow3Stats::recorder_wakes[k].exchange(0);
                        wake_ns[k] = Gow3Stats::recorder_wake_ns[k].exchange(0);
                    }
                    const u64 polls_skipped = Gow3Stats::semaphore_polls_skipped.exchange(0);
                    const auto each_us = [](u64 ns, u64 n) { return n ? ns / 1e3 / n : 0.0; };
                    std::printf("Stage B syscalls per frame: semaphore queries %.1f (%.2f us each; "
                                "other threads %.1f, %.2f us each), skipped by the poll interval "
                                "(%s) %.1f (~%.3f ms saved); recorder wakes %.1f (%.2f us each; "
                                "other threads %.1f, %.2f us each)\n",
                                queries[0] / frames_d, each_us(query_ns[0], queries[0]),
                                queries[1] / frames_d, each_us(query_ns[1], queries[1]),
                                Gow3Toggle2::Disabled(Gow3Toggle2::SemaphorePollInterval) ? "off"
                                                                                          : "on",
                                polls_skipped / frames_d,
                                polls_skipped * each_us(query_ns[0], queries[0]) / 1e3 / frames_d,
                                wakes[0] / frames_d, each_us(wake_ns[0], wakes[0]),
                                wakes[1] / frames_d, each_us(wake_ns[1], wakes[1]));
                    // gow3 C5 (impl-stage-b-bundle-2.md): the buffer cache's readback checks
                    // while copies were pending: answered without readback_mutex (no pending
                    // copy on the range's pages) or under it.
                    const u64 prechecks_free = Gow3Stats::readback_prechecks[0].exchange(0);
                    const u64 prechecks_locked = Gow3Stats::readback_prechecks[1].exchange(0);
                    std::printf("Readback pre-check per frame (C5 %s): without the lock %.1f, "
                                "locked %.1f\n",
                                Gow3Toggle2::Disabled(Gow3Toggle2::ReadbackPrecheck) ? "off" : "on",
                                prechecks_free / frames_d, prechecks_locked / frames_d);
                    // gow3 (impl-stage-b-per-draw-2.md): how often each per-draw cut applied.
                    // R2: residency checks answered by the bitmap or by the interval search.
                    // T1: texture binds of stages without images on the short path.
                    // T2 (GOW3_TEXSET_WIDE=1): memo lookups and hits of sets of 17-32 images
                    // and of stages without a prepared draw.
                    {
                        namespace TS = Gow3TexSet;
                        static u64 last[4]{};
                        const u64 now[4] = {TS::wide_lookups.load(), TS::wide_hits.load(),
                                            TS::unprepared_lookups.load(),
                                            TS::unprepared_hits.load()};
                        double d[4];
                        for (unsigned k = 0; k < 4; ++k) {
                            d[k] = (now[k] - last[k]) / frames_d;
                            last[k] = now[k];
                        }
                        const char* wide_state = std::getenv("GOW3_TEXSET_WIDE");
                        const bool wide_wanted = wide_state && wide_state[0] == '1';
                        std::printf(
                            "Stage B cuts per frame: residency bitmap (R2 %s) %.1f, searched "
                            "%.1f; no-image texture binds (T1 %s) %.1f; wide texture-set memo "
                            "(T2 %s): over 16 images %.1f lookups, %.1f hits; no prepared draw "
                            "%.1f lookups, %.1f hits; table %.1f MB\n",
                            Gow3Toggle2::Disabled(Gow3Toggle2::ResidencyBitmap) ? "off" : "on",
                            Gow3Stats::residency_checks[0].exchange(0) / frames_d,
                            Gow3Stats::residency_checks[1].exchange(0) / frames_d,
                            Gow3Toggle2::Disabled(Gow3Toggle2::NoImageFastPath) ? "off" : "on",
                            Gow3Stats::no_image_binds.exchange(0) / frames_d,
                            !wide_wanted ? "off"
                            : Gow3Toggle2::Disabled(Gow3Toggle2::TextureSetWide) ? "toggled off"
                                                                                 : "on",
                            d[0], d[1], d[2], d[3], TS::wide_bytes.load() / 1e6);
                    }
                    {
                        // gow3 (impl-stage-a-decode.md): stage A's constant ring bindings and
                        // pending-write checks per frame, A2's filter.
                        namespace SA = Gow3StageA;
                        std::array<double, SA::NumCounters> n{};
                        for (unsigned k = 0; k < SA::NumCounters; ++k) {
                            n[k] = SA::counters[k].exchange(0, std::memory_order_relaxed) / frames_d;
                        }
                        const char* filter = std::getenv("GOW3_PENDING_FILTER");
                        const bool filter_wanted = filter && filter[0] == '1';
                        const double scans = n[SA::pending_checks] - n[SA::pending_bounds] -
                                             n[SA::pending_filtered];
                        std::printf(
                            "Stage A cuts per frame: ring buffers %.1f (reused %.1f, copied %.1f); "
                            "pending-write checks %.1f (bounds %.1f, filter (A2 %s) %.1f, scans "
                            "%.1f of %.1f entries each); register words hashed %.1f (A3 %s)\n",
                            n[SA::ring_bindings], n[SA::ring_reused], n[SA::ring_copied],
                            n[SA::pending_checks], n[SA::pending_bounds],
                            !filter_wanted ? "off"
                            : Gow3Toggle2::Disabled(Gow3Toggle2::PendingWriteFilter) ? "toggled off"
                                                                                     : "on",
                            n[SA::pending_filtered], scans,
                            scans > 0 ? n[SA::pending_scanned] / scans : 0.0, n[SA::reg_words],
                            AmdGpu::Liverpool::PairedRegisterHash() ? "on" : "off");
                    }
                    // gow3 (impl-deck-cpu-bundle.md): the Vulkan recorder's hand-offs, sleeps
                    // and spin; in lean mode (second toggle word 32 clear, <= 8 threads) the
                    // label kicks that left it asleep and how the owed wakes were made.
                    const u64 spins = Gow3Stats::recorder_spins.exchange(0);
                    const u64 spin_ns = Gow3Stats::recorder_spin_ns.exchange(0);
                    std::printf(
                        "Recorder per frame (lean %s): hand-offs %.1f, sleeps %.1f, wakes %.1f; "
                        "label kicks left asleep %.1f, owed wakes made at the deadline %.1f, at "
                        "stage B idle / a wait / a forced kick %.1f, by the backstop %.1f; spin "
                        "%.3f ms (%.1f spins, %.1f found work)\n",
                        Gow3Stats::recorder_lean.load(std::memory_order_relaxed) == 1 ? "on"
                        : Gow3Toggle2::Disabled(Gow3Toggle2::RecorderLean)          ? "toggled off"
                                                                                    : "off",
                        Gow3Stats::recorder_handoffs.exchange(0) / frames_d,
                        Gow3Stats::recorder_sleeps.exchange(0) / frames_d,
                        (wakes[0] + wakes[1]) / frames_d,
                        Gow3Stats::recorder_lazy_kicks.exchange(0) / frames_d,
                        Gow3Stats::recorder_late_wakes[0].exchange(0) / frames_d,
                        Gow3Stats::recorder_late_wakes[1].exchange(0) / frames_d,
                        Gow3Stats::recorder_backstop_wakes.exchange(0) / frames_d,
                        spin_ns / 1e6 / frames_d, spins / frames_d,
                        Gow3Stats::recorder_spin_catches.exchange(0) / frames_d);
                    // gow3 #4 (impl-stage-b-bundle-2.md, gow3_stage_b_wait.h): stage B's blocking
                    // waits by site, the ones over 0.2 ms (the Deck's clock-plateau trigger), and
                    // the bounded spin before them (GOW3_STAGE_B_SPIN=1, second word 1024 off).
                    {
                        namespace SB = Gow3StageBWait;
                        u64 sleeps = 0, sleep_ns = 0, longs[SB::Sites], site_ns[SB::Sites];
                        for (unsigned k = 0; k < SB::Sites; ++k) {
                            sleeps += SB::sleeps[k].exchange(0);
                            site_ns[k] = SB::sleep_ns[k].exchange(0);
                            sleep_ns += site_ns[k];
                            longs[k] = SB::long_sleeps[k].exchange(0);
                        }
                        char spin_state[32];
                        if (!SB::Wanted()) {
                            std::snprintf(spin_state, sizeof(spin_state), "off");
                        } else if (!SB::Enabled()) {
                            std::snprintf(spin_state, sizeof(spin_state), "toggled off");
                        } else {
                            std::snprintf(spin_state, sizeof(spin_state), "on %llu us",
                                          static_cast<unsigned long long>(SB::SpinNs() / 1000));
                        }
                        std::printf(
                            "Stage B sleeps per frame (spin %s): over 0.2 ms %.2f (packet ring "
                            "%.2f, recorder %.2f, GPU waits %.2f, present %.2f); all %.1f (%.3f "
                            "ms: packet ring %.3f, recorder %.3f, GPU waits %.3f, present %.3f); "
                            "spins %.1f (%.3f ms, %.1f ended the wait)\n",
                            spin_state,
                            (longs[SB::PacketRing] + longs[SB::Recorder] + longs[SB::GpuTick] +
                             longs[SB::Present]) / frames_d,
                            longs[SB::PacketRing] / frames_d, longs[SB::Recorder] / frames_d,
                            longs[SB::GpuTick] / frames_d, longs[SB::Present] / frames_d,
                            sleeps / frames_d, sleep_ns / 1e6 / frames_d,
                            site_ns[SB::PacketRing] / 1e6 / frames_d,
                            site_ns[SB::Recorder] / 1e6 / frames_d,
                            site_ns[SB::GpuTick] / 1e6 / frames_d,
                            site_ns[SB::Present] / 1e6 / frames_d,
                            SB::spins.exchange(0) / frames_d, SB::spin_ns.exchange(0) / 1e6 / frames_d,
                            SB::spin_catches.exchange(0) / frames_d);
                    }
                    // gow3 (impl-cpu-waste.md, gow3_drain_wait.h): stage A's drains that waited
                    // for stage B, on today's path (pauses, then sched_yield) or the hybrid one
                    // (GOW3_DRAIN_HYBRID=1: pauses, then a futex sleep; second word 4096 off).
                    {
                        namespace DW = Gow3DrainWait;
                        char state[48];
                        if (!DW::Wanted()) {
                            std::snprintf(state, sizeof(state), "off");
                        } else if (!DW::Enabled()) {
                            std::snprintf(state, sizeof(state), "toggled off");
                        } else {
                            std::snprintf(state, sizeof(state), "on, spin %llu us, wake %u ahead",
                                          static_cast<unsigned long long>(DW::SpinNs() / 1000),
                                          DW::WakeAhead());
                        }
                        const u64 latency_n = DW::wake_latency_n.exchange(0);
                        const u64 latency_ns = DW::wake_latency_ns.exchange(0);
                        const u64 wakes = DW::wakes.exchange(0);
                        const u64 wake_call_ns = DW::wake_call_ns.exchange(0);
                        std::printf(
                            "Stage A drain waits per frame (hybrid %s): %.2f (%.3f ms); today's "
                            "wait: within the pauses %.2f, sched_yield calls %.0f; hybrid: %.2f, "
                            "ended in the spin %.2f, slept %.2f (sleeps %.2f, woken ahead %.2f, then "
                            "ended by the spin %.2f), spin %.3f ms, sleep %.3f ms, wake latency "
                            "%.1f us avg %.1f max, stage B wakes %.2f (%.2f us each)\n",
                            state, DW::waits.exchange(0) / frames_d,
                            DW::wait_ns.exchange(0) / 1e6 / frames_d,
                            DW::pause_only.exchange(0) / frames_d,
                            DW::yields.exchange(0) / frames_d,
                            DW::hybrid_waits.exchange(0) / frames_d,
                            DW::spin_ends.exchange(0) / frames_d, DW::slept.exchange(0) / frames_d,
                            DW::sleeps.exchange(0) / frames_d,
                            DW::woken_ahead.exchange(0) / frames_d,
                            DW::ahead_caught.exchange(0) / frames_d,
                            DW::spin_ns.exchange(0) / 1e6 / frames_d,
                            DW::sleep_ns.exchange(0) / 1e6 / frames_d,
                            latency_n ? latency_ns / 1e3 / latency_n : 0.0,
                            DW::wake_latency_max_ns.exchange(0) / 1e3, wakes / frames_d,
                            wakes ? wake_call_ns / 1e3 / wakes : 0.0);
                    }
                    // gow3 (gow3_signal_wait.h): the recording threads' waits for the recorder
                    // (deferred signals, host copies; GOW3_SIGNAL_SLEEP=1: pauses, then a futex
                    // sleep; second word 16777216 off).
                    {
                        namespace SW = Gow3SignalWait;
                        char state[40];
                        if (!SW::Wanted()) {
                            std::snprintf(state, sizeof(state), "off");
                        } else if (!SW::Enabled()) {
                            std::snprintf(state, sizeof(state), "toggled off");
                        } else {
                            std::snprintf(state, sizeof(state), "on, spin %llu us",
                                          static_cast<unsigned long long>(SW::SpinNs() / 1000));
                        }
                        std::printf("Recorder waits per frame (sleep %s): %.2f (%.3f ms); ended in "
                                    "the spin %.2f, slept %.2f (sleeps %.2f, %.3f ms), wakes %.2f\n",
                                    state, SW::waits.exchange(0) / frames_d,
                                    SW::wait_ns.exchange(0) / 1e6 / frames_d,
                                    SW::spin_ends.exchange(0) / frames_d,
                                    SW::slept.exchange(0) / frames_d,
                                    SW::sleeps.exchange(0) / frames_d,
                                    SW::sleep_ns.exchange(0) / 1e6 / frames_d,
                                    SW::wakes.exchange(0) / frames_d);
                    }
                    // gow3 (impl-texset-index.md): the texture-set memo's lookups by layout
                    // (compact, or the table as before with second word 2048), hits, misses by
                    // cause, sets the slow path kept or not, and the tables' memory.
                    {
                        namespace TS = Gow3TexSet;
                        static u64 last[9]{};
                        const u64 now[9] = {TS::lookups[0].load(),  TS::lookups[1].load(),
                                            TS::hits.load(),        TS::misses[0].load(),
                                            TS::misses[1].load(),   TS::misses[2].load(),
                                            TS::misses[3].load(),   TS::kept.load(),
                                            TS::dropped.load()};
                        double d[9];
                        for (unsigned k = 0; k < 9; ++k) {
                            d[k] = (now[k] - last[k]) / frames_d;
                            last[k] = now[k];
                        }
                        const double lookups = d[0] + d[1];
                        const int huge = TS::compact_huge.load(std::memory_order_relaxed);
                        std::printf(
                            "Texture-set memo per frame (compact layout %s): lookups %.1f "
                            "(compact %.1f, legacy table %.1f), hits %.1f (%.2f%%), misses: "
                            "other key %.1f, generation %.1f, image check %.1f, other %.1f; sets "
                            "kept %.1f, not kept %.1f; memory: compact %.1f MB (huge pages %s), "
                            "legacy table %.1f MB\n",
                            Gow3Toggle2::Disabled(Gow3Toggle2::TextureSetLegacy) ? "toggled off"
                                                                                 : "on",
                            lookups, d[0], d[1], d[2], lookups ? 100.0 * d[2] / lookups : 0.0,
                            d[3], d[4], d[5], d[6], d[7], d[8],
                            TS::compact_bytes.load(std::memory_order_relaxed) / 1e6,
                            huge == 1    ? "yes"
                            : huge == 0  ? "no"
                            : huge == -1 ? "unknown"
                                         : "n/a",
                            TS::legacy_bytes.load(std::memory_order_relaxed) / 1e6);
                    }
                    // gow3 D2 (impl-deck-cpu-bundle.md): scePthreadGetthreadid calls and the ones
                    // that asked the kernel (Linux: once per thread, cached after).
                    const u64 tid_calls =
                        __atomic_exchange_n(&runtime_thread_id_calls, 0, __ATOMIC_RELAXED);
                    const u64 tid_lookups =
                        __atomic_exchange_n(&runtime_thread_id_lookups, 0, __ATOMIC_RELAXED);
                    std::printf("Thread ids per frame: scePthreadGetthreadid %.0f, cache hits "
                                "%.0f, kernel lookups %.2f\n",
                                tid_calls / frames_d,
                                (tid_calls > tid_lookups ? tid_calls - tid_lookups : 0) / frames_d,
                                tid_lookups / frames_d);
                    // gow3 N0 (impl-draw-reduction.md): recorded draws per frame by class
                    // (render targets) and stage B's us per draw of each class.
                    std::fputs(Gow3DrawClasses::Report(frames).c_str(), stdout);
                    // gow3 U2 (impl-u2-direct-arena-writes.md): buffer uploads and precise CPU
                    // writes straight into the mapped arena vs through staging, why staging, what
                    // the direct writes saved; render pass ends and barrier batches of all causes.
                    static u64 window_pass_ends = 0;
                    const u64 pass_ends = Gow3Stats::render_pass_ends.load();
                    const char* u2_state =
                        !Gow3Stats::arena_mapped.load(std::memory_order_relaxed) ? "unavailable"
                        : Gow3Toggle2::Disabled(Gow3Toggle2::ArenaDirectWrites)  ? "off"
                                                                                 : "on";
                    std::printf(
                        "Arena writes per frame (U2 %s): direct %.1f (%.3f MB), staged %.1f (%.3f "
                        "MB); staged because: toggled off %.1f, GPU busy %.1f, staged copy queued "
                        "%.1f, not mapped %.1f, off the recording thread %.1f; saved %.1f copy commands, %.1f pass ends; pass ends "
                        "by staged copies %.1f; all pass ends %.1f, barrier batches %.1f\n",
                        u2_state, Gow3Stats::arena_direct_calls.exchange(0) / frames_d,
                        Gow3Stats::arena_direct_bytes.exchange(0) / 1e6 / frames_d,
                        Gow3Stats::arena_staged_calls.exchange(0) / frames_d,
                        Gow3Stats::arena_staged_bytes.exchange(0) / 1e6 / frames_d,
                        Gow3Stats::arena_fallbacks[Gow3Stats::ArenaToggledOff].exchange(0) / frames_d,
                        Gow3Stats::arena_fallbacks[Gow3Stats::ArenaGpuBusy].exchange(0) / frames_d,
                        Gow3Stats::arena_fallbacks[Gow3Stats::ArenaStagedPending].exchange(0) /
                            frames_d,
                        Gow3Stats::arena_fallbacks[Gow3Stats::ArenaNotMapped].exchange(0) / frames_d,
                        Gow3Stats::arena_fallbacks[Gow3Stats::ArenaOffThread].exchange(0) / frames_d,
                        Gow3Stats::arena_copies_saved.exchange(0) / frames_d,
                        Gow3Stats::arena_pass_ends_saved.exchange(0) / frames_d,
                        Gow3Stats::arena_staged_pass_ends.exchange(0) / frames_d,
                        (pass_ends - window_pass_ends) / frames_d,
                        Gow3Stats::barrier_flushes.exchange(0) / frames_d);
                    window_pass_ends = pass_ends;
                }
                if (std::getenv("GOW3_STREAM_CLEAN")) {
                    std::printf("Small read-only buffers/s: %.0f bound from the arena, %.0f uploaded "
                                "once, %.0f stream copies, %.0f found dynamic\n",
                                Gow3Stats::small_arena_binds.exchange(0) / window,
                                Gow3Stats::small_promotions.exchange(0) / window,
                                Gow3Stats::small_stream_copies.exchange(0) / window,
                                Gow3Stats::small_dynamic.exchange(0) / window);
                }
                // Synchronous buffer downloads (each a full GPU wait), by cause.
                static constexpr const char* reasons[] = {"read fault", "write fault",
                                                          "invalidation", "unmap"};
                std::printf("Buffer downloads/s (GPU waits):");
                for (unsigned r = 0; r < Gow3Stats::DownloadReasons; ++r) {
                    std::printf(" %s %.1f (%.2f MB, %.1f ms)", reasons[r],
                                Gow3Stats::download_count[r].exchange(0) / window,
                                Gow3Stats::download_bytes[r].exchange(0) / 1e6 / window,
                                Gow3Stats::download_ns[r].exchange(0) / 1e6 / window);
                }
                std::printf("; unmaps discarded %.1f; precise CPU writes %.1f (%.3f MB); on the readback "
                            "queue %.1f (finished page only %.1f); twinned (no wait) %.1f; write "
                            "faults with no GPU bytes on the page %.1f, left to the window: unknown "
                            "tick %.1f, too big for a twin %.1f\n",
                            Gow3Stats::unmap_discards.exchange(0) / window,
                            Gow3Stats::precise_writes.exchange(0) / window,
                            Gow3Stats::precise_write_bytes.exchange(0) / 1e6 / window,
                            Gow3Stats::fast_downloads.exchange(0) / window,
                            Gow3Stats::page_downloads.exchange(0) / window,
                            Gow3Stats::twin_faults.exchange(0) / window,
                            Gow3Stats::fault_no_gpu_bytes.exchange(0) / window,
                            Gow3Stats::fault_unknown_tick.exchange(0) / window,
                            Gow3Stats::fault_too_big_for_twin.exchange(0) / window);
                VideoCore::BufferCache::ReportTwins();
                if (Vulkan::WaitDiagnostics::Enabled()) {
                    VideoCore::BufferCache::ReportDownloadSites(window);
                }
                window_image_bytes = image_bytes;
                window_buffer_bytes = buffer_bytes;
            }
            presenter->GetRasterizer().GetPipelineCache().ReportCompilerStats();
            Vulkan::WaitDiagnostics::Report();
            // Frame pacing: spread of the guest flip intervals (judder that the mean hides).
            if (intervals.size() > 2) {
                std::vector<double> sorted = intervals;
                std::sort(sorted.begin(), sorted.end());
                const double median = sorted[sorted.size() / 2];
                pacing_median = median;
                double sum = 0, sq = 0;
                u32 spikes = 0;
                // Spike length: consecutive frames over 1.5x the median, and their time.
                u32 run = 0, longest_run = 0;
                double run_ms = 0, longest_run_ms = 0;
                for (const double ms : intervals) {
                    sum += ms;
                    sq += ms * ms;
                    if (ms > 1.5 * median) {
                        ++spikes;
                        ++run;
                        run_ms += ms;
                        if (run_ms > longest_run_ms) {
                            longest_run = run;
                            longest_run_ms = run_ms;
                        }
                    } else {
                        run = 0;
                        run_ms = 0;
                    }
                }
                const double mean = sum / intervals.size();
                std::printf("Frame pacing: median %.2f ms, stddev %.2f ms, p99 %.2f ms, "
                            "%u frames over 1.5x median, longest run %u frames (%.1f ms)",
                            median, std::sqrt(std::max(0.0, sq / intervals.size() - mean * mean)),
                            sorted[std::min(sorted.size() - 1, sorted.size() * 99 / 100)], spikes,
                            longest_run, longest_run_ms);
                // The window's slowest 2% (at least one): pooled over windows, a log reader gets the
                // 1% and 0.1% lows of a whole state (deck_summary.py), not one window's few frames.
                const std::size_t slowest = std::min(sorted.size(), sorted.size() / 50 + 1);
                std::printf("; %zu frames, slowest", sorted.size());
                for (std::size_t i = 0; i < slowest; ++i) {
                    std::printf(" %.2f", sorted[sorted.size() - 1 - i]);
                }
                std::printf(" ms\n");
            }
            intervals.clear();
            window_reset = true;
            window_start = now;
            frames = 0;
            worst_ms = 0;
            stalls_shown = 0;
            std::fill(std::begin(last_wait) + 1, std::begin(last_wait) + 6, 0);
        }
    }

    if (!req.completed) {
        CompleteFlip(req);
    }
}

static void PreciseWait(std::chrono::steady_clock::time_point slot,
                        std::chrono::microseconds spin = std::chrono::microseconds(100));

// gow3 GOW3_VBLANK_SYNC=1: the vblank tick follows the display's own vertical blank (DXGI
// WaitForVBlank on the primary output, the one SDL reports the refresh rate of), every Nth one
// for a tick below the refresh. T-V120 (120 Hz tick, VSync, 240 Hz display): frames held for
// exactly two refreshes were steady (p99 8.42 ms), but the tick ran on the PC clock, slid
// against the display and every ~2.5 min crossed a refresh boundary: waves of 12.5 / 4.17 ms
// pairs that grew denser and faded. Returns null (PC clock) when there is no output.
class DisplayVblank {
public:
    static DisplayVblank* Create(u32 tick_hz) {
#ifdef _WIN32
        const HMODULE dxgi = LoadLibraryA("dxgi.dll");
        using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
        const auto create = dxgi ? reinterpret_cast<CreateFactory>(
                                       reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")))
                                 : nullptr;
        IDXGIFactory1* factory = nullptr;
        if (!create || FAILED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
            return nullptr;
        }
        const HMONITOR primary = MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
        IDXGIOutput* found = nullptr;
        IDXGIAdapter1* adapter = nullptr;
        for (UINT a = 0; !found && factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
            IDXGIOutput* output = nullptr;
            for (UINT o = 0; !found && adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o) {
                DXGI_OUTPUT_DESC desc{};
                if (SUCCEEDED(output->GetDesc(&desc)) && desc.Monitor == primary) {
                    found = output;
                } else {
                    output->Release();
                }
            }
            adapter->Release();
        }
        factory->Release();
        if (!found) {
            std::printf("VideoOut: vblank sync: no display output, the PC clock paces\n");
            return nullptr;
        }
        const u32 refresh = Gow3DisplayRefreshHz();
        const u32 every = std::max(1u, (refresh + tick_hz / 2) / std::max(tick_hz, 1u));
        std::printf("VideoOut: vblank sync: tick on every %u of the display's vblanks (%u Hz)\n",
                    every, refresh);
        return new DisplayVblank(found, every);
#else
        return nullptr;
#endif
    }
    void Wait() {
#ifdef _WIN32
        for (u32 i = 0; i < every; ++i) {
            output->WaitForVBlank();
        }
#endif
    }

private:
#ifdef _WIN32
    DisplayVblank(IDXGIOutput* output_, u32 every_) : output(output_), every(every_) {}
    IDXGIOutput* output;
#endif
    u32 every = 1;
};

void VideoOutDriver::DropOlderFinished() {
    std::vector<Request> stale;
    {
        std::scoped_lock lock{mutex};
        size_t finished = 0; // frames finish in queue order
        while (finished < requests.size() &&
               presenter->WaitFrameReady(requests[finished].frame, {})) {
            ++finished;
        }
        while (finished > 1) {
            stale.push_back(requests.front());
            requests.pop_front();
            --finished;
        }
    }
    for (const auto& request : stale) {
        DropFlip(request);
    }
}

void VideoOutDriver::PacerThread(std::stop_token token) {
    Common::SetCurrentThreadName("gow3:Pacer");
    BoostPresentingThread("pacer");
    Common::SetCurrentThreadRealtime(std::chrono::milliseconds(8));
    auto next = std::chrono::steady_clock::now();
    u32 limit = 0;
    // The last part of the wait spins (GOW3_PACER_SPIN_US, default 700: past the timer's grid).
    const auto spin = std::chrono::microseconds([] {
        const char* env = std::getenv("GOW3_PACER_SPIN_US");
        return env ? std::atol(env) : 700L;
    }());
    while (!token.stop_requested()) {
        if (const u32 now_limit = pacer_limit.load(std::memory_order_relaxed); now_limit != limit) {
            limit = now_limit;
            next = std::chrono::steady_clock::now();
        }
        if (!limit) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        const auto period = std::chrono::nanoseconds(1000000000 / limit);
        PreciseWait(next, spin);
        // The newest finished frame; later than the slot only when none has finished yet.
        Request show{};
        while (!token.stop_requested() && pacer_limit.load(std::memory_order_relaxed) == limit) {
            DropOlderFinished();
            Request front{};
            {
                std::unique_lock lock{mutex};
                if (requests.empty()) {
                    request_cv.wait_for(lock, std::chrono::milliseconds(2),
                                        [&] { return !requests.empty(); });
                    continue;
                }
                front = requests.front();
                if (presenter->WaitFrameReady(front.frame, {})) {
                    requests.pop_front();
                    show = front;
                    break;
                }
            }
            presenter->WaitFrameReady(front.frame,
                                      std::chrono::steady_clock::now() + std::chrono::milliseconds(2));
        }
        if (!show) {
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        next = std::max(next + period, now - period);
        Flip(show);
        FRAME_END;
    }
}

void VideoOutDriver::DropFlip(const Request& req) {
    Gow3FlipTrace::Note("dropped", req.flip_arg);
    if (req.frame != nullptr) {
        presenter->DiscardFrame(req.frame);
    }
    if (!req.completed) {
        CompleteFlip(req);
    }
}

void VideoOutDriver::CompleteFlip(const Request& req) {
    // Update flip status.
    auto* port = req.port;
    {
        std::unique_lock lock{port->port_mutex};
        auto& flip_status = port->flip_status;
        flip_status.count++;
        flip_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
        flip_status.tsc = Libraries::Kernel::sceKernelReadTsc();
        flip_status.flip_arg = req.flip_arg;
        flip_status.current_buffer = req.index;
        if (req.eop) {
            --flip_status.gc_queue_num;
        }
        --flip_status.flip_pending_num;
    }

    // Trigger flip events for the port.
    for (auto event : port->flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->TriggerEvent(
                static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                Kernel::OrbisKernelEvent::Filter::VideoOut,
                reinterpret_cast<void*>(static_cast<u64>(OrbisVideoOutInternalEventId::Flip) |
                                        (req.flip_arg << 16)));
        }
    }

    Gow3FlipTrace::Note("flipdone", req.flip_arg);
    // Reset prev flip label
    if (port->prev_index != -1) {
        port->buffer_labels[port->prev_index] = 0;
        port->SignalVoLabel();
    }
    // save to prev buf index
    port->prev_index = req.index;
}

void VideoOutDriver::DrawBlankFrame() {
    RunPresenter([this] {
        const auto empty_frame = presenter->PrepareBlankFrame(true);
        presenter->Present(empty_frame, false, false);
    }, true);
}

void VideoOutDriver::DrawLastFrame() {
    RunPresenter([this] {
        const auto frame = presenter->PrepareLastFrame();
        if (frame != nullptr) {
            presenter->Present(frame, true);
        }
    }, true);
}

bool VideoOutDriver::SubmitFlip(VideoOutPort* port, s32 index, s64 flip_arg,
                                bool is_eop /*= false*/) {
    Gow3FlipTrace::Note(is_eop ? "irq" : "submit_direct", flip_arg);
    {
        std::unique_lock lock{port->port_mutex};
        if (index != -1 && port->flip_status.flip_pending_num > 16) {
            LOG_ERROR(Lib_VideoOut, "Flip queue is full");
            return false;
        }

        if (is_eop) {
            ++port->flip_status.gc_queue_num;
        }
        ++port->flip_status.flip_pending_num; // integral GPU and CPU pending flips counter
        port->flip_status.submit_tsc = Libraries::Kernel::sceKernelReadTsc();
    }

    if (!is_eop) {
        // Non EOP flips can arrive from any thread so ask GPU thread to perform them
        liverpool->SendCommand([=, this]() { SubmitFlipInternal(port, index, flip_arg, is_eop); });
    } else {
        SubmitFlipInternal(port, index, flip_arg, is_eop);
    }

    return true;
}

void VideoOutDriver::SubmitFlipInternal(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop) {
    Vulkan::Frame* frame;
    if (index == -1) {
        frame = presenter->PrepareBlankFrame(false);
    } else {
        const auto& buffer = port->buffer_slots[index];
        ASSERT_MSG(buffer.group_index >= 0, "Trying to flip an unregistered buffer!");
        const auto& group = port->groups[buffer.group_index];
        frame = presenter->PrepareFrame(group, buffer.address_left);
        presenter->TraceGpuReady(frame, flip_arg); // flip trace only
    }

    // gow3 newest-frame mode: the frame is copied out of the guest's buffer (PrepareFrame), so
    // its flip completes now: the guest's next frame no longer waits for the cap (T-L5: with
    // completion at present, the game waited for its buffer and no frame was ever dropped).
    const bool early = newest_active.load(std::memory_order_relaxed);
    const Request request{
        .frame = frame,
        .port = port,
        .flip_arg = flip_arg,
        .index = index,
        .eop = is_eop,
        .completed = early,
    };
    {
        std::scoped_lock lock{mutex};
        requests.push_back(request);
        Gow3FlipTrace::Note("queued", flip_arg);
    }
    if (early) {
        CompleteFlip(request);
        DropOlderFinished(); // frames back to the pool at once (the pool bounds the game)
    }
    request_cv.notify_all(); // the present thread and the pacer
}

// gow3: waits until `slot` within a few microseconds: a high-resolution waitable timer (one per
// thread) to `spin` before it, then a spin. The timer wakes on a ~0.5 ms grid here (T-L5 run 3:
// with a 0.1 ms spin the pacer's releases ran 0-0.5 ms late in a sawtooth, 8.33 mod 0.5 per frame).
static void PreciseWait(std::chrono::steady_clock::time_point slot, std::chrono::microseconds spin) {
#ifdef _WIN32
    thread_local HANDLE hires = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /* HIGH_RESOLUTION */,
                                                 TIMER_ALL_ACCESS);
    const auto coarse = slot - spin;
    if (const auto now = std::chrono::steady_clock::now(); hires && now < coarse) {
        LARGE_INTEGER due;
        due.QuadPart = -std::max<LONGLONG>(
            1, std::chrono::duration_cast<std::chrono::nanoseconds>(coarse - now).count() / 100);
        if (SetWaitableTimerEx(hires, &due, 0, nullptr, nullptr, nullptr, 0)) {
            WaitForSingleObject(hires, INFINITE);
        }
    }
#else
    std::this_thread::sleep_until(slot - spin);
#endif
    while (std::chrono::steady_clock::now() < slot) _mm_pause();
}

void VideoOutDriver::PresentThread(std::stop_token token) {
    const std::chrono::nanoseconds vblank_period(1000000000 /
                                                 EmulatorSettings.GetVblankFrequency());

    Common::SetCurrentThreadName("shadPS4:PresentThread");
    BoostPresentingThread("vblank");
    Common::SetCurrentThreadRealtime(vblank_period);
    // gow3 #8 (Linux, opt-in): GOW3_THREAD_NICE (nice, 1 ns timer slack).
    Gow3LinuxSched::VBlankThread();

    Common::AccurateTimer timer{vblank_period};

    // gow3: frame limit (see EmulatorSettings::GetFrameLimit). A request waits in the queue
    // until its slot; slots advance by one period (no drift) but never lag behind by more.
    // gow3: the in-game menu changes the limit live (Gow3Settings::fps_limit); GOW3_FPS_LIMIT
    // and the uncapped presets keep their fixed limit.
    // MediEvil: in the uncapped mode a lower in-game limit still applies on top.
    const bool env_limit = std::getenv("GOW3_FPS_LIMIT") != nullptr;
    const bool fixed_limit = env_limit || EmulatorSettings.GetVblankFrequency() == 480;
    // A limit at or above the vblank rate adds nothing (the vblank already paces the flips) and
    // a second clock of the same period beat against it: a flip ready at a vblank a moment
    // before its slot waited a whole extra vblank (stutter with a 60 FPS limit at 60 Hz,
    // MediEvil issue #3).
    const u32 vblank_hz = EmulatorSettings.GetVblankFrequency();
    // Without the driver's Reflex (AMD, Intel) the port's own limiter paces (run.py also sets the
    // precise limiter's switches for that case).
    const bool reflex = [this] {
        const char* env = std::getenv("GOW3_REFLEX");
        return env && env[0] == '1' && presenter->ReflexEnabled();
    }();
    u32 reflex_interval = ~0u;
    const auto current_limit = [&] {
        // -1: half the display refresh, so each frame shows for exactly two refreshes.
        const int raw = Gow3Settings::Get().fps_limit.load();
        const u32 user = u32(std::max(0, Gow3Graphics::ResolveFpsLimit(raw, Gow3DisplayRefreshHz(),
                                                                             Gow3Settings::Get().display_vrr)));
        u32 limit = user;
        if (fixed_limit) {
            const u32 base = EmulatorSettings.GetFrameLimit();
            limit = !env_limit && user && (!base || user < base) ? user : base;
        }
        // The toggle file's third word (diagnostics): a cap in FPS in place of the limit, live.
        if (const u64 cap = __atomic_load_n(&runtime_toggle_frame_cap, __ATOMIC_RELAXED)) {
            limit = u32(std::min<u64>(cap, 1000));
        }
        limit = limit >= vblank_hz ? 0u : limit;
        // With VSync the display paces the presents too: a limit at its refresh rate is another
        // clock beside the display's own (often 59.94 Hz) and repeats or drops a frame now and
        // then (issue #3, 4K TV at 60 Hz); the display alone keeps it even.
        if (Gow3Settings::Get().vsync.load() && limit + 1 >= Gow3DisplayRefreshHz()) {
            limit = 0u;
        }
        // gow3 GOW3_REFLEX=1: the driver paces (its minimum interval); our limiter steps aside so
        // there is one clock (T-F13: two limiters at one rate fight).
        if (reflex) {
            const u32 interval = limit ? 1000000u / limit : 0u;
            if (interval != reflex_interval) {
                reflex_interval = interval;
                RunPresenter([this, interval] { presenter->SetLatencyInterval(interval); });
            }
            return 0u;
        }
        return limit;
    };
    // Slots count as reached within half a vblank: timer jitter must not cost a whole vblank.
    // gow3 GOW3_PRECISE_LIMIT=1: frames are released at their slot, not up to half a vblank
    // early or a tick late. T-L2 (flip trace, 120 cap, G-Sync): the game's frames waited ~8 ms
    // for their slot, yet releases came 6.3-10.4 ms apart (8.33 +- one 480 Hz tick): the 0.1%
    // lows at 80-90 FPS. The wait is a high-resolution waitable timer (no spin: the first
    // version spun up to 1 ms on this realtime thread, T-L1b), the last ~0.1 ms a short spin.
    // gow3 GOW3_NEWEST_FRAME=1 (implies the precise limit): the game renders uncapped and each
    // slot shows the newest frame finished on the GPU; older ones complete unshown (DropFlip).
    static const bool newest_frame = [] {
        const char* env = std::getenv("GOW3_NEWEST_FRAME");
        return env && env[0] == '1';
    }();
    static const bool precise_limit = [] {
        const char* env = std::getenv("GOW3_PRECISE_LIMIT");
        return newest_frame || (env && env[0] == '1');
    }();
    const auto slot_slack = precise_limit ? std::chrono::nanoseconds(100000)
                                          : std::chrono::nanoseconds(500000000 / std::max(vblank_hz, 1u));
    const auto wait_slot = [](std::chrono::steady_clock::time_point slot) { PreciseWait(slot); };
    // gow3 GOW3_LIMIT_IN_PRESENT=1 (with the precise limiter): frames are released when they arrive
    // and the present waits for the slot after its acquire and blit, as the RTSS limiter does.
    static const bool limit_in_present = [] {
        const char* env = std::getenv("GOW3_LIMIT_IN_PRESENT");
        return env && env[0] == '1';
    }();
    static const bool limit_hold = [] {
        const char* env = std::getenv("GOW3_LIMIT_HOLD");
        return env && env[0] == '1';
    }();
    const auto slot_spin = std::chrono::microseconds([] {
        const char* env = std::getenv("GOW3_PACER_SPIN_US");
        return env ? std::atol(env) : 700L;
    }());
    // gow3 GOW3_PRECISE_TICK=1: the vblank tick runs on a fixed grid (no drift) and waits with
    // PreciseWait (high-resolution timer, then a spin of GOW3_TICK_SPIN_US, default 700) instead
    // of AccurateTimer's plain waitable timer. T-V120 (120 Hz tick, VSync): the steadiest pacing
    // yet, but ticks came +-0.7 ms off, and waves of 12.5 / 4.17 ms frames.
    static const bool precise_tick = [] {
        const char* env = std::getenv("GOW3_PRECISE_TICK");
        return env && env[0] == '1';
    }();
    const auto tick_spin = std::chrono::microseconds([] {
        const char* env = std::getenv("GOW3_TICK_SPIN_US");
        return env ? std::atol(env) : 700L;
    }());
    auto next_tick = std::chrono::steady_clock::now();
    if (precise_tick) {
        std::printf("VideoOut: precise vblank tick (spin %lld us)\n",
                    static_cast<long long>(tick_spin.count()));
    }
    static DisplayVblank* const display_vblank = [&]() -> DisplayVblank* {
        const char* env = std::getenv("GOW3_VBLANK_SYNC");
        return env && env[0] == '1' ? DisplayVblank::Create(vblank_hz) : nullptr;
    }();
    // gow3 GOW3_ONE_CLOCK=1 (with the precise tick and a frame limit): one clock paces everything.
    // The tick runs at limit x N (N whole ticks a frame, N = round(vblank Hz / limit)) and frames
    // are released on ticks only: on the Nth tick after the last release, else on the first tick
    // a late frame arrives (the count restarts there; the grid never moves). T-F13: two limiters
    // at one rate with independent phases (ours and RTSS at 120) fought and flickered, RTSS
    // alone (at 119, ours idle) held 8.403 +- 0.001 ms; our limiter's slots and the 960 Hz tick
    // were two such clocks.
    static const bool one_clock = [] {
        const char* env = std::getenv("GOW3_ONE_CLOCK");
        return env && env[0] == '1';
    }();
    u32 one_clock_ticks = 0; // N; 0: off
    // The grid: tick k is at grid_start + k * tick_period, never re-anchored (T-F14: re-anchoring on
    // a late tick moved the "one clock"; frames left after 7 or 9 ticks). Slots count grid ticks.
    std::chrono::steady_clock::time_point grid_start{};
    u64 grid_tick = 0, last_release_tick = 0;
    auto tick_period = std::chrono::nanoseconds(vblank_period);
    u32 frame_limit = current_limit();
    auto frame_period = frame_limit ? std::chrono::nanoseconds(1000000000 / frame_limit)
                                    : std::chrono::nanoseconds(0);
    auto next_flip = std::chrono::steady_clock::now();
    const auto set_one_clock = [&](u32 limit) {
        one_clock_ticks = 0;
        tick_period = std::chrono::nanoseconds(vblank_period);
        grid_start = {};
        grid_tick = last_release_tick = 0;
        if (one_clock && precise_tick && limit) {
            one_clock_ticks = std::max(1u, (vblank_hz + limit / 2) / limit);
            tick_period = std::chrono::nanoseconds(1000000000ll / (s64(limit) * one_clock_ticks));
            std::printf("VideoOut: one clock: %u ticks a frame at %u FPS (tick %.3f us)\n", one_clock_ticks,
                        limit, tick_period.count() / 1000.0);
        }
    };
    std::printf("VideoOut: vblank %u Hz, frame limit %u FPS\n",
                EmulatorSettings.GetVblankFrequency(), frame_limit);
    set_one_clock(frame_limit);

    const auto receive_request = [this] -> Request {
        std::scoped_lock lk{mutex};
        if (!requests.empty()) {
            const auto request = requests.front();
            requests.pop_front();
            return request;
        }
        return {};
    };

    // gow3: with a frame limit (uncapped presets) a queued flip is presented as soon as it
    // arrives and its slot allows, between vblanks, instead of on the next vblank tick.
    // gow3 Reflex: frames go to the driver the moment they are queued (T-F17: released on the next
    // 960 Hz tick instead, up to 1.1 ms late, the presents reached the driver 7.9-8.9 ms apart).
    bool immediate_flips = frame_limit != 0 || reflex;

    while (!token.stop_requested()) {
        Gow3LinuxSched::PollLive(); // GOW3_THREAD_NICE live (toggle second word 32768)
        if (const u32 limit = current_limit(); limit != frame_limit) {
            frame_limit = limit;
            frame_period = limit ? std::chrono::nanoseconds(1000000000 / limit) : std::chrono::nanoseconds(0);
            immediate_flips = limit != 0 || reflex;
            next_flip = std::chrono::steady_clock::now();
            std::printf("VideoOut: frame limit %u FPS\n", limit);
            set_one_clock(limit);
        }
        newest_active.store(newest_frame && frame_limit != 0, std::memory_order_relaxed);
        pacer_limit.store(newest_frame ? frame_limit : 0, std::memory_order_relaxed);
        if (display_vblank) {
            display_vblank->Wait();
        } else if (precise_tick && one_clock_ticks && frame_limit) {
            const auto now_tick = std::chrono::steady_clock::now();
            if (grid_start == std::chrono::steady_clock::time_point{}) {
                grid_start = now_tick;
            }
            ++grid_tick;
            if (grid_start + tick_period * grid_tick < now_tick - tick_period) {
                // a whole tick missed: the next grid tick, same phase
                grid_tick = u64((now_tick - grid_start) / tick_period) + 1;
            }
            PreciseWait(grid_start + tick_period * grid_tick, tick_spin);
        } else if (precise_tick) {
            PreciseWait(next_tick, tick_spin);
            next_tick = std::max(next_tick + tick_period, std::chrono::steady_clock::now() - tick_period);
        } else {
            timer.Start();
        }
        const auto tick_deadline = std::chrono::steady_clock::now() + vblank_period;

        if (DebugState.IsGuestThreadsPaused()) {
            DrawLastFrame();
            timer.End();
            continue;
        }

        // Check if it's time to take a request.
        auto& vblank_status = main_port.vblank_status;
        const auto now = std::chrono::steady_clock::now();
        // Precise: capped frames are released by the loop below at their slot, never at a tick
        // (the first tick after a slot was up to a tick late, T-L3).
        // Without a queued frame the tick still redraws (the save list and menus stay live while
        // the game draws nothing; skipping that froze the save list, T-L4).
        const auto redraw_idle = [&] {
            // gow3: the save list / menu / text entry stay live while the game draws nothing.
            static auto last_redraw = std::chrono::steady_clock::now();
            if (main_port.is_open && Gow3Overlay::CapturesInput() &&
                now - std::max(last_redraw, LastGameFlip()) >= std::chrono::milliseconds(16)) {
                last_redraw = now;
                DrawLastFrame();
            } else if (timer.GetTotalWait().count() < 0) { // Dont draw too fast
                if (!main_port.is_open) {
                    DrawBlankFrame();
                } else if (ImGui::Core::MustKeepDrawing()) {
                    DrawLastFrame();
                }
            }
        };
        const bool flip_slot = !frame_limit || (!precise_limit && now + slot_slack >= next_flip);
        const bool precise_capped = precise_limit && frame_limit;
        if (one_clock_ticks && frame_limit) {
            if (grid_tick >= last_release_tick + one_clock_ticks) {
                if (const Request request = receive_request()) {
                    last_release_tick = grid_tick;
                    Flip(request);
                    FRAME_END;
                } else {
                    redraw_idle();
                }
            }
        } else if ((flip_slot || precise_capped) && vblank_status.count % (main_port.flip_rate + 1) == 0) {
            Request request{};
            bool pending = false;
            if (flip_slot) {
                request = receive_request();
            } else {
                std::scoped_lock lk{mutex};
                pending = !requests.empty();
            }
            if (request && frame_limit) {
                next_flip = std::max(next_flip + frame_period, now - frame_period);
            }
            if (pending) {
                // released by the slot loop below
            } else if (!request) {
                // gow3: the save list / menu / text entry stay live while the game draws nothing.
                static auto last_redraw = std::chrono::steady_clock::now();
                if (main_port.is_open && Gow3Overlay::CapturesInput() &&
                    now - std::max(last_redraw, LastGameFlip()) >= std::chrono::milliseconds(16)) {
                    last_redraw = now;
                    DrawLastFrame();
                } else if (timer.GetTotalWait().count() < 0) { // Dont draw too fast
                    if (!main_port.is_open) {
                        DrawBlankFrame();
                    } else if (ImGui::Core::MustKeepDrawing()) {
                        DrawLastFrame();
                    }
                }
            } else {
                Flip(request);
                FRAME_END;
            }
        }

        {
            // Needs lock here as can be concurrently read by `sceVideoOutGetVblankStatus`
            std::scoped_lock lock{main_port.vo_mutex};

            // Trigger flip events for the port
            for (auto event : main_port.vblank_events) {
                auto equeue = Kernel::GetEqueue(event);
                if (equeue != nullptr) {
                    equeue->TriggerEvent(
                        static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                        Kernel::OrbisKernelEvent::Filter::VideoOut,
                        reinterpret_cast<void*>(
                            static_cast<u64>(OrbisVideoOutInternalEventId::Vblank) |
                            (vblank_status.count << 16)));
                }
            }

            // Update vblank status
            vblank_status.count++;
            vblank_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
            vblank_status.tsc = Libraries::Kernel::sceKernelReadTsc();
            main_port.vblank_cv.notify_all();
        }

        if (!immediate_flips || main_port.flip_rate != 0 || (one_clock_ticks && frame_limit)) {
            timer.End();
            continue;
        }
        if (newest_active.load(std::memory_order_relaxed)) {
            // the pacer thread releases frames
        } else
        while (!token.stop_requested()) {
            {
                std::unique_lock lk{mutex};
                if (!request_cv.wait_until(lk, tick_deadline,
                                           [&] { return !requests.empty(); })) {
                    break; // next vblank
                }
            }
            if (std::chrono::steady_clock::now() + slot_slack < next_flip) {
                // gow3 GOW3_LIMIT_HOLD=1 (with the precise limiter): a frame waiting for its slot
                // holds this thread until the slot, with no vblank ticks or events in between, as
                // the RTSS limiter does inside the present (T-F7/T-F8: with RTSS the game's submits
                // were 8.0-8.6 ms apart and the lows rose; with our ticks running on during the
                // wait, 7.5-9.1 ms).
                if (precise_limit && limit_in_present) {
                    // released now; the present waits for the slot (Presenter::SetPresentDeadline)
                    flip_deadline = next_flip;
                } else if (next_flip - slot_slack >= tick_deadline && !(precise_limit && limit_hold)) {
                    break;
                }
                if (precise_limit && limit_in_present) {
                } else if (precise_limit) {
                    // The slot itself: the long final spin (the timer wakes on a ~0.5 ms grid;
                    // with 0.1 ms the releases spread 7.6-9.1 ms at a 120 cap, T-F7, while the
                    // RTSS limiter held 8.33; GOW3_PACER_SPIN_US, default 700).
                    PreciseWait(next_flip, slot_spin);
                } else {
                    std::this_thread::sleep_until(next_flip - slot_slack);
                }
            }
            const auto now = std::chrono::steady_clock::now();
            const auto request = receive_request();
            if (request) {
                next_flip = std::max(next_flip + frame_period, now - frame_period);
                Flip(request);
                FRAME_END;
            }
        }
        if (precise_limit) {
            wait_slot(tick_deadline); // the next window starts on time (Sleep overshoots ~1 ms)
        } else {
            std::this_thread::sleep_until(tick_deadline);
        }
    }
}

} // namespace Libraries::VideoOut
