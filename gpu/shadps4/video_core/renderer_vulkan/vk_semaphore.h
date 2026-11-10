// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <queue>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

class Semaphore {
public:
    explicit Semaphore(const Instance& instance_);
    ~Semaphore();

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return KnownGpuTick() >= tick;
    }

    [[nodiscard]] u64 NextTick() noexcept {
        return current_tick.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] vk::Semaphore Handle() const noexcept {
        return semaphore.get();
    }

    /// Refresh the known GPU tick
    void Refresh();

    /// gow3: steady-clock nanoseconds (the stamps of RefreshedWithin).
    [[nodiscard]] static u64 NowNs() noexcept {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }

    /// gow3: whether some thread asked the GPU for its tick (Refresh, also at the end of Wait)
    /// less than `interval_ns` ago: KnownGpuTick() is then at most that old.
    [[nodiscard]] bool RefreshedWithin(u64 interval_ns) const noexcept {
        return NowNs() - refreshed_ns.load(std::memory_order_relaxed) < interval_ns;
    }

    /// Waits for a tick to be hit on the GPU
    void Wait(u64 tick);

protected:
    const Instance& instance;
    vk::UniqueSemaphore semaphore;    ///< Timeline semaphore.
    std::atomic<u64> gpu_tick{0};     ///< Current known GPU tick.
    std::atomic<u64> current_tick{1}; ///< Current logical tick.
    std::atomic<u64> refreshed_ns{0}; ///< gow3: NowNs() at the end of the last Refresh.
};

} // namespace Vulkan
