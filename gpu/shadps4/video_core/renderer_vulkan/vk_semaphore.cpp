// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#include "gow3_sections.h"
#include "gow3_stage_b_wait.h"
#include "gow3_toggles.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

#include "common/assert.h"

namespace Vulkan {

constexpr u64 WAIT_TIMEOUT = std::numeric_limits<u64>::max();

Semaphore::Semaphore(const Instance& instance_) : instance{instance_} {
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] =
        instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess, "Failed to create master semaphore: {}",
               vk::to_string(semaphore_result));
    semaphore = std::move(sem);
}

Semaphore::~Semaphore() = default;

void Semaphore::Refresh() {
    const u64 start_ns = Gow3Stats::enabled ? NowNs() : 0;
    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        ASSERT_MSG(counter_result == vk::Result::eSuccess,
                   "Failed to get master semaphore value: {}", vk::to_string(counter_result));
        counter = cntr;
        if (counter < this_tick) {
            break;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
    // gow3 (study-stage-b-per-draw.md C1): when the GPU's tick was last asked, by any thread
    // (RefreshedWithin), and the query's cost by thread kind (GOW3_FRAME_STATS).
    const u64 end_ns = NowNs();
    refreshed_ns.store(end_ns, std::memory_order_relaxed);
    if (Gow3Stats::enabled) {
        const unsigned kind = Gow3Sections::recording_thread ? 0 : 1;
        Gow3Stats::semaphore_queries[kind].fetch_add(1, std::memory_order_relaxed);
        Gow3Stats::semaphore_query_ns[kind].fetch_add(end_ns - start_ns, std::memory_order_relaxed);
    }
}

void Semaphore::Wait(u64 tick) {
    // No need to wait if the GPU is ahead of the tick
    if (IsFree(tick)) {
        return;
    }
    // Update the GPU tick and try again
    Refresh();
    if (IsFree(tick)) {
        return;
    }

    // gow3 #4 (gow3_stage_b_wait.h): stage B with GOW3_STAGE_B_SPIN=1 asks again every 20 us
    // (a query is an ioctl on RADV, ~5 us on the Deck) before it sleeps.
    if (Gow3StageBWait::Spin(
            [this, tick] {
                Refresh();
                return IsFree(tick);
            },
            20'000)) {
        return;
    }

    // If none of the above is hit, fallback to a regular wait
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    {
        Gow3StageBWait::SleepTimer sleep{Gow3StageBWait::GpuTick};
        while (instance.GetDevice().waitSemaphores(&wait_info, WAIT_TIMEOUT) !=
               vk::Result::eSuccess) {
        }
    }
    Refresh();
}

} // namespace Vulkan
