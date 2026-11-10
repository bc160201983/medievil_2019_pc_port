// SPDX-License-Identifier: GPL-2.0-or-later
// DrawPipe::Drain with GOW3_DRAIN_HYBRID (gpu/shim/gow3_drain_wait.h): every drain returns only
// once stage B has run every committed packet, on both paths (second toggle word 4096 flipped
// while it runs), with drains long enough to sleep and stage B idle long enough to sleep too.
// A watchdog fails the test on a lost wake. GOW3_FRAME_STATS=1 prints the counters.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>

#include "video_core/renderer_vulkan/vk_draw_pipe.h"

// The runtime's toggle words and fault recovery point (no guest process here).
extern "C" {
std::uint64_t runtime_disabled_optimizations = 0;
std::uint64_t runtime_disabled_optimizations2 = 0;
__thread sigjmp_buf* runtime_fault_recover = nullptr;
}

namespace {

std::atomic<u64> ran{0};
std::atomic<u64> progress{0};

void BusyNs(u64 ns) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::nanoseconds(ns);
    while (std::chrono::steady_clock::now() < until) {
    }
}

void Handler(void*, const u8* packet, u32 size) {
    if (size != sizeof(u32)) {
        std::abort();
    }
    u32 ns;
    std::memcpy(&ns, packet, sizeof(ns));
    BusyNs(ns);
    ran.fetch_add(1, std::memory_order_release);
}

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

} // namespace

int main() {
    SetEnv("GOW3_DRAIN_HYBRID", "1");
    if (!std::getenv("GOW3_DRAIN_SPIN_US")) {
        SetEnv("GOW3_DRAIN_SPIN_US", "5");
    }
    if (!std::getenv("GOW3_PIPE_SPIN_US")) {
        SetEnv("GOW3_PIPE_SPIN_US", "10");
    }
    std::atomic<bool> finished{false};
    std::thread watchdog([&] {
        u64 last = ~0ull;
        while (!finished.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            const u64 now = progress.load();
            if (!finished.load() && now == last) {
                std::printf("FAIL: no progress in 5 s (a lost wake?), %llu drains done\n",
                            static_cast<unsigned long long>(now));
                std::fflush(stdout);
                std::_Exit(1);
            }
            last = now;
        }
    });
    u64 committed = 0, drains = 0;
    {
        Vulkan::DrawPipe pipe{Handler, nullptr};
        std::mt19937 rng{1234};
        for (u32 round = 0; round < 6000; ++round) {
            if (round % 500 == 0) {
                // Both paths, switched live as the toggle file does.
                __atomic_xor_fetch(&runtime_disabled_optimizations2, Gow3Toggle2::DrainHybridOff,
                                   __ATOMIC_RELAXED);
            }
            const u32 packets = 1 + rng() % 40;
            for (u32 i = 0; i < packets; ++i) {
                const u32 ns = rng() % 8 == 0 ? 20'000 + rng() % 60'000 : 500 + rng() % 4'000;
                u8* data = pipe.Begin(sizeof(ns));
                std::memcpy(data, &ns, sizeof(ns));
                pipe.Commit(sizeof(ns));
                ++committed;
            }
            pipe.Drain();
            if (ran.load(std::memory_order_acquire) != committed) {
                std::printf("FAIL: drain returned with %llu of %llu packets run\n",
                            static_cast<unsigned long long>(ran.load()),
                            static_cast<unsigned long long>(committed));
                return 1;
            }
            ++drains;
            progress.store(drains);
            if (rng() % 16 == 0) {
                // Stage B runs dry and goes to sleep before the next packet.
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
    }
    finished.store(true);
    watchdog.join();
    namespace DW = Gow3DrainWait;
    std::printf("draw pipe drain test: OK, %llu packets, %llu drains (stats %s: waits %llu, "
                "pause-only %llu, yields %llu; hybrid %llu, spin ends %llu, slept %llu, sleeps "
                "%llu, woken ahead %llu, ahead caught %llu, stage B wakes %llu)\n",
                static_cast<unsigned long long>(committed),
                static_cast<unsigned long long>(drains), Gow3Stats::enabled ? "on" : "off",
                static_cast<unsigned long long>(DW::waits.load()),
                static_cast<unsigned long long>(DW::pause_only.load()),
                static_cast<unsigned long long>(DW::yields.load()),
                static_cast<unsigned long long>(DW::hybrid_waits.load()),
                static_cast<unsigned long long>(DW::spin_ends.load()),
                static_cast<unsigned long long>(DW::slept.load()),
                static_cast<unsigned long long>(DW::sleeps.load()),
                static_cast<unsigned long long>(DW::woken_ahead.load()),
                static_cast<unsigned long long>(DW::ahead_caught.load()),
                static_cast<unsigned long long>(DW::wakes.load()));
    return 0;
}
