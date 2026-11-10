// SPDX-License-Identifier: GPL-2.0-or-later
// Gow3SignalWait (gpu/shim/gow3_signal_wait.h): two waiter threads wait for a counter that a
// producer raises with random gaps (some inside the spin, most long enough to sleep), calling
// Notify after each raise as the recorder does. Every wait must return only once its target is
// reached, and a watchdog fails the test on a lost wake.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>

#include "gow3_signal_wait.h"

extern "C" {
std::uint64_t runtime_disabled_optimizations = 0;
std::uint64_t runtime_disabled_optimizations2 = 0;
}

namespace {

constexpr std::uint64_t Rounds = 4000;
std::atomic<std::uint64_t> counter{0};
std::atomic<std::uint64_t> progress{0};

void BusyNs(std::uint64_t ns) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::nanoseconds(ns);
    while (std::chrono::steady_clock::now() < until) {
    }
}

} // namespace

int main() {
    std::thread watchdog([] {
        std::uint64_t last = ~0ull;
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            const std::uint64_t now = progress.load();
            if (now >= 2 * Rounds) {
                return;
            }
            if (now == last) {
                std::fprintf(stderr, "signal-wait-test: no progress for 5 s (lost wake?) at %llu\n",
                             static_cast<unsigned long long>(now));
                std::_Exit(1);
            }
            last = now;
        }
    });
    std::thread producer([] {
        std::mt19937 rng{1234};
        std::uniform_int_distribution<int> gap(0, 9);
        for (std::uint64_t i = 1; i <= Rounds; ++i) {
            const int g = gap(rng);
            BusyNs(g < 3 ? 2000 : g < 8 ? 80000 : 400000); // 2 us, 80 us, 400 us
            counter.store(i, std::memory_order_release);
            Gow3SignalWait::Notify();
        }
    });
    auto waiter = [](std::uint64_t seed) {
        std::mt19937 rng{static_cast<unsigned>(seed)};
        for (std::uint64_t target = 1; target <= Rounds; target += 1 + rng() % 3) {
            Gow3SignalWait::Wait(
                [&] { return counter.load(std::memory_order_acquire) >= target; }, true);
            if (counter.load(std::memory_order_acquire) < target) {
                std::fprintf(stderr, "signal-wait-test: returned early (%llu < %llu)\n",
                             static_cast<unsigned long long>(counter.load()),
                             static_cast<unsigned long long>(target));
                std::_Exit(1);
            }
            progress.fetch_add(1);
        }
    };
    std::thread a(waiter, 1), b(waiter, 2);
    a.join();
    b.join();
    producer.join();
    progress.store(2 * Rounds);
    watchdog.join();
    namespace SW = Gow3SignalWait;
    std::printf("signal-wait-test: OK; waits %llu, ended in the spin %llu, slept %llu (sleeps %llu, "
                "%.1f ms), wakes %llu\n",
                static_cast<unsigned long long>(SW::waits.load()),
                static_cast<unsigned long long>(SW::spin_ends.load()),
                static_cast<unsigned long long>(SW::slept.load()),
                static_cast<unsigned long long>(SW::sleeps.load()), SW::sleep_ns.load() / 1e6,
                static_cast<unsigned long long>(SW::wakes.load()));
    return 0;
}
