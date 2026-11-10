// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: the futex calls of stage A's hybrid drain wait (gow3_drain_wait.h), out of line so that
// the header needs no system headers.

#include "gow3_drain_wait.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Gow3DrainWait {

void FutexWait(std::atomic<std::uint32_t>& word, std::uint32_t seen) {
    static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t));
#ifdef _WIN32
    WaitOnAddress(reinterpret_cast<volatile VOID*>(&word), &seen, sizeof(seen), INFINITE);
#else
    syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(&word), FUTEX_WAIT_PRIVATE, seen, nullptr,
            nullptr, 0);
#endif
}

void FutexWake(std::atomic<std::uint32_t>& word) {
#ifdef _WIN32
    WakeByAddressSingle(reinterpret_cast<PVOID>(&word));
#else
    syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(&word), FUTEX_WAKE_PRIVATE, 1, nullptr,
            nullptr, 0);
#endif
}

void FutexWakeAll(std::atomic<std::uint32_t>& word) {
#ifdef _WIN32
    WakeByAddressAll(reinterpret_cast<PVOID>(&word));
#else
    syscall(SYS_futex, reinterpret_cast<std::uint32_t*>(&word), FUTEX_WAKE_PRIVATE, INT32_MAX,
            nullptr, nullptr, 0);
#endif
}

} // namespace Gow3DrainWait
