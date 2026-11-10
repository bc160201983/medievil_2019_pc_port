// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 #7 / #8: core scheduling and priority for stage B and the vblank thread (gow3_linux_sched.h).

#include "gow3_linux_sched.h"
#include "gow3_toggles.h"

#ifdef _WIN32

namespace Gow3LinuxSched {
void StageBThread() {}
void VBlankThread() {}
void PollLive() {}
} // namespace Gow3LinuxSched

#else

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <dlfcn.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <unistd.h>

// Older headers (glibc < 2.35) lack the core-scheduling constants (linux/prctl.h values).
#ifndef PR_SCHED_CORE
#define PR_SCHED_CORE 62
#endif
#ifndef PR_SCHED_CORE_GET
#define PR_SCHED_CORE_GET 0
#endif
#ifndef PR_SCHED_CORE_CREATE
#define PR_SCHED_CORE_CREATE 1
#endif
#ifndef PR_SCHED_CORE_SCOPE_THREAD
#define PR_SCHED_CORE_SCOPE_THREAD 0
#endif

namespace Gow3LinuxSched {
namespace {

/// GOW3_STAGE_B_CORE_SCHED=1 at start.
bool CoreSchedWanted() {
    static const bool wanted = [] {
        const char* env = std::getenv("GOW3_STAGE_B_CORE_SCHED");
        return env && env[0] == '1';
    }();
    return wanted;
}

/// GOW3_THREAD_NICE at start: 1 = nice -10, -N = nice -N (at least -20), anything else 0 (off).
int WantedNice() {
    static const int nice = [] {
        const char* env = std::getenv("GOW3_THREAD_NICE");
        if (!env) {
            return 0;
        }
        if (env[0] == '1' && !env[1]) {
            return -10;
        }
        const int n = std::atoi(env);
        return n < 0 ? std::max(n, -20) : 0;
    }();
    return nice;
}

/// rtkit's MakeThreadHighPriority(tid, nice) over the system bus, through the host's libdbus
/// (loaded here: SDL loads it the same way; nothing links it). A private connection, closed after
/// the call, that does not exit the process when the bus goes away (libdbus's default).
bool RtkitHighPriority(pid_t tid, int nice, std::string& why) {
    void* dbus = dlopen("libdbus-1.so.3", RTLD_NOW | RTLD_LOCAL);
    if (!dbus) {
        why = "rtkit: libdbus-1.so.3 not found";
        return false;
    }
    // DBusError's layout (dbus-errors.h): name, message, five 1-bit fields, a pointer. Spare room
    // after it costs nothing.
    struct DBusError {
        const char* name;
        const char* message;
        unsigned int bits;
        void* padding;
        void* spare[4];
    };
    using ErrorInit = void (*)(DBusError*);
    using ErrorFree = void (*)(DBusError*);
    using BusGet = void* (*)(int, DBusError*);
    using SetExit = void (*)(void*, unsigned int);
    using NewCall = void* (*)(const char*, const char*, const char*, const char*);
    using AppendArgs = unsigned int (*)(void*, int, ...);
    using SendBlock = void* (*)(void*, void*, int, DBusError*);
    using Unref = void (*)(void*);
    const auto error_init = reinterpret_cast<ErrorInit>(dlsym(dbus, "dbus_error_init"));
    const auto error_free = reinterpret_cast<ErrorFree>(dlsym(dbus, "dbus_error_free"));
    const auto bus_get = reinterpret_cast<BusGet>(dlsym(dbus, "dbus_bus_get_private"));
    const auto set_exit = reinterpret_cast<SetExit>(dlsym(dbus, "dbus_connection_set_exit_on_disconnect"));
    const auto new_call = reinterpret_cast<NewCall>(dlsym(dbus, "dbus_message_new_method_call"));
    const auto append_args = reinterpret_cast<AppendArgs>(dlsym(dbus, "dbus_message_append_args"));
    const auto send_block =
        reinterpret_cast<SendBlock>(dlsym(dbus, "dbus_connection_send_with_reply_and_block"));
    const auto message_unref = reinterpret_cast<Unref>(dlsym(dbus, "dbus_message_unref"));
    const auto connection_close = reinterpret_cast<Unref>(dlsym(dbus, "dbus_connection_close"));
    const auto connection_unref = reinterpret_cast<Unref>(dlsym(dbus, "dbus_connection_unref"));
    if (!error_init || !error_free || !bus_get || !set_exit || !new_call || !append_args ||
        !send_block || !message_unref || !connection_close || !connection_unref) {
        why = "rtkit: libdbus-1.so.3 lacks a function";
        return false;
    }
    constexpr int SystemBus = 1;                                  // DBUS_BUS_SYSTEM
    constexpr int TypeUint64 = 't', TypeInt32 = 'i', TypeEnd = 0; // DBUS_TYPE_*
    DBusError error{};
    error_init(&error);
    void* bus = bus_get(SystemBus, &error);
    if (!bus) {
        why = std::string("rtkit: no system bus (") + (error.message ? error.message : "?") + ")";
        error_free(&error);
        return false;
    }
    set_exit(bus, 0);
    bool done = false;
    void* call = new_call("org.freedesktop.RealtimeKit1", "/org/freedesktop/RealtimeKit1",
                          "org.freedesktop.RealtimeKit1", "MakeThreadHighPriority");
    std::uint64_t thread = static_cast<std::uint64_t>(tid);
    std::int32_t priority = nice;
    if (call && append_args(call, TypeUint64, &thread, TypeInt32, &priority, TypeEnd)) {
        if (void* reply = send_block(bus, call, 1000, &error)) {
            message_unref(reply);
            done = true;
        } else {
            why = std::string("rtkit: ") + (error.name ? error.name : "no reply") +
                  (error.message ? std::string(" (") + error.message + ")" : std::string());
        }
    } else {
        why = "rtkit: could not build the call";
    }
    if (call) {
        message_unref(call);
    }
    error_free(&error);
    connection_close(bus);
    connection_unref(bus);
    return done; // libdbus stays loaded: it is not made to be unloaded
}

/// Thread `tid` at `nice`: setpriority first (RLIMIT_NICE or CAP_SYS_NICE), else rtkit.
void RaisePriority(const char* name, pid_t tid, int nice, const char* extra) {
    if (setpriority(PRIO_PROCESS, static_cast<id_t>(tid), nice) == 0) {
        std::printf("Thread priority (GOW3_THREAD_NICE): %s, tid %d, nice %d by setpriority%s\n", name,
                    int(tid), nice, extra);
        return;
    }
    const int direct_error = errno;
    rlimit limit{};
    const long rlimit_nice = getrlimit(RLIMIT_NICE, &limit) == 0 ? long(limit.rlim_cur) : -1;
    std::string why;
    if (RtkitHighPriority(tid, nice, why)) {
        errno = 0;
        const int now = getpriority(PRIO_PROCESS, static_cast<id_t>(tid));
        std::printf("Thread priority (GOW3_THREAD_NICE): %s, tid %d, nice %d through rtkit "
                    "(setpriority: %s, RLIMIT_NICE %ld)%s\n",
                    name, int(tid), errno ? nice : now, std::strerror(direct_error), rlimit_nice, extra);
        return;
    }
    std::printf("Thread priority (GOW3_THREAD_NICE): %s, tid %d, unchanged, not nice %d: setpriority: "
                "%s (RLIMIT_NICE %ld); %s%s\n",
                name, int(tid), nice, std::strerror(direct_error), rlimit_nice, why.c_str(), extra);
}

/// The threads at the start's nice (PollLive), and the live state (vblank thread only).
std::atomic<pid_t> stage_b_tid{0}, vblank_tid{0};
std::atomic<bool> live_busy{false};
bool live_off = false;

} // namespace

void StageBThread() {
    if (CoreSchedWanted()) {
        if (prctl(PR_SCHED_CORE, PR_SCHED_CORE_CREATE, 0, PR_SCHED_CORE_SCOPE_THREAD, 0) == 0) {
            unsigned long cookie = 0;
            prctl(PR_SCHED_CORE, PR_SCHED_CORE_GET, 0, PR_SCHED_CORE_SCOPE_THREAD, &cookie);
            std::printf("Stage B: core scheduling cookie %#lx (GOW3_STAGE_B_CORE_SCHED), tid %d: its SMT "
                        "sibling runs only this cookie while it runs\n",
                        cookie, int(gettid()));
        } else {
            const int error = errno;
            const char* hint = error == EINVAL   ? " (kernel without CONFIG_SCHED_CORE, or before 5.14)"
                               : error == ENODEV ? " (no SMT: the CPU has none or it is off)"
                                                 : "";
            std::printf("Stage B: no core scheduling cookie (GOW3_STAGE_B_CORE_SCHED): %s%s\n",
                        std::strerror(error), hint);
        }
    }
    if (const int nice = WantedNice()) {
        stage_b_tid.store(gettid(), std::memory_order_relaxed);
        RaisePriority("stage B (gow3:DrawRec)", gettid(), nice, "");
    }
}

void VBlankThread() {
    const int nice = WantedNice();
    if (!nice) {
        return;
    }
    // A 1 ns timer slack: the tick's sleep_until wakes on time instead of up to 50 us late.
    const bool slack = prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL) == 0;
    vblank_tid.store(gettid(), std::memory_order_relaxed);
    RaisePriority("vblank (shadPS4:PresentThread)", gettid(), nice,
                  slack ? "; timer slack 1 ns" : "; timer slack unchanged");
}

void PollLive() {
    const int nice = WantedNice();
    if (!nice) {
        return;
    }
    const bool off = Gow3Toggle2::Disabled(Gow3Toggle2::ThreadNiceOff);
    if (off == live_off || live_busy.exchange(true, std::memory_order_acquire)) {
        return; // unchanged, or the last switch is still being made (tried again next tick)
    }
    live_off = off;
    // rtkit is a D-Bus call of up to a second: not on the vblank thread. Nice 0 needs no privilege
    // (raising a thread's own nice value); the start's nice again goes the start's way.
    std::thread([off, nice] {
        const std::pair<const char*, pid_t> threads[] = {
            {"stage B (gow3:DrawRec)", stage_b_tid.load(std::memory_order_relaxed)},
            {"vblank (shadPS4:PresentThread)", vblank_tid.load(std::memory_order_relaxed)}};
        for (const auto& [name, tid] : threads) {
            if (!tid) {
                continue;
            }
            if (!off) {
                RaisePriority(name, tid, nice, " (live)");
            } else if (setpriority(PRIO_PROCESS, static_cast<id_t>(tid), 0) == 0) {
                std::printf("Thread priority (GOW3_THREAD_NICE): %s, tid %d, nice 0 (live, toggle file "
                            "second word 32768)\n",
                            name, int(tid));
            } else {
                std::printf("Thread priority (GOW3_THREAD_NICE): %s, tid %d, not back at nice 0: %s\n",
                            name, int(tid), std::strerror(errno));
            }
        }
        live_busy.store(false, std::memory_order_release);
    }).detach();
}

} // namespace Gow3LinuxSched

#endif
