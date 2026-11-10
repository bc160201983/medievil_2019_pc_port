// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: helper thread sizing. Counts follow the hardware threads this process may run on
// (the affinity mask, so `taskset` can emulate a Steam Deck), and speculative helpers run as
// SCHED_IDLE: they use cores the game leaves idle and never take time from its threads.

#pragma once

#include <algorithm>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <bit>
#include <windows.h>
#else
#include <sched.h>
#include <dirent.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#include <cstdlib>

namespace Gow3Threads {
#ifdef _WIN32
/// Hardware threads available to the process (its affinity mask).
inline unsigned Available() {
    DWORD_PTR process = 0, system = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system) && process) {
        return std::max(1, std::popcount(static_cast<unsigned long long>(process)));
    }
    return std::max(1u, std::thread::hardware_concurrency());
}

/// The calling thread only runs when cores are otherwise idle (lowest Windows priority).
inline void MakeBackground() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_IDLE);
}

/// gow3 GOW3_STAGE_B_CORE=1 (T-HZ4g: stage B's time per draw rose 20-40% in every part in the
/// slow frames, as if the thread got less of its core): the calling thread gets one logical
/// processor of the last physical core to itself; both of that core's logical processors leave
/// the process's default CPU sets, so no other thread of the process runs beside it.
inline bool DedicateCore(const char* name) {
    ULONG length = 0;
    GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
    std::vector<unsigned char> buffer(length);
    auto* info = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data());
    if (!length || !GetSystemCpuSetInformation(info, length, &length, GetCurrentProcess(), 0)) {
        return false;
    }
    struct Set {
        ULONG id;
        BYTE core, logical;
    };
    std::vector<Set> sets;
    for (ULONG offset = 0; offset < length;) {
        const auto* entry = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
        if (entry->Type == CpuSetInformation && entry->CpuSet.Group == 0) {
            sets.push_back({entry->CpuSet.Id, entry->CpuSet.CoreIndex,
                            entry->CpuSet.LogicalProcessorIndex});
        }
        offset += entry->Size;
    }
    if (sets.size() < 4) return false;
    BYTE core = 0;
    for (const auto& s : sets) core = std::max(core, s.core);
    std::vector<ULONG> others;
    ULONG own = 0;
    BYTE own_logical = 0, sibling_logical = 0xff;
    bool found = false;
    for (const auto& s : sets) {
        if (s.core != core) {
            others.push_back(s.id);
        } else if (!found) {
            own = s.id;
            own_logical = s.logical;
            found = true;
        } else {
            sibling_logical = s.logical;
        }
    }
    if (!found || others.empty() ||
        !SetProcessDefaultCpuSets(GetCurrentProcess(), others.data(), static_cast<ULONG>(others.size())) ||
        !SetThreadSelectedCpuSets(GetCurrentThread(), &own, 1)) {
        return false;
    }
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    std::printf("%s: dedicated core %u (logical processor %u; %s kept free), priority above normal\n",
                name, core, own_logical,
                sibling_logical == 0xff ? "no sibling" : ("sibling " + std::to_string(sibling_logical)).c_str());
    return true;
}
#else

/// Hardware threads available to the process.
inline unsigned Available() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        return std::max(1, CPU_COUNT(&set));
    }
    return std::max(1u, std::thread::hardware_concurrency());
}

/// The calling thread only runs on otherwise idle cores (falls back to the lowest nice level).
inline void MakeBackground() {
    sched_param param{};
    if (sched_setscheduler(0, SCHED_IDLE, &param) != 0) {
        setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 19);
    }
}

/// gow3 GOW3_STAGE_B_CORE=1 on Linux (the Deck: 4 cores, 8 threads, small caches; stage B
/// migrates between cores): the calling thread runs alone on one logical CPU of the highest
/// physical core; its sibling stays free, and every other thread of the process (those running
/// now; later ones inherit it from their creators) moves to the remaining cores.
inline bool DedicateCore(const char* name) {
    cpu_set_t all;
    CPU_ZERO(&all);
    if (sched_getaffinity(0, sizeof(all), &all) != 0 || CPU_COUNT(&all) < 4) {
        return false;
    }
    auto core_of = [](int cpu) {
        char path[96];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        int core = -1;
        if (FILE* file = std::fopen(path, "r")) {
            if (std::fscanf(file, "%d", &core) != 1) core = -1;
            std::fclose(file);
        }
        return core;
    };
    int last_core = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &all)) last_core = std::max(last_core, core_of(cpu));
    }
    cpu_set_t own, others;
    CPU_ZERO(&own);
    CPU_ZERO(&others);
    int own_cpu = -1, sibling = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &all)) continue;
        if (core_of(cpu) != last_core) {
            CPU_SET(cpu, &others);
        } else if (own_cpu < 0) {
            own_cpu = cpu;
        } else {
            sibling = cpu;
        }
    }
    if (last_core < 0 || own_cpu < 0 || CPU_COUNT(&others) == 0) {
        return false;
    }
    const pid_t self = gettid();
    unsigned moved = 0;
    if (DIR* tasks = opendir("/proc/self/task")) {
        while (const dirent* entry = readdir(tasks)) {
            const pid_t tid = static_cast<pid_t>(std::atoi(entry->d_name));
            if (tid > 0 && tid != self && sched_setaffinity(tid, sizeof(others), &others) == 0) {
                ++moved;
            }
        }
        closedir(tasks);
    }
    CPU_SET(own_cpu, &own);
    if (sched_setaffinity(0, sizeof(own), &own) != 0) {
        return false;
    }
    std::printf("%s: dedicated core %d (CPU %d; %s kept free), %u other threads moved\n", name,
                last_core, own_cpu,
                sibling < 0 ? "no sibling" : ("sibling CPU " + std::to_string(sibling)).c_str(), moved);
    return true;
}
#endif

} // namespace Gow3Threads
