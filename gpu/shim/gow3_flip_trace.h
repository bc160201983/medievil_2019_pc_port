// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 GOW3_FLIP_TRACE=<file> (diagnostics): one CSV line per stage of every flip, to see where
// frame pacing turns uneven under a frame cap: the game registers the flip (submit), the draw
// recording thread reaches it (irq), its present is recorded (queued), the present thread takes
// it (release), the swap thread has presented it (presented), the game's flip event fired
// (flipdone). Columns: stage, flip argument (the game's frame tag), nanoseconds since start. The
// first row, "start", holds the start itself in steady_clock nanoseconds (QueryPerformanceCounter on
// Windows), to line the trace up with an ETW / WPR capture.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace Gow3FlipTrace {
inline std::chrono::steady_clock::time_point Start() {
    static const auto start = std::chrono::steady_clock::now();
    return start;
}
inline std::FILE* File() {
    static std::FILE* file = [] {
        const char* path = std::getenv("GOW3_FLIP_TRACE");
        std::FILE* f = path && path[0] ? std::fopen(path, "w") : nullptr;
        if (f) {
            std::fputs("stage,flip_arg,ns\n", f);
            std::fprintf(f, "start,0,%lld\n",
                         static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                    Start().time_since_epoch()).count()));
        }
        return f;
    }();
    return file;
}
inline void Note(const char* stage, std::int64_t flip_arg) {
    std::FILE* file = File();
    if (!file) return;
    const auto start = Start();
    static std::mutex mutex;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start).count();
    std::scoped_lock lock{mutex};
    std::fprintf(file, "%s,%lld,%lld\n", stage, static_cast<long long>(flip_arg),
                 static_cast<long long>(ns));
    static unsigned lines;
    if (++lines % 2048 == 0) std::fflush(file);
}
} // namespace Gow3FlipTrace
