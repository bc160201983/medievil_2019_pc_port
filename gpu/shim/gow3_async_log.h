// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: diagnostics output written by a background thread. The slow-frame lines and the barrier
// report were printed on the flip path with an unbuffered stdout (one console write per printf,
// ~30 per slow frame): the printing itself delayed frames and turned one hitch into a burst
// (t-pan1). Lines are formatted on the caller's thread and queued; their order among themselves
// is kept, not their order relative to other output.
#pragma once

#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

namespace Gow3AsyncLog {

struct Queue {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::string> lines;
    std::thread worker;
};

inline Queue& Get() {
    static Queue* queue = [] {
        auto* q = new Queue; // never destroyed: written to until the process exits
        q->worker = std::thread([q] {
            for (;;) {
                std::deque<std::string> batch;
                {
                    std::unique_lock lock{q->mutex};
                    q->cv.wait(lock, [q] { return !q->lines.empty(); });
                    batch.swap(q->lines);
                }
                for (const auto& line : batch) {
                    std::fwrite(line.data(), 1, line.size(), stdout);
                }
                std::fflush(stdout);
            }
        });
        q->worker.detach();
        return q;
    }();
    return *queue;
}

/// Queues text (include the newline).
inline void Write(std::string text) {
    auto& q = Get();
    {
        std::scoped_lock lock{q.mutex};
        q.lines.push_back(std::move(text));
    }
    q.cv.notify_one();
}

/// Appends printf-style text to `out`.
inline void Append(std::string& out, const char* format, ...) {
    va_list args;
    va_start(args, format);
    va_list copy;
    va_copy(copy, args);
    const int size = std::vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    if (size > 0) {
        const size_t at = out.size();
        out.resize(at + size_t(size) + 1);
        std::vsnprintf(out.data() + at, size_t(size) + 1, format, args);
        out.resize(at + size_t(size));
    }
    va_end(args);
}

} // namespace Gow3AsyncLog
