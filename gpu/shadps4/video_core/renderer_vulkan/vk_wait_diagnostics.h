// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "gow3_async_log.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>

namespace Vulkan {
class WaitDiagnostics {
public:
    struct Entry {
        std::string_view file{};
        unsigned line{}, calls{};
        uint64_t submit_ns{}, wait_ns{}, max_wait_ns{};
    };
    void Add(std::string_view file, unsigned line, uint64_t submit, uint64_t wait) {
        std::scoped_lock lock{mutex};
        auto* entry = &entries.back();
        for (unsigned i = 0; i + 1 < entries.size(); ++i) {
            if (!entries[i].calls || (entries[i].file == file && entries[i].line == line)) {
                entry = &entries[i];
                entry->file = file;
                entry->line = line;
                break;
            }
        }
        if (entry == &entries.back()) entry->file = "other wait sites";
        ++entry->calls;
        entry->submit_ns += submit;
        entry->wait_ns += wait;
        entry->max_wait_ns = std::max(entry->max_wait_ns, wait);
    }
    std::array<Entry, 64> Take() {
        std::scoped_lock lock{mutex};
        const auto result = entries;
        entries = {};
        return result;
    }
    static bool Enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("GOW3_PERF_DIAG");
            return value && value[0] == '1';
        }();
        return enabled;
    }
    static WaitDiagnostics& Get() {
        static WaitDiagnostics stats;
        return stats;
    }
    /// gow3: Scheduler::WaitHostCopies callers that waited: submit_ns = for the copies on the
    /// recording thread, wait_ns = for the copy threads.
    static WaitDiagnostics& HostCopies() {
        static WaitDiagnostics stats;
        return stats;
    }
    /// gow3: Scheduler::EndRendering callers that ended an open render pass (a pass ended for a
    /// copy, e.g. a buffer upload, stores and reloads its targets).
    static WaitDiagnostics& RenderPassEnds() {
        static WaitDiagnostics stats;
        return stats;
    }
    /// gow3: what differed when an open render pass gave way to another (Scheduler::BeginRendering).
    static WaitDiagnostics& PassSwitches() {
        static WaitDiagnostics stats;
        return stats;
    }
    /// gow3: every barrier flush: who asked (site), what it covers; "(between passes)" when no
    /// render pass was open (the others also ended one).
    static WaitDiagnostics& BarrierBreaks() {
        static WaitDiagnostics stats;
        return stats;
    }
    /// gow3: a pass ended at a site, then the next pass opened on the same targets: a break that
    /// stored and reloaded the targets for nothing (keyed by the site that ended it).
    static WaitDiagnostics& PassReopens() {
        static WaitDiagnostics stats;
        return stats;
    }
    static void Report() {
        if (!Enabled()) return;
        // Written by a background thread (gow3_async_log.h): printed here, on the flip path, the
        // report delayed a frame every 5 s.
        std::string out;
        struct Flush {
            std::string& text;
            ~Flush() { Gow3AsyncLog::Write(std::move(text)); }
        } flush{out};
        auto rows = Get().Take();
        std::sort(rows.begin(), rows.end(), [](const Entry& a, const Entry& b) {
            return a.submit_ns + a.wait_ns > b.submit_ns + b.wait_ns;
        });
        for (const auto& row : rows) {
            if (!row.calls) continue;
            const auto filename = row.file.substr(row.file.find_last_of("/\\") + 1);
            Gow3AsyncLog::Append(out,"Wait profile: %.*s:%u; %u calls, CPU submit %.3f ms, semaphore wait %.3f ms, max wait %.3f ms\n",
                        int(filename.size()), filename.data(), row.line, row.calls,
                        row.submit_ns / 1e6, row.wait_ns / 1e6, row.max_wait_ns / 1e6);
        }
        rows = HostCopies().Take();
        std::sort(rows.begin(), rows.end(), [](const Entry& a, const Entry& b) {
            return a.submit_ns + a.wait_ns > b.submit_ns + b.wait_ns;
        });
        for (const auto& row : rows) {
            if (!row.calls) continue;
            const auto filename = row.file.substr(row.file.find_last_of("/\\") + 1);
            Gow3AsyncLog::Append(out,"Host copy waits: %.*s:%u; %u waits, recording thread %.3f ms, copy threads %.3f ms, max %.3f ms\n",
                        int(filename.size()), filename.data(), row.line, row.calls,
                        row.submit_ns / 1e6, row.wait_ns / 1e6, row.max_wait_ns / 1e6);
        }
        rows = RenderPassEnds().Take();
        std::sort(rows.begin(), rows.end(),
                  [](const Entry& a, const Entry& b) { return a.calls > b.calls; });
        for (const auto& row : rows) {
            if (!row.calls) continue;
            const auto filename = row.file.substr(row.file.find_last_of("/\\") + 1);
            Gow3AsyncLog::Append(out,"Render pass ends: %.*s:%u; %u\n", int(filename.size()), filename.data(),
                        row.line, row.calls);
        }
        rows = PassSwitches().Take();
        std::sort(rows.begin(), rows.end(),
                  [](const Entry& a, const Entry& b) { return a.calls > b.calls; });
        for (const auto& row : rows) {
            if (row.calls) {
                Gow3AsyncLog::Append(out,"Pass switch: %.*s; %u\n", int(row.file.size()), row.file.data(), row.calls);
            }
        }
        rows = BarrierBreaks().Take();
        std::sort(rows.begin(), rows.end(),
                  [](const Entry& a, const Entry& b) { return a.calls > b.calls; });
        for (const auto& row : rows) {
            if (row.calls) {
                Gow3AsyncLog::Append(out,"Barrier: %.*s; %u\n", int(row.file.size()), row.file.data(), row.calls);
            }
        }
        rows = PassReopens().Take();
        std::sort(rows.begin(), rows.end(),
                  [](const Entry& a, const Entry& b) { return a.calls > b.calls; });
        for (const auto& row : rows) {
            if (!row.calls) continue;
            const auto filename = row.file.substr(row.file.find_last_of("/\\") + 1);
            Gow3AsyncLog::Append(out,"Pass reopened on the same targets after an end at %.*s:%u; %u\n",
                        int(filename.size()), filename.data(), row.line, row.calls);
        }
    }
private:
    std::mutex mutex;
    std::array<Entry, 64> entries{};
};
}
