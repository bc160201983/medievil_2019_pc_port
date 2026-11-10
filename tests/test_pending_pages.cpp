// A2 (gpu/shim/gow3_stage_a.h; research/docs/impl-stage-a-decode.md): the pending-write filter.
// A model of Rasterizer::NotePendingGpuWrite / PendingWriteOverlaps (the same merge, prune and
// scan rules, a fake stage B position) runs once with the filter and once without over long
// random runs; every answer and the entry list after every step must be the same.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include "gow3_stage_a.h"

namespace {
struct Write {
    std::uint64_t begin, end, position;
    bool operator==(const Write&) const = default;
};

struct Model {
    bool filter = false;
    std::vector<Write> writes;
    std::uint64_t min = ~std::uint64_t{0}, max = 0;
    std::uint32_t checks = 0;
    Gow3StageA::PendingPages pages;
    std::uint64_t filtered = 0;

    void Note(std::uint64_t address, std::uint64_t size, std::uint64_t head) {
        if (!size) {
            return;
        }
        const std::uint64_t end = address + size;
        pages.Mark(address, end);
        for (auto& w : writes) {
            if (address <= w.end && w.begin <= end) {
                w.begin = std::min(w.begin, address);
                w.end = std::max(w.end, end);
                w.position = head;
                min = std::min(min, address);
                max = std::max(max, end);
                return;
            }
        }
        writes.push_back({address, end, head});
        min = std::min(min, address);
        max = std::max(max, end);
    }
    bool Overlaps(std::uint64_t address, std::uint64_t size, std::uint64_t consumed) {
        if (writes.empty() || address >= max || address + size <= min) {
            return false;
        }
        if ((++checks & 63) == 0) {
            std::erase_if(writes, [&](const Write& w) { return consumed >= w.position; });
            min = ~std::uint64_t{0};
            max = 0;
            pages.Clear();
            for (const auto& w : writes) {
                min = std::min(min, w.begin);
                max = std::max(max, w.end);
                pages.Mark(w.begin, w.end);
            }
        }
        if (size != 0 && filter && pages.Untouched(address, address + size)) {
            ++filtered;
            return false;
        }
        const std::uint64_t end = address + size;
        for (const auto& w : writes) {
            if (address < w.end && w.begin < end && consumed < w.position) {
                return true;
            }
        }
        return false;
    }
};

void Run(std::uint32_t seed) {
    std::mt19937_64 rng{seed};
    Model with, without;
    with.filter = true;
    without.filter = false;
    std::uint64_t head = 0, consumed = 0, yes = 0, queries = 0;
    // A few hot areas (UAVs, labels) spread over the address space, so the bounds rarely help.
    const std::uint64_t areas[] = {0x10'0000'0000ull, 0x10'4000'0000ull, 0x2f'b150'0000ull,
                                   0x30'0000'0000ull, 0x7f'0000'0000ull, 0xfb'ff00'0000ull};
    for (int step = 0; step < 2'000'000; ++step) {
        const std::uint64_t area = areas[rng() % std::size(areas)];
        const std::uint64_t address = area + (rng() % (1u << 26)) * 16;
        const unsigned kind = rng() % 100;
        if (kind < 6) {
            // A pending write: a label, a buffer, rarely one wider than the mark limit.
            const bool wide = kind == 0 && rng() % 200 == 0;
            const std::uint64_t size = wide       ? (std::uint64_t{20} << 20) + rng() % 4096
                                       : kind < 3 ? 4 + rng() % 8
                                                  : 256 + rng() % (2u << 20);
            ++head;
            with.Note(address, size, head);
            without.Note(address, size, head);
        } else if (kind < 9) {
            // Stage B moves on, a few packets behind stage A (now and then far behind).
            const std::uint64_t lag = rng() % 16 == 0 ? rng() % 200 : rng() % 6;
            consumed = std::max(consumed, head > lag ? head - lag : 0);
        } else {
            // A constant buffer check (up to 16 KiB, the stream threshold), sometimes size 0,
            // sometimes right next to or inside a written range.
            std::uint64_t at = address, size = rng() % 16385;
            if (kind < 30 && !with.writes.empty()) {
                const auto& w = with.writes[rng() % with.writes.size()];
                at = rng() % 2 ? w.end + rng() % 64 : w.begin - std::min<std::uint64_t>(w.begin, rng() % 128);
            }
            if (kind == 99) {
                size = 0;
            }
            const bool a = with.Overlaps(at, size, consumed);
            const bool b = without.Overlaps(at, size, consumed);
            assert(a == b);
            yes += a;
            ++queries;
        }
        assert(with.writes == without.writes && with.min == without.min && with.max == without.max);
    }
    std::printf("seed %u: %llu checks, %llu overlapped, %llu answered by the filter, %zu entries left\n",
                seed, static_cast<unsigned long long>(queries), static_cast<unsigned long long>(yes),
                static_cast<unsigned long long>(with.filtered), with.writes.size());
    assert(with.filtered > 0);
}

void Edges() {
    Gow3StageA::PendingPages p;
    assert(p.Untouched(0x1000, 0x2000));
    p.Mark(0x10000, 0x10001); // one byte of page 1
    assert(!p.Untouched(0x1ffff, 0x20000) && !p.Untouched(0xffff, 0x10001));
    assert(!p.Untouched(0x10000, 0x10000)); // an empty query is not filtered
    p.Mark(0, std::uint64_t{17} << 20);     // wider than 256 pages: everything
    assert(p.all && !p.Untouched(0x5000'0000, 0x5000'0010));
    p.Clear();
    assert(!p.all && p.Untouched(0x10000, 0x10001));
    p.Mark(0x30000, 0x30000); // empty: nothing
    assert(p.Untouched(0x30000, 0x30001));
    assert(!p.Untouched(0, std::uint64_t{17} << 16)); // a query over 16 pages: not filtered
}
} // namespace

int main() {
    Edges();
    for (std::uint32_t seed : {1u, 2u, 3u}) {
        Run(seed);
    }
    std::puts("pending-pages-test: OK");
    return 0;
}
