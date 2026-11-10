// The residency bitmap (gpu/shim/gow3_resident_blocks.h; research/docs/impl-stage-b-per-draw-2.md
// R2): over a long run of ranges made resident the way EnsureResident adds them (the gaps of a
// request) and of residency checks, "every bit set" answers exactly as "the interval search finds
// no gap" (BufferCache::resident_ranges, an IntervalList).
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>
#include "common/interval_set.h"
#include "gow3_resident_blocks.h"

namespace {
bool NoGap(const IntervalList<>& list, std::uint64_t start, std::uint64_t end) {
    bool gap = false;
    list.ForEachGap(start, end, [&](std::uint64_t, std::uint64_t) { gap = true; });
    return !gap;
}

void Run(std::uint32_t seed) {
    IntervalList<> ranges;
    Gow3Resident::Blocks bits;
    std::mt19937_64 rng{seed};
    constexpr std::uint64_t Chunk = Gow3Resident::Blocks::ChunkBlocks;
    std::uint64_t answers[2]{};
    for (int n = 0; n < 400'000; ++n) {
        // Requests around a few hot places, across word and chunk edges, of 1 to ~300 blocks
        // (a 64 KiB block: up to ~19 MiB), and now and then a wide one.
        const std::uint64_t base = (rng() % 48) * Chunk / 2 + Chunk - 200 + rng() % 400;
        const std::uint64_t length = rng() % 8 == 0 ? 1 + rng() % 3000 : 1 + rng() % 300;
        const std::uint64_t start = base + rng() % 40'000;
        const std::uint64_t end = start + length;
        const bool resident = bits.All(start, end);
        assert(resident == NoGap(ranges, start, end));
        ++answers[resident];
        if (!resident && rng() % 64 == 0) {
            // EnsureResident: each gap becomes resident (one Add each), the bitmap with it.
            std::vector<std::pair<std::uint64_t, std::uint64_t>> gaps;
            ranges.ForEachGap(start, end,
                              [&](std::uint64_t a, std::uint64_t b) { gaps.emplace_back(a, b); });
            for (const auto& [a, b] : gaps) {
                ranges.Add({a, b});
                bits.Set(a, b);
            }
            assert(bits.All(start, end) && NoGap(ranges, start, end));
        }
    }
    // Both answers came up often.
    assert(answers[0] > 10'000 && answers[1] > 10'000);
    std::printf("seed %u: %llu resident, %llu not\n", seed,
                static_cast<unsigned long long>(answers[1]),
                static_cast<unsigned long long>(answers[0]));
}
} // namespace

int main() {
    Gow3Resident::Blocks bits;
    // Edges: an empty range is never "all resident"; single blocks; word and chunk edges; the
    // top of the bitmap; past it.
    assert(!bits.All(5, 5));
    assert(!bits.All(0, 1));
    bits.Set(63, 65);
    assert(bits.All(63, 65) && bits.All(64, 65) && !bits.All(62, 64) && !bits.All(64, 66));
    constexpr std::uint64_t Chunk = Gow3Resident::Blocks::ChunkBlocks;
    bits.Set(Chunk - 1, Chunk + 1);
    assert(bits.All(Chunk - 1, Chunk + 1) && !bits.All(Chunk - 2, Chunk + 1));
    bits.Set(0, 64 * 3);
    assert(bits.All(0, 64 * 3) && !bits.All(0, 64 * 3 + 1));
    constexpr std::uint64_t Top = std::uint64_t{1} << Gow3Resident::Blocks::MaxBlockBits;
    bits.Set(Top - 2, Top);
    assert(bits.All(Top - 2, Top) && !bits.All(Top - 3, Top) && !bits.All(Top - 1, Top + 1));
    for (std::uint32_t seed : {1u, 2u, 3u}) {
        Run(seed);
    }
    std::printf("resident-blocks-test: pass\n");
    return 0;
}
