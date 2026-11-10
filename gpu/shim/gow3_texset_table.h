// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 (research/docs/impl-texset-index.md): the texture-set memo's slot choice, the memory of
// its compact layout, and its frame statistics. Both layouts (Rasterizer::TextureSet as before,
// and the compact one: a 64-byte index entry per slot with the images apart) use Choose, so a
// key finds, misses and replaces the same slot in either. Second toggle word 2048: the table as
// before (Gow3Toggle2::TextureSetLegacy).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace Gow3TexSet {

/// Slots of the memo, in either layout.
inline constexpr std::uint32_t Slots = 32768;

/// The slots a key may live in: key % Slots and, two-way (T-S1b), its neighbour index ^ 1.
struct Ways {
    std::uint32_t first, second;
};
constexpr Ways WaysOf(std::uint64_t key, bool two_way, std::uint32_t slots) {
    const auto index = static_cast<std::uint32_t>(key % slots);
    return {index, two_way ? index ^ 1u : index};
}
constexpr Ways WaysOf(std::uint64_t key, bool two_way) {
    return WaysOf(key, two_way, Slots);
}

/// The slot of the two to use: the one holding `key` (found), else the less recently used one
/// (the first on a tie). `Head` has `key` and `last_use`.
template <class Head>
std::uint32_t Choose(const Head* heads, Ways ways, std::uint64_t key, bool& found) {
    found = true;
    if (heads[ways.first].key == key) {
        return ways.first;
    }
    if (heads[ways.second].key == key) {
        return ways.second;
    }
    found = false;
    return heads[ways.first].last_use <= heads[ways.second].last_use ? ways.first : ways.second;
}

/// Zeroed memory for a large table, aligned to 2 MiB, every page written once at once (no
/// first-touch faults later, on stage B). Linux: advised for transparent huge pages before the
/// first write, then collapsed into them (MADV_COLLAPSE, Linux 6.1+) where the system allows.
class TableMemory {
public:
    TableMemory() = default;
    explicit TableMemory(std::size_t bytes);
    ~TableMemory();
    TableMemory(const TableMemory&) = delete;
    TableMemory& operator=(const TableMemory&) = delete;

    void* Data() const {
        return data;
    }
    std::size_t Bytes() const {
        return bytes;
    }
    /// Whether transparent huge pages back it: 1 yes (all of it), 0 no, -1 unknown (the kernel
    /// does not know MADV_COLLAPSE), -2 not on this system (Windows).
    int Huge() const {
        return huge;
    }

private:
    void* data = nullptr;
    std::size_t bytes = 0;
    int huge = -2;
};

/// The system's transparent huge page mode: "always", "madvise", "never", or "n/a".
const char* HugePageMode();

/// GOW3_FRAME_STATS counters ("Texture-set memo per frame" line), cumulative. Written by the
/// thread recording draws, one at a time (Bump: no locked add); read by the report.
inline std::atomic<std::uint64_t> lookups[2]{}; ///< compact, legacy
inline std::atomic<std::uint64_t> hits{0};
inline std::atomic<std::uint64_t> misses[4]{}; ///< other key, generation, image check, other
inline std::atomic<std::uint64_t> kept{0}, dropped{0};
/// T2 (impl-stage-b-per-draw-2.md; "Stage B cuts per frame" line): lookups and hits of sets of
/// 17-32 images (their own table; not in the counters above), and of stages without a prepared
/// draw (T# hashes computed at the bind; in the counters above too when 16 images or fewer).
inline std::atomic<std::uint64_t> wide_lookups{0}, wide_hits{0}, unprepared_lookups{0},
    unprepared_hits{0};
inline std::atomic<std::uint64_t> wide_bytes{0}; ///< the wide table, once made
/// The tables made so far (bytes) and whether huge pages back the compact one (TableMemory::Huge).
inline std::atomic<std::uint64_t> compact_bytes{0}, legacy_bytes{0};
inline std::atomic<int> compact_huge{-2};

inline void Bump(std::atomic<std::uint64_t>& counter) {
    counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

} // namespace Gow3TexSet
