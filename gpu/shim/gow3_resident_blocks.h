// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 (research/docs/impl-stage-b-per-draw-2.md, R2): which arena blocks are resident, as a
// bitmap beside BufferCache::resident_ranges. Residency only grows (nothing removes a range), so
// "every block of the range is set" is exactly "the interval search finds no gap". It answers
// ObtainBuffer's residency check (~2,100 calls a heavy Deck frame, 0.32 ms in the sorted-vector
// search) with a word or two of one chunk. Chunks of 2^16 blocks (8 KiB of bits) are made on
// their first set block; a chunk never made means no block in it is resident.
#pragma once

#include <array>
#include <cstdint>
#include <memory>

namespace Gow3Resident {

class Blocks {
public:
    static constexpr unsigned ChunkBits = 16;
    static constexpr std::uint64_t ChunkBlocks = std::uint64_t{1} << ChunkBits;
    static constexpr std::uint64_t ChunkWords = ChunkBlocks / 64;
    /// Block numbers below 2^MaxBlockBits (2^40 bytes of guest address in blocks of 16 KiB or
    /// more); 1,024 chunk pointers.
    static constexpr unsigned MaxBlockBits = 26;
    static constexpr std::uint64_t MaxChunks = std::uint64_t{1} << (MaxBlockBits - ChunkBits);

    /// Marks blocks [start, end) resident.
    void Set(std::uint64_t start, std::uint64_t end) {
        ForEachWord(start, end, [this](std::uint64_t chunk, std::uint64_t word, std::uint64_t mask) {
            auto& bits = chunks[chunk];
            if (!bits) {
                bits = std::make_unique<std::uint64_t[]>(ChunkWords); // zeroed
            }
            bits[word] |= mask;
            return true;
        });
    }

    /// Whether every block in [start, end) is resident (false for an empty range, as the
    /// callers never ask one; blocks past the bitmap are never resident).
    bool All(std::uint64_t start, std::uint64_t end) const {
        if (start >= end) {
            return false;
        }
        return ForEachWord(start, end,
                           [this](std::uint64_t chunk, std::uint64_t word, std::uint64_t mask) {
                               const auto& bits = chunks[chunk];
                               return bits && (bits[word] & mask) == mask;
                           });
    }

private:
    /// Calls f(chunk, word, mask) for each bitmap word the range touches, in order, while f
    /// returns true; false when f stopped it or the range leaves the bitmap.
    template <class F>
    bool ForEachWord(std::uint64_t start, std::uint64_t end, F&& f) const {
        if (end > (std::uint64_t{1} << MaxBlockBits)) {
            return false;
        }
        while (start < end) {
            const std::uint64_t chunk = start >> ChunkBits;
            const std::uint64_t word = (start & (ChunkBlocks - 1)) >> 6;
            const unsigned bit = static_cast<unsigned>(start & 63);
            const std::uint64_t word_end = (start | 63) + 1;
            const std::uint64_t stop = end < word_end ? end : word_end;
            const unsigned count = static_cast<unsigned>(stop - start);
            const std::uint64_t mask =
                (count == 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << count) - 1)) << bit;
            if (!f(chunk, word, mask)) {
                return false;
            }
            start = stop;
        }
        return true;
    }

    std::array<std::unique_ptr<std::uint64_t[]>, MaxChunks> chunks{};
};

} // namespace Gow3Resident
