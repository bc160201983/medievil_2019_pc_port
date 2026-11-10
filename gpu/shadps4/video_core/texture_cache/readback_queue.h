#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cassert>
#include <deque>
#include <utility>

namespace VideoCore {

// Copies retire in submission order, including overlapping guest ranges.
template <typename Payload>
class ReadbackQueue {
public:
    struct Entry {
        std::uint64_t address, size, tick;
        Payload payload;
        bool valid = true;
        bool watched = true; ///< still protects its pages (see ReleaseOutside)
        std::uint64_t seq = 0; ///< gow3: copy number of a tracked range (TextureCache::SnapshotReadback), 0 if none
        bool reads = false; ///< gow3: its pages also watch CPU reads (a first copy, GOW3_READBACK_FIRST_WAIT)
    };
    bool CanFit(std::uint64_t size) const {
        return entries.size() < 64 && size <= MaxBytes && bytes <= MaxBytes - size;
    }
    void Push(std::uint64_t address, std::uint64_t size, std::uint64_t tick, Payload payload,
              std::uint64_t seq = 0, bool reads = false) {
        assert(CanFit(size));
        entries.push_back({address, size, tick, std::move(payload)});
        entries.back().seq = seq;
        entries.back().reads = reads;
        bytes += size;
    }
    std::uint64_t RequiredTick(std::uint64_t address, std::uint64_t size) const {
        std::uint64_t tick = 0;
        for (const auto& entry : entries) {
            if (entry.valid && entry.watched && OverlapsPage(entry, address, size)) tick = entry.tick;
        }
        return tick;
    }
    template <typename Callback>
    void Cancel(std::uint64_t address, std::uint64_t size, Callback&& cancel) {
        for (auto& entry : entries) {
            if (entry.valid && OverlapsPage(entry, address, size)) {
                cancel(entry);
                entry.valid = false;
            }
        }
    }
    /// gow3: a CPU access to [address, address + size) that shares pages with small copies but
    /// not their bytes stops waiting for them: each such copy is unwatched (`release` drops its
    /// page protection) and still published when the GPU is done. Returns the copies released.
    template <typename Release>
    unsigned ReleaseOutside(std::uint64_t address, std::uint64_t size, std::uint64_t max_size,
                            Release&& release) {
        unsigned released = 0;
        for (auto& entry : entries) {
            if (entry.valid && entry.watched && entry.size <= max_size &&
                OverlapsPage(entry, address, size) &&
                !(address < entry.address + entry.size && entry.address < address + size)) {
                release(entry);
                entry.watched = false;
                ++released;
            }
        }
        return released;
    }
    template <typename Ready, typename Complete>
    void Retire(Ready&& ready, Complete&& complete) {
        while (!entries.empty() && ready(entries.front().tick)) {
            complete(entries.front());
            bytes -= entries.front().size;
            entries.pop_front();
        }
    }
    /// gow3: queued copies of [address, address + size) lose their copy number.
    void ClearSeq(std::uint64_t address, std::uint64_t size) {
        for (auto& entry : entries) {
            if (address < entry.address + entry.size && entry.address < address + size) entry.seq = 0;
        }
    }
    /// Calls `visit(address, size)` for every queued copy.
    template <typename Visit>
    void ForEach(Visit&& visit) const {
        for (const auto& entry : entries) {
            visit(entry.address, entry.size);
        }
    }

    /// Whether Retire would complete anything now (copies retire in order).
    template <typename Ready>
    bool FrontReady(Ready&& ready) const {
        return !entries.empty() && ready(entries.front().tick);
    }
    std::uint64_t Bytes() const { return bytes; }
    std::size_t Size() const { return entries.size(); }

    /// gow3 (impl-stage-b-bundle-2.md C5): the pages of the copies RequiredTick can still match
    /// (valid and watched) as at most N sorted ranges [first, second); the closest neighbours
    /// are merged first. Any range RequiredTick finds a copy for overlaps one of them (a
    /// superset: merging only adds pages). Returns the count.
    template <std::size_t N>
    std::size_t Spans(std::array<std::pair<std::uint64_t, std::uint64_t>, N>& out) const {
        static_assert(N > 0);
        std::array<std::pair<std::uint64_t, std::uint64_t>, 64> spans;
        std::size_t count = 0;
        for (const auto& entry : entries) {
            if (!entry.valid || !entry.watched) continue;
            if (count == spans.size()) {
                out[0] = {0, ~std::uint64_t{0}}; // more copies than expected: everything
                return 1;
            }
            const auto begin = entry.address & ~std::uint64_t{4095};
            const auto end = (entry.address + entry.size + 4095) & ~std::uint64_t{4095};
            spans[count++] = {begin, std::max(end, begin + 1)}; // an empty copy still matches
        }
        std::sort(spans.begin(), spans.begin() + count);
        std::size_t merged = 0;
        for (std::size_t i = 0; i < count; ++i) {
            if (merged && spans[i].first <= spans[merged - 1].second) {
                spans[merged - 1].second = std::max(spans[merged - 1].second, spans[i].second);
            } else {
                spans[merged++] = spans[i];
            }
        }
        while (merged > N) {
            std::size_t closest = 0;
            for (std::size_t i = 1; i + 1 < merged; ++i) {
                if (spans[i + 1].first - spans[i].second <
                    spans[closest + 1].first - spans[closest].second) {
                    closest = i;
                }
            }
            spans[closest].second = spans[closest + 1].second;
            std::copy(spans.begin() + closest + 2, spans.begin() + merged,
                      spans.begin() + closest + 1);
            --merged;
        }
        std::copy(spans.begin(), spans.begin() + merged, out.begin());
        return merged;
    }

private:
    static bool OverlapsPage(const Entry& entry, std::uint64_t address, std::uint64_t size) {
        if (!size) return false;
        const auto begin = entry.address & ~std::uint64_t{4095};
        const auto end = (entry.address + entry.size + 4095) & ~std::uint64_t{4095};
        return address < end && begin < address + size;
    }
    static constexpr std::uint64_t MaxBytes = 64ull << 20;
    std::deque<Entry> entries;
    std::uint64_t bytes = 0;
};
} // namespace VideoCore
