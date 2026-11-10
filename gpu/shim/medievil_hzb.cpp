// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: MediEvil HZB occlusion remap (research/docs/study-hzb-double-buffer.md, src/hzb_hook.h).
//
// UE4 renders the bounds of every primitive it tests into a 256x256 results texture and reads
// it on the CPU the next frame, by the index each primitive got when it was added (AddBounds).
// The port's GPU runs about a frame behind the game, so the copy the game maps is usually not
// done: stale readbacks then hand it an older frame's results under this frame's indices (wrong
// culling, T-HZ1) and precise ones wait for the GPU (T-HZ3). After the game's MapResults this
// serves the newest finished copy instead, remapped by primitive key (PrimitiveId, CustomIndex)
// from the indices of the frame that produced it to the indices the game reads now. Primitives
// the older frame did not test read as visible, the game's own rule for missing results. The
// game's own map is skipped after a first synchronising call: it waited for stage B every frame.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>
#include "core/memory.h"
#include "gow3_toggles.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

using VideoCore::TextureCache;

namespace {
using Key = std::uint64_t;
using Entry = std::pair<Key, std::uint32_t>; // key, results index

constexpr std::uint64_t ResultsBytes = 256 * 256 * 4;
constexpr std::uint32_t MaxIndex = 256 * 256;

// FHZBOcclusionTester (at FSceneViewState + 0x170).
constexpr std::size_t TesterInViewState = 0x170, ResultsBuffer = 0x40, ValidFrameNumber = 0x48;
// FSceneViewState::PrimitiveOcclusionHistorySet (TSet of 0x58-byte FPrimitiveOcclusionHistory).
constexpr std::size_t HistoryElements = 0x1360, HistoryNum = 0x1368, HashInline = 0x1398,
                      HashPointer = 0x13a0, HashSize = 0x13a8;
constexpr std::size_t HistoryStride = 0x58, PrimitiveId = 0x00, LastTestFrame = 0x2c,
                      TestIndex = 0x34, CustomIndex = 0x44, HashNext = 0x50;

template <typename T>
T Load(const std::uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/// Byte offset of a result in the texture (IsVisible, 0x1534260): 8x8 blocks, 32 blocks a row.
std::uint32_t Texel(std::uint32_t i) {
    const std::uint32_t x = (i & 7) | ((i >> 3) & 0xf8);
    const std::uint32_t y = ((i >> 3) & 7) | ((i >> 8) & 0x3ffff8);
    return 4 * (x + 256 * y);
}
/// Bytes of the texture that hold results 0..max_index (a block row is 8 texel rows).
std::uint64_t BytesFor(std::uint32_t max_index) {
    return std::min<std::uint64_t>(ResultsBytes, (std::uint64_t{max_index >> 8} + 1) * 8 * 256 * 4);
}

struct Frame {
    std::uint64_t seq = 0; ///< readback copy number of the results these indices address
    std::uint32_t max_index = 0;
    std::vector<Entry> entries; ///< sorted by key
};

struct Tester {
    std::uintptr_t tester = 0;
    std::uintptr_t address = 0; ///< guest address of the results texture (learnt at a sync)
    bool synced = false;
    std::uint64_t submits = 0;     ///< Submit calls so far: one results copy each
    std::uint64_t sync_submits = 0; ///< submits at the last sync: copy n is submit sync_submits + n
    std::array<Frame, 6> ring;
    std::vector<std::uint8_t> source, served;
    std::vector<Key> hidden_keys; ///< stats: keys read hidden in the last map, sorted
};

struct Stats {
    std::uint64_t calls = 0, invalid = 0, remapped = 0, syncs = 0, unavailable = 0, drift = 0;
    std::array<std::uint64_t, 7> age{}; // 0..4, >4 frames, no source
    std::uint64_t keys = 0, unmatched = 0, hidden = 0, hides = 0, unhides = 0;
    std::uint64_t hidden_min = ~std::uint64_t{0}, hidden_max = 0; // per map
    double us_total = 0, us_max = 0;
};

using MapResultsFn = void(__attribute__((sysv_abi))*)(void*, void*);
MapResultsFn map_results;
std::uintptr_t first_frame_buffer;
std::mutex mutex;
std::array<Tester, 8> testers;
std::vector<Entry> scratch;
Stats stats;
auto report_time = std::chrono::steady_clock::now();

Tester* Find(std::uint8_t* tester, bool create) {
    const auto key = reinterpret_cast<std::uintptr_t>(tester);
    auto state = std::ranges::find(testers, key, &Tester::tester);
    if (state != testers.end() || !create) return state != testers.end() ? &*state : nullptr;
    state = std::ranges::find(testers, std::uintptr_t{0}, &Tester::tester);
    if (state == testers.end()) return nullptr;
    state->tester = key;
    state->source.resize(ResultsBytes);
    state->served.resize(ResultsBytes);
    return &*state;
}

/// The histories the game looks up this frame: tested in the frame whose results it maps.
bool CollectKeys(const std::uint8_t* view_state, std::uint32_t frame, std::vector<Entry>& out,
                 std::uint32_t& max_index) {
    out.clear();
    max_index = 0;
    const auto* elements = Load<const std::uint8_t*>(view_state + HistoryElements);
    const auto num = Load<std::int32_t>(view_state + HistoryNum);
    auto* hash = Load<const std::int32_t*>(view_state + HashPointer);
    if (!hash) hash = reinterpret_cast<const std::int32_t*>(view_state + HashInline);
    const auto hash_size = Load<std::int32_t>(view_state + HashSize);
    if (num < 0 || hash_size <= 0 || hash_size > (1 << 22) || (hash_size & (hash_size - 1))) return false;
    if (num && !elements) return false;
    std::uint64_t steps = 0;
    for (std::int32_t bucket = 0; bucket < hash_size; ++bucket) {
        for (std::int32_t id = hash[bucket]; id != -1;) {
            if (id < 0 || id >= num || ++steps > (1u << 22)) return false;
            const std::uint8_t* h = elements + std::size_t(id) * HistoryStride;
            if ((Load<std::uint32_t>(h + LastTestFrame) & 0x7fffffff) == frame) {
                const auto index = Load<std::uint32_t>(h + TestIndex);
                if (index >= MaxIndex) return false;
                const Key key = Load<std::uint32_t>(h + PrimitiveId) |
                                Key{Load<std::uint32_t>(h + CustomIndex)} << 32;
                out.emplace_back(key, index);
                max_index = std::max(max_index, index);
            }
            id = Load<std::int32_t>(h + HashNext);
        }
    }
    std::ranges::sort(out, {}, &Entry::first);
    return true;
}

void Report(std::chrono::steady_clock::time_point now) {
    if (!Gow3Stats::enabled || now - report_time < std::chrono::seconds(5)) return;
    const auto& s = stats;
    const auto u = [](std::uint64_t v) { return static_cast<unsigned long long>(v); };
    std::printf("HZB remap (5 s): %llu maps (%llu invalid, %llu remapped, %llu syncs, %llu drift, "
                "%llu unavailable); age 0/1/2/3/4/>4/none %llu/%llu/%llu/%llu/%llu/%llu/%llu; keys %llu "
                "(%llu read visible, %llu hidden, %llu-%llu a map; flips %llu hidden, %llu visible); "
                "%.1f us avg, %.1f us max\n",
                u(s.calls), u(s.invalid), u(s.remapped), u(s.syncs), u(s.drift), u(s.unavailable),
                u(s.age[0]), u(s.age[1]), u(s.age[2]), u(s.age[3]), u(s.age[4]), u(s.age[5]),
                u(s.age[6]), u(s.keys), u(s.unmatched), u(s.hidden),
                u(s.hidden_max ? s.hidden_min : 0), u(s.hidden_max), u(s.hides), u(s.unhides), s.remapped ? s.us_total / s.remapped : 0.0,
                s.us_max);
    stats = {};
    report_time = now;
}

void Serve(std::uint8_t* tester, std::uint8_t* served) {
    std::memcpy(tester + ResultsBuffer, &served, sizeof(served));
}

/// Everything visible (no culling this frame): the whole texture, as the indices may be unknown.
void ServeVisible(std::uint8_t* tester, Tester& state) {
    std::memset(state.served.data(), 0xff, ResultsBytes);
    Serve(tester, state.served.data());
}

/// Runs the game's own MapResults (it waits for the copy of the latest Submit to be queued),
/// learns the results address and restarts the copy numbering from the next Submit.
void Sync(std::uint8_t* tester, void* command_list, Tester& state) {
    ++stats.syncs;
    map_results(tester, command_list);
    const auto valid = Load<std::uint32_t>(tester + ValidFrameNumber);
    const auto address = Load<std::uintptr_t>(tester + ResultsBuffer);
    if (valid == 0xffffffffu || !address || address == first_frame_buffer) return; // nothing mapped
    auto* rasterizer = Core::Memory::Instance()->GetRasterizer();
    if (!rasterizer) return;
    if (!rasterizer->GetTextureCache().SnapshotReadback(address, ResultsBytes, nullptr, 0, true).ok) return;
    state.address = address;
    state.sync_submits = state.submits;
    for (auto& f : state.ring) f.seq = 0;
    state.synced = true;
    ServeVisible(tester, state); // the guest copy is stale (T-HZ1); the remap starts next frame
}

void Map(std::uint8_t* tester, void* command_list) {
    Tester* state = Find(tester, true);
    if (!state) {
        ++stats.unavailable;
        map_results(tester, command_list);
        return;
    }
    if (!state->synced) {
        Sync(tester, command_list, *state);
        return;
    }
    // MapResults without the map: invalid results read the "no results" buffer.
    const auto valid = Load<std::uint32_t>(tester + ValidFrameNumber);
    if (valid == 0xffffffffu) {
        ++stats.invalid;
        if (!Load<std::uintptr_t>(tester + ResultsBuffer)) {
            std::memcpy(tester + ResultsBuffer, &first_frame_buffer, sizeof(first_frame_buffer));
        }
        return;
    }
    std::uint32_t max_index = 0;
    if (!CollectKeys(tester - TesterInViewState, valid, scratch, max_index)) {
        ++stats.unavailable;
        ServeVisible(tester, *state);
        return;
    }
    // Copy number of the latest Submit, whose indices the histories now hold.
    const std::uint64_t seq = state->submits - state->sync_submits;
    std::uint32_t copy_max = max_index;
    for (const auto& f : state->ring) {
        if (f.seq) copy_max = std::max(copy_max, f.max_index);
    }
    auto* rasterizer = Core::Memory::Instance()->GetRasterizer();
    const auto snap = rasterizer ? rasterizer->GetTextureCache().SnapshotReadback(
        state->address, ResultsBytes, state->source.data(), BytesFor(copy_max)) : TextureCache::ReadbackSnapshot{};
    if (!snap.ok) {
        ++stats.unavailable;
        ServeVisible(tester, *state);
        return;
    }
    if (snap.pushed > seq || seq - snap.pushed > 64) {
        // More copies than Submits (an extra download) or copies stopped: resynchronise.
        ++stats.drift;
        state->synced = false;
        ServeVisible(tester, *state);
        return;
    }
    std::uint8_t* served = state->served.data();
    if (scratch.empty() || !seq) {
        Serve(tester, served); // nothing is read this frame
        return;
    }
    auto slot = std::ranges::find(state->ring, seq, &Frame::seq);
    if (slot == state->ring.end()) slot = std::ranges::min_element(state->ring, {}, &Frame::seq);
    slot->seq = seq;
    slot->max_index = max_index;
    slot->entries.swap(scratch);
    const Frame& now = *slot;
    const Frame* source = nullptr;
    if (snap.published) {
        const auto found = std::ranges::find(state->ring, snap.published, &Frame::seq);
        if (found != state->ring.end()) source = &*found;
    }
    ++stats.age[!source ? 6 : std::min<std::uint64_t>(seq - snap.published, 5)];
    stats.keys += now.entries.size();
    if (!source) {
        stats.unmatched += now.entries.size();
        for (const auto& [key, index] : now.entries) std::memset(served + Texel(index), 0xff, 4);
    } else {
        // Both lists are sorted by key.
        auto it = source->entries.begin();
        const auto end = source->entries.end();
        const std::uint8_t* bytes = state->source.data();
        // Stats: keys hidden in this map against the last one, as flips per key (camera still:
        // a steady trickle of flips is visibility oscillating, not the view changing).
        const bool flips = Gow3Stats::enabled;
        auto was = state->hidden_keys.cbegin();
        const auto was_end = state->hidden_keys.cend();
        scratch.clear();
        std::uint64_t hidden = 0;
        for (const auto& [key, index] : now.entries) {
            while (it != end && it->first < key) ++it;
            bool is_hidden = false;
            if (it != end && it->first == key) {
                std::memcpy(served + Texel(index), bytes + Texel(it->second), 4);
                is_hidden = bytes[Texel(it->second)] == 0; // IsVisible reads the first byte
            } else {
                std::memset(served + Texel(index), 0xff, 4);
                ++stats.unmatched;
            }
            hidden += is_hidden;
            if (flips) {
                while (was != was_end && *was < key) ++was;
                const bool was_hidden = was != was_end && *was == key;
                stats.hides += is_hidden && !was_hidden;
                stats.unhides += was_hidden && !is_hidden;
                if (is_hidden) scratch.emplace_back(key, 0);
            }
        }
        stats.hidden += hidden;
        stats.hidden_min = std::min(stats.hidden_min, hidden);
        stats.hidden_max = std::max(stats.hidden_max, hidden);
        if (flips) {
            state->hidden_keys.clear();
            for (const auto& entry : scratch) state->hidden_keys.push_back(entry.first);
        }
    }
    Serve(tester, served);
    ++stats.remapped;
}
} // namespace

extern "C" void medievil_hzb_init(std::uintptr_t map_results_address, std::uintptr_t no_results) {
    map_results = reinterpret_cast<MapResultsFn>(map_results_address);
    first_frame_buffer = no_results;
}

/// In place of FHZBOcclusionTester::MapResults (src/hzb_hook.h).
extern "C" void __attribute__((sysv_abi)) medievil_hzb_map(void* tester, void* command_list) {
    const auto start = std::chrono::steady_clock::now();
    std::scoped_lock lock{mutex};
    ++stats.calls;
    const auto remapped = stats.remapped;
    Map(static_cast<std::uint8_t*>(tester), command_list);
    const auto now = std::chrono::steady_clock::now();
    if (stats.remapped != remapped) {
        const double us = std::chrono::duration<double, std::micro>(now - start).count();
        stats.us_total += us;
        stats.us_max = std::max(stats.us_max, us);
    }
    Report(now);
}

/// Before FHZBOcclusionTester::Submit: one results copy follows.
extern "C" void __attribute__((sysv_abi)) medievil_hzb_submit(void* tester) {
    std::scoped_lock lock{mutex};
    if (Tester* state = Find(static_cast<std::uint8_t*>(tester), true)) ++state->submits;
}
