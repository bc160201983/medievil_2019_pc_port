// The texture-set memo's slot choice and table memory (gpu/shim/gow3_texset_table.h;
// research/docs/impl-texset-index.md): Choose decides exactly as the lookup did before the
// compact layout, over a long run of lookups, replacements and dropped sets.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "gow3_texset_table.h"

using namespace Gow3TexSet;

namespace {
/// The slot as before the compact layout (key and last use where Rasterizer::TextureSet has them).
struct OldSet {
    std::uint64_t key = 0;
    const void* stage = nullptr;
    std::uint64_t generation = ~0ull, stamp = 0, last_use = 0, scene_generation = 0;
    std::uint32_t count = 0;
    std::uint64_t hashes[16]{};
    unsigned char entries[16 * 88]{};
};
/// The compact layout's index entry.
struct alignas(64) Head {
    std::uint64_t key = 0, last_use = 0, generation = ~0ull, stamp = 0, scene_generation = 0;
    const void* stage = nullptr;
    std::uint32_t count = 0;
};
static_assert(sizeof(Head) == 64);

/// The lookup's slot choice before the compact layout, verbatim (vk_rasterizer.cpp at 9e1f320).
OldSet* OldChoose(std::vector<OldSet>& texture_sets, std::uint64_t key, bool two_way,
                  bool& was_found) {
    const std::uint64_t index = key % texture_sets.size();
    OldSet* way0 = &texture_sets[index];
    OldSet* way1 = two_way ? &texture_sets[index ^ 1] : way0;
    OldSet* found = way0->key == key ? way0 : way1->key == key ? way1 : nullptr;
    auto& set = found ? *found : *(way0->last_use <= way1->last_use ? way0 : way1);
    was_found = found != nullptr;
    return &set;
}

/// `slots`: the table size (Slots, or T2's wide table of 1,024: WaysOf with a size).
void SameDecisions(bool two_way, std::uint32_t seed, std::uint32_t slots = Slots) {
    std::vector<OldSet> old_sets(slots);
    std::vector<Head> heads(slots);
    std::uint64_t old_tick = 0, tick = 0;
    std::mt19937_64 rng{seed};
    // A hot working set (the Deck: 99.3% hits), new keys, and keys that collide in a slot pair.
    std::vector<std::uint64_t> hot(slots < Slots ? slots / 4 : 3000);
    for (auto& key : hot) key = rng() | 1;
    std::uint64_t hits = 0, misses = 0;
    for (int n = 0; n < 2'000'000; ++n) {
        const std::uint64_t pick = rng() % 1000;
        std::uint64_t key;
        if (pick < 990) {
            key = hot[rng() % hot.size()];
        } else if (pick < 995) {
            key = rng() | 1;
        } else {
            // Same pair of slots as a hot key: the replacement order decides.
            key = (hot[rng() % hot.size()] + slots * (1 + rng() % 7)) | 1;
        }
        bool old_found = false, found = false;
        OldSet* old_set = OldChoose(old_sets, key, two_way, old_found);
        const std::uint32_t index =
            slots == Slots ? Choose(heads.data(), WaysOf(key, two_way), key, found)
                           : Choose(heads.data(), WaysOf(key, two_way, slots), key, found);
        assert(old_found == found);
        assert(old_set == &old_sets[index]);
        old_set->last_use = ++old_tick;
        heads[index].last_use = ++tick;
        if (!found) {
            old_set->key = key;
            heads[index].key = key;
            ++misses;
        } else {
            ++hits;
        }
        // The slow path drops some sets it could not keep (key 0, last use kept).
        if (rng() % 50 == 0) {
            old_set->key = 0;
            heads[index].key = 0;
        }
    }
    for (std::uint32_t i = 0; i < slots; ++i) {
        assert(old_sets[i].key == heads[i].key && old_sets[i].last_use == heads[i].last_use);
    }
    std::printf("%u slots, two-way %d: %llu hits, %llu misses, same slots and decisions\n",
                slots, int(two_way), static_cast<unsigned long long>(hits),
                static_cast<unsigned long long>(misses));
    assert(hits > misses);
}
} // namespace

int main() {
    // Slots: key % Slots and its neighbour.
    assert(WaysOf(5, true).first == 5 && WaysOf(5, true).second == 4);
    assert(WaysOf(Slots + 6, true).first == 6 && WaysOf(Slots + 6, true).second == 7);
    assert(WaysOf(9, false).first == 9 && WaysOf(9, false).second == 9);

    // A tie in last use replaces the first slot; a held key wins over a less recent slot.
    std::vector<Head> heads(Slots);
    bool found = true;
    assert(Choose(heads.data(), WaysOf(11, true), 11, found) == 11 && !found);
    heads[10].last_use = 1;
    heads[11].last_use = 2;
    assert(Choose(heads.data(), WaysOf(11, true), 11, found) == 10 && !found);
    heads[11].key = 11;
    assert(Choose(heads.data(), WaysOf(11, true), 11, found) == 11 && found);
    heads[10].key = Slots + 10 + 1; // the other key of the pair, in the second way
    assert(Choose(heads.data(), WaysOf(Slots + 11, true), Slots + 11, found) == 10 && found);

    SameDecisions(true, 1);
    SameDecisions(false, 2);
    // T2's wide table: the same choice over 1,024 slots; WaysOf with the default size as before.
    assert(WaysOf(Slots + 6, true, Slots).first == WaysOf(Slots + 6, true).first);
    assert(WaysOf(1024 + 6, true, 1024).first == 6 && WaysOf(1024 + 6, true, 1024).second == 7);
    SameDecisions(true, 3, 1024);
    SameDecisions(false, 4, 1024);

    // Table memory: zeroed, writable, rounded up to 2 MiB (aligned to it on Linux).
    {
        TableMemory memory(44'040'192 + 1);
        assert(memory.Data() != nullptr);
        assert(memory.Bytes() == 22 * (std::size_t{2} << 20));
#ifndef _WIN32
        assert(reinterpret_cast<std::uintptr_t>(memory.Data()) % (std::size_t{2} << 20) == 0);
        assert(memory.Huge() >= -1 && memory.Huge() <= 1);
#else
        assert(memory.Huge() == -2);
#endif
        const auto* bytes = static_cast<const unsigned char*>(memory.Data());
        for (std::size_t i = 0; i < memory.Bytes(); i += 4093) {
            assert(bytes[i] == 0);
        }
        std::memset(memory.Data(), 0xab, memory.Bytes());
        std::printf("table memory: %zu bytes, huge pages %d, system mode %s\n", memory.Bytes(),
                    memory.Huge(), HugePageMode());
    }
    const char* mode = HugePageMode();
    assert(!std::strcmp(mode, "always") || !std::strcmp(mode, "madvise") ||
           !std::strcmp(mode, "never") || !std::strcmp(mode, "n/a"));

    // Counters: one writer, no locked add.
    const auto before = kept.load();
    Bump(kept);
    Bump(kept);
    assert(kept.load() == before + 2);
    std::printf("texset-table-test: ok\n");
    return 0;
}
