// A3 (gpu/shim/gow3_stage_a.h; research/docs/impl-stage-a-decode.md): the register-packet
// checksum two words per step. The one-word hash is today's (checked against the loop it
// replaces); the paired hash tells apart every variant of register packets that differ in one
// bit, in one or two words, or in their order, as well as the one-word hash does; and the time
// per word of both.
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "gow3_stage_a.h"

namespace {
std::uint64_t Reference(std::uint64_t checksum, const std::uint32_t* words, std::uint32_t count) {
    for (std::uint32_t i = 0; i < count; ++i) { // liverpool.cpp before A3
        checksum = (checksum ^ words[i]) * 0x9E3779B97F4A7C15ull;
        checksum ^= checksum >> 29;
    }
    return checksum;
}

template <typename Hash>
std::size_t Collisions(Hash hash, std::uint32_t seed) {
    std::mt19937_64 rng{seed};
    // Hash -> the input it came from: two different inputs with one hash are a collision (the
    // same input made twice is not).
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> seen;
    std::size_t collisions = 0, made = 0;
    std::vector<std::uint32_t> base(18);
    // Register packets as UE4 sends them: a header, an offset, user data words (pointers, small
    // counts); variants of one base packet differ in one or two words, the hard case.
    for (int b = 0; b < 2000; ++b) {
        const std::uint32_t count = 2 + rng() % 17;
        for (auto& w : base) {
            w = rng() % 4 ? std::uint32_t(rng() & 0xffff) : std::uint32_t(rng());
        }
        for (int v = 0; v < 1000; ++v) {
            auto words = base;
            words[rng() % count] ^= std::uint32_t(1) << (rng() % 32);
            if (v % 2) {
                words[rng() % count] += std::uint32_t(1 + rng() % 4096);
            }
            const std::uint64_t seed_state = (b % 7) * 0x1234567ull; // a few running checksums
            const std::uint64_t h = hash(hash(seed_state, words.data(), count), base.data(), 3);
            if (words == base) {
                continue;
            }
            ++made;
            std::vector<std::uint32_t> input(words.begin(), words.begin() + count);
            input.push_back(std::uint32_t(b % 7));
            input.insert(input.end(), base.begin(), base.begin() + 3);
            const auto [at, added] = seen.try_emplace(h, input);
            collisions += !added && at->second != input;
        }
    }
    (void)made;
    return collisions;
}

template <typename Hash>
void BitFlips(Hash hash) {
    std::uint32_t words[16];
    for (std::uint32_t i = 0; i < 16; ++i) {
        words[i] = 0xC0001000u + i * 0x10001u;
    }
    std::unordered_set<std::uint64_t> seen{hash(7, words, 16)};
    for (std::uint32_t w = 0; w < 16; ++w) {
        for (std::uint32_t bit = 0; bit < 32; ++bit) {
            words[w] ^= 1u << bit;
            assert(seen.insert(hash(7, words, 16)).second);
            words[w] ^= 1u << bit;
        }
    }
    // Swapped neighbours (the same words in another order) differ too.
    for (std::uint32_t w = 0; w + 1 < 16; ++w) {
        std::swap(words[w], words[w + 1]);
        assert(seen.insert(hash(7, words, 16)).second);
        std::swap(words[w], words[w + 1]);
    }
}

template <typename Hash>
double NsPerWord(Hash hash) {
    std::vector<std::uint32_t> words(64);
    for (std::uint32_t i = 0; i < words.size(); ++i) {
        words[i] = i * 2654435761u;
    }
    std::uint64_t checksum = 1;
    const auto start = std::chrono::steady_clock::now();
    constexpr int Rounds = 2'000'000;
    for (int r = 0; r < Rounds; ++r) {
        checksum = hash(checksum, words.data() + (r & 31), 17); // a 17-word packet
    }
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
    volatile std::uint64_t sink = checksum;
    (void)sink;
    return ns / (double(Rounds) * 17);
}
} // namespace

int main() {
    std::mt19937_64 rng{9};
    for (int n = 0; n < 100000; ++n) {
        std::uint32_t words[20];
        const std::uint32_t count = rng() % 21;
        for (auto& w : words) {
            w = std::uint32_t(rng());
        }
        const std::uint64_t seed = rng();
        assert(Gow3StageA::HashWords(seed, words, count) == Reference(seed, words, count));
    }
    BitFlips(Gow3StageA::HashWords);
    BitFlips(Gow3StageA::HashWordsPaired);
    const std::size_t one = Collisions(Gow3StageA::HashWords, 1);
    const std::size_t paired = Collisions(Gow3StageA::HashWordsPaired, 1);
    std::printf("collisions over 2 million near-identical packets: one word %zu, paired %zu\n", one, paired);
    std::fflush(stdout);
    assert(one == 0 && paired == 0);
    std::printf("ns per word: one word %.3f, paired %.3f\n", NsPerWord(Gow3StageA::HashWords),
                NsPerWord(Gow3StageA::HashWordsPaired));
    std::puts("stage-a-hash-test: OK");
    return 0;
}
