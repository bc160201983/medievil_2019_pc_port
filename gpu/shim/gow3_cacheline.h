// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 #6 (research/docs/impl-stage-b-bundle-2.md; study-deck-apu-deep-dive.md 1.7): fields that
// one thread writes often and another reads often start a new 64-byte cache line, so a write no
// longer takes the line from the reader's L1 (a core-to-core transfer, worse on the Deck's Zen 2
// at a low fabric clock). GOW3_CACHE_LINE marks the first field of each group. Build switch:
// GOW3_CACHELINE_PADDING=OFF (CMake / build.sh) defines it to nothing: today's layout exactly.
#pragma once

#ifndef GOW3_CACHELINE_PADDING
#define GOW3_CACHELINE_PADDING 1
#endif
#if GOW3_CACHELINE_PADDING
#define GOW3_CACHE_LINE alignas(64)
#else
#define GOW3_CACHE_LINE
#endif
