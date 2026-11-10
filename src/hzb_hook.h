/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef MEDIEVIL_HZB_HOOK_H
#define MEDIEVIL_HZB_HOOK_H
#include <stdint.h>
#include <string.h>

/* MediEvil (CUSA11227) HZB occlusion remap (research/docs/study-hzb-double-buffer.md,
 * GOW3_HZB_REMAP=1): UE4's FHZBOcclusionTester::MapResults (01.00 image offset 0x15340a0) has one
 * caller, 0x158c4ac (rdi = FSceneViewState + 0x170, rsi = RHICmdList; the caller keeps rdx in
 * rbx across the call). The host helper takes its place and points the tester's results (+0x40)
 * at a remapped host copy. The same code in each game build, at its own offsets
 * (research/docs/plan-0102.md); the build whose signatures all match is used. */
typedef struct {
    uint32_t call, map_results, is_visible, history_walk, submit_call, submit, first_frame_buffer;
} MedievilHzbSites;
static const MedievilHzbSites medievil_hzb_builds[] = {
    {0x158c4ac, 0x15340a0, 0x1534260, 0x158d4df, 0x11573ec, 0x1534340, 0x5606ac8}, /* 01.00 */
    {0x158f12c, 0x1536d20, 0x1536ee0, 0x159015f, 0x115a06c, 0x1536fc0, 0x5606ac8}, /* 01.02 */
};
/* `bytes` at image+start, with a rel32 call at call_offset checked by its target instead (the
 * displacement differs between builds). */
static inline int medievil_bytes_with_call(const unsigned char *image, uint32_t start, const unsigned char *bytes,
                                           size_t n, size_t call_offset, uint32_t target) {
    int32_t rel;
    memcpy(&rel, image + start + call_offset + 1, 4);
    return !memcmp(image + start, bytes, call_offset + 1) &&
           !memcmp(image + start + call_offset + 5, bytes + call_offset + 5, n - call_offset - 5) &&
           (int64_t)start + (int64_t)call_offset + 5 + rel == (int64_t)target;
}
static const unsigned char medievil_hzb_call_bytes[] = {
    0x48,0x8d,0xba,0x70,0x01,0x00,0x00, /* lea rdi, [rdx+0x170] */
    0x48,0x89,0xd3,                     /* mov rbx, rdx */
    0xe8,0xef,0x7b,0xfa,0xff,           /* call MapResults */
    0x48,0x89,0xda                      /* mov rdx, rbx */
};
/* MapResults: [rdi+0x48] ValidFrameNumber. */
static const unsigned char medievil_hzb_map_bytes[] = {
    0x55,0x48,0x89,0xe5,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x83,0xec,0x18
};
/* IsVisible: ResultsBuffer at [rdi+0x40], 8x8 blocks of 4-byte texels, 256 a row. */
static const unsigned char medievil_hzb_visible_bytes[] = {
    0x55,0x48,0x89,0xe5,0x89,0xf1,0x89,0xf0,0x83,0xe6,0x07,0xc1,0xe8,0x08,0xc1,0xe9,0x03,0x89,0xca,
    0x25,0xf8,0xff,0x3f,0x00,0x83,0xe1,0x07,0x09,0xc1,0x48,0x8b,0x47,0x40
};
/* The game's occlusion history lookup: hash at ViewState+0x13a0 (0: inline at +0x1398), size
 * +0x13a8, elements 0x58 bytes, PrimitiveId +0, CustomIndex +0x44, next +0x50. */
static const unsigned char medievil_hzb_walk_bytes[] = {
    0x48,0x8b,0x8e,0xa0,0x13,0x00,0x00, /* mov rcx, [rsi+0x13a0] */
    0x8b,0xb6,0xa8,0x13,0x00,0x00,      /* mov esi, [rsi+0x13a8] */
    0x48,0x85,0xc9,0x48,0x0f,0x44,0x94,0x24,0xf8,0x01,0x00,0x00,0xff,0xce,0x21,0xc6,0x48,0x63,0xc6,
    0x48,0x09,0xca,0x8b,0x04,0x82,0x83,0xf8,0xff,0x0f,0x84,0xb2,0x01,0x00,0x00,
    0x48,0x8b,0x8c,0x24,0x20,0x01,0x00,0x00,0x48,0x8b,0x19,0x0f,0x1f,0x80,0x00,0x00,0x00,0x00,
    0x48,0x98,0x4c,0x6b,0xe0,0x58,0x42,0x39,0x3c,0x23,0x75,0x0a,0x42,0x8b,0x44,0x23,0x44
};

static inline int medievil_hzb_signatures_match(const unsigned char *image, uint64_t size,
                                                const MedievilHzbSites *s) {
    if (size < s->call + 0x40 || size < s->history_walk + sizeof(medievil_hzb_walk_bytes)) return 0;
    return medievil_bytes_with_call(image, s->call - 10, medievil_hzb_call_bytes, sizeof(medievil_hzb_call_bytes),
                                    10, s->map_results) &&
           !memcmp(image + s->map_results, medievil_hzb_map_bytes, sizeof(medievil_hzb_map_bytes)) &&
           !memcmp(image + s->is_visible, medievil_hzb_visible_bytes, sizeof(medievil_hzb_visible_bytes)) &&
           !memcmp(image + s->history_walk, medievil_hzb_walk_bytes, sizeof(medievil_hzb_walk_bytes));
}

/* The MapResults call jumps to the host helper (rdi tester, rsi RHICmdList), which calls the
 * game's MapResults itself only to synchronise: the game's map spins until the GPU side reaches
 * the copy's label (RHIMapStagingSurface, Sleep(0)), which held the render thread on stage B every
 * frame (T-HZ4c: ~430,000 Sleep(0) a second). movabs rax, helper; jmp rax. */
enum { MEDIEVIL_HZB_MAP_STUB_LENGTH = 12 };
static inline void medievil_hzb_build_map_stub(unsigned char code[MEDIEVIL_HZB_MAP_STUB_LENGTH], uintptr_t helper) {
    static const unsigned char pattern[MEDIEVIL_HZB_MAP_STUB_LENGTH] = {
        0x48,0xb8,0,0,0,0,0,0,0,0,          /* movabs rax, helper */
        0xff,0xe0                           /* jmp rax */
    };
    memcpy(code, pattern, sizeof(pattern));
    memcpy(code + 2, &helper, 8);
}

/* FHZBOcclusionTester::Submit (01.00: 0x1534340), one call per view at 0x11573ec (rdi tester, rsi
 * RHICmdList, rdx view), only with tested primitives: one results copy per call. The stub counts
 * it, then jumps to Submit. rsp is 16-byte aligned at the helper call (8 at entry, 3 pushes). */
static const unsigned char medievil_hzb_submit_call_bytes[] = {
    0x49,0x81,0xc7,0x70,0x01,0x00,0x00, /* add r15, 0x170 */
    0x48,0x89,0xda,                     /* mov rdx, rbx */
    0x4c,0x89,0xff,                     /* mov rdi, r15 */
    0xe8,0x4f,0xcf,0x3d,0x00            /* call Submit */
};
static const unsigned char medievil_hzb_submit_bytes[] = {
    0x55,0x48,0x89,0xe5,0x41,0x57,0x41,0x56,0x41,0x55,0x41,0x54,0x53,0x48,0x83,0xe4,0xe0
};
static inline int medievil_hzb_submit_signatures_match(const unsigned char *image, uint64_t size,
                                                       const MedievilHzbSites *s) {
    if (size < s->submit + 0x40 || size < s->submit_call + 0x40) return 0;
    return medievil_bytes_with_call(image, s->submit_call - 13, medievil_hzb_submit_call_bytes,
                                    sizeof(medievil_hzb_submit_call_bytes), 13, s->submit) &&
           !memcmp(image + s->submit, medievil_hzb_submit_bytes, sizeof(medievil_hzb_submit_bytes));
}
/* The build whose HZB code is all where its table says, or NULL. */
static inline const MedievilHzbSites *medievil_hzb_find(const unsigned char *image, uint64_t size) {
    for (size_t i = 0; i < sizeof(medievil_hzb_builds) / sizeof(medievil_hzb_builds[0]); ++i) {
        const MedievilHzbSites *s = &medievil_hzb_builds[i];
        if (medievil_hzb_signatures_match(image, size, s) && medievil_hzb_submit_signatures_match(image, size, s))
            return s;
    }
    return NULL;
}
enum { MEDIEVIL_HZB_SUBMIT_STUB_LENGTH = 30 };
static inline void medievil_hzb_build_submit_stub(unsigned char code[MEDIEVIL_HZB_SUBMIT_STUB_LENGTH],
                                                  uintptr_t helper, uintptr_t submit) {
    static const unsigned char pattern[MEDIEVIL_HZB_SUBMIT_STUB_LENGTH] = {
        0x57,0x56,0x52,                     /* push rdi; push rsi; push rdx */
        0x48,0xb8,0,0,0,0,0,0,0,0,          /* movabs rax, helper */
        0xff,0xd0,                          /* call rax */
        0x5a,0x5e,0x5f,                     /* pop rdx; pop rsi; pop rdi */
        0x48,0xb8,0,0,0,0,0,0,0,0,          /* movabs rax, submit */
        0xff,0xe0                           /* jmp rax */
    };
    memcpy(code, pattern, sizeof(pattern));
    memcpy(code + 5, &helper, 8);
    memcpy(code + 20, &submit, 8);
}
/* MapResults' "no results" buffer ({0xff}: everything visible): MedievilHzbSites::first_frame_buffer. */
#endif
