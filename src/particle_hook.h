/* SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef MEDIEVIL_PARTICLE_HOOK_H
#define MEDIEVIL_PARTICLE_HOOK_H
#include <stdint.h>
#include <string.h>

/* MediEvil (CUSA11227) GPU particle state flip guard (research/docs/study-particle-fixed-step.md).
 * UE 4.21's GPU particles keep their state in two texture pairs; FFXSystem::PreInitViews flips
 * which pair is "current" (AdvanceGPUParticleFrame, 01.00 0x23667e0: FrameIndex ^= 1 at
 * ParticleSimulationResources+0x1f8) and FFXSystem::PreRender simulates the main phase from the
 * other pair into it (SimulateGPUParticles 0x2366cb0, after UpdateMultiGPUResources' call at
 * 0x230df0a). With r.GPUParticle.FixDeltaSeconds 0 (the single pass) the simulation reads the
 * pair written one simulation ago, so a flip without a simulation, or a simulation without a
 * flip, loses a step; the default 1/30 s copies the result into the other pair every time
 * (the second pass) and does not care. The guard keeps one flip per simulation: a flip with no
 * simulation since the previous one is skipped, a simulation with no flip since the previous
 * one gets it first. Both hooks only count (and change nothing) with the guard off.
 * Per game build (research/docs/plan-0102.md: the particle code sits at +0x2dd0 in 01.02); the
 * build whose bytes match is used. */
typedef struct { uint32_t pre_init_jump, pre_render_call, advance, multi_gpu; } MedievilParticleSites;
static const MedievilParticleSites medievil_particle_builds[] = {
    {0x230de02, 0x230df0a, 0x23667e0, 0x236a050}, /* 01.00 */
    {0x2310bd2, 0x2310cda, 0x23695b0, 0x236ce20}, /* 01.02 */
};
/* PreInitViews' tail: pop rbp; jmp AdvanceGPUParticleFrame; pop rbp; ret */
static const unsigned char medievil_particle_pre_init_bytes[] = {0x5d, 0xe9};
/* PreRender: mov rdi, r14; mov rsi, rbx; call UpdateMultiGPUResources; mov rdi, rbx */
static const unsigned char medievil_particle_pre_render_bytes[] = {0x4c, 0x89, 0xf7, 0x48, 0x89, 0xde, 0xe8};
/* AdvanceGPUParticleFrame: push rbp; mov rbp, rsp; push rbx; push rax; mov rbx, rdi;
 * mov rdi, [rbx+0x78]; xor dword [rdi+0x1f8], 1 */
static const unsigned char medievil_particle_advance_bytes[] = {
    0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x7b, 0x78,
    0x83, 0xb7, 0xf8, 0x01, 0x00, 0x00, 0x01
};

static inline int medievil_particle_rel_to(const unsigned char *image, uint32_t at, uint32_t target) {
    int32_t rel;
    memcpy(&rel, image + at + 1, 4);
    return (int64_t)at + 5 + rel == (int64_t)target;
}
/* The build whose two hook sites (checked with their targets: the displacements differ between
 * builds) and AdvanceGPUParticleFrame's opening match, or NULL. */
static inline const MedievilParticleSites *medievil_particle_find(const unsigned char *image, uint64_t size) {
    for (size_t i = 0; i < sizeof(medievil_particle_builds) / sizeof(medievil_particle_builds[0]); ++i) {
        const MedievilParticleSites *s = &medievil_particle_builds[i];
        if (size < s->multi_gpu + 16 || size < s->advance + sizeof(medievil_particle_advance_bytes)) continue;
        if (!memcmp(image + s->pre_init_jump - 1, medievil_particle_pre_init_bytes, 2) &&
            medievil_particle_rel_to(image, s->pre_init_jump, s->advance) &&
            !memcmp(image + s->pre_render_call - 6, medievil_particle_pre_render_bytes, 7) &&
            medievil_particle_rel_to(image, s->pre_render_call, s->multi_gpu) &&
            !memcmp(image + s->advance, medievil_particle_advance_bytes, sizeof(medievil_particle_advance_bytes)))
            return s;
    }
    return NULL;
}

/* PreInitViews' jump to AdvanceGPUParticleFrame goes here: the argument registers are kept around
 * the helper (rdi: the FX system), which returns 1 to flip (jump on to the original) or 0 to
 * return to PreInitViews' caller. rsp is 16-byte aligned at the helper call (8 at entry, 6
 * pushes, 8 more). */
enum { MEDIEVIL_PARTICLE_ADVANCE_STUB_LENGTH = 53 };
static inline void medievil_particle_build_advance_stub(unsigned char code[MEDIEVIL_PARTICLE_ADVANCE_STUB_LENGTH],
                                                        uintptr_t helper, uintptr_t advance) {
    static const unsigned char pattern[MEDIEVIL_PARTICLE_ADVANCE_STUB_LENGTH] = {
        0x57,0x56,0x52,0x51,                /* push rdi; push rsi; push rdx; push rcx */
        0x41,0x50,0x41,0x51,                /* push r8; push r9 */
        0x48,0x83,0xec,0x08,                /* sub rsp, 8 */
        0x48,0xb8,0,0,0,0,0,0,0,0,          /* movabs rax, helper */
        0xff,0xd0,                          /* call rax */
        0x48,0x83,0xc4,0x08,                /* add rsp, 8 */
        0x41,0x59,0x41,0x58,                /* pop r9; pop r8 */
        0x59,0x5a,0x5e,0x5f,                /* pop rcx; pop rdx; pop rsi; pop rdi */
        0x84,0xc0,                          /* test al, al */
        0x74,0x0c,                          /* jz ret */
        0x48,0xb8,0,0,0,0,0,0,0,0,          /* movabs rax, advance */
        0xff,0xe0,                          /* jmp rax */
        0xc3                                /* ret */
    };
    memcpy(code, pattern, sizeof(pattern));
    memcpy(code + 14, &helper, 8);
    memcpy(code + 42, &advance, 8);
}

/* PreRender's call to UpdateMultiGPUResources goes here (before PrepareGPUSimulation and the
 * main-phase simulation): the helper (rdi: the FX system) returns 1 when this simulation needs
 * a flip first; then AdvanceGPUParticleFrame(rdi) is called; then on to UpdateMultiGPUResources
 * with the registers as they came (the return address is PreRender's). */
enum { MEDIEVIL_PARTICLE_PRE_RENDER_STUB_LENGTH = 69 };
static inline void medievil_particle_build_pre_render_stub(unsigned char code[MEDIEVIL_PARTICLE_PRE_RENDER_STUB_LENGTH],
                                                           uintptr_t helper, uintptr_t advance,
                                                           uintptr_t multi_gpu) {
    static const unsigned char pattern[MEDIEVIL_PARTICLE_PRE_RENDER_STUB_LENGTH] = {
        0x57,0x56,0x52,0x51,                /* push rdi; push rsi; push rdx; push rcx */
        0x41,0x50,0x41,0x51,                /* push r8; push r9 */
        0x48,0x83,0xec,0x08,                /* sub rsp, 8 */
        0x48,0xb8,0,0,0,0,0,0,0,0,          /* movabs rax, helper */
        0xff,0xd0,                          /* call rax */
        0x84,0xc0,                          /* test al, al */
        0x74,0x11,                          /* jz restore */
        0x48,0x8b,0x7c,0x24,0x30,           /* mov rdi, [rsp+0x30] (the saved rdi) */
        0x48,0xb8,0,0,0,0,0,0,0,0,          /* movabs rax, advance */
        0xff,0xd0,                          /* call rax */
        0x48,0x83,0xc4,0x08,                /* restore: add rsp, 8 */
        0x41,0x59,0x41,0x58,                /* pop r9; pop r8 */
        0x59,0x5a,0x5e,0x5f,                /* pop rcx; pop rdx; pop rsi; pop rdi */
        0x48,0xb8,0,0,0,0,0,0,0,0,          /* movabs rax, multi_gpu */
        0xff,0xe0                           /* jmp rax */
    };
    memcpy(code, pattern, sizeof(pattern));
    memcpy(code + 14, &helper, 8);
    memcpy(code + 35, &advance, 8);
    memcpy(code + 59, &multi_gpu, 8);
}
#endif
