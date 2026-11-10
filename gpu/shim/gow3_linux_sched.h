// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 #7 / #8 (research/docs/study-deck-apu-deep-dive.md 1.3, 4.1 and section 5;
// impl-deck-build-sched-pool.md): Linux scheduling for the threads on the frame's critical chain on
// the Steam Deck (4 cores, 8 threads, ~77 threads of ours at nice 0). Opt-in, read at start, no
// root, never realtime scheduling. Each prints one line per thread: what it did, or why it could not.
// - GOW3_STAGE_B_CORE_SCHED=1 (#7): stage B (gow3:DrawRec) gets a core-scheduling cookie of its own
//   (prctl PR_SCHED_CORE, Linux 5.14+ with CONFIG_SCHED_CORE): while it runs, its SMT sibling runs
//   no task without that cookie and idles instead, so stage B has the whole core (Zen 2 halves the
//   ROB and queues between busy siblings). When stage B sleeps the sibling runs anything.
// - GOW3_THREAD_NICE=1 (#8): stage B and the vblank thread (shadPS4:PresentThread) at nice -10
//   (GOW3_THREAD_NICE=-N: nice -N, down to -20), by setpriority when RLIMIT_NICE allows it, else
//   through rtkit's MakeThreadHighPriority (D-Bus, system bus; rtkit allows down to -15 by
//   default); the vblank thread's timer slack also drops to 1 ns (PR_SET_TIMERSLACK).
// Threads these two create later inherit the cookie and the nice level. The nice level switches
// live (PollLive, toggle file second word 32768: nice 0; clear: the start's nice again).
// Windows: nothing (the functions do nothing).
#pragma once

namespace Gow3LinuxSched {

/// On stage B's thread, once at its start.
void StageBThread();

/// On the vblank thread, once at its start.
void VBlankThread();

/// On the vblank thread, every tick: with GOW3_THREAD_NICE at start, the toggle file's second
/// word 32768 (Gow3Toggle2::ThreadNiceOff) puts both threads back at nice 0 live, and clearing it
/// gives them the start's nice again (a helper thread makes the calls; rtkit can take a while).
void PollLive();

} // namespace Gow3LinuxSched
