/* GOW3_WAIT_SITES=1 (diagnostics): which guest code yields, sleeps or waits, how often, and how
 * long the calls take against what they asked for. The guest's yield / sleep / condition variable /
 * semaphore / event flag / mutex calls record their return address (the call site in guest code);
 * whichever thread crosses a 5 s boundary prints the sites with the most time spent since the last
 * report, as guest+offset (eboot image offsets, for Ghidra: add 0x400000). Off: one load.
 * Panning stalls (2026-10-19): the game thread spins while workers hand work to each other; timed
 * waits on Windows round up to the timer resolution, which this shows as actual >> requested. */
#include "runtime.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SLOTS 4096
#define REPORT_NS UINT64_C(5000000000)
static const char *const kind_names[] = {"yield", "pthread_yield", "usleep(0)", "usleep", "nanosleep",
                                         "cond_wait", "cond_timedwait", "sema_wait", "eventflag_wait",
                                         "mutex_lock", "equeue_wait"};
typedef struct {
    uint64_t key;   /* return address | (kind+1) << 56; 0 = empty (atomic) */
    uint64_t calls, ns, requested; /* atomic */
    uint64_t reported_calls, reported_ns, reported_requested; /* the reporting thread's */
} Site;
static Site sites[SLOTS];
static uint64_t next_report, lost; /* atomic */
static int reporting;

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
uint64_t runtime_wait_now(void) { return now_ns() | 1; /* never 0: 0 means "not measured" */ }

int runtime_wait_sites_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *value = getenv("GOW3_WAIT_SITES");
        enabled = value && value[0] == '1';
    }
    return enabled;
}

static void report(uint64_t now) {
    Site *top[16] = {0};
    uint64_t top_ns[16] = {0};
    for (unsigned i = 0; i < SLOTS; ++i) {
        Site *s = &sites[i];
        if (!__atomic_load_n(&s->key, __ATOMIC_ACQUIRE)) continue;
        const uint64_t ns = __atomic_load_n(&s->ns, __ATOMIC_RELAXED) - s->reported_ns;
        for (unsigned j = 0; j < 16; ++j) {
            if (ns > top_ns[j]) {
                for (unsigned k = 15; k > j; --k) { top[k] = top[k - 1]; top_ns[k] = top_ns[k - 1]; }
                top[j] = s; top_ns[j] = ns;
                break;
            }
        }
    }
    const uintptr_t image = (uintptr_t)runtime_image_base();
    printf("Wait sites (5 s, by time spent; calls/s, avg requested / actual): %llu lost\n",
           (unsigned long long)__atomic_load_n(&lost, __ATOMIC_RELAXED));
    for (unsigned j = 0; j < 16 && top[j]; ++j) {
        Site *s = top[j];
        const uint64_t key = __atomic_load_n(&s->key, __ATOMIC_ACQUIRE),
                       calls = __atomic_load_n(&s->calls, __ATOMIC_RELAXED) - s->reported_calls,
                       requested = __atomic_load_n(&s->requested, __ATOMIC_RELAXED) - s->reported_requested;
        const uintptr_t at = (uintptr_t)(key & ((UINT64_C(1) << 56) - 1));
        const unsigned kind = (unsigned)(key >> 56) - 1;
        char where[40];
        if (image && at - image < 0x20000000) snprintf(where, sizeof(where), "guest+0x%llx", (unsigned long long)(at - image));
        else snprintf(where, sizeof(where), "%p", (void *)at);
        printf("  %8.0f/s  %-14s %-20s %8.1f ms total, avg %.1f / %.1f us\n", calls / 5.0, kind_names[kind], where,
               top_ns[j] / 1e6, calls ? requested / 1000.0 / calls : 0.0, calls ? top_ns[j] / 1000.0 / calls : 0.0);
    }
    for (unsigned i = 0; i < SLOTS; ++i) {
        sites[i].reported_calls = __atomic_load_n(&sites[i].calls, __ATOMIC_RELAXED);
        sites[i].reported_ns = __atomic_load_n(&sites[i].ns, __ATOMIC_RELAXED);
        sites[i].reported_requested = __atomic_load_n(&sites[i].requested, __ATOMIC_RELAXED);
    }
    (void)now;
}

void runtime_wait_site(void *return_address, unsigned kind, uint64_t requested_ns, uint64_t ns) {
    const uint64_t key = (uint64_t)(uintptr_t)return_address | (uint64_t)(kind + 1) << 56;
    unsigned i = (unsigned)((key * UINT64_C(0x9E3779B97F4A7C15)) >> 52) & (SLOTS - 1);
    for (unsigned probe = 0; probe < 64; ++probe, i = (i + 1) & (SLOTS - 1)) {
        uint64_t seen = __atomic_load_n(&sites[i].key, __ATOMIC_ACQUIRE);
        if (!seen) {
            uint64_t empty = 0;
            if (__atomic_compare_exchange_n(&sites[i].key, &empty, key, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) seen = key;
            else seen = empty;
        }
        if (seen != key) continue;
        __atomic_fetch_add(&sites[i].calls, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&sites[i].ns, ns, __ATOMIC_RELAXED);
        __atomic_fetch_add(&sites[i].requested, requested_ns, __ATOMIC_RELAXED);
        goto counted;
    }
    __atomic_fetch_add(&lost, 1, __ATOMIC_RELAXED);
counted:;
    const uint64_t now = now_ns(), due = __atomic_load_n(&next_report, __ATOMIC_RELAXED);
    if (!due) { uint64_t zero = 0; __atomic_compare_exchange_n(&next_report, &zero, now + REPORT_NS, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED); return; }
    if (now < due) return;
    int idle = 0;
    if (!__atomic_compare_exchange_n(&reporting, &idle, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
    if (__atomic_load_n(&next_report, __ATOMIC_RELAXED) == due) {
        __atomic_store_n(&next_report, now + REPORT_NS, __ATOMIC_RELAXED);
        report(now);
    }
    __atomic_store_n(&reporting, 0, __ATOMIC_RELEASE);
}
