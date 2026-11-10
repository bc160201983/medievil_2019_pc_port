/* PS4 rwlock handles (runtime_rwlock.c): validation through the live-lock set. Links only
   runtime_rwlock.c: cc -std=c11 -O2 -Isrc tests/test_rwlock_registry.c src/runtime_rwlock.c -pthread */
#define _GNU_SOURCE
#include "runtime.h"
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uintptr_t runtime_rwlock_resolve(const char *name);
typedef void Lock;
typedef int32_t (ABI *Init)(Lock **, void *const *, const char *);
typedef int32_t (ABI *Op)(Lock **);
#define GET(t, n) ((t)runtime_rwlock_resolve(n))
#define EINVAL_CODE ((int32_t)UINT32_C(0x80020016))
#define EPERM_CODE ((int32_t)UINT32_C(0x80020001))
#define EBUSY_CODE ((int32_t)UINT32_C(0x80020010))
#define EDEADLK_CODE ((int32_t)UINT32_C(0x8002000b))

static Init init;
static Op rd, wr, tryrd, trywr, unlock, destroy;

static void *reader(void *arg) {
    Lock **locks = arg;
    for (int round = 0; round < 2000; ++round) {
        Lock **l = &locks[round % 512];
        assert(rd(l) == 0);
        assert(unlock(l) == 0);
    }
    return NULL;
}

int main(void) {
    init = GET(Init, "6ULAa0fq4jA#libkernel");
    destroy = GET(Op, "BB+kb08Tl9A#libkernel");
    rd = GET(Op, "Ox9i0c7L5w0#libkernel");
    wr = GET(Op, "mqdNorrB+gI#libkernel");
    tryrd = GET(Op, "XD3mDeybCnk#libkernel");
    trywr = GET(Op, "bIHoZCTomsI#libkernel");
    unlock = GET(Op, "+L98PIbGttk#libkernel");
    assert(init && destroy && rd && wr && tryrd && trywr && unlock);

    /* Enough locks to grow the set several times (64 slots at first). */
    enum { N = 5000 };
    static Lock *locks[N];
    for (int i = 0; i < N; ++i) assert(init(&locks[i], NULL, "test") == 0);
    for (int i = 0; i < N; ++i) {
        assert(wr(&locks[i]) == 0);
        assert(trywr(&locks[i]) == EBUSY_CODE);   /* own write lock */
        assert(rd(&locks[i]) == EDEADLK_CODE);
        assert(unlock(&locks[i]) == 0);
        assert(unlock(&locks[i]) == EPERM_CODE);  /* not held */
    }

    /* Handles that are not live locks. */
    Lock *garbage = (Lock *)(uintptr_t)0x123450, *one = (Lock *)(uintptr_t)1;
    assert(rd(&garbage) == EINVAL_CODE);
    assert(unlock(&one) == EINVAL_CODE);

    /* Destroy every other lock: destroyed handles become 1 and no longer validate. */
    for (int i = 0; i < N; i += 2) {
        Lock *before = locks[i];
        assert(destroy(&locks[i]) == 0);
        assert(locks[i] == one);
        assert(rd(&locks[i]) == EINVAL_CODE);
        Lock *stale = before;
        assert(rd(&stale) == EINVAL_CODE);        /* the freed pointer itself */
    }
    for (int i = 1; i < N; i += 2) {
        assert(tryrd(&locks[i]) == 0);
        assert(unlock(&locks[i]) == 0);
    }

    /* Re-create into the deleted slots (and possibly reused addresses), then use them all. */
    for (int i = 0; i < N; i += 2) {
        locks[i] = NULL;
        assert(init(&locks[i], NULL, "again") == 0);
    }
    for (int i = 0; i < N; ++i) {
        assert(wr(&locks[i]) == 0);
        assert(unlock(&locks[i]) == 0);
    }

    /* A null handle is created on first use (static initialization). */
    Lock *lazy = NULL;
    assert(rd(&lazy) == 0 && lazy);
    assert(unlock(&lazy) == 0);
    assert(destroy(&lazy) == 0);

    /* Several threads on shared locks. */
    pthread_t threads[4];
    for (int t = 0; t < 4; ++t) assert(pthread_create(&threads[t], NULL, reader, locks) == 0);
    for (int t = 0; t < 4; ++t) pthread_join(threads[t], NULL);

    for (int i = 0; i < N; ++i) assert(destroy(&locks[i]) == 0);
    puts("rwlock registry: ok");
    return 0;
}
