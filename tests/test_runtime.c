#define _GNU_SOURCE
#include "runtime.h"
#include <stdlib.h>

/* The settings menu of the GPU library restarts through probe.c, which tests do not link. */
void runtime_restart(void) { abort(); }
/* The guest image lives in probe.c too: no image in tests (the UE4 hooks look for none). */
unsigned char *runtime_image_base(void) { return NULL; }
uint64_t runtime_image_size(void) { return 0; }
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

typedef int (ABI *Register)(GuestCallback);
typedef int (ABI *RegisterCxa)(void (ABI *)(void *), void *, void *);
typedef int (ABI *Acquire)(uint64_t *);
typedef void (ABI *Guard)(uint64_t *);
typedef int32_t (ABI *AttrInit)(void **);
typedef int32_t (ABI *AttrType)(void **, int);
typedef int32_t (ABI *MutexInit)(void **, void **, const char *);
typedef int32_t (ABI *HandleOp)(void **);
typedef int32_t (ABI *Allocate)(int64_t, int64_t, uint64_t, uint64_t, int, int64_t *);
typedef int32_t (ABI *Map)(void **, uint64_t, int, int, int64_t, uint64_t);
typedef int32_t (ABI *Unmap)(void *, uint64_t);
typedef int32_t (ABI *Release)(uint64_t, uint64_t);
#define GET(type, name) ((type)runtime_resolve(name, 0))
typedef void *(ABI *TlsAddress)(const uint64_t *);
typedef void *(ABI *ThreadSelf)(void);
typedef int32_t (ABI *ThreadGetAttr)(void *,void **);
typedef int32_t (ABI *ThreadAffinity)(void **,uint64_t *);
static void *main_identity;
static void *tls_worker(void *unused) {
    (void)unused;
    const uint64_t index[]={2,0};
    unsigned char *p=GET(TlsAddress,"vNe1w4diLCs#libkernel")(index);
    assert(p[0]==42 && p[1]==17 && p[31]==0);
    p[0]=99;
    assert(GET(ThreadSelf,"aI+OeCz8xrQ#libkernel")()!=main_identity);
    return NULL;
}
static void libc_support(void) {
    static const unsigned char initial[]={42,17};
    runtime_set_libc_tls(initial,sizeof(initial),32);
    const uint64_t index[]={2,0}, last[]={2,31};
    TlsAddress tls=GET(TlsAddress,"vNe1w4diLCs#libkernel");
    unsigned char *p=tls(index);
    assert(p[0]==42 && p[1]==17 && p[2]==0 && p[31]==0);
    assert(tls(last)==p+31 && tls(index)==p);
    p[0]=23;
    main_identity=GET(ThreadSelf,"aI+OeCz8xrQ#libkernel")();
    pthread_t worker;
    assert(pthread_create(&worker,NULL,tls_worker,NULL)==0);
    assert(pthread_join(worker,NULL)==0);
    assert(p[0]==23 && initial[0]==42);
    void *attr=NULL;
    AttrInit init=GET(AttrInit,"nsYoNRywwNg#libkernel");
    ThreadGetAttr get=GET(ThreadGetAttr,"x1X76arYMxU#libkernel");
    ThreadAffinity affinity=GET(ThreadAffinity,"8+s5BzZjxSg#libkernel");
    HandleOp destroy=GET(HandleOp,"62KCwEMmzcM#libkernel");
    uint64_t mask=0;
    assert((uint32_t)init(NULL)==0x80020016);
    assert(init(&attr)==0 && get(main_identity,&attr)==0);
    assert(affinity(&attr,&mask)==0 && mask==0x7f);
    assert((uint32_t)get(&mask,&attr)==0x80020003);
    assert((uint32_t)affinity(&attr,NULL)==0x80020016);
    assert(destroy(&attr)==0 && !attr);
    assert((uint32_t)affinity(&attr,&mask)==0x80020016);
    uint64_t param[8]={64};
    runtime_set_procparam(param);
    assert(GET(ThreadSelf,"959qrazPIrg#libkernel")()==param);
    typedef void (ABI *SetHeap)(void **);
    void *heap[10]={0};
    GET(SetHeap,"p5EcQeEeJAE#libkernel")(heap);
    assert(runtime_application_heap_api()==heap);
    GET(SetHeap,"p5EcQeEeJAE#libkernel")(NULL);
    assert(runtime_application_heap_api()==NULL);
    runtime_set_procparam(NULL);
}
static int order[512], count;
static Register register_plain;
static ABI void first(void) { order[count++]=1; }
static ABI void second(void) { order[count++]=2; }
static ABI void with_arg(void *p) { order[count++]=(int)(uintptr_t)p; }
static ABI void add_handler(void) { order[count++]=4; assert(register_plain(first)==0); }

static void exit_handlers(void) {
    register_plain=GET(Register,"8G2LB+A3rzg#libc");
    RegisterCxa cxa=GET(RegisterCxa,"tsvEmnenz48#libc");
    assert(register_plain && cxa);
    assert(register_plain(first)==0);
    assert(cxa(with_arg,(void *)3,(void *)11)==0);
    assert(register_plain(second)==0);
    runtime_finalize((void *)11);
    assert(count==1 && order[0]==3);
    runtime_finalize((void *)11); assert(count==1);
    runtime_finalize(NULL);
    assert(count==3 && order[1]==2 && order[2]==1);
    runtime_finalize(NULL); assert(count==3);
    assert(register_plain(add_handler)==0);
    runtime_finalize(NULL);
    assert(count==5 && order[3]==4 && order[4]==1);
    /* Exercise reallocating the registry and strict LIFO ordering. */
    count=0;
    for (uintptr_t i=0; i<200; ++i) assert(cxa(with_arg,(void *)i,(void *)12)==0);
    runtime_finalize((void *)12);
    assert(count==200);
    for (int i=0; i<200; ++i) assert(order[i]==199-i);
}
static void guards(void) {
    uint64_t state=0;
    Acquire acquire=GET(Acquire,"3GPpjQdAMTw#libc");
    Guard release=GET(Guard,"9rAeANT2tyE#libc"), abort_guard=GET(Guard,"2emaaluWzUw#libc");
    assert(acquire(&state)==1 && state==(UINT64_C(1)<<32));
    abort_guard(&state); assert(state==0);
    assert(acquire(&state)==1);
    release(&state); assert(state==1);
    assert(acquire(&state)==0);
}
static void mutexes(void) {
    AttrInit attr_init=GET(AttrInit,"F8bUHwAG284#libkernel");
    AttrType type=GET(AttrType,"iMp8QpE+XO4#libkernel");
    HandleOp attr_destroy=GET(HandleOp,"smWEktiyyG0#libkernel");
    MutexInit init=GET(MutexInit,"cmo1RIYva9o#libkernel");
    HandleOp lock=GET(HandleOp,"9UK1vLZQft4#libkernel"), unlock=GET(HandleOp,"tn3VlD0hG60#libkernel");
    HandleOp trylock=GET(HandleOp,"upoVrzMHFeE#libkernel"), destroy=GET(HandleOp,"2Of0f+3mhhE#libkernel");
    void *attr=NULL, *mutex=NULL;
    assert((uint32_t)attr_init(NULL)==0x80020016);
    assert(attr_init(&attr)==0);
    assert((uint32_t)type(&attr,99)==0x80020016);
    assert(type(&attr,2)==0 && init(&mutex,&attr,"recursive-test")==0);
    assert(attr_destroy(&attr)==0 && attr==NULL);
    assert(lock(&mutex)==0 && trylock(&mutex)==0);
    assert((uint32_t)destroy(&mutex)==0x80020010);
    assert(unlock(&mutex)==0 && unlock(&mutex)==0 && destroy(&mutex)==0);
    assert((uint32_t)lock(&mutex)==0x80020016);
    mutex=NULL;
    assert(lock(&mutex)==0); /* static initialization */
    assert((uint32_t)lock(&mutex)==0x8002000b);
    assert((uint32_t)trylock(&mutex)==0x80020010);
    assert(unlock(&mutex)==0 && destroy(&mutex)==0);
}
static void posix_mutexes(void) {
    AttrInit init=GET(AttrInit,"dQHWEsJtoE4#libkernel");
    AttrType type=GET(AttrType,"mDmgMOGVUqg#libkernel");
    HandleOp destroy_attr=GET(HandleOp,"HF7lK46xzjY#libkernel");
    typedef int32_t (ABI *PosixInit)(void **,void **);
    PosixInit mutex_init=GET(PosixInit,"ttHNfU+qDBU#libkernel");
    HandleOp lock=GET(HandleOp,"7H0iTOciTLo#libkernel");
    HandleOp unlock=GET(HandleOp,"2Z+PpY6CaJg#libkernel");
    HandleOp trylock=GET(HandleOp,"K-jXhbt2gn4#libkernel");
    HandleOp destroy=GET(HandleOp,"ltCfaGr2JGE#libkernel");
    void *a=NULL,*m=NULL;
    assert(init(NULL)==22 && init(&a)==0);
    assert(type(&a,99)==22 && type(&a,2)==0);
    assert(mutex_init(&m,&a)==0 && destroy_attr(&a)==0);
    assert(lock(&m)==0 && trylock(&m)==0 && destroy(&m)==16);
    assert(unlock(&m)==0 && unlock(&m)==0 && destroy(&m)==0);
    m=NULL;
    assert(lock(&m)==0 && lock(&m)==11 && trylock(&m)==16);
    assert(unlock(&m)==0 && destroy(&m)==0);
    assert(!runtime_resolve("dQHWEsJtoE4#libc",0));
}
static void wall_time(void) {
    typedef int (ABI *GetTime)(int64_t *,int32_t *);
    GetTime get=GET(GetTime,"n88vx3C5nW8#libkernel");
    struct { int64_t value[2]; uint64_t canary; } t={{0,0},0xabcdef};
    struct { int32_t value[2]; uint32_t canary; } z={{0,0},0x123456};
    time_t before=time(NULL);
    assert(get(t.value,z.value)==0);
    time_t after=time(NULL);
    assert(t.value[0]>=before && t.value[0]<=after);
    assert(t.value[1]>=0 && t.value[1]<1000000);
    assert(t.canary==0xabcdef && z.canary==0x123456);
    assert(get(NULL,NULL)==0 && get(NULL,z.value)==0);
    assert(!runtime_resolve("n88vx3C5nW8#libc",0));
}
/* Kept pool pages (Linux, runtime_memory.c pool_keep_locked): a range decommitted and committed
 * again at its address gets its own bytes back; a range committed elsewhere that has to take
 * those pages reads zero, not their bytes (the Hilltop Mausoleum's garbage lighting). With
 * GOW3_POOL_KEEP_PAGES=0 it only prints what the old behaviour does. */
typedef int32_t (ABI *PoolExpand)(int64_t, int64_t, uint64_t, uint64_t, int64_t *);
typedef int32_t (ABI *PoolReserve)(void *, uint64_t, uint64_t, int, void **);
typedef int32_t (ABI *PoolCommit)(void *, uint64_t, int, int, int);
typedef int32_t (ABI *PoolDecommit)(void *, uint64_t, int);
static void pool_pages(void) {
    PoolExpand expand=GET(PoolExpand,"qCSfqDILlns#libkernel");
    PoolReserve reserve=GET(PoolReserve,"pU-QydtGcGY#libkernel");
    PoolCommit commit=GET(PoolCommit,"Vzl66WmfLvk#libkernel");
    PoolDecommit decommit=GET(PoolDecommit,"LXo1tpFqJGs#libkernel");
    const char *keep_env=getenv("GOW3_POOL_KEEP_PAGES");
    const int keep=!(keep_env && keep_env[0]=='0');
    const uint64_t size=65536; /* 4 pool pages of 16 KiB: the whole pool */
    int64_t phys=-1;
    void *base=NULL;
    assert(expand(0,536870912,size,65536,&phys)==0);
    assert(reserve(NULL,2*size,65536,0,&base)==0 && base);
    unsigned char *one=base, *two=(unsigned char *)base+size;
    /* 1. own bytes back */
    assert(commit(one,size,0,3,0)==0);
    for (uint64_t p=0;p<size;p+=16384) memset(one+p,0xA5,64);
    assert(decommit(one,size,0)==0);
    assert(commit(one,size,0,3,0)==0);
    int own=1;
    for (uint64_t p=0;p<size;p+=16384) own&=one[p]==0xA5 && one[p+63]==0xA5;
    /* 2. another range on a full pool takes those pages: zero, not their bytes */
    assert(decommit(one,size,0)==0);
    assert(commit(two,size,0,3,0)==0);
    int zero2=1;
    for (uint64_t p=0;p<size;p+=16384) zero2&=two[p]==0 && two[p+63]==0;
    /* 3. and back: the first range takes the second's pages, zero again */
    for (uint64_t p=0;p<size;p+=16384) memset(two+p,0x5A,64);
    assert(decommit(two,size,0)==0);
    assert(commit(one,size,0,3,0)==0);
    int zero1=1;
    for (uint64_t p=0;p<size;p+=16384) zero1&=one[p]==0 && one[p+63]==0;
    assert(decommit(one,size,0)==0);
    printf("pool pages (kept %s): own bytes back %s, another range's bytes leaked %s / %s\n",
           keep ? "on" : "off", own ? "yes" : "no", zero2 ? "no" : "YES", zero1 ? "no" : "YES");
    if (keep) assert(own && zero2 && zero1);
}
static void direct_memory(void) {
    Allocate alloc=GET(Allocate,"rTXw65xmLIA#libkernel");
    Map map=GET(Map,"L-Q3LEjIbgA#libkernel");
    Unmap unmap=GET(Unmap,"cQke9UuBQOk#libkernel");
    Release release=GET(Release,"MBuItvba6z8#libkernel");
    int64_t a=-1,b=-1,c=-1;
    const uint64_t pool=536870912, length=16384;
    assert((uint32_t)alloc(0,pool,1,0,0,&a)==0x80020016 && a==-1);
    assert((uint32_t)alloc(0,8192,length,0,0,&a)==0x80020023);
    assert(alloc(0,pool,length,2097152,0,&a)==0 && a==0);
    assert(alloc(0,pool,length,2097152,0,&b)==0 && b==2097152);
    assert(alloc(0,pool,length,16384,0,&c)==0 && c==16384);
    void *x=NULL,*y=NULL,*invalid=NULL;
    assert((uint32_t)map(&invalid,length*2,3,0,b,0)==0x80020016);
    assert(map(&x,length,3,0,a,2097152)==0 && !((uintptr_t)x%2097152));
    assert(map(&y,length,3,0,a,0)==0);
    assert(*(uint64_t *)x==0);
    *(uint64_t *)x=UINT64_C(0xabcdef0123456789);
    assert(*(uint64_t *)y==UINT64_C(0xabcdef0123456789)); /* real shared backing */
    assert((uint32_t)unmap((char *)x+4096,length)==0x80020016); /* unaligned */
    assert(unmap(x,length)==0 && unmap(y,length)==0);
    assert(release(a,length)==0 && release(b,length)==0 && release(c,length)==0);
    assert(alloc(0,pool,length,0,0,&a)==0 && a==0);
    x=NULL; assert(map(&x,length,3,0,a,0)==0);
    assert(*(uint64_t *)x==0); /* released allocation does not leak previous data */
    assert(release(a,length)==0); /* release also unmaps owned mapping */
}
/* gow3 A1 (research/docs/impl-stage-a-decode.md): runtime_memory_copy_backing gives the bytes the
 * locked path gives (CopySparseSerial: mapped ranges from the backing view, the rest zero), or 0;
 * never a mix of two table states while maps and unmaps run on another thread. */
int runtime_memory_copy_backing(uintptr_t address, void *data, uint64_t size);
int runtime_memory_read_backing(uintptr_t address, void *data, uint64_t size);
int runtime_memory_region(uintptr_t address, uintptr_t *start, uintptr_t *end, int *mapped);
uint64_t runtime_memory_clamp(uintptr_t address, uint64_t size);
extern uint64_t runtime_disabled_optimizations2;
static void locked_copy(uintptr_t source, unsigned char *dest, uint64_t size) { /* gow3gpu.cpp's loop */
    while (size) {
        uintptr_t start=0, end=0; int mapped=0;
        if (!runtime_memory_region(source,&start,&end,&mapped)) end=source+size;
        const uint64_t n=size<end-source ? size : end-source;
        if (!(mapped && runtime_memory_read_backing(source,dest,n))) memset(dest,0,n);
        source+=n; dest+=n; size-=n;
    }
}
typedef struct { void *address; int64_t phys; _Atomic int stop; _Atomic long cycles; } FlipFixture;
static void *map_flipper(void *arg) { /* maps and unmaps one 2-page range over and over */
    FlipFixture *f=arg;
    Map map=GET(Map,"L-Q3LEjIbgA#libkernel");
    Unmap unmap=GET(Unmap,"cQke9UuBQOk#libkernel");
    while (!atomic_load(&f->stop)) {
        void *at=f->address;
        /* Pauses between the edits: copies land in both states and across edits. */
        const struct timespec pause={0,(long)(50000+atomic_load(&f->cycles)%7*20000)};
        assert(map(&at,2*16384,3,0x10,f->phys,0)==0 && at==f->address); /* MAP_FIXED */
        nanosleep(&pause,NULL);
        assert(unmap(f->address,2*16384)==0);
        nanosleep(&pause,NULL);
        atomic_fetch_add(&f->cycles,1);
    }
    return NULL;
}
static void backing_copies(void) {
    Allocate alloc=GET(Allocate,"rTXw65xmLIA#libkernel");
    Map map=GET(Map,"L-Q3LEjIbgA#libkernel");
    Unmap unmap=GET(Unmap,"cQke9UuBQOk#libkernel");
    Release release=GET(Release,"MBuItvba6z8#libkernel");
    const uint64_t page=16384, pool=536870912;
    int64_t a=-1, b=-1;
    assert(alloc(0,pool,4*page,0,0,&a)==0 && alloc(0,pool,2*page,0,0,&b)==0);
    void *x=NULL;
    assert(map(&x,4*page,3,0,a,0)==0);
    unsigned char *p=x;
    for (uint64_t i=0;i<4*page;++i) p[i]=(unsigned char)(i*7+3);
    assert(unmap(p+page,page)==0); /* a gap: reads as zero */
    static unsigned char got[4*16384+64], want[4*16384+64];
    const uint64_t offsets[]={0,5,page-100,page-1,page,page+1,2*page-3,3*page+17};
    const uint64_t sizes[]={1,64,256,4096,page,2*page,3*page-17};
    for (int round=0;round<2;++round) { /* round 1: the R1 index filled by the clamp first */
        for (size_t o=0;o<sizeof(offsets)/sizeof(*offsets);++o)
            for (size_t s=0;s<sizeof(sizes)/sizeof(*sizes);++s) {
                const uint64_t off=offsets[o], n=sizes[s];
                if (off+n>4*page) continue;
                const uintptr_t at=(uintptr_t)p+off;
                if (round) (void)runtime_memory_clamp(at,n);
                memset(got,0xee,sizeof(got)); memset(want,0xdd,sizeof(want));
                locked_copy(at,want,n);
                assert(runtime_memory_copy_backing(at,got,n)==1);
                assert(!memcmp(got,want,n) && got[n]==0xee);
            }
    }
    /* The live switch: second toggle word 1048576 sends every copy the locked way. */
    __atomic_or_fetch(&runtime_disabled_optimizations2,UINT64_C(1048576),__ATOMIC_RELAXED);
    assert(runtime_memory_copy_backing((uintptr_t)p,got,64)==0);
    __atomic_and_fetch(&runtime_disabled_optimizations2,~UINT64_C(1048576),__ATOMIC_RELAXED);
    /* Another thread maps and unmaps b at a fixed address: each copy of that range is all its
     * bytes or all zeros (one table state), and copies of the stable range stay exact. */
    void *y=NULL;
    assert(map(&y,2*page,3,0,b,0)==0);
    for (uint64_t i=0;i<2*page;++i) ((unsigned char *)y)[i]=(unsigned char)(i*13+1);
    static unsigned char pattern[2*16384], stable[4*16384];
    memcpy(pattern,y,2*page);
    assert(unmap(y,2*page)==0);
    locked_copy((uintptr_t)p,stable,4*page);
    FlipFixture f={y,b,0,0};
    pthread_t flipper;
    assert(pthread_create(&flipper,NULL,map_flipper,&f)==0);
    long copied=0, locked=0, mapped=0, empty=0;
    const time_t until=time(NULL)+10;
    for (long i=0; (i<1000 || atomic_load(&f.cycles)<300) && time(NULL)<until; ++i) {
        if (runtime_memory_copy_backing((uintptr_t)y,got,2*page)) {
            ++copied;
            if (!memcmp(got,pattern,2*page)) ++mapped;
            else {
                for (uint64_t k=0;k<2*page;++k) assert(got[k]==0);
                ++empty;
            }
        } else ++locked;
        if (runtime_memory_copy_backing((uintptr_t)p,got,4*page)) assert(!memcmp(got,stable,4*page));
    }
    atomic_store(&f.stop,1);
    assert(pthread_join(flipper,NULL)==0);
    printf("backing copies: %ld without the lock (%ld mapped, %ld unmapped), %ld the locked way, %ld map cycles\n",
           copied,mapped,empty,locked,(long)atomic_load(&f.cycles));
    assert(mapped>0 && empty>0 && atomic_load(&f.cycles)>0); /* both states seen without the lock */
    /* No writer now: every copy goes without the lock (a failed copy returns at once, so the
     * counts above are mostly fast retries while the other thread held the lock). */
    for (int i=0;i<1000;++i) {
        assert(runtime_memory_copy_backing((uintptr_t)y,got,2*page)==1);
        for (uint64_t k=0;k<2*page;++k) assert(got[k]==0); /* left unmapped */
        assert(runtime_memory_copy_backing((uintptr_t)p,got,4*page)==1 && !memcmp(got,stable,4*page));
    }
    assert(unmap(p,4*page)==0);
    assert(release(a,4*page)==0 && release(b,2*page)==0);
}
static void memory_primitives(void) {
    typedef void *(ABI *Set)(void *,int,size_t);
    typedef void *(ABI *Copy)(void *,const void *,size_t);
    typedef int (ABI *Compare)(const void *,const void *,size_t);
    char a[16],b[16];
    assert(GET(Set,"8zTFvBIAIN8#libc")(a,0x5a,sizeof(a))==a);
    assert(GET(Copy,"Q3VBxCXhUHs#libc")(b,a,sizeof(a))==b);
    assert(GET(Compare,"DfivPArhucg#libc")(a,b,sizeof(a))==0);
    memcpy(a,"abcdef",7);
    GET(Copy,"+P6FRGH4LfA#libc")(a+1,a,6);
    assert(memcmp(a,"aabcdef",7)==0);
    typedef size_t (ABI *Length)(const char *);
    Length length=GET(Length,"j4ViWNHEgww#libc");
    assert(length("")==0 && length("abc\0def")==3 && length("\xff\x80")==2);
}

static HandleOp rw_read, rw_write, rw_tryread, rw_trywrite, rw_unlock, rw_destroy;
typedef struct { int64_t seconds, nanoseconds; } TestTime;
typedef int32_t (ABI *TimedLock)(void **, const TestTime *);
static void rw_setup(void) {
    rw_read=GET(HandleOp,"Ox9i0c7L5w0#libkernel"); rw_write=GET(HandleOp,"mqdNorrB+gI#libkernel");
    rw_tryread=GET(HandleOp,"XD3mDeybCnk#libkernel"); rw_trywrite=GET(HandleOp,"bIHoZCTomsI#libkernel");
    rw_unlock=GET(HandleOp,"+L98PIbGttk#libkernel"); rw_destroy=GET(HandleOp,"BB+kb08Tl9A#libkernel");
    assert(rw_read && rw_write && rw_tryread && rw_trywrite && rw_unlock && rw_destroy);
}
static void rw_lifecycle(void) {
    rw_setup();
    MutexInit init=GET(MutexInit,"6ULAa0fq4jA#libkernel");
    void *lock;
    assert((uint32_t)init(NULL,NULL,NULL)==0x80020016);
    assert(init(&lock,NULL,"rw-test")==0);
    assert(rw_read(&lock)==0 && rw_read(&lock)==0);
    assert((uint32_t)rw_trywrite(&lock)==0x80020010);
    assert((uint32_t)rw_write(&lock)==0x8002000b);
    assert((uint32_t)rw_destroy(&lock)==0x80020010);
    assert(rw_unlock(&lock)==0 && rw_unlock(&lock)==0);
    assert((uint32_t)rw_unlock(&lock)==0x80020001);
    assert(rw_write(&lock)==0);
    assert((uint32_t)rw_tryread(&lock)==0x80020010);
    assert((uint32_t)rw_read(&lock)==0x8002000b);
    assert(rw_unlock(&lock)==0 && rw_destroy(&lock)==0);
    assert((uintptr_t)lock==1);
    assert((uint32_t)rw_read(&lock)==0x80020016);
    assert((uint32_t)rw_destroy(&lock)==0x80020016);
    lock=NULL;
    assert(rw_destroy(&lock)==0);
    assert(rw_read(&lock)==0 && rw_unlock(&lock)==0 && rw_destroy(&lock)==0);
    lock=(void *)(uintptr_t)0xdead;
    assert((uint32_t)rw_tryread(&lock)==0x80020016);
}
typedef struct {
    void *lock;
    pthread_barrier_t ready, release, writer_ready;
    _Atomic int value;
} RwFixture;
static void barrier(pthread_barrier_t *b) {
    int e=pthread_barrier_wait(b); assert(e==0 || e==PTHREAD_BARRIER_SERIAL_THREAD);
}
static void *reader_worker(void *context) {
    RwFixture *f=context;
    assert(rw_read(&f->lock)==0);
    barrier(&f->ready); barrier(&f->release);
    assert(atomic_load(&f->value)==0);
    assert(rw_unlock(&f->lock)==0); return NULL;
}
static void *writer_worker(void *context) {
    RwFixture *f=context;
    assert((uint32_t)rw_trywrite(&f->lock)==0x80020010);
    barrier(&f->writer_ready);
    assert(rw_write(&f->lock)==0);
    atomic_store(&f->value,1);
    assert(rw_unlock(&f->lock)==0); return NULL;
}
static void *holding_writer(void *context) {
    RwFixture *f=context;
    assert(rw_write(&f->lock)==0);
    barrier(&f->ready); barrier(&f->release);
    assert(rw_unlock(&f->lock)==0); return NULL;
}
static void fixture_init(RwFixture *f) {
    assert(pthread_barrier_init(&f->ready,NULL,2)==0);
    assert(pthread_barrier_init(&f->release,NULL,2)==0);
    assert(pthread_barrier_init(&f->writer_ready,NULL,2)==0);
}
static void fixture_destroy(RwFixture *f) {
    assert(rw_destroy(&f->lock)==0);
    assert(pthread_barrier_destroy(&f->ready)==0);
    assert(pthread_barrier_destroy(&f->release)==0);
    assert(pthread_barrier_destroy(&f->writer_ready)==0);
}
static void rw_concurrency(void) {
    rw_setup();
    RwFixture f={0}; fixture_init(&f);
    pthread_t reader,writer;
    assert(rw_read(&f.lock)==0);
    assert(pthread_create(&reader,NULL,reader_worker,&f)==0);
    barrier(&f.ready); /* two threads now hold read locks at once */
    assert(rw_unlock(&f.lock)==0);
    assert(pthread_create(&writer,NULL,writer_worker,&f)==0);
    barrier(&f.writer_ready);
    assert((uint32_t)rw_destroy(&f.lock)==0x80020010);
    assert(atomic_load(&f.value)==0);
    barrier(&f.release);
    assert(pthread_join(reader,NULL)==0 && pthread_join(writer,NULL)==0);
    assert(atomic_load(&f.value)==1);
    fixture_destroy(&f);
}
static void rw_timeouts(void) {
    rw_setup();
    TimedLock read_timed=GET(TimedLock,"iPtZRWICjrM#libkernel"), write_timed=GET(TimedLock,"adh--6nIqTk#libkernel");
    RwFixture f={0}; fixture_init(&f);
    pthread_t writer; assert(pthread_create(&writer,NULL,holding_writer,&f)==0);
    barrier(&f.ready);
    TestTime expired={0,0},invalid={0,1000000000};
    assert((uint32_t)read_timed(&f.lock,&expired)==0x8002003c);
    assert((uint32_t)write_timed(&f.lock,&expired)==0x8002003c);
    assert((uint32_t)read_timed(&f.lock,&invalid)==0x80020016);
    assert((uint32_t)rw_unlock(&f.lock)==0x80020001);
    barrier(&f.release); assert(pthread_join(writer,NULL)==0);
    assert(read_timed(&f.lock,&invalid)==0); /* available lock ignores deadline */
    assert(rw_unlock(&f.lock)==0);
    fixture_destroy(&f);
}
int main(int argc,char **argv) {
    setenv("GOW3_BACKING_COPY","1",1); setenv("GOW3_REGION_INDEX","1",1); /* A1 test: read once, at first use */
    runtime_start(0); assert(runtime_resolve("bzQExy189ZI#libc",0)==0);
    runtime_start(1);
    assert(runtime_resolve("bzQExy189ZI#wrong#module",0)==0);
    assert(runtime_resolve("unknown",0)==0);
    /* Keys are NID#library for every game; libScePosix shares libkernel's contracts. */
    assert(runtime_resolve("n88vx3C5nW8#libScePosix",0) &&
           runtime_resolve("n88vx3C5nW8#libScePosix",0)==runtime_resolve("n88vx3C5nW8#libkernel",0));
    assert(!strcmp(runtime_import_name("n88vx3C5nW8#libScePosix"),"gettimeofday"));
    assert(runtime_resolve("n88vx3C5nW8#libScePosixX",0)==0);
    assert(runtime_resolve("8zTFvBIAIN8#libc",1)==0);
    assert(runtime_resolve("f7uOxY9mM1U#libkernel",0)==0);
    uint64_t *canary=(uint64_t *)runtime_resolve("f7uOxY9mM1U#libkernel",1);
    assert(canary && *canary && !(*canary & 255));
    if (argc>1 && !strcmp(argv[1],"--guard-recursion")) {
        uint64_t guard=0; Acquire acquire=GET(Acquire,"3GPpjQdAMTw#libc");
        acquire(&guard); acquire(&guard); return 99;
    }
    if (argc>1 && !strcmp(argv[1],"--bad-tls")) {
        static const char data[4]={0};
        runtime_set_libc_tls(data,4,8);
        const uint64_t index[]={2,8};
        GET(TlsAddress,"vNe1w4diLCs#libkernel")(index); return 99;
    }
    if (argc>1 && !strcmp(argv[1],"--stack-failure")) {
        GET(GuestCallback,"Ou3iL1abvng#libkernel")(); return 99;
    }
    if (argc>1 && !strcmp(argv[1],"--rwlock-lifecycle")) { rw_lifecycle(); return 0; }
    if (argc>1 && !strcmp(argv[1],"--rwlock-concurrency")) { rw_concurrency(); return 0; }
    if (argc>1 && !strcmp(argv[1],"--rwlock-timeouts")) { rw_timeouts(); return 0; }
    /* A1 alone (the full run stops earlier today: posix_mutexes' destroy-while-locked check). */
    if (argc>1 && !strcmp(argv[1],"--backing-copies")) { direct_memory(); backing_copies(); return 0; }
    if (argc>1 && !strcmp(argv[1],"--pool-pages")) { pool_pages(); return 0; }
    libc_support(); posix_mutexes(); wall_time(); exit_handlers(); guards(); mutexes(); direct_memory(); memory_primitives(); backing_copies();
    rw_lifecycle(); rw_concurrency(); rw_timeouts();
    puts("PASS: callback lifecycle, guard ABI, mutex errors, shared direct memory, memory primitives, resolver scope");
    return 0;
}
