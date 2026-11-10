/* System functions added for MediEvil (CUSA11227, Unreal Engine 4), with the same offline,
 * single-user contracts as runtime_services.c:
 *   - kernel event flags (full implementation), small kernel/libc queries;
 *   - POSIX names for rwlocks and scheduling;
 *   - sockets unavailable (network unplugged);
 *   - NGS2 renders silence (no voices play yet);
 *   - Json2 parsing fails cleanly (callers fall back to defaults).
 * Every entry is an explicit contract; unknown functions still stop. */
#define _GNU_SOURCE
#define _CRT_RAND_S /* rand_s */
#include "runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#endif

#define ERR(n) ((int32_t)(UINT32_C(0x80020000)|(n)))
#define E_SRCH 3
#define E_BADF 9
#define E_NOMEM 12
#define E_BUSY 16
#define E_INVAL 22
#define E_NETDOWN 50
#define E_TIMEDOUT 60
#define E_CANCELED 85

/* ---- Event flags ---- */
#define EF_MAGIC 0x45464c47u
#define EF_ATTR_MULTI 0x20
#define EF_WAIT_AND 0x01
#define EF_WAIT_OR 0x02
#define EF_CLEAR_ALL 0x10
#define EF_CLEAR_PAT 0x20
typedef struct EventFlag {
    uint32_t magic, attr;
    uint64_t bits;
    unsigned waiters;
    uint64_t generation; /* bumped by cancel/delete so waiters can tell */
    int deleted;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    char name[32];
} EventFlag;
static size_t flags_created;

static EventFlag *flag(void *handle) {
    EventFlag *f=handle;
    return f && f->magic==EF_MAGIC && !f->deleted ? f : NULL;
}
static ABI int32_t ef_create(EventFlag **out, const char *name, uint32_t attr, uint64_t init, const void *param) {
    (void)param;
    if (!out) return ERR(E_INVAL);
    EventFlag *f=calloc(1,sizeof(*f));
    if (!f) return ERR(E_NOMEM);
    f->magic=EF_MAGIC; f->attr=attr; f->bits=init;
    pthread_mutex_init(&f->lock,NULL); pthread_cond_init(&f->cond,NULL);
    if (name) snprintf(f->name,sizeof(f->name),"%s",name);
    *out=f; ++flags_created;
    return 0;
}
static ABI int32_t ef_delete(EventFlag *handle) {
    EventFlag *f=flag(handle);
    if (!f) return ERR(E_SRCH);
    pthread_mutex_lock(&f->lock);
    f->deleted=1; ++f->generation;
    pthread_cond_broadcast(&f->cond);
    pthread_mutex_unlock(&f->lock);
    /* Freed lazily: a waiter may still hold the lock; the leak is bounded by deletions. */
    return 0;
}
static ABI int32_t ef_set(EventFlag *handle, uint64_t bits) {
    EventFlag *f=flag(handle);
    if (!f) return ERR(E_SRCH);
    pthread_mutex_lock(&f->lock);
    f->bits|=bits;
    pthread_cond_broadcast(&f->cond);
    pthread_mutex_unlock(&f->lock);
    return 0;
}
static ABI int32_t ef_clear(EventFlag *handle, uint64_t bits) {
    EventFlag *f=flag(handle);
    if (!f) return ERR(E_SRCH);
    pthread_mutex_lock(&f->lock);
    f->bits&=bits; /* PS4 contract: the argument is the mask of bits to keep */
    pthread_mutex_unlock(&f->lock);
    return 0;
}
static int ef_match(EventFlag *f, uint64_t bits, uint32_t mode) {
    return (mode & EF_WAIT_AND) ? (f->bits & bits)==bits : (f->bits & bits)!=0;
}
static void ef_consume(EventFlag *f, uint64_t bits, uint32_t mode, uint64_t *result) {
    if (result) *result=f->bits;
    if (mode & EF_CLEAR_ALL) f->bits=0;
    else if (mode & EF_CLEAR_PAT) f->bits&=~bits;
}
static int ef_valid_mode(uint32_t mode) {
    uint32_t wait=mode & 3, clear=mode & 0x30;
    return (wait==EF_WAIT_AND || wait==EF_WAIT_OR) && clear!=0x30 && !(mode & ~0x33u);
}
static ABI int32_t ef_poll(EventFlag *handle, uint64_t bits, uint32_t mode, uint64_t *result) {
    EventFlag *f=flag(handle);
    if (!f) return ERR(E_SRCH);
    if (!bits || !ef_valid_mode(mode)) return ERR(E_INVAL);
    pthread_mutex_lock(&f->lock);
    int32_t r=0;
    if (!(f->attr & EF_ATTR_MULTI) && f->waiters) r=ERR(1); /* EPERM: single-waiter flag in use */
    else if (ef_match(f,bits,mode)) ef_consume(f,bits,mode,result);
    else { if (result) *result=f->bits; r=ERR(E_BUSY); }
    pthread_mutex_unlock(&f->lock);
    return r;
}
static int32_t ef_wait_impl(EventFlag *handle, uint64_t bits, uint32_t mode, uint64_t *result, uint32_t *timeout_us);
static ABI int32_t ef_wait(EventFlag *handle, uint64_t bits, uint32_t mode, uint64_t *result, uint32_t *timeout_us) {
    const uint64_t requested=timeout_us ? (uint64_t)*timeout_us*1000 : 0;
    RUNTIME_WAIT_BEGIN();
    const int32_t r=ef_wait_impl(handle,bits,mode,result,timeout_us);
    RUNTIME_WAIT_END(8, requested);
    return r;
}
static int32_t ef_wait_impl(EventFlag *handle, uint64_t bits, uint32_t mode, uint64_t *result, uint32_t *timeout_us) {
    EventFlag *f=flag(handle);
    if (!f) return ERR(E_SRCH);
    if (!bits || !ef_valid_mode(mode)) return ERR(E_INVAL);
    pthread_mutex_lock(&f->lock);
    if (!(f->attr & EF_ATTR_MULTI) && f->waiters) { pthread_mutex_unlock(&f->lock); return ERR(1); }
    struct timespec deadline;
    if (timeout_us) {
        clock_gettime(CLOCK_REALTIME,&deadline);
        uint64_t ns=(uint64_t)deadline.tv_nsec+(uint64_t)*timeout_us*1000u;
        deadline.tv_sec+=(time_t)(ns/1000000000u); deadline.tv_nsec=(long)(ns%1000000000u);
    }
    uint64_t generation=f->generation;
    int32_t r=0;
    ++f->waiters;
    while (!ef_match(f,bits,mode)) {
        if (f->generation!=generation) { r=f->deleted ? ERR(E_SRCH) : ERR(E_CANCELED); break; }
        if (timeout_us) {
            if (pthread_cond_timedwait(&f->cond,&f->lock,&deadline)==ETIMEDOUT && !ef_match(f,bits,mode)) {
                r=ERR(E_TIMEDOUT); *timeout_us=0; break;
            }
        } else pthread_cond_wait(&f->cond,&f->lock);
    }
    --f->waiters;
    if (!r) ef_consume(f,bits,mode,result);
    else if (result) *result=f->bits;
    pthread_mutex_unlock(&f->lock);
    return r;
}
static ABI int32_t ef_cancel(EventFlag *handle, uint64_t bits, int32_t *waiters) {
    EventFlag *f=flag(handle);
    if (!f) return ERR(E_SRCH);
    pthread_mutex_lock(&f->lock);
    if (waiters) *waiters=(int32_t)f->waiters;
    f->bits=bits; ++f->generation;
    pthread_cond_broadcast(&f->cond);
    pthread_mutex_unlock(&f->lock);
    return 0;
}

/* ---- Small kernel and libc queries ---- */
/* Base PS4, or PS4 Pro with GOW3_NEO=1 (games then pick their Pro render path and resolution). */
static ABI int32_t is_neo_mode(void) {
    static int neo=-1;
    if (neo<0) { const char *v=getenv("GOW3_NEO"); neo=v && *v=='1'; if (neo) puts("Runtime: reporting a PS4 Pro (Neo mode)"); }
    return neo;
}
static ABI int32_t get_cpumode(void) { return 0; }        /* SCE_KERNEL_CPUMODE_6CPU */
/* gow3 GOW3_FRAME_STATS ("Thread ids" line, gpu driver.cpp): scePthreadGetthreadid calls and
 * the ones that asked the kernel (atomic). */
uint64_t runtime_thread_id_calls, runtime_thread_id_lookups;
int runtime_fault_stats_enabled(void);
static ABI int32_t thread_id(void) {
    const int stats=runtime_fault_stats_enabled();
    if (stats) __atomic_fetch_add(&runtime_thread_id_calls,1,__ATOMIC_RELAXED);
#ifdef _WIN32
    (void)stats;
    return (int32_t)GetCurrentThreadId(); /* a read of the thread block, no system call */
#else
    /* gow3 (spacepepperoni/medievil_ps4_Linux 1232650, impl-deck-cpu-bundle.md D2): UE4 asks
     * for it on every IsInRenderingThread()-style check (~150,000 calls in 4 s on its render
     * thread), and gettid is a system call each time. A thread's id never changes: cached. */
    static __thread int32_t id;
    if (!id) {
        id=(int32_t)syscall(SYS_gettid);
        if (stats) __atomic_fetch_add(&runtime_thread_id_lookups,1,__ATOMIC_RELAXED);
    }
    return id;
#endif
}
static ABI int32_t thread_name(void *thread, char *name) {
    if (!name) return ERR(E_INVAL);
    const char *n=runtime_thread_name(thread);
    if (!n) return ERR(E_SRCH);
    snprintf(name,32,"%s",n);
    return 0;
}
static ABI int32_t ok_noop(void) { return 0; }
static ABI void *null_pointer(void) { return NULL; }
static ABI int32_t not_found(void) { return ERR(E_SRCH); }
static ABI int32_t kernel_chmod(const char *path, uint16_t mode) { (void)path; (void)mode; return 0; }
static ABI int32_t kernel_utimes(const char *path, const void *times) { (void)path; (void)times; return 0; }
static ABI int32_t kernel_getdents(int fd, char *buffer, uint64_t size) {
    int64_t r=runtime_file_getdents(fd,buffer,size,NULL);
    return r<0 ? ERR(runtime_guest_errno((int)-r)) : (int32_t)r;
}
static ABI int32_t prt_aperture(int index, void **address, uint64_t *size) {
    if (index<0 || index>2 || !address || !size) return ERR(E_INVAL);
    *address=NULL; *size=0;
    return 0;
}
static ABI int32_t kernel_nanosleep(const struct timespec *request, struct timespec *remain) {
    if (!request) return ERR(E_INVAL);
    RUNTIME_WAIT_BEGIN();
#ifdef _WIN32
    compat_sleep_ns((uint64_t)request->tv_sec*1000000000u+(uint64_t)request->tv_nsec);
#else
    /* The guest's timespec (FreeBSD amd64) has the same layout as Linux x86-64's. */
    struct timespec left=*request;
    while (nanosleep(&left,&left) && errno==EINTR) {}
#endif
    RUNTIME_WAIT_END(4, (uint64_t)request->tv_sec*1000000000u+(uint64_t)request->tv_nsec);
    if (remain) remain->tv_sec=remain->tv_nsec=0;
    return 0;
}
/* sceKernelConvert{Localtime,Utc}...(time_t, ..., SceKernelTimezone*, int *dst) */
typedef struct { int32_t minuteswest, dsttime; } GuestTimezone;
static int32_t host_offset_seconds(time_t t, int *dst) {
    struct tm local, utc;
#ifdef _WIN32
    localtime_s(&local,&t); gmtime_s(&utc,&t);
#else
    localtime_r(&t,&local); gmtime_r(&t,&utc);
#endif
    if (dst) *dst=local.tm_isdst>0;
    utc.tm_isdst=local.tm_isdst;
    return (int32_t)difftime(mktime(&local),mktime(&utc));
}
static ABI int32_t local_to_utc(int64_t local, uint64_t unused, int64_t *utc, GuestTimezone *tz, int32_t *dst) {
    (void)unused;
    if (!utc) return ERR(E_INVAL);
    int d=0; int32_t off=host_offset_seconds((time_t)local,&d);
    *utc=local-off;
    if (tz) { tz->minuteswest=-off/60; tz->dsttime=d; }
    if (dst) *dst=d ? 3600 : 0;
    return 0;
}
static ABI int32_t utc_to_local(int64_t utc, int64_t *local, GuestTimezone *tz, int32_t *dst) {
    if (!local) return ERR(E_INVAL);
    int d=0; int32_t off=host_offset_seconds((time_t)utc,&d);
    *local=utc+off;
    if (tz) { tz->minuteswest=-off/60; tz->dsttime=d; }
    if (dst) *dst=d ? 3600 : 0;
    return 0;
}
static ABI uint64_t available_flexible(uint64_t *out) {
    if (!out) return (uint64_t)ERR(E_INVAL);
    *out=runtime_memory_flexible_available();
    return 0;
}
static ABI int32_t random_number(void *buffer, uint64_t size) {
    if (!buffer || size>64) return (int32_t)0x817C0016; /* SCE_RANDOM_ERROR_INVALID */
    unsigned char *p=buffer;
    for (uint64_t i=0;i<size;++i) {
#ifdef _WIN32
        unsigned v; rand_s(&v); p[i]=(unsigned char)v;
#else
        p[i]=(unsigned char)random();
#endif
    }
    return 0;
}

/* ---- POSIX rwlocks and scheduling (errno results, not SCE codes) ---- */
typedef int32_t (ABI *RwInit)(void **, void *const *, const char *);
typedef int32_t (ABI *RwOne)(void **);
static int32_t posix_code(int32_t r) { return r ? (int32_t)((uint32_t)r & 0xffff) : 0; }
static ABI int32_t posix_rw_init(void **rw, void *const *attr) {
    return posix_code(((RwInit)runtime_rwlock_resolve("6ULAa0fq4jA#libkernel"))(rw,attr,NULL));
}
#define POSIX_RW(name,nid) static ABI int32_t posix_rw_##name(void **rw) { \
    return posix_code(((RwOne)runtime_rwlock_resolve(nid))(rw)); }
POSIX_RW(destroy,"BB+kb08Tl9A#libkernel")
POSIX_RW(rdlock,"Ox9i0c7L5w0#libkernel")
POSIX_RW(wrlock,"mqdNorrB+gI#libkernel")
POSIX_RW(unlock,"+L98PIbGttk#libkernel")
typedef struct { int32_t priority; } GuestSchedParam;
static ABI int32_t posix_getschedparam(void *thread, int32_t *policy, GuestSchedParam *param) {
    (void)thread;
    if (policy) *policy=1; /* SCHED_FIFO, as the PS4 default */
    if (param) param->priority=700;
    return 0;
}
static ABI int32_t posix_setschedparam(void *thread, int32_t policy, const GuestSchedParam *param) {
    (void)thread; (void)policy; (void)param; return 0; /* host scheduler decides */
}
static ABI int32_t sched_max(int32_t policy) { (void)policy; return 767; }
static ABI int32_t sched_min(int32_t policy) { (void)policy; return 256; }
static ABI int32_t mutexattr_setprotocol(void *attr, int32_t protocol) { (void)attr; (void)protocol; return 0; }

/* ---- Sockets: network cable unplugged ---- */
static ABI int32_t socket_none(void) { *runtime_errno()=E_NETDOWN; return -1; }
static ABI int32_t socket_badf(void) { *runtime_errno()=E_BADF; return -1; }

/* ---- Coredump, PlayGo, AppContent ---- */
static ABI int32_t playgo_eta(int32_t handle, const uint16_t *chunks, uint32_t count, int64_t *eta) {
    (void)handle; (void)chunks; (void)count;
    if (!eta) return (int32_t)0x80B2000A;
    *eta=0; return 0;
}
static ABI int32_t playgo_todo(int32_t handle, void *list, uint32_t count, uint32_t *entries) {
    (void)handle; (void)list; (void)count;
    if (!entries) return (int32_t)0x80B2000A;
    *entries=0; return 0; /* whole package installed */
}
typedef struct { char data[16]; } MountPoint;
static ABI int32_t temp_mount2(uint32_t option, MountPoint *mount) {
    (void)option;
    if (!mount) return (int32_t)0x80D90002; /* SCE_APP_CONTENT_ERROR_PARAMETER */
    char host[1024];
    snprintf(host,sizeof(host),"%s/temp0",runtime_file_user_dir());
    mkdir(host,0755);
    if (runtime_file_mount("/temp0",host)) return (int32_t)0x80D90007; /* BUSY */
    memset(mount,0,sizeof(*mount));
    strcpy(mount->data,"/temp0");
    printf("Runtime: temporary data mounted at /temp0 -> %s\n",host);
    return 0;
}

/* ---- NGS2: silent renderer. Handles are distinct non-null tokens; voices never play. ---- */
static uintptr_t ngs2_next=0x1000;
static pthread_mutex_t ngs2_lock=PTHREAD_MUTEX_INITIALIZER;
static size_t ngs2_renders;
static uintptr_t ngs2_token(void) {
    pthread_mutex_lock(&ngs2_lock);
    uintptr_t t=(ngs2_next+=0x10);
    pthread_mutex_unlock(&ngs2_lock);
    return t;
}
static ABI int32_t ngs2_system_create(const void *option, const void *allocator, uintptr_t *out) {
    (void)option; (void)allocator;
    if (!out) return (int32_t)0x804A0003;
    *out=ngs2_token();
    puts("Runtime: NGS2 system created (silent renderer)");
    return 0;
}
static ABI int32_t ngs2_rack_create(uintptr_t system, uint32_t rack, const void *option, const void *allocator, uintptr_t *out) {
    (void)system; (void)rack; (void)option; (void)allocator;
    if (!out) return (int32_t)0x804A0003;
    *out=ngs2_token(); return 0;
}
static ABI int32_t ngs2_voice_handle(uintptr_t rack, uint32_t index, uintptr_t *out) {
    if (!out) return (int32_t)0x804A0003;
    *out=rack+0x100000+(uintptr_t)index*0x10; return 0;
}
static ABI int32_t ngs2_voice_state(uintptr_t voice, void *state, uint64_t size) {
    (void)voice;
    if (state && size) memset(state,0,size); /* stopped, not in use */
    return 0;
}
static ABI int32_t ngs2_voice_flags(uintptr_t voice, uint32_t *flags) {
    (void)voice;
    if (flags) *flags=0;
    return 0;
}
typedef struct { void *buffer; uint64_t size; uint32_t waveform, channels; } Ngs2RenderBuffer;
static ABI int32_t ngs2_render(uintptr_t system, Ngs2RenderBuffer *buffers, uint32_t count) {
    (void)system;
    for (uint32_t i=0;buffers && i<count;++i) if (buffers[i].buffer) memset(buffers[i].buffer,0,buffers[i].size);
    ++ngs2_renders;
    return 0;
}
static ABI int32_t ngs2_unsupported_waveform(void) { return (int32_t)0x804A0200; } /* invalid waveform */

/* ---- Json2 (C++ ABI, this in rdi). Parsing reports failure; values read as null/empty. ---- */
static unsigned char json_null[64];
static ABI void json_nothing(void *self) { (void)self; }
static ABI int32_t json_initialize(void *self, const void *param) { (void)self; (void)param; return 0; }
static ABI void json_notify_error(void *self, int32_t error, uint64_t size, void *user) {
    (void)self; (void)user;
    printf("Runtime: Json2 allocator error %d (size %llu)\n",error,(unsigned long long)size);
}
static ABI int32_t json_parse(void *self, void *value, const char *text, uint64_t size) {
    (void)self; (void)value;
    printf("Runtime: Json2 parse of %llu bytes not supported, reported as invalid\n",(unsigned long long)size);
    (void)text;
    return (int32_t)0x80920005; /* SCE_JSON_ERROR_PARSE_INVALID_CHAR */
}
static ABI uint64_t json_zero(void *self) { (void)self; return 0; }
static ABI void *json_null_value(void *self) { (void)self; return json_null; }
static ABI const char *json_c_str(void *self) { (void)self; return ""; }
static ABI int32_t json_empty(void *self) { (void)self; return 1; }
static ABI int32_t json_equals(void *self, const char *text) { (void)self; return text && !*text; }

/* ---- Console features with no PC counterpart: accepted and ignored ---- */
static ABI int32_t privacy_setting(int32_t type, int32_t *value) {
    (void)type;
    if (!value) return (int32_t)0x80A10003; /* SCE_SYSTEM_SERVICE_ERROR_PARAMETER */
    *value=0; return 0;
}

/* ---- Error dialog: logs the code the game wanted to show and finishes at once ---- */
static int32_t error_dialog_status=1; /* 1 INITIALIZED, 2 RUNNING, 3 FINISHED */
static ABI int32_t error_dialog_open(const int32_t *param) {
    if (!param) return (int32_t)0x80ED0003; /* SCE_ERROR_DIALOG_ERROR_PARAM_INVALID */
    printf("Runtime: error dialog requested for code 0x%08x (closed immediately)\n",(unsigned)param[1]);
    fflush(stdout);
    error_dialog_status=3;
    return 0;
}
static ABI int32_t error_dialog_status_get(void) { return error_dialog_status; }

/* ---- VoiceQoS (voice chat transport): endpoints exist, no packets ever arrive ---- */
static int32_t voice_next=1;
static ABI int32_t voiceqos_create(int32_t *out) {
    if (!out) return (int32_t)0x80A30002; /* SCE_VOICE_QOS_ERROR_INVALID_ARGUMENT */
    *out=__atomic_add_fetch(&voice_next,1,__ATOMIC_RELAXED);
    return 0;
}
static ABI int32_t voiceqos_get_attr(int32_t id, int32_t attr, void *value, int32_t size) {
    (void)id; (void)attr;
    if (value && size>0) memset(value,0,(size_t)size);
    return 0;
}
static ABI int32_t voiceqos_read(int32_t connection, void *data, uint32_t *size) {
    (void)connection; (void)data;
    if (size) *size=0;
    return 0;
}

/* ---- Dynamic modules: everything the game needs is linked up front ---- */
static ABI int32_t load_start_module(const char *path, uint64_t argc, const void *argv, uint32_t flags,
                                     const void *option, int32_t *result) {
    (void)argc; (void)argv; (void)flags; (void)option;
    printf("Runtime: sceKernelLoadStartModule(%s) -> not available\n",path ? path : "(null)");
    if (result) *result=0;
    return ERR(2); /* ENOENT */
}
static ABI int32_t posix_fcntl(int32_t fd, int32_t cmd, int64_t arg) {
    (void)fd; (void)arg;
    return cmd==1 || cmd==3 ? 0 : 0; /* F_GETFD/F_GETFL: no flags; set commands accepted */
}
static ABI int32_t posix_cancel(void *thread) { (void)thread; return E_SRCH; }

/* ---- HTTP extras: the network is unplugged ---- */
static ABI int32_t http_last_errno(int32_t request, int32_t *error) {
    (void)request;
    if (!error) return (int32_t)0x80431077;
    *error=65; /* EHOSTUNREACH */
    return 0;
}
static ABI int32_t http_network(void) { return (int32_t)0x80431063; } /* SCE_HTTP_ERROR_NETWORK */
static ABI int32_t http_bad_url(void) { return (int32_t)0x80431061; } /* SCE_HTTP_ERROR_INVALID_URL */

/* ---- Dialogs the offline console never shows: opening fails, status stays idle ---- */
static ABI int32_t dialog_unavailable(void) { return (int32_t)0x80B80001; } /* COMMON_DIALOG not available */
static ABI int32_t dialog_none(void) { return 0; }                           /* status NONE */

/* ---- Misc devices ---- */
static ABI int32_t net_mac(uint8_t *mac, int32_t flags) {
    (void)flags;
    if (!mac) return (int32_t)0x80410116;
    static const uint8_t fixed[6]={0x02,0x00,0x4d,0x45,0x44,0x49}; /* locally administered */
    memcpy(mac,fixed,6);
    return 0;
}
static ABI int32_t move_absent(void) { return (int32_t)0x80EE0002; }  /* SCE_MOVE_ERROR_NOT_INIT */
static ABI int32_t streaming_off(void) { return (int32_t)0x804E0002; } /* not streaming */
static ABI int32_t trophy_no_icon(void) { return (int32_t)0x80551616; } /* SCE_NP_TROPHY_ERROR_ICON_FILE_NOT_FOUND */
static ABI __attribute__((noreturn)) void load_exec(const char *path, const void *argv) {
    (void)argv;
    printf("Runtime: game asked to launch %s; exiting\n",path ? path : "(null)");
    fflush(NULL);
    _exit(0);
}

static const RuntimeExport exports[]={
    {"sceKernelLoadStartModule",load_start_module}, {"sceKernelDlsym",not_found},
    {"fcntl",posix_fcntl}, {"pthread_cancel",posix_cancel},
    {"sceHttpGetLastErrno",http_last_errno}, {"sceHttpAbortRequest",ok_noop},
    {"sceHttpGetAllResponseHeaders",http_network}, {"sceHttpGetMemoryPoolStats",http_network},
    {"sceHttpUriParse",http_bad_url}, {"sceHttpUriBuild",http_bad_url}, {"sceHttpUriEscape",http_bad_url},
    {"sceWebBrowserDialogOpen",dialog_unavailable}, {"sceWebBrowserDialogOpenForPredeterminedContent",dialog_unavailable},
    {"sceWebBrowserDialogUpdateStatus",dialog_none}, {"sceWebBrowserDialogGetResult",dialog_unavailable},
    {"sceWebBrowserDialogClose",ok_noop}, {"sceWebBrowserDialogResetCookie",ok_noop},
    {"sceGameCustomDataDialogOpenA",dialog_unavailable}, {"sceGameCustomDataDialogUpdateStatus",dialog_none},
    {"sceGameCustomDataDialogGetResultA",dialog_unavailable},
    {"sceInvitationDialogOpenA",dialog_unavailable}, {"sceInvitationDialogUpdateStatus",dialog_none},
    {"sceInvitationDialogGetResultA",dialog_unavailable},
    {"sceSigninDialogOpen",dialog_unavailable}, {"sceSigninDialogUpdateStatus",dialog_none},
    {"sceSigninDialogGetResult",dialog_unavailable},
    {"sceNpSnsFacebookDialogOpen",dialog_unavailable}, {"sceNpSnsFacebookDialogUpdateStatus",dialog_none},
    {"sceNpSnsFacebookDialogGetResult",dialog_unavailable},
    {"sceNpProfileDialogOpenA",dialog_unavailable}, {"sceNpCommerceDialogGetResult",dialog_unavailable},
    {"sceImeDialogGetPanelSizeExtended",dialog_unavailable},
    {"sceNetGetMacAddress",net_mac}, {"sceNetGetSockInfo",socket_badf},
    {"sceMoveGetDeviceInfo",move_absent}, {"sceMoveSetVibration",move_absent},
    {"sceGameLiveStreamingGetCurrentStatus2",streaming_off}, {"sceGameLiveStreamingGetProgramInfo",streaming_off},
    {"sceNpTrophyShowTrophyList",ok_noop}, {"sceNpTrophyCaptureScreenshot",ok_noop},
    {"sceNpTrophyGetGameIcon",trophy_no_icon}, {"sceNpTrophyGetGroupIcon",trophy_no_icon},
    {"sceNpTrophyGetTrophyIcon",trophy_no_icon}, {"sceNpTrophyGetGroupInfo",trophy_no_icon},
    {"sceSystemServiceLoadExec",load_exec},
    {"sceVoiceQoSInit",ok_noop}, {"sceVoiceQoSEnd",ok_noop},
    {"sceVoiceQoSCreateLocalEndpoint",voiceqos_create}, {"sceVoiceQoSDeleteLocalEndpoint",ok_noop},
    {"sceVoiceQoSCreateRemoteEndpoint",voiceqos_create}, {"sceVoiceQoSDeleteRemoteEndpoint",ok_noop},
    {"sceVoiceQoSConnect",voiceqos_create}, {"sceVoiceQoSDisconnect",ok_noop},
    {"sceVoiceQoSGetLocalEndpointAttribute",voiceqos_get_attr}, {"sceVoiceQoSSetLocalEndpointAttribute",ok_noop},
    {"sceVoiceQoSReadPacket",voiceqos_read}, {"sceVoiceQoSWritePacket",ok_noop},
    {"sceScreenShotEnable",ok_noop}, {"sceScreenShotDisable",ok_noop},
    {"sceSharePlayInitialize",ok_noop}, {"sceSharePlaySetProhibition",ok_noop}, {"sceSharePlayTerminate",ok_noop},
    {"sceGameLiveStreamingInitialize",ok_noop}, {"sceGameLiveStreamingTerminate",ok_noop},
    {"sceGameLiveStreamingEnableLiveStreaming",ok_noop},
    {"sceNetCtlTerm",ok_noop}, {"sceNpCommerceShowPsStoreIcon",ok_noop}, {"sceNpCommerceHidePsStoreIcon",ok_noop},
    {"sceSystemServiceGetPlatformPrivacySetting",privacy_setting},
    {"sceContentExportInit",ok_noop}, {"sceContentExportTerm",ok_noop},
    {"sceErrorDialogInitialize",ok_noop}, {"sceErrorDialogTerminate",ok_noop},
    {"sceErrorDialogOpen",error_dialog_open}, {"sceErrorDialogUpdateStatus",error_dialog_status_get},
    {"sceWebBrowserDialogInitialize",ok_noop}, {"sceWebBrowserDialogTerminate",ok_noop},
    {"sceGameCustomDataDialogInitialize",ok_noop}, {"sceGameCustomDataDialogTerminate",ok_noop},
    {"sceInvitationDialogInitialize",ok_noop}, {"sceInvitationDialogTerminate",ok_noop},
    /* Video recording (share button clips): accepted, nothing is recorded (as in shadPS4). */
    {"sceVideoRecordingSetInfo",ok_noop}, {"sceVideoRecordingQueryMemSize2",ok_noop},
    {"sceVideoRecordingOpen2",ok_noop}, {"sceVideoRecordingStart",ok_noop}, {"sceVideoRecordingStop",ok_noop},
    {"sceVideoRecordingClose",ok_noop}, {"sceVideoRecordingGetStatus",ok_noop},
    {"sceKernelCreateEventFlag",ef_create}, {"sceKernelDeleteEventFlag",ef_delete},
    {"sceKernelSetEventFlag",ef_set}, {"sceKernelClearEventFlag",ef_clear},
    {"sceKernelPollEventFlag",ef_poll}, {"sceKernelWaitEventFlag",ef_wait},
    {"sceKernelCancelEventFlag",ef_cancel},
    {"sceKernelIsNeoMode",is_neo_mode}, {"sceKernelGetCpumode",get_cpumode},
    {"scePthreadGetthreadid",thread_id}, {"scePthreadGetname",thread_name},
    {"sceKernelChmod",kernel_chmod}, {"sceKernelUtimes",kernel_utimes}, {"sceKernelGetdents",kernel_getdents},
    {"sceKernelGetPrtAperture",prt_aperture}, {"_nanosleep",kernel_nanosleep},
    {"sceKernelConvertLocaltimeToUtc",local_to_utc}, {"sceKernelConvertUtcToLocaltime",utc_to_local},
    {"sceKernelAvailableFlexibleMemorySize",available_flexible},
    {"_sceKernelRtldThreadAtexitIncrement",ok_noop}, {"_sceKernelRtldThreadAtexitDecrement",ok_noop},
    {"_sceKernelSetThreadAtexitCount",ok_noop}, {"_sceKernelSetThreadAtexitReport",ok_noop},
    {"_sceKernelSetThreadDtors",ok_noop},
    {"sceKernelIsAddressSanitizerEnabled",ok_noop},
    {"sceKernelGetSanitizerMallocReplaceExternal",null_pointer},
    {"sceKernelGetSanitizerNewReplaceExternal",null_pointer},
    {"sceKernelGetModuleInfoForUnwind",not_found}, {"sceKernelInternalMemoryGetModuleSegmentInfo",not_found},
    {"sceSysmoduleGetModuleInfoForUnwind",not_found}, {"sceLibcHeapGetTraceInfo",not_found},
    {"sceRandomGetRandomNumber",random_number},
    {"pthread_rwlock_init",posix_rw_init}, {"pthread_rwlock_destroy",posix_rw_destroy},
    {"pthread_rwlock_rdlock",posix_rw_rdlock}, {"pthread_rwlock_wrlock",posix_rw_wrlock},
    {"pthread_rwlock_unlock",posix_rw_unlock},
    {"pthread_getschedparam",posix_getschedparam}, {"pthread_setschedparam",posix_setschedparam},
    {"sched_get_priority_max",sched_max}, {"sched_get_priority_min",sched_min},
    {"pthread_mutexattr_setprotocol",mutexattr_setprotocol},
    {"socket",socket_none},
    {"accept",socket_badf}, {"bind",socket_badf}, {"connect",socket_badf}, {"getpeername",socket_badf},
    {"getsockname",socket_badf}, {"getsockopt",socket_badf}, {"setsockopt",socket_badf}, {"listen",socket_badf},
    {"recv",socket_badf}, {"recvfrom",socket_badf}, {"send",socket_badf}, {"sendto",socket_badf},
    {"shutdown",socket_badf}, {"select",socket_badf},
    {"sceCoredumpRegisterCoredumpHandler",ok_noop}, {"sceCoredumpWriteUserData",ok_noop},
    {"scePlayGoGetEta",playgo_eta}, {"scePlayGoGetToDoList",playgo_todo}, {"scePlayGoPrefetch",ok_noop},
    {"sceAppContentTemporaryDataMount2",temp_mount2},
    {"sceNgs2SystemCreateWithAllocator",ngs2_system_create}, {"sceNgs2SystemDestroy",ok_noop},
    {"sceNgs2RackCreateWithAllocator",ngs2_rack_create}, {"sceNgs2RackDestroy",ok_noop},
    {"sceNgs2RackGetVoiceHandle",ngs2_voice_handle}, {"sceNgs2VoiceControl",ok_noop},
    {"sceNgs2VoiceGetState",ngs2_voice_state}, {"sceNgs2VoiceGetStateFlags",ngs2_voice_flags},
    {"sceNgs2SystemRender",ngs2_render}, {"sceNgs2PanInit",ok_noop}, {"sceNgs2PanGetVolumeMatrix",ok_noop},
    {"sceNgs2ParseWaveformData",ngs2_unsupported_waveform}, {"sceNgs2CalcWaveformBlock",ngs2_unsupported_waveform},
    {"_ZN3sce4Json11InitializerC1Ev",json_nothing}, {"_ZN3sce4Json11InitializerD1Ev",json_nothing},
    {"_ZN3sce4Json11Initializer10initializeEPKNS0_13InitParameterE",json_initialize},
    {"_ZN3sce4Json12MemAllocatorC2Ev",json_nothing}, {"_ZN3sce4Json12MemAllocatorD2Ev",json_nothing},
    {"_ZN3sce4Json12MemAllocator11notifyErrorEimPv",json_notify_error},
    {"_ZN3sce4Json5ValueC1Ev",json_nothing}, {"_ZN3sce4Json5ValueD1Ev",json_nothing},
    {"_ZN3sce4Json6Parser5parseERNS0_5ValueEPKcm",json_parse},
    {"_ZN3sce4Json6StringC1ERKS1_",json_nothing}, {"_ZN3sce4Json6StringD1Ev",json_nothing},
    {"_ZNK3sce4Json5Value10getBooleanEv",json_zero}, {"_ZNK3sce4Json5Value10getIntegerEv",json_zero},
    {"_ZNK3sce4Json5Value11getUIntegerEv",json_zero}, {"_ZNK3sce4Json5Value5countEv",json_zero},
    {"_ZNK3sce4Json5Value9getStringEv",json_null_value},
    {"_ZNK3sce4Json5ValueixEPKc",json_null_value}, {"_ZNK3sce4Json5ValueixEm",json_null_value},
    {"_ZNK3sce4Json6String4sizeEv",json_zero}, {"_ZNK3sce4Json6String5c_strEv",json_c_str},
    {"_ZNK3sce4Json6String5emptyEv",json_empty}, {"_ZNK3sce4Json6StringeqEPKc",json_equals},
};
/* MediEvil native output (GOW3_UE4_SETRES=WxH, run.py): the base PS4 renderer makes its buffers at
 * the display size at start and only a live r.SetRes change resizes them (the debug menu's
 * Cinematic button). Once the game runs this sets the game's own r.SetRes console variable as
 * UE4's FSystemResolution::RequestResolutionChange does: flags = GetFlags() & 0xff000000 (vtable
 * +0x20), Set(value, flags) (vtable +0x68). Called from scePadReadState on the game's input thread.
 * Where the variable's IConsoleVariable* lives differs between eboots (US 01.00: 0x5c28328; the
 * 01.02 dump of PR #29: 0x5c28368). run.py finds it in RequestResolutionChange's own code
 * (ue4_override.setres_slot) and passes it as GOW3_UE4_SETRES_CVAR; without it nothing is set. */
static uintptr_t ue4_setres_slot(void) {
    static uintptr_t slot;
    static int read;
    if (!read) {
        read=1;
        const char *env=getenv("GOW3_UE4_SETRES_CVAR");
        slot=env && *env ? (uintptr_t)strtoull(env,NULL,0) : 0;
        if (!slot || slot+8>0x6000000) {
            slot=0;
            printf("Native output: r.SetRes not located in this game build; live resolution changes are off\n");
        }
    }
    return slot;
}
/* Sets a console variable on the calling (game) thread through `slot`, the image offset of its
 * IConsoleVariable* (a TAutoConsoleVariable's +8); 0 when the variable is not where expected. */
static int ue4_set_cvar(uintptr_t slot, const char *name, const char *text) {
    uint16_t value[24];
    size_t n=strlen(text);
    if (!n || n>=sizeof(value)/sizeof(value[0])) return 0;
    for (size_t i=0;i<=n;++i) value[i]=(uint16_t)(unsigned char)text[i];
    unsigned char *image=runtime_image_base();
    if (!image || !slot) return 0;
    const uintptr_t lo=(uintptr_t)image, hi=lo+0x6000000;
    uintptr_t cvar=*(const uintptr_t *)(image+slot);
    if (!cvar) { printf("UE4: %s not found\n",name); return 0; }
    uintptr_t vtable=*(const uintptr_t *)cvar;
    if (vtable<lo || vtable>=hi) { printf("UE4: unexpected %s object\n",name); return 0; }
    typedef uint32_t (ABI *GetFlags)(uintptr_t);
    typedef void (ABI *Set)(uintptr_t, const uint16_t *, uint32_t);
    uintptr_t get_flags=*(const uintptr_t *)(vtable+0x20), set=*(const uintptr_t *)(vtable+0x68);
    if (get_flags<lo || get_flags>=hi || set<lo || set>=hi) { printf("UE4: unexpected %s methods\n",name); return 0; }
    uint32_t flags=((GetFlags)get_flags)(cvar)&0xff000000u;
    ((Set)set)(cvar,value,flags);
    printf("UE4: %s %s (live)\n",name,text);
    return 1;
}
static int ue4_setres(const char *text) {
    const uintptr_t slot=ue4_setres_slot();
    return slot ? ue4_set_cvar(slot,"r.SetRes",text) : 0;
}

/* Console variables the in-game menu switches live, set on the next game-thread poll as r.SetRes
 * is. Per game build: {01.00, 01.02} (CUSA11227; research/docs/plan-0102.md). r.Shadow.Preshadows:
 * registered at 0x1611700, TAutoConsoleVariable at 0x5bcff28 (found as r.SetRes's 0x5c28320 is: the
 * lea of its name, the object beside it). The build whose registration bytes match is used. */
typedef struct { uintptr_t object, site; unsigned char code[7]; } Ue4CvarSite;
static const struct { const char *name; Ue4CvarSite at[2]; } ue4_cvars[]={
    /* site: lea rsi, [rip+name] in its registration */
    {"r.Shadow.Preshadows",{{0x5bcff28,0x1611700,{0x48,0x8d,0x35,0x1f,0x76,0x74,0x02}},{0x5bcff68,0x1614380,{0x48,0x8d,0x35,0x9f,0x7d,0x74,0x02}}}},
    /* Effects resolution; registered at 0x16651ee (0x145d993 only looks it up and keeps its data) */
    {"r.SeparateTranslucencyScreenPercentage",{{0x5bd2d68,0x16651ee,{0x48,0x8d,0x35,0x3d,0x8c,0x6f,0x02}},{0x5bd2da8,0x1667e6e,{0x48,0x8d,0x35,0xbd,0x93,0x6f,0x02}}}},
    /* Render scale (gow3_settings.cpp RequestRenderScale): the upsampling switch before the
     * percentage, as one poll sets them in this order. Objects from the registrations' own
     * `mov [object+8], rax` (research/docs/impl-live-render-scale.md). */
    {"r.TemporalAA.Upsampling",{{0x5c1a120,0x25662e6,{0x48,0x8d,0x35,0xbd,0x63,0x8d,0x01}},{0x5c1a160,0x25690b6,{0x48,0x8d,0x35,0x6d,0x66,0x8d,0x01}}}},
    {"r.ScreenPercentage",{{0x581a048,0x732124,{0x48,0x8d,0x35,0x27,0x56,0x53,0x03}},{0x581a068,0x734ce4,{0x48,0x8d,0x35,0xc7,0x5f,0x53,0x03}}}},
    /* The in-game menu's live graphics options (gow3_settings.h UE4LiveOptions); objects and sites
     * from research/docs/study-live-switches.md, each site's lea checked to load the cvar's name. */
    {"r.Shadow.MinResolution",{{0x5bd0068,0x1611cac,{0x48,0x8d,0x35,0x4f,0x7b,0x74,0x02}},{0x5bd00a8,0x161492c,{0x48,0x8d,0x35,0xcf,0x82,0x74,0x02}}}},
    {"r.Shadow.MaxResolution",{{0x581a210,0x7328e6,{0x48,0x8d,0x35,0x79,0x69,0x53,0x03}},{0x581a230,0x7354a6,{0x48,0x8d,0x35,0x19,0x73,0x53,0x03}}}},
    {"r.Shadow.DistanceScale",{{0x581a3c0,0x733053,{0x48,0x8d,0x35,0xfc,0x7e,0x53,0x03}},{0x581a3e0,0x735c13,{0x48,0x8d,0x35,0x9c,0x88,0x53,0x03}}}},
    {"r.Shadow.CSM.MaxCascades",{{0x5bc82a8,0x1586344,{0x48,0x8d,0x35,0xc7,0x88,0x7c,0x02}},{0x5bc82e8,0x1588fc4,{0x48,0x8d,0x35,0x47,0x90,0x7c,0x02}}}},
    {"r.ShadowQuality",{{0x5819f88,0x731dda,{0x48,0x8d,0x35,0xfd,0x46,0x53,0x03}},{0x5819fa8,0x73499a,{0x48,0x8d,0x35,0x9d,0x50,0x53,0x03}}}},
    {"r.ViewDistanceScale",{{0x581a2d0,0x732c2d,{0x48,0x8d,0x35,0x7e,0x73,0x53,0x03}},{0x581a2f0,0x7357ed,{0x48,0x8d,0x35,0x1e,0x7d,0x53,0x03}}}},
    {"r.StaticMeshLODDistanceScale",{{0x5bc8ef8,0x160d78a,{0x48,0x8d,0x35,0xe1,0x60,0x74,0x02}},{0x5bc8f38,0x161040a,{0x48,0x8d,0x35,0x61,0x68,0x74,0x02}}}},
    {"foliage.LODDistanceScale",{{0x5c0a180,0x2119693,{0x48,0x8d,0x35,0xaa,0xe7,0xce,0x01}},{0x5c0a1c0,0x211c423,{0x48,0x8d,0x35,0xfa,0xea,0xce,0x01}}}},
    {"r.AmbientOcclusionLevels",{{0x5b533c8,0x114b2cd,{0x48,0x8d,0x35,0x84,0x38,0xbb,0x02}},{0x5b53408,0x114df4d,{0x48,0x8d,0x35,0x04,0x40,0xbb,0x02}}}},
    {"r.SSR.Quality",{{0x5bc0228,0x14f84d3,{0x48,0x8d,0x35,0x0c,0xf2,0x84,0x02}},{0x5bc0268,0x14fb153,{0x48,0x8d,0x35,0x8c,0xf9,0x84,0x02}}}},
    {"r.VolumetricFog",{{0x5bd3bb0,0x1665ee2,{0x48,0x8d,0x35,0x0b,0x9f,0x6f,0x02}},{0x5bd3bf0,0x1668b62,{0x48,0x8d,0x35,0x8b,0xa6,0x6f,0x02}}}},
    {"r.MotionBlurQuality",{{0x5819fa0,0x731e43,{0x48,0x8d,0x35,0xb6,0x47,0x53,0x03}},{0x5819fc0,0x734a03,{0x48,0x8d,0x35,0x56,0x51,0x53,0x03}}}},
    {"r.DepthOfFieldQuality",{{0x581a030,0x7320b3,{0x48,0x8d,0x35,0x7c,0x53,0x53,0x03}},{0x581a050,0x734c73,{0x48,0x8d,0x35,0x1c,0x5d,0x53,0x03}}}},
    {"r.BloomQuality",{{0x5819f28,0x731c33,{0x48,0x8d,0x35,0x18,0x42,0x53,0x03}},{0x5819f48,0x7347f3,{0x48,0x8d,0x35,0xb8,0x4b,0x53,0x03}}}},
    {"r.LensFlareQuality",{{0x5819f10,0x731bca,{0x48,0x8d,0x35,0x43,0x41,0x53,0x03}},{0x5819f30,0x73478a,{0x48,0x8d,0x35,0xe3,0x4a,0x53,0x03}}}},
    {"r.SceneColorFringeQuality",{{0x5819f40,0x731c9c,{0x48,0x8d,0x35,0x3d,0x44,0x53,0x03}},{0x5819f60,0x73485c,{0x48,0x8d,0x35,0xdd,0x4d,0x53,0x03}}}},
    /* Single GPU particle pass (ue4_particle_pass_poll). TAutoConsoleVariable<float>, flags 0, default
     * 1/30 (vmovss of 0x3d088889 just before the site); the registration stores its result at
     * object+8 (01.00 0x5c14d20, 01.02 0x5c14d60). */
    {"r.GPUParticle.FixDeltaSeconds",{{0x5c14d18,0x23933b3,{0x48,0x8d,0x35,0xc0,0x16,0xa9,0x01}},{0x5c14d58,0x2396183,{0x48,0x8d,0x35,0xb0,0x19,0xa9,0x01}}}},
};
#define UE4_CVARS (sizeof(ue4_cvars)/sizeof(ue4_cvars[0]))
static char ue4_cvar_values[UE4_CVARS][16];
static int ue4_cvar_pending[UE4_CVARS];
void runtime_ue4_request_cvar(const char *name, const char *value) {
    if (!name || !value || strlen(value)>=sizeof(ue4_cvar_values[0])) return;
    for (size_t i=0;i<UE4_CVARS;++i) {
        if (strcmp(ue4_cvars[i].name,name)) continue;
        strcpy(ue4_cvar_values[i],value);
        __atomic_store_n(&ue4_cvar_pending[i],1,__ATOMIC_RELEASE);
    }
}

/* The single GPU particle pass (perf_particle_single_pass): r.GPUParticle.FixDeltaSeconds 0, one
 * simulation pass a frame with the real frame time, or the game's fixed 1/30 s step, which adds a
 * second, empty pass in some frames (~190 draws; research/docs/study-heavy-frame-map.md H3) but
 * keeps lone torches and the crypt ember right (plan-0102.md T-0102-4). The override pak sets the
 * start value (run.py: 0 unless GOW3_PARTICLE_FIXED_STEP=1); after that two switches decide it
 * live, at the game-thread poll: the toggle file's second word bit 8192 (Gow3Toggle2::
 * ParticleFixedStep) forces the fixed step whatever the menu says, and with the bit clear the
 * menu's choice (runtime_ue4_particle_single_pass) holds. The menu keeps its own choice while the
 * bit is set and gets it back when the bit clears. */
#define TOGGLE2_PARTICLE_FIXED_STEP UINT64_C(8192)
/* 1/30 as the game registers it (float 0x3d088889 in 01.00 and 01.02): 0.0333333 would be 9 ulps off. */
#define PARTICLE_FIXED_STEP "0.033333335"
extern uint64_t runtime_disabled_optimizations2; /* runtime_memory.c: the toggle file's second word */
static int particle_wanted=-1; /* the menu's choice: 1 single pass, 0 fixed step, -1 not drawn yet */
void runtime_ue4_particle_single_pass(int on) {
    __atomic_store_n(&particle_wanted,on ? 1 : 0,__ATOMIC_RELEASE);
}
static void ue4_particle_pass_poll(void) {
    static int start=-1, applied; /* the override pak's value; what the cvar holds now */
    if (start<0) {
        const char *fixed=getenv("GOW3_PARTICLE_FIXED_STEP");
        applied=start=!(fixed && !strcmp(fixed,"1"));
    }
    int wanted=__atomic_load_n(&particle_wanted,__ATOMIC_ACQUIRE);
    if (wanted<0) wanted=start; /* menu not opened yet: the ini's choice, as the game started */
    const int forced=(__atomic_load_n(&runtime_disabled_optimizations2,__ATOMIC_RELAXED)&TOGGLE2_PARTICLE_FIXED_STEP)!=0;
    const int on=forced ? 0 : wanted;
    if (on==applied) return;
    applied=on; /* once per switch, also when this eboot has no known address (logged once) */
    printf("Particle single pass: %s (live%s)\n",on ? "on" : "off",forced ? ", toggle file second word 8192" : "");
    runtime_ue4_request_cvar("r.GPUParticle.FixDeltaSeconds",on ? "0" : PARTICLE_FIXED_STEP);
}

/* GPU particle flip guard (src/particle_hook.h, installed by probe.c with
 * GOW3_PARTICLE_FLIP_GUARD=1): one state flip per main-phase simulation, so the single pass
 * never reads a pair older than the last simulation's output (research/docs/
 * study-particle-fixed-step.md). Both helpers run on the render thread (FSceneRenderer). The
 * toggle file's second word bit 16384 (Gow3Toggle2::ParticleFlipGuardOff) turns the guard off
 * live: the hooks then only count. Every 1,200 simulations a line gives the flips and
 * simulations since the previous line and the unpaired ones: a flip with no simulation since the
 * previous flip, a simulation with no flip since the previous simulation. */
#define TOGGLE2_PARTICLE_FLIP_GUARD_OFF UINT64_C(16384)
#define PARTICLE_FX_SLOTS 8
static struct { uintptr_t fx; int simulated; } particle_fx[PARTICLE_FX_SLOTS];
static uint64_t particle_flips, particle_sims, particle_lone_flips, particle_lone_sims;
static int particle_slot(uintptr_t fx) {
    for (int i=0;i<PARTICLE_FX_SLOTS;++i) {
        if (particle_fx[i].fx==fx) return i;
        if (!particle_fx[i].fx) { particle_fx[i].fx=fx; return i; }
    }
    return -1; /* more FX systems than slots: no guard for this one */
}
static int particle_guard_on(void) {
    return !(__atomic_load_n(&runtime_disabled_optimizations2,__ATOMIC_RELAXED)&TOGGLE2_PARTICLE_FLIP_GUARD_OFF);
}
static void particle_guard_report(void) {
    static int last_on=-1, quiet=-1;
    const int on=particle_guard_on();
    if (particle_sims<1200 && on==last_on) return;
    if (quiet<0) { /* GOW3_FPS_LOG=0: no periodic log lines (gow3_overlay.cpp LogPresent); a switch still prints */
        const char *fps_log=getenv("GOW3_FPS_LOG");
        quiet=fps_log && !strcmp(fps_log,"0");
    }
    if (quiet && on==last_on) {
        particle_flips=particle_sims=particle_lone_flips=particle_lone_sims=0;
        return;
    }
    printf("GPU particle flips: %llu flips, %llu simulations; %llu flips without a simulation (%s), "
           "%llu simulations without a flip (%s); guard %s\n",
           (unsigned long long)particle_flips,(unsigned long long)particle_sims,
           (unsigned long long)particle_lone_flips,on ? "skipped" : "kept",
           (unsigned long long)particle_lone_sims,on ? "flip added" : "none added",
           on ? "on" : "off (toggle file second word 16384)");
    last_on=on;
    particle_flips=particle_sims=particle_lone_flips=particle_lone_sims=0;
}
/* AdvanceGPUParticleFrame from PreInitViews: 1 flips (the original runs), 0 skips the flip. */
ABI int medievil_particle_advance(uintptr_t fx) {
    const int slot=particle_slot(fx);
    if (slot<0) return 1;
    int flip=1;
    if (!particle_fx[slot].simulated) {
        ++particle_lone_flips;
        flip=!particle_guard_on();
    }
    if (flip) { ++particle_flips; particle_fx[slot].simulated=0; }
    return flip;
}
/* PreRender, before the main-phase simulation: 1 flips first (AdvanceGPUParticleFrame). */
ABI int medievil_particle_pre_render(uintptr_t fx) {
    const int slot=particle_slot(fx);
    if (slot<0) return 0;
    int flip=0;
    if (particle_fx[slot].simulated) {
        ++particle_lone_sims;
        flip=particle_guard_on();
        if (flip) ++particle_flips;
    }
    particle_fx[slot].simulated=1;
    ++particle_sims;
    particle_guard_report();
    return flip;
}

/* A switch asked for by the in-game menu (window thread), done on the next game-thread poll. */
static char ue4_requested[24];
static int ue4_request_pending;
void runtime_ue4_request_setres(const char *text) {
    if (!text || strlen(text)>=sizeof(ue4_requested)) return;
    strcpy(ue4_requested,text);
    __atomic_store_n(&ue4_request_pending,1,__ATOMIC_RELEASE);
}

/* Called at every pad read. The start value (GOW3_UE4_SETRES) is set at 2 s, 5 s, 10 s and 20 s
 * of pad reads, as the viewport may not be up at the first ones (PR #29); a menu choice stops the
 * repeats. The pad reads are counted in every case: the menu's switches wait for 2 s of them. */
void runtime_ue4_setres_poll(void) {
    static int automatic_done;
    static unsigned reads;
    if (reads<1200) ++reads;
    if (__atomic_exchange_n(&ue4_request_pending,0,__ATOMIC_ACQ_REL)) {
        char text[sizeof(ue4_requested)];
        strcpy(text,ue4_requested);
        if (reads>=120) ue4_setres(text);
        automatic_done=1; /* a choice in the menu wins over the start setting */
    }
    /* Menu switches, and the ini's at start: kept until the main menu (after the engine has read
     * its own settings, which would otherwise win). */
    if (reads>=600) {
        ue4_particle_pass_poll(); /* requests its cvar for the loop below */
        for (size_t i=0;i<UE4_CVARS;++i) {
            if (!__atomic_exchange_n(&ue4_cvar_pending[i],0,__ATOMIC_ACQ_REL)) continue;
            char text[sizeof(ue4_cvar_values[0])];
            strcpy(text,ue4_cvar_values[i]);
            const unsigned char *image=runtime_image_base();
            const Ue4CvarSite *at=NULL;
            for (size_t b=0;image && b<sizeof(ue4_cvars[i].at)/sizeof(ue4_cvars[i].at[0]);++b) {
                const Ue4CvarSite *c=&ue4_cvars[i].at[b];
                if (runtime_image_size()>=c->object+16 && runtime_image_size()>=c->site+sizeof(c->code) &&
                    !memcmp(image+c->site,c->code,sizeof(c->code))) { at=c; break; }
            }
            if (!at) {
                printf("UE4: %s unavailable (not an analysed MediEvil eboot: 01.00 or 01.02)\n",ue4_cvars[i].name);
                continue;
            }
            ue4_set_cvar(at->object+8,ue4_cvars[i].name,text);
        }
    }
    if (automatic_done) return;
    const char *want=getenv("GOW3_UE4_SETRES");
    if (!want || !*want) { automatic_done=1; return; }
    if (reads==120 || reads==300 || reads==600 || reads==1200) ue4_setres(want);
    if (reads>=1200) automatic_done=1;
}

uintptr_t runtime_extra_resolve(const char *name) { return RUNTIME_LOOKUP(exports,name); }
void runtime_extra_report(void) {
    printf("Runtime: event flags created=%zu, NGS2 silent renders=%zu\n",flags_created,ngs2_renders);
}
