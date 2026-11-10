// gow3: host threads that may call guest code (AvPlayer allocator callbacks).
// Each thread gets a guest TCB (GS base, TLS) from the C runtime before running.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <system_error>
#include <thread>
#include "common/types.h"

extern "C" void runtime_thread_attach_host(const char* name);

namespace Libraries::Kernel {
class Thread {
public:
    Thread() = default;
    Thread(const Thread&) = delete;
    Thread& operator=(const Thread&) = delete;
    ~Thread() {
        Stop();
        // A thread destroying its own object (a player closed from its own thread) lets go.
        std::lock_guard lock(join_mutex);
        if (thread.joinable()) thread.detach();
    }
    void Run(std::function<void(std::stop_token)>&& func) {
        Stop();
        std::lock_guard lock(join_mutex);
        finished = std::make_shared<std::atomic<bool>>(false);
        thread = std::jthread([func = std::move(func), done = finished](std::stop_token stop) {
            runtime_thread_attach_host("gow3:hle");
            func(stop);
            done->store(true);
        });
        stop_source = thread.get_stop_source();
        id = thread.get_id();
    }
    // gow3: AvPlayer's decoder threads join themselves when they finish, its demuxer joins
    // them, and the game's sceAvPlayerStop joins all three, at the same time. A thread joining
    // itself returns at once (its owner joins it later; it no longer detaches), and one joiner
    // at a time: the self detach racing sceAvPlayerStop's join hung the game in pthread_join
    // after the first movie (black screen) or failed the join on Windows.
    // These are called from guest code, which a C++ exception cannot unwind: a failed join is
    // logged and the thread detached instead.
    void Join() {
        // Checked before the lock: the owner may hold it while joining this very thread.
        if (id.load() == std::this_thread::get_id()) return;
        std::lock_guard lock(join_mutex);
        if (!thread.joinable() || thread.get_id() == std::this_thread::get_id()) return;
        try {
            thread.join();
        } catch (const std::system_error& error) {
            // Windows (libc++ on Win32 threads): the join can fail although the thread is fine.
            // Wait for its function to return, so nothing it uses is freed under it, then let go.
            std::fprintf(stderr, "HLE thread: join failed (%s); waiting for it to finish\n", error.what());
            for (int waited = 0; finished && !finished->load() && waited < 5000; ++waited) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (thread.joinable()) thread.detach();
        }
        id = std::thread::id{};
    }
    // Running: started, and its function has not returned yet (a finished thread nobody has
    // joined yet counts as stopped, as when it detached itself). Never waits for a joiner.
    bool Joinable() const { return finished && !finished->load(); }
    // The stop request never waits for another joiner: the demuxer may be joining a decoder
    // that only ends once it is asked to stop.
    void Stop() {
        stop_source.request_stop();
        Join();
    }

private:
    std::mutex join_mutex;
    std::jthread thread;
    std::stop_source stop_source{std::nostopstate};
    std::atomic<std::thread::id> id;
    std::shared_ptr<std::atomic<bool>> finished;
};
} // namespace Libraries::Kernel
