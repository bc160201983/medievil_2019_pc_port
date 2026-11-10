// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/vk_frame_capture.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <format>
#include <mutex>

#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image_info.h"

namespace Vulkan {

namespace {

struct Target {
    VAddr address{};
    vk::Format format{};
    u32 width{}, height{};
    bool operator==(const Target&) const = default;
};

struct Entry {
    bool compute{};
    std::vector<Target> colors;
    Target depth;
    u32 draws{};
    u64 indices{};
    std::vector<u64> shaders; ///< vs/ps pairs (graphics) or cs (compute), first few
    std::vector<std::pair<Target, bool>> sampled; ///< textures (false) and storage images (true)
    std::string note;
    std::vector<std::string> buffers; ///< dumped constants, per draw
};

// GPU thread only.
std::vector<Entry> entries;
std::vector<std::pair<Target, bool>> pending_sampled;
std::vector<std::string> pending_buffers;

void AddBuffers(Entry& entry) {
    // Constants of full-screen passes (post-processing: camera matrices) and of the first draw
    // of each geometry pass.
    if (entry.draws <= 4) {
        for (auto& b : pending_buffers) {
            entry.buffers.push_back(std::move(b));
        }
    }
    pending_buffers.clear();
}
bool pass_open = false;
std::mutex display_mutex;
std::vector<VAddr> display_buffers;
// gow3: read on every draw; the game registers a handful of display buffers once.
std::array<std::atomic<VAddr>, 16> display_list{};
std::atomic<u32> display_count{0};

Target ToTarget(const VideoCore::ImageInfo& info) {
    return {info.guest_address, info.pixel_format, info.size.width, info.size.height};
}

std::string Describe(const Target& t) {
    if (!t.address) {
        return "-";
    }
    return std::format("{:#x} {} {}x{}", t.address, vk::to_string(t.format), t.width, t.height);
}

void AddSampled(Entry& entry) {
    for (const auto& s : pending_sampled) {
        if (entry.sampled.size() < 32 &&
            std::ranges::find(entry.sampled, s) == entry.sampled.end()) {
            entry.sampled.push_back(s);
        }
    }
    pending_sampled.clear();
}

void AddShader(Entry& entry, u64 hash) {
    if (entry.shaders.size() < 8 && std::ranges::find(entry.shaders, hash) == entry.shaders.end()) {
        entry.shaders.push_back(hash);
    }
}

std::mutex result_mutex;
std::string result; ///< for the menu (FrameCapture::LastResult)

void SetResult(std::string text) {
    std::printf("Frame capture: %s\n", text.c_str());
    std::scoped_lock lk{result_mutex};
    result = std::move(text);
}

// GOW3_CAPTURE_DIR, else <user>/captures (GOW3_GPU_USER_DIR, the saves folder).
std::filesystem::path CaptureDir() {
    if (const char* dir = std::getenv("GOW3_CAPTURE_DIR"); dir && dir[0]) {
        return dir;
    }
    const char* user = std::getenv("GOW3_GPU_USER_DIR");
    return std::filesystem::path(user && user[0] ? user : "user") / "captures";
}

void Write(VAddr presented) {
    const auto dir = CaptureDir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", std::localtime(&now));
    static u32 sequence = 0;
    const std::string path = (dir / std::format("frame_{}_{}.txt", stamp, ++sequence)).string();
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::printf("Frame capture: cannot write %s\n", path.c_str());
        SetResult("Capture failed: cannot write " + path);
        return;
    }
    std::fprintf(f, "presented buffer %#llx\n", static_cast<unsigned long long>(presented));
    u32 index = 0;
    for (const auto& e : entries) {
        if (e.compute) {
            std::fprintf(f, "\n#%u COMPUTE dispatches %u, cs", index++, e.draws);
        } else {
            std::fprintf(f, "\n#%u PASS draws %u (indices %llu), vs/ps", index++, e.draws,
                         static_cast<unsigned long long>(e.indices));
        }
        for (const u64 h : e.shaders) {
            std::fprintf(f, " %016llx", static_cast<unsigned long long>(h));
        }
        std::fprintf(f, "\n");
        for (size_t i = 0; i < e.colors.size(); ++i) {
            std::fprintf(f, "  color%zu %s\n", i, Describe(e.colors[i]).c_str());
        }
        if (!e.compute) {
            std::fprintf(f, "  depth  %s\n", Describe(e.depth).c_str());
        }
        for (const auto& [t, storage] : e.sampled) {
            std::fprintf(f, "  %s %s\n", storage ? "writes " : "samples", Describe(t).c_str());
        }
        if (!e.note.empty()) {
            std::fprintf(f, "  note: %s\n", e.note.c_str());
        }
        for (const auto& b : e.buffers) {
            std::fprintf(f, "%s\n", b.c_str());
        }
    }
    std::fclose(f);
    std::printf("Frame capture: %zu passes written to %s\n", entries.size(), path.c_str());
    SetResult(std::format("Frame captured ({} passes): {}", entries.size(), path));
}

} // namespace

void FrameCapture::OnFlip(VAddr presented_address) {
    last_presented.store(presented_address, std::memory_order_relaxed);
    flips.fetch_add(1, std::memory_order_release);
    static const char* trigger = std::getenv("GOW3_CAPTURE_TRIGGER");
    if (state.load(std::memory_order_relaxed) != Idle) {
        return;
    }
    bool arm = requested.exchange(false, std::memory_order_acq_rel);
    // The trigger file is looked for 4 times a second, not at every flip.
    static auto next_check = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    const bool check = !arm && trigger && now >= next_check;
    if (check) {
        next_check = now + std::chrono::milliseconds(250);
    }
    if (check && std::filesystem::exists(trigger)) {
        std::error_code ec;
        std::filesystem::remove(trigger, ec);
        arm = true;
    }
    if (arm) {
        state.store(Armed, std::memory_order_release);
        std::printf("Frame capture: armed\n");
    }
}

void FrameCapture::Request() {
    SetResult("Capturing the next frame...");
    requested.store(true, std::memory_order_release);
}

std::string FrameCapture::LastResult() {
    std::scoped_lock lk{result_mutex};
    return result;
}

void FrameCapture::Poll() {}

bool FrameCapture::IsDisplayBuffer(VAddr address) {
    const u32 count = display_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < count; ++i) {
        if (display_list[i].load(std::memory_order_relaxed) == address) {
            return true;
        }
    }
    if (count < display_list.size()) {
        return false;
    }
    std::scoped_lock lk{display_mutex};
    return std::ranges::find(display_buffers, address) != display_buffers.end();
}

void FrameCapture::AddDisplayBuffer(VAddr address) {
    std::scoped_lock lk{display_mutex};
    if (std::ranges::find(display_buffers, address) == display_buffers.end()) {
        display_buffers.push_back(address);
        const u32 count = display_count.load(std::memory_order_relaxed);
        if (count < display_list.size()) {
            display_list[count].store(address, std::memory_order_relaxed);
            display_count.store(count + 1, std::memory_order_release);
        }
    }
}

namespace {
struct TargetWatchSpec {
    u32 width = 0, height = 0, remaining = 0;
    char format[64] = {}; ///< optional ":Format" (vk::to_string, e.g. R16G16B16A16Sfloat)
};
TargetWatchSpec& WatchSpec() {
    static TargetWatchSpec spec = [] {
        TargetWatchSpec s;
        const char* env = std::getenv("GOW3_CAPTURE_ON_TARGET");
        if (env && std::sscanf(env, "%ux%u:%63s", &s.width, &s.height, s.format) >= 2) {
            const char* count = std::getenv("GOW3_CAPTURE_ON_TARGET_COUNT");
            s.remaining = count && *count ? u32(std::strtoul(count, nullptr, 10)) : 2;
        }
        return s;
    }();
    return spec;
}
} // namespace

bool FrameCapture::TargetWatch() {
    return WatchSpec().remaining != 0;
}

void FrameCapture::StartForTarget(const VideoCore::ImageInfo* const* colors, u32 num_colors) {
    auto& spec = WatchSpec();
    if (!spec.remaining || state.load(std::memory_order_relaxed) != Idle) {
        return;
    }
    for (u32 i = 0; i < num_colors; ++i) {
        if (colors[i] && colors[i]->size.width == spec.width &&
            colors[i]->size.height == spec.height &&
            (!spec.format[0] || vk::to_string(colors[i]->pixel_format) == spec.format)) {
            --spec.remaining;
            entries.clear();
            pending_sampled.clear();
            pending_buffers.clear();
            pass_open = false;
            state.store(Recording, std::memory_order_release);
            target_recording = true;
            std::printf("Frame capture: started at a %ux%u target (%#llx)\n", spec.width,
                        spec.height, static_cast<unsigned long long>(colors[i]->guest_address));
            return;
        }
    }
}

void FrameCapture::BeginPass(const VideoCore::ImageInfo* const* colors, u32 num_colors,
                             const VideoCore::ImageInfo* depth) {
    // gow3: a recording started by StartForTarget may see no frame boundary while a level loads.
    if (state.load(std::memory_order_relaxed) == Recording && entries.size() >= 400) {
        Write(last_presented.load(std::memory_order_relaxed));
        entries.clear();
        pending_sampled.clear();
        pending_buffers.clear();
        state.store(Idle, std::memory_order_release);
        target_recording = false;
        return;
    }
    std::vector<Target> targets;
    bool display = false;
    {
        std::scoped_lock lk{display_mutex};
        for (u32 i = 0; i < num_colors; ++i) {
            targets.push_back(colors[i] ? ToTarget(*colors[i]) : Target{});
            display |= colors[i] && std::ranges::find(display_buffers, colors[i]->guest_address) !=
                                        display_buffers.end();
        }
    }
    // Frame boundaries in the command stream: the pass that writes a display buffer as its only
    // target. gow3: God of War III also draws scene passes (two targets) into display buffers.
    display = display && targets.size() == 1;
    const bool new_pass = !(pass_open && !entries.empty() && !entries.back().compute &&
                            entries.back().colors == targets);
    // gow3: MediEvil writes a display buffer alone twice a frame (the composite, then the UI): a
    // capture that ended at the second one held only UI passes. It ends at a display pass that
    // follows a scene pass (one with a depth target).
    const bool saw_scene = std::ranges::any_of(
        entries, [](const Entry& e) { return !e.compute && e.depth.address != 0; });
    if (display && new_pass &&
        (state.load(std::memory_order_relaxed) != Recording || saw_scene)) {
        if (state.load(std::memory_order_relaxed) == Recording) {
            Write(last_presented.load(std::memory_order_relaxed));
            entries.clear();
            pending_sampled.clear();
            pending_buffers.clear();
            state.store(Idle, std::memory_order_release);
            target_recording = false;
            return;
        }
        entries.clear();
        pending_sampled.clear();
        pending_buffers.clear();
        pass_open = false;
        state.store(Recording, std::memory_order_release);
        target_recording = false;
    }
    if (state.load(std::memory_order_relaxed) != Recording) {
        pending_sampled.clear();
        pending_buffers.clear();
        return;
    }
    const Target d = depth ? ToTarget(*depth) : Target{};
    if (pass_open && !entries.empty() && !entries.back().compute &&
        entries.back().colors == targets && entries.back().depth == d) {
        return;
    }
    entries.push_back({.compute = false, .colors = std::move(targets), .depth = d});
    pass_open = true;
}

void FrameCapture::Draw(u64 vs_hash, u64 ps_hash, u32 num_indices, u32 num_instances) {
    if (state.load(std::memory_order_relaxed) != Recording || entries.empty() ||
        entries.back().compute) {
        return;
    }
    auto& e = entries.back();
    ++e.draws;
    e.indices += u64(num_indices) * std::max(num_instances, 1u);
    AddShader(e, vs_hash);
    AddShader(e, ps_hash);
    AddSampled(e);
    AddBuffers(e);
}

void FrameCapture::Dispatch(u64 cs_hash, u32 x, u32 y, u32 z) {
    if (state.load(std::memory_order_relaxed) != Recording) {
        pending_sampled.clear();
        pending_buffers.clear();
        return;
    }
    if (!entries.empty() && entries.back().compute && !entries.back().shaders.empty() &&
        entries.back().shaders[0] == cs_hash) {
        ++entries.back().draws;
    } else {
        entries.push_back({.compute = true, .draws = 1, .shaders = {cs_hash}});
        entries.back().note = std::format("groups {}x{}x{}", x, y, z);
    }
    AddSampled(entries.back());
    AddBuffers(entries.back());
    pass_open = false;
}

void FrameCapture::Buffer(u64 stage_hash, u32 slot, VAddr address, const void* data, u64 size) {
    // GOW3_CAPTURE_BUFFER_BYTES: how much of each buffer to print (default 1024; the view's
    // constants are ~3.3 KB).
    static const u64 limit = [] {
        const char* env = std::getenv("GOW3_CAPTURE_BUFFER_BYTES");
        return env && *env ? std::strtoull(env, nullptr, 10) : u64{1024};
    }();
    const u64 bytes = std::min<u64>(size, limit) & ~u64(3);
    std::string out = std::format("  buffer stage {:016x} slot {} at {:#x} size {}:", stage_hash,
                                  slot, address, size);
    const auto* words = static_cast<const float*>(data);
    // Flatbuf (address 0) is the stage's flattened user data: resource descriptors and bitfields,
    // not floats, so print it as the dwords it is. Other buffers are constant data: floats.
    const bool hex = address == 0;
    // IsValidGpuMapping only bounds the address range, so a null pointer still reaches here:
    // record the entry without reading it instead of faulting.
    if (data && bytes) {
        for (u64 i = 0; i < bytes / 4; ++i) {
            if (i % 8 == 0) {
                out += std::format("\n    [{:3}]", i);
            }
            out += hex ? std::format(" {:08x}", static_cast<const u32*>(data)[i])
                       : std::format(" {:12.6g}", words[i]);
        }
    }
    pending_buffers.push_back(std::move(out));
}

void FrameCapture::Sampled(const VideoCore::ImageInfo& info, bool storage) {
    if (pending_sampled.size() < 64) {
        pending_sampled.emplace_back(ToTarget(info), storage);
    }
}

void FrameCapture::Note(const char* text) {
    if (!entries.empty()) {
        entries.back().note += text;
    }
}

} // namespace Vulkan
