// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 (research/docs/impl-texset-index.md): memory for the texture-set memo's compact layout.
#include "gow3_texset_table.h"

#include <cstdio>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <sys/mman.h>
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
#endif

namespace Gow3TexSet {
namespace {
constexpr std::size_t HugeSize = std::size_t{2} << 20;
constexpr std::size_t PageSize = 4096;

void TouchPages(void* data, std::size_t bytes) {
    auto* bytes_ptr = static_cast<volatile unsigned char*>(data);
    for (std::size_t offset = 0; offset < bytes; offset += PageSize) {
        bytes_ptr[offset] = 0;
    }
}
} // namespace

TableMemory::TableMemory(std::size_t size) {
    const std::size_t rounded = (size + HugeSize - 1) / HugeSize * HugeSize;
#ifdef _WIN32
    data = VirtualAlloc(nullptr, rounded, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!data) {
        return;
    }
    bytes = rounded;
    TouchPages(data, bytes);
#else
    // Over-allocate by one huge page and trim to a 2 MiB-aligned range.
    void* raw = mmap(nullptr, rounded + HugeSize, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        return;
    }
    const auto start = reinterpret_cast<std::uintptr_t>(raw);
    const std::uintptr_t aligned = (start + HugeSize - 1) / HugeSize * HugeSize;
    if (aligned > start) {
        munmap(raw, aligned - start);
    }
    const std::uintptr_t tail = start + rounded + HugeSize - (aligned + rounded);
    if (tail) {
        munmap(reinterpret_cast<void*>(aligned + rounded), tail);
    }
    data = reinterpret_cast<void*>(aligned);
    bytes = rounded;
    madvise(data, bytes, MADV_HUGEPAGE);
    TouchPages(data, bytes);
    if (madvise(data, bytes, MADV_COLLAPSE) == 0) {
        huge = 1;
    } else {
        huge = errno == EINVAL ? -1 : 0;
    }
#endif
}

TableMemory::~TableMemory() {
    if (!data) {
        return;
    }
#ifdef _WIN32
    VirtualFree(data, 0, MEM_RELEASE);
#else
    munmap(data, bytes);
#endif
}

const char* HugePageMode() {
#ifdef _WIN32
    return "n/a";
#else
    static const char* mode = [] {
        char text[128] = {};
        if (FILE* file = std::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
            const std::size_t read = std::fread(text, 1, sizeof(text) - 1, file);
            text[read] = 0;
            std::fclose(file);
        }
        // The selected mode is in brackets: "always [madvise] never".
        static constexpr const char* names[] = {"always", "madvise", "never"};
        for (const char* name : names) {
            char selected[16];
            std::snprintf(selected, sizeof(selected), "[%s]", name);
            if (std::strstr(text, selected)) {
                return name;
            }
        }
        return "n/a";
    }();
    return mode;
#endif
}

} // namespace Gow3TexSet
