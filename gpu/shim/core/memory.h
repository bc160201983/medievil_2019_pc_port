// gow3: the MemoryManager surface used by the video core and GnmDriver.
// Guest memory lives at identical host addresses; the C runtime owns the VMAs.
#pragma once
#include "common/types.h"
#include "common/singleton.h"
#include "core/address_space.h"

namespace Vulkan { class Rasterizer; }

namespace Core {
class MemoryManager {
public:
    void SetRasterizer(Vulkan::Rasterizer* rasterizer_);
    AddressSpace& GetAddressSpace() { return impl; }
    bool IsValidGpuMapping(VAddr virtual_addr, u64 size) {
        // The PS4's GPU can only handle 40 bit addresses.
        return virtual_addr + size < 0x100'0000'0000ULL;
    }
    u64 ClampRangeSize(VAddr virtual_addr, u64 size);
    void CopySparseMemory(VAddr source, u8* dest, u64 size);
    bool TryWriteBacking(void* address, const void* data, u64 size);
    /// gow3 GOW3_GPU_FENCES: a fence label as one store (gow3gpu.cpp), through the backing view
    /// when `backing` (past the GPU side's page protection, as TryWriteBacking), else at its
    /// address. False when the label's memory is no longer mapped: it is dropped.
    bool StoreLabel(void* address, u64 value, u32 size, bool backing);
    /// gow3: reads guest memory past the GPU side's page protection (the backing view).
    bool ReadBacking(VAddr address, void* data, u64 size);
    /// CPU wrote guest memory outside page tracking (e.g. decoded video frames).
    void InvalidateMemory(VAddr address, u64 size);
    Vulkan::Rasterizer* GetRasterizer() const { return rasterizer; }

private:
    AddressSpace impl;
    Vulkan::Rasterizer* rasterizer{};
};
using Memory = Common::Singleton<MemoryManager>;
} // namespace Core
