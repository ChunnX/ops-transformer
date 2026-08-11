#ifndef FTSELF_CATLASS_GEMV_DEVICE_DEVICE_GEMV_HPP
#define FTSELF_CATLASS_GEMV_DEVICE_DEVICE_GEMV_HPP

#include_next "catlass/gemv/device/device_gemv.hpp"


#include "catlass/gemv/device/kernel_adapter.hpp"

#if defined(ENABLE_ASCENDC_DUMP)
#include "catlass/debug.hpp"
#endif

namespace FTSelf::Gemv::Device {

template <class GemvKernel>
class DeviceGemv {
public:
    using Kernel = GemvKernel;
    /// Argument structure: User API
    using Arguments = typename GemvKernel::Arguments;
    /// Argument structure: Kernel API
    using Params = typename GemvKernel::Params;
private:
    /// kernel API parameters object
    Params params_;
public:
    DeviceGemv() {}
    ~DeviceGemv() {}

    /// Access the Params structure
    Params const &params() const
    {
        return params_;
    }

    /// Determines whether the GEMV can execute the given problem.
    static Catlass::Status CanImplement(Arguments const &args)
    {
        if (GemvKernel::CanImplement(args)) {
            return Catlass::Status::kSuccess;
        } else {
            return Catlass::Status::kInvalid;
        }
    }

    /// Gets the workspace size
    static size_t GetWorkspaceSize(Arguments const &args)
    {
        size_t workspace_bytes = 0;
        workspace_bytes += GemvKernel::GetWorkspaceSize(args);
        return workspace_bytes;
    }

    /// Initializes GEMV state from arguments
    Catlass::Status Initialize(Arguments const &args, uint8_t *workspace = nullptr, aclrtStream stream = nullptr)
    {
        params_ = GemvKernel::ToUnderlyingArguments(args, workspace);
        return Catlass::Status::kSuccess;
    }

    inline Catlass::Status Run(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr)
    {
#if defined(ENABLE_ASCENDC_DUMP)
        uint8_t *ptrDump{nullptr};
        aclCheck(aclrtMalloc(reinterpret_cast<void **>(&ptrDump), ALL_DUMPSIZE, ACL_MEM_MALLOC_HUGE_FIRST));
        if (fftsAddr == 0) {
            FTSelf::KernelAdapter<GemvKernel><<<blockDim, nullptr, stream>>>(params_, ptrDump);
        } else {
            FTSelf::KernelAdapter<GemvKernel><<<blockDim, nullptr, stream>>>(params_, fftsAddr, ptrDump);
        }
        aclCheck(aclrtSynchronizeStream(stream));
        Adx::AdumpPrintWorkSpace(ptrDump, ALL_DUMPSIZE, stream, "device_gemv");
        aclCheck(aclrtFree(ptrDump));
#else
        if (fftsAddr == 0) {
            FTSelf::KernelAdapter<GemvKernel><<<blockDim, nullptr, stream>>>(params_);
        } else {
            FTSelf::KernelAdapter<GemvKernel><<<blockDim, nullptr, stream>>>(params_, fftsAddr);
        }
#endif
        return Catlass::Status::kSuccess;
    }

    inline Catlass::Status operator()(aclrtStream stream, uint32_t blockDim)
    {
        return Run(stream, blockDim, 0);
    }

    inline Catlass::Status operator()(aclrtStream stream, uint32_t blockDim, uint64_t fftsAddr)
    {
        return Run(stream, blockDim, fftsAddr);
    }
};

} // namespace FTSelf::Gemv::Device

#endif // FTSELF_CATLASS_GEMV_DEVICE_DEVICE_GEMV_HPP
