#ifndef CATLASS_GEMV_DEVICE_KERNEL_ADAPTER_SELF_HPP
#define CATLASS_GEMV_DEVICE_KERNEL_ADAPTER_SELF_HPP

#include "catlass/debug.hpp"

namespace FTSelf {
/// GEMV kernel adapter with optional dump workspace.
template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, GM_ADDR ptrDump = nullptr)
{
    Operator op;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);
#endif
    op(params);
}

template <class Operator>
CATLASS_GLOBAL void KernelAdapter(typename Operator::Params params, uint64_t fftsAddr, GM_ADDR ptrDump = nullptr)
{
    AscendC::SetSyncBaseAddr(fftsAddr);
    Operator op;
#if defined(ENABLE_ASCENDC_DUMP)
    AscendC::InitDump(false, ptrDump, ALL_DUMPSIZE);
#endif
    op(params);
}
} // namespace FTSelf

#endif
