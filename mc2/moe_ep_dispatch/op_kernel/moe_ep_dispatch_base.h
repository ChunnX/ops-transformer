/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_ep_dispatch_base.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_BASE_H
#define MOE_EP_DISPATCH_BASE_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"

namespace MoeEpDispatchBase {

#if defined(ENABLE_MOE_EP_KERNEL)

using namespace AscendC;

// The data and mask must be packed together; -1 must never alias valid bin 255.
template <Reg::HighLowPart part>
__simd_callee__ inline void PackHistogramPage(Reg::RegTensor<uint8_t> &packed, Reg::MaskReg &packedMask,
                                             __ubuf__ int16_t *src, int16_t base, uint16_t bins,
                                             Reg::MaskReg &tail)
{
    Reg::RegTensor<int16_t> ids;
    Reg::MaskReg valid;
    Reg::LoadAlign(ids, src);
    Reg::Adds(ids, ids, static_cast<int16_t>(-base), tail);
    // Unsigned comparison rejects both negative IDs and IDs outside this 256-bin page.
    Reg::Compares<uint16_t, CMPMODE::LT>(valid, (Reg::RegTensor<uint16_t> &)ids, bins, tail);
    Reg::Pack<uint8_t, int16_t, part>(packed, ids);
    Reg::Pack<part>(packedMask, valid);
}

// Keep filtering and narrowing in registers; one histogram input covers two int16 registers.
template <bool highBins>
__simd_vf__ __aicore__ inline void RouteHistogramsVf(
    __ubuf__ uint16_t *dst, __ubuf__ int16_t *src, uint32_t count, int16_t base, uint16_t bins)
{
    constexpr uint32_t lanes = GetVecLen() / sizeof(int16_t);
    uint16_t repeatTimes = (count + 2U * lanes - 1U) / (2U * lanes);
    Reg::RegTensor<uint8_t> packedLow, packedHigh, packed;
    Reg::RegTensor<uint16_t> low, high;
    Reg::MaskReg outputMask = Reg::CreateMask<uint16_t>();
    Reg::MaskReg mergeMask = Reg::CreateMask<uint8_t>();
    Reg::MaskReg maskLow, maskHigh, packedMask;
    Reg::Duplicate(low, 0);
    if constexpr (highBins) {
        Reg::Duplicate(high, 0);
    }
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        Reg::MaskReg tail = Reg::UpdateMask<int16_t>(count);
        PackHistogramPage<Reg::HighLowPart::LOWEST>(packedLow, maskLow, src + 2U * lanes * i, base, bins, tail);
        tail = Reg::UpdateMask<int16_t>(count);
        PackHistogramPage<Reg::HighLowPart::HIGHEST>(packedHigh, maskHigh, src + (2U * i + 1U) * lanes,
                                                  base, bins, tail);
        // Both Pack overloads zero the other half: preserve each result and explicitly merge them.
        Reg::Or<uint8_t>(packed, packedLow, packedHigh, mergeMask);
        Reg::Or(packedMask, maskLow, maskHigh, mergeMask);
        Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0,
                        Reg::HistogramsType::FREQUENCY>(low, packed, packedMask);
        if constexpr (highBins) {
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1,
                            Reg::HistogramsType::FREQUENCY>(high, packed, packedMask);
        }
    }
    Reg::StoreAlign(dst, low, outputMask);
    if constexpr (highBins) {
        Reg::StoreAlign(dst + lanes, high, outputMask);
    }
}

__aicore__ inline void GetRouteHistogram(LocalTensor<uint16_t> dstLocal, LocalTensor<int16_t> srcLocal,
                                        uint32_t count, uint32_t base, uint32_t bins)
{
    __ubuf__ int16_t *src = (__ubuf__ int16_t *)srcLocal.GetPhyAddr();
    __ubuf__ uint16_t *dst = (__ubuf__ uint16_t *)dstLocal.GetPhyAddr();
    if (bins <= GetVecLen() / sizeof(uint16_t)) {
        asc_vf_call<RouteHistogramsVf<false>>(dst, src, count, static_cast<int16_t>(base), static_cast<uint16_t>(bins));
    } else {
        asc_vf_call<RouteHistogramsVf<true>>(dst, src, count, static_cast<int16_t>(base), static_cast<uint16_t>(bins));
    }
    PipeBarrier<PIPE_V>();
}

template <typename ContextType>
__aicore__ inline GM_ADDR GetWindowAddrByRankId(__gm__ ContextType *context, uint32_t rankId,
                                                uint64_t offset)
{
    return (GM_ADDR)context->epHcclBuffer[rankId] + offset;
}

#endif

} // namespace MoeEpDispatchBase

namespace Mc2Kernel {

#if defined(ENABLE_MOE_EP_KERNEL)

using MoeEpDispatchBase::GetRouteHistogram;

template <typename ContextType>
__aicore__ inline GM_ADDR GetWinAddrByRankId(__gm__ ContextType *context, uint32_t rankId, uint64_t offset)
{
    return MoeEpDispatchBase::GetWindowAddrByRankId(context, rankId, offset);
}

#endif

} // namespace Mc2Kernel

#endif // MOE_EP_DISPATCH_BASE_H
