/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef ASCENDC_MHC_PRE_BACKWARD_C0_COPYOUT_H
#define ASCENDC_MHC_PRE_BACKWARD_C0_COPYOUT_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace MhcPreBackwardUtils {

// 默认Matmul策略在DAV3510上不开放L0C到UB通路。这里针对C0固定的
// FP32/ND输出使用DAV3510 Fixpipe，并按N方向将结果均分给两个AIV。
template <typename IMPL, class A_TYPE, class B_TYPE, class C_TYPE, const auto &MM_CFG>
class MhcC0SplitNCopyCubeOut {
    using DstT = typename C_TYPE::T;
    using SrcT = typename AscendC::GetMmDstType<typename A_TYPE::T>::Type;

    MATMUL_USE_MODULE(MatmulSubBlockInfo);

public:
    __aicore__ inline MhcC0SplitNCopyCubeOut() = default;

    template <bool enSequentialWrite = false, typename ScheduleContext = int>
    __aicore__ inline void Copy(const AscendC::LocalTensor<DstT> &dst,
                                const AscendC::LocalTensor<SrcT> &src,
                                int32_t curRow, int32_t curCol,
                                int32_t baseHeight, int32_t baseWidth,
                                int32_t baseBlockHeight, int32_t baseBlockWidth,
                                const ScheduleContext &context = 0)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        static_assert(AscendC::IsSameType<DstT, float>::value &&
                      AscendC::IsSameType<SrcT, float>::value,
                      "Mhc C0 direct CopyOut only supports FP32 -> FP32");
        static_assert(C_TYPE::format == CubeFormat::ND,
                      "Mhc C0 direct CopyOut only supports ND output");

        // 顺序写模式下，每个128行结果都从当前ping/pong槽首地址开始写。
        int64_t dstOffset = 0;
        if constexpr (!enSequentialWrite) {
            const int32_t halfStride = baseWidth >> 1;
            dstOffset = static_cast<int64_t>(curRow) * baseHeight * halfStride +
                        (static_cast<int64_t>(curCol) * baseWidth * baseHeight >> 1);
        }

        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = static_cast<uint16_t>(baseWidth);
        params.mSize = static_cast<uint16_t>(baseHeight);
        params.srcStride = AscendC::CeilAlign(baseHeight, AscendC::BLOCK_CUBE);
        params.dstStride = static_cast<uint16_t>(baseWidth >> 1);
        params.dualDstCtl = static_cast<uint8_t>(AscendC::McgShfMode::DUAL_DST_SPLIT_N);
        // 保留Matmul单元旗标，确保最后一次Fixpipe完成后才能发布跨核就绪信号。
        params.unitFlag = 3U;
        params.subBlockId = MATMUL_MODULE(MatmulSubBlockInfo)->GetSubBlockIdx();
        AscendC::Fixpipe<DstT, SrcT, AscendC::Impl::CFG_ROW_MAJOR_UB>(dst[dstOffset], src, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Mhc C0 direct CopyOut only supports DAV3510"); });
#endif
    }

    template <bool enSequentialWrite = false, typename ScheduleContext = int>
    __aicore__ inline void Copy(const AscendC::GlobalTensor<DstT> &dst,
                                const AscendC::LocalTensor<SrcT> &src,
                                int32_t curRow, int32_t curCol,
                                int32_t baseHeight, int32_t baseWidth,
                                int32_t baseBlockHeight, int32_t baseBlockWidth,
                                const ScheduleContext &context = 0)
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        // 混合通路中的GM tile继续使用标准ROW_MAJOR Fixpipe。
        // 当前调用始终为顺序写，目标GlobalTensor已经包含本tile的GM偏移。
        ASCENDC_ASSERT(enSequentialWrite,
                       { KERNEL_LOG(KERNEL_ERROR, "Mhc C0 GM CopyOut requires sequential write"); });
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> params;
        params.nSize = static_cast<uint16_t>(baseWidth);
        params.mSize = static_cast<uint16_t>(baseHeight);
        params.srcStride = AscendC::CeilAlign(baseHeight, AscendC::BLOCK_CUBE);
        params.dstStride = static_cast<uint16_t>(baseWidth);
        params.unitFlag = 3U;
        AscendC::Fixpipe<DstT, SrcT, AscendC::CFG_ROW_MAJOR>(dst, src, params);
#else
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Mhc C0 GM CopyOut only supports DAV3510"); });
#endif
    }
};

template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE,
          typename C_TYPE, typename BIAS_TYPE>
struct MhcC0SplitNPolicy
    : public AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using CopyCubeOut = MhcC0SplitNCopyCubeOut<IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG>;
};

} // namespace MhcPreBackwardUtils

#endif // ASCENDC_MHC_PRE_BACKWARD_C0_COPYOUT_H
