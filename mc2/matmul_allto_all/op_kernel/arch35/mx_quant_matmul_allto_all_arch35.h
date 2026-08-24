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
 * \file mx_quant_matmul_allto_all_arch35.h
 * \brief
 */

#ifndef MX_QUANT_MATMUL_ALLTO_ALL_ARCH35_H
#define MX_QUANT_MATMUL_ALLTO_ALL_ARCH35_H

namespace Mc2Kernel {
// ============================================================================
// 类型特征：定义打包因子（每个元素包含的标量数）
// eg: fp4（2个打包）和 fp8（1个独立）
// ============================================================================
template <typename T>
struct PackFactor {
    static constexpr uint64_t value = 1UL; // 默认：每个元素包含1个标量（如fp8）
};

template <>
struct PackFactor<fp4x2_e2m1_t> {
    static constexpr uint64_t value = 2UL; // fp4x2：2个fp4打包成1个元素
};

// ============================================================================
// 辅助函数：计算字节偏移
// 自动处理 fp4 和 fp8 的差异
// ============================================================================
template <typename T>
__aicore__ inline uint64_t ComputeByteOffset(uint64_t elementCount)
{
    return elementCount * sizeof(T) / PackFactor<T>::value;
}

// ============================================================================
// 辅助函数：计算GM地址
// 自动处理 fp4 和 fp8 的差异
// ============================================================================
template <typename T>
__aicore__ inline GM_ADDR ComputeGMAddr(GM_ADDR baseAddr, uint64_t elementCount)
{
    return baseAddr + ComputeByteOffset<T>(elementCount);
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType,
          bool UsingApaceImpl = false>
class MxQuantMatmulAlltoAllArch35 {
public:
    __aicore__ inline MxQuantMatmulAlltoAllArch35(SchedulerType *pipeLine) : pipeLine_(pipeLine){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR x1Scale, GM_ADDR x2Scale,
                                GM_ADDR workspaceGM, MatmulAlltoAllTilingDataType *tilingData, AscendC::TPipe *tPipe);
    __aicore__ inline void Process();

private:
    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t NUM_TWO = 2UL;

    SchedulerType *pipeLine_;
    SchedulerContextType pipeLineContext_;
    MatmulAlltoAllTilingDataType *tilingData_;
    AscendC::TPipe *tPipe_;
    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR x1Scale_;
    GM_ADDR x2Scale_;
    GM_ADDR workspaceGM_;
    GM_ADDR tempComputeOutGM_;
    GM_ADDR transOutGM_;

private:
    __aicore__ inline void ProcessTile(uint32_t taskCnt);
    __aicore__ inline void ProcessTail(uint32_t taskCnt);
    __aicore__ inline decltype(auto) GetTilingInfo()
    {
        if constexpr (UsingApaceImpl) {
            return (tilingData_->matmulAllToAllTilingInfo);
        } else {
            return (tilingData_->quantMatmulAlltoAllTilingInfo);
        }
    }

    __aicore__ inline auto GetTileTilingDataPtr()
    {
        if constexpr (UsingApaceImpl) {
            return &(tilingData_->tileQbmmTilingData);
        } else {
            return &(tilingData_->mc2QuantBmmV3TileTilingData);
        }
    }

    __aicore__ inline auto GetTailTilingDataPtr()
    {
        if constexpr (UsingApaceImpl) {
            return &(tilingData_->tailQbmmTilingData);
        } else {
            return &(tilingData_->mc2QuantBmmV3TailTilingData);
        }
    }
};


template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType,
          bool UsingApaceImpl>
__aicore__ inline void
MxQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType, UsingApaceImpl>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR workspaceGM,
    MatmulAlltoAllTilingDataType *tilingData, AscendC::TPipe *tPipe)
{
    // 获取tilingdata数据
    tilingData_ = tilingData;
    auto &&mc2Tiling_ = GetTilingInfo();
    // 管道初始化
    tPipe_ = tPipe;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    x1Scale_ = x1Scale;
    x2Scale_ = x2Scale;
    bias_ = bias;
    workspaceGM_ = workspaceGM;
    tempComputeOutGM_ = workspaceGM;
    if constexpr (UsingApaceImpl) {
        transOutGM_ = nullptr;
    } else {
        transOutGM_ = (GM_ADDR)(tempComputeOutGM_ + mc2Tiling_.mmResultLen);
    }

    // 初始化流水线
    pipeLine_->Init();
    pipeLine_->GetContext(&pipeLineContext_);
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType,
          bool UsingApaceImpl>
__aicore__ inline void
MxQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType, UsingApaceImpl>::Process()
{
    auto &&mc2Tiling_ = GetTilingInfo();
    // 启动主块流水
    if (mc2Tiling_.tileCnt > 0) {
        ProcessTile(mc2Tiling_.tileCnt);
    }

    // 启动尾块流水
    if (mc2Tiling_.tailCnt > 0) {
        ProcessTail(mc2Tiling_.tailCnt);
    }

    // 结束流水线
    pipeLine_->End();
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType,
          bool UsingApaceImpl>
__aicore__ inline void
MxQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType, UsingApaceImpl>::ProcessTile(
    uint32_t taskCnt)
{
    auto &&mc2Tiling_ = GetTilingInfo();
    const uint64_t rankNPerPeer = (uint64_t)mc2Tiling_.rankN / mc2Tiling_.rankDim;

    pipeLineContext_.computationContext->baseData.aGM = x1_;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM = tempComputeOutGM_;
    pipeLineContext_.computationContext->baseData.biasGM = bias_;
    pipeLineContext_.computationContext->baseData.aOffset =
        ComputeByteOffset<DTYPE_X1>((uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankK);
    pipeLineContext_.computationContext->baseData.bOffset = (uint64_t)0UL;
    if constexpr (UsingApaceImpl) {
        // Blaze writes [rank, M, N/rank] directly. A task advances inside one
        // rank block; rankStride selects the next Hcomm destination block.
        pipeLineContext_.computationContext->baseData.cOffset =
            ComputeByteOffset<DTYPE_Y>((uint64_t)mc2Tiling_.tileM * rankNPerPeer);
        pipeLineContext_.computationContext->additionalData.fullN = mc2Tiling_.rankN;
        pipeLineContext_.computationContext->additionalData.rankStride =
            (uint64_t)mc2Tiling_.rankM * rankNPerPeer;
        pipeLineContext_.computationContext->additionalData.rankDim = mc2Tiling_.rankDim;
    } else {
        pipeLineContext_.computationContext->baseData.cOffset =
            ComputeByteOffset<DTYPE_Y>((uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankN);
    }
    pipeLineContext_.computationContext->additionalData.x1ScaleOffset = ComputeByteOffset<AscendC::fp8_e8m0_t>(
        (uint64_t)mc2Tiling_.tileM * CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO);
    pipeLineContext_.computationContext->additionalData.x1Scale = x1Scale_;
    pipeLineContext_.computationContext->additionalData.x2Scale = x2Scale_;
    pipeLineContext_.computationContext->tilingDataPtr = GetTileTilingDataPtr();

    if constexpr (!UsingApaceImpl) {
        pipeLineContext_.transposeContext->transposeSrcAddr = tempComputeOutGM_;
        pipeLineContext_.transposeContext->transposeDstAddr = transOutGM_;
        pipeLineContext_.transposeContext->transposeSrcOffset =
            ComputeByteOffset<DTYPE_Y>((uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankN);
        pipeLineContext_.transposeContext->nextSrcBlockOffset = rankNPerPeer;
        pipeLineContext_.transposeContext->nextDstBlockOffset =
            (uint64_t)mc2Tiling_.rankM * rankNPerPeer;
        pipeLineContext_.transposeContext->transposeDstOffset =
            pipeLineContext_.transposeContext->transposeSrcOffset / mc2Tiling_.rankDim;
        pipeLineContext_.transposeContext->rankCnt = mc2Tiling_.rankDim;
        pipeLineContext_.transposeContext->innerAxis = rankNPerPeer;
        pipeLineContext_.transposeContext->transM = (uint64_t)mc2Tiling_.tileM;
        pipeLineContext_.transposeContext->innerOffsetIn = (uint64_t)mc2Tiling_.rankN;
        pipeLineContext_.transposeContext->innerOffsetOut = rankNPerPeer;
    }

    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tileCnt;
    pipeLineContext_.communicationContext->recvBuffer = y_;
    pipeLineContext_.communicationContext->sendOffset =
        (uint64_t)mc2Tiling_.tileM * rankNPerPeer * sizeof(DTYPE_Y);
    if constexpr (UsingApaceImpl) {
        pipeLineContext_.communicationContext->sendBuffer = tempComputeOutGM_;
        pipeLineContext_.communicationContext->sendCount = (uint64_t)mc2Tiling_.tileM * rankNPerPeer;
        pipeLineContext_.communicationContext->strideCount = (uint64_t)mc2Tiling_.rankM * rankNPerPeer;
    } else {
        pipeLineContext_.communicationContext->sendBuffer = pipeLineContext_.transposeContext->transposeDstAddr;
        pipeLineContext_.communicationContext->sendCount =
            (uint64_t)mc2Tiling_.tileM * pipeLineContext_.transposeContext->innerAxis;
        pipeLineContext_.communicationContext->strideCount = pipeLineContext_.transposeContext->nextDstBlockOffset;
    }
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->hcclDataType = mc2Tiling_.hcclDataType;

    pipeLine_->Process(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType,
          bool UsingApaceImpl>
__aicore__ inline void
MxQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType, UsingApaceImpl>::ProcessTail(
    uint32_t taskCnt)
{
    auto &&mc2Tiling_ = GetTilingInfo();
    const uint64_t rankNPerPeer = (uint64_t)mc2Tiling_.rankN / mc2Tiling_.rankDim;
    const uint64_t mainRows = (uint64_t)mc2Tiling_.tileM * mc2Tiling_.tileCnt;

    pipeLineContext_.computationContext->baseData.aGM =
        ComputeGMAddr<DTYPE_X1>(x1_, mainRows * mc2Tiling_.rankK);
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.aOffset =
        ComputeByteOffset<DTYPE_X1>((uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankK);
    pipeLineContext_.computationContext->baseData.bOffset = (uint64_t)0UL;
    if constexpr (UsingApaceImpl) {
        pipeLineContext_.computationContext->baseData.cGM =
            ComputeGMAddr<DTYPE_Y>(tempComputeOutGM_, mainRows * rankNPerPeer);
        pipeLineContext_.computationContext->baseData.cOffset =
            ComputeByteOffset<DTYPE_Y>((uint64_t)mc2Tiling_.tailM * rankNPerPeer);
        pipeLineContext_.computationContext->additionalData.fullN = mc2Tiling_.rankN;
        pipeLineContext_.computationContext->additionalData.rankStride =
            (uint64_t)mc2Tiling_.rankM * rankNPerPeer;
        pipeLineContext_.computationContext->additionalData.rankDim = mc2Tiling_.rankDim;
    } else {
        pipeLineContext_.computationContext->baseData.cGM =
            ComputeGMAddr<DTYPE_Y>(tempComputeOutGM_, mainRows * mc2Tiling_.rankN);
        pipeLineContext_.computationContext->baseData.cOffset =
            ComputeByteOffset<DTYPE_Y>((uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankN);
    }
    pipeLineContext_.computationContext->additionalData.x1ScaleOffset =
        (uint64_t)mc2Tiling_.tailM * CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO;
    pipeLineContext_.computationContext->additionalData.x1Scale = ComputeGMAddr<AscendC::fp8_e8m0_t>(
        x1Scale_, mainRows * CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO);
    pipeLineContext_.computationContext->additionalData.x2Scale = x2Scale_;
    pipeLineContext_.computationContext->tilingDataPtr = GetTailTilingDataPtr();

    if constexpr (!UsingApaceImpl) {
        pipeLineContext_.transposeContext->transposeSrcAddr =
            ComputeGMAddr<DTYPE_Y>(tempComputeOutGM_, mainRows * mc2Tiling_.rankN);
        pipeLineContext_.transposeContext->transposeDstAddr =
            transOutGM_ + mainRows * rankNPerPeer * sizeof(DTYPE_Y);
        pipeLineContext_.transposeContext->transposeSrcOffset =
            ComputeByteOffset<DTYPE_Y>((uint64_t)mc2Tiling_.tailM * mc2Tiling_.rankN);
        pipeLineContext_.transposeContext->transposeDstOffset =
            pipeLineContext_.transposeContext->transposeSrcOffset / (uint64_t)mc2Tiling_.rankDim;
        pipeLineContext_.transposeContext->transM = (uint64_t)mc2Tiling_.tailM;
    }

    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tailCnt;
    pipeLineContext_.communicationContext->recvBuffer = y_ + mainRows * rankNPerPeer * sizeof(DTYPE_Y);
    pipeLineContext_.communicationContext->sendOffset =
        (uint64_t)mc2Tiling_.tailM * rankNPerPeer * sizeof(DTYPE_Y);
    if constexpr (UsingApaceImpl) {
        pipeLineContext_.communicationContext->sendBuffer = pipeLineContext_.computationContext->baseData.cGM;
        pipeLineContext_.communicationContext->sendCount = (uint64_t)mc2Tiling_.tailM * rankNPerPeer;
        pipeLineContext_.communicationContext->strideCount = (uint64_t)mc2Tiling_.rankM * rankNPerPeer;
    } else {
        pipeLineContext_.communicationContext->sendBuffer = pipeLineContext_.transposeContext->transposeDstAddr;
        pipeLineContext_.communicationContext->sendCount =
            (uint64_t)mc2Tiling_.tailM * pipeLineContext_.transposeContext->innerAxis;
    }
    pipeLineContext_.communicationContext->recvOffset = pipeLineContext_.communicationContext->sendOffset;
    pipeLineContext_.communicationContext->hcclDataType = mc2Tiling_.hcclDataType;

    pipeLine_->Process(taskCnt);
}

} // namespace Mc2Kernel
#endif
