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
 * \brief matmul_allto_all MX量化主线pipeline路径（USING_APACE_IMPL=0）
 *
 * 数据流: MatMul → Transpose → AllToAll（与 allto_all_matmul 相反）
 *   1. AIC: matmul(x1, x2, scale) → mmResult（workspace）
 *   2. AIV: transpose(mmResult) → transposedResult
 *   3. AIV: AllToAll(transposedResult) → y
 *
 * Scale 在 matmul 阶段本地消费，无需跨卡通信。
 * 通信数据类型为 YType（matmul 输出），非 X1Type。
 */

#ifndef MX_QUANT_MATMUL_ALLTO_ALL_ARCH35_H
#define MX_QUANT_MATMUL_ALLTO_ALL_ARCH35_H

namespace Mc2Kernel {
template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType,
          bool IsMxFp4>
class MxQuantMatmulAlltoAllArch35 {
public:
    __aicore__ inline MxQuantMatmulAlltoAllArch35(SchedulerType *pipeLine, uint64_t yHcclDataType)
        : pipeLine_(pipeLine), yHcclDataType_(yHcclDataType){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
                                GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR workspaceGM,
                                MatmulAlltoAllTilingDataType *tilingData, AscendC::TPipe *tPipe);
    __aicore__ inline void Process();

private:
    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t NUM_TWO = 2UL;
    static constexpr uint64_t ALIGN_NUM = 512UL;

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
    // matmul_allto_all workspace 布局: mmResult → transOut → commOut
    GM_ADDR mmResultGM_;      // matmul 输出
    GM_ADDR transOutGM_;      // 转置后的 matmul 结果
    GM_ADDR commOutGM_;       // AllToAll 通信缓冲区
    uint64_t yHcclDataType_;  // 通信数据类型（YType 对应的 HCCL 类型）

private:
    __aicore__ inline void ProcessTile(uint32_t taskCnt);
    __aicore__ inline void ProcessTail(uint32_t taskCnt);
};


template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType,
          bool IsMxFp4>
__aicore__ inline void
MxQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType, IsMxFp4>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR x1Scale, GM_ADDR x2Scale,
    GM_ADDR workspaceGM, MatmulAlltoAllTilingDataType *tilingData, AscendC::TPipe *tPipe)
{
    tilingData_ = tilingData;
    auto &&mc2Tiling_ = tilingData_->quantMatmulAlltoAllTilingInfo;
    tPipe_ = tPipe;
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    x1Scale_ = x1Scale;
    x2Scale_ = x2Scale;
    workspaceGM_ = workspaceGM;

    // workspace 布局: mmResult → transOut → commOut
    // mmResult: matmul 输出，大小 = rankM * rankN * sizeof(YType)
    mmResultGM_ = workspaceGM;
    // transOut: 转置后的结果，大小 = rankM * rankN * sizeof(YType) (转置不改变大小)
    transOutGM_ = all2all_out;
    if (all2all_out == nullptr) {
        transOutGM_ = mmResultGM_ + mc2Tiling_.mmResultLen;
    }
    // commOut: AllToAll 接收缓冲区（最终输出 y）
    commOutGM_ = y;

    // 初始化流水线
    pipeLine_->Init();
    pipeLine_->GetContext(&pipeLineContext_);
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType,
          bool IsMxFp4>
__aicore__ inline void
MxQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType, IsMxFp4>::Process()
{
    if (pipeLine_ == nullptr || tilingData_ == nullptr) {
        return;
    }

    auto &&mc2Tiling_ = tilingData_->quantMatmulAlltoAllTilingInfo;

    if (mc2Tiling_.rankDim == 0) {
        return;
    }

    // matmul_allto_all: 不需要 ProcessScale（scale 本地消费）

    // 启动主块流水: MatMul → Transpose → AllToAll
    if (mc2Tiling_.tileCnt > 0) {
        ProcessTile(mc2Tiling_.tileCnt);
    }

    // 启动尾块流水
    if (mc2Tiling_.tailCnt > 0) {
        ProcessTail(mc2Tiling_.tailCnt);
    }

    pipeLine_->End();
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType,
          bool IsMxFp4>
__aicore__ inline void
MxQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType, IsMxFp4>::ProcessTile(
    uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->quantMatmulAlltoAllTilingInfo;
    uint64_t tileMMultiN = static_cast<uint64_t>(mc2Tiling_.tileM) * mc2Tiling_.rankN;

    // matmul 计算相关地址: x1 × x2 → mmResult
    pipeLineContext_.computationContext->baseData.aGM = x1_;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM = mmResultGM_;
    pipeLineContext_.computationContext->baseData.biasGM = bias_;
    pipeLineContext_.computationContext->baseData.aOffset =
        tileMMultiN * mc2Tiling_.rankK * sizeof(DTYPE_X1) / mc2Tiling_.rankK; // 按 tileM 步进
    pipeLineContext_.computationContext->baseData.bOffset = 0UL;
    pipeLineContext_.computationContext->baseData.cOffset =
        static_cast<uint64_t>(mc2Tiling_.tileM) * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->additionalData.x1Scale = x1Scale_;
    pipeLineContext_.computationContext->additionalData.x1ScaleOffset = 0UL;
    pipeLineContext_.computationContext->additionalData.x2Scale = x2Scale_;
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2QuantBmmV3TileTilingData);

    // 转置相关地址: mmResult → transOut
    pipeLineContext_.transposeContext->transposeSrcAddr = mmResultGM_;
    pipeLineContext_.transposeContext->transposeDstAddr = transOutGM_;
    pipeLineContext_.transposeContext->transposeSrcOffset =
        static_cast<uint64_t>(mc2Tiling_.tileM) * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->transposeDstOffset =
        static_cast<uint64_t>(mc2Tiling_.tileM) * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->nextSrcBlockOffset =
        static_cast<uint64_t>(mc2Tiling_.rankM) * mc2Tiling_.rankN / mc2Tiling_.rankDim;
    pipeLineContext_.transposeContext->nextDstBlockOffset = mc2Tiling_.rankN;
    pipeLineContext_.transposeContext->rankCnt = static_cast<uint64_t>(mc2Tiling_.rankDim);
    pipeLineContext_.transposeContext->innerAxis = mc2Tiling_.rankN;
    pipeLineContext_.transposeContext->transM = static_cast<uint64_t>(mc2Tiling_.tileM);
    pipeLineContext_.transposeContext->innerOffsetIn = mc2Tiling_.rankN;
    pipeLineContext_.transposeContext->innerOffsetOut = mc2Tiling_.rankN * mc2Tiling_.rankDim;

    // 通信相关地址: AllToAll(transOut → y)
    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tileCnt;
    pipeLineContext_.communicationContext->sendBuffer = transOutGM_;
    pipeLineContext_.communicationContext->recvBuffer = y_;
    pipeLineContext_.communicationContext->sendOffset = tileMMultiN * sizeof(DTYPE_Y);
    pipeLineContext_.communicationContext->recvOffset = tileMMultiN * sizeof(DTYPE_Y);
    pipeLineContext_.communicationContext->sendCount = tileMMultiN;
    pipeLineContext_.communicationContext->strideCount =
        static_cast<uint64_t>(mc2Tiling_.rankM) * mc2Tiling_.rankN / mc2Tiling_.rankDim;
    pipeLineContext_.communicationContext->hcclDataType = yHcclDataType_;

    pipeLine_->Process(taskCnt);
}

template <typename SchedulerType, typename SchedulerContextType, typename MatmulAlltoAllTilingDataType,
          bool IsMxFp4>
__aicore__ inline void
MxQuantMatmulAlltoAllArch35<SchedulerType, SchedulerContextType, MatmulAlltoAllTilingDataType, IsMxFp4>::ProcessTail(
    uint32_t taskCnt)
{
    auto &&mc2Tiling_ = tilingData_->quantMatmulAlltoAllTilingInfo;
    uint64_t tailMMultiN = static_cast<uint64_t>(mc2Tiling_.tailM) * mc2Tiling_.rankN;
    uint64_t tileCntMultiTileMMultiN =
        static_cast<uint64_t>(mc2Tiling_.tileCnt) * static_cast<uint64_t>(mc2Tiling_.tileM) * mc2Tiling_.rankN;

    // matmul 计算相关地址
    pipeLineContext_.computationContext->baseData.aGM = x1_;
    pipeLineContext_.computationContext->baseData.bGM = x2_;
    pipeLineContext_.computationContext->baseData.cGM = mmResultGM_;
    pipeLineContext_.computationContext->baseData.biasGM = bias_;
    pipeLineContext_.computationContext->baseData.cOffset =
        static_cast<uint64_t>(mc2Tiling_.tailM) * mc2Tiling_.rankN * sizeof(DTYPE_Y);
    pipeLineContext_.computationContext->additionalData.x1Scale = x1Scale_;
    pipeLineContext_.computationContext->additionalData.x2Scale = x2Scale_;
    pipeLineContext_.computationContext->tilingDataPtr = &(tilingData_->mc2QuantBmmV3TailTilingData);

    // 转置相关地址
    pipeLineContext_.transposeContext->transposeSrcAddr =
        mmResultGM_ + tileCntMultiTileMMultiN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->transposeDstAddr =
        transOutGM_ + tileCntMultiTileMMultiN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->transposeSrcOffset = tailMMultiN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->transposeDstOffset = tailMMultiN * sizeof(DTYPE_Y);
    pipeLineContext_.transposeContext->transM = static_cast<uint64_t>(mc2Tiling_.tailM);

    // 通信相关地址
    pipeLineContext_.communicationContext->taskCnt = mc2Tiling_.tailCnt;
    pipeLineContext_.communicationContext->sendBuffer =
        transOutGM_ + tileCntMultiTileMMultiN * sizeof(DTYPE_Y);
    pipeLineContext_.communicationContext->recvBuffer =
        y_ + tileCntMultiTileMMultiN * sizeof(DTYPE_Y);
    pipeLineContext_.communicationContext->sendOffset = tailMMultiN * sizeof(DTYPE_Y);
    pipeLineContext_.communicationContext->recvOffset = tailMMultiN * sizeof(DTYPE_Y);
    pipeLineContext_.communicationContext->sendCount = tailMMultiN;
    pipeLineContext_.communicationContext->strideCount =
        static_cast<uint64_t>(mc2Tiling_.rankM) * mc2Tiling_.rankN / mc2Tiling_.rankDim;
    pipeLineContext_.communicationContext->hcclDataType = yHcclDataType_;

    pipeLine_->Process(taskCnt);
}

} // namespace Mc2Kernel
#endif
