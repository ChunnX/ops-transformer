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
 * \file matmul_all_to_all_mx_quant_matmul_hcomm_impl.h
 * \brief Apace Blaze computation stage for the QuantMatmulAllToAll Hcomm pipeline.
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "../../matmul/quant_batch_matmul/matmul_all_to_all_qbmm_mx_kernel.h"
#include "../../../../mc2_templates/computation/matmul/mc2_mx_quant_matmul.h"
#include "../../../../mc2_templates/communication/communicator/hccl_communicator.h"
#include "../../../../mc2_templates/communication/primitives/hccl_primitives.h"
#include "../../../tiling/quant_matmul_tiling_data.h"

namespace Apace {

template <typename T>
struct MatmulAllToAllMxPackFactor {
    static constexpr uint64_t VALUE = 1UL;
};

template <>
struct MatmulAllToAllMxPackFactor<fp4x2_e2m1_t> {
    static constexpr uint64_t VALUE = 2UL;
};

template <typename T>
__aicore__ inline uint64_t MatmulAllToAllMxByteOffset(uint64_t elementCount)
{
    return elementCount * sizeof(T) / MatmulAllToAllMxPackFactor<T>::VALUE;
}

template <typename T>
__aicore__ inline GM_ADDR MatmulAllToAllMxGmAddr(GM_ADDR baseAddr, uint64_t elementCount)
{
    return baseAddr + MatmulAllToAllMxByteOffset<T>(elementCount);
}

template <typename T1, typename T2>
__aicore__ inline T1 MatmulAllToAllMxCeilDiv(T1 value, T2 divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

// Apace writes the matmul result directly in the rank-major layout consumed by
// Hcomm. These fields describe that output layout; no AIV transpose buffer is
// involved in this path.
struct QuantMatmulAllToAllMxAdditionalData {
    GM_ADDR x1Scale;
    GM_ADDR x2Scale;
    uint64_t x1ScaleOffset;
    uint64_t fullN;
    uint64_t rankStride;
    uint32_t rankDim;
};

template <typename ComputationContextType>
struct QuantMatmulAllToAllPipelineContext {
    ComputationContextType *computationContext;
    MC2KernelTemplate::MC2AlltoAllContext *communicationContext;
};

/**
 * Apace-specific pure-AIC pipeline: Blaze matmul -> Hcomm AllToAll.
 *
 * The legacy pipeline contains an AIV transpose stage between computation and
 * communication. The Apace matmul writes rank-major data directly, so this
 * pipeline executes both stages on AIC. SyncAll() follows the cube-only APACE
 * implementation and waits for every AIC core to finish writing GM before
 * AIC block 0 submits the Hcomm task. Do not use SyncAll<false>() here: that
 * mode also synchronizes with AIV cores, which are not launched by this path.
 */
template <typename ComputationType, typename CommunicationType, typename ContextType>
class QuantMatmulAllToAllPipeline {
public:
    __aicore__ inline QuantMatmulAllToAllPipeline(ComputationType *computeStage, CommunicationType *commStage)
        : computeStage_(computeStage), commStage_(commStage)
    {}

    __aicore__ inline void Init()
    {
        computeStage_->Init();
        commStage_->Init();
    }

    __aicore__ inline void GetContext(ContextType *context)
    {
        context->computationContext = computeStage_->GetContextPtr();
        context->communicationContext = commStage_->GetContextPtr();
    }

    __aicore__ inline void Process(uint32_t taskCnt)
    {
        commStage_->PrepareAll(taskCnt);
        for (uint32_t index = 0; index < taskCnt; ++index) {
            if ASCEND_IS_AIC {
                computeStage_->Process(index);
                AscendC::SyncAll();
                commStage_->Process(index);
            }
        }
    }

    __aicore__ inline void End()
    {
        computeStage_->End();
        commStage_->End();
    }

private:
    ComputationType *computeStage_;
    CommunicationType *commStage_;
};

/**
 * This class intentionally implements the same Init/GetContextPtr/Process/End
 * contract as the legacy MC2 computation component. Its Blaze kernel writes
 * [rank, M, N/rank] directly so Hcomm can consume the result without transpose.
 */
template <typename X1Type, typename X2Type, typename YType>
class QuantMatmulAllToAllMxComputation {
public:
    using TilingType = QuantMatmulTilingData;
    using ContextType = MC2KernelTemplate::MC2MMContext<QuantMatmulAllToAllMxAdditionalData, TilingType>;

    explicit __aicore__ inline QuantMatmulAllToAllMxComputation(AscendC::TPipe *pipe)
    {
        (void)pipe;
    }

    __aicore__ inline void Init() {}
    __aicore__ inline ContextType *GetContextPtr() { return &context_; }
    __aicore__ inline void Process(uint32_t taskIndex);
    __aicore__ inline void End() {}

private:
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::DNExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC, BiasType, LayoutBias>;
    using KernelImpl = Blaze::Gemm::Kernel::MatmulAllToAllQbmmMxKernel<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename KernelImpl::Params;
    using BlockMmadParams = typename KernelImpl::BlockMmadParams;
    using L1Params = typename KernelImpl::L1Params;
    using BlockSchedulerParams = typename KernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename KernelImpl::QBMMTiling;

    ContextType context_{};
    KernelImpl kernel_{};
};

template <typename X1Type, typename X2Type, typename YType>
__aicore__ inline void QuantMatmulAllToAllMxComputation<X1Type, X2Type, YType>::Process(uint32_t taskIndex)
{
    const TilingType &mmTile = *context_.tilingDataPtr;
    GM_ADDR aGM = context_.baseData.aGM + taskIndex * context_.baseData.aOffset;
    GM_ADDR cGM = context_.baseData.cGM + taskIndex * context_.baseData.cOffset;
    GM_ADDR x1Scale = context_.additionalData.x1Scale + taskIndex * context_.additionalData.x1ScaleOffset;

    ProblemShape problemShape{mmTile.m, mmTile.n, mmTile.k, 1UL};
    BlockMmadParams mmParams;
    mmParams.aGmAddr = aGM;
    mmParams.scaleAGmAddr = x1Scale;
    mmParams.bGmAddr = context_.baseData.bGM;
    mmParams.scaleBGmAddr = context_.additionalData.x2Scale;
    mmParams.cGmAddr = cGM;
    mmParams.biasGmAddr = context_.baseData.biasGM;

    L1Params l1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
    BlockSchedulerParams schedulerParams{
        mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
        mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};
    QBMMTiling qbmmParams{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c,
                          context_.baseData.biasGM != nullptr};
    Params params{problemShape, mmParams, l1Params, schedulerParams, qbmmParams,
                  context_.additionalData.fullN, context_.additionalData.rankStride,
                  context_.additionalData.rankDim};
    kernel_(params);
}

/**
 * Dedicated APACE implementation for MatmulAllToAll MXFP4/MXFP8.
 *
 * The complete APACE path stays here: tiling interpretation, Blaze output
 * layout, main/tail address calculation and Hcomm submission. The legacy
 * MxQuantMatmulAlltoAllArch35 wrapper is intentionally not reused, matching
 * the isolation pattern of AllToAllMxQuantMatmulHcommImpl.
 */
template <typename X1Type, typename X2Type, typename YType, typename TilingDataType,
          AscendC::HcclServerType ServerType>
class MatmulAllToAllMxQuantMatmulHcommImpl {
public:
    explicit __aicore__ inline MatmulAllToAllMxQuantMatmulHcommImpl(TilingDataType *tilingData)
        : tilingData_(tilingData), computation_(nullptr), communication_(tilingData),
          pipeline_(&computation_, &communication_)
    {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR x1Scale,
                                GM_ADDR x2Scale, GM_ADDR workspaceGM)
    {
        x1_ = x1;
        x2_ = x2;
        bias_ = bias;
        y_ = y;
        x1Scale_ = x1Scale;
        x2Scale_ = x2Scale;
        workspaceGM_ = workspaceGM;

        pipeline_.Init();
        pipeline_.GetContext(&pipelineContext_);
    }

    __aicore__ inline void Run()
    {
        const auto &tilingInfo = tilingData_->matmulAllToAllTilingInfo;
        if (tilingInfo.tileCnt > 0) {
            ProcessTile(tilingInfo.tileCnt);
        }
        if (tilingInfo.tailCnt > 0) {
            ProcessTail(tilingInfo.tailCnt);
        }
        pipeline_.End();
    }

private:
    using ComputationType = QuantMatmulAllToAllMxComputation<X1Type, X2Type, YType>;
    using ComputationContextType = typename ComputationType::ContextType;
    using CommunicationType = MC2KernelTemplate::HcclCommunicator<
        true, ServerType, MC2KernelTemplate::MC2AlltoAllContext, TilingDataType,
        MC2KernelTemplate::MC2AlltoAllPrimitives, 1, 0>;
    using PipelineContextType = QuantMatmulAllToAllPipelineContext<ComputationContextType>;
    using PipelineType = QuantMatmulAllToAllPipeline<ComputationType, CommunicationType, PipelineContextType>;

    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t MXFP_SCALE_FACTOR = 2UL;

    __aicore__ inline void ProcessTile(uint32_t taskCnt)
    {
        const auto &tilingInfo = tilingData_->matmulAllToAllTilingInfo;
        const uint64_t rankNPerPeer = static_cast<uint64_t>(tilingInfo.rankN) / tilingInfo.rankDim;
        auto *computationContext = pipelineContext_.computationContext;
        auto *communicationContext = pipelineContext_.communicationContext;

        computationContext->baseData.aGM = x1_;
        computationContext->baseData.bGM = x2_;
        computationContext->baseData.cGM = workspaceGM_;
        computationContext->baseData.biasGM = bias_;
        computationContext->baseData.aOffset = MatmulAllToAllMxByteOffset<X1Type>(
            static_cast<uint64_t>(tilingInfo.tileM) * tilingInfo.rankK);
        computationContext->baseData.bOffset = 0UL;
        computationContext->baseData.cOffset = MatmulAllToAllMxByteOffset<YType>(
            static_cast<uint64_t>(tilingInfo.tileM) * rankNPerPeer);
        computationContext->additionalData.x1Scale = x1Scale_;
        computationContext->additionalData.x2Scale = x2Scale_;
        computationContext->additionalData.x1ScaleOffset = MatmulAllToAllMxByteOffset<AscendC::fp8_e8m0_t>(
            static_cast<uint64_t>(tilingInfo.tileM) *
            MatmulAllToAllMxCeilDiv(tilingInfo.rankK, MXFP_GROUP_SIZE) * MXFP_SCALE_FACTOR);
        computationContext->additionalData.fullN = tilingInfo.rankN;
        computationContext->additionalData.rankStride =
            static_cast<uint64_t>(tilingInfo.rankM) * rankNPerPeer;
        computationContext->additionalData.rankDim = tilingInfo.rankDim;
        computationContext->tilingDataPtr = &(tilingData_->tileQbmmTilingData);

        communicationContext->taskCnt = tilingInfo.tileCnt;
        communicationContext->sendBuffer = workspaceGM_;
        communicationContext->recvBuffer = y_;
        communicationContext->sendOffset =
            static_cast<uint64_t>(tilingInfo.tileM) * rankNPerPeer * sizeof(YType);
        communicationContext->recvOffset = communicationContext->sendOffset;
        communicationContext->sendCount = static_cast<uint64_t>(tilingInfo.tileM) * rankNPerPeer;
        communicationContext->strideCount = static_cast<uint64_t>(tilingInfo.rankM) * rankNPerPeer;
        communicationContext->hcclDataType = tilingInfo.hcclDataType;

        pipeline_.Process(taskCnt);
    }

    __aicore__ inline void ProcessTail(uint32_t taskCnt)
    {
        const auto &tilingInfo = tilingData_->matmulAllToAllTilingInfo;
        const uint64_t rankNPerPeer = static_cast<uint64_t>(tilingInfo.rankN) / tilingInfo.rankDim;
        const uint64_t mainRows = static_cast<uint64_t>(tilingInfo.tileM) * tilingInfo.tileCnt;
        auto *computationContext = pipelineContext_.computationContext;
        auto *communicationContext = pipelineContext_.communicationContext;

        computationContext->baseData.aGM =
            MatmulAllToAllMxGmAddr<X1Type>(x1_, mainRows * tilingInfo.rankK);
        computationContext->baseData.bGM = x2_;
        computationContext->baseData.cGM =
            MatmulAllToAllMxGmAddr<YType>(workspaceGM_, mainRows * rankNPerPeer);
        computationContext->baseData.biasGM = bias_;
        computationContext->baseData.aOffset = MatmulAllToAllMxByteOffset<X1Type>(
            static_cast<uint64_t>(tilingInfo.tailM) * tilingInfo.rankK);
        computationContext->baseData.bOffset = 0UL;
        computationContext->baseData.cOffset = MatmulAllToAllMxByteOffset<YType>(
            static_cast<uint64_t>(tilingInfo.tailM) * rankNPerPeer);
        computationContext->additionalData.x1Scale = MatmulAllToAllMxGmAddr<AscendC::fp8_e8m0_t>(
            x1Scale_, mainRows * MatmulAllToAllMxCeilDiv(tilingInfo.rankK, MXFP_GROUP_SIZE) * MXFP_SCALE_FACTOR);
        computationContext->additionalData.x2Scale = x2Scale_;
        computationContext->additionalData.x1ScaleOffset = MatmulAllToAllMxByteOffset<AscendC::fp8_e8m0_t>(
            static_cast<uint64_t>(tilingInfo.tailM) *
            MatmulAllToAllMxCeilDiv(tilingInfo.rankK, MXFP_GROUP_SIZE) * MXFP_SCALE_FACTOR);
        computationContext->additionalData.fullN = tilingInfo.rankN;
        computationContext->additionalData.rankStride =
            static_cast<uint64_t>(tilingInfo.rankM) * rankNPerPeer;
        computationContext->additionalData.rankDim = tilingInfo.rankDim;
        computationContext->tilingDataPtr = &(tilingData_->tailQbmmTilingData);

        communicationContext->taskCnt = tilingInfo.tailCnt;
        communicationContext->sendBuffer = computationContext->baseData.cGM;
        communicationContext->recvBuffer = y_ + mainRows * rankNPerPeer * sizeof(YType);
        communicationContext->sendOffset =
            static_cast<uint64_t>(tilingInfo.tailM) * rankNPerPeer * sizeof(YType);
        communicationContext->recvOffset = communicationContext->sendOffset;
        communicationContext->sendCount = static_cast<uint64_t>(tilingInfo.tailM) * rankNPerPeer;
        communicationContext->strideCount = static_cast<uint64_t>(tilingInfo.rankM) * rankNPerPeer;
        communicationContext->hcclDataType = tilingInfo.hcclDataType;

        pipeline_.Process(taskCnt);
    }

    TilingDataType *tilingData_;
    ComputationType computation_;
    CommunicationType communication_;
    PipelineType pipeline_;
    PipelineContextType pipelineContext_{};
    GM_ADDR x1_{nullptr};
    GM_ADDR x2_{nullptr};
    GM_ADDR bias_{nullptr};
    GM_ADDR y_{nullptr};
    GM_ADDR x1Scale_{nullptr};
    GM_ADDR x2Scale_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
};

} // namespace Apace
