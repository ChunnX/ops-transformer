/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_CUBE_H
#define BLOCK_ATTN_RES_PREPARE_CUBE_H

#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "block_attn_res_prepare_tiling_data.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "vf/block_attn_res_prepare_vf.h"

namespace BlockAttnResPrepare {

using namespace AscendC;

struct BlockAttnResPrepareCubeAddressPlan {
    uint64_t qL1Offset = 0U;
    uint64_t vL1Offset = 0U;
    uint64_t eL1Offset = 0U;
    uint64_t vUbOffset = 0U;
    uint64_t dotUbOffset = 0U;
    uint64_t reduceUbOffset = 0U;
    uint64_t softmaxUbOffset = 0U;
};

struct BlockAttnResPrepareCubeGmParams {
    __gm__ float *blockResidual = nullptr;
    __gm__ float *effectiveQuery = nullptr;
    __gm__ int32_t *validBlocks = nullptr;
    __gm__ float *softmaxMax = nullptr;
    __gm__ float *weightedOutput = nullptr;
    __gm__ float *softmaxSum = nullptr;
    __gm__ float *workspace = nullptr;
};

struct BlockAttnResPrepareCubeWorkInfo {
    uint64_t tIndex = 0U;
    uint64_t sStart = 0U;
    uint32_t validS = 0U;
};

__aicore__ inline BlockAttnResPrepareCubeWorkInfo DecodeCubeWork(
    uint64_t workId, const optiling::BlockAttnResPrepareCubeTilingData &tiling)
{
    const uint64_t sTileIndex = workId % tiling.sTileNum;
    const uint64_t tIndex = workId / tiling.sTileNum;
    const uint64_t sStart = sTileIndex * tiling.baseS;
    const uint64_t remainingS = tiling.totalS - sStart;
    const uint32_t validS =
        static_cast<uint32_t>(remainingS < tiling.baseS ? remainingS : tiling.baseS);
    return {tIndex, sStart, validS};
}

template <typename T>
__aicore__ inline auto MakeCubeNDExtLayout(int64_t rows, int64_t columns, int64_t rowPitch)
{
    auto shape = AscendC::Te::MakeShape(
        AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, rows),
        AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, columns));
    auto stride = AscendC::Te::MakeStride(
        AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, rowPitch),
        AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
    return AscendC::Te::MakePatternLayout<AscendC::Te::NDExtLayoutPtn,
                                          AscendC::Te::LayoutTraitDefault<T>>(shape, stride);
}

class BlockAttnResPrepareCubeSync {
public:
    static constexpr uint16_t DOT_READY_FLAG = 0U;
    static constexpr uint16_t E_READY_FLAG = 1U;
    static constexpr uint16_t MM2_CONSUMED_FLAG = 2U;
    static constexpr uint8_t SYNC_MODE = 2U;

    __aicore__ inline static void NotifyDotReady()
    {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(DOT_READY_FLAG);
    }

    __aicore__ inline static void WaitDotReady()
    {
        CrossCoreWaitFlag(DOT_READY_FLAG);
    }

    __aicore__ inline static void NotifyEReady()
    {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(E_READY_FLAG);
    }

    __aicore__ inline static void WaitEReady()
    {
        CrossCoreWaitFlag(E_READY_FLAG);
    }

    __aicore__ inline static void NotifyMm2Consumed()
    {
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(MM2_CONSUMED_FLAG);
    }

    __aicore__ inline static void WaitMm2Consumed()
    {
        CrossCoreWaitFlag(MM2_CONSUMED_FLAG);
    }
};

using CubeMm1Block = Blaze::Gemm::Block::BlockMmad<
    Blaze::Gemm::MatmulMultiBlockBasic<0U, 0U, Blaze::Gemm::KernelMmadMultiBlockBasic, 0U>, float,
    AscendC::Te::NDExtLayoutPtn, float, AscendC::Te::DNExtLayoutPtn, float,
    AscendC::Te::NDExtLayoutPtn, float, AscendC::Te::NDExtLayoutPtn>;
using CubeMm2Block = Blaze::Gemm::Block::BlockMmad<
    Blaze::Gemm::MatmulMultiBlockBasic<0U, 0U, Blaze::Gemm::KernelMmadMultiBlockBasic, 0U>, float,
    AscendC::Te::NDExtLayoutPtn, float, AscendC::Te::NDExtLayoutPtn, float,
    AscendC::Te::NDExtLayoutPtn, float, AscendC::Te::NDExtLayoutPtn>;

class BlockAttnResPrepareCubeAicService {
public:
    __aicore__ inline void RunMm1(uint64_t workId, const BlockAttnResPrepareCubeGmParams &gm,
                                  const optiling::BlockAttnResPrepareCubeTilingData &tiling)
    {
        const BlockAttnResPrepareCubeWorkInfo work = DecodeCubeWork(workId, tiling);
        __gm__ float *workspace = GetCoreWorkspace(gm, tiling);
        __gm__ float *q = gm.effectiveQuery + work.sStart * tiling.totalD;
        __gm__ float *v = gm.blockResidual + work.tIndex * tiling.totalN * tiling.totalD;

        auto qTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(q),
            MakeCubeNDExtLayout<float>(work.validS, tiling.totalD, tiling.totalD));
        auto vLayout = AscendC::Te::FrameLayoutFormat<
            AscendC::Te::DNExtLayoutPtn, AscendC::Te::LayoutTraitDefault<float>>{}(
            static_cast<int64_t>(tiling.totalD), static_cast<int64_t>(tiling.totalN));
        auto vTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(v), vLayout);
        auto dotTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(workspace),
            MakeCubeNDExtLayout<float>(work.validS, tiling.nAlign, tiling.nAlign));
        auto dummyBias = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(workspace),
            MakeCubeNDExtLayout<float>(1, tiling.nAlign, tiling.nAlign));

        CubeMm1Block::Params params{};
        params.mL1 = tiling.sAlign;
        params.nL1 = tiling.nAlign;
        params.kL1 = tiling.baseD;
        params.mL0 = tiling.sAlign;
        params.nL0 = tiling.nAlign;
        params.kL0 = tiling.baseD < 64U ? tiling.baseD : 64U;
        params.l1Stages = 1U;
        params.l0cStages = 1U;
        params.biasGmAddr = nullptr;
        mm1_.Init(params);

        using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
        BlockShape blockShape{static_cast<int64_t>(work.validS), static_cast<int64_t>(tiling.totalN),
                              static_cast<int64_t>(tiling.totalD), 0};
        mm1_(qTensor, vTensor, dummyBias, dotTensor, blockShape);
    }

    __aicore__ inline void RunMm2(uint64_t workId, const BlockAttnResPrepareCubeGmParams &gm,
                                  const optiling::BlockAttnResPrepareCubeTilingData &tiling)
    {
        const BlockAttnResPrepareCubeWorkInfo work = DecodeCubeWork(workId, tiling);
        __gm__ float *workspace = GetCoreWorkspace(gm, tiling);
        auto eTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(workspace),
            MakeCubeNDExtLayout<float>(work.validS, tiling.nAlign, tiling.nAlign));
        auto dummyBias = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(workspace),
            MakeCubeNDExtLayout<float>(1, tiling.baseDAlign, tiling.baseDAlign));

        CubeMm2Block::Params params{};
        params.mL1 = tiling.sAlign;
        params.nL1 = tiling.baseDAlign;
        params.kL1 = tiling.nAlign;
        params.mL0 = tiling.sAlign;
        params.nL0 = tiling.baseDAlign;
        params.kL0 = tiling.nAlign;
        params.l1Stages = 1U;
        params.l0cStages = 1U;
        params.biasGmAddr = nullptr;
        mm2_.Init(params);

        for (uint32_t dLoop = 0; dLoop < tiling.dTileNum; ++dLoop) {
            const uint64_t dOffset = static_cast<uint64_t>(dLoop) * tiling.baseD;
            const uint64_t remainingD = tiling.totalD - dOffset;
            const uint32_t validD =
                static_cast<uint32_t>(remainingD < tiling.baseD ? remainingD : tiling.baseD);
            __gm__ float *v = gm.blockResidual + work.tIndex * tiling.totalN * tiling.totalD + dOffset;
            __gm__ float *o =
                gm.weightedOutput + (work.sStart * tiling.totalT + work.tIndex) * tiling.totalD + dOffset;
            auto vTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(v),
                MakeCubeNDExtLayout<float>(tiling.totalN, validD, tiling.totalD));
            auto oTensor = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(o),
                MakeCubeNDExtLayout<float>(work.validS, validD, tiling.totalT * tiling.totalD));
            using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
            BlockShape blockShape{static_cast<int64_t>(work.validS), static_cast<int64_t>(validD),
                                  static_cast<int64_t>(tiling.totalN), 0};
            mm2_(eTensor, vTensor, dummyBias, oTensor, blockShape);
        }
    }

private:
    __aicore__ inline static __gm__ float *GetCoreWorkspace(
        const BlockAttnResPrepareCubeGmParams &gm, const optiling::BlockAttnResPrepareCubeTilingData &tiling)
    {
        const uint64_t coreIndex = GetBlockIdx();
        return gm.workspace + tiling.workspaceOffset / sizeof(float) + coreIndex * tiling.workspacePerCoreElems;
    }

    CubeMm1Block mm1_{};
    CubeMm2Block mm2_{};
};

class BlockAttnResPrepareCubeAivService {
public:
    __aicore__ inline void LoadVAndReduce(uint64_t workId, uint32_t dLoop, uint32_t validN,
                                          const BlockAttnResPrepareCubeGmParams &gm,
                                          const BlockAttnResPrepareCubeAddressPlan &plan,
                                          const optiling::BlockAttnResPrepareCubeTilingData &tiling)
    {
        const BlockAttnResPrepareCubeWorkInfo work = DecodeCubeWork(workId, tiling);
        const uint64_t dOffset = static_cast<uint64_t>(dLoop) * tiling.baseD;
        const uint64_t remainingD = tiling.totalD - dOffset;
        const uint32_t validD = static_cast<uint32_t>(remainingD < tiling.baseD ? remainingD : tiling.baseD);
        const uint32_t bufferIndex = dLoop % tiling.vUbBufferNum;
        const uint64_t vByteOffset =
            plan.vUbOffset + static_cast<uint64_t>(bufferIndex) * tiling.vUbElems * sizeof(float);
        __gm__ float *v = gm.blockResidual + work.tIndex * tiling.totalN * tiling.totalD + dOffset;
        CopyGmToUb(vByteOffset, v, validN, validD, tiling.baseDAlign, tiling.totalD);
        WaitMte2ToVector(GetEvent(bufferIndex));

        __ubuf__ float *vAddr = GetUbAddr<float>(vByteOffset);
        __ubuf__ float *sumSquareAddr = GetUbAddr<float>(plan.reduceUbOffset);
        for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
            if (dLoop == 0U) {
                BlockAttnResPrepareVF::AccumulateSquare<true>(
                    vAddr + static_cast<uint64_t>(nIndex) * tiling.baseDAlign, sumSquareAddr, validD, nIndex);
            } else {
                BlockAttnResPrepareVF::AccumulateSquare<false>(
                    vAddr + static_cast<uint64_t>(nIndex) * tiling.baseDAlign, sumSquareAddr, validD, nIndex);
            }
        }
        WaitVectorToMte2(GetEvent(bufferIndex));
    }

    __aicore__ inline void FinalizeSoftmax(uint64_t workId, uint32_t validN,
                                           const BlockAttnResPrepareCubeGmParams &gm,
                                           const BlockAttnResPrepareCubeAddressPlan &plan,
                                           const optiling::BlockAttnResPrepareCubeTilingData &tiling)
    {
        const BlockAttnResPrepareCubeWorkInfo work = DecodeCubeWork(workId, tiling);
        __gm__ float *workspace = GetCoreWorkspace(gm, tiling);
        // The MM1 result is stored with the aligned N pitch.  Keep the same pitch when
        // moving it back to UB; using totalN here would reinterpret every row after the
        // first one for N values that are not Cube-aligned.
        CopyGmToUb(plan.dotUbOffset, workspace, work.validS, tiling.nAlign, tiling.nAlign, tiling.nAlign);
        WaitMte2ToVector(static_cast<event_t>(EVENT_ID0));

        __ubuf__ float *sumSquareAddr = GetUbAddr<float>(plan.reduceUbOffset);
        __ubuf__ float *dotAddr = GetUbAddr<float>(plan.dotUbOffset);
        __ubuf__ float *maxAddr = GetUbAddr<float>(plan.softmaxUbOffset);
        __ubuf__ float *sumAddr = maxAddr + tiling.sAlign;
        const float reciprocalD = 1.0F / static_cast<float>(tiling.totalD);
        for (uint32_t sIndex = 0; sIndex < work.validS; ++sIndex) {
            BlockAttnResPrepareVF::FinalizeCubeSoftmax(
                sumSquareAddr, dotAddr + static_cast<uint64_t>(sIndex) * tiling.nAlign, maxAddr + sIndex,
                sumAddr + sIndex, validN, tiling.nAlign, reciprocalD, tiling.epsilon);
        }

        WaitVectorToMte3(static_cast<event_t>(EVENT_ID0));
        // Preserve the zero-padded N tail in GM.  MM2 consumes an nAlign-wide E tile.
        CopyUbToGm(workspace, plan.dotUbOffset, work.validS, tiling.nAlign, tiling.nAlign, tiling.nAlign);
        CopyUbToGm(gm.softmaxMax + work.sStart * tiling.totalT + work.tIndex, plan.softmaxUbOffset, work.validS, 1,
                   tiling.totalT, 1);
        CopyUbToGm(gm.softmaxSum + work.sStart * tiling.totalT + work.tIndex,
                   plan.softmaxUbOffset + static_cast<uint64_t>(tiling.sAlign) * sizeof(float), work.validS, 1,
                   tiling.totalT, 1);
        WaitMte3ToVector(static_cast<event_t>(EVENT_ID0));
    }

    __aicore__ inline void RunN1FastPath(uint64_t workId, const BlockAttnResPrepareCubeGmParams &gm,
                                         const BlockAttnResPrepareCubeAddressPlan &plan,
                                         const optiling::BlockAttnResPrepareCubeTilingData &tiling)
    {
        const BlockAttnResPrepareCubeWorkInfo work = DecodeCubeWork(workId, tiling);
        const uint64_t vByteOffset = plan.vUbOffset;
        const uint64_t qByteOffset = plan.vUbOffset + tiling.vUbElems * sizeof(float);
        const uint64_t statByteOffset = plan.dotUbOffset;
        __ubuf__ float *statAddr = GetUbAddr<float>(statByteOffset);

        for (uint32_t localS = 0; localS < work.validS; ++localS) {
            const uint64_t globalS = work.sStart + localS;
            for (uint32_t dLoop = 0; dLoop < tiling.dTileNum; ++dLoop) {
                const uint64_t dOffset = static_cast<uint64_t>(dLoop) * tiling.baseD;
                const uint64_t remainingD = tiling.totalD - dOffset;
                const uint32_t validD =
                    static_cast<uint32_t>(remainingD < tiling.baseD ? remainingD : tiling.baseD);
                __gm__ float *v = gm.blockResidual + work.tIndex * tiling.totalN * tiling.totalD + dOffset;
                __gm__ float *q = gm.effectiveQuery + globalS * tiling.totalD + dOffset;
                CopyGmToUb(vByteOffset, v, 1, validD, tiling.baseDAlign, tiling.totalD);
                CopyGmToUb(qByteOffset, q, 1, validD, tiling.baseDAlign, tiling.totalD);
                const event_t eventId = GetEvent(dLoop & 1U);
                WaitMte2ToVector(eventId);

                __ubuf__ float *vAddr = GetUbAddr<float>(vByteOffset);
                __ubuf__ float *qAddr = GetUbAddr<float>(qByteOffset);
                if (dLoop == 0U) {
                    BlockAttnResPrepareVF::AccumulateSquareDot<true>(qAddr, vAddr, statAddr, validD, 0U);
                } else {
                    BlockAttnResPrepareVF::AccumulateSquareDot<false>(qAddr, vAddr, statAddr, validD, 0U);
                }
                WaitVectorToMte2(eventId);
                WaitVectorToMte3(eventId);
                __gm__ float *o =
                    gm.weightedOutput + (globalS * tiling.totalT + work.tIndex) * tiling.totalD + dOffset;
                CopyUbToGm(o, vByteOffset, 1, validD, tiling.totalD, tiling.baseDAlign);
                WaitMte3ToVector(eventId);
            }

            BlockAttnResPrepareVF::FinalizeSingleBlock(
                statAddr, 1.0F / static_cast<float>(tiling.totalD), tiling.epsilon);
            WaitVectorToMte3(static_cast<event_t>(EVENT_ID0));
            CopyUbToGm(gm.softmaxMax + globalS * tiling.totalT + work.tIndex,
                       statByteOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::MAX_OFFSET) * sizeof(float),
                       1, 1, 1, 1);
            CopyUbToGm(gm.softmaxSum + globalS * tiling.totalT + work.tIndex,
                       statByteOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::SUM_OFFSET) * sizeof(float),
                       1, 1, 1, 1);
            WaitMte3ToVector(static_cast<event_t>(EVENT_ID0));
        }
    }

private:
    __aicore__ inline static event_t GetEvent(uint32_t bufferIndex)
    {
        return bufferIndex == 0U ? static_cast<event_t>(EVENT_ID0) : static_cast<event_t>(EVENT_ID1);
    }

    __aicore__ inline static void WaitMte2ToVector(event_t eventId)
    {
        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);
    }

    __aicore__ inline static void WaitVectorToMte2(event_t eventId)
    {
        set_flag(PIPE_V, PIPE_MTE2, eventId);
        wait_flag(PIPE_V, PIPE_MTE2, eventId);
    }

    __aicore__ inline static void WaitVectorToMte3(event_t eventId)
    {
        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
    }

    __aicore__ inline static void WaitMte3ToVector(event_t eventId)
    {
        set_flag(PIPE_MTE3, PIPE_V, eventId);
        wait_flag(PIPE_MTE3, PIPE_V, eventId);
    }

    template <typename T>
    __aicore__ inline static __ubuf__ T *GetUbAddr(uint64_t byteOffset)
    {
        return AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, T>(byteOffset).Get();
    }

    __aicore__ inline static void CopyGmToUb(uint64_t dstByteOffset, __gm__ float *src, int64_t rows,
                                             int64_t columns, int64_t dstPitch, int64_t srcPitch)
    {
        auto dst = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(dstByteOffset),
            MakeCubeNDExtLayout<float>(rows, columns, dstPitch));
        auto source = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(src),
            MakeCubeNDExtLayout<float>(rows, columns, srcPitch));
        auto copy = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copy, dst, source);
    }

    __aicore__ inline static void CopyUbToGm(__gm__ float *dst, uint64_t srcByteOffset, int64_t rows,
                                             int64_t columns, int64_t dstPitch, int64_t srcPitch)
    {
        auto destination = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(dst),
            MakeCubeNDExtLayout<float>(rows, columns, dstPitch));
        auto source = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, float>(srcByteOffset),
            MakeCubeNDExtLayout<float>(rows, columns, srcPitch));
        auto copy = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});
        AscendC::Te::Copy(copy, destination, source);
    }

    __aicore__ inline static __gm__ float *GetCoreWorkspace(
        const BlockAttnResPrepareCubeGmParams &gm, const optiling::BlockAttnResPrepareCubeTilingData &tiling)
    {
        const uint64_t coreIndex = GetBlockIdx() / 2U;
        return gm.workspace + tiling.workspaceOffset / sizeof(float) + coreIndex * tiling.workspacePerCoreElems;
    }
};

template <typename AIC_SERVICE, typename AIV_SERVICE>
class BlockAttnResPrepareCubeTemplate {
public:
    __aicore__ inline BlockAttnResPrepareCubeTemplate(
        const optiling::BlockAttnResPrepareCubeTilingData *__restrict tilingData, AIC_SERVICE &aicService,
        AIV_SERVICE &aivService)
        : tilingData_(tilingData), aicService_(aicService), aivService_(aivService)
    {}

    __aicore__ inline bool Init(GM_ADDR blockResidual, GM_ADDR effectiveQuery, GM_ADDR validBlocks,
                                GM_ADDR softmaxMax, GM_ADDR weightedOutput, GM_ADDR softmaxSum, GM_ADDR workspace)
    {
        initialized_ = false;
        if (tilingData_ == nullptr || blockResidual == nullptr || effectiveQuery == nullptr ||
            validBlocks == nullptr || softmaxMax == nullptr || weightedOutput == nullptr || softmaxSum == nullptr ||
            workspace == nullptr) {
            return false;
        }
        if (!IsBufferNumValid(tilingData_->qL1BufferNum) || !IsBufferNumValid(tilingData_->vL1BufferNum) ||
            tilingData_->vUbBufferNum != 2U || tilingData_->qL1Elems == 0U || tilingData_->vL1Elems == 0U ||
            tilingData_->eL1Elems == 0U || tilingData_->vUbElems == 0U || tilingData_->dotUbElems == 0U ||
            tilingData_->reduceUbElems < BlockAttnResPrepareVF::FP32_REG_ELEMS ||
            tilingData_->softmaxUbElems < static_cast<uint64_t>(2U) * tilingData_->sAlign) {
            return false;
        }
        if (tilingData_->totalT == 0U || tilingData_->totalN == 0U || tilingData_->totalS == 0U ||
            tilingData_->totalD == 0U || tilingData_->totalWorkUnits == 0U || tilingData_->usedCoreNum == 0U ||
            tilingData_->usedCoreNum > tilingData_->totalWorkUnits ||
            tilingData_->bigCoreNum >= tilingData_->usedCoreNum || tilingData_->blockFactor == 0U ||
            tilingData_->baseT != 1U || tilingData_->baseS == 0U || tilingData_->baseD == 0U ||
            tilingData_->baseDAlign < tilingData_->baseD ||
            tilingData_->sTileNum == 0U || tilingData_->dTileNum == 0U ||
            tilingData_->sAlign < tilingData_->baseS || tilingData_->nAlign < tilingData_->totalN ||
            tilingData_->dAlign < tilingData_->totalD || tilingData_->workspacePerCoreElems <
                static_cast<uint64_t>(tilingData_->sAlign) * tilingData_->nAlign ||
            tilingData_->workspacePerCoreElems == 0U || tilingData_->aicCoreNum == 0U ||
            tilingData_->aivCoreNum == 0U) {
            return false;
        }

        gmParams_.blockResidual = reinterpret_cast<__gm__ float *>(blockResidual);
        gmParams_.effectiveQuery = reinterpret_cast<__gm__ float *>(effectiveQuery);
        gmParams_.validBlocks = reinterpret_cast<__gm__ int32_t *>(validBlocks);
        gmParams_.softmaxMax = reinterpret_cast<__gm__ float *>(softmaxMax);
        gmParams_.weightedOutput = reinterpret_cast<__gm__ float *>(weightedOutput);
        gmParams_.softmaxSum = reinterpret_cast<__gm__ float *>(softmaxSum);
        gmParams_.workspace = reinterpret_cast<__gm__ float *>(workspace);
        BuildAddressPlan();
        initialized_ = true;
        return true;
    }

    __aicore__ inline void Process()
    {
        if (!initialized_) {
            return;
        }
        if ASCEND_IS_AIV {
            if ((GetBlockIdx() & 1U) != 0U) {
                return;
            }
        }

        const uint32_t coreIndex = GetLogicalAicIndex();
        if (coreIndex >= tilingData_->usedCoreNum) {
            return;
        }
        const int32_t validNValue = ReadValidBlocks();
        if (validNValue < 1 || static_cast<uint64_t>(validNValue) > tilingData_->totalN) {
            return;
        }
        const uint32_t validN = static_cast<uint32_t>(validNValue);
        const uint64_t workStart = static_cast<uint64_t>(coreIndex) * tilingData_->blockFactor +
                                   (coreIndex < tilingData_->bigCoreNum ? coreIndex : tilingData_->bigCoreNum);
        const uint32_t workCount =
            tilingData_->blockFactor + (coreIndex < tilingData_->bigCoreNum ? 1U : 0U);

        for (uint32_t workOffset = 0; workOffset < workCount; ++workOffset) {
            const uint64_t workId = workStart + workOffset;
            if (validN == 1U) {
                if ASCEND_IS_AIV {
                    aivService_.RunN1FastPath(workId, gmParams_, addressPlan_, *tilingData_);
                }
                continue;
            }
            ProcessWork(workId, validN);
        }
    }

private:
    __aicore__ inline static bool IsBufferNumValid(uint32_t bufferNum)
    {
        return bufferNum == 1U || bufferNum == 2U;
    }

    __aicore__ inline uint32_t GetLogicalAicIndex() const
    {
        if ASCEND_IS_AIV {
            return GetBlockIdx() / 2U;
        }
        return GetBlockIdx();
    }

    __aicore__ inline void BuildAddressPlan()
    {
        addressPlan_.qL1Offset = 0U;
        addressPlan_.vL1Offset =
            static_cast<uint64_t>(tilingData_->qL1BufferNum) * tilingData_->qL1Elems * sizeof(float);
        addressPlan_.eL1Offset = addressPlan_.vL1Offset +
                                 static_cast<uint64_t>(tilingData_->vL1BufferNum) * tilingData_->vL1Elems *
                                     sizeof(float);
        addressPlan_.vUbOffset = 0U;
        addressPlan_.dotUbOffset =
            static_cast<uint64_t>(tilingData_->vUbBufferNum) * tilingData_->vUbElems * sizeof(float);
        addressPlan_.reduceUbOffset = addressPlan_.dotUbOffset + tilingData_->dotUbElems * sizeof(float);
        addressPlan_.softmaxUbOffset = addressPlan_.reduceUbOffset + tilingData_->reduceUbElems * sizeof(float);
    }

    __aicore__ inline int32_t ReadValidBlocks() const
    {
        auto validBlocksTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(gmParams_.validBlocks),
            MakeCubeNDExtLayout<int32_t>(1, 1, 1));
        return validBlocksTensor[AscendC::Te::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0))];
    }

    __aicore__ inline void ProcessWork(uint64_t workId, uint32_t validN)
    {
        if ASCEND_IS_AIC {
            aicService_.RunMm1(workId, gmParams_, *tilingData_);
            BlockAttnResPrepareCubeSync::NotifyDotReady();
        }
        if ASCEND_IS_AIV {
            for (uint32_t dLoop = 0; dLoop < tilingData_->dTileNum; ++dLoop) {
                aivService_.LoadVAndReduce(workId, dLoop, validN, gmParams_, addressPlan_, *tilingData_);
            }
            BlockAttnResPrepareCubeSync::WaitDotReady();
            aivService_.FinalizeSoftmax(workId, validN, gmParams_, addressPlan_, *tilingData_);
            BlockAttnResPrepareCubeSync::NotifyEReady();
        }
        if ASCEND_IS_AIC {
            BlockAttnResPrepareCubeSync::WaitEReady();
            aicService_.RunMm2(workId, gmParams_, *tilingData_);
            BlockAttnResPrepareCubeSync::NotifyMm2Consumed();
        }
        if ASCEND_IS_AIV {
            BlockAttnResPrepareCubeSync::WaitMm2Consumed();
        }
    }

    const optiling::BlockAttnResPrepareCubeTilingData *__restrict tilingData_ = nullptr;
    AIC_SERVICE &aicService_;
    AIV_SERVICE &aivService_;
    BlockAttnResPrepareCubeAddressPlan addressPlan_{};
    BlockAttnResPrepareCubeGmParams gmParams_{};
    bool initialized_ = false;
};

} // namespace BlockAttnResPrepare

#endif // BLOCK_ATTN_RES_PREPARE_CUBE_H
