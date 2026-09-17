/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_HPP
#define GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_HPP

#include "../../../attn_infra/gbsa_base_defs.hpp"
#include "../../../attn_infra/arch/gbsa_resource.hpp"
#include "../../../attn_infra/epilogue/gbsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/block/gbsa_block_epilogue_arch35_utils.hpp"
#include "../../../attn_infra/gbsa_gemm_coord.hpp"
#include "../../../tla/gbsa_tla_tensor.hpp"
#include "../../../tla/gbsa_tla_layout.hpp"

namespace NpuArch::Epilogue::Block {

enum class KvBaseTileRegSplitStagesHighPrec {
    ONE,
    TWO
};

template <class OutputType_, class LayoutS_>
class BlockEpilogue<EpilogueOnlineSoftmaxBsa, OutputType_, Gemm::GemmType<float, LayoutS_>> {
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = float;
    using LayoutOutput = typename OutputType_::Layout;

    static constexpr uint32_t ELE_NUM_PER_C0 = 16;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 128;
    static constexpr uint32_t SM_VREG_SIZE = 64;
    static constexpr uint32_t UB_S_P_BUF_STAGES = 2;
    static constexpr uint32_t UB_DM_BUF_MAX_STAGES = 3;
    static constexpr float MIN_VALUE = -3.402823466e+38F;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_,
                                    UBufTileHelper &uBufTileHelper)
        : scaleValue(scaleValue_)
    {
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        for (uint32_t i = 0; i < UB_S_P_BUF_STAGES; ++i) {
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.sStartOffset + uBufTileHelper.qBaseTilePerSubCore *
                                                      uBufTileHelper.kvBaseTilePerSubCore * sizeof(float) * i);
            lpUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(
                uBufTileHelper.pStartOffset + uBufTileHelper.qBaseTilePerSubCore *
                                                      uBufTileHelper.kvBaseTilePerSubCore * sizeof(ElementOutput) * i);
        }
        for (uint32_t i = 0; i < UB_DM_BUF_MAX_STAGES; ++i) {
            dmUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.dmStartOffset + uBufTileHelper.qBaseTilePerSubCore * sizeof(float) * i);
        }
        lmUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.lmStartOffset);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.gmStartOffset);
        llUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.llStartOffset);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.glStartOffset);
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyPUbToPL1(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t m)
    {
        AscendC::DataCopyParams repeatParams;
        repeatParams.blockCount = tla::get<1, 1>(srcTensor.shape());
        repeatParams.blockLen = m;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - m;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / ELE_NUM_PER_C0 - m;
        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());
        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <class TensorP>
    __aicore__ inline void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape, uint32_t isFirstKvSTile,
                                      uint32_t ubSBufId, uint32_t l1PBufId, Arch::CrossCoreFlag mm1ToSmFlag,
                                      Arch::CrossCoreFlag smToMm2Flag)
    {
        AscendC::printf("tkd fp32 online-softmax\n");
        uint32_t mCopyOffset = RoundUp(actualBlockShape.m(), 8) / 2;
        uint32_t m = actualBlockShape.m() < mCopyOffset ? actualBlockShape.m() : mCopyOffset;
        m = subBlockIdx_ == 0 ? m : actualBlockShape.m() - m;
        if (m == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        }

        uint32_t n = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        uint32_t blockStride = mRound;
        uint32_t tailN = (n - 1) % SM_VREG_SIZE + 1;

        __ubuf__ float *sAddr = (__ubuf__ float *)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ ElementOutput *pAddr = (__ubuf__ ElementOutput *)lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float *nowMaxAddr = (__ubuf__ float *)lmUbTensor.GetPhyAddr();
        __ubuf__ float *lastMaxAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *nowSumAddr = (__ubuf__ float *)llUbTensor.GetPhyAddr();
        __ubuf__ float *lastSumAddr = (__ubuf__ float *)glUbTensor.GetPhyAddr();
        __ubuf__ float *expMaxAddr = (__ubuf__ float *)dmUbTensor[l1PBufId].GetPhyAddr();

        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        uint32_t kvBaseTileRegStages = CeilDiv(n, SM_VREG_SIZE);
        if (kvBaseTileRegStages == 1) {
            ComputeScaleAndMax<KvBaseTileRegSplitStagesHighPrec::ONE>(sAddr, nowMaxAddr, m, tailN, scaleValue,
                                                                      nRound);
        } else if (kvBaseTileRegStages == 2) {
            ComputeScaleAndMax<KvBaseTileRegSplitStagesHighPrec::TWO>(sAddr, nowMaxAddr, m, tailN, scaleValue,
                                                                      nRound);
        }

        AscendC::printf("tkd S after ComputeScaleAndMax\n");
        AscendC::DumpTensor(lsUbTensor[ubSBufId], 0, 16);

        AscendC::printf("tkd LocalMax after ComputeScaleAndMax\n");
        AscendC::DumpTensor(lmUbTensor, 0, 16);


        if (!isFirstKvSTile) {
            UpdateMax(nowMaxAddr, lastMaxAddr, m);
            AscendC::printf("tkd LocalMax after UpdateMax\n");
            AscendC::DumpTensor(lmUbTensor, 0, 16);    
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        if (kvBaseTileRegStages == 1) {
            ComputeExpSubSum<KvBaseTileRegSplitStagesHighPrec::ONE>(pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN,
                                                                    blockStride, nRound);
        } else if (kvBaseTileRegStages == 2) {
            ComputeExpSubSum<KvBaseTileRegSplitStagesHighPrec::TWO>(pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN,
                                                                    blockStride, nRound);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        AscendC::printf("tkd P after ComputeExpSubSum, row 0 by C0 blocks\n");
        uint32_t pC0BlockNum = CeilDiv(n, ELE_NUM_PER_C0);
        for (uint32_t c0BlockIdx = 0; c0BlockIdx < pC0BlockNum; ++c0BlockIdx) {
            uint32_t logicalBegin = c0BlockIdx * ELE_NUM_PER_C0;
            uint32_t validElemNum = n - logicalBegin;
            validElemNum = validElemNum < ELE_NUM_PER_C0 ? validElemNum : ELE_NUM_PER_C0;
            uint32_t physicalOffset = c0BlockIdx * blockStride * ELE_NUM_PER_C0;
            AscendC::printf("tkd P C0 block %u, logical [%u, %u), physical offset %u\n", c0BlockIdx,
                            logicalBegin, logicalBegin + validElemNum, physicalOffset);
            AscendC::DumpTensor(lpUbTensor[ubSBufId][physicalOffset], 100 + c0BlockIdx, validElemNum);
        }

        AscendC::printf("tkd LocalSum after ComputeExpSubSum\n");
        AscendC::DumpTensor(llUbTensor, 0, 16);    

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId], ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile =
            GetTile(l1PTensorTla, tla::MakeCoord(subBlockIdx_ * mCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        if (isFirstKvSTile) {
            SaveFirstTileStats(lastSumAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, m);

            AscendC::printf("tkd Global Max after SaveFirstTileStats\n");
            AscendC::DumpTensor(gmUbTensor, 0, 16);
            AscendC::printf("tkd Global Sum after SaveFirstTileStats\n");
            AscendC::DumpTensor(glUbTensor, 0, 16);
        } else {
            UpdateStats(lastSumAddr, expMaxAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, m);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    template <KvBaseTileRegSplitStagesHighPrec stages>
    __simd_vf__ inline void ComputeScaleAndMax(__ubuf__ float *srcUb, __ubuf__ float *newMaxUb, uint16_t m,
                                               uint32_t tailN, float dScale, uint16_t s2BaseSize)
    {
        static_assert(stages == KvBaseTileRegSplitStagesHighPrec::ONE ||
                          stages == KvBaseTileRegSplitStagesHighPrec::TWO,
                      "ComputeScaleAndMax only supports ONE or TWO stages, please use the specialized versions.");
    }

    template <>
    __simd_vf__ inline void ComputeScaleAndMax<KvBaseTileRegSplitStagesHighPrec::ONE>(
        __ubuf__ float *srcUb, __ubuf__ float *newMaxUb, uint16_t m, uint32_t tailN, float dScale,
        uint16_t s2BaseSize)
    {
        using namespace AscendC::Reg;
        RegTensor<float> minVreg;
        RegTensor<float> srcVreg;
        RegTensor<float> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * s2BaseSize);
            Muls(srcVreg, srcVreg, dScale, pregFull);
            Select(srcVreg, srcVreg, minVreg, pregTail);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(srcUb + i * s2BaseSize, srcVreg, pregTail);
            ReduceMax(maxTmpVreg, srcVreg, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    template <>
    __simd_vf__ inline void ComputeScaleAndMax<KvBaseTileRegSplitStagesHighPrec::TWO>(
        __ubuf__ float *srcUb, __ubuf__ float *newMaxUb, uint16_t m, uint32_t tailN, float dScale,
        uint16_t s2BaseSize)
    {
        using namespace AscendC::Reg;
        RegTensor<float> srcVreg0;
        RegTensor<float> srcVreg1;
        RegTensor<float> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg0, srcUb + i * s2BaseSize);
            LoadAlign(srcVreg1, srcUb + i * s2BaseSize + FLOAT_VECTOR_SIZE);
            Muls(srcVreg0, srcVreg0, dScale, pregFull);
            Muls(srcVreg1, srcVreg1, dScale, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(srcUb + i * s2BaseSize, srcVreg0, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(srcUb + i * s2BaseSize + FLOAT_VECTOR_SIZE, srcVreg1,
                                                        pregTail);
            Max<float, MaskMergeMode::MERGING>(srcVreg0, srcVreg0, srcVreg1, pregTail);
            ReduceMax(maxTmpVreg, srcVreg0, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    __simd_vf__ inline void UpdateMax(__ubuf__ float *nowMaxUb, __ubuf__ float *lastMaxUb, uint32_t m)
    {
        using namespace AscendC::Reg;
        RegTensor<float> nowMaxVreg;
        RegTensor<float> lastMaxVreg;
        MaskReg preg = UpdateMask<float>(m);
        LoadAlign(nowMaxVreg, nowMaxUb);
        LoadAlign(lastMaxVreg, lastMaxUb);
        Max(nowMaxVreg, nowMaxVreg, lastMaxVreg, preg);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(nowMaxUb, nowMaxVreg, preg);
    }

    template <KvBaseTileRegSplitStagesHighPrec stages>
    __simd_vf__ inline void ComputeExpSubSum(__ubuf__ ElementOutput *expUb, __ubuf__ float *srcUb,
                                             __ubuf__ float *nowMaxUb, __ubuf__ float *expSumUb, uint16_t m,
                                             uint32_t tailN, uint32_t blockStride, uint16_t s2BaseSize)
    {
        static_assert(stages == KvBaseTileRegSplitStagesHighPrec::ONE ||
                          stages == KvBaseTileRegSplitStagesHighPrec::TWO,
                      "ComputeExpSubSum only supports ONE or TWO stages, please use the specialized versions.");
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum<KvBaseTileRegSplitStagesHighPrec::ONE>(
        __ubuf__ ElementOutput *expUb, __ubuf__ float *srcUb, __ubuf__ float *nowMaxUb, __ubuf__ float *expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t s2BaseSize)
    {
        using namespace AscendC::Reg;
        constexpr static CastTrait castTraitZero = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_ROUND};
        RegTensor<float> srcVreg;
        RegTensor<float> expVreg;
        RegTensor<float> expSumVreg;
        RegTensor<float> maxVreg;
        RegTensor<ElementOutput> pCastVreg;
        RegTensor<ElementOutput> pVreg;
        RegTensor<ElementOutput> pTmpVreg;
        UnalignReg expSumUreg;
        MaskReg pregTail = UpdateMask<float>(tailN);
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOutput = UpdateMask<ElementOutput>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maxVreg, nowMaxUb + i);
            LoadAlign(srcVreg, srcUb + i * s2BaseSize);
            FusedExpSub(expVreg, srcVreg, maxVreg, pregTail);
            ReduceSum(expSumVreg, expVreg, pregTail);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
            Cast<ElementOutput, float, castTraitZero>(pCastVreg, expVreg, pregFloatFull);
            DeInterleave(pVreg, pTmpVreg, pCastVreg, pCastVreg);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(expUb + i * ELE_NUM_PER_C0, pVreg, blockStride,
                                                                     pregOutput);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum<KvBaseTileRegSplitStagesHighPrec::TWO>(
        __ubuf__ ElementOutput *expUb, __ubuf__ float *srcUb, __ubuf__ float *nowMaxUb, __ubuf__ float *expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t s2BaseSize)
    {
        using namespace AscendC::Reg;
        constexpr static CastTrait castTraitZero = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_ROUND};
        RegTensor<float> srcVreg0;
        RegTensor<float> srcVreg1;
        RegTensor<float> expVreg0;
        RegTensor<float> expVreg1;
        RegTensor<float> expSumVreg;
        RegTensor<float> maxVreg;
        RegTensor<ElementOutput> pCastVreg0;
        RegTensor<ElementOutput> pCastVreg1;
        RegTensor<ElementOutput> pVreg;
        RegTensor<ElementOutput> pTmpVreg;
        UnalignReg expSumUreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(tailN);
        uint32_t outputN = FLOAT_VECTOR_SIZE + tailN;
        MaskReg pregOutput = UpdateMask<ElementOutput>(outputN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maxVreg, nowMaxUb + i);
            LoadAlign(srcVreg0, srcUb + i * s2BaseSize);
            LoadAlign(srcVreg1, srcUb + i * s2BaseSize + FLOAT_VECTOR_SIZE);
            FusedExpSub(expVreg0, srcVreg0, maxVreg, pregFull);
            FusedExpSub(expVreg1, srcVreg1, maxVreg, pregTail);
            Cast<ElementOutput, float, castTraitZero>(pCastVreg0, expVreg0, pregFull);
            Cast<ElementOutput, float, castTraitZero>(pCastVreg1, expVreg1, pregFull);
            DeInterleave(pVreg, pTmpVreg, pCastVreg0, pCastVreg1);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(expUb + i * ELE_NUM_PER_C0, pVreg, blockStride,
                                                                     pregOutput);
            Add<float, MaskMergeMode::MERGING>(expVreg0, expVreg0, expVreg1, pregTail);
            ReduceSum(expSumVreg, expVreg0, pregFull);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    __simd_vf__ inline void SaveFirstTileStats(__ubuf__ float *sumUb, __ubuf__ float *maxUb,
                                               __ubuf__ float *newSumUb, __ubuf__ float *newMaxUb, uint32_t m)
    {
        using namespace AscendC::Reg;
        RegTensor<float> sumVreg;
        RegTensor<float> maxVreg;
        MaskReg preg = UpdateMask<float>(m);
        LoadAlign(sumVreg, newSumUb);
        LoadAlign(maxVreg, newMaxUb);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb, sumVreg, preg);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb, maxVreg, preg);
    }

    __simd_vf__ inline void UpdateStats(__ubuf__ float *sumUb, __ubuf__ float *expMaxUb, __ubuf__ float *maxUb,
                                        __ubuf__ float *newSumUb, __ubuf__ float *newMaxUb, uint32_t m)
    {
        using namespace AscendC::Reg;
        RegTensor<float> lastMaxVreg;
        RegTensor<float> newMaxVreg;
        RegTensor<float> expMaxVreg;
        RegTensor<float> lastSumVreg;
        RegTensor<float> newSumVreg;
        RegTensor<float> updatedSumVreg;
        MaskReg preg = UpdateMask<float>(m);
        LoadAlign(lastMaxVreg, maxUb);
        LoadAlign(newMaxVreg, newMaxUb);
        FusedExpSub(expMaxVreg, lastMaxVreg, newMaxVreg, preg);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb, expMaxVreg, preg);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb, newMaxVreg, preg);
        LoadAlign(lastSumVreg, sumUb);
        LoadAlign(newSumVreg, newSumUb);
        Mul(updatedSumVreg, expMaxVreg, lastSumVreg, preg);
        Add(updatedSumVreg, updatedSumVreg, newSumVreg, preg);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb, updatedSumVreg, preg);
    }

private:
    float scaleValue;
    AscendC::LocalTensor<float> lsUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<ElementOutput> lpUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<float> lmUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> dmUbTensor[UB_DM_BUF_MAX_STAGES];
    AscendC::LocalTensor<float> llUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    uint32_t subBlockIdx_;
};

} // namespace NpuArch::Epilogue::Block

#endif // GBSA_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_HPP
