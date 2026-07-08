/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_S_TRANS_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_HIGH_PREC_S_TRANS_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/epilogue/block/block_epilogue_arch35_utils.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

template <
    class OutputType_,
    class LayoutS_>
class BlockEpilogue<
    // S is transposed, has shape [kvSTile, qSTile], softmax is done columnwise
    // P would be rearranged from [kvSTile, qSTile] to zN,
    // which can be viewed as from [qSTile, kvSTile] to nZ
    EpilogueOnlineSoftmaxBsa<true>,
    OutputType_,
    Gemm::GemmType<float, LayoutS_>>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = float;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    static constexpr uint32_t BLOCK_SIZE = 16;
    static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t VECTOR_SIZE = 128;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;
    static constexpr uint32_t MAX_UB_P_ELEM_NUM = 8320;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;

    static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_8 = 8;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    static constexpr uint32_t ROW_OPS_SPEC_MASK_2 = 2;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;

    static constexpr uint32_t SPLIT_COL_IDX_2 = 2;
    static constexpr uint32_t SPLIT_COL_IDX_3 = 3;
    static constexpr ElementInput MIN_VALUE = -65504.0f;
    static constexpr uint32_t HALF_REP_SIZE = 128;
    static constexpr uint32_t FLOAT_REP_SIZE = 64;
    static constexpr uint32_t BLOCK_REP_SIZE = 8;
    static constexpr uint32_t REPEAT_STRIDE = 1;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 128;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_VREG_SIZE = 256 / sizeof(ElementInput);
    static constexpr uint32_t UB_S_P_BUF_STAGES = 2;
    static constexpr uint32_t UB_DM_BUF_MAX_STAGES = 3;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_, UBufTileHelper &uBufTileHelper)
    {
        subBlockIdx_ = AscendC::GetSubBlockIdx();
        scaleValue = static_cast<ElementInput>(scaleValue_);
        uint32_t pExtraElemNum = Max(uBufTileHelper.qBaseTilePerSubCore, uBufTileHelper.kvBaseTilePerSubCore);
        for (uint32_t i = 0; i < UB_S_P_BUF_STAGES; i++) {
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(
                uBufTileHelper.sStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore * sizeof(ElementInput) * i);
            lpUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(
                uBufTileHelper.pStartOffset +
                (uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore + pExtraElemNum) *
                sizeof(ElementOutput) * i);
        }
        for (uint32_t i = 0; i < UB_DM_BUF_MAX_STAGES; i++) {
            dmUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.dmStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * sizeof(float) * i);
        }
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.gmStartOffset);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.glStartOffset);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(uBufTileHelper.lmStartOffset);
        llUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(uBufTileHelper.llStartOffset);
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
    }

    template <class TensorP>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag)
    {   
        uint32_t nCopyOffset = RoundUp(actualBlockShape.m(), 32) / 2;
        uint32_t n = actualBlockShape.m() < nCopyOffset ? actualBlockShape.m() : nCopyOffset;
        n = subBlockIdx_ == 0 ? n : actualBlockShape.m() - n;
        if (n == 0) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            return;
        }
        uint32_t m = actualBlockShape.n();
        uint16_t mRound = RoundUp(m, C0_NUM_PER_FRACTAL);
        uint16_t nRound = RoundUp(n, ELE_NUM_PER_C0);
        // alter nd2nz dst stride (typically 64->65) to alleviate bank conflict
        uint32_t blockStride = mRound / 2 + 1;
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;
        int16_t mLoops = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;
        __ubuf__ ElementOutput *pAddr = (__ubuf__ ElementOutput*) lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ ElementInput *sAddr = (__ubuf__ ElementInput*) lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float *lastMaxAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastMaxStartAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastSumAddr = (__ubuf__ float*) glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxAddr = (__ubuf__ float*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxStartAddr = (__ubuf__ float*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowSumAddr = (__ubuf__ float*) llUbTensor.GetPhyAddr();
        __ubuf__ float *expMaxUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId].GetPhyAddr();
        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        
        uint32_t mAligendTile = mRound / 4;
        uint32_t mFirstTile = m % mAligendTile;
        uint32_t mAligned16TileNum = m / mAligendTile;
        if (mAligned16TileNum == 4) {
            mFirstTile = mAligendTile;
        }
        if (isFirstKvSTile) {
            if (mAligned16TileNum == 0) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, false, MAligendTileNum::Zero>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 1) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, false, MAligendTileNum::One>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 2) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, false, MAligendTileNum::Two>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 3) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, false, MAligendTileNum::Three>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, false, MAligendTileNum::Four>(
                    sAddr, lastMaxAddr, lastMaxStartAddr, pAddr, lastSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            }
        } else {
            if (mAligned16TileNum == 0) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, true, MAligendTileNum::Zero>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 1) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, true, MAligendTileNum::One>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 2) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, true, MAligendTileNum::Two>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else if (mAligned16TileNum == 3) {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, true, MAligendTileNum::Three>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            } else {
                ComputeScaleAndMaxDn<ElementInput, ElementOutput, true, MAligendTileNum::Four>(
                    sAddr, nowMaxAddr, lastMaxAddr, pAddr, nowSumAddr, mRound, m, tailN, mFirstTile, scaleValue, 64, blockStride, nRound,
                    expMaxUbAddr, lastSumAddr);
            }
        }
        
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId],
            ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile = GetTile(l1PTensorTla,
                tla::MakeCoord(subBlockIdx_ * nCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        
        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = nRound / 16; // 分两次搬运
        dataCopyParams.blockLen = mRound / 2;
        dataCopyParams.srcStride = 1;
        dataCopyParams.dstStride = mRound / 2;
        DataCopy(l1PTensorTla.data()[subBlockIdx_ * mRound * nCopyOffset], 
            lpUbTensor[ubSBufId], dataCopyParams);
        DataCopy(l1PTensorTla.data()[mRound * 8 + subBlockIdx_ * mRound * nCopyOffset],
            lpUbTensor[ubSBufId][blockStride * 64], dataCopyParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
    }

private:
    ElementInput scaleValue;
    AscendC::LocalTensor<ElementInput> lsUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<ElementOutput> lpUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor[UB_DM_BUF_MAX_STAGES];
    AscendC::LocalTensor<ElementInput> lmUbTensor;
    AscendC::LocalTensor<ElementInput> llUbTensor;
    uint32_t subBlockIdx_;

    enum class MAligendTileNum {
        Zero = 0,
        One = 1,
        Two = 2,
        Three = 3,
        Four = 4
    };

    template <typename ElementS, typename ElementP, bool isUpdate, MAligendTileNum mTileNum>
    __simd_vf__ inline void ComputeScaleAndMaxDn(__ubuf__ ElementS *srcUb, __ubuf__ ElementS *newMaxUb, __ubuf__ ElementS *LastMaxUbStart, __ubuf__ ElementP *expUb,
        __ubuf__ ElementS *expSumUb, uint16_t mRound, uint16_t m, uint32_t tailN, uint32_t mFirstTile, ElementInput dScale, uint16_t S2BaseSize, 
        uint32_t blockStride, uint32_t repeatStride, __ubuf__ float *expMaxUb, __ubuf__ ElementS *lastExpSumUb)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<float> src0Vreg;
        RegTensor<float> src1Vreg;
        RegTensor<float> src2Vreg;
        RegTensor<float> src3Vreg;
        RegTensor<float> src4Vreg;

        RegTensor<float> exp0Fp32Vreg;
        RegTensor<float> exp1Fp32Vreg;
        RegTensor<float> exp2Fp32Vreg;
        RegTensor<float> exp3Fp32Vreg;
        
        RegTensor<float> max0Vreg;
        RegTensor<float> max1Vreg;
        RegTensor<float> max2Vreg;
        RegTensor<float> max3Vreg;

        RegTensor<float> expMaxVreg;
        RegTensor<float> updateExpSumVreg;

        RegTensor<float> sum0Vreg;
        RegTensor<float> sum1Vreg;
        RegTensor<float> sum2Vreg;
        RegTensor<float> sum3Vreg;
        RegTensor<float> lastExpSumVreg;

        RegTensor<ElementP> vreg_x_exp_even_16;
        RegTensor<ElementP> vreg_x_exp_odd_16;
        RegTensor<ElementP> vreg_x_exp_16_packa;
        RegTensor<ElementP> vreg_x_exp_16_pack;
        RegTensor<ElementP> vreg_x_exp_even_16_1;
        RegTensor<ElementP> vreg_x_exp_odd_16_1;
        RegTensor<ElementP> vreg_x_exp_16_1_pack;
        RegTensor<ElementP> vreg_x_exp_16_1_packa;

        constexpr static CastTrait castTraitZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg preg_all_b16 = CreateMask<uint16_t, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<float>(tailN);
        uint32_t sreg_92 = (uint32_t)128ULL;
        MaskReg preg_136 = UpdateMask<uint16_t>(sreg_92);
        // std::conditional_t<AscendC::IsSameType<ElementP, bfloat16_t>::value, 
        //                    __ubuf__ bfloat16_t*, 
        //                    __ubuf__ half*> x_exp_1 = expUb + (mRound * 4);
        __ubuf__ ElementP *x_exp_1 = expUb + (mRound * 4);

        __ubuf__ float *srcUb0 = srcUb;
        __ubuf__ float *srcUb1 = srcUb0 + S2BaseSize;
        __ubuf__ float *srcUb2 = srcUb0 + S2BaseSize * 2;
        __ubuf__ float *srcUb3 = srcUb0 + S2BaseSize * 3;
        __ubuf__ float *srcUb4 = srcUb0 + S2BaseSize * (m / 4) * 4;
        Duplicate(max0Vreg, MIN_VALUE);
        Duplicate(max1Vreg, MIN_VALUE);
        Duplicate(max2Vreg, MIN_VALUE);
        Duplicate(max3Vreg, MIN_VALUE);
        for (int32_t i = 0; i < int32_t(m / 4); i++) {
            LoadAlign(src0Vreg, srcUb0 + i * S2BaseSize * 4);
            LoadAlign(src1Vreg, srcUb1 + i * S2BaseSize * 4);
            LoadAlign(src2Vreg, srcUb2 + i * S2BaseSize * 4);
            LoadAlign(src3Vreg, srcUb3 + i * S2BaseSize * 4);
            Max(max0Vreg, max0Vreg, src0Vreg, pregTailN);
            Max(max1Vreg, max1Vreg, src1Vreg, pregTailN);
            Max(max2Vreg, max2Vreg, src2Vreg, pregTailN);
            Max(max3Vreg, max3Vreg, src3Vreg, pregTailN);
        }
        Max(max0Vreg, max0Vreg, max2Vreg, pregTailN);
        Max(max1Vreg, max1Vreg, max3Vreg, pregTailN);
        Max(max0Vreg, max0Vreg, max1Vreg, pregTailN);
        for (int32_t i = 0; i < int32_t(m % 4); i++) {
            LoadAlign(src4Vreg, srcUb4 + i * S2BaseSize);
            Max(max0Vreg, max0Vreg, src4Vreg, pregTailN);
        }
        Muls(max0Vreg, max0Vreg, dScale, pregTailN);

        if constexpr (isUpdate) {
            LoadAlign(max1Vreg, LastMaxUbStart);
            LoadAlign(lastExpSumVreg, lastExpSumUb);
            Max(max0Vreg, max0Vreg, max1Vreg, pregTailN);
            FusedExpSub(expMaxVreg, max1Vreg, max0Vreg, pregTailN);
            Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb, expMaxVreg, pregTailN);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(LastMaxUbStart, max0Vreg, pregTailN);
        }

        StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ float *&)newMaxUb, max0Vreg, pregFull);

        Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(sum0Vreg, 0, pregFull);
        Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(sum1Vreg, 0, pregFull);
        Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(sum2Vreg, 0, pregFull);
        Duplicate<float, AscendC::MicroAPI::MaskMergeMode::ZEROING, float>(sum3Vreg, 0, pregFull);

        for (int32_t i = 0; i < mFirstTile; i++) {
            LoadAlign(src0Vreg, srcUb + i * S2BaseSize);
            LoadAlign(src1Vreg, srcUb + mRound * S2BaseSize / 4 + i * S2BaseSize);
            LoadAlign(src2Vreg, srcUb + mRound * S2BaseSize / 2 + i * S2BaseSize);
            LoadAlign(src3Vreg, srcUb + mRound * S2BaseSize / 2 + mRound * S2BaseSize / 4 + i * S2BaseSize);

            Muls(src0Vreg, src0Vreg, dScale, pregTailN);
            Muls(src1Vreg, src1Vreg, dScale, pregTailN);
            Muls(src2Vreg, src2Vreg, dScale, pregTailN);
            Muls(src3Vreg, src3Vreg, dScale, pregTailN);

            FusedExpSub(exp0Fp32Vreg, src0Vreg, max0Vreg, pregTailN);
            FusedExpSub(exp1Fp32Vreg, src1Vreg, max0Vreg, pregTailN);
            FusedExpSub(exp2Fp32Vreg, src2Vreg, max0Vreg, pregTailN);
            FusedExpSub(exp3Fp32Vreg, src3Vreg, max0Vreg, pregTailN);

            Cast<ElementP, float, castTraitZero>(vreg_x_exp_even_16, exp0Fp32Vreg, pregFull);
            Cast<ElementP, float, castTraitZero>(vreg_x_exp_odd_16, exp2Fp32Vreg, pregFull);
            DeInterleave(vreg_x_exp_16_pack, vreg_x_exp_16_packa, vreg_x_exp_even_16, vreg_x_exp_odd_16);
            Cast<ElementP, float, castTraitZero>(vreg_x_exp_even_16_1, exp1Fp32Vreg, pregFull);
            Cast<ElementP, float, castTraitZero>(vreg_x_exp_odd_16_1, exp3Fp32Vreg, pregFull);
            DeInterleave(vreg_x_exp_16_1_pack, vreg_x_exp_16_1_packa, vreg_x_exp_even_16_1, vreg_x_exp_odd_16_1);
            StoreAlign<ElementP, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ ElementP *&)expUb), vreg_x_exp_16_pack, blockStride, 1, preg_136); 

            if constexpr (mTileNum >= MAligendTileNum::Zero) {
                Add(sum0Vreg, exp0Fp32Vreg, sum0Vreg, pregTailN);
            }
            if constexpr (mTileNum >= MAligendTileNum::Two) {
                Add(sum2Vreg, exp2Fp32Vreg, sum2Vreg, pregTailN);
            }
            StoreAlign<ElementP, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ((__ubuf__ ElementP *&)x_exp_1), vreg_x_exp_16_1_pack, blockStride, 1, preg_136);
            if constexpr (mTileNum >= MAligendTileNum::One) {
                Add(sum1Vreg, exp1Fp32Vreg, sum1Vreg, pregTailN);
            }
            if constexpr (mTileNum >= MAligendTileNum::Three) {
                Add(sum3Vreg, exp3Fp32Vreg, sum3Vreg, pregTailN);
            }
        }
        if constexpr (mTileNum < MAligendTileNum::Four) {
            for (int32_t i = mFirstTile; i < int32_t(mRound / 4); i++) {
                LoadAlign(src0Vreg, srcUb + i * S2BaseSize);
                LoadAlign(src1Vreg, srcUb + mRound * S2BaseSize / 4 + i * S2BaseSize);
                LoadAlign(src2Vreg, srcUb + mRound * S2BaseSize / 2 + i * S2BaseSize);
                LoadAlign(src3Vreg, srcUb + mRound * S2BaseSize / 2 + mRound * S2BaseSize / 4 + i * S2BaseSize);

                Muls(src0Vreg, src0Vreg, dScale, pregTailN);
                Muls(src1Vreg, src1Vreg, dScale, pregTailN);
                Muls(src2Vreg, src2Vreg, dScale, pregTailN);
                Muls(src3Vreg, src3Vreg, dScale, pregTailN);

                FusedExpSub(exp0Fp32Vreg, src0Vreg, max0Vreg, pregTailN);
                FusedExpSub(exp1Fp32Vreg, src1Vreg, max0Vreg, pregTailN);
                FusedExpSub(exp2Fp32Vreg, src2Vreg, max0Vreg, pregTailN);
                FusedExpSub(exp3Fp32Vreg, src3Vreg, max0Vreg, pregTailN);
                
                Cast<ElementP, float, castTraitZero>(vreg_x_exp_even_16, exp0Fp32Vreg, pregFull);
                Cast<ElementP, float, castTraitZero>(vreg_x_exp_odd_16, exp2Fp32Vreg, pregFull);
                DeInterleave(vreg_x_exp_16_pack, vreg_x_exp_16_packa, vreg_x_exp_even_16, vreg_x_exp_odd_16);
                Cast<ElementP, float, castTraitZero>(vreg_x_exp_even_16_1, exp1Fp32Vreg, pregFull);
                Cast<ElementP, float, castTraitZero>(vreg_x_exp_odd_16_1, exp3Fp32Vreg, pregFull);
                DeInterleave(vreg_x_exp_16_1_pack, vreg_x_exp_16_1_packa, vreg_x_exp_even_16_1, vreg_x_exp_odd_16_1);
                StoreAlign<ElementP, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ ElementP *&)expUb), vreg_x_exp_16_pack, blockStride, 1, preg_136); 

                if constexpr (mTileNum > MAligendTileNum::Zero) {
                    Add(sum0Vreg, exp0Fp32Vreg, sum0Vreg, pregTailN);
                }
                if constexpr (mTileNum > MAligendTileNum::Two) {
                    Add(sum2Vreg, exp2Fp32Vreg, sum2Vreg, pregTailN);
                }
                StoreAlign<ElementP, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ ElementP *&)x_exp_1), vreg_x_exp_16_1_pack, blockStride, 1, preg_136);
                if constexpr (mTileNum > MAligendTileNum::One) {
                    Add(sum1Vreg, exp1Fp32Vreg, sum1Vreg, pregTailN);
                }
            }
        }
        Add(sum0Vreg, sum0Vreg, sum2Vreg, pregFull);
        Add(sum1Vreg, sum1Vreg, sum3Vreg, pregFull);
        Add(sum0Vreg, sum0Vreg, sum1Vreg, pregFull);
        if constexpr (isUpdate) {
            Add(updateExpSumVreg, updateExpSumVreg, sum0Vreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(lastExpSumUb, updateExpSumVreg, pregFull);
        }
        StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ float *&)expSumUb, sum0Vreg, pregFull);
    }
};
}

#endif