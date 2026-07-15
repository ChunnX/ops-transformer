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
    static constexpr uint32_t ELE_NUM_PER_C0_FP8 = 32;

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

    static constexpr bool FULL_QUANT_FP8 = AscendC::IsSameType<ElementOutput, fp8_e4m3fn_t>::value;

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
        uint16_t s2BaseSize = 64;

        uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;
        __ubuf__ ElementOutput *pUbAddr = (__ubuf__ ElementOutput *)lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float *sUbAddr = (__ubuf__ ElementInput *)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float *gmUbAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *gmUbStartAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *glUbAddr = (__ubuf__ float *)glUbTensor.GetPhyAddr();
        __ubuf__ float *lmUbAddr = (__ubuf__ float *)lmUbTensor.GetPhyAddr();
        __ubuf__ float *lmUbStartAddr = (__ubuf__ float *)lmUbTensor.GetPhyAddr();
        __ubuf__ float *llUbAddr = (__ubuf__ float *)llUbTensor.GetPhyAddr();
        __ubuf__ float *dmUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId].GetPhyAddr();
        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        uint32_t mAligendTile = mRound / 4;
        uint32_t mFirstTile = m % mAligendTile;
        uint32_t mAligned16TileNum = m / mAligendTile;
        if (mAligned16TileNum == 4) {
            mFirstTile = mAligendTile;
        }

        ComputeLocalMax(sUbAddr, lmUbAddr, m, s2BaseSize, tailN, scaleValue);

        // AscendC::printf("tkd lsUbTensor[ubSBufId]\n");
        // AscendC::DumpTensor(lsUbTensor[ubSBufId], 0, 64);
        // AscendC::printf("tkd lmUbTensor\n");
        // AscendC::DumpTensor(lmUbTensor, 0, 16);

        if (!isFirstKvSTile) {
            UpdateHistoricalMax(lmUbAddr, gmUbAddr, tailN);
        }

        // AscendC::printf("tkd dmUbTensor[l1PBufId]\n");
        // AscendC::DumpTensor(dmUbTensor[l1PBufId], 0, 16);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);

        if (mAligned16TileNum == 0) {
            ComputePAndLocalSum<MAligendTileNum::Zero>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                       s2BaseSize, tailN, blockStride, scaleValue);
        } else if (mAligned16TileNum == 1) {
            ComputePAndLocalSum<MAligendTileNum::One>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                      s2BaseSize, tailN, blockStride, scaleValue);
        } else if (mAligned16TileNum == 2) {
            ComputePAndLocalSum<MAligendTileNum::Two>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                      s2BaseSize, tailN, blockStride, scaleValue);
        } else if (mAligned16TileNum == 3) {
            ComputePAndLocalSum<MAligendTileNum::Three>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                        s2BaseSize, tailN, blockStride, scaleValue);
        } else {
            ComputePAndLocalSum<MAligendTileNum::Four>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                       s2BaseSize, tailN, blockStride, scaleValue);
        }
        
        // AscendC::printf("tkd lpUbTensor[ubSBufId]\n");
        // AscendC::DumpTensor(lpUbTensor[ubSBufId], 0, 64);  
        // AscendC::printf("tkd llUbTensor\n");
        // AscendC::DumpTensor(llUbTensor, 0, 16);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId], ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile =
            GetTile(l1PTensorTla, tla::MakeCoord(subBlockIdx_ * nCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = nRound / 16; // 分两次搬运
        dataCopyParams.blockLen = mRound / 2;
        dataCopyParams.srcStride = 1;
        dataCopyParams.dstStride = mRound / 2;
        DataCopy(l1PTensorTla.data()[subBlockIdx_ * mRound * nCopyOffset], lpUbTensor[ubSBufId], dataCopyParams);
        DataCopy(l1PTensorTla.data()[subBlockIdx_ * mRound * nCopyOffset + mRound * 8],
                 lpUbTensor[ubSBufId][blockStride * 64], dataCopyParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        if (isFirstKvSTile) {
            SetGlobalMaxAndGlobalSum(lmUbAddr, gmUbAddr, llUbAddr, glUbAddr, tailN);

            // AscendC::printf("tkd SetGlobalMaxAndGlobalSum\n");
            // AscendC::printf("tkd gmUbTensor\n");
            // AscendC::DumpTensor(gmUbTensor, 0, 16);
            // AscendC::printf("tkd glUbTensor\n");
            // AscendC::DumpTensor(glUbTensor, 0, 16);

        } else {
            UpdateDeltaMaxGlobalMaxAndGlobalSum(lmUbAddr, gmUbAddr, llUbAddr, glUbAddr, dmUbAddr, tailN);

            // AscendC::printf("tkd UpdateDeltaMaxGlobalMaxAndGlobalSum\n");
            // AscendC::printf("tkd dmUbTensor[l1PBufId]\n");
            // AscendC::DumpTensor(dmUbTensor[l1PBufId], 0, 16);
            // AscendC::printf("tkd gmUbTensor\n");
            // AscendC::DumpTensor(gmUbTensor, 0, 16);
            // AscendC::printf("tkd glUbTensor\n");
            // AscendC::DumpTensor(glUbTensor, 0, 16);
        }
        AscendC::PipeBarrier<PIPE_V>();
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

    __simd_vf__ inline void ComputeLocalMax(__ubuf__ float *sUb, __ubuf__ float *lmUb, uint16_t m, uint16_t s2BaseSize,
                                            uint32_t tailN, float scaleValue)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> sVreg0;
        RegTensor<float> sVreg1;
        RegTensor<float> sVreg2;
        RegTensor<float> sVreg3;
        RegTensor<float> sVreg4;

        RegTensor<float> maxVreg0;
        RegTensor<float> maxVreg1;
        RegTensor<float> maxVreg2;
        RegTensor<float> maxVreg3;

        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        __ubuf__ float *sUb0 = sUb;
        __ubuf__ float *sUb1 = sUb0 + s2BaseSize;
        __ubuf__ float *sUb2 = sUb0 + s2BaseSize * 2;
        __ubuf__ float *sUb3 = sUb0 + s2BaseSize * 3;
        __ubuf__ float *sUb4 = sUb0 + s2BaseSize * (m / 4) * 4;

        Duplicate(maxVreg0, MIN_VALUE);
        Duplicate(maxVreg1, MIN_VALUE);
        Duplicate(maxVreg2, MIN_VALUE);
        Duplicate(maxVreg3, MIN_VALUE);

        for (uint16_t i = 0; i < m / 4; i++) {
            LoadAlign(sVreg0, sUb0 + i * s2BaseSize * 4);
            LoadAlign(sVreg1, sUb1 + i * s2BaseSize * 4);
            LoadAlign(sVreg2, sUb2 + i * s2BaseSize * 4);
            LoadAlign(sVreg3, sUb3 + i * s2BaseSize * 4);
            Max(maxVreg0, maxVreg0, sVreg0, maskReg32TailN);
            Max(maxVreg1, maxVreg1, sVreg1, maskReg32TailN);
            Max(maxVreg2, maxVreg2, sVreg2, maskReg32TailN);
            Max(maxVreg3, maxVreg3, sVreg3, maskReg32TailN);
        }

        Max(maxVreg0, maxVreg0, maxVreg2, maskReg32TailN);
        Max(maxVreg1, maxVreg1, maxVreg3, maskReg32TailN);
        Max(maxVreg0, maxVreg0, maxVreg1, maskReg32TailN);
        for (uint16_t i = 0; i < m % 4; i++) {
            LoadAlign(sVreg4, sUb4 + i * s2BaseSize);
            Max(maxVreg0, maxVreg0, sVreg4, maskReg32TailN);
        }
    
        Muls(maxVreg0, maxVreg0, scaleValue, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lmUb, maxVreg0, maskReg32TailN);
    }

    __simd_vf__ inline void UpdateHistoricalMax(__ubuf__ float *lmUb, __ubuf__ float *gmUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> lmVreg;
        RegTensor<float> gmVreg;
        RegTensor<float> hmVreg;

        MaskReg maskReg32TailN = UpdateMask<float>(tailN);
        MaskReg maskReg32All = CreateMask<float, MaskPattern::ALL>();

        LoadAlign(lmVreg, lmUb);
        LoadAlign(gmVreg, gmUb);
        Max(hmVreg, lmVreg, gmVreg, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lmUb, hmVreg, maskReg32All);
    }

    template <MAligendTileNum mTileNum>
    __simd_vf__ inline void ComputePAndLocalSum(__ubuf__ float *sUb, __ubuf__ ElementOutput *pUb, __ubuf__ float *lmUb,
                                                __ubuf__ float *llUb, uint16_t mRound, uint16_t m, uint32_t mFirstTile,
                                                uint16_t s2BaseSize, uint32_t tailN, uint32_t blockStride,
                                                float scaleValue)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> sVreg0;
        RegTensor<float> sVreg1;
        RegTensor<float> sVreg2;
        RegTensor<float> sVreg3;

        RegTensor<float> pVreg0;
        RegTensor<float> pVreg1;
        RegTensor<float> pVreg2;
        RegTensor<float> pVreg3;

        RegTensor<ElementOutput> pVreg160;
        RegTensor<ElementOutput> pVreg161;
        RegTensor<ElementOutput> pVreg162;
        RegTensor<ElementOutput> pVreg163;

        RegTensor<float> hmVreg;

        RegTensor<float> sumVreg0;
        RegTensor<float> sumVreg1;
        RegTensor<float> sumVreg2;
        RegTensor<float> sumVreg3;

        RegTensor<ElementOutput> deInterleaveVreg0;
        RegTensor<ElementOutput> deInterleaveVreg1;
        RegTensor<ElementOutput> deInterleaveVreg2;
        RegTensor<ElementOutput> deInterleaveVreg3;

        MaskReg maskReg32TailN = UpdateMask<float>(tailN);
        MaskReg maskReg32All = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskReg16VL128 = CreateMask<ElementOutput, MaskPattern::VL128>();

        constexpr static CastTrait castTraitZeroROUND = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        __ubuf__ ElementOutput *pUb0 = pUb;
        __ubuf__ ElementOutput *pUb1 = pUb + (mRound * 4); // *pUb1 = pUb + (mRound / 4 * 16)

        Duplicate<float, MaskMergeMode::ZEROING, float>(sumVreg0, 0, maskReg32All);
        Duplicate<float, MaskMergeMode::ZEROING, float>(sumVreg1, 0, maskReg32All);
        Duplicate<float, MaskMergeMode::ZEROING, float>(sumVreg2, 0, maskReg32All);
        Duplicate<float, MaskMergeMode::ZEROING, float>(sumVreg3, 0, maskReg32All);

        LoadAlign<float, LoadDist::DIST_BRC_B32>(hmVreg, lmUb);
        for (uint16_t i = 0; i < mFirstTile; i++) {
            LoadAlign(sVreg0, sUb + i * s2BaseSize);
            LoadAlign(sVreg1, sUb + mRound * s2BaseSize / 4 + i * s2BaseSize);
            LoadAlign(sVreg2, sUb + mRound * s2BaseSize / 2 + i * s2BaseSize);
            LoadAlign(sVreg3, sUb + mRound * s2BaseSize / 2 + mRound * s2BaseSize / 4 + i * s2BaseSize);

            Muls(sVreg0, sVreg0, scaleValue, maskReg32TailN);
            Muls(sVreg1, sVreg1, scaleValue, maskReg32TailN);
            Muls(sVreg2, sVreg2, scaleValue, maskReg32TailN);
            Muls(sVreg3, sVreg3, scaleValue, maskReg32TailN);

            // FusedExpSub(pVreg0, sVreg0, hmVreg, maskReg16TailN);
            // FusedExpSub(pVreg1, sVreg1, hmVreg, maskReg16TailN);
            // FusedExpSub(pVreg2, sVreg2, hmVreg, maskReg16TailN);
            // FusedExpSub(pVreg3, sVreg3, hmVreg, maskReg16TailN);

            Sub(sVreg0, sVreg0, hmVreg, maskReg32TailN);
            Sub(sVreg1, sVreg1, hmVreg, maskReg32TailN);
            Sub(sVreg2, sVreg2, hmVreg, maskReg32TailN);
            Sub(sVreg3, sVreg3, hmVreg, maskReg32TailN);

            Exp(pVreg0, sVreg0, maskReg32TailN);
            Exp(pVreg1, sVreg1, maskReg32TailN);
            Exp(pVreg2, sVreg2, maskReg32TailN);
            Exp(pVreg3, sVreg3, maskReg32TailN);

            Cast<ElementOutput, float, castTraitZeroROUND>(pVreg160, pVreg0, maskReg32All);
            Cast<ElementOutput, float, castTraitZeroROUND>(pVreg162, pVreg2, maskReg32All);
            DeInterleave(deInterleaveVreg0, deInterleaveVreg1, pVreg160, pVreg162);
            Cast<ElementOutput, float, castTraitZeroROUND>(pVreg161, pVreg1, maskReg32All);
            Cast<ElementOutput, float, castTraitZeroROUND>(pVreg163, pVreg3, maskReg32All);
            DeInterleave(deInterleaveVreg2, deInterleaveVreg3, pVreg161, pVreg163);

            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb0, deInterleaveVreg0, blockStride, 1, maskReg16VL128);
            if constexpr (mTileNum >= MAligendTileNum::Zero) {
                Add(sumVreg0, sumVreg0, pVreg0, maskReg32TailN);
            }
            if constexpr (mTileNum >= MAligendTileNum::One) {
                Add(sumVreg1, sumVreg1, pVreg1, maskReg32TailN);
            }
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb1, deInterleaveVreg2, blockStride, 1, maskReg16VL128);
            if constexpr (mTileNum >= MAligendTileNum::Two) {
                Add(sumVreg2, sumVreg2, pVreg2, maskReg32TailN);
            }
            if constexpr (mTileNum >= MAligendTileNum::Three) {
                Add(sumVreg3, sumVreg3, pVreg3, maskReg32TailN);
            }
        }
        if constexpr (mTileNum < MAligendTileNum::Four) {
            for (uint16_t i = mFirstTile; i < mRound / 4; i++) {
                LoadAlign(sVreg0, sUb + i * s2BaseSize);
                LoadAlign(sVreg1, sUb + mRound * s2BaseSize / 4 + i * s2BaseSize);
                LoadAlign(sVreg2, sUb + mRound * s2BaseSize / 2 + i * s2BaseSize);
                LoadAlign(sVreg3, sUb + mRound * s2BaseSize / 2 + mRound * s2BaseSize / 4 + i * s2BaseSize);

                Muls(sVreg0, sVreg0, scaleValue, maskReg32TailN);
                Muls(sVreg1, sVreg1, scaleValue, maskReg32TailN);
                Muls(sVreg2, sVreg2, scaleValue, maskReg32TailN);
                Muls(sVreg3, sVreg3, scaleValue, maskReg32TailN);

                // FusedExpSub(pVreg0, sVreg0, hmVreg, maskReg16TailN);
                // FusedExpSub(pVreg1, sVreg1, hmVreg, maskReg16TailN);
                // FusedExpSub(pVreg2, sVreg2, hmVreg, maskReg16TailN);
                // FusedExpSub(pVreg3, sVreg3, hmVreg, maskReg16TailN);

                Sub(sVreg0, sVreg0, hmVreg, maskReg32TailN);
                Sub(sVreg1, sVreg1, hmVreg, maskReg32TailN);
                Sub(sVreg2, sVreg2, hmVreg, maskReg32TailN);
                Sub(sVreg3, sVreg3, hmVreg, maskReg32TailN);

                Exp(pVreg0, sVreg0, maskReg32TailN);
                Exp(pVreg1, sVreg1, maskReg32TailN);
                Exp(pVreg2, sVreg2, maskReg32TailN);
                Exp(pVreg3, sVreg3, maskReg32TailN);

                Cast<ElementOutput, float, castTraitZeroROUND>(pVreg160, pVreg0, maskReg32All);
                Cast<ElementOutput, float, castTraitZeroROUND>(pVreg162, pVreg2, maskReg32All);
                DeInterleave(deInterleaveVreg0, deInterleaveVreg1, pVreg160, pVreg162);
                Cast<ElementOutput, float, castTraitZeroROUND>(pVreg161, pVreg1, maskReg32All);
                Cast<ElementOutput, float, castTraitZeroROUND>(pVreg163, pVreg3, maskReg32All);
                DeInterleave(deInterleaveVreg2, deInterleaveVreg3, pVreg161, pVreg163);

                StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    pUb0, deInterleaveVreg0, blockStride, 1, maskReg16VL128);
                if constexpr (mTileNum > MAligendTileNum::Zero) {
                    Add(sumVreg0, sumVreg0, pVreg0, maskReg32TailN);
                }
                if constexpr (mTileNum > MAligendTileNum::One) {
                    Add(sumVreg1, sumVreg1, pVreg1, maskReg32TailN);
                }
                StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    pUb1, deInterleaveVreg2, blockStride, 1, maskReg16VL128);
                if constexpr (mTileNum > MAligendTileNum::Two) {
                    Add(sumVreg2, sumVreg2, pVreg2, maskReg32TailN);
                }
            }
        }
        Add(sumVreg0, sumVreg0, sumVreg2, maskReg32All);
        Add(sumVreg1, sumVreg1, sumVreg3, maskReg32All);
        Add(sumVreg0, sumVreg0, sumVreg1, maskReg32All);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(llUb, sumVreg0, maskReg32All);
    }

    __simd_vf__ inline void SetGlobalMaxAndGlobalSum(__ubuf__ float *lmUb, __ubuf__ float *gmUb, __ubuf__ float *llUb,
                                      __ubuf__ float *glUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> lmVreg;
        RegTensor<float> llVreg;

        MaskReg maskReg32All = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        LoadAlign(lmVreg, lmUb);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUb, lmVreg, maskReg32All);

        LoadAlign(llVreg, llUb);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(glUb, llVreg, maskReg32All);
    }

    __simd_vf__ inline void UpdateDeltaMaxGlobalMaxAndGlobalSum(__ubuf__ float *lmUb, __ubuf__ float *gmUb, __ubuf__ float *llUb,
                                              __ubuf__ float *glUb, __ubuf__ float *dmUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> hmVreg;
        RegTensor<float> gmVreg;
        RegTensor<float> dmVreg;

        RegTensor<float> llVreg;
        RegTensor<float> glVreg;

        MaskReg maskReg32All = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        LoadAlign(hmVreg, lmUb);
        LoadAlign(gmVreg, gmUb);
        FusedExpSub(dmVreg, gmVreg, hmVreg, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dmUb, dmVreg, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUb, hmVreg, maskReg32TailN);

        LoadAlign(llVreg, llUb);
        LoadAlign(glVreg, glUb);
        Mul(glVreg, glVreg, dmVreg, maskReg32All);
        Add(glVreg, glVreg, llVreg, maskReg32All);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(glUb, glVreg, maskReg32All);
    }
};
}

#endif