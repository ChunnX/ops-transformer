/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_S_TRANS_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_S_TRANS_HPP

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
    Gemm::GemmType<half, LayoutS_>>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = half;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t BLOCK_SIZE_IN_BYTE = 32;
    // static constexpr uint32_t REPEAT_SIZE_IN_BYTE = 256;
    // static constexpr uint32_t FLOAT_BLOCK_SIZE = 8;
    // static constexpr uint32_t FLOAT_VECTOR_SIZE = 64;
    // static constexpr uint32_t HALF_VECTOR_SIZE = 128;
    // static constexpr uint32_t BLOCK_SIZE = 16;
    // static constexpr uint32_t UB_UINT8_VECTOR_SIZE = 1024;
    // static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    // static constexpr uint32_t VECTOR_SIZE = 128;
    // static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;
    // static constexpr uint32_t MAX_UB_P_ELEM_NUM = 8320;
    // static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t ELE_NUM_PER_C0 = 16;
    static constexpr uint32_t ELE_NUM_PER_C0_FP8 = 32;
    static constexpr uint32_t ELE_NUM_PER_DATABLOCK = 16;

    // static constexpr uint32_t REDUCE_UB_SIZE = 1024;
    // static constexpr uint32_t ROW_OPS_SPEC_MASK_32 = 32;
    // static constexpr uint32_t ROW_OPS_SPEC_MASK_8 = 8;
    // static constexpr uint32_t ROW_OPS_SPEC_MASK_4 = 4;
    // static constexpr uint32_t ROW_OPS_SPEC_MASK_2 = 2;
    // static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
    // static constexpr int64_t UB_FLOAT_LINE_SIZE = 64;

    // static constexpr uint32_t SPLIT_COL_IDX_2 = 2;
    // static constexpr uint32_t SPLIT_COL_IDX_3 = 3;
    static constexpr half MIN_VALUE = -65504.0f;
    // static constexpr uint32_t HALF_REP_SIZE = 128;
    // static constexpr uint32_t FLOAT_REP_SIZE = 64;
    // static constexpr uint32_t BLOCK_REP_SIZE = 8;
    // static constexpr uint32_t REPEAT_STRIDE = 1;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    // static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 128;
    // static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 64;
    // static constexpr uint32_t SM_VREG_SIZE = 256 / sizeof(ElementInput);
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
        int32_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;
        int32_t mLoops = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        // uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;
        uint16_t s2BaseSize = 64;

        uint32_t mAligendTile = mRound / 4;
        uint32_t mFirstTile = m % mAligendTile;
        uint32_t mAligned16TileNum = m / mAligendTile;
        if (mAligned16TileNum == 4) {
            mFirstTile = mAligendTile;
        }
        __ubuf__ half *pUbAddr = (__ubuf__ half *)lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ half *sUbAddr = (__ubuf__ half *)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float *gmUbAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *gmUbStartAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *glUbAddr = (__ubuf__ float *)glUbTensor.GetPhyAddr();
        __ubuf__ half *lmUbAddr = (__ubuf__ half *)lmUbTensor.GetPhyAddr();
        __ubuf__ half *lmUbStartAddr = (__ubuf__ half *)lmUbTensor.GetPhyAddr();
        __ubuf__ half *llUbAddr = (__ubuf__ half *)llUbTensor.GetPhyAddr();
        __ubuf__ float *dmUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId].GetPhyAddr();
        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        ComputeLocalMax(sUbAddr, lmUbAddr, m, s2BaseSize, tailN, scaleValue);
        if (!isFirstKvSTile) {
            UpdateHistoricalMax(lmUbAddr, gmUbAddr, tailN);
        }
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
        } else {
            UpdateDeltaMaxGlobalMaxAndGlobalSum(lmUbAddr, gmUbAddr, llUbAddr, glUbAddr, dmUbAddr, tailN);
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

    __simd_vf__ inline void ComputeLocalMax(__ubuf__ half *sUb, __ubuf__ half *lmUb, uint16_t m, uint16_t s2BaseSize,
                                            uint32_t tailN, half scaleValue)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<half> sVreg0;
        RegTensor<half> sVreg1;
        RegTensor<half> sVreg2;
        RegTensor<half> sVreg3;
        RegTensor<half> sVreg4;

        RegTensor<half> maxVreg0;
        RegTensor<half> maxVreg1;
        RegTensor<half> maxVreg2;
        RegTensor<half> maxVreg3;

        MaskReg maskReg16TailN = UpdateMask<half>(tailN);

        __ubuf__ half *sUb0 = sUb;
        __ubuf__ half *sUb1 = sUb0 + s2BaseSize;
        __ubuf__ half *sUb2 = sUb0 + s2BaseSize * 2;
        __ubuf__ half *sUb3 = sUb0 + s2BaseSize * 3;
        __ubuf__ half *sUb4 = sUb0 + s2BaseSize * (m / 4) * 4;

        Duplicate(maxVreg0, MIN_VALUE);
        Duplicate(maxVreg1, MIN_VALUE);
        Duplicate(maxVreg2, MIN_VALUE);
        Duplicate(maxVreg3, MIN_VALUE);

        for (uint16_t i = 0; i < m / 4; i++) {
            LoadAlign(sVreg0, sUb0 + i * s2BaseSize * 4);
            LoadAlign(sVreg1, sUb1 + i * s2BaseSize * 4);
            LoadAlign(sVreg2, sUb2 + i * s2BaseSize * 4);
            LoadAlign(sVreg3, sUb3 + i * s2BaseSize * 4);
            Max(maxVreg0, maxVreg0, sVreg0, maskReg16TailN);
            Max(maxVreg1, maxVreg1, sVreg1, maskReg16TailN);
            Max(maxVreg2, maxVreg2, sVreg2, maskReg16TailN);
            Max(maxVreg3, maxVreg3, sVreg3, maskReg16TailN);
        }

        Max(maxVreg0, maxVreg0, maxVreg2, maskReg16TailN);
        Max(maxVreg1, maxVreg1, maxVreg3, maskReg16TailN);
        Max(maxVreg0, maxVreg0, maxVreg1, maskReg16TailN);
        for (uint16_t i = 0; i < m % 4; i++) {
            LoadAlign(sVreg4, sUb4 + i * s2BaseSize);
            Max(maxVreg0, maxVreg0, sVreg4, maskReg16TailN);
        }
    
        Muls(maxVreg0, maxVreg0, scaleValue, maskReg16TailN);
        StoreAlign<half, StoreDist::DIST_NORM_B16>(lmUb, maxVreg0, maskReg16TailN);
    }

    __simd_vf__ inline void UpdateHistoricalMax(__ubuf__ half *lmUb, __ubuf__ float *gmUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<half> lmVreg;
        RegTensor<float> gmVreg;
        RegTensor<half> gmVregHalf;
        RegTensor<half> gmVregHalf0;
        RegTensor<half> gmVregHalf1;
        RegTensor<half> hmVreg;

        MaskReg maskReg16TailN = UpdateMask<half>(tailN);
        MaskReg maskReg32All = CreateMask<float, MaskPattern::ALL>();

        constexpr static CastTrait castTraitZeroROUND = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        LoadAlign(lmVreg, lmUb);
        LoadAlign(gmVreg, gmUb);
        Cast<half, float, castTraitZeroROUND>(gmVregHalf, gmVreg, maskReg32All);
        DeInterleave(gmVregHalf0, gmVregHalf1, gmVregHalf, gmVregHalf);
        Max(hmVreg, lmVreg, gmVregHalf0, maskReg16TailN);
        StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(lmUb, hmVreg, maskReg16TailN);
    }

    template <MAligendTileNum mTileNum>
    __simd_vf__ inline void ComputePAndLocalSum(__ubuf__ half *sUb, __ubuf__ ElementOutput *pUb, __ubuf__ half *lmUb,
                                                __ubuf__ half *llUb, uint16_t mRound, uint16_t m, uint32_t mFirstTile,
                                                uint16_t s2BaseSize, uint32_t tailN, uint32_t blockStride,
                                                half scaleValue)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<half> sVreg0;
        RegTensor<half> sVreg1;
        RegTensor<half> sVreg2;
        RegTensor<half> sVreg3;

        RegTensor<half> pVreg0;
        RegTensor<half> pVreg1;
        RegTensor<half> pVreg2;
        RegTensor<half> pVreg3;

        RegTensor<half> hmVreg;

        RegTensor<half> sumVreg0;
        RegTensor<half> sumVreg1;
        RegTensor<half> sumVreg2;
        RegTensor<half> sumVreg3;

        Duplicate(sumVreg0, 0);
        Duplicate(sumVreg1, 0);
        Duplicate(sumVreg2, 0);
        Duplicate(sumVreg3, 0);

        MaskReg maskReg16TailN = UpdateMask<half>(tailN);
        MaskReg maskReg16VL64 = CreateMask<half, MaskPattern::VL64>();

        __ubuf__ half *pUb0 = pUb;
        __ubuf__ half *pUb1 = pUb0 + (mRound * 4); // *pUb1 = pUb0 + (mRound / 4 * ELE_NUM_PER_DATABLOCK)
        __ubuf__ half *pUb2 = pUb0 + (mRound * 32 + 64); // *pUb2 = pUb0 + (mRound / 2 + 1) * 4 * ELE_NUM_PER_DATABLOCK
        __ubuf__ half *pUb3 = pUb2 + (mRound * 4); // *pUb3 = pUb2 + (mRound / 4 * ELE_NUM_PER_DATABLOCK)

        LoadAlign(hmVreg, lmUb);
        for (uint16_t i = 0; i < mFirstTile; i++) {
            LoadAlign(sVreg0, sUb + i * s2BaseSize);
            LoadAlign(sVreg1, sUb + mRound * s2BaseSize / 4 + i * s2BaseSize);
            LoadAlign(sVreg2, sUb + mRound * s2BaseSize / 2 + i * s2BaseSize);
            LoadAlign(sVreg3, sUb + mRound * s2BaseSize / 2 + mRound * s2BaseSize / 4 + i * s2BaseSize);

            Muls(sVreg0, sVreg0, scaleValue, maskReg16TailN);
            Muls(sVreg1, sVreg1, scaleValue, maskReg16TailN);
            Muls(sVreg2, sVreg2, scaleValue, maskReg16TailN);
            Muls(sVreg3, sVreg3, scaleValue, maskReg16TailN);

            Sub(sVreg0, sVreg0, hmVreg, maskReg16TailN);
            Sub(sVreg1, sVreg1, hmVreg, maskReg16TailN);
            Sub(sVreg2, sVreg2, hmVreg, maskReg16TailN);
            Sub(sVreg3, sVreg3, hmVreg, maskReg16TailN);

            Exp(pVreg0, sVreg0, maskReg16TailN);
            Exp(pVreg1, sVreg1, maskReg16TailN);
            Exp(pVreg2, sVreg2, maskReg16TailN);
            Exp(pVreg3, sVreg3, maskReg16TailN);

            if constexpr (mTileNum >= MAligendTileNum::Zero) {
                Add(sumVreg0, sumVreg0, pVreg0, maskReg16TailN);
            }
            if constexpr (mTileNum >= MAligendTileNum::One) {
                Add(sumVreg1, sumVreg1, pVreg1, maskReg16TailN);
            }
            if constexpr (mTileNum >= MAligendTileNum::Two) {
                Add(sumVreg2, sumVreg2, pVreg2, maskReg16TailN);
            }
            if constexpr (mTileNum >= MAligendTileNum::Three) {
                Add(sumVreg3, sumVreg3, pVreg3, maskReg16TailN);
            }

            StoreAlign<half, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb0, pVreg0, blockStride, 1, maskReg16VL64);
            StoreAlign<half, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb1, pVreg1, blockStride, 1, maskReg16VL64);
            StoreAlign<half, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb2, pVreg2, blockStride, 1, maskReg16VL64);
            StoreAlign<half, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb3, pVreg3, blockStride, 1, maskReg16VL64);
        }
        if constexpr (mTileNum < MAligendTileNum::Four) {
            for (uint16_t i = mFirstTile; i < mRound / 4; i++) {
                LoadAlign(sVreg0, sUb + i * s2BaseSize);
                LoadAlign(sVreg1, sUb + mRound * s2BaseSize / 4 + i * s2BaseSize);
                LoadAlign(sVreg2, sUb + mRound * s2BaseSize / 2 + i * s2BaseSize);
                LoadAlign(sVreg3, sUb + mRound * s2BaseSize / 2 + mRound * s2BaseSize / 4 + i * s2BaseSize);

                Muls(sVreg0, sVreg0, scaleValue, maskReg16TailN);
                Muls(sVreg1, sVreg1, scaleValue, maskReg16TailN);
                Muls(sVreg2, sVreg2, scaleValue, maskReg16TailN);
                Muls(sVreg3, sVreg3, scaleValue, maskReg16TailN);


                Sub(sVreg0, sVreg0, hmVreg, maskReg16TailN);
                Sub(sVreg1, sVreg1, hmVreg, maskReg16TailN);
                Sub(sVreg2, sVreg2, hmVreg, maskReg16TailN);
                Sub(sVreg3, sVreg3, hmVreg, maskReg16TailN);

                Exp(pVreg0, sVreg0, maskReg16TailN);
                Exp(pVreg1, sVreg1, maskReg16TailN);
                Exp(pVreg2, sVreg2, maskReg16TailN);
                Exp(pVreg3, sVreg3, maskReg16TailN);

                if constexpr (mTileNum > MAligendTileNum::Zero) {
                    Add(sumVreg0, sumVreg0, pVreg0, maskReg16TailN);
                }
                if constexpr (mTileNum > MAligendTileNum::One) {
                    Add(sumVreg1, sumVreg1, pVreg1, maskReg16TailN);
                }
                if constexpr (mTileNum > MAligendTileNum::Two) {
                    Add(sumVreg2, sumVreg2, pVreg2, maskReg16TailN);
                }

                StoreAlign<half, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    pUb0, pVreg0, blockStride, 1, maskReg16VL64);
                StoreAlign<half, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    pUb1, pVreg1, blockStride, 1, maskReg16VL64);
                StoreAlign<half, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    pUb2, pVreg2, blockStride, 1, maskReg16VL64);
            }
        }
        Add(sumVreg0, sumVreg0, sumVreg2, maskReg16TailN);
        Add(sumVreg1, sumVreg1, sumVreg3, maskReg16TailN);
        Add(sumVreg0, sumVreg0, sumVreg1, maskReg16TailN);
        StoreAlign<half, StoreDist::DIST_NORM_B16>(llUb, sumVreg0, maskReg16TailN);
    }

    __simd_vf__ inline void SetGlobalMaxAndGlobalSum(__ubuf__ half *lmUb, __ubuf__ float *gmUb, __ubuf__ half *llUb,
                                      __ubuf__ float *glUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<half> lmVreg;
        RegTensor<half> lmVreg0;
        RegTensor<half> lmVreg1;
        RegTensor<float> lmVregFloat;

        RegTensor<half> llVreg;
        RegTensor<half> llVreg0;
        RegTensor<half> llVreg1;
        RegTensor<float> llVregFloat;

        MaskReg maskReg16All = CreateMask<half, MaskPattern::ALL>();
        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        LoadAlign(lmVreg, lmUb);
        Interleave(lmVreg0, lmVreg1, lmVreg, lmVreg);
        Cast<float, half, castTraitZeroUNKNOWN>(lmVregFloat, lmVreg0, maskReg16All);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUb, lmVregFloat, maskReg32TailN);

        LoadAlign(llVreg, llUb);
        Interleave(llVreg0, llVreg1, llVreg, llVreg);
        Cast<float, half, castTraitZeroUNKNOWN>(llVregFloat, llVreg0, maskReg16All);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(glUb, llVregFloat, maskReg32TailN);
    }

    __simd_vf__ inline void UpdateDeltaMaxGlobalMaxAndGlobalSum(__ubuf__ half *lmUb, __ubuf__ float *gmUb, __ubuf__ half *llUb,
                                              __ubuf__ float *glUb, __ubuf__ float *dmUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<half> hmVreg;
        RegTensor<half> hmVreg0;
        RegTensor<half> hmVreg1;
        RegTensor<float> hmVregFloat;
        RegTensor<float> gmVreg;
        RegTensor<float> dmVreg;

        RegTensor<half> llVreg;
        RegTensor<half> llVreg0;
        RegTensor<half> llVreg1;
        RegTensor<float> llVregFloat;
        RegTensor<float> glVreg;

        MaskReg maskReg16All = CreateMask<half, MaskPattern::ALL>();
        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        LoadAlign(hmVreg, lmUb);
        LoadAlign(gmVreg, gmUb);
        Interleave(hmVreg0, hmVreg1, hmVreg, hmVreg);
        Cast<float, half, castTraitZeroUNKNOWN>(hmVregFloat, hmVreg0, maskReg16All);
        FusedExpSub(dmVreg, gmVreg, hmVregFloat, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dmUb, dmVreg, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUb, hmVregFloat, maskReg32TailN);

        LoadAlign(llVreg, llUb);
        LoadAlign(glVreg, glUb);
        Interleave(llVreg0, llVreg1, llVreg, llVreg);
        Cast<float, half, castTraitZeroUNKNOWN>(llVregFloat, llVreg0, maskReg16All);
        Mul(glVreg, glVreg, dmVreg, maskReg32TailN);
        Add(glVreg, glVreg, llVregFloat, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(glUb, glVreg, maskReg32TailN);
    }
};
}

#endif