/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_BF16_S_TRANS_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_BF16_S_TRANS_HPP

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
    Gemm::GemmType<bfloat16_t, LayoutS_>>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = bfloat16_t;

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
    // static constexpr half MIN_VALUE = -65504.0f;
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
        scaleValue = AscendC::ToBfloat16(scaleValue_);
        MIN_VALUE = AscendC::ToBfloat16(-3.389531390315715675e+38);

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
        lmUbTensor = resource.ubBuf.template GetBufferByByte<bfloat16_t>(uBufTileHelper.lmStartOffset);
        llUbTensor = resource.ubBuf.template GetBufferByByte<bfloat16_t>(uBufTileHelper.llStartOffset);
        lmUbFloatTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.lmStartOffset);
        llUbFloatTensor = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.llStartOffset);
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
        uint32_t tailNOdd = tailN / 2 ;
        uint32_t tailNEven = tailNOdd + tailN % 2;
        int32_t mLoops = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;
        uint16_t s2BaseSize = 64;
        uint32_t mAligendTile = mRound / 4;
        uint32_t mFirstTile = m % mAligendTile;
        uint32_t mAligned16TileNum = m / mAligendTile;
        if (mAligned16TileNum == 4) {
            mFirstTile = mAligendTile;
        }
        __ubuf__ bfloat16_t *pUbAddr = (__ubuf__ bfloat16_t *)lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ bfloat16_t *sUbAddr = (__ubuf__ bfloat16_t *)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float *gmUbAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *gmUbStartAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *glUbAddr = (__ubuf__ float *)glUbTensor.GetPhyAddr();
        __ubuf__ float *lmUbAddr = (__ubuf__ float*) lmUbFloatTensor.GetPhyAddr();
        __ubuf__ float *llUbAddr = (__ubuf__ float*) llUbFloatTensor.GetPhyAddr();
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
                                                       s2BaseSize, tailN, tailNOdd, tailNEven, blockStride, scaleValue);
        } else if (mAligned16TileNum == 1) {
            ComputePAndLocalSum<MAligendTileNum::One>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                      s2BaseSize, tailN, tailNOdd, tailNEven, blockStride, scaleValue);
        } else if (mAligned16TileNum == 2) {
            ComputePAndLocalSum<MAligendTileNum::Two>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                      s2BaseSize, tailN, tailNOdd, tailNEven, blockStride, scaleValue);
        } else if (mAligned16TileNum == 3) {
            ComputePAndLocalSum<MAligendTileNum::Three>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                        s2BaseSize, tailN, tailNOdd, tailNEven, blockStride,
                                                        scaleValue);
        } else {
            ComputePAndLocalSum<MAligendTileNum::Four>(sUbAddr, pUbAddr, lmUbAddr, llUbAddr, mRound, m, mFirstTile,
                                                       s2BaseSize, tailN, tailNOdd, tailNEven, blockStride, scaleValue);
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
    bfloat16_t scaleValue;
    AscendC::LocalTensor<bfloat16_t> lsUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<bfloat16_t> lpUbTensor[UB_S_P_BUF_STAGES];
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor[UB_DM_BUF_MAX_STAGES];
    AscendC::LocalTensor<bfloat16_t> lmUbTensor;
    AscendC::LocalTensor<bfloat16_t> llUbTensor;
    AscendC::LocalTensor<float> lmUbFloatTensor;
    AscendC::LocalTensor<float> llUbFloatTensor;
    uint32_t subBlockIdx_;
    bfloat16_t MIN_VALUE;

    enum class MAligendTileNum {
        Zero = 0,
        One = 1,
        Two = 2,
        Three = 3,
        Four = 4
    };

    __simd_vf__ inline void ComputeLocalMax(__ubuf__ bfloat16_t *sUb, __ubuf__ float *lmUb, uint16_t m,
                                            uint16_t s2BaseSize, uint32_t tailN, bfloat16_t scaleValue)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<bfloat16_t> sVreg0;
        RegTensor<bfloat16_t> sVreg1;
        RegTensor<bfloat16_t> sVreg2;
        RegTensor<bfloat16_t> sVreg3;
        RegTensor<bfloat16_t> sVreg4;

        RegTensor<bfloat16_t> maxVreg0;
        RegTensor<bfloat16_t> maxVreg1;
        RegTensor<bfloat16_t> maxVreg2;
        RegTensor<bfloat16_t> maxVreg3;

        RegTensor<bfloat16_t> scaleValueVreg;

        RegTensor<float> maxVregFloat;

        MaskReg maskReg16TailN = UpdateMask<bfloat16_t>(tailN);
        MaskReg maskReg16All = CreateMask<bfloat16_t, MaskPattern::ALL>();
        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitOneUNKNOWN = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        __ubuf__ bfloat16_t *sUb0 = sUb;
        __ubuf__ bfloat16_t *sUb1 = sUb0 + s2BaseSize;
        __ubuf__ bfloat16_t *sUb2 = sUb0 + s2BaseSize * 2;
        __ubuf__ bfloat16_t *sUb3 = sUb0 + s2BaseSize * 3;
        __ubuf__ bfloat16_t *sUb4 = sUb0 + s2BaseSize * (m / 4) * 4;

        Duplicate(maxVreg0, MIN_VALUE);
        Duplicate(maxVreg1, MIN_VALUE);
        Duplicate(maxVreg2, MIN_VALUE);
        Duplicate(maxVreg3, MIN_VALUE);
        Duplicate(scaleValueVreg, scaleValue);

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

        Mul(maxVreg0, maxVreg0, scaleValueVreg, maskReg16TailN);
        Interleave(maxVreg1, maxVreg2, maxVreg0, maxVreg0);
        Cast<float, bfloat16_t, castTraitZeroUNKNOWN>(maxVregFloat, maxVreg1, maskReg16All);

        StoreAlign<float, StoreDist::DIST_NORM_B32>(lmUb, maxVregFloat, maskReg32TailN);
    }

    __simd_vf__ inline void UpdateHistoricalMax(__ubuf__ float *lmUb, __ubuf__ float *gmUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> lmVreg;
        RegTensor<float> gmVreg;
        RegTensor<float> hmVreg;

        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        constexpr static CastTrait castTraitZeroROUND = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        LoadAlign(lmVreg, lmUb);
        LoadAlign(gmVreg, gmUb);
        Max(hmVreg, lmVreg, gmVreg, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lmUb, hmVreg, maskReg32TailN);
    }

    template <MAligendTileNum mTileNum>
    __simd_vf__ inline void
    ComputePAndLocalSum(__ubuf__ bfloat16_t *sUb, __ubuf__ bfloat16_t *pUb, __ubuf__ float *lmUb, __ubuf__ float *llUb,
                        uint16_t mRound, uint16_t m, uint32_t mFirstTile, uint16_t s2BaseSize, uint32_t tailN,
                        uint32_t tailNOdd, uint32_t tailNEven, uint32_t blockStride, bfloat16_t scaleValue)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<bfloat16_t> sVreg0;
        RegTensor<bfloat16_t> sVreg1;
        RegTensor<bfloat16_t> sVreg2;
        RegTensor<bfloat16_t> sVreg3;

        RegTensor<bfloat16_t> scaleValueVreg;

        RegTensor<bfloat16_t> pVreg0;
        RegTensor<bfloat16_t> pVreg1;
        RegTensor<bfloat16_t> pVreg2;
        RegTensor<bfloat16_t> pVreg3;

        RegTensor<float> pVregFloat0;
        RegTensor<float> pVregFloat1;
        RegTensor<float> pVregFloat2;
        RegTensor<float> pVregFloat3;
        RegTensor<float> pVregFloat4;
        RegTensor<float> pVregFloat5;
        RegTensor<float> pVregFloat6;
        RegTensor<float> pVregFloat7;

        RegTensor<bfloat16_t> pVregBfloat160;
        RegTensor<bfloat16_t> pVregBfloat161;
        RegTensor<bfloat16_t> pVregBfloat162;
        RegTensor<bfloat16_t> pVregBfloat163;
        RegTensor<bfloat16_t> pVregBfloat164;
        RegTensor<bfloat16_t> pVregBfloat165;
        RegTensor<bfloat16_t> pVregBfloat166;
        RegTensor<bfloat16_t> pVregBfloat167;

        RegTensor<float> hmVreg;

        RegTensor<float> sumVreg0;
        RegTensor<float> sumVreg1;
        RegTensor<float> sumVreg2;
        RegTensor<float> sumVreg3;

        Duplicate(sumVreg0, 0);
        Duplicate(sumVreg1, 0);
        Duplicate(sumVreg2, 0);
        Duplicate(sumVreg3, 0);
        Duplicate(scaleValueVreg, scaleValue);

        MaskReg maskReg16TailN = UpdateMask<bfloat16_t>(tailN);
        MaskReg maskReg16VL64 = CreateMask<bfloat16_t, MaskPattern::VL64>();
        MaskReg maskReg16All = CreateMask<bfloat16_t, MaskPattern::ALL>();
        MaskReg maskReg32TailNOdd = UpdateMask<float>(tailNOdd);
        MaskReg maskReg32TailNEven = UpdateMask<float>(tailNEven);
        MaskReg maskReg32All = CreateMask<float, MaskPattern::ALL>();

        constexpr static CastTrait castTraitZeroUNKNOWN = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        constexpr static CastTrait castTraitOneUNKNOWN = {
            RegLayout::ONE,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        constexpr static CastTrait castTraitZeroROUND = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        constexpr static CastTrait castTraitOneROUND = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };

        __ubuf__ bfloat16_t *pUb0 = pUb;
        __ubuf__ bfloat16_t *pUb1 = pUb0 + (mRound * 4); // *pUb1 = pUb0 + (mRound / 4 * ELE_NUM_PER_DATABLOCK)
        __ubuf__ bfloat16_t *pUb2 = pUb0 + (mRound * 32 + 64); // *pUb2 = pUb0 + (mRound / 2 + 1) * 4 * ELE_NUM_PER_DATABLOCK
        __ubuf__ bfloat16_t *pUb3 = pUb2 + (mRound * 4); // *pUb3 = pUb2 + (mRound / 4 * ELE_NUM_PER_DATABLOCK)

        LoadAlign(hmVreg, lmUb);
        for (uint16_t i = 0; i < mFirstTile; i++) {
            LoadAlign(sVreg0, sUb + i * s2BaseSize);
            LoadAlign(sVreg1, sUb + mRound * s2BaseSize / 4 + i * s2BaseSize);
            LoadAlign(sVreg2, sUb + mRound * s2BaseSize / 2 + i * s2BaseSize);
            LoadAlign(sVreg3, sUb + mRound * s2BaseSize / 2 + mRound * s2BaseSize / 4 + i * s2BaseSize);

            Mul(sVreg0, sVreg0, scaleValueVreg, maskReg16TailN);
            Mul(sVreg1, sVreg1, scaleValueVreg, maskReg16TailN);
            Mul(sVreg2, sVreg2, scaleValueVreg, maskReg16TailN);
            Mul(sVreg3, sVreg3, scaleValueVreg, maskReg16TailN);

            Cast<float, bfloat16_t, castTraitZeroUNKNOWN>(pVregFloat0, sVreg0, maskReg16All);
            Cast<float, bfloat16_t, castTraitOneUNKNOWN>(pVregFloat1, sVreg0, maskReg16All);
            Cast<float, bfloat16_t, castTraitZeroUNKNOWN>(pVregFloat2, sVreg0, maskReg16All);
            Cast<float, bfloat16_t, castTraitOneUNKNOWN>(pVregFloat3, sVreg0, maskReg16All);
            Cast<float, bfloat16_t, castTraitZeroUNKNOWN>(pVregFloat4, sVreg0, maskReg16All);
            Cast<float, bfloat16_t, castTraitOneUNKNOWN>(pVregFloat5, sVreg0, maskReg16All);
            Cast<float, bfloat16_t, castTraitZeroUNKNOWN>(pVregFloat6, sVreg0, maskReg16All);
            Cast<float, bfloat16_t, castTraitOneUNKNOWN>(pVregFloat7, sVreg0, maskReg16All);

            FusedExpSub(pVregFloat0, pVregFloat0, hmVreg, maskReg32TailNEven);
            FusedExpSub(pVregFloat1, pVregFloat1, hmVreg, maskReg32TailNOdd);
            FusedExpSub(pVregFloat2, pVregFloat2, hmVreg, maskReg32TailNEven);
            FusedExpSub(pVregFloat3, pVregFloat3, hmVreg, maskReg32TailNOdd);
            FusedExpSub(pVregFloat4, pVregFloat4, hmVreg, maskReg32TailNEven);
            FusedExpSub(pVregFloat5, pVregFloat5, hmVreg, maskReg32TailNOdd);
            FusedExpSub(pVregFloat6, pVregFloat6, hmVreg, maskReg32TailNEven);
            FusedExpSub(pVregFloat7, pVregFloat7, hmVreg, maskReg32TailNOdd);

            if constexpr (mTileNum >= MAligendTileNum::Zero) {
                Add(sumVreg0, sumVreg0, pVregFloat0, maskReg32TailNEven);
                Add(sumVreg0, sumVreg0, pVregFloat1, maskReg32TailNOdd);
            }
            if constexpr (mTileNum >= MAligendTileNum::One) {
                Add(sumVreg1, sumVreg1, pVregFloat2, maskReg32TailNEven);
                Add(sumVreg1, sumVreg1, pVregFloat3, maskReg32TailNOdd);
            }
            if constexpr (mTileNum >= MAligendTileNum::Two) {
                Add(sumVreg2, sumVreg2, pVregFloat4, maskReg32TailNEven);
                Add(sumVreg2, sumVreg2, pVregFloat5, maskReg32TailNOdd);
            }
            if constexpr (mTileNum >= MAligendTileNum::Three) {
                Add(sumVreg3, sumVreg3, pVregFloat6, maskReg32TailNEven);
                Add(sumVreg3, sumVreg3, pVregFloat7, maskReg32TailNOdd);
            }

            Cast<bfloat16_t, float, castTraitZeroROUND>(pVregBfloat160, pVregFloat0, maskReg32All);
            Cast<bfloat16_t, float, castTraitOneROUND>(pVregBfloat161, pVregFloat1, maskReg32All);
            Cast<bfloat16_t, float, castTraitZeroROUND>(pVregBfloat162, pVregFloat2, maskReg32All);
            Cast<bfloat16_t, float, castTraitOneROUND>(pVregBfloat163, pVregFloat3, maskReg32All);
            Cast<bfloat16_t, float, castTraitZeroROUND>(pVregBfloat164, pVregFloat4, maskReg32All);
            Cast<bfloat16_t, float, castTraitOneROUND>(pVregBfloat165, pVregFloat5, maskReg32All);
            Cast<bfloat16_t, float, castTraitZeroROUND>(pVregBfloat166, pVregFloat6, maskReg32All);
            Cast<bfloat16_t, float, castTraitOneROUND>(pVregBfloat167, pVregFloat7, maskReg32All);

            Or((RegTensor<uint16_t> &)pVreg0, (RegTensor<uint16_t> &)pVregBfloat160,
               (RegTensor<uint16_t> &)pVregBfloat161, maskReg16All);
            Or((RegTensor<uint16_t> &)pVreg1, (RegTensor<uint16_t> &)pVregBfloat162,
               (RegTensor<uint16_t> &)pVregBfloat163, maskReg16All);
            Or((RegTensor<uint16_t> &)pVreg2, (RegTensor<uint16_t> &)pVregBfloat164,
               (RegTensor<uint16_t> &)pVregBfloat165, maskReg16All);
            Or((RegTensor<uint16_t> &)pVreg3, (RegTensor<uint16_t> &)pVregBfloat166,
               (RegTensor<uint16_t> &)pVregBfloat167, maskReg16All);

            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb0, pVreg0, blockStride, 1, maskReg16VL64);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb1, pVreg1, blockStride, 1, maskReg16VL64);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb2, pVreg2, blockStride, 1, maskReg16VL64);
            StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                pUb3, pVreg3, blockStride, 1, maskReg16VL64);
        }
        if constexpr (mTileNum < MAligendTileNum::Four) {
            for (uint16_t i = mFirstTile; i < mRound / 4; i++) {
                LoadAlign(sVreg0, sUb + i * s2BaseSize);
                LoadAlign(sVreg1, sUb + mRound * s2BaseSize / 4 + i * s2BaseSize);
                LoadAlign(sVreg2, sUb + mRound * s2BaseSize / 2 + i * s2BaseSize);
                LoadAlign(sVreg3, sUb + mRound * s2BaseSize / 2 + mRound * s2BaseSize / 4 + i * s2BaseSize);

                Mul(sVreg0, sVreg0, scaleValueVreg, maskReg16TailN);
                Mul(sVreg1, sVreg1, scaleValueVreg, maskReg16TailN);
                Mul(sVreg2, sVreg2, scaleValueVreg, maskReg16TailN);
                Mul(sVreg3, sVreg3, scaleValueVreg, maskReg16TailN);

                Cast<float, bfloat16_t, castTraitZeroUNKNOWN>(pVregFloat0, sVreg0, maskReg16All);
                Cast<float, bfloat16_t, castTraitOneUNKNOWN>(pVregFloat1, sVreg0, maskReg16All);
                Cast<float, bfloat16_t, castTraitZeroUNKNOWN>(pVregFloat2, sVreg0, maskReg16All);
                Cast<float, bfloat16_t, castTraitOneUNKNOWN>(pVregFloat3, sVreg0, maskReg16All);
                Cast<float, bfloat16_t, castTraitZeroUNKNOWN>(pVregFloat4, sVreg0, maskReg16All);
                Cast<float, bfloat16_t, castTraitOneUNKNOWN>(pVregFloat5, sVreg0, maskReg16All);
                Cast<float, bfloat16_t, castTraitZeroUNKNOWN>(pVregFloat6, sVreg0, maskReg16All);
                Cast<float, bfloat16_t, castTraitOneUNKNOWN>(pVregFloat7, sVreg0, maskReg16All);

                FusedExpSub(pVregFloat0, pVregFloat0, hmVreg, maskReg32TailNEven);
                FusedExpSub(pVregFloat1, pVregFloat1, hmVreg, maskReg32TailNOdd);
                FusedExpSub(pVregFloat2, pVregFloat2, hmVreg, maskReg32TailNEven);
                FusedExpSub(pVregFloat3, pVregFloat3, hmVreg, maskReg32TailNOdd);
                FusedExpSub(pVregFloat4, pVregFloat4, hmVreg, maskReg32TailNEven);
                FusedExpSub(pVregFloat5, pVregFloat5, hmVreg, maskReg32TailNOdd);
                FusedExpSub(pVregFloat6, pVregFloat6, hmVreg, maskReg32TailNEven);
                FusedExpSub(pVregFloat7, pVregFloat7, hmVreg, maskReg32TailNOdd);

                if constexpr (mTileNum > MAligendTileNum::Zero) {
                    Add(sumVreg0, sumVreg0, pVregFloat0, maskReg32TailNEven);
                    Add(sumVreg0, sumVreg0, pVregFloat1, maskReg32TailNOdd);
                }
                if constexpr (mTileNum > MAligendTileNum::One) {
                    Add(sumVreg1, sumVreg1, pVregFloat2, maskReg32TailNEven);
                    Add(sumVreg1, sumVreg1, pVregFloat3, maskReg32TailNOdd);
                }
                if constexpr (mTileNum > MAligendTileNum::Two) {
                    Add(sumVreg2, sumVreg2, pVregFloat4, maskReg32TailNEven);
                    Add(sumVreg2, sumVreg2, pVregFloat5, maskReg32TailNOdd);
                }

                StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    pUb0, pVreg0, blockStride, 1, maskReg16VL64);
                StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    pUb1, pVreg1, blockStride, 1, maskReg16VL64);
                StoreAlign<bfloat16_t, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    pUb2, pVreg2, blockStride, 1, maskReg16VL64);
            }
        }
        Add(sumVreg0, sumVreg0, sumVreg2, maskReg16TailN);
        Add(sumVreg1, sumVreg1, sumVreg3, maskReg16TailN);
        Add(sumVreg0, sumVreg0, sumVreg1, maskReg16TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(llUb, sumVreg0, maskReg16TailN);
    }

    __simd_vf__ inline void SetGlobalMaxAndGlobalSum(__ubuf__ float *lmUb, __ubuf__ float *gmUb, __ubuf__ float *llUb,
                                                     __ubuf__ float *glUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> lmVreg;
        RegTensor<float> llVreg;

        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        LoadAlign(lmVreg, lmUb);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUb, lmVreg, maskReg32TailN);

        LoadAlign(llVreg, llUb);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(glUb, llVreg, maskReg32TailN);
    }

    __simd_vf__ inline void UpdateDeltaMaxGlobalMaxAndGlobalSum(__ubuf__ float *lmUb, __ubuf__ float *gmUb,
                                                                __ubuf__ float *llUb, __ubuf__ float *glUb,
                                                                __ubuf__ float *dmUb, uint32_t tailN)
    {
        using namespace AscendC::MicroAPI;

        RegTensor<float> hmVreg;
        RegTensor<float> gmVreg;
        RegTensor<float> dmVreg;

        RegTensor<float> llVreg;
        RegTensor<float> glVreg;

        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        LoadAlign(hmVreg, lmUb);
        LoadAlign(gmVreg, gmUb);
        FusedExpSub(dmVreg, gmVreg, hmVreg, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(dmUb, dmVreg, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUb, hmVreg, maskReg32TailN);

        LoadAlign(llVreg, llUb);
        LoadAlign(glVreg, glUb);
        Mul(glVreg, glVreg, dmVreg, maskReg32TailN);
        Add(glVreg, glVreg, llVreg, maskReg32TailN);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(glUb, glVreg, maskReg32TailN);
    }
};
}

#endif