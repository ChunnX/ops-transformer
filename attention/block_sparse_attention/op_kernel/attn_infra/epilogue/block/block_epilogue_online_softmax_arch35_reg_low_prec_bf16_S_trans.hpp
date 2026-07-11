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
        __ubuf__ bfloat16_t *pUb1 = pUb + (mRound * 4);  // *pUb1 = pUb + (mRound / 4 * 16)
        __ubuf__ bfloat16_t *pUb2 = pUb + (mRound * 32); // *pUb2 = pUb + (mRound / 2 * 64)
        __ubuf__ bfloat16_t *pUb3 = pUb + (mRound * 36); // *pUb3 = pUb + (mRound / 2 * 64 + mRound / 4 * 16)
        LoadAlign<float, LoadDist::DIST_BRC_B32>(hmVreg, lmUb);
        for (uint16_t i = 0; i < mFirstTile; i++) {
            LoadAlign(sVreg0, sUb + i * s2BaseSize);
            LoadAlign(sVreg1, sUb + mRound * s2BaseSize / 4 + i * s2BaseSize);
            LoadAlign(sVreg2, sUb + mRound * s2BaseSize / 2 + i * s2BaseSize);
            LoadAlign(sVreg3, sUb + mRound * s2BaseSize / 2 + mRound * s2BaseSize / 4 + i * s2BaseSize);

            Muls(sVreg0, sVreg0, scaleValue, maskReg16TailN);
            Muls(sVreg1, sVreg1, scaleValue, maskReg16TailN);
            Muls(sVreg2, sVreg2, scaleValue, maskReg16TailN);
            Muls(sVreg3, sVreg3, scaleValue, maskReg16TailN);

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

                Muls(sVreg0, sVreg0, scaleValue, maskReg16TailN);
                Muls(sVreg1, sVreg1, scaleValue, maskReg16TailN);
                Muls(sVreg2, sVreg2, scaleValue, maskReg16TailN);
                Muls(sVreg3, sVreg3, scaleValue, maskReg16TailN);

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

        RegTensor<bfloat16_t> lmVreg;
        RegTensor<bfloat16_t> llVreg;

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

    // template <typename ElementS, typename ElementP, bool isUpdate, MAligendTileNum mTileNum>
    // __simd_vf__ inline void ComputeScaleAndMaxDn(__ubuf__ half *sUb, __ubuf__ half *lmUb, __ubuf__ float *gmUb, __ubuf__ float *gmUbStart, __ubuf__ ElementP *pUb,
    //     __ubuf__ ElementS *expSumUb, uint16_t mRound, uint16_t m, uint32_t tailN, uint32_t mFirstTile, half scaleValue, uint16_t s2BaseSize, 
    //     uint32_t blockStride, uint32_t repeatStride, __ubuf__ float *dmUb, __ubuf__ ElementS *glUb)
    // {
    //     using namespace AscendC::MicroAPI;
    //     RegTensor<half> src0Vreg;
    //     RegTensor<half> src1Vreg;
    //     RegTensor<half> src2Vreg;
    //     RegTensor<half> src3Vreg;
    //     RegTensor<half> src4Vreg;

    //     RegTensor<half> exp0Vreg16;
    //     RegTensor<half> exp1Vreg16;
    //     RegTensor<half> exp2Vreg16;
    //     RegTensor<half> exp3Vreg16;
        
    //     RegTensor<half> max0Vreg;
    //     RegTensor<half> max1Vreg;
    //     RegTensor<half> max2Vreg;
    //     RegTensor<half> max3Vreg;

    //     RegTensor<float> dmVreg;
    //     RegTensor<float> updateExpSumVreg;

    //     RegTensor<half> sum0Vreg;
    //     RegTensor<half> sum1Vreg;
    //     RegTensor<half> sum2Vreg;
    //     RegTensor<half> sum3Vreg;
    //     RegTensor<float> glVreg;
    //     RegTensor<half> lmVreg;
    //     // RegTensor<ElementP> exp0Vreg16;
    //     // RegTensor<ElementP> exp1Vreg16;
    //     // RegTensor<ElementP> exp2Vreg16;
    //     // RegTensor<ElementP> exp3Vreg16;
    //     RegTensor<ElementP> deInterleave0Vreg;
    //     RegTensor<ElementP> deInterleave1Vreg;
    //     RegTensor<ElementP> deInterleave2Vreg;
    //     RegTensor<ElementP> deInterleave3Vreg;

    //     RegTensor<float> interleave0Vreg;
    //     RegTensor<float> interleave1Vreg;
    //     RegTensor<float> gmVreg;

    //     constexpr static CastTrait castTraitZeroROUND = {
    //         RegLayout::ZERO,
    //         SatMode::SAT,
    //         MaskMergeMode::ZEROING,
    //         AscendC::RoundMode::CAST_ROUND,
    //     };

    //     constexpr static CastTrait castTraitZeroUNKNOWN = {
    //         RegLayout::ZERO,
    //         SatMode::UNKNOWN,
    //         MaskMergeMode::ZEROING,
    //         AscendC::RoundMode::UNKNOWN,
    //     }

    //     MaskReg preg16Full = CreateMask<half, MaskPattern::ALL>();
    //     MaskReg preg32Full = CreateMask<float, MaskPattern::ALL>();
    //     MaskReg preg16TailN = UpdateMask<half>(tailN);
    //     MaskReg preg32TailN = UpdateMask<float>(tailN);
    //     uint32_t twoVecLenCombined = 128;
    //     MaskReg pregFp16VL128 = UpdateMask<uint16_t>(twoVecLenCombined);
    //     MaskReg pregFp8VL128 = UpdateMask<uint8_t>(twoVecLenCombined);
    //     __ubuf__ ElementP *p0Ub = pUb;
    //     __ubuf__ ElementP *p1Ub = pUb + (mRound * 4);

    //     __ubuf__ half *srcUb0 = sUb;
    //     __ubuf__ half *srcUb1 = srcUb0 + s2BaseSize;
    //     __ubuf__ half *srcUb2 = srcUb0 + s2BaseSize * 2;
    //     __ubuf__ half *srcUb3 = srcUb0 + s2BaseSize * 3;
    //     __ubuf__ half *srcUb4 = srcUb0 + s2BaseSize * (m / 4) * 4;
    //     Duplicate(max0Vreg, MIN_VALUE);
    //     Duplicate(max1Vreg, MIN_VALUE);
    //     Duplicate(max2Vreg, MIN_VALUE);
    //     Duplicate(max3Vreg, MIN_VALUE);
    //     for (int32_t i = 0; i < int32_t(m / 4); i++) {
    //         LoadAlign(src0Vreg, srcUb0 + i * s2BaseSize * 4);
    //         LoadAlign(src1Vreg, srcUb1 + i * s2BaseSize * 4);
    //         LoadAlign(src2Vreg, srcUb2 + i * s2BaseSize * 4);
    //         LoadAlign(src3Vreg, srcUb3 + i * s2BaseSize * 4);
    //         Max(max0Vreg, max0Vreg, src0Vreg, preg16TailN);
    //         Max(max1Vreg, max1Vreg, src1Vreg, preg16TailN);
    //         Max(max2Vreg, max2Vreg, src2Vreg, preg16TailN);
    //         Max(max3Vreg, max3Vreg, src3Vreg, preg16TailN);
    //     }
    //     Max(max0Vreg, max0Vreg, max2Vreg, preg16TailN);
    //     Max(max1Vreg, max1Vreg, max3Vreg, preg16TailN);
    //     Max(max0Vreg, max0Vreg, max1Vreg, preg16TailN);
    //     for (int32_t i = 0; i < int32_t(m % 4); i++) {
    //         LoadAlign(src4Vreg, srcUb4 + i * s2BaseSize);
    //         Max(max0Vreg, max0Vreg, src4Vreg, preg16TailN);
    //     }
    //     Muls(max0Vreg, max0Vreg, scaleValue, preg16TailN);

    //     StoreAlign<half, StoreDist::DIST_NORM_B16>(lmUb, max0Vreg, preg16Full);

    //     if constexpr (isUpdate) {
    //         // LoadAlign(gmVreg, gmUb);
    //         // Cast<half, float, castTraitZeroROUND>(lmVreg, gmVreg, pVreg, preg32Full);
    //         // DeInterleave(deInterleave0Vreg, deInterleave1Vreg, lmVreg, lmVreg);
    //         // Max(max0Vreg, max0Vreg, deInterleave0Vreg, preg16TailN);









    //         LoadAlign(glVreg, glUb);
    //         Max(max0Vreg, max0Vreg, max1Vreg, preg16TailN);
    //         FusedExpSub(dmVreg, max1Vreg, max0Vreg, preg16TailN);
    //         Mul(updateExpSumVreg, dmVreg, glVreg, preg16Full);
    //         StoreAlign<float, StoreDist::DIST_NORM_B32>(dmUb, dmVreg, preg16TailN);
    //         StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUbStart, max0Vreg, preg16TailN);
    //     } else {
    //         Interleave(interleave0Vreg, interleave0Vre1, max0Vreg, max0Vreg);
    //         Cast<float, half, castTraitZeroUNKNOWN>(gmVreg, interleave0Vreg, preg16Full);
    //         StoreAlign<float, StoreDist::DIST_NORM_B32>(gmUb, gmVreg, preg32Full);
    //     }

    //     // StoreAlign<half, AscendC::MicroAPI::StoreDist::DIST_NORM_B16>((__ubuf__ half *&)newMaxUb, max0Vreg, preg16Full);

    //     // Duplicate<half, AscendC::MicroAPI::MaskMergeMode::ZEROING, half>(sum0Vreg, 0, preg16Full);
    //     // Duplicate<half, AscendC::MicroAPI::MaskMergeMode::ZEROING, half>(sum1Vreg, 0, preg16Full);
    //     // Duplicate<half, AscendC::MicroAPI::MaskMergeMode::ZEROING, half>(sum2Vreg, 0, preg16Full);
    //     // Duplicate<half, AscendC::MicroAPI::MaskMergeMode::ZEROING, half>(sum3Vreg, 0, preg16Full);
    //     Duplicate(sum0Vreg, 0);
    //     Duplicate(sum1Vreg, 0);
    //     Duplicate(sum2Vreg, 0);
    //     Duplicate(sum3Vreg, 0);

    //     for (int32_t i = 0; i < mFirstTile; i++) {
    //         LoadAlign(src0Vreg, sUb + i * s2BaseSize);
    //         LoadAlign(src1Vreg, sUb + mRound * s2BaseSize / 4 + i * s2BaseSize);
    //         LoadAlign(src2Vreg, sUb + mRound * s2BaseSize / 2 + i * s2BaseSize);
    //         LoadAlign(src3Vreg, sUb + mRound * s2BaseSize / 2 + mRound * s2BaseSize / 4 + i * s2BaseSize);

    //         Muls(src0Vreg, src0Vreg, scaleValue, preg16TailN);
    //         Muls(src1Vreg, src1Vreg, scaleValue, preg16TailN);
    //         Muls(src2Vreg, src2Vreg, scaleValue, preg16TailN);
    //         Muls(src3Vreg, src3Vreg, scaleValue, preg16TailN);

    //         FusedExpSub(exp0Vreg16, src0Vreg, max0Vreg, preg16TailN);
    //         FusedExpSub(exp1Vreg16, src1Vreg, max0Vreg, preg16TailN);
    //         FusedExpSub(exp2Vreg16, src2Vreg, max0Vreg, preg16TailN);
    //         FusedExpSub(exp3Vreg16, src3Vreg, max0Vreg, preg16TailN);

    //         // Cast<ElementP, float, castTraitZeroROUND>(exp0Vreg16, exp0Vreg32, preg16Full);
    //         // Cast<ElementP, float, castTraitZeroROUND>(exp2Vreg16, exp2Vreg32, preg16Full);
    //         // DeInterleave(deInterleave0Vreg, deInterleave1Vreg, exp0Vreg16, exp2Vreg16);
    //         // Cast<ElementP, float, castTraitZeroROUND>(exp1Vreg16, exp1Vreg32, preg16Full);
    //         // Cast<ElementP, float, castTraitZeroROUND>(exp3Vreg16, exp3Vreg32, preg16Full);
    //         // DeInterleave(deInterleave2Vreg, deInterleave3Vreg, exp1Vreg16, exp3Vreg16);
    //         StoreAlign<ElementP, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
    //                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
    //             ((__ubuf__ ElementP *&)p0Ub), deInterleave0Vreg, blockStride, 1, pregFp16VL128);

    //         if constexpr (mTileNum >= MAligendTileNum::Zero) {
    //             Add(sum0Vreg, exp0Vreg16, sum0Vreg, preg16TailN);
    //         }
    //         if constexpr (mTileNum >= MAligendTileNum::Two) {
    //             Add(sum2Vreg, exp2Vreg16, sum2Vreg, preg16TailN);
    //         }
    //         StoreAlign<ElementP, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
    //                    AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
    //             ((__ubuf__ ElementP *&)p1Ub), deInterleave2Vreg, blockStride, 1, pregFp16VL128);
    //         if constexpr (mTileNum >= MAligendTileNum::One) {
    //             Add(sum1Vreg, exp1Vreg16, sum1Vreg, preg16TailN);
    //         }
    //         if constexpr (mTileNum >= MAligendTileNum::Three) {
    //             Add(sum3Vreg, exp3Vreg16, sum3Vreg, preg16TailN);
    //         }
    //     }
    //     if constexpr (mTileNum < MAligendTileNum::Four) {
    //         for (int32_t i = mFirstTile; i < int32_t(mRound / 4); i++) {
    //             LoadAlign(src0Vreg, sUb + i * s2BaseSize);
    //             LoadAlign(src1Vreg, sUb + mRound * s2BaseSize / 4 + i * s2BaseSize);
    //             LoadAlign(src2Vreg, sUb + mRound * s2BaseSize / 2 + i * s2BaseSize);
    //             LoadAlign(src3Vreg, sUb + mRound * s2BaseSize / 2 + mRound * s2BaseSize / 4 + i * s2BaseSize);

    //             Muls(src0Vreg, src0Vreg, scaleValue, preg16TailN);
    //             Muls(src1Vreg, src1Vreg, scaleValue, preg16TailN);
    //             Muls(src2Vreg, src2Vreg, scaleValue, preg16TailN);
    //             Muls(src3Vreg, src3Vreg, scaleValue, preg16TailN);

    //             FusedExpSub(exp0Vreg16, src0Vreg, max0Vreg, preg16TailN);
    //             FusedExpSub(exp1Vreg16, src1Vreg, max0Vreg, preg16TailN);
    //             FusedExpSub(exp2Vreg16, src2Vreg, max0Vreg, preg16TailN);
    //             FusedExpSub(exp3Vreg32, src3Vreg, max0Vreg, preg16TailN);

    //             // Cast<ElementP, float, castTraitZeroROUND>(exp0Vreg16, exp0Vreg32, preg16Full);
    //             // Cast<ElementP, float, castTraitZeroROUND>(exp2Vreg16, exp2Vreg32, preg16Full);
    //             // DeInterleave(deInterleave0Vreg, deInterleave1Vreg, exp0Vreg16, exp2Vreg16);
    //             // Cast<ElementP, float, castTraitZeroROUND>(exp1Vreg16, exp1Vreg32, preg16Full);
    //             // Cast<ElementP, float, castTraitZeroROUND>(exp3Vreg16, exp3Vreg32, preg16Full);
    //             // DeInterleave(deInterleave2Vreg, deInterleave3Vreg, exp1Vreg16, exp3Vreg16);
    //             StoreAlign<ElementP, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
    //                        AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
    //                 ((__ubuf__ ElementP *&)p0Ub), deInterleave0Vreg, blockStride, 1, pregFp16VL128);

    //             if constexpr (mTileNum > MAligendTileNum::Zero) {
    //                 Add(sum0Vreg, exp0Vreg32, sum0Vreg, preg16TailN);
    //             }
    //             if constexpr (mTileNum > MAligendTileNum::Two) {
    //                 Add(sum2Vreg, exp2Vreg32, sum2Vreg, preg16TailN);
    //             }
    //             StoreAlign<ElementP, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
    //                        AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
    //                 ((__ubuf__ ElementP *&)p1Ub), deInterleave2Vreg, blockStride, 1, pregFp16VL128);
    //             if constexpr (mTileNum > MAligendTileNum::One) {
    //                 Add(sum1Vreg, exp1Vreg32, sum1Vreg, preg16TailN);
    //             }
    //         }
    //     }
    //     Add(sum0Vreg, sum0Vreg, sum2Vreg, preg16Full);
    //     Add(sum1Vreg, sum1Vreg, sum3Vreg, preg16Full);
    //     Add(sum0Vreg, sum0Vreg, sum1Vreg, preg16Full);
    //     if constexpr (isUpdate) {
    //         Add(updateExpSumVreg, updateExpSumVreg, sum0Vreg, preg16Full);
    //         StoreAlign<float, StoreDist::DIST_NORM_B32>(glUb, updateExpSumVreg, preg16Full);
    //     }
    //     StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
    //         (__ubuf__ float *&)expSumUb, sum0Vreg, preg16Full);
    // }
};
}

#endif