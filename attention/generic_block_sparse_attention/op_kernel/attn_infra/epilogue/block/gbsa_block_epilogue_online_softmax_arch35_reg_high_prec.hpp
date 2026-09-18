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
    static constexpr uint32_t ELE_NUM_PER_DATABLOCK = 32 / sizeof(ElementOutput);
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
        uint32_t pNFractalNum = uBufTileHelper.kvBaseTilePerSubCore / ELE_NUM_PER_DATABLOCK;
        uint32_t pStageElemNum = uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore +
                                 (pNFractalNum - 1) * ELE_NUM_PER_DATABLOCK;
        for (uint32_t i = 0; i < UB_S_P_BUF_STAGES; ++i) {
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<float>(
                uBufTileHelper.sStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.kvBaseTilePerSubCore * sizeof(float) * i);
            lpUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(
                uBufTileHelper.pStartOffset + pStageElemNum * sizeof(ElementOutput) * i);
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
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / ELE_NUM_PER_C0 - m + 1;
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
        uint32_t mCopyOffset = RoundUp(actualBlockShape.m(), 2) / 2;
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
        uint32_t blockStride = mRound + 1;
        uint32_t tailN = (n - 1) % SM_VREG_SIZE + 1;

        __ubuf__ float *sUbAddr = (__ubuf__ float *)lsUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ ElementOutput *pUbAddr = (__ubuf__ ElementOutput *)lpUbTensor[ubSBufId].GetPhyAddr();
        __ubuf__ float *gmUbAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *glUbAddr = (__ubuf__ float *)glUbTensor.GetPhyAddr();
        __ubuf__ float *dmUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId].GetPhyAddr();

        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        uint32_t kvBaseTileRegStages = CeilDiv(n, SM_VREG_SIZE);
        if (kvBaseTileRegStages == 1) {
            if (isFirstKvSTile) {
                ComputeAndUpdateMax<true>(sUbAddr, gmUbAddr, dmUbAddr, m, tailN, scaleValue, nRound,
                                          tla::C<KvBaseTileRegSplitStagesHighPrec::ONE>{});
            } else {
                ComputeAndUpdateMax<false>(sUbAddr, gmUbAddr, dmUbAddr, m, tailN, scaleValue, nRound,
                                           tla::C<KvBaseTileRegSplitStagesHighPrec::ONE>{});
            }
        } else if (kvBaseTileRegStages == 2) {
            if (isFirstKvSTile) {
                ComputeAndUpdateMax<true>(sUbAddr, gmUbAddr, dmUbAddr, m, tailN, scaleValue, nRound,
                                          tla::C<KvBaseTileRegSplitStagesHighPrec::TWO>{});
            } else {
                ComputeAndUpdateMax<false>(sUbAddr, gmUbAddr, dmUbAddr, m, tailN, scaleValue, nRound,
                                           tla::C<KvBaseTileRegSplitStagesHighPrec::TWO>{});
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        if (kvBaseTileRegStages == 1) {
            if (isFirstKvSTile) {
                ComputePAndUpdateSum<true>(pUbAddr, sUbAddr, gmUbAddr, glUbAddr, dmUbAddr, m, tailN, blockStride,
                                           nRound, tla::C<KvBaseTileRegSplitStagesHighPrec::ONE>{});
            } else {
                ComputePAndUpdateSum<false>(pUbAddr, sUbAddr, gmUbAddr, glUbAddr, dmUbAddr, m, tailN, blockStride,
                                            nRound, tla::C<KvBaseTileRegSplitStagesHighPrec::ONE>{});
            }
        } else if (kvBaseTileRegStages == 2) {
            if (isFirstKvSTile) {
                ComputePAndUpdateSum<true>(pUbAddr, sUbAddr, gmUbAddr, glUbAddr, dmUbAddr, m, tailN, blockStride,
                                           nRound, tla::C<KvBaseTileRegSplitStagesHighPrec::TWO>{});
            } else {
                ComputePAndUpdateSum<false>(pUbAddr, sUbAddr, gmUbAddr, glUbAddr, dmUbAddr, m, tailN, blockStride,
                                            nRound, tla::C<KvBaseTileRegSplitStagesHighPrec::TWO>{});
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId], ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile =
            GetTile(l1PTensorTla, tla::MakeCoord(subBlockIdx_ * mCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    template <bool IsFirstKvSTile>
    __simd_vf__ inline void ComputeAndUpdateMax(__ubuf__ float *sUb, __ubuf__ float *gmUb, __ubuf__ float *dmUb,
                                                uint16_t m, uint32_t tailN, float scaleValue, uint16_t s2BaseSize,
                                                tla::C<KvBaseTileRegSplitStagesHighPrec::ONE>)
    {
        using namespace AscendC::Reg;
        RegTensor<float> minVreg;
        RegTensor<float> sVreg;
        RegTensor<float> lmVreg;
        RegTensor<float> gmVreg;
        RegTensor<float> hmVreg;
        RegTensor<float> dmVreg;
        UnalignReg gmUreg;
        UnalignReg dmUreg;
        MaskReg maskReg32All = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(sVreg, sUb + i * s2BaseSize);
            Muls(sVreg, sVreg, scaleValue, maskReg32All);
            Select(sVreg, sVreg, minVreg, maskReg32TailN);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sUb + i * s2BaseSize, sVreg, maskReg32TailN);
            ReduceMax(lmVreg, sVreg, maskReg32All);
            if constexpr (IsFirstKvSTile) {
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(gmUb, lmVreg, gmUreg, 1);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(gmVreg, gmUb);
                Max(hmVreg, lmVreg, gmVreg, maskReg32All);
                FusedExpSub(dmVreg, gmVreg, hmVreg, maskReg32All);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(gmUb, hmVreg, gmUreg, 1);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dmUb, dmVreg, dmUreg, 1);
            }
        }
        vstas(gmUreg, gmUb, 0, POST_UPDATE);
        if constexpr (!IsFirstKvSTile) {
            vstas(dmUreg, dmUb, 0, POST_UPDATE);
        }
    }

    template <bool IsFirstKvSTile>
    __simd_vf__ inline void ComputeAndUpdateMax(__ubuf__ float *sUb, __ubuf__ float *gmUb, __ubuf__ float *dmUb,
                                                uint16_t m, uint32_t tailN, float scaleValue, uint16_t s2BaseSize,
                                                tla::C<KvBaseTileRegSplitStagesHighPrec::TWO>)
    {
        using namespace AscendC::Reg;
        RegTensor<float> sVreg0;
        RegTensor<float> sVreg1;
        RegTensor<float> lmVreg;
        RegTensor<float> gmVreg;
        RegTensor<float> hmVreg;
        RegTensor<float> dmVreg;
        UnalignReg gmUreg;
        UnalignReg dmUreg;
        MaskReg maskReg32All = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskReg32TailN = UpdateMask<float>(tailN);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(sVreg0, sUb + i * s2BaseSize);
            LoadAlign(sVreg1, sUb + i * s2BaseSize + FLOAT_VECTOR_SIZE);
            Muls(sVreg0, sVreg0, scaleValue, maskReg32All);
            Muls(sVreg1, sVreg1, scaleValue, maskReg32All);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sUb + i * s2BaseSize, sVreg0, maskReg32All);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sUb + i * s2BaseSize + FLOAT_VECTOR_SIZE, sVreg1,
                                                        maskReg32TailN);
            Max<float, MaskMergeMode::MERGING>(sVreg0, sVreg0, sVreg1, maskReg32TailN);
            ReduceMax(lmVreg, sVreg0, maskReg32All);
            if constexpr (IsFirstKvSTile) {
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(gmUb, lmVreg, gmUreg, 1);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(gmVreg, gmUb);
                Max(hmVreg, lmVreg, gmVreg, maskReg32All);
                FusedExpSub(dmVreg, gmVreg, hmVreg, maskReg32All);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(gmUb, hmVreg, gmUreg, 1);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dmUb, dmVreg, dmUreg, 1);
            }
        }
        vstas(gmUreg, gmUb, 0, POST_UPDATE);
        if constexpr (!IsFirstKvSTile) {
            vstas(dmUreg, dmUb, 0, POST_UPDATE);
        }
    }

    template <bool IsFirstKvSTile>
    __simd_vf__ inline void ComputePAndUpdateSum(__ubuf__ ElementOutput *pUb, __ubuf__ float *sUb, __ubuf__ float *gmUb,
                                                 __ubuf__ float *glUb, __ubuf__ float *dmUb, uint16_t m, uint32_t tailN,
                                                 uint32_t blockStride, uint16_t s2BaseSize,
                                                 tla::C<KvBaseTileRegSplitStagesHighPrec::ONE>)
    {
        using namespace AscendC::Reg;
        constexpr static CastTrait castTraitZeroRound = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_ROUND};
        RegTensor<float> sVreg;
        RegTensor<float> pVreg;
        RegTensor<float> llVreg;
        RegTensor<float> hmVreg;
        RegTensor<float> dmVreg;
        RegTensor<float> glVreg;
        RegTensor<ElementOutput> pVreg16;
        RegTensor<ElementOutput> deInterleaveVreg;
        RegTensor<ElementOutput> deInterleaveTmpVreg;
        UnalignReg glUreg;
        uint32_t tailMaskCount = tailN;
        uint32_t outputMaskCount = tailN;
        MaskReg maskReg32TailN = UpdateMask<float>(tailMaskCount);
        MaskReg maskReg32All = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskReg16Output = UpdateMask<ElementOutput>(outputMaskCount);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(hmVreg, gmUb + i);
            LoadAlign(sVreg, sUb + i * s2BaseSize);
            FusedExpSub(pVreg, sVreg, hmVreg, maskReg32TailN);
            ReduceSum(llVreg, pVreg, maskReg32TailN);
            Cast<ElementOutput, float, castTraitZeroRound>(pVreg16, pVreg, maskReg32All);
            DeInterleave(deInterleaveVreg, deInterleaveTmpVreg, pVreg16, pVreg16);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(pUb + i * ELE_NUM_PER_C0, deInterleaveVreg,
                                                                     blockStride, maskReg16Output);
            if constexpr (IsFirstKvSTile) {
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(glUb, llVreg, glUreg, 1);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(glVreg, glUb);
                Mul(glVreg, glVreg, dmVreg, maskReg32All);
                Add(glVreg, glVreg, llVreg, maskReg32All);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(glUb, glVreg, glUreg, 1);
            }
        }
        vstas(glUreg, glUb, 0, POST_UPDATE);
    }

    template <bool IsFirstKvSTile>
    __simd_vf__ inline void ComputePAndUpdateSum(__ubuf__ ElementOutput *pUb, __ubuf__ float *sUb, __ubuf__ float *gmUb,
                                                 __ubuf__ float *glUb, __ubuf__ float *dmUb, uint16_t m, uint32_t tailN,
                                                 uint32_t blockStride, uint16_t s2BaseSize,
                                                 tla::C<KvBaseTileRegSplitStagesHighPrec::TWO>)
    {
        using namespace AscendC::Reg;
        constexpr static CastTrait castTraitZeroRound = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                         AscendC::RoundMode::CAST_ROUND};
        RegTensor<float> sVreg0;
        RegTensor<float> sVreg1;
        RegTensor<float> pVreg0;
        RegTensor<float> pVreg1;
        RegTensor<float> llVreg;
        RegTensor<float> hmVreg;
        RegTensor<float> dmVreg;
        RegTensor<float> glVreg;
        RegTensor<ElementOutput> pVreg160;
        RegTensor<ElementOutput> pVreg161;
        RegTensor<ElementOutput> deInterleaveVreg;
        RegTensor<ElementOutput> deInterleaveTmpVreg;
        UnalignReg glUreg;
        uint32_t tailMaskCount = tailN;
        uint32_t outputMaskCount = FLOAT_VECTOR_SIZE + tailN;
        MaskReg maskReg32All = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskReg32TailN = UpdateMask<float>(tailMaskCount);
        MaskReg maskReg16Output = UpdateMask<ElementOutput>(outputMaskCount);

        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(hmVreg, gmUb + i);
            LoadAlign(sVreg0, sUb + i * s2BaseSize);
            LoadAlign(sVreg1, sUb + i * s2BaseSize + FLOAT_VECTOR_SIZE);
            FusedExpSub(pVreg0, sVreg0, hmVreg, maskReg32All);
            FusedExpSub(pVreg1, sVreg1, hmVreg, maskReg32TailN);
            Cast<ElementOutput, float, castTraitZeroRound>(pVreg160, pVreg0, maskReg32All);
            Cast<ElementOutput, float, castTraitZeroRound>(pVreg161, pVreg1, maskReg32All);
            DeInterleave(deInterleaveVreg, deInterleaveTmpVreg, pVreg160, pVreg161);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(pUb + i * ELE_NUM_PER_C0, deInterleaveVreg,
                                                                     blockStride, maskReg16Output);
            Add<float, MaskMergeMode::MERGING>(pVreg0, pVreg0, pVreg1, maskReg32TailN);
            ReduceSum(llVreg, pVreg0, maskReg32All);
            if constexpr (IsFirstKvSTile) {
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(glUb, llVreg, glUreg, 1);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(glVreg, glUb);
                Mul(glVreg, glVreg, dmVreg, maskReg32All);
                Add(glVreg, glVreg, llVreg, maskReg32All);
                StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(glUb, glVreg, glUreg, 1);
            }
        }
        vstas(glUreg, glUb, 0, POST_UPDATE);
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
