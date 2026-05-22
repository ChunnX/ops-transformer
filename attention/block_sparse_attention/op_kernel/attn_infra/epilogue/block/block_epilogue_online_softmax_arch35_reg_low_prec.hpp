/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"

namespace NpuArch::Epilogue::Block {

template <typename DefaultType>
struct ElementPTmpTypeLookup {
    using type = DefaultType;
};

template <> 
struct ElementPTmpTypeLookup<float8_e4m3_t> {
    using type = half;
};

enum class KvBaseTileRegSplitStages {
    ONE,
    TWO
};

template <
    class OutputType_,
    class LayoutS_>
class BlockEpilogue<
    EpilogueOnlineSoftmaxBsa,
    OutputType_,
    Gemm::GemmType<half, LayoutS_>>
{
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsa;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementOutputTmp = typename ElementPTmpTypeLookup<ElementOutput>::type;


    using ElementInput = half;
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
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 16384;
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
    static constexpr uint32_t FP8_REP_SIZE = 256;
    static constexpr uint32_t HALF_REP_SIZE = 128;
    static constexpr uint32_t FLOAT_REP_SIZE = 64;
    static constexpr uint32_t BLOCK_REP_SIZE = 8;
    static constexpr uint32_t REPEAT_STRIDE = 1;
    static constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;
    static constexpr uint32_t SM_VREG_SIZE = 256 / sizeof(ElementInput);
    static constexpr uint32_t QUANT_MODE1_SCALE_BYTES = 1024;

    __aicore__ inline
    BlockEpilogue(Arch::Resource<ArchTag> &resource, float scaleValue_)
    {
        // Allocate UB space
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t LP_UB_TENSOR_OFFSET = 2 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t LM_UB_TENSOR_OFFSET = 7 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = LM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GM_UB_TENSOR_OFFSET + 64 * sizeof(float);
        constexpr uint32_t LL_UB_TENSOR_OFFSET = DM_UB_TENSOR_OFFSET + 3 * 64 * sizeof(float);
        constexpr uint32_t GL_UB_TENSOR_OFFSET = LL_UB_TENSOR_OFFSET +  64 * sizeof(float);

        subBlockIdx_ = AscendC::GetSubBlockIdx();

        scaleValue = static_cast<ElementInput>(scaleValue_);
        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        lpUbTensor = resource.ubBuf.template GetBufferByByte<ElementOutputTmp>(LP_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        lmUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LM_UB_TENSOR_OFFSET);
        llUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LL_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue()
    {
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline
    void CopyPUbToPL1(TensorDst const &dstTensor, TensorSrc const &srcTensor, uint32_t m)
    {
        const uint32_t blockCount = tla::get<1, 1>(srcTensor.shape());
        const uint32_t blockLen = tla::get<0, 0>(srcTensor.shape()) * tla::get<0, 1>(srcTensor.shape());
        const uint32_t dstOuterStrideCol = tla::get<1, 1>(dstTensor.stride());
        constexpr int32_t C0_SIZE = 32 / sizeof(typename TensorDst::Element);
        AscendC::DataCopyParams repeatParams;

        repeatParams.blockCount = blockCount;
        repeatParams.blockLen = m;
        repeatParams.srcStride = tla::get<1, 1>(srcTensor.stride()) / C0_SIZE - m;
        repeatParams.dstStride = tla::get<1, 1>(dstTensor.stride()) / C0_SIZE - m;

        auto dstOffset = dstTensor.layout()(dstTensor.coord());
        auto srcOffset = srcTensor.layout()(srcTensor.coord());

        AscendC::DataCopy(dstTensor.data()[dstOffset], srcTensor.data()[srcOffset], repeatParams);
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC set for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE>
    __aicore__ inline
    void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag)
    {
        // in mode 4, AIC wait for 2 AIVs seperately
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <class TensorP>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag,
         int actualRownum=0){

        struct EmptyTensor {};
        EmptyTensor dummyQS;
        EmptyTensor dummyKS;
        operator()<0>(l1PTensorTla, dummyQS, dummyKS,
                        actualBlockShape,
                        isFirstKvSTile, ubSBufId, l1PBufId,
                        mm1ToSmFlag, smToMm2Flag, AscendC::GlobalTensor<int32_t>(),
                        0, actualRownum);
    }
    
    template <int quant_mode, class TensorP, class TensorQS, class TensorKS, typename ElementIndex>
    __aicore__ inline
    void operator()(TensorP &l1PTensorTla, TensorQS &gmQSTensorTla, TensorKS &gmKSTensorTla,
         GemmCoord actualBlockShape,
        uint32_t isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
         Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag,
        AscendC::GlobalTensor<ElementIndex> sparseIndex, int blockNum=0, int actualRownum=0)
    {
        static_assert(quant_mode == 0 || quant_mode == 1);
        uint32_t mCopyOffset = (actualRownum > 0) ?
            static_cast<uint32_t>(actualRownum) / 2 : RoundUp(actualBlockShape.m(), 8) / 2;
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
        constexpr int16_t vlSize = static_cast<int16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
        int16_t nLoops = AscendC::CeilDivision(n, vlSize) - 1;
        uint32_t tailN = (n - 1) % vlSize + 1;
        int16_t mLoops = AscendC::CeilDivision(m, vlSize) - 1;
        uint32_t tailM = (m - 1) % vlSize + 1;
        uint32_t nPadding = (tailN + BLOCK_SIZE_IN_BYTE - 1) / BLOCK_SIZE_IN_BYTE * BLOCK_SIZE_IN_BYTE;
        __ubuf__ ElementOutputTmp *pAddr = (__ubuf__ ElementOutputTmp*) lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ ElementInput *sAddr = (__ubuf__ ElementInput*) lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].GetPhyAddr();
        __ubuf__ float *lastMaxAddr = (__ubuf__ float *)gmUbTensor.GetPhyAddr();
        __ubuf__ float *lastSumAddr = (__ubuf__ float*) glUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowMaxAddr = (__ubuf__ ElementInput*) lmUbTensor.GetPhyAddr();
        __ubuf__ ElementInput *nowSumAddr = (__ubuf__ ElementInput*) llUbTensor.GetPhyAddr();
        __ubuf__ float *expMaxUbAddr = (__ubuf__ float *)dmUbTensor[l1PBufId * DM_UB_GLOBAL_ELEM_NUM].GetPhyAddr();
        //__ubuf__ float *queryQuantScaleAddr = (__ubuf__ float*) quantUbTensor[ubSBufId].GetPhyAddr();
        //__ubuf__ float *keyQuantScaleAddr = (__ubuf__ float*) quantUbTensor[ubSBufId].GetPhyAddr();

        // wait QK Fixpipe finsh
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        if constexpr (quant_mode == 1){
            // 反量化场景， query 按照M方向64粒度共享scale，key按照N方向64粒度共享Scale
            auto qlayout = gmQSTensorTla.layout();
            auto klayout = gmKSTensorTla.layout();
            // 将N方向按blockNum切分后， 每个block的N长度
            uint32_t blockSize = AscendC::CeilDivision(n, blockNum);
            // M方向按照64粒度切分后的块树，每64行共享1个query_scale
            uint32_t subMLoops = AscendC::CeilDivision(m, 64);
            // 每个block的N方向按照64粒度切分后的块数，每64列共享1个key scale
            uint32_t subNLoops = AscendC::CeilDivision(blockSize, 64);
            // 加载query scale， 按照M方向64粒度，每个粒度1个float， 写入quantUbTensor前subMLoops个位置
            float qscale[2];
            float kscale[8];
            for(uint32_t i = 0; i < subMLoops; i++){
                auto qOffset = qlayout(tla::MakeCoord(i*2 + AscendC::GetSubBlockIdx(), static_cast<uint32_t>(0)));
                qscale[i] = gmQSTensorTla.data().GetValue(qOffset);
            }
            for(uint32_t i = 0; i < blockNum; i++){
                uint32_t index = sparseIndex.GetValue(i);
                for(uint32_t j = 0; j < subNLoops; j++){
                    auto kOffset = klayout(tla::MakeCoord(index, j));
                    kscale[i*subNLoops + j] = gmKSTensorTla.data().GetValue(index*2 + j);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
            DequantRes<ElementInput>(sAddr, m, nLoops, tailN, nRound, qscale[0],
                kscale[0],
                kscale[1],
                kscale[2],
                kscale[3]);
        }

        

        uint32_t kvBaseTileRegStages = CeilDiv(n, SM_VREG_SIZE);
        if (kvBaseTileRegStages == 1) {
            ComputeScaleAndMax<KvBaseTileRegSplitStages::ONE>(
                sAddr, nowMaxAddr, m, tailN, nPadding, scaleValue, nRound);
        } else if (kvBaseTileRegStages == 2) {
            ComputeScaleAndMax<KvBaseTileRegSplitStages::TWO>(
                sAddr, nowMaxAddr, m, tailN, nPadding, scaleValue, nRound);
        }
        if (!isFirstKvSTile) {
            UpdateMax(nowMaxAddr, lastMaxAddr, mLoops, tailM);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        if constexpr (quant_mode == 1){
            ComputeExpSubSum16WithQuantFp8<ElementOutputTmp, ElementInput>(
                pAddr, sAddr, nowMaxAddr, nowSumAddr, m, nLoops, tailN, blockStride, nRound
            );
        } else {
            if (kvBaseTileRegStages == 1) {
                ComputeExpSubSum16<KvBaseTileRegSplitStages::ONE>(
                    pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN, blockStride, nRound);
            } else if (kvBaseTileRegStages == 2) {
                ComputeExpSubSum16<KvBaseTileRegSplitStages::TWO>(
                    pAddr, sAddr, nowMaxAddr, nowSumAddr, m, tailN, blockStride, nRound);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);

        auto ubPLayoutTla = tla::MakeLayout<ElementOutput, LayoutOutput>(mRound, nRound);
        auto ubPTensorTla = tla::MakeTensor(lpUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM].template ReinterpretCast<ElementOutput>(),
            ubPLayoutTla, Arch::PositionUB{});
        auto ubPTensorTlaTile = GetTile(ubPTensorTla,
                tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto l1PTensorTlaTile = GetTile(l1PTensorTla,
                tla::MakeCoord(subBlockIdx_ * mCopyOffset, 0), tla::MakeShape(m, n));
        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);

        CopyPUbToPL1(l1PTensorTlaTile, ubPTensorTlaTile, m);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ubSBufId + 2);
        // crossCoreSync after PIPE_MTE1 move
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        mLoops = AscendC::CeilDivision(m, 64) - 1;
        tailM = (m - 1) % 64 + 1;
        if (isFirstKvSTile) {
            CastExpSumAndExpMax<ElementInput>(
                lastSumAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mLoops, tailM);
        } else {
            UpdateExpSumAndExpMax<ElementInput>(
                lastSumAddr, expMaxUbAddr, lastMaxAddr, nowSumAddr, nowMaxAddr, mLoops, tailM);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    ElementInput scaleValue;
    AscendC::LocalTensor<ElementInput> lsUbTensor;
    AscendC::LocalTensor<ElementOutputTmp> lpUbTensor;
    AscendC::LocalTensor<float> gmUbTensor;
    AscendC::LocalTensor<float> glUbTensor;
    AscendC::LocalTensor<float> dmUbTensor;
    AscendC::LocalTensor<ElementInput> lmUbTensor;
    AscendC::LocalTensor<ElementInput> llUbTensor;
    uint32_t subBlockIdx_;

    template <KvBaseTileRegSplitStages kvBaseTileRegSplitStages>
    __simd_vf__ inline void ComputeScaleAndMax(__ubuf__ ElementInput *srcUb, __ubuf__ ElementInput *newMaxUb,
        uint16_t m, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize)
    {
    }
    
    template <>
    __simd_vf__ inline void ComputeScaleAndMax<KvBaseTileRegSplitStages::ONE>(
        __ubuf__ ElementInput *srcUb, __ubuf__ ElementInput *newMaxUb,
        uint16_t m, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> minVreg;
        RegTensor<ElementInput> srcVreg;
        RegTensor<ElementInput> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg, srcUb + i * S2BaseSize);
            Muls(srcVreg, srcVreg, dScale, pregFull);
            Select(srcVreg, srcVreg, minVreg, pregTailN);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(
                srcUb + i * S2BaseSize, srcVreg, pregTailN);
            ReduceMax(maxTmpVreg, srcVreg, pregFull);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    template <>
    __simd_vf__ inline void ComputeScaleAndMax<KvBaseTileRegSplitStages::TWO>(
        __ubuf__ ElementInput *srcUb, __ubuf__ ElementInput *newMaxUb,
        uint16_t m, uint32_t tailN, uint32_t nPadding, ElementInput dScale, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> minVreg;
        RegTensor<ElementInput> srcVreg0;
        RegTensor<ElementInput> srcVreg1;
        RegTensor<ElementInput> maxTmpVreg;
        UnalignReg maxUreg;
        MaskReg pregCompare;
        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);

        Duplicate(minVreg, MIN_VALUE);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign(srcVreg0, srcUb + i * S2BaseSize);
            LoadAlign(srcVreg1, srcUb + i * S2BaseSize + HALF_REP_SIZE);
            Muls(srcVreg0, srcVreg0, dScale, pregFull);
            Muls(srcVreg1, srcVreg1, dScale, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(
                srcUb + i * S2BaseSize, srcVreg0, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(
                srcUb + i * S2BaseSize + HALF_REP_SIZE, srcVreg1, pregTailN);
            Max<ElementInput, MaskMergeMode::MERGING>(srcVreg0, srcVreg0, srcVreg1, pregTailN);
            ReduceMax(maxTmpVreg, srcVreg0, pregFull);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(newMaxUb, maxTmpVreg, maxUreg, 1);
        }
        vstas(maxUreg, newMaxUb, 0, POST_UPDATE);
    }

    __simd_vf__ inline void UpdateMax(
        __ubuf__ ElementInput *nowMaxUb, __ubuf__ float *lastMaxUb, uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;
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

        RegTensor<ElementInput> nowMaxVreg;
        RegTensor<float> lastMaxFloatVreg0;
        RegTensor<float> lastMaxFloatVreg1;
        RegTensor<ElementInput> maxVreg;
        RegTensor<ElementInput> lastMaxVreg;
        RegTensor<ElementInput> lastMaxVreg0;
        RegTensor<ElementInput> lastMaxVreg1;

        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<ElementInput>(tailM);
        for (uint16_t i = 0; i < mLoops; ++i) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    lastMaxFloatVreg0, lastMaxFloatVreg1, lastMaxUb + i * HALF_REP_SIZE);
            Cast<ElementInput, float, castTraitZero>(lastMaxVreg0, lastMaxFloatVreg0, pregFloatFull);
            Cast<ElementInput, float, castTraitOne>(lastMaxVreg1, lastMaxFloatVreg1, pregFloatFull);
            Or((RegTensor<uint16_t>&)lastMaxVreg,
                (RegTensor<uint16_t>&)lastMaxVreg0, (RegTensor<uint16_t>&)lastMaxVreg1,
                pregFull);
            LoadAlign(nowMaxVreg, nowMaxUb + i * HALF_REP_SIZE);
            Max(maxVreg, nowMaxVreg, lastMaxVreg, pregFull);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(nowMaxUb + i * HALF_REP_SIZE, maxVreg, pregFull);
        }
        if (tailM > 64) {
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    lastMaxFloatVreg0, lastMaxFloatVreg1, lastMaxUb + mLoops * HALF_REP_SIZE);
            Cast<ElementInput, float, castTraitZero>(lastMaxVreg0, lastMaxFloatVreg0, pregFloatFull);
            Cast<ElementInput, float, castTraitOne>(lastMaxVreg1, lastMaxFloatVreg1, pregFloatFull);
            Or((RegTensor<uint16_t>&)lastMaxVreg,
               (RegTensor<uint16_t>&)lastMaxVreg0, (RegTensor<uint16_t>&)lastMaxVreg1,
               pregFull);
            LoadAlign(nowMaxVreg, nowMaxUb + mLoops * HALF_REP_SIZE);
            Max(maxVreg, nowMaxVreg, lastMaxVreg, pregTailM);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(nowMaxUb + mLoops * HALF_REP_SIZE, maxVreg, pregTailM);
        } else {
            LoadAlign(lastMaxFloatVreg0, lastMaxUb + mLoops * HALF_REP_SIZE);
            Cast<ElementInput, float, castTraitZero>(lastMaxVreg, lastMaxFloatVreg0, pregFloatFull);
            DeInterleave(lastMaxVreg0, lastMaxVreg1, lastMaxVreg, lastMaxVreg);
            LoadAlign(nowMaxVreg, nowMaxUb + mLoops * HALF_REP_SIZE);
            Max(maxVreg, nowMaxVreg, lastMaxVreg0, pregTailM);
            StoreAlign<ElementInput, StoreDist::DIST_NORM_B16>(nowMaxUb + mLoops * HALF_REP_SIZE, maxVreg, pregTailM);
        }
    }

    template <typename ElementS, typename ElementP>
    __simd_vf__ inline void ComputeExpSubSum16WithQuantFp8(__ubuf__ ElementP *expUb, __ubuf__ ElementS *srcUb,
        __ubuf__ ElementS *nowMaxUb, __ubuf__ ElementS *expSumUb,
        uint16_t m, uint16_t nLoops, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        
        RegTensor<ElementS> expVreg;
        RegTensor<ElementS> expSumVreg;
        RegTensor<ElementS> maxVreg;
        RegTensor<ElementP> expDstVreg;

        RegTensor<float8_e4m3_t> outputVReg;
        constexpr static CastTrait castTraitHalf2FloatZero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };
        constexpr static CastTrait castTraitHalf2FloatOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_ROUND,
        };
        constexpr static CastTrait castTraitFp8Zero = {
            RegLayout::ZERO,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };
        constexpr static CastTrait castTraitFp8One = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };

        RegTensor<float> floatReg0;
        RegTensor<float> floatReg1;
        RegTensor<float8_e4m3_t> fp8Reg0;
        RegTensor<float8_e4m3_t> fp8Reg1;
        RegTensor<half> fp16Reg0;

        MaskReg pregHalfFull = CreateMask<ElementP, MaskPattern::ALL>();
        MaskReg pregInt16Full = CreateMask<int16_t, MaskPattern::ALL>();
        MaskReg pregFloatFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregAllFp8 = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg pregHalfFp8 = CreateMask<uint8_t, MaskPattern::VL128>();
        __ubuf__ float8_e4m3_t *pAddrFp8 = (__ubuf__ float8_e4m3_t *)expUb;

        UnalignReg expSumUreg;

        MaskReg pregFull = CreateMask<ElementS, MaskPattern::ALL>();
        uint32_t tailN1, tailN2, tailN3;
        tailN1=tailN2=tailN3=tailN;
        MaskReg pregTailN = UpdateMask<ElementS>(tailN1);
        MaskReg pregFloatTailN = UpdateMask<float>(tailN2);
        MaskReg pregFp8TailN = UpdateMask<float8_e4m3_t>(tailN3);
        constexpr static int FP8_ELE_NUM_PER_C0 = 32;

        for (uint16_t i = 0; i < m; ++i){
            LoadAlign<ElementS, LoadDist::DIST_BRC_B16>(maxVreg, nowMaxUb + i);
            Duplicate(expSumVreg, 0);
            for (uint16_t j = 0; j < nLoops; ++j){
                LoadAlign(expVreg, srcUb + i * S2BaseSize + j * HALF_REP_SIZE);
                Sub(expDstVreg, expVreg, maxVreg, pregFull);
                Exp(expDstVreg, expDstVreg, pregFull);
                Add(expSumVreg, expSumVreg, expDstVreg, pregFull);
                Cast<float, ElementP, castTraitHalf2FloatZero>(floatReg0, expDstVreg, pregHalfFull);
                Cast<float, ElementP, castTraitHalf2FloatOne>(floatReg1, expDstVreg, pregHalfFull);
                Muls(floatReg0, floatReg0, (float)448.0, pregFloatFull);
                Muls(floatReg1, floatReg1, (float)448.0, pregFloatFull);
                Cast<float8_e4m3_t, float, castTraitFp8Zero>(fp8Reg0, floatReg0, pregFloatFull);
                Cast<float8_e4m3_t, float, castTraitFp8One>(fp8Reg1, floatReg1, pregFloatFull);
                Or((RegTensor<uint8_t>&)fp8Reg0, (RegTensor<uint8_t>&)fp8Reg0, (RegTensor<uint8_t>&)fp8Reg1, pregAllFp8);
                Pack((RegTensor<uint16_t>&)fp16Reg0, (RegTensor<uint32_t>&)fp8Reg0);
                StoreAlign<float8_e4m3_t, DataCopyMode::DATA_BLOCK_COPY>(
                    pAddrFp8 + i * FP8_ELE_NUM_PER_C0 + j * blockStride * HALF_REP_SIZE,
                    (RegTensor<float8_e4m3_t>&)fp16Reg0, blockStride, pregHalfFp8);
            }
            LoadAlign(expVreg, srcUb + i * S2BaseSize + nLoops * HALF_REP_SIZE);
            Sub(expDstVreg, expVreg, maxVreg, pregTailN);
            Exp(expDstVreg, expDstVreg, pregTailN);
            Add<ElementP, MaskMergeMode::MERGING>(expSumVreg, expSumVreg, expDstVreg, pregTailN);
            Cast<float, ElementP, castTraitHalf2FloatZero>(floatReg0, expDstVreg, pregHalfFull);
            Cast<float, ElementP, castTraitHalf2FloatOne>(floatReg1, expDstVreg, pregHalfFull);
            Muls(floatReg0, floatReg0, (float)448.0, pregFloatFull);
            Muls(floatReg1, floatReg1, (float)448.0, pregFloatFull);
            Cast<float8_e4m3_t, float, castTraitFp8Zero>(fp8Reg0, floatReg0, pregFloatFull);
            Cast<float8_e4m3_t, float, castTraitFp8One>(fp8Reg1, floatReg1, pregFloatFull);
            Or((RegTensor<uint8_t>&)fp8Reg0, (RegTensor<uint8_t>&)fp8Reg0, (RegTensor<uint8_t>&)fp8Reg1, pregAllFp8);
            Pack((RegTensor<uint16_t>&)fp16Reg0, (RegTensor<uint32_t>&)fp8Reg0);
            StoreAlign<float8_e4m3_t, DataCopyMode::DATA_BLOCK_COPY>(
                pAddrFp8 + i * FP8_ELE_NUM_PER_C0 + nLoops * blockStride * HALF_REP_SIZE,
                (RegTensor<float8_e4m3_t>&)fp16Reg0, blockStride, pregFp8TailN);
            ReduceSum(expSumVreg, expSumVreg, pregFull);
            StoreUnAlign<ElementS, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);

        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);


    }

    template <KvBaseTileRegSplitStages kvBaseTileRegSplitStages>
    __simd_vf__ inline void ComputeExpSubSum16(__ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb,
        __ubuf__ ElementInput *nowMaxUb, __ubuf__ ElementInput *expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize)
    {
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum16<KvBaseTileRegSplitStages::ONE>(
        __ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb,
        __ubuf__ ElementInput *nowMaxUb, __ubuf__ ElementInput *expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> expVreg;
        RegTensor<ElementInput> expSumVreg;
        RegTensor<ElementInput> maxVreg;

        RegTensor<ElementOutput> expDstVreg;

        UnalignReg expSumUreg;

        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementInput, LoadDist::DIST_BRC_B16>(maxVreg, nowMaxUb + i);
            LoadAlign(expVreg, srcUb + i * S2BaseSize);
            Sub(expDstVreg, expVreg, maxVreg, pregTailN);
            Exp(expDstVreg, expDstVreg, pregTailN);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0,
                expDstVreg, blockStride, pregTailN);
            ReduceSum(expSumVreg, expDstVreg, pregTailN);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <>
    __simd_vf__ inline void ComputeExpSubSum16<KvBaseTileRegSplitStages::TWO>(
        __ubuf__ ElementOutput *expUb, __ubuf__ ElementInput *srcUb,
        __ubuf__ ElementInput *nowMaxUb, __ubuf__ ElementInput *expSumUb,
        uint16_t m, uint32_t tailN, uint32_t blockStride, uint16_t S2BaseSize)
    {
        using namespace AscendC::MicroAPI;
        RegTensor<ElementInput> expVreg0;
        RegTensor<ElementInput> expVreg1;
        RegTensor<ElementInput> expSumVreg;
        RegTensor<ElementInput> maxVreg;

        RegTensor<ElementOutput> expDstVreg0;
        RegTensor<ElementOutput> expDstVreg1;

        UnalignReg expSumUreg;

        MaskReg pregFull = CreateMask<ElementInput, MaskPattern::ALL>();
        MaskReg pregTailN = UpdateMask<ElementInput>(tailN);
        for (uint16_t i = 0; i < m; ++i) {
            LoadAlign<ElementInput, LoadDist::DIST_BRC_B16>(maxVreg, nowMaxUb + i);
            // Duplicate(expSumVreg, 0);
            LoadAlign(expVreg0, srcUb + i * S2BaseSize);
            LoadAlign(expVreg1, srcUb + i * S2BaseSize + HALF_REP_SIZE);
            Sub(expDstVreg0, expVreg0, maxVreg, pregFull);
            Sub(expDstVreg1, expVreg1, maxVreg, pregTailN);
            Exp(expDstVreg0, expDstVreg0, pregFull);
            Exp(expDstVreg1, expDstVreg1, pregTailN);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0,
                expDstVreg0, blockStride, pregFull);
            StoreAlign<ElementOutput, DataCopyMode::DATA_BLOCK_COPY>(
                expUb + i * ELE_NUM_PER_C0 + blockStride * ELE_NUM_PER_C0 * BLOCK_REP_SIZE,
                expDstVreg1, blockStride, pregTailN);
            Add<ElementOutput, MaskMergeMode::MERGING>(expDstVreg0, expDstVreg1, expDstVreg0, pregTailN);

            ReduceSum(expSumVreg, expDstVreg0, pregFull);
            StoreUnAlign<ElementInput, PostLiteral::POST_MODE_UPDATE>(expSumUb, expSumVreg, expSumUreg, 1);
        }
        vstas(expSumUreg, expSumUb, 0, POST_UPDATE);
    }

    template <typename ElementInput>
    __simd_vf__ inline void CastExpSumAndExpMax(__ubuf__ float *sumUb, __ubuf__ float *maxUb,
        __ubuf__ ElementInput *expSumUb, __ubuf__ ElementInput *nowMaxUb,
        uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        constexpr static CastTrait castTrait = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> nowMaxVreg0;
        RegTensor<ElementInput> nowMaxVreg1;
        RegTensor<ElementInput> nowMaxTmpVreg;
        RegTensor<float> nowMaxFloatVreg;
        RegTensor<ElementInput> brcExpSumVreg0;
        RegTensor<ElementInput> brcExpSumVreg1;
        RegTensor<ElementInput> brcExpSumTmpVreg;
        RegTensor<float> brcExpSumFloatVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);
        MaskReg pregNowMaxFull = CreateMask<ElementInput, MaskPattern::ALL>();
        for (int16_t i = 0; i < mLoops; ++i) {
            LoadAlign(nowMaxTmpVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
            Cast<float, ElementInput, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg, pregFull);

            LoadAlign(brcExpSumTmpVreg, expSumUb + i * FLOAT_REP_SIZE);
            Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
            Cast<float, ElementInput, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + i * FLOAT_REP_SIZE, brcExpSumFloatVreg, pregFull);
        }
        LoadAlign(nowMaxTmpVreg, nowMaxUb + mLoops * FLOAT_REP_SIZE);
        Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
        Cast<float, ElementInput, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + mLoops * FLOAT_REP_SIZE, nowMaxFloatVreg, pregTailM);

        LoadAlign(brcExpSumTmpVreg, expSumUb + mLoops * FLOAT_REP_SIZE);
        Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
        Cast<float, ElementInput, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + mLoops * FLOAT_REP_SIZE, brcExpSumFloatVreg, pregTailM);
    }

    template <typename ElementInput>
    __simd_vf__ inline void UpdateExpSumAndExpMax(__ubuf__ float *sumUb, __ubuf__ float *expMaxUb,
        __ubuf__ float *maxUb, __ubuf__ ElementInput *expSumUb, __ubuf__ ElementInput *nowMaxUb,
        uint16_t mLoops, uint32_t tailM)
    {
        using namespace AscendC::MicroAPI;

        constexpr static CastTrait castTrait = {
            RegLayout::ZERO,
            SatMode::UNKNOWN,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::UNKNOWN,
        };

        RegTensor<ElementInput> nowMaxVreg0;
        RegTensor<ElementInput> nowMaxVreg1;
        RegTensor<ElementInput> nowMaxTmpVreg;
        RegTensor<float> nowMaxFloatVreg;
        RegTensor<float> lastMaxVreg;
        RegTensor<float> expMaxVreg;
        RegTensor<float> lastExpSumVreg;
        RegTensor<ElementInput> brcExpSumVreg0;
        RegTensor<ElementInput> brcExpSumVreg1;
        RegTensor<ElementInput> brcExpSumTmpVreg;
        RegTensor<float> brcExpSumFloatVreg;
        RegTensor<float> updateExpSumVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTailM = UpdateMask<float>(tailM);
        MaskReg pregNowMaxFull = CreateMask<ElementInput, MaskPattern::ALL>();
        for (int16_t i = 0; i < mLoops; ++i) {
            LoadAlign(lastMaxVreg, maxUb + i * FLOAT_REP_SIZE);
            LoadAlign(nowMaxTmpVreg, nowMaxUb + i * FLOAT_REP_SIZE);
            Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
            Cast<float, ElementInput, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
            FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + i * FLOAT_REP_SIZE, expMaxVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + i * FLOAT_REP_SIZE, nowMaxFloatVreg, pregFull);

            LoadAlign(lastExpSumVreg, sumUb + i * FLOAT_REP_SIZE);
            LoadAlign(brcExpSumTmpVreg, expSumUb + i * FLOAT_REP_SIZE);
            Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
            Cast<float, ElementInput, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
            Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregFull);
            Add(updateExpSumVreg, updateExpSumVreg, brcExpSumFloatVreg, pregFull);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + i * FLOAT_REP_SIZE, updateExpSumVreg, pregFull);
        }
        LoadAlign(lastMaxVreg, maxUb + mLoops * FLOAT_REP_SIZE);
        LoadAlign(nowMaxTmpVreg, nowMaxUb + mLoops * FLOAT_REP_SIZE);
        Interleave(nowMaxVreg0, nowMaxVreg1, nowMaxTmpVreg, nowMaxTmpVreg);
        Cast<float, ElementInput, castTrait>(nowMaxFloatVreg, nowMaxVreg0, pregNowMaxFull);
        FusedExpSub(expMaxVreg, lastMaxVreg, nowMaxFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxUb + mLoops * FLOAT_REP_SIZE, expMaxVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(maxUb + mLoops * FLOAT_REP_SIZE, nowMaxFloatVreg, pregTailM);

        LoadAlign(lastExpSumVreg, sumUb + mLoops * FLOAT_REP_SIZE);
        LoadAlign(brcExpSumTmpVreg, expSumUb + mLoops * FLOAT_REP_SIZE);
        Interleave(brcExpSumVreg0, brcExpSumVreg1, brcExpSumTmpVreg, brcExpSumTmpVreg);
        Cast<float, ElementInput, castTrait>(brcExpSumFloatVreg, brcExpSumVreg0, pregNowMaxFull);
        Mul(updateExpSumVreg, expMaxVreg, lastExpSumVreg, pregTailM);
        Add(updateExpSumVreg, updateExpSumVreg, brcExpSumFloatVreg, pregTailM);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumUb + mLoops * FLOAT_REP_SIZE, updateExpSumVreg, pregTailM);
    }

        /**
     * @brief 反量化half类型的S矩阵
     * @details M方向和N方向均按64粒度共享一个scale值进行反量化。
     *          queryScaleUb按M方向64粒度组织，valueScaleUb按N方向64粒度组织（每64列2个float值）。
     *          通过Interleave/DeInterleave实现half到float的转换和反量化计算。
     * @param queryScaleUb  query反量化scale数组，按M方向64粒度存储
     * @param valueScaleUb  value反量化scale数组，按N方向64粒度存储
     * @param srcUb         输入输出half类型数据缓冲区（原地反量化）
     * @param m             M方向行数
     * @param nLoops        N方向完整128元素循环次数
     * @param tailN         N方向尾部剩余元素数
     * @param S2BaseSize    每行元素数（nRound，用于地址计算）
     */
    template <typename ElementS>
    __simd_vf__ inline void DequantRes(__ubuf__ ElementS *srcUb,
        uint16_t m, uint16_t nLoops, uint32_t tailN, uint16_t S2BaseSize, float qscale, float valueScale0, float valueScale1, float valueScale2, float valueScale3)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait castTraitHalf2FloatZero = {
                RegLayout::ZERO,
                SatMode::SAT,
                MaskMergeMode::ZEROING,
                AscendC::RoundMode::UNKNOWN,
            };
        constexpr static CastTrait castTraitHalf2FloatOne = {
                RegLayout::ONE,
                SatMode::SAT,
                MaskMergeMode::ZEROING,
                AscendC::RoundMode::UNKNOWN,
        };
        constexpr static CastTrait castTraitFloat2HalfZero = {
                RegLayout::ZERO,
                SatMode::SAT,
                MaskMergeMode::ZEROING,
                AscendC::RoundMode::CAST_RINT,
            };
        constexpr static CastTrait castTraitFloat2HalfOne = {
            RegLayout::ONE,
            SatMode::SAT,
            MaskMergeMode::ZEROING,
            AscendC::RoundMode::CAST_RINT,
        };
        RegTensor<ElementS> srcReg;
        RegTensor<float> queryScaleReg;
        RegTensor<float> keyScaleReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        RegTensor<ElementS> halfReg;
        RegTensor<float> floatReg0;
        RegTensor<float> floatReg1;
        RegTensor<float> floatReg2;
        RegTensor<float> floatReg3;
        RegTensor<ElementS> halfReg0;
        RegTensor<ElementS> halfReg1;
        MaskReg pregAllFp8 = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg pregHalfFull = CreateMask<ElementS, MaskPattern::ALL>();
        uint32_t tailFloat = (tailN + 1)/2;
        MaskReg pregTailFloat = CreateMask<float, MaskPattern::ALL>();
        uint32_t tailNtmp = tailN;
        MaskReg pregTailHalf = UpdateMask<ElementS>(tailNtmp);

        // 外层按M方向循环，每64行共享同一个query scale
        for (uint16_t i = 0; i < m; ++i) {
            // queryScaleUb按64粒度组织，每64行取1个scale值
            float query_scale = qscale;
            Duplicate<float, float>(queryScaleReg, query_scale);
            float lastScale0 = valueScale2;
            float lastScale1 = valueScale3;
            float curScale0 = valueScale0;
            float curScale1 = valueScale1;
            float tmp0, tmp1;
            // 处理N方向完整的128元素循环（每个循环体处理128个half元素）
            for (uint16_t j = 0; j < nLoops; ++j) {
                // valueScaleUb按N方向64粒度组织，每64列共享一个scale
                // 每个128元素循环包含2个64粒度，因此每个循环取2个key scale
                float key_scale1 = curScale0;
                float key_scale2 = curScale1;
                LoadAlign<ElementS>(srcReg, srcUb + i * S2BaseSize + j * HALF_REP_SIZE);
                // Step1: half -> float转换（RegLayout::ZERO/ONE分别取低64和高64元素）
                Cast<float, ElementS, castTraitHalf2FloatZero>(floatReg0, srcReg, pregHalfFull);
                Cast<float, ElementS, castTraitHalf2FloatOne>(floatReg1, srcReg, pregHalfFull);
                // Step2: Interleave将float数据重排，使floatReg2对应key_scale1作用的数据，floatReg3对应key_scale2
                Interleave(floatReg2, floatReg3, floatReg0, floatReg1);
                // Step3: 先乘以key scale，再乘以query scale，完成反量化
                Duplicate(keyScaleReg, key_scale1);
                Mul(floatReg2, keyScaleReg, floatReg2, pregFull);
                Mul(floatReg2, queryScaleReg, floatReg2, pregFull);
                Duplicate(keyScaleReg, key_scale2);
                Mul(floatReg3, keyScaleReg, floatReg3, pregFull);
                Mul(floatReg3, queryScaleReg, floatReg3, pregFull);
                // Step4: DeInterleave将数据还原回原来的布局
                DeInterleave(floatReg0, floatReg1, floatReg2, floatReg3);
                // Step5: float -> half转换回写（RegLayout::ZERO/ONE分别写入低64和高64元素）
                Cast<ElementS, float, castTraitFloat2HalfZero>(halfReg0, floatReg0, pregFull);
                Cast<ElementS, float, castTraitFloat2HalfOne>(halfReg1, floatReg1, pregFull);
                Or((RegTensor<uint8_t>&)halfReg, (RegTensor<uint8_t>&)halfReg0, (RegTensor<uint8_t>&)halfReg1, pregAllFp8);
                
                StoreAlign<ElementS, StoreDist::DIST_NORM_B16>(
                    srcUb + i * S2BaseSize + j * HALF_REP_SIZE, halfReg, pregHalfFull);
                tmp0 = curScale0;
                tmp1 = curScale1;
                curScale0 = lastScale0;
                curScale1 = lastScale1;
                lastScale0 = tmp0;
                lastScale1 = tmp1;
            }
            // 处理N方向尾部不足128个元素的情况
            if (tailN > 0) {
                float key_scale1 = curScale0;
                float key_scale2 = curScale1;
                LoadAlign<ElementS>(srcReg, srcUb + i * S2BaseSize + nLoops * HALF_REP_SIZE);
                Cast<float, ElementS, castTraitHalf2FloatZero>(floatReg0, srcReg, pregHalfFull);
                Cast<float, ElementS, castTraitHalf2FloatOne>(floatReg1, srcReg, pregHalfFull);
                Interleave(floatReg2, floatReg3, floatReg0, floatReg1);
                Duplicate(keyScaleReg, key_scale1);
                Mul(floatReg2, keyScaleReg, floatReg2, pregTailFloat);
                Mul(floatReg2, queryScaleReg, floatReg2, pregTailFloat);
                Duplicate(keyScaleReg, key_scale2);
                Mul(floatReg3, keyScaleReg, floatReg3, pregTailFloat);
                Mul(floatReg3, queryScaleReg, floatReg3, pregTailFloat);
                DeInterleave(floatReg0, floatReg1, floatReg2, floatReg3);
                Cast<ElementS, float, castTraitFloat2HalfZero>(halfReg0, floatReg0, pregTailFloat);
                Cast<ElementS, float, castTraitFloat2HalfOne>(halfReg1, floatReg1, pregTailFloat);
                Or((RegTensor<uint8_t>&)halfReg, (RegTensor<uint8_t>&)halfReg0, (RegTensor<uint8_t>&)halfReg1, pregAllFp8);
                StoreAlign<ElementS, StoreDist::DIST_NORM_B16>(
                    srcUb + i * S2BaseSize + nLoops * HALF_REP_SIZE, halfReg, pregTailHalf);
            }
            
        }
    }
    
};
}

#endif  // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_BSA_LOW_PREC_HPP
