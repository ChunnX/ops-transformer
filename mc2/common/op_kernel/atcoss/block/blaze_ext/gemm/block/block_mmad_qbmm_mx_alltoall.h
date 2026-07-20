/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_qbmm_mx_alltoall.h
 * \brief BlockMmad for AlltoAll: based on block_mmad_qbmm_mx.h with minimal rank-boundary changes.
 *
 * Key difference from block_mmad_qbmm_mx.h:
 * - operator() loop tracks rankIdx/localIter0; when crossing rank boundary, updates gmA/scaleA pointer
 * - CopyAInL1/CopyScalesInL1 use per-rank GM base address + localIter0 offset
 * - B/ScaleB use globalKOffset = rankIdx * rankKPerRank + localIter0 * kL1
 * - Assumption: rankKPerRank is divisible by kL1_
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "../policy/dispatch_policy_alltoall.h"
#include "blaze/gemm/block/block_mmad.h"
#include "tensor_api/tensor.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/tile/pad_mx_kl1.h"

namespace Blaze {
namespace Gemm {
namespace Block {
#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
template <
    uint64_t A_FULL_LOAD_MODE, bool ATOMIC_ADD, class AType_, class LayoutA_, class BType_, class LayoutB_,
    class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<
    MatmulWithScaleMxAlltoAll<A_FULL_LOAD_MODE, ATOMIC_ADD>, AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_,
    LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using BiasType = BiasType_;
    using DispatchPolicy = MatmulWithScaleMxAlltoAll<A_FULL_LOAD_MODE, ATOMIC_ADD>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    uint64_t k_;
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t scaleKL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    bool isBias_{false};
    static constexpr bool weightNz = IsWeightNz<LayoutB>::value;
    static constexpr bool transA = IsTrans<LayoutA>::value;
    static constexpr bool transB = IsTrans<LayoutB>::value;

    constexpr static uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
    constexpr static uint64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    constexpr static uint64_t SCALE_C0 = 2;
    constexpr static uint64_t SCALE_BUFFER_NUM = 2;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        transB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutAGM = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutScaleAGM = AscendC::Std::conditional_t<
        transA, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    uint64_t abL1LoopCnt_{0};
    uint64_t scaleLoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t splitKNum_{1};
    bool enableL0cPingPong_{false};

    uint32_t rankDim_{1};
    uint64_t rankKPerRank_{0};
    uint64_t scaleKPerRank_{0};
    uint64_t localKL1Iter_{0};
    uint64_t aRankStride_{0};
    uint64_t scaleARankStride_{0};
    uint64_t mPos_{0};
    __gm__ AType *aGmBaseAddr_{nullptr};
    __gm__ fp8_e8m0_t *scaleAGmBaseAddr_{nullptr};

    struct TileL1L0Param {
        uint64_t curM = 0;
        uint64_t curN = 0;
        uint64_t curGmAKL1 = 0;
        uint64_t curGmBKL1 = 0;
        uint64_t curPadAKL1 = 0;
        uint64_t curPadBKL1 = 0;
    };

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        GM_ADDR scaleAGmAddr{nullptr};
        GM_ADDR scaleBGmAddr{nullptr};
    };

    struct L1Params {
        uint64_t kL1;
        uint64_t scaleKL1;
        uint64_t l1BufNum;
    };

    struct AlltoAllParams {
        uint32_t rankDim;
        uint64_t rankKPerRank;
        uint64_t scaleKPerRank;
        uint64_t aRankStride;
        uint64_t scaleARankStride;
    };

    template <typename TSA, typename TSB>
    struct ScalePair {
        TSA scaleA;
        TSB scaleB;
    };

    __aicore__ inline BlockMmad()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(INPUT_BUFFER_FLAG_1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(M_MTE1_FLAG_1);
        AscendC::SetMMLayoutTransform(false);
    }

public:
    __aicore__ inline void Init(
        const ProblemShape& problemShape, const BlockShape& l0TileShape, const L1Params& l1Params, bool isBias,
        bool dbL0C, uint64_t splitKNum = 1)
    {
        k_ = AscendC::Te::Get<IDX_K_IDX>(problemShape);
        kL1_ = l1Params.kL1;
        scaleKL1_ = l1Params.scaleKL1;
        splitKNum_ = splitKNum;
        baseM_ = AscendC::Te::Get<IDX_M_IDX>(l0TileShape);
        baseN_ = AscendC::Te::Get<IDX_N_IDX>(l0TileShape);
        baseK_ = AscendC::Te::Get<IDX_K_IDX>(l0TileShape);
        isBias_ = isBias;
        l1BufNum_ = l1Params.l1BufNum;
        enableL0cPingPong_ = dbL0C;
        constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;
        bL1OneBuffer_ = (baseN_ * kL1_) >> sizeShift;
        scaleBL1OneBuffer_ = baseN_ * Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        biasL1OneBuffer_ = isBias_ ? baseN_ * sizeof(BiasType) : 0;
        aL1OneBuffer_ = (baseM_ * Blaze::Gemm::CeilAlign(kL1_, MXFP_DIVISOR_SIZE)) >> sizeShift;
        scaleAL1OneBuffer_ = baseM_ * Blaze::Gemm::CeilDiv(scaleKPerRank_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        for (int32_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
            uint64_t l1Offset = (AscendC::TOTAL_L1_SIZE >> 1) * (bufferId & 1);
            l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer_ * (bufferId >> 1);
            l1BufferBOffset_[bufferId] =
                l1Offset + aL1OneBuffer_ * (l1BufNum_ >> 1) + bL1OneBuffer_ * (bufferId >> 1);
        }
        for (int32_t bufferId = 0; bufferId < SCALE_BUFFER_NUM; bufferId++) {
            l1BufferScaleAOffset_[bufferId] = l1BufferBOffset_[bufferId] + bL1OneBuffer_ * (l1BufNum_ >> 1);
            l1BufferScaleBOffset_[bufferId] = l1BufferScaleAOffset_[bufferId] + scaleAL1OneBuffer_;
            l1BufferBiasOffset_[bufferId] = l1BufferScaleBOffset_[bufferId] + scaleBL1OneBuffer_;
        }
        kL1Iter_ = Blaze::Gemm::CeilDiv(k_, kL1_);
    }

    __aicore__ inline void SetAlltoAllParams(const AlltoAllParams& params)
    {
        rankDim_ = params.rankDim;
        rankKPerRank_ = params.rankKPerRank;
        scaleKPerRank_ = params.scaleKPerRank;
        aRankStride_ = params.aRankStride;
        scaleARankStride_ = params.scaleARankStride;
        localKL1Iter_ = Blaze::Gemm::CeilDiv(rankKPerRank_, kL1_);
    }

    __aicore__ inline void SetGmAddrs(GM_ADDR aGmAddr, GM_ADDR scaleAGmAddr)
    {
        aGmBaseAddr_ = reinterpret_cast<__gm__ AType *>(aGmAddr);
        scaleAGmBaseAddr_ = reinterpret_cast<__gm__ fp8_e8m0_t *>(scaleAGmAddr);
    }

    __aicore__ inline void SetMPos(uint64_t mPos) { mPos_ = mPos; }

    template <typename TensorScaleB>
    __aicore__ inline auto CopyScalesInL1(
        TensorScaleB const& gmScaleB, TileL1L0Param& tileL1L0Param, uint64_t scaleL1BufId,
        uint64_t iter0, uint64_t rankIdx, uint64_t localIter0, __gm__ fp8_e8m0_t *curScaleAGmAddr)
    {
        auto layoutScaleBL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            Blaze::Gemm::CeilDiv(scaleKL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, tileL1L0Param.curN);
        auto tensorScaleBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleBOffset_[scaleL1BufId]), layoutScaleBL1);
        auto CopyScaleGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});

        auto layoutScaleAL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
            tileL1L0Param.curM, Blaze::Gemm::CeilDiv(scaleKPerRank_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
        auto tensorScaleAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, fp8_e8m0_t>(l1BufferScaleAOffset_[scaleL1BufId]), layoutScaleAL1);

        if (localIter0 == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);

            uint64_t rankM = scaleARankStride_ / scaleKPerRank_;
            auto layoutScaleAGM = MakeLayoutScaleAGM{}(rankM, scaleKPerRank_);
            auto gmScaleARank = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(curScaleAGmAddr), layoutScaleAGM);
            auto gmBlockScaleA = gmScaleARank.Slice(
                AscendC::Te::MakeCoord(mPos_, 0),
                AscendC::Te::MakeShape(
                    tileL1L0Param.curM,
                    Blaze::Gemm::CeilDiv(scaleKPerRank_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE));
            AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleAL1, gmBlockScaleA);

            uint64_t globalKOffset = rankIdx * rankKPerRank_;
            uint64_t curScaleKL1 = scaleKL1_;
            if (globalKOffset + curScaleKL1 > k_) {
                curScaleKL1 = k_ - globalKOffset;
            }
            uint64_t globalScaleKOffset = Blaze::Gemm::CeilDiv(globalKOffset, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
            auto gmBlockScaleB = gmScaleB.Slice(
                AscendC::Te::MakeCoord(globalScaleKOffset, 0),
                AscendC::Te::MakeShape(
                    Blaze::Gemm::CeilDiv(curScaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                    tileL1L0Param.curN));
            AscendC::Te::Copy(CopyScaleGM2L1, tensorScaleBL1, gmBlockScaleB);
        }
        return ScalePair<decltype(tensorScaleAL1), decltype(tensorScaleBL1)>{
            tensorScaleAL1, tensorScaleBL1};
    }

    __aicore__ inline auto CopyAInL1(
        TileL1L0Param& tileL1L0Param, uint64_t l1BufId, uint64_t localIter0, __gm__ AType *curAGmAddr)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;

        auto layoutAL1 = MakeLayoutAL1{}(tileL1L0Param.curM, tileL1L0Param.curPadAKL1);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(l1BufferAOffset_[l1BufId]), layoutAL1);

        uint64_t rankM = aRankStride_ / rankKPerRank_;
        auto layoutAGM = MakeLayoutAGM{}(rankM, rankKPerRank_);
        auto gmARank = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(curAGmAddr), layoutAGM);
        auto gmBlockA = gmARank.Slice(
            AscendC::Te::MakeCoord(mPos_, localIter0 * kL1_),
            AscendC::Te::MakeShape(tileL1L0Param.curM, tileL1L0Param.curGmAKL1));
        Blaze::Gemm::Tile::PadMxKAL1::PadZero(tensorAL1, gmBlockA);
        AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
        return tensorAL1;
    }

    template <
        typename TensorScaleAL1, typename TensorScaleBL1, typename TensorAL1, typename TensorBL1,
        typename TensorBiasL1, typename TensorL0C>
    __aicore__ inline void Iterate(
        TileL1L0Param& tileL1L0Param, uint64_t iter0, uint64_t localIter0, TensorScaleAL1& tensorScaleAL1,
        TensorScaleBL1& tensorScaleBL1, TensorAL1& tensorAL1, TensorBL1& tensorBL1,
        TensorBiasL1& tensorBiasL1, TensorL0C& tensorL0C, uint64_t splitKIdx)
    {
        auto scaleKL1Len = Blaze::Gemm::CeilDiv(kL1_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        auto coordScaleKL1 = localIter0 * scaleKL1Len;
        auto tensorBlockScaleBL1 = tensorScaleBL1.Slice(
            AscendC::Te::MakeCoord(coordScaleKL1, 0), AscendC::Te::MakeShape(scaleKL1Len, tileL1L0Param.curN));
        auto tensorBlockScaleAL1 = tensorScaleAL1.Slice(
            AscendC::Te::MakeCoord(0, coordScaleKL1), AscendC::Te::MakeShape(tileL1L0Param.curM, scaleKL1Len));

        uint64_t kL0Iter = Blaze::Gemm::CeilDiv(tileL1L0Param.curGmBKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            auto curKL0 = (iter1 * baseK_ + baseK_ > tileL1L0Param.curPadBKL1) ?
                              (tileL1L0Param.curPadBKL1 - iter1 * baseK_) :
                              baseK_;
            uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
            uint16_t mte1WaitMFlag = (l0PingPong_ & 0x1) + M_MTE1_FLAG_0;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);

            auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>(
                tileL1L0Param.curM, curKL0);
            auto tensorAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
            auto tensorBlockAL1 = tensorAL1.Slice(
                AscendC::Te::MakeCoord(0, iter1 * baseK_), AscendC::Te::MakeShape(tileL1L0Param.curM, curKL0));
            auto CopyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
            AscendC::Te::Copy(CopyL12L0A, tensorAL0, tensorBlockAL1);

            auto CopyL12L0ScaleA = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleA{});
            auto layoutScaleAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZZLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                tileL1L0Param.curM, Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE);
            auto tensorScaleAL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleA, fp8_e8m0_t>(l0Offset >> 4), layoutScaleAL0);
            AscendC::Te::Copy(
                CopyL12L0ScaleA, tensorScaleAL0,
                tensorBlockScaleAL1.Slice(
                    AscendC::Te::MakeCoord(
                        0, iter1 * Blaze::Gemm::CeilDiv(baseK_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE),
                    AscendC::Te::MakeShape(
                        tileL1L0Param.curM, Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE)));

            auto layoutBt = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(
                1UL, Blaze::Gemm::CeilAlign(tileL1L0Param.curN, BLOCK_CUBE));
            auto tensorBt = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, float>(baseN_ * biasBufId_ * sizeof(float)),
                layoutBt);
            if (NeedBias(iter0, iter1, splitKIdx)) {
                auto CopyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
                AscendC::Te::Copy(CopyL12BT, tensorBt, tensorBiasL1);
            }

            auto layoutBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>(
                curKL0, tileL1L0Param.curN);
            auto tensorBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);
            auto tensorBlockBL1 = tensorBL1.Slice(
                AscendC::Te::MakeCoord(iter1 * baseK_, 0), AscendC::Te::MakeShape(curKL0, tileL1L0Param.curN));
            auto CopyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
            AscendC::Te::Copy(CopyL12L0B, tensorBL0, tensorBlockBL1);

            auto CopyL12L0ScaleB = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0ScaleB{});
            auto layoutScaleBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NNLayoutPtn, AscendC::Std::Int<SCALE_C0>>(
                Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, tileL1L0Param.curN);
            auto tensorScaleBL0 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0ScaleB, fp8_e8m0_t>(l0Offset >> 4), layoutScaleBL0);
            AscendC::Te::Copy(
                CopyL12L0ScaleB, tensorScaleBL0,
                tensorBlockScaleBL1.Slice(
                    AscendC::Te::MakeCoord(
                        iter1 * Blaze::Gemm::CeilDiv(baseK_, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, 0),
                    AscendC::Te::MakeShape(
                        Blaze::Gemm::CeilDiv(curKL0, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE, tileL1L0Param.curN)));

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0PingPong_ & 0x1);

            Mmad(tileL1L0Param, iter0, iter1, kL0Iter, curKL0, tensorL0C, tensorAL0, tensorBL0, tensorBt, splitKIdx);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(mte1WaitMFlag);
            l0PingPong_++;
        }
    }

    template <
        typename TensorB, typename TensorScaleB, typename TensorBias,
        typename TensorC>
    __aicore__ inline void operator()(
        TensorB const& gmB, TensorScaleB const& gmScaleB,
        TensorBias const& gmBias, TensorC const& gmC, BlockShape const& singleShape, uint64_t splitKIdx = 0)
    {
        TileL1L0Param tileL1L0Param;
        tileL1L0Param.curM = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        tileL1L0Param.curN = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
        auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            tileL1L0Param.curM, tileL1L0Param.curN);
        auto tensorL0C = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset), layoutL0C);

        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        constexpr uint64_t sizeShift = IsFp4<AType>() ? 1 : 0;

        __gm__ AType *curAGmAddr = aGmBaseAddr_;
        __gm__ fp8_e8m0_t *curScaleAGmAddr = scaleAGmBaseAddr_;
        uint32_t curRankIdx = 0;

        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            uint64_t scaleL1BufId = scaleLoopCnt_ & 1;

            uint64_t rankIdx = iter0 / localKL1Iter_;
            uint64_t localIter0 = iter0 % localKL1Iter_;

            if (rankIdx != curRankIdx) {
                curAGmAddr = aGmBaseAddr_ + (rankIdx * aRankStride_ >> sizeShift);
                curScaleAGmAddr = scaleAGmBaseAddr_ + rankIdx * scaleARankStride_;
                curRankIdx = rankIdx;
            }

            auto scalePair = CopyScalesInL1(gmScaleB, tileL1L0Param, scaleL1BufId, iter0, rankIdx, localIter0, curScaleAGmAddr);
            auto& tensorScaleAL1 = scalePair.scaleA;
            auto& tensorScaleBL1 = scalePair.scaleB;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            biasBufId_ = scaleL1BufId;
            tileL1L0Param.curGmBKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - iter0 * kL1_) : kL1_;
            tileL1L0Param.curPadBKL1 =
                Blaze::Gemm::CeilAlign(tileL1L0Param.curGmBKL1, MXFP_DIVISOR_SIZE);
            tileL1L0Param.curGmAKL1 = tileL1L0Param.curGmBKL1;
            tileL1L0Param.curPadAKL1 = tileL1L0Param.curPadBKL1;

            auto tensorAL1 = CopyAInL1(tileL1L0Param, l1BufId, localIter0, curAGmAddr);

            auto layoutBiasL1 = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn>{}(
                1UL, Blaze::Gemm::CeilAlign(tileL1L0Param.curN, BLOCK_CUBE));
            auto tensorBiasL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(l1BufferBiasOffset_[biasBufId_]), layoutBiasL1);
            if (isBias_ && iter0 == 0 && splitKIdx == 0) {
                AscendC::Te::Copy(copyGM2L1, tensorBiasL1, gmBias);
            }

            uint64_t globalKOffset = rankIdx * rankKPerRank_ + localIter0 * kL1_;
            auto layoutBL1 = MakeLayoutBL1{}(tileL1L0Param.curPadBKL1, tileL1L0Param.curN);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(l1BufferBOffset_[l1BufId]), layoutBL1);
            auto gmBlockB = gmB.Slice(
                AscendC::Te::MakeCoord(globalKOffset, 0),
                AscendC::Te::MakeShape(tileL1L0Param.curGmBKL1, tileL1L0Param.curN));
            Blaze::Gemm::Tile::PadMxKBL1::PadZero(tensorBL1, gmBlockB);
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmBlockB);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            Iterate(tileL1L0Param, iter0, localIter0, tensorScaleAL1, tensorScaleBL1, tensorAL1,
                    tensorBL1, tensorBiasL1, tensorL0C, splitKIdx);

            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            bool isLastOverall = (iter0 + 1 == kL1Iter_);
            if (localIter0 + 1 == localKL1Iter_ || isLastOverall) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(SCALE_BUFFER_FLAG_0 + scaleL1BufId);
                scaleLoopCnt_++;
            }
            abL1LoopCnt_++;
        }
        if (splitKIdx == splitKNum_ - 1) {
            if constexpr (AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<TensorC>, AscendC::Te::Location::UB>) {
                auto CopyL0C2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{});
                AscendC::Te::Copy(CopyL0C2UB.with(AscendC::Te::FixpipeParams(3)), gmC, tensorL0C);
            } else {
                auto CopyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
                AscendC::Te::Copy(CopyL0C2GM.with(AscendC::Te::FixpipeParams(3)), gmC, tensorL0C);
            }
            if (enableL0cPingPong_) {
                l0cPingPong_++;
            }
        }
    }

private:
    __aicore__ inline bool NeedBias(uint64_t kIter0, uint64_t kIter1, uint64_t kIter2)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0 && kIter2 == 0;
    }

    template <typename TensorL0CType, typename TensorAL0, typename TensorBL0, typename TensorBT>
    __aicore__ inline void Mmad(
        TileL1L0Param& tileL1L0Param, uint64_t iter0, uint64_t iter1, uint64_t kL0Iter, uint64_t curKL0,
        TensorL0CType& tensorL0C, TensorAL0 const& tensorAL0, TensorBL0 const& tensorBL0, TensorBT const& tensorBt,
        uint64_t splitKIdx)
    {
        AscendC::Te::MmadParams params;
        params.m = static_cast<uint16_t>(tileL1L0Param.curM);
        params.k = static_cast<uint16_t>(Blaze::Gemm::CeilAlign(curKL0, MXFP_DIVISOR_SIZE));
        params.n = static_cast<uint16_t>(tileL1L0Param.curN);
        params.unitFlag = (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter && (splitKIdx == splitKNum_ - 1)) ?
  	                      FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
  	    params.cmatrixInitVal = (iter0 == 0 && iter1 == 0 && splitKIdx == 0 && !isBias_);
  	    if (NeedBias(iter0, iter1, splitKIdx)) {
            AscendC::Te::Mmad(
                AscendC::Te::MmadAtom<
                    AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0C, tensorAL0, tensorBL0, tensorBt);
        } else {
            AscendC::Te::Mmad(
                AscendC::Te::MmadAtom<
                    AscendC::Te::MmadTraits<AscendC::Te::MmadOperation, Blaze::Gemm::Tile::MmadTraitMX>>{}
                    .with(params),
                tensorL0C, tensorAL0, tensorBL0);
        }
    }

    constexpr static uint16_t SCALE_BUFFER_FLAG_0 = 4;
    constexpr static uint16_t SCALE_BUFFER_FLAG_1 = 5;
    constexpr static uint16_t M_MTE1_FLAG_0 = 4;
    constexpr static uint16_t M_MTE1_FLAG_1 = 5;

    uint16_t biasBufId_ = 0;
    uint64_t biasL1OneBuffer_ = 0UL;
    uint64_t aL1OneBuffer_ = 0UL;
    uint64_t bL1OneBuffer_ = 0UL;
    uint64_t scaleAL1OneBuffer_ = 0UL;
    uint64_t scaleBL1OneBuffer_ = 0UL;
    uint64_t l1BufferAOffset_[4] = {0UL};
    uint64_t l1BufferBOffset_[4] = {0UL};
    uint64_t l1BufferScaleAOffset_[2] = {0UL};
    uint64_t l1BufferScaleBOffset_[2] = {0UL};
    uint64_t l1BufferBiasOffset_[2] = {0UL};
};
#endif
} // namespace Block
} // namespace Gemm
} // namespace Blaze
