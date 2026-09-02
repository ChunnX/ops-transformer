/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_mmad_a8w8_mix.h
 * \brief MIX template quantized matmul block — L0C output to UB via fixpipe (Tensor API).
 *        No scale/bias applied here; dequantization is handled by the epilogue on AIV side.
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
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

using AscendC::HardEvent;
using AscendC::IsSameType;
using AscendC::SetFlag;
using AscendC::WaitFlag;
using AscendC::SetMMLayoutTransform;

template <
    uint64_t FullLoadMode_, bool AtomicAdd_, class ScheduleType_, class AType_, class LayoutA_, class BTypeTuple_,
    class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<
    MatmulWithScaleMix<FullLoadMode_, AtomicAdd_, ScheduleType_>, AType_, LayoutA_, BTypeTuple_, LayoutB_, CType_,
    LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = typename AscendC::Std::tuple_element<0, BTypeTuple_>::type;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutBias = LayoutBias_;
    using L0CType =
        AscendC::Std::conditional_t<AscendC::IsSameType<AType, int8_t>::value, int32_t, float>;
    using DispatchPolicy = MatmulWithScaleMix<FullLoadMode_, AtomicAdd_, ScheduleType_>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        ProblemShape problemShape;
        BlockShape l0TileShape;
        uint64_t kAL1{0};
        uint64_t kBL1{0};
        uint64_t l1BufferNum{0};
        bool enableL0CPingPong{false};
    };

    __aicore__ inline BlockMmad()
    {
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        SetFlag<HardEvent::M_MTE1>(BASE_FLAG + M_MTE1_FLAG_0);
        SetFlag<HardEvent::M_MTE1>(BASE_FLAG + M_MTE1_FLAG_1);
        SetMMLayoutTransform(true);
    }

    __aicore__ inline ~BlockMmad()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0);
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_1);
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2);
        WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_3);
        WaitFlag<HardEvent::M_MTE1>(BASE_FLAG + M_MTE1_FLAG_0);
        WaitFlag<HardEvent::M_MTE1>(BASE_FLAG + M_MTE1_FLAG_1);
        SetMMLayoutTransform(false);
    }

    __aicore__ inline void Init(const Params& params)
    {
        k_ = AscendC::Te::Get<IDX_K_IDX>(params.problemShape);
        uint64_t baseM = AscendC::Te::Get<IDX_M_IDX>(params.l0TileShape);
        baseN_ = AscendC::Te::Get<IDX_N_IDX>(params.l0TileShape);
        baseK_ = AscendC::Te::Get<IDX_K_IDX>(params.l0TileShape);
        l1BufNum_ = params.l1BufferNum;
        enableL0cPingPong_ = params.enableL0CPingPong;
        const uint64_t kAL1 = params.kAL1;
        const uint64_t kBL1 = params.kBL1;
        const uint64_t l1BufferNum = params.l1BufferNum;
        uint64_t aL1OneBuffer = 0UL;
        uint64_t bL1OneBuffer = 0UL;
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == A_FULL_LOAD_MODE) {
            kBL1_ = kBL1;
            kAL1_ = kBL1_;
            kL1_ = kBL1;
            uint64_t mAlign = CeilAlign(baseM, TRANS_A ? static_cast<uint64_t>(C0_SIZE) : BLOCK_CUBE);
            uint64_t kAlign = CeilAlign(k_, TRANS_A ? BLOCK_CUBE : static_cast<uint64_t>(C0_SIZE));
            aL1OneBuffer = mAlign * kAlign;
            bL1OneBuffer = baseN_ * kL1_;
            kL1Iter_ = CeilDiv(k_, kL1_);
        } else {
            if (l1BufferNum == DOUBLE_BUFFER_COUNT) {
                kAL1_ = kAL1;
                kBL1_ = kBL1;
                aL1OneBuffer = baseM * kAL1_;
                bL1OneBuffer = baseN_ * kBL1_;
                kAL1Iter_ = CeilDiv(k_, kAL1_);
                kBL1Iter_ = CeilDiv(k_, kBL1_);
                if (kAL1 == kBL1) {
                    kL1_ = kAL1;
                    kL1Iter_ = CeilDiv(k_, kL1_);
                }
            } else {
                kL1_ = Min(kAL1, kBL1);
                aL1OneBuffer = baseM * kL1_;
                bL1OneBuffer = baseN_ * kL1_;
                kL1Iter_ = CeilDiv(k_, kL1_);
                kAL1_ = kL1_;
                kBL1_ = kL1_;
            }
        }
        GetL1BufferOffset(aL1OneBuffer, bL1OneBuffer);
    }

    template <typename TensorA, typename TensorB, typename TensorC>
    __aicore__ inline void operator()(const TensorA& gmA, const TensorB& gmB, TensorC ubC, BlockShape singleShape)
    {
        constexpr uint64_t halfL0cSize = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t curML1 = AscendC::Te::Get<IDX_M_TILEIDX>(singleShape);
        uint64_t curNL1 = AscendC::Te::Get<IDX_N_TILEIDX>(singleShape);
        AscendC::Te::MmadParams mmadParams;
        mmadParams.m = static_cast<uint16_t>(curML1);
        mmadParams.n = static_cast<uint16_t>(curNL1);
        uint64_t l0cOffset = (l0cPingPong_ & 1) * halfL0cSize;

        uint64_t curML1Aligned = CeilAlign(curML1, static_cast<uint64_t>(2));
        auto layoutL0C = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>(
            curML1Aligned, curNL1);
        auto c1Local = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, L0CType>(l0cOffset), layoutL0C);

        if (kAL1_ == kBL1_) {
            IterateABL1(gmA, gmB, mmadParams, c1Local, curML1, curNL1);
        } else if (kAL1_ > kBL1_) {
            IterateAL1BL1(gmA, gmB, mmadParams, c1Local, curML1, curNL1);
        } else {
            IterateBL1AL1(gmA, gmB, mmadParams, c1Local, curML1, curNL1);
        }

        auto copyL0C2UB = AscendC::Te::MakeCopy(
            AscendC::Te::CopyL0C2UB{}, Blaze::Gemm::Tile::CopyL0C2UBTraitMixSplitM{});
        AscendC::Te::Copy(copyL0C2UB.with(AscendC::Te::FixpipeParams(FINAL_ACCUMULATION)), ubC, c1Local);

        if (enableL0cPingPong_) {
            l0cPingPong_++;
        }
    }

private:
    __aicore__ inline void GetL1BufferOffset(uint64_t aL1OneBuffer, uint64_t bL1OneBuffer)
    {
        constexpr uint64_t halfL1Size = AscendC::TOTAL_L1_SIZE >> 1;
        uint64_t l1HalfBufNum = l1BufNum_ >> 1;
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
            for (uint16_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                uint64_t l1Offset = halfL1Size * (bufferId & 1);
                l1BufferAOffset_[bufferId] = l1Offset + aL1OneBuffer * (bufferId >> 1);
                l1BufferBOffset_[bufferId] =
                    l1Offset + aL1OneBuffer * l1HalfBufNum + bL1OneBuffer * (bufferId >> 1);
            }
        } else {
            l1BufferAOffset_[0] = bL1OneBuffer * l1HalfBufNum;
            uint64_t b1Offset = l1BufferAOffset_[0] + aL1OneBuffer >= halfL1Size
                                    ? l1BufferAOffset_[0] + aL1OneBuffer
                                    : halfL1Size;
            for (uint16_t bufferId = 0; bufferId < l1BufNum_; bufferId++) {
                l1BufferBOffset_[bufferId] = b1Offset * (bufferId & 1) + bL1OneBuffer * (bufferId >> 1);
            }
        }
    }

    template <typename TensorAL1, typename TensorBL1, typename C1Tensor>
    __aicore__ inline void Iterate(
        AscendC::Te::MmadParams& mmadParams, const TensorAL1& tensorAL1, const TensorBL1& tensorBL1, C1Tensor& c1Local,
        uint64_t curML1, uint64_t curNL1, uint64_t curInnerKL1, uint64_t aKPrefix, uint64_t bKPrefix,
        bool isL1LastRound, uint32_t initMode, uint64_t kL1OuterIdx, uint64_t kL1InnerIdx)
    {
        constexpr uint64_t halfL0Size = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        uint64_t kL0Iter = CeilDiv(curInnerKL1, baseK_);
        for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
            uint64_t curKL0 = (iter1 == kL0Iter - 1) ? (curInnerKL1 - iter1 * baseK_) : baseK_;
            uint64_t l0Offset = halfL0Size * (l0PingPong_ & 0x1);

            auto layoutAL0 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(
                    curML1, curKL0);
            auto layoutBL0 =
                AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(
                    curKL0, curNL1);
            auto l0aLocal = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
            auto l0bLocal = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);

            WaitFlag<HardEvent::M_MTE1>(BASE_FLAG + (l0PingPong_ & 0x1));

            auto aL1Sub = tensorAL1.Slice(
                AscendC::Te::MakeCoord(0, aKPrefix + iter1 * baseK_), AscendC::Te::MakeShape(curML1, curKL0));
            AscendC::Te::Copy(copyL12L0A, l0aLocal, aL1Sub);

            auto bL1Sub = tensorBL1.Slice(
                AscendC::Te::MakeCoord(bKPrefix + iter1 * baseK_, 0), AscendC::Te::MakeShape(curKL0, curNL1));
            AscendC::Te::Copy(copyL12L0B, l0bLocal, bL1Sub);

            SetFlag<HardEvent::MTE1_M>(l0PingPong_ & 0x1);
            WaitFlag<HardEvent::MTE1_M>(l0PingPong_ & 0x1);

            mmadParams.k = static_cast<uint16_t>(curKL0);
            mmadParams.unitFlag = (isL1LastRound && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
            mmadParams.cmatrixInitVal = (initMode == L0_INIT_MODE_ABL1)
                                            ? (kL1OuterIdx == 0 && iter1 == 0)
                                            : (kL1OuterIdx == 0 && kL1InnerIdx == 0 && iter1 == 0);

            constexpr auto mmadAtom =
                AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});
            AscendC::Te::Mmad(mmadAtom.with(mmadParams), c1Local, l0aLocal, l0bLocal);

            SetFlag<HardEvent::M_MTE1>(BASE_FLAG + (l0PingPong_ & 0x1));
            l0PingPong_++;
        }
    }

    template <typename TensorA, typename TensorB, typename C1Tensor>
    __aicore__ inline void IterateABL1(
        const TensorA& gmA, const TensorB& gmB, AscendC::Te::MmadParams& mmadParams, C1Tensor& c1Local,
        uint64_t curML1, uint64_t curNL1)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});

        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            uint64_t curKL1 = (iter0 == kL1Iter_ - 1) ? (k_ - iter0 * kL1_) : kL1_;
            uint16_t l1BufId = abL1LoopCnt_ & (l1BufNum_ - 1);
            WaitFlag<HardEvent::MTE1_MTE2>(l1BufId);

            uint64_t offsetAL1 = l1BufferAOffset_[l1BufId];
            auto layoutAL1 = MakeLayoutAL1{}(curML1, curKL1);
            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAL1), layoutAL1);
            if constexpr (DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
                auto gmTileA = gmA.Slice(
                    AscendC::Te::MakeCoord(0UL, iter0 * kAL1_), AscendC::Te::MakeShape(curML1, curKL1));
                AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);
            } else {
                if (abL1LoopCnt_ < kL1Iter_) {
                    offsetAL1 = l1BufferAOffset_[0] + iter0 * kL1_ *
                                 CeilAlign(curML1, TRANS_A ? C0_SIZE : BLOCK_CUBE);
                    tensorAL1 = AscendC::Te::MakeTensor(
                        AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAL1), layoutAL1);
                    auto gmTileA = gmA.Slice(
                        AscendC::Te::MakeCoord(0UL, iter0 * kL1_), AscendC::Te::MakeShape(curML1, curKL1));
                    AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);
                }
            }

            uint64_t offsetBL1 = l1BufferBOffset_[l1BufId];
            auto layoutBL1 = MakeLayoutBL1{}(curKL1, curNL1);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(offsetBL1), layoutBL1);
            auto gmTileB = gmB.Slice(
                AscendC::Te::MakeCoord(iter0 * kBL1_, 0UL), AscendC::Te::MakeShape(curKL1, curNL1));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);

            SetFlag<HardEvent::MTE2_MTE1>(l1BufId);
            WaitFlag<HardEvent::MTE2_MTE1>(l1BufId);

            Iterate(mmadParams, tensorAL1, tensorBL1, c1Local, curML1, curNL1, curKL1, 0UL, 0UL,
                    iter0 == kL1Iter_ - 1, L0_INIT_MODE_ABL1, iter0, 0UL);

            SetFlag<HardEvent::MTE1_MTE2>(l1BufId);
            abL1LoopCnt_++;
        }
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == A_FULL_LOAD_MODE) {
            abL1LoopCnt_ = 0;
        }
    }

    template <typename TensorA, typename TensorB, typename C1Tensor>
    __aicore__ inline void IterateAL1BL1(
        const TensorA& gmA, const TensorB& gmB, AscendC::Te::MmadParams& mmadParams, C1Tensor& c1Local,
        uint64_t curML1, uint64_t curNL1)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});

        for (uint64_t kAIter = 0; kAIter < kAL1Iter_; ++kAIter) {
            uint64_t curKAL1 = (kAIter == kAL1Iter_ - 1) ? (k_ - kAIter * kAL1_) : kAL1_;
            WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + aPingPongId_);

            uint64_t offsetAL1 = l1BufferAOffset_[aPingPongId_];
            auto layoutAL1 = MakeLayoutAL1{}(curML1, curKAL1);
            auto tensorAL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAL1), layoutAL1);
            auto gmTileA = gmA.Slice(
                AscendC::Te::MakeCoord(0UL, kAIter * kAL1_), AscendC::Te::MakeShape(curML1, curKAL1));
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);

            SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + aPingPongId_);
            WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + aPingPongId_);

            uint64_t kBL1IterLocal = CeilDiv(curKAL1, kBL1_);
            for (uint64_t kBIter = 0; kBIter < kBL1IterLocal; ++kBIter) {
                uint64_t curKBL1 = (kBIter == kBL1IterLocal - 1) ? (curKAL1 - kBIter * kBL1_) : kBL1_;
                WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bPingPongId_);

                uint64_t offsetBL1 = l1BufferBOffset_[bPingPongId_];
                auto layoutBL1 = MakeLayoutBL1{}(curKBL1, curNL1);
                auto tensorBL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(offsetBL1), layoutBL1);
                uint64_t bKStart = kAIter * kAL1_ + kBIter * kBL1_;
                auto gmTileB = gmB.Slice(
                    AscendC::Te::MakeCoord(bKStart, 0UL), AscendC::Te::MakeShape(curKBL1, curNL1));
                AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);

                SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + bPingPongId_);
                WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + bPingPongId_);

                uint64_t aKPrefix = kBIter * kBL1_;
                Iterate(mmadParams, tensorAL1, tensorBL1, c1Local, curML1, curNL1, curKBL1, aKPrefix, 0UL,
                        (kAIter == kAL1Iter_ - 1) && (kBIter == kBL1IterLocal - 1), L0_INIT_MODE_SPLIT, kAIter, kBIter);

                SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + bPingPongId_);
                bPingPongId_ = bPingPongId_ ^ 1;
                abL1LoopCnt_++;
            }
            SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + aPingPongId_);
            aPingPongId_ = aPingPongId_ ^ 1;
        }
        abL1LoopCnt_ = 0;
    }

    template <typename TensorA, typename TensorB, typename C1Tensor>
    __aicore__ inline void IterateBL1AL1(
        const TensorA& gmA, const TensorB& gmB, AscendC::Te::MmadParams& mmadParams, C1Tensor& c1Local,
        uint64_t curML1, uint64_t curNL1)
    {
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});

        for (uint64_t kBIter = 0; kBIter < kBL1Iter_; ++kBIter) {
            uint64_t curKBL1 = (kBIter == kBL1Iter_ - 1) ? (k_ - kBIter * kBL1_) : kBL1_;
            WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + bPingPongId_);

            uint64_t offsetBL1 = l1BufferBOffset_[bPingPongId_];
            auto layoutBL1 = MakeLayoutBL1{}(curKBL1, curNL1);
            auto tensorBL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(offsetBL1), layoutBL1);
            auto gmTileB = gmB.Slice(
                AscendC::Te::MakeCoord(kBIter * kBL1_, 0UL), AscendC::Te::MakeShape(curKBL1, curNL1));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);

            SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + bPingPongId_);
            WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_0 + bPingPongId_);

            uint64_t kAL1IterLocal = CeilDiv(curKBL1, kAL1_);
            for (uint64_t kAIter = 0; kAIter < kAL1IterLocal; ++kAIter) {
                uint64_t curKAL1 = (kAIter == kAL1IterLocal - 1) ? (curKBL1 - kAIter * kAL1_) : kAL1_;
                WaitFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aPingPongId_);

                uint64_t offsetAL1 = l1BufferAOffset_[aPingPongId_];
                auto layoutAL1 = MakeLayoutAL1{}(curML1, curKAL1);
                auto tensorAL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(offsetAL1), layoutAL1);
                uint64_t aKStart = kBIter * kBL1_ + kAIter * kAL1_;
                auto gmTileA = gmA.Slice(
                    AscendC::Te::MakeCoord(0UL, aKStart), AscendC::Te::MakeShape(curML1, curKAL1));
                AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);

                SetFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + aPingPongId_);
                WaitFlag<HardEvent::MTE2_MTE1>(INPUT_BUFFER_FLAG_2 + aPingPongId_);

                uint64_t bKPrefix = kAIter * kAL1_;
                Iterate(mmadParams, tensorAL1, tensorBL1, c1Local, curML1, curNL1, curKAL1, 0UL, bKPrefix,
                        (kBIter == kBL1Iter_ - 1) && (kAIter == kAL1IterLocal - 1), L0_INIT_MODE_SPLIT, kBIter, kAIter);

                SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_2 + aPingPongId_);
                aPingPongId_ = aPingPongId_ ^ 1;
                abL1LoopCnt_++;
            }
            SetFlag<HardEvent::MTE1_MTE2>(INPUT_BUFFER_FLAG_0 + bPingPongId_);
            bPingPongId_ = bPingPongId_ ^ 1;
        }
        abL1LoopCnt_ = 0;
    }

    uint64_t k_{0};
    uint64_t l1BufNum_{1};
    uint64_t kL1Iter_{0};
    uint64_t kAL1Iter_{0};
    uint64_t kBL1Iter_{0};
    uint64_t kL1_{1};
    uint64_t kAL1_{1};
    uint64_t kBL1_{1};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint16_t aPingPongId_{0};
    uint16_t bPingPongId_{0};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t l1BufferAOffset_[QUADRUPLE_BUFFER_COUNT] = {0UL};
    uint64_t l1BufferBOffset_[QUADRUPLE_BUFFER_COUNT] = {0UL};
    bool enableL0cPingPong_{false};

    static constexpr uint32_t L0_INIT_MODE_ABL1 = 0U;
    static constexpr uint32_t L0_INIT_MODE_SPLIT = 1U;
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr int32_t C0_SIZE = AscendC::Te::C0_ELEMENT<AType>;
    static constexpr uint16_t BASE_FLAG = 4;
    static constexpr uint16_t M_MTE1_FLAG_0 = 0;
    static constexpr uint16_t M_MTE1_FLAG_1 = 1;
    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Std::Int<C0_SIZE>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE>>>;
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
