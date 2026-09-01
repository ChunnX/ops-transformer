/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

constexpr uint64_t L1_STAGES = 2UL;
constexpr uint64_t L1_BUFFER_MASK = L1_STAGES - 1UL;

template <class AType_, class LayoutA_, class BType_, class LayoutB_, class L0CType_, class LayoutC_, class BiasType_,
          class LayoutBias_>
class BlockMmad<MatmulEmuSplitWeightPolicy, AType_, LayoutA_, BType_, LayoutB_, L0CType_, LayoutC_, BiasType_,
                LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = L0CType_;
    using L0CType = L0CType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using DispatchPolicy = MatmulEmuSplitWeightPolicy;
    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool TRANS_A = Blaze::Gemm::IsTrans<LayoutA_>::value;
    static constexpr bool TRANS_B = Blaze::Gemm::IsTrans<LayoutB_>::value;

    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<AType_>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType_>>>;
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType_>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<BType_>>>;

    struct Params {
        GM_ADDR xGmAddr{nullptr};
        GM_ADDR wHighGmAddr{nullptr};
        GM_ADDR wLowGmAddr{nullptr};
        uint64_t k{0UL};
        uint64_t kL1{0UL};
        uint32_t baseM{0};
        uint32_t baseN{0};
        uint32_t baseK{0};
        uint32_t usedCoreNum{0};
    };

    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            AscendC::SetMMLayoutTransform(true);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        k_ = params.k;
        baseM_ = params.baseM;
        baseN_ = params.baseN;
        baseK_ = params.baseK;
        kL1_ = params.kL1;
        l1LoopCnt_ = 0;
        l0PingPong_ = 0;

        uint64_t aL1OneSize = baseM_ * kL1_ * sizeof(AType_);
        uint64_t bL1OneSize = kL1_ * baseN_ * sizeof(BType_);
        constexpr uint64_t slotSize = AscendC::TOTAL_L1_SIZE / DOUBLE_BUFFER_COUNT;

        // 2buffer时：|APing,BLowPing,BHighPing---|APong,BLowPong,BHighPong---|
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            uint64_t base = slotSize * i;
            bufMgr_.InitAL1(i, base, i);
            bufMgr_.InitBL1(i, base + aL1OneSize, i);
            bufMgr_.InitBL1(i + L1_STAGES, base + aL1OneSize + bL1OneSize, i + L1_STAGES);
        }
        bufMgr_.InitL0();
        bufMgr_.InitL0C();
    }

    template <typename TensorA, typename TensorBHigh, typename TensorBLow, typename TensorCHigh, typename TensorCLow>
    __aicore__ inline void operator()(TensorA gmA, TensorBHigh gmBHigh, TensorBLow gmBLow, TensorCHigh ubBlockCHigh,
                                      TensorCLow ubBlockCLow, BlockShape singleShape, bool targetSubBlockId = false)
    {
        uint64_t kL1Iter = Blaze::Gemm::CeilDiv(k_, kL1_);

        auto gmTensors = AscendC::Std::make_tuple(gmA, gmBHigh, gmBLow);
        auto ubTensors = AscendC::Std::make_tuple(ubBlockCHigh, ubBlockCLow);

        const auto& l0cSlot1 = bufMgr_.GetL0CSlot(0);
        const auto& l0cSlot2 = bufMgr_.GetL0CSlot(1);
        auto curM = AscendC::Te::Get<MNK_M>(singleShape);
        auto curN = AscendC::Te::Get<MNK_N>(singleShape);
        auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<C0_SIZE_L0C>>{}(
            curM, curN);
        auto tensorL0C1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, L0CType_>(l0cSlot1.Addr()), layoutL0C);
        auto tensorL0C2 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, L0CType_>(l0cSlot2.Addr()), layoutL0C);
        auto l0CTensors = AscendC::Std::make_tuple(tensorL0C1, tensorL0C2);

        for (uint64_t iter0 = 0; iter0 < kL1Iter; ++iter0) {
            uint64_t l1BufId = l1LoopCnt_ & L1_BUFFER_MASK;
            uint64_t kL1Offset = iter0 * kL1_;
            uint64_t curKL1 = (iter0 + 1 == kL1Iter) ? (k_ - kL1Offset) : kL1_;

            const auto& aL1Slot = bufMgr_.GetL1ASlot(l1BufId);
            const auto& bLowL1Slot = bufMgr_.GetL1BSlot(l1BufId);
            const auto& bHighL1Slot = bufMgr_.GetL1BSlot(l1BufId + L1_STAGES);

            auto l1Slots = AscendC::Std::make_tuple(aL1Slot, bLowL1Slot, bHighL1Slot);
            auto l1Shape = BlockShape{curM, curN, static_cast<int64_t>(curKL1), static_cast<int64_t>(iter0)};
            auto l1Tensors = CopyL1FromGM(gmTensors, l1Shape, l1Slots);

            uint64_t kL0Iter = Blaze::Gemm::CeilDiv(curKL1, baseK_);
            for (uint16_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                auto kL0Offset = iter1 * baseK_;
                auto curKL0 = (kL0Offset + baseK_ > curKL1) ? (curKL1 - kL0Offset) : baseK_;
                const auto& l0Slot = bufMgr_.GetL0Slot(l0PingPong_ & 0x1);

                auto l0Shape = BlockShape{curM, curN, static_cast<int64_t>(curKL0), static_cast<int64_t>(kL0Offset)};
                auto l0Tensors = CopyL0FromL1(l1Tensors, l0Shape, l0Slot, l1Slots);

                bool isFirstKBeat = (iter0 == 0 && iter1 == 0);
                bool isLastKBeat = (iter0 + 1 == kL1Iter && iter1 + 1 == kL0Iter);
                {
                    auto l0Lock = l0Slot.LockM();
                    Compute(l0CTensors, l0Tensors, l0Shape, isFirstKBeat, isLastKBeat);
                }
                l0PingPong_++;
            }
            l1LoopCnt_++;
        }

        CopyL0CToUB(l0CTensors, ubTensors, targetSubBlockId);
    }

private:
    uint64_t k_{0UL};
    uint64_t kL1_{0UL};
    uint64_t baseM_{0UL};
    uint64_t baseN_{0UL};
    uint64_t baseK_{0UL};
    uint64_t l0PingPong_{0UL};
    uint64_t l1LoopCnt_{0UL};

    // Weight(B)切分成wHigh和wLow两部分, L1ASlots = L1_STAGES, L1BSlots = L1_STAGES * 2
    Blaze::Gemm::BufferManager<L1_STAGES, L1_STAGES * 2, DOUBLE_BUFFER_COUNT> bufMgr_;

    template <typename GmTensors, typename SlotsTuple>
    __aicore__ inline auto CopyL1FromGM(const GmTensors& gmTensors, const BlockShape& blockShape,
                                        const SlotsTuple& slotsTuple)
    {
        const auto& gmA = AscendC::Te::Get<0>(gmTensors);
        const auto& gmBHigh = AscendC::Te::Get<1>(gmTensors);
        const auto& gmBLow = AscendC::Te::Get<2>(gmTensors);

        auto curM = AscendC::Te::Get<MNK_M>(blockShape);
        auto curN = AscendC::Te::Get<MNK_N>(blockShape);
        auto curKL1 = AscendC::Te::Get<MNK_K>(blockShape);
        auto kIdx = static_cast<uint64_t>(AscendC::Te::Get<3>(blockShape));

        const auto& aL1Slot = AscendC::Te::Get<0>(slotsTuple);
        const auto& bLowL1Slot = AscendC::Te::Get<1>(slotsTuple);
        const auto& bHighL1Slot = AscendC::Te::Get<2>(slotsTuple);

        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});

        auto layoutAL1 = MakeLayoutAL1{}(curM, curKL1);
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType_>(aL1Slot.Addr()), layoutAL1);
        auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(0, kIdx * kL1_), AscendC::Te::MakeShape(curM, curKL1));
        {
            auto lock = aL1Slot.LockMte2();
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmBlockA);
        }

        auto layoutWL1 = MakeLayoutBL1{}(curKL1, curN);
        auto tensorBLowL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType_>(bLowL1Slot.Addr()), layoutWL1);
        auto gmBlockBLow = gmBLow.Slice(AscendC::Te::MakeCoord(kIdx * kL1_, 0), AscendC::Te::MakeShape(curKL1, curN));
        {
            auto lock = bLowL1Slot.LockMte2();
            AscendC::Te::Copy(copyGM2L1, tensorBLowL1, gmBlockBLow);
        }

        auto tensorBHighL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType_>(bHighL1Slot.Addr()), layoutWL1);
        auto gmBlockBHigh = gmBHigh.Slice(AscendC::Te::MakeCoord(kIdx * kL1_, 0), AscendC::Te::MakeShape(curKL1, curN));
        {
            auto lock = bHighL1Slot.LockMte2();
            AscendC::Te::Copy(copyGM2L1, tensorBHighL1, gmBlockBHigh);
        }

        return AscendC::Std::make_tuple(tensorAL1, tensorBLowL1, tensorBHighL1);
    }

    template <typename L1Tensors, typename SlotsTuple>
    __aicore__ inline auto CopyL0FromL1(const L1Tensors& l1Tensors, const BlockShape& blockShape,
                                        const Blaze::Gemm::BufferSlot& l0Slot, const SlotsTuple& slotsTuple)
    {
        const auto& tensorAL1 = AscendC::Te::Get<0>(l1Tensors);
        const auto& tensorBLowL1 = AscendC::Te::Get<1>(l1Tensors);
        const auto& tensorBHighL1 = AscendC::Te::Get<2>(l1Tensors);

        auto curM = AscendC::Te::Get<MNK_M>(blockShape);
        auto curN = AscendC::Te::Get<MNK_N>(blockShape);
        auto curKL0 = AscendC::Te::Get<MNK_K>(blockShape);
        auto kL0Offset = static_cast<uint64_t>(AscendC::Te::Get<3>(blockShape));

        const auto& aL1Slot = AscendC::Te::Get<0>(slotsTuple);
        const auto& bLowL1Slot = AscendC::Te::Get<1>(slotsTuple);
        const auto& bHighL1Slot = AscendC::Te::Get<2>(slotsTuple);

        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});

        auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn,
                                                      AscendC::Te::LayoutTraitDefault<AType_>>(curM, curKL0);
        auto tensorAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType_>(l0Slot.Addr()), layoutAL0);
        auto tensorBlockAL1 = tensorAL1.Slice(AscendC::Te::MakeCoord(0, kL0Offset),
                                              AscendC::Te::MakeShape(curM, curKL0));
        {
            auto l1LockA = aL1Slot.LockMte1();
            auto l0Lock = l0Slot.LockMte1();
            AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);
        }

        auto layoutBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn,
                                                      AscendC::Te::LayoutTraitDefault<BType_>>(curKL0, curN);
        auto tensorBL01 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType_>(l0Slot.Addr()), layoutBL0);
        auto tensorBlockBL1Low = tensorBLowL1.Slice(AscendC::Te::MakeCoord(kL0Offset, 0),
                                                    AscendC::Te::MakeShape(curKL0, curN));
        {
            auto l1LockBLow = bLowL1Slot.LockMte1();
            auto l0Lock = l0Slot.LockMte1();
            AscendC::Te::Copy(copyL12L0B, tensorBL01, tensorBlockBL1Low);
        }

        uint64_t l0bOneBufBytes = Blaze::Gemm::CeilAlign(static_cast<uint64_t>(curKL0),
                                                         static_cast<uint64_t>(C0_SIZE_L0C)) *
                                  Blaze::Gemm::CeilAlign(static_cast<uint64_t>(curN),
                                                         static_cast<uint64_t>(C0_SIZE_L0C)) *
                                  sizeof(BType_);
        auto tensorBL02 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType_>(l0Slot.Addr() + l0bOneBufBytes), layoutBL0);
        auto tensorBlockBL1High = tensorBHighL1.Slice(AscendC::Te::MakeCoord(kL0Offset, 0),
                                                      AscendC::Te::MakeShape(curKL0, curN));
        {
            auto l1LockBHigh = bHighL1Slot.LockMte1();
            auto l0Lock = l0Slot.LockMte1();
            AscendC::Te::Copy(copyL12L0B, tensorBL02, tensorBlockBL1High);
        }

        return AscendC::Std::make_tuple(tensorAL0, tensorBL01, tensorBL02);
    }

    template <typename L0CTensors, typename L0Tensors>
    __aicore__ inline void Compute(const L0CTensors& l0CTensors, const L0Tensors& l0Tensors,
                                   const BlockShape& blockShape, bool isFirstKBeat, bool isLastKBeat)
    {
        const auto& tensorL0C1 = AscendC::Te::Get<0>(l0CTensors);
        const auto& tensorL0C2 = AscendC::Te::Get<1>(l0CTensors);
        const auto& tensorAL0 = AscendC::Te::Get<0>(l0Tensors);
        const auto& tensorBL01 = AscendC::Te::Get<1>(l0Tensors);
        const auto& tensorBL02 = AscendC::Te::Get<2>(l0Tensors);

        auto curM = AscendC::Te::Get<MNK_M>(blockShape);
        auto curN = AscendC::Te::Get<MNK_N>(blockShape);
        auto curKL0 = AscendC::Te::Get<MNK_K>(blockShape);

        constexpr auto mmadAtom = AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});
        uint8_t mmadUnitFlag = isLastKBeat ? Blaze::Gemm::FINAL_ACCUMULATION : Blaze::Gemm::NON_FINAL_ACCUMULATION;
        bool mmadCmatrixInitVal = isFirstKBeat;

        AscendC::Te::MmadParams mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                                           static_cast<uint16_t>(curKL0), mmadUnitFlag, mmadCmatrixInitVal};
        AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C1, tensorAL0, tensorBL01);
        AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C2, tensorAL0, tensorBL02);
    }

    template <typename L0CTensors, typename UbTensors>
    __aicore__ inline void CopyL0CToUB(const L0CTensors& l0CTensors, const UbTensors& ubTensors, bool targetSubBlockId)
    {
        const auto& tensorL0C1 = AscendC::Te::Get<0>(l0CTensors);
        const auto& tensorL0C2 = AscendC::Te::Get<1>(l0CTensors);
        const auto& ubBlockCHigh = AscendC::Te::Get<0>(ubTensors);
        const auto& ubBlockCLow = AscendC::Te::Get<1>(ubTensors);

        AscendC::Te::FixpipeParams fixpParams{Blaze::Gemm::FINAL_ACCUMULATION, targetSubBlockId};
        auto copyL0C2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{});
        AscendC::Te::Copy(copyL0C2UB.with(fixpParams), ubBlockCLow, tensorL0C1);
        AscendC::Te::Copy(copyL0C2UB.with(fixpParams), ubBlockCHigh, tensorL0C2);
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
