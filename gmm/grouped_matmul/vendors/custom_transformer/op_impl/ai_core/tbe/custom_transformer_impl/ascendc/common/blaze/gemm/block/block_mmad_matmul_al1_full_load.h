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
 * \file block_mmad_matmul_al1_full_load.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <
    uint64_t FullLoadMode_, uint64_t FusedOpType_, class KernelSchedule_, class AType_, class LayoutA_, class BType_,
    class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<
    MatmulMultiBlockAFullLoad<FullLoadMode_, FusedOpType_, KernelSchedule_>, AType_, LayoutA_, BType_, LayoutB_, CType_,
    LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy = MatmulMultiBlockAFullLoad<FullLoadMode_, FusedOpType_, KernelSchedule_>;
    using TupleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using TripleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;

    // TRANS_A and TRANS_B
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ_FORMAT = IsWeightNz<LayoutB>::value;
    // AL1 Layout
    using MakeLayoutAL1 = AscendC::Std::conditional_t<
        TRANS_A, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>>;
    // BL1 Layout
    using MakeLayoutBL1 = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>>;

    // kernel params
    struct Params {
        GM_ADDR aGmAddr{nullptr};
        GM_ADDR bGmAddr{nullptr};
        GM_ADDR cGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        uint64_t oriK{0};
        uint64_t mL1{0};
        uint64_t kL1{0};
        uint32_t mL0{0};
        uint32_t nL0{0};
        uint32_t kL0{0};
        uint32_t l1Stages{1};
        uint16_t l0cStages{1};
    };

public:
    __aicore__ inline BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
            AscendC::SetMMLayoutTransform(true);
        }
    }

    __aicore__ inline ~BlockMmad()
    {
        if ASCEND_IS_NOT_AIV {
            for (uint16_t i = 0; i < MTE1_MTE2_EVENT_ID_NUM; i++) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(i);
            }
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(FIRST_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SIXTH_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(SEVENTH_FLAG);
            AscendC::SetMMLayoutTransform(false);
        }
    }

    __aicore__ inline void Init(const Params& params)
    {
        oriK_ = params.oriK;
        mL1_ = params.mL1;
        kL1_ = params.kL1;
        baseM_ = params.mL0;
        baseN_ = params.nL0;
        baseK_ = params.kL0;
        isBias_ = params.biasGmAddr != nullptr;
        l1Stages_ = params.l1Stages;
        enableL0cPingPong_ = params.l0cStages > 1;
        kAlign_ = Blaze::Gemm::CeilAlign(oriK_, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        // A全载
        aL1OneBuffer_ = mL1_ * kAlign_ * sizeof(AType);
        bL1OneBuffer_ = baseN_ * kL1_ * sizeof(BType);
        biasL1OneBuffer_ = baseN_ * sizeof(BiasType);
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
        isAL1Loaded_ = false;
        // 2 buffer: A | B0 B1 | Bias0 Bias1
        // 4 buffer: A | B0 B1 B2 B3 | Bias0 Bias1 Bias2 Bias3
        for (auto i = 0; i < l1Stages_; ++i) {
            bL1Buffer_[i] = aL1OneBuffer_ + bL1OneBuffer_ * i;
            biasL1Buffer_[i] = aL1OneBuffer_ + bL1OneBuffer_ * l1Stages_ + biasL1OneBuffer_ * i;
        }
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(
        TensorA& gmA, TensorB& gmB, TensorBias& gmBias, TensorC& gmC, TupleShape& blockShape)
    {
        static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
        static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;
        uint64_t curM = AscendC::Te::Get<MNK_M>(blockShape);
        uint64_t curN = AscendC::Te::Get<MNK_N>(blockShape);

        // A GM->L1 A全载模式
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto layoutAL1 = MakeLayoutAL1{}(curM, oriK_);
        auto tensorAL1 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(0UL), layoutAL1);
        if (!isAL1Loaded_) {
            AscendC::Te::Copy(copyGM2L1, tensorAL1, gmA);
            isAL1Loaded_ = true;
        }

        baseM_ = Min(curM, baseM_);
        mL1Iter_ = CeilDiv(curM, baseM_);
        kL1_ = Min(oriK_, kL1_);
        kL1Iter_ = CeilDiv(oriK_, kL1_);
        for (uint64_t iterM = 0; iterM < mL1Iter_; ++iterM) {
            auto tileM = (iterM + 1 == mL1Iter_) ? (curM - baseM_ * iterM) : baseM_;
            uint64_t l0cOffset = (l0cPingPong_ & 0x1) * HALF_L0C_SIZE;
            // LoC搬出
            auto layoutL0C =
                AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<16>>{}(tileM, curN);
            auto tensorL0C = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset), layoutL0C);

            for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
                auto curKL1 = (iter0 + 1 == kL1Iter_) ? (oriK_ - kL1_ * iter0) : kL1_;
                // 普通模板-2buffer-AL1搬入偏移位置：*AL1Ping*-BL1Ping-BiasPing|*AL1Pong*-BL1Pong-BiasPong
                uint64_t l1BufId = abL1LoopCnt_ & (l1Stages_ - 1);
                uint64_t btBufId = abL1LoopCnt_ & 0x1;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);

                // GM->L1
                TripleShape l1Shape{tileM, curN, curKL1};
                auto l1TensorTuple = CopyL1FromGM(gmB, gmBias, l1Shape, l1BufId, iter0);
                auto tensorBL1 = AscendC::Te::Get<0>(l1TensorTuple);
                auto tensorBiasL1 = AscendC::Te::Get<1>(l1TensorTuple);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

                uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
                for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                    uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                    uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                    uint64_t mte1Flag = ((l0PingPong_ & 0x1) + SIXTH_FLAG);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));

                    uint64_t aL1MOffset = iterM * baseM_;
                    uint64_t aL1KOffset = iter0 * kL1_ + iter1 * baseK_;
                    uint64_t bL1KOffset = iter1 * baseK_;
                    uint64_t bL1NOffset = 0UL;

                    // A L1->L0
                    TripleShape l0Shape{tileM, curN, curK0};
                    bool needBias = NeedProcessBias(iter0, iter1);
                    auto l0TensorTuple = CopyL0FromL1(
                        tensorAL1, tensorBL1, tensorBiasL1, l0Shape, l0Offset, needBias, btBufId, aL1MOffset,
                        aL1KOffset, bL1KOffset, bL1NOffset);
                    auto tensorAL0 = AscendC::Te::Get<0>(l0TensorTuple);
                    auto tensorBL0 = AscendC::Te::Get<1>(l0TensorTuple);
                    auto tensorBiasL0 = AscendC::Te::Get<2>(l0TensorTuple);

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<uint16_t>(mte1Flag));

                    bool initCmatrix = iter0 == 0 && iter1 == 0 && !isBias_;
                    uint8_t unitFlag =
                        ((iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION);
                    Compute(tensorAL0, tensorBL0, tensorBiasL0, tensorL0C, l0Shape, needBias, unitFlag, initCmatrix);

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));
                    l0PingPong_++;
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
                abL1LoopCnt_++;
            }
            // 数据搬出到GM
            AscendC::Te::FixpipeParams fixpParams{FINAL_ACCUMULATION};
            auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
            auto tensorgmC = gmC.Slice(AscendC::Te::MakeCoord(iterM * baseM_, 0), AscendC::Te::MakeShape(tileM, curN));
            AscendC::Te::Copy(copyL0C2GM.with(fixpParams), tensorgmC, tensorL0C);

            if (enableL0cPingPong_) {
                l0cPingPong_++;
            }
        }
    }

private:
    __aicore__ inline bool NeedProcessBias(uint64_t kIter0, uint64_t kIter1)
    {
        return isBias_ && kIter0 == 0 && kIter1 == 0;
    }

    template <typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL1FromGM(
        const TensorB& tensorB, const TensorBias& tensorBias, const TripleShape& l1Shape, uint64_t l1BufId, uint64_t kIdx)
    {
        uint64_t curN = AscendC::Te::Get<1>(l1Shape);
        uint64_t curKL1 = AscendC::Te::Get<2>(l1Shape);

        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        // B GM->L1
        auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(bL1Buffer_[l1BufId]), layoutBL1);
        auto gmTileB = tensorB.Slice(AscendC::Te::MakeCoord(kIdx * kL1_, 0), AscendC::Te::MakeShape(curKL1, curN));
        AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);

        // Bias GM->L1
        auto layoutBiasL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curN);
        auto tensorBiasL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(biasL1Buffer_[l1BufId]), layoutBiasL1);
        if (isBias_ && kIdx == 0) {
            AscendC::Te::Copy(copyGM2L1, tensorBiasL1, tensorBias);
        }

        return AscendC::Std::make_tuple(tensorBL1, tensorBiasL1);
    }

    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL0FromL1(
        const TensorA& tensorAL1, const TensorB& tensorBL1, const TensorBias& tensorBiasL1, const TripleShape& l0Shape,
        uint64_t l0Offset, bool needBias, uint64_t btBufId, uint64_t aL1MOffset, uint64_t aL1KOffset,
        uint64_t bL1KOffset, uint64_t bL1NOffset)
    {
        auto curM = AscendC::Te::Get<0>(l0Shape);
        auto curN = AscendC::Te::Get<1>(l0Shape);
        auto curK0 = AscendC::Te::Get<2>(l0Shape);

        // A L1->L0A
        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto layoutAL0 =
            AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(curM, curK0);
        auto tensorAL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Offset), layoutAL0);
        auto tensorBlockAL1 =
            tensorAL1.Slice(AscendC::Te::MakeCoord(aL1MOffset, aL1KOffset), AscendC::Te::MakeShape(curM, curK0));
        AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);

        // B L1->L0B
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        auto layoutBL0 =
            AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(curK0, curN);
        auto tensorBL0 =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Offset), layoutBL0);
        auto tensorBlockBL1 =
            tensorBL1.Slice(AscendC::Te::MakeCoord(bL1KOffset, bL1NOffset), AscendC::Te::MakeShape(curK0, curN));
        AscendC::Te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);

        // Bias L1->L0
        uint64_t nl1Align = Blaze::Gemm::CeilAlign(curN, static_cast<int64_t>(AscendC::BLOCK_CUBE));
        auto layoutBiasL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, nl1Align);
        auto offsetBiasL0 = baseN_ * btBufId * sizeof(float);
        auto tensorBiasL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, float>(offsetBiasL0), layoutBiasL0);
        if (needBias) {
            auto copyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
            AscendC::Te::Copy(copyL12BT, tensorBiasL0, tensorBiasL1);
        }

        return AscendC::Std::make_tuple(tensorAL0, tensorBL0, tensorBiasL0);
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void Compute(
        const TensorA& tensorAL0, const TensorB& tensorBL0, const TensorBias& tensorBiasL0, TensorC& tensorL0C,
        const TripleShape& l0Shape, bool needBias, uint8_t unitFlag, bool initCmatrix)
    {
        constexpr auto mmadAtom = AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});
        auto curM = AscendC::Te::Get<0>(l0Shape);
        auto curN = AscendC::Te::Get<1>(l0Shape);
        auto curK0 = AscendC::Te::Get<2>(l0Shape);
        // Mmad参数
        AscendC::Te::MmadParams mmadParams{
            static_cast<uint16_t>(curM), static_cast<uint16_t>(curN), static_cast<uint16_t>(curK0), unitFlag,
            initCmatrix};
        // 传入自定义Trait类型
        if (needBias) {
            AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
        } else {
            AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
        }
    }

private:
    static constexpr uint16_t MTE1_MTE2_EVENT_ID_NUM = 4;

    uint64_t oriK_{1};
    uint64_t mL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};

    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
    uint64_t biasL1OneBuffer_ = 0;
    uint64_t kL1Iter_{0};
    uint64_t mL1Iter_{0};
    uint64_t kAlign_{0};
    uint32_t l1Stages_{1};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    bool isAL1Loaded_{false};
    uint64_t bL1Buffer_[4] = {0};
    uint64_t biasL1Buffer_[4] = {0};
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze
