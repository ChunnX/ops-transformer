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
 * \file block_mmad_matmul_b_fullLoad_fixpipe_opti.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/tile/tile_trait.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <
    uint64_t L0COutModel_, uint64_t FullLoadMode_, uint64_t FusedOpType_, class KernelSchedule_, class AType_,
    class LayoutA_, class BType_, class LayoutB_, class CType_, class LayoutC_, class BiasType_, class LayoutBias_>
    class BlockMmad<
        MatmulMultiBlockFullLoadOrFixpipe<L0COutModel_, FullLoadMode_, FusedOpType_, KernelSchedule_>, AType_,
        LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_> {
public:
    using AType = AType_;
    using BType = BType_;
    using CType = CType_;
    using BiasType = BiasType_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutBias = LayoutBias_;
    using DispatchPolicy =
        MatmulMultiBlockFullLoadOrFixpipe<L0COutModel_, FullLoadMode_, FusedOpType_, KernelSchedule_>;
    using TupleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using TileShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;

    // TRANS_A and TRANS_B
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHTNZ_FORMAT = IsWeightNz<LayoutB>::value;
    // AIC(cube) <-> AIV(vector) cross-core sync flags, mirrored from the fixpipe opti kernel.
    // Per (baseM, baseN) N-chunk: AIC waits AIV_SYNC_AIC_FLAG (UB slot free) before fixpipe,
    // sets AIC_SYNC_AIV_FLAG (chunk ready) after fixpipe. splitM uses the +FLAG_ID_MAX mirror.
    static constexpr uint16_t AIC_SYNC_AIV_MODE_4 = 4;
    static constexpr uint16_t AIV_SYNC_AIC_FLAG = 4;  // AIV -> AIC (UB slot free)
    static constexpr uint16_t AIC_SYNC_AIV_FLAG = 6;  // AIC -> AIV (chunk ready)
    static constexpr uint16_t FLAG_ID_MAX = 16;
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
        GM_ADDR groupListGmAddr{nullptr};
        GM_ADDR workspaceGmAddr{nullptr};
        uint64_t oriK{0};
        uint64_t mL1{0};
        uint64_t nL1{0};
        uint64_t kL1{0};
        uint32_t mL0{0};
        uint32_t nL0{0};
        uint32_t kL0{0};
        uint32_t l1Stages{1};
        uint16_t l0cStages{1};
        uint64_t splitM{0};
        uint8_t ubDB{1};
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
        k_ = params.oriK;
        mL1_ = params.mL1;
        nL1_ = params.nL1;
        kL1_ = params.kL1;
        baseM_ = params.mL0;
        baseN_ = params.nL0;
        baseK_ = params.kL0;
        isBias_ = params.biasGmAddr != nullptr;
        l1Stages_ = params.l1Stages;
        splitM_ = params.splitM;
        ubDB_ = params.ubDB;
        enableL0cPingPong_ = params.l0cStages > 1;
        isBL1Loaded_ = false;
        kAlign_ = Blaze::Gemm::CeilAlign(k_, static_cast<uint64_t>(AscendC::BLOCK_CUBE));
        // b全载
        aL1OneBuffer_ = mL1_ * kL1_ * sizeof(AType);
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == B_FULL_LOAD_MODE) {
            bL1OneBuffer_ = nL1_ * kAlign_ * sizeof(BType);
        } else {
            bL1OneBuffer_ = nL1_ * kL1_ * sizeof(BType);
        }
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
        static constexpr uint64_t QUARTER_L1_SIZE = AscendC::TOTAL_L1_SIZE / QUADRUPLE_BUFFER_COUNT;
        // 2 buffer: A0 A1 | B | Bias
        // 4 buffer: A0 A1 A2 A3 | B | Bias
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == B_FULL_LOAD_MODE) {
            for (auto i = 0; i < l1Stages_; ++i) {
                aL1Buffer_[i] = aL1OneBuffer_ * i;
            }
        } else {
            for (auto i = 0; i < l1Stages_; ++i) {
                aL1Buffer_[i] = QUARTER_L1_SIZE * (QUADRUPLE_BUFFER_COUNT / l1Stages_) * i;
                bL1Buffer_[i] = aL1Buffer_[i] + aL1OneBuffer_;
                biasL1Buffer_[i] = bL1Buffer_[i] + bL1OneBuffer_;
            }
        }
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(
        TensorA& gmA, TensorB& gmB, TensorBias& gmBias, TensorC& tensorC, TupleShape& tileShape)
    {
        static constexpr uint64_t HALF_L0C_SIZE = AscendC::TOTAL_L0C_SIZE / DOUBLE_BUFFER_COUNT;
        static constexpr uint64_t HALF_L0_SIZE = AscendC::TOTAL_L0A_SIZE / DOUBLE_BUFFER_COUNT;

        cvPingPong_ = 0;
        uint64_t curM = AscendC::Te::Get<MNK_M>(tileShape);
        uint64_t curN = AscendC::Te::Get<MNK_N>(tileShape);
        uint64_t curK = AscendC::Te::Get<MNK_K>(tileShape);

        auto l1FullLoadTensorTuple = CopyL1FromGMbFullLoad(gmB, gmBias, curK, curN);
        auto tensorB = AscendC::Te::Get<0>(l1FullLoadTensorTuple);
        auto tensorBias = AscendC::Te::Get<1>(l1FullLoadTensorTuple);

        curBaseN_ = Min(curN, baseN_);
        nL1Iter_ = CeilDiv(curN, curBaseN_);
        kL1_ = Min(k_, kL1_);
        kL1Iter_ = CeilDiv(k_, kL1_);
        for (uint64_t iterN = 0; iterN < nL1Iter_; ++iterN) {
            auto tileN = (iterN + 1 == nL1Iter_) ? (curN - curBaseN_ * iterN) : curBaseN_;
            uint64_t l0cOffset = (l0cPingPong_ & 0x1) * HALF_L0C_SIZE;
            // LoC搬出
            auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Std::Int<16>>{}(curM, tileN);
            auto tensorL0C =
                AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cOffset), layoutL0C);
            for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
                auto curKL1 = (iter0 + 1 == kL1Iter_) ? (k_ - kL1_ * iter0) : kL1_;
                // 普通模板-2buffer-AL1搬入偏移位置：*AL1Ping*-BL1Ping-BiasPing|*AL1Pong*-BL1Pong-BiasPong
                uint64_t l1BufId = abL1LoopCnt_ & (l1Stages_ - 1);
                uint64_t btBufId = abL1LoopCnt_ & 0x1;
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
                // GM->L1
                TileShape l1Shape{curM, tileN, curKL1};
                auto l1TensorTuple = CopyL1FromGM(gmA, gmB, gmBias, l1Shape, l1BufId, iter0);
                auto tensorAL1 = AscendC::Te::Get<0>(l1TensorTuple);
                auto tensorBL1 =  AscendC::Te::Get<1>(l1TensorTuple);
                auto tensorBiasL1 = AscendC::Te::Get<2>(l1TensorTuple);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

                uint64_t kL0Iter = CeilDiv(curKL1, baseK_);
                for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                    uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                    uint64_t l0Offset = HALF_L0_SIZE * (l0PingPong_ & 0x1);
                    uint64_t mte1Flag = ((l0PingPong_ & 0x1) + SIXTH_FLAG);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<uint16_t>(mte1Flag));

                    uint64_t aL1MOffset = 0;
                    uint64_t aL1KOffset = iter1 * baseK_;
                    uint64_t bL1KOffset = aL1KOffset;
                    uint64_t bL1NOffset = iterN * curBaseN_;

                    if constexpr (DispatchPolicy::FULL_LOAD_MODE == B_FULL_LOAD_MODE) {
                        bL1KOffset = iter0 * kL1_ + iter1 * baseK_;
                        bL1NOffset = iterN * curBaseN_;
                        tensorBL1 = tensorB;
                        tensorBiasL1 = tensorBias.Slice(AscendC::Te::MakeCoord(0, iterN * curBaseN_), AscendC::Te::MakeShape(1, tileN));
                    } else {
                        // 非全载场景下b矩阵和bais已在kernel中slice切块处理
                    }

                    // A L1->L0
                    TileShape l0Shape{curM, tileN, curK0};
                    bool needBias = NeedProcessBias(iter0, iter1);
                    auto l0TensorTuple = CopyL0FromL1(
                        tensorAL1, tensorBL1, tensorBiasL1, l0Shape, l0Offset, aL1MOffset, aL1KOffset, bL1KOffset,
                        bL1NOffset, needBias, btBufId);
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

            // 数据搬出到GM/UB
            AscendC::Te::FixpipeParams fixpParams{FINAL_ACCUMULATION};
            if constexpr (DispatchPolicy::L0C2OUT_MODEL != ON_THE_FLY) {
                // 以 (baseM, baseN) 为基本块搬入 UB，ping-pong 两个槽位，与 AIV epilogue 逐块 cv 同步
                uint16_t slot = ((ubDB_ > 1) && (nL1Iter_ > 1)) ? static_cast<uint16_t>(cvPingPong_ & 0x1) : 0U;
                AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + slot);
                if (splitM_) {
                    AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIV_SYNC_AIC_FLAG + slot + FLAG_ID_MAX);
                }
                CopyOutFromL0C2UB(tensorC, tensorL0C, tileN, curM, slot);
                AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + slot);
                if (splitM_) {
                    AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_FIX>(AIC_SYNC_AIV_FLAG + slot + FLAG_ID_MAX);
                }
                cvPingPong_++;
            } else {
                auto tensorGmC = tensorC.Slice(AscendC::Te::MakeCoord(0, iterN * curBaseN_), AscendC::Te::MakeShape(curM, tileN));
                auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
                AscendC::Te::Copy(copyL0C2GM.with(fixpParams), tensorGmC, tensorL0C);
            }

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
    __aicore__ inline auto CopyL1FromGMbFullLoad(TensorB& gmB, TensorBias& gmBias, uint64_t curK, uint64_t curN)
    {
        uint64_t offsetBl1 = aL1OneBuffer_ * l1Stages_;
        uint64_t offsetBiasL1 = offsetBl1 + bL1OneBuffer_;

        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});

        auto layoutBL1 = MakeLayoutBL1{}(curK, curN);
        auto tensorB =
            AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(offsetBl1), layoutBL1);

        auto layoutBiasL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curN);
        auto tensorBias = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(offsetBiasL1), layoutBiasL1);

        // B GM->L1 全载模式
        if constexpr (DispatchPolicy::FULL_LOAD_MODE == B_FULL_LOAD_MODE) {
            if (!isBL1Loaded_) {
                AscendC::Te::Copy(copyGM2L1, tensorB, gmB);
                if (isBias_) {
                    AscendC::Te::Copy(copyGM2L1, tensorBias, gmBias);
                }
                isBL1Loaded_ = true;
            }
        }
        return AscendC::Std::make_tuple(tensorB, tensorBias);
    }

    template <typename TensorUB, typename TensorC>
    __aicore__ inline void CopyOutFromL0C2UB(TensorUB& tensorC, TensorC& tensorL0C, uint64_t tileN, uint64_t curM, uint16_t slotIdx)
    {
        // 数据搬出到UB，按 ping-pong 槽位 slotIdx 写入，槽位步长固定为 (baseM 对齐) * (baseN 对齐)
        AscendC::Te::FixpipeParams fixpParams{FINAL_ACCUMULATION};
        constexpr uint64_t c0Size = static_cast<uint64_t>(AscendC::Te::C0_ELEMENT<CType>);
        uint64_t tileNAlign = Blaze::Gemm::CeilAlign(tileN, c0Size);
        auto layoutUB = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(
            Blaze::Gemm::CeilAlign(curM, SPLIT_M_ALIGN), tileNAlign);
        constexpr int64_t ubHalfElems = static_cast<int64_t>(AscendC::TOTAL_UB_SIZE / sizeof(CType) / DOUBLE_BUFFER_COUNT);
        auto ubTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB>(tensorC.Data().Get() + slotIdx * ubHalfElems), layoutUB);
        if (splitM_) {
            auto copyL0C2UBSplitM =
                AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{}, Blaze::Gemm::Tile::CopyL0C2UBTraitSplitM{});
            AscendC::Te::Copy(copyL0C2UBSplitM.with(fixpParams), ubTensor, tensorL0C);
        } else {
            auto copyL0C2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2UB{});
            AscendC::Te::Copy(copyL0C2UB.with(fixpParams), ubTensor, tensorL0C);
        }
    }

    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL1FromGM(
        const TensorA& tensorA, const TensorB& tensorB, const TensorBias& tensorBias, const TileShape& l1Shape,
        uint64_t l1BufId, uint64_t kIdx)
    {
        uint64_t curM = AscendC::Te::Get<0>(l1Shape);
        uint64_t curN = AscendC::Te::Get<1>(l1Shape);
        uint64_t curKL1 = AscendC::Te::Get<2>(l1Shape);

        // A GM->L1
        auto layoutAL1 = MakeLayoutAL1{}(curM, curKL1);
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(aL1Buffer_[l1BufId]), layoutAL1);
        auto gmTileA = tensorA.Slice(AscendC::Te::MakeCoord(0, kIdx * kL1_), AscendC::Te::MakeShape(curM, curKL1));
        AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);

        auto layoutBL1 = MakeLayoutBL1{}(curKL1, curN);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(bL1Buffer_[l1BufId]), layoutBL1);
        auto layoutBiasL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curN);
        auto tensorBiasL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(biasL1Buffer_[l1BufId]), layoutBiasL1);

        if constexpr (DispatchPolicy::FULL_LOAD_MODE == NONE_FULL_LOAD_MODE) {
            // B GM->L1
            auto gmTileB = tensorB.Slice(AscendC::Te::MakeCoord(kIdx * kL1_, 0), AscendC::Te::MakeShape(curKL1, curN));
            AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);
            // Bias GM->L1
            if (isBias_ && kIdx == 0) {
                AscendC::Te::Copy(copyGM2L1, tensorBiasL1, tensorBias);
            }
        } else {
            // 全载场景已在kernel Tile层实现
        }

        return AscendC::Std::make_tuple(tensorAL1, tensorBL1, tensorBiasL1);
    }
    template <typename TensorA, typename TensorB, typename TensorBias>
    __aicore__ inline auto CopyL0FromL1(
        const TensorA& tensorAL1, const TensorB& tensorBL1, const TensorBias& tensorBiasL1, const TileShape& l0Shape,
        uint64_t l0Offset, uint64_t aL1MOffset, uint64_t aL1KOffset, uint64_t bL1KOffset, uint64_t bL1NOffset,
        bool needBias, uint64_t btBufId)
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
        uint64_t nL1Align = Blaze::Gemm::CeilAlign(curN, static_cast<int64_t>(AscendC::BLOCK_CUBE));
        auto layoutBiasL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, nL1Align);
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
        const TileShape& l0Shape, bool needBias, uint8_t unitFlag, bool initCmatrix)
    {
        constexpr auto mmadAtom = AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});
        auto curM = AscendC::Te::Get<0>(l0Shape);
        auto curN = AscendC::Te::Get<1>(l0Shape);
        auto curK0 = AscendC::Te::Get<2>(l0Shape);
        // Mmad参数
        AscendC::Te::MmadParams mmadParams{static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
                                           static_cast<uint16_t>(curK0), unitFlag, initCmatrix};
        // 传入自定义Trait类型
        if (needBias) {
            AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
        } else {
            AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
        }
    }

private:
    static constexpr uint64_t SPLIT_M_ALIGN = 2;
    static constexpr uint16_t DIMENSION_M = 0;
    static constexpr uint16_t DIMENSION_N = 1;
    static constexpr uint16_t DIMENSION_K = 2;
    static constexpr uint16_t MTE1_MTE2_EVENT_ID_NUM = 4;

    uint64_t k_{1};
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};
    uint64_t curBaseN_{16};
    uint64_t aL1OneBuffer_ = 0;
    uint64_t bL1OneBuffer_ = 0;
    uint64_t nL1Iter_{0};
    uint64_t kL1Iter_{0};
    uint64_t kAlign_{0};
    uint32_t l1Stages_{1};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    uint64_t ubDB_{0};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    bool splitM_{false};
    uint64_t aL1Buffer_[4] = {0};
    uint64_t bL1Buffer_[4] = {0};
    uint64_t biasL1Buffer_[4] = {0};
    bool isBL1Loaded_{false};
    uint64_t cvPingPong_{0};    // continuous ping-pong counter across all N chunks of all blocks
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze