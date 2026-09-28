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
 * \file block_mmad_matmul_basic_splitk.h
 * \brief 非量化基本 matmul mmad —— splitKIdx 增强版（源自 ops-tensor，apace 本地维护）。
 *
 * 本文件基于 ops-tensor 的 include/blaze/gemm/block/block_mmad_matmul_basic.h
 * （上游基线 de3bf6f），向后兼容地增加 splitKIdx 接口（L0C 跨调用累加 + 末段单次
 * fixpipe），供 all_to_all_mm 内核显式 include 使用：
 *   #include "apace/block/mmad/block_mmad_matmul_basic_splitk.h"
 * splitKNum=1 / splitKIdx=0 时行为与上游原版完全一致。
 *
 * 定义的是 blaze 的 BlockMmad<MatmulMultiBlockBasic<...>> 偏特化；已验证本文件的
 * include 闭包（block_mmad.h / copy_gm_to_l1.h 等）不会引入上游原版 basic.h，
 * 无重定义风险。ST 与 Kernel UT 均使用本副本（UT 经 -I op_kernel 根目录解析）。
 *
 * 相对上游基线的差异（与 fork shiziqito/ops-tensor splitk 分支的功能改动一致，
 * 不含其 tensor API 风格迁移）：
 * 1. Params 新增 splitKNum（split-K 分段数，1 = 独立 matmul）
 * 2. Init 读入 splitKNum_
 * 3. operator() 新增末参 splitKIdx：isFirstSplitK 清零 L0C 并加 bias，
 *    isLastSplitK 才触发 CopyL0CToGM 与 L0C ping-pong 翻转
 * 4. NeedProcessBias / CopyL1FromGM 增加 isFirstSplitK 门控（bias 只在首个 split 处理）
 * 5. initCmatrix 仅在首个 split（且无 bias 时）触发；末拍 update 标志仅在最后一个 split
 * 6. 移除 operator() 内的 kL1_ 二次钳制（多 split 调用下会破坏 Init 设定值）
 *
 * 注意：fork 改动合入上游 ops-tensor 且 OPTENSOR_TAG_ID 更新后，本文件应删除，
 * 内核 include 切回 "blaze/gemm/block/block_mmad_matmul_basic.h"。
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/buffer_manager.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/tile/copy_gm_to_l1.h"
// 原文件位于 blaze/gemm/block/ 下用相对路径 "#include "block_mmad.h""（同目录优先解析）；
// 本文件迁移到 apace/block/mmad/ 后必须用全路径，经 -I ops-tensor/include 解析
#include "blaze/gemm/block/block_mmad.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <uint64_t FullLoadMode_, uint64_t FusedOpType_, class KernelSchedule_, uint64_t NonContiguousType_,
          MatmulOutputMode OutputMode_, class AType_, class LayoutA_, class BType_, class LayoutB_, class CType_,
          class LayoutC_, class BiasType_, class LayoutBias_>
class BlockMmad<MatmulMultiBlockBasic<FullLoadMode_, FusedOpType_, KernelSchedule_, NonContiguousType_, OutputMode_>,
                AType_, LayoutA_, BType_, LayoutB_, CType_, LayoutC_, BiasType_, LayoutBias_> {
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
        MatmulMultiBlockBasic<FullLoadMode_, FusedOpType_, KernelSchedule_, NonContiguousType_, OutputMode_>;
    static constexpr uint64_t NON_CONTIGUOUS_TYPE = DispatchPolicy::NON_CONTIGUOUS_TYPE;
    using TupleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using TripleShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;

    // TRANS_A and TRANS_B
    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ_FORMAT = IsWeightNz<LayoutB>::value;
    static constexpr bool IS_INT8_OUT = AscendC::Std::is_same_v<CType, signed char>;
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
        uint64_t mL1{0};
        uint64_t nL1{0};
        uint64_t kL1{0};
        uint32_t mL0{0};
        uint32_t nL0{0};
        uint32_t kL0{0};
        uint32_t l1Stages{1};
        uint16_t l0cStages{1};
        uint64_t splitKNum{1}; // split-K 分段数（跨调用 L0C 累加），1 表示独立 matmul
        GM_ADDR scaleGmAddr{nullptr};
        uint64_t rowStride{0};
    };

public:
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

    __aicore__ inline void Init(const Params &params, bool isBias)
    {
        mL1_ = params.mL1;
        nL1_ = params.nL1;
        kL1_ = params.kL1;
        baseM_ = params.mL0;
        baseN_ = params.nL0;
        baseK_ = params.kL0;
        isBias_ = isBias;
        l1Stages_ = params.l1Stages;
        splitKNum_ = params.splitKNum;
        enableL0cPingPong_ = params.l0cStages > 1;
        l0PingPong_ = 0;
        abL1LoopCnt_ = 0;
        l0cPingPong_ = 0;
        uint64_t aL1OneSize = mL1_ * kL1_ * sizeof(AType);
        uint64_t bL1OneSize = nL1_ * kL1_ * sizeof(BType);
        // 把L1分成QUADRUPLE_BUFFER_COUNT个等大的slot
        constexpr uint64_t slotSize = AscendC::TOTAL_L1_SIZE / QUADRUPLE_BUFFER_COUNT;
        // 每个stage占几个slot
        uint64_t stride = QUADRUPLE_BUFFER_COUNT / l1Stages_;
        // 2buffer时：|APing,BPing,BiasPing------|APong,BPong,BiasPong---|
        // 4buffer时：|A0,B0,Bias0 |A1,B1,Bias1 |A2,B2,Bias2 |A3,B3,Bias3 |
        for (uint32_t i = 0; i < l1Stages_; ++i) {
            uint64_t base = slotSize * stride * i;
            bufMgr_.InitAL1(i, base, i);
            bufMgr_.InitBL1(i, base + aL1OneSize, i);
            bufMgr_.InitBias(i, base + aL1OneSize + bL1OneSize, i);
        }
        // the bias in BT must be float32
        bufMgr_.InitBT(sizeof(float) * baseN_);
        bufMgr_.InitL0();
        bufMgr_.InitL0C();

        // Scale L1 buffer for int8 output fixpipe quantization
        uint64_t biasL1OneSize = nL1_ * sizeof(BiasType);
        uint64_t lastSlotBase = slotSize * stride * (l1Stages_ - 1);
        bufMgr_.InitScaleL1(lastSlotBase + aL1OneSize + bL1OneSize + biasL1OneSize);
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void operator()(TensorA &gmA, TensorB &gmB, TensorBias &gmBias, TensorC &gmC,
                                      TupleShape &blockShape, __gm__ uint64_t *gmScalePtr = nullptr,
                                      uint64_t splitKIdx = 0)
    {
        // split-K 语义：多个 operator() 调用在 L0C 上跨调用累加，最后一段触发一次 fixpipe 搬出。
        // splitKIdx == 0 时负责清零 L0C 并累加 bias，splitKIdx == splitKNum_-1 时触发 CopyL0CToGM。
        const bool isFirstSplitK = splitKIdx == 0;
        const bool isLastSplitK = splitKIdx == splitKNum_ - 1;
        // m0 == m1 && n0 == n1
        int64_t curM = Blaze::Gemm::Min(AscendC::Te::Get<MNK_M>(blockShape), static_cast<int64_t>(baseM_));
        int64_t curN = Blaze::Gemm::Min(AscendC::Te::Get<MNK_N>(blockShape), static_cast<int64_t>(baseN_));
        uint64_t oriK = AscendC::Te::Get<MNK_K>(blockShape); // 非全载blockShape的K维度固定返回原始K
        const auto &l0cSlot = bufMgr_.GetL0CSlot(l0cPingPong_ & 0x1);
        // LoC搬出
        auto layoutL0C = AscendC::Te::FrameLayoutFormat<AscendC::Te::NZLayoutPtn, AscendC::Te::_16>{}(curM, curN);
        auto tensorL0C = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0C, float>(l0cSlot.Addr()), layoutL0C);

        gmScalePtr_ = gmScalePtr;
        curNOut_ = static_cast<uint64_t>(curN);

        kL1Iter_ = Blaze::Gemm::CeilDiv(oriK, kL1_);
        for (uint64_t iter0 = 0; iter0 < kL1Iter_; ++iter0) {
            auto curKL1 = (iter0 + 1 == kL1Iter_) ? (oriK - kL1_ * iter0) : kL1_;
            uint64_t l1BufId = abL1LoopCnt_ & (l1Stages_ - 1);
            uint64_t btBufId = abL1LoopCnt_ & 0x1;

            const auto &aL1Slot = bufMgr_.GetL1ASlot(l1BufId);
            const auto &bL1Slot = bufMgr_.GetL1BSlot(l1BufId);
            const auto &biasL1Slot = bufMgr_.GetL1BiasSlot(l1BufId);
            const auto &btSlot = bufMgr_.GetBTSlot(btBufId);
            const auto &scaleL1Slot = bufMgr_.GetScaleL1Slot();

            // GM->L1
            TripleShape l1Shape{curM, curN, static_cast<int64_t>(curKL1)};
            auto l1Slots = AscendC::Std::make_tuple(aL1Slot, bL1Slot, biasL1Slot, scaleL1Slot);
            auto l1TensorTuple = CopyL1FromGM(gmA, gmB, gmBias, l1Shape, l1Slots, iter0, isFirstSplitK);
            auto tensorAL1 = AscendC::Te::Get<0>(l1TensorTuple);
            auto tensorBL1 = AscendC::Te::Get<1>(l1TensorTuple);
            auto tensorBiasL1 = AscendC::Te::Get<2>(l1TensorTuple);

            uint64_t kL0Iter = Blaze::Gemm::CeilDiv(curKL1, baseK_);
            for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
                uint64_t curK0 = (iter1 + 1 == kL0Iter) ? (curKL1 - iter1 * baseK_) : baseK_;
                const auto &l0Slot = bufMgr_.GetL0Slot(l0PingPong_ & 0x1);

                // A L1->L0
                TripleShape l0Shape{curM, curN, static_cast<int64_t>(curK0)};
                bool needBias = NeedProcessBias(iter0, iter1, isFirstSplitK);
                auto l0TensorTuple = CopyL0FromL1(tensorAL1, tensorBL1, tensorBiasL1, l0Shape, l0Slot, baseK_ * iter1,
                                                  needBias, l1Slots, btSlot);
                auto tensorAL0 = AscendC::Te::Get<0>(l0TensorTuple);
                auto tensorBL0 = AscendC::Te::Get<1>(l0TensorTuple);
                auto tensorBiasL0 = AscendC::Te::Get<2>(l0TensorTuple);

                // 只有首个 split 清零 L0C（bias 场景由 bias 表初始化，不清零）
                bool initCmatrix = iter0 == 0 && iter1 == 0 && isFirstSplitK && !needBias;
                // 只有最后一个 split 的最后一拍才 update 单元标志
                bool isFinalAccumulation = (iter0 + 1 == kL1Iter_ && iter1 + 1 == kL0Iter) && isLastSplitK;
                uint8_t unitFlag = (isFinalAccumulation ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION);

                {
                    auto l0Lock = l0Slot.LockM();
                    auto btLock = btSlot.LockM();
                    Compute(tensorAL0, tensorBL0, tensorBiasL0, tensorL0C, l0Shape, needBias, unitFlag, initCmatrix);
                }
                l0PingPong_++;
            }
            abL1LoopCnt_++;
        }

        // 数据搬出到GM（仅最后一个 split 触发一次）
        if (isLastSplitK) {
            CopyL0CToGM(gmC, tensorL0C);

            if (enableL0cPingPong_) {
                l0cPingPong_++;
            }
        }
    }

private:
    __aicore__ inline bool NeedProcessBias(uint64_t kIter0, uint64_t kIter1, bool isFirstSplitK)
    {
        // bias 只在首个 split 的首次 K 迭代累加，避免跨 split 重复加 bias
        return isBias_ && kIter0 == 0 && kIter1 == 0 && isFirstSplitK;
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename SlotsTuple>
    __aicore__ inline auto CopyL1FromGM(const TensorA &tensorA, const TensorB &tensorB, const TensorBias &tensorBias,
                                        const TripleShape &l1Shape, const SlotsTuple &slotsTuple, uint64_t kIdx,
                                        bool isFirstSplitK)
    {
        constexpr bool IS_SLICE_POLICY =
            NON_CONTIGUOUS_TYPE == static_cast<uint64_t>(NoContiguousType::NON_CONTIGUOUS_TYPE_SLICE);
        constexpr bool IS_BATCHED_B_POLICY =
            NON_CONTIGUOUS_TYPE == static_cast<uint64_t>(NoContiguousType::NON_CONTIGUOUS_TYPE_BATCHED_B);
        uint64_t curML1 = AscendC::Te::Get<MNK_M>(l1Shape);
        uint64_t curNL1 = AscendC::Te::Get<MNK_N>(l1Shape);
        uint64_t curKL1 = AscendC::Te::Get<MNK_K>(l1Shape);
        const auto &aL1Slot = AscendC::Te::Get<0>(slotsTuple);
        const auto &bL1Slot = AscendC::Te::Get<1>(slotsTuple);
        const auto &biasL1Slot = AscendC::Te::Get<2>(slotsTuple);
        const auto &scaleL1Slot = AscendC::Te::Get<3>(slotsTuple);

        // A GM->L1
        auto layoutAL1 = MakeLayoutAL1{}(curML1, curKL1);
        auto copyGM2L1 = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2L1{});
        auto tensorAL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, AType>(aL1Slot.Addr()), layoutAL1);
        {
            auto lock = aL1Slot.LockMte2();
            if constexpr (IS_SLICE_POLICY) {
                auto layoutGmA = tensorA.Layout();
                auto sliceM = AscendC::Te::Get<0>(AscendC::Te::Get<1>(layoutGmA.Shape()));
                auto gmTileASlice =
                    tensorA.Slice(AscendC::Te::MakeCoord(0, AscendC::Te::MakeCoord(0, kIdx * kL1_)),
                                  AscendC::Te::MakeShape(curML1 / sliceM, AscendC::Te::MakeShape(sliceM, curKL1)));
                auto copyGM2L1Slice = AscendC::Te::MakeCopy(Blaze::Gemm::Tile::CopySliceGM2L1{});
                AscendC::Te::Copy(copyGM2L1Slice, tensorAL1, gmTileASlice);
            } else {
                auto gmTileA =
                    tensorA.Slice(AscendC::Te::MakeCoord(0, kIdx * kL1_), AscendC::Te::MakeShape(curML1, curKL1));
                AscendC::Te::Copy(copyGM2L1, tensorAL1, gmTileA);
            }
        }

        // B GM->L1
        auto layoutBL1 = MakeLayoutBL1{}(curKL1, curNL1);
        auto tensorBL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(bL1Slot.Addr()), layoutBL1);
        {
            auto lock = bL1Slot.LockMte2();
            if constexpr (IS_BATCHED_B_POLICY) {
                CopyBatchedBToConcatL1(copyGM2L1, tensorBL1, tensorB, bL1Slot.Addr(), curKL1, kIdx);
            } else {
                auto gmTileB =
                    tensorB.Slice(AscendC::Te::MakeCoord(kIdx * kL1_, 0), AscendC::Te::MakeShape(curKL1, curNL1));
                AscendC::Te::Copy(copyGM2L1, tensorBL1, gmTileB);
            }
        }
        // Bias GM->L1
        auto layoutBiasL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curNL1);
        auto tensorBiasL1 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BiasType>(biasL1Slot.Addr()), layoutBiasL1);
        if (isBias_ && kIdx == 0 && isFirstSplitK) {
            auto lock = biasL1Slot.LockMte2();
            AscendC::Te::Copy(copyGM2L1, tensorBiasL1, tensorBias);
        }

        // Scale GM->L1 (int8 output fixpipe quantization)
        if (kIdx == 0) {
            if constexpr (IS_INT8_OUT) {
                auto layoutScaleL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curNOut_);
                auto tensorScaleL1 = AscendC::Te::MakeTensor(
                    AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, uint64_t>(scaleL1Slot.Addr()), layoutScaleL1);
                auto layoutScaleGM = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curNOut_);
                auto gmScale = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(gmScalePtr_),
                                                       layoutScaleGM);
                auto lock = scaleL1Slot.LockMte2();
                AscendC::Te::Copy(copyGM2L1, tensorScaleL1, gmScale);
            }
        }

        return AscendC::Std::make_tuple(tensorAL1, tensorBL1, tensorBiasL1);
    }

    template <typename CopyGM2L1, typename TensorBL1, typename TensorB>
    __aicore__ inline void CopyBatchedBToConcatL1(const CopyGM2L1 &copyGM2L1, const TensorBL1 &tensorBL1,
                                                  const TensorB &tensorB, uint64_t bL1Address, uint64_t curKL1,
                                                  uint64_t kIdx)
    {
        using TensorBElementType = AscendC::Te::GetAttributeElementType<typename TensorB::elementType *>;
        using TensorBLayoutPattern = AscendC::Te::GetLayoutPattern<typename TensorB::layoutType>;
        using TensorBL1ElementType = AscendC::Te::GetAttributeElementType<typename TensorBL1::elementType *>;
        using TensorBL1LayoutPattern = AscendC::Te::GetLayoutPattern<typename TensorBL1::layoutType>;
        using ExpectedBL1LayoutPattern =
            AscendC::Std::conditional_t<TRANS_B, AscendC::Te::ZNLayoutPtn, AscendC::Te::NZLayoutPtn>;
        static_assert(AscendC::Std::is_same_v<TensorBElementType, BType>,
                      "Batched-B GM tensor element type must match BlockMmad BType.");
        static_assert(AscendC::Std::is_same_v<TensorBL1ElementType, BType>,
                      "Batched-B L1 tensor element type must match BlockMmad BType.");
        static_assert(AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<TensorB>, AscendC::Te::Location::GM>,
                      "Batched-B source tensor must reside in GM.");
        static_assert(AscendC::Std::is_same_v<AscendC::Te::GetMemLocation<TensorBL1>, AscendC::Te::Location::L1>,
                      "Batched-B destination tensor must reside in L1.");
        static_assert(AscendC::Std::is_same_v<TensorBLayoutPattern, LayoutB>,
                      "Batched-B GM tensor layout pattern must match BlockMmad LayoutB.");
        static_assert(AscendC::Std::is_same_v<TensorBL1LayoutPattern, ExpectedBL1LayoutPattern>,
                      "Batched-B L1 tensor layout must match the BlockMmad transpose mode.");
        static_assert(TensorB::layoutType::depth == AscendC::Te::FIVE_DIM_DATA,
                      "Batched-B source tensor must contain an outer batch and a two-dimensional matrix layout.");

        auto gmBLayout = tensorB.Layout();
        const uint64_t batchCount = static_cast<uint64_t>(AscendC::Te::Get<0>(gmBLayout.Shape()));
        using GmBLayoutPattern = AscendC::Te::GetLayoutPattern<typename TensorB::layoutType>;
        using BLayoutTrait = AscendC::Te::LayoutTraitDefault<BType>;
        auto singleGmBLayout = AscendC::Te::MakePatternLayout<GmBLayoutPattern, BLayoutTrait>(
            AscendC::Te::Get<1>(gmBLayout.Shape()), AscendC::Te::Get<1>(gmBLayout.Stride()));
        const uint64_t singleN = AscendC::Te::GetTotalColumnShape(singleGmBLayout);
        auto gmTileB = tensorB.Slice(AscendC::Te::MakeCoord(0UL, AscendC::Te::MakeCoord(kIdx * kL1_, 0UL)),
                                     AscendC::Te::MakeShape(batchCount, AscendC::Te::MakeShape(curKL1, singleN)));

        // Re-view the same BL1 storage as a batch so one GM2L1 copy packs the matrices into adjacent N ranges.
        // The original two-dimensional tensorBL1 remains the compute view used by L1-to-L0 and MMAD.
        auto concatBL1Layout = tensorBL1.Layout();
        auto singleBL1 = tensorBL1.Slice(AscendC::Te::MakeCoord(0UL, 0UL), AscendC::Te::MakeShape(curKL1, singleN));
        auto singleBL1Layout = singleBL1.Layout();
        const uint64_t batchStride = concatBL1Layout(AscendC::Te::MakeCoord(0UL, singleN));
        using BL1Layout = typename TensorBL1::layoutType;
        using BL1LayoutPattern = AscendC::Te::GetLayoutPattern<BL1Layout>;
        auto batchedBL1Layout = AscendC::Te::MakePatternLayout<BL1LayoutPattern, BLayoutTrait>(
            AscendC::Te::MakeShape(batchCount, singleBL1Layout.Shape()),
            AscendC::Te::MakeStride(batchStride, singleBL1Layout.Stride()));
        auto batchedBL1 = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, BType>(bL1Address),
                                                  batchedBL1Layout);
        AscendC::Te::Copy(copyGM2L1, batchedBL1, gmTileB);
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename SlotsTuple>
    __aicore__ inline auto CopyL0FromL1(const TensorA &tensorAL1, const TensorB &tensorBL1,
                                        const TensorBias &tensorBiasL1, const TripleShape &l0Shape,
                                        const BufferSlot &l0Slot, uint64_t kIdx, bool needBias,
                                        const SlotsTuple &slotsTuple, const BufferSlot &btSlot)
    {
        auto curM0 = AscendC::Te::Get<MNK_M>(l0Shape);
        auto curN0 = AscendC::Te::Get<MNK_N>(l0Shape);
        auto curK0 = AscendC::Te::Get<MNK_K>(l0Shape);
        const auto &aL1Slot = AscendC::Te::Get<0>(slotsTuple);
        const auto &bL1Slot = AscendC::Te::Get<1>(slotsTuple);
        const auto &biasL1Slot = AscendC::Te::Get<2>(slotsTuple);
        // A L1->L0A
        auto copyL12L0A = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0A{});
        auto layoutAL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NZLayoutPtn, AscendC::Te::LayoutTraitDefault<AType>>(
            curM0, curK0);
        auto tensorAL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0A, AType>(l0Slot.Addr()), layoutAL0);
        auto tensorBlockAL1 = tensorAL1.Slice(AscendC::Te::MakeCoord(0, kIdx), AscendC::Te::MakeShape(curM0, curK0));
        {
            auto l1LockA = aL1Slot.LockMte1();
            auto l0Lock = l0Slot.LockMte1();
            AscendC::Te::Copy(copyL12L0A, tensorAL0, tensorBlockAL1);
        }

        // B L1->L0B
        auto copyL12L0B = AscendC::Te::MakeCopy(AscendC::Te::CopyL12L0B{});
        auto layoutBL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::ZNLayoutPtn, AscendC::Te::LayoutTraitDefault<BType>>(
            curK0, curN0);
        auto tensorBL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::L0B, BType>(l0Slot.Addr()), layoutBL0);
        auto tensorBlockBL1 = tensorBL1.Slice(AscendC::Te::MakeCoord(kIdx, 0), AscendC::Te::MakeShape(curK0, curN0));
        {
            auto l1LockB = bL1Slot.LockMte1();
            auto l0Lock = l0Slot.LockMte1();
            AscendC::Te::Copy(copyL12L0B, tensorBL0, tensorBlockBL1);
        }

        // Bias L1->L0
        uint64_t nl1Align = Blaze::Gemm::CeilAlign(curN0, static_cast<int64_t>(AscendC::BLOCK_CUBE));
        auto layoutBiasL0 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, nl1Align);
        auto tensorBiasL0 = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::BIAS, float>(btSlot.Addr()), layoutBiasL0);
        if (needBias) {
            auto biasLock = biasL1Slot.LockMte1();
            auto btLock = btSlot.LockMte1();
            auto copyL12BT = AscendC::Te::MakeCopy(AscendC::Te::CopyL12BT{});
            AscendC::Te::Copy(copyL12BT, tensorBiasL0, tensorBiasL1);
        }

        return AscendC::Std::make_tuple(tensorAL0, tensorBL0, tensorBiasL0);
    }

    template <typename TensorC, typename TensorL0C>
    __aicore__ inline void CopyL0CToGM(TensorC &gmC, TensorL0C &tensorL0C)
    {
        AscendC::Te::FixpipeParams fixpParams{FINAL_ACCUMULATION};
        auto copyL0C2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyL0C2GM{});
        if constexpr (IS_INT8_OUT) {
            const auto &scaleL1Slot = bufMgr_.GetScaleL1Slot();
            auto layoutScaleL1 = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(1UL, curNOut_);
            auto tensorScaleL1 = AscendC::Te::MakeTensor(
                AscendC::Te::MakeMemPtr<AscendC::Te::Location::L1, uint64_t>(scaleL1Slot.Addr()), layoutScaleL1);
            auto Fixlock = scaleL1Slot.LockFix();
            AscendC::Te::Copy(copyL0C2GM.with(fixpParams), gmC, tensorL0C, tensorScaleL1);
        } else {
            AscendC::Te::Copy(copyL0C2GM.with(fixpParams), gmC, tensorL0C);
        }
    }

    template <typename TensorA, typename TensorB, typename TensorBias, typename TensorC>
    __aicore__ inline void Compute(const TensorA &tensorAL0, const TensorB &tensorBL0, const TensorBias &tensorBiasL0,
                                   TensorC &tensorL0C, const TripleShape &l0Shape, bool needBias, uint8_t unitFlag,
                                   bool initCmatrix)
    {
        constexpr auto mmadAtom = AscendC::Te::MakeMmad(AscendC::Te::MmadOperation{}, AscendC::Te::MmadTraitDefault{});
        auto curM0 = AscendC::Te::Get<MNK_M>(l0Shape);
        auto curN0 = AscendC::Te::Get<MNK_N>(l0Shape);
        auto curK0 = AscendC::Te::Get<MNK_K>(l0Shape);
        // Mmad参数
        AscendC::Te::MmadParams mmadParams{static_cast<uint16_t>(curM0), static_cast<uint16_t>(curN0),
                                           static_cast<uint16_t>(curK0), unitFlag, initCmatrix};
        // 传入自定义Trait类型
        if (needBias) {
            AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0, tensorBiasL0);
        } else {
            AscendC::Te::Mmad(mmadAtom.with(mmadParams), tensorL0C, tensorAL0, tensorBL0);
        }
    }

private:
    uint64_t mL1_{1};
    uint64_t nL1_{1};
    uint64_t kL1_{1};
    uint64_t baseM_{16};
    uint64_t baseN_{16};
    uint64_t baseK_{16};

    uint64_t kL1Iter_{0};
    uint64_t splitKNum_{1};
    uint32_t l1Stages_{1};
    uint64_t abL1LoopCnt_{0};
    uint64_t l0PingPong_{0};
    uint64_t l0cPingPong_{0};
    bool isBias_{false};
    bool enableL0cPingPong_{false};
    uint64_t curNOut_{0};
    __gm__ uint64_t *gmScalePtr_{nullptr};

    // 全流水线Buffer管理器, <MaxL1ASlots = 4, MaxL1BSlots = 4, MaxL0Slots = 2>
    BufferManager<4, 4, 2> bufMgr_;
};
} // namespace Block
} // namespace Gemm
} // namespace Blaze
