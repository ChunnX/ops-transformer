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
 * \file grouped_matmul_mx_kernel.h
 */

#pragma once

#if defined(ASCENDC_CPU_DEBUG)
#include "kernel_operator.h"
#elif ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

namespace {
// group list 的 三种类型
constexpr uint64_t GROUP_LIST_TYPE_OFFSET = 0UL; // group list 累计offset
constexpr uint64_t GROUP_LIST_TYPE_LENGTH = 1UL; // group list 存每组 token 数
constexpr uint64_t GROUP_LIST_TYPE_SPARSE = 2UL; // [专家id， token数]， 组与专家可多对一，允许中间 0 token 组
constexpr uint64_t SPARSE_GROUP_LIST_ITEM_STRIDE = 2UL;
constexpr uint64_t SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET = 1UL;
constexpr int64_t BLOCK_CUBE_MASK = BLOCK_CUBE - 1;
constexpr int64_t SCALE_CACHE_MASK = 0xff;

} // namespace

template <class ProblemShape, class BlockMmad, class BlockScheduler, class ScaleAType = fp8_e8m0_t,
          class ScaleBType = fp8_e8m0_t>
class GroupedMatmulMxKernel {
public:
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;

    static constexpr bool A_IS_FP8 = AscendC::IsSameType<AType, fp8_e4m3fn_t>::value ||
                                     AscendC::IsSameType<AType, fp8_e5m2_t>::value;
    static constexpr bool B_IS_FP8 = AscendC::IsSameType<BType, fp8_e4m3fn_t>::value ||
                                     AscendC::IsSameType<BType, fp8_e5m2_t>::value;
    static_assert((IsFp4<AType>() && IsFp4<BType>()) || (A_IS_FP8 && B_IS_FP8),
                  "GroupedMatmulMx only supports same bit-width MXFP4/MXFP8 AType and BType.");
    static_assert(AscendC::Std::is_one_of_v<CType, half, bfloat16_t, float>,
                  "GroupedMatmulMx only supports half/bfloat16_t/float CType.");
    static_assert(AscendC::Std::is_same_v<BiasType, float>, "GroupedMatmulMx only supports float BiasType.");
    static_assert(AscendC::Std::is_same_v<LayoutA, AscendC::Te::NDExtLayoutPtn>,
                  "GroupedMatmulMx supports ND LayoutA only (A transpose unsupported).");
    static_assert(AscendC::Std::is_one_of_v<LayoutB, AscendC::Te::NDExtLayoutPtn,
                                            AscendC::Te::DNExtLayoutPtn, AscendC::Te::NZLayoutPtn,
                                            AscendC::Te::ZNLayoutPtn>,
                  "GroupedMatmulMx only supports ND/DN/NZ/ZN LayoutB.");
    static_assert(!AscendC::Std::is_one_of_v<LayoutC, AscendC::Te::NZLayoutPtn, AscendC::Te::ZNLayoutPtn> &&
                      !AscendC::Std::is_one_of_v<LayoutBias, AscendC::Te::NZLayoutPtn,
                                                 AscendC::Te::ZNLayoutPtn>,
                  "GroupedMatmulMx does not support NZ/ZN LayoutC or LayoutBias.");

    static constexpr bool TRANS_A = IsTrans<LayoutA>::value;
    static constexpr bool TRANS_B = IsTrans<LayoutB>::value;
    static constexpr bool WEIGHT_NZ = IsWeightNz<LayoutB>::value;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockShape = typename BlockMmad::BlockShape;
    using SchedulerProblemShape = typename BlockScheduler::ProblemShape;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    static constexpr uint32_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    using MakeLayoutA = AscendC::Te::FrameLayoutFormat<LayoutA, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
    using MakeLayoutC =
        AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::Te::C0_ELEMENT<CType>>>;
    using MakeLayoutScaleA = AscendC::Std::conditional_t<
        TRANS_A, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleADNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB = AscendC::Std::conditional_t<
        TRANS_B, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    // 表按 grouplist 位置对齐（table[i] 对应 grouplist 第 i 项）
    struct GroupOffsetTables {
        const __gm__ uint64_t *aGroupAddrTable{nullptr};
        const __gm__ uint64_t *scaleAGroupAddrTable{nullptr};
        const __gm__ uint64_t *cGroupAddrTable{nullptr};
        const __gm__ uint64_t *bGroupAddrTable{nullptr};
        const __gm__ uint64_t *scaleBGroupAddrTable{nullptr};
    };

    struct GMMTiling {
        uint32_t groupNum;
        int64_t m;
        int64_t n;
        int64_t k;
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t kAL1;      // A 一次搬进 L1 的 K 元素数；填 0 兜底为 baseK
        uint32_t kBL1;      // 同上，B 侧
        uint32_t scaleKAL1; // scale K 窗口；填 0 兜底为 baseK（MX 要求 kAL1 为 64 倍数）
        uint8_t isBias;
        uint8_t dbL0C{DOUBLE_BUFFER_COUNT};         // L0C 中的 buffer 个数（1/2）
        uint8_t l1BufferStage{DOUBLE_BUFFER_COUNT}; // L1 缓冲 stage 数（2/3，block 层硬编码消费）
        uint8_t groupListType;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        GM_ADDR groupListGmAddr; // groupList 每个 group 的 token 数
        GMMTiling gmmParams;
        GroupOffsetTables groupOffsetTables; // 各 group 相对基址偏移表
    };

    __aicore__ inline GroupedMatmulMxKernel() {}
    __aicore__ inline ~GroupedMatmulMxKernel() {}

    __aicore__ inline void Init(const Params &params)
    {
        if ASCEND_IS_AIV {
            return;
        }
        tiling_ = params.gmmParams;
        const GMMTiling *gmmParams = &tiling_;
        aBasePtr_ = reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr);
        bBasePtr_ = reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr);
        cBasePtr_ = reinterpret_cast<__gm__ CType *>(params.mmadParams.cGmAddr);
        scaleABasePtr_ = reinterpret_cast<__gm__ ScaleAType *>(params.mmadParams.scaleAGmAddr);
        scaleBBasePtr_ = reinterpret_cast<__gm__ ScaleBType *>(params.mmadParams.scaleBGmAddr);
        if (gmmParams->isBias == 1) {
            biasBasePtr_ = reinterpret_cast<__gm__ float *>(params.mmadParams.biasGmAddr);
            isBias_ = true;
        } else {
            biasBasePtr_ = nullptr;
            isBias_ = false;
        }
        const ProblemShape initProblemShape{gmmParams->m, gmmParams->n, gmmParams->k, 0};
        problemShape_ = initProblemShape;
        groupNum_ = gmmParams->groupNum;
        curBaseM_ = gmmParams->baseM;
        groupListGmAddr_ = params.groupListGmAddr;
        offsetTables_ = params.groupOffsetTables;

        groupState_ = {};
        groupOffsets_ = {};

        if constexpr (!WEIGHT_NZ) {
            groupState_.perGroupB = gmmParams->n * gmmParams->k;
        } else {
            if constexpr (TRANS_B) {
                groupState_.perGroupB = static_cast<int64_t>(
                    Align16(gmmParams->n) * (IsFp4<AType>() ? Align64(gmmParams->k) : Align32(gmmParams->k)));
            } else {
                groupState_.perGroupB = static_cast<int64_t>(
                    (IsFp4<AType>() ? Align64(gmmParams->n) : Align32(gmmParams->n)) * Align16(gmmParams->k));
            }
        }

        const BlockShape l0Shape{static_cast<int64_t>(gmmParams->baseM), static_cast<int64_t>(gmmParams->baseN),
                                 static_cast<int64_t>(gmmParams->baseK), 0};
        const uint64_t kAL1 = gmmParams->kAL1 != 0 ? gmmParams->kAL1 : gmmParams->baseK;
        const uint64_t kBL1 = gmmParams->kBL1 != 0 ? gmmParams->kBL1 : gmmParams->baseK;
        const uint64_t scaleKAL1 = gmmParams->scaleKAL1 != 0 ? gmmParams->scaleKAL1 : gmmParams->baseK;
        const L1Params l1Params{kAL1, kBL1, scaleKAL1};
        const typename BlockMmad::MmadParams mmadParams{
            l0Shape, l1Params, isBias_, gmmParams->dbL0C == DOUBLE_BUFFER_COUNT, gmmParams->l1BufferStage};
        mmadOp_.Init(initProblemShape, mmadParams);
    }

    __aicore__ inline void Run()
    {
        if ASCEND_IS_AIV {
            return;
        }
        if (groupNum_ == 0) {
            return;
        }
        AscendC::GlobalTensor<int64_t> gmGroupList;
        gmGroupList.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupListGmAddr_));
        BlockScheduler scheduler(tiling_.baseM, tiling_.baseN, tiling_.baseK);
        SetSchedulerTailAlign(scheduler); // 设置尾块调度器能切的最小 block 大小
        const uint32_t lastGroupIdx = groupNum_ - 1;
        for (uint32_t groupIdx = 0; groupIdx < lastGroupIdx; ++groupIdx) {
            SetMNK(gmGroupList, groupIdx); // 按当前 group 的 M 重新均衡 baseM
            const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
            const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
            const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
            if (problemM <= 0 || problemN <= 0 || problemK <= 0) {
                continue;
            }
            BaseMBalance(scheduler, problemM, tiling_.baseM); // 按当前 group 的 M 重新均衡 baseM
            scheduler.UpdateNextProblem(
                SchedulerProblemShape{problemM, problemN, problemK, 0}); // 对当前的 groupidx 更新调度器状态
            ProcessSingleGroup<false>(scheduler, gmGroupList, groupIdx);
        }

        SetMNK(gmGroupList, lastGroupIdx);
        const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        if (problemM > 0 && problemN > 0 && problemK > 0) {
            BaseMBalance(scheduler, problemM, tiling_.baseM);
            scheduler.UpdateNextProblem(SchedulerProblemShape{problemM, problemN, problemK, 0});
            if (IsLastGroupAndNeedSplit(scheduler)) {
                scheduler.UpdateTailTile(); // 尾部均衡
                ProcessSingleGroup<true>(scheduler, gmGroupList, lastGroupIdx);
            } else {
                ProcessSingleGroup<false>(scheduler, gmGroupList, lastGroupIdx);
            }
        }
    }

private:
    __aicore__ inline void SetSchedulerTailAlign(BlockScheduler &scheduler)
    {
        // 调度器尾块切分，能切的最小block的大小
        constexpr uint32_t mTailAlign = 1;
        constexpr uint32_t nTailAlign = TRANS_B ? static_cast<uint32_t>(BLOCK_CUBE) : static_cast<uint32_t>(C0_SIZE);
        scheduler.SetTailAlign(mTailAlign, nTailAlign);
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2CacheHint(TensorB &gmB, TensorScaleB &gmScaleB, int64_t mSize, int64_t curBaseM,
                                          int64_t baseN)
    {
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        if constexpr (WEIGHT_NZ) {
            if (curBaseM >= mSize) {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
            } else {
                gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
            }
        } else {
            if constexpr (TRANS_B) {
                if (curBaseM >= mSize && (problemK & SCALE_CACHE_MASK) == 0) {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                }
            } else {
                if (curBaseM >= mSize && (problemN & SCALE_CACHE_MASK) == 0 && (baseN & SCALE_CACHE_MASK) == 0) {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_DISABLE);
                } else {
                    gmB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                    gmScaleB.SetL2CacheHint(AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
                }
            }
        }
    }

    __aicore__ inline void BaseMBalance(BlockScheduler &scheduler, int64_t m, int64_t baseM)
    {
        if (m <= 0) {
            return;
        }
        const int64_t safeBaseM = baseM > 0 ? baseM : static_cast<int64_t>(BLOCK_CUBE);
        const int64_t mCnt = (m + safeBaseM - 1) / safeBaseM;
        const int64_t balancedBaseM = (m + mCnt - 1) / mCnt;
        curBaseM_ = static_cast<uint32_t>((balancedBaseM + BLOCK_CUBE_MASK) & ~BLOCK_CUBE_MASK);
        scheduler.UpdateBaseM(curBaseM_);
    }

    __aicore__ inline bool IsLastGroupAndNeedSplit(const BlockScheduler &scheduler)
    {
        return (scheduler.GetEndBlockIdx() + 1) <= (AscendC::GetBlockNum() >> 1);
    }

    template <typename GroupListTensor>
    __aicore__ inline void SetMNK(const GroupListTensor &gmGroupList, uint32_t groupIdx)
    {
        const int64_t splitValue = GetSplitValueFromGroupList(gmGroupList, groupIdx); // 当前 groupIdx 的 tokens 数量
        const int64_t n = AscendC::Te::Get<MNK_N>(problemShape_);
        problemShape_ = ProblemShape{splitValue, n, AscendC::Te::Get<MNK_K>(problemShape_), 0};
    }

    template <typename GroupListTensor>
    __aicore__ inline int64_t GetSplitValueFromGroupList(const GroupListTensor &gmGroupList, uint32_t groupIdx)
    {
        int64_t splitValue = 0;
        if (tiling_.groupListType == GROUP_LIST_TYPE_OFFSET) {
            const int64_t offset = gmGroupList.GetValue(groupIdx);
            splitValue = offset - groupState_.preOffset;
            groupState_.preOffset = offset;
        } else if (tiling_.groupListType == GROUP_LIST_TYPE_LENGTH) {
            splitValue = gmGroupList.GetValue(groupIdx);
            groupState_.preOffset += splitValue;
        } else {
            const uint32_t splitValueIdx =
                groupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE + SPARSE_GROUP_LIST_SPLIT_VALUE_OFFSET;
            splitValue = gmGroupList.GetValue(splitValueIdx);
            groupState_.preOffset += splitValue;
        }
        return splitValue;
    }

    __aicore__ inline int64_t GetScaleK(int64_t k) const
    {
        return ((k + MXFP_DIVISOR_SIZE - 1) >> MXFP_DIVISOR_SHIFT) << MXFP_MULTI_BASE_SHIFT;
    }

    template <typename GroupListTensor>
    __aicore__ inline void UpdateBaseOffsets(const GroupListTensor &gmGroupList, uint32_t groupIdx)
    {
        uint32_t weightIdx = groupIdx;
        if (tiling_.groupListType == GROUP_LIST_TYPE_SPARSE) {
            weightIdx = static_cast<uint32_t>(gmGroupList.GetValue(groupIdx * SPARSE_GROUP_LIST_ITEM_STRIDE));
        }

        const int64_t m = AscendC::Te::Get<MNK_M>(problemShape_);
        const int64_t n = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t k = AscendC::Te::Get<MNK_K>(problemShape_);
        const int64_t prevSplitOffset = groupState_.preOffset - m;

        if (offsetTables_.aGroupAddrTable != nullptr) {
            groupOffsets_.a = static_cast<int64_t>(offsetTables_.aGroupAddrTable[groupIdx]);
        } else {
            groupOffsets_.a = IsFp4<AType>() ? ((prevSplitOffset * k) >> 1) : (prevSplitOffset * k);
        }

        if (offsetTables_.scaleAGroupAddrTable != nullptr) {
            groupOffsets_.scaleA = static_cast<int64_t>(offsetTables_.scaleAGroupAddrTable[groupIdx]);
        } else {
            groupOffsets_.scaleA = prevSplitOffset * GetScaleK(k);
        }

        if (offsetTables_.cGroupAddrTable != nullptr) {
            groupOffsets_.c = static_cast<int64_t>(offsetTables_.cGroupAddrTable[groupIdx]);
        } else {
            groupOffsets_.c = prevSplitOffset * n;
        }

        if (offsetTables_.bGroupAddrTable != nullptr) {
            groupOffsets_.b = static_cast<int64_t>(offsetTables_.bGroupAddrTable[groupIdx]);
        } else {
            groupOffsets_.b = groupState_.perGroupB * static_cast<int64_t>(weightIdx);
            if constexpr (IsFp4<BType>()) {
                groupOffsets_.b >>= 1;
            }
        }

        if (offsetTables_.scaleBGroupAddrTable != nullptr) {
            groupOffsets_.scaleB = static_cast<int64_t>(offsetTables_.scaleBGroupAddrTable[groupIdx]);
        } else {
            const int64_t scaleK = GetScaleK(k);
            groupOffsets_.scaleB = static_cast<int64_t>(weightIdx) * n * scaleK;
        }
        groupOffsets_.bias = static_cast<int64_t>(weightIdx) * n;
    }

    template <bool isLastGroupAndNeedSplit, typename GroupListTensor>
    __aicore__ inline void ProcessSingleGroup(BlockScheduler &scheduler, const GroupListTensor &gmGroupList,
                                              uint32_t groupIdx)
    {
        BlockCoord blockCoord;
        if (!scheduler.GetNextBlockCoord(blockCoord)) {
            return;
        }
        UpdateBaseOffsets(gmGroupList, groupIdx);

        mmadOp_.UpdateParamsForNextProblem(problemShape_);

        const int64_t problemM = AscendC::Te::Get<MNK_M>(problemShape_);
        const int64_t problemN = AscendC::Te::Get<MNK_N>(problemShape_);
        const int64_t problemK = AscendC::Te::Get<MNK_K>(problemShape_);
        const int64_t scaleK = GetScaleK(problemK);
        const int64_t baseN = static_cast<int64_t>(tiling_.baseN);
        auto layoutA = MakeLayoutA{}(problemM, problemK);
        auto layoutScaleA = MakeLayoutScaleA{}(problemM, scaleK);
        auto layoutB = MakeLayoutB{}(problemK, problemN);
        auto layoutScaleB = MakeLayoutScaleB{}(scaleK, problemN);
        auto layoutC = MakeLayoutC{}(problemM, problemN);
        auto gmA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(aBasePtr_ + groupOffsets_.a), layoutA);
        auto gmScaleA = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleABasePtr_ + groupOffsets_.scaleA), layoutScaleA);
        auto gmB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(bBasePtr_ + groupOffsets_.b), layoutB);
        auto gmScaleB = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(scaleBBasePtr_ + groupOffsets_.scaleB), layoutScaleB);
        auto gmC = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(cBasePtr_ + groupOffsets_.c), layoutC);
        auto biasPtr = isBias_ ? (biasBasePtr_ + groupOffsets_.bias) : nullptr;
        auto layoutBias =
            AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(static_cast<int64_t>(1), problemN);
        auto gmBias = AscendC::Te::MakeTensor(AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(biasPtr), layoutBias);

        if constexpr (!isLastGroupAndNeedSplit) {
            SetL2CacheHint(gmB, gmScaleB, problemM, static_cast<int64_t>(curBaseM_),
                           static_cast<int64_t>(tiling_.baseN));
        }

        do {
            const auto schedulerBlockShape = scheduler.GetBlockShape(blockCoord);
            const int64_t blockM = AscendC::Te::Get<MNK_M>(schedulerBlockShape);
            const int64_t blockN = AscendC::Te::Get<MNK_N>(schedulerBlockShape);
            if (blockM <= 0 || blockN <= 0) {
                continue;
            }
            const int64_t mSplitOffset =
                AscendC::Te::Get<MNK_K>(schedulerBlockShape); // 尾块切分时，被切块内部的偏移
            const int64_t nSplitOffset = AscendC::Te::Get<MNK_B>(schedulerBlockShape);
            const int64_t mBlockIdx = AscendC::Te::Get<MNK_M>(blockCoord);
            const int64_t nBlockIdx = AscendC::Te::Get<MNK_N>(blockCoord);
            const int64_t blockK = problemK;
            BlockShape blockShape{blockM, blockN, blockK, 0};
            const int64_t mPos = mBlockIdx * curBaseM_ + mSplitOffset;
            const int64_t nPos = nBlockIdx * baseN + nSplitOffset;

            auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mPos, static_cast<int64_t>(0)),
                                      AscendC::Te::MakeShape(blockM, blockK));
            auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(mPos, static_cast<int64_t>(0)),
                                                AscendC::Te::MakeShape(blockM, scaleK));
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), nPos),
                                      AscendC::Te::MakeShape(blockK, blockN));
            auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), nPos),
                                                AscendC::Te::MakeShape(scaleK, blockN));
            auto gmBlockBias = gmBias.Slice(AscendC::Te::MakeCoord(static_cast<int64_t>(0), nPos),
                                            AscendC::Te::MakeShape(static_cast<int64_t>(1), blockN));
            auto gmBlockC =
                gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos), AscendC::Te::MakeShape(blockM, blockN));
            mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, blockShape);
        } while (scheduler.GetNextBlockCoord(blockCoord));
    }

private:
    BlockMmad mmadOp_;
    ProblemShape problemShape_{};
    __gm__ AType *aBasePtr_{nullptr};
    __gm__ BType *bBasePtr_{nullptr};
    __gm__ CType *cBasePtr_{nullptr};
    __gm__ float *biasBasePtr_{nullptr};
    __gm__ ScaleAType *scaleABasePtr_{nullptr};
    __gm__ ScaleBType *scaleBBasePtr_{nullptr};
    GM_ADDR groupListGmAddr_{nullptr};
    GroupOffsetTables offsetTables_{};
    GMMTiling tiling_{};

    struct GroupState {
        int64_t preOffset{0}; // 已解析组的累计 token 总数（含当前组）
        int64_t perGroupB{0}; // 一组权重的步长（组间间隔）
    };
    struct GroupOffsets {
        int64_t a{0};
        int64_t scaleA{0};
        int64_t b{0};
        int64_t scaleB{0};
        int64_t c{0};
        int64_t bias{0};
    };

    GroupState groupState_{};
    GroupOffsets groupOffsets_{};
    uint32_t groupNum_{0};
    uint32_t curBaseM_{0};
    bool isBias_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
