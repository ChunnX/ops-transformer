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
 * \file quant_grouped_matmul_all_to_all_v_impl.h
 * \brief apace 库：QuantGroupedMatMulAlltoAllv 算子 MX(FP4/FP8) 路径的 GMM 计算适配 kernel。
 * MX 量化路径复用 apace 模板库的 GroupedMatmulMxKernel
 * （Blaze block_mmad_qgmm_mx + block_scheduler_gmm_swat_with_tail_split）。
 *
 * 对外接口（Init/Process/End）与 MC2KernelTemplate::QuantGroupedMatmul 保持一致，
 * 供算子 kernel 入口通过 MX_QUANT_MODE 编译期宏选择实现。
 *
 * workspace 布局（路由专家 GMM，rankDim=epWorldSize>1 后通信场景）:
 *   [0, groupListSize)              groupList: (expert,rank) 粒度 token 数，expert-major 序
 *   [+, cGroupOffsetTableSize)      C 输出偏移表（rank-major 物理布局，元素偏移）
 *   [+, bGroupOffsetTableSize)      B 权重偏移表（expert 粒度，元素偏移）
 *   [+, scaleBGroupOffsetTableSize) ScaleB 偏移表（expert 粒度，元素偏移）
 *   [+, ptrTableSize)               ptrTable 占位（与 TT 路径布局对齐）
 * 共享专家（isLocal=true，单 group）: 仅 groupList（1 项），其余偏移走公式（全 0）。
 */

#ifndef APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_IMPL_H__
#define APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_IMPL_H__

#include "../../../../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"
#include "apace/kernel/matmul/grouped_matmul/grouped_matmul_mx_kernel.h"

namespace Apace {

using namespace AscendC;
using namespace MC2KernelTemplate;

// groupList 起始槽位（与 mc2_templates compute 侧约定一致，当前为 0）
constexpr uint64_t GMM_A2AV_GROUP_LIST_INDEX = 0UL;

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
class QuantGroupedMatmulMx {
public:
    __aicore__ inline QuantGroupedMatmulMx() {}
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR weightScaleGM, GM_ADDR yGM,
                                GM_ADDR workspaceGM, const TilingDataType *tilingData,
                                const GmmTilingDataType *gmmTilingData, MC2_TILING_TYPE *gmmArrayAddrIn, TPipe *tPipe,
                                bool isA2avGmmFlag = false);
    __aicore__ inline void Process(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void Process(uint32_t expertIdx);
    __aicore__ inline void End();

private:
    __aicore__ inline void UpdateAddr(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void BuildGroupListAndOffsetTables(uint32_t startExpertIdx, uint32_t expertNum);
    __aicore__ inline void InitApaceKernel(uint32_t startExpertIdx, uint32_t expertNum);

    // ---- apace 模板类型组装 ----
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false>;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    // weight 不转置: [e, K, N] → B 为 ND；转置: [e, N, K] → B 为 DN
    using LayoutB = AscendC::Std::conditional_t<bTrans, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, xType, LayoutA, wType, LayoutB, yType, LayoutC,
                                                    float, LayoutBias>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using MxApaceKernel = Blaze::Gemm::Kernel::GroupedMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler>;
    using GMMTiling = typename MxApaceKernel::GMMTiling;
    using KernelParams = typename MxApaceKernel::Params;
    using GroupOffsetTables = typename MxApaceKernel::GroupOffsetTables;

    static constexpr bool IS_FP4 = Blaze::Gemm::IsFp4<xType>();
    static constexpr uint64_t PACKED_SHIFT = IS_FP4 ? 1U : 0U;

    GM_ADDR xGM_;
    GM_ADDR wGM_;
    GM_ADDR xScaleGM_;
    GM_ADDR weightScaleGM_;
    GM_ADDR yGM_;
    // 固定基址（Init 一次绑定），UpdateAddr 按批从基址绝对推导当前地址，禁止相对累加
    GM_ADDR xBaseGM_;
    GM_ADDR wBaseGM_;
    GM_ADDR xScaleBaseGM_;
    GM_ADDR wScaleBaseGM_;
    GM_ADDR yBaseGM_;
    GM_ADDR groupListGm_;
    GM_ADDR cGroupOffsetTableGm_;
    GM_ADDR bGroupOffsetTableGm_;
    GM_ADDR scaleBGroupOffsetTableGm_;
    GM_ADDR workspaceGM_;

    const TilingDataType *tilingData_;
    const GmmTilingDataType *gmmTilingData_;
    TPipe *tPipe_;
    uint64_t expertTokenNum_[MAX_EXPERT_PER_RANK] = {0};
    uint64_t expertTokenOffset_ = 0;
    uint64_t expertNumInOneRank_ = 0;
    uint64_t epWorldSize_ = 0;
    uint64_t h1_ = 0;
    uint64_t n1_ = 0;
    uint64_t bs_ = 0;
    uint64_t a_ = 0;
    uint64_t mMaxSize_ = 0;
    MxApaceKernel mxKernel_;
};

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void
QuantGroupedMatmulMx<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat, aTrans, bTrans,
                     isLocal, isA2avGmm>::Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR weightScaleGM,
                                               GM_ADDR yGM, GM_ADDR workspaceGM, const TilingDataType *tilingData,
                                               const GmmTilingDataType *gmmTilingData, MC2_TILING_TYPE *gmmArrayAddrIn,
                                               TPipe *tPipe, bool isA2avGmmFlag)
{
    if ASCEND_IS_AIV {
        return;
    }
    (void)gmmArrayAddrIn;
    (void)isA2avGmmFlag;
    xGM_ = xGM;
    wGM_ = weightGM;
    xScaleGM_ = xScaleGM;
    weightScaleGM_ = weightScaleGM;
    yGM_ = yGM;
    xBaseGM_ = xGM;
    wBaseGM_ = weightGM;
    xScaleBaseGM_ = xScaleGM;
    wScaleBaseGM_ = weightScaleGM;
    yBaseGM_ = yGM;
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    workspaceGM_ = workspaceGM;
    gmmTilingData_ = gmmTilingData;

    expertNumInOneRank_ = tilingData_->taskTilingInfo.e;
    epWorldSize_ = tilingData_->taskTilingInfo.epWorldSize;
    h1_ = tilingData_->taskTilingInfo.H1;
    n1_ = tilingData_->taskTilingInfo.N1;
    bs_ = tilingData_->taskTilingInfo.BS;
    a_ = tilingData_->taskTilingInfo.A;

    if constexpr (!isLocal) {
        uint64_t groupListSize = sizeof(int64_t) * expertNumInOneRank_ * epWorldSize_;
        uint64_t cGroupOffsetTableSize = sizeof(uint64_t) * expertNumInOneRank_ * epWorldSize_;
        uint64_t bGroupOffsetTableSize = sizeof(uint64_t) * expertNumInOneRank_ * epWorldSize_;
        uint64_t scaleBGroupOffsetTableSize = sizeof(uint64_t) * expertNumInOneRank_ * epWorldSize_;
        groupListGm_ = workspaceGM_;
        cGroupOffsetTableGm_ = groupListGm_ + groupListSize;
        bGroupOffsetTableGm_ = cGroupOffsetTableGm_ + cGroupOffsetTableSize;
        scaleBGroupOffsetTableGm_ = bGroupOffsetTableGm_ + bGroupOffsetTableSize;
    } else {
        groupListGm_ = workspaceGM_;
        cGroupOffsetTableGm_ = nullptr;
        bGroupOffsetTableGm_ = nullptr;
        scaleBGroupOffsetTableGm_ = nullptr;
    }

    // 每个 expert 的 token 总数（跨 rank 累计）
    const auto *opCnt = &tilingData_->taskTilingInfo.sendCnt[0];
    mMaxSize_ = 0;
    for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
        expertTokenNum_[e] = 0U;
        for (uint32_t i = 0U; i < epWorldSize_; i++) {
            expertTokenNum_[e] += static_cast<uint64_t>(opCnt[e + i * expertNumInOneRank_]);
        }
        mMaxSize_ = (expertTokenNum_[e] > mMaxSize_) ? expertTokenNum_[e] : mMaxSize_;
    }
}

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void QuantGroupedMatmulMx<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat,
                                            aTrans, bTrans, isLocal, isA2avGmm>::Process(uint32_t expertIdx)
{
    Process(expertIdx, 1);
}

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void QuantGroupedMatmulMx<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat,
                                            aTrans, bTrans, isLocal, isA2avGmm>::Process(uint32_t startExpertIdx,
                                                                                         uint32_t expertNum)
{
    if ASCEND_IS_AIV {
        return;
    }
    if constexpr (!isLocal) {
        uint64_t expertTokenNum = 0U;
        for (uint32_t e = 0U; e < expertNum; e++) {
            expertTokenNum += expertTokenNum_[startExpertIdx + e];
        }
        if (expertTokenNum == 0) {
            return;
        }
    }

    BuildGroupListAndOffsetTables(startExpertIdx, expertNum);
    UpdateAddr(startExpertIdx, expertNum);
    InitApaceKernel(startExpertIdx, expertNum);
    mxKernel_.Run();
}

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void QuantGroupedMatmulMx<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat,
                                            aTrans, bTrans, isLocal, isA2avGmm>::End()
{
    if ASCEND_IS_AIV {
        return;
    }
}

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void
QuantGroupedMatmulMx<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat, aTrans, bTrans,
                     isLocal, isA2avGmm>::BuildGroupListAndOffsetTables(uint32_t startExpertIdx, uint32_t expertNum)
{
    const auto *opCnt = &tilingData_->taskTilingInfo.sendCnt[0];
    __gm__ int64_t *groupListPtr = reinterpret_cast<__gm__ int64_t *>(groupListGm_);
    if constexpr (!isLocal) {
        __gm__ uint64_t *cGroupOffsetTable = reinterpret_cast<__gm__ uint64_t *>(cGroupOffsetTableGm_);
        __gm__ uint64_t *bGroupOffsetTable = reinterpret_cast<__gm__ uint64_t *>(bGroupOffsetTableGm_);
        __gm__ uint64_t *scaleBGroupOffsetTable = reinterpret_cast<__gm__ uint64_t *>(scaleBGroupOffsetTableGm_);

        uint64_t rankTotalInBatch[MAX_EP_RANK_SIZE] = {0};
        for (uint32_t r = 0; r < epWorldSize_; r++) {
            for (uint32_t e = 0; e < expertNum; e++) {
                uint32_t absExpertIdx = startExpertIdx + e;
                uint64_t cnt = static_cast<uint64_t>(opCnt[absExpertIdx + r * expertNumInOneRank_]);
                groupListPtr[GMM_A2AV_GROUP_LIST_INDEX + e * epWorldSize_ + r] = static_cast<int64_t>(cnt);
                rankTotalInBatch[r] += cnt;
            }
        }

        // C 输出偏移表：rank-major 物理布局（元素偏移，相对本批 C 基址）
        uint64_t rankBaseElemOffset[MAX_EP_RANK_SIZE] = {0};
        for (uint32_t r = 1; r < epWorldSize_; r++) {
            rankBaseElemOffset[r] = rankBaseElemOffset[r - 1] + rankTotalInBatch[r - 1] * n1_;
        }
        uint64_t expertRunningElemOffset[MAX_EP_RANK_SIZE] = {0};
        for (uint32_t r = 0; r < epWorldSize_; r++) {
            expertRunningElemOffset[r] = rankBaseElemOffset[r];
        }
        for (uint32_t e = 0; e < expertNum; e++) {
            for (uint32_t r = 0; r < epWorldSize_; r++) {
                uint32_t absExpertIdx = startExpertIdx + e;
                uint32_t groupIdx = e * epWorldSize_ + r;
                cGroupOffsetTable[groupIdx] = expertRunningElemOffset[r];
                expertRunningElemOffset[r] +=
                    static_cast<uint64_t>(opCnt[absExpertIdx + r * expertNumInOneRank_]) * n1_;
            }
        }

        // B 权重偏移表：expert 粒度（相对本批 W 基址，元素偏移）；
        // group 序 (e*ep+r) 与 expert 序不一致，无法走 kernel 默认公式，需显式建表。
        // ScaleB 偏移表：expert 粒度
        const uint64_t perExpertW = (h1_ * n1_) >> PACKED_SHIFT;
        const uint64_t scaleK = ((h1_ + Blaze::Gemm::MXFP_DIVISOR_SIZE - 1) >> Blaze::Gemm::MXFP_DIVISOR_SHIFT)
                                << Blaze::Gemm::MXFP_MULTI_BASE_SHIFT;
        for (uint32_t e = 0; e < expertNum; e++) {
            for (uint32_t r = 0; r < epWorldSize_; r++) {
                uint32_t groupIdx = e * epWorldSize_ + r;
                bGroupOffsetTable[groupIdx] = static_cast<uint64_t>(e) * perExpertW;
                scaleBGroupOffsetTable[groupIdx] = static_cast<uint64_t>(e) * n1_ * scaleK;
            }
        }
    } else {
        groupListPtr[GMM_A2AV_GROUP_LIST_INDEX] = static_cast<int64_t>(bs_);
    }
}

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void QuantGroupedMatmulMx<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat,
                                            aTrans, bTrans, isLocal, isA2avGmm>::UpdateAddr(uint32_t startExpertIdx,
                                                                                            uint32_t expertNum)
{
    uint64_t expertTokenNum = 0U;
    for (uint32_t e = 0U; e < expertNum; e++) {
        expertTokenNum += expertTokenNum_[startExpertIdx + e];
    }
    // A/B 为 MX 量化输入，FP4 时每 2 元素打包 1 字节，偏移需右移 1 位（与 apace 模板公式一致）；
    // A/scaleA 基址按批推进至全局 token 偏移（gmmX 为 expert-major 物理布局，走公式），
    // B/scaleB 基址推进至本批首个 expert（批内经偏移表按 expert 定位）；
    // 全部从固定基址绝对推导（expertTokenOffset_ 为批前累计，相对累加会重复计入历史批）
    const uint64_t scaleK = ((h1_ + Blaze::Gemm::MXFP_DIVISOR_SIZE - 1) >> Blaze::Gemm::MXFP_DIVISOR_SHIFT)
                            << Blaze::Gemm::MXFP_MULTI_BASE_SHIFT;
    xGM_ = xBaseGM_ + (((expertTokenOffset_ * h1_) >> PACKED_SHIFT) * sizeof(xType));
    wGM_ = wBaseGM_ + (((startExpertIdx * h1_ * n1_) >> PACKED_SHIFT) * sizeof(wType));
    yGM_ = yBaseGM_ + (expertTokenOffset_ * n1_ * sizeof(yType));
    xScaleGM_ = xScaleBaseGM_ + (expertTokenOffset_ * scaleK * sizeof(scaleType));
    weightScaleGM_ = wScaleBaseGM_ + (startExpertIdx * n1_ * scaleK * sizeof(scaleType));

    expertTokenOffset_ += expertTokenNum;
}

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
          CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
__aicore__ inline void
QuantGroupedMatmulMx<TilingDataType, GmmTilingDataType, xType, wType, scaleType, yType, wFormat, aTrans, bTrans,
                     isLocal, isA2avGmm>::InitApaceKernel(uint32_t startExpertIdx, uint32_t expertNum)
{
    (void)startExpertIdx;
    const auto *mmTilingData = &gmmTilingData_->mmTilingData;
    const auto *gmmQuantParams = &gmmTilingData_->gmmQuantParams;

    // 调度上界 M：路由专家取全专家最大 token 数；共享专家（isLocal 单组）实际行数为 BS
    const int64_t maxM = isLocal ? static_cast<int64_t>(bs_) : static_cast<int64_t>(mMaxSize_);
    GMMTiling gmmParams;
    gmmParams.groupNum = isLocal ? 1U : expertNum * static_cast<uint32_t>(epWorldSize_);
    gmmParams.m = maxM;
    gmmParams.n = static_cast<int64_t>(n1_);
    gmmParams.k = static_cast<int64_t>(h1_);
    gmmParams.baseM = static_cast<uint32_t>(mmTilingData->baseM);
    gmmParams.baseN = static_cast<uint32_t>(mmTilingData->baseN);
    gmmParams.baseK = static_cast<uint32_t>(mmTilingData->baseK);
    gmmParams.kAL1 = 0;      // 兜底为 baseK（正确性优先）
    gmmParams.kBL1 = 0;      // 兜底为 baseK
    gmmParams.scaleKAL1 = 0; // 兜底为 baseK
    gmmParams.isBias = static_cast<uint8_t>(mmTilingData->isBias != 0 ? 1 : 0);
    gmmParams.dbL0C = static_cast<uint8_t>(mmTilingData->dbL0C);
    gmmParams.l1BufferStage = Blaze::Gemm::DOUBLE_BUFFER_COUNT;
    gmmParams.groupListType = gmmQuantParams->groupListType;

    typename MxApaceKernel::BlockMmadParams mmadParams;
    mmadParams.aGmAddr = xGM_;
    mmadParams.bGmAddr = wGM_;
    mmadParams.cGmAddr = yGM_;
    mmadParams.biasGmAddr = nullptr;
    mmadParams.scaleAGmAddr = xScaleGM_;
    mmadParams.scaleBGmAddr = weightScaleGM_;

    GroupOffsetTables offsetTables;
    if constexpr (!isLocal) {
        offsetTables.aGroupAddrTable = nullptr; // A 走公式（expert-major 累积）
        offsetTables.scaleAGroupAddrTable = nullptr;
        offsetTables.cGroupAddrTable = reinterpret_cast<const __gm__ uint64_t *>(cGroupOffsetTableGm_);
        offsetTables.bGroupAddrTable = reinterpret_cast<const __gm__ uint64_t *>(bGroupOffsetTableGm_);
        offsetTables.scaleBGroupAddrTable = reinterpret_cast<const __gm__ uint64_t *>(scaleBGroupOffsetTableGm_);
    }

    KernelParams params;
    params.problemShape = ProblemShape{maxM, static_cast<int64_t>(n1_), static_cast<int64_t>(h1_), 0};
    params.mmadParams = mmadParams;
    params.groupListGmAddr = groupListGm_;
    params.gmmParams = gmmParams;
    params.groupOffsetTables = offsetTables;

    mxKernel_.Init(params);
    (void)expertNum;
}

} // namespace Apace
#endif // APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_IMPL_H__
