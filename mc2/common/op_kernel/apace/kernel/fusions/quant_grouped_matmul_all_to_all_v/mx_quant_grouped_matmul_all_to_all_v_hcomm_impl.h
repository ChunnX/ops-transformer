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
 * \file mx_quant_grouped_matmul_all_to_all_v_hcomm_impl.h
 * \brief MX 量化 GroupedMatmul + AllToAllV 融合 kernel 实现（Hcomm/CCU 变体）
 *
 * 框架形态参考 apace all_gather_quant_matmul 的 Hcomm 通路（Hccl 类 + tiling 内嵌
 * Mc2InitTiling/Mc2CcTiling），按本算子“先计算后通信”语义改造：
 *
 * Init():
 *   全核: hccl_.InitV2()/SetCcTilingV2()；
 *   AIC : 将 groupList（SPARSE：[专家id, token数]）与 rank-first cGroupOffsetTable
 *         写入 user workspace（各 AIC 写入相同值，与原算子模板一致）。
 *
 * Run():
 *   AIC: GroupedMatmulMxKernel 单次 Run 完成全部 e*ep 组 GMM，输出直写 AllToAllV
 *        发送缓冲（rank-first）；随后 mode-0 全 AIC barrier 闭合跨核产出依赖，
 *        mode-2 flag 通知 AIV；共享专家 MM（groupNum=1）与 AIV 通信重叠执行。
 *   AIV: 全部 sub-core WaitFlag 后，block 0 以按 rank 聚合的
 *        cnt/offset 提交一次 AlltoAllV<true> 并 Wait；SyncAll 后 Finalize。
 */

#pragma once

#include "include/tensor_api/tensor.h"
#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "apace/kernel/matmul/grouped_matmul/grouped_matmul_mx_kernel.h"
#include "apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/quant_grouped_matmul_all_to_all_v_tiling_data.h"

namespace Apace {

using namespace AscendC;

template <typename T>
constexpr uint64_t GetA2avHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::HCCL_DATA_TYPE_BFP16;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP16;
    }
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
class MxQuantGroupedMatmulAllToAllvHcommImpl {
public:
    using TilingData = Apace::hcommQuantGmmAlltoAllvTilingData;

    explicit __aicore__ inline MxQuantGroupedMatmulAllToAllvHcommImpl(TilingData *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmWeightGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM,
                                GM_ADDR mmxGM, GM_ADDR mmWeightGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM,
                                GM_ADDR yGM, GM_ADDR mmyGM, GM_ADDR userWorkspace);
    __aicore__ inline void Run();

private:
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;

    // gmmWeight 不转置: [e, K, N] → B 为 ND；转置: [e, N, K] → B 为 DN
    template <bool TransB>
    using LayoutBOf = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;

    template <bool TransB>
    using GmmKernelOf = Blaze::Gemm::Kernel::GroupedMatmulMxKernel<
        ProblemShape,
        Blaze::Gemm::Block::BlockMmad<Blaze::Gemm::GroupedMatmulWithScaleMx<0, false>, AType, LayoutA, BType,
                                      LayoutBOf<TransB>, CType, LayoutC, float, LayoutC>,
        Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit>;

    __aicore__ inline void InitBaseParams();
    __aicore__ inline void BuildGmmTables();
    __aicore__ inline void CommitAllToAllv();

    // 通用单次 GMM：主 GMM（SPARSE groupList + cGroupOffsetTable）与共享专家 MM（单组 LENGTH）复用
    template <bool TransB>
    __aicore__ inline void RunGmmKernel(const QuantMatmulTilingData &mmTile, uint32_t groupNum, int64_t maxM, int64_t n,
                                        int64_t k, GM_ADDR xGM, GM_ADDR wGM, GM_ADDR xScaleGM, GM_ADDR wScaleGM,
                                        GM_ADDR cGM, GM_ADDR groupListGm, uint8_t groupListType,
                                        const __gm__ uint64_t *cGroupTable);

    TilingData *tilingData_{nullptr};

    AscendC::Hccl<ServerType> hccl_{};
    AscendC::HcclHandle a2avHandle_{0};

    // 输入/输出地址
    GM_ADDR gmmxGM_{};
    GM_ADDR gmmWeightGM_{};
    GM_ADDR gmmxScaleGM_{};
    GM_ADDR gmmWeightScaleGM_{};
    GM_ADDR mmxGM_{};
    GM_ADDR mmWeightGM_{};
    GM_ADDR mmxScaleGM_{};
    GM_ADDR mmWeightScaleGM_{};
    GM_ADDR yGM_{};
    GM_ADDR mmyGM_{};

    // workspace 内部地址
    GM_ADDR sendBufGM_{};         // GMM 输出 / AllToAllV 发送缓冲（rank-first）
    GM_ADDR groupListGM_{};       // 主 GMM groupList（SPARSE）
    GM_ADDR cTableGM_{};          // 主 GMM cGroupOffsetTable
    GM_ADDR sharedGroupListGM_{}; // 共享专家单组 groupList

    // 派生维度
    uint32_t e_{0};
    uint32_t ep_{0};
    uint32_t groupNum_{0};
    uint64_t h1_{0};
    uint64_t n1_{0};
    uint64_t bs_{0};
    bool hasSharedMm_{false};

    static constexpr uint64_t dataDtype_ = GetA2avHcclDataType<CType>();
    // 核间同步 flag（与原算子 GmmA2avScheduler 一致：8=AIC barrier, 9=GMM done）
    static constexpr uint16_t AIC_BARRIER_FLAG = 8U;
    static constexpr uint16_t GMM_DONE_FLAG = 9U;
};

// =================================================================================
// 初始化
// =================================================================================

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
__aicore__ inline void
MxQuantGroupedMatmulAllToAllvHcommImpl<AType, BType, CType, ServerType, TransGmmB, TransMmB>::InitBaseParams()
{
    const TilingData &t = *tilingData_;
    e_ = t.e;
    ep_ = t.epWorldSize;
    groupNum_ = static_cast<uint32_t>(e_) * ep_;
    h1_ = t.h1;
    n1_ = t.n1;
    bs_ = t.bs;
    hasSharedMm_ = (t.hasSharedMm != 0U);
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
__aicore__ inline void
MxQuantGroupedMatmulAllToAllvHcommImpl<AType, BType, CType, ServerType, TransGmmB, TransMmB>::BuildGmmTables()
{
    // groupList 按 (expertIdx * ep + rankIdx) 专家主序展开，与 gmmX 布局一致；
    // A/scaleA 使用 kernel 默认累计 offset，B/scaleB 由 SPARSE 专家 id 定位，
    // C 由 cGroupOffsetTable 写成 rank-first，直接满足 AllToAllV 发送布局。
    __gm__ int64_t *groupList = reinterpret_cast<__gm__ int64_t *>(groupListGM_);
    __gm__ uint64_t *cTable = reinterpret_cast<__gm__ uint64_t *>(cTableGM_);
    const TilingData &t = *tilingData_;

    uint64_t rankTokens[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};
    for (uint32_t r = 0U; r < ep_; ++r) {
        for (uint32_t eIdx = 0U; eIdx < e_; ++eIdx) {
            rankTokens[r] += static_cast<uint64_t>(t.sendCnt[eIdx + r * e_]);
        }
    }
    uint64_t rankBase[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};
    for (uint32_t r = 1U; r < ep_; ++r) {
        rankBase[r] = rankBase[r - 1U] + rankTokens[r - 1U] * n1_;
    }
    uint64_t runningOff[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};
    for (uint32_t r = 0U; r < ep_; ++r) {
        runningOff[r] = rankBase[r];
    }
    for (uint32_t eIdx = 0U; eIdx < e_; ++eIdx) {
        for (uint32_t r = 0U; r < ep_; ++r) {
            const uint32_t groupIdx = eIdx * ep_ + r;
            const uint64_t cnt = static_cast<uint64_t>(t.sendCnt[eIdx + r * e_]);
            groupList[2UL * groupIdx] = static_cast<int64_t>(eIdx); // SPARSE: 专家 id
            groupList[2UL * groupIdx + 1UL] = static_cast<int64_t>(cnt);
            cTable[groupIdx] = runningOff[r];
            runningOff[r] += cnt * n1_;
        }
    }

    if (hasSharedMm_) {
        *reinterpret_cast<__gm__ int64_t *>(sharedGroupListGM_) = static_cast<int64_t>(bs_);
    }
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
__aicore__ inline void
MxQuantGroupedMatmulAllToAllvHcommImpl<AType, BType, CType, ServerType, TransGmmB, TransMmB>::Init(
    GM_ADDR gmmxGM, GM_ADDR gmmWeightGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightScaleGM, GM_ADDR mmxGM,
    GM_ADDR mmWeightGM, GM_ADDR mmxScaleGM, GM_ADDR mmWeightScaleGM, GM_ADDR yGM, GM_ADDR mmyGM, GM_ADDR userWorkspace)
{
    gmmxGM_ = gmmxGM;
    gmmWeightGM_ = gmmWeightGM;
    gmmxScaleGM_ = gmmxScaleGM;
    gmmWeightScaleGM_ = gmmWeightScaleGM;
    mmxGM_ = mmxGM;
    mmWeightGM_ = mmWeightGM;
    mmxScaleGM_ = mmxScaleGM;
    mmWeightScaleGM_ = mmWeightScaleGM;
    yGM_ = yGM;
    mmyGM_ = mmyGM;

    InitBaseParams();

    sendBufGM_ = userWorkspace;
    groupListGM_ = userWorkspace + tilingData_->wsGroupListOffset;
    cTableGM_ = userWorkspace + tilingData_->wsCTableOffset;
    sharedGroupListGM_ = userWorkspace + tilingData_->wsSharedGroupListOffset;

    // HCCL 初始化（全部核，与 HcclA2avOp/Hcomm 参考实现一致）
    hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(TilingData, mc2CcTiling)));

    if ASCEND_IS_AIC {
        // 各 AIC 写入相同表内容后各自消费（与原算子 QuantGroupedMatmul 模板一致）
        BuildGmmTables();
    }
}

// =================================================================================
// GMM 计算封装
// =================================================================================

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
template <bool TransB>
__aicore__ inline void
MxQuantGroupedMatmulAllToAllvHcommImpl<AType, BType, CType, ServerType, TransGmmB, TransMmB>::RunGmmKernel(
    const QuantMatmulTilingData &mmTile, uint32_t groupNum, int64_t maxM, int64_t n, int64_t k, GM_ADDR xGM,
    GM_ADDR wGM, GM_ADDR xScaleGM, GM_ADDR wScaleGM, GM_ADDR cGM, GM_ADDR groupListGm, uint8_t groupListType,
    const __gm__ uint64_t *cGroupTable)
{
    using GmmKernel = GmmKernelOf<TransB>;

    typename GmmKernel::GMMTiling gmmTiling{};
    gmmTiling.groupNum = groupNum;
    gmmTiling.m = maxM;
    gmmTiling.n = n;
    gmmTiling.k = k;
    gmmTiling.baseM = mmTile.baseM;
    gmmTiling.baseN = mmTile.baseN;
    gmmTiling.baseK = mmTile.baseK;
    gmmTiling.kAL1 = static_cast<uint32_t>(mmTile.stepK) * mmTile.baseK;
    gmmTiling.kBL1 = gmmTiling.kAL1;
    gmmTiling.scaleKAL1 = mmTile.scaleKL1;
    gmmTiling.isBias = 0U;
    gmmTiling.dbL0C = mmTile.dbL0c;
    gmmTiling.l1BufferStage = mmTile.nBufferNum;
    gmmTiling.groupListType = groupListType;

    typename GmmKernel::Params params{};
    params.problemShape = ProblemShape{maxM, n, k, 0};
    params.mmadParams = typename GmmKernel::BlockMmadParams{xGM, wGM, cGM, nullptr, xScaleGM, wScaleGM};
    params.groupListGmAddr = groupListGm;
    params.gmmParams = gmmTiling;
    params.groupOffsetTables.cGroupAddrTable = cGroupTable;

    GmmKernel kernel;
    kernel.Init(params);
    kernel.Run();
}

// =================================================================================
// AllToAllV 通信（AIV block 0）
// =================================================================================

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
__aicore__ inline void
MxQuantGroupedMatmulAllToAllvHcommImpl<AType, BType, CType, ServerType, TransGmmB, TransMmB>::CommitAllToAllv()
{
    const TilingData &t = *tilingData_;
    uint64_t sendCnt[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};
    uint64_t sendOff[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};
    uint64_t recvCnt[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};
    uint64_t recvOff[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};

    // 按 rank 聚合 token 数（sendCnt/recvCnt 按 [expertId + rank * e] 展开）
    for (uint32_t r = 0U; r < ep_; ++r) {
        for (uint32_t eIdx = 0U; eIdx < e_; ++eIdx) {
            sendCnt[r] += static_cast<uint64_t>(t.sendCnt[eIdx + r * e_]);
            recvCnt[r] += static_cast<uint64_t>(t.recvCnt[eIdx + r * e_]);
        }
    }
    // 发送缓冲/y 均为 rank-first 行布局，偏移按聚合 token 数 × N1 折算为元素数
    for (uint32_t r = 1U; r < ep_; ++r) {
        sendOff[r] = sendOff[r - 1U] + sendCnt[r - 1U] * n1_;
        recvOff[r] = recvOff[r - 1U] + recvCnt[r - 1U] * n1_;
    }
    for (uint32_t r = 0U; r < ep_; ++r) {
        sendCnt[r] *= n1_;
        recvCnt[r] *= n1_;
    }

    const auto dtype = static_cast<AscendC::HcclDataType>(dataDtype_);
    a2avHandle_ = hccl_.template AlltoAllV<true>(sendBufGM_, sendCnt, sendOff, dtype, yGM_, recvCnt, recvOff, dtype);
    hccl_.Wait(a2avHandle_);
}

// =================================================================================
// 主流程
// =================================================================================

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
__aicore__ inline void
MxQuantGroupedMatmulAllToAllvHcommImpl<AType, BType, CType, ServerType, TransGmmB, TransMmB>::Run()
{
    if ASCEND_IS_AIC {
        // 主 GMM：全部 e*ep 组一次算完，输出直写 rank-first 发送缓冲
        RunGmmKernel<TransGmmB>(
            tilingData_->gmmTile, groupNum_, static_cast<int64_t>(tilingData_->maxM), static_cast<int64_t>(n1_),
            static_cast<int64_t>(h1_), gmmxGM_, gmmWeightGM_, gmmxScaleGM_, gmmWeightScaleGM_, sendBufGM_, groupListGM_,
            Apace::GMM_A2AV_GROUP_LIST_TYPE_SPARSE, reinterpret_cast<const __gm__ uint64_t *>(cTableGM_));

        // 全 AIC barrier 闭合跨核 GMM 产出依赖，再通知 AIV 可通信
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AIC_BARRIER_FLAG);
        AscendC::CrossCoreWaitFlag(AIC_BARRIER_FLAG);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(GMM_DONE_FLAG);

        // 共享专家 MM 与 AIV 通信重叠（mmY 不参与通信，无跨核依赖）
        if (hasSharedMm_) {
            RunGmmKernel<TransMmB>(tilingData_->sharedMmTile, 1U, static_cast<int64_t>(bs_),
                                   static_cast<int64_t>(tilingData_->n2), static_cast<int64_t>(tilingData_->h2), mmxGM_,
                                   mmWeightGM_, mmxScaleGM_, mmWeightScaleGM_, mmyGM_, sharedGroupListGM_,
                                   Apace::GMM_A2AV_GROUP_LIST_TYPE_LENGTH, nullptr);
        }
        AscendC::SyncAll<false>();
    }

    if ASCEND_IS_AIV {
        AscendC::CrossCoreWaitFlag(GMM_DONE_FLAG);
        if (GetBlockIdx() == 0) {
            CommitAllToAllv();
        }
        AscendC::SyncAll<false>();
        hccl_.Finalize();
    }
}

} // namespace Apace
