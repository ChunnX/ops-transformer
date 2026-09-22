/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mx_quant_grouped_matmul_all_to_all_v_hcomm_impl.h
 * \brief MX 量化 GroupedMatmul + AllToAllV 融合 kernel 实现（Hcomm/CCU 变体，AIC-only）
 *
 * 框架形态对齐镜像算子 PR 11573（allto_allv_quant_grouped_matmul）的已验证 AIC-only 模型：
 * kernel 以 KERNEL_TYPE_AIC_ONLY 启动（tiling key 经 ASCENDC_TPL_KERNEL_TYPE_SEL 声明），
 * 无 AIV 参与、无 AIC/AIV 跨核 flag；全部 AIC 核执行同一流程，Hccl 客户端经
 * workingFlag_（GetBlockIdx()==0 的核）自动门控参数写入与 CCU doorbell。
 *
 * Init()（全部 AIC 核）:
 *   hccl_.InitV2()/SetCcTilingV2()；BuildGmmTables 写 groupList（SPARSE）与
 *   rank-first cGroupOffsetTable（各核写相同值）。
 *
 * Run()（专家分批通算掩盖流水，expertNumPerBatch = CeilDiv(e, 4)，totalLoopCnt ≤ 4）:
 *   for 批b: RunGmmKernel(本批，groupList/cTable 切片 + A/scaleA 基址推进)
 *            → PipeBarrier（排空管线，镜像经验）→ mode-0 全 AIC barrier
 *            （后通信方向必需：保证本批 sendBuf 多核写全部落 GM 后才可通信）
 *            → hccl_.AlltoAllV<true>(本批 chunk)（全核调用，CCU 异步执行）
 *   共享专家 MM（与末批通信重叠）→ 逐 handle Wait → Finalize。
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

    // 分批流水辅助
    __aicore__ inline uint32_t GetExpertNumPerBatch() const;
    __aicore__ inline uint64_t BatchTokenCnt(uint32_t startExpertIdx, uint32_t expertNum) const;
    __aicore__ inline AscendC::HcclHandle SubmitAllToAllvBatch(uint32_t startExpertIdx, uint32_t expertNum);

    // 通用单次 GMM：分批主 GMM（SPARSE groupList + cGroupOffsetTable 切片）与共享专家 MM（单组 LENGTH）复用
    template <bool TransB>
    __aicore__ inline void RunGmmKernel(const QuantMatmulTilingData &mmTile, uint32_t groupNum, int64_t maxM, int64_t n,
                                        int64_t k, GM_ADDR xGM, GM_ADDR wGM, GM_ADDR xScaleGM, GM_ADDR wScaleGM,
                                        GM_ADDR cGM, GM_ADDR groupListGm, uint8_t groupListType,
                                        const __gm__ uint64_t *cGroupTable);

    TilingData *tilingData_{nullptr};

    AscendC::Hccl<ServerType> hccl_{};
    AscendC::HcclHandle a2avHandles_[Apace::GMM_A2AV_MAX_COMM_LOOP_CNT]{};

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
    GM_ADDR sendBufGM_{};         // GMM 输出 / AllToAllV 发送缓冲（全局 rank-first）
    GM_ADDR groupListGM_{};       // 主 GMM groupList（SPARSE，专家主序）
    GM_ADDR cTableGM_{};          // 主 GMM cGroupOffsetTable（全局 rank-first 元素偏移）
    GM_ADDR sharedGroupListGM_{}; // 共享专家单组 groupList

    // 派生维度
    uint32_t e_{0};
    uint32_t ep_{0};
    uint64_t h1_{0};
    uint64_t n1_{0};
    uint64_t bs_{0};
    uint64_t scaleKLen_{0}; // 每 token 的 scaleA 元素数 = CeilDiv(H1,64)*2
    bool hasSharedMm_{false};

    static constexpr uint64_t dataDtype_ = GetA2avHcclDataType<CType>();
    // 全 AIC barrier flag（AIC-only 模型，全部核 Set+Wait；避开 CANN 内部 SyncAll 保留 ID 11~14）
    static constexpr uint16_t AIC_BARRIER_FLAG = 8U;
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
    h1_ = t.h1;
    n1_ = t.n1;
    bs_ = t.bs;
    hasSharedMm_ = (t.hasSharedMm != 0U);
    scaleKLen_ = (h1_ + Blaze::Gemm::MXFP_DIVISOR_SIZE - 1UL) / Blaze::Gemm::MXFP_DIVISOR_SIZE *
                 Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
__aicore__ inline void
MxQuantGroupedMatmulAllToAllvHcommImpl<AType, BType, CType, ServerType, TransGmmB, TransMmB>::BuildGmmTables()
{
    // groupList 按 (expertIdx * ep + rankIdx) 专家主序展开，与 gmmX 布局一致；
    // A/scaleA 使用 kernel 默认累计 offset，B/scaleB 由 SPARSE 专家 id 定位，
    // C 由 cGroupOffsetTable 写成全局 rank-first（分批通信按不相交连续 chunk 读取）。
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

    // HCCL 初始化（全部 AIC 核；workingFlag_ 仅 block 0 真正写共享状态，镜像同款）
    hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(TilingData, mc2CcTiling)));

    // 各核写入相同表内容后各自消费（与原算子 QuantGroupedMatmul 模板一致）
    BuildGmmTables();
}

// =================================================================================
// 分批流水辅助
// =================================================================================

// 与原通路 SetTilingCommonInfo 的 TARGET_COMM_ROUNDS=4 口径一致：
// expertNumPerBatch = CeilDiv(e, 4)，totalLoopCnt = CeilDiv(e, expertNumPerBatch) ≤ 4
template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
__aicore__ inline uint32_t MxQuantGroupedMatmulAllToAllvHcommImpl<AType, BType, CType, ServerType, TransGmmB,
                                                                  TransMmB>::GetExpertNumPerBatch() const
{
    constexpr uint32_t targetCommRounds = 4U;
    uint32_t expertNum = (e_ + targetCommRounds - 1U) / targetCommRounds;
    if (expertNum == 0U) {
        expertNum = 1U;
    }
    // 兜底：handle 数组上限约束（e ≤ 32 时 totalLoopCnt ≤ 4，不触发）
    const uint32_t minNumForHandleLimit =
        (e_ + Apace::GMM_A2AV_MAX_COMM_LOOP_CNT - 1U) / Apace::GMM_A2AV_MAX_COMM_LOOP_CNT;
    if (expertNum < minNumForHandleLimit) {
        expertNum = minNumForHandleLimit;
    }
    return expertNum;
}

// 本批全部 rank 的 token 总数（gmmX 专家主序下的本批 token 跨度）
template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
__aicore__ inline uint64_t MxQuantGroupedMatmulAllToAllvHcommImpl<AType, BType, CType, ServerType, TransGmmB,
                                                                  TransMmB>::BatchTokenCnt(uint32_t startExpertIdx,
                                                                                           uint32_t expertNum) const
{
    uint64_t tokens = 0UL;
    for (uint32_t eIdx = startExpertIdx; eIdx < startExpertIdx + expertNum; ++eIdx) {
        for (uint32_t r = 0U; r < ep_; ++r) {
            tokens += static_cast<uint64_t>(tilingData_->sendCnt[eIdx + r * e_]);
        }
    }
    return tokens;
}

// 提交本批 AllToAllV（非阻塞，返回 handle）：在全局 rank-first 的 sendBuf/y 布局中
// 读取/写入本批的不相交连续 chunk：
//   sendOff[r] = (rank r 全局基 + rank r 在 start 之前专家的 token 前缀) × N1
//   recvOff[r] 同构（y 按 recvCounts 源 rank 聚合顺序）
template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
__aicore__ inline AscendC::HcclHandle
MxQuantGroupedMatmulAllToAllvHcommImpl<AType, BType, CType, ServerType, TransGmmB, TransMmB>::SubmitAllToAllvBatch(
    uint32_t startExpertIdx, uint32_t expertNum)
{
    const TilingData &t = *tilingData_;
    uint64_t sendCnt[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};
    uint64_t sendOff[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};
    uint64_t recvCnt[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};
    uint64_t recvOff[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};

    // 一遍扫描：每 rank 全局 token 基（rank-first 布局）+ 本批计数
    uint64_t rankBaseS[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};
    uint64_t rankBaseR[Apace::GMM_A2AV_MAX_RANK_NUM] = {0UL};
    uint64_t runS = 0UL;
    uint64_t runR = 0UL;
    for (uint32_t r = 0U; r < ep_; ++r) {
        rankBaseS[r] = runS;
        rankBaseR[r] = runR;
        uint64_t batchS = 0UL;
        uint64_t batchR = 0UL;
        for (uint32_t eIdx = 0U; eIdx < e_; ++eIdx) {
            const uint64_t s = static_cast<uint64_t>(t.sendCnt[eIdx + r * e_]);
            const uint64_t rc = static_cast<uint64_t>(t.recvCnt[eIdx + r * e_]);
            runS += s;
            runR += rc;
            if (eIdx >= startExpertIdx && eIdx < startExpertIdx + expertNum) {
                batchS += s;
                batchR += rc;
            }
        }
        sendCnt[r] = batchS;
        recvCnt[r] = batchR;
    }
    // 偏移：rank 全局基 + 本批之前专家的 token 前缀（× N1 折算元素数）
    for (uint32_t r = 0U; r < ep_; ++r) {
        uint64_t preS = 0UL;
        uint64_t preR = 0UL;
        for (uint32_t eIdx = 0U; eIdx < startExpertIdx; ++eIdx) {
            preS += static_cast<uint64_t>(t.sendCnt[eIdx + r * e_]);
            preR += static_cast<uint64_t>(t.recvCnt[eIdx + r * e_]);
        }
        sendOff[r] = (rankBaseS[r] + preS) * n1_;
        recvOff[r] = (rankBaseR[r] + preR) * n1_;
    }
    for (uint32_t r = 0U; r < ep_; ++r) {
        sendCnt[r] *= n1_;
        recvCnt[r] *= n1_;
    }

    const auto dtype = static_cast<AscendC::HcclDataType>(dataDtype_);
    return hccl_.template AlltoAllV<true>(sendBufGM_, sendCnt, sendOff, dtype, yGM_, recvCnt, recvOff, dtype);
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
// 主流程（AIC-only 专家分批通算掩盖流水）
// =================================================================================

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool TransGmmB,
          bool TransMmB>
__aicore__ inline void
MxQuantGroupedMatmulAllToAllvHcommImpl<AType, BType, CType, ServerType, TransGmmB, TransMmB>::Run()
{
    const uint32_t expertNumPerBatch = GetExpertNumPerBatch();

    // 主 GMM 分批：本批 A/scaleA 基址推进至全局 token 偏移（gmmX 专家主序），
    // groupList/cTable 按批切片；提交本批 AllToAllV 后立即进入下一批 GMM（通算掩盖）
    uint64_t batchTokenOffset = 0UL;
    uint32_t submitted = 0U;
    for (uint32_t start = 0U; start < e_; start += expertNumPerBatch) {
        const uint32_t actualExpertNum = (start + expertNumPerBatch > e_) ? (e_ - start) : expertNumPerBatch;
        // FP4 打包口径与 kernel 内默认 A offset 一致：元素数 (token×K)>>1
        const uint64_t aElemUnits =
            Blaze::Gemm::IsFp4<AType>() ? ((batchTokenOffset * h1_) >> 1U) : (batchTokenOffset * h1_);
        RunGmmKernel<TransGmmB>(
            tilingData_->gmmTile, actualExpertNum * ep_, static_cast<int64_t>(tilingData_->maxM),
            static_cast<int64_t>(n1_), static_cast<int64_t>(h1_), gmmxGM_ + aElemUnits * sizeof(AType), gmmWeightGM_,
            gmmxScaleGM_ + batchTokenOffset * scaleKLen_, gmmWeightScaleGM_, sendBufGM_,
            groupListGM_ + static_cast<uint64_t>(start) * ep_ * 2UL * sizeof(int64_t),
            Apace::GMM_A2AV_GROUP_LIST_TYPE_SPARSE,
            reinterpret_cast<const __gm__ uint64_t *>(cTableGM_) + static_cast<uint64_t>(start) * ep_);

        // 镜像经验：GMM Run 后排空管线，保证 fixpipe 产出全部落 GM
        AscendC::PipeBarrier<PIPE_ALL>();

        // 后通信方向必需的全 AIC barrier：本批 sendBuf 由多核分块写入，
        // 必须全部落 GM 后才能提交本批 AllToAllV（mode-0 全核同步，与 CANN SyncAllImpl 同机制）
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AIC_BARRIER_FLAG);
        AscendC::CrossCoreWaitFlag(AIC_BARRIER_FLAG);

        // 全核提交（Hccl 客户端 workingFlag_ 自动门控：仅 block 0 写参数/敲 CCU doorbell，
        // 其余核仅推进本地句柄计数；CCU 异步执行，与下一批 GMM 重叠）
        a2avHandles_[submitted] = SubmitAllToAllvBatch(start, actualExpertNum);
        submitted++;
        batchTokenOffset += BatchTokenCnt(start, actualExpertNum);
    }

    // 共享专家 MM 与末批 AllToAllV 重叠（mmY 不参与通信，无跨核依赖）
    if (hasSharedMm_) {
        RunGmmKernel<TransMmB>(tilingData_->sharedMmTile, 1U, static_cast<int64_t>(bs_),
                               static_cast<int64_t>(tilingData_->n2), static_cast<int64_t>(tilingData_->h2), mmxGM_,
                               mmWeightGM_, mmxScaleGM_, mmWeightScaleGM_, mmyGM_, sharedGroupListGM_,
                               Apace::GMM_A2AV_GROUP_LIST_TYPE_LENGTH, nullptr);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    // 逐 handle Wait：多核协议下非工作核轮询 finishCntGM_（由工作核在 CKE 完成后推进）
    for (uint32_t i = 0U; i < submitted; ++i) {
        hccl_.Wait(a2avHandles_[i]);
    }
    hccl_.Finalize();
}

} // namespace Apace
