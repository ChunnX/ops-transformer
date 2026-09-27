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
 * \file quant_grouped_matmul_all_to_all_v_hcomm_impl.h
 * \brief AIC-only HCCL AllToAllV + Blaze MX GroupedMatMul implementation (GMM first, then comm).
 *
 * 结构对齐镜像算子 allto_allv_quant_grouped_mat_mul apace hcomm 实现的最新代码：
 *   - 直接复用算子 legacy tiling 数据（QuantGmmA2avTilingData），无独立紧凑结构，
 *     taskTilingInfo / gmmBaseTiling / sharedGmmTiling / hcclA2avTiling 设备侧直读；
 *   - SPARSE groupList（专家 id + token 数）与共享专家 mmGroupList 由 block0 在
 *     workspace 中构建后 SyncAll，B/scaleB 经 weightIdx 走 kernel 默认公式；
 *   - 每批 Blaze GroupedMatmulMxKernel 以栈上局部对象构造（Init + Run 后即析构）。
 *
 * 流水参考 gmm_a2av_scheduler.h 的批循环调度：批内 GMM 计算 → 核间同步 → 提交
 * 本批非阻塞 AllToAllV（通信由引擎异步执行，与后续批次计算重叠）；共享专家在批
 * 循环后执行（与末批通信重叠）；尾部统一 Wait 全部 handle → SyncAll → Finalize。
 *
 * workspace 布局（后通信模式，apace 通路）：
 *   [0, wsGmmOutputSize)                 → 路由专家 GMM 输出（sendBuf，批内 rank-major）
 *   [+, e*R*2*8)                         → SPARSE groupList（全批一次构建）
 *   [+, e*R*8)                           → C 输出偏移表（全批一次构建，按批切片消费）
 *   [+, 8)                               → 共享专家 mmGroupList（[0] = BS）
 *   （以上表格区复用原 gmmComputeWS 空间，容量 e*R*32+512+... ≥ e*R*24+8 恒成立）
 */

#ifndef APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__
#define APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__

#include "lib/hccl/hccl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "apace/kernel/matmul/grouped_matmul/grouped_matmul_mx_kernel.h"

namespace Apace {

template <typename T>
constexpr AscendC::HcclDataType GetGmmA2avYHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::HCCL_DATA_TYPE_BFP16;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP16;
    }
}

template <typename TilingDataType, typename AType, typename BType, typename CType, bool TransB, bool TransMmB,
          AscendC::HcclServerType ServerType>
class QuantGmmA2avHcommImpl {
public:
    explicit __aicore__ inline QuantGmmA2avHcommImpl(const TilingDataType *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR gmmX, GM_ADDR gmmXScale, GM_ADDR gmmWeight, GM_ADDR gmmWeightScale, GM_ADDR mmX,
                                GM_ADDR mmXScale, GM_ADDR mmWeight, GM_ADDR mmWeightScale, GM_ADDR mmY, GM_ADDR y,
                                GM_ADDR workspace)
    {
        xGM_ = gmmX;
        xScaleGM_ = gmmXScale;
        weightGM_ = gmmWeight;
        weightScaleGM_ = gmmWeightScale;
        yGM_ = y;
        mmXGM_ = mmX;
        mmXScaleGM_ = mmXScale;
        mmWeightGM_ = mmWeight;
        mmWeightScaleGM_ = mmWeightScale;
        mmYGM_ = mmY;
        gmmOutputGM_ = workspace;
        sparseGroupListGM_ = workspace + tilingData_->workspaceInfo.wsGmmOutputSize;
        cGroupOffsetTableGM_ = sparseGroupListGM_ + RankSize() * ExpertNum() * 2U * sizeof(int64_t);
        mmGroupListGM_ = cGroupOffsetTableGM_ + RankSize() * ExpertNum() * sizeof(uint64_t);
        BuildGroupLists();

        hccl_.InitV2(AscendC::GetHcclContext<0>(), &tilingData_->hcclA2avTiling.hcclInitTiling);
        hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(TilingDataType, hcclA2avTiling)) +
                            static_cast<uint64_t>(offsetof(MC2KernelTemplate::HcclA2avTilingInfo, a2avCcTiling)));
    }

    __aicore__ inline void Run()
    {
        uint64_t tokenPrefix = 0;
        for (uint32_t batch = 0; batch < BatchCount(); ++batch) {
            const uint64_t batchTokens = GetBatchTokenCount(batch);
            if (batchTokens != 0) {
                RunBatchMatmul(batch, tokenPrefix, batchTokens);
            }
            // AIC 核间同步：本批 sendBuf 多核写全部落 GM 后才可提交通信；
            // 空批也照常提交，保证各 rank 集合通信次数对齐
            AscendC::SyncAll();
            SubmitBatchComm(batch, tokenPrefix);
            // 排空本核数据搬移管线（含 block0 侧参数区拷贝）后再复用参数数组
            AscendC::PipeBarrier<PIPE_ALL>();
            tokenPrefix += batchTokens;
        }

        if (tilingData_->taskTilingInfo.BS != 0U && tilingData_->taskTilingInfo.H2 != 0U &&
            tilingData_->taskTilingInfo.N2 != 0U) {
            RunLocalMatmul();
        }

        for (uint32_t i = 0U; i < submitted_; ++i) {
            hccl_.Wait(a2avHandles_[i]);
        }
        AscendC::SyncAll();
        hccl_.Finalize();
    }

private:
    static constexpr uint32_t A2AV_GMM_MAX_RANK_NUM = 256U;
    static constexpr uint32_t GMM_A2AV_MAX_BATCH_NUM = 8U;
    static constexpr uint32_t MX_BASIC_FACTOR = 64U;
    static constexpr uint8_t GROUP_LIST_TYPE_OFFSET = 0U;
    static constexpr uint8_t GROUP_LIST_TYPE_SPARSE = 2U;
    static constexpr AscendC::HcclDataType Y_HCCL_DATA_TYPE = GetGmmA2avYHcclDataType<CType>();

    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGroupedMmadWithScaleMx>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    float, LayoutBias>;
    using GmmKernel = Blaze::Gemm::Kernel::GroupedMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler>;

    using MmLayoutB = AscendC::Std::conditional_t<TransMmB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using MmBlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, MmLayoutB, CType, LayoutC,
                                                      float, LayoutBias>;
    using MmKernel = Blaze::Gemm::Kernel::GroupedMatmulMxKernel<ProblemShape, MmBlockMmad, BlockScheduler>;

    __aicore__ inline uint32_t RankSize() const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.epWorldSize);
    }

    __aicore__ inline uint32_t ExpertNum() const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.e);
    }

    __aicore__ inline uint32_t BatchExpertNum() const
    {
        return tilingData_->taskTilingInfo.expertNum;
    }

    __aicore__ inline uint32_t BatchCount() const
    {
        return tilingData_->taskTilingInfo.totalLoopCount;
    }

    __aicore__ inline uint32_t GetSendRows(uint32_t expertIdx, uint32_t peer) const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.sendCnt[peer * ExpertNum() + expertIdx]);
    }

    __aicore__ inline uint32_t GetRecvRows(uint32_t expertIdx, uint32_t peer) const
    {
        return static_cast<uint32_t>(tilingData_->taskTilingInfo.recvCnt[peer * ExpertNum() + expertIdx]);
    }

    __aicore__ inline uint32_t ScaleKLen() const
    {
        return static_cast<uint32_t>((tilingData_->taskTilingInfo.H1 + MX_BASIC_FACTOR - 1U) / MX_BASIC_FACTOR * 2U);
    }

    __aicore__ inline void BuildGroupLists()
    {
        if (AscendC::GetBlockIdx() == 0U) {
            const uint32_t rankSize = RankSize();
            const uint32_t expertNum = ExpertNum();
            const uint32_t batchExpertNum = BatchExpertNum();
            const uint32_t n = static_cast<uint32_t>(tilingData_->taskTilingInfo.N1);
            __gm__ int64_t *sparseGroupList = reinterpret_cast<__gm__ int64_t *>(sparseGroupListGM_);
            __gm__ uint64_t *cGroupOffsetTable = reinterpret_cast<__gm__ uint64_t *>(cGroupOffsetTableGM_);

            // SPARSE groupList：组序 expert-major（g = expert * rankSize + rank）
            //   [g*2] = expert（Blaze weightIdx，直接索引 gmmWeight/gmmWeightScale 专家维）
            //   [g*2+1] = sendCnt[rank * e + expert]（gmmX 同序行数）
            for (uint32_t expert = 0U; expert < expertNum; ++expert) {
                for (uint32_t rank = 0U; rank < rankSize; ++rank) {
                    const uint32_t segment = expert * rankSize + rank;
                    sparseGroupList[segment * 2U] = static_cast<int64_t>(expert);
                    sparseGroupList[segment * 2U + 1U] = static_cast<int64_t>(GetSendRows(expert, rank));
                }
            }

            // C 偏移表：批内 rank-major 物理布局（元素偏移，相对各批 C 基址），
            // 全批一次构建，批 b 消费 [b*batchExpertNum*R, (b+E_b)*R) 切片：
            //   group(jj, r) 偏移 = rankBase_b[r] + Σ_{jj'<jj} cnt(start+jj', r) * n
            for (uint32_t batch = 0U; batch < BatchCount(); ++batch) {
                const uint32_t start = batch * batchExpertNum;
                const uint32_t actualExpertNum =
                    (start + batchExpertNum > expertNum) ? expertNum - start : batchExpertNum;
                const uint32_t batchGroupBase = start * rankSize;

                uint64_t rankTotal[A2AV_GMM_MAX_RANK_NUM] = {0};
                for (uint32_t jj = 0U; jj < actualExpertNum; ++jj) {
                    for (uint32_t rank = 0U; rank < rankSize; ++rank) {
                        rankTotal[rank] += static_cast<uint64_t>(GetSendRows(start + jj, rank));
                    }
                }
                uint64_t expertRunning[A2AV_GMM_MAX_RANK_NUM] = {0};
                for (uint32_t rank = 1U; rank < rankSize; ++rank) {
                    expertRunning[rank] = expertRunning[rank - 1] + rankTotal[rank - 1] * n;
                }
                for (uint32_t jj = 0U; jj < actualExpertNum; ++jj) {
                    for (uint32_t rank = 0U; rank < rankSize; ++rank) {
                        const uint32_t groupIdx = batchGroupBase + jj * rankSize + rank;
                        cGroupOffsetTable[groupIdx] = expertRunning[rank];
                        expertRunning[rank] += static_cast<uint64_t>(GetSendRows(start + jj, rank)) * n;
                    }
                }
            }

            // 共享专家单组 OFFSET 格式 groupList：[0] = BS
            *reinterpret_cast<__gm__ int64_t *>(mmGroupListGM_) = static_cast<int64_t>(tilingData_->taskTilingInfo.BS);
        }
        AscendC::SyncAll();
    }

    __aicore__ inline uint32_t GetBatchExpertNum(uint32_t batch) const
    {
        const uint32_t start = batch * BatchExpertNum();
        return (start + BatchExpertNum() > ExpertNum()) ? ExpertNum() - start : BatchExpertNum();
    }

    __aicore__ inline uint64_t GetBatchTokenCount(uint32_t batch) const
    {
        uint64_t tokens = 0;
        const uint32_t start = batch * BatchExpertNum();
        const uint32_t expertNum = GetBatchExpertNum(batch);
        for (uint32_t jj = 0U; jj < expertNum; ++jj) {
            for (uint32_t rank = 0U; rank < RankSize(); ++rank) {
                tokens += GetSendRows(start + jj, rank);
            }
        }
        return tokens;
    }

    __aicore__ inline void RunBatchMatmul(uint32_t batch, uint64_t tokenPrefix, uint64_t batchTokens)
    {
        const uint32_t expertNum = GetBatchExpertNum(batch);
        const uint32_t rankSize = RankSize();
        const uint32_t groupNum = expertNum * rankSize;
        const uint64_t k = tilingData_->taskTilingInfo.H1;
        const uint64_t n = tilingData_->taskTilingInfo.N1;
        const uint64_t dataAxis = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value ? (k + 1U) / 2U : k;
        typename GmmKernel::GMMTiling gmmTiling{};
        FillGmmTiling(gmmTiling, tilingData_->gmmBaseTiling, groupNum, batchTokens, n, k, GROUP_LIST_TYPE_SPARSE);

        typename GmmKernel::Params params{};
        params.problemShape =
            ProblemShape{static_cast<int64_t>(batchTokens), static_cast<int64_t>(n), static_cast<int64_t>(k), 0};
        params.mmadParams.aGmAddr = xGM_ + tokenPrefix * dataAxis;
        params.mmadParams.scaleAGmAddr = xScaleGM_ + tokenPrefix * ScaleKLen();
        params.mmadParams.bGmAddr = weightGM_;
        params.mmadParams.scaleBGmAddr = weightScaleGM_;
        params.mmadParams.cGmAddr = gmmOutputGM_ + tokenPrefix * n * sizeof(CType);
        params.mmadParams.biasGmAddr = nullptr;
        params.groupListGmAddr =
            sparseGroupListGM_ + static_cast<uint64_t>(batch) * BatchExpertNum() * rankSize * 2U * sizeof(int64_t);
        params.gmmParams = gmmTiling;
        params.groupOffsetTables = {};
        params.groupOffsetTables.cGroupAddrTable = reinterpret_cast<const __gm__ uint64_t *>(cGroupOffsetTableGM_) +
                                                   static_cast<uint64_t>(batch) * BatchExpertNum() * rankSize;

        {
            GmmKernel kernel;
            kernel.Init(params);
            kernel.Run();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RunLocalMatmul()
    {
        const uint64_t m = tilingData_->taskTilingInfo.BS;
        const uint64_t n = tilingData_->taskTilingInfo.N2;
        const uint64_t k = tilingData_->taskTilingInfo.H2;
        typename MmKernel::GMMTiling mmTiling{};
        FillGmmTiling(mmTiling, tilingData_->sharedGmmTiling, 1U, m, n, k, GROUP_LIST_TYPE_OFFSET);

        typename MmKernel::Params params{};
        params.problemShape =
            ProblemShape{static_cast<int64_t>(m), static_cast<int64_t>(n), static_cast<int64_t>(k), 0};
        params.mmadParams.aGmAddr = mmXGM_;
        params.mmadParams.scaleAGmAddr = mmXScaleGM_;
        params.mmadParams.bGmAddr = mmWeightGM_;
        params.mmadParams.scaleBGmAddr = mmWeightScaleGM_;
        params.mmadParams.cGmAddr = mmYGM_;
        params.mmadParams.biasGmAddr = nullptr;
        params.groupListGmAddr = mmGroupListGM_;
        params.gmmParams = mmTiling;
        params.groupOffsetTables = {};

        {
            MmKernel kernel;
            kernel.Init(params);
            kernel.Run();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SubmitBatchComm(uint32_t batch, uint64_t tokenPrefix)
    {
        // 通信参数为提交期临时数据（局部变量，避免成员占用 kernel 全生命周期栈空间）
        uint64_t sendCnt[A2AV_GMM_MAX_RANK_NUM] = {0};
        uint64_t sendOffset[A2AV_GMM_MAX_RANK_NUM] = {0};
        uint64_t recvCnt[A2AV_GMM_MAX_RANK_NUM] = {0};
        uint64_t recvOffset[A2AV_GMM_MAX_RANK_NUM] = {0};
        const uint32_t rankSize = RankSize();
        const uint32_t expertNum = ExpertNum();
        const uint32_t n = static_cast<uint32_t>(tilingData_->taskTilingInfo.N1);
        const uint32_t batchExpertNum = GetBatchExpertNum(batch);
        const uint32_t start = batch * BatchExpertNum();

        // send：gmmOutput 批间串接、批内 rank-major（sendOffset[0] = 批前缀，批内 peer 段前缀累加）
        // recv：y 全局 [peer][expert] rank-major（peer 段基 + peer 段内本批专家前缀）
        uint64_t sendRunning = tokenPrefix * n;
        uint64_t peerBase = 0;
        for (uint32_t peer = 0U; peer < rankSize; ++peer) {
            uint64_t sendBatchRows = 0;
            uint64_t recvBatchRows = 0;
            for (uint32_t jj = 0U; jj < batchExpertNum; ++jj) {
                sendBatchRows += GetSendRows(start + jj, peer);
                recvBatchRows += GetRecvRows(start + jj, peer);
            }
            uint64_t expertPrefix = 0;
            for (uint32_t j = 0U; j < start; ++j) {
                expertPrefix += GetRecvRows(j, peer);
            }
            sendCnt[peer] = sendBatchRows * n;
            sendOffset[peer] = sendRunning;
            sendRunning += sendBatchRows * n;
            recvCnt[peer] = recvBatchRows * n;
            recvOffset[peer] = (peerBase + expertPrefix) * n;

            uint64_t peerRows = 0;
            for (uint32_t j = 0U; j < expertNum; ++j) {
                peerRows += GetRecvRows(j, peer);
            }
            peerBase += peerRows * n;
        }

        a2avHandles_[submitted_] =
            hccl_.template AlltoAllV<true>(gmmOutputGM_, sendCnt, sendOffset, Y_HCCL_DATA_TYPE, yGM_, recvCnt,
                                           recvOffset, Y_HCCL_DATA_TYPE, static_cast<uint8_t>(1));
        submitted_++;
    }

    template <typename GmmTiling, typename GmmTilingData>
    __aicore__ inline void FillGmmTiling(GmmTiling &out, const GmmTilingData &in, uint32_t groupNum, int64_t m,
                                         int64_t n, int64_t k, uint8_t groupListType)
    {
        const auto &mmTiling = in.mmTilingData;
        out.groupNum = groupNum;
        out.m = m;
        out.n = n;
        out.k = k;
        out.baseM = static_cast<uint32_t>(mmTiling.baseM);
        out.baseN = static_cast<uint32_t>(mmTiling.baseN);
        out.baseK = static_cast<uint32_t>(mmTiling.baseK);
        out.kAL1 = static_cast<uint32_t>(mmTiling.stepKa * mmTiling.baseK);
        out.kBL1 = static_cast<uint32_t>(mmTiling.stepKb * mmTiling.baseK);
        out.scaleKAL1 = out.kAL1;
        out.isBias = 0U;
        out.dbL0C = static_cast<uint8_t>(mmTiling.dbL0C);
        out.l1BufferStage = 2U;
        out.groupListType = groupListType;
    }

    const TilingDataType *tilingData_{nullptr};
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle a2avHandles_[GMM_A2AV_MAX_BATCH_NUM]{};
    uint32_t submitted_{0};

    GM_ADDR xGM_{nullptr};
    GM_ADDR xScaleGM_{nullptr};
    GM_ADDR weightGM_{nullptr};
    GM_ADDR weightScaleGM_{nullptr};
    GM_ADDR yGM_{nullptr};
    GM_ADDR mmXGM_{nullptr};
    GM_ADDR mmXScaleGM_{nullptr};
    GM_ADDR mmWeightGM_{nullptr};
    GM_ADDR mmWeightScaleGM_{nullptr};
    GM_ADDR mmYGM_{nullptr};
    GM_ADDR gmmOutputGM_{nullptr};
    GM_ADDR sparseGroupListGM_{nullptr};
    GM_ADDR cGroupOffsetTableGM_{nullptr};
    GM_ADDR mmGroupListGM_{nullptr};
};

} // namespace Apace

#endif // APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__
