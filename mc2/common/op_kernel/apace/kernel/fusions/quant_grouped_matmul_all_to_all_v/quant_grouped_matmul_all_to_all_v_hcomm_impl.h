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
 * 结构对齐镜像算子 allto_allv_quant_grouped_mat_mul 的 apace hcomm 实现（PR 11573）：
 *   - 紧凑 tiling 数据（hcommGmmA2avTilingData）由 host 侧 PostTiling 直接覆写 tiling
 *     buffer，kernel 经 GET_TILING_DATA_WITH_STRUCT 解析；
 *   - 每批 Blaze GroupedMatmulMxKernel 以栈上局部对象构造（Init + Run 后即析构，
 *     核内事件与 L1 状态随批干净重建）；
 *   - SPARSE groupList 携带 expertId，B/scaleB 走 kernel 默认公式（weightIdx 索引专家维）。
 *
 * 流水参考 gmm_a2av_scheduler.h 的批循环调度：批内 GMM 计算 → 核间同步 → 提交本批
 * 非阻塞 AllToAllV（通信由引擎异步执行，与后续批次计算重叠）；共享专家在批循环后
 * 执行（与末批通信重叠）；尾部统一 Wait 全部 handle → SyncAll → Finalize。
 */

#ifndef APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__
#define APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__

#include "lib/hccl/hccl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "apace/kernel/matmul/grouped_matmul/grouped_matmul_mx_kernel.h"
#include "quant_grouped_matmul_all_to_all_v_tiling_data.h"

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

template <typename AType, typename BType, typename CType, bool TransB, bool TransMmB,
          AscendC::HcclServerType ServerType>
class QuantGmmA2avHcommImpl {
public:
    explicit __aicore__ inline QuantGmmA2avHcommImpl(hcommGmmA2avTilingData *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR gmmX, GM_ADDR gmmXScale, GM_ADDR gmmWeight, GM_ADDR gmmWeightScale, GM_ADDR mmX,
                                GM_ADDR mmXScale, GM_ADDR mmWeight, GM_ADDR mmWeightScale, GM_ADDR mmY, GM_ADDR y,
                                GM_ADDR workspace, GM_ADDR tilingGM)
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
        // apace 通路 userWorkspace 仅承载 GMM 输出缓冲（sendBuf），C 偏移表紧随其后
        gmmOutputGM_ = workspace;
        cGroupOffsetTableGM_ = workspace + tilingData_->cTableOffset;
        sparseGroupListGM_ = tilingGM + offsetof(hcommGmmA2avTilingData, sparseGroupList);
        mmGroupListGM_ = tilingGM + offsetof(hcommGmmA2avTilingData, mmGroupList);

        hccl_.InitV2(AscendC::GetHcclContext<0>(), &tilingData_->mc2InitTiling);
        hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(hcommGmmA2avTilingData, mc2CcTiling)));
    }

    __aicore__ inline void Run()
    {
        uint64_t tokenPrefix = 0;
        for (uint32_t batch = 0; batch < tilingData_->batchCount; ++batch) {
            const uint64_t batchTokens = GetBatchTokenCount(batch);
            if (batchTokens != 0) {
                RunBatchMatmul(batch, tokenPrefix);
            }
            // AIC 核间同步：本批 sendBuf 多核写全部落 GM 后才可提交通信；
            // 空批也照常提交，保证各 rank 集合通信次数对齐
            AscendC::SyncAll();
            SubmitBatchComm(batch, tokenPrefix);
            // 排空本核数据搬移管线（含 block0 侧参数区拷贝）后再复用参数数组
            AscendC::PipeBarrier<PIPE_ALL>();
            tokenPrefix += batchTokens;
        }

        if (tilingData_->localMatmul != 0U) {
            RunLocalMatmul();
        }

        for (uint32_t i = 0U; i < submitted_; ++i) {
            hccl_.Wait(a2avHandles_[i]);
        }
        AscendC::SyncAll();
        hccl_.Finalize();
    }

private:
    static constexpr uint64_t GROUP_LIST_TYPE_OFFSET = 0U;
    static constexpr uint64_t GROUP_LIST_TYPE_SPARSE = 2U;
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

    __aicore__ inline uint32_t GetBatchExpertNum(uint32_t batch) const
    {
        const uint32_t start = batch * tilingData_->batchExpertNum;
        return (start + tilingData_->batchExpertNum > tilingData_->expertNum) ? tilingData_->expertNum - start :
                                                                                tilingData_->batchExpertNum;
    }

    __aicore__ inline uint64_t GetBatchTokenCount(uint32_t batch) const
    {
        uint64_t tokens = 0;
        const uint32_t groupNum = GetBatchExpertNum(batch) * tilingData_->rankSize;
        const uint32_t firstGroup = batch * tilingData_->batchExpertNum * tilingData_->rankSize;
        for (uint32_t g = 0; g < groupNum; ++g) {
            tokens += static_cast<uint64_t>(tilingData_->sparseGroupList[(firstGroup + g) * 2U + 1U]);
        }
        return tokens;
    }

    __aicore__ inline void BuildCGroupOffsetTable(uint32_t batch, uint32_t expertNum)
    {
        // C 批内 rank-major 物理布局（元素偏移，相对本批 C 基址）：
        //   group(jj, r) 偏移 = rankBase[r] + Σ_{jj'<jj} cnt(jj', r) * n
        // 全核冗余写相同值（与原路由专家实现一致），无需核间同步
        const uint32_t rankSize = tilingData_->rankSize;
        const uint32_t n = tilingData_->n;
        const uint32_t firstGroup = batch * tilingData_->batchExpertNum * rankSize;
        __gm__ uint64_t *cTable = reinterpret_cast<__gm__ uint64_t *>(cGroupOffsetTableGM_);

        uint64_t rankTotal[GMM_A2AV_MAX_RANK_NUM] = {0};
        for (uint32_t jj = 0; jj < expertNum; ++jj) {
            for (uint32_t r = 0; r < rankSize; ++r) {
                rankTotal[r] +=
                    static_cast<uint64_t>(tilingData_->sparseGroupList[(firstGroup + jj * rankSize + r) * 2U + 1U]);
            }
        }
        uint64_t expertRunning[GMM_A2AV_MAX_RANK_NUM] = {0};
        for (uint32_t r = 1; r < rankSize; ++r) {
            expertRunning[r] = expertRunning[r - 1] + rankTotal[r - 1] * n;
        }
        for (uint32_t jj = 0; jj < expertNum; ++jj) {
            for (uint32_t r = 0; r < rankSize; ++r) {
                const uint32_t groupIdx = jj * rankSize + r;
                cTable[groupIdx] = expertRunning[r];
                expertRunning[r] +=
                    static_cast<uint64_t>(tilingData_->sparseGroupList[(firstGroup + groupIdx) * 2U + 1U]) * n;
            }
        }
    }

    __aicore__ inline void RunBatchMatmul(uint32_t batch, uint64_t tokenPrefix)
    {
        const uint32_t expertNum = GetBatchExpertNum(batch);
        const uint32_t rankSize = tilingData_->rankSize;
        const uint32_t groupNum = expertNum * rankSize;
        const uint64_t k = tilingData_->k;
        // FP4 (fp4x2) 1 字节装 2 个元素：A 行字节数 = (k+1)/2；scaleK = CeilDiv(k,32)*2（e8m0 1B）
        const uint64_t dataAxis = tilingData_->isMxFp4 != 0U ? (k + 1U) / 2U : k;
        const uint64_t scaleKLen = ((k + Blaze::Gemm::MXFP_DIVISOR_SIZE - 1) >> Blaze::Gemm::MXFP_DIVISOR_SHIFT)
                                   << Blaze::Gemm::MXFP_MULTI_BASE_SHIFT;

        BuildCGroupOffsetTable(batch, expertNum);

        typename GmmKernel::GMMTiling gmmTiling{};
        FillGmmTiling(gmmTiling, tilingData_->gmmTiling, groupNum, tilingData_->mMax, tilingData_->n, tilingData_->k,
                      GROUP_LIST_TYPE_SPARSE);

        typename GmmKernel::Params params{};
        params.problemShape =
            ProblemShape{static_cast<int64_t>(tilingData_->mMax), static_cast<int64_t>(tilingData_->n),
                         static_cast<int64_t>(tilingData_->k), 0};
        params.mmadParams.aGmAddr = xGM_ + tokenPrefix * dataAxis;
        params.mmadParams.scaleAGmAddr = xScaleGM_ + tokenPrefix * scaleKLen;
        params.mmadParams.bGmAddr = weightGM_;
        params.mmadParams.scaleBGmAddr = weightScaleGM_;
        params.mmadParams.cGmAddr = gmmOutputGM_ + tokenPrefix * tilingData_->n * sizeof(CType);
        params.mmadParams.biasGmAddr = nullptr;
        params.groupListGmAddr = sparseGroupListGM_ + static_cast<uint64_t>(batch) * tilingData_->batchExpertNum *
                                                          rankSize * 2U * sizeof(int64_t);
        params.gmmParams = gmmTiling;
        params.groupOffsetTables = {};
        params.groupOffsetTables.cGroupAddrTable = reinterpret_cast<const __gm__ uint64_t *>(cGroupOffsetTableGM_);

        {
            GmmKernel kernel;
            kernel.Init(params);
            kernel.Run();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RunLocalMatmul()
    {
        typename MmKernel::GMMTiling mmTiling{};
        FillGmmTiling(mmTiling, tilingData_->mmTiling, 1U, tilingData_->mmM, tilingData_->mmN, tilingData_->mmK,
                      GROUP_LIST_TYPE_OFFSET);

        typename MmKernel::Params params{};
        params.problemShape =
            ProblemShape{static_cast<int64_t>(tilingData_->mmM), static_cast<int64_t>(tilingData_->mmN),
                         static_cast<int64_t>(tilingData_->mmK), 0};
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
        uint64_t sendCnt[GMM_A2AV_MAX_RANK_NUM] = {0};
        uint64_t sendOffset[GMM_A2AV_MAX_RANK_NUM] = {0};
        uint64_t recvCnt[GMM_A2AV_MAX_RANK_NUM] = {0};
        uint64_t recvOffset[GMM_A2AV_MAX_RANK_NUM] = {0};
        const uint32_t rankSize = tilingData_->rankSize;
        const uint32_t expertNum = tilingData_->expertNum;
        const uint32_t n = tilingData_->n;
        const uint32_t batchExpertNum = GetBatchExpertNum(batch);
        const uint32_t start = batch * tilingData_->batchExpertNum;

        // send：gmmOutput 批间串接、批内 rank-major（sendOffset[0] = 批前缀，批内 peer 段前缀累加）
        // recv：y 全局 [peer][expert] rank-major（peer 段基 + peer 段内本批专家前缀）
        uint64_t sendRunning = tokenPrefix * n;
        uint64_t peerBase = 0;
        for (uint32_t r = 0; r < rankSize; ++r) {
            uint64_t sendBatchRows = 0;
            uint64_t recvBatchRows = 0;
            for (uint32_t jj = 0; jj < batchExpertNum; ++jj) {
                sendBatchRows +=
                    static_cast<uint64_t>(tilingData_->sparseGroupList[((start + jj) * rankSize + r) * 2U + 1U]);
                recvBatchRows += static_cast<uint64_t>(tilingData_->recvCount[r * expertNum + start + jj]);
            }
            uint64_t expertPrefix = 0;
            for (uint32_t j = 0; j < start; ++j) {
                expertPrefix += static_cast<uint64_t>(tilingData_->recvCount[r * expertNum + j]);
            }
            sendCnt[r] = sendBatchRows * n;
            sendOffset[r] = sendRunning;
            sendRunning += sendBatchRows * n;
            recvCnt[r] = recvBatchRows * n;
            recvOffset[r] = (peerBase + expertPrefix) * n;

            uint64_t peerRows = 0;
            for (uint32_t j = 0; j < expertNum; ++j) {
                peerRows += static_cast<uint64_t>(tilingData_->recvCount[r * expertNum + j]);
            }
            peerBase += peerRows * n;
        }

        a2avHandles_[submitted_] =
            hccl_.template AlltoAllV<true>(gmmOutputGM_, sendCnt, sendOffset, Y_HCCL_DATA_TYPE, yGM_, recvCnt,
                                           recvOffset, Y_HCCL_DATA_TYPE, static_cast<uint8_t>(1));
        submitted_++;
    }

    template <typename GmmTiling>
    __aicore__ inline void FillGmmTiling(GmmTiling &out, const GmmA2avMxTiling &in, uint32_t groupNum, int64_t m,
                                         int64_t n, int64_t k, uint8_t groupListType)
    {
        out.groupNum = groupNum;
        out.m = m;
        out.n = n;
        out.k = k;
        out.baseM = in.baseM;
        out.baseN = in.baseN;
        out.baseK = in.baseK;
        out.kAL1 = in.kAL1;
        out.kBL1 = in.kBL1;
        out.scaleKAL1 = in.scaleKAL1;
        out.isBias = 0U;
        out.dbL0C = in.dbL0C;
        out.l1BufferStage = in.l1BufferStage;
        out.groupListType = groupListType;
    }

    hcommGmmA2avTilingData *tilingData_{nullptr};
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
    GM_ADDR cGroupOffsetTableGM_{nullptr};
    GM_ADDR sparseGroupListGM_{nullptr};
    GM_ADDR mmGroupListGM_{nullptr};
};

} // namespace Apace

#endif // APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__
