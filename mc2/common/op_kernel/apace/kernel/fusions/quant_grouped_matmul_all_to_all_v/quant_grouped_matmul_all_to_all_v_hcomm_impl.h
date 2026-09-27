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
 * 结构参考镜像算子 allto_allv_quant_grouped_mat_mul apace hcomm 实现（直接复用算子
 * legacy tiling 数据），并按本算子两侧真机证据采用混合 kernel 生命周期：
 *   - 路由专家：持久成员 GmmKernel（ctor 一次预置核内事件、每批仅 re-Init）——
 *     409d5e04a 真机实证形态；BlockMmad ctor/dtor 为 SetFlag/WaitFlag 事件协议，
 *     每批栈上构造会重复 SetFlag（上批排空后事件已 SET），存在事件失步挂死风险
 *   - 共享专家：栈上局部 MmKernel（镜像 RunLocalMatmul 模式）——持久成员形态即
 *     历史挂死形态，栈上构造是镜像对其的解法
 *
 * Blaze 入参保持 409d5e04a（路由专家真机验证形态）：LENGTH groupList + B/scaleB
 * 显式偏移表（全核冗余写、每核写读一致，无跨核可见性依赖）、kAL1/kBL1/scaleKAL1
 * 置 0 走 baseK 兜底（单 L0 块 L1 窗口——多 L0 块窗口路径有共享专家挂死真机证据）。
 *
 * 流水参考 gmm_a2av_scheduler.h 的批循环调度：批内 GMM 计算 → 核间同步 → 提交
 * 本批非阻塞 AllToAllV（通信与后续批次计算重叠）；共享专家在批循环后执行（与末批
 * 通信重叠）；尾部统一 Wait 全部 handle → SyncAll → Finalize。
 *
 * workspace 布局（后通信模式，apace 通路，均位于 wsGmmComputeWS 区域内复用）：
 *   [wsGmmOutputSize, +e*R*8)                → groupList（LENGTH 计数，每批覆写）
 *   [+, +e*R*8)                              → C 输出偏移表（rank-major，每批覆写）
 *   [+, +e*R*8)                              → B 权重偏移表（批内 expert 粒度）
 *   [+, +e*R*8)                              → ScaleB 偏移表（批内 expert 粒度）
 *   [+, 8)                                   → 共享专家 mmGroupList（[0] = BS）
 */

#ifndef APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__
#define APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__

#include "lib/hccl/hccl.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "apace/kernel/matmul/grouped_matmul/grouped_matmul_mx_kernel.h"
#include "../../../../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"
#include "../../../../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/a2av_comm_params.h"

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
        // 表区从 GMM 输出缓冲之后开始（即复用原 gmmComputeWS 区域；全核冗余写，
        // 每核写读一致，无跨核可见性依赖，无需核间同步）
        GM_ADDR tableBase = workspace + tilingData_->workspaceInfo.wsGmmOutputSize;
        groupListGm_ = tableBase;
        cGroupOffsetTableGm_ = groupListGm_ + RankSize() * ExpertNum() * sizeof(int64_t);
        bGroupOffsetTableGm_ = cGroupOffsetTableGm_ + RankSize() * ExpertNum() * sizeof(uint64_t);
        scaleBGroupOffsetTableGm_ = bGroupOffsetTableGm_ + RankSize() * ExpertNum() * sizeof(uint64_t);
        mmGroupListGm_ = scaleBGroupOffsetTableGm_ + RankSize() * ExpertNum() * sizeof(uint64_t);

        // 共享专家单组 OFFSET 格式 groupList：[0] = BS（全核冗余写）
        *reinterpret_cast<__gm__ int64_t *>(mmGroupListGm_) = static_cast<int64_t>(tilingData_->taskTilingInfo.BS);

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
                RunBatchMatmul(batch, tokenPrefix);
            }
            // AIC 核间同步：本批 sendBuf 多核写全部落 GM 后才可提交通信；
            // 空批也照常提交，保证各 rank 集合通信次数对齐
            AscendC::SyncAll();
            SubmitBatchComm(batch);
            // 排空本核数据搬移管线（含 block0 侧参数区拷贝）后再复用参数数组
            AscendC::PipeBarrier<PIPE_ALL>();
            tokenPrefix += batchTokens;
        }

        // 共享专家位于批循环后、Wait 之前（镜像 RunLocalMatmul 位置语义：与末批
        // 在途通信重叠执行；Blaze 于 hccl_.Wait 之后执行为未验证序列，规避）；
        // Blaze 计算必须位于 Finalize 之前（既有真机结论）
        if (tilingData_->taskTilingInfo.BS != 0U && tilingData_->taskTilingInfo.H2 != 0U &&
            tilingData_->taskTilingInfo.N2 != 0U) {
            RunLocalMatmul();
        }

        // 尾部统一等待全部批次完成（全核，按提交顺序）；Wait 后排空管线再汇聚、
        // Finalize（对齐镜像 WaitExpert 的 Wait → PipeBarrier 模式）
        for (uint32_t i = 0U; i < submitted_; ++i) {
            hccl_.Wait(a2avHandles_[i]);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SyncAll();
        if (submitted_ > 0U) {
            hccl_.Finalize();
        }
    }

private:
    // rank 维数组容量沿用 legacy MAX_EP_RANK_SIZE(128)：SUPPORT_RANK_SIZE 上限 128，
    // 且 AIC 栈帧 32KB 受限（HcclMsgExt 注释明示），栈数组不得随意扩容
    static constexpr uint32_t A2AV_GMM_MAX_RANK_NUM = MC2KernelTemplate::MAX_EP_RANK_SIZE;
    static constexpr uint32_t GMM_A2AV_MAX_BATCH_NUM = 8U;
    static constexpr uint32_t MX_BASIC_FACTOR = 64U;
    static constexpr uint8_t GROUP_LIST_TYPE_OFFSET = 0U;
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

    __aicore__ inline void RunBatchMatmul(uint32_t batch, uint64_t tokenPrefix)
    {
        const uint32_t rankSize = RankSize();
        const uint32_t expertNum = GetBatchExpertNum(batch);
        const uint32_t groupNum = expertNum * rankSize;
        const uint32_t start = batch * BatchExpertNum();
        const uint64_t h1 = tilingData_->taskTilingInfo.H1;
        const uint64_t n1 = tilingData_->taskTilingInfo.N1;
        const uint64_t scaleK = ScaleKLen();
        const bool isFp4 = AscendC::IsSameType<AType, fp4x2_e2m1_t>::value;
        const uint64_t packedShift = isFp4 ? 1U : 0U;
        const uint64_t dataAxis = isFp4 ? (h1 + 1U) / 2U : h1;

        BuildBatchTables(start, expertNum, rankSize, n1, scaleK, packedShift);

        const int64_t maxM = static_cast<int64_t>(GetMaxExpertTokens());
        typename GmmKernel::GMMTiling gmmTiling{};
        FillGmmTiling(gmmTiling, tilingData_->gmmBaseTiling, groupNum, maxM, static_cast<int64_t>(n1),
                      static_cast<int64_t>(h1));

        typename GmmKernel::Params params{};
        params.problemShape = ProblemShape{maxM, static_cast<int64_t>(n1), static_cast<int64_t>(h1), 0};
        // A/scaleA/C 基址推进至批前全局 token 前缀；B/scaleB 基址推进至本批首个
        // expert（批内经偏移表按 expert 定位）
        params.mmadParams.aGmAddr = xGM_ + tokenPrefix * dataAxis;
        params.mmadParams.scaleAGmAddr = xScaleGM_ + tokenPrefix * scaleK;
        params.mmadParams.bGmAddr = weightGM_ + ((start * h1 * n1) >> packedShift);
        params.mmadParams.scaleBGmAddr = weightScaleGM_ + start * n1 * scaleK;
        params.mmadParams.cGmAddr = gmmOutputGM_ + tokenPrefix * n1 * sizeof(CType);
        params.mmadParams.biasGmAddr = nullptr;
        params.groupListGmAddr = groupListGm_;
        params.gmmParams = gmmTiling;
        params.groupOffsetTables = {};
        params.groupOffsetTables.cGroupAddrTable = reinterpret_cast<const __gm__ uint64_t *>(cGroupOffsetTableGm_);
        params.groupOffsetTables.bGroupAddrTable = reinterpret_cast<const __gm__ uint64_t *>(bGroupOffsetTableGm_);
        params.groupOffsetTables.scaleBGroupAddrTable =
            reinterpret_cast<const __gm__ uint64_t *>(scaleBGroupOffsetTableGm_);

        // 持久成员每批 re-Init（409d5e04a 实证形态：ctor 一次预置核内事件，避免逐批
        // SetFlag/WaitFlag 事件协议失步）
        gmmKernel_.Init(params);
        gmmKernel_.Run();
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BuildBatchTables(uint32_t start, uint32_t expertNum, uint32_t rankSize, uint64_t n1,
                                            uint64_t scaleK, uint64_t packedShift)
    {
        // 全核冗余写相同值（每核写读一致，无跨核可见性依赖，与 409d5e04a 形态一致）。
        // group 序 expert-major：g = jj * rankSize + rank（与 gmmX 行序一致）
        __gm__ int64_t *groupList = reinterpret_cast<__gm__ int64_t *>(groupListGm_);
        __gm__ uint64_t *cGroupOffsetTable = reinterpret_cast<__gm__ uint64_t *>(cGroupOffsetTableGm_);
        __gm__ uint64_t *bGroupOffsetTable = reinterpret_cast<__gm__ uint64_t *>(bGroupOffsetTableGm_);
        __gm__ uint64_t *scaleBGroupOffsetTable = reinterpret_cast<__gm__ uint64_t *>(scaleBGroupOffsetTableGm_);

        // groupList（LENGTH 计数）
        for (uint32_t jj = 0U; jj < expertNum; ++jj) {
            for (uint32_t rank = 0U; rank < rankSize; ++rank) {
                groupList[jj * rankSize + rank] = static_cast<int64_t>(GetSendRows(start + jj, rank));
            }
        }

        // C 偏移表：批内 rank-major 物理布局（元素偏移，相对本批 C 基址）
        uint64_t rankTotal[A2AV_GMM_MAX_RANK_NUM] = {0};
        for (uint32_t jj = 0U; jj < expertNum; ++jj) {
            for (uint32_t rank = 0U; rank < rankSize; ++rank) {
                rankTotal[rank] += GetSendRows(start + jj, rank);
            }
        }
        uint64_t expertRunning[A2AV_GMM_MAX_RANK_NUM] = {0};
        for (uint32_t rank = 1U; rank < rankSize; ++rank) {
            expertRunning[rank] = expertRunning[rank - 1] + rankTotal[rank - 1] * n1;
        }
        for (uint32_t jj = 0U; jj < expertNum; ++jj) {
            for (uint32_t rank = 0U; rank < rankSize; ++rank) {
                const uint32_t groupIdx = jj * rankSize + rank;
                cGroupOffsetTable[groupIdx] = expertRunning[rank];
                expertRunning[rank] += static_cast<uint64_t>(GetSendRows(start + jj, rank)) * n1;
            }
        }

        // B/scaleB 偏移表：批内 expert 粒度（相对本批 B/scaleB 基址，元素偏移）
        const uint64_t perExpertW = (tilingData_->taskTilingInfo.H1 * n1) >> packedShift;
        for (uint32_t jj = 0U; jj < expertNum; ++jj) {
            for (uint32_t rank = 0U; rank < rankSize; ++rank) {
                const uint32_t groupIdx = jj * rankSize + rank;
                bGroupOffsetTable[groupIdx] = static_cast<uint64_t>(jj) * perExpertW;
                scaleBGroupOffsetTable[groupIdx] = static_cast<uint64_t>(jj) * n1 * scaleK;
            }
        }
    }

    __aicore__ inline uint64_t GetMaxExpertTokens() const
    {
        uint64_t mMax = 0;
        for (uint32_t expert = 0U; expert < ExpertNum(); ++expert) {
            uint64_t tokens = 0;
            for (uint32_t rank = 0U; rank < RankSize(); ++rank) {
                tokens += GetSendRows(expert, rank);
            }
            mMax = tokens > mMax ? tokens : mMax;
        }
        return mMax;
    }

    __aicore__ inline void RunLocalMatmul()
    {
        const uint64_t m = tilingData_->taskTilingInfo.BS;
        const uint64_t n = tilingData_->taskTilingInfo.N2;
        const uint64_t k = tilingData_->taskTilingInfo.H2;
        typename MmKernel::GMMTiling mmTiling{};
        FillGmmTiling(mmTiling, tilingData_->sharedGmmTiling, 1U, static_cast<int64_t>(m), static_cast<int64_t>(n),
                      static_cast<int64_t>(k));

        typename MmKernel::Params params{};
        params.problemShape =
            ProblemShape{static_cast<int64_t>(m), static_cast<int64_t>(n), static_cast<int64_t>(k), 0};
        params.mmadParams.aGmAddr = mmXGM_;
        params.mmadParams.scaleAGmAddr = mmXScaleGM_;
        params.mmadParams.bGmAddr = mmWeightGM_;
        params.mmadParams.scaleBGmAddr = mmWeightScaleGM_;
        params.mmadParams.cGmAddr = mmYGM_;
        params.mmadParams.biasGmAddr = nullptr;
        params.groupListGmAddr = mmGroupListGm_;
        params.gmmParams = mmTiling;
        params.groupOffsetTables = {};

        // [临时诊断] 共享专家卡点二分（block0 门控共 2 条，ringbuffer 可靠预算内；
        // 上轮 S-param/S-addr 已采集：参数与 64B 对齐全部正常，5 条超出 ringbuffer
        // 容量致 S-stage 静默丢弃）。S-A 覆盖 ctor+Init，S-B 覆盖 Run：
        // 仅 S-A 出现 → 挂在 Run 内部（与 409d5e04a 历史 P3-post-init 后挂一致）；
        // S-A 也缺失 → ctor/Init 新断裂点
        MmKernel kernel;
        kernel.Init(params);
        if (AscendC::GetBlockIdx() == 0U) {
            AscendC::printf("[S-A] init-done\n");
        }
        kernel.Run();
        if (AscendC::GetBlockIdx() == 0U) {
            AscendC::printf("[S-B] run-done\n");
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SubmitBatchComm(uint32_t batch)
    {
        // 通信参数逐字节对齐 409d5e04a / 非 apace 通路：legacy A2avCommParams
        // （128 rank 容量、4KB 栈），公式为 CalcA2avCommAfterParams 同款伸缩和形态
        MC2KernelTemplate::A2avCommParams commParams;
        const uint32_t e = ExpertNum();
        const uint32_t rankDim = RankSize();
        const uint64_t n1 = tilingData_->taskTilingInfo.N1;
        const int32_t *sendCounts = tilingData_->taskTilingInfo.sendCnt;
        const int32_t *recvCounts = tilingData_->taskTilingInfo.recvCnt;
        const uint32_t start = batch * BatchExpertNum();
        const uint32_t expertNum = GetBatchExpertNum(batch);

        // 本批 send/recv 计数（元素单位，axis = N1）
        for (uint32_t i = 0U; i < rankDim; i++) {
            commParams.sendCnt[i] = 0UL;
            commParams.recvCnt[i] = 0UL;
            for (uint32_t expertIdx = start; expertIdx < start + expertNum; expertIdx++) {
                commParams.sendCnt[i] += static_cast<uint64_t>(sendCounts[expertIdx + i * e]) * n1;
                commParams.recvCnt[i] += static_cast<uint64_t>(recvCounts[expertIdx + i * e]) * n1;
            }
        }

        // send 偏移：GMM 输出为批间串接、批内 rank-major 布局
        // （本批起点 = 批前全部 token 累计；批内 rank 段首 = 本批起点 + 前 rank 段累计）
        uint64_t batchStartOffset = 0UL;
        for (uint32_t i = 0U; i < start; i++) {
            for (uint32_t j = 0U; j < rankDim; j++) {
                batchStartOffset += static_cast<uint64_t>(sendCounts[i + j * e]);
            }
        }
        batchStartOffset *= n1;
        commParams.sendOffset[0] = batchStartOffset;
        uint64_t rankRunningOffset = 0UL;
        for (uint32_t i = 1U; i < rankDim; i++) {
            for (uint32_t expertIdx = start; expertIdx < start + expertNum; expertIdx++) {
                rankRunningOffset += static_cast<uint64_t>(sendCounts[expertIdx + (i - 1) * e]) * n1;
            }
            commParams.sendOffset[i] = batchStartOffset + rankRunningOffset;
        }

        // recv 偏移：y 按 recvCounts 源 rank 聚合顺序（rank-major，伸缩和恒等式）
        commParams.recvOffset[0] = 0UL;
        for (uint32_t i = 0U; i < start; i++) {
            commParams.recvOffset[0] += static_cast<uint64_t>(recvCounts[i]) * n1;
        }
        for (uint32_t i = 1U; i < rankDim; i++) {
            commParams.recvOffset[i] = commParams.recvOffset[i - 1];
            for (uint32_t j = 0U; j < e; j++) {
                commParams.recvOffset[i] += static_cast<uint64_t>(recvCounts[start + (i - 1) * e + j]) * n1;
            }
        }

        a2avHandles_[submitted_] = hccl_.template AlltoAllV<true>(
            gmmOutputGM_, commParams.sendCnt, commParams.sendOffset, Y_HCCL_DATA_TYPE, yGM_, commParams.recvCnt,
            commParams.recvOffset, Y_HCCL_DATA_TYPE, static_cast<uint8_t>(1));
        submitted_++;
    }

    template <typename GmmTiling, typename GmmTilingData>
    __aicore__ inline void FillGmmTiling(GmmTiling &out, const GmmTilingData &in, uint32_t groupNum, int64_t m,
                                         int64_t n, int64_t k)
    {
        const auto &mmTiling = in.mmTilingData;
        out.groupNum = groupNum;
        out.m = m;
        out.n = n;
        out.k = k;
        out.baseM = static_cast<uint32_t>(mmTiling.baseM);
        out.baseN = static_cast<uint32_t>(mmTiling.baseN);
        out.baseK = static_cast<uint32_t>(mmTiling.baseK);
        // L1 K 窗口置 0 走 baseK 兜底（单 L0 块窗口）：多 L0 块窗口（stepKa×baseK）
        // 路径有共享专家 BlockMmad K 循环挂死的真机证据，且 409d5e04a 可跑形态即兜底值
        out.kAL1 = 0U;
        out.kBL1 = 0U;
        out.scaleKAL1 = 0U;
        out.isBias = static_cast<uint8_t>(mmTiling.isBias != 0 ? 1 : 0);
        out.dbL0C = static_cast<uint8_t>(mmTiling.dbL0C);
        out.l1BufferStage = Blaze::Gemm::DOUBLE_BUFFER_COUNT;
        // 路由专家沿用 tiling 的 LENGTH 格式；共享专家单组用 OFFSET
        out.groupListType =
            (&in == &tilingData_->sharedGmmTiling) ? GROUP_LIST_TYPE_OFFSET : in.gmmQuantParams.groupListType;
    }

    const TilingDataType *tilingData_{nullptr};
    GmmKernel gmmKernel_;
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
    GM_ADDR groupListGm_{nullptr};
    GM_ADDR cGroupOffsetTableGm_{nullptr};
    GM_ADDR bGroupOffsetTableGm_{nullptr};
    GM_ADDR scaleBGroupOffsetTableGm_{nullptr};
    GM_ADDR mmGroupListGm_{nullptr};
};

} // namespace Apace

#endif // APACE_QUANT_GROUPED_MATMUL_ALL_TO_ALL_V_HCOMM_IMPL_H__
