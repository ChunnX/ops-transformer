/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_quant_grouped_matmul_hcomm_impl.h
 * \brief AllToAllv + MX 量化 GroupedMatMul 融合实现（hcomm / HCCL 集合通信版本）
 *
 * 面向 AlltoAllvQuantGroupedMatMul 算子的 apace 切换路径（参照 AlltoAllMatmul 的
 * “MX 量化切 apace”模式）：
 *   - 通信：AIV0 按 expert 通过 Hccl<>::AlltoAllV 下发变长通信，数据与 scale 各一个句柄，
 *     通信输出按 expert-major / source-rank-minor 布局直接落入接收区；
 *   - 计算：等待当前 expert 通信完成、提交下一 expert 通信后，以“段式分组”消费当前数据，
 *     形成 expert 级通信/计算双缓冲；接收布局无需额外 permute——
 *     每个 (expert, srcRank) 段为独立计算分组（段内 A/scaleA/C 连续），
 *     免去通信后的独立 permute；
 *   - 计算 kernel 复用 apace 分组量化 Matmul（Blaze BlockMmad<GroupedMatmulWithScaleMx>）。
 *
 * 本实现运行在 MIX kernel：AIV 负责 HCCL AlltoAllV，AIC 负责 Blaze Matmul，
 * 两侧通过仓内 A2AV scheduler 同款 cross-core flag 协议同步通信完成事件。
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "../../matmul/grouped_matmul/qgmm_mx_kernel.h"
#include "include/tensor_api/tensor.h"
#include "allto_allv_quant_grouped_matmul_tiling_data.h"
#include "../../../../../../allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/common/a2av_common_tiling.h"

namespace Apace {

template <typename T>
constexpr uint64_t GetA2avHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return AscendC::HCCL_DATA_TYPE_INT8; // fp4 打包存储，按 int8 字节流通信
    } else {
        return AscendC::HCCL_DATA_TYPE_FP8E4M3;
    }
}

template <typename AType, typename BType, typename CType, bool TransB, bool TransMmB,
          AscendC::HcclServerType ServerType>
class AllToAllvQuantGmmHcommImpl {
public:
    explicit __aicore__ inline AllToAllvQuantGmmHcommImpl(hcommA2avQuantGmmTilingData *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline ~AllToAllvQuantGmmHcommImpl() = default;

    // Layout 定义：A/C 行主序；B 随 TransB；scale 布局与算子规格一致
    static constexpr int32_t SCALE_C0 = 2;
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = typename AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn,
                                                         AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using LayoutScaleA = AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleANDLayoutPtn,
                                                        AscendC::Std::Int<SCALE_C0>>;
    using LayoutScaleB = typename AscendC::Std::conditional_t<
        TransB, AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBDNLayoutPtn, AscendC::Std::Int<SCALE_C0>>,
        AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>>;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerGmmSwatWithTailSplit;
    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithScaleMx<0>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using GmmKernel = Apace::Kernel::QuantGroupedMatmulKernel<ProblemShape, BlockMmad, BlockScheduler>;
    using MmLayoutB = typename AscendC::Std::conditional_t<TransMmB, AscendC::Te::DNExtLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn>;
    using MmBlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, MmLayoutB, CType,
                                                      LayoutC, BiasType, LayoutBias>;
    using MmKernel = Apace::Kernel::QuantGroupedMatmulKernel<ProblemShape, MmBlockMmad, BlockScheduler>;

    static constexpr uint64_t xHcclDataType_ = GetA2avHcclDataType<AType>();
    static constexpr uint64_t mxScaleHcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E8M0;

    /*!
     * \brief 初始化并下发通信任务（AIV only）
     *
     * \param[in] tilingGM tiling buffer 首地址：GET_TILING_DATA_WITH_STRUCT 得到的是本地拷贝，
     *            段表（segList/segAOff/segCOff）以 __gm__ 指针按偏移直接引用 tiling buffer
     */
    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightGM,
                                GM_ADDR gmmWeightScaleGM, GM_ADDR mmxGM, GM_ADDR mmxScaleGM,
                                GM_ADDR mmWeightGM, GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                GM_ADDR permuteOutGM, GM_ADDR userWorkspaceGM, GM_ADDR tilingGM)
    {
        xGM_ = gmmxGM;
        xScaleGM_ = gmmxScaleGM;
        weightGM_ = gmmWeightGM;
        weightScaleGM_ = gmmWeightScaleGM;
        yGM_ = gmmyGM;
        mmxGM_ = mmxGM;
        mmxScaleGM_ = mmxScaleGM;
        mmWeightGM_ = mmWeightGM;
        mmWeightScaleGM_ = mmWeightScaleGM;
        mmyGM_ = mmyGM;
        // Each expert is communicated independently, so receive directly into the
        // expert-major output when requested. Otherwise use the workspace buffer.
        dataRecvGM_ = tilingData_->hasPermuteOut != 0U ? permuteOutGM : userWorkspaceGM + tilingData_->dataRecvOffset;
        scaleRecvGM_ = userWorkspaceGM + tilingData_->scaleRecvOffset;
        segListGm_ = reinterpret_cast<__gm__ const uint32_t *>(
            tilingGM + offsetof(hcommA2avQuantGmmTilingData, segList));
        segCOffGm_ =
            reinterpret_cast<__gm__ const uint32_t *>(tilingGM + offsetof(hcommA2avQuantGmmTilingData, segCOff));
        mmGroupListGm_ = reinterpret_cast<__gm__ const uint32_t *>(
            tilingGM + offsetof(hcommA2avQuantGmmTilingData, mmGroupList));

        if ASCEND_IS_AIC {
            return;
        }
        // AlltoAllV is an AIV-side HCCL primitive in the existing A2AV
        // scheduler. Every AIV initializes/finalizes its HCCL object; AIV0 is
        // the sole owner of the collective task sequence.
        hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
        hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(hcommA2avQuantGmmTilingData, mc2CcTiling)));
        if (IsCommunicationCore()) {
            if (tilingData_->expertNum > 0U) {
                IssueExpertCommunication(0U, 0U);
            }
        }
    }

    /*!
     * \brief 等待通信完成并执行段式分组量化 Matmul（AIC only）
     */
    __aicore__ inline void Run()
    {
        if ASCEND_IS_AIV {
            RunCommunication();
            return;
        }
        for (uint32_t expertIdx = 0; expertIdx < tilingData_->expertNum; ++expertIdx) {
            WaitCommunicationDone();
            RunExpertMatmul(expertIdx);
            // Keep every AIC on the same expert before reusing the cross-core
            // communication-ready event for the next expert.
            AscendC::CrossCoreSetFlag<MC2KernelTemplate::SYNC_MODE_AIC_BARRIER, PIPE_FIX>(
                MC2KernelTemplate::SYNC_FLAG_ID_AIC_BARRIER);
            AscendC::CrossCoreWaitFlag<MC2KernelTemplate::SYNC_MODE_AIC_BARRIER, PIPE_S>(
                MC2KernelTemplate::SYNC_FLAG_ID_AIC_BARRIER);
        }
        if (tilingData_->localMatmul != 0U) {
            RunLocalMatmul();
        }
    }

private:
    __aicore__ inline bool IsCommunicationCore() const
    {
        return AscendC::GetBlockIdx() == 0U;
    }

    __aicore__ inline void WaitCommunicationDone()
    {
        AscendC::CrossCoreWaitFlag<MC2KernelTemplate::SYNC_MODE_AIV_TO_AIC, PIPE_MTE2>(
            MC2KernelTemplate::SYNC_FLAG_ID_COMM_DONE);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        AscendC::CrossCoreWaitFlag<MC2KernelTemplate::SYNC_MODE_AIV_TO_AIC, PIPE_MTE2>(
            MC2KernelTemplate::SYNC_FLAG_ID_COMM_DONE + MC2KernelTemplate::SYNC_FLAG_AIV1_OFFSET);
#endif
    }

    __aicore__ inline void RunCommunication()
    {
        for (uint32_t expertIdx = 0; expertIdx < tilingData_->expertNum; ++expertIdx) {
            const uint32_t slot = expertIdx & 1U;
            if (IsCommunicationCore()) {
                hccl_.Wait(scaleHandle_[slot]);
                hccl_.Wait(dataHandle_[slot]);
                const uint32_t nextExpert = expertIdx + 1U;
                if (nextExpert < tilingData_->expertNum) {
                    IssueExpertCommunication(nextExpert, nextExpert & 1U);
                }
            }
            // Match A2avGmmScheduler: first make AIV0's wait visible to all
            // AIVs, then both vector sub-blocks notify their paired AIC.
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<MC2KernelTemplate::SYNC_MODE_AIV_TO_AIC, PIPE_V>(
                MC2KernelTemplate::SYNC_FLAG_ID_COMM_DONE +
                MC2KernelTemplate::SYNC_FLAG_AIV1_OFFSET * AscendC::GetSubBlockIdx());
        }
        AscendC::SyncAll<false>();
        hccl_.Finalize();
    }

    __aicore__ inline uint32_t GetSegmentRows(uint32_t segIdx) const
    {
        const uint32_t previous = (segIdx == 0U) ? 0U : tilingData_->segList[segIdx - 1U];
        return tilingData_->segList[segIdx] - previous;
    }

    __aicore__ inline void IssueExpertCommunication(uint32_t expertIdx, uint32_t slot)
    {
        const uint64_t dataAxis = tilingData_->isMxFp4 != 0U ? (static_cast<uint64_t>(tilingData_->k) + 1U) / 2U
                                                             : static_cast<uint64_t>(tilingData_->k);
        const uint64_t scaleAxis = tilingData_->scaleKLen;
        const uint32_t rankSize = tilingData_->rankSize;
        for (uint32_t peer = 0; peer < rankSize; ++peer) {
            const uint32_t segIdx = expertIdx * rankSize + peer;
            uint64_t sendRowOffset = 0U;
            for (uint32_t previousExpert = 0; previousExpert < expertIdx; ++previousExpert) {
                sendRowOffset += tilingData_->segSendRows[previousExpert * rankSize + peer];
            }
            const uint64_t sendRows = tilingData_->segSendRows[segIdx];
            const uint64_t recvRows = GetSegmentRows(segIdx);
            // Receive each expert directly in expert-major order. This is also the
            // externally visible permute_out layout and removes the former AIV permute.
            const uint64_t recvRowOffset = tilingData_->segCOff[segIdx];
            dataSendCnt_[slot][peer] = sendRows * dataAxis;
            dataSendOff_[slot][peer] = tilingData_->dataSendOff[peer] + sendRowOffset * dataAxis;
            dataRecvCnt_[slot][peer] = recvRows * dataAxis;
            dataRecvOff_[slot][peer] = recvRowOffset * dataAxis;
            scaleSendCnt_[slot][peer] = sendRows * scaleAxis;
            scaleSendOff_[slot][peer] = tilingData_->scaleSendOff[peer] + sendRowOffset * scaleAxis;
            scaleRecvCnt_[slot][peer] = recvRows * scaleAxis;
            scaleRecvOff_[slot][peer] = recvRowOffset * scaleAxis;
        }
        scaleHandle_[slot] = hccl_.template AlltoAllV<true>(
            xScaleGM_, scaleSendCnt_[slot], scaleSendOff_[slot],
            static_cast<AscendC::HcclDataType>(mxScaleHcclDataType_), scaleRecvGM_, scaleRecvCnt_[slot],
            scaleRecvOff_[slot], static_cast<AscendC::HcclDataType>(mxScaleHcclDataType_));
        dataHandle_[slot] = hccl_.template AlltoAllV<true>(
            xGM_, dataSendCnt_[slot], dataSendOff_[slot], static_cast<AscendC::HcclDataType>(xHcclDataType_),
            dataRecvGM_, dataRecvCnt_[slot], dataRecvOff_[slot], static_cast<AscendC::HcclDataType>(xHcclDataType_));
    }

    __aicore__ inline void RunExpertMatmul(uint32_t expertIdx)
    {
        typename GmmKernel::GMMTiling gmm;
        gmm.groupNum = tilingData_->rankSize;
        gmm.n = tilingData_->n;
        gmm.k = tilingData_->k;
        gmm.baseM = tilingData_->baseM;
        gmm.baseN = tilingData_->baseN;
        gmm.baseK = tilingData_->baseK;
        gmm.kAL1 = tilingData_->kAL1;
        gmm.kBL1 = tilingData_->kBL1;
        gmm.scaleKAL1 = tilingData_->scaleKAL1;
        gmm.isBias = 0;
        gmm.dbL0C = 2;
        gmm.l1BufferStage = 2;

        typename GmmKernel::Params params;
        params.aBasePtr = reinterpret_cast<__gm__ AType *>(dataRecvGM_);
        params.scaleABasePtr = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(scaleRecvGM_);
        params.bBasePtr = reinterpret_cast<__gm__ BType *>(weightGM_);
        params.scaleBBasePtr = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(weightScaleGM_);
        params.cBasePtr = reinterpret_cast<__gm__ CType *>(yGM_);
        const uint32_t firstSegment = expertIdx * tilingData_->rankSize;
        params.groupList = segListGm_ + firstSegment;
        // Communication receive data is already expert-major.
        params.segAOff = segCOffGm_ + firstSegment;
        params.segCOff = segCOffGm_ + firstSegment;
        params.segW = tilingData_->rankSize;
        params.expertBase = expertIdx;
        params.groupListBase = (firstSegment == 0U) ? 0U : tilingData_->segList[firstSegment - 1U];
        params.gmm = gmm;
        gmmKernel_(params);
    }

    __aicore__ inline void RunLocalMatmul()
    {
        typename MmKernel::GMMTiling mm;
        mm.groupNum = 1U;
        mm.n = tilingData_->mmN;
        mm.k = tilingData_->mmK;
        mm.baseM = tilingData_->mmBaseM;
        mm.baseN = tilingData_->mmBaseN;
        mm.baseK = tilingData_->mmBaseK;
        mm.kAL1 = tilingData_->mmKAL1;
        mm.kBL1 = tilingData_->mmKBL1;
        mm.scaleKAL1 = tilingData_->mmScaleKAL1;
        mm.isBias = 0;
        mm.dbL0C = 2;
        mm.l1BufferStage = 2;

        typename MmKernel::Params params;
        params.aBasePtr = reinterpret_cast<__gm__ AType *>(mmxGM_);
        params.scaleABasePtr = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(mmxScaleGM_);
        params.bBasePtr = reinterpret_cast<__gm__ BType *>(mmWeightGM_);
        params.scaleBBasePtr = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(mmWeightScaleGM_);
        params.cBasePtr = reinterpret_cast<__gm__ CType *>(mmyGM_);
        params.groupList = mmGroupListGm_;
        params.segAOff = nullptr;
        params.segCOff = nullptr;
        params.segW = 0U;
        params.expertBase = 0U;
        params.groupListBase = 0U;
        params.gmm = mm;
        mmKernel_(params);
    }
    hcommA2avQuantGmmTilingData *tilingData_{nullptr};
    AscendC::Hccl<ServerType> hccl_;
    GmmKernel gmmKernel_;
    MmKernel mmKernel_;
    AscendC::HcclHandle scaleHandle_[2] = {0, 0};
    AscendC::HcclHandle dataHandle_[2] = {0, 0};

    GM_ADDR xGM_{nullptr};
    GM_ADDR xScaleGM_{nullptr};
    GM_ADDR weightGM_{nullptr};
    GM_ADDR weightScaleGM_{nullptr};
    GM_ADDR yGM_{nullptr};
    GM_ADDR mmxGM_{nullptr};
    GM_ADDR mmxScaleGM_{nullptr};
    GM_ADDR mmWeightGM_{nullptr};
    GM_ADDR mmWeightScaleGM_{nullptr};
    GM_ADDR mmyGM_{nullptr};
    GM_ADDR dataRecvGM_{nullptr};
    GM_ADDR scaleRecvGM_{nullptr};
    // 段表 __gm__ 指针（引用 tiling buffer 原始地址，避免本地拷贝地址空间问题）
    __gm__ const uint32_t *segListGm_{nullptr};
    __gm__ const uint32_t *segCOffGm_{nullptr};
    __gm__ const uint32_t *mmGroupListGm_{nullptr};

    // AlltoAllV 本地参数表（元素计）
    uint64_t dataSendCnt_[2][A2AV_GMM_MAX_RANK_NUM] = {{0}};
    uint64_t dataSendOff_[2][A2AV_GMM_MAX_RANK_NUM] = {{0}};
    uint64_t dataRecvCnt_[2][A2AV_GMM_MAX_RANK_NUM] = {{0}};
    uint64_t dataRecvOff_[2][A2AV_GMM_MAX_RANK_NUM] = {{0}};
    uint64_t scaleSendCnt_[2][A2AV_GMM_MAX_RANK_NUM] = {{0}};
    uint64_t scaleSendOff_[2][A2AV_GMM_MAX_RANK_NUM] = {{0}};
    uint64_t scaleRecvCnt_[2][A2AV_GMM_MAX_RANK_NUM] = {{0}};
    uint64_t scaleRecvOff_[2][A2AV_GMM_MAX_RANK_NUM] = {{0}};
};

} // namespace Apace
