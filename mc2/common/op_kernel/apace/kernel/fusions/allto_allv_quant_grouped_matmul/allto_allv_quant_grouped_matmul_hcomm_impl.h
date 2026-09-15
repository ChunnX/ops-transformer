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
 *   - 计算：AIC 等待当前 expert 通信完成，以“段式分组”消费当前数据；
 *     AIV 可继续后续 expert 的通信。接收布局无需额外 permute——
 *     每个 (expert, srcRank) 段为独立计算分组（段内 A/scaleA/C 连续），
 *     免去通信后的独立 permute；
 *   - 计算 kernel 复用 apace 分组量化 Matmul（Blaze BlockMmad<GroupedMatmulWithScaleMx>）。
 *
 * 本实现运行在 MIX kernel：AIV 负责 HCCL AlltoAllV，AIC 负责 Blaze Matmul，
 * 两侧通过仓内 A2AV scheduler 同款 cross-core flag 协议同步通信完成事件。
 */

#pragma once

#include "kernel_operator.h"
#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qgmm_mx.h"
#include "blaze/gemm/block/block_scheduler_gmm_swat_with_tail_split.h"
#include "../../matmul/grouped_matmul/qgmm_mx_kernel.h"
#include "include/tensor_api/tensor.h"
#include "allto_allv_quant_grouped_matmul_tiling_data.h"

namespace Apace {

// Keep the MIX cross-core protocol local to the APACE implementation.  A
// common/APACE header must not include an operator-private header because the
// binary packaging step renames the operator source directory with an _apt
// suffix and does not preserve that source-tree-relative path.
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
static constexpr uint8_t A2AV_GMM_SYNC_MODE_AIV_TO_AIC = 4U;
static constexpr uint16_t A2AV_GMM_SYNC_FLAG_AIV1_OFFSET = 16U;
#else
static constexpr uint8_t A2AV_GMM_SYNC_MODE_AIV_TO_AIC = 2U;
static constexpr uint16_t A2AV_GMM_SYNC_FLAG_AIV1_OFFSET = 0U;
#endif
static constexpr uint8_t A2AV_GMM_SYNC_MODE_AIC_BARRIER = 0U;
static constexpr uint16_t A2AV_GMM_SYNC_FLAG_COMM_DONE = 8U;
static constexpr uint16_t A2AV_GMM_SYNC_FLAG_AIC_BARRIER = 9U;
// Disjoint from ready 8/24, barrier 9 and SyncAll's 11/12/13/14/28/29.
static constexpr uint16_t A2AV_GMM_SYNC_FLAG_COMM_CONSUMED = 10U;
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
    using DispatchPolicy =
        Blaze::Gemm::GroupedMatmulWithScaleMx<0, false, Blaze::Gemm::KernelGroupedMmadWithScaleMx>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC,
                                                    BiasType, LayoutBias>;
    using GmmKernel = Apace::Kernel::GroupedMatmulMxKernel<ProblemShape, BlockMmad, BlockScheduler>;
    using MmLayoutB = typename AscendC::Std::conditional_t<TransMmB, AscendC::Te::DNExtLayoutPtn,
                                                           AscendC::Te::NDExtLayoutPtn>;
    using MmBlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, MmLayoutB, CType,
                                                      LayoutC, BiasType, LayoutBias>;
    using MmKernel = Apace::Kernel::GroupedMatmulMxKernel<ProblemShape, MmBlockMmad, BlockScheduler>;

    static constexpr uint64_t xHcclDataType_ = GetA2avHcclDataType<AType>();

    /*!
     * \brief 初始化 HCCL；通信任务在所有 MIX 子核初始化完成后下发
     *
     * \param[in] tilingGM tiling buffer 首地址：GET_TILING_DATA_WITH_STRUCT 得到的是本地拷贝，
     *            groupList/A/C offset tables 以 __gm__ 指针按偏移直接引用 tiling buffer
     */
    __aicore__ inline void Init(GM_ADDR gmmxGM, GM_ADDR gmmxScaleGM, GM_ADDR gmmWeightGM,
                                GM_ADDR gmmWeightScaleGM, GM_ADDR mmxGM, GM_ADDR mmxScaleGM,
                                GM_ADDR mmWeightGM, GM_ADDR mmWeightScaleGM, GM_ADDR gmmyGM, GM_ADDR mmyGM,
                                GM_ADDR permuteOutGM, GM_ADDR userWorkspaceGM, GM_ADDR tilingGM)
    {
        Trace(0U);
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
        sparseGroupListGm_ = reinterpret_cast<__gm__ int64_t *>(
            tilingGM + offsetof(hcommA2avQuantGmmTilingData, sparseGroupList));
        aOffsetTableGm_ = reinterpret_cast<__gm__ const uint64_t *>(
            tilingGM + offsetof(hcommA2avQuantGmmTilingData, aOffsetTable));
        cOffsetTableGm_ = reinterpret_cast<__gm__ const uint64_t *>(
            tilingGM + offsetof(hcommA2avQuantGmmTilingData, cOffsetTable));
        mmGroupListGm_ = reinterpret_cast<__gm__ int64_t *>(
            tilingGM + offsetof(hcommA2avQuantGmmTilingData, mmGroupList));

        // Match the repository-native HcclA2avOp lifecycle: InitV2 and
        // SetCcTilingV2 run on every MIX sub-core. Only collective launch,
        // wait and finalize are restricted to AIV. Blaze objects are created
        // later, only within their non-overlapping AIC compute phases.
        hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
        hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(hcommA2avQuantGmmTilingData, mc2CcTiling)));
        rankDim_ = hccl_.GetRankDim();
        Trace(1U);
    }

    /*!
     * \brief 等待通信完成并执行段式分组量化 Matmul（AIC only）
     */
    __aicore__ inline void Run()
    {
        // Do not let AIV0 submit the first collective while another MIX
        // sub-core is still initializing its HCCL context.
        Trace(10U);
        AscendC::SyncAll<false>();
        Trace(11U);
        if ASCEND_IS_AIV {
            RunCommunication();
        } else if (tilingData_->diagnosticMode != A2AV_GMM_DIAG_COMM_ONLY) {
            // Match the native scheduler: shared-expert MM runs on AIC while AIV
            // starts the communication rounds, instead of serializing it after GMM.
            if (tilingData_->localMatmul != 0U &&
                (tilingData_->diagnosticMode == A2AV_GMM_DIAG_FULL ||
                 tilingData_->diagnosticMode == A2AV_GMM_DIAG_SHARED)) {
                Trace(20U);
                RunLocalMatmul();
                Trace(21U);
                AicBarrier();
                Trace(22U);
            }
            for (uint32_t expertIdx = 0; expertIdx < tilingData_->expertNum; ++expertIdx) {
                Trace(30U, expertIdx);
                WaitCommunicationDone();
                AcknowledgeCommunication();
                // WaitCommunicationDone blocks MTE2, not the scalar pipe.
                // This marker means the waits were issued, not completed.
                Trace(31U, expertIdx);
                if (tilingData_->diagnosticMode == A2AV_GMM_DIAG_FULL ||
                    tilingData_->diagnosticMode == A2AV_GMM_DIAG_ROUTED) {
                    RunExpertMatmul(expertIdx);
                } else {
                    // Even without a GEMM consumer, drain the MTE2 waits so
                    // comm+sync diagnostics actually test the hand-off.
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                Trace(32U, expertIdx);
                // Keep every AIC on the same expert before reusing the cross-core
                // communication-ready event for the next expert.
                AicBarrier();
                Trace(33U, expertIdx);
            }
        }

        // InitV2 ran on every MIX sub-core, so every sub-core must also reach
        // the lifecycle barrier before AIV finalizes HCCL.  Keeping this inside
        // RunCommunication deadlocks AIV after AIC returns from Run().
        Trace(90U);
        AscendC::SyncAll<false>();
        Trace(91U);
        if ASCEND_IS_AIV {
            Trace(92U);
            hccl_.Finalize();
            Trace(93U);
        }
    }

private:
    // Opt-in, bounded diagnostics. Print only the first logical MIX block and the
    // first four / last expert. Missing printf output alone is not evidence
    // that a stage was never entered (dump buffering / timeout may hide it).
    // 0/1: Init; 10/11: initial MIX barrier; 20/21: shared MM lifetime;
    // 22: shared AIC barrier; 30/31: comm wait issue; 32: routed MM/drain;
    // 33: expert AIC barrier; 40..44: prepare, scale submit/wait, data submit/wait;
    // 50/51: AIV barrier; 52: notification issued; 53: AIC acknowledgement received;
    // 90/91: final MIX barrier;
    // 92/93: Finalize. Only each AFTER marker demonstrates that boundary returned.
    __aicore__ inline void Trace(uint32_t stage, uint32_t expertIdx = 0U) const
    {
        if (tilingData_->diagnosticTrace == 0U || AscendC::GetBlockIdx() >= AscendC::GetTaskRation() ||
            (expertIdx >= 4U && expertIdx + 1U != tilingData_->expertNum)) {
            return;
        }
        if ASCEND_IS_AIC {
            AscendC::printf("A2AVGMM_APACE AIC stage=%u expert=%u mode=%u\n", stage, expertIdx,
                            tilingData_->diagnosticMode);
        } else {
            AscendC::printf("A2AVGMM_APACE AIV stage=%u expert=%u mode=%u sub=%u\n", stage, expertIdx,
                            tilingData_->diagnosticMode, static_cast<uint32_t>(AscendC::GetSubBlockIdx()));
        }
    }

    __aicore__ inline bool IsCommunicationCore() const
    {
        return AscendC::GetBlockIdx() == 0U;
    }

    __aicore__ inline void WaitCommunicationDone()
    {
        AscendC::CrossCoreWaitFlag<A2AV_GMM_SYNC_MODE_AIV_TO_AIC, PIPE_MTE2>(A2AV_GMM_SYNC_FLAG_COMM_DONE);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        AscendC::CrossCoreWaitFlag<A2AV_GMM_SYNC_MODE_AIV_TO_AIC, PIPE_MTE2>(
            A2AV_GMM_SYNC_FLAG_COMM_DONE + A2AV_GMM_SYNC_FLAG_AIV1_OFFSET);
#endif
    }

    __aicore__ inline void AicBarrier()
    {
        AscendC::CrossCoreSetFlag<A2AV_GMM_SYNC_MODE_AIC_BARRIER, PIPE_FIX>(A2AV_GMM_SYNC_FLAG_AIC_BARRIER);
        AscendC::CrossCoreWaitFlag<A2AV_GMM_SYNC_MODE_AIC_BARRIER, PIPE_S>(A2AV_GMM_SYNC_FLAG_AIC_BARRIER);
    }

    __aicore__ inline void AcknowledgeCommunication()
    {
        // Use the same pipe as the ready waits. AIV cannot advance until
        // this AIC has actually consumed both ready tokens.
        AscendC::CrossCoreSetFlag<A2AV_GMM_SYNC_MODE_AIV_TO_AIC, PIPE_MTE2>(
            A2AV_GMM_SYNC_FLAG_COMM_CONSUMED);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
        AscendC::CrossCoreSetFlag<A2AV_GMM_SYNC_MODE_AIV_TO_AIC, PIPE_MTE2>(
            A2AV_GMM_SYNC_FLAG_COMM_CONSUMED + A2AV_GMM_SYNC_FLAG_AIV1_OFFSET);
#endif
    }

    __aicore__ inline void RunCommunication()
    {
        for (uint32_t expertIdx = 0; expertIdx < tilingData_->expertNum; ++expertIdx) {
            if (IsCommunicationCore()) {
                RunExpertCommunication(expertIdx);
            }
            // Match A2avGmmScheduler: first make AIV0's wait visible to all
            // AIVs, then both vector sub-blocks notify their paired AIC.
            Trace(50U, expertIdx);
            AscendC::SyncAll<true>();
            Trace(51U, expertIdx);
            if (tilingData_->diagnosticMode != A2AV_GMM_DIAG_COMM_ONLY) {
                // Each AIV has IDs 0..15; AIV1's ID 8 maps to AIC ID 24.
                AscendC::CrossCoreSetFlag<A2AV_GMM_SYNC_MODE_AIV_TO_AIC, PIPE_V>(A2AV_GMM_SYNC_FLAG_COMM_DONE);
                Trace(52U, expertIdx);
                // Bound outstanding notifications to one, even if shared MM
                // is slow or many experts are empty. Reuse only after the AIC
                // consumes this ready token; next comm can still overlap GEMM.
                AscendC::CrossCoreWaitFlag<A2AV_GMM_SYNC_MODE_AIV_TO_AIC, PIPE_S>(
                    A2AV_GMM_SYNC_FLAG_COMM_CONSUMED);
                Trace(53U, expertIdx);
            }
        }
    }

    __aicore__ inline void RunExpertCommunication(uint32_t expertIdx)
    {
        PrepareExpertCommunication(expertIdx);
        Trace(40U, expertIdx);
        // MX scale is an E8M0 byte payload.  The HCCL server validates each
        // collective against the data type used to build its CC tiling, so
        // transport the scale bytes with the same one-byte type as X.  The
        // collective only copies bytes; the E8M0 bit pattern is unchanged.
        scaleHandle_ = hccl_.template AlltoAllV<true>(
            xScaleGM_, scaleSendCnt_, scaleSendOff_, static_cast<AscendC::HcclDataType>(xHcclDataType_), scaleRecvGM_,
            scaleRecvCnt_, scaleRecvOff_, static_cast<AscendC::HcclDataType>(xHcclDataType_));
        Trace(41U, expertIdx);
        hccl_.Wait(scaleHandle_);
        Trace(42U, expertIdx);
        dataHandle_ = hccl_.template AlltoAllV<true>(
            xGM_, dataSendCnt_, dataSendOff_, static_cast<AscendC::HcclDataType>(xHcclDataType_), dataRecvGM_,
            dataRecvCnt_, dataRecvOff_, static_cast<AscendC::HcclDataType>(xHcclDataType_));
        Trace(43U, expertIdx);
        hccl_.Wait(dataHandle_);
        Trace(44U, expertIdx);
    }

    __aicore__ inline uint32_t GetSegmentRows(uint32_t segIdx) const
    {
        return static_cast<uint32_t>(tilingData_->sparseGroupList[segIdx * 2U + 1U]);
    }

    __aicore__ inline void PrepareExpertCommunication(uint32_t expertIdx)
    {
        const uint64_t dataAxis = tilingData_->isMxFp4 != 0U ? (static_cast<uint64_t>(tilingData_->k) + 1U) / 2U
                                                             : static_cast<uint64_t>(tilingData_->k);
        const uint64_t scaleAxis = tilingData_->scaleKLen;
        const uint32_t rankSize = rankDim_;
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
            const uint64_t recvRowOffset = tilingData_->cOffsetTable[segIdx] / tilingData_->n;
            dataSendCnt_[peer] = sendRows * dataAxis;
            dataSendOff_[peer] = tilingData_->dataSendOff[peer] + sendRowOffset * dataAxis;
            dataRecvCnt_[peer] = recvRows * dataAxis;
            dataRecvOff_[peer] = recvRowOffset * dataAxis;
            scaleSendCnt_[peer] = sendRows * scaleAxis;
            scaleSendOff_[peer] = tilingData_->scaleSendOff[peer] + sendRowOffset * scaleAxis;
            scaleRecvCnt_[peer] = recvRows * scaleAxis;
            scaleRecvOff_[peer] = recvRowOffset * scaleAxis;
        }
    }

    __aicore__ inline void RunExpertMatmul(uint32_t expertIdx)
    {
        typename GmmKernel::GMMTiling gmm;
        gmm.groupNum = tilingData_->rankSize;
        gmm.m = 0;
        for (uint32_t peer = 0; peer < tilingData_->rankSize; ++peer) {
            const uint32_t segIdx = expertIdx * tilingData_->rankSize + peer;
            gmm.m += tilingData_->sparseGroupList[segIdx * 2U + 1U];
        }
        if (gmm.m == 0) {
            return;
        }
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
        gmm.groupListType = Apace::Kernel::GROUP_LIST_TYPE_SPARSE;

        typename GmmKernel::Params params;
        params.mmadParams.aGmAddr = dataRecvGM_;
        params.mmadParams.scaleAGmAddr = scaleRecvGM_;
        params.mmadParams.bGmAddr = weightGM_;
        params.mmadParams.scaleBGmAddr = weightScaleGM_;
        params.mmadParams.cGmAddr = yGM_;
        params.mmadParams.biasGmAddr = nullptr;
        const uint32_t firstSegment = expertIdx * tilingData_->rankSize;
        params.groupListGmAddr = reinterpret_cast<GM_ADDR>(sparseGroupListGm_ + firstSegment * 2U);
        params.groupOffsetTables.aOffsetTable = aOffsetTableGm_ + firstSegment;
        params.groupOffsetTables.cOffsetTable = cOffsetTableGm_ + firstSegment;
        params.gmmParams = gmm;
        if ASCEND_IS_AIC {
            // BlockMmad seeds fixed HardEvent IDs in its constructor and
            // drains them in its destructor. Never overlap two such owners.
            {
                GmmKernel gmmKernel;
                gmmKernel(params);
            }
            // Finish destructor waits AND the final FIX write before another
            // BlockMmad instance can reseed events / reuse L0C and L1.
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void RunLocalMatmul()
    {
        typename MmKernel::GMMTiling mm;
        mm.groupNum = 1U;
        mm.m = tilingData_->mmM;
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
        mm.groupListType = Apace::Kernel::GROUP_LIST_TYPE_OFFSET;

        typename MmKernel::Params params;
        params.mmadParams.aGmAddr = mmxGM_;
        params.mmadParams.scaleAGmAddr = mmxScaleGM_;
        params.mmadParams.bGmAddr = mmWeightGM_;
        params.mmadParams.scaleBGmAddr = mmWeightScaleGM_;
        params.mmadParams.cGmAddr = mmyGM_;
        params.mmadParams.biasGmAddr = nullptr;
        params.groupListGmAddr = reinterpret_cast<GM_ADDR>(mmGroupListGm_);
        params.groupOffsetTables = {};
        params.gmmParams = mm;
        if ASCEND_IS_AIC {
            {
                MmKernel mmKernel;
                mmKernel(params);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
    hcommA2avQuantGmmTilingData *tilingData_{nullptr};
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle scaleHandle_{0};
    AscendC::HcclHandle dataHandle_{0};
    uint32_t rankDim_{0};

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
    // Group metadata __gm__ pointers reference the serialized tiling buffer.
    __gm__ int64_t *sparseGroupListGm_{nullptr};
    __gm__ const uint64_t *aOffsetTableGm_{nullptr};
    __gm__ const uint64_t *cOffsetTableGm_{nullptr};
    __gm__ int64_t *mmGroupListGm_{nullptr};

    // AlltoAllV 本地参数表（元素计）
    uint64_t dataSendCnt_[A2AV_GMM_HCCL_PARAM_RANK_NUM] = {0};
    uint64_t dataSendOff_[A2AV_GMM_HCCL_PARAM_RANK_NUM] = {0};
    uint64_t dataRecvCnt_[A2AV_GMM_HCCL_PARAM_RANK_NUM] = {0};
    uint64_t dataRecvOff_[A2AV_GMM_HCCL_PARAM_RANK_NUM] = {0};
    uint64_t scaleSendCnt_[A2AV_GMM_HCCL_PARAM_RANK_NUM] = {0};
    uint64_t scaleSendOff_[A2AV_GMM_HCCL_PARAM_RANK_NUM] = {0};
    uint64_t scaleRecvCnt_[A2AV_GMM_HCCL_PARAM_RANK_NUM] = {0};
    uint64_t scaleRecvOff_[A2AV_GMM_HCCL_PARAM_RANK_NUM] = {0};
};

} // namespace Apace
