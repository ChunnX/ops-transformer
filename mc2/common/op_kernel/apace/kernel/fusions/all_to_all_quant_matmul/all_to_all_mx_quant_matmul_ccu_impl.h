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
 * \file all_to_all_mx_quant_matmul_ccu_impl.h
 * \brief AllToAll MX Quant Matmul（CCU 版）AIC 侧融合实现
 *
 * 通信同步原语（PrepareAllToAllComm/CommitCcuComm/WaitCcuCommDone）在
 * core/ccu_comm/all_to_all/ccu_all_to_all_comm.h；CCU 侧 kernel 见
 * core/ccu_comm/all_to_all/ccu_all_to_all_kernel.asc。
 * 本文件：CcuCommWaitPolicy（qbmm CommPolicy 注入位，逐 tile 轮询 round-done，
 * 与 matmul 消费节奏耦合——tile 级通算 overlap 的来源）+ 融合 matmul impl +
 * AllToAllMatmulProcess（AIC 编排：帧首事件清扫 → prepare → commit → matmul）。
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "../../matmul/quant_batch_matmul/all_to_all_qbmm_mx_kernel.h"
#include "../../../utils/op_state_dump.h"
#include "apace/core/ccu_comm/all_to_all/ccu_all_to_all_comm.h"
#include "all_to_all_mx_quant_matmul_ccu_tiling_data.h"

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"

namespace Apace {

// CCU 直调版通信等待策略（注入 AllToAllQbmmMxKernel 的 CommPolicy 位）：
// tileIdx==0 前置等待 scale 轮（eventVa[1]），随后等待 data 轮 tileIdx（eventVa[tileIdx+2]）
struct CcuCommWaitPolicy {
    const CcuMappedAddrs *mappedAddrs_{nullptr};

    __aicore__ inline void WaitTile(uint32_t tileIdx)
    {
        if (tileIdx == 0) {
            WaitCcuCommDone(mappedAddrs_->eventVa[1]);
        }
        WaitCcuCommDone(mappedAddrs_->eventVa[tileIdx + 2]);
    }
};

// CCU 直调版 AllToAll + MXFP8 MatMul 融合 impl（镜像 AllToAllMxQuantMatmulHcommImpl，
// Init/Run 去 HCCL 化——通信由 streamCcu 上的 CCU kernel 独立执行）
template <typename X1Type, typename X2Type, typename YType>
class AllToAllMxQuantMatmulCcuImpl {
public:
    explicit __aicore__ inline AllToAllMxQuantMatmulCcuImpl(QuantMatmulTilingData *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x1Scale, GM_ADDR x2, GM_ADDR x2Scale,
                                GM_ADDR y, GM_ADDR commOut, GM_ADDR commScale,
                                const CcuMappedAddrs &mappedAddrs,
                                uint32_t rankId, uint32_t rankSize, uint32_t tileM)
    {
        x1_ = x1;
        x1Scale_ = x1Scale;
        x2_ = x2;
        x2Scale_ = x2Scale;
        y_ = y;
        commOutGM_ = commOut;
        commX1ScaleGM1_ = commScale;
        mappedAddrs_ = mappedAddrs;
        rankId_ = rankId;
        rankSize_ = rankSize;
        tileM_ = tileM;
        originalM_ = static_cast<uint64_t>(tilingData_->m);
        quantMatmulKernelImpl_.GetCommPolicy().mappedAddrs_ = &mappedAddrs_;
        // CCU 版暂不接入 DFX 状态 dump（无独立 dump workspace；nullptr Init 在
        // MC2_DFX_ENABLE 开/关两种实现下均为安全空操作）
        opStateDump_.Init(nullptr, nullptr, 0);
    }

    // 融合 REMOTE 单趟（localMatmul=0：self rank 直读 x1 零拷贝，跨源 L0C 累加）
    __aicore__ inline void Run()
    {
        MatmulProcess(MatmulMode::REMOTE);
    }

    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::dn_ext_layout_ptn;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = asc::te::nd_ext_layout_ptn;
    using BiasType = float;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC, BiasType, LayoutBias>;
    using QuantMatmulKernelImpl = Blaze::Gemm::Kernel::AllToAllQbmmMxKernel<
        ProblemShape, BlockMmad, BlockScheduler, CcuCommWaitPolicy>;

    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using MatmulMode = typename QuantMatmulKernelImpl::MatmulMode;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;
    Mc2Kernel::OpStateDump opStateDump_{};

private:
    __aicore__ inline void SetupParams(Params &out, MatmulMode matmulMode)
    {
        const QuantMatmulTilingData &mmTile = *tilingData_;

        ProblemShape problemShape{mmTile.m, mmTile.n, mmTile.k, 1UL};

        BlockMmadParams mmadParams;
        mmadParams.aGmAddr = commOutGM_;
        mmadParams.scaleAGmAddr = commX1ScaleGM1_;
        mmadParams.bGmAddr = x2_;
        mmadParams.scaleBGmAddr = x2Scale_;
        mmadParams.cGmAddr = y_;
        mmadParams.biasGmAddr = 0;

        LocalParams localParams{rankId_, rankSize_, originalM_, x1_, x1Scale_,
                                0UL, rankSize_, matmulMode, tileM_};
        L1Params l1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1,
                          mmTile.nBufferNum};
        BlockSchedulerParams schedulerParams{
            mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
            mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};
        QBMMTiling qbmmParams{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c, 0U};

        out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
    }

    __aicore__ inline void MatmulProcess(MatmulMode matmulMode)
    {
        Params params;
        SetupParams(params, matmulMode);
        quantMatmulKernelImpl_(params, opStateDump_);
    }

    QuantMatmulTilingData *tilingData_;

    GM_ADDR x1_;
    GM_ADDR x1Scale_;
    GM_ADDR x2_;
    GM_ADDR x2Scale_;
    GM_ADDR y_;
    GM_ADDR commOutGM_;
    GM_ADDR commX1ScaleGM1_;
    CcuMappedAddrs mappedAddrs_;

    uint32_t rankId_{0};
    uint32_t rankSize_{0};
    uint32_t tileM_{0};
    uint64_t originalM_{0};
};

} // namespace Apace

// AIC 侧融合主逻辑：block 0 清 CKE 事件残值 → prepare → commit（顺序约束：CCU 只唤醒一次，
// 清零在 commit 之前保证不吞 round-done）；全核融合 matmul（逐 tile WaitTile 与 CCU 通信 overlap）
__aicore__ inline void AllToAllMatmulProcess(
    __gm__ void *x1, __gm__ void *x1Scale,
    __gm__ void *x2, __gm__ void *x2Scale,
    __gm__ void *output, __gm__ void *workspace,
    ccuAllToAllMatmulTilingData tilingData)
{
    const CcuCommTilingData &comm = tilingData.comm;

    __gm__ uint8_t *wsBytes = reinterpret_cast<__gm__ uint8_t *>(workspace);
    __gm__ void *recvBuf = workspace;
    __gm__ void *recvScaleBuf = reinterpret_cast<__gm__ void *>(
        wsBytes + comm.workspaceScaleOffset);

    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) == 0) {
        const uint32_t totalRounds = comm.dataCommTurn + 1;
        for (uint32_t i = 0; i <= totalRounds; i++) {
            __gm__ uint64_t *evAddr =
                reinterpret_cast<__gm__ uint64_t *>(comm.mappedAddrs.eventVa[i]);
            AscendC::WriteGmByPassDCache<uint64_t>(evAddr, 0);
        }
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
        PrepareAllToAllComm(comm, x1, x1Scale, recvBuf, recvScaleBuf);
        CommitCcuComm(comm.mappedAddrs);
    }

    Apace::AllToAllMxQuantMatmulCcuImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t> matmulImpl(
        &tilingData.matmul);
    matmulImpl.Init(reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(x1)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(x1Scale)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(x2)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(x2Scale)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(output)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(recvBuf)),
                    reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(recvScaleBuf)),
                    comm.mappedAddrs, comm.rankId, comm.rankSize, comm.tileM);
    matmulImpl.Run();
}
