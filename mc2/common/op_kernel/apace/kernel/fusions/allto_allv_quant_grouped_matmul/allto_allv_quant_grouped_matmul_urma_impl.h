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
 * \file allto_allv_quant_grouped_matmul_urma_impl.h
 * \brief AllToAllv + MX 量化 GroupedMatMul 融合实现（URMA PUT 版本）
 *
 * Run():
 *   AIV: RunAllToAllv() 按 (dst, expert) 段直接写入对端窗口的 expert-major 位置
 *        （通信即 permute，免去独立 permute 步骤），本地段经 UB 中转拷贝入窗，
 *        Drain + 跨卡 TeamBarrier 保证全局可见后通知 AIC；
 *   AIC: RunGroupedMatmul() 等待通信完成后，以 groupList 驱动分组 MX 量化 Matmul。
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "include/tensor_api/tensor.h"

#include "adv_api/hcomm/hcomm.h"
#include "apace/kernel/matmul/grouped_matmul/qgmm_mx_kernel.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "allto_allv_quant_grouped_matmul_tiling_data.h"
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h" // CommContext

#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/core/aiv_comm/barrier/barrier_ubmem.h"
#include "apace/tiling/comm_tiling_data.h"

namespace Apace {

using namespace AscendC;
using Blaze::Gemm::CeilDiv;
using namespace Apace::AivComm;

/*!
 * \brief AllToAllv 量化分组矩阵乘核心实现
 *
 * 模板参数：
 *   AType/BType: 量化数据类型（fp8_e4m3fn_t / fp8_e5m2_t）
 *   CType:       输出类型（bfloat16_t / half_t）
 *   TransB:      权重是否转置存放（[E, N, K]）
 */
template <typename AType, typename BType, typename CType, bool TransB>
class AllToAllvQuantGroupedMatmulUrmaImpl {
public:
    __aicore__ inline AllToAllvQuantGroupedMatmulUrmaImpl() {}
    __aicore__ inline ~AllToAllvQuantGroupedMatmulUrmaImpl() {}

    using TypeScaleA = ::fp8_e8m0_t;
    using TypeScaleB = ::fp8_e8m0_t;
    static constexpr int32_t SCALE_C0 = 2;

    // Layout 定义：A/C 行主序；B 随 TransB；scale 布局与算子规格一致
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

    __aicore__ inline void Init(__gm__ CommContext *hcommCtx, GM_ADDR xGM, GM_ADDR xScaleGM, GM_ADDR weightGM,
                                GM_ADDR weightScaleGM, GM_ADDR yGM, GM_ADDR tilingGM);
    __aicore__ inline void Run();

private:
    // ---------------- AIV 侧：通信与本地落窗 ----------------
    __aicore__ inline void RunAllToAllv();
    __aicore__ inline void CopyLocalSegments();
    __aicore__ inline void GmToGmCopy(GM_ADDR srcAddr, GM_ADDR dstAddr, uint64_t bytes);
    // ---------------- AIC 侧：分组量化 Matmul ----------------
    __aicore__ inline void RunGroupedMatmul();

    // ---------------- 通信与上下文 ----------------
    __gm__ CommUbmemContext *syncBuffer_{nullptr};
    __gm__ CommUdmaContext *udmaCtx_{nullptr};
    Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> comm_;
    TeamBarrier teamBarrier_;
    GmmKernel gmmKernel_;

    const __gm__ AllToAllvQuantGmmTilingData *tiling_{nullptr};
    GM_ADDR selfWin_{nullptr}; // 本 rank 通信窗口（HCCL buffer）
    GM_ADDR xGM_{nullptr};
    GM_ADDR xScaleGM_{nullptr};
    GM_ADDR weightGM_{nullptr};
    GM_ADDR weightScaleGM_{nullptr};
    GM_ADDR yGM_{nullptr};

    uint32_t rankId_{0};
    uint32_t rankSize_{0};
    uint32_t blockIdx_{0};
    uint64_t copyBufOffset_{0}; // UB 内本地拷贝中转缓冲偏移
};

// =================================================================================
// 公有方法实现
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransB>
__aicore__ inline void AllToAllvQuantGroupedMatmulUrmaImpl<AType, BType, CType, TransB>::Init(
    __gm__ CommContext *hcommCtx, GM_ADDR xGM, GM_ADDR xScaleGM, GM_ADDR weightGM, GM_ADDR weightScaleGM, GM_ADDR yGM,
    GM_ADDR tilingGM)
{
    tiling_ = reinterpret_cast<const __gm__ AllToAllvQuantGmmTilingData *>(tilingGM);
    xGM_ = xGM;
    xScaleGM_ = xScaleGM;
    weightGM_ = weightGM;
    weightScaleGM_ = weightScaleGM;
    yGM_ = yGM;

    syncBuffer_ = &(hcommCtx->ubmemCtx);
    udmaCtx_ = &(hcommCtx->udmaCtx);
    rankId_ = udmaCtx_->rankId;
    rankSize_ = udmaCtx_->rankSize;
    blockIdx_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
    selfWin_ = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[rankId_]);

    // UB 布局：[0, 512) 通信工作区；[512, 512 + 32K) 本地拷贝中转
    uint32_t ubOffset = 0;
    auto commBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    copyBufOffset_ = ubOffset;
    auto barrierBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset);
    teamBarrier_.Init(barrierBuf.Get(), syncBuffer_, rankSize_, blockIdx_);
    comm_.Init(commBuf.Get(), COMM_WORKSPACE_SIZE);
}

template <typename AType, typename BType, typename CType, bool TransB>
__aicore__ inline void AllToAllvQuantGroupedMatmulUrmaImpl<AType, BType, CType, TransB>::Run()
{
    if ASCEND_IS_AIV {
        RunAllToAllv();
    }
    if ASCEND_IS_AIC {
        RunGroupedMatmul();
    }
}

// =================================================================================
// AIV 侧实现：AllToAllv（直接 expert-major 落窗）+ 本地段拷贝
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransB>
__aicore__ inline void AllToAllvQuantGroupedMatmulUrmaImpl<AType, BType, CType, TransB>::RunAllToAllv()
{
    const uint32_t expertNum = tiling_->expertNum;
    const uint64_t k = tiling_->k;
    const uint64_t scaleKLen = tiling_->scaleKLen;

    // 1. 远程段：blockIdx < rankSize 的 block 负责目标 rank == blockIdx（与 CollectiveCommBase 划分一致）
    if (blockIdx_ < rankSize_) {
        const uint32_t dst = blockIdx_;
        if (dst != rankId_) {
            const ChannelHandle channel = static_cast<ChannelHandle>(udmaCtx_->channelHandles[dst]);
            GM_ADDR dstWin = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[dst]);
            for (uint32_t e = 0; e < expertNum; ++e) {
                const uint32_t cnt = tiling_->sendCnt[dst][e];
                if (cnt == 0) {
                    continue;
                }
                const uint32_t srcRow = tiling_->sendSrcRow[dst][e];
                const uint32_t dstRow = tiling_->sendDstRow[dst][e];
                // 行分片，单次 WriteNbi 控制在 A2AV_GMM_COMM_TILE_BYTES 内
                uint64_t tileRows = A2AV_GMM_COMM_TILE_BYTES / k;
                if (tileRows == 0) {
                    tileRows = 1;
                }
                if (tileRows > cnt) {
                    tileRows = cnt;
                }
                for (uint32_t row = 0; row < cnt; row += static_cast<uint32_t>(tileRows)) {
                    uint64_t rows = tileRows;
                    if (static_cast<uint64_t>(cnt) - row < rows) {
                        rows = static_cast<uint64_t>(cnt) - row;
                    }
                    const uint64_t aBytes = rows * k;
                    GM_ADDR srcA = reinterpret_cast<GM_ADDR>(xGM_) + (srcRow + row) * k;
                    GM_ADDR dstA = dstWin + (dstRow + row) * k;
                    int32_t ret = comm_.WriteNbi(channel, dstA, srcA, aBytes);
                    ascendc_assert(ret == 0, "A2AV write A failed, ret=%d, dst=%u, expert=%u", ret, dst, e);

                    const uint64_t scaleBytes = rows * scaleKLen;
                    GM_ADDR srcS = reinterpret_cast<GM_ADDR>(xScaleGM_) + (srcRow + row) * scaleKLen;
                    GM_ADDR dstS = dstWin + tiling_->dstWinScaleOffset[dst] + (dstRow + row) * scaleKLen;
                    ret = comm_.WriteNbi(channel, dstS, srcS, scaleBytes);
                    ascendc_assert(ret == 0, "A2AV write scale failed, ret=%d, dst=%u, expert=%u", ret, dst, e);
                }
            }
            int32_t ret = comm_.Drain(channel);
            ascendc_assert(ret == 0, "A2AV drain failed, ret=%d, dst=%u", ret, dst);
        }
    }

    // 2. 本地段（src == dst == rankId）：经 UB 中转拷入本窗口 expert-major 位置，按专家分片到各 block
    CopyLocalSegments();

    // 3. 跨卡屏障：保证所有 rank 对本窗口的写入完成（每个 block 负责 dst==blockIdx 的写入）
    if (blockIdx_ < rankSize_) {
        teamBarrier_.CrossDevice();
    }

    // 4. 核内同步（本地拷贝完成）后通知 AIC
    AscendC::SyncAll<true>();
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0);

    // 通信接口使用了 channel 内的字段，刷新 GM 地址保证跨 kernel launch 的 GM 状态与 aiCore cache 一致
    dcci(reinterpret_cast<__gm__ void *>(udmaCtx_->commBufferAddrs),
         static_cast<uint64_t>(CacheLine::ENTIRE_DATA_CACHE), static_cast<uint64_t>(DcciDst::CACHELINE_OUT));
}

template <typename AType, typename BType, typename CType, bool TransB>
__aicore__ inline void AllToAllvQuantGroupedMatmulUrmaImpl<AType, BType, CType, TransB>::CopyLocalSegments()
{
    const uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
    const uint32_t expertNum = tiling_->expertNum;
    const uint64_t k = tiling_->k;
    const uint64_t scaleKLen = tiling_->scaleKLen;

    for (uint32_t e = 0; e < expertNum; ++e) {
        if ((e % blockNum) != blockIdx_) {
            continue;
        }
        const uint32_t cnt = tiling_->recvCnt[rankId_][e];
        if (cnt == 0) {
            continue;
        }
        const uint32_t srcRow = tiling_->sendSrcRow[rankId_][e];
        const uint32_t dstRow = tiling_->recvDstRow[rankId_][e];
        GmToGmCopy(reinterpret_cast<GM_ADDR>(xGM_) + static_cast<uint64_t>(srcRow) * k,
                   reinterpret_cast<GM_ADDR>(selfWin_) + static_cast<uint64_t>(dstRow) * k,
                   static_cast<uint64_t>(cnt) * k);
        GmToGmCopy(reinterpret_cast<GM_ADDR>(xScaleGM_) + static_cast<uint64_t>(srcRow) * scaleKLen,
                   reinterpret_cast<GM_ADDR>(selfWin_) + tiling_->winScaleOffset +
                       static_cast<uint64_t>(dstRow) * scaleKLen,
                   static_cast<uint64_t>(cnt) * scaleKLen);
    }
}

template <typename AType, typename BType, typename CType, bool TransB>
__aicore__ inline void AllToAllvQuantGroupedMatmulUrmaImpl<AType, BType, CType, TransB>::GmToGmCopy(
    GM_ADDR srcAddr, GM_ADDR dstAddr, uint64_t bytes)
{
    constexpr uint64_t COPY_TILE = 32UL * 1024UL; // 32KB 中转
    auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
    auto copyUB2GM = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});
    for (uint64_t off = 0; off < bytes; off += COPY_TILE) {
        const uint64_t remain = bytes - off;
        const uint64_t cur = remain < COPY_TILE ? remain : COPY_TILE;
        auto layout = AscendC::Te::FrameLayoutFormat<AscendC::Te::NDExtLayoutPtn,
                                                     AscendC::Te::LayoutTraitDefault<uint8_t>>{}(cur, 1);
        auto ubT = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(copyBufOffset_), layout);
        auto gmS = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(srcAddr + off), layout);
        auto gmD = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(dstAddr + off), layout);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(1);
        AscendC::Te::Copy(copyGM2UB, ubT, gmS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(1);

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(1);
        AscendC::Te::Copy(copyUB2GM, gmD, ubT);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(1);
    }
}

// =================================================================================
// AIC 侧实现：分组 MX 量化 Matmul
// =================================================================================

template <typename AType, typename BType, typename CType, bool TransB>
__aicore__ inline void AllToAllvQuantGroupedMatmulUrmaImpl<AType, BType, CType, TransB>::RunGroupedMatmul()
{
    // 等待 AIV 侧全部通信与本地落窗完成
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);

    typename GmmKernel::GMMTiling gmm;
    gmm.groupNum = tiling_->expertNum;
    gmm.n = tiling_->n;
    gmm.k = tiling_->k;
    gmm.baseM = tiling_->baseM;
    gmm.baseN = tiling_->baseN;
    gmm.baseK = tiling_->baseK;
    gmm.kAL1 = tiling_->kAL1;
    gmm.kBL1 = tiling_->kBL1;
    gmm.scaleKAL1 = tiling_->scaleKAL1;
    gmm.isBias = 0;
    gmm.dbL0C = 2;
    gmm.l1BufferStage = 2;

    typename GmmKernel::Params params;
    params.aBasePtr = reinterpret_cast<__gm__ AType *>(selfWin_);
    params.scaleABasePtr = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(
        reinterpret_cast<uint64_t>(selfWin_) + tiling_->winScaleOffset);
    params.bBasePtr = reinterpret_cast<__gm__ BType *>(weightGM_);
    params.scaleBBasePtr = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(weightScaleGM_);
    params.cBasePtr = reinterpret_cast<__gm__ CType *>(yGM_);
    params.groupList = &(tiling_->groupListCum[0]);
    params.gmm = gmm;
    gmmKernel_(params);
}

} // namespace Apace
