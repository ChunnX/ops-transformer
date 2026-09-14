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
 * \file all_gather_mx_matmul_urma_impl.h
 * \brief AllGatherQuantMatmul 算子实现，基于 FragmentTensor + 通算解耦（UDMA 通信）
 */

#pragma once

#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_tiling_data.h"
#include "apace/kernel/matmul/quant_batch_matmul/all_gather_qbmm_mx_kernel.h"
#include "adv_api/hcomm/hcomm.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/core/aiv_comm/collective_comm_api.h"
#include "apace/core/aiv_comm/barrier/barrier_ubmem.h"
#include "apace/tiling/comm_tiling_data.h"
#include "include/tensor_api/tensor.h"
#include "../../../utils/op_state_dump.h"

namespace Apace {

using namespace AscendC;
using namespace Apace::AivComm;

using LayoutA = asc::te::nd_ext_layout_ptn;
using LayoutB = asc::te::dn_ext_layout_ptn;
using LayoutC = asc::te::nd_ext_layout_ptn;
using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

struct UrmaCommWaitPolicy {
    __aicore__ inline void WaitTile(uint32_t tileIdx)
    {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx);
    }
};

template <typename AType, typename BType, typename CType>
class AllGatherMxMatmulUrmaImpl {
public:
    __aicore__ inline AllGatherMxMatmulUrmaImpl() {}
    __aicore__ inline void Init(__gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM,
                                GM_ADDR bScaleGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR gatherOutGM,
                                GM_ADDR workspaceGM, const AllGatherMxMatmulUrmaTilingData *tilingData);
    __aicore__ inline void Process();

    using QuantMatmulKernelImpl = AllGatherQbmmMxKernel<AType, BType, CType, UrmaCommWaitPolicy>;
    using KernelParams = typename QuantMatmulKernelImpl::Params;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using FragmentParams = typename QuantMatmulKernelImpl::FragmentParams;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
    __aicore__ inline void InitBaseParams(const AllGatherMxMatmulUrmaTilingData *td);

    __aicore__ inline void AllGatherProcess();
    __aicore__ inline void MatmulProcess();

    __aicore__ inline bool IsCommCore() const;
    __aicore__ inline bool IsCopyCore() const;
    __aicore__ inline void CopyDataFromWin(uint32_t round);
    __aicore__ inline void CopyBytesPingPong(__gm__ int8_t *src, __gm__ int8_t *dst, uint64_t len);

    __aicore__ inline GM_ADDR GetWinDataRegionBase();
    __aicore__ inline GM_ADDR GetWinScaleRegionBase();

    // 通信
    using AllGatherCommData = Apace::AivComm::CollectiveComm<Apace::AivComm::CommCollectiveOp::AllGather,
                                                             Apace::AivComm::CommMode::PUT, AType, TeamBarrier>;
    using AllGatherCommScale =
        Apace::AivComm::CollectiveComm<Apace::AivComm::CommCollectiveOp::AllGather, Apace::AivComm::CommMode::PUT,
                                       AscendC::fp8_e8m0_t, TeamBarrier>;
    AllGatherCommData allGatherData_;
    AllGatherCommScale allGatherScale_;
    CommTilingData commTilingData_{};
    CommTilingData commTilingScale_{};
    TeamBarrier teamBarrier_;

    static constexpr uint32_t kPaddingLength = 16;

    __gm__ CommContext *hcommCtx_{};
    __gm__ CommUdmaContext *udmaCtx_{};
    __gm__ CommUbmemContext *ubmemCtx_{};

    GM_ADDR aGM_{};
    GM_ADDR aScaleGM_{};
    GM_ADDR bGM_{};
    GM_ADDR bScaleGM_{};
    GM_ADDR cGM_{};
    GM_ADDR biasGM_{};
    GM_ADDR gatherOutGM_{};

    uint32_t rankId_{};
    uint32_t rankSize_{};
    uint32_t blockNum_{};
    uint32_t copyCoreNum_{};
    uint32_t copyUbOffset_{};
    uint32_t m_{};
    uint32_t k_{};
    uint32_t n_{};
    uint32_t tileCnt_{};
    uint32_t tileM_{};
    uint32_t tailCnt_{};
    uint32_t tailM_{};
    uint32_t paddedTailM_{};
    uint32_t commTurn_{};
    uint64_t headRows_{};
    uint64_t scaleKLen_{};
    uint64_t dataBytesPerMRow_{};
    uint64_t scaleBytesPerMRow_{};
    uint64_t cBytesPerM_{};
    uint64_t scaleKGroups_{};
    uint64_t dataRegionBytes_{};
    uint64_t scaleRegionBytes_{};

    const AllGatherMxMatmulUrmaTilingData *tilingData_{};
    Mc2Kernel::OpStateDump opStateDump_{};
    static constexpr uint8_t POS_COMM_BEFORE = 0U;
    static constexpr uint8_t POS_COMP_CUBE_MATMUL_BEFORE = 1U;

    // ---- gather_out 拷贝（win -> UB -> gather_out）----
    using CopyGM2UB = decltype(asc::te::make_copy(asc::te::copy_gm_to_ub{}));
    using CopyUB2GM = decltype(asc::te::make_copy(asc::te::copy_ub_to_gm{}));
    using MakeLayoutByte =
        asc::te::frame_layout_format<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<int8_t>>;
    CopyGM2UB copyGM2UB_ = asc::te::make_copy(asc::te::copy_gm_to_ub{});
    CopyUB2GM copyUB2GM_ = asc::te::make_copy(asc::te::copy_ub_to_gm{});

    static constexpr uint16_t COPY_EVT_ID0 = 0;
    static constexpr uint16_t COPY_EVT_ID1 = 1;
    static constexpr uint32_t COPY_CHUNK_BYTES = 32U * 1024U;
    static constexpr uint32_t COPY_UB_ALIGN = 32U;
};

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::Init(
    __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
    GM_ADDR biasGM, GM_ADDR gatherOutGM, GM_ADDR workspaceGM, const AllGatherMxMatmulUrmaTilingData *tilingData)
{
    tilingData_ = tilingData;
    hcommCtx_ = hcommCtx;
    aGM_ = aGM;
    aScaleGM_ = aScaleGM;
    bGM_ = bGM;
    bScaleGM_ = bScaleGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    gatherOutGM_ = gatherOutGM;
#if MC2_DFX_ENABLE
    opStateDump_.Init(workspaceGM, &tilingData->dumpInfo.workspaceLayout, tilingData->mmTile.usedCoreNum);
#endif

    InitBaseParams(tilingData);

    udmaCtx_ = &(hcommCtx_->udmaCtx);
    ubmemCtx_ = &(hcommCtx_->ubmemCtx);
    rankId_ = udmaCtx_->rankId;
    rankSize_ = udmaCtx_->rankSize;
    // 核划分：后 rankSize_ 个核通信，前 copyCoreNum_ 个核做 win->UB->gather_out 拷贝
    blockNum_ = static_cast<uint32_t>(AscendC::GetBlockNum());
    copyCoreNum_ = blockNum_ - rankSize_;
    uint32_t commJobIdx = static_cast<uint32_t>(GetBlockIdx()) - copyCoreNum_;

    // Rank-major 布局：每个 rank 占完整的 M 行。data + scale 两段连续存放。
    //   Win buffer 内存布局：
    // ┌──────────────────────────────────────────────────┐
    // │  rank0 M行 data │ ... │ rankR-1 M行 data │          ← data 段
    // │  rank0 M行 scale│ ... │ rankR-1 M行 scale│         ← scale 段 (offset = dataRegionBytes_)
    // └──────────────────────────────────────────────────┘
    dataRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * dataBytesPerMRow_;
    scaleRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * scaleBytesPerMRow_;

    // 静态 tensor 替代 tpipe buffer
    uint32_t ubOffset = 0;
    auto commBuf = asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto commScaleBuf = asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>(ubOffset);
    ubOffset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = asc::te::make_mem_ptr<asc::te::location::ub, uint8_t>(ubOffset);
    teamBarrier_.Init(barrierBuf.get(), ubmemCtx_, rankSize_, commJobIdx);
    // 拷贝双 buffer 紧随 barrier 缓冲之后，按 COPY_UB_ALIGN 对齐
    ubOffset += BARRIER_FLAG_SIZE;
    copyUbOffset_ = (ubOffset + COPY_UB_ALIGN - 1) / COPY_UB_ALIGN * COPY_UB_ALIGN;

    commTilingData_.splitAxisTileSize = tileM_;  // 每个 tile 搬 tileM 行
    commTilingData_.splitAxisTileCnt = tileCnt_; // head 段 tile 数量
    commTilingData_.splitAxisTailSize = tailM_;  // tail 行数
    commTilingData_.splitAxisTailCnt = tailCnt_;
    // 通信按 Dtype 元素个数计：FP4(fp4x2) 每元素 2 个逻辑值，故按字节行宽折算
    commTilingData_.nonSplitAxisSize = dataBytesPerMRow_ / sizeof(AType); // 非切分轴k

    commTilingScale_.splitAxisTileSize = tileM_;
    commTilingScale_.splitAxisTileCnt = tileCnt_;
    commTilingScale_.splitAxisTailSize = tailM_;
    commTilingScale_.splitAxisTailCnt = tailCnt_;
    scaleKLen_ = scaleKGroups_ * static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE);
    commTilingScale_.nonSplitAxisSize = scaleKLen_;
    allGatherData_.template Init<BARRIER_NONE>(udmaCtx_, teamBarrier_, commTilingData_, aGM_, commBuf.get(), rankSize_,
                                               commJobIdx);
    allGatherScale_.Init(udmaCtx_, teamBarrier_, commTilingScale_, aScaleGM_, commScaleBuf.get(), rankSize_, commJobIdx,
                         dataRegionBytes_);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::InitBaseParams(
    const AllGatherMxMatmulUrmaTilingData *td)
{
    const auto &ct = td->commTile;
    tileCnt_ = static_cast<uint32_t>(ct.splitAxisTileCnt);
    tileM_ = static_cast<uint32_t>(ct.splitAxisTileSize);
    tailCnt_ = static_cast<uint32_t>(ct.splitAxisTailCnt);
    tailM_ = static_cast<uint32_t>(ct.splitAxisTailSize);
    k_ = td->mmTile.k;
    n_ = td->mmTile.n;
    m_ = static_cast<uint32_t>(ct.splitAxisTileSize * ct.splitAxisTileCnt + ct.splitAxisTailSize * ct.splitAxisTailCnt);
    commTurn_ = tileCnt_ + tailCnt_;
    paddedTailM_ = (tailM_ > 0) ? ((tailM_ + kPaddingLength - 1) / kPaddingLength * kPaddingLength) : 0U;
    headRows_ = static_cast<uint64_t>(tileCnt_) * tileM_;

    scaleKGroups_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(k_), Blaze::Gemm::MXFP_DIVISOR_SIZE);
    // FP4 (fp4x2) 1 字节装 2 个元素：每行字节数 = k/2；FP8 等 = k * sizeof(AType)
    dataBytesPerMRow_ =
        Blaze::Gemm::IsFp4<AType>() ? (static_cast<uint64_t>(k_) >> 1) : static_cast<uint64_t>(k_) * sizeof(AType);
    scaleBytesPerMRow_ =
        scaleKGroups_ * static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE) * sizeof(AscendC::fp8_e8m0_t);
    cBytesPerM_ = static_cast<uint64_t>(n_) * sizeof(CType);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::Process()
{
    if ASCEND_IS_AIV {
        AllGatherProcess();
    }
    if ASCEND_IS_AIC {
        MatmulProcess();
    }
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::MatmulProcess()
{
    KernelParams params;
    params.mmTile = &tilingData_->mmTile;
    params.qbmmParams = {tilingData_->mmTile.baseM, tilingData_->mmTile.baseN, tilingData_->mmTile.baseK,
                         tilingData_->mmTile.dbL0c};
    params.fragParams = {tileCnt_,
                         tileM_,
                         tailCnt_,
                         tailM_,
                         paddedTailM_,
                         commTurn_,
                         headRows_,
                         rankId_,
                         rankSize_,
                         m_,
                         static_cast<uint64_t>(k_),
                         static_cast<uint64_t>(n_),
                         scaleKLen_};
    params.aGM = aGM_;
    params.aScaleGM = aScaleGM_;
    params.bGM = bGM_;
    params.bScaleGM = bScaleGM_;
    params.cGM = cGM_;
    params.biasGM = biasGM_;
    params.isBias = (tilingData_->isBias != 0);
    params.winDataBase = GetWinDataRegionBase();
    params.winScaleBase = GetWinScaleRegionBase();
    params.dataBytesPerMRow = dataBytesPerMRow_;
    params.scaleBytesPerMRow = scaleBytesPerMRow_;
    params.cBytesPerM = cBytesPerM_;
    opStateDump_.DoDump(DUMP_FIELD_STEP, POS_COMP_CUBE_MATMUL_BEFORE, static_cast<uint8_t>(Utils::RT_PHASE_COMM_WAIT));
    quantMatmulKernelImpl_(params, opStateDump_);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline GM_ADDR AllGatherMxMatmulUrmaImpl<AType, BType, CType>::GetWinDataRegionBase()
{
    return reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[rankId_]);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline GM_ADDR AllGatherMxMatmulUrmaImpl<AType, BType, CType>::GetWinScaleRegionBase()
{
    return GetWinDataRegionBase() + dataRegionBytes_;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::AllGatherProcess()
{
    // 预触发 dependTileIdx=0：自身数据始终就绪，AIC 可直接消费。
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0);

    opStateDump_.DoDump(DUMP_FIELD_STEP, POS_COMM_BEFORE, static_cast<uint8_t>(Utils::RT_PHASE_COMM_COMMIT));
    for (uint32_t round = 0; round < commTurn_; ++round) {
        if (IsCommCore()) {
            allGatherScale_.Commit();
            allGatherData_.Commit();
            opStateDump_.DoDump(DUMP_FIELD_COMMIT);
            allGatherData_.template Wait<BARRIER_DEVICE>();
            opStateDump_.DoDump(DUMP_FIELD_WAIT);
        }
        if (IsCopyCore() && round > 0) {
            CopyDataFromWin(round - 1);
        }
        AscendC::SyncAll<true>();
        // round 0 对应 dependTileIdx=1，AIC 侧按 dependTileIdx 等待对应的远端 round 数据。
        CrossCoreSetFlag<0x2, PIPE_MTE3>(round + 1);
    }
    if (IsCopyCore()) {
        CopyDataFromWin(commTurn_ - 1);
    }
    AscendC::SyncAll<true>();
    allGatherScale_.Finalize();
    allGatherData_.Finalize();
    opStateDump_.DoDump(DUMP_FIELD_PHASE, 0, static_cast<uint8_t>(Utils::RT_PHASE_COMM_FINALIZE));
    // 通信接口使用了 channel 内的字段，需要刷新 GM 地址保证跨 kernel launch 的 GM 状态与 aiCore cache 一致
    // 后续通信接口会内置刷 cache 机制，届时可移除此处调用
    dcci(reinterpret_cast<__gm__ void *>(udmaCtx_->commBufferAddrs),
         static_cast<uint64_t>(CacheLine::ENTIRE_DATA_CACHE), static_cast<uint64_t>(DcciDst::CACHELINE_OUT));
}

template <typename AType, typename BType, typename CType>
__aicore__ inline bool AllGatherMxMatmulUrmaImpl<AType, BType, CType>::IsCommCore() const
{
    return static_cast<uint32_t>(GetBlockIdx()) >= copyCoreNum_;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline bool AllGatherMxMatmulUrmaImpl<AType, BType, CType>::IsCopyCore() const
{
    return static_cast<uint32_t>(GetBlockIdx()) < copyCoreNum_;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::CopyDataFromWin(uint32_t round)
{
    if (gatherOutGM_ == nullptr) {
        return;
    }

    uint64_t rowBytes = dataBytesPerMRow_;
    uint64_t rankStride = static_cast<uint64_t>(m_) * rowBytes;
    uint64_t roundStride = static_cast<uint64_t>(tileM_) * rowBytes;
    uint64_t roundOffset = static_cast<uint64_t>(round) * roundStride;
    uint32_t tileRows = (round < tileCnt_) ? tileM_ : tailM_;
    uint64_t roundTotalRows = static_cast<uint64_t>(rankSize_) * tileRows;

    uint64_t core = static_cast<uint64_t>(GetBlockIdx());
    uint64_t rowsPerCore = roundTotalRows / copyCoreNum_;
    uint64_t rem = roundTotalRows % copyCoreNum_;
    uint64_t rowStart = core * rowsPerCore + (core < rem ? core : rem);
    uint64_t rowEnd = rowStart + rowsPerCore + (core < rem ? 1 : 0);
    if (rowStart >= rowEnd) {
        return;
    }

    __gm__ int8_t *winBase = reinterpret_cast<__gm__ int8_t *>(GetWinDataRegionBase());
    __gm__ int8_t *outBase = reinterpret_cast<__gm__ int8_t *>(gatherOutGM_);

    uint64_t rankStart = rowStart / tileRows;
    uint64_t rankEnd = (rowEnd - 1) / tileRows;
    for (uint64_t r = rankStart; r <= rankEnd; ++r) {
        uint64_t rowInTileStart = (r == rankStart) ? (rowStart % tileRows) : 0;
        uint64_t rowInTileEnd = (r == rankEnd) ? ((rowEnd - 1) % tileRows + 1) : tileRows;
        uint64_t off = r * rankStride + roundOffset + rowInTileStart * rowBytes;
        uint64_t len = (rowInTileEnd - rowInTileStart) * rowBytes;
        CopyBytesPingPong(winBase + off, outBase + off, len);
    }
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::CopyBytesPingPong(__gm__ int8_t *src,
                                                                                         __gm__ int8_t *dst, uint64_t len)
{
    uint32_t chunk = COPY_CHUNK_BYTES;
    uint64_t done = 0;
    SetFlag<HardEvent::MTE3_MTE2>(COPY_EVT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(COPY_EVT_ID1);
    while (done < len) {
        uint32_t curLen = (len - done) > chunk ? chunk : static_cast<uint32_t>(len - done);
        uint32_t bufIdx = static_cast<uint32_t>((done / chunk) & 1);
        uint16_t evt = (bufIdx == 0) ? COPY_EVT_ID0 : COPY_EVT_ID1;
        uint32_t ubOff = copyUbOffset_ + bufIdx * chunk;
        uint32_t ubLen = (curLen + COPY_UB_ALIGN - 1) / COPY_UB_ALIGN * COPY_UB_ALIGN;

        auto ubTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::ub, int8_t>(ubOff),
                                             MakeLayoutByte{}(1, ubLen));
        auto srcTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(src + done),
                                              MakeLayoutByte{}(1, curLen));
        WaitFlag<HardEvent::MTE3_MTE2>(evt);
        asc::te::copy(copyGM2UB_, ubTensor, srcTensor);
        SetFlag<HardEvent::MTE2_MTE3>(evt);
        WaitFlag<HardEvent::MTE2_MTE3>(evt);
        auto dstTensor = asc::te::make_tensor(asc::te::make_mem_ptr<asc::te::location::gm>(dst + done),
                                              MakeLayoutByte{}(1, curLen));
        asc::te::copy(copyUB2GM_, dstTensor, ubTensor);
        SetFlag<HardEvent::MTE3_MTE2>(evt);
        done += curLen;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(COPY_EVT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(COPY_EVT_ID1);
}

} // namespace Apace

__global__ __aicore__ void AllGatherQuantMatmulKernel(__gm__ Apace::AivComm::CommContext *hcommCtx, GM_ADDR aGM,
                                                      GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
                                                      GM_ADDR biasGM, AllGatherMxMatmulUrmaTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    Apace::AllGatherMxMatmulUrmaImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t> impl;
    impl.Init(hcommCtx, aGM, aScaleGM, bGM, bScaleGM, cGM, biasGM, nullptr, nullptr, &tilingData);
    impl.Process();
}
