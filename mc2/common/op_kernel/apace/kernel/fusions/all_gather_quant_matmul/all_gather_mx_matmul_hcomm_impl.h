/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_mx_matmul_hcomm_impl.h
 * \brief AllGather + QuantMatmul 融合实现（Hcomm/CCU 纯 C 核通信 + BlockMmad 直调计算）
 *
 * 计算侧复用 Blaze::Gemm::Kernel::AllGatherQbmmMxBlockMmadKernel（直接调用 ops-tensor 的
 * BlockMmad，与 all_to_all_mx_quant_matmul_hcomm_impl.h 同构），不经过 apace FragmentTensor。
 *
 * 通信状态由 HcommCommState 容器存储，AllGatherMxQuantMatmulHcommImpl 直接持有；
 * 等待策略由 HcommCommWaitPolicy 承担，通过 state_ 指针引用 HcommCommState，
 * kernel 内逐 tile 通过 commPolicy_.WaitTile() 等待通信完成。
 *
 * Init():
 *   AIC: hccl_.InitV2()/SetCcTilingV2()，批量下发 AllGather<true> 通信任务：
 *        - scale：全量一次性提交（repeat=1），gather buffer 上拼成 [rankSize*m, scaleKLen]；
 *        - dataHead：repeat=tileCnt，每轮搬 tileM 行，拼成各 rank 块内的前 headRows 行；
 *        - dataTail：repeat=tailCnt（通常为 1），搬尾块行；
 *        最终 gatherData buffer 为连续的 [rankSize*m, kForComm*bytes] 大矩阵。
 *        最后通过 GetCommPolicy().state_ = &commState_ 将 WaitPolicy 绑定到通信状态。
 *
 * Run():
 *   AIC: 执行 BlockMmad kernel（gather buffer 上按调度器逐 tile 计算 + per-tile wait）
 *        → SyncAll → hccl_.Finalize()。
 *
 * 等待语义（waitIdx 单调递增，与 kernel 的 CalcDependTileIdx 约定一致）：
 *   waitIdx == 0        : 本地数据，无需等待（保留）；
 *   waitIdx == 1        : 首次远端等待，先等 scale（全量）再等 dataHead 第 0 轮；
 *   waitIdx <= tileCnt  : 等 dataHead（repeat=tileCnt，每次 Wait 消费一轮）；
 *   waitIdx >  tileCnt  : 等 dataTail（repeat=tailCnt，每次 Wait 消费一轮）。
 *
 * \note 本实现仅启动 AIC 核（cube-only）。若需启动 AIV 核，须为 Init/Run 中的通信下发、
 *       Wait、Finalize 及 SyncAll 增加 AIC 守卫（if ASCEND_IS_AIC），否则 AIV 误参与 HCCL
 *       调用将导致 Prepare/Wait 失配或死锁。
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#define WINDOW_LEN 1L // 调度器窗口设置为1，非侵入式修改
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#undef WINDOW_LEN
#include "../../matmul/quant_batch_matmul/all_gather_qbmm_mx_blockmmad_kernel.h"
#include "all_gather_mx_matmul_hcomm_tiling_data.h"
#include "include/tensor_api/tensor.h"

namespace Apace {

template <typename T>
constexpr uint64_t GetAgHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return AscendC::HCCL_DATA_TYPE_UINT8;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP8E4M3;
    }
}

template <AscendC::HcclServerType ServerType>
struct HcommCommState {
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle scaleHandle_{0};
    AscendC::HcclHandle dataHeadHandle_{0};
    AscendC::HcclHandle dataTailHandle_{0};
    uint32_t tileCnt_{0};
};

template <AscendC::HcclServerType ServerType>
struct HcommCommWaitPolicy {
    HcommCommState<ServerType> *state_{nullptr};

    __aicore__ inline void WaitTile(uint32_t tileIdx)
    {
        // waitIdx=0 为本地数据，无需等待通信
        if (tileIdx == 0) {
            return;
        }
        // waitIdx=1: 首次远端等待，先等 scale（全量），掩盖 matmul 头开销
        if (tileIdx == 1) {
            state_->hccl_.Wait(state_->scaleHandle_);
        }
        // waitIdx=1..tileCnt (HEAD): 等 data head（repeat=tileCnt，每次 Wait 消费一轮）
        if (tileIdx <= state_->tileCnt_) {
            state_->hccl_.Wait(state_->dataHeadHandle_);
        } else {
            // waitIdx>tileCnt (TAIL): 等 data tail
            state_->hccl_.Wait(state_->dataTailHandle_);
        }
    }
};

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool IsBTrans = false>
class AllGatherMxQuantMatmulHcommImpl {
public:
    explicit __aicore__ inline AllGatherMxQuantMatmulHcommImpl(Apace::hcommAllGatherMatmulTilingData *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR biasGM,
                                GM_ADDR cGM, GM_ADDR gatherOut, GM_ADDR workspaceGM);
    __aicore__ inline void Run();

    // Layout 定义
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB =
        typename AscendC::Std::conditional_t<IsBTrans, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    // 组件定义
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, AType>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
    using QuantMatmulKernelImpl = Blaze::Gemm::Kernel::AllGatherQbmmMxBlockMmadKernel<
        ProblemShape, BlockMmad, BlockScheduler, HcommCommWaitPolicy<ServerType>>;

    // 参数类型
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;
    HcommCommState<ServerType> commState_;

private:
    __aicore__ inline void InitBaseParams();
    __aicore__ inline void CommitAllGather();
    __aicore__ inline void SetupKernelParams(Params &params);

    Apace::hcommAllGatherMatmulTilingData *tilingData_;

    GM_ADDR aGM_{};
    GM_ADDR aScaleGM_{};
    GM_ADDR bGM_{};
    GM_ADDR bScaleGM_{};
    GM_ADDR biasGM_{};
    GM_ADDR cGM_{};
    GM_ADDR workspaceGM_{};
    GM_ADDR gatherDataAddr_{};
    GM_ADDR gatherScaleAddr_{};

    uint32_t rankId_{};
    uint32_t rankSize_{};
    uint32_t m_{};          // 每 rank 的 M 行数
    uint32_t k_{};
    uint32_t n_{};
    uint32_t tileCnt_{};    // dataHead 通信轮数
    uint32_t tileM_{};      // dataHead 单轮行数
    uint32_t tailCnt_{};    // dataTail 通信轮数
    uint32_t tailM_{};      // dataTail 单轮行数
    uint32_t commTurn_{};   // 总通信轮数 = tileCnt + tailCnt
    uint64_t headRows_{};   // dataHead 覆盖行数 = tileCnt * tileM
    uint64_t scaleKLen_{};  // 每 scale 行长度 = CeilDiv(K, 64) * 2
    uint64_t kForComm_{};   // 通信用 K 元素数（FP4 时 = CeilDiv(K, 2)，即打包后字节）
    uint64_t dataBytesPerMRow_{}; // 每 M 行数据字节数
    uint64_t scaleBytesPerMRow_{}; // 每 M 行 scale 字节数
    uint64_t cBytesPerM_{};
    uint64_t scaleRegionBytes_{};
    uint64_t dataRegionBytes_{};
    bool isBias_{false};

    static constexpr uint64_t mxScaleHcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E8M0;
    static constexpr uint64_t dataHcclDataType_ = GetAgHcclDataType<AType>();
    static constexpr uint64_t MXFP_DATA_NUM_PER_BYTE = 2UL;
};

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool IsBTrans>
__aicore__ inline void
AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4, IsBTrans>::InitBaseParams()
{
    const auto &ct = tilingData_->commTile;
    tileCnt_ = static_cast<uint32_t>(ct.splitAxisTileCnt);
    tileM_ = static_cast<uint32_t>(ct.splitAxisTileSize);
    tailCnt_ = static_cast<uint32_t>(ct.splitAxisTailCnt);
    tailM_ = static_cast<uint32_t>(ct.splitAxisTailSize);
    k_ = tilingData_->mmTile.k;
    n_ = tilingData_->mmTile.n;
    m_ = static_cast<uint32_t>(ct.splitAxisTileSize * ct.splitAxisTileCnt +
                               ct.splitAxisTailSize * ct.splitAxisTailCnt);
    commTurn_ = tileCnt_ + tailCnt_;
    headRows_ = static_cast<uint64_t>(tileCnt_) * tileM_;

    const uint64_t kFull = static_cast<uint64_t>(k_);
    scaleKLen_ = Blaze::Gemm::CeilDiv(kFull, static_cast<uint64_t>(Blaze::Gemm::MXFP_DIVISOR_SIZE)) *
                 static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE);
    // FP4: 2 个元素打包为 1 字节，通信 count/stride 及行字节偏移均需用 CeilDiv(K, 2) 折半
    kForComm_ = kFull;
    if constexpr (IsMxFp4) {
        kForComm_ = Blaze::Gemm::CeilDiv(kForComm_, MXFP_DATA_NUM_PER_BYTE);
    }
    dataBytesPerMRow_ = kForComm_ * sizeof(AType);
    scaleBytesPerMRow_ = scaleKLen_ * sizeof(AscendC::fp8_e8m0_t);
    cBytesPerM_ = static_cast<uint64_t>(n_) * sizeof(CType);
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool IsBTrans>
__aicore__ inline void
AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4, IsBTrans>::Init(
    GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut,
    GM_ADDR workspaceGM)
{
    aGM_ = aGM;
    aScaleGM_ = aScaleGM;
    bGM_ = bGM;
    bScaleGM_ = bScaleGM;
    biasGM_ = biasGM;
    isBias_ = (biasGM != nullptr);
    cGM_ = cGM;
    workspaceGM_ = workspaceGM;

    InitBaseParams();

    // HCCL initialization
    commState_.hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    commState_.hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(Apace::hcommAllGatherMatmulTilingData, mc2CcTiling)));
    rankId_ = commState_.hccl_.GetRankId();
    rankSize_ = commState_.hccl_.GetRankDim();

    // Gather buffer: scale 始终用 workspace; data 用 gatherOut（当可用）或 workspace
    scaleRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * scaleBytesPerMRow_;
    dataRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * dataBytesPerMRow_;
    gatherScaleAddr_ = workspaceGM_;
    if (tilingData_->gatherLen == 0 && gatherOut != nullptr) {
        gatherDataAddr_ = gatherOut;
    } else {
        gatherDataAddr_ = workspaceGM_ + scaleRegionBytes_;
    }

    // Submit AllGather (non-blocking <true>)
    CommitAllGather();

    // Bind CommPolicy to CommState
    quantMatmulKernelImpl_.GetCommPolicy().state_ = &commState_;
    commState_.tileCnt_ = tileCnt_;
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool IsBTrans>
__aicore__ inline void
AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4, IsBTrans>::CommitAllGather()
{
    // Scale: 全量一次性提交（repeat=1）。
    // 语义：rank r 的 scale 落在 dst + r * strideCount 处，拼成 [rankSize*m, scaleKLen] 连续布局
    commState_.scaleHandle_ =
        commState_.hccl_.template AllGather<true>(aScaleGM_, gatherScaleAddr_,
                                                  static_cast<uint64_t>(m_) * scaleKLen_,
                                                  static_cast<AscendC::HcclDataType>(mxScaleHcclDataType_),
                                                  static_cast<uint64_t>(m_) * scaleKLen_, 1);

    // Data Head: 批量提交所有 head tile（repeat=tileCnt）。
    // task i: src = aGM + i*tileM*kForComm, rank r 的数据落在 dst + i*tileM*kForComm + r*m*kForComm，
    // 即 gather buffer 中 rank r 块内的第 [i*tileM, (i+1)*tileM) 行
    if (tileCnt_ > 0) {
        commState_.dataHeadHandle_ = commState_.hccl_.template AllGather<true>(
            aGM_, gatherDataAddr_, static_cast<uint64_t>(tileM_) * kForComm_,
            static_cast<AscendC::HcclDataType>(dataHcclDataType_),
            static_cast<uint64_t>(m_) * kForComm_, static_cast<uint8_t>(tileCnt_));
    }

    // Data Tail: 批量提交所有 tail tile（repeat=tailCnt）。
    // 尾块地址偏移按字节计算：headRows * dataBytesPerMRow（FP4 时已折半，避免 2x 偏移错误）
    if (tailCnt_ > 0) {
        uint64_t headOffset = headRows_ * dataBytesPerMRow_;
        commState_.dataTailHandle_ = commState_.hccl_.template AllGather<true>(
            aGM_ + headOffset, gatherDataAddr_ + headOffset, static_cast<uint64_t>(tailM_) * kForComm_,
            static_cast<AscendC::HcclDataType>(dataHcclDataType_),
            static_cast<uint64_t>(m_) * kForComm_, static_cast<uint8_t>(tailCnt_));
    }
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool IsBTrans>
__aicore__ inline void
AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4, IsBTrans>::SetupKernelParams(
    Params &params)
{
    const auto &mmTile = tilingData_->mmTile;
    const uint64_t totalM = static_cast<uint64_t>(rankSize_) * m_;

    ProblemShape problemShape{static_cast<int64_t>(totalM), mmTile.n, mmTile.k, 1};

    BlockMmadParams mmadParams;
    mmadParams.aGmAddr = gatherDataAddr_;
    mmadParams.scaleAGmAddr = gatherScaleAddr_;
    mmadParams.bGmAddr = bGM_;
    mmadParams.scaleBGmAddr = bScaleGM_;
    mmadParams.cGmAddr = cGM_;
    mmadParams.biasGmAddr = isBias_ ? biasGM_ : nullptr;

    L1Params l1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};

    BlockSchedulerParams schedulerParams{
        mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
        mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};

    QBMMTiling qbmmParams{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c,
                          static_cast<uint32_t>(isBias_ ? 1 : 0)};

    LocalParams localParams{rankSize_, m_, tileM_, tileCnt_, tailM_, tailCnt_, commTurn_};

    params = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool IsBTrans>
__aicore__ inline void AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4, IsBTrans>::Run()
{
    Params params;
    SetupKernelParams(params);
    quantMatmulKernelImpl_(params);
    AscendC::SyncAll();
    commState_.hccl_.Finalize();
}

} // namespace Apace
