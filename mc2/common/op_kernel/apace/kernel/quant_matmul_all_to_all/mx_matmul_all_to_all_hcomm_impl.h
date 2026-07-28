/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file matmul_all_to_all_mx_impl.h
 * \brief MatmulAllToAll MX 算子 kernel 实现（先计算后通信，拉 C 方向）
 *
 * 核心设计：
 * - 不使用 GemmKernel 包装，直接组合 QuantBatchMatmulKernel + AIV 通信
 * - AIC 负责计算：委托给 QuantBatchMatmulKernel，遍历所有 rank 的 B 段做 QBMM，结果写 Win 区
 * - AIV 负责通信：通过 AllToAll GET 从其他卡的 Win 区拉回本 rank 的 C 段到 cGM
 * - 流水线：bufferCount 个槽位环形复用，AIC 逐 tile 产出，AIV 逐 tile 消费
 * - LocalDelay=true：本 rank C 段后置计算直接写 cGM，可能与 AIV GET 并行
 * - Scale 全在本卡，不参与通信
 *
 * 本文件只负责整体编排与通信，matmul 调度逻辑见 mx_matmul_all_to_all_mm_dispatch.h
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_all_to_all_tiling_data.h"
#include "adv_api/hcomm/hcomm.h"

#include "qbmm_kernel.h"
#include "../../block/aiv_comm/collective_comm_api.h"
#include "../../block/aiv_comm/collective_comm_context.h"
#include "../../tiling/comm_tiling_data.h"
#include "../../block/aiv_comm/barrier/barrier_ubmem.h"

namespace MatmulAllToAllMxImpl {

using namespace AscendC;
using namespace Apace::AivComm;

template <typename AType, typename BType, typename CType, bool LocalDelay>
class MatmulAllToAllMxImpl {
public:
  __aicore__ inline MatmulAllToAllMxImpl() {};
  __aicore__ inline ~MatmulAllToAllMxImpl() {}

  __aicore__ inline void Init(__gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR aScaleGM,
    GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM, const MatmulAllToAllTilingData *tilingData,
    TPipe* pipe);
  __aicore__ inline void Run();

private:
  __aicore__ inline void AllToAllProcess();

  QuantBatchMatmulKernel<AType, BType, CType, LocalDelay> qbmm_;

  __gm__ CommContext *hcommCtx_{};
  __gm__ CommUbmemContext* barrierCtx_{};
  __gm__ CommUdmaContext* udmaCtx_{};
  TPipe* pipe_{};

  CommTilingData commTilingData_;
  CollectiveComm<CommCollectiveOp::AllToAll, CommMode::GET, CType, TeamBarrier, CommAddrLayout::DEFAULT> allToAll_;

  uint64_t ubOffset_{0};
  uint32_t rankSize_{0};
  TeamBarrier teamBarrier_;
};

// =================================================================================
// 公有方法实现
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::Init(
  __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR aScaleGM,
  GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM, const MatmulAllToAllTilingData *tilingData,
  TPipe* pipe)
{
  hcommCtx_ = hcommCtx;
  pipe_ = pipe;

  barrierCtx_ = &(hcommCtx_->ubmemCtx);
  udmaCtx_ = &(hcommCtx_->udmaCtx);
  rankSize_ = udmaCtx_->rankSize;

  qbmm_.Init(udmaCtx_->rankId, udmaCtx_->rankSize,
    aGM, aScaleGM, bGM, bScaleGM, cGM, tilingData);

  BaseParams& base = qbmm_.GetBase();
  base.selfCBase = base.cGm + static_cast<uint64_t>(base.rankId) * base.axisM * base.cByteStrideN;
  base.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[base.rankId]);
  base.winGm = base.selfWinAddr;

  commTilingData_ = tilingData->alltoallTilingData.commTilingData;

  auto commPtr = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset_);
  ubOffset_ += COMM_WORKSPACE_SIZE;
  auto barrierPtr = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(ubOffset_);
  teamBarrier_.Init(barrierPtr.Get(), barrierCtx_, udmaCtx_->rankSize, static_cast<uint32_t>(GetBlockIdx()));
  ubOffset_ += UB_SIZE;

  allToAll_.Init(udmaCtx_, teamBarrier_, commTilingData_, cGM, commPtr.Get(),
    static_cast<uint32_t>(udmaCtx_->rankSize), static_cast<uint32_t>(GetBlockIdx()));
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::Run()
{
  if ASCEND_IS_AIC {
    qbmm_.MatmulProcess();
    if constexpr (LocalDelay) {
      AscendC::PipeBarrier<PIPE_ALL>();
      qbmm_.LocalCompute();
    }
  }
  if ASCEND_IS_AIV {
    AllToAllProcess();
  }
}

// =================================================================================
// 通信实现 (AIV 侧)
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void MatmulAllToAllMxImpl<AType, BType, CType, LocalDelay>::AllToAllProcess()
{
  const auto& base = qbmm_.GetBase();
  for (uint32_t tid = 0; tid < base.totalTileCnt; ++tid) {
    CrossCoreWaitFlag<0x2, PIPE_S>(tid);
    SyncAll<true>();
    if (GetBlockIdx() < rankSize_) {
      allToAll_.Commit();
      allToAll_.Wait(true);
    }
    if (base.bufferCount < base.totalTileCnt && tid < base.totalTileCnt - base.bufferCount) {
      CrossCoreSetFlag<0x2, PIPE_MTE3>(tid);
    }
  }
  if (GetBlockIdx() < rankSize_) {
    allToAll_.Finalize();
  }
}

// =================================================================================
// Kernel 入口（4 种 fp8 类型组合，统一 TransA=false, TransB=true, 输出 bfloat16_t）
// =================================================================================

template <bool LocalDelay>
__global__ __aicore__ void MatmulAllToAllMxKernelE4M3E4M3(
  __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR aScaleGM,
  GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM, MatmulAllToAllTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  TPipe pipe;
  MatmulAllToAllMxImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, LocalDelay> impl;
  impl.Init(hcommCtx, aGM, aScaleGM, bGM, bScaleGM, cGM, &tilingData, &pipe);
  impl.Run();
}

template <bool LocalDelay>
__global__ __aicore__ void MatmulAllToAllMxKernelE5M2E5M2(
  __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR aScaleGM,
  GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM, MatmulAllToAllTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  TPipe pipe;
  MatmulAllToAllMxImpl<fp8_e5m2_t, fp8_e5m2_t, bfloat16_t, LocalDelay> impl;
  impl.Init(hcommCtx, aGM, aScaleGM, bGM, bScaleGM, cGM, &tilingData, &pipe);
  impl.Run();
}

template <bool LocalDelay>
__global__ __aicore__ void MatmulAllToAllMxKernelE4M3E5M2(
  __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR aScaleGM,
  GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM, MatmulAllToAllTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  TPipe pipe;
  MatmulAllToAllMxImpl<fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, LocalDelay> impl;
  impl.Init(hcommCtx, aGM, aScaleGM, bGM, bScaleGM, cGM, &tilingData, &pipe);
  impl.Run();
}

template <bool LocalDelay>
__global__ __aicore__ void MatmulAllToAllMxKernelE5M2E4M3(
  __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR aScaleGM,
  GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM, MatmulAllToAllTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  TPipe pipe;
  MatmulAllToAllMxImpl<fp8_e5m2_t, fp8_e4m3fn_t, bfloat16_t, LocalDelay> impl;
  impl.Init(hcommCtx, aGM, aScaleGM, bGM, bScaleGM, cGM, &tilingData, &pipe);
  impl.Run();
}

} // namespace MatmulAllToAllMxImpl
