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
 * \file quant_batch_matmul_mx_kernel.h
 * \brief MatmulAllToAll MX 专用kernel: 逐rank遍历x2的N列, 输出rank-major
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"

#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define QBMM_MX_KERNEL_CLASS_TEM_PARAMS \
  template <class ProblemShape, class BlockMmad, class BlockScheduler>
#define QBMM_MX_KERNEL_FUNC_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler

using namespace AscendC;
using AscendC::Te::Get;

QBMM_MX_KERNEL_CLASS_TEM_PARAMS
class QuantBatchMatmulMxKernel {
public:
  __aicore__ inline QuantBatchMatmulMxKernel() {};
  __aicore__ inline ~QuantBatchMatmulMxKernel() {}

  using AType = typename BlockMmad::AType;
  using BType = typename BlockMmad::BType;
  using CType = typename BlockMmad::CType;
  using BiasType = typename BlockMmad::BiasType;
  using LayoutA = typename BlockMmad::LayoutA;
  using LayoutB = typename BlockMmad::LayoutB;
  using LayoutC = typename BlockMmad::LayoutC;
  using TypeScaleA = ::fp8_e8m0_t;
  using TypeScaleB = ::fp8_e8m0_t;

  static constexpr bool weightNz = BlockMmad::WEIGHT_NZ;
  static constexpr bool transA = BlockMmad::TRANS_A;
  static constexpr bool transB = BlockMmad::TRANS_B;
  static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
  static constexpr int64_t kCacheLineAlignMask = IsFp4<AType>() ? 0xff : 0x7f;
  static constexpr int32_t SCALE_C0 = 2;

  using BlockMmadParams = typename BlockMmad::Params;
  using L1Params = typename BlockMmad::L1Params;
  using BlockSchedulerParams = typename BlockScheduler::Params;

  using BlockShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;
  using BlockCoord = Te::Coord<int64_t, int64_t, int64_t, int64_t>;

  using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
  using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
  using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE>>;
  using MakeLayoutScaleA = Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<SCALE_C0>>;
  using MakeLayoutScaleB = Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<SCALE_C0>>;

  struct QBMMTiling {
    enum BiasMode : uint32_t {
      BIAS_DISABLED = 0,
      BIAS_ENABLED = 1
    };
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    BiasMode isBias;
    uint32_t dbL0c;
    bool disableCL2Cache;
  };

  struct LocalParams {
    uint64_t fullN;    // totalN, 用于layoutB
    uint64_t rankDim;  // rank数量
  };

  struct Params {
    ProblemShape problemShape;  // M=tileM, N=perRankN, K, batch=1
    BlockMmadParams mmadParams;
    L1Params l1Params;
    BlockSchedulerParams schParams;
    QBMMTiling qbmmParams;
    LocalParams localParams;
  };

  __aicore__ inline void InitMmad(const Params &params) {
    if ASCEND_IS_AIV { return; }
    isBias_ = (params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED);
    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    mmadOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias_,
                 params.qbmmParams.dbL0c > 1);
  }

  __aicore__ inline void operator()(const Params &params) {
    if ASCEND_IS_AIV { return; }

    int64_t M = Te::Get<MNK_M>(params.problemShape);   // tileM
    int64_t rankN = Te::Get<MNK_N>(params.problemShape); // perRankN
    int64_t K = Te::Get<MNK_K>(params.problemShape);
    int64_t fullN = static_cast<int64_t>(params.localParams.fullN);
    uint64_t rankDim = params.localParams.rankDim;
    if (rankDim == 0 || fullN != rankN * static_cast<int64_t>(rankDim)) { return; }

    int64_t scaleKLen = Blaze::Gemm::CeilDiv(K, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    auto layoutA = MakeLayoutA{}(M, K);
    auto layoutScaleA = MakeLayoutScaleA{}(M, scaleKLen);
    auto layoutB = MakeLayoutB{}(K, fullN);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, fullN);
    auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, fullN);
    auto layoutC = MakeLayoutC{}(M, rankN);

    __gm__ AType *aAddr = reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr);
    __gm__ BType *bAddr = reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr);
    __gm__ CType *cAddr = reinterpret_cast<__gm__ CType *>(params.mmadParams.cGmAddr);
    __gm__ BiasType *biasAddr = isBias_ ? reinterpret_cast<__gm__ BiasType *>(params.mmadParams.biasGmAddr) : nullptr;
    __gm__ TypeScaleA *scaleAAddr = reinterpret_cast<__gm__ TypeScaleA *>(params.mmadParams.scaleAGmAddr);
    __gm__ TypeScaleB *scaleBAddr = reinterpret_cast<__gm__ TypeScaleB *>(params.mmadParams.scaleBGmAddr);

    auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aAddr), layoutA);
    auto gmScaleA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleAAddr), layoutScaleA);
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bAddr), layoutB);
    auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBAddr), layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasAddr), layoutBias);

    for (uint32_t rankIdx = 0; rankIdx < rankDim; ++rankIdx) {
      // C输出rank-major: cAddr + rankIdx * M(tileM) * rankN
      __gm__ CType *rankCAddr = cAddr + static_cast<uint64_t>(rankIdx) * static_cast<uint64_t>(M) * static_cast<uint64_t>(rankN);
      auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(rankCAddr), layoutC);

      // 本rank在x2中的N偏移
      int64_t fullNPos = static_cast<int64_t>(rankIdx) * rankN;

      BlockScheduler bs(params.problemShape, params.schParams);
      BlockCoord blockIdx;
      int64_t mPos = 0L, nPos = 0L;
      constexpr int64_t kPos = 0L;

      while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape =
          bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
        if (Te::Get<IDX_M_TILEIDX>(singleShape) <= 0 || Te::Get<IDX_N_TILEIDX>(singleShape) <= 0) { return; }

        bs.GetTileCoord(blockIdx, mPos, nPos);
        auto gmBlockA = gmA.Slice(Te::MakeCoord(mPos, kPos),
          Te::MakeShape(Te::Get<IDX_M_TILEIDX>(singleShape), K));
        auto gmBlockScaleA = gmScaleA.Slice(Te::MakeCoord(mPos, kPos),
          Te::MakeShape(Te::Get<IDX_M_TILEIDX>(singleShape), scaleKLen));
        auto gmBlockB = gmB.Slice(Te::MakeCoord(kPos, fullNPos + nPos),
          Te::MakeShape(K, Te::Get<IDX_N_TILEIDX>(singleShape)));
        auto gmBlockScaleB = gmScaleB.Slice(Te::MakeCoord(kPos, fullNPos + nPos),
          Te::MakeShape(scaleKLen, Te::Get<IDX_N_TILEIDX>(singleShape)));
        auto gmBlockBias = gmBias.Slice(Te::MakeCoord(0L, fullNPos + nPos),
          Te::MakeShape(1L, Te::Get<IDX_N_TILEIDX>(singleShape)));
        auto gmBlockC = gmC.Slice(Te::MakeCoord(mPos, nPos),
          Te::MakeShape(Te::Get<IDX_M_TILEIDX>(singleShape), Te::Get<IDX_N_TILEIDX>(singleShape)));
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
      }
    }
  }

private:
  BlockMmad mmadOp_;
  bool isBias_{false};
};

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
