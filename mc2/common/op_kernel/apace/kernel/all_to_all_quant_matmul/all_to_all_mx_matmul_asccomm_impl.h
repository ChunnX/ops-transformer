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
 * \file all_to_all_mx_matmul_asccomm_impl.h
 * \brief AlltoAll MX Quant Matmul — 通信+计算融合实现
 *
 * Run():
 *   AIV: RunAllToAll() — Wait 已在 Init 中批量下发的 AlltoAll<true> 通信任务
 *   AIC: MatmulProcess()       ->  SetupComputeContext(i) + ExecuteGemm(i)
 *
 * AIV side: Init 中通过 AlltoAll<true> + repeat 一次下发所有通信任务，Run 阶段仅 Wait + 同步.
 * AIC side: pure computation, directly invokes BlockMmad with inlined scheduling.
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "apace/block/blaze_ext/gemm/policy/dispatch_policy_alltoall.h"
#include "apace/block/blaze_ext/gemm/block/block_mmad_qbmm_mx_alltoall.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "tensor_api/tensor.h"
#include "apace/utils/qbmm_tiling_data.h"

namespace Apace {

__aicore__ inline uint64_t CeilDivU64(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
class AllToAllMxQuantMatmulImpl {
private:
  using LayoutA = AscendC::Te::NDExtLayoutPtn;
  using LayoutB = AscendC::Te::DNExtLayoutPtn;
  using LayoutC = AscendC::Te::NDExtLayoutPtn;
  using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
  using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
  using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

  using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMxAlltoAll<0, false>;
  using BlockScheduler =
      Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, X1Type>;
  using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC,
                                                   float, LayoutC>;

  using L1Params = typename BlockMmad::L1Params;
  using SchParams = typename BlockScheduler::Params;

  static constexpr int64_t C0_SIZE = Blaze::Gemm::IsFp4<X1Type>() ? Blaze::Gemm::C0_SIZE_B4 : Blaze::Gemm::C0_SIZE_B8;
  static constexpr int64_t kCacheLineAlignMask = Blaze::Gemm::IsFp4<X1Type>() ? 0xff : 0x7f;
  static constexpr int32_t SCALE_C0 = 2;

  using MakeLayoutB = AscendC::Te::FrameLayoutFormat<LayoutB, AscendC::Std::Int<C0_SIZE>>;
  using MakeLayoutC = AscendC::Te::FrameLayoutFormat<LayoutC, AscendC::Std::Int<AscendC::AuxGetC0Size<YType>()>>;
  using MakeLayoutScaleB = AscendC::Te::FrameLayoutFormat<AscendC::Te::ScaleBNDLayoutPtn, AscendC::Std::Int<SCALE_C0>>;

public:
  __aicore__ inline AllToAllMxQuantMatmulImpl(AlltoAllMatmulTilingDataType *tilingData,
                                               AscendC::TPipe *tPipe, uint64_t x1HcclDataType)
      : tilingData_(tilingData), tPipe_(tPipe),
        x1HcclDataType_(x1HcclDataType){};

  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
                               GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR workspaceGM);
  __aicore__ inline void Run();

private:
  static constexpr AscendC::HcclServerType kServerType = AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU;
  AscendC::Hccl<kServerType> hccl_;
  bool notifyFlag_{false};
  AscendC::HcclHandle scaleHandle_{0};
  AscendC::HcclHandle dataHeadHandle_{0};
  AscendC::HcclHandle dataTailHandle_{0};

  AlltoAllMatmulTilingDataType *tilingData_;
  AscendC::TPipe *tPipe_;

  GM_ADDR x1_;
  GM_ADDR x2_;
  GM_ADDR y_;
  GM_ADDR bias_;
  GM_ADDR x1Scale_;
  GM_ADDR x2Scale_;
  GM_ADDR workspaceGM_;
  GM_ADDR commX1ScaleGM1_;
  GM_ADDR commOutGM_;

  uint64_t mxScaleHcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E8M0;
  uint64_t x1HcclDataType_ = AscendC::HCCL_DATA_TYPE_FP8E4M3;

  uint64_t rankForComm_{0};
  uint32_t scaleKPerRank_{0};

  static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
  static constexpr uint64_t NUM_TWO = 2UL;
  static constexpr uint64_t ALIGN_NUM = 512UL;

  struct BaseParams {
    struct BaseGmAddrs {
      GM_ADDR aGM;
      GM_ADDR bGM;
      GM_ADDR cGM;
      uint64_t aOffset;
      uint64_t bOffset;
      uint64_t cOffset;
    } baseData;

    struct AdditionalData {
      GM_ADDR x1Scale;
      GM_ADDR x2Scale;
      uint64_t x1ScaleOffset;
      uint32_t rankDim;
      uint32_t rankN;
      uint32_t rankM;
      uint32_t scaleKPerRank;
    } additionalData;

    QuantBatchMatmulTilingData *tilingDataPtr;
  } computeContext_;

  __aicore__ inline void RunAllToAll();
  __aicore__ inline void MatmulProcess();
  __aicore__ inline void SetupComputeContext(uint32_t loopIdx);
  __aicore__ inline void ExecuteGemm(uint32_t taskIndex);
};

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AllToAllMxQuantMatmulImpl<X1Type, X2Type, YType, CommDataTypeX1,
                           AlltoAllMatmulTilingDataType, IsMxFp4>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR x1_scale, GM_ADDR x2_scale,
    GM_ADDR workspaceGM)
{
  auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
  x1_ = x1;
  x2_ = x2;
  y_ = y;
  bias_ = bias;
  x1Scale_ = x1_scale;
  x2Scale_ = x2_scale;
  workspaceGM_ = workspaceGM;
  commX1ScaleGM1_ = workspaceGM;
  uint64_t x1ScaleLen = CeilDivU64(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO *
                        static_cast<uint64_t>(mc2Tiling_.rankM) * sizeof(AscendC::fp8_e8m0_t);
  x1ScaleLen = CeilDivU64(x1ScaleLen, ALIGN_NUM) * ALIGN_NUM;
  commOutGM_ = workspaceGM + x1ScaleLen;

  if constexpr (IsMxFp4) {
    rankForComm_ = CeilDivU64(mc2Tiling_.rankK, 2) / mc2Tiling_.rankDim;
  } else {
    rankForComm_ = mc2Tiling_.rankK / mc2Tiling_.rankDim;
  }
  scaleKPerRank_ = static_cast<uint32_t>(
      CeilDivU64(mc2Tiling_.rankK / mc2Tiling_.rankDim, MXFP_GROUP_SIZE) * NUM_TWO);

  notifyFlag_ = false;
  if ASCEND_IS_AIV {
    if (AscendC::GetBlockIdx() == 0) {
      notifyFlag_ = true;
    }
  }
  hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
  hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(AlltoAllMatmulTilingDataType, mc2CcTiling)));

  if (notifyFlag_ && (mc2Tiling_.tileCnt > 0 || mc2Tiling_.tailCnt > 0)) {
    uint64_t dataStrideCount = static_cast<uint64_t>(mc2Tiling_.rankM) * rankForComm_;

    uint64_t scaleSendCount = static_cast<uint64_t>(mc2Tiling_.rankM) * scaleKPerRank_;
    scaleHandle_ = hccl_.template AlltoAll<true>(
        x1Scale_, commX1ScaleGM1_, scaleSendCount,
        static_cast<AscendC::HcclDataType>(mxScaleHcclDataType_),
        scaleSendCount, 1);

    if (mc2Tiling_.tileCnt > 0) {
      uint64_t headSendCount = static_cast<uint64_t>(mc2Tiling_.tileM) * rankForComm_;
      dataHeadHandle_ = hccl_.template AlltoAll<true>(
          x1_, commOutGM_, headSendCount,
          static_cast<AscendC::HcclDataType>(x1HcclDataType_),
          dataStrideCount, static_cast<uint8_t>(mc2Tiling_.tileCnt));
    }

    if (mc2Tiling_.tailCnt > 0) {
      uint64_t headOffset = static_cast<uint64_t>(mc2Tiling_.tileCnt) *
                            mc2Tiling_.tileM * rankForComm_ * sizeof(X1Type);
      uint64_t tailSendCount = static_cast<uint64_t>(mc2Tiling_.tailM) * rankForComm_;
      dataTailHandle_ = hccl_.template AlltoAll<true>(
          x1_ + headOffset, commOutGM_ + headOffset, tailSendCount,
          static_cast<AscendC::HcclDataType>(x1HcclDataType_),
          dataStrideCount, static_cast<uint8_t>(mc2Tiling_.tailCnt));
    }
  }
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AllToAllMxQuantMatmulImpl<X1Type, X2Type, YType, CommDataTypeX1,
                           AlltoAllMatmulTilingDataType, IsMxFp4>::Run()
{
  if (tilingData_ == nullptr) {
    return;
  }
  auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
  if (mc2Tiling_.rankDim == 0) {
    return;
  }
  if (mc2Tiling_.tileCnt == 0 && mc2Tiling_.tailCnt == 0) {
    return;
  }

  if ASCEND_IS_AIV {
    RunAllToAll();
  }

  if ASCEND_IS_AIC {
    MatmulProcess();
  }

  if (notifyFlag_) {
    hccl_.Finalize();
  }
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AllToAllMxQuantMatmulImpl<X1Type, X2Type, YType, CommDataTypeX1,
                           AlltoAllMatmulTilingDataType, IsMxFp4>::RunAllToAll()
{
  auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;

  if (notifyFlag_) {
    hccl_.Wait(scaleHandle_);
  }

  for (uint32_t i = 0; i < mc2Tiling_.tileCnt; ++i) {
    if (notifyFlag_) {
      hccl_.Wait(dataHeadHandle_);
    }
    AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(8);
    AscendC::CrossCoreWaitFlag(8);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
  }

  for (uint32_t i = 0; i < mc2Tiling_.tailCnt; ++i) {
    if (notifyFlag_) {
      hccl_.Wait(dataTailHandle_);
    }
    AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(8);
    AscendC::CrossCoreWaitFlag(8);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
  }
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AllToAllMxQuantMatmulImpl<X1Type, X2Type, YType, CommDataTypeX1,
                           AlltoAllMatmulTilingDataType, IsMxFp4>::MatmulProcess()
{
  auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
  uint32_t totalTileCnt = mc2Tiling_.tileCnt + mc2Tiling_.tailCnt;

  for (uint32_t i = 0; i < totalTileCnt; ++i) {
    AscendC::CrossCoreWaitFlag(9);
    SetupComputeContext(i);
    uint32_t localIdx = (i < mc2Tiling_.tileCnt) ? i : (i - mc2Tiling_.tileCnt);
    ExecuteGemm(localIdx);
  }
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AllToAllMxQuantMatmulImpl<X1Type, X2Type, YType, CommDataTypeX1,
                           AlltoAllMatmulTilingDataType, IsMxFp4>::SetupComputeContext(uint32_t loopIdx)
{
  auto &&mc2Tiling_ = tilingData_->alltoAllQuantMatmulTilingInfo;
  bool isHead = (loopIdx < mc2Tiling_.tileCnt);
  uint32_t curM = isHead ? mc2Tiling_.tileM : mc2Tiling_.tailM;
  uint64_t headRows = static_cast<uint64_t>(mc2Tiling_.tileCnt) * mc2Tiling_.tileM;
  uint64_t rankForComm = rankForComm_;
  uint32_t scaleKPerRank = scaleKPerRank_;

  auto *ctx = &computeContext_;
  if (isHead) {
    ctx->baseData.aGM = commOutGM_;
    ctx->baseData.cGM = y_;
    ctx->additionalData.x1Scale = commX1ScaleGM1_;
  } else {
    ctx->baseData.aGM = commOutGM_ + headRows * rankForComm * sizeof(X1Type);
    ctx->baseData.cGM = y_ + headRows * mc2Tiling_.rankN * sizeof(YType);
    ctx->additionalData.x1Scale =
        commX1ScaleGM1_ + headRows * scaleKPerRank * sizeof(AscendC::fp8_e8m0_t);
  }
  ctx->baseData.bGM = x2_;
  ctx->baseData.aOffset = static_cast<uint64_t>(curM) * rankForComm * sizeof(X1Type);
  ctx->baseData.bOffset = 0UL;
  ctx->baseData.cOffset = static_cast<uint64_t>(curM) * mc2Tiling_.rankN * sizeof(YType);
  ctx->additionalData.x1ScaleOffset = static_cast<uint64_t>(curM) * scaleKPerRank * sizeof(AscendC::fp8_e8m0_t);
  ctx->additionalData.x2Scale = x2Scale_;
  ctx->additionalData.rankDim = mc2Tiling_.rankDim;
  ctx->additionalData.rankN = mc2Tiling_.rankN;
  ctx->additionalData.rankM = mc2Tiling_.rankM;
  ctx->additionalData.scaleKPerRank = scaleKPerRank;
  ctx->tilingDataPtr = isHead ? &(tilingData_->blazeMmTileTilingData) : &(tilingData_->blazeMmTailTilingData);
}

template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, bool IsMxFp4>
__aicore__ inline void
AllToAllMxQuantMatmulImpl<X1Type, X2Type, YType, CommDataTypeX1,
                           AlltoAllMatmulTilingDataType, IsMxFp4>::ExecuteGemm(uint32_t taskIndex)
{
  GM_ADDR aGM = computeContext_.baseData.aGM + taskIndex * computeContext_.baseData.aOffset;
  GM_ADDR bGM = computeContext_.baseData.bGM + taskIndex * computeContext_.baseData.bOffset;
  GM_ADDR cGM = computeContext_.baseData.cGM + taskIndex * computeContext_.baseData.cOffset;
  GM_ADDR x1Scale = computeContext_.additionalData.x1Scale + taskIndex * computeContext_.additionalData.x1ScaleOffset;
  GM_ADDR x2Scale = computeContext_.additionalData.x2Scale;

  const auto &mmTile = *computeContext_.tilingDataPtr;
  uint32_t rankDim = computeContext_.additionalData.rankDim;
  uint64_t rankKPerRank = static_cast<uint64_t>(mmTile.k) / rankDim;
  uint64_t scaleKPerRank = computeContext_.additionalData.scaleKPerRank;
  uint64_t aRankStride = static_cast<uint64_t>(computeContext_.additionalData.rankM) * rankKPerRank;
  uint64_t scaleARankStride = static_cast<uint64_t>(computeContext_.additionalData.rankM) * scaleKPerRank;

  ProblemShape problemShape;
  AscendC::Std::get<Blaze::Gemm::MNK_M>(problemShape) = static_cast<int64_t>(mmTile.m);
  AscendC::Std::get<Blaze::Gemm::MNK_N>(problemShape) = static_cast<int64_t>(mmTile.n);
  AscendC::Std::get<Blaze::Gemm::MNK_K>(problemShape) = static_cast<int64_t>(mmTile.k);
  AscendC::Std::get<Blaze::Gemm::MNK_B>(problemShape) = 1;

  SchParams sch;
  sch.baseM = 256;
  sch.baseN = 128;
  sch.mTailTile = mmTile.mTailTile;
  sch.nTailTile = mmTile.nTailTile;
  sch.mBaseTailSplitCnt = mmTile.mBaseTailSplitCnt;
  sch.nBaseTailSplitCnt = mmTile.nBaseTailSplitCnt;
  sch.mTailMain = mmTile.mTailMain;
  sch.nTailMain = mmTile.nTailMain;

  BlockScheduler bs(problemShape, sch);

  BlockMmad mmadOp;
  BlockShape l0TileShape{256, 128, static_cast<int64_t>(mmTile.baseK), 0};
  L1Params l1;
  l1.kL1 = static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK;
  l1.scaleKL1 = mmTile.scaleKL1;
  l1.l1BufNum = mmTile.nBufferNum;
  mmadOp.Init(problemShape, l0TileShape, l1, false, false);

  typename BlockMmad::AlltoAllParams alltoallParams;
  alltoallParams.rankDim = rankDim;
  alltoallParams.rankKPerRank = rankKPerRank;
  alltoallParams.scaleKPerRank = scaleKPerRank;
  alltoallParams.aRankStride = aRankStride;
  alltoallParams.scaleARankStride = scaleARankStride;
  mmadOp.SetAlltoAllParams(alltoallParams);
  mmadOp.SetGmAddrs(aGM, x1Scale);

  auto scaleKLen = Blaze::Gemm::CeilDiv(
      AscendC::Te::Get<Blaze::Gemm::MNK_K>(problemShape),
      static_cast<int64_t>(Blaze::Gemm::MXFP_DIVISOR_SIZE)) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

  auto layoutB = MakeLayoutB{}(
      AscendC::Te::Get<Blaze::Gemm::MNK_K>(problemShape),
      AscendC::Te::Get<Blaze::Gemm::MNK_N>(problemShape));
  auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen,
      AscendC::Te::Get<Blaze::Gemm::MNK_N>(problemShape));
  auto layoutBias = AscendC::Te::MakeFrameLayout<AscendC::Te::NDExtLayoutPtn>(
      1L, AscendC::Te::Get<Blaze::Gemm::MNK_N>(problemShape));
  auto layoutC = MakeLayoutC{}(
      AscendC::Te::Get<Blaze::Gemm::MNK_M>(problemShape),
      AscendC::Te::Get<Blaze::Gemm::MNK_N>(problemShape));

  auto gmB = AscendC::Te::MakeTensor(
      AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
          reinterpret_cast<__gm__ X2Type*>(bGM)), layoutB);
  auto gmScaleB = AscendC::Te::MakeTensor(
      AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
          reinterpret_cast<__gm__ AscendC::fp8_e8m0_t*>(x2Scale)), layoutScaleB);
  auto gmBias = AscendC::Te::MakeTensor(
      AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
          reinterpret_cast<__gm__ float*>(static_cast<uintptr_t>(0))), layoutBias);
  auto gmC = AscendC::Te::MakeTensor(
      AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(
          reinterpret_cast<__gm__ YType*>(cGM)), layoutC);

  {
    auto& mTailTile = sch.mTailTile;
    auto& nTailTile = sch.nTailTile;
    if ((bs.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum()) {
      bs.UpdateTailTile(mTailTile, nTailTile);
    }
  }

  {
    bool fullMTile = (256 >= AscendC::Te::Get<Blaze::Gemm::MNK_M>(problemShape));
    if (fullMTile) {
      int64_t scaleNStrideBytes =
          AscendC::Te::Get<Blaze::Gemm::MNK_N>(problemShape) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
      int64_t scaleBaseNStrideBytes = 128 * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
      bool scaleAlign = (scaleNStrideBytes & 0x7f) == 0 && (scaleBaseNStrideBytes & 0x7f) == 0;
      gmScaleB.SetL2CacheHint(scaleAlign ?
          AscendC::Te::CacheMode::CACHE_MODE_DISABLE : AscendC::Te::CacheMode::CACHE_MODE_NORMAL);

      bool bAlign = (AscendC::Te::Get<Blaze::Gemm::MNK_N>(problemShape) & kCacheLineAlignMask) == 0 &&
                    (128 & kCacheLineAlignMask) == 0;
      gmB.SetL2CacheHint(bAlign ?
          AscendC::Te::CacheMode::CACHE_MODE_DISABLE : AscendC::Te::CacheMode::CACHE_MODE_NORMAL);
    }
  }

  BlockCoord blockIdx;
  int64_t mPos = 0L;
  int64_t nPos = 0L;
  constexpr int64_t kPos = 0L;
  while (bs.GetTileIdx(blockIdx)) {
    BlockShape singleShape =
        bs.template GetBlockShape<Blaze::Gemm::QuantMode::MX_PERGROUP_MODE,
                                  Blaze::Gemm::QuantMode::MX_PERGROUP_MODE, false>(blockIdx);
    if (AscendC::Te::Get<Blaze::Gemm::IDX_M_TILEIDX>(singleShape) <= 0 ||
        AscendC::Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape) <= 0) {
      return;
    }

    bs.GetTileCoord(blockIdx, mPos, nPos);
    mmadOp.SetMPos(static_cast<uint64_t>(mPos));

    auto gmBlockB = gmB.Slice(
        AscendC::Te::MakeCoord(kPos, nPos),
        AscendC::Te::MakeShape(
            AscendC::Te::Get<Blaze::Gemm::MNK_K>(problemShape),
            AscendC::Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape)));
    auto gmBlockScaleB = gmScaleB.Slice(
        AscendC::Te::MakeCoord(kPos, nPos),
        AscendC::Te::MakeShape(scaleKLen,
            AscendC::Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape)));
    auto gmBlockBias = gmBias.Slice(
        AscendC::Te::MakeCoord(0L, nPos),
        AscendC::Te::MakeShape(1L,
            AscendC::Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape)));
    auto gmBlockC = gmC.Slice(
        AscendC::Te::MakeCoord(mPos, nPos),
        AscendC::Te::MakeShape(
            AscendC::Te::Get<Blaze::Gemm::IDX_M_TILEIDX>(singleShape),
            AscendC::Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape)));

    mmadOp(gmBlockB, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
  }
}

} // namespace Apace
