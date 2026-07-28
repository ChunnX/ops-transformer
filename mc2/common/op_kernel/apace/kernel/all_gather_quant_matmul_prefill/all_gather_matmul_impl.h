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
 * \file all_gather_matmul_impl.h
 * \brief AllGatherQuantMatmul 算子实现，基于 FragmentTensor + 通算解耦。
 */

#pragma once

#include "kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apace/kernel/all_gather_quant_matmul_prefill/all_gather_matmul_tiling_data.h"
#include "adv_api/hcomm/hcomm.h"
#include "apace/block/aiv_comm/collective_comm_context.h"
#include "apace/block/aiv_comm/collective_comm_api.h"
#include "apace/block/aiv_comm/barrier/barrier_ubmem.h"
#include "apace/tiling/comm_tiling_data.h"

#include "blaze/gemm/kernel/kernel_qbmm_mx.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "apace/block/blaze_ext/gemm/block/block_mmad_qbmm_mx_fragment.h"
#include "apace/basic/fragment_tensor/fragment_tensor.h"
#include "apace/basic/fragment_tensor/fragment_tensor_api.h"

namespace AllGatherQuantMatmulImpl {

using namespace AscendC;
using namespace Apace::AivComm;

using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::DNExtLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

enum RegionTag { HEAD, MAIN, TAIL };

template <typename AType, typename BType, typename CType>
class AllGatherQuantMatmulImpl {
public:
  __aicore__ inline AllGatherQuantMatmulImpl() {}
  __aicore__ inline void Init(__gm__ CommContext *hcommCtx, TPipe* pipe,
    GM_ADDR aGM, GM_ADDR aScaleGM,
    GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
    const AllGatherMatmulTilingData *tilingData);
  __aicore__ inline void Process();

  using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0>;
  using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
  using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
    ProblemShape, 0, LayoutA, LayoutB, AType>;
  using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, float, LayoutC>;
  using GemmKernel = Blaze::Gemm::Kernel::GemmUniversal<
    ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler,
    AscendC::Std::enable_if_t<
      AscendC::Std::is_same_v<Blaze::Gemm::KernelMmadWithScaleMx,
                  typename BlockMmad::DispatchPolicy::ScheduleType>>>;
  using GemmParams = typename GemmKernel::Params;
  using BlockMmadParams = typename BlockMmad::Params;
  using L1Params = typename BlockMmad::L1Params;
  using SchParams = typename BlockScheduler::Params;
  using QBMMTiling = typename GemmKernel::QBMMTiling;

  using BlockMmadFragC = Blaze::Gemm::Block::BlockMmadFragCOut<
    0, false, AType, LayoutA, BType, LayoutB, CType, LayoutC, float, LayoutC>;
  using FragL1Params = typename BlockMmadFragC::L1Params;
  using FragBlockShape = typename BlockMmadFragC::BlockShape;
  using FragTensorA = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT,
                            typename GemmKernel::MakeLayoutA, AType>;
  using FragScaleA = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT,
                            typename GemmKernel::MakeLayoutScaleA, AscendC::fp8_e8m0_t>;
  using FragTensorC = Apace::Basic::FragmentTensor<2, Apace::Basic::MAX_FRAGMENT_COUNT,
                            typename GemmKernel::MakeLayoutC, CType>;

private:
  __aicore__ inline void InitBaseParams(const AllGatherMatmulTilingData *td);

  __aicore__ inline void AllGatherProcess();
  __aicore__ inline void MatmulProcess();
  __aicore__ inline void BuildFragmentTensors();
  __aicore__ inline void EnsureWinRankBasesReady();
  __aicore__ inline void BuildMainFragment(uint32_t roundIdx);
  __aicore__ inline void BuildTailFragment();
  __aicore__ inline void UpdateMainRoundAddrs(uint32_t roundIdx);
  __aicore__ inline Apace::Basic::FragmentParam<2> MakeFragParam(
      uint64_t fragSize, uint64_t realFragSize, uint32_t fragCnt, uint64_t shape1) const;
    
   struct TileCtx {
    uint32_t dependId;
    uint32_t roundIdx;
    RegionTag region;
    int64_t regionMPos;
    const FragTensorA* fragA;
    const FragScaleA* fragScaleA;
    const FragTensorC* fragC;
    uint64_t rankCnt;
  };
  __aicore__ inline TileCtx ResolveTileCtx(int64_t mPos,
      int64_t headMainRows, int64_t mainRoundRows, int64_t mainSectionRows) const;

  __aicore__ inline GM_ADDR GetWinDataRegionBase();
  __aicore__ inline GM_ADDR GetWinScaleRegionBase();

  // 通信
  using AllGatherCommData = Apace::AivComm::CollectiveComm<
    Apace::AivComm::CommCollectiveOp::AllGather,
    Apace::AivComm::CommMode::PUT, AType, TeamBarrier, CommAddrLayout::DEFAULT>;
  using AllGatherCommScale = Apace::AivComm::CollectiveComm<
    Apace::AivComm::CommCollectiveOp::AllGather,
    Apace::AivComm::CommMode::PUT, AscendC::fp8_e8m0_t, TeamBarrier, CommAddrLayout::DEFAULT>;
  AllGatherCommData all_gather_data_;
  AllGatherCommScale all_gather_scale_;
  CommTilingData commTilingData_{};
  CommTilingData commTilingScale_{};
  TeamBarrier teamBarrier_;
  TBuf<TPosition::VECCALC> commBuf_;
  TBuf<TPosition::VECCALC> barrierBuf_;

  __gm__ CommContext *hcommCtx_{};
  __gm__ CommUdmaContext* udmaCtx_{};
  __gm__ CommUbmemContext* ubmemCtx_{};
  TPipe* pipe_{};

  GM_ADDR aGM_{};
  GM_ADDR aScaleGM_{};
  GM_ADDR bGM_{};
  GM_ADDR bScaleGM_{};
  GM_ADDR cGM_{};

  uint32_t rankId_{};
  uint32_t rankSize_{};
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

  const AllGatherMatmulTilingData *tilingData_{};

  // fragment tensor
  FragTensorA headFragA_{};
  FragScaleA headFragScaleA_{};
  FragTensorC headFragC_{};
  FragTensorA curMainA_{};
  FragScaleA curMainScaleA_{};
  FragTensorC curMainC_{};
  FragTensorA tailFragA_{};
  FragScaleA tailFragScaleA_{};
  FragTensorC tailFragC_{};

  // 地址状态
  GM_ADDR cFragAddrs_[8]{};
  GM_ADDR winDataRankBase_[8]{};
  GM_ADDR winScaleRankBase_[8]{};
  uint32_t curMainRoundIdx_{0xFFFFFFFF};
  bool winBasesReady_{false};
  bool tailBuilt_{false};
  uint64_t curRoundDataOff_{};
  uint64_t curRoundScaleOff_{};
  uint64_t curRoundCOff_{};

  // tile 步长
  uint64_t tileMDataStride_{};
  uint64_t tileMScaleStride_{};
  uint64_t tileMCStride_{};
};

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQuantMatmulImpl<AType, BType, CType>::Init(
  __gm__ CommContext *hcommCtx, TPipe* pipe,
  GM_ADDR aGM, GM_ADDR aScaleGM,
  GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
  const AllGatherMatmulTilingData *tilingData)
{
  tilingData_ = tilingData;
  hcommCtx_ = hcommCtx;
  pipe_ = pipe;
  aGM_ = aGM;
  aScaleGM_ = aScaleGM;
  bGM_ = bGM;
  bScaleGM_ = bScaleGM;
  cGM_ = cGM;

  InitBaseParams(tilingData);

  udmaCtx_ = &(hcommCtx_->udmaCtx);
  ubmemCtx_ = &(hcommCtx_->ubmemCtx);
  rankId_ = udmaCtx_->rankId;
  rankSize_ = udmaCtx_->rankSize;

  // Rank-major 布局：每个 rank 占完整的 M 行。data + scale 两段连续存放。
  //   Win buffer 内存布局：
  // ┌──────────────────────────────────────────────────┐
  // │  rank0 M行 data │ ... │ rankR-1 M行 data │          ← data 段
  // │  rank0 M行 scale│ ... │ rankR-1 M行 scale│         ← scale 段 (offset = dataRegionBytes_)
  // └──────────────────────────────────────────────────┘
  dataRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * dataBytesPerMRow_;
  scaleRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * scaleBytesPerMRow_;

  pipe_->InitBuffer(commBuf_, COMM_MAX_RANK_NUM);
  LocalTensor<uint8_t> commTensor = commBuf_.Get<uint8_t>();
  __ubuf__ uint8_t* commUbBuf = reinterpret_cast<__ubuf__ uint8_t*>(commTensor.GetPhyAddr());

  pipe_->InitBuffer(barrierBuf_, UB_SIZE);
  LocalTensor<uint8_t> barrierTensor = barrierBuf_.Get<uint8_t>();
  __ubuf__ uint8_t* barrierUbBuf = reinterpret_cast<__ubuf__ uint8_t*>(barrierTensor.GetPhyAddr());
  teamBarrier_.Init(barrierUbBuf, ubmemCtx_, rankSize_,
                    static_cast<uint32_t>(GetBlockIdx()));

  commTilingData_.splitAxisTileSize = tileM_;   // 每个 tile 搬 tileM 行
  commTilingData_.splitAxisTileCnt = tileCnt_;   // head 段 tile 数量
  commTilingData_.splitAxisTailSize = tailM_;   // tail 行数
  commTilingData_.splitAxisTailCnt = tailCnt_;
  commTilingData_.nonSplitAxisSize = k_;    // 非切分轴k

  commTilingScale_.splitAxisTileSize = tileM_;
  commTilingScale_.splitAxisTileCnt = tileCnt_;
  commTilingScale_.splitAxisTailSize = tailM_;
  commTilingScale_.splitAxisTailCnt = tailCnt_;
  scaleKLen_ = scaleKGroups_ * static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE);
  commTilingScale_.nonSplitAxisSize = scaleKLen_;

  all_gather_data_.Init(udmaCtx_, teamBarrier_, commTilingData_,
                        aGM_, commUbBuf, rankSize_,
                        static_cast<uint32_t>(GetBlockIdx()));
  all_gather_scale_.Init(udmaCtx_, teamBarrier_, commTilingScale_,
                         aScaleGM_, commUbBuf, rankSize_,
                         static_cast<uint32_t>(GetBlockIdx()), dataRegionBytes_);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQuantMatmulImpl<AType, BType, CType>::InitBaseParams(
  const AllGatherMatmulTilingData *td)
{
  m_ = td->m;
  k_ = td->k;
  n_ = td->n;
  tileCnt_ = td->tileCnt;
  tileM_ = td->tileM;
  tailCnt_ = td->tailCnt;
  tailM_ = td->tailM;
  paddedTailM_ = td->paddedTailM;
  commTurn_ = td->commTurn;
  headRows_ = static_cast<uint64_t>(tileCnt_) * tileM_;

  scaleKGroups_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(k_), Blaze::Gemm::MXFP_DIVISOR_SIZE);
  dataBytesPerMRow_ = static_cast<uint64_t>(k_) * sizeof(AType);
  scaleBytesPerMRow_ = scaleKGroups_ * static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE) * sizeof(AscendC::fp8_e8m0_t);
  cBytesPerM_ = static_cast<uint64_t>(n_) * sizeof(CType);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQuantMatmulImpl<AType, BType, CType>::Process()
{
  if ASCEND_IS_AIV {
    AllGatherProcess();
  }
  if ASCEND_IS_AIC {
    MatmulProcess();
  }
}

template <typename AType, typename BType, typename CType>
__aicore__ inline GM_ADDR AllGatherQuantMatmulImpl<AType, BType, CType>::GetWinDataRegionBase()
{
  return reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[rankId_]);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline GM_ADDR AllGatherQuantMatmulImpl<AType, BType, CType>::GetWinScaleRegionBase()
{
  return GetWinDataRegionBase() + dataRegionBytes_;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQuantMatmulImpl<AType, BType, CType>::AllGatherProcess()
{
  // 预触发 dependId=0：自身数据始终就绪，AIC 可直接消费。
  CrossCoreSetFlag<0x2, PIPE_MTE3>(0);

  for (uint32_t round = 0; round < commTurn_; ++round) {
    if (static_cast<uint32_t>(GetBlockIdx()) < rankSize_) {
      all_gather_scale_.Commit();
      all_gather_data_.Commit();
      all_gather_data_.Wait();
    }
    AscendC::SyncAll<true>();
    // round 0 对应 dependId=1，AIC 侧按 dependId 等待对应的远端 round 数据。
    CrossCoreSetFlag<0x2, PIPE_MTE3>(round + 1);
  }
  all_gather_scale_.Finalize();
  all_gather_data_.Finalize();
}

template <typename AType, typename BType, typename CType>
__aicore__ inline Apace::Basic::FragmentParam<2>
AllGatherQuantMatmulImpl<AType, BType, CType>::MakeFragParam(
    uint64_t fragSize, uint64_t realFragSize, uint32_t fragCnt, uint64_t shape1) const
{
  Apace::Basic::FragmentParam<2> param{};
  param.assembleAxis = 0;
  param.assembledShape[0] = fragSize * fragCnt;
  param.assembledShape[1] = shape1;
  param.fragmentSize = fragSize;
  param.realFragmentSize = realFragSize;
  param.fragmentCnt = fragCnt;
  return param;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQuantMatmulImpl<AType, BType, CType>::BuildFragmentTensors()
{
  constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;

  for (uint32_t r = 0; r < rankSize_; ++r) {
    cFragAddrs_[r] = cGM_ + static_cast<uint64_t>(r) * m_ * cBytesPerM_;
  }

  tileMDataStride_ = static_cast<uint64_t>(tileM_) * dataBytesPerMRow_;
  tileMScaleStride_ = static_cast<uint64_t>(tileM_) * scaleBytesPerMRow_;
  tileMCStride_ = static_cast<uint64_t>(tileM_) * cBytesPerM_;

  // HEAD：自身数据，单 fragment tensor。
  {
    GM_ADDR addrListA[MAX_FRAG] = {aGM_};
    GM_ADDR addrListScale[MAX_FRAG] = {aScaleGM_};
    GM_ADDR addrListC[MAX_FRAG] = {cFragAddrs_[rankId_]};
    headFragA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, typename GemmKernel::MakeLayoutA, AType>(
        MakeFragParam(headRows_, headRows_, 1, k_), addrListA);
    headFragScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, typename GemmKernel::MakeLayoutScaleA, AscendC::fp8_e8m0_t>(
        MakeFragParam(headRows_, headRows_, 1, scaleKLen_), addrListScale);
    headFragC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, typename GemmKernel::MakeLayoutC, CType>(
        MakeFragParam(headRows_, headRows_, 1, n_), addrListC);
  }
  // main / tail 的 fragment tensor 延迟到首次使用时构建，构建开销被 comm wait 掩盖。
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQuantMatmulImpl<AType, BType, CType>::EnsureWinRankBasesReady()
{
  if (winBasesReady_) { 
    return;
  }
  GM_ADDR winDataBase = GetWinDataRegionBase();
  GM_ADDR winScaleBase = GetWinScaleRegionBase();
  for (uint32_t r = 0; r < rankSize_; ++r) {
    winDataRankBase_[r] = winDataBase + static_cast<uint64_t>(r) * m_ * dataBytesPerMRow_;
    winScaleRankBase_[r] = winScaleBase + static_cast<uint64_t>(r) * m_ * scaleBytesPerMRow_;
  }
  winBasesReady_ = true;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQuantMatmulImpl<AType, BType, CType>::BuildMainFragment(uint32_t roundIdx)
{
  EnsureWinRankBasesReady();

  constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;

  // 绝对偏移：支持任意起始 roundIdx（调度器按 mPos 升序，首次必为 0）
  curRoundDataOff_ = static_cast<uint64_t>(roundIdx) * tileMDataStride_;
  curRoundScaleOff_ = static_cast<uint64_t>(roundIdx) * tileMScaleStride_;
  curRoundCOff_ = static_cast<uint64_t>(roundIdx) * tileMCStride_;

  GM_ADDR addrListA[MAX_FRAG];
  GM_ADDR addrListScale[MAX_FRAG];
  GM_ADDR addrListC[MAX_FRAG];
  uint32_t fragIdx = 0;
  for (uint32_t r = 0; r < rankSize_; ++r) {
    if (r == rankId_) { 
      continue; 
    }
    addrListA[fragIdx] = winDataRankBase_[r] + curRoundDataOff_;
    addrListScale[fragIdx] = winScaleRankBase_[r] + curRoundScaleOff_;
    addrListC[fragIdx] = cFragAddrs_[r] + curRoundCOff_;
    fragIdx++;
  }
  curMainA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, typename GemmKernel::MakeLayoutA, AType>(
      MakeFragParam(tileM_, tileM_, rankSize_ - 1, k_), addrListA);
  curMainScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, typename GemmKernel::MakeLayoutScaleA, AscendC::fp8_e8m0_t>(
      MakeFragParam(tileM_, tileM_, rankSize_ - 1, scaleKLen_), addrListScale);
  curMainC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, typename GemmKernel::MakeLayoutC, CType>(
      MakeFragParam(tileM_, tileM_, rankSize_ - 1, n_), addrListC);
  curMainRoundIdx_ = roundIdx;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQuantMatmulImpl<AType, BType, CType>::BuildTailFragment()
{
  EnsureWinRankBasesReady();

  constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;

  uint64_t tailRowOff = headRows_;
  uint64_t tailDataOff = tailRowOff * dataBytesPerMRow_;
  uint64_t tailScaleOff = tailRowOff * scaleBytesPerMRow_;
  uint64_t tailCOff = tailRowOff * cBytesPerM_;

  GM_ADDR addrListA[MAX_FRAG];
  GM_ADDR addrListScale[MAX_FRAG];
  GM_ADDR addrListC[MAX_FRAG];
  for (uint32_t r = 0; r < rankSize_; ++r) {
    if (r == rankId_) {
      addrListA[r] = aGM_ + tailDataOff;
      addrListScale[r] = aScaleGM_ + tailScaleOff;
    } else {
      addrListA[r] = winDataRankBase_[r] + tailDataOff;
      addrListScale[r] = winScaleRankBase_[r] + tailScaleOff;
    }
    addrListC[r] = cFragAddrs_[r] + tailCOff;
  }
  tailFragA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, typename GemmKernel::MakeLayoutA, AType>(
      MakeFragParam(paddedTailM_, tailM_, rankSize_, k_), addrListA);
  tailFragScaleA_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, typename GemmKernel::MakeLayoutScaleA, AscendC::fp8_e8m0_t>(
      MakeFragParam(paddedTailM_, tailM_, rankSize_, scaleKLen_), addrListScale);
  tailFragC_ = Apace::Basic::MakeFragmentTensor<2, MAX_FRAG, typename GemmKernel::MakeLayoutC, CType>(
      MakeFragParam(paddedTailM_, tailM_, rankSize_, n_), addrListC);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQuantMatmulImpl<AType, BType, CType>::UpdateMainRoundAddrs(uint32_t roundIdx)
{
  uint32_t delta = roundIdx - curMainRoundIdx_;
  curRoundDataOff_ += static_cast<uint64_t>(delta) * tileMDataStride_;
  curRoundScaleOff_ += static_cast<uint64_t>(delta) * tileMScaleStride_;
  curRoundCOff_ += static_cast<uint64_t>(delta) * tileMCStride_;

  constexpr uint32_t MAX_FRAG = Apace::Basic::MAX_FRAGMENT_COUNT;
  GM_ADDR addrListA[MAX_FRAG];
  GM_ADDR addrListScale[MAX_FRAG];
  GM_ADDR addrListC[MAX_FRAG];
  uint32_t fragIdx = 0;
  for (uint32_t r = 0; r < rankSize_; ++r) {
    if (r == rankId_) {
      continue;
    }
    addrListA[fragIdx] = winDataRankBase_[r] + curRoundDataOff_;
    addrListScale[fragIdx] = winScaleRankBase_[r] + curRoundScaleOff_;
    addrListC[fragIdx] = cFragAddrs_[r] + curRoundCOff_;
    fragIdx++;
  }
  curMainA_.UpdateAddrList(addrListA);
  curMainScaleA_.UpdateAddrList(addrListScale);
  curMainC_.UpdateAddrList(addrListC);
  curMainRoundIdx_ = roundIdx;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline typename AllGatherQuantMatmulImpl<AType, BType, CType>::TileCtx
AllGatherQuantMatmulImpl<AType, BType, CType>::ResolveTileCtx(
    int64_t mPos, int64_t headMainRows, int64_t mainRoundRows, int64_t mainSectionRows) const
{
  TileCtx ctx{};
  if (mPos < headMainRows) {
    ctx.dependId = 0;
    ctx.region = HEAD;
    ctx.regionMPos = mPos;
    ctx.fragA = &headFragA_;
    ctx.fragScaleA = &headFragScaleA_;
    ctx.fragC = &headFragC_;
    ctx.rankCnt = static_cast<uint64_t>(rankSize_ - 1);
  } else if (mPos < headMainRows + mainSectionRows) {
    int64_t relPos = mPos - headMainRows;
    ctx.roundIdx = static_cast<uint32_t>(relPos / mainRoundRows);
    ctx.dependId = ctx.roundIdx + 1;
    ctx.region = MAIN;
    ctx.regionMPos = relPos % mainRoundRows;
    ctx.fragA = &curMainA_;
    ctx.fragScaleA = &curMainScaleA_;
    ctx.fragC = &curMainC_;
    ctx.rankCnt = static_cast<uint64_t>(rankSize_ - 1);
  } else {
    ctx.dependId = commTurn_;
    ctx.region = TAIL;
    ctx.regionMPos = mPos - headMainRows - mainSectionRows;
    ctx.fragA = &tailFragA_;
    ctx.fragScaleA = &tailFragScaleA_;
    ctx.fragC = &tailFragC_;
    ctx.rankCnt = static_cast<uint64_t>(rankSize_);
  }
  return ctx;
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherQuantMatmulImpl<AType, BType, CType>::MatmulProcess()
{
  BuildFragmentTensors();

  const auto &mmT = tilingData_->mmTile;
  const int64_t Ki = static_cast<int64_t>(k_);
  const int64_t Ni = static_cast<int64_t>(n_);
  const int64_t scaleKLen = Blaze::Gemm::CeilDiv(Ki, static_cast<int64_t>(Blaze::Gemm::MXFP_DIVISOR_SIZE)) *
                            Blaze::Gemm::MXFP_MULTI_BASE_SIZE;    // todo

  // B / scaleB / bias 全局共享。
  auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BType *>(bGM_)),
               typename GemmKernel::MakeLayoutB{}(Ki, Ni));
  auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                     reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(bScaleGM_)),
                     typename GemmKernel::MakeLayoutScaleB{}(scaleKLen, Ni));
  __gm__ float *biasNull = nullptr;
  auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasNull),
                  Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Ni));

  // 总 logical M = head + main 段 + tail。
  bool hasTail = (tailCnt_ > 0 && tailM_ > 0);
  const int64_t headMainRows = static_cast<int64_t>(headRows_);
  const int64_t mainRoundRows = static_cast<int64_t>(rankSize_ - 1) * static_cast<int64_t>(tileM_);   // 单主round的行数
  const int64_t mainSectionRows = static_cast<int64_t>(tileCnt_) * mainRoundRows;   // 主round所有行
  const int64_t tailRows = hasTail ? static_cast<int64_t>(rankSize_) * static_cast<int64_t>(paddedTailM_) : 0;
  const int64_t totalLogicalM = headMainRows + mainSectionRows + tailRows;  // 逻辑总行数

  typename BlockScheduler::Params schParams;
  schParams.baseM = mmT.baseM;
  schParams.baseN = mmT.baseN;
  schParams.mTailTile = mmT.mTailTile;
  schParams.nTailTile = mmT.nTailTile;
  schParams.mBaseTailSplitCnt = mmT.mBaseTailSplitCnt;
  schParams.nBaseTailSplitCnt = mmT.nBaseTailSplitCnt;
  schParams.mTailMain = mmT.mTailMain;
  schParams.nTailMain = mmT.nTailMain;

  FragBlockShape l0TileShape{static_cast<int64_t>(mmT.baseM), static_cast<int64_t>(mmT.baseN),
                             static_cast<int64_t>(mmT.baseK), 0L};  // batch=0，单 batch，不切分
  FragL1Params fragL1{static_cast<uint64_t>(mmT.stepK) * mmT.baseK, mmT.scaleKL1, mmT.nBufferNum};

  ProblemShape problemShape{totalLogicalM, Ni, Ki, 1};
  BlockScheduler sch(problemShape, schParams);
  BlockMmadFragC mmadFrag;
  mmadFrag.Init(problemShape, l0TileShape, fragL1, false, mmT.dbL0c > 1);

  uint32_t waitedMask = 0U;
  // dependId=0 由 AIV 预触发，无需等待。

  Te::Coord<int64_t, int64_t, int64_t, int64_t> blockIdx;
  int64_t mPos = 0L, nPos = 0L;

  while (sch.GetTileIdx(blockIdx)) {
    auto singleShape = sch.template GetBlockShape<
      Blaze::Gemm::QuantMode::MX_PERGROUP_MODE, Blaze::Gemm::QuantMode::MX_PERGROUP_MODE,
      BlockMmadFragC::weightNz>(blockIdx);
    int64_t curMtile = Te::Get<0>(singleShape);
    int64_t curNtile = Te::Get<1>(singleShape);
    if (curMtile <= 0 || curNtile <= 0) { 
      break;
     }
    sch.GetTileCoord(blockIdx, mPos, nPos); // 根据调度器得到当前block的m/n坐标

    auto ctx = ResolveTileCtx(mPos, headMainRows, mainRoundRows, mainSectionRows);  // 用于计算当前块用于fragment的context

    // 按位记录已等待过的 dependId，避免重复 wait。
    if ((waitedMask & (1U << ctx.dependId)) == 0U) {
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(ctx.dependId);
      waitedMask |= (1U << ctx.dependId);
    }

    auto gmBlockB = gmB.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(Ki, curNtile));
    auto gmBlockScaleB = gmScaleB.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(scaleKLen, curNtile));
    auto gmBlockBias = gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, curNtile));

    auto coordA = Te::MakeCoord(ctx.regionMPos, 0L);
    auto coordC = Te::MakeCoord(ctx.regionMPos, nPos);
    auto shapeA = Te::MakeShape(curMtile, Ki);
    auto shapeScaleA = Te::MakeShape(curMtile, scaleKLen);
    auto shapeC = Te::MakeShape(curMtile, curNtile);

    // MAIN/TAIL 的 fragment tensor 延迟到首次命中时构建/更新地址。
    if (ctx.region == MAIN) {
      if (curMainRoundIdx_ == 0xFFFFFFFF) {
        BuildMainFragment(ctx.roundIdx);
      } else if (curMainRoundIdx_ != ctx.roundIdx) {
        UpdateMainRoundAddrs(ctx.roundIdx);  // 非首主轮，用累加更新 todo,更好写法
      }
    } else if (ctx.region == TAIL && !tailBuilt_) {
      BuildTailFragment();
      tailBuilt_ = true;
    }

    auto blockA = ctx.fragA->Slice(coordA, shapeA);
    auto blockScaleA = ctx.fragScaleA->Slice(coordA, shapeScaleA);
    auto blockC = ctx.fragC->Slice(coordC, shapeC);
    mmadFrag(blockA, gmBlockB, blockScaleA, gmBlockScaleB, gmBlockBias,
             cFragAddrs_, static_cast<uint64_t>(m_), static_cast<uint64_t>(tileM_),
             static_cast<uint64_t>(tileCnt_), static_cast<uint64_t>(tailM_),
             ctx.rankCnt, Ni, singleShape, ctx.regionMPos, nPos, 0, blockC);
  }

  // 确保所有 dependId 均已 wait（收尾清理）。
  for (uint32_t t = 0; t <= commTurn_; ++t) {
    if ((waitedMask & (1U << t)) == 0U) {
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(t);
    }
  }
}

} // namespace AllGatherQuantMatmulImpl

__global__ __aicore__ void AllGatherQuantMatmulKernel(
  __gm__ Apace::AivComm::CommContext *hcommCtx,
  GM_ADDR aGM, GM_ADDR aScaleGM,
  GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
  AllGatherMatmulTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;
  AllGatherQuantMatmulImpl::AllGatherQuantMatmulImpl<
    fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t> impl;
  impl.Init(hcommCtx, &pipe, aGM, aScaleGM, bGM, bScaleGM, cGM, &tilingData);
  impl.Process();
}
