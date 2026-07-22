/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

/*!
 * \file qbmm_kernel.h
 * \brief MatmulAllToAll MX 算子的 matmul 调度实现（仅负责 AIC 侧计算调度，不涉及通信）
 *
 * 职责：
 * - 持有 BlockMmad + BlockScheduler，按 tiling 调度 QBMM 计算
 * - 管理计算所需的 GM 地址、tiling 参数、tile 切分
 * - 提供 MatmulProcess / LocalCompute 入口供主类调用
 * - 不感知通信上下文（CommContext/UDMA/UBMEM），rankId/rankSize 由主类注入
 * - Scale 全在本卡，不参与通信
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "matmul_all_to_all_tiling_data.h"

#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "tensor_api/tensor.h"

namespace MatmulAllToAllMxImpl {

using namespace AscendC;
using namespace Blaze::Gemm;

using LayoutA = AscendC::Te::NDExtLayoutPtn;
using LayoutB = AscendC::Te::DNExtLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;
using LayoutBias = AscendC::Te::NDExtLayoutPtn;
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

struct BaseParams {
  GM_ADDR winGm{};
  GM_ADDR selfWinAddr{};
  GM_ADDR aGm{};
  GM_ADDR scaleAGm{};
  GM_ADDR bGm{};
  GM_ADDR scaleBGm{};
  GM_ADDR cGm{};
  GM_ADDR selfCBase{};

  uint32_t rankId{0};
  uint32_t rankSize{0};
  uint32_t tileCnt{0};
  uint32_t tileM{0};
  uint32_t tailM{0};
  uint32_t tailCnt{0};
  uint32_t totalTileCnt{0};
  uint32_t bufferCount{0};

  uint64_t axisM{0};
  uint64_t axisK{0};
  uint64_t axisNPerRank{0};
  uint64_t maxTileM{0};

  uint64_t aByteStrideK{0};
  uint64_t cByteStrideN{0};
  uint64_t scaleAByteStride{0};
  uint64_t scaleAElemCnt{0};

  uint64_t tileChunkBytes{0};
  uint64_t bufferSlotBytes{0};
};

template <typename AType, typename BType, typename CType, bool LocalDelay>
class QuantBatchMatmulKernel {
public:
  __aicore__ inline QuantBatchMatmulKernel() {};
  __aicore__ inline ~QuantBatchMatmulKernel() {}

  __aicore__ inline void Init(uint32_t rankId, uint32_t rankSize,
    GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
    const MatmulAllToAllTilingData *tilingData);

  __aicore__ inline void MatmulProcess();
  __aicore__ inline void LocalCompute();

  __aicore__ inline BaseParams& GetBase() { return base_; }
  __aicore__ inline const BaseParams& GetBase() const { return base_; }

  using TypeScaleA = ::fp8_e8m0_t;
  using TypeScaleB = ::fp8_e8m0_t;
  using BiasType = float;
  static constexpr int32_t SCALE_C0 = 2;
  static constexpr uint16_t MTE1_MTE2_EVENT_FLAG = 6;

  using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0>;
  using BlockScheduler =
    Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, AType>;
  using BlockMmad =
    Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;

  static constexpr bool weightNz = BlockMmad::WEIGHT_NZ;
  static constexpr bool transA = BlockMmad::TRANS_A;
  static constexpr bool transB = BlockMmad::TRANS_B;
  static constexpr int64_t C0_SIZE = Blaze::Gemm::IsFp4<AType>() ? Blaze::Gemm::C0_SIZE_B4 : Blaze::Gemm::C0_SIZE_B8;
  static constexpr int64_t kCacheLineAlignMask = Blaze::Gemm::IsFp4<AType>() ? 0xff : 0x7f;

  using BlockShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;
  using BlockCoord = Te::Coord<int64_t, int64_t, int64_t, int64_t>;

  using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
  using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
  using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE>>;
  using MakeLayoutScaleA = Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<SCALE_C0>>;
  using MakeLayoutScaleB = Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<SCALE_C0>>;

  using BlockMmadParams = typename BlockMmad::Params;
  using L1Params = typename BlockMmad::L1Params;
  using BlockSchedulerParams = typename BlockScheduler::Params;

  struct QBMMTiling {
    enum BiasMode : uint32_t {
      BIAS_DISABLED = 0,
      BIAS_ENABLED = 1
    };
    uint32_t batchCount;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    BiasMode isBias;
    uint32_t dbL0c;
    bool disableCL2Cache;
  };

  struct LocalParams {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t origM;
    uint32_t selfBatchIdx;
    GM_ADDR cGmSelfAddr;
    bool skipSelfBatch;
  };

  struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    L1Params l1Params;
    BlockSchedulerParams schParams;
    QBMMTiling qbmmParams;
    LocalParams localParams;
  };

private:
  __aicore__ inline void InitBaseParams(const MatmulAllToAllTilingData *td);
  __aicore__ inline void SetupParams(const QuantMatmulTilingData &mmTile,
                     uint32_t batchSize, uint32_t batchRankOffset,
                     uint64_t origM, bool isLocal, Params &out);
  __aicore__ inline void RunMatmul(const Params &params);
  __aicore__ inline void InitMatmul(const Params &params);
  __aicore__ inline void ResetGmAddr(const Params &params);
  __aicore__ inline void ProcessSingleBatch(const Params &params, BlockScheduler &bs,
                        uint64_t restBatch, bool isTailRound);

  template <typename TensorB, typename TensorScaleB, typename TensorC>
  __aicore__ inline void SetL2Cache(
    const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB &gmB,
    TensorScaleB &gmScaleB, TensorC &gmC);
  template <typename TensorScaleB>
  __aicore__ inline void SetScaleL2Cache(
    const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB);

  BlockMmad mmadOp_;
  BaseParams base_;

  const MatmulAllToAllTilingData *tilingData_{};

  __gm__ AType *aGmAddr_{};
  __gm__ BType *bGmAddr_{};
  __gm__ CType *cGmAddr_{};
  __gm__ CType *cGmSelfAddr_{};
  __gm__ BiasType *biasGmAddr_{};
  __gm__ AscendC::fp8_e8m0_t *scaleAGmAddr_{};
  __gm__ AscendC::fp8_e8m0_t *scaleBGmAddr_{};

  bool isBias_{false};
  bool needUpdateTail_{false};
  bool disableCL2Cache_{false};
};

// =================================================================================
// 公有方法实现
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void QuantBatchMatmulKernel<AType, BType, CType, LocalDelay>::Init(
  uint32_t rankId, uint32_t rankSize,
  GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
  const MatmulAllToAllTilingData *tilingData)
{
  tilingData_ = tilingData;
  base_.rankId = rankId;
  base_.rankSize = rankSize;
  base_.aGm = aGM;
  base_.scaleAGm = aScaleGM;
  base_.bGm = bGM;
  base_.scaleBGm = bScaleGM;
  base_.cGm = cGM;

  InitBaseParams(tilingData);
}

// =================================================================================
// 参数推导与构造
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void QuantBatchMatmulKernel<AType, BType, CType, LocalDelay>::InitBaseParams(
const MatmulAllToAllTilingData *tilingData)
{
  const auto& commTd = tilingData->alltoallTilingData.commTilingData;
  base_.axisK = tilingData->tileQbmmTilingData.k;
  base_.axisNPerRank = tilingData->tileQbmmTilingData.n;
  base_.tileCnt = static_cast<uint32_t>(commTd.splitAxisTileCnt);
  base_.tileM = static_cast<uint32_t>(commTd.splitAxisTileSize);
  base_.tailM = static_cast<uint32_t>(commTd.splitAxisTailSize);
  base_.tailCnt = static_cast<uint32_t>(commTd.splitAxisTailCnt);
  base_.totalTileCnt = base_.tileCnt + base_.tailCnt;
  base_.axisM = static_cast<uint64_t>(base_.tileCnt) * base_.tileM +
                static_cast<uint64_t>(base_.tailCnt) * base_.tailM;
  base_.maxTileM = (base_.tileM > base_.tailM) ? base_.tileM : base_.tailM;
  base_.cByteStrideN = base_.axisNPerRank * sizeof(CType);
  uint32_t scaleKGroups = (base_.axisK + Blaze::Gemm::MXFP_DIVISOR_SIZE - 1) / Blaze::Gemm::MXFP_DIVISOR_SIZE;
  base_.scaleAElemCnt = scaleKGroups * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
  base_.aByteStrideK = base_.axisK * sizeof(AType);
  base_.scaleAByteStride = base_.scaleAElemCnt * sizeof(AscendC::fp8_e8m0_t);
  base_.bufferCount = tilingData->alltoallTilingData.bufferSize;
  base_.tileChunkBytes = base_.maxTileM * base_.cByteStrideN;
  base_.bufferSlotBytes = base_.rankSize * base_.tileChunkBytes;
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void QuantBatchMatmulKernel<AType, BType, CType, LocalDelay>::SetupParams(
  const QuantMatmulTilingData &mmTile,
  uint32_t batchSize, uint32_t batchRankOffset,
  uint64_t origM, bool isLocal, Params &out)
{
  AscendC::Std::get<Blaze::Gemm::MNK_M>(out.problemShape) = static_cast<int64_t>(mmTile.m);
  AscendC::Std::get<Blaze::Gemm::MNK_N>(out.problemShape) = static_cast<int64_t>(base_.axisNPerRank);
  AscendC::Std::get<Blaze::Gemm::MNK_K>(out.problemShape) = static_cast<int64_t>(base_.axisK);
  AscendC::Std::get<Blaze::Gemm::MNK_B>(out.problemShape) = static_cast<int64_t>(batchSize);
  constexpr uint64_t sizeShift = Blaze::Gemm::IsFp4<AType>() ? 1 : 0;
  out.mmadParams = {base_.aGm,
    base_.bGm + ((static_cast<uint64_t>(batchRankOffset) * base_.axisNPerRank * base_.axisK) >> sizeShift),
    isLocal ? base_.selfCBase : base_.winGm, nullptr,
    base_.scaleAGm, base_.scaleBGm + static_cast<uint64_t>(batchRankOffset) * base_.axisNPerRank * base_.scaleAElemCnt};
  out.l1Params = {static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
  out.schParams = {mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
           mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};
  out.qbmmParams = {batchSize, mmTile.baseM, mmTile.baseN, mmTile.baseK,
           QBMMTiling::BIAS_DISABLED, mmTile.dbL0c,
           LocalDelay && !isLocal};
  out.localParams = {base_.rankId, base_.rankSize, origM, base_.rankId,
           base_.selfCBase,
           LocalDelay && !isLocal};
}

// =================================================================================
// 计算逻辑实现 (AIC 侧)
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void QuantBatchMatmulKernel<AType, BType, CType, LocalDelay>::MatmulProcess()
{
  uint32_t remoteBatchSize = LocalDelay ? base_.rankSize - 1 : base_.rankSize;
  uint32_t remoteBatchRankOffset = 0;
  Params param;
  SetupParams(tilingData_->tileQbmmTilingData, remoteBatchSize, remoteBatchRankOffset,
        base_.maxTileM, false, param);

  GM_ADDR aCur = base_.aGm;
  GM_ADDR aScaleCur = base_.scaleAGm;
  GM_ADDR cSelfCur = base_.selfCBase;
  uint32_t lastTiling = 0;
  for (uint32_t tid = 0; tid < base_.totalTileCnt; ++tid) {
    if (base_.bufferCount < base_.totalTileCnt && tid >= base_.bufferCount) {
      CrossCoreWaitFlag<0x2, PIPE_M>(tid - base_.bufferCount);
    }
    bool isTail = (tid >= base_.tileCnt);
    uint32_t curTileM = isTail ? base_.tailM : base_.tileM;
    uint32_t curTiling = isTail ? 1 : 0;
    if (curTiling != lastTiling) {
      SetupParams(isTail ? tilingData_->tailQbmmTilingData : tilingData_->tileQbmmTilingData,
        remoteBatchSize, remoteBatchRankOffset, base_.maxTileM, false, param);
      lastTiling = curTiling;
    }
    uint32_t bufIdx = tid % base_.bufferCount;
    GM_ADDR cCur = base_.selfWinAddr + bufIdx * base_.bufferSlotBytes;
    param.mmadParams.aGmAddr = aCur;
    param.mmadParams.scaleAGmAddr = aScaleCur;
    param.mmadParams.cGmAddr = cCur;
    if (!LocalDelay) { param.localParams.cGmSelfAddr = cSelfCur; }
    RunMatmul(param);

    CrossCoreSetFlag<0x2, PIPE_FIX>(tid);
    aCur += curTileM * base_.aByteStrideK;
    aScaleCur += curTileM * base_.scaleAByteStride;
    cSelfCur += curTileM * base_.cByteStrideN;

    if (base_.tailCnt > 0 && tid < base_.totalTileCnt - 1) {
      AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    }
  }
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void QuantBatchMatmulKernel<AType, BType, CType, LocalDelay>::LocalCompute()
{
  Params localParam;
  SetupParams(tilingData_->localQbmmTilingData, 1, base_.rankId, base_.axisM, true, localParam);
  RunMatmul(localParam);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void QuantBatchMatmulKernel<AType, BType, CType, LocalDelay>::RunMatmul(const Params &params)
{
  InitMatmul(params);

  BlockScheduler bs(params.problemShape, params.schParams);
  BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
  bool enableL0CPingPong = (params.qbmmParams.dbL0c > 1);
  mmadOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias_, enableL0CPingPong);

  int64_t N = Te::Get<MNK_N>(params.problemShape);
  int64_t K = Te::Get<MNK_K>(params.problemShape);
  constexpr uint64_t sizeShift = Blaze::Gemm::IsFp4<AType>() ? 1 : 0;
  int64_t scaleKLen =
    Blaze::Gemm::CeilDiv(K, static_cast<int64_t>(Blaze::Gemm::MXFP_DIVISOR_SIZE)) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

  uint32_t batchCnt = params.qbmmParams.batchCount;
  uint64_t singleBatchTileCnt = bs.GetTotalCnt();
  uint64_t tailRoundStart =
    (singleBatchTileCnt * batchCnt / AscendC::GetBlockNum()) * AscendC::GetBlockNum();

  for (uint32_t b = 0; b < batchCnt; ++b) {
    uint64_t curBatch = static_cast<uint64_t>(b) + 1;
    bool isTailRound = curBatch * singleBatchTileCnt > tailRoundStart;
    uint64_t restBatch = batchCnt - curBatch;

    uint32_t realRank = params.localParams.skipSelfBatch
      ? ((b < params.localParams.selfBatchIdx) ? b : b + 1) : b;

    ResetGmAddr(params);
    bGmAddr_ += (static_cast<uint64_t>(realRank) * N * K) >> sizeShift;
    scaleBGmAddr_ += static_cast<uint64_t>(realRank) * N * static_cast<uint64_t>(scaleKLen);

    if (!params.localParams.skipSelfBatch && b == params.localParams.selfBatchIdx) {
      cGmAddr_ = cGmSelfAddr_;
    } else {
      cGmAddr_ += static_cast<uint64_t>(realRank) * params.localParams.origM * N;
    }

    ProcessSingleBatch(params, bs, restBatch, isTailRound);
  }
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void QuantBatchMatmulKernel<AType, BType, CType, LocalDelay>::InitMatmul(const Params &params)
{
  if ASCEND_IS_AIV {
    return;
  }
  isBias_ = (params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED);
  disableCL2Cache_ = params.qbmmParams.disableCL2Cache;
  needUpdateTail_ = false;
  ResetGmAddr(params);
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void QuantBatchMatmulKernel<AType, BType, CType, LocalDelay>::ResetGmAddr(const Params &params)
{
  if ASCEND_IS_AIV {
    return;
  }
  aGmAddr_ = reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr);
  bGmAddr_ = reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr);
  cGmAddr_ = reinterpret_cast<__gm__ CType *>(params.mmadParams.cGmAddr);
  cGmSelfAddr_ = reinterpret_cast<__gm__ CType *>(params.localParams.cGmSelfAddr);
  scaleAGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(params.mmadParams.scaleAGmAddr);
  scaleBGmAddr_ = reinterpret_cast<__gm__ AscendC::fp8_e8m0_t *>(params.mmadParams.scaleBGmAddr);
  if (isBias_) {
    biasGmAddr_ = reinterpret_cast<__gm__ BiasType *>(params.mmadParams.biasGmAddr);
  }
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
__aicore__ inline void QuantBatchMatmulKernel<AType, BType, CType, LocalDelay>::ProcessSingleBatch(
  const Params &params, BlockScheduler &bs, uint64_t restBatch, bool isTailRound)
{
  auto scaleKLen =
    Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(Blaze::Gemm::MXFP_DIVISOR_SIZE)) *
    Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
  auto layoutA = MakeLayoutA{}(Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_K>(params.problemShape));
  auto layoutScaleA = MakeLayoutScaleA{}(Te::Get<MNK_M>(params.problemShape), scaleKLen);
  auto layoutB = MakeLayoutB{}(Te::Get<MNK_K>(params.problemShape), Te::Get<MNK_N>(params.problemShape));
  auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, Te::Get<MNK_N>(params.problemShape));
  auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Te::Get<MNK_N>(params.problemShape));
  auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_N>(params.problemShape));

  auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aGmAddr_), layoutA);
  auto gmScaleA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleAGmAddr_), layoutScaleA);
  auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), layoutB);
  auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBGmAddr_), layoutScaleB);
  auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasGmAddr_), layoutBias);
  auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cGmAddr_), layoutC);

  auto& mTailTile = params.schParams.mTailTile;
  auto& nTailTile = params.schParams.nTailTile;
  if (needUpdateTail_ ||
    (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
              AscendC::GetBlockNum())) {
    needUpdateTail_ = true;
    bs.UpdateTailTile(mTailTile, nTailTile);
  }
  SetL2Cache(
    params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, params.l1Params.scaleKL1, gmB, gmScaleB,
    gmC);

  BlockCoord blockIdx;
  int64_t mPos = 0L;
  int64_t nPos = 0L;
  constexpr int64_t kPos = 0L;
  while (bs.GetTileIdx(blockIdx)) {
    BlockShape singleShape =
      bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
    if (Te::Get<Blaze::Gemm::IDX_M_TILEIDX>(singleShape) <= 0 || Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape) <= 0) {
      return;
    }

    bs.GetTileCoord(blockIdx, mPos, nPos);
    auto gmBlockA = gmA.Slice(
      Te::MakeCoord(mPos, kPos),
      Te::MakeShape(Te::Get<Blaze::Gemm::IDX_M_TILEIDX>(singleShape), Te::Get<MNK_K>(params.problemShape)));
    auto gmBlockScaleA =
      gmScaleA.Slice(Te::MakeCoord(mPos, kPos), Te::MakeShape(Te::Get<Blaze::Gemm::IDX_M_TILEIDX>(singleShape), scaleKLen));
    auto gmBlockB = gmB.Slice(
      Te::MakeCoord(kPos, nPos),
      Te::MakeShape(Te::Get<MNK_K>(params.problemShape), Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape)));
    auto gmBlockScaleB =
      gmScaleB.Slice(Te::MakeCoord(kPos, nPos), Te::MakeShape(scaleKLen, Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape)));
    auto gmBlockBias =
      gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape)));
    auto gmBlockC = gmC.Slice(
      Te::MakeCoord(mPos, nPos),
      Te::MakeShape(Te::Get<Blaze::Gemm::IDX_M_TILEIDX>(singleShape), Te::Get<Blaze::Gemm::IDX_N_TILEIDX>(singleShape)));
    mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape);
  }
  bs.UpdateNextBatchBlockRoundParams();
}

// =================================================================================
// L2 Cache 优化
// =================================================================================

template <typename AType, typename BType, typename CType, bool LocalDelay>
template <typename TensorScaleB>
__aicore__ inline void QuantBatchMatmulKernel<AType, BType, CType, LocalDelay>::SetScaleL2Cache(
  const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB)
{
  if (Te::Get<MNK_B>(problemShape) != 1) {
    return;
  }
  if constexpr (transB) {
    const int64_t scaleKRowBytes =
      Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(problemShape), static_cast<int64_t>(Blaze::Gemm::MXFP_DIVISOR_SIZE)) *
      Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
    const int64_t scaleKL1RowBytes = Blaze::Gemm::CeilDiv(scaleKL1, Blaze::Gemm::MXFP_DIVISOR_SIZE) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
    const bool scaleAlignForL2Stream = (scaleKRowBytes & kCacheLineAlignMask) == 0 &&
      (scaleKL1RowBytes & kCacheLineAlignMask) == 0;
    gmScaleB.SetL2CacheHint(
      scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
  } else {
    const int64_t scaleNStrideBytes = Te::Get<MNK_N>(problemShape) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
    const int64_t scaleBaseNStrideBytes = baseN * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
    const bool scaleAlignForL2Stream = (scaleNStrideBytes & kCacheLineAlignMask) == 0 &&
      (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
    gmScaleB.SetL2CacheHint(
      scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
  }
}

template <typename AType, typename BType, typename CType, bool LocalDelay>
template <typename TensorB, typename TensorScaleB, typename TensorC>
__aicore__ inline void QuantBatchMatmulKernel<AType, BType, CType, LocalDelay>::SetL2Cache(
  const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB &gmB,
  TensorScaleB &gmScaleB, TensorC &gmC)
{
  if (disableCL2Cache_) {
    gmC.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
  }

  const bool fullMTile = curBaseM >= Te::Get<MNK_M>(problemShape);
  if (!fullMTile) {
    return;
  }

  SetScaleL2Cache(problemShape, baseN, scaleKL1, gmScaleB);

  if constexpr (weightNz) {
    gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
  } else {
    if constexpr (transB) {
      bool bAlignForL2Stream = (Te::Get<MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
      gmB.SetL2CacheHint(
        bAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
      bool bAlignForL2Stream =
        (Te::Get<MNK_N>(problemShape) & kCacheLineAlignMask) == 0 && (baseN & kCacheLineAlignMask) == 0;
      gmB.SetL2CacheHint(
        bAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    }
  }
}

} // namespace MatmulAllToAllMxImpl
