// license

#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "include/tensor_api/tensor.h"

#include "adv_api/hcomm/hcomm.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "qbmm_mx_kernel.h"
#include "all_to_all_matmul_tiling_data.h"

#include "atcoss/block/aiv_comm/collective_comm_api.h"
#include "atcoss/block/aiv_comm/collective_comm_context.h"
#include "atcoss/tiling/comm_tiling_data.h"
#include "atcoss/block/aiv_comm/barrier/barrier_ubmem.h"

#include "all_to_all_asccomm_impl.h"

struct allToAllMatmulCcuTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    CommTilingData commTilingData;
    QuantMatmulTilingData tileQbmmTilingData;
    QuantMatmulTilingData tailQbmmTilingData;
    QuantMatmulTilingData localQbmmTilingData;
    uint32_t localMatmul{0};
    uint32_t bufferSize{0};
    uint64_t hcclDataType{0};
};

namespace Atcoss {

using namespace AscendC;
using namespace Blaze::Gemm;
using Atcoss::Utils::CommTilingData;
using AscendC::Te::Get;
using namespace Atcoss::AivComm;

using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
class AllToAllQuantMatmulCcuImpl {
public:
  __aicore__ inline AllToAllQuantMatmulCcuImpl() {};
  __aicore__ inline ~AllToAllQuantMatmulCcuImpl() {}

  __aicore__ inline void Init(__gm__ CommContext *hcommCtx,
                 GM_ADDR aGM, GM_ADDR scaleAGM,
                 GM_ADDR bGM, GM_ADDR scaleBGM,
                 GM_ADDR cGM,
                 allToAllMatmulCcuTilingData *tilingData,
                 TPipe* pipe);

  __aicore__ inline void Run();

  using TypeA = AType;
  using TypeB = BType;
  using TypeC = CType;
  using TypeScaleA = ::fp8_e8m0_t;
  using TypeScaleB = ::fp8_e8m0_t;
  static constexpr int32_t SCALE_C0 = 2;
  static constexpr uint16_t MTE1_MTE2_EVENT_FLAG = 6;

  using LayoutA = typename AscendC::Std::conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
  using LayoutB = typename AscendC::Std::conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
  using LayoutC = AscendC::Te::NDExtLayoutPtn;
  using LayoutBias = AscendC::Te::NDExtLayoutPtn;
  using BiasType = float;
  using LayoutScaleA = typename AscendC::Te::FrameLayoutFormat<
    AscendC::Std::conditional_t<TransA, AscendC::Te::ScaleADNLayoutPtn, AscendC::Te::ScaleANDLayoutPtn>,
    AscendC::Std::Int<SCALE_C0>>;
  using LayoutScaleB = typename AscendC::Te::FrameLayoutFormat<
    AscendC::Std::conditional_t<TransB, AscendC::Te::ScaleBDNLayoutPtn, AscendC::Te::ScaleBNDLayoutPtn>,
    AscendC::Std::Int<SCALE_C0>>;

  using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape,
    NONE_FULL_LOAD_MODE, LayoutA, LayoutB, TypeA>;
  using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>;
  using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC, BiasType, LayoutBias>;
  using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernelSwat<ProblemShape, BlockMmad, BlockScheduler>;

  using Params = typename QuantMatmulKernelImpl::Params;
  using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
  using L1Params = typename QuantMatmulKernelImpl::L1Params;
  using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
  using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
  using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
  using MatmulMode = typename QuantMatmulKernelImpl::MatmulMode;

  QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
  Atcoss::AivComm::Ccu::CcuAllToAllComm<allToAllMatmulCcuTilingData> allToAll_;
  CommTilingData commTilingDataA_;
  CommTilingData commTilingDataScaleA_;
  uint64_t scaleWinOffset_{0};
  TPipe* pipe_{};

  struct BaseParams {
    GM_ADDR selfWinAddr{};
    GM_ADDR aGm{};
    GM_ADDR scaleAGm{};
    GM_ADDR bGm{};
    GM_ADDR scaleBGm{};
    GM_ADDR cGm{};

    uint32_t rankId{0};
    uint32_t headTileCnt{0};
    uint32_t tailTileCnt{0};
    uint32_t tileCnt{0};
    int32_t  commTurn{0};
    uint32_t axisN{0};
    uint32_t rankSize{0};

    uint64_t axisM{0};
    uint64_t axisKa{0};
    uint64_t commMSize{0};
    uint64_t bufferSize{0};
    uint64_t headMSize{0};
    uint64_t tailMSize{0};
    uint64_t scaleKaSize{0};

    uint64_t bytesPerMRow{0};
    uint64_t bufferBlockSize{0};
  } baseParams_;

  allToAllMatmulCcuTilingData* tilingData_{nullptr};

  __aicore__ inline void InitBaseParams(const allToAllMatmulCcuTilingData *tilingData);
  __aicore__ inline void SetupParams(const QuantMatmulTilingData* mmTile, uint32_t loopIdx,
                     Params& out, MatmulMode matmulMode);

  __aicore__ inline void RunAllToAll();
  __aicore__ inline void LocalMatmulProcess();
  __aicore__ inline void MatmulProcess();

  __aicore__ inline uint64_t CalcMOffset(uint32_t loopIdx) const;
  __aicore__ inline uint64_t GetCurMSize(uint32_t loopIdx) const;
  __aicore__ inline GM_ADDR GetDataAddrGm(uint32_t bufferId) const;
  __aicore__ inline GM_ADDR GetScaleAddrGm() const;
};

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulCcuImpl<AType, BType, CType, TransA, TransB>::Init(
  __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM, GM_ADDR cGM,
  allToAllMatmulCcuTilingData *tilingData, TPipe* pipe)
{
  tilingData_ = tilingData;
  pipe_ = pipe;
  baseParams_.aGm = aGM;
  baseParams_.scaleAGm = scaleAGM;
  baseParams_.bGm = bGM;
  baseParams_.scaleBGm = scaleBGM;
  baseParams_.cGm = cGM;

  auto *udmaCtx = &(hcommCtx->udmaCtx);
  baseParams_.rankId = udmaCtx->rankId;
  baseParams_.rankSize = udmaCtx->rankSize;

  InitBaseParams(tilingData);

  baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx->commBufferAddrs[baseParams_.rankId]);

  commTilingDataA_ = {
    baseParams_.headMSize, static_cast<uint64_t>(baseParams_.headTileCnt),
    baseParams_.tailMSize, static_cast<uint64_t>(baseParams_.tailTileCnt),
    baseParams_.axisKa
  };

  uint64_t scaleKLen = baseParams_.scaleKaSize / sizeof(TypeScaleA);
  commTilingDataScaleA_ = {
    baseParams_.axisM, 1, 0, 0, scaleKLen
  };

  scaleWinOffset_ = baseParams_.rankSize * baseParams_.axisM * baseParams_.axisKa;

  allToAll_.Init(tilingData, baseParams_.aGm, baseParams_.selfWinAddr,
                 commTilingDataA_, tilingData->hcclDataType, sizeof(AType), 0);
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulCcuImpl<AType, BType, CType, TransA, TransB>::Run()
{
  if ASCEND_IS_AIV {
    RunAllToAll();
  }

  if ASCEND_IS_AIC {
    if (tilingData_->localMatmul == 1) {
      LocalMatmulProcess();
    }
    MatmulProcess();
  }
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulCcuImpl<AType, BType, CType, TransA, TransB>::RunAllToAll()
{
  uint32_t totalTileCnt = baseParams_.headTileCnt + baseParams_.tailTileCnt;
  for (uint32_t tid = 0; tid < totalTileCnt; ++tid) {
    CrossCoreWaitFlag<0x2, PIPE_S>(tid);
    if (tid == 0) {
      allToAll_.SetContext(baseParams_.scaleAGm, baseParams_.selfWinAddr,
                           commTilingDataScaleA_, AscendC::HCCL_DATA_TYPE_FP8E8M0,
                           sizeof(TypeScaleA), scaleWinOffset_);
      allToAll_.Commit();
      allToAll_.Wait();
      allToAll_.SetContext(baseParams_.aGm, baseParams_.selfWinAddr,
                           commTilingDataA_, tilingData_->hcclDataType,
                           sizeof(AType), 0);
    }
    allToAll_.Commit();
    allToAll_.Wait();
    AscendC::SyncAll<true>();
    if (baseParams_.bufferSize < totalTileCnt && tid < totalTileCnt - baseParams_.bufferSize) {
      CrossCoreSetFlag<0x2, PIPE_MTE3>(tid);
    }
  }

  allToAll_.Finalize();
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulCcuImpl<AType, BType, CType, TransA, TransB>::InitBaseParams(
  const allToAllMatmulCcuTilingData *tilingData)
{
  baseParams_.headMSize = tilingData->commTilingData.splitAxisTileSize;
  baseParams_.tailMSize = tilingData->commTilingData.splitAxisTailSize;
  baseParams_.headTileCnt = tilingData->commTilingData.splitAxisTileCnt;
  baseParams_.tailTileCnt = tilingData->commTilingData.splitAxisTailCnt;
  baseParams_.tileCnt = tilingData->commTilingData.splitAxisTileCnt
                      + tilingData->commTilingData.splitAxisTailCnt;

  baseParams_.axisM = baseParams_.headTileCnt * baseParams_.headMSize + baseParams_.tailTileCnt * baseParams_.tailMSize;
  baseParams_.axisKa = tilingData->tileQbmmTilingData.k;
  baseParams_.axisN = tilingData->tileQbmmTilingData.n;
  baseParams_.scaleKaSize = CeilDiv(baseParams_.axisKa, Blaze::Gemm::MXFP_DIVISOR_SIZE) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

  baseParams_.commTurn = baseParams_.tileCnt;
  baseParams_.commMSize = baseParams_.headMSize;
  baseParams_.bufferSize = tilingData->bufferSize;

  baseParams_.bytesPerMRow = baseParams_.axisKa * sizeof(AType);
  baseParams_.bufferBlockSize = baseParams_.rankSize * baseParams_.commMSize * baseParams_.bytesPerMRow;
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulCcuImpl<AType, BType, CType, TransA, TransB>::SetupParams(
  const QuantMatmulTilingData* mmTile, uint32_t loopIdx, Params& out, MatmulMode matmulMode)
{
  ProblemShape problemShape{mmTile->m, mmTile->n, mmTile->k, 1UL};
  BlockMmadParams mmadParams;
  LocalParams localParams{baseParams_.rankId, baseParams_.rankSize, baseParams_.axisM, baseParams_.aGm,
    baseParams_.scaleAGm, tilingData_->localMatmul, 1UL, matmulMode, loopIdx};
  L1Params l1Params{static_cast<uint64_t>(mmTile->stepK) * mmTile->baseK, mmTile->scaleKL1,
    mmTile->nBufferNum};

  if (matmulMode == MatmulMode::LOCAL) {
    mmadParams.cGmAddr = baseParams_.cGm;
  } else {
    if (tilingData_->localMatmul == 1) {
      localParams.splitKNum = baseParams_.rankSize - 1;
    } else {
      localParams.splitKNum = baseParams_.rankSize;
    }
  }

  mmadParams.bGmAddr = baseParams_.bGm;
  mmadParams.scaleBGmAddr = baseParams_.scaleBGm;

  BlockSchedulerParams schedulerParams{
    mmTile->baseM, mmTile->baseN, mmTile->mTailTile, mmTile->nTailTile,
    mmTile->mBaseTailSplitCnt, mmTile->nBaseTailSplitCnt, mmTile->mTailMain, mmTile->nTailMain};
  QBMMTiling qbmmParams{mmTile->baseM, mmTile->baseN, mmTile->baseK, mmTile->dbL0c, false};

  out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulCcuImpl<AType, BType, CType, TransA, TransB>::LocalMatmulProcess()
{
  Params localParams;
  SetupParams(&tilingData_->localQbmmTilingData, 0, localParams, MatmulMode::LOCAL);
  quantMatmulKernelImpl_(localParams);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulCcuImpl<AType, BType, CType, TransA, TransB>::MatmulProcess()
{
  Params params;
  bool deferredSync = (tilingData_->localMatmul == 2);
  MatmulMode mode = deferredSync ? MatmulMode::DEFERRED_SYNC : MatmulMode::REMOTE;
  uint32_t totalTileCnt = baseParams_.headTileCnt + baseParams_.tailTileCnt;
  for (uint32_t mLoopIdx = 0; mLoopIdx < totalTileCnt; ++mLoopIdx) {
    if (baseParams_.bufferSize < totalTileCnt && mLoopIdx >= baseParams_.bufferSize) {
      CrossCoreWaitFlag<0x2, PIPE_M>(mLoopIdx - baseParams_.bufferSize);
    }
    const QuantMatmulTilingData* mmTile = (mLoopIdx < baseParams_.headTileCnt)
        ? &tilingData_->tileQbmmTilingData
        : &tilingData_->tailQbmmTilingData;
    SetupParams(mmTile, mLoopIdx, params, mode);

    uint32_t bufferId = mLoopIdx % baseParams_.bufferSize;
    uint64_t mOffset = CalcMOffset(mLoopIdx);

    params.mmadParams.aGmAddr = GetDataAddrGm(bufferId);
    params.mmadParams.scaleAGmAddr = GetScaleAddrGm() + mOffset * baseParams_.scaleKaSize;
    params.mmadParams.cGmAddr = baseParams_.cGm + mOffset * baseParams_.axisN * sizeof(CType);
    params.localParams.localAGmAddr = baseParams_.aGm + mOffset * baseParams_.axisKa;
    params.localParams.localScaleAGmAddr = baseParams_.scaleAGm + mOffset * baseParams_.scaleKaSize;
    params.localParams.mLoopIdx = mLoopIdx;

    if (!deferredSync) {
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(mLoopIdx);
    }
    quantMatmulKernelImpl_(params);

    CrossCoreSetFlag<0x2, PIPE_FIX>(mLoopIdx);

    if (baseParams_.tailTileCnt > 0 && mLoopIdx < totalTileCnt - 1) {
      AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    }
  }
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline uint64_t AllToAllQuantMatmulCcuImpl<AType, BType, CType, TransA, TransB>::CalcMOffset(uint32_t loopIdx) const
{
  if (loopIdx < baseParams_.headTileCnt) {
    return loopIdx * baseParams_.headMSize;
  }
  return baseParams_.headTileCnt * baseParams_.headMSize + (loopIdx - baseParams_.headTileCnt) * baseParams_.tailMSize;
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline uint64_t AllToAllQuantMatmulCcuImpl<AType, BType, CType, TransA, TransB>::GetCurMSize(uint32_t loopIdx) const
{
  if (loopIdx < baseParams_.headTileCnt) {
    return baseParams_.headMSize;
  }
  return baseParams_.tailMSize;
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline GM_ADDR AllToAllQuantMatmulCcuImpl<AType, BType, CType, TransA, TransB>::GetDataAddrGm(uint32_t bufferId) const
{
  return baseParams_.selfWinAddr + bufferId * baseParams_.bufferBlockSize;
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline GM_ADDR AllToAllQuantMatmulCcuImpl<AType, BType, CType, TransA, TransB>::GetScaleAddrGm() const
{
  return baseParams_.selfWinAddr + baseParams_.bufferSize * baseParams_.bufferBlockSize;
}

} // namespace Atcoss

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE4M3E4M3_Ccu(
                          __gm__ Atcoss::AivComm::CommContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulCcuTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;
  Atcoss::AllToAllQuantMatmulCcuImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData, &pipe);
  impl.Run();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE5M2E5M2_Ccu(
                          __gm__ Atcoss::AivComm::CommContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulCcuTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;
  Atcoss::AllToAllQuantMatmulCcuImpl<fp8_e5m2_t, fp8_e5m2_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData, &pipe);
  impl.Run();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE4M3E5M2_Ccu(
                          __gm__ Atcoss::AivComm::CommContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulCcuTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;
  Atcoss::AllToAllQuantMatmulCcuImpl<fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData, &pipe);
  impl.Run();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE5M2E4M3_Ccu(
                          __gm__ Atcoss::AivComm::CommContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulCcuTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;
  Atcoss::AllToAllQuantMatmulCcuImpl<fp8_e5m2_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData, &pipe);
  impl.Run();
}
