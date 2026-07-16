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

namespace Atcoss {

using namespace AscendC;
using namespace Blaze::Gemm;
using Atcoss::Utils::CommTilingData;
using AscendC::Te::Get;
using namespace Atcoss::AivComm;

// 定义问题形状：[M, N, K, Batch]
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

/**
 * @brief All2All-Matmul 核心实现类 (Hcomm GET 版本，计算通信融合)
 * 实现思路：
 * 1. AIC 负责本地计算 和远程数据计算。
 * 2. AIV 负责背景通信，通过 Hcomm GET 接口从其他卡拉取 A 和 ScaleA 数据。
 * 3. 通过流水线掩盖通信开销。
 * 4. 计算与通信逻辑合并于此类，便于精细化流水线控制。
 */
template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
class AllToAllQuantMatmulHcommImpl {
public:
  __aicore__ inline AllToAllQuantMatmulHcommImpl() {};
  __aicore__ inline ~AllToAllQuantMatmulHcommImpl() {}

  /**
   * @brief 初始化算子状态和参数
   */
  __aicore__ inline void Init(__gm__ CommContext *hcommCtx,
                 GM_ADDR aGM, GM_ADDR scaleAGM,
                 GM_ADDR bGM, GM_ADDR scaleBGM,
                 GM_ADDR cGM,
                 const allToAllMatmulTilingData *tilingData,
                 TPipe* pipe);
  /**
   * @brief 执行算子逻辑（包含 AIC/AIV 分离逻辑）
   */
  __aicore__ inline void Run();

  using TypeA = AType;
  using TypeB = BType;
  using TypeC = CType;
  using TypeScaleA = ::fp8_e8m0_t;
  using TypeScaleB = ::fp8_e8m0_t;
  static constexpr int32_t SCALE_C0 = 2;
  static constexpr uint16_t MTE1_MTE2_EVENT_FLAG = 6;

  // Layout 定义
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

  // 组件定义
  using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape,
    NONE_FULL_LOAD_MODE, LayoutA, LayoutB, TypeA>;
  using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<NONE_FULL_LOAD_MODE, false>;
  using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC, BiasType, LayoutBias>;
  using QuantMatmulKernelImpl = Kernel::QuantMatmulMxKernelSwat<ProblemShape, BlockMmad, BlockScheduler>;

  // 参数类型
  using Params = typename QuantMatmulKernelImpl::Params;
  using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
  using L1Params = typename QuantMatmulKernelImpl::L1Params;
  using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
  using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
  using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
  using MatmulMode = typename QuantMatmulKernelImpl::MatmulMode;

  QuantMatmulKernelImpl quantMatmulKernelImpl_;

private:
  // ---------------- 通信相关组件与参数 ----------------
  __gm__ CommUbmemContext* syncBuffer_{};
  __gm__ CommUdmaContext* udmaCtx_{};
  CommTilingData commTilingDataA_;
  CommTilingData commTilingDataScaleA_;
  CollectiveComm<CommCollectiveOp::AllToAll, CommMode::PUT, AType, TeamBarrier, CommAddrLayout::CONTIGUOUS> allToAllA_;
  CollectiveComm<
    CommCollectiveOp::AllToAll, CommMode::PUT, TypeScaleA, TeamBarrier, CommAddrLayout::CONTIGUOUS> allToAllScaleA_;
  TPipe* pipe_{};
  TBuf<TPosition::VECCALC> commBuf_;
  TBuf<TPosition::VECCALC> commScaleBuf_;
  TBuf<TPosition::VECCALC> barrierBuf_;
  TeamBarrier teamBarrier_;

  struct BaseParams {
    GM_ADDR selfWinAddr{}; // 通信窗口地址
    GM_ADDR aGm{};
    GM_ADDR scaleAGm{};
    GM_ADDR bGm{};
    GM_ADDR scaleBGm{};
    GM_ADDR cGm{};

    uint32_t rankId{0};
    uint32_t headTileCnt{0}; // 长块数量
    uint32_t tailTileCnt{0}; // 短块数量
    uint32_t tileCnt{0};     // 总块数量
    int32_t  commTurn{0};     // 总流水步数
    uint32_t axisN{0};     // N 轴大小
    uint32_t rankSize{0};    // 卡数

    uint64_t axisM{0};     // M 轴总大小
    uint64_t axisKa{0};    // K 轴大小
    uint64_t commMSize{0};   // 每次通信的 M 大小
    uint64_t bufferSize{0};   // 缓冲区大小
    uint64_t headMSize{0};   // 长块 M 大小
    uint64_t tailMSize{0};   // 短块 M 大小
    uint64_t scaleKaSize{0};   // Scale 的 K 轴字节长度

    uint64_t bytesPerMRow{0};     // 数据每行的字节数
    uint64_t bufferBlockSize{0};  // 单个 bufferId 块的总字节数
  } baseParams_;

  const allToAllMatmulTilingData* tilingData_{nullptr};

  // ---------------- 私有方法 ----------------
  __aicore__ inline void InitBaseParams(const allToAllMatmulTilingData *tilingData);
  __aicore__ inline void SetupParams(const QuantMatmulTilingData* mmTile, uint32_t loopIdx,
                     Params& out, MatmulMode matmulMode);
  
  __aicore__ inline void RunAllToAll();  // AIV 通信任务
  __aicore__ inline void LocalMatmulProcess(); // AIC 本地计算任务
  __aicore__ inline void MatmulProcess();    // AIC 远程计算任务

  __aicore__ inline uint64_t CalcMOffset(uint32_t loopIdx) const;
  __aicore__ inline uint64_t GetCurMSize(uint32_t loopIdx) const;
  __aicore__ inline GM_ADDR GetDataAddrGm(uint32_t bufferId) const;
  __aicore__ inline GM_ADDR GetScaleAddrGm() const;
};

// =================================================================================
// 公有方法实现
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::Init(
  __gm__ CommContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM, GM_ADDR cGM,
  const allToAllMatmulTilingData *tilingData, TPipe* pipe)
{
  tilingData_ = tilingData;
  pipe_ = pipe;
  baseParams_.aGm = aGM;
  baseParams_.scaleAGm = scaleAGM;
  baseParams_.bGm = bGM;
  baseParams_.scaleBGm = scaleBGM;
  baseParams_.cGm = cGM;

  syncBuffer_ = &(hcommCtx->ubmemCtx);
  udmaCtx_ = &(hcommCtx->udmaCtx);
  baseParams_.rankId = udmaCtx_->rankId;
  baseParams_.rankSize = udmaCtx_->rankSize;

  InitBaseParams(tilingData);

  baseParams_.selfWinAddr = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[baseParams_.rankId]);

  pipe_->InitBuffer(commBuf_, COMM_MAX_RANK_NUM);
  LocalTensor<uint8_t> commTensor = commBuf_.Get<uint8_t>();
  __ubuf__ uint8_t* commUbBuf = (__ubuf__ uint8_t*)commTensor.GetPhyAddr();

  pipe_->InitBuffer(commScaleBuf_, COMM_MAX_RANK_NUM);
  LocalTensor<uint8_t> commScaleTensor = commScaleBuf_.Get<uint8_t>();
  __ubuf__ uint8_t* commScaleUbBuf = (__ubuf__ uint8_t*)commScaleTensor.GetPhyAddr();

  pipe_->InitBuffer(barrierBuf_, UB_SIZE);
  LocalTensor<uint8_t> barrierTensor = barrierBuf_.Get<uint8_t>();
  __ubuf__ uint8_t* barrierUbBuf = (__ubuf__ uint8_t*)barrierTensor.GetPhyAddr();
  teamBarrier_.Init(barrierUbBuf, syncBuffer_, baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));

  commTilingDataA_ = {
    baseParams_.headMSize, static_cast<uint64_t>(baseParams_.headTileCnt),
    baseParams_.tailMSize, static_cast<uint64_t>(baseParams_.tailTileCnt),
    baseParams_.axisKa
  };
  allToAllA_.Init(udmaCtx_, teamBarrier_, commTilingDataA_, baseParams_.aGm, commUbBuf,
    baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));

  uint64_t scaleKLen = baseParams_.scaleKaSize / sizeof(TypeScaleA);
  commTilingDataScaleA_ = {
    baseParams_.axisM, 1, 0, 0,scaleKLen
  };

  uint64_t scaleWinOffset = baseParams_.rankSize * baseParams_.axisM * baseParams_.axisKa;
  allToAllScaleA_.Init(udmaCtx_, teamBarrier_, commTilingDataScaleA_, baseParams_.scaleAGm, commScaleUbBuf,
    baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()), scaleWinOffset);
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::Run()
{
  if ASCEND_IS_AIV {
    RunAllToAll(); // AIV 通信
  }

  if ASCEND_IS_AIC {
    if (tilingData_->localMatmul == 1) {
      LocalMatmulProcess(); // AIC local计算前置
    }
    MatmulProcess();    // AIC 通信块计算（带同步等待）
  }
}

// =================================================================================
// 通信与流水线控制实现 (AIV 侧)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::RunAllToAll()
{
  uint32_t totalTileCnt = baseParams_.headTileCnt + baseParams_.tailTileCnt;
  for (uint32_t tid = 0; tid < totalTileCnt; ++tid) {
    CrossCoreWaitFlag<0x2, PIPE_S>(tid);
    if (tid == 0) {
      allToAllScaleA_.Commit(); // scale一次通信。
    }
    allToAllA_.Commit();
    allToAllA_.Wait();
    if (baseParams_.bufferSize < totalTileCnt && tid < totalTileCnt - baseParams_.bufferSize) {
      CrossCoreSetFlag<0x2, PIPE_MTE3>(tid);
    }
  }

  allToAllScaleA_.Finalize();
  allToAllA_.Finalize();
}

// =================================================================================
// 计算逻辑实现 (AIC 侧)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::InitBaseParams(
  const allToAllMatmulTilingData *tilingData)
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
  // 计算 MXFP8 格式下 Scale 每一行的字节数
  baseParams_.scaleKaSize = CeilDiv(baseParams_.axisKa, Blaze::Gemm::MXFP_DIVISOR_SIZE) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

  baseParams_.commTurn = baseParams_.tileCnt;
  baseParams_.commMSize = baseParams_.headMSize;
  baseParams_.bufferSize = tilingData->bufferSize;

  // 计算地址偏移所需的参数
  baseParams_.bytesPerMRow = baseParams_.axisKa * sizeof(AType);
  baseParams_.bufferBlockSize = baseParams_.rankSize * baseParams_.commMSize * baseParams_.bytesPerMRow;
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::SetupParams(
  const QuantMatmulTilingData* mmTile, uint32_t loopIdx, Params& out, MatmulMode matmulMode)
{
  ProblemShape problemShape{mmTile->m, mmTile->n, mmTile->k, 1UL};
  BlockMmadParams mmadParams;
  // mLoopIdx 由 MatmulProcess 在 per-tile 循环中更新（DEFERRED_SYNC 模式下 kernel 内部要用）
  LocalParams localParams{baseParams_.rankId, baseParams_.rankSize, baseParams_.axisM, baseParams_.aGm,
    baseParams_.scaleAGm, tilingData_->localMatmul, 1UL, matmulMode, loopIdx};
  L1Params l1Params{static_cast<uint64_t>(mmTile->stepK) * mmTile->baseK, mmTile->scaleKL1,
    mmTile->nBufferNum};

  if (matmulMode == MatmulMode::LOCAL) {
    mmadParams.cGmAddr = baseParams_.cGm; // 本地模式直接写入输出
  } else {
    if (tilingData_->localMatmul == 1) {
      localParams.splitKNum = baseParams_.rankSize - 1; // 低精度模式通信块计算只有rankSize-1个远程卡参与
    } else {
      // localMatmul==0（融合）和==2（DEFERRED_SYNC）都需要 splitKNum=rankSize
      localParams.splitKNum = baseParams_.rankSize;
    }
  }

  mmadParams.bGmAddr = baseParams_.bGm;
  mmadParams.scaleBGmAddr = baseParams_.scaleBGm;

  // 调度器参数
  BlockSchedulerParams schedulerParams{
    mmTile->baseM, mmTile->baseN, mmTile->mTailTile, mmTile->nTailTile,
    mmTile->mBaseTailSplitCnt, mmTile->nBaseTailSplitCnt, mmTile->mTailMain, mmTile->nTailMain};
  // 基础 Tiling 参数
  QBMMTiling qbmmParams{mmTile->baseM, mmTile->baseN, mmTile->baseK, mmTile->dbL0c, false};

  out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::LocalMatmulProcess()
{
  Params localParams;
  SetupParams(&tilingData_->localQbmmTilingData, 0, localParams, MatmulMode::LOCAL);
  quantMatmulKernelImpl_(localParams);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::MatmulProcess()
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

    // DEFERRED_SYNC 模式下 wait_flag 由 kernel 内部发出（仅一次，在 self rank mmad 之后）；
    // 其它模式下此处外部 wait_flag 保留原行为。
    if (!deferredSync) {
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(mLoopIdx);
    }
    quantMatmulKernelImpl_(params);

    // 通知通信层：当前计算已完成，空间可释放（针对流水线控制）
    CrossCoreSetFlag<0x2, PIPE_FIX>(mLoopIdx);

    // 非整切场景下，长块和短块使用不同的 tiling data，
    // 需要 FIX 流水完成输出写入后再进入下一个 tile，避免 mmadOp_ 重新 Init 时冲突
    if (baseParams_.tailTileCnt > 0 && mLoopIdx < totalTileCnt - 1) {
      AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    }
  }
}

// =================================================================================
// 辅助方法实现
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline uint64_t AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::CalcMOffset(uint32_t loopIdx) const
{
  if (loopIdx < baseParams_.headTileCnt) {
    return loopIdx * baseParams_.headMSize;
  }
  return baseParams_.headTileCnt * baseParams_.headMSize + (loopIdx - baseParams_.headTileCnt) * baseParams_.tailMSize;
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline uint64_t AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::GetCurMSize(uint32_t loopIdx) const
{
  if (loopIdx < baseParams_.headTileCnt) {
    return baseParams_.headMSize;
  }
  return baseParams_.tailMSize;
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline GM_ADDR AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::GetDataAddrGm(uint32_t bufferId) const
{
  return baseParams_.selfWinAddr + bufferId * baseParams_.bufferBlockSize;
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline GM_ADDR AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::GetScaleAddrGm() const
{
  return baseParams_.selfWinAddr + baseParams_.bufferSize * baseParams_.bufferBlockSize;
}

} // namespace Atcoss

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE4M3E4M3_Hcomm(
                          __gm__ Atcoss::AivComm::CommContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;
  Atcoss::AllToAllQuantMatmulHcommImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData, &pipe);
  impl.Run();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE5M2E5M2_Hcomm(
                          __gm__ Atcoss::AivComm::CommContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;
  Atcoss::AllToAllQuantMatmulHcommImpl<fp8_e5m2_t, fp8_e5m2_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData, &pipe);
  impl.Run();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE4M3E5M2_Hcomm(
                          __gm__ Atcoss::AivComm::CommContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;
  Atcoss::AllToAllQuantMatmulHcommImpl<fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData, &pipe);
  impl.Run();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE5M2E4M3_Hcomm(
                          __gm__ Atcoss::AivComm::CommContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  AscendC::TPipe pipe;
  Atcoss::AllToAllQuantMatmulHcommImpl<fp8_e5m2_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData, &pipe);
  impl.Run();
}
