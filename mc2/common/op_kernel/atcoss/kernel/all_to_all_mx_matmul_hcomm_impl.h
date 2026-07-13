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
#include "../tiling/quant_matmul_tiling_data.h"
#include "../tiling/all_to_all_matmul_tiling_data.h"

#include "atcoss/block/aiv_comm/all_to_all_udma_get.h"
#include "atcoss/single_core/aiv_comm/asc_comm/asc_comm_context.h"
#include "atcoss/utils/comm_tiling_data.h"

namespace AllToAllQuantMatmulHcommImpl {

using namespace AscendC;
using namespace Blaze::Gemm;
using Atcoss::Utils::CommTilingData;
using AscendC::Te::Get;

// 定义问题形状：[M, N, K, Batch]
using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

/**
 * @brief All2All-Matmul 核心实现类 (Hcomm GET 版本，计算通信融合)
 * 实现思路：
 * 1. AIC 负责本地计算 和远程数据计算。
 * 2. AIV 负责背景通信，通过 Hcomm GET 接口从其他卡拉取 A 和 ScaleA 数据。
 * 3. 通过流水线 掩盖通信开销。
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
  __aicore__ inline void Init(__gm__ CommChannelContext *hcommCtx,
                 GM_ADDR aGM, GM_ADDR scaleAGM,
                 GM_ADDR bGM, GM_ADDR scaleBGM,
                 GM_ADDR cGM,
                 const allToAllMatmulTilingData *tilingData);
  /**
   * @brief 执行算子逻辑（包含 AIC/AIV 分离逻辑）
   */
  __aicore__ inline void Process();

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

  // 矩阵乘模式
  enum class MatmulMode : uint32_t {
    NORMAL = 0,
    LOCAL = 1,
    REMOTE = 2,
    DEFERRED_SYNC = 3
  };

  /**
   * @brief Tiling 配置
   */
  struct QBMMTiling {
    enum BiasMode : uint32_t {
      BIAS_DISABLED = 0,
      BIAS_ENABLED = 1
    };
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t dbL0C;
    uint32_t isBias;
  };

  /**
   * @brief 本地 Rank 相关参数
   */
  struct LocalParams {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t originalM;    // 单卡负责的总 M 行数
    GM_ADDR  localAGmAddr;
    GM_ADDR  localScaleAGmAddr;
    uint32_t localMatmul; // 1 means local matmul
    uint32_t splitKNum;
    MatmulMode matmulMode; // 0: NORMAL, 1: LOCAL, 2: REMOTE, 3: DEFERRED_SYNC
    uint32_t mLoopIdx;     // DEFERRED_SYNC 模式下用于 CrossCoreWaitFlag 的 idx
  };

  // 参数类型定义
  using BlockMmadParams = typename BlockMmad::Params;
  using L1Params = typename BlockMmad::L1Params;
  using BlockSchedulerParams = typename BlockScheduler::Params;

  /**
   * @brief 顶层参数结构
   */
  struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    L1Params l1Params;
    BlockSchedulerParams schParams;
    QBMMTiling qbmmParams;
    LocalParams localParams;
  };

  // 常量与布局别名
  static constexpr bool weightNz = BlockMmad::weightNz;
  static constexpr bool transA = BlockMmad::transA;
  static constexpr bool transB = BlockMmad::transB;
  static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
  static constexpr int64_t kCacheLineAlignMask = IsFp4<AType>() ? 0xff : 0x7f;

  using BlockShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;
  using BlockCoord = Te::Coord<int64_t, int64_t, int64_t, int64_t>;
  using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
  using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
  using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE>>;
  using MakeLayoutScaleA = Std::conditional_t<
    transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<SCALE_C0>>,
    Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<SCALE_C0>>>;
  using MakeLayoutScaleB = Std::conditional_t<
    transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<SCALE_C0>>,
    Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<SCALE_C0>>>;

private:
  // ---------------- 通信相关组件与参数 ----------------
  CommContext commCtx_{};
  HcommComm<HcommCollectiveOp::AllToAll, HcommMode::GET, AType> allToAllA_;
  HcommComm<HcommCollectiveOp::AllToAll, HcommMode::GET, TypeScaleA> allToAllScaleA_;

  struct BaseParams {
    GM_ADDR winGm{};       // 通信窗口地址
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

  // ---------------- 计算相关组件与状态 ----------------
  BlockMmad mmadOp_;

  bool isAtomicAdd_{false}; // REMOTE 模式且 localMatmul==1 时需要原子累加到 C
  bool isBias_{false};
  bool needUpdateTail_{false};

  // ---------------- 私有方法 ----------------
  __aicore__ inline void InitBaseParams(const allToAllMatmulTilingData *tilingData);
  __aicore__ inline void SetupParams(const QuantMatmulTilingData* mmTile, uint32_t loopIdx,
                     Params& out, MatmulMode matmulMode);
  
  __aicore__ inline void AllToAllProcess();  // AIV 通信任务
  __aicore__ inline void LocalMatmulProcess(); // AIC 本地计算任务
  __aicore__ inline void MatmulProcess();    // AIC 远程计算任务

  __aicore__ inline void RunMatmul(const Params &params); // 计算入口
  __aicore__ inline void InitMatmul(const Params &params);
  __aicore__ inline void ProcessSingleBatch(const Params &params, BlockScheduler& bs,
                        uint64_t restBatch, bool isTailRound);

  template <typename TensorB, typename TensorScaleB, typename TensorC>
  __aicore__ inline void SetL2Cache(
    const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB& gmB,
    TensorScaleB &gmScaleB, TensorC& gmC);
  template <typename TensorScaleB>
  __aicore__ inline void SetScaleL2Cache(
    const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB);

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
  __gm__ CommChannelContext *hcommCtx, GM_ADDR aGM, GM_ADDR scaleAGM, GM_ADDR bGM, GM_ADDR scaleBGM, GM_ADDR cGM,
  const allToAllMatmulTilingData *tilingData)
{
  tilingData_ = tilingData;
  baseParams_.aGm = aGM;
  baseParams_.scaleAGm = scaleAGM;
  baseParams_.bGm = bGM;
  baseParams_.scaleBGm = scaleBGM;
  baseParams_.cGm = cGM;

  // 提前获取 rank 信息，供 InitBaseParams 使用
  baseParams_.rankId = hcommCtx->rankId;
  baseParams_.rankSize = hcommCtx->rankSize;
  commCtx_ = {hcommCtx, baseParams_.rankSize, baseParams_.rankId};

  // 从 hcommCtx 获取通信窗口地址
  baseParams_.winGm = reinterpret_cast<GM_ADDR>(hcommCtx->localBufferAddr);

  InitBaseParams(tilingData);

  // 构造 A 矩阵的通信 Tiling (第二维度为 K)
  CommTilingData commTilingDataA = {
    baseParams_.headMSize, static_cast<uint64_t>(baseParams_.headTileCnt),
    baseParams_.tailMSize, static_cast<uint64_t>(baseParams_.tailTileCnt),
    baseParams_.axisKa,
    baseParams_.bufferSize
  };
  allToAllA_.Init(&commCtx_, baseParams_.aGm, commTilingDataA, baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));

  // 构造 ScaleA 的通信 Tiling (第二维度为 scaleKLen)
  uint64_t scaleKLen = baseParams_.scaleKaSize / sizeof(TypeScaleA);
  CommTilingData commTilingDataScaleA = {
    baseParams_.headMSize, static_cast<uint64_t>(baseParams_.headTileCnt),
    baseParams_.tailMSize, static_cast<uint64_t>(baseParams_.tailTileCnt),
    scaleKLen,
    baseParams_.bufferSize
  };
  allToAllScaleA_.Init(&commCtx_, baseParams_.scaleAGm, commTilingDataScaleA, baseParams_.rankSize, static_cast<uint32_t>(GetBlockIdx()));
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::Process()
{
  if ASCEND_IS_AIV {
    AllToAllProcess(); // AIV 通信
  }

  if ASCEND_IS_AIC {
    if (tilingData_->commTilingData.localMatmul == 1) {
      LocalMatmulProcess(); // AIC local计算前置
    }
    MatmulProcess();    // AIC 通信块计算（带同步等待）
  }
}

// =================================================================================
// 通信与流水线控制实现 (AIV 侧)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::AllToAllProcess()
{
  // 先行拉取全局 Scale
  allToAllScaleA_.Commit();
  allToAllScaleA_.Wait();

  for (uint32_t mLoopIdx = 0; mLoopIdx < baseParams_.commTurn; ++mLoopIdx) {
    // 流水线深度控制，防止覆盖尚未被 AIC 读取的旧数据
    if (mLoopIdx >= baseParams_.bufferSize) {
      CrossCoreWaitFlag<0x2, PIPE_MTE3>(mLoopIdx - baseParams_.bufferSize);
    }

    // 拉取当前数据块
    allToAllA_.Commit();
    allToAllA_.Wait();

    // 设置 Flag，通知 AIC 数据已就绪
    CrossCoreSetFlag<0x2, PIPE_MTE3>(mLoopIdx);
  }

  allToAllA_.Finalize();
  allToAllScaleA_.Finalize();
}

// =================================================================================
// 计算逻辑实现 (AIC 侧)
// =================================================================================

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::InitBaseParams(
  const allToAllMatmulTilingData *tilingData)
{
  baseParams_.headMSize = tilingData->commTilingData.headMSize;
  baseParams_.tailMSize = tilingData->commTilingData.tailMSize;
  baseParams_.headTileCnt = tilingData->commTilingData.headTileCnt;
  baseParams_.tailTileCnt = tilingData->commTilingData.tailTileCnt;
  baseParams_.tileCnt = tilingData->commTilingData.tileCnt;

  baseParams_.axisM = baseParams_.headTileCnt * baseParams_.headMSize + baseParams_.tailTileCnt * baseParams_.tailMSize;
  baseParams_.axisKa = tilingData->tileQbmmTilingData.k;
  baseParams_.axisN = tilingData->tileQbmmTilingData.n;
  // 计算 MXFP8 格式下 Scale 每一行的字节数
  baseParams_.scaleKaSize = CeilDiv(baseParams_.axisKa, Blaze::Gemm::MXFP_DIVISOR_SIZE) * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;

  baseParams_.commTurn = baseParams_.tileCnt;
  baseParams_.commMSize = baseParams_.headMSize;
  baseParams_.bufferSize = tilingData->commTilingData.bufferSize;

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
    baseParams_.scaleAGm, tilingData_->commTilingData.localMatmul, 1UL, matmulMode, loopIdx};
  L1Params l1Params{static_cast<uint64_t>(mmTile->stepK) * mmTile->baseK, mmTile->scaleKL1,
    mmTile->nBufferNum};

  if (matmulMode == MatmulMode::LOCAL) {
    mmadParams.cGmAddr = baseParams_.cGm; // 本地模式直接写入输出
  } else {
    if (tilingData_->commTilingData.localMatmul == 1) {
      localParams.splitKNum = baseParams_.rankSize - 1; // 低精度模式通信块计算只有rankSize-1个远程卡参与
    } else {
      // localMatmul==0（融合）和 ==2（DEFERRED_SYNC）都需要 splitKNum=rankSize
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
  RunMatmul(localParams);
  AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::MatmulProcess()
{
  Params params;
  bool deferredSync = (tilingData_->commTilingData.localMatmul == 2);
  MatmulMode mode = deferredSync ? MatmulMode::DEFERRED_SYNC : MatmulMode::REMOTE;
  for (int32_t mLoopIdx = 0; mLoopIdx < baseParams_.tileCnt; ++mLoopIdx) {
    const QuantMatmulTilingData* mmTile = (static_cast<uint32_t>(mLoopIdx) < baseParams_.headTileCnt)
        ? &tilingData_->tileQbmmTilingData
        : &tilingData_->tailQbmmTilingData;
    SetupParams(mmTile, mLoopIdx, params, mode);

    uint32_t bufferId = mLoopIdx & (baseParams_.bufferSize - 1);
    uint64_t mOffset = CalcMOffset(mLoopIdx);

    // 更新当前流水步的地址偏移
    params.mmadParams.aGmAddr = GetDataAddrGm(bufferId);
    params.mmadParams.scaleAGmAddr = GetScaleAddrGm() + mOffset * baseParams_.scaleKaSize;
    params.mmadParams.cGmAddr = baseParams_.cGm + mOffset * baseParams_.axisN * sizeof(CType);
    params.localParams.localAGmAddr = baseParams_.aGm + mOffset * baseParams_.axisKa;
    params.localParams.localScaleAGmAddr = baseParams_.scaleAGm + mOffset * baseParams_.scaleKaSize;
    params.localParams.mLoopIdx = static_cast<uint32_t>(mLoopIdx);

    // DEFERRED_SYNC 模式下 wait_flag 由 kernel 内部发出（仅一次，在 self rank mmad 之后）；
    // 其它模式下此处外部 wait_flag 保留原行为。
    if (!deferredSync) {
      CrossCoreWaitFlag<0x2, PIPE_MTE2>(mLoopIdx);
    }
    
    RunMatmul(params);

    // 通知通信层：当前计算已完成，空间可释放（针对流水线控制）
    if ((baseParams_.commTurn > baseParams_.bufferSize) && (mLoopIdx < (baseParams_.commTurn - baseParams_.bufferSize))) {
      CrossCoreSetFlag<0x2, PIPE_FIX>(mLoopIdx);
    }

    // 非整切场景下，长块和短块使用不同的 tiling data，
    // 需要 FIX 流水完成输出写入后再进入下一个 tile，避免 mmadOp_ 重新 Init 时冲突
    if (baseParams_.tailTileCnt > 0 && mLoopIdx < baseParams_.tileCnt - 1) {
      AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_FLAG);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    }
  }
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::RunMatmul(const Params &params)
{
  InitMatmul(params);

  if (isAtomicAdd_) {
    AscendC::SetAtomicAdd<CType>(); // 开启原子累加
  }
  BlockScheduler bs(params.problemShape, params.schParams);

  BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
  mmadOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias_, params.qbmmParams.dbL0C > 1,
    params.localParams.splitKNum);

  ProcessSingleBatch(params, bs, 0, true);

  if (isAtomicAdd_) {
    AscendC::SetAtomicNone();
  }
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::InitMatmul(const Params &params)
{
  if ASCEND_IS_AIV {
    return;
  }
  if (params.qbmmParams.isBias == QBMMTiling::BIAS_ENABLED) {
    isBias_ = true;
  }

  if ((params.localParams.matmulMode == MatmulMode::REMOTE) && (params.localParams.localMatmul == 1)) {
    isAtomicAdd_ = true; // REMOTE 模式且 localMatmul==1 时需要原子累加
  }
  needUpdateTail_ = false;
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::ProcessSingleBatch(
  const Params &params, BlockScheduler& bs, uint64_t restBatch, bool isTailRound)
{
  auto rankId = params.localParams.rankId;
  auto rankSize = params.localParams.rankSize;
  auto oriM = params.localParams.originalM;
  bool localFirst = ((params.localParams.matmulMode == MatmulMode::LOCAL) && (params.localParams.localMatmul == 1));
  // DEFERRED_SYNC: per-tile 内 local 先算驻留 L0C → wait_flag → remote 累加 → 单次 fixpipe。
  // wait_flag 必须由 kernel 内部发出（在 self rank 的 mmad 之后、其它 rank 的 mmad 之前）。
  bool deferredSync = (params.localParams.matmulMode == MatmulMode::DEFERRED_SYNC);
  auto scaleKLen =
    Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(params.problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
    MXFP_MULTI_BASE_SIZE;
  auto kSize = Te::Get<MNK_K>(params.problemShape);

  // 构建各 Tensor 的全局布局
  auto layoutA = MakeLayoutA{}(rankSize * Te::Get<MNK_M>(params.problemShape), kSize);
  auto layoutALocal = MakeLayoutA{}(rankSize * oriM, kSize);
  auto layoutScaleA = MakeLayoutScaleA{}(rankSize * oriM, scaleKLen);

  auto layoutB = MakeLayoutB{}(rankSize * kSize, Te::Get<MNK_N>(params.problemShape));
  auto layoutScaleB = MakeLayoutScaleB{}(rankSize * scaleKLen, Te::Get<MNK_N>(params.problemShape));
  auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Te::Get<MNK_N>(params.problemShape));
  auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_N>(params.problemShape));

  // 创建 Tensor 句柄 (直接从 params 获取地址，消除冗余成员变量)
  auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
    reinterpret_cast<__gm__ AType*>(params.mmadParams.aGmAddr)), layoutA);
  auto gmALocal = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
    reinterpret_cast<__gm__ AType*>(params.localParams.localAGmAddr)), layoutALocal); // local输入
  auto gmScaleA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
    reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.mmadParams.scaleAGmAddr)), layoutScaleA);
  auto gmScaleALocal = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
    reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.localParams.localScaleAGmAddr)), layoutScaleA);
  auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
    reinterpret_cast<__gm__ BType*>(params.mmadParams.bGmAddr)), layoutB);
  auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
    reinterpret_cast<__gm__ ::fp8_e8m0_t*>(params.mmadParams.scaleBGmAddr)), layoutScaleB);
  auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
    reinterpret_cast<__gm__ BiasType*>(params.mmadParams.biasGmAddr)), layoutBias);
  auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
    reinterpret_cast<__gm__ CType*>(params.mmadParams.cGmAddr)), layoutC);

  // 尾块更新逻辑
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
  // DEFERRED_SYNC 模式下，wait_flag 在第一个 block 的 self-mmad 之后只调用一次。
  // CrossCoreSetFlag/WaitFlag 是 1:1 配对（基于 NotifyEvent/WaitEvent），不能多次 Wait 同一 flagId。
  bool firstBlockWaited = false;
  // 遍历当前块的调度任务
  while (bs.GetTileIdx(blockIdx)) {
    BlockShape singleShape =
      bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
    if ((Te::Get<IDX_M_TILEIDX>(singleShape) <= 0) || (Te::Get<IDX_N_TILEIDX>(singleShape) <= 0)) {
      return;
    }

    bs.GetTileCoord(blockIdx, mPos, nPos);
    // 切分输出块：地址基址已在外部按流水步偏移，此处仅按调度器位置切局部块
    auto gmBlockC =
      gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos),
            AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
    auto gmBlockBias =
      gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, Te::Get<IDX_N_TILEIDX>(singleShape)));

    if (localFirst) {
      // LOCAL 模式：计算本 Rank 的 A 和 本 Rank 的 B 对应部分
      auto actualMPos = rankId * oriM + mPos;
      auto gmBlockA = gmALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), kSize));
      auto gmBlockScaleA = gmScaleALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
      auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(rankId * kSize, nPos), AscendC::Te::MakeShape(kSize, Get<MNK_N>(singleShape)));
      auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(rankId * scaleKLen, nPos), AscendC::Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));

      mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape, 0);
      continue;
    }

    bool localProcessed = false;
    uint32_t remoteRankCnt = 0;

    if (deferredSync) {
      // DEFERRED_SYNC 模式：
      // Phase 1: 本 rank 的 local A × 本 rank 的 B 段 → L0C（reset，remoteRankCnt=0）
      //          此处读 GM 的 localAGmAddr，不依赖通信，可与 AIV 的 Hcomm GET 通信并行。
      auto selfMPos = rankId * oriM + mPos;
      auto gmBlockA_self = gmALocal.Slice(AscendC::Te::MakeCoord(selfMPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), kSize));
      auto gmBlockScaleA_self = gmScaleALocal.Slice(AscendC::Te::MakeCoord(selfMPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
      auto gmBlockB_self = gmB.Slice(AscendC::Te::MakeCoord(rankId * kSize, nPos), AscendC::Te::MakeShape(kSize, Get<MNK_N>(singleShape)));
      auto gmBlockScaleB_self = gmScaleB.Slice(AscendC::Te::MakeCoord(rankId * scaleKLen, nPos), AscendC::Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));
      
      mmadOp_(gmBlockA_self, gmBlockB_self, gmBlockScaleA_self, gmBlockScaleB_self, gmBlockBias, gmBlockC, singleShape, 0);

      // Phase 2: 在第一个 block 处插入 wait_flag（PIPE_MTE2 阻塞后续 shmem 读，
      //          但 self rank 的 MTE2 已 queue 完，self MMAD 可与 wait 并行）
      if (!firstBlockWaited) {
        CrossCoreWaitFlag<0x2, PIPE_MTE2>(params.localParams.mLoopIdx);
        firstBlockWaited = true;
      }
      localProcessed = true;
      remoteRankCnt = 1;
    }

    // Phase 3 / REMOTE 模式：遍历 rank 进行计算
    for (uint64_t rank = 0; rank < rankSize; rank++) {
      if (rank == rankId) {
        if (localProcessed || params.localParams.localMatmul == 1) {
          continue; // DEFERRED_SYNC 已处理，或 REMOTE 模式跳过本卡
        }
      }

      auto actualMPos = rank * oriM + mPos;
      auto actualCommMPos = rank * Te::Get<MNK_M>(params.problemShape) + mPos;
      
      // 根据是否为本卡，选择从本地 GM 或通信 Buffer 切片
      auto gmBlockA = (rank == rankId) ? 
        gmALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), kSize)) :
        gmA.Slice(AscendC::Te::MakeCoord(actualCommMPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), kSize));
      
      auto gmBlockScaleA = (rank == rankId) ?
        gmScaleALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen)) :
        gmScaleA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos), AscendC::Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));

      auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(rank * kSize, nPos), AscendC::Te::MakeShape(kSize, Get<MNK_N>(singleShape)));
      auto gmBlockScaleB = gmScaleB.Slice(AscendC::Te::MakeCoord(rank * scaleKLen, nPos), AscendC::Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));

      // L0C上累加
      mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape, remoteRankCnt);
      remoteRankCnt++;
    }
  }
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
template <typename TensorScaleB>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::SetScaleL2Cache(
  const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB)
{
  if (Te::Get<MNK_B>(problemShape) != 1) {
    return;
  }
  if constexpr (transB) {
    const int64_t scaleKRowBytes =
      Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
      MXFP_MULTI_BASE_SIZE;
    const int64_t scaleKL1RowBytes = Blaze::Gemm::CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    // 0x7f: 128B cache line alignment for mx scale GM streaming
    const bool scaleAlignForL2Stream = (scaleKRowBytes & kCacheLineAlignMask) == 0 &&
      (scaleKL1RowBytes & kCacheLineAlignMask) == 0;
    gmScaleB.SetL2CacheHint(
      scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
  } else {
    const int64_t scaleNStrideBytes = Te::Get<MNK_N>(problemShape) * MXFP_MULTI_BASE_SIZE;
    const int64_t scaleBaseNStrideBytes = baseN * MXFP_MULTI_BASE_SIZE;
    // 0x7f: 128B cache line alignment for mx scale GM streaming
    const bool scaleAlignForL2Stream = (scaleNStrideBytes & kCacheLineAlignMask) == 0 &&
      (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
    gmScaleB.SetL2CacheHint(
      scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
  }
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
template <typename TensorB, typename TensorScaleB, typename TensorC>
__aicore__ inline void AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::SetL2Cache(
  const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB& gmB,
  TensorScaleB &gmScaleB, TensorC& gmC)
{
  if (isAtomicAdd_) {
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
  return baseParams_.winGm + bufferId * baseParams_.bufferBlockSize;
}

template<typename AType, typename BType, typename CType, bool TransA, bool TransB>
__aicore__ inline GM_ADDR AllToAllQuantMatmulHcommImpl<AType, BType, CType, TransA, TransB>::GetScaleAddrGm() const
{
  return baseParams_.winGm + baseParams_.bufferSize * baseParams_.bufferBlockSize;
}

} // namespace AllToAllQuantMatmulHcommImpl

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE4M3E4M3_Hcomm(__gm__ CommChannelContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  AllToAllQuantMatmulHcommImpl::AllToAllQuantMatmulHcommImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData);
  impl.Process();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE5M2E5M2_Hcomm(__gm__ CommChannelContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  AllToAllQuantMatmulHcommImpl::AllToAllQuantMatmulHcommImpl<fp8_e5m2_t, fp8_e5m2_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData);
  impl.Process();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE4M3E5M2_Hcomm(__gm__ CommChannelContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  AllToAllQuantMatmulHcommImpl::AllToAllQuantMatmulHcommImpl<fp8_e4m3fn_t, fp8_e5m2_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData);
  impl.Process();
}

[[bisheng::core_ratio(1, 1)]]
__global__ __aicore__ __schedmode__(1) void  AllToAllQuantMatmulKernelE5M2E4M3_Hcomm(__gm__ CommChannelContext *hcommCtx,
                           GM_ADDR aGM, GM_ADDR scaleAGM,
                           GM_ADDR bGM, GM_ADDR scaleBGM,
                           GM_ADDR cGM,
                           allToAllMatmulTilingData tilingData)
{
  AllToAllQuantMatmulHcommImpl::AllToAllQuantMatmulHcommImpl<fp8_e5m2_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, cGM, &tilingData);
  impl.Process();
}