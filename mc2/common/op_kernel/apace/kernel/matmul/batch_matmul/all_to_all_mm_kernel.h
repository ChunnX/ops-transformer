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
 * \file all_to_all_mm_kernel.h
 * \brief 非量化场景 all-to-all MatMul 实现（GM 累加版）。
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#define WINDOW_LEN 1L // 调度器窗口设置为1，非侵入式修改
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#undef WINDOW_LEN

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define MM_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler, class CommPolicy>
#define MM_KERNEL_FUNC_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler, CommPolicy

using namespace AscendC;
using AscendC::Te::Get;

/**
 * @brief 非量化 all-to-all MatMul 内核实现
 * 该类负责具体的矩阵乘块调度和计算，支持本地(LOCAL)和远程(REMOTE)两种切片模式。
 * 与量化版(all_to_all_qbmm_mx_kernel.h)的差异：
 * 1. 非量化 mmad(block_mmad_matmul_basic.h)为自包含 matmul，无 splitKIdx 接口，无法跨调用保持 L0C 状态，
 *    因此统一采用 GM 累加：第一次写入覆盖写（atomic 关），其余写入原子累加（atomic 开）。
 * 2. 调度器复用 BlockSchedulerQuantBatchMatmulV3（调度逻辑与量化无关），QuantMode 传 DEFAULT 关闭量化对齐。
 * 3. 基本 mmad 每次调用都会累加 bias，kernel 侧按"首次调用真 bias / 其余调用 zero-bias"控制（见 LocalParams.zeroBiasGmAddr）。
 * 4. 通信等待逻辑经 CommPolicy 策略类注入（组合模式）：调用点直接 commPolicy_.WaitTile(tileIdx)，
 *    编译期由模板参数绑定具体策略，无需继承与 static_cast。
 */

template <class ProblemShape, class BlockMmad, class BlockScheduler, class CommPolicy>
class AllToAllMmKernel {
public:
    __aicore__ inline AllToAllMmKernel() {}
    __aicore__ inline ~AllToAllMmKernel() {}

    static constexpr bool weightNz = BlockMmad::WEIGHT_NZ_FORMAT;
    static constexpr bool transA = BlockMmad::TRANS_A;
    static constexpr bool transB = BlockMmad::TRANS_B;

    using BlockMmadParams = typename BlockMmad::Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutBias = typename BlockMmad::LayoutBias;
    static constexpr int64_t kCacheLineAlignMask = 0x7f;

    using BlockShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;
    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<Te::C0_ELEMENT<AType>>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<Te::C0_ELEMENT<BType>>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<Te::C0_ELEMENT<CType>>>;
    /**
     * @brief 算子模式：REMOTE (远程同步数据后计算), LOCAL (仅本地计算),
     *                 DEFERRED_SYNC (per-tile 本地先算写 GM → wait_flag → 远程原子累加)
     */
    enum class MatmulMode : uint32_t {
        REMOTE = 0,
        LOCAL = 1,
        DEFERRED_SYNC = 2
    };
    /**
     * @brief Tiling 配置
     */
    struct MMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
    };

    /**
     * @brief 本地 Rank 相关参数
     */
    struct LocalParams {
        uint32_t rankId;
        uint32_t rankSize;
        uint64_t originalM; // 单卡负责的总 M 行数
        GM_ADDR localAGmAddr; // 本地输入基址
        GM_ADDR zeroBiasGmAddr; // zero-bias workspace（baseN 大小，tiling 分配并清零，kernel 只读）
        uint32_t localMatmul; // 0: remote, 1: DEFERRED_SYNC 语义, 2: GM 累加
        MatmulMode matmulMode; // 1: LOCAL, 2: REMOTE, 3: DEFERRED_SYNC
        uint32_t headTileSize;
    };

    /**
     * @brief 顶层参数结构
     */
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams;
        BlockSchedulerParams schParams;
        MMTiling mmTiling;
        LocalParams localParams;
    };

public:
    __aicore__ inline void Init(const Params &params);
    __aicore__ inline void Run(const Params &params);
    __aicore__ inline void operator()(const Params &params)
    {
        Run(params);
    }

    __aicore__ inline CommPolicy &GetCommPolicy()
    {
        return commPolicy_;
    }

private:
    __aicore__ inline void ResetGmAddr(const Params &params);
    __aicore__ inline void ProcessSingleBatch(const Params &params, BlockScheduler &bs, uint64_t restBatch,
                                              bool isTailRound);
    __aicore__ inline int32_t CalcDependTileIdx(int64_t mPos, uint32_t headTileSize, uint32_t totalTiles) const;

    template <typename TensorB, typename TensorC>
    __aicore__ inline void SetL2Cache(const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN,
                                      TensorB &gmB, TensorC &gmC);

private:
    BlockMmad mmadOp_;
    CommPolicy commPolicy_;

    __gm__ AType *aGmAddr_; // 远程数据基址（通信缓冲区）
    __gm__ AType *localAGmAddr_; // 本地数据基址
    __gm__ BType *bGmAddr_;
    __gm__ CType *cGmAddr_; // 输出基址（已根据流水步偏移）
    __gm__ BiasType *biasGmAddr_ = nullptr;
    __gm__ BiasType *zeroBiasGmAddr_ = nullptr;

    bool isAtomicAdd_{false}; // GM累加模式下需要原子累加到 C
    bool isBias_{false};
    bool needUpdateTail_{false};
};

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::Run(const Params &params)
{
    Init(params);

    if (isAtomicAdd_) {
        AscendC::SetAtomicAdd<CType>(); // 开启原子累加
    }
    BlockScheduler bs(params.problemShape, params.schParams);

    // GM累加模式（模式3-REMOTE）：bias 已在 LOCAL 阶段写入 GM，mmad 必须跳过 bias，否则重复累加导致精度偏差
    BlockMmadParams mmadParams = params.mmadParams;
    if ((params.localParams.matmulMode == MatmulMode::REMOTE) && (params.localParams.localMatmul == 2)) {
        mmadParams.biasGmAddr = nullptr;
    }
    mmadOp_.Init(mmadParams);

    ProcessSingleBatch(params, bs, 0, true);

    if (isAtomicAdd_) {
        AscendC::SetAtomicNone();
    }
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::Init(const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }
    isBias_ = (params.mmadParams.biasGmAddr != nullptr);

    if ((params.localParams.matmulMode == MatmulMode::REMOTE) && (params.localParams.localMatmul == 2)) {
        isAtomicAdd_ = true; // GM累加模式（local已写GM）需要原子累加
        isBias_ = false; // GM累加模式：bias 已在 RunLocalMatmul(LOCAL) 阶段写入 GM，
                         // REMOTE 阶段必须跳过 bias，否则会重复累加 bias 导致精度偏差
    }
    needUpdateTail_ = false;
    ResetGmAddr(params);
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::ResetGmAddr(const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmAddr_ = reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType *>(params.mmadParams.cGmAddr);
    localAGmAddr_ = reinterpret_cast<__gm__ AType *>(params.localParams.localAGmAddr);
    biasGmAddr_ = reinterpret_cast<__gm__ BiasType *>(params.mmadParams.biasGmAddr);
    zeroBiasGmAddr_ = reinterpret_cast<__gm__ BiasType *>(params.localParams.zeroBiasGmAddr);
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline int32_t AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::CalcDependTileIdx(
    int64_t mPos, uint32_t headTileSize, uint32_t totalTiles) const
{
    int32_t tileIdx = static_cast<int32_t>(mPos / headTileSize);
    if (tileIdx >= totalTiles) {
        tileIdx = totalTiles - 1;
    }
    return tileIdx;
}

MM_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorC>
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::SetL2Cache(const ProblemShape &problemShape,
                                                                               uint64_t curBaseM, uint64_t baseN,
                                                                               TensorB &gmB, TensorC &gmC)
{
    if (isAtomicAdd_) {
        gmC.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
    }

    const bool fullMTile = curBaseM >= Te::Get<MNK_M>(problemShape);
    if (!fullMTile) {
        return;
    }

    if constexpr (weightNz) {
        gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
    } else {
        if constexpr (transB) {
            bool bAlignForL2Stream = (Te::Get<MNK_K>(problemShape) & kCacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(bAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE :
                                                   Te::CacheMode::CACHE_MODE_NORMAL);
        } else {
            bool bAlignForL2Stream =
                (Te::Get<MNK_N>(problemShape) & kCacheLineAlignMask) == 0 && (baseN & kCacheLineAlignMask) == 0;
            gmB.SetL2CacheHint(bAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE :
                                                   Te::CacheMode::CACHE_MODE_NORMAL);
        }
    }
}

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::ProcessSingleBatch(const Params &params,
                                                                                        BlockScheduler &bs,
                                                                                        uint64_t restBatch,
                                                                                        bool isTailRound)
{
    auto rankId = params.localParams.rankId;
    auto rankSize = params.localParams.rankSize;
    auto oriM = params.localParams.originalM;
    bool localFirst = ((params.localParams.matmulMode == MatmulMode::LOCAL) && (params.localParams.localMatmul == 2));
    // DEFERRED_SYNC: per-tile 内 local 先算写 GM → wait_flag → remote 原子累加。
    // wait_flag 必须由 kernel 内部发出（在 self rank 的 mmad 之后、其它 rank 的 mmad 之前）。
    bool deferredSync = (params.localParams.matmulMode == MatmulMode::DEFERRED_SYNC);

    // 构建各 Tensor 的全局布局
    auto layoutA = MakeLayoutA{}(rankSize * params.localParams.originalM, Te::Get<MNK_K>(params.problemShape));
    auto layoutALocal = MakeLayoutA{}(rankSize * oriM, Te::Get<MNK_K>(params.problemShape));
    auto layoutB = MakeLayoutB{}(rankSize * Te::Get<MNK_K>(params.problemShape), Te::Get<MNK_N>(params.problemShape));
    auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Te::Get<MNK_N>(params.problemShape));
    auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_N>(params.problemShape));
    // zero-bias workspace 全 0，切片始终从 (0,0) 出发，位置无关
    auto layoutZeroBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, params.mmTiling.baseN);

    // 创建 Tensor 句柄
    auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aGmAddr_), layoutA);
    auto gmALocal = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(localAGmAddr_), layoutALocal);
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), layoutB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasGmAddr_), layoutBias);
    auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cGmAddr_), layoutC);
    auto gmZeroBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(zeroBiasGmAddr_), layoutZeroBias);

    // 尾块更新逻辑
    auto &mTailTile = params.schParams.mTailTile;
    auto &nTailTile = params.schParams.nTailTile;
    if (needUpdateTail_ ||
        (isTailRound && ((bs.GetEndBlockIdx() + 1) + (restBatch * bs.GetTotalCnt())) * mTailTile * nTailTile <=
                            AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    SetL2Cache(params.problemShape, params.mmTiling.baseM, params.mmTiling.baseN, gmB, gmC);

    BlockCoord blockIdx;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L;
    uint32_t totalTiles = (oriM + params.localParams.headTileSize - 1) / params.localParams.headTileSize;
    int32_t readyTileIdx = -1;
    // 遍历当前块的调度任务
    while (bs.GetTileIdx(blockIdx)) {
        // 复用量化调度器，QuantMode 传 DEFAULT 关闭量化对齐逻辑
        auto schedShape = bs.template GetBlockShape<QuantMode::DEFAULT, QuantMode::DEFAULT, weightNz>(blockIdx);
        if ((Te::Get<IDX_M_TILEIDX>(schedShape) <= 0) || (Te::Get<IDX_N_TILEIDX>(schedShape) <= 0)) {
            return;
        }
        bs.GetTileCoord(blockIdx, mPos, nPos);
        // 量化调度器 dims2/3 为 split offset，非量化 mmad 需要 blockShape 的 K 维为当前 rank 的 K 段长度
        BlockShape singleShape{Te::Get<IDX_M_TILEIDX>(schedShape), Te::Get<IDX_N_TILEIDX>(schedShape),
                               Te::Get<MNK_K>(params.problemShape), 0L};
        // 切分输出块：地址基址已在外部按流水步偏移，此处仅按调度器位置切局部块
        auto gmBlockC = gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos),
                                  AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
        auto gmBlockBias =
            gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, Te::Get<IDX_N_TILEIDX>(singleShape)));
        auto gmBlockZeroBias =
            gmZeroBias.Slice(Te::MakeCoord(0L, 0L), Te::MakeShape(1L, Te::Get<IDX_N_TILEIDX>(singleShape)));

        if (localFirst) {
            // LOCAL 模式：计算本 Rank 的 A 和本 Rank 的 B 对应部分，覆盖写 GM（atomic 关）
            auto actualMPos = rankId * oriM + mPos;
            auto gmBlockA =
                gmALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                               AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(rankId * Get<MNK_K>(params.problemShape), nPos),
                                      AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape),
                                                             Get<MNK_N>(singleShape)));

            mmadOp_(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, singleShape);
        } else if (deferredSync) {
            // DEFERRED_SYNC 模式（GM 累加版）：
            // Phase 1: 本 rank 的 local A × 本 rank 的 B 段 → GM 覆盖写
            //          此处读 GM 的 localAGmAddr_，不依赖通信，可与 AIV 的 UDMA put 并行。
            auto selfMPos = rankId * oriM + mPos;
            auto gmBlockA_self =
                gmALocal.Slice(AscendC::Te::MakeCoord(selfMPos, kPos),
                               AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
            auto gmBlockB_self =
                gmB.Slice(AscendC::Te::MakeCoord(rankId * Get<MNK_K>(params.problemShape), nPos),
                          AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
            mmadOp_(gmBlockA_self, gmBlockB_self, gmBlockBias, gmBlockC, singleShape);

            // Phase 2: 在 self rank mmad 之后 wait，阻塞后续 shmem 读（去重：同一 tile 只 wait 一次）
            int64_t blockM = Te::Get<IDX_M_TILEIDX>(singleShape);
            int32_t dependTileIdx = CalcDependTileIdx(mPos + blockM - 1, params.localParams.headTileSize, totalTiles);
            while (readyTileIdx < dependTileIdx) {
                readyTileIdx++;
                commPolicy_.WaitTile(readyTileIdx);
            }
            // Phase 3: 遍历其它 rank，在 GM 上原子累加（zero-bias）
            AscendC::PipeBarrier<PIPE_ALL>(); // 确保覆盖写 fixpipe 完成后开启原子累加（dav-3510 无 PIPE_FIX）
            AscendC::SetAtomicAdd<CType>();
            for (uint64_t rank = 0; rank < rankSize; rank++) {
                if (rank == rankId)
                    continue;
                auto actualMPos = rank * oriM + mPos;
                auto gmBlockA_remote =
                    gmA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                              AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
                auto gmBlockB_r =
                    gmB.Slice(AscendC::Te::MakeCoord(rank * Get<MNK_K>(params.problemShape), nPos),
                              AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
                mmadOp_(gmBlockA_remote, gmBlockB_r, gmBlockZeroBias, gmBlockC, singleShape);
            }
            AscendC::PipeBarrier<PIPE_ALL>(); // 确保最后原子 fixpipe 完成后关闭原子累加（dav-3510 无 PIPE_FIX）
            AscendC::SetAtomicNone();
        } else {
            // REMOTE 模式：等待通信后遍历各 rank 的 A × B
            int64_t blockM = Te::Get<IDX_M_TILEIDX>(singleShape);
            int32_t dependTileIdx = CalcDependTileIdx(mPos + blockM - 1, params.localParams.headTileSize, totalTiles);
            // 等待当前 block 依赖的通信 tile 完成（去重：同一 tile 只 wait 一次）
            while (readyTileIdx < dependTileIdx) {
                readyTileIdx++;
                commPolicy_.WaitTile(readyTileIdx);
            }
            if (isAtomicAdd_) {
                // GM累加模式（localMatmul==2）：self 已由 LOCAL 阶段写入 GM，此处全部原子累加
                for (uint64_t rank = 0; rank < rankSize; rank++) {
                    if (rank == rankId)
                        continue;
                    auto actualMPos = rank * oriM + mPos;
                    auto gmBlockA =
                        gmA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                                  AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
                    auto gmBlockB =
                        gmB.Slice(AscendC::Te::MakeCoord(rank * Get<MNK_K>(params.problemShape), nPos),
                                  AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
                    // mmad 已跳过 bias（isBias_=false），此处传 zero-bias 即可
                    mmadOp_(gmBlockA, gmBlockB, gmBlockZeroBias, gmBlockC, singleShape);
                }
            } else {
                // 远程融合模式（localMatmul==0）：首次调用覆盖写 + 真 bias，其余原子累加 + zero-bias
                bool isFirstWrite = true;
                for (uint64_t rank = 0; rank < rankSize; rank++) {
                    auto actualMPos = rank * oriM + mPos;
                    // 从通信buffer上切片
                    auto gmBlockA =
                        gmA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                                  AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));

                    if (rank == rankId) {
                        gmBlockA = gmALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                                                  AscendC::Te::MakeShape(Get<MNK_M>(singleShape),
                                                                         Get<MNK_K>(params.problemShape)));
                    }

                    auto gmBlockB =
                        gmB.Slice(AscendC::Te::MakeCoord(rank * Get<MNK_K>(params.problemShape), nPos),
                                  AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));

                    // 首次写入覆盖写（atomic 关），其后原子累加（atomic 开）
                    if (isFirstWrite) {
                        mmadOp_(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, singleShape);
                        isFirstWrite = false;
                        AscendC::PipeBarrier<PIPE_ALL>(); // 确保覆盖写 fixpipe 完成后开启原子累加（dav-3510 无 PIPE_FIX）
                        AscendC::SetAtomicAdd<CType>();
                    } else {
                        mmadOp_(gmBlockA, gmBlockB, gmBlockZeroBias, gmBlockC, singleShape);
                    }
                }
                AscendC::PipeBarrier<PIPE_ALL>(); // 确保最后原子 fixpipe 完成后关闭原子累加（dav-3510 无 PIPE_FIX）
                AscendC::SetAtomicNone();
            }
        }
    }

    // 存在尾核没用满核，所以这里要等flag兜底
    if (!localFirst) {
        while (readyTileIdx < totalTiles - 1) {
            readyTileIdx++;
            commPolicy_.WaitTile(readyTileIdx);
        }
    }
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
