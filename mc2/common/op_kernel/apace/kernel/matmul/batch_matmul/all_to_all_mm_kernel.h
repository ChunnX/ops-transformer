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
 * \brief 非量化场景 all-to-all MatMul 实现。
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
// splitKIdx 增强版基本 mmad（上游 de3bf6f 基线 + L0C 跨调用累加，apace 本地维护，
// fork 合入上游后切回 "blaze/gemm/block/block_mmad_matmul_basic.h" 并删除该文件）
#include "apace/block/mmad/block_mmad_matmul_basic_splitk.h"
#define WINDOW_LEN 1L // 调度器窗口设置为1，非侵入式修改
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
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
 * 核心机制：
 * 1. 非量化 mmad(block_mmad_matmul_basic.h) 已支持 splitKIdx 接口：同一个输出块由 rankSize 次
 *    mmad 调用在 L0C 上跨调用累加，splitKIdx==0 清零 L0C 并加 bias，splitKIdx==splitKNum-1 触发单次 fixpipe。
 * 2. 调度器使用非量化 BlockSchedulerMatmulBasic：按下标 for 循环遍历（blockIdx 步进 coreNums），
 *    GetBlockShape 返回 {m, n, k_, batch_}（K 即每 rank 的 K 段长度），尾块由 Params 静态切分，
 *    不参与尾块切分的核会收到空子片 {0,0,k_,batch_}，kernel 需 continue 跳过。
 * 3. bias 由 mmad 按 isFirstSplitK 自动只加一次。
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
     *                 DEFERRED_SYNC (per-tile 本地先算驻留 L0C → wait_flag → 远程累加 → 单次 fixpipe)
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
        uint32_t localMatmul; // 0: remote, 1: DEFERRED_SYNC 语义, 2: GM 累加
        MatmulMode matmulMode; // 1: LOCAL, 2: REMOTE, 3: DEFERRED_SYNC
        uint32_t headTileSize;
        bool skipCommWait{false}; // MM_ONLY 测试模式：跳过所有 WaitTile（窗口数据已由 host 预填）
    };

    /**
     * @brief 顶层参数结构
     */
    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmadParams; // 含 splitKNum，由融合接口按模式设置
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

    bool isAtomicAdd_{false}; // GM累加模式（模式3-REMOTE）需要原子累加到 C
    bool isBias_{false};
};

MM_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllToAllMmKernel<MM_KERNEL_FUNC_TEM_PARAMS>::Run(const Params &params)
{
    Init(params);

    if (isAtomicAdd_) {
        AscendC::SetAtomicAdd<CType>(); // 开启原子累加（模式3-REMOTE 单次 fixpipe 累加到 LOCAL 已写的 GM）
    }
    BlockScheduler bs(params.problemShape, params.schParams);

    // GM累加模式（模式3-REMOTE）：bias 已在 LOCAL 阶段写入 GM，mmad 必须跳过 bias，否则重复累加导致精度偏差
    BlockMmadParams mmadParams = params.mmadParams;
    if ((params.localParams.matmulMode == MatmulMode::REMOTE) && (params.localParams.localMatmul == 2)) {
        mmadParams.biasGmAddr = nullptr;
    }
    mmadOp_.Init(mmadParams, isBias_);

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
    // DEFERRED_SYNC: per-tile 内 local 先算驻留 L0C → wait_flag → remote 累加 → 单次 fixpipe。
    // wait_flag 必须由 kernel 内部发出（在 self rank 的 mmad 之后、其它 rank 的 mmad 之前）。
    bool deferredSync = (params.localParams.matmulMode == MatmulMode::DEFERRED_SYNC);

    // 构建各 Tensor 的全局布局
    auto layoutA = MakeLayoutA{}(rankSize * params.localParams.originalM, Te::Get<MNK_K>(params.problemShape));
    auto layoutALocal = MakeLayoutA{}(rankSize * oriM, Te::Get<MNK_K>(params.problemShape));
    auto layoutB = MakeLayoutB{}(rankSize * Te::Get<MNK_K>(params.problemShape), Te::Get<MNK_N>(params.problemShape));
    auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Te::Get<MNK_N>(params.problemShape));
    auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_N>(params.problemShape));

    // 创建 Tensor 句柄
    auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aGmAddr_), layoutA);
    auto gmALocal = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(localAGmAddr_), layoutALocal);
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), layoutB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasGmAddr_), layoutBias);
    auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cGmAddr_), layoutC);

    SetL2Cache(params.problemShape, params.mmTiling.baseM, params.mmTiling.baseN, gmB, gmC);

    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L;
    uint32_t totalTiles = (oriM + params.localParams.headTileSize - 1) / params.localParams.headTileSize;
    int32_t readyTileIdx = -1;
    int64_t curBlockIdx = AscendC::GetBlockIdx();
    if (curBlockIdx >= bs.GetCoreNums()) {
        return; // 核数超出实际所需（如 blockNums < coreNums），直接退出
    }
    int64_t coreNums = AscendC::GetBlockNum();
    // 遍历当前块的调度任务（按下标步进，空子片 continue 跳过）
    for (int64_t blockIdx = curBlockIdx; blockIdx < bs.GetBlockNums(); blockIdx += coreNums) {
        // 非量化调度器返回 {blkM, blkN, k_, batch_}，K 维即当前 rank 的 K 段长度
        auto singleShape = bs.template GetBlockShape<transB, BType>(blockIdx);
        if ((Te::Get<IDX_M_TILEIDX>(singleShape) <= 0) || (Te::Get<IDX_N_TILEIDX>(singleShape) <= 0)) {
            continue; // 空子片：不参与该尾块切分的核，跳过本次迭代
        }
        // GetBlockCoord 必须在 GetBlockShape 之后调用（mSplitOffset_ 由 GetBlockShape 计算）
        auto blockCoord = bs.GetBlockCoord(blockIdx);
        mPos = Te::Get<MNK_M>(blockCoord);
        nPos = Te::Get<MNK_N>(blockCoord);
        // 切分输出块：地址基址已在外部按流水步偏移，此处仅按调度器位置切局部块
        auto gmBlockC = gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos),
                                  AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
        auto gmBlockBias =
            gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, Te::Get<IDX_N_TILEIDX>(singleShape)));

        if (localFirst) {
            // LOCAL 模式：计算本 Rank 的 A 和本 Rank 的 B 对应部分，splitKIdx=0（独立 matmul）
            auto actualMPos = rankId * oriM + mPos;
            auto gmBlockA =
                gmALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                               AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
            auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(rankId * Get<MNK_K>(params.problemShape), nPos),
                                      AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));

            mmadOp_(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, singleShape, nullptr, 0);
        } else if (deferredSync) {
            // DEFERRED_SYNC 模式：
            // Phase 1: 本 rank 的 local A × 本 rank 的 B 段 → L0C（reset，splitKIdx=0）
            //          此处读 GM 的 localAGmAddr_，不依赖通信，可与 AIV 的 UDMA put 并行。
            auto selfMPos = rankId * oriM + mPos;
            auto gmBlockA_self =
                gmALocal.Slice(AscendC::Te::MakeCoord(selfMPos, kPos),
                               AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
            auto gmBlockB_self =
                gmB.Slice(AscendC::Te::MakeCoord(rankId * Get<MNK_K>(params.problemShape), nPos),
                          AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
            mmadOp_(gmBlockA_self, gmBlockB_self, gmBlockBias, gmBlockC, singleShape, nullptr, 0);

            // Phase 2: 在 self rank mmad 之后 wait，阻塞后续 shmem 读（去重：同一 tile 只 wait 一次）
            int64_t blockM = Te::Get<IDX_M_TILEIDX>(singleShape);
            int32_t dependTileIdx = CalcDependTileIdx(mPos + blockM - 1, params.localParams.headTileSize, totalTiles);
            if (!params.localParams.skipCommWait) {
                while (readyTileIdx < dependTileIdx) {
                    readyTileIdx++;
                    commPolicy_.WaitTile(readyTileIdx);
                }
            }
            // Phase 3: 遍历其它 rank，在 L0C 上累加（最后一个 rank 触发 fixpipe）
            uint32_t remoteRankCnt = 1;
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
                mmadOp_(gmBlockA_remote, gmBlockB_r, gmBlockBias, gmBlockC, singleShape, nullptr, remoteRankCnt);
                remoteRankCnt++;
            }
        } else {
            // REMOTE 模式：遍历除本 Rank 外的所有其他卡发送过来的数据，L0C 累加，最后一个 rank 触发 fixpipe
            auto remoteRankCnt = 0UL;
            int64_t blockM = Te::Get<IDX_M_TILEIDX>(singleShape);
            int32_t dependTileIdx = CalcDependTileIdx(mPos + blockM - 1, params.localParams.headTileSize, totalTiles);
            // 等待当前 block 依赖的通信 tile 完成（去重：同一 tile 只 wait 一次）
            if (!params.localParams.skipCommWait) {
                while (readyTileIdx < dependTileIdx) {
                    readyTileIdx++;
                    commPolicy_.WaitTile(readyTileIdx);
                }
            }
            for (uint64_t rank = 0; rank < rankSize; rank++) {
                auto actualMPos = rank * oriM + mPos;
                // 从通信buffer上切片
                auto gmBlockA =
                    gmA.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                              AscendC::Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));

                if (rank == rankId) {
                    if (params.localParams.localMatmul == 2) {
                        continue; // GM累加模式：self rank 已在 RunLocalMatmul 计算，REMOTE 阶段跳过
                    } else {
                        gmBlockA = gmALocal.Slice(AscendC::Te::MakeCoord(actualMPos, kPos),
                                                  AscendC::Te::MakeShape(Get<MNK_M>(singleShape),
                                                                         Get<MNK_K>(params.problemShape)));
                    }
                }

                auto gmBlockB =
                    gmB.Slice(AscendC::Te::MakeCoord(rank * Get<MNK_K>(params.problemShape), nPos),
                              AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));

                // L0C上累加（splitKIdx==0 时 mmad 内部清零 L0C 并加 bias）
                mmadOp_(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, singleShape, nullptr, remoteRankCnt);
                remoteRankCnt++;
            }
        }
    }

    // 存在尾核没用满核，所以这里要等flag兜底
    if (!localFirst) {
        if (!params.localParams.skipCommWait) {
            while (readyTileIdx < totalTiles - 1) {
                readyTileIdx++;
                commPolicy_.WaitTile(readyTileIdx);
            }
        }
    }
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
