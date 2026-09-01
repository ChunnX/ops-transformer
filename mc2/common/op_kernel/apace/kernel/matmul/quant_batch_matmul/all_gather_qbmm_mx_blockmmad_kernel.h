/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_qbmm_mx_blockmmad_kernel.h
 * \brief AllGather + MX 量化 Matmul 计算内核 — 直接调用 ops-tensor(Blaze) BlockMmad 实现
 *
 * 与 FragmentTensor 版本 (all_gather_qbmm_mx_kernel.h) 的核心差异：
 * 计算侧不构建 HEAD/MAIN/TAIL 虚拟 M 空间与散片 fragment，而是把通信产出的 gather buffer
 * 视为连续的 [rankSize * mPerRank, K] 大矩阵，调度器直接在其上做标准 GEMM 切分；
 * 每个 tile 依据其覆盖的最后一行所属的通信轮次，经 CommPolicy 策略等待通信完成，
 * 随后单次调用 BlockMmad 完成 A(整 K) x B(整 K) 计算（无跨 rank L0C 累加）。
 *
 * 通信等待语义（WaitTile 约定，与 AllToAllQbmmMxKernel 一致的单调递增去重等待）：
 *   dependTileIdx = 0          : 无依赖（不发生：gather 行恒属某个通信轮次，仅在无尾块且行属本地段时可能为 0 前缀）
 *   dependTileIdx = 1..tileCnt : 依赖 dataHead 第 (dependTileIdx-1) 轮
 *   dependTileIdx > tileCnt    : 依赖 dataTail 第 (dependTileIdx-tileCnt-1) 轮
 * 由于调度器 M 轴单调推进且等待按 readyTileIdx 去重递增，同一轮次仅 Wait 一次。
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
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#define WINDOW_LEN 1L // 调度器窗口设置为1，非侵入式修改
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#undef WINDOW_LEN

namespace Blaze {
namespace Gemm {
namespace Kernel {

#define AG_BMMA_KERNEL_CLASS_TEM_PARAMS \
    template <class ProblemShape, class BlockMmad, class BlockScheduler, class CommPolicy>
#define AG_BMMA_KERNEL_FUNC_TEM_PARAMS ProblemShape, BlockMmad, BlockScheduler, CommPolicy

using namespace AscendC;
using AscendC::Te::Get;

/**
 * @brief AllGather + MX 量化矩阵乘内核实现（BlockMmad 直调版）
 * 该类负责在 gather buffer（连续 [rankSize*mPerRank, K] 布局）上的矩阵乘块调度和计算。
 * 通信等待逻辑经 CommPolicy 策略类注入（组合模式）：基类持 commPolicy_ 成员对象，
 * 调用点直接 commPolicy_.WaitTile(tileIdx)，编译期由模板参数绑定具体策略。
 */
template <class ProblemShape, class BlockMmad, class BlockScheduler, class CommPolicy>
class AllGatherQbmmMxBlockMmadKernel {
public:
    __aicore__ inline AllGatherQbmmMxBlockMmadKernel() {}
    __aicore__ inline ~AllGatherQbmmMxBlockMmadKernel() {}

    static constexpr bool weightNz = BlockMmad::WEIGHT_NZ;
    static constexpr bool transA = BlockMmad::TRANS_A;
    static constexpr bool transB = BlockMmad::TRANS_B;

    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t kCacheLineAlignMask = IsFp4<AType>() ? 0xff : 0x7f;
    static constexpr int32_t SCALE_C0 = 2;

    using BlockShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    using BlockSchedulerParams = typename BlockScheduler::Params;
    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE>>;
    using MakeLayoutScaleA =
        Std::conditional_t<transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<SCALE_C0>>,
                           Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<SCALE_C0>>>;
    using MakeLayoutScaleB =
        Std::conditional_t<transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<SCALE_C0>>,
                           Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<SCALE_C0>>>;

    /**
     * @brief Tiling 配置（对标 blaze QBMMTiling）
     */
    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0c;
        uint32_t isBias;
    };

    /**
     * @brief AllGather 通信布局相关参数
     */
    struct LocalParams {
        uint32_t rankSize;   // 通信 rank 数
        uint64_t mPerRank;   // 每 rank 的 M 行数（gather buffer 按 rank 顺序拼接）
        uint32_t tileM;      // dataHead 单轮通信覆盖的行数
        uint32_t tileCnt;    // dataHead 通信轮数
        uint32_t tailM;      // dataTail 单轮通信覆盖的行数
        uint32_t tailCnt;    // dataTail 通信轮数
        uint32_t commTurn;   // 总通信轮数 = tileCnt + tailCnt
    };

    /**
     * @brief 顶层参数结构
     */
    struct Params {
        ProblemShape problemShape; // M = rankSize * mPerRank（gather 后总行数）
        BlockMmadParams mmadParams;
        L1Params l1Params;
        BlockSchedulerParams schParams;
        QBMMTiling qbmmParams;
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
    __aicore__ inline void ProcessSingleBatch(const Params &params, BlockScheduler &bs);
    __aicore__ inline uint32_t CalcDependTileIdx(int64_t lastRow, const LocalParams &local) const;

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN,
                                      uint64_t scaleKL1, TensorB &gmB, TensorScaleB &gmScaleB);

    template <typename TensorScaleB>
    __aicore__ inline void SetScaleL2Cache(const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1,
                                           TensorScaleB &gmScaleB);

private:
    BlockMmad mmadOp_;
    CommPolicy commPolicy_;

    __gm__ AType *aGmAddr_;      // gather buffer 上的 A 基址（连续 [rankSize*mPerRank, K]）
    __gm__ BType *bGmAddr_;      // 本卡权重 B 基址（全量 K）
    __gm__ CType *cGmAddr_;      // 输出基址（[rankSize*mPerRank, N]）
    __gm__ BiasType *biasGmAddr_ = nullptr;
    __gm__ ::fp8_e8m0_t *scaleAGmAddr_; // gather buffer 上的 A scale 基址
    __gm__ ::fp8_e8m0_t *scaleBGmAddr_;

    bool isBias_{false};
    bool needUpdateTail_{false};
};

AG_BMMA_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxBlockMmadKernel<AG_BMMA_KERNEL_FUNC_TEM_PARAMS>::Run(const Params &params)
{
    Init(params);

    const auto &local = params.localParams;
    // 只通信不计算（N=0）或空 M 场景：跳过调度与 matmul，仅按轮次等待通信完成后返回
    if ((Te::Get<MNK_N>(params.problemShape) <= 0) || (Te::Get<MNK_M>(params.problemShape) <= 0)) {
        uint32_t readyTileIdx = 0;
        while (readyTileIdx < local.commTurn) {
            readyTileIdx++;
            commPolicy_.WaitTile(readyTileIdx);
        }
        return;
    }

    BlockScheduler bs(params.problemShape, params.schParams);

    BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
    mmadOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias_, params.qbmmParams.dbL0c > 1, 1);

    ProcessSingleBatch(params, bs);
}

AG_BMMA_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxBlockMmadKernel<AG_BMMA_KERNEL_FUNC_TEM_PARAMS>::Init(const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }
    isBias_ = (params.mmadParams.biasGmAddr != nullptr) && (params.qbmmParams.isBias != 0);
    needUpdateTail_ = false;
    ResetGmAddr(params);
}

AG_BMMA_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxBlockMmadKernel<AG_BMMA_KERNEL_FUNC_TEM_PARAMS>::ResetGmAddr(
    const Params &params)
{
    if ASCEND_IS_AIV {
        return;
    }
    aGmAddr_ = reinterpret_cast<__gm__ AType *>(params.mmadParams.aGmAddr);
    bGmAddr_ = reinterpret_cast<__gm__ BType *>(params.mmadParams.bGmAddr);
    cGmAddr_ = reinterpret_cast<__gm__ CType *>(params.mmadParams.cGmAddr);
    scaleAGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(params.mmadParams.scaleAGmAddr);
    scaleBGmAddr_ = reinterpret_cast<__gm__ ::fp8_e8m0_t *>(params.mmadParams.scaleBGmAddr);
    biasGmAddr_ = reinterpret_cast<__gm__ BiasType *>(params.mmadParams.biasGmAddr);
}

AG_BMMA_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline uint32_t AllGatherQbmmMxBlockMmadKernel<AG_BMMA_KERNEL_FUNC_TEM_PARAMS>::CalcDependTileIdx(
    int64_t lastRow, const LocalParams &local) const
{
    // gather buffer 按 rank 顺序拼接，行 lastRow 落在某个 rank 的 local row = lastRow % mPerRank
    const int64_t localRow = lastRow % static_cast<int64_t>(local.mPerRank);
    const int64_t headRows = static_cast<int64_t>(local.tileCnt) * static_cast<int64_t>(local.tileM);
    if (localRow < headRows) {
        // dataHead 第 localRow / tileM 轮（waitIdx 从 1 起，0 保留）
        return static_cast<uint32_t>(localRow / static_cast<int64_t>(local.tileM)) + 1U;
    }
    // dataTail 第 (localRow - headRows) / tailM 轮，clamp 到最后一轮
    const int64_t tailIdx = (localRow - headRows) / static_cast<int64_t>(local.tailM);
    const int64_t maxTailIdx = static_cast<int64_t>(local.tailCnt) - 1;
    const int64_t clamped = tailIdx > maxTailIdx ? maxTailIdx : tailIdx;
    return static_cast<uint32_t>(local.tileCnt + clamped + 1);
}

AG_BMMA_KERNEL_CLASS_TEM_PARAMS
template <typename TensorScaleB>
__aicore__ inline void AllGatherQbmmMxBlockMmadKernel<AG_BMMA_KERNEL_FUNC_TEM_PARAMS>::SetScaleL2Cache(
    const ProblemShape &problemShape, uint64_t baseN, uint64_t scaleKL1, TensorScaleB &gmScaleB)
{
    if (Te::Get<MNK_B>(problemShape) != 1) {
        return;
    }
    if constexpr (transB) {
        const int64_t scaleKRowBytes =
            Blaze::Gemm::CeilDiv(Te::Get<MNK_K>(problemShape), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        const int64_t scaleKL1RowBytes =
            Blaze::Gemm::CeilDiv(scaleKL1, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        // 0x7f: 128B cache line alignment for mx scale GM streaming
        const bool scaleAlignForL2Stream =
            (scaleKRowBytes & kCacheLineAlignMask) == 0 && (scaleKL1RowBytes & kCacheLineAlignMask) == 0;
        gmScaleB.SetL2CacheHint(scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE :
                                                        Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
        const int64_t scaleNStrideBytes = Te::Get<MNK_N>(problemShape) * MXFP_MULTI_BASE_SIZE;
        const int64_t scaleBaseNStrideBytes = baseN * MXFP_MULTI_BASE_SIZE;
        // 0x7f: 128B cache line alignment for mx scale GM streaming
        const bool scaleAlignForL2Stream =
            (scaleNStrideBytes & kCacheLineAlignMask) == 0 && (scaleBaseNStrideBytes & kCacheLineAlignMask) == 0;
        gmScaleB.SetL2CacheHint(scaleAlignForL2Stream ? Te::CacheMode::CACHE_MODE_DISABLE :
                                                        Te::CacheMode::CACHE_MODE_NORMAL);
    }
}

AG_BMMA_KERNEL_CLASS_TEM_PARAMS
template <typename TensorB, typename TensorScaleB>
__aicore__ inline void AllGatherQbmmMxBlockMmadKernel<AG_BMMA_KERNEL_FUNC_TEM_PARAMS>::SetL2Cache(
    const ProblemShape &problemShape, uint64_t curBaseM, uint64_t baseN, uint64_t scaleKL1, TensorB &gmB,
    TensorScaleB &gmScaleB)
{
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

AG_BMMA_KERNEL_CLASS_TEM_PARAMS
__aicore__ inline void AllGatherQbmmMxBlockMmadKernel<AG_BMMA_KERNEL_FUNC_TEM_PARAMS>::ProcessSingleBatch(
    const Params &params, BlockScheduler &bs)
{
    const auto &local = params.localParams;
    const int64_t totalM = static_cast<int64_t>(local.rankSize) * static_cast<int64_t>(local.mPerRank);
    const int64_t Ki = Te::Get<MNK_K>(params.problemShape);
    const int64_t Ni = Te::Get<MNK_N>(params.problemShape);
    const int64_t scaleKLen =
        Blaze::Gemm::CeilDiv(Ki, static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    // 构建各 Tensor 的全局布局：A/ScaleA 为 gather 后的连续 [rankSize*mPerRank, K] 大矩阵
    auto layoutA = MakeLayoutA{}(totalM, Ki);
    auto layoutScaleA = MakeLayoutScaleA{}(totalM, scaleKLen);
    auto layoutB = MakeLayoutB{}(Ki, Ni);
    auto layoutScaleB = MakeLayoutScaleB{}(scaleKLen, Ni);
    auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, Ni);
    auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(params.problemShape), Ni);

    // 创建 Tensor 句柄
    auto gmA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(aGmAddr_), layoutA);
    auto gmScaleA = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleAGmAddr_), layoutScaleA);
    auto gmB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(bGmAddr_), layoutB);
    auto gmScaleB = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(scaleBGmAddr_), layoutScaleB);
    auto gmBias = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(biasGmAddr_), layoutBias);
    auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(cGmAddr_), layoutC);

    // 尾块更新逻辑
    auto &mTailTile = params.schParams.mTailTile;
    auto &nTailTile = params.schParams.nTailTile;
    if (needUpdateTail_ || ((bs.GetEndBlockIdx() + 1) * mTailTile * nTailTile <= AscendC::GetBlockNum())) {
        needUpdateTail_ = true;
        bs.UpdateTailTile(mTailTile, nTailTile);
    }
    SetL2Cache(params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, params.l1Params.scaleKL1, gmB,
               gmScaleB);

    BlockCoord blockIdx;
    int64_t mPos = 0L;
    int64_t nPos = 0L;
    constexpr int64_t kPos = 0L;
    uint32_t readyTileIdx = 0;
    // 遍历当前块的调度任务
    while (bs.GetTileIdx(blockIdx)) {
        BlockShape singleShape =
            bs.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(blockIdx);
        if ((Te::Get<IDX_M_TILEIDX>(singleShape) <= 0) || (Te::Get<IDX_N_TILEIDX>(singleShape) <= 0)) {
            return;
        }
        int64_t blockM = Te::Get<IDX_M_TILEIDX>(singleShape);

        bs.GetTileCoord(blockIdx, mPos, nPos);
        // 等待当前 tile 覆盖的最后一行所依赖的通信轮次（去重：同一轮次只 wait 一次）
        uint32_t dependTileIdx = CalcDependTileIdx(mPos + blockM - 1, local);
        while (readyTileIdx < dependTileIdx) {
            readyTileIdx++;
            commPolicy_.WaitTile(readyTileIdx);
        }

        // gather buffer 连续布局：A/ScaleA/C 直接按 (mPos, nPos) 切片，B/scaleB/bias 全量 K
        auto gmBlockA = gmA.Slice(AscendC::Te::MakeCoord(mPos, kPos),
                                  AscendC::Te::MakeShape(blockM, Get<MNK_K>(params.problemShape)));
        auto gmBlockScaleA = gmScaleA.Slice(AscendC::Te::MakeCoord(mPos, kPos),
                                            AscendC::Te::MakeShape(blockM, scaleKLen));
        auto gmBlockB = gmB.Slice(AscendC::Te::MakeCoord(kPos, nPos),
                                  AscendC::Te::MakeShape(Get<MNK_K>(params.problemShape),
                                                         Get<IDX_N_TILEIDX>(singleShape)));
        auto gmBlockScaleB = gmScaleB.Slice(
            AscendC::Te::MakeCoord(kPos, nPos),
            AscendC::Te::MakeShape(scaleKLen, Get<IDX_N_TILEIDX>(singleShape)));
        auto gmBlockBias =
            gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, Te::Get<IDX_N_TILEIDX>(singleShape)));
        auto gmBlockC = gmC.Slice(AscendC::Te::MakeCoord(mPos, nPos),
                                  AscendC::Te::MakeShape(blockM, Get<IDX_N_TILEIDX>(singleShape)));

        // 输出 tile 单次写出，无跨 rank L0C 累加（splitKIdx 恒 0）
        mmadOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape, 0);
    }

    // 存在尾核没用满核，因此这里要按轮次等待兜底（每个下发通信的核都需消费完所有轮次）
    while (readyTileIdx < local.commTurn) {
        readyTileIdx++;
        commPolicy_.WaitTile(readyTileIdx);
    }
}
} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
