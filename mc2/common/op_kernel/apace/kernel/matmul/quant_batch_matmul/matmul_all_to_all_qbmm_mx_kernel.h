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
 * \file matmul_all_to_all_qbmm_mx_kernel.h
 * \brief Blaze MXFP4/MXFP8 matmul used before MatmulAlltoAll communication.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "include/tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Kernel {

using namespace AscendC;
using AscendC::Te::Get;

// MX/HiF4 量化的 scale 类型、scale K 向单元数与 framing C0 统一使用 ops-tensor 公共函数
// (blaze/gemm/utils/common_utils.h): MxScaleType<T> / MxScaleKLen<T>(k) / GetScaleC0<T>() / IsHif4<T>(),
// 均按数据类型(AType/BType)键控: MX → fp8_e8m0(1字节, 每64元素2单元, C0=2),
// HiF4(hifloat4x2_t) → hif4_scale(4字节: E6M2+E1_8+E1_16, 每64元素1单元, C0=1)。

/**
 * The kernel computes [M, K] x [K, fullN] one destination rank at a time.
 * Each rank's [M, fullN/rankDim] result is written directly to a rank-major
 * Hcomm send buffer, removing the former AIV transpose stage.
 */
template <class ProblemShape, class BlockMmad, class BlockScheduler>
class MatmulAllToAllQbmmMxKernel {
public:
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using AType = typename BlockMmad::AType;
    using BType = typename BlockMmad::BType;
    using CType = typename BlockMmad::CType;
    using BiasType = typename BlockMmad::BiasType;
    // Scale 元素类型由 ops-tensor 公共 MxScaleType 按数据类型导出:
    // MX 量化 → fp8_e8m0_t, HiF4 量化(hifloat4x2_t) → hif4_scale。
    // A/B 两侧 scale 元素类型对称共用(同 kernel_qbmm_mx.h 的单一 ScaleType), 取自 BlockMmad 成员
    // (block_mmad_qbmm_mx.h: ScaleType = MxScaleType<AType>, 按 AType 键控)。
    using ScaleType = typename BlockMmad::ScaleType;
    static_assert(Std::is_same_v<MxScaleType<AType>, MxScaleType<BType>>,
                  "A/B scale element types must be identical (MX/HiF4 symmetric quant)");
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using BlockSchedulerParams = typename BlockScheduler::Params;

    static constexpr bool weightNz = BlockMmad::WEIGHT_NZ;
    static constexpr bool transA = BlockMmad::TRANS_A;
    static constexpr bool transB = BlockMmad::TRANS_B;
    static constexpr int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    static constexpr int64_t CACHE_LINE_ALIGN_MASK = IsFp4<AType>() ? 0xff : 0x7f;

    using BlockShape = Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using MakeLayoutA = Te::FrameLayoutFormat<LayoutA, Std::Int<C0_SIZE>>;
    using MakeLayoutB = Te::FrameLayoutFormat<LayoutB, Std::Int<C0_SIZE>>;
    using MakeLayoutC = Te::FrameLayoutFormat<LayoutC, Std::Int<C0_SIZE>>;
    // Scale framing C0 由 ops-tensor 公共 GetScaleC0 按数据类型导出(MX/E8M0 → 2, HiF4 → 1)
    using MakeLayoutScaleA =
        Std::conditional_t<transA, Te::FrameLayoutFormat<Te::ScaleADNLayoutPtn, Std::Int<GetScaleC0<AType>()>>,
                           Te::FrameLayoutFormat<Te::ScaleANDLayoutPtn, Std::Int<GetScaleC0<AType>()>>>;
    using MakeLayoutScaleB =
        Std::conditional_t<transB, Te::FrameLayoutFormat<Te::ScaleBDNLayoutPtn, Std::Int<GetScaleC0<BType>()>>,
                           Te::FrameLayoutFormat<Te::ScaleBNDLayoutPtn, Std::Int<GetScaleC0<BType>()>>>;

    struct QBMMTiling {
        uint32_t baseM;
        uint32_t baseN;
        uint32_t baseK;
        uint32_t dbL0C;
        uint32_t isBias;
    };

    // Describes the GM address stride between two adjacent communication slices.
    struct SliceParams {
        uint64_t aByteStride;
        uint64_t scaleAByteStride;
        uint64_t cByteStride;
        uint32_t sliceCount;
    };

    struct Params {
        ProblemShape problemShape;
        BlockMmadParams mmParams;
        L1Params l1Params;
        BlockSchedulerParams schedulerParams;
        QBMMTiling qbmmParams;
        uint64_t rankStride;
        uint32_t rankDim;
        SliceParams sliceParams;
    };

    __aicore__ inline void operator()(const Params &params)
    {
        Init(params);
        Run(params);
    }

    // Inject communication as a policy so QBMM owns the compute/sync/communication sequence.
    template <typename CommPolicy>
    __aicore__ inline void operator()(const Params &params, CommPolicy &commPolicy)
    {
        RunSliceGroup(params, commPolicy);
    }

private:
    static constexpr uint16_t AIC_BARRIER_FLAG_ID = 8U;

    BlockMmad mmOp_;

    __aicore__ inline void Init(const Params &params)
    {
        if (params.rankDim == 0) {
            return;
        }
        BlockShape l0TileShape{params.qbmmParams.baseM, params.qbmmParams.baseN, params.qbmmParams.baseK, 0};
        const bool isBias = params.qbmmParams.isBias != 0;
        mmOp_.Init(params.problemShape, l0TileShape, params.l1Params, isBias, params.qbmmParams.dbL0C > 1, 1U);
    }

    __aicore__ inline void Run(const Params &params);

    template <typename CommPolicy>
    __aicore__ inline void RunSliceGroup(const Params &groupParams, CommPolicy &commPolicy)
    {
        const auto &sliceParams = groupParams.sliceParams;
        if (groupParams.rankDim == 0 || sliceParams.sliceCount == 0) {
            return;
        }

        // All slices in a group share one BlockMmad resource lifetime.
        Init(groupParams);

        Params params = groupParams;
        for (uint32_t sliceIdx = 0; sliceIdx < sliceParams.sliceCount; ++sliceIdx) {
            Run(params);

            // Do not submit this slice until every AIC has completed its FixPipe write to GM.
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AIC_BARRIER_FLAG_ID);
            AscendC::CrossCoreWaitFlag<0, PIPE_S>(AIC_BARRIER_FLAG_ID);
            commPolicy.IssueSlice(sliceIdx);

            if (sliceIdx + 1 < sliceParams.sliceCount) {
                params.mmParams.aGmAddr += sliceParams.aByteStride;
                params.mmParams.scaleAGmAddr += sliceParams.scaleAByteStride;
                params.mmParams.cGmAddr += sliceParams.cByteStride;
            }
        }
    }

    template <typename TensorB, typename TensorScaleB>
    __aicore__ inline void SetL2Cache(const ProblemShape &shape, uint64_t curBaseM, uint64_t curBaseN,
                                      uint64_t scaleKL1, TensorB &gmB, TensorScaleB &gmScaleB);
};

template <class ProblemShape, class BlockMmad, class BlockScheduler>
template <typename TensorB, typename TensorScaleB>
__aicore__ inline void MatmulAllToAllQbmmMxKernel<ProblemShape, BlockMmad, BlockScheduler>::SetL2Cache(
    const ProblemShape &shape, uint64_t curBaseM, uint64_t curBaseN, uint64_t scaleKL1, TensorB &gmB,
    TensorScaleB &gmScaleB)
{
    const bool fullMTile = curBaseM >= Te::Get<MNK_M>(shape);
    if (!fullMTile) {
        return;
    }

    // 每64元素的 scale 字节数: MX 为 2(E8M0), HiF4 为 4(hif4_scale), 单元数由公共 MxScaleKLen 按数据类型导出
    const int64_t scaleBytesPerDivisor =
        static_cast<int64_t>(MxScaleKLen<BType>(MXFP_DIVISOR_SIZE)) * static_cast<int64_t>(sizeof(ScaleType));
    if constexpr (transB) {
        const int64_t scaleKRowBytes =
            static_cast<int64_t>(MxScaleKLen<BType>(Te::Get<MNK_K>(shape))) * static_cast<int64_t>(sizeof(ScaleType));
        const int64_t scaleKL1RowBytes =
            static_cast<int64_t>(MxScaleKLen<BType>(static_cast<uint64_t>(scaleKL1))) *
            static_cast<int64_t>(sizeof(ScaleType));
        // 128B cache line alignment for scale GM streaming
        const bool scaleAlign =
            (scaleKRowBytes & CACHE_LINE_ALIGN_MASK) == 0 && (scaleKL1RowBytes & CACHE_LINE_ALIGN_MASK) == 0;
        gmScaleB.SetL2CacheHint(scaleAlign ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    } else {
        const int64_t scaleNStrideBytes = Te::Get<MNK_N>(shape) * scaleBytesPerDivisor;
        const int64_t scaleBaseNStrideBytes = static_cast<int64_t>(curBaseN) * scaleBytesPerDivisor;
        // 128B cache line alignment for scale GM streaming
        const bool scaleAlign =
            (scaleNStrideBytes & CACHE_LINE_ALIGN_MASK) == 0 && (scaleBaseNStrideBytes & CACHE_LINE_ALIGN_MASK) == 0;
        gmScaleB.SetL2CacheHint(scaleAlign ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    }

    if constexpr (weightNz) {
        gmB.SetL2CacheHint(Te::CacheMode::CACHE_MODE_DISABLE);
    } else {
        const bool bAlign = (Te::Get<MNK_K>(shape) & CACHE_LINE_ALIGN_MASK) == 0;
        gmB.SetL2CacheHint(bAlign ? Te::CacheMode::CACHE_MODE_DISABLE : Te::CacheMode::CACHE_MODE_NORMAL);
    }
}

template <class ProblemShape, class BlockMmad, class BlockScheduler>
__aicore__ inline void MatmulAllToAllQbmmMxKernel<ProblemShape, BlockMmad, BlockScheduler>::Run(const Params &params)
{
    const int64_t rankN = Te::Get<MNK_N>(params.problemShape);
    if (params.rankDim == 0) {
        return;
    }
    const int64_t fullN = rankN * static_cast<int64_t>(params.rankDim);

    // Scale K 方向单元数(按 scale 元素计), 由公共 MxScaleKLen 按数据类型导出:
    // MX 为 E8M0 单元(K/64*2), HiF4 为 hif4_scale 单元(K/64)
    const auto scaleKLen = MxScaleKLen<AType>(Te::Get<MNK_K>(params.problemShape));
    auto layoutA = MakeLayoutA{}(Te::Get<MNK_M>(params.problemShape), Te::Get<MNK_K>(params.problemShape));
    auto layoutScaleA = MakeLayoutScaleA{}(Te::Get<MNK_M>(params.problemShape), scaleKLen);
    auto layoutB = MakeLayoutB{}(Te::Get<MNK_K>(params.problemShape), fullN);
    auto layoutScaleB = MakeLayoutScaleB{}(MxScaleKLen<BType>(Te::Get<MNK_K>(params.problemShape)), fullN);
    auto layoutC = MakeLayoutC{}(Te::Get<MNK_M>(params.problemShape), rankN);
    auto layoutBias = Te::MakeFrameLayout<Te::NDExtLayoutPtn>(1L, fullN);

    auto gmA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ AType *>(params.mmParams.aGmAddr)), layoutA);
    auto gmScaleA = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ScaleType *>(params.mmParams.scaleAGmAddr)),
        layoutScaleA);
    auto gmB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BType *>(params.mmParams.bGmAddr)), layoutB);
    auto gmScaleB = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ ScaleType *>(params.mmParams.scaleBGmAddr)),
        layoutScaleB);
    auto gmBias = Te::MakeTensor(
        Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ BiasType *>(params.mmParams.biasGmAddr)), layoutBias);

    SetL2Cache(params.problemShape, params.qbmmParams.baseM, params.qbmmParams.baseN, params.l1Params.scaleKL1, gmB,
               gmScaleB);
    for (uint32_t rankIdx = 0; rankIdx < params.rankDim; ++rankIdx) {
        BlockScheduler scheduler(params.problemShape, params.schedulerParams);
        if ((scheduler.GetEndBlockIdx() + 1) * params.schedulerParams.mTailTile * params.schedulerParams.nTailTile <=
            AscendC::GetBlockNum()) {
            scheduler.UpdateTailTile(params.schedulerParams.mTailTile, params.schedulerParams.nTailTile);
        }

        auto gmRankC =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(reinterpret_cast<__gm__ CType *>(params.mmParams.cGmAddr) +
                                                            rankIdx * params.rankStride),
                           layoutC);
        BlockCoord blockIdx;
        int64_t mPos = 0;
        int64_t nPos = 0;
        while (scheduler.GetTileIdx(blockIdx)) {
            BlockShape singleShape =
                scheduler.template GetBlockShape<QuantMode::MX_PERGROUP_MODE, QuantMode::MX_PERGROUP_MODE, weightNz>(
                    blockIdx);
            if (Te::Get<IDX_M_TILEIDX>(singleShape) <= 0 || Te::Get<IDX_N_TILEIDX>(singleShape) <= 0) {
                return;
            }
            scheduler.GetTileCoord(blockIdx, mPos, nPos);
            const int64_t fullNPos = static_cast<int64_t>(rankIdx) * rankN + nPos;
            // HiF4+非转置(ND)修复: ScaleBND 布局下 scale 的硬件搬运(asc_copy_gm2l1_nd2nz)按 e8m0
            // 半字栅格(2字节)计列偏移, 而张量 Slice 按 hif4_scale 元素栅格(4字节)计列坐标, 列坐标>0
            // 时该 N tile 的 scale 整体错位。正确的列字节偏移 nPos*2 恰等于列坐标 (nPos>>1) 的
            // Slice 偏移 (nPos>>1)*4, 故对 hif4+ND 将 Slice 列坐标折半(与 AlltoAllMatmul 及
            // ops-tensor kernel_qbmm_mx_without_batch.h 的同族修复保持同构)。
            // 折半精确的前提是列坐标恒偶: baseN 为 GetShapeWithDataType<hif4> 对齐后的偶数(64/256),
            // 尾拆按 CUBE_BLOCK(16) 对齐, 均偶; e8m0(C0=2, 列栅格天然 2 字节)与转置 DN(列偏移在
            // 4 字节栅格)路径不受影响, 走原逻辑。
            int64_t scaleBColPos = fullNPos;
            if constexpr (IsHif4<AType>() && !transB) {
                scaleBColPos = fullNPos >> 1;
            }
            auto gmBlockA = gmA.Slice(Te::MakeCoord(mPos, 0L),
                                      Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_K>(params.problemShape)));
            auto gmBlockScaleA =
                gmScaleA.Slice(Te::MakeCoord(mPos, 0L), Te::MakeShape(Get<MNK_M>(singleShape), scaleKLen));
            auto gmBlockB = gmB.Slice(Te::MakeCoord(0L, fullNPos),
                                      Te::MakeShape(Get<MNK_K>(params.problemShape), Get<MNK_N>(singleShape)));
            auto gmBlockScaleB =
                gmScaleB.Slice(Te::MakeCoord(0L, scaleBColPos), Te::MakeShape(scaleKLen, Get<MNK_N>(singleShape)));
            auto gmBlockC = gmRankC.Slice(Te::MakeCoord(mPos, nPos),
                                          Te::MakeShape(Get<MNK_M>(singleShape), Get<MNK_N>(singleShape)));
            auto gmBlockBias = gmBias.Slice(Te::MakeCoord(0L, fullNPos), Te::MakeShape(1L, Get<MNK_N>(singleShape)));
            mmOp_(gmBlockA, gmBlockB, gmBlockScaleA, gmBlockScaleB, gmBlockBias, gmBlockC, singleShape, 0);
        }
    }
}

} // namespace Kernel
} // namespace Gemm
} // namespace Blaze
