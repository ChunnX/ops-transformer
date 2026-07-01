/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_aiv_mode_dequant.h
 * \brief 反量化（Dequant）执行器，封装AIV核心上的分块反量化逻辑
 *
 * 支持两种反量化模式：
 *   - Per-Channel反量化：C_int32 × perChannelScale → D_float16/bfloat16
 *     每个输出列使用一个scale因子，scale形状为[N]
 *   - Per-Token反量化：D_float16/bfloat16 × perTokenScale → D_float16/bfloat16
 *     每个输入行使用一个scale因子，scale形状为[M]
 *   - Per-Channel + Per-Token组合：C_int32 × perChannelScale × perTokenScale → D
 *
 * 反量化流水线（在BlockEpilogue中执行）：
 *   1. 从GM拷贝C/scale/perTokenScale到UB
 *   2. 将int32的C cast到float32
 *   3. 执行广播乘法：C_fp32 × scale（行广播）和/或 C_fp32 × perTokenScale（列广播）
 *   4. 将float32结果cast回输出类型（half/bfloat16）
 *   5. 从UB拷贝结果到GM
 */
#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_DEQUANT_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_DEQUANT_H

#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/catlass.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/cross_core_sync.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/resource.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/layout.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/detail/callback.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm_coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/matrix_coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/block/block_epilogue.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tile_broadcast_mul.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tile_swizzle.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/block_mmad.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/block/block_swizzle.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/gemm_type.hpp"
#include "matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h"
#include "matmul_reduce_scatter_aiv_mode_util.h"
#include "tile_broadcast_add.hpp"

using namespace matmulReduceScatterV2_util;
namespace dequant {
template <typename BiasType, typename OutputType>
class DequantRunner {
public:
    using ArchTag = Arch::AtlasA2;
    using ScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using BiasGType = Gemm::GemmType<BiasType, layout::VectorLayout>;
    using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
    using DType = Gemm::GemmType<OutputType, layout::RowMajor>;
    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using RowBroadcastAddType = Gemm::GemmType<float, layout::RowMajor>;
    
    using EpilogueTileShape = MatrixShape<TILE_SHAPE_64, TILE_SHAPE_128>;
    // 行广播乘法Tile：C_fp32[i,j] × scale[j]（scale沿行方向广播）
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileRowBroadcastAdd = Epilogue::Tile::TileRowBroadcastAdd<ArchTag, RowBroadcastAddType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    // 列广播乘法Tile：result[i,j] × perTokenScale[i]（perTokenScale沿列方向广播）
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    // Epilogue层的Tile拷贝和调度策略
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    // BlockEpilogue：反量化的核心执行单元，包含双缓冲UB管理和分块循环
    using BlockEpilogue =
        Epilogue::Block::BlockEpilogue<ArchTag, CType, ScaleType, PerTokenScaleType, BiasGType, DType,
        TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul,
        TileCopy, TileScheduler>;

    uint32_t DefaultSwizzleDirect = 0;   // 默认Swizzle方向
    uint32_t DefaultSwizzleCount = 3;    // 默认Swizzle块数量

    __aicore__ explicit DequantRunner() = default;

    // 根据loop偏移量获取分块的M/N坐标索引（使用Swizzle调度）
    __aicore__ inline GemmCoord GetBlockIdCoord(int32_t loopOffset, int32_t mLoop, int32_t nLoop, int32_t swizzlDirect,
                                                int32_t swizzlCount)
    {
        uint32_t kIdx = 0;
        int64_t mIdx, nIdx;
        GetSwizzledBlockIdx(loopOffset, mLoop, nLoop, swizzlDirect, swizzlCount, mIdx, nIdx);
        return GemmCoord{static_cast<uint32_t>(mIdx), static_cast<uint32_t>(nIdx), kIdx}; // idx在uint32_t范围内
    }

    // 根据分块索引计算分块在矩阵中的起始位置坐标
    __aicore__ inline GemmCoord GetBlockLocCoord(GemmCoord blockIdxCoord)
    {
        return GemmCoord{blockIdxCoord.m() * m0, blockIdxCoord.n() * n0, 0};
    }

    // 根据分块索引和位置计算分块的实际大小（最后一个分块可能不满）
    __aicore__ inline GemmCoord GetBlockSizeCoord(GemmCoord blockIdxCoord, GemmCoord blockLocCoord, int32_t mLoop,
                                                  int32_t mSize, int32_t nLoop, int32_t nSize, int32_t kSize)
    {
        uint32_t mActual = (blockIdxCoord.m() == (mLoop - 1)) ? (mSize - blockLocCoord.m()) : m0;
        uint32_t nActual = (blockIdxCoord.n() == (nLoop - 1)) ? (nSize - blockLocCoord.n()) : n0;
        uint32_t kActual = kSize;
        return GemmCoord{mActual, nActual, kActual};
    }

    // RunMatmulReduceScatter: 反量化主执行函数
    // 遍历当前AIV核心负责的所有分块，对int32累加结果做反量化并写入peermem或输出GM
    __aicore__ inline void RunMatmulReduceScatter(DEQUANT_ARGS_FUN())
    {
        m0 = tileM0;
        n0 = tileN0;
        // 每个rank的最终输出M维度 = M / rankSize
        uint32_t finalM = rowNum / rankSize;
        uint32_t finalN = colNum;
        // 计算M/N方向的分块循环次数
        uint32_t mLoop = DivCeil(finalM, m0);
        uint32_t nLoop = DivCeil(finalN, n0);
        uint32_t blockLoop = mLoop * nLoop;
        uint32_t totalLoop = blockLoop * rankSize;
        uint32_t blockSize = m0 * n0;
        // 每次通信的计算循环数 = pValue × coreNum
        uint32_t loopNumPerCom = pValue * coreNum;
        uint32_t loopSt = calIdx * loopNumPerCom;
        // 每个rank在peermem中的分块大小
        uint32_t peerMemBlockSize = blockSize * loopNumPerCom / rankSize;

        // 实例化BlockEpilogue反量化执行器
        BlockEpilogue blockEpilogue(resource);
        layout::RowMajor layoutBlock{m0, n0};
        layout::RowMajor layoutOutput{finalM, finalN};
        __gm__ OutputType *gmPeerMem = reinterpret_cast<__gm__ OutputType *>(peerMem);
        __gm__ OutputType *gmOutput = reinterpret_cast<__gm__ OutputType *>(output);
        __gm__ BiasType *gmbias = reinterpret_cast<__gm__ BiasType *>(biasptr);
        for (int32_t p = 0; p < pValue; p++) {
            int32_t loopIdx = loopSt + p * coreNum + coreIdx;
            if (loopIdx >= totalLoop) {
                break;
            }
            // 计算目标rank索引和块内索引
            int32_t dstRankIdx = loopIdx % rankSize;
            int32_t inRankIdx = loopIdx / rankSize;
            // 获取分块坐标信息
            GemmCoord blockIdxCoord = GetBlockIdCoord(inRankIdx, mLoop, nLoop, swizzlDirect, swizzlCount);
            GemmCoord blockLocCoord = GetBlockLocCoord(blockIdxCoord);
            GemmCoord blockSizeCoord = GetBlockSizeCoord(blockIdxCoord, blockLocCoord, mLoop, finalM, nLoop, finalN, 0);

            layout::VectorLayout layoutPerChannelScale{blockSizeCoord.n()};
            layout::VectorLayout layoutBias{blockSizeCoord.n()};
            layout::VectorLayout layoutPerTokenScale{blockSizeCoord.m()};
            // 计算在peermem中的数据偏移
            uint32_t dataBlockOffset = dstRankIdx * peerMemBlockSize + (loopIdx - loopSt) / rankSize * blockSize;
            layout::RowMajor layoutDst = layoutBlock;
            __gm__ OutputType *gmDst = gmPeerMem + dataBlockOffset;
            // 如果目标rank是本rank，直接写入输出GM
            if (dstRankIdx == rankIdx) {
                MatrixCoord offsetC{blockLocCoord.m(), blockLocCoord.n()};
                layoutDst = layoutOutput;
                gmDst = gmOutput + layoutOutput.GetOffset(offsetC);
            }
            __gm__ BiasType *gmBiasOffset = (gmbias != nullptr) ? (gmbias + blockLocCoord.n()) : nullptr;
            if (needPerChannel && needPerToken) {
                // Per-Channel + Per-Token组合反量化
                blockEpilogue(perChannelScale + blockLocCoord.n(), layoutPerChannelScale,
                              perTokenScale + dstRankIdx * finalM + blockLocCoord.m(), layoutPerTokenScale,
                              gmBiasOffset, layoutBias, workspace + dataBlockOffset, layoutBlock,
                              gmDst, layoutDst,
                              blockSizeCoord);
            } else if (needPerChannel) {
                blockEpilogue(perChannelScale + blockLocCoord.n(), layoutPerChannelScale,
                              gmBiasOffset, layoutBias,
                              workspace + dataBlockOffset, layoutBlock,
                              gmDst, layoutDst,
                              blockSizeCoord);
            } else if (needPerToken) {
                blockEpilogue(perTokenScale + dstRankIdx * finalM + blockLocCoord.m(), layoutPerTokenScale,
                              gmBiasOffset, layoutBias,
                              gmPeerMem + dataBlockOffset, layoutBlock,
                              gmDst, layoutDst,
                              blockSizeCoord);
            }
        }
    }

private:
    uint32_t m0;    // M方向分块大小
    uint32_t n0;    // N方向分块大小
};
} // namespace dequant
#endif // MATMUL_REDUCE_SCATTER_AIV_MODE_DEQUANT_H
