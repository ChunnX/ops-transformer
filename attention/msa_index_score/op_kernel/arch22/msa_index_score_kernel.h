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
 * \file msa_index_score_kernel.h
 * \brief MsaIndexScore Atlas A2/A3 核函数：AIC 做 paged QK^T，AIV 做 mask + 分段 RowMax。
 *
 * 数据流（A2 无 L0C->UB 直通通路，S 必须经 GM workspace 中转）：
 *   query/key_cache --(AIC: catlass BlockMmad)--> L0C --FIXPIPE--> workspace S tile
 *                   --(AIV: mask + 分段 RowMax + scale)--> score
 */

#ifndef MSA_INDEX_SCORE_KERNEL_H
#define MSA_INDEX_SCORE_KERNEL_H

#include "kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"

#include "../msa_index_score_common.h"
#include "msa_index_score_task.h"
#include "msa_index_score_epilogue.h"

namespace MsaIndexScoreNs {

template <class ElementInput_>
class MsaIndexScoreKernel {
public:
    using ElementInput = ElementInput_;
    using ArchTag = Catlass::Arch::AtlasA2;

    // Q 视作 [M, headDim] 行主序（行 stride = strideQn）。
    using LayoutQ = Catlass::layout::RowMajor;
    // K^T 视作 [headDim, blockSize] 列主序（列 stride = strideKvToken），
    // 即直接把 page 内的 [blockSize, headDim] 子矩阵转置成 B 操作数。
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutS = Catlass::layout::RowMajor;

    using QType = Catlass::Gemm::GemmType<ElementInput, LayoutQ>;
    using KType = Catlass::Gemm::GemmType<ElementInput, LayoutK>;
    using SType = Catlass::Gemm::GemmType<float, LayoutS>;

    // P0 采用最简流水的 Pingpong 策略；ENABLE_UNIT_FLAG 关闭以走显式 M_FIX/FIX_M 握手。
    using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<false>;
    using L1TileShape = Catlass::GemmShape<MSA_ROW_TILE_M, MSA_BLOCK_SIZE, MSA_K_TILE>;
    using L0TileShape = L1TileShape;
    using BlockMmad =
        Catlass::Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, QType, KType, SType>;

    static constexpr uint32_t REVERSE_DEPTH = MSA_WORKSPACE_STAGES - 1;
    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;

    __aicore__ inline MsaIndexScoreKernel() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR keyCache, GM_ADDR blockTable, GM_ADDR scale,
                                GM_ADDR cuSeqLenQ, GM_ADDR kvLen, GM_ADDR startLoc, GM_ADDR score,
                                GM_ADDR workspace, const MsaIndexScoreTilingData *__restrict tiling)
    {
        tiling_ = tiling;
        gQuery_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementInput *>(query));
        gKeyCache_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementInput *>(keyCache));
        gBlockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
        gScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale));
        gCuSeqLenQ_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cuSeqLenQ));
        gKvLen_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(kvLen));
        gStartLoc_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(startLoc));
        gScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(score));
        gWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        scheduler_.Init(tiling_->batch, tiling_->numQHeads, tiling_->maxBlocksPerBatch,
                        tiling_->scoreBlockStride, tiling_->fillInvalidTail != 0U, gCuSeqLenQ_, gKvLen_,
                        gStartLoc_);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProcessCube();
        }
        if ASCEND_IS_AIV {
            ProcessVector();
        }
    }

private:
    __aicore__ inline void ProcessCube()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH> flagSReady{MSA_FLAG_S_READY,
                                                                          MSA_FLAG_S_READY_REVERSE};

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreWsBase =
            static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        const uint32_t totalTasks = scheduler_.TotalTasks();

        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            for (uint32_t st = 0; st < task.numSTiles; ++st) {
                const uint64_t sBase =
                    coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                ComputeSTile(blockMmad, task, st * MSA_BLOCKS_PER_STILE, sBase);
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagSReady);
                ++tileSeq;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    /// 产出一个 S tile：逐 page 经 block_table 间接寻址，做 [mActual, blockSize] = Q * K^T。
    __aicore__ inline void ComputeSTile(BlockMmad &blockMmad, const MsaTask &task, uint32_t blkBase,
                                        uint64_t sBase)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const LayoutQ layoutQ(mActual, headDim, tiling_->strideQn);
        const LayoutK layoutK(headDim, MSA_BLOCK_SIZE, tiling_->strideKvToken);
        const LayoutS layoutS(mActual, MSA_BLOCK_SIZE, STILE_WIDTH);
        const Catlass::GemmCoord shape{mActual, MSA_BLOCK_SIZE, headDim};
        // 行 r 对应 (token, head)，head 为低位，且 strideQt == numQHeads * strideQn，
        // 因此整个 [totalQ * numQHeads, headDim] 视图是行 stride 为 strideQn 的行主序矩阵。
        const uint64_t qOffset = static_cast<uint64_t>(task.globalRowBase) * tiling_->strideQn;

        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            const int32_t page =
                gBlockTable_.GetValue(task.batchIdx * tiling_->maxBlocksPerBatch + blk);
            const uint64_t kOffset = static_cast<uint64_t>(page) * tiling_->strideKvBlock;
            blockMmad(gQuery_[qOffset], layoutQ, gKeyCache_[kOffset], layoutK,
                      gWorkspace_[sBase + j * MSA_BLOCK_SIZE], layoutS, shape);
        }
    }

    __aicore__ inline void ProcessVector()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        MsaSegRowMaxEpilogue epilogue;
        epilogue.Init(resource, tiling_->numQHeads, tiling_->strideOutHead, tiling_->strideOutToken,
                      tiling_->strideScaleT, tiling_->strideScaleN, tiling_->hasScale != 0U, gScore_,
                      gScale_);
        Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH> flagSReady{MSA_FLAG_S_READY,
                                                                          MSA_FLAG_S_READY_REVERSE};

        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t subIdx = AscendC::GetSubBlockIdx();
        const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreWsBase =
            static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        const uint32_t totalTasks = scheduler_.TotalTasks();

        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            for (uint32_t st = 0; st < task.numSTiles; ++st) {
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagSReady);
                const uint64_t sBase =
                    coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                epilogue.ProcessSTile(gWorkspace_[sBase], task, st * MSA_BLOCKS_PER_STILE, subIdx,
                                      subBlockNum);
                ++tileSeq;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    const MsaIndexScoreTilingData *__restrict tiling_ = nullptr;
    MsaTaskScheduler scheduler_;

    AscendC::GlobalTensor<ElementInput> gQuery_;
    AscendC::GlobalTensor<ElementInput> gKeyCache_;
    AscendC::GlobalTensor<int32_t> gBlockTable_;
    AscendC::GlobalTensor<float> gScale_;
    AscendC::GlobalTensor<int32_t> gCuSeqLenQ_;
    AscendC::GlobalTensor<int32_t> gKvLen_;
    AscendC::GlobalTensor<int32_t> gStartLoc_;
    AscendC::GlobalTensor<float> gScore_;
    AscendC::GlobalTensor<float> gWorkspace_;
};

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_KERNEL_H
