/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * \file moe_token_permute_regbase.h
 * \brief Ascend 950 RegBase implementation for MoeTokenPermute.
 */
#ifndef MOE_TOKEN_PERMUTE_REGBASE_H
#define MOE_TOKEN_PERMUTE_REGBASE_H

#include <cstdint>

// The Arch35 routing helpers are shared at source level. Keep this compatibility
// data private to this translation unit so MoeTokenPermute retains its tiling ABI.
struct MoeV2VBSComputeTilingData {
    int64_t needCoreNum;
    int64_t perCoreElements;
    int64_t perCoreLoops;
    int64_t perCorePerLoopElements;
    int64_t perCoreLastLoopElements;
    int64_t lastCoreElements;
    int64_t lastCoreLoops;
    int64_t lastCorePerLoopElements;
    int64_t lastCoreLastLoopElements;
    int64_t oneLoopMaxElements;
};

struct MoeV2VMSMiddleComputeTilingData {
    int64_t needCoreNum;
};

struct MoeV2SortOutComputeTilingData {
    int64_t oneLoopMaxElements;
};

struct MoeV2GatherOutComputeTilingData {
    int64_t needCoreNum;
    int64_t activateRows;
    int64_t perCoreRows;
    int64_t perCorePerLoopRows;
    int64_t perCoreLastLoopRows;
    int64_t lastCoreRows;
    int64_t lastCorePerLoopRows;
    int64_t lastCoreLastLoopRows;
    int64_t perCoreLoops;
    int64_t lastCoreLoops;
    int64_t perLoopCols;
    int64_t lastLoopCols;
    int64_t colLoops;
    int64_t scatterMode;
};

struct MoeInitRoutingV2TilingData {
    int64_t coreNum;
    int64_t n;
    int64_t cols;
    int64_t k;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensCountOrCumsumFlag;
    int64_t expertTokensBeforeCapacityFlag;
    MoeV2VBSComputeTilingData vbsComputeParamsOp;
    MoeV2VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    MoeV2SortOutComputeTilingData sortOutComputeParamsOp;
    MoeV2GatherOutComputeTilingData srcToDstComputeParamsOp;
    MoeV2GatherOutComputeTilingData srcToDstCapacityComputeParamsOp;
    MoeV2GatherOutComputeTilingData gatherOutComputeParamsOp;
};

#include "../../../moe_init_routing_v2/op_kernel/arch35/moe_v2_sort_one_core.h"
#include "../../../moe_init_routing_v2/op_kernel/arch35/moe_v2_sort_multi_core.h"
#include "../../../moe_init_routing_v2/op_kernel/arch35/moe_v2_src_to_dst_op_simt.h"
#include "../../../moe_init_routing_v2/op_kernel/arch35/moe_v2_gather_out_for_simt.h"

namespace MoeTokenPermute {
using namespace AscendC;

__aicore__ inline void InitRegBaseSortTiling(MoeInitRoutingV2TilingData &dst,
                                             const MoeTokenPermuteTilingData *src)
{
    dst.coreNum = src->coreNum;
    dst.n = src->n;
    dst.cols = src->cols;
    dst.k = src->topK;
    dst.expertCapacity = 0;
    dst.expertNum = 0;
    dst.dropPadMode = 0;
    dst.expertTokensCountOrCumsumFlag = 0;
    dst.expertTokensBeforeCapacityFlag = 0;

    dst.vbsComputeParamsOp.needCoreNum = src->vbsComputeParamsOp.needCoreNum;
    dst.vbsComputeParamsOp.perCoreElements = src->vbsComputeParamsOp.perCoreElements;
    dst.vbsComputeParamsOp.perCoreLoops = src->vbsComputeParamsOp.perCoreLoops;
    dst.vbsComputeParamsOp.perCorePerLoopElements = src->vbsComputeParamsOp.perCorePerLoopElements;
    dst.vbsComputeParamsOp.perCoreLastLoopElements = src->vbsComputeParamsOp.perCoreLastLoopElements;
    dst.vbsComputeParamsOp.lastCoreElements = src->vbsComputeParamsOp.lastCoreElements;
    dst.vbsComputeParamsOp.lastCoreLoops = src->vbsComputeParamsOp.lastCoreLoops;
    dst.vbsComputeParamsOp.lastCorePerLoopElements = src->vbsComputeParamsOp.lastCorePerLoopElements;
    dst.vbsComputeParamsOp.lastCoreLastLoopElements = src->vbsComputeParamsOp.lastCoreLastLoopElements;
    dst.vbsComputeParamsOp.oneLoopMaxElements = src->vbsComputeParamsOp.oneLoopMaxElements;
    // The shared Arch35 sort has no lastCoreWSindex field; host tiling keeps all cores at equal merge depth.
    dst.vmsMiddleComputeParamsOp.needCoreNum = src->vmsMiddleComputeParamsOp.needCoreNum;
    dst.sortOutComputeParamsOp.oneLoopMaxElements = src->sortOutComputeParamsOp.oneLoopMaxElements;
}

__aicore__ inline void InitRegBaseGatherTiling(MoeInitRoutingV2TilingData &dst,
                                               const MoeTokenPermuteTilingData *src)
{
    const auto &indexCopy = src->indexCopyComputeParamsOp;
    MoeV2GatherOutComputeTilingData gather{};
    gather.needCoreNum = indexCopy.needCoreNum;
    gather.activateRows = indexCopy.numOutTokens;
    gather.perCoreRows = indexCopy.coreCalcNum;
    gather.perCorePerLoopRows = indexCopy.onceIndicesTokenNums;
    gather.perCoreLastLoopRows = indexCopy.frontCoreLastTokenNums;
    gather.lastCoreRows = indexCopy.coreCalcTail;
    gather.lastCorePerLoopRows = indexCopy.onceUbTokenNums;
    gather.lastCoreLastLoopRows = indexCopy.tailLastIndicesLastTokenNums;
    // The shared gather helper consumes these two loop-count fields in the
    // opposite core branches, so store them in the order it expects.
    gather.perCoreLoops = indexCopy.tailCoreLoop;
    gather.lastCoreLoops = indexCopy.frontCoreLoop;
    gather.perLoopCols = indexCopy.oneTokenOnceMove;
    gather.lastLoopCols = indexCopy.oneTokenlastMove;
    gather.colLoops = indexCopy.oneTokenMoveTimes;
    gather.scatterMode = 0;

    dst.srcToDstComputeParamsOp = gather;
    dst.srcToDstCapacityComputeParamsOp = gather;
    dst.gatherOutComputeParamsOp = gather;
}

__aicore__ inline void InitRegBaseCompatTiling(MoeInitRoutingV2TilingData &dst,
                                               const MoeTokenPermuteTilingData *src)
{
    InitRegBaseSortTiling(dst, src);
    InitRegBaseGatherTiling(dst, src);
}

template <typename TokenType, typename IndexType>
class MoeTokenPermuteRegBase {
public:
    __aicore__ inline void Process(GM_ADDR tokens, GM_ADDR indices, GM_ADDR permuteTokens, GM_ADDR sortedIndices,
                                   GM_ADDR workspace, const MoeTokenPermuteTilingData *tilingData)
    {
        if (tilingData->n <= 0 || tilingData->topK <= 0) {
            return;
        }

        MoeInitRoutingV2TilingData compatTiling{};
        InitRegBaseCompatTiling(compatTiling, tilingData);

        TPipe sortPipe;
        if (tilingData->vbsComputeParamsOp.needCoreNum <= 1) {
            MoeInitRoutingV2::MoeV2SortOneCore<IndexType> sortOp;
            sortOp.Init(indices, nullptr, nullptr, workspace, &compatTiling, &sortPipe);
            sortOp.Process();
        } else {
            MoeInitRoutingV2::MoeV2SortMultiCore<IndexType> sortOp;
            sortOp.Init(indices, nullptr, nullptr, workspace, &compatTiling, &sortPipe);
            sortOp.Process();
        }
        sortPipe.Destroy();

        MoeInitRoutingV2::MoeV2SrcToDstOpSimt inverseOp;
        inverseOp.Init(sortedIndices, workspace, &compatTiling);
        inverseOp.Process();

        if (tilingData->cols <= 0 || tilingData->indexCopyComputeParamsOp.numOutTokens <= 0) {
            return;
        }

        TPipe gatherPipe;
        MoeInitRoutingV2::MoeV2GatherOutSimt<TokenType> gatherOp;
        gatherOp.Init(tokens, sortedIndices, permuteTokens, workspace, &compatTiling, &gatherPipe);
        gatherOp.Process();
        gatherPipe.Destroy();
    }
};

} // namespace MoeTokenPermute
#endif // MOE_TOKEN_PERMUTE_REGBASE_H
