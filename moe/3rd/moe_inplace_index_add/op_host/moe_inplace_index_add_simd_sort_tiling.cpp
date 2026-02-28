/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_inplace_index_simd_sort_tiling.cpp
 * \brief
 */

#include <cstdint>
#include "tiling_base/tiling_templates_registry.h"
#include "moe_inplace_index_add_tiling.h"
#include "moe_inplace_index_add_simd_sort_tiling.h"

namespace optiling
{
constexpr int64_t SIMD_DB_BUFFER = 2;
constexpr int64_t SIZE_TWO = 2;
constexpr int64_t ALIGN_SIZE = 32;
constexpr int64_t INT32_BYTES = 4;
constexpr int64_t RESERVE_SIZE = 128;
constexpr int64_t FP32_BYTES = 4;
constexpr int64_t MIN_HANDLE_SIZE = 128;
constexpr int64_t AFTER_HANDLE_THRESHOLD = 1024;
constexpr int64_t INDICES_MIN_BLOCK_SIZE = 8192;
constexpr uint64_t SIMD_SORT_TILING_KEY = 200000;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16L * 1024L * 1024L;

static const std::set<ge::DataType> VAR_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT64, ge::DT_INT32,
                                                 ge::DT_INT16, ge::DT_INT8,    ge::DT_UINT8, ge::DT_BOOL};

bool MoeInplaceIndexAddSimdSortTiling::IsCapable()
{
    if (isSimdSort_ == 1) {
        return true;
    }
    return false;
}

int64_t MoeInplaceIndexAddSimdSortTiling::GetIndicesAlignBlockSize(int64_t indicesFactor)
{
    /* ubIndexFactor_: indicesindicesQue + (sortIndicesQue + 2 * shiftOfset) +
                       (uniqueIdCntQue_ + 1) + (updateSumIdxQue_ + 1) + originIdxQue
    */ 
    auto ubBlock = Ops::Base::GetUbBlockSize(context_);
    int64_t occupy = MoeCeilAlign(indicesFactor * indicesTypeSize_, ubBlock) +
                    (MoeCeilAlign(indicesFactor * (indicesTypeSize_), ubBlock) + SIZE_TWO * ALIGN_SIZE) +
                    (MoeCeilAlign(indicesFactor * INT32_BYTES, ubBlock) + ALIGN_SIZE) +
                    (MoeCeilAlign(indicesFactor * indicesTypeSize_, ubBlock) + ALIGN_SIZE) + 
                     MoeCeilAlign(indicesFactor * INT32_BYTES, ubBlock) +
                     GetSortTmpSize(indicesDtype_, indicesFactor, false);
    return occupy;
}

int64_t MoeInplaceIndexAddSimdSortTiling::GetAfterAlignBlockSize(int64_t indicesFactor, int64_t afterFactor)
{
    /* ubIndexFactor_ * afterAxisAlign: updatesQue_ + updateSumQue_ */ 
    auto ubBlock = Ops::Base::GetUbBlockSize(context_);
    int64_t occupy = indicesFactor * MoeCeilAlign((varTypeSize_) * afterFactor, ubBlock) + 
                     indicesFactor * MoeCeilAlign((FP32_BYTES) * afterFactor, ubBlock);
    return occupy;
}

void MoeInplaceIndexAddSimdSortTiling::DoOpTilingSplitAfterSort()
{
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / SIMD_DB_BUFFER);
    halfUbSize = halfUbSize - MIN_HANDLE_SIZE * FP32_BYTES;//maxScore
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    /* split afterAxis */
    eachCoreAfterAxisCount_ = Ops::Base::CeilDiv(afterAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(afterAxis_, eachCoreAfterAxisCount_);
    tailCoreAfterAxisCount_ = afterAxis_ - eachCoreAfterAxisCount_ * (usedCoreNumBefore_ - 1);

    auto ubBlock = Ops::Base::GetUbBlockSize(context_);
    /* 同地址优化:搬入多少行indices,就搬入相同行数的updates
    * ubIndexFactor_: indiecesQue + (sortIndicesQue + 2 * shiftOfset) + originIdxQue +
    *                 (uniqueIdCntQue_ + 1) + updateSumIdxQue_,
    * ubIndexFactor_ * eachCoreAfterAxisCount_: updatesQue_ + updateSumQue_
    */
    int64_t occupy = MoeCeilAlign(indicesTypeSize_, ubBlock) +
                     MoeCeilAlign(indicesTypeSize_ + SIZE_TWO * ALIGN_SIZE, ubBlock) + 
                     MoeCeilAlign(INT32_BYTES, ubBlock) + 
                     MoeCeilAlign(INT32_BYTES * SIZE_TWO, ubBlock) +
                     MoeCeilAlign(indicesTypeSize_, ubBlock) + 
                     MoeCeilAlign(varTypeSize_ * eachCoreAfterAxisCount_, ubBlock) + 
                     MoeCeilAlign(FP32_BYTES * eachCoreAfterAxisCount_, ubBlock) + 
                     GetSortTmpSize(indicesDtype_, 1, false);
    int64_t indicesSize = MoeCeilAlign(indicesTypeSize_, ubBlock) +
                          MoeCeilAlign(indicesTypeSize_ + SIZE_TWO * ALIGN_SIZE, ubBlock) + 
                          MoeCeilAlign(INT32_BYTES, ubBlock) + 
                          MoeCeilAlign(INT32_BYTES * SIZE_TWO, ubBlock) +
                          MoeCeilAlign(indicesTypeSize_, ubBlock) +
                          GetSortTmpSize(indicesDtype_, 1, false);                     
    if (occupy > halfUbSize) {
        int64_t indicesUbSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesSize);
        /* indicesBuf_ + outOfstBuf_ */
        ubIndexFactor_ = Ops::Base::CeilAlign(indicesUbSize, ALIGN_SIZE) / indicesSize;
        afterAxisFactor_ = (halfUbSize - ubIndexFactor_ * indicesSize) / ubIndexFactor_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, ALIGN_SIZE) / varTypeSize_;
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(eachCoreAfterAxisCount_, alignNum);
        ubIndexFactor_ = halfUbSize / (afterAxisFactor_ * (varTypeSize_ + FP32_BYTES) + indicesSize);
        
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            --ubIndexFactor_;
            restSize = halfUbSize - (MoeCeilAlign(ubIndexFactor_ * indicesTypeSize_, ubBlock) +
                                    MoeCeilAlign(ubIndexFactor_ * (indicesTypeSize_ + SIZE_TWO * ALIGN_SIZE), ubBlock) + 
                                    MoeCeilAlign(ubIndexFactor_ * INT32_BYTES, ubBlock) + 
                                    MoeCeilAlign(ubIndexFactor_ * (INT32_BYTES * SIZE_TWO), ubBlock) +
                                    MoeCeilAlign(ubIndexFactor_ * indicesTypeSize_, ubBlock) + 
                                    ubIndexFactor_ * MoeCeilAlign((FP32_BYTES) * eachCoreAfterAxisCount_, ubBlock) + 
                                    ubIndexFactor_ * MoeCeilAlign((varTypeSize_) * eachCoreAfterAxisCount_, ubBlock) + 
                                    GetSortTmpSize(indicesDtype_, ubIndexFactor_, false));
            if (ubIndexFactor_ > indicesAxis_) {
                ubIndexFactor_ = indicesAxis_;
                break;
            }
        }
    }

    /* 每个核分的indices相同 */
    indicesLoopSize_ = Ops::Base::CeilDiv(indicesAxis_, ubIndexFactor_);
    indiceAxisTailNum_ = indicesAxis_ - (indicesLoopSize_ - 1) * ubIndexFactor_;

    /* 主核循环次数 */
    updateLoopSize_ = Ops::Base::CeilDiv(eachCoreAfterAxisCount_, afterAxisFactor_);
    /* 主核尾loop处理afterAxis大小 */
    updateTailNum_ = eachCoreAfterAxisCount_ - (updateLoopSize_ - 1) * afterAxisFactor_;

    /* 尾核循环次数 */
    tailUpdateLoopSize_ = Ops::Base::CeilDiv(tailCoreAfterAxisCount_, afterAxisFactor_);
    /* 尾核尾loop处理afterAxis大小 */
    tailUpdateAxisNum_ = tailCoreAfterAxisCount_ - (tailUpdateLoopSize_ - 1) * afterAxisFactor_;
    isSplitAfterAxis_ = 1;
}

void MoeInplaceIndexAddSimdSortTiling::DoOpTilingSplitIndicesSort()
{
    auto ubBlock = Ops::Base::GetUbBlockSize(context_);
    int64_t alignNum = ubBlock / varTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / SIMD_DB_BUFFER);

    /* split indices分核 */
    eachCoreIndexCount_ = Ops::Base::CeilDiv(indicesAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(indicesAxis_, eachCoreIndexCount_);
    tailCoreIndexCount_ = indicesAxis_ - eachCoreIndexCount_ * (usedCoreNumBefore_ - 1);

    int64_t indicesSize = indicesTypeSize_ + (indicesTypeSize_ + SIZE_TWO * ubBlock) + 
                          INT32_BYTES + INT32_BYTES + indicesTypeSize_;
    int64_t indicesAlignSize = GetIndicesAlignBlockSize(ubIndexFactor_);
    int64_t afterAlignSize = GetAfterAlignBlockSize(ubIndexFactor_, afterAxis_);
    int64_t oneBlockSize = indicesAlignSize + afterAlignSize;

    if (oneBlockSize > halfUbSize) {
        int64_t indicesUbSize = std::min(INDICES_MIN_BLOCK_SIZE, GetIndicesAlignBlockSize(indicesAxis_));
        ubIndexFactor_ = Ops::Base::CeilAlign(indicesUbSize, static_cast<int64_t>(ubBlock)) / indicesSize;

        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            --ubIndexFactor_;
            restSize = INDICES_MIN_BLOCK_SIZE - GetIndicesAlignBlockSize(ubIndexFactor_);
        }
        ubIndexFactor_ = ubIndexFactor_ > eachCoreIndexCount_ ? eachCoreIndexCount_ : ubIndexFactor_;

        int64_t aviableSizeForUpdate = halfUbSize - GetIndicesAlignBlockSize(ubIndexFactor_);
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxis_, alignNum);
        restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            afterAxisFactor_ -= alignNum;
            restSize = aviableSizeForUpdate - GetAfterAlignBlockSize(ubIndexFactor_, afterAxisFactor_);
        }
        afterAxisFactor_ = afterAxisFactor_ == 0 ? 1 : afterAxisFactor_;
    } else {
        afterAxisFactor_ = std::min(AFTER_HANDLE_THRESHOLD / varTypeSize_, afterAxis_);
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxisFactor_, alignNum);

        ubIndexFactor_ = halfUbSize / GetAfterAlignBlockSize(1, afterAxisFactor_);
        int64_t restSize = static_cast<int64_t>(-1);
        ubIndexFactor_ += 1;
        while (restSize <= 0) {
            --ubIndexFactor_;
            restSize = halfUbSize - (GetIndicesAlignBlockSize(ubIndexFactor_) + 
                                     GetAfterAlignBlockSize(ubIndexFactor_, afterAxisFactor_));
        }
        ubIndexFactor_ = ubIndexFactor_ > eachCoreIndexCount_ ? eachCoreIndexCount_ : ubIndexFactor_;
    }
    /* 每个核分的update相同 */
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;

    mainCoreIndicesLoop_ = Ops::Base::CeilDiv(eachCoreIndexCount_, ubIndexFactor_);
    tailCoreIndicesLoop_ = Ops::Base::CeilDiv(tailCoreIndexCount_, ubIndexFactor_);
    mainCoreTailIndices_ = eachCoreIndexCount_ - (mainCoreIndicesLoop_ - 1) * ubIndexFactor_;
    tailCoreTailIndices_ = tailCoreIndexCount_ - (tailCoreIndicesLoop_ - 1) * ubIndexFactor_;
    isSplitIndicesAxis_ = 1;
}

void MoeInplaceIndexAddSimdSortTiling::DoOpTilingSplitPreSort()
{
    int64_t alignNum = ALIGN_SIZE / varTypeSize_;
    int64_t halfUbSize = static_cast<int64_t>((ubSize_ - RESERVE_SIZE) / SIMD_DB_BUFFER);

    /* split preaxis */
    eachCorePreAxisCount_ = Ops::Base::CeilDiv(preAxis_, totalCoreNum_);
    usedCoreNumBefore_ = Ops::Base::CeilDiv(preAxis_, eachCorePreAxisCount_);
    tailCorePreAxisCount_ = preAxis_ - eachCorePreAxisCount_ * (usedCoreNumBefore_ - 1);
    
    auto ubBlock = Ops::Base::GetUbBlockSize(context_);
    /* 同地址优化:搬入多少行indices,就搬入相同行数的updates
    * ubIndexFactor_: indiecesQue + (sortIndicesQue + 2 * shiftOfset) + originIdxQue +
    *                 (uniqueIdCntQue_ + 1) + updateSumIdxQue_,
    * ubIndexFactor_ * afterAxis_: updatesQue_ + updateSumQue_
    */
    int64_t indicesSize = MoeCeilAlign(indicesTypeSize_, ubBlock) +
                          MoeCeilAlign(indicesTypeSize_ + SIZE_TWO * ALIGN_SIZE, ubBlock) + 
                          MoeCeilAlign(INT32_BYTES, ubBlock) + 
                          MoeCeilAlign(INT32_BYTES * SIZE_TWO, ubBlock) +
                          MoeCeilAlign(indicesTypeSize_, ubBlock) +
                          GetSortTmpSize(indicesDtype_, 1, false);
    int64_t occupy = MoeCeilAlign(indicesTypeSize_, ubBlock) +
                     MoeCeilAlign(indicesTypeSize_ + SIZE_TWO * ALIGN_SIZE, ubBlock) + 
                     MoeCeilAlign(INT32_BYTES, ubBlock) + 
                     MoeCeilAlign(INT32_BYTES * SIZE_TWO, ubBlock) +
                     MoeCeilAlign(indicesTypeSize_, ubBlock) + 
                     MoeCeilAlign(varTypeSize_ * afterAxis_, ubBlock) + 
                     MoeCeilAlign(FP32_BYTES * afterAxis_, ubBlock) + 
                     GetSortTmpSize(indicesDtype_, 1, false);
    if (occupy > halfUbSize) {
        int64_t indicesUbSize = std::min(INDICES_MIN_BLOCK_SIZE, indicesAxis_ * indicesSize);
        /* indicesBuf_ + outOfstBuf_ */
        ubIndexFactor_ = Ops::Base::CeilAlign(indicesUbSize, ALIGN_SIZE) / indicesSize;
        afterAxisFactor_ = (halfUbSize - ubIndexFactor_ * indicesSize) / ubIndexFactor_;
        afterAxisFactor_ = Ops::Base::FloorAlign(afterAxisFactor_, ALIGN_SIZE) / varTypeSize_;
    } else {
        afterAxisFactor_ = Ops::Base::CeilAlign(afterAxis_, alignNum);
        ubIndexFactor_ = halfUbSize / (afterAxisFactor_ * (varTypeSize_ + FP32_BYTES) + indicesSize);
        
        int64_t restSize = static_cast<int64_t>(-1);
        while (restSize <= 0) {
            --ubIndexFactor_;
            restSize = halfUbSize - (MoeCeilAlign(ubIndexFactor_ * indicesTypeSize_, ubBlock) +
                                    MoeCeilAlign(ubIndexFactor_ * (indicesTypeSize_ + SIZE_TWO * ALIGN_SIZE), ubBlock) + 
                                    MoeCeilAlign(ubIndexFactor_ * INT32_BYTES, ubBlock) + 
                                    MoeCeilAlign(ubIndexFactor_ * (INT32_BYTES * SIZE_TWO), ubBlock) +
                                    MoeCeilAlign(ubIndexFactor_ * indicesTypeSize_, ubBlock) + 
                                    ubIndexFactor_ * MoeCeilAlign((varTypeSize_) * eachCoreAfterAxisCount_, ubBlock) + 
                                    ubIndexFactor_ * MoeCeilAlign((FP32_BYTES) * eachCoreAfterAxisCount_, ubBlock) + 
                                    GetSortTmpSize(indicesDtype_, ubIndexFactor_, false));
            if (ubIndexFactor_ > indicesAxis_) {
                ubIndexFactor_ = indicesAxis_;
                break;
            }
        }
    }
    /* 每个核分的update相同 */
    updateLoopSize_ = Ops::Base::CeilDiv(afterAxis_, afterAxisFactor_);
    updateTailNum_ = afterAxis_ - (updateLoopSize_ - 1) * afterAxisFactor_;

    /* 每个核处理的indices相同 */
    indicesLoopSize_ = Ops::Base::CeilDiv(indicesAxis_, ubIndexFactor_);
    indiceAxisTailNum_ = indicesAxis_ - (indicesLoopSize_ - 1) * ubIndexFactor_;
    isSplitPreAxis_ = 1;
}

ge::graphStatus MoeInplaceIndexAddSimdSortTiling::DoOpTiling()
{
    usedCoreNum_ = totalCoreNum_;
    if (afterAxis_ * varTypeSize_ >= AFTER_HANDLE_THRESHOLD * usedCoreNum_) {
        DoOpTilingSplitAfterSort();
    } else if (indicesAxis_ > preAxis_) {
        DoOpTilingSplitIndicesSort();
    } else {
        DoOpTilingSplitPreSort();
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddSimdSortTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInplaceIndexAddSimdSortTiling::GetTilingKey() const
{
    uint64_t tilingKey = SIMD_SORT_TILING_KEY;
    return tilingKey;
}

ge::graphStatus MoeInplaceIndexAddSimdSortTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInplaceIndexAddSimdSortTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    tilingKey_ = GetTilingKey();
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    context_->SetTilingKey(tilingKey_);
    workspaces[0] = workspaceSize_;
    auto res = context_->SetLocalMemorySize(ubSize_);
    MOE_OP_TILING_CHECK(
        (res != ge::GRAPH_SUCCESS),
        MOE_VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeInplaceIndexAddSimdSortTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "preAxis: " << tilingData_.get_preAxis();
    info << ", varInAxis: " << tilingData_.get_varInAxis();
    info << ", updatesInAxis: " << tilingData_.get_updatesInAxis();
    info << ", indicesStride: " << tilingData_.get_indicesStride();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", afterAxis: " << tilingData_.get_afterAxis();
    info << ", tilingKey: " << tilingKey_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

void MoeInplaceIndexAddSimdSortTiling::SetTilingData()
{
    tilingData_.set_preAxis(preAxis_);
    tilingData_.set_varInAxis(varInAxis_);
    tilingData_.set_updatesInAxis(updatesInAxis_);
    tilingData_.set_afterAxis(afterAxis_);
    tilingData_.set_usedCoreNumBefore(usedCoreNumBefore_);
    tilingData_.set_ubIndexFactor(ubIndexFactor_);
    tilingData_.set_eachCoreIndexCount(eachCoreIndexCount_);
    tilingData_.set_tailCoreIndexCount(tailCoreIndexCount_);
    tilingData_.set_eachCorePreAxisCount(eachCorePreAxisCount_);
    tilingData_.set_eachCoreAfterAxisCount(eachCoreAfterAxisCount_);
    tilingData_.set_tailCorePreAxisCount(tailCorePreAxisCount_);
    tilingData_.set_tailCoreAfterAxisCount(tailCoreAfterAxisCount_);
    tilingData_.set_indicesStride(indicesStride_);

    tilingData_.set_mainCoreIndicesLoop(mainCoreIndicesLoop_);
    tilingData_.set_tailCoreIndicesLoop(tailCoreIndicesLoop_);
    tilingData_.set_mainCoreTailIndices(mainCoreTailIndices_);
    tilingData_.set_tailCoreTailIndices(tailCoreTailIndices_);

    tilingData_.set_updateLoopSize(updateLoopSize_);
    tilingData_.set_updateTailNum(updateTailNum_);
    tilingData_.set_indicesLoopSize(indicesLoopSize_);
    tilingData_.set_indiceAxisTailNum(indiceAxisTailNum_);
    tilingData_.set_tailUpdateLoopSize(tailUpdateLoopSize_);
    tilingData_.set_tailUpdateAxisNum(tailUpdateAxisNum_);
    tilingData_.set_isSplitPreAxis(isSplitPreAxis_);
    tilingData_.set_isSplitAfterAxis(isSplitAfterAxis_);
    tilingData_.set_isSplitIndicesAxis(isSplitIndicesAxis_);
    tilingData_.set_afterAxisFactor(afterAxisFactor_);
    tilingData_.set_isWithAlpha(withAlpha_);
    return;
}

REGISTER_OPS_TILING_TEMPLATE(MoeInplaceIndexAdd, MoeInplaceIndexAddSimdSortTiling, 0);
}  // namespace optiling
