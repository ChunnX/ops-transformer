/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_mla_value_tiling_nonquant.h"
#include "flash_mla_value_tiling_common.h"
#include "../../../incre_flash_attention/op_host/incre_flash_attention_tiling_base.h"
#include "log/log.h"
#include "err/ops_err.h"
#include <algorithm>
#include <numeric>
#include <cstring>

namespace FlashMla {
namespace Arch22 {
using namespace optiling;
constexpr uint64_t PRE_LOAD_NUM_MLA = 2;
template <typename T>
inline T Align(T num, T rnd)
{
    return rnd == 0 ? 0 : ((num + rnd - 1) / rnd * rnd);
}

// Prepare sequence views and the original tiler state, run the V4 scheduling
// methods below, then export dynamic fields. Host owns static fields and launch.
MlaStatus MlaNonQuantValueState::Compute(const MlaScheduleConfig &config, const MlaPlatformInfo &platform,
                                         MlaSeqView seqQ, MlaSeqView seqKv, MlaValuePatch &patch)
{
    InitializeMlaPatch(config, platform, MlaFamily::ARCH22_MLA, patch);
    // Descriptor checks have already run in ComputeMlaValueTiling.
    auto status = ValidateMlaSeqValues(config, seqQ, seqKv);
    if (status != MlaStatus::SUCCESS) {
        return patch.header.status = status;
    }
    if (config.queryDataType != kMlaFp16 && config.queryDataType != kMlaBf16) {
        return patch.header.status = MlaStatus::UNSUPPORTED;
    }
    auto qTensor = MlaSeqTensor(seqQ);
    auto kvTensor = MlaSeqTensor(seqKv);
    MlaValueParser parser;
    parser.opName_ = "FlashMlaMetadata";
    parser.bSize_ = config.batchSize;
    parser.s1Size_ = static_cast<uint32_t>(config.qSeqSize);
    parser.s2Size_ = static_cast<int64_t>(config.kvSeqSize);
    parser.qLayout_ = MlaFiaLayout(config.layoutQ);
    parser.kvLayout_ = MlaFiaLayout(config.layoutKv);
    parser.blockSize_ = config.blockSize;
    parser.kvStorageMode_ = config.blockSize ? KvStorageMode::PAGE_ATTENTION : KvStorageMode::BATCH_CONTINUOUS;
    parser.opParamInfo_.actualSeqLengthsQ.tensor = config.seqQPresent ? &qTensor : nullptr;
    parser.opParamInfo_.actualSeqLengths.tensor = config.seqKvPresent ? &kvTensor : nullptr;
    if (config.layoutQ == MlaLayout::TND) {
        parser.ComputeQMax(seqQ.data, config.batchSize);
    } else {
        parser.qSize_.push_back(config.qSeqSize);
    }
    if (config.blockSize != 0) {
        if (parser.GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS) {
            return patch.header.status = MlaStatus::INVALID_SEQ;
        }
        parser.s2Size_ = static_cast<int64_t>(parser.maxBlockNumPerBatch_) * config.blockSize;
    } else if (config.layoutKv == MlaLayout::TND && config.seqKvPresent) {
        parser.GetMaxActualSeq(&kvTensor, parser.kvLayout_, parser.s2Size_);
    } else {
        parser.kvListSeqLens_.push_back(config.kvSeqSize);
        parser.kvSize_.push_back(config.kvSeqSize);
    }
    parser.GetActualSeqInfo();

    // The same original value checks run before the split in the FIA Host path.
    MlaValueSeqCheck checker;
    checker.opParamInfo_ = parser.opParamInfo_;
    checker.bSize_ = parser.bSize_;
    checker.s1Size_ = parser.s1Size_;
    checker.s2Size_ = parser.s2Size_;
    checker.qTSize_ = static_cast<uint32_t>(config.qSeqSize);
    checker.kTSize_ = static_cast<uint32_t>(config.kvSeqSize);
    checker.qLayout_ = parser.qLayout_;
    checker.kvLayout_ = parser.kvLayout_;
    if (checker.CheckFeatureActualSeqLensQData() != ge::GRAPH_SUCCESS ||
        checker.CheckFeatureActualSeqLensKvData() != ge::GRAPH_SUCCESS) {
        return patch.header.status = MlaStatus::INVALID_SEQ;
    }
    // Original split counters multiply Q by heads and round up in uint32_t.
    // Reject an unsupported workload before any of those calculations run.
    const uint64_t groupedQuerySize = static_cast<uint64_t>(parser.s1Size_) * config.numHeads;
    if (groupedQuerySize > UINT32_MAX - (mBaseSize_ - 1U)) {
        return patch.header.status = MlaStatus::OVERFLOW;
    }
    // FIAChecker consumes the parser's lengths: stored S for padded dense
    // layouts, per-batch differences for TND, and no kvSize_ entries for PA.
    // The sequence checker has different semantics and cannot replace it.
    // Preserve the original exception for a zero KV length.
    if (config.sparseMode == SPARSE_MODE_TREE) {
        const size_t count = std::min(parser.qSize_.size(), parser.kvSize_.size());
        for (size_t i = 0; i < count; ++i) {
            if (parser.kvSize_[i] != 0 && parser.qSize_[i] > parser.kvSize_[i]) {
                return patch.header.status = MlaStatus::INVALID_SEQ;
            }
        }
    }
    FiaTilingInfo info{};
    info.opName = "FlashMlaMetadata";
    info.bSize = config.batchSize;
    info.n1Size = info.gSize = config.numHeads;
    info.n2Size = 1;
    info.qkHeadDim = info.vHeadDim = 512;
    info.ropeHeadDim = 64;
    info.s1Size = parser.s1Size_;
    info.s2Size = parser.s2Size_;
    info.inputLayout = MlaTilingLayout(config.layoutQ);
    info.inputKvLayout = MlaTilingLayout(config.layoutKv);
    info.actualLenQDims = parser.actualLenQDims_;
    info.actualLenKvDims = parser.actualLenDims_;
    info.isAccumQSeq = parser.isAccumQSeq_;
    info.isAccumKVSeq = parser.isAccumKVSeq_;
    info.actualSeqLenFlag = config.seqKvPresent;
    info.batchContinuousFlag = true;
    info.isSameActualseq = parser.isSameActualseq_;
    info.needInit = parser.needInit_;
    info.pageAttentionFlag = config.blockSize != 0;
    info.blockSize = config.blockSize;
    // Original GetFiaTilingInfo overrides this field with blockTableDim2_ on
    // ASCEND910B. It is the stored table row stride, whereas parser.s2Size_
    // above deliberately follows the maximum number of actually used blocks.
    info.maxBlockNumPerBatch = config.blockSize == 0 ? 0 : config.kvSeqSize / config.blockSize;
    info.sparseMode = config.sparseMode;
    info.attenMaskFlag = config.hasAttenMask;
    info.preToken = config.preTokens;
    info.nextToken = config.nextTokens;
    info.opParamInfo = parser.opParamInfo_;
    fiaInfo_ = &info;
    aicNum_ = platform.aicNum;
    aivNum_ = platform.aivNum;
    cvRatio_ = platform.aivNum / platform.aicNum;
    // The producer computes only user workspace; Host adds its system reservation.
    libapiSize_ = 0;
    InitParams();
    Split();
    tilingData_.baseParams.s1Size = info.s1Size;
    tilingData_.baseParams.s2Size = info.s2Size;
    tilingData_.baseParams.needInit = info.needInit;
    tilingData_.baseParams.usedCoreNum = usedCoreNum_;
    FillTilingWorkspaceParams();
    CalcWorkspaceSize();
    static_assert(sizeof(tilingData_) == kMlaNonQuantPayloadBytes, "MLA value tiling layout");
    CopyMlaDynamicFields(MlaFamily::ARCH22_MLA, patch.payload, reinterpret_cast<const uint8_t *>(&tilingData_));
    // The producer computes user workspace. Host tiling adds its own system reservation.
    patch.header.workspaceNeed = workspaceSize_;
    patch.header.originalBlockDim = usedCoreNum_;
    patch.header.dynamicFlags = splitKVFlag_ ? kMlaFlagFd : 0;
    patch.header.maxQ = parser.s1Size_;
    // The original parser's initialization flags follow the query layout.
    // Metadata needs KV-layout lengths: PA is absolute even with a TND query.
    patch.header.maxKv = checker.kvSize_.empty() ?
                             0 :
                             static_cast<uint64_t>(*std::max_element(checker.kvSize_.begin(), checker.kvSize_.end()));
    for (uint32_t i = 0; i < config.batchSize; ++i) {
        const uint64_t q = checker.qSize_.empty() ?
                               config.qSeqSize :
                               static_cast<uint64_t>(checker.qSize_[checker.qSize_.size() == 1 ? 0 : i]);
        uint64_t square;
        if (!MlaCheckedMul(q, q, square) ||
            !MlaCheckedAdd(patch.header.qSeqSquareSum, square, patch.header.qSeqSquareSum)) {
            return patch.header.status = MlaStatus::OVERFLOW;
        }
    }
    return patch.header.status = MlaStatus::SUCCESS;
}

MlaStatus MlaNonQuantValueState::WorkspaceCapacity(const MlaScheduleConfig &config, const MlaPlatformInfo &platform,
                                                   uint64_t &capacity)
{
    FiaTilingInfo info{};
    info.qkHeadDim = info.vHeadDim = 512;
    info.n2Size = 1;
    fiaInfo_ = &info;
    aicNum_ = coreNum_ = platform.aicNum;
    libapiSize_ = 0;
    headDimAlign_ = 512;
    CalcMaxWorkspaceSize();
    return MlaCheckedAdd(platform.sysWorkspaceSize, workspaceSize_, capacity) ? MlaStatus::SUCCESS :
                                                                                MlaStatus::OVERFLOW;
}

// Mechanically extracted from
// attention/fused_infer_attention_score/op_host/arch22/fused_infer_attention_score_tiling_check_feature.cpp. Private
// FlashMla value-state methods; the original operator is unchanged.

ge::graphStatus MlaValueSeqCheck::GetActualSeqLenSize(const gert::Tensor *tensor, const FiaLayout &layout,
                                                      const std::string &actualSeqLenName, const std::string &attrName,
                                                      uint32_t &size)
{
    if (tensor == nullptr) {
        OP_LOGE(opName_, "when layout of %s is %u, %s must be provided.", attrName.c_str(),
                static_cast<uint32_t>(layout), actualSeqLenName.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "%s shape size is %ld, it should be greater than 0.", actualSeqLenName.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaValueSeqCheck::CheckFeatureActualSeqLensQData()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        qSize_.push_back(s1Size_);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t *actualSeq = opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>();
    if (actualSeq == nullptr) {
        qSize_.push_back(s1Size_);
        return ge::GRAPH_SUCCESS;
    }

    if (GetActualSeqLenSize(opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, ACTUAL_SEQ_Q_LEN_NAME, QUERY_NAME,
                            actualSeqLengthsQSize_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t loop = std::min(actualSeqLengthsQSize_, bSize_);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS1 = 0;
        if (qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD) {
            OP_CHECK_IF(actualSeq[i] < 0,
                        OP_LOGE(opName_, "when %s's layout is %u, %s[%u] should not be a negative number, but got %ld.",
                                QUERY_NAME.c_str(), static_cast<uint32_t>(qLayout_), ACTUAL_SEQ_Q_LEN_NAME.c_str(), i,
                                actualSeq[i]),
                        return ge::GRAPH_FAILED);

            OP_CHECK_IF(i > 0U && (actualSeq[i] < actualSeq[i - 1U]),
                        OP_LOGE(opName_, "when %s's layout is %u, %s[%u](%ld) should not be less than %s[%u](%ld).",
                                QUERY_NAME.c_str(), static_cast<uint32_t>(qLayout_), ACTUAL_SEQ_Q_LEN_NAME.c_str(), i,
                                actualSeq[i], ACTUAL_SEQ_Q_LEN_NAME.c_str(), (i - 1U), actualSeq[i - 1U]),
                        return ge::GRAPH_FAILED);

            tmpS1 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS1 = actualSeq[i];
        }
        if (tmpS1 > static_cast<int64_t>(s1Size_) || tmpS1 < 0) {
            OP_LOGE(opName_, "%s[%u] computed is %ld, it should be in range [0, Q_S(%u)].",
                    ACTUAL_SEQ_Q_LEN_NAME.c_str(), i, tmpS1, s1Size_);
            return ge::GRAPH_FAILED;
        }
        qSize_.push_back(tmpS1);
    }

    OP_CHECK_IF((qLayout_ == FiaLayout::TND) && (qTSize_ != actualSeq[actualSeqLengthsQSize_ - 1]),
                OP_LOGE(opName_, "when %s's layout is %u, T(%u) should be equal to the last element of %s(%ld).",
                        QUERY_NAME.c_str(), static_cast<uint32_t>(qLayout_), qTSize_, ACTUAL_SEQ_Q_LEN_NAME.c_str(),
                        actualSeq[actualSeqLengthsQSize_ - 1]),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaValueSeqCheck::CheckFeatureActualSeqLensKvData()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        kvSize_.push_back(s2Size_);
        return ge::GRAPH_SUCCESS;
    }

    const int64_t *actualSeq = opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>();
    if (actualSeq == nullptr) {
        kvSize_.push_back(s2Size_);
        return ge::GRAPH_SUCCESS;
    }

    if (GetActualSeqLenSize(opParamInfo_.actualSeqLengths.tensor, kvLayout_, ACTUAL_SEQ_KV_LEN_NAME, KEY_NAME,
                            actualSeqLengthsKvSize_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t loop = std::min(actualSeqLengthsKvSize_, bSize_);
    for (uint32_t i = 0; i < loop; i++) {
        int64_t tmpS2 = 0;
        if (kvLayout_ == FiaLayout::TND || kvLayout_ == FiaLayout::NTD) {
            OP_CHECK_IF(actualSeq[i] < 0,
                        OP_LOGE(opName_, "when kv's layout is %u, %s[%u] should not be a negative number, but got %ld.",
                                static_cast<uint32_t>(kvLayout_), ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, actualSeq[i]),
                        return ge::GRAPH_FAILED);

            OP_CHECK_IF(i > 0U && (actualSeq[i] < actualSeq[i - 1U]),
                        OP_LOGE(opName_, "when kv's layout is %u, %s[%u](%ld) should not be less than %s[%u](%ld).",
                                static_cast<uint32_t>(kvLayout_), ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, actualSeq[i],
                                ACTUAL_SEQ_KV_LEN_NAME.c_str(), (i - 1U), actualSeq[i - 1U]),
                        return ge::GRAPH_FAILED);

            tmpS2 = (i == 0U) ? actualSeq[0] : (actualSeq[i] - actualSeq[i - 1U]);
        } else {
            tmpS2 = actualSeq[i];
        }

        OP_CHECK_IF(tmpS2 < 0 || tmpS2 > s2Size_,
                    OP_LOGE(opName_, "%s(%u) is %ld, it should be in range [0, KV_S(%ld)].",
                            ACTUAL_SEQ_KV_LEN_NAME.c_str(), i, tmpS2, s2Size_),
                    return ge::GRAPH_FAILED);
        kvSize_.push_back(tmpS2);
    }

    OP_CHECK_IF((kvLayout_ == FiaLayout::TND) && (kTSize_ != actualSeq[actualSeqLengthsKvSize_ - 1]),
                OP_LOGE(opName_, "when kv's layout is %u, T(%u) should be equal to the last element of %s(%ld).",
                        static_cast<uint32_t>(kvLayout_), kTSize_, ACTUAL_SEQ_KV_LEN_NAME.c_str(),
                        actualSeq[actualSeqLengthsKvSize_ - 1]),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Mechanically extracted from
// attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp. Private FlashMla
// value-state methods; the original operator is unchanged.

ge::graphStatus MlaValueParser::GetMaxActualSeq(const gert::Tensor *actualSeqLensTensor, FiaLayout layout,
                                                int64_t &maxActualSeqLen)
{
    maxActualSeqLen = 0;
    const int64_t *actualLenData = actualSeqLensTensor->GetData<int64_t>();
    if (actualLenData != nullptr) {
        uint32_t loop = std::min(static_cast<uint32_t>(actualSeqLensTensor->GetShapeSize()), bSize_);
        for (uint32_t i = 0; i < loop; i++) {
            int64_t tmpS = 0;
            if (layout == FiaLayout::TND || layout == FiaLayout::NTD) {
                tmpS = (i == 0U) ? actualLenData[0] : (actualLenData[i] - actualLenData[i - 1U]);
            } else {
                tmpS = actualLenData[i];
            }
            kvSize_.push_back(tmpS);
            maxActualSeqLen = std::max(maxActualSeqLen, tmpS);
        }
    }
    if (isMaxWorkspace_) {
        maxActualSeqLen = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaValueParser::GetMaxBlockNumPerBatch()
{
    if (antiQuantFlag_) {
        uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
        if (dimNum != 2U) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "block_table", std::to_string(dimNum).c_str(), "2");
            return ge::GRAPH_FAILED;
        }
        maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    } else {
        const gert::Tensor *actSeqLenKV = opParamInfo_.actualSeqLengths.tensor;
        int32_t actualSeqKVPerBatch = 0;
        int32_t blockNumPerBatch = 0;
        int64_t blockNumValid = 0;
        maxBlockNumPerBatch_ = 0;
        if (actSeqLenKV == nullptr || actSeqLenKV->GetData<int64_t>() == nullptr || blockSize_ == 0) {
            return ge::GRAPH_SUCCESS;
        }
        uint32_t loops = std::min(bSize_, static_cast<uint32_t>(actSeqLenKV->GetShapeSize()));
        for (uint32_t i = 0; i < loops; i++) {
            actualSeqKVPerBatch = (actSeqLenKV->GetShapeSize() > 1) ?
                                      static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[i]) :
                                      static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[0]);
            blockNumPerBatch = (static_cast<int64_t>(actualSeqKVPerBatch) + blockSize_ - 1) / blockSize_;
            blockNumValid += blockNumPerBatch;
            if (blockNumPerBatch > maxBlockNumPerBatch_) {
                maxBlockNumPerBatch_ = blockNumPerBatch;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

void MlaValueParser::ProcessKVActualSeqLoop(const int64_t *actualLenData, uint32_t loop)
{
    int64_t preActualLen = 0;
    for (uint32_t i = 0; i < loop; i++) {
        int64_t actLen = actualLenData[i];
        if (antiQuantFlag_) {
            if (qLayout_ == FiaLayout::TND && i > 0 && kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
                actLen -= actualLenData[i - 1];
            }
            if (s1Size_ == 1) {
                needInit_ = needInit_ || (actLen == 0);
            }
        } else {
            if (qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD) {
                actLen -= preActualLen;
                preActualLen = actualLenData[i];
            }
            needInit_ = needInit_ || (actLen == 0);
            if (actLen != s2Size_ && qLayout_ != FiaLayout::TND && qLayout_ != FiaLayout::NTD) {
                needInit_ = 1;
            }
        }
        maxActualseq_ = maxActualseq_ < actLen ? actLen : maxActualseq_;
        if (actualLenData[i] != actualLenData[0]) {
            isSameActualseq_ = false;
        }
    }
}

void MlaValueParser::ProcessQActualSeqLoop(const int64_t *actualLenQData, uint32_t loop)
{
    int64_t preActualLen = 0;
    for (uint32_t i = 0; i < loop; i++) {
        int64_t actLen = actualLenQData[i];
        if (qLayout_ == FiaLayout::TND || qLayout_ == FiaLayout::NTD) {
            actLen -= preActualLen;
            preActualLen = actualLenQData[i];
        }
        if (actLen != static_cast<int64_t>(s1Size_)) {
            needInit_ = true;
            break;
        }
    }
}

ge::graphStatus MlaValueParser::GetActualSeqInfo()
{
    maxActualseq_ = 0;
    if (isMaxWorkspace_) {
        maxActualseq_ = s2Size_;
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
        actualLenDims_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
        if ((kvLayout_ == FiaLayout::TND) || (kvLayout_ == FiaLayout::NTD)) {
            isAccumKVSeq_ = true;
        }
        if (opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>() != nullptr) {
            const int64_t *actualLenData = opParamInfo_.actualSeqLengths.tensor->GetData<int64_t>();
            uint32_t loop = ((actualLenDims_ == 1) && (kvListSeqLens_.size() == 1)) ? 1 : bSize_;
            loop = std::min(loop, actualLenDims_);
            ProcessKVActualSeqLoop(actualLenData, loop);
        }
    } else {
        maxActualseq_ = s2Size_;
    }

    if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) {
        const int64_t *actualLenQData = opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>();
        actualLenQDims_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
        if ((qLayout_ == FiaLayout::TND) || (qLayout_ == FiaLayout::NTD)) {
            isAccumQSeq_ = true;
        }
        if (opParamInfo_.actualSeqLengthsQ.tensor->GetData<int64_t>() != nullptr) {
            uint32_t loop = std::min(bSize_, actualLenQDims_);
            ProcessQActualSeqLoop(actualLenQData, loop);
        }
    }

    if ((quantMode_ == FiaQuantMode::ANTI_QUANT) && (s1Size_ == 1)) {
        if (qLayout_ != FiaLayout::TND) {
            actualLenQDims_ = 0;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void MlaValueParser::ComputeQMax(const int64_t *actualSeqQ, uint32_t b)
{
    int64_t qActualSeqMax = 0;
    for (uint32_t i = 0; i < b; i++) {
        int64_t tmpS1 = (i == 0U) ? actualSeqQ[0] : (actualSeqQ[i] - actualSeqQ[i - 1U]);
        qSize_.push_back(tmpS1);
        if (tmpS1 > qActualSeqMax) {
            qActualSeqMax = tmpS1;
        }
    }
    s1Size_ = static_cast<uint32_t>(qActualSeqMax);
}
// Mechanically extracted from attention/fused_infer_attention_score/op_host/arch22/fia_tiling_nonquant_mla.cpp.
// Private FlashMla value-state methods; the original operator is unchanged.

bool MlaNonQuantValueState::IsFlashDecode()
{
    uint32_t tndFDCoreArrLen = tilingData_.fdParams.numOfFdHead;
    return tndFDCoreArrLen > 0U;
}

bool MlaNonQuantValueState::DealSameSeqEachBatch() const
{
    if (!fiaInfo_->batchContinuousFlag) {
        if (fiaInfo_->actualSeqLenFlag) {
            return fiaInfo_->isSameActualseq;
        } else {
            return fiaInfo_->isSameSeqAllKVTensor;
        }
    } else {
        return fiaInfo_->isSameActualseq;
    }
}

void MlaNonQuantValueState::EmptyTensorProcess() const
{
    if (fiaInfo_->s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        fiaInfo_->s2Size = 1024;
    }
}

void MlaNonQuantValueState::InitParams()
{
    perfMode_ = FiaTemplateId::HIGH_PERFORMANCE_MLA;
    coreNum_ = aicNum_;
    numBlocks_ =
        aicNum_; // Tiling下沉首次Tiling也会校验numBlocks_是否为0，为避免拦截报错，将numBlocks_设置为aicNum_，实际不生效

    headDimAlign_ = Align(fiaInfo_->qkHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    EmptyTensorProcess();
}

void MlaNonQuantValueState::CalcInnerSize(uint32_t seqSize)
{
    sInnerSize_ = 512U;
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && fiaInfo_->inputLayout != TilingKeyLayout::TND) {
        if (seqSize == 256U) {
            sInnerSize_ = 128U;
        } else if (seqSize > 256U && seqSize <= sInnerSize_) {
            sInnerSize_ = (sInnerSize_ + 1U) / 2U;
        }
    }

    // PA特性泛化场景，blockSize可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
    if (fiaInfo_->pageAttentionFlag && fiaInfo_->blockSize != 0) {
        uint32_t blockSize = static_cast<uint32_t>(fiaInfo_->blockSize);
        if ((sInnerSize_ > blockSize) && (sInnerSize_ % blockSize != 0)) {
            sInnerSize_ = (sInnerSize_ / blockSize) * blockSize;
        }
    }

    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
    // tiling下沉 && flash decoder场景时，sInnerSize_基块大小不按照真实值修改
    // 否则会导致 tiling下沉 && flash decoder 场景时开辟workspace空间大小小于真实运行时所需的workspace大小
    if (sInnerSize_ > seqSize) {
        sInnerSize_ = seqSize;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CalcMBaseSize();
}

void MlaNonQuantValueState::CalcMBaseSize()
{
    uint32_t bN2 = fiaInfo_->bSize * fiaInfo_->n2Size;
    uint32_t s1G = fiaInfo_->s1Size * fiaInfo_->gSize;
    if (bN2 == 0U || aicNum_ == 0U) {
        return;
    }
    // 若BN2能整除核数或者KVS不等长，不考虑调整mBaseSize
    if (bN2 % aicNum_ == 0U || !DealSameSeqEachBatch()) {
        return;
    }

    // 求B*N2与核数的最小公倍数
    uint32_t originalA = bN2;
    uint32_t originalB = aicNum_;
    // 欧几里得算法计算最大公约数
    while (originalB != 0U) {
        originalA %= originalB;
        std::swap(originalA, originalB);
    }

    // 计算最小公倍数
    uint32_t lcm = (bN2 / originalA) * aicNum_;
    uint32_t splitOfS1G = lcm / bN2; // 求S1G被切多少份，可以不开FD
    // 如果不能等分S1G，则不切S2无法做到负载不均衡，不修改mBaseSize，直接按照最大mBaseSize进行切分
    if (splitOfS1G == 0U || s1G % splitOfS1G != 0U) {
        return;
    }
    uint32_t mBaseSizeTmp = s1G / splitOfS1G; // 计算可以等分S1G的mBaseSize
    // 128：最小的mBaseSize，计算的理论mBaseSize需要在原定mBaseSize和最小mBaseSize区间内
    if (mBaseSizeTmp > mBaseSize_ || mBaseSizeTmp < 128U) {
        return;
    }

    mBaseSize_ = mBaseSizeTmp; // 满足以上条件则更新mBaseSize
}

void MlaNonQuantValueState::CreateSplitInput(BaseInfo &baseInfo, SplitParam &splitParam) const
{
    // 构造分核输入参数
    baseInfo.bSize = fiaInfo_->bSize;
    baseInfo.n2Size = fiaInfo_->n2Size;
    baseInfo.gSize = fiaInfo_->gSize;
    baseInfo.s2Size = static_cast<uint32_t>(fiaInfo_->s2Size);
    baseInfo.s1Size = fiaInfo_->s1Size;
    baseInfo.actualLenQDims = fiaInfo_->actualLenQDims;
    baseInfo.actualLenKvDims = fiaInfo_->actualLenKvDims;
    baseInfo.preToken = fiaInfo_->preToken;
    baseInfo.nextToken = fiaInfo_->nextToken;
    baseInfo.isS1G = fiaInfo_->inputLayout == TilingKeyLayout::TND ||
                     fiaInfo_->inputLayout == TilingKeyLayout::BSH_BSND; // 使用枚举映射
    baseInfo.sparseMode = fiaInfo_->sparseMode;
    baseInfo.attenMaskFlag = fiaInfo_->attenMaskFlag;

    if (fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor != nullptr) {
        baseInfo.isAccumSeqS1 = fiaInfo_->isAccumQSeq;
        baseInfo.actualSeqS1Size.reserve(fiaInfo_->bSize);
        const int64_t *s1Ptr = fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>();
        // A one-element view is broadcast by GetS1SeqSize; copy only stored values.
        const uint32_t count = std::min(fiaInfo_->bSize, fiaInfo_->actualLenQDims);
        for (uint32_t i = 0; i < count; ++i) {
            baseInfo.actualSeqS1Size.emplace_back(s1Ptr[i]);
        }
    }
    if (fiaInfo_->opParamInfo.actualSeqLengths.tensor != nullptr) {
        baseInfo.isAccumSeqS2 = fiaInfo_->isAccumKVSeq;
        baseInfo.actualSeqS2Size.reserve(fiaInfo_->bSize);
        const int64_t *s2Ptr = fiaInfo_->opParamInfo.actualSeqLengths.tensor->GetData<int64_t>();
        // A one-element view is broadcast by GetS2SeqSize; copy only stored values.
        const uint32_t count = std::min(fiaInfo_->bSize, fiaInfo_->actualLenKvDims);
        for (uint32_t i = 0; i < count; ++i) {
            baseInfo.actualSeqS2Size.emplace_back(s2Ptr[i]);
        }
    }
    splitParam.mBaseSize = mBaseSize_;
    splitParam.s2BaseSize = sInnerSize_;
    splitParam.gS1BaseSizeOfFd = mFdBaseSize_;
}

void MlaNonQuantValueState::SetSplitOutput(const SplitResult &res)
{
    uint32_t *bN2EndPtr = tilingData_.outerSplitParams.bN2End;
    uint32_t *gS1EndPtr = tilingData_.outerSplitParams.gS1End;
    uint32_t *s2EndPtr = tilingData_.outerSplitParams.s2End;
    uint32_t *bN2IdxOfFdHead = tilingData_.fdParams.bN2IdxOfFdHead;
    uint32_t *gS1IdxOfFdHead = tilingData_.fdParams.gS1IdxOfFdHead;
    uint32_t *s2SplitNumOfFdHead = tilingData_.fdParams.s2SplitNumOfFdHead;
    uint32_t *s2SplitStartIdxOfCore = tilingData_.fdParams.s2SplitStartIdxOfCore;
    uint32_t *gS1SplitNumOfFdHead = tilingData_.fdParams.gS1SplitNumOfFdHead;
    uint32_t *gS1LastPartSizeOfFdHead = tilingData_.fdParams.gS1LastPartSizeOfFdHead;
    uint32_t *gS1IdxEndOfFdHeadSplit = tilingData_.fdParams.gS1IdxEndOfFdHeadSplit;
    uint32_t *gS1IdxEndOfFdHead = tilingData_.fdParams.gS1IdxEndOfFdHead;

    for (uint32_t i = 0; i < res.usedCoreNum; ++i) {
        bN2EndPtr[i] = res.bN2End[i];
        gS1EndPtr[i] = res.gS1End[i];
        s2EndPtr[i] = res.s2End[i];
        bN2IdxOfFdHead[i] = res.fdRes.bN2IdxOfFdHead[i];
        gS1IdxOfFdHead[i] = res.fdRes.gS1IdxOfFdHead[i];
        s2SplitNumOfFdHead[i] = res.fdRes.s2SplitNumOfFdHead[i];
        s2SplitStartIdxOfCore[i] = res.fdRes.s2SplitStartIdxOfCore[i];
        gS1SplitNumOfFdHead[i] = res.fdRes.gS1SplitNumOfFdHead[i];
        gS1LastPartSizeOfFdHead[i] = res.fdRes.gS1LastPartSizeOfFdHead[i];
    }

    // SplitResult allocates exactly aicNum * vecCubeRatio entries. CV1 must not
    // read the CV2-only half (ASAN regression: B4/H8/Q4/KV4096, AIC24/CV1).
    for (uint32_t i = 0; i < res.usedCoreNum * res.vecCubeRatio; ++i) {
        gS1IdxEndOfFdHead[i] = res.fdRes.gS1IdxEndOfFdHead[i];
        gS1IdxEndOfFdHeadSplit[i] = res.fdRes.gS1IdxEndOfFdHeadSplit[i];
    }
}

void MlaNonQuantValueState::Split()
{
    CalcInnerSize(static_cast<uint32_t>(fiaInfo_->s2Size));

    // 构造分核输入参数
    BaseInfo baseInfo{};
    SplitParam splitParam{};
    CreateSplitInput(baseInfo, splitParam);

    // 构造分核输出参数
    SplitResult res{aicNum_, cvRatio_};
    Arch22::SplitCore(aicNum_, baseInfo, splitParam, res);
    if (res.numOfFdHead > aicNum_ || res.usedCoreNum > aicNum_ || res.maxS2SplitNum > aicNum_ + 1U) {
        OP_LOGE(fiaInfo_->opName, "used_core_num: %u, num_of_fd_head: %u, max_s2_split_num: %u, aic_num: %u",
                res.usedCoreNum, res.numOfFdHead, res.maxS2SplitNum, aicNum_);
    }
    SetSplitOutput(res);
    tilingData_.innerSplitParams.mBaseSize = mBaseSize_;
    tilingData_.innerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.fdParams.numOfFdHead = res.numOfFdHead;
    tilingData_.fdParams.gS1BaseSizeOfFd = mFdBaseSize_;
    usedCoreNum_ = res.usedCoreNum;

    // kvSplitPart_,用于lse out workspace计算
    if (IsFlashDecode()) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = res.maxS2SplitNum;
        tilingData_.fdParams.usedVecNumOfFd = res.usedVecNumOfFd;
    }
    CalcMmResSize();
}

void MlaNonQuantValueState::CalcMmResSize()
{
    int64_t mSize = std::min(fiaInfo_->gSize * fiaInfo_->s1Size, mBaseSize_);
    mSize = Align(mSize, static_cast<int64_t>(16U));
    mm1ResSize_ = static_cast<int64_t>(sInnerSizeAlign_) * mSize;
    mm2ResSize_ = static_cast<int64_t>(headDimAlign_) * mSize;
}

void MlaNonQuantValueState::CalcMaxMmResSize()
{
    constexpr int64_t S2_SIZE_MAX = 512;
    mm1ResSize_ = S2_SIZE_MAX * static_cast<int64_t>(mBaseSize_);
    mm2ResSize_ = static_cast<int64_t>(headDimAlign_) * static_cast<int64_t>(mBaseSize_);
}

void MlaNonQuantValueState::FillTilingWorkspaceParams()
{
    uint32_t accumOutElemNumOfOneFdTask = fiaInfo_->n2Size * mBaseSize_ * headDimAlign_;
    // max和sum总共两份
    uint32_t sumAndMaxElemNumOfOneFdTask = 2 * fiaInfo_->n2Size * mBaseSize_ * (BYTE_BLOCK / sizeof(float));
    // 每个核可能有头规约和尾规约，一共两份规约信息
    constexpr uint32_t MAX_FD_TASK_NUMS = 2;
    tilingData_.workspaceParams.fdAccumOutSize = aicNum_ * MAX_FD_TASK_NUMS * accumOutElemNumOfOneFdTask;
    tilingData_.workspaceParams.fdLogSumExpSize = aicNum_ * MAX_FD_TASK_NUMS * sumAndMaxElemNumOfOneFdTask;
    tilingData_.workspaceParams.mm1ResSize = mm1ResSize_;
    tilingData_.workspaceParams.mm2ResSize = mm2ResSize_;
}

uint32_t MlaNonQuantValueState::CalcFlashDecodeParamNums(const uint32_t coreNum) const
{
    return coreNum * 2U * fiaInfo_->n2Size * mBaseSize_; // 每个核可能有头规约和尾规约，一共两份规约信息
}

uint64_t MlaNonQuantValueState::CalcNormalWorkspaceSize(uint32_t coreNum, int64_t mm1ResSize, int64_t mm2ResSize,
                                                        uint32_t mBaseSize) const
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;     // 4: fp32
    constexpr uint32_t V1_RES_ELEM_SIZE = 2;      // 2: fp16/bf16
    constexpr uint32_t MM2_RES_ELEM_SIZE = 4;     // 4: fp32
    constexpr uint32_t V2_RES_ELEM_SIZE = 4;      // 4: fp32
    constexpr uint32_t N_UPDATE_ELEM_SIZE = 4;    // 4: int32
    constexpr uint32_t SOFTMAX_SUM_ELEM_SIZE = 4; // 4: int32

    uint64_t workspaceSize = 0;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm1ResSize * MM1_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm1ResSize * V1_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm2ResSize * MM2_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm2ResSize * V2_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mBaseSize * N_UPDATE_ELEM_SIZE;    // aMla nUpdate, mBaseSize=128
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mBaseSize * SOFTMAX_SUM_ELEM_SIZE; // aMla softmaxSum, mBaseSize=128
    return workspaceSize;
}

uint64_t MlaNonQuantValueState::CalcFlashDecodeWorkspace(const uint32_t coreNum) const
{
    uint64_t flashDecodeParamNums = static_cast<uint64_t>(CalcFlashDecodeParamNums(coreNum));
    uint64_t accumOutSize = flashDecodeParamNums * static_cast<uint64_t>(headDimAlign_);
    uint64_t logSumExpSize = 2 * flashDecodeParamNums * (BYTE_BLOCK / sizeof(float)); // log和sum的存储空间一致,需要2份
    uint64_t workspaceSize = (accumOutSize + logSumExpSize) * sizeof(float);
    return workspaceSize;
}

void MlaNonQuantValueState::CalcWorkspaceSize()
{
    workspaceSize_ = libapiSize_;
    workspaceSize_ += CalcNormalWorkspaceSize(coreNum_, mm1ResSize_, mm2ResSize_, mBaseSize_);
    if (splitKVFlag_) {
        workspaceSize_ += CalcFlashDecodeWorkspace(coreNum_);
    }
}

void MlaNonQuantValueState::CalcMaxWorkspaceSize()
{
    CalcMaxMmResSize();
    workspaceSize_ = libapiSize_;
    workspaceSize_ += CalcNormalWorkspaceSize(coreNum_, mm1ResSize_, mm2ResSize_, mBaseSize_);
    workspaceSize_ += CalcFlashDecodeWorkspace(aicNum_);
}
} // namespace Arch22
} // namespace FlashMla
