/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_mla_value_tiling_legacy.h"
#include "flash_mla_value_tiling_common.h"
#include "log/log.h"
#include "err/ops_err.h"
#include <algorithm>
#include <numeric>
#include <cstring>

namespace FlashMla {
namespace Arch22 {
using namespace optiling;
using namespace ge;
constexpr int64_t mFdBaseSizeMla = 8;
constexpr uint32_t MAX_SPLIT_RATIO = 2;
constexpr uint32_t MAX_AIC_CORE_NUM = kMlaMaxAicNum + 1; // Original tiling arrays include an aligned sentinel.
constexpr uint32_t SEQ_LEN_MIN = 4096;
constexpr uint32_t SEQ_LEN_MAX = 5120;
constexpr uint32_t SEQ_LEN_MIN_V2 = 37000;
constexpr uint32_t SEQ_LEN_MAX_V2 = 65536;
constexpr uint32_t HEAD_DIM = 512;
constexpr uint32_t HEAD_DIM_V = 512;
constexpr uint32_t SPARSE_MODE = 3;
constexpr uint32_t PRE_LOAD_NUM = 4;
constexpr uint32_t BUF_NUM = 6;
constexpr uint32_t PRE_LOAD_NUM_MLA = 2;
template <typename T>
inline T Align(T num, T rnd)
{
    return rnd == 0 ? 0 : ((num + rnd - 1) / rnd * rnd);
}
static int64_t CeilDivision(int64_t num1, int64_t num2)
{
    return num2 == 0 ? 0 : (num1 + num2 - 1) / num2;
}

// Prepare sequence views and the original tiler state, run the V4 scheduling
// methods below, then export dynamic fields. Host owns static fields and launch.
void MlaLegacyValueState::Prepare(const MlaScheduleConfig &c, const MlaPlatformInfo &p,
                                  optiling::IncreFlashAttentionContext &context)
{
    ifaContext_ = &context;
    numHeads_ = nNumOfQInOneGroup_ = groupSplitSize_ = c.numHeads;
    numKvHeads_ = 1;
    batchSizeQ_ = batchSize_ = c.batchSize;
    headDim_ = headDimV_ = headDimAlign_ = headDimVAlign_ = 512;
    headDimRope_ = 64;
    qSeqSize_ = s1SplitSize_ = static_cast<uint32_t>(c.qSeqSize);
    seqSize_ = sMax_ = static_cast<uint32_t>(c.kvSeqSize);
    inputLayout_ = static_cast<IfaLayout>(MlaTilingLayout(c.layoutQ));
    inputKvLayout_ = static_cast<IfaLayout>(MlaTilingLayout(c.layoutKv));
    pageAttentionFlag_ = c.blockSize != 0;
    blockSize_ = c.blockSize;
    maxBlockNumPerBatch_ = c.blockSize ? c.kvSeqSize / c.blockSize : 0;
    if (!pageAttentionFlag_) {
        kvListSeqLens_.push_back(c.kvSeqSize);
    }
    inputQType_ = inputKvType_ = static_cast<ge::DataType>(c.queryDataType);
    outputType_ = c.queryDataType == kMlaInt8 ? ge::DT_BF16 : inputQType_;
    quantFlag_ = c.queryDataType == kMlaInt8;
    ropeFlag_ = true;
    perfMode_ = IfaPerfMode::CUBE_VIEW_MM_MLA;
    blockTypeSize_ = sizeof(float);
    attenMaskFlag_ = c.hasAttenMask;
    sparseMode_ = c.sparseMode;
    aicNum_ = p.aicNum;
    aivNum_ = p.aivNum;
    cvRatio_ = p.aivNum / p.aicNum;
    // Preserve user workspace formulas; only Host WorkspaceCapacity adds W_sys.
    libapiSize_ = 0;
}

MlaStatus MlaLegacyValueState::Compute(const MlaScheduleConfig &config, const MlaPlatformInfo &platform,
                                       MlaSeqView seqQ, MlaSeqView seqKv, MlaValuePatch &patch)
{
    InitializeMlaPatch(config, platform, MlaFamily::LEGACY_MLA, patch);
    // Descriptor checks have already run in ComputeMlaValueTiling.
    auto status = ValidateMlaSeqValues(config, seqQ, seqKv);
    if (status != MlaStatus::SUCCESS) {
        return patch.header.status = status;
    }
    const bool ignoreQuerySeq = IsMlaQuerySeqIgnored(config.queryDataType, config.layoutQ == MlaLayout::TND);
    // Keep the real descriptor in the signature without constructing an unused legacy view.
    auto qTensor = MlaSeqTensor(ignoreQuerySeq ? MlaSeqView{} : seqQ);
    auto kvTensor = MlaSeqTensor(seqKv);
    gert::CompileTimeTensorDesc qDesc;
    qDesc.SetDataType(static_cast<ge::DataType>(config.queryDataType));
    const int64_t qStorage = static_cast<int64_t>(config.qSeqSize);
    const int64_t kvStorage = static_cast<int64_t>(config.kvSeqSize);
    gert::StorageShape qShape({qStorage}, {qStorage});
    gert::StorageShape kvShape({kvStorage}, {kvStorage});
    int64_t queryQuantMode = 3;
    int64_t kvQuantMode = 0;
    optiling::IncreFlashAttentionContext context{};
    context.opName = "FlashMlaMetadata";
    context.query.shape = &qShape;
    context.query.desc = &qDesc;
    context.key.shape = &kvShape;
    context.actualSeqLengthsQ.tensor = config.seqQPresent && !ignoreQuerySeq ? &qTensor : nullptr;
    context.actualSeqLengths.tensor = config.seqKvPresent ? &kvTensor : nullptr;
    context.queryQuantMode = &queryQuantMode;
    context.keyAntiquantMode = context.valueAntiquantMode = &kvQuantMode;
    Prepare(config, platform, context);
    if (config.layoutQ == MlaLayout::TND) {
        if (QKVPreProcess4TND("TND") != ge::GRAPH_SUCCESS) {
            return patch.header.status = MlaStatus::INVALID_SEQ;
        }
        s1SplitSize_ = qSeqSize_;
    }
    if (ProcessActualSeqLen() != ge::GRAPH_SUCCESS) {
        return patch.header.status = MlaStatus::INVALID_SEQ;
    }
    if (config.sparseMode == 9 && CheckTreeSparseMask() != ge::GRAPH_SUCCESS) {
        return patch.header.status = MlaStatus::INVALID_SEQ;
    }
    IsFdBalanceCase();
    balanceModeFlag_ = IsBalanceSplitCore();
    SetCoreNum();
    if (ZeroTensorProcess() != ge::GRAPH_SUCCESS || Split() != ge::GRAPH_SUCCESS) {
        return patch.header.status = MlaStatus::INVALID_SEQ;
    }
    tilingDataMla_.baseParams.seqSize = sMax_;
    tilingDataMla_.baseParams.qSeqSize = qSeqSize_;
    FillTilingSplitKVMla();
    FillTilingCoreParamsMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
    if (CalcWorkSpace() != ge::GRAPH_SUCCESS) {
        return patch.header.status = MlaStatus::CAPACITY_EXCEEDED;
    }
    static_assert(sizeof(tilingDataMla_) == kMlaLegacyPayloadBytes, "MLA value tiling layout");
    CopyMlaDynamicFields(MlaFamily::LEGACY_MLA, patch.payload, reinterpret_cast<const uint8_t *>(&tilingDataMla_));
    // The producer computes user workspace. Host tiling adds its own system reservation.
    patch.header.workspaceNeed = workspaceSize_;
    patch.header.originalBlockDim = usedCoreNum_;
    patch.header.dynamicFlags = (splitKVFlag_ ? kMlaFlagFd : 0) | (balanceModeFlag_ ? kMlaFlagBalance : 0);
    patch.header.maxQ = qSeqSize_;
    patch.header.maxKv = maxActualseq_;
    patch.header.qSeqSquareSum = qSeqSquareSum_;
    return patch.header.status = MlaStatus::SUCCESS;
}

MlaStatus MlaLegacyValueState::WorkspaceCapacity(const MlaScheduleConfig &config, const MlaPlatformInfo &platform,
                                                 uint64_t &capacity)
{
    // isWorkspace_ is the original descriptor-only upper-bound path. No sequence is synthesized.
    optiling::IncreFlashAttentionContext context{};
    Prepare(config, platform, context);
    isWorkspace_ = true;
    maxActualseq_ = sMax_;
    balanceModeFlag_ = IsBalanceSplitCore();
    SetCoreNum();
    if (ZeroTensorProcess() != ge::GRAPH_SUCCESS || Split() != ge::GRAPH_SUCCESS ||
        CalcWorkSpace() != ge::GRAPH_SUCCESS) {
        return MlaStatus::CAPACITY_EXCEEDED;
    }
    return MlaCheckedAdd(platform.sysWorkspaceSize, workspaceSize_, capacity) ? MlaStatus::SUCCESS :
                                                                                MlaStatus::OVERFLOW;
}

// Mechanically extracted from attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp.
// Private FlashMla value-state methods; the original operator is unchanged.

void MlaLegacyValueState::SetCoreNum()
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        coreNum_ = aicNum_; // use aic num in 310p
    } else {
        if (balanceModeFlag_) {
            coreNum_ = aicNum_;
            return;
        }
        if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD ||
            perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
            coreNum_ = aicNum_; // use aiv num when compute mm by vec
        } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
            coreNum_ = aicNum_;
        } else {
            coreNum_ = aivNum_; // split core by vec core
        }
    }
}

bool MlaLegacyValueState::CheckCommonConditions(const ValidityConfigFD &config) const
{
    return CheckBatchAndQSeqSize(config.validBatchSizes, config.validQSeqSizes) &&
           CheckHeadDimensions(config.numHeads, config.numKvHeads, config.headDim, config.headDimV) &&
           CheckQuantizationFlags(config.sparseMode) && CheckActualSeqLengths(config.expectedActualSeqLength);
}

bool MlaLegacyValueState::CheckBatchAndQSeqSize(const std::vector<int32_t> &validBatchSizes,
                                                const std::vector<int32_t> &validQSeqSizes) const
{
    if (std::find(validBatchSizes.begin(), validBatchSizes.end(), batchSize_) == validBatchSizes.end())
        return false;

    if (std::find(validQSeqSizes.begin(), validQSeqSizes.end(), qSeqSize_) == validQSeqSizes.end())
        return false;

    return true;
}

bool MlaLegacyValueState::CheckHeadDimensions(int32_t numHeads, int32_t numKvHeads, int32_t headDim,
                                              int32_t headDimV) const
{
    return (numHeads_ == numHeads && numKvHeads_ == numKvHeads && headDim_ == headDim && headDimV_ == headDimV);
}

bool MlaLegacyValueState::CheckQuantizationFlags(int32_t sparseMode) const
{
    if (!quantFlag_)
        return false;

    if (sparseMode_ != sparseMode)
        return false;

    if (inputKvLayout_ != IfaLayout::NZ || !pageAttentionFlag_)
        return false;

    int64_t queryQuantMode = ifaContext_->queryQuantMode ? *ifaContext_->queryQuantMode : 0;
    int64_t keyAntiQuantMode = ifaContext_->keyAntiquantMode ? *ifaContext_->keyAntiquantMode : 0;
    int64_t valueAntiQuantMode = ifaContext_->valueAntiquantMode ? *ifaContext_->valueAntiquantMode : 0;

    if (queryQuantMode != DEQUANT_PER_TOKEN_HEAD_MODE || keyAntiQuantMode != DEQUANT_PER_CHANNEL_MODE ||
        valueAntiQuantMode != DEQUANT_PER_CHANNEL_MODE)
        return false;

    return true;
}

bool MlaLegacyValueState::CheckActualSeqLengths(int64_t expectedActualSeqLength) const
{
    const int64_t *actualSeqKv = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
    if (actualSeqKv == nullptr)
        return false;

    for (uint32_t bIdx = 0U; bIdx < batchSize_; ++bIdx) {
        int64_t s2 = (actualLenDims_ == 1U) ? actualSeqKv[0] : actualSeqKv[bIdx];

        if (expectedActualSeqLength >= 0) {
            if (s2 < SEQ_LEN_MIN_V2 || s2 > SEQ_LEN_MAX_V2)
                return false;
        } else {
            if (s2 < SEQ_LEN_MIN || s2 > SEQ_LEN_MAX)
                return false;
        }
    }

    return true;
}

bool MlaLegacyValueState::IsValidFlag3B()
{
    ValidityConfigFD cfg{
        .validBatchSizes = {1, 2, 4}, // 表示有效的BatchSize大小
        .validQSeqSizes = {4},        // 表示有效的Qseq大小
        .numHeads = 8,
        .numKvHeads = 1,
        .headDim = HEAD_DIM,
        .headDimV = HEAD_DIM_V,
        .sparseMode = SPARSE_MODE,
        .expectedActualSeqLength = -1 // 表示 [4096, 5120]
    };
    return CheckCommonConditions(cfg);
}

bool MlaLegacyValueState::IsValidFlag560B()
{
    ValidityConfigFD cfg{.validBatchSizes = {4},
                         .validQSeqSizes = {2, 3, 4},
                         .numHeads = 64,
                         .numKvHeads = 1,
                         .headDim = HEAD_DIM,
                         .headDimV = HEAD_DIM_V,
                         .sparseMode = SPARSE_MODE,
                         .expectedActualSeqLength = -1};
    return CheckCommonConditions(cfg);
}

bool MlaLegacyValueState::IsValidFlag()
{
    ValidityConfigFD cfg{
        .validBatchSizes = {1},
        .validQSeqSizes = {2},
        .numHeads = 128,
        .numKvHeads = 1,
        .headDim = HEAD_DIM,
        .headDimV = HEAD_DIM_V,
        .sparseMode = SPARSE_MODE,
        .expectedActualSeqLength = 1 // 1表示整网场景下s2范围：[37000,65536]
    };
    return CheckCommonConditions(cfg);
}

void MlaLegacyValueState::IsFdBalanceCase()
{
    if (IsValidFlag() || IsValidFlag3B() || IsValidFlag560B()) {
        tilingDataMla_.tndSplitCoreParams.FdBalanceFlag = 1;
    } else {
        tilingDataMla_.tndSplitCoreParams.FdBalanceFlag = 0;
    }
}

bool MlaLegacyValueState::IsBalanceSplitCore()
{
    if (tilingDataMla_.tndSplitCoreParams.FdBalanceFlag == 1) {
        return true;
    }

    if (perfMode_ != IfaPerfMode::CUBE_VIEW_MM_MLA) {
        return false;
    }

    if (inputLayout_ == IfaLayout::TND) {
        return true;
    }

    if (antiQuantFlag_) {
        return false;
    }

    if (inputLayout_ == IfaLayout::BNSD) {
        return false;
    }

    if (isWorkspace_) { // tiling下沉不走balance
        return false;
    }

    return !DealSameSeqEachBatch();
}

ge::graphStatus MlaLegacyValueState::QKVPreProcess4TND(const std::string layout)
{
    auto qType = ifaContext_->query.desc->GetDataType();
    uint32_t qTypeSize = GetTypeSize(qType);
    if (qTypeSize == NUM_BYTES_UNDEF) {
        OP_LOGE(ifaContext_->opName, "Data type %d is not currently supported.", static_cast<int32_t>(qType));
        return ge::GRAPH_FAILED;
    }

    if (isWorkspace_) { // TND+tiling下沉场景，不做校验
        tSeqSize_ = qSeqSize_ = ifaContext_->query.shape->GetStorageShape().GetDim(0);
        OP_CHECK_IF((tSeqSize_ > 1024U * 1024U / qTypeSize), // T不大于1M
                    OP_LOGE(ifaContext_->opName, "%s query T should <= 1M", layout.c_str()), return ge::GRAPH_FAILED);
    } else {
        // actualSeqLengths非空校验
        OP_CHECK_IF(
            (ifaContext_->actualSeqLengthsQ.tensor == nullptr) ||
                (ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>() == nullptr),
            OP_LOGE(ifaContext_->opName, "%s the query's actual sequence lengths should not be null!", layout.c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((ifaContext_->actualSeqLengths.tensor == nullptr) ||
                        (ifaContext_->actualSeqLengths.tensor->GetData<int64_t>() == nullptr),
                    OP_LOGE(ifaContext_->opName, "%s the key/value's actual sequence lengths should not be null!",
                            layout.c_str()),
                    return ge::GRAPH_FAILED);

        actualLenQDims_ = ifaContext_->actualSeqLengthsQ.tensor->GetShapeSize();
        actualLenDims_ = ifaContext_->actualSeqLengths.tensor->GetShapeSize();

        OP_CHECK_IF((actualLenQDims_ <= 0U),
                    OP_LOGE(ifaContext_->opName, "%s the query's actual sequence lengths shape size(%u) should > 0",
                            layout.c_str(), actualLenQDims_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((actualLenQDims_ != actualLenDims_),
                    OP_LOGE(ifaContext_->opName,
                            "%s the query's actual sequence lengths shape size(%u) should equal the key/value's actual "
                            "sequence lengths shape size(%u)",
                            layout.c_str(), actualLenQDims_, actualLenDims_),
                    return ge::GRAPH_FAILED);

        batchSizeQ_ = batchSize_ = ifaContext_->actualSeqLengthsQ.tensor->GetShapeSize();
        tSeqSize_ = ifaContext_->query.shape->GetStorageShape().GetDim(0);
        uint32_t tKVSeqSize = ifaContext_->key.shape->GetStorageShape().GetDim(0);
        OP_CHECK_IF((tSeqSize_ > 1024U * 1024U / qTypeSize), // T不大于1M
                    OP_LOGE(ifaContext_->opName, "%s query T should <= 1M", layout.c_str()), return ge::GRAPH_FAILED);

        const int64_t *actualSeqQTnd = ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>();
        const int64_t *actualSeqKVTnd = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
        std::vector<int64_t> actualSeqQ(actualLenQDims_);
        int64_t tmpQSeqSize = 0;

        for (int32_t b = 0; b < static_cast<int>(actualLenQDims_); b++) {
            actualSeqQ[b] = (b <= 0) ? actualSeqQTnd[0] : (actualSeqQTnd[b] - actualSeqQTnd[b - 1]);
            OP_CHECK_IF((actualSeqQ[b] < 0) || (actualSeqQ[b] > 32), // 32 MTP最大QS
                        OP_LOGE(ifaContext_->opName,
                                "%s QS(%ld) of batch(%d) computed by the query's actual sequence lengths should be in "
                                "range [0, 32].",
                                layout.c_str(), actualSeqQ[b], b),
                        return ge::GRAPH_FAILED);
            tmpQSeqSize = std::max(tmpQSeqSize, actualSeqQ[b]);
            qSeqSquareSum_ += actualSeqQ[b] * actualSeqQ[b];
        }

        OP_CHECK_IF((tSeqSize_ != actualSeqQTnd[actualLenQDims_ - 1]),
                    OP_LOGE(ifaContext_->opName,
                            "%s T(%u) should be equal to the last element of the query's actual sequence lengths(%ld).",
                            layout.c_str(), tSeqSize_, actualSeqQTnd[actualLenQDims_ - 1]),
                    return ge::GRAPH_FAILED);

        if (!pageAttentionFlag_) {
            OP_CHECK_IF(
                (tKVSeqSize != actualSeqKVTnd[actualLenDims_ - 1]),
                OP_LOGE(ifaContext_->opName,
                        "%s T(%u) should be equal to the last element of the key/value's actual sequence lengths(%ld).",
                        layout.c_str(), tKVSeqSize, actualSeqKVTnd[actualLenDims_ - 1]),
                return ge::GRAPH_FAILED);
        }

        qSeqSize_ = static_cast<uint32_t>(tmpQSeqSize);
        OP_LOGI(ifaContext_->opName, "TND MAX actualSeqQ:%u", qSeqSize_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaLegacyValueState::ZeroTensorProcess()
{
    if (sMax_ == 0U) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        seqSize_ = 1024U;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaLegacyValueState::ProcessActualSeqLen()
{
    if (inputLayout_ == IfaLayout::TND) {
        if (isWorkspace_) {
            actualSeqLenFlag_ = true;
            maxActualseq_ = sMax_;
            return ge::GRAPH_SUCCESS;
        }

        if (CheckActualSeqLens() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        actualSeqLenFlag_ = true;
        maxActualseq_ = sMax_;
    } else {
        if (ifaContext_->actualSeqLengths.tensor == nullptr) {
            OP_LOGD(ifaContext_->opName, "the key/value's actual sequence lengths is nullptr");
            maxActualseq_ = sMax_;

            // pa场景必须带actual_seq_lens；第1次tiling调用时(isWorkspace为true)
            // actualSeqLengths会被强制置None，需要跳过校验
            OP_LOGD(ifaContext_->opName, "isWorkspace: %d", isWorkspace_);
            if (pageAttentionFlag_ && (!isWorkspace_)) {
                OP_LOGE(ifaContext_->opName,
                        "the key/value's actual sequence lengths is null, it must exist in pageAttention scene");
                return ge::GRAPH_FAILED;
            }
            return ge::GRAPH_SUCCESS;
        }
        OP_LOGD(ifaContext_->opName, "the key/value's actual sequence lengths is not nullptr");

        actualSeqLenFlag_ = true;
        actualLenDims_ = static_cast<uint32_t>(ifaContext_->actualSeqLengths.tensor->GetShapeSize());
        OP_LOGD(ifaContext_->opName, "number of elements in the key/value's actual sequence lengths is %u",
                actualLenDims_);
        if (actualLenDims_ == 0U) {
            // pa场景必须带actual_seq_lens
            if (pageAttentionFlag_) {
                OP_LOGW(ifaContext_->opName,
                        "the key/value's actual sequence lengths size[%u] can not be zero in pageAttention scene",
                        actualLenDims_);
            }
            maxActualseq_ = sMax_;
            return ge::GRAPH_SUCCESS;
        }

        OP_CHECK_IF(
            actualLenDims_ != 1U && actualLenDims_ < batchSize_,
            OP_LOGE(
                ifaContext_->opName,
                "the key/value's actual sequence lengths size[%u] should be greater than q batch[%u] or equal to 1.",
                actualLenDims_, batchSize_),
            return ge::GRAPH_FAILED);

        actualLenDims_ = std::min(actualLenDims_, batchSize_);
    }

    if (ParseActualSeqLens() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaLegacyValueState::CheckActualSeqLens()
{
    // Q的actual_seq要求非空
    if (ifaContext_->actualSeqLengthsQ.tensor == nullptr) {
        OP_LOGE(ifaContext_->opName, "TND the query's actual sequence lengths should not be null!");
        return ge::GRAPH_FAILED;
    }
    actualLenQDims_ = static_cast<uint32_t>(ifaContext_->actualSeqLengthsQ.tensor->GetShapeSize());
    if (actualLenQDims_ == 0U) {
        OP_LOGE(ifaContext_->opName, "TND actualLenQDims_ is 0!");
        return ge::GRAPH_FAILED;
    }

    // KV的actual_seq要求非空
    if (ifaContext_->actualSeqLengths.tensor == nullptr) {
        OP_LOGE(ifaContext_->opName, "TND the key/value's actual sequence lengths should not be null!");
        return ge::GRAPH_FAILED;
    }
    actualLenDims_ = ifaContext_->actualSeqLengths.tensor->GetShapeSize();
    if (actualLenDims_ == 0U) {
        OP_LOGE(ifaContext_->opName, "TND actualLenDims_ is 0!");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaLegacyValueState::ParseActualSeqLens()
{
    const int64_t *actualLenData = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
    if (actualLenData != nullptr) {
        OP_LOGD(ifaContext_->opName, "the data for the actual sequence lengths of key/value is not nullptr");
        uint32_t loop = ((actualLenDims_ == 1U) && (kvListSeqLens_.size() == 1U)) ? 1U : batchSize_;
        for (uint32_t i = 0U; i < loop; i++) {
            int64_t actLen = (actualLenDims_ == 1U) ? actualLenData[0] : actualLenData[i];
            OP_CHECK_IF(actLen < 0, // actualSeqLengths必须大于0
                        OP_LOGE(ifaContext_->opName,
                                "the value of the key/value's actual sequence lengths[%u] must be greater than or "
                                "equal to 0, but it is %ld",
                                i, actLen),
                        return ge::GRAPH_FAILED);
            OP_LOGI(ifaContext_->opName, "The value of the key/value's actual sequence lengths[%u] is %ld.", i, actLen);
            if (!pageAttentionFlag_) {
                uint32_t seqSize = (kvListSeqLens_.size() == 1) ? kvListSeqLens_[0] : kvListSeqLens_[i];
                OP_CHECK_IF(static_cast<uint32_t>(actLen) > seqSize,
                            OP_LOGE(ifaContext_->opName,
                                    "the key/value's actual sequence lengths[%u](%ld) cannot be greater than "
                                    "seq_length(%u) in input key.",
                                    i, actLen, seqSize),
                            return ge::GRAPH_FAILED);
            }
            maxActualseq_ =
                maxActualseq_ < static_cast<uint32_t>(actLen) ? static_cast<uint32_t>(actLen) : maxActualseq_;
            if (actLen == 0) {
                hasZeroActualseq_ = true;
            }
            // actLen already applies the one-element broadcast above. The raw
            // index is out of bounds for PA with B > 1 and actualLenDims_ == 1.
            if (actLen != actualLenData[0]) {
                isSameActualseq_ = false;
            }
        }
    } else {
        OP_LOGD(ifaContext_->opName, "data of the key/value's actual sequence lengths is nullptr");
        // pa场景必须带actual_seq_lens
        if (pageAttentionFlag_ && (!isWorkspace_)) {
            OP_LOGW(ifaContext_->opName,
                    "data of the key/value's actual sequence lengths can not be nullptr in pageAttention scene");
        }
        maxActualseq_ = sMax_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaLegacyValueState::Split()
{
    amlaMode_ = GetAmlaMode();
    if (balanceModeFlag_) {
        if (isWorkspace_) {
            CalcInnerSize(seqSize_);
            OP_LOGI(ifaContext_->opName, "aicNum_(use for usedCoreNum_ and kvSplitPart_):%u\n", aicNum_);
            usedCoreNum_ = aicNum_;
            splitKVFlag_ = true;
            kvSplit_++;
            kvSplitPart_ = aicNum_;
            return ge::GRAPH_SUCCESS;
        } else if (tilingDataMla_.tndSplitCoreParams.FdBalanceFlag == 1) {
            return SplitBalancedFd();
        } else {
            return SplitBalanced();
        }
    } else {
        return SplitUnbalanced();
    }
}

void MlaLegacyValueState::GetActualSeqInfo(const int64_t *actualSeqKv, ActualSeqInfo &actualSeqInfo) const
{
    uint32_t bSize = batchSize_;
    if (inputLayout_ == IfaLayout::TND) {
        // TND格式，actual_seq_q定义为累积长度，这里做转化再分核
        const int64_t *actualSeqQTnd = ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>();
        actualSeqInfo.actualSeqQ[0] = actualSeqQTnd[0];
        for (int32_t b = 1; b < static_cast<int>(bSize); b++) {
            actualSeqInfo.actualSeqQ[b] = actualSeqQTnd[b] - actualSeqQTnd[b - 1];
            if (actualLenDims_ != 1U) {
                actualSeqInfo.maxActualseqkv = std::max(actualSeqInfo.maxActualseqkv, actualSeqKv[b]);
            }
        }
    } else {
        for (int32_t b = 0; b < static_cast<int>(bSize); b++) {
            actualSeqInfo.actualSeqQ[b] = qSeqSize_; // 需要检查
            if (actualLenDims_ != 1U) {
                actualSeqInfo.maxActualseqkv = std::max(actualSeqInfo.maxActualseqkv, actualSeqKv[b]);
            }
        }
    }
}

void MlaLegacyValueState::GetSeqTilingInfo(const int64_t *actualSeqKv, const ActualSeqInfo &actualSeqInfo,
                                           SeqTilingInfo &seqTilingInfo) const
{
    uint32_t bSize = batchSize_;
    uint32_t souter = s1SplitSize_;
    uint32_t s2BasicSize = sInnerSize_;
    uint32_t n2Size = numKvHeads_;
    uint64_t totalS2Length = 0U;
    for (uint32_t bIdx = 0U; bIdx < bSize; bIdx++) {
        uint32_t s1 = actualSeqInfo.actualSeqQ[bIdx];
        uint64_t actualSeqLen = static_cast<uint64_t>(actualSeqKv[0]);
        if (actualLenDims_ != 1U) {
            actualSeqLen = static_cast<uint64_t>(actualSeqKv[bIdx]);
        }
        if (actualSeqInfo.maxActualseqkv <= 0) { // kv actseq全为0
            seqTilingInfo.s1OuterNum[bIdx] = 1U; // kv actseq全为0时，s1不切
            seqTilingInfo.s2OuterNum[bIdx] = 1U; // kv actseq全为0时，每个batch s2份数强制为1
        } else {
            seqTilingInfo.s1OuterNum[bIdx] = (s1 + (souter - 1U)) / souter; // s1总共切多少分，即线段个数
            seqTilingInfo.s2OuterNum[bIdx] =
                static_cast<uint32_t>((actualSeqLen + s2BasicSize - 1U) / s2BasicSize); // 线段长度
        }

        uint32_t bTotalS2Length = seqTilingInfo.s1OuterNum[bIdx] * seqTilingInfo.s2OuterNum[bIdx] *
                                  n2Size; // 线段个数*线段长度*N2=当前batch线段总长度

        totalS2Length += bTotalS2Length; // 更新一次总长度
        OP_LOGI(ifaContext_->opName, "s1OuterNum[%u]:%u, s2OuterNum[%u]:%u\n", bIdx, seqTilingInfo.s1OuterNum[bIdx],
                bIdx, seqTilingInfo.s2OuterNum[bIdx]);
        if ((seqTilingInfo.s1OuterNum[bIdx] > 0U) && (seqTilingInfo.s2OuterNum[bIdx] > 0U)) {
            seqTilingInfo.lastValidBIdx = bIdx;
        }
    }
    if (totalS2Length > coreNum_) {
        seqTilingInfo.avgS2Length = (totalS2Length + coreNum_ - 1U) / coreNum_; // 平均长度向上取整
    }
    OP_LOGI(ifaContext_->opName, "totalS2Length:%lu, avgS2Length:%lu, coreNum_:%u\n", totalS2Length,
            seqTilingInfo.avgS2Length, coreNum_);
}

void MlaLegacyValueState::FillBalancedSplitCoreInfo(const TilingIndexes &tilingIdx, BalancedSplitTilingInfo &tilingInfo)
{
    uint32_t *coreBEnd = tilingDataMla_.increFlashAttentionCoreParams.coreBEnd;
    uint32_t *coreS1OuterEnd = tilingDataMla_.increFlashAttentionCoreParams.coreS1OuterEnd;
    uint32_t *coreS2End = tilingDataMla_.increFlashAttentionCoreParams.coreS2End;

    coreBEnd[tilingInfo.currCoreIdx] = tilingIdx.bIdx;
    coreS1OuterEnd[tilingInfo.currCoreIdx] = tilingIdx.s1Idx;
    coreS2End[tilingInfo.currCoreIdx] = tilingIdx.s2Idx;
    tilingInfo.coreLoad[tilingInfo.currCoreIdx] = tilingInfo.accumS2Length;
    tilingInfo.currCoreIdx += 1U;
}

void MlaLegacyValueState::EndSplitForCurrentCore(const TilingIndexes &tilingIdx, const SeqTilingInfo &seqTilingInfo,
                                                 uint32_t &currKvSplitPart, BalancedSplitTilingInfo &tilingInfo)
{
    uint32_t *balanceFDCoreStartKVSplitNum = tilingDataMla_.tndSplitCoreParams.balanceFDCoreStartKVSplitNum;
    tilingInfo.accumS2Length += 1U;
    // 更新当前核的End分核信息
    FillBalancedSplitCoreInfo(tilingIdx, tilingInfo);
    if (tilingIdx.s2Idx <
        seqTilingInfo.s2OuterNum[tilingIdx.bIdx] - 1U) { // 只有切到S2的中间位置，才涉及规约，将currKvSplitPart加1
        currKvSplitPart += 1U;
        balanceFDCoreStartKVSplitNum[tilingInfo.currCoreIdx] = currKvSplitPart - 1U;
    } else {
        balanceFDCoreStartKVSplitNum[tilingInfo.currCoreIdx] = 0U;
    }
    tilingInfo.needUpdate = false;
}

void MlaLegacyValueState::SplitBalancedForEachHeadFd(uint32_t bIdx, const SeqTilingInfo &seqTilingInfo,
                                                     BalancedSplitTilingInfo &tilingInfo,
                                                     std::vector<int64_t> &gS1SplitNumOfFdHead, uint32_t s1)
{
    uint32_t *balanceFDCoreBArr = tilingDataMla_.tndSplitCoreParams.balanceFDCoreBArr;
    uint32_t *balanceFDCoreS1Arr = tilingDataMla_.tndSplitCoreParams.balanceFDCoreS1Arr;
    uint32_t *balanceFDCoreKVSplitArr = tilingDataMla_.tndSplitCoreParams.balanceFDCoreKVSplitArr;
    uint32_t *balanceFDCoreStartKVSplitNum = tilingDataMla_.tndSplitCoreParams.balanceFDCoreStartKVSplitNum;
    balanceFDCoreStartKVSplitNum[0] = 0U;
    uint32_t *gS1LastPartSizeOfFdHead = tilingDataMla_.tndSplitCoreParams.gS1LastPartSizeOfFdHead;

    for (uint32_t s1OuterIdx = 0U; s1OuterIdx < seqTilingInfo.s1OuterNum[bIdx]; s1OuterIdx++) {
        uint32_t currKvSplitPart = 1U; // [B,N2,S1]确定后，S2被切了几份
        for (uint32_t s2Idx = 0U; s2Idx < seqTilingInfo.s2OuterNum[bIdx]; s2Idx++) {
            // 计算当前的目标权重
            uint64_t targetS2Length = static_cast<uint64_t>(tilingInfo.currCoreIdx + 1U) * seqTilingInfo.avgS2Length;
            if (tilingInfo.accumS2Length + 1U >= targetS2Length && tilingInfo.currCoreIdx + 2U <= aicNum_) {
                EndSplitForCurrentCore(TilingIndexes(bIdx, s1OuterIdx, s2Idx), seqTilingInfo, currKvSplitPart,
                                       tilingInfo);
            } else {
                tilingInfo.accumS2Length += 1U;
                tilingInfo.needUpdate = true;
            }
        }
        tilingInfo.maxKvSplitPart = std::max(tilingInfo.maxKvSplitPart, currKvSplitPart);
        if (currKvSplitPart > 1U) {
            // S2被切过了，需要规约，记录[B,N,S1]三根轴的idx和切分份数，用于规约
            balanceFDCoreBArr[tilingInfo.tndFDCoreArrLen] = bIdx;
            balanceFDCoreS1Arr[tilingInfo.tndFDCoreArrLen] = s1OuterIdx;
            balanceFDCoreKVSplitArr[tilingInfo.tndFDCoreArrLen] = currKvSplitPart;
            int64_t gSize = nNumOfQInOneGroup_;
            int64_t currFdS1gSize = (s1OuterIdx == seqTilingInfo.s1OuterNum[bIdx] - 1) ?
                                        s1 * gSize - (seqTilingInfo.s1OuterNum[bIdx] - 1) * s1SplitSize_ * gSize :
                                        s1SplitSize_ * gSize; // 处理尾块
            int64_t currFdS1gSplitSize = (currFdS1gSize + mFdBaseSizeMla - 1) / mFdBaseSizeMla;
            int64_t currFdS1gLastPartSize = currFdS1gSize % mFdBaseSizeMla;
            if (currFdS1gLastPartSize == 0U) {
                currFdS1gLastPartSize = mFdBaseSizeMla;
            }
            gS1SplitNumOfFdHead[tilingInfo.tndFDCoreArrLen] = currFdS1gSplitSize;
            gS1LastPartSizeOfFdHead[tilingInfo.tndFDCoreArrLen] = currFdS1gLastPartSize;
            tilingInfo.tndFDCoreArrLen += 1U;
        }
    }
}

void MlaLegacyValueState::SplitBalancedForEachHead(uint32_t bIdx, const SeqTilingInfo &seqTilingInfo,
                                                   BalancedSplitTilingInfo &tilingInfo)
{
    uint32_t *balanceFDCoreBArr = tilingDataMla_.tndSplitCoreParams.balanceFDCoreBArr;
    uint32_t *balanceFDCoreS1Arr = tilingDataMla_.tndSplitCoreParams.balanceFDCoreS1Arr;
    uint32_t *balanceFDCoreKVSplitArr = tilingDataMla_.tndSplitCoreParams.balanceFDCoreKVSplitArr;
    uint32_t *balanceFDCoreStartKVSplitNum = tilingDataMla_.tndSplitCoreParams.balanceFDCoreStartKVSplitNum;
    balanceFDCoreStartKVSplitNum[0] = 0U;

    for (uint32_t s1OuterIdx = 0U; s1OuterIdx < seqTilingInfo.s1OuterNum[bIdx]; s1OuterIdx++) {
        uint32_t currKvSplitPart = 1U; // [B,N2,S1]确定后，S2被切了几份
        for (uint32_t s2Idx = 0U; s2Idx < seqTilingInfo.s2OuterNum[bIdx]; s2Idx++) {
            // 计算当前的目标权重
            uint64_t targetS2Length = static_cast<uint64_t>(tilingInfo.currCoreIdx + 1U) * seqTilingInfo.avgS2Length;
            if (tilingInfo.accumS2Length + 1U >= targetS2Length && tilingInfo.currCoreIdx + 2U <= aicNum_) {
                EndSplitForCurrentCore(TilingIndexes(bIdx, s1OuterIdx, s2Idx), seqTilingInfo, currKvSplitPart,
                                       tilingInfo);
            } else {
                tilingInfo.accumS2Length += 1U;
                tilingInfo.needUpdate = true;
            }
        }
        tilingInfo.maxKvSplitPart = std::max(tilingInfo.maxKvSplitPart, currKvSplitPart);
        if (currKvSplitPart > 1U) {
            // S2被切过了，需要规约，记录[B,N,S1]三根轴的idx和切分份数，用于规约
            balanceFDCoreBArr[tilingInfo.tndFDCoreArrLen] = bIdx;
            balanceFDCoreS1Arr[tilingInfo.tndFDCoreArrLen] = s1OuterIdx;
            balanceFDCoreKVSplitArr[tilingInfo.tndFDCoreArrLen] = currKvSplitPart;
            tilingInfo.tndFDCoreArrLen += 1U;
        }
    }
}

void MlaLegacyValueState::SplitFDMLa(uint32_t tndFDCoreArrLen, std::vector<int64_t> &gS1SplitNumOfFdHead,
                                     uint32_t *s2SplitNumOfFdHead, uint32_t aivCoreNum, SeqTilingInfo &seqTilingInfo)
{
    uint32_t *gS1IdxEndOfFdHead = tilingDataMla_.tndSplitCoreParams.gS1IdxEndOfFdHead;
    uint32_t *gS1IdxEndOfFdHeadSplit = tilingDataMla_.tndSplitCoreParams.gS1IdxEndOfFdHeadSplit;

    uint32_t totalFDLoad = 0U;
    uint32_t totalFDHeadSplit = 0U;

    for (uint32_t i = 0U; i < tndFDCoreArrLen; i++) {
        totalFDLoad += s2SplitNumOfFdHead[i] * gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += gS1SplitNumOfFdHead[i];
    }
    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    uint32_t maxVectorNum =
        std::min(totalFDHeadSplit,
                 aivCoreNum); // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    double loadThrOfVector =
        static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum); // 初始化vector的负载上限
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0U;
    uint32_t preTmpFDIndexEndOfFdHead = 0U;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0U;

    for (uint32_t i = 0U; i < tndFDCoreArrLen; i++) {
        uint32_t fDKVSplitNum = s2SplitNumOfFdHead[i];
        for (uint32_t gS1SplitIdx = 0U; gS1SplitIdx < gS1SplitNumOfFdHead[i]; gS1SplitIdx++) {
            double remainSpace = loadThrOfVector - loadOfCurVector; // 计算当前vector剩余负载空间
            // 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if (fDKVSplitNum > remainSpace * MAX_SPLIT_RATIO) {
                gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1U;
                totalFDLoad -= static_cast<uint32_t>(loadOfCurVector); // 当前未分配的总负载
                loadThrOfVector =
                    static_cast<double>(totalFDLoad) /
                    static_cast<double>(
                        maxVectorNum -
                        curCoreIndex); // 根据剩余负载和剩余可用vector更新负载上限，保证最后一个vector能分配所有负载
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    tilingDataMla_.tndSplitCoreParams.usedVecNumOfFd = curCoreIndex + 1U;
}

ge::graphStatus MlaLegacyValueState::SplitBalancedFd()
{
    CalcInnerSize(seqSize_);

    uint32_t s1gBasicSize = FIA_BALANCE_SG_BASIC_SIZE;
    uint32_t bSize = batchSize_;
    uint32_t gSize = nNumOfQInOneGroup_;
    uint32_t n2Size = numKvHeads_;
    s1SplitSize_ = s1gBasicSize / gSize; // QS的切分
    uint32_t souter = s1SplitSize_;

    // 负载均衡场景G轴不切
    groupSplitSize_ = nNumOfQInOneGroup_;
    gOuter_ = (nNumOfQInOneGroup_ + groupSplitSize_ - 1U) / groupSplitSize_;

    const int64_t *actualSeqKv = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
    ActualSeqInfo actualSeqInfo(bSize, actualSeqKv[0]);
    GetActualSeqInfo(actualSeqKv, actualSeqInfo);

    OP_LOGI(ifaContext_->opName, "bSize:%u, gSize:%u, n2Size:%u, souter:%u\n", bSize, gSize, n2Size, souter);
    // 分核主流程
    // 计算线段总长度和平均长度
    SeqTilingInfo seqTilingInfo(bSize);
    GetSeqTilingInfo(actualSeqKv, actualSeqInfo, seqTilingInfo);
    std::vector<int64_t> gS1SplitNumOfFdHead(MAX_AIC_CORE_NUM * 2);

    // 启动分核
    BalancedSplitTilingInfo tilingInfo(coreNum_);
    for (uint32_t bIdx = 0U; bIdx < bSize; bIdx++) {
        uint32_t s1 = actualSeqInfo.actualSeqQ[bIdx];
        int64_t s2 = actualLenDims_ == 1U ? actualSeqKv[0] : actualSeqKv[bIdx]; // 线段长度
        OP_LOGI(ifaContext_->opName, "bIdx:%u, s1:%u, s2:%ld\n", bIdx, s1, s2);
        for (uint32_t nIdx = 0U; nIdx < n2Size; nIdx++) {
            SplitBalancedForEachHeadFd(bIdx, seqTilingInfo, tilingInfo, gS1SplitNumOfFdHead, s1);
        }
    }
    uint32_t *gS1SplitNumOfFdHeadMla = tilingDataMla_.tndSplitCoreParams.gS1SplitNumOfFdHeadMla;
    for (uint32_t i = 0U; i < tilingInfo.tndFDCoreArrLen; ++i) {
        gS1SplitNumOfFdHeadMla[i] = gS1SplitNumOfFdHead[i];
    }
    if (tilingInfo.needUpdate) {
        // 更新最后一个核的End分核信息
        FillBalancedSplitCoreInfo(
            TilingIndexes(seqTilingInfo.lastValidBIdx, seqTilingInfo.s1OuterNum[seqTilingInfo.lastValidBIdx] - 1U,
                          seqTilingInfo.s2OuterNum[seqTilingInfo.lastValidBIdx] - 1U),
            tilingInfo);
    }
    uint32_t lastValidBMoreIdx = seqTilingInfo.lastValidBIdx + 1U;
    if (IsKvZeroBatchSplit(tilingInfo.needUpdate, lastValidBMoreIdx, bSize, seqTilingInfo.s1OuterNum,
                           seqTilingInfo.s2OuterNum)) {
        FillBalancedSplitCoreInfo(TilingIndexes(lastValidBMoreIdx, 0U, 0U), tilingInfo);
    }

    usedCoreNum_ = tilingInfo.currCoreIdx;
    tilingDataMla_.tndSplitCoreParams.tndFDCoreArrLen = tilingInfo.tndFDCoreArrLen;
    OP_LOGI(ifaContext_->opName, "usedCoreNum_:%u", usedCoreNum_);
    OP_LOGI(ifaContext_->opName, "tnd FD Core Array Length:%u", tilingInfo.tndFDCoreArrLen);
    OP_LOGI(ifaContext_->opName, "max Kv Split Part:%u", tilingInfo.maxKvSplitPart);
    OP_LOGI(ifaContext_->opName, "average S2 Length:%lu", seqTilingInfo.avgS2Length);
    OP_LOGI(ifaContext_->opName, "sInnerSize_:%u", sInnerSize_);

    if (IsFlashDecode(coreNum_, perfMode_)) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = tilingInfo.maxKvSplitPart;
        // 使能fd负载均衡
        uint32_t *balanceFDCoreS1Arr = tilingDataMla_.tndSplitCoreParams.balanceFDCoreS1Arr;
        uint32_t *s2SplitNumOfFdHead = tilingDataMla_.tndSplitCoreParams.balanceFDCoreKVSplitArr;
        uint32_t aivCoreNumOfFd = (aivNum_ / aicNum_) * usedCoreNum_;
        SplitFDMLa(tilingInfo.tndFDCoreArrLen, gS1SplitNumOfFdHead, s2SplitNumOfFdHead, aivCoreNumOfFd,
                   seqTilingInfo); // 待补充
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaLegacyValueState::SplitBalanced()
{
    CalcInnerSize(seqSize_);

    uint32_t s1gBasicSize = FIA_BALANCE_SG_BASIC_SIZE;
    uint32_t bSize = batchSize_;
    uint32_t gSize = nNumOfQInOneGroup_;
    uint32_t n2Size = numKvHeads_;
    s1SplitSize_ = s1gBasicSize / gSize; // QS的切分
    uint32_t souter = s1SplitSize_;

    // 负载均衡场景G轴不切
    groupSplitSize_ = nNumOfQInOneGroup_;
    gOuter_ = (nNumOfQInOneGroup_ + groupSplitSize_ - 1U) / groupSplitSize_;

    const int64_t *actualSeqKv = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
    ActualSeqInfo actualSeqInfo(bSize, actualSeqKv[0]);
    GetActualSeqInfo(actualSeqKv, actualSeqInfo);

    OP_LOGI(ifaContext_->opName, "bSize:%u, gSize:%u, n2Size:%u, souter:%u\n", bSize, gSize, n2Size, souter);
    // 分核主流程
    // 计算线段总长度和平均长度
    SeqTilingInfo seqTilingInfo(bSize);
    GetSeqTilingInfo(actualSeqKv, actualSeqInfo, seqTilingInfo);

    // 启动分核
    BalancedSplitTilingInfo tilingInfo(coreNum_);
    for (uint32_t bIdx = 0U; bIdx < bSize; bIdx++) {
        uint32_t s1 = actualSeqInfo.actualSeqQ[bIdx];
        int64_t s2 = actualLenDims_ == 1U ? actualSeqKv[0] : actualSeqKv[bIdx]; // 线段长度
        OP_LOGI(ifaContext_->opName, "bIdx:%u, s1:%u, s2:%ld\n", bIdx, s1, s2);
        for (uint32_t nIdx = 0U; nIdx < n2Size; nIdx++) {
            SplitBalancedForEachHead(bIdx, seqTilingInfo, tilingInfo);
        }
    }
    if (tilingInfo.needUpdate) {
        // 更新最后一个核的End分核信息
        FillBalancedSplitCoreInfo(
            TilingIndexes(seqTilingInfo.lastValidBIdx, seqTilingInfo.s1OuterNum[seqTilingInfo.lastValidBIdx] - 1U,
                          seqTilingInfo.s2OuterNum[seqTilingInfo.lastValidBIdx] - 1U),
            tilingInfo);
    }
    uint32_t lastValidBMoreIdx = seqTilingInfo.lastValidBIdx + 1U;
    if (IsKvZeroBatchSplit(tilingInfo.needUpdate, lastValidBMoreIdx, bSize, seqTilingInfo.s1OuterNum,
                           seqTilingInfo.s2OuterNum)) {
        FillBalancedSplitCoreInfo(TilingIndexes(lastValidBMoreIdx, 0U, 0U), tilingInfo);
    }

    if (inputLayout_ == IfaLayout::TND && tSeqSize_ != 0U && tilingInfo.currCoreIdx == 0U) {
        // No Q/KV work was assigned, but the nonempty TND output still needs zeroing.
        // Core 0 visits every batch through the kernel's existing zero-length path.
        FillBalancedSplitCoreInfo(TilingIndexes(bSize - 1U, 0U, 0U), tilingInfo);
    }

    usedCoreNum_ = tilingInfo.currCoreIdx;
    tilingDataMla_.tndSplitCoreParams.tndFDCoreArrLen = tilingInfo.tndFDCoreArrLen;
    OP_LOGI(ifaContext_->opName, "usedCoreNum_:%u", usedCoreNum_);
    OP_LOGI(ifaContext_->opName, "tnd FD Core Array Length:%u", tilingInfo.tndFDCoreArrLen);
    OP_LOGI(ifaContext_->opName, "max Kv Split Part:%u", tilingInfo.maxKvSplitPart);
    OP_LOGI(ifaContext_->opName, "average S2 Length:%lu", seqTilingInfo.avgS2Length);
    OP_LOGI(ifaContext_->opName, "sInnerSize_:%u", sInnerSize_);

    if (IsFlashDecode(coreNum_, perfMode_)) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = tilingInfo.maxKvSplitPart;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaLegacyValueState::SplitUnbalanced()
{
    if (amlaMode_ == IfaAmlaMode::AMLA_3BUF) {
        gMax_ = 512U; // 3buf场景下切512
    }
    if (gqaMtpFlag_) {
        gMax_ = antiQuantFlag_ ? 128U : 256U; // 伪量化场景下，g切32，非伪量化场景切64
    }

    if (ropeFlag_ || gqaMtpFlag_) {
        if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
            s1SplitSize_ = gMax_ / nNumOfQInOneGroup_;
            if (s1SplitSize_ > qSeqSize_) {
                s1SplitSize_ = qSeqSize_;
            }
            s1Outer_ = (qSeqSize_ + s1SplitSize_ - 1U) / s1SplitSize_;
        } else {
            // mla 模式支持G切分
            groupSplitSize_ = gMax_ / qSeqSize_;
            groupSplitSize_ -= (groupSplitSize_ & 1U);
            if (groupSplitSize_ > nNumOfQInOneGroup_) {
                groupSplitSize_ = nNumOfQInOneGroup_;
            }
            gOuter_ = (nNumOfQInOneGroup_ + groupSplitSize_ - 1U) / groupSplitSize_;
        }
    }

    if (IsFlashDecode(coreNum_, perfMode_)) {
        OP_LOGI(ifaContext_->opName, "Enable flashdecode.");
        splitKVFlag_ = true;
        kvSplit_++;
        return SplitBNS();
    }

    CalcInnerSize(seqSize_);
    return SplitBN();
}

ge::graphStatus MlaLegacyValueState::SplitBN()
{
    uint32_t bn;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
        bn = batchSize_ * numKvHeads_ * s1Outer_;
    } else {
        bn = batchSize_ * numKvHeads_ * gOuter_;
    }

    for (auto &elem : startIdxEachCore_) {
        elem = bn;
    }

    if (isSysPrefixTiling_) {
        return SplitBN_V0();
    }

    if (actualLenDims_ == 1U || bn <= coreNum_ || (actualLenDims_ == 0U && kvListSeqLens_.size() == 1U) ||
        DealSameSeqEachBatch()) {
        return SplitBN_V0();
    }

    std::vector<int64_t> validArray;
    if (actualLenDims_ > 0U) {
        const int64_t *actualLenData = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
        validArray = InitSparseValidArray(actualLenData);
    } else {
        validArray = InitSparseValidArray(&kvListSeqLens_[0]);
    }

    SetSparseStartIdx(validArray, bn, coreNum_, startIdxEachCore_, CeilDivision(bn, coreNum_));

    usedCoreNum_ = coreNum_;
    return ge::GRAPH_SUCCESS;
}

void MlaLegacyValueState::GetEstimatedLoad(int64_t &estimatedLoad) const
{
    if (antiQuantFlag_ && estimatedLoad < MSD_VEC_LOAD) {
        estimatedLoad = MSD_VEC_LOAD;
    } else if (estimatedLoad == 0) {
        // 空tensor输出也计入负载，否则当最后一个batch为空tensor时，分核算法会将该batch优化掉
        estimatedLoad = 1;
    }
}

std::vector<int64_t> MlaLegacyValueState::InitSparseValidArray(const int64_t *actualLens) const
{
    uint32_t outer;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
        outer = s1Outer_;
    } else {
        outer = gOuter_;
    }
    std::vector<int64_t> res((batchSize_ * numKvHeads_ * outer));
    for (uint32_t b = 0U; b < batchSize_; b++) {
        for (uint32_t n = 0U; n < numKvHeads_ * outer; n++) {
            int64_t estimatedLoad = static_cast<int64_t>(seqSize_);
            if (actualLens != nullptr) {
                estimatedLoad = actualLens[b];
                GetEstimatedLoad(estimatedLoad);
            }
            res[b * numKvHeads_ * outer + n] = estimatedLoad;
        }
    }
    return res;
}

bool MlaLegacyValueState::BalanceLoad(const std::vector<int64_t> &sparseValidArray, int64_t totalSize,
                                      int64_t validAivNum, std::vector<int64_t> &localValue,
                                      std::vector<int64_t> &sparseStartIdx) const
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
    int64_t tmpMaxVal = maxVal;

    // 从前往后遍历
    for (int64_t idx = 1; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
            localValue[idx - 1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        } else if (start == totalSize) {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    // 从后往前遍历
    for (int64_t idx = validAivNum - 1; idx > 0; --idx) {
        int64_t start = sparseStartIdx[idx];
        if (start == totalSize) {
            if (sparseStartIdx[idx - 1] == totalSize) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] = sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else if (start > 0) {
            if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] += sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    return (tmpMaxVal >= maxVal) ? false : true;
}

void MlaLegacyValueState::InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t totalSize,
                                        int64_t validAivNum, const std::vector<int64_t> &sparseStartIdx,
                                        std::vector<int64_t> &localValue) const
{
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
        if (start < totalSize) {
            localValue[idx] = std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0);
        } else {
            break;
        }
    }
}

void MlaLegacyValueState::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t totalSize,
                                            int64_t validAivNum, uint32_t *sparseStartIdx,
                                            int64_t splitFactorSize) const
{
    // initLoad: 使用均分策略, 保证后续不会比均分差
    std::vector<int64_t> localSparseStartIdx(MAX_CORE_NUM, totalSize);
    for (uint32_t idx = 0; idx < MAX_CORE_NUM; ++idx) {
        localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
    }
    std::vector<int64_t> localValue(validAivNum, 0);
    InitLoadValue(sparseValidArray, totalSize, validAivNum, localSparseStartIdx, localValue);

    // 负载均衡粗调
    std::vector<int64_t> tmpLocalValue(validAivNum, 0);
    std::vector<int64_t> tmpsparseStartIdx(MAX_CORE_NUM, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0);
    int64_t avgVal = CeilDivision(sparseArraySum, validAivNum);

    tmpsparseStartIdx[0] = 0;
    for (uint32_t idx = 1; idx < MAX_CORE_NUM; ++idx) {
        int64_t start = tmpsparseStartIdx[idx - 1];
        int64_t singleLoadValue = 0;
        tmpsparseStartIdx[idx] = start;
        while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
            singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
            tmpsparseStartIdx[idx] += 1;
        }

        if ((start + 1) < tmpsparseStartIdx[idx]) {
            int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
            if ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) {
                tmpsparseStartIdx[idx] = tmpsparseStartIdx[idx] - 1;
                singleLoadValue = redoSingleLoadValue;
            }
            sparseArraySum -= singleLoadValue;
            avgVal = CeilDivision(sparseArraySum, (validAivNum - static_cast<int64_t>(idx)));
        }
    }

    InitLoadValue(sparseValidArray, totalSize, validAivNum, tmpsparseStartIdx, tmpLocalValue);

    // 负载均衡精调
    while (BalanceLoad(sparseValidArray, totalSize, validAivNum, tmpLocalValue, tmpsparseStartIdx)) {
        // 根据负载均衡是否能得到更好预测结果决定是否结束循环
    }

    // exchange initLoad and 负载均衡
    if ((*std::max_element(localValue.begin(), localValue.end())) >
        (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
        localSparseStartIdx.swap(tmpsparseStartIdx);
        localValue.swap(tmpLocalValue);
    }
    for (uint32_t idx = 0; idx < MAX_CORE_NUM; ++idx) {
        sparseStartIdx[idx] = localSparseStartIdx[idx];
    }
}

ge::graphStatus MlaLegacyValueState::SplitBN_V0()
{
    uint32_t bn;
    if (inputLayout_ == IfaLayout::BSH_BSND || (gqaKvNZFlag_ && inputLayout_ == IfaLayout::TND)) {
        bn = batchSize_ * numKvHeads_ * s1Outer_;
    } else {
        bn = batchSize_ * numKvHeads_ * gOuter_;
    }
    // MLA场景，在单核的最大负载不变的情况下，尽可能少的启动核
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        coreNum_ = static_cast<uint32_t>(CeilDivision(
            static_cast<int64_t>(bn), CeilDivision(static_cast<int64_t>(bn), static_cast<int64_t>(aicNum_))));
    }
    formerCoreNum_ = bn % coreNum_;
    if (formerCoreNum_ == 0U) {
        blockSplitBn2Range_ = bn / coreNum_;
        tailSplitedBatchRange_ = blockSplitBn2Range_;
    } else {
        blockSplitBn2Range_ = bn / coreNum_ + 1U;
        tailSplitedBatchRange_ = blockSplitBn2Range_ - 1U;
    }

    usedCoreNum_ = bn > coreNum_ ? coreNum_ : bn;

    for (uint32_t i = 0U; i < formerCoreNum_; i++) {
        startIdxEachCore_[i] = blockSplitBn2Range_ * i;
    }

    uint32_t formerBase = formerCoreNum_ * blockSplitBn2Range_;
    for (uint32_t i = formerCoreNum_; i < usedCoreNum_; i++) {
        startIdxEachCore_[i] = formerBase + tailSplitedBatchRange_ * (i - formerCoreNum_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaLegacyValueState::SplitBNS()
{
    formerCoreNum_ = 0U;
    blockSplitBn2Range_ = 1U;
    tailSplitedBatchRange_ = 1U;

    uint32_t bn = batchSize_ * numKvHeads_;
    kvSplitPart_ = coreNum_ / bn;
    while (((maxActualseq_ / kvSplitPart_) < 512U) && (kvSplitPart_ > 1U)) { // 512, 经验值
        kvSplitPart_--;
    }

    usedCoreNum_ = bn * kvSplitPart_;
    uint32_t computeSeqSize = (seqSize_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    if (pageAttentionFlag_) {
        computeSeqSize = Align(computeSeqSize, blockSize_);
    }
    if (inputKvType_ == ge::DT_INT4) {
        computeSeqSize = Align(computeSeqSize, 2U);
    }
    CalcInnerSize(computeSeqSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MlaLegacyValueState::CalcInnerSize(uint32_t seqSize)
{
    /**
     * sInnerSize：s2的切分大小，直接决定了MM的singleN/K和vector的切块大小，但当前切分也并非适用所有case。
     * 1、非GQA场景：按照vector的最大基本块8*1024进行切分，sInnerSize=8192
     * 2、GQA场景：(1) 非伪量化：vector计算比较少，cube MTE2bound，
     *                          因此，cube发射大块，减少通信次数。sInnerSize=8192
     *            (2) 伪量化：vector比较重，尽量较少vector的循环次数,
     *                          因此，cube发小块，期望vector尽量被cube的mte2掩盖。sInnerSize=1024
     */
    sInnerSize_ = MAX_SPLIT_SIZE; // 8192
    if (antiQuantFlag_) {
        if (nNumOfQInOneGroup_ > 1U) {
            if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
                sInnerSize_ = 2048U;
            } else {
                sInnerSize_ = 1024U;
            }
        } else if (nNumOfQInOneGroup_ == 1) {
            if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
                sInnerSize_ = 1024U;
            }
        }
    } else {
        /** 当前版本限制workspace大小不超过32MB，否则会影响网络中前后算子性能，
         *  GQA场景下 nNumOfQInOneGroup_和sInnerSize_切分大小直接影响workspace大小,
         *  具体计算参考CalcWorkSpace函数，这里根据nNumOfQInOneGroup_将sInnerSize_
         *  分为8192，4096，2048三档，nNumOfQInOneGroup_增大时减小sInnerSize_，
         *  保证最终workspace大小不超过32MB。
         */
        uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
        uint32_t idx = std::min(nNumOfQInOneGroup_ / 5U, 2U);
        sInnerSize_ = sInnerSize[idx];
    }
    // for ifa_cube_310P
    if ((socVersion_ == IfaSocVersion::SOC_ASCEND_310P) && (nNumOfQInOneGroup_ * qSeqSize_ > 1U)) {
        auto bmm1BufferSize = 40U * 1024U;
        sInnerSize_ = std::min(bmm1BufferSize / nNumOfQInOneGroup_ / qSeqSize_ / static_cast<uint32_t>(sizeof(float)),
                               MAX_SPLIT_SIZE);
        sInnerSize_ = (sInnerSize_ / 128U) * 128U;
    }

    if (ropeFlag_) {
        sInnerSize_ = 512U;
        // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
        if (!isWorkspace_ && splitKVFlag_ && inputLayout_ != IfaLayout::TND) {
            if (seqSize == 256U) {
                sInnerSize_ = 128U;
            } else if (seqSize > 256U && seqSize <= sInnerSize_) {
                sInnerSize_ = (sInnerSize_ + 1U) / 2U;
            }
        }
        if (antiQuantFlag_) {
            sInnerSize_ = 1024U; // 1024: 伪量化场景下，sInnerSize_设置为1024
        }
    }

    // PA特性泛化场景，blockSize_可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
    if (pageAttentionFlag_ && blockSize_ != 0U) {
        if (sInnerSize_ % blockSize_ != 0U) {
            sInnerSize_ = (sInnerSize_ / blockSize_) * blockSize_;
        }
    }

    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
    // tiling下沉 && flash decoder场景时，sInnerSize_基块大小不按照真实值修改
    // 否则会导致 tiling下沉 && flash decoder 场景时开辟workspace空间大小小于真实运行时所需的workspace大小
    if (sInnerSize_ > seqSize && (!(isWorkspace_ && splitKVFlag_))) {
        sInnerSize_ = seqSize;
    }
    if (inputKvType_ == ge::DT_INT4) {
        sInnerSize_ = Align(sInnerSize_, 2U);
        sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
        sInnerSizeAlign_ = Align(sInnerSize_, 64U); // 元素个数按照基本块大小对齐
    } else {
        sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    }

    CheckUbSpace();
    return ge::GRAPH_SUCCESS;
}

void MlaLegacyValueState::SetDealBN2Num()
{
    uint32_t elemNum = Align(maxActualseq_, 32U) * ((nNumOfQInOneGroup_ + 1U) / 2U);
    dealBN2Num_ = (8192U + elemNum - 1U) / elemNum;
    if (dealBN2Num_ < DEAL_BN2_NUM) {
        dealBN2Num_ = DEAL_BN2_NUM;
    }
}

bool MlaLegacyValueState::CalcUbBmm()
{
    uint32_t cubeMSize = nNumOfQInOneGroup_ * qSeqSize_;
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        uint32_t maxMSize = (inputLayout_ == IfaLayout::TND) ? FIA_BALANCE_SG_BASIC_SIZE : gMax_;
        if (cubeMSize > maxMSize) {
            cubeMSize = maxMSize;
        }
        if (amlaMode_ != IfaAmlaMode::DISABLE_AMLA) {
            cubeMSize = Align(cubeMSize, 16U);
        }
    }
    if (ropeFlag_) {
        mmResUbSize_ = static_cast<size_t>(sInnerSizeAlign_) * static_cast<size_t>(cubeMSize);
        bmm2ResUbSize_ = static_cast<size_t>(headDimAlign_) * static_cast<size_t>(cubeMSize);
    } else {
        mmResUbSize_ =
            static_cast<size_t>(msdIterNum_) * static_cast<size_t>(sInnerSizeAlign_) * static_cast<size_t>(cubeMSize);
        if (slidingFlag_) {
            bmm2ResUbSize_ =
                static_cast<size_t>(msdIterNum_) * static_cast<size_t>(headDimVAlign_) * static_cast<size_t>(cubeMSize);
        } else {
            bmm2ResUbSize_ =
                static_cast<size_t>(msdIterNum_) * static_cast<size_t>(headDimAlign_) * static_cast<size_t>(cubeMSize);
        }
    }

    qPreSizeMla_ = static_cast<size_t>(msdIterNum_) * static_cast<size_t>(nNumOfQInOneGroup_ * (headDimAlign_ + 64U)) *
                   static_cast<size_t>(qSeqSize_);
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
        SetDealBN2Num();
        mmResUbSize_ *= dealBN2Num_;
        bmm2ResUbSize_ *= dealBN2Num_;
    }

    if (isSysPrefixTiling_) {
        mmResUbSize_ *= batchSizeQ_;
        bmm2ResUbSize_ *= batchSizeQ_;
    }
    return true;
}

std::pair<uint32_t, uint32_t> MlaLegacyValueState::GetPreLoadNumAndActCoreNum() const
{
    uint32_t preLoadNum = 1U;
    uint32_t actCoreNum = coreNum_;
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM || perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
        preLoadNum = PRE_LOAD_NUM;
    } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_DD) {
        preLoadNum = BUF_NUM;
    } else if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        preLoadNum = PRE_LOAD_NUM_MLA;
        if (isWorkspace_) {
            actCoreNum = aicNum_;
        }
    }
    return std::make_pair(preLoadNum, actCoreNum);
}

void MlaLegacyValueState::CalcWorkSpaceForBmmAll(const IfaWorkSpaceSizeParams &params, uint32_t preLoadNum,
                                                 uint32_t actCoreNum)
{
    workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * params.mmResElemSize);
    if (ropeFlag_) {
        workspaceSize_ +=
            preLoadNum *
            static_cast<size_t>(static_cast<float>(mmResUbSize_ * msdIterNum_ * actCoreNum * params.vec1ResElemSize) *
                                params.kvDtypeRatio);
    } else {
        workspaceSize_ +=
            preLoadNum * static_cast<size_t>(static_cast<float>(mmResUbSize_ * actCoreNum * params.vec1ResElemSize) *
                                             params.kvDtypeRatio);
    }
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * params.bmm2ResElemSize;
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * params.vec2ResElemSize;
    if (ropeFlag_) {
        workspaceSize_ += preLoadNum * static_cast<size_t>(
                                           static_cast<float>(qPreSizeMla_ * actCoreNum * params.qPreProcResElemSize) *
                                           params.kvDtypeRatio);
    } else {
        workspaceSize_ +=
            preLoadNum *
            static_cast<size_t>(static_cast<float>(bmm2ResUbSize_ * actCoreNum * params.qPreProcResElemSize) *
                                params.kvDtypeRatio);
    }
    workspaceSize_ +=
        static_cast<size_t>(preLoadNum) * gMax_ * actCoreNum * params.nUpdateElemSize; // aMla nUpdate, Gmax=128
    workspaceSize_ +=
        static_cast<size_t>(preLoadNum) * gMax_ * actCoreNum * params.softmaxSumElemSize; // aMla softmaxSum, Gmax=128
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD) {
        uint32_t tmpWsNum = 1U; // 1块workspace: aMaxBmm1Gm
        if (antiquantMode_ == PER_TOKEN_MODE) {
            tmpWsNum = 4U; // 4块workspace: aMaxBmm1Gm/qRowSumGm/softmaxResAmaxGm/softmaxResRowSumGm
        }
        workspaceSize_ += preLoadNum *
                          static_cast<size_t>(static_cast<float>(actCoreNum * dealBN2Num_ * nNumOfQInOneGroup_ * 32U)) *
                          tmpWsNum;
        workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(
                                           actCoreNum * dealBN2Num_ * nNumOfQInOneGroup_ * 32U)); // SoftmaxSumGm
    }
}

ge::graphStatus MlaLegacyValueState::CalcWorkSpace()
{
    IfaWorkSpaceSizeParams params{};
    if (antiQuantFlag_) {
        if (ropeFlag_) {
            params.mmResElemSize = 2U;   // 2: S已经被转换成FP16存在workspace上
            params.bmm2ResElemSize = 2U; // 2: O已经被转换成FP16在workspace上
            params.vec2ResElemSize = 2U;
        } else {
            params.mmResElemSize = 4U;   // 4:int32
            params.bmm2ResElemSize = 4U; // 4:int32
            params.vec2ResElemSize = 4U; // 4:float
        }
        params.vec1ResElemSize = 1U;                                   // 1:int8
        params.qPreProcResElemSize = 1U;                               // int8
        params.kvDtypeRatio = inputKvType_ == ge::DT_INT4 ? 0.5 : 1.0; // 0.5:int4 1.0:elseType
    }

    workspaceSize_ = libapiSize_;
    std::pair<uint32_t, uint32_t> preLoadInfo = GetPreLoadNumAndActCoreNum();
    uint32_t preLoadNum = preLoadInfo.first;
    uint32_t actCoreNum = preLoadInfo.second;
    if (perfMode_ != IfaPerfMode::BMM_ALL_BY_VEC) {
        CalcWorkSpaceForBmmAll(params, preLoadNum, actCoreNum);
    }
    CalcFDWorkSpace(actCoreNum);

    uint32_t mmPACallBackDataSize = 64; // 64: matmul回调信息需要7个uint32值，dcci cacheline需要64B对齐
    if (pageAttentionFlag_) {
        workspaceSize_ += static_cast<size_t>(actCoreNum) * static_cast<size_t>(mmPACallBackDataSize) *
                          static_cast<size_t>(2); // bmm1 bmm2 2份
    }

    if (isSysPrefixTiling_) {
        if (antiQuantFlag_) {
            size_t blockSize = static_cast<size_t>(nNumOfQInOneGroup_) * static_cast<size_t>(BYTE_BLOCK) *
                               static_cast<size_t>(batchSizeQ_);
            workspaceSize_ += actCoreNum * blockSize * 4U; // 4, rowMax1 rowMax2 rowSum1 rowSum2
        }

        size_t blockSize = static_cast<size_t>(nNumOfQInOneGroup_) * static_cast<size_t>(BYTE_BLOCK) *
                           static_cast<size_t>(batchSizeQ_);
        workspaceSize_ += actCoreNum * blockSize * 3U; // 3, sum log exp

        if (!antiQuantFlag_) {
            workspaceSize_ += static_cast<size_t>(batchSizeQ_) * static_cast<size_t>(nNumOfQInOneGroup_) *
                              static_cast<size_t>(headDimAlign_) * static_cast<size_t>(2) *
                              static_cast<size_t>(actCoreNum); // 2:fp16/bf16
        }
    }

    if (ifaContext_->workSpaces) {
        ifaContext_->workSpaces[0] = workspaceSize_;
    }
    return ge::GRAPH_SUCCESS;
}

void MlaLegacyValueState::CalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (perfMode_ == IfaPerfMode::CUBE_VIEW_MM_MLA) {
        MLACalcFDWorkSpace(actCoreNum);
    } else {
        NormalCalcFDWorkSpace(actCoreNum);
    }
}

void MlaLegacyValueState::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0U;
        uint32_t logSumExpSize = 0U;
        uint32_t FDParamNums = balanceModeFlag_ ? CalcBalanceFDParamNums(actCoreNum) : CalcUnbalanceFDParamNums();
        if (slidingFlag_) {
            accumOutSize = FDParamNums * headDimVAlign_;
        } else {
            accumOutSize = FDParamNums * headDimAlign_;
        }
        logSumExpSize = NUM2 * FDParamNums * (BYTE_BLOCK / blockTypeSize_); // log和sum的存储空间一致，共需要2份内存
        workspaceSize_ += static_cast<size_t>((accumOutSize + logSumExpSize)) * static_cast<size_t>(blockTypeSize_);
        if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
            workspaceSize_ += static_cast<size_t>(actCoreNum) * 32U; // 每个核SyncAll软同步需要32Byte记录状态
        }
    }
}

void MlaLegacyValueState::MLACalcFDWorkSpace(const uint32_t actCoreNum)
{
    uint32_t accumOutSize = 0U;
    uint32_t logSumExpSize = 0U;

    // BALANCE
    uint32_t FDParamNumsBalance = CalcBalanceFDParamNums(actCoreNum);
    uint32_t accumOutSizeBalance = FDParamNumsBalance * headDimAlign_;
    uint32_t logSumExpSizeBalance = 2U * FDParamNumsBalance * (BYTE_BLOCK / blockTypeSize_);

    // UNBALANCE
    uint32_t FDParamNumsUnbalance = CalcUnbalanceFDParamNums();
    uint32_t accumOutSizeUnbalance = FDParamNumsUnbalance * headDimAlign_;
    uint32_t logSumExpSizeUnbalance = 2U * FDParamNumsUnbalance * (BYTE_BLOCK / blockTypeSize_);

    if (isWorkspace_) { // tiling下沉场景，无法通过actual_seq_kv判断走哪种分核，针对不同layout采取不同workspace分配策略
        if (inputLayout_ == IfaLayout::TND) {
            accumOutSize = accumOutSizeBalance;
            logSumExpSize = logSumExpSizeBalance;
        } else if (inputLayout_ == IfaLayout::BSH_BSND) {
            accumOutSize = std::max(accumOutSizeBalance, accumOutSizeUnbalance);
            logSumExpSize = std::max(logSumExpSizeBalance, logSumExpSizeUnbalance);
        } else {
            accumOutSize = accumOutSizeUnbalance;
            logSumExpSize = logSumExpSizeUnbalance;
        }
        workspaceSize_ += static_cast<size_t>((accumOutSize + logSumExpSize)) * static_cast<size_t>(blockTypeSize_);
    } else {
        NormalCalcFDWorkSpace(actCoreNum);
    }
}

uint32_t MlaLegacyValueState::CalcBalanceFDParamNums(const uint32_t actCoreNum) const
{
    return actCoreNum * 2U * numKvHeads_ * FIA_BALANCE_SG_BASIC_SIZE; // 每个核可能有头规约和尾规约，一共两份规约信息
}

uint32_t MlaLegacyValueState::CalcUnbalanceFDParamNums() const
{
    return batchSizeQ_ * numHeads_ * kvSplitPart_;
}

void MlaLegacyValueState::FillTilingSplitKVMla()
{
    tilingDataMla_.splitKVParams.s2 = kvSplitPart_;
    uint32_t sInnerLoopSize_ = (maxActualseq_ + (kvSplitPart_ - 1U)) / kvSplitPart_;
    if (pageAttentionFlag_) {
        sInnerLoopSize_ = Align(sInnerLoopSize_, blockSize_);
        OP_LOGD(ifaContext_->opName, "PA FlashDecode is enabled, sInnerLoopSize is %u, blockSize is %u",
                sInnerLoopSize_, blockSize_);
    }
    if (inputKvType_ == ge::DT_INT4) {
        sInnerLoopSize_ = Align(sInnerLoopSize_, 2U);
    }
    tilingDataMla_.splitKVParams.sInnerLoopSize = sInnerLoopSize_;
    if (balanceModeFlag_) {
        tilingDataMla_.splitKVParams.accumOutSize = aicNum_ * 2U * numKvHeads_ * FIA_BALANCE_SG_BASIC_SIZE *
                                                    headDimAlign_; // 每个核可能有头规约和尾规约，一共两份规约信息
        tilingDataMla_.splitKVParams.logSumExpSize =
            2U * aicNum_ * 2U * numKvHeads_ *
            FIA_BALANCE_SG_BASIC_SIZE * // 每个核可能有头规约和尾规约，一共两份规约信息;sum + max
            (BYTE_BLOCK / blockTypeSize_);
    } else {
        tilingDataMla_.splitKVParams.accumOutSize = batchSizeQ_ * numHeads_ * kvSplitPart_ * headDimAlign_;
        tilingDataMla_.splitKVParams.logSumExpSize =
            2U * batchSizeQ_ * numHeads_ * kvSplitPart_ * (BYTE_BLOCK / blockTypeSize_); // 2: sum + max
    }
    if (!splitKVFlag_) {
        tilingDataMla_.splitKVParams.s2 = 0;
    }
}

void MlaLegacyValueState::FillTilingCoreParamsMla()
{
    uint32_t *coreStartIdx = tilingDataMla_.increFlashAttentionCoreParams.coreSidxEnd;
    memcpy_s(coreStartIdx, MAX_AIC_CORE_NUM * sizeof(uint32_t), startIdxEachCore_, MAX_AIC_CORE_NUM * sizeof(uint32_t));
}

void MlaLegacyValueState::FillTilingSingleCoreParamsMla()
{
    tilingDataMla_.increFlashAttentionSingleCoreParams.singleProcessSInnerSize = sInnerSize_;
    tilingDataMla_.increFlashAttentionSingleCoreParams.usedCoreNum = usedCoreNum_;
    tilingDataMla_.increFlashAttentionSingleCoreParams.groupSplitSize = groupSplitSize_;
    tilingDataMla_.increFlashAttentionSingleCoreParams.s1SplitSize = s1SplitSize_;
}

void MlaLegacyValueState::FillTilingSingleCoreTensorSizeMla()
{
    tilingDataMla_.increFlashAttentionSingleCoreTensorSize.mmResUbSize = mmResUbSize_;
    tilingDataMla_.increFlashAttentionSingleCoreTensorSize.bmm2ResUbSize = bmm2ResUbSize_;
}

IfaAmlaMode MlaLegacyValueState::GetAmlaMode() const
{
    // 1.只有NZ才应该走进这个函数，提前判断, 2.伪量化和全量化不走amla
    if ((!ropeFlag_) || inputKvLayout_ != IfaLayout::NZ || antiQuantFlag_ || quantFlag_) {
        return IfaAmlaMode::DISABLE_AMLA;
    }
    OP_LOGI(ifaContext_->opName, "input Key and Value format is NZ and blockSize_[%u], nNumOfQInOneGroup_[%u]",
            blockSize_, nNumOfQInOneGroup_);
    // 当前暂时只支持 大于24batch G128 MTP1,3 512对齐 走3buf模板
    if (blockSize_ == 128U) {
        return IfaAmlaMode::AMLA; // s1 * G <= 128 nz amla
    }
    return IfaAmlaMode::DISABLE_AMLA;
}

// Mechanically extracted from attention/incre_flash_attention/op_host/incre_flash_attention_tiling_check.cpp.
// Private FlashMla value-state methods; the original operator is unchanged.

ge::graphStatus MlaLegacyValueState::CheckUbSpace()
{
    // Only the BMM buffer sizes feed MLA scheduling/workspace. The discarded
    // softmax/select scratch-size queries belonged to Host tiling.
    if (!CalcUbBmm()) {
        return false;
    }
    return true;
}

bool MlaLegacyValueState::IsFlashDecode(uint32_t coreNum, IfaPerfMode perfMode)
{
    if (socVersion_ == IfaSocVersion::SOC_ASCEND_310P) {
        perfMode = IfaPerfMode::BMM_ALL_BY_VEC;
        if (nNumOfQInOneGroup_ * qSeqSize_ > 1U) {
            OP_LOGD(ifaContext_->opName, "flash decode has not been implemented when 310P and GQA > 1.");
            return false;
        }
    }
    if (balanceModeFlag_) {
        uint32_t tndFDCoreArrLen = tilingDataMla_.tndSplitCoreParams.tndFDCoreArrLen;
        return tndFDCoreArrLen > 0U;
    }
    if (pageAttentionFlag_ && socVersion_ == IfaSocVersion::SOC_ASCEND_910B &&
        perfMode != IfaPerfMode::CUBE_VIEW_MM_MLA && maxActualseq_ <= 1024U) { // 1024, 经验值
        return false;
    }
    constexpr uint32_t maxQNum = 128U;
    if (ropeFlag_ && (qSeqSize_ > 1U || nNumOfQInOneGroup_ > maxQNum)) {
        return false; // 投机推理 和 G > 64 切G待实现
    }

    if (gqaMtpFlag_) {
        return false; // 投机推理FD待实现
    }

    if (CheckCoreOkFlag(coreNum, perfMode)) {
        return true;
    }

    return false;
}

bool MlaLegacyValueState::CheckCoreOkFlag(uint32_t coreNum, IfaPerfMode perfMode) const
{
    bool coreOkFlag = false;
    float flashDecodeBNRatio = static_cast<float>(0.4); // 0.4, 经验值
    if (perfMode == IfaPerfMode::CUBE_VIEW_MM_FULL_LOAD || perfMode == IfaPerfMode::CUBE_VIEW_MM ||
        perfMode == IfaPerfMode::CUBE_VIEW_MM_MLA || perfMode == IfaPerfMode::CUBE_VIEW_MM_DD) {
        flashDecodeBNRatio = 0.5; // 0.5, 经验值
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) <= flashDecodeBNRatio * coreNum);
    } else if (perfMode == IfaPerfMode::BMM_ALL_BY_VEC) {
        flashDecodeBNRatio = 0.5; // 0.5, 全V模板可以按0.5切分
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    } else {
        coreOkFlag = (static_cast<float>(batchSize_ * numKvHeads_) < flashDecodeBNRatio * coreNum);
    }

    if (coreOkFlag && (nNumOfQInOneGroup_ == 1U)) {
        OP_LOGD(ifaContext_->opName, "Flash Decode Split kv."); // 非gqa时这里只判断bn是否满足
        return true;
    }

    if (coreOkFlag && (maxActualseq_ >= 2048U)) { // 2048, 在flash decode + gqa时的经验值
        OP_LOGD(ifaContext_->opName, "Flash Decode And GQA Split kv.");
        return true;
    }
    return false;
}

bool MlaLegacyValueState::DealSameSeqEachBatch() const
{
    if (!batchContinuousFlag_) {
        if (actualSeqLenFlag_) {
            return isSameActualseq_;
        } else {
            return isSameSeqAllKVTensor_;
        }
    } else {
        return isSameActualseq_;
    }
}

bool MlaLegacyValueState::IsKvZeroBatchSplit(bool needUpdate, uint32_t lastValidBMoreIdx, uint32_t bSize,
                                             const std::vector<uint32_t> &s1OuterNum,
                                             const std::vector<uint32_t> &s2OuterNum) const
{
    if (needUpdate) {
        return false;
    }
    if ((lastValidBMoreIdx <= 0U) || (lastValidBMoreIdx >= bSize)) {
        return false;
    }
    if (s1OuterNum[lastValidBMoreIdx] == 0) {
        return false;
    }
    if (s2OuterNum[lastValidBMoreIdx] != 0) {
        return false;
    }
    return true;
}

// Mechanically extracted from attention/incre_flash_attention/op_host/incre_flash_attention_tiling_check.cpp.
// Private FlashMla value-state methods; the original operator is unchanged.

ge::graphStatus MlaLegacyValueState::CheckTreeSparseMask()
{
    // 仅支持MLA全量化，且要求s2 >= s1
    OP_CHECK_IF(socVersion_ != IfaSocVersion::SOC_ASCEND_910B,
                OP_LOGE(ifaContext_->opName, "Tree sparse9 is only supported in ASCEND_910B."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!(ropeFlag_ && quantFlag_),
                OP_LOGE(ifaContext_->opName, "Tree Sparse9 is only supported in MLA full quant situation."),
                return ge::GRAPH_FAILED);

    // tiling下沉场景，获取不到actualseqlen,不进行校验
    if (isWorkspace_) {
        return ge::GRAPH_SUCCESS;
    }

    // 补充s2 >= s1的拦截，在MLA全量化场景，必须开启PA(NZ)，所以actualSeqKV不进行累加
    if (inputLayout_ == IfaLayout::TND) {
        const int64_t *actualSeqQTnd = ifaContext_->actualSeqLengthsQ.tensor->GetData<int64_t>();
        const int64_t *actualSeqKVTnd = ifaContext_->actualSeqLengths.tensor->GetData<int64_t>();
        int64_t qActSize = 0;
        int64_t kvActSize = 0;

        // s2=0 的 batch（入图padding或空tensor场景）不校验 s1<=s2
        for (int32_t b = 0; b < actualLenQDims_; b++) {
            kvActSize = actualSeqKVTnd[b];
            if (kvActSize == 0) {
                continue;
            }
            qActSize = (b == 0) ? actualSeqQTnd[0] : (actualSeqQTnd[b] - actualSeqQTnd[b - 1]);
            OP_CHECK_IF(qActSize > kvActSize,
                        OP_LOGE(ifaContext_->opName,
                                "In MLA full quant situation, when sparse is %u, qSize(%ld) should be less than or "
                                "equal to kvSize(%ld).",
                                sparseMode_, qActSize, kvActSize),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(qSeqSize_ > seqSize_,
                    OP_LOGE(ifaContext_->opName,
                            "In MLA full quant situation, when sparse is %u, qSize(%u) should be less than or equal "
                            "to kvSize(%u).",
                            sparseMode_, qSeqSize_, seqSize_),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace Arch22
} // namespace FlashMla
