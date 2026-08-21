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
 * \file fused_infer_attention_score_tiling_v4.cpp
 * \brief
 */

#include "../../../common/op_host/fia_tiling_templates_registry.h"
#include "fused_infer_attention_score_tiling_v4.h"
#include "../fused_infer_attention_score_tiling_info_parser.h"
#include "../checkers/fia_checker.h"
#include "log/log.h"

namespace optiling {
namespace {
void PrintFiaDiagSequence(const FiaTilingInfo &fiaInfo)
{
    const gert::Tensor *actualQTensor = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    const gert::Tensor *actualKvTensor = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    const int64_t qDims = actualQTensor == nullptr ? 0 : actualQTensor->GetShapeSize();
    const int64_t kvDims = actualKvTensor == nullptr ? 0 : actualKvTensor->GetShapeSize();
    const int64_t *actualQ = actualQTensor == nullptr ? nullptr : actualQTensor->GetData<int64_t>();
    const int64_t *actualKv = actualKvTensor == nullptr ? nullptr : actualKvTensor->GetData<int64_t>();
    const int64_t diagDims = qDims > kvDims ? qDims : kvDims;

    OP_LOGI(fiaInfo.opName,
            "[FIA_DIAG][HOST][SEQ_META] qDims=%lld kvDims=%lld qData=%u kvData=%u accumQ=%u accumKV=%u",
            qDims, kvDims, actualQ != nullptr, actualKv != nullptr, fiaInfo.isAccumQSeq, fiaInfo.isAccumKVSeq);
    for (int64_t index = 0; index < diagDims; ++index) {
        const int64_t qValue = actualQ != nullptr && index < qDims ? actualQ[index] : -1;
        const int64_t kvValue = actualKv != nullptr && index < kvDims ? actualKv[index] : -1;
        OP_LOGI(fiaInfo.opName, "[FIA_DIAG][HOST][SEQ] index=%lld actualQ=%lld actualKV=%lld", index, qValue,
                kvValue);
    }
}

void PrintFiaDiagPadding(const FiaTilingInfo &fiaInfo)
{
    const gert::Tensor *qPaddingTensor = fiaInfo.opParamInfo.queryPaddingSize.tensor;
    const gert::Tensor *kvPaddingTensor = fiaInfo.opParamInfo.kvPaddingSize.tensor;
    const int64_t *qPadding = qPaddingTensor == nullptr ? nullptr : qPaddingTensor->GetData<int64_t>();
    const int64_t *kvPadding = kvPaddingTensor == nullptr ? nullptr : kvPaddingTensor->GetData<int64_t>();
    OP_LOGI(fiaInfo.opName,
            "[FIA_DIAG][HOST][PADDING] qFlag=%u qValue=%lld kvFlag=%u kvValue=%lld",
            fiaInfo.qPaddingSizeFlag, qPadding == nullptr ? -1 : qPadding[0], fiaInfo.kvPaddingSizeFlag,
            kvPadding == nullptr ? -1 : kvPadding[0]);
}

void PrintFiaDiagBlockTable(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION || fiaInfo.opParamInfo.blockTable.tensor == nullptr) {
        return;
    }
    const gert::Tensor *blockTableTensor = fiaInfo.opParamInfo.blockTable.tensor;
    const int32_t *blockTable = blockTableTensor->GetData<int32_t>();
    const int64_t blockTableSize = blockTableTensor->GetShapeSize();
    const gert::Tensor *actualKvTensor = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    const int64_t *actualKv = actualKvTensor == nullptr ? nullptr : actualKvTensor->GetData<int64_t>();
    const int64_t actualKvDims = actualKvTensor == nullptr ? 0 : actualKvTensor->GetShapeSize();
    OP_LOGI(fiaInfo.opName,
            "[FIA_DIAG][HOST][PA_META] data=%u elements=%lld dim2=%u blockSize=%d totalBlocks=%u",
            blockTable != nullptr, blockTableSize, fiaInfo.maxBlockNumPerBatch, fiaInfo.blockSize,
            fiaInfo.totalBlockNum);
    if (blockTable == nullptr || actualKv == nullptr || fiaInfo.blockSize <= 0 ||
        fiaInfo.maxBlockNumPerBatch == 0) {
        return;
    }

    for (uint32_t batch = 0; batch < fiaInfo.bSize && batch < static_cast<uint32_t>(actualKvDims); ++batch) {
        const int64_t usedBlocks = (actualKv[batch] + fiaInfo.blockSize - 1) / fiaInfo.blockSize;
        const int64_t rowOffset = static_cast<int64_t>(batch) * fiaInfo.maxBlockNumPerBatch;
        int64_t invalidCount = 0;
        int64_t firstInvalidSlot = -1;
        int32_t firstInvalidValue = -1;
        const int64_t scanBlocks = usedBlocks < fiaInfo.maxBlockNumPerBatch ? usedBlocks : fiaInfo.maxBlockNumPerBatch;
        for (int64_t slot = 0; slot < scanBlocks && rowOffset + slot < blockTableSize; ++slot) {
            const int32_t blockId = blockTable[rowOffset + slot];
            if (blockId < 0 || static_cast<uint32_t>(blockId) >= fiaInfo.totalBlockNum) {
                ++invalidCount;
                if (firstInvalidSlot < 0) {
                    firstInvalidSlot = slot;
                    firstInvalidValue = blockId;
                }
            }
        }
        const int32_t firstBlock = usedBlocks > 0 && rowOffset < blockTableSize ? blockTable[rowOffset] : -1;
        const int64_t lastOffset = rowOffset + usedBlocks - 1;
        const int32_t lastBlock = usedBlocks > 0 && lastOffset < blockTableSize ? blockTable[lastOffset] : -1;
        OP_LOGI(fiaInfo.opName,
                "[FIA_DIAG][HOST][PA] batch=%u actualKV=%lld used=%lld first=%d last=%d invalid=%lld "
                "firstBadSlot=%lld firstBadValue=%d",
                batch, actualKv[batch], usedBlocks, firstBlock, lastBlock, invalidCount, firstInvalidSlot,
                firstInvalidValue);
    }
}

void PrintFiaDiagInfo(const FiaTilingInfo &fiaInfo)
{
    OP_LOGI(fiaInfo.opName,
            "[FIA_DIAG][HOST][CASE] arch=%d layout=%s b=%u n1=%u n2=%u g=%u s1=%u s2=%lld d=%u dv=%u "
            "ropeD=%u",
            static_cast<int32_t>(fiaInfo.npuArch), fiaInfo.opParamInfo.layOut, fiaInfo.bSize, fiaInfo.n1Size,
            fiaInfo.n2Size, fiaInfo.gSize, fiaInfo.s1Size, fiaInfo.s2Size, fiaInfo.qkHeadDim, fiaInfo.vHeadDim,
            fiaInfo.ropeHeadDim);
    OP_LOGI(fiaInfo.opName,
            "[FIA_DIAG][HOST][FEATURE] qType=%d kvType=%d outType=%d quant=%u fullQuant=%u mla=%u rope=%u "
            "kvStorage=%u mask=%u sparse=%d pre=%lld next=%lld postQuant=%u lse=%u",
            static_cast<int32_t>(fiaInfo.inputQType), static_cast<int32_t>(fiaInfo.inputKvType),
            static_cast<int32_t>(fiaInfo.outputType), static_cast<uint32_t>(fiaInfo.quantMode),
            static_cast<uint32_t>(fiaInfo.fullQuantMode), static_cast<uint32_t>(fiaInfo.mlaMode),
            static_cast<uint32_t>(fiaInfo.ropeMode), static_cast<uint32_t>(fiaInfo.kvStorageMode),
            fiaInfo.attenMaskFlag, fiaInfo.sparseMode, fiaInfo.preToken, fiaInfo.nextToken,
            fiaInfo.isOutQuantEnable, fiaInfo.softmaxLseFlag);
    OP_LOGI(fiaInfo.opName,
            "[FIA_DIAG][HOST][PA_CONFIG] enabled=%u blockSize=%d dim2=%u totalBlocks=%u kvLayout=%u "
            "transpose=%u",
            fiaInfo.pageAttentionFlag, fiaInfo.blockSize, fiaInfo.maxBlockNumPerBatch, fiaInfo.totalBlockNum,
            fiaInfo.kvLayoutType, static_cast<uint32_t>(fiaInfo.outLayout));
    OP_LOGI(fiaInfo.opName,
            "[FIA_DIAG][HOST][CACHE] continuous=%u viewStride=%u keyBn=%llu keyN2=%llu valueBn=%llu "
            "valueN2=%llu kRopeBn=%llu kRopeN2=%llu",
            fiaInfo.batchContinuousFlag, fiaInfo.hasViewStride, fiaInfo.keyBnStride, fiaInfo.keyN2Stride,
            fiaInfo.valueBnStride, fiaInfo.valueN2Stride, fiaInfo.kRopeBnStride, fiaInfo.kRopeN2Stride);
    PrintFiaDiagSequence(fiaInfo);
    PrintFiaDiagPadding(fiaInfo);
    PrintFiaDiagBlockTable(fiaInfo);
}
} // namespace

ge::graphStatus TilingFusedInferAttentionScoreV4(gert::TilingContext *context) {
    // Parse -> Check -> DoOpTiling
    FiaTilingInfo fiaInfo;
    FiaInfoParser fiaInfoParser(context);
    if (fiaInfoParser.Parse(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PrintFiaDiagInfo(fiaInfo);

    FIAChecker fiaChecker;
    fiaChecker.Init(fiaInfo);
    // Check函数只做校验，不能修改fiaInfo中的信息
    if (fiaChecker.Process(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(fiaInfo.opName, "[FIA_DIAG][HOST][TEMPLATE] registry_begin");
    const ge::graphStatus status = FiaTilingRegistry::GetInstance().DoTilingImpl(context, &fiaInfo);
    OP_LOGI(fiaInfo.opName, "[FIA_DIAG][HOST][TEMPLATE] registry_end status=%d", status);
    return status;
}

} // namespace optiling
