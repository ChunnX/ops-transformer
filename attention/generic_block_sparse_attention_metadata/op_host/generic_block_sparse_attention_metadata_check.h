/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_CHECK_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_CHECK_H

#include <cstring>
#include <string>

#include "opdev/data_type_utils.h"
#include "opdev/op_log.h"
#include "../generic_block_sparse_attention_metadata.h"

namespace {

bool GbsaTensorValid(const aclTensor *tensor)
{
    return tensor != nullptr && tensor->GetViewShape().GetDimNum() > 0;
}

bool GbsaIsLayout(const char *layout, const char *expected)
{
    return layout != nullptr && std::strcmp(layout, expected) == 0;
}

aclnnStatus CheckGbsaTensorDataType(const aclTensor *tensor, aclDataType expectedDataType, const char *expectedTypeName,
                                   const char *tensorName)
{
    aclDataType dataType = ACL_DT_UNDEFINED;
    aclGetDataType(tensor, &dataType);
    if (dataType != expectedDataType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be %s.", tensorName, expectedTypeName);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaTensorDim(const aclTensor *tensor, int64_t expectedDim, const char *tensorName)
{
    if (tensor->GetViewShape().GetDimNum() != expectedDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s dimension must be %ld.", tensorName, expectedDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaRequiredInt32Tensor(const aclTensor *tensor, const char *tensorName)
{
    if (!GbsaTensorValid(tensor)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be provided.", tensorName);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return CheckGbsaTensorDataType(tensor, ACL_INT32, "INT32", tensorName);
}

aclnnStatus CheckGbsaRequiredInputs(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                                   const aclTensor *metadata)
{
    aclnnStatus status = CheckGbsaRequiredInt32Tensor(sparseBlockIdx, "sparseBlockIdx");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaRequiredInt32Tensor(sparseBlockCount, "SparseBlockCount");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaRequiredInt32Tensor(metadata, "metadata");
}

aclnnStatus CheckGbsaMetadataShape(const aclTensor *metadata)
{
    aclnnStatus status = CheckGbsaTensorDim(metadata, 1, "metadata");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (metadata->GetViewShape().GetDim(0) !=
        static_cast<int64_t>(optiling::generic_block_sparse_attention_metadata::METADATA_TOTAL_SIZE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "metadata shape must be [%u].",
                optiling::generic_block_sparse_attention_metadata::METADATA_TOTAL_SIZE);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaPositiveScalarAttr(int64_t attrValue, const char *attrName)
{
    if (attrValue <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be greater than 0, but got %lld.", attrName, attrValue);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaScalarAttrs(int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads, int64_t numKvHeads,
                                int64_t headDim, int64_t blockShapeX, int64_t blockShapeY, int64_t isPackedGQA)
{
    const struct {
        int64_t value;
        const char *name;
    } positiveAttrs[] = {{maxQSeqLen, "maxQSeqLen"}, {maxKvSeqLen, "maxKvSeqLen"}, {numQHeads, "numQHeads"},
                         {numKvHeads, "numKvHeads"}, {headDim, "headDim"},         {blockShapeY, "blockShapeY"}};
    for (const auto &attr : positiveAttrs) {
        const aclnnStatus status = CheckGbsaPositiveScalarAttr(attr.value, attr.name);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    if (blockShapeX != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeX currently only supports 1, but got %lld.", blockShapeX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (blockShapeY % 16 != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeY must be a multiple of 16, but got %lld.", blockShapeY);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (isPackedGQA != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "isPackedGQA currently only supports 1, but got %lld.", isPackedGQA);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaLayouts(const char *qInputLayout, const char *kvInputLayout)
{
    const bool qLayoutValid =
        GbsaIsLayout(qInputLayout, "TND") || GbsaIsLayout(qInputLayout, "BNSD") || GbsaIsLayout(qInputLayout, "BSND");
    const bool kvLayoutValid = GbsaIsLayout(kvInputLayout, "TND") || GbsaIsLayout(kvInputLayout, "BNSD") ||
                               GbsaIsLayout(kvInputLayout, "BSND") || GbsaIsLayout(kvInputLayout, "PAGED_BBND") ||
                               GbsaIsLayout(kvInputLayout, "PAGED_BNBD");
    if (!qLayoutValid || !kvLayoutValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported qInputLayout or kvInputLayout.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaOptionalSeqLength(const aclTensor *tensor, int64_t batch, aclDataType dataType,
                                      int64_t expectedElements, const char *typeName, const char *tensorName)
{
    if (!GbsaTensorValid(tensor)) {
        return ACLNN_SUCCESS;
    }
    aclnnStatus status = CheckGbsaTensorDataType(tensor, dataType, typeName, tensorName);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaTensorDim(tensor, 1, tensorName);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (tensor->GetViewShape().GetDim(0) != expectedElements) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s shape does not match batch %lld.", tensorName, batch);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus GetGbsaBatch(const aclTensor *sparseBlockIdx, const aclTensor *cuSeqLengthsOptional,
                        const char *qInputLayout, int64_t &batch)
{
    if (!GbsaIsLayout(qInputLayout, "TND")) {
        batch = sparseBlockIdx->GetViewShape().GetDim(0);
        return batch >= 0 ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID;
    }
    if (!GbsaTensorValid(cuSeqLengthsOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths is required for TND query.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cuSeqLengthsOptional->GetViewShape().GetDimNum() != 1 || cuSeqLengthsOptional->GetViewShape().GetDim(0) < 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths shape must be [batch + 1].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    batch = cuSeqLengthsOptional->GetViewShape().GetDim(0) - 1;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaSparseTensorShapes(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                                       int64_t maxQSeqLen, int64_t numQHeads, int64_t numKvHeads, int64_t blockShapeX,
                                       int64_t isPackedGQA, const char *qInputLayout, int64_t batch)
{
    const int64_t sparseHeads = isPackedGQA == 1 ? numKvHeads : numQHeads;
    if (GbsaIsLayout(qInputLayout, "TND")) {
        if (CheckGbsaTensorDim(sparseBlockIdx, 3, "sparseBlockIdx") != ACLNN_SUCCESS ||
            CheckGbsaTensorDim(sparseBlockCount, 2, "SparseBlockCount") != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (sparseBlockIdx->GetViewShape().GetDim(0) != sparseHeads ||
            sparseBlockCount->GetViewShape().GetDim(0) != sparseHeads ||
            sparseBlockIdx->GetViewShape().GetDim(1) != sparseBlockCount->GetViewShape().GetDim(1) ||
            sparseBlockIdx->GetViewShape().GetDim(2) <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TND sparse block tensor shapes do not match attrs.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    if (CheckGbsaTensorDim(sparseBlockIdx, 4, "sparseBlockIdx") != ACLNN_SUCCESS ||
        CheckGbsaTensorDim(sparseBlockCount, 3, "SparseBlockCount") != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t qBlocks = maxQSeqLen / blockShapeX + static_cast<int64_t>(maxQSeqLen % blockShapeX != 0);
    if (sparseBlockIdx->GetViewShape().GetDim(0) != batch || sparseBlockCount->GetViewShape().GetDim(0) != batch ||
        sparseBlockIdx->GetViewShape().GetDim(1) != sparseHeads ||
        sparseBlockCount->GetViewShape().GetDim(1) != sparseHeads ||
        sparseBlockIdx->GetViewShape().GetDim(2) != qBlocks || sparseBlockCount->GetViewShape().GetDim(2) != qBlocks ||
        sparseBlockIdx->GetViewShape().GetDim(3) <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSND/BNSD sparse block tensor shapes do not match attrs.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaSeqLengthInputs(const aclTensor *cuSeqLengthsOptional, const aclTensor *cuSeqLengthsKvOptional,
                                    const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKvOptional,
                                    int64_t batch, const char *qInputLayout, const char *kvInputLayout)
{
    if (GbsaIsLayout(qInputLayout, "TND") && !GbsaTensorValid(cuSeqLengthsOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths is required for TND query.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool kvSeqLengthsRequired = GbsaIsLayout(kvInputLayout, "TND") || GbsaIsLayout(kvInputLayout, "PAGED_BBND") ||
                                      GbsaIsLayout(kvInputLayout, "PAGED_BNBD");
    if (kvSeqLengthsRequired && !GbsaTensorValid(cuSeqLengthsKvOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengthsKv is required for TND or paged KV.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    aclnnStatus status =
        CheckGbsaOptionalSeqLength(cuSeqLengthsOptional, batch, ACL_INT64, batch + 1, "INT64", "cuSeqLengths");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaOptionalSeqLength(cuSeqLengthsKvOptional, batch, ACL_INT64, batch + 1, "INT64", "cuSeqLengthsKv");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaOptionalSeqLength(seqUsedQOptional, batch, ACL_INT32, batch, "INT32", "seqUsedQ");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaOptionalSeqLength(seqUsedKvOptional, batch, ACL_INT32, batch, "INT32", "seqUsedKv");
}

aclnnStatus CheckGbsaQuantType(int64_t quantType, const std::string &socVersion)
{
    if (quantType != 0 && socVersion.find("Ascend950") == std::string::npos) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantType is only supported on Atlas A5.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaMaskAttrs(int64_t maskType, int64_t windowSizeLeft, int64_t windowSizeRight)
{
    if (maskType < 0 || maskType > 2 || (maskType != 2 && (windowSizeLeft != -1 || windowSizeRight != -1)) ||
        (maskType == 2 && (windowSizeLeft < 0 || windowSizeRight < 0))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maskType/windowSize attrs are invalid.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaMetadataParams(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                                   const aclTensor *cuSeqLengthsOptional, const aclTensor *cuSeqLengthsKvOptional,
                                   const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKvOptional,
                                   int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads, int64_t numKvHeads,
                                   int64_t headDim, int64_t blockShapeX, int64_t blockShapeY, int64_t isPackedGQA,
                                   const char *qInputLayout, const char *kvInputLayout, int64_t maskType,
                                   int64_t quantType, int64_t windowSizeLeft, int64_t windowSizeRight,
                                   const std::string &socVersion, const aclTensor *metadata)
{
    aclnnStatus status = CheckGbsaRequiredInputs(sparseBlockIdx, sparseBlockCount, metadata);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaScalarAttrs(maxQSeqLen, maxKvSeqLen, numQHeads, numKvHeads, headDim, blockShapeX, blockShapeY,
                                 isPackedGQA);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaLayouts(qInputLayout, kvInputLayout);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaMetadataShape(metadata);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    int64_t batch = 0;
    status = GetGbsaBatch(sparseBlockIdx, cuSeqLengthsOptional, qInputLayout, batch);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaSparseTensorShapes(sparseBlockIdx, sparseBlockCount, maxQSeqLen, numQHeads, numKvHeads,
                                        blockShapeX, isPackedGQA, qInputLayout, batch);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaSeqLengthInputs(cuSeqLengthsOptional, cuSeqLengthsKvOptional, seqUsedQOptional, seqUsedKvOptional,
                                     batch, qInputLayout, kvInputLayout);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaQuantType(quantType, socVersion);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaMaskAttrs(maskType, windowSizeLeft, windowSizeRight);
}

} // namespace

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_CHECK_H
