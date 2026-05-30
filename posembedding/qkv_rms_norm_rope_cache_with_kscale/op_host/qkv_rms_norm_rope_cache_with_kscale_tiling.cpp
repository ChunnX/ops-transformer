/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qkv_rms_norm_rope_cache_with_kscale_tiling.h"

#include "log/log.h"
#include "qkv_rms_norm_rope_cache_with_kscale_contract.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <limits>

namespace optiling {
namespace {
using qkv_rms_norm_rope_cache_with_kscale::ContractInput;
using qkv_rms_norm_rope_cache_with_kscale::ContractDtype;
using qkv_rms_norm_rope_cache_with_kscale::ContractStatus;
using qkv_rms_norm_rope_cache_with_kscale::ContractStatusName;
using qkv_rms_norm_rope_cache_with_kscale::ContractTiling;
using qkv_rms_norm_rope_cache_with_kscale::CalcTileNum;
using qkv_rms_norm_rope_cache_with_kscale::TensorContractInfo;
using qkv_rms_norm_rope_cache_with_kscale::TensorShapeInfo;
using qkv_rms_norm_rope_cache_with_kscale::ValidateAndBuildTiling;

constexpr size_t kQkvIndex = 0;
constexpr size_t kQGammaIndex = 1;
constexpr size_t kKGammaIndex = 2;
constexpr size_t kCosSinIndex = 3;
constexpr size_t kSlotMappingIndex = 4;
constexpr size_t kKCacheIndex = 5;
constexpr size_t kVCacheIndex = 6;
constexpr size_t kKScaleCacheIndex = 7;
constexpr size_t kCuSeqLenIndex = 8;
constexpr size_t kRotationIndex = 9;
constexpr size_t kVScaleIndex = 10;

constexpr size_t kHeadNumsAttrIndex = 0;
constexpr size_t kEpsilonAttrIndex = 1;

constexpr uint64_t kTilingKeyArch35D128 = 350128;
constexpr float kDefaultEpsilon = 1e-6f;

TensorShapeInfo ToShapeInfo(const gert::StorageShape *shape)
{
    TensorShapeInfo info;
    if (shape == nullptr) {
        info.present = false;
        return info;
    }
    const gert::Shape &storageShape = shape->GetStorageShape();
    info.rank = static_cast<uint32_t>(storageShape.GetDimNum());
    if (info.rank > 0) {
        info.dim0 = static_cast<uint64_t>(storageShape.GetDim(0));
    }
    if (info.rank > 1) {
        info.dim1 = static_cast<uint64_t>(storageShape.GetDim(1));
    }
    if (info.rank > 2) {
        info.dim2 = static_cast<uint64_t>(storageShape.GetDim(2));
    }
    if (info.rank > 3) {
        info.dim3 = static_cast<uint64_t>(storageShape.GetDim(3));
    }
    return info;
}

ContractDtype ToContractDtype(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BF16:
            return ContractDtype::kBf16;
        case ge::DT_FLOAT:
            return ContractDtype::kFloat;
        case ge::DT_INT32:
            return ContractDtype::kInt32;
        case ge::DT_FLOAT8_E4M3FN:
            return ContractDtype::kFloat8E4M3fn;
        default:
            return ContractDtype::kUnknown;
    }
}

TensorContractInfo ToTensorInfo(gert::TilingContext *context, size_t index)
{
    TensorContractInfo info;
    info.shape = ToShapeInfo(context->GetInputShape(index));
    const auto desc = context->GetInputDesc(index);
    if (desc != nullptr) {
        info.dtype = ToContractDtype(desc->GetDataType());
    }
    return info;
}

TensorContractInfo ToOptionalTensorInfo(gert::TilingContext *context, size_t index)
{
    TensorContractInfo info;
    info.shape = ToShapeInfo(context->GetOptionalInputShape(index));
    const auto desc = context->GetOptionalInputDesc(index);
    if (desc != nullptr) {
        info.dtype = ToContractDtype(desc->GetDataType());
    }
    return info;
}

ge::graphStatus BuildContractInput(gert::TilingContext *context, ContractInput &input, float &epsilon)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto headNums = attrs->GetListInt(kHeadNumsAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNums);
    OP_CHECK_IF(headNums->GetSize() != 3,
        OP_LOGE(context->GetNodeName(), "head_nums must have exactly 3 values."), return ge::GRAPH_FAILED);
    const auto *headNumsData = headNums->GetData();
    for (size_t i = 0; i < 3; ++i) {
        OP_CHECK_IF(headNumsData[i] <= 0 ||
            headNumsData[i] > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
            OP_LOGE(context->GetNodeName(), "head_nums[%zu] must be positive uint32.", i),
            return ge::GRAPH_FAILED);
    }

    const auto qkv = ToTensorInfo(context, kQkvIndex);
    const auto cosSin = ToTensorInfo(context, kCosSinIndex);
    const auto kCache = ToTensorInfo(context, kKCacheIndex);
    const auto cuSeqLen = ToTensorInfo(context, kCuSeqLenIndex);
    const auto &qkvShape = qkv.shape;
    const auto &cosSinShape = cosSin.shape;
    const auto &kCacheShape = kCache.shape;
    const auto &cuSeqLenShape = cuSeqLen.shape;
    if (qkvShape.rank >= 3) {
        input.totalTokens = static_cast<uint32_t>(qkvShape.dim0);
        input.headDim = static_cast<uint32_t>(qkvShape.dim2);
    }
    if (cuSeqLenShape.rank == 1 && cuSeqLenShape.dim0 > 0) {
        input.batch = static_cast<uint32_t>(cuSeqLenShape.dim0 - 1);
    }
    if (cosSinShape.rank >= 1) {
        input.maxSeqLen = static_cast<uint32_t>(cosSinShape.dim0);
    }
    if (kCacheShape.rank >= 4) {
        input.blockNum = static_cast<uint32_t>(kCacheShape.dim0);
        input.blockSize = static_cast<uint32_t>(kCacheShape.dim2);
    }

    input.numQHeads = static_cast<uint32_t>(headNumsData[0]);
    input.numKHeads = static_cast<uint32_t>(headNumsData[1]);
    input.numVHeads = static_cast<uint32_t>(headNumsData[2]);

    input.qkv = qkv;
    input.qGamma = ToTensorInfo(context, kQGammaIndex);
    input.kGamma = ToTensorInfo(context, kKGammaIndex);
    input.cosSin = cosSin;
    input.slotMapping = ToTensorInfo(context, kSlotMappingIndex);
    input.kCache = kCache;
    input.vCache = ToTensorInfo(context, kVCacheIndex);
    input.kScaleCache = ToTensorInfo(context, kKScaleCacheIndex);
    input.cuSeqLen = cuSeqLen;
    input.rotation = ToOptionalTensorInfo(context, kRotationIndex);
    input.vScale = ToOptionalTensorInfo(context, kVScaleIndex);

    const float *epsilonAttr = attrs->GetFloat(kEpsilonAttrIndex);
    epsilon = epsilonAttr == nullptr ? kDefaultEpsilon : *epsilonAttr;
    return ge::GRAPH_SUCCESS;
}

void FillTilingData(const ContractInput &input, const ContractTiling &contractTiling, float epsilon,
    QkvRmsNormRopeCacheWithKScaleTilingData &tilingData)
{
    tilingData.set_totalTokens(input.totalTokens);
    tilingData.set_batch(input.batch);
    tilingData.set_numQHeads(input.numQHeads);
    tilingData.set_numKHeads(input.numKHeads);
    tilingData.set_numVHeads(input.numVHeads);
    tilingData.set_headDim(input.headDim);
    tilingData.set_maxSeqLen(input.maxSeqLen);
    tilingData.set_blockNum(input.blockNum);
    tilingData.set_blockSize(input.blockSize);
    tilingData.set_qkvStrideT((input.numQHeads + input.numKHeads + input.numVHeads) * input.headDim);
    tilingData.set_qkvStrideH(input.headDim);
    tilingData.set_cacheStrideBlock(input.numKHeads * input.blockSize * input.headDim);
    tilingData.set_cacheStrideHead(input.blockSize * input.headDim);
    tilingData.set_cacheStrideToken(input.headDim);
    tilingData.set_tokenTile(contractTiling.tokenTile);
    tilingData.set_tokenTilePerAiv(contractTiling.tokenTilePerAiv);
    tilingData.set_rowTile(contractTiling.rowTile);
    tilingData.set_rowTileAligned(contractTiling.rowTileAligned);
    tilingData.set_qkHeadTile(contractTiling.qkHeadTile);
    tilingData.set_vHeadTile(contractTiling.vHeadTile);
    tilingData.set_dimTile(contractTiling.dimTile);
    tilingData.set_flags(QKV_RRC_FLAG_D128_FAST | QKV_RRC_FLAG_ROTATION_L1_RESIDENT |
        QKV_RRC_FLAG_V_CACHE_WRITE_AFTER_A_READY | QKV_RRC_FLAG_TOKEN_TILE_EVEN | QKV_RRC_FLAG_L0C_DOUBLE_BUFFER);
    tilingData.set_epsilon(epsilon);
}

} // namespace

ge::graphStatus Tiling4QkvRmsNormRopeCacheWithKScale(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto compileInfo = context->GetCompileInfo<QkvRmsNormRopeCacheWithKScaleCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_CHECK_IF(compileInfo->aicNum == 0,
        OP_LOGE(context->GetNodeName(), "aicNum in compile info must be greater than 0."),
        return ge::GRAPH_FAILED);

    ContractInput input;
    float epsilon = kDefaultEpsilon;
    OP_CHECK_IF(BuildContractInput(context, input, epsilon) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "failed to build QkvRmsNormRopeCacheWithKScale contract input."),
        return ge::GRAPH_FAILED);

    ContractTiling contractTiling;
    const ContractStatus status = ValidateAndBuildTiling(input, contractTiling);
    OP_CHECK_IF(status != ContractStatus::kSuccess,
        OP_LOGE(context->GetNodeName(), "QkvRmsNormRopeCacheWithKScale contract failed: %s",
            ContractStatusName(status)),
        return ge::GRAPH_FAILED);

    QkvRmsNormRopeCacheWithKScaleTilingData tilingData;
    FillTilingData(input, contractTiling, epsilon, tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData()->GetData());
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetTilingKey(kTilingKeyArch35D128);
    context->SetBlockDim(std::min(CalcTileNum(input.totalTokens, contractTiling.tokenTile), compileInfo->aicNum));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4QkvRmsNormRopeCacheWithKScale(gert::TilingParseContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto compileInfo = context->GetCompiledInfo<QkvRmsNormRopeCacheWithKScaleCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aicNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAic());
    compileInfo->aivNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo->l0cSize);
    OP_CHECK_IF(compileInfo->aicNum == 0 || compileInfo->aivNum == 0 || compileInfo->ubSize == 0 ||
        compileInfo->l1Size == 0 || compileInfo->l0cSize == 0,
        OP_LOGE(context->GetNodeName(), "invalid platform resource for QkvRmsNormRopeCacheWithKScale."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QkvRmsNormRopeCacheWithKScale)
    .Tiling(Tiling4QkvRmsNormRopeCacheWithKScale)
    .TilingParse<QkvRmsNormRopeCacheWithKScaleCompileInfo>(TilingPrepare4QkvRmsNormRopeCacheWithKScale);

} // namespace optiling
