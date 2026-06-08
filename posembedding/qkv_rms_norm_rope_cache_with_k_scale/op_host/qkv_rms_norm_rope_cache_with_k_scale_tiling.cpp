/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qkv_rms_norm_rope_cache_with_k_scale_tiling.h"

#include "../op_kernel/qkv_rms_norm_rope_cache_with_k_scale_tiling_key.h"
#include "log/log.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_contract.h"
#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
using qkv_rms_norm_rope_cache_with_k_scale::ContractDtype;
using qkv_rms_norm_rope_cache_with_k_scale::ContractInput;
using qkv_rms_norm_rope_cache_with_k_scale::ContractStatus;
using qkv_rms_norm_rope_cache_with_k_scale::ContractStatusName;
using qkv_rms_norm_rope_cache_with_k_scale::ContractTiling;
using qkv_rms_norm_rope_cache_with_k_scale::TensorContractInfo;
using qkv_rms_norm_rope_cache_with_k_scale::TensorShapeInfo;
using qkv_rms_norm_rope_cache_with_k_scale::TensorStrideInfo;
using qkv_rms_norm_rope_cache_with_k_scale::ValidateAndBuildTiling;
using namespace QkvRmsNormRopeCacheWithKScaleTiling;

constexpr size_t kQkvIndex = 0;
constexpr size_t kQGammaIndex = 1;
constexpr size_t kKGammaIndex = 2;
constexpr size_t kCosSinIndex = 3;
constexpr size_t kSlotMappingIndex = 4;
constexpr size_t kKCacheIndex = 5;
constexpr size_t kVCacheIndex = 6;
constexpr size_t kKScaleCacheIndex = 7;
constexpr size_t kCuSeqLenIndex = 8;
constexpr size_t kQRotationIndex = 9;
constexpr size_t kKRotationIndex = 10;
constexpr size_t kVScaleIndex = 11;

constexpr size_t kHeadNumsAttrIndex = 0;
constexpr size_t kEpsilonAttrIndex = 1;

constexpr float kDefaultEpsilon = 1e-6f;
constexpr size_t kWorkspaceCount = 1;
constexpr size_t kReservedWorkspaceSize = 16UL * 1024UL * 1024UL;

TensorShapeInfo ToShapeInfo(const gert::Shape &shape)
{
    TensorShapeInfo info;
    info.rank = static_cast<uint32_t>(shape.GetDimNum());
    if (info.rank > 0) {
        info.dim0 = static_cast<uint64_t>(shape.GetDim(0));
    }
    if (info.rank > 1) {
        info.dim1 = static_cast<uint64_t>(shape.GetDim(1));
    }
    if (info.rank > 2) {
        info.dim2 = static_cast<uint64_t>(shape.GetDim(2));
    }
    if (info.rank > 3) {
        info.dim3 = static_cast<uint64_t>(shape.GetDim(3));
    }
    return info;
}

TensorShapeInfo ToShapeInfo(const gert::StorageShape *shape)
{
    if (shape == nullptr) {
        TensorShapeInfo info;
        info.present = false;
        return info;
    }
    return ToShapeInfo(shape->GetStorageShape());
}

TensorStrideInfo ToStrideInfo(const gert::Stride &stride)
{
    TensorStrideInfo info;
    info.rank = static_cast<uint32_t>(stride.GetDimNum());
    if (info.rank > 0) {
        info.stride0 = static_cast<uint64_t>(stride.GetStride(0));
    }
    if (info.rank > 1) {
        info.stride1 = static_cast<uint64_t>(stride.GetStride(1));
    }
    if (info.rank > 2) {
        info.stride2 = static_cast<uint64_t>(stride.GetStride(2));
    }
    if (info.rank > 3) {
        info.stride3 = static_cast<uint64_t>(stride.GetStride(3));
    }
    return info;
}

TensorStrideInfo MakeContiguousStrideInfo(const gert::Shape &shape)
{
    gert::Stride stride;
    stride.SetDimNum(shape.GetDimNum());
    int64_t currentStride = 1;
    for (int32_t dim = static_cast<int32_t>(shape.GetDimNum()) - 1; dim >= 0; --dim) {
        stride.SetStride(dim, currentStride);
        currentStride *= shape.GetDim(dim);
    }
    return ToStrideInfo(stride);
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

const char *ContractDtypeName(ContractDtype dtype)
{
    switch (dtype) {
        case ContractDtype::kBf16:
            return "bf16";
        case ContractDtype::kFloat:
            return "float";
        case ContractDtype::kInt32:
            return "int32";
        case ContractDtype::kFloat8E4M3fn:
            return "float8_e4m3fn";
        default:
            return "unknown";
    }
}

void FillInputDtype(gert::TilingContext *context, size_t index, TensorContractInfo &info)
{
    const auto desc = context->GetInputDesc(index);
    if (desc != nullptr) {
        info.dtype = ToContractDtype(desc->GetDataType());
    }
}

TensorContractInfo ToTensorInfo(gert::TilingContext *context, size_t index)
{
    TensorContractInfo info;
    info.shape = ToShapeInfo(context->GetInputShape(index));
    FillInputDtype(context, index, info);
    return info;
}

TensorContractInfo ToCacheTensorInfo(gert::TilingContext *context, size_t index)
{
    TensorContractInfo info;
    const auto storageShape = context->GetInputShape(index);
    if (storageShape == nullptr) {
        info.shape.present = false;
        info.stride.present = false;
        FillInputDtype(context, index, info);
        return info;
    }

    if (context->InputIsView(index)) {
        const auto inputStride = context->GetInputStride(index);
        if (inputStride != nullptr && inputStride->GetDimNum() != 0) {
            info.shape = ToShapeInfo(storageShape->GetShape());
            info.stride = ToStrideInfo(*inputStride);
            FillInputDtype(context, index, info);
            return info;
        }
    }

    const gert::Shape &shape = storageShape->GetStorageShape();
    info.shape = ToShapeInfo(shape);
    info.stride = MakeContiguousStrideInfo(shape);
    FillInputDtype(context, index, info);
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

void LogTensorInfo(const char *opName, const char *tensorName, const TensorContractInfo &info)
{
    OP_LOGD(opName,
            "QkvRmsNormRopeCacheWithKScale input tensor %s: dtype=%s(%u) shapePresent=%u shapeRank=%u "
            "shape={%lu,%lu,%lu,%lu} stridePresent=%u strideRank=%u stride={%lu,%lu,%lu,%lu}.",
            tensorName, ContractDtypeName(info.dtype), static_cast<uint32_t>(info.dtype),
            info.shape.present ? 1U : 0U, info.shape.rank, info.shape.dim0, info.shape.dim1, info.shape.dim2,
            info.shape.dim3, info.stride.present ? 1U : 0U, info.stride.rank, info.stride.stride0,
            info.stride.stride1, info.stride.stride2, info.stride.stride3);
}

void LogContractInput(const char *opName, const ContractInput &input, float epsilon)
{
    OP_LOGD(opName,
            "QkvRmsNormRopeCacheWithKScale parsed attrs/input summary: head_nums={%lu,%lu,%lu} epsilon=%f "
            "T=%lu batch=%lu headDim=%lu maxSeqLen=%lu blockNum=%lu blockSize=%lu aicNum=%lu.",
            input.numQHeads, input.numKHeads, input.numVHeads, epsilon, input.totalTokens, input.batch, input.headDim,
            input.maxSeqLen, input.blockNum, input.blockSize, input.aicNum);
    LogTensorInfo(opName, "qkv", input.qkv);
    LogTensorInfo(opName, "qGamma", input.qGamma);
    LogTensorInfo(opName, "kGamma", input.kGamma);
    LogTensorInfo(opName, "cosSin", input.cosSin);
    LogTensorInfo(opName, "slotMapping", input.slotMapping);
    LogTensorInfo(opName, "kCache", input.kCache);
    LogTensorInfo(opName, "vCache", input.vCache);
    LogTensorInfo(opName, "kScaleCache", input.kScaleCache);
    LogTensorInfo(opName, "cuSeqLen", input.cuSeqLen);
    LogTensorInfo(opName, "qRotation", input.qRotation);
    LogTensorInfo(opName, "kRotation", input.kRotation);
    LogTensorInfo(opName, "vScale", input.vScale);
}

size_t CalcOpWorkspaceSize(const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo)
{
    return static_cast<size_t>(compileInfo.opWorkspaceSize);
}

size_t CalcWorkspaceSize(size_t opWorkspaceSize)
{
    return opWorkspaceSize + kReservedWorkspaceSize;
}

ge::graphStatus SetWorkspace(gert::TilingContext *context, const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo,
                             size_t &workspaceSize)
{
    size_t *workspaces = context->GetWorkspaceSizes(kWorkspaceCount);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context->GetNodeName(), "get workspace failed."),
                return ge::GRAPH_FAILED);
    const size_t opWorkspaceSize = CalcOpWorkspaceSize(compileInfo);
    workspaceSize = CalcWorkspaceSize(opWorkspaceSize);
    workspaces[0] = workspaceSize;
    OP_LOGD(context->GetNodeName(),
            "QkvRmsNormRopeCacheWithKScale workspace: opWorkspaceSize=%zu reservedWorkspaceSize=%zu total=%zu.",
            opWorkspaceSize, kReservedWorkspaceSize, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

void LogTilingData(const char *opName, const QkvRmsNormRopeCacheWithKScaleTilingData &tilingData,
                   const ContractTiling &contractTiling, uint64_t tilingKey, uint64_t blockDim, size_t dataSize,
                   size_t workspaceSize)
{
    OP_LOGD(opName,
            "QkvRmsNormRopeCacheWithKScale final tilingData: totalTokens=%lu batch=%lu Nq=%lu Nk=%lu Nv=%lu "
            "headDim=%lu blockSize=%lu coreTokenTile=%lu coreGroupNum=%lu tokenTile=%lu epsilon=%f.",
            tilingData.totalTokens, tilingData.batch, tilingData.numQHeads, tilingData.numKHeads,
            tilingData.numVHeads, tilingData.headDim, tilingData.blockSize, tilingData.coreTokenTile,
            tilingData.coreGroupNum, tilingData.tokenTile, tilingData.epsilon);
    OP_LOGD(opName,
            "QkvRmsNormRopeCacheWithKScale final strides: kCache={%lu,%lu,%lu} vCache={%lu,%lu,%lu} "
            "kScaleCache={%lu,%lu,%lu}.",
            tilingData.kCacheStrideBlock, tilingData.kCacheStrideHead, tilingData.kCacheStrideToken,
            tilingData.vCacheStrideBlock, tilingData.vCacheStrideHead, tilingData.vCacheStrideToken,
            tilingData.kScaleCacheStrideBlock, tilingData.kScaleCacheStrideHead, tilingData.kScaleCacheStrideToken);
    OP_LOGD(opName,
            "QkvRmsNormRopeCacheWithKScale final launch: tilingKey=%lu blockDim=%lu tilingDataSize=%zu workspaceSize=%zu "
            "qkHeadTile=%lu vHeadTile=%lu rowTile=%lu rowTileAligned=%lu tokenTilePerAiv=%lu dimTile=%lu.",
            tilingKey, blockDim, dataSize, workspaceSize, contractTiling.qkHeadTile, contractTiling.vHeadTile,
            contractTiling.rowTile, contractTiling.rowTileAligned, contractTiling.tokenTilePerAiv,
            contractTiling.dimTile);
}

ge::graphStatus BuildContractInput(gert::TilingContext *context, ContractInput &input, float &epsilon)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto headNums = attrs->GetListInt(kHeadNumsAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNums);
    OP_CHECK_IF(headNums->GetSize() != 3, OP_LOGE(context->GetNodeName(), "head_nums must have exactly 3 values."),
                return ge::GRAPH_FAILED);
    const auto *headNumsData = headNums->GetData();
    for (size_t i = 0; i < 3; ++i) {
        OP_CHECK_IF(headNumsData[i] <= 0, OP_LOGE(context->GetNodeName(), "head_nums[%zu] must be positive.", i),
                    return ge::GRAPH_FAILED);
    }

    const auto qkv = ToTensorInfo(context, kQkvIndex);
    const auto cosSin = ToTensorInfo(context, kCosSinIndex);
    const auto kCache = ToCacheTensorInfo(context, kKCacheIndex);
    const auto cuSeqLen = ToTensorInfo(context, kCuSeqLenIndex);
    const auto &qkvShape = qkv.shape;
    const auto &cosSinShape = cosSin.shape;
    const auto &kCacheShape = kCache.shape;
    const auto &cuSeqLenShape = cuSeqLen.shape;
    if (qkvShape.rank >= 3) {
        input.totalTokens = qkvShape.dim0;
        input.headDim = qkvShape.dim2;
    }
    if (cuSeqLenShape.rank == 1 && cuSeqLenShape.dim0 > 0) {
        input.batch = cuSeqLenShape.dim0 - 1;
    }
    if (cosSinShape.rank >= 1) {
        input.maxSeqLen = cosSinShape.dim0;
    }
    if (kCacheShape.rank >= 4) {
        input.blockNum = kCacheShape.dim0;
        input.blockSize = kCacheShape.dim2;
    }

    input.numQHeads = static_cast<uint64_t>(headNumsData[0]);
    input.numKHeads = static_cast<uint64_t>(headNumsData[1]);
    input.numVHeads = static_cast<uint64_t>(headNumsData[2]);

    input.qkv = qkv;
    input.qGamma = ToTensorInfo(context, kQGammaIndex);
    input.kGamma = ToTensorInfo(context, kKGammaIndex);
    input.cosSin = cosSin;
    input.slotMapping = ToTensorInfo(context, kSlotMappingIndex);
    input.kCache = kCache;
    input.vCache = ToCacheTensorInfo(context, kVCacheIndex);
    input.kScaleCache = ToCacheTensorInfo(context, kKScaleCacheIndex);
    input.cuSeqLen = cuSeqLen;
    input.qRotation = ToOptionalTensorInfo(context, kQRotationIndex);
    input.kRotation = ToOptionalTensorInfo(context, kKRotationIndex);
    input.vScale = ToOptionalTensorInfo(context, kVScaleIndex);

    const float *epsilonAttr = attrs->GetFloat(kEpsilonAttrIndex);
    epsilon = epsilonAttr == nullptr ? kDefaultEpsilon : *epsilonAttr;
    return ge::GRAPH_SUCCESS;
}

void FillTilingData(const ContractInput &input, const ContractTiling &contractTiling, float epsilon,
                    QkvRmsNormRopeCacheWithKScaleTilingData &tilingData)
{
    tilingData.totalTokens = input.totalTokens;
    tilingData.batch = input.batch;
    tilingData.numQHeads = input.numQHeads;
    tilingData.numKHeads = input.numKHeads;
    tilingData.numVHeads = input.numVHeads;
    tilingData.headDim = input.headDim;
    tilingData.blockSize = input.blockSize;
    tilingData.coreTokenTile = contractTiling.coreTokenTile;
    tilingData.coreGroupNum = contractTiling.coreGroupNum;
    tilingData.kCacheStrideBlock = contractTiling.kCacheStrideBlock;
    tilingData.kCacheStrideHead = contractTiling.kCacheStrideHead;
    tilingData.kCacheStrideToken = contractTiling.kCacheStrideToken;
    tilingData.vCacheStrideBlock = contractTiling.vCacheStrideBlock;
    tilingData.vCacheStrideHead = contractTiling.vCacheStrideHead;
    tilingData.vCacheStrideToken = contractTiling.vCacheStrideToken;
    tilingData.kScaleCacheStrideBlock = contractTiling.kScaleCacheStrideBlock;
    tilingData.kScaleCacheStrideHead = contractTiling.kScaleCacheStrideHead;
    tilingData.kScaleCacheStrideToken = contractTiling.kScaleCacheStrideToken;
    tilingData.tokenTile = contractTiling.tokenTile;
    tilingData.epsilon = epsilon;
}

} // namespace

ge::graphStatus Tiling4QkvRmsNormRopeCacheWithKScale(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const char *opName = context->GetNodeName();
    OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale tiling start.");
    auto compileInfo = context->GetCompileInfo<QkvRmsNormRopeCacheWithKScaleCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    OP_LOGD(opName,
            "QkvRmsNormRopeCacheWithKScale compile info: aicNum=%u aivNum=%u ubSize=%lu l1Size=%lu l0cSize=%lu "
            "opWorkspaceSize=%lu.",
            compileInfo->aicNum, compileInfo->aivNum, compileInfo->ubSize, compileInfo->l1Size,
            compileInfo->l0cSize, compileInfo->opWorkspaceSize);
    OP_CHECK_IF(compileInfo->aicNum == 0,
                OP_LOGE(context->GetNodeName(), "aicNum in compile info must be greater than 0."),
                return ge::GRAPH_FAILED);

    ContractInput input;
    float epsilon = kDefaultEpsilon;
    OP_CHECK_IF(BuildContractInput(context, input, epsilon) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "failed to build QkvRmsNormRopeCacheWithKScale contract input."),
                return ge::GRAPH_FAILED);
    input.aicNum = compileInfo->aicNum;
    LogContractInput(opName, input, epsilon);

    ContractTiling contractTiling;
    const ContractStatus status = ValidateAndBuildTiling(input, contractTiling, opName);
    OP_CHECK_IF(status != ContractStatus::kSuccess,
                OP_LOGE(context->GetNodeName(), "QkvRmsNormRopeCacheWithKScale contract failed: %s",
                        ContractStatusName(status)),
                return ge::GRAPH_FAILED);

    QkvRmsNormRopeCacheWithKScaleTilingData tilingData;
    FillTilingData(input, contractTiling, epsilon, tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData()->GetData());
    const size_t tilingDataSize = sizeof(QkvRmsNormRopeCacheWithKScaleTilingData);
    OP_CHECK_IF(tilingDataSize % sizeof(uint64_t) != 0,
                OP_LOGE(context->GetNodeName(), "tiling data size[%zu] is not aligned to 8.", tilingDataSize),
                return ge::GRAPH_FAILED);
    const auto ret = memcpy_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
                              reinterpret_cast<const void *>(&tilingData), tilingDataSize);
    OP_CHECK_IF(ret != EOK,
                OP_LOGE(context->GetNodeName(), "copy QkvRmsNormRopeCacheWithKScale tiling data failed, ret=%d.", ret),
                return ge::GRAPH_FAILED);
    context->GetRawTilingData()->SetDataSize(tilingDataSize);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TPL_HEAD_DIM_D128);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(contractTiling.coreGroupNum);
    size_t workspaceSize = 0;
    OP_CHECK_IF(SetWorkspace(context, *compileInfo, workspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "set QkvRmsNormRopeCacheWithKScale workspace failed."),
                return ge::GRAPH_FAILED);
    LogTilingData(opName, tilingData, contractTiling, tilingKey, contractTiling.coreGroupNum, tilingDataSize,
                  workspaceSize);
    OP_LOGD(opName, "QkvRmsNormRopeCacheWithKScale tiling success.");
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
    compileInfo->opWorkspaceSize = static_cast<uint64_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    OP_LOGD(context->GetNodeName(),
            "QkvRmsNormRopeCacheWithKScale tiling prepare resources: aicNum=%u aivNum=%u ubSize=%lu l1Size=%lu "
            "l0cSize=%lu opWorkspaceSize=%lu.",
            compileInfo->aicNum, compileInfo->aivNum, compileInfo->ubSize, compileInfo->l1Size,
            compileInfo->l0cSize, compileInfo->opWorkspaceSize);
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
