/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gmm_csv_ge_parse_utils.h"
#include "gmm_csv_parse_utils.h"
#include "op_common/op_host/util/math_util.h"
#include "tiling_case_executor.h"
#include "platform/platform_infos_def.h"
#include "base/registry/op_impl_space_registry_v2.h"

#include "../../../op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.h"
#include "../../../op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling.h"
#include "../../../op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling_key.h"

namespace {
using QkvTiling = optiling::QkvRmsNormRopeCacheWithKScale::QkvRmsNormRopeCacheWithKScaleBaseTiling;
using ContractInput = optiling::QkvRmsNormRopeCacheWithKScale::ContractInput;
using TensorContractInfo = optiling::QkvRmsNormRopeCacheWithKScale::TensorContractInfo;
using TilingData = optiling::QkvRmsNormRopeCacheWithKScaleTilingData;

constexpr uint64_t TEST_BLOCK_SIZE = 128U;
constexpr uint64_t TEST_OP_WORKSPACE_SIZE = 4096U;
constexpr uint64_t RESERVED_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t TEST_LAYOUT_NTD = 0U;
constexpr uint64_t TEST_LAYOUT_TND = 1U;
constexpr uint64_t INPUT_LAYOUT_TILING_KEY_SHIFT = 8U;

struct TilingRunOptions {
    uint64_t aicNum = 32U;
    bool aivNumPresent = false;
    uint64_t aivNum = 0U;
    bool headNumsPresent = false;
    std::vector<int64_t> headNums;
    std::string socVersion = "Ascend950";
};

struct TensorSpec {
    gert::StorageShape shape;
    ge::DataType dtype = ge::DT_FLOAT;
    ge::Format format = ge::FORMAT_ND;
    bool isView = false;
    gert::Stride stride;
};

bool ExecuteTilingForInput(const ContractInput &input, TilingInfo &tilingInfo, uint64_t aicNum = 32);
bool ExecuteTilingForInput(const ContractInput &input, TilingInfo &tilingInfo, const TilingRunOptions &options);

gert::StorageShape MakeStorageShape(const gert::Shape &shape)
{
    gert::StorageShape storageShape;
    for (uint64_t i = 0; i < shape.GetDimNum(); ++i) {
        storageShape.MutableOriginShape().AppendDim(shape.GetDim(i));
        storageShape.MutableStorageShape().AppendDim(shape.GetDim(i));
    }
    return storageShape;
}

template <typename... Args>
void SetShape(TensorContractInfo &tensor, Args... dims)
{
    tensor.shape = gert::Shape();
    (tensor.shape.AppendDim(static_cast<int64_t>(dims)), ...);
    tensor.shapePresent = true;
}

void SetShape(TensorContractInfo &tensor, const std::vector<int64_t> &dims)
{
    tensor.shape = gert::Shape();
    for (const auto dim : dims) {
        tensor.shape.AppendDim(dim);
    }
    tensor.shapePresent = true;
}

template <typename... Args>
gert::Stride Stride(Args... strides)
{
    gert::Stride stride;
    stride.SetDimNum(sizeof...(strides));
    int64_t index = 0;
    ((stride.SetStride(index, static_cast<int64_t>(strides)), ++index), ...);
    return stride;
}

void SetStride(TensorContractInfo &tensor, const std::vector<int64_t> &strides)
{
    tensor.stride = gert::Stride();
    tensor.stride.SetDimNum(strides.size());
    for (uint64_t i = 0; i < strides.size(); ++i) {
        tensor.stride.SetStride(static_cast<int64_t>(i), strides[i]);
    }
    tensor.stridePresent = true;
}

void RefreshShapes(ContractInput &input)
{
    const uint64_t qkvN = input.numQHeads + input.numKHeads + input.numVHeads;
    if (input.inputLayout == TEST_LAYOUT_TND) {
        SetShape(input.qkv, input.totalTokens, qkvN, input.headDim);
    } else {
        SetShape(input.qkv, qkvN, input.totalTokens, input.headDim);
    }
    SetShape(input.qGamma, input.headDim);
    SetShape(input.kGamma, input.headDim);
    SetShape(input.cosSin, input.maxSeqLen, input.headDim);
    SetShape(input.slotMapping, input.totalTokens);
    SetShape(input.kCache, input.blockNum, input.numKHeads, input.blockSize, input.headDim);
    input.kCache.stride = Stride(input.numKHeads * input.blockSize * input.headDim,
        input.blockSize * input.headDim, input.headDim, 1);
    input.kCache.stridePresent = true;
    SetShape(input.vCache, input.blockNum, input.numVHeads, input.blockSize, input.headDim);
    input.vCache.stride = Stride(input.numVHeads * input.blockSize * input.headDim,
        input.blockSize * input.headDim, input.headDim, 1);
    input.vCache.stridePresent = true;
    SetShape(input.kScaleCache, input.blockNum, input.numKHeads, input.blockSize, 1);
    input.kScaleCache.stride = Stride(input.numKHeads * input.blockSize, input.blockSize, 1, 1);
    input.kScaleCache.stridePresent = true;
    SetShape(input.cuSeqLen, static_cast<uint64_t>(input.batch) + 1);
    SetShape(input.actualSeqLens, static_cast<uint64_t>(input.batch));
    SetShape(input.qRotation, input.headDim, input.headDim);
    SetShape(input.kRotation, input.headDim, input.headDim);
    SetShape(input.vScale, input.numVHeads);
}

ContractInput BuildInput(uint64_t totalTokens = 128, uint64_t numQHeads = 16, uint64_t numKHeads = 2,
    uint64_t numVHeads = 2, uint64_t headDim = 128)
{
    ContractInput input;
    input.totalTokens = totalTokens;
    input.batch = 1;
    input.numQHeads = numQHeads;
    input.numKHeads = numKHeads;
    input.numVHeads = numVHeads;
    input.headDim = headDim;
    input.maxSeqLen = 256;
    input.blockNum = 8;
    input.blockSize = 128;

    input.qkv.dtype = ge::DT_BF16;
    input.qGamma.dtype = ge::DT_FLOAT;
    input.kGamma.dtype = ge::DT_FLOAT;
    input.cosSin.dtype = ge::DT_FLOAT;
    input.slotMapping.dtype = ge::DT_INT32;
    input.kCache.dtype = ge::DT_FLOAT8_E4M3FN;
    input.vCache.dtype = ge::DT_FLOAT8_E4M3FN;
    input.kScaleCache.dtype = ge::DT_FLOAT;
    input.cuSeqLen.dtype = ge::DT_INT32;
    input.actualSeqLens.dtype = ge::DT_INT32;
    input.qRotation.dtype = ge::DT_BF16;
    input.kRotation.dtype = ge::DT_BF16;
    input.vScale.dtype = ge::DT_FLOAT;
    RefreshShapes(input);
    return input;
}

struct ConcurrentTilingCase {
    uint64_t totalTokens;
    uint64_t numQHeads;
    uint64_t numKHeads;
    uint64_t numVHeads;
    uint64_t headDim;
    uint64_t aicNum;
    ge::graphStatus status;
    uint64_t tokenTile;
    uint64_t coreTokenTile;
    uint64_t coreGroupNum;
    uint64_t inputLayout = TEST_LAYOUT_NTD;
};

struct ConcurrentTilingResult {
    bool ok = true;
    uint64_t failedCase = 0;
    uint32_t failedIteration = 0;
    bool success = false;
    uint64_t tokenTile = 0;
    uint64_t coreTokenTile = 0;
    uint64_t coreGroupNum = 0;
    uint64_t blockNum = 0;
};

optiling::QkvRmsNormRopeCacheWithKScaleCompileInfo BuildCompileInfo(const TilingRunOptions &options)
{
    optiling::QkvRmsNormRopeCacheWithKScaleCompileInfo compileInfo;
    compileInfo.aicNum = static_cast<uint32_t>(options.aicNum);
    compileInfo.aivNum = static_cast<uint32_t>(
        options.aivNumPresent ? options.aivNum : options.aicNum * QkvTiling::AIV_PER_AIC);
    compileInfo.ubSize = 262144U;
    compileInfo.l1Size = 524288U;
    compileInfo.l0cSize = 131072U;
    compileInfo.opWorkspaceSize = TEST_OP_WORKSPACE_SIZE;
    return compileInfo;
}

TensorSpec MakeTensorSpec(const TensorContractInfo &tensor)
{
    TensorSpec spec;
    spec.shape = MakeStorageShape(tensor.shape);
    spec.dtype = tensor.dtype;
    spec.format = ge::FORMAT_ND;
    spec.isView = tensor.stridePresent;
    spec.stride = tensor.stride;
    return spec;
}

TensorSpec MakeEmptyTensorSpec(ge::DataType dtype)
{
    TensorSpec spec;
    spec.shape = gert::StorageShape();
    spec.dtype = dtype;
    spec.format = ge::FORMAT_ND;
    return spec;
}

std::vector<TensorSpec> BuildTilingInputSpecs(const ContractInput &input)
{
    std::vector<TensorSpec> inputs;
    inputs.reserve(13U);
    inputs.push_back(MakeTensorSpec(input.qkv));
    inputs.push_back(MakeTensorSpec(input.qGamma));
    inputs.push_back(MakeTensorSpec(input.kGamma));
    inputs.push_back(MakeTensorSpec(input.cosSin));
    inputs.push_back(MakeTensorSpec(input.slotMapping));
    inputs.push_back(MakeTensorSpec(input.kCache));
    inputs.push_back(MakeTensorSpec(input.vCache));
    inputs.push_back(MakeTensorSpec(input.kScaleCache));
    inputs.push_back(MakeTensorSpec(input.cuSeqLen));
    inputs.push_back(MakeTensorSpec(input.actualSeqLens));
    inputs.push_back(input.qRotation.shapePresent ? MakeTensorSpec(input.qRotation) : MakeEmptyTensorSpec(ge::DT_UNDEFINED));
    inputs.push_back(input.kRotation.shapePresent ? MakeTensorSpec(input.kRotation) : MakeEmptyTensorSpec(ge::DT_UNDEFINED));
    inputs.push_back(input.vScale.shapePresent ? MakeTensorSpec(input.vScale) : MakeEmptyTensorSpec(ge::DT_UNDEFINED));
    return inputs;
}

std::vector<TensorSpec> BuildTilingOutputSpecs(const ContractInput &input)
{
    const bool isTnd = input.inputLayout == TEST_LAYOUT_TND;
    std::vector<TensorSpec> outputs;
    outputs.reserve(5U);
    gert::Shape qOutShape;
    qOutShape.AppendDim(static_cast<int64_t>(isTnd ? input.totalTokens : input.numQHeads));
    qOutShape.AppendDim(static_cast<int64_t>(isTnd ? input.numQHeads : input.totalTokens));
    qOutShape.AppendDim(static_cast<int64_t>(input.headDim));
    outputs.push_back({MakeStorageShape(qOutShape), ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, false, gert::Stride()});
    gert::Shape scaleShape;
    scaleShape.AppendDim(static_cast<int64_t>(isTnd ? input.totalTokens : input.numQHeads));
    scaleShape.AppendDim(static_cast<int64_t>(isTnd ? input.numQHeads : input.totalTokens));
    outputs.push_back({MakeStorageShape(scaleShape), ge::DT_FLOAT, ge::FORMAT_ND, false, gert::Stride()});
    outputs.push_back(MakeTensorSpec(input.kCache));
    outputs.push_back(MakeTensorSpec(input.vCache));
    outputs.push_back(MakeTensorSpec(input.kScaleCache));
    return outputs;
}

std::vector<gert::TilingContextPara::OpAttr> BuildTilingAttrs(
    const ContractInput &input, const TilingRunOptions &options)
{
    std::vector<int64_t> headNums;
    if (options.headNumsPresent) {
        headNums = options.headNums;
    } else {
        headNums.reserve(3U);
        headNums.push_back(static_cast<int64_t>(input.numQHeads));
        headNums.push_back(static_cast<int64_t>(input.numKHeads));
        headNums.push_back(static_cast<int64_t>(input.numVHeads));
    }

    std::vector<gert::TilingContextPara::OpAttr> attrs;
    attrs.reserve(3U);
    attrs.push_back(gert::TilingContextPara::OpAttr(
        "head_nums", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(headNums)));
    attrs.push_back(gert::TilingContextPara::OpAttr("inputLayout",
        Ops::Transformer::AnyValue::CreateFrom<std::string>(input.inputLayout == TEST_LAYOUT_TND ? "TND" : "NTD")));
    attrs.push_back(
        gert::TilingContextPara::OpAttr("epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)));
    return attrs;
}

std::unique_ptr<gert::Tensor> MakeTensor(const TensorSpec &spec)
{
    return std::make_unique<gert::Tensor>(spec.shape, gert::StorageFormat(spec.format, spec.format,
        gert::ExpandDimsType()), gert::TensorPlacement::kOnHost, spec.dtype, nullptr);
}

std::unique_ptr<gert::TensorV2> MakeViewTensor(const TensorSpec &spec)
{
    return std::make_unique<gert::TensorV2>(spec.shape, gert::StorageFormat(spec.format, spec.format,
        gert::ExpandDimsType()), gert::TensorPlacement::kOnHost, spec.dtype, nullptr, nullptr, spec.stride, 0);
}

void BuildTensorLists(const std::vector<TensorSpec> &specs, std::vector<uint32_t> &irInstances,
    std::vector<gert::Tensor *> &tensors, std::vector<std::unique_ptr<gert::Tensor>> &tensorKeepAlive,
    std::vector<std::unique_ptr<gert::TensorV2>> &viewTensorKeepAlive)
{
    irInstances.reserve(specs.size());
    tensors.reserve(specs.size());
    for (const auto &spec : specs) {
        if (spec.shape.GetStorageShape().GetDimNum() == 0) {
            irInstances.push_back(0U);
            continue;
        }
        irInstances.push_back(1U);
        if (spec.isView) {
            auto tensor = MakeViewTensor(spec);
            tensors.push_back(reinterpret_cast<gert::Tensor *>(tensor.get()));
            viewTensorKeepAlive.push_back(std::move(tensor));
        } else {
            auto tensor = MakeTensor(spec);
            tensors.push_back(tensor.get());
            tensorKeepAlive.push_back(std::move(tensor));
        }
    }
}

void AppendAttrs(gert::TilingContextFaker &contextFaker, const std::vector<gert::TilingContextPara::OpAttr> &attrs)
{
    for (const auto &attrInfo : attrs) {
        switch (attrInfo.attr_.type_) {
            case Ops::Transformer::AnyValue::ValueType::VT_BOOL:
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<bool *>(attrInfo.attr_.valuePtr_.get()));
                break;
            case Ops::Transformer::AnyValue::ValueType::VT_INT:
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<int64_t *>(attrInfo.attr_.valuePtr_.get()));
                break;
            case Ops::Transformer::AnyValue::ValueType::VT_FLOAT:
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<float *>(attrInfo.attr_.valuePtr_.get()));
                break;
            case Ops::Transformer::AnyValue::ValueType::VT_STRING:
                contextFaker.Attr(attrInfo.attrName_,
                    ge::AscendString(reinterpret_cast<std::string *>(attrInfo.attr_.valuePtr_.get())->c_str()));
                break;
            case Ops::Transformer::AnyValue::ValueType::VT_LIST_INT:
                contextFaker.Attr(attrInfo.attrName_,
                    *reinterpret_cast<std::vector<int64_t> *>(attrInfo.attr_.valuePtr_.get()));
                break;
            default:
                throw std::invalid_argument("unsupported qkv tiling attr type");
        }
    }
}

void InitPlatformInfo(gert::TilingContext *context, const TilingRunOptions &options)
{
    const std::string npuArch = options.socVersion == "Ascend950" ? "3510" : "2201";
    std::map<std::string, std::string> socInfos = {
        {"ai_core_cnt", std::to_string(options.aicNum)},
        {"cube_core_cnt", std::to_string(options.aicNum)},
        {"vector_core_cnt", std::to_string(options.aicNum * QkvTiling::AIV_PER_AIC)},
        {"core_type_list", "CubeCore,VectorCore"},
        {"l2_size", "33554432"},
    };
    std::map<std::string, std::string> aicoreSpec = {
        {"ub_size", "262144"},
        {"l1_size", "524288"},
        {"l0_a_size", "65536"},
        {"l0_b_size", "65536"},
        {"l0_c_size", "131072"},
        {"bt_size", "0"},
        {"load3d_constraints", "1"},
        {"cube_freq", "cube_freq"},
    };
    std::map<std::string, std::string> intrinsics = {
        {"Intrinsic_data_move_l12ub", "float16"},
        {"Intrinsic_data_move_l0c2ub", "float16"},
    };
    std::map<std::string, std::string> versions = {
        {"NpuArch", npuArch},
        {"Short_SoC_version", options.socVersion},
    };
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    context->GetPlatformInfo()->SetPlatformRes("version", versions);
}

bool ExecuteQkvTiling(const ContractInput &input, const TilingRunOptions &options,
    optiling::QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo, TilingInfo &tilingInfo)
{
    gert::TilingContextFaker contextFaker;
    auto inputSpecs = BuildTilingInputSpecs(input);
    auto outputSpecs = BuildTilingOutputSpecs(input);
    std::vector<uint32_t> inputIrInstances;
    std::vector<uint32_t> outputIrInstances;
    std::vector<gert::Tensor *> inputTensors;
    std::vector<gert::Tensor *> outputTensors;
    std::vector<std::unique_ptr<gert::Tensor>> inputTensorKeepAlive;
    std::vector<std::unique_ptr<gert::Tensor>> outputTensorKeepAlive;
    std::vector<std::unique_ptr<gert::TensorV2>> inputViewTensorKeepAlive;
    std::vector<std::unique_ptr<gert::TensorV2>> outputViewTensorKeepAlive;
    BuildTensorLists(inputSpecs, inputIrInstances, inputTensors, inputTensorKeepAlive, inputViewTensorKeepAlive);
    BuildTensorLists(outputSpecs, outputIrInstances, outputTensors, outputTensorKeepAlive, outputViewTensorKeepAlive);
    contextFaker.IrInstanceNum(inputIrInstances, outputIrInstances)
        .InputTensors(inputTensors)
        .OutputTensors(outputTensors);
    AppendAttrs(contextFaker, BuildTilingAttrs(input, options));

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto *workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());
    auto contextHolder = contextFaker.SetOpType("QkvRmsNormRopeCacheWithKScale")
                             .CompileInfo(&compileInfo)
                             .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                             .TilingData(tilingData.get())
                             .Workspace(workspace)
                             .Build();
    auto *tilingContext = contextHolder.GetContext();
    InitPlatformInfo(tilingContext, options);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto tilingFunc = spaceRegistry->GetOpImpl("QkvRmsNormRopeCacheWithKScale")->tiling;
    if (tilingFunc(tilingContext) != ge::GRAPH_SUCCESS) {
        return false;
    }

    tilingInfo.tilingKey = tilingContext->GetTilingKey();
    tilingInfo.blockNum = tilingContext->GetBlockDim();
    size_t workspaceCount = tilingContext->GetWorkspaceNum();
    if (workspaceCount > 0U) {
        auto workspaceSizes = tilingContext->GetWorkspaceSizes(workspaceCount);
        for (size_t i = 0U; i < workspaceCount; ++i) {
            tilingInfo.workspaceSizes.push_back(workspaceSizes[i]);
        }
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    tilingInfo.tilingData = std::make_unique<uint8_t[]>(rawTilingData->GetDataSize());
    tilingInfo.tilingDataSize = rawTilingData->GetDataSize();
    std::memcpy(tilingInfo.tilingData.get(), rawTilingData->GetData(), rawTilingData->GetDataSize());
    return true;
}

bool ExecuteTilingForInput(const ContractInput &input, TilingInfo &tilingInfo, uint64_t aicNum)
{
    TilingRunOptions options;
    options.aicNum = aicNum;
    return ExecuteTilingForInput(input, tilingInfo, options);
}

bool ExecuteTilingForInput(const ContractInput &input, TilingInfo &tilingInfo, const TilingRunOptions &options)
{
    auto compileInfo = BuildCompileInfo(options);
    return ExecuteQkvTiling(input, options, compileInfo, tilingInfo);
}

bool MatchesConcurrentTilingCase(const ConcurrentTilingCase &item, bool success, const TilingInfo &tilingInfo)
{
    if (success != (item.status == ge::GRAPH_SUCCESS)) {
        return false;
    }
    if (!success) {
        return true;
    }
    if (tilingInfo.tilingDataSize < sizeof(optiling::QkvRmsNormRopeCacheWithKScaleTilingData) ||
        tilingInfo.tilingData == nullptr || tilingInfo.workspaceSizes.empty()) {
        return false;
    }

    const auto *tilingData =
        reinterpret_cast<const optiling::QkvRmsNormRopeCacheWithKScaleTilingData *>(tilingInfo.tilingData.get());
    const int64_t expectedWorkspace = static_cast<int64_t>(RESERVED_WORKSPACE_SIZE + TEST_OP_WORKSPACE_SIZE);
    return tilingInfo.blockNum == item.coreGroupNum && tilingInfo.workspaceSizes[0] == expectedWorkspace &&
           tilingData->totalTokens == item.totalTokens && tilingData->qHeadNum == item.numQHeads &&
           tilingData->kvHeadNum == item.numKHeads &&
           tilingData->headDim == item.headDim && tilingData->blockSize == TEST_BLOCK_SIZE &&
           tilingData->tokenTile == item.tokenTile && tilingData->coreTokenTile == item.coreTokenTile &&
           tilingData->coreGroupNum == item.coreGroupNum;
}

uint64_t ParseU64(const std::string &value, uint64_t defaultValue = 0)
{
    const std::string trimmed = ops::ut::Trim(value);
    return trimmed.empty() ? defaultValue : static_cast<uint64_t>(std::stoull(trimmed));
}

struct CsvTilingCase {
    std::string caseName;
    std::string updates;
    uint64_t totalTokens = 128;
    uint64_t numQHeads = 16;
    uint64_t numKHeads = 2;
    uint64_t numVHeads = 2;
    uint64_t headDim = 128;
    uint64_t aicNum = 32;
    uint64_t inputLayout = TEST_LAYOUT_NTD;
    ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS;
    uint64_t tokenTile = 0;
    uint64_t tokenTilePerAiv = 0;
    uint64_t rowTile = 0;
    uint64_t rowTileAligned = 0;
    uint64_t coreTokenTile = 0;
    uint64_t coreGroupNum = 0;
    uint64_t kvStrideBlock = 0;
    uint64_t kvStrideHead = 0;
    uint64_t kvStrideToken = 0;
    uint64_t kScaleStrideBlock = 0;
    uint64_t kScaleStrideHead = 0;
    uint64_t kScaleStrideToken = 0;
    std::string checkSpec;
    std::map<std::string, std::string> spec;

    bool HasSpec(const std::string &key) const
    {
        return spec.find(key) != spec.end();
    }

    uint64_t SpecU64(const std::string &key, uint64_t defaultValue = 0) const
    {
        const auto it = spec.find(key);
        return it == spec.end() ? defaultValue : ParseU64(it->second);
    }
};

std::string CsvColumn(const std::vector<std::string> &cols, uint64_t index)
{
    return index < cols.size() ? ops::ut::Trim(cols[index]) : std::string();
}

uint64_t ParseLayout(const std::string &value)
{
    const std::string layout = ops::ut::ToLower(ops::ut::Trim(value));
    return layout == "tnd" ? TEST_LAYOUT_TND : TEST_LAYOUT_NTD;
}

ge::graphStatus ParseGraphStatus(const std::string &value)
{
    return ops::ut::ToLower(ops::ut::Trim(value)) == "success" ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

TensorContractInfo *FindTensor(ContractInput &input, const std::string &name)
{
    if (name == "qkv") {
        return &input.qkv;
    }
    if (name == "qGamma") {
        return &input.qGamma;
    }
    if (name == "kGamma") {
        return &input.kGamma;
    }
    if (name == "cosSin") {
        return &input.cosSin;
    }
    if (name == "slotMapping") {
        return &input.slotMapping;
    }
    if (name == "kCache") {
        return &input.kCache;
    }
    if (name == "vCache") {
        return &input.vCache;
    }
    if (name == "kScaleCache") {
        return &input.kScaleCache;
    }
    if (name == "cuSeqLen") {
        return &input.cuSeqLen;
    }
    if (name == "actualSeqLens") {
        return &input.actualSeqLens;
    }
    if (name == "qRotation") {
        return &input.qRotation;
    }
    if (name == "kRotation") {
        return &input.kRotation;
    }
    if (name == "vScale") {
        return &input.vScale;
    }
    return nullptr;
}

std::map<std::string, std::string> ParseSpec(const std::string &spec)
{
    std::map<std::string, std::string> result;
    const std::string trimmedSpec = ops::ut::Trim(spec);
    if (trimmedSpec.empty() || ops::ut::ToLower(trimmedSpec) == "none") {
        return result;
    }

    std::vector<std::string> items;
    ops::ut::SplitStr2Vec(trimmedSpec, ";", items);
    for (const auto &item : items) {
        std::vector<std::string> keyValue;
        ops::ut::SplitStr2Vec(item, "=", keyValue);
        if (keyValue.size() >= 2U) {
            result[ops::ut::Trim(keyValue[0])] = ops::ut::Trim(keyValue[1]);
        }
    }
    return result;
}

void ApplyTensorUpdate(ContractInput &input, const std::string &update)
{
    const std::string trimmed = ops::ut::Trim(update);
    if (trimmed.empty() || ops::ut::ToLower(trimmed) == "none") {
        return;
    }
    const auto equalPos = trimmed.find('=');
    const auto dotPos = trimmed.find('.');
    if (equalPos == std::string::npos || dotPos == std::string::npos || dotPos > equalPos) {
        throw std::invalid_argument("bad tensor update: " + trimmed);
    }
    const std::string tensorName = ops::ut::Trim(trimmed.substr(0, dotPos));
    const std::string field = ops::ut::Trim(trimmed.substr(dotPos + 1, equalPos - dotPos - 1));
    const std::string value = ops::ut::Trim(trimmed.substr(equalPos + 1));
    TensorContractInfo *tensor = FindTensor(input, tensorName);
    if (tensor == nullptr) {
        throw std::invalid_argument("unknown tensor in update: " + trimmed);
    }
    if (field == "shape") {
        SetShape(*tensor, ops::ut::ParseI64List(value));
    } else if (field == "dtype") {
        tensor->dtype = ops::ut::ParseGeDtype(value);
    } else if (field == "present") {
        tensor->shapePresent = ops::ut::ParseBool(value);
    } else if (field == "stride") {
        SetStride(*tensor, ops::ut::ParseI64List(value));
    } else if (field == "strideDimNum") {
        tensor->stride.SetDimNum(ParseU64(value));
        tensor->stridePresent = true;
    } else {
        throw std::invalid_argument("unknown tensor update field: " + trimmed);
    }
}

void ApplyRunOptionUpdate(TilingRunOptions &options, const std::string &field, const std::string &value)
{
    if (field == "aicNum") {
        options.aicNum = ParseU64(value);
    } else if (field == "aivNum") {
        options.aivNum = ParseU64(value);
        options.aivNumPresent = true;
    } else if (field == "headNums") {
        options.headNums = ops::ut::ParseI64List(value);
        options.headNumsPresent = true;
    } else if (field == "socVersion") {
        options.socVersion = value;
    } else {
        throw std::invalid_argument("unknown run option field: " + field);
    }
}

void ApplyCsvUpdate(ContractInput &input, TilingRunOptions &options, const std::string &update)
{
    const std::string trimmed = ops::ut::Trim(update);
    if (trimmed.empty() || ops::ut::ToLower(trimmed) == "none") {
        return;
    }
    const auto equalPos = trimmed.find('=');
    const auto dotPos = trimmed.find('.');
    if (equalPos == std::string::npos) {
        throw std::invalid_argument("bad csv update: " + trimmed);
    }
    if (dotPos == std::string::npos || dotPos > equalPos) {
        ApplyRunOptionUpdate(options, ops::ut::Trim(trimmed.substr(0, equalPos)),
            ops::ut::Trim(trimmed.substr(equalPos + 1)));
        return;
    }
    ApplyTensorUpdate(input, trimmed);
}

void ApplyCsvUpdates(ContractInput &input, TilingRunOptions &options, const std::string &updatesText)
{
    std::vector<std::string> updates;
    ops::ut::SplitStr2Vec(updatesText, ";", updates);
    for (const auto &update : updates) {
        ApplyCsvUpdate(input, options, update);
    }
}

CsvTilingCase ParseCsvCase(const std::vector<std::string> &cols)
{
    CsvTilingCase item;
    item.caseName = CsvColumn(cols, 0);
    item.updates = CsvColumn(cols, 1);
    item.totalTokens = ParseU64(CsvColumn(cols, 2), 128U);
    item.numQHeads = ParseU64(CsvColumn(cols, 3), 16U);
    item.numKHeads = ParseU64(CsvColumn(cols, 4), 2U);
    item.numVHeads = ParseU64(CsvColumn(cols, 5), 2U);
    item.headDim = ParseU64(CsvColumn(cols, 6), 128U);
    item.aicNum = ParseU64(CsvColumn(cols, 7), 32U);
    item.inputLayout = ParseLayout(CsvColumn(cols, 8));
    item.expectedStatus = ParseGraphStatus(CsvColumn(cols, 9));
    item.tokenTile = ParseU64(CsvColumn(cols, 10));
    item.tokenTilePerAiv = ParseU64(CsvColumn(cols, 11));
    item.rowTile = ParseU64(CsvColumn(cols, 12));
    item.rowTileAligned = ParseU64(CsvColumn(cols, 13));
    item.coreTokenTile = ParseU64(CsvColumn(cols, 14));
    item.coreGroupNum = ParseU64(CsvColumn(cols, 15));
    item.kvStrideBlock = ParseU64(CsvColumn(cols, 16));
    item.kvStrideHead = ParseU64(CsvColumn(cols, 17));
    item.kvStrideToken = ParseU64(CsvColumn(cols, 18));
    item.kScaleStrideBlock = ParseU64(CsvColumn(cols, 19));
    item.kScaleStrideHead = ParseU64(CsvColumn(cols, 20));
    item.kScaleStrideToken = ParseU64(CsvColumn(cols, 21));
    item.checkSpec = CsvColumn(cols, 22);
    item.spec = ParseSpec(item.checkSpec);
    return item;
}

std::vector<CsvTilingCase> LoadCsvTilingCases(const std::string &csvFilePath)
{
    std::ifstream in(csvFilePath);
    EXPECT_TRUE(in.is_open()) << "Failed to open CSV file: " << csvFilePath;

    std::vector<CsvTilingCase> cases;
    std::string line;
    uint64_t lineNo = 0U;
    while (std::getline(in, line)) {
        ++lineNo;
        const std::string trimmed = ops::ut::Trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        std::vector<std::string> cols;
        ops::ut::SplitStr2Vec(trimmed, ",", cols);
        if (!cols.empty() && ops::ut::Trim(cols[0]) == "caseName") {
            continue;
        }
        if (cols.size() != 23U) {
            ADD_FAILURE() << "Bad csv row column count in " << csvFilePath << ": line=" << lineNo;
            continue;
        }
        try {
            cases.push_back(ParseCsvCase(cols));
        } catch (const std::exception &error) {
            ADD_FAILURE() << ops::ut::BuildCsvParseErrorMessage(csvFilePath, lineNo, CsvColumn(cols, 0), error);
        }
    }
    EXPECT_FALSE(cases.empty()) << "No valid cases parsed from CSV: " << csvFilePath;
    return cases;
}

const std::vector<CsvTilingCase> &GetCsvTilingCases()
{
    static const auto cases = LoadCsvTilingCases(ops::ut::ResolveCsvPath(
        "test_qkv_rms_norm_rope_cache_with_k_scale_tiling.csv",
        "posembedding/qkv_rms_norm_rope_cache_with_k_scale/tests/ut/op_host", __FILE__));
    return cases;
}

std::string MakeCsvTilingCaseName(const testing::TestParamInfo<CsvTilingCase> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

uint64_t ExpectTileFromSpec(const CsvTilingCase &item, const ContractInput &input, const TilingData &tiling,
    const std::string &prefix)
{
    const uint64_t tokenOffset = item.SpecU64(prefix + "TokenOffset");
    const uint64_t inputSize = item.SpecU64(prefix + "InputSize");
    uint64_t tokenSize = 0U;
    if (tokenOffset < input.totalTokens && inputSize != 0U) {
        tokenSize = std::min(inputSize, std::min(tiling.tokenTile, input.totalTokens - tokenOffset));
    }

    EXPECT_EQ(tokenOffset, item.SpecU64(prefix + "ExpectOffset")) << "caseName=" << item.caseName;
    EXPECT_EQ(tokenSize, item.SpecU64(prefix + "ExpectSize")) << "caseName=" << item.caseName;
    EXPECT_EQ(tokenSize * (tiling.qHeadNum + tiling.kvHeadNum), item.SpecU64(prefix + "ExpectRowSize"))
        << "caseName=" << item.caseName;

    return tokenSize;
}

void ExpectTilingKey(const CsvTilingCase &item, const ContractInput &input, const TilingInfo &tilingInfo)
{
    const uint64_t expectedLayout = input.inputLayout == TEST_LAYOUT_TND ? QKV_RRC_LAYOUT_TND : QKV_RRC_LAYOUT_NTD;
    EXPECT_EQ(static_cast<uint64_t>(tilingInfo.tilingKey), expectedLayout << INPUT_LAYOUT_TILING_KEY_SHIFT)
        << "caseName=" << item.caseName;
}

void ExpectCoreRangeFromSpec(const CsvTilingCase &item, const ContractInput &input, const TilingData &tiling)
{
    const uint64_t coreIndex = item.SpecU64("coreIndex");
    const uint64_t rangeBegin = coreIndex * tiling.coreTokenTile;
    const uint64_t rangeEnd = std::min(input.totalTokens, rangeBegin + tiling.coreTokenTile);
    EXPECT_LT(coreIndex, tiling.coreGroupNum) << "caseName=" << item.caseName;
    EXPECT_EQ(rangeBegin, item.SpecU64("rangeBegin")) << "caseName=" << item.caseName;
    EXPECT_EQ(rangeEnd, item.SpecU64("rangeEnd")) << "caseName=" << item.caseName;
}

void ExpectCoreCoverFromSpec(const CsvTilingCase &item, const ContractInput &input, const TilingData &tiling)
{
    const uint64_t lastCoreIndex = item.SpecU64("lastCoreIndex");
    const uint64_t lastBegin = lastCoreIndex * tiling.coreTokenTile;
    const uint64_t lastEnd = std::min(input.totalTokens, lastBegin + tiling.coreTokenTile);
    EXPECT_LT(lastCoreIndex, tiling.coreGroupNum) << "caseName=" << item.caseName;
    EXPECT_EQ(lastBegin, item.SpecU64("lastBegin")) << "caseName=" << item.caseName;
    EXPECT_EQ(lastEnd, item.SpecU64("lastEnd")) << "caseName=" << item.caseName;
    EXPECT_EQ(lastEnd, item.SpecU64("coveredEnd", input.totalTokens)) << "caseName=" << item.caseName;
}

void ExpectCoreLocalTilesFromSpec(const CsvTilingCase &item, const ContractInput &input, const TilingData &tiling)
{
    ExpectCoreRangeFromSpec(item, input, tiling);
    const uint64_t tile0TokenSize = ExpectTileFromSpec(item, input, tiling, "tile0");
    if (!item.HasSpec("tile1TokenOffset")) {
        return;
    }

    const uint64_t tile1TokenSize = ExpectTileFromSpec(item, input, tiling, "tile1");
    EXPECT_EQ(tile0TokenSize + tile1TokenSize, item.SpecU64("rangeEnd") - item.SpecU64("rangeBegin"))
        << "caseName=" << item.caseName;
}

bool IsKnownSpecKey(const std::string &key)
{
    return key == "tileTokenOffset" || key == "tileInputSize" || key == "tileExpectOffset" ||
           key == "tileExpectSize" || key == "tileExpectRowSize" || key == "coreIndex" ||
           key == "rangeBegin" || key == "rangeEnd" || key == "lastCoreIndex" || key == "lastBegin" ||
           key == "lastEnd" || key == "coveredEnd" || key == "tile0TokenOffset" || key == "tile0InputSize" ||
           key == "tile0ExpectOffset" || key == "tile0ExpectSize" || key == "tile0ExpectRowSize" ||
           key == "tile1TokenOffset" || key == "tile1InputSize" || key == "tile1ExpectOffset" ||
           key == "tile1ExpectSize" || key == "tile1ExpectRowSize";
}

void ExpectCsvSpecChecks(const CsvTilingCase &item, const ContractInput &input, const TilingData &tiling)
{
    for (const auto &spec : item.spec) {
        EXPECT_TRUE(IsKnownSpecKey(spec.first)) << "caseName=" << item.caseName << ", specKey=" << spec.first;
    }
    if (item.HasSpec("tileTokenOffset")) {
        ExpectTileFromSpec(item, input, tiling, "tile");
    }
    if (item.HasSpec("lastCoreIndex")) {
        ExpectCoreCoverFromSpec(item, input, tiling);
    }
    if (item.HasSpec("tile0TokenOffset")) {
        ExpectCoreLocalTilesFromSpec(item, input, tiling);
    } else if (item.HasSpec("coreIndex")) {
        ExpectCoreRangeFromSpec(item, input, tiling);
    }
}

class QkvRmsNormRopeCacheWithKScaleCsvTiling : public testing::TestWithParam<CsvTilingCase> {};

} // namespace

TEST_P(QkvRmsNormRopeCacheWithKScaleCsvTiling, RunsCase)
{
    const auto &item = GetParam();
    auto input = BuildInput(item.totalTokens, item.numQHeads, item.numKHeads, item.numVHeads, item.headDim);
    input.inputLayout = item.inputLayout;
    RefreshShapes(input);
    TilingRunOptions options;
    options.aicNum = item.aicNum;
    ApplyCsvUpdates(input, options, item.updates);

    TilingInfo tilingInfo;
    const bool success = ExecuteTilingForInput(input, tilingInfo, options);
    const auto status = success ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    EXPECT_EQ(status, item.expectedStatus) << "caseName=" << item.caseName;
    if (status != ge::GRAPH_SUCCESS) {
        return;
    }
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(optiling::QkvRmsNormRopeCacheWithKScaleTilingData));
    ASSERT_NE(tilingInfo.tilingData, nullptr);
    const auto &tiling =
        *reinterpret_cast<const optiling::QkvRmsNormRopeCacheWithKScaleTilingData *>(tilingInfo.tilingData.get());

    EXPECT_EQ(tiling.totalTokens, input.totalTokens);
    EXPECT_EQ(tiling.batch, input.batch);
    EXPECT_EQ(tiling.qHeadNum, input.numQHeads);
    EXPECT_EQ(tiling.kvHeadNum, input.numKHeads);
    EXPECT_EQ(tiling.headDim, input.headDim);
    EXPECT_EQ(tiling.blockSize, input.blockSize);
    EXPECT_EQ(tiling.tokenTile, item.tokenTile);
    const uint64_t tokenTilePerAiv = Ops::Base::CeilDiv(tiling.tokenTile, QkvTiling::AIV_PER_AIC);
    const uint64_t qkHeadNum = tiling.qHeadNum + tiling.kvHeadNum;
    const uint64_t rowTile = tiling.tokenTile * qkHeadNum;
    EXPECT_EQ(tokenTilePerAiv, item.tokenTilePerAiv);
    EXPECT_EQ(rowTile, item.rowTile);
    EXPECT_EQ(Ops::Base::CeilAlign(rowTile, 16UL), item.rowTileAligned);
    EXPECT_EQ(tiling.coreTokenTile, item.coreTokenTile);
    EXPECT_EQ(tiling.coreGroupNum, item.coreGroupNum);
    EXPECT_EQ(qkHeadNum, input.numQHeads + input.numKHeads);
    EXPECT_LE(tokenTilePerAiv * qkHeadNum, QkvTiling::QK_OUTPUT_ROWS_PER_AIV);
    EXPECT_LE(tokenTilePerAiv * (qkHeadNum + tiling.kvHeadNum), QkvTiling::QKV_INPUT_ROWS_PER_AIV);
    EXPECT_LE(tokenTilePerAiv * tiling.kvHeadNum, QkvTiling::V_OUTPUT_ROWS_PER_AIV);
    EXPECT_EQ(tiling.kvCacheStrideBlock, item.kvStrideBlock);
    EXPECT_EQ(tiling.kvCacheStrideHead, item.kvStrideHead);
    EXPECT_EQ(tiling.kvCacheStrideToken, item.kvStrideToken);
    EXPECT_EQ(tiling.kScaleCacheStrideBlock, item.kScaleStrideBlock);
    EXPECT_EQ(tiling.kScaleCacheStrideHead, item.kScaleStrideHead);
    EXPECT_EQ(tiling.kScaleCacheStrideToken, item.kScaleStrideToken);
    EXPECT_FLOAT_EQ(tiling.epsilon, 1e-6F);
    ExpectTilingKey(item, input, tilingInfo);
    ExpectCsvSpecChecks(item, input, tiling);
}

INSTANTIATE_TEST_SUITE_P(CsvCases, QkvRmsNormRopeCacheWithKScaleCsvTiling,
    testing::ValuesIn(GetCsvTilingCases()), MakeCsvTilingCaseName);

TEST(QkvRmsNormRopeCacheWithKScaleBaseTiling, RealTilingIsStableAcrossThreads)
{
    constexpr uint32_t THREAD_COUNT = 8U;
    constexpr uint32_t ITERATIONS = 64U;
    std::vector<ConcurrentTilingCase> cases;
    cases.reserve(5U);
    cases.push_back({128, 16, 2, 2, 128, 32, ge::GRAPH_SUCCESS, 14, 4, 32});
    cases.push_back({512, 64, 8, 8, 128, 32, ge::GRAPH_SUCCESS, 2, 16, 32});
    cases.push_back({1024, 16, 2, 2, 128, 32, ge::GRAPH_SUCCESS, 14, 32, 32});
    cases.push_back({128, 128, 8, 8, 128, 32, ge::GRAPH_FAILED, 0, 0, 0});
    cases.push_back({128, 16, 2, 2, 64, 32, ge::GRAPH_FAILED, 0, 0, 0});
    std::atomic<bool> start(false);
    std::array<ConcurrentTilingResult, THREAD_COUNT> results;
    std::array<std::thread, THREAD_COUNT> threads;

    for (uint32_t threadIdx = 0U; threadIdx < THREAD_COUNT; ++threadIdx) {
        threads[threadIdx] = std::thread([threadIdx, &cases, &results, &start]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (uint32_t iter = 0U; iter < ITERATIONS; ++iter) {
                const uint64_t caseIdx = (static_cast<uint64_t>(threadIdx) + iter) % cases.size();
                const auto &item = cases[caseIdx];
                auto input = BuildInput(item.totalTokens, item.numQHeads, item.numKHeads, item.numVHeads, item.headDim);
                input.inputLayout = item.inputLayout;
                RefreshShapes(input);
                TilingInfo tilingInfo;
                const bool success = ExecuteTilingForInput(input, tilingInfo, item.aicNum);
                if (!MatchesConcurrentTilingCase(item, success, tilingInfo)) {
                    results[threadIdx].ok = false;
                    results[threadIdx].failedCase = caseIdx;
                    results[threadIdx].failedIteration = iter;
                    results[threadIdx].success = success;
                    results[threadIdx].blockNum = tilingInfo.blockNum;
                    if (success && tilingInfo.tilingDataSize >= sizeof(optiling::QkvRmsNormRopeCacheWithKScaleTilingData) &&
                        tilingInfo.tilingData != nullptr) {
                        const auto *tilingData = reinterpret_cast<const optiling::QkvRmsNormRopeCacheWithKScaleTilingData *>(
                            tilingInfo.tilingData.get());
                        results[threadIdx].tokenTile = tilingData->tokenTile;
                        results[threadIdx].coreTokenTile = tilingData->coreTokenTile;
                        results[threadIdx].coreGroupNum = tilingData->coreGroupNum;
                    }
                    return;
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto &thread : threads) {
        thread.join();
    }

    for (uint32_t threadIdx = 0U; threadIdx < THREAD_COUNT; ++threadIdx) {
        EXPECT_TRUE(results[threadIdx].ok)
            << "threadIdx=" << threadIdx << " failedCase=" << results[threadIdx].failedCase
            << " failedIteration=" << results[threadIdx].failedIteration
            << " success=" << results[threadIdx].success
            << " blockNum=" << results[threadIdx].blockNum
            << " tokenTile=" << results[threadIdx].tokenTile
            << " coreTokenTile=" << results[threadIdx].coreTokenTile
            << " coreGroupNum=" << results[threadIdx].coreGroupNum;
    }
}
