/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_grad_tiling.cpp
 * \brief CompressorGrad arch22 (A3 / ascend910_93) tiling.
 *
 * Validates with the BSH subset of the arch35 Check* series plus the A3
 * implementation's constraints (TH layout rejected, nblk/nblkIn checks), then
 * computes the schedule the same way the verified standalone A3 host did:
 * cast/scatter core clamps, segLen 64->32->16 UB downgrade, shape-adaptive
 * gDx split, and the five-block fp16 workspace layout.
 */

#include <functional>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "compressor_grad_tiling_arch22.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace {

template <typename T>
std::string to_string(const T& value)
{
    if (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        return std::to_string(value);
    }
}

template <typename T>
void LogErrorNumberSupport(const std::vector<T>& expectNumberList, const T& actualValue, const std::string& name,
                           const std::string subName, const char* opName)
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, name, to_string(actualValue),
                                          subName + " only supports " + oss.str());
}

template <typename T>
ge::graphStatus CheckFeatureValueSupport(const T* featureValue, const std::vector<T>& expectFeatureValList,
                                         const std::string& name, const char* opName)
{
    if (std::find(expectFeatureValList.begin(), expectFeatureValList.end(), *featureValue) ==
        expectFeatureValList.end()) {
        LogErrorNumberSupport(expectFeatureValList, *featureValue, name, "feature value", opName);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CheckAttrValueSupportInterval(const T* attrValue, const uint32_t minVal, const uint32_t maxVal,
                                              const std::string& name, const char* opName)
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    std::string attr_value = "attr value";
    if (*attrValue < minVal || *attrValue > maxVal) {
        std::ostringstream oss;
        oss << "[" << minVal << ", " << maxVal << "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, name, std::to_string(*attrValue),
                                              "attr value only supports value in range " + oss.str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CheckAttrValueSupportList(const T* attrValue, const std::vector<T>& expectAttrValList,
                                          const std::string& name, const char* opName)
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectAttrValList.begin(), expectAttrValList.end(), *attrValue) == expectAttrValList.end()) {
        LogErrorNumberSupport(expectAttrValList, *attrValue, name, "attr value", opName);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static std::string DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("CompressorGrad", "datatype", std::to_string(static_cast<int32_t>(type)),
                                              "not support");
        return "UNDEFINED";
    }
}

void LogErrorDtypeSupport(const std::vector<ge::DataType>& expectDtypeList, const ge::DataType& actualDtype,
                          const std::string& name, const char* opName)
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << DataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_DTYPE(opName, name, DataTypeToSerialString(actualDtype), oss.str());
}

ge::graphStatus LogErrorShapeConsistency(const std::string& name, const gert::StorageShape* shape,
                                         const uint32_t& dimNum, const std::string& subName, const uint32_t& expectNum,
                                         const char* opName)
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t actualNum = shape->GetStorageShape().GetDim(dimNum);
    OP_CHECK_IF(actualNum != expectNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, name, "dim " + std::to_string(dimNum) + "=" + std::to_string(actualNum),
                    "should be equal to " + subName + ": " + std::to_string(expectNum)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

struct ShapeCheckItem {
    const char* tensorName;
    const gert::StorageShape* shape;
    uint32_t dimIdx;
    const char* expectName;
    uint32_t expectValue;
};

ge::graphStatus CheckShapeList(const char* opName, const ShapeCheckItem* items, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        if (LogErrorShapeConsistency(items[i].tensorName, items[i].shape, items[i].dimIdx, items[i].expectName,
                                     items[i].expectValue, opName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ── 辅助函数：检查单个 tensor 的 shape 和 desc 均不为空 ──
ge::graphStatus CheckTensorShapeAndDesc(const char* opName, const char* tensorName,
                                        const gert::CompileTimeTensorDesc* desc, const gert::StorageShape* shape)
{
    OP_CHECK_IF(shape == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, tensorName, "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(desc == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, tensorName, "desc is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

struct TensorCheckItem {
    const char* name;
    const gert::CompileTimeTensorDesc* desc;
    const gert::StorageShape* shape;
};

ge::graphStatus CheckTensorList(const char* opName, const TensorCheckItem* items, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        if (CheckTensorShapeAndDesc(opName, items[i].name, items[i].desc, items[i].shape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus CompressorGradTilingA3::ConvertRequiredParams(gert::TilingContext& context,
                                                              CompressorGradContext& compressorGradContext)
{
    compressorGradContext.x.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    compressorGradContext.x.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    OP_CHECK_IF(compressorGradContext.x.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorGradContext.opName, X_NAME, "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compressorGradContext.x.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorGradContext.opName, X_NAME, "desc is nullptr"),
                return ge::GRAPH_FAILED);
    compressorGradContext.wkv.desc = context.GetRequiredInputDesc(WEIGHT_KV_INPUT_INDEX);
    compressorGradContext.wkv.shape = context.GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);
    compressorGradContext.wgate.desc = context.GetRequiredInputDesc(WEIGHT_WGATE_INPUT_INDEX);
    compressorGradContext.wgate.shape = context.GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);
    compressorGradContext.dCmpKv.desc = context.GetRequiredInputDesc(D_CMP_KV_INPUT_INDEX);
    compressorGradContext.dCmpKv.shape = context.GetRequiredInputShape(D_CMP_KV_INPUT_INDEX);
    compressorGradContext.softmaxScore.desc = context.GetRequiredInputDesc(SOFTMAX_SCORE_INPUT_INDEX);
    compressorGradContext.softmaxScore.shape = context.GetRequiredInputShape(SOFTMAX_SCORE_INPUT_INDEX);
    compressorGradContext.kv.desc = context.GetRequiredInputDesc(KV_INPUT_INDEX);
    compressorGradContext.kv.shape = context.GetRequiredInputShape(KV_INPUT_INDEX);

    compressorGradContext.dX.desc = context.GetOutputDesc(D_X_OUTPUT_INDEX);
    compressorGradContext.dX.shape = context.GetOutputShape(D_X_OUTPUT_INDEX);
    compressorGradContext.dWkv.desc = context.GetOutputDesc(D_WKV_OUTPUT_INDEX);
    compressorGradContext.dWkv.shape = context.GetOutputShape(D_WKV_OUTPUT_INDEX);
    compressorGradContext.dWgate.desc = context.GetOutputDesc(D_WGATE_OUTPUT_INDEX);
    compressorGradContext.dWgate.shape = context.GetOutputShape(D_WGATE_OUTPUT_INDEX);
    compressorGradContext.dApe.desc = context.GetOutputDesc(D_APE_OUTPUT_INDEX);
    compressorGradContext.dApe.shape = context.GetOutputShape(D_APE_OUTPUT_INDEX);

    compressorGradContext.dtype = compressorGradContext.x.desc->GetDataType();
    auto xDimNum = compressorGradContext.x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_3) {
        compressorGradContext.layout = LayoutType::LAYOUT_BSH;
    } else if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_2) {
        compressorGradContext.layout = LayoutType::LAYOUT_TH;
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(compressorGradContext.opName, X_NAME, std::to_string(xDimNum),
                                                 "x dimension should be 2 or 3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void CompressorGradTilingA3::ConvertOptionalParams(gert::TilingContext& context,
                                                   CompressorGradContext& compressorGradContext)
{
    compressorGradContext.cuSeqlens.desc = context.GetOptionalInputDesc(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.cuSeqlens.shape = context.GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.seqUsed.desc = context.GetOptionalInputDesc(SEQ_USED_INPUT_INDEX);
    compressorGradContext.seqUsed.shape = context.GetOptionalInputShape(SEQ_USED_INPUT_INDEX);
    compressorGradContext.startPos.desc = context.GetOptionalInputDesc(START_POS_INPUT_INDEX);
    compressorGradContext.startPos.shape = context.GetOptionalInputShape(START_POS_INPUT_INDEX);
}

ge::graphStatus CompressorGradTilingA3::ConvertContext(gert::TilingContext& context,
                                                       CompressorGradContext& compressorGradContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "opName", "got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    compressorGradContext.opName = context.GetNodeName();
    compressorGradContext.opType = context.GetNodeType();
    compressorGradContext.platformInfo = context.GetPlatformInfo();
    OP_CHECK_IF(ConvertRequiredParams(context, compressorGradContext) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    ConvertOptionalParams(context, compressorGradContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.GetNodeName(), "attrs", "got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    compressorGradContext.coff = attrs->GetAttrPointer<uint32_t>(COFF_ATTR_INDEX);
    compressorGradContext.cmpRatio = attrs->GetAttrPointer<uint32_t>(CMP_RATIO_ATTR_INDEX);

    OP_CHECK_IF(
        context.GetWorkspaceSizes(1) == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.GetNodeName(), "workSpaceSize", "got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    compressorGradContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "platformInfo", "is nullptr"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();

    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(
        aicNum_ == 0 || aivNum_ == 0,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "aicNum/aivNum", "num of core obtained is 0"),
        return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::SetBaseInfo()
{
    // arch22 仅支持 BSH（TH 已在 CheckLayoutSupport 拦截）
    baseParams_.batchSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
    baseParams_.seqSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    baseParams_.hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_2);
    baseParams_.tokenSize = baseParams_.batchSize * baseParams_.seqSize;

    coff_ = context_->coff == nullptr ? COFF_VALUE : static_cast<uint8_t>(*context_->coff);
    baseParams_.headDim = context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) / coff_;
    baseParams_.featureDim = context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
    baseParams_.cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    baseParams_.usedCoreNum = aicNum_;

    // A3 派生量（与已验证 A3 host 的公式一致）
    tPad_ = (static_cast<int64_t>(baseParams_.tokenSize) + CG_A3_NZ_TILE - 1) / CG_A3_NZ_TILE * CG_A3_NZ_TILE;
    nblk_ = baseParams_.seqSize / baseParams_.cmpRatio; // complete blocks only
    nblkIn_ = context_->dCmpKv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    dnPlanes_ = (context_->dtype == ge::DT_FLOAT16) ? 2 : 1;

    OP_LOGI(context_->opName, "[TILING][arch22] bSize:%u sSize:%u tSize:%u cmpRatio:%u coff:%u", baseParams_.batchSize,
            baseParams_.seqSize, baseParams_.tokenSize, baseParams_.cmpRatio, coff_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CalcSchedule()
{
    const int64_t tLen = baseParams_.tokenSize;
    const int64_t hSize = baseParams_.hiddenSize;
    const int64_t cSize = coff_ * baseParams_.headDim;
    const int64_t nbTotal = static_cast<int64_t>(baseParams_.batchSize) * nblk_;
    const int64_t nRows = static_cast<int64_t>(coff_) * baseParams_.cmpRatio;

    // ---- cast 核数（A3 host: min(aivNum, max(1, T_pad + 2*C))）----
    int64_t totalRows = tPad_ + 2 * cSize;
    castCoreNum_ = std::min<int64_t>(aivNum_, std::max<int64_t>(1, totalRows));

    // ---- scatter 核数与每核块数 ----
    scatterCoreNum_ = std::min<int64_t>(aivNum_, nbTotal);
    blocksPerCore_ = (nbTotal + scatterCoreNum_ - 1) / scatterCoreNum_;

    // ---- SEG: UB budget 降档 64 -> 32 -> 16（arch2201 TBuf ceiling 184KB）----
    //   bytes(S) = (2*cr + 3*N + 5)*S*4 + (planes-1)*cr*S*4 + S*ioT(2B)
    //              + slots*planes*cr*S*2
    const int64_t planes = dnPlanes_;
    const int64_t slots = (coff_ == 2) ? 2 : 1;
    const int64_t cr = baseParams_.cmpRatio;
    auto segBytes = [&](int64_t s) {
        return (2 * cr + 3 * nRows + 5) * s * 4 + (planes - 1) * cr * s * 4 + s * 2 + slots * planes * cr * s * 2;
    };
    segLen_ = 64;
    if (segBytes(64) > static_cast<int64_t>(CG_A3_SCATTER_UB_LIMIT) || baseParams_.headDim < 64) {
        segLen_ = 32;
    }
    if (segBytes(segLen_) > static_cast<int64_t>(CG_A3_SCATTER_UB_LIMIT)) {
        segLen_ = 16;
    }
    // final UB guardrail: reject shapes that overflow even after the downgrade
    OP_CHECK_IF(
        segBytes(16) > static_cast<int64_t>(CG_A3_SCATTER_UB_LIMIT),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "cmpRatio/headDim",
                                              std::to_string(cr) + "/" + std::to_string(baseParams_.headDim),
                                              "scatter UB budget exceeded even at segLen=16; shape not supported"),
        return ge::GRAPH_FAILED);

    // ---- gemm dx/dw 核数划分（A3 R7 形状自适应公式）----
    // Post-R6/R7-s1 the dw branch is far cheaper per output tile (dn A tiles
    // L2-hot + x B contiguous), so mid-size shapes are dx-bound at the even
    // split. Tile-count cost model calibrated on dev10: dx tile ~1.0; dw tile
    // ~0.62 when the x workspace stays L2-resident, ~1.16 when it is
    // HBM-streamed. Shift cores dx-ward only when dx clearly dominates
    // (>=1.25x), clamp to [half, half+4]: balanced shapes keep the even split.
    const int64_t gemmCores = aicNum_;
    int64_t gDx = std::max<int64_t>(1, gemmCores / 2);
    {
        const int64_t nM = (tLen + CG_A3_NZ_TILE - 1) / CG_A3_NZ_TILE;
        const int64_t nH = hSize / CG_A3_NZ_TILE;  // H % 128 == 0 (checked)
        const int64_t nCn = cSize / CG_A3_NZ_TILE; // C % 128 == 0
        const int64_t nPairRow = (nH + 1) / 2;
        const bool fp16 = (context_->dtype == ge::DT_FLOAT16);
        const double dxCost = static_cast<double>(nM * nPairRow * (cSize / CG_A3_NZ_TILE)) * (fp16 ? 8 : 6);
        const double dwTiles = static_cast<double>(nCn * nH * (tPad_ / CG_A3_NZ_TILE)) * (fp16 ? 5 : 3);
        const bool xResident = (tPad_ * hSize * 2) <= static_cast<int64_t>(128) * 1024 * 1024;
        const double dwCost = dwTiles * (xResident ? 0.62 : 1.16);
        if (dxCost >= 1.25 * dwCost) {
            int64_t g = static_cast<int64_t>(static_cast<double>(gemmCores) * dxCost / (dxCost + dwCost) + 0.5);
            g = std::min<int64_t>(std::max<int64_t>(g, gemmCores / 2), gemmCores / 2 + 4);
            g = std::min<int64_t>(g, gemmCores);
            g = std::min(g, std::max<int64_t>(1, nM * nPairRow));
            gDx = g;
        }
    }
    gemmDxCores_ = gDx;

    OP_LOGI(context_->opName, "[TILING][arch22] castCores:%ld scatterCores:%ld blocksPerCore:%ld segLen:%ld gDx:%ld",
            castCoreNum_, scatterCoreNum_, blocksPerCore_, segLen_, gemmDxCores_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CalcWorkSpace()
{
    // fp16 path: dn workspaces carry HiLo dual planes (dn = hi + lo, plane
    // stride T_pad*C); bf16 stays single-plane.
    // One flat fp16 workspace, five sub-blocks (same layout as the verified
    // standalone A3 host): xF16 | wkvF16 | wgateF16 | dnKv | dnSc.
    // T_pad/H/C are all 128-aligned, so every block keeps 32KB alignment.
    constexpr int64_t f16Size = 2; // fp16 element bytes
    const int64_t hSize = baseParams_.hiddenSize;
    const int64_t cSize = coff_ * baseParams_.headDim;
    const int64_t xBytes = tPad_ * hSize * f16Size;
    const int64_t wBytes = cSize * hSize * f16Size;
    const int64_t dnBytes = dnPlanes_ * tPad_ * cSize * f16Size;

    xF16WsOff_ = 0;
    wkvF16WsOff_ = xF16WsOff_ + xBytes;
    wgateF16WsOff_ = wkvF16WsOff_ + wBytes;
    dnKvWsOff_ = wgateF16WsOff_ + wBytes;
    dnScWsOff_ = dnKvWsOff_ + dnBytes;
    const int64_t userWsBytes = dnScWsOff_ + dnBytes;

    workspaceSize_ = libapiSize_ + static_cast<size_t>(userWsBytes);
    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }

    OP_LOGI(context_->opName, "[TILING][arch22] workspaceSize = %zu (x=%ld w=2x%ld dn=2x%ld planes=%ld)",
            workspaceSize_, xBytes, wBytes, dnBytes, dnPlanes_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::SetTilingData()
{
    tilingData_->batchSize = baseParams_.batchSize;
    tilingData_->seqSize = baseParams_.seqSize;
    tilingData_->tLen = baseParams_.tokenSize;
    tilingData_->tPad = tPad_;
    tilingData_->hiddenSize = baseParams_.hiddenSize;
    tilingData_->headDim = baseParams_.headDim;
    tilingData_->totalHeadDim = coff_ * baseParams_.headDim;
    tilingData_->cmpRatio = baseParams_.cmpRatio;
    tilingData_->coff = coff_;
    tilingData_->cmpRowCnt = coff_ * baseParams_.cmpRatio;
    tilingData_->nblk = nblk_;
    tilingData_->nblkIn = nblkIn_;
    tilingData_->dnPlanes = dnPlanes_;
    tilingData_->castCoreNum = castCoreNum_;
    tilingData_->scatterCoreNum = scatterCoreNum_;
    tilingData_->blocksPerCore = blocksPerCore_;
    tilingData_->segLen = segLen_;
    tilingData_->gemmDxCores = gemmDxCores_;
    tilingData_->xF16WsOff = xF16WsOff_;
    tilingData_->wkvF16WsOff = wkvF16WsOff_;
    tilingData_->wgateF16WsOff = wgateF16WsOff_;
    tilingData_->dnKvWsOff = dnKvWsOff_;
    tilingData_->dnScWsOff = dnScWsOff_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::GenTilingKey() const
{
    // 0:BF16, 1:FP16（与 arch35 编码一致）
    uint8_t dtype = 0;
    auto xDtype = context_->x.desc->GetDataType();
    if (xDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    // arch22 模板空间只有 D_TYPE 一个字段（ASCENDC_TPL_UINT_DECL(D_TYPE, 2bit, LIST, 0, 1)）
    context_->tilingKey = GET_TPL_TILING_KEY(dtype);

    OP_LOGI(context_->opName, "CompressorGrad[arch22] dtype:%hhu tilingKey:%lu", dtype, context_->tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDimNumInLayoutSupport(const std::string& layout,
                                                                   const gert::StorageShape* shape,
                                                                   const std::string& name) const
{
    const auto& dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, name, std::to_string(shape->GetStorageShape().GetDimNum()),
                    "when layout is " + layout + ", dimension should be " + std::to_string(dimIt->second)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDtypeSupport(const gert::CompileTimeTensorDesc* desc,
                                                          const std::string& name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context_->opName, name, "datatype support list should be specify in DTYPE_SUPPORT_MAP"),
                    return ge::GRAPH_FAILED);
        auto& expectDtypeList = it->second;
        OP_CHECK_IF(
            std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name, context_->opName),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDimNumSupport(const gert::StorageShape* shape,
                                                           const std::string& name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto& it = DIM_NUM_MAP.find(name);
    OP_CHECK_IF(it == DIM_NUM_MAP.end(),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, name,
                                                         "dim number support list should be specify in DIM_NUM_MAP"),
                return ge::GRAPH_FAILED);
    auto& expectDimNumList = it->second;
    OP_CHECK_IF(std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
                    expectDimNumList.end(),
                LogErrorNumberSupport(expectDimNumList, static_cast<uint32_t>(shape->GetStorageShape().GetDimNum()),
                                      name, "dimension", context_->opName),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaX() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->x.desc, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->x.shape, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport("BSH", context_->x.shape, X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaWkv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wkv.desc, WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wkv.shape, WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaWgate() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wgate.desc, WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wgate.shape, WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaDCmpKv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dCmpKv.desc, D_CMP_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dCmpKv.shape, D_CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaSoftmaxScore() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->softmaxScore.desc, SOFTMAX_SCORE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->softmaxScore.shape, SOFTMAX_SCORE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaKV() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->kv.desc, KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->kv.shape, KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaCuSeqlens() const
{
    if (context_->cuSeqlens.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cuSeqlens.desc, CU_SEQLENS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cuSeqlens.shape, CU_SEQLENS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaSeqused() const
{
    if (context_->seqUsed.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->seqUsed.desc, SEQUSED_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->seqUsed.shape, SEQUSED_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaStartPos() const
{
    if (context_->startPos.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->startPos.desc, START_POS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->startPos.shape, START_POS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaDX() const
{
    if (context_->dX.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dX.desc, D_X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dX.shape, D_X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaDWkv() const
{
    if (context_->dWkv.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dWkv.desc, D_WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dWkv.shape, D_WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaDWgate() const
{
    if (context_->dWgate.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dWgate.desc, D_WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dWgate.shape, D_WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaDApe() const
{
    if (context_->dApe.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dApe.desc, D_APE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dApe.shape, D_APE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaCmpRatio() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupportInterval(context_->cmpRatio, MIN_CMP_RATIO, MAX_CMP_RATIO,
                                                           CMP_RATIO_NAME, context_->opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaCoff() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupportList(context_->coff, COFF, COFF_NAME, context_->opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckLayoutSupport() const
{
    // arch22 实现仅支持 BSH 布局，TH 显式拦截
    OP_CHECK_IF(context_->layout != LayoutType::LAYOUT_BSH,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, X_NAME, std::to_string(context_->x.shape->GetStorageShape().GetDimNum()) + "D",
                    "CompressorGrad on arch22 (Ascend910_93) only supports BSH layout, TH is not supported"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckFeature() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureValueSupport(&baseParams_.headDim, HEAD_DIM, "headDim", context_->opName)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseParams_.hiddenSize > MAX_HIDDEN_SIZE || baseParams_.hiddenSize < MIN_HIDDEN_SIZE ||
            baseParams_.hiddenSize % ALIGN_FACTOR_HIDDEN_SIZE != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "hiddenSize", std::to_string(baseParams_.hiddenSize),
                                              "should be within [" + std::to_string(MIN_HIDDEN_SIZE) + ", " +
                                                  std::to_string(MAX_HIDDEN_SIZE) + "] and be 512-aligned"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckShapeConsistency() const
{
    uint8_t coff = coff_;
    auto coffD = baseParams_.headDim * coff;
    const char* opName = context_->opName;
    // 公共部分：cuSeqlens/seqUsed/startPos + wkv/wgate 的 hiddenSize 和 coff*headDim
    const ShapeCheckItem commonChecks[] = {
        {"cuSeqlens", context_->cuSeqlens.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize+1", baseParams_.batchSize + 1},
        {"seqUsed", context_->seqUsed.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
        {"startPos", context_->startPos.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
        {"wkv", context_->wkv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "hiddenSize", baseParams_.hiddenSize},
        {"wgate", context_->wgate.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "hiddenSize", baseParams_.hiddenSize},
        {"wkv", context_->wkv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "coff*headDim", static_cast<uint32_t>(coffD)},
        {"wgate", context_->wgate.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "coff*headDim", static_cast<uint32_t>(coffD)},
    };
    if (CheckShapeList(opName, commonChecks, sizeof(commonChecks) / sizeof(commonChecks[0])) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // BSH 3D 分支（TH 已在 CheckLayoutSupport 拦截）
    const ShapeCheckItem checks3D[] = {
        {"dCmpKv", context_->dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
        {"dCmpKv", context_->dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "headDim", baseParams_.headDim},
        {"softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
        {"softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio",
         coff * baseParams_.cmpRatio},
        {"softmaxScore", context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_3, "headDim", baseParams_.headDim},
        {"kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
        {"kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio", coff * baseParams_.cmpRatio},
        {"kv", context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_3, "headDim", baseParams_.headDim},
    };
    if (CheckShapeList(opName, checks3D, sizeof(checks3D) / sizeof(checks3D[0])) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc* desc,
                                                               const std::string& name) const
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(
        actualDtype != context_->dtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->opName, name, DataTypeToSerialString(actualDtype),
                                              "should be same with x: " + DataTypeToSerialString(context_->dtype)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDtypeConsistency() const
{
    if (CheckDtypeConsistencyX(context_->wkv.desc, WKV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->wgate.desc, WGATE_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dCmpKv.desc, D_CMP_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dX.desc, D_X_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dWkv.desc, D_WKV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dWgate.desc, D_WGATE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDimNumConsistency() const
{
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(xDimNum != context_->dX.shape->GetStorageShape().GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, "d_x", std::to_string(context_->dX.shape->GetStorageShape().GetDimNum()),
                    "dim num should be equal to x: " + std::to_string(xDimNum)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xDimNum != context_->dCmpKv.shape->GetStorageShape().GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, "d_cmp_kv", std::to_string(context_->dCmpKv.shape->GetStorageShape().GetDimNum()),
                    "dim num should be equal to x: " + std::to_string(xDimNum)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xDimNum != context_->softmaxScore.shape->GetStorageShape().GetDimNum() - 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, "softmax_score",
                    std::to_string(context_->softmaxScore.shape->GetStorageShape().GetDimNum()),
                    "dim num should be x dim + 1: " + std::to_string(xDimNum + 1)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xDimNum != context_->kv.shape->GetStorageShape().GetDimNum() - 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, "kv", std::to_string(context_->kv.shape->GetStorageShape().GetDimNum()),
                    "dim num should be x dim + 1: " + std::to_string(xDimNum + 1)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckBlockDimConstrain() const
{
    uint32_t minBlockNum = baseParams_.headDim / D_BASE_SIZE; // D_BASE_SIZE is the largest dBaseSize
    OP_CHECK_IF(aicNum_ < minBlockNum,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "aicNum", std::to_string(aicNum_),
                                                      "should not be less than " + std::to_string(minBlockNum)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckA3Constraints() const
{
    // A3 实现语义：仅处理完整压缩块（nblk = S / cmpRatio），不完整尾部块不落 dn
    OP_CHECK_IF(nblk_ < 1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->opName, "seqSize/cmpRatio",
                    std::to_string(baseParams_.seqSize) + "/" + std::to_string(baseParams_.cmpRatio),
                    "arch22 requires at least one complete block (seqSize >= cmpRatio)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        nblkIn_ < nblk_,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->opName, "d_cmp_kv", "dim1=" + std::to_string(nblkIn_),
                                              "d_cmp_kv blocks must cover complete blocks: " + std::to_string(nblk_)),
        return ge::GRAPH_FAILED);
    const int64_t smBlocks = context_->softmaxScore.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    const int64_t kvBlocks = context_->kv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    OP_CHECK_IF(
        smBlocks != nblkIn_ || kvBlocks != nblkIn_,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->opName, "softmax_score/kv", "dim1=" + std::to_string(smBlocks) + "/" + std::to_string(kvBlocks),
            "softmax_score/kv block dim should be equal to d_cmp_kv: " + std::to_string(nblkIn_)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckMultiParaConsistency() const
{
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS || CheckDtypeConsistency() != ge::GRAPH_SUCCESS ||
        CheckDimNumConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaX() || ge::GRAPH_SUCCESS != CheckSingleParaWkv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWgate() || ge::GRAPH_SUCCESS != CheckSingleParaDCmpKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSoftmaxScore() || ge::GRAPH_SUCCESS != CheckSingleParaKV() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens() || ge::GRAPH_SUCCESS != CheckSingleParaSeqused() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos() || ge::GRAPH_SUCCESS != CheckSingleParaDX() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWkv() || ge::GRAPH_SUCCESS != CheckSingleParaDWgate() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDApe() || ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCoff()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckRequiredInOutExistence() const
{
    const TensorCheckItem requiredTensors[] = {
        {"x", context_->x.desc, context_->x.shape},
        {"wkv", context_->wkv.desc, context_->wkv.shape},
        {"wgate", context_->wgate.desc, context_->wgate.shape},
        {"d_cmp_kv", context_->dCmpKv.desc, context_->dCmpKv.shape},
        {"softmax_score", context_->softmaxScore.desc, context_->softmaxScore.shape},
        {"kv", context_->kv.desc, context_->kv.shape},
        {"d_x", context_->dX.desc, context_->dX.shape},
        {"d_wkv", context_->dWkv.desc, context_->dWkv.shape},
        {"d_wgate", context_->dWgate.desc, context_->dWgate.shape},
        {"d_ape", context_->dApe.desc, context_->dApe.shape},
    };
    if (CheckTensorList(context_->opName, requiredTensors, sizeof(requiredTensors) / sizeof(requiredTensors[0])) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // arch22 仅 BSH：cu_seqlens 必须为空（TH 已在 CheckLayoutSupport 拦截）
    OP_CHECK_IF(
        context_->cuSeqlens.desc != nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens", "in BSH layout, must be nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->cuSeqlens.shape != nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens", "in BSH layout, must be nullptr"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(context_->cmpRatio == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cmp_ratio", "attr is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckEmptyTensor() const
{
    // CompressorGrad 不支持空 tensor：所有输入/输出 shapeSize 必须 > 0，空则直接拦截
    if (context_->x.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->wkv.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->wgate.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->dCmpKv.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->softmaxScore.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->kv.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->dX.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->dWkv.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->dWgate.shape->GetStorageShape().GetShapeSize() == 0 ||
        context_->dApe.shape->GetStorageShape().GetShapeSize() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->opName, "x", "0",
            "CompressorGrad does not support empty tensor: all inputs/outputs shapeSize must be > 0");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::RunBigKernelTiling(CompressorGradTilingDataA3* tilingData)
{
    tilingData_ = tilingData;
    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> tilingFuncs{std::bind(&CompressorGradTilingA3::GetNpuInfo, this),
                                            std::bind(&CompressorGradTilingA3::CheckRequiredParaExistence, this),
                                            std::bind(&CompressorGradTilingA3::CheckEmptyTensor, this),
                                            std::bind(&CompressorGradTilingA3::CheckLayoutSupport, this),
                                            std::bind(&CompressorGradTilingA3::CheckSinglePara, this),
                                            std::bind(&CompressorGradTilingA3::SetBaseInfo, this),
                                            std::bind(&CompressorGradTilingA3::CheckFeature, this),
                                            std::bind(&CompressorGradTilingA3::CheckMultiParaConsistency, this),
                                            std::bind(&CompressorGradTilingA3::CheckBlockDimConstrain, this),
                                            std::bind(&CompressorGradTilingA3::CheckA3Constraints, this),
                                            std::bind(&CompressorGradTilingA3::CalcSchedule, this),
                                            std::bind(&CompressorGradTilingA3::CalcWorkSpace, this),
                                            std::bind(&CompressorGradTilingA3::SetTilingData, this),
                                            std::bind(&CompressorGradTilingA3::GenTilingKey, this)};
    for (const auto& func : tilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // 单入口 MIX_AIC_1_2 kernel：blockDim 计 AIC 核数（AIV 自动 x2）
    context_->blockDim = aicNum_;

    OP_LOGI(context_->opName, "Run big kernel (arch22)");

    return ge::GRAPH_SUCCESS;
}

CMP_GRAD_EXTERN_C ge::graphStatus TilingCompressorGradArch22(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "context", "is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling (arch22)");

    CompressorGradContext compressorGradContext{};
    if (CompressorGradTilingA3::ConvertContext(*context, compressorGradContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "context",
            "error occurred while converting tilingContext to CompressorGrad context");
        return ge::GRAPH_FAILED;
    }

    CompressorGradTilingDataA3* tilingData = context->GetTilingData<CompressorGradTilingDataA3>();
    OP_CHECK_IF(tilingData == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorGradContext.opName, "tilingData", "is nullptr"),
                return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);

    CompressorGradTilingA3 compressorGradTiling(&compressorGradContext);
    if (compressorGradTiling.RunBigKernelTiling(tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(compressorGradContext.tilingKey);
    context->SetBlockDim(compressorGradContext.blockDim);
    OP_LOGI(compressorGradContext.opName, "block dim: %u.", compressorGradContext.blockDim);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
