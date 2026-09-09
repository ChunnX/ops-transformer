// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <torch/extension.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_TWO = 2;
const int64_t MX_SCALE_GROUP_SIZE = 32;
const int64_t PACKED_FLOAT4_ELEMENTS_PER_BYTE = 2;
constexpr int64_t GE_DTYPE_UNDEFINED = 28;

bool HasTensor(const c10::optional<at::Tensor> &tensor)
{
    return tensor.has_value() && tensor.value().defined();
}

std::tuple<at::Tensor, at::Tensor> NpuMegaMoe(
    const at::Tensor &context, const at::Tensor &x, const at::Tensor &topkIds, const at::Tensor &topkWeights,
    const std::vector<at::Tensor> &weight1, const std::vector<at::Tensor> &weight2, int64_t moeExpertNum,
    int64_t epWorldSize, int64_t cclBufferSize, const c10::optional<std::vector<at::Tensor>> &weightScales1,
    const c10::optional<std::vector<at::Tensor>> &weightScales2, const c10::optional<std::vector<at::Tensor>> &bias1,
    const c10::optional<std::vector<at::Tensor>> &bias2, const c10::optional<at::Tensor> &xActiveMask,
    const c10::optional<at::Tensor> &scales,
    const c10::optional<std::vector<at::Tensor>> &sharedWeight1,
    const c10::optional<std::vector<at::Tensor>> &sharedWeight2,
    const c10::optional<std::vector<at::Tensor>> &sharedWeightScales1,
    const c10::optional<std::vector<at::Tensor>> &sharedWeightScales2,
    const c10::optional<std::vector<at::Tensor>> &sharedBias1,
    const c10::optional<std::vector<at::Tensor>> &sharedBias2, const c10::optional<at::Tensor> &maskBuffer,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t combineQuantMode, std::string commAlg,
    int64_t numMaxTokensPerRank, std::string activation, std::vector<float> activationParams,
    c10::optional<int64_t> dispatchQuantOutDtype, c10::optional<int64_t> sharedExpertQuantOutDtype,
    c10::optional<int64_t> weight1Type, c10::optional<int64_t> weight2Type, c10::optional<int64_t> sharedWeight1Type,
    c10::optional<int64_t> sharedWeight2Type, c10::optional<int64_t> topoType, c10::optional<int64_t> rankNumPerServer,
    int64_t topkWeightsType)
{
    TORCH_CHECK((epWorldSize > 0), "The ep_world_sizes should be greater than 0, current is: ", epWorldSize);
    TORCH_CHECK((x.dim() == DIM_TWO) && (topkIds.dim() == DIM_TWO), "The x and topk_ids should be 2D");

    const auto xScalarType = x.scalar_type();
    const bool isFp8X = xScalarType == at::ScalarType::Float8_e5m2 ||
                        xScalarType == at::ScalarType::Float8_e4m3fn;
    const bool isPackedFp4X = xScalarType == at::kByte || xScalarType == at::ScalarType::Float4_e2m1fn_x2;
    const bool isPreQuantizedX = isFp8X || isPackedFp4X;
    TORCH_CHECK(
        ((xScalarType == at::kBFloat16) || (xScalarType == at::kHalf) || isPreQuantizedX) &&
            (topkIds.scalar_type() == at::kInt),
        "dtype of x should be bfloat16, float16, float8_e5m2, float8_e4m3fn, or uint8/float4_e2m1fn_x2 "
        "for packed FP4; dtype of topk_ids should be int32.");

    aclDataType xAclDtype = isPackedFp4X ? ACL_FLOAT4_E2M1 : ConvertToAclDataType(xScalarType);
    const bool hasScales = HasTensor(scales);
    TORCH_CHECK(isPreQuantizedX ? hasScales : !hasScales,
                isPreQuantizedX ? "scales is required for pre-quantized x."
                                : "scales must be None when x is bfloat16 or float16.");
    if (isPreQuantizedX) {
        TORCH_CHECK(dispatchQuantMode == 4, "dispatch_quant_mode must be 4 for pre-quantized x.");
        TORCH_CHECK(topoType.value_or(0) == 0, "pre-quantized x is supported only by the A5 MTE path (topo_type=0).");
        TORCH_CHECK(!dispatchQuantOutDtype.has_value(),
                    "dispatch_quant_out_dtype must be None for pre-quantized x; the dispatch dtype is inferred "
                    "from x.");
        TORCH_CHECK(!sharedExpertQuantOutDtype.has_value(),
                    "shared_expert_quant_out_dtype must be None for pre-quantized x; its dtype is inferred from x.");
    }
    aclDataType dispatchQuantAclDtype = dispatchQuantOutDtype.has_value() ?
                                            GetAclDataType(dispatchQuantOutDtype.value()) :
                                            ACL_DT_UNDEFINED;

    auto xSize = x.sizes();
    int64_t bs = xSize[0];
    int64_t h = xSize[1];
    if (isPackedFp4X) {
        TORCH_CHECK(h <= std::numeric_limits<int64_t>::max() / PACKED_FLOAT4_ELEMENTS_PER_BYTE,
                    "The packed FP4 hidden dimension is too large.");
        h *= PACKED_FLOAT4_ELEMENTS_PER_BYTE;
    }
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous.");
    if (hasScales) {
        const at::Tensor &scaleTensor = scales.value();
        const auto scaleScalarType = scaleTensor.scalar_type();
        TORCH_CHECK(scaleScalarType == at::kByte || scaleScalarType == at::ScalarType::Float8_e8m0fnu,
                    "scales only supports torch.uint8 storage or torch.float8_e8m0fnu.");
        TORCH_CHECK(scaleTensor.dim() == DIM_TWO, "scales must be 2D [BS, CeilDiv(H, 32)].");
        const int64_t expectedScaleWidth = (h + MX_SCALE_GROUP_SIZE - 1) / MX_SCALE_GROUP_SIZE;
        TORCH_CHECK(scaleTensor.size(0) == bs && scaleTensor.size(1) == expectedScaleWidth,
                    "scales shape must be [BS, CeilDiv(H, 32)] = [", bs, ", ", expectedScaleWidth, "], but got ",
                    scaleTensor.sizes(), ".");
        TORCH_CHECK(scaleTensor.device() == x.device(), "scales must be on the same device as x.");
        TORCH_CHECK(scaleTensor.is_contiguous(), "scales must be contiguous.");
    }
    if (maskBuffer.has_value()) {
        const at::Tensor &mask = maskBuffer.value();
        TORCH_CHECK(mask.scalar_type() == at::kInt, "mask_buffer dtype must be int32.");
        TORCH_CHECK(mask.dim() == 1 && mask.numel() == epWorldSize, "mask_buffer shape must be [ep_world_size].");
        TORCH_CHECK(mask.device() == x.device(), "mask_buffer must be on the same device as x.");
        TORCH_CHECK(mask.is_contiguous(), "mask_buffer must be contiguous.");
    }

    at::TensorList weight1Ref = weight1;
    at::TensorList weight2Ref = weight2;

    auto toTensorList = [](const c10::optional<std::vector<at::Tensor>> &opt) -> at::TensorList {
        return opt.has_value() ? at::TensorList(opt.value()) : at::TensorList();
    };
    at::TensorList weightScales1Ref = toTensorList(weightScales1);
    at::TensorList weightScales2Ref = toTensorList(weightScales2);
    at::TensorList bias1Ref = toTensorList(bias1);
    at::TensorList bias2Ref = toTensorList(bias2);

    aclDataType weight1RefDtype = weight1Type.has_value() ? GetAclDataType(weight1Type.value()) :
                                                            ConvertToAclDataType(weight1Ref[0].scalar_type());
    aclDataType weightScales1Dtype;
    if (weight1RefDtype == aclDataType::ACL_FLOAT8_E5M2 || weight1RefDtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight1RefDtype == aclDataType::ACL_FLOAT4_E2M1) {
        weightScales1Dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weightScales1Dtype = aclDataType::ACL_UINT64;
    }

    aclDataType weight2RefDtype = weight2Type.has_value() ? GetAclDataType(weight2Type.value()) :
                                                            ConvertToAclDataType(weight2Ref[0].scalar_type());
    aclDataType weightScales2Dtype;
    if (weight2RefDtype == aclDataType::ACL_FLOAT8_E5M2 || weight2RefDtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight2RefDtype == aclDataType::ACL_FLOAT4_E2M1) {
        weightScales2Dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weightScales2Dtype = aclDataType::ACL_UINT64;
    }

    const bool dispatchQuantOutIsFp4 =
        dispatchQuantOutDtype.has_value() && dispatchQuantOutDtype.value() == static_cast<int64_t>(DType::FLOAT4_E2M1);
    const bool sharedQuantOutIsFp4 = sharedExpertQuantOutDtype.has_value() ?
                                         sharedExpertQuantOutDtype.value() == static_cast<int64_t>(DType::FLOAT4_E2M1) :
                                         dispatchQuantOutIsFp4;
    if (dispatchQuantOutIsFp4 || sharedQuantOutIsFp4) {
        TORCH_CHECK(h % 2 == 0, "The last dim input shape must be divisible by 2 if "
                                "an expert quant output type is torch_npu.float4_e2m1");
    }

    int64_t localMoeExpertNum = 1;
    localMoeExpertNum = moeExpertNum / epWorldSize;
    at::Tensor expertTokenNums;
    expertTokenNums = at::empty({localMoeExpertNum}, x.options().dtype(at::kInt));

    std::string commAlgStr = std::string(commAlg);
    char *commAlgPtr = const_cast<char *>(commAlg.c_str());

    std::string activationStr = std::string(activation);
    char *activationPtr = const_cast<char *>(activationStr.c_str());

    // The eager and graph Python paths validate and serialize activationParams before calling C++.
    int64_t topoTypeValue = topoType.value_or(0);
    int64_t rankNumPerServerValue = rankNumPerServer.value_or(2);

    // 这里传给 ACLNN 的是 GE dtype 属性；GE DT_UNDEFINED 为 28，与 ACL_DT_UNDEFINED=-1 不同。
    int64_t dispatchQuantResultType = dispatchQuantOutDtype.has_value() ?
                                                  static_cast<int64_t>(dispatchQuantAclDtype) :
                                                  GE_DTYPE_UNDEFINED;
    int64_t sharedExpertQuantResultType = sharedExpertQuantOutDtype.has_value() ?
                                              static_cast<int64_t>(GetAclDataType(sharedExpertQuantOutDtype.value())) :
                                              dispatchQuantResultType;

    at::ScalarType outputScalarType = isPreQuantizedX ? at::kBFloat16 : xScalarType;
    at::Tensor y = at::empty({bs, h}, x.options().dtype(outputScalarType));

    at::Tensor scalesTensor = hasScales ? scales.value() : at::Tensor();
    TensorWrapper xWrapper = {x, xAclDtype};
    TensorWrapper scalesWrapper = {scalesTensor, hasScales ? ACL_FLOAT8_E8M0 : ACL_DT_UNDEFINED};

    TensorListWrapper weight1Wrapper = {weight1Ref, weight1RefDtype};
    TensorListWrapper weight2Wrapper = {weight2Ref, weight2RefDtype};
    TensorListWrapper weightScales1Wrapper = {weightScales1Ref, weightScales1Dtype};
    TensorListWrapper weightScales2Wrapper = {weightScales2Ref, weightScales2Dtype};
    TensorListWrapper bias1Wrapper = {bias1Ref, aclDataType::ACL_FLOAT};
    TensorListWrapper bias2Wrapper = {bias2Ref, aclDataType::ACL_FLOAT};

    at::TensorList sharedWeight1Ref = toTensorList(sharedWeight1);
    at::TensorList sharedWeight2Ref = toTensorList(sharedWeight2);
    at::TensorList sharedWeightScales1Ref = toTensorList(sharedWeightScales1);
    at::TensorList sharedWeightScales2Ref = toTensorList(sharedWeightScales2);
    at::TensorList sharedBias1Ref = toTensorList(sharedBias1);
    at::TensorList sharedBias2Ref = toTensorList(sharedBias2);

    aclDataType sharedWeight1RefDtype =
        sharedWeight1Type.has_value() ?
            GetAclDataType(sharedWeight1Type.value()) :
            (sharedWeight1Ref.size() > 0 ? ConvertToAclDataType(sharedWeight1Ref[0].scalar_type()) : weight1RefDtype);
    aclDataType sharedWeight2RefDtype =
        sharedWeight2Type.has_value() ?
            GetAclDataType(sharedWeight2Type.value()) :
            (sharedWeight2Ref.size() > 0 ? ConvertToAclDataType(sharedWeight2Ref[0].scalar_type()) : weight2RefDtype);
    const auto getWeightScaleDtype = [](aclDataType weightDtype) {
        return weightDtype == aclDataType::ACL_FLOAT8_E5M2 || weightDtype == aclDataType::ACL_FLOAT8_E4M3FN ||
                       weightDtype == aclDataType::ACL_FLOAT4_E2M1 ?
                   aclDataType::ACL_FLOAT8_E8M0 :
                   aclDataType::ACL_UINT64;
    };

    TensorListWrapper sharedWeight1Wrapper = {sharedWeight1Ref, sharedWeight1RefDtype};
    TensorListWrapper sharedWeight2Wrapper = {sharedWeight2Ref, sharedWeight2RefDtype};
    TensorListWrapper sharedWeightScales1Wrapper = {sharedWeightScales1Ref, getWeightScaleDtype(sharedWeight1RefDtype)};
    TensorListWrapper sharedWeightScales2Wrapper = {sharedWeightScales2Ref, getWeightScaleDtype(sharedWeight2RefDtype)};
    TensorListWrapper sharedBias1Wrapper = {sharedBias1Ref, aclDataType::ACL_FLOAT};
    TensorListWrapper sharedBias2Wrapper = {sharedBias2Ref, aclDataType::ACL_FLOAT};

    ACLNN_CMD(aclnnMegaMoe, context, xWrapper, topkIds, topkWeights, weight1Wrapper, weight2Wrapper,
              weightScales1Wrapper, weightScales2Wrapper, bias1Wrapper, bias2Wrapper, xActiveMask, scalesWrapper,
              sharedWeight1Wrapper, sharedWeight2Wrapper, sharedWeightScales1Wrapper, sharedWeightScales2Wrapper,
              sharedBias1Wrapper, sharedBias2Wrapper, maskBuffer, moeExpertNum, epWorldSize, cclBufferSize,
              maxRecvTokenNum, dispatchQuantMode, dispatchQuantResultType, sharedExpertQuantResultType,
              combineQuantMode, commAlgPtr, numMaxTokensPerRank, activationPtr, activationParams, topoTypeValue,
              rankNumPerServerValue, topkWeightsType, y, expertTokenNums);

    return std::tie(y, expertTokenNums);
}

namespace {
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int64_t MB_SIZE = 1024LL * 1024LL;
constexpr int64_t RESERVED_SPACE_SIZE = 10LL * 1024 * 1024;
constexpr int64_t MAX_EXPERTS_PER_RANK_A2A3 = 128LL;
constexpr int64_t SYNC_STATE_RESERVED_SIZE = 512LL * 1024;
constexpr int64_t PEERMEM_MIN_RANK_SYNC_SIZE = 48LL * 1024LL;
constexpr int64_t PEERMEM_SYNC_COUNT_REGION_SIZE = 12LL * 1024LL;
constexpr int64_t PEERMEM_SYNC_SLOT_SIZE = 64LL;
constexpr int64_t MXFP_SCALE_GROUP_NUM = 32LL;
constexpr int64_t MXFP_MULTI_BASE_SIZE = 2LL;
constexpr int64_t Y_DTYPE_SIZE = 2LL;
constexpr int64_t URMA_H_ALIGN = 1024LL;
constexpr int64_t MOE_PERMUTE_CHUNK = 1024LL;
// 异常 Dump 区
constexpr int64_t EXCEPTION_DUMP_REGION_SIZE = 60LL * 1024LL;
// rankSyncInWorld 同步区
constexpr int64_t PEERMEM_DATA_OFFSET = 60LL * 1024LL;

int64_t CeilAlign(int64_t val, int64_t align)
{
    return (val + align - 1) / align * align;
}

// A2 minimum buffer size (MB).
// Matches tiling_arch22.cpp CalcLeastCclBufferSize with isA3=false.
int64_t CalcLeastCclBufferSizeA2(int64_t maxRecvTokenNum, int64_t h, int64_t epWorldSize, bool isQuantRouting,
                                 int64_t bs, int64_t topK)
{
    // Data block 1: TokenPerExpert
    // EP × CeilAlign(EP × MAX_EXPERTS_PER_RANK_A2A3 + 1, 128) × 4B
    int64_t offsetTokenPerExpert = epWorldSize * CeilAlign(epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 + 1, ALIGN_128) *
                                   static_cast<int64_t>(sizeof(int32_t));

    // Data block 2: tensors
    // ===== winIn =====
    int64_t offsetAAfterDispatch =
        maxRecvTokenNum * (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
    int64_t offsetD = bs * topK * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t winInTensorSize = offsetAAfterDispatch + offsetD;

    // ===== winOut =====
    // winOut A 区按单 chunk（不超过 bs）的 permute 输出分配，逐 chunk 复用，
    // 与 tiling_arch22.cpp CalcLeastCclBufferSize 的 A2 分支一致
    const int64_t chunkTokensA2 = std::min(bs, MOE_PERMUTE_CHUNK);
    int64_t offsetA =
        chunkTokensA2 * topK * (!isQuantRouting ? h * static_cast<int64_t>(sizeof(int16_t)) : (h + ALIGN_512));
    int64_t offsetC = maxRecvTokenNum * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t winOutTensorSize = offsetA + offsetC;
    int64_t offsetTensor = std::max(winInTensorSize, winOutTensorSize);
    if (isQuantRouting) {
        offsetTensor += maxRecvTokenNum * static_cast<int64_t>(sizeof(float));
    }

    // Data block 3: sync flags
    int64_t offsetFlag = epWorldSize * ALIGN_512;                 // CrossRankSync
    offsetFlag += epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 * 64LL; // DispatchFlag
    offsetFlag += epWorldSize * 64LL;                             // AllGatherFlag

    return (offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE + MB_SIZE) / MB_SIZE;
}

// A3 minimum buffer size (MB).
// Matches tiling_arch22.cpp CalcLeastCclBufferSize with isA3=true.
int64_t CalcLeastCclBufferSizeA3(int64_t h, int64_t epWorldSize, bool isQuantRouting, int64_t bs, int64_t topK)
{
    // Data block 1: TokenPerExpert
    // EP × CeilAlign(EP × MAX_EXPERTS_PER_RANK_A2A3 + 1, 128) × 4B
    int64_t offsetTokenPerExpert = epWorldSize * CeilAlign(epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 + 1, ALIGN_128) *
                                   static_cast<int64_t>(sizeof(int32_t));

    // Data block 2: tensors (winIn only, no winOut)
    // Window layout is consistent with kernel PeermemInfo: number of rows per chunk (not exceeding bs)
    const int64_t chunkTokens = std::min(bs, MOE_PERMUTE_CHUNK);
    int64_t offsetAAfterDispatch =
        chunkTokens * topK * (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
    int64_t offsetD = chunkTokens * topK * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t offsetTensor = offsetAAfterDispatch + offsetD;
    if (isQuantRouting) {
        offsetTensor += chunkTokens * topK * static_cast<int64_t>(sizeof(float));
    }

    // Data block 3: sync flags
    int64_t offsetFlag = std::max(epWorldSize * ALIGN_512, SYNC_STATE_RESERVED_SIZE);

    return (offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE + MB_SIZE) / MB_SIZE;
}

// The Torch JIT wheel does not package op_kernel headers. Keep this pure sizing mirror synchronized with
// mc2/mega_moe/op_kernel/arch35/common/mega_moe_peermem.h, which remains the host/device layout source of truth.
int64_t CalcTokenScaleBytesA5(int64_t hidden, int64_t numTopk, int64_t topkWeightsType)
{
    int64_t mxScaleNum = (hidden + ALIGN_32 - 1) / ALIGN_32;
    int64_t dataBytes = CeilAlign(hidden, ALIGN_256);
    int64_t tokenBytes = CeilAlign(dataBytes + mxScaleNum, ALIGN_32);
    if (topkWeightsType == 1) {
        int64_t weightBytes = CeilAlign(numTopk * static_cast<int64_t>(sizeof(float)), ALIGN_32);
        tokenBytes = CeilAlign(tokenBytes + weightBytes, ALIGN_32);
    }
    return tokenBytes;
}

int64_t CalcCombineTokenBytesA5(int64_t hidden, int64_t combineQuantMode)
{
    if (combineQuantMode == 0) {
        return hidden * Y_DTYPE_SIZE;
    }
    int64_t tokenStorageBytes = CeilAlign(hidden, ALIGN_256);
    int64_t scaleCount = (hidden + MXFP_SCALE_GROUP_NUM - 1) / MXFP_SCALE_GROUP_NUM;
    int64_t storedScaleBytes = CeilAlign(scaleCount, MXFP_MULTI_BASE_SIZE);
    return CeilAlign(tokenStorageBytes + storedScaleBytes, ALIGN_32);
}

// Preserve the A5 MTE sizing baseline, including its non-quant combine width and exception-dump prefix.
int64_t CalcMteCclBufferSizeA5(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                               int64_t hidden, int64_t topkWeightsType)
{
    int64_t expertPerRank = moeExpertNum / epWorldSize;

    // Compact route-index receive area.
    int64_t routeIndexAlignSize = CeilAlign(numMaxTokensPerRank * static_cast<int64_t>(sizeof(int32_t)), ALIGN_32);
    int64_t routeRecvSize = CeilAlign(expertPerRank * epWorldSize * routeIndexAlignSize, ALIGN_512);

    // Expert-major raw count table: [localExpert][sourceRank].
    int64_t expertCountRecvSize =
        CeilAlign(expertPerRank * epWorldSize * static_cast<int64_t>(sizeof(int32_t)), ALIGN_512);

    int64_t tokenBytes = CalcTokenScaleBytesA5(hidden, numTopk, topkWeightsType);
    int64_t dispatchRecordAreaSize = CeilAlign(numMaxTokensPerRank * tokenBytes, ALIGN_512);

    int64_t combineSendSize = CeilAlign(numMaxTokensPerRank * numTopk * hidden * Y_DTYPE_SIZE, ALIGN_512);

    int64_t totalBytes = EXCEPTION_DUMP_REGION_SIZE + PEERMEM_DATA_OFFSET + routeRecvSize + expertCountRecvSize +
                         dispatchRecordAreaSize + combineSendSize;

    return totalBytes;
}

// A5 URMA peermem minimum size in bytes. The symmetric layout uses capacity, never the current-rank BS.
int64_t CalcUrmaCclBufferSizeA5(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                                int64_t hidden, int64_t combineQuantMode, int64_t topkWeightsType, int64_t serverNum)
{
    int64_t expertPerRank = moeExpertNum / epWorldSize;
    int64_t rankSyncSize = epWorldSize * PEERMEM_SYNC_SLOT_SIZE;
    int64_t dataOffset =
        CeilAlign(std::max(rankSyncSize, PEERMEM_MIN_RANK_SYNC_SIZE) + PEERMEM_SYNC_COUNT_REGION_SIZE, ALIGN_512);

    int64_t routeCapacity = numMaxTokensPerRank * numTopk;
    int64_t alignedRouteCount = CeilAlign(routeCapacity * static_cast<int64_t>(sizeof(int32_t)), ALIGN_256) /
                                static_cast<int64_t>(sizeof(int32_t));
    int64_t maskAlignSize = CeilAlign(alignedRouteCount / 8, ALIGN_32);
    int64_t maskRecvSize = CeilAlign(expertPerRank * epWorldSize * (maskAlignSize + ALIGN_32), ALIGN_512);
    int64_t expertCountRecvSize =
        CeilAlign(expertPerRank * epWorldSize * static_cast<int64_t>(sizeof(int32_t)), ALIGN_512);

    int64_t tokenBytes = CalcTokenScaleBytesA5(hidden, numTopk, topkWeightsType);
    int64_t relayRecordBytes = CeilAlign(tokenBytes, ALIGN_512);
    int64_t relayDataSize = CeilAlign(numMaxTokensPerRank * relayRecordBytes * serverNum, ALIGN_512);
    int64_t relayFlagSize =
        CeilAlign(serverNum * numMaxTokensPerRank * static_cast<int64_t>(sizeof(uint64_t)), ALIGN_512);

    int64_t combineTokenBytes = CalcCombineTokenBytesA5(hidden, combineQuantMode);
    int64_t combineSendSize = CeilAlign(numMaxTokensPerRank * numTopk * combineTokenBytes, ALIGN_512);

    return dataOffset + maskRecvSize + expertCountRecvSize + relayDataSize + relayFlagSize + combineSendSize;
}
} // namespace

int64_t GetMegaMoeCclBufferSize(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                                int64_t hidden, int64_t maxRecvTokenNum, int64_t dispatchQuantMode,
                                c10::optional<int64_t> dispatchQuantOutDtype, int64_t combineQuantMode,
                                std::string commAlg, int64_t topkWeightsType, int64_t serverNum)
{
    // Zero selects the initial MTE layout; a confirmed cross-server context supplies the actual count (> 1).
    TORCH_CHECK(serverNum >= 0, "server_num must be non-negative, but got ", serverNum);
    const char *socName = aclrtGetSocName();
    bool isA2 = (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr);
    bool isA3 = (socName != nullptr && std::strstr(socName, "Ascend910_93") != nullptr);
    if (isA2 || isA3) {
        TORCH_CHECK(serverNum == 0, "server_num is only supported by the Ascend950 channel backend");
        TORCH_CHECK(epWorldSize == 2 || epWorldSize == 4 || epWorldSize == 8 || epWorldSize == 16 ||
                        epWorldSize == 32 || epWorldSize == 48 || epWorldSize == 64 || epWorldSize == 96 ||
                        epWorldSize == 128,
                    "ep_world_size only support {2, 4, 8, 16, 32, 48, 64, 96, 128} on A2/A3, but got ", epWorldSize);
        TORCH_CHECK(hidden >= 1024 && hidden <= 8192 && hidden % 512 == 0,
                    "hidden only support [1024, 8192] and hidden % 512 == 0 on A2/A3, but got ", hidden);
        TORCH_CHECK(numMaxTokensPerRank >= 1, "num_max_tokens_per_rank should be >= 1 on A2/A3, but got ",
                    numMaxTokensPerRank);
        TORCH_CHECK(moeExpertNum >= 1 && moeExpertNum <= 2048,
                    "moe_expert_num only support [1, 2048] on A2/A3, but got ", moeExpertNum);
        TORCH_CHECK(numTopk >= 1 && numTopk <= 16, "num_topk only support [1, 16] on A2/A3, but got ", numTopk);
        TORCH_CHECK(dispatchQuantMode == 0 || dispatchQuantMode == 2 || dispatchQuantMode == 4,
                    "dispatch_quant_mode only support {0, 2, 4} on A2/A3, but got ", dispatchQuantMode);

        bool isQuantRouting = (dispatchQuantMode == 4);
        // max_recv_token_num 为 0 时自动计算为 bs * epWorldSize * min(topK, expertPerRank)，
        if (maxRecvTokenNum == 0) {
            int64_t expertPerRank = moeExpertNum / epWorldSize;
            maxRecvTokenNum = numMaxTokensPerRank * epWorldSize * std::min(numTopk, expertPerRank);
        }
        if (isA3) {
            return CalcLeastCclBufferSizeA3(hidden, epWorldSize, isQuantRouting, numMaxTokensPerRank, numTopk);
        }
        return CalcLeastCclBufferSizeA2(maxRecvTokenNum, hidden, epWorldSize, isQuantRouting, numMaxTokensPerRank,
                                        numTopk);
    }

    // A5 / 950 checks are aligned with the arch35 host tiling contract.
    TORCH_CHECK(epWorldSize >= 2 && epWorldSize <= 1024, "ep_world_size only support in [2, 1024], but got ",
                epWorldSize);
    TORCH_CHECK(hidden >= 1024 && hidden <= 8192, "hidden only support in [1024, 8192], but got ", hidden);
    int64_t hiddenAlignment = serverNum > 0 ? URMA_H_ALIGN : ALIGN_32;
    TORCH_CHECK(hidden % hiddenAlignment == 0, "hidden must be a multiple of ", hiddenAlignment,
                " for the selected communication topology, but got ", hidden);
    TORCH_CHECK(
        numMaxTokensPerRank >= 1 && static_cast<uint64_t>(numMaxTokensPerRank) <= std::numeric_limits<uint32_t>::max(),
        "num_max_tokens_per_rank should be in [1, UINT32_MAX], but got ", numMaxTokensPerRank);
    TORCH_CHECK(maxRecvTokenNum >= 0, "max_recv_token_num should be non-negative, but got ", maxRecvTokenNum);
    TORCH_CHECK(moeExpertNum >= epWorldSize && moeExpertNum <= 2048 && moeExpertNum % epWorldSize == 0,
                "moe_expert_num should be in [ep_world_size, 2048] and divisible by ep_world_size, but got ",
                moeExpertNum, " and ep_world_size ", epWorldSize);
    TORCH_CHECK(numTopk >= 1 && numTopk <= 32, "num_topk only support in [1, 32], but got ", numTopk);
    TORCH_CHECK(topkWeightsType == 0 || topkWeightsType == 1, "topk_weights_type only support 0 or 1, but got ",
                topkWeightsType);
    TORCH_CHECK(combineQuantMode == 0 || combineQuantMode == 3 || combineQuantMode == 4,
                "combine_quant_mode only support 0, 3 or 4 on Ascend950, but got ", combineQuantMode);

    if (serverNum > 0) {
        TORCH_CHECK(serverNum > 1 && serverNum <= epWorldSize && epWorldSize % serverNum == 0,
                    "server_num should be in [2, ep_world_size] and divide ep_world_size, but got ", serverNum,
                    " and ep_world_size ", epWorldSize);
        int64_t routeCapacity = numMaxTokensPerRank * numTopk;
        TORCH_CHECK(routeCapacity <= std::numeric_limits<int32_t>::max(),
                    "num_max_tokens_per_rank * num_topk should be <= INT32_MAX for URMA, but got ", routeCapacity);
        int64_t expertPerRank = moeExpertNum / epWorldSize;
        int64_t maxOutputCapacity = numMaxTokensPerRank * epWorldSize * std::min(numTopk, expertPerRank);
        TORCH_CHECK(maxOutputCapacity <= std::numeric_limits<int32_t>::max(),
                    "maximum receive token capacity should be <= INT32_MAX for URMA, but got ", maxOutputCapacity);
        TORCH_CHECK(maxRecvTokenNum <= maxOutputCapacity,
                    "max_recv_token_num should not exceed the URMA maximum receive token capacity ", maxOutputCapacity,
                    ", but got ", maxRecvTokenNum);
        return CalcUrmaCclBufferSizeA5(epWorldSize, moeExpertNum, numMaxTokensPerRank, numTopk, hidden,
                                       combineQuantMode, topkWeightsType, serverNum);
    }
    return CalcMteCclBufferSizeA5(epWorldSize, moeExpertNum, numMaxTokensPerRank, numTopk, hidden, topkWeightsType);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_mega_moe", &NpuMegaMoe, "npu_mega_moe");
    m.def("get_mega_moe_ccl_buffer_size", &GetMegaMoeCclBufferSize, "get_mega_moe_ccl_buffer_size",
          py::arg("ep_world_size"), py::arg("moe_expert_num"), py::arg("num_max_tokens_per_rank"), py::arg("num_topk"),
          py::arg("hidden"), py::arg("max_recv_token_num"), py::arg("dispatch_quant_mode"),
          py::arg("dispatch_quant_out_dtype"), py::arg("combine_quant_mode"), py::arg("comm_alg"),
          py::arg("topk_weights_type"), py::arg("server_num") = 0);
}

} // namespace op_api
