/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file latent_ep.cpp
 * \brief
 */

#include <algorithm>
#include <cstring>
#include <vector>
#include <acl/acl_base.h>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace LatentEp {
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
const int DIM_ONE = 1;
const int DIM_TWO = 2;
const int64_t RANK_NUM_PER_NODE = 8;
const int64_t SEND_COUNT_MEMORY_SIZE = 2;
const int64_t PACKED_FLOAT4_ELEMENTS_PER_BYTE = 2;

bool HasTensor(const c10::optional<at::Tensor> &tensor)
{
    return tensor.has_value() && tensor.value().defined();
}

aclDataType GetDynamicScalesDtype(const at::Tensor &x, const c10::optional<at::Tensor> &scales,
                                  c10::optional<int64_t> scales_dtype, int64_t quant_mode)
{
    aclDataType dynamic_scale_dtype = ACL_FLOAT;
    if (quant_mode == QuantMode::QUANT_MODE_NO_QUANT) {
        aclDataType scale_value_type =
            HasTensor(scales) ? ConvertToAclDataType(scales.value().scalar_type()) : ACL_FLOAT;
        dynamic_scale_dtype = (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) ?
                                  ACL_FLOAT :
                                  (scales_dtype.has_value() ? GetAclDataType(scales_dtype.value()) : scale_value_type);
    } else if (quant_mode == QuantMode::QUANT_MODE_MX || quant_mode == QuantMode::QUANT_MODE_MX_CLIP) {
        dynamic_scale_dtype = ACL_FLOAT8_E8M0;
    }
    return dynamic_scale_dtype;
}

std::vector<int64_t> GetDynamicScalesShape(const c10::optional<at::Tensor> &scales, int64_t quant_mode, int64_t a,
                                           int64_t h)
{
    constexpr int64_t perGroupSize = 128;
    constexpr int64_t mxQuantSize = 32;
    std::vector<int64_t> shape{a};
    if (quant_mode == QuantMode::QUANT_MODE_NO_QUANT && HasTensor(scales)) {
        TORCH_CHECK(scales.value().dim() >= DIM_TWO, "Expected scales to be at least 2D.");
        shape = {a, scales.value().sizes()[1]};
    } else if (quant_mode == QuantMode::QUANT_MODE_PERTOKEN) {
        shape = {a};
    } else if (quant_mode == QuantMode::QUANT_MODE_PERGROUP) {
        shape = {a, (h + perGroupSize - 1) / perGroupSize};
    } else if (quant_mode == QuantMode::QUANT_MODE_MX || quant_mode == QuantMode::QUANT_MODE_MX_CLIP) {
        shape = {a, ((h + mxQuantSize - 1) / mxQuantSize + 1) / 2 * 2};
    }
    return shape;
}

at::ScalarType GetStorageScalarType(aclDataType dtype)
{
    switch (dtype) {
        case ACL_UINT8:
        case ACL_INT4:
        case ACL_HIFLOAT8:
        case ACL_FLOAT8_E8M0:
        case ACL_FLOAT4_E2M1:
        case ACL_FLOAT4_E1M2:
            return at::kByte;
        case ACL_INT8:
            return at::kChar;
        case ACL_INT16:
            return at::kShort;
        case ACL_INT32:
            return at::kInt;
        case ACL_INT64:
            return at::kLong;
        case ACL_FLOAT16:
            return at::kHalf;
        case ACL_FLOAT:
            return at::kFloat;
        case ACL_DOUBLE:
            return at::kDouble;
        case ACL_BOOL:
            return at::kBool;
        case ACL_BF16:
            return at::kBFloat16;
        case ACL_FLOAT8_E5M2:
            return at::ScalarType::Float8_e5m2;
        case ACL_FLOAT8_E4M3FN:
            return at::ScalarType::Float8_e4m3fn;
        default:
            TORCH_CHECK(false, "unsupported acl dtype for tensor allocation.");
    }
    return at::kByte;
}

bool IsPackedFloat4(aclDataType dtype)
{
    return dtype == ACL_FLOAT4_E2M1 || dtype == ACL_FLOAT4_E1M2;
}

bool IsAscend950()
{
    static const char *soc_name = aclrtGetSocName();
    return soc_name != nullptr && std::strstr(soc_name, "Ascend950") != nullptr;
}

/**
 * @brief Warpper for moe_distribute_dispatch
 */
tensor_list dispatch(const at::Tensor &context, const at::Tensor &x, const at::Tensor &expertIds, int64_t epWorldSize,
                     int64_t epRankId, int64_t moeExpertNum, int64_t cclBufferSize,
                     const c10::optional<at::Tensor> &sharedExpertInput, const c10::optional<at::Tensor> &scales,
                     const c10::optional<at::Tensor> &xActiveMask, const c10::optional<at::Tensor> &expertScales,
                     const c10::optional<at::Tensor> &elasticInfo, const c10::optional<at::Tensor> &performanceInfo,
                     int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
                     int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType,
                     std::string commAlg, int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
                     c10::optional<int64_t> yDtype, c10::optional<int64_t> xDtype, c10::optional<int64_t> scalesDtype)
{
    TORCH_CHECK((x.dim() == DIM_TWO) && (expertIds.dim() == DIM_TWO), "The x and expert_ids should be 2D");
    TORCH_CHECK((epWorldSize > 0), "The ep_world_sizes should be greater than 0, current is: ", epWorldSize);
    TORCH_CHECK((epRankId >= 0) && (epRankId < epWorldSize), "ep_rank_id should be in [0, ep_world_size), but got",
                " ep_world_size: ", epWorldSize, ", ep_rank_id: ", epRankId, ". ");
    TORCH_CHECK((sharedExpertRankNum >= 0) && (sharedExpertRankNum < epWorldSize),
                "shared_expert_rank_num should be in [0, ep_world_size), but got", " ep_world_size: ", epWorldSize,
                ", shared_expert_rank_num: ", sharedExpertRankNum, ". ");
    bool isSharedDefault = ((sharedExpertNum == 1) && (sharedExpertRankNum == 0));
    bool isNoShared = ((sharedExpertNum == 0) && (sharedExpertRankNum == 0));
    bool isValidShared = ((sharedExpertNum > 0) && ((sharedExpertRankNum / sharedExpertNum) > 0) &&
                          ((sharedExpertRankNum % sharedExpertNum) == 0));
    TORCH_CHECK(isSharedDefault || isNoShared || isValidShared,
                "shared_expert_num and shared_expertrank_num have obvious value situations: "
                "1. shared_expert_num is 1, shared_expert_rank_num is 0; 2. shared_expert num is 0, "
                "shared_expert_rank_num is 0; 3. shared_expert_num in (0, shared_expert_rank_num] and "
                "shared_expert_rank_num % shared_expert_num = 0. but the current input value is ",
                " shared_expert_num: ", sharedExpertNum, ", shared_expert_rank_num: ", sharedExpertRankNum, ". ");
    TORCH_CHECK((expertTokenNumsType == 0) || (expertTokenNumsType == 1),
                "The expert_token_nums_type should be 0 or 1.");
    TORCH_CHECK(IsAscend950(), "latent_ep dispatch requires Ascend950.");
    const bool independent = HasTensor(sharedExpertInput);
    if (independent) {
        TORCH_CHECK(sharedExpertInput->dim() == DIM_TWO && sharedExpertInput->size(0) == x.size(0) &&
                        sharedExpertInput->size(1) == x.size(1) * 2 &&
                        sharedExpertInput->scalar_type() == x.scalar_type() &&
                        sharedExpertInput->device() == x.device(),
                    "shared_expert_input must be [BS, 2*H] with the same dtype/device as x.");
        TORCH_CHECK(tpWorldSize == 1 && tpRankId == 0 && !HasTensor(elasticInfo) && sharedExpertNum > 0 &&
                        sharedExpertRankNum > 0,
                    "Independent shared experts require TP=1 and shared ranks, without elastic_info.");
        TORCH_CHECK(x.size(1) >= 1024 && x.size(1) <= 4096 && commAlg == "fullmesh_v2",
                    "Independent shared dispatch requires latent H in [1024,4096] and fullmesh_v2.");
        TORCH_CHECK(quantMode == 0 || quantMode == 4 || quantMode == 5 || yDtype.has_value(),
                    "y_dtype is required for quant_mode 1, 2, 3.");
    }
    auto xSize = x.sizes();
    auto expertIdsSize = expertIds.sizes();

    int64_t bs = xSize[0];
    int64_t h = xSize[1];
    int64_t k = expertIdsSize[1];

    // a2 expert_shard_type、shared_expert_rank_num 应为0
    bool sharedFront = (expertShardType == 0);
    int64_t localMoeExpertNum = 1;
    int64_t globalBsReal = (globalBs == 0) ? (bs * epWorldSize) : globalBs;
    int64_t a = 0;
    int64_t epRecvCntNum = 0;
    bool isSharedExpert = (sharedFront && epRankId < sharedExpertRankNum);
    if (isSharedExpert) {
        localMoeExpertNum = 1;
        int64_t maxBs = globalBsReal / epWorldSize; // 前面已有拦截，保证ep_world_size > 0
        int64_t rankNumPerSharedExpert =
            sharedExpertRankNum / sharedExpertNum; // 前面已有拦截, 保证进入该分支时shared_expert_num > 0
        int64_t maxSharedGroupNum = (epWorldSize + rankNumPerSharedExpert - 1) / rankNumPerSharedExpert;
        a = maxBs * maxSharedGroupNum;
    } else {
        localMoeExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum);
        a = globalBsReal * std::min(localMoeExpertNum, k);
    }
    if (sharedFront && elasticInfo.has_value()) {
        if ((isSharedDefault) || (isNoShared)) {
            localMoeExpertNum = std::max(localMoeExpertNum, moeExpertNum / (epWorldSize - sharedExpertRankNum));
            a = globalBsReal * std::min(localMoeExpertNum, k);
        } else {
            int64_t maxBs = globalBsReal / epWorldSize;
            int64_t rankNumPerSharedExpert = sharedExpertRankNum / sharedExpertNum;
            int64_t maxSharedGroupNum = (epWorldSize + rankNumPerSharedExpert - 1) / rankNumPerSharedExpert;
            a = std::max(maxBs * maxSharedGroupNum,
                         globalBsReal * std::min(moeExpertNum / (epWorldSize - sharedExpertRankNum), k));
            localMoeExpertNum = std::max(localMoeExpertNum, moeExpertNum / (epWorldSize - sharedExpertRankNum));
        }
    }
    if (tpWorldSize == DIM_TWO) {
        epRecvCntNum = epWorldSize * localMoeExpertNum * tpWorldSize;
    } else {
        epRecvCntNum = epWorldSize * localMoeExpertNum;
    }

    aclDataType outputAclDtype = ACL_INT8;
    if (quantMode == QuantMode::QUANT_MODE_NO_QUANT) {
        outputAclDtype = ConvertToAclDataType(x.scalar_type());
    }
    if (yDtype.has_value()) {
        outputAclDtype = GetAclDataType(yDtype.value());
    }
    if (!yDtype.has_value() && (quantMode == 4 || quantMode == 5)) {
        outputAclDtype = ACL_FLOAT8_E4M3FN;
    }
    // All ranks use the same y_dtype for MoE communication, including shared expert ranks.
    // Preserve it before selecting this rank's local expand_x dtype.
    const int64_t moeAclDtype = static_cast<int64_t>(outputAclDtype);
    TORCH_CHECK(!independent || (quantMode != 4 && quantMode != 5) || outputAclDtype == ACL_FLOAT8_E4M3FN,
                "Independent shared MX quantization requires E4M3FN.");
    const bool independentSharedRank = independent && isSharedExpert;
    if (independentSharedRank) {
        outputAclDtype = ConvertToAclDataType(x.scalar_type());
        h *= 2;
    }
    auto outputDtype = GetStorageScalarType(outputAclDtype);
    bool hasExpertScales = HasTensor(expertScales);

    at::Tensor expandX{nullptr};
    at::Tensor dynamicScales{nullptr};
    at::Tensor assistInfoForcombine{nullptr};
    at::Tensor expertTokenNums{nullptr};
    at::Tensor epRecvCounts{nullptr};
    at::Tensor tpRecvCounts{nullptr};
    at::Tensor expandScales{nullptr};
    {
        auto localDevice = c10::Device(x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        expandX = at::empty({std::max(a, a * tpWorldSize), h}, x.options().dtype(outputDtype));
        if (yDtype.has_value() && IsPackedFloat4(outputAclDtype) && !HasTensor(scales)) {
            TORCH_CHECK(h % PACKED_FLOAT4_ELEMENTS_PER_BYTE == 0,
                        "The last dim input shape must be divisible by 2 if y dtype is packed float4.");
            expandX = at::empty({std::max(a, a * tpWorldSize), h / PACKED_FLOAT4_ELEMENTS_PER_BYTE},
                                x.options().dtype(outputDtype));
        }

        aclDataType dynamicScalesAclDtype =
            (independentSharedRank ? ACL_FLOAT : GetDynamicScalesDtype(x, scales, scalesDtype, quantMode));
        auto dynamicScalesDtype = GetStorageScalarType(dynamicScalesAclDtype);
        if (IsAscend950()) {
            auto dynamicScalesShape = independentSharedRank ?
                                          std::vector<int64_t>{a} :
                                          GetDynamicScalesShape(scales, quantMode, std::max(a, a * tpWorldSize), h);
            dynamicScales = at::empty(dynamicScalesShape, x.options().dtype(dynamicScalesDtype));
        } else if (tpWorldSize == 0) {
            dynamicScales = at::empty({a}, x.options().dtype(dynamicScalesDtype));
        } else {
            dynamicScales = at::empty({a * tpWorldSize}, x.options().dtype(dynamicScalesDtype));
        }

        expertTokenNums = at::empty({localMoeExpertNum}, x.options().dtype(at::kLong));
        if (hasExpertScales && !IsAscend950()) {
            epRecvCntNum = epWorldSize * localMoeExpertNum +
                           SEND_COUNT_MEMORY_SIZE * globalBsReal * k * (epWorldSize / RANK_NUM_PER_NODE);
        }
        epRecvCounts = at::empty({epRecvCntNum}, x.options().dtype(at::kInt));
        tpRecvCounts = at::empty({tpWorldSize}, x.options().dtype(at::kInt));
        assistInfoForcombine = at::empty({std::max(bs * k, a * 128)}, x.options().dtype(at::kInt));
        expandScales = at::empty({hasExpertScales ? a : 0}, x.options().dtype(at::kFloat));
    }

    std::string commAlgStr = std::string(commAlg);
    char *commAlgPtr = const_cast<char *>(commAlgStr.c_str());

    aclDataType xAclDtype = xDtype.has_value() ? GetAclDataType(xDtype.value()) : ConvertToAclDataType(x.scalar_type());
    at::Tensor scalesTensor = HasTensor(scales) ? scales.value() : at::Tensor();
    aclDataType scalesAclDtype =
        scalesDtype.has_value() ?
            GetAclDataType(scalesDtype.value()) :
            ConvertToAclDataType(scalesTensor.defined() ? scalesTensor.scalar_type() : at::kFloat);
    aclDataType dynamicScalesAclDtype =
        (independentSharedRank ? ACL_FLOAT : GetDynamicScalesDtype(x, scales, scalesDtype, quantMode));
    TensorWrapper xWrapper = {x, xAclDtype};
    TensorWrapper scalesWrapper = {scalesTensor, scalesAclDtype};
    TensorWrapper expandXWrapper = {expandX, outputAclDtype};
    TensorWrapper dynamicScalesWrapper = {dynamicScales, dynamicScalesAclDtype};

    ACLNN_CMD(aclnnMoeDistributeDispatchV6, context, xWrapper, expertIds, sharedExpertInput, scalesWrapper, xActiveMask,
              expertScales, elasticInfo, performanceInfo, epWorldSize, epRankId, moeExpertNum, cclBufferSize,
              tpWorldSize, tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBsReal,
              expertTokenNumsType, commAlgPtr, zeroExpertNum, copyExpertNum, constExpertNum, moeAclDtype,
              expandXWrapper, dynamicScalesWrapper, assistInfoForcombine, expertTokenNums, epRecvCounts, tpRecvCounts,
              expandScales);

    return std::tie(expandX, dynamicScales, assistInfoForcombine, expertTokenNums, epRecvCounts, tpRecvCounts,
                    expandScales);
}

std::tuple<at::Tensor, at::Tensor> combine(
    const at::Tensor &context, const at::Tensor &expandX, const at::Tensor &expertIds,
    const at::Tensor &assistInfoForCombine, const at::Tensor &epSendCounts, const at::Tensor &expertScales,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t cclBufferSize,
    const c10::optional<at::Tensor> &tpSendCounts, const c10::optional<at::Tensor> &xActiveMask,
    const c10::optional<at::Tensor> &expandScales, const c10::optional<at::Tensor> &sharedExpertX,
    const c10::optional<at::Tensor> &elasticInfo, const c10::optional<at::Tensor> &oriX,
    const c10::optional<at::Tensor> &constExpertAlpha1, const c10::optional<at::Tensor> &constExpertAlpha2,
    const c10::optional<at::Tensor> &constExpertV, const c10::optional<at::Tensor> &performanceInfo,
    int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t globalBs, int64_t commQuantMode, std::string commAlg, int64_t zeroExpertNum,
    int64_t copyExpertNum, int64_t constExpertNum, bool enableIndependentSharedExpert)
{
    TORCH_CHECK((expandX.dim() == DIM_TWO) && (expertIds.dim() == DIM_TWO), "The x and expert_ids should be 2D");
    TORCH_CHECK((expandX.scalar_type() == at::kBFloat16) || (expandX.scalar_type() == at::kHalf) ||
                    (expandX.scalar_type() == at::kInt),
                "dtype of expand_x should be BFloat16, Float16 or Int, but got " +
                    std::string(c10::toString(expandX.scalar_type())));
    TORCH_CHECK(expertIds.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got " + std::string(c10::toString(expertIds.scalar_type())));
    auto expandXSize = expandX.sizes();
    auto expertIdsSize = expertIds.sizes();

    int64_t bs = expertIdsSize[0];
    int64_t h = expandXSize[1];
    TORCH_CHECK(IsAscend950(), "latent_ep combine requires Ascend950.");
    if (enableIndependentSharedExpert) {
        TORCH_CHECK(
            tpWorldSize == 1 && tpRankId == 0 && !HasTensor(elasticInfo) && !HasTensor(sharedExpertX) &&
                sharedExpertNum > 0 && sharedExpertRankNum > 0 && sharedExpertRankNum < epWorldSize,
            "Independent shared combine requires TP=1 and shared ranks; shared_expert_x/elastic_info must be absent.");
        TORCH_CHECK(commQuantMode == 0 || commQuantMode == 2 || commQuantMode == 4,
                    "Independent shared combine supports communication quantization 0, 2, 4.");
        if (epRankId < sharedExpertRankNum) {
            TORCH_CHECK(h % 2 == 0, "Shared H must be even.");
            h /= 2;
        }
        TORCH_CHECK(h >= 1024 && h <= 4096, "Independent shared latent H must be in [1024,4096].");
    }
    int64_t globalBsReal = (globalBs == 0) ? (bs * epWorldSize) : globalBs;

    at::Tensor output;
    {
        auto localDevice = c10::Device(expandX.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        if (expandX.scalar_type() != at::kInt) {
            output = at::empty({bs, h}, expandX.options().dtype(expandX.scalar_type()));
        } else {
            output = at::empty({bs, h}, expandX.options().dtype(at::kBFloat16));
        }
    }
    at::Tensor sharedOutput =
        enableIndependentSharedExpert ? at::empty({bs, h * 2}, output.options()) : at::empty({0}, output.options());
    c10::optional<at::Tensor> nulltensor = c10::nullopt;
    int64_t outDtype = 0;
    int64_t groupListType = 0;

    std::string commAlgStr = std::string(commAlg);
    char *commAlgPtr = const_cast<char *>(commAlgStr.c_str());

    ACLNN_CMD(aclnnMoeDistributeCombineV6, context, expandX, expertIds, assistInfoForCombine, epSendCounts,
              expertScales, tpSendCounts, xActiveMask, nulltensor, nulltensor, nulltensor, expandScales, sharedExpertX,
              elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, epWorldSize,
              epRankId, moeExpertNum, cclBufferSize, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
              sharedExpertRankNum, globalBsReal, outDtype, commQuantMode, groupListType, commAlgPtr, zeroExpertNum,
              copyExpertNum, constExpertNum, enableIndependentSharedExpert, output, sharedOutput);

    return std::make_tuple(output, sharedOutput);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dispatch", &dispatch, "Dispatch latent and original tokens");
    m.def("combine", &combine, "Combine MoE and shared expert results");
}
} // namespace LatentEp
