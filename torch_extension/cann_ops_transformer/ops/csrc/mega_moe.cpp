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
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_TWO = 2;

std::tuple<at::Tensor, at::Tensor>
NpuMegaMoe(const at::Tensor &context, const at::Tensor &x, const at::Tensor &topkIds, const at::Tensor &topkWeights,
           const std::vector<at::Tensor> &weight1, const std::vector<at::Tensor> &weight2, int64_t moeExpertNum,
           int64_t epWorldSize, int64_t cclBufferSize, const c10::optional<std::vector<at::Tensor>> &weightScales1,
           const c10::optional<std::vector<at::Tensor>> &weightScales2,
           const c10::optional<std::vector<at::Tensor>> &bias1, const c10::optional<std::vector<at::Tensor>> &bias2,
           const c10::optional<at::Tensor> &xActiveMask, int64_t maxRecvTokenNum, int64_t dispatchQuantMode,
           int64_t combineQuantMode, std::string commAlg, int64_t numMaxTokensPerRank, std::string activation,
           c10::optional<float> activationClamp, c10::optional<int64_t> dispatchQuantOutDtype,
           c10::optional<int64_t> weight1Type, c10::optional<int64_t> weight2Type, c10::optional<int64_t> topoType,
           c10::optional<int64_t> rankNumPerServer)
{
    TORCH_CHECK((epWorldSize > 0), "The ep_world_sizes should be greater than 0, current is: ", epWorldSize);
    TORCH_CHECK((x.dim() == DIM_TWO) && (topkIds.dim() == DIM_TWO), "The x and topk_ids should be 2D");
    TORCH_CHECK(((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf)) &&
                    (topkIds.scalar_type() == at::kInt),
                "dtype of x should be bfloat16, float16, dtype of topk_ids should be int.");

    at::TensorList weight1Ref = weight1;
    at::TensorList weight2Ref = weight2;

    at::TensorList weightScales1Ref;
    if (weightScales1.has_value()) {
        weightScales1Ref = at::TensorList(weightScales1.value());
    } else {
        weightScales1Ref = at::TensorList();
    }
    at::TensorList weightScales2Ref;
    if (weightScales2.has_value()) {
        weightScales2Ref = at::TensorList(weightScales2.value());
    } else {
        weightScales2Ref = at::TensorList();
    }
    at::TensorList bias1Ref;
    if (bias1.has_value()) {
        bias1Ref = at::TensorList(bias1.value());
    } else {
        bias1Ref = at::TensorList();
    }
    at::TensorList bias2Ref;
    if (bias2.has_value()) {
        bias2Ref = at::TensorList(bias2.value());
    } else {
        bias2Ref = at::TensorList();
    }

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

    auto xSize = x.sizes();
    auto topkIdsSize = topkIds.sizes();
    int64_t bs = xSize[0];
    int64_t h = xSize[1];
    int64_t k = topkIdsSize[1];

    if ((dispatchQuantOutDtype.has_value()) &&
        (dispatchQuantOutDtype.value() == static_cast<int64_t>(DType::FLOAT4_E2M1))) {
        TORCH_CHECK(h % 2 == 0, "The last dim input shape must be divisible by 2 if "
                                "dispatch quant output type is torch_npu.float4_e2m1");
    }

    int64_t localMoeExpertNum = 1;
    localMoeExpertNum = moeExpertNum / epWorldSize;
    at::Tensor expertTokenNums;
    expertTokenNums = at::empty({localMoeExpertNum}, x.options().dtype(at::kInt));

    std::string commAlgStr = std::string(commAlg);
    char *commAlgPtr = const_cast<char *>(commAlg.c_str());

    std::string activationStr = std::string(activation);
    char *activationPtr = const_cast<char *>(activationStr.c_str());

    float activationClampValue = activationClamp.value_or(std::numeric_limits<float>::max());
    int64_t topoTypeValue = topoType.value_or(0);
    int64_t rankNumPerServerValue = rankNumPerServer.value_or(2);

    int64_t dispatchQuantResultType =
        dispatchQuantOutDtype.has_value() ? static_cast<int64_t>(GetAclDataType(dispatchQuantOutDtype.value())) : 28;

    at::Tensor y;
    y = at::empty({bs, h}, topkIds.options().dtype(x.scalar_type()));

    TensorListWrapper weight1Wrapper = {weight1Ref, weight1RefDtype};
    TensorListWrapper weight2Wrapper = {weight2Ref, weight2RefDtype};
    TensorListWrapper weightScales1Wrapper = {weightScales1Ref, weightScales1Dtype};
    TensorListWrapper weightScales2Wrapper = {weightScales2Ref, weightScales2Dtype};
    TensorListWrapper bias1Wrapper = {bias1Ref, aclDataType::ACL_FLOAT};
    TensorListWrapper bias2Wrapper = {bias2Ref, aclDataType::ACL_FLOAT};

    ACLNN_CMD(aclnnMegaMoe, context, x, topkIds, topkWeights, weight1Wrapper, weight2Wrapper, weightScales1Wrapper,
              weightScales2Wrapper, bias1Wrapper, bias2Wrapper, xActiveMask, moeExpertNum, epWorldSize, cclBufferSize,
              maxRecvTokenNum, dispatchQuantMode, dispatchQuantResultType, combineQuantMode, commAlgPtr,
              numMaxTokensPerRank, activationPtr, activationClampValue, topoTypeValue,
              rankNumPerServerValue, y, expertTokenNums);

    return std::tie(y, expertTokenNums);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_mega_moe", &NpuMegaMoe, "npu_mega_moe");
}

} // namespace op_api
