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
 * \file moe_distribute_dispatch.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;
const int DIM_ONE = 1;
const int DIM_TWO = 2;

/**
 * @brief Warpper for moe_distribute_dispatch
 */
tensor_list NpuMoeDistributeDispatch(
    const at::Tensor &context, const at::Tensor &x, const at::Tensor &expertIds, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, int64_t cclBufferSize, const c10::optional<at::Tensor> &scales,
    const c10::optional<at::Tensor> &xActiveMask, const c10::optional<at::Tensor> &expertScales,
    const c10::optional<at::Tensor> &elasticInfo, const c10::optional<at::Tensor> &performanceInfo, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode,
    int64_t globalBs, int64_t expertTokenNumsType, std::string commAlg, int64_t zeroExpertNum, int64_t copyExpertNum,
    int64_t constExpertNum, c10::optional<int64_t> yDtype, c10::optional<int64_t> xDtype,
    c10::optional<int64_t> scalesDtype)
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

    auto outputDtype = at::kChar;
    if (quantMode == QuantMode::QUANT_MODE_NO_QUANT) {
        outputDtype = x.scalar_type();
    }
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
        if (tpWorldSize == 0) {
            dynamicScales = at::empty({a}, x.options().dtype(at::kFloat));
        } else {
            dynamicScales = at::empty({a * tpWorldSize}, x.options().dtype(at::kFloat));
        }

        expertTokenNums = at::empty({localMoeExpertNum}, x.options().dtype(at::kLong));
        if (expertScales.has_value() && expertScales.value().defined()) {
            // 2: 2 buffer, 8 ranknum per server
            epRecvCntNum = epWorldSize * localMoeExpertNum + 2 * globalBsReal * k * (epWorldSize / 8);
        }
        epRecvCounts = at::empty({epRecvCntNum}, x.options().dtype(at::kInt));
        tpRecvCounts = at::empty({tpWorldSize}, x.options().dtype(at::kInt));
        assistInfoForcombine = at::empty({std::max(bs * k, a * 128)}, x.options().dtype(at::kInt));
        expandScales = at::empty({a}, x.options().dtype(at::kFloat));
    }

    std::string commAlgStr = std::string(commAlg);
    char *commAlgPtr = const_cast<char *>(commAlgStr.c_str());

    ACLNN_CMD(aclnnMoeDistributeDispatchV5, context, x, expertIds, scales, xActiveMask, expertScales, elasticInfo,
              performanceInfo, epWorldSize, epRankId, moeExpertNum, cclBufferSize, tpWorldSize, tpRankId,
              expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBsReal, expertTokenNumsType,
              commAlgPtr, zeroExpertNum, copyExpertNum, constExpertNum, expandX, dynamicScales, assistInfoForcombine,
              expertTokenNums, epRecvCounts, tpRecvCounts, expandScales);

    return std::tie(expandX, dynamicScales, assistInfoForcombine, expertTokenNums, epRecvCounts, tpRecvCounts,
                    expandScales);
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_distribute_dispatch", &NpuMoeDistributeDispatch, "moe_distribute_dispatch");
}
} // namespace op_api
