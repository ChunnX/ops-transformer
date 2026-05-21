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
const int DIM_ONE = 1;
const int DIM_TWO = 2;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_notify_dispatch(const at::Tensor &sendData, const at::Tensor &tokenPerExpertData, int64_t sendCount,
                    int64_t numTokens, const std::string &commGroup, int64_t rankSize, int64_t rankId,
                    int64_t localRankSize, int64_t localRankId)
{
    TORCH_CHECK((rankSize > 0), "The rankSize should be greater than 0, current is: ", rankSize);
    TORCH_CHECK((localRankSize > 0), "The localRankSize should be greater than 0, current is: ", localRankSize);
    TORCH_CHECK((sendCount >= 0), "The sendCount should be >= 0, current is: ", sendCount);
    TORCH_CHECK((numTokens >= 0), "The numTokens should be >= 0, current is: ", numTokens);

    at::Tensor sendDataOffset = at::empty({rankSize}, sendData.options().dtype(at::kLong));
    at::Tensor recvData = at::empty({sendCount}, sendData.options().dtype(sendData.scalar_type()));
    at::Tensor totalRecvTokens = at::empty({1}, sendData.options().dtype(at::kLong));
    at::Tensor recvCount = at::empty({rankSize}, sendData.options().dtype(at::kLong));
    at::Tensor recvOffset = at::empty({rankSize}, sendData.options().dtype(at::kLong));
    at::Tensor maxBs = at::empty({1}, sendData.options().dtype(at::kLong));
    at::Tensor recvTokensPerExpert = at::empty({rankSize}, sendData.options().dtype(at::kLong));

    ACLNN_CMD(aclnnNotifyDispatch, sendData, tokenPerExpertData, sendCount, numTokens, commGroup.c_str(), rankSize,
              rankId, localRankSize, localRankId, sendDataOffset, recvData, totalRecvTokens, recvCount, recvOffset,
              maxBs, recvTokensPerExpert);

    return std::tie(sendDataOffset, recvData, totalRecvTokens, recvCount, recvOffset, maxBs, recvTokensPerExpert);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_notify_dispatch", &npu_notify_dispatch, "npu_notify_dispatch");
}

} // namespace op_api