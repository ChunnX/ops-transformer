/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bandwidth_test.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor>;
const int DIM_TWO = 2;

tensor_list npu_bandwidth_test(const at::Tensor &x, const at::Tensor &dstrank_id,
                               std::string group, int64_t world_size, int64_t max_bs,
                               int64_t mode, std::string comm_alg, int64_t aiv_num)
{
    TORCH_CHECK((x.dim() == DIM_TWO), "The x should be 2D");
    TORCH_CHECK((dstrank_id.dim() == 1), "The dstrank_id should be 1D");
    TORCH_CHECK((world_size > 0), "The world_size should be greater than 0, current is: ", world_size);
    TORCH_CHECK((max_bs > 0), "The max_bs should be greater than 0, current is: ", max_bs);
    TORCH_CHECK((mode == 0 || mode == 1), "The mode should be 0 or 1, current is: ", mode);

    auto x_size = x.sizes();
    int64_t h = x_size[1];

    int64_t max_recv_cnt = max_bs * world_size;
    
    at::Tensor y{nullptr};
    at::Tensor receive_cnt{nullptr};
    {
        auto localDevice = c10::Device(x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        y = at::empty({max_recv_cnt, h}, x.options());
        receive_cnt = at::empty({world_size}, x.options().dtype(at::kInt));
    }
    std::string group_str = std::string(group);
    std::string comm_alg_str = std::string(comm_alg);
    char *group_ptr = const_cast<char *>(group_str.c_str());
    char *comm_alg_ptr = const_cast<char *>(comm_alg_str.c_str());

    ACLNN_CMD(aclnnBandwidthTest, x, dstrank_id, group_ptr, world_size,
              max_bs, mode, comm_alg_ptr, aiv_num, y, receive_cnt);

    return std::tie(y, receive_cnt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_bandwidth_test", &npu_bandwidth_test, "bandwidth_test");
}
} // op_api