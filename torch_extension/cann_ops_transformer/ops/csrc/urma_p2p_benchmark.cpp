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
 * \file urma_p2p_benchmark.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
const int DIM_TWO = 2;

tensor_list npu_urma_p2p_benchmark(const at::Tensor &x,
                                   std::string group, int64_t world_size, int64_t max_bs,
                                   int64_t mode, int64_t ccl_buffer_size,
                                   int64_t dst_rank_id, int64_t aiv_num)
{
    TORCH_CHECK((x.dim() == DIM_TWO), "The x should be 2D");
    TORCH_CHECK((world_size > 0), "The world_size should be greater than 0, current is: ", world_size);
    TORCH_CHECK((max_bs > 0), "The max_bs should be greater than 0, current is: ", max_bs);
    TORCH_CHECK((mode == 0 || mode == 1 || mode == 2), "The mode should be 0, 1 or 2, current is: ", mode);

    auto x_size = x.sizes();
    int64_t h = x_size[1];

    int64_t max_recv_cnt = max_bs * world_size;

    at::Tensor y{nullptr};
    at::Tensor receive_cnt{nullptr};
    at::Tensor performance_info{nullptr};
    {
        auto localDevice = c10::Device(x.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        y = at::empty({max_recv_cnt, h}, x.options());
        receive_cnt = at::empty({world_size}, x.options().dtype(at::kInt));
        performance_info = at::empty({12}, x.options().dtype(at::kLong));  // 9个timing slot + 3个padding, DataCopy对齐
    }

    char *group_ptr = const_cast<char *>(group.c_str());

    ACLNN_CMD(aclnnUrmaP2pBenchmark, x, group_ptr, world_size,
              max_bs, mode, ccl_buffer_size, dst_rank_id, aiv_num, y, receive_cnt, performance_info);

    return std::tie(y, receive_cnt, performance_info);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_urma_p2p_benchmark", &npu_urma_p2p_benchmark, "urma_p2p_benchmark");
}
} // op_api