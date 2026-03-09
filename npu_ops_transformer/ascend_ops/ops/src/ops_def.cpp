/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/extension.h>
#include <ATen/ATen.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dummy", []() { return "ascend_ops extension loaded"; });
}

TORCH_LIBRARY(ascend_ops, m) {
    m.def("detection(Tensor expand_x, str group_ep, int ep_world_size, int ep_rank_id, "
                "str group_detection, int detection_world_size, int detection_rank_id, str group_barrier, "
                "int barrier_world_size, int barrier_rank_id, int test_timeout = 150) -> Tensor");

    m.def("reset(Tensor detection_result, int ep_world_size, int ep_rank_id, "
          "str group_detection, int detection_world_size, int detection_rank_id, int needSync) -> bool");
}