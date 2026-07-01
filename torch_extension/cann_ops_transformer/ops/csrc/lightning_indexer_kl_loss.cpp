/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

at::Tensor lightning_indexer_kl_loss(
    const at::Tensor &target_score,
    const at::Tensor &index_probs,
    double eps)
{
    TORCH_CHECK((eps > 0), "eps should be greater than 0, current is: ", eps);
    TORCH_CHECK((target_score.scalar_type() == index_probs.scalar_type()),
        "target_score and index_probs should have the same dtype.");
    TORCH_CHECK((target_score.dim() >= 2 && target_score.dim() <= 3),
        "target_score.dim() should be in [2, 3], current is: ", target_score.dim());
    TORCH_CHECK((index_probs.dim() >= 2 && index_probs.dim() <= 3),
        "index_probs.dim() should be in [2, 3], current is: ", index_probs.dim());

    at::Tensor loss{nullptr};
    {
        auto local_device = c10::Device(target_score.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        // 输出为标量，分配一个单元素 tensor
        loss = at::empty({}, target_score.options());
    }

    ACLNN_CMD(aclnnLightningIndexerKlLoss, target_score, index_probs, eps, loss);

    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lightning_indexer_kl_loss", &lightning_indexer_kl_loss, "lightning_indexer_kl_loss");
}

}  // namespace op_api
