/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_H
#define OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor stem_oam_prep_varlen_q_torch(
    const at::Tensor& q,
    const at::Tensor& qSeqLens,
    const at::Tensor& cuSeqLensQ,
    /* * optional */ const c10::optional<at::Tensor>& qScaleOpt,
    const c10::optional<at::Tensor>& kScaleOpt,
    int64_t stemBlockSize,
    int64_t stemStride);

}  // namespace ascend_kernel

#endif