/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <ATen/ATen.h>

namespace LightningAttn::Wrapper::Forward::lo_cmm128
{
/**
 * @namespace LightningAttn::Wrapper::Forward::lo_cmm
 * @brief Forward pass implementation
 */

std::tuple<at::Tensor, at::Tensor> run(const at::Tensor& query,
                                       const at::Tensor& key,
                                       const at::Tensor& value,
                                       const at::Tensor& slopeRate,
                                       const bool isTraining);

} // namespace LightningAttn::Wrapper::Forward::lo_cmm128
