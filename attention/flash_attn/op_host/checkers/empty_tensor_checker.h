/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file empty_tensor_checker.h
 * \brief Empty tensor scenario checker for npu_flash_attn (特殊场景处理)
 */

#ifndef EMPTY_TENSOR_CHECKER_H
#define EMPTY_TENSOR_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class EmptyTensorChecker : public BaseChecker {
public:
    EmptyTensorChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~EmptyTensorChecker() override = default;

    bool IsEmptyTensorScenario(const FaTilingInfo &faInfo);
};

} // namespace optiling
#endif // EMPTY_TENSOR_CHECKER_H