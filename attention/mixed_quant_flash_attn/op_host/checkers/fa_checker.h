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
 * \file fa_checker.h
 * \brief FAChecker using Composite Pattern - manages all checkers uniformly
 */

#ifndef MIXED_QUANT_FLASH_ATTN_FA_CHECKER_H
#define MIXED_QUANT_FLASH_ATTN_FA_CHECKER_H

#include <memory>
#include <vector>
#include <functional>
#include "tiling/tiling_api.h"
#include "base_checker.h"

#include "./quant_checker.h"
#include "./seq_len_checker.h"
#include "./common_checker.h"
#include "./mask_checker.h"
#include "./metadata_checker.h"
#include "./paged_attention_checker.h"
#include "./sinks_checker.h"
#include "./softmax_lse_checker.h"

namespace optiling {
namespace mixed_quant_flash_attn {

class FAChecker {
public:
    FAChecker() = default;
    ~FAChecker() = default;

    ge::graphStatus Init(const FaTilingInfo& faInfo);
    ge::graphStatus Process(const FaTilingInfo& faInfo);

    ge::graphStatus CheckSinglePara(const FaTilingInfo& faInfo);
    ge::graphStatus CheckParaExistence(const FaTilingInfo& faInfo);
    ge::graphStatus CheckFeature(const FaTilingInfo& faInfo);
    ge::graphStatus CheckMultiPara(const FaTilingInfo& faInfo);

private:
    using CheckMethod = std::function<ge::graphStatus(FABaseChecker*, const FaTilingInfo&)>;

    ge::graphStatus RunCheck(const CheckMethod& method, const FaTilingInfo& faInfo);

    void RegisterCheckers();

    std::vector<std::unique_ptr<FABaseChecker>> checkers_;
};

} // namespace mixed_quant_flash_attn
} // namespace optiling
#endif // MIXED_QUANT_FLASH_ATTN_FA_CHECKER_H
