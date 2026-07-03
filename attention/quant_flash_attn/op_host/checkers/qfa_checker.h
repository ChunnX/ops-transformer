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
 * \file qfa_checker.h
 * \brief QFAChecker - manages all checkers for QuantFlashAttn
 */

#ifndef QUANT_FLASH_ATTN_QFA_CHECKER_H
#define QUANT_FLASH_ATTN_QFA_CHECKER_H

#include <memory>
#include <vector>
#include <functional>
#include "tiling/tiling_api.h"

#include "../qfa_tiling_info.h"
#include "../qfa_tiling_shape.h"
#include "../quant_flash_attn_tiling_utils.h"

namespace optiling {
namespace quant_flash_attn {

class QfaChecker {
public:
    QfaChecker() = default;
    ~QfaChecker() = default;

    ge::graphStatus Init(const QfaTilingInfo &qfaInfo);
    ge::graphStatus Process(const QfaTilingInfo &qfaInfo);

    ge::graphStatus CheckSinglePara(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckParaExistence(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckFeature(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckMultiPara(const QfaTilingInfo &qfaInfo);

private:
    ge::graphStatus CheckDtypeSupport(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckQuantMode(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckLayoutSupport(const QfaTilingInfo &qfaInfo);
    ge::graphStatus CheckMaskParams(const QfaTilingInfo &qfaInfo);
};

} // namespace quant_flash_attn
} // namespace optiling
#endif // QUANT_FLASH_ATTN_QFA_CHECKER_H
