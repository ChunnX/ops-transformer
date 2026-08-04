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
 * \file softmax_lse_checker.h
 * \brief SoftmaxLseChecker validates the SoftmaxLse参数组:
 *        return_softmax_lse, softmax_lse.
 */

#ifndef QSMLA_SOFTMAX_LSE_CHECKER_H
#define QSMLA_SOFTMAX_LSE_CHECKER_H

#include "base_checker.h"

namespace optiling {

class SoftmaxLseChecker : public QSMLABaseChecker {
public:
    SoftmaxLseChecker() = default;
    ~SoftmaxLseChecker() override = default;

    ge::graphStatus CheckSinglePara(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckParaExistence(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckCrossFeature(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckMultiParaConsistency(const QSMLATilingInfo &info) const override;
};

} // namespace optiling
#endif // QSMLA_SOFTMAX_LSE_CHECKER_H
