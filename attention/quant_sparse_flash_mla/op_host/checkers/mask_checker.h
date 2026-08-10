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
 * \file mask_checker.h
 * \brief MaskChecker validates the Mask参数组: ori_mask_mode, cmp_mask_mode,
 *        ori_win_left, ori_win_right.
 */

#ifndef QSMLA_MASK_CHECKER_H
#define QSMLA_MASK_CHECKER_H

#include "base_checker.h"

namespace optiling {

class MaskChecker : public QSMLABaseChecker {
public:
    MaskChecker() = default;
    ~MaskChecker() override = default;

    ge::graphStatus CheckSinglePara(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckParaExistence(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckCrossFeature(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckMultiParaConsistency(const QSMLATilingInfo &info) const override;

private:
    ge::graphStatus CheckOriMaskMode(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckCmpMaskMode(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckOriWinLeftRight(const QSMLATilingInfo &info) const;
};

} // namespace optiling
#endif // QSMLA_MASK_CHECKER_H
