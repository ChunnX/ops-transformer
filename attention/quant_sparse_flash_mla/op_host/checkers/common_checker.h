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
 * \file common_checker.h
 * \brief CommonChecker validates the 公共参数组: q, ori_kv, cmp_kv, q_descale,
 *        ori_kv_descale, cmp_kv_descale, metadata, quant_mode, softmax_scale,
 *        layout_q, layout_kv, attention_out.
 */

#ifndef QSMLA_COMMON_CHECKER_H
#define QSMLA_COMMON_CHECKER_H

#include "base_checker.h"

namespace optiling {

class CommonChecker : public QSMLABaseChecker {
public:
    CommonChecker() = default;
    ~CommonChecker() override = default;

    ge::graphStatus CheckSinglePara(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckParaExistence(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckCrossFeature(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckMultiParaConsistency(const QSMLATilingInfo &info) const override;

private:
    // --- Single parameter checks (dtype / shape / format) ---
    ge::graphStatus CheckQuery(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckOriKv(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckCmpKv(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckDescale(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckMetadata(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckAttentionOut(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckQuantModeAttr(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckLayoutAttr(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckAxis(const QSMLATilingInfo &info) const;

    // --- Existence checks ---
    ge::graphStatus CheckRequiredExistence(const QSMLATilingInfo &info) const;

    // --- Cross-feature checks ---
    ge::graphStatus CheckLayoutQKvConsistency(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckQuantModeDtype(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckOriCmpKvDtypeConsistency(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckKvContiguous(const QSMLATilingInfo &info) const;

    // --- Multi-parameter consistency ---
    ge::graphStatus CheckQueryOutShapeConsistency(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckDescaleShapeConsistency(const QSMLATilingInfo &info) const;
};

} // namespace optiling
#endif // QSMLA_COMMON_CHECKER_H
