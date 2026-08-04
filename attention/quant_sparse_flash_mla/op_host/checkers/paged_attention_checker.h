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
 * \file paged_attention_checker.h
 * \brief PagedAttentionChecker validates the Paged Attention参数组:
 *        ori_block_table, cmp_block_table.
 */

#ifndef QSMLA_PAGED_ATTENTION_CHECKER_H
#define QSMLA_PAGED_ATTENTION_CHECKER_H

#include "base_checker.h"

namespace optiling {

class PagedAttentionChecker : public QSMLABaseChecker {
public:
    PagedAttentionChecker() = default;
    ~PagedAttentionChecker() override = default;

    ge::graphStatus CheckSinglePara(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckParaExistence(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckCrossFeature(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckMultiParaConsistency(const QSMLATilingInfo &info) const override;

private:
    ge::graphStatus CheckOriBlockTable(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckCmpBlockTable(const QSMLATilingInfo &info) const;
};

} // namespace optiling
#endif // QSMLA_PAGED_ATTENTION_CHECKER_H
