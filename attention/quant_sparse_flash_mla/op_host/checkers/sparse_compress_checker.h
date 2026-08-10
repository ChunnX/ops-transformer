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
 * \file sparse_compress_checker.h
 * \brief SparseCompressChecker validates the 稀疏压缩参数组: cmp_ratio,
 *        cmp_residual_kv, ori_sparse_indices, cmp_sparse_indices,
 *        ori_topk_length, cmp_topk_length, topk_value_mode.
 */

#ifndef QSMLA_SPARSE_COMPRESS_CHECKER_H
#define QSMLA_SPARSE_COMPRESS_CHECKER_H

#include "base_checker.h"

namespace optiling {

class SparseCompressChecker : public QSMLABaseChecker {
public:
    SparseCompressChecker() = default;
    ~SparseCompressChecker() override = default;

    ge::graphStatus CheckSinglePara(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckParaExistence(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckCrossFeature(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckMultiParaConsistency(const QSMLATilingInfo &info) const override;

private:
    ge::graphStatus CheckCmpRatio(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckCmpResidualKv(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckOriSparseIndices(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckCmpSparseIndices(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckOriTopkLength(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckCmpTopkLength(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckTopkValueMode(const QSMLATilingInfo &info) const;

    ge::graphStatus CheckCmpResidualKvExistence(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckOriTopkLengthExistence(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckCmpTopkLengthExistence(const QSMLATilingInfo &info) const;

    ge::graphStatus CheckSparseIndicesShapeMatchQ(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckTopkLengthShapeMatchQ(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckN2Consistency(const QSMLATilingInfo &info) const;
};

} // namespace optiling
#endif // QSMLA_SPARSE_COMPRESS_CHECKER_H
