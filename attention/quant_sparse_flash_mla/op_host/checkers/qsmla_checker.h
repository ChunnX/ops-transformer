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
 * \file qsmla_checker.h
 * \brief QSMLAChecker is the orchestrator that owns all sub-checkers and drives
 *        the four-phase check pipeline: CheckSinglePara → CheckParaExistence →
 *        CheckCrossFeature → CheckMultiParaConsistency.
 *
 *        Sub-checkers (one per parameter group):
 *          - CommonChecker         公共参数组
 *          - MaskChecker           Mask参数组
 *          - SeqLensChecker        SeqLens参数组
 *          - SparseCompressChecker 稀疏压缩参数组
 *          - PagedAttentionChecker Paged Attention参数组
 *          - SinksChecker          Sinks参数组
 *          - SoftmaxLseChecker     SoftmaxLse参数组
 */

#ifndef QSMLA_CHECKER_H
#define QSMLA_CHECKER_H

#include <memory>
#include "base_checker.h"
#include "common_checker.h"
#include "mask_checker.h"
#include "seqlens_checker.h"
#include "sparse_compress_checker.h"
#include "paged_attention_checker.h"
#include "sinks_checker.h"
#include "softmax_lse_checker.h"

namespace optiling {

class QSMLAChecker {
public:
    QSMLAChecker() = default;
    ~QSMLAChecker() = default;

    ge::graphStatus Process(const QSMLATilingInfo &info);

private:
    std::unique_ptr<CommonChecker> commonChecker_;
    std::unique_ptr<MaskChecker> maskChecker_;
    std::unique_ptr<SeqLensChecker> seqlensChecker_;
    std::unique_ptr<SparseCompressChecker> sparseCompressChecker_;
    std::unique_ptr<PagedAttentionChecker> pagedAttentionChecker_;
    std::unique_ptr<SinksChecker> sinksChecker_;
    std::unique_ptr<SoftmaxLseChecker> softmaxLseChecker_;
};

} // namespace optiling
#endif // QSMLA_CHECKER_H
