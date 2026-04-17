/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fa_checker.h
 * \brief
 */

#ifndef FA_CHECKER_H
#define FA_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

#include "./seq_len_checker.h"
#include "./base_checker.h"
#include "./empty_tensor_checker.h"
#include "./layout_checker.h"
#include "./mask_checker.h"
#include "./metadata_checker.h"
#include "./paged_attention_checker.h"
#include "./sinks_checker.h"
#include "./shape_checker.h"
#include "./softmax_lse_checker.h"

namespace optiling {
class FAChecker {
public:
    FAChecker() = default;
    ~FAChecker() = default;
    ge::graphStatus Init(const FaTilingInfo &faInfo);
    ge::graphStatus Process(const FaTilingInfo &faInfo);

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo);
    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeature(const FaTilingInfo &faInfo);
    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo);

private:
    std::unique_ptr<ActualSeqLenChecker> actualSeqLenChecker_;
    std::unique_ptr<EmptyTensorChecker> emptyTensorChecker_;
    std::unique_ptr<LayoutChecker> layoutChecker_;
    std::unique_ptr<MaskChecker> maskChecker_;
    std::unique_ptr<MetadataChecker> metadataChecker_;
    std::unique_ptr<PagedAttentionChecker> pagedAttentionChecker_;
    std::unique_ptr<SinksChecker> sinksChecker_;
    std::unique_ptr<ShapeChecker> shapeChecker_;
    std::unique_ptr<SoftmaxLSEChecker> softmaxLSEChecker_;

    bool enableNonQuant_ = false;
    bool enableFullQuant_ = false;
    bool enableAntiQuant_ = false;
};

} // namespace optiling
#endif // FA_CHECKER_H