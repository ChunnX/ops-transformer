/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file paged_attention_checker.h
 * \brief
 */

#ifndef PAGED_ATTENTION_CHECKER_H
#define PAGED_ATTENTION_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class PagedAttentionChecker : public BaseChecker {
public:
    PagedAttentionChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~PagedAttentionChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckFeature(const FaTilingInfo &faInfo) override;
    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo) override;

private:
    // 公共校验函数

    ge::graphStatus CheckBlockTableDtype(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQDtypeSupport(const FaTilingInfo &faInfo);
    ge::graphStatus CheckBlockTableExistence(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeatureExistence(const FaTilingInfo &faInfo);
    ge::graphStatus CheckSeqLengthKVExistence(const FaTilingInfo &faInfo);

    int64_t GetMaxBlockNumPerBatch(const FaTilingInfo &faInfo);
    ge::graphStatus CheckMaskShape(const FaTilingInfo &faInfo);

    ge::graphStatus CheckBlockTableShape(const FaTilingInfo &faInfo);
    ge::graphStatus CheckBlockSize(const FaTilingInfo &faInfo);
    ge::graphStatus CheckPADimNum(const FaTilingInfo &faInfo);

    ge::graphStatus CheckPACacheShape(const FaTilingInfo &faInfo, const gert::Shape tempShape,
        const std::string& inputName);
};

} // namespace optiling
#endif // PAGED_ATTENTION_CHECKER_H