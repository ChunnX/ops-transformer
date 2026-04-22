/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fa_checker.cpp
 * \brief FAChecker implementation using Composite Pattern
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fa_tiling_info.h"
#include "../flash_attn_tiling_constants.h"
#include "fa_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

// RegisterCheckers - 注册所有 checker
// 新增 checker 时在此方法中添加: checkers_.push_back(std::make_unique<XxxChecker>());
void FAChecker::RegisterCheckers()
{
    checkers_.push_back(std::make_unique<ActualSeqLenChecker>());
    checkers_.push_back(std::make_unique<LayoutChecker>());
    checkers_.push_back(std::make_unique<MaskChecker>());
    checkers_.push_back(std::make_unique<MetadataChecker>());
    checkers_.push_back(std::make_unique<PagedAttentionChecker>());
    checkers_.push_back(std::make_unique<SinksChecker>());
    checkers_.push_back(std::make_unique<ShapeChecker>());
    checkers_.push_back(std::make_unique<SoftmaxLSEChecker>());

    // EmptyTensorChecker 不加入 checkers_，因其有特殊处理逻辑（空tensor场景直接返回成功）
    emptyTensorChecker_ = std::make_unique<EmptyTensorChecker>();
}

ge::graphStatus FAChecker::Init(const FaTilingInfo &faInfo)
{
    (void)faInfo;
    RegisterCheckers();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FAChecker::RunCheck(const CheckMethod &method, const FaTilingInfo &faInfo)
{
    for (const auto &checker : checkers_) {
        if (ge::GRAPH_SUCCESS != method(checker.get(), faInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FAChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    return RunCheck([](FABaseChecker *c, const FaTilingInfo &info) { return c->CheckSinglePara(info); }, faInfo);
}

ge::graphStatus FAChecker::CheckParaExistence(const FaTilingInfo &faInfo)
{
    return RunCheck([](FABaseChecker *c, const FaTilingInfo &info) { return c->CheckParaExistence(info); }, faInfo);
}

ge::graphStatus FAChecker::CheckFeature(const FaTilingInfo &faInfo)
{
    return RunCheck([](FABaseChecker *c, const FaTilingInfo &info) { return c->CheckFeature(info); }, faInfo);
}

ge::graphStatus FAChecker::CheckMultiPara(const FaTilingInfo &faInfo)
{
    return RunCheck([](FABaseChecker *c, const FaTilingInfo &info) { return c->CheckMultiPara(info); }, faInfo);
}

ge::graphStatus FAChecker::Process(const FaTilingInfo &faInfo)
{
    if (emptyTensorChecker_->IsEmptyTensorScenario(faInfo)) {
        OP_LOGI(faInfo.opName, "Empty tensor scenario detected, skip normal validation.");
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != emptyTensorChecker_->CheckEmptyTensor(faInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckSinglePara(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckParaExistence(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckFeature(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckMultiPara(faInfo)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling