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
 * \file qfa_checker.cpp
 * \brief QFAChecker implementation
 */

#include "qfa_checker.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;
namespace optiling {
namespace quant_flash_attn {

ge::graphStatus QfaChecker::Init(const QfaTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaChecker::Process(const QfaTilingInfo &qfaInfo)
{
    if (CheckParaExistence(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckSinglePara(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMultiPara(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFeature(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaChecker::CheckSinglePara(const QfaTilingInfo &qfaInfo)
{
    if (CheckDtypeSupport(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckQuantMode(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckLayoutSupport(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMaskParams(qfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaChecker::CheckParaExistence(const QfaTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaChecker::CheckFeature(const QfaTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaChecker::CheckMultiPara(const QfaTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaChecker::CheckDtypeSupport(const QfaTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaChecker::CheckQuantMode(const QfaTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaChecker::CheckLayoutSupport(const QfaTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QfaChecker::CheckMaskParams(const QfaTilingInfo &qfaInfo)
{
    (void)qfaInfo;
    return ge::GRAPH_SUCCESS;
}

} // namespace quant_flash_attn
} // namespace optiling
