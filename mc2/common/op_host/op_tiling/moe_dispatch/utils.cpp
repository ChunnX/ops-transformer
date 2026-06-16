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
 * \file utils.cpp
 * \brief MOE Distribute Dispatch Tiling 工具函数实现
 */

#include "utils.h"
#include "mc2_log.h"

namespace moe_dispatch_utils {

ge::graphStatus CheckTensorRules(gert::TilingContext *ctx, const TensorCheckRule *rules, uint32_t count)
{
    if (ctx == nullptr || rules == nullptr) {
        OPS_LOG_E("CheckTensorRules", "invalid params: ctx or rules is nullptr.");
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0U; i < count; ++i) {
        const TensorCheckRule &rule = rules[i];

        // 获取 tensor desc
        const gert::CompileTimeTensorDesc *desc = nullptr;
        const gert::StorageShape *shape = nullptr;

        if (rule.type == moe_dispatch_defs::TensorType::INPUT ||
            rule.type == moe_dispatch_defs::TensorType::OPTIONAL_INPUT) {
            desc = ctx->GetInputDesc(rule.index);
            shape = ctx->GetInputShape(rule.index);
        } else if (rule.type == moe_dispatch_defs::TensorType::OUTPUT) {
            desc = ctx->GetOutputDesc(rule.index);
            shape = ctx->GetOutputShape(rule.index);
        }

        // 检查 desc 是否为空
        if (desc == nullptr) {
            OPS_LOG_E("CheckTensorRules", "tensor %s desc is nullptr.", rule.name);
            return ge::GRAPH_FAILED;
        }

        // 检查维度
        if (CheckTensorDimNum(shape, rule.expectedDim) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E("CheckTensorRules", "tensor %s dim check failed, expected %u.", rule.name, rule.expectedDim);
            return ge::GRAPH_FAILED;
        }

        // 检查 dtype
        if (rule.dtypeCount > 0U) {
            if (CheckTensorDtypeIn(desc, rule.validDtypes, rule.dtypeCount) != ge::GRAPH_SUCCESS) {
                OPS_LOG_E("CheckTensorRules", "tensor %s dtype check failed.", rule.name);
                return ge::GRAPH_FAILED;
            }
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckOneTensorRule(gert::TilingContext *ctx, const TensorCheckRule &rule)
{
    return CheckTensorRules(ctx, &rule, 1U);
}

ge::graphStatus CheckAttrRules(gert::TilingContext *ctx, const AttrCheckRule *rules, uint32_t count)
{
    if (ctx == nullptr || rules == nullptr) {
        OPS_LOG_E("CheckAttrRules", "invalid params: ctx or rules is nullptr.");
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0U; i < count; ++i) {
        const AttrCheckRule &rule = rules[i];

        // 获取属性值
        const int64_t *attrPtr = ctx->GetAttrs()->GetAttrPointer<int64_t>(rule.attrIndex);

        // 检查必选属性
        if (rule.required && attrPtr == nullptr) {
            OPS_LOG_E("CheckAttrRules", "attr %s is required but nullptr.", rule.attrName);
            return ge::GRAPH_FAILED;
        }

        // 可选属性为空时跳过
        if (attrPtr == nullptr) {
            continue;
        }

        int64_t attrValue = *attrPtr;

        // 检查范围
        if (attrValue < static_cast<int64_t>(rule.minVal) || attrValue > static_cast<int64_t>(rule.maxVal)) {
            OPS_LOG_E("CheckAttrRules", "attr %s value %ld out of range [%u, %u].", rule.attrName, attrValue,
                      rule.minVal, rule.maxVal);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTensorFormatRules(gert::TilingContext *ctx, const TensorFormatRule *rules, uint32_t count,
                                       const char *nodeName)
{
    if (ctx == nullptr || rules == nullptr) {
        OPS_LOG_E("CheckTensorFormatRules", "invalid params: ctx or rules is nullptr.");
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0U; i < count; ++i) {
        const TensorFormatRule &rule = rules[i];

        // 获取 tensor desc
        const gert::CompileTimeTensorDesc *desc = nullptr;
        if (rule.type == moe_dispatch_defs::TensorType::INPUT ||
            rule.type == moe_dispatch_defs::TensorType::OPTIONAL_INPUT) {
            desc = ctx->GetInputDesc(rule.index);
        } else if (rule.type == moe_dispatch_defs::TensorType::OUTPUT) {
            desc = ctx->GetOutputDesc(rule.index);
        }

        // 检查 desc 是否为空
        if (desc == nullptr) {
            OPS_LOG_E(nodeName, "tensor %s desc is nullptr.", rule.name);
            return ge::GRAPH_FAILED;
        }

        // 获取 format
        ge::Format format = desc->GetStorageFormat();
        if (rule.checkPrimary) {
            format = static_cast<ge::Format>(ge::GetPrimaryFormat(format));
        }

        // 检查 format
        if (format != rule.expectedFormat) {
            OPS_LOG_E(nodeName, "tensor %s format is invalid, expected %d, got %d.", rule.name,
                      static_cast<int32_t>(rule.expectedFormat), static_cast<int32_t>(format));
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace moe_dispatch_utils
