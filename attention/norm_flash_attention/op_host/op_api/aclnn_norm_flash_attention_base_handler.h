/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_NORM_FLASH_ATTENTION_BASE_HANDLER_H
#define OP_HOST_OP_API_ACLNN_NORM_FLASH_ATTENTION_BASE_HANDLER_H

#include "aclnn_norm_flash_attention_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

namespace norm_flash_attention_base {

using namespace op;

constexpr int64_t DIM_IDX_0 = 0L;
constexpr int64_t DIM_IDX_1 = 1L;
constexpr int64_t DIM_IDX_2 = 2L;
constexpr int64_t DIM_IDX_3 = 3L;
constexpr size_t INPUT_DIM_LIMIT = 4UL;
constexpr size_t OUTPUT_DIM_LIMIT = 4UL;

const std::initializer_list<DataType> INPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16
};

const std::initializer_list<DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16
};

class NormFlashAttentionBaseHandler : public norm_flash_attention::NormFlashAttentionHandler {
protected:
    bool CheckInputOutDims() override
    {
        auto queryDimNum = params_.query->GetViewShape().GetDimNum();
        if (queryDimNum != INPUT_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "query tensor dim num must be %zu, but got %zu",
                    INPUT_DIM_LIMIT, queryDimNum);
            return false;
        }

        auto keyDimNum = params_.key->GetViewShape().GetDimNum();
        if (keyDimNum != INPUT_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "key tensor dim num must be %zu, but got %zu",
                    INPUT_DIM_LIMIT, keyDimNum);
            return false;
        }

        auto valueDimNum = params_.value->GetViewShape().GetDimNum();
        if (valueDimNum != INPUT_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "value tensor dim num must be %zu, but got %zu",
                    INPUT_DIM_LIMIT, valueDimNum);
            return false;
        }

        auto outDimNum = params_.attentionOut->GetViewShape().GetDimNum();
        if (outDimNum != OUTPUT_DIM_LIMIT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "attentionOut tensor dim num must be %zu, but got %zu",
                    OUTPUT_DIM_LIMIT, outDimNum);
            return false;
        }

        return true;
    }

    bool CheckInputOutShape() override
    {
        auto queryShape = params_.query->GetViewShape();
        auto keyShape = params_.key->GetViewShape();
        auto valueShape = params_.value->GetViewShape();

        // B, S, H, D 应一致
        int64_t queryB = queryShape.GetDim(DIM_IDX_0);
        int64_t keyB = keyShape.GetDim(DIM_IDX_0);
        int64_t valueB = valueShape.GetDim(DIM_IDX_0);
        if (queryB != keyB || queryB != valueB) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "batch dim mismatch: query(%ld), key(%ld), value(%ld)",
                    queryB, keyB, valueB);
            return false;
        }

        int64_t queryH = queryShape.GetDim(DIM_IDX_2);
        int64_t keyH = keyShape.GetDim(DIM_IDX_2);
        int64_t valueH = valueShape.GetDim(DIM_IDX_2);
        if (queryH != keyH || queryH != valueH) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "head dim mismatch: query(%ld), key(%ld), value(%ld)",
                    queryH, keyH, valueH);
            return false;
        }

        int64_t queryD = queryShape.GetDim(DIM_IDX_3);
        int64_t keyD = keyShape.GetDim(DIM_IDX_3);
        int64_t valueD = valueShape.GetDim(DIM_IDX_3);
        if (queryD != keyD || queryD != valueD) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "head_dim mismatch: query(%ld), key(%ld), value(%ld)",
                    queryD, keyD, valueD);
            return false;
        }

        return true;
    }

    bool CheckDtypeValid() override
    {
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.query, INPUT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.key, INPUT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.value, INPUT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params_.attentionOut, OUTPUT_DTYPE_SUPPORT_LIST, return false);

        auto queryDtype = params_.query->GetDataType();
        auto keyDtype = params_.key->GetDataType();
        auto valueDtype = params_.value->GetDataType();
        auto outDtype = params_.attentionOut->GetDataType();

        if (queryDtype != keyDtype || queryDtype != valueDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "input dtype mismatch: query(%s), key(%s), value(%s)",
                    ToString(queryDtype).GetString(),
                    ToString(keyDtype).GetString(),
                    ToString(valueDtype).GetString());
            return false;
        }

        if (outDtype != queryDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "attentionOut dtype (%s) must equal query dtype (%s)",
                    ToString(outDtype).GetString(),
                    ToString(queryDtype).GetString());
            return false;
        }

        return true;
    }

    bool CheckFormat() override
    {
        if (IsPrivateFormat(params_.query->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "query tensor format must be ND, current format is %s",
                    ToString(params_.query->GetStorageFormat()).GetString());
            return false;
        }

        if (IsPrivateFormat(params_.key->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "key tensor format must be ND, current format is %s",
                    ToString(params_.key->GetStorageFormat()).GetString());
            return false;
        }

        if (IsPrivateFormat(params_.value->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "value tensor format must be ND, current format is %s",
                    ToString(params_.value->GetStorageFormat()).GetString());
            return false;
        }

        if (IsPrivateFormat(params_.attentionOut->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "attentionOut tensor format must be ND, current format is %s",
                    ToString(params_.attentionOut->GetStorageFormat()).GetString());
            return false;
        }

        return true;
    }
};

} // namespace norm_flash_attention_base

#endif // OP_HOST_OP_API_ACLNN_NORM_FLASH_ATTENTION_BASE_HANDLER_H
