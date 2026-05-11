/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_scatter_pa_kv_cache.h"

#include <new>

#include "scatter_pa_kv_cache.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const std::initializer_list<op::DataType> KEY_VALUE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16,
    DataType::DT_INT8, DataType::DT_UINT8,
    DataType::DT_INT16, DataType::DT_UINT16,
    DataType::DT_INT32, DataType::DT_UINT32,
    DataType::DT_HIFLOAT8, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN
};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT32, DataType::DT_INT64
};

static aclnnStatus CheckNullptr(
    const aclTensor *key,
    const aclTensor *keyCacheRef,
    const aclTensor *slotMapping,
    const aclTensor *value,
    const aclTensor *valueCacheRef,
    const uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(keyCacheRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(slotMapping != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    
    if (value != nullptr) {
        CHECK_RET(valueCacheRef != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtypeValid(
    const aclTensor *key,
    const aclTensor *keyCacheRef,
    const aclTensor *slotMapping,
    const aclTensor *value,
    const aclTensor *valueCacheRef,
    const aclTensor *compressLensOptional,
    const aclTensor *compressSeqOffsetOptional,
    const aclTensor *seqLensOptional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(key, KEY_VALUE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(keyCacheRef, KEY_VALUE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(slotMapping, INDEX_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

    auto keyDtype = key->GetDataType();
    auto keyCacheDtype = keyCacheRef->GetDataType();
    if (keyDtype != keyCacheDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The data type of key[%s] and keyCacheRef[%s] are not equal.",
                op::ToString(DataType(keyDtype)).GetString(),
                op::ToString(DataType(keyCacheDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (value != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(value, KEY_VALUE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT(valueCacheRef, KEY_VALUE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

        auto valueDtype = value->GetDataType();
        auto valueCacheDtype = valueCacheRef->GetDataType();
        if (keyDtype != valueDtype || keyDtype != valueCacheDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The data type of key[%s], value[%s], valueCacheRef[%s] are not equal.",
                    op::ToString(DataType(keyDtype)).GetString(),
                    op::ToString(DataType(valueDtype)).GetString(),
                    op::ToString(DataType(valueCacheDtype)).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    auto slotMappingDtype = slotMapping->GetDataType();
    if (compressLensOptional != nullptr) {
        if (slotMappingDtype != compressLensOptional->GetDataType()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The data type of slotMapping[%s] and compressLensOptional[%s] are not equal.",
                    op::ToString(DataType(slotMappingDtype)).GetString(),
                    op::ToString(DataType(compressLensOptional->GetDataType())).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (compressSeqOffsetOptional != nullptr) {
        if (slotMappingDtype != compressSeqOffsetOptional->GetDataType()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The data type of slotMapping[%s] and compressSeqOffsetOptional[%s] are not equal.",
                    op::ToString(DataType(slotMappingDtype)).GetString(),
                    op::ToString(DataType(compressSeqOffsetOptional->GetDataType())).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (seqLensOptional != nullptr) {
        if (slotMappingDtype != seqLensOptional->GetDataType()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The data type of slotMapping[%s] and seqLensOptional[%s] are not equal.",
                    op::ToString(DataType(slotMappingDtype)).GetString(),
                    op::ToString(DataType(seqLensOptional->GetDataType())).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const aclTensor *key, const aclTensor *value)
{
    auto keyDimNum = key->GetViewShape().GetDimNum();
    auto valueDimNum = value != nullptr ? value->GetViewShape().GetDimNum() : 0;

    if (keyDimNum != 3 && keyDimNum != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dim num of key must be 3 or 4, but got %ld.",
                keyDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (value != nullptr && valueDimNum != 0 && valueDimNum != 3 && valueDimNum != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The dim num of value must be 0, 3 or 4, but got %ld.",
                valueDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static bool IsSupportedFormat(ge::Format format)
{
    return format == ge::FORMAT_ND ||
           format == ge::FORMAT_FRACTAL_NZ;
}

static aclnnStatus CheckFormat(
    const aclTensor *key,
    const aclTensor *keyCacheRef,
    const aclTensor *slotMapping,
    const aclTensor *value,
    const aclTensor *valueCacheRef,
    char *cacheModeOptional)
{
    auto keyFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(key->GetStorageFormat()));
    auto slotMappingFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(slotMapping->GetStorageFormat()));
    
    if (!IsSupportedFormat(keyFormat)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "key format[%s] is not supported, only support ND or FRACTAL_NZ.",
                op::ToString(keyFormat).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (slotMappingFormat != ge::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "slotMapping format[%s] is not supported, only support ND.",
                op::ToString(slotMappingFormat).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto keyCacheFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(keyCacheRef->GetStorageFormat()));
    if (cacheModeOptional != nullptr && strcmp(cacheModeOptional, "PA_NZ") == 0) {
        if (ge::GetPrimaryFormat(keyCacheFormat) != ge::FORMAT_FRACTAL_NZ) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When cacheMode is PA_NZ, keyCacheRef must be FRACTAL_NZ format, but got %s.",
                    op::ToString(keyCacheFormat).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (ge::GetPrimaryFormat(keyCacheFormat) != ge::FORMAT_ND) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "When cacheMode is Norm or null, keyCacheRef must be ND format, but got %s.",
                    op::ToString(keyCacheFormat).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (value != nullptr) {
        auto valueFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(value->GetStorageFormat()));
        if (!IsSupportedFormat(valueFormat)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "value format[%s] is not supported, only support ND or FRACTAL_NZ.",
                    op::ToString(valueFormat).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }

        auto valueCacheFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(valueCacheRef->GetStorageFormat()));
        if (cacheModeOptional != nullptr && strcmp(cacheModeOptional, "PA_NZ") == 0) {
            if (ge::GetPrimaryFormat(valueCacheFormat) != ge::FORMAT_FRACTAL_NZ) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "When cacheMode is PA_NZ, valueCacheRef must be FRACTAL_NZ format, but got %s.",
                        op::ToString(valueCacheFormat).GetString());
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (ge::GetPrimaryFormat(valueCacheFormat) != ge::FORMAT_ND) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                        "When cacheMode is Norm or null, valueCacheRef must be ND format, but got %s.",
                        op::ToString(valueCacheFormat).GetString());
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }

    return ACLNN_SUCCESS;
}

static bool CheckIsEmptyTensor(const aclTensor *key)
{
    return key->GetViewShape().GetShapeSize() == 0;
}

} // namespace

aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(
    const aclTensor *key,
    aclTensor *keyCacheRef,
    const aclTensor *slotMapping,
    const aclTensor *value,
    aclTensor *valueCacheRef,
    const aclTensor *compressLensOptional,
    const aclTensor *compressSeqOffsetOptional,
    const aclTensor *seqLensOptional,
    char *cacheModeOptional,
    char *scatterModeOptional,
    const aclIntArray *stridesOptional,
    const aclIntArray *offsetsOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnScatterPaKvCache,
                   DFX_IN(key, keyCacheRef, slotMapping, value, valueCacheRef,
                          compressLensOptional, compressSeqOffsetOptional, seqLensOptional,
                          cacheModeOptional, scatterModeOptional, stridesOptional, offsetsOptional),
                   DFX_OUT(keyCacheRef, valueCacheRef));

    CHECK_RET(CheckNullptr(key, keyCacheRef, slotMapping, value, valueCacheRef,
                           workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    if (CheckIsEmptyTensor(key)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "[ScatterPaKvCache] do not support empty key.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckShape(key, value) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    
    CHECK_RET(CheckFormat(key, keyCacheRef, slotMapping, value, valueCacheRef,
                          cacheModeOptional) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckDtypeValid(key, keyCacheRef, slotMapping, value, valueCacheRef,
                              compressLensOptional, compressSeqOffsetOptional, seqLensOptional) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto keyContiguous = l0op::Contiguous(key, uniqueExecutor.get());
    CHECK_RET(keyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyCacheContiguous = const_cast<aclTensor *>(l0op::Contiguous(keyCacheRef, uniqueExecutor.get()));
    CHECK_RET(keyCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto slotMappingContiguous = l0op::Contiguous(slotMapping, uniqueExecutor.get());
    CHECK_RET(slotMappingContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueContiguous = value;
    auto valueCacheContiguous = valueCacheRef;
    if (value != nullptr) {
        valueContiguous = l0op::Contiguous(value, uniqueExecutor.get());
        CHECK_RET(valueContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        valueCacheContiguous = const_cast<aclTensor *>(l0op::Contiguous(valueCacheRef, uniqueExecutor.get()));
        CHECK_RET(valueCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto compressLensContiguous = l0op::Contiguous(compressLensOptional, uniqueExecutor.get());
    auto compressSeqOffsetContiguous = l0op::Contiguous(compressSeqOffsetOptional, uniqueExecutor.get());
    auto seqLensContiguous = l0op::Contiguous(seqLensOptional, uniqueExecutor.get());

    auto scatterResult = l0op::ScatterPaKvCache(
        keyContiguous, keyCacheContiguous, slotMappingContiguous,
        valueContiguous, valueCacheContiguous,
        compressLensContiguous, compressSeqOffsetContiguous, seqLensContiguous,
        cacheModeOptional, scatterModeOptional, stridesOptional, offsetsOptional,
        uniqueExecutor.get());
    CHECK_RET(std::get<0>(scatterResult) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto keyCacheViewCopy = l0op::ViewCopy(std::get<0>(scatterResult), keyCacheRef, uniqueExecutor.get());
    CHECK_RET(keyCacheViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (value != nullptr) {
        auto valueCacheViewCopy = l0op::ViewCopy(std::get<1>(scatterResult), valueCacheRef, uniqueExecutor.get());
        CHECK_RET(valueCacheViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnScatterPaKvCache(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnScatterPaKvCache);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "ScatterPaKvCache launch aicore failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif