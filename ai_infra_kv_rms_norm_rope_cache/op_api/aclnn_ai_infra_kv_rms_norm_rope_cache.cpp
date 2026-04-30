/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_ai_infra_kv_rms_norm_rope_cache.h"
#include "ai_infra_kv_rms_norm_rope_cache.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t KV_DIM_NUM = 4;
constexpr size_t K_CACHE_DIM_NUM = 4;
constexpr size_t CKV_CACHE_DIM_NUM = 4;

const std::initializer_list<DataType> KV_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16
};
const std::initializer_list<DataType> CACHE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT8
};
const std::initializer_list<DataType> INDEX_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT64
};


// 辅助函数1：将 Shape 对象（GetViewShape/GetStorageShape 返回值）转为字符串
std::string ShapeToString(const Shape& shape) {
    int dim_num = shape.GetDimNum();  // 获取维度数量
    std::string shape_str = "[";
    for (int i = 0; i < dim_num; ++i) {
        shape_str += std::to_string(shape.GetDim(i));  // 获取第i维的大小
        if (i != dim_num - 1) {
            shape_str += ", ";
        }
    }
    shape_str += "]";
    return shape_str;
}

// 辅助函数2：将 Strides 对象（GetViewStrides 返回值）转为字符串
std::string StridesToString(const op::internal::SmallVector<long int, 25, op::internal::PoolAllocator<long int>>& vec, 
                               const std::string& name = "shape") {
    if (vec.empty()) {
        return "[] (empty " + name + ")";
    }

    std::string str = "[";
    // 限制打印维度数（避免超长输出）
    size_t print_num = std::min(vec.size(), static_cast<size_t>(10));
    for (size_t i = 0; i < print_num; ++i) {
        str += std::to_string(vec[i]);  // 用[]访问SmallVector元素
        if (i != print_num - 1) {
            str += ", ";
        }
    }
    if (vec.size() > print_num) {
        str += ", ...] (total " + std::to_string(vec.size()) + " dims)";
    } else {
        str += "]";
    }
    return str;
}

aclnnStatus CheckAiInfraKvRmsNormRopeCacheParams(
    const aclTensor* kv, const aclTensor* gamma, const aclTensor* cos, const aclTensor* sin,
    const aclTensor* index, aclTensor* k_cache_ref, aclTensor* ckv_cache_ref,
    char* cache_mode, char* rotary_mode, char* quant_mode,
    const aclTensor* k_rope_out, const aclTensor* c_kv_out)
{
    OP_CHECK_NULL(kv, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(gamma, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(cos, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(sin, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(index, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(ckv_cache_ref, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(k_rope_out, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(c_kv_out, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(cache_mode, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(rotary_mode, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(quant_mode, return ACLNN_ERR_INNER_NULLPTR);

    OP_CHECK_WRONG_DIMENSION(kv, KV_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(ckv_cache_ref, CKV_CACHE_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_DTYPE_NOT_SUPPORT(kv, KV_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(gamma, KV_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(cos, KV_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(sin, KV_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(index, INDEX_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(ckv_cache_ref, CACHE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    if (k_cache_ref != nullptr){
        OP_CHECK_NULL(k_cache_ref, return ACLNN_ERR_INNER_NULLPTR);
        OP_CHECK_WRONG_DIMENSION(k_cache_ref, K_CACHE_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_SUPPORT(k_cache_ref, CACHE_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus AiInfraKvRmsNormRopeCacheCommonProcess(
    const aclTensor* kv, const aclTensor* gamma, const aclTensor* cos, const aclTensor* sin,
    const aclTensor* index, aclTensor* k_cache_ref, aclTensor* ckv_cache_ref,
    const aclTensor* k_rope_scale, const aclTensor* c_kv_scale, 
    const aclTensor* k_rope_offset, const aclTensor* c_kv_offset, const aclTensor* v, 
    double epsilon, char* cache_mode, char* rotary_mode, char* quant_mode, bool is_output_kv,
    const aclTensor* k_rope_out, const aclTensor* c_kv_out,
    aclOpExecutor *executor)
{
    auto ret = CheckAiInfraKvRmsNormRopeCacheParams(kv, gamma, cos, sin, index, k_cache_ref, ckv_cache_ref,
        cache_mode, rotary_mode, quant_mode, k_rope_out, c_kv_out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *kv_final = l0op::Contiguous(kv, executor);
    CHECK_COND(kv_final != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous kv failed.");
    const aclTensor *gamma_final = l0op::Contiguous(gamma, executor);
    CHECK_COND(gamma_final != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous gamma failed.");
    const aclTensor *cos_final = l0op::Contiguous(cos, executor);
    CHECK_COND(cos_final != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous cos failed.");
    const aclTensor *sin_final = l0op::Contiguous(sin, executor);
    CHECK_COND(sin_final != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous sin failed.");
    const aclTensor *index_final = l0op::Contiguous(index, executor);
    CHECK_COND(index_final != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous index failed.");

    aclTensor *k_cache_final = nullptr;
    if(k_cache_ref != nullptr) {
        k_cache_final = executor->CreateView(
            k_cache_ref, k_cache_ref->GetViewShape(), k_cache_ref->GetStorageShape(),
            k_cache_ref->GetViewStrides(), k_cache_ref->GetViewOffset()
        );
        CHECK_COND(k_cache_final != nullptr, ACLNN_ERR_INNER_NULLPTR, "CreateView for k_cache failed.");
    } else {
        std::vector<int64_t> emptyShape = {0};
        aclDataType dtype = static_cast<aclDataType>(kv_final->GetDataType());
        k_cache_final = aclCreateTensor(
          emptyShape.data(),    // view_dims: shape {0}
          emptyShape.size(),    // view_dims_num: 1
          dtype,// data_type
          nullptr,              // stride: nullptr 表示连续
          0,                    // offset: 0
          ACL_FORMAT_ND,        // format
          emptyShape.data(),    // storage_dims: 同 view_dims
          emptyShape.size(),    // storage_dims_num: 1
          nullptr               // tensor_data: 无实际数据
      );
    }
    aclTensor *ckv_cache_final = executor->CreateView(
        ckv_cache_ref, ckv_cache_ref->GetViewShape(), ckv_cache_ref->GetStorageShape(),
        ckv_cache_ref->GetViewStrides(), ckv_cache_ref->GetViewOffset()
    );
    CHECK_COND(ckv_cache_final != nullptr, ACLNN_ERR_INNER_NULLPTR, "CreateView for ckv_cache failed.");
    
    // 处理可选参数
    const aclTensor *k_rope_scale_final = nullptr;
    if(k_rope_scale != nullptr) {
        k_rope_scale_final = l0op::Contiguous(k_rope_scale, executor);
        CHECK_COND(k_rope_scale != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous k_rope_scale failed.");
    }
    const aclTensor *c_kv_scale_final = nullptr;
    if(c_kv_scale != nullptr) {
        c_kv_scale_final = l0op::Contiguous(c_kv_scale, executor);
        CHECK_COND(c_kv_scale_final != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous c_kv_scale failed.");
    }
    const aclTensor *k_rope_offset_final = nullptr;
    if(k_rope_offset != nullptr) {
        k_rope_offset_final = l0op::Contiguous(k_rope_offset, executor);
        CHECK_COND(k_rope_offset_final != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous k_rope_offset failed.");
    }
    const aclTensor *c_kv_offset_final = nullptr;
    if(c_kv_offset != nullptr) {
        c_kv_offset_final = l0op::Contiguous(c_kv_offset, executor);
        CHECK_COND(c_kv_offset_final != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous c_kv_offset failed.");
    }
    const aclTensor *v_final = nullptr;
    if(v != nullptr) {
        v_final = l0op::Contiguous(v, executor);
        CHECK_COND(v_final != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous v failed.");
    }

    // out is always contiguous. passed directly to l0op.
    bool out = l0op::AiInfraKvRmsNormRopeCache(
        kv_final, gamma_final, cos_final, sin_final, index_final, k_cache_final, ckv_cache_final,
        k_rope_scale_final, c_kv_scale_final, k_rope_offset_final, c_kv_offset_final, v_final, 
        epsilon, cache_mode, rotary_mode, quant_mode, is_output_kv, k_rope_out, c_kv_out, executor);
    CHECK_RET(out, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAiInfraKvRmsNormRopeCacheGetWorkspaceSize(
    const aclTensor* kv, const aclTensor* gamma, const aclTensor* cos, const aclTensor* sin, const aclTensor* index,
    aclTensor* kCacheRef, aclTensor* ckvCacheRef,
    const aclTensor* kRopeScaleOptional, const aclTensor* ckvScaleOptional,
    const aclTensor* kRopeOffsetOptional, const aclTensor* cKvOffsetOptional, const aclTensor* vOptional,
    double epsilon, char* cacheModeOptional, char* rotaryModeOptional, char* quantModeOptional, bool isOutputKv,
    aclTensor* kRopeOut, aclTensor* cKvOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAiInfraKvRmsNormRopeCache,
        DFX_IN(kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef,
                kRopeScaleOptional, ckvScaleOptional, kRopeOffsetOptional, cKvOffsetOptional, vOptional),
        DFX_OUT(kRopeOut, cKvOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    auto ret = AiInfraKvRmsNormRopeCacheCommonProcess(
        kv, gamma, cos, sin, index, kCacheRef, ckvCacheRef,
        kRopeScaleOptional, ckvScaleOptional, kRopeOffsetOptional, cKvOffsetOptional, vOptional, 
        epsilon, cacheModeOptional, rotaryModeOptional, quantModeOptional, isOutputKv, 
        kRopeOut, cKvOut,
        uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAiInfraKvRmsNormRopeCache(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAiInfraKvRmsNormRopeCache);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}


}  // namespace

#ifdef __cplusplus
}
#endif
