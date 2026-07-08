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
 * \file scatter_pa_kv_cache_with_k_scale.cpp
 * \brief torch_npu閫傞厤锛歴catter_pa_kv_cache_with_k_scale绠楀瓙鐨凜++缁戝畾
 */

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "aclnn_common.h"

namespace op_api {

/**
 * scatter_pa_kv_cache_with_k_scale绠楀瓙鐨凜++瀹炵幇
 *
 * 鍔熻兘锛氬皢FP8鏍煎紡鐨刱ey/value鍜屽搴旂殑key_scale鏇存柊鍒癒V cache涓? *
 * @param key [num_tokens, num_head, k_head_size] FP8鏍煎紡鐨刱ey杈撳叆
 * @param value [num_tokens, num_head, v_head_size] FP8鏍煎紡鐨剉alue杈撳叆
 * @param key_cache [num_blocks, num_head, block_size, k_head_size] FP8鏍煎紡鐨刱ey cache
 * @param value_cache [num_blocks, num_head, block_size, v_head_size] FP8鏍煎紡鐨剉alue cache
 * @param slot_mapping [num_tokens] int32/int64锛屾瘡涓猼oken鍦╟ache涓殑鍋忕Щ
 * @param key_scale [num_tokens, num_head] float32鏍煎紡鐨刱ey scale杈撳叆
 * @param key_scale_cache [num_blocks, num_head, block_size, 1] float32鏍煎紡鐨刱ey scale cache
 * @param cache_layout cache甯冨眬鏍煎紡瀛楃涓诧紙榛樿"BNBD"锛? *
 * @return tuple(key_cache_out, value_cache_out, key_scale_cache_out)
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor>
ScatterPaKvCacheWithKScale(const at::Tensor &key, const at::Tensor &value, const at::Tensor &keyCache,
                           const at::Tensor &valueCache, const at::Tensor &slotMapping, const at::Tensor &keyScale,
                           const at::Tensor &keyScaleCache, std::string cacheLayout)
{
    // 妫€鏌ヨ緭鍏ensor缁村害
    TORCH_CHECK(key.dim() == 3, "key must be 3D tensor [num_tokens, num_head, k_head_size]");
    TORCH_CHECK(value.dim() == 3, "value must be 3D tensor [num_tokens, num_head, v_head_size]");
    TORCH_CHECK(keyCache.dim() == 4, "key_cache must be 4D tensor [num_blocks, num_head, block_size, k_head_size]");
    TORCH_CHECK(valueCache.dim() == 4, "value_cache must be 4D tensor [num_blocks, num_head, block_size, v_head_size]");
    TORCH_CHECK(slotMapping.dim() == 1, "slot_mapping must be 1D tensor [num_tokens]");
    TORCH_CHECK(keyScale.dim() == 2, "key_scale must be 2D tensor [num_tokens, num_head]");
    TORCH_CHECK(keyScaleCache.dim() == 4, "key_scale_cache must be 4D tensor [num_blocks, num_head, block_size, 1]");

    // check dtype consistency
    TORCH_CHECK(key.dtype() == value.dtype(), "key and value must have same dtype");
    TORCH_CHECK(key.dtype() == keyCache.dtype(), "key and key_cache must have same dtype");
    TORCH_CHECK(key.dtype() == valueCache.dtype(), "key and value_cache must have same dtype");
    TORCH_CHECK(key.dtype() == at::kFloat8_e5m2 || key.dtype() == at::kFloat8_e4m3fn,
                "key dtype must be float8_e5m2 or float8_e4m3fn");
    TORCH_CHECK(keyScale.dtype() == at::kFloat, "key_scale must be float32");
    TORCH_CHECK(keyScaleCache.dtype() == at::kFloat, "key_scale_cache must be float32");
    TORCH_CHECK(slotMapping.dtype() == at::kInt || slotMapping.dtype() == at::kLong,
                "slot_mapping dtype must be int32 or int64");

    // create output tensor (same shape and dtype as input cache)
    at::Tensor keyCacheOut = keyCache.clone();
    at::Tensor valueCacheOut = valueCache.clone();
    at::Tensor keyScaleCacheOut = keyScaleCache.clone();

    // 杞崲cache_layout涓篶har鎸囬拡
    char *cacheLayoutPtr = const_cast<char *>(cacheLayout.c_str());

    // 璋冪敤ACLNN绠楀瓙
    ACLNN_CMD(aclnnScatterPaKvCacheWithKScale, key, value, keyCacheOut, valueCacheOut, slotMapping, keyScale,
              keyScaleCacheOut, cacheLayoutPtr);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(keyCacheOut, valueCacheOut, keyScaleCacheOut);
}

// 缁戝畾C++鍑芥暟鍒癙ython妯″潡
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("scatter_pa_kv_cache_with_k_scale", &ScatterPaKvCacheWithKScale,
          "scatter_pa_kv_cache_with_k_scale operator for torch_npu");
}

} // namespace op_api
