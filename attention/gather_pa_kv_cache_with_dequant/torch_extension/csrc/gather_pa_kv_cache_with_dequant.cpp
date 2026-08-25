/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file gather_pa_kv_cache_with_dequant.cpp
 * \brief torch_npu适配：gather_pa_kv_cache_with_dequant算子的C++绑定
 *
 * 功能：从分页 KV cache 中 gather 指定 token 并反量化为 BF16，原地写入 key_ref/value_ref。
 * 封装 aclnnGatherPaKvCacheWithDequant（自定义算子，op-package 安装后由 libcust_opapi.so 提供）。
 *
 * NZ 物理布局说明：
 *   key_cache/value_cache 在 vLLM-Ascend C8 路径下物理布局为 NZ（ACL_FORMAT_FRACTAL_NZ），
 *   逻辑 tensor 仍为 ND 4D/5D。aclnn_common.h 的 ConvertType(at::Tensor) 通过公开导出 API
 *   at_npu::native::get_npu_format() 取真实 format，对非 base format 的 tensor 会把 NZ format
 *   与 storage_sizes_ 正确传给 aclCreateTensor，实现 NZ cache 零拷贝。此过程不触碰
 *   NPUStorageImpl（其 typeinfo 被 -fvisibility=hidden 隐藏，不可直接 include），故本文件
 *   仅依赖 aclnn_common.h 即可，无需 dlsym 绕开。
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

// aclDataType 枚举值（acl_base.h）：ACL_BF16 = 27
constexpr int64_t ACL_BF16_TYPE = 27;
constexpr int64_t DIM_FOUR = 4;
constexpr int64_t DIM_FIVE = 5;

/**
 * gather_pa_kv_cache_with_dequant算子的C++实现
 *
 * 功能：从分页 KV cache 中按 block_tables/seq_lens gather 指定 token，并反量化为 BF16
 *       原地写入 key_ref/value_ref（调用方预分配，算子原地 mutate）
 *
 * @param key_cache        INT8  [num_blocks, block_size, num_heads_k, head_dim_k]
 *                         （NZ 物理布局时逻辑 shape 不变，format=ACL_FORMAT_FRACTAL_NZ）
 * @param value_cache      FP8 (float8_e4m3fn) [num_blocks, block_size, num_heads_v, head_dim_v]
 * @param key_scale_cache  FLOAT  Norm: [num_blocks, block_size/num_heads_k, num_heads_k]
 *                                      或 HBS: [num_blocks, num_heads_k, block_size]
 *                         C8:   [num_heads_k, head_dim_k]    (per-channel)
 * @param value_scale      FLOAT  [num_heads_v, head_dim_v]   (per-head-per-channel, static)
 * @param block_tables     INT32  [num_batches, max_blocks_per_seq]
 * @param seq_lens         INT32  [num_batches] (或 [num_batches+1] cumsum)
 * @param key_ref          OUT BF16 [num_tokens, num_heads_k, head_dim_k]（原地写）
 * @param value_ref        OUT BF16 [num_tokens, num_heads_v, head_dim_v]（原地写）
 * @param seq_offset       optional INT32，token 起始偏移
 * @param cache_mode       "Norm" | "C8"，控制 scale 布局与 gather 逻辑
 * @param is_seq_lens_cumsum  seq_lens 是否为前缀和形式
 * @param key_dst_type     目标 dtype（aclDataType，BF16=27）
 * @param value_dst_type   目标 dtype（aclDataType，BF16=27）
 */
void GatherPaKvCacheWithDequant(const at::Tensor &key_cache, const at::Tensor &value_cache,
                                const at::Tensor &key_scale_cache, const at::Tensor &value_scale,
                                const at::Tensor &block_tables, const at::Tensor &seq_lens,
                                at::Tensor &key_ref, at::Tensor &value_ref,
                                const c10::optional<at::Tensor> &seq_offset,
                                std::string cache_mode, bool is_seq_lens_cumsum,
                                int64_t key_dst_type, int64_t value_dst_type)
{
    // 检查输入tensor维度
    TORCH_CHECK(key_cache.dim() == DIM_FOUR || key_cache.dim() == DIM_FIVE,
                "key_cache must be 4D or 5D, current dim is: ", key_cache.dim());
    TORCH_CHECK(value_cache.dim() == DIM_FOUR || value_cache.dim() == DIM_FIVE,
                "value_cache must be 4D or 5D, current dim is: ", value_cache.dim());
    TORCH_CHECK(block_tables.dim() == 2, "block_tables must be 2D [batch, max_blocks], current dim is: ",
                block_tables.dim());
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D, current dim is: ", seq_lens.dim());
    TORCH_CHECK(key_ref.dim() == 3, "key_ref must be 3D [num_tokens, num_heads_k, head_dim_k], current dim is: ",
                key_ref.dim());
    TORCH_CHECK(value_ref.dim() == 3, "value_ref must be 3D [num_tokens, num_heads_v, head_dim_v], current dim is: ",
                value_ref.dim());

    // 检查dtype
    TORCH_CHECK(key_cache.scalar_type() == at::kChar, "key_cache must be INT8, current is: ",
                key_cache.scalar_type());
    TORCH_CHECK(value_cache.scalar_type() == at::kFloat8_e4m3fn,
                "value_cache must be FP8 (float8_e4m3fn), current is: ", value_cache.scalar_type());
    TORCH_CHECK(key_scale_cache.scalar_type() == at::kFloat,
                "key_scale_cache must be FLOAT32, current is: ", key_scale_cache.scalar_type());
    TORCH_CHECK(value_scale.scalar_type() == at::kFloat,
                "value_scale must be FLOAT32, current is: ", value_scale.scalar_type());
    TORCH_CHECK(key_ref.scalar_type() == at::kBFloat16, "key_ref must be BF16, current is: ",
                key_ref.scalar_type());
    TORCH_CHECK(value_ref.scalar_type() == at::kBFloat16, "value_ref must be BF16, current is: ",
                value_ref.scalar_type());

    // 检查 cache_mode 取值（白名单与 op_host tiling.cpp 一致: Norm / C8 / Norm_NZ / C8_NZ）
    // 语义由子串决定: 含 "C8" → per-channel scale; 含 "NZ" → NZ 物理布局; "Norm" → ND + per-token scale
    TORCH_CHECK(cache_mode == "Norm" || cache_mode == "C8" || cache_mode == "Norm_NZ" ||
                    cache_mode == "C8_NZ",
                "cache_mode must be one of 'Norm'/'C8'/'Norm_NZ'/'C8_NZ', current is: ", cache_mode);

    // 检查目标 dtype（当前仅支持 BF16）
    TORCH_CHECK(key_dst_type == ACL_BF16_TYPE,
                "key_dst_type must be ACL_BF16(27), current is: ", key_dst_type);
    TORCH_CHECK(value_dst_type == ACL_BF16_TYPE,
                "value_dst_type must be ACL_BF16(27), current is: ", value_dst_type);

    // 转换 cache_mode 为 char 指针（aclnn 签名要求 char*）
    char *cacheModePtr = const_cast<char *>(cache_mode.c_str());

    // 调用ACLNN算子（inplace更新 key_ref/value_ref，无返回值）
    // 参数顺序与 aclnnGatherPaKvCacheWithDequantGetWorkspaceSize 严格一致。
    ACLNN_CMD(aclnnGatherPaKvCacheWithDequant, key_cache, value_cache, key_scale_cache, value_scale,
              block_tables, seq_lens, key_ref, value_ref, seq_offset, cacheModePtr, is_seq_lens_cumsum,
              key_dst_type, value_dst_type);
}

// 绑定C++函数到Python模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gather_pa_kv_cache_with_dequant", &GatherPaKvCacheWithDequant,
          "gather_pa_kv_cache_with_dequant operator for torch_npu");
}

} // namespace op_api
