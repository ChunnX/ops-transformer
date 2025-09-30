/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
* @brief Implement MlaProlog.

* @par Inputs:
* @li token_x: A matrix Tensor. The type support int8 and bf16.
* @li weight_dq: A matrix Tensor. The downsampling weight matrix of query. The type support int8 and bf16.
* @li weight_uq_qr: A matrix Tensor. The upsampling and positional encoding weight matrix of query.
* The type support int8 and bf16.
* @li weight_uk: A matrix Tensor. The second upsampling weight matrix of query. The type support int8 and bf16.
* @li weight_dkv_kr: A matrix Tensor. The upsampling and positional encoding weight matrix of key.
* The type support int8 and bf16.
* @li rmsnorm_gamma_cq: A matrix Tensor. The gamma factor for the rmsnorm of query. The type support float16 and bf16.
* @li rmsnorm_gamma_ckv: A matrix Tensor. The gamma factor for the rmsnorm of key. The type support float16 and bf16.
* @li rope_sin: A matrix Tensor. The position encoding sin information of each token. The type support float16 and bf16.
* @li rope_cos: A matrix Tensor. The position encoding cos information of each token. The type support float16 and bf16.
* @li cache_index: A matrix Tensor. The index of the cache in each batch. The type support int64.
* @li kv_cache: A matrix Tensor. The type support float16 and bf16.
* @li kr_cache: A matrix Tensor. The type support float16 and bf16.
* @li dequant_scale_x: A matrix Tensor. This parameter is used for dequantization after downsampling when tokenX is of the int8 type. The quantization mode of tokenX is per-token.
* The type support float.
* @li dequant_scale_w_dq: A matrix Tensor. This parameter is used for dequantization after downsampling when tokenX is of the int8 type. The quantization mode is per-channel.
* The type support float.
* @li dequantScaleWUqQr: A matrix Tensor. Parameter used for dequantization after matrix multiplication during dynamic quantization of MatmulQcQr.
* The type support float.
* @li dequant_scale_w_dkv_kr: A matrix Tensor. This parameter is used for quantization after MatmulCkvKr when tokenX is of the int8 type.
* The type support float.
* @li quantScaleCkv: A matrix Tensor. Parameter used for quantizing the RmsNormCkv output. The parameter is aclTensor on the device side.
* The type support float.
* @li quantScaleCkr: A matrix Tensor. This parameter is used for quantizing the RoPEKr output. It is aclTensor on the device side.
* The type support float.
* @li smoothScalesCq: A matrix Tensor. Smoothquant parameter required for dynamic quantization of RmsNormDq output.

* @par Attributes:
* @li rmsnorm_epsilon_cq: An optional float. The epsilon factor for the rmsnorm of query. Default: 1e-5.
* @li rmsnorm_epsilon_ckv: An optional float. The epsilon factor for the rmsnorm of key. Default: 1e-5.
* @li cache_mode: An optional int. The mode of kvcache. Default: PA_BSND.

* @par Outputs:
* query: A matrix Tensor. The type support float16 and bf16.
* query_rope: A matrix Tensor. The type support float16 and bf16.
* kv_cache_out: A matrix Tensor. The type support float16 and bf16.
* kr_cache_out: A matrix Tensor. The type support float16 and bf16.\n

* @attention Constraints:
*
*/
REG_OP(MlaProlog)
    .INPUT(token_x, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_dq, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uq_qr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(weight_uk, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight_dkv_kr, TensorType({DT_INT8, DT_BF16}))
    .INPUT(rmsnorm_gamma_cq, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rmsnorm_gamma_ckv, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_sin, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(rope_cos, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(cache_index, TensorType({DT_INT64}))
    .INPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(dequant_scale_x, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dq, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_uq_qr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale_w_dkv_kr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckv, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale_ckr, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scales_cq, TensorType({DT_FLOAT}))
    .OUTPUT(query, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kv_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(kr_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .ATTR(rmsnorm_epsilon_cq, Float, 1e-05f)
    .ATTR(rmsnorm_epsilon_ckv, Float, 1e-05f)
    .ATTR(cache_mode, String, "PA_BSND")
    .OP_END_FACTORY_REG(MlaProlog)