/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file aclnn_kv_cache_turbo_quant.h
 * @brief Encode entry used by the verified launcher.
 *
 * The runnable implementation is examples/test_aclnn_kv_cache_turbo_quant.cpp.
 * Norm and division go through aclnnLinalgVectorNorm and aclnnDiv, the same
 * path torch.linalg.vector_norm uses, so Lloyd-Max and QJL decisions match the golden.
 */
#ifndef ACLNN_KV_CACHE_TURBO_QUANT_H_
#define ACLNN_KV_CACHE_TURBO_QUANT_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void kvtq_cast_fp32(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *kv, uint8_t *fp32, uint32_t rowCount,
                    uint32_t coreNum);
void mm_fp32_transb(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *a, uint8_t *b, uint8_t *c,
                    uint8_t *workspace, uint8_t *tiling);
void kvtq_quantize(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *rotated, uint8_t *quantIdx,
                   uint8_t *residual, uint32_t rowCount, uint32_t mseBits, uint32_t coreNum);
void kvtq_store_bf16(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *normFp, uint8_t *gammaFp, uint8_t *normBf,
                     uint8_t *gammaBf, uint32_t rowCount, uint32_t coreNum);
void kvtq_sign(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *projected, uint8_t *quantQjl, uint32_t rowCount,
               uint32_t coreNum);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_KV_CACHE_TURBO_QUANT_H_
