/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#if __CCE_AICORE__ == 310
#include "arc35/generic_block_sparse_attention_grad.h"
#else
#include "arc22/generic_block_sparse_attention_grad_interface.h"
#endif
// ============================================================================
// Kernel Entry Point
// ============================================================================

/**
 * @brief GenericBlockSparseAttentionGrad算子的kernel入口函数
 *
 * 这是NPU上执行的入口点，使用extern "C"以避免C++名称修饰，
 * 使用__global__和__aicore__修饰符标识为全局kernel函数
 */
extern "C" __global__ __aicore__ void generic_block_sparse_attention_grad(__gm__ uint8_t *query,
                                                                          __gm__ uint8_t *key,
                                                                          __gm__ uint8_t *value,
                                                                          __gm__ uint8_t *dout,
                                                                          __gm__ uint8_t *out,
                                                                          __gm__ uint8_t *softmaxLse,
                                                                          __gm__ uint8_t *rsvdBlockIdx,
                                                                          __gm__ uint8_t *rsvdBlockCount,
                                                                          __gm__ uint8_t *metadata,
                                                                          __gm__ uint8_t *attentionMask,
                                                                          __gm__ uint8_t *cuSeqLengthsQ,
                                                                          __gm__ uint8_t *cuSeqLengthsKv,
                                                                          __gm__ uint8_t *sequsedQ,
                                                                          __gm__ uint8_t *sequsedKv,
                                                                          __gm__ uint8_t *blockShape,
                                                                          __gm__ uint8_t *dq,
                                                                          __gm__ uint8_t *dk,
                                                                          __gm__ uint8_t *dv,
                                                                          __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(100)) {
        GBSAG::GenericBlockSparseAttentionGradInfer<half, 0>(
            query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attentionMask,
            cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, blockShape, dq, dk, dv, user, tiling);
    } else if (TILING_KEY_IS(101)) {
        GBSAG::GenericBlockSparseAttentionGradInfer<half, 1>(
            query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attentionMask,
            cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, blockShape, dq, dk, dv, user, tiling);
    } else if (TILING_KEY_IS(102)) {
        GBSAG::GenericBlockSparseAttentionGradInfer<half, 2>(
            query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attentionMask,
            cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, blockShape, dq, dk, dv, user, tiling);
    } else if (TILING_KEY_IS(110)) {
        GBSAG::GenericBlockSparseAttentionGradInfer<bfloat16_t, 0>(
            query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attentionMask,
            cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, blockShape, dq, dk, dv, user, tiling);
    } else if (TILING_KEY_IS(111)) {
        GBSAG::GenericBlockSparseAttentionGradInfer<bfloat16_t, 1>(
            query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attentionMask,
            cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, blockShape, dq, dk, dv, user, tiling);
    } else if (TILING_KEY_IS(112)) {
        GBSAG::GenericBlockSparseAttentionGradInfer<bfloat16_t, 2>(
            query, key, value, dout, out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata, attentionMask,
            cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, blockShape, dq, dk, dv, user, tiling);
    }
}
