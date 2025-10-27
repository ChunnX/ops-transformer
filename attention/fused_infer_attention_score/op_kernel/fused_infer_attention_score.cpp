/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "fused_infer_attention_score_tilingkey.h"
// ifa must include before pfa
#define FIA_ENABLE_MLA
#include "../../incre_flash_attention/op_kernel/incre_flash_attention.cpp"
#include "../../prompt_flash_attention/op_kernel/prompt_flash_attention.cpp"

#include "fused_infer_attention_score_v3.cpp"

extern "C" __global__ __aicore__ void fused_infer_attention_score(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                             __gm__ uint8_t* pse_shift, __gm__ uint8_t* attenMask,
                                                             __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                             __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                             __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                             __gm__ uint8_t* quant_offset2, __gm__ uint8_t* antiquantScale,
                                                             __gm__ uint8_t* antiquantOffset, __gm__ uint8_t* blocktable,
                                                             __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                                             __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
                                                             __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset,
                                                             __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                                             __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope, __gm__ uint8_t* keyRopeAntiquantScale, __gm__ uint8_t* dequantScaleQuery,
                                                             __gm__ uint8_t* learnableSink, __gm__ uint8_t* qStartIdx, __gm__ uint8_t* kvStartIdx,
                                                             __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                                             __gm__ uint8_t* tiling) {
  // judge ifa or pfa or fia by range of tilingKey
  if(TILING_KEY_VAR >= PFA_FlAG_TILING) { // 10^18
      prompt_flash_attention_FIAS(query, key, value, pse_shift, attenMask, actualSeqLengths, 
                                  actualSeqLengthsKV, deq_scale1, quant_scale1,
                                  deq_scale2, quant_scale2, quant_offset2, antiquantScale, 
                                  antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize, 
                                  keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, 
                                  valueAntiquantOffset, keySharedPrefix, valueSharedPrefix, 
                                  actualSharedPrefixLen, queryRope, keyRope, learnableSink, 
                                  attentionOut, softmaxLse, workspace, tiling);
  } else if (TILING_KEY_VAR >= FIA_FLAG_TILING) { // 10^17
    fused_infer_attention(query, key, value, pse_shift, attenMask, actualSeqLengths,
                          actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                          quant_offset2, antiquantScale, antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize,
                          keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                          keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, keyRopeAntiquantScale,
                          attentionOut, softmaxLse, workspace, tiling);
  } else {
    incre_flash_attention_FIAS(query, key, value, pse_shift, attenMask, actualSeqLengths,
                          actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                          quant_offset2, antiquantScale, antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize,
                          keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                          keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, keyRopeAntiquantScale, dequantScaleQuery,
                          attentionOut, softmaxLse, workspace, tiling);
  }
}
