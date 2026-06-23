/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "kernel_operator.h"
#include "sparse_attention_score_prefill_kernel_interface.cpp"

constexpr uint64_t SASA_PREFILL_BF16_D128_TILING = 20001;

extern "C" __global__ __aicore__ void sparse_attention_score_prefill(
    __gm__ uint8_t* query,
    __gm__ uint8_t* key,
    __gm__ uint8_t* value,
    __gm__ uint8_t* selectIdx,
    __gm__ uint8_t* blockTable,
    __gm__ uint8_t* kvToQSelectIdx,
    __gm__ uint8_t* kvToQSelectNum,
    __gm__ uint8_t* kvToQSlotIdx,
    __gm__ uint8_t* kvBlockValidSize,
    __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* actualSeqLengthsKv,
    __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* workspace,
    __gm__ uint8_t* tiling)
{
    if (TILING_KEY_VAR == SASA_PREFILL_BF16_D128_TILING) {
        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 310)
        TILING_KEY_IS(SASA_PREFILL_BF16_D128_TILING);
        SasaPrefillInferIntf<bfloat16_t, bfloat16_t, float>(
            query, key, value, selectIdx, blockTable,
            kvToQSelectIdx, kvToQSelectNum, kvToQSlotIdx, kvBlockValidSize,
            actualSeqLengths, actualSeqLengthsKv,
            attentionOut, user, tiling);
#endif
    }
}
