// ENABLE_CV_COMM_VIA_SSBUF 单点定义于 static_slot.h（本文件首个包含头，
// 先于 kernel_operator.h 生效，驱动 SSBUF 硬件通路选择）。
#include "static_slot.h"
#include "attention_sync.h"
#include "attention_cube.h"
#include "attention_vec0_quant.h"
#include "attention_vec1_softmax.h"
#include "attention_vec1_output.h"
#include "sage_attention2_template_tiling_key.h"

#if SAGE_ATTN_KERNEL_API && SAGE_ATTN_DEVICE_COMPILE
#ifndef REGISTER_TILING_DEFAULT
#define REGISTER_TILING_DEFAULT(tiling_struct)
#endif
#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    for (uint32_t i = 0; i < sizeof(tiling_struct); ++i) { reinterpret_cast<uint8_t *>(&tiling_data)[i] = tiling_arg[i]; }
#endif
#endif

extern "C" __global__ __aicore__ void sage_attention2(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if SAGE_ATTN_DEVICE_COMPILE
    REGISTER_TILING_DEFAULT(sage_attn::SageAttention2TilingData);
    GET_TILING_DATA_WITH_STRUCT(sage_attn::SageAttention2TilingData, t, tiling);
    AscendC::SetSysWorkspaceForce(workspace);
    __gm__ uint8_t *user_ws = workspace + AscendC::RESERVED_WORKSPACE;
    sage_attn::SageAttnLaunch::RunByDtype(t.input_dtype, query, key, value, attentionOut, softmaxLse, user_ws, &t);
#else
    (void)workspace;
    const auto *t = reinterpret_cast<const sage_attn::SageAttention2TilingData *>(tiling);
    sage_attn::SageAttnLaunch::RunByDtype(t->input_dtype, query, key, value, attentionOut, softmaxLse, nullptr, t);
#endif
}
