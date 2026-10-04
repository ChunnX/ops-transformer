// ENABLE_CV_COMM_VIA_SSBUF 单点定义于 static_slot.h（本文件首个包含头，
// 先于 kernel_operator.h 生效，驱动 SSBUF 硬件通路选择）。
#include "static_slot.h"
#include "attention_sync.h"
#include "attention_cube.h"
#include "attention_vec0_quant.h"
#include "attention_vec1_softmax.h"
#include "attention_vec1_output.h"
#include "sage_attention2_template_tiling_key.h"

#if SAGE_ATTN_DEVICE_COMPILE
#define DEBUG_PHASE_5 1

template <class ByteCountT>
__inline__ __attribute__((always_inline)) __aicore__ void InitTilingDataHelper(
    const __gm__ uint8_t *p_tilingdata, uint8_t *tilingdata, ByteCountT all_bytes)
{
    const uint64_t copy_bytes = static_cast<uint64_t>(all_bytes);
#if defined(ASCENDC_CPU_DEBUG)
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)p_tilingdata;
    uint32_t *dst = (uint32_t *)tilingdata;
    for (uint64_t i = 0; i < (copy_bytes + 3) / 4; i++) {
        *(dst + i) = *(src + i);
    }
#elif defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__) || \
      defined(__GET_CODE_CHANNEL__) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201))
    copy_data_align64(tilingdata, const_cast<__gm__ uint8_t *>(p_tilingdata), copy_bytes);
#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    uint32_t len_burst = (copy_bytes + 31) / 32;
    copy_gm_to_ubuf_align_v2(tilingdata_in_ub, const_cast<__gm__ uint8_t *>(p_tilingdata),
        0, 1, len_burst * 32, 0, 0, false, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    copy_data_align64(tilingdata, tilingdata_in_ub, copy_bytes);
#endif
    pipe_barrier(PIPE_ALL);
}

#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingDataHelper(tiling_arg, reinterpret_cast<uint8_t *>(&tiling_data), sizeof(tiling_struct))
#endif


extern "C" __global__ __aicore__ void sage_attention2(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(sage_attn::SageAttention2TilingData);
    GET_TILING_DATA_WITH_STRUCT(sage_attn::SageAttention2TilingData, t, tiling);
    AscendC::SetSysWorkspaceForce(workspace);
    __gm__ uint8_t *user_ws = workspace + AscendC::RESERVED_WORKSPACE;
#if defined(DEBUG_PHASE_5)
    // 实例化内核并执行 Init
    sage_attn::SageAttnKernel<sage_attn::kTplDtypeFp16> kernel;
    kernel.Init(query, key, value, attentionOut, softmaxLse, user_ws, &t);
    // 仅在 Cube (AIC) 执行纯 MMAD 计算与 Fixpipe 搬出测试，Vector 0 和 Vector 1 直接退出
    if ASCEND_IS_AIC {
        kernel.TestCubeStandalone();
    }
    return;
#endif
    sage_attn::SageAttnLaunch::RunByDtype(t.input_dtype, query, key, value, attentionOut, softmaxLse, user_ws, &t);
}
#else
extern "C" void sage_attention2(
    uint8_t *query, uint8_t *key, uint8_t *value,
    uint8_t *attentionOut, uint8_t *softmaxLse,
    uint8_t *workspace, uint8_t *tiling)
{
    (void)workspace;
    const auto *t = reinterpret_cast<const sage_attn::SageAttention2TilingData *>(tiling);
    sage_attn::SageAttnLaunch::RunByDtype(t->input_dtype, query, key, value, attentionOut, softmaxLse, nullptr, t);
}
#endif
