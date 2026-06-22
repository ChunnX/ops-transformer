#include "kernel_operator.h"
#include "FaTilingKey_tilingkey.h"
#include "kernel.cpp"
template <uint64_t NeedAttnMask>
__global__ AICORE void flash_attention_score(__gm__ uint8_t* query_0, __gm__ uint8_t* key_0, __gm__ uint8_t* value_0, __gm__ uint8_t* real_shift_0, __gm__ uint8_t* drop_mask_0, __gm__ uint8_t* padding_mask_0, __gm__ uint8_t* atten_mask_0, __gm__ uint8_t* prefix_0, __gm__ uint8_t* actual_seq_qlen_0, __gm__ uint8_t* actual_seq_kvlen_0, __gm__ uint8_t* q_start_idx_0, __gm__ uint8_t* kv_start_idx_0, __gm__ uint8_t* d_scale_q_0, __gm__ uint8_t* d_scale_k_0, __gm__ uint8_t* d_scale_v_0, __gm__ uint8_t* query_rope_0, __gm__ uint8_t* key_rope_0, __gm__ uint8_t* sink_0, __gm__ uint8_t* p_scale_0, __gm__ uint8_t* softmax_max_0, __gm__ uint8_t* softmax_sum_0, __gm__ uint8_t* softmax_out_0, __gm__ uint8_t* attention_out_0, __gm__ uint8_t* workspace_0, __gm__ uint8_t* tiling_0_ptr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    flash_attention_score_impl<NeedAttnMask>(query_0, key_0, value_0, real_shift_0, drop_mask_0, padding_mask_0, atten_mask_0, prefix_0, actual_seq_qlen_0, actual_seq_kvlen_0, q_start_idx_0, kv_start_idx_0, d_scale_q_0, d_scale_k_0, d_scale_v_0, query_rope_0, key_rope_0, sink_0, p_scale_0, softmax_max_0, softmax_sum_0, softmax_out_0, attention_out_0, workspace_0, tiling_0_ptr);
}
