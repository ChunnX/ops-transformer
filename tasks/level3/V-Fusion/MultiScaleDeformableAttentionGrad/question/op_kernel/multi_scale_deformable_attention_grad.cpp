#include "kernel_operator.h"

extern "C" __global__ __aicore__ void multi_scale_deformable_attention_grad(
    GM_ADDR value_gm, GM_ADDR spatial_shapes_gm,
    GM_ADDR level_start_index_gm, GM_ADDR sampling_loc_gm,
    GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
    GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm,
    GM_ADDR grad_attn_weight_gm, GM_ADDR workspace,
    GM_ADDR tiling)
{
    // TODO: 实现多尺度可变形注意力反向传播 kernel
    // 1. GET_TILING_DATA 获取 tiling 参数
    // 2. 对每个 (batch, head, level, point, query) 计算：
    //    a. 双线性插值采样 value 张量
    //    b. 计算对 value、sampling_locations、attention_weights 的梯度
    //    c. 使用 AtomicAdd 累加梯度到输出
}
