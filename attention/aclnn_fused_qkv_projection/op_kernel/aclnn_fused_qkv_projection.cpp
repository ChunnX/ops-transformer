#include "aclnn_fused_qkv_projection.h"
#include "aclnn_fused_qkv_projection_tiling_key.h"

template <uint64_t schMode, uint64_t dtype>
__global__ __aicore__ void aclnn_fused_qkv_projection(
    GM_ADDR hiddenStates, GM_ADDR weight, GM_ADDR bias,
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    REGISTER_TILING_DEFAULT(AclnnFusedQkvProjectionTilingData);
    GET_TILING_DATA_WITH_STRUCT(AclnnFusedQkvProjectionTilingData, tilingData, tiling);
    
    if constexpr (dtype == TPL_DTYPE_FLOAT16) {
        KernelFusedQkvProjection<half> op;
        op.Init(hiddenStates, weight, bias, query, key, value, workspace, &tilingData);
        op.Process();
    } else {
        KernelFusedQkvProjection<float> op;
        op.Init(hiddenStates, weight, bias, query, key, value, workspace, &tilingData);
        op.Process();
    }
}
