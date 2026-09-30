#include "../orchestration/single_pass_engine.h"
#include "sage_attention2_tiling_data.h"
#include "sage_attention2_template_tiling_key.h"

template <int INPUT_DTYPE, bool WIDE>
__global__ __aicore__ void sage_attention2(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse, GM_ADDR workspace,
                                           GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(SageAttention2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SageAttention2TilingData, tilingData, tiling);
    GM_ADDR work = AscendC::GetUserWorkspace(workspace);

    static_assert(INPUT_DTYPE == SAGE_TPL_FP16 || INPUT_DTYPE == SAGE_TPL_BF16, "Unsupported input dtype for SageAttention2");

    if constexpr (INPUT_DTYPE == SAGE_TPL_BF16) {
        sage::orchestration::AttentionEngine<bfloat16_t, WIDE>::Execute(q, k, v, out, lse, work, tilingData);
    } else {
        sage::orchestration::AttentionEngine<half, WIDE>::Execute(q, k, v, out, lse, work, tilingData);
    }
}
