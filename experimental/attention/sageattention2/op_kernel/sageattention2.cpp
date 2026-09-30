#include "single_pass_engine.h"
#include "sage_attention2_tiling_data.h"
#include "sage_attention2_template_tiling_key.h"

namespace sage::entry {

// Validates that template input type corresponds to supported IEEE/Brain floating point formats
template <int DTypeCode>
__aicore__ inline constexpr bool IsPrecisionSupported() noexcept {
    return (DTypeCode == SAGE_TPL_BF16) || (DTypeCode == SAGE_TPL_FP16);
}

// Routes kernel parameters to the unified AttentionEngine with workspace resolution
template <int DTypeCode, bool WidePipeline, bool Dp128, bool CausalAttention>
__aicore__ inline void RouteAttentionEngine(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse, GM_ADDR workspace, const SageAttention2TilingData &td) {
    static_assert(IsPrecisionSupported<DTypeCode>(), "Unsupported precision dtype for SageAttention2");
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if constexpr (DTypeCode == SAGE_TPL_BF16) {
        sage::orchestration::AttentionEngine<bfloat16_t, WidePipeline>::template Execute<Dp128, CausalAttention>(
            q, k, v, out, lse, userWorkspace, td);
    } else if constexpr (DTypeCode == SAGE_TPL_FP16) {
        sage::orchestration::AttentionEngine<half, WidePipeline>::template Execute<Dp128, CausalAttention>(
            q, k, v, out, lse, userWorkspace, td);
    }
}

} // namespace sage::entry

template <int INPUT_DTYPE, bool WIDE, bool DP128, bool CAUSAL>
__global__ __aicore__ void sage_attention2(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(SageAttention2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SageAttention2TilingData, tilingData, tiling);

    sage::entry::RouteAttentionEngine<INPUT_DTYPE, WIDE, DP128, CAUSAL>(q, k, v, out, lse, workspace, tilingData);
}
