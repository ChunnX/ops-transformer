/*!
 * \file grouped_bias_add_grad.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "grouped_bias_add_grad_base.h"
#include "grouped_bias_add_grad_unequal_c_perf.h"

using namespace GroupedBiasAddGradAll;

#define TWO_DIMS_HALF_PERF 1001010
#define TWO_DIMS_FLOAT_PERF 1001011
#define TWO_DIMS_BFLOAT16_PERF 1001012
#define TWO_DIMS_HALF_USEUB_PERF 1001110
#define TWO_DIMS_FLOAT_USEUB_PERF 1001111
#define TWO_DIMS_BFLOAT16_USEUB_PERF 1001112

#define TWO_DIMS_HALF_GRP64_PERF 1011010
#define TWO_DIMS_FLOAT_GRP64_PERF 1011011
#define TWO_DIMS_BFLOAT16_GRP64_PERF 1011012
#define TWO_DIMS_HALF_USEUB_GRP64_PERF 1011110
#define TWO_DIMS_FLOAT_USEUB_GRP64_PERF 1011111
#define TWO_DIMS_BFLOAT16_USEUB_GRP64_PERF 1011112

template <typename T, typename G, const uint32_t USE_TYPE>
__aicore__ inline void InvokeTemplateGroupedUnequalCPerf(
    GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias, GM_ADDR userWS,
    const GroupedBiasAddGradTilingData& tilingData)
{
    GroupedBiasAddGradUnequalCPerf<T, G, USE_TYPE> op;
    op.Init(grad_y, group_idx, grad_bias, userWS, tilingData);
    op.Process();
}

extern "C" __global__ __aicore__ void grouped_bias_add_grad(
    GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias, GM_ADDR workspace, GM_ADDR tiling_data)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling_data);

    if (TILING_KEY_IS(TWO_DIMS_FLOAT_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<float, int32_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<half, int32_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<bfloat16_t, int32_t, USE_WS>(
            grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_FLOAT_USEUB_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<float, int32_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF_USEUB_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<half, int32_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16_USEUB_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<bfloat16_t, int32_t, USE_UB>(
            grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_FLOAT_USEUB_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<float, int64_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF_USEUB_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<half, int64_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16_USEUB_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<bfloat16_t, int64_t, USE_UB>(
            grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_FLOAT_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<float, int64_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<half, int64_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<bfloat16_t, int64_t, USE_WS>(
            grad_y, group_idx, grad_bias, userWS, tilingData);
    }
}