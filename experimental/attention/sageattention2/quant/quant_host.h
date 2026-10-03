#ifndef SAGE_ATTN_QUANT_HOST_H
#define SAGE_ATTN_QUANT_HOST_H

#include <cstdint>
#include <cstddef>

#include "quant_common.h"

namespace sage_attn {
namespace quant {

/**
 * 数学与硬件常量：唯一定义位于 ../sageattn_fp8/ascendc/tiling.h（sage_attn 命名空间），
 * 此处经 using 引入 quant 命名空间，禁止再复制数值。
 */
using sage_attn::LOG2E;
using sage_attn::S_FP8_OFFSET;
using sage_attn::SKIP_SCALE_TAU;
using sage_attn::INT8_QUANT_MAX;
using sage_attn::SCALE_EPSILON;
using sage_attn::SCALE_MAX_FP16_ACCUM;
using sage_attn::SCALE_MAX_FP32_ACCUM;
using sage_attn::FP8_E4M3_MAX;
constexpr uint32_t ALIGN_BYTES_32 = 32;                      // MTE2 / UB 32 字节对齐基准

/**
 * V 量化饱和上限三档取值（fp32+fp16 档 2.25，其余档位兜底 448.0）——唯一定义；
 * aclnn ComputeHostTiling 与 quant_fp8 均经此函数取值。
 *
 * 原 GetDefaultSmScale / FoldSmScaleLog2e / GetPaddedHeadDim（默认 sm_scale、
 * log2e 折叠、头维补齐）已删除：三者的唯一定义分别收敛于
 * python error_check.compute_sm_scale / pad_head_dim 与
 * aclnn/sage_attn_host_tiling.h 的 ComputeHostTiling，Host 侧不再保留副本。
 * 原 GetQ/K/VScaleCount、GetCompact*BlockBytes、ComputeKm、ComputeLseCorrection、
 * ConvertLseExp2ToOut 为无调用方死代码，已删除（Q/K scale 计数唯一定义在
 * quant_common.h 的 QScaleCount/KScaleCount）。
 */
inline float GetVScaleMax(uint32_t pv_accum_dtype) noexcept {
    return (pv_accum_dtype == PV_ACCUM_FP32_FP16) ? SCALE_MAX_FP16_ACCUM : SCALE_MAX_FP32_ACCUM;
}

inline uint32_t Align32(uint32_t size) noexcept {
    return ((size + ALIGN_BYTES_32 - 1) / ALIGN_BYTES_32) * ALIGN_BYTES_32;
}

} // namespace quant
} // namespace sage_attn

#endif // SAGE_ATTN_QUANT_HOST_H
