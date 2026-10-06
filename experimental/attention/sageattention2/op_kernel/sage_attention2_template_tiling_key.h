#pragma once

/**
 * \file sage_attention2_template_tiling_key.h
 * \brief M5 编译期位型特化：输入位型档位常量、归一与 tiling key 编码的唯一定义点
 *        （遵循标准单头模板档位声明模式）。
 *
 * canonical 模板档位机制（ASCENDC_TPL_ARGS_DECL / GET_TPL_TILING_KEY，头
 * ascendc/host_api/tiling/template_argument.h）按构建形态分三态（CANN 9.2 实测）：
 * - ASCENDC_TPL_PRE：opc autogen 预解析态，宏展开为 @@ 标记供解析器提取档位表；
 * - ASCENDC_TPL_KERNEL：autogen 设备核编译态，decl/SEL 宏恒空展开，头本身设备安全；
 * - 其余（宿主 tiling/裸编译）：ParamStruct 表 + GET_TPL_TILING_KEY 运行时编码，
 *   依赖 iostream/printf——设备编译（裸 bisheng -xasc，无 ASCENDC_TPL_KERNEL 注入）
 *   不可纳入本头，否则 host 分支的 printf/iostream 在设备域硬报错。
 * 故引入守卫为：宿主编译可纳入（GET_TPL_TILING_KEY 消费方 op_host 需要）；
 * 设备编译仅在 OPP autogen 注入 ASCENDC_TPL_KERNEL 时纳入。缺失时本头的编码域
 * 常量独立成立，核侧经 SageAttnLaunch::RunByDtype 运行时一次分派（prologue 单次，
 * 热路径零位型分支），两形态同档位实例的算术逐位一致。
 */

#include <cstdint>

#include "../sageattn_fp8/ascendc/tiling.h" // __aicore__ 宏桩首个定义点（宿主展开为空）

#if defined(__has_include)
#if __has_include("ascendc/host_api/tiling/template_argument.h")
#if !defined(SAGE_ATTN_DEVICE_COMPILE) || !SAGE_ATTN_DEVICE_COMPILE || defined(ASCENDC_TPL_KERNEL)
#include "ascendc/host_api/tiling/template_argument.h"
#define SAGE_HAS_CANONICAL_TPL_ARGS 1
#endif
#endif
#endif

namespace sage_attn {

inline constexpr uint32_t kTplDtypeShift = 20;
inline constexpr uint32_t kTplDtypeFp16 = 1; // 1=fp16
inline constexpr uint32_t kTplDtypeBf16 = 2; // 2=bf16

/**
 * 输入位型归一（0 视作 fp16 缺省）。host/kernel 共用唯一定义点，
 * 消费侧禁止再展开三元式（原 SageAttnKernel::DtypeMode 的定义迁移至此）。
 */
[[nodiscard]] __aicore__ __host__ inline __attribute__((always_inline)) constexpr uint32_t NormDtypeMode(uint32_t raw)
{
    return (raw == kTplDtypeBf16) ? kTplDtypeBf16 : kTplDtypeFp16;
}

/**
 * tiling key 单点编码（裸 bisheng 门禁回落形态）：入参为原始/归一化 input_dtype
 * 编码，返回 dtype nibble（docs/sageattention2_design.md "tilingkey 规划策略"
 * 冻结 bits 20-23 = input_dtype；其余 key 位 M5 未启用恒 0）。canonical 机制
 * 可用时 host 侧经 SAGE_TPL_TILING_KEY 宏走 GET_TPL_TILING_KEY 单点。
 */
[[nodiscard]] __aicore__ __host__ inline __attribute__((always_inline)) constexpr uint32_t TplTilingKey(uint32_t dtype_mode)
{
    return NormDtypeMode(dtype_mode) << kTplDtypeShift;
}
} // namespace sage_attn

#ifdef SAGE_HAS_CANONICAL_TPL_ARGS
// canonical 模板档位声明（标准置头单点，全局域——GET_TPL_TILING_KEY
// 展开按无限定名取 g_tilingDeclareParams）：宿主 tiling TU 依此获得档位参数表
// （static 内链，随 TU 局部）；autogen 设备核态（ASCENDC_TPL_KERNEL）宏恒空展开；
// PRE 预解析态提取档位表生成框架分派。值域 1/2 与 kTplDtypeFp16/kTplDtypeBf16
// 一致（字面量直书：PRE 解析器不做 constexpr 求值，标准字面量形态）。
ASCENDC_TPL_ARGS_DECL(SageAttention2, ASCENDC_TPL_DTYPE_DECL(INPUT_DTYPE, 1, 2));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(INPUT_DTYPE, 1, 2)));
#endif

/**
 * host tiling / kernel 共用的 tiling key 计算宏：canonical 机制可用时对齐
 * 标准的 GET_TPL_TILING_KEY 单点；裸 bisheng 门禁为本仓冻结编码。
 */
#ifdef SAGE_HAS_CANONICAL_TPL_ARGS
#define SAGE_TPL_TILING_KEY(dtype_mode) GET_TPL_TILING_KEY(dtype_mode)
#else
#define SAGE_TPL_TILING_KEY(dtype_mode) ::sage_attn::TplTilingKey(dtype_mode)
#endif
