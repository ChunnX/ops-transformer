#ifndef SAGE_ATTN_PROTO_H
#define SAGE_ATTN_PROTO_H

#include <cstdint>
#include <cstring>

// 接口数值编码（gran/accum/layout）与共享数学常量的唯一定义位于
// ../sageattn_fp8/ascendc/tiling.h（Host/Device 共享 ABI）；本头经 include
// 复用同一套枚举，并在下方提供唯一的 "字符串属性 -> 数值编码" 映射，
// 禁止在任何 TU 再引入第二套编码表或映射表。
//
// （原移植态两段式 C-ABI 声明、aclTensor/aclOpExecutor 本地桩与错误码词汇
//  已随 aclnn/sage_attn_host.cpp 删除收敛至交付态：两段式 ABI 唯一面为
//  op_api/aclnn_sage_attention2.h 的字符串属性签名；错误码词汇表归
//  sage_attn_host_tiling.h。本头只保留编码唯一定义点。）
#include "../sageattn_fp8/ascendc/tiling.h"

namespace sage_attn {

/**
 * 接口字符串协议 -> 数值编码的唯一映射（C++ 侧唯一定义点）。
 * op_host TilingImpl/InferShape（gert 字符串属性）与 GE 缺省折叠均经此映射；
 * 非法串按冻结签名默认值兜底（per_thread / fp32+fp16 / HND）；严格校验由各
 * 入口的前置检查负责（op_api 对 layout 白名单校验，非法取值直接报错）。
 */
// 冻结签名默认值（C++ 侧唯一定义点；op_host 缺省折叠与 op_api 文档共用；
// 与任务书 §2.3 及 Python 侧 error_check.DEFAULT_* 逐项对应）
constexpr const char *kDefaultLayout = "HND";
constexpr const char *kDefaultQkQuantGran = "per_thread";
constexpr const char *kDefaultPvAccumDtype = "fp32+fp16";

inline int64_t TensorLayoutCodeFromString(const char *layout)
{
    return (std::strcmp(layout, "NHD") == 0) ? static_cast<int64_t>(LAYOUT_NHD)
                                             : static_cast<int64_t>(LAYOUT_HND);
}

inline int64_t QuantGranCodeFromString(const char *gran)
{
    return (std::strcmp(gran, "per_warp") == 0) ? static_cast<int64_t>(QUANT_GRAN_PER_WARP)
                                : static_cast<int64_t>(QUANT_GRAN_PER_THREAD);
}

inline int64_t PvAccumCodeFromString(const char *accum)
{
    if (std::strcmp(accum, "fp32") == 0) {
        return static_cast<int64_t>(PV_ACCUM_FP32);
    }
    if (std::strcmp(accum, "fp32+fp32") == 0) {
        return static_cast<int64_t>(PV_ACCUM_FP32_FP32);
    }
    return static_cast<int64_t>(PV_ACCUM_FP32_FP16);
}

} // namespace sage_attn

#endif // SAGE_ATTN_PROTO_H
