#ifndef SAGE_ATTENTION2_DEF_H
#define SAGE_ATTENTION2_DEF_H

#include <cstddef>

#include "graph/operator_reg.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"

namespace ge {

/**
 * @brief SageAttention2: 针对 Ascend CANN 官方标准 REG_OP 算子原型注册
 *
 * 输入/输出/属性契约：
 * 1. query/key/value: 必选输入，支持 DT_FLOAT16 / DT_BF16。
 * 2. key_mean: 可选输入，Key 均值向量 (smooth_k 启用时使用)，类型为 DT_FLOAT。
 * 3. lse_correction: 可选输入，LSE 修正项张量，类型为 DT_FLOAT。
 * 4. output: 必选输出，形状与数据类型同 query。
 * 5. lse: 可选输出，Softmax 归一化对数和，类型为 DT_FLOAT。
 * 6. 属性：tensor_layout("HND"), is_causal(false), qk_quant_gran("per_thread"),
 *    sm_scale(0.0f), pv_accum_dtype("fp32+fp16"), smooth_k(true), smooth_v(false)。
 */
REG_OP(SageAttention2)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(output, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(lse, TensorType({DT_FLOAT}))
    .ATTR(tensor_layout, String, "HND")
    .ATTR(is_causal, Bool, false)
    .ATTR(qk_quant_gran, String, "per_thread")
    .ATTR(sm_scale, Float, 0.0f)
    .ATTR(pv_accum_dtype, String, "fp32+fp16")
    .ATTR(smooth_k, Bool, true)
    .ATTR(smooth_v, Bool, false)
    .OP_END_FACTORY_REG(SageAttention2);

} // namespace ge

namespace ops {

class SageAttention2 : public OpDef {
public:
    explicit SageAttention2(const char *name);
};

/**
 * SageAttention2 参数索引（唯一定义点，严格对齐 REG_OP 与 OpDef 声明顺序）
 */
namespace sage_attn2_idx {
constexpr size_t IN_QUERY = 0;
constexpr size_t IN_KEY = 1;
constexpr size_t IN_VALUE = 2;

constexpr size_t OUT_OUTPUT = 0;
constexpr size_t OUT_LSE = 1;

constexpr size_t ATTR_TENSOR_LAYOUT = 0;
constexpr size_t ATTR_IS_CAUSAL = 1;
constexpr size_t ATTR_QK_QUANT_GRAN = 2;
constexpr size_t ATTR_SM_SCALE = 3;
constexpr size_t ATTR_PV_ACCUM_DTYPE = 4;
constexpr size_t ATTR_SMOOTH_K = 5;
constexpr size_t ATTR_SMOOTH_V = 6;
} // namespace sage_attn2_idx

/**
 * RuntimeAttrs 字符串属性读取封装（唯一定义点）：
 * 标准 CANN gert::RuntimeAttrs::GetStr(idx) -> const char*
 */
inline const char *ReadStringAttr(const gert::RuntimeAttrs *attrs, size_t idx)
{
    return (attrs != nullptr) ? attrs->GetStr(idx) : nullptr;
}

ge::graphStatus InferShapeSageAttention2(gert::InferShapeContext *context);
ge::graphStatus InferDataTypeSageAttention2(gert::InferDataTypeContext *context);

} // namespace ops

#endif // SAGE_ATTENTION2_DEF_H
