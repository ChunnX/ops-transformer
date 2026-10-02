/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file sage_attention2_def.cpp
 * \brief Ascend CANN 官方标准 REG_OP 与 OpDef 算子原型注册
 */

#include "sage_attention2_def.h"

namespace ge {

/**
 * @brief SageAttention2: 针对 Ascend 950PR 硬件深度优化的极速 FP8/INT8 量化注意力算子。
 *
 * 核心创新点与特性：
 * 1. 严格支持 qk_quant_gran 量化粒度配置 (per_thread 与 per_warp)，绝不阉割。
 * 2. 严格支持 pv_accum_dtype 累加精度档位 (fp32, fp32+fp32, fp32+fp16)。
 * 3. 支持平滑化因子 smooth_k 与 smooth_v。
 * 4. 支持因果掩码 (is_causal) 与任意 BNSD/HND, BSND/NHD 张量布局。
 *
 * @par Inputs:
 * @li query: [B, Hq, Sq, D] 或 [B, Sq, Hq, D]，数据类型支持 float16, bfloat16。
 * @li key: [B, Hkv, Sk, D] 或 [B, Sk, Hkv, D]，数据类型支持 float16, bfloat16。
 * @li value: [B, Hkv, Sk, D] 或 [B, Sk, Hkv, D]，数据类型支持 float16, bfloat16。
 * @li key_mean: [B, Hkv, D]，可选输入，Key 均值向量 (smooth_k 启用时使用)，类型为 float32 或 float16。
 * @li lse_correction: [B, Hq, Sq]，可选输入，LSE 修正项张量，类型为 float32。
 *
 * @par Outputs:
 * @li output: [B, Hq, Sq, D] 或 [B, Sq, Hq, D]，形状与数据类型同 query。
 * @li lse: [B, Hq, Sq]，可选输出，Softmax 归一化对数和，类型为 float32。
 *
 * @par Attributes:
 * @li tensor_layout: String，张量内存排布格式 ("BNSD" 或 "BSND")，默认 "BNSD"。
 * @li is_causal: Bool，因果注意力下三角掩码开关，默认 false。
 * @li qk_quant_gran: String，QK 矩阵乘量化粒度 ("per_thread", "per_warp")，默认 "per_thread"。
 * @li sm_scale: Float，Softmax 缩放系数，默认 0.0f (0.0 表示内部采用 1.0 / sqrt(head_dim)，负值非法)。
 * @li pv_accum_dtype: String，PV 矩阵乘累加中间精度 ("fp32", "fp32+fp32", "fp32+fp16")，默认 "fp32+fp16"。
 * @li smooth_k: Bool，Key 平滑减法开关，默认 true。
 * @li smooth_v: Bool，Value 平滑减法开关，默认 false。
 */
REG_OP(SageAttention2)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_mean, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(lse_correction, TensorType({DT_FLOAT}))
    .OUTPUT(output, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(lse, TensorType({DT_FLOAT}))
    .ATTR(tensor_layout, String, "BNSD")
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
    explicit SageAttention2(const char *name) : OpDef(name)
    {
        // 1. 必选输入张量（DataTypeList/FormatList 为本版本 OpDef 标准列表接口，
        //    FormatList 单元素 ND 可与多 dtype 选项对齐）
        this->Input("query")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("key")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("value")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        // 2. 可选平滑与修正输入张量
        this->Input("key_mean")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("lse_correction")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        // 3. 输出张量
        this->Output("output")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("lse")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        // 4. 算子属性声明（OPTIONAL + 冻结默认值，对齐开源 core.py 签名：
        //    gran 默认 per_thread、accum 默认 fp32+fp16，消除 REQUIRED 却带默认值的矛盾）
        this->Attr("tensor_layout").AttrType(OPTIONAL).String("BNSD");
        this->Attr("is_causal").AttrType(OPTIONAL).Bool(false);
        this->Attr("qk_quant_gran").AttrType(OPTIONAL).String("per_thread");
        this->Attr("sm_scale").AttrType(OPTIONAL).Float(0.0f);
        this->Attr("pv_accum_dtype").AttrType(OPTIONAL).String("fp32+fp16");
        this->Attr("smooth_k").AttrType(OPTIONAL).Bool(true);
        this->Attr("smooth_v").AttrType(OPTIONAL).Bool(false);
        // return_lse 不设属性：由 lse 输出张量的可选性承载，OP_ATTR 保持 7 项与注册一致。

        // 5. AI Core 硬件平台配置与执行入口映射（收敛为 Ascend 950PR 单平台）
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicShapeSupportFlag(true)
            .DynamicRankSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "sageattention2")
            .ExtendCfgInfo("opInterface.value", "sage_attention2");

        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(SageAttention2);

} // namespace ops
