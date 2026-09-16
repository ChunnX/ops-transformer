/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_token_permute_with_ep_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class MoeTokenPermuteWithEp : public OpDef
{
public:
    explicit MoeTokenPermuteWithEp(const char* name) : OpDef(name)
    {
        this->Input("tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("probs")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("permute_tokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("sorted_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // [上游决策点] permute_probs 维持 REQUIRED（保持与 A2 ascend910b/910_93 对外行为兼容）：
        // aclnn 文档（docs/aclnnMoeTokenPermuteWithEp.md probsOptional 条目）承诺"传入空则不输出
        // permuteProbsOut"，但 REQUIRED 声明下 aclnn 层自动生成的非空检查会拒绝该缺省调用
        // （实测 161001 "Check permuteProbsOut != nullptr failed"），存在文档与实现矛盾；
        // 若需兑现文档承诺需改为 OPTIONAL，或修订文档（950 适配实测：OPTIONAL 下 aclnn/geir 双路径
        // 缺省调用均通过；见 gen_simt_code 实验报告 01/02）。kernel 侧已通过
        // srcProbsGm.GetPhyAddr()==nullptr 保护，probs 缺省且 permute_probs 传入有效 tensor 时可正常执行。
        // 图模式（GEIR/GE）不受此声明影响：GE 对未连接的 REQUIRED 输出宽容，配合 infershape 的
        // probs dtype 回退可正常构图（实验验证，probs 缺省构图可编译执行）。
        this->Output("permute_probs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("range").AttrType(OPTIONAL).ListInt({0, 0});
        // 已知实现缺陷（950 适配实测，全平台一致）: aclnn 文档承诺"numOutTokens 在 rangeOptional
        // 为空时生效（0=全量/>0=切片/<0=负切片）"，但真实链路中"未传 range"会按本默认值以
        // ListInt({0,0}) 填充到达 tiling（tiling 探针实测：aclnn 直调 rangeOptional=nullptr 与
        // GE 图模式不设置该属性，到达形态均为 size=2 的 [0,0]，非空指针），tiling 恒走 range 分支，
        // else 回退分支（end=*numOutTokensPtr）不可达——num_out_tokens 实际永远不生效。
        // "不传 range + num_out_tokens!=0" 组合的输出为全量（[0,0] 环绕公式），与文档语义不符；
        // 950 适配实测该场景与 910 行为一致（R1a/R1b 双向实验），修复会使 910 存量 workaround
        // 调用（全量分配）从成功变报错，为保持 A2 兼容暂不修复，待上游决策：
        //   方案 a) tiling 对 [0,0] 回退 num_out_tokens（对齐文档，改变 910 行为）
        //   方案 b) 移除本默认值让 nullptr 到达（上游级修改，影响所有未设置该属性的存量构图）
        //   方案 c) 修订 aclnn 文档（删去 num_out_tokens 的生效承诺）
        this->Attr("num_out_tokens").AttrType(OPTIONAL).Int(0);
        this->Attr("padded_mode").AttrType(OPTIONAL).Bool(false);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(MoeTokenPermuteWithEp);
} // namespace ops
