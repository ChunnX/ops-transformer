/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_COMBINE_REGISTRATION_H
#define MC2_COMBINE_REGISTRATION_H

#include "register/op_def_registry.h"

namespace ops {
namespace Mc2OpDef {

inline OpAICoreConfig BuildCombineAICoreConfig(const char *jitFlag)
{
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("prebuildPattern.value", "Opaque")
        .ExtendCfgInfo("jitCompile.flag", jitFlag)
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
    return config;
}

inline OpAICoreConfig BuildCombineAICoreConfig(const char *jitFlag, const char *opFile)
{
    auto config = BuildCombineAICoreConfig(jitFlag);
    config.ExtendCfgInfo("opFile.value", opFile);
    return config;
}

inline void DefineCombineOutputX(OpDef &op)
{
    op.Output("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
}

inline void DefineCombineEpAttributes(OpDef &op)
{
    op.Attr("ep_world_size").AttrType(REQUIRED).Int();
    op.Attr("ep_rank_id").AttrType(REQUIRED).Int();
    op.Attr("moe_expert_num").AttrType(REQUIRED).Int();
}

inline void DefineCombineTpAttributes(OpDef &op)
{
    op.Attr("tp_world_size").AttrType(OPTIONAL).Int(0);
    op.Attr("tp_rank_id").AttrType(OPTIONAL).Int(0);
}

inline void DefineCombineExpertAttributes(OpDef &op, int64_t sharedExpertNum)
{
    op.Attr("expert_shard_type").AttrType(OPTIONAL).Int(0);
    op.Attr("shared_expert_num").AttrType(OPTIONAL).Int(sharedExpertNum);
    op.Attr("shared_expert_rank_num").AttrType(OPTIONAL).Int(0);
    op.Attr("global_bs").AttrType(OPTIONAL).Int(0);
}

inline void DefineCombineOutputAttributes(OpDef &op)
{
    op.Attr("out_dtype").AttrType(OPTIONAL).Int(0);
    op.Attr("comm_quant_mode").AttrType(OPTIONAL).Int(0);
    op.Attr("group_list_type").AttrType(OPTIONAL).Int(0);
}

inline void DefineCombineSpecialExpertAttributes(OpDef &op)
{
    op.Attr("zero_expert_num").AttrType(OPTIONAL).Int(0);
    op.Attr("copy_expert_num").AttrType(OPTIONAL).Int(0);
    op.Attr("const_expert_num").AttrType(OPTIONAL).Int(0);
}

// Registration order is part of the positional attribute ABI used by tiling.
inline void DefineCombineAttributes(OpDef &op, int64_t sharedExpertNum)
{
    op.Attr("group_ep").AttrType(REQUIRED).String();
    DefineCombineEpAttributes(op);
    op.Attr("group_tp").AttrType(OPTIONAL).String("");
    DefineCombineTpAttributes(op);
    DefineCombineExpertAttributes(op, sharedExpertNum);
    DefineCombineOutputAttributes(op);
}

inline void DefineCombineStageAttributes(OpDef &op)
{
    op.Attr("group_ep").AttrType(REQUIRED).String();
    DefineCombineEpAttributes(op);
    DefineCombineExpertAttributes(op, 1);
    op.Attr("comm_quant_mode").AttrType(OPTIONAL).Int(0);
    op.Attr("comm_type").AttrType(OPTIONAL).Int(0);
    op.Attr("comm_alg").AttrType(OPTIONAL).String("");
}

} // namespace Mc2OpDef
} // namespace ops

#endif // MC2_COMBINE_REGISTRATION_H
