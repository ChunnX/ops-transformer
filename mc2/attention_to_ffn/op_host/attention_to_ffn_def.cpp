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
 * \file attention_to_ffn_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
// AttentionToFFN算子定义类，用于将Attention阶段的输出传递到FFN阶段
class AttentionToFFN : public OpDef {
public:
    explicit AttentionToFFN(const char *name) : OpDef(name) {
        // 输入x：Attention阶段的输出张量，支持float16和bfloat16数据类型
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // 输入session_id：会话标识，用于区分不同的推理会话
        this->Input("session_id")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // 输入micro_batch_id：微批次标识，用于标识当前微批次
        this->Input("micro_batch_id")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // 输入layer_id：层标识，用于标识当前Transformer层
        this->Input("layer_id")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // 输入expert_ids：专家标识列表，用于MoE场景下的专家选择
        this->Input("expert_ids")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // 输入expert_rank_table：专家排名表，记录专家的排名信息
        this->Input("expert_rank_table")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // 输入scales：量化缩放因子，可选输入，用于量化场景
        this->Input("scales")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // 输入active_mask：激活掩码，可选输入，用于标识活跃token
        this->Input("active_mask")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // 属性group：通信组名称，用于HCCL通信
        this->Attr("group").AttrType(REQUIRED).String();
        // 属性world_size：全局通信世界大小
        this->Attr("world_size").AttrType(REQUIRED).Int();
        // 属性ffn_token_info_table_shape：FFN阶段token信息表的形状
        this->Attr("ffn_token_info_table_shape").AttrType(REQUIRED).ListInt();
        // 属性ffn_token_data_shape：FFN阶段token数据的形状
        this->Attr("ffn_token_data_shape").AttrType(REQUIRED).ListInt();
        // 属性attn_token_info_table_shape：Attention阶段token信息表的形状
        this->Attr("attn_token_info_table_shape").AttrType(REQUIRED).ListInt();
        // 属性moe_expert_num：MoE专家数量
        this->Attr("moe_expert_num").AttrType(REQUIRED).Int();
        // 属性quant_mode：量化模式，可选属性
        this->Attr("quant_mode").AttrType(OPTIONAL).Int();
        // 属性sync_flag：同步标志，可选属性
        this->Attr("sync_flag").AttrType(OPTIONAL).Int();
        // 属性ffn_start_rank_id：FFN起始rank标识，可选属性
        this->Attr("ffn_start_rank_id").AttrType(OPTIONAL).Int();
    
        // AICore算子配置：支持动态编译静态化、动态格式、动态维度、动态形状
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        // 添加ascend910_93芯片配置
        this->AICore().AddConfig("ascend910_93", aicore_config);
        // MC2配置：指定HCCL通信组
        this->MC2().HcclGroup({"group"});
    }
};

// 注册AttentionToFFN算子
OP_ADD(AttentionToFFN);
}  // namespace ops
