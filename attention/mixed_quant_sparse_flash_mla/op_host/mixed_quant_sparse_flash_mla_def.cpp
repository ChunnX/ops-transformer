/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mixed_quant_sparse_flash_mla_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> mqsmlaBf16Dtype = {ge::DT_BF16};
static const std::vector<ge::DataType> mqsmlaKvDtype = {ge::DT_FLOAT8_E4M3FN};
static const std::vector<ge::DataType> mqsmlaIdxDtype = {ge::DT_INT32};
static const std::vector<ge::DataType> mqsmlaFp32Dtype = {ge::DT_FLOAT};
static const std::vector<ge::Format> mqsmlaNdFmt = {ge::FORMAT_ND};

class MixedQuantSparseFlashMla : public OpDef {
public:
    explicit MixedQuantSparseFlashMla(const char *name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataTypeList(mqsmlaBf16Dtype).FormatList(mqsmlaNdFmt).AutoContiguous();
        this->Input("ori_kv").ParamType(OPTIONAL).DataType(mqsmlaKvDtype).FormatList(mqsmlaNdFmt).IgnoreContiguous();
        this->Input("cmp_kv").ParamType(OPTIONAL).DataType(mqsmlaKvDtype).FormatList(mqsmlaNdFmt).IgnoreContiguous();
        this->Input("ori_sparse_indices").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("cmp_sparse_indices").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("ori_block_table").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("cmp_block_table").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("cu_seqlens_q").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("cu_seqlens_ori_kv").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("cu_seqlens_cmp_kv").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("seqused_q").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("seqused_ori_kv").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("seqused_cmp_kv").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("cmp_residual_kv").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("ori_topk_length").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("cmp_topk_length").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Input("sinks").ParamType(OPTIONAL).DataTypeList(mqsmlaFp32Dtype).FormatList(mqsmlaNdFmt).AutoContiguous();
        this->Input("metadata").ParamType(OPTIONAL).DataTypeList(mqsmlaIdxDtype).FormatList(mqsmlaNdFmt)
            .AutoContiguous();
        this->Output("attn_out").ParamType(REQUIRED).DataTypeList(mqsmlaBf16Dtype).FormatList(mqsmlaNdFmt);
        this->Output("softmax_lse").ParamType(OPTIONAL).DataTypeList(mqsmlaFp32Dtype).FormatList(mqsmlaNdFmt);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int();
        this->Attr("rope_head_dim").AttrType(OPTIONAL).Int();
        this->Attr("softmax_scale").AttrType(OPTIONAL).Float();
        this->Attr("cmp_ratio").AttrType(OPTIONAL).Int();
        this->Attr("ori_mask_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("cmp_mask_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("ori_win_left").AttrType(OPTIONAL).Int(-1); // ori_win_left默认值-1
        this->Attr("ori_win_right").AttrType(OPTIONAL).Int(-1); // ori_win_right默认值-1
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("BSND");
        this->Attr("topk_value_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};
OP_ADD(MixedQuantSparseFlashMla);
} // namespace ops
