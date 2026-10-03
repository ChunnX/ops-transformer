/*!
 * \file sage_attention2_def.cpp
 * \brief Ascend CANN 官方标准 OpDef 算子定义与硬件配置
 */

#include "sage_attention2_def.h"

namespace ops {

SageAttention2::SageAttention2(const char *name) : OpDef(name)
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
        // DEFECT-L-1 同口径：km 内核 float* 直读，位型仅 DT_FLOAT，不得放行 DT_FLOAT16
        this->Input("key_mean")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
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

        // 4. 算子属性声明（OPTIONAL + 冻结默认值，对齐开源 core.py 签名）
        this->Attr("tensor_layout").AttrType(OPTIONAL).String("HND");
        this->Attr("is_causal").AttrType(OPTIONAL).Bool(false);
        this->Attr("qk_quant_gran").AttrType(OPTIONAL).String("per_thread");
        this->Attr("sm_scale").AttrType(OPTIONAL).Float(0.0f);
        this->Attr("pv_accum_dtype").AttrType(OPTIONAL).String("fp32+fp16");
        this->Attr("smooth_k").AttrType(OPTIONAL).Bool(true);
        this->Attr("smooth_v").AttrType(OPTIONAL).Bool(false);

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


OP_ADD(SageAttention2);

} // namespace ops
