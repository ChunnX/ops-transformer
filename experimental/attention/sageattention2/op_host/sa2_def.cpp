// 模块说明：SageAttention2（INT8 QK^T + FP8 PV）OpDef 注册：q/k/v(FP16/BF16, ND) → out/lse，attr 7 项全部 REQUIRED。
#include <register/op_def_registry.h>

namespace ops {

class SageAttention2 : public OpDef {
public:
    explicit SageAttention2(const char *name) : OpDef(name)
    {
        // 输入仅 FP16/BF16（支持 FP8 PV 路径），ND 布局，AutoContiguous 由框架补齐
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // out 与 q 同 dtype（pad 维由内核裁剪回 head_dim_og）；lse 固定 FP32
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("lse")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();

        // attr 声明序即 aclnn/python 传参序，禁止增删改序；括号内为原型值，REQUIRED 由调用方必传
        this->Attr("sm_scale").AttrType(REQUIRED).Float(0.0F);
        this->Attr("smooth_k").AttrType(REQUIRED).Bool(true);
        this->Attr("is_causal").AttrType(REQUIRED).Bool(false);
        this->Attr("return_lse").AttrType(REQUIRED).Bool(false);
        this->Attr("qk_quant_gran").AttrType(REQUIRED).Int(1);  // 1 = kPerThread（开源默认粒度）
        this->Attr("pv_accum_dtype").AttrType(REQUIRED).Int(2); // 2 = kFp32Fp16（开源默认两级累加）
        this->Attr("smooth_v").AttrType(REQUIRED).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicShapeSupportFlag(true)
            .DynamicRankSupportFlag(false)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "sa2_kernel")
            .ExtendCfgInfo("opInterface.value", "sa2_attention2");
        // 950PR 原生 FP8 + MIX 1 AIC:2 AIV，仅注册该目标硬件
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(SageAttention2);
} // namespace ops
