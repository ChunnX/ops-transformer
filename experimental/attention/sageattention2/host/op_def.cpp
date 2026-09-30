/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {

/**
 * @brief Operator Definition for SageAttention2 on Ascend 950PR.
 *
 * Exposes the INT8 QK^T + MXFP8 PV fused attention kernel with dynamic shape
 * and dynamic rank support for CANN 9.2.0-beta.1.
 */
class SageAttention2 : public OpDef {
public:
    explicit SageAttention2(const char *name) : OpDef(name)
    {
        DeclareTensorInputs();
        DeclareTensorOutputs();
        DeclareOperatorAttributes();
        DeclareHardwareTarget();
    }

private:
    void DeclareTensorInputs()
    {
        const std::vector<ge::DataType> supportedTypes = {ge::DT_FLOAT16, ge::DT_BF16};
        const std::vector<ge::Format> supportedFormats = {ge::FORMAT_ND, ge::FORMAT_ND};

        Input("q").ParamType(REQUIRED).DataType(supportedTypes).Format(supportedFormats);
        Input("k").ParamType(REQUIRED).DataType(supportedTypes).Format(supportedFormats);
        Input("v").ParamType(REQUIRED).DataType(supportedTypes).Format(supportedFormats);
    }

    void DeclareTensorOutputs()
    {
        const std::vector<ge::DataType> outTypes = {ge::DT_FLOAT16, ge::DT_BF16};
        const std::vector<ge::DataType> lseTypes = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND};

        Output("out").ParamType(REQUIRED).DataType(outTypes).Format(formats);
        Output("lse").ParamType(REQUIRED).DataType(lseTypes).Format(formats);
    }

    void DeclareOperatorAttributes()
    {
        Attr("sm_scale").AttrType(REQUIRED).Float();
        Attr("smooth_k").AttrType(REQUIRED).Bool();
        Attr("is_causal").AttrType(REQUIRED).Bool();
        Attr("return_lse").AttrType(REQUIRED).Bool();
        // SageQuantGran and SagePvAccum encoded representations from common/types.h
        Attr("qk_quant_gran").AttrType(REQUIRED).Int();
        Attr("pv_accum_dtype").AttrType(REQUIRED).Int();
        Attr("smooth_v").AttrType(REQUIRED).Bool();
    }

    void DeclareHardwareTarget()
    {
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicShapeSupportFlag(true)
            .DynamicRankSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "sageattention2")
            .ExtendCfgInfo("opInterface.value", "sage_attention2");
        AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(SageAttention2);

} // namespace ops
