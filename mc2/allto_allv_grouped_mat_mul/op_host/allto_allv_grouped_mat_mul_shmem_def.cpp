/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def_registry.h"

namespace ops {
class AlltoAllvGroupedMatMulShmem : public OpDef {
public:
    explicit AlltoAllvGroupedMatMulShmem(const char *name) : OpDef(name)
    {
        this->Input("gmm_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gmm_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("gmm_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("ep_world_size").AttrType(REQUIRED).Int();
        this->Attr("send_counts").AttrType(REQUIRED).ListInt();
        this->Attr("recv_counts").AttrType(REQUIRED).ListInt();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        // Deliberately no MC2().HcclGroup(): communication uses the initialized process-local SHMEM state.
    }
};

OP_ADD(AlltoAllvGroupedMatMulShmem);
} // namespace ops
