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
 * \file engram_fetch_grad_def.cpp
 * \brief 算子信息库定义 — EngramFetchGrad（反向，新算子）
 *
 * 单融合 kernel，3 步:
 * ① grad_sorted = gradFetched[perm]
 * ② a2a(grad_sorted; sendCounts↔recvCounts) → recvGrad
 * ③ owner unique + fp32 scatter-add (recvLocalEntry, recvGrad) → gradUnique + uniqueLocalEntry
 *
 * 输入: gradFetched + 前向 save-for-backward (perm/sendCounts/recvCounts/recvLocalEntry)
 * 输出: gradUniqueOut (fp32) + uniqueLocalEntryOut (int32 1D sparse index) + numUniqueOut (int32)
 */

#include "register/op_def_registry.h"

namespace ops {
class EngramFetchGrad : public OpDef {
public:
    explicit EngramFetchGrad(const char *name) : OpDef(name)
    {
        // Input 0: 通信上下文
        this->Input("commContext")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        // Input 1: 前向 fetched 的梯度 gradFetched (T, H) bf16
        this->Input("gradFetched")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        // Input 2: perm (T,) int32 — 前向 save-for-backward
        this->Input("perm")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        // Input 3: sendCounts (W,) int32 — 前向 save-for-backward
        this->Input("sendCounts")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        // Input 4: recvCounts (W,) int32 — 前向 save-for-backward
        this->Input("recvCounts")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        // Input 5: recvLocalEntry (R,) int32 — 前向 save-for-backward
        this->Input("recvLocalEntry")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        // Output 0: gradUnique (R_max, H) fp32 — 前 numUnique 有效
        this->Output("gradUniqueOut")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        // Output 1: uniqueLocalEntry (R_max,) int32
        this->Output("uniqueLocalEntryOut")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        // Output 2: numUnique (1,) int32
        this->Output("numUniqueOut")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});

        this->Attr("hidden_size").AttrType(REQUIRED).Int();
        this->Attr("num_entries_per_rank").AttrType(REQUIRED).Int();

        OpAICoreConfig aicore_config_950;
        aicore_config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend950", aicore_config_950);
    }
};
OP_ADD(EngramFetchGrad);
} // namespace ops
