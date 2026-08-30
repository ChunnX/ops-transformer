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
 * \file moe_tutel_combine_x_tiling.h
 * \brief MoeTutelCombineX Ascend950 tiling declaration.
 */

#ifndef OPS_OP_TILING_RUNTIME_MOE_TUTEL_COMBINE_X_TILING_H_
#define OPS_OP_TILING_RUNTIME_MOE_TUTEL_COMBINE_X_TILING_H_

#include "../../op_kernel/arch35/moe_tutel_combine_x_tiling_data.h"
#include "op_host/tiling_base.h"

namespace optiling {

struct MoeTutelCombineXCompileInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};

class MoeTutelCombineXTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeTutelCombineXTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckRouteShape(const gert::Shape &gatesShape);
    ge::graphStatus CheckAddressRange();
    ge::graphStatus SolveHFactor();
    void SolveRowFactor();
    ge::graphStatus SolveKFactor();
    // 给定 (H 分块, token 分块, k 分块) 时 kernel 侧的 UB 占用字节数。
    // 与 op_kernel/moe_tutel_combine_x.cpp 的 InitBuffer 序列逐项对应，改一处必须改另一处。
    int64_t CalcUbBytes(int64_t hChunk, int64_t rowF, int64_t kF) const;

private:
    int64_t totalCoreNum_{0};
    int64_t usableUb_{0};
    int64_t dtypeSize_{0};

    int64_t expertNum_{0};
    int64_t capacity_{0};
    int64_t hidden_{0};
    int64_t topK_{0};
    int64_t samples_{0};

    int64_t usedCoreNum_{0};
    int64_t rowOfFormerBlock_{0};
    int64_t rowOfTailBlock_{0};
    int64_t rowFactor_{1};
    int64_t kFactor_{1};
    int64_t hFactor_{0};

    uint64_t workspaceSize_{0};
    ge::DataType dtype_{ge::DT_UNDEFINED};
    MoeTutelCombineXTilingData tilingData_;
};
} // namespace optiling
#endif // OPS_OP_TILING_RUNTIME_MOE_TUTEL_COMBINE_X_TILING_H_
