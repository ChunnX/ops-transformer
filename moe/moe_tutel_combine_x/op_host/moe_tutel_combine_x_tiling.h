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
 * \brief
 */

#ifndef OPS_OP_TILING_RUNTIME_MOE_TUTEL_COMBINE_X_TILING_H_
#define OPS_OP_TILING_RUNTIME_MOE_TUTEL_COMBINE_X_TILING_H_

#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"

namespace optiling {

struct MoeTutelCombineXCompileInfo {
    int64_t coreNum{0};
    int64_t ubSize{0};
};

BEGIN_TILING_DATA_DEF(MoeTutelCombineXTilingData)
TILING_DATA_FIELD_DEF(int64_t, capacity);         // C = y_grad.dim(1)
TILING_DATA_FIELD_DEF(int64_t, hidden);           // H = y_grad.dim(2)
TILING_DATA_FIELD_DEF(int64_t, topK);             // K = gates.dim(0)
TILING_DATA_FIELD_DEF(int64_t, samples);          // S = gates.dim(1)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);      // 实际派活的核数
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock); // 非尾核各领多少个 token
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);   // 尾核领多少个 token
TILING_DATA_FIELD_DEF(int64_t, rowFactor);        // 内层一次处理多少个 token
TILING_DATA_FIELD_DEF(int64_t, kFactor);          // 内层一次处理多少个 k
TILING_DATA_FIELD_DEF(int64_t, hFactor);          // H 分块长度，等于 hidden 时为整行全载
TILING_DATA_FIELD_DEF(int64_t, hChunkAlignT);     // CeilAlign(hFactor, 32 / sizeof(T))
TILING_DATA_FIELD_DEF(int64_t, hChunkAlignF32);   // CeilAlign(hFactor, 8)
TILING_DATA_FIELD_DEF(int64_t, routeStride);      // 路由片在 UB 里的行跨距 = CeilAlign(rowFactor, 16)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeTutelCombineX, MoeTutelCombineXTilingData)

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
    int64_t blockElemT_{0};

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

    int64_t workspaceSize_{0};
    ge::DataType dtype_{ge::DT_UNDEFINED};
    MoeTutelCombineXTilingData tilingData_;
};
} // namespace optiling
#endif // OPS_OP_TILING_RUNTIME_MOE_TUTEL_COMBINE_X_TILING_H_
