/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_NORM_POST_PREONLY_TILING_H_
#define MHC_NORM_POST_PREONLY_TILING_H_

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "err/ops_err.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MhcNormPostPreonlyTilingData)
// 形状信息，totalTokens为展平后的B*S
TILING_DATA_FIELD_DEF(uint32_t, totalTokens); // M = B*S
TILING_DATA_FIELD_DEF(uint32_t, headCount);   // N固定为4
TILING_DATA_FIELD_DEF(uint32_t, embedDim);    // D按16个元素对齐

// 核映射，沿totalTokens均衡切分
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCoreSp);
TILING_DATA_FIELD_DEF(uint32_t, blockPivot);

// D轴分块
TILING_DATA_FIELD_DEF(uint32_t, dimTile);     // 单块元素数
TILING_DATA_FIELD_DEF(uint32_t, dimLoop);     // 分块数量
TILING_DATA_FIELD_DEF(uint32_t, lastDimTile); // 尾块元素数，0表示整除

// MHC_Pre参数，pre分支只使用N行
TILING_DATA_FIELD_DEF(uint32_t, phiRows); // N固定为4
TILING_DATA_FIELD_DEF(uint32_t, phiCols); // N*D

// 数值稳定项
TILING_DATA_FIELD_DEF(float, normEps);
TILING_DATA_FIELD_DEF(float, mhcPreNormEps);
TILING_DATA_FIELD_DEF(float, hcEps);

// 预先计算的浮点倒数，避免Kernel侧整数转浮点
TILING_DATA_FIELD_DEF(float, invD); // 1.0f / D
TILING_DATA_FIELD_DEF(float, invK); // 1.0f / (N*D)

// 是否在MHC_Post与MHC_Pre之间执行RMSNorm_mid
TILING_DATA_FIELD_DEF(uint32_t, useRmsMid); // 0表示关闭，1表示开启
// A5执行方案。bit2选择双sibling AIV模板，其他组合回退单AIV。
TILING_DATA_FIELD_DEF(uint32_t, coresPerToken); // 取值为1或2

// 是否返回h_in_f32
TILING_DATA_FIELD_DEF(int8_t, returnHInF32); // TILING_DATA_FIELD_DEF接口不支持bool
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcNormPostPreonly, MhcNormPostPreonlyTilingData)

struct MhcNormPostPreonlyCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t systemWorkspaceSize{0UL};
};

class MhcNormPostPreonlyTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MhcNormPostPreonlyTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {}

    ~MhcNormPostPreonlyTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckAttrs();
    ge::graphStatus CheckInputDtype() const;
    void FillTilingData();
    void PrintTilingData() const;

    MhcNormPostPreonlyTilingData tilingData_;
    uint32_t coreNumAiv_{0U};
    uint32_t totalTokens_{0U};
    uint32_t headCount_{0U};
    uint32_t embedDim_{0U};
    uint32_t phiRows_{0U};
    uint32_t phiCols_{0U};
    uint32_t usedCoreNum_{0U};
    uint32_t rowsPerCore_{0U};
    uint32_t rowsPerCoreSp_{0U};
    uint32_t blockPivot_{0U};
    uint32_t dimTile_{0U};
    uint32_t dimLoop_{0U};
    uint32_t lastDimTile_{0U};
    uint32_t coresPerToken_{1U};
    float normEps_{1e-6F};
    float mhcPreNormEps_{1e-6F};
    float hcEps_{1e-6F};
    bool useRmsMid_{false};
    bool returnHInF32_{false};
    bool enableDualSibling_{false};
    size_t systemWorkspaceSize_{0U};
    size_t workspaceSize_{0U};
};

} // namespace optiling

#endif
