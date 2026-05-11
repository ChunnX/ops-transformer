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
 * \file matmul_allto_all_v2_tiling_910b.h
 * \brief
 */

#ifndef MATMUL_ALLTO_ALL_V2_TILING_910B_H
#define MATMUL_ALLTO_ALL_V2_TILING_910B_H

#include "securec.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "../matmul_allto_all_v2_tiling_base.h"
#include "../common/matmul_allto_all_v2_util_tiling.h"
#include "../../../op_kernel/arch22/matmul_allto_all_v2_tiling.h"
#include "../../../op_kernel/arch22/matmul_allto_all_v2_tiling_key.h"

namespace MC2Tiling {
using namespace optiling;

struct MatmulAlltoAllV2TilingValue {
    int32_t value = -1;
    std::map<int, std::vector<std::vector<int>>> conditionMap = {};
    explicit MatmulAlltoAllV2TilingValue(int32_t v = -1, std::map<int, std::vector<std::vector<int>>> m = {})
        : value(v), conditionMap(std::move(m))
    {
    }
};

class MatmulAlltoAllV2Tiling910B : public MatmulAlltoAllV2TilingBase {
public:
    explicit MatmulAlltoAllV2Tiling910B(gert::TilingContext *context);
    ~MatmulAlltoAllV2Tiling910B() override = default;

protected:
    bool IsCapable() override; // 恒为true
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    ge::graphStatus CheckOpInputInfo(MatmulAlltoAllV2Info &info);
    ge::graphStatus CheckAndSetAttrsInfo(MatmulAlltoAllV2Info &info);
    ge::graphStatus CheckTensorDataType(MatmulAlltoAllV2Info &info);
    ge::graphStatus CheckShapeInfo(MatmulAlltoAllV2Info &info);
    ge::graphStatus DoMmCommTiling(CoCTiling &cocTilingData, MatmulAlltoAllV2Info &info);
    void DoTwoRankTiling(CoCTiling &cocTilingData, MatmulAlltoAllV2Info &info);
    void DoFourRankTiling(CoCTiling &cocTilingData, MatmulAlltoAllV2Info &info);
    void DoEightRankTiling(CoCTiling &cocTilingData, MatmulAlltoAllV2Info &info);
    void CalTilingParam(CoCTiling &cocTilingData, const std::map<int *, MatmulAlltoAllV2TilingValue> &TilingParamMap,
                        MatmulAlltoAllV2Info &info);
    int32_t GetValueFromMKNConditionMap(int32_t m, int32_t k, int32_t n, int32_t defaultValue,
                                        std::map<int, std::vector<std::vector<int>>> conditionMap);
    ge::graphStatus SetHcclTiling(MatmulAlltoAllV2TilingData *tilingData);
    void PrintMatmulAlltoAllV2TilingData(CoCTiling &cocTilingData, MatmulAlltoAllV2Info &info);
    void SetTilingKey();

private:
    uint64_t tilingKey_;
    bool isQuantBF16 = false;
    bool needTransX2 = false;
    bool hasBias = false;
    uint32_t quantType = 0;
    uint32_t tileM0 = 0;
    uint32_t tileN0 = 0;
    uint32_t numBlocks = 1U;
};
} // namespace MC2Tiling
#endif
