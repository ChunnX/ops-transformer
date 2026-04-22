/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_mat_mul_allto_allv_tiling.h
 * \brief
 */
#ifndef MC2_GROUPED_MATMUL_ALLTO_ALLV_TILING_STRUCT_H
#define MC2_GROUPED_MATMUL_ALLTO_ALLV_TILING_STRUCT_H

#include "grouped_mat_mul_allto_allv_tiling_base.h"
#include "../../op_kernel/grouped_mat_mul_allto_allv_tiling.h"
#include "../../op_kernel/grouped_mat_mul_allto_allv_tiling_key.h"

namespace optiling {
class GmmAlltoAllvTilingStruct : public GmmAlltoAllvTilingBase
{
public:
    explicit GmmAlltoAllvTilingStruct(gert::TilingContext* context) : GmmAlltoAllvTilingBase(context){};
    virtual bool NeedToCheckRecvSendCounts() = 0;
    virtual std::vector<int64_t> GetEpWorldSizeOptional() = 0;
    bool CheckSendCnt(
        const gert::RuntimeAttrs* attrs,
        int64_t A,  int64_t H, int64_t eExpert, int64_t epWorldSize,
        gert::TilingContext* context);
    bool CheckRecvCnt(
        const gert::RuntimeAttrs* attrs,
        int64_t BsK, int64_t H, int64_t eExpert, int64_t epWorldSize,
        gert::TilingContext* context);
    bool CheckSendCntAndRecvCnt(
        const gert::RuntimeAttrs* attrs, int64_t BsK, int64_t A, int64_t H, int64_t eExpert, int64_t epWorldSize,
        gert::TilingContext* context);
    bool CheckCoreDimensionsAndCommunication(
        GroupedMatMulAlltoAllvTilingData* tilingData,
        const gert::StorageShape* gmmX, const gert::StorageShape* gmmWeight,
        const gert::StorageShape* y,
        const gert::RuntimeAttrs* attrs, gert::TilingContext* context);
    bool CheckDimValue(
        GroupedMatMulAlltoAllvTilingData* tilingData, const gert::StorageShape* gmmX,
        const gert::StorageShape* gmmWeight, const gert::StorageShape* mmX,
        const gert::StorageShape* mmWeight, const gert::StorageShape* y,
        const gert::StorageShape* mmY, const gert::RuntimeAttrs* attrs,
        gert::TilingContext* context);
    bool CheckInputAndOutput(gert::TilingContext* context, GroupedMatMulAlltoAllvTilingData* tilingData);
    ge::graphStatus GroupedMatMulAlltoAllvDoOpTilingFunc(gert::TilingContext* context);
    bool CheckEpWorldSizeConstraints(
        GroupedMatMulAlltoAllvTilingData* tilingData, gert::TilingContext* context);

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
};
} // namespace optiling
#endif