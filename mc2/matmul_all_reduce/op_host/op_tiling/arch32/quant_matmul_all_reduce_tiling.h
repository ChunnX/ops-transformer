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
 * \file quant_matmul_all_reduce_tiling.h
 * \brief
 */
#ifndef QUANT_MATMUL_ALL_REDUCE_TILING_H
#define QUANT_MATMUL_ALL_REDUCE_TILING_H

#include "../matmul_all_reduce_tiling_base.h"
#include "../../../op_kernel/matmul_all_reduce_tiling_key.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(QuantMatmulAllReduceTilingData)
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(Mc2QuantBatchMatmulV3TilingData, tilematmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(Mc2QuantBatchMatmulV3TilingData, tailmatmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_4104, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_20488, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_8, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_16392, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_4136, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_20520, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_40, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_16424, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(QuantMatmulAllReduceTilingDataOp, QuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_9, QuantMatmulAllReduceTilingData);

struct QuantMatmulTPLParam{
    uint64_t trans{65535};
    uint64_t isPertoken{65535};
};

class QuantMatmulAllReduceTiling : public MatmulAllReduceTilingBase
{
    friend class QuantTilingTransferHelper;
    friend class QuantMatmulAllReduceAddRmsNormTiling;

public:
    explicit QuantMatmulAllReduceTiling(gert::TilingContext* context);
    QuantMatmulAllReduceTiling(
        gert::TilingContext* context, MMRCtxInfo* mmrCtxInfo, QuantMatmulAllReduceTilingData* out);
    ~QuantMatmulAllReduceTiling() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Msg& MutableMc2MsgData() override;

    RCSTiling& MutableRCSTilingData() override;

    TCubeTiling& MutableTCubeTileTilingData() override;

    TCubeTiling& MutableTCubeTailTilingData() override;

    ge::graphStatus DoQuantTiling();

    ge::graphStatus CheckInput() override;

    ge::graphStatus CheckDequantScaleType();

private:
    ge::graphStatus CheckAxisSize();
    QuantMatmulAllReduceTilingData quantMatmulAllReduceTilingDataSelf_;
    QuantMatmulAllReduceTilingData& quantMatmulAllReduceTilingData_;
    uint64_t myWorkSpaceSize_{0U};
    bool isCommInt8Enable_ = false;
    QuantMatmulTPLParam quantMatmulTPLParam_;
};

class QuantTilingTransferHelper : public Mc2QuantBatchMatmulV3Tiling
{
public:
    QuantTilingTransferHelper(
        QuantMatmulAllReduceTiling& quantMatmulAllReduceTiling, Mc2QuantBatchMatmulV3TilingData& data);
    const gert::Shape GetX1Shape(const size_t index) override;
    const gert::Shape GetX2Shape(const size_t index) override;
    const gert::Shape& GetScaleShape(const size_t index) override;
    const gert::StorageShape* GetPertokenShape(const size_t index) override;
    const gert::StorageShape* GetBiasShape(const size_t index) override;
    ge::graphStatus GetShapeAttrsInfo() override;
    void PrintTilingInputParam(Mc2QuantBatchMatmulInfo quantBatchMatmulInfo);
    ge::graphStatus PostTiling() override;
    QuantMatmulTPLParam GetQuantMatmulTPLParam();

private:
    QuantMatmulAllReduceTiling& tilingProcesser_;
};
} // namespace optiling
#endif // QUANT_MATMUL_ALL_REDUCE_TILING_H