/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unquant_matmul_all_reduce_tiling_310.h
 * \brief
 */
#ifndef UNQUANT_MATMUL_ALL_REDUCE_TILING_310_H
#define UNQUANT_MATMUL_ALL_REDUCE_TILING_310_H

#include "../matmul_all_reduce_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UnQuantMatmulAllReduceTilingData)
TILING_DATA_FIELD_DEF_STRUCT(Mc2Msg, msg);
TILING_DATA_FIELD_DEF_STRUCT(RCSTiling, param);
TILING_DATA_FIELD_DEF_STRUCT(MatmulTilingData, tilematmulTiling);
TILING_DATA_FIELD_DEF_STRUCT(MatmulTilingData, tailmatmulTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000002000, UnQuantMatmulAllReduceTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduce_10000000000000002001, UnQuantMatmulAllReduceTilingData);

class UnQuantMatmulAllReduceTiling310 : public MatmulAllReduceTilingBase
{
    class UnQuantTilingTransferHelper : public matmul_v3::MatmulV3BaseTiling
    {
    public:
        UnQuantTilingTransferHelper(
            UnQuantMatmulAllReduceTiling310& unquantMatmulAllReduceTiling, MatmulTilingData& data)
            : MatmulV3BaseTiling(unquantMatmulAllReduceTiling.context_, &data),
              tilingProcesser_(unquantMatmulAllReduceTiling)
        {}

        ge::graphStatus CheckInputInfo()
        {
            return ge::GRAPH_SUCCESS;
        }

        ge::graphStatus GetShapeAttrsInfo() override
        {
            OP_LOGI(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
            if (CheckInputInfo() == ge::GRAPH_FAILED) {
                return ge::GRAPH_FAILED;
            }

            auto&& tilingArgs = tilingProcesser_.args_;
            args_.opName = tilingProcesser_.opName_;
            args_.isATrans = tilingArgs.isATrans;
            args_.isBTrans = tilingArgs.isBTrans;
            args_.isHf32 = false;
            args_.hasBias = tilingArgs.isBias;
            args_.nd2nzA = false;
            args_.nd2nzB = false;
            args_.aType = tilingArgs.geAType;
            args_.bType = tilingArgs.geBType;
            args_.cType = tilingArgs.geCType;
            args_.biasType = tilingArgs.geBiasType;
            args_.aFormat = ge::FORMAT_ND;
            args_.bFormat = ge::FORMAT_FRACTAL_NZ;
            args_.outFormat = ge::FORMAT_ND;
            args_.mValue = tilingArgs.mValue;
            args_.kValue = tilingArgs.kValue;
            args_.nValue = tilingArgs.nValue;
            args_.l2Ratio = 0;
            return ge::GRAPH_SUCCESS;
        }

        ge::graphStatus PostTiling() override
        {
            tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
            OP_LOGI(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
            return ge::GRAPH_SUCCESS;
        }

    private:
        UnQuantMatmulAllReduceTiling310& tilingProcesser_;
    };

public:
    explicit UnQuantMatmulAllReduceTiling310(gert::TilingContext* context) : MatmulAllReduceTilingBase(context)
    {
        unquantMatmulAllReduceTilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    }
    ~UnQuantMatmulAllReduceTiling310() override = default;

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

    ge::graphStatus DoUnQuantTiling();

private:
    UnQuantMatmulAllReduceTilingData unquantMatmulAllReduceTilingData_;
    uint64_t myWorkSpaceSize_{0U};
};
} // namespace optiling
#endif // UNQUANT_MATMUL_ALL_REDUCE_TILING_310_H
