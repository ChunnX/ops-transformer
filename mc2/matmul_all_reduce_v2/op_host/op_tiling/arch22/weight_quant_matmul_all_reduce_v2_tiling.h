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
 * \file weight_quant_matmul_all_reduce_v2_tiling.h
 * \brief
 */
#ifndef WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
#define WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
#include "../matmul_all_reduce_v2_tiling_base.h"
#include "../../../op_kernel/matmul_all_reduce_v2_tiling_key.h"
#include "../../../op_kernel/arch22/weight_quant_matmul_all_reduce_v2_tiling_data.h"
#include "weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom.h"

namespace optiling {

struct WeightQuantMatmulTPLParam {
    uint64_t subAlgorithmCustom{65535};
    bool hasAntiquantOffset{0};
    uint64_t antiquantType{65535};
    bool transA{0};
    bool transB{0};
};

constexpr int64_t ANTIQUANT_GROUP_SIZE_MIN_VALUE = 32;

class WeightQuantMatmulAllReduceV2Tiling : public MatmulAllReduceV2TilingBase {
    friend class WeightQuantTilingTransferHelper;
    friend class WeightQuantMatmulAllReduceV2AddRmsNormTiling;

public:
    explicit WeightQuantMatmulAllReduceV2Tiling(gert::TilingContext *context);
    WeightQuantMatmulAllReduceV2Tiling(gert::TilingContext *context, MMRCtxInfo *mmrCtxInfo,
                                     Mc2Tiling::WeightQuantMatmulAllReduceV2TilingData *out);
    ~WeightQuantMatmulAllReduceV2Tiling() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Tiling::Mc2Msg &MutableMc2MsgData() override
    {
        return weightQuantMatmulAllReduceV2TilingData_.msg;
    }

    Mc2Tiling::RCSTiling &MutableRCSTilingData() override
    {
        return weightQuantMatmulAllReduceV2TilingData_.param;
    }

    AscendC::tiling::TCubeTiling &MutableTCubeTileTilingData() override
    {
        return weightQuantMatmulAllReduceV2TilingData_.tilematmulTiling.matmulTiling;
    }

    AscendC::tiling::TCubeTiling &MutableTCubeTailTilingData() override
    {
        return weightQuantMatmulAllReduceV2TilingData_.tailmatmulTiling.matmulTiling;
    }

    ge::graphStatus DoWeightQuantTiling();

    void DoEmptyTensorTiling() override;

    ge::graphStatus CheckInput() override;

    CutResult GetTilingResult() override;

private:
    ge::graphStatus CheckAxisSize();
    Mc2Tiling::WeightQuantMatmulAllReduceV2TilingData weightQuantMatmulAllReduceV2TilingDataSelf_{};
    Mc2Tiling::WeightQuantMatmulAllReduceV2TilingData &weightQuantMatmulAllReduceV2TilingData_;
    uint64_t myWorkSpaceSize_{0U};
    WeightQuantMatmulTPLParam weightQuantMatmulTPLParam_;
};

class WeightQuantTilingTransferHelper : public Mc2WeightQuantBatchMatmulV2TilingCustom {
public:
    WeightQuantTilingTransferHelper(WeightQuantMatmulAllReduceV2Tiling &weightQuantMatmulAllReduceV2Tiling,
                                    Mc2WeightQuantBatchMatmulV2TilingData &data)
        : Mc2WeightQuantBatchMatmulV2TilingCustom(weightQuantMatmulAllReduceV2Tiling.context_, &data),
          tilingProcesser_(weightQuantMatmulAllReduceV2Tiling)
    {
    }
    ge::graphStatus GetShapeAttrsInfo() override;
    void PrintTilingInputParam(Mc2WeightQuantBatchMatmulInfo &weightQuantBatchMatmulInfo);
    ge::graphStatus PostTiling() override;
    WeightQuantMatmulTPLParam GetWeightQuantMatmulTPLParam();

private:
    WeightQuantMatmulAllReduceV2Tiling &tilingProcesser_;
};
} // namespace optiling
#endif // WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_H
