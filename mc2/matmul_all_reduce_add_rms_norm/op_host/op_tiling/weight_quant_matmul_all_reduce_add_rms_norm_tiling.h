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
 * \file weight_quant_matmul_all_reduce_add_rms_norm_tiling.h
 * \brief
 */
#ifndef _WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#define _WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_
#include <memory>
#include "../../../matmul_all_reduce/op_host/op_tiling/arch32/weight_quant_matmul_all_reduce_tiling.h"
#include "common_add_rms_norm_tiling.h"
#include "context_transfer.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(WeightQuantMatmulAllReduceAddRmsNormTilingData)
TILING_DATA_FIELD_DEF_STRUCT(WeightQuantMatmulAllReduceTilingData, weightQuantMatmulAllReduceTilingData);
TILING_DATA_FIELD_DEF_STRUCT(MC2AddRMSNormTilingData, addRMSNormTileTilingData);
TILING_DATA_FIELD_DEF_STRUCT(MC2AddRMSNormTilingData, addRMSNormTailTilingData);
TILING_DATA_FIELD_DEF_STRUCT(AddRMSNormTilingeKeyData, addRmsNormTilingeKeyData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_2293772, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_3342348, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_69402636, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_70451212, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_4390924, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_5439500, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_71499788, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_72548364, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_6488076, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_73596940, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_7536652, WeightQuantMatmulAllReduceAddRmsNormTilingData);
REGISTER_TILING_DATA_CLASS(MatmulAllReduceAddRmsNorm_74645516, WeightQuantMatmulAllReduceAddRmsNormTilingData);

class WeightQuantMMNTilingTransferHelper;
class WeightQuantMatmulAllReduceAddRmsNormTiling : public TilingBaseClass
{
    friend class WeightQuantMMNTilingTransferHelper;

public:
    explicit WeightQuantMatmulAllReduceAddRmsNormTiling(gert::TilingContext* context);
    ~WeightQuantMatmulAllReduceAddRmsNormTiling() override = default;

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus CheckMRNInput(const MRNCtxInfo& mrnCtxInfo);

private:
    bool HasTail() const;
    MRNCtxInfo mrnCtxInfo_;
    WeightQuantMatmulAllReduceAddRmsNormTilingData tilingData_;
    bool hasTail_;
    TilingOut tilingOutAddRmsNormTile_;
    TilingOut tilingOutAddRmsNormTail_;
    std::unique_ptr<WeightQuantMMNTilingTransferHelper> helper_;
};

class WeightQuantMMNTilingTransferHelper : public WeightQuantMatmulAllReduceTiling
{
public:
    WeightQuantMMNTilingTransferHelper(
        WeightQuantMatmulAllReduceAddRmsNormTiling& weightQuantMatmulAllReduceAddRmsNormTiling,
        WeightQuantMatmulAllReduceTilingData& data);
    ge::graphStatus GetShapeAttrsInfo() override;

private:
    WeightQuantMatmulAllReduceAddRmsNormTiling& tilingProcesser_;
};
} // namespace optiling

#endif // _WEIGHT_QUANT_MATMUL_ALL_REDUCE_ADD_RMS_NORM_TILING_H_