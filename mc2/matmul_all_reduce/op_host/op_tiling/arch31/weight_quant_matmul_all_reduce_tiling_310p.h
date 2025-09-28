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
 * \file weight_quant_matmul_all_reduce_tiling_310p.h
 * \brief
 */
#ifndef WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_310P_H
#define WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_310P_H
#include "../matmul_all_reduce_tiling.h"

namespace optiling {
class WeightQuantMatmulAllReduceTiling310P : public MatmulAllReduceTilingBase
{
    class WeightQuantTilingTransferHelper : public WeightQuantBatchMatmulV2WeightNz
    {
    public:
        WeightQuantTilingTransferHelper(
            WeightQuantMatmulAllReduceTiling310P& weightQuantMatmulAllReduceTiling,
            WeightQuantBatchMatmulV2NzTilingData& data)
            : WeightQuantBatchMatmulV2WeightNz(weightQuantMatmulAllReduceTiling.context_, &data),
              tilingProcesser_(weightQuantMatmulAllReduceTiling)
        {}
        ge::graphStatus GetShapeAttrsInfo() override
        {
            OP_LOGI(tilingProcesser_.opName_, "Start assemble input params for matmul tiling");
            auto&& tilingArgs = tilingProcesser_.args_;
            inputParams_.opName = tilingProcesser_.opName_;
            inputParams_.transA = tilingArgs.isATrans;
            inputParams_.transB = tilingArgs.isBTrans;
            inputParams_.hasBias = tilingArgs.isBias;
            inputParams_.hasAntiQuantOffset = tilingProcesser_.HasAntiQuantOffset();
            inputParams_.mSize = tilingArgs.mValue;
            inputParams_.kSize = tilingArgs.kValue;
            inputParams_.nSize = tilingArgs.nValue;
            inputParams_.aDtype = tilingArgs.geAType;
            inputParams_.bDtype = tilingArgs.geBType;
            inputParams_.cDtype = tilingArgs.geCType;
            inputParams_.biasDtype = tilingArgs.geBiasType;
            inputParams_.antiQuantType = tilingProcesser_.antiQuantType_;
            inputParams_.groupSize = tilingProcesser_.antiGroupSize_;
            inputParams_.quantType = tilingProcesser_.quantType_;
            PrintTilingInputParam(inputParams_);
            return ge::GRAPH_SUCCESS;
        }
        void PrintTilingInputParam(WeightQuantBatchMatmulInfo& weightQuantBatchMatmulInfo)
        {
            OP_LOGD(
                tilingProcesser_.opName_, " transA_ %d transB_ %d, hasBias_ %d, hasAntiQuantOffset_ %d, ",
                weightQuantBatchMatmulInfo.transA, weightQuantBatchMatmulInfo.transB,
                weightQuantBatchMatmulInfo.hasBias, weightQuantBatchMatmulInfo.hasAntiQuantOffset);
            OP_LOGD(
                tilingProcesser_.opName_, "mSize_ %ld kSize_ %ldnSize_ %ld groupSize_ %ld",
                weightQuantBatchMatmulInfo.mSize, weightQuantBatchMatmulInfo.kSize, weightQuantBatchMatmulInfo.nSize,
                weightQuantBatchMatmulInfo.groupSize);
            OP_LOGD(
                tilingProcesser_.opName_, "aDtype_ %d bDtype_ %d cDtype_ %d biasDtype_ %d",
                static_cast<int32_t>(weightQuantBatchMatmulInfo.aDtype),
                static_cast<int32_t>(weightQuantBatchMatmulInfo.bDtype),
                static_cast<int32_t>(weightQuantBatchMatmulInfo.cDtype),
                static_cast<int32_t>(weightQuantBatchMatmulInfo.biasDtype));
            OP_LOGD(
                tilingProcesser_.opName_, "antiQuantType_ %d quantType_ %d",
                static_cast<int32_t>(weightQuantBatchMatmulInfo.antiQuantType),
                static_cast<int32_t>(weightQuantBatchMatmulInfo.quantType));
        }

        void PrintTilingData(bool debugLevel)
        {
            if (debugLevel && CheckLogLevel(OP, DLOG_DEBUG) != 1) {
                return;
            }
            std::stringstream ss;
            ss << "cubeBlockDimN: " << static_cast<uint32_t>(tilingData_->get_cubeBlockDimN())
               << " cubeBlockDimM: " << static_cast<uint32_t>(tilingData_->get_cubeBlockDimM())
               << " AL1Pingpong: " << static_cast<uint32_t>(tilingData_->get_AL1Pingpong())
               << " BL1Pingpong: " << static_cast<uint32_t>(tilingData_->get_BL1Pingpong())
               << " kAlign: " << tilingData_->get_kAlign() << " nAlign: " << tilingData_->get_nAlign()
               << " mSize: " << tilingData_->get_mSize() << " kSize: " << tilingData_->get_kSize()
               << " nSize: " << tilingData_->get_nSize() << " groupSize: " << tilingData_->get_groupSize()
               << " mAubSize: " << tilingData_->get_mAubSize() << " kAubSize: " << tilingData_->get_kAubSize()
               << " nBubSize: " << tilingData_->get_nBubSize() << " kBubSize: " << tilingData_->get_kBubSize()
               << " mCubSize: " << tilingData_->get_mCubSize() << " nCubSize: " << tilingData_->get_nCubSize()
               << " mAL1Size: " << tilingData_->get_mAL1Size() << " kAL1Size: " << tilingData_->get_kAL1Size()
               << " nBL1Size: " << tilingData_->get_nBL1Size() << " kBL1Size: " << tilingData_->get_kBL1Size();

            int32_t logLevel = debugLevel ? DLOG_DEBUG : DLOG_ERROR;
            OPS_LOG_FULL(logLevel, inputParams_.opName, "tiling data: %s", ss.str().c_str());
            PrintMatMulTiling(logLevel);
        }

        void PrintMatMulTiling(int32_t logLevel)
        {
            std::stringstream ss;
            auto& matmulTiling = tilingData_->matmulTiling;
            ss << "usedCoreNum " << matmulTiling.get_usedCoreNum() << " M " << matmulTiling.get_M() << " N "
               << matmulTiling.get_N() << " Ka " << matmulTiling.get_Ka() << " Kb " << matmulTiling.get_Kb()
               << " singleCoreM " << matmulTiling.get_singleCoreM() << " singleCoreN " << matmulTiling.get_singleCoreN()
               << " singleCoreK " << matmulTiling.get_singleCoreK() << " baseM " << matmulTiling.get_baseM()
               << " baseN " << matmulTiling.get_baseN() << " baseK " << matmulTiling.get_baseK() << " depthA1 "
               << matmulTiling.get_depthA1() << " depthB1 " << matmulTiling.get_depthB1() << " stepM "
               << matmulTiling.get_stepM() << " stepN " << matmulTiling.get_stepN() << " isBias "
               << matmulTiling.get_isBias() << " transLength " << matmulTiling.get_transLength() << " iterateOrder "
               << matmulTiling.get_iterateOrder() << " shareMode " << matmulTiling.get_shareMode() << " shareL1Size "
               << matmulTiling.get_shareL1Size() << " shareL0CSize " << matmulTiling.get_shareL0CSize()
               << " shareUbSize " << matmulTiling.get_shareUbSize() << " batchM " << matmulTiling.get_batchM()
               << " batchN " << matmulTiling.get_batchN() << " stepKa " << matmulTiling.get_stepKa() << " stepKb "
               << matmulTiling.get_stepKb() << " dbL0A " << matmulTiling.get_dbL0A() << " dbL0B "
               << matmulTiling.get_dbL0B() << " dbL0C " << matmulTiling.get_dbL0C();

            OPS_LOG_FULL(logLevel, inputParams_.opName, "matmul tiling: %s", ss.str().c_str());
        }

        ge::graphStatus PostTiling() override
        {
            PrintTilingData(true);
            tilingProcesser_.myWorkSpaceSize_ = std::max(tilingProcesser_.myWorkSpaceSize_, workspaceSize_);
            OP_LOGI(tilingProcesser_.opName_, " set mm workspace size %lu to mc2", tilingProcesser_.myWorkSpaceSize_);
            return ge::GRAPH_SUCCESS;
        }

    private:
        WeightQuantMatmulAllReduceTiling310P& tilingProcesser_;
    };

public:
    explicit WeightQuantMatmulAllReduceTiling310P(gert::TilingContext* context) : MatmulAllReduceTilingBase(context)
    {
        weightQuantMatmulAllReduceTilingData_.SetDataPtr(context_->GetRawTilingData()->GetData());
    }

    ~WeightQuantMatmulAllReduceTiling310P() override = default;

protected:
    bool IsCapable() override;

    void UpdateCommOffset();

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Msg& MutableMc2MsgData() override;

    RCSTiling& MutableRCSTilingData() override;

    TCubeTiling& MutableTCubeTileTilingData() override;

    TCubeTiling& MutableTCubeTailTilingData() override;

    ge::graphStatus DoWeightQuantTiling();

private:
    WeightQuantMatmulAllReduceNzTilingData weightQuantMatmulAllReduceTilingData_;
    uint64_t myWorkSpaceSize_{0U};
    uint64_t tileTilingKey_{0U};
};
} // namespace optiling
#endif // WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_310P_H
