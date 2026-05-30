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
 * \file weight_quant_matmul_all_reduce_tiling_950.h
 * \brief
 */
#ifndef WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_950_H
#define WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_950_H
#include "../matmul_all_reduce_tiling_base.h"
#include "weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom.h"
#include "../../../op_kernel/arch35/matmul_all_reduce_tiling_struct_ar35.h"

namespace optiling {
struct WeightQuantMMAllReduceTilingKeyParams
{
    bool transB{false};
    bool biasIsExist{false};
    uint8_t antiQuantType{0};
    uint8_t templateCustom{0};
    uint8_t quantType{0};
    bool hasAntiQuantOffset{false};
    bool isBiasFp32{false};
    uint8_t weightFormat{0};
};

class WeightQuantMatmulAllReduceTilingA5 : public MatmulAllReduceTilingBase
{
    friend class WeightQuantTilingTransferHelperA5;
    friend class WeightQuantAsTilingTransferHelper;

public:
    explicit WeightQuantMatmulAllReduceTilingA5(gert::TilingContext* context);
    ~WeightQuantMatmulAllReduceTilingA5() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus GetWorkspaceSize() override;

    ge::graphStatus PostTiling() override;

    Mc2Tiling::RCSTiling& MutableRCSTilingData() override
    {
        if (antiQuantType_ != AntiQuantType::PER_GROUP) {
            return weightQuantMatmulAllReduceA5Fp8TilingData_.param;
        }
        return weightQuantMatmulAllReduceA5TilingData_.param;
    }

    ::TCubeTiling& MutableTCubeTileTilingData() override
    {
        if (antiQuantType_ != AntiQuantType::PER_GROUP) {
            return weightQuantMatmulAllReduceA5Fp8TilingData_.tileMmASTiling.matmulTiling;
        }
        return weightQuantMatmulAllReduceA5TilingData_.tileRegBaseMmTiling.matmulTiling;
    }

    ::TCubeTiling& MutableTCubeTailTilingData() override
    {
        if (antiQuantType_ != AntiQuantType::PER_GROUP) {
            return weightQuantMatmulAllReduceA5Fp8TilingData_.tailMmASTiling.matmulTiling;
        }
        return weightQuantMatmulAllReduceA5TilingData_.tailRegBaseMmTiling.matmulTiling;
    }

    void PrintExtendMatmulTiling(bool isTail) override;
    void PrintMatmulAsTiling(bool isTail);

    ge::graphStatus DoWeightQuantTiling();
    ge::graphStatus DoWeightQuantAsTiling();

    void DoEmptyTensorTiling() override;

    ge::graphStatus CheckInput() override;

    ge::graphStatus GetWorkspaceSizeForA2ARSAG();
    ge::graphStatus SetMc2HcommAllReduce(const char* groupName, const uint32_t reduceType);
    ge::graphStatus SetMc2HcommTwoShot(const char* groupName, const uint32_t reduceType);
    ge::graphStatus SetMc2Hcomm();

    CutResult GetTilingResult() override;

private:
    ge::graphStatus CheckBiasInput();
    ge::graphStatus CheckAxisSize();
    Mc2Tiling::WeightQuantMatmulAllReduceA5TilingData weightQuantMatmulAllReduceA5TilingDataSelf_{};
    Mc2Tiling::WeightQuantMatmulAllReduceA5Fp8TilingData weightQuantMatmulAllReduceA5Fp8TilingDataSelf_{};

    Mc2Tiling::WeightQuantMatmulAllReduceA5TilingData& weightQuantMatmulAllReduceA5TilingData_;
    Mc2Tiling::WeightQuantMatmulAllReduceA5Fp8TilingData& weightQuantMatmulAllReduceA5Fp8TilingData_;
    uint64_t myWorkSpaceSize_{0U};
    bool isWeightFp8Hif8_{false};
    WeightQuantMMAllReduceTilingKeyParams WeightQuantTPLPatams_;
};

class WeightQuantTilingTransferHelperA5 : public Mc2WeightQuantBatchMatmulV2RegBase
{
public:
    WeightQuantTilingTransferHelperA5(
        WeightQuantMatmulAllReduceTilingA5& weightQuantMatmulAllReduceTiling,
        Mc2WeightQuantBatchMatmulV2RegBaseTilingData& data)
        : Mc2WeightQuantBatchMatmulV2RegBase(weightQuantMatmulAllReduceTiling.context_),
          tilingProcesser_(weightQuantMatmulAllReduceTiling),
          data_(data)
    {
        Mc2WeightQuantBatchMatmulV2RegBase::InitCompileInfo();
    }
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus PostTiling() override;
    WeightQuantMMAllReduceTilingKeyParams GetWeightQuantMMAllReduceTPLParam();
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus MatmulDoTiling()
    {
        if (DoTiling() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        data_.cubeBlockDimN = tilDat_->cubeBlockDimN;
        data_.cubeBlockDimM = tilDat_->cubeBlockDimM;
        data_.reserve1 = tilDat_->reserve1;
        data_.vecCoreParallel = tilDat_->vecCoreParallel;
        data_.AL1Pingpong = tilDat_->AL1Pingpong;
        data_.BL1Pingpong = tilDat_->BL1Pingpong;
        data_.kSize = tilDat_->kSize;
        data_.nSize = tilDat_->nSize;
        data_.groupSize = tilDat_->groupSize;
        data_.mSize = tilDat_->mSize;
        data_.nBubSize = tilDat_->nBubSize;
        data_.kBubSize = tilDat_->kBubSize;

        data_.matmulTiling.M = tilDat_->matmulTiling.M;
        data_.matmulTiling.Ka = tilDat_->matmulTiling.Ka;
        data_.matmulTiling.N = tilDat_->matmulTiling.N;
        data_.matmulTiling.Kb = tilDat_->matmulTiling.Kb;
        data_.matmulTiling.singleCoreM = tilDat_->matmulTiling.singleCoreM;
        data_.matmulTiling.singleCoreN = tilDat_->matmulTiling.singleCoreN;
        data_.matmulTiling.singleCoreK = tilDat_->matmulTiling.singleCoreK;
        data_.matmulTiling.baseM = tilDat_->matmulTiling.baseM;
        data_.matmulTiling.baseN = tilDat_->matmulTiling.baseN;
        data_.matmulTiling.baseK = tilDat_->matmulTiling.baseK;
        data_.matmulTiling.dbL0A = tilDat_->matmulTiling.dbL0A;
        data_.matmulTiling.dbL0B = tilDat_->matmulTiling.dbL0B;
        data_.matmulTiling.dbL0C = tilDat_->matmulTiling.dbL0C;
        data_.matmulTiling.stepM = tilDat_->matmulTiling.stepM;
        data_.matmulTiling.stepN = tilDat_->matmulTiling.stepN;
        data_.matmulTiling.stepKa = tilDat_->matmulTiling.stepKa;
        data_.matmulTiling.stepKb = tilDat_->matmulTiling.stepKb;
        data_.matmulTiling.depthA1 = tilDat_->matmulTiling.depthA1;
        data_.matmulTiling.depthB1 = tilDat_->matmulTiling.depthB1;
        data_.matmulTiling.iterateOrder = tilDat_->matmulTiling.iterateOrder;
        data_.matmulTiling.isBias = tilDat_->matmulTiling.isBias;
        data_.matmulTiling.shareMode = tilDat_->matmulTiling.shareMode;
        data_.matmulTiling.shareL1Size = tilDat_->matmulTiling.shareMode;
        data_.matmulTiling.shareL0CSize = tilDat_->matmulTiling.shareL0CSize;
        return ge::GRAPH_SUCCESS;
    }

private:
    WeightQuantMatmulAllReduceTilingA5& tilingProcesser_;
    Mc2WeightQuantBatchMatmulV2RegBaseTilingData& data_;
};

class WeightQuantAsTilingTransferHelper : public Mc2weight_quant_batch_matmul_v2::Mc2WeightQuantBatchMatmulV2TilingAS
{
public:
    WeightQuantAsTilingTransferHelper(
        WeightQuantMatmulAllReduceTilingA5& weightQuantMatmulAllReduceTiling,
        Mc2WeightQuantBatchMatmulV2ASTilingData& data)
        : Mc2WeightQuantBatchMatmulV2TilingAS(weightQuantMatmulAllReduceTiling.context_),
          tilingProcesser_(weightQuantMatmulAllReduceTiling),
          mmASTilingdata_(data)
    {
        Mc2WeightQuantBatchMatmulV2TilingAS::InitCompileInfo();
    }
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus PostTiling() override;
    void PrintTilingInputParam(std::unique_ptr<Mc2WeightQuantBatchMatmulInfo>& matmulInfo);
    WeightQuantMMAllReduceTilingKeyParams GetWeightQuantASMMAllReduceTPLParam();
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus MatmulDoTiling()
    {
        if (DoTiling() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        mmASTilingdata_.cubeBlockDimM = tilDat_->cubeBlockDimM;
        mmASTilingdata_.cubeBlockDimN = tilDat_->cubeBlockDimN;
        mmASTilingdata_.hasBias = tilDat_->hasBias;
        mmASTilingdata_.firstTailBlockCount = tilDat_->firstTailBlockCount;
        mmASTilingdata_.secondTailBlockCount = tilDat_->secondTailBlockCount;
        mmASTilingdata_.weightL2Cacheable = tilDat_->weightL2Cacheable;
        mmASTilingdata_.mainBlockL1Size = tilDat_->mainBlockL1Size;
        mmASTilingdata_.firstTailBlockL1Size = tilDat_->firstTailBlockL1Size;
        mmASTilingdata_.secondTailBlockL1Size = tilDat_->secondTailBlockL1Size;
        mmASTilingdata_.aPreloadSize = tilDat_->aPreloadSize;
        mmASTilingdata_.groupSize = tilDat_->groupSize;
        mmASTilingdata_.mainBlockCount = tilDat_->mainBlockCount;
        mmASTilingdata_.mSize = tilDat_->mSize;
        mmASTilingdata_.kSize = tilDat_->kSize;
        mmASTilingdata_.nSize = tilDat_->nSize;

        mmASTilingdata_.matmulTiling.usedCoreNum = tilDat_->matmulTiling.usedCoreNum;
        mmASTilingdata_.matmulTiling.M = tilDat_->matmulTiling.M;
        mmASTilingdata_.matmulTiling.Ka = tilDat_->matmulTiling.Ka;
        mmASTilingdata_.matmulTiling.N = tilDat_->matmulTiling.N;
        mmASTilingdata_.matmulTiling.Kb = tilDat_->matmulTiling.Kb;
        mmASTilingdata_.matmulTiling.singleCoreM = tilDat_->matmulTiling.singleCoreM;
        mmASTilingdata_.matmulTiling.singleCoreN = tilDat_->matmulTiling.singleCoreN;
        mmASTilingdata_.matmulTiling.singleCoreK = tilDat_->matmulTiling.singleCoreK;
        mmASTilingdata_.matmulTiling.baseM = tilDat_->matmulTiling.baseM;
        mmASTilingdata_.matmulTiling.baseN = tilDat_->matmulTiling.baseN;
        mmASTilingdata_.matmulTiling.baseK = tilDat_->matmulTiling.baseK;
        mmASTilingdata_.matmulTiling.dbL0A = tilDat_->matmulTiling.dbL0A;
        mmASTilingdata_.matmulTiling.dbL0B = tilDat_->matmulTiling.dbL0B;
        mmASTilingdata_.matmulTiling.dbL0C = tilDat_->matmulTiling.dbL0C;
        mmASTilingdata_.matmulTiling.stepM = tilDat_->matmulTiling.stepM;
        mmASTilingdata_.matmulTiling.stepN = tilDat_->matmulTiling.stepN;
        mmASTilingdata_.matmulTiling.stepKa = tilDat_->matmulTiling.stepKa;
        mmASTilingdata_.matmulTiling.stepKb = tilDat_->matmulTiling.stepKb;
        mmASTilingdata_.matmulTiling.depthA1 = tilDat_->matmulTiling.depthA1;
        mmASTilingdata_.matmulTiling.depthB1 = tilDat_->matmulTiling.depthB1;
        mmASTilingdata_.matmulTiling.iterateOrder = tilDat_->matmulTiling.iterateOrder;
        mmASTilingdata_.matmulTiling.isBias = tilDat_->matmulTiling.isBias;
        mmASTilingdata_.matmulTiling.shareMode = tilDat_->matmulTiling.shareMode;
        mmASTilingdata_.matmulTiling.shareL1Size = tilDat_->matmulTiling.shareMode;
        mmASTilingdata_.matmulTiling.shareL0CSize = tilDat_->matmulTiling.shareL0CSize;
        return ge::GRAPH_SUCCESS;
    }

private:
    WeightQuantMatmulAllReduceTilingA5& tilingProcesser_;
    Mc2WeightQuantBatchMatmulV2ASTilingData& mmASTilingdata_;
};

} // namespace optiling
#endif // WEIGHT_QUANT_MATMUL_ALL_REDUCE_TILING_950_H
