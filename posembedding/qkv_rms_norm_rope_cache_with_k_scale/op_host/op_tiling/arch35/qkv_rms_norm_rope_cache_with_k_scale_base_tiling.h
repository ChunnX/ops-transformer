/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASE_TILING_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASE_TILING_H_

#include <cstdint>
#include <string>

#include "op_host/tiling_base.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_tiling.h"

namespace optiling {
namespace QkvRmsNormRopeCacheWithKScale {

struct TensorContractInfo {
    gert::Shape shape;
    gert::Stride stride;
    ge::DataType dtype = ge::DT_UNDEFINED;
    bool shapePresent = false;
    bool stridePresent = false;
};

struct ContractInput {
    uint64_t totalTokens = 0;
    uint64_t batch = 0;
    uint64_t numQHeads = 0;
    uint64_t numKHeads = 0;
    uint64_t numVHeads = 0;
    uint64_t headDim = 0;
    uint64_t maxSeqLen = 0;
    uint64_t blockNum = 0;
    uint64_t blockSize = 0;

    TensorContractInfo qkv;
    TensorContractInfo qGamma;
    TensorContractInfo kGamma;
    TensorContractInfo cosSin;
    TensorContractInfo slotMapping;
    TensorContractInfo kCache;
    TensorContractInfo vCache;
    TensorContractInfo kScaleCache;
    TensorContractInfo cuSeqLen;
    TensorContractInfo qRotation;
    TensorContractInfo kRotation;
    TensorContractInfo vScale;
};

struct ContractTiling {
    uint64_t numQHeads = 0;
    uint64_t numKHeads = 0;
    uint64_t numVHeads = 0;
    uint64_t coreTokenTile = 0;
    uint64_t coreGroupNum = 0;
    uint64_t tokenTile = 0;
    uint64_t tokenTilePerAiv = 0;
    uint64_t rowTile = 0;
    uint64_t rowTileAligned = 0;
    uint64_t dimTile = 0;
    uint64_t kvCacheStrideBlock = 0;
    uint64_t kvCacheStrideHead = 0;
    uint64_t kvCacheStrideToken = 0;
    uint64_t kScaleCacheStrideBlock = 0;
    uint64_t kScaleCacheStrideHead = 0;
    uint64_t kScaleCacheStrideToken = 0;
};

struct TileParam {
    uint64_t tileId = 0;
    uint64_t tokenOffset = 0;
    uint64_t tokenSize = 0;
    uint64_t qkHeadOffset = 0;
    uint64_t qkHeadSize = 0;
    uint64_t vHeadOffset = 0;
    uint64_t vHeadSize = 0;
    uint64_t rowOffset = 0;
    uint64_t rowSize = 0;
};

struct TokenTileSelection {
    uint64_t tokenTile = 0;
    uint64_t tokenTilePerAiv = 0;
};

class QkvRmsNormRopeCacheWithKScaleBaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    static constexpr uint64_t HEAD_DIM_FAST = 128;
    static constexpr uint64_t MAX_TOKEN_TILE = 16;
    static constexpr uint64_t L0A_FULL_DIM_SEGMENT_ROWS = 128;
    static constexpr uint64_t L0C_MAX_ROWS = 256;
    static constexpr uint64_t QKV_INPUT_ROWS_PER_AIV = 160;
    static constexpr uint64_t QK_OUTPUT_ROWS_PER_AIV = 128;
    static constexpr uint64_t V_OUTPUT_ROWS_PER_AIV = 80;
    static constexpr uint64_t DIM_TILE_FAST = 128;
    static constexpr uint64_t AIV_PER_AIC = 2;

    static ge::graphStatus ValidateContractInput(const ContractInput &input,
                                                 const std::string &opName = "QkvRmsNormRopeCacheWithKScale")
        __attribute__((visibility("default")));
    static ge::graphStatus BuildContractTiling(const ContractInput &input, uint64_t aicNum, ContractTiling &tiling,
                                               const std::string &opName = "QkvRmsNormRopeCacheWithKScale")
        __attribute__((visibility("default")));
    explicit QkvRmsNormRopeCacheWithKScaleBaseTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }
    ~QkvRmsNormRopeCacheWithKScaleBaseTiling() override = default;

    void Reset(gert::TilingContext *context) override;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    void Reset();

    static uint64_t CalcQkAlignedRows(uint64_t tokenTile, uint64_t numQHeads, uint64_t numKHeads);
    ge::graphStatus ValidateHeadNums(const std::string &opName) const;
    ge::graphStatus ValidateScalarInputs(const std::string &opName) const;
    static ge::graphStatus ValidateDtype(const std::string &opName, const char *tensorName,
                                         const TensorContractInfo &tensor, ge::DataType expected);
    ge::graphStatus ValidateDtypes(const std::string &opName) const;
    static ge::graphStatus ValidateShapeRank(const TensorContractInfo &tensor, const char *tensorName,
                                             size_t expectedRank, const std::string &opName);
    ge::graphStatus ValidateShapeInputsForDerivedFields(const std::string &opName) const;
    ge::graphStatus ValidateNonCacheShapeRanks(const std::string &opName) const;
    ge::graphStatus ValidateQkvAndGammaShapes(const std::string &opName) const;
    ge::graphStatus ValidatePositionInputShapes(const std::string &opName) const;
    ge::graphStatus ValidateRotationAndScaleShapes(const std::string &opName) const;
    ge::graphStatus ValidateNonCacheShapes(const std::string &opName) const;
    ge::graphStatus ValidateCacheShapes(const std::string &opName) const;
    ge::graphStatus ValidateShapes(const std::string &opName) const;
    ge::graphStatus ValidateStrides(const std::string &opName) const;
    ge::graphStatus ValidateMinimumTokenTileFeasible(const std::string &opName) const;
    bool TrySelectTokenTile(uint64_t tokenTile, const std::string &opName, TokenTileSelection &selection) const;
    ge::graphStatus SelectTokenTile(const std::string &opName, TokenTileSelection &selection) const;
    void FillContractTiling(const TokenTileSelection &selection);
    ge::graphStatus ValidateContractInput(const std::string &opName) const;
    ge::graphStatus BuildContractTiling(const std::string &opName);
    static ge::graphStatus MakeContiguousStrideInfo(const gert::Shape &shape, const std::string &opName,
                                                    gert::Stride &strideInfo);
    static const char *DtypeName(ge::DataType dtype);
    static ge::graphStatus ToTensorInfo(gert::TilingContext *context, const std::string &opName, const char *tensorName,
                                        size_t index, TensorContractInfo &info);
    static ge::graphStatus ToCacheTensorInfo(gert::TilingContext *context, const std::string &opName, size_t index,
                                             TensorContractInfo &info);
    static TensorContractInfo ToOptionalTensorInfo(gert::TilingContext *context, size_t index);
    static void LogTensorInfo(const std::string &opName, const char *tensorName, const TensorContractInfo &info);
    void LogContractInput(const std::string &opName) const;
    static ge::graphStatus SetWorkspace(gert::TilingContext *context, const std::string &opName,
                                        const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo,
                                        size_t &workspaceSize);
    void LogTilingData() const;
    ge::graphStatus ParseHeadNumsAttr(const std::string &opName);
    float ParseEpsilonAttr() const;
    ge::graphStatus FillShapeDerivedFields(const std::string &opName);
    ge::graphStatus FillRequiredTensorInputs(const std::string &opName);
    void FillOptionalTensorInputs();
    ge::graphStatus BuildContractInput(const std::string &opName);
    void FillTilingData();
    static ge::graphStatus ValidateCompileInfo(const std::string &opName,
                                               const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo);
    static ge::graphStatus CopyRawTilingData(gert::TilingContext *context, const std::string &opName,
                                             const QkvRmsNormRopeCacheWithKScaleTilingData &tilingData,
                                             size_t &tilingDataSize);

    const QkvRmsNormRopeCacheWithKScaleCompileInfo *compileInfo_ = nullptr;
    ContractInput input_;
    ContractTiling contractTiling_;
    QkvRmsNormRopeCacheWithKScaleTilingData tilingData_;
    uint64_t aicNum_ = 0;
    float epsilon_ = 1e-6f;
    size_t tilingDataSize_ = 0;
};

} // namespace QkvRmsNormRopeCacheWithKScale
} // namespace optiling

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASE_TILING_H_
