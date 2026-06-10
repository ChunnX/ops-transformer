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

struct TensorShapeInfo {
    uint32_t rank = 0;
    uint64_t dim0 = 0;
    uint64_t dim1 = 0;
    uint64_t dim2 = 0;
    uint64_t dim3 = 0;
    bool present = true;
};

struct TensorStrideInfo {
    uint32_t rank = 0;
    uint64_t stride0 = 0;
    uint64_t stride1 = 0;
    uint64_t stride2 = 0;
    uint64_t stride3 = 0;
    bool present = true;
};

struct TensorContractInfo {
    TensorShapeInfo shape;
    TensorStrideInfo stride;
    ge::DataType dtype = ge::DT_UNDEFINED;
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
    uint64_t numQkHeads = 0;
    uint64_t numQkvHeads = 0;
    uint64_t coreTokenTile = 0;
    uint64_t coreGroupNum = 0;
    uint64_t tokenTile = 0;
    uint64_t tokenTilePerAiv = 0;
    uint64_t rowTile = 0;
    uint64_t rowTileAligned = 0;
    uint64_t dimTile = 0;
    uint64_t kCacheStrideBlock = 0;
    uint64_t kCacheStrideHead = 0;
    uint64_t kCacheStrideToken = 0;
    uint64_t vCacheStrideBlock = 0;
    uint64_t vCacheStrideHead = 0;
    uint64_t vCacheStrideToken = 0;
    uint64_t kScaleCacheStrideBlock = 0;
    uint64_t kScaleCacheStrideHead = 0;
    uint64_t kScaleCacheStrideToken = 0;
};

struct TileParam {
    uint64_t tileId = 0;
    uint64_t bufferId = 0;
    uint64_t tokenOffset = 0;
    uint64_t tokenSize = 0;
    uint64_t qkHeadOffset = 0;
    uint64_t qkHeadSize = 0;
    uint64_t vHeadOffset = 0;
    uint64_t vHeadSize = 0;
    uint64_t rowOffset = 0;
    uint64_t rowSize = 0;
};

struct AivSlice {
    uint64_t tokenOffset = 0;
    uint64_t tokenSize = 0;
    uint64_t rowOffset = 0;
    uint64_t rowSize = 0;
};

struct TokenTileSelection {
    uint64_t tokenTile = 0;
    uint64_t tokenTilePerAiv = 0;
};

class __attribute__((visibility("default"))) QkvRmsNormRopeCacheWithKScaleTiling
    : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    static constexpr uint64_t kHeadDimFast = 128;
    static constexpr uint64_t kMaxTokenTile = 16;
    static constexpr uint64_t kL0aMaxRows = 128;
    static constexpr uint64_t kL0cMaxRows = 256;
    static constexpr uint64_t kUbInputRowsPerAiv = 160;
    static constexpr uint64_t kUbOutputRowsPerAiv = 128;
    static constexpr uint64_t kFp8NtdStagingRowsPerAiv = 80;
    static constexpr uint64_t kDimTileFast = 128;
    static constexpr uint64_t kAivPerAic = 2;

    static ge::graphStatus ValidateContractInput(
        const ContractInput &input, const std::string &opName = "QkvRmsNormRopeCacheWithKScale");
    static ge::graphStatus BuildContractTiling(
        const ContractInput &input, uint64_t aicNum, ContractTiling &tiling,
        const std::string &opName = "QkvRmsNormRopeCacheWithKScale");
    static AivSlice MakeAivSlice(const TileParam &tile, uint64_t qkHeadTile, uint64_t aivLocalId);

    explicit QkvRmsNormRopeCacheWithKScaleTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {}
    ~QkvRmsNormRopeCacheWithKScaleTiling() override = default;

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

    static bool IsShape1D(const TensorShapeInfo &shape, uint64_t d0);
    static bool IsShape2D(const TensorShapeInfo &shape, uint64_t d0, uint64_t d1);
    static bool IsShape3D(const TensorShapeInfo &shape, uint64_t d0, uint64_t d1, uint64_t d2);
    static bool IsShape4D(const TensorShapeInfo &shape, uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3);
    static bool HasDtype(const TensorContractInfo &tensor, ge::DataType expected);
    static bool IsCacheStrideValid(const TensorStrideInfo &stride, uint64_t headDim);
    static bool IsScaleStrideValid(const TensorStrideInfo &stride);
    static bool CacheNonInnermostStridesMatch(const TensorStrideInfo &lhs, const TensorStrideInfo &rhs);
    static uint64_t CalcQkAlignedRows(uint64_t tokenTile, uint64_t numQHeads, uint64_t numKHeads);
    static void LogContractInputSummary(const std::string &opName, const ContractInput &input, uint64_t aicNum,
                                        const ContractTiling &tiling);
    static void LogTokenTileCandidate(const std::string &opName, uint64_t tokenTilePerAiv, uint64_t tokenTile,
                                      uint64_t rowTile, uint64_t rowTileAligned, uint64_t qkAlignedRows, bool fitsUb,
                                      bool fitsL0a, bool fitsL0c, bool accepted);
    static void FillContractHeadNums(const ContractInput &input, ContractTiling &tiling);
    static ge::graphStatus ValidateScalarInputs(const ContractInput &input, const std::string &opName);
    static ge::graphStatus ValidateContractHeadNums(const ContractTiling &tiling, const std::string &opName);
    static ge::graphStatus ValidateDtype(const std::string &opName, const char *tensorName,
                                         const TensorContractInfo &tensor, ge::DataType expected);
    static ge::graphStatus ValidateRequiredDtypes(const ContractInput &input, const std::string &opName);
    static ge::graphStatus ValidateCacheDtypes(const ContractInput &input, const std::string &opName);
    static ge::graphStatus ValidateOptionalDtypes(const ContractInput &input, const std::string &opName);
    static ge::graphStatus ValidateDtypes(const ContractInput &input, const std::string &opName);
    static ge::graphStatus ValidateShapes(const ContractInput &input, const ContractTiling &tiling,
                                          const std::string &opName);
    static ge::graphStatus ValidateStrides(const ContractInput &input, const std::string &opName);
    static bool TrySelectTokenTile(const ContractTiling &tiling, uint64_t fp8NtdStagingHeads, uint64_t tokenTile,
                                   const std::string &opName,
                                   TokenTileSelection &selection);
    static ge::graphStatus SelectTokenTile(const ContractTiling &tiling, const std::string &opName,
                                           TokenTileSelection &selection);
    static void FillContractTiling(const ContractInput &input, uint64_t aicNum,
                                   const TokenTileSelection &selection, ContractTiling &tiling);
    static void LogContractTilingResult(const std::string &opName, const ContractTiling &tiling);
    static TensorShapeInfo ToShapeInfo(const gert::Shape &shape);
    static TensorShapeInfo ToShapeInfo(const gert::StorageShape *shape);
    static TensorStrideInfo ToStrideInfo(const gert::Stride &stride);
    static bool HasUsableBlockDims(const TensorShapeInfo &shape);
    static bool BlockDimsMatch(const TensorShapeInfo &lhs, const TensorShapeInfo &rhs);
    static ge::graphStatus MakeContiguousStrideInfo(const gert::Shape &shape, const std::string &opName,
                                                    TensorStrideInfo &strideInfo);
    static const char *DtypeName(ge::DataType dtype);
    static void FillInputDtype(gert::TilingContext *context, size_t index, TensorContractInfo &info);
    static TensorContractInfo ToTensorInfo(gert::TilingContext *context, size_t index);
    static ge::graphStatus ToCacheTensorInfo(gert::TilingContext *context, const std::string &opName, size_t index,
                                             TensorContractInfo &info);
    static TensorContractInfo ToOptionalTensorInfo(gert::TilingContext *context, size_t index);
    static void LogTensorInfo(const std::string &opName, const char *tensorName, const TensorContractInfo &info);
    static void LogContractInput(const std::string &opName, const ContractInput &input, uint64_t aicNum,
                                 float epsilon);
    static size_t CalcOpWorkspaceSize(const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo);
    static size_t CalcWorkspaceSize(size_t opWorkspaceSize);
    static ge::graphStatus SetWorkspace(gert::TilingContext *context, const std::string &opName,
                                        const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo,
                                        size_t &workspaceSize);
    static void LogTilingData(const std::string &opName,
                              const QkvRmsNormRopeCacheWithKScaleTilingData &tilingData,
                              const ContractTiling &contractTiling, uint64_t tilingKey, uint64_t blockDim,
                              size_t dataSize, size_t workspaceSize);
    static ge::graphStatus ParseHeadNumsAttr(gert::TilingContext *context, const std::string &opName,
                                             ContractInput &input);
    static float ParseEpsilonAttr(gert::TilingContext *context);
    static void FillShapeDerivedFields(const TensorContractInfo &qkv, const TensorContractInfo &cosSin,
                                       const TensorContractInfo &cuSeqLen, ContractInput &input);
    static void DeriveBlockShape(const std::string &opName, ContractInput &input);
    static ge::graphStatus FillRequiredTensorInputs(gert::TilingContext *context, const std::string &opName,
                                                    ContractInput &input);
    static void FillOptionalTensorInputs(gert::TilingContext *context, ContractInput &input);
    static ge::graphStatus BuildContractInput(gert::TilingContext *context, const std::string &opName,
                                              ContractInput &input, float &epsilon);
    static void FillTilingData(const ContractInput &input, const ContractTiling &contractTiling, float epsilon,
                               QkvRmsNormRopeCacheWithKScaleTilingData &tilingData);
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
