/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "../../../op_kernel/arch35/weight_quant_basic_block/weight_quant_tiling_key.h"
#include "blaze/gemm/utils/a8w4_fixed_tile_resource.h"
#include "gmm_csv_ge_parse_utils.h"
#include "tiling_case_executor.h"

extern "C" ge::graphStatus TilingGroupedA8W4BasicApiForTest(gert::TilingContext *context);

namespace {
using TensorDescription = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;
using A8W4TilingData = GroupedMatmulTilingData::GMMA8W4BasicApiTilingData;

constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
constexpr uint64_t A8W4_ND_PERGROUP_TILING_KEY = 0x20600300UL;
constexpr uint64_t A8W4_ND_PERCHANNEL_TILING_KEY = 0x20600200UL;
constexpr uint64_t A8W4_NZ_PERGROUP_TILING_KEY = 0x20600301UL;
constexpr uint64_t A8W4_NZ_PERCHANNEL_TILING_KEY = 0x20600201UL;

uint64_t AlignWorkspace(uint64_t value)
{
    constexpr uint64_t align = 512UL;
    return (value + align - 1UL) / align * align;
}

static_assert(sizeof(A8W4TilingData) % sizeof(uint64_t) == 0);

uint64_t GetExpectedA8W4TilingKey(ge::Format weightFormat, uint64_t cQuantType)
{
    if (weightFormat == ge::FORMAT_FRACTAL_NZ) {
        return cQuantType == WQGMM_PER_GROUP ? A8W4_NZ_PERGROUP_TILING_KEY :
                                              A8W4_NZ_PERCHANNEL_TILING_KEY;
    }
    return cQuantType == WQGMM_PER_GROUP ? A8W4_ND_PERGROUP_TILING_KEY :
                                          A8W4_ND_PERCHANNEL_TILING_KEY;
}

struct A8W4Case {
    int64_t m = 512;
    int64_t k = 1024;
    int64_t n = 256;
    int64_t groupNum = 2;
    int64_t scaleGroupNum = 4;
    int64_t groupListTypeAttr = 0;
    ge::DataType weightDtype = ge::DT_INT4;
    ge::DataType outDtype = ge::DT_BF16;
    ge::Format weightFormat = ge::FORMAT_ND;
    bool hasBias = true;
    bool hasOffset = false;
    bool hasAntiquantScale = false;
    bool hasAntiquantOffset = false;
    bool specialWeightFormat = false;
    bool invalidPackedStorageShape = false;
    bool apiNormalizedPackedInt32 = false;
    std::vector<int64_t> tuningConfig = {0};
};

gert::StorageShape EmptyShape()
{
    return gert::StorageShape();
}

TensorDescription MakeTensor(const std::vector<int64_t> &shape, ge::DataType dtype,
                             ge::Format format = ge::FORMAT_ND)
{
    return {ops::ut::MakeGertStorageShape(shape), dtype, format};
}

TensorDescription MakeTensor(const std::vector<int64_t> &originShape, const std::vector<int64_t> &storageShape,
                             ge::DataType dtype, ge::Format format)
{
    return {ops::ut::MakeGertStorageShape(originShape, storageShape), dtype, format};
}

std::vector<OpAttr> MakeAttrs(const A8W4Case &param)
{
    return {
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupListTypeAttr)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(param.tuningConfig)},
    };
}

optiling::GMMCompileInfo MakeCompileInfo()
{
    return {
        32,                                      // aicNum
        64,                                      // aivNum
        262144,                                  // ubSize
        524288,                                  // l1Size
        134217728,                               // l2Size
        262144,                                  // l0CSize
        65536,                                   // l0ASize
        65536,                                   // l0BSize
        platform_ascendc::SocVersion::ASCEND950, // socVersion
        NpuArch::DAV_3510,
    };
}

gert::TilingContextPara MakeContext(const A8W4Case &param, optiling::GMMCompileInfo *compileInfo)
{
    std::vector<int64_t> weightOriginShape =
        param.specialWeightFormat ?
            std::vector<int64_t>{param.groupNum, param.n, param.k} :
            std::vector<int64_t>{param.groupNum, param.k, param.n};
    std::vector<int64_t> weightStorageShape = weightOriginShape;
    if (param.weightFormat == ge::FORMAT_FRACTAL_NZ) {
        weightStorageShape =
            param.specialWeightFormat ?
                std::vector<int64_t>{param.groupNum, (param.k + 63) / 64, (param.n + 15) / 16, 16, 64} :
                std::vector<int64_t>{param.groupNum, (param.n + 63) / 64, (param.k + 15) / 16, 16, 64};
        if (param.weightDtype == ge::DT_INT32) {
            weightStorageShape.back() /= 8;
        }
    } else if (param.weightDtype == ge::DT_INT32 || param.apiNormalizedPackedInt32) {
        weightStorageShape.back() /= 8;
    }
    if (param.invalidPackedStorageShape) {
        ++weightStorageShape.back();
    }

    const gert::StorageShape offsetShape =
        param.hasOffset
            ? ops::ut::MakeGertStorageShape(std::vector<int64_t>{param.groupNum, 1, param.n})
            : EmptyShape();
    const gert::StorageShape biasShape =
        param.hasBias ? ops::ut::MakeGertStorageShape(std::vector<int64_t>{param.groupNum, param.n}) : EmptyShape();
    const gert::StorageShape antiquantScaleShape =
        param.hasAntiquantScale
            ? ops::ut::MakeGertStorageShape(std::vector<int64_t>{param.groupNum, 1, param.n})
            : EmptyShape();
    const gert::StorageShape antiquantOffsetShape =
        param.hasAntiquantOffset
            ? ops::ut::MakeGertStorageShape(std::vector<int64_t>{param.groupNum, 1, param.n})
            : EmptyShape();
    std::vector<TensorDescription> inputs = {
        MakeTensor({param.m, param.k}, ge::DT_INT8), // x
        MakeTensor(weightOriginShape, weightStorageShape, param.weightDtype, param.weightFormat),
        {biasShape, ge::DT_FLOAT, ge::FORMAT_ND}, // bias
        MakeTensor({param.groupNum, param.scaleGroupNum, param.n}, ge::DT_UINT64), // scale
        {offsetShape, ge::DT_FLOAT, ge::FORMAT_ND},                                 // offset
        {antiquantScaleShape, ge::DT_FLOAT, ge::FORMAT_ND},                        // antiquantScale
        {antiquantOffsetShape, ge::DT_FLOAT, ge::FORMAT_ND},                       // antiquantOffset
        MakeTensor({param.groupNum}, ge::DT_INT64),                                // groupList
        MakeTensor({param.m}, ge::DT_FLOAT),                                       // perTokenScale
    };
    std::vector<TensorDescription> outputs = {
        MakeTensor({param.m, param.n}, param.outDtype),
    };
    return gert::TilingContextPara("GroupedMatmul", inputs, outputs, MakeAttrs(param), compileInfo, "3510", 32,
                                   262144);
}

bool ExecuteA8W4Tiling(const A8W4Case &param, TilingInfo &tilingInfo)
{
    auto compileInfo = MakeCompileInfo();
    auto context = MakeContext(param, &compileInfo);
    return ExecuteTiling(context, tilingInfo, TilingGroupedA8W4BasicApiForTest);
}

bool ExecuteA8W4TilingWithCompileInfo(
    const A8W4Case &param, optiling::GMMCompileInfo &compileInfo, TilingInfo &tilingInfo)
{
    auto context = MakeContext(param, &compileInfo);
    return ExecuteTiling(context, tilingInfo, TilingGroupedA8W4BasicApiForTest);
}

bool ExecuteA8W4ProductionTiling(const A8W4Case &param, TilingInfo &tilingInfo)
{
    auto compileInfo = MakeCompileInfo();
    auto context = MakeContext(param, &compileInfo);
    return ExecuteTiling(context, tilingInfo);
}

const A8W4TilingData *GetTilingData(const TilingInfo &tilingInfo)
{
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(A8W4TilingData));
    return reinterpret_cast<const A8W4TilingData *>(tilingInfo.tilingData.get());
}

TEST(GroupedA8W4BasicApiArch35Tiling, ExactTilingKeys)
{
    EXPECT_EQ(
        GET_TPL_TILING_KEY(
            WQGMM_ND, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST, WQGMM_PER_GROUP, WQGMM_NONE,
            WQGMM_NO_TRANS, WQGMM_NO_TRANS, WQGMM_A8W4_FIXED_TEMPLATE, WQGMM_NOT_SINGLE_MULTI_SINGLE,
            WQGMM_VDEFAULT, WQGMM_MULTI_SCALE_DEQUANT),
        A8W4_ND_PERGROUP_TILING_KEY);
    EXPECT_EQ(
        GET_TPL_TILING_KEY(
            WQGMM_ND, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST, WQGMM_PER_CHANNEL, WQGMM_NONE,
            WQGMM_NO_TRANS, WQGMM_NO_TRANS, WQGMM_A8W4_FIXED_TEMPLATE, WQGMM_NOT_SINGLE_MULTI_SINGLE,
            WQGMM_VDEFAULT, WQGMM_MULTI_SCALE_DEQUANT),
        A8W4_ND_PERCHANNEL_TILING_KEY);
    EXPECT_EQ(
        GET_TPL_TILING_KEY(
            WQGMM_FRACTAL_NZ, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST, WQGMM_PER_GROUP, WQGMM_NONE,
            WQGMM_NO_TRANS, WQGMM_NO_TRANS, WQGMM_A8W4_FIXED_TEMPLATE, WQGMM_NOT_SINGLE_MULTI_SINGLE,
            WQGMM_VDEFAULT, WQGMM_MULTI_SCALE_DEQUANT),
        A8W4_NZ_PERGROUP_TILING_KEY);
    EXPECT_EQ(
        GET_TPL_TILING_KEY(
            WQGMM_FRACTAL_NZ, WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST, WQGMM_PER_CHANNEL, WQGMM_NONE,
            WQGMM_NO_TRANS, WQGMM_NO_TRANS, WQGMM_A8W4_FIXED_TEMPLATE, WQGMM_NOT_SINGLE_MULTI_SINGLE,
            WQGMM_VDEFAULT, WQGMM_MULTI_SCALE_DEQUANT),
        A8W4_NZ_PERCHANNEL_TILING_KEY);
}

TEST(GroupedA8W4BasicApiArch35Tiling, V5SymmetricPergroupUsesProductionRouter)
{
    A8W4Case param;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4ProductionTiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, A8W4_ND_PERGROUP_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmQuantParams.bQuantMode,
              static_cast<uint32_t>(optiling::QuantMode::PERGROUP_MODE));
}

TEST(GroupedA8W4BasicApiArch35Tiling, V5AsymmetricPerchannelUsesProductionRouter)
{
    A8W4Case param;
    param.k = 1001;
    param.scaleGroupNum = 1;
    param.hasOffset = true;
    param.outDtype = ge::DT_FLOAT16;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4ProductionTiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, A8W4_ND_PERCHANNEL_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmQuantParams.bQuantMode,
              static_cast<uint32_t>(optiling::QuantMode::PERCHANNEL_MODE));
    EXPECT_EQ(tilingData->mmTilingData.k, 1001U);
}

TEST(GroupedA8W4BasicApiArch35Tiling, V5ApiNormalizedPackedInt32UsesProductionRouter)
{
    A8W4Case param;
    // This is the descriptor state produced by UnpackB32ToB4 for an ND packed-INT32 input:
    // logical dtype/shape are INT4 [E,K,N], while storage retains [E,K,N/8].
    param.apiNormalizedPackedInt32 = true;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4ProductionTiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, A8W4_ND_PERGROUP_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->a8w4Params.weightPackedInt32, 1U);
}

TEST(GroupedA8W4BasicApiArch35Tiling, SymmetricPergroupUsesOneQuantGroupPerL1Window)
{
    A8W4Case param;
    param.groupListTypeAttr = 99; // The documented A8W4 path ignores this value and treats it as count mode.

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, GetExpectedA8W4TilingKey(param.weightFormat, WQGMM_PER_GROUP));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmQuantParams.groupListType, 1);
    EXPECT_EQ(tilingData->gmmQuantParams.bQuantMode,
              static_cast<uint32_t>(optiling::QuantMode::PERGROUP_MODE));
    EXPECT_EQ(tilingData->a8w4Params.quantGroupSize, 256U);
    EXPECT_EQ(tilingData->a8w4Params.quantGroupNum, 4U);
    EXPECT_EQ(tilingData->a8w4Params.dequantMode, 0U);
    EXPECT_EQ(tilingData->mmTilingData.baseM, 256U);
    EXPECT_EQ(tilingData->mmTilingData.baseN, 256U);
    EXPECT_EQ(tilingData->mmTilingData.baseK, 128U);
    EXPECT_EQ(tilingData->mmTilingData.kAL1, 256U);
    EXPECT_EQ(tilingData->mmTilingData.kBL1, 256U);
    EXPECT_EQ(tilingData->mmTilingData.dbL0C, 1U);
    EXPECT_EQ(tilingData->a8w4Params.coreNum, 32U);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE);
}

TEST(GroupedA8W4BasicApiArch35Tiling, AsymmetricPerchannelUsesOffsetAndTilePipelineWorkspace)
{
    A8W4Case param;
    param.hasOffset = true;
    param.scaleGroupNum = 1;
    param.outDtype = ge::DT_FLOAT16;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, GetExpectedA8W4TilingKey(param.weightFormat, WQGMM_PER_CHANNEL));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->gmmQuantParams.bQuantMode,
              static_cast<uint32_t>(optiling::QuantMode::PERCHANNEL_MODE));
    EXPECT_EQ(tilingData->a8w4Params.quantGroupNum, 1U);
    EXPECT_EQ(tilingData->a8w4Params.dequantMode, 1U);
    EXPECT_EQ(tilingData->a8w4Params.hasOffset, 1U);
    EXPECT_EQ(tilingData->a8w4Params.enableWeightPreprocess, 1U);
    EXPECT_EQ(tilingData->a8w4Params.expandedWeightStrideBytes,
              static_cast<uint64_t>(param.k * param.n));
    EXPECT_EQ(tilingData->mmTilingData.baseM, 256U);
    EXPECT_EQ(tilingData->mmTilingData.kAL1, 256U);
    EXPECT_EQ(tilingData->mmTilingData.kBL1, 256U);
    EXPECT_EQ(tilingData->mmTilingData.dbL0C, 1U);
    EXPECT_EQ(tilingData->a8w4Params.coreNum, 32U);
    const uint64_t expandedSize = AlignWorkspace(static_cast<uint64_t>(param.groupNum * param.k * param.n));
    const uint64_t tileStride = AlignWorkspace(256UL * 256UL * sizeof(uint16_t));
    const uint64_t tileSize = 32UL * tileStride;
    const uint64_t rowSumSize = AlignWorkspace(static_cast<uint64_t>(param.m) * sizeof(float));
    EXPECT_EQ(tilingData->a8w4Params.expandedWeightOffsetBytes, 0U);
    EXPECT_EQ(tilingData->a8w4Params.expandedWeightSizeBytes, expandedSize);
    EXPECT_EQ(tilingData->a8w4Params.tileWorkspaceOffsetBytes, expandedSize);
    EXPECT_EQ(tilingData->a8w4Params.tileWorkspaceStrideBytes, tileStride);
    EXPECT_EQ(tilingData->a8w4Params.tileWorkspaceSizeBytes, tileSize);
    EXPECT_EQ(tilingData->a8w4Params.rowSumOffsetBytes, expandedSize + tileSize);
    EXPECT_EQ(tilingData->a8w4Params.rowSumSizeBytes, rowSumSize);
    EXPECT_EQ(tilingData->a8w4Params.userWorkspaceSizeBytes, expandedSize + tileSize + rowSumSize);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE + expandedSize + tileSize + rowSumSize);
}

TEST(GroupedA8W4BasicApiArch35Tiling, FixedTilePhysicalFootprintSeparatesBAndScaleForSmallN)
{
    using Resource = Blaze::Gemm::A8W4FixedTileResource;
    constexpr uint64_t baseM = 16UL;
    constexpr uint64_t baseN = 16UL;
    constexpr uint64_t kL1 = 256UL;
    constexpr uint64_t aBytes = Resource::GetAL1Bytes(baseM, kL1);
    constexpr uint64_t bBytes = Resource::GetBL1Bytes(baseN, kL1);
    constexpr uint64_t scaleBytes = Resource::GetScaleL1Bytes(baseN);
    constexpr uint64_t scaleOffset = aBytes + bBytes;

    static_assert(aBytes == 4096UL);
    static_assert(bBytes == 8192UL);
    static_assert(scaleBytes == 128UL);
    static_assert(scaleOffset == 12288UL);
    EXPECT_EQ(Resource::GetL1StageBytes(baseM, baseN, kL1, kL1), 12416UL);
    EXPECT_EQ(scaleOffset, aBytes + bBytes);
}

TEST(GroupedA8W4BasicApiArch35Tiling, PerchannelNTailsUseLogicalTileAndPhysicalNzFootprint)
{
    using Resource = Blaze::Gemm::A8W4FixedTileResource;
    for (const int64_t n : {8L, 16L, 24L, 32L, 248L, 256L}) {
        A8W4Case param;
        param.m = 8;
        param.k = 512;
        param.n = n;
        param.scaleGroupNum = 1;
        param.hasOffset = true;
        param.outDtype = ge::DT_FLOAT16;

        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo)) << "N=" << n;
        const auto *tilingData = GetTilingData(tilingInfo);
        ASSERT_NE(tilingData, nullptr);
        const uint64_t expectedBaseN = Resource::AlignUp(static_cast<uint64_t>(n), Resource::CUBE_BLOCK);
        EXPECT_EQ(tilingData->mmTilingData.baseM, 16U);
        EXPECT_EQ(tilingData->mmTilingData.baseN, expectedBaseN);
        EXPECT_EQ(Resource::GetBL1Bytes(expectedBaseN, tilingData->mmTilingData.kBL1),
                  Resource::AlignUp(256UL, Resource::CUBE_BLOCK) *
                      Resource::AlignUp(expectedBaseN, Resource::INT8_C0));
    }
}

TEST(GroupedA8W4BasicApiArch35Tiling, MaximumFixedTileMatchesA5ResourceBudget)
{
    using Resource = Blaze::Gemm::A8W4FixedTileResource;
    EXPECT_EQ(Resource::GetAL1Bytes(256UL, 256UL), 64UL * 1024UL);
    EXPECT_EQ(Resource::GetBL1Bytes(256UL, 256UL), 64UL * 1024UL);
    EXPECT_EQ(Resource::GetScaleL1Bytes(256UL), 2UL * 1024UL);
    EXPECT_EQ(Resource::GetL1StageBytes(256UL, 256UL, 256UL, 256UL), 130UL * 1024UL);
    EXPECT_EQ(Resource::GetRequiredL1Bytes(256UL, 256UL, 256UL, 256UL), 260UL * 1024UL);
    EXPECT_EQ(Resource::L0_BUFFER_NUM * Resource::GetL0ABytes(256UL, 128UL), 64UL * 1024UL);
    EXPECT_EQ(Resource::L0_BUFFER_NUM * Resource::GetL0BBytes(256UL, 128UL), 64UL * 1024UL);
    EXPECT_EQ(Resource::GetL0CBytes(256UL, 256UL), 256UL * 1024UL);
    EXPECT_EQ(Resource::GetEpilogueUbBytes(256UL, 256UL), 130UL * 1024UL);
}

TEST(GroupedA8W4BasicApiArch35Tiling, FixedTileRejectsInsufficientOnChipMemory)
{
    A8W4Case param;
    param.hasOffset = true;
    param.scaleGroupNum = 1;
    param.outDtype = ge::DT_FLOAT16;

    for (const int resource : {0, 1, 2, 3, 4}) {
        auto compileInfo = MakeCompileInfo();
        if (resource == 0) {
            compileInfo.l1Size = 260UL * 1024UL - 1UL;
        } else if (resource == 1) {
            compileInfo.l0ASize = 64UL * 1024UL - 1UL;
        } else if (resource == 2) {
            compileInfo.l0BSize = 64UL * 1024UL - 1UL;
        } else if (resource == 3) {
            compileInfo.l0CSize = 256UL * 1024UL - 1UL;
        } else {
            compileInfo.ubSize = 130UL * 1024UL - 1UL;
        }
        TilingInfo tilingInfo;
        EXPECT_FALSE(ExecuteA8W4TilingWithCompileInfo(param, compileInfo, tilingInfo))
            << "resource=" << resource;
    }
}

TEST(GroupedA8W4BasicApiArch35Tiling, AsymmetricPerchannelNzUsesNzTilingKey)
{
    A8W4Case param;
    param.k = 1001;
    param.n = 248;
    param.hasOffset = true;
    param.scaleGroupNum = 1;
    param.outDtype = ge::DT_FLOAT16;
    param.weightFormat = ge::FORMAT_FRACTAL_NZ;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, A8W4_NZ_PERCHANNEL_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    constexpr uint64_t expectedStride = 1008UL * 256UL;
    EXPECT_EQ(tilingData->a8w4Params.expandedWeightStrideBytes, expectedStride);
    const uint64_t expandedSize = AlignWorkspace(2UL * expectedStride);
    const uint64_t tileStride = AlignWorkspace(256UL * 256UL * sizeof(uint16_t));
    const uint64_t tileSize = 32UL * tileStride;
    const uint64_t rowSumSize = AlignWorkspace(static_cast<uint64_t>(param.m) * sizeof(float));
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE + expandedSize + tileSize + rowSumSize);
}

TEST(GroupedA8W4BasicApiArch35Tiling, PerchannelInt4AndPackedInt32ShareWorkspaceAndLogicalTiling)
{
    A8W4Case int4Param;
    int4Param.k = 1001;
    int4Param.scaleGroupNum = 1;
    int4Param.hasOffset = true;
    int4Param.outDtype = ge::DT_FLOAT16;
    A8W4Case packedParam = int4Param;
    packedParam.weightDtype = ge::DT_INT32;

    TilingInfo int4TilingInfo;
    TilingInfo packedTilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(int4Param, int4TilingInfo));
    ASSERT_TRUE(ExecuteA8W4Tiling(packedParam, packedTilingInfo));
    EXPECT_EQ(int4TilingInfo.tilingKey, packedTilingInfo.tilingKey);
    EXPECT_EQ(int4TilingInfo.workspaceSizes, packedTilingInfo.workspaceSizes);
    const auto *int4TilingData = GetTilingData(int4TilingInfo);
    const auto *packedTilingData = GetTilingData(packedTilingInfo);
    ASSERT_NE(int4TilingData, nullptr);
    ASSERT_NE(packedTilingData, nullptr);
    EXPECT_EQ(int4TilingData->a8w4Params.expandedWeightStrideBytes,
              packedTilingData->a8w4Params.expandedWeightStrideBytes);
    EXPECT_EQ(int4TilingData->mmTilingData.k, packedTilingData->mmTilingData.k);
    EXPECT_EQ(int4TilingData->mmTilingData.n, packedTilingData->mmTilingData.n);
}

TEST(GroupedA8W4BasicApiArch35Tiling, Float16WithoutOffsetStillUsesPergroup)
{
    A8W4Case param;
    param.outDtype = ge::DT_FLOAT16;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, GetExpectedA8W4TilingKey(param.weightFormat, WQGMM_PER_GROUP));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->a8w4Params.dequantMode, 0U);
    EXPECT_EQ(tilingData->a8w4Params.quantGroupNum, 4U);
}

TEST(GroupedA8W4BasicApiArch35Tiling, PackedInt32KeepsLogicalOriginShape)
{
    A8W4Case param;
    param.weightDtype = ge::DT_INT32;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, GetExpectedA8W4TilingKey(param.weightFormat, WQGMM_PER_GROUP));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->a8w4Params.weightPackedInt32, 1U);
    EXPECT_EQ(tilingData->mmTilingData.k, static_cast<uint32_t>(param.k));
    EXPECT_EQ(tilingData->mmTilingData.n, static_cast<uint32_t>(param.n));
}

TEST(GroupedA8W4BasicApiArch35Tiling, Int4AndPackedInt32ShareKeyAndLogicalTiling)
{
    A8W4Case int4Param;
    A8W4Case packedParam;
    packedParam.weightDtype = ge::DT_INT32;

    TilingInfo int4TilingInfo;
    TilingInfo packedTilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(int4Param, int4TilingInfo));
    ASSERT_TRUE(ExecuteA8W4Tiling(packedParam, packedTilingInfo));
    EXPECT_EQ(int4TilingInfo.tilingKey, packedTilingInfo.tilingKey);

    const auto *int4TilingData = GetTilingData(int4TilingInfo);
    const auto *packedTilingData = GetTilingData(packedTilingInfo);
    ASSERT_NE(int4TilingData, nullptr);
    ASSERT_NE(packedTilingData, nullptr);
    EXPECT_EQ(int4TilingData->mmTilingData.m, packedTilingData->mmTilingData.m);
    EXPECT_EQ(int4TilingData->mmTilingData.n, packedTilingData->mmTilingData.n);
    EXPECT_EQ(int4TilingData->mmTilingData.k, packedTilingData->mmTilingData.k);
    EXPECT_EQ(int4TilingData->a8w4Params.weightPackedInt32, 0U);
    EXPECT_EQ(packedTilingData->a8w4Params.weightPackedInt32, 1U);
}

TEST(GroupedA8W4BasicApiArch35Tiling, PackedInt32StorageIsEightToOne)
{
    A8W4Case param;
    param.weightDtype = ge::DT_INT32;
    auto compileInfo = MakeCompileInfo();
    const auto context = MakeContext(param, &compileInfo);
    const auto &weightShape = context.inputTensorDesc_[1].shape_;
    EXPECT_EQ(weightShape.GetOriginShape().GetDim(2), param.n);
    EXPECT_EQ(weightShape.GetStorageShape().GetDim(2), param.n / 8);
}

TEST(GroupedA8W4BasicApiArch35Tiling, PackedInt32NzStorageIsEightToOne)
{
    A8W4Case param;
    param.weightDtype = ge::DT_INT32;
    param.weightFormat = ge::FORMAT_FRACTAL_NZ;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, A8W4_NZ_PERGROUP_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->a8w4Params.weightPackedInt32, 1U);
}

TEST(GroupedA8W4BasicApiArch35Tiling, RejectsMalformedPackedInt32StorageShape)
{
    A8W4Case param;
    param.weightDtype = ge::DT_INT32;
    param.invalidPackedStorageShape = true;

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteA8W4Tiling(param, tilingInfo));
}

TEST(GroupedA8W4BasicApiArch35Tiling, SpecialEnkNzWeightIsRecorded)
{
    A8W4Case param;
    param.k = 2048;
    param.n = 2048;
    param.scaleGroupNum = param.k / 256;
    param.weightFormat = ge::FORMAT_FRACTAL_NZ;
    param.specialWeightFormat = true;
    param.tuningConfig = {0, 1};

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, GetExpectedA8W4TilingKey(param.weightFormat, WQGMM_PER_GROUP));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->a8w4Params.specialWeightFormat, 1U);
    EXPECT_EQ(tilingData->mmTilingData.k, 2048U);
    EXPECT_EQ(tilingData->mmTilingData.n, 2048U);
}

TEST(GroupedA8W4BasicApiArch35Tiling, PerchannelSpecialEnkUsesStandardW8NzWorkspaceStride)
{
    A8W4Case param;
    param.k = 2050;
    param.n = 2040;
    param.scaleGroupNum = 1;
    param.hasOffset = true;
    param.outDtype = ge::DT_FLOAT16;
    param.weightFormat = ge::FORMAT_FRACTAL_NZ;
    param.specialWeightFormat = true;
    param.tuningConfig = {0, 1};

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, A8W4_NZ_PERCHANNEL_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->a8w4Params.specialWeightFormat, 1U);
    constexpr uint64_t expectedStandardNzStride = 2064UL * 2048UL;
    EXPECT_EQ(tilingData->a8w4Params.expandedWeightStrideBytes, expectedStandardNzStride);
    const uint64_t expandedSize = AlignWorkspace(2UL * expectedStandardNzStride);
    const uint64_t tileStride = AlignWorkspace(256UL * 256UL * sizeof(uint16_t));
    const uint64_t tileSize = 32UL * tileStride;
    const uint64_t rowSumSize = AlignWorkspace(static_cast<uint64_t>(param.m) * sizeof(float));
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE + expandedSize + tileSize + rowSumSize);
}

TEST(GroupedA8W4BasicApiArch35Tiling, ExpectedTokensDoNotEnableUnimplementedWeightPreprocess)
{
    A8W4Case param;
    param.tuningConfig = {128}; // 128 > N/4(64)

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->a8w4Params.expectedTokenNum, 128U);
    EXPECT_EQ(tilingData->a8w4Params.enableWeightPreprocess, 0U);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], SYS_WORKSPACE_SIZE);
}

TEST(GroupedA8W4BasicApiArch35Tiling, RejectsWrongPergroupScaleShape)
{
    A8W4Case param;
    param.scaleGroupNum = 1;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteA8W4Tiling(param, tilingInfo));
}

TEST(GroupedA8W4BasicApiArch35Tiling, RejectsBf16WithOffset)
{
    A8W4Case param;
    param.hasOffset = true;
    param.scaleGroupNum = 1;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteA8W4Tiling(param, tilingInfo));
}

TEST(GroupedA8W4BasicApiArch35Tiling, RejectsKNotDivisibleBy256)
{
    A8W4Case param;
    param.k = 1152;
    param.scaleGroupNum = 4;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteA8W4Tiling(param, tilingInfo));
}

TEST(GroupedA8W4BasicApiArch35Tiling, AsymmetricPerchannelAcceptsUnalignedLogicalK)
{
    A8W4Case param;
    param.k = 1001;
    param.scaleGroupNum = 1;
    param.hasOffset = true;
    param.outDtype = ge::DT_FLOAT16;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, A8W4_ND_PERCHANNEL_TILING_KEY);
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->mmTilingData.k, 1001U);
    EXPECT_EQ(tilingData->a8w4Params.quantGroupNum, 1U);
}

TEST(GroupedA8W4BasicApiArch35Tiling, AsymmetricPerchannelPadsSmallKTilingWindow)
{
    A8W4Case param;
    param.k = 127;
    param.scaleGroupNum = 1;
    param.hasOffset = true;
    param.outDtype = ge::DT_FLOAT16;

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteA8W4Tiling(param, tilingInfo));
    const auto *tilingData = GetTilingData(tilingInfo);
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->mmTilingData.k, 127U);
    EXPECT_EQ(tilingData->mmTilingData.kAL1, 256U);
    EXPECT_EQ(tilingData->mmTilingData.kBL1, 256U);
}

TEST(GroupedA8W4BasicApiArch35Tiling, RejectsMissingBias)
{
    A8W4Case param;
    param.hasBias = false;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteA8W4Tiling(param, tilingInfo));
}

TEST(GroupedA8W4BasicApiArch35Tiling, RejectsNonEmptyAntiquantScale)
{
    A8W4Case param;
    param.hasAntiquantScale = true;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteA8W4Tiling(param, tilingInfo));
}

TEST(GroupedA8W4BasicApiArch35Tiling, RejectsNonEmptyAntiquantOffset)
{
    A8W4Case param;
    param.hasAntiquantOffset = true;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteA8W4Tiling(param, tilingInfo));
}

TEST(GroupedA8W4BasicApiArch35Tiling, RejectsSpecialWeightWithoutNzConversion)
{
    A8W4Case param;
    param.specialWeightFormat = true;
    param.tuningConfig = {0, 1};
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteA8W4Tiling(param, tilingInfo));
}
} // namespace
