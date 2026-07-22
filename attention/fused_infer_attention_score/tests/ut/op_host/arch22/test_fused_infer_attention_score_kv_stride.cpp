/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>

#include "../../../../op_host/checkers/common_checker.h"

namespace FusedInferAttentionScoreUT {
namespace {

void SetStride(gert::Stride &stride, std::initializer_list<int64_t> values)
{
    stride.SetDimNum(static_cast<uint32_t>(values.size()));
    uint32_t index = 0;
    for (int64_t value : values) {
        stride.SetStride(index++, value);
    }
}

optiling::FiaTilingInfo MakeFiaInfo(gert::StorageShape &keyShape, gert::StorageShape &valueShape,
                                   gert::Stride &keyStride, gert::Stride &valueStride, NpuArch npuArch,
                                   optiling::FiaLayout kvLayout, bool pageAttention = true)
{
    optiling::FiaTilingInfo fiaInfo;
    fiaInfo.opName = "FiaKvStrideTest";
    fiaInfo.opParamInfo.key.shape = &keyShape;
    fiaInfo.opParamInfo.value.shape = &valueShape;
    fiaInfo.keyStrides = &keyStride;
    fiaInfo.valueStrides = &valueStride;
    fiaInfo.npuArch = npuArch;
    fiaInfo.kvLayout = kvLayout;
    fiaInfo.pageAttentionFlag = pageAttention;
    fiaInfo.kvStorageMode = pageAttention ? optiling::KvStorageMode::PAGE_ATTENTION
                                          : optiling::KvStorageMode::BATCH_CONTINUOUS;
    return fiaInfo;
}

TEST(FiaKvStrideChecker, Arch22AcceptsIndependentDim0Strides)
{
    gert::StorageShape keyShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::StorageShape valueShape({8, 2, 16, 64}, {8, 2, 16, 64});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {8192, 2048, 128, 1});
    SetStride(valueStride, {6144, 1024, 64, 1});
    auto fiaInfo = MakeFiaInfo(keyShape, valueShape, keyStride, valueStride, NpuArch::DAV_2201,
                               optiling::FiaLayout::BnNBsD);
    optiling::CommonChecker checker(true, false, false);

    EXPECT_EQ(checker.CheckKVContiguous(fiaInfo), ge::GRAPH_SUCCESS);
}

TEST(FiaKvStrideChecker, Arch22RejectsDim1Stride)
{
    gert::StorageShape keyShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::StorageShape valueShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {8192, 4096, 128, 1});
    SetStride(valueStride, {4096, 2048, 128, 1});
    auto fiaInfo = MakeFiaInfo(keyShape, valueShape, keyStride, valueStride, NpuArch::DAV_2201,
                               optiling::FiaLayout::BnNBsD);
    optiling::CommonChecker checker(true, false, false);

    EXPECT_EQ(checker.CheckKVContiguous(fiaInfo), ge::GRAPH_FAILED);
}

TEST(FiaKvStrideChecker, Arch35KeepsDim1Support)
{
    gert::StorageShape keyShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::StorageShape valueShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {8192, 4096, 128, 1});
    SetStride(valueStride, {4096, 2048, 128, 1});
    auto fiaInfo = MakeFiaInfo(keyShape, valueShape, keyStride, valueStride, NpuArch::DAV_3510,
                               optiling::FiaLayout::BnNBsD);
    optiling::CommonChecker checker(true, false, false);

    EXPECT_EQ(checker.CheckKVContiguous(fiaInfo), ge::GRAPH_SUCCESS);
}

TEST(FiaKvStrideChecker, RejectsBroadcastDim0Stride)
{
    gert::StorageShape keyShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::StorageShape valueShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {0, 2048, 128, 1});
    SetStride(valueStride, {4096, 2048, 128, 1});
    auto fiaInfo = MakeFiaInfo(keyShape, valueShape, keyStride, valueStride, NpuArch::DAV_2201,
                               optiling::FiaLayout::BnNBsD);
    optiling::CommonChecker checker(true, false, false);

    EXPECT_EQ(checker.CheckKVContiguous(fiaInfo), ge::GRAPH_FAILED);
}

TEST(FiaKvStrideChecker, Arch35RejectsBroadcastDim1Stride)
{
    gert::StorageShape keyShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::StorageShape valueShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {2048, 0, 128, 1});
    SetStride(valueStride, {4096, 2048, 128, 1});
    auto fiaInfo = MakeFiaInfo(keyShape, valueShape, keyStride, valueStride, NpuArch::DAV_3510,
                               optiling::FiaLayout::BnNBsD);
    optiling::CommonChecker checker(true, false, false);

    EXPECT_EQ(checker.CheckKVContiguous(fiaInfo), ge::GRAPH_FAILED);
}

TEST(FiaKvStrideChecker, NonPaStillRejectsNonContiguousCache)
{
    gert::StorageShape keyShape({2, 2, 16, 128}, {2, 2, 16, 128});
    gert::StorageShape valueShape({2, 2, 16, 128}, {2, 2, 16, 128});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {8192, 2048, 128, 1});
    SetStride(valueStride, {4096, 2048, 128, 1});
    auto fiaInfo = MakeFiaInfo(keyShape, valueShape, keyStride, valueStride, NpuArch::DAV_2201,
                               optiling::FiaLayout::BNSD, false);
    optiling::CommonChecker checker(true, false, false);

    EXPECT_EQ(checker.CheckKVContiguous(fiaInfo), ge::GRAPH_FAILED);
}

TEST(FiaKvStrideChecker, UnsupportedFullQuantRejectsNonContiguousCache)
{
    gert::StorageShape keyShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::StorageShape valueShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {8192, 2048, 128, 1});
    SetStride(valueStride, {6144, 2048, 128, 1});
    auto fiaInfo = MakeFiaInfo(keyShape, valueShape, keyStride, valueStride, NpuArch::DAV_3510,
                               optiling::FiaLayout::BnNBsD);
    fiaInfo.fullQuantMode = optiling::FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT;
    optiling::CommonChecker checker(false, true, false);

    EXPECT_EQ(checker.CheckKVContiguousForQuantMode(fiaInfo), ge::GRAPH_FAILED);
}

TEST(FiaKvStrideChecker, AntiquantRejectsNonContiguousCache)
{
    gert::StorageShape keyShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::StorageShape valueShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {8192, 2048, 128, 1});
    SetStride(valueStride, {4096, 2048, 128, 1});
    auto fiaInfo = MakeFiaInfo(keyShape, valueShape, keyStride, valueStride, NpuArch::DAV_3510,
                               optiling::FiaLayout::BnNBsD);
    optiling::CommonChecker checker(false, false, true);

    EXPECT_EQ(checker.CheckKVContiguousForQuantMode(fiaInfo), ge::GRAPH_FAILED);
}

TEST(FiaKvStrideChecker, Arch35WhitelistedFullQuantKeepsDim0AndDim1Support)
{
    gert::StorageShape keyShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::StorageShape valueShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {8192, 4096, 128, 1});
    SetStride(valueStride, {6144, 3072, 128, 1});
    auto fiaInfo = MakeFiaInfo(keyShape, valueShape, keyStride, valueStride, NpuArch::DAV_3510,
                               optiling::FiaLayout::BnNBsD);
    fiaInfo.fullQuantMode = optiling::FiaFullQuantMode::QK_PER_TOKEN_HEAD_V_PER_HEAD;
    optiling::CommonChecker checker(false, true, false);

    EXPECT_EQ(checker.CheckKVContiguousForQuantMode(fiaInfo), ge::GRAPH_SUCCESS);
}

TEST(FiaKvStrideChecker, Arch35WhitelistedFullQuantRejectsInnerStride)
{
    gert::StorageShape keyShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::StorageShape valueShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {8192, 4096, 256, 1});
    SetStride(valueStride, {4096, 2048, 128, 1});
    auto fiaInfo = MakeFiaInfo(keyShape, valueShape, keyStride, valueStride, NpuArch::DAV_3510,
                               optiling::FiaLayout::BnNBsD);
    fiaInfo.fullQuantMode = optiling::FiaFullQuantMode::QKV_MXFP8_FULL_QUANT;
    optiling::CommonChecker checker(false, true, false);

    EXPECT_EQ(checker.CheckKVContiguousForQuantMode(fiaInfo), ge::GRAPH_FAILED);
}

} // namespace
} // namespace FusedInferAttentionScoreUT
