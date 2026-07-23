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

#include "../../../../op_host/checkers/paged_attention_checker.h"
#include "../../../../op_host/fused_infer_attention_score_tiling_info_parser.h"

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

optiling::FiaTilingInfo MakePagedInfo(NpuArch npuArch, optiling::FiaLayout kvLayout, int32_t keyDim,
                                     int32_t valueDim)
{
    optiling::FiaTilingInfo fiaInfo;
    fiaInfo.opName = "FiaKvStrideTest";
    fiaInfo.npuArch = npuArch;
    fiaInfo.socVersion = npuArch == NpuArch::DAV_3510 ? platform_ascendc::SocVersion::ASCEND950
                                                       : platform_ascendc::SocVersion::ASCEND910B;
    fiaInfo.kvLayout = kvLayout;
    fiaInfo.kvStorageMode = optiling::KvStorageMode::PAGE_ATTENTION;
    fiaInfo.pageAttentionFlag = true;
    fiaInfo.hasViewStride = true;
    fiaInfo.blockSize = 128;
    fiaInfo.qkHeadDim = 128;
    fiaInfo.s1Size = 1;
    fiaInfo.keyNonContigDim = keyDim;
    fiaInfo.valueNonContigDim = valueDim;
    fiaInfo.keyBnStride = keyDim == 0 ? 8192 : 4096;
    fiaInfo.valueBnStride = valueDim == 0 ? 6144 : 4096;
    fiaInfo.keyN2Stride = keyDim == 1 ? 4096 : 2048;
    fiaInfo.valueN2Stride = valueDim == 1 ? 3072 : 2048;
    return fiaInfo;
}

TEST(FiaKvStrideParser, PrecomputesIndependentKeyAndValueDim0Strides)
{
    gert::StorageShape keyShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::StorageShape valueShape({8, 2, 16, 64}, {8, 2, 16, 64});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {8192, 2048, 128, 1});
    SetStride(valueStride, {6144, 1024, 64, 1});

    optiling::FiaInfoParser parser(nullptr);
    parser.hasViewStride_ = true;
    parser.kvStorageMode_ = optiling::KvStorageMode::PAGE_ATTENTION;
    parser.kvLayout_ = optiling::FiaLayout::BnNBsD;
    parser.kCache_ = {&keyShape};
    parser.vCache_ = {&valueShape};
    parser.kStrideCache_ = {&keyStride};
    parser.vStrideCache_ = {&valueStride};
    parser.keyStrides_ = &keyStride;
    parser.valueStrides_ = &valueStride;

    parser.GetKvIsContiguous();
    ASSERT_EQ(parser.GetKvStrideValues(), ge::GRAPH_SUCCESS);

    EXPECT_EQ(parser.keyNonContigDim_, 0);
    EXPECT_EQ(parser.valueNonContigDim_, 0);
    EXPECT_EQ(parser.keyBnStride_, 8192U);
    EXPECT_EQ(parser.valueBnStride_, 6144U);
    EXPECT_EQ(parser.keyN2Stride_, 2048U);
    EXPECT_EQ(parser.valueN2Stride_, 1024U);
}

TEST(FiaKvStrideParser, DistinguishesKeyOnlyViewAndDim1Stride)
{
    gert::StorageShape keyShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::StorageShape valueShape({8, 2, 16, 128}, {8, 2, 16, 128});
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {8192, 4096, 128, 1});
    SetStride(valueStride, {4096, 2048, 128, 1});

    optiling::FiaInfoParser parser(nullptr);
    parser.hasViewStride_ = true;
    parser.kvStorageMode_ = optiling::KvStorageMode::PAGE_ATTENTION;
    parser.kvLayout_ = optiling::FiaLayout::BnNBsD;
    parser.kCache_ = {&keyShape};
    parser.vCache_ = {&valueShape};
    parser.kStrideCache_ = {&keyStride};
    parser.vStrideCache_ = {&valueStride};
    parser.keyStrides_ = &keyStride;
    parser.valueStrides_ = &valueStride;

    parser.GetKvIsContiguous();
    ASSERT_EQ(parser.GetKvStrideValues(), ge::GRAPH_SUCCESS);

    EXPECT_EQ(parser.keyNonContigDim_, 1);
    EXPECT_EQ(parser.valueNonContigDim_, -1);
    EXPECT_EQ(parser.keyBnStride_, 8192U);
    EXPECT_EQ(parser.valueBnStride_, 4096U);
}

TEST(FiaKvStrideParser, ExtractsKeyRopeDim0Stride)
{
    gert::StorageShape keyShape({8, 1, 16, 512}, {8, 1, 16, 512});
    gert::StorageShape valueShape({8, 1, 16, 512}, {8, 1, 16, 512});
    gert::Stride keyStride;
    gert::Stride valueStride;
    gert::Stride keyRopeStride;
    SetStride(keyStride, {8192, 8192, 512, 1});
    SetStride(valueStride, {8192, 8192, 512, 1});
    SetStride(keyRopeStride, {2048, 1024, 64, 1});

    optiling::FiaInfoParser parser(nullptr);
    parser.hasViewStride_ = true;
    parser.kvStorageMode_ = optiling::KvStorageMode::PAGE_ATTENTION;
    parser.kvLayout_ = optiling::FiaLayout::BnNBsD;
    parser.kCache_ = {&keyShape};
    parser.vCache_ = {&valueShape};
    parser.kStrideCache_ = {&keyStride};
    parser.vStrideCache_ = {&valueStride};
    parser.keyStrides_ = &keyStride;
    parser.valueStrides_ = &valueStride;
    parser.kRopeStrides_ = &keyRopeStride;

    ASSERT_EQ(parser.GetKvStrideValues(), ge::GRAPH_SUCCESS);

    EXPECT_EQ(parser.kRopeBnStride_, 2048U);
    EXPECT_EQ(parser.kRopeN2Stride_, 1024U);
}

TEST(FiaKvStrideParser, DetectsNonContiguousStrideInLaterTensorListElement)
{
    gert::StorageShape firstShape({1, 2, 16, 128}, {1, 2, 16, 128});
    gert::StorageShape secondShape({1, 2, 16, 128}, {1, 2, 16, 128});
    gert::Stride firstStride;
    gert::Stride secondStride;
    SetStride(firstStride, {4096, 2048, 128, 1});
    SetStride(secondStride, {8192, 4096, 128, 1});

    optiling::FiaInfoParser parser(nullptr);
    parser.hasViewStride_ = true;
    parser.kCache_ = {&firstShape, &secondShape};
    parser.vCache_ = {&firstShape, &firstShape};
    parser.kStrideCache_ = {&firstStride, &secondStride};
    parser.vStrideCache_ = {&firstStride, &firstStride};

    parser.GetKvIsContiguous();

    EXPECT_EQ(parser.keyNonContigDim_, 1);
    EXPECT_EQ(parser.valueNonContigDim_, -1);
}

TEST(FiaKvStrideParser, RejectsStrideRankTooSmallForBnNBsD)
{
    gert::Stride keyStride;
    gert::Stride valueStride;
    SetStride(keyStride, {8192, 4096, 128, 1});
    SetStride(valueStride, {4096});

    optiling::FiaInfoParser parser(nullptr);
    parser.opName_ = "FiaKvStrideTest";
    parser.hasViewStride_ = true;
    parser.kvStorageMode_ = optiling::KvStorageMode::PAGE_ATTENTION;
    parser.kvLayout_ = optiling::FiaLayout::BnNBsD;
    parser.keyStrides_ = &keyStride;
    parser.valueStrides_ = &valueStride;

    EXPECT_EQ(parser.GetKvStrideValues(), ge::GRAPH_FAILED);
}

TEST(FiaKvStrideChecker, Arch22AcceptsDim0ForBnNBsD)
{
    auto fiaInfo = MakePagedInfo(NpuArch::DAV_2201, optiling::FiaLayout::BnNBsD, 0, 0);
    optiling::PagedAttentionChecker checker(true, false, false);

    EXPECT_EQ(checker.CheckMultiParaConsistency(fiaInfo), ge::GRAPH_SUCCESS);
}

TEST(FiaKvStrideChecker, Arch22AcceptsDim0ForBnBsHAndNz)
{
    optiling::PagedAttentionChecker checker(true, false, false);
    auto bnBsHInfo = MakePagedInfo(NpuArch::DAV_2201, optiling::FiaLayout::BnBsH, 0, 0);
    auto nzInfo = MakePagedInfo(NpuArch::DAV_2201, optiling::FiaLayout::NZ, 0, 0);

    EXPECT_EQ(checker.CheckMultiParaConsistency(bnBsHInfo), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckMultiParaConsistency(nzInfo), ge::GRAPH_SUCCESS);
}

TEST(FiaKvStrideChecker, Arch22AcceptsDim0KeyRopeForSplitMla)
{
    optiling::PagedAttentionChecker checker(true, false, false);
    auto keyRopeInfo = MakePagedInfo(NpuArch::DAV_2201, optiling::FiaLayout::BnNBsD, -1, -1);
    keyRopeInfo.keyRopeNonContigDim = 0;
    keyRopeInfo.kRopeBnStride = 8192;

    keyRopeInfo.mlaMode = optiling::MlaMode::ROPE_SPLIT_D128;
    EXPECT_EQ(checker.CheckMultiParaConsistency(keyRopeInfo), ge::GRAPH_SUCCESS);
    keyRopeInfo.mlaMode = optiling::MlaMode::ROPE_SPLIT_D512;
    EXPECT_EQ(checker.CheckMultiParaConsistency(keyRopeInfo), ge::GRAPH_SUCCESS);
}

TEST(FiaKvStrideChecker, Arch22RejectsDim1AndKeyRopeOutsideSplitMla)
{
    optiling::PagedAttentionChecker checker(true, false, false);
    auto dim1Info = MakePagedInfo(NpuArch::DAV_2201, optiling::FiaLayout::BnNBsD, 1, -1);
    EXPECT_EQ(checker.CheckMultiParaConsistency(dim1Info), ge::GRAPH_FAILED);

    auto keyRopeDim1Info = MakePagedInfo(NpuArch::DAV_2201, optiling::FiaLayout::BnNBsD, -1, -1);
    keyRopeDim1Info.mlaMode = optiling::MlaMode::ROPE_SPLIT_D512;
    keyRopeDim1Info.keyRopeNonContigDim = 1;
    keyRopeDim1Info.kRopeN2Stride = 4096;
    EXPECT_EQ(checker.CheckMultiParaConsistency(keyRopeDim1Info), ge::GRAPH_FAILED);

    auto nonMlaInfo = MakePagedInfo(NpuArch::DAV_2201, optiling::FiaLayout::BnNBsD, -1, -1);
    nonMlaInfo.keyRopeNonContigDim = 0;
    nonMlaInfo.kRopeBnStride = 8192;
    EXPECT_EQ(checker.CheckMultiParaConsistency(nonMlaInfo), ge::GRAPH_FAILED);
}

TEST(FiaKvStrideChecker, Arch35AcceptsDim1ForBnNBsDAndNz)
{
    optiling::PagedAttentionChecker checker(true, false, false);
    auto bnNBsDInfo = MakePagedInfo(NpuArch::DAV_3510, optiling::FiaLayout::BnNBsD, 1, 1);
    auto nzInfo = MakePagedInfo(NpuArch::DAV_3510, optiling::FiaLayout::NZ, 1, 1);

    EXPECT_EQ(checker.CheckMultiParaConsistency(bnNBsDInfo), ge::GRAPH_SUCCESS);
    EXPECT_EQ(checker.CheckMultiParaConsistency(nzInfo), ge::GRAPH_SUCCESS);
}

TEST(FiaKvStrideChecker, Arch35RejectsDim1ForBnBsH)
{
    auto fiaInfo = MakePagedInfo(NpuArch::DAV_3510, optiling::FiaLayout::BnBsH, 1, -1);
    optiling::PagedAttentionChecker checker(true, false, false);

    EXPECT_EQ(checker.CheckMultiParaConsistency(fiaInfo), ge::GRAPH_FAILED);
}

TEST(FiaKvStrideChecker, QuantizedTemplatesUseExplicitStrideWhitelist)
{
    auto unsupported = MakePagedInfo(NpuArch::DAV_3510, optiling::FiaLayout::BnNBsD, 0, 0);
    unsupported.fullQuantMode = optiling::FiaFullQuantMode::QKV_PER_TENSOR_FULL_QUANT;
    optiling::PagedAttentionChecker fullQuantChecker(false, true, false);
    EXPECT_EQ(fullQuantChecker.CheckMultiParaConsistency(unsupported), ge::GRAPH_FAILED);

    auto supported = MakePagedInfo(NpuArch::DAV_3510, optiling::FiaLayout::BnNBsD, 1, 1);
    supported.fullQuantMode = optiling::FiaFullQuantMode::QK_PER_TOKEN_HEAD_V_PER_HEAD;
    EXPECT_EQ(fullQuantChecker.CheckMultiParaConsistency(supported), ge::GRAPH_SUCCESS);

    auto mxfp8 = MakePagedInfo(NpuArch::DAV_3510, optiling::FiaLayout::BnNBsD, 1, 1);
    mxfp8.fullQuantMode = optiling::FiaFullQuantMode::QKV_MXFP8_FULL_QUANT;
    mxfp8.blockSize = 512;
    EXPECT_EQ(fullQuantChecker.CheckMultiParaConsistency(mxfp8), ge::GRAPH_SUCCESS);

    auto antiquant = MakePagedInfo(NpuArch::DAV_3510, optiling::FiaLayout::BnNBsD, 0, 0);
    optiling::PagedAttentionChecker antiquantChecker(false, false, true);
    EXPECT_EQ(antiquantChecker.CheckMultiParaConsistency(antiquant), ge::GRAPH_FAILED);
}

TEST(FiaKvStrideChecker, RejectsNonPaAndZeroForwardedStride)
{
    optiling::PagedAttentionChecker checker(true, false, false);
    auto nonPaInfo = MakePagedInfo(NpuArch::DAV_2201, optiling::FiaLayout::BNSD, 0, 0);
    nonPaInfo.kvStorageMode = optiling::KvStorageMode::BATCH_CONTINUOUS;
    nonPaInfo.pageAttentionFlag = false;
    EXPECT_EQ(checker.CheckMultiParaConsistency(nonPaInfo), ge::GRAPH_FAILED);

    auto zeroStrideInfo = MakePagedInfo(NpuArch::DAV_2201, optiling::FiaLayout::BnNBsD, 0, -1);
    zeroStrideInfo.keyBnStride = 0;
    EXPECT_EQ(checker.CheckMultiParaConsistency(zeroStrideInfo), ge::GRAPH_FAILED);
}

} // namespace
} // namespace FusedInferAttentionScoreUT
