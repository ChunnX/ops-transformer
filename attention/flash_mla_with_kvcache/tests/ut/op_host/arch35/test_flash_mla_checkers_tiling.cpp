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
#include "../../../../op_host/checkers/common_checker.h"
#include "../../../../op_host/checkers/flash_mla_with_kvcache_checker.h"
#include "../../../../op_host/checkers/mask_checker.h"
#include "../../../../op_host/checkers/paged_attention_checker.h"
#include "../../../../op_host/checkers/sinks_checker.h"
#include "../../../../op_host/checkers/softmax_lse_checker.h"
#include "../../../../op_host/flash_mla_with_kvcache_tiling_info_parser.h"

namespace FlashMlaWithKvcacheUT {
using namespace optiling::flash_mla_with_kvcache;
using Layout = FlashMlaWithKvcacheLayout;

// A complete valid TND -> NTD, PA_NZ input. Each negative case changes one rule.
class FlashMlaCheckerTest : public testing::Test {
protected:
    FlashMlaWithKvcacheTilingInfo info{};
    gert::StorageShape q{{8, 64, 576}, {8, 64, 576}};
    gert::StorageShape k{{1, 1, 36, 128, 16}, {1, 1, 36, 128, 16}};
    gert::StorageShape out{{64, 8, 512}, {64, 8, 512}};
    gert::StorageShape lse{{64, 8}, {64, 8}};
    gert::CompileTimeTensorDesc halfDesc, intDesc, floatDesc, maskDesc;
    gert::Tensor table, cache, cu, metadata, mask, sink;
    int64_t headDimV = 512;
    int64_t returnLse = 1;

    void SetUp() override
    {
        info.opName = "FlashMlaCheckerTest";
        info.qLayout = Layout::TND;
        info.kvLayout = Layout::PA_NZ;
        info.outLayout = Layout::NTD;
        info.bSize = 2;
        info.n1Size = 64;
        info.n2Size = 1;
        info.qTSize = info.s1Size = 8;
        info.s2Size = info.blockSize = 128;
        info.totalBlockNum = 1;
        info.qkHeadDim = info.vHeadDim = 512;
        info.pageAttentionFlag = true;
        info.maxSeqQ = info.maxSeqKv = -1;
        info.maskMode = 0;
        info.winLeft = info.winRight = -1;
        info.softmaxLseFlag = true;
        info.opParamInfo.headDimV = &headDimV;
        info.opParamInfo.returnSoftMaxLse = &returnLse;
        for (auto *desc : {&halfDesc, &intDesc, &floatDesc, &maskDesc}) {
            desc->SetOriginFormat(ge::FORMAT_ND);
        }
        halfDesc.SetDataType(ge::DT_FLOAT16);
        intDesc.SetDataType(ge::DT_INT32);
        floatDesc.SetDataType(ge::DT_FLOAT);
        maskDesc.SetDataType(ge::DT_INT8);
        info.opParamInfo.query.shape = &q;
        info.opParamInfo.query.desc = &halfDesc;
        info.opParamInfo.kCache.shape = &k;
        info.opParamInfo.kCache.desc = &halfDesc;
        info.opParamInfo.attnOut.shape = &out;
        info.opParamInfo.attnOut.desc = &halfDesc;
        info.opParamInfo.lseOut.shape = &lse;
        info.opParamInfo.lseOut.desc = &floatDesc;
        table.MutableStorageShape() = gert::Shape({2, 1});
        cache.MutableStorageShape() = gert::Shape({2});
        cu.MutableStorageShape() = gert::Shape({3});
        metadata.MutableStorageShape() = gert::Shape({4096});
        mask.MutableStorageShape() = gert::Shape({2048, 2048});
        sink.MutableStorageShape() = gert::Shape({64});
        info.opParamInfo.blockTable.tensor = &table;
        info.opParamInfo.blockTable.desc = &intDesc;
        info.opParamInfo.cacheSeqlens.tensor = &cache;
        info.opParamInfo.cacheSeqlens.desc = &intDesc;
        info.opParamInfo.cuSeqlensQ.tensor = &cu;
        info.opParamInfo.cuSeqlensQ.desc = &intDesc;
        info.opParamInfo.metadata.tensor = &metadata;
        info.opParamInfo.metadata.desc = &intDesc;
        info.opParamInfo.attnMask.desc = &maskDesc;
        info.opParamInfo.sinks.desc = &floatDesc;
    }

    ge::graphStatus Check()
    {
        FlashMlaWithKvcacheChecker checker;
        checker.Init(info);
        return checker.Process(info);
    }
};

TEST_F(FlashMlaCheckerTest, ValidComposite)
{
    EXPECT_EQ(Check(), ge::GRAPH_SUCCESS);
}

TEST_F(FlashMlaCheckerTest, MissingRequiredDescriptors)
{
    for (auto *param : {&info.opParamInfo.query, &info.opParamInfo.kCache, &info.opParamInfo.attnOut}) {
        auto *saved = param->desc;
        param->desc = nullptr;
        EXPECT_EQ(Check(), ge::GRAPH_FAILED);
        param->desc = saved;
    }
}

TEST_F(FlashMlaCheckerTest, EmptyTensorFlag)
{
    info.emptyTensorFlag = true;
    EXPECT_EQ(Check(), ge::GRAPH_FAILED);
}

TEST_F(FlashMlaCheckerTest, LayoutCompatibilityMatrix)
{
    CommonChecker checker;
    for (auto query : {Layout::BSND, Layout::BNSD, Layout::TND, Layout::NTD}) {
        for (auto kv : {Layout::PA_BNBD, Layout::PA_BBND, Layout::PA_NZ, Layout::BNSD}) {
            for (auto output : {Layout::BSND, Layout::BNSD, Layout::TND, Layout::NTD}) {
                SCOPED_TRACE(testing::Message() << "q=" << int(query) << " kv=" << int(kv) << " out=" << int(output));
                info.qLayout = query;
                info.kvLayout = kv;
                info.outLayout = output;
                const bool expected = query != Layout::NTD && kv != Layout::BNSD &&
                                      (output == query || (query == Layout::TND && output == Layout::NTD));
                EXPECT_EQ(checker.CheckSinglePara(info) == ge::GRAPH_SUCCESS &&
                              checker.CheckMultiParaLayout(info) == ge::GRAPH_SUCCESS,
                          expected);
            }
        }
    }
}

class FlashMlaHeadTest : public FlashMlaCheckerTest, public testing::WithParamInterface<int> {};
TEST_P(FlashMlaHeadTest, QueryAndSinkHeads)
{
    const int heads = GetParam();
    info.n1Size = heads;
    CommonChecker common;
    EXPECT_EQ(common.CheckAxis(info) == ge::GRAPH_SUCCESS, heads >= 1 && heads <= 128);
    if (heads < 1 || heads > 128) {
        return;
    }
    SinksChecker sinks;
    info.opParamInfo.sinks.tensor = &sink;
    sink.MutableStorageShape() = gert::Shape({heads});
    EXPECT_EQ(sinks.CheckSinglePara(info), ge::GRAPH_SUCCESS);
    sink.MutableStorageShape() = gert::Shape({heads + 1});
    EXPECT_EQ(sinks.CheckSinglePara(info), ge::GRAPH_FAILED);
}
INSTANTIATE_TEST_SUITE_P(Bounds, FlashMlaHeadTest, testing::Range(-1, 130));

TEST_F(FlashMlaCheckerTest, InvalidKvHeadsBeforeDivision)
{
    for (int heads : {-1, 0, 2}) {
        SCOPED_TRACE(heads);
        info.n2Size = heads;
        EXPECT_EQ(Check(), ge::GRAPH_FAILED);
    }
}

class FlashMlaBlockSizeTest : public FlashMlaCheckerTest, public testing::WithParamInterface<int> {};
TEST_P(FlashMlaBlockSizeTest, RangeAndAlignment)
{
    const int size = GetParam();
    info.blockSize = size;
    PagedAttentionChecker checker;
    EXPECT_EQ(checker.CheckSinglePara(info) == ge::GRAPH_SUCCESS, size >= 16 && size <= 1024 && size % 16 == 0);
}
INSTANTIATE_TEST_SUITE_P(Bounds, FlashMlaBlockSizeTest, testing::Range(0, 1041));

TEST_F(FlashMlaCheckerTest, RejectEveryZeroQueryAndCacheAxis)
{
    FlashMlaWithKvcacheInfoParser parser(nullptr);
    parser.opName_ = "FlashMlaEmptyTensorTest";
    parser.opParamInfo_.query.shape = &q;
    parser.opParamInfo_.kCache.shape = &k;
    const std::vector<std::vector<gert::Shape>> shapes = {{{8, 64, 576}, {2, 4, 64, 576}, {2, 64, 4, 576}},
                                                          {{1, 128, 1, 576}, {1, 1, 128, 576}, {1, 1, 36, 128, 16}}};
    size_t input = 0;
    for (auto *tensor : {&q, &k}) {
        for (const auto &shape : shapes[input]) {
            for (size_t axis = 0; axis < shape.GetDimNum(); ++axis) {
                SCOPED_TRACE(testing::Message()
                             << "input=" << input << " rank=" << shape.GetDimNum() << " axis=" << axis);
                tensor->MutableStorageShape() = shape;
                tensor->MutableStorageShape().SetDim(axis, 0);
                parser.emptyTensorFlag_ = false;
                EXPECT_EQ(parser.GetEmptyTensorFlag(), ge::GRAPH_FAILED);
                EXPECT_TRUE(parser.emptyTensorFlag_);
                tensor->MutableStorageShape() = shape;
            }
        }
        ++input;
    }
    parser.emptyTensorFlag_ = false;
    EXPECT_EQ(parser.GetEmptyTensorFlag(), ge::GRAPH_SUCCESS);
}

TEST_F(FlashMlaCheckerTest, MaskModesAndPresence)
{
    for (int mode : {-1, 0, 1, 2, 3, 4, 5}) {
        for (bool present : {false, true}) {
            SCOPED_TRACE(testing::Message() << "mode=" << mode << " present=" << present);
            info.maskMode = mode;
            info.opParamInfo.attnMask.tensor = present ? &mask : nullptr;
            const bool expected = (mode == 0 && !present) || ((mode == 3 || mode == 4) && present);
            EXPECT_EQ(Check() == ge::GRAPH_SUCCESS, expected);
        }
    }
}

TEST_F(FlashMlaCheckerTest, BandWindowBounds)
{
    info.maskMode = 4;
    info.opParamInfo.attnMask.tensor = &mask;
    for (int64_t value : {-2L, -1L, 0L, int64_t(INT32_MAX), int64_t(INT32_MAX) + 1}) {
        SCOPED_TRACE(value);
        info.winLeft = value;
        EXPECT_EQ(Check() == ge::GRAPH_SUCCESS, value >= -1 && value <= INT32_MAX);
        info.winLeft = -1;
        info.winRight = value;
        EXPECT_EQ(Check() == ge::GRAPH_SUCCESS, value >= -1 && value <= INT32_MAX);
        info.winRight = -1;
    }
    for (int mode : {0, 3}) {
        info.maskMode = mode;
        info.opParamInfo.attnMask.tensor = mode == 0 ? nullptr : &mask;
        info.winLeft = 0;
        EXPECT_EQ(Check(), ge::GRAPH_FAILED);
        info.winLeft = -1;
        info.winRight = 0;
        EXPECT_EQ(Check(), ge::GRAPH_FAILED);
        info.winRight = -1;
    }
}

TEST_F(FlashMlaCheckerTest, MaskDtypeRankShapeAndDescriptor)
{
    info.maskMode = 4;
    info.opParamInfo.attnMask.tensor = &mask;
    for (auto dtype : {ge::DT_INT8, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16}) {
        maskDesc.SetDataType(dtype);
        EXPECT_EQ(Check() == ge::GRAPH_SUCCESS, dtype == ge::DT_INT8);
    }
    maskDesc.SetDataType(ge::DT_INT8);
    for (auto shape : {gert::Shape({2048}), gert::Shape({1, 2048, 2048}), gert::Shape({2047, 2048}),
                       gert::Shape({2048, 2047}), gert::Shape({0, 2048})}) {
        mask.MutableStorageShape() = shape;
        EXPECT_EQ(Check(), ge::GRAPH_FAILED);
    }
    mask.MutableStorageShape() = gert::Shape({2048, 2048});
    info.opParamInfo.attnMask.desc = nullptr;
    EXPECT_EQ(Check(), ge::GRAPH_FAILED);
}

TEST_F(FlashMlaCheckerTest, SinkDtypeRankAndOptionalPlaceholders)
{
    info.opParamInfo.sinks.tensor = &sink;
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32}) {
        floatDesc.SetDataType(dtype);
        SinksChecker checker;
        EXPECT_EQ(checker.CheckSinglePara(info) == ge::GRAPH_SUCCESS, dtype == ge::DT_FLOAT);
    }
    floatDesc.SetDataType(ge::DT_FLOAT);
    sink.MutableStorageShape() = gert::Shape({1, 64});
    EXPECT_EQ(Check(), ge::GRAPH_FAILED);
    sink.MutableStorageShape() = gert::Shape({64});
    info.opParamInfo.sinks.desc = nullptr;
    EXPECT_EQ(Check(), ge::GRAPH_FAILED);
    info.opParamInfo.sinks.desc = &floatDesc;
    sink.MutableStorageShape() = gert::Shape({0});
    EXPECT_EQ(Check(), ge::GRAPH_SUCCESS);
    info.opParamInfo.sinks.tensor = nullptr;
    EXPECT_EQ(Check(), ge::GRAPH_SUCCESS);
}

TEST_F(FlashMlaCheckerTest, LseShapeForEachQueryLayout)
{
    SoftmaxLSEChecker checker;
    info.s1Size = 4;
    for (auto layout : {Layout::BSND, Layout::BNSD, Layout::TND}) {
        info.qLayout = layout;
        const gert::Shape expected = layout == Layout::TND ? gert::Shape({64, 8}) : gert::Shape({2, 64, 4});
        lse.MutableStorageShape() = expected;
        EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_SUCCESS);
        for (size_t axis = 0; axis < expected.GetDimNum(); ++axis) {
            for (int64_t size : {int64_t(0), expected.GetDim(axis) + 1}) {
                SCOPED_TRACE(testing::Message() << "layout=" << int(layout) << " axis=" << axis << " size=" << size);
                lse.MutableStorageShape().SetDim(axis, size);
                EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
                lse.MutableStorageShape() = expected;
            }
        }
        lse.MutableStorageShape() = gert::Shape({1});
        EXPECT_EQ(checker.CheckMultiPara(info), ge::GRAPH_FAILED);
    }
}

TEST_F(FlashMlaCheckerTest, LseDescriptorDtypeAndDisabledOutput)
{
    floatDesc.SetDataType(ge::DT_FLOAT16);
    EXPECT_EQ(Check(), ge::GRAPH_FAILED);
    floatDesc.SetDataType(ge::DT_FLOAT);
    info.opParamInfo.lseOut.desc = nullptr;
    EXPECT_EQ(Check(), ge::GRAPH_FAILED);
    info.opParamInfo.lseOut.desc = &floatDesc;
    info.opParamInfo.lseOut.shape = nullptr;
    EXPECT_EQ(Check(), ge::GRAPH_FAILED);
    info.softmaxLseFlag = false;
    returnLse = 0;
    info.opParamInfo.lseOut.desc = nullptr;
    EXPECT_EQ(Check(), ge::GRAPH_SUCCESS);
}
} // namespace FlashMlaWithKvcacheUT
