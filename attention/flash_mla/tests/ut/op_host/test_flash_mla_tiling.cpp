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

#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "../../../op_host/flash_mla_value_tiling.h"
#include "../../../../flash_mla_metadata/op_host/flash_mla_metadata_config.h"
#include "../../../op_kernel/arch22/flash_mla_tiling_adapter.h"
#include "../../../op_kernel/flash_mla_metadata_protocol.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
using Para = gert::TilingContextPara;
using Tensor = Para::TensorDescription;
using FlashMla::MlaFamily;
using FlashMla::MlaLayout;
using FlashMla::MlaNonQuantTiling;
using FlashMla::MlaStaticTiling;

enum InputIndex : size_t {
    QUERY,
    KEY,
    VALUE,
    MASK,
    SEQ_Q,
    SEQ_KV,
    BLOCK_TABLE,
    KEY_SCALE,
    VALUE_SCALE,
    QUERY_ROPE,
    KEY_ROPE,
    QUERY_SCALE,
    METADATA
};
enum AttrIndex : size_t {
    NUM_HEADS,
    SCALE,
    PRE_TOKENS,
    NEXT_TOKENS,
    LAYOUT,
    KV_HEADS,
    SPARSE,
    PRECISE,
    BLOCK_SIZE,
    LSE,
    KEY_MODE,
    VALUE_MODE,
    QUERY_MODE
};

gert::StorageShape StorageShape(const std::vector<int64_t> &dims)
{
    gert::StorageShape shape;
    for (auto dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

Tensor Desc(const std::vector<int64_t> &dims, ge::DataType dtype)
{
    return Tensor(StorageShape(dims), dtype, ge::FORMAT_ND);
}

void Shape(Para &para, size_t index, const std::vector<int64_t> &dims)
{
    para.inputTensorDesc_[index].shape_ = StorageShape(dims);
}

template <typename T>
void Attr(Para &para, size_t index, T value)
{
    para.attrs_[index].attr_ = Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

std::string Platform(uint32_t aic = 24, uint32_t aiv = 48)
{
    return "{\"hardware_info\":{\"CORE_NUM\":" + std::to_string(aic) + ",\"cube_core_cnt\":" + std::to_string(aic) +
           ",\"vector_core_cnt\":" + std::to_string(aiv) +
           ",\"UB_SIZE\":196608,\"L1_SIZE\":524288,\"L0A_SIZE\":65536,\"L0B_SIZE\":65536,"
           "\"L0C_SIZE\":131072,\"L2_SIZE\":33554432}}";
}

std::vector<int64_t> DenseShape(const std::string &layout, int64_t batch, int64_t heads, int64_t seq, int64_t dim)
{
    if (layout == "BSH") {
        return {batch, seq, heads * dim};
    }
    if (layout == "BSND") {
        return {batch, seq, heads, dim};
    }
    if (layout == "BNSD") {
        return {batch, heads, seq, dim};
    }
    return {seq, heads, dim};
}

Para MakeTilingCase(const std::string &layout = "BNSD", ge::DataType dtype = ge::DT_FLOAT16, int64_t batch = 2,
                    int64_t heads = 8, int64_t qSize = 1, int64_t kvSize = 128, int64_t blockSize = 0, bool nz = false)
{
    // FlashMla's compile information is empty; the SDK builder still requires a non-null pointer.
    static uint8_t compileInfo = 0;
    const auto qLayout = layout.substr(0, layout.find('_'));
    const bool quant = dtype == ge::DT_INT8;
    const auto ropeType = quant ? ge::DT_BF16 : dtype;
    const auto q = DenseShape(qLayout, batch, heads, qSize, 512);
    const auto qr = DenseShape(qLayout, batch, heads, qSize, 64);
    auto k = DenseShape(qLayout, batch, 1, kvSize, 512);
    auto kr = DenseShape(qLayout, batch, 1, kvSize, 64);
    if (blockSize != 0) {
        const int64_t blocks = batch * 4;
        k = nz ? std::vector<int64_t>{blocks, 1, quant ? 16 : 32, blockSize, quant ? 32 : 16} :
                 std::vector<int64_t>{blocks, 1, blockSize, 512};
        kr = nz ? std::vector<int64_t>{blocks, 1, 4, blockSize, 16} : std::vector<int64_t>{blocks, 1, blockSize, 64};
    }
    auto output = q;
    if (layout.find('_') != std::string::npos) {
        output =
            qLayout == "TND" ? std::vector<int64_t>{heads, qSize, 512} : std::vector<int64_t>{heads, batch, qSize, 512};
    }
    std::vector<Tensor> inputs = {Desc(q, dtype),
                                  Desc(k, dtype),
                                  Desc(k, dtype),
                                  Desc({}, ge::DT_BOOL),
                                  Desc({}, ge::DT_INT64),
                                  Desc({}, ge::DT_INT64),
                                  Desc({}, ge::DT_INT32),
                                  Desc({}, ge::DT_FLOAT),
                                  Desc({}, ge::DT_FLOAT),
                                  Desc(qr, ropeType),
                                  Desc(kr, ropeType),
                                  Desc({}, ge::DT_FLOAT),
                                  Desc({FlashMla::kMlaMetadataElements}, ge::DT_INT32)};
    if (qLayout == "TND") {
        inputs[SEQ_Q] = Desc({batch}, ge::DT_INT64);
        inputs[SEQ_KV] = Desc({batch}, ge::DT_INT64);
    }
    if (blockSize != 0) {
        inputs[BLOCK_TABLE] = Desc({batch, (kvSize + blockSize - 1) / blockSize}, ge::DT_INT32);
        inputs[SEQ_KV] = Desc({batch}, ge::DT_INT64);
    }
    if (quant) {
        auto scaleShape = q;
        scaleShape.pop_back();
        if (qLayout == "BSH") {
            scaleShape.push_back(heads);
        }
        inputs[QUERY_SCALE] = Desc(scaleShape, ge::DT_FLOAT);
        inputs[KEY_SCALE] = Desc({1}, ge::DT_FLOAT);
        inputs[VALUE_SCALE] = Desc({1}, ge::DT_FLOAT);
    }
    auto lse = qLayout == "TND" ? std::vector<int64_t>{qSize, heads, 1} : std::vector<int64_t>{batch, heads, qSize, 1};
    return Para("FlashMla", inputs, {Desc(output, ropeType), Desc(lse, ge::DT_FLOAT)},
                {{"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(heads)},
                 {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.125F)},
                 {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT32_MAX)},
                 {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(INT32_MAX)},
                 {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
                 {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                 {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(blockSize)},
                 {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
                 {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                 {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(quant ? 3 : 0)}},
                &compileInfo, "Ascend910B", 24, 196608, sizeof(MlaStaticTiling), Platform());
}

Para QuantCase(const std::string &layout = "BSND", int64_t qSize = 1, int64_t batch = 4)
{
    return MakeTilingCase(layout, ge::DT_INT8, batch, 8, qSize, 512, 128, true);
}

void ReadTiling(const Para &para, TilingInfo &result, MlaStaticTiling &tiling)
{
    ASSERT_TRUE(ExecuteTiling(para, result));
    ASSERT_NE(result.tilingData, nullptr);
    ASSERT_EQ(result.tilingDataSize, sizeof(tiling));
    std::memcpy(&tiling, result.tilingData.get(), sizeof(tiling));
    EXPECT_EQ(result.tilingKey, tiling.launch.staticLaunchKey);
    EXPECT_EQ(result.blockNum, tiling.launch.physicalBlockDim);
    ASSERT_EQ(result.workspaceSizes.size(), 1U);
    EXPECT_EQ(result.workspaceSizes[0], tiling.launch.workspaceCap);
}

template <typename T>
T Payload(const MlaStaticTiling &tiling)
{
    T result{};
    std::memcpy(&result, tiling.payload, sizeof(result));
    return result;
}

struct DenseCase {
    const char *layout;
    MlaLayout qLayout;
    uint32_t outputLayout;
    uint64_t tilingKey;
};

class FlashMlaDenseTiling : public testing::TestWithParam<DenseCase> {};

TEST_P(FlashMlaDenseTiling, StaticLaunchAndPayload)
{
    const auto &test = GetParam();
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        SCOPED_TRACE(dtype);
        auto para = MakeTilingCase(test.layout, dtype, 2, 8, 3);
        TilingInfo actual;
        MlaStaticTiling tiling{};
        ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
        EXPECT_EQ(actual.tilingKey, test.tilingKey + (dtype == ge::DT_BF16 ? 0x1000 : 0));
        EXPECT_EQ(tiling.launch.family, MlaFamily::ARCH22_MLA);
        EXPECT_EQ(tiling.launch.layoutQ, test.qLayout);
        EXPECT_EQ(tiling.launch.layoutKv, test.qLayout);
        EXPECT_EQ(tiling.launch.aicNum, 24U);
        EXPECT_EQ(tiling.launch.aivNum, 48U);
        EXPECT_EQ(tiling.launch.cvRatio, 2U);
        EXPECT_GT(actual.blockNum, 0U);
        EXPECT_GT(tiling.launch.workspaceCap, tiling.launch.sysWorkspaceSize);
        EXPECT_EQ(tiling.launch.qStorageSize, 3U);
        EXPECT_EQ(tiling.launch.kvStorageSize, 128U);
        EXPECT_EQ(tiling.launch.outputElements, test.qLayout == MlaLayout::TND ? 3U * 8U * 512U : 6U * 8U * 512U);
        EXPECT_EQ(tiling.launch.lseElements, test.qLayout == MlaLayout::TND ? 24U : 48U);
        const auto payload = Payload<MlaNonQuantTiling>(tiling);
        EXPECT_EQ(payload.baseParams.bSize, 2U);
        EXPECT_EQ(payload.baseParams.gSize, 8U);
        EXPECT_EQ(payload.baseParams.n2Size, 1U);
        EXPECT_EQ(payload.baseParams.headDim, 512U);
        EXPECT_EQ(payload.baseParams.headDimRope, 64U);
        EXPECT_FLOAT_EQ(payload.baseParams.scaleValue, 0.125F);
        EXPECT_EQ(payload.baseParams.outputLayout, test.outputLayout);
        EXPECT_EQ(payload.baseParams.accumQSeqFlag, test.qLayout == MlaLayout::TND ? 1U : 0U);
        EXPECT_EQ(payload.baseParams.accumKVSeqFlag, test.qLayout == MlaLayout::TND ? 1U : 0U);
        EXPECT_EQ(payload.baseParams.softmaxLseFlag, 1U);
    }
}

INSTANTIATE_TEST_SUITE_P(Layouts, FlashMlaDenseTiling,
                         testing::Values(DenseCase{"BSH", MlaLayout::BSH, 0, 0x110001},
                                         DenseCase{"BSND", MlaLayout::BSND, 0, 0x110111},
                                         DenseCase{"BNSD", MlaLayout::BNSD, 1, 0x110221},
                                         DenseCase{"TND", MlaLayout::TND, 3, 0x110331},
                                         DenseCase{"BSH_NBSD", MlaLayout::BSH, 4, 0x110001},
                                         DenseCase{"BSND_NBSD", MlaLayout::BSND, 4, 0x110111},
                                         DenseCase{"BNSD_NBSD", MlaLayout::BNSD, 4, 0x110221},
                                         DenseCase{"TND_NTD", MlaLayout::TND, 5, 0x110331}),
                         [](const testing::TestParamInfo<DenseCase> &info) { return info.param.layout; });

TEST(FlashMlaStaticTiling, PageCacheLayoutsAndBlockTableRowStride)
{
    for (size_t rank : {3U, 4U, 5U}) {
        SCOPED_TRACE(rank);
        auto para = MakeTilingCase("BNSD", ge::DT_FLOAT16, 2, 8, 1, 512, 128, rank == 5);
        if (rank == 3) {
            Shape(para, KEY, {8, 128, 512});
            Shape(para, VALUE, {8, 128, 512});
            Shape(para, KEY_ROPE, {8, 128, 64});
        }
        TilingInfo actual;
        MlaStaticTiling tiling{};
        ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
        EXPECT_EQ(tiling.launch.pageAttention, 1U);
        EXPECT_EQ(tiling.launch.layoutKv, rank == 3 ? MlaLayout::BSH : (rank == 4 ? MlaLayout::BNSD : MlaLayout::NZ));
        EXPECT_EQ(tiling.launch.kvStorageSize, 512U);
        const auto payload = Payload<MlaNonQuantTiling>(tiling);
        EXPECT_EQ(payload.pageAttenParams.blockSize, 128U);
        EXPECT_EQ(payload.pageAttenParams.maxBlockNumPerBatch, 4U);
        EXPECT_EQ(payload.baseParams.actualSeqS2Dims, 2U);
    }
}

// Use the public metadata semantics, then apply its value patch to the real
// consumer tiling. This detects producer/consumer signature and static-field drift.
TEST_P(FlashMlaDenseTiling, NonQuantMetadataIsSharedByFp16AndBf16)
{
    using namespace FlashMla;
    const auto &test = GetParam();
    MlaScheduleConfig config;
    config.numHeads = 8;
    config.batchSize = 2;
    config.qSeqSize = 3;
    config.kvSeqSize = 128;
    ASSERT_TRUE(ConfigureMlaMetadata(test.layout, 0, config));
    std::vector<int64_t> seqQ, seqKv;
    if (test.qLayout == MlaLayout::TND) {
        seqQ = {1, 3};
        seqKv = {64, 128};
        config.seqQPresent = config.seqKvPresent = 1;
        config.seqQCount = config.seqKvCount = 2;
    }
    MlaPlatformInfo platform;
    platform.aicNum = 24;
    platform.aivNum = 48;
    MlaValuePatch patch{};
    ASSERT_EQ(ComputeMlaValueTiling(config, platform, {seqQ.data(), seqQ.size()}, {seqKv.data(), seqKv.size()}, patch),
              MlaStatus::SUCCESS);
    alignas(64) MlaMetadata metadata{};
    ASSERT_EQ(WriteMlaMetadata(&metadata, sizeof(metadata), patch), MlaStatus::SUCCESS);
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        SCOPED_TRACE(dtype);
        TilingInfo actual;
        MlaStaticTiling tiling{};
        ASSERT_NO_FATAL_FAILURE(ReadTiling(MakeTilingCase(test.layout, dtype, 2, 8, 3), actual, tiling));
        auto payload = Payload<MlaNonQuantTiling>(tiling);
        ASSERT_EQ(ApplyMlaValuePatch(metadata, tiling.launch, &payload, sizeof(payload)), MlaStatus::SUCCESS);
        EXPECT_EQ(tiling.launch.queryDataType, dtype == ge::DT_FLOAT16 ? kMlaFp16 : kMlaBf16);
        EXPECT_EQ(payload.baseParams.outputLayout, test.outputLayout);
        uint64_t workspace = 0;
        EXPECT_EQ(FinalizeMlaTiling(tiling.launch, metadata.header, payload, tiling.launch.aicNum, workspace),
                  MlaStatus::SUCCESS);
    }
}

TEST_P(FlashMlaDenseTiling, PagedMetadataIsIndependentOfKvStorageLayout)
{
    using namespace FlashMla;
    const auto &test = GetParam();
    for (uint32_t ratio : {1U, 2U}) {
        for (uint32_t blockSize : {112U, 128U}) {
            SCOPED_TRACE(ratio);
            SCOPED_TRACE(blockSize);
            MlaScheduleConfig config;
            config.numHeads = 8;
            config.batchSize = 2;
            config.qSeqSize = test.qLayout == MlaLayout::TND ? 3 : 1;
            config.kvSeqSize = blockSize * 4;
            config.blockSize = blockSize;
            ASSERT_TRUE(ConfigureMlaMetadata(test.layout, 0, config));
            EXPECT_EQ(config.layoutKv, MlaLayout::BSH);
            std::vector<int64_t> seqQ;
            if (test.qLayout == MlaLayout::TND) {
                seqQ = {1, 3};
                config.seqQPresent = 1;
                config.seqQCount = 2;
            }
            // Decreasing PA lengths must remain absolute even for TND queries.
            const std::vector<int64_t> seqKv = {blockSize * 3, blockSize};
            config.seqKvPresent = 1;
            config.seqKvCount = 2;
            MlaPlatformInfo platform;
            platform.aicNum = 24;
            platform.aivNum = 24 * ratio;
            MlaValuePatch patch{};
            ASSERT_EQ(ComputeMlaValueTiling(config, platform, {seqQ.data(), seqQ.size()}, {seqKv.data(), seqKv.size()},
                                            patch),
                      MlaStatus::SUCCESS);
            alignas(64) MlaMetadata metadata{};
            ASSERT_EQ(WriteMlaMetadata(&metadata, sizeof(metadata), patch), MlaStatus::SUCCESS);
            for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
                for (size_t rank : {3U, 4U, 5U}) {
                    SCOPED_TRACE(dtype);
                    SCOPED_TRACE(rank);
                    auto para = MakeTilingCase(test.layout, dtype, 2, 8, config.qSeqSize, config.kvSeqSize, blockSize,
                                               rank == 5);
                    para.socInfoString_ = Platform(24, 24 * ratio);
                    if (rank == 3) {
                        Shape(para, KEY, {8, blockSize, 512});
                        Shape(para, VALUE, {8, blockSize, 512});
                        Shape(para, KEY_ROPE, {8, blockSize, 64});
                    }
                    TilingInfo actual;
                    MlaStaticTiling tiling{};
                    ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
                    auto payload = Payload<MlaNonQuantTiling>(tiling);
                    ASSERT_EQ(ApplyMlaValuePatch(metadata, tiling.launch, &payload, sizeof(payload)),
                              MlaStatus::SUCCESS);
                    const auto kvLayout = rank == 3 ? MlaLayout::BSH : (rank == 4 ? MlaLayout::BNSD : MlaLayout::NZ);
                    EXPECT_EQ(tiling.launch.layoutKv, kvLayout);
                    EXPECT_EQ(payload.baseParams.accumKVSeqFlag, 0U);
                    EXPECT_EQ(payload.baseParams.outputLayout, test.outputLayout);
                    uint64_t workspace = 0;
                    EXPECT_EQ(
                        FinalizeMlaTiling(tiling.launch, metadata.header, payload, tiling.launch.aicNum, workspace),
                        MlaStatus::SUCCESS);
                    // Compare every scheduling byte against the actual dtype/cache
                    // configuration, not just the normalized signature.
                    auto consumerConfig = config;
                    consumerConfig.layoutKv = kvLayout;
                    consumerConfig.queryDataType = dtype == ge::DT_FLOAT16 ? kMlaFp16 : kMlaBf16;
                    MlaValuePatch expected{};
                    ASSERT_EQ(ComputeMlaValueTiling(consumerConfig, platform, {seqQ.data(), seqQ.size()},
                                                    {seqKv.data(), seqKv.size()}, expected),
                              MlaStatus::SUCCESS);
                    EXPECT_EQ(std::memcmp(&patch, &expected, sizeof(patch)), 0);
                }
            }
        }
    }
}

TEST(FlashMlaStaticTiling, QuantModeSelectsMatchingConsumerFamily)
{
    using namespace FlashMla;
    for (const char *layout : {"BSH", "BSND", "TND", "BSH_NBSD", "BSND_NBSD", "TND_NTD"}) {
        SCOPED_TRACE(layout);
        const bool tnd = std::string(layout).find("TND") == 0;
        auto para = QuantCase(layout, tnd ? 2 : 1, 2);
        TilingInfo actual;
        MlaStaticTiling tiling{};
        ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
        MlaScheduleConfig config;
        config.numHeads = 8;
        config.batchSize = 2;
        config.qSeqSize = tnd ? 2 : 1;
        config.kvSeqSize = 512;
        config.blockSize = 128;
        const std::vector<int64_t> seqQ = tnd ? std::vector<int64_t>{1, 2} : std::vector<int64_t>{};
        const std::vector<int64_t> seqKv{128, 512};
        config.seqQPresent = tnd;
        config.seqQCount = seqQ.size();
        config.seqKvPresent = 1;
        config.seqKvCount = seqKv.size();
        MlaPlatformInfo platform;
        platform.aicNum = tiling.launch.aicNum;
        platform.aivNum = tiling.launch.aivNum;
        for (int64_t quantMode : {0, 1}) {
            ASSERT_TRUE(ConfigureMlaMetadata(layout, quantMode, config));
            MlaValuePatch patch{};
            ASSERT_EQ(ComputeMlaValueTiling(config, platform, {seqQ.data(), seqQ.size()}, {seqKv.data(), seqKv.size()},
                                            patch),
                      MlaStatus::SUCCESS);
            alignas(64) MlaMetadata metadata{};
            ASSERT_EQ(WriteMlaMetadata(&metadata, sizeof(metadata), patch), MlaStatus::SUCCESS);
            auto payload = Payload<MlaLegacyTiling>(tiling);
            EXPECT_EQ(ApplyMlaValuePatch(metadata, tiling.launch, &payload, sizeof(payload)),
                      quantMode == 1 ? MlaStatus::SUCCESS : MlaStatus::ABI_MISMATCH);
            if (quantMode == 1) {
                EXPECT_EQ(config.layoutKv, MlaLayout::NZ);
                uint64_t workspace = 0;
                EXPECT_EQ(FinalizeMlaTiling(tiling.launch, metadata.header, payload, tiling.launch.aicNum, workspace),
                          MlaStatus::SUCCESS);
            }
        }
    }
}

TEST(FlashMlaStaticTiling, EmptyPageTableRejectsNonEmptyInputs)
{
    auto para = MakeTilingCase("BNSD", ge::DT_FLOAT16, 1, 8, 1, 16, 16);
    Shape(para, BLOCK_TABLE, {1, 0});
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(FlashMlaStaticTiling, EmptyPageTablePreservesEmptyInputInitialization)
{
    for (const char *emptyInput : {"query", "cache", "batch"}) {
        SCOPED_TRACE(emptyInput);
        const bool emptyBatch = std::strcmp(emptyInput, "batch") == 0;
        const bool emptyQuery = std::strcmp(emptyInput, "query") == 0;
        auto para = MakeTilingCase("BNSD", ge::DT_FLOAT16, emptyBatch ? 0 : 1, 8, emptyQuery ? 0 : 1, 16, 16);
        Shape(para, BLOCK_TABLE, {emptyBatch ? 0 : 1, 0});
        if (std::strcmp(emptyInput, "cache") == 0) {
            Shape(para, KEY, {0, 1, 16, 512});
            Shape(para, VALUE, {0, 1, 16, 512});
            Shape(para, KEY_ROPE, {0, 1, 16, 64});
        }
        TilingInfo actual;
        MlaStaticTiling tiling{};
        ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
        EXPECT_EQ(tiling.launch.family, MlaFamily::NO_SCHEDULE);
        EXPECT_EQ(tiling.launch.kvStorageSize, 0U);
        EXPECT_EQ(tiling.launch.workspaceCap, tiling.launch.sysWorkspaceSize);
        EXPECT_EQ(tiling.launch.payloadAbiId, 0U);
    }
}

TEST(FlashMlaStaticTiling, PagedZeroKvMaskSurvivesProductionMetadataMerge)
{
    using namespace FlashMla;
    auto para = MakeTilingCase("BNSD", ge::DT_FLOAT16, 1, 8, 1, 16, 16);
    Shape(para, MASK, {1, 1, 16});
    int64_t seqKv = 0;
    para.inputTensorDesc_[SEQ_KV] = Tensor(StorageShape({1}), ge::DT_INT64, ge::FORMAT_ND, true, &seqKv);
    TilingInfo actual;
    MlaStaticTiling tiling{};
    ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
    ASSERT_EQ(tiling.launch.family, MlaFamily::ARCH22_MLA);
    ASSERT_EQ(tiling.launch.maskDim2, 16U);

    MlaScheduleConfig config;
    config.numHeads = 8;
    config.batchSize = 1;
    config.qSeqSize = 1;
    config.kvSeqSize = 16;
    config.layoutQ = MlaLayout::BNSD;
    config.layoutKv = MlaLayout::BNSD;
    config.blockSize = 16;
    config.preTokens = INT32_MAX;
    config.nextTokens = INT32_MAX;
    config.seqKvPresent = 1;
    config.seqKvCount = 1;
    config.hasAttenMask = 1;
    MlaPlatformInfo platform;
    platform.aicNum = tiling.launch.aicNum;
    platform.aivNum = tiling.launch.aivNum;
    platform.sysWorkspaceSize = tiling.launch.sysWorkspaceSize;
    MlaValuePatch patch{};
    ASSERT_EQ(ComputeMlaValueTiling(config, platform, {}, {&seqKv, 1}, patch), MlaStatus::SUCCESS);
    alignas(64) MlaMetadata metadata{};
    ASSERT_EQ(WriteMlaMetadata(&metadata, sizeof(metadata), patch), MlaStatus::SUCCESS);
    auto payload = Payload<MlaNonQuantTiling>(tiling);
    ASSERT_EQ(ApplyMlaValuePatch(metadata, tiling.launch, &payload, sizeof(payload)), MlaStatus::SUCCESS);
    EXPECT_EQ(metadata.header.maxKv, 0U);
    // FIA retains an inner-size surrogate for empty KV; it is not a mask extent.
    EXPECT_EQ(payload.baseParams.s2Size, 1024U);
    uint64_t workspaceLayout = 0;
    EXPECT_EQ(FinalizeMlaTiling(tiling.launch, metadata.header, payload, tiling.launch.aicNum, workspaceLayout),
              MlaStatus::SUCCESS);
    EXPECT_LE(workspaceLayout, tiling.launch.workspaceCap);
}

TEST(FlashMlaStaticTiling, TndPageKvUsesAbsoluteSequences)
{
    auto para = MakeTilingCase("TND_NTD", ge::DT_BF16, 2, 8, 6, 512, 128);
    TilingInfo actual;
    MlaStaticTiling tiling{};
    ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
    const auto payload = Payload<MlaNonQuantTiling>(tiling);
    EXPECT_EQ(payload.baseParams.actualSeqS1Dims, 2U);
    EXPECT_EQ(payload.baseParams.actualSeqS2Dims, 2U);
    EXPECT_EQ(payload.baseParams.accumQSeqFlag, 1U);
    EXPECT_EQ(payload.baseParams.accumKVSeqFlag, 0U);
    EXPECT_EQ(payload.baseParams.outputLayout, 5U);
}

TEST(FlashMlaStaticTiling, Int8StaticPayload)
{
    for (const char *layout : {"BSH", "BSND", "TND_NTD"}) {
        SCOPED_TRACE(layout);
        auto para = QuantCase(layout);
        TilingInfo actual;
        MlaStaticTiling tiling{};
        ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
        EXPECT_EQ(tiling.launch.family, MlaFamily::LEGACY_MLA);
        EXPECT_EQ(tiling.launch.queryDataType, FlashMla::kMlaInt8);
        EXPECT_EQ(tiling.launch.layoutKv, MlaLayout::NZ);
        EXPECT_EQ(tiling.launch.pageAttention, 1U);
        const auto payload = Payload<FlashMla::MlaLegacyTiling>(tiling);
        EXPECT_EQ(payload.baseParams.batchSize, 4U);
        EXPECT_EQ(payload.baseParams.blockSize, 128U);
        EXPECT_EQ(payload.baseParams.maxBlockNumPerBatch, 4U);
        EXPECT_EQ(payload.baseParams.nNumOfQInOneGroup, 8U);
        EXPECT_EQ(payload.baseParams.actualLenDims, 4U);
        EXPECT_EQ(payload.baseParams.actualLenQDims, tiling.launch.layoutQ == MlaLayout::TND ? 4U : 0U);
        EXPECT_EQ(payload.baseParams.attenMaskSize, 0U);
        EXPECT_FLOAT_EQ(payload.baseParams.scaleValue, 0.125F);
    }
}

TEST(FlashMlaStaticTiling, DisabledLseAndExplicitSingleKvHeadMatchFia)
{
    auto para = MakeTilingCase("BSH", ge::DT_FLOAT16, 2, 1);
    Attr<int64_t>(para, KV_HEADS, 0);
    Attr<bool>(para, LSE, false);
    para.outputTensorDesc_[1] = Desc({0}, ge::DT_FLOAT);
    TilingInfo actual;
    EXPECT_FALSE(ExecuteTiling(para, actual));
    Attr<int64_t>(para, KV_HEADS, 1);
    MlaStaticTiling tiling{};
    ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
    EXPECT_EQ(tiling.launch.lseElements, 0U);
    EXPECT_EQ(tiling.launch.softmaxLseFlag, 0U);
    EXPECT_EQ(Payload<MlaNonQuantTiling>(tiling).baseParams.gSize, 1U);
}

TEST(FlashMlaStaticTiling, DefaultKvHeadPreservesOriginalEmptyAndInt8Paths)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (auto sizes : {std::pair<int64_t, int64_t>{0, 128}, {1, 0}}) {
            auto para = MakeTilingCase("BSH", dtype, 2, 1, sizes.first, sizes.second);
            Attr<int64_t>(para, KV_HEADS, 0);
            TilingInfo actual;
            EXPECT_TRUE(ExecuteTiling(para, actual));
        }
    }
    auto quant = MakeTilingCase("BSH", ge::DT_INT8, 2, 1, 1, 128, 128, true);
    Attr<int64_t>(quant, KV_HEADS, 0);
    TilingInfo actual;
    EXPECT_TRUE(ExecuteTiling(quant, actual));
}

TEST(FlashMlaStaticTiling, EmptyBatchOrQueryOrKvUsesVectorInitialization)
{
    for (auto dims : {std::vector<int64_t>{0, 3, 128}, {2, 0, 128}, {2, 3, 0}}) {
        auto para = MakeTilingCase("BNSD", ge::DT_BF16, dims[0], 8, dims[1], dims[2]);
        TilingInfo actual;
        MlaStaticTiling tiling{};
        ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
        EXPECT_EQ(tiling.launch.family, MlaFamily::NO_SCHEDULE);
        EXPECT_EQ(actual.tilingKey, 0x101000);
        EXPECT_EQ(actual.blockNum, 48U);
        EXPECT_EQ(tiling.launch.workspaceCap, tiling.launch.sysWorkspaceSize);
        EXPECT_EQ(tiling.launch.payloadAbiId, 0U);
        const uint8_t zero[FlashMla::kMlaMaxPayloadBytes]{};
        EXPECT_EQ(std::memcmp(tiling.payload, zero, sizeof(zero)), 0);
    }
}

TEST(FlashMlaStaticTiling, SupportsA2A3AndEffectiveCoreQuotas)
{
    for (const char *soc : {"Ascend910B", "Ascend910_93"}) {
        for (auto cores : {std::vector<uint32_t>{1, 1}, {1, 2}, {25, 50}}) {
            SCOPED_TRACE(soc);
            SCOPED_TRACE(cores[1]);
            auto para = MakeTilingCase();
            para.socVersion_ = soc;
            para.socInfoString_ = Platform(cores[0], cores[1]);
            TilingInfo actual;
            MlaStaticTiling tiling{};
            ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
            EXPECT_EQ(tiling.launch.aicNum, cores[0]);
            EXPECT_EQ(tiling.launch.aivNum, cores[1]);
            EXPECT_EQ(tiling.launch.cvRatio, cores[1] / cores[0]);
            EXPECT_EQ(actual.tilingKey, 0x110220 + cores[1] / cores[0] - 1);
        }
    }
}

TEST(FlashMlaStaticTiling, SequenceDescriptorsAllowBroadcastPaddingAndEmpty)
{
    for (int64_t count : {0, 1, 4, 65537}) {
        auto para = MakeTilingCase("BNSD", ge::DT_FLOAT16, 4);
        Shape(para, SEQ_Q, {count});
        Shape(para, SEQ_KV, {count});
        TilingInfo actual;
        MlaStaticTiling tiling{};
        ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
        const auto payload = Payload<MlaNonQuantTiling>(tiling);
        EXPECT_EQ(payload.baseParams.actualSeqS1Dims, count);
        EXPECT_EQ(payload.baseParams.actualSeqS2Dims, count);
    }
    auto quant = QuantCase();
    Shape(quant, SEQ_Q, {2}); // Non-TND INT8 ignores Q descriptor count even when it is below B.
    TilingInfo actual;
    MlaStaticTiling tiling{};
    ASSERT_NO_FATAL_FAILURE(ReadTiling(quant, actual, tiling));
    EXPECT_EQ(Payload<FlashMla::MlaLegacyTiling>(tiling).baseParams.actualLenQDims, 0U);
}

TEST(FlashMlaStaticTiling, LogicalBatchAndHeadCountBoundaries)
{
    for (int64_t batch : {65536, 65537}) {
        // Empty Q isolates the logical batch limit without allocating a large workspace.
        auto para = MakeTilingCase("BNSD", ge::DT_FLOAT16, batch, 128, 0);
        TilingInfo result;
        EXPECT_EQ(ExecuteTiling(para, result), batch == 65536);
    }
    auto para = MakeTilingCase("BSND", ge::DT_BF16, 2, 128);
    TilingInfo result;
    MlaStaticTiling tiling{};
    ASSERT_NO_FATAL_FAILURE(ReadTiling(para, result, tiling));
    EXPECT_EQ(Payload<MlaNonQuantTiling>(tiling).baseParams.gSize, 128U);
}

TEST(FlashMlaStaticTiling, PageBlockSizeBoundaries)
{
    for (int64_t block : {16, 1024}) {
        auto para = MakeTilingCase("BSH", ge::DT_FLOAT16, 2, 8, 1, 2048, block);
        TilingInfo result;
        MlaStaticTiling tiling{};
        ASSERT_NO_FATAL_FAILURE(ReadTiling(para, result, tiling));
        const auto page = Payload<MlaNonQuantTiling>(tiling).pageAttenParams;
        EXPECT_EQ(page.blockSize, block);
        EXPECT_EQ(page.maxBlockNumPerBatch, 2048 / block);
    }
}

TEST(FlashMlaStaticTiling, LaunchDoesNotReadSequenceValues)
{
    auto para = MakeTilingCase("BNSD", ge::DT_FLOAT16, 2, 8, 3, 512, 128);
    int64_t seq[2] = {1, 128};
    para.inputTensorDesc_[SEQ_KV] = Tensor(StorageShape({2}), ge::DT_INT64, ge::FORMAT_ND, true, seq);
    TilingInfo before;
    MlaStaticTiling first{};
    ASSERT_NO_FATAL_FAILURE(ReadTiling(para, before, first));
    seq[0] = -1;
    seq[1] = INT64_MAX; // Value validation belongs to the metadata producer on the execution stream.
    TilingInfo after;
    MlaStaticTiling second{};
    ASSERT_NO_FATAL_FAILURE(ReadTiling(para, after, second));
    EXPECT_EQ(std::memcmp(&first, &second, sizeof(first)), 0);
    EXPECT_EQ(Payload<MlaNonQuantTiling>(second).pageAttenParams.maxBlockNumPerBatch, 4U);
}

TEST(FlashMlaStaticTiling, PageCacheAllowsOnlyDimZeroStrideChanges)
{
    auto para = MakeTilingCase("BNSD", ge::DT_FLOAT16, 2, 8, 1, 512, 128);
    for (size_t input : {KEY, VALUE, KEY_ROPE}) {
        auto &desc = para.inputTensorDesc_[input];
        const int64_t dim = input == KEY_ROPE ? 64 : 512;
        desc.hasStride_ = true;
        desc.stride_ = gert::Stride{256 * dim, 128 * dim, dim, 1};
    }
    TilingInfo actual;
    MlaStaticTiling tiling{};
    ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
    const auto payload = Payload<MlaNonQuantTiling>(tiling);
    EXPECT_EQ(payload.kvStrideParams.keyBnStride, 256U * 512U);
    EXPECT_EQ(payload.kvStrideParams.valueBnStride, 256U * 512U);
    EXPECT_EQ(payload.kvStrideParams.keyRopeBnStride, 256U * 64U);
    para.inputTensorDesc_[KEY].stride_ = gert::Stride{256 * 512, 128 * 512, 513, 1};
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

TEST(FlashMlaStaticTiling, SparseMasksAndNormalizedWindow)
{
    for (int64_t sparse : {0, 3, 4, 9}) {
        auto para = MakeTilingCase("BNSD", ge::DT_FLOAT16, 2, 8, 3);
        Attr<int64_t>(para, SPARSE, sparse);
        para.inputTensorDesc_[MASK] =
            Desc(sparse == 3 || sparse == 4 ? std::vector<int64_t>{2048, 2048} : std::vector<int64_t>{2, 3, 128},
                 ge::DT_BOOL);
        Attr<int64_t>(para, PRE_TOKENS, INT64_MAX);
        Attr<int64_t>(para, NEXT_TOKENS, -INT32_MAX);
        TilingInfo actual;
        MlaStaticTiling tiling{};
        ASSERT_NO_FATAL_FAILURE(ReadTiling(para, actual, tiling));
        const auto mask = Payload<MlaNonQuantTiling>(tiling).maskParams;
        EXPECT_EQ(mask.sparseMode, sparse);
        EXPECT_EQ(mask.attenMaskFlag, 1U);
        EXPECT_EQ(mask.preToken, INT32_MAX);
        EXPECT_EQ(mask.nextToken, sparse == 3 || sparse == 9 ? 0 : -INT32_MAX);
        EXPECT_EQ(tiling.launch.treeMask, sparse == 9 ? 1U : 0U);
    }
}

TEST(FlashMlaStaticTiling, MaskRanksAndDtypes)
{
    for (auto dtype : {ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8}) {
        for (auto dims : {std::vector<int64_t>{1, 3, 128}, {2, 1, 3, 128}}) {
            auto para = MakeTilingCase("BNSD", ge::DT_FLOAT16, 2, 8, 3);
            para.inputTensorDesc_[MASK] = Desc(dims, dtype);
            TilingInfo result;
            MlaStaticTiling tiling{};
            ASSERT_NO_FATAL_FAILURE(ReadTiling(para, result, tiling));
            EXPECT_EQ(tiling.launch.maskRank, dims.size());
            EXPECT_EQ(tiling.launch.maskBatchStride, dims[0] == 1 ? 0U : 384U);
            EXPECT_EQ(Payload<MlaNonQuantTiling>(tiling).maskParams.attenMaskStride, 128U);
        }
    }
    for (auto dims : {std::vector<int64_t>{1, 2048, 2048}, {1, 1, 2048, 2048}}) {
        auto para = MakeTilingCase();
        Attr<int64_t>(para, SPARSE, 3);
        para.inputTensorDesc_[MASK] = Desc(dims, ge::DT_BOOL);
        TilingInfo result;
        MlaStaticTiling tiling{};
        ASSERT_NO_FATAL_FAILURE(ReadTiling(para, result, tiling));
        EXPECT_EQ(Payload<MlaNonQuantTiling>(tiling).maskParams.attenMaskStride, 2048U);
    }
    auto tree = MakeTilingCase("TND", ge::DT_FLOAT16, 2, 8, 5);
    Attr<int64_t>(tree, SPARSE, 9);
    tree.inputTensorDesc_[MASK] = Desc({13}, ge::DT_BOOL);
    TilingInfo result;
    MlaStaticTiling tiling{};
    ASSERT_NO_FATAL_FAILURE(ReadTiling(tree, result, tiling));
    EXPECT_EQ(tiling.launch.maskRank, 1U);
    EXPECT_EQ(tiling.launch.maskBatchStride, 1U);
}

TEST(FlashMlaStaticTiling, Int8QueryLengthBoundaryWithSparseMask)
{
    for (int64_t qSize : {2, 32, 33}) {
        auto para = QuantCase("BSND", qSize);
        Attr<int64_t>(para, SPARSE, 3);
        para.inputTensorDesc_[MASK] = Desc({2048, 2048}, ge::DT_UINT8);
        TilingInfo result;
        EXPECT_EQ(ExecuteTiling(para, result), qSize <= 32);
    }
}

struct InvalidTilingCase {
    const char *name;
    bool quant;
    std::function<void(Para &)> mutate;
};

class FlashMlaInvalidStaticTiling : public testing::TestWithParam<InvalidTilingCase> {};

TEST_P(FlashMlaInvalidStaticTiling, RejectsInvalidContract)
{
    const auto &test = GetParam();
    auto para = test.quant ? QuantCase() : MakeTilingCase();
    test.mutate(para);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

INSTANTIATE_TEST_SUITE_P(
    Invalid, FlashMlaInvalidStaticTiling,
    testing::Values(
        InvalidTilingCase{"MissingAttributes", false, [](Para &p) { p.attrs_.clear(); }},
        InvalidTilingCase{"MissingQuantMode", false, [](Para &p) { p.attrs_.pop_back(); }},
        InvalidTilingCase{"ZeroHeads", false, [](Para &p) { Attr<int64_t>(p, NUM_HEADS, 0); }},
        InvalidTilingCase{"TooManyHeads", false, [](Para &p) { Attr<int64_t>(p, NUM_HEADS, 129); }},
        InvalidTilingCase{"MultipleKvHeads", false, [](Para &p) { Attr<int64_t>(p, KV_HEADS, 2); }},
        InvalidTilingCase{"MissingKvHeadsForGqa", false, [](Para &p) { Attr<int64_t>(p, KV_HEADS, 0); }},
        InvalidTilingCase{"NegativeBlockSize", false, [](Para &p) { Attr<int64_t>(p, BLOCK_SIZE, -1); }},
        InvalidTilingCase{"SparseOutOfRange", false, [](Para &p) { Attr<int64_t>(p, SPARSE, 10); }},
        InvalidTilingCase{"UnsupportedSparseMode", false, [](Para &p) { Attr<int64_t>(p, SPARSE, 2); }},
        InvalidTilingCase{"InnerPreciseOutOfRange", false, [](Para &p) { Attr<int64_t>(p, PRECISE, 4); }},
        InvalidTilingCase{"MissingQueryRope", false, [](Para &p) { Shape(p, QUERY_ROPE, {}); }},
        InvalidTilingCase{"MissingKey", false, [](Para &p) { Shape(p, KEY, {}); }},
        InvalidTilingCase{"MultipleDynamicKeys", false, [](Para &p) {
            p.inputTensorDesc_.insert(p.inputTensorDesc_.begin() + KEY, p.inputTensorDesc_[KEY]);
            p.inputInstanceNum_ = {1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1};
            p.outputInstanceNum_ = {1, 1};
        }},
        InvalidTilingCase{"MultipleDynamicValues", false, [](Para &p) {
            p.inputTensorDesc_.insert(p.inputTensorDesc_.begin() + VALUE, p.inputTensorDesc_[VALUE]);
            p.inputInstanceNum_ = {1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1};
            p.outputInstanceNum_ = {1, 1};
        }},
        InvalidTilingCase{"MissingMetadata", false, [](Para &p) { Shape(p, METADATA, {}); }},
        InvalidTilingCase{"MetadataDtype", false, [](Para &p) { p.inputTensorDesc_[METADATA].dtype_ = ge::DT_INT64; }},
        InvalidTilingCase{"MetadataLength", false, [](Para &p) { Shape(p, METADATA, {479}); }},
        InvalidTilingCase{"MetadataRank", false, [](Para &p) { Shape(p, METADATA, {1, 480}); }},
        InvalidTilingCase{"QueryDtype", false, [](Para &p) { p.inputTensorDesc_[QUERY].dtype_ = ge::DT_FLOAT; }},
        InvalidTilingCase{"KeyDtypeMismatch", false, [](Para &p) { p.inputTensorDesc_[KEY].dtype_ = ge::DT_BF16; }},
        InvalidTilingCase{"ValueDtypeMismatch", false, [](Para &p) { p.inputTensorDesc_[VALUE].dtype_ = ge::DT_BF16; }},
        InvalidTilingCase{"RopeDtypeMismatch", false, [](Para &p) {
            p.inputTensorDesc_[QUERY_ROPE].dtype_ = ge::DT_BF16;
        }},
        InvalidTilingCase{"OutputDtypeMismatch", false, [](Para &p) { p.outputTensorDesc_[0].dtype_ = ge::DT_BF16; }},
        InvalidTilingCase{"LseDtype", false, [](Para &p) { p.outputTensorDesc_[1].dtype_ = ge::DT_FLOAT16; }},
        InvalidTilingCase{"QueryHeadDim", false, [](Para &p) { Shape(p, QUERY, {2, 8, 1, 256}); }},
        InvalidTilingCase{"QueryRopeHeadDim", false, [](Para &p) { Shape(p, QUERY_ROPE, {2, 8, 1, 32}); }},
        InvalidTilingCase{"QueryRopeSequenceMismatch", false, [](Para &p) { Shape(p, QUERY_ROPE, {2, 8, 2, 64}); }},
        InvalidTilingCase{"QueryNegativeDimension", false, [](Para &p) { Shape(p, QUERY, {2, 8, -1, 512}); }},
        InvalidTilingCase{"QueryDimensionOverflow", false, [](Para &p) {
            Shape(p, QUERY, {2, 8, int64_t{UINT32_MAX} + 1, 512});
        }},
        InvalidTilingCase{"UnsupportedLayoutSuffix", false, [](Para &p) { Attr<std::string>(p, LAYOUT, "BNSD_BNSD"); }},
        InvalidTilingCase{"KvBatchMismatch", false, [](Para &p) { Shape(p, KEY, {3, 1, 128, 512}); }},
        InvalidTilingCase{"KvSequenceMismatch", false, [](Para &p) { Shape(p, VALUE, {2, 1, 127, 512}); }},
        InvalidTilingCase{"KvRopeHeadDim", false, [](Para &p) { Shape(p, KEY_ROPE, {2, 1, 128, 32}); }},
        InvalidTilingCase{"UnexpectedScale", false, [](Para &p) { Shape(p, KEY_SCALE, {1}); }},
        InvalidTilingCase{"NonquantQueryMode", false, [](Para &p) { Attr<int64_t>(p, QUERY_MODE, 3); }},
        InvalidTilingCase{"KeyAntiquantMode", false, [](Para &p) { Attr<int64_t>(p, KEY_MODE, 1); }},
        InvalidTilingCase{"ValueAntiquantMode", false, [](Para &p) { Attr<int64_t>(p, VALUE_MODE, 1); }},
        InvalidTilingCase{"SequenceDtype", false, [](Para &p) { p.inputTensorDesc_[SEQ_Q] = Desc({2}, ge::DT_INT32); }},
        InvalidTilingCase{"SequenceRank", false, [](Para &p) { Shape(p, SEQ_KV, {1, 2}); }},
        InvalidTilingCase{"SequenceCountBelowBatch", false, [](Para &p) {
            p = MakeTilingCase("BNSD", ge::DT_FLOAT16, 4);
            Shape(p, SEQ_Q, {2});
        }},
        InvalidTilingCase{"TndMissingQuerySequence", false, [](Para &p) {
            p = MakeTilingCase("TND");
            Shape(p, SEQ_Q, {});
        }},
        InvalidTilingCase{"TndMissingKvSequence", false, [](Para &p) {
            p = MakeTilingCase("TND");
            Shape(p, SEQ_KV, {});
        }},
        InvalidTilingCase{"TndKvCountMismatch", false, [](Para &p) {
            p = MakeTilingCase("TND");
            Shape(p, SEQ_KV, {3});
        }},
        InvalidTilingCase{"OutputShapeMismatch", false, [](Para &p) {
            p.outputTensorDesc_[0] = Desc({2, 8, 1, 256}, ge::DT_FLOAT16);
        }},
        InvalidTilingCase{"LseShapeMismatch", false, [](Para &p) {
            p.outputTensorDesc_[1] = Desc({2, 8, 1}, ge::DT_FLOAT);
        }},
        InvalidTilingCase{"MaskDtype", false, [](Para &p) {
            p.inputTensorDesc_[MASK] = Desc({2, 1, 128}, ge::DT_FLOAT);
        }},
        InvalidTilingCase{"OrdinaryMaskRankTwo", false, [](Para &p) {
            p.inputTensorDesc_[MASK] = Desc({1, 128}, ge::DT_BOOL);
        }},
        InvalidTilingCase{"MaskHeadAxis", false, [](Para &p) {
            p.inputTensorDesc_[MASK] = Desc({2, 8, 1, 128}, ge::DT_BOOL);
        }},
        InvalidTilingCase{"MaskBatch", false, [](Para &p) {
            p.inputTensorDesc_[MASK] = Desc({3, 1, 128}, ge::DT_BOOL);
        }},
        InvalidTilingCase{"MaskQueryExtent", false, [](Para &p) {
            p = MakeTilingCase("BNSD", ge::DT_FLOAT16, 2, 8, 3);
            p.inputTensorDesc_[MASK] = Desc({2, 2, 128}, ge::DT_BOOL);
        }},
        InvalidTilingCase{"SparseWithoutMask", false, [](Para &p) { Attr<int64_t>(p, SPARSE, 3); }},
        InvalidTilingCase{"CompressedMaskSize", false, [](Para &p) {
            Attr<int64_t>(p, SPARSE, 3);
            p.inputTensorDesc_[MASK] = Desc({2048, 1024}, ge::DT_BOOL);
        }},
        InvalidTilingCase{"NegativeTokenWindow", false, [](Para &p) {
            p.inputTensorDesc_[MASK] = Desc({2, 1, 128}, ge::DT_BOOL);
            Attr<int64_t>(p, PRE_TOKENS, -2);
            Attr<int64_t>(p, NEXT_TOKENS, 1);
        }},
        InvalidTilingCase{"UnsupportedSoc", false, [](Para &p) { p.socVersion_ = "Ascend950"; }},
        InvalidTilingCase{"ZeroCores", false, [](Para &p) { p.socInfoString_ = Platform(0, 0); }},
        InvalidTilingCase{"TooManyCores", false, [](Para &p) { p.socInfoString_ = Platform(26, 52); }},
        InvalidTilingCase{"InvalidCvRatio", false, [](Para &p) { p.socInfoString_ = Platform(24, 47); }},
        InvalidTilingCase{"InsufficientTilingBuffer", false, [](Para &p) {
            p.tilingDataSize_ = sizeof(MlaStaticTiling) - 1;
        }},
        InvalidTilingCase{"Int8MissingScale", true, [](Para &p) { Shape(p, QUERY_SCALE, {}); }},
        InvalidTilingCase{"Int8ScaleDtype", true, [](Para &p) { p.inputTensorDesc_[KEY_SCALE].dtype_ = ge::DT_BF16; }},
        InvalidTilingCase{"Int8QueryScaleShape", true, [](Para &p) { Shape(p, QUERY_SCALE, {4, 8}); }},
        InvalidTilingCase{"Int8KvScaleShape", true, [](Para &p) { Shape(p, VALUE_SCALE, {2}); }},
        InvalidTilingCase{"Int8QueryMode", true, [](Para &p) { Attr<int64_t>(p, QUERY_MODE, 0); }},
        InvalidTilingCase{"Int8RopeDtype", true, [](Para &p) { p.inputTensorDesc_[KEY_ROPE].dtype_ = ge::DT_FLOAT16; }},
        InvalidTilingCase{"Int8InnerPrecise", true, [](Para &p) { Attr<int64_t>(p, PRECISE, 2); }},
        InvalidTilingCase{"Int8NonPowerOfTwoHeads", true, [](Para &p) {
            p = MakeTilingCase("BSND", ge::DT_INT8, 4, 3, 1, 512, 128, true);
        }},
        InvalidTilingCase{"Int8BlockMustBe128", true, [](Para &p) {
            p = MakeTilingCase("BSND", ge::DT_INT8, 4, 8, 1, 512, 64, true);
        }},
        InvalidTilingCase{"Int8TndTokenOverflow", true, [](Para &p) { p = QuantCase("TND", 1048577); }},
        InvalidTilingCase{"Int8RequiresPa", true, [](Para &p) { Shape(p, BLOCK_TABLE, {}); }},
        InvalidTilingCase{"PageMissingKvSequence", true, [](Para &p) { Shape(p, SEQ_KV, {}); }},
        InvalidTilingCase{"PageEmptyKvSequence", true, [](Para &p) { Shape(p, SEQ_KV, {0}); }},
        InvalidTilingCase{"PageBlockTableDtype", true, [](Para &p) {
            p.inputTensorDesc_[BLOCK_TABLE].dtype_ = ge::DT_INT64;
        }},
        InvalidTilingCase{"PageBlockTableRank", true, [](Para &p) { Shape(p, BLOCK_TABLE, {4}); }},
        InvalidTilingCase{"PageBlockTableBatch", true, [](Para &p) { Shape(p, BLOCK_TABLE, {3, 4}); }},
        InvalidTilingCase{"PageZeroBlockSize", true, [](Para &p) { Attr<int64_t>(p, BLOCK_SIZE, 0); }},
        InvalidTilingCase{"PageUnalignedBlockSize", true, [](Para &p) { Attr<int64_t>(p, BLOCK_SIZE, 17); }},
        InvalidTilingCase{"PageTooLargeBlockSize", true, [](Para &p) { Attr<int64_t>(p, BLOCK_SIZE, 1040); }},
        InvalidTilingCase{"PageCapacityOverflow", true, [](Para &p) {
            Shape(p, BLOCK_TABLE, {4, int64_t{UINT32_MAX}});
        }},
        InvalidTilingCase{"Int8NzInnerAxis", true, [](Para &p) { Shape(p, KEY, {16, 1, 16, 128, 16}); }},
        InvalidTilingCase{"Int8BnsdQuery", true, [](Para &p) { p = QuantCase("BNSD"); }},
        InvalidTilingCase{"Int8MultiTokenWithoutMask", true, [](Para &p) { p = QuantCase("BSND", 2); }},
        InvalidTilingCase{"Int8KvCountBelowBatch", true, [](Para &p) { Shape(p, SEQ_KV, {2}); }},
        InvalidTilingCase{"Int8CompressedMaskRank", true, [](Para &p) {
            p = QuantCase("BSND", 2);
            Attr<int64_t>(p, SPARSE, 3);
            p.inputTensorDesc_[MASK] = Desc({1, 2048, 2048}, ge::DT_BOOL);
        }},
        InvalidTilingCase{"Int8SingleTokenWithMask", true, [](Para &p) {
            Attr<int64_t>(p, SPARSE, 3);
            p.inputTensorDesc_[MASK] = Desc({2048, 2048}, ge::DT_BOOL);
        }}),
    [](const testing::TestParamInfo<InvalidTilingCase> &info) { return info.param.name; });

TEST(FlashMlaStaticTiling, NullContext)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("FlashMla");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    EXPECT_EQ(impl->tiling(nullptr), ge::GRAPH_FAILED);
}
} // namespace
