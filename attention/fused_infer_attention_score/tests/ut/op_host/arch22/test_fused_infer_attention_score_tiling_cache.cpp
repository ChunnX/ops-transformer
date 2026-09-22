// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include <gtest/gtest.h>
#include <map>
#include <thread>
#include "exe_graph/runtime/tensor.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "../../../../op_host/fused_infer_attention_score_tiling_cache.h"

namespace optiling {
namespace {
using Cache = FusedInferAttentionScoreTilingCache;
using Recorder = FiaTilingScheduleRecorder;

struct CacheContext {
    fe::PlatFormInfos platform;
    gert::TilingContextFaker faker;
    decltype(gert::TilingData::CreateCap(4096)) data = gert::TilingData::CreateCap(4096);
    decltype(gert::ContinuousVector::Create<size_t>(1)) workspace =
        gert::ContinuousVector::Create<size_t>(1);
    std::vector<std::unique_ptr<gert::TensorV2>> tensors;
    int64_t qLength = 1;
    int64_t kvLength = 4;

    gert::Tensor *Tensor(const gert::StorageShape &shape, ge::DataType dtype, void *value = nullptr)
    {
        auto tensor = std::make_unique<gert::TensorV2>(
            shape, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
            gert::TensorPlacement::kOnHost, dtype, value);
        auto *ptr = reinterpret_cast<gert::Tensor *>(tensor.get());
        tensors.push_back(std::move(tensor));
        return ptr;
    }

    explicit CacheContext(bool lse = false, uint32_t dynamicCount = 1, int64_t hostCount = 1,
                          int64_t secondKeyLength = 4)
    {
        platform.Init();
        std::map<std::string, std::string> soc = {
            {"ai_core_cnt", "20"}, {"cube_core_cnt", "20"}, {"vector_core_cnt", "40"},
            {"core_type_list", "CubeCore,VectorCore"}, {"l2_size", "201326592"}};
        std::map<std::string, std::string> spec = {
            {"ub_size", "196608"}, {"l1_size", "524288"}, {"l0_a_size", "65536"},
            {"l0_b_size", "65536"}, {"l0_c_size", "131072"}};
        std::map<std::string, std::string> version = {
            {"NpuArch", "2201"}, {"Short_SoC_version", "Ascend910B"}};
        platform.SetPlatformRes("SoCInfo", soc);
        platform.SetPlatformRes("AICoreSpec", spec);
        platform.SetPlatformRes("version", version);
        platform.SetCoreNumByCoreType("AICore");
        std::vector<uint32_t> counts(31, 0);
        counts[0] = counts[5] = counts[6] = 1;
        counts[1] = counts[2] = dynamicCount;
        std::vector<gert::Tensor *> inputs = {
            Tensor({{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT16)};
        for (uint32_t i = 0; i < dynamicCount * 2; ++i) {
            const int64_t length = i == 1 ? secondKeyLength : 4;
            inputs.push_back(Tensor({{1, 1, length, 128}, {1, 1, length, 128}}, ge::DT_FLOAT16));
        }
        inputs.push_back(Tensor({{1}, {1}}, ge::DT_INT64, &qLength));
        inputs.push_back(Tensor({{hostCount}, {hostCount}}, ge::DT_INT64, &kvLength));
        std::vector<gert::Tensor *> outputs = {
            Tensor({{1, 1, 1, 128}, {1, 1, 1, 128}}, ge::DT_FLOAT16),
            Tensor({{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_FLOAT)};
        faker.SetOpType("FusedInferAttentionScore").IrInstanceNum(counts, {1, 1})
            .InputTensors(inputs).OutputTensors(outputs)
            .Attr("num_heads", int64_t{1}).Attr("scale", 1.0f)
            .Attr("pre_tokens", int64_t{2147483647}).Attr("next_tokens", int64_t{2147483647})
            .Attr("input_layout", ge::AscendString("BNSD"))
            .Attr("num_key_value_heads", int64_t{1}).Attr("sparse_mode", int64_t{0})
            .Attr("inner_precise", int64_t{0}).Attr("block_size", int64_t{0})
            .Attr("antiquant_mode", int64_t{0}).Attr("softmax_lse_flag", lse)
            .Attr("key_antiquant_mode", int64_t{0}).Attr("value_antiquant_mode", int64_t{0})
            .Attr("query_quant_mode", int64_t{0}).Attr("pse_type", int64_t{0})
            .Attr("out_dtype", int64_t{0})
            .PlatformInfo(&platform).DeterministicInfo(0).TilingData(data.get())
            .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace.get()));
    }
};

TEST(FiaTilingCache, HostValuesAndPlatformSeparateEntriesWithoutUsingAddresses)
{
    CacheContext first, second;
    auto a = first.faker.Build();
    auto b = second.faker.Build();
    std::string keyA, keyB;
    ASSERT_TRUE(Cache::BuildKey(a.GetContext(), keyA));
    ASSERT_TRUE(Cache::BuildKey(b.GetContext(), keyB));
    EXPECT_EQ(keyA, keyB);
    second.kvLength = 2;
    ASSERT_TRUE(Cache::BuildKey(b.GetContext(), keyB));
    EXPECT_NE(keyA, keyB);
    second.kvLength = 4;
    std::map<std::string, std::string> soc = {
        {"ai_core_cnt", "10"}, {"cube_core_cnt", "10"}, {"vector_core_cnt", "20"},
        {"core_type_list", "CubeCore,VectorCore"}, {"l2_size", "201326592"}};
    second.platform.SetPlatformRes("SoCInfo", soc);
    ASSERT_TRUE(Cache::BuildKey(b.GetContext(), keyB));
    EXPECT_NE(keyA, keyB);
}

TEST(FiaTilingCache, BoolAttributeAndDynamicTailParticipateInKey)
{
    CacheContext first(false, 2), second(true, 2);
    auto a = first.faker.Build();
    auto b = second.faker.Build();
    std::string keyA, keyB;
    ASSERT_TRUE(Cache::BuildKey(a.GetContext(), keyA));
    ASSERT_TRUE(Cache::BuildKey(b.GetContext(), keyB));
    EXPECT_NE(keyA, keyB);

    // Change the second K tensor, beyond the first element of a dynamic input.
    CacheContext third(false, 2, 1, 8);
    auto c = third.faker.Build();
    ASSERT_TRUE(Cache::BuildKey(c.GetContext(), keyB));
    EXPECT_NE(keyA, keyB);
}

TEST(FiaTilingCache, OversizedHostValuesAndDynamicListsBypassInsteadOfTruncating)
{
    // Only one value is backed by memory: BuildKey must reject before reading it.
    CacheContext largeHost(false, 1, 65537), largeList(false, 1025);
    auto a = largeHost.faker.Build();
    auto b = largeList.faker.Build();
    std::string key;
    EXPECT_FALSE(Cache::BuildKey(a.GetContext(), key));
    EXPECT_FALSE(Cache::BuildKey(b.GetContext(), key));
}

TEST(FiaTilingCache, StrideChangesSeparateEntries)
{
    CacheContext first, second;
    auto setStride = [](gert::Stride &stride, int64_t dim0) {
        stride.SetDimNum(4);
        stride.SetStride(0, dim0);
        stride.SetStride(1, 512);
        stride.SetStride(2, 128);
        stride.SetStride(3, 1);
    };
    setStride(first.tensors[1]->MutableStride(), 512);
    setStride(second.tensors[1]->MutableStride(), 1024);
    auto a = first.faker.Build();
    auto b = second.faker.Build();
    std::string keyA, keyB;
    ASSERT_TRUE(Cache::BuildKey(a.GetContext(), keyA));
    ASSERT_TRUE(Cache::BuildKey(b.GetContext(), keyB));
    EXPECT_NE(keyA, keyB);
}

TEST(FiaTilingCache, RestoreRejectsSmallBufferAndRecorderDoesNotLeakBetweenContexts)
{
    CacheContext first, second;
    auto a = first.faker.Build();
    auto b = second.faker.Build();
    Recorder::Scope outer(a.GetContext());
    ASSERT_EQ(Recorder::Set(a.GetContext(), 1), ge::GRAPH_SUCCESS);
    {
        Recorder::Scope inner(b.GetContext());
        uint32_t mode = 0;
        EXPECT_FALSE(inner.Get(mode));
        Cache::TilingResult result;
        result.tilingDataBlob.resize(4097);
        EXPECT_EQ(Cache::Restore(b.GetContext(), result), ge::GRAPH_FAILED);
    }
    ASSERT_EQ(Recorder::Set(a.GetContext(), 0), ge::GRAPH_SUCCESS);
    uint32_t mode = 1;
    ASSERT_TRUE(outer.Get(mode));
    EXPECT_EQ(mode, 0);
}

TEST(FiaTilingCache, ResultReusableInAnotherContextBeforeAnyRun)
{
    CacheContext first, second;
    auto a = first.faker.Build();
    auto b = second.faker.Build();
    std::string keyA, keyB;
    ASSERT_TRUE(Cache::BuildKey(a.GetContext(), keyA));
    ASSERT_TRUE(Cache::BuildKey(b.GetContext(), keyB));
    Cache cache;
    Cache::TilingResult snapshot;
    {
        Recorder::Scope scope(a.GetContext());
        const char blob[] = {'a', '\0', 'b'};
        std::memcpy(a.GetContext()->GetRawTilingData()->GetData(), blob, sizeof(blob));
        a.GetContext()->GetRawTilingData()->SetDataSize(sizeof(blob));
        a.GetContext()->SetTilingKey(1234);
        a.GetContext()->SetBlockDim(20);
        a.GetContext()->GetWorkspaceSizes(1)[0] = 4096;
        ASSERT_EQ(Recorder::Set(a.GetContext(), 1), ge::GRAPH_SUCCESS);
        ASSERT_TRUE(Cache::Snapshot(a.GetContext(), scope, snapshot));
        ASSERT_TRUE(cache.Add(keyA, snapshot));
    }
    // No ACLNN Run or executor finalization has taken place.
    auto hit = cache.Get(keyB);
    ASSERT_NE(hit, nullptr);
    a.GetContext()->GetRawTilingData()->GetData()[0] = 'z'; // snapshot must own a deep copy
    Recorder::Scope restored(b.GetContext());
    ASSERT_EQ(Cache::Restore(b.GetContext(), *hit), ge::GRAPH_SUCCESS);
    EXPECT_EQ(b.GetContext()->GetTilingKey(), 1234);
    EXPECT_EQ(b.GetContext()->GetBlockDim(), 20);
    EXPECT_EQ(b.GetContext()->GetWorkspaceSizes(1)[0], 4096);
    EXPECT_EQ(b.GetContext()->GetRawTilingData()->GetDataSize(), 3);
    EXPECT_EQ(std::memcmp(b.GetContext()->GetRawTilingData()->GetData(), "a\0b", 3), 0);
    uint32_t mode = 0;
    ASSERT_TRUE(restored.Get(mode));
    EXPECT_EQ(mode, 1);
    hit = nullptr;
}

TEST(FiaTilingCache, EvictionKeepsOutstandingSnapshotsValid)
{
    Cache cache(2);
    Cache::TilingResult result;
    result.tilingDataBlob = "first";
    ASSERT_TRUE(cache.Add("a", result));
    auto snapshot = cache.Get("a");
    ASSERT_TRUE(cache.Add("b", result));
    ASSERT_NE(cache.Get("a"), nullptr); // b becomes least recently used
    ASSERT_TRUE(cache.Add("c", result));
    EXPECT_EQ(cache.Get("b"), nullptr);
    ASSERT_TRUE(cache.Add("d", result)); // evict a while a reader still owns its snapshot
    EXPECT_EQ(cache.Get("a"), nullptr);
    ASSERT_NE(snapshot, nullptr);
    EXPECT_EQ(snapshot->tilingDataBlob, "first");
    Cache tooSmall(2, 1);
    EXPECT_FALSE(tooSmall.Add("a", result));
}

TEST(FiaTilingCache, ConcurrentLookupAndEviction)
{
    Cache cache(2);
    auto work = [&cache](int index) {
        Cache::TilingResult result;
        result.tilingKey = index;
        const auto key = std::to_string(index);
        for (int i = 0; i < 100; ++i) {
            EXPECT_TRUE(cache.Add(key, result));
            auto hit = cache.Get(key);
            if (hit != nullptr) { EXPECT_EQ(hit->tilingKey, index); }
        }
    };
    std::thread a(work, 1), b(work, 2), c(work, 3);
    a.join();
    b.join();
    c.join();
}
} // namespace
} // namespace optiling
