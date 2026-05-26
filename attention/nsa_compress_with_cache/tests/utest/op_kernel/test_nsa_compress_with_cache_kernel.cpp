/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include "tikicpulib.h"
#include "tiling_case_executor.h"
#include "../op_host/ncwc_tiling_ut_common.h"

using namespace ncwc_tiling_ut;

extern "C" __global__ __aicore__ void nsa_compress_with_cache(
    __gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* slotMapping, __gm__ uint8_t* outputCacheIn,
    __gm__ uint8_t* actSeqLen, __gm__ uint8_t* blockTable, __gm__ uint8_t* outputCacheOut, __gm__ uint8_t* workspace,
    __gm__ uint8_t* tiling);

using NcwcKernelFunc = void (*)(__gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*,
                                __gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*);

struct NamedKernelCase {
    std::string name;
    CaseParams params;
};

class NsaCompressWithCacheKernel : public testing::TestWithParam<NamedKernelCase> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressWithCacheKernel SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressWithCacheKernel TearDown" << std::endl;
    }
};

namespace {

size_t Product(const std::vector<int64_t>& dims)
{
    if (dims.empty()) {
        return 0;
    }
    return std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1),
                           [](size_t acc, int64_t dim) { return acc * static_cast<size_t>(dim); });
}

size_t GetElemSize(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return sizeof(float);
    }
    return sizeof(uint16_t);
}

uint8_t* AllocGm(size_t bytes)
{
    if (bytes == 0) {
        return nullptr;
    }
    return reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes));
}

void RunKernelCase(const CaseParams& params)
{
    CaseShapes shapes;
    auto para = BuildTilingContextPara(params, shapes);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    ASSERT_EQ(static_cast<uint64_t>(params.expectTilingKey), static_cast<uint64_t>(tilingInfo.tilingKey));

    const size_t elemSize = GetElemSize(params.dtype);
    const size_t inputBytes = Product(shapes.input) * elemSize;
    const size_t weightBytes = Product(shapes.weight) * elemSize;
    const size_t slotMappingBytes = Product(shapes.slotMapping) * sizeof(int32_t);
    const size_t outputCacheBytes = Product(shapes.outputCache) * elemSize;
    const size_t actSeqLenBytes = Product({static_cast<int64_t>(shapes.actSeqLenData.size())}) * sizeof(int64_t);
    const size_t blockTableBytes = Product(shapes.blockTable) * sizeof(int32_t);

    uint8_t* input = AllocGm(inputBytes);
    uint8_t* weight = AllocGm(weightBytes);
    uint8_t* slotMapping = AllocGm(slotMappingBytes);
    uint8_t* outputCacheIn = AllocGm(outputCacheBytes);
    uint8_t* actSeqLen = AllocGm(actSeqLenBytes);
    uint8_t* blockTable = AllocGm(blockTableBytes);
    uint8_t* outputCacheOut = AllocGm(outputCacheBytes);

    memcpy(slotMapping, shapes.slotMappingData.data(), slotMappingBytes);
    memcpy(actSeqLen, shapes.actSeqLenData.data(), actSeqLenBytes);
    memcpy(blockTable, shapes.blockTableData.data(), blockTableBytes);

    size_t workspaceBytes = 0;
    for (size_t ws : tilingInfo.workspaceSizes) {
        workspaceBytes += ws;
    }
    uint8_t* workspace = AllocGm(workspaceBytes);
    uint8_t* tiling = AllocGm(tilingInfo.tilingDataSize);
    memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    auto kernelFunc = reinterpret_cast<NcwcKernelFunc>(nsa_compress_with_cache);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(kernelFunc, 1, input, weight, slotMapping, outputCacheIn, actSeqLen, blockTable, outputCacheOut,
                workspace, tiling);
}

NamedKernelCase MakeCase(const char* name, CaseParams params)
{
    params.expectResult = ge::GRAPH_SUCCESS;
    return {name, params};
}

std::vector<NamedKernelCase> BuildKernelCases()
{
    const std::vector<int64_t> smallActSeq = {144, 192, 10, 45};
    const auto actSeq64 = ActSeqLenList64Batch();
    const auto actSeq32 = ActSeqLenList32Batch();

    return {
        MakeCase("NsaCompressWithCache_Case_0",
                 {4, 8, 192, ge::DT_FLOAT16, smallActSeq, "TND", 64, 16, 1, 128, 2}),
        MakeCase("NsaCompressWithCache_Case_1", {4, 8, 192, ge::DT_BF16, smallActSeq, "TND", 64, 16, 1, 128, 2}),
        MakeCase("NsaCompressWithCache_Case_2",
                 {4, 32, 192, ge::DT_FLOAT16, smallActSeq, "TND", 16, 16, 1, 128, 1}),
        MakeCase("NsaCompressWithCache_Case_3", {4, 32, 192, ge::DT_BF16, smallActSeq, "TND", 16, 16, 1, 128, 1}),
        MakeCase("NsaCompressWithCache_Case_4",
                 {4, 37, 256, ge::DT_FLOAT16, std::vector<int64_t>{144, 1, 10, 45}, "TND", 64, 16, 1, 128, 0}),
        MakeCase("NsaCompressWithCache_Case_5",
                 {4, 37, 256, ge::DT_BF16, std::vector<int64_t>{144, 1, 10, 45}, "TND", 64, 16, 1, 128, 0}),
        MakeCase("NsaCompressWithCache_Case_6",
                 {4, 1, 192, ge::DT_FLOAT16, smallActSeq, "TND", 48, 16, 1, 128, 2}),
        MakeCase("NsaCompressWithCache_Case_7", {4, 1, 192, ge::DT_BF16, smallActSeq, "TND", 48, 16, 1, 128, 2}),
        MakeCase("NsaCompressWithCache_Case_8",
                 {4, 13, 256, ge::DT_FLOAT16, smallActSeq, "TND", 48, 16, 1, 128, 0}),
        MakeCase("NsaCompressWithCache_Case_9", {4, 13, 256, ge::DT_BF16, smallActSeq, "TND", 48, 16, 1, 128, 0}),
        MakeCase("NsaCompressWithCache_Case_10",
                 {4, 64, 96, ge::DT_FLOAT16, smallActSeq, "TND", 32, 16, 1, 128, 1}),
        MakeCase("NsaCompressWithCache_Case_11", {4, 64, 96, ge::DT_BF16, smallActSeq, "TND", 32, 16, 1, 128, 1}),
        MakeCase("NsaCompressWithCache_Case_12",
                 {64, 8, 192, ge::DT_FLOAT16, actSeq64, "TND", 64, 16, 1, 128, 2}),
        MakeCase("NsaCompressWithCache_Case_13", {64, 8, 192, ge::DT_BF16, actSeq64, "TND", 64, 16, 1, 128, 2}),
        MakeCase("NsaCompressWithCache_Case_14",
                 {32, 32, 192, ge::DT_FLOAT16, actSeq64, "TND", 16, 16, 1, 128, 1}),
        MakeCase("NsaCompressWithCache_Case_15", {32, 32, 192, ge::DT_BF16, actSeq64, "TND", 16, 16, 1, 128, 1}),
        MakeCase("NsaCompressWithCache_Case_16",
                 {2, 37, 256, ge::DT_FLOAT16, std::vector<int64_t>{144, 192}, "TND", 64, 16, 1, 128, 0}),
        MakeCase("NsaCompressWithCache_Case_17",
                 {2, 37, 256, ge::DT_BF16, std::vector<int64_t>{144, 192}, "TND", 64, 16, 1, 128, 0}),
        MakeCase("NsaCompressWithCache_Case_18",
                 {64, 1, 192, ge::DT_FLOAT16, actSeq64, "TND", 48, 16, 1, 128, 2}),
        MakeCase("NsaCompressWithCache_Case_19", {64, 1, 192, ge::DT_BF16, actSeq64, "TND", 48, 16, 1, 128, 2}),
        MakeCase("NsaCompressWithCache_Case_20",
                 {4, 13, 256, ge::DT_FLOAT16, std::vector<int64_t>{144, 192, 144, 144}, "TND", 48, 16, 1, 128, 0}),
        MakeCase("NsaCompressWithCache_Case_21",
                 {4, 13, 256, ge::DT_BF16, std::vector<int64_t>{144, 192, 144, 144}, "TND", 48, 16, 1, 128, 0}),
        MakeCase("NsaCompressWithCache_Case_22",
                 {32, 64, 96, ge::DT_FLOAT16, actSeq32, "TND", 32, 16, 1, 128, 1}),
        MakeCase("NsaCompressWithCache_Case_23", {32, 64, 96, ge::DT_BF16, actSeq32, "TND", 32, 16, 1, 128, 1}),
        MakeCase("NsaCompressWithCache_Case_24",
                 {4, 32, 192, ge::DT_BF16, std::vector<int64_t>{1, 2, 3, 4}, "TND", 16, 16, 1, 128, 1}),
    };
}

}  // namespace

TEST_P(NsaCompressWithCacheKernel, RunCase)
{
    RunKernelCase(GetParam().params);
}

INSTANTIATE_TEST_SUITE_P(NsaCompressWithCache, NsaCompressWithCacheKernel, testing::ValuesIn(BuildKernelCases()),
                         [](const testing::TestParamInfo<NamedKernelCase>& info) { return info.param.name; });
