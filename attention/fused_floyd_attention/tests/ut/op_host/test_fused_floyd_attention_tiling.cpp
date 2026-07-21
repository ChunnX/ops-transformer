/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include "tiling/platform/platform_ascendc.h"
#include "../../../../common/include/op_host/tiling_base.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class FusedFloydAttentionTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedFloydAttentionTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedFloydAttentionTiling TearDown" << std::endl;
    }
};

TEST_F(FusedFloydAttentionTiling, fused_floyd_attention_bf16_has_mask)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        48, 24, 196608, 524288, 65536, 65536, 131072, 201326592, 24, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "FusedFloydAttention",
        {
            {{{1, 1, 16, 128, 64}, {1, 1, 16, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // q
            {{{1, 1, 16, 256, 64}, {1, 1, 16, 256, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // k1
            {{{1, 1, 16, 256, 64}, {1, 1, 16, 256, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // v1
            {{{1, 1, 256, 128, 64}, {1, 1, 256, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // k2
            {{{1, 1, 256, 128, 64}, {1, 1, 256, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND}, // v2
            {{{1, 1, 16, 1, 256}, {1, 1, 16, 1, 256}}, ge::DT_BOOL, ge::FORMAT_ND}, // atten_mask
        },
        {
            // 输出Tensor
            {{{1, 1, 16, 128, 8}, {1, 1, 16, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_max
            {{{1, 1, 16, 128, 8}, {1, 1, 16, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_sum
            {{{1, 1, 16, 128, 64}, {1, 1, 16, 128, 64}}, ge::DT_BF16, ge::FORMAT_ND}  // attention_out
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)}
        },
        &compileInfo,"Ascend910B",48,196608,8192);
    int64_t expectTilingKey = 10000000001022030943UL;
    std::string expectTilingData = "1 16 1 128 256 256 64 1 4575657222473777152 0 0 0 0 0 16973827 1099511627776 1 0 0 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 2 128 0 137438953504 1 274877907008 1 4294967297 1 42949672992 68719476740 1 4294967297 1 8 0 0 0 0 281474976776192 0 70368744308736 70368744194048 562949953437696 0 0 0 0 0 0 0 0 0 274877906945 274877907200 274877907008 274877907200 1099511627840 4294967360 4294967297 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 42949672961 4294967360 274877906945 1099511627786 4294967297 42949673024 4294967360 1099511627777 10 137438953473 274877907200 137438953536 274877907200 1099511627808 4294967360 4294967297 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 4294967297 4294967328 274877907072 1099511627777 549755813889 4294967360 4294967328 1099511627840 4 274877906945 1099511627840 274877907200 1099511627840 274877907008 4294967552 4294967297 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 137438953473 4294967360 1099511627777 274877906976 4294967297 137438953728 4294967360 274877906945 0 137438953473 1099511627840 137438953728 1099511627840 274877906976 4294967552 4294967297 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 4294967297 4294967328 1099511627840 1099511627777 549755813889 4294967360 4294967328 274877907008 4 8796093022209 4294969344 34359738376 8796093022209 4294969344 34359738376 1 0 68719476737 274877907072 4294968320 274877906945 549755813952 8796093153280 1024 1024 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {268435456};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// A5 tiling 验证用例：打印 tiling 信息到 stdout 和文件
TEST_F(FusedFloydAttentionTiling, fused_floyd_attention_a5_dump_tiling)
{
    // A5 平台参数: Ascend950PR PCIE, 28 cube / 56 vector, UB=248KB, L1=512KB, L0C=256KB
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        56, 28, 253952, 524288, 65536, 65536, 262144, 117440512, 28,
        platform_ascendc::SocVersion::ASCEND950};
    // 与 example 一致的最小 shape: B=1, H=1, N=16, M=128, K=128, D=32
    gert::TilingContextPara tilingContextPara(
        "FusedFloydAttention",
        {
            {{{1, 1, 16, 128, 32}, {1, 1, 16, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q
            {{{1, 1, 16, 128, 32}, {1, 1, 16, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // k1
            {{{1, 1, 16, 128, 32}, {1, 1, 16, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // v1
            {{{1, 1, 128, 128, 32}, {1, 1, 128, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // k2
            {{{1, 1, 128, 128, 32}, {1, 1, 128, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // v2
            {{{1, 1, 16, 1, 128}, {1, 1, 16, 1, 128}}, ge::DT_BOOL, ge::FORMAT_ND}, // atten_mask
        },
        {
            {{{1, 1, 16, 128, 8}, {1, 1, 16, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_max
            {{{1, 1, 16, 128, 8}, {1, 1, 16, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_sum
            {{{1, 1, 16, 128, 32}, {1, 1, 16, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}  // attention_out
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)}
        },
        &compileInfo, "Ascend950", 56, 253952, 8192);

    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(ok) << "A5 tiling failed";

    std::ostringstream oss;
    oss << "========== FusedFloydAttention A5 Tiling Dump ==========\n";
    oss << "[Global]\n";
    oss << "  tilingKey    : " << tilingInfo.tilingKey << "\n";
    oss << "  blockNum     : " << tilingInfo.blockNum << "\n";
    oss << "  tilingDataSize : " << tilingInfo.tilingDataSize << "\n";
    for (size_t i = 0; i < tilingInfo.workspaceSizes.size(); i++) {
        oss << "  workspace[" << i << "] : " << tilingInfo.workspaceSizes[i] << " ("
            << tilingInfo.workspaceSizes[i] / 1024 / 1024 << " MB)\n";
    }

    // tiling data 按 int64_t 数组解析（buffer 是紧凑序列化格式）
    // 字段顺序见 FusedFloydAttentionGeneralTilingData（tiling.h:346-359）:
    //   InputParams → MultiCoreParams → CoreParams → TensorSizeParams
    //   → Bmm1Tiling(TCubeTiling) → Bmm1k1Tiling → Bmm2Tiling → Bmm2v2Tiling
    //   → SoftmaxFlashTiling → transposeTilingData → ...
    const int64_t *raw = reinterpret_cast<const int64_t *>(tilingInfo.tilingData.get());
    size_t totalInt64 = tilingInfo.tilingDataSize / sizeof(int64_t);

    oss << "\n[Raw Tiling Data (int64_t array, " << totalInt64 << " elements)]\n";
    oss << "  字段顺序: InputParams → MultiCoreParams → CoreParams → TensorSizeParams\n";
    oss << "           → Bmm1Tiling → Bmm1k1Tiling → Bmm2Tiling → Bmm2v2Tiling\n";
    oss << "           → SoftmaxFlashTiling → transposeTilingData → transposeTilingDataTailCore → dropmaskParams\n\n";
    for (size_t i = 0; i < totalInt64; i++) {
        oss << "  [" << std::setw(3) << i << "] " << raw[i] << "\n";
    }

    oss << "\n========== End of Tiling Dump ==========\n";

    std::cout << oss.str();
    std::ofstream ofs("/mnt/workspace/gitCode/cannbot-skills/docs/fused_floyd_attention_a5_tiling_dump.txt");
    ofs << oss.str();
    ofs.close();
    std::cout << "[INFO] Tiling dump saved to docs/fused_floyd_attention_a5_tiling_dump.txt" << std::endl;
}

// A3 tiling 验证用例：用与 A5 相同的 shape，打印 tiling 信息便于对比
TEST_F(FusedFloydAttentionTiling, fused_floyd_attention_a3_dump_tiling)
{
    // A3 平台参数: Ascend910_93, 24 cube / 48 vector, UB=192KB, L1=512KB, L0C=128KB
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        48, 24, 196608, 524288, 65536, 65536, 131072, 201326592, 24,
        platform_ascendc::SocVersion::ASCEND910B};
    // 与 A5 用例完全相同的 shape: B=1, H=1, N=16, M=128, K=128, D=32
    gert::TilingContextPara tilingContextPara(
        "FusedFloydAttention",
        {
            {{{1, 1, 16, 128, 32}, {1, 1, 16, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q
            {{{1, 1, 16, 128, 32}, {1, 1, 16, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // k1
            {{{1, 1, 16, 128, 32}, {1, 1, 16, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // v1
            {{{1, 1, 128, 128, 32}, {1, 1, 128, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // k2
            {{{1, 1, 128, 128, 32}, {1, 1, 128, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // v2
            {{{1, 1, 16, 1, 128}, {1, 1, 16, 1, 128}}, ge::DT_BOOL, ge::FORMAT_ND}, // atten_mask
        },
        {
            {{{1, 1, 16, 128, 8}, {1, 1, 16, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_max
            {{{1, 1, 16, 128, 8}, {1, 1, 16, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_sum
            {{{1, 1, 16, 128, 32}, {1, 1, 16, 128, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND}  // attention_out
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)}
        },
        &compileInfo, "Ascend910B", 48, 196608, 8192);

    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(ok) << "A3 tiling failed";

    std::ostringstream oss;
    oss << "========== FusedFloydAttention A3 Tiling Dump ==========\n";
    oss << "[Global]\n";
    oss << "  tilingKey    : " << tilingInfo.tilingKey << "\n";
    oss << "  blockNum     : " << tilingInfo.blockNum << "\n";
    oss << "  tilingDataSize : " << tilingInfo.tilingDataSize << "\n";
    for (size_t i = 0; i < tilingInfo.workspaceSizes.size(); i++) {
        oss << "  workspace[" << i << "] : " << tilingInfo.workspaceSizes[i] << " ("
            << tilingInfo.workspaceSizes[i] / 1024 / 1024 << " MB)\n";
    }

    const int64_t *raw = reinterpret_cast<const int64_t *>(tilingInfo.tilingData.get());
    size_t totalInt64 = tilingInfo.tilingDataSize / sizeof(int64_t);

    oss << "\n[Raw Tiling Data (int64_t array, " << totalInt64 << " elements)]\n";
    oss << "  字段顺序: InputParams → MultiCoreParams → CoreParams → TensorSizeParams\n";
    oss << "           → Bmm1Tiling → Bmm1k1Tiling → Bmm2Tiling → Bmm2v2Tiling\n";
    oss << "           → SoftmaxFlashTiling → transposeTilingData → transposeTilingDataTailCore → dropmaskParams\n\n";
    for (size_t i = 0; i < totalInt64; i++) {
        oss << "  [" << std::setw(3) << i << "] " << raw[i] << "\n";
    }

    oss << "\n========== End of Tiling Dump ==========\n";

    std::cout << oss.str();
    std::ofstream ofs("/mnt/workspace/gitCode/cannbot-skills/docs/fused_floyd_attention_a3_tiling_dump.txt");
    ofs << oss.str();
    ofs.close();
    std::cout << "[INFO] Tiling dump saved to docs/fused_floyd_attention_a3_tiling_dump.txt" << std::endl;
}

TEST_F(FusedFloydAttentionTiling, fused_floyd_attention_fp16_has_mask)
{
    Ops::Transformer::OpTiling::FlashAttentionScoreGradCompileInfo compileInfo = {
        48, 24, 196608, 524288, 65536, 65536, 131072, 201326592, 24, platform_ascendc::SocVersion::ASCEND910B};
    gert::TilingContextPara tilingContextPara(
        "FusedFloydAttention",
        {
            {{{1, 1, 16, 128, 64}, {1, 1, 16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // q
            {{{1, 1, 16, 256, 64}, {1, 1, 16, 256, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // k1
            {{{1, 1, 16, 256, 64}, {1, 1, 16, 256, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // v1
            {{{1, 1, 256, 128, 64}, {1, 1, 256, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // k2
            {{{1, 1, 256, 128, 64}, {1, 1, 256, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // v2
            {{{1, 1, 16, 1, 256}, {1, 1, 16, 1, 256}}, ge::DT_BOOL, ge::FORMAT_ND}, // atten_mask
        },
        {
            // 输出Tensor
            {{{1, 1, 16, 128, 8}, {1, 1, 16, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_max
            {{{1, 1, 16, 128, 8}, {1, 1, 16, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_sum
            {{{1, 1, 16, 128, 64}, {1, 1, 16, 128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND}  // attention_out
        },
        {
            {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)}
        },
        &compileInfo,"Ascend910B",48,196608,8192);
    int64_t expectTilingKey = 10000000001022030943UL;
    std::string expectTilingData = "1 16 1 128 256 256 64 1 4575657222473777152 0 0 0 0 0 16973827 1099511627776 1 0 0 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 274877907008 2 128 0 137438953504 1 274877907008 1 4294967297 1 42949672992 68719476740 1 4294967297 1 8 0 0 0 0 281474976776192 0 70368744308736 70368744194048 562949953437696 0 0 0 0 0 0 0 0 0 274877906945 274877907200 274877907008 274877907200 1099511627840 4294967360 4294967297 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 42949672961 4294967360 274877906945 1099511627786 4294967297 42949673024 4294967360 1099511627777 10 137438953473 274877907200 137438953536 274877907200 1099511627808 4294967360 4294967297 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 4294967297 4294967328 274877907072 1099511627777 549755813889 4294967360 4294967328 1099511627840 4 274877906945 1099511627840 274877907200 1099511627840 274877907008 4294967552 4294967297 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 137438953473 4294967360 1099511627777 274877906976 4294967297 137438953728 4294967360 274877906945 0 137438953473 1099511627840 137438953728 1099511627840 274877906976 4294967552 4294967297 1 0 2250700302057472 131072 4294967297 4294967297 4294967297 0 8589934594 4294967297 4294967328 1099511627840 1099511627777 549755813889 4294967360 4294967328 274877907008 4 8796093022209 4294969344 34359738376 8796093022209 4294969344 34359738376 1 0 68719476737 274877907072 4294968320 274877906945 549755813952 8796093153280 1024 1024 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {268435456};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
