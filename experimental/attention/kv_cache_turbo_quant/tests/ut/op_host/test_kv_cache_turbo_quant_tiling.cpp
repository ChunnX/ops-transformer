/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_kv_cache_turbo_quant_tiling.cpp
 * \brief KvCacheTurboQuant Tiling 单元测试（op_host 层）
 *
 * 覆盖：
 *   - 合法路径：任务书 5 个自测 case 的典型 shape（decode b1/b64、prefill t2048 × mse_bits 2/3/4）
 *     与 head 数上下界（4 / 32）；断言 tiling 成功、TilingKey = 0、workspace 非空
 *   - 非法路径：与 `kv_cache_turbo_quant_tiling.cpp` 的校验一一对应，断言 tiling 拒绝（返回失败）
 *       `kv_vectors` rank≠3 / head_dim≠128 / head 数 <4 或 >32 / 空 tensor（tokens=0）/ dtype≠BF16
 *       `rotation_matrix`、`qjl_matrix` shape≠[128,128] 或 dtype≠FP32
 *       `mse_bits` ∉ {2,3,4}
 *
 * 说明：本用例只消费 tiling 输出（TilingKey / workspace / TilingData），不执行 kernel；
 * 数值正确性由 `tests/` 下的端到端对拍与精度套件保证（见 `tests/README.md`）。
 */

#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../../op_host/kv_cache_turbo_quant_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

namespace {
constexpr int64_t kHeadDim = 128;
constexpr int64_t kQjlBytesPerRow = 16; // ceil(head_dim / 8)

int64_t IdxBytesPerRow(int64_t mseBits)
{
    return (kHeadDim * mseBits + 7) / 8;
}

gert::TilingContextPara MakePara(std::initializer_list<int64_t> kvDims,
                                 std::initializer_list<int64_t> rotDims,
                                 std::initializer_list<int64_t> qjlDims,
                                 int64_t mseBits,
                                 ge::DataType kvDtype = DT_BF16,
                                 ge::DataType rotDtype = DT_FLOAT,
                                 ge::DataType qjlDtype = DT_FLOAT)
{
    // CompileInfo 需覆盖 ExecuteTiling 调用期；内容不被 TilingFunc 读取（平台信息经 PlatformInfo 提供）
    static optiling::KvCacheTurboQuantCompileInfo compileInfo;

    std::vector<int64_t> kv(kvDims);
    const int64_t tokens = kv.size() >= 3 ? kv[0] : 1;
    const int64_t heads = kv.size() >= 2 ? kv[1] : 1;
    const int64_t idxBytes = IdxBytesPerRow(mseBits);

    return gert::TilingContextPara(
        "KvCacheTurboQuant",
        {
            {gert::StorageShape(kvDims, kvDims), kvDtype, FORMAT_ND},
            {gert::StorageShape(rotDims, rotDims), rotDtype, FORMAT_ND},
            {gert::StorageShape(qjlDims, qjlDims), qjlDtype, FORMAT_ND},
        },
        {
            {gert::StorageShape({tokens, heads, idxBytes}, {tokens, heads, idxBytes}), DT_UINT8, FORMAT_ND},
            {gert::StorageShape({tokens, heads, kQjlBytesPerRow}, {tokens, heads, kQjlBytesPerRow}), DT_UINT8,
             FORMAT_ND},
            {gert::StorageShape({tokens, heads}, {tokens, heads}), DT_BF16, FORMAT_ND},
            {gert::StorageShape({tokens, heads}, {tokens, heads}), DT_BF16, FORMAT_ND},
        },
        {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(mseBits)}},
        &compileInfo);
}

const std::initializer_list<int64_t> kMatrixDims{kHeadDim, kHeadDim};
} // namespace

class KvCacheTurboQuantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "KvCacheTurboQuantTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KvCacheTurboQuantTilingTest TearDown" << std::endl;
    }
};

// ============================================================================
// 1. 合法路径（任务书 5 个自测 case + head 数上下界）
// ============================================================================
TEST_F(KvCacheTurboQuantTilingTest, DecodeB1Bits3)
{
    TilingInfo tilingInfo;
    auto para = MakePara({1, 8, kHeadDim}, kMatrixDims, kMatrixDims, 3);
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    ASSERT_FALSE(tilingInfo.workspaceSizes.empty());
    EXPECT_GT(tilingInfo.workspaceSizes[0], 0);
}

TEST_F(KvCacheTurboQuantTilingTest, DecodeB64Bits3)
{
    TilingInfo tilingInfo;
    auto para = MakePara({64, 8, kHeadDim}, kMatrixDims, kMatrixDims, 3);
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
}

TEST_F(KvCacheTurboQuantTilingTest, PrefillT2048Bits3)
{
    TilingInfo tilingInfo;
    auto para = MakePara({2048, 8, kHeadDim}, kMatrixDims, kMatrixDims, 3);
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    ASSERT_FALSE(tilingInfo.workspaceSizes.empty());
    // workspace = libapi 保留区 + 中间量（u/y/r/p/norm），应显著大于 0
    EXPECT_GT(tilingInfo.workspaceSizes[0], 1024);
    EXPECT_GT(tilingInfo.tilingDataSize, 0);
}

TEST_F(KvCacheTurboQuantTilingTest, MseBits2)
{
    TilingInfo tilingInfo;
    auto para = MakePara({2048, 8, kHeadDim}, kMatrixDims, kMatrixDims, 2);
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
}

TEST_F(KvCacheTurboQuantTilingTest, MseBits4)
{
    TilingInfo tilingInfo;
    auto para = MakePara({2048, 8, kHeadDim}, kMatrixDims, kMatrixDims, 4);
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
}

TEST_F(KvCacheTurboQuantTilingTest, HeadsLowerBound4)
{
    TilingInfo tilingInfo;
    auto para = MakePara({32, 4, kHeadDim}, kMatrixDims, kMatrixDims, 3);
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
}

TEST_F(KvCacheTurboQuantTilingTest, HeadsUpperBound32)
{
    TilingInfo tilingInfo;
    auto para = MakePara({8, 32, kHeadDim}, kMatrixDims, kMatrixDims, 3);
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0);
}

// ============================================================================
// 2. 非法路径：kv_vectors
// ============================================================================
TEST_F(KvCacheTurboQuantTilingTest, InvalidHeadsTooSmall)
{
    TilingInfo tilingInfo;
    auto para = MakePara({8, 2, kHeadDim}, kMatrixDims, kMatrixDims, 3);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

TEST_F(KvCacheTurboQuantTilingTest, InvalidHeadsTooLarge)
{
    TilingInfo tilingInfo;
    auto para = MakePara({8, 64, kHeadDim}, kMatrixDims, kMatrixDims, 3);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

TEST_F(KvCacheTurboQuantTilingTest, InvalidKvRank2)
{
    TilingInfo tilingInfo;
    auto para = MakePara({8, kHeadDim}, kMatrixDims, kMatrixDims, 3);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

TEST_F(KvCacheTurboQuantTilingTest, InvalidHeadDim64)
{
    TilingInfo tilingInfo;
    auto para = MakePara({8, 8, 64}, kMatrixDims, kMatrixDims, 3);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

TEST_F(KvCacheTurboQuantTilingTest, InvalidEmptyTokens)
{
    TilingInfo tilingInfo;
    auto para = MakePara({0, 8, kHeadDim}, kMatrixDims, kMatrixDims, 3);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// 行数上界：token×head 必须落在「16 对齐 + usedCoreNum 分块都不回绕」的范围内。
// 536870792 × 8 = 4,294,966,336：仍然 <= UINT32_MAX（旧的 `> UINT32_MAX` 拦不住），
// 但 rowsPerCore × usedCoreNum 会越过 uint32 → alignedRows 回绕成 0 → 工作区退化、
// 内核按 0 行工作区写 3300 万行。收紧后的上界必须在 tiling 阶段就拒掉它。
// 注意：上界 = UINT32_MAX − 16×aicNum，本用例的输入按 UT 平台桩的 coreNum=64
// （tests/ut/framework_normal/common/tiling_context_faker.h 的默认值）落在界外；
// 若平台桩核数变成 <= 41，该输入会落到界内、此用例将失去区分度。
TEST_F(KvCacheTurboQuantTilingTest, InvalidTotalRowsTooLarge)
{
    TilingInfo tilingInfo;
    auto para = MakePara({536870792, 8, kHeadDim}, kMatrixDims, kMatrixDims, 3);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

TEST_F(KvCacheTurboQuantTilingTest, InvalidKvDtypeFp32)
{
    TilingInfo tilingInfo;
    auto para = MakePara({8, 8, kHeadDim}, kMatrixDims, kMatrixDims, 3, DT_FLOAT, DT_FLOAT, DT_FLOAT);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

// ============================================================================
// 3. 非法路径：矩阵与属性
// ============================================================================
TEST_F(KvCacheTurboQuantTilingTest, InvalidRotationShape)
{
    TilingInfo tilingInfo;
    auto para = MakePara({8, 8, kHeadDim}, {64, 64}, kMatrixDims, 3);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

TEST_F(KvCacheTurboQuantTilingTest, InvalidQjlDtypeBf16)
{
    TilingInfo tilingInfo;
    auto para = MakePara({8, 8, kHeadDim}, kMatrixDims, kMatrixDims, 3, DT_BF16, DT_FLOAT, DT_BF16);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

TEST_F(KvCacheTurboQuantTilingTest, InvalidMseBits5)
{
    TilingInfo tilingInfo;
    auto para = MakePara({8, 8, kHeadDim}, kMatrixDims, kMatrixDims, 5);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}

TEST_F(KvCacheTurboQuantTilingTest, InvalidMseBits1)
{
    TilingInfo tilingInfo;
    auto para = MakePara({8, 8, kHeadDim}, kMatrixDims, kMatrixDims, 1);
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo));
}
