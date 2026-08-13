/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// BAB 模板 Tiling UT
//
// 覆盖场景:
//   1. fp16/bf16/fp32 三种 dtype
//   2. dCosFlag=0 (无 query/key 可选输入) 与 dCosFlag=1 (有 query/key, 触发 cosPartial 融合路径)
//   3. nQ == nK 与 nQ != nK (触发 SET/ACCUMULATE 顺序逻辑)
//   4. BSND (4D) 与 TND (3D) 布局
//   5. D 对齐 (128) 与 D 非对齐 (48, D/2=24 非 16 对齐 → 拆半搬运分支)
//
// 说明: expectTilingKey / expectTilingData 的 golden 值依赖 GEN_REDUCE_TILING_KEY 宏的实际编码,
//       无法静态推导。此处采用 ExecuteTiling 动态捕获 + 结构断言 (success / key>0 / workspace 合理),
//       保证 UT 可立即运行。若需升级为 golden-match, 取消注释各 case 末尾的 printf, 首次运行捕获
//       tilingKey 与 tilingData 字符串后, 替换为 ExecuteTestCase(..., expectTilingKey, expectTilingData, ...)
//       形式 (参考 rotary_position_embedding_grad/tests/ut/op_host 同名文件)。

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

class ApplyRotaryPosEmbGradBabTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyRotaryPosEmbGradBabTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyRotaryPosEmbGradBabTiling TearDown" << std::endl;
    }

    // BAB 模板仅在 Ascend950 (regbase) 上生效
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};

    // 统一校验: tiling 成功 + key 有效 + workspace 合理
    void CheckBabTiling(const gert::TilingContextPara& para, bool expectDcos, uint64_t expectWorkspaceCnt)
    {
        TilingInfo tilingInfo;
        bool ok = ExecuteTiling(para, tilingInfo);
        EXPECT_TRUE(ok) << "BAB tiling should succeed";
        EXPECT_GT(tilingInfo.tilingKey, 0) << "tilingKey should be positive";
        if (expectDcos) {
            EXPECT_FALSE(tilingInfo.workspaceSizes.empty()) << "dCosFlag=1 requires workspace for partial sum";
            for (size_t i = 0; i < tilingInfo.workspaceSizes.size(); i++) {
                EXPECT_GT(tilingInfo.workspaceSizes[i], 0u) << "workspace[" << i << "] should be non-zero";
            }
        }
        EXPECT_EQ(tilingInfo.workspaceSizes.size(), expectWorkspaceCnt)
            << "workspace count mismatch, got " << tilingInfo.workspaceSizes.size();
        std::cout << "  [captured] tilingKey=" << tilingInfo.tilingKey
                  << " workspaceSizes.size=" << tilingInfo.workspaceSizes.size() << std::endl;
    }
};

// =====================================================================================
// dCosFlag=0 系列 (无 query/key 可选输入 → 不计算 grad_cos/grad_sin)
// 4 inputs: grad_query_embed, grad_key_embed, cos, sin
// 2 outputs: grad_query, grad_key
// =====================================================================================

// fp16 BSND D=128 dCosFlag=0
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_dcos0)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_query_embed
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_key_embed
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // cos (cosb=1)
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // sin
        },
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_query
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_key
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},  // BSND
        },
        &compileInfo);
    CheckBabTiling(para, false, 1);
}

// fp32 BSND D=128 dCosFlag=0
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp32_bsnd_dcos0)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    CheckBabTiling(para, false, 1);
}

// =====================================================================================
// dCosFlag=1 系列 (有 query/key 可选输入 → 计算 grad_cos/grad_sin, 触发 cosPartial 融合)
// 6 inputs: grad_query_embed, grad_key_embed, cos, sin, query, key
// 4 outputs: grad_query, grad_key, grad_cos, grad_sin
// =====================================================================================

// fp16 BSND D=128 dCosFlag=1, nQ==nK
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_dcos1)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // query
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // key
        },
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_cos
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // grad_sin
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    CheckBabTiling(para, true, 1);
}

// bf16 BSND D=128 dCosFlag=1
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_bf16_bsnd_dcos1)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    CheckBabTiling(para, true, 1);
}

// =====================================================================================
// nQ != nK 系列 — 触发 SET/ACCUMULATE 顺序逻辑与 tail tile 差异
// =====================================================================================

// fp16 BSND nQ=8 nK=4 dCosFlag=1 (Q 路 head 多, 多次 idxN 迭代 K 提前耗尽)
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_nq_neq_nk)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // nQ=8
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // nK=4
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 8, 128}, {2, 64, 8, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 128}, {2, 64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 128}, {1, 64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    CheckBabTiling(para, true, 1);
}

// =====================================================================================
// TND 布局 (3D shape) — 退化为 B=1 的 BSND
// =====================================================================================

// fp16 TND D=128 dCosFlag=1
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_tnd_dcos1)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // [T, N, D]
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // cos [T, 1, D]
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 4, 128}, {64, 4, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1, 128}, {64, 1, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},  // TND
        },
        &compileInfo);
    CheckBabTiling(para, true, 1);
}

// =====================================================================================
// D 非对齐系列 — D=48, D/2=24 非 16 对齐 → 拆半搬运分支
// =====================================================================================

// fp16 BSND D=48 dCosFlag=1 (非对齐: 走 srcStride=0 拆半 DataCopyPad 分支)
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_d48_unaligned)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    CheckBabTiling(para, true, 1);
}

// fp16 BSND D=48 dCosFlag=0 (非对齐 + 无 cos/sin grad)
TEST_F(ApplyRotaryPosEmbGradBabTiling, bab_fp16_bsnd_d48_unaligned_dcos0)
{
    optiling::ApplyRotaryPosEmbGradCompileInfo compileInfo = {};
    gert::TilingContextPara para("ApplyRotaryPosEmbGrad",
        {
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 64, 1, 48}, {1, 64, 1, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2, 64, 4, 48}, {2, 64, 4, 48}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"rotary_mode", Ops::Transformer::AnyValue::CreateFrom<string>("half")},
            {"layout", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    CheckBabTiling(para, false, 1);
}
