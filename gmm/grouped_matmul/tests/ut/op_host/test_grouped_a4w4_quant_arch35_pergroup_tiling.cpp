/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_a4w4_quant_arch35_pergroup_tiling.cpp
 * \brief Host tiling UT for GroupedA4W4Quant (INT4×INT4) pergroup groupSize 约束。
 *
 * 验证 GMM_A5_A4W4_PERGROUPSIZE_CONSTRAINT.md §3 偶数规格：
 *   - groupSize ∈ {偶数} 且整除 K -> tiling 成功，quantGroupNum = K/groupSize，baseK=128
 *     覆盖 256/512/128（128 倍数）、192/64（32 倍数非 128）、2（最小偶数非 32 倍数）。
 *   - groupSize 奇数 -> tiling 失败（A1：grouped_a4w4_quant_tiling.cpp
 *     的 (k/G) % A4W4_QUANT_GROUP_SIZE 校验，A4W4_QUANT_GROUP_SIZE=2）。
 *   - perchannel（scale 2D）对照：quantGroupNum=1。
 *
 * 路由：TilingGMM 在 npuArch==DAV_3510 && xDType==DT_INT4 && weightDtype==DT_INT4 时走
 *       GroupedA4W4QuantTiling（grouped_matmul_tiling.cpp:2256）。
 *
 * 输入契约（GroupedMatmul 9 输入）：x(0) weight(1) bias(2) scale(3) offset(4)
 *   antiquantScale(5) antiquantOffset(6) groupList(7) perTokenScale(8)。
 *   A4W4：x [M,K] INT4，weight [E,K,N] INT4，scale [E,G,N] UINT64(pergroup)/[E,N] UINT64(perchannel)，
 *   groupList [E] INT64(cumsum, 末值=M)，perTokenScale [M] FLOAT，y [M,N] FP16。
 *
 * 注：本测试只覆盖 host tiling 层。kernel 端到端精度 UT 受 A4W4_BLAZE_ENABLED 未定义阻塞，
 *     见约束文档 §6 deferred TODO。
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "../../../op_host/op_tiling/arch35/grouped_a4w4_quant_tiling.h"
#include "../../../op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.h"  // AIC_AIV_CORE_RATIO
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"

using namespace std;
using namespace ge;

namespace {

constexpr uint32_t DEFAULT_AIC_NUM = 32;

// A4W4 GetPlatformInfo 要求 aic:aiv=1:2；AIC_AIV_CORE_RATIO=2 满足。
optiling::GMMCompileInfo MakeAscend950CompileInfo()
{
    return {
        DEFAULT_AIC_NUM,                                      // aicNum
        DEFAULT_AIC_NUM * optiling::AIC_AIV_CORE_RATIO,       // aivNum = 2*aic
        262144,                                               // ubSize
        524288,                                               // l1Size
        134217728,                                            // l2Size
        262144,                                               // l0CSize
        65536,                                                // l0ASize
        65536,                                                // l0BSize
        platform_ascendc::SocVersion::ASCEND950,              // socVersion
        NpuArch::DAV_3510,
    };
}

gert::StorageShape MakeEmptyShape()
{
    return gert::StorageShape();
}

gert::TilingContextPara::TensorDescription MakeTensorDesc(const vector<int64_t> &dims, ge::DataType dtype,
                                                          ge::Format format = ge::FORMAT_ND)
{
    return {ops::ut::MakeGertStorageShape(dims, dims), dtype, format};
}

// GroupedMatmul attrs（索引）：0 split_item, 1 dtype, 2 transpose_weight, 3 transpose_x,
//   4 group_type, 5 group_list_type, 6 act_type, 7 tuning_config。
// A4W4 约束：transpose_x=false, group_type=0(SPLIT_M), act_type=0, split_item=3(单输出 y)。
vector<gert::TilingContextPara::OpAttr> GetA4W4Attrs(bool transposeWeight)
{
    return {
        {"split_item", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
        {"dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(transposeWeight)},
        {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"act_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"tuning_config", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>({0})},
    };
}

// 驱动 A4W4 tiling。pergroup 时 scale=[E,G,N]（G=k/groupSize）；perchannel 时 scale=[E,N]（groupSize=0）。
// 返回 tiling 是否成功，tilingInfo 传出用于断言。
bool RunA4W4Tiling(int64_t m, int64_t k, int64_t n, int64_t e, int64_t groupSize, bool isPergroup,
                   TilingInfo &tilingInfo)
{
    auto compileInfo = MakeAscend950CompileInfo();
    const int64_t G = isPergroup ? (k / groupSize) : 0;  // K 方向量化组数 = scale.dim(1)

    // groupList [E] cumsum（均分 M，末值修正为 M）
    vector<int64_t> groupList(static_cast<size_t>(e));
    const int64_t per = m / e;
    for (int64_t i = 0; i < e; ++i) {
        groupList[static_cast<size_t>(i)] = (i + 1) * per;
    }
    groupList[static_cast<size_t>(e - 1)] = m;

    // scale: pergroup [E,G,N]，perchannel [E,N]；dtype UINT64
    gert::StorageShape scaleShape = isPergroup
        ? ops::ut::MakeGertStorageShape({e, G, n}, {e, G, n})
        : ops::ut::MakeGertStorageShape({e, n}, {e, n});

    vector<gert::TilingContextPara::TensorDescription> inputDescs = {
        MakeTensorDesc({m, k}, ge::DT_INT4),                       // x [M,K] INT4
        MakeTensorDesc({e, k, n}, ge::DT_INT4),                    // weight [E,K,N] INT4 (transpose_weight=false)
        {MakeEmptyShape(), ge::DT_FLOAT16, ge::FORMAT_ND},         // bias (A4W4 无 bias, empty)
        {scaleShape, ge::DT_UINT64, ge::FORMAT_ND},                // scale (UINT64)
        {MakeEmptyShape(), ge::DT_FLOAT, ge::FORMAT_ND},           // offset (empty)
        {MakeEmptyShape(), ge::DT_FLOAT16, ge::FORMAT_ND},         // antiquantScale (empty)
        {MakeEmptyShape(), ge::DT_FLOAT16, ge::FORMAT_ND},         // antiquantOffset (empty)
        MakeTensorDesc({e}, ge::DT_INT64),                         // groupList [E]
        MakeTensorDesc({m}, ge::DT_FLOAT),                         // perTokenScale [M] FLOAT
    };
    vector<gert::TilingContextPara::TensorDescription> outputDescs = {
        MakeTensorDesc({m, n}, ge::DT_FLOAT16),                    // y [M,N] FP16
    };

    gert::TilingContextPara ctx(
        "GroupedMatmul", inputDescs, outputDescs, GetA4W4Attrs(false),
        &compileInfo, "3510", compileInfo.aicNum, compileInfo.ubSize);
    return ExecuteTiling(ctx, tilingInfo);
}

// 取 A4W4 tiling data 的 gmmA4W4Params 用于断言。
const GroupedMatmulTilingData::GMMBaseParamsA4W4 *GetA4W4Params(const TilingInfo &info)
{
    return &reinterpret_cast<const GroupedMatmulTilingData::GMMA4W4QuantTilingData *>(info.tilingData.get())
                ->gmmA4W4Params;
}

} // namespace

class TestGroupedA4W4QuantPergroupArch35Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// PlanA：groupSize=256（128 倍数）接受，quantGroupNum=K/256=4，baseK=128。
TEST_F(TestGroupedA4W4QuantPergroupArch35Tiling, pergroupGroupSize256Accepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunA4W4Tiling(/*m=*/128, /*k=*/1024, /*n=*/256, /*e=*/2, /*groupSize=*/256, true, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMA4W4QuantTilingData));
    const auto *p = GetA4W4Params(info);
    EXPECT_EQ(p->quantGroupNum, 4U);   // G = 1024/256
    EXPECT_EQ(p->baseK, 128U);         // pergroup 固定 baseK=128
    EXPECT_EQ(p->k, 1024U);
    EXPECT_EQ(p->n, 256U);
}

// PlanA：groupSize=512（128 倍数，大 groupSize 精度更优）接受，quantGroupNum=2，baseK=128。
TEST_F(TestGroupedA4W4QuantPergroupArch35Tiling, pergroupGroupSize512Accepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunA4W4Tiling(128, 1024, 256, 2, /*groupSize=*/512, true, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMA4W4QuantTilingData));
    const auto *p = GetA4W4Params(info);
    EXPECT_EQ(p->quantGroupNum, 2U);   // G = 1024/512
    EXPECT_EQ(p->baseK, 128U);
}

// PlanA 基线：groupSize=128 接受，quantGroupNum=8，baseK=128。
TEST_F(TestGroupedA4W4QuantPergroupArch35Tiling, pergroupGroupSize128BaselineAccepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunA4W4Tiling(128, 1024, 256, 2, /*groupSize=*/128, true, info));
    const auto *p = GetA4W4Params(info);
    EXPECT_EQ(p->quantGroupNum, 8U);   // G = 1024/128
    EXPECT_EQ(p->baseK, 128U);
}

// 偶数规格接受：groupSize=192（32 倍数非 128 倍数，偶数）。K=960 % G=5 == 0，(K/G)=192 为偶数 -> 接受。
// quantGroupNum = 960/192 = 5，baseK=128。
TEST_F(TestGroupedA4W4QuantPergroupArch35Tiling, pergroupGroupSize192EvenAccepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunA4W4Tiling(128, /*k=*/960, 256, 2, /*groupSize=*/192, true, info));
    const auto *p = GetA4W4Params(info);
    EXPECT_EQ(p->quantGroupNum, 5U);   // G = 960/192
    EXPECT_EQ(p->baseK, 128U);
}

// 偶数规格接受：groupSize=64（<128，32 倍数，偶数）。K=512 % G=8 == 0，(K/G)=64 为偶数 -> 接受。
// quantGroupNum = 512/64 = 8，baseK=128。
TEST_F(TestGroupedA4W4QuantPergroupArch35Tiling, pergroupGroupSize64EvenAccepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunA4W4Tiling(128, /*k=*/512, 256, 2, /*groupSize=*/64, true, info));
    const auto *p = GetA4W4Params(info);
    EXPECT_EQ(p->quantGroupNum, 8U);   // G = 512/64
    EXPECT_EQ(p->baseK, 128U);
}

// 偶数规格接受：groupSize=2（最小偶数，非 32 倍数）。K=1024 % G=512 == 0，(K/G)=2 为偶数 -> 接受。
// quantGroupNum = 1024/2 = 512，baseK=128。注意：groupSize=2 非 32 倍数，L0B ZN 末步对齐为 deferred
TEST_F(TestGroupedA4W4QuantPergroupArch35Tiling, pergroupGroupSize2MinEvenAccepted)
{
    TilingInfo info;
    ASSERT_TRUE(RunA4W4Tiling(128, /*k=*/1024, 256, 2, /*groupSize=*/2, true, info));
    const auto *p = GetA4W4Params(info);
    EXPECT_EQ(p->quantGroupNum, 512U);   // G = 1024/2
    EXPECT_EQ(p->baseK, 128U);
}

// 偶数规格拒绝：groupSize=3（奇数）。K=192 % G=64 == 0（过 K%G 校验），(K/G)=3 非偶数 -> A1 拒绝。
TEST_F(TestGroupedA4W4QuantPergroupArch35Tiling, pergroupGroupSize3OddRejected)
{
    TilingInfo info;
    EXPECT_FALSE(RunA4W4Tiling(128, /*k=*/192, 256, 2, /*groupSize=*/3, true, info));
}

// 偶数规格拒绝：groupSize=129（>128 奇数）。K=4128 % G=32 == 0，(K/G)=129 非偶数 -> A1 拒绝。
TEST_F(TestGroupedA4W4QuantPergroupArch35Tiling, pergroupGroupSize129OddRejected)
{
    TilingInfo info;
    EXPECT_FALSE(RunA4W4Tiling(128, /*k=*/4128, 256, 2, /*groupSize=*/129, true, info));
}

// 对照：perchannel（scale 2D [E,N]）-> quantGroupNum=1，baseK=min(K,128)=128。
TEST_F(TestGroupedA4W4QuantPergroupArch35Tiling, perchannelScale2DQuantGroupNumOne)
{
    TilingInfo info;
    ASSERT_TRUE(RunA4W4Tiling(128, 1024, 256, 2, /*groupSize=*/0, /*isPergroup=*/false, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(GroupedMatmulTilingData::GMMA4W4QuantTilingData));
    const auto *p = GetA4W4Params(info);
    EXPECT_EQ(p->quantGroupNum, 1U);   // perchannel
    EXPECT_EQ(p->baseK, 128U);         // min(1024,128)
}

// P2-1：perchannel K 非 32 倍数 -> host CheckA4W4Params 拒绝（int8 C0_SIZE，L0B 对齐）。
// K=100 过 n%8/K>0 校验，在 K%32 校验处失败。
TEST_F(TestGroupedA4W4QuantPergroupArch35Tiling, perchannelKNotMultipleOf32Rejected)
{
    TilingInfo info;
    EXPECT_FALSE(RunA4W4Tiling(128, /*k=*/100, 256, 2, /*groupSize=*/0, /*isPergroup=*/false, info));
}

// P2-2：perchannel K=0 -> host CheckA4W4Params 拒绝（k 必须 > 0）。
TEST_F(TestGroupedA4W4QuantPergroupArch35Tiling, perchannelKZeroRejected)
{
    TilingInfo info;
    EXPECT_FALSE(RunA4W4Tiling(128, /*k=*/0, 256, 2, /*groupSize=*/0, /*isPergroup=*/false, info));
}
