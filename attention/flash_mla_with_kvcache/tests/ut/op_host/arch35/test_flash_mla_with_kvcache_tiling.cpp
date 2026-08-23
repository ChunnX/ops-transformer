/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "../flash_mla_with_kvcache_param.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/flash_mla_with_kvcache_tiling_data.h"

namespace FlashMlaWithKvcacheUT {

namespace {

using gert::TilingContextPara;

std::vector<TilingContextPara::OpAttr> BuildMlaAttrs(const FlashMlaWithKvcacheHostUtParamBase &p)
{
    return {
        {"head_dim_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.head_dim_v)},
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(p.softmax_scale)},
        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.mask_mode)},
        {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.max_seqlen_q)},
        {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.max_seqlen_kv)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(p.layout_q)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(p.layout_kv)},
        {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(p.layout_out)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.return_softmax_lse)},
    };
}

// m0070 input order: q, k_cache, block_table, cache_seqlens, cu_seqlens_q, seqused_q, attn_mask, metadata
std::vector<TilingContextPara::TensorDescription> BuildMlaInputs(const FlashMlaWithKvcacheTilingUtParam &p)
{
    return {p.q, p.k_cache, p.block_table, p.cache_seqlens, p.cu_seqlens_q, p.seqused_q, p.attn_mask, p.metadata};
}

TilingContextPara BuildTilingContext(const FlashMlaWithKvcacheTilingUtParam &p)
{
    return TilingContextPara("FlashMlaWithKvcache", BuildMlaInputs(p), {p.attn_out, p.softmax_lse}, BuildMlaAttrs(p),
                             const_cast<FlashMlaWithKvcacheCompileInfo *>(&p.compileInfo), "Ascend950", 64, 262144, 8192);
}

// Const-data tensor: cu_seqlens_q / cache_seqlens host data driving the R2 seq-lens path
TilingContextPara::TensorDescription ConstInt32Tensor(const std::vector<int32_t> &data)
{
    auto *buf = new std::vector<int32_t>(data); // intentionally leaked: valid for the whole test process
    return TilingContextPara::TensorDescription(
        gert::StorageShape({static_cast<int64_t>(data.size())}, {static_cast<int64_t>(data.size())}), ge::DT_INT32,
        ge::FORMAT_ND, true, buf->data());
}

FlashMlaWithKvcacheTilingUtParam MakeBaseMlaParam(int64_t tTokens)
{
    FlashMlaWithKvcacheTilingUtParam p;
    p.q = TilingContextPara::TensorDescription(
        gert::StorageShape({tTokens, 8, 576}, {tTokens, 8, 576}), ge::DT_FLOAT16, ge::FORMAT_ND);
    p.k_cache = TilingContextPara::TensorDescription(
        gert::StorageShape({4, 1, 36, 64, 16}, {4, 1, 36, 64, 16}), ge::DT_FLOAT16, ge::FORMAT_ND);
    p.block_table = TilingContextPara::TensorDescription(
        gert::StorageShape({2, 16}, {2, 16}), ge::DT_INT32, ge::FORMAT_ND);
    p.cache_seqlens = TilingContextPara::TensorDescription(
        gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND);
    p.cu_seqlens_q = TilingContextPara::TensorDescription(
        gert::StorageShape({3}, {3}), ge::DT_INT32, ge::FORMAT_ND);
    p.metadata = TilingContextPara::TensorDescription(
        gert::StorageShape({4096}, {4096}), ge::DT_INT32, ge::FORMAT_ND);
    p.attn_out = TilingContextPara::TensorDescription(
        gert::StorageShape({tTokens, 8, 512}, {tTokens, 8, 512}), ge::DT_FLOAT16, ge::FORMAT_ND);
    p.layout_q = "TND";
    p.layout_kv = "PA_NZ";
    p.layout_out = "TND";
    p.softmax_scale = 0.125f;
    return p;
}

// 5-field key encoding (flash_mla_with_kvcache_template_tiling_key.h bit layout): InOutLayoutType<<0 |
// Config<<8 | HasAttenMask<<18 | KvLayoutType<<19 | IsFd<<21
// InOutLayoutType: TND=2 / BSND(BSH)=1 / BNSD=0；KvLayoutType: PA_NZ=3 / PA_BNBD=2 / PA_BBND=1
// （取值以本算子 template_tiling_key.h 的枚举常量为准，勿照抄 flash_attn 的编号）
// legacy tiling key 值（T2/T14 记录）：isFd=0 形式 1575170=prefill m0f0 / 1837314=m1f0；
// isFd=1 形式 3672322 / 3934466，为主链（SplitPolicy 恒 flashDecodeFlag_=true）刷新后的 host 真值。
// f=0 legacy 值仅公式级保留，host 恒 isFd=1。
uint64_t EncodeMlaTilingKey(bool hasAttenMask, bool isFd, uint64_t inOutLayoutType = 2, uint64_t kvLayoutType = 3)
{
    return (inOutLayoutType << 0) | (9ULL << 8) | (static_cast<uint64_t>(hasAttenMask) << 18) |
           (kvLayoutType << 19) | (static_cast<uint64_t>(isFd) << 21);
}

} // namespace

class FlashMlaWithKvcacheArch35TilingTest : public testing::TestWithParam<FlashMlaWithKvcacheTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashMlaWithKvcache Arch35 TilingTest SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "FlashMlaWithKvcache Arch35 TilingTest TearDown" << std::endl;
    }
};

TEST_P(FlashMlaWithKvcacheArch35TilingTest, tiling)
{
    auto param = GetParam();
    ExecuteTestCase(BuildTilingContext(param), param.expectResult, param.expectTilingKey, param.expectTilingDataHash,
                    param.expectWorkspaces, 0, true);
}

INSTANTIATE_TEST_SUITE_P(FlashMlaWithKvcache, FlashMlaWithKvcacheArch35TilingTest,
                         testing::ValuesIn(GetCasesFromCsv<FlashMlaWithKvcacheTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
                         PrintCaseInfoString<FlashMlaWithKvcacheTilingUtParam>);

// ============================================================================
// C5 topic cases: metadata passthrough + FIA workspace formulas (R3)
// ============================================================================

// metadata passthrough (D7): metadata tensor present -> the tiling result is the constant-fill
// branch; base params are parse constants, metadata content/size never read
TEST(FlashMlaWithKvcacheArch35Tiling, MetadataPassthroughConstantFill)
{
    FlashMlaWithKvcacheTilingUtParam param = MakeBaseMlaParam(192);
    param.cu_seqlens_q = ConstInt32Tensor({0, 64, 192}); // prefill lens: 64 / 128, no batch <= 1

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(BuildTilingContext(param), tilingInfo));
    // host 真值（T14 漂移修复：SplitPolicy 恒 flashDecodeFlag_=true，与 flash_attn tiling.cpp:129 对齐，
    // prefill 也走 isFd=1 派发 → key = 3672322）——key 断言以 host 实测为准
    EXPECT_EQ(tilingInfo.tilingKey, EncodeMlaTilingKey(false, true));

    auto *tiling = reinterpret_cast<const optiling::FlashMlaWithKvcacheTilingData *>(tilingInfo.tilingData.get());
    const optiling::FlashMlaWithKvcacheBaseParams &base = tiling->baseTiling.flashMlaWithKvcacheBaseParams;
    EXPECT_EQ(base.bSize, 2);
    EXPECT_EQ(base.t1Size, 192);
    EXPECT_EQ(base.s1Size, 192);
    EXPECT_EQ(base.n2Size, 1);
    EXPECT_EQ(base.gSize, 8);
    EXPECT_EQ(base.dSize, 512);
    EXPECT_EQ(base.dSizeV, 512);
    EXPECT_EQ(base.dSizeRope, 64);
    EXPECT_EQ(base.actualSeqLengthsQSize, 3);  // cu_seqlens_q elements = b + 1
    EXPECT_EQ(base.actualSeqLengthsKVSize, 2); // cache_seqlens elements = b
    EXPECT_EQ(base.isKvContinuous, 0);
    EXPECT_EQ(base.isSoftMaxLseEnable, 0);
    EXPECT_EQ(base.outputLayout, 3); // KERNEL_LAYOUT_TND (out==q, T11 flip from NTD/5)
    EXPECT_EQ(base.coreNum, tilingInfo.blockNum);

    // multi-section passthrough: a 2-section-capacity metadata tensor yields byte-identical tiling data
    FlashMlaWithKvcacheTilingUtParam multiSection = MakeBaseMlaParam(192);
    multiSection.cu_seqlens_q = ConstInt32Tensor({0, 64, 192});
    multiSection.cache_seqlens = ConstInt32Tensor({128, 128});
    multiSection.metadata = TilingContextPara::TensorDescription(
        gert::StorageShape({12000}, {12000}), ge::DT_INT32, ge::FORMAT_ND);
    TilingInfo multiSectionInfo;
    ASSERT_TRUE(ExecuteTiling(BuildTilingContext(multiSection), multiSectionInfo));
    ASSERT_EQ(multiSectionInfo.tilingDataSize, tilingInfo.tilingDataSize);
    EXPECT_EQ(std::memcmp(multiSectionInfo.tilingData.get(), tilingInfo.tilingData.get(),
                          tilingInfo.tilingDataSize),
              0);
}

// flashDecodeFlag_ 语义（T14/learnings：ACLNN 直调路径 host 侧不能读 GM 的 seq-lens 值——
// GetData<int32_t>() 返回设备指针，解引用即段错误；与 flash_attn 同策略，按 shape/attr 推导）：
//   SplitPolicy 恒置 flashDecodeFlag_=true（metadata 透传 + 兜底两条路径，与 flash_attn tiling.cpp:129
//   对齐）——prefill/decode 均走 isFd=1 派发；prefill 的 KEY-1 断言已按 host 真值刷新（T14 漂移修复）。
// CalcWorkspaceSize delta 仍匹配 FIA 公式：
// (numBlocks*PRE_LOAD(2)*64*512 + 2*numBlocks*2*64*8) * sizeof(float)
// 注意：varlen 混合 decode 的逐 batch 判断在 ACLNN 路径粒度不足（F6 记录项）；实际长度
// 由 kernel 侧 ACTLEN_T=uint32_t 解析器按 GM 值工作，不受影响（R2）。
TEST(FlashMlaWithKvcacheArch35Tiling, SeqLensDecodeFlagAndWorkspace)
{
    FlashMlaWithKvcacheTilingUtParam prefill = MakeBaseMlaParam(192);
    TilingInfo prefillInfo;
    ASSERT_TRUE(ExecuteTiling(BuildTilingContext(prefill), prefillInfo));
    // host 真值：SplitPolicy 恒 flashDecodeFlag_=true（与 flash_attn tiling.cpp:129 对齐），
    // prefill 同样走 isFd=1 派发（T14 漂移修复，key 断言以 host 实测为准）
    EXPECT_EQ(prefillInfo.tilingKey, EncodeMlaTilingKey(false, true));
    size_t prefillWorkspace = prefillInfo.workspaceSizes.empty() ? 0 : prefillInfo.workspaceSizes[0];

    FlashMlaWithKvcacheTilingUtParam decode = MakeBaseMlaParam(1); // q TSIZE=1 -> s1Size<=1 -> decode
    TilingInfo decodeInfo;
    ASSERT_TRUE(ExecuteTiling(BuildTilingContext(decode), decodeInfo));
    EXPECT_EQ(decodeInfo.tilingKey, EncodeMlaTilingKey(false, true));
    ASSERT_FALSE(decodeInfo.workspaceSizes.empty());
    constexpr uint64_t preLoad = 2;
    constexpr uint64_t mSize = 64;
    constexpr uint64_t dVSize = 512;
    constexpr uint64_t lseSize = 8;
    uint64_t expectedDelta =
        (static_cast<uint64_t>(decodeInfo.blockNum) * preLoad * mSize * dVSize +
         2 * static_cast<uint64_t>(decodeInfo.blockNum) * preLoad * mSize * lseSize) *
        sizeof(float);
    // host 真值（T6 后）：prefill/decode 均 isFd=1 且 numBlocks 相同（metadata 透传恒全核调度），
    // workspace = defaultSysWorkspaceSize + expectedDelta 两路径相等；旧"decode=prefill+delta"
    // 前提（prefill 无 FD 增量）已随 isFd 恒真失效，改为三者一致的组成校验：
    //   workspaceSizes[0] == defaultSysWorkspaceSize + expectedDelta == prefillWorkspace
    uint64_t prefillDelta =
        (static_cast<uint64_t>(prefillInfo.blockNum) * preLoad * mSize * dVSize +
         2 * static_cast<uint64_t>(prefillInfo.blockNum) * preLoad * mSize * lseSize) *
        sizeof(float);
    EXPECT_EQ(prefillWorkspace - prefillDelta, decodeInfo.workspaceSizes[0] - expectedDelta);
    EXPECT_EQ(decodeInfo.workspaceSizes[0], prefillWorkspace);

    auto *tiling = reinterpret_cast<const optiling::FlashMlaWithKvcacheTilingData *>(decodeInfo.tilingData.get());
    EXPECT_EQ(tiling->baseTiling.flashMlaWithKvcacheWorkspaceParams.accumOutSize,
              static_cast<uint32_t>(decodeInfo.blockNum * preLoad * mSize * dVSize));
    EXPECT_EQ(tiling->baseTiling.flashMlaWithKvcacheWorkspaceParams.logSumExpSize,
              static_cast<uint32_t>(decodeInfo.blockNum * preLoad * mSize * lseSize));

    // LSE does not change the workspace sizes (same formula, just an output flag)
    FlashMlaWithKvcacheTilingUtParam decodeLse = decode;
    decodeLse.return_softmax_lse = 1;
    decodeLse.softmax_lse = TilingContextPara::TensorDescription(
        gert::StorageShape({1, 8}, {1, 8}), ge::DT_FLOAT, ge::FORMAT_ND);
    TilingInfo decodeLseInfo;
    ASSERT_TRUE(ExecuteTiling(BuildTilingContext(decodeLse), decodeLseInfo));
    EXPECT_EQ(decodeLseInfo.workspaceSizes[0], decodeInfo.workspaceSizes[0]);
}

// needInit 语义（T14）：shape 级推导 needInit_ = (qTSize == 0) || (s2Size == 0)。
// SUCCESS 几何下总 token 数 > 0 且 KV 侧非空 -> needInit == 0；总 token 数为 0 的输入
// 在 IsCapable 的 emptyTensor 门即被拒绝（CSV EmptyTensor 行锁定 GRAPH_FAILED），
// 故 needInit == 1 在透传路径不可达——逐 batch 空行检测依赖 kernel 侧 GM 值解析（R2/F6）。
TEST(FlashMlaWithKvcacheArch35Tiling, NeedInitShapeDerivedAndEmptyRejected)
{
    FlashMlaWithKvcacheTilingUtParam filled = MakeBaseMlaParam(192);
    TilingInfo filledInfo;
    ASSERT_TRUE(ExecuteTiling(BuildTilingContext(filled), filledInfo));
    auto *tiling = reinterpret_cast<const optiling::FlashMlaWithKvcacheTilingData *>(filledInfo.tilingData.get());
    EXPECT_EQ(tiling->baseTiling.flashMlaWithKvcacheEmptyTensorParams.needInit, 0);
    // 空 Q（T=0）在 check/IsCapable 层被拒绝，不会走到产出 needInit=1 的 tiling data
    FlashMlaWithKvcacheTilingUtParam empty = MakeBaseMlaParam(0);
    ExecuteTestCase(BuildTilingContext(empty), ge::GRAPH_FAILED, 0, "", {}, 0, true);
}

// metadata is mandatory (D7): without a metadata input the case is rejected, so the
// CalcMaxWorkspaceSize fallback (fia:134) stays dead code by design
TEST(FlashMlaWithKvcacheArch35Tiling, MetadataMissingRejected)
{
    FlashMlaWithKvcacheTilingUtParam param = MakeBaseMlaParam(192);
    param.metadata = TD_DEFAULT; // optional input not instantiated
    ExecuteTestCase(BuildTilingContext(param), ge::GRAPH_FAILED, 0, "", {}, 0, true);
}

} // namespace FlashMlaWithKvcacheUT
