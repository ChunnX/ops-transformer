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
#include <cstdint>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_und_gen_qkv_rms_norm_rope_cache_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void und_gen_qkv_rms_norm_rope_cache(
    GM_ADDR und_qkv, GM_ADDR und_weights_q, GM_ADDR und_weights_k, GM_ADDR cos_sin_cache, GM_ADDR k_cache,
    GM_ADDR v_cache, GM_ADDR slot_mapping, GM_ADDR positions, GM_ADDR gen_qkv, GM_ADDR gen_weights_q,
    GM_ADDR gen_weights_k, GM_ADDR cat_indices, GM_ADDR q, GM_ADDR k_cache_out, GM_ADDR v_cache_out,
    GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t MROPE_AXIS_NUM = 3;
constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024;
// 输出缓冲预先填哨兵，跑完检查「不该被写的区域仍是哨兵」，用来抓 CopyOut 的寻址越界。
// 计算尚未实现，输出值本身无意义，但写到哪些位置是确定的：
//   q       写满 [0, total*Hq*D)，其后的 guard 区不得被碰
//   k/v cache 只写 slot_mapping 命中的行（本 UT 是 0..total-1），其余行不得被碰
constexpr uint8_t SENTINEL = 0xAB;
constexpr size_t GUARD_BYTES = 4096;

bool AllSentinel(const uint8_t* p, size_t begin, size_t end)
{
    for (size_t i = begin; i < end; ++i) {
        if (p[i] != SENTINEL) {
            return false;
        }
    }
    return true;
}

struct TestConfig {
    int64_t undLen;
    int64_t genLen;
    int64_t nq;
    int64_t nk;
    int64_t nv;
    int64_t maxPos;
    uint32_t blockDim;
    bool hasGen;
    bool hasCatIndices;
    int64_t mropeSection[MROPE_AXIS_NUM];
    // 单次 UB 内处理的 token 数。真机由 host CalUbTiling 按 UB 容量反推，CPU UT 里
    // 直接指定，取值要能凑出「整块 + 尾块」两种 tile，覆盖 outQue 的分段偏移
    int64_t ubFactor;
};

static inline int64_t CeilDiv(int64_t x, int64_t y)
{
    return (y != 0) ? (x + y - 1) / y : 0;
}

void RunTestWithConfig(const TestConfig& cfg)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    const int64_t total = cfg.undLen + cfg.genLen;
    const int64_t numHead = cfg.nq + cfg.nk + cfg.nv;
    const int64_t blockNum = CeilDiv(total, BLOCK_SIZE) + 1;

    size_t undQkvSize = cfg.undLen * numHead * HEAD_DIM * sizeof(bfloat16_t);
    size_t genQkvSize = std::max<int64_t>(cfg.genLen, 1) * numHead * HEAD_DIM * sizeof(bfloat16_t);
    size_t weightSize = HEAD_DIM * sizeof(bfloat16_t);
    size_t cosSinSize = cfg.maxPos * HEAD_DIM * sizeof(float);
    size_t kCacheSize = blockNum * BLOCK_SIZE * cfg.nk * HEAD_DIM * sizeof(bfloat16_t);
    size_t vCacheSize = blockNum * BLOCK_SIZE * cfg.nv * HEAD_DIM * sizeof(bfloat16_t);
    size_t slotMappingSize = total * sizeof(int64_t);
    size_t positionsSize = MROPE_AXIS_NUM * total * sizeof(int64_t);
    size_t catIndicesSize = total * sizeof(int64_t);
    size_t qSize = total * cfg.nq * HEAD_DIM * sizeof(bfloat16_t);
    size_t tilingDataSize = sizeof(UndGenQkvRmsNormRopeCacheTilingData);

    uint8_t* undQkv = (uint8_t*)AscendC::GmAlloc(undQkvSize);
    uint8_t* undWeightsQ = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* undWeightsK = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* cosSinCache = (uint8_t*)AscendC::GmAlloc(cosSinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* vCache = (uint8_t*)AscendC::GmAlloc(vCacheSize);
    uint8_t* slotMapping = (uint8_t*)AscendC::GmAlloc(slotMappingSize);
    uint8_t* positions = (uint8_t*)AscendC::GmAlloc(positionsSize);
    uint8_t* genQkv = (uint8_t*)AscendC::GmAlloc(genQkvSize);
    uint8_t* genWeightsQ = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* genWeightsK = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* catIndices = (uint8_t*)AscendC::GmAlloc(catIndicesSize);
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize + GUARD_BYTES);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(WORKSPACE_SIZE);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    memset(undQkv, 0, undQkvSize);
    memset(genQkv, 0, genQkvSize);
    memset(undWeightsQ, 0, weightSize);
    memset(undWeightsK, 0, weightSize);
    memset(genWeightsQ, 0, weightSize);
    memset(genWeightsK, 0, weightSize);
    memset(cosSinCache, 0, cosSinSize);
    memset(kCache, SENTINEL, kCacheSize);
    memset(vCache, SENTINEL, vCacheSize);
    memset(q, SENTINEL, qSize + GUARD_BYTES);

    // slot_mapping：连续占位，保证唯一且不越界
    auto* slotMap = reinterpret_cast<int64_t*>(slotMapping);
    for (int64_t i = 0; i < total; ++i) {
        slotMap[i] = i;
    }
    // positions：三轴同值，落在 [0, maxPos)
    auto* pos = reinterpret_cast<int64_t*>(positions);
    for (int64_t axis = 0; axis < MROPE_AXIS_NUM; ++axis) {
        for (int64_t i = 0; i < total; ++i) {
            pos[axis * total + i] = i % cfg.maxPos;
        }
    }
    // cat_indices：und/gen 交错，覆盖乱序寻址
    auto* cat = reinterpret_cast<int64_t*>(catIndices);
    for (int64_t i = 0; i < total; ++i) {
        cat[i] = (i % 2 == 0) ? (i / 2) % std::max<int64_t>(cfg.undLen, 1)
                              : cfg.undLen + (i / 2) % std::max<int64_t>(cfg.genLen, 1);
    }

    auto* tilingData = reinterpret_cast<UndGenQkvRmsNormRopeCacheTilingData*>(tiling);
    tilingData->totalTokens = total;
    tilingData->undLen = cfg.undLen;
    tilingData->genLen = cfg.genLen;
    tilingData->numHead = numHead;
    tilingData->numHeadQ = cfg.nq;
    tilingData->numHeadK = cfg.nk;
    tilingData->numHeadV = cfg.nv;
    tilingData->headDim = HEAD_DIM;
    tilingData->maxPos = cfg.maxPos;
    tilingData->blockNum = blockNum;
    tilingData->blockSize = BLOCK_SIZE;
    tilingData->hasGen = static_cast<int64_t>(cfg.hasGen);
    tilingData->hasCatIndices = static_cast<int64_t>(cfg.hasCatIndices);
    tilingData->mropeSectionT = cfg.mropeSection[0];
    tilingData->mropeSectionH = cfg.mropeSection[1];
    tilingData->mropeSectionW = cfg.mropeSection[2];
    tilingData->epsilon = 1e-6f;
    tilingData->reciprocal = 1.0f / static_cast<float>(HEAD_DIM);
    // 与 host CalBlockTiling 一致的余数分配：前 formerCoreNum 个核多处理 1 个 token
    const int64_t usedCoreNum = std::min(static_cast<int64_t>(cfg.blockDim), total);
    const int64_t baseFactor = total / usedCoreNum;
    const int64_t formerCoreNum = total % usedCoreNum;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->formerCoreNum = formerCoreNum;
    tilingData->blockFactor = formerCoreNum > 0 ? baseFactor + 1 : baseFactor;
    tilingData->tailBlockFactor = baseFactor;
    // 每核最多也只有 blockFactor 个 token，ubFactor 再大也用不上（与 host CalUbTiling 同规则）
    tilingData->ubFactor = std::min(cfg.ubFactor, tilingData->blockFactor);

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(und_gen_qkv_rms_norm_rope_cache, cfg.blockDim, undQkv, undWeightsQ, undWeightsK, cosSinCache, kCache,
                vCache, slotMapping, positions, cfg.hasGen ? genQkv : nullptr, cfg.hasGen ? genWeightsQ : nullptr,
                cfg.hasGen ? genWeightsK : nullptr, cfg.hasCatIndices ? catIndices : nullptr, q, kCache, vCache,
                workspace, tiling);

    // 寻址校验：q 尾部 guard 区、以及 k/v cache 中 slot_mapping 未命中的行必须原封不动
    EXPECT_TRUE(AllSentinel(q, qSize, qSize + GUARD_BYTES)) << "q 写越界，冲掉了尾部 guard 区";
    const size_t kUsedBytes = total * cfg.nk * HEAD_DIM * sizeof(bfloat16_t);
    const size_t vUsedBytes = total * cfg.nv * HEAD_DIM * sizeof(bfloat16_t);
    EXPECT_TRUE(AllSentinel(kCache, kUsedBytes, kCacheSize)) << "k_cache 写到了 slot_mapping 未命中的行";
    EXPECT_TRUE(AllSentinel(vCache, vUsedBytes, vCacheSize)) << "v_cache 写到了 slot_mapping 未命中的行";

    AscendC::GmFree(undQkv);
    AscendC::GmFree(undWeightsQ);
    AscendC::GmFree(undWeightsK);
    AscendC::GmFree(cosSinCache);
    AscendC::GmFree(kCache);
    AscendC::GmFree(vCache);
    AscendC::GmFree(slotMapping);
    AscendC::GmFree(positions);
    AscendC::GmFree(genQkv);
    AscendC::GmFree(genWeightsQ);
    AscendC::GmFree(genWeightsK);
    AscendC::GmFree(catIndices);
    AscendC::GmFree(q);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
} // namespace

class und_gen_qkv_rms_norm_rope_cache_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "und_gen_qkv_rms_norm_rope_cache_test SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "und_gen_qkv_rms_norm_rope_cache_test TearDown" << std::endl;
    }
};

// NOTE: kernel 计算实现尚未补齐（见 op_kernel/arch35/*_regbase.h 的 TODO），
//       当前用例仅验证 kernel 可正常拉起、参数与 TilingData 结构对齐、无越界访问。
//       实现落地后需要补：与 tests/golden.py 对齐的数值校验、tail block 分支。
//       可选输入缺省（gen=None / cat_indices=None）当前不支持，由 host tiling 拦截，不在此覆盖。

TEST_F(und_gen_qkv_rms_norm_rope_cache_test, launch_h8_1_1_full_optional)
{
    TestConfig cfg;
    cfg.undLen = 5;
    cfg.genLen = 3;
    cfg.nq = 8;
    cfg.nk = 1;
    cfg.nv = 1;
    cfg.maxPos = 32;
    cfg.blockDim = 1;
    cfg.hasGen = true;
    cfg.hasCatIndices = true;
    cfg.mropeSection[0] = 16;
    cfg.mropeSection[1] = 16;
    cfg.mropeSection[2] = 16;
    cfg.ubFactor = 3; // 单核 8 个 token -> tile 3/3/2，尾块非满
    RunTestWithConfig(cfg);
}

TEST_F(und_gen_qkv_rms_norm_rope_cache_test, launch_h16_2_2_multi_core)
{
    TestConfig cfg;
    cfg.undLen = 80;
    cfg.genLen = 16;
    cfg.nq = 16;
    cfg.nk = 2;
    cfg.nv = 2;
    cfg.maxPos = 32;
    cfg.blockDim = 8;
    cfg.hasGen = true;
    cfg.hasCatIndices = true;
    cfg.mropeSection[0] = 16;
    cfg.mropeSection[1] = 16;
    cfg.mropeSection[2] = 16;
    cfg.ubFactor = 5; // 每核 12 个 token -> tile 5/5/2
    RunTestWithConfig(cfg);
}

TEST_F(und_gen_qkv_rms_norm_rope_cache_test, launch_plain_rope)
{
    // mrope_section 退化为 [D/2, 0, 0]（等价标准 RoPE）；
    // 可选输入当前必须全部提供，缺省场景由 host tiling 拦截，不在 kernel UT 覆盖
    TestConfig cfg;
    cfg.undLen = 127;
    cfg.genLen = 1;
    cfg.nq = 8;
    cfg.nk = 1;
    cfg.nv = 1;
    cfg.maxPos = 128;
    cfg.blockDim = 4;
    cfg.hasGen = true;
    cfg.hasCatIndices = true;
    cfg.mropeSection[0] = HEAD_DIM / 2;
    cfg.mropeSection[1] = 0;
    cfg.mropeSection[2] = 0;
    cfg.ubFactor = 7; // 每核 32 个 token -> tile 7*4 + 4
    RunTestWithConfig(cfg);
}
