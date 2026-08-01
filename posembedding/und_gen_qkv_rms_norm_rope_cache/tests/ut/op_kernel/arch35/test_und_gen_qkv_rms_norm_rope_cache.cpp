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
    uint8_t* q = (uint8_t*)AscendC::GmAlloc(qSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(WORKSPACE_SIZE);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    memset(undQkv, 0, undQkvSize);
    memset(genQkv, 0, genQkvSize);
    memset(undWeightsQ, 0, weightSize);
    memset(undWeightsK, 0, weightSize);
    memset(genWeightsQ, 0, weightSize);
    memset(genWeightsK, 0, weightSize);
    memset(cosSinCache, 0, cosSinSize);
    memset(kCache, 0, kCacheSize);
    memset(vCache, 0, vCacheSize);
    memset(q, 0, qSize);

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
    // TODO(tiling): 切分实现后，这里改用 host tiling 产出的 usedCoreNum/blockFactor/ubFactor
    tilingData->usedCoreNum = cfg.blockDim;
    tilingData->blockFactor = CeilDiv(total, static_cast<int64_t>(cfg.blockDim));
    tilingData->tailBlockFactor = total - tilingData->blockFactor * (cfg.blockDim - 1);
    tilingData->ubFactor = 1;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(und_gen_qkv_rms_norm_rope_cache, cfg.blockDim, undQkv, undWeightsQ, undWeightsK, cosSinCache, kCache,
                vCache, slotMapping, positions, cfg.hasGen ? genQkv : nullptr, cfg.hasGen ? genWeightsQ : nullptr,
                cfg.hasGen ? genWeightsK : nullptr, cfg.hasCatIndices ? catIndices : nullptr, q, kCache, vCache,
                workspace, tiling);

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
//       实现落地后需要补：与 golden.py 对齐的数值校验、tail block、gen=None、cat_indices=None 分支。

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
    RunTestWithConfig(cfg);
}

TEST_F(und_gen_qkv_rms_norm_rope_cache_test, launch_prefill_only_plain_rope)
{
    // gen_qkv/gen_weights/cat_indices 均缺省，mrope_section 退化为 [D/2, 0, 0]
    TestConfig cfg;
    cfg.undLen = 128;
    cfg.genLen = 0;
    cfg.nq = 8;
    cfg.nk = 1;
    cfg.nv = 1;
    cfg.maxPos = 128;
    cfg.blockDim = 4;
    cfg.hasGen = false;
    cfg.hasCatIndices = false;
    cfg.mropeSection[0] = HEAD_DIM / 2;
    cfg.mropeSection[1] = 0;
    cfg.mropeSection[2] = 0;
    RunTestWithConfig(cfg);
}
