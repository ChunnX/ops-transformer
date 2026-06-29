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
 * \file test_kv_compress_epilog.cpp
 * \brief KvCompressEpilog op_kernel UT (arch35 / ascend950).
 *
 * 用 ICPU_RUN_KF 在 CPU 仿真上真正执行 kernel，覆盖三种量化模式:
 *   quantMode=0  group quant + bf16 scale (roundScale true/false)
 *   quantMode=1  group quant + e8m0 scale
 *   quantMode=2  hifloat8 整行量化
 * 校验: kernel 运行不崩溃 + 写入 slot 的 cache 字节非全 0 (sanity)。
 *
 * 说明: KvCompressEpilogTilingData 结构体由 UT 框架根据 op_host BEGIN_TILING_DATA_DEF
 * 自动生成 (强制 -include 进本编译单元), 因此这里直接使用其纯结构体成员赋值, 不再单独定义。
 */
#include <array>
#include <vector>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <sstream>
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

using namespace std;

extern "C" __global__ __aicore__ void kv_compress_epilog(
    GM_ADDR cache, GM_ADDR x, GM_ADDR slot_mapping, GM_ADDR cache_out,
    GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr int64_t SLICE_SIZE = 64;
constexpr int64_t QUANT_GROUP = 128;
constexpr int64_t QUANT_MODE_GROUP_BF16 = 0;
constexpr int64_t QUANT_MODE_GROUP_E8M0 = 1;
constexpr int64_t QUANT_MODE_HIFLOAT = 2;
constexpr size_t SYS_WORKSPACE = 16 * 1024 * 1024;

inline int64_t CeilDiv(int64_t a, int64_t b) { return b == 0 ? a : (a + b - 1) / b; }
inline int64_t RoundUp(int64_t a, int64_t b) { return CeilDiv(a, b) * b; }

template <typename T>
T* GmAllocWrapper(size_t size)
{
    T* ptr = reinterpret_cast<T*>(AscendC::GmAlloc(size));
    assert(ptr != nullptr && "GM allocation failed");
    return ptr;
}

// 把 bf16 GM 用 1.0 (0x3F80) 与 -0.5 (0xBF00) 交替填充, 保证每行有非零最大值。
void FillBf16(uint8_t* gm, int64_t numElem)
{
    uint16_t* p = reinterpret_cast<uint16_t*>(gm);
    for (int64_t i = 0; i < numElem; ++i) {
        p[i] = (i & 1) ? 0xBF00 /*-0.5*/ : 0x3F80 /*1.0*/;
    }
}
}  // namespace

struct KvParams {
    int64_t bs;
    int64_t d;
    int64_t quantMode;
    int64_t roundScale;
    int64_t blockDim;
    bool withSkip;  // 部分 slot 置 -1, 验证 skip 分支
    std::string desc;
};

class KvCompressEpilogKernelTest : public testing::TestWithParam<KvParams> {
protected:
    void ComputeTiling(KvCompressEpilogTilingData& t, const KvParams& p)
    {
        int64_t scaleCol;
        int64_t concatCol;
        int64_t kvCacheCol;
        int64_t padCol;
        if (p.quantMode == QUANT_MODE_HIFLOAT) {
            scaleCol = 0;
            concatCol = p.d;
            kvCacheCol = p.d;
            padCol = 0;
        } else {
            scaleCol = CeilDiv(p.d - SLICE_SIZE, SLICE_SIZE);
            int64_t scaleBytes = (p.quantMode == QUANT_MODE_GROUP_BF16) ? 2 : 1;
            concatCol = p.d - SLICE_SIZE + SLICE_SIZE * 2 + scaleCol * scaleBytes;
            kvCacheCol = RoundUp(concatCol, QUANT_GROUP);
            padCol = kvCacheCol - concatCol;
        }
        int64_t coreNum = p.blockDim;
        int64_t rowOfFormer = CeilDiv(p.bs, coreNum);
        int64_t usedCore = std::min(CeilDiv(p.bs, rowOfFormer), coreNum);
        int64_t rowOfTail = p.bs - (usedCore - 1) * rowOfFormer;
        int64_t rowFactor = rowOfFormer;  // 小 shape 一次处理整块, UB 充足

        t.bs = p.bs;
        t.d = p.d;
        t.kvCacheCol = kvCacheCol;
        t.kvCacheRowStride = kvCacheCol;
        t.kvCacheBlockSize = 1;
        t.kvCacheBlockStride = kvCacheCol;
        t.scaleCol = scaleCol;
        t.concatCol = concatCol;
        t.padCol = padCol;
        t.quantMode = p.quantMode;
        t.roundScale = p.roundScale;
        t.perGroupSize = QUANT_GROUP;
        t.rowOfFormerBlock = rowOfFormer;
        t.rowOfTailBlock = rowOfTail;
        t.rowLoopOfFormerBlock = CeilDiv(rowOfFormer, rowFactor);
        t.rowLoopOfTailBlock = CeilDiv(rowOfTail, rowFactor);
        t.rowFactor = rowFactor;
        t.tailRowFactorOfFormerBlock = (rowOfFormer % rowFactor == 0) ? rowFactor : rowOfFormer % rowFactor;
        t.tailRowFactorOfTailBlock = (rowOfTail % rowFactor == 0) ? rowFactor : rowOfTail % rowFactor;
        t.scalesAttr = 1.0f;

        usedCore_ = usedCore;
        kvCacheCol_ = kvCacheCol;
    }

    int64_t usedCore_ = 1;
    int64_t kvCacheCol_ = 0;
};

TEST_P(KvCompressEpilogKernelTest, RunKernel)
{
    auto p = GetParam();
    std::cout << "[KvCompressEpilog] " << p.desc << " bs=" << p.bs << " d=" << p.d
              << " quantMode=" << p.quantMode << " roundScale=" << p.roundScale
              << " blockDim=" << p.blockDim << std::endl;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    KvCompressEpilogTilingData tilingData;
    std::memset(&tilingData, 0, sizeof(tilingData));
    ComputeTiling(tilingData, p);

    // slot 上限 (= bs-1), cache 大小按 (maxSlot+1)*kvCacheCol 预留并留余量。
    int64_t cacheRows = p.bs + 1;
    size_t shapeX = static_cast<size_t>(p.bs) * p.d * sizeof(uint16_t);  // bf16
    size_t shapeSlot = static_cast<size_t>(p.bs) * sizeof(int32_t);
    size_t shapeCache = static_cast<size_t>(cacheRows) * kvCacheCol_ * sizeof(uint8_t) + 512;

    uint8_t* xGm = GmAllocWrapper<uint8_t>(shapeX);
    uint8_t* slotGm = GmAllocWrapper<uint8_t>(shapeSlot);
    uint8_t* cacheGm = GmAllocWrapper<uint8_t>(shapeCache);
    uint8_t* cacheOutGm = GmAllocWrapper<uint8_t>(shapeCache);
    uint8_t* workspace = GmAllocWrapper<uint8_t>(SYS_WORKSPACE);
    uint8_t* tilingGm = GmAllocWrapper<uint8_t>(sizeof(KvCompressEpilogTilingData));

    FillBf16(xGm, static_cast<int64_t>(p.bs) * p.d);
    std::memset(cacheGm, 0, shapeCache);
    std::memset(cacheOutGm, 0, shapeCache);
    std::memset(workspace, 0, SYS_WORKSPACE);
    std::memcpy(tilingGm, &tilingData, sizeof(KvCompressEpilogTilingData));

    int32_t* slot = reinterpret_cast<int32_t*>(slotGm);
    for (int64_t i = 0; i < p.bs; ++i) {
        slot[i] = static_cast<int32_t>(i);
    }
    int64_t skippedSlots = 0;
    if (p.withSkip) {
        for (int64_t i = 0; i < p.bs; i += 3) {  // 每 3 行 skip 一行
            slot[i] = -1;
            ++skippedSlots;
        }
    }

    uint32_t blockDim = static_cast<uint32_t>(usedCore_);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(kv_compress_epilog, blockDim, cacheGm, xGm, slotGm, cacheOutGm, workspace, tilingGm);

    // sanity: 至少一个有效 slot 的 cache 行被写入了非零字节。
    bool anyNonZero = false;
    for (int64_t i = 0; i < p.bs && !anyNonZero; ++i) {
        if (slot[i] == -1) {
            continue;
        }
        uint8_t* row = cacheGm + slot[i] * kvCacheCol_;
        for (int64_t c = 0; c < kvCacheCol_; ++c) {
            if (row[c] != 0) {
                anyNonZero = true;
                break;
            }
        }
    }
    if (p.bs - skippedSlots > 0) {
        EXPECT_TRUE(anyNonZero) << "no written cache row is non-zero";
    }

    AscendC::GmFree(xGm);
    AscendC::GmFree(slotGm);
    AscendC::GmFree(cacheGm);
    AscendC::GmFree(cacheOutGm);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tilingGm);
}

INSTANTIATE_TEST_SUITE_P(
    QuantModes, KvCompressEpilogKernelTest,
    testing::Values(
        KvParams{32, 256, QUANT_MODE_GROUP_BF16, 1, 1, false, "group_bf16_round"},
        KvParams{32, 256, QUANT_MODE_GROUP_BF16, 0, 1, false, "group_bf16_noround"},
        KvParams{32, 256, QUANT_MODE_GROUP_E8M0, 1, 1, false, "group_e8m0_round"},
        KvParams{32, 256, QUANT_MODE_GROUP_E8M0, 0, 1, false, "group_e8m0_noround"},
        KvParams{32, 256, QUANT_MODE_HIFLOAT, 1, 1, false, "hifloat_row"},
        KvParams{32, 128, QUANT_MODE_GROUP_BF16, 1, 1, false, "group_bf16_d128"},
        KvParams{48, 256, QUANT_MODE_GROUP_E8M0, 1, 2, false, "group_e8m0_multicore"},
        KvParams{30, 256, QUANT_MODE_GROUP_BF16, 1, 1, true, "group_bf16_with_skip"}));

// 结构体字段 sanity (保留原 tiling-data 校验意图: 字段可写、布局可用)。
TEST(KvCompressEpilogTilingStruct, FieldsWritable)
{
    KvCompressEpilogTilingData t;
    std::memset(&t, 0, sizeof(t));
    t.bs = 1024;
    t.d = 256;
    t.quantMode = 1;
    t.roundScale = 1;
    t.scalesAttr = 1.0f;
    EXPECT_EQ(t.bs, 1024);
    EXPECT_EQ(t.d, 256);
    EXPECT_EQ(t.quantMode, 1);
    EXPECT_EQ(t.roundScale, 1);
    EXPECT_FLOAT_EQ(t.scalesAttr, 1.0f);
    EXPECT_GT(sizeof(KvCompressEpilogTilingData), 0u);
}
