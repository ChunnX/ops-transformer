/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sparse_attn_sharedkv.cpp
 * \brief SparseAttnSharedkv AICore kernel smoke tests on CPU simulator.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "test_sparse_attn_sharedkv.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

extern "C" __global__ __aicore__ void sparse_attn_sharedkv(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
    __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKV, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling);

class SparseAttnSharedkvKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseAttnSharedkvKernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseAttnSharedkvKernelTest TearDown" << std::endl;
    }
};

TEST_F(SparseAttnSharedkvKernelTest, test_swa_fp16_tnd_pa_nd)
{
#ifdef __CCE_KT_TEST__
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    constexpr uint32_t kTokenNum = 4U;
    constexpr uint32_t kHeadNum = 64U;
    constexpr uint32_t kHeadDim = 512U;
    constexpr uint32_t kKvHeadNum = 1U;
    constexpr uint32_t kBlockSize = 128U;
    constexpr uint32_t kOriBlockNum = 4U;
    constexpr uint32_t kKvSeqLen = 512U;
    constexpr uint32_t kBlockDim = 24U;
    constexpr size_t kWorkspaceBytes = 64U * 1024U * 1024U;

    const size_t qBytes = static_cast<size_t>(kTokenNum) * kHeadNum * kHeadDim * sizeof(half);
    const size_t oriKvBytes =
        static_cast<size_t>(kOriBlockNum) * kBlockSize * kKvHeadNum * kHeadDim * sizeof(half);
    const size_t attnOutBytes = qBytes;
    const size_t sinksBytes = kHeadNum * sizeof(float);
    const size_t metadataBytes = optiling::SAS_META_SIZE * sizeof(int32_t);
    const size_t tilingBytes = sizeof(optiling::SparseAttnSharedkvTilingData);

    uint8_t *query = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(qBytes));
    uint8_t *oriKv = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(oriKvBytes));
    uint8_t *attentionOut = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(attnOutBytes));
    uint8_t *sinks = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(sinksBytes));
    uint8_t *metadata = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(metadataBytes));
    uint8_t *workspace = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(kWorkspaceBytes));
    uint8_t *tiling = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(tilingBytes));

    int32_t cuSeqlensQHost[2] = {0, static_cast<int32_t>(kTokenNum)};
    int32_t seqUsedKvHost[1] = {static_cast<int32_t>(kKvSeqLen)};
    int32_t oriBlockTableHost[4] = {0, 1, 2, 3};

    uint8_t *cuSeqlensQ = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(sizeof(cuSeqlensQHost)));
    uint8_t *seqUsedKv = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(sizeof(seqUsedKvHost)));
    uint8_t *oriBlockTable = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(sizeof(oriBlockTableHost)));

    std::memcpy(cuSeqlensQ, cuSeqlensQHost, sizeof(cuSeqlensQHost));
    std::memcpy(seqUsedKv, seqUsedKvHost, sizeof(seqUsedKvHost));
    std::memcpy(oriBlockTable, oriBlockTableHost, sizeof(oriBlockTableHost));
    std::memset(query, 0, qBytes);
    std::memset(oriKv, 0, oriKvBytes);
    std::memset(attentionOut, 0, attnOutBytes);
    std::memset(sinks, 0, sinksBytes);

    auto *tilingData = reinterpret_cast<optiling::SparseAttnSharedkvTilingData *>(tiling);
    sas_ut::InitSwaTilingData(*tilingData, kTokenNum, kKvSeqLen, kOriBlockNum, kBlockSize);
    sas_ut::InitMetadataGm(reinterpret_cast<int32_t *>(metadata), 1U, kKvHeadNum);

    ICPU_SET_TILING_KEY(sas_ut::kTndPaNdSwaTilingKey);
    ICPU_RUN_KF(sparse_attn_sharedkv, kBlockDim, query, oriKv, nullptr, nullptr, nullptr, oriBlockTable, nullptr,
                cuSeqlensQ, nullptr, nullptr, nullptr, seqUsedKv, sinks, metadata, attentionOut, nullptr, workspace,
                tiling);

    AscendC::GmFree(query);
    AscendC::GmFree(oriKv);
    AscendC::GmFree(attentionOut);
    AscendC::GmFree(sinks);
    AscendC::GmFree(metadata);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    AscendC::GmFree(cuSeqlensQ);
    AscendC::GmFree(seqUsedKv);
    AscendC::GmFree(oriBlockTable);
#else
    GTEST_SKIP() << "Kernel UT requires __CCE_KT_TEST__";
#endif
}

TEST_F(SparseAttnSharedkvKernelTest, test_scfa_fp16_tnd_pa_nd)
{
#ifdef __CCE_KT_TEST__
    AscendC::SetKernelMode(KernelMode::MIX_MODE);

    constexpr uint32_t kTokenNum = 4U;
    constexpr uint32_t kHeadNum = 64U;
    constexpr uint32_t kHeadDim = 512U;
    constexpr uint32_t kKvHeadNum = 1U;
    constexpr uint32_t kBlockSize = 128U;
    constexpr uint32_t kOriBlockNum = 4U;
    constexpr uint32_t kCmpBlockNum = 1U;
    constexpr uint32_t kKvSeqLen = 512U;
    constexpr uint32_t kTopK = 512U;
    constexpr uint32_t kBlockDim = 24U;
    constexpr size_t kWorkspaceBytes = 128U * 1024U * 1024U;

    const size_t qBytes = static_cast<size_t>(kTokenNum) * kHeadNum * kHeadDim * sizeof(half);
    const size_t oriKvBytes =
        static_cast<size_t>(kOriBlockNum) * kBlockSize * kKvHeadNum * kHeadDim * sizeof(half);
    const size_t cmpKvBytes =
        static_cast<size_t>(kCmpBlockNum) * kBlockSize * kKvHeadNum * kHeadDim * sizeof(half);
    const size_t cmpSparseBytes = static_cast<size_t>(kTokenNum) * kKvHeadNum * kTopK * sizeof(int32_t);
    const size_t attnOutBytes = qBytes;
    const size_t sinksBytes = kHeadNum * sizeof(float);
    const size_t metadataBytes = optiling::SAS_META_SIZE * sizeof(int32_t);
    const size_t tilingBytes = sizeof(optiling::SparseAttnSharedkvTilingData);

    uint8_t *query = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(qBytes));
    uint8_t *oriKv = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(oriKvBytes));
    uint8_t *cmpKv = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(cmpKvBytes));
    uint8_t *cmpSparseIndices = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(cmpSparseBytes));
    uint8_t *attentionOut = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(attnOutBytes));
    uint8_t *sinks = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(sinksBytes));
    uint8_t *metadata = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(metadataBytes));
    uint8_t *workspace = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(kWorkspaceBytes));
    uint8_t *tiling = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(tilingBytes));

    int32_t cuSeqlensQHost[2] = {0, static_cast<int32_t>(kTokenNum)};
    int32_t seqUsedKvHost[1] = {static_cast<int32_t>(kKvSeqLen)};
    int32_t oriBlockTableHost[4] = {0, 1, 2, 3};
    int32_t cmpBlockTableHost[1] = {0};

    uint8_t *cuSeqlensQ = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(sizeof(cuSeqlensQHost)));
    uint8_t *seqUsedKv = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(sizeof(seqUsedKvHost)));
    uint8_t *oriBlockTable = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(sizeof(oriBlockTableHost)));
    uint8_t *cmpBlockTable = reinterpret_cast<uint8_t *>(AscendC::GmAlloc(sizeof(cmpBlockTableHost)));

    std::memcpy(cuSeqlensQ, cuSeqlensQHost, sizeof(cuSeqlensQHost));
    std::memcpy(seqUsedKv, seqUsedKvHost, sizeof(seqUsedKvHost));
    std::memcpy(oriBlockTable, oriBlockTableHost, sizeof(oriBlockTableHost));
    std::memcpy(cmpBlockTable, cmpBlockTableHost, sizeof(cmpBlockTableHost));
    std::memset(query, 0, qBytes);
    std::memset(oriKv, 0, oriKvBytes);
    std::memset(cmpKv, 0, cmpKvBytes);
    std::memset(cmpSparseIndices, 0, cmpSparseBytes);
    std::memset(attentionOut, 0, attnOutBytes);
    std::memset(sinks, 0, sinksBytes);

    auto *tilingData = reinterpret_cast<optiling::SparseAttnSharedkvTilingData *>(tiling);
    sas_ut::InitScfaTilingData(*tilingData, kTokenNum, kKvSeqLen, kOriBlockNum, kCmpBlockNum, kBlockSize);
    sas_ut::InitMetadataGm(reinterpret_cast<int32_t *>(metadata), 1U, kKvHeadNum);

    ICPU_SET_TILING_KEY(sas_ut::kTndPaNdScfaTilingKey);
    ICPU_RUN_KF(sparse_attn_sharedkv, kBlockDim, query, oriKv, cmpKv, nullptr, cmpSparseIndices, oriBlockTable,
                cmpBlockTable, cuSeqlensQ, nullptr, nullptr, nullptr, seqUsedKv, sinks, metadata, attentionOut, nullptr,
                workspace, tiling);

    AscendC::GmFree(query);
    AscendC::GmFree(oriKv);
    AscendC::GmFree(cmpKv);
    AscendC::GmFree(cmpSparseIndices);
    AscendC::GmFree(attentionOut);
    AscendC::GmFree(sinks);
    AscendC::GmFree(metadata);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    AscendC::GmFree(cuSeqlensQ);
    AscendC::GmFree(seqUsedKv);
    AscendC::GmFree(oriBlockTable);
    AscendC::GmFree(cmpBlockTable);
#else
    GTEST_SKIP() << "Kernel UT requires __CCE_KT_TEST__";
#endif
}
