/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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
#include "../op_host/arch22/ncai_tiling_ut_common.h"

using namespace ncai_tiling_ut;

extern "C" __global__ __aicore__ void nsa_compress_attention_infer(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* attentionMaskOptional,
    __gm__ uint8_t* blockTableOptional, __gm__ uint8_t* actualQSeqLenOptional, __gm__ uint8_t* actualCmpKvSeqLenOptional,
    __gm__ uint8_t* actualSelKvSeqLenOptional, __gm__ uint8_t* topkMaskOptional, __gm__ uint8_t* output,
    __gm__ uint8_t* topkIndicesOut, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling);

using NcaiKernelFunc = void (*)(__gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*,
                                __gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*,
                                __gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*);

class NsaCompressAttentionInferKernel : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NsaCompressAttentionInferKernel SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NsaCompressAttentionInferKernel TearDown" << std::endl;
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

    const size_t queryBytes = Product(shapes.query) * sizeof(uint16_t);
    const size_t keyBytes = Product(shapes.key) * sizeof(uint16_t);
    const size_t valueBytes = Product(shapes.value) * sizeof(uint16_t);
    const size_t maskBytes = Product({128, 128}) * sizeof(bool);
    const size_t blockTableBytes = Product(shapes.blockTable) * sizeof(int32_t);
    const size_t actualQBytes = Product({static_cast<int64_t>(shapes.actualQ.size())}) * sizeof(int64_t);
    const size_t actualCmpBytes = Product({static_cast<int64_t>(shapes.actualCmpKv.size())}) * sizeof(int64_t);
    const size_t actualSelBytes = Product({static_cast<int64_t>(shapes.actualSelKv.size())}) * sizeof(int64_t);
    const size_t topkMaskBytes = Product({20, 4096}) * sizeof(bool);
    const size_t outputBytes = Product(shapes.output) * sizeof(uint16_t);
    const size_t topkOutBytes = Product(shapes.topkIndicesOut) * sizeof(int32_t);

    uint8_t* query = AllocGm(queryBytes);
    uint8_t* key = AllocGm(keyBytes);
    uint8_t* value = AllocGm(valueBytes);
    uint8_t* attenMask = AllocGm(maskBytes);
    uint8_t* blockTable = AllocGm(blockTableBytes);
    uint8_t* actualQ = AllocGm(actualQBytes);
    uint8_t* actualCmp = AllocGm(actualCmpBytes);
    uint8_t* actualSel = AllocGm(actualSelBytes);
    uint8_t* topkMask = AllocGm(topkMaskBytes);
    uint8_t* output = AllocGm(outputBytes);
    uint8_t* topkOut = AllocGm(topkOutBytes);

    memcpy(actualQ, shapes.actualQ.data(), actualQBytes);
    memcpy(actualCmp, shapes.actualCmpKv.data(), actualCmpBytes);
    memcpy(actualSel, shapes.actualSelKv.data(), actualSelBytes);

    size_t workspaceBytes = 0;
    for (size_t ws : tilingInfo.workspaceSizes) {
        workspaceBytes += ws;
    }
    uint8_t* workspace = AllocGm(workspaceBytes);
    uint8_t* tiling = AllocGm(tilingInfo.tilingDataSize);
    memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    auto kernelFunc = reinterpret_cast<NcaiKernelFunc>(nsa_compress_attention_infer);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    ICPU_RUN_KF(kernelFunc, tilingInfo.blockNum, query, key, value, attenMask, blockTable, actualQ, actualCmp,
                actualSel, topkMask, output, topkOut, workspace, tiling);
}

}  // namespace

TEST_F(NsaCompressAttentionInferKernel, case_nsa_compress_attention_infer_0_kernel)
{
    RunKernelCase(LegacyCase0Tnd());
}

TEST_F(NsaCompressAttentionInferKernel, case_nsa_compress_attention_infer_1_kernel)
{
    RunKernelCase(LegacyCase1Bsnd());
}

TEST_F(NsaCompressAttentionInferKernel, case_nsa_compress_attention_infer_aclgraph_kernel)
{
    RunKernelCase(LegacyCaseAclGraphTnd());
}

TEST_F(NsaCompressAttentionInferKernel, case_nsa_compress_attention_infer_mtp_kernel)
{
    RunKernelCase(LegacyCaseMtpTnd());
}
