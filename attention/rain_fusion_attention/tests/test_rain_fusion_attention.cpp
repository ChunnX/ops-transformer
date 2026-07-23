/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn_rain_fusion_attention.h"

enum TestResult {
    TEST_PASS,
    TEST_FAIL,
    TEST_SKIP
};

struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    int skipped = 0;
};

struct TestConfig {
    int64_t batch;
    int64_t qSeqlen;
    int64_t kvSeqlen;
    int64_t numHeads;
    int64_t numKvHeads;
    int64_t headDim;
    int64_t blockShapeX;
    int64_t blockShapeY;
    float sparsity;
    std::string testName;
};

static aclrtStream g_stream = nullptr;
static TestStats g_stats;

static inline float HalfToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f32;
    if (exp == 0) {
        f32 = (sign << 31) | (mant << 13);
    } else if (exp == 0x1f) {
        f32 = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f32, sizeof(float));
    return result;
}

static inline uint16_t FloatToHalf(float f)
{
    uint32_t f32;
    std::memcpy(&f32, &f, sizeof(uint32_t));
    uint32_t sign = (f32 >> 16) & 0x8000;
    int32_t exp  = static_cast<int32_t>((f32 >> 23) & 0xff) - 127;
    uint32_t mant = f32 & 0x7fffff;
    if (exp > 15) return static_cast<uint16_t>(sign | 0x7c00);
    if (exp < -14) return static_cast<uint16_t>(sign);
    uint32_t hExp = static_cast<uint32_t>(exp + 15);
    uint32_t hMant = (mant >> 13) & 0x3ff;
    uint16_t rounding = (mant >> 12) & 1;
    return static_cast<uint16_t>(sign | (hExp << 10) | hMant + rounding);
}

void GenerateRandomFp16(std::vector<uint16_t>& data, int64_t count)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    data.resize(count);
    for (int64_t i = 0; i < count; ++i) {
        data[i] = FloatToHalf(dis(gen));
    }
}

void GenerateSparsePattern(
    int32_t qBlockNum, int32_t kvBlockNum, int32_t numHeads,
    float sparsity,
    std::vector<int64_t> &selectIdx,
    std::vector<int64_t> &selectNumIdx)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    int32_t maxKvBlockNum = kvBlockNum;
    selectIdx.resize(qBlockNum * numHeads * maxKvBlockNum, -1);
    selectNumIdx.resize(qBlockNum * numHeads, 0);

    for (int32_t qb = 0; qb < qBlockNum; ++qb) {
        for (int32_t h = 0; h < numHeads; ++h) {
            std::vector<int32_t> selected;
            for (int32_t kvb = 0; kvb < kvBlockNum; ++kvb) {
                if (dis(gen) < sparsity) {
                    selected.push_back(kvb);
                }
            }
            if (selected.empty()) selected.push_back(0);

            int32_t baseIdx = (qb * numHeads + h) * maxKvBlockNum;
            for (size_t i = 0; i < selected.size(); ++i) {
                selectIdx[static_cast<size_t>(baseIdx) + i] = static_cast<int64_t>(selected[i]);
            }
            selectNumIdx[qb * numHeads + h] = static_cast<int64_t>(selected.size());
        }
    }
}

void RainFusionAttentionCpuRef(
    const uint16_t* qHost,
    const uint16_t* kHost,
    const uint16_t* vHost,
    const int64_t* selIdxHost,
    const int64_t* selNumHost,
    float* oRef,
    int64_t batch,
    int64_t qSeqlen,
    int64_t kvSeqlen,
    int64_t numHeads,
    int64_t numKvHeads,
    int64_t headDim,
    int64_t blockShapeX,
    int64_t blockShapeY,
    int64_t maxKvBlockNum,
    float scale)
{
    int64_t totalQTokens = batch * qSeqlen;
    int64_t totalKvTokens = batch * kvSeqlen;
    int64_t qBlockNum = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int64_t groupSize = numHeads / numKvHeads;

    int64_t qStrideT = numHeads * headDim;
    int64_t kvStrideT = numKvHeads * headDim;

    std::fill(oRef, oRef + totalQTokens * numHeads * headDim, 0.0f);

    for (int64_t b = 0; b < batch; ++b) {
        int64_t qBatchOffset = b * qSeqlen * qStrideT;
        int64_t kvBatchOffset = b * kvSeqlen * kvStrideT;
        int64_t selBatchOffset = b * qBlockNum;

        for (int64_t qb = 0; qb < qBlockNum; ++qb) {
            int64_t qStart = qb * blockShapeX;
            int64_t qEnd = std::min(qStart + blockShapeX, qSeqlen);
            int64_t qBlockSize = qEnd - qStart;

            for (int64_t h = 0; h < numHeads; ++h) {
                int64_t kvH = h / groupSize;
                int64_t selIdxBase = (selBatchOffset + qb) * numHeads + h;
                int64_t numSel = selNumHost[selIdxBase];

                if (numSel <= 0) {
                    continue;
                }

                std::vector<float> Q_chunk(qBlockSize * headDim);
                for (int64_t i = 0; i < qBlockSize; ++i) {
                    int64_t qRow = qBatchOffset + (qStart + i) * qStrideT + h * headDim;
                    for (int64_t d = 0; d < headDim; ++d) {
                        Q_chunk[i * headDim + d] = HalfToFloat(qHost[qRow + d]);
                    }
                }

                std::vector<float> O_block(qBlockSize * headDim, 0.0f);
                std::vector<float> m(qBlockSize, -std::numeric_limits<float>::infinity());
                std::vector<float> l(qBlockSize, 0.0f);

                for (int64_t si = 0; si < numSel; ++si) {
                    int64_t kvBlockIdx = selIdxHost[selIdxBase * maxKvBlockNum + si];
                    if (kvBlockIdx < 0) break;

                    int64_t kvStart = kvBlockIdx * blockShapeY;
                    int64_t kvEnd = std::min(kvStart + blockShapeY, kvSeqlen);
                    int64_t kvBlockSize = kvEnd - kvStart;

                    std::vector<float> K_chunk(kvBlockSize * headDim);
                    std::vector<float> V_chunk(kvBlockSize * headDim);
                    for (int64_t i = 0; i < kvBlockSize; ++i) {
                        int64_t kvRow = kvBatchOffset + (kvStart + i) * kvStrideT + kvH * headDim;
                        for (int64_t d = 0; d < headDim; ++d) {
                            K_chunk[i * headDim + d] = HalfToFloat(kHost[kvRow + d]);
                            V_chunk[i * headDim + d] = HalfToFloat(vHost[kvRow + d]);
                        }
                    }

                    std::vector<float> S(qBlockSize * kvBlockSize);
                    for (int64_t i = 0; i < qBlockSize; ++i) {
                        for (int64_t j = 0; j < kvBlockSize; ++j) {
                            float sum = 0.0f;
                            for (int64_t d = 0; d < headDim; ++d) {
                                sum += Q_chunk[i * headDim + d] *
                                       K_chunk[j * headDim + d];
                            }
                            S[i * kvBlockSize + j] = sum * scale;
                        }
                    }

                    for (int64_t i = 0; i < qBlockSize; ++i) {
                        float rowMax = -std::numeric_limits<float>::infinity();
                        for (int64_t j = 0; j < kvBlockSize; ++j) {
                            rowMax = std::max(rowMax, S[i * kvBlockSize + j]);
                        }
                        float mNew = std::max(m[i], rowMax);

                        float rowSum = 0.0f;
                        for (int64_t j = 0; j < kvBlockSize; ++j) {
                            float pVal = std::exp(S[i * kvBlockSize + j] - mNew);
                            S[i * kvBlockSize + j] = pVal;
                            rowSum += pVal;
                        }
                        float lNew = std::exp(m[i] - mNew) * l[i] + rowSum;

                        float rescaleO = std::exp(m[i] - mNew);
                        for (int64_t d = 0; d < headDim; ++d) {
                            O_block[i * headDim + d] *= rescaleO;
                        }
                        for (int64_t j = 0; j < kvBlockSize; ++j) {
                            float pVal = S[i * kvBlockSize + j];
                            for (int64_t d = 0; d < headDim; ++d) {
                                O_block[i * headDim + d] += pVal *
                                    V_chunk[j * headDim + d];
                            }
                        }

                        m[i] = mNew;
                        l[i] = lNew;
                    }
                }

                for (int64_t i = 0; i < qBlockSize; ++i) {
                    float invL = 1.0f / l[i];
                    for (int64_t d = 0; d < headDim; ++d) {
                        O_block[i * headDim + d] *= invL;
                    }
                }

                for (int64_t i = 0; i < qBlockSize; ++i) {
                    int64_t oRow = qBatchOffset + (qStart + i) * qStrideT + h * headDim;
                    for (int64_t d = 0; d < headDim; ++d) {
                        oRef[oRow + d] = O_block[i * headDim + d];
                    }
                }
            }
        }
    }
}

bool CompareResults(const uint16_t* oNpu, const float* oRef, int64_t totalElements,
                    float atol, float rtol, float* outMaxDiff, int64_t* outMaxIdx)
{
    float maxDiff = 0.0f;
    int64_t maxIdx = -1;
    bool allPass = true;

    for (int64_t i = 0; i < totalElements; ++i) {
        float npuVal = HalfToFloat(oNpu[i]);
        float refVal = oRef[i];
        float diff = std::fabs(npuVal - refVal);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxIdx = i;
        }
        float threshold = atol + rtol * std::fabs(refVal);
        if (diff > threshold) {
            allPass = false;
        }
    }

    *outMaxDiff = maxDiff;
    *outMaxIdx = maxIdx;
    return allPass;
}

TestResult RunTest(const TestConfig &config)
{
    std::cout << "\n--- Test: " << config.testName << " ---" << std::endl;
    std::cout << "Config: batch=" << config.batch
              << ", qSeq=" << config.qSeqlen
              << ", kvSeq=" << config.kvSeqlen
              << ", heads=" << config.numHeads
              << ", blockShape=[" << config.blockShapeX << "," << config.blockShapeY << "]" << std::endl;

    int64_t totalQTokens = config.batch * config.qSeqlen;
    int64_t totalKvTokens = config.batch * config.kvSeqlen;
    int32_t qBlockNum = static_cast<int32_t>((config.qSeqlen + config.blockShapeX - 1) / config.blockShapeX);
    int32_t kvBlockNum = static_cast<int32_t>((config.kvSeqlen + config.blockShapeY - 1) / config.blockShapeY);

    void *qDev = nullptr;
    void *kDev = nullptr;
    void *vDev = nullptr;
    void *oDev = nullptr;
    void *selIdxDev = nullptr;
    void *selNumDev = nullptr;

    aclTensor *qTensor = nullptr;
    aclTensor *kTensor = nullptr;
    aclTensor *vTensor = nullptr;
    aclTensor *oTensor = nullptr;
    aclTensor *selIdxTensor = nullptr;
    aclTensor *selNumTensor = nullptr;
    aclIntArray *blockShapeArray = nullptr;
    aclIntArray *actualSeqLenQ = nullptr;
    aclIntArray *actualSeqLenKv = nullptr;

    try {
        int64_t qElementCount = totalQTokens * config.numHeads * config.headDim;
        int64_t kvElementCount = totalKvTokens * config.numKvHeads * config.headDim;
        int64_t oElementCount = qElementCount;

        size_t qSize = qElementCount * sizeof(uint16_t);
        size_t kvSize = kvElementCount * sizeof(uint16_t);

        std::vector<uint16_t> qHost(qElementCount);
        std::vector<uint16_t> kHost(kvElementCount);
        std::vector<uint16_t> vHost(kvElementCount);
        std::vector<uint16_t> oHostInit(oElementCount, 0);

        GenerateRandomFp16(qHost, qElementCount);
        GenerateRandomFp16(kHost, kvElementCount);
        GenerateRandomFp16(vHost, kvElementCount);

        aclrtMalloc(&qDev, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&kDev, kvSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&vDev, kvSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&oDev, qSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclrtMemcpy(qDev, qSize, qHost.data(), qSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(kDev, kvSize, kHost.data(), kvSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(vDev, kvSize, vHost.data(), kvSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(oDev, qSize, oHostInit.data(), qSize, ACL_MEMCPY_HOST_TO_DEVICE);

        std::vector<int64_t> selIdxHost, selNumHost;
        GenerateSparsePattern(qBlockNum, kvBlockNum, static_cast<int32_t>(config.numHeads),
            config.sparsity, selIdxHost, selNumHost);

        int64_t maxKvBlockNum = static_cast<int64_t>(kvBlockNum);

        size_t selIdxSize = selIdxHost.size() * sizeof(int64_t);
        size_t selNumSize = selNumHost.size() * sizeof(int64_t);

        aclrtMalloc(&selIdxDev, selIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&selNumDev, selNumSize, ACL_MEM_MALLOC_HUGE_FIRST);

        aclrtMemcpy(selIdxDev, selIdxSize, selIdxHost.data(), selIdxSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(selNumDev, selNumSize, selNumHost.data(), selNumSize, ACL_MEMCPY_HOST_TO_DEVICE);

        std::vector<int64_t> qShape = {totalQTokens, config.numHeads, config.headDim};
        std::vector<int64_t> kvShape = {totalKvTokens, config.numKvHeads, config.headDim};
        std::vector<int64_t> selIdxShape = {
            static_cast<int64_t>(qBlockNum) * config.batch,
            config.numHeads,
            maxKvBlockNum
        };
        std::vector<int64_t> selNumShape = {
            static_cast<int64_t>(qBlockNum) * config.batch,
            config.numHeads
        };

        qTensor = aclCreateTensor(qShape.data(), qShape.size(), ACL_FLOAT16,
                                  nullptr, 0, ACL_FORMAT_ND, qShape.data(), qShape.size(), qDev);
        kTensor = aclCreateTensor(kvShape.data(), kvShape.size(), ACL_FLOAT16,
                                  nullptr, 0, ACL_FORMAT_ND, kvShape.data(), kvShape.size(), kDev);
        vTensor = aclCreateTensor(kvShape.data(), kvShape.size(), ACL_FLOAT16,
                                  nullptr, 0, ACL_FORMAT_ND, kvShape.data(), kvShape.size(), vDev);
        oTensor = aclCreateTensor(qShape.data(), qShape.size(), ACL_FLOAT16,
                                  nullptr, 0, ACL_FORMAT_ND, qShape.data(), qShape.size(), oDev);
        selIdxTensor = aclCreateTensor(selIdxShape.data(), selIdxShape.size(), ACL_INT64,
                                       nullptr, 0, ACL_FORMAT_ND,
                                       selIdxShape.data(), selIdxShape.size(), selIdxDev);
        selNumTensor = aclCreateTensor(selNumShape.data(), selNumShape.size(), ACL_INT64,
                                       nullptr, 0, ACL_FORMAT_ND,
                                       selNumShape.data(), selNumShape.size(), selNumDev);

        int64_t blockShapeData[2] = {config.blockShapeX, config.blockShapeY};
        blockShapeArray = aclCreateIntArray(blockShapeData, 2);

        std::vector<int64_t> actualSeqQData(config.batch, config.qSeqlen);
        std::vector<int64_t> actualSeqKvData(config.batch, config.kvSeqlen);
        actualSeqLenQ = aclCreateIntArray(actualSeqQData.data(), static_cast<int64_t>(config.batch));
        actualSeqLenKv = aclCreateIntArray(actualSeqKvData.data(), static_cast<int64_t>(config.batch));

        char qLayoutStr[] = "TND";
        char kvLayoutStr[] = "TND";
        double scaleValue = 1.0 / std::sqrt(static_cast<double>(config.headDim));

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        aclnnStatus ret = aclnnRainFusionAttentionGetWorkspaceSize(
            qTensor, kTensor, vTensor,
            selIdxTensor, selNumTensor,
            blockShapeArray,
            nullptr,
            actualSeqLenQ, actualSeqLenKv,
            nullptr,
            qLayoutStr, kvLayoutStr,
            config.numKvHeads, 0,
            scaleValue, 1, 128,
            oTensor, nullptr,
            &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("GetWorkspaceSize failed with code: " + std::to_string(ret));
        }

        void *workspace = nullptr;
        if (workspaceSize > 0) {
            aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        ret = aclnnRainFusionAttention(workspace, workspaceSize, executor, g_stream);
        if (ret != ACL_SUCCESS) {
            if (workspace) aclrtFree(workspace);
            throw std::runtime_error("RainFusionAttention failed with code: " + std::to_string(ret));
        }

        aclrtSynchronizeStream(g_stream);

        std::vector<uint16_t> oNpu(oElementCount);
        aclrtMemcpy(oNpu.data(), qSize, oDev, qSize, ACL_MEMCPY_DEVICE_TO_HOST);

        std::vector<float> oRef(oElementCount);
        RainFusionAttentionCpuRef(
            qHost.data(), kHost.data(), vHost.data(),
            selIdxHost.data(), selNumHost.data(),
            oRef.data(),
            config.batch, config.qSeqlen, config.kvSeqlen,
            config.numHeads, config.numKvHeads, config.headDim,
            config.blockShapeX, config.blockShapeY,
            maxKvBlockNum, static_cast<float>(scaleValue));

        float atol = 5e-2f;
        float rtol = 5e-2f;
        float maxDiff = 0.0f;
        int64_t maxIdx = -1;
        bool cmpResult = CompareResults(oNpu.data(), oRef.data(), oElementCount,
                                        atol, rtol, &maxDiff, &maxIdx);
        if (cmpResult) {
            std::cout << "Result: PASS (max_diff=" << maxDiff << ")" << std::endl;
        } else {
            std::cout << "Result: FAIL - numerical mismatch" << std::endl;
            std::cout << "  max_diff = " << maxDiff
                      << " at index " << maxIdx
                      << " (atol=" << atol << ", rtol=" << rtol << ")" << std::endl;
        }

        if (workspace) aclrtFree(workspace);
        aclDestroyTensor(qTensor);
        aclDestroyTensor(kTensor);
        aclDestroyTensor(vTensor);
        aclDestroyTensor(oTensor);
        aclDestroyTensor(selIdxTensor);
        aclDestroyTensor(selNumTensor);
        aclDestroyIntArray(blockShapeArray);
        aclDestroyIntArray(actualSeqLenQ);
        aclDestroyIntArray(actualSeqLenKv);

        aclrtFree(qDev);
        aclrtFree(kDev);
        aclrtFree(vDev);
        aclrtFree(oDev);
        aclrtFree(selIdxDev);
        aclrtFree(selNumDev);

        return cmpResult ? TEST_PASS : TEST_FAIL;
    } catch (const std::exception &e) {
        std::cerr << "Result: FAIL - " << e.what() << std::endl;

        if (qTensor) aclDestroyTensor(qTensor);
        if (kTensor) aclDestroyTensor(kTensor);
        if (vTensor) aclDestroyTensor(vTensor);
        if (oTensor) aclDestroyTensor(oTensor);
        if (selIdxTensor) aclDestroyTensor(selIdxTensor);
        if (selNumTensor) aclDestroyTensor(selNumTensor);
        if (blockShapeArray) aclDestroyIntArray(blockShapeArray);
        if (actualSeqLenQ) aclDestroyIntArray(actualSeqLenQ);
        if (actualSeqLenKv) aclDestroyIntArray(actualSeqLenKv);

        if (qDev) aclrtFree(qDev);
        if (kDev) aclrtFree(kDev);
        if (vDev) aclrtFree(vDev);
        if (oDev) aclrtFree(oDev);
        if (selIdxDev) aclrtFree(selIdxDev);
        if (selNumDev) aclrtFree(selNumDev);

        return TEST_FAIL;
    }
}

int main()
{
    std::cout << "======================================" << std::endl;
    std::cout << "  RainFusionAttention Test Suite" << std::endl;
    std::cout << "======================================\n" << std::endl;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtCreateStream(&g_stream);

    std::vector<TestConfig> tests = {
        {1, 256, 512, 8, 8, 128, 64, 64, 0.3f, "Basic TND Format"},
        {1, 256, 512, 8, 8, 128, 32, 32, 0.3f, "Small Block Shape (32x32)"},
        {1, 512, 1024, 8, 8, 128, 128, 128, 0.25f, "Large Block Shape (128x128)"},
        {1, 256, 512, 8, 8, 128, 64, 128, 0.3f, "Asymmetric Block Shape (64x128)"},
        {1, 250, 500, 8, 8, 128, 64, 64, 0.3f, "Non-aligned Sequence Length"},
        {1, 64, 128, 8, 8, 128, 32, 32, 0.5f, "Small Sequence Length"},
        {1, 256, 512, 1, 1, 128, 64, 64, 0.3f, "Single Head"},
        {2, 128, 256, 8, 8, 128, 64, 64, 0.3f, "Multiple Batch (2)"},
        {1, 256, 512, 8, 8, 128, 64, 64, 0.1f, "High Sparsity (10%)"},
        {1, 256, 512, 8, 8, 128, 64, 64, 0.8f, "Low Sparsity (80%)"},
    };

    for (const auto &test : tests) {
        g_stats.total++;
        TestResult result = RunTest(test);

        switch (result) {
            case TEST_PASS:
                g_stats.passed++;
                break;
            case TEST_FAIL:
                g_stats.failed++;
                break;
            case TEST_SKIP:
                g_stats.skipped++;
                break;
        }
    }

    std::cout << "\n======================================" << std::endl;
    std::cout << "  Test Summary" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Total tests:   " << g_stats.total << std::endl;
    std::cout << "Passed:        " << g_stats.passed << " ("
              << (100.0 * g_stats.passed / g_stats.total) << "%)" << std::endl;
    std::cout << "Failed:        " << g_stats.failed << std::endl;
    std::cout << "Skipped:       " << g_stats.skipped << std::endl;
    std::cout << "======================================\n" << std::endl;

    aclrtDestroyStream(g_stream);
    aclrtResetDevice(0);
    aclFinalize();

    return (g_stats.failed == 0) ? 0 : 1;
}
