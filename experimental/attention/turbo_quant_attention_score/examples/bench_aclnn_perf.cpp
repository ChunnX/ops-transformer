/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_turbo_quant_attention_score.h"

#define CHECK_ACL(expr) \
    do { \
        const auto status = (expr); \
        if (status != ACL_SUCCESS) { \
            std::printf("%s failed: %d\n", #expr, static_cast<int>(status)); \
            return 1; \
        } \
    } while (0)

namespace {
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t MSE_BITS = 3;
constexpr int WARMUP = 10;
constexpr int SAMPLES = 60;
constexpr int DROP_MAX = 10;

uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

template <typename T>
int CreateTensor(const std::vector<T> &data, const std::vector<int64_t> &shape, aclDataType dtype, void **device,
                 aclTensor **tensor)
{
    const size_t bytes = data.size() * sizeof(T);
    CHECK_ACL(aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(*device, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *device);
    return *tensor == nullptr ? 1 : ACL_SUCCESS;
}

struct CaseCfg {
    const char *name;
    int64_t tokens;
    int64_t qHeads;
    int64_t kvHeads;
    int64_t kvLen;
    double baselineUs;
};

double MeanDropMax(std::vector<double> times)
{
    std::sort(times.begin(), times.end());
    const int keep = std::max(1, static_cast<int>(times.size()) - DROP_MAX);
    const double sum = std::accumulate(times.begin(), times.begin() + keep, 0.0);
    return sum / static_cast<double>(keep);
}

int RunCase(const CaseCfg &cfg)
{
    const int64_t numBlocks = (cfg.kvLen + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t idxBytes = (HEAD_DIM * MSE_BITS + 7) / 8;
    const int64_t qjlBytes = (HEAD_DIM + 7) / 8;
    const int64_t maxKv = cfg.kvLen;

    std::vector<uint16_t> query(cfg.tokens * cfg.qHeads * HEAD_DIM);
    std::vector<uint16_t> queryQjl(query.size());
    for (size_t i = 0; i < query.size(); ++i) {
        const float v = static_cast<float>((static_cast<int>(i % 17) - 8) * 0.03125F);
        query[i] = FloatToBf16Bits(v);
        queryQjl[i] = FloatToBf16Bits(-v * 0.5F);
    }
    std::vector<uint8_t> idx(numBlocks * BLOCK_SIZE * cfg.kvHeads * idxBytes);
    std::vector<uint8_t> qjl(numBlocks * BLOCK_SIZE * cfg.kvHeads * qjlBytes);
    for (size_t i = 0; i < idx.size(); ++i) {
        idx[i] = static_cast<uint8_t>((i * 3U) & 0xFFU);
    }
    for (size_t i = 0; i < qjl.size(); ++i) {
        qjl[i] = static_cast<uint8_t>((i * 5U) & 0xFFU);
    }
    std::vector<uint16_t> norm(numBlocks * BLOCK_SIZE * cfg.kvHeads);
    std::vector<uint16_t> gamma(norm.size());
    for (size_t i = 0; i < norm.size(); ++i) {
        norm[i] = FloatToBf16Bits(1.0F);
        gamma[i] = FloatToBf16Bits(0.5F);
    }
    std::vector<int32_t> table(numBlocks);
    for (int64_t i = 0; i < numBlocks; ++i) {
        table[i] = static_cast<int32_t>(i);
    }
    std::vector<int32_t> seq{static_cast<int32_t>(cfg.kvLen)};
    std::vector<float> out(cfg.tokens * cfg.qHeads * maxKv, 0.0F);

    void *qDev = nullptr;
    void *qjlDev = nullptr;
    void *idxDev = nullptr;
    void *qjlPDev = nullptr;
    void *normDev = nullptr;
    void *gammaDev = nullptr;
    void *tableDev = nullptr;
    void *seqDev = nullptr;
    void *outDev = nullptr;
    aclTensor *qT = nullptr;
    aclTensor *qjlT = nullptr;
    aclTensor *idxT = nullptr;
    aclTensor *qjlPT = nullptr;
    aclTensor *normT = nullptr;
    aclTensor *gammaT = nullptr;
    aclTensor *tableT = nullptr;
    aclTensor *seqT = nullptr;
    aclTensor *outT = nullptr;

    const std::vector<int64_t> qShape{cfg.tokens, cfg.qHeads, HEAD_DIM};
    const std::vector<int64_t> idxShape{numBlocks, BLOCK_SIZE, cfg.kvHeads, idxBytes};
    const std::vector<int64_t> qjlShape{numBlocks, BLOCK_SIZE, cfg.kvHeads, qjlBytes};
    const std::vector<int64_t> scShape{numBlocks, BLOCK_SIZE, cfg.kvHeads};
    const std::vector<int64_t> tableShape{1, numBlocks};
    const std::vector<int64_t> seqShape{1};
    const std::vector<int64_t> outShape{cfg.tokens, cfg.qHeads, maxKv};

    CHECK_ACL(CreateTensor(query, qShape, ACL_BF16, &qDev, &qT));
    CHECK_ACL(CreateTensor(queryQjl, qShape, ACL_BF16, &qjlDev, &qjlT));
    CHECK_ACL(CreateTensor(idx, idxShape, ACL_UINT8, &idxDev, &idxT));
    CHECK_ACL(CreateTensor(qjl, qjlShape, ACL_UINT8, &qjlPDev, &qjlPT));
    CHECK_ACL(CreateTensor(norm, scShape, ACL_BF16, &normDev, &normT));
    CHECK_ACL(CreateTensor(gamma, scShape, ACL_BF16, &gammaDev, &gammaT));
    CHECK_ACL(CreateTensor(table, tableShape, ACL_INT32, &tableDev, &tableT));
    CHECK_ACL(CreateTensor(seq, seqShape, ACL_INT32, &seqDev, &seqT));
    CHECK_ACL(CreateTensor(out, outShape, ACL_FLOAT, &outDev, &outT));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    auto launch = [&]() -> int {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        CHECK_ACL(aclnnTurboQuantAttentionScoreGetWorkspaceSize(qT, qjlT, idxT, qjlPT, normT, gammaT, tableT, seqT,
                                                                MSE_BITS, outT, &workspaceSize, &executor));
        void *workspace = nullptr;
        if (workspaceSize > 0) {
            CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        CHECK_ACL(aclnnTurboQuantAttentionScore(workspace, workspaceSize, executor, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        return 0;
    };

    for (int i = 0; i < WARMUP; ++i) {
        if (launch() != 0) {
            return 1;
        }
    }
    std::vector<double> times;
    times.reserve(SAMPLES);
    for (int i = 0; i < SAMPLES; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (launch() != 0) {
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    const double meanUs = MeanDropMax(times);
    const double ratio = meanUs / cfg.baselineUs;
    std::printf("case=%s latency_us=%.3f baseline_us=%.3f ratio=%.3f pass_1p2x=%s\n", cfg.name, meanUs, cfg.baselineUs,
                ratio, ratio <= 1.2 ? "YES" : "NO");

    aclDestroyTensor(qT);
    aclDestroyTensor(qjlT);
    aclDestroyTensor(idxT);
    aclDestroyTensor(qjlPT);
    aclDestroyTensor(normT);
    aclDestroyTensor(gammaT);
    aclDestroyTensor(tableT);
    aclDestroyTensor(seqT);
    aclDestroyTensor(outT);
    aclrtFree(qDev);
    aclrtFree(qjlDev);
    aclrtFree(idxDev);
    aclrtFree(qjlPDev);
    aclrtFree(normDev);
    aclrtFree(gammaDev);
    aclrtFree(tableDev);
    aclrtFree(seqDev);
    aclrtFree(outDev);
    aclrtDestroyStream(stream);
    return 0;
}
} // namespace

int main(int argc, char **argv)
{
    // Paged Full-QK baselines from task book; prefill_kv32k left blank there,
    // so use kv4k * (32768/4096) linear extrapolation.
    const CaseCfg cases[] = {
        {"gqa_decode_kv4k", 1, 32, 8, 4096, 1501.634},        {"gqa_decode_kv32k", 1, 32, 8, 32768, 9278.507},
        {"gqa_decode_kv128k", 1, 32, 8, 131072, 35681.690},   {"gqa_prefill_kv4k", 512, 32, 8, 4096, 75984.253},
        {"gqa_prefill_kv32k", 512, 32, 8, 32768, 607874.024},
    };
    std::string only;
    if (argc > 1) {
        only = argv[1];
    }
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    for (const auto &cfg : cases) {
        if (!only.empty() && only != cfg.name) {
            continue;
        }
        if (RunCase(cfg) != 0) {
            return 1;
        }
    }
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
