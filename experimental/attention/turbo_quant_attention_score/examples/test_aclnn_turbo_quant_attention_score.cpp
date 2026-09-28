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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_turbo_quant_attention_score.h"

#define CHECK_ACL(expr) \
    do { \
        const auto status = (expr); \
        if (status != ACL_SUCCESS) { \
            std::printf("%s failed: %d\n", #expr, static_cast<int>(status)); \
            return status; \
        } \
    } while (0)

namespace {
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t MSE_BITS = 3;
constexpr float PI = 3.14159265358979323846F;
constexpr float CENTROIDS3[] = {-0.19020693F,   -0.1187859178F, -0.06682205945F, -0.02166347019F,
                                0.02166347019F, 0.06682205945F, 0.1187859178F,   0.19020693F};

uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

std::vector<uint16_t> ToBf16(const std::vector<float> &src)
{
    std::vector<uint16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = FloatToBf16Bits(src[i]);
    }
    return dst;
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
    return *tensor == nullptr ? -1 : ACL_SUCCESS;
}

void PackMse3(const std::vector<uint8_t> &indices, uint8_t *packed)
{
    for (int64_t t = 0; t < 16; ++t) {
        uint32_t word = 0;
        for (int64_t k = 0; k < 8; ++k) {
            word |= (static_cast<uint32_t>(indices[t * 8 + k] & 7U) << (3 * k));
        }
        packed[t * 3 + 0] = static_cast<uint8_t>(word & 0xFFU);
        packed[t * 3 + 1] = static_cast<uint8_t>((word >> 8) & 0xFFU);
        packed[t * 3 + 2] = static_cast<uint8_t>((word >> 16) & 0xFFU);
    }
}

void PackQjl(const std::vector<int8_t> &signs, uint8_t *packed)
{
    for (int64_t b = 0; b < 16; ++b) {
        uint8_t value = 0;
        for (int64_t s = 0; s < 8; ++s) {
            if (signs[b * 8 + s] > 0) {
                value |= static_cast<uint8_t>(1U << s);
            }
        }
        packed[b] = value;
    }
}

std::vector<float> Golden(const std::vector<float> &query, const std::vector<float> &queryQjl,
                          const std::vector<uint8_t> &idxPacked, const std::vector<uint8_t> &qjlPacked,
                          const std::vector<float> &norm, const std::vector<float> &gamma,
                          const std::vector<int32_t> &blockTable, const std::vector<int32_t> &seqLens, int64_t tokens,
                          int64_t qHeads, int64_t kvHeads, int64_t maxBlocks, int64_t maxKvLen)
{
    const float coeff = std::sqrt(PI / 2.0F) / static_cast<float>(HEAD_DIM);
    const int64_t groups = qHeads / kvHeads;
    const int64_t batch = static_cast<int64_t>(seqLens.size());
    std::vector<float> out(tokens * qHeads * maxKvLen, 0.0F);

    for (int64_t t = 0; t < tokens; ++t) {
        const int64_t batchIdx = (batch == 1) ? 0 : t;
        for (int64_t h = 0; h < qHeads; ++h) {
            const int64_t kvHead = h / groups;
            const int64_t qBase = (t * qHeads + h) * HEAD_DIM;
            for (int64_t pos = 0; pos < maxKvLen; ++pos) {
                const int64_t outOff = (t * qHeads + h) * maxKvLen + pos;
                if (pos >= seqLens[batchIdx]) {
                    out[outOff] = -INFINITY;
                    continue;
                }
                const int64_t phys = blockTable[batchIdx * maxBlocks + pos / BLOCK_SIZE];
                const int64_t offset = pos % BLOCK_SIZE;
                const int64_t scalar = (phys * BLOCK_SIZE + offset) * kvHeads + kvHead;
                const uint8_t *idxBytes = idxPacked.data() + scalar * 48;
                const uint8_t *qjlBytes = qjlPacked.data() + scalar * 16;

                float mainDot = 0.0F;
                float corrDot = 0.0F;
                int64_t dim = 0;
                for (int64_t trip = 0; trip < 16; ++trip) {
                    const uint32_t word = static_cast<uint32_t>(idxBytes[trip * 3]) |
                                          (static_cast<uint32_t>(idxBytes[trip * 3 + 1]) << 8) |
                                          (static_cast<uint32_t>(idxBytes[trip * 3 + 2]) << 16);
                    for (int64_t s = 0; s < 24; s += 3) {
                        const uint32_t index = (word >> s) & 7U;
                        mainDot += query[qBase + dim] * CENTROIDS3[index];
                        const float sign = ((qjlBytes[dim / 8] >> (dim % 8)) & 1) ? 1.0F : -1.0F;
                        corrDot += queryQjl[qBase + dim] * sign;
                        ++dim;
                    }
                }
                out[outOff] = norm[scalar] * mainDot + coeff * gamma[scalar] * corrDot;
            }
        }
    }
    return out;
}
} // namespace

int main()
{
    // Small smoke: T=1, H=2, KvH=1, 1 block, valid kv_len=4 (rest -inf to 128).
    constexpr int64_t T = 1;
    constexpr int64_t H = 2;
    constexpr int64_t KVH = 1;
    constexpr int64_t NUM_BLOCKS = 1;
    constexpr int64_t MAX_BLOCKS = 1;
    constexpr int64_t MAX_KV = BLOCK_SIZE;
    constexpr int64_t SEQ_LEN = 4;
    constexpr int64_t IDX_BYTES = 48;
    constexpr int64_t QJL_BYTES = 16;

    std::vector<float> query(T * H * HEAD_DIM);
    std::vector<float> queryQjl(T * H * HEAD_DIM);
    for (size_t i = 0; i < query.size(); ++i) {
        query[i] = static_cast<float>(static_cast<int32_t>(i % 11U) - 5) * 0.0625F;
        queryQjl[i] = static_cast<float>(static_cast<int32_t>((i * 3U) % 13U) - 6) * 0.03125F;
    }

    std::vector<uint8_t> idxPacked(NUM_BLOCKS * BLOCK_SIZE * KVH * IDX_BYTES, 0);
    std::vector<uint8_t> qjlPacked(NUM_BLOCKS * BLOCK_SIZE * KVH * QJL_BYTES, 0);
    std::vector<float> norm(NUM_BLOCKS * BLOCK_SIZE * KVH, 1.0F);
    std::vector<float> gamma(NUM_BLOCKS * BLOCK_SIZE * KVH, 1.0F);
    for (int64_t pos = 0; pos < SEQ_LEN; ++pos) {
        std::vector<uint8_t> indices(HEAD_DIM);
        std::vector<int8_t> signs(HEAD_DIM);
        for (int64_t d = 0; d < HEAD_DIM; ++d) {
            indices[d] = static_cast<uint8_t>((pos + d) % 8);
            signs[d] = ((pos + d) & 1) ? 1 : -1;
        }
        PackMse3(indices, idxPacked.data() + pos * KVH * IDX_BYTES);
        PackQjl(signs, qjlPacked.data() + pos * KVH * QJL_BYTES);
        norm[pos] = 1.0F + 0.1F * static_cast<float>(pos);
        gamma[pos] = 0.5F + 0.1F * static_cast<float>(pos);
    }

    std::vector<int32_t> blockTable{0};
    std::vector<int32_t> seqLens{static_cast<int32_t>(SEQ_LEN)};
    const auto expected =
        Golden(query, queryQjl, idxPacked, qjlPacked, norm, gamma, blockTable, seqLens, T, H, KVH, MAX_BLOCKS, MAX_KV);
    std::vector<float> output(expected.size(), 0.0F);
    const auto queryBf16 = ToBf16(query);
    const auto queryQjlBf16 = ToBf16(queryQjl);
    const auto normBf16 = ToBf16(norm);
    const auto gammaBf16 = ToBf16(gamma);

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    void *qDev = nullptr;
    void *qjlDev = nullptr;
    void *idxDev = nullptr;
    void *qjlPDev = nullptr;
    void *normDev = nullptr;
    void *gammaDev = nullptr;
    void *tableDev = nullptr;
    void *seqDev = nullptr;
    void *outDev = nullptr;
    aclTensor *qTensor = nullptr;
    aclTensor *qjlTensor = nullptr;
    aclTensor *idxTensor = nullptr;
    aclTensor *qjlPTensor = nullptr;
    aclTensor *normTensor = nullptr;
    aclTensor *gammaTensor = nullptr;
    aclTensor *tableTensor = nullptr;
    aclTensor *seqTensor = nullptr;
    aclTensor *outTensor = nullptr;

    const std::vector<int64_t> qShape{T, H, HEAD_DIM};
    const std::vector<int64_t> idxShape{NUM_BLOCKS, BLOCK_SIZE, KVH, IDX_BYTES};
    const std::vector<int64_t> qjlShape{NUM_BLOCKS, BLOCK_SIZE, KVH, QJL_BYTES};
    const std::vector<int64_t> scalarShape{NUM_BLOCKS, BLOCK_SIZE, KVH};
    const std::vector<int64_t> tableShape{1, MAX_BLOCKS};
    const std::vector<int64_t> seqShape{1};
    const std::vector<int64_t> outShape{T, H, MAX_KV};

    CHECK_ACL(CreateTensor(queryBf16, qShape, ACL_BF16, &qDev, &qTensor));
    CHECK_ACL(CreateTensor(queryQjlBf16, qShape, ACL_BF16, &qjlDev, &qjlTensor));
    CHECK_ACL(CreateTensor(idxPacked, idxShape, ACL_UINT8, &idxDev, &idxTensor));
    CHECK_ACL(CreateTensor(qjlPacked, qjlShape, ACL_UINT8, &qjlPDev, &qjlPTensor));
    CHECK_ACL(CreateTensor(normBf16, scalarShape, ACL_BF16, &normDev, &normTensor));
    CHECK_ACL(CreateTensor(gammaBf16, scalarShape, ACL_BF16, &gammaDev, &gammaTensor));
    CHECK_ACL(CreateTensor(blockTable, tableShape, ACL_INT32, &tableDev, &tableTensor));
    CHECK_ACL(CreateTensor(seqLens, seqShape, ACL_INT32, &seqDev, &seqTensor));
    CHECK_ACL(CreateTensor(output, outShape, ACL_FLOAT, &outDev, &outTensor));

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnTurboQuantAttentionScoreGetWorkspaceSize(qTensor, qjlTensor, idxTensor, qjlPTensor, normTensor,
                                                            gammaTensor, tableTensor, seqTensor, MSE_BITS, outTensor,
                                                            &workspaceSize, &executor));
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnTurboQuantAttentionScore(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(output.data(), output.size() * sizeof(float), outDev, output.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST));

    float maxError = 0.0F;
    for (size_t i = 0; i < output.size(); ++i) {
        const bool expMask = std::isinf(expected[i]) || expected[i] < -1.0e30F;
        const bool outMask = std::isinf(output[i]) || output[i] < -1.0e30F;
        if (expMask && outMask) {
            continue;
        }
        maxError = std::max(maxError, std::fabs(output[i] - expected[i]));
    }
    std::printf("max_abs_error=%g, %s\n", maxError, maxError < 1.0e-2F ? "PASSED" : "FAILED");

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyTensor(qTensor);
    aclDestroyTensor(qjlTensor);
    aclDestroyTensor(idxTensor);
    aclDestroyTensor(qjlPTensor);
    aclDestroyTensor(normTensor);
    aclDestroyTensor(gammaTensor);
    aclDestroyTensor(tableTensor);
    aclDestroyTensor(seqTensor);
    aclDestroyTensor(outTensor);
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
    aclrtResetDevice(0);
    aclFinalize();
    return maxError < 1.0e-2F ? 0 : 1;
}
