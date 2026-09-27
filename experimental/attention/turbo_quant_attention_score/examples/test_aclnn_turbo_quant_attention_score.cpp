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
 * test_aclnn_turbo_quant_attention_score.cpp — 设备端自测 harness
 *
 * 精度：host golden 与 kernel 采用同一算法（fp32、pairwise 归约树、小端 bit 解包、
 *       Compare/Select 质心查表、-inf masking），该算法已与 golden.py 语义做过
 *       CPU 双路径仿真对拍（sim_kernel_vs_golden.py，8 case 全 PASS）。
 *       判定：有效位置 |a-b| <= 1e-3 + 1e-3*|b|；-inf 位置位模式精确一致。
 * 性能：decode 三档 + prefill_kv4k 预热 3 次 + 计时 10 次（aclrt 事件）。
 *
 * 编译（实例上，装好 custom run 包后）：
 *   g++ -O2 -std=c++17 test_aclnn_turbo_quant_attention_score.cpp -o tqas_test \
 *       -I/usr/local/Ascend/ascend-toolkit/latest/include \
 *       -I$CUSTOM_INC -L$CUSTOM_LIB -lascendcl -lopapi -lpthread
 */
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_turbo_quant_attention_score.h"

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FATAL] " << msg << std::endl; \
            return false; \
        } \
    } while (0)

namespace {

constexpr int32_t D = 128;
constexpr int32_t BLK = 128;
constexpr int32_t QJL_BYTES = 16;             // qjl_dim=128, 1 bit/channel
constexpr float COEF = 0.009791516697777345f; // sqrt(pi/2)/128

const float CENT2[4] = {-0.1335033178f, -0.04002048075f, 0.04002048075f, 0.1335033178f};
const float CENT3[8] = {-0.19020693f,   -0.1187859178f, -0.06682205945f, -0.02166347019f,
                        0.02166347019f, 0.06682205945f, 0.1187859178f,   0.19020693f};
const float CENT4[16] = {-0.2414890379f,  -0.1828317791f,  -0.1429702938f,  -0.1109927073f,
                         -0.08325428516f, -0.05802082643f, -0.03428063914f, -0.01134236995f,
                         0.01134236995f,  0.03428063914f,  0.05802082643f,  0.08325428516f,
                         0.1109927073f,   0.1429702938f,   0.1828317791f,   0.2414890379f};
const float *Centroids(int bits)
{
    return bits == 2 ? CENT2 : (bits == 3 ? CENT3 : CENT4);
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed)
        : s(seed)
    {}
    uint32_t Next()
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(s >> 33);
    }
    float Uniform(float lo, float hi)
    {
        return lo + (hi - lo) * (Next() >> 8) / 16777216.0f;
    }
    uint8_t Byte()
    {
        return static_cast<uint8_t>(Next() & 0xFF);
    }
};

float RneBf16(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, 4);
    uint32_t rounded = (u + 0x7FFFU + ((u >> 16) & 1)) >> 16;
    uint32_t bits = rounded << 16;
    std::memcpy(&v, &bits, 4);
    return v;
}

uint16_t F32ToBf16Bits(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return static_cast<uint16_t>((u + 0x7FFFU + ((u >> 16) & 1)) >> 16);
}

float Bf16BitsToF32(uint16_t b)
{
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// fp32 pairwise 归约树点积（与 kernel TreeReduce8 + SumPartials 一致）
float TreeDot(const float *a, const float *b)
{
    float p[D];
    for (int i = 0; i < D; i++) {
        p[i] = a[i] * b[i];
    }
    for (int off = D / 2; off >= 8; off >>= 1) {
        for (int i = 0; i < off; i++) {
            p[i] = p[i] + p[i + off];
        }
    }
    float s = p[0];
    for (int i = 1; i < 8; i++) {
        s += p[i];
    }
    return s;
}

// 与 kernel UnpackIdx 一致：每 bits 字节一个小端字，拆 8 个 lane
void UnpackIdx(const uint8_t *row, int bits, float *yhat)
{
    const float *cent = Centroids(bits);
    for (int g = 0; g < D / 8; g++) {
        uint32_t word = 0;
        for (int by = 0; by < bits; by++) {
            word |= static_cast<uint32_t>(row[g * bits + by]) << (8 * by);
        }
        for (int l = 0; l < 8; l++) {
            yhat[g * 8 + l] = cent[(word >> (l * bits)) & ((1 << bits) - 1)];
        }
    }
}

void UnpackSigns(const uint8_t *row, float *sign)
{
    for (int by = 0; by < 16; by++) {
        uint32_t v = row[by];
        for (int l = 0; l < 8; l++) {
            sign[by * 8 + l] = ((v >> l) & 1U) ? 1.0f : -1.0f;
        }
    }
}

// NPU HBM 空闲/总字节（查询失败返回 0）
void HbmInfo(size_t &freeBytes, size_t &totalBytes)
{
    freeBytes = 0;
    totalBytes = 0;
    if (aclrtGetMemInfo(ACL_HBM_MEM, &freeBytes, &totalBytes) != ACL_SUCCESS) {
        freeBytes = 0;
        totalBytes = 0;
    }
}

// 负向用例：key_cache_norm/gamma 的 num_blocks 少于 key_cache_idx 时必须被拒绝
// （kernel 按物理块号直接寻址 norm/gamma 平面，shape 不一致会越界读 GM）
bool RunShortNormCase(aclrtStream stream)
{
    const int64_t T = 1, B = 1, QH = 32, KH = 8, bits = 3;
    const int64_t maxBlocks = 4;
    const int64_t idxBytes = D * bits / 8;
    const int64_t maxKv = maxBlocks * BLK;

    std::vector<uint16_t> qBf16(T * QH * D, 0), qqBf16(T * QH * D, 0);
    std::vector<uint8_t> idx(static_cast<size_t>(maxBlocks) * BLK * KH * idxBytes, 0);
    std::vector<uint8_t> qjl(static_cast<size_t>(maxBlocks) * BLK * KH * QJL_BYTES, 0);
    // 故意少一个 block
    std::vector<uint16_t> ngBf16(static_cast<size_t>(maxBlocks - 1) * BLK * KH, 0);
    std::vector<int32_t> bt(B * maxBlocks), seqLens(B, maxKv);
    for (int64_t i = 0; i < B * maxBlocks; i++) {
        bt[i] = static_cast<int32_t>(i);
    }

    void *qDev = nullptr, *qqDev = nullptr, *idxDev = nullptr, *qjlDev = nullptr, *ngDev = nullptr, *btDev = nullptr,
         *slDev = nullptr, *outDev = nullptr;
    const size_t qBytes = qBf16.size() * 2, ngBytes = ngBf16.size() * 2, outBytes = T * QH * maxKv * 4;
    bool ok = aclrtMalloc(&qDev, qBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&qqDev, qBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&idxDev, idx.size(), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&qjlDev, qjl.size(), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&ngDev, ngBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&btDev, bt.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&slDev, seqLens.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    if (!ok) {
        std::cout << "[FAIL] neg_short_norm malloc failed" << std::endl;
        return false;
    }

    std::vector<int64_t> qShape = {T, QH, D};
    std::vector<int64_t> idxShape = {maxBlocks, BLK, KH, idxBytes};
    std::vector<int64_t> qjlShape = {maxBlocks, BLK, KH, QJL_BYTES};
    std::vector<int64_t> ngShape = {maxBlocks - 1, BLK, KH};
    std::vector<int64_t> btShape = {B, maxBlocks}, slShape = {B}, outShape = {T, QH, maxKv};
    aclTensor *qT = aclCreateTensor(qShape.data(), 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, qShape.data(), 3, qDev);
    aclTensor *qqT = aclCreateTensor(qShape.data(), 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, qShape.data(), 3, qqDev);
    aclTensor *idxT =
        aclCreateTensor(idxShape.data(), 4, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, idxShape.data(), 4, idxDev);
    aclTensor *qjlT =
        aclCreateTensor(qjlShape.data(), 4, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, qjlShape.data(), 4, qjlDev);
    aclTensor *normT =
        aclCreateTensor(ngShape.data(), 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, ngShape.data(), 3, ngDev);
    aclTensor *gamT = aclCreateTensor(ngShape.data(), 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, ngShape.data(), 3, ngDev);
    aclTensor *btT = aclCreateTensor(btShape.data(), 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, btShape.data(), 2, btDev);
    aclTensor *slT = aclCreateTensor(slShape.data(), 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, slShape.data(), 1, slDev);
    aclTensor *outT =
        aclCreateTensor(outShape.data(), 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, outShape.data(), 3, outDev);

    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus st = aclnnTurboQuantAttentionScoreGetWorkspaceSize(qT, qqT, idxT, qjlT, normT, gamT, btT, slT, bits,
                                                                   outT, &wsSize, &executor);
    // tiling 应在下发前拦下该 shape
    bool rejected = (st != ACL_SUCCESS);
    std::cout << "[" << (rejected ? "PASS" : "FAIL")
              << "] neg_short_norm (norm/gamma 少 1 个 block) rejected=" << rejected << " aclnnStatus=" << st
              << std::endl;

    aclDestroyTensor(qT);
    aclDestroyTensor(qqT);
    aclDestroyTensor(idxT);
    aclDestroyTensor(qjlT);
    aclDestroyTensor(normT);
    aclDestroyTensor(gamT);
    aclDestroyTensor(btT);
    aclDestroyTensor(slT);
    aclDestroyTensor(outT);
    aclrtFree(qDev);
    aclrtFree(qqDev);
    aclrtFree(idxDev);
    aclrtFree(qjlDev);
    aclrtFree(ngDev);
    aclrtFree(btDev);
    aclrtFree(slDev);
    aclrtFree(outDev);
    (void)stream;
    return rejected;
}

struct Case {
    std::string name;
    int64_t numQTokens;
    int64_t batch;
    int64_t kvLen; // 各请求等长（edge_seq_tail 用 seqLens 覆盖）
    int64_t qHeads;
    int64_t kvHeads;
    int bits;
    bool perf;
    int64_t cmpTokens;            // host golden 只比对前 cmpTokens 个 token（0 = 全部）
    std::vector<int32_t> seqLens; // 空 = 全部 kvLen
};

bool RunCase(const Case &c, aclrtStream stream)
{
    const int64_t T = c.numQTokens, B = c.batch, QH = c.qHeads, KH = c.kvHeads;
    const int64_t G = QH / KH;
    const int64_t idxBytes = D * c.bits / 8;
    const int64_t maxBlocks = (c.kvLen + BLK - 1) / BLK;
    const int64_t numBlocks = maxBlocks; // 官方 case：物理块数恰好等于需求
    const int64_t maxKv = maxBlocks * BLK;
    std::vector<int32_t> seqLens = c.seqLens.empty() ? std::vector<int32_t>(B, c.kvLen) : c.seqLens;

    // ---- 输入生成（确定性） ----
    Rng rng(20260923ULL + static_cast<uint64_t>(c.bits) * 7919ULL + static_cast<uint64_t>(T) * 131ULL +
            static_cast<uint64_t>(c.kvLen));
    std::vector<uint16_t> qBf16(T * QH * D), qqBf16(T * QH * D);
    std::vector<float> qF(T * QH * D), qqF(T * QH * D);
    for (int64_t i = 0; i < T * QH * D; i++) {
        qF[i] = RneBf16(rng.Uniform(-1.0f, 1.0f));
        qqF[i] = RneBf16(rng.Uniform(-1.0f, 1.0f));
        qBf16[i] = F32ToBf16Bits(qF[i]);
        qqBf16[i] = F32ToBf16Bits(qqF[i]);
    }
    std::vector<uint8_t> idx(numBlocks * BLK * KH * idxBytes);
    std::vector<uint8_t> qjl(numBlocks * BLK * KH * 16);
    for (size_t i = 0; i < idx.size(); i++)
        idx[i] = rng.Byte();
    for (size_t i = 0; i < qjl.size(); i++)
        qjl[i] = rng.Byte();
    std::vector<uint16_t> normBf16(numBlocks * BLK * KH), gamBf16(numBlocks * BLK * KH);
    for (int64_t i = 0; i < numBlocks * BLK * KH; i++) {
        normBf16[i] = F32ToBf16Bits(rng.Uniform(0.1f, 2.0f));
        gamBf16[i] = F32ToBf16Bits(rng.Uniform(0.1f, 2.0f));
    }
    // block_table：随机置换（全部指向合法物理块，覆盖间接寻址）
    std::vector<int32_t> bt(B * maxBlocks);
    std::vector<int32_t> perm(numBlocks);
    for (int64_t i = 0; i < numBlocks; i++)
        perm[i] = static_cast<int32_t>(i);
    for (int64_t i = numBlocks - 1; i > 0; i--) {
        int64_t j = rng.Next() % (i + 1);
        std::swap(perm[i], perm[j]);
    }
    for (int64_t b = 0; b < B; b++) {
        for (int64_t i = 0; i < maxBlocks; i++) {
            bt[b * maxBlocks + i] = perm[i];
        }
    }

    // ---- host golden（kernel 同算法；仅算需要比对的 token 子集） ----
    const int64_t cmpT = (c.cmpTokens > 0 && c.cmpTokens < T) ? c.cmpTokens : T;
    const int64_t cmpKvMax = c.kvLen; // 有效区间长度（-inf 尾部单独校验）
    std::vector<float> golden(static_cast<size_t>(cmpT) * QH * cmpKvMax, 0.0f);
    {
        float yhat[D], sign[D], qRow[D], qqRow[D];
        for (int64_t t = 0; t < cmpT; t++) {
            int64_t b = (B == 1) ? 0 : t;
            for (int64_t h = 0; h < QH; h++) {
                int64_t k = h / G;
                for (int i = 0; i < D; i++) {
                    qRow[i] = qF[t * QH * D + h * D + i];
                    qqRow[i] = qqF[t * QH * D + h * D + i];
                }
                for (int64_t i = 0; i < cmpKvMax; i++) {
                    int64_t phys = bt[b * maxBlocks + i / BLK];
                    int64_t off = i % BLK;
                    int64_t rowBase = (phys * BLK + off) * KH + k;
                    UnpackIdx(&idx[rowBase * idxBytes], c.bits, yhat);
                    UnpackSigns(&qjl[rowBase * 16], sign);
                    float main = TreeDot(qRow, yhat);
                    float corr = TreeDot(qqRow, sign);
                    float normV = Bf16BitsToF32(normBf16[rowBase]);
                    float gamV = Bf16BitsToF32(gamBf16[rowBase]);
                    golden[(t * QH + h) * cmpKvMax + i] =
                        (i < seqLens[b]) ? normV * main + COEF * gamV * corr : -INFINITY;
                }
            }
        }
    }

    // ---- 设备内存 ----
    void *qDev = nullptr, *qqDev = nullptr, *idxDev = nullptr, *qjlDev = nullptr, *normDev = nullptr, *gamDev = nullptr,
         *btDev = nullptr, *slDev = nullptr, *outDev = nullptr;
    size_t hbmFree0 = 0, hbmTotal = 0;
    HbmInfo(hbmFree0, hbmTotal);
    const size_t qBytes = qBf16.size() * 2;
    const size_t outBytes = static_cast<size_t>(T) * QH * maxKv * 4;
    CHECK(aclrtMalloc(&qDev, qBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, c.name + ": malloc q");
    CHECK(aclrtMalloc(&qqDev, qBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, c.name + ": malloc qq");
    CHECK(aclrtMalloc(&idxDev, idx.size(), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, c.name + ": malloc idx");
    CHECK(aclrtMalloc(&qjlDev, qjl.size(), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, c.name + ": malloc qjl");
    CHECK(aclrtMalloc(&normDev, normBf16.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc norm");
    CHECK(aclrtMalloc(&gamDev, gamBf16.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc gam");
    CHECK(aclrtMalloc(&btDev, bt.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc bt");
    CHECK(aclrtMalloc(&slDev, seqLens.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc sl");
    CHECK(aclrtMalloc(&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, c.name + ": malloc out");
    size_t hbmFreeAlloc = 0, hbmTotal2 = 0;
    HbmInfo(hbmFreeAlloc, hbmTotal2);
    CHECK(aclrtMemcpy(qDev, qBytes, qBf16.data(), qBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, "H2D q");
    CHECK(aclrtMemcpy(qqDev, qBytes, qqBf16.data(), qBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, "H2D qq");
    CHECK(aclrtMemcpy(idxDev, idx.size(), idx.data(), idx.size(), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, "H2D idx");
    CHECK(aclrtMemcpy(qjlDev, qjl.size(), qjl.data(), qjl.size(), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, "H2D qjl");
    CHECK(aclrtMemcpy(normDev, normBf16.size() * 2, normBf16.data(), normBf16.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE) ==
              ACL_SUCCESS,
          "H2D norm");
    CHECK(aclrtMemcpy(gamDev, gamBf16.size() * 2, gamBf16.data(), gamBf16.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE) ==
              ACL_SUCCESS,
          "H2D gam");
    CHECK(aclrtMemcpy(btDev, bt.size() * 4, bt.data(), bt.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
          "H2D bt");
    CHECK(aclrtMemcpy(slDev, seqLens.size() * 4, seqLens.data(), seqLens.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE) ==
              ACL_SUCCESS,
          "H2D sl");

    std::vector<int64_t> qShape = {T, QH, D};
    std::vector<int64_t> idxShape = {numBlocks, BLK, KH, idxBytes};
    std::vector<int64_t> qjlShape = {numBlocks, BLK, KH, 16};
    std::vector<int64_t> ngShape = {numBlocks, BLK, KH};
    std::vector<int64_t> btShape = {B, maxBlocks};
    std::vector<int64_t> slShape = {B};
    std::vector<int64_t> outShape = {T, QH, maxKv};
    aclTensor *qT = aclCreateTensor(qShape.data(), 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, qShape.data(), 3, qDev);
    aclTensor *qqT = aclCreateTensor(qShape.data(), 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, qShape.data(), 3, qqDev);
    aclTensor *idxT =
        aclCreateTensor(idxShape.data(), 4, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, idxShape.data(), 4, idxDev);
    aclTensor *qjlT =
        aclCreateTensor(qjlShape.data(), 4, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, qjlShape.data(), 4, qjlDev);
    aclTensor *normT =
        aclCreateTensor(ngShape.data(), 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, ngShape.data(), 3, normDev);
    aclTensor *gamT =
        aclCreateTensor(ngShape.data(), 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, ngShape.data(), 3, gamDev);
    aclTensor *btT = aclCreateTensor(btShape.data(), 2, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, btShape.data(), 2, btDev);
    aclTensor *slT = aclCreateTensor(slShape.data(), 1, ACL_INT32, nullptr, 0, ACL_FORMAT_ND, slShape.data(), 1, slDev);
    aclTensor *outT =
        aclCreateTensor(outShape.data(), 3, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, outShape.data(), 3, outDev);
    CHECK(qT && qqT && idxT && qjlT && normT && gamT && btT && slT && outT, c.name + ": create tensor");

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus st = aclnnTurboQuantAttentionScoreGetWorkspaceSize(qT, qqT, idxT, qjlT, normT, gamT, btT, slT, c.bits,
                                                                   outT, &workspaceSize, &executor);
    CHECK(st == ACL_SUCCESS, c.name + ": GetWorkspaceSize ret=" + std::to_string(st));
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc ws");
    }

    st = aclnnTurboQuantAttentionScore(workspace, workspaceSize, executor, stream);
    CHECK(st == ACL_SUCCESS, c.name + ": aclnnTurboQuantAttentionScore ret=" + std::to_string(st));
    {
        aclError syncErr = aclrtSynchronizeStream(stream);
        const char *errMsg = aclGetRecentErrMsg();
        CHECK(syncErr == ACL_SUCCESS,
              c.name + ": sync ret=" + std::to_string(syncErr) + " msg=" + (errMsg ? errMsg : "(null)"));
    }

    // ---- 内存 ----
    {
        const size_t kvEncodedBytes = idx.size() + qjl.size() + normBf16.size() * 2 + gamBf16.size() * 2;
        const size_t kvBaselineBytes = static_cast<size_t>(numBlocks) * BLK * KH * D * 2; // bf16 K cache
        size_t hbmFreeExec = 0, hbmTotal3 = 0;
        HbmInfo(hbmFreeExec, hbmTotal3);
        const size_t ioBytes = qBytes * 2 + idx.size() + qjl.size() + normBf16.size() * 2 + gamBf16.size() * 2 +
                               bt.size() * 4 + seqLens.size() * 4 + outBytes;
        std::cout << "  [MEM] " << c.name << " kv_cache_encoded=" << kvEncodedBytes
                  << "B kv_cache_bf16_baseline=" << kvBaselineBytes
                  << "B compression=" << (kvBaselineBytes / static_cast<double>(kvEncodedBytes))
                  << "x workspace=" << workspaceSize << "B io_tensors=" << ioBytes
                  << "B hbm_free(before/alloc/exec)=" << hbmFree0 << "/" << hbmFreeAlloc << "/" << hbmFreeExec
                  << "B hbm_total=" << hbmTotal << "B" << std::endl;
    }

    // ---- 性能 ----
    if (c.perf) {
        aclrtEvent ev0, ev1;
        aclrtCreateEvent(&ev0);
        aclrtCreateEvent(&ev1);
        for (int i = 0; i < 3; i++) {
            CHECK(aclnnTurboQuantAttentionScore(workspace, workspaceSize, executor, stream) == ACL_SUCCESS,
                  "perf warmup");
        }
        CHECK(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "perf warmup sync");
        float totalUs = 0.0f;
        const int iters = 10;
        for (int i = 0; i < iters; i++) {
            CHECK(aclrtRecordEvent(ev0, stream) == ACL_SUCCESS, "rec0");
            CHECK(aclnnTurboQuantAttentionScore(workspace, workspaceSize, executor, stream) == ACL_SUCCESS,
                  "perf exec");
            CHECK(aclrtRecordEvent(ev1, stream) == ACL_SUCCESS, "rec1");
            CHECK(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "perf sync");
            float ms = 0.0f;
            CHECK(aclrtEventElapsedTime(&ms, ev0, ev1) == ACL_SUCCESS, "elapsed");
            totalUs += ms * 1000.0f;
        }
        std::cout << "  [PERF] " << c.name << " avg " << totalUs / iters << " us over " << iters << " iters"
                  << std::endl;
        aclrtDestroyEvent(ev0);
        aclrtDestroyEvent(ev1);
    }

    // ---- 回读比对（比对区间 = 前 cmpT 个 token × 前 kvLen 列；含 -inf 位模式） ----
    const size_t rowBytes = static_cast<size_t>(maxKv) * 4;
    const size_t d2hRows = static_cast<size_t>(cmpT) * QH;
    std::vector<uint8_t> hOut(d2hRows * rowBytes);
    for (size_t r = 0; r < d2hRows; r++) {
        CHECK(aclrtMemcpy(hOut.data() + r * rowBytes, rowBytes, static_cast<uint8_t *>(outDev) + r * rowBytes, rowBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              "D2H out");
    }

    size_t bad = 0, infBad = 0, total = 0;
    double maxAbs = 0.0;
    size_t firstBadIdx = SIZE_MAX;
    for (int64_t t = 0; t < cmpT; t++) {
        for (int64_t h = 0; h < QH; h++) {
            const float *row = reinterpret_cast<const float *>(hOut.data() + (t * QH + h) * rowBytes);
            for (int64_t i = 0; i < cmpKvMax; i++) {
                float gv = golden[(t * QH + h) * cmpKvMax + i];
                float av = row[i];
                total++;
                if (std::isinf(gv)) {
                    uint32_t au, gu;
                    std::memcpy(&au, &av, 4);
                    std::memcpy(&gu, &gv, 4);
                    if (au != gu) {
                        infBad++;
                        if (firstBadIdx == SIZE_MAX)
                            firstBadIdx = (t * QH + h) * cmpKvMax + i;
                    }
                } else {
                    double d = std::fabs(static_cast<double>(av) - gv);
                    maxAbs = d > maxAbs ? d : maxAbs;
                    if (d > 1e-3 + 1e-3 * std::fabs(static_cast<double>(gv))) {
                        bad++;
                        if (firstBadIdx == SIZE_MAX)
                            firstBadIdx = (t * QH + h) * cmpKvMax + i;
                    }
                }
            }
        }
    }
    bool ok = (bad == 0) && (infBad == 0);
    std::cout << "[" << (ok ? "PASS" : "FAIL") << "] " << c.name << " T=" << T << " B=" << B << " kv=" << c.kvLen
              << " bits=" << c.bits << " | bad=" << bad << " inf_bad=" << infBad << " max_abs=" << maxAbs
              << " n=" << total << std::endl;
    if (!ok && firstBadIdx != SIZE_MAX) {
        int64_t t = firstBadIdx / (QH * cmpKvMax);
        int64_t h = (firstBadIdx / cmpKvMax) % QH;
        int64_t i = firstBadIdx % cmpKvMax;
        std::cout << "  first bad t=" << t << " h=" << h << " i=" << i
                  << " got=" << reinterpret_cast<const float *>(hOut.data() + (t * QH + h) * rowBytes)[i]
                  << " expect=" << golden[firstBadIdx] << std::endl;
    }
    if (!ok && T <= 2) { // debug：按 (h, lb) 统计 -inf 分布，定位坏 item 聚类
        std::cout << "  -inf@valid map (h x lb, '.'=ok, '#'=all-bad, digit=partial/16):" << std::endl;
        for (int64_t h = 0; h < QH && h < 32; h++) {
            std::cout << "  h" << h << (h < 10 ? " " : "") << " ";
            for (int64_t lb = 0; lb < maxBlocks && lb < 64; lb++) {
                int badCnt = 0;
                const float *row = reinterpret_cast<const float *>(hOut.data() + (0 * QH + h) * rowBytes);
                for (int64_t off = 0; off < BLK; off++) {
                    int64_t i = lb * BLK + off;
                    if (i >= cmpKvMax)
                        break;
                    float gv = golden[(0 * QH + h) * cmpKvMax + i];
                    float av = row[i];
                    bool gInf = std::isinf(gv);
                    if ((gInf != std::isinf(av)) || (!gInf && std::fabs(av - gv) > 1e-3 + 1e-3 * std::fabs(gv))) {
                        badCnt++;
                    }
                }
                std::cout << (badCnt == 0 ? "." : (badCnt >= BLK ? "#" : std::to_string(badCnt / 16)));
            }
            std::cout << std::endl;
        }
    }

    aclDestroyTensor(qT);
    aclDestroyTensor(qqT);
    aclDestroyTensor(idxT);
    aclDestroyTensor(qjlT);
    aclDestroyTensor(normT);
    aclDestroyTensor(gamT);
    aclDestroyTensor(btT);
    aclDestroyTensor(slT);
    aclDestroyTensor(outT);
    aclrtFree(qDev);
    aclrtFree(qqDev);
    aclrtFree(idxDev);
    aclrtFree(qjlDev);
    aclrtFree(normDev);
    aclrtFree(gamDev);
    aclrtFree(btDev);
    aclrtFree(slDev);
    aclrtFree(outDev);
    if (workspace) {
        aclrtFree(workspace);
    }
    return ok;
}

} // namespace

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl;
        return -1;
    }
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    std::vector<Case> cases = {
        // 官方 5 case（shape 与 case.json 一致）
        {"gqa_decode_kv4k", 1, 1, 4096, 32, 8, 3, true, 0, {}},
        {"gqa_decode_kv32k", 1, 1, 32768, 32, 8, 3, true, 0, {}},
        {"gqa_decode_kv128k", 1, 1, 131072, 32, 8, 3, true, 0, {}},
        {"gqa_prefill_kv4k", 512, 1, 4096, 32, 8, 3, true, 0, {}},
        {"gqa_prefill_kv32k", 512, 1, 32768, 32, 8, 3, true, 64, {}}, // host golden 采样 64 token
        // 边界 case
        {"edge_mse2_kv4k", 1, 1, 4096, 32, 8, 2, false, 0, {}},
        {"edge_mse4_kv4k", 1, 1, 4096, 32, 8, 4, false, 0, {}},
        {"edge_seq_tail", 2, 2, 5120, 32, 8, 3, false, 0, {5000, 123}}, // 尾块 + 变长 batch
    };

    int passed = 0, failed = 0;
    for (auto &c : cases) {
        if (RunCase(c, stream)) {
            ++passed;
        } else {
            ++failed;
        }
    }
    // 负向用例：KV cache 四个输入 num_blocks 不一致必须被拦下
    if (RunShortNormCase(stream)) {
        ++passed;
    } else {
        ++failed;
    }
    std::cout << "\n==== summary: " << passed << " passed, " << failed << " failed ====" << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return failed == 0 ? 0 : 1;
}
