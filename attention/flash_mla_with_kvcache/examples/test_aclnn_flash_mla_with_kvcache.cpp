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
 * \file test_aclnn_flash_mla_with_kvcache.cpp
 * \brief aclnn FlashMlaWithKvcache end-to-end example with independent CPU golden verification.
 *
 * Pipeline per case:
 *   1. aclnnFlashMlaWithKvcacheMetadataGetWorkspaceSize / aclnnFlashMlaWithKvcacheMetadata  -> multi-section metadata (INT32)
 *   2. aclnnFlashMlaWithKvcacheGetWorkspaceSize / aclnnFlashMlaWithKvcache                  -> attn_out (NTD) + softmax_lse
 *   3. independent naive-attention golden (fp64) compared with the
 *      flash_attn pytests thresholds (attention/flash_attn/tests/pytests/utils/compare.py,
 *      check_result: fp16 atol=2.5e-5 rtol=5e-3; bf16 atol=1e-4 rtol=7.8125e-3; failRatio<=0.5%)
 *
 * Geometry (routing set): q TND [T,n1,576] (nope512+rope64 merged), k_cache PA_NZ
 * (Bn,N_kv,D1,Bs,D0) = (numBlocks,1,36,blockSize,16), block_table [b,maxBlocks], cache_seqlens [b],
 * cu_seqlens_q [b+1], attn_mask INT8 [2048,2048] causal (mask_mode=3 = RIGHT_DOWN), attrs
 * head_dim_v=512, softmax_scale=1/sqrt(512), layout_q=TND layout_kv=NZ layout_out=NTD,
 * return_softmax_lse=1. N_kv=1 (single latent KV head).
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnn/opdev/bfloat16.h"
#include "../op_api/aclnn_flash_mla_with_kvcache.h"
#include "../../flash_mla_with_kvcache_metadata/op_host/op_api/aclnn_flash_mla_with_kvcache_metadata.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr int64_t QK_HEAD_DIM = 512;                   // nope width (head_dim_v)
constexpr int64_t ROPE_HEAD_DIM = 64;                  // rope lanes merged at the tail of the 576-wide q/k_cache
constexpr int64_t MLA_D = QK_HEAD_DIM + ROPE_HEAD_DIM; // 576
constexpr int64_t BLOCK_SIZE = 128;                    // paged-attention block size (tokens) for the NZ k_cache
constexpr int64_t NZ_D0 = 16;                          // NZ inner depth (32B / typeSize(fp16))
constexpr int64_t NZ_D1 = MLA_D / NZ_D0;               // 36

constexpr int64_t AIC_CORE_NUM = 36;
constexpr int64_t AIV_CORE_NUM = 72;
constexpr int64_t METADATA_ALIGN = 4096; // words (same as flash_attn.py / flash_mla_with_kvcache.py alignment)

// --- compare.py thresholds (get_tolerance / check_result) -------------------------------
constexpr double FP16_ATOL = 2.5e-5;
constexpr double FP16_RTOL = 5.0e-3;
constexpr double BF16_ATOL = 1.0e-4;
constexpr double BF16_RTOL = 7.8125e-3;
constexpr double FAIL_RATIO_LIMIT = 5.0e-3;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

// Deterministic pseudo-random generator (xorshift64) so goldens are reproducible.
struct Rng {
    uint64_t state_;
    explicit Rng(uint64_t seed)
        : state_(seed)
    {}
    double Next() // uniform in (-0.5, 0.5)
    {
        state_ ^= state_ << 13;
        state_ ^= state_ >> 7;
        state_ ^= state_ << 17;
        return (static_cast<double>(state_ & 0xFFFFFFu) / 16777216.0) - 0.5;
    }
};

// raw byte-buffer variant (used for output tensors whose dtype size differs from the host
// staging vector, e.g. fp16/bf16 outputs staged in a 16-bit word vector).
int CreateRawAclTensor(const void *hostBytes, int64_t sizeBytes, const std::vector<int64_t> &shape,
                       aclDataType dataType, void **deviceAddr, aclTensor **tensor)
{
    auto ret = aclrtMalloc(deviceAddr, sizeBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, sizeBytes, hostBytes, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    return CreateRawAclTensor(hostData.data(), GetShapeSize(shape) * sizeof(T), shape, dataType, deviceAddr, tensor);
}

// Mirror of flash_mla_with_kvcache.py/_calculate_metadata_size + flash_mla_with_kvcache.cpp/CalculateMlaMetadataSizeBytes (B2/D16):
// words = ((36+72)*batch*1+1)*16, bytes aligned to 4096 (kv head num == 1 for MLA).
int64_t RequiredMetadataBytes(int64_t batchSize)
{
    int64_t words = (AIC_CORE_NUM + AIV_CORE_NUM) * batchSize * 1 + 1;
    words *= 16;
    int64_t bytes = words * static_cast<int64_t>(sizeof(int32_t));
    return ((bytes + METADATA_ALIGN * 4 - 1) / (METADATA_ALIGN * 4)) * (METADATA_ALIGN * 4);
}

// ------------------------------------------------------------------------------------------
// Case description: one (batched, variable-length) MLA attention problem.
// ------------------------------------------------------------------------------------------
struct Case {
    const char *name;
    aclDataType dtype; // ACL_FLOAT16 or ACL_BF16
    int64_t batch;
    std::vector<int64_t> qLens; // per-batch query lengths (sum == T)
    int64_t kvLen;              // uniform per-batch KV length
    int64_t numHeadsQ;
    bool expectMultiSection; // expect metadata sectionNum >= 2
};

struct PreparedInputs {
    std::vector<op::fp16_t> qFp16;
    std::vector<op::bfloat16> qBf16;
    std::vector<op::fp16_t> kCacheFp16;
    std::vector<op::bfloat16> kCacheBf16;
    std::vector<int32_t> blockTable;
    std::vector<int32_t> cacheSeqlens;
    std::vector<int32_t> cuSeqlensQ;
    std::vector<int8_t> attnMask;
    std::vector<int32_t> metadata;
    int64_t totalBlocks = 0;
    int64_t maxBlocksPerBatch = 0;
    int64_t totalTokens = 0;
    // logical (un-NZ'd) per-batch flat data for the golden: kv[b * kvLen * MLA_D + s * MLA_D + d]
    std::vector<float> kvLogical;
    // logical per-batch query (rope lanes included): q[b * qLen * n1 * MLA_D + (i*n1+n) * MLA_D + d]
    std::vector<float> qLogical;
};

// Build all host-side inputs for a case. q / k values are drawn from the same deterministic
// RNG stream used later by the golden, so the device run and the golden see identical data.
PreparedInputs PrepareCase(const Case &cs, uint64_t seed)
{
    PreparedInputs in;
    in.totalTokens = 0;
    for (auto l : cs.qLens) {
        in.totalTokens += l;
    }
    int64_t blockSize = BLOCK_SIZE;
    in.maxBlocksPerBatch = (cs.kvLen + blockSize - 1) / blockSize;
    in.totalBlocks = cs.batch * in.maxBlocksPerBatch;
    int64_t numBlocks = in.totalBlocks;

    // Query: layout TND [T, n1, 576] (fp16/bf16 host vectors built below).
    Rng rng(seed);
    in.qLogical.assign(in.totalTokens * cs.numHeadsQ * MLA_D, 0.0f);
    {
        int64_t tok = 0;
        for (int64_t b = 0; b < cs.batch; ++b) {
            int64_t qLen = cs.qLens[b];
            for (int64_t i = 0; i < qLen; ++i) {
                for (int64_t n = 0; n < cs.numHeadsQ; ++n) {
                    for (int64_t d = 0; d < MLA_D; ++d) {
                        in.qLogical[(tok * cs.numHeadsQ + n) * MLA_D + d] = static_cast<float>(rng.Next());
                    }
                }
                tok++;
            }
        }
    }
    in.qFp16.resize(in.totalTokens * cs.numHeadsQ * MLA_D);
    in.qBf16.resize(in.totalTokens * cs.numHeadsQ * MLA_D);
    for (int64_t i = 0; i < in.totalTokens * cs.numHeadsQ * MLA_D; ++i) {
        const float v = in.qLogical[i];
        in.qFp16[i] = op::fp16_t(v);
        in.qBf16[i] = op::bfloat16(v);
    }

    // KV: logical per-batch flat [kvLen*576]; then packed into PA_NZ blocks (unused lanes of
    // the last block are zero; blocks for shorter batches are never referenced by block_table).
    in.kvLogical.assign(cs.batch * cs.kvLen * MLA_D, 0.0f);
    std::vector<float> kBlockFlat(numBlocks * 1 * NZ_D1 * BLOCK_SIZE * NZ_D0, 0.0f);
    for (int64_t b = 0; b < cs.batch; ++b) {
        for (int64_t s = 0; s < cs.kvLen; ++s) {
            for (int64_t d = 0; d < MLA_D; ++d) {
                in.kvLogical[(b * cs.kvLen + s) * MLA_D + d] = static_cast<float>(rng.Next());
            }
        }
        // NZ pack: k_cache[block][0][d1][s_in_block][d0] = kv[b][blockBase*128 + s_in][d1*16+d0]
        for (int64_t block = 0; block < in.maxBlocksPerBatch; ++block) {
            for (int64_t s = 0; s < BLOCK_SIZE; ++s) {
                int64_t g = block * BLOCK_SIZE + s;
                if (g >= cs.kvLen) {
                    continue;
                }
                for (int64_t d1 = 0; d1 < NZ_D1; ++d1) {
                    for (int64_t d0 = 0; d0 < NZ_D0; ++d0) {
                        size_t idx = (b * in.maxBlocksPerBatch + block) * (NZ_D1 * BLOCK_SIZE * NZ_D0) +
                                     d1 * (BLOCK_SIZE * NZ_D0) + s * NZ_D0 + d0;
                        kBlockFlat[idx] = in.kvLogical[(b * cs.kvLen + g) * MLA_D + d1 * NZ_D0 + d0];
                    }
                }
            }
        }
    }
    in.kCacheFp16.resize(kBlockFlat.size());
    in.kCacheBf16.resize(kBlockFlat.size());
    for (size_t i = 0; i < kBlockFlat.size(); ++i) {
        in.kCacheFp16[i] = op::fp16_t(kBlockFlat[i]);
        in.kCacheBf16[i] = op::bfloat16(kBlockFlat[i]);
    }

    // block_table [b, maxBlocksPerBatch] rows: [b*max + 0..max)  (uniform 1:1 block layout)
    in.blockTable.assign(cs.batch * in.maxBlocksPerBatch, 0);
    for (int64_t b = 0; b < cs.batch; ++b) {
        for (int64_t blk = 0; blk < in.maxBlocksPerBatch; ++blk) {
            in.blockTable[b * in.maxBlocksPerBatch + blk] = static_cast<int32_t>(b * in.maxBlocksPerBatch + blk);
        }
    }

    in.cacheSeqlens.assign(cs.batch, static_cast<int32_t>(cs.kvLen));
    in.cuSeqlensQ.assign(cs.batch + 1, 0);
    for (int64_t b = 0; b < cs.batch; ++b) {
        in.cuSeqlensQ[b + 1] = in.cuSeqlensQ[b] + static_cast<int32_t>(cs.qLens[b]);
    }

    // attn_mask: INT8 [2048,2048] dense causal (mask_mode=3 = RIGHT_DOWN; gate requires non-null).
    // FIA MASK_BOOL 语义: 1 = 需要掩(blocked), 0 = 允许(allow)。causal 掩上三角 (s > t)。
    in.attnMask.assign(2048 * 2048, 0);
    for (int64_t t = 0; t < 2048; ++t) {
        for (int64_t s = t + 1; s < 2048; ++s) {
            in.attnMask[t * 2048 + s] = 1;
        }
    }

    // metadata buffer: capacity per the (36+72)*batch*1+1 formula, aligned to 4096 words.
    int64_t words = ((AIC_CORE_NUM + AIV_CORE_NUM) * cs.batch * 1 + 1) * 16;
    int64_t alignedWords = ((words + METADATA_ALIGN - 1) / METADATA_ALIGN) * METADATA_ALIGN;
    in.metadata.assign(alignedWords, 0);
    return in;
}

// ------------------------------------------------------------------------------------------
// Independent CPU golden: naive masked attention in double precision.
// O[n][t][d] = sum_j P[t][j] * kv[batchOf(t)][j][d],  d in [0,512) (nope only; rope lanes of
// the output are NOT produced — the kernel outputs exactly head_dim_v=512 columns).
// LSE[n][t] = max_j(S) + log(sum_j exp(S[j]-max)), over allowed (unmasked) j.
// ------------------------------------------------------------------------------------------
struct GoldenResult {
    std::vector<double> o;   // [numHeadsQ * T * 512]
    std::vector<double> lse; // [numHeadsQ * T]
};

GoldenResult ComputeGolden(const Case &cs, const PreparedInputs &in, double scale, bool causal)
{
    // device-exact 参考：golden 用与送入 kernel 完全相同的 16bit 量化输入计算，
    // 避免 fp32 中间值 cast 干扰（fp16/bf16 分别按各自 rounding）。
    std::vector<double> qExact(in.qLogical.size()), kvExact(in.kvLogical.size());
    if (cs.dtype == ACL_FLOAT16) {
        for (size_t i = 0; i < qExact.size(); ++i) {
            qExact[i] = static_cast<double>(static_cast<float>(in.qFp16[i]));
        }
        for (size_t i = 0; i < kvExact.size(); ++i) {
            kvExact[i] = static_cast<double>(static_cast<float>(op::fp16_t(in.kvLogical[i])));
        }
    } else {
        for (size_t i = 0; i < qExact.size(); ++i) {
            qExact[i] = static_cast<double>(static_cast<float>(in.qBf16[i]));
        }
        for (size_t i = 0; i < kvExact.size(); ++i) {
            kvExact[i] = static_cast<double>(static_cast<float>(op::bfloat16(in.kvLogical[i])));
        }
    }

    std::vector<int64_t> tOfBatch(cs.batch + 1, 0);
    for (int64_t b = 0; b < cs.batch; ++b) {
        tOfBatch[b + 1] = tOfBatch[b] + cs.qLens[b];
    }
    GoldenResult g;
    g.o.assign(cs.numHeadsQ * in.totalTokens * QK_HEAD_DIM, 0.0);
    g.lse.assign(cs.numHeadsQ * in.totalTokens, 0.0);
    std::vector<double> scores(cs.kvLen, 0.0);
    for (int64_t b = 0; b < cs.batch; ++b) {
        const double *qBase = &qExact[(tOfBatch[b] * cs.numHeadsQ) * MLA_D];
        // mask_mode=3 (RIGHT_DOWN) 的因果窗：允许 j <= i + (kvLen - qLens[b])（FIA 语义，
        // 对角在 KV 右端；prefill 扩展场景 kvS>>qS 时只挡最后几把 key）。kvLen==qLens 时退化为 j<=i。
        for (int64_t i = 0; i < cs.qLens[b]; ++i) {
            for (int64_t n = 0; n < cs.numHeadsQ; ++n) {
                const double *qrow = qBase + (i * cs.numHeadsQ + n) * MLA_D;
                double maxS = -1e300;
                for (int64_t j = 0; j < cs.kvLen; ++j) {
                    const double *krow = &kvExact[(b * cs.kvLen + j) * MLA_D];
                    double s = 0.0;
                    for (int64_t d = 0; d < MLA_D; ++d) {
                        s += qrow[d] * krow[d];
                    }
                    s *= scale;
                    if (causal && j > i + (cs.kvLen - cs.qLens[b])) {
                        s = -1e300; // masked out (never contributes to max/sum)
                    }
                    scores[j] = s;
                    if (s > maxS) {
                        maxS = s;
                    }
                }
                double sum = 0.0;
                for (int64_t j = 0; j < cs.kvLen; ++j) {
                    if (scores[j] > -1e299) {
                        sum += exp(scores[j] - maxS);
                    }
                }
                double lse = maxS + log(sum);
                int64_t t = tOfBatch[b] + i;
                g.lse[n * in.totalTokens + t] = lse;
                for (int64_t j = 0; j < cs.kvLen; ++j) {
                    double p = (scores[j] > -1e299) ? exp(scores[j] - maxS) / sum : 0.0;
                    const double *vrow = &kvExact[(b * cs.kvLen + j) * MLA_D];
                    double *orow = &g.o[(n * in.totalTokens + t) * QK_HEAD_DIM];
                    for (int64_t d = 0; d < QK_HEAD_DIM; ++d) {
                        orow[d] += p * vrow[d];
                    }
                }
            }
        }
    }
    return g;
}

// Compare with the flash_attn compare.py semantics: torch.isclose(rtol, atol), fail ratio <= 0.5%.
bool CompareWithTolerance(const char *tag, const std::vector<double> &expect, const std::vector<float> &actual,
                          int64_t expectStride, int64_t actualStride, double atol, double rtol)
{
    int64_t n = expect.size();
    int64_t failCnt = 0;
    double maxAbs = 0.0, sumAbs = 0.0, maxRel = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double e = expect[i];
        double a = actual[i];
        double diff = fabs(e - a);
        sumAbs += diff;
        if (diff > maxAbs) {
            maxAbs = diff;
        }
        double rel = diff / (fabs(e) + 1e-6);
        if (rel > maxRel) {
            maxRel = rel;
        }
        // isclose semantics with the python-side comparison
        if (!(diff <= atol + rtol * fabs(e))) {
            failCnt++;
        }
        (void)expectStride;
        (void)actualStride;
    }
    double failRatio = static_cast<double>(failCnt) / static_cast<double>(n);
    bool passed = failRatio <= FAIL_RATIO_LIMIT;
    LOG_PRINT("  [%s] elems=%ld fail=%ld (%.4f%%) maxAbs=%.3e meanAbs=%.3e maxRel=%.3e atol=%.2e rtol=%.2e -> %s\n",
              tag, (long)n, (long)failCnt, failRatio * 100.0, maxAbs, sumAbs / n, maxRel, atol, rtol,
              passed ? "PASS" : "FAIL");
    return passed;
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<Case> cases = {
        {"fp16-causal-b2", ACL_FLOAT16, 2, {4, 4}, 4, 2, false},
        {"bf16-causal-b2", ACL_BF16, 2, {4, 4}, 4, 2, false},
        {"fp16-causal-b8-multisection", ACL_FLOAT16, 8, {8, 8, 8, 8, 8, 8, 8, 8}, 32768, 17, true},
    };

    const double softmaxScale = 1.0 / sqrt(static_cast<double>(QK_HEAD_DIM));

    int totalPass = 0;
    int totalFail = 0;
    for (const auto &cs : cases) {
        LOG_PRINT("\n==================== case: %s ====================\n", cs.name);
        PreparedInputs in = PrepareCase(cs, 0x5EEDull ^ static_cast<uint64_t>(cs.batch * 100003 + cs.kvLen));

        // ---- device buffers / tensors ------------------------------------------------
        void *qDev = nullptr, *kDev = nullptr, *btDev = nullptr, *clDev = nullptr, *cuDev = nullptr;
        void *maskDev = nullptr, *metaDev = nullptr, *outDev = nullptr, *lseDev = nullptr;
        aclTensor *qT = nullptr, *kT = nullptr, *btT = nullptr, *clT = nullptr, *cuT = nullptr;
        aclTensor *maskT = nullptr, *metaT = nullptr, *outT = nullptr, *lseT = nullptr;

        std::vector<int64_t> qShape = {in.totalTokens, cs.numHeadsQ, MLA_D};
        std::vector<int64_t> kShape = {in.totalBlocks, 1, NZ_D1, BLOCK_SIZE, NZ_D0};
        std::vector<int64_t> btShape = {cs.batch, in.maxBlocksPerBatch};
        std::vector<int64_t> clShape = {cs.batch};
        std::vector<int64_t> cuShape = {cs.batch + 1};
        std::vector<int64_t> maskShape = {2048, 2048};
        std::vector<int64_t> metaShape = {static_cast<int64_t>(in.metadata.size())};
        std::vector<int64_t> outShape = {cs.numHeadsQ, in.totalTokens, QK_HEAD_DIM}; // NTD = (N,T,D)
        std::vector<int64_t> lseShape = {cs.numHeadsQ, in.totalTokens};

        aclDataType qDtype = cs.dtype; // fp16/bf16 with matching host vector
        if (cs.dtype == ACL_FLOAT16) {
            ret = CreateAclTensor(in.qFp16, qShape, &qDev, qDtype, &qT);
            ret = ret == ACL_SUCCESS ? CreateAclTensor(in.kCacheFp16, kShape, &kDev, qDtype, &kT) : ret;
        } else {
            ret = CreateAclTensor(in.qBf16, qShape, &qDev, qDtype, &qT);
            ret = ret == ACL_SUCCESS ? CreateAclTensor(in.kCacheBf16, kShape, &kDev, qDtype, &kT) : ret;
        }
        ret = ret == ACL_SUCCESS ? CreateAclTensor(in.blockTable, btShape, &btDev, ACL_INT32, &btT) : ret;
        ret = ret == ACL_SUCCESS ? CreateAclTensor(in.cacheSeqlens, clShape, &clDev, ACL_INT32, &clT) : ret;
        ret = ret == ACL_SUCCESS ? CreateAclTensor(in.cuSeqlensQ, cuShape, &cuDev, ACL_INT32, &cuT) : ret;
        ret = ret == ACL_SUCCESS ? CreateAclTensor(in.attnMask, maskShape, &maskDev, ACL_INT8, &maskT) : ret;
        ret = ret == ACL_SUCCESS ? CreateAclTensor(in.metadata, metaShape, &metaDev, ACL_INT32, &metaT) : ret;
        std::vector<uint16_t> outRaw(outShape[0] * outShape[1] * outShape[2], 0);
        ret = ret == ACL_SUCCESS ? CreateRawAclTensor(outRaw.data(), outRaw.size() * sizeof(uint16_t), outShape, qDtype, &outDev,
                                                      &outT) :
                                   ret;
        std::vector<float> lseHost(lseShape[0] * lseShape[1], 0.0f);
        ret = ret == ACL_SUCCESS ? CreateAclTensor(lseHost, lseShape, &lseDev, ACL_FLOAT, &lseT) : ret;
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("CreateAclTensor failed: %d\n", ret);
            return ret;
        }

        // ---- 1. metadata op -------------------------------------------------------------
        char layoutQ[] = "TND";
        char layoutKv[] = "PA_NZ";
        char layoutOut[] = "NTD"; // MLA 输出头主序 (N,T,D)
        void *cuDevM = nullptr;
        void *seqKvDevM = nullptr;
        aclTensor *cuTM = nullptr;
        aclTensor *seqKvTM = nullptr;
        std::vector<int64_t> cuShapeM = {cs.batch + 1};
        std::vector<int64_t> seqKvShapeM = {cs.batch};
        ret = CreateAclTensor(in.cuSeqlensQ, cuShapeM, &cuDevM, ACL_INT32, &cuTM);
        ret = ret == ACL_SUCCESS ? CreateAclTensor(in.cacheSeqlens, seqKvShapeM, &seqKvDevM, ACL_INT32, &seqKvTM) : ret;
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("CreateAclTensor(metadata seq-lens) failed: %d\n", ret);
            return ret;
        }
        uint64_t wsSize = 0;
        aclOpExecutor *executor = nullptr;
        // 16 参契约 (cuSeqlensQ, cacheSeqlens, sequsedQ, batchSize, maxSeqlenQ, maxSeqlenKv,
        // numHeadsQ, numHeadsKv, headDim, maskMode, layoutQ, layoutKv, layoutOut, metaData...)
        // headDim=512: MLA checker 强制 head_dim==512（qk nope 宽）。
        ret = aclnnFlashMlaWithKvcacheMetadataGetWorkspaceSize(cuTM, seqKvTM, nullptr, cs.batch, cs.qLens[0], cs.kvLen,
                                                    cs.numHeadsQ, 1, 512, 3, layoutQ, layoutKv, layoutOut, metaT,
                                                    &wsSize, &executor);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("aclnnFlashMlaWithKvcacheMetadataGetWorkspaceSize failed: %d\n", ret);
            return ret;
        }
        void *wsAddr = nullptr;
        if (wsSize > 0U) {
            ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (!CHECK_RET(ret == ACL_SUCCESS)) {
                LOG_PRINT("metadata workspace malloc failed: %d\n", ret);
                return ret;
            }
        }
        ret = aclnnFlashMlaWithKvcacheMetadata(wsAddr, wsSize, executor, stream);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("aclnnFlashMlaWithKvcacheMetadata failed: %d\n", ret);
            return ret;
        }
        ret = aclrtSynchronizeStream(stream);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("metadata sync failed: %d\n", ret);
            return ret;
        }
        ret = aclrtMemcpy(in.metadata.data(), in.metadata.size() * sizeof(int32_t), metaDev,
                          in.metadata.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("metadata copyback failed: %d\n", ret);
            return ret;
        }
        aclDestroyTensor(cuTM);
        aclrtFree(cuDevM);
        aclDestroyTensor(seqKvTM);
        aclrtFree(seqKvDevM);
        int32_t sectionNum = in.metadata[0];
        LOG_PRINT("  metadata: sectionNum=%d (expected %s)\n", sectionNum,
                  cs.expectMultiSection ? ">=2" : ">=1");
        if (cs.expectMultiSection && sectionNum < 2) {
            LOG_PRINT("  [FAIL] expected >=2 sections but got %d\n", sectionNum);
            totalFail++;
            continue;
        }

        // ---- 2. main op -------------------------------------------------------------------
        wsSize = 0;
        executor = nullptr;
        ret = aclnnFlashMlaWithKvcacheGetWorkspaceSize(qT, kT, btT, clT, cuT, nullptr, maskT, metaT, QK_HEAD_DIM, softmaxScale, 3,
                                                       cs.qLens[0] > 8 ? cs.qLens[0] : -1,
                                                       cs.qLens[0] > 8 ? cs.kvLen : -1, layoutQ, layoutKv, layoutOut, 1, outT,
                                                       lseT, &wsSize, &executor);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("aclnnFlashMlaWithKvcacheGetWorkspaceSize failed: %d\n", ret);
            return ret;
        }
        void *mainWs = nullptr;
        if (wsSize > 0U) {
            ret = aclrtMalloc(&mainWs, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (!CHECK_RET(ret == ACL_SUCCESS)) {
                LOG_PRINT("workspace malloc failed: %d\n", ret);
                return ret;
            }
        }
        ret = aclnnFlashMlaWithKvcache(mainWs, wsSize, executor, stream);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("aclnnFlashMlaWithKvcache failed: %d\n", ret);
            return ret;
        }
        ret = aclrtSynchronizeStream(stream);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("main sync failed: %d\n", ret);
            return ret;
        }
        ret = aclrtMemcpy(outRaw.data(), outRaw.size() * sizeof(uint16_t), outDev, outRaw.size() * sizeof(uint16_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("out copyback failed: %d\n", ret);
            return ret;
        }
        ret = aclrtMemcpy(lseHost.data(), lseHost.size() * sizeof(float), lseDev, lseHost.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("lse copyback failed: %d\n", ret);
            return ret;
        }

        // Convert the raw fp16/bf16 output words to float for the comparison.
        std::vector<float> outRounded(outRaw.size(), 0.0f);
        for (size_t i = 0; i < outRaw.size(); ++i) {
            if (cs.dtype == ACL_FLOAT16) {
                outRounded[i] = static_cast<float>(op::fp16_t(outRaw[i]));
            } else {
                outRounded[i] = static_cast<float>(op::bfloat16(outRaw[i], op::bfloat16::from_bits()));
            }
        }

        // ---- 3. golden + compare -----------------------------------------------------------
        GoldenResult golden = ComputeGolden(cs, in, softmaxScale, /*causal=*/true);
        // actual NTD (N,T,D): flatten [N][T][D]; expect [N][T][D]
        std::vector<double> expectO(golden.o.begin(), golden.o.end());
        double atol = (cs.dtype == ACL_FLOAT16) ? FP16_ATOL : BF16_ATOL;
        double rtol = (cs.dtype == ACL_FLOAT16) ? FP16_RTOL : BF16_RTOL;
        bool okO = CompareWithTolerance("attn_out", expectO, outRounded, 0, 0, atol, rtol);
        bool okL = CompareWithTolerance("softmax_lse", golden.lse, lseHost, 0, 0, 1e-4, 1e-3);
        bool ok = okO && okL;
        LOG_PRINT("  ==> case %s: %s\n", cs.name, ok ? "PASS" : "FAIL");
        totalPass += ok ? 1 : 0;
        totalFail += ok ? 0 : 1;

        if (wsAddr != nullptr) {
            aclrtFree(wsAddr);
        }
        aclDestroyTensor(qT);
        aclDestroyTensor(kT);
        aclDestroyTensor(btT);
        aclDestroyTensor(clT);
        aclDestroyTensor(cuT);
        aclDestroyTensor(maskT);
        aclDestroyTensor(metaT);
        aclDestroyTensor(outT);
        aclDestroyTensor(lseT);
        aclrtFree(qDev);
        aclrtFree(kDev);
        aclrtFree(btDev);
        aclrtFree(clDev);
        aclrtFree(cuDev);
        aclrtFree(maskDev);
        aclrtFree(metaDev);
        aclrtFree(outDev);
        aclrtFree(lseDev);
        if (mainWs != nullptr) {
            aclrtFree(mainWs);
        }
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("\n==================== summary: pass=%d fail=%d ====================\n", totalPass, totalFail);
    return totalFail == 0 ? 0 : 1;
}
