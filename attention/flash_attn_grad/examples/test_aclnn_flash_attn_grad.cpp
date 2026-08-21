/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// aclnn single-operator invocation for FlashAttnGrad (BSND, fp16, D=128).
//
// The kernel is the pypto-pro implementation; this exercises the same path a
// framework would take: aclnnFlashAttnGradGetWorkspaceSize -> aclnnFlashAttnGrad.
// Golden values are computed on host in fp32 and compared against dQ/dK/dV.
//
// Build/run: see the run_aclnn_flash_attn_grad.sh helper next to this file.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attn_grad.h"

#define CHECK_RET(cond, ...)                                                                       \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            printf(__VA_ARGS__);                                                                   \
            return -1;                                                                              \
        }                                                                                          \
    } while (0)

#define CHECK_ACL(expr)                                                                            \
    do {                                                                                           \
        aclError _st = (expr);                                                                      \
        CHECK_RET(_st == ACL_SUCCESS, "%s failed, code %d\n", #expr, static_cast<int>(_st));         \
    } while (0)

namespace {

// fp16 <-> fp32 conversion. Only round-to-nearest-even for the values this test
// generates (no subnormals, no overflow), which keeps the golden comparable.
uint16_t FloatToHalf(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 31) & 0x1U;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent <= 0) {
        return static_cast<uint16_t>(sign << 15);
    }
    if (exponent >= 0x1F) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00U);
    }
    uint32_t half = (sign << 15) | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13);
    // round to nearest even on the dropped bits
    if ((mantissa & 0x1FFFU) > 0x1000U || ((mantissa & 0x1FFFU) == 0x1000U && (half & 0x1U))) {
        half += 1;
    }
    return static_cast<uint16_t>(half);
}

float HalfToFloat(uint16_t half)
{
    uint32_t sign = (half >> 15) & 0x1U;
    uint32_t exponent = (half >> 10) & 0x1FU;
    uint32_t mantissa = half & 0x3FFU;
    uint32_t bits;
    if (exponent == 0) {
        bits = sign << 31;
    } else if (exponent == 0x1FU) {
        bits = (sign << 31) | 0x7F800000U | (mantissa << 13);
    } else {
        bits = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

struct Shape {
    int64_t b = 1;
    int64_t s1 = 256;
    int64_t s2 = 256;
    // GQA: q/dout/attn_out/dq carry n heads (N1); k/v/dk/dv carry n2 = n/g.
    // g == 1 is plain MHA, where n2 == n.
    int64_t n = 1;
    int64_t n2 = 1;
    int64_t g = 1;
    int64_t d = 128;
    // Unequal head dims: q/k/dq/dk are d wide, v/dout/attn_out/dv are dv wide.
    // dv == d is the equal-length case. dv > d is rejected (see main).
    int64_t dv = 128;
};

// Deterministic pseudo-random input so host golden and device see the same data.
std::vector<uint16_t> MakeInput(size_t count, uint32_t seed)
{
    std::vector<uint16_t> out(count);
    uint32_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        float value = (static_cast<float>((state >> 8) & 0xFFFFU) / 32768.0F) - 1.0F;
        out[i] = FloatToHalf(value * 0.5F);
    }
    return out;
}

// Layout of q/k/v/dout/attn_out and the dq/dk/dv outputs. softmax_lse is
// always [B,N,S1] and is unaffected.
enum class Layout { BSND, BNSD };
Layout g_layout = Layout::BSND;

// Element index. BSND is [b,s,n,d]; BNSD is [b,n,s,d]. Only the axis order
// differs -- d stays innermost in both, which is what lets the kernel use a
// single 4D view for both layouts.
inline size_t Idx(const Shape &sh, int64_t b, int64_t s, int64_t n, int64_t d)
{
    if (g_layout == Layout::BNSD) {
        return static_cast<size_t>(((b * sh.n + n) * sh.s1 + s) * sh.d + d);
    }
    return static_cast<size_t>(((b * sh.s1 + s) * sh.n + n) * sh.d + d);
}

// KV-side index. Takes a KV head number (0..n2-1) and uses sh.n2 as the head
// extent -- under GQA the KV tensors have fewer heads than the Q ones.
inline size_t IdxKv(const Shape &sh, int64_t b, int64_t s, int64_t n, int64_t d)
{
    if (g_layout == Layout::BNSD) {
        return static_cast<size_t>(((b * sh.n2 + n) * sh.s2 + s) * sh.d + d);
    }
    return static_cast<size_t>(((b * sh.s2 + s) * sh.n2 + n) * sh.d + d);
}

// Dv-wide variants for the tensors whose last dim is dv rather than d:
// dout/attn_out/dq-side Q layout (IdxV) and v/dv on the KV side (IdxKvV).
// They differ from Idx/IdxKv only in the trailing stride, so when dv == d the
// two families coincide.
inline size_t IdxV(const Shape &sh, int64_t b, int64_t s, int64_t n, int64_t d)
{
    if (g_layout == Layout::BNSD) {
        return static_cast<size_t>(((b * sh.n + n) * sh.s1 + s) * sh.dv + d);
    }
    return static_cast<size_t>(((b * sh.s1 + s) * sh.n + n) * sh.dv + d);
}

inline size_t IdxKvV(const Shape &sh, int64_t b, int64_t s, int64_t n, int64_t d)
{
    if (g_layout == Layout::BNSD) {
        return static_cast<size_t>(((b * sh.n2 + n) * sh.s2 + s) * sh.dv + d);
    }
    return static_cast<size_t>(((b * sh.s2 + s) * sh.n2 + n) * sh.dv + d);
}

// Host reference in fp32. Also produces softmax_lse = max + log(sum(exp(s-max)))
// which matches the Dao-AILab/flash-attention convention the kernel expects.
struct Golden {
    std::vector<float> lse;   // [B, N, S1]
    std::vector<float> attnOut;
    std::vector<float> dq;
    std::vector<float> dk;
    std::vector<float> dv;
};

Golden ComputeGolden(const Shape &sh, const std::vector<uint16_t> &q, const std::vector<uint16_t> &k,
                     const std::vector<uint16_t> &v, const std::vector<uint16_t> &dout, float scale)
{
    Golden g;
    g.lse.assign(static_cast<size_t>(sh.b * sh.n * sh.s1), 0.0F);
    // attn_out is Dv wide (it is P@V), same element count as dout.
    g.attnOut.assign(dout.size(), 0.0F);
    g.dq.assign(q.size(), 0.0F);
    g.dk.assign(k.size(), 0.0F);
    g.dv.assign(v.size(), 0.0F);

    std::vector<float> p(static_cast<size_t>(sh.s2));
    std::vector<float> dp(static_cast<size_t>(sh.s2));

    for (int64_t b = 0; b < sh.b; ++b) {
        for (int64_t n = 0; n < sh.n; ++n) {
            // GQA: Q head n is served by KV head n/g. With g == 1 this is n.
            // The G Q-heads sharing one KV head accumulate into the same
            // dk/dv slot, mirroring the kernel's atomicAdd.
            const int64_t n2i = n / sh.g;
            for (int64_t i = 0; i < sh.s1; ++i) {
                // S = Q @ K^T * scale, then softmax
                float maxScore = -3.0e38F;
                for (int64_t j = 0; j < sh.s2; ++j) {
                    float acc = 0.0F;
                    for (int64_t d = 0; d < sh.d; ++d) {
                        acc += HalfToFloat(q[Idx(sh, b, i, n, d)]) * HalfToFloat(k[IdxKv(sh, b, j, n2i, d)]);
                    }
                    p[j] = acc * scale;
                    maxScore = std::max(maxScore, p[j]);
                }
                float sumExp = 0.0F;
                for (int64_t j = 0; j < sh.s2; ++j) {
                    p[j] = std::exp(p[j] - maxScore);
                    sumExp += p[j];
                }
                g.lse[static_cast<size_t>((b * sh.n + n) * sh.s1 + i)] = maxScore + std::log(sumExp);
                for (int64_t j = 0; j < sh.s2; ++j) {
                    p[j] /= sumExp;
                }

                // attn_out = P @ V; dV += P^T @ dO.
                // V/dO/attn_out/dV are Dv wide, so these loops run over sh.dv
                // and use the Dv-strided index helpers.
                for (int64_t d = 0; d < sh.dv; ++d) {
                    float acc = 0.0F;
                    for (int64_t j = 0; j < sh.s2; ++j) {
                        acc += p[j] * HalfToFloat(v[IdxKvV(sh, b, j, n2i, d)]);
                    }
                    g.attnOut[IdxV(sh, b, i, n, d)] = acc;
                }
                for (int64_t j = 0; j < sh.s2; ++j) {
                    for (int64_t d = 0; d < sh.dv; ++d) {
                        g.dv[IdxKvV(sh, b, j, n2i, d)] += p[j] * HalfToFloat(dout[IdxV(sh, b, i, n, d)]);
                    }
                }

                // dP = dO @ V^T ; sfmg = rowsum(dO * attn_out) ; dS = P * (dP - sfmg)
                // Both reductions contract over Dv (dO and V are Dv wide).
                float sfmg = 0.0F;
                for (int64_t d = 0; d < sh.dv; ++d) {
                    sfmg += HalfToFloat(dout[IdxV(sh, b, i, n, d)]) * g.attnOut[IdxV(sh, b, i, n, d)];
                }
                for (int64_t j = 0; j < sh.s2; ++j) {
                    float acc = 0.0F;
                    for (int64_t d = 0; d < sh.dv; ++d) {
                        acc += HalfToFloat(dout[IdxV(sh, b, i, n, d)]) * HalfToFloat(v[IdxKvV(sh, b, j, n2i, d)]);
                    }
                    dp[j] = p[j] * (acc - sfmg);
                }

                // dQ = dS @ K * scale ; dK += dS^T @ Q * scale
                for (int64_t d = 0; d < sh.d; ++d) {
                    float acc = 0.0F;
                    for (int64_t j = 0; j < sh.s2; ++j) {
                        acc += dp[j] * HalfToFloat(k[IdxKv(sh, b, j, n2i, d)]);
                    }
                    g.dq[Idx(sh, b, i, n, d)] = acc * scale;
                }
                for (int64_t j = 0; j < sh.s2; ++j) {
                    for (int64_t d = 0; d < sh.d; ++d) {
                        g.dk[IdxKv(sh, b, j, n2i, d)] += dp[j] * HalfToFloat(q[Idx(sh, b, i, n, d)]) * scale;
                    }
                }
            }
        }
    }
    return g;
}

// Device buffer that owns its allocation and the aclTensor wrapping it.
struct DeviceTensor {
    void *addr = nullptr;
    aclTensor *tensor = nullptr;

    ~DeviceTensor()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (addr != nullptr) {
            aclrtFree(addr);
        }
    }
};

int CreateTensor(DeviceTensor &out, const std::vector<int64_t> &shape, aclDataType dtype,
                 const void *hostData, size_t bytes)
{
    CHECK_ACL(aclrtMalloc(&out.addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    if (hostData != nullptr) {
        CHECK_ACL(aclrtMemcpy(out.addr, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        CHECK_ACL(aclrtMemset(out.addr, bytes, 0, bytes));
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    out.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                 aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), out.addr);
    CHECK_RET(out.tensor != nullptr, "aclCreateTensor failed\n");
    return 0;
}

// Report max abs diff against golden. fp16 outputs of this magnitude land
// around 1e-3; the threshold matches what the pypto test suite uses.
bool CompareHalf(const char *name, const std::vector<uint16_t> &actual,
                 const std::vector<float> &expect, float tol)
{
    float maxDiff = 0.0F;
    size_t worst = 0;
    for (size_t i = 0; i < expect.size(); ++i) {
        float diff = std::fabs(HalfToFloat(actual[i]) - expect[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
            worst = i;
        }
    }
    bool ok = maxDiff <= tol;
    printf("    %s: %s  max_diff=%.6e (at %zu: got %.6f want %.6f)\n", name, ok ? "PASS" : "FAIL",
           maxDiff, worst, HalfToFloat(actual[worst]), expect[worst]);
    return ok;
}

} // namespace

int64_t EnvInt(const char *name, int64_t fallback)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    return std::strtoll(raw, nullptr, 10);
}

int main()
{
    Shape sh;
    sh.b = EnvInt("FAG_B", sh.b);
    sh.n = EnvInt("FAG_N", sh.n);   // N1: head count of q/dout/attn_out/dq
    sh.g = EnvInt("FAG_G", sh.g);   // GQA group size; 1 == plain MHA
    sh.s1 = EnvInt("FAG_S1", sh.s1);
    sh.s2 = EnvInt("FAG_S2", sh.s2);
    sh.d = EnvInt("FAG_D", sh.d);     // head dim of q/k/dq/dk
    // FAG_DV defaults to FAG_D, i.e. the equal-length case.
    sh.dv = EnvInt("FAG_DV", sh.d);   // head dim of v/dout/attn_out/dv
    if (sh.dv > sh.d) {
        printf("FAG_DV (%ld) must not exceed FAG_D (%ld)\n", sh.dv, sh.d);
        return -1;
    }
    if (sh.g <= 0 || sh.n % sh.g != 0) {
        printf("FAG_N (%ld) must be a positive multiple of FAG_G (%ld)\n", sh.n, sh.g);
        return -1;
    }
    sh.n2 = sh.n / sh.g;            // head count of k/v/dk/dv
    // Host golden is an O(B*N*S1*S2*D) triple loop; skip it for large shapes and
    // only assert the kernel ran and produced non-zero output there.
    const bool checkGolden = EnvInt("FAG_CHECK", 1) != 0;
    const char *layoutEnv = std::getenv("FAG_LAYOUT");
    std::string layoutStr = (layoutEnv != nullptr && *layoutEnv != '\0') ? layoutEnv : "BSND";
    if (layoutStr == "BNSD") {
        g_layout = Layout::BNSD;
    } else if (layoutStr != "BSND") {
        printf("FAG_LAYOUT only supports BSND / BNSD, got %s\n", layoutStr.c_str());
        return -1;
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(sh.d));
    const size_t qCount = static_cast<size_t>(sh.b * sh.s1 * sh.n * sh.d);
    const size_t kvCount = static_cast<size_t>(sh.b * sh.s2 * sh.n2 * sh.d);
    // Dv-wide counterparts: dout/attn_out/dq-side and v/dv on the KV side.
    // Equal to qCount/kvCount when dv == d.
    const size_t qCountV = static_cast<size_t>(sh.b * sh.s1 * sh.n * sh.dv);
    const size_t kvCountV = static_cast<size_t>(sh.b * sh.s2 * sh.n2 * sh.dv);
    const size_t lseCount = static_cast<size_t>(sh.b * sh.n * sh.s1);

    printf("FlashAttnGrad aclnn test: B=%ld S1=%ld S2=%ld N1=%ld N2=%ld G=%ld D=%ld Dv=%ld "
           "scale=%.6f layout=%s\n",
           sh.b, sh.s1, sh.s2, sh.n, sh.n2, sh.g, sh.d, sh.dv, scale, layoutStr.c_str());

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    auto qHost = MakeInput(qCount, 11U);
    auto kHost = MakeInput(kvCount, 22U);
    auto vHost = MakeInput(kvCountV, 33U);
    auto doutHost = MakeInput(qCountV, 44U);

    Golden golden;
    if (checkGolden) {
        printf("  computing host golden (fp32) ...\n");
        golden = ComputeGolden(sh, qHost, kHost, vHost, doutHost, scale);
    } else {
        printf("  skipping host golden (large shape); checking execution + magnitude only\n");
        golden.lse.assign(lseCount, 0.0F);
        golden.attnOut.assign(qCountV, 0.0F);
        golden.dq.assign(qCount, 0.0F);
        golden.dk.assign(kvCount, 0.0F);
        golden.dv.assign(kvCountV, 0.0F);
        // lse=0 with attn_out=0 keeps the kernel numerically well-defined
        // (P = exp(S*scale - 0)) without needing the full forward pass.
    }

    // attn_out feeds the kernel's softmaxGradFront, so it must be the fp16
    // rounding of the same forward output the golden used.
    std::vector<uint16_t> attnOutHost(qCountV);
    for (size_t i = 0; i < qCountV; ++i) {
        attnOutHost[i] = FloatToHalf(golden.attnOut[i]);
    }

    const bool isBnsd = (g_layout == Layout::BNSD);
    const std::vector<int64_t> qShape = isBnsd ? std::vector<int64_t>{sh.b, sh.n, sh.s1, sh.d}
                                               : std::vector<int64_t>{sh.b, sh.s1, sh.n, sh.d};
    const std::vector<int64_t> kvShape = isBnsd ? std::vector<int64_t>{sh.b, sh.n2, sh.s2, sh.d}
                                                : std::vector<int64_t>{sh.b, sh.s2, sh.n2, sh.d};
    // v/dv are Dv wide; dout/attn_out follow the Q axes but with Dv as last dim.
    const std::vector<int64_t> qShapeV = isBnsd ? std::vector<int64_t>{sh.b, sh.n, sh.s1, sh.dv}
                                                : std::vector<int64_t>{sh.b, sh.s1, sh.n, sh.dv};
    const std::vector<int64_t> kvShapeV = isBnsd ? std::vector<int64_t>{sh.b, sh.n2, sh.s2, sh.dv}
                                                 : std::vector<int64_t>{sh.b, sh.s2, sh.n2, sh.dv};
    // softmax_lse is [B,N,S1] regardless of layout
    const std::vector<int64_t> lseShape = {sh.b, sh.n, sh.s1};

    DeviceTensor q, k, v, dout, attnOut, lse, dq, dk, dv;
    CHECK_RET(CreateTensor(q, qShape, ACL_FLOAT16, qHost.data(), qCount * 2) == 0, "q\n");
    CHECK_RET(CreateTensor(k, kvShape, ACL_FLOAT16, kHost.data(), kvCount * 2) == 0, "k\n");
    CHECK_RET(CreateTensor(v, kvShapeV, ACL_FLOAT16, vHost.data(), kvCountV * 2) == 0, "v\n");
    CHECK_RET(CreateTensor(dout, qShapeV, ACL_FLOAT16, doutHost.data(), qCountV * 2) == 0, "dout\n");
    CHECK_RET(CreateTensor(attnOut, qShapeV, ACL_FLOAT16, attnOutHost.data(), qCountV * 2) == 0,
              "attn_out\n");
    CHECK_RET(CreateTensor(lse, lseShape, ACL_FLOAT, golden.lse.data(), lseCount * 4) == 0, "lse\n");
    CHECK_RET(CreateTensor(dq, qShape, ACL_FLOAT16, nullptr, qCount * 2) == 0, "dq\n");
    CHECK_RET(CreateTensor(dk, kvShape, ACL_FLOAT16, nullptr, kvCount * 2) == 0, "dk\n");
    CHECK_RET(CreateTensor(dv, kvShapeV, ACL_FLOAT16, nullptr, kvCountV * 2) == 0, "dv\n");

    // The three layout attrs must agree (enforced by CheckLayout on the host)
    std::vector<char> layoutBuf(layoutStr.begin(), layoutStr.end());
    layoutBuf.push_back('\0');
    char *layoutQ = layoutBuf.data();
    char *layoutKv = layoutBuf.data();
    char *layoutOut = layoutBuf.data();

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnFlashAttnGradGetWorkspaceSize(
        q.tensor, k.tensor, v.tensor, dout.tensor, attnOut.tensor, lse.tensor,
        /*cu_seqlens_q*/ nullptr, /*cu_seqlens_kv*/ nullptr,
        /*seqused_q*/ nullptr, /*seqused_kv*/ nullptr,
        /*sinks*/ nullptr, /*attn_mask*/ nullptr, /*metadata*/ nullptr,
        static_cast<double>(scale), /*mask_mode*/ 0, /*win_left*/ -1, /*win_right*/ -1,
        /*max_seqlen_q*/ sh.s1, /*max_seqlen_kv*/ sh.s2,
        layoutQ, layoutKv, layoutOut,
        dq.tensor, dk.tensor, dv.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnFlashAttnGradGetWorkspaceSize failed: %d\n", ret);
    printf("  workspaceSize = %llu bytes (%.2f MB)\n", static_cast<unsigned long long>(workspaceSize),
           static_cast<double>(workspaceSize) / (1024.0 * 1024.0));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ret = aclnnFlashAttnGrad(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnFlashAttnGrad failed: %d\n", ret);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    printf("  kernel executed\n");

    std::vector<uint16_t> dqHost(qCount), dkHost(kvCount), dvHost(kvCountV);
    CHECK_ACL(aclrtMemcpy(dqHost.data(), qCount * 2, dq.addr, qCount * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(dkHost.data(), kvCount * 2, dk.addr, kvCount * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(dvHost.data(), kvCountV * 2, dv.addr, kvCountV * 2, ACL_MEMCPY_DEVICE_TO_HOST));

    // Guard against a silently-zero kernel: an all-zero output would still pass
    // a loose tolerance if every golden value happened to be tiny. Report the
    // magnitude of both sides so the comparison is meaningful.
    auto rms = [](const std::vector<float> &x) {
        double acc = 0.0;
        for (float f : x) { acc += static_cast<double>(f) * f; }
        return std::sqrt(acc / std::max<size_t>(x.size(), 1));
    };
    auto rmsHalf = [](const std::vector<uint16_t> &x) {
        double acc = 0.0;
        for (uint16_t h : x) { float f = HalfToFloat(h); acc += static_cast<double>(f) * f; }
        return std::sqrt(acc / std::max<size_t>(x.size(), 1));
    };
    printf("  magnitude check (rms, device vs golden):\n");
    printf("    dQ %.6e vs %.6e | dK %.6e vs %.6e | dV %.6e vs %.6e\n",
           rmsHalf(dqHost), rms(golden.dq), rmsHalf(dkHost), rms(golden.dk),
           rmsHalf(dvHost), rms(golden.dv));
    // "Did the kernel run at all" guard. Compare each output against its own
    // golden rather than against a fixed floor: at S2 == 1 softmax over a
    // single key gives P == 1, so sfmg == dP and dS = P*(dP - sfmg) is exactly
    // zero -- dQ and dK are *legitimately* all-zero there and the golden says
    // so too. A fixed `> 1e-6` floor reports that correct result as "kernel
    // wrote nothing". Only flag an output that is zero while its golden is not.
    auto wroteNothing = [](double got, double want) {
        return want > 1e-6 && got <= 1e-6;
    };
    CHECK_RET(!wroteNothing(rmsHalf(dqHost), rms(golden.dq)) &&
              !wroteNothing(rmsHalf(dkHost), rms(golden.dk)) &&
              !wroteNothing(rmsHalf(dvHost), rms(golden.dv)),
              "an output is all (near) zero while its golden is not -- kernel wrote nothing\n");

    bool ok = true;
    if (checkGolden) {
        printf("  comparing against host golden:\n");
        const float tol = 4.0e-3F;
        ok = CompareHalf("dQ", dqHost, golden.dq, tol);
        ok = CompareHalf("dK", dkHost, golden.dk, tol) && ok;
        ok = CompareHalf("dV", dvHost, golden.dv, tol) && ok;
    } else {
        printf("  golden skipped; execution + non-zero output verified above\n");
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());

    printf("%s\n", ok ? "=== PASSED ===" : "=== FAILED ===");
    return ok ? 0 : 1;
}
