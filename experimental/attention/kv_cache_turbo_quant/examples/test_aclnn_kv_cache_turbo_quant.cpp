/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_linalg_vector_norm.h"
#include "aclnnop/aclnn_mul.h"
#include "tiling/matmul/bmm_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void kvtq_cast_fp32(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *kv, uint8_t *fp32,
                              uint32_t rowCount, uint32_t coreNum);
extern "C" void mm_fp32_transb(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *a, uint8_t *b, uint8_t *c,
                              uint8_t *workspace, uint8_t *tiling);
extern "C" void kvtq_quantize(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *rotated, uint8_t *quantIdx,
                             uint8_t *residual, uint32_t rowCount, uint32_t mseBits, uint32_t coreNum);
extern "C" void kvtq_store_bf16(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *normFp, uint8_t *gammaFp,
                               uint8_t *normBf, uint8_t *gammaBf, uint32_t rowCount, uint32_t coreNum);
extern "C" void kvtq_sign(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *projected, uint8_t *quantQjl,
                         uint32_t rowCount, uint32_t coreNum);

#define CHECK_NN(expr)                                                                                                \
    do {                                                                                                              \
        aclnnStatus _s = (expr);                                                                                      \
        if (_s != 0) {                                                                                                \
            std::printf("aclnn %d at %s\n", static_cast<int>(_s), #expr);                                            \
            return 1;                                                                                                 \
        }                                                                                                            \
    } while (0)

namespace {

constexpr int kDim = 128;

float kCent3[8] = {-1.902069300e-01f, -1.187859178e-01f, -6.682205945e-02f, -2.166347019e-02f,
                   2.166347019e-02f,  6.682205945e-02f,  1.187859178e-01f,  1.902069300e-01f};

uint16_t FloatToBf16(float x)
{
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<uint16_t>(u >> 16);
}

float Bf16ToFloat(uint16_t b)
{
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

uint32_t XorShift(uint32_t &s)
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

float RandUniform(uint32_t &s, float lo, float hi)
{
    const float u = static_cast<float>(XorShift(s) >> 8) / static_cast<float>(1u << 24);
    return lo + (hi - lo) * u;
}

#define CHECK_ACL(expr)                                                                                               \
    do {                                                                                                              \
        aclError _e = (expr);                                                                                         \
        if (_e != ACL_SUCCESS) {                                                                                      \
            std::printf("ACL error %d at %s\n", static_cast<int>(_e), #expr);                                        \
            return 1;                                                                                                 \
        }                                                                                                            \
    } while (0)

struct Device {
    aclrtStream stream = nullptr;
    uint32_t aiv = 1;
    uint32_t aic = 1;
    platform_ascendc::PlatformAscendC *platform = nullptr;
};

int PrepareTiling(Device &dev, int rows, AscendC::tiling::TCubeTiling &tiling, void *dTiling)
{
    matmul_tiling::MultiCoreMatmulTiling tilingApi{*dev.platform};
    tilingApi.SetDim(static_cast<int32_t>(dev.aic));
    tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                       false);
    tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT,
                       true);
    tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                          matmul_tiling::DataType::DT_FLOAT);
    tilingApi.SetShape(rows, kDim, kDim);
    tilingApi.SetOrgShape(rows, kDim, kDim);
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);
    const int64_t ret = tilingApi.GetTiling(tiling);
    if (ret == -1) {
        std::printf("GetTiling failed rows=%d\n", rows);
        return 1;
    }
    CHECK_ACL(aclrtMemcpy(dTiling, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));
    return 0;
}

struct ExactOps {
    int64_t dim = 1;
    float ord = 2.f;
    aclScalar *ordS = nullptr;
    aclIntArray *dims = nullptr;
    aclTensor *xSrc = nullptr;
    aclTensor *xDst = nullptr;
    aclTensor *xNorm = nullptr;
    aclTensor *rSrc = nullptr;
    aclTensor *rDst = nullptr;
    aclTensor *rNorm = nullptr;
    aclTensor *gammaFp = nullptr;
    aclOpExecutor *exXN = nullptr;
    aclOpExecutor *exXD = nullptr;
    aclOpExecutor *exRN = nullptr;
    aclOpExecutor *exRD = nullptr;
    aclOpExecutor *exMul = nullptr;
    uint64_t wsXN = 0;
    uint64_t wsXD = 0;
    uint64_t wsRN = 0;
    uint64_t wsRD = 0;
    uint64_t wsMul = 0;
};

aclTensor *MakeMatrix(void *ptr, int64_t rows, int64_t cols)
{
    const int64_t dims[2] = {rows, cols};
    const int64_t stride[2] = {cols, 1};
    return aclCreateTensor(dims, 2, ACL_FLOAT, stride, 0, ACL_FORMAT_ND, dims, 2, ptr);
}

aclTensor *MakeColumn(void *ptr, int64_t rows)
{
    const int64_t dims[2] = {rows, 1};
    const int64_t stride[2] = {1, 1};
    return aclCreateTensor(dims, 2, ACL_FLOAT, stride, 0, ACL_FORMAT_ND, dims, 2, ptr);
}

int PrepareExact(ExactOps &op, int rows, void *fpX, void *unit, void *xNorm, void *residual, void *rNorm, void *gammaFp)
{
    op.ordS = aclCreateScalar(&op.ord, ACL_FLOAT);
    op.dims = aclCreateIntArray(&op.dim, 1);
    op.xSrc = MakeMatrix(fpX, rows, kDim);
    op.xDst = MakeMatrix(unit, rows, kDim);
    op.xNorm = MakeColumn(xNorm, rows);
    op.rSrc = MakeMatrix(residual, rows, kDim);
    op.rDst = MakeMatrix(unit, rows, kDim);
    op.rNorm = MakeColumn(rNorm, rows);
    op.gammaFp = MakeColumn(gammaFp, rows);
    if (op.ordS == nullptr || op.dims == nullptr || op.xSrc == nullptr || op.xDst == nullptr || op.xNorm == nullptr ||
        op.rSrc == nullptr || op.rDst == nullptr || op.rNorm == nullptr || op.gammaFp == nullptr) {
        std::printf("aclCreateTensor failed\n");
        return 1;
    }
    CHECK_NN(aclnnLinalgVectorNormGetWorkspaceSize(op.xSrc, op.ordS, op.dims, true, ACL_FLOAT, op.xNorm, &op.wsXN,
                                                   &op.exXN));
    CHECK_NN(aclSetAclOpExecutorRepeatable(op.exXN));
    CHECK_NN(aclnnDivGetWorkspaceSize(op.xSrc, op.xNorm, op.xDst, &op.wsXD, &op.exXD));
    CHECK_NN(aclSetAclOpExecutorRepeatable(op.exXD));
    CHECK_NN(aclnnLinalgVectorNormGetWorkspaceSize(op.rSrc, op.ordS, op.dims, true, ACL_FLOAT, op.rNorm, &op.wsRN,
                                                   &op.exRN));
    CHECK_NN(aclSetAclOpExecutorRepeatable(op.exRN));
    CHECK_NN(aclnnDivGetWorkspaceSize(op.rSrc, op.rNorm, op.rDst, &op.wsRD, &op.exRD));
    CHECK_NN(aclSetAclOpExecutorRepeatable(op.exRD));
    CHECK_NN(aclnnMulGetWorkspaceSize(op.xNorm, op.rNorm, op.gammaFp, &op.wsMul, &op.exMul));
    CHECK_NN(aclSetAclOpExecutorRepeatable(op.exMul));
    return 0;
}

void DestroyExact(ExactOps &op)
{
    if (op.exXN != nullptr) {
        aclDestroyAclOpExecutor(op.exXN);
    }
    if (op.exXD != nullptr) {
        aclDestroyAclOpExecutor(op.exXD);
    }
    if (op.exRN != nullptr) {
        aclDestroyAclOpExecutor(op.exRN);
    }
    if (op.exRD != nullptr) {
        aclDestroyAclOpExecutor(op.exRD);
    }
    if (op.exMul != nullptr) {
        aclDestroyAclOpExecutor(op.exMul);
    }
    aclDestroyTensor(op.xSrc);
    aclDestroyTensor(op.xDst);
    aclDestroyTensor(op.xNorm);
    aclDestroyTensor(op.rSrc);
    aclDestroyTensor(op.rDst);
    aclDestroyTensor(op.rNorm);
    aclDestroyTensor(op.gammaFp);
    aclDestroyScalar(op.ordS);
    aclDestroyIntArray(op.dims);
    op = ExactOps{};
}

void Launch(Device &dev, int rows, int bits, uint32_t vecCores, uint32_t cubeCores, ExactOps &exact, void *dKv, void *dH,
            void *dS, void *dUnit, void *dY, void *dR, void *dP, void *dIdx, void *dQjl, void *dNormFp, void *dNorm,
            void *dGammaFp, void *dGamma, void *dWork, void *dTiling)
{
    auto *ws = static_cast<uint8_t *>(dWork);
    kvtq_cast_fp32(vecCores, nullptr, dev.stream, (uint8_t *)dKv, (uint8_t *)dR, static_cast<uint32_t>(rows), vecCores);
    aclnnLinalgVectorNorm(ws, exact.wsXN, exact.exXN, dev.stream);
    aclnnDiv(ws, exact.wsXD, exact.exXD, dev.stream);
    mm_fp32_transb(cubeCores, nullptr, dev.stream, (uint8_t *)dUnit, (uint8_t *)dH, (uint8_t *)dY, (uint8_t *)dWork,
                   (uint8_t *)dTiling);
    kvtq_quantize(vecCores, nullptr, dev.stream, (uint8_t *)dY, (uint8_t *)dIdx, (uint8_t *)dR,
                  static_cast<uint32_t>(rows), static_cast<uint32_t>(bits), vecCores);
    aclnnLinalgVectorNorm(ws, exact.wsRN, exact.exRN, dev.stream);
    aclnnDiv(ws, exact.wsRD, exact.exRD, dev.stream);
    aclnnMul(ws, exact.wsMul, exact.exMul, dev.stream);
    kvtq_store_bf16(vecCores, nullptr, dev.stream, (uint8_t *)dNormFp, (uint8_t *)dGammaFp, (uint8_t *)dNorm,
                    (uint8_t *)dGamma, static_cast<uint32_t>(rows), vecCores);
    mm_fp32_transb(cubeCores, nullptr, dev.stream, (uint8_t *)dUnit, (uint8_t *)dS, (uint8_t *)dP, (uint8_t *)dWork,
                   (uint8_t *)dTiling);
    kvtq_sign(vecCores, nullptr, dev.stream, (uint8_t *)dP, (uint8_t *)dQjl, static_cast<uint32_t>(rows), vecCores);
}

int RunCase(const char *name, int rows, int bits, Device &dev, bool bench)
{
    const int idxBytes = (kDim * bits) / 8;
    const int qjlBytes = kDim / 8;
    std::vector<float> x(static_cast<size_t>(rows) * kDim);
    std::vector<float> H(static_cast<size_t>(kDim) * kDim);
    std::vector<float> S(static_cast<size_t>(kDim) * kDim);
    uint32_t seed = 0x1234u + static_cast<uint32_t>(rows) * 17u + static_cast<uint32_t>(bits);
    if (const char *seedText = std::getenv("KVTQ_SEED")) {
        seed += static_cast<uint32_t>(std::strtoul(seedText, nullptr, 10));
    }
    for (float &v : x) {
        v = RandUniform(seed, -2.f, 2.f);
    }
    for (float &v : H) {
        v = RandUniform(seed, -1.f, 1.f);
    }
    for (float &v : S) {
        v = RandUniform(seed, -4.f, 4.f);
    }
    std::vector<uint16_t> xBf(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        xBf[i] = FloatToBf16(x[i]);
        x[i] = Bf16ToFloat(xBf[i]);
    }
    AscendC::tiling::TCubeTiling tiling{};
    void *dKv = nullptr;
    void *dH = nullptr;
    void *dS = nullptr;
    void *dUnit = nullptr;
    void *dY = nullptr;
    void *dR = nullptr;
    void *dP = nullptr;
    void *dIdx = nullptr;
    void *dQjl = nullptr;
    void *dNormFp = nullptr;
    void *dRNorm = nullptr;
    void *dNorm = nullptr;
    void *dGammaFp = nullptr;
    void *dGamma = nullptr;
    void *dWork = nullptr;
    void *dTiling = nullptr;
    CHECK_ACL(aclrtMalloc(&dKv, xBf.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dH, H.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dS, S.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dUnit, x.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dY, x.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dR, x.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dP, x.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dIdx, static_cast<size_t>(rows) * static_cast<size_t>(idxBytes), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dQjl, static_cast<size_t>(rows) * static_cast<size_t>(qjlBytes), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dNormFp, static_cast<size_t>(rows) * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dRNorm, static_cast<size_t>(rows) * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dNorm, static_cast<size_t>(rows) * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dGammaFp, static_cast<size_t>(rows) * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dGamma, static_cast<size_t>(rows) * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dWork, 64 * 1024 * 1024, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dTiling, sizeof(tiling), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(dKv, xBf.size() * 2, xBf.data(), xBf.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dH, H.size() * 4, H.data(), H.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dS, S.size() * 4, S.data(), S.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    if (PrepareTiling(dev, rows, tiling, dTiling)) {
        return 1;
    }
    ExactOps exact;
    if (PrepareExact(exact, rows, dR, dUnit, dNormFp, dR, dRNorm, dGammaFp)) {
        return 1;
    }
    int32_t cube = 1;
    for (int32_t c = static_cast<int32_t>(dev.aic); c >= 1; --c) {
        if (rows % c == 0 && ((rows / c) % 128 == 0 || rows / c < 128)) {
            cube = c;
            break;
        }
    }
    if (rows < 128) {
        cube = 1;
    }
    tiling.usedCoreNum = cube;
    CHECK_ACL(aclrtMemcpy(dTiling, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));
    uint32_t vecCores = std::min(dev.aiv, static_cast<uint32_t>(rows));
    while (vecCores > 1 && static_cast<uint32_t>(rows) / vecCores < 8) {
        vecCores /= 2;
    }
    const uint32_t cubeCores = static_cast<uint32_t>(cube);
    Launch(dev, rows, bits, vecCores, cubeCores, exact, dKv, dH, dS, dUnit, dY, dR, dP, dIdx, dQjl, dNormFp, dNorm,
           dGammaFp, dGamma, dWork, dTiling);
    CHECK_ACL(aclrtSynchronizeStream(dev.stream));

    const size_t nIdx = static_cast<size_t>(rows) * static_cast<size_t>(idxBytes);
    const size_t nQjl = static_cast<size_t>(rows) * static_cast<size_t>(qjlBytes);
    std::vector<uint8_t> idx(nIdx);
    std::vector<uint8_t> qjl(nQjl);
    std::vector<uint16_t> norm(static_cast<size_t>(rows));
    std::vector<uint16_t> gamma(static_cast<size_t>(rows));
    CHECK_ACL(aclrtMemcpy(idx.data(), idx.size(), dIdx, idx.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(qjl.data(), qjl.size(), dQjl, qjl.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(norm.data(), norm.size() * 2, dNorm, norm.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(gamma.data(), gamma.size() * 2, dGamma, gamma.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    std::printf("[%s] rows=%d bits=%d vec=%u cube=%u\n", name, rows, bits, vecCores, cubeCores);

    double us = 0.0;
    if (bench) {
        for (int i = 0; i < 5; ++i) {
            Launch(dev, rows, bits, vecCores, cubeCores, exact, dKv, dH, dS, dUnit, dY, dR, dP, dIdx, dQjl, dNormFp,
                   dNorm, dGammaFp, dGamma, dWork, dTiling);
        }
        CHECK_ACL(aclrtSynchronizeStream(dev.stream));
        aclrtEvent st = nullptr;
        aclrtEvent ed = nullptr;
        CHECK_ACL(aclrtCreateEvent(&st));
        CHECK_ACL(aclrtCreateEvent(&ed));
        constexpr int kIters = 20;
        CHECK_ACL(aclrtRecordEvent(st, dev.stream));
        for (int i = 0; i < kIters; ++i) {
            Launch(dev, rows, bits, vecCores, cubeCores, exact, dKv, dH, dS, dUnit, dY, dR, dP, dIdx, dQjl, dNormFp,
                   dNorm, dGammaFp, dGamma, dWork, dTiling);
        }
        CHECK_ACL(aclrtRecordEvent(ed, dev.stream));
        CHECK_ACL(aclrtSynchronizeStream(dev.stream));
        float ms = 0.f;
        CHECK_ACL(aclrtEventElapsedTime(&ms, st, ed));
        us = static_cast<double>(ms) * 1000.0 / kIters;
        std::printf("  latency %.3f us\n", us);
        aclrtDestroyEvent(st);
        aclrtDestroyEvent(ed);
    }
    if (const char *dump = std::getenv("KVTQ_DUMP")) {
        const std::string dir = std::string(dump) + "/" + name;
        const std::string cmd = "mkdir -p " + dir;
        if (std::system(cmd.c_str()) != 0) {
            std::printf("dump mkdir failed\n");
        }
        auto writeBin = [&](const char *file, const void *data, size_t bytes) {
            FILE *fp = std::fopen((dir + "/" + file).c_str(), "wb");
            if (fp == nullptr) {
                return;
            }
            std::fwrite(data, 1, bytes, fp);
            std::fclose(fp);
        };
        writeBin("x.bin", x.data(), x.size() * 4);
        writeBin("H.bin", H.data(), H.size() * 4);
        writeBin("S.bin", S.data(), S.size() * 4);
        writeBin("idx.bin", idx.data(), idx.size());
        writeBin("qjl.bin", qjl.data(), qjl.size());
        writeBin("norm.bin", norm.data(), norm.size() * 2);
        writeBin("gamma.bin", gamma.data(), gamma.size() * 2);
        writeBin("bits.txt", &bits, sizeof(bits));
    }
    DestroyExact(exact);
    aclrtFree(dKv);
    aclrtFree(dH);
    aclrtFree(dS);
    aclrtFree(dUnit);
    aclrtFree(dY);
    aclrtFree(dR);
    aclrtFree(dP);
    aclrtFree(dIdx);
    aclrtFree(dQjl);
    aclrtFree(dNormFp);
    aclrtFree(dRNorm);
    aclrtFree(dNorm);
    aclrtFree(dGammaFp);
    aclrtFree(dGamma);
    aclrtFree(dWork);
    aclrtFree(dTiling);
    double cap = 1.0e9;
    if (std::strcmp(name, "gqa_decode_b1") == 0) {
        cap = 149.168;
    } else if (std::strcmp(name, "gqa_decode_b64") == 0) {
        cap = 155.278;
    } else if (std::strcmp(name, "gqa_prefill_t2048") == 0) {
        cap = 385.542;
    } else if (std::strcmp(name, "gqa_mse2_t2048") == 0) {
        cap = 360.796;
    } else if (std::strcmp(name, "gqa_mse4_t2048") == 0) {
        cap = 528.650;
    }
    if (!bench) {
        std::printf("  accuracy_only\n");
        return 0;
    }
    const bool latOk = us < cap;
    std::printf("  latency_result %s cap=%.3f us\n", latOk ? "PASS" : "FAIL", cap);
    return latOk ? 0 : 1;
}

int RunRecon(Device &dev)
{
    constexpr int kRows = 256;
    constexpr int kBits = 3;
    std::vector<float> H(static_cast<size_t>(kDim) * kDim, 0.f);
    H[0] = 1.f;
    for (int n = 1; n < kDim; n <<= 1) {
        std::vector<float> next(static_cast<size_t>(kDim) * kDim, 0.f);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const float v = H[static_cast<size_t>(i) * kDim + j];
                next[static_cast<size_t>(i) * kDim + j] = v;
                next[static_cast<size_t>(i) * kDim + (j + n)] = v;
                next[static_cast<size_t>(i + n) * kDim + j] = v;
                next[static_cast<size_t>(i + n) * kDim + (j + n)] = -v;
            }
        }
        H.swap(next);
    }
    const float hScale = 1.f / std::sqrt(static_cast<float>(kDim));
    for (float &v : H) {
        v *= hScale;
    }
    uint32_t seed = 0x51u;
    auto gauss = [&]() {
        const float u1 = std::max(RandUniform(seed, 0.f, 1.f), 1e-6f);
        const float u2 = RandUniform(seed, 0.f, 1.f);
        return std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * static_cast<float>(M_PI) * u2);
    };
    std::vector<float> S(static_cast<size_t>(kDim) * kDim);
    for (float &v : S) {
        v = gauss();
    }
    std::vector<float> x(static_cast<size_t>(kRows) * kDim);
    for (float &v : x) {
        v = gauss();
    }
    std::vector<uint16_t> xBf(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        xBf[i] = FloatToBf16(x[i]);
        x[i] = Bf16ToFloat(xBf[i]);
    }
    const float *cent = kCent3;
    AscendC::tiling::TCubeTiling tiling{};
    void *dKv = nullptr;
    void *dH = nullptr;
    void *dS = nullptr;
    void *dUnit = nullptr;
    void *dY = nullptr;
    void *dR = nullptr;
    void *dP = nullptr;
    void *dIdx = nullptr;
    void *dQjl = nullptr;
    void *dNormFp = nullptr;
    void *dRNorm = nullptr;
    void *dNorm = nullptr;
    void *dGammaFp = nullptr;
    void *dGamma = nullptr;
    void *dWork = nullptr;
    void *dTiling = nullptr;
    const size_t idxBytes = static_cast<size_t>(kRows) * 48;
    const size_t qjlBytes = static_cast<size_t>(kRows) * 16;
    CHECK_ACL(aclrtMalloc(&dKv, xBf.size() * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dH, H.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dS, S.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dUnit, x.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dY, x.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dR, x.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dP, x.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dIdx, idxBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dQjl, qjlBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dNormFp, static_cast<size_t>(kRows) * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dRNorm, static_cast<size_t>(kRows) * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dNorm, static_cast<size_t>(kRows) * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dGammaFp, static_cast<size_t>(kRows) * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dGamma, static_cast<size_t>(kRows) * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dWork, 64 * 1024 * 1024, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&dTiling, sizeof(tiling), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(dKv, xBf.size() * 2, xBf.data(), xBf.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dH, H.size() * 4, H.data(), H.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dS, S.size() * 4, S.data(), S.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    if (PrepareTiling(dev, kRows, tiling, dTiling)) {
        return 1;
    }
    ExactOps exact;
    if (PrepareExact(exact, kRows, dR, dUnit, dNormFp, dR, dRNorm, dGammaFp)) {
        return 1;
    }
    tiling.usedCoreNum = 2;
    CHECK_ACL(aclrtMemcpy(dTiling, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE));
    Launch(dev, kRows, kBits, 32, 2, exact, dKv, dH, dS, dUnit, dY, dR, dP, dIdx, dQjl, dNormFp, dNorm, dGammaFp,
           dGamma, dWork, dTiling);
    CHECK_ACL(aclrtSynchronizeStream(dev.stream));
    std::vector<uint8_t> idx(idxBytes);
    std::vector<uint8_t> qjl(qjlBytes);
    std::vector<uint16_t> norm(kRows);
    std::vector<uint16_t> gamma(kRows);
    CHECK_ACL(aclrtMemcpy(idx.data(), idx.size(), dIdx, idx.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(qjl.data(), qjl.size(), dQjl, qjl.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(norm.data(), norm.size() * 2, dNorm, norm.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(gamma.data(), gamma.size() * 2, dGamma, gamma.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));

    const float qjlScale = std::sqrt(static_cast<float>(M_PI) / 2.f) / static_cast<float>(kDim);
    double mseSum = 0.0;
    double mseMax = 0.0;
    std::vector<double> ipRel;
    ipRel.reserve(static_cast<size_t>(kRows) * 8);
    std::vector<float> q(kDim);
    for (int r = 0; r < kRows; ++r) {
        float yhat[kDim];
        float signs[kDim];
        for (int g = 0; g < kDim / 8; ++g) {
            uint32_t word = 0;
            for (int b = 0; b < kBits; ++b) {
                word |= static_cast<uint32_t>(idx[static_cast<size_t>(r * 48 + g * kBits + b)]) << (8 * b);
            }
            const uint32_t signByte = qjl[static_cast<size_t>(r * 16 + g)];
            for (int lane = 0; lane < 8; ++lane) {
                const int code = static_cast<int>((word >> (lane * kBits)) & ((1u << kBits) - 1u));
                yhat[g * 8 + lane] = cent[code];
                signs[g * 8 + lane] = ((signByte >> lane) & 1u) ? 1.f : -1.f;
            }
        }
        const float nrm = Bf16ToFloat(norm[static_cast<size_t>(r)]);
        const float gam = Bf16ToFloat(gamma[static_cast<size_t>(r)]);
        const float rMag = nrm > 0.f ? gam / nrm : 0.f;
        float dir[kDim];
        float dirSq = 0.f;
        for (int i = 0; i < kDim; ++i) {
            float acc = 0.f;
            for (int j = 0; j < kDim; ++j) {
                acc += S[static_cast<size_t>(j) * kDim + i] * signs[j];
            }
            dir[i] = acc;
            dirSq += acc * acc;
        }
        const float dirNorm = std::sqrt(dirSq);
        float recon[kDim];
        for (int i = 0; i < kDim; ++i) {
            recon[i] = yhat[i] + (dirNorm > 0.f ? rMag * dir[i] / dirNorm : 0.f);
        }
        (void)qjlScale;
        float xhat[kDim];
        double num = 0.0;
        double den = 0.0;
        const float *xr = x.data() + static_cast<size_t>(r) * kDim;
        for (int i = 0; i < kDim; ++i) {
            float acc = 0.f;
            for (int k = 0; k < kDim; ++k) {
                acc += H[static_cast<size_t>(k) * kDim + i] * recon[k];
            }
            xhat[i] = nrm * acc;
            const double d = static_cast<double>(xhat[i] - xr[i]);
            num += d * d;
            den += static_cast<double>(xr[i]) * xr[i];
        }
        const double rel = den > 0.0 ? num / den : 0.0;
        mseSum += rel;
        mseMax = std::max(mseMax, rel);
        for (int t = 0; t < 8; ++t) {
            for (int i = 0; i < kDim; ++i) {
                q[static_cast<size_t>(i)] = xr[i] + 0.35f * gauss();
            }
            double ip = 0.0;
            double iph = 0.0;
            double qn = 0.0;
            double xn = 0.0;
            for (int i = 0; i < kDim; ++i) {
                ip += static_cast<double>(q[static_cast<size_t>(i)]) * xr[i];
                iph += static_cast<double>(q[static_cast<size_t>(i)]) * xhat[i];
                qn += static_cast<double>(q[static_cast<size_t>(i)]) * q[static_cast<size_t>(i)];
                xn += static_cast<double>(xr[i]) * xr[i];
            }
            const double cosAbs = std::fabs(ip) / (std::sqrt(qn) * std::sqrt(xn) + 1e-12);
            if (cosAbs > 0.2 && std::fabs(ip) > 1e-3) {
                ipRel.push_back(std::fabs(iph - ip) / std::fabs(ip));
            }
        }
    }
    std::sort(ipRel.begin(), ipRel.end());
    const double mseMean = mseSum / kRows;
    const double ipP95 = ipRel.empty() ? 0.0 : ipRel[static_cast<size_t>(0.95 * (ipRel.size() - 1))];
    std::printf("[recon] rows=%d relMSE mean=%.4f max=%.4f ipRel_p95(|cos|>0.2)=%.4f n=%zu\n", kRows, mseMean, mseMax,
                ipP95, ipRel.size());
    DestroyExact(exact);
    aclrtFree(dKv);
    aclrtFree(dH);
    aclrtFree(dS);
    aclrtFree(dUnit);
    aclrtFree(dY);
    aclrtFree(dR);
    aclrtFree(dP);
    aclrtFree(dIdx);
    aclrtFree(dQjl);
    aclrtFree(dNormFp);
    aclrtFree(dRNorm);
    aclrtFree(dNorm);
    aclrtFree(dGammaFp);
    aclrtFree(dGamma);
    aclrtFree(dWork);
    aclrtFree(dTiling);
    return (mseMean < 0.05 && ipP95 < 0.1) ? 0 : 1;
}

} // namespace

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    Device dev;
    dev.platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    dev.aiv = dev.platform->GetCoreNumAiv();
    dev.aic = dev.platform->GetCoreNumAic();
    std::printf("aic=%u aiv=%u\n", dev.aic, dev.aiv);
    CHECK_ACL(aclrtCreateStream(&dev.stream));
    int failed = 0;
    failed |= RunCase("gqa_decode_b1", 8, 3, dev, true);
    failed |= RunCase("gqa_decode_b64", 64 * 8, 3, dev, true);
    failed |= RunCase("gqa_prefill_t2048", 2048 * 8, 3, dev, true);
    failed |= RunCase("gqa_mse2_t2048", 2048 * 8, 2, dev, true);
    failed |= RunCase("gqa_mse4_t2048", 2048 * 8, 4, dev, true);
    failed |= RunCase("gqa_heads4_t32", 32 * 4, 3, dev, false);
    failed |= RunCase("gqa_heads32_t8", 8 * 32, 3, dev, false);
    failed |= RunRecon(dev);
    aclrtDestroyStream(dev.stream);
    aclrtResetDevice(0);
    aclFinalize();
    std::printf(failed ? "FAILED\n" : "PASSED\n");
    return failed ? 1 : 0;
}
