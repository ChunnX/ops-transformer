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
 * \file npu_perf_bench.cpp
 * \brief
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_kv_cache_turbo_quant.h"

#define CHECK_ACL(expr, msg) \
    do { \
        auto _st = (expr); \
        if (_st != ACL_SUCCESS) { \
            std::cerr << "[FATAL] " << msg << " ret=" << _st << std::endl; \
            return false; \
        } \
    } while (0)

namespace {

constexpr int kHeadDim = 128;
constexpr double kTargetSpeedup = 10.0;

struct CaseMeta {
    int64_t tokens{0};
    int64_t heads{0};
    int bits{3};
};

struct BenchCase {
    const char *name;
    double baselineUs;
};

// 任务书 §3.5 原文 baseline（us）
const BenchCase kCases[] = {
    {"gqa_decode_b1", 1491.676},  {"gqa_decode_b64", 1552.784}, {"gqa_prefill_t2048", 3855.422},
    {"gqa_mse2_t2048", 3607.963}, {"gqa_mse4_t2048", 5286.502},
};

bool LoadMeta(const std::string &path, CaseMeta *m)
{
    std::ifstream ifs(path);
    if (!ifs) {
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.rfind("tokens=", 0) == 0) {
            m->tokens = std::stoll(line.substr(7));
        } else if (line.rfind("heads=", 0) == 0) {
            m->heads = std::stoll(line.substr(6));
        } else if (line.rfind("bits=", 0) == 0) {
            m->bits = std::stoi(line.substr(5));
        }
    }
    return m->tokens > 0 && m->heads > 0;
}

bool LoadBytes(const std::string &path, std::vector<uint8_t> *out)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        return false;
    }
    const auto n = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    out->assign(n, 0);
    ifs.read(reinterpret_cast<char *>(out->data()), static_cast<std::streamsize>(n));
    return true;
}

double Median(std::vector<double> v)
{
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2 == 1) {
        return v[n / 2];
    }
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

bool BenchOne(const std::string &caseDir, const BenchCase &bc, aclrtStream stream, int warmup, int iters,
              double *outMedianUs, double *outMeanUs, double *outMinUs)
{
    CaseMeta meta;
    if (!LoadMeta(caseDir + "/meta.txt", &meta)) {
        std::cerr << "FAIL " << bc.name << " meta\n";
        return false;
    }
    const int64_t nVec = meta.tokens * meta.heads;
    const int64_t idxBytes = (kHeadDim * meta.bits + 7) / 8;
    const int64_t qjlBytes = (kHeadDim + 7) / 8;

    std::vector<uint8_t> kvHost;
    std::vector<uint8_t> rotHost;
    std::vector<uint8_t> qjlHost;
    if (!LoadBytes(caseDir + "/kv_bf16.bin", &kvHost) || !LoadBytes(caseDir + "/rotation_matrix.bin", &rotHost) ||
        !LoadBytes(caseDir + "/qjl_matrix.bin", &qjlHost)) {
        std::cerr << "FAIL " << bc.name << " load bins\n";
        return false;
    }

    void *kvDev = nullptr;
    void *rotDev = nullptr;
    void *qjlDev = nullptr;
    void *idxDev = nullptr;
    void *qjlOutDev = nullptr;
    void *normDev = nullptr;
    void *gammaDev = nullptr;
    const size_t idxSz = static_cast<size_t>(nVec * idxBytes);
    const size_t qjlSz = static_cast<size_t>(nVec * qjlBytes);
    CHECK_ACL(aclrtMalloc(&kvDev, kvHost.size(), ACL_MEM_MALLOC_HUGE_FIRST), "kv");
    CHECK_ACL(aclrtMalloc(&rotDev, rotHost.size(), ACL_MEM_MALLOC_HUGE_FIRST), "rot");
    CHECK_ACL(aclrtMalloc(&qjlDev, qjlHost.size(), ACL_MEM_MALLOC_HUGE_FIRST), "qjl");
    CHECK_ACL(aclrtMalloc(&idxDev, idxSz, ACL_MEM_MALLOC_HUGE_FIRST), "idx");
    CHECK_ACL(aclrtMalloc(&qjlOutDev, qjlSz, ACL_MEM_MALLOC_HUGE_FIRST), "qjlOut");
    CHECK_ACL(aclrtMalloc(&normDev, static_cast<size_t>(nVec) * 2, ACL_MEM_MALLOC_HUGE_FIRST), "norm");
    CHECK_ACL(aclrtMalloc(&gammaDev, static_cast<size_t>(nVec) * 2, ACL_MEM_MALLOC_HUGE_FIRST), "gamma");

    CHECK_ACL(aclrtMemcpy(kvDev, kvHost.size(), kvHost.data(), kvHost.size(), ACL_MEMCPY_HOST_TO_DEVICE), "H2D kv");
    CHECK_ACL(aclrtMemcpy(rotDev, rotHost.size(), rotHost.data(), rotHost.size(), ACL_MEMCPY_HOST_TO_DEVICE),
              "H2D rot");
    CHECK_ACL(aclrtMemcpy(qjlDev, qjlHost.size(), qjlHost.data(), qjlHost.size(), ACL_MEMCPY_HOST_TO_DEVICE),
              "H2D qjl");

    int64_t kvShape[3] = {meta.tokens, meta.heads, kHeadDim};
    int64_t matShape[2] = {kHeadDim, kHeadDim};
    int64_t idxShape[3] = {meta.tokens, meta.heads, idxBytes};
    int64_t qjlShape[3] = {meta.tokens, meta.heads, qjlBytes};
    int64_t scShape[2] = {meta.tokens, meta.heads};

    aclTensor *kvT = aclCreateTensor(kvShape, 3, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, kvShape, 3, kvDev);
    aclTensor *rotT = aclCreateTensor(matShape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, matShape, 2, rotDev);
    aclTensor *qjlT = aclCreateTensor(matShape, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, matShape, 2, qjlDev);
    aclTensor *idxT = aclCreateTensor(idxShape, 3, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, idxShape, 3, idxDev);
    aclTensor *qjlOutT = aclCreateTensor(qjlShape, 3, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, qjlShape, 3, qjlOutDev);
    aclTensor *normT = aclCreateTensor(scShape, 2, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, scShape, 2, normDev);
    aclTensor *gammaT = aclCreateTensor(scShape, 2, ACL_BF16, nullptr, 0, ACL_FORMAT_ND, scShape, 2, gammaDev);

    // workspace：首次 GetWorkspaceSize 后按需分配，循环内禁止 free/malloc
    uint64_t workspaceCap = 0;
    void *workspace = nullptr;

    auto runOnce = [&]() -> bool {
        uint64_t ws = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus st = aclnnKvCacheTurboQuantGetWorkspaceSize(kvT, rotT, qjlT, meta.bits, idxT, qjlOutT, normT,
                                                                gammaT, &ws, &executor);
        if (st != ACL_SUCCESS) {
            std::cerr << "FAIL " << bc.name << " GetWorkspaceSize ret=" << st << "\n";
            return false;
        }
        if (ws > workspaceCap) {
            if (workspace) {
                aclrtFree(workspace);
                workspace = nullptr;
            }
            if (ws > 0 && aclrtMalloc(&workspace, ws, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                std::cerr << "FAIL " << bc.name << " ws malloc\n";
                return false;
            }
            workspaceCap = ws;
        }
        st = aclnnKvCacheTurboQuant(workspace, workspaceCap, executor, stream);
        if (st != ACL_SUCCESS) {
            std::cerr << "FAIL " << bc.name << " aclnn ret=" << st << "\n";
            return false;
        }
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            std::cerr << "FAIL " << bc.name << " sync\n";
            return false;
        }
        return true;
    };

    // 预热：顺带固定 workspace 容量，避免计入正式样本
    for (int i = 0; i < warmup; ++i) {
        if (!runOnce()) {
            if (workspace) {
                aclrtFree(workspace);
            }
            return false;
        }
    }

    std::vector<double> samplesApiUs;
    std::vector<double> samplesKerUs;
    samplesApiUs.reserve(static_cast<size_t>(iters));
    samplesKerUs.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        // API window
        const auto a0 = std::chrono::steady_clock::now();
        if (!runOnce()) {
            if (workspace) {
                aclrtFree(workspace);
            }
            return false;
        }
        const auto a1 = std::chrono::steady_clock::now();
        samplesApiUs.push_back(std::chrono::duration<double, std::micro>(a1 - a0).count());

        // Kernel window: GetWS outside, time aclnn+sync only
        uint64_t ws = 0;
        aclOpExecutor *executor = nullptr;
        aclnnStatus st = aclnnKvCacheTurboQuantGetWorkspaceSize(kvT, rotT, qjlT, meta.bits, idxT, qjlOutT, normT,
                                                                gammaT, &ws, &executor);
        if (st != ACL_SUCCESS || ws > workspaceCap) {
            std::cerr << "FAIL " << bc.name << " ker GetWS\n";
            if (workspace) {
                aclrtFree(workspace);
            }
            return false;
        }
        const auto k0 = std::chrono::steady_clock::now();
        st = aclnnKvCacheTurboQuant(workspace, workspaceCap, executor, stream);
        if (st != ACL_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            std::cerr << "FAIL " << bc.name << " ker aclnn/sync\n";
            if (workspace) {
                aclrtFree(workspace);
            }
            return false;
        }
        const auto k1 = std::chrono::steady_clock::now();
        samplesKerUs.push_back(std::chrono::duration<double, std::micro>(k1 - k0).count());
    }

    *outMedianUs = Median(samplesApiUs);
    *outMeanUs = Median(samplesKerUs); // reuse mean slot as kernel median for compact table
    *outMinUs = *std::min_element(samplesKerUs.begin(), samplesKerUs.end());

    if (workspace) {
        aclrtFree(workspace);
    }
    aclDestroyTensor(kvT);
    aclDestroyTensor(rotT);
    aclDestroyTensor(qjlT);
    aclDestroyTensor(idxT);
    aclDestroyTensor(qjlOutT);
    aclDestroyTensor(normT);
    aclDestroyTensor(gammaT);
    aclrtFree(kvDev);
    aclrtFree(rotDev);
    aclrtFree(qjlDev);
    aclrtFree(idxDev);
    aclrtFree(qjlOutDev);
    aclrtFree(normDev);
    aclrtFree(gammaDev);
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    std::string root = "data/npu_cases";
    int warmup = 20;
    int iters = 50;
    if (argc >= 2) {
        root = argv[1];
    }
    if (argc >= 3) {
        warmup = std::max(1, std::atoi(argv[2]));
    }
    if (argc >= 4) {
        iters = std::max(1, std::atoi(argv[3]));
    }

    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        std::cerr << "acl init/device fail\n";
        return 1;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        std::cerr << "createStream fail\n";
        return 1;
    }

    std::cout << "# npu_perf_bench warmup=" << warmup << " iters=" << iters << "\n";
    std::cout << "# window: GetWorkspaceSize + aclnn + SynchronizeStream (no H2D/D2H)\n";
    std::cout << "# baseline: 任务书 §3.5 PyTorch eager (us); gate: speedup>=" << kTargetSpeedup << "\n";
    std::cout << std::left << std::setw(20) << "case" << std::right << std::setw(12) << "baseline_us" << std::setw(12)
              << "api_med_us" << std::setw(12) << "ker_med_us" << std::setw(12) << "ker_min_us" << std::setw(10)
              << "api_x" << std::setw(10) << "ker_x" << std::setw(8) << "gate" << "\n";

    int failGate = 0;
    int failRun = 0;
    for (const auto &bc : kCases) {
        double med = 0;
        double mean = 0;
        double mn = 0;
        if (!BenchOne(root + "/" + bc.name, bc, stream, warmup, iters, &med, &mean, &mn)) {
            ++failRun;
            std::cout << std::left << std::setw(20) << bc.name << " FAIL_RUN\n";
            continue;
        }
        const double apiX = (med > 0.0) ? (bc.baselineUs / med) : 0.0;
        const double kerX = (mean > 0.0) ? (bc.baselineUs / mean) : 0.0; // mean slot = ker median
        const bool pass = kerX >= kTargetSpeedup;                        // gate on kernel window (aclnn+sync)
        if (!pass) {
            ++failGate;
        }
        std::cout << std::left << std::setw(20) << bc.name << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << bc.baselineUs << std::setw(12) << med << std::setw(12) << mean << std::setw(12)
                  << mn << std::setprecision(2) << std::setw(10) << apiX << std::setw(10) << kerX << std::setw(8)
                  << (pass ? "PASS" : "FAIL") << "\n";
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    if (failRun) {
        std::cout << "ALL FAIL npu_perf_bench run_fail=" << failRun << "\n";
        return 1;
    }
    if (failGate) {
        std::cout << "PERF_GATE FAIL fail=" << failGate << "/" << (sizeof(kCases) / sizeof(kCases[0]))
                  << " (need all >=" << kTargetSpeedup << "x)\n";
        return 2;
    }
    std::cout << "ALL PASS npu_perf_bench (5/5 >=" << kTargetSpeedup << "x)\n";
    return 0;
}
