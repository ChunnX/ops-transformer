/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file test_aclnn_kv_cache_turbo_quant.cpp
 *
 * aclnn accuracy + performance self-test for KvCacheTurboQuant.
 * Accuracy: compares against golden .bin files produced by tests/gen_test_data.py
 * (CPU port of the task golden.py with identical fp32 math).
 * Performance: reports average us/call and the speedup vs the task-provided
 * PyTorch eager baselines (see the task specification, section 3.5).
 *
 * Usage:
 *   test_aclnn_kv_cache_turbo_quant <data_dir> [perf_iters]
 * e.g.
 *   test_aclnn_kv_cache_turbo_quant /data/ma-user/work/ASCEND_DEVELOP/test_data 200
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kv_cache_turbo_quant.h"

#define CHECK_RET(cond, msg)                                         \
    do {                                                             \
        if (!(cond)) {                                               \
            std::cerr << "[FATAL] " << msg << std::endl;             \
            return false;                                            \
        }                                                            \
    } while (0)

namespace {

constexpr int64_t HEAD_DIM = 128;

struct CaseInfo {
    std::string name;
    int64_t numTokens;
    int64_t numKvHeads;
    int64_t mseBits;
    double baselineUs;   // PyTorch eager baseline from the task specification
};

const std::vector<CaseInfo> kCases = {
    {"gqa_decode_b1", 1, 8, 3, 1491.676},
    {"gqa_decode_b64", 64, 8, 3, 1552.784},
    {"gqa_prefill_t2048", 2048, 8, 3, 3855.422},
    {"gqa_mse2_t2048", 2048, 8, 2, 3607.963},
    {"gqa_mse4_t2048", 2048, 8, 4, 5286.502},
};

template <typename T>
bool LoadBin(const std::string &path, std::vector<T> &out)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        std::cerr << "[FATAL] cannot open " << path << std::endl;
        return false;
    }
    const size_t bytes = static_cast<size_t>(ifs.tellg());
    out.resize(bytes / sizeof(T));
    ifs.seekg(0);
    ifs.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(ifs);
}

uint16_t Fp32ToBf16Bits(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;   // round to nearest even
    return static_cast<uint16_t>(u >> 16);
}

float Bf16BitsToFp32(uint16_t v)
{
    uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct CaseResult {
    bool ioOk = false;
    size_t idxMismatch = 0;
    size_t idxTotal = 0;
    size_t qjlMismatch = 0;
    size_t qjlTotal = 0;
    size_t normMismatch = 0;
    size_t normTotal = 0;
    size_t gammaMismatch = 0;
    size_t gammaTotal = 0;
    double avgUs = 0.0;
};

bool RunCase(const std::string &dataDir, const CaseInfo &ci, int perfIters, aclrtStream stream, CaseResult &res)
{
    const std::string dir = dataDir + "/" + ci.name + "/";
    const int64_t totalVec = ci.numTokens * ci.numKvHeads;
    const int64_t idxBytes = HEAD_DIM / 8 * ci.mseBits;
    const int64_t qjlBytes = HEAD_DIM / 8;

    std::vector<uint16_t> kvHost;
    std::vector<float> rotHost, qjlHost;
    std::vector<uint8_t> idxGold, qjlGold;
    std::vector<uint16_t> normGold, gammaGold;
    if (!LoadBin(dir + "kv_vectors.bin", kvHost) || !LoadBin(dir + "rotation_matrix.bin", rotHost) ||
        !LoadBin(dir + "qjl_matrix.bin", qjlHost) || !LoadBin(dir + "quant_idx.bin", idxGold) ||
        !LoadBin(dir + "quant_qjl.bin", qjlGold) || !LoadBin(dir + "quant_norm.bin", normGold) ||
        !LoadBin(dir + "quant_gamma.bin", gammaGold)) {
        return false;
    }

    const size_t kvBytes = kvHost.size() * sizeof(uint16_t);
    const size_t rotBytes = rotHost.size() * sizeof(float);
    const size_t qjlBytesTotal = qjlHost.size() * sizeof(float);
    const size_t idxOutBytes = static_cast<size_t>(totalVec) * idxBytes;
    const size_t qjlOutBytes = static_cast<size_t>(totalVec) * qjlBytes;
    const size_t normOutBytes = static_cast<size_t>(totalVec) * sizeof(uint16_t);

    void *kvDev = nullptr, *rotDev = nullptr, *qjlDev = nullptr;
    void *idxDev = nullptr, *qjlOutDev = nullptr, *normDev = nullptr, *gammaDev = nullptr;
    CHECK_RET(aclrtMalloc(&kvDev, kvBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc kv");
    CHECK_RET(aclrtMalloc(&rotDev, rotBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc rot");
    CHECK_RET(aclrtMalloc(&qjlDev, qjlBytesTotal, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc qjl");
    CHECK_RET(aclrtMalloc(&idxDev, idxOutBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc idx");
    CHECK_RET(aclrtMalloc(&qjlOutDev, qjlOutBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc qjlout");
    CHECK_RET(aclrtMalloc(&normDev, normOutBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc norm");
    CHECK_RET(aclrtMalloc(&gammaDev, normOutBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc gamma");
    CHECK_RET(aclrtMemcpy(kvDev, kvBytes, kvHost.data(), kvBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS, "H2D kv");
    CHECK_RET(aclrtMemcpy(rotDev, rotBytes, rotHost.data(), rotBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
              "H2D rot");
    CHECK_RET(aclrtMemcpy(qjlDev, qjlBytesTotal, qjlHost.data(), qjlBytesTotal, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
              "H2D qjl");

    std::vector<int64_t> kvShape = {ci.numTokens, ci.numKvHeads, HEAD_DIM};
    std::vector<int64_t> matShape = {HEAD_DIM, HEAD_DIM};
    std::vector<int64_t> idxShape = {ci.numTokens, ci.numKvHeads, idxBytes};
    std::vector<int64_t> qjlShape = {ci.numTokens, ci.numKvHeads, qjlBytes};
    std::vector<int64_t> normShape = {ci.numTokens, ci.numKvHeads};

    aclTensor *kvT = aclCreateTensor(kvShape.data(), kvShape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND,
                                     kvShape.data(), kvShape.size(), kvDev);
    aclTensor *rotT = aclCreateTensor(matShape.data(), matShape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                                      matShape.data(), matShape.size(), rotDev);
    aclTensor *qjlT = aclCreateTensor(matShape.data(), matShape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                                      matShape.data(), matShape.size(), qjlDev);
    aclTensor *idxT = aclCreateTensor(idxShape.data(), idxShape.size(), ACL_UINT8, nullptr, 0, ACL_FORMAT_ND,
                                      idxShape.data(), idxShape.size(), idxDev);
    aclTensor *qjlOutT = aclCreateTensor(qjlShape.data(), qjlShape.size(), ACL_UINT8, nullptr, 0, ACL_FORMAT_ND,
                                         qjlShape.data(), qjlShape.size(), qjlOutDev);
    aclTensor *normT = aclCreateTensor(normShape.data(), normShape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND,
                                       normShape.data(), normShape.size(), normDev);
    aclTensor *gammaT = aclCreateTensor(normShape.data(), normShape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND,
                                        normShape.data(), normShape.size(), gammaDev);
    CHECK_RET(kvT && rotT && qjlT && idxT && qjlOutT && normT && gammaT, "create tensor");

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus st = aclnnKvCacheTurboQuantGetWorkspaceSize(kvT, rotT, qjlT, ci.mseBits, idxT, qjlOutT, normT,
                                                            gammaT, &workspaceSize, &executor);
    CHECK_RET(st == ACL_SUCCESS, ci.name + " GetWorkspaceSize ret=" + std::to_string(st));

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, "malloc ws");
    }

    // ---- accuracy run ----
    st = aclnnKvCacheTurboQuant(workspace, workspaceSize, executor, stream);
    CHECK_RET(st == ACL_SUCCESS, ci.name + " aclnn ret=" + std::to_string(st));
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "sync");

    std::vector<uint8_t> idxOut(idxOutBytes), qjlOut(qjlOutBytes);
    std::vector<uint16_t> normOut(totalVec), gammaOut(totalVec);
    CHECK_RET(aclrtMemcpy(idxOut.data(), idxOutBytes, idxDev, idxOutBytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              "D2H idx");
    CHECK_RET(aclrtMemcpy(qjlOut.data(), qjlOutBytes, qjlOutDev, qjlOutBytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
              "D2H qjl");
    CHECK_RET(aclrtMemcpy(normOut.data(), normOutBytes, normDev, normOutBytes, ACL_MEMCPY_DEVICE_TO_HOST) ==
                  ACL_SUCCESS, "D2H norm");
    CHECK_RET(aclrtMemcpy(gammaOut.data(), normOutBytes, gammaDev, normOutBytes, ACL_MEMCPY_DEVICE_TO_HOST) ==
                  ACL_SUCCESS, "D2H gamma");

    res.ioOk = true;
    res.idxTotal = idxOutBytes;
    res.qjlTotal = qjlOutBytes;
    res.normTotal = static_cast<size_t>(totalVec);
    res.gammaTotal = static_cast<size_t>(totalVec);
    for (size_t i = 0; i < idxOutBytes; ++i) {
        if (idxOut[i] != idxGold[i]) {
            ++res.idxMismatch;
        }
    }
    for (size_t i = 0; i < qjlOutBytes; ++i) {
        if (qjlOut[i] != qjlGold[i]) {
            ++res.qjlMismatch;
        }
    }
    for (size_t i = 0; i < res.normTotal; ++i) {
        if (normOut[i] != normGold[i]) {
            ++res.normMismatch;
        }
        if (gammaOut[i] != gammaGold[i]) {
            ++res.gammaMismatch;
        }
    }

    // ---- optional dump of outputs for offline comparison ----
    if (getenv("KVTQ_DUMP") != nullptr) {
        auto dumpU8 = [&](const char *path, const std::vector<uint8_t> &buf) {
            std::ofstream ofs(path, std::ios::binary);
            ofs.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
        };
        auto dumpU16 = [&](const char *path, const std::vector<uint16_t> &buf) {
            std::ofstream ofs(path, std::ios::binary);
            ofs.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size() * 2));
        };
        const std::string dd = dataDir + "/" + ci.name + "/dev_";
        dumpU8((dd + "quant_idx.bin").c_str(), idxOut);
        dumpU8((dd + "quant_qjl.bin").c_str(), qjlOut);
        dumpU16((dd + "quant_norm.bin").c_str(), normOut);
        dumpU16((dd + "quant_gamma.bin").c_str(), gammaOut);
    }

    // ---- performance run ----
    const int warmup = std::min(20, std::max(3, perfIters / 10));
    for (int i = 0; i < warmup; ++i) {
        aclnnKvCacheTurboQuant(workspace, workspaceSize, executor, stream);
    }
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "warmup sync");
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < perfIters; ++i) {
        aclnnKvCacheTurboQuant(workspace, workspaceSize, executor, stream);
    }
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, "perf sync");
    const auto t1 = std::chrono::steady_clock::now();
    res.avgUs = std::chrono::duration<double, std::micro>(t1 - t0).count() / perfIters;

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
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <data_dir> [perf_iters]" << std::endl;
        return -1;
    }
    const std::string dataDir = argv[1];
    const int perfIters = (argc > 2) ? std::atoi(argv[2]) : 200;

    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cerr << "aclInit failed" << std::endl;
        return -1;
    }
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // acceptance thresholds: idx/qjl mismatch ratio <= 1e-5 (case.json err_threshold),
    // bf16 norm/gamma mismatch ratio <= 1e-3 (fp32 accumulation-order noise only)
    int failed = 0;
    std::cout << "==== KvCacheTurboQuant aclnn self-test ====" << std::endl;
    for (const auto &ci : kCases) {
        CaseResult res;
        if (!RunCase(dataDir, ci, perfIters, stream, res)) {
            std::cout << "[FATAL] case " << ci.name << " aborted" << std::endl;
            ++failed;
            continue;
        }
        const double idxRatio = static_cast<double>(res.idxMismatch) / std::max<size_t>(res.idxTotal, 1);
        const double qjlRatio = static_cast<double>(res.qjlMismatch) / std::max<size_t>(res.qjlTotal, 1);
        const double normRatio = static_cast<double>(res.normMismatch) / std::max<size_t>(res.normTotal, 1);
        const double gammaRatio = static_cast<double>(res.gammaMismatch) / std::max<size_t>(res.gammaTotal, 1);
        const bool accOk = (idxRatio <= 1e-5) && (qjlRatio <= 1.1e-5) && (normRatio <= 1e-3) && (gammaRatio <= 1e-3);
        const bool perfOk = res.avgUs * 10.0 <= ci.baselineUs;
        if (!accOk || !perfOk) {
            ++failed;
        }
        std::cout << "[" << ((accOk && perfOk) ? "PASS" : "FAIL") << "] " << ci.name
                  << " idx_mismatch=" << res.idxMismatch << "/" << res.idxTotal
                  << " qjl_mismatch=" << res.qjlMismatch << "/" << res.qjlTotal
                  << " norm_mismatch=" << res.normMismatch << "/" << res.normTotal
                  << " gamma_mismatch=" << res.gammaMismatch << "/" << res.gammaTotal
                  << " | perf=" << res.avgUs << " us, baseline=" << ci.baselineUs
                  << " us, speedup=" << (ci.baselineUs / res.avgUs) << "x" << std::endl;
    }

    std::cout << "==== summary: " << (kCases.size() - failed) << " passed, " << failed << " failed ====" << std::endl;

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return failed == 0 ? 0 : 1;
}
