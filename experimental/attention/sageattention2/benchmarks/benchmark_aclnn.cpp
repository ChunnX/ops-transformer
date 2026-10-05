/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Standalone ACL runtime benchmark: no torch, Python tensor or Python launch.
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_infer_attention_score_v5.h"
#include "aclnn_sage_attention2.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
static void check(int64_t status, const char *what)
{
    if (status) {
        throw std::runtime_error(std::string(what) + ": " + std::to_string(status));
    }
}
struct Runtime {
    aclrtStream stream = nullptr;
    Runtime()
    {
        check(aclInit(nullptr), "aclInit");
        check(aclrtSetDevice(0), "aclrtSetDevice");
        check(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
    ~Runtime()
    {
        aclrtSynchronizeStream(stream);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }
};
struct Tensor {
    aclTensor *handle = nullptr;
    void *data = nullptr;
    size_t bytes = 0;
    Tensor(std::vector<int64_t> dims, aclDataType type, const void *host = nullptr)
    {
        std::vector<int64_t> strides(dims.size());
        int64_t count = 1;
        for (int i = int(dims.size()) - 1; i >= 0; --i) {
            strides[i] = count;
            count *= dims[i];
        }
        bytes = count * (type == ACL_BOOL ? 1 : 2);
        check(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "tensor malloc");
        if (host) {
            check(aclrtMemcpy(data, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "input copy");
        }
        handle = aclCreateTensor(dims.data(), dims.size(), type, strides.data(), 0, ACL_FORMAT_ND, dims.data(),
                                 dims.size(), data);
        if (!handle) {
            throw std::runtime_error("aclCreateTensor failed");
        }
    }
    Tensor(const Tensor &) = delete;
    ~Tensor()
    {
        if (handle) {
            aclDestroyTensor(handle);
        }
        if (data) {
            aclrtFree(data);
        }
    }
    std::vector<uint16_t> read() const
    {
        std::vector<uint16_t> host(bytes / 2);
        check(aclrtMemcpy(host.data(), bytes, data, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "output copy");
        return host;
    }
};
struct Workspace {
    void *data = nullptr;
    uint64_t capacity = 0;
    void reserve(uint64_t size)
    {
        if (size > capacity) {
            if (data) {
                check(aclrtFree(data), "workspace free");
            }
            check(aclrtMalloc(&data, size, ACL_MEM_MALLOC_HUGE_FIRST), "workspace malloc");
            capacity = size;
        }
    }
    ~Workspace()
    {
        if (data) {
            aclrtFree(data);
        }
    }
};
static uint16_t encode(float x, bool bf16)
{
    if (!bf16) {
        return aclFloatToFloat16(x);
    }
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    bits += 0x7fff + ((bits >> 16) & 1);
    return uint16_t(bits >> 16);
}
static float decode(uint16_t x, bool bf16)
{
    if (!bf16) {
        return aclFloat16ToFloat(x);
    }
    uint32_t bits = uint32_t(x) << 16;
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}
int main(int argc, char **argv)
try {
    if (argc != 9) {
        throw std::runtime_error("usage: benchmark_aclnn SEQ BATCH KV_HEADS BF16 CAUSAL WARMUP_CALLS REPEATS OUTPUT_JSON");
    }
    int64_t seq = std::stoll(argv[1]), batch = std::stoll(argv[2]), hk = std::stoll(argv[3]);
    bool bf16 = std::stoi(argv[4]) != 0, causal = std::stoi(argv[5]) != 0;
    int warmupCalls = std::stoi(argv[6]);
    int repeats = std::stoi(argv[7]);
    const char *headsEnv = std::getenv("SAGE_BENCH_HEADS");
    const char *dimEnv = std::getenv("SAGE_BENCH_DIM");
    int64_t heads = headsEnv ? std::stoll(headsEnv) : 32;
    int64_t dim = dimEnv ? std::stoll(dimEnv) : 128;
    if (heads <= 0 || dim <= 0 || dim > 128 || seq <= 0 || batch <= 0 || hk <= 0 || heads % hk || warmupCalls < 1 ||
        repeats < 1) {
        throw std::runtime_error("invalid arguments");
    }
    Runtime runtime;
    auto fiaPrepare = &aclnnFusedInferAttentionScoreV5GetWorkspaceSize;
    auto fiaRun = &aclnnFusedInferAttentionScoreV5;
    const char *fiaLibrary = std::getenv("FIA_OPAPI_LIBRARY");
    if (fiaLibrary) {
        void *library = dlopen(fiaLibrary, RTLD_NOW | RTLD_LOCAL);
        if (!library) {
            throw std::runtime_error(dlerror());
        }
        fiaPrepare =
            reinterpret_cast<decltype(fiaPrepare)>(dlsym(library, "aclnnFusedInferAttentionScoreV5GetWorkspaceSize"));
        fiaRun = reinterpret_cast<decltype(fiaRun)>(dlsym(library, "aclnnFusedInferAttentionScoreV5"));
        if (!fiaPrepare || !fiaRun) {
            throw std::runtime_error("FIA V5 symbols missing from selected library");
        }
    }
    Dl_info fiaInfo{};
    if (!dladdr(reinterpret_cast<void *>(fiaRun), &fiaInfo) || !fiaInfo.dli_fname) {
        throw std::runtime_error("Cannot identify FIA API provider");
    }
    std::mt19937 rng(71);
    std::normal_distribution<float> normal(0, 1);
    auto random = [&](size_t count) {
        std::vector<uint16_t> h(count);
        for (auto &x : h) {
            x = encode(normal(rng), bf16);
        }
        return h;
    };
    aclDataType dtype = bf16 ? ACL_BF16 : ACL_FLOAT16;
    auto qh = random(batch * heads * seq * dim);
    Tensor q({batch, heads, seq, dim}, dtype, qh.data());
    qh.clear();
    qh.shrink_to_fit();
    auto kh = random(batch * hk * seq * dim);
    Tensor k({batch, hk, seq, dim}, dtype, kh.data());
    kh.clear();
    kh.shrink_to_fit();
    auto vh = random(batch * hk * seq * dim);
    Tensor v({batch, hk, seq, dim}, dtype, vh.data());
    vh.clear();
    vh.shrink_to_fit();
    Tensor out({batch, heads, seq, dim}, dtype), fiaOut({batch, heads, seq, dim}, dtype);
    std::unique_ptr<Tensor> mask;
    if (causal) {
        std::vector<uint8_t> h(size_t{2048} * 2048);
        for (int i = 0; i < 2048; ++i) {
            for (int j = i + 1; j < 2048; ++j) {
                h[i * 2048 + j] = 1;
            }
        }
        mask = std::make_unique<Tensor>(std::vector<int64_t>{2048, 2048}, ACL_BOOL, h.data());
    }
    const aclTensor *keyArray[] = {k.handle};
    const aclTensor *valueArray[] = {v.handle};
    auto keyList = aclCreateTensorList(keyArray, 1);
    auto valueList = aclCreateTensorList(valueArray, 1);
    if (!keyList || !valueList) {
        throw std::runtime_error("aclCreateTensorList failed");
    }
    Workspace sageWs, fiaWs;
    auto prepare = [&](bool sage, uint64_t &bytes, aclOpExecutor *&executor) {
        if (sage) {
            check(aclnnSageAttention2GetWorkspaceSize(q.handle, k.handle, v.handle, "HND", causal, "per_token",
                                                      1 / std::sqrt(static_cast<double>(dim)), "fp32+fp32", true,
                                                      false, out.handle, nullptr, &bytes, &executor),
                  "Sage GetWorkspaceSize");
        } else {
            char layout[] = "BNSD";
            check(fiaPrepare(q.handle, keyList, valueList, nullptr, mask ? mask->handle : nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                             nullptr, nullptr, nullptr, nullptr, heads, 1 / std::sqrt(static_cast<double>(dim)),
                             2147483647, 2147483647, layout, hk, causal ? 2 : 0, 0, 0, 0, false, 0, 0, 0, 0,
                             fiaOut.handle, nullptr, &bytes, &executor),
                  "FIA V5 GetWorkspaceSize");
        }
    };
    float sageTotalMs = 0.0f, fiaTotalMs = 0.0f;
    auto measure = [&](bool sage) {
        Workspace &ws = sage ? sageWs : fiaWs;
        auto run = [&]() {
            uint64_t bytes = 0;
            aclOpExecutor *executor = nullptr;
            prepare(sage, bytes, executor);
            ws.reserve(bytes);
            check(sage ? aclnnSageAttention2(ws.data, bytes, executor, runtime.stream) :
                         fiaRun(ws.data, bytes, executor, runtime.stream),
                  "operator run");
        };
        // Match test_perf.py: fixed warmup calls, followed by one synchronization.
        for (int i = 0; i < warmupCalls; ++i) {
            run();
        }
        check(aclrtSynchronizeStream(runtime.stream), "warmup sync");
        aclrtEvent begin = nullptr, end = nullptr;
        check(aclrtCreateEvent(&begin), "create event");
        check(aclrtCreateEvent(&end), "create event");
        check(aclrtRecordEvent(begin, runtime.stream), "record begin");
        // Match the task's batched timer. Call the preparation API on each iteration;
        // both API phases run inside the batch, without per-call synchronization.
        for (int i = 0; i < repeats; ++i) {
            run();
        }
        check(aclrtRecordEvent(end, runtime.stream), "record end");
        check(aclrtSynchronizeStream(runtime.stream), "batch sync");
        float totalMs = 0.0f;
        check(aclrtEventElapsedTime(&totalMs, begin, end), "elapsed time");
        check(aclrtDestroyEvent(begin), "destroy event");
        check(aclrtDestroyEvent(end), "destroy event");
        (sage ? sageTotalMs : fiaTotalMs) = totalMs;
        std::cout << "MEASURED " << (sage ? "sage" : "fia") << ": " << totalMs / repeats
                  << " ms; cooldown 10s" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));
        return totalMs / repeats;
    };
    float fiaMs = measure(false);
    float sageMs = measure(true);
    auto result = out.read(), truth = fiaOut.read();
    double maxAbs = 0, sumAbs = 0, sumSq = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        double e = double(decode(result[i], bf16)) - decode(truth[i], bf16);
        if (!std::isfinite(e)) {
            throw std::runtime_error("non-finite output");
        }
        maxAbs = std::max(maxAbs, std::abs(e));
        sumAbs += std::abs(e);
        sumSq += e * e;
    }
    std::ostringstream report;
    report << std::setprecision(17);
    report << "{\n  \"path\":\"standalone C++ ACL runtime\",\n  \"sage_api\":\"aclnnSageAttention2\",\n  "
              "\"fia_api\":\"aclnnFusedInferAttentionScoreV5\",\n  \"timing\":\""
           << "ACL device Event batch total / repeats; both API phases inside batch; workspace/output reused"
           << "\",\n  \"status\":\"performance development, full "
              "accuracy acceptance separate\",\n  \"shape\":["
           << batch << "," << heads << "," << seq << "," << dim
           << "],\n  \"profile_only\":" << "false"
           << ",\n  \"operator_stages\":[\"Fused\"],\n  \"order\":\"fia-first\",\n  "
              "\"pv_quantization\":\"mxfp8_e4m3_e8m0\",\n  \"pv_accumulation\":\"fp32+fp32\""
           << ",\n  \"fia_api_library\":\"" << fiaInfo.dli_fname << "\",\n  \"kv_heads\":" << hk
           << ",\n  \"bf16\":" << (bf16 ? "true" : "false") << ",\n  \"causal\":" << (causal ? "true" : "false")
           << ",\n  \"warmup_calls\":" << warmupCalls << ",\n  \"repeats\":" << repeats;
    {
        report << ",\n  \"sage_ms\":" << sageMs << ",\n  \"fia_ms\":" << fiaMs << ",\n  \"speedup\":" << fiaMs / sageMs;
        report << ",\n  \"sage_batch_total_ms\":" << sageTotalMs
               << ",\n  \"fia_batch_total_ms\":" << fiaTotalMs;
    }
    report << ",\n  \"max_abs_vs_fia\":" << maxAbs
           << ",\n  \"mean_abs_vs_fia\":" << sumAbs / static_cast<double>(result.size())
           << ",\n  \"rmse_vs_fia\":" << std::sqrt(sumSq / static_cast<double>(result.size())) << "\n}\n";
    std::ofstream file(argv[8]);
    if (!file) {
        throw std::runtime_error("cannot open output JSON");
    }
    file << report.str();
    std::cout << report.str();
    aclDestroyTensorList(keyList);
    aclDestroyTensorList(valueList);
    // aclDestroyTensorList also destroys its member tensor descriptors.
    // Tensor still owns the device allocation, but must not free the descriptor twice.
    k.handle = nullptr;
    v.handle = nullptr;
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
}
