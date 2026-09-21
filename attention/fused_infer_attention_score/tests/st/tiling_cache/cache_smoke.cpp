// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// Device regression: build/run instructions are in README.md.
#include <acl/acl.h>
#include <opdev/op_executor.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "aclnn_fused_infer_attention_score.h"
#include "aclnn_fused_infer_attention_score_v2.h"
#include "aclnn_fused_infer_attention_score_v3.h"
#include "aclnn_fused_infer_attention_score_v4.h"

void Check(int status, const char *operation)
{
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
}

struct Inputs {
    std::vector<void *> allocations;
    aclTensor *query = nullptr;
    aclTensorList *key = nullptr;
    aclTensorList *value = nullptr;
    aclTensor *output = nullptr;
    aclTensor *lse = nullptr;
    void *outputData = nullptr;

    aclTensor *MakeTensor(const std::vector<int64_t> &shape, const std::vector<uint16_t> &data,
                          void **address = nullptr)
    {
        void *device = nullptr;
        // Empty LSE uses a valid allocation, but has zero logical elements.
        const size_t bytes = data.empty() ? sizeof(uint16_t) : data.size() * sizeof(uint16_t);
        Check(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
        allocations.push_back(device);
        if (!data.empty()) {
            Check(aclrtMemcpy(device, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
        }
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size(); i > 1; --i) {
            strides[i - 2] = strides[i - 1] * shape[i - 1];
        }
        auto *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, strides.data(), 0,
                                       ACL_FORMAT_ND, shape.data(), shape.size(), device);
        if (tensor == nullptr) { throw std::runtime_error("create tensor"); }
        if (address != nullptr) { *address = device; }
        return tensor;
    }

    explicit Inputs(bool second)
    {
        query = MakeTensor({1, 1, 1, 128}, std::vector<uint16_t>(128, 0));
        auto *k = MakeTensor({1, 1, 4, 128}, std::vector<uint16_t>(512, 0));
        // V positions are [0,0,2,2] or [4,4,6,6], represented as IEEE fp16.
        std::vector<uint16_t> values(512, second ? 0x4400 : 0x0000);
        for (size_t i = 256; i < values.size(); ++i) { values[i] = second ? 0x4600 : 0x4000; }
        auto *v = MakeTensor({1, 1, 4, 128}, values);
        const aclTensor *keys[] = {k};
        const aclTensor *valuesList[] = {v};
        key = aclCreateTensorList(keys, 1);
        value = aclCreateTensorList(valuesList, 1);
        if (key == nullptr || value == nullptr) { throw std::runtime_error("create tensor list"); }
        output = MakeTensor({1, 1, 1, 128}, std::vector<uint16_t>(128, 0xffff), &outputData);
        // The hand-written Inner API expects an LSE descriptor even when LSE is disabled.
        const int64_t zero = 0, stride = 1;
        lse = aclCreateTensor(&zero, 1, ACL_FLOAT, &stride, 0, ACL_FORMAT_ND, &zero, 1, outputData);
        if (lse == nullptr) { throw std::runtime_error("create LSE descriptor"); }
    }

    ~Inputs()
    {
        aclDestroyTensor(query);
        // aclDestroyTensorList also destroys its tensor descriptors.
        aclDestroyTensorList(key);
        aclDestroyTensorList(value);
        aclDestroyTensor(output);
        aclDestroyTensor(lse);
        for (void *address : allocations) { aclrtFree(address); }
    }
};

aclnnStatus GetWorkspace(int version, Inputs &in, const aclIntArray *qLen, const aclIntArray *kvLen,
                        double scale, uint64_t *size, aclOpExecutor **executor)
{
    char layout[] = "BNSD";
    switch (version) {
        case 1:
            return aclnnFusedInferAttentionScoreGetWorkspaceSize(
                in.query, in.key, in.value, nullptr, nullptr, qLen,
                kvLen, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, 1,
                scale, INT32_MAX, INT32_MAX, layout, 1, 0,
                0, 0, 0, false, in.output, in.lse,
                size, executor);
        case 2:
            return aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
                in.query, in.key, in.value, nullptr, nullptr, qLen,
                kvLen, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                1, scale, INT32_MAX, INT32_MAX, layout, 1,
                0, 0, 0, 0, false, 0,
                0, in.output, in.lse, size, executor);
        case 3:
            return aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
                in.query, in.key, in.value, nullptr, nullptr, qLen,
                kvLen, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, 1, scale, INT32_MAX,
                INT32_MAX, layout, 1, 0, 0, 0,
                0, false, 0, 0, in.output, in.lse,
                size, executor);
        case 4:
            return aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
                in.query, in.key, in.value, nullptr, nullptr, qLen,
                kvLen, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, 1,
                scale, INT32_MAX, INT32_MAX, layout, 1, 0,
                0, 0, 0, false, 0, 0,
                0, in.output, in.lse, size, executor);
        default: throw std::runtime_error("version must be 1..4");
    }
}

aclnnStatus Run(int version, void *workspace, uint64_t size, aclOpExecutor *executor, aclrtStream stream)
{
    switch (version) {
        case 1: return aclnnFusedInferAttentionScore(workspace, size, executor, stream);
        case 2: return aclnnFusedInferAttentionScoreV2(workspace, size, executor, stream);
        case 3: return aclnnFusedInferAttentionScoreV3(workspace, size, executor, stream);
        case 4: return aclnnFusedInferAttentionScoreV4(workspace, size, executor, stream);
        default: throw std::runtime_error("invalid version");
    }
}

struct Prepared {
    uint64_t size = 0;
    aclOpExecutor *executor = nullptr;
};

Prepared Prepare(int version, Inputs &in, int64_t length, double scale)
{
    const int64_t one = 1;
    auto *qLen = aclCreateIntArray(&one, 1);
    auto *kvLen = aclCreateIntArray(&length, 1);
    if (qLen == nullptr || kvLen == nullptr) { throw std::runtime_error("create lengths"); }
    Prepared prepared;
    auto status = GetWorkspace(version, in, qLen, kvLen, scale, &prepared.size, &prepared.executor);
    aclDestroyIntArray(qLen);
    aclDestroyIntArray(kvLen);
    Check(status, "GWS");
    if (GetOpExecCacheFromExecutor(prepared.executor) != nullptr) {
        throw std::runtime_error("executor cache must be disabled to isolate tiling cache");
    }
    return prepared;
}

void Execute(int version, Inputs &in, const Prepared &prepared, uint16_t expectedValue, aclrtStream stream)
{
    const auto size = prepared.size;
    void *workspace = nullptr;
    if (size != 0) { Check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST), "workspace"); }
    auto status = Run(version, workspace, size, prepared.executor, stream);
    Check(aclrtSynchronizeStream(stream), "synchronize");
    if (workspace != nullptr) { aclrtFree(workspace); }
    Check(status, "Run");
    std::vector<uint16_t> result(128);
    Check(aclrtMemcpy(result.data(), result.size() * 2, in.outputData, result.size() * 2,
                      ACL_MEMCPY_DEVICE_TO_HOST), "read output");
    for (auto value : result) {
        if (value != expectedValue) { throw std::runtime_error("incorrect output: stale address or sequence length"); }
    }
}

void Pair(int version, Inputs &first, Inputs &second, int64_t length, double scale,
          uint16_t firstValue, uint16_t secondValue, aclrtStream stream)
{
    std::cout << "BEGIN pair length=" << length << " scale=" << scale << std::endl;
    auto a = Prepare(version, first, length, scale);
    auto b = Prepare(version, second, length, scale);
    std::cout << "BOTH GWS DONE BEFORE RUN" << std::endl;
    Execute(version, first, a, firstValue, stream);
    Execute(version, second, b, secondValue, stream);
    std::cout << "PASS pair" << std::endl;
}

int main(int argc, char **argv)
{
    try {
        const int version = argc > 1 ? std::stoi(argv[1]) : 4;
        if (version < 1 || version > 4) { throw std::runtime_error("version must be 1..4"); }
        if (std::getenv("DISABLE_L2_CACHE") == nullptr) {
            throw std::runtime_error("set DISABLE_L2_CACHE=1");
        }
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "set device");
        aclrtStream stream = nullptr;
        Check(aclrtCreateStream(&stream), "create stream");
        {
            Inputs first(false), second(true);
            Pair(version, first, second, 4, 1.0, 0x3c00, 0x4500, stream); // mean 1 / 5
            Pair(version, first, second, 2, 1.0, 0x0000, 0x4400, stream); // changed host value
            Pair(version, first, second, 4, 0.5, 0x3c00, 0x4500, stream); // changed attribute
        }
        Check(aclrtDestroyStream(stream), "destroy stream");
        Check(aclrtResetDevice(0), "reset device");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
