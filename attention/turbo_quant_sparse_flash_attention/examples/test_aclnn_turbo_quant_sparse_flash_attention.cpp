/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_turbo_quant_sparse_flash_attention.h"

namespace {
void Check(int status, const char *operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes)
        : size(std::max<size_t>(bytes, 1))
    {
        Check(aclrtMalloc(&data, size, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~DeviceBuffer()
    {
        aclrtFree(data);
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    void *data = nullptr;
    size_t size;
};

class Tensor {
public:
    template <typename T>
    Tensor(const std::vector<T> &host, const std::vector<int64_t> &shape, aclDataType dtype,
           std::vector<int64_t> strides = {}, int64_t offset = 0)
        : storage(host.size() * sizeof(T))
    {
        if (!host.empty()) {
            Check(aclrtMemcpy(storage.data, storage.size, host.data(), host.size() * sizeof(T),
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy H2D");
        }
        if (strides.empty()) {
            strides.resize(shape.size(), 1);
            for (size_t axis = shape.size(); axis > 1; --axis) {
                strides[axis - 2] = strides[axis - 1] * shape[axis - 1];
            }
        }
        const int64_t storageElements = static_cast<int64_t>(host.size());
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), offset, ACL_FORMAT_ND,
                                 &storageElements, 1, storage.data);
        if (tensor == nullptr) {
            throw std::runtime_error("aclCreateTensor returned nullptr");
        }
    }
    ~Tensor()
    {
        aclDestroyTensor(tensor);
    }
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    DeviceBuffer storage;
    aclTensor *tensor = nullptr;
};

float Decode16(uint16_t bits, aclDataType dtype)
{
    if (dtype == ACL_BF16) {
        const uint32_t word = static_cast<uint32_t>(bits) << 16;
        float result;
        std::memcpy(&result, &word, sizeof(result));
        return result;
    }
    const int exponent = (bits >> 10) & 31;
    const int fraction = bits & 1023;
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24) :
                                            std::ldexp(1.0F + fraction / 1024.0F, exponent - 15);
    return bits & 0x8000 ? -magnitude : magnitude;
}

void RunCase(aclrtStream stream, aclDataType dtype, bool bnbd, int32_t queryLength, int32_t kvLength)
{
    constexpr int64_t blockSize = 16;
    constexpr int64_t pageBytes = blockSize * 386;
    constexpr int64_t pageStride = pageBytes + 257;
    constexpr int64_t offset = 131;
    std::vector<int8_t> cache(offset + pageStride + pageBytes + 64, 113);
    // Select physical page 1, token 0. All latent nibbles are code 8, rope is zero, scale is fp16 1.0.
    auto *slot = cache.data() + offset + pageStride;
    std::fill(slot, slot + 256, static_cast<int8_t>(0x88));
    std::fill(slot + 256, slot + 386, 0);
    slot[385] = 0x3c;
    const std::vector<int64_t> shape =
        bnbd ? std::vector<int64_t>{2, 1, blockSize, 386} : std::vector<int64_t>{2, blockSize, 1, 386};
    const std::vector<int64_t> strides =
        bnbd ? std::vector<int64_t>{pageStride, pageBytes, 386, 1} : std::vector<int64_t>{pageStride, 386, 386, 1};
    Tensor q(std::vector<uint16_t>(queryLength * 8 * 576, 0), {queryLength, 8, 576}, dtype);
    Tensor kv(cache, shape, ACL_INT8, strides, offset);
    Tensor indices(std::vector<int32_t>(queryLength, 0), {queryLength, 1, 1}, ACL_INT32);
    Tensor table(std::vector<int32_t>{1}, {1, 1}, ACL_INT32);
    Tensor queryLengths(std::vector<int32_t>{queryLength}, {1}, ACL_INT32);
    Tensor kvLengths(std::vector<int32_t>{kvLength}, {1}, ACL_INT32);
    Tensor output(std::vector<uint16_t>(queryLength * 8 * 512, 0xffff), {queryLength, 8, 512}, dtype);
    Tensor maximum(std::vector<float>(queryLength * 8, 1.0F), {1, queryLength, 8}, ACL_FLOAT);
    Tensor total(std::vector<float>(queryLength * 8, 1.0F), {1, queryLength, 8}, ACL_FLOAT);
    char queryLayout[] = "TND";
    char bsndLayout[] = "PA_BSND";
    char bnbdLayout[] = "PA_BNBD";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    Check(aclnnTurboQuantSparseFlashAttentionGetWorkspaceSize(
              q.tensor, kv.tensor, kv.tensor, indices.tensor, nullptr, nullptr, table.tensor, queryLengths.tensor,
              kvLengths.tensor, 1.0, 3, 3, 1, queryLayout, bnbd ? bnbdLayout : bsndLayout, 3, INT64_MAX, INT64_MAX, 2,
              1, 128, 64, true, output.tensor, maximum.tensor, total.tensor, &workspaceSize, &executor),
          "aclnnTurboQuantSparseFlashAttentionGetWorkspaceSize");
    DeviceBuffer workspace(workspaceSize);
    Check(aclnnTurboQuantSparseFlashAttention(workspaceSize == 0 ? nullptr : workspace.data, workspaceSize, executor,
                                              stream),
          "aclnnTurboQuantSparseFlashAttention");
    Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    if (queryLength != 0) {
        std::vector<uint16_t> actual(queryLength * 8 * 512);
        Check(aclrtMemcpy(actual.data(), actual.size() * sizeof(uint16_t), output.storage.data,
                          actual.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST),
              "output D2H");
        const float expected = kvLength == 0 ? 0.0F : 0.00547294F;
        for (auto element : actual) {
            if (std::abs(Decode16(element, dtype) - expected) > 2.0e-5F) {
                throw std::runtime_error("attention output mismatch");
            }
        }
        std::vector<float> maxHost(8), sumHost(8);
        Check(aclrtMemcpy(maxHost.data(), 32, maximum.storage.data, 32, ACL_MEMCPY_DEVICE_TO_HOST), "max D2H");
        Check(aclrtMemcpy(sumHost.data(), 32, total.storage.data, 32, ACL_MEMCPY_DEVICE_TO_HOST), "sum D2H");
        for (size_t row = 0; row < maxHost.size(); ++row) {
            if (maxHost[row] != (kvLength == 0 ? -2.0e38F : 0.0F) || sumHost[row] != (kvLength == 0 ? 0.0F : 1.0F)) {
                throw std::runtime_error("softmax output mismatch");
            }
        }
    }
    std::vector<int8_t> cacheAfter(cache.size());
    Check(aclrtMemcpy(cacheAfter.data(), cacheAfter.size(), kv.storage.data, cacheAfter.size(),
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "cache D2H");
    if (cacheAfter != cache) {
        throw std::runtime_error("input cache or padding modified");
    }
}
} // namespace

int main()
{
    aclrtStream stream = nullptr;
    bool initialized = false;
    bool deviceSet = false;
    int result = 0;
    try {
        Check(aclInit(nullptr), "aclInit");
        initialized = true;
        // Logical device 0: select the physical device with ASCEND_RT_VISIBLE_DEVICES before launching.
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        deviceSet = true;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        for (auto dtype : {ACL_FLOAT16, ACL_BF16}) {
            for (bool bnbd : {false, true}) {
                for (int32_t queryLength : {0, 1}) {
                    for (int32_t kvLength : {0, 1}) {
                        RunCase(stream, dtype, bnbd, queryLength, kvLength);
                    }
                }
            }
        }
        std::cout << "PASS: 16 ACLNN cases (FP16/BF16, BSND/BNBD, empty Q/KV, strided cache with offset)\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        aclrtResetDevice(0);
    }
    if (initialized) {
        aclFinalize();
    }
    return result;
}
