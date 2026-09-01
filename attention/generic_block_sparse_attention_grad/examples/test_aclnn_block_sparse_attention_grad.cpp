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
 * \file test_aclnn_block_sparse_attention_grad.cpp
 * \brief aclnnGenericBlockSparseAttentionGrad TND example.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad.h"
#include "aclnnop/aclnn_generic_block_sparse_attention_grad_metadata.h"

namespace {

constexpr int32_t kDeviceId = 0;
constexpr int64_t kBatchSize = 1;
constexpr int64_t kNumQHeads = 1;
constexpr int64_t kNumKvHeads = 1;
constexpr int64_t kMaxQSeqlen = 128;
constexpr int64_t kMaxKvSeqlen = 128;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kBlockShapeX = 1;
constexpr int64_t kBlockShapeY = 128;
constexpr int64_t kMetadataHeaderWords = 32;
constexpr int64_t kMetadataTaskWords = 8;
constexpr int64_t kMetadataMaxAicCores = 64;
constexpr int64_t kMetadataWords = kMetadataHeaderWords + kMetadataMaxAicCores * kMetadataTaskWords;

#define CHECK_ACL(call) \
    do { \
        const aclError status = (call); \
        if (status != ACL_SUCCESS) { \
            std::fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(status)); \
            return status; \
        } \
    } while (0)

int64_t ElementCount(const std::vector<int64_t> &shape)
{
    int64_t count = 1;
    for (const int64_t dimension : shape) {
        count *= dimension;
    }
    return count;
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[index] = shape[index + 1] * strides[index + 1];
    }
    return strides;
}

struct TensorResource {
    aclTensor *tensor = nullptr;
    void *deviceAddress = nullptr;

    ~TensorResource()
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
        if (deviceAddress != nullptr) {
            aclrtFree(deviceAddress);
        }
    }
};

struct DeviceBuffer {
    void *address = nullptr;

    ~DeviceBuffer()
    {
        if (address != nullptr) {
            aclrtFree(address);
        }
    }
};

template <typename T>
aclError CreateTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                      TensorResource *resource)
{
    if (resource == nullptr || shape.empty() || hostData.size() != static_cast<size_t>(ElementCount(shape))) {
        return ACL_ERROR_INVALID_PARAM;
    }

    const size_t bytes = hostData.size() * sizeof(T);
    aclError status = aclrtMalloc(&resource->deviceAddress, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) {
        return status;
    }
    status = aclrtMemcpy(resource->deviceAddress, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (status != ACL_SUCCESS) {
        return status;
    }

    const std::vector<int64_t> strides = ContiguousStrides(shape);
    resource->tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                                       shape.data(), shape.size(), resource->deviceAddress);
    return resource->tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

aclError AllocateWorkspace(uint64_t workspaceSize, DeviceBuffer *workspace)
{
    if (workspaceSize == 0) {
        return ACL_SUCCESS;
    }
    return aclrtMalloc(&workspace->address, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

aclError VerifyFinite(const TensorResource &tensor, size_t elementCount, const char *name)
{
    std::vector<op::fp16_t> hostData(elementCount);
    const size_t bytes = hostData.size() * sizeof(op::fp16_t);
    CHECK_ACL(aclrtMemcpy(hostData.data(), bytes, tensor.deviceAddress, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    for (const op::fp16_t &value : hostData) {
        if (!std::isfinite(static_cast<float>(value))) {
            std::fprintf(stderr, "%s contains NaN or Inf\n", name);
            return ACL_ERROR_FAILURE;
        }
    }
    return ACL_SUCCESS;
}

aclError RunCase(aclrtStream stream)
{
    const int64_t kvBlockNum = (kMaxKvSeqlen + kBlockShapeY - 1) / kBlockShapeY;
    const int64_t totalQTokens = kBatchSize * kMaxQSeqlen;
    const int64_t totalKvTokens = kBatchSize * kMaxKvSeqlen;
    const std::vector<int64_t> queryShape = {totalQTokens, kNumQHeads, kHeadDim};
    const std::vector<int64_t> kvShape = {totalKvTokens, kNumKvHeads, kHeadDim};
    const std::vector<int64_t> lseShape = {totalQTokens, kNumQHeads, 1};
    const std::vector<int64_t> indexShape = {kBatchSize, kNumKvHeads, kvBlockNum, kMaxQSeqlen};
    const std::vector<int64_t> countShape = {kBatchSize, kNumKvHeads, kvBlockNum};
    const std::vector<int64_t> cuSeqShape = {kBatchSize + 1};
    const std::vector<int64_t> metadataShape = {kMetadataWords};

    const size_t queryElements = static_cast<size_t>(ElementCount(queryShape));
    const size_t kvElements = static_cast<size_t>(ElementCount(kvShape));
    const double scaleValue = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    const float lseValue = static_cast<float>(kHeadDim * scaleValue + std::log(static_cast<double>(kMaxKvSeqlen)));

    std::vector<op::fp16_t> queryHostData(queryElements, 1.0f);
    std::vector<op::fp16_t> keyHostData(kvElements, 1.0f);
    std::vector<op::fp16_t> valueHostData(kvElements, 1.0f);
    std::vector<op::fp16_t> dOutHostData(queryElements, 0.1f);
    std::vector<op::fp16_t> outHostData(queryElements, 1.0f);
    std::vector<float> lseHostData(static_cast<size_t>(ElementCount(lseShape)), lseValue);
    std::vector<int32_t> sparseBlockIdxHostData(static_cast<size_t>(ElementCount(indexShape)), -1);
    std::vector<int32_t> sparseBlockCountHostData(static_cast<size_t>(ElementCount(countShape)), kMaxQSeqlen);
    for (int64_t qIndex = 0; qIndex < kMaxQSeqlen; ++qIndex) {
        sparseBlockIdxHostData[static_cast<size_t>(qIndex)] = static_cast<int32_t>(qIndex);
    }
    std::vector<int64_t> cuSeqHostData = {0, kMaxQSeqlen};
    std::vector<int64_t> metadataHostData(static_cast<size_t>(kMetadataWords), 0);
    std::vector<op::fp16_t> dQueryHostData(queryElements, 0.0f);
    std::vector<op::fp16_t> dKeyHostData(kvElements, 0.0f);
    std::vector<op::fp16_t> dValueHostData(kvElements, 0.0f);

    TensorResource query;
    TensorResource key;
    TensorResource value;
    TensorResource dOut;
    TensorResource out;
    TensorResource lse;
    TensorResource sparseBlockIdx;
    TensorResource sparseBlockCount;
    TensorResource cuSeqLengthsQ;
    TensorResource cuSeqLengthsKv;
    TensorResource metadata;
    TensorResource dQuery;
    TensorResource dKey;
    TensorResource dValue;

    CHECK_ACL(CreateTensor(queryHostData, queryShape, ACL_FLOAT16, &query));
    CHECK_ACL(CreateTensor(keyHostData, kvShape, ACL_FLOAT16, &key));
    CHECK_ACL(CreateTensor(valueHostData, kvShape, ACL_FLOAT16, &value));
    CHECK_ACL(CreateTensor(dOutHostData, queryShape, ACL_FLOAT16, &dOut));
    CHECK_ACL(CreateTensor(outHostData, queryShape, ACL_FLOAT16, &out));
    CHECK_ACL(CreateTensor(lseHostData, lseShape, ACL_FLOAT, &lse));
    CHECK_ACL(CreateTensor(sparseBlockIdxHostData, indexShape, ACL_INT32, &sparseBlockIdx));
    CHECK_ACL(CreateTensor(sparseBlockCountHostData, countShape, ACL_INT32, &sparseBlockCount));
    CHECK_ACL(CreateTensor(cuSeqHostData, cuSeqShape, ACL_INT64, &cuSeqLengthsQ));
    CHECK_ACL(CreateTensor(cuSeqHostData, cuSeqShape, ACL_INT64, &cuSeqLengthsKv));
    CHECK_ACL(CreateTensor(metadataHostData, metadataShape, ACL_INT64, &metadata));
    CHECK_ACL(CreateTensor(dQueryHostData, queryShape, ACL_FLOAT16, &dQuery));
    CHECK_ACL(CreateTensor(dKeyHostData, kvShape, ACL_FLOAT16, &dKey));
    CHECK_ACL(CreateTensor(dValueHostData, kvShape, ACL_FLOAT16, &dValue));

    int64_t blockShapeValues[] = {kBlockShapeX, kBlockShapeY};
    aclIntArray *blockShape = aclCreateIntArray(blockShapeValues, 2);
    if (blockShape == nullptr) {
        return ACL_ERROR_FAILURE;
    }
    char layoutQ[] = "TND";
    char layoutKv[] = "TND";
    constexpr int64_t kIsPackedGqa = 1;
    constexpr int64_t kMaskType = 1;
    constexpr int64_t kSoftmaxPrecision = 0;
    constexpr int64_t kWinLeft = -1;
    constexpr int64_t kWinRight = -1;

    uint64_t metadataWorkspaceSize = 0;
    aclOpExecutor *metadataExecutor = nullptr;
    aclError status = aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize(
        sparseBlockIdx.tensor, sparseBlockCount.tensor, cuSeqLengthsQ.tensor, cuSeqLengthsKv.tensor, nullptr, nullptr,
        kMaxQSeqlen, kMaxKvSeqlen, kNumQHeads, kNumKvHeads, kHeadDim, blockShape, kIsPackedGqa, layoutQ, layoutKv,
        kMaskType, kSoftmaxPrecision, kWinLeft, kWinRight, metadata.tensor, &metadataWorkspaceSize, &metadataExecutor);
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "metadata phase failed: %d\n", static_cast<int>(status));
        aclDestroyIntArray(blockShape);
        return status;
    }
    DeviceBuffer metadataWorkspace;
    status = AllocateWorkspace(metadataWorkspaceSize, &metadataWorkspace);
    if (status == ACL_SUCCESS) {
        status = aclnnGenericBlockSparseAttentionGradMetadata(metadataWorkspace.address, metadataWorkspaceSize,
                                                              metadataExecutor, stream);
    }
    if (status == ACL_SUCCESS) {
        status = aclrtSynchronizeStream(stream);
    }
    if (status != ACL_SUCCESS) {
        aclDestroyIntArray(blockShape);
        return status;
    }

    uint64_t gradWorkspaceSize = 0;
    aclOpExecutor *gradExecutor = nullptr;
    status = aclnnGenericBlockSparseAttentionGradGetWorkspaceSize(
        query.tensor, key.tensor, value.tensor, dOut.tensor, out.tensor, lse.tensor, sparseBlockIdx.tensor,
        sparseBlockCount.tensor, metadata.tensor, nullptr, cuSeqLengthsQ.tensor, cuSeqLengthsKv.tensor, nullptr,
        nullptr, blockShape, kIsPackedGqa, layoutQ, layoutKv, scaleValue, kMaskType, kSoftmaxPrecision, kWinLeft,
        kWinRight, dQuery.tensor, dKey.tensor, dValue.tensor, &gradWorkspaceSize, &gradExecutor);
    if (status == ACL_SUCCESS) {
        DeviceBuffer gradWorkspace;
        status = AllocateWorkspace(gradWorkspaceSize, &gradWorkspace);
        if (status == ACL_SUCCESS) {
            status =
                aclnnGenericBlockSparseAttentionGrad(gradWorkspace.address, gradWorkspaceSize, gradExecutor, stream);
        }
        if (status == ACL_SUCCESS) {
            status = aclrtSynchronizeStream(stream);
        }
    }
    aclDestroyIntArray(blockShape);
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "gradient phase failed: %d\n", static_cast<int>(status));
        return status;
    }

    CHECK_ACL(VerifyFinite(dQuery, queryElements, "dQuery"));
    CHECK_ACL(VerifyFinite(dKey, kvElements, "dKey"));
    CHECK_ACL(VerifyFinite(dValue, kvElements, "dValue"));
    std::printf("aclnnGenericBlockSparseAttentionGrad passed (metadata workspace=%lu, grad workspace=%lu).\n",
                metadataWorkspaceSize, gradWorkspaceSize);
    return ACL_SUCCESS;
}

} // namespace

int main()
{
    aclError status = aclInit(nullptr);
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "aclInit failed: %d\n", static_cast<int>(status));
        return 1;
    }
    status = aclrtSetDevice(kDeviceId);
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSetDevice failed: %d\n", static_cast<int>(status));
        aclFinalize();
        return 1;
    }

    aclrtStream stream = nullptr;
    status = aclrtCreateStream(&stream);
    if (status != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtCreateStream failed: %d\n", static_cast<int>(status));
    } else {
        status = RunCase(stream);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(kDeviceId);
    aclFinalize();
    return status == ACL_SUCCESS ? 0 : 1;
}
