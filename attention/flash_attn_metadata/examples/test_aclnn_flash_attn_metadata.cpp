/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attn_metadata.h"
#include "../op_kernel_aicpu/flash_attn_metadata.h"
#include "securec.h"

using namespace std;
using namespace optiling;

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
    // Fixed writing method, AscendCL initialization.
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

static void DumpMeta(void* data) {
    
    SplitCoreMode splitCoreMode = static_cast<SplitCoreMode>(((uint32_t*)data)[0]);
    printf("splitCoreMode:%u\n", splitCoreMode);
    if (splitCoreMode == SplitCoreMode::SPLIT_S1OUT_CUBE) {
        optiling::detail::S1outFaMetaData* metaDataPtr = (optiling::detail::S1outFaMetaData*)(data);
        for (uint32_t i = 0; i < AIC_CORE_NUM; i++){
            printf("coreId:%u enable:%d M_stride:%d M_num:%d firstFullLoadS1OuterIdx:%d\n",
                    i, metaDataPtr->s1outMetadata[i][S1OUT_CORE_ENABLE_INDEX], metaDataPtr->s1outMetadata[i][S1OUT_M_STRIDE_INDEX],
                    metaDataPtr->s1outMetadata[i][S1OUT_M_NUM_INDEX], metaDataPtr->s1outMetadata[i][S1OUT_FIRST_FULL_LOAD_S1_OUTET_IDX_INDEX]);
        }
    }
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to request device side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { 
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); 
        return ret;
    }
    // Call aclrtMemcpy to copy host side data to device side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
        return ret;
    }

    // Calculate the strides of continuous tensors.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor interface to create aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Fixed writing method)  device/stream initialization. Refer to AscendCL's list of external interfaces.
    // Fill in the deviceId based on your actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret); 
        return ret;
    }

    // 2. To construct input and output, it is necessary to customize the construction according to the API interface.
    
    int32_t batchSize = 4; // N
    int32_t numHeads = 8; // N
    int32_t numKeyValueHeads = numHeads;
    int32_t qS = 1024;
    int32_t kvS = 1024;
    int32_t headDim = 128;
    int32_t ropeHeadDim = 0;
    static const uint32_t aicCoreNum = 32;
 	static const uint32_t aivCoreNum = 64;

    double scaleValue = 1 / sqrt(headDim); // 1/sqrt(d)
    double sparse_block_size = 16;
    int32_t preTokens = 2147483647;
    int32_t nextTokens = 2147483647;
    string layout_query = "BSND";
    char layoutQuery[layout_query.length() + 1];
    strcpy(layoutQuery, layout_query.c_str());
    string layout_kv = "BSND";
    char layoutKv[layout_kv.length() + 1];
    strcpy(layoutKv, layout_kv.c_str());
    int32_t sparseMode = 0;
    int32_t attentionMode = 0;
    bool return_softmax_lse = false;

    std::vector<int64_t> actualSeqLengthsQueryShape = {batchSize};
    std::vector<int64_t> actualSeqLengthsKvShape = {batchSize};
    std::vector<int64_t> metadataShape = {1024};
    void *actualSeqLengthsQueryDeviceAddr = nullptr;
    void *actualSeqLengthsKvDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    aclTensor* actualSeqLengthsQueryTensor = nullptr;
    aclTensor* actualSeqLengthsKvTensor = nullptr;
    aclTensor* metadataTensor = nullptr;
    int64_t actualSeqLengthsQueryShapeSize = GetShapeSize(actualSeqLengthsQueryShape);
    int64_t actualSeqLengthsKvShapeSize = GetShapeSize(actualSeqLengthsKvShape);
    int64_t metadataShapeSize = GetShapeSize(metadataShape);
    std::vector<int32_t> actualSeqLengthsQueryHostData(actualSeqLengthsQueryShapeSize, qS);
    std::vector<int32_t> actualSeqLengthsKvHostData(actualSeqLengthsKvShapeSize, kvS);
    std::vector<int32_t> metadataHostData(metadataShapeSize, 1);
    


    // Create actual_seq_lengths_query aclTensor.
    ret = CreateAclTensor(actualSeqLengthsQueryHostData, actualSeqLengthsQueryShape, &actualSeqLengthsQueryDeviceAddr, aclDataType::ACL_INT32, &actualSeqLengthsQueryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create actual_seq_lengths_kv aclTensor.
    ret = CreateAclTensor(actualSeqLengthsKvHostData, actualSeqLengthsKvShape, &actualSeqLengthsKvDeviceAddr, aclDataType::ACL_INT32, &actualSeqLengthsKvTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create metadata aclTensor.
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadataTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;

    char socVersion[] = "ascend950";
    printf("start aclnnFlashAttnMetadata\n");
    ret = aclnnFlashAttnMetadataGetWorkspaceSize(
        actualSeqLengthsQueryTensor, actualSeqLengthsKvTensor, numHeads, numKeyValueHeads,
        headDim, sparseMode, batchSize, qS, kvS,
        metadataTensor,
        &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("aclnnFlashAttnMetadataGetWorkspaceSize %d\n",
            ret);
        return -1;
    }

    ret = aclnnFlashAttnMetadata(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("aclnnFlashAttnMetadata %d\n", ret);
        return -1;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSynchronizeStream %d\n", ret);
        return -1;
    }

    ret = aclrtMemcpy(metadataHostData.data(),
                        metadataHostData.size() * sizeof(metadataHostData[0]), metadataDeviceAddr,
                        optiling::FA_META_SIZE * sizeof(int32_t),
                        ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        printf("aclrtMemcpy %d\n", ret);
        return -1;
    }
    
    DumpMeta(&metadataHostData[0]);
    // return 0;

    aclDestroyTensor(metadataTensor);
    aclrtFree(metadataDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
