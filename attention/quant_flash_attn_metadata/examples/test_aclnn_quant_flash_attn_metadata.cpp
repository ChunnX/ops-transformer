/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License").
 * Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS
 * SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root
 * of the software repository
 * for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_flash_attn_metadata.h"
#include "../op_kernel_aicpu/quant_flash_attn_metadata.h"
#include "securec.h"

using namespace std;
using namespace optiling;

#define CHECK_RET(cond) ((cond) ? true : (false))

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

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

static void DumpMeta(void *data)
{
    int32_t sectionNum = static_cast<int32_t>(((int32_t *)data)[0]);
    optiling::detail::QFaMetaData qfaMetadata(data, sectionNum);
    printf("sectionNum:%d\n", sectionNum);
    for (uint32_t sectionId = 0; sectionId < sectionNum; ++sectionId) {
        printf("sectionIdx:%d\n", sectionId);
        for (size_t i = 0; i < optiling::AIC_CORE_NUM; ++i) {
            printf("bn2 start: %d\n", qfaMetadata.getQFaMetadata(sectionId, i, optiling::QFA_BN2_START_INDEX));
            printf("m start: %d\n", qfaMetadata.getQFaMetadata(sectionId, i, optiling::QFA_M_START_INDEX));
            printf("s2 start: %d\n", qfaMetadata.getQFaMetadata(sectionId, i, optiling::QFA_S2_START_INDEX));
            printf("bn2 end: %d\n", qfaMetadata.getQFaMetadata(sectionId, i, optiling::QFA_BN2_END_INDEX));
            printf("m end: %d\n", qfaMetadata.getQFaMetadata(sectionId, i, optiling::QFA_M_END_INDEX));
            printf("s2 end: %d\n", qfaMetadata.getQFaMetadata(sectionId, i, optiling::QFA_S2_END_INDEX));
            printf("first qfd data ws idx: %d\n",
                   qfaMetadata.getQFaMetadata(sectionId, i, optiling::QFA_FIRST_QFD_DATA_WORKSPACE_IDX_INDEX));
        }

        for (size_t i = 0; i < optiling::AIV_CORE_NUM; ++i) {
            printf("bn2 idx: %d\n", qfaMetadata.getQFdMetadata(sectionId, i, optiling::QFD_BN2_IDX_INDEX));
            printf("m idx: %d\n", qfaMetadata.getQFdMetadata(sectionId, i, optiling::QFD_M_IDX_INDEX));
            printf("fd workspace idx: %d\n",
                   qfaMetadata.getQFdMetadata(sectionId, i, optiling::QFD_WORKSPACE_IDX_INDEX));
            printf("fd workspace num: %d\n",
                   qfaMetadata.getQFdMetadata(sectionId, i, optiling::QFD_WORKSPACE_NUM_INDEX));
            printf("m start: %d\n", qfaMetadata.getQFdMetadata(sectionId, i, optiling::QFD_M_START_INDEX));
            printf("m num: %d\n", qfaMetadata.getQFdMetadata(sectionId, i, optiling::QFD_M_NUM_INDEX));
        }
    }
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
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

int main()
{
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

    int32_t batchSize = 4;
    int32_t numHeadsQ = 32;
    int32_t numHeadsKv = numHeadsQ;
    int32_t maxSeqlenQ = 1;
    int32_t maxSeqlenKv = 8192;
    int32_t headDim = 128;

    int32_t winLeft = -1;
    int32_t winRight = -1;
    string layout_q = "BSND";
    char layoutQuery[layout_q.length() + 1];
    strcpy(layoutQuery, layout_q.c_str());
    string layout_kv = "BSND";
    char layoutKv[layout_kv.length() + 1];
    strcpy(layoutKv, layout_kv.c_str());
    string layout_out = "BSND";
    char layoutOut[layout_out.length() + 1];
    strcpy(layoutOut, layout_out.c_str());
    int32_t maskMode = 3;

    int64_t metadataSize = ((36 + 72) * batchSize * numHeadsKv + 1) * 16;
    int64_t alignedSize = ((metadataSize + 4095) / 4096) * 4096;

    std::vector<int64_t> sequsedQShape = {batchSize};
    std::vector<int64_t> sequsedKvShape = {batchSize};
    std::vector<int64_t> metadataShape = {alignedSize};
    void *sequsedQDeviceAddr = nullptr;
    void *sequsedKvDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    aclTensor *sequsedQTensor = nullptr;
    aclTensor *sequsedKvTensor = nullptr;
    aclTensor *metadataTensor = nullptr;
    int64_t sequsedQShapeSize = GetShapeSize(sequsedQShape);
    int64_t sequsedKvShapeSize = GetShapeSize(sequsedKvShape);
    int64_t metadataShapeSize = GetShapeSize(metadataShape);
    std::vector<int32_t> sequsedQHostData(sequsedQShapeSize, maxSeqlenQ);
    std::vector<int32_t> sequsedKvHostData(sequsedKvShapeSize, maxSeqlenKv);
    std::vector<int32_t> metadataHostData(metadataShapeSize, 0);


    ret =
        CreateAclTensor(sequsedQHostData, sequsedQShape, &sequsedQDeviceAddr, aclDataType::ACL_INT32, &sequsedQTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret = CreateAclTensor(sequsedKvHostData, sequsedKvShape, &sequsedKvDeviceAddr, aclDataType::ACL_INT32,
                          &sequsedKvTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    ret =
        CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadataTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    aclOpExecutor *executor = nullptr;
    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    char socVersion[] = "ascend950";
    int32_t qQuantMode = 0;
    int32_t kQuantMode = 0;
    int32_t vQuantMode = 0;
    int32_t qDtype = 0; // FP16
    int32_t kDtype = 0; // FP16
    int32_t vDtype = 0; // FP16

    printf("start aclnnQuantFlashAttnMetadata\n");
    ret = aclnnQuantFlashAttnMetadataGetWorkspaceSize(
        nullptr, nullptr, sequsedQTensor, sequsedKvTensor, batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv,
        headDim, qQuantMode, kQuantMode, vQuantMode, qDtype, kDtype, vDtype, maskMode, winLeft, winRight, layoutQuery,
        layoutKv, layoutOut, metadataTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("aclnnQuantFlashAttnMetadataGetWorkspaceSize %d\n", ret);
        return -1;
    }

    ret = aclnnQuantFlashAttnMetadata(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("aclnnQuantFlashAttnMetadata %d\n", ret);
        return -1;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSynchronizeStream %d\n", ret);
        return -1;
    }

    ret = aclrtMemcpy(metadataHostData.data(), metadataHostData.size() * sizeof(metadataHostData[0]),
                      metadataDeviceAddr, metadataShapeSize, ACL_MEMCPY_DEVICE_TO_HOST);
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
