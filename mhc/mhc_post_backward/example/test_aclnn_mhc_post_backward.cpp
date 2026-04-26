/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_post_backward.h"
#include "securec.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

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

} // namespace

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
    // Shape: TND (T=1024, n=4, D=2560)
    const int64_t T = 1024;
    const int64_t n = 4;
    const int64_t D = 2560;

    std::vector<int64_t> grad_outputShape = {T, n, D};
    std::vector<int64_t> xShape = {T, n, D};
    std::vector<int64_t> h_resShape = {T, n, n};
    std::vector<int64_t> h_outShape = {T, D};
    std::vector<int64_t> h_postShape = {T, n};
    std::vector<int64_t> grad_xShape = {T, n, D};
    std::vector<int64_t> grad_h_resShape = {T, n, n};
    std::vector<int64_t> grad_h_outShape = {T, D};
    std::vector<int64_t> grad_h_postShape = {T, n};

    void *grad_outputDeviceAddr = nullptr;
    void *xDeviceAddr = nullptr;
    void *h_resDeviceAddr = nullptr;
    void *h_outDeviceAddr = nullptr;
    void *h_postDeviceAddr = nullptr;
    void *grad_xDeviceAddr = nullptr;
    void *grad_h_resDeviceAddr = nullptr;
    void *grad_h_outDeviceAddr = nullptr;
    void *grad_h_postDeviceAddr = nullptr;

    aclTensor *grad_outputTensor = nullptr;
    aclTensor *xTensor = nullptr;
    aclTensor *h_resTensor = nullptr;
    aclTensor *h_outTensor = nullptr;
    aclTensor *h_postTensor = nullptr;
    aclTensor *grad_xTensor = nullptr;
    aclTensor *grad_h_resTensor = nullptr;
    aclTensor *grad_h_outTensor = nullptr;
    aclTensor *grad_h_postTensor = nullptr;

    int64_t grad_outputShapeSize = GetShapeSize(grad_outputShape);
    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t h_resShapeSize = GetShapeSize(h_resShape);
    int64_t h_outShapeSize = GetShapeSize(h_outShape);
    int64_t h_postShapeSize = GetShapeSize(h_postShape);
    int64_t grad_xShapeSize = GetShapeSize(grad_xShape);
    int64_t grad_h_resShapeSize = GetShapeSize(grad_h_resShape);
    int64_t grad_h_outShapeSize = GetShapeSize(grad_h_outShape);
    int64_t grad_h_postShapeSize = GetShapeSize(grad_h_postShape);

    // Create inpit data with default values
    std::vector<uinit16_t> grad_outputHostData(grad_outputShapeSize, 1);   // FP16/BF16 data
    std::vector<uinit16_t> xHostData(xShapeSize, 1);                       // FP16/BF16 data
    std::vector<float> h_resHostData(h_resShapeSize, 0.25f);               // FP32 data double-stochastistic constrained
    std::vector<uinit16_t> h_outHostData(h_outShapeSize, 1);               // FP16/BF16 data
    std::vector<float> h_postHostData(h_postShapeSize, 1.0f);              // FP32 data
    
    std::vector<uinit16_t> grad_xHostData(grad_xShapeSize, 0);             // FP16/BF16 output
    std::vector<float> grad_h_resHostData(grad_h_resShapeSize, 0.25f);     // FP32 output
    std::vector<uinit16_t> grad_h_outHostData(grad_h_outShapeSize, 1);     // FP16/BF16 output
    std::vector<float> grad_h_postHostData(grad_h_postShapeSize, 1.0f);    // FP32 output

    // Create grad_output aclTensor.
    ret = CreateAclTensor(grad_outputHostData, grad_outputShape, &grad_outputDeviceAddr,
        aclDataType::ACL_FLOAT16, &grad_outputTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create h_res aclTensor.
    ret = CreateAclTensor(h_resHostData, h_resShape, &h_resDeviceAddr, aclDataType::ACL_FLOAT, &h_resTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create h_out aclTensor.
    ret = CreateAclTensor(h_outHostData, h_outShape, &h_outDeviceAddr, aclDataType::ACL_FLOAT16, &h_outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create h_post aclTensor.
    ret = CreateAclTensor(h_postHostData, h_postShape, &h_postDeviceAddr, aclDataType::ACL_FLOAT, &h_postTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // Create grad_x aclTensor.
    ret = CreateAclTensor(grad_xHostData, grad_xShape, &grad_xDeviceAddr, aclDataType::ACL_FLOAT16, &grad_xTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create grad_h_res aclTensor.
    ret = CreateAclTensor(grad_h_resHostData, grad_h_resShape, &grad_h_resDeviceAddr,
        aclDataType::ACL_FLOAT, &grad_h_resTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create grad_h_out aclTensor.
    ret = CreateAclTensor(grad_h_outHostData, grad_h_outShape, &grad_h_outDeviceAddr,
        aclDataType::ACL_FLOAT16, &grad_h_outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }
    // Create grad_h_post aclTensor.
    ret = CreateAclTensor(grad_h_postHostData, grad_h_postShape, &grad_h_postDeviceAddr,
        aclDataType::ACL_FLOAT, &grad_h_postTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        return ret;
    }

    // 3. Call CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // Call the first interface.
    ret = aclnnMhcPostBackwardGetWorkspaceSize(
        grad_outputTensor, xTensor, h_resTensor, h_outTensor, h_postTensor,
        grad_xTensor, grad_h_resTensor, grad_h_outTensor, grad_h_postTensor,
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    // Apply for device memory based on the workspaceSize calculated from the first interface paragraph.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }
    // Call the second interface.
    ret = aclnnMhcPostBackward(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnMhcPostBackward failed. ERROR: %d\n", ret);
        return ret;
    }

    // 4. (Fixed writing method) Synchronize and wait for task execution to end.
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 5. Retrieve the output value, copy the result from the device side memory to the host side, and modify it
    // Retrive grad_x
    std::vector<uint16_t> grad_xResultData(grad_xShapeSize, 0);
    ret = aclrtMemcpy(grad_xResultData.data(), grad_xResultData.size() * sizeof(grad_xResultData[0]), grad_xDeviceAddr,
                      grad_xShapeSize * sizeof(grad_xResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy grad_x result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < grad_xShapeSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // Retrive grad_h_res
    std::vector<uint16_t> grad_h_resResultData(grad_h_resShapeSize, 0);
    ret = aclrtMemcpy(grad_h_resResultData.data(), grad_h_resResultData.size() * sizeof(grad_h_resResultData[0]),
        grad_h_resDeviceAddr, grad_h_resShapeSize * sizeof(grad_h_resResultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy grad_h_res result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < grad_h_resShapeSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // Retrive grad_h_out
    std::vector<uint16_t> grad_h_outResultData(grad_h_outShapeSize, 0);
    ret = aclrtMemcpy(grad_h_outResultData.data(), grad_h_outResultData.size() * sizeof(grad_h_outResultData[0]),
        grad_h_outDeviceAddr, grad_h_outShapeSize * sizeof(grad_h_outResultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy grad_h_out result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < grad_h_outShapeSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // Retrive grad_h_post
    std::vector<uint16_t> grad_h_postResultData(grad_h_postShapeSize, 0);
    ret = aclrtMemcpy(grad_h_postResultData.data(), grad_h_postResultData.size() * sizeof(grad_h_postResultData[0]),
        grad_h_postDeviceAddr, grad_h_postShapeSize * sizeof(grad_h_postResultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy grad_h_post result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < grad_h_postShapeSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release resources.
    aclDestroyTensor(grad_outputTensor);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(h_resTensor);
    aclDestroyTensor(h_outTensor);
    aclDestroyTensor(h_postTensor);
    aclDestroyTensor(grad_xTensor);
    aclDestroyTensor(grad_h_resTensor);
    aclDestroyTensor(grad_h_outTensor);
    aclDestroyTensor(grad_h_postTensor);

    aclrtFree(grad_outputDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(h_resDeviceAddr);
    aclrtFree(h_outDeviceAddr);
    aclrtFree(h_postDeviceAddr);
    aclrtFree(grad_xDeviceAddr);
    aclrtFree(grad_h_resDeviceAddr);
    aclrtFree(grad_h_outDeviceAddr);
    aclrtFree(grad_h_postDeviceAddr);
    
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    
    return 0;
}