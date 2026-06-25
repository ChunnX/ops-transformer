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
 * \file test_aclnn_inplace_partial_rotary_mul_grad_bab.cpp
 * \brief BAB template test for InplacePartialRotaryMulGrad (1S1D, interleave mode).
 *
 * BAB template constraints:
 *   - Layout: BSND (dy shape [B, S, N, D])
 *   - B=1, N=1 (current implementation limitation)
 *   - cosb_==1 (cos/sin broadcast on B dimension)
 *   - rotary_mode=1 (interleave) only
 *
 * Interleave gradient formula:
 *   dx[2k]   = cos[2k]   * dy[2k]   + sin[2k+1] * dy[2k+1]
 *   dx[2k+1] = cos[2k+1] * dy[2k+1] - sin[2k]   * dy[2k]
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_inplace_partial_rotary_mul_grad.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// Reference: interleave gradient computation
void InterleaveGradRef(const std::vector<float> &dy, const std::vector<float> &cos, const std::vector<float> &sin,
                       std::vector<float> &dx, int64_t s, int64_t d)
{
    for (int64_t si = 0; si < s; si++) {
        for (int64_t k = 0; k < d / 2; k++) {
            int64_t idx0 = si * d + 2 * k;
            int64_t idx1 = si * d + 2 * k + 1;
            dx[idx0] = cos[idx0] * dy[idx0] + sin[idx1] * dy[idx1];
            dx[idx1] = cos[idx1] * dy[idx1] - sin[idx0] * dy[idx0];
        }
    }
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // BAB 1S1D layout: dy=[1, S, 1, D], cos=[1, S, 1, D], cosb_=1
    int64_t B = 1;
    int64_t S = 2;
    int64_t N = 1;
    int64_t D = 4;

    std::vector<int64_t> dyShape = {B, S, N, D};
    std::vector<int64_t> cosShape = {1, S, 1, D};  // cosb_=1: B dimension broadcast
    std::vector<int64_t> sinShape = {1, S, 1, D};
    int64_t rotaryMode = 1;  // interleave
    std::vector<int64_t> partialSliceData = {0, D};

    // Test data: 2 sequences, 4 elements each (interleave mode)
    // S=0: dy=[1.0, 2.0, 3.0, 4.0], cos=[0.5, 0.6, 0.7, 0.8], sin=[0.1, 0.2, 0.3, 0.4]
    // S=1: dy=[5.0, 6.0, 7.0, 8.0], cos=[0.5, 0.6, 0.7, 0.8], sin=[0.1, 0.2, 0.3, 0.4]
    //
    // Expected (interleave grad):
    //   dx[2k]   = cos[2k]*dy[2k]     + sin[2k+1]*dy[2k+1]
    //   dx[2k+1] = cos[2k+1]*dy[2k+1] - sin[2k]*dy[2k]
    //
    // S=0: dx[0]=0.5*1.0+0.2*2.0=0.9  dx[1]=0.6*2.0-0.1*1.0=1.1
    //      dx[2]=0.7*3.0+0.4*4.0=3.7  dx[3]=0.8*4.0-0.3*3.0=2.3
    // S=1: dx[4]=0.5*5.0+0.2*6.0=3.7  dx[5]=0.6*6.0-0.1*5.0=3.1
    //      dx[6]=0.7*7.0+0.4*8.0=8.1  dx[7]=0.8*8.0-0.3*7.0=4.3
    std::vector<float> dyHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> cosHostData = {0.5f, 0.6f, 0.7f, 0.8f, 0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<float> sinHostData = {0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> expectedDx = {0.9f, 1.1f, 3.7f, 2.3f, 3.7f, 3.1f, 8.1f, 4.3f};

    std::vector<float> referenceDx(expectedDx.size(), 0.0f);
    InterleaveGradRef(dyHostData, cosHostData, sinHostData, referenceDx, S, D);

    void *dyRefDeviceAddr = nullptr;
    void *cosDeviceAddr = nullptr;
    void *sinDeviceAddr = nullptr;
    aclTensor *dyRef = nullptr;
    aclTensor *cos = nullptr;
    aclTensor *sin = nullptr;
    aclTensor *dcosOut = nullptr;
    aclTensor *dsinOut = nullptr;

    ret = CreateAclTensor(dyHostData, dyShape, &dyRefDeviceAddr, aclDataType::ACL_FLOAT, &dyRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *partialSlice = aclCreateIntArray(partialSliceData.data(), partialSliceData.size());

    std::vector<int64_t> emptyShape = {1, 1, 1, 0};
    std::vector<int64_t> emptyStride(emptyShape.size(), 0);
    dcosOut = aclCreateTensor(emptyShape.data(), emptyShape.size(), aclDataType::ACL_FLOAT, emptyStride.data(), 0,
                              aclFormat::ACL_FORMAT_ND, emptyShape.data(), emptyShape.size(), nullptr);
    dsinOut = aclCreateTensor(emptyShape.data(), emptyShape.size(), aclDataType::ACL_FLOAT, emptyStride.data(), 0,
                              aclFormat::ACL_FORMAT_ND, emptyShape.data(), emptyShape.size(), nullptr);

    // First stage: get workspace size
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    ret = aclnnInplacePartialRotaryMulGradGetWorkspaceSize(dyRef, cos, sin, nullptr, rotaryMode, partialSlice, dcosOut,
                                                            dsinOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnInplacePartialRotaryMulGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Second stage: execute
    ret = aclnnInplacePartialRotaryMulGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplacePartialRotaryMulGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Read back result (inplace: dyRef was overwritten with dx)
    auto size = GetShapeSize(dyShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dyRefDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // Verify results
    const float epsilon = 1e-6f;
    bool allPassed = true;
    for (int64_t i = 0; i < size; i++) {
        bool ok = std::fabs(resultData[i] - referenceDx[i]) < epsilon;
        if (!ok) {
            LOG_PRINT("MISMATCH at [%ld]: got %.6f, expected %.6f (reference %.6f)\n", i, resultData[i], expectedDx[i],
                      referenceDx[i]);
            allPassed = false;
        }
    }

    if (allPassed) {
        LOG_PRINT("BAB test PASSED: all %ld elements match reference.\n", size);
        LOG_PRINT("Result: ");
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("%.6f ", resultData[i]);
        }
        LOG_PRINT("\n");
    } else {
        LOG_PRINT("BAB test FAILED.\n");
    }

    // Cleanup
    aclDestroyIntArray(partialSlice);
    aclDestroyTensor(dyRef);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(dcosOut);
    aclDestroyTensor(dsinOut);

    aclrtFree(dyRefDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return allPassed ? 0 : 1;
}
