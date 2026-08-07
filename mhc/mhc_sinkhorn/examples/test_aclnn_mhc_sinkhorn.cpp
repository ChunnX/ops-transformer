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
 * \file test_aclnn_mhc_sinkhorn.cpp
 * \brief Test aclnnMhcSinkhorn parameter validation: outFlag=0 with invalid output shapes
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_sinkhorn.h"

#define CHECK_RET(cond, return_expr) \
    do { \
        if (!(cond)) { \
            return_expr; \
        } \
    } while (0)

#define LOG_PRINT(message, ...) \
    do { \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

int InitAcl(int32_t device_id, aclrtContext &context, aclrtStream &stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed: %d\n", ret); return -1);
    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed: %d\n", ret); return -1);
    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed: %d\n", ret); return -1);
    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed: %d\n", ret); return -1);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed: %d\n", ret); return -1);
    return 0;
}

int main()
{
    // ========== Init ==========
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    int ret = InitAcl(device_id, context, stream);
    CHECK_RET(ret == 0, LOG_PRINT("InitAcl failed\n"); return -1);

    // ========== Test 1: outFlag=0 via nullptr normOut/sumOut ==========
    LOG_PRINT("=== Test 1: outFlag=0 (nullptr). Expect SUCCESS. ===\n");
    {
        std::vector<int64_t> x_shape = {1024, 4, 4};
        std::vector<float> x_host(GetShapeSize(x_shape), 1.0f);

        void *x_dev = nullptr;
        void *out_dev = nullptr;
        aclTensor *x_t = nullptr;
        aclTensor *out_t = nullptr;

        ret = aclrtMalloc(&x_dev, GetShapeSize(x_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("x malloc failed\n"); return -1);
        ret = aclrtMemcpy(x_dev, GetShapeSize(x_shape) * sizeof(float), x_host.data(),
                          GetShapeSize(x_shape) * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        std::vector<int64_t> strides(x_shape.size(), 1);
        for (int64_t i = x_shape.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * x_shape[i + 1];
        x_t = aclCreateTensor(x_shape.data(), x_shape.size(), ACL_FLOAT, strides.data(), 0,
                              ACL_FORMAT_ND, x_shape.data(), x_shape.size(), x_dev);
        CHECK_RET(x_t != nullptr, LOG_PRINT("x_t create failed\n"); return -1);

        ret = aclrtMalloc(&out_dev, GetShapeSize(x_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        out_t = aclCreateTensor(x_shape.data(), x_shape.size(), ACL_FLOAT, nullptr, 0,
                                ACL_FORMAT_ND, x_shape.data(), x_shape.size(), out_dev);

        uint64_t ws = 0;
        aclOpExecutor *exec = nullptr;
        // normOut=nullptr, sumOut=nullptr → outFlag inferred as 0
        aclnnStatus s = aclnnMhcSinkhornGetWorkspaceSize(x_t, 1e-6f, 20, out_t, nullptr, nullptr, &ws, &exec);
        if (s == ACLNN_SUCCESS) {
            LOG_PRINT("Test 1 PASSED: outFlag=0 with nullptr returned SUCCESS.\n");
        } else {
            LOG_PRINT("Test 1 FAILED: outFlag=0 with nullptr returned %d.\n", s);
        }
        aclDestroyTensor(x_t);
        aclDestroyTensor(out_t);
        aclrtFree(x_dev);
        aclrtFree(out_dev);
    }

    // ========== Test 2: outFlag=0 via shape{1} normOut/sumOut ==========
    LOG_PRINT("=== Test 2: outFlag=0 (shape{1}). Expect SUCCESS. ===\n");
    {
        std::vector<int64_t> x_shape = {1024, 4, 4};
        std::vector<int64_t> small_shape = {1};
        std::vector<float> x_host(GetShapeSize(x_shape), 1.0f);

        void *x_dev = nullptr;
        void *out_dev = nullptr;
        void *n_dev = nullptr;
        void *s_dev = nullptr;
        aclTensor *x_t = nullptr;
        aclTensor *out_t = nullptr;
        aclTensor *n_t = nullptr;
        aclTensor *s_t = nullptr;

        aclrtMalloc(&x_dev, GetShapeSize(x_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(x_dev, GetShapeSize(x_shape) * sizeof(float), x_host.data(),
                    GetShapeSize(x_shape) * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        std::vector<int64_t> strides(x_shape.size(), 1);
        for (int64_t i = x_shape.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * x_shape[i + 1];
        x_t = aclCreateTensor(x_shape.data(), x_shape.size(), ACL_FLOAT, strides.data(), 0,
                              ACL_FORMAT_ND, x_shape.data(), x_shape.size(), x_dev);

        aclrtMalloc(&out_dev, GetShapeSize(x_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        out_t = aclCreateTensor(x_shape.data(), x_shape.size(), ACL_FLOAT, nullptr, 0,
                                ACL_FORMAT_ND, x_shape.data(), x_shape.size(), out_dev);

        // Pass shape{1} normOut/sumOut → outFlag inferred as 0
        aclrtMalloc(&n_dev, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        n_t = aclCreateTensor(small_shape.data(), 1, ACL_FLOAT, nullptr, 0,
                              ACL_FORMAT_ND, small_shape.data(), 1, n_dev);
        aclrtMalloc(&s_dev, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        s_t = aclCreateTensor(small_shape.data(), 1, ACL_FLOAT, nullptr, 0,
                              ACL_FORMAT_ND, small_shape.data(), 1, s_dev);

        uint64_t ws = 0;
        aclOpExecutor *exec = nullptr;
        aclnnStatus st = aclnnMhcSinkhornGetWorkspaceSize(x_t, 1e-6f, 20, out_t, n_t, s_t, &ws, &exec);
        if (st == ACLNN_SUCCESS) {
            LOG_PRINT("Test 2 PASSED: outFlag=0 with shape{1} returned SUCCESS.\n");
        } else {
            LOG_PRINT("Test 2 FAILED: outFlag=0 with shape{1} returned %d.\n", st);
        }
        aclDestroyTensor(x_t);
        aclDestroyTensor(out_t);
        aclDestroyTensor(n_t);
        aclDestroyTensor(s_t);
        aclrtFree(x_dev);
        aclrtFree(out_dev);
        aclrtFree(n_dev);
        aclrtFree(s_dev);
    }

    // ========== Test 3: Invalid multi-dim shape for normOut with outFlag=0 intent ==========
    LOG_PRINT("=== Test 3: 3 outputs, normOut/sumOut shape mismatch. Expect ACLNN_ERR_PARAM_INVALID. ===\n");
    {
        std::vector<int64_t> x_shape = {1024, 4, 4};
        // wrong multi-dim shapes (not 1D, not {1})
        std::vector<int64_t> norm_bad = {40, 1024, 4, 8};
        std::vector<int64_t> sum_bad = {40, 1024, 8};
        std::vector<float> x_host(GetShapeSize(x_shape), 1.0f);

        void *x_dev = nullptr;
        void *out_dev = nullptr;
        void *n_dev = nullptr;
        void *s_dev = nullptr;
        aclTensor *x_t = nullptr;
        aclTensor *out_t = nullptr;
        aclTensor *n_t = nullptr;
        aclTensor *s_t = nullptr;

        aclrtMalloc(&x_dev, GetShapeSize(x_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(x_dev, GetShapeSize(x_shape) * sizeof(float), x_host.data(),
                    GetShapeSize(x_shape) * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        std::vector<int64_t> strides(x_shape.size(), 1);
        for (int64_t i = x_shape.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * x_shape[i + 1];
        x_t = aclCreateTensor(x_shape.data(), x_shape.size(), ACL_FLOAT, strides.data(), 0,
                              ACL_FORMAT_ND, x_shape.data(), x_shape.size(), x_dev);

        aclrtMalloc(&out_dev, GetShapeSize(x_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        out_t = aclCreateTensor(x_shape.data(), x_shape.size(), ACL_FLOAT, nullptr, 0,
                                ACL_FORMAT_ND, x_shape.data(), x_shape.size(), out_dev);

        aclrtMalloc(&n_dev, GetShapeSize(norm_bad) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        n_t = aclCreateTensor(norm_bad.data(), norm_bad.size(), ACL_FLOAT, nullptr, 0,
                              ACL_FORMAT_ND, norm_bad.data(), norm_bad.size(), n_dev);
        aclrtMalloc(&s_dev, GetShapeSize(sum_bad) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        s_t = aclCreateTensor(sum_bad.data(), sum_bad.size(), ACL_FLOAT, nullptr, 0,
                              ACL_FORMAT_ND, sum_bad.data(), sum_bad.size(), s_dev);

        uint64_t ws = 0;
        aclOpExecutor *exec = nullptr;
        // 3 non-null outputs with non-{1} shapes → outFlag inferred as 1
        // but shapes don't match outFlag=1 expectations → should be caught
        aclnnStatus st = aclnnMhcSinkhornGetWorkspaceSize(x_t, 1e-6f, 20, out_t, n_t, s_t, &ws, &exec);
        if (st != ACLNN_SUCCESS) {
            LOG_PRINT("Test 3 PASSED: Invalid shapes correctly rejected with error %d.\n", st);
        } else {
            LOG_PRINT("Test 3 FAILED: Invalid shapes were NOT rejected!\n");
        }
        aclDestroyTensor(x_t);
        aclDestroyTensor(out_t);
        aclDestroyTensor(n_t);
        aclDestroyTensor(s_t);
        aclrtFree(x_dev);
        aclrtFree(out_dev);
        aclrtFree(n_dev);
        aclrtFree(s_dev);
    }

    // ========== Test 4: mixed state (one nullptr, one valid) ==========
    LOG_PRINT("=== Test 4: Mixed state (normOut nullptr, sumOut valid). Expect ACLNN_ERR_PARAM_INVALID. ===\n");
    {
        std::vector<int64_t> x_shape = {1024, 4, 4};
        std::vector<int64_t> sum_shape = {40 * 1024 * 8};
        std::vector<float> x_host(GetShapeSize(x_shape), 1.0f);

        void *x_dev = nullptr;
        void *out_dev = nullptr;
        void *s_dev = nullptr;
        aclTensor *x_t = nullptr;
        aclTensor *out_t = nullptr;
        aclTensor *s_t = nullptr;

        aclrtMalloc(&x_dev, GetShapeSize(x_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(x_dev, GetShapeSize(x_shape) * sizeof(float), x_host.data(),
                    GetShapeSize(x_shape) * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        std::vector<int64_t> strides(x_shape.size(), 1);
        for (int64_t i = x_shape.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * x_shape[i + 1];
        x_t = aclCreateTensor(x_shape.data(), x_shape.size(), ACL_FLOAT, strides.data(), 0,
                              ACL_FORMAT_ND, x_shape.data(), x_shape.size(), x_dev);

        aclrtMalloc(&out_dev, GetShapeSize(x_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        out_t = aclCreateTensor(x_shape.data(), x_shape.size(), ACL_FLOAT, nullptr, 0,
                                ACL_FORMAT_ND, x_shape.data(), x_shape.size(), out_dev);

        aclrtMalloc(&s_dev, GetShapeSize(sum_shape) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        s_t = aclCreateTensor(sum_shape.data(), 1, ACL_FLOAT, nullptr, 0,
                              ACL_FORMAT_ND, sum_shape.data(), 1, s_dev);

        uint64_t ws = 0;
        aclOpExecutor *exec = nullptr;
        // normOut=nullptr, sumOut valid → mixed → should be rejected
        aclnnStatus st = aclnnMhcSinkhornGetWorkspaceSize(x_t, 1e-6f, 20, out_t, nullptr, s_t, &ws, &exec);
        if (st != ACLNN_SUCCESS) {
            LOG_PRINT("Test 4 PASSED: Mixed state correctly rejected with error %d.\n", st);
        } else {
            LOG_PRINT("Test 4 FAILED: Mixed state was NOT rejected!\n");
        }
        aclDestroyTensor(x_t);
        aclDestroyTensor(out_t);
        aclDestroyTensor(s_t);
        aclrtFree(x_dev);
        aclrtFree(out_dev);
        aclrtFree(s_dev);
    }

    // Cleanup
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(device_id);
    aclFinalize();
    LOG_PRINT("All tests done.\n");
    return 0;
}