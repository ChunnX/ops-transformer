/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file test_aclnn_pdist.cpp
 * \brief aclnn eager-mode test case for Pdist.
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include "securec.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

// aclnnPdist is provided by the built-in CANN libopapi.so; the CANN installs used here ship no
// aclnnop/aclnn_pdist.h, so the two-stage entries are declared inline, matching the built-in
// prototype exactly (note: p is float, out is non-const).
#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnPdistGetWorkspaceSize(const aclTensor *x, float p, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnPdist(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

// Host-side reference for a small fp32 case (upper-triangular, row-major).
void PdistGolden(const std::vector<float> &x, int64_t n, int64_t m, double p, std::vector<float> &golden)
{
    golden.assign(static_cast<size_t>(n * (n - 1) / 2), 0.0f);
    size_t idx = 0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i + 1; j < n; j++) {
            double s = 0.0;
            for (int64_t k = 0; k < m; k++) {
                double d = std::fabs(static_cast<double>(x[i * m + k]) - static_cast<double>(x[j * m + k]));
                if (p == 0.0) {
                    s += (d != 0.0) ? 1.0 : 0.0;
                } else if (std::isinf(p)) {
                    s = (k == 0) ? d : std::max(s, d);
                } else {
                    s += std::pow(d, p);
                }
            }
            if (p != 0.0 && !std::isinf(p)) {
                s = std::pow(s, 1.0 / p);
            }
            golden[idx++] = static_cast<float>(s);
        }
    }
}

// One fp32 case: build x [N, M], run aclnnPdist with norm p, compare against the host golden.
int RunCaseFp32(aclrtStream stream, int64_t N, int64_t M, double p)
{
    std::vector<int64_t> xShape = {N, M};
    std::vector<int64_t> yShape = {N * (N - 1) / 2};

    void *xDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    aclTensor *x = nullptr;
    aclTensor *y = nullptr;

    std::vector<float> xHostData(N * M);
    for (int64_t i = 0; i < N * M; i++) {
        xHostData[i] = static_cast<float>((i * 37 % 19) - 9) * 0.25f;
    }
    std::vector<float> yHostData(yShape[0], 0.0f);

    auto ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnPdistGetWorkspaceSize(x, p, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPdistGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnPdist(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPdist failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<float> resultData(yShape[0], 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), yDeviceAddr,
                      yShape[0] * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::vector<float> golden;
    PdistGolden(xHostData, N, M, p, golden);
    int64_t mismatch = 0;
    double maxAbsErr = 0.0;
    for (size_t i = 0; i < golden.size(); i++) {
        double err = std::fabs(static_cast<double>(resultData[i]) - static_cast<double>(golden[i]));
        if (err > 1e-3 + 1e-3 * std::fabs(static_cast<double>(golden[i]))) {
            mismatch++;
        }
        maxAbsErr = std::max(maxAbsErr, err);
    }
    LOG_PRINT("[fp32 N=%ld M=%ld p=%.4g] total %ld pairs, mismatch %ld, maxAbsErr %.3e -> %s\n",
              N, M, p, (long)golden.size(), (long)mismatch, maxAbsErr, mismatch == 0 ? "pass" : "FAIL");

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return mismatch == 0 ? 0 : 1;
}

// One fp16 case: output compared against the fp32 golden with the fp16 tolerance.
int RunCaseFp16(aclrtStream stream, int64_t N, int64_t M, double p)
{
    std::vector<int64_t> xShape = {N, M};
    std::vector<int64_t> yShape = {N * (N - 1) / 2};

    void *xDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    aclTensor *x = nullptr;
    aclTensor *y = nullptr;

    std::vector<float> xHostData(N * M);
    for (int64_t i = 0; i < N * M; i++) {
        xHostData[i] = static_cast<float>((i * 37 % 19) - 9) * 0.25f;
    }
    std::vector<aclFloat16> xHostHalf(xHostData.size());
    for (size_t i = 0; i < xHostData.size(); i++) {
        xHostHalf[i] = aclFloatToFloat16(xHostData[i]);
    }
    std::vector<aclFloat16> yHostHalf(yShape[0], aclFloatToFloat16(0.0f));

    auto ret = CreateAclTensor(xHostHalf, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostHalf, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnPdistGetWorkspaceSize(x, p, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPdistGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnPdist(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPdist failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<float> resultData(yShape[0], 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), yDeviceAddr,
                      yShape[0] * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // fp16 payload copied as raw halfs: expand back to float.
    {
        std::vector<aclFloat16> halfView(yShape[0]);
        errno_t mcRet = memcpy_s(halfView.data(), halfView.size() * sizeof(aclFloat16), resultData.data(),
                                 yShape[0] * sizeof(aclFloat16));
        CHECK_RET(mcRet == 0, LOG_PRINT("memcpy_s halfs failed\n"); return -1);
        for (int64_t i = 0; i < yShape[0]; i++) {
            resultData[i] = aclFloat16ToFloat(halfView[i]);
        }
    }

    std::vector<float> golden;
    PdistGolden(xHostData, N, M, p, golden);
    int64_t mismatch = 0;
    double maxAbsErr = 0.0;
    for (size_t i = 0; i < golden.size(); i++) {
        double err = std::fabs(static_cast<double>(resultData[i]) - static_cast<double>(golden[i]));
        if (err > 1e-1 + 5e-3 * std::fabs(static_cast<double>(golden[i]))) {
            mismatch++;
        }
        maxAbsErr = std::max(maxAbsErr, err);
    }
    LOG_PRINT("[fp16 N=%ld M=%ld p=%.4g] total %ld pairs, mismatch %ld, maxAbsErr %.3e -> %s\n",
              N, M, p, (long)golden.size(), (long)mismatch, maxAbsErr, mismatch == 0 ? "pass" : "FAIL");

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return mismatch == 0 ? 0 : 1;
}

int main()
{
    // 1. device/context/stream init
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Case sweep: fp32 across all PBranch classes (p=0/1/2/0.5/inf/general) plus
    // a large-M multi-block case, and fp16 for the two performance-representative norms.
    int failures = 0;
    failures += RunCaseFp32(stream, 64, 16, 2.0) ? 1 : 0;      // PBranch 2 (Euclidean)
    failures += RunCaseFp32(stream, 64, 16, 1.0) ? 1 : 0;      // PBranch 3 (Manhattan)
    failures += RunCaseFp32(stream, 64, 16, 0.0) ? 1 : 0;      // PBranch 0 (Hamming)
    failures += RunCaseFp32(stream, 64, 16, 0.5) ? 1 : 0;      // PBranch 4 (fractional)
    failures += RunCaseFp32(stream, 64, 16, std::numeric_limits<double>::infinity()) ? 1 : 0; // PBranch 1 (Chebyshev)
    failures += RunCaseFp32(stream, 64, 16, 3.0) ? 1 : 0;      // PBranch 5 (general)
    failures += RunCaseFp32(stream, 130, 300, 2.0) ? 1 : 0;    // multi-M-block + pair tail
    failures += RunCaseFp16(stream, 64, 16, 2.0) ? 1 : 0;      // fp16 widen/narrow chain
    failures += RunCaseFp16(stream, 64, 16, 1.0) ? 1 : 0;

    LOG_PRINT(failures == 0 ? "ALL CASES PASS\n" : "FAIL: %d cases out of tolerance\n", failures);

    // 3. Cleanup.
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return failures == 0 ? 0 : 1;
}
