/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_flash_attn_grad.h"

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

template <typename T>
int CreateEmptyAclTensor(const std::vector<int64_t> &shape, void **deviceAddr, aclDataType dataType,
                         aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // ==================== 1. 初始化 ACL 环境 ====================
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // ==================== 2. 定义算子输入/输出 shape ====================
    // BSND layout: q=[B,Sq,N,D], k=[B,Skv,N,D], v=[B,Skv,N,D], dout=[B,Sq,N,D]
    // 量化场景 D=128, 量化块大小 512
    int64_t B = 1;
    int64_t Sq = 512;
    int64_t Skv = 512;
    int64_t N = 1;
    int64_t D = 128;
    int64_t quantBlockSize = 512;
    int64_t s1AlignQuant = (Sq + quantBlockSize - 1) / quantBlockSize; // ceil(Sq/512) = 1
    int64_t s2AlignQuant = (Skv + quantBlockSize - 1) / quantBlockSize; // ceil(Skv/512) = 1

    std::vector<int64_t> qShape = {B, Sq, N, D};
    std::vector<int64_t> kShape = {B, Skv, N, D};
    std::vector<int64_t> vShape = {B, Skv, N, D};
    std::vector<int64_t> doutShape = {B, Sq, N, D};
    std::vector<int64_t> attnOutShape = {B, Sq, N, D};
    std::vector<int64_t> qDescaleShape = {B, N, s1AlignQuant, 1};
    std::vector<int64_t> kDescaleShape = {B, N, s2AlignQuant, 1};
    std::vector<int64_t> vDescaleShape = {B, N, s2AlignQuant, 1};
    std::vector<int64_t> doDescaleShape = {B, N, s1AlignQuant, 1};
    std::vector<int64_t> pScaleShape = {1};
    std::vector<int64_t> dsScaleShape = {1};
    std::vector<int64_t> softmaxLseShape = {B, N, Sq, 1};
    std::vector<int64_t> metadataShape = {64};

    // 输出 shape
    std::vector<int64_t> dqShape = {B, Sq, N, D};
    std::vector<int64_t> dkShape = {B, Skv, N, D};
    std::vector<int64_t> dvShape = {B, Skv, N, D};
    std::vector<int64_t> dsinkShape = {N};

    // ==================== 3. 准备 host 数据并创建 device tensor ====================
    void *qDeviceAddr = nullptr;
    void *kDeviceAddr = nullptr;
    void *vDeviceAddr = nullptr;
    void *doutDeviceAddr = nullptr;
    void *attnOutDeviceAddr = nullptr;
    void *qDescaleDeviceAddr = nullptr;
    void *kDescaleDeviceAddr = nullptr;
    void *vDescaleDeviceAddr = nullptr;
    void *doDescaleDeviceAddr = nullptr;
    void *pScaleDeviceAddr = nullptr;
    void *dsScaleDeviceAddr = nullptr;
    void *softmaxLseDeviceAddr = nullptr;
    void *metadataDeviceAddr = nullptr;
    void *dqDeviceAddr = nullptr;
    void *dkDeviceAddr = nullptr;
    void *dvDeviceAddr = nullptr;
    void *dsinkDeviceAddr = nullptr;

    aclTensor *q = nullptr;
    aclTensor *k = nullptr;
    aclTensor *v = nullptr;
    aclTensor *dout = nullptr;
    aclTensor *attnOut = nullptr;
    aclTensor *qDescale = nullptr;
    aclTensor *kDescale = nullptr;
    aclTensor *vDescale = nullptr;
    aclTensor *doDescale = nullptr;
    aclTensor *pScale = nullptr;
    aclTensor *dsScale = nullptr;
    aclTensor *softmaxLse = nullptr;
    aclTensor *metadata = nullptr;
    aclTensor *dq = nullptr;
    aclTensor *dk = nullptr;
    aclTensor *dv = nullptr;
    aclTensor *dsink = nullptr;

    // HIFLOAT8 量化输入 (uint8_t 承载)
    std::vector<uint8_t> qHostData(GetShapeSize(qShape), 0);
    std::vector<uint8_t> kHostData(GetShapeSize(kShape), 0);
    std::vector<uint8_t> vHostData(GetShapeSize(vShape), 0);
    std::vector<uint8_t> doutHostData(GetShapeSize(doutShape), 0);
    // attn_out: BF16
    std::vector<short> attnOutHostData(GetShapeSize(attnOutShape), 0);
    // descale / scale: FP32
    std::vector<float> qDescaleHostData(GetShapeSize(qDescaleShape), 1.0f);
    std::vector<float> kDescaleHostData(GetShapeSize(kDescaleShape), 1.0f);
    std::vector<float> vDescaleHostData(GetShapeSize(vDescaleShape), 1.0f);
    std::vector<float> doDescaleHostData(GetShapeSize(doDescaleShape), 1.0f);
    std::vector<float> pScaleHostData = {1.0f};
    std::vector<float> dsScaleHostData = {1.0f};
    std::vector<float> softmaxLseHostData(GetShapeSize(softmaxLseShape), 0.0f);
    // metadata: INT32
    std::vector<int32_t> metadataHostData(GetShapeSize(metadataShape), 0);

    // 创建输入 tensor
    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_HIFLOAT8, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_HIFLOAT8, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_HIFLOAT8, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(doutHostData, doutShape, &doutDeviceAddr, aclDataType::ACL_HIFLOAT8, &dout);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_BFLOAT16, &attnOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qDescaleHostData, qDescaleShape, &qDescaleDeviceAddr, aclDataType::ACL_FLOAT, &qDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kDescaleHostData, kDescaleShape, &kDescaleDeviceAddr, aclDataType::ACL_FLOAT, &kDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vDescaleHostData, vDescaleShape, &vDescaleDeviceAddr, aclDataType::ACL_FLOAT, &vDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(doDescaleHostData, doDescaleShape, &doDescaleDeviceAddr, aclDataType::ACL_FLOAT, &doDescale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(pScaleHostData, pScaleShape, &pScaleDeviceAddr, aclDataType::ACL_FLOAT, &pScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dsScaleHostData, dsScaleShape, &dsScaleDeviceAddr, aclDataType::ACL_FLOAT, &dsScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT,
                          &softmaxLse);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 tensor (BF16 / FP32)
    ret = CreateEmptyAclTensor<short>(dqShape, &dqDeviceAddr, aclDataType::ACL_BFLOAT16, &dq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateEmptyAclTensor<short>(dkShape, &dkDeviceAddr, aclDataType::ACL_BFLOAT16, &dk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateEmptyAclTensor<short>(dvShape, &dvDeviceAddr, aclDataType::ACL_BFLOAT16, &dv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateEmptyAclTensor<float>(dsinkShape, &dsinkDeviceAddr, aclDataType::ACL_FLOAT, &dsink);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // ==================== 4. 设置属性并调用算子 ====================
    int64_t quantMode = 0;      // 0: HIFLOAT8
    double softmaxScale = 0.088388; // 1/sqrt(128)
    int64_t maskMode = 0;       // 0: ALL_MASK
    int64_t winLeft = -1;
    int64_t winRight = -1;
    int64_t maxSeqlenQ = Sq;
    int64_t maxSeqlenKv = Skv;
    char layoutQ[5] = {'B', 'S', 'N', 'D', 0};
    char layoutK[5] = {'B', 'S', 'N', 'D', 0};

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnQuantFlashAttnGradGetWorkspaceSize(
        q, k, v, dout, attnOut, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, softmaxLse,
        nullptr, nullptr, nullptr, nullptr, nullptr, metadata,
        quantMode, softmaxScale, maskMode, winLeft, winRight, maxSeqlenQ, maxSeqlenKv, layoutQ, layoutK,
        dq, dk, dv, dsink, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnQuantFlashAttnGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnQuantFlashAttnGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantFlashAttnGrad failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnQuantFlashAttnGrad succeeded.\n");

    // ==================== 5. 释放资源 ====================
    aclDestroyTensor(q);
    aclDestroyTensor(k);
    aclDestroyTensor(v);
    aclDestroyTensor(dout);
    aclDestroyTensor(attnOut);
    aclDestroyTensor(qDescale);
    aclDestroyTensor(kDescale);
    aclDestroyTensor(vDescale);
    aclDestroyTensor(doDescale);
    aclDestroyTensor(pScale);
    aclDestroyTensor(dsScale);
    aclDestroyTensor(softmaxLse);
    aclDestroyTensor(metadata);
    aclDestroyTensor(dq);
    aclDestroyTensor(dk);
    aclDestroyTensor(dv);
    aclDestroyTensor(dsink);
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(doutDeviceAddr);
    aclrtFree(attnOutDeviceAddr);
    aclrtFree(qDescaleDeviceAddr);
    aclrtFree(kDescaleDeviceAddr);
    aclrtFree(vDescaleDeviceAddr);
    aclrtFree(doDescaleDeviceAddr);
    aclrtFree(pScaleDeviceAddr);
    aclrtFree(dsScaleDeviceAddr);
    aclrtFree(softmaxLseDeviceAddr);
    aclrtFree(metadataDeviceAddr);
    aclrtFree(dqDeviceAddr);
    aclrtFree(dkDeviceAddr);
    aclrtFree(dvDeviceAddr);
    aclrtFree(dsinkDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
