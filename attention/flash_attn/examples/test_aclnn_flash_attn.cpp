/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_flash_attn.cpp
 * \brief FlashAttn算子调用示例（推理场景）
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attn.h"

namespace {

#define CHECK_RET(cond)  ((cond) ? true : (false))

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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

} // namespace

int main()
{
    // 1. （固定写法）device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. 构造输入与输出
    // BNSD推理场景：B=1, N_q=8, N_kv=2 (GQA 4:1), S_q=S_kv=128, D=64
    const int64_t B   = 1;
    const int64_t N_q = 8;
    const int64_t N_kv = 2;   // GQA：每2个q heads共享1个kv head
    const int64_t S_q = 128;
    const int64_t S_kv = 128;
    const int64_t D   = 64;

    // Q/Out: BNSD (B, N, S, D)
    // KV:    BSND (B, S, N, D) — layoutKv仅支持BSND/TND/PA_ND/PA_Nz
    std::vector<int64_t> qShape   = {B, N_q,  S_q,  D};   // BNSD
    std::vector<int64_t> kShape   = {B, S_kv, N_kv, D};   // BSND
    std::vector<int64_t> vShape   = {B, S_kv, N_kv, D};   // BSND
    std::vector<int64_t> outShape = {B, N_q,  S_q,  D};   // BNSD

    void *qDeviceAddr   = nullptr;
    void *kDeviceAddr   = nullptr;
    void *vDeviceAddr   = nullptr;
    void *outDeviceAddr = nullptr;

    aclTensor *qTensor   = nullptr;
    aclTensor *kTensor   = nullptr;
    aclTensor *vTensor   = nullptr;
    aclTensor *outTensor = nullptr;

    // 初始化host数据（FP16用uint16_t模拟，实际使用half类型）
    std::vector<uint16_t> qHostData(GetShapeSize(qShape),   0x3C00);  // 1.0 in FP16
    std::vector<uint16_t> kHostData(GetShapeSize(kShape),   0x3C00);
    std::vector<uint16_t> vHostData(GetShapeSize(vShape),   0x3C00);
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);

    ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT16, &qTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }
    ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT16, &kTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT16, &vTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &outTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) { return ret; }

    // 属性参数
    // softmaxMode=0.0f 表示使用默认缩放系数 1/sqrt(D)
    float   softmaxMode      = 0.0f;
    int64_t maskMode         = 0;      // 0: 无掩码
    int64_t winLeft          = 0;
    int64_t winRight         = 0;
    const char *layoutQ      = "BNSD";
    const char *layoutKv     = "BSND";  // KV布局仅支持BSND/TND/PA_ND/PA_Nz
    const char *layoutOut    = "BNSD";
    int64_t returnSoftmaxLse = 0;      // 推理场景不输出softmax_lse
    int64_t deterministic    = 0;

    // 3. 调用aclnnFlashAttn两段式接口
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    // 第一段：计算workspace大小
    ret = aclnnFlashAttnGetWorkspaceSize(
        qTensor, kTensor, vTensor,
        nullptr,    // blockTableOptional: 无PA
        nullptr,    // cuSeqlensQOptional
        nullptr,    // cuSeqlensKvOptional
        nullptr,    // sequsedQOptional
        nullptr,    // sequsedKvOptional
        nullptr,    // sinksOptional
        nullptr,    // metadataOptional
        softmaxMode, maskMode, winLeft, winRight,
        layoutQ, layoutKv, layoutOut,
        returnSoftmaxLse, deterministic,
        outTensor,
        nullptr,    // softmaxLseOptional: 推理场景不输出
        &workspaceSize, &executor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFlashAttnGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }
    else {
        LOG_PRINT("aclnnFlashAttnGetWorkspaceSize success. \n");
    }

    // 根据workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0U) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    // 第二段：执行计算
    ret = aclnnFlashAttn(workspaceAddr, workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFlashAttn failed. ERROR: %d\n", ret);
        return ret;
    }
    else {
        LOG_PRINT("aclnnFlashAttn success. \n");
    }

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }

    // 5. 获取输出结果，将device侧内存拷贝至host侧
    auto outSize = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(outSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                      outDeviceAddr, outSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
    }
    for (int64_t i = 0; i < std::min(outSize, int64_t(8)); i++) {
        LOG_PRINT("attentionOut[%ld] = 0x%04x (FP16)\n", i, resultData[i]);
    }

    // 6. 释放aclTensor资源
    aclDestroyTensor(qTensor);
    aclDestroyTensor(kTensor);
    aclDestroyTensor(vTensor);
    aclDestroyTensor(outTensor);

    // 7. 释放device资源
    aclrtFree(qDeviceAddr);
    aclrtFree(kDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0U) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}