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
 * \file test_aclnn_bsa_select_block_mask.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_bsa_select_block_mask.h"

using namespace std;

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
            return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    if (shape.empty()) {
        LOG_PRINT("CreateAclTensor: ERROR - shape is empty\n");
        return -1;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            LOG_PRINT("CreateAclTensor: ERROR - shape[%zu]=%ld is invalid\n", i, shape[i]);
            return -1;
        }
    }
    
    auto size = GetShapeSize(shape) * sizeof(T);
    
    if (hostData.size() != static_cast<size_t>(GetShapeSize(shape))) {
        LOG_PRINT("CreateAclTensor: ERROR - hostData size mismatch: %zu vs %ld\n", 
                  hostData.size(), GetShapeSize(shape));
        return -1;
    }
    
    *deviceAddr = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); 
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return ret);
    
    std::vector<int64_t> strides(shape.size(), 1);
    if (shape.size() > 1) {
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    *tensor = nullptr;
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed - returned nullptr\n"); 
              aclrtFree(*deviceAddr); *deviceAddr = nullptr; return -1);
    return 0;
}

int main() {
    // 1. device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 设置核心参数 (以BNSD Layout为例)
    int32_t batch = 1;
    int32_t numHeads = 4;
    int32_t numKHeads = 4;      // 仅支持MHA: N == KN
    int32_t qSeqlen = 256 * 1024; // 256K
    int32_t kSeqlen = 256 * 1024; // 256K
    int32_t headDim = 128;
    int32_t blockShapeX = 128;
    int32_t blockShapeY = 128;

    // 块数量计算
    int32_t ceilQ = (qSeqlen + blockShapeX - 1) / blockShapeX;
    int32_t ceilK = (kSeqlen + blockShapeY - 1) / blockShapeY;

    // 3. 构建张量Shape
    std::vector<int64_t> qShape = {batch, numHeads, qSeqlen, headDim};
    std::vector<int64_t> kShape = {batch, numKHeads, kSeqlen, headDim};
    std::vector<int64_t> maskShape = {batch, numHeads, ceilQ, ceilK};

    // 4. 分配并初始化Host数据
    int64_t qSize = GetShapeSize(qShape);
    int64_t kSize = GetShapeSize(kShape);

    std::vector<op::fp16_t> qData(qSize, 0.1f);
    std::vector<op::fp16_t> kData(kSize, 0.1f);
    
    // mask输出初始化为0
    std::vector<int8_t> maskOutData(GetShapeSize(maskShape), 0);

    // 创建所有aclTensor
    void *qAddr = nullptr, *kAddr = nullptr, *maskAddr = nullptr;
    aclTensor *qTensor = nullptr, *kTensor = nullptr, *maskTensor = nullptr;

    CreateAclTensor(qData, qShape, &qAddr, aclDataType::ACL_FLOAT16, &qTensor);
    CreateAclTensor(kData, kShape, &kAddr, aclDataType::ACL_FLOAT16, &kTensor);
    CreateAclTensor(maskOutData, maskShape, &maskAddr, aclDataType::ACL_INT8, &maskTensor);

    // 5. 创建aclIntArray属性参数
    std::vector<int64_t> blockShapeVec = {blockShapeX, blockShapeY};
    aclIntArray *blockShapeArr = aclCreateIntArray(blockShapeVec.data(), blockShapeVec.size());

    // 6. 标量与字符串参数配置
    char queryLayoutBuffer[16] = "BNSD";
    char keyLayoutBuffer[16] = "BNSD";
    double scaleValue = 1.0 / std::sqrt(static_cast<double>(headDim));
    double sparsity = 0.5;

    // 7. 调用第一段接口: GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    LOG_PRINT("Calling aclnnBSASelectBlockMaskGetWorkspaceSize...\n");
    ret = aclnnBSASelectBlockMaskGetWorkspaceSize(
        qTensor, 
        kTensor, 
        blockShapeArr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        queryLayoutBuffer,
        keyLayoutBuffer,
        static_cast<int64_t>(numKHeads),
        scaleValue, 
        sparsity, 
        maskTensor,
        &workspaceSize, 
        &executor
    );

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    CHECK_RET(executor != nullptr, LOG_PRINT("executor is null after GetWorkspaceSize\n"); return -1);
    LOG_PRINT("Workspace size required: %lu bytes\n", workspaceSize);

    // 8. 分配workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 9. 调用第二段接口: 执行计算
    LOG_PRINT("Calling aclnnBSASelectBlockMask...\n");
    ret = aclnnBSASelectBlockMask(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBSASelectBlockMask failed. ERROR: %d\n", ret); return ret);

    // 10. 同步Stream，等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret); 

    // 11. 将结果拷贝回Host侧打印 (blockSparseMaskOut)
    int64_t maskSize = GetShapeSize(maskShape);
    ret = aclrtMemcpy(maskOutData.data(), maskSize * sizeof(int8_t), maskAddr, maskSize * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed.\n"); return ret);

    LOG_PRINT("Execution Success! BlockSparseMask output (first 20 elements):\n");
    int64_t displayCount = (maskSize < 20) ? maskSize : 20;
    for (int64_t i = 0; i < displayCount; i++) {
        LOG_PRINT("  mask index %ld: %u\n", i, static_cast<unsigned int>(maskOutData[i]));
    }

    // 12. 释放所有资源
    LOG_PRINT("Cleaning up resources...\n");

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    
    aclrtFree(qAddr);
    aclrtFree(kAddr);
    aclrtFree(maskAddr);

    aclDestroyTensor(qTensor);
    aclDestroyTensor(kTensor);
    aclDestroyTensor(maskTensor);
    
    aclDestroyIntArray(blockShapeArr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    LOG_PRINT("BSASelectBlockMask Test completed successfully!\n");
    return 0;
}
