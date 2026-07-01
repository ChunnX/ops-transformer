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
 * \file test_aclnn_grouped_matmul_v5_weight_nz.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_weight_nz.h"
#include "aclnnop/aclnn_npu_format_cast.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)                                                                              \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            Finalize(deviceId, stream);                                                                                \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1L;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(const std::vector<T> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                        void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

// 创建 NZ 格式 weight 的两种方式：
//   方式一（推荐）：ND→NZ 在线转换，适用于 FLOAT4_E2M1 等需要格式转换的场景
//   方式二：直接创建 NZ Tensor（需数据已预转为 NZ layout，见下方 CreateAclTensorNz）

template <typename T>
int CreateAclTensorWithFormat(const std::vector<T> &hostData, const std::vector<int64_t> &shape, int64_t **storageShape,
                              uint64_t *storageShapeSize, void **deviceAddr, aclDataType dataType, aclTensor **tensor,
                              aclFormat format)
{
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, format, *storageShape, *storageShapeSize,
                              *deviceAddr);
    return 0;
}

// 方式一：ND → NZ 在线转换（含 NpuFormatCast + Cast 两步）
//   适用 FLOAT4_E2M1 weight，从 ND 格式转换为亲核 NZ 格式
//   FLOAT4_E2M1 实际按 FLOAT32 存储（每 32bit 装 8 个 F4），
//   NpuFormatCast 后需 Cast 转为正确的 FLOAT4_E2M1 类型
template <typename T>
int CreateAclTensorNzViaCast(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                             aclDataType dataType, aclTensor **tensor, aclrtStream &stream)
{
    // Step 1: 创建 ND 格式的 src Tensor（以 FLOAT32 承载 FLOAT4_E2M1 数据）
    void *srcDeviceAddr = nullptr;
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(&srcDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(srcDeviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    aclTensor *srcTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT32, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), srcDeviceAddr);

    // Step 2: NpuFormatCast 计算 ND→NZ 转换后的 shape 和格式
    int64_t *dstShape = nullptr;
    uint64_t dstShapeSize = 0;
    int actualFormat;
    ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, 29, aclFormat::ACL_FORMAT_ND, &dstShape, &dstShapeSize,
                                                   &actualFormat);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
              return ret);

    // Step 3: 计算 NZ dst 内存大小，申请 dst Tensor
    void *dstDeviceAddr = nullptr;
    uint64_t nzSize = 1;
    for (uint64_t i = 0; i < dstShapeSize; i++) {
        nzSize *= dstShape[i];
    }
    ret = aclrtMalloc(&dstDeviceAddr, nzSize * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Step 4: 调用 NpuFormatCast 执行 ND→NZ 转换
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;
    aclTensor *dstTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT32, nullptr, 0,
                                           static_cast<aclFormat>(actualFormat), dstShape, dstShapeSize, dstDeviceAddr);
    ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

    // Step 5: NpuFormatCast 后 Cast 将 FLOAT32 转为 FLOAT4_E2M1
    //   重建 src tensor 指向 dstDeviceAddr（Cast 的 src 是 FormatCast 的输出）
    aclDestroyTensor(srcTensor);
    aclrtFree(srcDeviceAddr);
    aclDestroyTensor(dstTensor);
    srcTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT32, nullptr, 0,
                                static_cast<aclFormat>(actualFormat), dstShape, dstShapeSize, dstDeviceAddr);
    dstTensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0,
                                static_cast<aclFormat>(actualFormat), dstShape, dstShapeSize, dstDeviceAddr);
    ret = aclnnCastGetWorkspaceSize(srcTensor, static_cast<aclDataType>(dataType), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

    aclDestroyTensor(srcTensor);
    if (workspaceAddr) aclrtFree(workspaceAddr);
    *tensor = dstTensor;
    *deviceAddr = dstDeviceAddr;
    return ACL_SUCCESS;
}

// 方式二：直接创建 NZ Tensor（数据已预转为 NZ layout）
int CreateAclTensorNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    std::vector<int64_t> strides_ = {256L * 4L, 1L, 4L};
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    int64_t E = shape[0];
    int64_t K = shape[1];
    int64_t N = shape[2];
    std::vector<int64_t> shapeNz = {E, N / 64, K / 16, 16L, 64L};

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                              shapeNz.data(), shapeNz.size(), *deviceAddr);
    // aclInitTensor(*tensor, shape.data(), shape.size(), aclDataType::ACL_FLOAT4_E2M1, strides_.data(), 0,
    // aclFormat::ACL_FORMAT_FRACTAL_NZ, shape.data(), shape.size(), &deviceAddr);
    return 0;
}
template <typename T>
int CreateAclTensorListNz(const std::vector<T> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                          void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; ++i) {
        int ret = CreateAclTensorNz<T>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}
void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnGourpedMatmulTest(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 512L;
    int64_t k = 256L;
    int64_t n = 64L;
    int64_t groupnum = 2L;
    std::vector<std::vector<int64_t>> xShape = {{m, k}};
    std::vector<std::vector<int64_t>> weightShape = {{groupnum, k, n}};
    std::vector<std::vector<int64_t>> weightTransposedShape = {{groupnum, k, n}};
    std::vector<std::vector<int64_t>> biasShape = {{groupnum, n}};
    std::vector<std::vector<int64_t>> antiquantScaleShape = {{groupnum, k / 32, n}};
    std::vector<std::vector<int64_t>> pertokenShape = {{m, k / 32}};
    std::vector<std::vector<int64_t>> yShape = {{m, n}};
    std::vector<int64_t> groupListShape = {{groupnum}};
    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *antiquantScaleDeviceAddr = nullptr;
    void *pertokenDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    aclTensorList *x = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *weightTransposed = nullptr;
    aclTensorList *bias = nullptr;
    aclTensor *groupedList = nullptr;
    aclTensorList *scale = nullptr;
    aclTensorList *offset = nullptr;
    aclTensorList *antiquantScale = nullptr;
    aclTensorList *antiquantOffset = nullptr;
    aclTensorList *perTokenScale = nullptr;
    aclTensorList *activationInput = nullptr;
    aclTensorList *activationQuantScale = nullptr;
    aclTensorList *activationQuantOffset = nullptr;
    aclTensorList *out = nullptr;
    aclTensorList *activationFeatureOut = nullptr;
    aclTensorList *dynQuantScaleOut = nullptr;
    int64_t splitItem = 3L;
    int64_t groupType = 0L;
    int64_t groupListType = 0L;
    int64_t actType = 0L;
    std::vector<int8_t> xHostData(m * k, 10);
    std::vector<int8_t> weightHostData(groupnum * k * n / 2, 17);           // 0001 0001
    std::vector<int8_t> weightTransposedHostData(groupnum * k * n / 2, 17); // 0001 0001
    std::vector<int16_t> biasHostData(groupnum * n, 10);
    std::vector<uint16_t> yHostData(m * n, 0);
    std::vector<int8_t> antiquantScaleHostData(groupnum * n, 1);
    std::vector<int8_t> pertokenHostData(m * k / 32, 1);

    std::vector<int64_t> groupListData = {256L, 512L};

    // 创建x aclTensorList
    ret = CreateAclTensorList(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> xTensorPtr(x, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    // 创建weight aclTensorList —— NZ 格式 weight 两种创建方式：
    //   方式一（ND→NZ 在线转换）：CreateAclTensorNzViaCast（适用于 FLOAT4_E2M1 等需要格式转换的场景）
    //   方式二（直接 NZ）：CreateAclTensorListNz（适用于数据已预转为 NZ layout 的场景，当前使用）
    aclTensor *weightTensorSingle = nullptr;
    ret = CreateAclTensorNzViaCast<int8_t>(weightHostData, weightShape[0], &weightDeviceAddr,
                                           aclDataType::ACL_FLOAT4_E2M1, &weightTensorSingle, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    weight = aclCreateTensorList(&weightTensorSingle, 1);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorPtr(weight,
                                                                                           aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    // 创建bias aclTensorList
    ret = CreateAclTensorList(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_BF16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> biasTensorPtr(bias, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
    // 创建pertoken aclTensorList
    ret = CreateAclTensorList(pertokenHostData, pertokenShape, &pertokenDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                              &perTokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> pertokenTensorPtr(perTokenScale,
                                                                                             aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> pertokenDeviceAddrPtr(pertokenDeviceAddr, aclrtFree);
    // 创建antiquantScale aclTensorList
    ret = CreateAclTensorList(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr,
                              aclDataType::ACL_FLOAT8_E8M0, &antiquantScale);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> antiquantTensorPtr(antiquantScale,
                                                                                              aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
    // 创建y aclTensorList
    ret = CreateAclTensorList(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> yTensorPtr(out, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);

    // 创建group_list aclTensor
    ret = CreateAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                   &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupedList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(workspaceAddr, aclrtFree);
    // 调用aclnnGroupedMatmulWeightNz第一段接口
    workspaceSize = 0;
    ret = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, nullptr, groupedList, activationInput,
        activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, nullptr, 0, out,
        activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnGroupedMatmulWeightNz第二段接口
    ret = aclnnGroupedMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    for (int i = 0; i < 1; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t j = 0; j < 10; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
        }
    }
    return ACL_SUCCESS;
}
int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGourpedMatmulTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}