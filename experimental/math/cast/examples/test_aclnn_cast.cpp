/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cast.h"

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
  // 固定写法，初始化
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
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void PrintValue(float value) {
  LOG_PRINT("%f", static_cast<double>(value));
}

void PrintValue(double value) {
  LOG_PRINT("%f", value);
}

void PrintValue(bool value) {
  LOG_PRINT("%d", static_cast<int>(value));
}

template <typename T>
void PrintValue(T value) {
  LOG_PRINT("%lld", static_cast<long long>(value));
}

template <typename InT, typename OutT>
int RunCastCase(const std::string& name, const std::vector<int64_t>& shape, const std::vector<InT>& selfHostData,
                aclDataType selfType, aclDataType outType, aclrtStream stream) {
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  std::vector<OutT> outHostData(GetShapeSize(shape), static_cast<OutT>(0));

  // 创建self aclTensor
  auto ret = CreateAclTensor(selfHostData, shape, &selfDeviceAddr, selfType, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, outType, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // 调用aclnnCast第一段接口
  ret = aclnnCastGetWorkspaceSize(self, outType, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > static_cast<uint64_t>(0)) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnCast第二段接口
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(shape);
  std::vector<OutT> resultData(size, static_cast<OutT>(0));
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(OutT), outDeviceAddr, size * sizeof(OutT),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("[%s]\n", name.c_str());
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: ", i);
    PrintValue(resultData[i]);
    LOG_PRINT("\n");
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. 释放device 资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > static_cast<uint64_t>(0)) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API文档
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 默认跑当前 310P camodel 已验证的 smoke case；完整多 dtype case 在真机或完整镜像中打开。
  std::vector<int64_t> shape = {4, 2};
  ret = RunCastCase<double, float>("double_to_float_smoke", shape, {0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1},
                                   ACL_DOUBLE, ACL_FLOAT, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  if (std::getenv("CAST_EXAMPLE_FULL") != nullptr) {
    ret = RunCastCase<float, int32_t>("float_to_int32", shape,
                                      {-3.5f, -1.0f, 0.0f, 1.4f, 2.5f, 3.9f, 8.0f, 16.0f}, ACL_FLOAT, ACL_INT32,
                                      stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = RunCastCase<int32_t, float>("int32_to_float", shape, {-8, -1, 0, 1, 2, 3, 127, 1024}, ACL_INT32,
                                      ACL_FLOAT, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = RunCastCase<int8_t, float>("int8_to_float", shape, {-8, -4, -1, 0, 1, 2, 7, 12}, ACL_INT8, ACL_FLOAT,
                                     stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = RunCastCase<uint8_t, uint8_t>("uint8_to_bool", shape, {0, 1, 2, 3, 4, 5, 6, 7}, ACL_UINT8, ACL_BOOL,
                                       stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = RunCastCase<uint8_t, int32_t>("bool_to_int32", shape, {0, 1, 1, 0, 1, 0, 1, 0}, ACL_BOOL, ACL_INT32,
                                        stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }

  // 7. 释放device 资源
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
