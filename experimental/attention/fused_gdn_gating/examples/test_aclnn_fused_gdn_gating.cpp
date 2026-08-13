/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_fused_gdn_gating.cpp
 * \brief Standalone aclnn example for FusedGdnGating on 910B (BF16_FLOAT) and 310P (FP16).
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_gdn_gating.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

void PrintOutHalfResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<uint16_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("half_result[%ld] is: 0x%04x\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
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

uint16_t FloatToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  uint16_t sign = (x >> 16) & 0x8000;
  int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = x & 0x7FFFFF;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7C00;
  return sign | (exp << 10) | (mantissa >> 13);
}

int DoFusedGdnGating(aclrtStream stream, aclTensor* aLog, aclTensor* a, aclTensor* b,
                     aclTensor* dtBias, float beta, float threshold,
                     aclTensor* g, aclTensor* betaOutput) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  auto ret = aclnnFusedGdnGatingGetWorkspaceSize(aLog, a, b, dtBias, beta, threshold,
                                                  g, betaOutput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedGdnGatingGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnFusedGdnGating(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedGdnGating failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return 0;
}

int test_910b_bf16_float(aclrtStream stream) {
  LOG_PRINT("=== Running 910B BF16_FLOAT test ===\n");
  int64_t batch = 4;
  int64_t numHeads = 8;

  std::vector<int64_t> aLogShape = {numHeads};
  std::vector<int64_t> aShape = {batch, numHeads};
  std::vector<int64_t> bShape = {batch, numHeads};
  std::vector<int64_t> dtBiasShape = {numHeads};
  std::vector<int64_t> gShape = {1, batch, numHeads};
  std::vector<int64_t> betaOutputShape = {1, batch, numHeads};

  void* aLogDeviceAddr = nullptr;
  void* aDeviceAddr = nullptr;
  void* bDeviceAddr = nullptr;
  void* dtBiasDeviceAddr = nullptr;
  void* gDeviceAddr = nullptr;
  void* betaOutputDeviceAddr = nullptr;

  aclTensor* aLog = nullptr;
  aclTensor* a = nullptr;
  aclTensor* b = nullptr;
  aclTensor* dtBias = nullptr;
  aclTensor* g = nullptr;
  aclTensor* betaOutput = nullptr;

  std::vector<float> aLogHostData(numHeads, 0.5);
  std::vector<uint16_t> aHostData(batch * numHeads, 0);
  std::vector<uint16_t> bHostData(batch * numHeads, 0);
  std::vector<float> dtBiasHostData(numHeads, 0.1);
  std::vector<float> gHostData(batch * numHeads, 0);
  std::vector<uint16_t> betaOutputHostData(batch * numHeads, 0);

  auto ret = CreateAclTensor(aLogHostData, aLogShape, &aLogDeviceAddr, aclDataType::ACL_FLOAT, &aLog);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_BF16, &a);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(bHostData, bShape, &bDeviceAddr, aclDataType::ACL_BF16, &b);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dtBiasHostData, dtBiasShape, &dtBiasDeviceAddr, aclDataType::ACL_FLOAT, &dtBias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gHostData, gShape, &gDeviceAddr, aclDataType::ACL_FLOAT, &g);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(betaOutputHostData, betaOutputShape, &betaOutputDeviceAddr, aclDataType::ACL_BF16, &betaOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = DoFusedGdnGating(stream, aLog, a, b, dtBias, 1.0f, 20.0f, g, betaOutput);
  if (ret != 0) {
    LOG_PRINT("910B BF16_FLOAT test failed, SOC may not be 910B.\n");
  } else {
    LOG_PRINT("g output:\n");
    PrintOutResult(gShape, &gDeviceAddr);
    LOG_PRINT("910B BF16_FLOAT test passed.\n");
  }

  aclDestroyTensor(aLog);
  aclDestroyTensor(a);
  aclDestroyTensor(b);
  aclDestroyTensor(dtBias);
  aclDestroyTensor(g);
  aclDestroyTensor(betaOutput);
  aclrtFree(aLogDeviceAddr);
  aclrtFree(aDeviceAddr);
  aclrtFree(bDeviceAddr);
  aclrtFree(dtBiasDeviceAddr);
  aclrtFree(gDeviceAddr);
  aclrtFree(betaOutputDeviceAddr);
  return ret;
}

int test_310p_fp16(aclrtStream stream) {
  LOG_PRINT("=== Running 310P FP16 test ===\n");
  int64_t batch = 4;
  int64_t numHeads = 8;

  std::vector<int64_t> aLogShape = {numHeads};
  std::vector<int64_t> aShape = {batch, numHeads};
  std::vector<int64_t> bShape = {batch, numHeads};
  std::vector<int64_t> dtBiasShape = {numHeads};
  std::vector<int64_t> gShape = {1, batch, numHeads};
  std::vector<int64_t> betaOutputShape = {1, batch, numHeads};

  void* aLogDeviceAddr = nullptr;
  void* aDeviceAddr = nullptr;
  void* bDeviceAddr = nullptr;
  void* dtBiasDeviceAddr = nullptr;
  void* gDeviceAddr = nullptr;
  void* betaOutputDeviceAddr = nullptr;

  aclTensor* aLog = nullptr;
  aclTensor* a = nullptr;
  aclTensor* b = nullptr;
  aclTensor* dtBias = nullptr;
  aclTensor* g = nullptr;
  aclTensor* betaOutput = nullptr;

  std::vector<uint16_t> aLogHostData(numHeads, FloatToHalf(0.5));
  std::vector<uint16_t> aHostData(batch * numHeads, FloatToHalf(0.3));
  std::vector<uint16_t> bHostData(batch * numHeads, FloatToHalf(0.2));
  std::vector<uint16_t> dtBiasHostData(numHeads, FloatToHalf(0.1));
  std::vector<float> gHostData(batch * numHeads, 0);
  std::vector<uint16_t> betaOutputHostData(batch * numHeads, 0);

  auto ret = CreateAclTensor(aLogHostData, aLogShape, &aLogDeviceAddr, aclDataType::ACL_FLOAT16, &aLog);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_FLOAT16, &a);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(bHostData, bShape, &bDeviceAddr, aclDataType::ACL_FLOAT16, &b);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dtBiasHostData, dtBiasShape, &dtBiasDeviceAddr, aclDataType::ACL_FLOAT16, &dtBias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gHostData, gShape, &gDeviceAddr, aclDataType::ACL_FLOAT, &g);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(betaOutputHostData, betaOutputShape, &betaOutputDeviceAddr, aclDataType::ACL_FLOAT16, &betaOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = DoFusedGdnGating(stream, aLog, a, b, dtBias, 1.0f, 20.0f, g, betaOutput);
  if (ret != 0) {
    LOG_PRINT("310P FP16 test failed.\n");
  } else {
    LOG_PRINT("g output:\n");
    PrintOutResult(gShape, &gDeviceAddr);
    LOG_PRINT("betaOutput output:\n");
    PrintOutHalfResult(betaOutputShape, &betaOutputDeviceAddr);
    LOG_PRINT("310P FP16 test passed.\n");
  }

  aclDestroyTensor(aLog);
  aclDestroyTensor(a);
  aclDestroyTensor(b);
  aclDestroyTensor(dtBias);
  aclDestroyTensor(g);
  aclDestroyTensor(betaOutput);
  aclrtFree(aLogDeviceAddr);
  aclrtFree(aDeviceAddr);
  aclrtFree(bDeviceAddr);
  aclrtFree(dtBiasDeviceAddr);
  aclrtFree(gDeviceAddr);
  aclrtFree(betaOutputDeviceAddr);
  return ret;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  ret = test_910b_bf16_float(stream);
  if (ret != 0) {
    LOG_PRINT("910B test failed or not supported, falling back to 310P FP16 test.\n");
    test_310p_fp16(stream);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
