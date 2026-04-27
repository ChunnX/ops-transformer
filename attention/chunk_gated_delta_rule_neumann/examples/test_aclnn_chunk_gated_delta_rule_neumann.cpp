/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_gated_delta_rule.h"

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

void PrintBf16Result(std::vector<int64_t>& shape, void** deviceAddr, int maxPrint = 10) {
  auto size = GetShapeSize(shape);
  std::vector<uint16_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(uint16_t),
                         *deviceAddr, size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return);
  
  int printCount = std::min((int64_t)maxPrint, size);
  for (int i = 0; i < printCount; i++) {
    uint16_t bf16_val = resultData[i];
    uint32_t float_bits = ((uint32_t)bf16_val) << 16;
    float float_val;
    memcpy(&float_val, &float_bits, sizeof(float));
    LOG_PRINT("result[%d] = %f (bf16: 0x%04x)\n", i, float_val, bf16_val);
  }
  LOG_PRINT("Total elements: %ld, printed %d\n", size, printCount);
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int64_t B = 2;
  int64_t T = 128;
  int64_t nk = 4;
  int64_t nv = 4;
  int64_t dk = 64;
  int64_t dv = 64;
  float scaleValue = 1.0f;
  int64_t hasGamma = 1;

  std::vector<int64_t> queryShape = {T, nk, dk};
  std::vector<int64_t> keyShape = {T, nk, dk};
  std::vector<int64_t> valueShape = {T, nv, dv};
  std::vector<int64_t> betaShape = {T, nv};
  std::vector<int64_t> initialStateShape = {B, nv, dv, dk};
  std::vector<int64_t> seqlensShape = {B};
  std::vector<int64_t> gShape = {T, nv};
  std::vector<int64_t> outShape = {T, nv, dv};
  std::vector<int64_t> finalStateShape = {B, nv, dv, dk};

  int64_t querySize = GetShapeSize(queryShape);
  int64_t keySize = GetShapeSize(keyShape);
  int64_t valueSize = GetShapeSize(valueShape);
  int64_t betaSize = GetShapeSize(betaShape);
  int64_t stateSize = GetShapeSize(initialStateShape);
  int64_t seqlensSize = GetShapeSize(seqlensShape);
  int64_t gSize = GetShapeSize(gShape);
  int64_t outSize = GetShapeSize(outShape);

  void* queryDeviceAddr = nullptr;
  void* keyDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* initialStateDeviceAddr = nullptr;
  void* seqlensDeviceAddr = nullptr;
  void* gDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* finalStateDeviceAddr = nullptr;

  aclTensor* queryTensor = nullptr;
  aclTensor* keyTensor = nullptr;
  aclTensor* valueTensor = nullptr;
  aclTensor* betaTensor = nullptr;
  aclTensor* initialStateTensor = nullptr;
  aclTensor* seqlensTensor = nullptr;
  aclTensor* gTensor = nullptr;
  aclTensor* outTensor = nullptr;
  aclTensor* finalStateTensor = nullptr;

  std::vector<uint16_t> queryHostData(querySize, 0x3f80);
  std::vector<uint16_t> keyHostData(keySize, 0x3f80);
  std::vector<uint16_t> valueHostData(valueSize, 0x3f80);
  std::vector<uint16_t> betaHostData(betaSize, 0x3f80);
  std::vector<uint16_t> stateHostData(stateSize, 0x3f80);
  std::vector<float> gHostData(gSize, -0.5f);
  std::vector<int32_t> seqlensHostData = {64, 64};
  std::vector<uint16_t> outHostData(outSize, 0);
  std::vector<uint16_t> finalStateHostData(stateSize, 0);

  LOG_PRINT("Creating tensors...\n");

  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_BF16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create query tensor failed. ERROR: %d\n", ret); return ret);
  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_BF16, &keyTensor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create key tensor failed. ERROR: %d\n", ret); return ret);
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_BF16, &valueTensor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create value tensor failed. ERROR: %d\n", ret); return ret);
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_BF16, &betaTensor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create beta tensor failed. ERROR: %d\n", ret); return ret);
  ret = CreateAclTensor(stateHostData, initialStateShape, &initialStateDeviceAddr, aclDataType::ACL_BF16, &initialStateTensor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create initialState tensor failed. ERROR: %d\n", ret); return ret);
  ret = CreateAclTensor(seqlensHostData, seqlensShape, &seqlensDeviceAddr, aclDataType::ACL_INT32, &seqlensTensor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create seqlens tensor failed. ERROR: %d\n", ret); return ret);

  if (hasGamma) {
    ret = CreateAclTensor(gHostData, gShape, &gDeviceAddr, aclDataType::ACL_FLOAT, &gTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create g tensor failed. ERROR: %d\n", ret); return ret);
  }

  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create out tensor failed. ERROR: %d\n", ret); return ret);
  ret = CreateAclTensor(finalStateHostData, finalStateShape, &finalStateDeviceAddr, aclDataType::ACL_BF16, &finalStateTensor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create finalState tensor failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("All tensors created successfully.\n");
  LOG_PRINT("Input shapes: query(%ld,%ld,%ld), key(%ld,%ld,%ld), value(%ld,%ld,%ld)\n",
            T, nk, dk, T, nk, dk, T, nv, dv);
  LOG_PRINT("State shape: (%ld,%ld,%ld,%ld)\n", B, nv, dv, dk);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  LOG_PRINT("Calling aclnnChunkGatedDeltaRuleNeumannGetWorkspaceSize...\n");
  
  ret = aclnnChunkGatedDeltaRuleNeumannGetWorkspaceSize(
      queryTensor, keyTensor, valueTensor, betaTensor,
      initialStateTensor, seqlensTensor, gTensor, scaleValue,
      outTensor, finalStateTensor, &workspaceSize, &executor);
  
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclnnChunkGatedDeltaRuleNeumannGetWorkspaceSize failed. ERROR: %d\n", ret);
    return ret;
  }

  LOG_PRINT("Workspace size: %lu bytes\n", workspaceSize);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  LOG_PRINT("Calling aclnnChunkGatedDeltaRuleNeumann...\n");
  ret = aclnnChunkGatedDeltaRuleNeumann(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkGatedDeltaRuleNeumann failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("Synchronizing stream...\n");
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("\n=== Output Results ===\n");
  LOG_PRINT("Output (out):\n");
  PrintBf16Result(outShape, &outDeviceAddr, 5);
  LOG_PRINT("\nFinal State:\n");
  PrintBf16Result(finalStateShape, &finalStateDeviceAddr, 5);

  LOG_PRINT("\nReleasing resources...\n");
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(betaTensor);
  aclDestroyTensor(initialStateTensor);
  aclDestroyTensor(seqlensTensor);
  if (gTensor) aclDestroyTensor(gTensor);
  aclDestroyTensor(outTensor);
  aclDestroyTensor(finalStateTensor);

  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(initialStateDeviceAddr);
  aclrtFree(seqlensDeviceAddr);
  if (gDeviceAddr) aclrtFree(gDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(finalStateDeviceAddr);
  if (workspaceSize > 0) aclrtFree(workspaceAddr);

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  LOG_PRINT("Test completed successfully!\n");
  return 0;
}