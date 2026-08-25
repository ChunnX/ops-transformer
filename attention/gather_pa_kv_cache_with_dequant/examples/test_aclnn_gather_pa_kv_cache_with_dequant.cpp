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
 * \file test_aclnn_gather_pa_kv_cache_with_dequant.cpp
 * \brief Example test for fused GatherPaKvCache + Dequant operator
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gather_pa_kv_cache_with_dequant.h"

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

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // Test configuration
  // num_blocks=2, block_size=4, num_heads_k=8, head_dim_k=16
  // num_heads_v=8, head_dim_v=16
  // batch=2, max_seq_len=8
  std::vector<int64_t> keyCacheShape = {2, 4, 8, 16};      // [num_blocks, block_size, num_heads_k, head_dim_k]
  std::vector<int64_t> keyScaleShape = {2, 4, 8};          // [num_blocks, block_size, num_heads_k]
  std::vector<int64_t> valueCacheShape = {2, 4, 8, 16};    // [num_blocks, block_size, num_heads_v, head_dim_v]
  std::vector<int64_t> valueScaleShape = {8, 16};          // [num_heads_v, head_dim_v]
  std::vector<int64_t> blockTablesShape = {2, 2};          // [batch, max_blocks_per_seq]
  std::vector<int64_t> seqLensShape = {2};                 // [batch]
  std::vector<int64_t> keyShape = {10, 8, 16};             // [num_tokens, num_heads_k, head_dim_k]
  std::vector<int64_t> valueShape = {10, 8, 16};           // [num_tokens, num_heads_v, head_dim_v]

  void* keyCacheDeviceAddr = nullptr;
  void* keyScaleDeviceAddr = nullptr;
  void* valueCacheDeviceAddr = nullptr;
  void* valueScaleDeviceAddr = nullptr;
  void* blockTablesDeviceAddr = nullptr;
  void* seqLensDeviceAddr = nullptr;
  void* keyDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;

  aclTensor* keyCache = nullptr;
  aclTensor* keyScaleCache = nullptr;
  aclTensor* valueCache = nullptr;
  aclTensor* valueScale = nullptr;
  aclTensor* blockTables = nullptr;
  aclTensor* seqLens = nullptr;
  aclTensor* key = nullptr;
  aclTensor* value = nullptr;

  // Initialize test data
  std::vector<int8_t> keyCacheHostData(GetShapeSize(keyCacheShape), 1);
  std::vector<float> keyScaleHostData(GetShapeSize(keyScaleShape), 0.5f);
  std::vector<uint8_t> valueCacheHostData(GetShapeSize(valueCacheShape), 2);
  std::vector<float> valueScaleHostData(GetShapeSize(valueScaleShape), 0.25f);
  std::vector<int32_t> blockTablesHostData = {0, 1, 1, 0};  // batch 0 uses blocks [0,1], batch 1 uses blocks [1,0]
  std::vector<int32_t> seqLensHostData = {6, 4};            // batch 0 has 6 tokens, batch 1 has 4 tokens
  std::vector<uint16_t> keyHostData(GetShapeSize(keyShape), 0);
  std::vector<uint16_t> valueHostData(GetShapeSize(valueShape), 0);

  // Create tensors
  ret = CreateAclTensor(keyCacheHostData, keyCacheShape, &keyCacheDeviceAddr, ACL_INT8, &keyCache);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create keyCache tensor failed\n"); return ret);

  ret = CreateAclTensor(keyScaleHostData, keyScaleShape, &keyScaleDeviceAddr, ACL_FLOAT, &keyScaleCache);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create keyScaleCache tensor failed\n"); return ret);

  ret = CreateAclTensor(valueCacheHostData, valueCacheShape, &valueCacheDeviceAddr, ACL_FLOAT8_E4M3FN, &valueCache);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create valueCache tensor failed\n"); return ret);

  ret = CreateAclTensor(valueScaleHostData, valueScaleShape, &valueScaleDeviceAddr, ACL_FLOAT, &valueScale);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create valueScale tensor failed\n"); return ret);

  ret = CreateAclTensor(blockTablesHostData, blockTablesShape, &blockTablesDeviceAddr, ACL_INT32, &blockTables);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create blockTables tensor failed\n"); return ret);

  ret = CreateAclTensor(seqLensHostData, seqLensShape, &seqLensDeviceAddr, ACL_INT32, &seqLens);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create seqLens tensor failed\n"); return ret);

  ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, ACL_BF16, &key);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create key tensor failed\n"); return ret);

  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, ACL_BF16, &value);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create value tensor failed\n"); return ret);

  // Call the fused operator
  char* cacheMode = "Norm";
  const bool isSeqLensCumsum = false;
  
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  ret = aclnnGatherPaKvCacheWithDequantGetWorkspaceSize(
      keyCache, keyScaleCache, valueCache, valueScale,
      blockTables, seqLens, key, value,
      nullptr,  // seqOffsetOptional
      cacheMode, isSeqLensCumsum,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherPaKvCacheWithDequantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnGatherPaKvCacheWithDequant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherPaKvCacheWithDequant failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Copy results back to host
  auto keySize = GetShapeSize(keyShape) * sizeof(uint16_t);
  ret = aclrtMemcpy(keyHostData.data(), keySize, keyDeviceAddr, keySize, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy key result failed. ERROR: %d\n", ret); return ret);

  auto valueSize = GetShapeSize(valueShape) * sizeof(uint16_t);
  ret = aclrtMemcpy(valueHostData.data(), valueSize, valueDeviceAddr, valueSize, ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy value result failed. ERROR: %d\n", ret); return ret);

  // Print some results
  LOG_PRINT("Key output (first 10 elements):\n");
  for (int i = 0; i < 10 && i < keyHostData.size(); i++) {
    LOG_PRINT("  key[%d] = 0x%04x\n", i, keyHostData[i]);
  }

  LOG_PRINT("Value output (first 10 elements):\n");
  for (int i = 0; i < 10 && i < valueHostData.size(); i++) {
    LOG_PRINT("  value[%d] = 0x%04x\n", i, valueHostData[i]);
  }

  // Cleanup
  aclDestroyTensor(keyCache);
  aclDestroyTensor(keyScaleCache);
  aclDestroyTensor(valueCache);
  aclDestroyTensor(valueScale);
  aclDestroyTensor(blockTables);
  aclDestroyTensor(seqLens);
  aclDestroyTensor(key);
  aclDestroyTensor(value);

  aclrtFree(keyCacheDeviceAddr);
  aclrtFree(keyScaleDeviceAddr);
  aclrtFree(valueCacheDeviceAddr);
  aclrtFree(valueScaleDeviceAddr);
  aclrtFree(blockTablesDeviceAddr);
  aclrtFree(seqLensDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  LOG_PRINT("Test completed successfully!\n");
  return 0;
}
