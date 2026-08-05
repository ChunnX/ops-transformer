/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_flash_attention_score_grad_v4_tnd_deter_band.cpp
 * \brief TND deterministic example for TND deter BAND. Expect DeterType=4.
 *        Check plog: tiling is / isTndSwizzle / isDeterministic[4]
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score_grad.h"

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
  // Enable deterministic for TND deter / swizzle path
  int64_t det = 1;
  ret = aclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC, det);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCtxSetSysParamOpt deterministic failed. ERROR: %d\n", ret); return ret);
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
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
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

  // Case: band, expect DeterType=4, IsTndSwizzle=1
  // Large shape needed for enableSwizzle (exceed L2 / fill aicNum).
  int64_t N1 = 8;
  int64_t N2 = 8;
  int64_t T1 = 12288;  // 4096 + 8192
  int64_t T2 = 12288;
  int64_t D = 128;

  int64_t q_size = T1 * N1 * D;
  int64_t kv_size = T2 * N2 * D;
  int64_t atten_mask_size = 2048 * 2048;
  int64_t softmax_size = T1 * N1 * 8;

  std::vector<int64_t> qShape = {T1, N1, D};
  std::vector<int64_t> kShape = {T2, N2, D};
  std::vector<int64_t> vShape = {T2, N2, D};
  std::vector<int64_t> dxShape = {T1, N1, D};
  std::vector<int64_t> attenmaskShape = {2048, 2048};
  std::vector<int64_t> softmaxMaxShape = {T1, N1, 8};
  std::vector<int64_t> softmaxSumShape = {T1, N1, 8};
  std::vector<int64_t> attentionInShape = {T1, N1, D};
  std::vector<int64_t> dqShape = {T1, N1, D};
  std::vector<int64_t> dkShape = {T2, N2, D};
  std::vector<int64_t> dvShape = {T2, N2, D};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* attenmaskDeviceAddr = nullptr;
  void* softmaxMaxDeviceAddr = nullptr;
  void* softmaxSumDeviceAddr = nullptr;
  void* attentionInDeviceAddr = nullptr;
  void* dqDeviceAddr = nullptr;
  void* dkDeviceAddr = nullptr;
  void* dvDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* pse = nullptr;
  aclTensor* dropMask = nullptr;
  aclTensor* padding = nullptr;
  aclTensor* attenmask = nullptr;
  aclTensor* queryRope = nullptr;
  aclTensor* keyRope = nullptr;
  aclTensor* dScaleQ = nullptr;
  aclTensor* dScaleK = nullptr;
  aclTensor* dScaleV = nullptr;
  aclTensor* dScaleDy = nullptr;
  aclTensor* dScaleO = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxIn = nullptr;
  aclTensor* attentionIn = nullptr;
  aclTensor* dq = nullptr;
  aclTensor* dk = nullptr;
  aclTensor* dv = nullptr;
  aclTensor* dpse = nullptr;
  aclTensor* dqRope = nullptr;
  aclTensor* dkRope = nullptr;

  std::vector<float> qHostData(q_size, 1.0f);
  std::vector<float> kHostData(kv_size, 1.0f);
  std::vector<float> vHostData(kv_size, 1.0f);
  std::vector<float> dxHostData(q_size, 1.0f);
  std::vector<uint8_t> attenmaskHostData(atten_mask_size, 0);
  std::vector<float> softmaxMaxHostData(softmax_size, 3.0f);
  std::vector<float> softmaxSumHostData(softmax_size, 3.0f);
  std::vector<float> attentionInHostData(q_size, 1.0f);
  std::vector<float> dqHostData(q_size, 0.0f);
  std::vector<float> dkHostData(kv_size, 0.0f);
  std::vector<float> dvHostData(kv_size, 0.0f);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_FLOAT, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attenmaskHostData, attenmaskShape, &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxMaxHostData, softmaxMaxShape, &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxSumHostData, softmaxSumShape, &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attentionInHostData, attentionInShape, &attentionInDeviceAddr, aclDataType::ACL_FLOAT, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dqHostData, dqShape, &dqDeviceAddr, aclDataType::ACL_FLOAT, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dkHostData, dkShape, &dkDeviceAddr, aclDataType::ACL_FLOAT, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dvHostData, dvShape, &dvDeviceAddr, aclDataType::ACL_FLOAT, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> prefixOp = {0};
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  std::vector<int64_t> actualSeqQLenOp = {4096, 12288};  // per-batch: 4096, 8192
  std::vector<int64_t> actualSeqKVLenOp = {4096, 12288};

  aclIntArray *prefix = aclCreateIntArray(prefixOp.data(), 1);
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  aclIntArray* actualSeqQLen = aclCreateIntArray(actualSeqQLenOp.data(), 2);
  aclIntArray* actualSeqKVLen = aclCreateIntArray(actualSeqKVLenOp.data(), 2);

  double scaleValue = 1.0 / sqrt(128);
  double keepProb = 1.0;
  int64_t preTokens = 1024;
  int64_t nextTokens = 1024;
  int64_t headNum = 8;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 4;
  int64_t pseType = 1;
  int64_t outDtype = 1;
  int64_t seed = 0;
  int64_t offset = 0;
  char inputlayOut[5] = {'T', 'N', 'D', 0};

  LOG_PRINT("[FAG-TND-DETER] case=band, sparseMode=%ld, pre/next=%ld/%ld, T1/T2=%ld/%ld, expectDeterType=4\n",
            sparseMode, preTokens, nextTokens, T1, T2);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnFlashAttentionScoreGradV4GetWorkspaceSize(
      q, k, v, dx, pse, dropMask, padding, attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, nullptr,
      queryRope, keyRope, dScaleQ, dScaleK, dScaleV, dScaleDy, dScaleO, prefix, actualSeqQLen, actualSeqKVLen,
      qStartIdx, kvStartIdx, scaleValue, keepProb, preTokens, nextTokens, headNum, inputlayOut, nullptr, innerPrecise,
      sparseMode, outDtype, pseType, seed, offset, dq, dk, dv, dqRope, dkRope, dpse, nullptr, &workspaceSize,
      &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnFlashAttentionScoreGradV4GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnFlashAttentionScoreGradV4(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlashAttentionScoreGradV4 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("[FAG-TND-DETER] case=band finished. Check plog for: tiling is / isTndSwizzle / isDeterministic[4]\n");

  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(dx);
  aclDestroyTensor(attenmask);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyTensor(attentionIn);
  aclDestroyTensor(dq);
  aclDestroyTensor(dk);
  aclDestroyTensor(dv);
  aclDestroyIntArray(prefix);
  aclDestroyIntArray(qStartIdx);
  aclDestroyIntArray(kvStartIdx);
  aclDestroyIntArray(actualSeqQLen);
  aclDestroyIntArray(actualSeqKVLen);

  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(attenmaskDeviceAddr);
  aclrtFree(softmaxMaxDeviceAddr);
  aclrtFree(softmaxSumDeviceAddr);
  aclrtFree(attentionInDeviceAddr);
  aclrtFree(dqDeviceAddr);
  aclrtFree(dkDeviceAddr);
  aclrtFree(dvDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
