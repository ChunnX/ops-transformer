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
 * \file test_aclnn_flash_attention_score_grad_v4_tnd_deter_causal_right_down.cpp
 * \brief TND deterministic RIGHT_DOWN_CAUSAL (sparseMode=3, S1!=S2). Expect DeterType=5, isTndSwizzle=1.
 *        Type 5 only when swizzle on; S1==S2 would stay DeterType=3.
 *        Tiling-only: stops after GetWorkspaceSize (no kernel) for fast iteration.
 *        Plog: isExceedL2Cache / enableSwizzle / isTndSwizzle / isDeterministic[5] / tiling is
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
  int64_t det = 1;
  ret = aclrtCtxSetSysParamOpt(ACL_OPT_DETERMINISTIC, det);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCtxSetSysParamOpt deterministic failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

// Device-only alloc (memset), skip host H2D — enough for tiling / GetWorkspaceSize.
int CreateAclTensorZero(const std::vector<int64_t>& shape, size_t elementSize, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
  auto size = static_cast<size_t>(GetShapeSize(shape)) * elementSize;
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemset(*deviceAddr, size, 0, size);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemset failed. ERROR: %d\n", ret); return ret);

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

  // RIGHT_DOWN + S1!=S2 + swizzle → DeterType=5. Moderate size; unequal Q/KV keep TND + type5.
  int64_t N1 = 8;
  int64_t N2 = 8;
  int64_t T1 = 6144;   // 2048 + 4096
  int64_t T2 = 9216;   // 3072 + 6144
  int64_t D = 128;

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

  ret = CreateAclTensorZero(qShape, sizeof(float), &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorZero(kShape, sizeof(float), &kDeviceAddr, aclDataType::ACL_FLOAT, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorZero(vShape, sizeof(float), &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorZero(dxShape, sizeof(float), &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorZero(attenmaskShape, sizeof(uint8_t), &attenmaskDeviceAddr, aclDataType::ACL_UINT8, &attenmask);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorZero(softmaxMaxShape, sizeof(float), &softmaxMaxDeviceAddr, aclDataType::ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorZero(softmaxSumShape, sizeof(float), &softmaxSumDeviceAddr, aclDataType::ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorZero(attentionInShape, sizeof(float), &attentionInDeviceAddr, aclDataType::ACL_FLOAT, &attentionIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorZero(dqShape, sizeof(float), &dqDeviceAddr, aclDataType::ACL_FLOAT, &dq);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorZero(dkShape, sizeof(float), &dkDeviceAddr, aclDataType::ACL_FLOAT, &dk);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorZero(dvShape, sizeof(float), &dvDeviceAddr, aclDataType::ACL_FLOAT, &dv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> prefixOp = {0};
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  std::vector<int64_t> actualSeqQLenOp = {2048, 6144};   // per-batch: 2048, 4096
  std::vector<int64_t> actualSeqKVLenOp = {3072, 9216};  // per-batch: 3072, 6144 (S1!=S2)

  aclIntArray *prefix = aclCreateIntArray(prefixOp.data(), 1);
  aclIntArray *qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray *kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);
  aclIntArray* actualSeqQLen = aclCreateIntArray(actualSeqQLenOp.data(), 2);
  aclIntArray* actualSeqKVLen = aclCreateIntArray(actualSeqKVLenOp.data(), 2);

  double scaleValue = 1.0 / sqrt(128);
  double keepProb = 1.0;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = 8;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 3;
  int64_t pseType = 1;
  int64_t outDtype = 1;
  int64_t seed = 0;
  int64_t offset = 0;
  char inputlayOut[5] = {'T', 'N', 'D', 0};

  LOG_PRINT("[FAG-TND-DETER] causal_right_down tiling-only: N=%ld T1/T2=%ld/%ld sparseMode=%ld expect "
            "DeterType=5 enableSwizzle=1 isTndSwizzle=1\n",
            N1, T1, T2, sparseMode);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // API order: sparseMode, pseType, seed, offset, outDtype
  ret = aclnnFlashAttentionScoreGradV4GetWorkspaceSize(
      q, k, v, dx, pse, dropMask, padding, attenmask, softmaxMax, softmaxSum, softmaxIn, attentionIn, nullptr,
      queryRope, keyRope, dScaleQ, dScaleK, dScaleV, dScaleDy, dScaleO, prefix, actualSeqQLen, actualSeqKVLen,
      qStartIdx, kvStartIdx, scaleValue, keepProb, preTokens, nextTokens, headNum, inputlayOut, nullptr, innerPrecise,
      sparseMode, pseType, seed, offset, outDtype, dq, dk, dv, dqRope, dkRope, dpse, nullptr, &workspaceSize,
      &executor);
  if (ret != ACL_SUCCESS) {
    const char* errMsg = aclGetRecentErrMsg();
    LOG_PRINT("aclnnFlashAttentionScoreGradV4GetWorkspaceSize failed. ERROR: %d\n", ret);
    LOG_PRINT("aclGetRecentErrMsg: %s\n", errMsg != nullptr ? errMsg : "(null)");
    LOG_PRINT("Also grep plog for: Tiling failed|OP_LOGE|invalid|EZ1001|FlashAttentionScoreGrad\n");
    return ret;
  }

  LOG_PRINT("[FAG-TND-DETER] GetWorkspaceSize ok, workspaceSize=%lu. Skip kernel.\n", workspaceSize);
  LOG_PRINT("[FAG-TND-DETER] grep plog: isExceedL2Cache / enableSwizzle / isTndSwizzle / "
            "isDeterministic\\[5\\] / tiling is\n");

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
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
