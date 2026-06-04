/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_flash_attention_workspace.cpp
 * \brief Test workspace size for L2 cache optimization — phase 1 only (GetWorkspaceSize).
 *
 * Cases:
 *   SBH:  100 random — B in [5,7], S1 in [30001,80000] %32!=0, S2 in [180,256] %32!=0, headNum=1, D=128
 *   BSND:  40 random — B in [1,16], S1,S2 in [1025,8192] %32!=0 (not both >3000), N in (32,128] %16!=0, D=128
 *   BNSD:  40 random — same params as BSND, layout=BNSD
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score.h"

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

int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(float);
  if (dataType == ACL_BF16 || dataType == ACL_FLOAT16) {
    size = GetShapeSize(shape) * sizeof(uint16_t);
  } else if (dataType == ACL_UINT8) {
    size = GetShapeSize(shape) * sizeof(uint8_t);
  }
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

// Random integer in [low, high] that is NOT a multiple of mod
static int64_t RandNotMod(int64_t low, int64_t high, int64_t mod) {
  int64_t x = low + rand() % (high - low + 1);
  if (x % mod == 0) {
    x++;
    if (x > high) x = high - 1;
  }
  return x;
}

static int TestWorkspaceSize(const char* label,
                             const std::vector<int64_t>& qShape,
                             const std::vector<int64_t>& kShape,
                             const std::vector<int64_t>& vShape,
                             int64_t headNum, char* layout,
                             aclDataType dtype) {
  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* attentionOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  void* qAddr = nullptr;
  void* kAddr = nullptr;
  void* vAddr = nullptr;
  void* outAddr = nullptr;
  void* smMaxAddr = nullptr;
  void* smSumAddr = nullptr;

  int ret = CreateAclTensor(qShape, &qAddr, dtype, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kShape, &kAddr, dtype, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vShape, &vAddr, dtype, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(qShape, &outAddr, dtype, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // softmaxMax/Sum shape: (B, N, S1, 8) for SBH/BSND/BNSD
  // Determine B, N, S1 from layout
  int64_t B_dim, N_dim, S1_dim;
  if (layout[0] == 'S' && layout[1] == 'B') { // SBH
    S1_dim = qShape[0]; B_dim = qShape[1]; N_dim = headNum;
  } else if (layout[0] == 'B' && layout[1] == 'S') { // BSND
    B_dim = qShape[0]; S1_dim = qShape[1]; N_dim = qShape[2];
  } else { // BNSD
    B_dim = qShape[0]; N_dim = qShape[1]; S1_dim = qShape[2];
  }
  std::vector<int64_t> smShape = {B_dim, N_dim, S1_dim, 8};
  ret = CreateAclTensor(smShape, &smMaxAddr, ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(smShape, &smSumAddr, ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> prefixOp = {0};
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  aclIntArray* prefix = aclCreateIntArray(prefixOp.data(), 1);
  aclIntArray* qStartIdx = aclCreateIntArray(qStartIdxOp.data(), 1);
  aclIntArray* kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), 1);

  double scaleValue = 1.0 / sqrt(128.0);
  double keepProb = 1.0;
  int64_t preTokens = INT64_MAX;
  int64_t nextTokens = INT64_MAX;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnFlashAttentionScoreV2GetWorkspaceSize(
      q, k, v, nullptr, nullptr, nullptr, nullptr,
      prefix, qStartIdx, kvStartIdx,
      scaleValue, keepProb, preTokens, nextTokens, headNum, layout,
      innerPrecise, sparseMode, pseType,
      softmaxMax, softmaxSum, nullptr, attentionOut,
      &workspaceSize, &executor);

  if (ret != ACL_SUCCESS) {
    LOG_PRINT("  [%s] FAILED ret=%d\n", label, ret);
  } else {
    LOG_PRINT("  [%s] workspaceSize = %lu bytes (%.2f MB)\n", label, workspaceSize, workspaceSize / (1024.0 * 1024.0));
  }

  // Cleanup
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyIntArray(prefix);
  aclDestroyIntArray(qStartIdx);
  aclDestroyIntArray(kvStartIdx);
  aclrtFree(qAddr);
  aclrtFree(kAddr);
  aclrtFree(vAddr);
  aclrtFree(outAddr);
  aclrtFree(smMaxAddr);
  aclrtFree(smSumAddr);

  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  srand(42);
  const aclDataType dtype = ACL_BF16;

  // =========================================================================
  // SBH: 100 random cases
  //   B in [5,7], S1 in [30001,80000] %32!=0, S2 in [180,256] %32!=0,
  //   headNum=1, D=128
  //   Tensor shape: (S1, B, H) where H = headNum * D = 128
  // =========================================================================
  LOG_PRINT("=== SBH cases (100) ===\n");
  for (int i = 0; i < 100; i++) {
    int64_t B  = 5 + rand() % 3;                              // [5, 7]
    int64_t S1 = RandNotMod(30001, 80000, 32);
    int64_t S2 = RandNotMod(180, 256, 32);
    int64_t headNum = 1;
    int64_t D = 128;
    int64_t H = headNum * D;

    std::vector<int64_t> qShape = {S1, B, H};
    std::vector<int64_t> kShape = {S2, B, H};
    std::vector<int64_t> vShape = {S2, B, H};
    char layout[] = "SBH";

    char label[64];
    snprintf(label, sizeof(label), "sbh_%03d", i);
    TestWorkspaceSize(label, qShape, kShape, vShape, headNum, layout, dtype);
  }

  // =========================================================================
  // BSND: 40 random cases
  //   B in [1,16], S1,S2 in [1025,8192] %32!=0 (not both >3000),
  //   N in (32,128] %16!=0, D=128
  //   Tensor shape: (B, S, N, D)
  // =========================================================================
  LOG_PRINT("\n=== BSND cases (40) ===\n");
  for (int i = 0; i < 40; i++) {
    int64_t B  = 1 + rand() % 16;                              // [1, 16]
    int64_t S1 = RandNotMod(1025, 8192, 32);
    int64_t S2 = RandNotMod(1025, 8192, 32);
    if (S1 > 3000 && S2 > 3000) {
      if (rand() % 2 == 0) S1 = RandNotMod(1025, 3000, 32);
      else                  S2 = RandNotMod(1025, 3000, 32);
    }
    int64_t N  = RandNotMod(33, 127, 16);                      // (32, 128] %16!=0
    int64_t D  = 128;

    std::vector<int64_t> qShape = {B, S1, N, D};
    std::vector<int64_t> kShape = {B, S2, N, D};
    std::vector<int64_t> vShape = {B, S2, N, D};
    char layout[] = "BSND";

    char label[64];
    snprintf(label, sizeof(label), "bsnd_%03d", i);
    TestWorkspaceSize(label, qShape, kShape, vShape, N, layout, dtype);
  }

  // =========================================================================
  // BNSD: 40 random cases — same params, (B, N, S, D)
  // =========================================================================
  LOG_PRINT("\n=== BNSD cases (40) ===\n");
  for (int i = 0; i < 40; i++) {
    int64_t B  = 1 + rand() % 16;
    int64_t S1 = RandNotMod(1025, 8192, 32);
    int64_t S2 = RandNotMod(1025, 8192, 32);
    if (S1 > 3000 && S2 > 3000) {
      if (rand() % 2 == 0) S1 = RandNotMod(1025, 3000, 32);
      else                  S2 = RandNotMod(1025, 3000, 32);
    }
    int64_t N  = RandNotMod(33, 127, 16);
    int64_t D  = 128;

    std::vector<int64_t> qShape = {B, N, S1, D};
    std::vector<int64_t> kShape = {B, N, S2, D};
    std::vector<int64_t> vShape = {B, N, S2, D};
    char layout[] = "BNSD";

    char label[64];
    snprintf(label, sizeof(label), "bnsd_%03d", i);
    TestWorkspaceSize(label, qShape, kShape, vShape, N, layout, dtype);
  }

  // Cleanup ACL
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  LOG_PRINT("\nAll done.\n");
  return 0;
}
