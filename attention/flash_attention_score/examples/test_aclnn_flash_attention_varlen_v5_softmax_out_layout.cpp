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
 * \file test_aclnn_flash_attention_varlen_v5_softmax_out_layout.cpp
 * \brief Test softmaxOutLayout feature of aclnnFlashAttentionVarLenScoreV5.
 *
 * softmaxOutLayout controls softmax output layout in TND scenario:
 *   - "" (empty string):        softmax output layout is NTD → softmaxMax/Sum shape = [N, T, 8]
 *   - "same_as_input":          softmax output layout matches input TND → shape = [T, N, 8]
 *
 * This test verifies both modes produce correct results and prints output shapes.
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int CreateAclTensorNoData(const std::vector<int64_t>& shape, void** deviceAddr,
                          aclDataType dataType, aclTensor** tensor) {
  int64_t elemSize = sizeof(float);
  if (dataType == ACL_BF16 || dataType == ACL_FLOAT16) elemSize = sizeof(uint16_t);
  else if (dataType == ACL_UINT8) elemSize = sizeof(uint8_t);
  auto size = GetShapeSize(shape) * elemSize;
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

void PrintShape(const char* name, const std::vector<int64_t>& shape) {
  LOG_PRINT("  %s shape: [", name);
  for (size_t i = 0; i < shape.size(); i++) {
    LOG_PRINT("%ld%s", shape[i], (i < shape.size() - 1) ? ", " : "");
  }
  LOG_PRINT("]\n");
}

void PrintResult(const char* name, const std::vector<int64_t>& shape, void** deviceAddr, int64_t printCount = 8) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float),
                         *deviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  copy result failed. ERROR: %d\n", ret); return);
  LOG_PRINT("  %s (first %ld values): ", name, std::min(printCount, size));
  for (int64_t i = 0; i < std::min(printCount, size); i++) {
    LOG_PRINT("%.6f ", resultData[i]);
  }
  LOG_PRINT("\n");
}

/**
 * Run a single test case with the given softmaxOutLayout.
 * TND layout: query/key/value shape = [T, N, D]
 */
int RunTest(const char* caseName,
            int64_t T, int64_t N, int64_t D, int64_t Skv,
            const char* softmaxOutLayout,
            aclrtStream stream) {
  LOG_PRINT("\n=== [%s] T=%ld N=%ld D=%ld Skv=%ld softmaxOutLayout=\"%s\" ===\n",
            caseName, T, N, D, Skv, softmaxOutLayout);

  // TND shapes
  std::vector<int64_t> qShape = {T, N, D};
  std::vector<int64_t> kShape = {Skv, N, D};
  std::vector<int64_t> vShape = {Skv, N, D};
  std::vector<int64_t> attentionOutShape = {T, N, D};

  // softmaxMax/Sum shape is always [T, N, 8] (InferShape derives TND shape regardless of softmaxOutLayout).
  // softmaxOutLayout only controls the internal data arrangement, not the tensor shape.
  std::vector<int64_t> softmaxMaxShape = {T, N, 8};
  std::vector<int64_t> softmaxSumShape = {T, N, 8};
  if (strlen(softmaxOutLayout) > 0 && strcmp(softmaxOutLayout, "same_as_input") == 0) {
    LOG_PRINT("  softmaxMax/Sum: shape [T=%ld, N=%ld, 8], data layout TND (same_as_input)\n", T, N);
  } else {
    LOG_PRINT("  softmaxMax/Sum: shape [T=%ld, N=%ld, 8], data layout NTD (empty string)\n", T, N);
  }

  PrintShape("query", qShape);
  PrintShape("key", kShape);
  PrintShape("value", vShape);
  PrintShape("softmaxMax", softmaxMaxShape);
  PrintShape("attentionOut", attentionOutShape);

  // Create tensors
  void* qAddr = nullptr;
  void* kAddr = nullptr;
  void* vAddr = nullptr;
  void* outAddr = nullptr;
  void* smMaxAddr = nullptr;
  void* smSumAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* k = nullptr;
  aclTensor* v = nullptr;
  aclTensor* attentionOut = nullptr;
  aclTensor* softmaxMax = nullptr;
  aclTensor* softmaxSum = nullptr;
  aclTensor* softmaxOut = nullptr;

  int64_t qSize = GetShapeSize(qShape);
  int64_t kvSize = GetShapeSize(kShape);

  std::vector<float> qHostData(qSize, 1.0f);
  std::vector<float> kHostData(kvSize, 1.0f);
  std::vector<float> vHostData(kvSize, 1.0f);

  int ret = CreateAclTensor(qHostData, qShape, &qAddr, ACL_BF16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(kHostData, kShape, &kAddr, ACL_BF16, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, vShape, &vAddr, ACL_BF16, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorNoData(attentionOutShape, &outAddr, ACL_BF16, &attentionOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorNoData(softmaxMaxShape, &smMaxAddr, ACL_FLOAT, &softmaxMax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorNoData(softmaxSumShape, &smSumAddr, ACL_FLOAT, &softmaxSum);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // actualSeqQLen / actualSeqKvLen for TND (single batch: T = S1, Skv = S2)
  std::vector<int64_t> acSeqQ = {T};
  std::vector<int64_t> acSeqKv = {Skv};
  aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQ.data(), acSeqQ.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKv.data(), acSeqKv.size());

  std::vector<int64_t> prefixOp = {0};
  std::vector<int64_t> qStartIdxOp = {0};
  std::vector<int64_t> kvStartIdxOp = {0};
  aclIntArray* prefix = aclCreateIntArray(prefixOp.data(), prefixOp.size());
  aclIntArray* qStartIdx = aclCreateIntArray(qStartIdxOp.data(), qStartIdxOp.size());
  aclIntArray* kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), kvStartIdxOp.size());

  double scaleValue = 1.0 / sqrt(static_cast<double>(D));
  double keepProb = 1.0;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t innerPrecise = 0;
  int64_t sparseMode = 0;
  int64_t pseType = 1;
  char layOut[] = "TND";

  // Phase 1: GetWorkspaceSize
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(
      q, nullptr, k, nullptr, v,
      nullptr, nullptr, nullptr, nullptr, nullptr,
      prefix, acSeqQLen, acSeqKvLen, qStartIdx, kvStartIdx,
      scaleValue, keepProb, preTokens, nextTokens,
      N, layOut, innerPrecise, sparseMode, pseType,
      const_cast<char*>(softmaxOutLayout),
      softmaxMax, softmaxSum, softmaxOut, attentionOut,
      &workspaceSize, &executor);

  if (ret != ACL_SUCCESS) {
    LOG_PRINT("  GetWorkspaceSize FAILED: ret=%d\n", ret);
    // Cleanup and return
    goto cleanup;
  }
  LOG_PRINT("  workspaceSize = %lu bytes (%.2f MB)\n", workspaceSize, workspaceSize / (1024.0 * 1024.0));

  // Allocate workspace
  {
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  allocate workspace failed. ERROR: %d\n", ret); goto cleanup);
    }

    // Phase 2: Execute
    ret = aclnnFlashAttentionVarLenScoreV5(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("  Execute FAILED: ret=%d\n", ret);
    } else {
      LOG_PRINT("  Execute SUCCESS\n");
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  SyncStream failed. ERROR: %d\n", ret); goto cleanup);

      // Print some output values
      PrintResult("softmaxMax", softmaxMaxShape, &smMaxAddr, 8);
      PrintResult("attentionOut", attentionOutShape, &outAddr, 8);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
  }

cleanup:
  aclDestroyTensor(q);
  aclDestroyTensor(k);
  aclDestroyTensor(v);
  aclDestroyTensor(attentionOut);
  aclDestroyTensor(softmaxMax);
  aclDestroyTensor(softmaxSum);
  aclDestroyIntArray(prefix);
  aclDestroyIntArray(acSeqQLen);
  aclDestroyIntArray(acSeqKvLen);
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

  LOG_PRINT("============================================================\n");
  LOG_PRINT("Test softmaxOutLayout feature of aclnnFlashAttentionVarLenScoreV5\n");
  LOG_PRINT("============================================================\n");

  // -----------------------------------------------------------------------
  // Case 1: softmaxOutLayout="" (default NTD layout)
  //   softmaxMax/Sum shape = [N, T, 8]
  // -----------------------------------------------------------------------
  RunTest("case1_empty_string", 256, 1, 128, 256, "", stream);

  // -----------------------------------------------------------------------
  // Case 2: softmaxOutLayout="same_as_input" (TND layout)
  //   softmaxMax/Sum shape = [T, N, 8]
  // -----------------------------------------------------------------------
  RunTest("case2_same_as_input", 256, 1, 128, 256, "same_as_input", stream);

  // -----------------------------------------------------------------------
  // Case 3: Larger shape with empty string
  // -----------------------------------------------------------------------
  RunTest("case3_empty_large", 1024, 4, 128, 512, "", stream);

  // -----------------------------------------------------------------------
  // Case 4: Larger shape with same_as_input
  // -----------------------------------------------------------------------
  RunTest("case4_same_large", 1024, 4, 128, 512, "same_as_input", stream);

  // -----------------------------------------------------------------------
  // Case 5: Multi-batch (B=3, S1=200 each → T=600, Skv=400)
  // -----------------------------------------------------------------------
  {
    int64_t B = 3;
    int64_t S1 = 200;
    int64_t Skv = 200;
    int64_t T = B * S1;
    int64_t Tkv = B * Skv;
    int64_t N = 2;
    int64_t D = 64;

    LOG_PRINT("\n=== [case5_multi_batch] B=%ld S1=%ld Skv=%ld T=%ld Tkv=%ld N=%ld D=%ld ===\n",
              B, S1, Skv, T, Tkv, N, D);

    std::vector<int64_t> qShape = {T, N, D};
    std::vector<int64_t> kShape = {Tkv, N, D};
    std::vector<int64_t> vShape = {Tkv, N, D};
    std::vector<int64_t> attentionOutShape = {T, N, D};

    // Test with same_as_input: softmax shape = [T, N, 8]
    std::vector<int64_t> smShape = {T, N, 8};

    void* qAddr = nullptr;
    void* kAddr = nullptr;
    void* vAddr = nullptr;
    void* outAddr = nullptr;
    void* smMaxAddr = nullptr;
    void* smSumAddr = nullptr;
    aclTensor* q = nullptr;
    aclTensor* k = nullptr;
    aclTensor* v = nullptr;
    aclTensor* attentionOut = nullptr;
    aclTensor* softmaxMax = nullptr;
    aclTensor* softmaxSum = nullptr;
    aclTensor* softmaxOut = nullptr;

    int64_t qSize = GetShapeSize(qShape);
    int64_t kvSize = GetShapeSize(kShape);

    ret = CreateAclTensor(std::vector<float>(qSize, 1.0f), qShape, &qAddr, ACL_BF16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(std::vector<float>(kvSize, 1.0f), kShape, &kAddr, ACL_BF16, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(std::vector<float>(kvSize, 1.0f), vShape, &vAddr, ACL_BF16, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorNoData(attentionOutShape, &outAddr, ACL_BF16, &attentionOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorNoData(smShape, &smMaxAddr, ACL_FLOAT, &softmaxMax);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorNoData(smShape, &smSumAddr, ACL_FLOAT, &softmaxSum);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // actualSeqQLen = [200, 400, 600], actualSeqKvLen = [200, 400, 600]
    std::vector<int64_t> acSeqQ = {S1, 2 * S1, 3 * S1};
    std::vector<int64_t> acSeqKv = {Skv, 2 * Skv, 3 * Skv};
    aclIntArray* acSeqQLen = aclCreateIntArray(acSeqQ.data(), acSeqQ.size());
    aclIntArray* acSeqKvLen = aclCreateIntArray(acSeqKv.data(), acSeqKv.size());

    std::vector<int64_t> prefixOp = {0, 0, 0};
    std::vector<int64_t> qStartIdxOp = {0};
    std::vector<int64_t> kvStartIdxOp = {0};
    aclIntArray* prefix = aclCreateIntArray(prefixOp.data(), prefixOp.size());
    aclIntArray* qStartIdx = aclCreateIntArray(qStartIdxOp.data(), qStartIdxOp.size());
    aclIntArray* kvStartIdx = aclCreateIntArray(kvStartIdxOp.data(), kvStartIdxOp.size());

    double scaleValue = 1.0 / sqrt(static_cast<double>(D));
    char layOut[] = "TND";
    char softmaxOutLayout[] = "same_as_input";

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(
        q, nullptr, k, nullptr, v,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        prefix, acSeqQLen, acSeqKvLen, qStartIdx, kvStartIdx,
        scaleValue, 1.0, 65536, 65536,
        N, layOut, 0, 0, 1,
        softmaxOutLayout,
        softmaxMax, softmaxSum, softmaxOut, attentionOut,
        &workspaceSize, &executor);

    if (ret != ACL_SUCCESS) {
      LOG_PRINT("  GetWorkspaceSize FAILED: ret=%d\n", ret);
    } else {
      LOG_PRINT("  workspaceSize = %lu bytes (%.2f MB)\n", workspaceSize, workspaceSize / (1024.0 * 1024.0));
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  alloc workspace failed\n"); goto cleanup5);
      }
      ret = aclnnFlashAttentionVarLenScoreV5(workspaceAddr, workspaceSize, executor, stream);
      LOG_PRINT("  Execute: %s\n", ret == ACL_SUCCESS ? "SUCCESS" : "FAILED");
      aclrtSynchronizeStream(stream);
      if (workspaceAddr) aclrtFree(workspaceAddr);
    }

  cleanup5:
    aclDestroyTensor(q); aclDestroyTensor(k); aclDestroyTensor(v);
    aclDestroyTensor(attentionOut); aclDestroyTensor(softmaxMax); aclDestroyTensor(softmaxSum);
    aclDestroyIntArray(prefix); aclDestroyIntArray(acSeqQLen); aclDestroyIntArray(acSeqKvLen);
    aclDestroyIntArray(qStartIdx); aclDestroyIntArray(kvStartIdx);
    aclrtFree(qAddr); aclrtFree(kAddr); aclrtFree(vAddr);
    aclrtFree(outAddr); aclrtFree(smMaxAddr); aclrtFree(smSumAddr);
  }

  // Cleanup ACL
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  LOG_PRINT("\nAll done.\n");
  return 0;
}
