/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_mixed_quant_sparse_flash_mla.cpp
 * \brief MixedQuantSparseFlashMla + MixedQuantSparseFlashMlaMetadata 算子调用示例
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mixed_quant_sparse_flash_mla.h"
#include "aclnnop/aclnn_mixed_quant_sparse_flash_mla_metadata.h"

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

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void Bf16FloatSwap(uint16_t& hi, uint16_t& lo, float f)
{
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  hi = static_cast<uint16_t>(bits >> 16);
  lo = static_cast<uint16_t>(bits & 0xffff);
}

float Bf16ToFloat(uint16_t h)
{
  uint32_t bits = static_cast<uint32_t>(h) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void PrintBf16OutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
  auto size = GetShapeSize(shape);
  std::vector<uint16_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size && i < 10; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, Bf16ToFloat(resultData[i]));
  }
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream)
{
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
  auto size = GetShapeSize(shape) * sizeof(T);
  if (size > 0) {
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  } else {
    *deviceAddr = nullptr;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

std::vector<uint16_t> MakeBf16Data(int64_t size, float value)
{
  std::vector<uint16_t> data(static_cast<size_t>(size), 0);
  for (int64_t i = 0; i < size; i++) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    data[i] = static_cast<uint16_t>(bits >> 16);
  }
  return data;
}

}  // namespace

int main()
{
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 5;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &context, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // SWA场景：仅传入ori_kv，PA_BBND布局
  int64_t B = 1;
  int64_t S1 = 1;
  int64_t S2 = 8193;
  int64_t N1 = 64;
  int64_t N2 = 1;
  int64_t D = 512;
  // quant_mode=1时，KV的D为608：rope(64*2=128B) + nope(448B) + scale(7*2=14B) + pad(18B) = 608B
  int64_t D_KV = 608;
  int64_t oriBlockSize = 128;
  int64_t cmpRatio = 1;
  int64_t quantMode = 1;
  int64_t ropeHeadDim = 64;
  int64_t oriWinLeft = 127;
  int64_t oriWinRight = 0;
  int64_t oriMaskMode = 4;
  int64_t cmpMaskMode = 0;
  double softmaxScale = 1.0 / sqrt(static_cast<double>(D));

  int64_t T1 = B * S1;
  int64_t oriBlockNum = ((S2 + oriBlockSize - 1) / oriBlockSize) * B;

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> qShape = {T1, N1, D};
  std::vector<int64_t> oriKvShape = {oriBlockNum, oriBlockSize, N2, D_KV};
  std::vector<int64_t> oriBlockTableShape = {B, (S2 + oriBlockSize - 1) / oriBlockSize};
  std::vector<int64_t> seqUsedOriKvShape = {B};
  std::vector<int64_t> sinksShape = {N1};
  std::vector<int64_t> metadataShape = {1024};
  std::vector<int64_t> attnOutShape = {T1, N1, D};
  std::vector<int64_t> softmaxLseShape = {T1, N1, 1};
  // optional 输入传 shape 为 {0} 的空 tensor。
  std::vector<int64_t> emptyShape = {0};

  void* qDeviceAddr = nullptr;
  void* oriKvDeviceAddr = nullptr;
  void* cmpKvDeviceAddr = nullptr;
  void* oriSparseIndicesDeviceAddr = nullptr;
  void* cmpSparseIndicesDeviceAddr = nullptr;
  void* oriBlockTableDeviceAddr = nullptr;
  void* cmpBlockTableDeviceAddr = nullptr;
  void* cuSeqLensQDeviceAddr = nullptr;
  void* cuSeqLensOriKvDeviceAddr = nullptr;
  void* cuSeqLensCmpKvDeviceAddr = nullptr;
  void* seqUsedQDeviceAddr = nullptr;
  void* seqUsedOriKvDeviceAddr = nullptr;
  void* seqUsedCmpKvDeviceAddr = nullptr;
  void* cmpResidualKvDeviceAddr = nullptr;
  void* oriTopkLengthDeviceAddr = nullptr;
  void* cmpTopkLengthDeviceAddr = nullptr;
  void* sinksDeviceAddr = nullptr;
  void* metadataDeviceAddr = nullptr;
  void* attnOutDeviceAddr = nullptr;
  void* softmaxLseDeviceAddr = nullptr;

  aclTensor* q = nullptr;
  aclTensor* oriKv = nullptr;
  aclTensor* cmpKv = nullptr;
  aclTensor* oriSparseIndices = nullptr;
  aclTensor* cmpSparseIndices = nullptr;
  aclTensor* oriBlockTable = nullptr;
  aclTensor* cmpBlockTable = nullptr;
  aclTensor* cuSeqLensQ = nullptr;
  aclTensor* cuSeqLensOriKv = nullptr;
  aclTensor* cuSeqLensCmpKv = nullptr;
  aclTensor* seqUsedQ = nullptr;
  aclTensor* seqUsedOriKv = nullptr;
  aclTensor* seqUsedCmpKv = nullptr;
  aclTensor* cmpResidualKv = nullptr;
  aclTensor* oriTopkLength = nullptr;
  aclTensor* cmpTopkLength = nullptr;
  aclTensor* sinks = nullptr;
  aclTensor* metadata = nullptr;
  aclTensor* attnOut = nullptr;
  aclTensor* softmaxLse = nullptr;

  int64_t qSize = GetShapeSize(qShape);
  int64_t oriKvSize = GetShapeSize(oriKvShape);
  int64_t oriBlockTableSize = GetShapeSize(oriBlockTableShape);
  int64_t attnOutSize = GetShapeSize(attnOutShape);
  int64_t softmaxLseSize = GetShapeSize(softmaxLseShape);

  std::vector<uint16_t> qHostData = MakeBf16Data(qSize, 1.0f);
  // ori_kv为量化KV，整体类型为float8_e4m3fn，每个元素1字节
  std::vector<uint8_t> oriKvHostData(oriKvSize, 63);
  std::vector<int32_t> oriBlockTableHostData(oriBlockTableSize);
  std::iota(oriBlockTableHostData.begin(), oriBlockTableHostData.end(), 0);
  std::vector<int32_t> emptyHostData;
  std::vector<int32_t> seqUsedOriKvHostData(B, static_cast<int32_t>(S2));
  std::vector<float> sinksHostData(N1, 1.0f);
  std::vector<int32_t> metadataHostData(1024, 0);
  std::vector<uint16_t> attnOutHostData = MakeBf16Data(attnOutSize, 0.0f);
  std::vector<float> softmaxLseHostData(softmaxLseSize, 0.0f);

  ret = CreateAclTensor(qHostData, qShape, &qDeviceAddr, aclDataType::ACL_BF16, &q);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(oriKvHostData, oriKvShape, &oriKvDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &oriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cmpKvDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &cmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &oriSparseIndicesDeviceAddr, aclDataType::ACL_INT32, &oriSparseIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cmpSparseIndicesDeviceAddr, aclDataType::ACL_INT32, &cmpSparseIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(oriBlockTableHostData, oriBlockTableShape, &oriBlockTableDeviceAddr, aclDataType::ACL_INT32, &oriBlockTable);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cmpBlockTableDeviceAddr, aclDataType::ACL_INT32, &cmpBlockTable);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensQDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensOriKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cuSeqLensCmpKvDeviceAddr, aclDataType::ACL_INT32, &cuSeqLensCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &seqUsedQDeviceAddr, aclDataType::ACL_INT32, &seqUsedQ);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(seqUsedOriKvHostData, seqUsedOriKvShape, &seqUsedOriKvDeviceAddr, aclDataType::ACL_INT32, &seqUsedOriKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &seqUsedCmpKvDeviceAddr, aclDataType::ACL_INT32, &seqUsedCmpKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cmpResidualKvDeviceAddr, aclDataType::ACL_INT32, &cmpResidualKv);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &oriTopkLengthDeviceAddr, aclDataType::ACL_INT32, &oriTopkLength);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(emptyHostData, emptyShape, &cmpTopkLengthDeviceAddr, aclDataType::ACL_INT32, &cmpTopkLength);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sinksHostData, sinksShape, &sinksDeviceAddr, aclDataType::ACL_FLOAT, &sinks);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(metadataHostData, metadataShape, &metadataDeviceAddr, aclDataType::ACL_INT32, &metadata);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(attnOutHostData, attnOutShape, &attnOutDeviceAddr, aclDataType::ACL_BF16, &attnOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(softmaxLseHostData, softmaxLseShape, &softmaxLseDeviceAddr, aclDataType::ACL_FLOAT, &softmaxLse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  char layoutQ[] = "TND";
  char layoutKv[] = "PA_BBND";

  uint64_t metadataWorkspaceSize = 0;
  aclOpExecutor* metadataExecutor = nullptr;

  // 3. 调用metadata算子
  ret = aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize(
      cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv,
      seqUsedQ, seqUsedOriKv, seqUsedCmpKv,
      cmpResidualKv, oriTopkLength, cmpTopkLength,
      N1, N2, D, quantMode,
      B, S1, S2, 0,
      0, 0,
      ropeHeadDim, cmpRatio,
      oriMaskMode, cmpMaskMode,
      oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      true, false,
      metadata,
      &metadataWorkspaceSize, &metadataExecutor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* metadataWorkspaceAddr = nullptr;
  if (metadataWorkspaceSize > 0) {
    ret = aclrtMalloc(&metadataWorkspaceAddr, metadataWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate metadata workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnMixedQuantSparseFlashMlaMetadata(metadataWorkspaceAddr, metadataWorkspaceSize, metadataExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMixedQuantSparseFlashMlaMetadata failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after metadata failed. ERROR: %d\n", ret); return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // 5. 调用主算子
  ret = aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
      q, oriKv, cmpKv,
      oriSparseIndices, cmpSparseIndices,
      oriBlockTable, cmpBlockTable,
      cuSeqLensQ, cuSeqLensOriKv, cuSeqLensCmpKv,
      seqUsedQ, seqUsedOriKv, seqUsedCmpKv,
      cmpResidualKv, oriTopkLength, cmpTopkLength,
      sinks, metadata,
      quantMode, ropeHeadDim,
      softmaxScale, cmpRatio,
      oriMaskMode, cmpMaskMode,
      oriWinLeft, oriWinRight,
      layoutQ, layoutKv,
      1,
      false,
      attnOut, softmaxLse,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMixedQuantSparseFlashMlaGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnMixedQuantSparseFlashMla(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMixedQuantSparseFlashMla failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 6.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintBf16OutResult(attnOutShape, &attnOutDeviceAddr);

  // 7. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(q);
  aclDestroyTensor(oriKv);
  aclDestroyTensor(cmpKv);
  aclDestroyTensor(oriSparseIndices);
  aclDestroyTensor(cmpSparseIndices);
  aclDestroyTensor(oriBlockTable);
  aclDestroyTensor(cmpBlockTable);
  aclDestroyTensor(cuSeqLensQ);
  aclDestroyTensor(cuSeqLensOriKv);
  aclDestroyTensor(cuSeqLensCmpKv);
  aclDestroyTensor(seqUsedQ);
  aclDestroyTensor(seqUsedOriKv);
  aclDestroyTensor(seqUsedCmpKv);
  aclDestroyTensor(cmpResidualKv);
  aclDestroyTensor(oriTopkLength);
  aclDestroyTensor(cmpTopkLength);
  aclDestroyTensor(sinks);
  aclDestroyTensor(metadata);
  aclDestroyTensor(attnOut);
  aclDestroyTensor(softmaxLse);
  
  // 8. 释放device资源
  aclrtFree(qDeviceAddr);
  aclrtFree(oriKvDeviceAddr);
  aclrtFree(oriBlockTableDeviceAddr);
  aclrtFree(sinksDeviceAddr);
  aclrtFree(metadataDeviceAddr);
  aclrtFree(attnOutDeviceAddr);
  aclrtFree(softmaxLseDeviceAddr);
  if (metadataWorkspaceSize > 0) {
    aclrtFree(metadataWorkspaceAddr);
  }
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
