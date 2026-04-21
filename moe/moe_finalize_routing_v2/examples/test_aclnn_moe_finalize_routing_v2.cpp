/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_finalize_routing_v2.h"
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

namespace {
/* expandedRowIdx 语义见 aclnnMoeFinalizeRoutingV2：dropPadMode 为 1/3 时可为 -1（无效槽位） */
constexpr int32_t kExpandedRowIdxInvalid = -1;

/** Host 侧按 IEEE BF16 位模式与 float 互转（acl BF16 tensor 元素为 2 字节） */
inline uint16_t FloatToBf16Host(float f)
{
    uint32_t u = 0;
    static_assert(sizeof(float) == sizeof(uint32_t), "float size");
    std::memcpy(&u, &f, sizeof(u));
    return static_cast<uint16_t>((u + 0x8000u) >> 16);
}

inline float Bf16BitsToFloat(uint16_t h)
{
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

/** 全向量快速统计：区分「整段全零」与「仅前段/某段为 0」（例如多 H-slice 时列未落在首轮） */
inline void LogFloatOutSummary(const char* tag, const std::vector<float>& v, float eps = 1e-5f)
{
    int64_t nonZero = 0;
    int64_t firstNz = -1;
    float maxAbs = 0.0f;
    for (size_t i = 0; i < v.size(); ++i) {
        const float a = std::fabs(v[i]);
        if (a > eps) {
            ++nonZero;
            if (firstNz < 0) {
                firstNz = static_cast<int64_t>(i);
            }
            if (a > maxAbs) {
                maxAbs = a;
            }
        }
    }
    LOG_PRINT(
        "%s summary: len=%zu nonZero=%lld firstNonZeroIdx=%lld maxAbs=%f "
        "(若 hSliceNum=1 仍全零：更像内核/旧 Duplicate 路径问题；若 firstNonZeroIdx>0：核对 slice 写回起点)\n",
        tag, v.size(), static_cast<long long>(nonZero), static_cast<long long>(firstNz), maxAbs);
}

inline void LogBf16OutSummary(const char* tag, const std::vector<uint16_t>& v, float eps = 1e-5f)
{
    int64_t nonZero = 0;
    int64_t firstNz = -1;
    float maxAbs = 0.0f;
    for (size_t i = 0; i < v.size(); ++i) {
        const float f = Bf16BitsToFloat(v[i]);
        const float a = std::fabs(f);
        if (a > eps) {
            ++nonZero;
            if (firstNz < 0) {
                firstNz = static_cast<int64_t>(i);
            }
            if (a > maxAbs) {
                maxAbs = a;
            }
        }
    }
    LOG_PRINT(
        "%s summary: len=%zu nonZero=%lld firstNonZeroIdx=%lld maxAbs=%f "
        "(BF16 切 H 时若 firstNonZeroIdx>3 则前 4 列全零不一定矛盾)\n",
        tag, v.size(), static_cast<long long>(nonZero), static_cast<long long>(firstNz), maxAbs);
}
} // namespace

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，资源初始化
    auto  ret = aclInit(nullptr);
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

/**
 * 构造「乒侧 Db0 路由无效(-1)、Db1 路由有效」+ dropPadMode=1/3，用于暴露 K4 CutH 路径上 Db0/Db1 UB 处理问题。
 *
 * membase：LoadBias 全载成功时常走 AllBias；要落到 **CutH + K4**，需 k_=4、较大 H，使 CalcTilingData 走 CutH，
 * 从而在 CalcTilingKeyWithScales 中选 GetTilingKeyForK4()：
 * - FP32：**20009**（DTYPE_FLOAT_CUTH_K4_V2 → fp_cuth_k4）
 * - BF16：**20011**（DTYPE_BF16_CUTH_K4_V2 → bf16_cuth_k4）
 * 若日志 tilingkey 不对，可增大 kHidden（如 16384）直到出现 CutH。
 */

/** FP32：目标 FpCuthK4 / key 20009 */
static int RunCaseDb0InvalidExpert1ValidDropPadColumnFp32(aclrtStream stream)
{
    LOG_PRINT(
        "\n--- Case (FP32, target FpCuthK4 / key 20009): dropPadMode=1, K=4, "
        "expandedRowIdx={-1,0,-1,-1} ---\n");

    const int64_t kHidden = 8192;
    const int64_t kExperts = 4;

    const std::vector<int64_t> expandedXShape = {4, 2, kHidden};
    const std::vector<int64_t> biasShape = {4, kHidden};
    const std::vector<int64_t> scalesShape = {1, kExperts};
    const std::vector<int64_t> expandedExpertIdxShape = {1, kExperts};
    const std::vector<int64_t> expandedRowIdxShape = {kExperts};
    const std::vector<int64_t> outShape = {1, kHidden};
    const int64_t dropPadMode = 1;

    std::vector<float> expandedXHostData(static_cast<size_t>(GetShapeSize(expandedXShape)), 0.0f);
    for (int i = 0; i < 4; ++i) {
        expandedXHostData[static_cast<size_t>(i)] = 100.0f + static_cast<float>(i);
    }
    std::vector<float> biasHostData(static_cast<size_t>(GetShapeSize(biasShape)), 0.0f);
    {
        const float rowBias[4] = {0.5f, 1.5f, 2.5f, 3.5f};
        for (int e = 0; e < 4; ++e) {
            for (int64_t h = 0; h < kHidden; ++h) {
                biasHostData[static_cast<size_t>(e * kHidden + h)] = rowBias[e];
            }
        }
    }
    std::vector<float> scalesHostData(static_cast<size_t>(kExperts), 1.0f);
    std::vector<int32_t> expandedExpertIdxHostData = {0, 1, 2, 3};
    std::vector<int32_t> expandedRowIdxHostData = {
        kExpandedRowIdxInvalid, 0, kExpandedRowIdxInvalid, kExpandedRowIdxInvalid};
    std::vector<float> outHostData(static_cast<size_t>(GetShapeSize(outShape)), 0.0f);

    void* expandedXAddr = nullptr;
    void* biasAddr = nullptr;
    void* scalesDeviceAddr = nullptr;
    void* expandedExpertIdxAddr = nullptr;
    void* expandedRowIdxAddr = nullptr;
    void* outDeviceAddr = nullptr;

    aclTensor* expandedX = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* scales = nullptr;
    aclTensor* expandedExpertIdx = nullptr;
    aclTensor* expandedRowIdx = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXAddr, aclDataType::ACL_FLOAT, &expandedX);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor expandedX failed\n"); return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor bias failed\n"); return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor scales failed\n"); return ret);
    ret = CreateAclTensor(expandedExpertIdxHostData, expandedExpertIdxShape, &expandedExpertIdxAddr,
                          aclDataType::ACL_INT32, &expandedExpertIdx);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor expandedExpertIdx failed\n"); return ret);
    ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxAddr, aclDataType::ACL_INT32,
                          &expandedRowIdx);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor expandedRowIdx failed\n"); return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor out failed\n"); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMoeFinalizeRoutingV2GetWorkspaceSize(expandedX, expandedRowIdx,
                                                      nullptr /* x1 */, nullptr /* x2 */, bias, scales, expandedExpertIdx,
                                                      dropPadMode, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnMoeFinalizeRoutingV2GetWorkspaceSize (FP32 K4 case) failed. ERROR: %d\n", ret);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnMoeFinalizeRoutingV2(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnMoeFinalizeRoutingV2 (FP32 K4 case) failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outShape);
    std::vector<float> result(size, 0.0f);
    ret = aclrtMemcpy(result.data(), result.size() * sizeof(result[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT(
        "drop pad FP32 K4 case: first 4 columns (expect ~(100+i)+bias_expert1 ~101.5..104.5). "
        "Check log DumpTilingInfo tilingkey:20009 for DTYPE_FLOAT_CUTH_K4_V2.\n");
    LogFloatOutSummary("FP32 K4 out", result);
    const int64_t kPrint = (size < 4) ? size : 4;
    for (int64_t i = 0; i < kPrint; ++i) {
        LOG_PRINT("  result[%ld] = %f\n", i, result[static_cast<size_t>(i)]);
    }

    aclDestroyTensor(expandedX);
    aclDestroyTensor(bias);
    aclDestroyTensor(scales);
    aclDestroyTensor(expandedExpertIdx);
    aclDestroyTensor(expandedRowIdx);
    aclDestroyTensor(out);
    aclrtFree(expandedXAddr);
    aclrtFree(biasAddr);
    aclrtFree(scalesDeviceAddr);
    aclrtFree(expandedExpertIdxAddr);
    aclrtFree(expandedRowIdxAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

/** BF16：目标 Bf16CuthK4 / key 20011 */
static int RunCaseDb0InvalidExpert1ValidDropPadColumnBf16(aclrtStream stream)
{
    LOG_PRINT(
        "\n--- Case (BF16, target Bf16CuthK4 / key 20011): dropPadMode=1, K=4, "
        "expandedRowIdx={-1,0,-1,-1} ---\n");

    const int64_t kHidden = 8192; // 不足触发 CutH 时请加大该值
    const int64_t kExperts = 4;

    const std::vector<int64_t> expandedXShape = {4, 2, kHidden}; // [E,C,H]，E>=4 以支持 expert_idx 0..3
    const std::vector<int64_t> biasShape = {4, kHidden};
    const std::vector<int64_t> scalesShape = {1, kExperts};
    const std::vector<int64_t> expandedExpertIdxShape = {1, kExperts};
    const std::vector<int64_t> expandedRowIdxShape = {kExperts};
    const std::vector<int64_t> outShape = {1, kHidden};
    const int64_t dropPadMode = 1;

    std::vector<uint16_t> expandedXHostData(GetShapeSize(expandedXShape), UINT16_C(0));
    // [4,2,H] 线性布局：下标 0..3 对应 expert0、capacity0 的前 4 个 hidden（expanded_row_idx==0 时读取）
    for (int i = 0; i < 4; ++i) {
        expandedXHostData[static_cast<size_t>(i)] = FloatToBf16Host(100.0f + static_cast<float>(i));
    }
    std::vector<uint16_t> biasHostData(GetShapeSize(biasShape), UINT16_C(0));
    {
        const float rowBias[4] = {0.5f, 1.5f, 2.5f, 3.5f};
        for (int e = 0; e < 4; ++e) {
            const uint16_t be = FloatToBf16Host(rowBias[e]);
            for (int64_t h = 0; h < kHidden; ++h) {
                biasHostData[static_cast<size_t>(e * kHidden + h)] = be;
            }
        }
    }
    std::vector<uint16_t> scalesHostData(static_cast<size_t>(kExperts), FloatToBf16Host(1.0f));
    std::vector<int32_t> expandedExpertIdxHostData = {0, 1, 2, 3};
    std::vector<int32_t> expandedRowIdxHostData = {
        kExpandedRowIdxInvalid, 0, kExpandedRowIdxInvalid, kExpandedRowIdxInvalid};
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), UINT16_C(0));

    void* expandedXAddr = nullptr;
    void* biasAddr = nullptr;
    void* scalesDeviceAddr = nullptr;
    void* expandedExpertIdxAddr = nullptr;
    void* expandedRowIdxAddr = nullptr;
    void* outDeviceAddr = nullptr;

    aclTensor* expandedX = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* scales = nullptr;
    aclTensor* expandedExpertIdx = nullptr;
    aclTensor* expandedRowIdx = nullptr;
    aclTensor* out = nullptr;

    auto ret = CreateAclTensor(expandedXHostData, expandedXShape, &expandedXAddr, aclDataType::ACL_BF16, &expandedX);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor expandedX failed\n"); return ret);
    ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_BF16, &bias);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor bias failed\n"); return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_BF16, &scales);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor scales failed\n"); return ret);
    ret = CreateAclTensor(expandedExpertIdxHostData, expandedExpertIdxShape, &expandedExpertIdxAddr,
                          aclDataType::ACL_INT32, &expandedExpertIdx);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor expandedExpertIdx failed\n"); return ret);
    ret = CreateAclTensor(expandedRowIdxHostData, expandedRowIdxShape, &expandedRowIdxAddr, aclDataType::ACL_INT32,
                          &expandedRowIdx);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor expandedRowIdx failed\n"); return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
    CHECK_RET(ret == 0, LOG_PRINT("CreateAclTensor out failed\n"); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMoeFinalizeRoutingV2GetWorkspaceSize(expandedX, expandedRowIdx,
                                                    nullptr /* x1 */, nullptr /* x2 */, bias, scales, expandedExpertIdx,
                                                    dropPadMode, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnMoeFinalizeRoutingV2GetWorkspaceSize (drop pad case) failed. ERROR: %d\n", ret);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnMoeFinalizeRoutingV2(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnMoeFinalizeRoutingV2 (drop pad case) failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultBf16(size, UINT16_C(0));
    ret = aclrtMemcpy(resultBf16.data(), resultBf16.size() * sizeof(resultBf16[0]), outDeviceAddr,
                      size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT(
        "drop pad BF16 case: first 4 columns (decoded float; expect ~(100+i)+bias_expert1 ~101.5..104.5). "
        "Check log DumpTilingInfo tilingkey:20011 for DTYPE_BF16_CUTH_K4_V2.\n");
    LogBf16OutSummary("BF16 K4 out", resultBf16);
    const int64_t kPrint = (size < 4) ? size : 4;
    for (int64_t i = 0; i < kPrint; ++i) {
        LOG_PRINT("  result[%ld] = %f (bf16 bits 0x%04x)\n", i, Bf16BitsToFloat(resultBf16[static_cast<size_t>(i)]),
                  static_cast<unsigned>(resultBf16[static_cast<size_t>(i)]));
    }

    aclDestroyTensor(expandedX);
    aclDestroyTensor(bias);
    aclDestroyTensor(scales);
    aclDestroyTensor(expandedExpertIdx);
    aclDestroyTensor(expandedRowIdx);
    aclDestroyTensor(out);
    aclrtFree(expandedXAddr);
    aclrtFree(biasAddr);
    aclrtFree(scalesDeviceAddr);
    aclrtFree(expandedExpertIdxAddr);
    aclrtFree(expandedRowIdxAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT(
      "=== K4 CutH regression: FP32 key 20009 (fp_cuth_k4), then BF16 key 20011 (bf16_cuth_k4) ===\n");

  ret = RunCaseDb0InvalidExpert1ValidDropPadColumnFp32(stream);
  CHECK_RET(ret == 0,
            LOG_PRINT("RunCaseDb0InvalidExpert1ValidDropPadColumnFp32 failed. ERROR: %d\n", ret); return ret);

  ret = RunCaseDb0InvalidExpert1ValidDropPadColumnBf16(stream);
  CHECK_RET(ret == 0,
            LOG_PRINT("RunCaseDb0InvalidExpert1ValidDropPadColumnBf16 failed. ERROR: %d\n", ret); return ret);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}