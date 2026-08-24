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
 * \file test_aclnn_und_gen_qkv_rms_norm_rope_cache.cpp
 * \brief 性能采集驱动：典型 shape，KV Cache device 侧 memset，slot/cat 打散，固定 seed
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_und_gen_qkv_rms_norm_rope_cache.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                            \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {
int64_t EnvI64(const char* name, int64_t def)
{
    const char* v = std::getenv(name);
    return (v && *v) ? std::strtoll(v, nullptr, 10) : def;
}

// 默认即典型 case：T = 6652 + 4100 = 10752，heads (8,1,1)，KV Cache 23568 个 block
const int64_t HEAD_DIM   = EnvI64("HEAD_DIM", 128);
const int64_t BLOCK_SIZE = EnvI64("BLOCK_SIZE", 128);
const int64_t NUM_HEADS_Q = EnvI64("HQ", 8);
const int64_t NUM_HEADS_K = EnvI64("HK", 1);
const int64_t NUM_HEADS_V = EnvI64("HV", 1);
const int64_t UND_LEN    = EnvI64("UND_LEN", 6652);
const int64_t GEN_LEN    = EnvI64("GEN_LEN", 4100);
const int64_t MAX_POS    = EnvI64("MAX_POS", 16384);
const int64_t NUM_BLOCKS = EnvI64("NUM_BLOCKS", 23568);
const int64_t ITERS      = EnvI64("ITERS", 20);
const int64_t WARMUP     = EnvI64("WARMUP", 3);
constexpr int64_t MROPE_AXIS_NUM = 3;
constexpr float NORM_EPS = 1e-6f;

// bf16 用 uint16_t 承载：取 float32 的高 16 位，就近舍入
uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t rounded = bits + 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(rounded >> 16);
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
                    aclDataType dataType, aclTensor** tensor, aclFormat format = ACL_FORMAT_ND)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}
// KV Cache 太大（典型 case 每个 771MB），只在 device 侧 memset，不分配 host 副本、不走 H2D
int CreateAclTensorDevZero(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                           aclTensor** tensor, size_t elemSize)
{
    auto size = static_cast<size_t>(GetShapeSize(shape)) * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemset(*deviceAddr, size, 0, size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemset failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}
} // namespace

int main()
{
    // 1. 初始化设备与流
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t total = UND_LEN + GEN_LEN;
    const int64_t numHead = NUM_HEADS_Q + NUM_HEADS_K + NUM_HEADS_V;
    const int64_t blockNum = NUM_BLOCKS > 0 ? NUM_BLOCKS : ((total + BLOCK_SIZE - 1) / BLOCK_SIZE + 1);

    // 2. 构造输入 shape
    std::vector<int64_t> undQkvShape = {UND_LEN, numHead, HEAD_DIM};
    std::vector<int64_t> genQkvShape = {GEN_LEN, numHead, HEAD_DIM};
    std::vector<int64_t> weightShape = {HEAD_DIM};
    std::vector<int64_t> cosSinShape = {MAX_POS, HEAD_DIM};
    std::vector<int64_t> kCacheShape = {blockNum, BLOCK_SIZE, NUM_HEADS_K, HEAD_DIM};
    std::vector<int64_t> vCacheShape = {blockNum, BLOCK_SIZE, NUM_HEADS_V, HEAD_DIM};
    std::vector<int64_t> slotMappingShape = {total};
    std::vector<int64_t> positionsShape = {MROPE_AXIS_NUM, total};
    std::vector<int64_t> catIndicesShape = {total};
    std::vector<int64_t> qShape = {total, NUM_HEADS_Q, HEAD_DIM};

    // 3. 造 host 数据（bf16 用 uint16_t 承载）
    // QKV 填一段随位置缓慢变化的非零数据，避免全零输入让样例退化成恒等于 0 的空跑
    std::vector<uint16_t> undQkvHost(GetShapeSize(undQkvShape));
    for (size_t i = 0; i < undQkvHost.size(); ++i) {
        undQkvHost[i] = FloatToBf16(0.05f * static_cast<float>(i % 17) - 0.4f);
    }
    std::vector<uint16_t> genQkvHost(GetShapeSize(genQkvShape));
    for (size_t i = 0; i < genQkvHost.size(); ++i) {
        genQkvHost[i] = FloatToBf16(0.05f * static_cast<float>(i % 13) - 0.3f);
    }
    // RMSNorm 权重：und 段取 1.0，gen 段取 0.5，便于区分两段权重是否按 catIndices 正确选中
    std::vector<uint16_t> undWeightsQHost(HEAD_DIM, FloatToBf16(1.0f));
    std::vector<uint16_t> undWeightsKHost(HEAD_DIM, FloatToBf16(1.0f));
    std::vector<uint16_t> genWeightsQHost(HEAD_DIM, FloatToBf16(0.5f));
    std::vector<uint16_t> genWeightsKHost(HEAD_DIM, FloatToBf16(0.5f));
    // cos/sin 缓存表：前 D/2 列为 cos、后 D/2 列为 sin，按标准 RoPE 的 theta 生成
    std::vector<float> cosSinHost(GetShapeSize(cosSinShape));
    for (int64_t pos = 0; pos < MAX_POS; ++pos) {
        for (int64_t i = 0; i < HEAD_DIM / 2; ++i) {
            float freq = 1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(HEAD_DIM));
            float angle = static_cast<float>(pos) * freq;
            cosSinHost[pos * HEAD_DIM + i] = std::cos(angle);
            cosSinHost[pos * HEAD_DIM + HEAD_DIM / 2 + i] = std::sin(angle);
        }
    }
    // KV Cache 由调用方预分配并原地更新，未被 slot_mapping 命中的行保持传入值
    std::vector<uint16_t> qHost(GetShapeSize(qShape), 0);

    // slot_mapping 由调用方预计算：slot = blockIdx * blockSize + rowIdx
    // slot 在整个 paged cache 上打散：散射写才是真实负载，顺序 slot 会被 L2 吃掉。seed 固定以便 A/B
    std::mt19937_64 rng(20260817ULL);
    const int64_t totalSlots = blockNum * BLOCK_SIZE;
    std::vector<int64_t> allSlots(static_cast<size_t>(totalSlots));
    std::iota(allSlots.begin(), allSlots.end(), 0);
    std::shuffle(allSlots.begin(), allSlots.end(), rng);
    std::vector<int64_t> slotMappingHost(allSlots.begin(), allSlots.begin() + total);
    // positions：三轴位置
    std::vector<int64_t> positionsHost(MROPE_AXIS_NUM * total);
    for (int64_t axis = 0; axis < MROPE_AXIS_NUM; ++axis) {
        for (int64_t t = 0; t < total; ++t) {
            positionsHost[axis * total + t] = t % MAX_POS;
        }
    }
    // cat_indices：und/gen 交错，out_t -> src_t
    std::vector<int64_t> catIndicesHost(total);
    std::iota(catIndicesHost.begin(), catIndicesHost.end(), 0);
    std::shuffle(catIndicesHost.begin(), catIndicesHost.end(), rng);

    // 4. 创建 aclTensor
    void *undQkvDev = nullptr, *genQkvDev = nullptr, *undWqDev = nullptr, *undWkDev = nullptr;
    void *genWqDev = nullptr, *genWkDev = nullptr, *cosSinDev = nullptr, *kCacheDev = nullptr;
    void *vCacheDev = nullptr, *slotMappingDev = nullptr, *positionsDev = nullptr, *catIndicesDev = nullptr;
    void* qDev = nullptr;
    aclTensor *undQkv = nullptr, *genQkv = nullptr, *undWq = nullptr, *undWk = nullptr, *genWq = nullptr;
    aclTensor *genWk = nullptr, *cosSin = nullptr, *kCacheRef = nullptr, *vCacheRef = nullptr, *slotMapping = nullptr;
    aclTensor *positions = nullptr, *catIndices = nullptr, *qOut = nullptr;

    ret = CreateAclTensor(undQkvHost, undQkvShape, &undQkvDev, ACL_BF16, &undQkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(undWeightsQHost, weightShape, &undWqDev, ACL_BF16, &undWq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(undWeightsKHost, weightShape, &undWkDev, ACL_BF16, &undWk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosSinHost, cosSinShape, &cosSinDev, ACL_FLOAT, &cosSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorDevZero(kCacheShape, &kCacheDev, ACL_BF16, &kCacheRef, sizeof(uint16_t));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorDevZero(vCacheShape, &vCacheDev, ACL_BF16, &vCacheRef, sizeof(uint16_t));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(slotMappingHost, slotMappingShape, &slotMappingDev, ACL_INT64, &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(positionsHost, positionsShape, &positionsDev, ACL_INT64, &positions);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(genQkvHost, genQkvShape, &genQkvDev, ACL_BF16, &genQkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(genWeightsQHost, weightShape, &genWqDev, ACL_BF16, &genWq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(genWeightsKHost, weightShape, &genWkDev, ACL_BF16, &genWk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(catIndicesHost, catIndicesShape, &catIndicesDev, ACL_INT64, &catIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qHost, qShape, &qDev, ACL_BF16, &qOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // mrope_section：三轴分段
    std::vector<int64_t> mropeSectionData = {16, 16, 16};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionData.data(), mropeSectionData.size());

    // 5/6. 两段式接口，跑 WARMUP + ITERS 轮。executor 一次性，每轮都要重新取
    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    LOG_PRINT("T=%ld (und %ld + gen %ld), heads (%ld,%ld,%ld), blockNum=%ld, warmup=%ld, iters=%ld\n",
              total, UND_LEN, GEN_LEN, NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V, blockNum, WARMUP, ITERS);
    for (int64_t it = 0; it < WARMUP + ITERS; ++it) {
        aclOpExecutor* executor = nullptr;
        ret = aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize(
            undQkv, undWq, undWk, cosSin, kCacheRef, vCacheRef, slotMapping, positions, genQkv, genWq, genWk,
            catIndices, NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V, NORM_EPS, mropeSection, qOut, &workspaceSize,
            &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("GetWorkspaceSize failed at iter %ld. ERROR: %d\n", it, ret); return ret);
        if (workspaceSize > 0 && workspaceAddr == nullptr) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        ret = aclnnUndGenQkvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("execute failed at iter %ld. ERROR: %d\n", it, ret); return ret);
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("sync failed at iter %ld. ERROR: %d\n", it, ret); return ret);
    }

    // 7. 拷回结果：qOut 直接读取；k_cache/v_cache 按 slot_mapping 定位对应行
    auto qSize = GetShapeSize(qShape);
    std::vector<uint16_t> qResult(qSize, 0);
    ret = aclrtMemcpy(qResult.data(), qResult.size() * sizeof(qResult[0]), qDev, qSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy qOut result failed. ERROR: %d\n", ret); return ret);

    // 只回读 qOut 做存活性检查；k_cache 典型 case 771MB，全量 D2H 没有意义
    constexpr int64_t PRINT_NUM = 8;
    for (int64_t i = 0; i < PRINT_NUM; ++i) {
        LOG_PRINT("qOut[%ld] = %f\n", i, Bf16ToFloat(qResult[i]));
    }
    LOG_PRINT("run aclnnUndGenQkvRmsNormRopeCache success, qOut size = %ld\n", qSize);

    // 8. 释放资源
    aclDestroyTensor(undQkv);
    aclDestroyTensor(undWq);
    aclDestroyTensor(undWk);
    aclDestroyTensor(cosSin);
    aclDestroyTensor(kCacheRef);
    aclDestroyTensor(vCacheRef);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(positions);
    aclDestroyTensor(genQkv);
    aclDestroyTensor(genWq);
    aclDestroyTensor(genWk);
    aclDestroyTensor(catIndices);
    aclDestroyTensor(qOut);
    aclDestroyIntArray(mropeSection);

    aclrtFree(undQkvDev);
    aclrtFree(undWqDev);
    aclrtFree(undWkDev);
    aclrtFree(cosSinDev);
    aclrtFree(kCacheDev);
    aclrtFree(vCacheDev);
    aclrtFree(slotMappingDev);
    aclrtFree(positionsDev);
    aclrtFree(genQkvDev);
    aclrtFree(genWqDev);
    aclrtFree(genWkDev);
    aclrtFree(catIndicesDev);
    aclrtFree(qDev);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
