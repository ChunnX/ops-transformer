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
 * \brief aclnnUndGenQkvRmsNormRopeCache 两段式接口调用样例（含 slot_mapping 预计算）
 */

#include <iostream>
#include <vector>
#include <cstring>
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
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t NUM_HEADS_Q = 8;
constexpr int64_t NUM_HEADS_K = 1;
constexpr int64_t NUM_HEADS_V = 1;
constexpr int64_t UND_LEN = 5;
constexpr int64_t GEN_LEN = 3;
constexpr int64_t MAX_POS = 32;
constexpr int64_t MROPE_AXIS_NUM = 3;
constexpr float NORM_EPS = 1e-6f;

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
    const int64_t blockNum = (total + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;

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

    // 3. 造 host 数据（bf16 用 uint16_t 承载，示例中填 0）
    std::vector<uint16_t> undQkvHost(GetShapeSize(undQkvShape), 0);
    std::vector<uint16_t> genQkvHost(GetShapeSize(genQkvShape), 0);
    std::vector<uint16_t> undWeightsQHost(HEAD_DIM, 0);
    std::vector<uint16_t> undWeightsKHost(HEAD_DIM, 0);
    std::vector<uint16_t> genWeightsQHost(HEAD_DIM, 0);
    std::vector<uint16_t> genWeightsKHost(HEAD_DIM, 0);
    std::vector<float> cosSinHost(GetShapeSize(cosSinShape), 0.0f);
    std::vector<uint16_t> kCacheHost(GetShapeSize(kCacheShape), 0);
    std::vector<uint16_t> vCacheHost(GetShapeSize(vCacheShape), 0);
    std::vector<uint16_t> qHost(GetShapeSize(qShape), 0);

    // slot_mapping 由调用方预计算：slot = blockNum * blockSize + rowIdx
    std::vector<int64_t> slotMappingHost(total);
    for (int64_t t = 0; t < total; ++t) {
        int64_t blockIdx = t / BLOCK_SIZE;
        int64_t rowIdx = t % BLOCK_SIZE;
        slotMappingHost[t] = blockIdx * BLOCK_SIZE + rowIdx;
    }
    // positions：三轴位置
    std::vector<int64_t> positionsHost(MROPE_AXIS_NUM * total);
    for (int64_t axis = 0; axis < MROPE_AXIS_NUM; ++axis) {
        for (int64_t t = 0; t < total; ++t) {
            positionsHost[axis * total + t] = t % MAX_POS;
        }
    }
    // cat_indices：und/gen 交错，out_t -> src_t
    std::vector<int64_t> catIndicesHost = {0, 5, 1, 6, 2, 7, 3, 4};

    // 4. 创建 aclTensor
    void *undQkvDev = nullptr, *genQkvDev = nullptr, *undWqDev = nullptr, *undWkDev = nullptr;
    void *genWqDev = nullptr, *genWkDev = nullptr, *cosSinDev = nullptr, *kCacheDev = nullptr;
    void *vCacheDev = nullptr, *slotMappingDev = nullptr, *positionsDev = nullptr, *catIndicesDev = nullptr;
    void* qDev = nullptr;
    aclTensor *undQkv = nullptr, *genQkv = nullptr, *undWq = nullptr, *undWk = nullptr, *genWq = nullptr;
    aclTensor *genWk = nullptr, *cosSin = nullptr, *kCache = nullptr, *vCache = nullptr, *slotMapping = nullptr;
    aclTensor *positions = nullptr, *catIndices = nullptr, *q = nullptr;

    ret = CreateAclTensor(undQkvHost, undQkvShape, &undQkvDev, ACL_BF16, &undQkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(undWeightsQHost, weightShape, &undWqDev, ACL_BF16, &undWq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(undWeightsKHost, weightShape, &undWkDev, ACL_BF16, &undWk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(cosSinHost, cosSinShape, &cosSinDev, ACL_FLOAT, &cosSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kCacheHost, kCacheShape, &kCacheDev, ACL_BF16, &kCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vCacheHost, vCacheShape, &vCacheDev, ACL_BF16, &vCache);
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
    ret = CreateAclTensor(qHost, qShape, &qDev, ACL_BF16, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // mrope_section：三轴分段
    std::vector<int64_t> mropeSectionData = {16, 16, 16};
    aclIntArray* mropeSection = aclCreateIntArray(mropeSectionData.data(), mropeSectionData.size());

    // 5. 第一段接口：计算 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize(
        undQkv, undWq, undWk, cosSin, kCache, vCache, slotMapping, positions, genQkv, genWq, genWk, catIndices,
        NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V, NORM_EPS, mropeSection, q, kCache, vCache, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 6. 第二段接口：执行计算
    ret = aclnnUndGenQkvRmsNormRopeCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUndGenQkvRmsNormRopeCache failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 拷回结果：q 直接读取；k_cache/v_cache 按 slot_mapping 定位对应行
    auto qSize = GetShapeSize(qShape);
    std::vector<uint16_t> qResult(qSize, 0);
    ret = aclrtMemcpy(qResult.data(), qResult.size() * sizeof(qResult[0]), qDev, qSize * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy q result failed. ERROR: %d\n", ret); return ret);

    auto kCacheSize = GetShapeSize(kCacheShape);
    std::vector<uint16_t> kCacheResult(kCacheSize, 0);
    ret = aclrtMemcpy(kCacheResult.data(), kCacheResult.size() * sizeof(kCacheResult[0]), kCacheDev,
                      kCacheSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy k_cache result failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("run aclnnUndGenQkvRmsNormRopeCache success, q size = %ld, k_cache size = %ld\n", qSize, kCacheSize);

    // 8. 释放资源
    aclDestroyTensor(undQkv);
    aclDestroyTensor(undWq);
    aclDestroyTensor(undWk);
    aclDestroyTensor(cosSin);
    aclDestroyTensor(kCache);
    aclDestroyTensor(vCache);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(positions);
    aclDestroyTensor(genQkv);
    aclDestroyTensor(genWq);
    aclDestroyTensor(genWk);
    aclDestroyTensor(catIndices);
    aclDestroyTensor(q);
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
