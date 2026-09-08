/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_roi_extractor.cpp
 * \brief aclnn call example for RoiExtractor (single-level fp16 run).
 */

#include <aclnn/aclnn_base.h>
#include <iostream>

constexpr int64_t kNumLevels = 1;   // provide features_0 only; features_1..7 = nullptr
constexpr int64_t kR = 32;
constexpr int64_t kN = 1;
constexpr int64_t kC = 16;
constexpr int64_t kH = 32;
constexpr int64_t kW = 32;
constexpr int64_t kPooledH = 7;
constexpr int64_t kPooledW = 7;

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) {
        std::cerr << "acl init failed" << std::endl;
        return 1;
    }
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // features_0: [N, C, H, W] fp16 ND (contiguous)
    int64_t featShape[4] = {kN, kC, kH, kW};
    aclTensor *features_0 = nullptr;
    void *featDev = nullptr;
    int64_t featSize = kN * kC * kH * kW;
    aclrtMalloc(&featDev, featSize * sizeof(int16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(featDev, 0, featSize * sizeof(int16_t), 0, featSize * sizeof(int16_t));
    features_0 = aclCreateTensor(featShape, 4, ACL_FLOAT16, featDev, 0, nullptr, nullptr);

    // rois: [R, 5] fp16
    int64_t roisShape[2] = {kR, 5};
    aclTensor *rois = nullptr;
    void *roisDev = nullptr;
    aclrtMalloc(&roisDev, kR * 5 * sizeof(int16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(roisDev, 0, kR * 5 * sizeof(int16_t), 0, kR * 5 * sizeof(int16_t));
    rois = aclCreateTensor(roisShape, 2, ACL_FLOAT16, roisDev, 0, nullptr, nullptr);

    // y: [R, C, pooled_h, pooled_w] fp16
    int64_t yShape[4] = {kR, kC, kPooledH, kPooledW};
    aclTensor *y = nullptr;
    void *yDev = nullptr;
    int64_t ySize = kR * kC * kPooledH * kPooledW;
    aclrtMalloc(&yDev, ySize * sizeof(int16_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(yDev, 0, ySize * sizeof(int16_t), 0, ySize * sizeof(int16_t));
    y = aclCreateTensor(yShape, 4, ACL_FLOAT16, yDev, 0, nullptr, nullptr);

    // attrs (SPEC §5 fixed order; defaults shown)
    int64_t finestScale = 56;
    double roiScaleFactor = 0.0;
    float spatialScale[4] = {0.25f, 0.125f, 0.0625f, 0.03125f};
    int64_t pooledHeight = kPooledH;
    int64_t pooledWidth = kPooledW;
    int64_t sampleNum = 0;
    const char *poolMode = "avg";
    bool aligned = true;

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus status = aclnnRoiExtractorGetWorkspaceSize(
        features_0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        rois, nullptr, y,
        finestScale, roiScaleFactor, spatialScale, 4,
        pooledHeight, pooledWidth, sampleNum, poolMode, aligned,
        &workspaceSize, &executor);
    if (status != ACLNN_SUCCESS) {
        std::cerr << "get workspace size failed: " << status << std::endl;
        return 1;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    status = aclnnRoiExtractor(workspace, workspaceSize, executor, stream);
    if (status != ACLNN_SUCCESS) {
        std::cerr << "aclnnRoiExtractor failed: " << status << std::endl;
        return 1;
    }
    aclrtSynchronizeStream(stream);

    std::cout << "RoiExtractor aclnn example done" << std::endl;
    return 0;
}
