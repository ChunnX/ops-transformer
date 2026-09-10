/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the LICENSE file in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_roi_extractor.cpp
 * \brief aclnnRoiExtractor example: 4-level FPN feature extraction.
 */
#include <cstdint>
#include <cstdio>
#include <vector>
#include "aclnn/opdev/platform.h"
#include "common.h"
#include "aclnn_roi_extractor.h"

using namespace op;

constexpr int64_t LEVELS = 4;
constexpr int64_t N = 1;
constexpr int64_t C = 64;
constexpr int64_t K = 16;
constexpr int64_t POOLED_H = 7;
constexpr int64_t POOLED_W = 7;

static bool RunRoiExtractor()
{
    int64_t levelH[LEVELS] = {128, 64, 32, 16};
    int64_t levelW[LEVELS] = {128, 64, 32, 16};
    float spatialScales[LEVELS] = {0.25f, 0.125f, 0.0625f, 0.03125f};

    auto featsArr = new aclTensor *[LEVELS];
    for (int64_t l = 0; l < LEVELS; l++) {
        std::vector<int64_t> dims = {N, C, levelH[l], levelW[l]};
        std::vector<int64_t> strides = {C * levelH[l] * levelW[l], levelH[l] * levelW[l], levelW[l], 1};
        int64_t size = N * C * levelH[l] * levelW[l];
        featsArr[l] = CreateAclTensor(dims, strides, ACL_FLOAT, ACL_FORMAT_ND, size);
    }
    auto featsList = aclCreateTensorList(const_cast<const aclTensor **>(featsArr), LEVELS);

    std::vector<int64_t> roisDims = {K, 5};
    std::vector<int64_t> roisStrides = {5, 1};
    auto rois = CreateAclTensor(roisDims, roisStrides, ACL_FLOAT, ACL_FORMAT_ND, K * 5);

    std::vector<int64_t> yDims = {K, C, POOLED_H, POOLED_W};
    std::vector<int64_t> yStrides = {C * POOLED_H * POOLED_W, POOLED_H * POOLED_W, POOLED_W, 1};
    auto y = CreateAclTensor(yDims, yStrides, ACL_FLOAT, ACL_FORMAT_ND, K * C * POOLED_H * POOLED_W);

    auto scaleArr = aclCreateFloatArray(spatialScales, LEVELS);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnRoiExtractorGetWorkspaceSize(featsList, rois, nullptr, 56, 0.0, scaleArr, POOLED_H, POOLED_W, 0,
                                                 const_cast<char *>("avg"), true, y, &workspaceSize, &executor);
    if (ret != ACLNN_SUCCESS) {
        printf("aclnnRoiExtractorGetWorkspaceSize failed, ret=%d.\n", static_cast<int>(ret));
        return false;
    }
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    ret = aclnnRoiExtractor(workspace, workspaceSize, executor, GetCurrentStream());
    if (ret != ACLNN_SUCCESS) {
        printf("aclnnRoiExtractor failed, ret=%d.\n", static_cast<int>(ret));
        return false;
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    printf("aclnnRoiExtractor succeeded.\n");
    return true;
}

int main(int argc, char **argv)
{
    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        printf("Set device failed.\n");
        return -1;
    }
    bool ok = RunRoiExtractor();
    aclrtResetDevice(0);
    return ok ? 0 : -1;
}
