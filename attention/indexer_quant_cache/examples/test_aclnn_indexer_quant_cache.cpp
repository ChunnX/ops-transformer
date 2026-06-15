/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "aclnn/aclnn_base.h"
#include "aclnn_indexer_quant_cache.h"

int main()
{
    // 1. 创建输入 tensor
    // cache (in-place input/output): [2048, 128], dtype=float8_e4m3fn
    int64_t cacheShape[] = {2048, 128};
    aclTensor *cache = aclCreateTensor(
        cacheShape, 2, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, ACL_FORMAT_ND);

    // cacheScale (in-place input/output): [2048, 1], dtype=float32
    int64_t scaleShape[] = {2048, 1};
    aclTensor *cacheScale = aclCreateTensor(
        scaleShape, 2, ACL_FLOAT, ACL_FORMAT_ND, ACL_FORMAT_ND);

    // x: [1024, 128], dtype=float16
    int64_t xShape[] = {1024, 128};
    aclTensor *x = aclCreateTensor(
        xShape, 2, ACL_FLOAT16, ACL_FORMAT_ND, ACL_FORMAT_ND);

    // slotMapping: [1024], dtype=int32
    int64_t slotShape[] = {1024};
    aclTensor *slotMapping = aclCreateTensor(
        slotShape, 1, ACL_INT32, ACL_FORMAT_ND, ACL_FORMAT_ND);

    // 2. 设置属性
    int64_t quantMode = 1;   // Normal block quantization
    bool roundScale = true;
    float xScale = 1.0f;

    // 3. 第一阶段：获取 workspace 大小和 executor
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus status = aclnnIndexerQuantCacheGetWorkspaceSize(
        cache, cacheScale,
        x, slotMapping,
        quantMode, roundScale, xScale,
        &workspaceSize, &executor);

    if (status != ACLNN_SUCCESS) {
        std::cerr << "GetWorkspaceSize failed: " << status << std::endl;
        return -1;
    }

    // 4. 分配 workspace
    void *workspace = nullptr;
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);

    // 5. 第二阶段：执行算子
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    status = aclnnIndexerQuantCache(workspace, workspaceSize, executor, stream);

    if (status != ACLNN_SUCCESS) {
        std::cerr << "Execute failed: " << status << std::endl;
        return -1;
    }

    aclrtSynchronizeStream(stream);

    // 6. 清理
    aclDestroyTensor(cache);
    aclDestroyTensor(cacheScale);
    aclDestroyTensor(x);
    aclDestroyTensor(slotMapping);
    aclrtFree(workspace);
    aclrtDestroyStream(stream);

    std::cout << "IndexerQuantCache test passed!" << std::endl;
    return 0;
}
