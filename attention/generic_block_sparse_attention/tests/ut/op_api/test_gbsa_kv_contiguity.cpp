/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "opdev/op_errno.h"
#include "attention/generic_block_sparse_attention/op_host/op_api/aclnn_generic_block_sparse_attention.h"

namespace {
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;

TensorPtr MakeTensor(const std::vector<int64_t> &shape, aclDataType dtype, int holeAxis = -1)
{
    std::vector<int64_t> strides(shape.size());
    int64_t storageSize = 1;
    for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis) {
        if (axis == holeAxis) {
            storageSize *= 2;
        }
        strides[axis] = storageSize;
        storageSize *= shape[axis];
    }
    return TensorPtr(aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                    ACL_FORMAT_ND, &storageSize, 1, nullptr), aclDestroyTensor);
}

class GbsaKvContiguityTest : public testing::TestWithParam<int> {};

TEST_P(GbsaKvContiguityTest, RejectsNonContiguousTndBeforeExecution)
{
    // Independently exercise K/V holes on each axis, including the PA-only dim0 case.
    const int axis = GetParam() % 3;
    const bool stridedKey = GetParam() < 3;
    auto query = MakeTensor({4, 4, 128}, ACL_FLOAT16);
    auto key = MakeTensor({256, 2, 128}, ACL_FLOAT16, stridedKey ? axis : -1);
    auto value = MakeTensor({256, 2, 128}, ACL_FLOAT16, stridedKey ? -1 : axis);
    auto indices = MakeTensor({2, 4, 2}, ACL_INT32);
    auto counts = MakeTensor({2, 4}, ACL_INT32);
    auto output = MakeTensor({4, 4, 128}, ACL_FLOAT16);
    ASSERT_NE(query, nullptr);
    ASSERT_NE(key, nullptr);
    ASSERT_NE(value, nullptr);
    ASSERT_NE(indices, nullptr);
    ASSERT_NE(counts, nullptr);
    ASSERT_NE(output, nullptr);
    const int64_t blockDims[] = {1, 128};
    std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> blockShape(
        aclCreateIntArray(blockDims, 2), aclDestroyIntArray);
    ASSERT_NE(blockShape, nullptr);
    char layout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    EXPECT_EQ(aclnnGenericBlockSparseAttentionGetWorkspaceSize(
                  query.get(), key.get(), value.get(), indices.get(), counts.get(),
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                  nullptr, nullptr, nullptr, blockShape.get(), 1, layout, layout, 0.08838834764831845,
                  1, 0, 0.0, 0, -1, -1, 0, output.get(), nullptr, &workspaceSize, &executor),
              ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

INSTANTIATE_TEST_SUITE_P(KeyAndValueAxes, GbsaKvContiguityTest, testing::Range(0, 6));
} // namespace
