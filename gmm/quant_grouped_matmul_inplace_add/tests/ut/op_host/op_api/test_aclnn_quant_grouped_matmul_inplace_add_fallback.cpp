/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

extern "C" {
ge::graphStatus QuantGroupedMatmulInplaceAddFallbackUT_NullContext();
aclTensor *QuantGroupedMatmulInplaceAddFallbackUT_NullTensor(bool enableTranspose, size_t index);
}

namespace {

TEST(aclnn_qgmm_inplace_add_fallback, ge_tensor_to_acl_tensor_null_input)
{
    auto *ret = QuantGroupedMatmulInplaceAddFallbackUT_NullTensor(false, 0);
    EXPECT_EQ(ret, nullptr);
}

TEST(aclnn_qgmm_inplace_add_fallback, execute_func_null_context_fail)
{
    auto ret = QuantGroupedMatmulInplaceAddFallbackUT_NullContext();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

} // namespace
