/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_matmul_all_reduce_v2_st.cpp
 * \brief System test for aclnn_weight_quant_matmul_all_reduce_v2 (MatmulAllReduceV2)
 */

#include <gtest/gtest.h>
#include "aclnn_weight_quant_matmul_all_reduce_v2.h"

// MatmulAllReduceV2 的 weight quant ST 测试与 MatmulAllReduce 完全一致
// 由于功能相同，可以直接引用原算子的测试逻辑

namespace matmul_all_reduce_v2 {
namespace st {

class WeightQuantMatmulAllReduceV2St : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// 系统测试用例
TEST_F(WeightQuantMatmulAllReduceV2St, BasicTest) {
    // TODO: 添加测试逻辑
    // 由于功能与原算子一致，测试逻辑可以复用
}

} // namespace st
} // namespace matmul_all_reduce_v2