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
 * \file aclnn_matmul_all_reduce_v3.cpp
 * \brief MatmulAllReduceV2 v3 API - reuse MatmulAllReduce implementation
 */

// 引用原算子的 v3 接口实现
// 由于接口逻辑完全一致，可以直接 include 原文件
#include "../matmul_all_reduce/op_api/aclnn_matmul_all_reduce_v3.cpp"

// 注意：原文件中已经定义了 aclnnMatmulAllReduceV3GetWorkspaceSize 和 aclnnMatmulAllReduceV3
// MatmulAllReduceV2 使用相同的接口实现，因为功能完全一致