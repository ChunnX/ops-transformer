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
 * \file aclnn_weight_quant_matmul_all_reduce_v2.cpp
 * \brief MatmulAllReduceV2 weight quant v2 API - reuse MatmulAllReduce implementation
 */

// 引用原算子的 weight quant v2 接口实现
// 由于接口逻辑完全一致，可以直接 include 原文件
#include "../matmul_all_reduce/op_api/aclnn_weight_quant_matmul_all_reduce_v2.cpp"

// 注意：原文件中已经定义了 aclnnWeightQuantMatmulAllReduceV2GetWorkspaceSize 和 aclnnWeightQuantMatmulAllReduceV2
// MatmulAllReduceV2 使用相同的接口实现，因为功能完全一致