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
 * \file mc2_moe_context_tensor.h
 * \brief
 */

 #include "mc2_moe_context.h"
 #include "opdev/platform.h"
 #include "hccl/hcom.h"
 #include "hccl/hccl_rank_graph.h"

namespace Mc2MoeDistributeContext {

constexpr uint64_t MaxContextTagSize = 255; // 最大上下文标签大小

extern inline aclnnStatus GetMc2ContextTensor(const char* groupEp, const char* opName, uint64_t& hcclBuffSize, aclTensor* &mc2Context)
    return;
}

#endif // MC2_MOE_CONTEXT_H