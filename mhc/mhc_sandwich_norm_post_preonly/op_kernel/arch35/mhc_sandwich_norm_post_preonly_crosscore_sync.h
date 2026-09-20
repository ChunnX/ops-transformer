/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

#ifndef MHC_SANDWICH_NORM_POST_PREONLY_CROSSCORE_SYNC_H_
#define MHC_SANDWICH_NORM_POST_PREONLY_CROSSCORE_SYNC_H_

// mode1只在同一物理AI Core的两个sibling AIV之间握手。
// 调用点保持无条件控制流，保证每个lane、每个token各执行且只执行一次Set/Wait。
__aicore__ inline void MhcPairReadySync()
{
    CrossCoreSetFlag<0x1, PIPE_MTE3>(PAIR_READY_FLAG);
    CrossCoreWaitFlag<0x1, PIPE_MTE2>(PAIR_READY_FLAG);
}

#endif // MHC_SANDWICH_NORM_POST_PREONLY_CROSSCORE_SYNC_H_
