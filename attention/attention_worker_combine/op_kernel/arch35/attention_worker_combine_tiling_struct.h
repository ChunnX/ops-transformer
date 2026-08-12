/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attention_worker_combine_tiling_struct.h
 * \brief
 */

#ifndef ATTENTION_WORKER_COMBINE_TILING_STRUCT_H_
#define ATTENTION_WORKER_COMBINE_TILING_STRUCT_H_

namespace optiling {

struct AttentionWorkerCombineRegbaseTilingData {
    int64_t usedCoreNum{0};
    int64_t BS{0};
    int64_t K{0};
    int64_t H{0};
    int64_t needSchedule{0};
    int64_t BsSplitFactor{0};
    int64_t BsSplitCoreNum{0};
    int64_t mainCoreBsLoopNum{0};
    int64_t tailCoreBsLoopNum{0};
    int64_t HSplitFactor{0};
    int64_t HSplitTailFactor{0};
    int64_t HSplitCoreNum{0};
    int64_t mainCoreHLoopNum{0};
    int64_t tailCoreHLoopNum{0};
    int64_t KSplitFactor{0};
    int64_t KSplitTailFactor{0};
    int64_t KSplitLoopNum{0};
};

} // namespace optiling

#endif // ATTENTION_WORKER_COMBINE_TILING_STRUCT_H_
