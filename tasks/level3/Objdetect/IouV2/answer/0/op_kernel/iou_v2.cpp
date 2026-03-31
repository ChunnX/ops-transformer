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
 * \file iou_v2.cpp
 * \brief
 */

#include "iou_v2_align_iof.h"
#include "iou_v2_align_iou.h"
#include "iou_v2_not_align_iof.h"
#include "iou_v2_not_align_iou.h"

using namespace AscendC;
using namespace IouV2;

extern "C" __global__ __aicore__ void iou_v2(
    GM_ADDR bboxes,
    GM_ADDR gtboxes,
    GM_ADDR overlap,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
#define INIT_AND_PROCESS                             \
    op.Init(bboxes, gtboxes, overlap, &tiling_data); \
    op.Process()

    if (TILING_KEY_IS(4))
    {
        KernelIouV2Align<float> op;
        INIT_AND_PROCESS;
    }
    else if (TILING_KEY_IS(7))
    {
        KernelIouV2NotAlign<float> op;
        INIT_AND_PROCESS;
    }
    else if (TILING_KEY_IS(14))
    {
        KernelIofV2Align<float> op;
        INIT_AND_PROCESS;
    }
    else if (TILING_KEY_IS(17))
    {
        KernelIofV2NotAlign<float> op;
        INIT_AND_PROCESS;
    }
    else if (TILING_KEY_IS(5))
    {
        KernelIouV2Align<half> op;
        INIT_AND_PROCESS;
    }
    else if (TILING_KEY_IS(8))
    {
        KernelIouV2NotAlign<half> op;
        INIT_AND_PROCESS;
    }
    else if (TILING_KEY_IS(15))
    {
        KernelIofV2Align<half> op;
        INIT_AND_PROCESS;
    }
    else if (TILING_KEY_IS(18))
    {
        KernelIofV2NotAlign<half> op;
        INIT_AND_PROCESS;
#if __CCE_AICORE__ != 200 && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    }
    else if (TILING_KEY_IS(6))
    {
        KernelIouV2Align<bfloat16_t> op;
        INIT_AND_PROCESS;
    }
    else if (TILING_KEY_IS(9))
    {
        KernelIouV2NotAlign<bfloat16_t> op;
        INIT_AND_PROCESS;
    }
    else if (TILING_KEY_IS(16))
    {
        KernelIofV2Align<bfloat16_t> op;
        INIT_AND_PROCESS;
    }
    else if (TILING_KEY_IS(19))
    {
        KernelIofV2NotAlign<bfloat16_t> op;
        INIT_AND_PROCESS;
#endif
    }
}
