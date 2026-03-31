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
}
