/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TRACE_CONFIG_H
#define MEGA_MOE_TRACE_CONFIG_H

// 主流程诊断开关：同时控制 kernel、InferShape 和 tiling；诊断构建时手动改为 1。
#define MEGA_MOE_ENABLE_A4W4_WAVE_TRACE 0

#if MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 0 && MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 1
#error "MEGA_MOE_ENABLE_A4W4_WAVE_TRACE must be 0 or 1"
#endif

#endif // MEGA_MOE_TRACE_CONFIG_H
