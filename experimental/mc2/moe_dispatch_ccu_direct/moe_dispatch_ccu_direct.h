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
 * \file moe_dispatch_ccu_direct.h
 * \brief CCU dispatch setup / teardown <<<>>> 直调封装函数声明（供 host 侧调用）。
 */

#ifndef MOE_DISPATCH_CCU_DIRECT_H
#define MOE_DISPATCH_CCU_DIRECT_H

#include <cstdint>

// CCU dispatch setup —— <<<>>> 直调入口
void moe_distribute_dispatch_ccu_setup_do(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *expertIds, uint8_t *scales, uint8_t *xActiveMask,
    uint8_t *YOut, uint8_t *expandIdxOut, uint8_t *commCmdInfoOut,
    uint8_t *workspaceGM, uint8_t *tilingGM);

// CCU dispatch teardown —— <<<>>> 直调入口
void moe_distribute_dispatch_ccu_teardown_do(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *y, uint8_t *expertIds, uint8_t *commCmdInfo,
    uint8_t *expandXOut, uint8_t *dynamicScalesOut, uint8_t *assistInfoForCombineOut,
    uint8_t *expertTokenNumsOut, uint8_t *workspaceGM, uint8_t *tilingGM);

#endif  // MOE_DISPATCH_CCU_DIRECT_H
