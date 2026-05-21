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
 * \file moe_combine_ccu_direct.h
 * \brief CCU combine setup / teardown <<<>>> 直调封装函数声明（供 host 侧调用）。
 */

#ifndef MOE_COMBINE_CCU_DIRECT_H
#define MOE_COMBINE_CCU_DIRECT_H

#include <cstdint>

// CCU combine setup —— <<<>>> 直调入口
void moe_distribute_combine_ccu_setup_do(
    uint32_t blockDim, void *stream,
    uint8_t *expandX, uint8_t *expertIds, uint8_t *assistInfoForCombine, uint8_t *quantExpandX,
    uint8_t *commCmdInfoOut, uint8_t *workspaceGM, uint8_t *tilingGM);

// CCU combine teardown —— <<<>>> 直调入口
void moe_distribute_combine_ccu_teardown_do(
    uint32_t blockDim, void *stream,
    uint8_t *expandX, uint8_t *quantExpandX, uint8_t *expertIds, uint8_t *expandIdx,
    uint8_t *expertScales, uint8_t *commCmdInfo, uint8_t *xActiveMask, uint8_t *sharedExpertX,
    uint8_t *XOut, uint8_t *workspaceGM, uint8_t *tilingGM);

#endif  // MOE_COMBINE_CCU_DIRECT_H
