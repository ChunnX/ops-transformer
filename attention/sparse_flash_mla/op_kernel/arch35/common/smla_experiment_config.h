/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SMLA_EXPERIMENT_CONFIG_H
#define SMLA_EXPERIMENT_CONFIG_H

// Shared by the AI Core kernel and its AICPU metadata producer.
// 0 original; 1 split reference; 2 control; 3 balanced; 4 slot1 phase; 5 S64 balanced.
#ifndef SMLA_L1_LAYOUT_MODE
#define SMLA_L1_LAYOUT_MODE 0
#endif

// Private diagnostic package only; a magic metadata marker is also required.
#ifndef SMLA_BANK_DIAGNOSTIC
#define SMLA_BANK_DIAGNOSTIC 0
#endif

#endif // SMLA_EXPERIMENT_CONFIG_H
