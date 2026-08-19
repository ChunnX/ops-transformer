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
 * \file fia_exp_stage_switch.h
 * \brief arch35 FIA MLA decode 性能分阶段归因实验的阶段开关编译宏
 *
 * 修改本文件宏值（或编译时 -D 覆盖）即可一键开启/关闭各阶段的"计算主体"，用于性能分阶段归因实验：
 *   1 = 开启该阶段计算（默认）
 *   0 = 关闭该阶段计算（仅保留同步骨架）
 *
 * 注意：所有同步骨架（WaitCrossCore/SetCrossCore/SetFlag/WaitFlag/event 配对/拷贝指令）无论宏开关
 * 与否都必须保留（防死锁铁律），宏只包"计算主体"，不得包含任何同步语句。
 * 默认全 1 = 完整功能，性能与基线一致。
 */
#ifndef FIA_EXP_STAGE_SWITCH_H_
#define FIA_EXP_STAGE_SWITCH_H_

// MLA 性能阶段开关（性能实验用）: 1=开启该阶段计算, 0=关闭(保留同步骨架)
#ifndef FIA_EXP_STAGE_C1
#define FIA_EXP_STAGE_C1 1
#endif
// C2 / V1 / V2 / FD 同理
#ifndef FIA_EXP_STAGE_C2
#define FIA_EXP_STAGE_C2 1
#endif
#ifndef FIA_EXP_STAGE_V1
#define FIA_EXP_STAGE_V1 1
#endif
#ifndef FIA_EXP_STAGE_V2
#define FIA_EXP_STAGE_V2 1
#endif
#ifndef FIA_EXP_STAGE_FD
#define FIA_EXP_STAGE_FD 1
#endif

#endif // FIA_EXP_STAGE_SWITCH_H_