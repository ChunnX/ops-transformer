/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mhc_pre_split_post_res_tiling_key.h
 * \brief MhcPreSplitPostRes 模板参数 schMode 与 TilingKey 的映射声明（APT 链路）
 *
 * Kernel 入口以 template <uint32_t schMode> + if constexpr 编译期派发 SINGLE/MULTI_TILE，
 * 不再使用 TILING_KEY_IS 运行时分支；模板实参由 Host 侧 GET_TPL_TILING_KEY(schMode) 编码进
 * TilingKey（单参数 1bit，编码值即 schMode 原值 0/1），框架按下方 SEL 表实例化对应 kernel。
 *
 * 本文件 #define 与 op_host/mhc_pre_split_post_res_tiling.h 的
 * optiling::MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE/MULTI_TILE 常量同值互锚
 * （kernel 侧无法 include op_host 头，Host/UT 侧以 tiling.h 为源，两侧改动必须同步）。
 */

#ifndef __OP_KERNEL_MHC_PRE_SPLIT_POST_RES_TILING_KEY_H__
#define __OP_KERNEL_MHC_PRE_SPLIT_POST_RES_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

// schMode 取值：0=SINGLE_TILE（dimLoop==1，D<=2560），1=MULTI_TILE（dimLoop>1）
#define MHC_PRS_SCH_MODE_SINGLE_TILE 0
#define MHC_PRS_SCH_MODE_MULTI_TILE 1

ASCENDC_TPL_ARGS_DECL(MhcPreSplitPostRes,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, MHC_PRS_SCH_MODE_SINGLE_TILE,
                                            MHC_PRS_SCH_MODE_MULTI_TILE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          MHC_PRS_SCH_MODE_SINGLE_TILE)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          MHC_PRS_SCH_MODE_MULTI_TILE)));

#endif  // __OP_KERNEL_MHC_PRE_SPLIT_POST_RES_TILING_KEY_H__
