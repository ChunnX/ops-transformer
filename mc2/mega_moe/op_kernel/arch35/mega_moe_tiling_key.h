/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mega_moe_tiling_key.h
 * \brief 定义 Host 与 Device 共用的 MegaMoE 模板参数及其 TilingKey 编解码规则。
 */

#ifndef MEGA_MOE_TILING_KEY_H
#define MEGA_MOE_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

namespace Mc2Tiling {
// 以下数值是写入 TilingKey 的紧凑内部编码，不是 C++ 数据类型本身：
//   1. Host 从算子属性读取配置；
//   2. GET_TPL_TILING_KEY 将四个编码合成为一个 tilingKey；
//   3. 运行时根据 tilingKey 选择 mega_moe_apt.cpp 中对应的模板实例。
//
// DispatchQuantMode：Dispatch 激活量化算法。目前 Arch35 只支持 4（MXFP 分组量化）。
#define DISPATCH_QUANT_MODE_MXFP 4
// DispatchQuantOutType：源 rank 的 BF16 x 在通信前量化成的数据类型；scale 使用 E8M0。
#define DISPATCH_QUANT_OUT_DTYPE_E5M2 3
#define DISPATCH_QUANT_OUT_DTYPE_E4M3FN 4
#define DISPATCH_QUANT_OUT_DTYPE_E2M1 5
// CombineQuantOutType：专家输出写回源 rank 时是否量化，以及量化后的 FP8 类型。
#define COMBINE_QUANT_OUT_TYPE_NO_QUANT 0
#define COMBINE_QUANT_OUT_TYPE_E5M2 3
#define COMBINE_QUANT_OUT_TYPE_E4M3FN 4
// CommModeType：跨 rank 数据搬运实现；MTE 为通信窗口读写，URMA 为分层通信。
#define TILINGKEY_TPL_MTE 0
#define TILINGKEY_TPL_URMA 1

// 四项声明顺序必须与 Host 的 GET_TPL_TILING_KEY(...) 实参顺序、设备入口的四个模板形参顺序一致。
// ASCENDC_TPL_* 宏会为候选值生成模板实例，并在运行时用 tilingKey 选中其中一个实例。
ASCENDC_TPL_ARGS_DECL(MegaMoe,
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_DISPATCH_QUANT_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                            DISPATCH_QUANT_MODE_MXFP),
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_DISPATCH_QUANT_OUT_DTYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                            DISPATCH_QUANT_OUT_DTYPE_E5M2, DISPATCH_QUANT_OUT_DTYPE_E4M3FN,
                                            DISPATCH_QUANT_OUT_DTYPE_E2M1),
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_COMBINE_QUANT_OUT_TYPE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST,
                                            COMBINE_QUANT_OUT_TYPE_NO_QUANT, COMBINE_QUANT_OUT_TYPE_E5M2,
                                            COMBINE_QUANT_OUT_TYPE_E4M3FN),
                      ASCENDC_TPL_UINT_DECL(TILINGKEY_COMM_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            TILINGKEY_TPL_MTE, TILINGKEY_TPL_URMA), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(TILINGKEY_DISPATCH_QUANT_MODE, ASCENDC_TPL_UI_LIST, DISPATCH_QUANT_MODE_MXFP),
        ASCENDC_TPL_UINT_SEL(TILINGKEY_DISPATCH_QUANT_OUT_DTYPE, ASCENDC_TPL_UI_LIST, DISPATCH_QUANT_OUT_DTYPE_E5M2,
                             DISPATCH_QUANT_OUT_DTYPE_E4M3FN, DISPATCH_QUANT_OUT_DTYPE_E2M1),
        ASCENDC_TPL_UINT_SEL(TILINGKEY_COMBINE_QUANT_OUT_TYPE, ASCENDC_TPL_UI_LIST, COMBINE_QUANT_OUT_TYPE_NO_QUANT,
                             COMBINE_QUANT_OUT_TYPE_E5M2, COMBINE_QUANT_OUT_TYPE_E4M3FN),
        ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_MTE, TILINGKEY_TPL_URMA),
        ASCENDC_TPL_TILING_STRUCT_SEL(MegaMoeTilingData)), );
} // namespace Mc2Tiling
#endif
