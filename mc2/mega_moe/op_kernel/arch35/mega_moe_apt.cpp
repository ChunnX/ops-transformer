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
 * \file mega_moe_apt.cpp
 * \brief MegaMoE 的设备侧入口：按通信拓扑选择 MTE 或 URMA 实现。
 */

#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)
#define ENABLE_TENSOR_API
#endif

#include "kernel_operator.h"

#ifdef ENABLE_TENSOR_API
#include "mega_moe.h"
#include "mega_moe_layered.h"
#endif

#include "mega_moe_tiling.h"
#include "mega_moe_tiling_key.h"

using namespace AscendC;
#ifdef ENABLE_TENSOR_API
using namespace MegaMoeImpl;
#endif

// 这四个模板参数不是 mega_moe() 的运行时形参，而是从 Host 设置的 TilingKey 解码得到：
//   DispatchQuantMode    <- 属性 dispatch_quant_mode，当前 Arch35 仅接受 4（MXFP）；
//   DispatchQuantOutType <- 属性 dispatch_quant_out_dtype，经 Host 映射为 3(E5M2)/4(E4M3FN)/5(E2M1)；
//   CombineQuantOutType  <- 属性 combine_quant_mode，取 0(不量化)/3(E5M2)/4(E4M3FN)；
//   CommModeType         <- TilingKey 的通信模式位，取 0(MTE)/1(URMA)。
// Host 通过 GET_TPL_TILING_KEY 生成 key，AscendC 再从 mega_moe_tiling_key.h 声明的候选组合中选择实例。
// DTYPE_X、DTYPE_Y 等 DTYPE_* 宏则由算子输入/输出描述在编译时生成，与上述四个属性参数来源不同。
template <uint8_t DispatchQuantMode, uint8_t DispatchQuantOutType, uint8_t CombineQuantOutType, uint8_t CommModeType>
__global__ __aicore__ void mega_moe(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                    GM_ADDR weight2, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR bias1,
                                    GM_ADDR bias2, GM_ADDR xActiveMask, GM_ADDR scales, GM_ADDR yOut,
                                    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // bias1/bias2 是公共算子接口保留形参，当前 Arch35 GMM 路径未下传使用；
    // scales 虽传入 Init，当前 MTE 实现也不读取它。
    // 真正参与 MX GMM 的 scale 是 weightScales1/2 与运行时生成的激活 scale。
    InitSocState();
    // 每个逻辑 block 由 1 个 AIC（矩阵乘）和 2 个 AIV（路由、向量处理、通信控制）协作完成。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // tilingGM 是 Host tiling 阶段写入的运行时数据区；它保存 bs、H、topK、EP 大小等形状与资源参数。
    // 这些字段与模板参数不同：模板参数决定编译期分支，tilingData 决定本次调用的循环边界和地址布局。
    REGISTER_TILING_DEFAULT(MegaMoeTilingData);
    GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingData, tilingData, tilingGM);
#if defined(ENABLE_TENSOR_API) && defined(ORIG_DTYPE_X) && (ORIG_DTYPE_X == DT_BF16) && defined(ORIG_DTYPE_Y) &&       \
    (ORIG_DTYPE_Y == DT_BF16) && defined(ORIG_DTYPE_WEIGHT1) &&                                                        \
    ((ORIG_DTYPE_WEIGHT1 == DT_FLOAT8_E5M2) || (ORIG_DTYPE_WEIGHT1 == DT_FLOAT8_E4M3FN) ||                             \
     (ORIG_DTYPE_WEIGHT1 == DT_FLOAT4_E2M1)) &&                                                                        \
    defined(ORIG_DTYPE_WEIGHT2) && (ORIG_DTYPE_WEIGHT2 == ORIG_DTYPE_WEIGHT1)
    if constexpr (CommModeType == TILINGKEY_TPL_MTE) {
        if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
            // MTE 路径：源 rank 先写量化通信窗口，专家 rank 根据 mask 主动 Pull 所需路由项。
            // 两个“OutType”继续向下传给 MegaMoe，分别成为 QuantMode 和 CombineQuantMode 模板参数。
            MegaMoe<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DTYPE_WEIGHT1, DispatchQuantOutType, CombineQuantOutType> op;
            op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask, weightScales1, weightScales2,
                    scales, yOut, expertTokenNumsOut, workspaceGM, &tilingData);
            op.Process();
        }
    } else if constexpr (CommModeType == TILINGKEY_TPL_URMA) {
        if constexpr (DispatchQuantMode == DISPATCH_QUANT_MODE_MXFP) {
            // URMA 路径采用分层通信实现；算子语义仍是 Dispatch → Expert FFN → Combine。
            MegaMoeLayered<DTYPE_X, DTYPE_Y, DTYPE_TOPK_WEIGHTS, DTYPE_WEIGHT1, DispatchQuantOutType,
                           CombineQuantOutType>
                op;
            op.Init(context, x, topkIds, topkWeights, weight1, weight2, xActiveMask, weightScales1, weightScales2,
                    scales, yOut, expertTokenNumsOut, workspaceGM, &tilingData);
            op.Process();
        }
    }
#endif
}
