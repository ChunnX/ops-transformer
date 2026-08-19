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
 * \file ffn_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FFN_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FFN_H_
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FFNBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
TILING_DATA_FIELD_DEF(uint32_t, k1);
TILING_DATA_FIELD_DEF(uint32_t, n1);
TILING_DATA_FIELD_DEF(uint32_t, n2);
TILING_DATA_FIELD_DEF(uint32_t, expertNum);
TILING_DATA_FIELD_DEF(uint32_t, maxTokens);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, activeType);
TILING_DATA_FIELD_DEF(uint64_t, workspace1Size);
TILING_DATA_FIELD_DEF(uint64_t, workspace2Size);
TILING_DATA_FIELD_DEF(uint32_t, syncWorkspaceSize);
TILING_DATA_FIELD_DEF(uint32_t, dataTypeSize);
TILING_DATA_FIELD_DEF(uint32_t, scale1GroupNum);
TILING_DATA_FIELD_DEF(uint32_t, scale2GroupNum);
TILING_DATA_FIELD_DEF(uint32_t, tokensIndexFlag);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FFNBaseParamsOp, FFNBaseParams)

BEGIN_TILING_DATA_DEF(FFNSingleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, baseM1);
TILING_DATA_FIELD_DEF(uint32_t, baseN1);
TILING_DATA_FIELD_DEF(uint32_t, baseN2);
TILING_DATA_FIELD_DEF(uint32_t, ubCalSize);
TILING_DATA_FIELD_DEF(uint32_t, ubRestBytes);
TILING_DATA_FIELD_DEF(uint32_t, mm1ResUbSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FFNSingleCoreParamsOp, FFNSingleCoreParams)

BEGIN_TILING_DATA_DEF(FFNTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FFNSingleCoreParams, ffnSingleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(FFNBaseParams, ffnBaseParams);
// ---- arch35 (ascend950) fused gelu 路径新增字段：两个 matmul 的紧凑 tiling + hidden/workspace 信息 ----
TILING_DATA_FIELD_DEF(uint32_t, v1UpUsedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, v1UpM);
TILING_DATA_FIELD_DEF(uint32_t, v1UpN);
TILING_DATA_FIELD_DEF(uint32_t, v1UpK);
TILING_DATA_FIELD_DEF(uint32_t, v1UpML1);
TILING_DATA_FIELD_DEF(uint32_t, v1UpNL1);
TILING_DATA_FIELD_DEF(uint32_t, v1UpKL1);
TILING_DATA_FIELD_DEF(uint32_t, v1UpBaseM);
TILING_DATA_FIELD_DEF(uint32_t, v1UpBaseN);
TILING_DATA_FIELD_DEF(uint32_t, v1UpBaseK);
TILING_DATA_FIELD_DEF(uint32_t, v1UpSkSingleCoreK);
TILING_DATA_FIELD_DEF(uint32_t, v1UpMTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, v1UpNTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, v1UpMBaseTailSplitCnt);
TILING_DATA_FIELD_DEF(uint32_t, v1UpNBaseTailSplitCnt);
TILING_DATA_FIELD_DEF(uint32_t, v1UpMTailMain);
TILING_DATA_FIELD_DEF(uint32_t, v1UpNTailMain);
TILING_DATA_FIELD_DEF(uint8_t, v1UpIsHf32);
TILING_DATA_FIELD_DEF(uint8_t, v1UpL1BufferNum);
TILING_DATA_FIELD_DEF(uint8_t, v1UpL0cDB);
TILING_DATA_FIELD_DEF(uint8_t, v1UpUbDB);
TILING_DATA_FIELD_DEF(uint32_t, v1DownUsedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, v1DownM);
TILING_DATA_FIELD_DEF(uint32_t, v1DownN);
TILING_DATA_FIELD_DEF(uint32_t, v1DownK);
TILING_DATA_FIELD_DEF(uint32_t, v1DownML1);
TILING_DATA_FIELD_DEF(uint32_t, v1DownNL1);
TILING_DATA_FIELD_DEF(uint32_t, v1DownKL1);
TILING_DATA_FIELD_DEF(uint32_t, v1DownBaseM);
TILING_DATA_FIELD_DEF(uint32_t, v1DownBaseN);
TILING_DATA_FIELD_DEF(uint32_t, v1DownBaseK);
TILING_DATA_FIELD_DEF(uint32_t, v1DownSkSingleCoreK);
TILING_DATA_FIELD_DEF(uint32_t, v1DownMTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, v1DownNTailCnt);
TILING_DATA_FIELD_DEF(uint32_t, v1DownMBaseTailSplitCnt);
TILING_DATA_FIELD_DEF(uint32_t, v1DownNBaseTailSplitCnt);
TILING_DATA_FIELD_DEF(uint32_t, v1DownMTailMain);
TILING_DATA_FIELD_DEF(uint32_t, v1DownNTailMain);
TILING_DATA_FIELD_DEF(uint8_t, v1DownIsHf32);
TILING_DATA_FIELD_DEF(uint8_t, v1DownL1BufferNum);
TILING_DATA_FIELD_DEF(uint8_t, v1DownL0cDB);
TILING_DATA_FIELD_DEF(uint8_t, v1DownUbDB);
TILING_DATA_FIELD_DEF(uint32_t, v1HiddenOffset);
TILING_DATA_FIELD_DEF(uint32_t, v1HiddenRows);
TILING_DATA_FIELD_DEF(uint32_t, v1HiddenCols);
TILING_DATA_FIELD_DEF(uint8_t, v1IsFp16); // 0: x/weight bf16; 1: x/weight fp16（arch35 新增 fp16 支持）
TILING_DATA_FIELD_DEF(uint8_t, v1BiasIsBf16); // 0: bias float32; 1: bias bf16（arch35 与 linear 对齐）
TILING_DATA_FIELD_DEF(uint8_t, v1BiasIsFp16); // 0: 非 fp16 bias; 1: bias fp16（与 v1BiasIsBf16 互斥）
TILING_DATA_FIELD_DEF(uint8_t, v1HasBias); // 0: 无 bias（swiglu 模型场景）；1: 有 bias
TILING_DATA_FIELD_DEF(uint8_t, v1TransB); // 0: 权重 canonical [K,N]；1: 权重 linear [N,K]，kernel 内 transB
TILING_DATA_FIELD_DEF(uint8_t, v1SwigluSingle); // swiglu 单 matmul（N=2H，B 装载交错）；0 回退旧三段
TILING_DATA_FIELD_DEF(uint8_t, v1UpFullLoad); // up 的 L1 全载模式（0/1/2），host 官方资格判定后透传 kernel
TILING_DATA_FIELD_DEF(uint8_t, v1DownFullLoad); // down 的 L1 全载模式（0/1/2）
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FFN, FFNTilingData)
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FFN_H_