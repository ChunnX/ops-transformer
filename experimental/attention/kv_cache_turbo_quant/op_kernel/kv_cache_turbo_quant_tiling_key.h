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
 * \file kv_cache_turbo_quant_tiling_key.h
 * \brief host（kv_cache_turbo_quant_tiling.cpp）与 kernel（kv_cache_turbo_quant.cpp）共用的 tilingkey 常量。
 *
 * tilingkey 编码 mse_bits 主量化位宽：2/3/4 bit 对应 202/203/204（= 200 + mse_bits）。
 * 位宽编译期固化后 kernel 侧生成无分支的量化/打包代码。
 *
 * 必须用 #define 宏（仓内惯例，chunk_gated_delta_rule 同款）：出包工具链对内核二进制做
 * tilingKey 提取时只解析宏字面值，constexpr 变量解析不到会全部编成 tilingKey=0，
 * 真机运行时框架报 "Cannot find tilingKey[203] in kernel json" → aclnn 161002 全挂。
 */
#ifndef KV_CACHE_TURBO_QUANT_TILING_KEY_H
#define KV_CACHE_TURBO_QUANT_TILING_KEY_H

#define KVTQ_TILING_KEY_MSE2 202UL
#define KVTQ_TILING_KEY_MSE3 203UL
#define KVTQ_TILING_KEY_MSE4 204UL

#endif // KV_CACHE_TURBO_QUANT_TILING_KEY_H
