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
 * \file moe_distribute_combine_ccu_teardown.h
 * \brief A6 MoeDistributeCombineCcuTeardown ascendc kernel 接口声明（v11，对称 dispatch_teardown）。
 *
 * 端到端 IO 契约严格对齐库上 op_def
 *   （ops-transformer/mc2/moe_distribute_combine_teardown/op_host/moe_distribute_combine_teardown_def.cpp）：
 *
 *   Inputs (op_def 顺序):
 *     expand_x (REQUIRED, BF16/FP16)        ← combine_setup 的 expand_x（同一份）
 *     quant_expand_x (REQUIRED, INT8)       ← combine_setup 输出
 *     expert_ids (REQUIRED, INT32)          ← 原始 expertIds
 *     expand_idx (REQUIRED, INT32)          ← dispatch_setup 输出
 *     expert_scales (REQUIRED, FLOAT)       ← 每个 expert 对该 token 的加权
 *     comm_cmd_info (REQUIRED, INT32)       ← combine_setup 输出
 *     x_active_mask (OPTIONAL, BOOL)
 *     shared_expert_x (OPTIONAL, BF16/FP16)
 *   Outputs:
 *     x (REQUIRED, BF16/FP16)               ← 最终还原结果，shape (Bs, H)
 *
 * 双流并行架构（与 dispatch_teardown 完全对称）：
 *   - combine_setup CCU 立即返回（异步），通过 0.5RTT (tag = MOE_COMBINE_CCU_TAG) 反向发数据
 *   - combine_teardown AIV 内部 WaitForAllSrcRanks 轮询 commCmdInfo ready 区 [0,R) 等齐
 *   - 等齐后以 K 为 stride 取 commWindow[i*K]（j=0 副本）→ xOut[i]；identity 下 commWindow[i*K]==x[i]
 *   - v20.6 ★ combine commWindow 为【干净槽】(tokenMsgLen，无 metadata)，stride 改 tokenMsgLen
 *
 * 简化范围（demo）：
 *   - dtype 写死 bfloat16_t
 *   - block 数 = 1
 *   - K-stride 取 j=0 副本（非加权求和）：生产版需对每 i 的 K 个副本 Cast+Muls expertScales+Add
 *   - quant_expand_x / x_active_mask / shared_expert_x device 端不读
 */

#ifndef MOE_DISTRIBUTE_COMBINE_CCU_TEARDOWN_H
#define MOE_DISTRIBUTE_COMBINE_CCU_TEARDOWN_H

#include <cstdint>

// ----- v11 Tiling 结构（host 填充 + device 读取） -----
// 彻底精简版：只保留 Process 实际读取的 tiling 字段。
//   已删 device 不读的字段：epRankId / moeExpertNum / moeExpertPerRankNum /
//     tokenMsgLen（Process 未用）/ segBytes（Process 未用）。
struct MoeDistributeCombineCcuTeardownTilingData {
    uint32_t epWorldSize;        // WaitForAllSrcRanks 用
    uint32_t bs;
    uint32_t topK;
    uint32_t hiddenSize;         // Process copyElemsPerToken 用（dataBytesPerToken）
    uint32_t commQuantMode;      // 0=BF16（2*H 字节）/ 1=MXFP8（H+H/32 字节）：切 copyLen
    uint64_t tokenMsgLen;        // v20.6 ★ combine commWindow 每槽 stride（干净，无 metadata）= Align256(H)*2 / Align32(H+H/32)
};

#endif // MOE_DISTRIBUTE_COMBINE_CCU_TEARDOWN_H
