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
 * v11 双流并行架构（与 dispatch_teardown v11 完全对称）：
 *   - combine_setup CCU 立即返回（异步），通过 0.5RTT (tag = MOE_COMBINE_CCU_TAG) 反向发数据
 *   - combine_teardown AIV 内部 WaitForAllSrcRanks 轮询 commCmdInfo sentinel 等齐
 *   - 等齐后做加权求和：对每个本卡 (i, j)，从 yCombine[rank 段, slot 槽] 读 expert 结果，
 *     乘以 expert_scales[i*K+j]，累加到 x[i]
 *
 * 简化范围（v11 demo）：
 *   - dtype 写死 bfloat16_t
 *   - block 数 = 1
 *   - 加权求和路径：v11 简化为"K 次覆盖"（每个 (i, j) DataCopy yCombine → x[i]，
 *     最后一次覆盖；正确加权求和待生产版实施。详见 OPEN_QUESTIONS）
 *   - quant_expand_x / x_active_mask / shared_expert_x device 端不读
 */

#ifndef MOE_DISTRIBUTE_COMBINE_CCU_TEARDOWN_H
#define MOE_DISTRIBUTE_COMBINE_CCU_TEARDOWN_H

#include <cstdint>

// ----- v11 Tiling 结构（host 填充 + device 读取） -----
struct MoeDistributeCombineCcuTeardownTilingData {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t bs;
    uint32_t topK;
    uint32_t hiddenSize;
    uint32_t commQuantMode;       // 0=非量化（v11 仅支持）；> 0 量化路径暂未实现
    uint64_t tokenMsgLen;          // = Align256(H) * sizeof(YOutType)
    uint64_t tokenWithMetaLen;     // v15 ★ = tokenMsgLen + 32（yCombine 中每槽含 token data + metadata）
    uint64_t segBytes;             // = bs * topK * tokenWithMetaLen
};

#endif // MOE_DISTRIBUTE_COMBINE_CCU_TEARDOWN_H
