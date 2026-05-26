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
 * \file moe_distribute_combine_ccu_setup.h
 * \brief A6 CCU MoeDistributeCombineCcuSetup 算子三件套声明（v11，对称 dispatch_setup）。
 *
 * 端到端 IO 契约严格对齐库上 op_def
 *   （ops-transformer/mc2/moe_distribute_combine_setup/op_host/moe_distribute_combine_setup_def.cpp）：
 *
 *   Inputs:  expand_x (REQUIRED, BF16/FP16) ← 来自 dispatch_teardown.expand_x 经 expert 处理后的结果
 *            expert_ids (REQUIRED, INT32)   ← 原始 expertIds（与 dispatch 同款）
 *            assist_info_for_combine (REQUIRED, INT32) ← dispatch_teardown.assist_info_for_combine
 *   Outputs: quant_expand_x (REQUIRED, INT8) ← v11 简化版 host 归零（不做量化）
 *            comm_cmd_info  (REQUIRED, INT32) ← 给 combine_teardown 用的 cmd 信息
 *   Attrs:   group_ep (REQUIRED, String)
 *            ep_world_size / ep_rank_id / moe_expert_num (REQUIRED, Int)
 *            expert_shard_type (OPT, Int=0)
 *            shared_expert_num (OPT, Int=1)
 *            shared_expert_rank_num (OPT, Int=0)
 *            global_bs (OPT, Int=0)
 *            comm_quant_mode (OPT, Int=0)  ← 注意是 comm_quant_mode，不是 quant_mode（与 dispatch 不同）
 *            comm_type (OPT, Int=0)
 *            comm_alg (OPT, String="")
 *
 * v11 双流并行架构（与 dispatch v11 完全对称）：
 *   - combine_setup CCU：反向发 expand_x 数据到对端 yCombine window，立即返回（异步）
 *   - combine_teardown AIV：内部 WaitForAllSrcRanks 轮询等齐 → 加权求和 → 输出 x
 *
 * 数据流方向（反向于 dispatch）：
 *   dispatch 时：本卡 token → 对端 y[本卡 epRankId 段]
 *   combine  时：本卡 expand_x[src 段] → 对端 r 的 yCombine[本卡 epRankId 段]
 *                也就是说：本卡作为"combine src"，反向把数据送回原 token 主人。
 *
 * 简化范围（v11）：
 *   - 仅 BF16 非量化（comm_quant_mode = 0；quant_expand_x 输出 host 归零）
 *   - sharedExpertRankNum = 0，sharedExpertNum = 0
 *   - 反向通信简化：本卡 expand_x 整段（[r 段] 共 bs*topK 个槽位）发回对端
 */

#ifndef MOE_DISTRIBUTE_COMBINE_CCU_SETUP_H
#define MOE_DISTRIBUTE_COMBINE_CCU_SETUP_H

#include <cstdint>
#include <string>
#include <vector>
#include "moe_combine_ccu_api.h"
#include "ccu_kernel_arg.h"
#include "ccu_kernel_signature.h"
#include "ccu_task_arg_v1.h"
#include "hccl_types.h"

namespace hcomm {

// 注册期参数（attr）：与 op_def 注册顺序一一对应
struct CcuKernelArgMoeDistributeCombineCcuSetup : public CcuKernelArg {
    std::string groupEp;             // REQUIRED, op_def 第 1 个 attr
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t expertShardType;        // OPT, 默认 0
    uint32_t sharedExpertNum;        // OPT, 默认 1
    uint32_t sharedExpertRankNum;    // OPT, 默认 0
    uint32_t globalBs;               // OPT, 默认 0
    uint32_t commQuantMode;          // OPT, 默认 0（combine 用 commQuantMode 不是 dispatch 的 quantMode）
    uint32_t commType;               // OPT, 默认 0
    std::string commAlg;             // OPT, 默认 ""

    uint32_t bs;                      // shape 派生量
    uint32_t topK;
    uint32_t hiddenSize;
    uint64_t tokenMsgLen;             // = Align256(H) * sizeof(ExpandXType)
    uint32_t moeExpertPerRankNum;     // = moeExpertNum / (epWorldSize - sharedExpertRankNum)

    CcuKernelArgMoeDistributeCombineCcuSetup(
        const std::string &groupEp_,
        uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
        uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
        uint32_t expertShardType_ = 0, uint32_t sharedExpertNum_ = 1,
        uint32_t sharedExpertRankNum_ = 0, uint32_t globalBs_ = 0,
        uint32_t commQuantMode_ = 0, uint32_t commType_ = 0,
        const std::string &commAlg_ = "");

    CcuKernelSignature GetKernelSignature() const override;
};

// 每次 launch 动态参数：字段顺序对齐 op_def Input/Output
//   v16 ★ commWindowAddr 复用 dispatch 同一块 GM buffer（hccl 开辟的中间 buffer）
//   combine 阶段 commWindow layout: (bs*K, tokenWithMetaLen) 单段
//   dispatch 阶段 commWindow layout: (rankSize, bs*K, tokenWithMetaLen) 按段
//   同一块 GM, alloc size = max(两者) = rankSize * bs * K * tokenWithMetaLen
struct CcuTaskArgMoeDistributeCombineCcuSetup : public CcuTaskArg {
    // ----- Inputs (op_def) -----
    uint64_t expandXAddr;            // (A, tokenWithMetaLen) ← v15 紧密+metadata，dispatch_teardown 输出
    uint64_t expertIdsAddr;          // (Bs, K) INT32 ← v16 不再使用（metadata 自带定位信息）
    uint64_t assistInfoForCombineAddr; // INT32 ← v15 紧密三元组数组，v16 device 端不直接读（metadata 内嵌）
    // ----- Outputs (op_def) -----
    uint64_t quantExpandXAddr;       // INT8 ← 简化版 host 归零
    uint64_t commCmdInfoAddr;        // INT32 ← 给 combine_teardown 用
    // ----- Internal (v16 工程内部 scratch) -----
    uint64_t commWindowAddr;         // v16 ★ 复用 dispatch 同一块 hccl commWindow
                                      //   combine 写入 layout: 按 slot=srcSlot*K+dstK 直接索引（不分段）
                                      //   bs*K 个槽（前 0..K-1 都是 token 0 副本，K..2K-1 都是 token 1 副本...）
    uint64_t routingTableAddr;       // [Bs*K] INT32 ← v16 device 端不再使用（保留 IO 契约）

    CcuTaskArgMoeDistributeCombineCcuSetup(
        uint64_t expandXAddr_, uint64_t expertIdsAddr_, uint64_t assistInfoForCombineAddr_,
        uint64_t quantExpandXAddr_, uint64_t commCmdInfoAddr_,
        uint64_t commWindowAddr_, uint64_t routingTableAddr_);
};

// device 算法
class CcuKernelMoeDistributeCombineCcuSetup : public Mc2CcuKernelBase {
public:
    explicit CcuKernelMoeDistributeCombineCcuSetup(const CcuKernelArg &arg);

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    uint32_t epWorldSize_;
    uint32_t epRankId_;
    uint32_t moeExpertNum_;
    uint32_t moeExpertPerRankNum_;
    uint32_t bs_;
    uint32_t topK_;
    uint64_t tokenMsgLen_;

    CcuRep::Variable expandXAddrVar_;
    CcuRep::Variable commCmdInfoAddrVar_;
    CcuRep::Variable commWindowAddrVar_;                  // v16 ★ 替代 yCombineAddrVar_
    CcuRep::Variable routingTableAddrVar_;
    CcuRep::Variable tokenMsgLenVar_;
    std::vector<CcuRep::Variable> remoteCommWindowAddrs_; // v16 ★ 各对端 commWindow 起址
    CcuRep::LocalAddr localSrc_;
    CcuRep::LocalAddr localDst_;
    CcuRep::RemoteAddr remoteDst_;
    CcuRep::CompletedEvent event_;
};

} // namespace hcomm

#endif // MOE_DISTRIBUTE_COMBINE_CCU_SETUP_H
