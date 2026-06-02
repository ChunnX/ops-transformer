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
 * 双流并行架构（与 dispatch 完全对称）：
 *   - combine_setup CCU：反向发 expand_x 数据到对端 commWindow（复用 dispatch 同款），立即返回（异步）
 *   - combine_teardown AIV：内部 WaitForAllSrcRanks 轮询等齐 → K-stride 取 j=0 副本 → 输出 x
 *
 * 数据流方向（反向于 dispatch）：
 *   dispatch 时：本卡 token → 对端 commWindow[本卡 epRankId 段]
 *   combine  时：本卡 expand_x[src 段] → 对端 srcRank 的 commWindow[dstByteOff 槽]
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

// ============ 注册期参数（ATTR）============
//   彻底精简版：只保留 device 端实际使用的 attr。
//   已删 op_def 占位 attr：groupEp / expertShardType / sharedExpertNum / sharedExpertRankNum /
//     globalBs / commType / commAlg（device 不读）。
struct CcuKernelArgMoeDistributeCombineCcuSetup : public CcuKernelArg {
    // ----- ATTR -----
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;            // 仅用于派生 moeExpertPerRankNum
    uint32_t commQuantMode;           // 0=BF16 / 1=MXFP8（切 tokenMsgLen 公式；注意非 dispatch 的 quantMode）
    // ----- shape 派生量 -----
    uint32_t bs;
    uint32_t topK;
    uint32_t hiddenSize;
    uint64_t tokenMsgLen;             // commQuantMode=0: Align256(H)*2；=1: Align32(H+H/32)
    uint32_t moeExpertPerRankNum;     // = moeExpertNum / epWorldSize（host attr 派生量；v20.7 后 device 不再读——段5/7 aBound 已删）

    CcuKernelArgMoeDistributeCombineCcuSetup(
        uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
        uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
        uint32_t commQuantMode_ = 0);

    CcuKernelSignature GetKernelSignature() const override;
};

// ============ 每次 launch 动态参数（INPUT / OUTPUT / INTERNAL）============
//   已删 device 不读的 op_def 占位：expertIdsAddr / assistInfoForCombineAddr（GeneArgs 从未传，
//     metadata 自带定位）/ quantExpandXAddr（host 归零）/ routingTableAddr（Load 了但算法不用）。
struct CcuTaskArgMoeDistributeCombineCcuSetup : public CcuTaskArg {
    // ----- INPUT -----
    uint64_t expandXAddr;            // v20.6 ★ (A, tokenMsgLen) 干净数据（无 metadata）：dispatch_teardown 输出，反向数据源
    uint64_t assistInfoAddr;         // v20.7 ★ (A,1)int64 = dstByteOff/token：dispatch_teardown 写，combine 按段消费的回送偏移
    // ----- OUTPUT -----
    uint64_t commCmdInfoAddr;        // INT32 ready 区 + int64 sendBytes[0..R)/negCnt[0..R)（dispatch_teardown 写、本 setup 读）
    // ----- INTERNAL -----
    uint64_t commWindowAddr;         // 复用 dispatch 同款 hccl commWindow；按 assistInfo.dstByteOff 直接索引
    // ----- TOKEN（需求2 ★，与 dispatch_setup 同款 hccl 范式：每块本地 buffer 一个 host GetTokenInfo token）-----
    uint64_t expandXToken;           // ★ 本卡 expandX（反向数据源）token（段 7 localSrc）
    // v20.7 ★ 删 assistInfoToken：assistInfo 由 LoadVariable 读（读 GM 标量不需 token），无消费者
    uint64_t commWindowToken;        // 本卡 commWindow token（段 7 自反自 localDst）
    uint64_t commCmdInfoToken;       // 本卡 commCmdInfo token（段 5 读 sendBytes/negCnt + 段 10 ready 写）
    // ----- 收口 ★ ready 写源（替代段 10 原 R 次自造 StoreVariableAt → 1 次 hccl 现役 LocalCopyNb）-----
    uint64_t oneBufAddr;             // 收口 ★ oneBuf GM 起址（host 填 R 个 int32=1）
    uint64_t oneBufToken;            // 收口 ★ oneBuf token

    CcuTaskArgMoeDistributeCombineCcuSetup(
        uint64_t expandXAddr_, uint64_t assistInfoAddr_, uint64_t commCmdInfoAddr_, uint64_t commWindowAddr_,
        uint64_t expandXToken_, uint64_t commWindowToken_, uint64_t commCmdInfoToken_,
        uint64_t oneBufAddr_, uint64_t oneBufToken_);
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
    uint32_t bs_;
    uint32_t topK_;
    uint64_t tokenMsgLen_;

    CcuRep::Variable expandXAddrVar_;
    CcuRep::Variable assistInfoAddrVar_;                  // v20.6 ★ assistInfo 路由元信息起址
    CcuRep::Variable commCmdInfoAddrVar_;
    CcuRep::Variable commWindowAddrVar_;                  // v16 ★ 替代 yCombineAddrVar_
    CcuRep::Variable tokenMsgLenVar_;
    // ----- v19 ★ token 寄存器（需求2，hccl 范式）-----
    CcuRep::Variable expandXTokenVar_;                    // 本地 expandX 数据源 token
    CcuRep::Variable commWindowTokenVar_;                 // 本地 commWindow token
    CcuRep::Variable commCmdInfoTokenVar_;                // 本地 commCmdInfo token（段 10 ready 写 dst.token）
    CcuRep::Variable oneBufAddrVar_;                      // 收口 ★ oneBuf 地址（段 10 ready 写 src.addr）
    CcuRep::Variable oneBufTokenVar_;                     // 收口 ★ oneBuf token（段 10 ready 写 src.token）
    std::vector<CcuRep::Variable> remoteCommWindowAddrs_; // v16 ★ 各对端 commWindow 起址
    std::vector<CcuRep::Variable> remoteCommWindowTokens_;  // v19 ★ 各对端 commWindow token
    CcuRep::LocalAddr localSrc_;
    CcuRep::LocalAddr localDst_;
    CcuRep::RemoteAddr remoteDst_;
    CcuRep::CompletedEvent event_;
    CcuRep::CcuBuf localCopyBuf_;   // ★ A6 新 LocalCopyNb 签名要求的 UB 暂存 buf（见 .cpp InitResource 注释）
};

} // namespace hcomm

#endif // MOE_DISTRIBUTE_COMBINE_CCU_SETUP_H
