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
 * \file moe_distribute_dispatch_ccu_teardown.h
 * \brief A6 CCU MoeDistributeDispatchCcuTeardown 算子三件套声明（v4，端到端 IO 契约对齐库上 op_def）。
 *
 * 入参出参严格对齐 ops-transformer/mc2/moe_distribute_dispatch_teardown 的 op_def：
 *   Inputs:  x (REQUIRED)
 *            y (REQUIRED)
 *            expert_ids (REQUIRED, INT32)
 *            comm_cmd_info (REQUIRED, INT32)
 *   Outputs: expand_x (REQUIRED)
 *            dynamic_scales (REQUIRED, FLOAT 或 FLOAT8_E8M0)
 *            assist_info_for_combine (REQUIRED, INT32)
 *            expert_token_nums (REQUIRED, INT64)
 *   Attrs:   group_ep (REQUIRED, String)
 *            ep_world_size, ep_rank_id, moe_expert_num (REQUIRED, Int)
 *            expert_shard_type (OPT, Int=0)
 *            shared_expert_num (OPT, Int=1)
 *            shared_expert_rank_num (OPT, Int=0)
 *            quant_mode (OPT, Int=0)
 *            global_bs (OPT, Int=0)
 *            expert_token_nums_type (OPT, Int=1)
 *            comm_type (OPT, Int=0)
 *            comm_alg (OPT, String="")
 *
 * v3→v4 变更：
 *   - KernelArg attr：新增 groupEp (REQUIRED)；删 subCommRanks；commAlg uint32→string；
 *     默认值修正 (sharedExpertNum 0→1, commType 2→0)。
 *   - TaskArg：字段顺序与 op_def 严格对齐；删除 expectedPeerCount（device 端用 epWorldSize_-1 推导）；
 *     dynamic_scales 从 OPTIONAL（默认 0）改为 REQUIRED（caller 必须 alloc buffer）。
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H
#define MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H

#include <cstdint>
#include <string>
#include <vector>
#include "moe_dispatch_ccu_api.h"
#include "ccu_kernel_arg.h"
#include "ccu_kernel_signature.h"
#include "ccu_task_arg_v1.h"
#include "hccl_types.h"

namespace hcomm {

struct CcuKernelArgMoeDistributeDispatchCcuTeardown : public CcuKernelArg {
    std::string groupEp;             // REQUIRED, 签名级占位
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t expertShardType;        // OPT, 默认 0
    uint32_t sharedExpertNum;        // OPT, 默认 1
    uint32_t sharedExpertRankNum;    // OPT, 默认 0
    uint32_t quantMode;              // OPT, 默认 0
    uint32_t globalBs;               // OPT, 默认 0
    uint32_t expertTokenNumsType;    // OPT, 默认 1
    uint32_t commType;               // OPT, 默认 0
    std::string commAlg;             // OPT, 默认 ""

    uint32_t bs;
    uint32_t topK;
    uint32_t hiddenSize;
    uint64_t tokenMsgLen;             // v6: Align256(hiddenSize) * sizeof(YOutType)（非量化）
    uint32_t moeExpertPerRankNum;    // = op_def 的 localExpertNum

    CcuKernelArgMoeDistributeDispatchCcuTeardown(
        const std::string &groupEp_,
        uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
        uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
        uint32_t expertShardType_ = 0, uint32_t sharedExpertNum_ = 1,
        uint32_t sharedExpertRankNum_ = 0, uint32_t quantMode_ = 0,
        uint32_t globalBs_ = 0, uint32_t expertTokenNumsType_ = 1,
        uint32_t commType_ = 0, const std::string &commAlg_ = "");

    CcuKernelSignature GetKernelSignature() const override;
};

// 每次 launch 参数：字段顺序严格对齐 op_def 的 Input/Output。
//   Inputs:  x, y, expert_ids, comm_cmd_info
//   Outputs: expand_x, dynamic_scales, assist_info_for_combine, expert_token_nums
//   dynamic_scales 改为 REQUIRED；v3 简化版本下 device 端不写、host driver sync 后归零。
struct CcuTaskArgMoeDistributeDispatchCcuTeardown : public CcuTaskArg {
    // ----- Inputs (op_def) -----
    uint64_t xAddr;
    uint64_t yAddr;                       // setup 写入的 window 起址
    uint64_t expertIdsAddr;
    uint64_t commCmdInfoAddr;             // 含每 src rank 发送 token 数（setup 跨卡写入）
    // ----- Outputs (op_def) -----
    uint64_t expandXAddr;                 // [rankSize*bs, H]（v3 整段预留布局）
    uint64_t dynamicScalesAddr;           // [bs*topK] FLOAT；v3 简化版 host 归零
    uint64_t assistInfoForCombineAddr;    // INT32 前缀和路由表
    uint64_t expertTokenNumsAddr;         // INT64 (moeExpertPerRankNum,)；v3 简化版 host 归零

    CcuTaskArgMoeDistributeDispatchCcuTeardown(
        uint64_t xAddr_, uint64_t yAddr_, uint64_t expertIdsAddr_, uint64_t commCmdInfoAddr_,
        uint64_t expandXAddr_, uint64_t dynamicScalesAddr_,
        uint64_t assistInfoForCombineAddr_, uint64_t expertTokenNumsAddr_);
};

class CcuKernelMoeDistributeDispatchCcuTeardown : public Mc2CcuKernelBase {
public:
    explicit CcuKernelMoeDistributeDispatchCcuTeardown(const CcuKernelArg &arg);

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    uint32_t epWorldSize_;               // 用来在 Algorithm 内部推导 expectedPeerCount = epWorldSize - 1
    uint32_t epRankId_;
    uint32_t moeExpertPerRankNum_;
    uint32_t bs_;
    uint32_t topK_;
    uint64_t tokenMsgLen_;

    CcuRep::Variable yAddrVar_;
    CcuRep::Variable expandXAddrVar_;
    CcuRep::Variable assistInfoAddrVar_;
    CcuRep::Variable commCmdInfoAddrVar_;
    CcuRep::Variable expectedPeerCountVar_;   // = epWorldSize_ - 1（host const，Var=Imm 合法）
    CcuRep::Variable tokenMsgLenVar_;
    CcuRep::LocalAddr srcAddr_;
    CcuRep::LocalAddr dstAddr_;
    CcuRep::CompletedEvent event_;
};

} // namespace hcomm

#endif // MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H
