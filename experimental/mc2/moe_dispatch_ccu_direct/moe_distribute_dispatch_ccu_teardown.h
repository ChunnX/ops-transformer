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
 * \brief A6 CCU MoeDistributeDispatchCcuTeardown 算子三件套声明（dispatch 接收侧，v3）。
 *
 * v3 关键修复：
 *   - 不再做 Var × Var 字节数 (srcTokens[r] * tokenMsgLen)：每段固定整段拷贝
 *     `bs * tokenMsgLen` 字节（host const），expand_x 整体布局 [r*bs : (r+1)*bs]，
 *     未使用槽位填 0。assist_info_for_combine 前缀和仍按精确 srcTokens 累加。
 *   - 不再做 Var / const expert_token_nums 均分：device 端跳过 expertTokenNums 写入，
 *     host 端在 sync 后 memset 归零作为简化版默认值（端到端 IO 契约：字段存在且
 *     可被 caller 读到一个合法值）。
 *
 * 端到端 IO 契约保留：所有原有输入/输出字段在 TaskArg 中存在，可被现网 caller 直接对接。
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H
#define MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H

#include <cstdint>
#include <string>
#include <vector>
#include "moe_dispatch_ccu_api.h"
#include "ccu_kernel_arg.h"
#include "ccu_task_arg_v1.h"
#include "hccl_types.h"

namespace hcomm {

struct CcuKernelArgMoeDistributeDispatchCcuTeardown : public CcuKernelArg {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t expertShardType;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    uint32_t quantMode;
    uint32_t globalBs;
    uint32_t commType;
    uint32_t commAlg;
    uint32_t expertTokenNumsType;
    uint32_t bs;
    uint32_t topK;
    uint32_t hiddenSize;
    uint64_t tokenMsgLen;
    uint32_t moeExpertPerRankNum;
    std::vector<uint32_t> subCommRanks;

    CcuKernelArgMoeDistributeDispatchCcuTeardown(
        uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
        uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
        std::vector<uint32_t> subCommRanks_,
        uint32_t expertTokenNumsType_ = 1,
        uint32_t expertShardType_ = 0, uint32_t sharedExpertNum_ = 0,
        uint32_t sharedExpertRankNum_ = 0, uint32_t quantMode_ = 0,
        uint32_t globalBs_ = 0, uint32_t commType_ = 2, uint32_t commAlg_ = 0);

    std::string GetKernelSignature() override;
};

// 每次 launch 参数：expectedPeerCount = rankSize - 1（不是字节数）。
// expertTokenNumsAddr 字段保留（IO 契约），v3 device 端不再写入，host 端 sync 后归零。
struct CcuTaskArgMoeDistributeDispatchCcuTeardown : public CcuTaskArg {
    uint64_t xAddr;
    uint64_t yAddr;                       // setup 写入的 window 起址
    uint64_t expertIdsAddr;
    uint64_t commCmdInfoAddr;             // setup 写入的 commCmdInfo（含每 src rank 发送 token 数）
    uint64_t expandXAddr;                 // [rankSize*bs, H] BF16（v3 整段预留布局）
    uint64_t dynamicScalesAddr;           // OPT 简化版传 0
    uint64_t assistInfoForCombineAddr;    // INT32 前缀和路由表
    uint64_t expertTokenNumsAddr;         // INT64/INT32 (moeExpertPerRankNum,)；v3 device 端不写
    uint32_t expectedPeerCount;           // = rankSize - 1，给 WaitAllPeersArriveNotify

    CcuTaskArgMoeDistributeDispatchCcuTeardown(
        uint64_t xAddr_, uint64_t yAddr_, uint64_t expertIdsAddr_, uint64_t commCmdInfoAddr_,
        uint64_t expandXAddr_, uint64_t assistInfoForCombineAddr_, uint64_t expertTokenNumsAddr_,
        uint32_t expectedPeerCount_, uint64_t dynamicScalesAddr_ = 0);
};

class CcuKernelMoeDistributeDispatchCcuTeardown : public Mc2CcuKernelBase {
public:
    explicit CcuKernelMoeDistributeDispatchCcuTeardown(const CcuKernelArg &arg);

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    uint32_t epWorldSize_;
    uint32_t epRankId_;
    uint32_t moeExpertPerRankNum_;
    uint32_t bs_;
    uint32_t topK_;
    uint64_t tokenMsgLen_;

    CcuRep::Variable yAddrVar_;
    CcuRep::Variable expandXAddrVar_;
    CcuRep::Variable assistInfoAddrVar_;
    CcuRep::Variable commCmdInfoAddrVar_;
    CcuRep::Variable expectedPeerCountVar_;
    CcuRep::Variable tokenMsgLenVar_;
    CcuRep::LocalAddr srcAddr_;
    CcuRep::LocalAddr dstAddr_;
    CcuRep::CompletedEvent event_;
};

} // namespace hcomm

#endif // MOE_DISTRIBUTE_DISPATCH_CCU_TEARDOWN_H
