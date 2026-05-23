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
 * \file moe_distribute_dispatch_ccu_setup.h
 * \brief A6 CCU MoeDistributeDispatchCcuSetup 算子三件套声明（dispatch 发送侧，v3）。
 *
 * 端到端 IO 契约严格对齐现网 moe_distribute_dispatch_setup 算子；
 * 实现走 A6 CCU 0.5RTT 路径（hcomm::CcuKernel 子类 + 0.5RTT 上层接口）。
 *
 * v3 关键变更（绕过 CCU Variable 算术严格清单约束）：
 *   1. 引入 routingTableAddr：host 端预算 expertId → rank 的映射，
 *      device 侧 LoadVariable 查表，去除 Var/const 除法；
 *   2. peerOccBytes_：per-peer 字节偏移累加器，替代 Var*const 乘法；
 *   3. 第 7 阶段 scratch 用 commCmdInfo 后段（[epWorldSize, 2*epWorldSize) 槽位）独占，
 *      消除原 commCmdInfo[0] 既是正式槽位又是 scratch 的双 race。
 *
 * 简化范围：仅 BF16 非量化，sharedExpertRankNum=0，OPT 字段保留位置（传 0）。
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_CCU_SETUP_H
#define MOE_DISTRIBUTE_DISPATCH_CCU_SETUP_H

#include <cstdint>
#include <string>
#include <vector>
#include "moe_dispatch_ccu_api.h"
#include "ccu_kernel_arg.h"
#include "ccu_task_arg_v1.h"
#include "hccl_types.h"

namespace hcomm {

// 注册期参数：attr 一一对应现网 op_def（一个通信域内 immutable），与 KernelSignature 绑定。
struct CcuKernelArgMoeDistributeDispatchCcuSetup : public CcuKernelArg {
    // attr
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t expertShardType;        // 简化版固定 0
    uint32_t sharedExpertNum;        // 简化版固定 0
    uint32_t sharedExpertRankNum;    // 简化版固定 0
    uint32_t quantMode;              // 简化版固定 0（非量化）
    uint32_t globalBs;
    uint32_t commType;               // 2 = URMA / CCU
    uint32_t commAlg;
    // shape 与派生量
    uint32_t bs;
    uint32_t topK;
    uint32_t hiddenSize;
    uint64_t tokenMsgLen;             // hiddenSize * sizeof(bfloat16)
    uint32_t moeExpertPerRankNum;     // moeExpertNum / (epWorldSize - sharedExpertRankNum)
    // 通信域拓扑
    std::vector<uint32_t> subCommRanks;

    CcuKernelArgMoeDistributeDispatchCcuSetup(
        uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
        uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
        std::vector<uint32_t> subCommRanks_,
        uint32_t expertShardType_ = 0, uint32_t sharedExpertNum_ = 0,
        uint32_t sharedExpertRankNum_ = 0, uint32_t quantMode_ = 0,
        uint32_t globalBs_ = 0, uint32_t commType_ = 2, uint32_t commAlg_ = 0);

    std::string GetKernelSignature() override;
};

// 每次 launch 动态参数：与现网 setup 算子输入/输出一一对应。
//
// v3 新增：routingTableAddr —— host 端在 Launch 前预算好的 [bs*topK] int32 数组，
//   routingTable[i*topK + j] = expertIds[i*topK + j] / moeExpertPerRankNum
//   （即 token (i, j) 应该路由到哪个 rank）。device 端通过 LoadVariable
//   查表，绕过 CCU Variable 不支持 Var/const 除法的约束。
//
// expertIdsAddr 字段保留（端到端 IO 契约对齐），device 端不再 Load 它。
struct CcuTaskArgMoeDistributeDispatchCcuSetup : public CcuTaskArg {
    // 输入
    uint64_t xAddr;                   // [Bs, H] BF16     required
    uint64_t expertIdsAddr;           // [Bs, topK] INT32 required（device 端不再读，host 用它算 routingTable）
    uint64_t scalesAddr;              // OPT 简化版传 0
    uint64_t xActiveMaskAddr;         // OPT 简化版传 0
    uint64_t routingTableAddr;        // [Bs*topK] INT32 ★v3 新增（host 预算）
    // 输出
    uint64_t yAddr;                   // 接收 window 起址（本卡待发数据缓冲，对端 GM）
    uint64_t expandIdxAddr;           // [Bs*topK] INT32
    uint64_t commCmdInfoAddr;         // INT32，前 epWorldSize 槽是正式槽位，后 epWorldSize 槽是 scratch

    CcuTaskArgMoeDistributeDispatchCcuSetup(
        uint64_t xAddr_, uint64_t expertIdsAddr_, uint64_t routingTableAddr_,
        uint64_t yAddr_, uint64_t expandIdxAddr_, uint64_t commCmdInfoAddr_,
        uint64_t scalesAddr_ = 0, uint64_t xActiveMaskAddr_ = 0);
};

// device 算法：dispatch 发送侧（按 routingTable 路由，用 0.5RTT WriteNbWithArriveNotify）
class CcuKernelMoeDistributeDispatchCcuSetup : public Mc2CcuKernelBase {
public:
    explicit CcuKernelMoeDistributeDispatchCcuSetup(const CcuKernelArg &arg);

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    // 配置缓存（构造时从 KernelArg 读出）
    uint32_t epWorldSize_;
    uint32_t epRankId_;
    uint32_t moeExpertPerRankNum_;    // v3 device 端不再用（保留供 host 端预算 routingTable 时参考）
    uint32_t bs_;
    uint32_t topK_;
    uint64_t tokenMsgLen_;

    // device 句柄（InitResource 阶段创建）
    CcuRep::Variable xAddrVar_;
    CcuRep::Variable yAddrVar_;
    CcuRep::Variable expandIdxAddrVar_;
    CcuRep::Variable commCmdInfoAddrVar_;
    CcuRep::Variable routingTableAddrVar_;        // v3 ★
    CcuRep::Variable tokenMsgLenVar_;
    std::vector<CcuRep::Variable> remoteYAddrs_;  // PreSync 后拿到的各对端 yAddr
    std::vector<CcuRep::Variable> peerOccurCnts_; // 每对端累计发送 token 数（写 expand_idx）
    std::vector<CcuRep::Variable> peerOccBytes_;  // v3 ★ 每对端累计已发字节数（替代 Var*const 乘法）
    CcuRep::LocalAddr localSrc_;
    CcuRep::LocalAddr localDst_;
    CcuRep::RemoteAddr remoteDst_;
    CcuRep::CompletedEvent event_;
};

} // namespace hcomm

#endif // MOE_DISTRIBUTE_DISPATCH_CCU_SETUP_H
