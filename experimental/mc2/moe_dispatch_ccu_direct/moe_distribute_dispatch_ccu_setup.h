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
 * \brief A6 CCU MoeDistributeDispatchCcuSetup 算子三件套声明（v4，端到端 IO 契约对齐库上 op_def）。
 *
 * 入参出参严格对齐 ops-transformer/mc2/moe_distribute_dispatch_setup 的 op_def：
 *   Inputs:  x (REQUIRED, BF16/FP16/...)
 *            expert_ids (REQUIRED, INT32)
 *            scales (OPTIONAL, FLOAT)
 *            x_active_mask (OPTIONAL, BOOL)
 *   Outputs: y (REQUIRED)
 *            expand_idx (REQUIRED, INT32)
 *            comm_cmd_info (REQUIRED, INT32)
 *   Attrs:   group_ep (REQUIRED, String)
 *            ep_world_size (REQUIRED, Int)
 *            ep_rank_id (REQUIRED, Int)
 *            moe_expert_num (REQUIRED, Int)
 *            expert_shard_type (OPT, Int=0)
 *            shared_expert_num (OPT, Int=1)
 *            shared_expert_rank_num (OPT, Int=0)
 *            quant_mode (OPT, Int=0)
 *            global_bs (OPT, Int=0)
 *            comm_type (OPT, Int=0)
 *            comm_alg (OPT, String="")
 *
 * v3→v4 变更：
 *   - KernelArg attr：新增 groupEp (REQUIRED)；删除 subCommRanks（库上无此 attr，v3 算法亦未用）；
 *     commAlg 类型 uint32→std::string；默认值修正（sharedExpertNum 0→1，commType 2→0，commAlg ""→""）。
 *   - TaskArg：字段顺序与 op_def Input/Output 严格对齐 (x, expertIds, scales, xActiveMask, y,
 *     expandIdx, commCmdInfo)；routingTableAddr 保留为 v3 算法内部 scratch（库上 op_def 无此字段，
 *     仅因 CCU Variable 不支持 Var/const 除法、必须 host 端预算 LUT 而存在）。
 *
 * 简化范围：仅 BF16 非量化，sharedExpertRankNum=0，scales/xActiveMask 输入字段保留位置传 0。
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_CCU_SETUP_H
#define MOE_DISTRIBUTE_DISPATCH_CCU_SETUP_H

#include <cstdint>
#include <string>
#include <vector>
#include "moe_dispatch_ccu_api.h"
#include "ccu_kernel_arg.h"
#include "ccu_kernel_signature.h"
#include "ccu_task_arg_v1.h"
#include "hccl_types.h"

namespace hcomm {

// 注册期参数 (attr)：与 op_def 注册顺序一一对应；同一通信域内 immutable，绑 KernelSignature。
struct CcuKernelArgMoeDistributeDispatchCcuSetup : public CcuKernelArg {
    // op_def attrs
    std::string groupEp;             // REQUIRED, 库上 HCCL 通信域名；v3 hcomm 路径不用，签名级占位
    uint32_t epWorldSize;            // REQUIRED
    uint32_t epRankId;               // REQUIRED
    uint32_t moeExpertNum;           // REQUIRED
    uint32_t expertShardType;        // OPT, 默认 0；简化版固定 0
    uint32_t sharedExpertNum;        // OPT, 默认 1（与 op_def 一致）
    uint32_t sharedExpertRankNum;    // OPT, 默认 0；简化版固定 0
    uint32_t quantMode;              // OPT, 默认 0；简化版固定 0（非量化）
    uint32_t globalBs;               // OPT, 默认 0 → 算出 bs*epWorldSize
    uint32_t commType;               // OPT, 默认 0（与 op_def 一致）
    std::string commAlg;             // OPT, 默认 ""（与 op_def 一致；v3 之前误写成 uint32）

    // shape 与派生量（kernel 内部用，不来自 op_def attr）
    uint32_t bs;
    uint32_t topK;
    uint32_t hiddenSize;
    uint64_t tokenMsgLen;            // v6: Align256(hiddenSize) * sizeof(YOutType)（非量化）；
                                     //     来源：aclnnMoeDistributeDispatchSetup.md §约束 5
    uint32_t moeExpertPerRankNum;    // moeExpertNum / (epWorldSize - sharedExpertRankNum)，
                                     //     即 op_def 的 localExpertNum

    CcuKernelArgMoeDistributeDispatchCcuSetup(
        const std::string &groupEp_,
        uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
        uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
        uint32_t expertShardType_ = 0, uint32_t sharedExpertNum_ = 1,
        uint32_t sharedExpertRankNum_ = 0, uint32_t quantMode_ = 0,
        uint32_t globalBs_ = 0, uint32_t commType_ = 0,
        const std::string &commAlg_ = "");

    CcuKernelSignature GetKernelSignature() const override;
};

// 每次 launch 动态参数：字段顺序严格对齐 op_def 的 Input/Output。
//   Inputs:  x, expert_ids, scales, x_active_mask
//   Outputs: y, expand_idx, comm_cmd_info
//   + commWindow / routingTable: v13 工程内部 scratch（库上 op_def 无此字段）
//
// v13 ★ 关键架构纠正（op_def yAddr 字段语义恢复库上）：
//   v12 之前把 op_def yAddr 当作"对端接收 window"用——这跟库上 yAddr（src 端发送缓冲）
//   字段语义颠倒。v13 引入独立的 commWindowAddr 字段作为接收 window，op_def yAddr
//   保留 caller 占位 alloc 但 device 端不读不写。
//   详见 OPEN_QUESTIONS P0-0a "差异 1" 修复。
struct CcuTaskArgMoeDistributeDispatchCcuSetup : public CcuTaskArg {
    // ----- Inputs (op_def) -----
    uint64_t xAddr;                  // [Bs, H]    REQUIRED (BF16/FP16/...) —— v15 ★ device 端不直接读
                                      //   （host 端把 x 重排拼上 metadata 后 device 端读 xWithMetaAddr）
    uint64_t expertIdsAddr;          // [Bs, topK] REQUIRED (INT32)；host 用它算 routingTable
    uint64_t scalesAddr;             // OPTIONAL (FLOAT)；简化版传 0
    uint64_t xActiveMaskAddr;        // OPTIONAL (BOOL)；简化版传 0
    // ----- Outputs (op_def) -----
    uint64_t yAddr;                  // v13 ★ caller 按 op_def 公式 alloc 占位，device 端不读不写
                                      //   （库上 yAddr 语义为 src 端发送缓冲，v13 CCU 0.5RTT 路径不需要
                                      //    发送侧重排——直接 per-token RemoteWrite 到对端 commWindow）
    uint64_t expandIdxAddr;          // [Bs*topK] INT32
    uint64_t commCmdInfoAddr;        // INT32；前 epWorldSize 槽是正式槽，后 epWorldSize 槽是 scratch
    // ----- Internal (v13/v15 工程内部 scratch，库上 op_def 无) -----
    uint64_t commWindowAddr;         // v13 ★ 对端接收 window：对端 setup 通过 0.5RTT WriteNbWithArriveNotify
                                      //   写到本卡 commWindow[本卡视角的 src rank 段]；teardown 读取整理
                                      //   v15 ★ shape: (rankSize * bs * topK, tokenWithMetaLen)
                                      //          每槽 = tokenMsgLen data + 32B metadata 三元组
    uint64_t routingTableAddr;       // [Bs*topK] INT32, host 预算 = expertIds / moeExpertPerRankNum
    uint64_t xWithMetaAddr;          // v15 ★ host 预拼装的 (Bs*K, tokenWithMetaLen) buffer：
                                      //   每行 idx=i*K+j 内容 = x[i] data + (selfRank, i, j) 三元组 + pad to 32B
                                      //   段 7 阶段 B per-token WriteNb 直接从这个 buffer 读
                                      //   一次性 RDMA 把 data + metadata 整体送过去

    CcuTaskArgMoeDistributeDispatchCcuSetup(
        uint64_t xAddr_, uint64_t expertIdsAddr_,
        uint64_t scalesAddr_, uint64_t xActiveMaskAddr_,
        uint64_t yAddr_, uint64_t expandIdxAddr_, uint64_t commCmdInfoAddr_,
        uint64_t commWindowAddr_, uint64_t routingTableAddr_,
        uint64_t xWithMetaAddr_);
};

// device 算法
class CcuKernelMoeDistributeDispatchCcuSetup : public Mc2CcuKernelBase {
public:
    explicit CcuKernelMoeDistributeDispatchCcuSetup(const CcuKernelArg &arg);

protected:
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override;

private:
    uint32_t epWorldSize_;
    uint32_t epRankId_;
    uint32_t moeExpertNum_;            // v8 ★ DoMainLoop 阶段 A 按 expertId 展开 CCU_IF（per-expert expand_idx 计数）
    uint32_t moeExpertPerRankNum_;
    uint32_t bs_;
    uint32_t topK_;
    uint64_t tokenMsgLen_;

    CcuRep::Variable xAddrVar_;
    CcuRep::Variable expertIdsAddrVar_;          // v8 ★ DoMainLoop 阶段 A 读 expertId
    CcuRep::Variable commWindowAddrVar_;         // v13 ★ 替代 v12 之前的 yAddrVar_（op_def yAddr 不再用作接收 window）
    CcuRep::Variable expandIdxAddrVar_;
    CcuRep::Variable commCmdInfoAddrVar_;
    CcuRep::Variable routingTableAddrVar_;
    CcuRep::Variable xWithMetaAddrVar_;          // v15 ★ host 预拼装的 (Bs*K, tokenWithMetaLen) buffer 起址
    CcuRep::Variable tokenMsgLenVar_;
    CcuRep::Variable tokenWithMetaLenVar_;       // v15 ★ = tokenMsgLen + 32（段 4/5/7 用）
    std::vector<CcuRep::Variable> remoteCommWindowAddrs_;   // v13 ★ PreSync 后拿到的各对端 commWindow 起址
    std::vector<CcuRep::Variable> expertOccCnts_; // v8 ★ per-expert 累计计数（对齐库上 expand_idx "给同一专家发送的 token 个数"语义）
    std::vector<CcuRep::Variable> peerOccBytes_;
    CcuRep::LocalAddr localSrc_;
    CcuRep::LocalAddr localDst_;
    CcuRep::RemoteAddr remoteDst_;
    CcuRep::CompletedEvent event_;
};

} // namespace hcomm

#endif // MOE_DISTRIBUTE_DISPATCH_CCU_SETUP_H
