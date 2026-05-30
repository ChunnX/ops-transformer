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

// ============ 注册期参数（ATTR）============
//   彻底精简版：只保留 device 端实际使用的 attr。
//   已删 op_def 占位 attr（device 不读）：groupEp / expertShardType / sharedExpertNum /
//     sharedExpertRankNum / globalBs / commType / commAlg / moeExpertPerRankNum（派生量，dispatch 不用）。
struct CcuKernelArgMoeDistributeDispatchCcuSetup : public CcuKernelArg {
    // ----- ATTR -----
    uint32_t epWorldSize;            // EP 通信域大小
    uint32_t epRankId;               // 本卡 EP rank id
    uint32_t moeExpertNum;           // MoE 总 expert 数（段 7 阶段 A per-expert 计数循环上界）
    uint32_t quantMode;              // 0=BF16 / 1=MXFP8（切 tokenMsgLen 公式）
    // ----- shape 派生量（非 attr，kernel 内部）-----
    uint32_t bs;
    uint32_t topK;
    uint32_t hiddenSize;
    uint64_t tokenMsgLen;            // quantMode=0: Align256(H)*2；quantMode=1: Align32(H+H/32)

    CcuKernelArgMoeDistributeDispatchCcuSetup(
        uint32_t epWorldSize_, uint32_t epRankId_, uint32_t moeExpertNum_,
        uint32_t bs_, uint32_t topK_, uint32_t hiddenSize_,
        uint32_t quantMode_ = 0);

    CcuKernelSignature GetKernelSignature() const override;
};

// ============ 每次 launch 动态参数（INPUT / OUTPUT / INTERNAL）============
//   字段分组清晰标注；构造函数顺序 = 字段顺序。
//   已删 device 不读的 op_def 占位：xAddr（v15 后改读 xWithMeta）/ scalesAddr / xActiveMaskAddr
//     （GeneArgs 从未传）/ yAddr（v13 后纯占位 alloc）。
struct CcuTaskArgMoeDistributeDispatchCcuSetup : public CcuTaskArg {
    // ----- INPUT -----
    uint64_t expertIdsAddr;          // [Bs, topK] INT32：段 7 阶段 A 读 expertId
    // ----- OUTPUT -----
    uint64_t expandIdxAddr;          // [Bs*topK] INT32：段 7 阶段 A 写
    uint64_t commCmdInfoAddr;        // INT32：段 8 写（前 epWorldSize 槽正式 + 后段 scratch）
    // ----- INTERNAL（工程内部 scratch，库上 op_def 无）-----
    uint64_t commWindowAddr;         // 对端接收 window (rankSize*bs*topK, tokenWithMetaLen)：
                                      //   每槽 = tokenMsgLen data + 32B metadata 三元组
    uint64_t routingTableAddr;       // [Bs*topK] INT32：host 预算 = expertId / moeExpertPerRankNum
    uint64_t xWithMetaAddr;          // host 预拼装 (Bs*K, tokenWithMetaLen)：
                                      //   每行 = x[i] data + (selfRank,i,j,dstByteOff) metadata + pad

    CcuTaskArgMoeDistributeDispatchCcuSetup(
        uint64_t expertIdsAddr_, uint64_t expandIdxAddr_, uint64_t commCmdInfoAddr_,
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
    uint32_t bs_;
    uint32_t topK_;
    uint64_t tokenMsgLen_;

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
