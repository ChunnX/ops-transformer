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
 *
 * ★ 数据宽度分歧（CCU 实现 vs 库上 op_def）：上方契约标注的 expert_ids / expand_idx 库上为 INT32，
 *   但本 CCU 实现里它们被 device 端 LoadVariable / StoreVariable 读写——CCU Variable 即 8 字节 XN 寄存器
 *   （CCU_RESOURCE_XN_PER_SIZE=8），Load/Store 一次必搬 8 字节，故这些 GM 标量（含 routingTable 与每行
 *   metadata 4 字段）一律按 INT64 排布、按 8 步进，否则 8 字节读会把相邻字段卷入高 32 位污染比较。
 *   driver 据此用 int64 buffer；真接库上 op（INT32）时需 host 侧做一次 int32→int64 转置。
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
    uint64_t expertIdsAddr;          // [Bs, topK] INT64：op 输入（v20.6 后 CCU 不消费——routing 改读 host 预算的 routingTable）
    // ----- OUTPUT -----
    //   ★ v20.6：expandIdx 不再由本 CCU 写——它对 xOut==x 闭环不参与（仅服务 expert 分组），
    //     是 expertIds 的确定性函数，改由 host driver 预算填充（见 driver）。故 TaskArg 不再携带其地址/token。
    uint64_t commCmdInfoAddr;        // INT32：本卡命令区（ready[0..R) + v20.6 新增 per-rank 计数区，见 .cpp 段 10/teardown）
    // ----- INTERNAL（工程内部 scratch，库上 op_def 无）-----
    uint64_t commWindowAddr;         // 对端接收 window (rankSize*bs*topK, tokenWithMetaLen)：
                                      //   每槽 = tokenMsgLen data + 32B metadata（4×int64: srcRank/srcSlot/dstK/dstByteOff）
    uint64_t routingTableAddr;       // [Bs*topK] INT64（★ LoadVariable 读 8 字节）：host 预算 = expertId / moeExpertPerRankNum
    uint64_t xWithMetaAddr;          // host 预拼装 (Bs*K, tokenWithMetaLen)：
                                      //   每行 = x[i] data + (selfRank,i,j,dstByteOff) 4×int64 metadata（32B，无 pad）
    // ----- TOKEN（需求2 ★：host GetTokenInfo(addr,size) 算，device 读写 GM 的寻址凭证）-----
    //   token 范式（亲核实 hccl all_to_all_v / all_reduce / reduce_scatter_v mesh1d）：
    //     - hcomm 所有访存指令底层 = (地址寄存器, token 寄存器) 对（见 ccu_microcode_v1.h
    //       TransMem/StoreXToMem/LoadXFromMem 都带 dstToken/srcToken 形参）。
    //     - LocalAddr 与 RemoteAddr 都有 .token 字段，WriteNb/LocalCopyNb 前两端 .token 必须设。
    //     - token 跟随"buffer 所属卡"：本地 buffer 用本地 token（host GetTokenInfo(addr,size) → Load
    //       进 device Variable）；对端 buffer 用对端 token（CreateVariable(ch,TOKEN_XN,&v) + PreSync
    //       NotifyRecord 交换）。每块独立 buffer 一个独立 token（GetTokenInfo(addr,size) 各算各的）。
    //   据此本算子需要 3 个本地 token + 1 类对端 token（v20.6：删除 expandIdx 那路，CCU 不再写任何标量 GM）：
    //     本地：xWithMeta（段 7 数据源）、commWindow（段 7 自发自落地）、commCmdInfo（段 10 本地置 ready）
    //     对端：commWindow（段 7 跨卡写数据）——PreSync 交换（commCmdInfo 纯本地不交换）
    uint64_t xWithMetaToken;         // ★ 本卡 xWithMeta（数据源）的 token（段 7 localSrc）
    uint64_t commWindowToken;        // 本卡 commWindow 的 token（段 7 自发自 localDst + 本地落地）
    uint64_t commCmdInfoToken;       // 本卡 commCmdInfo 的 token（段 10 ready 写 LocalCopyNb 的 dst token）
    // ----- 收口 ★ ready 写源（替代段 10 原 R 次自造 StoreVariableAt → 1 次 hccl 现役 LocalCopyNb）-----
    //   oneBuf 是本卡 GM buffer（host 端填 epWorldSize 个 int32=1），段 10 整块拷到 commCmdInfo ready 区。
    //   token 跟随 buffer：oneBuf 自己 host GetTokenInfo(addr,size) 算，不复用别块 token。
    uint64_t oneBufAddr;             // 收口 ★ oneBuf GM 起址
    uint64_t oneBufToken;            // 收口 ★ oneBuf token

    CcuTaskArgMoeDistributeDispatchCcuSetup(
        uint64_t expertIdsAddr_, uint64_t commCmdInfoAddr_,
        uint64_t commWindowAddr_, uint64_t routingTableAddr_, uint64_t xWithMetaAddr_,
        uint64_t xWithMetaToken_,
        uint64_t commWindowToken_, uint64_t commCmdInfoToken_,
        uint64_t oneBufAddr_, uint64_t oneBufToken_);
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
    uint32_t bs_;
    uint32_t topK_;
    uint64_t tokenMsgLen_;

    // v20.7 ★ 删 expertIdsAddrVar_：device 用 routingTable 定 dst 卡、不读 expertId（expertIdsAddr 仅留 TaskArg 作 op 输入占位）
    CcuRep::Variable commWindowAddrVar_;         // v13 ★ 替代 v12 之前的 yAddrVar_（op_def yAddr 不再用作接收 window）
    CcuRep::Variable commCmdInfoAddrVar_;
    CcuRep::Variable routingTableAddrVar_;
    CcuRep::Variable xWithMetaAddrVar_;          // v15 ★ host 预拼装的 (Bs*K, tokenWithMetaLen) buffer 起址
    CcuRep::Variable tokenMsgLenVar_;
    CcuRep::Variable tokenWithMetaLenVar_;       // v15 ★ = tokenMsgLen + 32（段 4/5/7 用）
    // ----- v19 ★ token 寄存器（需求2，严格对齐 hccl mesh1d 范式）-----
    //   本地 token：Load 自 SQE（host GetTokenInfo 算）；对端 token：PreSync NotifyRecord 交换
    CcuRep::Variable xWithMetaTokenVar_;         // 本地 xWithMeta 数据源 token（段 7 localSrc.token）
    CcuRep::Variable commWindowTokenVar_;        // 本地 commWindow token（段 7 自发自 localDst.token）
    CcuRep::Variable commCmdInfoTokenVar_;       // 本地 commCmdInfo token（段 10 ready 写 LocalCopyNb dst.token）
    CcuRep::Variable oneBufAddrVar_;             // 收口 ★ oneBuf 地址（段 10 ready 写 LocalCopyNb src.addr）
    CcuRep::Variable oneBufTokenVar_;            // 收口 ★ oneBuf token（段 10 ready 写 LocalCopyNb src.token）
    std::vector<CcuRep::Variable> remoteCommWindowAddrs_;   // v13 ★ PreSync 后各对端 commWindow 起址
    std::vector<CcuRep::Variable> remoteCommWindowTokens_;  // PreSync 后各对端 commWindow token（段 7 remoteDst.token）
    std::vector<CcuRep::Variable> peerOccBytes_;
    CcuRep::LocalAddr localSrc_;
    CcuRep::LocalAddr localDst_;
    CcuRep::RemoteAddr remoteDst_;
    CcuRep::CompletedEvent event_;
    CcuRep::CcuBuf localCopyBuf_;   // ★ A6 新 LocalCopyNb 签名要求的 UB 暂存 buf（见 .cpp InitResource 注释）
};

} // namespace hcomm

#endif // MOE_DISTRIBUTE_DISPATCH_CCU_SETUP_H
