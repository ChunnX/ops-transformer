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
 * \file dispatch_setup_ccu.cpp
 * \brief dispatch-ccu CCU setup kernel 实现。GeneArgs 已实现；Algorithm 为 P0 步骤 4 骨架 stub。
 */
#include "dispatch_setup_ccu.h"

DispatchSetupCcu::DispatchSetupCcu(const hcomm::CcuKernelArg& arg)
    : hcomm::CcuKernel(arg)
{
    // 注册期：从 KernelArg 静态配置中取出 rank 拓扑
    const auto& a = static_cast<const DispatchSetupKernelArg&>(arg);
    worldSize_ = a.worldSize_;
    myRank_    = a.myRank_;
    param_     = a.param_;
}

// GeneArgs：动态地址打包成 uint64 数组。次序 = { x, expertIds, remoteVaTable, tokenNum, hidden }，
// Algorithm() 内 Load(...) 必须按同一次序加载，否则 TaskArg 在 xn 错位（设计文档失败模式 FM2）。
std::vector<uint64_t> DispatchSetupCcu::GeneArgs(const hcomm::CcuTaskArg& arg)
{
    const auto& a = static_cast<const DispatchSetupTaskArg&>(arg);
    return { a.xAddr_, a.expertIdsAddr_, a.remoteVaTableAddr_, a.tokenNum_, a.hidden_ };
}

// Algorithm：CCU 通信算法主体。
// ===== TODO(P0 步骤 4)：按设计文档 §8 + hccl/src/ops/scatter/template/ccu/ 的
//       CcuKernelScatterMesh1D 实现 token 散发 =====
//   Phase A InitResource：for r in worldSize → CreateRemoteAddr() / CreateLocalAddr() / CreateVariable()
//   Phase B LoadArgs    ：Load(xAddr) / Load(expertIdsAddr) / Load(remoteVaTable) / Load(tokenNum) / Load(hidden)
//                         —— 次序须与上面 GeneArgs 输出严格一致
//   Phase C 散发        ：Loop("send_tokens") → 按 expertIds 算目的 rank →
//                         WriteNb(channels_[dstRank], dstAddr, srcAddr, lenVar, ev) → WaitEvent(ev)
//   Phase D 通知        ：for peer r != myRank_ → NotifyRecord(channels_[r], myRank_, 1)
// 当前为骨架占位：直接返回成功，使 host 编排链路（建链 / 注册 / 双流 / launch）可先行联调。
HcclResult DispatchSetupCcu::Algorithm()
{
    return HCCL_SUCCESS;
}

// GetKernelSignature：用于 kernel 去重 / 缓存。
// ===== TODO(P0 步骤 4)：参考 hccl ccu_kernel_utils.h 的 GenerateCcuKernelSignature 填充签名字段。=====
hcomm::CcuKernelSignature DispatchSetupKernelArg::GetKernelSignature() const
{
    hcomm::CcuKernelSignature signature;
    return signature;
}
