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
 * \file moe_dispatch_ccu_api.h
 * \brief A6 0.5RTT 特性需要的 ccuKernel 上层接口占位声明。
 *
 * 这 4 个接口在当前 hcomm 仓 (ccu_kernel.h) 里尚未实现，驱动层 opcode
 *   CCU_U_OP_GET_HIGH_PERF_XN   (102)  Get high-performance XN register range
 *   CCU_U_OP_SET_TIF_SPLIT_SIZE (127)  Set count unit for 0.5RTT
 *   CCU_U_OP_SET_XN_TOTAL_CNT   (128)  Set compare register for 0.5RTT
 *   CCU_U_OP_SET_TRANM_NTF_HVAL (129)  Set high 32bits of notify value
 * 已就绪。本算子工程把它们封装为 Mc2CcuKernelBase 的 protected 成员声明，dispatch
 * setup/teardown 的 CcuKernel 子类直接调用；链接期由 SE 在 hcomm 仓补全实现。
 *
 * 4 个接口语义（按 A6_CCU_0.5RTT 方案文档）：
 *   RecordExpectCount        发送方把"我会发多少字节"写入对端 channel 的 wishCntXn
 *   WriteNbWithArriveNotify  发送方写数据 + 把本次长度 atomic add 到对端 wtCount
 *   WaitAllPeersArriveNotify 接收方等本端 totalCntXn 达到 target
 *   ClearAllPeersArriveNotify 清零本端 totalCntXn 与所有 wishCntXn
 *
 * 配套 ChannelDesc.channelGroupTag 字段（方案文档增量定义、仓里待补）：
 *   channel 通过 tag 关联到 CCU 上一组 wishCntXn/totalCntXn 资源（CCU 最多 4 组），
 *   一个算子用一组 tag。本算子 tag 见下方宏。
 */

#ifndef MOE_DISPATCH_CCU_API_H
#define MOE_DISPATCH_CCU_API_H

#include <string>
#include "ccu_kernel.h"

// v5 工程私有宏：hcomm 公开头未暴露，自定义最小实现。
//   语义：调用 expr，若返回非 HCCL_SUCCESS 则原样返回。
//   不与 hcomm 内部 CHK_RET 冲突（不同编译单元，遵循 ODR）。
#ifndef CHK_RET
#define CHK_RET(expr)                              \
    do {                                           \
        HcclResult __ccuRet = (expr);              \
        if (__ccuRet != HCCL_SUCCESS) {            \
            return __ccuRet;                       \
        }                                          \
    } while (0)
#endif

namespace hcomm {

// 0.5RTT 上层接口的占位基类：dispatch setup/teardown 的 CcuKernel 子类继承它，
// 即可在 Algorithm 中按成员函数语义调用 4 个 0.5RTT API；实现待 SE 补全。
class Mc2CcuKernelBase : public CcuKernel {
public:
    using CcuKernel::CcuKernel;

protected:
    // 将 var 中的值写到 channel 中以 tag 标记的对端 countXn 作为 wishCount。
    HcclResult RecordExpectCount(const ChannelHandle channel,
                                  const CcuRep::Variable &var, std::string tag);

    // 从 loc 写长为 len 的数据到 rem，并把本次写的数据量
    // 累加到 channel 对端 tag 标记的 countXn 的 wtCount。
    HcclResult WriteNbWithArriveNotify(const ChannelHandle channel,
                                       const CcuRep::RemoteAddr &rem,
                                       const CcuRep::LocalAddr &loc,
                                       const CcuRep::Variable &len, std::string tag);

    // 等本端以 tag 标记的 totalCntXn 达到 target 时解除等待。
    HcclResult WaitAllPeersArriveNotify(const CcuRep::Variable &target, std::string tag);

    // 清除本端 tag 标记的 totalCntXn 与所有 wishCntXn。
    HcclResult ClearAllPeersArriveNotify(std::string tag);

    // ============ 标量写 GM（Variable 动态地址版）============
    //
    // 【为何需要】CCU 算法要把 device 标量（Variable，如 expand_idx 计数）写回 GM 的"运行期动态地址"
    //   （base 来自 Load，非 host 编译期立即数）。
    //
    // 【真相：hcomm 早有此能力，只是 pkg_inc CcuKernel 漏暴露】
    //   legacy CcuContext 有两个 StoreVariable 重载（ccu_ctx.h:148/150，实现 ccu_context.cpp:285/295）：
    //       void StoreVariable(const Variable &var, uint64_t addr);        // 立即数地址
    //       void StoreVariable(const Variable &var, const Variable &src);  // ★ Variable 动态地址（无 token）
    //   mc2 通算融合的 mc2_context.cpp 正用第二个写本卡 HBM 信号（StoreVariable(sig, hbmSigAddr)，
    //   hbmSigAddr 是 Variable、无 token）——证明"写本卡 HBM 标量"不需要调用方传 token。
    //   而 pkg_inc 新体系 CcuKernel（我们继承的）只暴露了立即数版（ccu_kernel.h:172），Variable 地址版
    //   漏暴露（对比 Load 有 Variable 地址版 :169，Store 没有——不对称遗漏）。
    //
    // 【本工程处理】占位声明 Variable 地址版（语义=legacy CcuContext::StoreVariable(Variable,Variable)），
    //   用标准名表达正确意图、-fsyntax-only 通过；待 SE 把该重载补进 pkg_inc CcuKernel（照搬 legacy 实现、
    //   非新造）后即用基类版、删本占位。expand_idx 写本卡 HBM 输出 buffer，无需 token。
    using CcuKernel::StoreVariable;   // 引入基类立即数版，避免 name hiding
    void StoreVariable(const CcuRep::Variable &val, const CcuRep::Variable &src);
};

} // namespace hcomm

// 本算子使用的 0.5RTT 资源组 tag（与 ChannelDesc.channelGroupTag 对应，CCU 上至多 4 组）。
#define MOE_DISPATCH_CCU_TAG  "moe_dispatch_05rtt"

#endif // MOE_DISPATCH_CCU_API_H
