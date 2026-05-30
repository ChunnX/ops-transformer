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

    // ============ v19 ★ 标量写 GM（动态地址 + token）占位声明 ============
    //
    // 【为何需要】CCU 算法要把 device 端算出的标量（Variable，如 expand_idx 计数、tokenCount、
    //   ready 状态位）写回 GM 的"运行期动态地址"（base 来自 Load，非 host 编译期立即数）。
    //
    // 【公开 API 缺口（亲核实 ccu_kernel.h CANN 9.1.0）】CcuKernel 公开的标量写只有一个：
    //       void StoreVariable(const CcuRep::Variable &var, uint64_t addr);
    //   它的 addr 是 **host 编译期立即数 uint64_t**，且 **无 token 形参**。
    //   而 Variable / Address 都不能隐式转 uint64_t（实测编译 no known conversion），
    //   故无法把"动态地址 Variable"喂给它 → 公开 API 写不了动态地址标量。
    //
    // 【底层硬件其实支持】ccu_microcode_v1.h:901
    //       void StoreXToMem(CcuInstr*, uint16_t dst, uint16_t dstToken, uint16_t src, uint16_t len, ...);
    //   dst / dstToken 都是寄存器号 → 微码层完全支持 (动态地址寄存器 + token)，只是上层未封装。
    //
    // 【本工程处理】与上面 4 个 0.5RTT API 同样的"占位 extern、链接期 SE 补全"模式：
    //   声明一个语义精确的 helper，让算法表达正确意图、-fsyntax-only 通过；定义由 SE 在 hcomm
    //   仓补（直接转发到 CcuV2::StoreXToMem，把 addr/token 落成寄存器号）。
    //   ⚠️ 这是 OPEN_QUESTION：上板前必须确认 hcomm 已补此 API（或改走 AIV teardown 产出这些标量）。
    //
    // 语义：把 val 写到 GM 地址 addr（Variable，运行期动态）处的 4 字节，访存用 token 鉴权。
    HcclResult StoreVariableAt(const CcuRep::Variable &val, const CcuRep::Variable &addr,
                               const CcuRep::Variable &token);
};

} // namespace hcomm

// 本算子使用的 0.5RTT 资源组 tag（与 ChannelDesc.channelGroupTag 对应，CCU 上至多 4 组）。
#define MOE_DISPATCH_CCU_TAG  "moe_dispatch_05rtt"

#endif // MOE_DISPATCH_CCU_API_H
