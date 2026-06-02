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
 *   RecordExpectCount         发送方把"我会发多少字节"写入对端 channel 的 wishCntXn（SE 对齐：+sem+mask）
 *   WriteNbWithArriveNotify   发送方写数据 + 把本次长度 atomic add 到对端 wtCount
 *   WaitallpeersarriveNotify  接收方等本端 totalCntXn 达到 target（SE 官方名，注意大小写）
 *   clearallpeersarriveNotify 清零本端 totalCntXn 与所有 wishCntXn（SE 官方名，小写 c）
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
    //   ★ SE 对齐：新增 sem + mask 完成事件（写 wishCnt 是异步 issue，经 sem 的 mask 位标记本地完成）。
    HcclResult RecordExpectCount(const ChannelHandle channel, const CcuRep::Variable &var,
                                  const std::string &tag, CcuRep::CompletedEvent sem, uint16_t mask);

    // 从 loc 写长为 len 的数据到 rem，并把本次写的数据量
    // 累加到 channel 对端 tag 标记的 countXn 的 wtCount。（SE 对齐：tag 取 const 引用）
    HcclResult WriteNbWithArriveNotify(const ChannelHandle channel,
                                       const CcuRep::RemoteAddr &rem,
                                       const CcuRep::LocalAddr &loc,
                                       const CcuRep::Variable &len, const std::string &tag);

    // 等本端以 tag 标记的 totalCntXn 达到 target 时解除等待。
    //   ★ SE 官方名 Waitallpeersarrive（注意大小写：首字母大写 W、中段全小写、尾 Notify）；tag const 引用。
    HcclResult WaitallpeersarriveNotify(const CcuRep::Variable &target, const std::string &tag);

    // 清除本端 tag 标记的 totalCntXn 与所有 wishCntXn。
    //   ★ SE 官方名 clearallpeersarrive（注意：首字母小写 c、中段全小写、尾 Notify）；tag const 引用。
    HcclResult clearallpeersarriveNotify(const std::string &tag);

    // ============ 标量写 GM 动态地址：v20.6 已彻底取消（无 StoreVariableAt 依赖）============
    //   历史：曾用自造 StoreVariableAt(val, addrVar, token) 把 device 标量（expand_idx）写回 GM 动态地址。
    //   SE 对齐确认 hcomm 标准 StoreVariable(Variable val, Variable addr)（2 参无 token，动态地址）不支持。
    //   v20.6 决策（Req5 最优解）：dispatch_setup 不再写任何标量 GM——expand_idx 改由 host driver 预算，
    //   per-rank 计数/total 改由 teardown AIV（普通 GlobalTensor.SetValue，非 CCU 微码）产出。
    //   故本 CCU 接口层【不再需要】任何"写动态 GM 地址"的自造微码，SE 依赖清零。
};

} // namespace hcomm

// 本算子使用的 0.5RTT 资源组 tag（与 ChannelDesc.channelGroupTag 对应，CCU 上至多 4 组）。
#define MOE_DISPATCH_CCU_TAG  "moe_dispatch_05rtt"

#endif // MOE_DISPATCH_CCU_API_H
