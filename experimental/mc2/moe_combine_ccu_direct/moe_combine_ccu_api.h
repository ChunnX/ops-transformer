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
 * \file moe_combine_ccu_api.h
 * \brief A6 CCU combine_setup 的 0.5RTT 资源 tag + 占位基类（v21：自包含，与 dispatch 用例完全解耦）。
 *
 * v21 变更：不再 #include "moe_dispatch_ccu_api.h"——combine 工程独立编译，不依赖 dispatch 目录任何文件。
 *   本头自带 Mc2CcuKernelBase + CHK_RET + 4 个 0.5RTT API 声明，定义与 dispatch 完全一致（同一
 *   hcomm::Mc2CcuKernelBase 类，链接到 A6 hcomm 同一实现，ODR 安全：combine 编译单元只 include 本头、
 *   dispatch 编译单元只 include moe_dispatch_ccu_api.h，两者不会同时出现在一个 TU）。唯一区别是 tag。
 *
 * 4 个 0.5RTT API 语义（按 A6_CCU_0.5RTT 方案文档，实现链接期由 A6 hcomm 提供）：
 *   RecordExpectCount         发送方把"我会发多少字节"写入对端 channel 的 wishCntXn（含 sem+mask 完成事件）
 *   WriteNbWithArriveNotify   发送方写数据 + 把本次长度 atomic add 到对端 wtCount
 *   WaitallpeersarriveNotify  接收方等本端 totalCntXn 达到 target
 *   clearallpeersarriveNotify 清零本端 totalCntXn 与所有 wishCntXn
 */

#ifndef MOE_COMBINE_CCU_API_H
#define MOE_COMBINE_CCU_API_H

#include <string>
#include "ccu_kernel.h"

// 工程私有宏（与 dispatch 同款；#ifndef 守护避免与 dispatch api.h 在同一 TU 重复定义——本工程不会同时 include）。
//   语义：调用 expr，若返回非 HCCL_SUCCESS 则原样返回。
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

// 0.5RTT 上层接口占位基类：combine_setup 的 CcuKernel 子类继承它；实现由 A6 hcomm 补全。
//   ★ 定义须与 moe_dispatch_ccu_api.h 的 Mc2CcuKernelBase 逐字一致（同一类，ODR）。
class Mc2CcuKernelBase : public CcuKernel {
public:
    using CcuKernel::CcuKernel;

protected:
    HcclResult RecordExpectCount(const ChannelHandle channel, const CcuRep::Variable &var,
                                  const std::string &tag, CcuRep::CompletedEvent sem, uint16_t mask);

    HcclResult WriteNbWithArriveNotify(const ChannelHandle channel,
                                       const CcuRep::RemoteAddr &rem,
                                       const CcuRep::LocalAddr &loc,
                                       const CcuRep::Variable &len, const std::string &tag);

    HcclResult WaitallpeersarriveNotify(const CcuRep::Variable &target, const std::string &tag);

    HcclResult clearallpeersarriveNotify(const std::string &tag);
};

} // namespace hcomm

// 本算子使用的 0.5RTT 资源组 tag（与 dispatch 不同，独立累加，dispatch/combine 流水互不干扰）。
#define MOE_COMBINE_CCU_TAG  "moe_combine_05rtt"

#endif // MOE_COMBINE_CCU_API_H
