/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  dispatch_setup_ccu_api.h —— A6 CCU 0.5RTT 数据面 4 接口的 ccuKernel 上层占位声明。
 *
 *  逐字复刻自 a6 已验证参考
 *    /home/developer/hcomm-test/src/moe_distribute_dispatch_ccu_setup_test/
 *        ccu_kernel_impl/moe_dispatch_ccu_api.h
 *  这 4 个接口在 hcomm 公共头 (ccu_kernel.h) 里【未声明】，由本工程封装为 Mc2CcuKernelBase
 *  的 protected 成员；链接期由 SE 在 hcomm 仓补全实现（a6 driver opcode 102/127/128/129 已就绪）。
 *  ★【铁律】类名 hcomm::Mc2CcuKernelBase 与 4 个方法签名【必须与参考逐字一致】，否则 mangled name
 *    对不上 SE 在 libhcomm 的实现 → 链接 undefined。改它前先 diff 参考。
 *
 *  4 接口语义（A6_CCU_0.5RTT 方案文档）：
 *    RecordExpectCount         发送方把"我会发多少字节"写入对端 channel 的 wishCntXn（SE 对齐：+sem+mask）
 *    WriteNbWithArriveNotify   发送方写数据 + 把本次写的字节数 atomic add 到对端 wtCount
 *    WaitallpeersarriveNotify  接收方等本端 totalCntXn 达到 target（SE 官方名，注意大小写）
 *    clearallpeersarriveNotify 清零本端 totalCntXn 与所有 wishCntXn（SE 官方名，首字母小写 c）
 */
#ifndef DISPATCH_SETUP_CCU_API_H
#define DISPATCH_SETUP_CCU_API_H

#include <string>
#include "ccu_kernel.h"   // 解析到 CANN pkg_inc/hcomm/ccu/ccu_kernel.h（CcuKernel / CcuRep::* / ChannelHandle）

// 工程私有 CHK_RET：调用 expr，非 HCCL_SUCCESS 即原样返回（#ifndef 守护，与 dispatch_setup_ccu.h
//   / hcomm 内部宏不冲突，ODR 各编译单元独立）。
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

// 0.5RTT 上层接口占位基类：继承 CcuKernel，仅【声明】4 个数据面 API（impl 待 SE 链接期补全）。
// ★ 与参考 moe_dispatch_ccu_api.h 的 Mc2CcuKernelBase 逐字一致（类名 / 签名 / protected），保证链得到 SE 实现。
class Mc2CcuKernelBase : public CcuKernel {
public:
    using CcuKernel::CcuKernel;

protected:
    // 把 var 的值作为 wishCount 写到 channel 以 tag 标记的对端 countXn（sem 的 mask 位标记本地完成）。
    HcclResult RecordExpectCount(const ChannelHandle channel, const CcuRep::Variable &var,
                                  const std::string &tag, CcuRep::CompletedEvent sem, uint16_t mask);

    // 从 loc 写长为 len 的数据到 rem，并把本次字节数累加到 channel 对端 tag 标记的 countXn 的 wtCount。
    HcclResult WriteNbWithArriveNotify(const ChannelHandle channel,
                                       const CcuRep::RemoteAddr &rem,
                                       const CcuRep::LocalAddr &loc,
                                       const CcuRep::Variable &len, const std::string &tag);

    // 等本端以 tag 标记的 totalCntXn 达到 target 时解除等待。
    HcclResult WaitallpeersarriveNotify(const CcuRep::Variable &target, const std::string &tag);

    // 清除本端 tag 标记的 totalCntXn 与所有 wishCntXn。
    HcclResult clearallpeersarriveNotify(const std::string &tag);
};

} // namespace hcomm

// 本算子 0.5RTT 资源组 tag（= 驱动 HcclChannelDesc.ccuUbAttr.resGroupTag，必须一字一致）。
//   ★ 复用参考已验证的 key 字符串——它只是 CCU 资源组键、非语义名；与 dispatch_min_driver.cpp 的
//     resGroupTag 保持一致即可（CCU 最多 4 组，本算子用一组）。零偏离已验证路径。
#define DISPATCH_SETUP_CCU_TAG  "moe_dispatch_05rtt"

#endif // DISPATCH_SETUP_CCU_API_H
