/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  combine_setup_ccu_api.h —— A6 CCU 0.5RTT 数据面 4 接口的上层占位声明（combine 版，独立资源组 tag）。
 *  与 dispatch_setup_ccu_api.h 逐字同款（同一 hcomm::Mc2CcuKernelBase 声明，SE 链接期补全实现），
 *  仅资源组 tag 换成 combine 专属，与 dispatch 互不干扰（CCU 至多 4 组 wishCntXn/totalCntXn）。
 *
 *  ★【铁律】类名 hcomm::Mc2CcuKernelBase + 4 方法签名（大小写：Waitallpeers 大 W / clearallpeers 小 c）
 *    必须与参考逐字一致，否则 mangled name 对不上 SE 实现 → 链接 undefined。
 *  ★ combine_setup 与 dispatch_setup 是【独立 standalone 二进制】（各自 min_driver + cmake），
 *    故各自声明 Mc2CcuKernelBase 无 ODR 冲突。
 */
#ifndef COMBINE_SETUP_CCU_API_H
#define COMBINE_SETUP_CCU_API_H

#include <string>
#include "ccu_kernel.h"

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

// combine 0.5RTT 资源组 tag（≠ dispatch，独立组；= 驱动 HcclChannelDesc.ccuUbAttr.resGroupTag，一字一致）。
#define COMBINE_SETUP_CCU_TAG  "moe_combine_05rtt"

#endif // COMBINE_SETUP_CCU_API_H
