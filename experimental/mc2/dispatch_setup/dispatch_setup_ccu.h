/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  dispatch_setup_ccu.h
 *  --------------------
 *  CCU dispatch_setup：用 a6 已验证的 0.5RTT 跨卡写（WriteNbWithArriveNotify）把"就绪值 1"
 *  写进【对端】卡的状态区——卡0 写卡1、卡1 写卡0；本卡状态区由对端写、本卡 teardown(AIV) 轮询 slot0。
 *
 *  为什么从"本地 LocalCopyNb 自写"改成"对端跨卡写"：
 *    a6 上 6 参 LocalCopyNb 是接口团队新开发、未验证；上板诊断坐实它静默不写
 *    （CCU 跑完无 fault、但 statusGm 全 0）。改用 a6 已验证功能正常的 WriteNbWithArriveNotify。
 *
 *  协议（裁剪自 a6 已验证参考 moe_distribute_dispatch_ccu_setup.cpp 的 0.5RTT 数据面）：
 *    clear → PreSync(NotifyRecord/Wait 换对端 status 地址+token) → RecordExpectCount →
 *    WriteNbWithArriveNotify(oneBuf→对端 status slot0) → WaitallpeersarriveNotify(R-1)。
 *    payload 极简：参考的逐 token 数据面(routing/sendBytes/DoMainLoop 循环)全删，只剩一次定长写。
 *
 *  基类：Mc2CcuKernelBase（提供 4 个数据面 API，见 dispatch_setup_ccu_api.h）；
 *    channels_ / NotifyRecord / NotifyWait / CreateVariable(channel,XN) / CreateRemoteAddr
 *    来自 CcuKernel 基类（CANN pkg ccu_kernel.h:109/113/130/132/134 坐实）。
 *
 *  状态区契约（与 dispatch_teardown AIV 唯一耦合点，字节级一致）：
 *    statusGm：int32 GM，slot0=1 表示就绪；R 个 int32 槽（32B 对齐，teardown 一次搬 32B 读 slot0）。
 *
 *  a5/a6 隔离：跨卡 0.5RTT 数据面 + channel-XN PreSync 为 a6-only（SE 链接期补全）；
 *    .h（类/成员声明）在 a5 编得过（CANN 头已有 RemoteAddr/CreateRemoteAddr/channel-XN CreateVariable）；
 *    .cpp 的 Algorithm 体整体 #ifdef DISPATCH_CCU_A6、a5 桩 return（不引用 a6-only 数据面符号 → a5 链接通过）。
 */
#ifndef DISPATCH_SETUP_CCU_H
#define DISPATCH_SETUP_CCU_H

#include <vector>
#include <cstdint>
#include "dispatch_setup_ccu_api.h"  // Mc2CcuKernelBase + 4 数据面 API + CHK_RET + DISPATCH_SETUP_CCU_TAG（含 ccu_kernel.h）

namespace mc2 {

using namespace hcomm;

// 状态区元素类型与"就绪"值（与 teardown AIV 读端严格一致）。
constexpr uint32_t DISPATCH_STATUS_DTYPE_BYTES = sizeof(int32_t);  // 4
constexpr int32_t  DISPATCH_READY_VALUE = 1;

// -------- KernelArg：构造期固定标量（epWorldSize 决定对端数 / 状态区槽数；epRankId 本卡号）--------
class CcuKernelArgDispatchSetup : public CcuKernelArg {
public:
    CcuKernelArgDispatchSetup(uint32_t epWorldSize, uint32_t epRankId)
        : epWorldSize(epWorldSize), epRankId(epRankId) {}

    CcuKernelSignature GetKernelSignature() const override {
        return CcuKernelSignature();   // 最小版默认构造（[T11] 若 a6 需 name/attrs 唯一键再补）
    }

    uint32_t epWorldSize{1};
    uint32_t epRankId{0};
};

// -------- TaskArg：每次 Launch 传入的 device GM 地址 + token（GeneArgs/Load 顺序须三处一致）--------
//   statusAddr：本卡状态区——被【对端】WriteNb 写、本卡 teardown 读；PreSync 把其 addr+token 交换给对端。
//   oneBufAddr：host 预填 slot0=1 的源 buffer——本卡作为发送方的 localSrc。
class CcuTaskArgDispatchSetup : public CcuTaskArg {
public:
    CcuTaskArgDispatchSetup(uint64_t statusAddr, uint64_t statusToken,
                            uint64_t oneBufAddr, uint64_t oneBufToken)
        : statusAddr(statusAddr), statusToken(statusToken),
          oneBufAddr(oneBufAddr), oneBufToken(oneBufToken) {}

    uint64_t statusAddr{0};
    uint64_t statusToken{0};
    uint64_t oneBufAddr{0};
    uint64_t oneBufToken{0};
};

// -------- CCU kernel：Algorithm() 内建微码图，跨卡把 1 写到【对端】状态区 --------
class CcuKernelDispatchSetup : public Mc2CcuKernelBase {
public:
    explicit CcuKernelDispatchSetup(const CcuKernelArg &arg) : Mc2CcuKernelBase(arg) {
        const auto *kernelArg = dynamic_cast<const CcuKernelArgDispatchSetup *>(&arg);
        epWorldSize_ = kernelArg->epWorldSize;
        epRankId_    = kernelArg->epRankId;
    }
    ~CcuKernelDispatchSetup() override {}

    HcclResult Algorithm() override;

    // host→device 参数序列：device 侧 Load 顺序必须与此【逐字节一致】，否则读到错位地址/token。
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override {
        const auto *taskArg = dynamic_cast<const CcuTaskArgDispatchSetup *>(&arg);
        return {taskArg->statusAddr, taskArg->statusToken,
                taskArg->oneBufAddr, taskArg->oneBufToken};
    }

private:
    uint32_t epWorldSize_{1};
    uint32_t epRankId_{0};

    // ---- 段 1 在 Algorithm() 内创建的 CCU 资源 ----
    CcuRep::Variable statusAddrVar_;    // 本卡 status 起址（Load；PreSync 交换给对端）
    CcuRep::Variable statusTokenVar_;   // 本卡 status token（Load；PreSync 交换给对端）
    CcuRep::Variable oneBufAddrVar_;    // 本卡 oneBuf 源起址（Load；段 7 localSrc.addr）
    CcuRep::Variable oneBufTokenVar_;   // 本卡 oneBuf 源 token（Load；段 7 localSrc.token）
    std::vector<CcuRep::Variable> remoteStatusAddrs_;   // PreSync 后各对端 status 起址（段 7 remoteDst.addr）
    std::vector<CcuRep::Variable> remoteStatusTokens_;  // PreSync 后各对端 status token（段 7 remoteDst.token）
    CcuRep::LocalAddr  localSrc_;       // 段 7 数据源（本卡 oneBuf）
    CcuRep::RemoteAddr remoteDst_;      // 段 7 跨卡写目标（对端 status）
    CcuRep::CompletedEvent event_;      // 段 6 RecordExpectCount 完成事件
};

}  // namespace mc2

#endif  // DISPATCH_SETUP_CCU_H
