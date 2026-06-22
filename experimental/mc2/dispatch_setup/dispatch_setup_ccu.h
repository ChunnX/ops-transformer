/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 *  dispatch_setup_ccu.h
 *  ---------------------
 *  最小 CCU dispatch_setup：用 CCU 语义（LocalCopyNb）往【本卡状态区】写 1。
 *
 *  设计来源：裁剪自 a6 已跑通参考
 *    /home/developer/hcomm-test/src/ccu_kernel_test/ccu_kernel_impl/
 *        moe_distribute_dispatch_ccu_setup.{h,cpp}
 *  只保留其“段 10 ready 写”——把 host 预填 1 的 oneBuf 整块 LocalCopyNb 到本卡
 *  状态区。其余跨卡 0.5RTT 流程（PreSync / RecordExpectCount /
 *  WriteNbWithArriveNotify / Waitallpeersarrive）全部删除：
 *    1) 本任务只需“本卡写 1、本卡 teardown 读 1”，单 die 不跨卡；
 *    2) 纯本地 LocalCopyNb 由基类 CcuKernel 直接提供（见 hcomm
 *       ccu_kernel/ccu_kernel.h:334 LocalCopyNb 重载），不需要 Mc2CcuKernelBase
 *       那 4 个尚待 SE 链接期补全的 0.5RTT 接口，故直接继承 CcuKernel。
 *
 *  状态区契约（与 dispatch_teardown AIV 唯一耦合点，字节级必须一致）：
 *    - statusGm：专用 GM buffer（对标参考 commCmdInfo），int32 元素，值 1 表示就绪；
 *    - 单 die / R=1 时只有 1 个 int32 槽（len = R*sizeof(int32) = 4 字节）；
 *    - LocalCopyNb 是裸字节 DMA，状态字保持 4 字节 int32，【不】受 CCU Variable/
 *      Load/Store 恒 8 字节规则约束（那条规则只管 LoadVariable/StoreVariable）。
 *
 *  a6-TODO（闭源、需手抄校准；本机为 a5 只作参考、不在此编译）：
 *    [T1] LocalCopyNb 在 a6 是 6 参签名 (dst, src, len, CcuBuf*, bufCount, event)，
 *         且 UB 暂存 buf 必须 CreateBlockCcuBuf(1U,&buf) 块分配（a5 头签名不同）。
 *    [T2] WaitEvent：a6 参考用 event_.SetMask(1U)+WaitEvent(event_) 1 参形式；
 *         a5 头是 WaitEvent(event, mask) 2 参。本文件按 a6 参考写。
 *    [T3] CcuKernelSignature 的构造：此处用默认构造，若 a6 需带 name/attrs 唯一化
 *         键，按 a6 头补。
 */

#ifndef DISPATCH_SETUP_CCU_H
#define DISPATCH_SETUP_CCU_H

#include <vector>
#include <cstdint>
#include "ccu_kernel.h"   // 解析到 CANN pkg_inc/hcomm/ccu/ccu_kernel.h（提供 CcuKernel / CcuRep::*）

// 工程私有 CHK_RET：调用 expr，非 HCCL_SUCCESS 即原样返回（与 hcomm 内部宏不冲突，ODR 各编译单元独立）。
#ifndef CHK_RET
#define CHK_RET(expr)                              \
    do {                                           \
        HcclResult __ccuRet = (expr);              \
        if (__ccuRet != HCCL_SUCCESS) {            \
            return __ccuRet;                       \
        }                                          \
    } while (0)
#endif

namespace mc2 {

using namespace hcomm;

// 本算子的状态区元素类型与“就绪”值（与 teardown AIV 读端严格一致）。
constexpr uint32_t DISPATCH_STATUS_DTYPE_BYTES = sizeof(int32_t);  // 4
constexpr int32_t  DISPATCH_READY_VALUE = 1;

// -------- KernelArg：构造期固定的标量属性（编译期/建图期可见）--------
//   单 die 最小版只需 epWorldSize（决定状态区有几个 int32 槽，单 die=1）与 epRankId。
class CcuKernelArgDispatchSetup : public CcuKernelArg {
public:
    CcuKernelArgDispatchSetup(uint32_t epWorldSize, uint32_t epRankId)
        : epWorldSize(epWorldSize), epRankId(epRankId) {}

    // 见 ccu_demo.h：签名用于框架生成 kernel 唯一键。最小版用默认构造（[T3]）。
    CcuKernelSignature GetKernelSignature() const override {
        return CcuKernelSignature();
    }

    uint32_t epWorldSize{1};
    uint32_t epRankId{0};
};

// -------- TaskArg：每次 Launch 传入的 device GM 地址 + 其 token --------
//   token 由 host 端 CcuRep::GetTokenInfo(addr, bytes) 算得，每块独立 buffer 各算各的。
class CcuTaskArgDispatchSetup : public CcuTaskArg {
public:
    CcuTaskArgDispatchSetup(uint64_t statusAddr, uint64_t statusToken,
                            uint64_t oneBufAddr, uint64_t oneBufToken)
        : statusAddr(statusAddr), statusToken(statusToken),
          oneBufAddr(oneBufAddr), oneBufToken(oneBufToken) {}

    uint64_t statusAddr{0};   // 本卡状态区 GM 起址（teardown 也读这里）
    uint64_t statusToken{0};  // statusAddr 的 token
    uint64_t oneBufAddr{0};   // host 预填 R 个 int32=1 的源 buffer GM 起址
    uint64_t oneBufToken{0};  // oneBufAddr 的 token
};

// -------- CCU kernel：Algorithm() 内建微码图，把 1 写到本卡状态区 --------
class CcuKernelDispatchSetup : public CcuKernel {
public:
    explicit CcuKernelDispatchSetup(const CcuKernelArg &arg) : CcuKernel(arg) {
        const auto *kernelArg = dynamic_cast<const CcuKernelArgDispatchSetup *>(&arg);
        epWorldSize_ = kernelArg->epWorldSize;
        epRankId_    = kernelArg->epRankId;
    }
    ~CcuKernelDispatchSetup() override {}

    // 设备侧算法：段 1 建资源 → 段 2 Load SQE（顺序须与 GeneArgs 完全一致）→ 段 10 LocalCopyNb 写 1。
    HcclResult Algorithm() override;

    // host→device 参数序列：device 侧 Load 顺序必须与此【逐字节一致】，否则读到错地址/错 token。
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override {
        const auto *taskArg = dynamic_cast<const CcuTaskArgDispatchSetup *>(&arg);
        return {taskArg->statusAddr, taskArg->statusToken,
                taskArg->oneBufAddr, taskArg->oneBufToken};
    }

private:
    uint32_t epWorldSize_{1};
    uint32_t epRankId_{0};

    // ---- 段 1 在 Algorithm() 内创建的 CCU 资源（成员仅为可读性，亦可局部变量）----
    CcuRep::Variable   statusAddrVar_;
    CcuRep::Variable   statusTokenVar_;
    CcuRep::Variable   oneBufAddrVar_;
    CcuRep::Variable   oneBufTokenVar_;
    CcuRep::LocalAddr  readyDst_;
    CcuRep::LocalAddr  readySrc_;
    CcuRep::CompletedEvent event_;
    CcuRep::CcuBuf     localCopyBuf_;   // [T1] A6 LocalCopyNb 要求的 UB 暂存 buf（块分配）
};

}  // namespace mc2

#endif  // DISPATCH_SETUP_CCU_H
