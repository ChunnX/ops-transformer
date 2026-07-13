/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  dispatch_setup_ccu.h —— CCU dispatch_setup（真实逐 token 跨卡搬运，A6 视角，mxfp8）。
 *  --------------------------------------------------------------------------------------
 *  设计（见 mc2_integration_mirror/SETUP_DESIGN.md，技术总监 2026-07-01 拍板 host 预算簿记）：
 *    setup CCU 只搬 token 数据（payload+scale），路由用 host 预算的 routeTable；wishCnt/expand_idx
 *    由 driver host 预算（CCU 硬件不支持动态标量 GM 写 StoreVariableAt）。
 *
 *  与 A6-proven 参考 moe_distribute_dispatch_ccu_setup.{h,cpp} 的三处承重适配：
 *    1. 数据区块序 rank-major → expert-major（block=(ge%E)*R+my_rank，块内 pos=出现次数）——
 *       落点全由 host 预算的 dataDstByteOff/scaleDstByteOff 决定（device 不算 ge/E、不乘法）。
 *    2. payload 与 scale 两个独立 GM 区（参考是拼接一条）——每 token 两次跨卡写。
 *    3. wishCnt 不由 CCU 产（host 预算），故无 metadata、无标量 GM 写。
 *
 *  产出（写到对端卡，= 已测 dispatch_teardown 的输入布局）：
 *    对端 dataRegion  [R*E*bs, h]    fp8_e4m3fn  expert-major，token payload 落 base+dataDstByteOff
 *    对端 scaleRegion [R*E*bs, h/32] e8m0        同块结构，scale 落 base+scaleDstByteOff
 *    本卡 commCmdInfo ready[0..R)=1（teardown AIV 轮询）
 *
 *  基类 Mc2CcuKernelBase（4 SE 数据面 API，见 dispatch_setup_ccu_api.h）；PreSync/CreateVariable(ch,XN)/
 *    CreateRemoteAddr/LocalCopyNb/WriteNbWithArriveNotify 来自 CcuKernel（ccu_kernel.h）。
 *
 *  a5/a6 隔离：Algorithm 体整体 #ifdef DISPATCH_CCU_A6；a5 桩 return（不引用 a6-only SE 符号 → a5 链接过）。
 */
#ifndef DISPATCH_SETUP_CCU_H
#define DISPATCH_SETUP_CCU_H

#include <vector>
#include <cstdint>
#include "dispatch_setup_ccu_api.h"   // Mc2CcuKernelBase + 4 SE API + CHK_RET + DISPATCH_SETUP_CCU_TAG

namespace mc2 {

using namespace hcomm;

// -------- KernelArg：构造期标量 --------
class CcuKernelArgDispatchSetup : public CcuKernelArg {
public:
    CcuKernelArgDispatchSetup(uint32_t epWorldSize, uint32_t epRankId,
                              uint32_t expertPerRank, uint32_t bs, uint32_t topK, uint32_t hiddenSize)
        : epWorldSize(epWorldSize), epRankId(epRankId), expertPerRank(expertPerRank),
          bs(bs), topK(topK), hiddenSize(hiddenSize),
          hBytes(static_cast<uint64_t>(hiddenSize)),              // fp8 payload：1 字节/元素
          scaleBytes(static_cast<uint64_t>(hiddenSize) / 32U) {}  // e8m0：h/32 字节（group=32）

    // ★ launch 前框架用签名做 kernel 身份识别/微码缓存键（对齐 a6-proven 参考：Append 名+全部标量参）。
    //   空签名在小 arg 数（status 版 4 个）侥幸能过，14 参数据面必须填充，否则 HcclCcuKernelLaunch ret=4
    //   (HCCL_E_INTERNAL, 进不了 Algorithm)。
    CcuKernelSignature GetKernelSignature() const override {
        CcuKernelSignature sig;
        sig.Append(std::string("Mc2DispatchSetupCcu"));
        sig.Append(epWorldSize);
        sig.Append(epRankId);
        sig.Append(expertPerRank);
        sig.Append(bs);
        sig.Append(topK);
        sig.Append(hiddenSize);
        return sig;
    }

    uint32_t epWorldSize{1};
    uint32_t epRankId{0};
    uint32_t expertPerRank{1};   // E（本设计 device 不用它算 ge/E，仅记录；路由已 host 预算）
    uint32_t bs{1};
    uint32_t topK{1};
    uint32_t hiddenSize{0};
    uint64_t hBytes{0};          // = h
    uint64_t scaleBytes{0};      // = h/32
};

// -------- TaskArg：每次 Launch 的 device GM 地址 + token（GeneArgs/Load 三处严格同序）--------
//   routeTable：host 预算 [bs*topK, 3] int64（★ LoadVariable 读，int64 排布步进 8）：
//     每 (i,j) = {dstRank, dataDstByteOff, scaleDstByteOff}。
//     dataDstByteOff = (block*bs+pos)*h；scaleDstByteOff = (block*bs+pos)*(h/32)；
//     block=(ge%E)*R+my_rank；pos=该 ge 在本卡 expert_ids 行主序出现次数（=expand_idx）。
//   dataRegion/scaleRegion：本卡【接收】数据区（对端往这写 → PreSync 广播本卡地址给对端；自发自也写这）。
class CcuTaskArgDispatchSetup : public CcuTaskArg {
public:
    CcuTaskArgDispatchSetup(
        uint64_t xDataAddr, uint64_t xScaleAddr, uint64_t routeTableAddr,
        uint64_t dataRegionAddr, uint64_t scaleRegionAddr,
        uint64_t commCmdInfoAddr, uint64_t oneBufAddr,
        uint64_t xDataToken, uint64_t xScaleToken, uint64_t routeTableToken,
        uint64_t dataRegionToken, uint64_t scaleRegionToken,
        uint64_t commCmdInfoToken, uint64_t oneBufToken)
        : xDataAddr(xDataAddr), xScaleAddr(xScaleAddr), routeTableAddr(routeTableAddr),
          dataRegionAddr(dataRegionAddr), scaleRegionAddr(scaleRegionAddr),
          commCmdInfoAddr(commCmdInfoAddr), oneBufAddr(oneBufAddr),
          xDataToken(xDataToken), xScaleToken(xScaleToken), routeTableToken(routeTableToken),
          dataRegionToken(dataRegionToken), scaleRegionToken(scaleRegionToken),
          commCmdInfoToken(commCmdInfoToken), oneBufToken(oneBufToken) {}

    uint64_t xDataAddr{0}, xScaleAddr{0}, routeTableAddr{0};
    uint64_t dataRegionAddr{0}, scaleRegionAddr{0};
    uint64_t commCmdInfoAddr{0}, oneBufAddr{0};
    uint64_t xDataToken{0}, xScaleToken{0}, routeTableToken{0};
    uint64_t dataRegionToken{0}, scaleRegionToken{0};
    uint64_t commCmdInfoToken{0}, oneBufToken{0};
};

// -------- CCU kernel：Algorithm() 建微码图，逐 token 把 payload+scale 写到对端 expert-major 数据区 --------
class CcuKernelDispatchSetup : public Mc2CcuKernelBase {
public:
    explicit CcuKernelDispatchSetup(const CcuKernelArg &arg) : Mc2CcuKernelBase(arg) {
        const auto *a = dynamic_cast<const CcuKernelArgDispatchSetup *>(&arg);
        epWorldSize_  = a->epWorldSize;
        epRankId_     = a->epRankId;
        bs_           = a->bs;
        topK_         = a->topK;
        hBytes_       = a->hBytes;
        scaleBytes_   = a->scaleBytes;
    }
    ~CcuKernelDispatchSetup() override {}

    HcclResult Algorithm() override;

    // host→device 参数序列（device Load 顺序必须逐字节同序）。
    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override {
        const auto *t = dynamic_cast<const CcuTaskArgDispatchSetup *>(&arg);
        std::vector<uint64_t> full = {
            t->xDataAddr, t->xScaleAddr, t->routeTableAddr,
            t->dataRegionAddr, t->scaleRegionAddr, t->commCmdInfoAddr, t->oneBufAddr,
            t->xDataToken, t->xScaleToken, t->routeTableToken,
            t->dataRegionToken, t->scaleRegionToken, t->commCmdInfoToken, t->oneBufToken,
        };
#ifdef LAUNCH_ARGN
        // ★ 隔离探针 B：只返回前 LAUNCH_ARGN 个 arg，扫 arg 数看 ops-transformer 框架 launch 何时开始 ret=4
        //   （9fc2045 用 4 个能跑；本数据面 14 个）。★必配 -DLAUNCH_PROBE（Algorithm 早返回），否则 Load 少参越界。
        if (static_cast<size_t>(LAUNCH_ARGN) < full.size()) full.resize(static_cast<size_t>(LAUNCH_ARGN));
#endif
        return full;
    }

private:
    uint32_t epWorldSize_{1};
    uint32_t epRankId_{0};
    uint32_t bs_{1};
    uint32_t topK_{1};
    uint64_t hBytes_{0};
    uint64_t scaleBytes_{0};

    // ---- Algorithm() 内建的 CCU 资源 ----
    CcuRep::Variable xDataAddrVar_, xScaleAddrVar_, routeTableAddrVar_;
    CcuRep::Variable dataRegionAddrVar_, scaleRegionAddrVar_, commCmdInfoAddrVar_, oneBufAddrVar_;
    CcuRep::Variable xDataTokenVar_, xScaleTokenVar_, routeTableTokenVar_;
    CcuRep::Variable dataRegionTokenVar_, scaleRegionTokenVar_, commCmdInfoTokenVar_, oneBufTokenVar_;
    // PreSync 交换来的各对端【接收】区起址+token（写对端时用）
    std::vector<CcuRep::Variable> remoteDataAddrs_, remoteDataTokens_;
    std::vector<CcuRep::Variable> remoteScaleAddrs_, remoteScaleTokens_;
    std::vector<CcuRep::Variable> sendBytesPerPeer_;   // 段5 累加 → 段6 RecordExpectCount
    CcuRep::LocalAddr  localSrcData_, localSrcScale_, localDstData_, localDstScale_;
    CcuRep::RemoteAddr remoteDstData_, remoteDstScale_;
    CcuRep::CompletedEvent event_;
    CcuRep::CcuBuf localCopyBuf_;   // A6 LocalCopyNb 的 UB 暂存 buf（CreateBlockCcuBuf）
};

}  // namespace mc2

#endif  // DISPATCH_SETUP_CCU_H
