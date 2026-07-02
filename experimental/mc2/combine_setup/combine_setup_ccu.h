/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 *  combine_setup_ccu.h —— CCU combine_setup（真实逐 token 送回，A6 视角，mxfp8）。
 *  ------------------------------------------------------------------------------------
 *  与 dispatch_setup 完全对称、方向反转（见 SETUP_DESIGN.md §3）：
 *    源 = 本卡 expandX(mxfp8: payload+scale, A 行, expert-major)；按 host 预算 assistTable 送回 owner 卡的
 *    combine 数据区（每全局专家一块 ge, 块内 pos=expand_idx）。padding 行 ownerRank=-1(0xFFFF..FF) 哨兵 → 跳过（与 .cpp / host OWNER_PAD_SENTINEL=-1 对齐，勿写 R）。
 *    产出 = 已测 combine_teardown 的输入布局。
 *
 *  assistTable [A, 3] int64（host 预算，LoadVariable 读，int64 步进 8）：每行 = {ownerRank, dataDstByteOff,
 *    scaleDstByteOff}；dataDstByteOff=(ge*bs+expand_idx)*h；scaleDstByteOff=(ge*bs+expand_idx)*(h/32)。
 *  A = R*bs*min(topk,E)（expandX 行上限，host 展开上界）。
 *
 *  a5/a6 隔离：Algorithm 体 #ifdef COMBINE_CCU_A6；a5 桩 return。
 */
#ifndef COMBINE_SETUP_CCU_H
#define COMBINE_SETUP_CCU_H

#include <vector>
#include <cstdint>
#include "combine_setup_ccu_api.h"

namespace mc2 {

using namespace hcomm;

// -------- KernelArg：构造期标量 --------
class CcuKernelArgCombineSetup : public CcuKernelArg {
public:
    CcuKernelArgCombineSetup(uint32_t epWorldSize, uint32_t epRankId,
                             uint32_t expertPerRank, uint32_t bs, uint32_t topK, uint32_t hiddenSize)
        : epWorldSize(epWorldSize), epRankId(epRankId), expertPerRank(expertPerRank),
          bs(bs), topK(topK), hiddenSize(hiddenSize),
          hBytes(static_cast<uint64_t>(hiddenSize)),
          scaleBytes(static_cast<uint64_t>(hiddenSize) / 32U),
          A(static_cast<uint64_t>(epWorldSize) * bs * ((topK < expertPerRank) ? topK : expertPerRank)) {}

    // ★ 同 dispatch：launch 前框架用签名做 kernel 身份/微码缓存键，14 参数据面必须填充（空 → ret=4）。
    CcuKernelSignature GetKernelSignature() const override {
        CcuKernelSignature sig;
        sig.Append(std::string("Mc2CombineSetupCcu"));
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
    uint32_t expertPerRank{1};
    uint32_t bs{1};
    uint32_t topK{1};
    uint32_t hiddenSize{0};
    uint64_t hBytes{0};
    uint64_t scaleBytes{0};
    uint64_t A{0};               // = R*bs*min(topk,E)
};

// -------- TaskArg --------
class CcuTaskArgCombineSetup : public CcuTaskArg {
public:
    CcuTaskArgCombineSetup(
        uint64_t expandXDataAddr, uint64_t expandXScaleAddr, uint64_t assistTableAddr,
        uint64_t combineDataAddr, uint64_t combineScaleAddr,
        uint64_t commCmdInfoAddr, uint64_t oneBufAddr,
        uint64_t expandXDataToken, uint64_t expandXScaleToken, uint64_t assistTableToken,
        uint64_t combineDataToken, uint64_t combineScaleToken,
        uint64_t commCmdInfoToken, uint64_t oneBufToken)
        : expandXDataAddr(expandXDataAddr), expandXScaleAddr(expandXScaleAddr), assistTableAddr(assistTableAddr),
          combineDataAddr(combineDataAddr), combineScaleAddr(combineScaleAddr),
          commCmdInfoAddr(commCmdInfoAddr), oneBufAddr(oneBufAddr),
          expandXDataToken(expandXDataToken), expandXScaleToken(expandXScaleToken), assistTableToken(assistTableToken),
          combineDataToken(combineDataToken), combineScaleToken(combineScaleToken),
          commCmdInfoToken(commCmdInfoToken), oneBufToken(oneBufToken) {}

    uint64_t expandXDataAddr{0}, expandXScaleAddr{0}, assistTableAddr{0};
    uint64_t combineDataAddr{0}, combineScaleAddr{0};
    uint64_t commCmdInfoAddr{0}, oneBufAddr{0};
    uint64_t expandXDataToken{0}, expandXScaleToken{0}, assistTableToken{0};
    uint64_t combineDataToken{0}, combineScaleToken{0};
    uint64_t commCmdInfoToken{0}, oneBufToken{0};
};

// -------- CCU kernel --------
class CcuKernelCombineSetup : public Mc2CcuKernelBase {
public:
    explicit CcuKernelCombineSetup(const CcuKernelArg &arg) : Mc2CcuKernelBase(arg) {
        const auto *a = dynamic_cast<const CcuKernelArgCombineSetup *>(&arg);
        epWorldSize_ = a->epWorldSize;
        epRankId_    = a->epRankId;
        A_           = a->A;
        hBytes_      = a->hBytes;
        scaleBytes_  = a->scaleBytes;
    }
    ~CcuKernelCombineSetup() override {}

    HcclResult Algorithm() override;

    std::vector<uint64_t> GeneArgs(const CcuTaskArg &arg) override {
        const auto *t = dynamic_cast<const CcuTaskArgCombineSetup *>(&arg);
        return {
            t->expandXDataAddr, t->expandXScaleAddr, t->assistTableAddr,
            t->combineDataAddr, t->combineScaleAddr, t->commCmdInfoAddr, t->oneBufAddr,
            t->expandXDataToken, t->expandXScaleToken, t->assistTableToken,
            t->combineDataToken, t->combineScaleToken, t->commCmdInfoToken, t->oneBufToken,
        };
    }

private:
    uint32_t epWorldSize_{1};
    uint32_t epRankId_{0};
    uint64_t A_{0};
    uint64_t hBytes_{0};
    uint64_t scaleBytes_{0};

    CcuRep::Variable expandXDataAddrVar_, expandXScaleAddrVar_, assistTableAddrVar_;
    CcuRep::Variable combineDataAddrVar_, combineScaleAddrVar_, commCmdInfoAddrVar_, oneBufAddrVar_;
    CcuRep::Variable expandXDataTokenVar_, expandXScaleTokenVar_, assistTableTokenVar_;
    CcuRep::Variable combineDataTokenVar_, combineScaleTokenVar_, commCmdInfoTokenVar_, oneBufTokenVar_;
    std::vector<CcuRep::Variable> remoteDataAddrs_, remoteDataTokens_;
    std::vector<CcuRep::Variable> remoteScaleAddrs_, remoteScaleTokens_;
    std::vector<CcuRep::Variable> sendBytesPerPeer_;
    CcuRep::LocalAddr  localSrcData_, localSrcScale_, localDstData_, localDstScale_;
    CcuRep::RemoteAddr remoteDstData_, remoteDstScale_;
    CcuRep::CompletedEvent event_;
    CcuRep::CcuBuf localCopyBuf_;
};

}  // namespace mc2

#endif  // COMBINE_SETUP_CCU_H
