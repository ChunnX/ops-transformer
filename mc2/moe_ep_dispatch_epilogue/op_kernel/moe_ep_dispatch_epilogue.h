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
 * \file moe_ep_dispatch_epilogue.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_EPILOGUE_H
#define MOE_EP_DISPATCH_EPILOGUE_H

#include "kernel_operator.h"
#include "moe_ep_dispatch_epilogue_tiling_key.h"

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#include "../common/op_kernel/mc2_moe_context.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#endif

#include "moe_ep_dispatch_epilogue_tiling.h"

namespace MoeEpDispatchEpilogueImpl {

using namespace AscendC;

static constexpr uint32_t UB_ALIGN = 32U;
static constexpr uint32_t WIN_ADDR_ALIGN = 512;
static constexpr uint32_t RECV_META_FIELDS = 4;

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
class MoeEpDispatchEpilogue {
public:
    __aicore__ inline MoeEpDispatchEpilogue() {};
    __aicore__ inline void Init(
        GM_ADDR context,
        GM_ADDR dstBufferSlotIdx,
        GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR cachedRecvSrcMetadata,
        GM_ADDR recvX,
        GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales,
        GM_ADDR workspace, GM_ADDR tilingGM,
        TPipe* pipe, const MoeEpDispatchEpilogueInfo* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputePrefixSums();
    __aicore__ inline void CopyFromWindowByCachedMeta();
    // AtomicAdd 行号抢占路径: 用每专家原子计数器替代 CountHits+跨核前缀和,
    // 单趟 flatten 拷出, 段内行序非确定(按专家分段仍正确, 4元组自描述可回散)。
    __aicore__ inline void InitAtomicCounters();
    __aicore__ inline void CopyFromWindowByAtomic();

    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId,
                                       uint32_t &endId, uint32_t &sendNum);
    __aicore__ inline GM_ADDR GetWinAddrByRankId(__gm__ Mc2Aclnn::Mc2MoeContext *ctx, uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)ctx->epHcclBuffer_[rankId] + offset;
    }

    TPipe* tpipe_{nullptr};
    __gm__ Mc2Aclnn::Mc2MoeContext* mc2Context_{nullptr};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    GlobalTensor<int32_t> numRecvPerRankGm_;
    GlobalTensor<int64_t> numRecvPerExpertGm_;

    GlobalTensor<XType> recvXGm_;
    GlobalTensor<float> recvTopkWeightsGm_;
    GlobalTensor<int32_t> recvSrcMetadataGm_;
    GlobalTensor<ScalesType> recvScalesGm_;
    GlobalTensor<int32_t> cachedRecvSrcMetadataGm_;   // cached 路径专用：来自上一轮 dispatch 的 recv_src_metadata

    GlobalTensor<int32_t> hitCountGm_;

    LocalTensor<XType> ubToken_;
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<int32_t> ubRecvCnt_;
    LocalTensor<int64_t> ubExpertPfx_;
    LocalTensor<float> ubStageWeights_;
    LocalTensor<int32_t> ubStageMeta_;
    LocalTensor<ScalesType> ubScales_;

    TBuf<QuePosition::VECIN> ubTokenBuf_;
    TBuf<QuePosition::VECIN> ubMetaBuf_;
    TBuf<QuePosition::VECIN> ubRecvCntBuf_;
    TBuf<QuePosition::VECIN> ubExpertPfxBuf_;
    TBuf<QuePosition::VECIN> ubStageWeightsBuf_;
    TBuf<QuePosition::VECIN> ubStageMetaBuf_;
    TBuf<QuePosition::VECIN> ubScalesBuf_;

    uint32_t expertSum_{0};
    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR localWinAddr_{nullptr};
    uint32_t scalesOffset_{0};
    uint32_t scalesBytes_{0};
    uint32_t scalesElems_{0};
    uint32_t metaOffset_{0};
    uint32_t metaBytes_{0};
    uint32_t paddedMetaElems_{0};
    uint32_t axisKAlign_{0};
    uint32_t numLocalExperts_{0};
    uint32_t aivNum_{0};
    uint32_t axisK_{0};
    uint32_t axisH_{0};
    uint32_t epWorldSize_{0};
    uint32_t numMaxTokensPerRank_{0};
    uint32_t perSlotBytes_{0};
    uint64_t winDataOffset_{0};
};

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId, uint32_t &sendNum)
{
    sendNum = curSendCnt / curUseAivNum;
    uint32_t remainderNum = curSendCnt % curUseAivNum;
    uint32_t newAivId = aivId_;
    startId = sendNum * newAivId;
    if (newAivId < remainderNum) {
        sendNum += 1;
        startId += newAivId;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::Init(
    GM_ADDR context,
    GM_ADDR dstBufferSlotIdx,
    GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR cachedRecvSrcMetadata,
    GM_ADDR recvX,
    GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales,
    GM_ADDR workspace, GM_ADDR tilingGM,
    TPipe* pipe, const MoeEpDispatchEpilogueInfo* tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspace;
    numLocalExperts_ = tilingData->cfg.numLocalExperts;
    aivNum_ = tilingData->aivNum;
    axisK_ = tilingData->cfg.topK;
    axisH_ = tilingData->cfg.hidden;
    epWorldSize_ = tilingData->cfg.epWorldSize;
    numMaxTokensPerRank_ = tilingData->cfg.numMaxTokensPerRank;
    perSlotBytes_ = tilingData->cfg.perSlotBytes;
    winDataOffset_ = tilingData->winDataOffset;

    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::Mc2MoeContext*>(context);
    epRankId_ = mc2Context_->epRankId;
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_);
    metaOffset_ = Ceil((uint32_t)(axisH_ * sizeof(XType)), UB_ALIGN) * UB_ALIGN;

    numRecvPerRankGm_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRank);
    numRecvPerExpertGm_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpert);
    cachedRecvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)cachedRecvSrcMetadata);
    recvXGm_.SetGlobalBuffer((__gm__ XType*)recvX);
    recvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadata);
    if constexpr (HasTopkWeights) {
        recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float*)recvTopkWeights);
    }

    hitCountGm_.SetGlobalBuffer((__gm__ int32_t*)(workspace));

    uint32_t ubTokenBytes     = Ceil((uint32_t)(axisH_ * sizeof(XType)), UB_ALIGN) * UB_ALIGN;

    axisKAlign_ = Ceil(axisK_, 8U) * 8U;
    metaBytes_ = (2 * axisKAlign_) * (uint32_t)sizeof(int32_t) + UB_ALIGN;
    paddedMetaElems_ = Ceil(2 * axisKAlign_ + 2, 8U) * 8U;
    uint32_t maxSlotsPerAiv = Ceil(numMaxTokensPerRank_, aivNum_);
    uint32_t ubMetaBytes = Ceil((uint32_t)(maxSlotsPerAiv * paddedMetaElems_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubRecvCntBytes = Ceil((uint32_t)(epWorldSize_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubExpertPfxBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(ubRecvCntBuf_, ubRecvCntBytes);
    tpipe_->InitBuffer(ubExpertPfxBuf_, ubExpertPfxBytes);
    tpipe_->InitBuffer(ubTokenBuf_, ubTokenBytes);
    ubRecvCnt_ = ubRecvCntBuf_.Get<int32_t>();
    ubToken_ = ubTokenBuf_.Get<XType>();
    ubExpertPfx_ = ubExpertPfxBuf_.Get<int64_t>();

    if constexpr (!IsCached) {
        tpipe_->InitBuffer(ubMetaBuf_, ubMetaBytes);
        ubMeta_ = ubMetaBuf_.Get<int32_t>();
    }

    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesOffset_ = Ceil((uint32_t)(axisH_ * sizeof(XType)), UB_ALIGN) * UB_ALIGN;
        scalesBytes_ = tilingData->cfg.scalesBytes;
        scalesElems_ = (scalesBytes_ == 0U) ? 0U : (scalesBytes_ / sizeof(ScalesType));
        uint32_t scalesBytesAlign = Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN;
        metaOffset_ += scalesBytesAlign;
        tpipe_->InitBuffer(ubScalesBuf_, scalesBytesAlign);
        ubScales_ = ubScalesBuf_.Get<ScalesType>();
        recvScalesGm_.SetGlobalBuffer((__gm__ ScalesType *)recvScales);
    }

    DataCopyExtParams expertPfxCopyParams{1U, static_cast<uint32_t>(numLocalExperts_ * sizeof(int64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int64_t> expertPfxPadParams{false, 0U, 0U, 0};
    DataCopyPad(ubExpertPfx_, numRecvPerExpertGm_, expertPfxCopyParams, expertPfxPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t expertSum = 0;
    for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
        expertSum += static_cast<uint32_t>(ubExpertPfx_.GetValue(localExpertIdx));
    }
    expertSum_ = expertSum;

    DataCopyExtParams recvCntCopyParams{1U, static_cast<uint32_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> recvCntPadParams{false, 0U, 0U, 0};
    DataCopyPad(ubRecvCnt_, numRecvPerRankGm_, recvCntCopyParams, recvCntPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::Process()
{
    if constexpr (!IsCached) {
        // AtomicAdd 行号抢占路径, 省掉 CountHits 扫描与跨核前缀回读。
        ComputePrefixSums();
        if (expertSum_ != 0) {
            InitAtomicCounters();
            SyncAll<true>();
            CopyFromWindowByAtomic();
        }
    } else {
        CopyFromWindowByCachedMeta();
    }
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::ComputePrefixSums()
{
    int64_t cumulativeRowOffset = 0;
    for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
        int64_t expertTokenCnt = ubExpertPfx_.GetValue(localExpertIdx);
        ubExpertPfx_.SetValue(localExpertIdx, cumulativeRowOffset);
        cumulativeRowOffset += expertTokenCnt;
    }
    SyncFunc<AscendC::HardEvent::S_V>();
}

template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CopyFromWindowByCachedMeta()
{
    if (expertSum_ == 0) { return; }

    uint32_t startId, endId, cnt;
    SplitToCore(expertSum_, aivNum_, startId, endId, cnt);
    if (startId >= endId) { return; }

    uint32_t ubMetaBytes = Ceil(metaBytes_, UB_ALIGN) * UB_ALIGN;
    uint32_t ubStageWeightsBytes = Ceil((uint32_t)(cnt * sizeof(float)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubStageMetaBytes = Ceil((uint32_t)(cnt * RECV_META_FIELDS * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(ubMetaBuf_, ubMetaBytes);
    tpipe_->InitBuffer(ubStageWeightsBuf_, ubStageWeightsBytes);
    tpipe_->InitBuffer(ubStageMetaBuf_, ubStageMetaBytes);
    ubMeta_ = ubMetaBuf_.Get<int32_t>();
    ubStageWeights_ = ubStageWeightsBuf_.Get<float>();
    ubStageMeta_ = ubStageMetaBuf_.Get<int32_t>();

    DataCopyExtParams metaInParams{1U, static_cast<uint32_t>(cnt * RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
    DataCopyPad(ubStageMeta_,
                cachedRecvSrcMetadataGm_[(int64_t)startId * RECV_META_FIELDS], metaInParams, metaPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t i = 0; i < cnt; ++i) {
        uint32_t metaBase = i * RECV_META_FIELDS;
        int32_t srcRankId   = ubStageMeta_.GetValue(metaBase + 0);
        int32_t srcTopkIdx  = ubStageMeta_.GetValue(metaBase + 2);
        int32_t slotIdx     = ubStageMeta_.GetValue(metaBase + 3);

        GM_ADDR slotAddr = localWinAddr_
            + (int64_t)srcRankId * numMaxTokensPerRank_ * perSlotBytes_
            + (int64_t)slotIdx * perSlotBytes_;

        GlobalTensor<XType> srcTokenTensor;
        srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(slotAddr), axisH_);
        DataCopyParams tokenCopyParams{1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
        DataCopyPadParams tokenPadParams{false, 0, 0, 0};
        DataCopyPad(ubToken_, srcTokenTensor, tokenCopyParams, tokenPadParams);

        if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
            GlobalTensor<ScalesType> srcScalesTensor;
            srcScalesTensor.SetGlobalBuffer(
                reinterpret_cast<__gm__ ScalesType*>(slotAddr + scalesOffset_), scalesElems_);
            DataCopyParams scalesCopyParams{1U,
                static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
            DataCopyPadParams scalesPadParams{false, 0, 0, 0};
            DataCopyPad(ubScales_, srcScalesTensor, scalesCopyParams, scalesPadParams);
        }

        if constexpr (HasTopkWeights) {
            GlobalTensor<int32_t> srcMetaGm;
            srcMetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slotAddr + metaOffset_));
            DataCopyExtParams slotMetaParams{1U, metaBytes_, 0U, 0U, 0U};
            DataCopyPad(ubMeta_, srcMetaGm, slotMetaParams, metaPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            float weight = ubMeta_.ReinterpretCast<float>().GetValue(axisKAlign_ + srcTopkIdx);
            ubStageWeights_.SetValue(i, weight);
            SyncFunc<AscendC::HardEvent::S_MTE2>();
        }

        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        uint32_t globalRow = startId + i;
        DataCopyPad(recvXGm_[(int64_t)globalRow * axisH_], ubToken_, tokenCopyParams);
        if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
            DataCopyParams scalesCopyParams{1U,
                static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
            DataCopyPad(recvScalesGm_[(int64_t)globalRow * scalesElems_], ubScales_, scalesCopyParams);
        }
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    }

    if constexpr (HasTopkWeights) {
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams weightOutParams{1U, static_cast<uint32_t>(cnt * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(recvTopkWeightsGm_[startId], ubStageWeights_, weightOutParams);
    }
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopyExtParams metaOutParams{1U, static_cast<uint32_t>(cnt * RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(recvSrcMetadataGm_[(int64_t)startId * RECV_META_FIELDS], ubStageMeta_, metaOutParams);
}

// aiv0 用专家段基址初始化每专家原子计数器(复用 hitCountGm_ 即 workspace 前 E 个 int32)。
// 计数器初值 = expertSegmentBase[e] = ComputePrefixSums 后的 ubExpertPfx_[e]。
// 全核 SyncAll 后, 各核 AtomicAdd(counter[e], 1) 抢占的行号恰落在 [base[e], base[e]+numRecvPerExpert[e])。
template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::InitAtomicCounters()
{
    if (aivId_ != 0) { return; }
    for (uint32_t e = 0; e < numLocalExperts_; ++e) {
        int32_t base = static_cast<int32_t>(ubExpertPfx_.GetValue(e));
        hitCountGm_.SetValue(e, base);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

// AtomicAdd 行号抢占路径: flatten 单趟(rank × slot × topk), 无 targetExpert 外层。
// 每个 (token, topk) 命中后, 原子地从其所属专家段的计数器取一个行号, token 即时写、
// 权重/4元组元数据用标量 GM 写(S pipe, 无额外同步点)。省掉 CountHits+SyncAll扫描+跨核前缀回读。
template <typename XType, typename ScalesType, uint32_t IsCached, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogue<XType, ScalesType, IsCached, HasTopkWeights>::CopyFromWindowByAtomic()
{
    __gm__ int32_t* counterBase = hitCountGm_.GetPhyAddr(0);

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        int32_t slotCnt = ubRecvCnt_.GetValue(rankId);
        if (slotCnt == 0) { continue; }

        uint32_t slotStart, slotEnd, slotCntPerAiv;
        SplitToCore(static_cast<uint32_t>(slotCnt), aivNum_, slotStart, slotEnd, slotCntPerAiv);
        if (slotStart >= slotEnd) { continue; }

        GM_ADDR srcRankBase = localWinAddr_
                              + (int64_t)rankId * numMaxTokensPerRank_ * perSlotBytes_;
        GM_ADDR firstSlotAddrThisAiv = srcRankBase + (int64_t)slotStart * perSlotBytes_;

        GlobalTensor<int32_t> srcMetaGm;
        srcMetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(firstSlotAddrThisAiv + metaOffset_));
        DataCopyExtParams metaCopyParams{static_cast<uint16_t>(slotCntPerAiv), metaBytes_,
            perSlotBytes_ - metaBytes_, 0, 0};
        DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
        DataCopyPad(ubMeta_, srcMetaGm, metaCopyParams, metaPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t localSlot = 0; localSlot < slotCntPerAiv; ++localSlot) {
            uint32_t metaBase = localSlot * paddedMetaElems_;

            for (uint32_t topkIdx = 0; topkIdx < axisK_; ++topkIdx) {
                int32_t expertId = ubMeta_.GetValue(metaBase + topkIdx);
                if (expertId < 0) { continue; }
                uint32_t dstRank = static_cast<uint32_t>(expertId) / numLocalExperts_;
                if (dstRank != epRankId_) { continue; }
                uint32_t localExpertId = static_cast<uint32_t>(expertId) % numLocalExperts_;

                // 原子抢占该专家段内的行号(返回旧值即写入行)。
                int32_t row = AscendC::AtomicAdd<int32_t>(counterBase + localExpertId, 1);
                uint32_t globalRow = static_cast<uint32_t>(row);

                uint32_t slotIdx = slotStart + localSlot;
                GM_ADDR slotAddr = srcRankBase + (int64_t)slotIdx * perSlotBytes_;
                GlobalTensor<XType> srcTokenTensor;
                srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(slotAddr), axisH_);
                DataCopyParams tokenCopyParams{1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
                DataCopyPadParams tokenPadParams{false, 0, 0, 0};
                DataCopyPad(ubToken_, srcTokenTensor, tokenCopyParams, tokenPadParams);
                if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                    GlobalTensor<ScalesType> srcScalesTensor;
                    srcScalesTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScalesType*>(slotAddr + scalesOffset_),
                        scalesElems_);
                    DataCopyParams scalesCopyParams{1U,
                        static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
                    DataCopyPadParams scalesPadParams{false, 0, 0, 0};
                    DataCopyPad(ubScales_, srcScalesTensor, scalesCopyParams, scalesPadParams);
                }
                SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                DataCopyPad(recvXGm_[(int64_t)globalRow * axisH_], ubToken_, tokenCopyParams);
                if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
                    DataCopyParams scalesCopyParams{1U,
                        static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
                    DataCopyPad(recvScalesGm_[(int64_t)globalRow * scalesElems_], ubScales_, scalesCopyParams);
                }
                SyncFunc<AscendC::HardEvent::MTE3_S>();

                // 权重 + 4元组元数据: 标量 GM 写(S pipe), 行号离散故无法段内突发, 改为 per-hit 直写。
                if constexpr (HasTopkWeights) {
                    float weights = ubMeta_.ReinterpretCast<float>().GetValue(metaBase + axisKAlign_ + topkIdx);
                    recvTopkWeightsGm_.SetValue(globalRow, weights);
                }
                recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 0,
                    ubMeta_.GetValue(metaBase + 2 * axisKAlign_));
                recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 1,
                    ubMeta_.GetValue(metaBase + 2 * axisKAlign_ + 1));
                recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 2,
                    static_cast<int32_t>(topkIdx));
                recvSrcMetadataGm_.SetValue(globalRow * RECV_META_FIELDS + 3,
                    static_cast<int32_t>(slotIdx));
                SyncFunc<AscendC::HardEvent::S_MTE2>();
                SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
    // 确保标量 GM 写(权重/元组)在 kernel 结束前全部落地。
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

}  // namespace MoeEpDispatchEpilogueImpl

#endif  // MOE_EP_DISPATCH_EPILOGUE_H
