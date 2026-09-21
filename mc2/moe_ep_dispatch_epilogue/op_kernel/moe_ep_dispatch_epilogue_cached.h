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
 * \file moe_ep_dispatch_epilogue_cached.h
 * \brief cached 路径：直接消费上一轮 dispatch 落下的 recv_src_metadata / rank offsets 搬运 recv_x。
 *
 * 公共状态、Init 前半段、WaitDispatch 与工具函数都在 moe_ep_dispatch_epilogue_base.h；
 * 本文件只有这条路径的成员与流水。
 */

#ifndef MOE_EP_DISPATCH_EPILOGUE_CACHED_H
#define MOE_EP_DISPATCH_EPILOGUE_CACHED_H

#include "moe_ep_dispatch_epilogue_base.h"

namespace MoeEpDispatchEpilogueImpl {

#if defined(ENABLE_MOE_EP_DISPATCH_EPILOGUE_KERNEL)

using namespace AscendC;

// 本路径专有的常量（公共量见 base）。
static constexpr uint8_t BUFFER_NUM = 2;
// Five-field metadata needs 20 bytes per row. A 4096-row tile keeps this cached staging buffer at 80 KiB.
static constexpr uint32_t CACHED_META_TILE = 4096U;

template <typename XType, typename ScalesType, bool HasTopkWeights>
class MoeEpDispatchEpilogueCached : public MoeEpDispatchEpilogueBase<XType, ScalesType, HasTopkWeights> {
public:
    __aicore__ inline MoeEpDispatchEpilogueCached(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR numRecvPerRank,
                                GM_ADDR numRecvPerExpert, GM_ADDR cachedRecvSrcMetadata, GM_ADDR recvX,
                                GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales, GM_ADDR workspace,
                                GM_ADDR tilingGM, TPipe *pipe, const MoeEpDispatchEpilogueInfo *tilingData);
    __aicore__ inline void Process();

private:
    using Base = MoeEpDispatchEpilogueBase<XType, ScalesType, HasTopkWeights>;

    // 基类是依赖基类（dependent base），派生类的非限定查找不会进去，必须显式引入。这里一次性列全，
    // 换来的是下面所有算法代码保持原样、不带 this-> 前缀，重构前后可以直接逐行对照。
    using Base::tpipe_;
    using Base::mc2Context_;
    using Base::diagWriter_;
    using Base::epRankId_;
    using Base::networkMode_;
    using Base::aivId_;
    using Base::numRecvPerExpertGm_;
    using Base::xGm_;
    using Base::recvXGm_;
    using Base::recvTopkWeightsGm_;
    using Base::recvSrcMetadataGm_;
    using Base::recvRankOffsetsGm_;
    using Base::recvScalesGm_;
    using Base::ubRankOffsets_;
    using Base::ubExpertPfx_;
    using Base::ubWaitStatus_;
    using Base::ubWaitSum_;
    using Base::ubRankOffsetsBuf_;
    using Base::ubExpertPfxBuf_;
    using Base::waitStatusBuf_;
    using Base::waitSumBuf_;
    using Base::sharedTmpBuf_;
    using Base::expertSum_;
    using Base::workspaceGM_;
    using Base::localWinAddr_;
    using Base::localSlotStateWinAddr_;
    using Base::scalesOffset_;
    using Base::scalesStride_;
    using Base::scalesBytes_;
    using Base::scalesBytesAlign_;
    using Base::scalesElems_;
    using Base::metaOffset_;
    using Base::tokenQueueBufBytes_;
    using Base::metaBytes_;
    using Base::axisKAlign_;
    using Base::hAlignSize_;
    using Base::numLocalExperts_;
    using Base::aivNum_;
    using Base::axisK_;
    using Base::axisH_;
    using Base::epWorldSize_;
    using Base::numMaxTokensPerRank_;
    using Base::perSlotBytes_;
    using Base::dispatchNotifyCount_;
    using Base::totalNotifyCnt_;
    using Base::winDataOffset_;
    using Base::slotWinStateOffset_;
    using Base::InitCommon;
    using Base::FinalizeInit;
    using Base::WaitDispatch;
    using Base::SplitToCore;
    using Base::GetWinAddrByRankId;
    using Base::ReduceSumWorkNeedSize;

    // 本路径的搬运按「行」组织：一行 = 上一轮 dispatch 留下的一条 recv_src_metadata。
    struct CachedRow {
        GM_ADDR slotAddr{nullptr}; // 本行 token 在窗口里的槽地址
        uint32_t recvXRow{0};      // recv_x / weights 的目标行号
        int32_t srcRankId{0};
        int32_t tokenIdx{0};
        int32_t srcTopkIdx{0};
    };

    // ---------------- 行区间与 UB ----------------
    __aicore__ inline void CopyFromWindowByCachedMeta();
    __aicore__ inline bool ClaimCachedRows(uint32_t &startRow, uint32_t &endRow);
    __aicore__ inline void InitCachedBuffers();
    __aicore__ inline void CopyCachedRankOffsets();

    // ---------------- 拍级：CACHED_META_TILE 行 metadata ----------------
    __aicore__ inline void CopyCachedTile(uint32_t tileStart, uint32_t tileCnt);
    __aicore__ inline void LoadCachedMetaTile(uint32_t tileStart, uint32_t tileCnt);
    __aicore__ inline void FlushCachedMetaTile(uint32_t tileStart, uint32_t tileCnt);

    // ---------------- 行级：读 metadata → 发读 → 落 recv_x/scales → 落 weight ----------------
    __aicore__ inline void CopyCachedRow(uint32_t rowInTile);
    __aicore__ inline bool ReadCachedRow(uint32_t rowInTile, CachedRow &row);
    __aicore__ inline LocalTensor<XType> LoadCachedToken(const CachedRow &row);
    __aicore__ inline void LoadCachedWeight(const CachedRow &row);
    __aicore__ inline void StoreCachedToken(const CachedRow &row);
    __aicore__ inline void StoreCachedWeight(const CachedRow &row);

    GlobalTensor<int32_t> cachedRecvSrcMetadataGm_; // 来自上一轮 dispatch 的 recv_src_metadata
    GlobalTensor<int32_t> cachedRecvRankOffsetsGm_; // 来自上一轮 dispatch 的 rank offsets
    LocalTensor<float> ubStageWeights_;
    LocalTensor<int32_t> ubStageMeta_;
    LocalTensor<int32_t> ubMeta_;

    TBuf<QuePosition::VECIN> ubStageWeightsBuf_;
    TBuf<QuePosition::VECIN> ubStageMetaBuf_;
    TBuf<QuePosition::VECIN> ubMetaBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tokenQueue_;
};

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
    GM_ADDR cachedRecvSrcMetadata, GM_ADDR recvX, GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales,
    GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe, const MoeEpDispatchEpilogueInfo *tilingData)
{
    InitCommon(context, x, numRecvPerExpert, recvX, recvSrcMetadata, recvTopkWeights, recvScales, workspace, tilingGM,
               pipe, tilingData);

    // topkIdx / numRecvPerRank 只有非 cached 路径用得到，本路径不绑。
    cachedRecvSrcMetadataGm_.SetGlobalBuffer((__gm__ int32_t *)cachedRecvSrcMetadata);
    cachedRecvRankOffsetsGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(cachedRecvSrcMetadata + tilingData->metadataRankOffsetsOffset));

    // hAlignSize_ / scalesBytesAlign_ 由 InitCommon 算好（含 fp8 分支），这里只做拼装。
    tokenQueueBufBytes_ = hAlignSize_ + scalesBytesAlign_;
    tpipe_->InitBuffer(tokenQueue_, BUFFER_NUM, tokenQueueBufBytes_);

    FinalizeInit();
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::Process()
{
    WaitDispatch();
    SyncAll<true>();
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_WAIT_DONE);
    CopyCachedRankOffsets();
    CopyFromWindowByCachedMeta();
    // The cached metadata path ends with MTE3_MTE2 for UB reuse.  Add an explicit producer-completion edge for
    // the following combine kernel, which consumes all five metadata fields and recv_rank_offsets.
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_OUTPUT_DONE);
    MoeEpCompletion::DrainChannels(mc2Context_, epWorldSize_, tpipe_);
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::CopyCachedRankOffsets()
{
    if (aivId_ != 0U) {
        return;
    }

    DataCopyExtParams offsetsCopyParams{1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> offsetsPadParams{false, 0U, 0U, 0};
    DataCopyPad(ubRankOffsets_, cachedRecvRankOffsetsGm_, offsetsCopyParams, offsetsPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopyPad(recvRankOffsetsGm_, ubRankOffsets_, offsetsCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

// 搬运总入口：先认领行区间，再按 CACHED_META_TILE 行一拍地过。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::CopyFromWindowByCachedMeta()
{
    uint32_t startRow = 0U;
    uint32_t endRow = 0U;
    if (!ClaimCachedRows(startRow, endRow)) {
        return;
    }
    InitCachedBuffers();
    for (uint32_t tileStart = startRow; tileStart < endRow;) {
        const uint32_t tileCnt = (endRow - tileStart > CACHED_META_TILE) ? CACHED_META_TILE : (endRow - tileStart);
        CopyCachedTile(tileStart, tileCnt);
        tileStart += tileCnt;
    }
}

// 本核负责的 cached metadata 行区间（全局行号）。返回 false 表示本核没活干。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline bool MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::ClaimCachedRows(
    uint32_t &startRow, uint32_t &endRow)
{
    if (expertSum_ == 0) {
        return false;
    }

    uint32_t rowCnt = 0;
    SplitToCore(expertSum_, aivNum_, startRow, endRow, rowCnt);
    return startRow < endRow;
}

// cached 路径的 UB 是有活干才要，所以这份分配压在认领行区间之后，不放进 Init。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::InitCachedBuffers()
{
    uint32_t ubMetaBytes = Ceil(metaBytes_, UB_ALIGN) * UB_ALIGN;
    // Cached metadata is rank ordered, while weights remain recv_x ordered. A single aligned staging block is enough
    // for the optional per-row scatter and avoids reserving one padded block per metadata row.
    uint32_t ubStageWeightsBytes = UB_ALIGN;
    uint32_t ubStageMetaBytes =
        Ceil((uint32_t)(CACHED_META_TILE * RECV_META_FIELDS * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(ubMetaBuf_, ubMetaBytes);
    tpipe_->InitBuffer(ubStageWeightsBuf_, ubStageWeightsBytes);
    tpipe_->InitBuffer(ubStageMetaBuf_, ubStageMetaBytes);
    ubMeta_ = ubMetaBuf_.Get<int32_t>();
    ubStageWeights_ = ubStageWeightsBuf_.Get<float>();
    ubStageMeta_ = ubStageMetaBuf_.Get<int32_t>();
}

// 一拍：把这段 metadata 读进来，逐行搬 token/scales/weights，最后把这段 metadata 原样写出去。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::CopyCachedTile(
    uint32_t tileStart, uint32_t tileCnt)
{
    LoadCachedMetaTile(tileStart, tileCnt);
    for (uint32_t i = 0; i < tileCnt; ++i) {
        CopyCachedRow(i);
    }
    FlushCachedMetaTile(tileStart, tileCnt);
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::LoadCachedMetaTile(
    uint32_t tileStart, uint32_t tileCnt)
{
    DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
    DataCopyExtParams metaInParams{1U, static_cast<uint32_t>(tileCnt * RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(ubStageMeta_, cachedRecvSrcMetadataGm_[(int64_t)tileStart * RECV_META_FIELDS], metaInParams,
                metaPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

// 逐行搬完之后，把这段 metadata 落到输出的 recv_src_metadata（源 buffer 是上一轮 dispatch 留下的另一块）。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::FlushCachedMetaTile(
    uint32_t tileStart, uint32_t tileCnt)
{
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopyExtParams metaOutParams{1U, static_cast<uint32_t>(tileCnt * RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U,
                                    0U};
    DataCopyPad(recvSrcMetadataGm_[(int64_t)tileStart * RECV_META_FIELDS], ubStageMeta_, metaOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
}

// 一行：读 metadata → 发读 → 落 recv_x/scales → 落 weight。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::CopyCachedRow(uint32_t rowInTile)
{
    CachedRow row;
    if (!ReadCachedRow(rowInTile, row)) {
        return;
    }
    LocalTensor<XType> tokenTensor = LoadCachedToken(row);
    if constexpr (HasTopkWeights) {
        LoadCachedWeight(row);
    }
    // 队列握手的 EnQue 必须压在 weight 那段读之后，DeQue 在下面 StoreCachedToken 的开头配平。
    tokenQueue_.EnQue(tokenTensor);
    StoreCachedToken(row);
    if constexpr (HasTopkWeights) {
        StoreCachedWeight(row);
    }
}

// 取一行的槽地址与目标行号。recvXIdx 是上一轮非 cached Epilogue 产出的，拿它当 recv_x 行号之前必须先
// 夹范围：cached 句柄是用户可见的，畸形 metadata 不能变成一次越界的 GM 写。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline bool MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::ReadCachedRow(
    uint32_t rowInTile, CachedRow &row)
{
    uint32_t metaBase = rowInTile * RECV_META_FIELDS;
    row.srcRankId = ubStageMeta_.GetValue(metaBase + META_SRC_RANK_OFFSET);
    row.tokenIdx = ubStageMeta_.GetValue(metaBase + META_TOKEN_IDX_OFFSET);
    row.srcTopkIdx = ubStageMeta_.GetValue(metaBase + META_TOPK_IDX_OFFSET);
    int32_t slotIdx = ubStageMeta_.GetValue(metaBase + META_SLOT_IDX_OFFSET);
    int32_t recvXIdx = ubStageMeta_.GetValue(metaBase + META_RECV_X_IDX_OFFSET);
    if (recvXIdx < 0 || recvXIdx >= static_cast<int32_t>(expertSum_)) {
        return false;
    }
    row.recvXRow = static_cast<uint32_t>(recvXIdx);
    row.slotAddr = localWinAddr_ + (int64_t)row.srcRankId * numMaxTokensPerRank_ * perSlotBytes_ +
                   (int64_t)slotIdx * perSlotBytes_;
    return true;
}

// token/scales 的读。direct 自段 x 未入窗口（dispatch 已省略本端拷贝），x 直读输入；hybrid 保持读窗口。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline LocalTensor<XType> MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::LoadCachedToken(
    const CachedRow &row)
{
    DataCopyParams tokenCopyParams{1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyPadParams tokenPadParams{false, 0, 0, 0};
    bool directSelfRank = (networkMode_ != NETWORK_HYBRID) && (row.srcRankId == static_cast<int32_t>(epRankId_));
    GlobalTensor<XType> srcTokenTensor;
    if (directSelfRank) {
        srcTokenTensor = xGm_[static_cast<int64_t>(row.tokenIdx) * axisH_];
    } else {
        srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(row.slotAddr), axisH_);
    }
    LocalTensor<XType> tokenTensor = tokenQueue_.AllocTensor<XType>();
    DataCopyPad(tokenTensor, srcTokenTensor, tokenCopyParams, tokenPadParams);
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        GlobalTensor<ScalesType> srcScalesTensor;
        srcScalesTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScalesType *>(row.slotAddr + scalesOffset_),
                                        scalesElems_);
        DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
        DataCopyPadParams scalesPadParams{false, 0, 0, 0};
        DataCopyPad(tokenTensor[scalesStride_].template ReinterpretCast<ScalesType>(), srcScalesTensor, scalesCopyParams,
                    scalesPadParams);
    }
    return tokenTensor;
}

// 本行的 topk weight 在槽 meta 里，所以要单独读一小段槽 meta。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::LoadCachedWeight(
    const CachedRow &row)
{
    DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
    GlobalTensor<int32_t> srcMetaGm;
    srcMetaGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(row.slotAddr + metaOffset_));
    DataCopyExtParams slotMetaParams{1U, metaBytes_, 0U, 0U, 0U};
    DataCopyPad(ubMeta_, srcMetaGm, slotMetaParams, metaPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    float weight = ubMeta_.ReinterpretCast<float>().GetValue(axisKAlign_ + row.srcTopkIdx);
    ubStageWeights_.SetValue(0, weight);
    SyncFunc<AscendC::HardEvent::S_MTE2>();
}

// recv_x（fp8 含 scales）的写。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::StoreCachedToken(
    const CachedRow &row)
{
    DataCopyParams tokenCopyParams{1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    LocalTensor<XType> tokenOut = tokenQueue_.DeQue<XType>();
    DataCopyPad(recvXGm_[(int64_t)row.recvXRow * axisH_], tokenOut, tokenCopyParams);
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
        DataCopyPad(recvScalesGm_[(int64_t)row.recvXRow * scalesElems_],
                    tokenOut[scalesStride_].template ReinterpretCast<ScalesType>(), scalesCopyParams);
    }
    tokenQueue_.FreeTensor(tokenOut);
}

// weight 走一块全核共用的 UB 暂存，所以写完要等 MTE3 落地，下一行才能复用这块暂存。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueCached<XType, ScalesType, HasTopkWeights>::StoreCachedWeight(
    const CachedRow &row)
{
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyExtParams weightOutParams{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(recvTopkWeightsGm_[row.recvXRow], ubStageWeights_, weightOutParams);
    // Wait before reusing the single aligned staging block for the next metadata row.
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

#endif

} // namespace MoeEpDispatchEpilogueImpl

#endif // MOE_EP_DISPATCH_EPILOGUE_CACHED_H
