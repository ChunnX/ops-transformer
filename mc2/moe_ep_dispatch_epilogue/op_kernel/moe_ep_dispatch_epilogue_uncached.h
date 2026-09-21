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
 * \file moe_ep_dispatch_epilogue_uncached.h
 * \brief 非 cached 路径：逐核扫窗区数命中 + workspace 全核规约，按 slot 切片分核，按专家重排写出
 *        recv_x / metadata / weights。
 *
 * 【隔离实验分支】本文件保留 b7c8b67ed 的三头文件拆分与细粒度函数结构，但把分核策略恢复成
 * 重构之前那一版（slot 切片 + joint count 规约），用来单独度量「分核策略」和「重构」各自的影响。
 * 逐条指令的先后顺序与原 moe_ep_dispatch_epilogue.h 保持一致，调用顺序就是流水顺序。
 *
 * 公共状态、Init 前半段、WaitDispatch 与工具函数都在 moe_ep_dispatch_epilogue_base.h；
 * 本文件只有这条路径的成员与流水。
 */

#ifndef MOE_EP_DISPATCH_EPILOGUE_UNCACHED_H
#define MOE_EP_DISPATCH_EPILOGUE_UNCACHED_H

#include "moe_ep_dispatch_epilogue_base.h"

namespace MoeEpDispatchEpilogueImpl {

#if defined(ENABLE_MOE_EP_DISPATCH_EPILOGUE_KERNEL)

using namespace AscendC;

// 本路径专有的常量（公共量见 base）。
static constexpr uint32_t META_EXTRA_FIELDS = 2U;
static constexpr uint32_t HIT_ROW_OFFSET = 0U;
static constexpr uint32_t HIT_TOPK_OFFSET = 1U;
static constexpr uint32_t HIT_ENTRY_SIZE = 2U;
static constexpr uint32_t SLOTS_TILE = 128U;

// 本路径上三处流水缓冲的份数。都是人工调的旋钮：调大 → 重叠更深、更抗上游抖动；调小 → 省 UB。
// 三处互相独立，可以单独调。都是「全局拍号 % 份数」选 buffer，所以份数改动不需要动任何下标表达式。
//   META_RING  : tile 元数据的 MTE2 → S 环（本轮不整拍做向量比较，所以落点是标量 pipe）。
//   TOKEN_RING : token/scales 的 MTE2 → MTE3 环，让这一拍的读盖在上一拍的写里。
//   STAGE_RING : stage meta/weights 的 S → MTE3 环。份数 >= 3 之后，同一份的下一次使用恒在 STAGE_RING
//                拍之后，稳态的 per-slot 等待就顶掉了原来 tile 尾那次「等整条 MTE3 排空」。
static constexpr uint32_t META_RING = 2U;
static constexpr uint32_t TOKEN_RING = 3U;
static constexpr uint32_t STAGE_RING = 3U;

// 预读扫描的起点哨兵：表示「还没发过任何一拍的 meta」，从第一个有活的 rank 的第一拍开始找。
static constexpr uint32_t TILE_NONE = 0xFFFFFFFFU;

template <typename XType, typename ScalesType, bool HasTopkWeights>
class MoeEpDispatchEpilogueUncached : public MoeEpDispatchEpilogueBase<XType, ScalesType, HasTopkWeights> {
public:
    __aicore__ inline MoeEpDispatchEpilogueUncached(){};
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

    // ---------------- 行号与命中数：进搬运前的一次性准备 ----------------
    // 每个核都完整扫一遍窗口里所有有活的 slot，把自己负责的那一段数出来，再落 workspace 做全核规约。
    __aicore__ inline void ComputePrefixSums();
    __aicore__ inline void CountHits();
    __aicore__ inline void ScanRankHitTiles(uint32_t rankId, uint32_t slotStart, uint32_t slotCntPerAiv);
    __aicore__ inline void CountTileHits(uint32_t rankId, uint32_t tileCnt);
    // 全核规约出 joint count，再各自折算 metadata / recv_x 的行号游标。
    __aicore__ inline void BuildRankRowStarts();

    // 搬运主循环的游标与拍号。原来是 CopyFromWindowByExpert 里的一堆局部量，拆成一堆小函数之后提到成员，
    // 否则每个函数都得拖一串引用参数。PrepareCopy 会先 Reset，所以下面这些初值只是兜底。
    struct TileCursor {
        uint32_t rankId{0};
        uint32_t tileStart{0};
        uint32_t tileCnt{0};
        uint32_t slotStart{0}; // 本核在这个 rank 上分到的 slot 区间起点
        uint32_t nextRank{0};
        uint32_t nextTileStart{0};
        uint32_t nextTileCnt{0};
        uint32_t nextSlotStart{0};
        uint32_t metaSeq{0};
        uint32_t metaBuf{0};
        uint32_t metaRingBase{0};
        uint32_t tokenSeq{0};
        uint32_t stageSeq{0};
        uint32_t stageBuf{0};
        GM_ADDR srcRankBase{nullptr};
        bool hasTile{false};
        bool hasNext{false};

        __aicore__ inline void Reset()
        {
            rankId = 0U;
            tileStart = 0U;
            tileCnt = 0U;
            slotStart = 0U;
            nextRank = 0U;
            nextTileStart = 0U;
            nextTileCnt = 0U;
            nextSlotStart = 0U;
            metaSeq = 0U;
            metaBuf = 0U;
            metaRingBase = 0U;
            tokenSeq = 0U;
            stageSeq = 0U;
            stageBuf = 0U;
            srcRankBase = nullptr;
            hasTile = false;
            hasNext = false;
        }
    };

    // 一个槽在搬运期间的标量上下文。这些量都是「读一次、后面几个循环反复用」，攒在一起是为了让每个小函数
    // 只接一个引用，而不是六七个下标。
    struct SlotHeader {
        uint32_t metaBase{0};    // 本槽 meta 在 ubMeta_ 里的起点
        uint32_t slotIdx{0};     // 本槽在源 rank 内的槽位号（slotStart + tileStart + localSlot），写进 metadata
        uint32_t tokenBuf{0};    // 本槽占的 token 环份号
        int32_t srcRankMeta{0};  // 发端 rank
        int32_t tokenIdxMeta{0}; // 发端 token 号
    };

    // ---------------- 拍级：一个 SLOTS_TILE 大小的槽位窗口 ----------------
    // 把 (rank, tile) 两层循环拍平成「取下一个 tile」。预读 meta 必须知道下一拍在哪，而下一拍会跨 rank，
    // 每个 rank 的槽位数又只有标量读得到；epWorldSize_ 很小，逐 rank 扫一遍的代价可以忽略。
    // slot 仍是按核切片的，所以这里要把本核的 slotStart 一并带出来。
    __aicore__ inline bool NextTile(uint32_t rankId, uint32_t tileStart, uint32_t &nextRank, uint32_t &nextTileStart,
                                    uint32_t &nextTileCnt, uint32_t &nextSlotStart);
    __aicore__ inline void IssueMeta(uint32_t rankId, uint32_t slotIdx, uint32_t tileCnt, uint32_t buf);
    __aicore__ inline void PrepareCopy(); // 常量与游标复位，发第一拍的 meta 读
    __aicore__ inline void BeginTile();   // 等本拍 meta、预读下一拍、换 rank 时更新直读判断
    __aicore__ inline void EndTile();     // 归还本拍 meta、把游标推到下一拍

    // ---------------- 槽级：一拍里的一个槽 ----------------
    // 一个槽的搬运拆成七步，每步一个函数：开槽 → 发读 → 收集 → 等读 → 占 stage → 发写 → 落 metadata。
    // 调用顺序就是流水顺序，中间不允许再插别的指令，同步指令都留在各自那一步里。
    __aicore__ inline void OpenSlot(uint32_t localSlot, SlotHeader &slot);
    __aicore__ inline void IssueTokenLoad(const SlotHeader &slot);
    __aicore__ inline uint32_t CollectHits(const SlotHeader &slot);
    __aicore__ inline void AwaitTokenLoad(const SlotHeader &slot);
    __aicore__ inline void ReleaseTokenBuf(const SlotHeader &slot); // 无命中时直接把这一份还给 MTE2
    __aicore__ inline uint32_t AcquireStageBuf();
    __aicore__ inline void FillRecvAndStage(const SlotHeader &slot, uint32_t hitCnt, uint32_t stageOff);
    __aicore__ inline void FlushStage(const SlotHeader &slot, uint32_t hitCnt, uint32_t stageOff);
    __aicore__ inline void CopySlot(uint32_t localSlot);
    __aicore__ inline void CopyFromWindowByExpert();

    bool isDirectSelfRank_{false}; // 当前遍历的源 rank 是否为 direct 本端段
    GlobalTensor<int32_t> numRecvPerRankGm_;
    // joint 计数矩阵占本 op 用户 workspace 的最前面一段，所以不需要偏移量，直接绑 workspace 首地址。
    GlobalTensor<int32_t> rankExpertHitCountGm_;

    LocalTensor<int32_t> ubRecvCnt_;
    LocalTensor<int32_t> ubHitCount_;
    LocalTensor<int32_t> ubRankExpertHitCount_;
    LocalTensor<int32_t> ubRankExpertRowStart_;
    LocalTensor<int64_t> ubRowStart_;
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<int32_t> ubTopkIds_;
    LocalTensor<int32_t> ubTargetExpertId_;
    LocalTensor<int64_t> ubHitCountRowI64_;
    LocalTensor<float> ubStageWeightsRing_;
    LocalTensor<int32_t> ubStageMetaRing_;
    LocalTensor<XType> tokenRing_;
    LocalTensor<int64_t> ubHitList_;
    LocalTensor<int32_t> ubLocalCursor_;

    TBuf<QuePosition::VECIN> ubRecvCntBuf_;
    TBuf<QuePosition::VECIN> ubHitCountBuf_;
    TBuf<QuePosition::VECIN> ubRankExpertHitCountBuf_;
    TBuf<QuePosition::VECIN> ubRankExpertRowStartBuf_;
    TBuf<QuePosition::VECIN> ubRowStartBuf_;
    TBuf<QuePosition::VECIN> ubMetaBuf_;
    TBuf<QuePosition::VECIN> ubTopkIdsBuf_;
    TBuf<QuePosition::VECIN> ubTargetExpertIdBuf_;
    TBuf<QuePosition::VECIN> ubHitCountRowI64Buf_;
    TBuf<QuePosition::VECIN> ubLocalCursorBuf_;
    TBuf<QuePosition::VECIN> ubHitListBuf_;
    // 流水缓冲全部自管，份数见 *_RING 常量。ubMetaBuf_ 同时也是 meta 环的载体。
    TBuf<QuePosition::VECIN> tokenRingBuf_;
    TBuf<QuePosition::VECIN> ubStageWeightsRingBuf_;
    TBuf<QuePosition::VECIN> ubStageMetaRingBuf_;

    // 三处自管环的「一份」有多少个元素，用来做 份号 * 每份元素数 的偏移。
    uint32_t tokenRingElems_{0};
    uint32_t stageSlotElems_{0};
    uint32_t metaRingElems_{0};
    uint32_t paddedMetaElems_{0};
    uint32_t paddedTopkElems_{0};
    // joint 计数矩阵的行 / 整表长度。必须补到 ELEM_ALIGN 的整数倍：消费者会按 rank * rankExpertRowStride_
    // 切片，而 VEC 访问 UB 要求 32 字节对齐；numLocalExperts_ 本身没有对齐保证。
    uint32_t rankExpertRowStride_{0};
    uint32_t rankExpertCountStride_{0};
    // 搬运主循环的游标与拍号，字段见 TileCursor。
    TileCursor cur_;
    // 三处自管环的事件，份数各自等于对应 *_RING。全部走 AllocEventID 拿真份数：FetchEventID 不置占用位、
    // 恒返回同一个下标，一份以上的环会退化成一条 id，等待就分不出是哪一份。
    int32_t metaEvtMte2ToS_[META_RING] = {0};
    int32_t metaEvtSToMte2_[META_RING] = {0};
    int32_t tokenEvtFill_[TOKEN_RING] = {0}; // MTE2_MTE3
    int32_t tokenEvtFree_[TOKEN_RING] = {0}; // MTE3_MTE2
    int32_t stageEvtMte3ToS_[STAGE_RING] = {0};
    // stage 的 S → MTE3 握手是 Set/Wait 紧邻的，任何时刻只有一发在飞，一条 id 就够。
    int32_t stageEvtSToMte3_{0};
};

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
    GM_ADDR cachedRecvSrcMetadata, GM_ADDR recvX, GM_ADDR recvSrcMetadata, GM_ADDR recvTopkWeights, GM_ADDR recvScales,
    GM_ADDR workspace, GM_ADDR tilingGM, TPipe *pipe, const MoeEpDispatchEpilogueInfo *tilingData)
{
    InitCommon(context, x, numRecvPerExpert, recvX, recvSrcMetadata, recvTopkWeights, recvScales, workspace, tilingGM,
               pipe, tilingData);

    // topkIdx / cachedRecvSrcMetadata 只有 cached 路径用得到，本路径不绑。
    numRecvPerRankGm_.SetGlobalBuffer((__gm__ int32_t *)numRecvPerRank);
    rankExpertHitCountGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace));

    rankExpertRowStride_ = Ceil(numLocalExperts_, ELEM_ALIGN) * ELEM_ALIGN;
    rankExpertCountStride_ = rankExpertRowStride_ * epWorldSize_;
    paddedMetaElems_ = Ceil(META_TOPK_SECTION * axisKAlign_ + META_EXTRA_FIELDS, ELEM_ALIGN) * ELEM_ALIGN;
    paddedTopkElems_ = axisKAlign_;
    uint32_t ubRecvCntBytes = Ceil((uint32_t)(epWorldSize_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubHitCountBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubRankExpertCountBytes = rankExpertCountStride_ * static_cast<uint32_t>(sizeof(int32_t));
    uint32_t ubRowStartBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubMetaBytes = Ceil((uint32_t)(SLOTS_TILE * paddedMetaElems_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubTopkIdsBytes = Ceil((uint32_t)(SLOTS_TILE * paddedTopkElems_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubHitCountRowI64Bytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
    // 一份 stage:最多 axisK_ 个命中，每个占 ELEM_ALIGN 个元素。份数见 STAGE_RING。
    stageSlotElems_ = axisK_ * ELEM_ALIGN;
    uint32_t ubStageSlotBytes = stageSlotElems_ * static_cast<uint32_t>(sizeof(int32_t));
    uint32_t ubStageMetaBytes = ubStageSlotBytes * STAGE_RING;
    uint32_t ubStageWeightsBytes = ubStageSlotBytes * STAGE_RING;
    metaRingElems_ = ubMetaBytes / static_cast<uint32_t>(sizeof(int32_t));
    uint32_t ubHitListBytes = Ceil((uint32_t)(axisK_ * HIT_ENTRY_SIZE * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubLocalCursorBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;

    tpipe_->InitBuffer(ubRecvCntBuf_, ubRecvCntBytes);
    tpipe_->InitBuffer(ubHitCountBuf_, ubHitCountBytes);
    tpipe_->InitBuffer(ubRankExpertHitCountBuf_, ubRankExpertCountBytes);
    tpipe_->InitBuffer(ubRankExpertRowStartBuf_, ubRankExpertCountBytes);
    tpipe_->InitBuffer(ubRowStartBuf_, ubRowStartBytes);
    tpipe_->InitBuffer(ubMetaBuf_, ubMetaBytes * META_RING);
    tpipe_->InitBuffer(ubTopkIdsBuf_, ubTopkIdsBytes);
    tpipe_->InitBuffer(ubTargetExpertIdBuf_, ubTopkIdsBytes);
    tpipe_->InitBuffer(ubHitCountRowI64Buf_, ubHitCountRowI64Bytes);
    tpipe_->InitBuffer(ubStageWeightsRingBuf_, ubStageWeightsBytes);
    tpipe_->InitBuffer(ubStageMetaRingBuf_, ubStageMetaBytes);
    tpipe_->InitBuffer(ubHitListBuf_, ubHitListBytes);
    tpipe_->InitBuffer(ubLocalCursorBuf_, ubLocalCursorBytes);
    ubRecvCnt_ = ubRecvCntBuf_.Get<int32_t>();
    ubHitCount_ = ubHitCountBuf_.Get<int32_t>();
    ubRankExpertHitCount_ = ubRankExpertHitCountBuf_.Get<int32_t>();
    ubRankExpertRowStart_ = ubRankExpertRowStartBuf_.Get<int32_t>();
    ubRowStart_ = ubRowStartBuf_.Get<int64_t>();
    ubMeta_ = ubMetaBuf_.Get<int32_t>();
    ubTopkIds_ = ubTopkIdsBuf_.Get<int32_t>();
    ubTargetExpertId_ = ubTargetExpertIdBuf_.Get<int32_t>();
    ubHitCountRowI64_ = ubHitCountRowI64Buf_.Get<int64_t>();
    ubStageWeightsRing_ = ubStageWeightsRingBuf_.Get<float>();
    ubStageMetaRing_ = ubStageMetaRingBuf_.Get<int32_t>();
    ubHitList_ = ubHitListBuf_.Get<int64_t>();
    ubLocalCursor_ = ubLocalCursorBuf_.Get<int32_t>();
    // 三处环的事件各按份数取，互相不复用。
    for (uint32_t ringIdx = 0; ringIdx < META_RING; ++ringIdx) {
        metaEvtMte2ToS_[ringIdx] = static_cast<int32_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_S>());
        metaEvtSToMte2_[ringIdx] = static_cast<int32_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::S_MTE2>());
    }
    for (uint32_t ringIdx = 0; ringIdx < TOKEN_RING; ++ringIdx) {
        tokenEvtFill_[ringIdx] = static_cast<int32_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
        tokenEvtFree_[ringIdx] = static_cast<int32_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    }
    for (uint32_t ringIdx = 0; ringIdx < STAGE_RING; ++ringIdx) {
        stageEvtMte3ToS_[ringIdx] = static_cast<int32_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_S>());
    }
    stageEvtSToMte3_ = static_cast<int32_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::S_MTE3>());

    // hAlignSize_ / scalesBytesAlign_ 由 InitCommon 算好（含 fp8 分支），这里只做拼装。
    tokenQueueBufBytes_ = hAlignSize_ + scalesBytesAlign_;
    tokenRingElems_ = tokenQueueBufBytes_ / static_cast<uint32_t>(sizeof(XType));
    tpipe_->InitBuffer(tokenRingBuf_, TOKEN_RING * tokenQueueBufBytes_);
    tokenRing_ = tokenRingBuf_.Get<XType>();

    FinalizeInit();
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::Process()
{
    ComputePrefixSums();
    WaitDispatch();
    SyncAll<true>();
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_WAIT_DONE);
    CountHits();
    // joint count 是全核各扫一段拼出来的，规约完之后每个核才能算出自己的 metadata / recv_x 行号。
    SyncAll<true>();

    // Every core builds its own [rank][expert] metadata cursors from the shared joint-count matrix.
    // The starts stay in local UB, so no post-build cross-core synchronization is needed.
    BuildRankRowStarts();

    uint32_t totalHits = 0;
    for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
        totalHits += static_cast<uint32_t>(ubHitCount_.GetValue(localExpertIdx));
    }

    if (totalHits != 0) {
        CopyFromWindowByExpert();
    }
    // recv_x/metadata/weights are consumed immediately by MoeEpCombine in continue mode.  Do not rely on the
    // diagnostic write below (or on kernel-tail draining) to complete preceding MTE3 writes.
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    diagWriter_.RunPosRecord(MOE_EP_DISPATCH_EPILOGUE_RUN_POS_OUTPUT_DONE);
    MoeEpCompletion::DrainChannels(mc2Context_, epWorldSize_, tpipe_);
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::ComputePrefixSums()
{
    int64_t cumulativeRowOffset = 0;
    for (uint32_t localExpertIdx = 0; localExpertIdx < numLocalExperts_; ++localExpertIdx) {
        int64_t expertTokenCnt = ubExpertPfx_.GetValue(localExpertIdx);
        ubExpertPfx_.SetValue(localExpertIdx, cumulativeRowOffset);
        cumulativeRowOffset += expertTokenCnt;
    }
    SyncFunc<AscendC::HardEvent::S_V>();
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::CountHits()
{
    DataCopyExtParams recvCntCopyParams{1U, static_cast<uint32_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> recvCntPadParams{false, 0U, 0U, 0};
    DataCopyPad(ubRecvCnt_, numRecvPerRankGm_, recvCntCopyParams, recvCntPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    Duplicate(ubHitCount_, (int32_t)0, numLocalExperts_);
    Duplicate(ubRankExpertHitCount_, (int32_t)0, rankExpertCountStride_);

    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        int32_t slotCnt = ubRecvCnt_.GetValue(rankId);
        if (slotCnt == 0) {
            continue;
        }

        uint32_t slotStart, slotEnd, slotCntPerAiv;
        SplitToCore(static_cast<uint32_t>(slotCnt), aivNum_, slotStart, slotEnd, slotCntPerAiv);
        if (slotStart >= slotEnd) {
            continue;
        }
        ScanRankHitTiles(rankId, slotStart, slotCntPerAiv);
    }

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    // 每核一份的 hitCount 矩阵不再落 GM：它是 CopyFromWindowByExpert 里唯一的使用者，而那处已经改成
    // 对 ubRankExpertHitCount_ 的 rank 维求和，不需要再绕一趟 GM。ubHitCount_ 本身还有用（本核累计值）。
    DataCopyExtParams jointCountCopyParams{1U, rankExpertCountStride_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U,
                                           0U};
    DataCopyPad(rankExpertHitCountGm_[(int64_t)aivId_ * rankExpertCountStride_], ubRankExpertHitCount_,
                jointCountCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

// 扫一个源 rank 上分给本核的那一段槽位，逐拍数命中。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::ScanRankHitTiles(
    uint32_t rankId, uint32_t slotStart, uint32_t slotCntPerAiv)
{
    GM_ADDR srcRankBase = localWinAddr_ + (int64_t)rankId * numMaxTokensPerRank_ * perSlotBytes_;
    GlobalTensor<int32_t> srcTopkIdsGm;
    uint32_t topkBytes = axisK_ * sizeof(int32_t);

    for (uint32_t tileStart = 0; tileStart < slotCntPerAiv; tileStart += SLOTS_TILE) {
        uint32_t tileCnt = (slotCntPerAiv - tileStart > SLOTS_TILE) ? SLOTS_TILE : (slotCntPerAiv - tileStart);

        srcTopkIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            srcRankBase + (int64_t)(slotStart + tileStart) * perSlotBytes_ + metaOffset_));
        DataCopyExtParams topkCopyParams{static_cast<uint16_t>(tileCnt), static_cast<uint32_t>(topkBytes),
                                         static_cast<int64_t>(perSlotBytes_ - topkBytes), 0, 0};
        DataCopyPadExtParams<int32_t> topkPadParams{true, 0, static_cast<uint8_t>(paddedTopkElems_ - axisK_), -1};
        DataCopyPad(ubTopkIds_, srcTopkIdsGm, topkCopyParams, topkPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();

        CountTileHits(rankId, tileCnt);
        SyncFunc<AscendC::HardEvent::V_MTE2>();
    }
}

// 一拍里对每个本端专家做一次 EQ 比较 + GatherMask，把命中数累加进本核的 hitCount 表。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::CountTileHits(
    uint32_t rankId, uint32_t tileCnt)
{
    int32_t calCnt = static_cast<int32_t>(tileCnt * paddedTopkElems_);
    constexpr int32_t CMP_ALIGN = ALIGNED_LEN_256 / sizeof(int32_t);
    int32_t calCntAlign = Ceil(calCnt, CMP_ALIGN) * CMP_ALIGN;
    // ubMetaBuf_ 在这一段里当共享临时区用（V 写掩码、V 读掩码），与 meta 环错开使用。
    LocalTensor<uint8_t> sharedTmpInt8 = ubMetaBuf_.Get<uint8_t>();
    LocalTensor<uint32_t> sharedTmpInt32 = ubMetaBuf_.Get<uint32_t>();

    for (uint32_t localExpertId = 0; localExpertId < numLocalExperts_; ++localExpertId) {
        int32_t targetExpertId = static_cast<int32_t>(epRankId_ * numLocalExperts_ + localExpertId);
        uint64_t rsvdCnt = 0;
        CompareScalar(sharedTmpInt8, ubTopkIds_, static_cast<int32_t>(targetExpertId), AscendC::CMPMODE::EQ,
                      calCntAlign);
        GatherMask(ubTargetExpertId_, ubTopkIds_, sharedTmpInt32, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t curExpertCnt = rsvdCnt;
        int32_t currentHits = ubHitCount_.GetValue(localExpertId);
        ubHitCount_.SetValue(localExpertId, currentHits + curExpertCnt);
        uint32_t rankExpertIndex = rankId * rankExpertRowStride_ + localExpertId;
        int32_t rankExpertHits = ubRankExpertHitCount_.GetValue(rankExpertIndex);
        ubRankExpertHitCount_.SetValue(rankExpertIndex, rankExpertHits + curExpertCnt);
    }
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::BuildRankRowStarts()
{
    // CountHits used ubRankExpertHitCount_ as an MTE3 source. Finish that read before reusing the buffer as the
    // per-rank prefix contributed by cores preceding this core.
    SyncFunc<AscendC::HardEvent::MTE3_V>();

    // H[c][r][e] counts expanded records. Reduce all cores and the cores before this AIV separately.
    // Flat rows are [rank][expert], padded only at the end of each core row.
    Duplicate(ubRankExpertRowStart_, (int32_t)0, rankExpertCountStride_);
    Duplicate(ubRankExpertHitCount_, (int32_t)0, rankExpertCountStride_);

    // Reuse metadata staging for complete core-row tiles; even the maximum 2048-expert row fits.
    uint32_t ubMetaBytes = Ceil((uint32_t)(SLOTS_TILE * paddedMetaElems_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t jointRowBytes = rankExpertCountStride_ * sizeof(int32_t);
    uint32_t coreRowsPerTile = ubMetaBytes / jointRowBytes;
    coreRowsPerTile = coreRowsPerTile > aivNum_ ? aivNum_ : coreRowsPerTile;

    DataCopyPadExtParams<int32_t> jointMatrixPadParams{false, 0U, 0U, 0};
    for (uint32_t coreBase = 0; coreBase < aivNum_; coreBase += coreRowsPerTile) {
        uint32_t rowsThisTile = (aivNum_ - coreBase > coreRowsPerTile) ? coreRowsPerTile : (aivNum_ - coreBase);
        DataCopyExtParams jointMatrixCopyParams{1U, rowsThisTile * jointRowBytes, 0U, 0U, 0U};
        DataCopyPad(ubMeta_, rankExpertHitCountGm_[(int64_t)coreBase * rankExpertCountStride_], jointMatrixCopyParams,
                    jointMatrixPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();

        for (uint32_t localCoreId = 0; localCoreId < rowsThisTile; ++localCoreId) {
            uint32_t coreId = coreBase + localCoreId;
            LocalTensor<int32_t> jointCountRow = ubMeta_[localCoreId * rankExpertCountStride_];
            Add(ubRankExpertRowStart_, ubRankExpertRowStart_, jointCountRow, rankExpertCountStride_);
            if (coreId < aivId_) {
                Add(ubRankExpertHitCount_, ubRankExpertHitCount_, jointCountRow, rankExpertCountStride_);
            }
        }
        SyncFunc<AscendC::HardEvent::V_MTE2>();
    }

    // P(c,r,e) = offsets[r] + sum_{e'<e,c'} H[c'][r][e'] + sum_{c'<c} H[c'][r][e].
    // recv_x stays expert/core/rank ordered, so (rank, expert, core, per-group cursor) is (rank, recv_x_idx) order.
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t rankPrefix = 0;
    for (uint32_t rankId = 0; rankId < epWorldSize_; ++rankId) {
        if (aivId_ == 0U) {
            ubRankOffsets_.SetValue(rankId, rankPrefix);
        }
        for (uint32_t expertId = 0; expertId < numLocalExperts_; ++expertId) {
            uint32_t rankExpertIndex = rankId * rankExpertRowStride_ + expertId;
            int32_t jointTotal = ubRankExpertRowStart_.GetValue(rankExpertIndex);
            int32_t corePrefix = ubRankExpertHitCount_.GetValue(rankExpertIndex);
            ubRankExpertRowStart_.SetValue(rankExpertIndex, rankPrefix + corePrefix);
            rankPrefix += jointTotal;
        }
    }
    if (aivId_ == 0U) {
        ubRankOffsets_.SetValue(epWorldSize_, rankPrefix);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams offsetsCopyParams{1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(int32_t)), 0U, 0U,
                                            0U};
        DataCopyPad(recvRankOffsetsGm_, ubRankOffsets_, offsetsCopyParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline bool MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::NextTile(
    uint32_t rankId, uint32_t tileStart, uint32_t &nextRank, uint32_t &nextTileStart, uint32_t &nextTileCnt,
    uint32_t &nextSlotStart)
{
    for (uint32_t r = rankId; r < epWorldSize_; ++r) {
        int32_t slotCnt = ubRecvCnt_.GetValue(r);
        if (slotCnt == 0) {
            continue;
        }
        uint32_t slotStart, slotEnd, slotCntPerAiv;
        SplitToCore(static_cast<uint32_t>(slotCnt), aivNum_, slotStart, slotEnd, slotCntPerAiv);
        if (slotStart >= slotEnd) {
            continue;
        }
        // 同一个 rank 内接着上一拍往后走；跨到下一个 rank 就从它的第 0 拍开始。TILE_NONE 表示还没发过任何一拍。
        uint32_t start = (r == rankId && tileStart != TILE_NONE) ? (tileStart + SLOTS_TILE) : 0U;
        if (start >= slotCntPerAiv) {
            continue;
        }
        nextRank = r;
        nextTileStart = start;
        nextTileCnt = (slotCntPerAiv - start > SLOTS_TILE) ? SLOTS_TILE : (slotCntPerAiv - start);
        nextSlotStart = slotStart;
        return true;
    }
    return false;
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::IssueMeta(uint32_t rankId,
                                                                                                   uint32_t slotIdx,
                                                                                                   uint32_t tileCnt,
                                                                                                   uint32_t buf)
{
    DataCopyExtParams metaCopyParams{static_cast<uint16_t>(tileCnt), metaBytes_, perSlotBytes_ - metaBytes_, 0, 0};
    DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
    GlobalTensor<int32_t> srcMetaGm;
    srcMetaGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(localWinAddr_ + (int64_t)rankId * numMaxTokensPerRank_ * perSlotBytes_ +
                                           (int64_t)slotIdx * perSlotBytes_ + metaOffset_));
    DataCopyPad(ubMeta_[buf * metaRingElems_], srcMetaGm, metaCopyParams, metaPadParams);
    SetFlag<AscendC::HardEvent::MTE2_S>(metaEvtMte2ToS_[buf]);
}

// 搬运主循环。三段一拍：进这一拍（等 meta、预读下一拍）、处理这一拍（逐槽搬）、出这一拍（归还 meta、推游标）。
// 三段的调用顺序就是流水顺序，中间不允许插别的指令。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::CopyFromWindowByExpert()
{
    PrepareCopy();
    while (cur_.hasTile) {
        BeginTile();
        for (uint32_t localSlot = 0; localSlot < cur_.tileCnt; ++localSlot) {
            CopySlot(localSlot);
        }
        EndTile();
    }
    // 这里不再做 tile 尾的 MTE3 排空：Caller(Process) 紧接着就有 SyncFunc<MTE3_S> 兜住 recv_x/metadata/weights
    // 的可见性，环的事件又是 AllocEventID 私有、不会串到别处，所以函数返回时的 pipe 状态不需要在这里再等一次。
}

// 进搬运前的准备：算行号游标、复位拍号，并把第一拍的 meta 读发出去。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::PrepareCopy()
{
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    // recv_x 段起点 = 专家前缀 + 本核之前所有核在该专家上的命中数。
    // BuildRankRowStarts 已经把后者按 [rank][expert] 存进 ubRankExpertHitCount_，而且只累加了 coreId < aivId_
    // 的核，所以对 rank 求和恰好就是「本核之前所有核在该专家上的命中数」：
    //     sum_r sum_{c<aivId_} H[c][r][e]  ==  sum_{c<aivId_} H[c][e]
    // 原来那个循环是 aivId_ 次串行 DMA（每次先把某核的 hitCount 从 GM 搬回来）、每次夹两个 pipe 同步，
    // 是 CopyExpert 段里最长的一段串行标量；现在换成 epWorldSize_ 次纯向量 Add，标量只剩发指令。
    Duplicate(ubHitCount_, (int32_t)0, numLocalExperts_);
    for (uint32_t prefixRank = 0; prefixRank < epWorldSize_; ++prefixRank) {
        Add(ubHitCount_, ubHitCount_, ubRankExpertHitCount_[prefixRank * rankExpertRowStride_], numLocalExperts_);
    }
    Cast(ubHitCountRowI64_, ubHitCount_, RoundMode::CAST_NONE, numLocalExperts_);
    Add(ubRowStart_, ubExpertPfx_, ubHitCountRowI64_, numLocalExperts_);

    Duplicate(ubLocalCursor_, (int32_t)0, numLocalExperts_);
    SyncFunc<AscendC::HardEvent::V_S>();

    // 拍平成一个「取下一个 tile」的循环。原来的 (rank, tile) 双层结构没法把 meta 读提前一拍发出去：
    // 预读必须知道下一拍在哪，而下一拍会跨 rank。
    cur_.Reset();
    cur_.hasTile = NextTile(0U, TILE_NONE, cur_.rankId, cur_.tileStart, cur_.tileCnt, cur_.slotStart);
    if (cur_.hasTile) { // 第一拍的 meta 读先发出去，它的延迟盖在下面 PrepareCopy 剩下的标量里
        IssueMeta(cur_.rankId, cur_.slotStart + cur_.tileStart, cur_.tileCnt, 0U);
    }
}

// 进这一拍：等本拍的 meta 到、把下一拍定位并预读、更新 direct 直读判断。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::BeginTile()
{
    // 当前拍的 meta 读是上一拍发出去的，这里只等它到。
    cur_.metaBuf = cur_.metaSeq % META_RING;
    WaitFlag<AscendC::HardEvent::MTE2_S>(metaEvtMte2ToS_[cur_.metaBuf]);
    ++cur_.metaSeq;
    // 先把下一拍定位并预读，再处理当前拍：读延迟盖在当前拍的 MTE3 写里，标量 pipe 不用等。
    cur_.hasNext = NextTile(cur_.rankId, cur_.tileStart, cur_.nextRank, cur_.nextTileStart, cur_.nextTileCnt,
                            cur_.nextSlotStart);
    if (cur_.hasNext) {
        const uint32_t nextMetaBuf = cur_.metaSeq % META_RING;
        if (cur_.metaSeq >= META_RING) { // 这一份上一次用是 META_RING 拍之前，先确认标量已经读完
            WaitFlag<AscendC::HardEvent::S_MTE2>(metaEvtSToMte2_[nextMetaBuf]);
        }
        IssueMeta(cur_.nextRank, cur_.nextSlotStart + cur_.nextTileStart, cur_.nextTileCnt, nextMetaBuf);
    }

    // direct 本端x 直读输入；hybrid 保持读窗口
    isDirectSelfRank_ = (networkMode_ != NETWORK_HYBRID) && (cur_.rankId == static_cast<uint32_t>(epRankId_));
    cur_.srcRankBase = localWinAddr_ + (int64_t)cur_.rankId * numMaxTokensPerRank_ * perSlotBytes_;
    cur_.metaRingBase = cur_.metaBuf * metaRingElems_;
}

// 出这一拍：归还本拍的 meta 份，把游标推到下一拍。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::EndTile()
{
    // 这一拍的 meta 已经被标量全部读完（上面所有 ubMeta_ 的 GetValue），可以还给 MTE2 了。
    // 只有还有下一拍时才发：这样 S_MTE2 上「每次 Wait 恰好一次 Set」，不攒多余信用。
    if (cur_.hasNext) {
        SetFlag<AscendC::HardEvent::S_MTE2>(metaEvtSToMte2_[cur_.metaBuf]);
    }
    cur_.rankId = cur_.nextRank;
    cur_.tileStart = cur_.nextTileStart;
    cur_.tileCnt = cur_.nextTileCnt;
    cur_.slotStart = cur_.nextSlotStart;
    cur_.hasTile = cur_.hasNext;
}

// 一个槽的搬运。七步的顺序就是流水顺序：读先发出去盖住下面的标量收集，写要等 stage 份空出来。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::CopySlot(uint32_t localSlot)
{
    SlotHeader slot;
    OpenSlot(localSlot, slot);
    IssueTokenLoad(slot);
    const uint32_t hitCnt = CollectHits(slot);
    AwaitTokenLoad(slot);
    if (hitCnt == 0) {
        // 没有消费者，这一拍的数据作废，直接把这份 buffer 还给 MTE2。
        ReleaseTokenBuf(slot);
        return;
    }
    const uint32_t stageOff = AcquireStageBuf();
    FillRecvAndStage(slot, hitCnt, stageOff);
    FlushStage(slot, hitCnt, stageOff);
}

// 开槽：拼出本槽在 ubMeta_ / 源 rank 里的位置，并读槽头。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::OpenSlot(uint32_t localSlot,
                                                                                                  SlotHeader &slot)
{
    slot.metaBase = cur_.metaRingBase + localSlot * paddedMetaElems_;
    slot.slotIdx = cur_.slotStart + cur_.tileStart + localSlot;
    slot.srcRankMeta = ubMeta_.GetValue(slot.metaBase + META_TOPK_SECTION * axisKAlign_);
    slot.tokenIdxMeta = ubMeta_.GetValue(slot.metaBase + META_TOPK_SECTION * axisKAlign_ + 1);
    slot.tokenBuf = cur_.tokenSeq % TOKEN_RING;
}

// 发出本槽 token/scales 的读。内容只取决于 slot，与命中集合无关，所以放在命中循环之前发：
// MTE2 这一拍的读和下面 loop A 的标量收集并行。原来 Alloc/读在 loop A 之后，MTE3 只能干等读完成。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::IssueTokenLoad(
    const SlotHeader &slot)
{
    DataCopyParams tokenCopyParams{1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyPadParams tokenPadParams{false, 0, 0, 0};

    const bool tokenReused = (cur_.tokenSeq >= TOKEN_RING);
    ++cur_.tokenSeq;
    LocalTensor<XType> tokenOut = tokenRing_[slot.tokenBuf * tokenRingElems_];
    if (tokenReused) { // 这一份上一次用于 TOKEN_RING 拍之前，等 MTE3 写完再覆盖
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tokenEvtFree_[slot.tokenBuf]);
    }
    GM_ADDR slotAddr = cur_.srcRankBase + (int64_t)slot.slotIdx * perSlotBytes_;
    GlobalTensor<XType> srcTokenTensor;
    if (isDirectSelfRank_) { // direct 自段 x 未入窗口，直读输入 x
        srcTokenTensor = xGm_[static_cast<int64_t>(slot.tokenIdxMeta) * axisH_];
    } else {
        srcTokenTensor.SetGlobalBuffer(reinterpret_cast<__gm__ XType *>(slotAddr), axisH_);
    }
    DataCopyPad(tokenOut, srcTokenTensor, tokenCopyParams, tokenPadParams);
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        GlobalTensor<ScalesType> srcScalesTensor;
        srcScalesTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ScalesType *>(slotAddr + scalesOffset_), scalesElems_);
        DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
        DataCopyPadParams scalesPadParams{false, 0, 0, 0};
        DataCopyPad(tokenOut[scalesStride_].template ReinterpretCast<ScalesType>(), srcScalesTensor, scalesCopyParams,
                    scalesPadParams);
    }
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(tokenEvtFill_[slot.tokenBuf]);
}

// loop A：逐 topk 位置判断专家是否落在本核区间，命中的记进 ubHitList_。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline uint32_t MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::CollectHits(
    const SlotHeader &slot)
{
    const int32_t rankExpertBase = static_cast<int32_t>(epRankId_ * numLocalExperts_);
    const int32_t rankExpertEnd = rankExpertBase + static_cast<int32_t>(numLocalExperts_);
    uint32_t hitCnt = 0;
    for (uint32_t topkIdx = 0; topkIdx < axisK_; ++topkIdx) {
        int32_t expertId = ubMeta_.GetValue(slot.metaBase + topkIdx);
        if (expertId < rankExpertBase || expertId >= rankExpertEnd) {
            continue;
        }
        uint32_t localExpertId = static_cast<uint32_t>(expertId - rankExpertBase);
        int64_t expertRowStart = ubRowStart_.GetValue(localExpertId);
        int32_t cursor = ubLocalCursor_.GetValue(localExpertId);
        ubLocalCursor_.SetValue(localExpertId, cursor + 1);
        int64_t recvXRow = expertRowStart + cursor;
        ubHitList_.SetValue(hitCnt * HIT_ENTRY_SIZE + HIT_ROW_OFFSET, recvXRow);
        ubHitList_.SetValue(hitCnt * HIT_ENTRY_SIZE + HIT_TOPK_OFFSET, static_cast<int64_t>(topkIdx));
        hitCnt++;
    }
    return hitCnt;
}

// 等本槽的读落地。命中与否都要等：MTE2_MTE3 上每个槽恰好一 Set 一 Wait 才配平，少一次 Wait 就会给
// 这一份攒下信用，让后面某次等待提前放行（MTE3 会读到还没写完的 buffer）。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::AwaitTokenLoad(
    const SlotHeader &slot)
{
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(tokenEvtFill_[slot.tokenBuf]);
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::ReleaseTokenBuf(
    const SlotHeader &slot)
{
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(tokenEvtFree_[slot.tokenBuf]);
}

// 占一份 stage，返回它在 ubStage*Ring_ 里的偏移。份数 STAGE_RING 保证同一份的下一次使用恒在 STAGE_RING
// 拍之后，所以这里稳态的这次等待就顶掉了原来 tile 尾那次「等整条 MTE3 排空」。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline uint32_t MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::AcquireStageBuf()
{
    cur_.stageBuf = cur_.stageSeq % STAGE_RING;
    if (cur_.stageSeq >= STAGE_RING) {
        WaitFlag<AscendC::HardEvent::MTE3_S>(stageEvtMte3ToS_[cur_.stageBuf]);
    }
    ++cur_.stageSeq;
    return cur_.stageBuf * stageSlotElems_;
}

// loop A'：把这一槽的 recv_x/scales 读发给 MTE3，同时把 stage 的 weights/metadata 填好。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::FillRecvAndStage(
    const SlotHeader &slot, uint32_t hitCnt, uint32_t stageOff)
{
    DataCopyParams tokenCopyParams{1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    LocalTensor<XType> tokenOut = tokenRing_[slot.tokenBuf * tokenRingElems_];
    // token 的 MTE3_MTE2 归还放在这个循环之后：recvXGm_/recvScalesGm_ 的 DataCopyPad 就是从 tokenOut 读的。
    for (uint32_t i = 0; i < hitCnt; i++) {
        int64_t recvXRow = ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_ROW_OFFSET);
        uint32_t topkIdx = static_cast<uint32_t>(ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_TOPK_OFFSET));

        DataCopyPad(recvXGm_[recvXRow * axisH_], tokenOut, tokenCopyParams);
        if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
            DataCopyParams scalesCopyParams{1U, static_cast<uint16_t>(scalesElems_ * sizeof(ScalesType)), 0U, 0U};
            DataCopyPad(recvScalesGm_[recvXRow * scalesElems_],
                        tokenOut[scalesStride_].template ReinterpretCast<ScalesType>(), scalesCopyParams);
        }

        if constexpr (HasTopkWeights) {
            float weights = ubMeta_.ReinterpretCast<float>().GetValue(slot.metaBase + axisKAlign_ + topkIdx);
            ubStageWeightsRing_.SetValue(stageOff + i * ELEM_ALIGN, weights);
        }
        ubStageMetaRing_.SetValue(stageOff + i * ELEM_ALIGN + META_SRC_RANK_OFFSET, slot.srcRankMeta);
        ubStageMetaRing_.SetValue(stageOff + i * ELEM_ALIGN + META_TOKEN_IDX_OFFSET, slot.tokenIdxMeta);
        ubStageMetaRing_.SetValue(stageOff + i * ELEM_ALIGN + META_TOPK_IDX_OFFSET, static_cast<int32_t>(topkIdx));
        ubStageMetaRing_.SetValue(stageOff + i * ELEM_ALIGN + META_SLOT_IDX_OFFSET, static_cast<int32_t>(slot.slotIdx));
        ubStageMetaRing_.SetValue(stageOff + i * ELEM_ALIGN + META_RECV_X_IDX_OFFSET,
                                  static_cast<int32_t>(recvXRow));
    }
    // 这一份的 MTE3 写已全部入队，MTE2 可以在 TOKEN_RING 拍后覆盖它。
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(tokenEvtFree_[slot.tokenBuf]);
}

// loop B：把 stage 里的 weights/metadata 落到 GM。stage 的 S → MTE3 握手（Set/Wait 紧邻，任何时刻只有一发
// 在飞，所以一条 id 就够）也在这里，因为标量写的就是下面要读的那些 buffer。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::FlushStage(
    const SlotHeader &slot, uint32_t hitCnt, uint32_t stageOff)
{
    DataCopyExtParams metaOutParams{1U, static_cast<uint32_t>(RECV_META_FIELDS * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyExtParams weightOutParams{1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    const int32_t rankExpertBase = static_cast<int32_t>(epRankId_ * numLocalExperts_);

    SetFlag<AscendC::HardEvent::S_MTE3>(stageEvtSToMte3_);
    WaitFlag<AscendC::HardEvent::S_MTE3>(stageEvtSToMte3_);
    // 从 ubStage*Ring_ 读，所以归还要压在这个循环的 SetFlag<MTE3_S> 之前。
    for (uint32_t i = 0; i < hitCnt; i++) {
        int64_t recvXRow = ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_ROW_OFFSET);
        if constexpr (HasTopkWeights) {
            DataCopyPad(recvTopkWeightsGm_[recvXRow], ubStageWeightsRing_[stageOff + i * ELEM_ALIGN], weightOutParams);
        }
        // Output metadata in source-row order without changing token/weight/scales placement.
        uint32_t topkIdx = static_cast<uint32_t>(ubHitList_.GetValue(i * HIT_ENTRY_SIZE + HIT_TOPK_OFFSET));
        uint32_t localExpertId = static_cast<uint32_t>(ubMeta_.GetValue(slot.metaBase + topkIdx) - rankExpertBase);
        uint32_t rankExpertIndex = cur_.rankId * rankExpertRowStride_ + localExpertId;
        int32_t metadataRow = ubRankExpertRowStart_.GetValue(rankExpertIndex);
        DataCopyPad(recvSrcMetadataGm_[(int64_t)metadataRow * RECV_META_FIELDS],
                    ubStageMetaRing_[stageOff + i * ELEM_ALIGN], metaOutParams);
        ubRankExpertRowStart_.SetValue(rankExpertIndex, metadataRow + 1);
    }
    SetFlag<AscendC::HardEvent::MTE3_S>(stageEvtMte3ToS_[cur_.stageBuf]);
}

#endif

} // namespace MoeEpDispatchEpilogueImpl

#endif // MOE_EP_DISPATCH_EPILOGUE_UNCACHED_H
