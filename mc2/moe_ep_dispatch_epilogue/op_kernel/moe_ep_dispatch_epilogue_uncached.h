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
 * \brief 非 cached 路径：读窗口里的 [源 rank][本端专家] 计数表分核，按专家重排写出 recv_x / metadata / weights。
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
// 整拍比较的元素数向上取整粒度。CompareScalar 按 256B 一个 repeat（int32 是 64 个元素）工作，而随后的
// And 是按字节数的，字节数得再满足 32B 对齐 —— 取 4 个 repeat（256 个元素 = 1024B = 32 个 32B 块），
// 两个约束一次到位。代价是每拍最多多比较 255 个元素，ubMeta_ 尾部留了同样多的余量。
static constexpr uint32_t CMP_ALIGN_ELEMS = 4U * ALIGNED_LEN_256 / sizeof(int32_t);
// 一个 uint32 掩码最多装 32 个 topk 位，而 host 侧 K_MAX = 32，两边必须一起改。
static constexpr uint32_t TOPK_MASK_BITS = 32U;

// 本路径上三处流水缓冲的份数。都是人工调的旋钮：调大 → 重叠更深、更抗上游抖动；调小 → 省 UB。
// 三处互相独立，可以单独调。都是「全局拍号 % 份数」选 buffer，所以份数改动不需要动任何下标表达式。
//   META_RING  : tile 元数据的 MTE2 → V → S 环。份数 >= 2 就能把下一个 tile 的 meta 读提前一拍发出去，
//                让读延迟盖在当前拍的 MTE3 写上，而不是让标量 pipe 干等。
//   TOKEN_RING : token/scales 的 MTE2 → MTE3 环，让这一拍的读盖在上一拍的写里。
//                与 meta 环不同，只有命中本核专家的槽位才会占用一份。
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
    __aicore__ inline void ComputePrefixSums();
    __aicore__ inline void CountHits();
    __aicore__ inline void BuildRankRowStarts();

    // 搬运主循环的游标与拍号。原来是 CopyFromWindowByExpert 里的一堆局部量，拆成一堆小函数之后提到成员，
    // 否则每个函数都得拖一串引用参数。PrepareCopy 会先 Reset，所以下面这些初值只是兜底。
    struct TileCursor {
        uint32_t rankId{0};
        uint32_t tileStart{0};
        uint32_t tileCnt{0};
        uint32_t nextRank{0};
        uint32_t nextTileStart{0};
        uint32_t nextTileCnt{0};
        uint32_t metaSeq{0};
        uint32_t metaBuf{0};
        uint32_t metaRingBase{0};
        uint32_t tokenSeq{0};
        uint32_t stageSeq{0};
        uint32_t stageBuf{0};
        // 游标是按 (源 rank, 专家) 计的，用哨兵表示「还没进过任何 rank」。
        uint32_t cursorRank{TILE_NONE};
        bool hasTile{false};
        bool hasNext{false};

        __aicore__ inline void Reset()
        {
            rankId = 0U;
            tileStart = 0U;
            tileCnt = 0U;
            nextRank = 0U;
            nextTileStart = 0U;
            nextTileCnt = 0U;
            metaSeq = 0U;
            metaBuf = 0U;
            metaRingBase = 0U;
            tokenSeq = 0U;
            stageSeq = 0U;
            stageBuf = 0U;
            cursorRank = TILE_NONE;
            hasTile = false;
            hasNext = false;
        }
    };

    // 一个槽在搬运期间的标量上下文。这些量都是「读一次、后面几个循环反复用」，攒在一起是为了让每个小函数
    // 只接一个引用，而不是六七个下标。
    struct SlotHeader {
        uint32_t metaBase{0};      // 本槽 meta 在 ubMeta_ 里的起点
        uint32_t slotIdx{0};       // 本槽在源 rank 内的槽位号（tileStart + localSlot），写进 metadata
        uint32_t hitMask{0};       // 本槽命中本核专家的 topk 位集
        uint32_t tokenBuf{0};      // 本槽占的 token 环份号
        int32_t srcRankMeta{0};    // 发端 rank
        int32_t tokenIdxMeta{0};   // 发端 token 号
    };

    // ---------------- 拍级：一个 SLOTS_TILE 大小的槽位窗口 ----------------
    // 把 (rank, tile) 两层循环拍平成「取下一个 tile」。预读 meta 必须知道下一拍在哪，而下一拍会跨 rank，
    // 每个 rank 的槽位数又只有标量读得到；epWorldSize_ 很小，逐 rank 扫一遍的代价可以忽略。
    // 槽位不再按核切片：本核负责的源 rank 的全部槽位都要过一遍，靠专家掩码过滤。
    __aicore__ inline bool NextTile(uint32_t rankId, uint32_t tileStart, uint32_t &nextRank, uint32_t &nextTileStart,
                                    uint32_t &nextTileCnt);
    __aicore__ inline void IssueMeta(uint32_t rankId, uint32_t tileStart, uint32_t tileCnt, uint32_t buf);
    // 整拍一次性判出「哪些 (槽, topk) 命中本核的专家」。边界随 rank 变，所以放在拍内算。
    __aicore__ inline void BuildTileHitMask(uint32_t metaRingBase, uint32_t tileCnt);
    __aicore__ inline void PrepareCopy();                     // 常量与游标复位，发第一拍的 meta 读
    __aicore__ inline void BeginTile();                       // 等本拍 meta、预读下一拍、跨 rank 复位游标
    __aicore__ inline void EndTile();                         // 归还本拍 meta、把游标推到下一拍
    __aicore__ inline void ResetExpertCursors(uint32_t rankId); // 换源 rank 时清 recv_x 游标

    // ---------------- 槽级：一拍里的一个槽 ----------------
    // 一个槽的搬运拆成七步，每步一个函数：定命中 → 发读 → 收集 → 等读 → 占 stage → 发写 → 落 metadata。
    // 调用顺序就是流水顺序，中间不允许再插别的指令，同步指令都留在各自那一步里。
    __aicore__ inline bool OpenSlot(uint32_t localSlot, SlotHeader &slot);
    __aicore__ inline void IssueTokenLoad(const SlotHeader &slot);
    __aicore__ inline uint32_t CollectHits(const SlotHeader &slot);
    __aicore__ inline void AwaitTokenLoad(const SlotHeader &slot);
    __aicore__ inline uint32_t AcquireStageBuf();
    __aicore__ inline void FillRecvAndStage(const SlotHeader &slot, uint32_t hitCnt, uint32_t stageOff);
    __aicore__ inline void FlushStage(const SlotHeader &slot, uint32_t hitCnt, uint32_t stageOff);
    __aicore__ inline void CopySlot(uint32_t localSlot);
    __aicore__ inline void CopyFromWindowByExpert();

    bool isDirectSelfRank_{false}; // 当前遍历的源 rank 是否为 direct 本端段
    GlobalTensor<int32_t> numRecvPerRankGm_;
    // dispatch 落在本端窗口里的 [源 rank][本端专家] token 计数表。每个二元组只有一个写者，
    // 所以「按二元组分核」+「读这张表」就能免掉原来那套 workspace 全核规约。
    GlobalTensor<int32_t> windowRankExpertCountGm_;

    // 本核负责的 (源 rank, 本端专家) 任务区间。task 号 = rank * numLocalExperts_ + localExpert。
    uint32_t taskBegin_{0};
    uint32_t taskEnd_{0};
    uint32_t taskCount_{0};
    uint32_t taskHits_{0};
    // taskBegin_/taskEnd_ 折算出来的源 rank 区间 [rankBegin_, rankEnd_)。
    uint32_t rankBegin_{0};
    uint32_t rankEnd_{0};

    LocalTensor<int32_t> ubRankExpertCount_; // 窗口计数表的 UB 影子，[rank][expert]，行步长 rankExpertRowStride_
    LocalTensor<int32_t> ubTaskMetaStart_;   // 每个本核任务的 metadata 下一行行号
    LocalTensor<int64_t> ubTaskXStart_;      // 每个本核任务的 recv_x 起始行号
    LocalTensor<int32_t> ubMeta_;
    LocalTensor<uint8_t> ubHitMask_;      // 整拍命中掩码，每个元素一位
    LocalTensor<uint8_t> ubHitMaskUpper_; // 同一块 buffer 的后半，用来和 ubHitMask_ 做 And
    LocalTensor<int32_t> ubRecvCnt_;
    LocalTensor<XType> tokenRing_;
    LocalTensor<float> ubStageWeightsRing_;
    LocalTensor<int32_t> ubStageMetaRing_;
    LocalTensor<int32_t> ubLocalCursor_;
    LocalTensor<int64_t> ubHitList_;

    TBuf<QuePosition::VECIN> ubRankExpertCountBuf_;
    TBuf<QuePosition::VECIN> ubTaskMetaStartBuf_;
    TBuf<QuePosition::VECIN> ubTaskXStartBuf_;
    TBuf<QuePosition::VECIN> ubMetaBuf_;
    TBuf<> ubHitMaskBuf_;
    TBuf<QuePosition::VECIN> ubRecvCntBuf_;
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
    // 窗口计数表在 UB 里每跑一个 rank 行步长。必须补到 ELEM_ALIGN 的整数倍：BuildTileHitMask 之外的消费者
    // 会按 rank * rankExpertRowStride_ 切片，而 VEC 访问 UB 要求 32 字节对齐；numLocalExperts_ 本身没有
    // 对齐保证（host 侧就是 numExperts / epWorldSize，例：numExperts=4, epWorldSize=2 → 2）。
    // host 侧窗口里的行步长则是 AlignMoeEpWin(numLocalExperts * 4)，两者不是一个值，别混。
    uint32_t rankExpertRowStride_{0};
    uint32_t rankExpertTableElems_{0}; // 整表元素数 = rankExpertRowStride_ * epWorldSize_
    uint32_t hitMaskStride_{0};        // 一份命中掩码占的字节数（含尾部余量），也是两份掩码之间的间隔
    // 掩码相关的两个量只跟 axisK_ 有关，进搬运循环前算一次：每槽取几个字节的掩码位、有效位屏蔽。
    uint32_t topkMaskBytes_{0};
    uint32_t validTopkMask_{0};
    // 搬运主循环的游标与拍号，字段见 TileCursor。
    TileCursor cur_;
    // 三处自管环的事件，份数各自等于对应 *_RING。全部走 AllocEventID 拿真份数：FetchEventID 不置占用位、
    // 恒返回同一个下标，一份以上的环会退化成一条 id，等待就分不出是哪一份。
    // meta 环要过一道 V（整拍比较），所以 MTE2 的落点是 V 而不是 S。
    int32_t metaEvtMte2ToV_[META_RING] = {0};
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
    // 命中数不再自己扫窗区数：dispatch 已经把 [源 rank][本端专家] 的 token 数写进本端窗口，
    // 且每个二元组恰好一个写者，所以按这个二元组给核分活就够了。
    windowRankExpertCountGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(GetWinAddrByRankId(mc2Context_, epRankId_, tilingData->expertCountWinOffset)));
    SplitToCore(epWorldSize_ * numLocalExperts_, aivNum_, taskBegin_, taskEnd_, taskCount_);
    rankBegin_ = taskBegin_ / numLocalExperts_;
    rankEnd_ = Ceil(taskEnd_, numLocalExperts_);

    rankExpertRowStride_ = Ceil(numLocalExperts_, ELEM_ALIGN) * ELEM_ALIGN;
    rankExpertTableElems_ = rankExpertRowStride_ * epWorldSize_;
    paddedMetaElems_ = Ceil(META_TOPK_SECTION * axisKAlign_ + META_EXTRA_FIELDS, ELEM_ALIGN) * ELEM_ALIGN;
    uint32_t ubRankExpertCountBytes = rankExpertTableElems_ * static_cast<uint32_t>(sizeof(int32_t));
    // taskCount_ 可能为 0（核比任务多），InitBuffer 不接受 0 长度，至少留一个元素。
    uint32_t allocatedTasks = taskCount_ == 0U ? 1U : taskCount_;
    uint32_t ubTaskMetaStartBytes =
        Ceil(allocatedTasks * static_cast<uint32_t>(sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubTaskXStartBytes = Ceil(allocatedTasks * static_cast<uint32_t>(sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubMetaBytes = Ceil((uint32_t)(SLOTS_TILE * paddedMetaElems_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    metaRingElems_ = ubMetaBytes / static_cast<uint32_t>(sizeof(int32_t));
    // 整拍比较会按 CMP_ALIGN_ELEMS 向上取整，最多多读 CMP_ALIGN_ELEMS - 1 个元素；补一小段尾巴，
    // 让最后一份 meta 的那次比较不越出本 buffer。
    uint32_t ubMetaSlackBytes = CMP_ALIGN_ELEMS * static_cast<uint32_t>(sizeof(int32_t));
    // 掩码一个元素一位，正好是 meta 元素数的 1/ELEM_ALIGN；尾部同样要留出「向上取整到 CMP_ALIGN_ELEMS」
    // 那一截，所以按 32B 取整之后再补 32B。
    hitMaskStride_ = Ceil(SLOTS_TILE * paddedMetaElems_ / ELEM_ALIGN, UB_ALIGN) * UB_ALIGN + UB_ALIGN;
    uint32_t ubRecvCntBytes = Ceil((uint32_t)(epWorldSize_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;
    // 一份 stage:最多 axisK_ 个命中，每个占 ELEM_ALIGN 个元素。份数见 STAGE_RING。
    stageSlotElems_ = axisK_ * ELEM_ALIGN;
    uint32_t ubStageSlotBytes = stageSlotElems_ * static_cast<uint32_t>(sizeof(int32_t));
    uint32_t ubStageMetaBytes = ubStageSlotBytes * STAGE_RING;
    uint32_t ubStageWeightsBytes = ubStageSlotBytes * STAGE_RING;
    uint32_t ubHitListBytes = Ceil((uint32_t)(axisK_ * HIT_ENTRY_SIZE * sizeof(int64_t)), UB_ALIGN) * UB_ALIGN;
    uint32_t ubLocalCursorBytes = Ceil((uint32_t)(numLocalExperts_ * sizeof(int32_t)), UB_ALIGN) * UB_ALIGN;

    tpipe_->InitBuffer(ubRecvCntBuf_, ubRecvCntBytes);
    tpipe_->InitBuffer(ubRankExpertCountBuf_, ubRankExpertCountBytes);
    tpipe_->InitBuffer(ubTaskMetaStartBuf_, ubTaskMetaStartBytes);
    tpipe_->InitBuffer(ubTaskXStartBuf_, ubTaskXStartBytes);
    tpipe_->InitBuffer(ubMetaBuf_, META_RING * ubMetaBytes + ubMetaSlackBytes);
    tpipe_->InitBuffer(ubHitMaskBuf_, hitMaskStride_ * 2U);
    tpipe_->InitBuffer(ubStageWeightsRingBuf_, ubStageWeightsBytes);
    tpipe_->InitBuffer(ubStageMetaRingBuf_, ubStageMetaBytes);
    tpipe_->InitBuffer(ubHitListBuf_, ubHitListBytes);
    tpipe_->InitBuffer(ubLocalCursorBuf_, ubLocalCursorBytes);
    ubRecvCnt_ = ubRecvCntBuf_.Get<int32_t>();
    ubRankExpertCount_ = ubRankExpertCountBuf_.Get<int32_t>();
    ubTaskMetaStart_ = ubTaskMetaStartBuf_.Get<int32_t>();
    ubTaskXStart_ = ubTaskXStartBuf_.Get<int64_t>();
    ubMeta_ = ubMetaBuf_.Get<int32_t>();
    ubHitMask_ = ubHitMaskBuf_.Get<uint8_t>();
    ubHitMaskUpper_ = ubHitMask_[hitMaskStride_];
    ubStageWeightsRing_ = ubStageWeightsRingBuf_.Get<float>();
    ubStageMetaRing_ = ubStageMetaRingBuf_.Get<int32_t>();
    ubHitList_ = ubHitListBuf_.Get<int64_t>();
    ubLocalCursor_ = ubLocalCursorBuf_.Get<int32_t>();
    // 三处环的事件各按份数取，互相不复用。
    for (uint32_t ringIdx = 0; ringIdx < META_RING; ++ringIdx) {
        metaEvtMte2ToV_[ringIdx] = static_cast<int32_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>());
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
    // CountHits 之后不再需要 SyncAll：命中数直接来自窗口，每个 (源 rank, 本端专家) 只有一个写者，
    // metadata 行号和 recv_x 行号都能在本核内独立算出来，不再有跨核依赖。
    BuildRankRowStarts();

    if (taskHits_ != 0U) {
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
    taskHits_ = 0U;
    DataCopyExtParams recvCntCopyParams{1U, static_cast<uint32_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> recvCntPadParams{false, 0U, 0U, 0};
    DataCopyPad(ubRecvCnt_, numRecvPerRankGm_, recvCntCopyParams, recvCntPadParams);

    // 命中数直接读 dispatch 落的 [源 rank][本端专家] 表。两条 dispatch（direct / hybrid）都在 count-ready
    // 之前把它写到本端窗口 expertCountWinOffset，行在窗口里按 WIN_ADDR_ALIGN 对齐、在 UB 里按
    // rankExpertRowStride_ 对齐，WaitDispatch 已经等到了 count-ready。
    // 没分到任务的核也要读：core 0 还要靠这张表算 recv_rank_offsets 的尾巴。
    uint32_t rowBytes = numLocalExperts_ * static_cast<uint32_t>(sizeof(int32_t));
    uint32_t windowRowBytes = Ceil(rowBytes, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    DataCopyExtParams countCopyParams{static_cast<uint16_t>(epWorldSize_), rowBytes, windowRowBytes - rowBytes, 0U, 0U};
    DataCopyPadExtParams<int32_t> countPadParams{true, 0U,
                                                 static_cast<uint8_t>(rankExpertRowStride_ - numLocalExperts_), 0};
    DataCopyPad(ubRankExpertCount_, windowRankExpertCountGm_, countCopyParams, countPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 本核只认领自己那几个任务。任务为空时这个循环不跑，taskHits_ 保持 0，Process 会跳过搬运。
    for (uint32_t task = taskBegin_; task < taskEnd_; ++task) {
        uint32_t rank = task / numLocalExperts_;
        uint32_t expert = task % numLocalExperts_;
        taskHits_ += static_cast<uint32_t>(ubRankExpertCount_.GetValue(rank * rankExpertRowStride_ + expert));
    }
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::BuildRankRowStarts()
{
    // core 0 即使没分到任务也要走完全程，因为它额外负责写完整的 recv_rank_offsets 尾巴；
    // 其余核按自己的任务区间走，没任务的直接返回。
    uint32_t loopRankEnd = aivId_ == 0U ? epWorldSize_ : rankEnd_;
    if (loopRankEnd == 0U) {
        return;
    }

    // metadata 行号：源 rank 大序、专家次之、槽位再次之，完全由窗口计数表决定。
    // recv_x 行号：专家大序、源 rank 次之、槽位再次之 —— 同一个专家里，前面的源 rank 先占行。
    // 两者都是纯本核计算，不再有跨核规约；core 0 顺带把 rank 前缀抄进 ubRankOffsets_。
    int32_t metadataPrefix = 0;
    for (uint32_t rank = 0; rank < loopRankEnd; ++rank) {
        if (aivId_ == 0U) {
            ubRankOffsets_.SetValue(rank, metadataPrefix);
        }
        for (uint32_t expert = 0; expert < numLocalExperts_; ++expert) {
            uint32_t task = rank * numLocalExperts_ + expert;
            if (task >= taskBegin_ && task < taskEnd_) {
                uint32_t localTask = task - taskBegin_;
                ubTaskMetaStart_.SetValue(localTask, metadataPrefix);
                // 本任务负责的 (rank, expert) 在 recv_x 里的起点 = 该专家的全局前缀
                // + 这个专家上前面的源 rank 收到的 token 数。numRecvPerExpert_[e] 本身就是本端窗口
                // 计数表按源 rank 求和的结果，所以这两项天然自洽。
                int64_t xStart = ubExpertPfx_.GetValue(expert);
                for (uint32_t prevRank = 0; prevRank < rank; ++prevRank) {
                    xStart += ubRankExpertCount_.GetValue(prevRank * rankExpertRowStride_ + expert);
                }
                ubTaskXStart_.SetValue(localTask, xStart);
            }
            metadataPrefix += ubRankExpertCount_.GetValue(rank * rankExpertRowStride_ + expert);
        }
    }
    if (aivId_ == 0U) {
        ubRankOffsets_.SetValue(epWorldSize_, metadataPrefix);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams offsetsCopyParams{1U, static_cast<uint32_t>((epWorldSize_ + 1U) * sizeof(int32_t)), 0U, 0U,
                                            0U};
        DataCopyPad(recvRankOffsetsGm_, ubRankOffsets_, offsetsCopyParams);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline bool MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::NextTile(
    uint32_t rankId, uint32_t tileStart, uint32_t &nextRank, uint32_t &nextTileStart, uint32_t &nextTileCnt)
{
    // 按 (源 rank, 本端专家) 分活之后，槽位不再按核切片：本核要扫完自己负责的那些源 rank 的全部槽位，
    // 靠专家掩码过滤掉不属于自己的。所以这里只按 rank 往右走，没有 slotStart。
    for (uint32_t r = rankId; r < rankEnd_; ++r) {
        int32_t slotCnt = ubRecvCnt_.GetValue(r);
        if (slotCnt == 0) {
            continue;
        }
        // 同一个 rank 内接着上一拍往后走；跨到下一个 rank 就从它的第 0 拍开始。TILE_NONE 表示还没发过任何一拍。
        uint32_t start = (r == rankId && tileStart != TILE_NONE) ? (tileStart + SLOTS_TILE) : 0U;
        if (start >= static_cast<uint32_t>(slotCnt)) {
            continue;
        }
        uint32_t remaining = static_cast<uint32_t>(slotCnt) - start;
        nextRank = r;
        nextTileStart = start;
        nextTileCnt = (remaining > SLOTS_TILE) ? SLOTS_TILE : remaining;
        return true;
    }
    return false;
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::IssueMeta(uint32_t rankId,
                                                                                                   uint32_t tileStart,
                                                                                                   uint32_t tileCnt,
                                                                                                   uint32_t buf)
{
    DataCopyExtParams metaCopyParams{static_cast<uint16_t>(tileCnt), metaBytes_, perSlotBytes_ - metaBytes_, 0, 0};
    DataCopyPadExtParams<int32_t> metaPadParams{false, 0, 0, 0};
    GlobalTensor<int32_t> srcMetaGm;
    srcMetaGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(localWinAddr_ + (int64_t)rankId * numMaxTokensPerRank_ * perSlotBytes_ +
                                           (int64_t)tileStart * perSlotBytes_ + metaOffset_));
    DataCopyPad(ubMeta_[buf * metaRingElems_], srcMetaGm, metaCopyParams, metaPadParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(metaEvtMte2ToV_[buf]);
}

template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::BuildTileHitMask(
    uint32_t metaRingBase, uint32_t tileCnt)
{
    // 本核在这个源 rank 上认领的专家区间。rankId 只会在 [rankBegin_, rankEnd_) 上取值，而
    // rankEnd_ = Ceil(taskEnd_, numLocalExperts_)，所以 taskEnd_ - rankId * numLocalExperts_ 恒 >= 1，不会下溢。
    uint32_t rankId = cur_.rankId;
    uint32_t firstTask = rankId * numLocalExperts_;
    uint32_t expertBegin = taskBegin_ > firstTask ? taskBegin_ - firstTask : 0U;
    uint32_t expertEnd = taskEnd_ - firstTask;
    expertEnd = expertEnd < numLocalExperts_ ? expertEnd : numLocalExperts_;
    int32_t assignedExpertBegin = static_cast<int32_t>(epRankId_ * numLocalExperts_ + expertBegin);
    int32_t assignedExpertEnd = static_cast<int32_t>(epRankId_ * numLocalExperts_ + expertEnd);

    // 掩码的写 / 比较读之间的 S→V、V→S 边由调用点（tile 循环里那道 S_V、这里结尾这道 V_S）保证。
    uint32_t compareCount = Ceil(tileCnt * paddedMetaElems_, CMP_ALIGN_ELEMS) * CMP_ALIGN_ELEMS;
    // 两条比较加一条与，一次判完整拍：[begin, end) 之外全是 0。原来是一个专家一条 CompareScalar + GatherMask。
    CompareScalar(ubHitMask_, ubMeta_[metaRingBase], assignedExpertBegin, AscendC::CMPMODE::GE, compareCount);
    CompareScalar(ubHitMaskUpper_, ubMeta_[metaRingBase], assignedExpertEnd, AscendC::CMPMODE::LT, compareCount);
    And(ubHitMask_, ubHitMask_, ubHitMaskUpper_, compareCount / ELEM_ALIGN);
    // 掩码是 V 写的，下面标量要按位读它。
    SyncFunc<AscendC::HardEvent::V_S>();
}

// 搬运主循环。三段一拍：进这一拍（等 meta、预读下一拍）、处理这一拍（整拍判命中 + 逐槽搬）、
// 出这一拍（归还 meta、推游标）。三段的调用顺序就是流水顺序，中间不允许插别的指令。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::CopyFromWindowByExpert()
{
    PrepareCopy();
    while (cur_.hasTile) {
        BeginTile();
        BuildTileHitMask(cur_.metaRingBase, cur_.tileCnt);
        for (uint32_t localSlot = 0; localSlot < cur_.tileCnt; ++localSlot) {
            CopySlot(localSlot);
        }
        EndTile();
    }
    // 这里不再做 tile 尾的 MTE3 排空：Caller(Process) 紧接着就有 SyncFunc<MTE3_S> 兜住 recv_x/metadata/weights
    // 的可见性，环的事件又是 AllocEventID 私有、不会串到别处，所以函数返回时的 pipe 状态不需要在这里再等一次。
}

// 进搬运前的准备：算掩码常量、复位游标，并把第一拍的 meta 读发出去（它的延迟盖在下面按拍处理里）。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::PrepareCopy()
{
    topkMaskBytes_ = Ceil(axisK_, ELEM_ALIGN);
    // 掩码是按整张 meta 布局算的，每个槽只取前 axisK_ 位；高位可能是 padding 或 weights 段，必须掩掉。
    validTopkMask_ = 0xFFFFFFFFU >> (TOPK_MASK_BITS - axisK_);

    // 拍平成一个「取下一个 tile」的循环。
    cur_.Reset();
    cur_.hasTile = NextTile(rankBegin_, TILE_NONE, cur_.rankId, cur_.tileStart, cur_.tileCnt);
    if (cur_.hasTile) {
        IssueMeta(cur_.rankId, cur_.tileStart, cur_.tileCnt, 0U);
    }
}

// 进这一拍：等本拍的 meta 到、把下一拍定位并预读、跨 rank 复位游标。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::BeginTile()
{
    // 当前拍的 meta 读是上一拍发出去的，这里只等它到。
    cur_.metaBuf = cur_.metaSeq % META_RING;
    WaitFlag<AscendC::HardEvent::MTE2_V>(metaEvtMte2ToV_[cur_.metaBuf]);
    ++cur_.metaSeq;
    cur_.metaRingBase = cur_.metaBuf * metaRingElems_;
    // 先把下一拍定位并预读，再处理当前拍：读延迟盖在当前拍的 MTE3 写里，标量 pipe 不用等。
    cur_.hasNext = NextTile(cur_.rankId, cur_.tileStart, cur_.nextRank, cur_.nextTileStart, cur_.nextTileCnt);
    if (cur_.hasNext) {
        const uint32_t nextMetaBuf = cur_.metaSeq % META_RING;
        if (cur_.metaSeq >= META_RING) { // 这一份上一次用是 META_RING 拍之前，先确认标量已经读完
            WaitFlag<AscendC::HardEvent::S_MTE2>(metaEvtSToMte2_[nextMetaBuf]);
        }
        IssueMeta(cur_.nextRank, cur_.nextTileStart, cur_.nextTileCnt, nextMetaBuf);
    }
    // 上一拍标量对游标和掩码的写要落定，这一拍的 Duplicate / 整拍比较才能覆盖它们。屏障必须压在
    // Duplicate 之前：SyncFunc<S_V> 只约束它「之后」的向量指令，挡不住这条写在它前面的 Duplicate。
    SyncFunc<AscendC::HardEvent::S_V>();
    if (cur_.rankId != cur_.cursorRank) {
        ResetExpertCursors(cur_.rankId);
    }
}

// 出这一拍：归还本拍的 meta 份，把游标推到下一拍。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::EndTile()
{
    // 这一拍的 meta 已经被读完，可以还给 MTE2 了。只有还有下一拍时才发：这样 S_MTE2 上「每次 Wait
    // 恰好一次 Set」，不攒多余信用。
    // 读 ubMeta_ 的有两条 pipe：标量（GetValue）和向量（整拍比较），而 S_MTE2 只约束标量。这里不用
    // 再补一道 V_MTE2：本拍 BuildTileHitMask 结尾的 V_S 已经让标量 pipe 停到「本拍向量读完」之后，
    // 所以这道 SetFlag 本身就蕴含了向量侧也读完了，下一拍 IssueMeta 覆盖这个 buffer 是安全的。
    if (cur_.hasNext) {
        SetFlag<AscendC::HardEvent::S_MTE2>(metaEvtSToMte2_[cur_.metaBuf]);
    }
    cur_.rankId = cur_.nextRank;
    cur_.tileStart = cur_.nextTileStart;
    cur_.tileCnt = cur_.nextTileCnt;
    cur_.hasTile = cur_.hasNext;
}

// 换源 rank：每个 (rank, expert) 的 recv_x 游标都从 0 开始，跨 rank 要清零（rank 内各拍累计）。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::ResetExpertCursors(
    uint32_t rankId)
{
    cur_.cursorRank = rankId;
    Duplicate(ubLocalCursor_, (int32_t)0, numLocalExperts_);
    // direct 本端x 直读输入；hybrid 保持读窗口
    isDirectSelfRank_ = (networkMode_ != NETWORK_HYBRID) && (rankId == static_cast<uint32_t>(epRankId_));
}

// 一个槽的搬运。七步的顺序就是流水顺序：读先发出去盖住下面的标量收集，写要等 stage 份空出来。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline void MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::CopySlot(uint32_t localSlot)
{
    SlotHeader slot;
    if (!OpenSlot(localSlot, slot)) {
        return;
    }
    IssueTokenLoad(slot);
    const uint32_t hitCnt = CollectHits(slot);
    AwaitTokenLoad(slot);
    const uint32_t stageOff = AcquireStageBuf();
    FillRecvAndStage(slot, hitCnt, stageOff);
    FlushStage(slot, hitCnt, stageOff);
}

// 打开本槽：拼出本槽的命中掩码、读槽头、占下这一槽的 token 环份。返回 false 表示整槽跳过。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline bool MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::OpenSlot(uint32_t localSlot,
                                                                                                  SlotHeader &slot)
{
    slot.metaBase = cur_.metaRingBase + localSlot * paddedMetaElems_;
    slot.slotIdx = cur_.tileStart + localSlot;
    // 这一槽命中本核专家的 topk 下标集合。掩码是按整拍算的，这里只把该槽那几位拼成一个 uint32；
    // 掩码 buffer 不随 meta 成环，所以这里的下标只含拍内槽号，不带 metaRingBase。
    uint32_t hitMask = 0U;
    for (uint32_t byte = 0; byte < topkMaskBytes_; ++byte) {
        hitMask |= static_cast<uint32_t>(ubHitMask_.GetValue(localSlot * paddedMetaElems_ / ELEM_ALIGN + byte))
                   << (byte * ELEM_ALIGN);
    }
    slot.hitMask = hitMask & validTopkMask_;
    if (slot.hitMask == 0U) {
        // 本槽没有命中本核的专家。任务按 (rank, expert) 分，命中天然稀疏，这里必须连 token/scales
        // 都不去读：原来「每槽无条件先发 token 读」在这个分核方式下会白读绝大多数槽。
        return false;
    }
    slot.srcRankMeta = ubMeta_.GetValue(slot.metaBase + META_TOPK_SECTION * axisKAlign_);
    slot.tokenIdxMeta = ubMeta_.GetValue(slot.metaBase + META_TOPK_SECTION * axisKAlign_ + 1);
    slot.tokenBuf = cur_.tokenSeq % TOKEN_RING;
    return true;
}

// 发出本槽 token/scales 的读。内容只取决于 slot，与命中集合无关，所以放在命中循环之前发：
// MTE2 这一拍的读和下面 loop A 的标量收集并行。
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
    GM_ADDR slotAddr =
        localWinAddr_ + (int64_t)cur_.rankId * numMaxTokensPerRank_ * perSlotBytes_ + (int64_t)slot.slotIdx * perSlotBytes_;
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

// loop A：按位置位迭代命中项，把每项的 (recv_x 行, topk) 记进 ubHitList_。掩码已经把「专家落在本核区间」
// 这个条件判完了，所以这里省掉了原来每个 topk 一次的范围判断，迭代次数恰好等于命中数。
template <typename XType, typename ScalesType, bool HasTopkWeights>
__aicore__ inline uint32_t MoeEpDispatchEpilogueUncached<XType, ScalesType, HasTopkWeights>::CollectHits(
    const SlotHeader &slot)
{
    const int32_t rankExpertBase = static_cast<int32_t>(epRankId_ * numLocalExperts_);
    uint32_t hitCnt = 0;
    for (uint32_t pendingMask = slot.hitMask; pendingMask != 0U; pendingMask &= (pendingMask - 1U)) {
        uint32_t topkIdx = static_cast<uint32_t>(ScalarGetSFFValue<1>(static_cast<uint64_t>(pendingMask)));
        int32_t expertId = ubMeta_.GetValue(slot.metaBase + topkIdx);
        uint32_t localExpertId = static_cast<uint32_t>(expertId - rankExpertBase);
        uint32_t localTask = cur_.rankId * numLocalExperts_ + localExpertId - taskBegin_;
        int64_t expertRowStart = ubTaskXStart_.GetValue(localTask);
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

// 占一份 stage，返回它在 ubStage*Ring_ 里的偏移。掩码非 0 才有这一槽，所以 hitCnt >= 1，
// 不存在「占了 stage 又没人消费」的分支。
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
        uint32_t localTask = cur_.rankId * numLocalExperts_ + localExpertId - taskBegin_;
        int32_t metadataRow = ubTaskMetaStart_.GetValue(localTask);
        DataCopyPad(recvSrcMetadataGm_[(int64_t)metadataRow * RECV_META_FIELDS],
                    ubStageMetaRing_[stageOff + i * ELEM_ALIGN], metaOutParams);
        ubTaskMetaStart_.SetValue(localTask, metadataRow + 1);
    }
    SetFlag<AscendC::HardEvent::MTE3_S>(stageEvtMte3ToS_[cur_.stageBuf]);
}

#endif

} // namespace MoeEpDispatchEpilogueImpl

#endif // MOE_EP_DISPATCH_EPILOGUE_UNCACHED_H
