/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_fetch_local_arch35.h
 * \brief engram_fetch算子arch35全量本NPU(local)模式kernel实现
 * fetch_mode=1: 每个rank持有全局全量表, indices即本表行号, 取数纯本地GM拷贝, 无跨NPU通信
 */

#ifndef ENGRAM_FETCH_LOCAL_ARCH35_H
#define ENGRAM_FETCH_LOCAL_ARCH35_H

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_ENGRAM_FETCH_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_tiling_data.h"
#include "../engram_fetch_utils.h"
#include "adv_api/index/arithprogression.h"

namespace Mc2Kernel {

#if defined(ENABLE_ENGRAM_FETCH_KERNEL)

template <AscendC::HardEvent event>
__aicore__ inline void EngramFetchLocalSyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// 推理local模式: fetched[t] = localTable[indices[t]]
// 表基址 = commBuffer[rankId](host pinned注册内存的设备映射地址), entry即全局行号(无rank偏移)
class EngramFetchLocalArch35 {
public:
    __aicore__ inline EngramFetchLocalArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR sfTable,
                                GM_ADDR fetchedSf, GM_ADDR workspaceGM, AscendC::TPipe *pipe,
                                const EngramFetchTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void GatherSf(int32_t entryIdx, uint64_t globalTokenIdx);
    __aicore__ inline void CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen);
    __aicore__ inline void GatherRange(uint32_t start, uint32_t end);

    AscendC::TPipe *tpipe_{nullptr};
    GM_ADDR indicesGM_{nullptr};
    GM_ADDR fetchedGM_{nullptr};
    __gm__ uint8_t *sfTableGM_{nullptr};
    GM_ADDR fetchedSfGM_{nullptr};
    __gm__ EngramCommContext *ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{1};
    int32_t numEntriesPerRank_{0};
    int64_t numTokens_{0};
    int64_t hiddenBytes_{0};
    int64_t numSfPacks_{0};
    int64_t sfElemSize_{0};
    uint64_t ubSize_{0};
    uint64_t tableBase_{0};
    uint32_t indicesBatchSize_{0};

    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, RELAY_BUFFER_NUM> relayQue_;
    AscendC::TBuf<> indicesBuf_;
};

__aicore__ inline void EngramFetchLocalArch35::Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
                                                    GM_ADDR sfTable, GM_ADDR fetchedSf, GM_ADDR workspaceGM,
                                                    AscendC::TPipe *pipe, const EngramFetchTilingData *tilingData)
{
    tpipe_ = pipe;
    indicesGM_ = indices;
    fetchedGM_ = fetched;
    fetchedSfGM_ = fetchedSf;
    sfTableGM_ = reinterpret_cast<__gm__ uint8_t *>(sfTable);
    aivId_ = AscendC::GetBlockIdx();
    totalBlocks_ = AscendC::GetBlockNum();
    (void)workspaceGM;

    ctxPtr_ = (__gm__ EngramCommContext *)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;

    AscendC::GlobalTensor<uint64_t> commBufferGm;
    commBufferGm.SetGlobalBuffer((__gm__ uint64_t *)&ctxPtr_->commBuffer[0]);
    tableBase_ = commBufferGm.GetValue(rankId_);

    numEntriesPerRank_ = tilingData->numEntriesPerRank;
    numTokens_ = tilingData->numTokens;
    hiddenBytes_ = tilingData->hiddenBytes;
    ubSize_ = tilingData->ubSize;
    numSfPacks_ = tilingData->numSfPacks;
    sfElemSize_ = tilingData->sfElemSize;

    tpipe_->InitBuffer(relayQue_, RELAY_BUFFER_NUM, TILE_BYTES);

    constexpr uint64_t ubReserved = 8U * 1024U;
    uint64_t usedUb = TILE_BYTES * RELAY_BUFFER_NUM;
    uint64_t availableUb = (ubSize_ > usedUb + ubReserved) ? (ubSize_ - usedUb - ubReserved) : 0U;
    uint32_t maxBatchSize = static_cast<uint32_t>(availableUb / sizeof(int32_t));
    uint32_t numTokens = static_cast<uint32_t>(numTokens_);
    indicesBatchSize_ = (numTokens <= maxBatchSize) ? numTokens : maxBatchSize;
    if (indicesBatchSize_ == 0U) {
        indicesBatchSize_ = 1U;
    }
    uint32_t indicesBufSize = AscendC::Ceil(indicesBatchSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
}

__aicore__ inline void EngramFetchLocalArch35::CopyIndicesToUb(uint32_t indicesBatchStart,
                                                               uint32_t indicesBatchLen)
{
    AscendC::GlobalTensor<int32_t> indicesGlobal;
    indicesGlobal.SetGlobalBuffer((__gm__ int32_t *)indicesGM_);
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    AscendC::DataCopyExtParams params{1U, indicesBatchLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(indicesLocal, indicesGlobal[indicesBatchStart], params, pad);
}

__aicore__ inline void EngramFetchLocalArch35::GatherSf(int32_t entryIdx, uint64_t globalTokenIdx)
{
    uint32_t sfBytes = static_cast<uint32_t>(numSfPacks_) * static_cast<uint32_t>(sfElemSize_);
    GM_ADDR sfSrc = sfTableGM_ + static_cast<uint64_t>(entryIdx) * sfBytes;
    GM_ADDR sfDst = fetchedSfGM_ + globalTokenIdx * sfBytes;
    AscendC::GlobalTensor<uint8_t> sfSrcGm;
    AscendC::GlobalTensor<uint8_t> sfDstGm;
    sfSrcGm.SetGlobalBuffer((__gm__ uint8_t *)sfSrc);
    sfDstGm.SetGlobalBuffer((__gm__ uint8_t *)sfDst);
    AscendC::LocalTensor<uint8_t> sfTmp = relayQue_.AllocTensor<uint8_t>();
    AscendC::DataCopyExtParams sfCopyIn{1U, sfBytes, 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<uint8_t> sfPadIn{false, 0, 0, 0};
    AscendC::DataCopyPad(sfTmp, sfSrcGm, sfCopyIn, sfPadIn);
    relayQue_.EnQue<uint8_t>(sfTmp);
    sfTmp = relayQue_.DeQue<uint8_t>();
    AscendC::DataCopyExtParams sfCopyOut{1U, sfBytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(sfDstGm, sfTmp, sfCopyOut);
    relayQue_.FreeTensor<uint8_t>(sfTmp);
}

__aicore__ inline void EngramFetchLocalArch35::GatherRange(uint32_t start, uint32_t end)
{
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    bool hasSf = (numSfPacks_ > 0 && sfTableGM_ != nullptr && fetchedSfGM_ != nullptr);
    uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);

    uint32_t cur = start;
    while (cur < end) {
        uint32_t batchLen = indicesBatchSize_;
        if (cur + batchLen > end) {
            batchLen = end - cur;
        }
        CopyIndicesToUb(cur, batchLen);
        EngramFetchLocalSyncFunc<AscendC::HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < batchLen; i++) {
            int32_t entry = indicesLocal.GetValue(i);
            if (entry < 0 || entry >= numEntriesPerRank_) {
                continue;
            }
            uint64_t globalTokenIdx = static_cast<uint64_t>(cur) + i;
            GM_ADDR dst = fetchedGM_ + globalTokenIdx * hiddenBytes;
            GM_ADDR src = (GM_ADDR)tableBase_ + static_cast<uint64_t>(entry) * hiddenBytes;
            LocalCopySlice(dst, src, hiddenBytes);
            if (hasSf) {
                GatherSf(entry, globalTokenIdx);
            }
        }
        cur += batchLen;
    }
}

__aicore__ inline void EngramFetchLocalArch35::Process()
{
    if ASCEND_IS_AIV {
        if (numEntriesPerRank_ == 0 || numTokens_ == 0) {
            return;
        }
        uint32_t numTokens = static_cast<uint32_t>(numTokens_);
        uint32_t totalPerCore = (numTokens + totalBlocks_ - 1U) / totalBlocks_;
        if (totalPerCore == 0U) {
            totalPerCore = 1U;
        }
        uint32_t start = aivId_ * totalPerCore;
        uint32_t end = start + totalPerCore;
        if (end > numTokens) {
            end = numTokens;
        }
        GatherRange(start, end);
        EngramFetchLocalSyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

__aicore__ inline void EngramFetchLocalArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    AscendC::GlobalTensor<uint8_t> srcGm;
    AscendC::GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);

    uint32_t tileLen = TILE_BYTES;
    uint64_t off = 0;
    while (off < len) {
        uint64_t thisLen = (len - off > TILE_BYTES) ? tileLen : (len - off);

        AscendC::LocalTensor<uint8_t> tmp = relayQue_.AllocTensor<uint8_t>();
        AscendC::DataCopy(tmp, srcGm[off], thisLen);
        relayQue_.EnQue<uint8_t>(tmp);
        tmp = relayQue_.DeQue<uint8_t>();
        AscendC::DataCopy(dstGm[off], tmp, thisLen);
        relayQue_.FreeTensor<uint8_t>(tmp);

        off += thisLen;
    }
}

// 训练local模式: fetched[t] = localTable[indices[t]] + 生成与TRAIN_MODE同构的反向上下文
// perm=恒等置换; sendCounts/recvCounts仅[me]=T其余0; recvLocalEntry=indices(token序); numRecv=T
class EngramFetchTrainLocalArch35 {
public:
    __aicore__ inline EngramFetchTrainLocalArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR permOut,
                                GM_ADDR sendCountsOut, GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut,
                                GM_ADDR numRecvOut, GM_ADDR workspaceGM, GM_ADDR localStorageAddr,
                                AscendC::TPipe *pipe, const EngramFetchTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen);
    __aicore__ inline void GatherRange(uint32_t start, uint32_t end);
    __aicore__ inline void WriteCounters();

    AscendC::TPipe *tpipe_{nullptr};
    GM_ADDR indicesGM_{nullptr};
    GM_ADDR fetchedGM_{nullptr};
    GM_ADDR permOutGM_{nullptr};
    GM_ADDR sendCountsOutGM_{nullptr};
    GM_ADDR recvCountsOutGM_{nullptr};
    GM_ADDR recvLocalEntryOutGM_{nullptr};
    GM_ADDR numRecvOutGM_{nullptr};
    __gm__ EngramCommContext *ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{1};
    int32_t numEntriesPerRank_{0};
    uint32_t numTokens_{0};
    int64_t hiddenBytes_{0};
    uint64_t ubSize_{0};
    uint64_t localStorageAddr_{0};
    uint32_t indicesBatchSize_{0};

    AscendC::TBuf<> pingBuf_;
    AscendC::TBuf<> pongBuf_;
    int32_t ppEvtMte2ToMte3_[2] = {0, 0};
    int32_t ppEvtMte3ToMte2_[2] = {0, 0};
    AscendC::TBuf<> indicesBuf_;
    AscendC::TBuf<> permBuf_;
    AscendC::TBuf<> tempBuf_;
};

__aicore__ inline void EngramFetchTrainLocalArch35::Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
                                                         GM_ADDR permOut, GM_ADDR sendCountsOut,
                                                         GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut,
                                                         GM_ADDR numRecvOut, GM_ADDR workspaceGM,
                                                         GM_ADDR localStorageAddr, AscendC::TPipe *pipe,
                                                         const EngramFetchTilingData *tilingData)
{
    tpipe_ = pipe;
    indicesGM_ = indices;
    fetchedGM_ = fetched;
    permOutGM_ = permOut;
    sendCountsOutGM_ = sendCountsOut;
    recvCountsOutGM_ = recvCountsOut;
    recvLocalEntryOutGM_ = recvLocalEntryOut;
    numRecvOutGM_ = numRecvOut;
    (void)workspaceGM;
    aivId_ = AscendC::GetBlockIdx();
    totalBlocks_ = AscendC::GetBlockNum();

    ctxPtr_ = (__gm__ EngramCommContext *)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;

    numEntriesPerRank_ = tilingData->numEntriesPerRank;
    numTokens_ = static_cast<uint32_t>(tilingData->numTokens);
    hiddenBytes_ = tilingData->hiddenBytes;
    ubSize_ = tilingData->ubSize;

    AscendC::GlobalTensor<int64_t> localStorageTensor;
    localStorageTensor.SetGlobalBuffer((__gm__ int64_t *)localStorageAddr);
    localStorageAddr_ = static_cast<uint64_t>(localStorageTensor.GetValue(0));

    tpipe_->InitBuffer(pingBuf_, TILE_BYTES);
    tpipe_->InitBuffer(pongBuf_, TILE_BYTES);
    ppEvtMte2ToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    ppEvtMte2ToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    ppEvtMte3ToMte2_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    ppEvtMte3ToMte2_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

    // sendCounts 区为 rankSize*8 个 int32(32B 步长契约), tempBuf 覆盖 sendCounts/recvCounts/numRecv 写出
    uint32_t tempBufSize = AscendC::Ceil(numRanks_ * 8U * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tempBuf_, tempBufSize);

    constexpr uint64_t ubReserved = 8U * 1024U;
    uint64_t usedUb = TILE_BYTES * 2U + tempBufSize;
    uint64_t availableUb = (ubSize_ > usedUb + ubReserved) ? (ubSize_ - usedUb - ubReserved) : 0U;
    uint32_t maxBatchSize = static_cast<uint32_t>(availableUb / (sizeof(int32_t) * 2U));
    indicesBatchSize_ = (numTokens_ <= maxBatchSize) ? numTokens_ : maxBatchSize;
    if (indicesBatchSize_ == 0U) {
        indicesBatchSize_ = 1U;
    }
    uint32_t indicesBufSize = AscendC::Ceil(indicesBatchSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
    uint32_t permBufSize = AscendC::Ceil(indicesBatchSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(permBuf_, permBufSize);
}

__aicore__ inline void EngramFetchTrainLocalArch35::CopyIndicesToUb(uint32_t indicesBatchStart,
                                                                    uint32_t indicesBatchLen)
{
    AscendC::GlobalTensor<int32_t> indicesGlobal;
    indicesGlobal.SetGlobalBuffer((__gm__ int32_t *)indicesGM_);
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    AscendC::DataCopyExtParams params{1U, indicesBatchLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(indicesLocal, indicesGlobal[indicesBatchStart], params, pad);
}

__aicore__ inline void EngramFetchTrainLocalArch35::GatherRange(uint32_t start, uint32_t end)
{
    AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    AscendC::LocalTensor<int32_t> permLocal = permBuf_.Get<int32_t>();
    uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);

    uint32_t cur = start;
    while (cur < end) {
        uint32_t batchLen = indicesBatchSize_;
        if (cur + batchLen > end) {
            batchLen = end - cur;
        }
        CopyIndicesToUb(cur, batchLen);
        EngramFetchLocalSyncFunc<AscendC::HardEvent::MTE2_S>();

        // perm 恒等置换: perm[cur + i] = cur + i
        AscendC::ArithProgression<int32_t>(permLocal, static_cast<int32_t>(cur), 1, batchLen);
        AscendC::PipeBarrier<PIPE_V>();
        EngramFetchLocalSyncFunc<AscendC::HardEvent::V_MTE3>();
        AscendC::GlobalTensor<int32_t> permGm;
        permGm.SetGlobalBuffer((__gm__ int32_t *)permOutGM_);
        AscendC::DataCopyExtParams permParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(permGm[cur], permLocal, permParams);

        // recvLocalEntry = indices(token序): 收到的请求即本卡 indices
        EngramFetchLocalSyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        AscendC::GlobalTensor<int32_t> recvEntryGm;
        recvEntryGm.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM_);
        AscendC::DataCopyPad(recvEntryGm[cur], indicesLocal, permParams);
        EngramFetchLocalSyncFunc<AscendC::HardEvent::MTE3_S>();

        for (uint32_t i = 0; i < batchLen; i++) {
            int32_t entry = indicesLocal.GetValue(i);
            if (entry < 0 || entry >= numEntriesPerRank_) {
                continue;
            }
            uint64_t globalTokenIdx = static_cast<uint64_t>(cur) + i;
            GM_ADDR dst = fetchedGM_ + globalTokenIdx * hiddenBytes;
            GM_ADDR src = (GM_ADDR)localStorageAddr_ + static_cast<uint64_t>(entry) * hiddenBytes;
            LocalCopySlice(dst, src, hiddenBytes);
        }
        cur += batchLen;
    }
}

__aicore__ inline void EngramFetchTrainLocalArch35::WriteCounters()
{
    AscendC::LocalTensor<int32_t> tempLocal = tempBuf_.Get<int32_t>();
    uint32_t sendCountsInt32 = numRanks_ * 8U;

    // sendCounts: 整区零填后仅置 [me] 槽 = numTokens(其余 rank 槽保持 0)
    AscendC::Duplicate<int32_t>(tempLocal, 0, sendCountsInt32);
    AscendC::PipeBarrier<PIPE_V>();
    EngramFetchLocalSyncFunc<AscendC::HardEvent::V_S>();
    tempLocal.SetValue(rankId_ * 8U, static_cast<int32_t>(numTokens_));
    EngramFetchLocalSyncFunc<AscendC::HardEvent::S_MTE3>();
    AscendC::GlobalTensor<int32_t> sendCountsGm;
    sendCountsGm.SetGlobalBuffer((__gm__ int32_t *)sendCountsOutGM_);
    AscendC::DataCopyExtParams sendCountsParams{1U, static_cast<uint32_t>(sendCountsInt32 * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(sendCountsGm, tempLocal, sendCountsParams);
    EngramFetchLocalSyncFunc<AscendC::HardEvent::MTE3_S>();

    // recvCounts: 零填后仅置 [me] = numTokens
    AscendC::Duplicate<int32_t>(tempLocal, 0, numRanks_);
    AscendC::PipeBarrier<PIPE_V>();
    EngramFetchLocalSyncFunc<AscendC::HardEvent::V_S>();
    tempLocal.SetValue(rankId_, static_cast<int32_t>(numTokens_));
    EngramFetchLocalSyncFunc<AscendC::HardEvent::S_MTE3>();
    AscendC::GlobalTensor<int32_t> recvCountsGm;
    recvCountsGm.SetGlobalBuffer((__gm__ int32_t *)recvCountsOutGM_);
    AscendC::DataCopyExtParams recvCountsParams{1U, static_cast<uint32_t>(numRanks_ * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(recvCountsGm, tempLocal, recvCountsParams);
    EngramFetchLocalSyncFunc<AscendC::HardEvent::MTE3_S>();

    // numRecv = numTokens
    tempLocal.SetValue(0, static_cast<int32_t>(numTokens_));
    EngramFetchLocalSyncFunc<AscendC::HardEvent::S_MTE3>();
    AscendC::GlobalTensor<int32_t> numRecvGm;
    numRecvGm.SetGlobalBuffer((__gm__ int32_t *)numRecvOutGM_);
    AscendC::DataCopyExtParams numRecvParams{1U, sizeof(int32_t), 0U, 0U, 0U};
    AscendC::DataCopyPad(numRecvGm, tempLocal, numRecvParams);
    EngramFetchLocalSyncFunc<AscendC::HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainLocalArch35::Process()
{
    if ASCEND_IS_AIV {
        if (numEntriesPerRank_ == 0 || numTokens_ == 0) {
            return;
        }
        uint32_t totalPerCore = (numTokens_ + totalBlocks_ - 1U) / totalBlocks_;
        if (totalPerCore == 0U) {
            totalPerCore = 1U;
        }
        uint32_t start = aivId_ * totalPerCore;
        uint32_t end = start + totalPerCore;
        if (end > numTokens_) {
            end = numTokens_;
        }
        GatherRange(start, end);
        if (aivId_ == 0U) {
            WriteCounters();
        }
    }
}

__aicore__ inline void EngramFetchTrainLocalArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    AscendC::LocalTensor<uint8_t> tmp0 = pingBuf_.Get<uint8_t>();
    AscendC::LocalTensor<uint8_t> tmp1 = pongBuf_.Get<uint8_t>();
    AscendC::GlobalTensor<uint8_t> srcGm;
    AscendC::GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);

    uint32_t tileLen = TILE_BYTES;
    uint64_t off = 0;
    uint32_t tileIdx = 0;
    while (off < len) {
        uint64_t thisLen = (len - off > TILE_BYTES) ? tileLen : (len - off);
        uint32_t bufIdx = tileIdx % 2U;
        AscendC::LocalTensor<uint8_t> tmp = (bufIdx == 0U) ? tmp0 : tmp1;

        if (tileIdx >= 2U) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        }

        AscendC::DataCopyExtParams mte2Params{1U, static_cast<uint32_t>(thisLen), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<uint8_t> mte2Pad{false, 0, 0, 0};
        AscendC::DataCopyPad(tmp, srcGm[off], mte2Params, mte2Pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        AscendC::DataCopyParams mte3Params{1U, static_cast<uint16_t>(thisLen), 0U, 0U};
        AscendC::DataCopyPad(dstGm[off], tmp, mte3Params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);

        off += thisLen;
        tileIdx++;
    }

    if (tileIdx >= 1U) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[(tileIdx - 1U) % 2U]);
    }
    if (tileIdx >= 2U) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[tileIdx % 2U]);
    }
}

#endif

} // namespace Mc2Kernel

#endif
