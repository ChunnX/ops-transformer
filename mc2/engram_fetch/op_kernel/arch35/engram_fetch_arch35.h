#ifndef ENGRAM_FETCH_ARCH35_H
#define ENGRAM_FETCH_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_tiling_data.h"
#include "../engram_fetch.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"

namespace EngramFetchImpl {

using namespace AscendC;
using namespace Mc2Aclnn;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

class EngramFetchArch35 {
public:
    __aicore__ inline EngramFetchArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR workspaceGM,
                                TPipe *pipe, const EngramFetchTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void CopyGmToUb();

    TPipe *tpipe_{nullptr};
    GM_ADDR indicesGM_{nullptr};
    GM_ADDR fetchedGM_{nullptr};
    const EngramFetchTilingData *tilingData_{nullptr};
    __gm__ EngramCommContext *ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    uint32_t numEntriesPerRank_{0};
    uint32_t numTokens_{0};
    uint32_t hiddenBytes_{0};

    TBuf<> dataBuf0_;
    TBuf<> dataBuf1_;
    int32_t ppEvtMte2ToMte3_[2] = {0, 0};
    int32_t ppEvtMte3ToMte2_[2] = {0, 0};
    TBuf<> indicesBuf_;
    TBuf<> hcommBuf_;
    TBuf<> countsBuf_;
    TBuf<> offsetsBuf_;
    TBuf<> bucketPosBuf_;
    TBuf<> commBufferBuf_;
    TBuf<> hcommHandleBuf_;
    LocalTensor<uint8_t> hcommTensor_;

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
};

__aicore__ inline void EngramFetchArch35::Init(
    GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR workspaceGM,
    TPipe *pipe, const EngramFetchTilingData *tilingData)
{
    tpipe_ = pipe;
    indicesGM_ = indices;
    fetchedGM_ = fetched;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();

    ctxPtr_ = (__gm__ EngramCommContext *)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;

    numEntriesPerRank_ = tilingData_->numEntriesPerRank;
    numTokens_ = tilingData_->numTokens;
    hiddenBytes_ = tilingData_->hiddenBytes;

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(dataBuf0_, TILE_BYTES);
    tpipe_->InitBuffer(dataBuf1_, TILE_BYTES);
    ppEvtMte2ToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte2ToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte3ToMte2_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    ppEvtMte3ToMte2_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    uint32_t indicesBufSize = Ceil(numTokens_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);

    uint32_t countsBufSize = Ceil(numRanks_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(countsBuf_, countsBufSize);
    uint32_t offsetsBufSize = Ceil((numRanks_ + 1U) * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(offsetsBuf_, offsetsBufSize);
    uint32_t bucketPosBufSize = Ceil(numTokens_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(bucketPosBuf_, bucketPosBufSize);
    uint32_t commBufferBufSize = Ceil(numRanks_ * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(commBufferBuf_, commBufferBufSize);
    uint32_t hcommHandleBufSize = Ceil(numRanks_ * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(hcommHandleBuf_, hcommHandleBufSize);
}

__aicore__ inline void EngramFetchArch35::CopyGmToUb()
{
    GlobalTensor<int32_t> indicesGlobal;
    indicesGlobal.SetGlobalBuffer((__gm__ int32_t *)indicesGM_);
    LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    DataCopyExtParams params{1U, numTokens_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(indicesLocal, indicesGlobal, params, pad);

    LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    GlobalTensor<uint64_t> commBufferGm;
    commBufferGm.SetGlobalBuffer((__gm__ uint64_t *)&ctxPtr_->commBuffer[0]);
    DataCopyExtParams cpComm{1U, numRanks_ * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<uint64_t> padComm{false, 0, 0, 0};
    DataCopyPad(commBufferLocal, commBufferGm, cpComm, padComm);

    LocalTensor<uint64_t> hcommHandleLocal = hcommHandleBuf_.Get<uint64_t>();
    GlobalTensor<uint64_t> hcommHandleGm;
    hcommHandleGm.SetGlobalBuffer((__gm__ uint64_t *)&ctxPtr_->hcommHandle_[0]);
    DataCopyExtParams cpHandle{1U, numRanks_ * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<uint64_t> padHandle{false, 0, 0, 0};
    DataCopyPad(hcommHandleLocal, hcommHandleGm, cpHandle, padHandle);

    SyncFunc<HardEvent::MTE2_S>();
}

__aicore__ inline void EngramFetchArch35::Process()
{
    if ASCEND_IS_AIV {
        uint32_t totalBlocks = GetBlockNum();

        CopyGmToUb();

        if (numEntriesPerRank_ == 0) {
            return;
        }

        LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
        LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
        LocalTensor<uint64_t> hcommHandleLocal = hcommHandleBuf_.Get<uint64_t>();
        LocalTensor<uint32_t> bucketCounts = countsBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> bucketOffsets = offsetsBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> bucketPos = bucketPosBuf_.Get<uint32_t>();

        for (uint32_t r = 0; r < numRanks_; r++) {
            bucketCounts(r) = 0;
        }
        for (uint32_t i = 0; i < numTokens_; i++) {
            int32_t globalIdx = indicesLocal(i);
            if (globalIdx < 0) {
                continue;
            }
            uint32_t gIdx = static_cast<uint32_t>(globalIdx);
            uint32_t ownerRank = gIdx / numEntriesPerRank_;
            if (ownerRank >= numRanks_) {
                continue;
            }
            bucketCounts(ownerRank) += 1;
        }

        bucketOffsets(0) = 0;
        for (uint32_t r = 0; r < numRanks_; r++) {
            bucketOffsets(r + 1) = bucketOffsets(r) + bucketCounts(r);
        }

        for (uint32_t r = 0; r < numRanks_; r++) {
            bucketCounts(r) = 0;
        }
        for (uint32_t i = 0; i < numTokens_; i++) {
            int32_t globalIdx = indicesLocal(i);
            if (globalIdx < 0) {
                continue;
            }
            uint32_t gIdx = static_cast<uint32_t>(globalIdx);
            uint32_t ownerRank = gIdx / numEntriesPerRank_;
            if (ownerRank >= numRanks_) {
                continue;
            }
            uint32_t pos = bucketOffsets(ownerRank) + bucketCounts(ownerRank);
            bucketPos(pos) = i;
            bucketCounts(ownerRank) += 1;
        }

        for (uint32_t ownerRank = aivId_; ownerRank < numRanks_; ownerRank += totalBlocks) {
            uint32_t idxStart = ownerRank * numEntriesPerRank_;
            uint32_t bucketStart = bucketOffsets(ownerRank);
            uint32_t bucketEnd = bucketOffsets(ownerRank + 1);
            uint32_t cnt = bucketEnd - bucketStart;

            uint32_t channelIdx = ownerRank;
            uint32_t pendingReadCount = 0;
            for (uint32_t b = 0; b < cnt; b++) {
                uint32_t i = bucketPos(bucketStart + b);
                int32_t globalIdx = indicesLocal(i);
                uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - idxStart;
                GM_ADDR dst = fetchedGM_ + static_cast<uint64_t>(i) * hiddenBytes_;

                if (ownerRank == rankId_) {
                    GM_ADDR src =
                        (GM_ADDR)commBufferLocal(rankId_) + static_cast<uint64_t>(localEntryIdx) * hiddenBytes_;
                    LocalCopySlice(dst, src, hiddenBytes_);
                } else {
                    GM_ADDR remoteSrcAddr =
                        (GM_ADDR)commBufferLocal(ownerRank) + static_cast<uint64_t>(localEntryIdx) * hiddenBytes_;
                    hcomm_.ReadNbi<false>(hcommHandleLocal(channelIdx), dst, remoteSrcAddr, hiddenBytes_);
                    pendingReadCount++;
                }
            }

            if (pendingReadCount > 0) {
                hcomm_.Commit(hcommHandleLocal(channelIdx));
            }
        }
    }
}

__aicore__ inline void EngramFetchArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    LocalTensor<uint8_t> tmp0 = dataBuf0_.Get<uint8_t>();
    LocalTensor<uint8_t> tmp1 = dataBuf1_.Get<uint8_t>();
    GlobalTensor<uint8_t> srcGm;
    GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};

    uint64_t off = 0;
    uint32_t tileIdx = 0;
    while (off < len) {
        uint64_t thisLen = (len - off > TILE_BYTES) ? TILE_BYTES : (len - off);
        DataCopyExtParams cp{1U, static_cast<uint32_t>(thisLen), 0U, 0U, 0U};
        uint32_t bufIdx = tileIdx % 2U;
        LocalTensor<uint8_t> tmp = (bufIdx == 0U) ? tmp0 : tmp1;

        if (tileIdx >= 2U) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        }

        DataCopyPad(tmp, srcGm[off], cp, pad);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        DataCopyPad(dstGm[off], tmp, cp);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);

        off += thisLen;
        tileIdx++;
    }

    if (tileIdx >= 1U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[(tileIdx - 1U) % 2U]);
    }
    if (tileIdx >= 2U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[tileIdx % 2U]);
    }
}
} // namespace EngramFetchImpl

#endif