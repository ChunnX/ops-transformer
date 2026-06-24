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

    __aicore__ inline void Init(GM_ADDR bufferHandle, GM_ADDR indices, GM_ADDR fetched, GM_ADDR workspaceGM,
                                TPipe *pipe, const EngramFetchTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);

    TPipe *tpipe_{nullptr};
    GM_ADDR indicesGM_{nullptr};
    GM_ADDR fetchedGM_{nullptr};
    const EngramFetchTilingData *tilingData_{nullptr};
    __gm__ EngramBufferHandle *bhPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    uint32_t numEntriesPerRank_{0};
    uint32_t numTokens_{0};
    uint32_t hiddenBytes_{0};

    TBuf<> dataBuf_;
    TBuf<> indicesBuf_;
    TBuf<> hcommBuf_;
    LocalTensor<uint8_t> hcommTensor_;

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
};

__aicore__ inline void EngramFetchArch35::Init(
    GM_ADDR bufferHandle, GM_ADDR indices, GM_ADDR fetched, GM_ADDR workspaceGM,
    TPipe *pipe, const EngramFetchTilingData *tilingData)
{
    tpipe_ = pipe;
    indicesGM_ = indices;
    fetchedGM_ = fetched;
    tilingData_ = tilingData;
    aivId_ = GetBlockIdx();

    bhPtr_ = (__gm__ EngramBufferHandle *)bufferHandle;
    rankId_ = bhPtr_->rankId;
    numRanks_ = bhPtr_->rankSize;

    numEntriesPerRank_ = tilingData_->numEntriesPerRank;
    numTokens_ = tilingData_->numTokens;
    hiddenBytes_ = tilingData_->hiddenBytes;

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(dataBuf_, TILE_BYTES);
    uint32_t indicesBufSize = Ceil(numTokens_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
    
}

__aicore__ inline void EngramFetchArch35::Process()
{
    if ASCEND_IS_AIV {
        uint32_t totalBlocks = GetBlockNum();

        GlobalTensor<int32_t> indicesGlobal;
        indicesGlobal.SetGlobalBuffer((__gm__ int32_t *)indicesGM_);
        LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
        DataCopyExtParams params{1U, numTokens_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(indicesLocal, indicesGlobal, params, pad);
        SyncFunc<HardEvent::MTE2_S>();

        for (uint32_t ownerRank = aivId_; ownerRank < numRanks_; ownerRank += totalBlocks) {
            uint32_t idxStart = ownerRank * numEntriesPerRank_;
            uint32_t idxEnd = idxStart + numEntriesPerRank_;

            uint32_t pendingReadCount = 0;
            uint32_t channelIdx = ownerRank;
            for (uint32_t i = 0; i < numTokens_; i++) {
                int32_t globalIdx = indicesLocal(i);
                if (globalIdx < 0) {
                    continue;
                }
                if (static_cast<uint32_t>(globalIdx) < idxStart || static_cast<uint32_t>(globalIdx) >= idxEnd) {
                    continue;
                }

                uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - idxStart;
                GM_ADDR dst = fetchedGM_ + static_cast<uint64_t>(i) * hiddenBytes_;

                if (ownerRank == rankId_) {
                    GM_ADDR src =
                        (GM_ADDR)bhPtr_->commBuffer[rankId_] + static_cast<uint64_t>(localEntryIdx) * hiddenBytes_;
                    LocalCopySlice(dst, src, hiddenBytes_);
                } else {
                    GM_ADDR remoteSrcAddr =
                        (GM_ADDR)bhPtr_->commBuffer[ownerRank] + static_cast<uint64_t>(localEntryIdx) * hiddenBytes_;
                    hcomm_.ReadNbi<false>(bhPtr_->hcommHandle_[channelIdx], dst, remoteSrcAddr, hiddenBytes_);
                    pendingReadCount++;
                }
            }

            if (pendingReadCount > 0) {
                hcomm_.Commit(bhPtr_->hcommHandle_[channelIdx]);
            }
        }
    }
}

__aicore__ inline void EngramFetchArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    LocalTensor<uint8_t> tmp = dataBuf_.Get<uint8_t>();
    GlobalTensor<uint8_t> srcGm;
    GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);
    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
    uint64_t off = 0;
    while (off < len) {
        uint64_t thisLen = (len - off > TILE_BYTES) ? TILE_BYTES : (len - off);
        DataCopyExtParams cp{1U, static_cast<uint32_t>(thisLen), 0U, 0U, 0U};
        DataCopyPad(tmp, srcGm[off], cp, pad);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyPad(dstGm[off], tmp, cp);
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        off += thisLen;
    }
}
} // namespace EngramFetchImpl

#endif