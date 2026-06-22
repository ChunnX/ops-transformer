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
    uint32_t hiddenDim_{0};
    uint32_t hiddenBytes_{0};
    uint32_t aivNum_{0};

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
    // if (GetBlockIdx() > 0) {
    //     return;
    // }
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
    hiddenDim_ = tilingData_->hiddenDim;
    hiddenBytes_ = tilingData_->hiddenBytes;
    aivNum_ = tilingData_->aivNum;

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(dataBuf_, TILE_BYTES);
    uint32_t indicesBufSize = Ceil(numTokens_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
    if (GetBlockIdx() == 0) {
        AscendC::printf("[kernel] rankId_: %lu\n", rankId_);
        AscendC::printf("[kernel] numRanks_: %lu\n", numRanks_);
        for (int i = 0; i < 8; ++i) {
          AscendC::printf("[kernel] rankid: %lu, bhPtr_->commBuffer[%d]: %lu\n", 
                    rankId_, i, bhPtr_->commBuffer[i]);
      }

        AscendC::printf("[kernel] rankid: %lu, bhPtr_->hcommHandle_[0]: %lu\n", rankId_, bhPtr_->hcommHandle_[0]);
        AscendC::printf("[kernel] rankid: %lu, bhPtr_->hcommHandle_[1]: %lu\n", rankId_, bhPtr_->hcommHandle_[1]);
        AscendC::printf("[kernel] rankid: %lu, bhPtr_->hcommHandle_[2]: %lu\n", rankId_, bhPtr_->hcommHandle_[2]);
        AscendC::printf("[kernel] rankid: %lu, bhPtr_->hcommHandle_[3]: %lu\n", rankId_, bhPtr_->hcommHandle_[3]);
        AscendC::printf("[kernel] fetched: %lu\n", (uint64_t)fetchedGM_);
    }
}

__aicore__ inline void EngramFetchArch35::Process()
{
    if ASCEND_IS_AIV {
        if (aivId_ >= numRanks_) {
            return;
        }
        uint32_t ownerRank = aivId_;

        // DMA: 全部 indices 从 GM 搬入 UB
        GlobalTensor<int32_t> indicesGlobal;
        indicesGlobal.SetGlobalBuffer((__gm__ int32_t *)indicesGM_);
        LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
        DataCopyExtParams params{1U, numTokens_ * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(indicesLocal, indicesGlobal, params, pad);
        PipeBarrier<PIPE_ALL>();
        // PipeBarrier<PIPE_MTE2>();
        uint32_t rankStart = ownerRank * numEntriesPerRank_;
        uint32_t rankEnd = rankStart + numEntriesPerRank_;

        for (uint32_t i = 0; i < numTokens_; i++) {
            int32_t globalIdx = indicesLocal(i);
            if (globalIdx < 0) {
                continue;
            }
            if (static_cast<uint32_t>(globalIdx) < rankStart ||
                static_cast<uint32_t>(globalIdx) >= rankEnd) {
                continue;
            }

            uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - rankStart;
            GM_ADDR dst = fetchedGM_ + static_cast<uint64_t>(i) * hiddenBytes_;

            if (ownerRank == rankId_) {
                GM_ADDR src = (GM_ADDR)bhPtr_->commBuffer[rankId_] +
                              static_cast<uint64_t>(localEntryIdx) * hiddenBytes_;
                PipeBarrier<PIPE_ALL>();
                LocalCopySlice(dst, src, hiddenBytes_);
                PipeBarrier<PIPE_ALL>();
            } 
            else {
                AscendC::printf("[aiv]ssssss\n");
                PipeBarrier<PIPE_ALL>();
                // uint32_t channelIdx = ownerRank > rankId_ ? ownerRank - 1 : ownerRank;
                uint32_t channelIdx = ownerRank;
                AscendC::printf("[kernel] channelIdx: %u\n", channelIdx);
                GM_ADDR remoteSrcAddr =
                    (GM_ADDR)bhPtr_->commBuffer[ownerRank] +
                    static_cast<uint64_t>(localEntryIdx) * hiddenBytes_;
                PipeBarrier<PIPE_ALL>();
                AscendC::printf("[kernel] rankid: %lu, bhPtr_->hcommHandle_[channelIdx]: %lu, dst = %lu ,remoteSrcAddr = %lu \n", rankId_, bhPtr_->hcommHandle_[channelIdx], (uint64_t)dst, (uint64_t)remoteSrcAddr);
                hcomm_.ReadNbi<true>(bhPtr_->hcommHandle_[channelIdx], dst, remoteSrcAddr, hiddenBytes_);
                PipeBarrier<PIPE_ALL>();
                hcomm_.Drain(bhPtr_->hcommHandle_[channelIdx]);
                PipeBarrier<PIPE_ALL>();
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
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(tmp, srcGm[off], cp, pad);
        PipeBarrier<PIPE_ALL>();


        //  if (off == 0) {
        //     printf("=== SRC dump src: %lu dst %lu ===\n", (uint64_t)src,(uint64_t)dst);
        //     LocalTensor<float16_t> tmpBf16 = tmp.ReinterpretCast<float16_t>();
        //     uint32_t numBf16Elems = static_cast<uint32_t>(thisLen) / 2;
        //     printf("=== SRC dump (off=0, %u bf16 elems) ===\n", numBf16Elems);
        //     for (uint32_t j = 0; j < numBf16Elems && j < 10; j++) {
        //         printf("src[%u] = %f\n", j, static_cast<float>(tmpBf16(j)));
        //     }
        // }

        //PipeBarrier<PIPE_MTE2>();
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(dstGm[off], tmp, cp);
        //PipeBarrier<PIPE_MTE3>();
        PipeBarrier<PIPE_ALL>();

        //  if (off == 0) {
        //     DataCopyPad(tmp, dstGm[off], cp, pad);
        //     PipeBarrier<PIPE_MTE2>();
        //     LocalTensor<float16_t> dstBf16 = tmp.ReinterpretCast<float16_t>();
        //     uint32_t numBf16Elems = static_cast<uint32_t>(thisLen) / 2;
        //      printf("=== DST dump src: %lu dst %lu ===\n", (uint64_t)src,(uint64_t)dst);
        //     printf("=== DST dump after copy (off=0, %u bf16 elems) ===\n", numBf16Elems);
        //     for (uint32_t j = 0; j < numBf16Elems && j < 10; j++) {
        //         printf("dst[%u] = %f\n", j, static_cast<float>(dstBf16(j)));
        //     }
        // }
        off += thisLen;
    }
}
} // namespace EngramFetchImpl

#endif