/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.hiascend.com/license/
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*!
 * \file engram_fetch_infer_base_arch35.h
 * \brief engram_fetch算子arch35推理模式kernel公共基类
 * 封装 sliced(切片跨NPU) 与 local(全量本NPU) 两种推理kernel的共同逻辑:
 * 通信上下文/tiling解析、indices分批加载、行级GM拷贝、sf gather等
 */

#ifndef ENGRAM_FETCH_INFER_BASE_ARCH35_H
#define ENGRAM_FETCH_INFER_BASE_ARCH35_H

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

namespace Mc2Kernel {

#if defined(ENABLE_ENGRAM_FETCH_KERNEL)

template <AscendC::HardEvent event>
__aicore__ inline void EngramFetchSyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

class EngramFetchInferBase {
public:
    __aicore__ inline EngramFetchInferBase() = default;

protected:
    // 推理模式公共初始化: GM输入指针 + 通信上下文解析 + tiling解析 + relay队列
    __aicore__ inline void InitInferBase(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR sfTable,
                                         GM_ADDR fetchedSf, AscendC::TPipe *pipe,
                                         const EngramFetchTilingData *tilingData)
    {
        tpipe_ = pipe;
        indicesGM_ = indices;
        fetchedGM_ = fetched;
        fetchedSfGM_ = fetchedSf;
        sfTableGM_ = reinterpret_cast<__gm__ uint8_t *>(sfTable);
        aivId_ = AscendC::GetBlockIdx();

        ctxPtr_ = (__gm__ EngramCommContext *)commContext;
        rankId_ = ctxPtr_->rankId;
        numRanks_ = ctxPtr_->rankSize;

        numEntriesPerRank_ = tilingData->numEntriesPerRank;
        numTokens_ = tilingData->numTokens;
        hiddenBytes_ = tilingData->hiddenBytes;
        ubSize_ = tilingData->ubSize;
        numSfPacks_ = tilingData->numSfPacks;
        sfElemSize_ = tilingData->sfElemSize;

        tpipe_->InitBuffer(relayQue_, RELAY_BUFFER_NUM, TILE_BYTES);
    }

    // 按各模式已占用的UB(usedUb)与每index所需字节数(bytesPerIndice)计算indices分批大小并分配缓冲
    __aicore__ inline void InitIndicesBuffer(uint64_t usedUb, uint32_t bytesPerIndice)
    {
        constexpr uint64_t ubReserved = 8U * 1024U;
        uint64_t availableUb = (ubSize_ > usedUb + ubReserved) ? (ubSize_ - usedUb - ubReserved) : 0U;
        uint32_t maxBatchSize = static_cast<uint32_t>(availableUb / bytesPerIndice);
        uint32_t numTokens = static_cast<uint32_t>(numTokens_);
        indicesBatchSize_ = (numTokens <= maxBatchSize) ? numTokens : maxBatchSize;
        if (indicesBatchSize_ == 0U) {
            indicesBatchSize_ = 1U;
        }
        uint32_t indicesBufSize = AscendC::Ceil(indicesBatchSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
    }

    __aicore__ inline bool HasSf() const
    {
        return (numSfPacks_ > 0 && sfTableGM_ != nullptr && fetchedSfGM_ != nullptr);
    }

    __aicore__ inline void CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen)
    {
        AscendC::GlobalTensor<int32_t> indicesGlobal;
        indicesGlobal.SetGlobalBuffer((__gm__ int32_t *)indicesGM_);
        AscendC::LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
        AscendC::DataCopyExtParams params{1U, indicesBatchLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(indicesLocal, indicesGlobal[indicesBatchStart], params, pad);
    }

    __aicore__ inline void GatherSf(int32_t globalIdx, uint64_t globalTokenIdx)
    {
        uint32_t sfBytes = static_cast<uint32_t>(numSfPacks_) * static_cast<uint32_t>(sfElemSize_);
        GM_ADDR sfSrc = sfTableGM_ + static_cast<uint64_t>(globalIdx) * sfBytes;
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

    // 行级 GM->UB->GM 中转拷贝(按 TILE_BYTES 分片)
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
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
            AscendC::DataCopyExtParams mte2Params{1U, static_cast<uint32_t>(thisLen), 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<uint8_t> mte2Pad{false, 0, 0, 0};
            AscendC::DataCopyPad(tmp, srcGm[off], mte2Params, mte2Pad);
            relayQue_.EnQue<uint8_t>(tmp);
            tmp = relayQue_.DeQue<uint8_t>();
            AscendC::DataCopyParams mte3Params{1U, static_cast<uint16_t>(thisLen), 0U, 0U};
            AscendC::DataCopyPad(dstGm[off], tmp, mte3Params);
            relayQue_.FreeTensor<uint8_t>(tmp);

            off += thisLen;
        }
    }

    AscendC::TPipe *tpipe_{nullptr};
    GM_ADDR indicesGM_{nullptr};
    GM_ADDR fetchedGM_{nullptr};
    __gm__ uint8_t *sfTableGM_{nullptr};
    GM_ADDR fetchedSfGM_{nullptr};
    __gm__ EngramCommContext *ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    int32_t numEntriesPerRank_{0};
    int64_t numTokens_{0};
    int64_t hiddenBytes_{0};
    int64_t numSfPacks_{0};
    int64_t sfElemSize_{0};
    uint64_t ubSize_{0};
    uint32_t indicesBatchSize_{0};

    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, RELAY_BUFFER_NUM> relayQue_;
    AscendC::TBuf<> indicesBuf_;
};

#endif

} // namespace Mc2Kernel

#endif
