/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_to_attention_v2_tilling_key.h
 * \brief
 */
#ifndef FFN_TO_ATTENTION_URMA_H
#define FFN_TO_ATTENTION_URMA_H
 
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ffn_to_attention_v2_tiling.h"
#include "adv_api/hcomm/hcomm.h"
// ... 其他 common include ...
 
namespace FFNToAttentionImpl {
 
#define TemplateFFNToAttentionUrmaTypeClass typename xType, bool isInputRankTable
#define TemplateFFNToAttentionUrmaTypeFunc xType, isInputRankTable
 
template <TemplateFFNToAttentionUrmaTypeClass>
class FFNToAttentionUrma {
public:
    __aicore__ inline FFNToAttentionUrma(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR sessionIds, GM_ADDR microBatchIds,
                                GM_ADDR tokenIds, GM_ADDR expertOffsets, GM_ADDR actualTokenNum,
                                GM_ADDR attnRankTable, GM_ADDR workspaceGM, TPipe *pipe,
                                const FFNToAttentionV2TilingData *tilingData);
    __aicore__ inline void Process();
 
private:
    // ====== 从 UB-MEM 版本完整复制的函数（逻辑不变） ======
    __aicore__ inline void ReadTokenMetaData(ReadTokenMetaDataStruct &metaDataStruct, uint32_t YOffset);
 
    // ====== 需要改造的函数 ======
    __aicore__ inline GM_ADDR GetWindowAddr(uint32_t curAttenWorkRank);
 
    // ====== URMA 新增函数 ======
    __aicore__ inline void HcommInit();
 
    // ====== 成员变量 ======
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    GM_ADDR workspaceGM_;
    uint32_t rankId_{0};  // 新增：当前卡 rankId（UB-MEM 版本未显式保存）
 
    // --- 与 UB-MEM 版本相同的成员 ---
    TPipe *tpipe_{nullptr};
    GlobalTensor<xType> xGMTensor_;
    // ... 其他成员 ...
    __gm__ Mc2Aclnn::AttentionFFNContext *mc2Context_{nullptr};
};
 
} // namespace FFNToAttentionImpl
#endif

// 在 URMA 模板中内联
__aicore__ inline uint64_t GetUrmaCommHandle(uint32_t dstRank, uint32_t srcRank)
{
    uint32_t index = dstRank > srcRank ? dstRank - 1 : dstRank;
    return mc2Context_->hcommHandle_[index];
}


template <TemplateFFNToAttentionUrmaTypeClass>
__aicore__ inline void FFNToAttentionUrma<TemplateFFNToAttentionUrmaTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        // URMA: 初始化 hcomm
        HcommInit();
 
        uint32_t calStartTokenId = 0;
        uint32_t sendTokenNum = actualTokenNum_ / aivNum_;
        uint32_t remainderRankNum = actualTokenNum_ % aivNum_;
        if (aivId_ < remainderRankNum) {
            sendTokenNum += 1;
            calStartTokenId += aivId_;
        } else {
            calStartTokenId += remainderRankNum;
        }
        if (calStartTokenId >= actualTokenNum_) {
            sendTokenNum = 0;
        }
 
        GM_ADDR curRankWinAddr;
        GlobalTensor<xType> tokenDataGMTensor;
        GlobalTensor<int32_t> tokenInfoTableGMTensor;
        DataCopyExtParams dataCopyParams = {1U, sizeof(int32_t), 0U, 0U, 0U};
        DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(axisH_ * sizeof(xType)), 0U, 0U, 0U};
        DataCopyPadExtParams<xType> copyPadExtParams{false, 0U, 0U, 0U};
 
        for (uint32_t tokenCnt = 0; tokenCnt < sendTokenNum; ++tokenCnt) {
            uint32_t YOffset = aivId_ + (tokenCnt * aivNum_);
 
            // 1. 加载 token（不变）
            xTmpTensor_ = xQueue_.AllocTensor<xType>();
            DataCopyPad(xTmpTensor_, xGMTensor_[YOffset * axisH_], xCopyParams, copyPadExtParams);
 
            // 2. 计算偏移数据（不变）
            ReadTokenMetaDataStruct metaDataStruct;
            ReadTokenMetaData(metaDataStruct, YOffset);
 
            // 3. 计算远端 win 区地址（不变）
            curRankWinAddr = GetWindowAddr(metaDataStruct.curAttenWorkRank);
 
            // 4. 计算 info 和 data 在 win 区的地址偏移（不变）
            uint64_t tokenInfoTableOffset = metaDataStruct.curMicroBatchIds * batchSizeSendCnt_;
            uint64_t tokenDataOffset =
                (metaDataStruct.curMicroBatchIds * perMicroBatchOffset_ +
                 metaDataStruct.curTokenBatchOffset * perExpertOffset_ +
                 metaDataStruct.curTokenTopkOffset * axisHS_) * sizeof(xType);
            GM_ADDR remoteDataAddr = curRankWinAddr + winTokenInfoTableSize_ + tokenDataOffset;
            GM_ADDR remoteFlagAddr = curRankWinAddr +
                (tokenInfoTableOffset + metaDataStruct.curTokenBatchOffset * expertNumPerToken_ +
                 metaDataStruct.curTokenTopkOffset) * sizeof(int32_t);
 
            // 5. UB→GM 中转（hcomm_.WriteNbi 源地址必须是 GM）
            if (tokenCnt == 0) {
                SyncFunc<AscendC::HardEvent::S_MTE3>();
            }
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<xType>();
            GM_ADDR localTmpGm = workspaceGM_ + aivId_ * hSize_;
            GlobalTensor<xType> localTmpGmTensor;
            localTmpGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(localTmpGm));
            DataCopyPad(localTmpGmTensor, xTmpTensor_, xCopyParams);
            xQueue_.FreeTensor<xType>(xTmpTensor_);
            SyncFunc<AscendC::HardEvent::MTE3_S>();  // 等本地 GM 写完
 
            // 6. 通过 URMA 发送 token + flag 到远端 win
            uint64_t channel = GetUrmaCommHandle(mc2Context_, metaDataStruct.curAttenWorkRank, rankId_);
            hcomm_.WriteWithNotifyNbi(channel, remoteDataAddr, localTmpGm,
                                      static_cast<int64_t>(axisH_ * sizeof(xType)),
                                      remoteFlagAddr, uint64_t(1));
        }
 
        // URMA: 冲刷所有使用过的通道
        DrainUsedChannels();
    }
}


template <bool RankTableMode, uint8_t ArchTag, uint8_t CommModeType>
__global__ __aicore__ void ffn_to_attention_v2(...)
{
    REGISTER_TILING_DEFAULT(FFNToAttentionV2TilingData);
    TPipe pipe;
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        GET_TILING_DATA_WITH_STRUCT(FFNToAttentionV2TilingData, tilingData, tilingGM);
        if constexpr (CommModeType == TILINGKEY_COMM_MTE) {
            // 现有 UB-MEM 路径（不变）
            FFNToAttention<DTYPE_X, RankTableMode, FFNToAttentionContextType::MC2_CONTEXT> op;
            op.Init(...);
            op.Process();
        } else if constexpr (CommModeType == TILINGKEY_COMM_URMA) {
            // 新增 URMA 路径
            FFNToAttentionUrma<DTYPE_X, RankTableMode> op;
            op.Init(mc2Context, x, sessionIds, microBatchIds, tokenIds, expertOffsets,
                    actualTokenNum, attnRankTable, workspaceGM, &pipe, &tilingData);
            op.Process();
        }
    }
}