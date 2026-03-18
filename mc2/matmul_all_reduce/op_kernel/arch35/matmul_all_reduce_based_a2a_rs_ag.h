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
 * \file matmul_all_reduce_based_a2a_rs_ag.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_BASED_A2A_RS_AG_H
#define MATMUL_ALL_REDUCE_BASED_A2A_RS_AG_H

#include "matmul_all_reduce_base.h"
namespace MatmulAllReduceImpl {
using namespace GmUbGmCopyImpl;
using namespace AiVReduceSumCastFp32Impl;
constexpr uint32_t A2A_VSUM_AG_MAX_HANDLE_ID_NUM = 16;
template <typename XType, typename YType, Mc2CoreType CoreType>
class MatmulAllReduceBase<XType, YType, CoreType, true>
{
public:
    __aicore__ inline MatmulAllReduceBase(
        MC2GmAddrs* addrs, QuantGmAddrs* quantAddrs, ArnGmAddrs* arnAddrs, MC2TilingHeader* tilingData, TPipe* tPipe)
        : addrs_(addrs), quantAddrs_(quantAddrs), arnAddrs_(arnAddrs), tilingData_(tilingData), tPipe_(tPipe)
    {
        paramInTiling_ = &tilingData->param;
        rankNum_ = paramInTiling_->rankDim;
    }

    __aicore__ inline void Init()
    {
        notifyFlag_ = (g_coreType == AscendC::AIV && GetBlockIdx() == 0);
        hccl_.InitV2(GetHcclContext<0>(), tilingData_);
        hccl_.SetCcTilingV2(offsetof(MC2TilingHeader, mc2CcTiling));
        hccl_.SetCcTilingV2(offsetof(MC2TilingHeader, mc2CcTilingComm));

        __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
        OOMInit(context);
#if defined(__DAV_C310__)
        addrs_->cGM = addrs_->workspaceGM + paramInTiling_->nd2NzWorkLen + paramInTiling_->biasLen;
#else
        if (msgInTiling_->useBufferType == MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN &&
            context->config.determinism != 1) {
            addrs_->cGM = hccl_.GetWindowsInAddr(hccl_.GetRankId());
        }
#endif
        tailFlag_ = (paramInTiling_->tailCnt != 0U);
        addFlag_ = (paramInTiling_->isAdd != 0U);
        isOneTileFlag_ = (paramInTiling_->tileCnt==1U) && (paramInTiling_->tailCnt==0U);

        const uint64_t mVal = isOneTileFlag_ ? ((uint64_t)paramInTiling_->rankM) : (uint64_t)tileInfo_.mmTiling->M;
        tileInfo_.aOffset = mVal * (uint64_t)tileInfo_.mmTiling->Ka;
        tileInfo_.aAddrOffset = tileInfo_.aOffset * sizeof(XType);
        if (AscendC::IsSameType<XType, fp4x2_e2m1_t>::value) {
            // In 4-bits scenario, the data length is 0.5, the size of Xtype is 1, it should be divided by 2.
            tileInfo_.aAddrOffset = tileInfo_.aAddrOffset / 2;
        }
        tileInfo_.cOffset = mVal * (uint64_t)tileInfo_.mmTiling->N;
        tileInfo_.cAddrOffset = tileInfo_.cOffset * sizeof(YType);
        if (tailFlag_) {
            tailInfo_.aOffset = (uint64_t)tailInfo_.mmTiling->M * (uint64_t)tailInfo_.mmTiling->Ka;
            tailInfo_.aAddrOffset = tailInfo_.aOffset * sizeof(XType);
            if (AscendC::IsSameType<XType, fp4x2_e2m1_t>::value || AscendC::IsSameType<XType, fp4x2_e1m2_t>::value) {
                // In 4-bits scenario, the data length is 0.5, the size of Xtype is 1, it should be divided by 2.
                tailInfo_.aAddrOffset = tailInfo_.aAddrOffset / 2;
            }
            tailInfo_.cOffset = (uint64_t)tailInfo_.mmTiling->M * (uint64_t)tailInfo_.mmTiling->N;
            tailInfo_.cAddrOffset = tailInfo_.cOffset * sizeof(YType);
        }
        CalcNeededPad(mVal, (uint64_t)tailInfo_.mmTiling->M);
        CalcGMAddr();
    }

    __aicore__ inline void CalcGMAddr()
    {
        uint64_t tileAndTailNum = paramInTiling_->tileCnt + paramInTiling_->tailCnt;
        uint64_t padLen = tileAndTailNum * rankNum_;
        uint64_t padAddrSize = padLen * sizeof(YType);
        cgmAddr_ = tileInfo_.cAddrOffset * paramInTiling_->tileCnt + tailInfo_.cAddrOffset * paramInTiling_->tailCnt;
        cgmLen_ = tileInfo_.cOffset * paramInTiling_->tileCnt + tailInfo_.cOffset * paramInTiling_->tailCnt;
        all2allInGM_ = addrs_->cGM;
        all2allOutGM_ = all2allInGM_ + cgmAddr_ + padAddrSize;       // MM结果
        reduceSumInGM_ = all2allOutGM_;
        reduceSumOutGM_ = reduceSumInGM_ + cgmAddr_ + padAddrSize;   // alltoall结果
        allgatherInGM_ = reduceSumOutGM_;
        if (!needPad_) {         // 是否需要内存拷贝
            allgatherOutGM_ = addrs_->outputGM;
        } else {
            allgatherOutGM_ = reduceSumOutGM_ + (cgmAddr_ + padAddrSize) / rankNum_; // reduceSum结果
        }
        PrePareHCCL();
    }

    __aicore__ inline void PrePareHCCL()
    {
        for (uint32_t i = 0U; i < paramInTiling_->tileCnt; i++){
            uint64_t indexOffsetTile = tileInfo_.cAddrOffset * i;
            uint64_t alignedIndexOffsetTile = CeilAlign(tileInfo_.cOffset, rankNum_) * sizeof(YType) * i;
            all2allSendGM_[i] = all2allInGM_ + indexOffsetTile;                         // all2allSendBuff 切块之间是连续的
            all2allRecvGM_[i] = all2allOutGM_ + alignedIndexOffsetTile;                 // all2allRecvBuff 切块之间是非连续的
            allgatherSendGM_[i] = allgatherInGM_ + alignedIndexOffsetTile / rankNum_;   // allgatherInBuff 切块之间是非连续的
            allgatherRecvGM_[i] = allgatherOutGM_ + indexOffsetTile;                    // allgatherOutBuff 切块之间是连续的
            uint64_t ceilDataCount = CeilDiv(tileInfo_.cOffset, rankNum_);

            all2allHandleId_[i] = hccl_.AlltoAll<false>(
                all2allSendGM_[i], all2allRecvGM_[i], ceilDataCount, HCCL_DATA_TYPE);
        }

        for (uint32_t i = 0U; i < paramInTiling_->tailCnt; i++){
            uint64_t indexOffsetTail = tileInfo_.cAddrOffset * paramInTiling_->tileCnt + tailInfo_.cAddrOffset * i;
            uint64_t alignedIndexOffsetTail = CeilAlign(tileInfo_.cOffset, rankNum_) * sizeof(YType) * paramInTiling_->tileCnt 
                                            + CeilAlign(tailInfo_.cOffset, rankNum_) * sizeof(YType) * i;
            uint64_t index = paramInTiling_->tileCnt + i;
            all2allSendGM_[index] = all2allInGM_ + indexOffsetTail;                             // all2allSendBuff 切块之间是连续的
            all2allRecvGM_[index] = all2allOutGM_ + alignedIndexOffsetTail;                     // all2allRecvBuff 切块之间是非连续的
            allgatherSendGM_[index] = allgatherInGM_ + alignedIndexOffsetTail / rankNum_;       // allgatherInBuff 切块之间是非连续的
            allgatherRecvGM_[index] = allgatherOutGM_ + indexOffsetTail;                        // allgatherOutBuff 切块之间是连续的
            uint64_t ceilDataCount = CeilDiv(tailInfo_.cOffset, rankNum_);
            
            all2allHandleId_[index] = hccl_.AlltoAll<false>(
                all2allSendGM_[index], all2allRecvGM_[index], ceilDataCount, HCCL_DATA_TYPE);
        }

        for (uint32_t i = 0U; i < paramInTiling_->tileCnt; i++){
            uint64_t ceilDataCount = CeilDiv(tileInfo_.cOffset, rankNum_);
            allgatherHandleId_[i] = hccl_.AllGather<false>(
                allgatherSendGM_[i], allgatherRecvGM_[i], ceilDataCount, HCCL_DATA_TYPE, 0, 1);
        }

        for (uint32_t i = 0U; i < paramInTiling_->tailCnt; i++){
            const uint64_t index = paramInTiling_->tileCnt + i;
            uint64_t ceilDataCount = CeilDiv(tailInfo_.cOffset, rankNum_);
            allgatherHandleId_[index] = hccl_.AllGather<false>(
                allgatherSendGM_[index], allgatherRecvGM_[index], ceilDataCount, HCCL_DATA_TYPE, 0, 1);
        }
    }

protected:
    __aicore__ inline void CalcNeededPad(uint64_t mTileValue, uint64_t mTailValue)
    {
        uint32_t tilePad = (rankNum_ - (mTileValue % rankNum_)) % rankNum_;
        uint32_t tailPad = (rankNum_ - (mTailValue % rankNum_)) % rankNum_;
        needPad_ = (tilePad != 0 || tailPad != 0);
    }

    __aicore__ inline void PostProcEachTurn(AscendC::HcclHandle handleId, uint64_t aOffset, uint64_t cOffset, uint64_t index = 0)
    {
        if (addFlag_ && addrs_->cGM != addrs_->addGM) {
            SyncAll<false>();
            MatmulAllReduceAddX3Kernel<YType>(
                addrs_->cGM, addrs_->addGM, cOffset / sizeof(YType), paramInTiling_->addX3UbCnt, tPipe_);
            addrs_->addGM += cOffset;
        }

        addrs_->aGM += aOffset;
        addrs_->cGM += cOffset;
        SyncAll<false>();
        if (notifyFlag_) {
            hccl_.Commit(all2allHandleId_[index]);
        }
    }
    __aicore__ inline void WaitAlltoAllEachTurn(bool tailFlag, uint32_t turnCnt){
        if (notifyFlag_) {
            for (uint32_t i = 0U; i < turnCnt; ++i) {
                const uint64_t index = tailFlag ? i + paramInTiling_->tileCnt : i;
                hccl_.Wait(all2allHandleId_[index]);
            }
        }
        SyncAll();
    }

    __aicore__ inline void ReduceSumAndAllGather()
    {
        if ASCEND_IS_AIV {
            // ReduceSum计算
            uint64_t aivNum = GetBlockNum() * GetTaskRation();
            for (int i = 0; i < paramInTiling_->tileCnt; i++){
                tPipe_->Reset();
                uint64_t ceilDataCount = CeilDiv(tileInfo_.cOffset, rankNum_);
                reduceSum_.Init(ceilDataCount, 0, rankNum_, aivNum, reduceSumInGM_, reduceSumOutGM_, tPipe_);
                reduceSum_.ExecuteReduceSum();
                reduceSumInGM_ += CeilAlign(tileInfo_.cOffset, rankNum_) * sizeof(YType);               // reduceSumIn 切块之间是非连续的
                reduceSumOutGM_ += CeilAlign(tileInfo_.cOffset, rankNum_) * sizeof(YType) / rankNum_;   // reduceSumOut 切块之间是非连续的
            }

            for (int i = 0; i < paramInTiling_->tailCnt; i++){
                tPipe_->Reset();
                uint64_t ceilDataCount = CeilDiv(tailInfo_.cOffset, rankNum_);
                reduceSum_.Init(ceilDataCount, 0, rankNum_, aivNum, reduceSumInGM_, reduceSumOutGM_, tPipe_);
                reduceSum_.ExecuteReduceSum();
                reduceSumInGM_ += CeilAlign(tailInfo_.cOffset, rankNum_) * sizeof(YType);
                reduceSumOutGM_ += CeilAlign(tailInfo_.cOffset, rankNum_) * sizeof(YType) / rankNum_;
            }
        }
        SyncAll();
        if (notifyFlag_) {
            for (int i = 0; i < paramInTiling_->tileCnt + paramInTiling_->tailCnt; i++) {
                hccl_.Commit(allgatherHandleId_[i]);
            }
        }
    }

    __aicore__ inline void HcclFinalize()
    {
        if (notifyFlag_) {
            for (int i = 0; i < paramInTiling_->tileCnt + paramInTiling_->tailCnt; i++) {
                hccl_.Wait(allgatherHandleId_[i]);
            }
        }

        if (needPad_){
            if ASCEND_IS_AIV {
                uint64_t aivNum = GetBlockNum() * GetTaskRation();
                // DataCopy
                SyncAll();
                tPipe_->Reset();
                dataCopy_.Init(cgmLen_, aivNum, allgatherOutGM_, addrs_->outputGM, tPipe_);
                dataCopy_.Process();
                SyncAll();
            }
        }
        if (notifyFlag_) {
            hccl_.Finalize();
        }
    }
    uint32_t rankNum_ = 0UL;
    uint64_t cgmLen_ = 0UL;
    uint64_t cgmAddr_ = 0UL;
    QuantGmAddrs* quantAddrs_;
    MC2GmAddrs* addrs_;
    ArnGmAddrs* arnAddrs_;
    Mc2Tiling::RCSTiling* paramInTiling_;
    Mc2Tiling::Mc2Msg* msgInTiling_;
    MC2TilingHeader* tilingData_;
    MC2TileInfo tileInfo_, tailInfo_;
    TPipe* tPipe_;
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
    bool notifyFlag_;
    bool tailFlag_;
    bool isOneTileFlag_;
    bool addFlag_;
    bool needPad_;
    
    AscendC::HcclHandle all2allHandleId_[A2A_VSUM_AG_MAX_HANDLE_ID_NUM] = {0};
    AscendC::HcclHandle allgatherHandleId_[A2A_VSUM_AG_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR all2allSendGM_[A2A_VSUM_AG_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR all2allRecvGM_[A2A_VSUM_AG_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allgatherSendGM_[A2A_VSUM_AG_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allgatherRecvGM_[A2A_VSUM_AG_MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR all2allInGM_;
    GM_ADDR all2allOutGM_;
    GM_ADDR reduceSumInGM_;
    GM_ADDR reduceSumOutGM_;
    GM_ADDR allgatherInGM_;
    GM_ADDR allgatherOutGM_;
private:
    ReduceSumForAlltoAll<YType> reduceSum_;     // AIV ReduceSum相关实现
    GmUbGmCopy<YType> dataCopy_;        // dataCopy实现
};
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_BASED_A2A_RS_AG_H