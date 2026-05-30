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
 * \file matmul_all_reduce_based_all_reduce.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_BASED_ALL_REDUCE_H
#define MATMUL_ALL_REDUCE_BASED_ALL_REDUCE_H
#include "matmul_all_reduce_base.h"
namespace MatmulAllReduceImpl {
template <typename XType, typename YType, Mc2CoreType CoreType, int commMode>
class MatmulAllReduceBase<XType, YType, CoreType, false, commMode>
{
public:
    __aicore__ inline MatmulAllReduceBase(
        MC2GmAddrs* addrs, QuantGmAddrs* quantAddrs, ArnGmAddrs* arnAddrs, MC2TilingHeader* tilDat, TPipe* tPipe)
        : addrs_(addrs), quantAddrs_(quantAddrs), arnAddrs_(arnAddrs), tilDat_(tilDat), tPipe_(tPipe)
    {
        if constexpr (CoreType == Mc2CoreType::ON_CUBE) {
            notifyFlag_ = (GetBlockIdx() == 0);
        } else {
            notifyFlag_ = (g_coreType == AscendC::AIV && GetBlockIdx() == 0);
        }
        paramCfg3_ = &tilDat->param;
    }

    __aicore__ inline void Init()
    {
        hccl_.InitV2(GetHcclContext<0>(), tilDat_);
        hccl_.SetCcTilingV2(offsetof(MC2TilingHeader, mc2CcTiling));
        __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
        OOMInit(context);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        addrs_->cGM = addrs_->workspaceGM + paramCfg3_->nd2NzWorkLen + paramCfg3_->biasLen;
#else
        if (msgInTiling_->useBufferType == MC2_BUFFER_TYPE::MC2_BUFFER_TYPE_WINDOW_IN &&
            context->config.determinism != 1) {
            addrs_->cGM = hccl_.GetWindowsInAddr(hccl_.GetRankId());
        }
#endif
        addFlag_ = (paramCfg3_->isAdd != 0U);
        tailFl3_ = (paramCfg3_->tailCnt != 0U);
        isOneTileFlag_ = (paramCfg3_->tileLoopCnt == 1U) && (paramCfg3_->tailCnt == 0U);
        const uint64_t mVal = isOneTileFlag_ ? ((uint64_t)paramCfg3_->rankM) : (uint64_t)tileInf5_.mmTil5->M;

        tileInf5_.aOff = mVal * (uint64_t)tileInf5_.mmTil5->Ka;
        if (AscendC::IsSameType<XType, fp4x2_e2m1_t>::value) {
            // In 4-bits scenario, the data length is 0.5, the size of Xtype is 1, it should be divided by 2.
            tileInf5_.aPtrOffset = tileInf5_.aOff * sizeof(XType) / 2;
        } else {
            tileInf5_.aPtrOffset = tileInf5_.aOff * sizeof(XType);
        }
        tileInf5_.cOff = mVal * (uint64_t)tileInf5_.mmTil5->N;
        tileInf5_.cPtrOffset = tileInf5_.cOff * sizeof(YType);
        if (tailFl3_) {
            tailInf5_.aOff = (uint64_t)tailInf5_.mmTil5->M * (uint64_t)tailInf5_.mmTil5->Ka;
            if (AscendC::IsSameType<XType, fp4x2_e2m1_t>::value || AscendC::IsSameType<XType, fp4x2_e1m2_t>::value) {
                // In 4-bits scenario, the data length is 0.5, the size of Xtype is 1, it should be divided by 2.
                tailInf5_.aPtrOffset = tailInf5_.aOff * sizeof(XType) / 2;
            } else {
                tailInf5_.aPtrOffset = tailInf5_.aOff * sizeof(XType);
            }
            tailInf5_.cOff = (uint64_t)tailInf5_.mmTil5->M * (uint64_t)tailInf5_.mmTil5->N;
            tailInf5_.cPtrOffset = tailInf5_.cOff * sizeof(YType);
        }

        if (notifyFlag_) {
            tileInf5_.hcclHandleId = hccl_.template AllReduce(
                addrs_->cGM, addrs_->outputGM, tileInf5_.cOff, HCCL_DATA_TYPE, AscendC::HCCL_REDUCE_SUM,
                paramCfg3_->tileLoopCnt);
            if (tailFl3_) {
                const uint64_t offset = tileInf5_.cPtrOffset * paramCfg3_->tileLoopCnt;
                tailInf5_.hcclHandleId = hccl_.template AllReduce(
                    addrs_->cGM + offset, addrs_->outputGM + offset, tailInf5_.cOff, HCCL_DATA_TYPE,
                    AscendC::HCCL_REDUCE_SUM, paramCfg3_->tailCnt);
            }
        }
    }

protected:
#if (ORIG_DTYPE_X1 == DT_BF16)
    __aicore__ inline void PreProcForBiasOnVector()
    {
        if (paramCfg3_->biasLen == 0U) {
            return;
        }

        TBuf<TPosition::VECCALC> tmpBuf;
        tPipe_->InitBuffer(tmpBuf, TOTAL_UB_SIZE);
        CastBFtoFloatOnAiv0(addrs_->workspaceGM, addrs_->biasGM, paramCfg3_->rankN, tmpBuf);
        SyncAll<false>();
        addrs_->biasGM = addrs_->workspaceGM;
        addrs_->workspaceGM += paramCfg3_->biasLen;
    }
#endif

    __aicore__ inline void PostProcEachTurn(AscendC::HcclHandle handleId, uint64_t aOff, uint64_t cOff, uint64_t index = 0)
    {
        if (addFlag_ && addrs_->cGM != addrs_->addGM) {
            Mc2SyncAll<CoreType>();
            MatmulAllReduceAddX3Kernel<YType>(
                addrs_->cGM, addrs_->addGM, cOff / sizeof(YType), paramCfg3_->addX3UbCnt, tPipe_);
            addrs_->addGM += cOff;
        }

        addrs_->aGM += aOff;
        addrs_->cGM += cOff;

        Mc2SyncAll<CoreType>();
        if (notifyFlag_) {
            hccl_.Commit(handleId);
        }
    }

    __aicore__ inline void HcclFinalize()
    {
        if (notifyFlag_) {
            for (uint32_t idx = 0; idx < paramCfg3_->tileLoopCnt; ++i) {
                hccl_.Wait(tileInf5_.hcclHandleId);
            }

            if (tailFl3_) {
                for (uint32_t idx = 0; idx < paramCfg3_->tailCnt; ++i) {
                    hccl_.Wait(tailInf5_.hcclHandleId);
                }
            }
        }

        Mc2SyncAll<CoreType>();
        if (notifyFlag_) {
            hccl_.Finalize();
        }
    }

    MC2GmAddrs* addrs_;
    QuantGmAddrs* quantAddrs_;
    ArnGmAddrs* arnAddrs_;
    Mc2Tiling::Mc2Msg* msgInTiling_;
    Mc2Tiling::RCSTiling* paramCfg3_;
    MC2TileInfo tileInf5_, tailInf5_;
    MC2TilingHeader* tilDat_;
    TPipe* tPipe_;
    typename HcclTypeSelector<commMode>::type hccl_;
    bool notifyFlag_;
    bool addFlag_;
    bool tailFl3_;
    bool isOneTileFlag_;
};
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_BASED_ALL_REDUCE_H