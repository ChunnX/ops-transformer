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
 * \file moe_distribute_a2_constant.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_A2_ADUMP_H
#define MOE_DISTRIBUTE_A2_ADUMP_H
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "moe_distribute_a2_constant.h"
namespace Mc2A2Kernel {

using namespace AscendC;
class MoeDistributeA2ADump {
public:
    __aicore__ inline void InitWinState(GM_ADDR aDumpWinStartAddr, uint32_t aivId,
                                    uint32_t epRankIdHccl, uint32_t epWorldSizeHccl, uint32_t epRankIdOriginal,
                                    uint32_t moeExpertNum, uint32_t epWorldSizeOriginal, uint32_t globalBs,
                                    uint32_t bufferId, uint32_t isLayered, uint32_t aivNum)
    {
        selfDataStatusGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)(aDumpWinStartAddr + aivId * WIN_ADDR_ALIGN));
        uint32_t dataSize = UB_ALIGN * 2U;
        auto dataStateLocalTensor64 = AscendC::LocalTensor<uint64_t>{AscendC::TPosition::LCM, 0, static_cast<uint32_t>(dataSize / sizeof(uint64_t))};
        dataStateLocalTensor_ = AscendC::LocalTensor<uint32_t>{AscendC::TPosition::LCM, 0, static_cast<uint32_t>(dataSize / sizeof(uint32_t))};
        DataCopy(dataStateLocalTensor_, selfDataStatusGMTensor_, UB_ALIGN / sizeof(uint32_t));
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        dataStateLocalTensor_.SetValue(BUFFERID_POS, bufferId);
        dataStateLocalTensor_.SetValue(OPOSITION_POS, RUNPOS_INIT);   
        dataStateLocalTensor_.SetValue(TILING_EPRANKID_POS, epRankIdOriginal);
        dataStateLocalTensor_.SetValue(MOE_NUM_POS, moeExpertNum);
        dataStateLocalTensor_.SetValue(TILING_WORLDSIZE_POS, epWorldSizeOriginal);
        dataStateLocalTensor_.SetValue(GLOBALBS_POS, globalBs);
        dataStateLocalTensor_.SetValue(ISLAYERED_POS, isLayered);
        dataStateLocalTensor_.SetValue(AIVNUM_POS, aivNum);
        uint64_t opCnt = dataStateLocalTensor64.GetValue(OP_CNT_POSUL);
        opCnt = opCnt + 1;
        dataStateLocalTensor64.SetValue(OP_CNT_POSUL, opCnt);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(selfDataStatusGMTensor_, dataStateLocalTensor_, dataSize);
        if ((epRankIdOriginal != epRankIdHccl) || (epWorldSizeOriginal != epWorldSizeHccl)) {
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            DataCopyParams hcclDatacopyParams{1U, HCCL_DFX_NUM * sizeof(uint32_t), 0U, 0U};
            dataStateLocalTensor_.SetValue(HCCL_EPRANKId_POS, epRankIdHccl);
            dataStateLocalTensor_.SetValue(HCCL_WORLDSIZE_POS, epWorldSizeHccl);
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyPad(selfDataStatusGMTensor_[HCCL_DFX_POS], dataStateLocalTensor_, hcclDatacopyParams);
        }
    }

    __aicore__ inline void RunPosRecord(const uint32_t runPos) {
        dataStateLocalTensor_.SetValue(0, runPos);
        auto dataStateParams = DataCopyExtParams{1U, sizeof(uint32_t), 0U, 0U, 0U};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor_[OPOSITION_POS], dataStateLocalTensor_, dataStateParams);
    }

    __aicore__ inline void UpdateAivTaskInfo(const uint32_t firstEpRankId, const uint32_t epRankNum) {
        dataStateLocalTensor_.SetValue(0, firstEpRankId);
        dataStateLocalTensor_.SetValue(1, epRankNum);
        auto dataStateParams = DataCopyExtParams{1U, 2 * sizeof(uint32_t), 0U, 0U, 0U};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor_[AIV_TASK_FIRST_EPRANKID_POS], dataStateLocalTensor_, dataStateParams);
    }

    __aicore__ inline void UpdateAivArrivedEpRank(const uint32_t firstEpRankId, const uint32_t epRankId) {
        uint32_t loc = (epRankId - firstEpRankId) / sizeof(uint32_t);
        uint32_t relativeLoc = (epRankId - firstEpRankId) % sizeof(uint32_t);
        DataCopy(dataStateLocalTensor_, selfDataStatusGMTensor_[AIV_TASK_ARRIVED_EPRANKID_START_POS + loc], UB_ALIGN / sizeof(uint32_t));
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        uint32_t status = dataStateLocalTensor_.GetValue(0);
        status |= (1U << relativeLoc);
        dataStateLocalTensor_.SetValue(0, status);
        auto dataStateParams = DataCopyExtParams{1U, sizeof(uint32_t), 0U, 0U, 0U};
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyPad(selfDataStatusGMTensor_[AIV_TASK_ARRIVED_EPRANKID_START_POS + loc], dataStateLocalTensor_, dataStateParams);
    }

private:
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
};
}


#endif
