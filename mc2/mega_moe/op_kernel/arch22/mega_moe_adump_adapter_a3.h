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
 * \file mega_moe_adump_adapter_a3.h
 * \brief 实现 MegaMoe 算子的 adump 记录引擎，适配 A3 架构
 */
#ifndef MEGA_MOE_ADUMP_ADAPTER_A3
#define MEGA_MOE_ADUMP_ADAPTER_A3

#include "../../../common/op_kernel/mc2_adump_engine.h"

constexpr uint32_t MEGA_MOE_ADUMP_PER_CORE_SIZE = 256;
constexpr uint32_t MEGA_MOE_ADUMP_RESERVED_SIZE = 1024 * 1024;

struct MegaMoeAdumpSharedField {
    uint32_t cclBufferSize;
    uint32_t worldSize;
    uint32_t rankId;
    uint32_t aivNum;

    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t isTransposeW1;
    uint32_t isTransposeW2;
    uint32_t hasBias1;
    uint32_t hasBias2;

    uint32_t expertNumPerRank;
    uint32_t topK;
    uint32_t maxRecvTokenNum;
    uint32_t dispatchQuantMode;
    uint32_t dispatchQuantOutDtype;
    uint32_t combineQuantMode;
    uint32_t commAlgCode;
    uint32_t numMaxTokensPerRank;
    uint32_t activationCode;
    float activationClamp;
};

struct MegaMoeAdumpBasicInfo {
    uint32_t aivId;
    uint32_t currentStageId;
    uint64_t enterTimeStamp;
    uint64_t exitTimeStamp;
};

struct MegaMoeAdumpStageApplyXActiveMaskInfo {
    uint32_t totalElements;
    uint32_t startIdx;
    uint32_t endIdx;
    uint32_t hasXActiveMask;
};

struct MegaMoeAdumpStageInitRoutingInfo {
    uint32_t isQuantRouting;
    uint32_t m;
    uint32_t topK;
    uint32_t expertNum;
};

struct MegaMoeAdumpStageAllGatherPushInfo {
    uint32_t targetRankId;
    uint32_t pushCompleted;
};

struct MegaMoeAdumpStageAllGatherWaitInfo {
    uint32_t targetRankId;
    uint32_t waitCompleted;
    uint32_t checkIdx;
};

struct MegaMoeAdumpStageGetCumsumInfo {
    uint32_t isCore0;
};

struct MegaMoeAdumpStageNotifyAicGmm1Info {
    uint32_t syncgmm1Idx;
};

struct MegaMoeAdumpStagePullTokensInfo {
    uint32_t groupIdx;
    uint32_t expertPerRank;
    uint32_t targetRankId;
    uint32_t rowSrc;
    uint32_t prevGroupSum;
    uint32_t currentM;
    uint32_t rowStart;
    uint32_t rows;
};

struct MegaMoeAdumpStageEpilogue1WaitGmm1Info {
    uint32_t syncIdx;
    uint32_t isW4A8;
};

struct MegaMoeAdumpStageEpilogue1ProcessInfo {
    uint32_t rowStartThisCore;
    uint32_t curRowNum;
    uint32_t syncIdx;
    uint32_t isW4A8;
};

struct MegaMoeAdumpStageEpilogue1NotifyGmm2Info {
    uint32_t syncIdx;
};

struct MegaMoeAdumpStageCombineWaitGmm2Info {
    uint32_t groupIdx;
    uint32_t flagId;
};

struct MegaMoeAdumpStageCombineProcessInfo {
    uint32_t groupIdx;
    uint32_t expertPerRank;
    uint32_t targetRankId;
    uint32_t isCombineV1;
    uint32_t srcRowOffset;
    uint32_t dataRows;
    uint32_t dstRowOffset;
};

struct MegaMoeAdumpStageResetInfo {
    uint32_t resetCoreIdx;
};

struct MegaMoeAdumpStageCrossRankSync2Info {
    uint32_t targetRankId;
    uint32_t syncCompleted;
    uint32_t isV2;
    uint32_t syncCount;
};

struct MegaMoeAdumpStageUnpermuteInfo {
    uint32_t totalTokens;
    uint32_t n2;
    uint32_t topK;
    uint32_t coreNumUsed;
};

enum StageA3 : uint32_t {
    STAGE_INIT = 0,
    STAGE_APPLY_XACTIVATE_MASK,
    STAGE_INIT_ROUTING,
    STAGE_CROSS_RANK_SYNC_ALLGATHER_PUSH,
    STAGE_CROSS_RANK_SYNC_ALLGATHER_WAIT,
    STAGE_GET_CUMSUM,
    STAGE_NOTIFY_AIC_GMM1,
    STAGE_PULL_TOKENS,
    STAGE_EPILOGUE1_WAIT_GMM1,
    STAGE_EPILOGUE1_PROCESS,
    STAGE_EPILOGUE1_NOTIFY_GMM2,
    STAGE_COMBINE_WAIT_GMM2,
    STAGE_COMBINE_PROCESS,
    STAGE_RESET,
    STAGE_CROSS_RANK_SYNC2,
    STAGE_UNPERMUTE,
    STAGE_END
};

struct MegaMoeAdumpCoreCustomInfoA3 {
    MegaMoeAdumpBasicInfo basicInfo;
    union MegaMoeAdumpStageInfo {
        MegaMoeAdumpStageApplyXActiveMaskInfo applyXActiveMaskInfo;
        MegaMoeAdumpStageInitRoutingInfo initRoutingInfo;
        MegaMoeAdumpStageAllGatherPushInfo allGatherPushInfo;
        MegaMoeAdumpStageAllGatherWaitInfo allGatherWaitInfo;
        MegaMoeAdumpStageGetCumsumInfo getCumsumInfo;
        MegaMoeAdumpStageNotifyAicGmm1Info notifyAicGmm1Info;
        MegaMoeAdumpStagePullTokensInfo pullTokensInfo;
        MegaMoeAdumpStageEpilogue1WaitGmm1Info epilogue1WaitGmm1Info;
        MegaMoeAdumpStageEpilogue1ProcessInfo epilogue1ProcessInfo;
        MegaMoeAdumpStageEpilogue1NotifyGmm2Info epilogue1NotifyGmm2Info;
        MegaMoeAdumpStageCombineWaitGmm2Info combineWaitGmm2Info;
        MegaMoeAdumpStageCombineProcessInfo combineProcessInfo;
        MegaMoeAdumpStageResetInfo resetInfo;
        MegaMoeAdumpStageCrossRankSync2Info crossRankSync2Info;
        MegaMoeAdumpStageUnpermuteInfo unpermuteInfo;
    } stageInfo;
};

class MegaMoeAdumpAdapterA3 {
public:
    __aicore__ inline MegaMoeAdumpAdapterA3() = default;
    __aicore__ inline ~MegaMoeAdumpAdapterA3() = default;

    __aicore__ inline void Init(GM_ADDR dumpBase, MegaMoeAdumpSharedField& sharedField)
    {
        uint32_t aivId = AscendC::GetBlockIdx();

        MC2Adump::AdumpConfig config = {
            .opType = MC2Adump::OP_TYPE_MEGA_MOE,
            .aivId = aivId,
            .sharedFieldSize = sizeof(MegaMoeAdumpSharedField),
            .stageNum = StageA3::STAGE_END,
            .timeOutCycles = 10 * 50 * 1000 * 1000,
            .perCoreSize = MEGA_MOE_ADUMP_PER_CORE_SIZE,
            .enableCacheFlush = true,
        };
        adumpEngine_.Init(dumpBase, config);

        if (aivId == 0) {
            InitAdumpSharedField(sharedField);
        }

        auto* field = GetAdumpCoreCustomField();
        field->basicInfo.aivId = aivId;
        field->basicInfo.currentStageId = StageA3::STAGE_INIT;
        field->basicInfo.enterTimeStamp = 0;
        field->basicInfo.exitTimeStamp = 0;
    }

    __aicore__ inline uint32_t GetAdumpTotalSize()
    {
        return sizeof(MC2Adump::AdumpHeader) + sizeof(MegaMoeAdumpSharedField) +
               MEGA_MOE_ADUMP_PER_CORE_SIZE * AscendC::GetBlockNum();
    }

    __aicore__ inline void EnterStage(uint32_t stageId)
    {
        auto* field = GetAdumpCoreCustomField();
        field->basicInfo.exitTimeStamp = AscendC::GetTimeStamp();
        field->basicInfo.currentStageId = stageId;
        field->basicInfo.enterTimeStamp = AscendC::GetTimeStamp();
    }

    __aicore__ inline void ExitStage()
    {
        auto* field = GetAdumpCoreCustomField();
        field->basicInfo.exitTimeStamp = AscendC::GetTimeStamp();
        adumpEngine_.FlushCache();
    }

    template<typename StageInfoT>
    __aicore__ inline void UpdateStageInfo(const StageInfoT& info)
    {
        auto* field = GetAdumpCoreCustomField();
        static_assert(sizeof(StageInfoT) <= sizeof(field->stageInfo),
                     "StageInfo size exceeds union size");
        *reinterpret_cast<StageInfoT*>(&field->stageInfo) = info;
        adumpEngine_.FlushCache();
    }

private:
    __aicore__ inline void InitAdumpSharedField(MegaMoeAdumpSharedField& sharedField)
    {
        __gm__ MegaMoeAdumpSharedField* adumpSharedField = adumpEngine_.GetSharedFieldPtr();
        adumpSharedField->cclBufferSize = sharedField.cclBufferSize;
        adumpSharedField->worldSize = sharedField.worldSize;
        adumpSharedField->rankId = sharedField.rankId;
        adumpSharedField->aivNum = sharedField.aivNum;
        adumpSharedField->m = sharedField.m;
        adumpSharedField->k = sharedField.k;
        adumpSharedField->n = sharedField.n;
        adumpSharedField->isTransposeW1 = sharedField.isTransposeW1;
        adumpSharedField->isTransposeW2 = sharedField.isTransposeW2;
        adumpSharedField->hasBias1 = sharedField.hasBias1;
        adumpSharedField->hasBias2 = sharedField.hasBias2;
        adumpSharedField->expertNumPerRank = sharedField.expertNumPerRank;
        adumpSharedField->topK = sharedField.topK;
        adumpSharedField->maxRecvTokenNum = sharedField.maxRecvTokenNum;
        adumpSharedField->dispatchQuantMode = sharedField.dispatchQuantMode;
        adumpSharedField->dispatchQuantOutDtype = sharedField.dispatchQuantOutDtype;
        adumpSharedField->combineQuantMode = sharedField.combineQuantMode;
        adumpSharedField->commAlgCode = sharedField.commAlgCode;
        adumpSharedField->numMaxTokensPerRank = sharedField.numMaxTokensPerRank;
        adumpSharedField->activationCode = sharedField.activationCode;
        adumpSharedField->activationClamp = sharedField.activationClamp;
    }

    __aicore__ inline __gm__ MegaMoeAdumpSharedField* GetAdumpSharedField()
    {
        return (__gm__ MegaMoeAdumpSharedField*)adumpEngine_.GetSharedFieldPtr();
    }

    __aicore__ inline __gm__ MegaMoeAdumpCoreCustomInfoA3* GetAdumpCoreCustomField()
    {
        return adumpEngine_.GetCoreCustomFieldPtr<MegaMoeAdumpCoreCustomInfoA3>();
    }

private:
    MC2Adump::MC2AdumpEngine adumpEngine_;
};

#endif
