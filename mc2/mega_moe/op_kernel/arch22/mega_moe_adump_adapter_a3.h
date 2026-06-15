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

namespace MC2MegaMoeAdump {

constexpr uint32_t ADUMP_UB_SIZE_BYTES = 1024;

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

enum MegaMoeAdumpSharedFieldPos : uint32_t {
    CCL_BUFFER_SIZE_POS = 0,
    WORLD_SIZE_POS = 1,
    RANK_ID_POS = 2,
    AIV_NUM_POS = 3,
    M_POS = 4,
    K_POS = 5,
    N_POS = 6,
    IS_TRANSPOSE_W1_POS = 7,
    IS_TRANSPOSE_W2_POS = 8,
    HAS_BIAS1_POS = 9,
    HAS_BIAS2_POS = 10,
    EXPERT_NUM_PER_RANK_POS = 11,
    TOPK_POS = 12,
    MAX_RECV_TOKEN_NUM_POS = 13,
    DISPATCH_QUANT_MODE_POS = 14,
    DISPATCH_QUANT_OUT_DTYPE_POS = 15,
    COMBINE_QUANT_MODE_POS = 16,
    COMM_ALG_CODE_POS = 17,
    NUM_MAX_TOKENS_PER_RANK_POS = 18,
    ACTIVATION_CODE_POS = 19,
    ACTIVATION_CLAMP_POS = 20,
    SHARED_FIELD_POS_END,
};

struct MegaMoeAdumpCoreCustomInfoA3 {
    uint32_t aivId;
    uint32_t currentStageId;
    uint32_t startTimeStampHigh32Bit;
    uint32_t startTimeStampLow32Bit;
};

enum MegaMoeAdumpCoreBasicInfoPos : uint32_t {
    AIVID_POS = 0,
    CURRENT_STAGEID_POS = 1,
    START_TIMESTAMP_HIGH_32BIT_POS = 2,
    START_TIMESTAMP_LOW_32BIT_POS = 3,
    CORE_BASIC_INFO_POS_END,
};

enum StageA3 : uint32_t {
    STAGE_INIT = 0,
    STAGE_APPLY_XACTIVATE_MASK,
    STAGE_INIT_ROUTING,
    STAGE_CROSS_RANK_SYNC,
    STAGE_GET_CUMSUM,
    STAGE_PULL_TOKENS,
    STAGE_EPILOGUE1,
    STAGE_COMBINE,
    STAGE_RESET,
    STAGE_CROSS_RANK_SYNC2,
    STAGE_UNPERMUTE,
    STAGE_END
};

class MegaMoeAdumpAdapterA3 {
public:
    __aicore__ inline void Init(GM_ADDR dumpBase)
    {
        static_assert(sizeof(MegaMoeAdumpSharedField) == SHARED_FIELD_POS_END * sizeof(uint32_t), 
            "struct MegaMoeAdumpSharedField should be the same as enum MegaMoeAdumpSharedFieldPos");
        static_assert(sizeof(MegaMoeAdumpCoreCustomInfoA3) == CORE_BASIC_INFO_POS_END * sizeof(uint32_t), 
            "struct MegaMoeAdumpCoreCustomInfoA3 should be the same as enum MegaMoeAdumpCoreBasicInfoPos");
        
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t sharedFieldSize = RoundUp(sizeof(MegaMoeAdumpSharedField), MC2Adump::ADUMP_GM_ALIGN_BYTES);
        uint32_t customFieldSize = RoundUp(sizeof(MegaMoeAdumpCoreCustomInfoA3), MC2Adump::ADUMP_GM_ALIGN_BYTES);

        MC2Adump::AdumpConfig config = {
            .opType = MC2Adump::OP_TYPE_MEGA_MOE,
            .aivId = aivId,
            .ubStartAddr = 0,
            .ubSizeBytes = ADUMP_UB_SIZE_BYTES,
            .sharedFieldSize = sharedFieldSize,
            .customFieldSize = customFieldSize,
            .stageNum = StageA3::STAGE_END,
            .timeOutCycles = 10 * 50 * 1000 * 1000,
        };
        adumpEngine_.Init(dumpBase, config);
        InitCoreBasicInfo();
    }

    __aicore__ inline void FillSharedField(MegaMoeAdumpSharedField& sharedField)
    {
        // 仅0核完成共享区域初始化
        uint32_t aivId = AscendC::GetBlockIdx();
        if (aivId == 0) {
            InitAdumpSharedField(sharedField);
        }
    }

    __aicore__ inline void FillCoreBaseInfo()
    {
        
    }

    __aicore__ inline uint32_t GetAdumpTotalGmSizeBytes()
    {
        uint32_t headerSize = RoundUp(sizeof(MC2Adump::AdumpHeader), MC2Adump::ADUMP_GM_ALIGN_BYTES);
        uint32_t sharedFieldSize = RoundUp(sizeof(MegaMoeAdumpSharedField), MC2Adump::ADUMP_GM_ALIGN_BYTES);
        uint32_t customFieldSize = RoundUp(sizeof(MegaMoeAdumpCoreCustomInfoA3), MC2Adump::ADUMP_GM_ALIGN_BYTES);
        return headerSize + sharedFieldSize + customFieldSize * AscendC::GetBlockNum();
    }

    __aicore__ inline uint32_t GetAdumpTotalUbSizeBytes()
    {
        return ADUMP_UB_SIZE_BYTES;
    }

    __aicore__ inline void UpdateAdumpCurrentStateId(uint32_t stageId)
    {
        uint32_t maxElemNum = 0;
        LocalTensor<uint32_t> adumpU32LocalTensor = GetAdumpLocalTensor<uint32_t>(maxElemNum);
        adumpU32LocalTensor.SetValue(0, stageId);

        GM_ADDR adumpCoreCustomFieldAddr = GetAdumpCoreCustomFieldAddr();
        GlobalTensor<uint32_t> adumpU32GlobalTensor;
        adumpU32GlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)adumpCoreCustomFieldAddr);

        PipeBarrier<PIPE_ALL>();
        DataCopyPad(adumpU32GlobalTensor[MegaMoeAdumpCoreBasicInfoPos::CURRENT_STAGEID_POS], adumpU32LocalTensor[1], {1, sizeof(uint32_t), 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
  
private:
    __aicore__ inline void InitAdumpSharedField(MegaMoeAdumpSharedField& sharedField)
    {
        uint32_t maxElemNum = 0;
        LocalTensor<uint32_t> adumpU32LocalTensor = GetAdumpLocalTensor<uint32_t>(maxElemNum);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::CCL_BUFFER_SIZE_POS, sharedField.cclBufferSize);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::WORLD_SIZE_POS, sharedField.worldSize);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::RANK_ID_POS, sharedField.rankId);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::AIV_NUM_POS, sharedField.aivNum);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::M_POS, sharedField.m);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::K_POS, sharedField.k);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::N_POS, sharedField.n);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::IS_TRANSPOSE_W1_POS, sharedField.isTransposeW1);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::IS_TRANSPOSE_W2_POS, sharedField.isTransposeW2);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::HAS_BIAS1_POS, sharedField.hasBias1);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::HAS_BIAS2_POS, sharedField.hasBias2);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::EXPERT_NUM_PER_RANK_POS, sharedField.expertNumPerRank);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::TOPK_POS, sharedField.topK);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::MAX_RECV_TOKEN_NUM_POS, sharedField.maxRecvTokenNum);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::DISPATCH_QUANT_MODE_POS, sharedField.dispatchQuantMode);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::DISPATCH_QUANT_OUT_DTYPE_POS, sharedField.dispatchQuantOutDtype);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::COMBINE_QUANT_MODE_POS, sharedField.combineQuantMode);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::COMM_ALG_CODE_POS, sharedField.commAlgCode);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::NUM_MAX_TOKENS_PER_RANK_POS, sharedField.numMaxTokensPerRank);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::ACTIVATION_CODE_POS, sharedField.activationCode);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpSharedFieldPos::ACTIVATION_CLAMP_POS, *reinterpret_cast<uint32_t*>(&sharedField.activationClamp));

        GM_ADDR adumpSharedFieldAddr = GetAdumpSharedFieldAddr();
        GlobalTensor<uint32_t> adumpU32GlobalTensor;
        adumpU32GlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)adumpSharedFieldAddr);

        PipeBarrier<PIPE_ALL>();
        DataCopyPad(adumpU32GlobalTensor, adumpU32LocalTensor, {1, MegaMoeAdumpSharedFieldPos::SHARED_FIELD_POS_END, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitCoreBasicInfo()
    {
        uint32_t aivId = AscendC::GetBlockIdx();
        uint32_t currentStageId = 0;
        uint64_t startTimeStamp = AscendC::GetSystemCycle();

        uint32_t maxElemNum = 0;
        LocalTensor<uint32_t> adumpU32LocalTensor = GetAdumpLocalTensor<uint32_t>(maxElemNum);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpCoreBasicInfoPos::AIVID_POS, aivId);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpCoreBasicInfoPos::CURRENT_STAGEID_POS, currentStageId);
        adumpU32LocalTensor.SetValue(MegaMoeAdumpCoreBasicInfoPos::START_TIMESTAMP_HIGH_32BIT_POS, (uint32_t)(startTimeStamp >> 32));
        adumpU32LocalTensor.SetValue(MegaMoeAdumpCoreBasicInfoPos::START_TIMESTAMP_LOW_32BIT_POS, (uint32_t)(startTimeStamp & 0x00000000FFFFFFFF));

        GM_ADDR adumpCoreCustomFieldAddr = GetAdumpCoreCustomFieldAddr();
        GlobalTensor<uint32_t> adumpU32GlobalTensor;
        adumpU32GlobalTensor.SetGlobalBuffer((__gm__ uint32_t*)adumpCoreCustomFieldAddr);

        PipeBarrier<PIPE_ALL>();
        DataCopyPad(adumpU32GlobalTensor, adumpU32LocalTensor, {1, MegaMoeAdumpCoreBasicInfoPos::CORE_BASIC_INFO_POS_END, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline GM_ADDR GetAdumpSharedFieldAddr()
    {
        return adumpEngine_.GetSharedFieldAddr();
    }
    
    __aicore__ inline GM_ADDR GetAdumpCoreCustomFieldAddr()
    {
        return adumpEngine_.GetCoreCustomFieldAddr();
    }

    template<typename T>
    __aicore__ inline LocalTensor<T> GetAdumpLocalTensor(uint32_t &elemNum)
    {
        elemNum = adumpEngine_.GetAdumpUbSizeBytes() / sizeof(T);
        return adumpEngine_.GetAdumpU8LocalTensor().template ReinterpretCast<T>();
    }

private:
    MC2Adump::MC2AdumpEngine adumpEngine_;
};
} //namespace MC2MegaMoeAdump

#endif  // MEGA_MOE_ADUMP_ADAPTER_A3