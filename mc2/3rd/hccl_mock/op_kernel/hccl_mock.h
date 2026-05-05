/* * Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#ifndef HCCL_MOCK_H
#define HCCL_MOCK_H

#ifdef ENABLE_SINGLE_NPU_MOCK

#include "kernel_operator.h"

namespace MC2Mock {

template<HcclServerType serverType>
class Hccl {
public:
    __aicore__ inline void InitV2(GM_ADDR hcclContextGm, const void *hcclInitTiling) {}
    __aicore__ inline void SetCcTilingV2(uint64_t hcclCcTilingOffset) {}
    __aicore__ inline uint32_t GetRankId() { return rankId_; }
    __aicore__ inline uint32_t GetRankDim() { return epWorldSize_; }
    __aicore__ inline void SetRankInfo(uint32_t rankId, uint32_t epWorldSize) {
        rankId_ = rankId;
        epWorldSize_ = epWorldSize;
    }
    
    template<bool inlineMode>
    __aicore__ inline HcclHandle AlltoAllV(__gm__ uint8_t *sendBuf, uint64_t *sendCnt,
        uint64_t *sOffset, HcclDataType sType, __gm__ uint8_t *recvBuf,
        uint64_t *recvCnt, uint64_t *rOffset, HcclDataType rType)
    {
        if ASCEND_IS_AIC { return 0; }
        if ASCEND_IS_AIV { if (GetBlockIdx() != 0) { return 0; }
        
        AscendC::printf("[MOCK] AlltoAllV: ep=%u rank=%u sType=%d rType=%d\n", epWorldSize_, rankId_, (int)sType, (int)rType);
        }
        
        for (uint32_t i = 0U; i < epWorldSize_; i++) {
            if ASCEND_IS_AIV {
                AscendC::printf("[MOCK] peer=%u: sendCnt=%lu sOffset=%lu recvCnt=%lu rOffset=%lu\n",
                    i, sendCnt[i], sOffset[i], recvCnt[i], rOffset[i]);
            }
        }
        
        for (uint32_t i = 0U; i < epWorldSize_; i++) {
            if (recvCnt[i] > 0UL) {
                if ASCEND_IS_AIV {
                    AscendC::printf("[MOCK] peer=%u: SKIPPING DataCopy (commented out for debug)\n", i);
                }
            }
        }
        
        if ASCEND_IS_AIV {
            AscendC::printf("[MOCK] AlltoAllV done (no actual copy)\n");
        }
        return handleId_++;
    }
    
    __aicore__ inline void Wait(HcclHandle handleId) {}
    __aicore__ inline void Finalize() {}
    
    uint32_t rankId_ = 0U;
    uint32_t epWorldSize_ = 2U;
    HcclHandle handleId_ = 0U;
};

using HcclAicpu = Hccl<HCCL_SERVER_TYPE_AICPU>;
using HcclCcu = Hccl<HCCL_SERVER_TYPE_CCU>;

}

#define Hccl MC2Mock::Hccl

#endif

#endif
