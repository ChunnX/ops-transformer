/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file hccl_shmem.h
 * \brief
 */
#ifndef SHMEM_H
#define SHMEM_H
#include "kernel_operator.h"
#include "gmmsq_base.h"

template <AscendC::HardEvent event>
__aicore__ inline void SetWaitFlag()
{
    AscendC::SetFlag<event>(EVENT_ID7);
    AscendC::WaitFlag<event>(EVENT_ID7);
}

template<typename T>
CATLASS_DEVICE void gm_dcci(__gm__ T * addr) {
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(addr));

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

CATLASS_DEVICE int32_t gm_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    do {
        if (AscendC::ReadGmByPassDCache(sig_addr) == cmp_val) {
            return *sig_addr;
        }
        if (AscendC::ReadGmByPassDCache(sig_addr) == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true);
    return -1;
}

CATLASS_DEVICE void gm_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val) {
    do {
        if (AscendC::ReadGmByPassDCache(sig_addr) != cmp_val) {
            return;
        }
    } while (true);
}

CATLASS_DEVICE void AtomicAddFlag(__gm__ int32_t* flagPtr, int32_t flagNum, int32_t flagValue = 1)
{
    SetWaitFlag<AscendC::HardEvent::MTE3_S>();
    AscendC::AtomicAdd(flagPtr, flagValue);
}

CATLASS_DEVICE void AtomicAddFlagVec(__gm__ int32_t* flagPtr, int32_t flagNum, int32_t flagValue = 1)
{
    SetWaitFlag<AscendC::HardEvent::MTE3_S>();
    SetWaitFlag<AscendC::HardEvent::MTE3_V>();
    AscendC::SetAtomicAdd<int32_t>();
    AscendC::LocalTensor<int32_t> ub0{AscendC::TPosition::VECIN, 200 * 1024, 0};
    AscendC::GlobalTensor<int32_t> flagGm;
    flagGm.SetGlobalBuffer(flagPtr);
    AscendC::Duplicate(ub0, flagValue, flagNum); // 64 * 16
    SetWaitFlag<AscendC::HardEvent::V_MTE3>();
    SetWaitFlag<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopy(flagGm, ub0, flagNum);
    SetWaitFlag<AscendC::HardEvent::MTE3_S>();
    AscendC::SetAtomicNone();
    SetWaitFlag<AscendC::HardEvent::S_MTE3>();
}

class HcclShmem {
public:
   #ifdef HCCL_COMM    // HCCL needs to initialize the HCCL context
        __gm__ HcclOpResParamCustom *WinContext_{nullptr};
        Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
        AscendC::LocalTensor<int32_t> ub;
        CATLASS_DEVICE
        HcclShmem(){
            auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
            WinContext_ = (__gm__ HcclOpResParamCustom *)contextGM0;

            m_rank = WinContext_->localUsrRankId;
            m_rankSize = WinContext_->rankSize;
            m_segmentSize = WinContext_->winSize;
        }
    #else
        CATLASS_DEVICE
        HcclShmem(GM_ADDR* symmetricPtr_, size_t rank_, size_t rankSize_){
            symmetricPtr = symmetricPtr_;
            m_rank = rank_;
            m_rankSize = rankSize_;
        }
    #endif

    CATLASS_DEVICE
    GM_ADDR operator() () const {   // No parameters: return pointer to local peermem
        #ifdef HCCL_COMM
            return (GM_ADDR)(WinContext_->localWindowsIn);
        #else
            return symmetricPtr[m_rank];
        #endif
    }

    CATLASS_DEVICE
    GM_ADDR operator() (int32_t index) const {  // With index parameter: return pointer to the base address of remote peermem
        #ifdef HCCL_COMM
            return (GM_ADDR)((index == m_rank) ? WinContext_->localWindowsIn :
                                    ((HcclRankRelationResV2Custom *)(WinContext_->remoteRes[index].nextDevicePtr))->windowsIn);
        #else
            return symmetricPtr[index];
        #endif
    }

    CATLASS_DEVICE
    GM_ADDR operator () (int64_t offset, int32_t rankId) const  {  
        #ifdef HCCL_COMM
            if (offset < 0 || offset >= m_segmentSize) {
                return nullptr;
            }
            if (rankId < 0 || rankId >= m_rankSize) {
                return nullptr;
            }
            return (GM_ADDR)((rankId == m_rank) ? WinContext_->localWindowsIn :
                                    ((HcclRankRelationResV2Custom *)(WinContext_->remoteRes[rankId].nextDevicePtr))->windowsIn) + offset;
        #else
            return symmetricPtr[rankId] + offset;
        #endif
    }

    CATLASS_DEVICE
    GM_ADDR operator () (GM_ADDR peerAddr, int32_t rankId) const  {  
        #ifdef HCCL_COMM
            if (offset < 0 || offset >= m_segmentSize) {
                return nullptr;
            }
            if (rankId < 0 || rankId >= m_rankSize) {
                return nullptr;
            }
            return (GM_ADDR)((rankId == m_rank) ? WinContext_->localWindowsIn :
                                    ((HcclRankRelationResV2Custom *)(WinContext_->remoteRes[rankId].nextDevicePtr))->windowsIn) + offset;
        #else
            int64_t peerOffset = peerAddr - symmetricPtr[m_rank];
            return symmetricPtr[rankId] + peerOffset;
        #endif
    }


    CATLASS_DEVICE
    size_t SegmentSize() const {
        return m_segmentSize;
    }

    CATLASS_DEVICE
    int32_t RankSize() const {
        return m_rankSize;
    }


    CATLASS_DEVICE
    ~HcclShmem() {
    }


    CATLASS_DEVICE
    void CrossRankSync() const {
        uint64_t flag_offset = 0;
        __gm__ int32_t* sync_counter = (__gm__ int32_t*)(*this)() + flag_offset;
        __gm__ int32_t* sync_base = (__gm__ int32_t*)(*this)() + flag_offset + 2048;
        
        int vec_id = AscendC::GetBlockIdx();
        int vec_size = AscendC::GetBlockNum();
        int count = *(sync_base + vec_id * 16) + 1;
 
        for(int i = vec_id; i < m_rankSize; i += vec_size) {
            __gm__ int32_t* sync_remote = (__gm__ int32_t*)((*this)(i)) + flag_offset + m_rank * 16;
            AscendC::WriteGmByPassDCache(sync_remote, count);
            auto sync_check = sync_counter + i * 16;
            gm_signal_wait_until_eq_for_barrier(sync_check, count);
        }
        AscendC::SyncAll<true>();
        *(sync_base + vec_id * 16) = count;
    }

    CATLASS_DEVICE
    __gm__ int32_t* SyncBaseAddr() {
        uint64_t flag_offset = 0;
        return (__gm__ int32_t*)(*this)() + flag_offset + 2048;
    }

private:
    GM_ADDR* symmetricPtr;
    int32_t m_rank;
    int32_t m_rankSize;
    size_t m_segmentSize;
};

#endif