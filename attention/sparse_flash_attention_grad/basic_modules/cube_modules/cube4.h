/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cube4.h
 * \brief Formula: dk = s^T * q
 * l0_a： dimG * selectedBlockSize * sizof(T1) (26-28k/28-30k)
 * l0_b： dimG * 512 * sizof(T1) (0k-16k)
 * l0_c： dimG * dimDqk * sizof(flot) (0-12k / 64k-76k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube4Process(const int64_t dsGmOffset, const int64_t queryGmOffset, const int64_t queryRopeGmOffset, const int64_t indicesGmOffset,
                         const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx, 
                         const int64_t lastBlockSize, const bool isLastBasicBlock)
{
    uint32_t dLoopTimes = (dimDTotal + 127) / N_SPLIT_SIZE;
    uint32_t perLoopDSize = N_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDTotal - (dLoopTimes - 1) * perLoopDSize;
    
    uint32_t blockOffset = M_SPLIT_SIZE / selectedBlockSize; // 128 / 1 = 128

    MMParam mmParam;
    mmParam.singleM = selectedBlockSize * blockOffset;
    mmParam.singleK = dimG;
    mmParam.isFixOut = false;
    mmParam.isLeftTranspose = true;
    mmParam.isRightTranspose = true;
    mmParam.dstStride = dimDTotal * dimN2;

    CopyGmToL1(l1_ds_tensor, dsWorkspaceGm[dsGmOffset], dimG, selectedCntOffset * selectedBlockSize, TOTAL_BLOCK_SIZE);
    for (int32_t mIdx = blkCntOffset; mIdx < blkCntOffset + selectedCntOffset; mIdx+=blockOffset) {
        int32_t topkIdx = topkIndicesGm[indicesGmOffset].GetValue(mIdx);
        int32_t startS2Idx = topkIdx * selectedBlockSize * dimN2 * dimDTotal;
        int32_t l1Offset = (mIdx - blkCntOffset) * selectedBlockSize * dimGAlign;

        mmParam.singleN = perLoopDSize;
        mmParam.singleM = min(selectedBlockSize * blockOffset, selectedCntOffset * selectedBlockSize - (mIdx - blkCntOffset) * selectedBlockSize);

        LocalTensor<T1> current_l1_ds_tensor, l1_query_tensor;
        current_l1_ds_tensor = l1_ds_tensor[l1Offset];
        int64_t currentQueryOffset;
        
        bool isLast = isLastBasicBlock && (mIdx + blockOffset >= blkCntOffset + selectedCntOffset);
        for (uint32_t dIdx = 0;dIdx < dLoopTimes - 1; dIdx++) {
            LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_];
            l1_query_tensor = l1_query_tensors[ping_pong_flag_l1_query_];
            
            currentQueryOffset = queryGmOffset + dIdx * N_SPLIT_SIZE;
            WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
            CopyGmToL1(l1_query_tensor, queryGm[currentQueryOffset], dimG, N_SPLIT_SIZE, dimDqk);

            int64_t currentOutGmOffset = outGmOffset + dIdx * perLoopDSize;

            // l0a复用
            uint32_t l0a_ping_pong_flag = ping_pong_flag_l0a_;
            MmadInnerWithSync<T1, false, true>(l0cTensor, current_l1_ds_tensor, l1_query_tensor,
                    aL0TensorPingPong, bL0TensorPingPong,
                    mmParam, l0a_ping_pong_flag, ping_pong_flag_l0b_, ping_pong_flag_l0c_, dIdx == 0, dkWorkspaceGm[currentOutGmOffset]);
            ScatterFixOutWithSync<true>(dkWorkspaceGm[currentOutGmOffset], topkIndicesGm[indicesGmOffset + mIdx], l0cTensor, mmParam, selectedBlockSize, blockOffset, dimN2, L0C_EVENTS[ping_pong_flag_l0c_], lastBlockSize, isLast);
            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
            UpdatePingPongFlag(ping_pong_flag_l0c_);
            UpdatePingPongFlag(ping_pong_flag_l1_query_);
        }

        mmParam.singleN = tailLoopDSize;
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_];
        l1_query_tensor = l1_query_tensors[ping_pong_flag_l1_query_];
        
        currentQueryOffset = HAS_ROPE ? queryRopeGmOffset : queryGmOffset + (dLoopTimes - 1) * K_SPLIT_SIZE;
        GlobalTensor<T1> srcGm = HAS_ROPE ? queryRopeGm[currentQueryOffset] : queryGm[currentQueryOffset];
        int64_t destStride = HAS_ROPE ? dimRope : dimDqk;
        WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
        CopyGmToL1(l1_query_tensor, srcGm, dimG, tailLoopDSize, destStride);

        int64_t currentOutGmOffset = outGmOffset + (dLoopTimes - 1) * perLoopDSize;
        
        MmadInnerWithSync<T1, false, true>(l0cTensor, current_l1_ds_tensor, l1_query_tensor,
                aL0TensorPingPong, bL0TensorPingPong,
                mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_, ping_pong_flag_l0c_, true, dkWorkspaceGm[currentOutGmOffset]);
        ScatterFixOutWithSync<true>(dkWorkspaceGm[currentOutGmOffset], topkIndicesGm[indicesGmOffset + mIdx], l0cTensor, mmParam, selectedBlockSize, blockOffset, dimN2, L0C_EVENTS[ping_pong_flag_l0c_], lastBlockSize, isLast);
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_QUERY_EVENTS[ping_pong_flag_l1_query_]);
        UpdatePingPongFlag(ping_pong_flag_l0c_);
        UpdatePingPongFlag(ping_pong_flag_l1_query_);
    }
}