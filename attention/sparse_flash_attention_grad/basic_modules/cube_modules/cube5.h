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
 * \file cube5.h
 * \brief Formula: dv = p^T * dy
 * l0_a： dimG * dimDqk * sizof(T1) (30k-32k/32-34k)
 * l0_b： dimG * 512 * sizof(T1) (48-64k)
 * l0_c： dimG * dimDqk * sizof(float) (0-12k / 64k-76k)
 */
template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube5Process(const int64_t pGmOffset, const int64_t dyGmOffset, const int64_t indicesGmOffset,
                         const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx, 
                         const int64_t lastBlockSize, const bool isLastBasicBlock)
{
    uint32_t dLoopTimes = (dimDv + 127) / N_SPLIT_SIZE;
    uint32_t perLoopDSize = N_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDv - (dLoopTimes - 1) * perLoopDSize;
    uint32_t blockOffset = M_SPLIT_SIZE / selectedBlockSize; // 128 / 1 = 128

    MMParam mmParam;
    mmParam.singleM = selectedBlockSize * blockOffset;
    mmParam.singleN = perLoopDSize;
    mmParam.singleK = dimG;
    mmParam.isFixOut = false;
    mmParam.isLeftTranspose = true;
    mmParam.isRightTranspose = true;
    mmParam.dstStride = dimDv * dimN2;

    LocalTensor<T1> l1_dy_tensor = l1_dy_tensors[mmPingPongIdx];
    for (int32_t mIdx = blkCntOffset; mIdx < blkCntOffset + selectedCntOffset; mIdx+=blockOffset) {
        LocalTensor<T1> current_l1_dy_tensor, l1_p_tensor;
        l1_p_tensor = l1_common_tensors[ping_pong_flag_l1_common_];

        mmParam.singleM = min(selectedBlockSize * blockOffset, selectedCntOffset * selectedBlockSize - (mIdx - blkCntOffset) * selectedBlockSize);

        bool isLast = isLastBasicBlock && (mIdx + blockOffset >= blkCntOffset + selectedCntOffset);
        WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
        CopyGmToL1(l1_p_tensor, pWorkspaceGm[pGmOffset + (mIdx - blkCntOffset) * selectedBlockSize], dimG, mmParam.singleM, 512);
        for (int32_t dIdx = 0; dIdx < dLoopTimes; dIdx++) {
            current_l1_dy_tensor = l1_dy_tensor[dIdx * perLoopDSize * dimGAlign];
            LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_ & 1];
            
            int64_t currentOutGmOffset = outGmOffset + dIdx * perLoopDSize;
            // l0a复用
            uint32_t l0a_ping_pong_flag = ping_pong_flag_l0a_;
            MmadInnerWithSync<T1, false, true>(l0cTensor, l1_p_tensor, current_l1_dy_tensor,
                    aL0TensorPingPong, bL0TensorPingPong,
                    mmParam, l0a_ping_pong_flag, ping_pong_flag_l0b_, ping_pong_flag_l0c_, dIdx == 0, dvWorkspaceGm[currentOutGmOffset]);
            ScatterFixOutWithSync<true>(dvWorkspaceGm[currentOutGmOffset], topkIndicesGm[indicesGmOffset + mIdx], l0cTensor, mmParam, selectedBlockSize, blockOffset, dimN2, L0C_EVENTS[ping_pong_flag_l0c_], lastBlockSize, isLast);
            UpdatePingPongFlag(ping_pong_flag_l0c_);
        }
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
        UpdatePingPongFlag(ping_pong_flag_l0a_);
        UpdatePingPongFlag(ping_pong_flag_l1_common_);
    }
}