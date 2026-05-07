/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
using namespace AscendC;

namespace BSA_ARC35 {

template <typename BSA_TYPE>
class CubeOp {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    int32_t batch_num_;
    int32_t q_seq_len_;
    int32_t kv_seq_len_;
    int32_t q_group_;
    int32_t q_head_num_;
    int32_t kv_head_num_;
    int32_t head_dim_;
    int32_t head_dim_align_;
    int32_t ping_pong_flag_inner{0};
    int32_t ping_pong_flag_outer{0};
    // l1_tensor
    LocalTensor<INPUT_TYPE> query_l1_tensor_;
    LocalTensor<INPUT_TYPE> key_l1_tensor_;
    LocalTensor<INPUT_TYPE> l1_a_tensor_ping_;
    LocalTensor<INPUT_TYPE> l1_a_tensor_pong_;
    LocalTensor<INPUT_TYPE> l1_b_tensor_ping_;
    LocalTensor<INPUT_TYPE> l1_b_tensor_pong_;
    // l0_tensor
    LocalTensor<INPUT_TYPE> l0_a_tensor_ping_;
    LocalTensor<INPUT_TYPE> l0_a_tensor_pong_;
    LocalTensor<INPUT_TYPE> l0_b_tensor_ping_;
    LocalTensor<INPUT_TYPE> l0_b_tensor_pong_;
    LocalTensor<INPUT_TYPE> l0_c_tensor_ping_;
    LocalTensor<INPUT_TYPE> l0_c_tensor_pong_;

    static constexpr int32_t L1_SIZE = 512 * 1024;
    static constexpr int32_t L0_AB_SIZE = 64 * 1024;
    static constexpr int32_t BASE_BLOCK_SIZE = 128 * 128;

public:
    __aicore__ inline CubeOp(){};
    __aicore__ inline void Init(const TILING_CLASS *tilingData, TPipe *tPipe)
    {
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->kv_seq_len_ = tilingData->kvSeqLen;
        this->q_group_ = tilingData->qGroup;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        this->head_dim_align_ = RoundUp(head_dim_, 16);
        TBuf<TPosition::A1> l1_buffer_;
        TBuf<TPosition::A2> l0_a_buffer_;
        TBuf<TPosition::B2> l0_b_buffer_;
        TBuf<TPosition::C2> l0_c_buffer_;
        tPipe->InitBuffer(l1_buffer_, L1_SIZE);
        tPipe->InitBuffer(l0_a_buffer_, L0_AB_SIZE);
        tPipe->InitBuffer(l0_b_buffer_, L0_AB_SIZE);

        uint32_t l1_offset = 0;
        l1_a_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(128 * 1024 / sizeof(INPUT_TYPE), l1_offset);
        l1_offset += 128 * 1024;
        l1_a_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(128 * 1024 / sizeof(INPUT_TYPE), l1_offset);
        l1_offset += 128 * 1024;
        l1_b_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(128 * 1024 / sizeof(INPUT_TYPE), l1_offset);
        l1_offset += 128 * 1024;
        l1_b_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(128 * 1024 / sizeof(INPUT_TYPE), l1_offset);
        l1_offset += 128 * 1024;

        l0_a_tensor_ping_ = l0_a_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 0);
        l0_a_tensor_pong_ = l0_a_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 32 * 1024);
        l0_b_tensor_ping_ = l0_b_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 0);
        l0_b_tensor_pong_ = l0_b_buffer_.GetWithOffset<INPUT_TYPE>(32 * 1024 / sizeof(INPUT_TYPE), 32 * 1024);
        l0_c_tensor_ping_ = l0_c_buffer_.GetWithOffset<INPUT_TYPE>(64 * 1024 / sizeof(INPUT_TYPE), 0);
        l0_c_tensor_pong_ = l0_c_buffer_.GetWithOffset<INPUT_TYPE>(64 * 1024 / sizeof(INPUT_TYPE), 64 * 1024);
    }

    __aicore__ inline void ComputeMM12(GlobalTensor<INPUT_TYPE> leftGm, GlobalTensor<INPUT_TYPE> rightGm,
                                       RunTimeInfo *runTimeInfo)
    {
        int32_t mLoop = CeilDiv(runTimeInfo->s1Len, 128);
        int32_t nLoop = CeilDiv(runTimeInfo->s2Len, 128);
        int32_t mTail = (runTimeInfo->s1Len % 128) == 0 ? 128 : (runTimeInfo->s1Len % 128);
        int32_t nTail = (runTimeInfo->s2Len % 128) == 0 ? 128 : (runTimeInfo->s2Len % 128);

        for (int32_t nIdx = 0; nIdx < nLoop; nIdx++) {
            int32_t nProcess = (nIdx == nLoop - 1) ? nTail : 128;
            int32_t nProcessAlign = RoundUp(nProcess, 16);
            LocalTensor<INPUT_TYPE> l0_b_tensor = ping_pong_flag_outer ? l0_b_tensor_ping_ : l0_b_tensor_pong_;

            for (int32_t mIdx = 0; mIdx < mLoop; mIdx++) {
                int32_t mProcess = (mIdx == mLoop - 1) ? mTail : 128;
                int32_t mProcessAlign = RoundUp(mProcess, 16);
                LocalTensor<INPUT_TYPE> l1_a_tensor = ping_pong_flag_inner ? l1_a_tensor_ping_ : l1_a_tensor_pong_;
                LocalTensor<INPUT_TYPE> l0_a_tensor = ping_pong_flag_inner ? l0_a_tensor_ping_ : l0_a_tensor_pong_;

                load_data_gm_2_l0<true, false>(l0_a_tensor, l1_a_tensor, leftGm, mProcess, head_dim_, mProcessAlign,
                                               head_dim_align_, head_dim_);


                ping_pong_flag_inner = 1 - ping_pong_flag_inner;
            }

            ping_pong_flag_outer = 1 - ping_pong_flag_outer;
        }
    }

private:
    template <bool NZ>
    __aicore__ inline void load_data_gm_2_l1(LocalTensor<INPUT_TYPE> dstL1Tensor, GlobalTensor<INPUT_TYPE> srcGmTensor,
                                             const int32_t mSize, const int32_t kSize, const int32_t mSizeAlign,
                                             const int32_t kSizeAlign, const int32_t srcStride)
    {
        if constexpr (NZ) {
            AscendC::DataCopyParams intriParams;
            intriParams.blockCount = kSizeAlign / 16;
            intriParams.blockLen = mSizeAlign * 16 * sizeof(INPUT_TYPE) / 32;
            intriParams.srcStride = (srcStride - mSizeAlign) * 16 * sizeof(INPUT_TYPE) / 32;
            intriParams.dstStride = 0;
            DataCopy(dstL1Tensor, srcGmTensor, intriParams);
        } else {
            Nd2NzParams nd2nzPara;
            nd2nzPara.ndNum = 1;
            nd2nzPara.nValue = mSize;
            nd2nzPara.dValue = kSize;
            nd2nzPara.srcDValue = srcStride;
            nd2nzPara.dstNzC0Stride = mSizeAlign;
            nd2nzPara.dstNzNStride = 1;
            nd2nzPara.srcNdMatrixStride = 0;
            nd2nzPara.dstNzMatrixStride = 0;
            AscendC::DataCopy(dstL1Tensor, srcGmTensor, nd2nzPara);
        }
    }

    template <bool A_MATRIX, bool NZ>
    __aicore__ inline void load_data_gm_2_l0(LocalTensor<INPUT_TYPE> dstL0Tensor, LocalTensor<INPUT_TYPE> dstL1Tensor,
                                             GlobalTensor<INPUT_TYPE> srcGmTensor, const int32_t mSize,
                                             const int32_t kSize, const int32_t srcKSize, const int32_t mSizeAlign,
                                             const int32_t kSizeAlign, const bool mte2Copy)
    {
        if (mte2Copy) {
            load_data_gm_2_l1<NZ>(dstL1Tensor, srcGmTensor, mSize, kSize, mSizeAlign, kSizeAlign, srcKSize);
            SET_FLAG(MTE2, MTE1, EVENT_ID0);
            WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
        }

        AscendC::LoadData2dParams loadData2dParams;
        loadData2dParams.repeatTimes = kSizeAlign / 16;
        loadData2dParams.srcStride = mSizeAlign / 16;
        if constexpr (A_MATRIX) {
            loadData2dParams.ifTranspose = false;
        } else {
            loadData2dParams.ifTranspose = true;
        }

        for (int32_t i = 0; i < mSizeAlign / 16; i++) {
            AscendC::LoadData(dstL0Tensor[i * kSizeAlign * 16], dstL1Tensor[i * 256], loadData2dParams);
        }
    }

    template <bool A_MATRIX, bool NZ>
    __aicore__ inline void
    load_data_gm_2_l0_trans(LocalTensor<INPUT_TYPE> dstL0Tensor, LocalTensor<INPUT_TYPE> dstL1Tensor,
                            GlobalTensor<INPUT_TYPE> srcGmTensor, const int32_t mSize, const int32_t kSize,
                            const int32_t srcKSize, const int32_t mSizeAlign, const int32_t kSizeAlign,
                            const bool mte2Copy)
    {
        if (mte2Copy) {
            load_data_gm_2_l1<NZ>(dstL1Tensor, srcGmTensor, mSize, kSize, mSizeAlign, kSizeAlign, srcKSize);
            SET_FLAG(MTE2, MTE1, EVENT_ID0);
            WAIT_FLAG(MTE2, MTE1, EVENT_ID0);
        }

        AscendC::LoadData2dParams loadData2dParams;
        loadData2dParams.repeatTimes = mSizeAlign * kSizeAlign / 256;
        loadData2dParams.srcStride = 1;
        if constexpr (A_MATRIX) {
            loadData2dParams.ifTranspose = true;
        } else {
            loadData2dParams.ifTranspose = false;
        }

        AscendC::LoadData(dstL0Tensor, dstL1Tensor, loadData2dParams);
    }
};

} // namespace BSA_ARC35
