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
class VecOp {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    uint32_t v_core_num_;
    uint32_t v_core_idx_;
    uint32_t v_sub_core_idx_;
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
    float softmax_scale_{0.0f};
    uint32_t base_m;
    uint32_t base_n;
    uint32_t vec_base_m;
    uint32_t vec_base_n;
    GlobalTensor<float> lse_gm_;
    LocalTensor<float> lse_tensor_;
    LocalTensor<INPUT_TYPE> softmax_res_tensor_;
    LocalTensor<INPUT_TYPE> softmax_res_nz_tensor_;

    LocalTensor<float> vec_in_ping_;
    LocalTensor<float> vec_in_pong_;
    LocalTensor<INPUT_TYPE> vec_out_ping_;
    LocalTensor<INPUT_TYPE> vec_out_pong_;
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t C0_SIZE = 16;
    static constexpr uint32_t BLOCK_FP32 = BLOCK_SIZE / sizeof(float);
    static constexpr uint32_t BLOCK_INPUT = BLOCK_SIZE / sizeof(INPUT_TYPE);
    constexpr static uint32_t POST_TILE_LEN = 20 * 1024; // POST一次处理元素的个数

public:
    __aicore__ inline VecOp(){};

    __aicore__ inline void Init(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attention_out,
                                GM_ADDR softmaxLse, GM_ADDR blockSparseMask, GM_ADDR blockShape, GM_ADDR attentionMask,
                                GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                GM_ADDR workspace, const TILING_CLASS *tilingData, TBuf<TPosition::VECCALC> &ub_buffer,
                                uint32_t ub_offset)
    {
        this->v_core_num_ = tilingData->cubeCoreNum * 2;
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->kv_seq_len_ = tilingData->kvSeqLen;
        this->q_group_ = tilingData->qGroup;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        this->softmax_scale_ = tilingData->softmaxScale;
        this->head_dim_align_ = RoundUp(head_dim_, 16);
        this->base_m = tilingData->baseM;
        this->base_n = tilingData->baseN;
        this->vec_base_m = tilingData->baseM / 2;
        this->vec_base_n = tilingData->baseN;
        v_core_idx_ = GetBlockIdx();
        v_sub_core_idx_ = GetSubBlockIdx();

        // gm_tensor
        lse_gm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        // local_tensor
        lse_tensor_ = ub_buffer.GetWithOffset<float>(vec_base_m, ub_offset);
        ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);

        softmax_res_tensor_ = ub_buffer.GetWithOffset<INPUT_TYPE>(vec_base_m * vec_base_n, ub_offset);
        ub_offset += vec_base_m * vec_base_n * sizeof(INPUT_TYPE);
        softmax_res_nz_tensor_ = ub_buffer.GetWithOffset<INPUT_TYPE>(vec_base_m * vec_base_n, ub_offset);
        ub_offset += vec_base_m * vec_base_n * sizeof(INPUT_TYPE);
    }

    __aicore__ inline void SendVecSoftmax(const LocalTensor<INPUT_TYPE> &dst_l1_tensor,
                                          const LocalTensor<float> &src_ub_tensor, const RunTimeInfo &runTimeInfo)
    {
        CrossCoreWaitFlag<4, PIPE_V>(FLAG_C1_V1);
        ComputeSoftmax(dst_l1_tensor, src_ub_tensor, runTimeInfo);
        CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_V1_C3);
    }

    __aicore__ inline void SendVecSoftmaxGrad(const LocalTensor<INPUT_TYPE> &dst_l1_tensor,
                                              const LocalTensor<float> &src_ub_tensor, const RunTimeInfo &runTimeInfo)
    {
        // int32_t mLoop = runTimeInfo->s1Loop;
        // int32_t nLoop = runTimeInfo->s2Loop;
        // int32_t mTail = runTimeInfo->s1Tail;
        // int32_t nTail = runTimeInfo->s2Tail;

        // for (int32_t nIdx = 0; nIdx < nLoop; nIdx++) {
        //     int32_t nProcess = (nIdx == nLoop - 1) ? nTail : 128;
        //     int32_t n_process_align = RoundUp(nProcess, 16);

        //     for (int32_t mIdx = 0; mIdx < mLoop; mIdx++) {
        //         int32_t mProcess = (mIdx == mLoop - 1) ? mTail : 128;
        //         int32_t mProcessAlign = RoundUp(mProcess, 16);

        //         CrossCoreWaitFlag<2, PIPE_FIX>(FLAG_C1_V1);
        //         ping_pong_flag_inner = 1 - ping_pong_flag_inner;
        //     }

        //     ping_pong_flag_outer = 1 - ping_pong_flag_outer;
        // }
    }


    __aicore__ inline void SendVecPost(const GlobalTensor<INPUT_TYPE> &dq_out_gm,
                                       const GlobalTensor<INPUT_TYPE> &dk_out_gm,
                                       const GlobalTensor<INPUT_TYPE> &dv_out_gm,
                                       const GlobalTensor<float> &dq_workspace, const GlobalTensor<float> &dk_workspace,
                                       const GlobalTensor<float> &dv_workspace, const TILING_CLASS *tilingData,
                                       TBuf<TPosition::VECCALC> &ub_buffer)
    {
        uint64_t dq_size = tilingData->dqSize;
        uint32_t ub_offset = 0;
        vec_in_ping_ = ub_buffer.GetWithOffset<float>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(float);
        vec_in_pong_ = ub_buffer.GetWithOffset<float>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(float);
        vec_out_ping_ = ub_buffer.GetWithOffset<INPUT_TYPE>(POST_TILE_LEN, ub_offset);
        ub_offset = POST_TILE_LEN * sizeof(INPUT_TYPE);
        vec_out_pong_ = ub_buffer.GetWithOffset<INPUT_TYPE>(POST_TILE_LEN, ub_offset);
        ub_offset = POST_TILE_LEN * sizeof(INPUT_TYPE);

        ComputeVecPost<false>(dv_out_gm, dv_workspace, tilingData->dkSize);
    }

private:
    __aicore__ inline void ComputeSoftmax(const LocalTensor<INPUT_TYPE> &dst_l1_tensor,
                                          const LocalTensor<float> &src_ub_tensor, const RunTimeInfo &runTimeInfo)
    { /*
       * function: Compute simple softmax
       * input shape：[s1LenAlign / 2, s2LenAlign]
       * out shape:   [s1LenAlign / 2, s2LenAlign]
       * dtype:       float
       */
        int32_t m_process = runTimeInfo.s1Len;
        int32_t sub_m_process = runTimeInfo.s1LenAlign / 2;
        int32_t n_process_align = runTimeInfo.s2LenAlign;
        int32_t data_size = sub_m_process * n_process_align;
        int32_t sub_real_m_process =
            v_sub_core_idx_ == 0 ? sub_m_process : (m_process - sub_m_process); // GM CopyIn仍采用实际的mProcess
        uint64_t lse_offset = runTimeInfo.lseGmOffset + v_sub_core_idx_ * sub_m_process * q_head_num_;
        uint64_t l1_offset = v_sub_core_idx_ * sub_m_process * C0_SIZE;
        printf("lse_offset: %d\n", lse_offset);

        Muls(src_ub_tensor, src_ub_tensor, softmax_scale_, data_size);
        PipeBarrier<PIPE_V>();

        CopyInLSE(lse_tensor_, lse_gm_[lse_offset], sub_real_m_process, 0);
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);

        SubBrcb(src_ub_tensor, src_ub_tensor, lse_tensor_, sub_m_process, n_process_align);
        PipeBarrier<PIPE_V>();

        Exp(src_ub_tensor, src_ub_tensor, data_size);
        PipeBarrier<PIPE_V>();

        Cast(softmax_res_tensor_, src_ub_tensor, AscendC::RoundMode::CAST_RINT, data_size);
        PipeBarrier<PIPE_V>();
        DumpTensor(src_ub_tensor, 0, 32);

        ND2NZ(softmax_res_nz_tensor_, softmax_res_tensor_, sub_m_process, n_process_align);
        PipeBarrier<PIPE_V>();
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);

        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = n_process_align / C0_SIZE;
        dataCopyParams.blockLen = sub_m_process * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (m_process - sub_m_process) * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        DataCopy(dst_l1_tensor[l1_offset], softmax_res_nz_tensor_, dataCopyParams);
    }

    __aicore__ inline void CopyInLSE(const LocalTensor<float> &dstTensor, const GlobalTensor<float> &srcTensor,
                                     const int32_t count, const int32_t srcStride)
    {
        /*
         * function: Copy lse from global memory to local memory
         * input shape：(b, n, s, 1) or (t, n, 1)
         * out shape:   (s, 8)
         * dtype:       float
         */

        DataCopyPad(dstTensor, srcTensor,
                    {static_cast<uint16_t>(count), static_cast<uint32_t>(1 * sizeof(float)),
                     static_cast<uint32_t>(srcStride), 0, 0},
                    {false, 0, 0, 0});
    }


    __aicore__ inline void SubBrcb(const LocalTensor<float> &dstTensor, const LocalTensor<float> &src0Tensor,
                                   const LocalTensor<float> &src1Tensor, const uint64_t row, const uint64_t col)
    {
        /*
         * function: dstTensor = src0Tensor - src1Tensor
         * input shape：[m_align, n_align]
         * out shape:   [m_align, n_align]
         * dtype:       float
         */
        for (uint32_t i = 0; i < row; i++) {
            float sub_value = src1Tensor.GetValue(i * BLOCK_FP32);
            SET_FLAG(S, V, EVENT_ID0);
            WAIT_FLAG(S, V, EVENT_ID0);
            Subs(dstTensor[i * col], src0Tensor[i * col], sub_value, col);
        }
    }

    __aicore__ inline void ND2NZ(const LocalTensor<INPUT_TYPE> &dstTensor, const LocalTensor<INPUT_TYPE> &srcTensor,
                                 const uint64_t row, const uint64_t col)
    {
        uint64_t src_offset = 0;
        uint64_t dst_offset = 0;
        uint32_t C0_NUM = CeilDiv<uint32_t>(col, C0_SIZE);

        DataCopyParams repeatParams;
        repeatParams.blockCount = C0_NUM;
        repeatParams.blockLen = C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        repeatParams.srcStride = 0;
        repeatParams.dstStride = (row - 1) * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;

        for (uint64_t i = 0; i < row; i++) {
            src_offset = i * col;
            dst_offset = i * C0_SIZE;
            DataCopy(dstTensor[dst_offset], srcTensor[src_offset], repeatParams);
        }
    }

    template <bool MULS>
    __aicore__ inline void ComputeVecPost(const GlobalTensor<INPUT_TYPE> &dst_tensor,
                                          const GlobalTensor<float> &src_tensor, uint64_t totalElem)
    {
        LocalTensor<float> vecIn = vec_in_ping_;
        LocalTensor<INPUT_TYPE> vecOut = vec_out_ping_;
        uint64_t perCoreElem = (totalElem + v_core_num_ - 1) / v_core_num_;
        uint64_t coreStart = v_core_idx_ * perCoreElem;
        uint64_t coreEnd = (coreStart + perCoreElem > totalElem) ? totalElem : (coreStart + perCoreElem);
        uint32_t coreElem = coreEnd - coreStart;
        if (coreStart >= totalElem) {
            return;
        }
        uint32_t loop = CeilDiv<uint64_t>(coreElem, POST_TILE_LEN);
        uint32_t process_size = POST_TILE_LEN;
        uint32_t tail = coreElem % POST_TILE_LEN;
        // 由于DataCopyPad最多处理65535，因此tail部分分成align_tail和pad_tail计算
        uint32_t pad_tail, align_tail;
        if (tail <= 65535) {
            pad_tail = tail;
            align_tail = 0;
        } else {
            align_tail = tail / 16 * 16;
            pad_tail = tail - align_tail;
        }

        for (uint64_t i = 0; i < loop; i++) {
            if (unlikely(i == loop - 1)) {
                if (align_tail == 0) {
                    break;
                }
                process_size = align_tail;
            }
            uint64_t gm_offset = coreStart + i * POST_TILE_LEN;

            DataCopy(vecIn, src_tensor[gm_offset], process_size); // dataSize(fp32) align 32B
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);

            if constexpr (MULS) {
                Muls(vecIn, vecIn, softmax_scale_, process_size);
                PipeBarrier<PIPE_V>();
            }
            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_RINT, process_size);
            PipeBarrier<PIPE_V>();

            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            DataCopy(dst_tensor[gm_offset], vecOut, process_size);
        }

        if (pad_tail > 0) {
            uint32_t pad_tail_align = RoundUp<uint32_t>(pad_tail, 16);
            uint64_t gm_offset = coreStart + (loop - 1) * POST_TILE_LEN + align_tail;
            DataCopyParams copyParam;
            copyParam.blockCount = 1;
            copyParam.blockLen = pad_tail * sizeof(float);
            copyParam.srcStride = 0;
            copyParam.dstStride = 0;
            DataCopyPad(vecIn, src_tensor[gm_offset], copyParam, {false, 0, 0, 0});
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            if constexpr (MULS) {
                Muls(vecIn, vecIn, softmax_scale_, pad_tail_align);
                PipeBarrier<PIPE_V>();
            }

            Cast(vecOut, vecIn, AscendC::RoundMode::CAST_RINT, pad_tail_align);
            PipeBarrier<PIPE_V>();

            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            copyParam.blockLen = pad_tail * sizeof(INPUT_TYPE);
            DataCopyPad(dst_tensor[gm_offset], vecOut, copyParam);
        }
    }
};

} // namespace BSA_ARC35
