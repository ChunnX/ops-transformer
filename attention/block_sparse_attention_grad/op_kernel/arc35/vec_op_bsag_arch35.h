/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "vector_api/vf_cast_nd2nz.h"
using namespace AscendC;

namespace BSA_ARC35 {

struct EvenCoreInfo {
    uint32_t data_size{0};        // 总共需要处理的元素个数
    uint32_t start_idx{0};        // 每个core处理的起始idx
    uint32_t len{0};              // 每个core处理的元素个数
    uint32_t loop_num{0};         // 每个core需要循环的次数
    uint32_t max_process_size{0}; // 每个core单次循环最大处理的元素个数
    uint32_t align_tail{0};       // 每个core处理的对齐尾块, tail = align_tail + pad_tail
    uint32_t pad_tail{0};         // 每个core处理的pad尾块, tail = align_tail + pad_tail
};

template <typename BSA_TYPE>
class VecOp {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
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
    GM_ADDR act_seq_q_len;
    GlobalTensor<float> lse_gm_;
    GlobalTensor<float> sftg_workspace_;
    GlobalTensor<float> det_dq_gm_;
    GlobalTensor<float> det_dk_gm_;
    GlobalTensor<float> det_dv_gm_;
    GlobalTensor<int64_t> deter_offset_gm_;
    GM_ADDR det_dq_base_{nullptr};
    GM_ADDR det_dk_base_{nullptr};
    GM_ADDR det_dv_base_{nullptr};
    uint32_t cube_core_num_{0};
    uint32_t cube_core_idx_{0};
    uint64_t det_dq_tile_elems_{0};
    uint64_t det_dkv_tile_elems_{0};
    uint32_t q_stride_{0};
    uint32_t kv_stride_{0};
    LocalTensor<float> lse_tensor_;
    LocalTensor<float> lse_tensor_ping_;
    LocalTensor<float> lse_tensor_pong_;
    LocalTensor<float> sftg_front_tensor_;
    LocalTensor<float> sftg_front_tensor_ping_;
    LocalTensor<float> sftg_front_tensor_pong_;
    LocalTensor<INPUT_TYPE> softmax_res_nz_tensor_;
    LocalTensor<INPUT_TYPE> sftg_res_nz_tensor_;
    LocalTensor<int64_t> deter_offset_tensor_;
    LocalTensor<float> det_acc_ub_;
    LocalTensor<float> det_src_ub_;

    LocalTensor<float> vec_in_ping_;
    LocalTensor<float> vec_in_pong_;
    LocalTensor<INPUT_TYPE> vec_out_ping_;
    LocalTensor<INPUT_TYPE> vec_out_pong_;
    // sftg
    LocalTensor<INPUT_TYPE> dy_in_ping_;
    LocalTensor<INPUT_TYPE> dy_in_pong_;
    LocalTensor<INPUT_TYPE> attention_in_ping_;
    LocalTensor<INPUT_TYPE> attention_in_pong_;
    LocalTensor<float> dy_out_ping_;
    LocalTensor<float> dy_out_pong_;
    LocalTensor<float> attention_out_ping_;
    LocalTensor<float> attention_out_pong_;
    LocalTensor<float> sftg_front_ping_;
    LocalTensor<float> sftg_front_pong_;
    LocalTensor<uint8_t> sftg_tmp_tensor;
    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t C0_SIZE = 16;
    static constexpr uint32_t BLOCK_FP32 = BLOCK_SIZE / sizeof(float);
    static constexpr uint32_t BLOCK_INPUT = BLOCK_SIZE / sizeof(INPUT_TYPE);
    constexpr static uint32_t PRE_TILE_LEN = 60 * 1024;  // pre一次处理元素的个数
    constexpr static uint32_t POST_TILE_LEN = 20 * 1024; // POST一次处理元素的个数
    constexpr static int64_t OUTINDEX = -1;
    constexpr static uint32_t INT64_BLOCK_NUM = 32 / sizeof(int64_t);
    constexpr static uint32_t DET_OFFSET_NUM = 4;
    constexpr static uint32_t MAX_CUBE_CORES = 64;

    struct DeterAddrGroup {
        uint32_t group_num{0};
        uint32_t group_begin[MAX_CUBE_CORES];
        uint32_t group_size[MAX_CUBE_CORES];
        int64_t group_off[MAX_CUBE_CORES];
        int64_t group_len[MAX_CUBE_CORES];
        uint32_t core_list[MAX_CUBE_CORES];
    };
    // runtInfo
    int32_t s1_process_;
    int32_t s1_process_align_;
    int32_t s2_process_align_;
    int32_t half_s1_process_align_;
    int32_t data_size;
    int32_t half_s1_process_real_;
    int64_t lse_gm_offset_;
    int64_t sftg_gm_offset_;
    int64_t l1_offset_;
    TEventID event_ping_ = EVENT_ID3;
    TEventID event_pong_ = EVENT_ID4;
    TEventID event_id;

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
        this->act_seq_q_len = actualQseqlen;
        v_core_idx_ = GetBlockIdx();
        v_sub_core_idx_ = GetSubBlockIdx();

        if constexpr (INPUT_LAYOUT == BSND) {
            q_stride_ = q_head_num_ * head_dim_;
            kv_stride_ = kv_head_num_ * head_dim_;
        } else if constexpr (INPUT_LAYOUT == BNSD) {
            q_stride_ = head_dim_;
            kv_stride_ = head_dim_;
        } else {
            q_stride_ = q_head_num_ * head_dim_;
            kv_stride_ = kv_head_num_ * head_dim_;
        }

        // gm_tensor
        lse_gm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        sftg_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->sftgWorkspaceOffset));
        if constexpr (DETERMINISTIC_ENABLE) {
            det_dq_tile_elems_ = static_cast<uint64_t>(base_m) * head_dim_;
            det_dkv_tile_elems_ = static_cast<uint64_t>(base_n) * head_dim_;
            cube_core_num_ = static_cast<uint32_t>(tilingData->cubeCoreNum);
            cube_core_idx_ = v_core_idx_ / 2;
            det_dq_base_ = workspace + tilingData->detDqWorkspaceOffset;
            det_dk_base_ = workspace + tilingData->detDkWorkspaceOffset;
            det_dv_base_ = workspace + tilingData->detDvWorkspaceOffset;
            det_dq_gm_.SetGlobalBuffer(
                (__gm__ float *)(det_dq_base_ + cube_core_idx_ * det_dq_tile_elems_ * sizeof(float)));
            det_dk_gm_.SetGlobalBuffer(
                (__gm__ float *)(det_dk_base_ + cube_core_idx_ * det_dkv_tile_elems_ * sizeof(float)));
            det_dv_gm_.SetGlobalBuffer(
                (__gm__ float *)(det_dv_base_ + cube_core_idx_ * det_dkv_tile_elems_ * sizeof(float)));
            deter_offset_gm_.SetGlobalBuffer((__gm__ int64_t *)(workspace + tilingData->detOffsetWorkspaceOffset));
        }
        /*
         * UB 布局（主循环常驻，从 Process 传入的 ub_offset 接着排）
         * 总预算 UB_SIZE = 247KB。Process 已占用：
         *   mm1/mm2 pingpong: 4 * vec_base_m * vec_base_n * sizeof(float)
         *   例 baseM=N=128: 4 * 64 * 128 * 4 = 128KB
         * 本 Init 继续占用：
         *   softmax_res_nz / sftg_res_nz: 2 * vec_base_m * vec_base_n * sizeof(INPUT_TYPE)
         *   lse ping/pong + sftg_front ping/pong: 4 * vec_base_m * BLOCK_FP32 * sizeof(float)
         *   deter_offset: cube_core_num * DET_OFFSET_NUM * sizeof(int64_t)
         *   det_acc / det_src: 2 * CeilDiv(max(baseM,baseN), v_core_num) * head_dim * sizeof(float)
         * 例 fp16 + cubeCoreNum=5: VecOp 常驻 ≈ 40KB + 13.2KB，主循环合计 ≈ 181KB，剩余 ≈ 66KB。
         * Pre / SftgFront / Post 从 offset 0 覆盖整块 UB，与主循环错开。
         */
        softmax_res_nz_tensor_ = ub_buffer.GetWithOffset<INPUT_TYPE>(vec_base_m * vec_base_n, ub_offset);
        ub_offset += vec_base_m * vec_base_n * sizeof(INPUT_TYPE);
        sftg_res_nz_tensor_ = ub_buffer.GetWithOffset<INPUT_TYPE>(vec_base_m * vec_base_n, ub_offset);
        ub_offset += vec_base_m * vec_base_n * sizeof(INPUT_TYPE);
        lse_tensor_ping_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
        ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
        lse_tensor_pong_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
        ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
        sftg_front_tensor_ping_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
        ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
        sftg_front_tensor_pong_ = ub_buffer.GetWithOffset<float>(vec_base_m * BLOCK_FP32, ub_offset);
        ub_offset += vec_base_m * BLOCK_FP32 * sizeof(float);
        if constexpr (DETERMINISTIC_ENABLE) {
            uint32_t offset_elems = cube_core_num_ * DET_OFFSET_NUM;
            deter_offset_tensor_ = ub_buffer.GetWithOffset<int64_t>(offset_elems, ub_offset);
            ub_offset += offset_elems * sizeof(int64_t);
            uint32_t max_axis = base_m > base_n ? base_m : base_n;
            uint32_t max_my_rows = CeilDiv<uint32_t>(max_axis, v_core_num_);
            uint32_t ub_tile = max_my_rows * static_cast<uint32_t>(head_dim_);
            det_acc_ub_ = ub_buffer.GetWithOffset<float>(ub_tile, ub_offset);
            ub_offset += ub_tile * sizeof(float);
            det_src_ub_ = ub_buffer.GetWithOffset<float>(ub_tile, ub_offset);
            ub_offset += ub_tile * sizeof(float);
        }
    }

    __aicore__ inline void SendVecPre(const GlobalTensor<float> &dq_workspace, const GlobalTensor<float> &dk_workspace,
                                      const GlobalTensor<float> &dv_workspace, const GlobalTensor<INPUT_TYPE> &dy_gm,
                                      const GlobalTensor<INPUT_TYPE> &out_gm, const GlobalTensor<float> &sftg_workspace,
                                      const TILING_CLASS *tilingData, TBuf<TPosition::VECCALC> &ub_buffer)
    {
        constexpr static uint32_t PRE_TILE_LEN = 20 * 1024; // PRE一次处理元素的个数
        EvenCoreInfo info;
        vec_in_ping_ = ub_buffer.GetWithOffset<float>(PRE_TILE_LEN, 0);
        Duplicate(vec_in_ping_, (float)0.0, PRE_TILE_LEN);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        ComputeEvenCoreInfo(info, tilingData->dqSize, PRE_TILE_LEN);
        ComputeVecPre(dq_workspace, vec_in_ping_, info);
        ComputeEvenCoreInfo(info, tilingData->dkSize, PRE_TILE_LEN);
        ComputeVecPre(dk_workspace, vec_in_ping_, info);
        ComputeVecPre(dv_workspace, vec_in_ping_, info);
    }

    __aicore__ inline void WriteInvalidOffsetToGM(uint32_t pingpong_idx)
    {
        WriteOffsetToGM(pingpong_idx, OUTINDEX, OUTINDEX, 0, 0, 0);
    }

    /*
     * 从 GM 一次读齐所有 cube 核 offset，按输出地址归类后统一累加写回。
     * dq 按 s1 切分：每个 vector 核处理同一段 s1_idx；dk/dv 按 s2 切分：每个 vector 核处理同一段 s2_idx。
     */
    __aicore__ inline void SendVecDeterPost(const GlobalTensor<float> &dq_workspace,
                                            const GlobalTensor<float> &dk_workspace,
                                            const GlobalTensor<float> &dv_workspace, uint32_t pingpong_idx)
    {
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        uint32_t offset_elems = cube_core_num_ * DET_OFFSET_NUM;
        DataCopy(deter_offset_tensor_, deter_offset_gm_[pingpong_idx * offset_elems], offset_elems);
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);

        int64_t q_off[MAX_CUBE_CORES];
        int64_t s1_len[MAX_CUBE_CORES];
        int64_t k_off[MAX_CUBE_CORES];
        int64_t s2_len[MAX_CUBE_CORES];
        CollectDeterCoreInfo(q_off, s1_len, k_off, s2_len);

        DeterAtomicAddOneOutput(dq_workspace, det_dq_base_, det_dq_tile_elems_, q_off, s1_len, q_stride_, pingpong_idx,
                                static_cast<uint32_t>(base_m));
        DeterAtomicAddOneOutput(dk_workspace, det_dk_base_, det_dkv_tile_elems_, k_off, s2_len, kv_stride_, pingpong_idx,
                                static_cast<uint32_t>(base_n));
        DeterAtomicAddOneOutput(dv_workspace, det_dv_base_, det_dkv_tile_elems_, k_off, s2_len, kv_stride_, pingpong_idx,
                                static_cast<uint32_t>(base_n));
    }

    __aicore__ inline void SendVecSftgFront(const GlobalTensor<INPUT_TYPE> &dy_gm,
                                            const GlobalTensor<INPUT_TYPE> &out_gm,
                                            const GlobalTensor<float> &sftg_workspace, const TILING_CLASS *tilingData,
                                            TBuf<TPosition::VECCALC> &ub_buffer)
    {
        EvenCoreInfo info;
        constexpr static uint32_t SFTG_TILE_LEN = 8 * 1024; // POST一次处理元素的个数
        uint32_t process_s1_size = SFTG_TILE_LEN / head_dim_;
        uint32_t ub_offset = 0;
        uint32_t sftg_ping_pong_idx = 0;
        dy_in_ping_ = ub_buffer.GetWithOffset<INPUT_TYPE>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(INPUT_TYPE);
        dy_in_pong_ = ub_buffer.GetWithOffset<INPUT_TYPE>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(INPUT_TYPE);
        attention_in_ping_ = ub_buffer.GetWithOffset<INPUT_TYPE>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(INPUT_TYPE);
        attention_in_pong_ = ub_buffer.GetWithOffset<INPUT_TYPE>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(INPUT_TYPE);
        dy_out_ping_ = ub_buffer.GetWithOffset<float>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(float);
        dy_out_pong_ = ub_buffer.GetWithOffset<float>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(float);
        attention_out_ping_ = ub_buffer.GetWithOffset<float>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(float);
        attention_out_pong_ = ub_buffer.GetWithOffset<float>(SFTG_TILE_LEN, ub_offset);
        ub_offset += SFTG_TILE_LEN * sizeof(float);
        sftg_front_ping_ = ub_buffer.GetWithOffset<float>(process_s1_size * 8, ub_offset);
        ub_offset += process_s1_size * 8 * sizeof(float);
        sftg_front_pong_ = ub_buffer.GetWithOffset<float>(process_s1_size * 8, ub_offset);
        ub_offset += process_s1_size * 8 * sizeof(float);
        sftg_tmp_tensor = ub_buffer.GetWithOffset<uint8_t>(tilingData->sftgTmpSpaceSize, ub_offset);
        ub_offset += tilingData->sftgTmpSpaceSize;

        SET_FLAG(MTE3, MTE2, event_ping_);
        SET_FLAG(MTE3, MTE2, event_pong_);
        for (uint32_t b_idx = 0; b_idx < batch_num_; b_idx++) {
            int64_t current_q_seqlen;
            int64_t last_seq_total_len;
            if constexpr (INPUT_LAYOUT == TND) {
                current_q_seqlen = GetSeqLen(b_idx, act_seq_q_len);
                last_seq_total_len = b_idx == 0 ? 0 : GetSeqTotalLen(b_idx - 1, act_seq_q_len);
            } else {
                current_q_seqlen = q_seq_len_;
                last_seq_total_len = 0;
            }
            for (uint32_t n1_idx = 0; n1_idx < q_head_num_; n1_idx++) {
                int64_t in_gm_offset;
                int64_t out_gm_offset;

                if constexpr (INPUT_LAYOUT == BSND) {
                    in_gm_offset = b_idx * current_q_seqlen * q_head_num_ * head_dim_ + n1_idx * head_dim_;
                    out_gm_offset = b_idx * q_head_num_ * current_q_seqlen * 8 + n1_idx * current_q_seqlen * 8;
                } else if constexpr (INPUT_LAYOUT == BNSD) {
                    in_gm_offset =
                        b_idx * q_head_num_ * current_q_seqlen * head_dim_ + n1_idx * current_q_seqlen * head_dim_;
                    out_gm_offset = b_idx * q_head_num_ * current_q_seqlen * 8 + n1_idx * current_q_seqlen * 8;
                } else if constexpr (INPUT_LAYOUT == TND) {
                    in_gm_offset = last_seq_total_len * q_head_num_ * head_dim_ + n1_idx * head_dim_;
                    out_gm_offset = last_seq_total_len * q_head_num_ * 8 + n1_idx * current_q_seqlen * 8;
                }
                TEventID event_id = sftg_ping_pong_idx ? event_ping_ : event_pong_;
                ComputeEvenCoreInfo(info, current_q_seqlen, process_s1_size);

                WAIT_FLAG(MTE3, MTE2, event_id);
                ComputeSoftmaxGradFront(sftg_workspace[out_gm_offset], dy_gm[in_gm_offset], out_gm[in_gm_offset], info,
                                        tilingData, sftg_ping_pong_idx);
                SET_FLAG(MTE3, MTE2, event_id);
                sftg_ping_pong_idx = 1 - sftg_ping_pong_idx;
            }
        }

        WAIT_FLAG(MTE3, MTE2, event_ping_);
        WAIT_FLAG(MTE3, MTE2, event_pong_);
    }

    __aicore__ inline void SendVecSftPreProcess(const RunTimeInfo &runTimeInfo, uint32_t pingpong_idx)
    {
        s1_process_ = runTimeInfo.s1Len;
        s1_process_align_ = runTimeInfo.s1LenAlign;
        s2_process_align_ = runTimeInfo.s2LenAlign;
        half_s1_process_align_ = s1_process_align_ / 2;
        data_size = half_s1_process_align_ * s2_process_align_;
        half_s1_process_real_ = v_sub_core_idx_ == 0 ?
                                    half_s1_process_align_ :
                                    (s1_process_ - half_s1_process_align_); // GM CopyIn仍采用实际的mProcess
        sftg_gm_offset_ = runTimeInfo.sftgGmOffset + v_sub_core_idx_ * half_s1_process_align_ * 8;
        if constexpr (INPUT_LAYOUT == TND) {
            lse_gm_offset_ = runTimeInfo.lseGmOffset + v_sub_core_idx_ * half_s1_process_align_ * q_head_num_;
        } else {
            lse_gm_offset_ = runTimeInfo.lseGmOffset + v_sub_core_idx_ * half_s1_process_align_;
        }
        l1_offset_ = v_sub_core_idx_ * half_s1_process_align_ * C0_SIZE;

        if (half_s1_process_real_ > 0) {
            lse_tensor_ = pingpong_idx == 0 ? lse_tensor_ping_ : lse_tensor_pong_;
            sftg_front_tensor_ = pingpong_idx == 0 ? sftg_front_tensor_ping_ : sftg_front_tensor_pong_;
            event_id = pingpong_idx == 0 ? event_ping_ : event_pong_;

            WAIT_FLAG(V, MTE2, event_id);
            CopyInLSE(lse_tensor_, lse_gm_[lse_gm_offset_], half_s1_process_real_);
            DataCopy(sftg_front_tensor_, sftg_workspace_[sftg_gm_offset_], half_s1_process_real_ * 8);
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
        }
    }

    __aicore__ inline void SendVecSoftmax(const LocalTensor<INPUT_TYPE> &dst_l1_tensor,
                                          const LocalTensor<float> &src_ub_tensor, const RunTimeInfo &runTimeInfo)
    { /*
       * function: Compute simple softmax
       * input shape：[s1LenAlign / 2, s2LenAlign]
       * out shape:   [s1LenAlign / 2, s2LenAlign]
       * dtype:       float
       */
        if (half_s1_process_real_ <= 0) {
            return;
        }

        SimpleSoftmax((__ubuf__ float *)src_ub_tensor.GetPhyAddr(), (__ubuf__ float *)src_ub_tensor.GetPhyAddr(),
                      (__ubuf__ float *)lse_tensor_.GetPhyAddr(), half_s1_process_align_, s2_process_align_);

        CastND2NZ<INPUT_TYPE>(softmax_res_nz_tensor_, src_ub_tensor, half_s1_process_align_, s2_process_align_);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);

        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s2_process_align_ / C0_SIZE;
        dataCopyParams.blockLen = half_s1_process_align_ * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride =
            (s1_process_align_ - half_s1_process_align_) * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        DataCopy(dst_l1_tensor[l1_offset_], softmax_res_nz_tensor_, dataCopyParams);
    }

    __aicore__ inline void SendVecSoftmaxGrad(const LocalTensor<INPUT_TYPE> &dst_l1_tensor,
                                              const LocalTensor<float> &softmax_ub_tensor,
                                              const LocalTensor<float> &src_ub_tensor, const RunTimeInfo &runTimeInfo)
    {
        /*
         * function: Compute softmaxGrad
         * input shape：[s1LenAlign / 2, s2LenAlign]
         * out shape:   [s1LenAlign / 2, s2LenAlign]
         * dtype:       float
         */
        if (half_s1_process_real_ <= 0) {
            return;
        }
        ComputeSoftmaxGrad((__ubuf__ float *)src_ub_tensor.GetPhyAddr(), (__ubuf__ float *)src_ub_tensor.GetPhyAddr(),
                           (__ubuf__ float *)softmax_ub_tensor.GetPhyAddr(),
                           (__ubuf__ float *)sftg_front_tensor_.GetPhyAddr(), half_s1_process_align_,
                           s2_process_align_);

        CastND2NZ<INPUT_TYPE>(sftg_res_nz_tensor_, src_ub_tensor, half_s1_process_align_, s2_process_align_);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);

        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = s2_process_align_ / C0_SIZE;
        dataCopyParams.blockLen = half_s1_process_align_ * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride =
            (s1_process_align_ - half_s1_process_align_) * C0_SIZE * sizeof(INPUT_TYPE) / BLOCK_SIZE;
        DataCopy(dst_l1_tensor[l1_offset_], sftg_res_nz_tensor_, dataCopyParams);
        SET_FLAG(V, MTE2, event_id);
    }

    __aicore__ inline void SendVecPost(const GlobalTensor<INPUT_TYPE> &dq_out_gm,
                                       const GlobalTensor<INPUT_TYPE> &dk_out_gm,
                                       const GlobalTensor<INPUT_TYPE> &dv_out_gm,
                                       const GlobalTensor<float> &dq_workspace, const GlobalTensor<float> &dk_workspace,
                                       const GlobalTensor<float> &dv_workspace, const TILING_CLASS *tilingData,
                                       TBuf<TPosition::VECCALC> &ub_buffer)
    {
        int64_t dq_size = tilingData->dqSize;
        int64_t dkv_size = tilingData->dkSize;
        uint32_t ub_offset = 0;
        uint32_t post_ping_pong_idx = 0;
        vec_in_ping_ = ub_buffer.GetWithOffset<float>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(float);
        vec_in_pong_ = ub_buffer.GetWithOffset<float>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(float);
        vec_out_ping_ = ub_buffer.GetWithOffset<INPUT_TYPE>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(INPUT_TYPE);
        vec_out_pong_ = ub_buffer.GetWithOffset<INPUT_TYPE>(POST_TILE_LEN, ub_offset);
        ub_offset += POST_TILE_LEN * sizeof(INPUT_TYPE);
        SET_FLAG(MTE3, MTE2, event_ping_);
        SET_FLAG(MTE3, MTE2, event_pong_);

        EvenCoreInfo info;
        ComputeEvenCoreInfo(info, dkv_size, POST_TILE_LEN);
        ComputeVecPost<false>(dv_out_gm, dv_workspace, info, post_ping_pong_idx);
        ComputeVecPost<true>(dk_out_gm, dk_workspace, info, post_ping_pong_idx);
        ComputeEvenCoreInfo(info, dq_size, POST_TILE_LEN);
        ComputeVecPost<true>(dq_out_gm, dq_workspace, info, post_ping_pong_idx);

        WAIT_FLAG(MTE3, MTE2, event_ping_);
        WAIT_FLAG(MTE3, MTE2, event_pong_);
    }

private:

    __aicore__ inline void WriteOffsetToGM(uint32_t pingpong_idx, int64_t queryGmOffset, int64_t keyGmOffset,
                                           int64_t s1Len, int64_t s2Len, int64_t is_singlekv_last)
    {
        if constexpr (!DETERMINISTIC_ENABLE) {
            return;
        }
        if(v_sub_core_idx_ != 0) {
            return;
        }
        int64_t gm_offset = pingpong_idx * cube_core_num_ * DET_OFFSET_NUM + cube_core_idx_ * DET_OFFSET_NUM;
        deter_offset_tensor_.SetValue(0, queryGmOffset);
        deter_offset_tensor_.SetValue(1, s1Len);
        if (is_singlekv_last) {
            deter_offset_tensor_.SetValue(2, keyGmOffset);
            deter_offset_tensor_.SetValue(3, s2Len);
        } else {
            deter_offset_tensor_.SetValue(2, OUTINDEX);
            deter_offset_tensor_.SetValue(3, 0);
        }

        SET_FLAG(S, MTE3, EVENT_ID0);
        WAIT_FLAG(S, MTE3, EVENT_ID0);
        DataCopy(deter_offset_gm_[gm_offset], deter_offset_tensor_, DET_OFFSET_NUM);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    }

    __aicore__ inline void GetVecAxisSlice(uint32_t total_rows, uint32_t &my_start, uint32_t &my_len)
    {
        uint32_t rows_per_core = CeilDiv<uint32_t>(total_rows, v_core_num_);
        my_start = v_core_idx_ * rows_per_core;
        if (my_start >= total_rows) {
            my_len = 0;
            return;
        }
        uint32_t remain = total_rows - my_start;
        my_len = remain < rows_per_core ? remain : rows_per_core;
    }

    __aicore__ inline void CollectDeterCoreInfo(int64_t *q_off, int64_t *s1_len, int64_t *k_off, int64_t *s2_len)
    {
        for (uint32_t c = 0; c < cube_core_num_; c++) {
            uint32_t base = c * DET_OFFSET_NUM;
            q_off[c] = deter_offset_tensor_.GetValue(base);
            s1_len[c] = deter_offset_tensor_.GetValue(base + 1);
            k_off[c] = deter_offset_tensor_.GetValue(base + 2);
            s2_len[c] = deter_offset_tensor_.GetValue(base + 3);
        }
    }

    __aicore__ inline void GroupSameAddr(const int64_t *gm_off, const int64_t *row_len, DeterAddrGroup &group)
    {
        uint8_t used[MAX_CUBE_CORES] = {0};
        group.group_num = 0;
        uint32_t list_pos = 0;
        for (uint32_t c0 = 0; c0 < cube_core_num_; c0++) {
            if (used[c0] || gm_off[c0] == OUTINDEX) {
                continue;
            }
            uint32_t g = group.group_num;
            group.group_begin[g] = list_pos;
            group.group_off[g] = gm_off[c0];
            group.group_len[g] = row_len[c0];
            uint32_t cnt = 0;
            for (uint32_t c = c0; c < cube_core_num_; c++) {
                if (gm_off[c] != gm_off[c0]) {
                    continue;
                }
                used[c] = 1;
                group.core_list[list_pos++] = c;
                cnt++;
                if (row_len[c] > group.group_len[g]) {
                    group.group_len[g] = row_len[c];
                }
            }
            group.group_size[g] = cnt;
            group.group_num++;
        }
    }

    /*
     * 每个 vector 核处理同一段 axis 下标（dq 切 s1=baseM，dkv 切 s2=baseN）。
     * 按 cube 核号顺序把 det tile AtomicAdd 到 workspace，避免同地址 UB 分组写回丢尾块。
     */
    __aicore__ inline void DeterAtomicAddOneOutput(const GlobalTensor<float> &out_ws, GM_ADDR det_base,
                                                   uint64_t tile_elems, const int64_t *gm_off, const int64_t *row_len,
                                                   uint32_t out_stride, uint32_t pingpong_idx, uint32_t full_axis)
    {
        uint32_t my_start = 0;
        uint32_t my_len = 0;
        GetVecAxisSlice(full_axis, my_start, my_len);
        if (my_len == 0) {
            return;
        }

        for (uint32_t c = 0; c < cube_core_num_; c++) {
            if (gm_off[c] == OUTINDEX) {
                continue;
            }
            if (my_start >= static_cast<uint32_t>(row_len[c])) {
                continue;
            }
            uint32_t valid_rows = my_len;
            uint32_t remain = static_cast<uint32_t>(row_len[c]) - my_start;
            if (valid_rows > remain) {
                valid_rows = remain;
            }

            GlobalTensor<float> det_c;
            uint64_t slot =
                (static_cast<uint64_t>(pingpong_idx) * cube_core_num_ + static_cast<uint64_t>(c)) * tile_elems;
            det_c.SetGlobalBuffer((__gm__ float *)(det_base + slot * sizeof(float)));
            DataCopy(det_src_ub_, det_c[static_cast<int64_t>(my_start) * head_dim_], valid_rows * head_dim_);
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);
            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);

            int64_t out_offset = gm_off[c] + static_cast<int64_t>(my_start) * out_stride;
            DataCopyExtParams copy_out;
            copy_out.blockCount = static_cast<uint16_t>(valid_rows);
            copy_out.blockLen = static_cast<uint32_t>(head_dim_ * sizeof(float));
            copy_out.srcStride = 0;
            copy_out.dstStride = static_cast<uint32_t>((out_stride - head_dim_) * sizeof(float));
            AscendC::SetAtomicAdd<float>();
            DataCopyPad(out_ws[out_offset], det_src_ub_, copy_out);
            AscendC::SetAtomicNone();
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        }
    }

    __aicore__ inline void ComputeVecPre(const GlobalTensor<float> &dst_tensor, const LocalTensor<float> &src_tensor,
                                         const EvenCoreInfo &info)
    {
        if (info.start_idx >= info.data_size) {
            return;
        }
        uint32_t process_size = info.max_process_size;
        for (uint32_t i = 0; i < info.loop_num; i++) {
            if (unlikely(i == info.loop_num - 1)) {
                process_size = info.align_tail;
                if (process_size == 0) {
                    break;
                }
            }

            int64_t gm_offset = (int64_t)info.start_idx + i * info.max_process_size;
            DataCopy(dst_tensor[gm_offset], src_tensor, process_size);
        }

        if (info.pad_tail > 0) {
            int64_t gm_offset = (int64_t)info.start_idx + (info.loop_num - 1) * info.max_process_size + info.align_tail;
            DataCopyParams copyParam;
            copyParam.blockCount = 1;
            copyParam.blockLen = info.pad_tail * sizeof(float);
            copyParam.srcStride = 0;
            copyParam.dstStride = 0;
            DataCopyPad(dst_tensor[gm_offset], src_tensor, copyParam);
        }
    }

    __aicore__ inline void ComputeSoftmaxGradFront(const GlobalTensor<float> &dst_gm,
                                                   const GlobalTensor<INPUT_TYPE> &dy_gm,
                                                   const GlobalTensor<INPUT_TYPE> &out_gm, const EvenCoreInfo &info,
                                                   const TILING_CLASS *tilingData, const uint32_t sftg_ping_pong_idx)
    {
        if (info.start_idx >= info.data_size) {
            return;
        }
        LocalTensor<INPUT_TYPE> dy_tensor = sftg_ping_pong_idx ? dy_in_ping_ : dy_in_pong_;
        LocalTensor<INPUT_TYPE> attention_tensor = sftg_ping_pong_idx ? attention_in_ping_ : attention_in_pong_;
        LocalTensor<float> dy_out_tensor = sftg_ping_pong_idx ? dy_out_ping_ : dy_out_pong_;
        LocalTensor<float> attention_out_tensor = sftg_ping_pong_idx ? attention_out_ping_ : attention_out_pong_;
        LocalTensor<float> sftg_front_tensor = sftg_ping_pong_idx ? sftg_front_ping_ : sftg_front_pong_;
        uint32_t process_size = info.max_process_size; // 表示处理S1方向的元素个数
        uint32_t src_stride;
        if constexpr (INPUT_LAYOUT == BSND) {
            src_stride = q_head_num_ * head_dim_;
        } else if constexpr (INPUT_LAYOUT == BNSD) {
            src_stride = head_dim_;
        } else if constexpr (INPUT_LAYOUT == TND) {
            src_stride = q_head_num_ * head_dim_;
        }

        for (uint32_t i = 0; i < info.loop_num; i++) {
            if (unlikely(i == info.loop_num - 1)) {
                process_size = info.align_tail + info.pad_tail;
                if (process_size == 0) {
                    break;
                }
            }
            int64_t in_gm_offset = (int64_t)(info.start_idx + i * info.max_process_size) * src_stride;

            DataCopyParams copyParam;
            copyParam.blockCount = process_size;
            copyParam.blockLen = head_dim_ * sizeof(INPUT_TYPE);
            copyParam.srcStride = (src_stride - head_dim_) * sizeof(INPUT_TYPE);
            copyParam.dstStride = 0;
            DataCopyPad(dy_tensor, dy_gm[in_gm_offset], copyParam, {false, 0, 0, 0});
            DataCopyPad(attention_tensor, out_gm[in_gm_offset], copyParam, {false, 0, 0, 0});
            SET_FLAG(MTE2, V, EVENT_ID0);
            WAIT_FLAG(MTE2, V, EVENT_ID0);

            Cast(dy_out_tensor, dy_tensor, RoundMode::CAST_NONE, process_size * head_dim_);
            Cast(attention_out_tensor, attention_tensor, RoundMode::CAST_NONE, process_size * head_dim_);
            PipeBarrier<PIPE_V>();

            uint32_t intput_shape_arry[2] = {static_cast<uint32_t>(process_size), static_cast<uint32_t>(head_dim_)};
            uint32_t out_shape_arry[2] = {static_cast<uint32_t>(process_size), static_cast<uint32_t>(8)};

            dy_out_tensor.SetShapeInfo(ShapeInfo(2, intput_shape_arry, AscendC::DataFormat::ND));
            attention_out_tensor.SetShapeInfo(ShapeInfo(2, intput_shape_arry, AscendC::DataFormat::ND));
            sftg_front_tensor.SetShapeInfo(ShapeInfo(2, out_shape_arry, AscendC::DataFormat::ND));
            bool isBasicBlock = process_size % 8 == 0;
            if (likely(isBasicBlock)) {
                SoftmaxGradFront<float, true>(sftg_front_tensor, dy_out_tensor, attention_out_tensor, sftg_tmp_tensor,
                                              tilingData->softmaxGradFrontTilingData);
            } else {
                SoftmaxGradFront<float, false>(sftg_front_tensor, dy_out_tensor, attention_out_tensor, sftg_tmp_tensor,
                                               tilingData->softmaxGradFrontTilingData);
            }
            PipeBarrier<PIPE_V>();

            SET_FLAG(V, MTE3, EVENT_ID0);
            WAIT_FLAG(V, MTE3, EVENT_ID0);
            int64_t out_gm_offset = (int64_t)(info.start_idx + i * info.max_process_size) * 8;
            DataCopy(dst_gm[out_gm_offset], sftg_front_tensor, process_size * 8);
            SET_FLAG(MTE3, MTE2, EVENT_ID0);
            WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        }
    }

    template <bool MULS>
    __aicore__ inline void ComputeVecPost(const GlobalTensor<INPUT_TYPE> &dst_tensor,
                                          const GlobalTensor<float> &src_tensor, const EvenCoreInfo &info,
                                          uint32_t &post_ping_pong_idx)
    {
        if (info.start_idx >= info.data_size) {
            return;
        }

        LocalTensor<float> vecIn;
        LocalTensor<INPUT_TYPE> vecOut;
        uint32_t process_size = info.max_process_size;

        for (uint32_t i = 0; i < info.loop_num; i++) {
            if (unlikely(i == info.loop_num - 1)) {
                process_size = info.align_tail;
                if (process_size == 0) {
                    break;
                }
            }
            vecIn = post_ping_pong_idx ? vec_in_ping_ : vec_in_pong_;
            vecOut = post_ping_pong_idx ? vec_out_ping_ : vec_out_pong_;
            event_id = post_ping_pong_idx ? event_ping_ : event_pong_;
            int64_t gm_offset = (int64_t)info.start_idx + i * info.max_process_size;

            WAIT_FLAG(MTE3, MTE2, event_id);
            DataCopy(vecIn, src_tensor[gm_offset], process_size);
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
            SET_FLAG(MTE3, MTE2, event_id);
            post_ping_pong_idx = 1 - post_ping_pong_idx;
        }

        if (info.pad_tail > 0) {
            vecIn = post_ping_pong_idx ? vec_in_ping_ : vec_in_pong_;
            vecOut = post_ping_pong_idx ? vec_out_ping_ : vec_out_pong_;
            event_id = post_ping_pong_idx ? event_ping_ : event_pong_;
            int64_t gm_offset = (int64_t)info.start_idx + (info.loop_num - 1) * info.max_process_size + info.align_tail;
            uint32_t pad_tail_align = RoundUp<uint32_t>(info.pad_tail, 16);
            DataCopyParams copyParam;
            copyParam.blockCount = 1;
            copyParam.blockLen = info.pad_tail * sizeof(float);
            copyParam.srcStride = 0;
            copyParam.dstStride = 0;

            WAIT_FLAG(MTE3, MTE2, event_id);
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
            copyParam.blockLen = info.pad_tail * sizeof(INPUT_TYPE);
            DataCopyPad(dst_tensor[gm_offset], vecOut, copyParam);
            SET_FLAG(MTE3, MTE2, event_id);
            post_ping_pong_idx = 1 - post_ping_pong_idx;
        }
    }

    __simd_vf__ inline void SimpleSoftmax(__ubuf__ float *dstTensor, __ubuf__ float *src0Tensor,
                                          __ubuf__ float *src1Tensor, const uint32_t row, const uint32_t col)
    {
        AscendC::Reg::RegTensor<float> src_reg;
        AscendC::Reg::RegTensor<float> lse_reg;
        AscendC::Reg::RegTensor<float> scale_reg;
        AscendC::Reg::MaskReg msk_reg;
        constexpr static uint16_t repeat_size = 256 / sizeof(float);
        uint16_t repeat_times = (col + repeat_size - 1) / repeat_size;
        uint32_t ub_offset = 0;
        Duplicate(scale_reg, softmax_scale_);

        for (int32_t i = 0; i < row; i++) {
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(lse_reg, src1Tensor + i * 8);
            uint32_t count = col;
            ub_offset = i * col;

            for (int32_t j = 0; j < repeat_times; j++) {
                msk_reg = AscendC::Reg::UpdateMask<float>(count);

                LoadAlign(src_reg, src0Tensor + ub_offset);
                Mul(src_reg, src_reg, scale_reg, msk_reg);
                Sub(src_reg, src_reg, lse_reg, msk_reg);
                Exp(src_reg, src_reg, msk_reg);
                StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstTensor + ub_offset, src_reg, msk_reg);
                ub_offset += repeat_size;
            }
        }
    }

    __simd_vf__ inline void ComputeSoftmaxGrad(__ubuf__ float *dstTensor, __ubuf__ float *src0Tensor,
                                               __ubuf__ float *src1Tensor, __ubuf__ float *sftFrontTensor,
                                               const uint32_t row, const uint32_t col)
    {
        AscendC::Reg::RegTensor<float> src_reg;
        AscendC::Reg::RegTensor<float> sft_front_reg;
        AscendC::Reg::RegTensor<float> mul_reg;
        AscendC::Reg::MaskReg msk_reg;
        constexpr static uint16_t repeat_size = 256 / sizeof(float);
        uint16_t repeat_times = (col + repeat_size - 1) / repeat_size;
        uint32_t ub_offset = 0;

        for (int32_t i = 0; i < row; i++) {
            LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(sft_front_reg, sftFrontTensor + i * 8);
            uint32_t count = col;
            ub_offset = i * col;

            for (int32_t j = 0; j < repeat_times; j++) {
                msk_reg = AscendC::Reg::UpdateMask<float>(count);

                LoadAlign(src_reg, src0Tensor + ub_offset);
                Sub(src_reg, src_reg, sft_front_reg, msk_reg);
                LoadAlign(mul_reg, src1Tensor + ub_offset);
                Mul(src_reg, src_reg, mul_reg, msk_reg);
                StoreAlign<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(dstTensor + ub_offset, src_reg, msk_reg);
                ub_offset += repeat_size;
            }
        }
    }

    __aicore__ inline void ComputeEvenCoreInfo(EvenCoreInfo &info, const uint32_t data_size,
                                               const uint32_t max_process_size)
    {
        /*
         * function: even process data_size
         */
        uint32_t per_core_size = CeilDiv<uint32_t>(data_size, v_core_num_);
        info.start_idx = v_core_idx_ * per_core_size;
        info.len = (info.start_idx + per_core_size > data_size) ? data_size - info.start_idx : per_core_size;
        info.max_process_size = max_process_size;
        info.data_size = data_size;
        info.loop_num = CeilDiv<uint32_t>(info.len, max_process_size);

        uint32_t tail = info.len % max_process_size;
        // 由于DataCopyPad最多处理65535，因此tail部分分成align_tail和pad_tail计算
        if (tail == 0) {
            info.align_tail = max_process_size;
            info.pad_tail = 0;
        } else {
            info.align_tail = tail / 16 * 16;
            info.pad_tail = tail - info.align_tail;
        }
    }

    __aicore__ inline void CopyInLSE(const LocalTensor<float> &dstTensor, const GlobalTensor<float> &srcTensor,
                                     const int32_t count)
    {
        /*
         * function: Copy lse from global memory to local memory
         * input shape：(b, n, s, 1) or (t, n, 1)
         * out shape:   (s, 8)
         * dtype:       float
         */
        uint32_t src_stride;
        if constexpr (INPUT_LAYOUT == BSND) {
            src_stride = 0;
        } else if (INPUT_LAYOUT == BNSD) {
            src_stride = 0;
        } else if (INPUT_LAYOUT == TND) {
            src_stride = (q_head_num_ - 1) * sizeof(float);
        }

        DataCopyPad(dstTensor, srcTensor,
                    {static_cast<uint16_t>(count), static_cast<uint32_t>(1 * sizeof(float)),
                     static_cast<uint32_t>(src_stride), 0, 0},
                    {false, 0, 0, 0});
    }
};

} // namespace BSA_ARC35
