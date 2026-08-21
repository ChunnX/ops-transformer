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
 * \file generic_sparse_attention_grad.h
 * \brief Arch35 Process / CubeProcess / VectorProcess for GenericSparseAttentionGrad

 *        (design §3.5 pipeline: Pre once → [Gather+MM12] || [Vector+MM345] → Post once).
 */
#pragma once
#include "common_header.h"
#include "addr_compute.h"
#include "cube_op.h"
#include "vec_op.h"
using namespace AscendC;

namespace GSAG_ARC35 {

template <typename INPUT_TYPE, uint32_t INPUT_LAYOUT, class TILING_CLASS, bool DETERMINISTIC_ENABLE>
struct GSAG_TYPE {
    using input_type = INPUT_TYPE;
    static constexpr uint32_t input_layout = INPUT_LAYOUT;
    using tiling_class = TILING_CLASS;
    static constexpr bool deterministic_enable = DETERMINISTIC_ENABLE;
};

template <typename GSAG_TYPE>
class GenericSparseAttentionGradArch35 {
    using INPUT_TYPE = typename GSAG_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = GSAG_TYPE::input_layout;
    using TILING_CLASS = typename GSAG_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = GSAG_TYPE::deterministic_enable;

private:
    RunTimeInfo runTimeInfo_[2];
    RunTimeInfo dqScatterInfo_[2];
    uint8_t dqScatterPending_[2] {0, 0};
    AddrComputeModule<GSAG_TYPE> addr_;
    GlobalTensor<INPUT_TYPE> query_gm_, key_gm_, val_gm_, dout_gm_, attention_out_gm_;
    GlobalTensor<INPUT_TYPE> dq_gm_, dk_gm_, dv_gm_;
    GlobalTensor<float> dq_workspace_, dk_workspace_, dv_workspace_, sftg_workspace_;
    TBuf<TPosition::VECCALC> ub_buffer_;
    TBuf<TPosition::A1> l1_buffer_;
    LocalTensor<float> mm1_res_ub_tensor_ping_, mm1_res_ub_tensor_pong_;
    LocalTensor<float> mm2_res_ub_tensor_ping_, mm2_res_ub_tensor_pong_;
    LocalTensor<float> mm1_res_ub_tensor_, mm2_res_ub_tensor_;
    // AIV gather staging: Q/dO are first read as ND, then converted to these
    // NZ buffers before the ordinary UB -> L1 DataCopy.
    LocalTensor<INPUT_TYPE> query_nz_ub_tensor_, dout_nz_ub_tensor_;
    LocalTensor<INPUT_TYPE> gather_q_nd_ub_, gather_dout_nd_ub_;
    LocalTensor<INPUT_TYPE> p_l1_tensor_ping_, p_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> ds_l1_tensor_ping_, ds_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> p_l1_tensor_, ds_l1_tensor_;
    // Filled by AIV0/AIV1 before AIC starts MM12.
    LocalTensor<INPUT_TYPE> query_l1_tensor_ping_, query_l1_tensor_pong_;
    LocalTensor<INPUT_TYPE> dout_l1_tensor_ping_, dout_l1_tensor_pong_;
    GlobalTensor<int32_t> sparse_idx_gm_;
    int32_t q_head_num_ { 0 };
    int32_t head_dim_ { 0 };
    uint32_t vec_ub_matrix_elements_ { 0 };
    uint32_t vec_base_m_ { 0 };
    uint32_t taskId = 0;
    uint32_t ping_pong_idx = 0;
    uint32_t last_ping_pong_idx = 0;
    uint32_t ub_offset_ = 0;
    uint32_t l1_offset_ = 0;
    static constexpr int32_t UB_SIZE = 247 * 1024;
    static constexpr int32_t L1_SIZE = 512 * 1024;

    // step: 0 Process; 1-2 Cube pre-wait FLAG_CUBE_POST; 10-14 Vec pre;
    // 20 gather entry; 203 partition; 204 zero UB; 205 GM→UB done; 206 ND2NZ done;
    // 207 before UB→L1; 208 UB→L1 done; 21 gather exit (legacy);
    // 220-224 Vec gather set-flag; 301-304 Cube gather wait;
    // 30-37 Cube main-loop (MM12/MM345 commented in Pre+Gather debug);
    // 38 Cube loop exit; 39 Cube SyncAll; 40-42 Vec loop/post.
    __aicore__ inline void GsagLog(int32_t step)
    {
        int32_t isAic = 0;
        if ASCEND_IS_AIC {
            isAic = 1;
        }
        AscendC::printf("[GSAG] aic=%d blk=%d sub=%d step=%d task=%d\n", isAic,
                        static_cast<int32_t>(GetBlockIdx()), static_cast<int32_t>(GetSubBlockIdx()), step,
                        static_cast<int32_t>(taskId));
    }

    // AIV0/AIV1 row partition: must match Softmax dualDst (mAlign/2) and CopyQAndDoutToL1.
    __aicore__ inline int32_t GatherAivRowCount(int32_t m, int32_t mAlign, int32_t aivHalfIdx) const
    {
        if (m < static_cast<int32_t>(C0_SIZE)) {
            return aivHalfIdx == 0 ? m : 0;
        }
        const int32_t halfAlign = mAlign / 2;
        const int32_t rowBegin = aivHalfIdx * halfAlign;
        int32_t rowCount = m - rowBegin;
        if (rowCount < 0) {
            rowCount = 0;
        }
        if (rowCount > halfAlign) {
            rowCount = halfAlign;
        }
        return rowCount;
    }

    __aicore__ inline bool GatherNeedAiv1(int32_t m, int32_t mAlign) const
    {
        return GatherAivRowCount(m, mAlign, 1) > 0;
    }

public:
    __aicore__ inline GenericSparseAttentionGradArch35(){};

    /**
     * Entry: wire GM, allocate ping-pong UB/L1, dispatch AIC / AIV pipelines.
     * IR order matches host: query,key,value,dout,out,lse,rsvd_idx,rsvd_count,metadata,
     * attenMask?,cuQ?,cuKv?,sequsedQ?,sequsedKv?
     */
    __aicore__ inline void Process(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dout, GM_ADDR attention_out,
                                   GM_ADDR softmaxLse, GM_ADDR rsvdBlockIdx, GM_ADDR rsvdBlockCount, GM_ADDR metadata,
                                   GM_ADDR attenMask, GM_ADDR cuSeqLengthsQ, GM_ADDR cuSeqLengthsKv,
                                   GM_ADDR sequsedQ, GM_ADDR sequsedKv, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                   GM_ADDR workspace, const TILING_CLASS *tilingData,
                                   TPipe *tPipe)
    {
        AscendC::PRINTF(
            "GenericSparseAttentionGradArch35 Process(full Pre+Gather+MM12+Vec+MM345, blockIdx=%d)\n",
            GetBlockIdx());

        GsagLog(0);
        uint32_t base_m = tilingData->baseM;
        uint32_t base_n = tilingData->baseN;
        uint32_t vec_base_m = base_m / 2;
        uint32_t vec_base_n = base_n;

        query_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)query);
        key_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)key);
        val_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
        dout_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dout);
        attention_out_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)attention_out);
        dq_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dq);
        dk_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dk);
        dv_gm_.SetGlobalBuffer((__gm__ INPUT_TYPE *)dv);
        dq_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dqWorkspaceOffset));
        dk_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dkWorkspaceOffset));
        dv_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->dvWorkspaceOffset));
        sftg_workspace_.SetGlobalBuffer((__gm__ float *)(workspace + tilingData->sftgWorkspaceOffset));
        sparse_idx_gm_.SetGlobalBuffer((__gm__ int32_t *)rsvdBlockIdx);
        q_head_num_ = static_cast<int32_t>(tilingData->qHeadNum);
        head_dim_ = static_cast<int32_t>(tilingData->headDim);
        vec_ub_matrix_elements_ = vec_base_m * vec_base_n;
        vec_base_m_ = vec_base_m;

        addr_.Init(tilingData, cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, rsvdBlockIdx, rsvdBlockCount,
                   metadata);
        tPipe->InitBuffer(ub_buffer_, UB_SIZE);
        tPipe->InitBuffer(l1_buffer_, L1_SIZE);

        //todo: 检查这里的UB分配是否超过247KB
        // mm1/mm2 ping-pong: Cube Fixpipe S/dP here; Vec Softmax/Grad reads here.
        // Gather ND/NZ always overlays the same physical region (bf16/fp16 only):
        //   gatherBytes=(8M+4)N*2 <= mmBytes=16MN; float input would not fit.
        // Timeline: Gather → Set gather flag → Cube Fixpipe mm[ping] → Softmax.
        // During Gather both mm slots are dead for Vec; ND↔NZ stay distinct views.
        static_assert(sizeof(INPUT_TYPE) == 2, "Gather/mm UB overlay requires bf16/fp16");
        const uint32_t mmSlotElems = vec_base_m * vec_base_n;
        const uint32_t mmUbBase = ub_offset_;
        mm1_res_ub_tensor_ping_ = ub_buffer_.GetWithOffset<float>(mmSlotElems, ub_offset_);
        ub_offset_ += mmSlotElems * sizeof(float);
        mm1_res_ub_tensor_pong_ = ub_buffer_.GetWithOffset<float>(mmSlotElems, ub_offset_);
        ub_offset_ += mmSlotElems * sizeof(float);
        mm2_res_ub_tensor_ping_ = ub_buffer_.GetWithOffset<float>(mmSlotElems, ub_offset_);
        ub_offset_ += mmSlotElems * sizeof(float);
        mm2_res_ub_tensor_pong_ = ub_buffer_.GetWithOffset<float>(mmSlotElems, ub_offset_);
        ub_offset_ += mmSlotElems * sizeof(float);
        // +1 row per C0 column for TransdataND2NZ deconflict pad.
        const uint32_t nz_ub_elements = (vec_base_m + 1) * vec_base_n;
        uint32_t gOff = mmUbBase;
        gather_q_nd_ub_ = ub_buffer_.GetWithOffset<INPUT_TYPE>(vec_ub_matrix_elements_, gOff);
        gOff += vec_ub_matrix_elements_ * sizeof(INPUT_TYPE);
        gather_dout_nd_ub_ = ub_buffer_.GetWithOffset<INPUT_TYPE>(vec_ub_matrix_elements_, gOff);
        gOff += vec_ub_matrix_elements_ * sizeof(INPUT_TYPE);
        query_nz_ub_tensor_ = ub_buffer_.GetWithOffset<INPUT_TYPE>(nz_ub_elements, gOff);
        gOff += nz_ub_elements * sizeof(INPUT_TYPE);
        dout_nz_ub_tensor_ = ub_buffer_.GetWithOffset<INPUT_TYPE>(nz_ub_elements, gOff);
        (void)gOff;

        p_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        p_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        ds_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);
        ds_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * base_n, l1_offset_);
        l1_offset_ += base_m * base_n * sizeof(INPUT_TYPE);

        const uint32_t head_dim_align = RoundUp(static_cast<uint32_t>(head_dim_), C0_SIZE);
        query_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align, l1_offset_);
        l1_offset_ += base_m * head_dim_align * sizeof(INPUT_TYPE);
        query_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align, l1_offset_);
        l1_offset_ += base_m * head_dim_align * sizeof(INPUT_TYPE);
        dout_l1_tensor_ping_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align, l1_offset_);
        l1_offset_ += base_m * head_dim_align * sizeof(INPUT_TYPE);
        dout_l1_tensor_pong_ = l1_buffer_.GetWithOffset<INPUT_TYPE>(base_m * head_dim_align, l1_offset_);
        l1_offset_ += base_m * head_dim_align * sizeof(INPUT_TYPE);

        (void)attenMask;

        if ASCEND_IS_AIC {
            CubeProcess(query, key, value, dout, attention_out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata,
                        cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, dq, dk, dv, workspace, tilingData, tPipe);
        }
        if ASCEND_IS_AIV {
            VectorProcess(query, key, value, dout, attention_out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata,
                          cuSeqLengthsQ, cuSeqLengthsKv, sequsedQ, sequsedKv, dq, dk, dv, workspace, tilingData, tPipe);
        }
    }

    __aicore__ inline int64_t GetSparseTokenGmOffset(const RunTimeInfo &runTimeInfo, const int32_t qTok) const
    {
        if constexpr (INPUT_LAYOUT == TND) {
            return (static_cast<int64_t>(runTimeInfo.last_q_seq_sum) + qTok) * q_head_num_ * head_dim_ +
                   static_cast<int64_t>(runTimeInfo.n1Idx) * head_dim_;
        } else if constexpr (INPUT_LAYOUT == BSND) {
            return static_cast<int64_t>(runTimeInfo.bIdx) * runTimeInfo.cur_q_seq_len * q_head_num_ * head_dim_ +
                   static_cast<int64_t>(qTok) * q_head_num_ * head_dim_ +
                   static_cast<int64_t>(runTimeInfo.n1Idx) * head_dim_;
        } else {
            return static_cast<int64_t>(runTimeInfo.bIdx) * q_head_num_ * runTimeInfo.cur_q_seq_len * head_dim_ +
                   static_cast<int64_t>(runTimeInfo.n1Idx) * runTimeInfo.cur_q_seq_len * head_dim_ +
                   static_cast<int64_t>(qTok) * head_dim_;
        }
    }

    /**
     * AIV0/AIV1 each own s1LenAlign/2 rows (16-row fractal). ND gather lands in
     * UB, TransdataND2NZ writes [K1, vec_base_m+1, C0], then CopyUB2L1-style
     * DataCopy scatters into Cube NZ L1 [K1, s1LenAlign, C0].
     */
    __aicore__ inline void CopyQAndDoutToL1(const RunTimeInfo &runTimeInfo, const uint32_t pingPongIdx)
    {
        GsagLog(20);
        const int32_t m = static_cast<int32_t>(runTimeInfo.s1Len);
        const int32_t mAlign = static_cast<int32_t>(runTimeInfo.s1LenAlign);
        if (m <= 0 || mAlign <= 0) {
            GsagLog(21);
            return;
        }

        // MIX_AIC_1_2: AIV subblock 0/1. Partition MUST match Cube Fixpipe dualDstCtl=1
        // and Vec Softmax (half = mAlign/2). Align16(m/2) diverges when m%32!=0 (e.g. m=68).
        const int32_t aivHalfIdx = static_cast<int32_t>(GetSubBlockIdx());
        const int32_t halfAlign = mAlign / 2;
        const int32_t totalAlign = mAlign;
        int32_t rowBegin = 0;
        int32_t shardAlign = halfAlign;
        int32_t rowCount = 0;
        if (m < static_cast<int32_t>(C0_SIZE)) {
            // Only AIV0 writes the full mAlign tile (zero-padded) for Cube LoadData.
            if (aivHalfIdx != 0) {
                GsagLog(21);
                return;
            }
            rowBegin = 0;
            shardAlign = mAlign;
            rowCount = m;
        } else {
            rowBegin = aivHalfIdx * halfAlign;
            rowCount = m - rowBegin;
            if (rowCount < 0) {
                rowCount = 0;
            }
            if (rowCount > halfAlign) {
                rowCount = halfAlign;
            }
            shardAlign = halfAlign;
        }

        if (rowCount <= 0) {
            GsagLog(21);
            return;
        }
        GsagLog(203);

        // Dedicated events — avoid EVENT_ID0 (Vector) and EVENT_ID3/4 (vec/cube ping-pong).
        constexpr event_t evtGatherVmte2 = EVENT_ID2;
        constexpr event_t evtGatherMte2v = EVENT_ID5;
        constexpr event_t evtGatherVmte3 = EVENT_ID6;
        constexpr event_t evtGatherMte3mte2 = EVENT_ID7;

        // Gather ND staging aliases mm1/mm2 physical UB (see Process alloc). Softmax does not
        // touch these views until after gather Set; Cube Fixpipe mm[ping] starts only after Wait gather.
        LocalTensor<INPUT_TYPE> qUb = gather_q_nd_ub_;
        LocalTensor<INPUT_TYPE> doutUb = gather_dout_nd_ub_;

        const int32_t ndElems = static_cast<int32_t>(vec_ub_matrix_elements_);
        Duplicate(qUb, static_cast<INPUT_TYPE>(0), ndElems);
        Duplicate(doutUb, static_cast<INPUT_TYPE>(0), ndElems);
        SET_FLAG(V, MTE2, evtGatherVmte2);
        WAIT_FLAG(V, MTE2, evtGatherVmte2);
        GsagLog(204);

        for (int32_t row = 0; row < rowCount; ++row) {
            const int32_t qTok = sparse_idx_gm_.GetValue(runTimeInfo.sparseIdxOffset + rowBegin + row);
            const int64_t gmOffset = GetSparseTokenGmOffset(runTimeInfo, qTok);
            DataCopy(qUb[row * head_dim_], query_gm_[gmOffset], head_dim_);
            DataCopy(doutUb[row * head_dim_], dout_gm_[gmOffset], head_dim_);
        }
        SET_FLAG(MTE2, V, evtGatherMte2v);
        WAIT_FLAG(MTE2, V, evtGatherMte2v);
        GsagLog(205);
        GsagLog(201);

        // VF always converts the full vec_base_m ND tile (FAG CopyUB2L1).
        const uint32_t vfM = vec_base_m_;
        TransdataND2NZ<INPUT_TYPE>(query_nz_ub_tensor_, qUb, vfM, head_dim_);
        TransdataND2NZ<INPUT_TYPE>(dout_nz_ub_tensor_, doutUb, vfM, head_dim_);
        SET_FLAG(V, MTE3, evtGatherVmte3);
        WAIT_FLAG(V, MTE3, evtGatherVmte3);
        GsagLog(206);
        GsagLog(202);

        // Copy a full 16-row-aligned shard into L1 NZ. Rows beyond rowCount
        // were zero-filled in UB, so writing the full aligned shard keeps the
        // L1 fractal layout regular for Cube consumers.
        const int32_t copyRows = shardAlign;
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(head_dim_ / C0_SIZE);
        copyParams.blockLen = static_cast<uint16_t>(
            copyRows * static_cast<int32_t>(C0_SIZE) * static_cast<int32_t>(sizeof(INPUT_TYPE)) / 32);
        copyParams.srcStride = static_cast<uint16_t>(
            (static_cast<int32_t>(vfM) + 1 - copyRows) * static_cast<int32_t>(C0_SIZE) *
            static_cast<int32_t>(sizeof(INPUT_TYPE)) / 32);
        copyParams.dstStride = static_cast<uint16_t>(
            (totalAlign - copyRows) * static_cast<int32_t>(C0_SIZE) * static_cast<int32_t>(sizeof(INPUT_TYPE)) / 32);

        LocalTensor<INPUT_TYPE> qL1 =
            pingPongIdx == 0 ? query_l1_tensor_ping_ : query_l1_tensor_pong_;
        LocalTensor<INPUT_TYPE> doutL1 =
            pingPongIdx == 0 ? dout_l1_tensor_ping_ : dout_l1_tensor_pong_;
        const int32_t l1RowOffset = rowBegin * static_cast<int32_t>(C0_SIZE);
        AscendC::printf(
            "[GSAG-G] blk=%d sub=%d task=%d ping=%d m=%d ma=%d ha=%d ta=%d rb=%d rc=%d cr=%d hd=%d vfM=%d bc=%u bl=%u ss=%u ds=%u l1off=%d\n",
            static_cast<int32_t>(GetBlockIdx()), static_cast<int32_t>(GetSubBlockIdx()),
            static_cast<int32_t>(taskId), static_cast<int32_t>(pingPongIdx), m, mAlign, halfAlign,
            totalAlign, rowBegin, rowCount, copyRows, head_dim_, static_cast<int32_t>(vfM),
            static_cast<uint32_t>(copyParams.blockCount),
            static_cast<uint32_t>(copyParams.blockLen), static_cast<uint32_t>(copyParams.srcStride),
            static_cast<uint32_t>(copyParams.dstStride), l1RowOffset);
        GsagLog(207);
        GsagLog(2071);
        DataCopy(qL1[l1RowOffset], query_nz_ub_tensor_, copyParams);
        GsagLog(2072);
        GsagLog(2073);
        DataCopy(doutL1[l1RowOffset], dout_nz_ub_tensor_, copyParams);
        GsagLog(2074);
        SET_FLAG(MTE3, MTE2, evtGatherMte3mte2);
        WAIT_FLAG(MTE3, MTE2, evtGatherMte3mte2);
        GsagLog(208);
        GsagLog(21);
    }

    __aicore__ inline void CubeProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dout, GM_ADDR attention_out,
                                       GM_ADDR softmaxLse, GM_ADDR rsvdBlockIdx, GM_ADDR rsvdBlockCount,
                                       GM_ADDR metadata, GM_ADDR cuSeqLengthsQ, GM_ADDR cuSeqLengthsKv,
                                       GM_ADDR sequsedQ, GM_ADDR sequsedKv, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                       GM_ADDR workspace, const TILING_CLASS *tilingData,
                                       TPipe *tPipe)
    {
        CubeOp<GSAG_TYPE> cubeOp;
        cubeOp.Init(tilingData, tPipe, l1_buffer_, l1_offset_, query_l1_tensor_ping_, query_l1_tensor_pong_,
                    dout_l1_tensor_ping_, dout_l1_tensor_pong_, rsvdBlockIdx, workspace);
        GsagLog(1);
        CrossCoreWaitFlag(FLAG_CUBE_POST);
        GsagLog(2);

        (void)query;
        (void)key;
        (void)value;
        (void)dout;
        (void)attention_out;
        (void)softmaxLse;
        (void)rsvdBlockCount;
        (void)metadata;
        (void)cuSeqLengthsQ;
        (void)cuSeqLengthsKv;
        (void)sequsedQ;
        (void)sequsedKv;
        (void)dq;
        (void)dk;
        (void)dv;
        (void)workspace;

        // Pipeline: Gather → MM12 || Vector Softmax/Grad → MM345 → Post (align BSAG).
        while (true) {
            ping_pong_idx = taskId % 2;
            last_ping_pong_idx = 1 - ping_pong_idx;
            mm1_res_ub_tensor_ = ping_pong_idx ? mm1_res_ub_tensor_ping_ : mm1_res_ub_tensor_pong_;
            mm2_res_ub_tensor_ = ping_pong_idx ? mm2_res_ub_tensor_ping_ : mm2_res_ub_tensor_pong_;
            p_l1_tensor_ = last_ping_pong_idx ? p_l1_tensor_ping_ : p_l1_tensor_pong_;
            ds_l1_tensor_ = last_ping_pong_idx ? ds_l1_tensor_ping_ : ds_l1_tensor_pong_;
            addr_.GetRunTimeInfo(runTimeInfo_[ping_pong_idx]);

            if (runTimeInfo_[ping_pong_idx].need_compute) {
                if (runTimeInfo_[ping_pong_idx].use_sparse_gather) {
                    const uint32_t gatherFlag =
                        ping_pong_idx == 0 ? FLAG_V_GATHER_C_PING : FLAG_V_GATHER_C_PONG;
                    const int32_t m = static_cast<int32_t>(runTimeInfo_[ping_pong_idx].s1Len);
                    const int32_t mAlign = static_cast<int32_t>(runTimeInfo_[ping_pong_idx].s1LenAlign);
                    // Release Vec (taskId>0) then Wait gather with nothing in between.
                    // Vec SoftmaxGrad of previous task blocks on MM345_DONE until here,
                    // so Cube Wait is always armed before Vec Set gather.
                    if (taskId > 0) {
                        CrossCoreSetFlag<2, PIPE_FIX>(FLAG_C_MM345_DONE);
                    }
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(gatherFlag);
                    GsagLog(302);
                    if (GatherNeedAiv1(m, mAlign)) {
                        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(gatherFlag + AIV0_AIV1_FLAG_OFFSET);
                    }
                    GsagLog(304);
                }
                GsagLog(31);
                cubeOp.SendMatmulQK(query_gm_, key_gm_, mm1_res_ub_tensor_, runTimeInfo_[ping_pong_idx], ping_pong_idx);
                GsagLog(315);
                CrossCoreSetFlag<2, PIPE_FIX>(FLAG_C1_V1);
                GsagLog(32);
                cubeOp.SendMatmulDyV(dout_gm_, val_gm_, mm2_res_ub_tensor_, runTimeInfo_[ping_pong_idx], ping_pong_idx);
                GsagLog(325);
                CrossCoreSetFlag<2, PIPE_FIX>(FLAG_C2_V2);
                GsagLog(33);
            }

            if (taskId > 0 && runTimeInfo_[last_ping_pong_idx].need_compute) {
                GsagLog(34);
                CrossCoreWaitFlag<2, PIPE_MTE1>(FLAG_V1_C3);
                GsagLog(35);
                cubeOp.SendMatmulDv(p_l1_tensor_, dv_workspace_, runTimeInfo_[last_ping_pong_idx], last_ping_pong_idx);
                GsagLog(36);
                CrossCoreWaitFlag<2, PIPE_MTE1>(FLAG_V2_C45);
                GsagLog(37);
                cubeOp.SendMatmulDq(ds_l1_tensor_, dq_workspace_, runTimeInfo_[last_ping_pong_idx], last_ping_pong_idx);
                cubeOp.SendMatmulDk(ds_l1_tensor_, dk_workspace_, runTimeInfo_[last_ping_pong_idx], last_ping_pong_idx);
                SET_FLAG(MTE1, MTE2, EVENT_ID0);
                WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
            }

            if (runTimeInfo_[ping_pong_idx].need_compute == false) {
                GsagLog(38);
                break;
            }
            taskId++;
        }
        cubeOp.Destroy();
        GsagLog(39);
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_CUBE_POST);
    }

    __aicore__ inline void VectorProcess(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dout,
                                         GM_ADDR attention_out, GM_ADDR softmaxLse, GM_ADDR rsvdBlockIdx,
                                         GM_ADDR rsvdBlockCount, GM_ADDR metadata, GM_ADDR cuSeqLengthsQ,
                                         GM_ADDR cuSeqLengthsKv, GM_ADDR sequsedQ, GM_ADDR sequsedKv,
                                         GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace,
                                         const TILING_CLASS *tilingData, TPipe *tPipe)
    {
        VecOp<GSAG_TYPE> vecOp;
        vecOp.Init(dout, query, key, value, attention_out, softmaxLse, rsvdBlockIdx, rsvdBlockCount, metadata,
                   cuSeqLengthsQ, cuSeqLengthsKv, dq, dk, dv, workspace, tilingData, ub_buffer_, ub_offset_);

        (void)tPipe;
        // GsagLog(10);

        vecOp.SendVecPre(dq_workspace_, dk_workspace_, dv_workspace_, dout_gm_, attention_out_gm_, sftg_workspace_,
                         tilingData, ub_buffer_);
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        // GsagLog(11);
        vecOp.SendVecSftgFront(dout_gm_, attention_out_gm_, sftg_workspace_, tilingData, ub_buffer_);
        PipeBarrier<PIPE_ALL>();
        // GsagLog(12);
        SyncAll();
        // GsagLog(13);
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_CUBE_POST);
        // GsagLog(14);

        vecOp.SetFlag();
        while (true) {
            ping_pong_idx = taskId % 2;
            mm1_res_ub_tensor_ = ping_pong_idx ? mm1_res_ub_tensor_ping_ : mm1_res_ub_tensor_pong_;
            mm2_res_ub_tensor_ = ping_pong_idx ? mm2_res_ub_tensor_ping_ : mm2_res_ub_tensor_pong_;
            p_l1_tensor_ = ping_pong_idx ? p_l1_tensor_ping_ : p_l1_tensor_pong_;
            ds_l1_tensor_ = ping_pong_idx ? ds_l1_tensor_ping_ : ds_l1_tensor_pong_;
            addr_.GetRunTimeInfo(runTimeInfo_[ping_pong_idx]);

            if (runTimeInfo_[ping_pong_idx].need_compute) {
                if (runTimeInfo_[ping_pong_idx].use_sparse_gather) {
                    const int32_t m = static_cast<int32_t>(runTimeInfo_[ping_pong_idx].s1Len);
                    const int32_t mAlign = static_cast<int32_t>(runTimeInfo_[ping_pong_idx].s1LenAlign);
                    const int32_t aivHalfIdx = static_cast<int32_t>(GetSubBlockIdx());
                    // task0: Cube is already in Wait gather before Vec Softmax of any prior task.
                    // task>0: wait until Cube finished prior MM345 so Wait gather is armed.
                    if (taskId > 0) {
                        GsagLog(220);
                        CrossCoreWaitFlag<2, PIPE_MTE3>(FLAG_C_MM345_DONE);
                        GsagLog(221);
                        // After Wait: Cube completed MM345 for the prior tile that lived in
                        // this ping slot (schedule: MM345 at Cube task T-1 uses last=T%2).
                        // taskId==1 Wait is vacuous (no prior MM345); scatter from taskId>=2.
                        if (taskId >= 2 && dqScatterPending_[ping_pong_idx]) {
                            vecOp.ScatterDqSel(dqScatterInfo_[ping_pong_idx], ub_buffer_, ping_pong_idx);
                            dqScatterPending_[ping_pong_idx] = 0;
                        }
                    }
                    CopyQAndDoutToL1(runTimeInfo_[ping_pong_idx], ping_pong_idx);
                    if (GatherAivRowCount(m, mAlign, aivHalfIdx) > 0) {
                        const uint32_t gatherFlagBase =
                            ping_pong_idx == 0 ? FLAG_V_GATHER_C_PING : FLAG_V_GATHER_C_PONG;
                        const uint32_t gatherFlag =
                            gatherFlagBase + static_cast<uint32_t>(aivHalfIdx) * AIV0_AIV1_FLAG_OFFSET;
                        GsagLog(223);
                        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(gatherFlag);
                        GsagLog(24);
                    }
                }
                vecOp.SendVecSftPreProcess(runTimeInfo_[ping_pong_idx], ping_pong_idx);
                GsagLog(25);
                CrossCoreWaitFlag<2, PIPE_V>(FLAG_C1_V1);
                GsagLog(26);
                vecOp.SendVecSoftmax(p_l1_tensor_, mm1_res_ub_tensor_, runTimeInfo_[ping_pong_idx]);
                GsagLog(262);
                CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_V1_C3);
                GsagLog(27);
                CrossCoreWaitFlag<2, PIPE_V>(FLAG_C2_V2);
                GsagLog(29);
                vecOp.SendVecSoftmaxGrad(ds_l1_tensor_, mm1_res_ub_tensor_, mm2_res_ub_tensor_,
                                         runTimeInfo_[ping_pong_idx]);
                CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_V2_C45);
                // Snapshot for deferred dQ scatter (Cube MM4 Fixpipes dQ_sel later).
                if (runTimeInfo_[ping_pong_idx].use_sparse_gather) {
                    dqScatterInfo_[ping_pong_idx] = runTimeInfo_[ping_pong_idx];
                    dqScatterPending_[ping_pong_idx] = 1;
                }
            }

            if (runTimeInfo_[ping_pong_idx].need_compute == false) {
                GsagLog(40);
                break;
            }
            taskId++;
        }
        vecOp.WaitFlag();
        GsagLog(41);
        PipeBarrier<PIPE_ALL>();
        CrossCoreWaitFlag(FLAG_CUBE_POST);
        SyncAll();
        // Final MM345 completed with CUBE_POST; drain any pending dQ_sel scatters.
        for (uint32_t s = 0; s < 2; ++s) {
            if (dqScatterPending_[s]) {
                vecOp.ScatterDqSel(dqScatterInfo_[s], ub_buffer_, s);
                dqScatterPending_[s] = 0;
            }
        }
        GsagLog(42);
        vecOp.SendVecPost(dq_gm_, dk_gm_, dv_gm_, dq_workspace_, dk_workspace_, dv_workspace_, tilingData, ub_buffer_);

    }
};

} // namespace GSAG_ARC35
