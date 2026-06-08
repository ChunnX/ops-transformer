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
using namespace AscendC;

namespace BSA_ARC35 {

class SingleBlock {
public:
    __aicore__ inline void Update(int32_t &cur_s1_idx, int32_t &cur_s1_len)
    {
        /*
         *  Single块内只遍历S1方向。
         */
        cur_s1_idx = s1_start_idx;
        cur_s1_len = s1_start_idx + s1_base_size < s1_end_idx ? s1_base_size : (s1_end_idx - s1_start_idx);

        s1_start_idx += cur_s1_len;
        if (s1_start_idx >= s1_end_idx) {
            is_finish = true;
        }
    }

    __aicore__ inline void Reset(int32_t s1_start_idx, int32_t s1_len)
    {
        is_finish = false;
        this->s1_start_idx = s1_start_idx;
        this->s1_len = s1_len;
        s1_end_idx = s1_start_idx + s1_len;
    }

    __aicore__ inline void RecordRunTimeInfo(const RunTimeInfo &runTimeInfo)
    {
        this->runTimeInfoBak_.bIdx = runTimeInfo.bIdx;
        this->runTimeInfoBak_.last_q_seq_sum = runTimeInfo.last_q_seq_sum;
        this->runTimeInfoBak_.last_kv_seq_sum = runTimeInfo.last_kv_seq_sum;
        this->runTimeInfoBak_.cur_q_seq_len = runTimeInfo.cur_q_seq_len;
        this->runTimeInfoBak_.cur_kv_seq_len = runTimeInfo.cur_kv_seq_len;
        this->runTimeInfoBak_.s1Idx = runTimeInfo.s1Idx;
        this->runTimeInfoBak_.s2Idx = runTimeInfo.s2Idx;
        this->runTimeInfoBak_.n1Idx = runTimeInfo.n1Idx;
        this->runTimeInfoBak_.n2Idx = runTimeInfo.n2Idx;
        this->runTimeInfoBak_.s1Len = runTimeInfo.s1Len;
        this->runTimeInfoBak_.s2Len = runTimeInfo.s2Len;
        this->runTimeInfoBak_.s1LenAlign = runTimeInfo.s1LenAlign;
        this->runTimeInfoBak_.s2LenAlign = runTimeInfo.s2LenAlign;
        this->runTimeInfoBak_.kv_ping_pong_idx = runTimeInfo.kv_ping_pong_idx;
    }

    template <uint32_t INPUT_LAYOUT>
    __aicore__ inline void UpdateRunTimeInfo(RunTimeInfo &runTimeInfo, const int32_t q_head_num,
                                             const int32_t kv_head_num, const int32_t head_dim)
    {
        int32_t base_s1_idx, base_s1_len;
        this->Update(base_s1_idx, base_s1_len);

        runTimeInfo.bIdx = this->runTimeInfoBak_.bIdx;
        runTimeInfo.last_q_seq_sum = this->runTimeInfoBak_.last_q_seq_sum;
        runTimeInfo.last_kv_seq_sum = this->runTimeInfoBak_.last_kv_seq_sum;
        runTimeInfo.cur_q_seq_len = this->runTimeInfoBak_.cur_q_seq_len;
        runTimeInfo.cur_kv_seq_len = this->runTimeInfoBak_.cur_kv_seq_len;
        runTimeInfo.s1Idx = base_s1_idx;
        runTimeInfo.s2Idx = this->runTimeInfoBak_.s2Idx;
        runTimeInfo.n1Idx = this->runTimeInfoBak_.n1Idx;
        runTimeInfo.n2Idx = this->runTimeInfoBak_.n2Idx;
        runTimeInfo.s1Len = base_s1_len;
        runTimeInfo.s2Len = this->runTimeInfoBak_.s2Len;
        runTimeInfo.s1LenAlign = RoundUp(base_s1_len, 16);
        runTimeInfo.s2LenAlign = this->runTimeInfoBak_.s2LenAlign;
        runTimeInfo.queryGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num, head_dim,
                                         runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.keyGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(runTimeInfo.last_kv_seq_sum, runTimeInfo.cur_kv_seq_len, kv_head_num, head_dim,
                                         runTimeInfo.bIdx, runTimeInfo.s2Idx, runTimeInfo.n2Idx);
        runTimeInfo.lseGmOffset =
            GetLseGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num,
                                         runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.sftgGmOffset =
            GetSftgGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num,
                                          runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.need_compute = 1;
        runTimeInfo.need_copy_kv = 0;
        runTimeInfo.is_singlekv_last = is_finish;
        runTimeInfo.kv_ping_pong_idx = this->runTimeInfoBak_.kv_ping_pong_idx;
    }

    __aicore__ inline bool IsFinish()
    {
        return is_finish;
    }
    __aicore__ inline void SetBaseBlock(uint32_t base_size)
    {
        this->s1_base_size = base_size;
    }

private:
    int32_t s1_start_idx{0};
    int32_t s1_len{0};
    int32_t s1_end_idx{0};
    int32_t s1_base_size{128};
    bool is_finish{true};
    RunTimeInfo runTimeInfoBak_;
};

template <typename BSA_TYPE>
class AddrComputeModule {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    GM_ADDR actualQseqlen_;
    GM_ADDR actualKvseqlen_;
    GM_ADDR blockSparseMask_;
    int32_t batch_num_;
    int32_t q_seq_len_;
    int32_t kv_seq_len_;
    int32_t q_group_;
    int32_t q_head_num_;
    int32_t kv_head_num_;
    int32_t head_dim_;
    int32_t bIdx_{0};             // 当前batch计算到的位置
    int32_t s1Idx_{0};            // 当前s1方向计算到的位置
    int32_t s2Idx_{0};            // 当前s2方向计算到的位置
    int32_t n1Idx_{0};            // 当前n1方向计算到的位置
    int32_t cur_q_seq_len_{0};    // 当前batch的q_seq_len
    int32_t cur_kv_seq_len_{0};   // 当前batch的kv_seq_len
    int32_t last_q_seq_sum_{0};   // 上一个batch的q_seq_len的累加和
    int32_t last_kv_seq_sum_{0};  // 上一个batch的kv_seq_len的累加和
    int32_t cube_core_idx_{0};    // 实际cube核的Idx
    int32_t cube_core_num_{0};    // cube核的数量
    int32_t base_m_{0};           // 每个cube核计算的s1方向的长度
    int32_t base_n_{0};           // 每个cube核计算的s2方向的长度
    int32_t first_loop_{0};       // 当前循环次数(包括sparse跳过的计算)
    int32_t current_cube_idx_{0}; // 当前cube计算到的位置
    int32_t q_block_num_{0};
    int32_t kv_block_num_{0};
    int32_t block_x_{0};
    int32_t block_y_{0};
    int32_t single_m_{0};
    int32_t kv_ping_pong_idx_{0};
    SingleBlock single_block_;

public:
    __aicore__ inline void Init(const TILING_CLASS *tilingData, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                GM_ADDR blockSparseMask)
    {
        this->batch_num_ = tilingData->batchNum;
        this->q_seq_len_ = tilingData->qSeqLen;
        this->kv_seq_len_ = tilingData->kvSeqLen;
        this->q_group_ = tilingData->qGroup;
        this->q_head_num_ = tilingData->qHeadNum;
        this->kv_head_num_ = tilingData->kvHeadNum;
        this->head_dim_ = tilingData->headDim;
        this->cube_core_num_ = tilingData->cubeCoreNum;
        this->actualQseqlen_ = actualQseqlen;
        this->actualKvseqlen_ = actualKvseqlen;
        this->blockSparseMask_ = blockSparseMask;
        this->block_x_ = tilingData->BlockX;
        this->block_y_ = tilingData->BlockY;
        this->base_m_ = tilingData->baseM;
        this->base_n_ = tilingData->baseN;
        this->single_m_ = 1024;
        single_block_.SetBaseBlock(this->base_m_);
        if constexpr (INPUT_LAYOUT == TND) {
            UpdateSeqLen();
            int32_t max_q_seq_len_ = 0;
            int32_t max_kv_seq_len_ = 0;
            for (int32_t i = 0; i < batch_num_; i++) {
                int64_t q_seq_len = GetSeqLen(i, actualQseqlen_);
                int64_t kv_seq_len = GetSeqLen(i, actualKvseqlen_);
                max_q_seq_len_ = max(max_q_seq_len_, q_seq_len);
                max_kv_seq_len_ = max(max_kv_seq_len_, kv_seq_len);
            }
            q_block_num_ = CeilDiv(max_q_seq_len_, block_x_);
            kv_block_num_ = CeilDiv(max_kv_seq_len_, block_y_);
        } else {
            cur_q_seq_len_ = q_seq_len_;
            cur_kv_seq_len_ = kv_seq_len_;
            last_q_seq_sum_ = 0;
            last_kv_seq_sum_ = 0;
            q_block_num_ = CeilDiv(q_seq_len_, block_x_);
            kv_block_num_ = CeilDiv(kv_seq_len_, block_y_);
        }

        if ASCEND_IS_AIC {
            this->cube_core_idx_ = GetBlockIdx();
        }
        if ASCEND_IS_AIV {
            this->cube_core_idx_ = GetBlockIdx() / 2;
        }
    }

    __aicore__ inline void GetRunTimeInfo(RunTimeInfo &runTimeInfo)
    {
        runTimeInfo.need_compute = 0;

        if (!single_block_.IsFinish()) {
            // 处理完Single块，更新RunTimeInfo
            single_block_.UpdateRunTimeInfo<INPUT_LAYOUT>(runTimeInfo, q_head_num_, kv_head_num_, head_dim_);
            return;
        }

        while (true) {
            if (InitStartIdx()) {
                break;
            }

            if (IsValidBlock(q_head_num_, q_block_num_, kv_block_num_, bIdx_, n1Idx_, s1Idx_ / block_x_,
                             s2Idx_ / block_y_, blockSparseMask_)) {
                RunTimeInfoRecord(runTimeInfo);
            }

            if (current_cube_idx_ && current_cube_idx_ % cube_core_num_ == 0) {
                // 所有cube核分配到任务，暂时跳出循环，等下次分配
                current_cube_idx_ = 0;
                break;
            }
        }
    }

private:
    __aicore__ inline void UpdateSeqLen()
    {
        if constexpr (INPUT_LAYOUT != TND) {
            return;
        }
        cur_q_seq_len_ = GetSeqLen(bIdx_, actualQseqlen_);
        cur_kv_seq_len_ = GetSeqLen(bIdx_, actualKvseqlen_);
        while ((cur_q_seq_len_ == 0 || cur_kv_seq_len_ == 0) && bIdx_ < batch_num_ - 1) {
            bIdx_++;
            cur_q_seq_len_ = GetSeqLen(bIdx_, actualQseqlen_);
            cur_kv_seq_len_ = GetSeqLen(bIdx_, actualKvseqlen_);
        }
        last_q_seq_sum_ = bIdx_ > 0 ? GetSeqTotalLen(bIdx_ - 1, actualQseqlen_) : 0;
        last_kv_seq_sum_ = bIdx_ > 0 ? GetSeqTotalLen(bIdx_ - 1, actualKvseqlen_) : 0;
    }

    __aicore__ inline int32_t GetBlockLen(int32_t sIdx, int32_t sLen, int32_t single_size)
    {
        return sIdx + single_size < sLen ? single_size : (sLen - sIdx);
    }

    __aicore__ inline bool InitStartIdx()
    {
        // 遍历顺序 s2->s1->n1->batch
        int32_t recoderS1 = GetBlockLen(s1Idx_, cur_q_seq_len_, single_m_);
        int32_t recoderS2 = GetBlockLen(s2Idx_, cur_kv_seq_len_, base_n_);

        if (unlikely(first_loop_ == 0)) {
            first_loop_ = 1;
            return false;
        }

        if (s2Idx_ < cur_kv_seq_len_ - recoderS2) {
            s2Idx_ += recoderS2;
            return false;
        }

        if (s1Idx_ < cur_q_seq_len_ - recoderS1) {
            s1Idx_ += recoderS1;
            s2Idx_ = 0;
            return false;
        }

        if (n1Idx_ < q_head_num_ - 1) {
            s1Idx_ = 0;
            s2Idx_ = 0;
            n1Idx_++;
            return false;
        }

        if (bIdx_ < batch_num_ - 1) {
            s1Idx_ = 0;
            s2Idx_ = 0;
            n1Idx_ = 0;
            bIdx_++;
            if constexpr (INPUT_LAYOUT == TND) {
                UpdateSeqLen();
            }
            return false;
        }

        return true;
    }

    __aicore__ inline void RunTimeInfoRecord(RunTimeInfo &runTimeInfo)
    {
        if (current_cube_idx_ % cube_core_num_ != cube_core_idx_) {
            current_cube_idx_++;
            return;
        }
        int32_t single_s1_len = GetBlockLen(s1Idx_, cur_q_seq_len_, single_m_);
        single_block_.Reset(s1Idx_, single_s1_len);
        int32_t base_s1_idx;
        int32_t base_s1_len;
        int32_t base_s2_len = GetBlockLen(s2Idx_, cur_kv_seq_len_, base_n_);
        int32_t n2Idx = n1Idx_ / q_group_;

        single_block_.Update(base_s1_idx, base_s1_len);
        runTimeInfo.bIdx = bIdx_;
        runTimeInfo.last_q_seq_sum = last_q_seq_sum_;
        runTimeInfo.last_kv_seq_sum = last_kv_seq_sum_;
        runTimeInfo.cur_q_seq_len = cur_q_seq_len_;
        runTimeInfo.cur_kv_seq_len = cur_kv_seq_len_;
        runTimeInfo.s1Idx = base_s1_idx;
        runTimeInfo.s2Idx = s2Idx_;
        runTimeInfo.n1Idx = n1Idx_;
        runTimeInfo.n2Idx = n2Idx;
        runTimeInfo.s1Len = base_s1_len;
        runTimeInfo.s2Len = base_s2_len;
        runTimeInfo.s1LenAlign = RoundUp(base_s1_len, 16);
        runTimeInfo.s2LenAlign = RoundUp(base_s2_len, 16);
        runTimeInfo.queryGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_, head_dim_,
                                         runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.keyGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(runTimeInfo.last_kv_seq_sum, runTimeInfo.cur_kv_seq_len, kv_head_num_,
                                         head_dim_, runTimeInfo.bIdx, runTimeInfo.s2Idx, runTimeInfo.n2Idx);
        runTimeInfo.lseGmOffset =
            GetLseGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_,
                                         runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.sftgGmOffset =
            GetSftgGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_,
                                          runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.need_compute = 1;
        runTimeInfo.need_copy_kv = 1;
        runTimeInfo.kv_ping_pong_idx = kv_ping_pong_idx_;
        runTimeInfo.is_singlekv_last = single_block_.IsFinish();
        single_block_.RecordRunTimeInfo(runTimeInfo);
        kv_ping_pong_idx_ = 1 - kv_ping_pong_idx_;
        current_cube_idx_++;
    }
};
} // namespace BSA_ARC35
