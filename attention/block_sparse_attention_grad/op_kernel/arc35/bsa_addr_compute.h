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
#include "bsa_common_header.h"
using namespace AscendC;

namespace BSA_ARC35 {

template <typename BSA_TYPE>
class AddrComputeModule {
protected:
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;
    // 是否走 host 预生成的 block index workspace 遍历（NonZero 优化路径）
    static constexpr bool INDEX_ENABLE = BSA_TYPE::index_enable;
    static_assert(INDEX_ENABLE,
                  "BSAG arch35: 确定性与非确定性路径统一使用 NonZero index 遍历，不再读 blockSparseMask 判块");
    GM_ADDR actualQseqlen_;
    GM_ADDR actualKvseqlen_;
    GM_ADDR indexWorkspace_;
    GM_ADDR indexShapeWorkspace_;
    int32_t batch_num_;
    int32_t q_seq_len_;
    int32_t kv_seq_len_;
    int32_t q_group_;
    int32_t q_head_num_;
    int32_t kv_head_num_;
    int32_t head_dim_;
    int32_t bIdx_{0};            // 当前batch计算到的位置
    int32_t s1Idx_{0};           // 当前s1方向计算到的位置
    int32_t s2Idx_{0};           // 当前s2方向计算到的位置
    int32_t n1Idx_{0};           // 当前n1方向计算到的位置
    int32_t cur_q_seq_len_{0};   // 当前batch的q_seq_len
    int32_t cur_kv_seq_len_{0};  // 当前batch的kv_seq_len
    int32_t last_q_seq_sum_{0};  // 上一个batch的q_seq_len的累加和
    int32_t last_kv_seq_sum_{0}; // 上一个batch的kv_seq_len的累加和
    int32_t cube_core_idx_{0};   // 实际cube核的Idx
    int32_t cube_core_num_{0};   // cube核的数量
    int32_t base_m_{0};          // 每个cube核计算的s1方向的长度
    int32_t base_n_{0};          // 每个cube核计算的s2方向的长度
    int32_t q_block_num_{0};
    int32_t kv_block_num_{0};
    int32_t block_x_{0};
    int32_t block_y_{0};
    int32_t max_q_seq_len_{0};
    int32_t max_kv_seq_len_{0};
    uint64_t loop_num_{0};
    int32_t block_global_idx{0};
    int32_t block_local_idx{0};
    uint64_t vaild_block_num{0};
    // 确定性路径核间分工（与 host 侧 GetWorkspaceSize 的 dqSplitNum 计算一致）：
    //   组 g = batch * kvHeadNum + n2；单位 u = g * k + p，p ∈ [0, k) 为该组 kv_block 的子段号，
    //   子段 p 覆盖 kv_block 区间 [p*KV/k, (p+1)*KV/k)。cube 核 c 负责所有 u % cubeCoreNum == c 的单位。
    //   同一 dq 槽位 (b, n1, s1) 的各个 kv_block 恰好落在不同子段 p 上、每段只有一个拥有者，
    //   因此每段 buffer 单写者、累加顺序 = 程序顺序，vector 端再按段号固定升序归约 => bitwise 可复现。
    //   组内 s2 只有一段（k == 1）时所有核写到同一份 dq buffer 的不重叠槽位，无需分段归约。
    int32_t dq_split_num_{1}; // k：每组的 kv_block 子段数，同时也是 dq 分段数（非确定性为 1）
    int32_t group_num_{1};    // G = batchNum * kvHeadNum
    int64_t dq_size_{0};      // dq 元素总量
    // s1 相位遍历：把 q_block 空间均分为 s1_phase_num_ 段，外层按相位推进，使同一核重复访问同一段
    // dq buffer 的距离从整份 dq 缩到 1/s1PhaseNum，避免 k 份 dq 分段整体超出 L2 后原子加回写 HBM。
    int32_t s1_phase_num_{1};
    int32_t cur_phase_{0};    // 当前相位
    int32_t it_unit_{0};      // 当前遍历到的单位编号（步长 = cube_core_num_）
    int32_t it_n1_{0};        // 单位内当前 (b,q_head) 的偏移（0..q_group-1）
    int32_t it_s2_{0};        // 单位内子段 s2 偏移
    int32_t it_pos_{0};       // 当前 entry 子区间已消费到的下标
    int32_t it_end_{0};       // 当前 entry 子区间的上界
    int64_t it_dq_offset_{0}; // 当前单位对应的 dq 段 workspace 偏移（子段号 * dqSize）
    __gm__ int32_t *index_ptr_{nullptr};

public:
    __aicore__ inline void Init(const TILING_CLASS *tilingData, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                GM_ADDR workspace)
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
        this->indexWorkspace_ = workspace + tilingData->indexWorkspaceOffset;
        this->indexShapeWorkspace_ = workspace + tilingData->indexShapeWorkspaceOffset;
        this->block_x_ = tilingData->BlockX;
        this->block_y_ = tilingData->BlockY;
        this->base_m_ = tilingData->baseM;
        this->base_n_ = tilingData->baseN;
        if constexpr (INPUT_LAYOUT == TND) {
            UpdateSeqLenByBatch(bIdx_);
            max_q_seq_len_ = 0;
            max_kv_seq_len_ = 0;
            for (int32_t i = 0; i < batch_num_; i++) {
                int64_t q_seq_len = GetSeqLen(i, actualQseqlen_);
                int64_t kv_seq_len = GetSeqLen(i, actualKvseqlen_);
                max_q_seq_len_ = IMax(max_q_seq_len_, q_seq_len);
                max_kv_seq_len_ = IMax(max_kv_seq_len_, kv_seq_len);
            }
            q_block_num_ = CeilDiv(max_q_seq_len_, block_x_);
            kv_block_num_ = CeilDiv(max_kv_seq_len_, block_y_);
        } else {
            cur_q_seq_len_ = q_seq_len_;
            cur_kv_seq_len_ = kv_seq_len_;
            last_q_seq_sum_ = 0;
            last_kv_seq_sum_ = 0;
            max_q_seq_len_ = q_seq_len_;
            max_kv_seq_len_ = kv_seq_len_;
            q_block_num_ = CeilDiv(q_seq_len_, block_x_);
            kv_block_num_ = CeilDiv(kv_seq_len_, block_y_);
        }
        // 确定性路径：dq 子段数 k / s1 相位数由 tiling 下发（非确定性恒为 1）
        this->dq_size_ = static_cast<int64_t>(tilingData->dqSize);
        this->dq_split_num_ = (tilingData->dqSplitNum > 0) ? static_cast<int32_t>(tilingData->dqSplitNum) : 1;
        this->s1_phase_num_ = (tilingData->s1PhaseNum > 0) ? static_cast<int32_t>(tilingData->s1PhaseNum) : 1;
        this->group_num_ = batch_num_ * IMax(kv_head_num_, 1);
        if (this->group_num_ < 1) {
            this->group_num_ = 1;
        }
        this->index_ptr_ = (__gm__ int32_t *)indexWorkspace_;
        if ASCEND_IS_AIC {
            this->cube_core_idx_ = GetBlockIdx();
        }
        if ASCEND_IS_AIV {
            this->cube_core_idx_ = GetBlockIdx() / 2;
        }
    }

    // 将 block index 展平下标还原为 (batch, q_head, q_block, kv_block)。
    // kernel 侧 blockSparseMask 布局为 [B, N1, KV_BLOCK, Q_BLOCK]（aclnn 适配层已把公开的
    // [B, N1, Q_BLOCK, KV_BLOCK] 转置），因此最内层维度是 Q_BLOCK。
    __aicore__ inline void unravel_index(int32_t flat_index, int32_t &b, int32_t &n, int32_t &s1_block,
                                         int32_t &s2_block)
    {
        s1_block = flat_index % q_block_num_;
        int32_t remaining = flat_index / q_block_num_;
        s2_block = remaining % kv_block_num_;
        remaining = remaining / kv_block_num_;
        n = remaining % q_head_num_;
        b = remaining / q_head_num_;
    }

    // TND 下每个 batch 的真实序列长度不同，切换 batch 时需要刷新当前长度与前缀和
    __aicore__ inline void UpdateSeqLenByBatch(int32_t b_idx)
    {
        if constexpr (INPUT_LAYOUT != TND) {
            return;
        }
        cur_q_seq_len_ = GetSeqLen(b_idx, actualQseqlen_);
        cur_kv_seq_len_ = GetSeqLen(b_idx, actualKvseqlen_);
        last_q_seq_sum_ = b_idx > 0 ? GetSeqTotalLen(b_idx - 1, actualQseqlen_) : 0;
        last_kv_seq_sum_ = b_idx > 0 ? GetSeqTotalLen(b_idx - 1, actualKvseqlen_) : 0;
    }

    __aicore__ inline void GetRunTimeInfo(RunTimeInfo &runTimeInfo)
    {
        // 确定性与非确定性路径统一按 host 预生成的 block index 遍历，不再读 blockSparseMask 判块
        GetRunTimeInfoByIndex(runTimeInfo);
    }

protected:
    // 每个核连续取 INDEX_BLOCK_NUM 个有效块，让相邻任务尽量落在同一 KV 块上，提高 K/V 复用率
    static constexpr int32_t INDEX_BLOCK_NUM = 16;

    // 取第 flat_index 个有效块填充 runTimeInfo；返回 false 表示该块已超出真实序列长度、需跳过
    __aicore__ inline bool FillRunTimeInfoByIndex(int32_t flat_index, RunTimeInfo &runTimeInfo)
    {
        int32_t b_idx;
        int32_t n1_idx;
        int32_t s1_block;
        int32_t s2_block;
        unravel_index(flat_index, b_idx, n1_idx, s1_block, s2_block);
        if (b_idx != bIdx_) {
            bIdx_ = b_idx;
            UpdateSeqLenByBatch(b_idx);
        }
        n1Idx_ = n1_idx;
        s1Idx_ = s1_block * base_m_;
        s2Idx_ = s2_block * base_n_;
        if (s1Idx_ >= cur_q_seq_len_ || s2Idx_ >= cur_kv_seq_len_) {
            // mask 末尾的补齐块（TND 短序列、block 未铺满）不产生任务
            return false;
        }
        RunTimeInfoRecord(runTimeInfo, s1Idx_, s2Idx_, GetBlockLen(s1Idx_, cur_q_seq_len_, base_m_),
                          GetBlockLen(s2Idx_, cur_kv_seq_len_, base_n_));
        return true;
    }

    // NonZero 优化路径：按 host 预生成的 block index workspace 顺序取任务
    __aicore__ inline void GetRunTimeInfoByIndex(RunTimeInfo &runTimeInfo)
    {
        if constexpr (DETERMINISTIC_ENABLE) {
            GetRunTimeInfoByUnit(runTimeInfo);
            return;
        }
        runTimeInfo.need_compute = 0;
        if (unlikely(loop_num_ == 0)) {
            this->vaild_block_num = ((__gm__ uint64_t *)indexShapeWorkspace_)[2];
            block_global_idx = cube_core_idx_;
            block_local_idx = 0;
            loop_num_ = 1;
        }
        while (true) {
            if (block_local_idx >= INDEX_BLOCK_NUM) {
                block_global_idx += cube_core_num_;
                block_local_idx = 0;
            }
            int64_t index_workspace_offset = static_cast<int64_t>(block_global_idx) * INDEX_BLOCK_NUM + block_local_idx;
            if (index_workspace_offset >= static_cast<int64_t>(vaild_block_num)) {
                return; // 有效块 index 已耗尽
            }
            block_local_idx++;
            loop_num_++;
            int32_t block_sparse_idx = index_ptr_[index_workspace_offset];
            if (FillRunTimeInfoByIndex(block_sparse_idx, runTimeInfo)) {
                return;
            }
        }
    }

    // 相位对应的 s1 block 下界 / 上界：把 [0, q_block_num_) 均分成 s1_phase_num_ 段。
    __aicore__ inline int32_t PhaseLoBlock(int32_t phase) const
    {
        return static_cast<int32_t>((static_cast<int64_t>(phase) * q_block_num_) / s1_phase_num_);
    }

    __aicore__ inline int32_t PhaseHiBlock(int32_t phase) const
    {
        return static_cast<int32_t>((static_cast<int64_t>(phase + 1) * q_block_num_) / s1_phase_num_);
    }

    // 二分：第一个满足 (entry / q_block_num_) >= group_key 的 entry 下标（group_key 单调不减）
    __aicore__ inline int32_t LowerBoundGroup(int32_t group_key) const
    {
        int32_t lo = 0;
        int32_t hi = static_cast<int32_t>(vaild_block_num);
        while (lo < hi) {
            int32_t mid = lo + ((hi - lo) >> 1);
            if (index_ptr_[mid] / q_block_num_ < group_key) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    // 二分：同 group 内第一个满足 (entry % q_block_num_) >= target 的 entry 下标（s1 单调不减）
    __aicore__ inline int32_t LowerBoundS1(int32_t lo, int32_t hi, int32_t target) const
    {
        while (lo < hi) {
            int32_t mid = lo + ((hi - lo) >> 1);
            if (index_ptr_[mid] % q_block_num_ < target) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    // 确定性路径：按 (相位 -> 本核负责的单位 -> q_head -> s2 子段) 顺序，用二分定位本核任务区间，
    // 只访问属于本核的 entry（不再线性扫全表），同时让同一段 dq buffer 的重复访问距离缩小到 1/相位数。
    __aicore__ inline void GetRunTimeInfoByUnit(RunTimeInfo &runTimeInfo)
    {
        runTimeInfo.need_compute = 0;
        if (unlikely(loop_num_ == 0)) {
            this->vaild_block_num = ((__gm__ uint64_t *)indexShapeWorkspace_)[2];
            // 相位不是越多越好：dk/dv 每个相位都要落盘一次，稀疏 mask 下 (s2, 相位) 落盘次数几乎不随
            // 稀疏度下降，相位过多反而更慢。这里按运行期有效块数收敛相位数（稠密保持 tiling 值）。
            if (s1_phase_num_ > 1 && kv_block_num_ > 0) {
                int32_t phase_cap = static_cast<int32_t>(vaild_block_num / (static_cast<uint64_t>(kv_block_num_) * 4));
                if (phase_cap < 1) {
                    phase_cap = 1;
                }
                if (phase_cap < s1_phase_num_) {
                    s1_phase_num_ = phase_cap;
                }
            }
            loop_num_ = 1;
            cur_phase_ = 0;
            it_unit_ = cube_core_idx_;
            it_n1_ = 0;
            it_s2_ = 0;
            it_pos_ = 0;
            it_end_ = 0;
        }
        while (true) {
            if (it_pos_ >= it_end_) {
                if (!AdvanceUnitGroup()) {
                    return; // 本核任务已全部取完
                }
            }
            int32_t block_sparse_idx = index_ptr_[it_pos_++];
            runTimeInfo.dqWsOffset = it_dq_offset_;
            runTimeInfo.s1Phase = cur_phase_;
            if (FillRunTimeInfoByIndex(block_sparse_idx, runTimeInfo)) {
                return;
            }
        }
    }

    // 找到下一个存在有效块的 (相位, 单位, q_head, s2) entry 子区间 [it_pos_, it_end_)；返回 false 表示结束
    __aicore__ inline bool AdvanceUnitGroup()
    {
        const int32_t vaild = static_cast<int32_t>(vaild_block_num);
        const int32_t k = dq_split_num_;
        const int32_t total_units = group_num_ * k;
        while (cur_phase_ < s1_phase_num_) {
            const int32_t ph_lo = PhaseLoBlock(cur_phase_);
            const int32_t ph_hi = PhaseHiBlock(cur_phase_);
            while (it_unit_ < total_units) {
                const int32_t unit = it_unit_;
                const int32_t sub = unit % k;   // kv_block 子段号 = dq 段号
                const int32_t group = unit / k; // (batch, kv_head)
                const int32_t n2 = group % kv_head_num_;
                const int32_t b_idx = group / kv_head_num_;
                const int32_t s2_lo = static_cast<int32_t>((static_cast<int64_t>(sub) * kv_block_num_) / k);
                const int32_t s2_hi = static_cast<int32_t>((static_cast<int64_t>(sub + 1) * kv_block_num_) / k);
                const int32_t n1_base = n2 * q_group_;
                const int32_t n1_num = IMin(q_group_, q_head_num_ - n1_base);
                while (it_n1_ < n1_num) {
                    const int32_t n1 = n1_base + it_n1_;
                    while (it_s2_ < s2_hi - s2_lo) {
                        const int32_t s2_block = s2_lo + it_s2_;
                        it_s2_++;
                        if (s2_block >= kv_block_num_ || ph_lo >= ph_hi) {
                            continue;
                        }
                        const int32_t group_key = (b_idx * q_head_num_ + n1) * kv_block_num_ + s2_block;
                        const int32_t g_lo = LowerBoundGroup(group_key);
                        if (g_lo >= vaild || index_ptr_[g_lo] / q_block_num_ != group_key) {
                            continue; // 稀疏 mask 下该 (b, q_head, kv_block) 无有效块
                        }
                        const int32_t g_hi = LowerBoundGroup(group_key + 1);
                        const int32_t lo = LowerBoundS1(g_lo, g_hi, ph_lo);
                        const int32_t hi = (ph_hi >= q_block_num_) ? g_hi : LowerBoundS1(lo, g_hi, ph_hi);
                        if (lo < hi) {
                            it_pos_ = lo;
                            it_end_ = hi;
                            it_dq_offset_ = static_cast<int64_t>(sub) * dq_size_;
                            return true;
                        }
                    }
                    it_s2_ = 0;
                    it_n1_++;
                }
                it_n1_ = 0;
                it_unit_ += cube_core_num_;
            }
            cur_phase_++;
            it_unit_ = cube_core_idx_;
            it_n1_ = 0;
            it_s2_ = 0;
        }
        it_pos_ = 0;
        it_end_ = 0;
        return false;
    }

private:
    __aicore__ inline void RunTimeInfoRecord(RunTimeInfo &runTimeInfo, int32_t vaild_s1_idx, int32_t vaild_s2_idx,
                                             int32_t vaild_s1_len, int32_t vaild_s2_len)
    {
        int32_t n2Idx = n1Idx_ / q_group_;
        runTimeInfo.bIdx = bIdx_;
        runTimeInfo.last_q_seq_sum = last_q_seq_sum_;
        runTimeInfo.cur_q_seq_len = cur_q_seq_len_;
        runTimeInfo.s1Idx = vaild_s1_idx;
        runTimeInfo.s2Idx = vaild_s2_idx;
        runTimeInfo.n1Idx = n1Idx_;
        runTimeInfo.n2Idx = n2Idx;
        runTimeInfo.s1Len = vaild_s1_len;
        runTimeInfo.s2Len = vaild_s2_len;
        runTimeInfo.s1LenAlign = RoundUp(vaild_s1_len, 16);
        runTimeInfo.s2LenAlign = RoundUp(vaild_s2_len, 16);
        runTimeInfo.queryGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_, head_dim_,
                                         runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.keyGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(last_kv_seq_sum_, cur_kv_seq_len_, kv_head_num_, head_dim_, runTimeInfo.bIdx,
                                         runTimeInfo.s2Idx, runTimeInfo.n2Idx);
        runTimeInfo.lseGmOffset =
            GetLseGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_,
                                         runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.sftgGmOffset =
            GetSftgGmOffset<INPUT_LAYOUT>(runTimeInfo.last_q_seq_sum, runTimeInfo.cur_q_seq_len, q_head_num_,
                                          runTimeInfo.bIdx, runTimeInfo.s1Idx, runTimeInfo.n1Idx);
        runTimeInfo.need_compute = 1;
    }
};

} // namespace BSA_ARC35
