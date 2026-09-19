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

// 两条路径共用同一套 RunTimeInfo 填充与 K/V 复用判定，仅“取哪一个 block”的遍历方式不同：
//   - 确定性路径：直读 blockSparseMask 判块，按 latin-square 轮转调度取 (w, x, y)；
//   - 非确定性路径：按 host 侧 NonZero 预 kernel 生成的 block index workspace 顺序取块。
template <typename BSA_TYPE>
class AddrComputeModule {
protected:
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    static constexpr uint32_t INPUT_LAYOUT = BSA_TYPE::input_layout;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

    GM_ADDR actualQseqlen_;
    GM_ADDR actualKvseqlen_;
    GM_ADDR blockSparseMask_;   // 确定性路径直读的 block 稀疏 mask
    GM_ADDR indexWorkspace_;    // 非确定性路径：有效块下标表
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

    // 确定性路径调度状态：把全部 (batch, q_head) × (s1 块) × (s2 块) 任务按 latin-square
    // 轮转分给 cube 核——第 r 轮第 j 个核处理 (w, x, y)，同一轮内各核的 x、y 两两不同，
    // 因此各核并发写 dq/dk/dv 命中的地址互不重叠；再配合每轮一次跨核 barrier，
    // fp32 原子加的先后顺序被固定成“轮次序” => bitwise 可复现。
    int32_t s1_outer_{0};     // m = ceil(maxQSeqLen / baseM)
    int32_t s2_outer_{0};     // n = ceil(maxKvSeqLen / baseN)
    uint32_t deter_latin_r_{0};
    uint32_t deter_max_round_{0};
    // K/V 复用分组状态：同一 latin 分组内连续 m 轮的 y（KV 块）不变，分组未关闭时 K/V 常驻
    // L1、dk/dv 在 L0C 内继续累加；关闭时换一块 L1 槽位并重置 L0C 累加器。
    // 稀疏 mask 下的空轮不参与、也不打断分组，因此用核内自持状态判断，而不是比较相邻任务 offset。
    uint32_t dkv_group_open_{0};
    int32_t kv_ping_pong_idx_{0};
    // latin 分组的尾部有效性缓存：同一分组（ceil(r/m) 相同）内 w、y 固定，x 随轮次每步 +1（模 m），
    // 因此进入分组时按轮转顺序扫一遍 mask 的 q_block 行、记下“最后一个有效 x 的轮转位置”，
    // 之后每轮只做一次整数比较——替代原来每轮 O(m) 的 mask 扫描（每次迭代都带整数除模 + 逐字节 GM 标量读）。
    // mask 已由 aclnn 适配层转成 [B, N1, KV_BLOCK, Q_BLOCK] 连续布局，
    // 固定 (bIdx, n1Idx, kv_block) 时 q_block 方向是连续字节，天然适合批量读。
    uint32_t grp_g_{0};              // 缓存对应的分组序号 ceil(r/m)，0 表示未缓存
    uint32_t grp_last_valid_pos_{0}; // 分组内最后一个有效 x 的轮转位置；0 表示本轮之后无有效任务
    ConstInfo const_info_;

    __gm__ int32_t *index_ptr_{nullptr};

public:
    __aicore__ inline void Init(const TILING_CLASS *tilingData, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                GM_ADDR workspace, GM_ADDR blockSparseMask)
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
        s1_outer_ = CeilDiv(max_q_seq_len_, base_m_);
        s2_outer_ = CeilDiv(max_kv_seq_len_, base_n_);
        if constexpr (DETERMINISTIC_ENABLE) {
            deter_max_round_ = CalcDeterMaxRound();
        }
        const_info_.q_head_num = q_head_num_;
        const_info_.kv_head_num = kv_head_num_;
        const_info_.block_x = block_x_;
        const_info_.block_y = block_y_;
        const_info_.head_dim = head_dim_;
        const_info_.q_block_num = q_block_num_;
        const_info_.kv_block_num = kv_block_num_;
        this->index_ptr_ = (__gm__ int32_t *)indexWorkspace_;
        if ASCEND_IS_AIC {
            this->cube_core_idx_ = GetBlockIdx();
        }
        if ASCEND_IS_AIV {
            this->cube_core_idx_ = GetBlockIdx() / 2;
        }
    }

    __aicore__ inline uint32_t GetDeterMaxRound() const
    {
        return deter_max_round_;
    }

    __aicore__ inline void GetRunTimeInfo(RunTimeInfo &runTimeInfo)
    {
        if constexpr (DETERMINISTIC_ENABLE) {
            GetRunTimeInfoDeter(runTimeInfo);
            return;
        }
        GetRunTimeInfoByIndex(runTimeInfo);
    }

protected:
    // ------------------------------------------------------------------
    // 非确定性路径：NonZero 预 kernel 产出的有效块下标表
    // ------------------------------------------------------------------

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

    __aicore__ inline void GetRunTimeInfoByIndex(RunTimeInfo &runTimeInfo)
    {
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

    // ------------------------------------------------------------------
    // 确定性路径：latin-square 无冲突轮转
    // ------------------------------------------------------------------

    // 总轮数只与形状有关、与 mask 稀疏度无关（稀疏 mask 下空轮直接产出 need_compute = 0）
    __aicore__ inline uint32_t CalcDeterMaxRound()
    {
        int64_t m = s1_outer_;
        int64_t n = s2_outer_;
        int64_t k = cube_core_num_;
        if (m <= 0 || n <= 0 || k <= 0) {
            return 0;
        }
        if (q_group_ <= 1) {
            int64_t b = static_cast<int64_t>(batch_num_) * q_head_num_;
            k = IMin(k, b * m);
            return static_cast<uint32_t>(m * ICeil(b * n, k));
        }
        int64_t b = static_cast<int64_t>(batch_num_) * kv_head_num_;
        int64_t g = q_group_;
        k = IMin(IMin(k, b * g * m), b * n);
        int64_t R = IMax(IMax(ICeil(b * n * g, k), ICeil(n, m)), g);
        return static_cast<uint32_t>(R * m);
    }

    __aicore__ inline bool MapDenseIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r, int64_t &w,
                                         int64_t &x, int64_t &y)
    {
        k = IMin(k, b * m);
        if (j > k || j < 1 || r < 1) {
            return false;
        }
        int64_t p = (ICeil(r, m) - 1) * k + j;
        w = p % b;
        w = (w != 0) ? w : b;
        y = ICeil(p, b);
        int64_t y1 = y % m;
        y1 = (y1 != 0) ? y1 : m;
        int64_t r1 = r % m;
        r1 = (r1 != 0) ? r1 : m;
        x = y1 + r1 - 1;
        if (x > m) {
            x -= m;
        }
        return (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n);
    }

    __aicore__ inline bool MapGqaIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t core_id, int64_t round_id,
                                       int64_t g, int64_t &bn1, int64_t &x, int64_t &y)
    {
        k = IMin(IMin(k, b * g * m), b * n);
        int64_t R = IMax(IMax(ICeil(b * n * g, k), ICeil(n, m)), g);
        if (core_id < 1 || core_id > k || round_id < 1 || round_id > R * m) {
            return false;
        }
        int64_t ID = (core_id - 1) * R + ICeil(round_id, m);
        int64_t local_id = round_id % m;
        local_id = local_id != 0 ? local_id : m;
        if (ID > g * n * b) {
            return false;
        }
        int64_t N = b * g;
        int64_t b_id = ID % N;
        b_id = b_id != 0 ? b_id : N;
        b_id = ICeil(b_id, g);
        y = ICeil(ID, N);
        int64_t w = ID % g;
        w = w != 0 ? w : g;
        int64_t gcd = IGcd(N, R);
        int64_t t1 = R / gcd;
        int64_t t2 = N / gcd;
        int64_t t1_new = t1 * m;
        int64_t y1 = y % t1_new;
        y1 = y1 != 0 ? y1 : t1_new;
        int64_t offset = ICeil(y1, t1);
        if (t1_new < n) {
            int64_t n1 = (n % t1_new);
            n1 = n1 != 0 ? n1 : t1_new;
            if (y <= n - n1) {
                int64_t delta = ICeil(y, t1_new);
                ID += delta;
                if (ID > (delta - 1) * t2 * m * R + offset * t2 * R) {
                    ID -= t2 * R;
                }
                b_id = ID % N;
                b_id = b_id != 0 ? b_id : N;
                b_id = ICeil(b_id, g);
                w = ID % g;
                w = w != 0 ? w : g;
                y = ICeil(ID, N);
            }
        }
        x = local_id + offset - 1;
        if (x > m) {
            x -= m;
        }
        bn1 = w + (b_id - 1) * g;
        return (bn1 >= 1 && x >= 1 && x <= m && y >= 1 && y <= n);
    }

    // 进入一个 latin 分组时建立「尾部是否还有有效块」的缓存。
    //
    // 同一分组内（ceil(r/m) 相同、core j 固定）w、y 不变，x 随轮次每步 +1（模 m），
    // 因此只需扫一遍本分组剩余的轮转位置，记下最后一个有效 x 的位置；之后每轮的
    // is_kv_end 退化成一次整数比较。相比原来每轮 MostValidInGroup 的 O(m) 扫描：
    //   - 去掉了每轮迭代里的 MapDenseIndex/MapGqaIndex（含 IGcd、多次整数除模）
    //   - mask 读取沿途连续推进（q_block 方向连续），不再逐字节算 offset 读 GM
    // 单次分组扫描量 = 本分组剩余轮数，摊到每轮约 1 次迭代，且都是寄存器算术。
    __aicore__ inline void BuildGroupTailCache(int64_t G, int64_t m, uint32_t pos, int64_t x, int32_t s1Idx,
                                               int32_t s2Idx, int32_t bIdx, int32_t n1Idx, int32_t cur_q)
    {
        grp_g_ = static_cast<uint32_t>(G);
        grp_last_valid_pos_ = 0;
        int64_t pos_max = m;
        if (G * m > static_cast<int64_t>(deter_max_round_)) {
            pos_max = static_cast<int64_t>(deter_max_round_) - (G - 1) * m;
        }
        if (pos_max <= static_cast<int64_t>(pos)) {
            return;
        }
        // x 的上界：s1Idx = (x-1)*base_m_ 必须 < cur_q
        const int64_t x_max = ICeil(static_cast<int64_t>(cur_q), static_cast<int64_t>(base_m_));
        // 固定 (bIdx, n1Idx, kv_block) 后 mask 行的基址（q_block = 0 处的字节偏移）
        const int64_t qbn = const_info_.q_block_num;
        const int64_t kbn = const_info_.kv_block_num;
        const int64_t kv_block_idx = static_cast<int64_t>(s2Idx) / block_y_;
        __gm__ uint8_t *mask_row = blockSparseMask_ + static_cast<int64_t>(bIdx) * (const_info_.q_head_num * qbn * kbn) +
                                   static_cast<int64_t>(n1Idx) * (qbn * kbn) + kv_block_idx * qbn;
        // 用 qb + frac 表示 acc = (x-1)*base_m_ = qb*block_x_ + frac，推进时只做加减
        // （base_m_ <= block_x_，故每次最多进一位），避免每次迭代做除法
        int64_t xq = x;
        int32_t qb = s1Idx / block_x_;
        int32_t frac = s1Idx % block_x_;
        for (int64_t pp = static_cast<int64_t>(pos) + 1; pp <= pos_max; ++pp) {
            ++xq;
            if (xq > m) {
                xq = 1;
                qb = 0;
                frac = 0;
            } else {
                frac += base_m_;
                if (frac >= block_x_) {
                    frac -= block_x_;
                    ++qb;
                }
            }
            if (xq > x_max) {
                continue;
            }
            if (mask_row[qb] != 0) {
                grp_last_valid_pos_ = static_cast<uint32_t>(pp);
            }
        }
    }

    __aicore__ inline void GetRunTimeInfoDeter(RunTimeInfo &runTimeInfo)
    {
        runTimeInfo.need_compute = 0;
        deter_latin_r_++;
        int64_t m = s1_outer_;
        int64_t n = s2_outer_;
        int64_t k = cube_core_num_;
        int64_t j = cube_core_idx_ + 1;
        int64_t r = static_cast<int64_t>(deter_latin_r_);
        if (deter_latin_r_ > deter_max_round_ || m <= 0 || n <= 0 || k <= 0) {
            return;
        }
        int64_t w = 0;
        int64_t x = 0;
        int64_t y = 0;
        bool ok = false;
        int64_t bflat = q_group_ <= 1 ? static_cast<int64_t>(batch_num_) * q_head_num_ :
                                        static_cast<int64_t>(batch_num_) * kv_head_num_;
        if (q_group_ <= 1) {
            ok = MapDenseIndex(k, m, n, bflat, j, r, w, x, y);
        } else {
            ok = MapGqaIndex(k, m, n, bflat, j, r, q_group_, w, x, y);
        }
        if (!ok) {
            return;
        }
        int64_t bn1 = w - 1;
        int32_t bIdx = static_cast<int32_t>(bn1 / q_head_num_);
        int32_t n1Idx = static_cast<int32_t>(bn1 % q_head_num_);
        int32_t n2Idx = n1Idx / q_group_;
        int32_t s1Idx = static_cast<int32_t>((x - 1) * base_m_);
        int32_t s2Idx = static_cast<int32_t>((y - 1) * base_n_);
        int32_t cur_q = q_seq_len_;
        int32_t cur_kv = kv_seq_len_;
        int32_t last_q = 0;
        int32_t last_kv = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            if (bIdx < 0 || bIdx >= batch_num_) {
                return;
            }
            cur_q = GetSeqLen(bIdx, actualQseqlen_);
            cur_kv = GetSeqLen(bIdx, actualKvseqlen_);
            last_q = bIdx > 0 ? GetSeqTotalLen(bIdx - 1, actualQseqlen_) : 0;
            last_kv = bIdx > 0 ? GetSeqTotalLen(bIdx - 1, actualKvseqlen_) : 0;
        } else if (bIdx < 0 || bIdx >= batch_num_) {
            return;
        }
        if (s1Idx >= cur_q || s2Idx >= cur_kv || s1Idx < 0 || s2Idx < 0) {
            return;
        }
        if (!IsValidBlock(const_info_, bIdx, n1Idx, s1Idx / block_x_, s2Idx / block_y_, blockSparseMask_)) {
            return;
        }
        int32_t s1Len = GetBlockLen(s1Idx, cur_q, base_m_);
        int32_t s2Len = GetBlockLen(s2Idx, cur_kv, base_n_);
        runTimeInfo.bIdx = bIdx;
        runTimeInfo.last_q_seq_sum = last_q;
        runTimeInfo.last_kv_seq_sum = last_kv;
        runTimeInfo.cur_q_seq_len = cur_q;
        runTimeInfo.cur_kv_seq_len = cur_kv;
        runTimeInfo.s1Idx = s1Idx;
        runTimeInfo.s2Idx = s2Idx;
        runTimeInfo.n1Idx = n1Idx;
        runTimeInfo.n2Idx = n2Idx;
        runTimeInfo.s1Len = s1Len;
        runTimeInfo.s2Len = s2Len;
        runTimeInfo.s1LenAlign = RoundUp(s1Len, 16);
        runTimeInfo.s2LenAlign = RoundUp(s2Len, 16);
        runTimeInfo.queryGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(last_q, cur_q, q_head_num_, head_dim_, bIdx, s1Idx, n1Idx);
        runTimeInfo.keyGmOffset =
            GetQKVGmOffset<INPUT_LAYOUT>(last_kv, cur_kv, kv_head_num_, head_dim_, bIdx, s2Idx, n2Idx);
        runTimeInfo.lseGmOffset = GetLseGmOffset<INPUT_LAYOUT>(last_q, cur_q, q_head_num_, bIdx, s1Idx, n1Idx);
        runTimeInfo.sftgGmOffset = GetSftgGmOffset<INPUT_LAYOUT>(last_q, cur_q, q_head_num_, bIdx, s1Idx, n1Idx);
        runTimeInfo.need_compute = 1;
        const int64_t g_idx = ICeil(r, m);                                   // latin 分组序号
        const uint32_t pos = static_cast<uint32_t>(r - (g_idx - 1) * m);     // 分组内轮转位置 1..m
        if (!dkv_group_open_) {
            // 开启新分组：换一块 K/V L1 槽位，dk/dv 的 L0C 累加器重置（cmatrixInitVal = 1）
            kv_ping_pong_idx_ = (kv_ping_pong_idx_ + 1) % 3;
            runTimeInfo.is_kv_change = 1;
            dkv_group_open_ = 1;
            // 本分组的第一轮有效任务：一次性把“本组后面还有没有有效块”算好
            BuildGroupTailCache(g_idx, m, pos, x, s1Idx, s2Idx, bIdx, n1Idx, cur_q);
        } else {
            runTimeInfo.is_kv_change = 0;
        }
        runTimeInfo.kv_ping_pong_idx = kv_ping_pong_idx_;
        // 本组内最后一轮有效任务 => 收尾（fixpipe 落盘 dk/dv）
        const bool last = (static_cast<uint32_t>(g_idx) == grp_g_) && (pos >= grp_last_valid_pos_);
        runTimeInfo.is_kv_end = last ? 1 : 0;
        if (last) {
            dkv_group_open_ = 0;
        }
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
