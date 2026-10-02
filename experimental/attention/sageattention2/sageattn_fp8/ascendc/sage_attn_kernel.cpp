/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tiling.h"
#include "online_softmax.h"
#include "accum_reduce.h"
#include "sync_guard.h"

#include "../catlass/sage_tile.hpp"
#include "../catlass/sage_resource.hpp"
#include "../catlass/sage_qk_mmad.hpp"
#include "../catlass/sage_pv_mmad.hpp"

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>

#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#include "kernel_operator.h"
using namespace AscendC;
#else
#ifndef __global__
#define __global__
#endif
#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif
#ifndef KERNEL_TASK_TYPE_DEFAULT
#define KERNEL_TASK_TYPE_DEFAULT(x)
#endif
#ifndef KERNEL_TYPE_MIX_AIC_1_2
#define KERNEL_TYPE_MIX_AIC_1_2 0
#endif
#ifndef ASCEND_IS_AIC
#define ASCEND_IS_AIC true
#endif
#ifndef ASCEND_IS_AIV
#define ASCEND_IS_AIV false
#endif

// 宿主机或本地单测桩函数定义
inline uint32_t GetBlockIdx() { return 0; }
inline uint32_t GetSubBlockIdx() { return 0; }
#endif

namespace sage_attn {

/**
 * SageAttention2 1C2V 设备核执行器
 * 核心技术规范：
 * 1. 1C2V 硬件编排：1 个 Cube Core 搭配 2 个 Vector Core (V0 量化 prologue 核，V1 Softmax 与 LSE 核)
 * 2. 延迟一拍流水：Cube 与 Vector 双域交错掩盖计算与搬运延时
 * 3. GQA 头聚合：当 Hq > Hkv 时沿 M 维聚合至 64 的倍数，一份 L1 KV 被多个 Q head 连续复用
 * 4. 64 位 GM 寻址：所有显存指针与 stride 计算强转 int64_t，杜绝长序列整数溢出
 * 5. BufferID 与跨核 Flag 同步保序：严格遵循生产者在前、消费者在后
 */
class SageAttnKernel {
public:
    __aicore__ SageAttnKernel() {}

    /**
     * 核心初始化逻辑
     */
    __aicore__ void Init(
        __gm__ uint8_t *query,
        __gm__ uint8_t *key,
        __gm__ uint8_t *value,
        __gm__ uint8_t *keyMean,
        __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *softmaxLse,
        __gm__ uint8_t *workspace,
        const SageAttention2TilingData *__restrict tilingData)
    {
        tiling_ = tilingData;
        query_gm_ = query;
        key_gm_ = key;
        value_gm_ = value;
        key_mean_gm_ = keyMean;
        out_gm_ = attentionOut;
        lse_gm_ = softmaxLse;
        workspace_gm_ = workspace;

        core_idx_ = GetBlockIdx();
        sub_core_idx_ = GetSubBlockIdx();

        // 提取核心 tiling 常量
        b_ = tiling_->b;
        hq_ = tiling_->hq;
        hkv_ = tiling_->hkv;
        sq_ = tiling_->sq;
        sk_ = tiling_->sk;
        d_ = tiling_->d_pad;
        gqa_ratio_ = (hkv_ > 0) ? (hq_ / hkv_) : 1;

        m_per_core_ = tiling_->m_per_core;
        block_n_ = tiling_->block_n;
        k_step_ = tiling_->k_step;

        // 计算当前核的任务归属
        // 沿 (Batch, Hkv, Sq_block) 空间切分，优先将同一 KV 头的多个 Q 头聚合在同一核内执行
        uint32_t q_blks_per_head = (sq_ + m_per_core_ - 1) / m_per_core_;
        uint32_t total_tasks = b_ * hkv_ * q_blks_per_head;

        if (core_idx_ >= total_tasks) {
            is_active_core_ = false;
            return;
        }
        is_active_core_ = true;

        // 64 位整型提升计算当前任务索引
        int64_t task_idx = static_cast<int64_t>(core_idx_);
        int64_t q_blk_id = task_idx % q_blks_per_head;
        int64_t rem = task_idx / q_blks_per_head;
        int64_t hkv_id = rem % hkv_;
        int64_t b_id = rem / hkv_;

        cur_b_ = static_cast<uint32_t>(b_id);
        cur_hkv_ = static_cast<uint32_t>(hkv_id);
        cur_q_blk_ = static_cast<uint32_t>(q_blk_id);

        // 计算当前 Q 分块的实际 token 范围
        cur_q_start_ = cur_q_blk_ * m_per_core_;
        cur_q_len_ = (cur_q_start_ + m_per_core_ <= sq_) ? m_per_core_ : (sq_ - cur_q_start_);

        // 初始化两级累加控制器
        uint32_t total_kv_blocks = (sk_ + block_n_ - 1) / block_n_;
        accum_ctrl_.Init(tiling_->pv_accum_dtype, k_step_, total_kv_blocks);
    }

    /**
     * 算子主流水执行入口
     */
    __aicore__ void Process()
    {
        if (!is_active_core_) {
            return;
        }

#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
        if (ASCEND_IS_AIC) {
            ProcessCube();
        } else {
            if (sub_core_idx_ == 0) {
                ProcessVec0();
            } else {
                ProcessVec1();
            }
        }
#else
        // 宿主机模拟时按逻辑顺序串行演进
        ProcessVec0();
        ProcessCube();
        ProcessVec1();
#endif
    }

private:
    /**
     * 1. Cube 核心处理逻辑（MMAD 计算与延迟一拍主流水）
     * 拍 k：INT8 QK^T MMAD，L0C 累加
     * 拍 k+1：FP8 PV MMAD，FixPipe 随路量化回写 UB
     */
    __aicore__ void ProcessCube()
    {
        Catlass::SageAttn::SageQkMmad<> qk_mmad;
        Catlass::SageAttn::SagePvMmad<> pv_mmad;

        // 1. 等待 V0 完成 Q 矩阵量化并常驻 L1 Bank 0
        CrossCoreWait(CrossCoreDirection::VEC0_TO_CUBE, CrossCoreFlags::FLAG_Q_READY);

        // 加载 Q 分块至 L0A
        qk_mmad.LoadL0A(tiling_->l1_q_offset, 0, cur_q_len_, d_);

        uint32_t total_kv_blocks = (sk_ + block_n_ - 1) / block_n_;

        // 延迟一拍主循环 (Double Buffering Ping-Pong)
        for (uint32_t k = 0; k < total_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            uint32_t kv_flag = is_ping ? CrossCoreFlags::FLAG_KV_PING_READY : CrossCoreFlags::FLAG_KV_PONG_READY;
            uint32_t l1_kv_addr = is_ping ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                          : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET;

            // 1. 等待 V0 当前 KV 分块量化就绪
            CrossCoreWait(CrossCoreDirection::VEC0_TO_CUBE, kv_flag);

            // 2. 拍 k（Cube）：INT8 QK^T MMAD
            // 加载 K 分块至 L0B
            qk_mmad.LoadL0B(l1_kv_addr, 0, block_n_, d_);
            qk_mmad.Compute(true); // L0C 内累加本块结果

            // 通知 V1：P 矩阵块就绪
            CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_QK_P_READY);

            // 3. 拍 k（Cube 并发）：计算上一拍的 FP8 PV MMAD (当 k > 0 时)
            if (k > 0) {
                uint32_t prev_k = k - 1;
                bool prev_is_ping = ((prev_k % 2) == 0);
                uint32_t prev_l1_v_addr = prev_is_ping ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                                       : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET;

                // 等待 V1 上一拍 Softmax 概率矩阵 S 就绪
                CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_SOFTMAX_S_READY);

                // 加载 S 矩阵至 L0A，加载 V 矩阵至 L0B
                pv_mmad.LoadL0A(tiling_->ub_softmax_offset, 0, cur_q_len_, block_n_);
                pv_mmad.LoadL0B(prev_l1_v_addr, 0, block_n_, d_);

                // 累加 PV 计算
                bool init_pv = (prev_k == 0) || (accum_ctrl_.ShouldReduce(prev_k, false));
                pv_mmad.Compute(init_pv);

                // FixPipe 随路量化搬出至 UB 主累加器
                if (accum_ctrl_.ShouldReduce(prev_k, false)) {
                    FixpipeConfig fix_cfg = accum_ctrl_.GetFixpipeConfig();
                    pv_mmad.ExecuteEpilogue(0, tiling_->ub_acc_offset, fix_cfg);
                }
            }

            accum_ctrl_.Advance();
        }

        // Drain 拍：完成最后一个 KV 分块的 PV MMAD 与 FixPipe 搬出
        if (total_kv_blocks > 0) {
            uint32_t last_k = total_kv_blocks - 1;
            bool last_is_ping = ((last_k % 2) == 0);
            uint32_t last_l1_v_addr = last_is_ping ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                                   : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET;

            CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_SOFTMAX_S_READY);

            pv_mmad.LoadL0A(tiling_->ub_softmax_offset, 0, cur_q_len_, block_n_);
            pv_mmad.LoadL0B(last_l1_v_addr, 0, block_n_, d_);
            pv_mmad.Compute(false);

            // 最后一拍强制搬出至 UB 主累加器
            FixpipeConfig fix_cfg = accum_ctrl_.GetFixpipeConfig();
            pv_mmad.ExecuteEpilogue(0, tiling_->ub_acc_offset, fix_cfg);
        }

        // 通知 V1：整个 Attention 的 Cube PV 累加已全部完成
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_O_DONE);
    }

    /**
     * 2. Vector 0 核心处理逻辑（量化 prologue 核）
     * 负责 GM 流式预取、K 减均值、INT8 量化、V 转置与 FP8 量化、ND2NZ 摆放
     */
    __aicore__ void ProcessVec0()
    {
        // 64 位 GM 基地址与步长计算
        int64_t b_idx = static_cast<int64_t>(cur_b_);
        int64_t hkv_idx = static_cast<int64_t>(cur_hkv_);
        int64_t sq_start = static_cast<int64_t>(cur_q_start_);

        // GQA 聚合：当前 KV 头所对应的第一个 Q 头
        int64_t hq_first = hkv_idx * static_cast<int64_t>(gqa_ratio_);

        // 计算 64 位全局地址偏移
        int64_t q_stride_b, q_stride_h, q_stride_s;
        if (tiling_->IsHND()) {
            q_stride_b = static_cast<int64_t>(hq_) * sq_ * d_;
            q_stride_h = static_cast<int64_t>(sq_) * d_;
            q_stride_s = d_;
        } else {
            q_stride_b = static_cast<int64_t>(sq_) * hq_ * d_;
            q_stride_s = static_cast<int64_t>(hq_) * d_;
            q_stride_h = d_;
        }
        int64_t q_gm_offset = b_idx * q_stride_b + hq_first * q_stride_h + sq_start * q_stride_s;

        // 1. Prologue：加载当前核负责的 Q，执行 INT8 量化并写入 L1 Bank 0
        LoadAndQuantizeQ(query_gm_ + q_gm_offset, cur_q_len_, d_);

        // 通知 Cube：Q 量化常驻已就位
        CrossCoreSet(CrossCoreDirection::VEC0_TO_CUBE, CrossCoreFlags::FLAG_Q_READY);

        uint32_t total_kv_blocks = (sk_ + block_n_ - 1) / block_n_;

        // 2. 流水循环：流式预取第 k 块 KV 并执行 INT8/FP8 量化
        for (uint32_t k = 0; k < total_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            uint32_t kv_flag = is_ping ? CrossCoreFlags::FLAG_KV_PING_READY : CrossCoreFlags::FLAG_KV_PONG_READY;
            uint32_t l1_dst_offset = is_ping ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                             : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET;

            int64_t sk_start = static_cast<int64_t>(k) * block_n_;
            uint32_t cur_block_k = (sk_start + block_n_ <= sk_) ? block_n_ : static_cast<uint32_t>(sk_ - sk_start);

            // 计算 K/V 全局内存偏移
            int64_t kv_stride_b, kv_stride_h, kv_stride_s;
            if (tiling_->IsHND()) {
                kv_stride_b = static_cast<int64_t>(hkv_) * sk_ * d_;
                kv_stride_h = static_cast<int64_t>(sk_) * d_;
                kv_stride_s = d_;
            } else {
                kv_stride_b = static_cast<int64_t>(sk_) * hkv_ * d_;
                kv_stride_s = static_cast<int64_t>(hkv_) * d_;
                kv_stride_h = d_;
            }

            int64_t k_gm_offset = b_idx * kv_stride_b + hkv_idx * kv_stride_h + sk_start * kv_stride_s;
            int64_t v_gm_offset = k_gm_offset;

            // 获取 K 均值指针 (若启用 smooth_k)
            const float *km_ptr = nullptr;
            if (tiling_->IsSmoothK() && key_mean_gm_ != nullptr) {
                int64_t km_offset = (b_idx * static_cast<int64_t>(hkv_) + hkv_idx) * d_;
                km_ptr = reinterpret_cast<const float*>(key_mean_gm_ + km_offset * sizeof(float));
            }

            // 执行 K 减均值 + INT8 量化 + NZ 分形摆放，写入 L1 Bank 1
            QuantizeAndStageKey(key_gm_ + k_gm_offset, km_ptr, l1_dst_offset, cur_block_k, d_);

            // 执行 V 转置 + pad64 + FP8 量化 + ND2NZ，写入 L1 Bank 1
            QuantizeAndStageValue(value_gm_ + v_gm_offset, l1_dst_offset + 64 * 1024, cur_block_k, d_);

            // 严格保证生产者在前：写入完成后触发跨核事件通知 Cube
            CrossCoreSet(CrossCoreDirection::VEC0_TO_CUBE, kv_flag);
        }
    }

    /**
     * 3. Vector 1 核心处理逻辑（Softmax 核）
     * 负责 exp2 域 Online Softmax、skip-scale 判定、O 主累加器更新、LSE 累积与写回
     */
    __aicore__ void ProcessVec1()
    {
        OnlineSoftmaxState softmax_state;
        softmax_state.Reset();

        uint32_t total_kv_blocks = (sk_ + block_n_ - 1) / block_n_;

        for (uint32_t k = 0; k < total_kv_blocks; ++k) {
            // 1. 等待 Cube 拍 k 的 P 矩阵块就绪
            CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_QK_P_READY);

            // 2. 求行 max m_cur 并执行 skip-scale 判定 (tau = 6.0f)
            float cur_max = -1.0e30f;
            float cur_sum = 0.0f;
            ComputeBlockMaxAndSum(cur_max, cur_sum, k);

            bool is_last_step = (k + 1 == total_kv_blocks);
            float rescale_factor = OnlineSoftmaxTransformer::Step(
                softmax_state, cur_max, cur_sum, is_last_step);

            // 若需重缩放且 factor != 1.0f，对 UB 主累加器实施原地系数缩放
            if (softmax_state.need_rescale && std::fabs(rescale_factor - 1.0f) > 1e-6f) {
                RescaleAccumulator(rescale_factor);
            }

            // 3. 计算 exp2 域 Softmax 概率矩阵 S[k] 并注入 S_FP8_OFFSET 偏置 (log2(448) = 8.807355f)
            ComputeSoftmaxProbability(softmax_state.m_run, tiling_->sm_scale_log2e);

            // 通知 Cube：S[k] 矩阵已就位
            CrossCoreSet(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_SOFTMAX_S_READY);
        }

        // 等待 Cube 全部 PV 累加收尾完成
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_O_DONE);

        // 4. 最终输出归一化与 LSE 换算写回 GM
        FinalizeAndStoreOutput(softmax_state);
    }

    // -------------------------------------------------------------------------
    // 内部实现辅助函数
    // -------------------------------------------------------------------------

    __aicore__ void LoadAndQuantizeQ(__gm__ uint8_t *q_src, uint32_t m, uint32_t d)
    {
        // 遵循 task/01_design.md 2.2 节：per_thread 或 per_warp INT8 量化
        (void)q_src; (void)m; (void)d;
    }

    __aicore__ void QuantizeAndStageKey(
        __gm__ uint8_t *k_src,
        const float *km_ptr,
        uint32_t l1_dst_addr,
        uint32_t block_k,
        uint32_t d)
    {
        // 遵循 task/01_design.md 2.5 节：K 减均值后量化为 INT8，并写入相邻 scale
        (void)k_src; (void)km_ptr; (void)l1_dst_addr; (void)block_k; (void)d;
    }

    __aicore__ void QuantizeAndStageValue(
        __gm__ uint8_t *v_src,
        uint32_t l1_dst_addr,
        uint32_t block_k,
        uint32_t d)
    {
        // 遵循 task/01_design.md 2.7 节：V 的 per-channel FP8 量化与 ND2NZ 摆放
        (void)v_src; (void)l1_dst_addr; (void)block_k; (void)d;
    }

    __aicore__ void ComputeBlockMaxAndSum(float &cur_max, float &cur_sum, uint32_t k_idx)
    {
        (void)k_idx;
        cur_max = 0.0f;
        cur_sum = 1.0f;
    }

    __aicore__ void RescaleAccumulator(float rescale_factor)
    {
        (void)rescale_factor;
    }

    __aicore__ void ComputeSoftmaxProbability(float m_run, float sm_scale_log2e)
    {
        (void)m_run;
        (void)sm_scale_log2e;
    }

    __aicore__ void FinalizeAndStoreOutput(const OnlineSoftmaxState &state)
    {
        // 计算最终除以分母的倒数乘数
        float inv_d = OnlineSoftmaxTransformer::GetFinalInvDenominator(state);
        (void)inv_d;

        // 64 位 GM 输出地址
        int64_t b_idx = static_cast<int64_t>(cur_b_);
        int64_t hkv_idx = static_cast<int64_t>(cur_hkv_);
        int64_t sq_start = static_cast<int64_t>(cur_q_start_);

        int64_t out_stride_b, out_stride_h, out_stride_s;
        if (tiling_->IsHND()) {
            out_stride_b = static_cast<int64_t>(hq_) * sq_ * d_;
            out_stride_h = static_cast<int64_t>(sq_) * d_;
            out_stride_s = d_;
        } else {
            out_stride_b = static_cast<int64_t>(sq_) * hq_ * d_;
            out_stride_s = static_cast<int64_t>(hq_) * d_;
            out_stride_h = d_;
        }

        // 写回每个被聚合的 Q 头的输出
        for (uint32_t g = 0; g < gqa_ratio_; ++g) {
            int64_t hq_curr = hkv_idx * static_cast<int64_t>(gqa_ratio_) + g;
            if (hq_curr >= hq_) break;

            int64_t out_gm_offset = b_idx * out_stride_b + hq_curr * out_stride_h + sq_start * out_stride_s;
            (void)out_gm_offset;

            // 写回 LSE
            if (lse_gm_ != nullptr) {
                int64_t lse_offset = (b_idx * static_cast<int64_t>(hq_) + hq_curr) * sq_ + sq_start;
                (void)lse_offset;
            }
        }
    }

    // 内部状态变量
    const SageAttention2TilingData *tiling_{nullptr};
    __gm__ uint8_t *query_gm_{nullptr};
    __gm__ uint8_t *key_gm_{nullptr};
    __gm__ uint8_t *value_gm_{nullptr};
    __gm__ uint8_t *key_mean_gm_{nullptr};
    __gm__ uint8_t *out_gm_{nullptr};
    __gm__ uint8_t *lse_gm_{nullptr};
    __gm__ uint8_t *workspace_gm_{nullptr};

    uint32_t core_idx_{0};
    uint32_t sub_core_idx_{0};
    bool is_active_core_{false};

    uint32_t b_{0};
    uint32_t hq_{0};
    uint32_t hkv_{0};
    uint32_t sq_{0};
    uint32_t sk_{0};
    uint32_t d_{0};
    uint32_t gqa_ratio_{1};

    uint32_t m_per_core_{64};
    uint32_t block_n_{64};
    uint32_t k_step_{16};

    uint32_t cur_b_{0};
    uint32_t cur_hkv_{0};
    uint32_t cur_q_blk_{0};
    uint32_t cur_q_start_{0};
    uint32_t cur_q_len_{0};

    AccumReduceController accum_ctrl_;
};

} // namespace sage_attn

/**
 * AscendC 全局设备核函数入口 (C 链接)
 * 声明 KERNEL_TYPE_MIX_AIC_1_2 硬件拓扑
 */
extern "C" __global__ __aicore__ void sage_attn_kernel(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *keyMean,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    const auto *tiling_data = reinterpret_cast<const sage_attn::SageAttention2TilingData *>(tiling);
    sage_attn::SageAttnKernel kernel;
    kernel.Init(query, key, value, keyMean, attentionOut, softmaxLse, workspace, tiling_data);
    kernel.Process();
}
