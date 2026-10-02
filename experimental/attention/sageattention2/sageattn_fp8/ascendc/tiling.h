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

#ifndef SAGE_ATTN_FP8_ASCENDC_TILING_H
#define SAGE_ATTN_FP8_ASCENDC_TILING_H

#include <cstdint>
#include <cstddef>

namespace sage_attn {

/**
 * 片上存储容量与硬件拓扑常量（白皮书表 4-2 与第 3 节硬件规格）
 */
constexpr uint32_t L1_TOTAL_SIZE = 512 * 1024;         // L1 总容量 512KB
constexpr uint32_t L1_BANK_SIZE = 256 * 1024;          // 单 Bank 容量 256KB
constexpr uint32_t L1_BANK0_OFFSET = 0;                // Bank 0 起始偏移 (0~256KB, Q 与 Q scale)
constexpr uint32_t L1_BANK1_OFFSET = 256 * 1024;       // Bank 1 起始偏移 (256~512KB, KV 与 KV scale)

constexpr uint32_t UB_TOTAL_SIZE_PER_V = 256 * 1024;   // 单 Vector 核心 UB 容量 256KB (V0 256KB, V1 256KB)
constexpr uint32_t UB_TOTAL_SIZE = 512 * 1024;         // 每 AI Core 包含 2 个 Vector 核心，UB 总量 512KB

constexpr uint32_t L0A_SIZE = 64 * 1024;               // L0A 矩阵乘输入 A 寄存器 64KB
constexpr uint32_t L0B_SIZE = 64 * 1024;               // L0B 矩阵乘输入 B 寄存器 64KB
constexpr uint32_t L0C_SIZE = 256 * 1024;              // L0C 累加器 256KB

constexpr uint32_t ATLAS_950_AIC_NUM = 32;             // Atlas 950PR 满配 32 个 AI Core
constexpr uint32_t ATLAS_950_AIV_NUM = 64;             // Atlas 950PR 满配 64 个 Vector Core (1C2V 架构)

/**
 * 布局编码枚举（对齐 core.py 行 745）
 */
enum TensorLayout : uint32_t {
    LAYOUT_NHD = 0,
    LAYOUT_HND = 1
};

/**
 * 量化粒度枚举（对齐 core.py 行 747）
 */
enum QkQuantGran : uint32_t {
    QUANT_GRAN_PER_WARP = 2,
    QUANT_GRAN_PER_THREAD = 3
};

/**
 * PV 累加精度档位枚举（task/01_design.md 2.4 节冻结）
 */
enum PvAccumDtype : uint32_t {
    PV_ACCUM_FP32 = 0,       // 全程 FP32 累加
    PV_ACCUM_FP32_FP32 = 1,  // 硬件累加，每 k_step 周期归约至 FP32 buffer
    PV_ACCUM_FP32_FP16 = 2   // 低精度累加，每 k_step 周期归约至 FP16 buffer (默认验收档)
};

/**
 * TilingData 结构体（Host 与 Device 共享只读）
 * 使用 __attribute__((packed, aligned(32))) 强制 32 字节对齐
 * 结构体总大小严格补齐为 32 字节整数倍 (160 字节)
 */
struct __attribute__((packed, aligned(32))) SageAttention2TilingData {
    // 1. shape 字段 (28 字节)
    uint32_t b;               // 批次大小 batch_size
    uint32_t hq;              // Query 头的数量 num_qo_heads
    uint32_t hkv;             // Key/Value 头的数量 num_kv_heads
    uint32_t sq;              // Query 序列长度 qo_len
    uint32_t sk;              // Key/Value 序列长度 kv_len
    uint32_t d_og;            // 原始头维度 head_dim_og (<= 128)
    uint32_t d_pad;           // 对齐填充后的头维度 (64 或 128)

    // 2. 模式字段 (24 字节)
    uint32_t is_causal;       // 是否因果注意力 (0=False, 1=True)
    uint32_t tensor_layout;   // 张量布局 (0=NHD, 1=HND)
    uint32_t qk_quant_gran;   // QK 量化粒度 (2=per_warp, 3=per_thread)
    uint32_t pv_accum_dtype;  // PV 累加精度 (0=fp32, 1=fp32+fp32, 2=fp32+fp16)
    uint32_t smooth_k;        // 是否启用 smooth_k (0=False, 1=True)
    uint32_t smooth_v;        // 是否启用 smooth_v (0=False, 1=True)

    // 3. 数值字段 (8 字节)
    float sm_scale_log2e;     // Host 端折叠后的缩放因子: sm_scale * log2(e) (约 1.44269504)
    float v_scale_max;        // V 量化饱和上限 (fp32+fp16 为 2.25f, 其余为 448.0f)

    // 4. 分块字段 (12 字节)
    uint32_t m_per_core;      // 每个核处理的 Query 行数 (GQA 聚合后对齐为 64 的倍数)
    uint32_t block_n;         // KV 分块大小 (推荐 64 或 128)
    uint32_t k_step;          // 两级累加归约周期步长 (8, 16 或 32)

    // 5. 分核字段 (20 字节)
    uint32_t core_num_used;   // 实际调度的 AI Core 数量 (<= 32)
    uint32_t q_blk_num;       // Query 维度分块数
    uint32_t kv_blk_num;      // Key/Value 维度分块数
    uint32_t split_kv_num;    // Split-KV 切分数 (用于尾核利用)
    uint32_t is_merge;        // 是否为 Split-KV 二次归并核标志

    // 6. 布局与片上/全局偏移字段 (44 字节)
    uint32_t l1_q_offset;          // L1 Bank 0 中 Q 矩阵的字节偏移
    uint32_t l1_q_scale_offset;    // L1 Bank 0 中 Q scale 的字节偏移
    uint32_t l1_kv_offset;         // L1 Bank 1 中 KV 矩阵的起始字节偏移
    uint32_t l1_kv_scale_offset;   // L1 Bank 1 中 KV scale 的起始字节偏移
    uint32_t ub_acc_offset;        // V1 UB 中 O 累加器起始字节偏移
    uint32_t ub_softmax_offset;    // V1 UB 中 Softmax 中间结果起始字节偏移
    uint32_t ub_quant_offset;      // V0 UB 中量化临时缓冲起始字节偏移
    uint64_t workspace_lse_offset; // GM 工作区中 LSE 张量偏移 (字节)
    uint64_t workspace_vm_offset;  // GM 工作区中 V 均值张量偏移 (字节, 仅用于 fuse_v_mean)

    // 7. 补齐保留字段 (24 字节) 使结构体总大小为 160 字节 (32 字节整数倍)
    uint8_t reserved[24];

    // 便捷访问辅助函数
    uint32_t GetGqaRatio() const {
        return (hkv > 0) ? (hq / hkv) : 1;
    }

    bool IsCausal() const {
        return is_causal != 0;
    }

    bool IsSmoothK() const {
        return smooth_k != 0;
    }

    bool IsSmoothV() const {
        return smooth_v != 0;
    }

    bool IsMerge() const {
        return is_merge != 0;
    }

    bool IsHND() const {
        return tensor_layout == LAYOUT_HND;
    }

    bool IsPerThread() const {
        return qk_quant_gran == QUANT_GRAN_PER_THREAD;
    }

    bool IsPerWarp() const {
        return qk_quant_gran == QUANT_GRAN_PER_WARP;
    }
};

// 静态断言验证结构体大小和对齐属性
static_assert(sizeof(SageAttention2TilingData) == 160, "SageAttention2TilingData size must be exactly 160 bytes");
static_assert(sizeof(SageAttention2TilingData) % 32 == 0, "SageAttention2TilingData size must be a multiple of 32 bytes");
static_assert(alignof(SageAttention2TilingData) == 32, "SageAttention2TilingData must be 32-byte aligned");

using TilingData = SageAttention2TilingData;

} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_TILING_H
