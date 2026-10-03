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

#ifndef SAGE_ATTN_HOST_TILING_H
#define SAGE_ATTN_HOST_TILING_H

/**
 * \file sage_attn_host_tiling.h
 * \brief 闭式 Host Tiling 计算核心（inline 头文件实现）
 *
 * ComputeHostTiling 为纯函数（仅依赖 tiling.h / catlass / quant_host 头），
 * 同时被 aclnn/sage_attn_host.cpp（两段式 C 接口）与
 * op_host/sage_attention2_tiling.cpp（gert TilingImpl）包含，
 * 保证 GetWorkspaceSize 阶段与 TilingImpl 阶段结果位元一致。
 * 以 inline 头文件形式组织，使 op_host 库无需链接 op_api 层目标即可获得定义。
 */

#include "sage_attn_proto.h"
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/catlass/sage_tile.hpp"
#include "../sageattn_fp8/catlass/sage_resource.hpp"
#include "../quant/quant_host.h"

#include <cmath>
#include <cstring>
#include <cstdlib>

// 平台核数查询（opdev/platform.h）依赖 libnnopbase 运行时符号；op_api 层 TU
// （aclnn/sage_attn_host.cpp）在包含本头文件前定义 SAGE_ATTN_HAVE_PLATFORM_INFO
// 以启用动态核数探测。op_host 库（被 op_build dlopen）不得引用 libnnopbase
// 符号，走环境变量覆盖 + 默认 32 核的闭式路径。
#if defined(SAGE_ATTN_HAVE_PLATFORM_INFO) && defined(SAGE_HAS_CANN_ACLNN_BASE)
#include "opdev/platform.h"
#endif

namespace sage_attn {

/**
 * 获取硬件可用 AI Core 核心数量
 * 动态检测 32 核 (Atlas 950PR 满配) 与 28 核 (减配版)
 */
inline uint32_t GetHardwareCoreNum()
{
    // 1. 优先读取用户显式环境变量覆盖 (如自动化测试或特定拓扑)
    const char *env_cores = std::getenv("SAGEATTN_NUM_CORES");
    if (env_cores != nullptr) {
        int c = std::atoi(env_cores);
        if (c > 0 && c <= 64) {
            return static_cast<uint32_t>(c);
        }
    }

#if defined(SAGE_ATTN_HAVE_PLATFORM_INFO) && defined(SAGE_HAS_CANN_ACLNN_BASE)
    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t cubeCores = npuInfo.GetCubeCoreNum();
    if (cubeCores > 0) {
        return cubeCores;
    }
#endif

    // 默认 Atlas 950PR 标配 32 个 AI Core
    return ATLAS_950_AIC_NUM;
}

/**
 * 闭式超轻量微秒级 Host Tiling 计算核心
 * 核心设计指标：
 * 1. 扁平代数闭式计算，无复杂动态循环与大表搜索，平均启动开销 < 2 微秒
 * 2. 彻底打赢对 FIA 3.7 万行无 cache 的 Host Launch 时延之战
 * 3. 动态支持 32 核与 28 核
 * 4. 强制 64 位整型提升计算显存空间与地址偏移
 *
 * inline 唯一定义（本头被 aclnn/sage_attn_host.cpp 与 op_host/sage_attention2_tiling.cpp
 * 同时包含，且二者在独立构建目标 sageattention2_host 中共链——无 inline 会在共享库
 * 链接期报 duplicate symbol，R6 审查实测捕获）；inline 弱符号即 proto.h 注释所述
 * "弱符号，避免 op_host 库反向依赖 op_api 层目标"的落地形式。
 */
inline aclnnStatus ComputeHostTiling(
    uint32_t b,
    uint32_t sq,
    uint32_t sk,
    uint32_t hq,
    uint32_t hkv,
    int dtype_mode,      // 输入位型分派：1=fp16, 2=bf16，写 TilingData.input_dtype
    uint32_t d_og,
    double smScale,
    int64_t qkQuantGran,
    int64_t pvAccumDtype,
    bool isCausal,
    int64_t tensorLayout,
    bool smoothK,
    bool smoothV,
    bool hasKeyMean,
    bool hasLse,
    SageAttention2TilingData &tiling_out,
    uint64_t &workspace_bytes_out)
{
    // 1. 头维度对齐与校验 (head_dim 小于等于 64 补 64，(64, 128] 补 128)
    if (d_og == 0 || d_og > 128) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint32_t d_pad = (d_og <= 64) ? 64 : 128;

    // 2. GQA 整除性校验与比例计算
    if (hkv == 0 || (hq % hkv) != 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint32_t gqa_ratio = hq / hkv;

    // 3. 量化粒度与累加档位校验
    if (qkQuantGran != QUANT_GRAN_PER_WARP && qkQuantGran != QUANT_GRAN_PER_THREAD) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (pvAccumDtype < PV_ACCUM_FP32 || pvAccumDtype > PV_ACCUM_FP32_FP16) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // tensor_layout 编码域校验：仅接受冻结编码 NHD(0)/HND(1)，非法数值显式报错
    // 不静默归一（与 canonical op_api 的"非法布局报错不降级"口径及同函数
    // gran/accum 严格校验一致；GE 字符串映射层经 *CodeFromString 恒产 0/1，不受影响）
    if (tensorLayout != LAYOUT_NHD && tensorLayout != LAYOUT_HND) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 4. 清空结构体 (160 字节，32 字节对齐)
    std::memset(&tiling_out, 0, sizeof(SageAttention2TilingData));

    // 5. 填充 Shape 字段 (28 字节)
    tiling_out.b = b;
    tiling_out.hq = hq;
    tiling_out.hkv = hkv;
    tiling_out.sq = sq;
    tiling_out.sk = sk;
    tiling_out.d_og = d_og;
    tiling_out.d_pad = d_pad;

    // 6. 填充模式字段 (24 字节)
    tiling_out.is_causal = isCausal ? 1 : 0;
    tiling_out.tensor_layout = (tensorLayout == LAYOUT_HND) ? LAYOUT_HND : LAYOUT_NHD;
    tiling_out.qk_quant_gran = static_cast<uint32_t>(qkQuantGran);
    tiling_out.pv_accum_dtype = static_cast<uint32_t>(pvAccumDtype);
    tiling_out.smooth_k = (smoothK && hasKeyMean) ? 1 : 0;
    // 对齐设计文档：若非全精度 fp32 档，则忽略 smooth_v
    tiling_out.smooth_v = (smoothV && pvAccumDtype == PV_ACCUM_FP32) ? 1 : 0;

    // 7. 预计算与常量折叠数值字段 (8 字节)
    // sm_scale 冻结语义：默认值 1/sqrt(d) 仅由上层桥接在
    // Python sm_scale=None 时折叠后传入；C 接口收到 0 或负值一律判非法报错
    if (!(smScale > 0.0) || !std::isfinite(smScale)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    float base_sm_scale = static_cast<float>(smScale);
    tiling_out.sm_scale_log2e = base_sm_scale * quant::LOG2E;
    tiling_out.v_scale_max = quant::GetVScaleMax(static_cast<uint32_t>(pvAccumDtype));
    // s_fp8_offset 契约冻结：仅允许缺省 0（基线，448· 经乘法承担）或冻结常量
    // S_FP8_OFFSET（映射回基线 0，op_kernel Init 归一化）。
    tiling_out.s_fp8_offset = quant::S_FP8_OFFSET;
    // 输入位型分派：fp16/bf16 三方（def 注册、op_api、核内读写）一致
    tiling_out.input_dtype = static_cast<uint8_t>(dtype_mode);

    // 8. 闭式分块计算 (分块字段，12 字节)
    // m_per_core 沿 GQA 头聚合：kernel 的 M 维打包 = 组内各 Q 头行连续
    //（TotalRows = gqa·cur_q_len，行容量 kKernelMaxRows = 128），故
    // gqa·m_per_core <= 128 为硬约束；同时 m_per_core >= 32 保证
    // per_thread 32-token 子块的 scale 组不跨 tile 切割。gqa>4 时打包不下，显式拒绝。
    uint32_t m_per_core = 64;
    if (gqa_ratio > 1) {
        if (gqa_ratio > 4u) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        m_per_core = (gqa_ratio == 2u) ? 64u : 32u;
    }
    tiling_out.m_per_core = m_per_core;

    // block_n：KV 分块大小，推荐 64 或 128 (满足 L1 Bank 1 256KB Ping-Pong 与 FixPipe 吞吐平衡)
    uint32_t block_n = 64;
    if (sk >= 2048) {
        block_n = 128;
    }
    tiling_out.block_n = block_n;

    // 运行时资源台账 fail-fast：
    // O 累加器/S 双槽/L0C/L1 的足迹随 (gqa·m_per_core, block_n, d_pad) 变化，
    // 必须在发射前拒绝越界分块。
    const uint32_t total_rows = m_per_core * gqa_ratio;
    if (Catlass::SageAttn::SageResourceManager::InspectTilingPlan(total_rows, block_n, d_pad) !=
        Catlass::SageAttn::FootprintViolation::kNone) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // k_step：两级累加归约周期步长 (默认 16)
    tiling_out.k_step = DEFAULT_K_STEP; // 唯一定义点 tiling.h

    // 9. 分核计算与 Split-KV 判定 (分核字段，20 字节)
    uint32_t hw_cores = GetHardwareCoreNum();
    uint32_t q_blk_num = (sq + m_per_core - 1) / m_per_core;
    uint32_t kv_blk_num = (sk + block_n - 1) / block_n;

    tiling_out.q_blk_num = q_blk_num;
    tiling_out.kv_blk_num = kv_blk_num;

    // 任务切分单位为 (Batch, Hkv, Q_blk)；64 位乘法防回绕
    const uint64_t total_tasks_64 = static_cast<uint64_t>(b) * hkv * q_blk_num;
    if (total_tasks_64 > UINT32_MAX) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint32_t total_tasks = static_cast<uint32_t>(total_tasks_64);
    uint32_t split_kv_num = 1;

    // 尾核利用优化：当任务总数不足以填满硬件核心时，自动沿 KV 序列维实施 Split-KV
    if (total_tasks < hw_cores && kv_blk_num > 1) {
        uint32_t factor = (hw_cores + total_tasks - 1) / total_tasks;
        split_kv_num = (factor < kv_blk_num) ? factor : kv_blk_num;
        if (split_kv_num > 8) {
            split_kv_num = 8;
        }
    }
    tiling_out.split_kv_num = split_kv_num;
    tiling_out.is_merge = 0;

    uint32_t scheduled_tasks = total_tasks * split_kv_num;
    tiling_out.core_num_used = (scheduled_tasks < hw_cores) ? scheduled_tasks : hw_cores;

    // 10. Die 亲和性多核调度分核计算 (Atlas 950PR 双 Die / 8-Group 拓扑)
    // 硬件拓扑：Atlas 950PR 标配 2 个计算 Die，每个 Die 拥有独立的片上 L2 Cache 与 AI Core 组 (各 4 个 Core Group)
    // 若简单按线性顺序平铺 task_id，同一 Query 块对应的任务会频繁跨 Die 调度，导致跨 Die 互联总线拥塞与远端 L2 Cache Miss
    // 判定准则：无 Split-KV 切分、核心数 >= 2 且为偶数 (标配 32 核或减配 28 核)，且任务数 >= 2 且为偶数时启用 Die 亲和性调度
    uint32_t die_num = ATLAS_950_DIE_NUM;
    uint32_t active_cores = tiling_out.core_num_used;
    bool enable_die = (split_kv_num == 1) &&
                      (active_cores >= die_num) && ((active_cores % die_num) == 0) &&
                      (total_tasks >= die_num) && ((total_tasks % die_num) == 0);

    tiling_out.enable_die_affinity = enable_die ? 1 : 0;
    tiling_out.die_num = die_num;
    tiling_out.cores_per_die = (active_cores >= die_num) ? (active_cores / die_num) : active_cores;
    tiling_out.tasks_per_die = (total_tasks + die_num - 1) / die_num;

    // 11. 片上存储与全局内存偏移布局字段 (44 字节)
    tiling_out.l1_q_offset = Catlass::SageAttn::L1DoubleBankContract::Q_DATA_OFFSET;
    tiling_out.l1_q_scale_offset = Catlass::SageAttn::L1DoubleBankContract::Q_SCALE_OFFSET;
    tiling_out.l1_kv_offset = Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET;
    // l1_kv_scale_offset：KV 槽内 Scale 保留区起点，与 Bank 0 的 Q Data 区同宽
    // （Q_MAX_BYTES = 128 行 × 128 列 INT8 = 16KB，唯一定义见 sage_tile.hpp，替代原 16 * 1024 字面量）
    tiling_out.l1_kv_scale_offset = Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET +
                                    Catlass::SageAttn::L1DoubleBankContract::Q_MAX_BYTES;

    tiling_out.ub_acc_offset = Catlass::SageAttn::Vector1UbBudget::O_ACCUM_OFFSET;
    tiling_out.ub_softmax_offset = Catlass::SageAttn::Vector1UbBudget::FIXPIPE_RECV_OFFSET;
    tiling_out.ub_quant_offset = Catlass::SageAttn::Vector0UbBudget::QUANT_WORKSPACE_OFFSET;

    // 12. Workspace 空间计算与 64 位整型提升 (杜绝 4GB 整数溢出)
    uint64_t current_ws_offset = 0;
    auto align512 = [](uint64_t size) -> uint64_t {
        return ((size + 511ULL) / 512ULL) * 512ULL;
    };

    // (1) LSE 临时显存空间 (Split-KV 跨核归并或 smooth_k 修正)
    if (split_kv_num > 1 || hasLse) {
        tiling_out.workspace_lse_offset = current_ws_offset;
        int64_t lse_elems = static_cast<int64_t>(b) * hq * sq * split_kv_num;
        uint64_t lse_bytes = align512(static_cast<uint64_t>(lse_elems) * sizeof(float));
        current_ws_offset += lse_bytes;
    } else {
        tiling_out.workspace_lse_offset = 0;
    }

    // (2) Split-KV partial outputs 临时累加显存
    if (split_kv_num > 1) {
        int64_t partial_o_elems = static_cast<int64_t>(b) * hq * sq * d_pad * split_kv_num;
        uint64_t partial_o_bytes = align512(static_cast<uint64_t>(partial_o_elems) * sizeof(float));
        current_ws_offset += partial_o_bytes;
    }

    // (3) fuse_v_mean 临时均值空间
    if (tiling_out.IsSmoothV()) {
        tiling_out.workspace_vm_offset = current_ws_offset;
        int64_t vm_elems = static_cast<int64_t>(b) * hkv * d_pad;
        uint64_t vm_bytes = align512(static_cast<uint64_t>(vm_elems) * sizeof(float));
        current_ws_offset += vm_bytes;
    } else {
        tiling_out.workspace_vm_offset = 0;
    }

    // (4) A7 V0→V1 scale 暂存条带（task/12_device_data_path.md 第四节；2026-10-03）：
    // 布局常量唯一定义 tiling.h STAGE_*（每核 STAGE_CORE_STRIDE 字节 × 参与核数），
    // kernel 侧以 core_idx × STAGE_CORE_STRIDE 寻址 GetUserWorkspace 用户区。
    current_ws_offset += align512(static_cast<uint64_t>(tiling_out.core_num_used) *
                                  sage_attn::STAGE_CORE_STRIDE);

    workspace_bytes_out = current_ws_offset;
    return ACLNN_SUCCESS;
}

} // namespace sage_attn

#endif // SAGE_ATTN_HOST_TILING_H
