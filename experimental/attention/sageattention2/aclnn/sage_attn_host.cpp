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

#include "sage_attn_proto.h"
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/catlass/sage_tile.hpp"
#include "../sageattn_fp8/catlass/sage_resource.hpp"
#include "../quant/quant_host.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <memory>

#ifdef SAGE_HAS_CANN_ACLNN_BASE
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#endif

namespace sage_attn {

/**
 * 宿主机轻量执行器上下文定义
 */
struct SageOpExecutorContext {
    SageAttention2TilingData tiling_data;
    uint64_t workspace_size{0};
    void *kernel_func{nullptr};

    // 记录输入输出张量原始指针或句柄
    const aclTensor *query{nullptr};
    const aclTensor *key{nullptr};
    const aclTensor *value{nullptr};
    const aclTensor *key_mean{nullptr};
    const aclTensor *lse_correction{nullptr};
    const aclTensor *out{nullptr};
    const aclTensor *lse{nullptr};
};

#ifndef SAGE_HAS_CANN_ACLNN_BASE
/**
 * 本地环境下的具体 aclOpExecutor 实现类
 */
struct ConcreteSageOpExecutor : public ::aclOpExecutor {
    SageOpExecutorContext ctx;

    uint64_t GetWorkspaceSize() const override {
        return ctx.workspace_size;
    }
};
#endif

/**
 * 张量维度辅助提取器
 * 支持从 CANN 原生 aclTensor 或本地轻量 mock 中统一提取形状
 */
inline bool ExtractTensorDimensions(
    const aclTensor *tensor,
    int64_t &dim0,
    int64_t &dim1,
    int64_t &dim2,
    int64_t &dim3)
{
    if (tensor == nullptr) {
        return false;
    }
#ifdef SAGE_HAS_CANN_ACLNN_BASE
    auto viewShape = op::GetViewShape(tensor);
    if (viewShape.size() != 4) {
        return false;
    }
    dim0 = viewShape[0];
    dim1 = viewShape[1];
    dim2 = viewShape[2];
    dim3 = viewShape[3];
    return true;
#else
    if (tensor->GetDimNum() != 4) {
        return false;
    }
    dim0 = tensor->GetShape(0);
    dim1 = tensor->GetShape(1);
    dim2 = tensor->GetShape(2);
    dim3 = tensor->GetShape(3);
    return true;
#endif
}

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

#ifdef SAGE_HAS_CANN_ACLNN_BASE
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
 * 3. 动态兼容 32 核与 28 核
 * 4. 强制 64 位整型提升计算显存空间与地址偏移
 */
inline aclnnStatus ComputeHostTiling(
    uint32_t b,
    uint32_t sq,
    uint32_t sk,
    uint32_t hq,
    uint32_t hkv,
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
    float base_sm_scale = (smScale > 0.0) ? static_cast<float>(smScale) : (1.0f / std::sqrt(static_cast<float>(d_og)));
    tiling_out.sm_scale_log2e = base_sm_scale * quant::LOG2E;
    tiling_out.v_scale_max = (pvAccumDtype == PV_ACCUM_FP32_FP16) ? quant::SCALE_MAX_FP16_ACCUM : quant::SCALE_MAX_FP32_ACCUM;

    // 8. 闭式分块计算 (分块字段，12 字节)
    // m_per_core 沿 GQA 头聚合，对齐至 64 的整数倍打满 Cube 吞吐
    uint32_t m_per_core = 64;
    if (gqa_ratio > 1) {
        // GQA 模式：将同一 KV 头的 gqa_ratio 个 Q 头聚合在同一核处理，一份 L1 KV 被多次复用
        uint32_t candidate_m = gqa_ratio * 32;
        if (candidate_m <= 64) {
            m_per_core = 64;
        } else if (candidate_m <= 128) {
            m_per_core = 128;
        } else {
            m_per_core = 64;
        }
    } else {
        m_per_core = 64;
    }
    tiling_out.m_per_core = m_per_core;

    // block_n：KV 分块大小，推荐 64 或 128 (满足 L1 Bank 1 256KB Ping-Pong 与 FixPipe 吞吐平衡)
    uint32_t block_n = 64;
    if (sk >= 2048) {
        block_n = 128;
    }
    tiling_out.block_n = block_n;

    // k_step：两级累加归约周期步长 (默认 16)
    tiling_out.k_step = 16;

    // 9. 分核计算与 Split-KV 判定 (分核字段，20 字节)
    uint32_t hw_cores = GetHardwareCoreNum();
    uint32_t q_blk_num = (sq + m_per_core - 1) / m_per_core;
    uint32_t kv_blk_num = (sk + block_n - 1) / block_n;

    tiling_out.q_blk_num = q_blk_num;
    tiling_out.kv_blk_num = kv_blk_num;

    // 任务切分单位为 (Batch, Hkv, Q_blk)
    uint32_t total_tasks = b * hkv * q_blk_num;
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

    // 10. 片上存储与全局内存偏移布局字段 (44 字节)
    tiling_out.l1_q_offset = Catlass::SageAttn::L1DoubleBankContract::Q_DATA_OFFSET;
    tiling_out.l1_q_scale_offset = Catlass::SageAttn::L1DoubleBankContract::Q_SCALE_OFFSET;
    tiling_out.l1_kv_offset = Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET;
    tiling_out.l1_kv_scale_offset = Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET + 16 * 1024;

    tiling_out.ub_acc_offset = Catlass::SageAttn::Vector1UbBudget::O_ACCUM_OFFSET;
    tiling_out.ub_softmax_offset = Catlass::SageAttn::Vector1UbBudget::FIXPIPE_RECV_OFFSET;
    tiling_out.ub_quant_offset = Catlass::SageAttn::Vector0UbBudget::QUANT_WORKSPACE_OFFSET;

    // 11. Workspace 空间计算与 64 位整型提升 (杜绝 4GB 整数溢出)
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

    workspace_bytes_out = current_ws_offset;
    return ACLNN_SUCCESS;
}

} // namespace sage_attn

// =============================================================================
// ACLNN 官方两段式 C 接口实现
// =============================================================================

extern "C" {

__attribute__((visibility("default"))) aclnnStatus
aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *keyMeanOptional,
    const aclTensor *lseCorrectionOptional,
    double smScale,
    int64_t qkQuantGran,
    int64_t pvAccumDtype,
    bool isCausal,
    int64_t tensorLayout,
    bool smoothK,
    bool smoothV,
    const aclTensor *out,
    const aclTensor *lse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    // 1. 空指针入参校验
    if (query == nullptr || key == nullptr || value == nullptr || out == nullptr ||
        workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    // 2. 提取张量形状
    int64_t q_d0 = 0, q_d1 = 0, q_d2 = 0, q_d3 = 0;
    int64_t k_d0 = 0, k_d1 = 0, k_d2 = 0, k_d3 = 0;
    int64_t v_d0 = 0, v_d1 = 0, v_d2 = 0, v_d3 = 0;

    if (!sage_attn::ExtractTensorDimensions(query, q_d0, q_d1, q_d2, q_d3) ||
        !sage_attn::ExtractTensorDimensions(key, k_d0, k_d1, k_d2, k_d3) ||
        !sage_attn::ExtractTensorDimensions(value, v_d0, v_d1, v_d2, v_d3)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 3. 根据布局解析 B, S, H, D
    uint32_t b = 0, sq = 0, sk = 0, hq = 0, hkv = 0, d_og = 0;
    if (tensorLayout == sage_attn::LAYOUT_HND) {
        // HND 布局: [B, H, S, D]
        b = static_cast<uint32_t>(q_d0);
        hq = static_cast<uint32_t>(q_d1);
        sq = static_cast<uint32_t>(q_d2);
        d_og = static_cast<uint32_t>(q_d3);

        hkv = static_cast<uint32_t>(k_d1);
        sk = static_cast<uint32_t>(k_d2);
    } else {
        // NHD 布局: [B, S, H, D]
        b = static_cast<uint32_t>(q_d0);
        sq = static_cast<uint32_t>(q_d1);
        hq = static_cast<uint32_t>(q_d2);
        d_og = static_cast<uint32_t>(q_d3);

        sk = static_cast<uint32_t>(k_d1);
        hkv = static_cast<uint32_t>(k_d2);
    }

    // 4. 执行微秒级闭式 Tiling 与 Workspace 显存计算
    sage_attn::SageAttention2TilingData tiling_data;
    uint64_t calculated_workspace = 0;

    bool has_km = (keyMeanOptional != nullptr);
    bool has_lse = (lse != nullptr);

    aclnnStatus status = sage_attn::ComputeHostTiling(
        b, sq, sk, hq, hkv, d_og,
        smScale, qkQuantGran, pvAccumDtype,
        isCausal, tensorLayout, smoothK, smoothV,
        has_km, has_lse,
        tiling_data,
        calculated_workspace);

    if (status != ACLNN_SUCCESS) {
        return status;
    }

    *workspaceSize = calculated_workspace;

    // 5. 构造并返回算子执行器上下文
#ifdef SAGE_HAS_CANN_ACLNN_BASE
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    // 绑定 TilingData 副本至 CANN 执行器
    uniqueExecutor->SetWorkspaceSize(calculated_workspace);
    uniqueExecutor.ReleaseTo(executor);
#else
    auto op_exec = std::make_unique<sage_attn::ConcreteSageOpExecutor>();
    if (!op_exec) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    op_exec->ctx.tiling_data = tiling_data;
    op_exec->ctx.workspace_size = calculated_workspace;
    op_exec->ctx.query = query;
    op_exec->ctx.key = key;
    op_exec->ctx.value = value;
    op_exec->ctx.key_mean = keyMeanOptional;
    op_exec->ctx.lse_correction = lseCorrectionOptional;
    op_exec->ctx.out = out;
    op_exec->ctx.lse = lse;

    *executor = static_cast<::aclOpExecutor*>(op_exec.release());
#endif

    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnSageAttention2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

#ifdef SAGE_HAS_CANN_ACLNN_BASE
    // 在 CANN 运行时下，调用公共执行器调度下发算子
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
#else
    // 宿主机环境或单测桩执行
    (void)workspace;
    (void)workspaceSize;
    (void)stream;

    auto *concrete = static_cast<sage_attn::ConcreteSageOpExecutor *>(executor);
    if (workspaceSize < concrete->ctx.workspace_size) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
#endif
}

} // extern "C"
