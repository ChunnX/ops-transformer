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
#include "autotune_tiling_bank.h"
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
 * 双 Die 拓扑亲和调度判定准则 (inspired by competitor common.h):
 * Atlas 950PR 标配 2 个计算 Die，各拥有 16 个 AI Core 与 64MB 独立片上 L2 Cache。
 * 当参与调度的核心数为偶数 (标配 32 核或减配 28 核) 且 (b * hkv) 能被 2 整除时，
 * 满足双 Die 亲和调度条件：
 * - Die 0 (核 0..cores/2-1) 亲和前半段 KV heads/batches [0, (b * hkv)/2)
 * - Die 1 (核 cores/2..cores-1) 亲和后半段 KV heads/batches [(b * hkv)/2, b * hkv)
 * 使得对应 KV 头所有 Q 块任务全部收敛在同一 Die 内部物理核上执行，
 * 保证 100% 的 KV reuse 命中对应 Die 的 64MB 本地 L2 Cache，彻底消除跨 Die 互联颠簸 (cross-Die bouncing)。
 */
inline bool UseDieGroups(uint32_t cores, uint32_t b, uint32_t hkv) noexcept
{
    return cores >= 2 && (cores % 2 == 0) && ((static_cast<uint64_t>(b) * hkv) % 2 == 0);
}

/**
 * 计算指定物理核心归属的计算 Die (Die 0: 0..cores/2-1, Die 1: cores/2..cores-1)
 */
inline uint32_t GetDieIndexForCore(uint32_t core, uint32_t cores = ATLAS_950_AIC_NUM) noexcept
{
    return (core < (cores / 2)) ? 0 : 1;
}

/**
 * 计算指定 Batch / Hkv 头在双 Die 拓扑下的亲和归属 Die
 */
inline uint32_t GetKvHeadDieAffinity(uint32_t b_idx, uint32_t hkv_idx, uint32_t b, uint32_t hkv) noexcept
{
    uint64_t total_bh = static_cast<uint64_t>(b) * hkv;
    uint64_t cur_bh = static_cast<uint64_t>(b_idx) * hkv + hkv_idx;
    return (cur_bh < (total_bh / 2)) ? 0 : 1;
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
    // 任务书 §2.2：因果掩码仅当 Sq == Sk 合法（DEFECT-L-2：原仅 python/op_api 入口
    // 拒绝，移植态 C-ABI 与 gert tiling 路径可达 Sq>Sk causal，共享点一处拒绝全覆盖）
    if (isCausal && sq != sk) {
        return ACLNN_ERR_PARAM_INVALID;
    }
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

    // 8. 闭式分块计算 (分块字段，12 字节) 与 Auto-Tuned Tiling 调优知识库查询
    // 优先查询 Ascend 950PR 经验调优知识库 (autotune::QueryTunedTiling)：
    // 命中调优用例 (P-01 ~ P-07) 时应用实测最优超参 (m_per_core, block_n, k_step,
    // causal_early_exit, use_die_grouped)；未命中或动态 shape 时平滑回退至健壮的闭式推导。
    const autotune::TunedTilingEntry *tuned = nullptr;
    const char *env_tune_case = std::getenv("SAGEATTN_TUNE_CASE");
    if (env_tune_case != nullptr) {
        tuned = autotune::QueryTunedTiling(env_tune_case);
    }
    if (tuned == nullptr) {
        tuned = autotune::QueryTunedTiling(b, sq, sk, hq, hkv, d_og, dtype_mode, isCausal);
    }

    uint32_t m_per_core = 64;
    if (gqa_ratio > 1) {
        // GQA 分块打包（官方 500 用例含 32:4 即 8:1）：行打包上限 MAX_PACK_ROWS，
        // 单任务打包 min(gqa_ratio, MAX_PACK_ROWS / m_per_core) 个 Q 头，超出按 chunk
        // 拆多任务（kernel DecodeTask 同式推导，任务空间 b*hkv*chunks*q_blks）。
        if (gqa_ratio > MAX_GQA_RATIO) {
            // 上限 8：SimMemory/GM scale 暂存条带按 8 头配额冻结（官方套件最大 8:1）
            return ACLNN_ERR_PARAM_INVALID;
        }
        m_per_core = (gqa_ratio == 2u) ? 64u : 32u;
    }
    uint32_t block_n = (sk >= 2048) ? 128 : 64;
    uint32_t k_step = DEFAULT_K_STEP;
    bool tuned_die_grouped = false;
    bool has_tuned = false;
    uint8_t causal_early_exit_val = isCausal ? 1 : 0;

    if (tuned != nullptr) {
        has_tuned = true;
        m_per_core = tuned->m_per_core;
        block_n = tuned->block_n;
        k_step = tuned->k_step;
        causal_early_exit_val = tuned->causal_early_exit ? 1 : 0;
        tuned_die_grouped = tuned->use_die_grouped;
    }

    tiling_out.m_per_core = m_per_core;
    tiling_out.block_n = block_n;
    tiling_out.k_step = k_step;
    tiling_out.causal_early_exit = causal_early_exit_val;

    // 运行时资源台账 fail-fast：
    // O 累加器/S 双槽/L0C/L1 的足迹随 (gqa·m_per_core, block_n, d_pad) 变化，
    // 必须在发射前拒绝越界分块。
    const uint32_t pack_heads = (gqa_ratio > 0)
        ? ((gqa_ratio < MAX_PACK_ROWS / m_per_core) ? gqa_ratio : MAX_PACK_ROWS / m_per_core)
        : 1u;
    const uint32_t total_rows = m_per_core * pack_heads;
    if (Catlass::SageAttn::SageResourceManager::InspectTilingPlan(total_rows, block_n, d_pad) !=
        Catlass::SageAttn::FootprintViolation::kNone) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 9. 分核计算与 Split-KV 判定 (分核字段，20 字节)
    // 分块计数的向上取整统一走官方 CATLASS 对齐组件（catlass/detail/alignment.hpp
    // 全局 ::CeilDiv，constexpr 纯整数算术，与手写 (a+b-1)/b 位元一致）
    uint32_t hw_cores = GetHardwareCoreNum();
    uint32_t q_blk_num = ::CeilDiv(sq, m_per_core);
    uint32_t kv_blk_num = ::CeilDiv(sk, block_n);

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
        uint32_t factor = ::CeilDiv(hw_cores, total_tasks);
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
    // 竞品 common.h DieTask 启发与双 Die 拓扑亲和调度：
    // 硬件拓扑：Atlas 950PR 标配 2 个计算 Die，每个 Die 拥有独立的片上 64MB L2 Cache 与 AI Core 组 (各 4 个 Core Group)
    // 判定准则：当参与调度核心数为 32 (或偶数核心) 且 (b * hkv) 能被 2 整除时，
    // 将任务严格按 Die 切分，Die 0 (核 0..cores/2-1) 亲和前半段 KV heads/batches [0, (b*hkv)/2)，
    // Die 1 (核 cores/2..cores-1) 亲和后半段 KV heads/batches [(b*hkv)/2, b*hkv)。
    // 这保证了 100% 的 KV reuse 命中对应 Die 的 64MB 本地 L2 Cache，彻底避免跨 Die 互联总线颠簸。
    uint32_t die_num = ATLAS_950_DIE_NUM;
    uint32_t active_cores = tiling_out.core_num_used;
    const bool is_b_hkv_even = ((static_cast<uint64_t>(b) * hkv) % 2ULL == 0ULL);

    bool enable_die = false;
    if (has_tuned) {
        enable_die = tuned_die_grouped && (split_kv_num == 1) &&
                     (active_cores >= die_num) && ((active_cores % die_num) == 0) &&
                     is_b_hkv_even;
    } else {
        enable_die = (split_kv_num == 1) &&
                     (active_cores >= die_num) && ((active_cores % die_num) == 0) &&
                     (total_tasks >= die_num) && ((total_tasks % die_num) == 0) &&
                     is_b_hkv_even;
    }

    tiling_out.enable_die_affinity = enable_die ? 1 : 0;
    tiling_out.die_num = die_num;
    tiling_out.cores_per_die = (active_cores >= die_num) ? (active_cores / die_num) : active_cores;
    tiling_out.tasks_per_die = ::CeilDiv(total_tasks, die_num);

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
    // 512B 对齐统一走官方 CATLASS 对齐组件（::RoundUp<512u>，constexpr）
    uint64_t current_ws_offset = 0;
    auto align512 = [](uint64_t size) -> uint64_t {
        return ::RoundUp<512u>(size);
    };

    // (1) LSE 临时显存空间 (Split-KV 跨核归并；smooth_k 的 lse_correction 由桥接层
    //     经 lseCorrection GM 入参直传，不经 workspace)
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
