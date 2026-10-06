#ifndef SAGE_ATTN_HOST_TILING_H
#define SAGE_ATTN_HOST_TILING_H
#pragma once

/**
 * \file sage_attn_host_tiling.h
 * \brief 闭式 Host Tiling 计算核心（inline 头文件实现）
 *
 * ComputeHostTiling 为纯函数（仅依赖 tiling.h / catlass / quant_host 头），
 * 被 op_host/sage_attention2_tiling.cpp（gert TilingImpl）与宿主单测
 * tests/test_aclnn_host.cpp 包含，保证 TilingImpl 序列化位元与单测锁值一致；
 * 交付态两段式 aclnn 入口（op_api 层）的 Workspace 预算同式收敛于此。
 * 以 inline 头文件形式组织，使 op_host 库无需链接 op_api 层目标即可获得定义。
 */

#include "sage_attn_proto.h"
#include "autotune_tiling_bank.h"
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/catlass/sage_tile.hpp"
#include "../sageattn_fp8/catlass/sage_resource.hpp"
#include "../quant/quant_host.h"

#include <cassert>
#include <cmath>
#include <cstdlib>

// aclnn 状态码词汇表（ComputeHostTiling 返回值；原 sage_attn_proto.h 承载，
// 移植态两段式删除后归本头）：CANN 头可用时复用真实定义，宿主单测等无 CANN
// include 环境使用取值对齐 CANN opdev/op_errno.h 的本地桩。
#if defined(__has_include)
#if __has_include("aclnn/aclnn_base.h")
#include "aclnn/aclnn_base.h"
#if __has_include("opdev/op_errno.h")
#include "opdev/op_errno.h"
#endif
#define SAGE_HAS_CANN_ACLNN_BASE 1
#endif
#endif

#ifndef SAGE_HAS_CANN_ACLNN_BASE
using aclnnStatus = int32_t;
constexpr aclnnStatus ACLNN_SUCCESS = 0;
constexpr aclnnStatus ACLNN_ERR_PARAM_INVALID = 161002;
#endif

#if (defined(__linux__) || defined(__unix__)) && !defined(__NPU_DEVICE__)
#include <dlfcn.h>
#endif

namespace sage_attn {

/**
 * 获取硬件可用 AI Core 核心数量
 * 动态支持 32 核 (Atlas 950PR 满配) 与 28 核 (减配版)：
 * 优先级 1：SAGEATTN_NUM_CORES 环境变量显式覆盖（自动化测试与特定拓扑模拟）；
 * 优先级 2：动态查询 ACL 运行时属性（对齐 CANN 官方最佳实践，避免硬编码，且无编译期强符号链接依赖）；
 * 优先级 3：默认 Atlas 950PR 标配 32 个 AI Core。
 */
[[nodiscard]] inline uint32_t GetHardwareCoreNum()
{
    const char *env_cores = std::getenv("SAGEATTN_NUM_CORES");
    if (env_cores != nullptr) {
        int c = std::atoi(env_cores);
        if (c > 0 && c <= 64) {
            return static_cast<uint32_t>(c);
        }
    }

#if (defined(__linux__) || defined(__unix__)) && !defined(__NPU_DEVICE__)
    auto getDevice = reinterpret_cast<int (*)(int32_t *)>(dlsym(RTLD_DEFAULT, "aclrtGetDevice"));
    auto getInfo = reinterpret_cast<int (*)(uint32_t, int, int64_t *)>(dlsym(RTLD_DEFAULT, "aclrtGetDeviceInfo"));
    if (getDevice != nullptr && getInfo != nullptr) {
        int32_t device = 0;
        int64_t aic = 0;
        // 0 == ACL_SUCCESS, 102 == ACL_DEV_ATTR_CUBE_CORE_NUM, 103 == ACL_DEV_ATTR_VECTOR_CORE_NUM
        if (getDevice(&device) == 0 && getInfo(static_cast<uint32_t>(device), 102, &aic) == 0) {
            if (aic > 0 && aic <= 64) {
                int64_t aiv = 0;
                // 拓扑安全校验：若成功探测到 AIV 核心数，核验其是否满足昇腾 950 1C:2V 拓扑（aiv >= 2 * aic）
                if (getInfo(static_cast<uint32_t>(device), 103, &aiv) == 0 && aiv > 0) {
                    if (aiv < 2 * aic) {
                        return ATLAS_950_AIC_NUM;
                    }
                }
                return static_cast<uint32_t>(aic);
            }
        }
    }
#endif

    // 默认 Atlas 950PR 标配 32 个 AI Core
    return ATLAS_950_AIC_NUM;
}

/**
 * \brief 官方 CATLASS 49 FAI 移植：GetCutBlockNums 等差数列求和扣除因果全零块
 * 用于 Host 端计算因果注意力下的实际有效计算块数与精确负载均衡。
 */
[[nodiscard]] constexpr inline int64_t GetCutBlockNums(
    int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner, int64_t sOuter, int64_t token)
{
    if (sInner == 0 || sOuter == 0) {
        return 0;
    }
    int64_t blockNums = 0;
    int64_t blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
    int64_t outDivIn = sOuter > sInner ? sOuter / sInner : 1;
    int64_t InDivOut = sInner > sOuter ? sInner / sOuter : 1;
    int64_t tolerance = 0;
    int64_t smallSize = 0;
    if (outDivIn >= 1) {
        tolerance = outDivIn;
        smallSize = sInner;
    } else {
        tolerance = InDivOut;
        smallSize = sOuter;
    }
    int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
    int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
    int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength - blockToken) / smallSize - tolerance;
    int64_t tmpInnerCutBlockNums =
        (innerCutBlockNums > 0) ?
            (innerCutBlockNums % tolerance + innerCutBlockNums) * (innerCutBlockNums / tolerance + 1) / 2 :
            0;
    blockNums += tmpInnerCutBlockNums;
    int64_t tmpInnerCutBlockLeftNums =
        (innerCutBlockLeftNums > 0) ?
            (innerCutBlockLeftNums % tolerance + innerCutBlockLeftNums) * (innerCutBlockLeftNums / tolerance + 1) / 2 :
            0;
    blockNums -= tmpInnerCutBlockLeftNums;
    int64_t tmpInnerCutBlockDownNums =
        (innerCutBlockDownNums > 0) ?
            (innerCutBlockDownNums % tolerance + innerCutBlockDownNums) * (innerCutBlockDownNums / tolerance + 1) / 2 :
            0;
    blockNums -= tmpInnerCutBlockDownNums;
    return blockNums;
}

[[nodiscard]] constexpr inline int64_t GetCalcBlockNumsOneHead(
    int64_t actualSeqLength, int64_t actualSeqLengthKV, int64_t sOuterSize, int64_t sInnerSize, bool isCausal)
{
    int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
    int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize;
    int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
    if (!isCausal) {
        return toCalcBlockNums;
    }
    int64_t blockSeqLengthKV = innerBlockNums * sInnerSize;
    int64_t blockSeqLength = outerBlockNums * sOuterSize;
    toCalcBlockNums -= GetCutBlockNums(
        blockSeqLengthKV, blockSeqLength, sInnerSize, sOuterSize, 0);
    return toCalcBlockNums;
}

/**
 * 闭式超轻量微秒级 Host Tiling 计算核心
 * 核心设计指标：
 * 1. 扁平代数闭式计算，无复杂动态循环与大表搜索，平均启动开销 < 2 微秒
 * 2. 彻底打赢对 FIA 3.7 万行无 cache 的 Host Launch 时延之战
 * 3. 动态支持 32 核与 28 核
 * 4. 强制 64 位整型提升计算显存空间与地址偏移
 *
 * inline 唯一定义（本头被 op_host/sage_attention2_tiling.cpp 与
 * tests/test_aclnn_host.cpp 同时包含，且二者在独立构建目标 sageattention2_host
 * 中共链——无 inline 会在共享库链接期报 duplicate symbol，R6 审查实测捕获）；
 * inline 弱符号即"避免 op_host 库反向依赖 op_api 层目标"的落地形式。
 */
[[nodiscard]] inline aclnnStatus ComputeHostTiling(
    uint32_t b,
    uint32_t sq,
    uint32_t sk,
    uint32_t hq,
    uint32_t hkv,
    int dtype_mode,      // 输入位型分派档位（编码唯一定义 op_kernel/sage_attention2_template_tiling_key.h），写 TilingData.input_dtype
    uint32_t d_og,
    double smScale,
    int64_t qkQuantGran,
    int64_t pvAccumDtype,
    bool isCausal,
    int64_t tensorLayout,
    bool smoothK,
    bool smoothV,
    bool hasLse,
    SageAttention2TilingData &tiling_out,
    uint64_t &workspace_bytes_out,
    uint32_t hw_cores_in = 0)
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
    if (qkQuantGran != static_cast<int64_t>(QkQuantGran::QUANT_GRAN_PER_WARP) &&
        qkQuantGran != static_cast<int64_t>(QkQuantGran::QUANT_GRAN_PER_THREAD)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (pvAccumDtype < static_cast<int64_t>(PvAccumDtype::PV_ACCUM_FP32) ||
        pvAccumDtype > static_cast<int64_t>(PvAccumDtype::PV_ACCUM_FP32_FP16)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // tensor_layout 编码域校验：仅接受冻结编码 NHD(0)/HND(1)，非法数值显式报错
    // 不静默归一（与 canonical op_api 的"非法布局报错不降级"口径及同函数
    // gran/accum 严格校验一致；GE 字符串映射层经 *CodeFromString 恒产 0/1，不受影响）
    if (tensorLayout != static_cast<int64_t>(TensorLayout::LAYOUT_NHD) &&
        tensorLayout != static_cast<int64_t>(TensorLayout::LAYOUT_HND)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 4. 清空结构体 (160 字节，32 字节对齐)
    tiling_out = {};

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
    tiling_out.tensor_layout = static_cast<uint32_t>(
        (tensorLayout == static_cast<int64_t>(TensorLayout::LAYOUT_HND))
            ? TensorLayout::LAYOUT_HND : TensorLayout::LAYOUT_NHD);
    tiling_out.qk_quant_gran = static_cast<uint32_t>(qkQuantGran);
    tiling_out.pv_accum_dtype = static_cast<uint32_t>(pvAccumDtype);
    (void)smoothK;
    tiling_out.smooth_k = 0; // 标准 7 参收敛无 keyMean 输入，恒置 0
    // 对齐设计文档：若非全精度 fp32 档，则忽略 smooth_v
    tiling_out.smooth_v = (smoothV && pvAccumDtype == static_cast<int64_t>(PvAccumDtype::PV_ACCUM_FP32)) ? 1 : 0;
    tiling_out.has_lse = hasLse ? 1u : 0u;

    // 7. 预计算与常量折叠数值字段 (8 字节)
    // sm_scale 冻结语义：默认值 1/sqrt(d) 由上层桥接在
    // Python sm_scale=None 时折叠后传入；gert 路径 0/缺省已由 TilingImpl 折叠、
    // 负值已被上游拒绝，本检查覆盖直连调用方（tests/test_aclnn_host.cpp）
    // 与漏网的非有限值
    if (!(smScale > 0.0) || !std::isfinite(smScale)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    float base_sm_scale = static_cast<float>(smScale);
    tiling_out.sm_scale_log2e = base_sm_scale * sage_attn::LOG2E;
    tiling_out.v_scale_max = quant::GetVScaleMax(static_cast<uint32_t>(pvAccumDtype));
    // （原 s_fp8_offset TilingData 字段已删除：448· 放大恒经乘法启用、LSE 出口恒以
    //  S_FP8_OFFSET 常量补扣，唯一定义 sageattn_fp8/ascendc/tiling.h。）
    // 输入位型分派：fp16/bf16 三方（def 注册、op_api、核内读写）一致
    tiling_out.input_dtype = static_cast<uint8_t>(dtype_mode);

    // 8. 闭式分块计算 (分块字段，12 字节) 与 Auto-Tuned Tiling 调优知识库查询
    // 优先查询 Ascend 950PR 经验调优知识库 (autotune::QueryTunedTiling)：
    // 命中调优用例 (P-01 ~ P-07) 时应用实测最优超参 (m_per_core, block_n, k_step,
    // use_die_grouped)；未命中或动态 shape 时平滑回退至健壮的闭式推导。
    const autotune::TunedTilingEntry *tuned = nullptr;
#if defined(SAGE_ATTN_AUTOTUNE_HOOKS)
    // 调优会话专用：按用例 ID 直查知识库（正式构建无此运行期入口，见上方宏门控注释）
    const char *env_tune_case = std::getenv("SAGEATTN_TUNE_CASE");
    if (env_tune_case != nullptr) {
        tuned = autotune::QueryTunedTiling(env_tune_case);
    }
#endif
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

    if (tuned != nullptr) {
        has_tuned = true;
        m_per_core = tuned->m_per_core;
        block_n = tuned->block_n;
        k_step = tuned->k_step;
        tuned_die_grouped = tuned->use_die_grouped;
    }

    // Auto-Tune 调优会话钩子（SAGE_ATTN_AUTOTUNE_HOOKS 编译期门控）：
    // 仅调优构建（build_opp.sh SAGE_AUTOTUNE_HOOKS=1 编入该宏）
    // 定义该宏时编入 SAGEATTN_FORCE_MPERCORE/_BLOCKN/_KSTEP/_DIE/_SPLITKV 与
    // SAGEATTN_TUNE_CASE 运行期覆盖入口，供网格实测调优会话逐配置切换；
    // 正式发布构建（缺省不定义宏）该段整体不参与编译——零 autotune 代码、
    // 零 getenv 运行时判断，发射路径只剩编译期 constexpr 调优知识库查询。
    // 覆盖取值域与 kernel/UB 台账一致（m 32/64/128、n 64/128——A7 dot32 区
    // static_assert 封顶 128、k 8/16/32，越界显式拒绝）；die 仅 0/1、split 仅
    // ≥1 语义化生效，其余解析值视同未设置（不覆盖、不报错）；
    // 生效覆盖值与库查询值同经 InspectTilingPlan 足迹 fail-fast，不静默降级。
#if defined(SAGE_ATTN_AUTOTUNE_HOOKS)
    auto env_forced = [](const char *name) -> int64_t {
        const char *raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0') {
            return -1;
        }
        char *end = nullptr;
        const long long parsed = std::strtoll(raw, &end, 10);
        if (end == raw || parsed < 0) {
            return -1;
        }
        return static_cast<int64_t>(parsed);
    };

    const int64_t forced_m = env_forced("SAGEATTN_FORCE_MPERCORE");
    if (forced_m >= 0) {
        if (forced_m != 32 && forced_m != 64 && forced_m != 128) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        m_per_core = static_cast<uint32_t>(forced_m);
        has_tuned = true;
    }
    const int64_t forced_n = env_forced("SAGEATTN_FORCE_BLOCKN");
    if (forced_n >= 0) {
        if (forced_n != 64 && forced_n != 128) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        block_n = static_cast<uint32_t>(forced_n);
        has_tuned = true;
    }
    const int64_t forced_k = env_forced("SAGEATTN_FORCE_KSTEP");
    if (forced_k >= 0) {
        if (forced_k != 8 && forced_k != 16 && forced_k != 32) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        k_step = static_cast<uint32_t>(forced_k);
        has_tuned = true;
    }
    bool die_forced = false;
    const int64_t forced_die = env_forced("SAGEATTN_FORCE_DIE");
    if (forced_die == 0 || forced_die == 1) {
        tuned_die_grouped = (forced_die == 1);
        die_forced = true;
    }
#else
    constexpr bool die_forced = false;
#endif

    tiling_out.m_per_core = m_per_core;
    tiling_out.block_n = block_n;
    tiling_out.k_step = k_step;

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
    uint32_t hw_cores = (hw_cores_in > 0) ? hw_cores_in : GetHardwareCoreNum();
    uint32_t q_blk_num = ::CeilDiv(sq, m_per_core);
    uint32_t kv_blk_num = ::CeilDiv(sk, block_n);

    tiling_out.q_blk_num = q_blk_num;
    tiling_out.kv_blk_num = kv_blk_num;

    const int64_t effective_blocks_per_head = GetCalcBlockNumsOneHead(sq, sk, m_per_core, block_n, isCausal);
    if (sq > 0 && sk > 0) {
        assert(effective_blocks_per_head > 0);
    }
    (void)effective_blocks_per_head;

    // 任务切分单位为 (Batch, Hkv, Q_blk)；64 位乘法防回绕
    const uint64_t total_tasks_64 = static_cast<uint64_t>(b) * hkv * q_blk_num;
    if (total_tasks_64 > UINT32_MAX) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint32_t total_tasks = static_cast<uint32_t>(total_tasks_64);
    uint32_t split_kv_num = 1;

    // 尾核利用优化：当任务总数不足以填满硬件核心时，自动沿 KV 序列维实施 Split-KV
    // 注意：因果掩码下存在下三角不等长依赖，内核未接入跨核 Merge 算子前严禁自动提升 Split-KV
    if (!isCausal && total_tasks < hw_cores && kv_blk_num > 1) {
        uint32_t factor = ::CeilDiv(hw_cores, total_tasks);
        split_kv_num = (factor < kv_blk_num) ? factor : kv_blk_num;
        if (split_kv_num > 8) {
            split_kv_num = 8;
        }
    }
#if defined(SAGE_ATTN_AUTOTUNE_HOOKS)
    const int64_t forced_split = env_forced("SAGEATTN_FORCE_SPLITKV");
    if (forced_split >= 1) {
        if (forced_split > 8 || forced_split > static_cast<int64_t>(kv_blk_num)) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (isCausal && forced_split > 1) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        split_kv_num = static_cast<uint32_t>(forced_split);
    }
#endif
    tiling_out.split_kv_num = split_kv_num;
    tiling_out.is_merge = 0;

    uint32_t scheduled_tasks = total_tasks * split_kv_num;
    tiling_out.core_num_used = (scheduled_tasks < hw_cores) ? scheduled_tasks : hw_cores;

    // 10. Die 亲和性多核调度分核计算（STARS 2.0 双 Die / 8-Group 拓扑，M6 硬件特性 4）
    // 接口边界（考证证据链唯一定义点 ascendc/tiling.h STARS_MAX_GROUP_NUM 组注释）：
    // 本 CANN 版本 host tiling 层无 STARS Group 直控公开 API，按仓内既有惯例以
    // TilingData 第 7 段四字段编码 2 个物理 Die Group 配置（enable_die_affinity=1 即
    // 下发 Die 亲和调度），核内按 GetBlockIdx 经 DieTask/DieTaskEnd 解码，运行时
    // STARS 调度器按 blockDim 波次将 Batch/Head 分块下沉至各 Die 硬件队列纳秒级发射
    // （HSCB 为纯硬件总线，无软件编程面）。
    // 闭式映射与判据（数学论证）：
    // - 核域任务总数 T4 = b·hkv·gqa_chunk·q_blk（与 kernel DecodeTask 同式），
    //   DieTask/DieTaskEnd 双射闭合的必要条件是 T4 为偶数（奇数总数丢尾任务）；
    // - 两分支共同硬前置 b·hkv 偶：T3 = b·hkv·q_blk 随之恒偶，T4 = T3·gqa_chunk
    //   恒偶，核内映射无尾任务遗漏（安全性）；非调优域另加 T3 偶与 T3 ≥ 2，在
    //   b·hkv 偶下属显式防御性冗余（恒真蕴含）；
    // - (b,h) 块整对齐定理：b·hkv 偶 ⇔ Die 边界 T4/2 = (b·hkv/2)·(gqa_chunk·q_blk)
    //   为 (b,h) 基块粒度 gqa_chunk·q_blk 的整数倍 ⇔ 每个 Batch/Head 基块整块归属
    //   单一 Die（其 KV 流 100% 命中本 Die 本地 L2，零跨 Die 重取）。
    // 硬件拓扑：Atlas 950PR 标配 2 个计算 Die（2 Die × 4 STARS Group × 4 核），每 Die
    // 独立 64MB L2；Die 0 (核 0..cores/2-1) 亲和前半段任务，Die 1 (核 cores/2..cores-1)
    // 亲和后半段任务，两 Die 间零任务交叉、零 KV 跨 Die 重取。
    uint32_t die_num = ATLAS_950_DIE_NUM;
    uint32_t active_cores = tiling_out.core_num_used;
    const bool is_b_hkv_even = ((static_cast<uint64_t>(b) * hkv) % 2ULL == 0ULL);

    bool enable_die = false;
    if (has_tuned || die_forced) {
        // 库命中 / 调优覆盖态：die 意愿取库值或 SAGEATTN_FORCE_DIE，硬性前置
        // （无 Split-KV、核心数偶数且 ≥ Die 数、b*hkv 偶数）仍然全部保留
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

    // (1) A7 V0→V1 scale 暂存条带（STAGE_CORE_STRIDE × core_num_used；
    //     kernel 侧以 core_idx × STAGE_CORE_STRIDE 从 workspace 起始处定址）
    current_ws_offset += align512(static_cast<uint64_t>(tiling_out.core_num_used) *
                                  sage_attn::STAGE_CORE_STRIDE);

    // (2) LSE 临时显存空间 (Split-KV 跨核归并)
    if (split_kv_num > 1 || hasLse) {
        tiling_out.workspace_lse_offset = current_ws_offset;
        int64_t lse_elems = static_cast<int64_t>(b) * hq * sq * split_kv_num;
        uint64_t lse_bytes = align512(static_cast<uint64_t>(lse_elems) * sizeof(float));
        current_ws_offset += lse_bytes;
    } else {
        tiling_out.workspace_lse_offset = 0;
    }

    // (3) Split-KV partial outputs 临时累加显存
    if (split_kv_num > 1) {
        int64_t partial_o_elems = static_cast<int64_t>(b) * hq * sq * d_pad * split_kv_num;
        uint64_t partial_o_bytes = align512(static_cast<uint64_t>(partial_o_elems) * sizeof(float));
        current_ws_offset += partial_o_bytes;
    }

    // (4) fuse_v_mean 临时均值空间
    if (tiling_out.IsSmoothV()) {
        tiling_out.workspace_vm_offset = current_ws_offset;
        int64_t vm_elems = static_cast<int64_t>(b) * hkv * d_pad;
        uint64_t vm_bytes = align512(static_cast<uint64_t>(vm_elems) * sizeof(float));
        current_ws_offset += vm_bytes;
    } else {
        tiling_out.workspace_vm_offset = 0;
    }

    // (5) Step 1: 自研 KV 前置单次流式量化 workspace 空间
    // 预留各核心的局部预量化张量缓冲与 Scale 空间 (按 core_num_used 隔离分配)
    const KvPreQuantWorkspaceOffsets core_ws = ComputeKvPreQuantCoreOffsets(sk, d_pad);
    current_ws_offset += align512(core_ws.total_core_bytes * tiling_out.core_num_used);

    workspace_bytes_out = current_ws_offset;
    return ACLNN_SUCCESS;
}

} // namespace sage_attn

#endif // SAGE_ATTN_HOST_TILING_H
