#pragma once

// =============================================================================
// sm100_mqa_logits_with_topk.cuh
// =============================================================================
// DeepGEMM SM100 MQA Logits + 融合 TopK 内核
//
// 整体两阶段结构:
//   Phase 1（原 DeepGEMM 逻辑，未改动核心）:
//     Q @ K^T (UMMA/Tensor Core) → relu → ×weights → sum over heads
//     → 写 logits_buffer (bf16, GMEM)
//   Phase 2（新增 TopK）:
//     8-warp 分段 → 每 warp 局部 radix TopK-512 → 4096 候选合并
//     → 分布式 radix rounds 求 kth → 分布式 selection 写 sparse_indices
//
// 12-warp 分工（384 线程 = 128 specialized + 256 math）:
//   warp 0~7:  math warpgroups —— UMMA 计算 + Phase 2 TopK
//   warp 8:    TMA Q 加载
//   warp 9:    TMA KV 加载 + barrier 初始化
//   warp 10:   MMA issue（UTCCP + UMMA 发射）
//   warp 11:   idle（仅做寄存器重分配）
// =============================================================================

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/mqa_logits.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>

// For CUDART_INF_F used in TopK
#include <math_constants.h>

namespace deep_gemm {

// 环形流水级计数器: 避免 % 运算（ptxas 对 TMEM 路径的取模优化很差）
// 维护 (stage_idx, phase) 对: 绕圈时 stage_idx 归零、phase 翻转
template <uint32_t kNumStages>
struct RingPipeline {
    uint32_t stage_idx = 0, phase = 0;

    CUTLASS_DEVICE cute::tuple<uint32_t, uint32_t> advance(const uint32_t& step = 1) {
        const uint32_t current_stage_idx = stage_idx, current_phase = phase;
        stage_idx += step;
        if (stage_idx >= kNumStages) {
            stage_idx -= kNumStages;
            phase ^= 1;
        }
        return {current_stage_idx, current_phase};
    }
};

// 把运行时"有效 token 数"折叠成编译期常量 cute::Int<k>，
// 使 token 循环可全展开（编译器需要常量界来 unroll）
template <uint32_t kBlockQ, uint32_t kCandidate = kBlockQ, typename Fn>
CUTLASS_DEVICE void dispatch_num_block_tokens(const uint32_t& num_block_tokens, Fn&& fn) {
    if constexpr (kCandidate <= 1) {
        fn(cute::Int<1>{});
    } else if (num_block_tokens >= kCandidate) {
        fn(cute::Int<kCandidate>{});
    } else {
        dispatch_num_block_tokens<kBlockQ, kCandidate - 1>(num_block_tokens, static_cast<Fn&&>(fn));
    }
}

// =============================================================================
// warp_bucket_topk —— warp 级 8-bit radix-select TopK
// =============================================================================
// 算法（4 轮 × 8bit）:
//   每轮: 对"前缀匹配"的元素建 256 桶直方图 → 前缀和定位 cutoff 桶
//         → 高于 cutoff 的计入 definite、cutoff 桶内的继续下一轮
//   4 轮后得到 kth（第 K 大值的精确位模式）
// Final: u > kth 直接写出（definite），u == kth 补足（boundary），其余丢弃
//
// 模板参数:
//   kSparseCount : TopK 的 K
//   kCached      : true=logits 在 SMEM 缓存里（ld_shared 读）；false=GMEM bf16 兜底
//   kUseMap      : true=输出索引取 imap[i]（合并阶段的候选重映射）；false=out_base+i
// 参数:
//   logits  : 输入缓冲（kCached 时为 float 缓存，否则 bf16 行）
//   out_base: 输出索引的基址（真实 k_start）
//   k_len   : 元素个数（logits[0 .. k_len)）
//   smem_vals: SMEM 256 桶直方图工作区
//   smem_idxs: SMEM boundary 暂存区（32 lane × kPerLane）
//   kth_slot: 非空时只算 kth 写回并返回（供分布式 stage3/4 拆分使用）
// =============================================================================
template <uint32_t kSparseCount, bool kCached, bool kUseMap = false>
CUTLASS_DEVICE void warp_bucket_topk(
    const void* __restrict__ logits,
    // local buffer, reads logits[i], i in [0, k_len)
    const uint32_t out_base,            // output index base (real k_start)
    const uint32_t k_len,               // local buffer length
    int32_t* __restrict__ out_indices,
    float* __restrict__ smem_vals,
    int32_t* __restrict__ smem_idxs,
    const int32_t* __restrict__ imap,   // when kUseMap: output = imap[i] instead of out_base+i
    const uint32_t lane_idx,
    uint32_t* __restrict__ kth_slot = nullptr   // when non-null: only compute kth, skip output
) {
    // 输出索引映射器: kUseMap 时输出重映射表的第 i 项（候选合并场景），
    // 否则输出"基址 + 位置"（常规场景）
    auto get_idx = [&](uint32_t i) -> int32_t {
        if constexpr (kUseMap) return imap[i];
        else return static_cast<int32_t>(out_base + i);
    };

    // 边界 1: 空区间 → 全填 -1
    if (k_len == 0) {
        if (lane_idx < kSparseCount) out_indices[lane_idx] = -1;
        __syncwarp();
        return;
    }

    // 边界 2: 元素数 ≤ K → 全部入选，直接按位置写索引（无需排序）
    if (k_len <= kSparseCount) {
        for (uint32_t i = lane_idx; i < kSparseCount; i += 32)
            out_indices[i] = (i < k_len) ? get_idx(i) : -1;
        __syncwarp();
        return;
    }

    // float → 可比较 uint32 的单调映射:
    //   正数: 翻转符号位（正数之间 uint 越大 = float 越大）
    //   负数: 整体取反（更负 → uint 更小）
    auto to_sortable = [](float f) -> uint32_t {
        uint32_t u = __float_as_uint(f);
        return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
    };

    // 元素读取器（按 kCached 编译期选择视图，避免 generic load）
    const uint32_t* logits_u = reinterpret_cast<const uint32_t*>(logits);
    const nv_bfloat16* logits_b = reinterpret_cast<const nv_bfloat16*>(logits);
    auto load_u32 = [&](uint32_t i) -> uint32_t {
        if constexpr (kCached) return ptx::ld_shared(logits_u + i);
        else return __float_as_uint(__bfloat162float(logits_b[i]));
    };

    // SMEM: 256-bucket histogram
    uint32_t* histogram = reinterpret_cast<uint32_t*>(smem_vals);

    // radix-select 状态:
    //   kth_value     : 已确定的 kth 前缀（每轮拼接一个字节）
    //   definite_count: 已锁定"必进 TopK"的元素数（高于 cutoff 桶的）
    //   remaining     : 还需要从存活集合里选的个数
    uint32_t kth_value = 0;
    uint32_t definite_count = 0;
    uint32_t remaining = kSparseCount;

    // === 2 轮 8-bit radix select（字节从高到低） ===
    // logits 为 bf16 提升值: 低 16 位恒为 0，只有字节 3/2 携带信息，
    // 后 2 轮（字节 1/0）恒空转，故只跑 2 轮——结果与 4 轮逐位一致。
    for (int round = 0; round < 2 && remaining > 0; ++round) {
        const int shift = (3 - round) * 8;                 // 本轮看第几个字节
        const uint32_t prefix_mask = (round == 0) ? 0u : (0xFFFFFFFFu << (32 - round * 8));
        const uint32_t prefix_val = kth_value & prefix_mask; // 前几轮已确定的前缀

        // --- 步骤 1: 清 256 桶直方图（32 lane 各清 8 个） ---
        for (uint32_t i = lane_idx; i < 256; i += 32)
            histogram[i] = 0;
        __syncwarp();

        // --- 步骤 2: 建直方图 ---
        // 只统计"与前几轮 kth 前缀完全匹配"的存活元素；
        // 按本轮字节落入 256 桶之一，atomicAdd 计数
        // （round 0 前缀为空 → 全部元素参与，热点桶可能产生原子争用）
        #pragma unroll 4
        for (uint32_t i = lane_idx; i < k_len; i += 32) {
            uint32_t u = to_sortable(__uint_as_float(load_u32(i)));
            if ((u & prefix_mask) == prefix_val) {
                uint32_t bucket = (u >> shift) & 0xFF;
                atomicAdd(&histogram[bucket], 1u);
            }
        }
        __syncwarp();

        // --- 步骤 3: 定位 cutoff 桶（第 K 大的元素所在桶） ---
        // 布局: lane i 负责桶 [i*8, i*8+7]，先各自读出自己的 8 个计数
        uint32_t my_counts[8];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            my_counts[i] = histogram[lane_idx * 8 + i];

        uint32_t my_sum = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) my_sum += my_counts[i];

        // warp 内 inclusive 前缀和（shfl_up 蝴蝶算法）:
        // lane i 得到"桶 0..(i*8+7)"的累计数
        uint32_t incl = my_sum;
        for (int s = 1; s < 32; s <<= 1) {
            uint32_t t = __shfl_up_sync(0xFFFFFFFF, incl, s);
            if (lane_idx >= s) incl += t;
        }
        const uint32_t total = __shfl_sync(0xFFFFFFFF, incl, 31);   // 存活总数 = lane31 的值

        // 高于本 lane 所有桶的元素数
        const uint32_t count_above_me = total - incl;

        // 在自己的 8 个桶内从高往低扫，找满足条件的 cutoff:
        //   count_above(严格高于该桶) < remaining 且 +本桶 ≥ remaining
        //   即"第 remaining 大的元素就在这个桶里"
        uint32_t prefix_in_lane = 0;
        int32_t my_cutoff = -1;
        uint32_t my_count_above_cutoff = 0;

        #pragma unroll
        for (int i = 7; i >= 0; i--) {
            uint32_t count_above = count_above_me + prefix_in_lane;
            uint32_t count_ge = count_above + my_counts[i];
            if (count_above < remaining && count_ge >= remaining) {
                my_cutoff = static_cast<int32_t>(lane_idx * 8 + i);
                my_count_above_cutoff = count_above;
            }
            prefix_in_lane += my_counts[i];
        }

        // 32 个 lane 中只有 1 个找到 cutoff，ballot 选出它，shfl 广播其值
        uint32_t cutoff_mask = __ballot_sync(0xFFFFFFFF, my_cutoff >= 0);
        if (cutoff_mask == 0) break;
        uint32_t cutoff_lane = __ffs(cutoff_mask) - 1;
        uint32_t cutoff_bucket = static_cast<uint32_t>(
            __shfl_sync(0xFFFFFFFF, static_cast<uint32_t>(my_cutoff), cutoff_lane));
        uint32_t count_above_cutoff = __shfl_sync(0xFFFFFFFF, my_count_above_cutoff, cutoff_lane);

        // 拼接 kth 字节、累计 definite、更新 remaining
        kth_value |= (cutoff_bucket << shift);
        definite_count += count_above_cutoff;
        remaining = kSparseCount - definite_count;
    }

    // kth_slot 模式（分布式 stage3/4 使用）: 只导出 kth，selection 由调用方另行处理
    if (kth_slot != nullptr) {
        if (lane_idx == 0) *kth_slot = kth_value;
        __syncwarp();
        return;
    }

    // === Final: 按 kth 做一次分类写出 ===
    // definite (u > kth) 直接进输出；boundary (u == kth) 暂存后补足
    const uint32_t kth_full = kth_value;
    constexpr uint32_t kPerLane = (kSparseCount + 31) / 32;   // 每 lane 的 boundary 暂存上限

    // --- 单遍扫描: 同时数 definite、暂存 boundary ---
    uint32_t my_def = 0;
    uint32_t my_collected = 0;
    #pragma unroll 4
    for (uint32_t i = lane_idx; i < k_len; i += 32) {
        uint32_t u = to_sortable(__uint_as_float(load_u32(i)));
        if (u > kth_full) {
            my_def++;
        } else if (u == kth_full && my_collected < kPerLane) {
            smem_idxs[lane_idx * kPerLane + my_collected] = get_idx(i);
            my_collected++;
        }
    }

    // --- definite 的 warp 内 exclusive 前缀和 → 各 lane 写出起始偏移 ---
    uint32_t def_prefix = my_def;
    for (int s = 1; s < 32; s <<= 1) {
        uint32_t t = __shfl_up_sync(0xFFFFFFFF, def_prefix, s);
        if (lane_idx >= s) def_prefix += t;
    }
    const uint32_t def_exclusive = def_prefix - my_def;
    const uint32_t total_definite = __shfl_sync(0xFFFFFFFF, def_prefix, 31);

    // --- 第二遍扫描: 按偏移写出 definite 索引 ---
    uint32_t def_pos = def_exclusive;
    #pragma unroll 4
    for (uint32_t i = lane_idx; i < k_len; i += 32) {
        uint32_t u = to_sortable(__uint_as_float(load_u32(i)));
        if (u > kth_full) {
            out_indices[def_pos] = get_idx(i);
            def_pos++;
        }
    }
    __syncwarp();

    uint32_t col_prefix = my_collected;
    for (int s = 1; s < 32; s <<= 1) {
        uint32_t t = __shfl_up_sync(0xFFFFFFFF, col_prefix, s);
        if (lane_idx >= s) col_prefix += t;
    }
    const uint32_t col_exclusive = col_prefix - my_collected;
    const uint32_t total_collected = __shfl_sync(0xFFFFFFFF, col_prefix, 31);

    // --- boundary 从暂存区补足到 definite 之后（上限 remaining_slots） ---
    const uint32_t remaining_slots = kSparseCount - total_definite;
    for (uint32_t i = 0; i < my_collected; ++i) {
        uint32_t out_pos = total_definite + col_exclusive + i;
        if (out_pos < kSparseCount)
            out_indices[out_pos] = smem_idxs[lane_idx * kPerLane + i];
    }

    // --- 尾部空位填 -1（padding 语义） ---
    if (lane_idx == 0) {
        const uint32_t filled = total_definite +
            (total_collected < remaining_slots ? total_collected : remaining_slots);
        for (uint32_t i = filled; i < kSparseCount; ++i)
            out_indices[i] = -1;
    }
    __syncwarp();
}

// =============================================================================
// core_impl —— 两阶段主函数
// 参数说明:
//   logits_stride : 每 Q token 一行的 logits 宽度（对齐后）
//   logits_buffer : Phase 1 输出（bf16, [num_q_tokens, logits_stride]）
//   sparse_indices: Phase 2 输出（int32, [num_q_tokens, kSparseCount]）
//   5 个 tensor_map: TMA 描述符（Q/SF_Q/KV/SF_KV/weights）
//   make_scheduler: paged 或 non-paged 的调度器工厂 lambda
// =============================================================================
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF, bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kSparseCount,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t, typename MakeScheduler,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_DEVICE void sm100_mqa_logits_with_topk_core_impl(
    const uint32_t logits_stride,
    nv_bfloat16* __restrict__ logits_buffer,            // Phase 1 output (temp, bf16)
    int32_t* __restrict__ sparse_indices,              // Phase 2 output (final)
    const cute::TmaDescriptor& tensor_map_q,
    const cute::TmaDescriptor& tensor_map_sf_q,
    const cute::TmaDescriptor& tensor_map_kv,
    const cute::TmaDescriptor& tensor_map_sf_kv,
    const cute::TmaDescriptor& tensor_map_weights,
    const MakeScheduler& make_scheduler
) {
    constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;

    // 线程身份（block=SM、warp=32线程组、lane=warp内线程号）
    const auto sm_idx = blockIdx.x;                      // 本 block 对应第几个 SM（0..147）
    const auto warp_idx = cutlass::canonical_warp_idx_sync();  // 0..11
    const auto warpgroup_idx = warp_idx / 4;             // UMMA 以 4 warp 为一组
    const auto lane_idx = ptx::get_lane_idx();           // 0..31
    constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;  // 专用 warp 的起始编号 = 8
    constexpr uint32_t kNumTopkWarps = 8;                // Phase 2 参与 TopK 的 warp 数


    // 预取 TMA 描述符（warp 8 负责）: 提前把 5 个张量描述符拉进常量内存
    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_sf_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }

    // ---- 编译期常量 ----
    // TMEM 3 级流水（accum 累加器三级缓冲）
    static constexpr uint32_t kNumTmemStages = 3;
    static constexpr uint32_t kNumUTCCPAlignedElems = 128;
    static constexpr uint32_t UMMA_M = 128;
    static constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;
    static constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
    static constexpr uint32_t kNumSFQ  = kIsMXSF ? math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems) : 0;
    static constexpr uint32_t kNumSFKV = kIsMXSF ? math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems) : 0;
    static constexpr uint32_t kRealNumSFQ = BLOCK_Q * kNumHeads;
    static constexpr uint32_t kNumQKBytesPerToken = kIsFP4 ? (kHeadDim / 2) : kHeadDim;
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kNumQKBytesPerToken;
    static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE = kIsMXSF ? (kRealNumSFQ * sizeof(int)) : 0;
    static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = kIsMXSF ? (kNumSFKV * sizeof(int)) : (SPLIT_KV * sizeof(float));
    static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * sizeof(reduce_dtype_t);

    // 静态约束: 128 专用线程 + 128 倍数的 math 线程; SPLIT_KV 与 UMMA_M 对齐
    DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0, "Invalid threads");
    DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and SPLIT_KV % kNumUTCCPAlignedElems == 0, "Invalid `SPLIT_KV`");

    // SMEM 布局: Phase 1 用 SharedStorage（Q/KV/SF/weights 多级缓冲 + mbarrier），
    // Phase 2 复用同一块 smem_buffer 做 TopK 工作区（见后文 SMEM 布局注释）
    using SharedStorage = layout::MQALogitsSharedStorage<kNumHeads, kHeadDim, kIsMXSF, BLOCK_Q, SPLIT_KV,
                                                         kNumQStages, kNumKVStages, kNumTmemStages, qk_dtype_t, reduce_dtype_t>;
    extern __shared__ __align__(SharedStorage::kSwizzleAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);

    // TMEM 列布局: accum 区 + SFQ 区 + SFKV 区（TMEM 共 512 列）
    constexpr uint32_t kNumAccumTmemCols = BLOCK_Q * kNumHeads * kNumTmemStages;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 + kNumSFKV / 32>();
    constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
    DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

    // ---- barrier 初始化（warp 9，单线程执行） ----
    // 每组 mbarrier 成对: full_* = 数据就绪信号, empty_* = 缓冲空闲信号
    // arrive 计数含义: 生产者(1) + 消费者(n)，全到齐才放行
    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumQStages; ++ i) {
            smem.full_q_barriers[i].init(1);
            smem.empty_q_barriers[i].init(kNumMathThreads + 32);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            smem.full_kv_barriers[i].init(1);
            smem.empty_kv_barriers[i].init(kIsMXSF ? 1 : kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++i) {
            smem.full_tmem_barriers[i].init(1);
            smem.empty_tmem_barriers[i].init(128);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncwarp();

    // ---- TMEM 分配（warp 10）: 每个 CTA 只能发一次 tcgen05.alloc，内部 elect_one_sync ----
    // 分配结果写入 SMEM 供全 block 读取
    if (warp_idx == kSpecWarpStart + 2)
        cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
    __syncthreads();

    // 每个 Q token 的 KV 有效区间（scheduler 填）
    uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];

    // 三套环形流水计数器: Q / KV / TMEM（stage 数 = 模板参数）
    RingPipeline<kNumQStages> q_pipeline;
    RingPipeline<kNumKVStages> kv_pipeline;
    RingPipeline<kNumTmemStages> tmem_pipeline;

    // 寄存器预算: 专用 warp 只留 56 个，其余让给 math warp（224 个）做 UMMA 深流水
    constexpr uint32_t kNumSpecializedRegisters = 56;
    constexpr uint32_t kNumMathRegisters = 224;

    // 等所有前置 kernel 完成（programmatic dependent launch）
    cudaGridDependencySynchronize();

    // ====== Phase 1 warp 8: TMA Q 加载 ======
    // 职责: 循环拉取本 SM 负责的每个 q_block，把 Q/SF_Q/weights 经 TMA 搬进
    //       SMEM 对应 stage，用 full_q_barrier 通知下游
    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();  // 让出寄存器给 math warp
        if (cute::elect_one_sync()) {
            auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

                const uint32_t q_token_base = scheduler.get_q_tma_token_base(q_block_idx);
                tma::copy<kNumQKBytesPerToken, BLOCK_Q * kNumHeads, 0>(
                    &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
                    smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
                if constexpr (kIsMXSF)
                    tma::copy<BLOCK_Q * kNumHeads, 1, 0>(&tensor_map_sf_q, &smem.full_q_barriers[q_stage_idx], smem.smem_sf_q[q_stage_idx], 0, q_token_base);
                tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, &smem.full_q_barriers[q_stage_idx], smem.smem_weights[q_stage_idx], 0, q_token_base);
                smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
            }
        }
        __syncwarp();

    // ====== Phase 1 warp 9: TMA KV 加载 ======
    // 职责: 按 split 循环加载 KV 页/段到 SMEM；
    //       paged 模式先经 block_table 把逻辑页换算成物理页坐标（warp 内缓存 32 页）
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        uint32_t cached_kv_page_base = 0;
        uint32_t cached_kv_page_coord = 0;
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            cached_kv_page_base = cute::numeric_limits<uint32_t>::max();
            #pragma unroll 1
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                if constexpr (decltype(scheduler)::kIsPaged) {
                    constexpr uint32_t kPageKV = decltype(scheduler)::kPageKV;
                    constexpr uint32_t kNumPagesPerSplit = decltype(scheduler)::kNumPagesPerSplit;
                    DG_STATIC_ASSERT(kNumPagesPerSplit <= 32, "Split spans more pages than a warp can cache");

                    const uint32_t kv_page_base = (kv_base + kv_split_idx) * kNumPagesPerSplit;
                    if (kv_page_base < cached_kv_page_base or kv_page_base + kNumPagesPerSplit > cached_kv_page_base + 32) {
                        cached_kv_page_base = (kv_page_base / 32) * 32;
                        cached_kv_page_coord = scheduler.get_kv_page_coord_by_page_offset(cached_kv_page_base + lane_idx);
                    }

                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    if (cute::elect_one_sync())
                        smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
                    __syncwarp();

                    int page_coords[kNumPagesPerSplit];
                    #pragma unroll
                    for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++ page_idx) {
                        const auto src_lane = static_cast<int>(kv_page_base - cached_kv_page_base + page_idx);
                        page_coords[page_idx] = __shfl_sync(0xffffffff, cached_kv_page_coord, src_lane);
                    }

                    if (cute::elect_one_sync()) {
                        #pragma unroll
                        for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++ page_idx) {
                            tma::copy<kNumQKBytesPerToken, kPageKV, 0, qk_dtype_t, true>(
                                &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                                smem.smem_kv[kv_stage_idx] + page_idx * kPageKV * kNumQKBytesPerToken,
                                0, 0, 1, page_coords[page_idx]);
                            tma::copy<kPageKV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                                     smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV,
                                                     0, page_coords[page_idx]);
                        }
                        smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                    }
                    __syncwarp();
                } else if (cute::elect_one_sync()) {
                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);

                    const uint32_t kv_tma_offset = scheduler.get_kv_tma_offset(kv_base, kv_split_idx);
                    tma::copy<kNumQKBytesPerToken, SPLIT_KV, 0>(
                        &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                        smem.smem_kv[kv_stage_idx], 0, kv_tma_offset);
                    tma::copy<SPLIT_KV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                              smem.smem_sf_kv[kv_stage_idx], kv_tma_offset, 0);
                    smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
                }
                __syncwarp();
            }
        }

    // ====== Phase 1 warp 10: MMA issue ======
    // 职责: ① MX 模式先把 scale factor 在 SMEM 内转置(UTCCP 要求 K-major)
    //       ② 用 UTCCP 把 SF 拷进 TMEM
    //       ③ 逐 warpgroup 发射 UMMA 指令（tcgen05.mma），做完 commit 到 TMEM barrier
    } else if (warp_idx == kSpecWarpStart + 2) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);

        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            if constexpr (kIsMXSF) {
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                    utccp_required_smem_warp_transpose(smem_ptr);
                }
                cutlass::arch::fence_view_async_shared();
                #pragma unroll
                for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++ i) {
                    auto smem_ptr = smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
                    mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                    if (cute::elect_one_sync())
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFQ + i * 4);
                    __syncwarp();
                }
            }

            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

                if constexpr (kIsMXSF) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                        auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                        utccp_required_smem_warp_transpose(smem_ptr);
                    }
                    cutlass::arch::fence_view_async_shared();
                }

                if (cute::elect_one_sync()) {
                    if constexpr (kIsMXSF) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, kTmemStartColOfSFKV + i * 4);
                        }
                    }
                    #pragma unroll
                    for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                        CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

                        smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
                        ptx::tcgen05_after_thread_sync();

                        if constexpr (kIsMXSF) {
                            DG_STATIC_ASSERT((not kIsFP4 and kHeadDim == 32) or kHeadDim == 64 or kHeadDim == 128, "Invalid head dim");

                            constexpr uint32_t kPackFactor = kIsFP4 ? 2 : 1;
                            constexpr uint32_t kQKSwizzleMode = kHeadDim / kPackFactor;

                            using mma_op_t = cute::conditional_t<kIsFP4, ptx::SM100_MMA_MXF4_SS, ptx::SM100_MMA_MXF8F6F4_SS>;
                            auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<qk_dtype_t, qk_dtype_t, float, cutlass::float_ue8m0_t,
                                                                                       UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
                            #pragma unroll
                            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                                auto runtime_instr_desc = mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k * kPackFactor, k * kPackFactor);
                                auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(
                                    smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                                auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kQKSwizzleMode>(
                                    smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                                mma_op_t::fma(
                                    a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                                    kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
                            }
                        } else {
                            auto instr_desc = cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
                                                                            UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
                            auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                            #pragma unroll
                            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                                auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                    smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                                auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                    smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                                ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, k, runtime_instr_desc);
                            }
                        }

                        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                                     ::"r"(cute::cast_smem_ptr_to_uint(&smem.full_tmem_barriers[tmem_stage_idx])));
                    }
                }
                __syncwarp();
                if constexpr (kIsMXSF)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_kv_barriers[kv_stage_idx]));
            }
            smem.empty_q_barriers[q_stage_idx].arrive();
        }

    // ====== Phase 1 warp 11: idle ======
    // 职责: 什么都不做，只做寄存器让渡（保持 4-warp 对齐的 warpgroup 结构）
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    // ====== Phase 1 warp 0-7: math warpgroups（计算主力） ======
    // 职责: 等 TMEM 累加完成 → 读回 → relu ×weight ×scale → sum over heads
    //       → 写 logits_buffer
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_warpgroup_idx = warpgroup_idx;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;
        DG_STATIC_ASSERT(kNumMathWarpGroups <= kNumTmemStages, "Math warp groups exceed TMEM stages");
        tmem_pipeline.advance(math_warpgroup_idx);

        constexpr bool kIsReduceBF16 = not cute::is_same_v<reduce_dtype_t, float>;
        DG_STATIC_ASSERT(not kIsReduceBF16 or kNumHeads % 2 == 0, "bf16 weights need even kNumHeads");
        DG_STATIC_ASSERT(kNumHeads == 4 or kNumHeads == 8 or kNumHeads == 16 or kNumHeads == 32 or kNumHeads == 64,
                         "Unsupported TMEM load size");
        using weights_reg_dtype_t = cute::conditional_t<kIsReduceBF16, nv_bfloat162, float>;
        constexpr uint32_t kNumWeightsRegPerToken = kIsReduceBF16 ? (kNumHeads / 2) : kNumHeads;
        weights_reg_dtype_t weights[BLOCK_Q][kNumWeightsRegPerToken];
        float accum[kNumHeads];

        auto tmem_load_no_fence = [](auto num_elems_t, const uint32_t& addr, float* load_dst) {
            constexpr uint32_t N = decltype(num_elems_t)::value;
            using Loader = cute::conditional_t<N == 2,  cute::SM100_TMEM_LOAD_32dp32b2x,
                           cute::conditional_t<N == 4,  cute::SM100_TMEM_LOAD_32dp32b4x,
                           cute::conditional_t<N == 8,  cute::SM100_TMEM_LOAD_32dp32b8x,
                           cute::conditional_t<N == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                           cute::conditional_t<N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                                                        cute::SM100_TMEM_LOAD_32dp32b64x>>>>>;
            [&]<size_t... Is>(cute::index_sequence<Is...>) {
                Loader::copy(addr, reinterpret_cast<uint32_t*>(load_dst)[Is]...);
            }(cute::make_index_sequence<N>{});
        };

        // 循环消费本 SM 分配到的 q_block（与 TMA 侧共享同一 scheduler 迭代序）
        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);   // 等 Q 数据就绪

            // 单个 q_block 的处理 lambda（BLOCK_Q 个 token，编译期展开）
            const auto process_q_block = [&](auto num_valid_tokens_t) {
                constexpr uint32_t kNumValidTokens = decltype(num_valid_tokens_t)::value;

                // ---- 预载本块所有 token 的 weights 到寄存器（bf16 打包 / fp32 向量读） ----
                #pragma unroll
                for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                    const auto smem_weights_row = smem.smem_weights[q_stage_idx] + i * kNumHeads;
                    if constexpr (kIsReduceBF16) {
                        const auto packed_row = reinterpret_cast<const uint32_t*>(smem_weights_row);
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumHeads / 2; ++ j) {
                            const auto packed = ptx::ld_shared(packed_row + j);
                            weights[i][j] = ptx::exchange(*reinterpret_cast<const nv_bfloat162*>(&packed), 0);
                        }
                    } else {
                        // vectorized broadcast load: one ld.shared.v4 per 4 weights
                        const uint32_t* wrow = reinterpret_cast<const uint32_t*>(smem_weights_row);
                        #pragma unroll
                        for (uint32_t j = 0; j < kNumHeads / 4; ++ j) {
                            uint32_t v0, v1, v2, v3;
                            asm("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
                                : "l"(__cvta_generic_to_shared(wrow + j * 4)));
                            weights[i][j * 4 + 0] = __uint_as_float(v0);
                            weights[i][j * 4 + 1] = __uint_as_float(v1);
                            weights[i][j * 4 + 2] = __uint_as_float(v2);
                            weights[i][j * 4 + 3] = __uint_as_float(v3);
                        }
                    }
                }

                // ---- 按 KV split 循环: 每个 math 线程负责 1 个 KV token 的 logit ----
                for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                    // 本线程负责的 KV 全局列号（256 math 线程覆盖 SPLIT_KV=256）
                    auto kv_offset = scheduler.get_logits_col(kv_base, kv_split_idx, math_thread_idx);

                    reduce_dtype_t scale_kv = static_cast<reduce_dtype_t>(0.0f);
                    uint32_t kv_stage_idx = 0;
                    if constexpr (not kIsMXSF) {
                        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx_local, kv_phase);
                        kv_stage_idx = kv_stage_idx_local;
                        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
                        scale_kv = static_cast<reduce_dtype_t>(ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + math_thread_idx));
                    }

                    CUTE_TIE_DECL(tmem_pipeline.advance(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
                    smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
                    ptx::tcgen05_after_thread_sync();

                    if constexpr (not kIsMXSF)
                        smem.empty_kv_barriers[kv_stage_idx].arrive();

                    #pragma unroll
                    for (uint32_t i = 0; i < kNumValidTokens; ++ i) {
                        uint32_t tmem_addr = tmem_stage_idx * UMMA_N + i * kNumHeads;
                        if constexpr (kNumHeads == 8) {
                            tmem_load_no_fence(cute::Int<kNumHeads>{}, tmem_addr, accum);
                            cutlass::arch::fence_view_async_tmem_load();
                        } else if constexpr (kNumHeads == 16) {
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr, accum);
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr + kNumHeads / 2, accum + kNumHeads / 2);
                            cutlass::arch::fence_view_async_tmem_load();
                        } else {
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr, accum);
                            cutlass::arch::fence_view_async_tmem_load();
                            tmem_load_no_fence(cute::Int<kNumHeads / 2>{}, tmem_addr + kNumHeads / 2, accum + kNumHeads / 2);
                            cutlass::arch::fence_view_async_tmem_load();
                        }

                        if (i == kNumValidTokens - 1) {
                            ptx::tcgen05_before_thread_sync();
                            smem.empty_tmem_barriers[tmem_stage_idx].arrive();
                        }

                        reduce_dtype_t reduced;
                        if constexpr (kIsReduceBF16) {
                            auto sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                            auto sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                            const auto transform = [&](const uint32_t& j, const nv_bfloat162& sum) {
                                const auto a = ptx::cvt_relu_bf16x2_f32(make_float2(accum[j], accum[j + 1]));
                                const auto b = weights[i][j / 2];
                                return __hfma2(a, b, sum);
                            };

                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads; j += 4) {
                                sum_0 = transform(j, sum_0);
                                sum_1 = transform(j + 2, sum_1);
                            }

                            auto sum = __hadd2_rn(sum_0, sum_1);
                            reduced = __hadd_rn(sum.x, sum.y);
                        } else {
                            auto sum_0 = make_float2(0, 0);
                            auto sum_1 = make_float2(0, 0);
                            const auto transform = [&](const uint32_t& j, const float2& sum) {
                                auto a_0 = make_float2(accum[j], accum[j + 1]);
                                auto a_1 = make_float2(fabsf(accum[j]), fabsf(accum[j + 1]));
                                auto b = make_float2(weights[i][j], weights[i][j + 1]);
                                return __ffma2_rn(__fadd2_rn(a_0, a_1), b, sum);
                            };

                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads; j += 4) {
                                sum_0 = transform(j, sum_0);
                                sum_1 = transform(j + 2, sum_1);
                            }

                            auto sum = __fadd2_rn(sum_0, sum_1);
                            reduced = (sum.x + sum.y) / 2;
                        }

                        // ---- 写回 logits_buffer（bf16）: compressed 按相对偏移，否则绝对 KV 列 ----
                        auto result = static_cast<float>(
                            kIsMXSF ? reduced : reduced * scale_kv);

                        const auto q_offset = scheduler.get_logits_row(q_block_idx, i) * static_cast<uint64_t>(logits_stride);
                        if constexpr (kIsCompressedLogits) {
                            const uint32_t rel_kv = kv_offset - seq_k_start[i];
                            const uint32_t len = seq_k_end[i] - seq_k_start[i];
                            if (rel_kv < len)
                                logits_buffer[q_offset + rel_kv] = __float2bfloat16(result);
                        } else {
                            if (kv_offset < logits_stride)
                                logits_buffer[q_offset + kv_offset] = __float2bfloat16(result);
                        }
                    }
                }
            };

            if constexpr (decltype(scheduler)::kHasPartialBlock)
                dispatch_num_block_tokens<BLOCK_Q>(scheduler.get_num_block_tokens(q_block_idx), process_q_block);
            else
                process_q_block(cute::Int<BLOCK_Q>{});

            smem.empty_q_barriers[q_stage_idx].arrive();
        }

        // math warp 组内同步，然后 warp0 释放 TMEM（Phase 2 不再需要）
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
    }

    // ========================================================================
    // Phase 1 → Phase 2 边界
    // ① __threadfence: 本 block 的 logits 写推送到 L2（别的 SM 的 Phase 2
    //    要消费这些行——跨 block 可见性的关键，缺失会偶发 5M-cycle stall）
    // ② __syncthreads: 块内 12 warp 对齐（warps 8-11 也到此再分道）
    // ========================================================================
    __threadfence();
    __syncthreads();
    // ========================================================================
    // PHASE 2: 8-warp split TopK with distributed final radix.
    // NOTE: all __syncthreads() are executed uniformly by the WHOLE block
    // (warps 8-11 also arrive), only the work is guarded by warp_idx checks.
    // ========================================================================
    {
        // ---- Phase 2 SMEM 布局（复用 Phase 1 的 smem_buffer，约 105KB） ----
        //   [0, 16KB)     merge_vals  float[4096]   全局候选值
        //   [16K, 32KB)   merge_idx   int32[4096]   全局候选索引
        //   [32K, 33KB)   ghist       uint32[256]   warp0 归并的全局直方图
        //   [33K, ~33.1K) gslots      uint32[16]    跨 warp 广播槽（kth/cutoff/前缀）
        //   [之后]        每 warp 独立区域（stride kPerWarpBytes）:
        //                    wcache  float[1024]   段缓存（4KB）
        //                    whist   uint32[256]   私有直方图（1KB）
        //                    wstage  int32[...]    boundary 暂存（2KB）
        //                    wcand   int32[512]    本 warp 局部 TopK 的候选索引（2KB）
        constexpr uint32_t kSegFloats = 1024;
        constexpr uint32_t kCandPerWarp = kSparseCount;
        const uint32_t kStageBytes = ((kSparseCount + 31) * 4 > 2048) ? ((kSparseCount + 31) * 4) : 2048;
        const uint32_t kPerWarpBytes = kSegFloats * 4 + 256 * 4 + kStageBytes + kCandPerWarp * 4;

        constexpr uint32_t kNumCand = kNumTopkWarps * kCandPerWarp;  // 4096 候选总量
        constexpr uint32_t kSlice = kCandPerWarp;                    // 每 warp 512 片
        float* merge_vals = reinterpret_cast<float*>(smem_buffer);
        int32_t* merge_idx = reinterpret_cast<int32_t*>(merge_vals + kNumCand);
        uint32_t* ghist = reinterpret_cast<uint32_t*>(merge_idx + kNumCand);
        uint32_t* gslots = ghist + 256;
        uint8_t* warp_base = reinterpret_cast<uint8_t*>(gslots + 8 + kNumTopkWarps);

        uint8_t* wregion = warp_base + static_cast<uint64_t>(warp_idx) * kPerWarpBytes;
        float* wcache = reinterpret_cast<float*>(wregion);
        float* whist = reinterpret_cast<float*>(wregion + kSegFloats * 4);
        int32_t* wstage = reinterpret_cast<int32_t*>(wregion + kSegFloats * 4 + 256 * 4);
        int32_t* wcand = reinterpret_cast<int32_t*>(wregion + kSegFloats * 4 + 256 * 4 + kStageBytes);

        const uint32_t sms = gridDim.x;
        const bool i_active = (warp_idx < kNumTopkWarps);   // 只有 warp 0-7 干活；barrier 全体统一执行

        auto scheduler = make_scheduler(sm_idx, seq_k_start, seq_k_end);
        const uint32_t num_q_tokens = [&] {
            if constexpr (decltype(scheduler)::kIsPaged) return scheduler.num_q_tokens_total;
            else return scheduler.num_q_tokens;
        }();

        // 每个 SM 按 stride=148 认领 token（串行处理，全 block 协作每个 token）
        for (uint32_t q_token = sm_idx; q_token < num_q_tokens; q_token += sms) {
            uint32_t k_start, k_end;
            if constexpr (decltype(scheduler)::kIsPaged) {
                k_start = 0;
                k_end = scheduler.context_lens[q_token];
            } else {
                k_start = cute::min(scheduler.cu_seq_len_k_start[q_token], scheduler.num_kv_tokens);
                k_end = cute::min(scheduler.cu_seq_len_k_end[q_token], scheduler.num_kv_tokens);
            }
            const uint32_t k_len = k_end > k_start ? (k_end - k_start) : 0;
            const nv_bfloat16* logits_row = logits_buffer + static_cast<uint64_t>(q_token) * logits_stride;
            int32_t* out_ptr = sparse_indices + static_cast<uint64_t>(q_token) * kSparseCount;


            // 边缘 1: 空 token → 全 -1
            if (i_active && k_len == 0) {
                for (uint32_t i = lane_idx; i < kSparseCount; i += 32)
                    out_ptr[i] = -1;
            }
            __syncthreads();
            if (k_len == 0) continue;
            // 边缘 2: k_len ≤ K → 全部入选直写
            if (k_len > 0 && k_len <= kSparseCount) {
                for (uint32_t i = warp_idx * 32 + lane_idx; i < kSparseCount; i += kNumTopkWarps * 32)
                    out_ptr[i] = (i < k_len) ? static_cast<int32_t>(k_start + i) : -1;
                __syncthreads();
                continue;
            }

            // 段划分: warp w 负责 [w*segL, min(k_len,(w+1)*segL))
            const uint32_t segL = (k_len + kNumTopkWarps - 1) / kNumTopkWarps;
            const uint32_t my_start = warp_idx * segL;
            const uint32_t my_len = (my_start < k_len) ? ((my_start + segL > k_len) ? k_len - my_start : segL) : 0;

            __threadfence();   // 读前 fence: 配合 writer 侧 fence 保证本 token 行可见

            // ==== stage 1: 每 warp 段内局部 TopK ====
            // 段 ≤ 1024: 先整段载入 SMEM 缓存（coalesced + bf16→float），再跑局部 radix
            // 段 > 1024（长上下文兜底）: 直接对 GMEM bf16 行跑（kCached=false）
            // 输出 wcand[0..512) = 本段 top-512 的绝对 KV 索引
            if (i_active && my_len > 0 && my_len <= kSegFloats) {
                #pragma unroll 4
                for (uint32_t i = lane_idx; i < my_len; i += 32)
                    ptx::st_shared(reinterpret_cast<uint32_t*>(wcache) + i,
                                   __float_as_uint(__bfloat162float(logits_row[k_start + my_start + i])));
                __syncwarp();
                warp_bucket_topk<kSparseCount, true>(wcache, k_start + my_start, my_len,
                                                     wcand, whist, wstage, nullptr, lane_idx);
            } else if (i_active && my_len > 0) {
                warp_bucket_topk<kSparseCount, false>(logits_row + k_start + my_start,
                                                      k_start + my_start, my_len,
                                                      wcand, whist, wstage, nullptr, lane_idx);
            } else if (i_active) {
                for (uint32_t i = lane_idx; i < kCandPerWarp; i += 32)
                    wcand[i] = -1;
            }
            __syncthreads();

            // ==== stage 2: 构建全局候选数组 ====
            // 正确性依据: 元素不在本段 top-512 → 本段有 ≥512 个更大 → 必不在全局 top-512
            // 8 warp × 32 lane 分担 4096 项: 取候选索引 + 回推段内位置读值
            // 无效候选（短 token）填 -INF/-1 哨兵（to_sortable 下排最低）
            if (i_active) {
                const bool all_seg_cached = (k_len <= kNumTopkWarps * kSegFloats);
                const uint32_t kCandOffU32 = kSegFloats + 256 + kStageBytes / 4;
                const int32_t* cand_base = reinterpret_cast<int32_t*>(warp_base);
                const uint32_t* cache_base = reinterpret_cast<const uint32_t*>(warp_base);
                #pragma unroll 4
                for (uint32_t j = warp_idx * 32 + lane_idx; j < kNumCand; j += kNumTopkWarps * 32) {
                    uint32_t w = j / kCandPerWarp;
                    uint32_t pos = j % kCandPerWarp;
                    int32_t c = cand_base[w * (kPerWarpBytes / 4) + kCandOffU32 + pos];
                    if (c >= 0) {
                        merge_idx[j] = c;
                        if (all_seg_cached) {
                            const uint32_t cache_pos = static_cast<uint32_t>(c) - k_start - w * segL;
                            merge_vals[j] = __uint_as_float(
                                ptx::ld_shared(cache_base + w * (kPerWarpBytes / 4) + cache_pos));
                        } else {
                            merge_vals[j] = __bfloat162float(logits_row[static_cast<uint32_t>(c) - k_start]);
                        }
                    } else {
                        merge_vals[j] = -CUDART_INF_F;
                        merge_idx[j] = -1;
                    }
                }
            }
            __syncthreads();

            // ==== stage 3: 分布式 radix rounds（8 warp × 512 片） ====
            // 与 warp_bucket_topk 的 rounds 等价，但直方图拆分:
            //   每 warp 扫自己的片 → 私有 hist → warp0 归并 8 份 → 全局 cutoff
            // 共享量（cutoff/cao/valid）经 gslots + __syncthreads 广播
            {
                const float* mv3 = merge_vals;
                auto to_sortable3 = [](float f) -> uint32_t {
                    uint32_t u = __float_as_uint(f);
                    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
                };
                uint32_t kth = 0, def_cnt = 0, rem = kSparseCount;
                // bf16 数据: 只跑 2 轮（字节 3/2），后 2 轮恒空转
                for (int round = 0; round < 2 && rem > 0; ++round) {
                    const int shift = (3 - round) * 8;
                    const uint32_t pmask = (round == 0) ? 0u : (0xFFFFFFFFu << (32 - round * 8));
                    const uint32_t pval = kth & pmask;

                    // ① 各 warp 清私有直方图 + 扫自己的 512 片（前缀命中才计数）
                    if (i_active) {
                        uint32_t* wh = reinterpret_cast<uint32_t*>(whist);
                        for (uint32_t i = lane_idx; i < 256; i += 32)
                            wh[i] = 0;
                        __syncwarp();
                        #pragma unroll 4
                        for (uint32_t i = lane_idx; i < kSlice; i += 32) {
                            uint32_t u = to_sortable3(__uint_as_float(
                                ptx::ld_shared(reinterpret_cast<const uint32_t*>(mv3) + warp_idx * kSlice + i)));
                            if ((u & pmask) == pval) {
                                uint32_t b = (u >> shift) & 0xFF;
                                atomicAdd(&wh[b], 1u);
                            }
                        }
                    }
                    __syncthreads();

                    // ② warp0 归并 8 份私有直方图 → 全局 ghist
                    if (warp_idx == 0) {
                        const uint32_t* wp = reinterpret_cast<const uint32_t*>(warp_base);
                        for (uint32_t i = lane_idx; i < 256; i += 32) {
                            uint32_t s = 0;
                            #pragma unroll
                            for (uint32_t w = 0; w < kNumTopkWarps; ++w)
                                s += wp[w * (kPerWarpBytes / 4) + kSegFloats + i];
                            ghist[i] = s;
                        }
                    }
                    __syncthreads();

                    // ③ warp0 从全局直方图定位 cutoff（32 lane × 8 桶 + shfl 前缀，同 v2 逻辑）
                    if (warp_idx == 0) {
                        uint32_t mc[8];
                        #pragma unroll
                        for (int i = 0; i < 8; i++) mc[i] = ghist[lane_idx * 8 + i];
                        uint32_t my_sum = 0;
                        #pragma unroll
                        for (int i = 0; i < 8; i++) my_sum += mc[i];
                        uint32_t incl = my_sum;
                        for (int s = 1; s < 32; s <<= 1) {
                            uint32_t t = __shfl_up_sync(0xFFFFFFFF, incl, s);
                            if (lane_idx >= s) incl += t;
                        }
                        const uint32_t total = __shfl_sync(0xFFFFFFFF, incl, 31);
                        const uint32_t count_above_me = total - incl;
                        uint32_t prefix_in_lane = 0;
                        int32_t my_cutoff = -1;
                        uint32_t my_cao = 0;
                        #pragma unroll
                        for (int i = 7; i >= 0; i--) {
                            uint32_t ca = count_above_me + prefix_in_lane;
                            uint32_t cge = ca + mc[i];
                            if (ca < rem && cge >= rem) {
                                my_cutoff = static_cast<int32_t>(lane_idx * 8 + i);
                                my_cao = ca;
                            }
                            prefix_in_lane += mc[i];
                        }
                        uint32_t m = __ballot_sync(0xFFFFFFFF, my_cutoff >= 0);
                        gslots[2] = (m != 0);
                        if (m) {
                            uint32_t cl = __ffs(m) - 1;
                            gslots[0] = __shfl_sync(0xFFFFFFFF, static_cast<uint32_t>(my_cutoff), cl);
                            gslots[1] = __shfl_sync(0xFFFFFFFF, my_cao, cl);
                        }
                    }
                    __syncthreads();
                    // ④ 全体 warp 统一更新 radix 状态（kth 字节拼接 / definite / remaining）
                    if (gslots[2] == 0) break;
                    kth |= (gslots[0] << shift);
                    def_cnt += gslots[1];
                    rem = kSparseCount - def_cnt;
                }
                // 发布最终 kth 到 gslots[0]，供 stage 4 读取
                if (warp_idx == 0) gslots[0] = kth;
            }
            __syncthreads();
            const uint32_t kth_full = gslots[0];

            // ==== stage 4: 分布式 selection（8 warp × 512 片 + 块级前缀） ====
            // ① 每 warp 数自己的 definite + 暂存 boundary
            // ② 块级前缀算全局写偏移（warp 内 xor-reduce + warp0 串行前缀）
            // ③ 写 definite → ④ 写 boundary → ⑤ 尾部填 -1
            {
                constexpr uint32_t kPerLane2 = (kSparseCount + 31) / 32;
                const float* mv = merge_vals;
                auto to_sortable2 = [](float f) -> uint32_t {
                    uint32_t u = __float_as_uint(f);
                    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
                };
                auto load_val = [&](uint32_t i) -> uint32_t {
                    return ptx::ld_shared(reinterpret_cast<const uint32_t*>(mv) + warp_idx * kSlice + i);
                };

                // ① 单遍扫描: 数 definite（u>kth）+ 暂存 boundary（u==kth，上限 kPerLane2）
                uint32_t my_def = 0, my_col = 0;
                if (i_active) {
                    #pragma unroll 4
                    for (uint32_t i = lane_idx; i < kSlice; i += 32) {
                        uint32_t u = to_sortable2(__uint_as_float(load_val(i)));
                        if (u > kth_full) {
                            my_def++;
                        } else if (u == kth_full && my_col < kPerLane2) {
                            wstage[lane_idx * kPerLane2 + my_col] =
                                merge_idx[warp_idx * kSlice + i];
                            my_col++;
                        }
                    }
                }

                // ② definite 的块级前缀:
                //    warp 内 xor-reduce 到独立变量 warp_def（保持 my_def 原值！），
                //    写槽 → warp0 串行前缀改写成 exclusive → 全 block 读回基址
                uint32_t warp_def = my_def;
                if (i_active) {
                    for (int s = 16; s > 0; s >>= 1)
                        warp_def += __shfl_xor_sync(0xFFFFFFFF, warp_def, s);
                    if (lane_idx == 0) gslots[8 + warp_idx] = warp_def;
                }
                __syncthreads();
                if (warp_idx == 0) {
                    uint32_t acc = 0;
                    for (uint32_t w = 0; w < kNumTopkWarps; ++w) {
                        uint32_t v = gslots[8 + w];
                        gslots[8 + w] = acc;
                        acc += v;
                    }
                    gslots[0] = acc;   // total definite
                }
                __syncthreads();
                const uint32_t total_definite = gslots[0];

                // ③ 写 definite: 全局基址 = 块级基址 + warp 内 exclusive 前缀
                if (i_active) {
                    uint32_t def_prefix = my_def;
                    for (int s = 1; s < 32; s <<= 1) {
                        uint32_t t = __shfl_up_sync(0xFFFFFFFF, def_prefix, s);
                        if (lane_idx >= s) def_prefix += t;
                    }
                    uint32_t run = gslots[8 + warp_idx] + (def_prefix - my_def);
                    #pragma unroll 4
                    for (uint32_t i = lane_idx; i < kSlice; i += 32) {
                        uint32_t u = to_sortable2(__uint_as_float(load_val(i)));
                        if (u > kth_full) {
                            out_ptr[run] = merge_idx[warp_idx * kSlice + i];
                            run++;
                        }
                    }
                }
                __syncthreads();

                // ④ boundary 的同构块级前缀（warp 内归约 + 槽位前缀）
                uint32_t my_wc = 0;
                if (i_active) {
                    for (int s = 16; s > 0; s >>= 1)
                        my_wc += __shfl_xor_sync(0xFFFFFFFF, my_col, s);
                    if (lane_idx == 0) gslots[8 + warp_idx] = my_wc;
                }
                __syncthreads();
                if (warp_idx == 0) {
                    uint32_t acc = 0;
                    for (uint32_t w = 0; w < kNumTopkWarps; ++w) {
                        uint32_t v = gslots[8 + w];
                        gslots[8 + w] = acc;
                        acc += v;
                    }
                    gslots[1] = acc;   // total collected
                }
                __syncthreads();
                const uint32_t col_warp_base = gslots[8 + warp_idx];

                if (i_active) {
                    uint32_t col_incl = my_col;
                    for (int s = 1; s < 32; s <<= 1) {
                        uint32_t t = __shfl_up_sync(0xFFFFFFFF, col_incl, s);
                        if (lane_idx >= s) col_incl += t;
                    }
                    const uint32_t col_lane_base = col_incl - my_col;
                    for (uint32_t j = 0; j < my_col; ++j) {
                        uint32_t pos = total_definite + col_warp_base + col_lane_base + j;
                        if (pos < kSparseCount)
                            out_ptr[pos] = wstage[lane_idx * kPerLane2 + j];
                    }
                }
                __syncthreads();

                // ⑤ 尾部空位填 -1（warp0 包办）
                if (warp_idx == 0) {
                    const uint32_t rem_slots = kSparseCount - total_definite;
                    const uint32_t filled = total_definite +
                        (gslots[1] < rem_slots ? gslots[1] : rem_slots);
                    for (uint32_t i = filled + lane_idx; i < kSparseCount; i += 32)
                        out_ptr[i] = -1;
                }
            }

        }
    }

    __syncthreads();
}

// =============================================================================
// Entry point: Non-paged (contiguous KV) with TopK
// =============================================================================
template <uint32_t kNumHeads, uint32_t kHeadDim,
          bool kIsMXSF, bool kIsCompressedLogits,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kSparseCount,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_mqa_logits_with_topk(
    const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
    const uint32_t logits_stride,
    const uint32_t* cu_seq_len_k_start,
    const uint32_t* cu_seq_len_k_end,
    nv_bfloat16* __restrict__ logits_buffer,
    int32_t* __restrict__ sparse_indices,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights
) {
    const auto make_scheduler = [&](const uint32_t& sm_idx, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        return sched::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs>(
            sm_idx, num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end, seq_k_start, seq_k_end);
    };

    sm100_mqa_logits_with_topk_core_impl<
        kNumHeads, kHeadDim, kIsMXSF, kIsCompressedLogits, BLOCK_Q, SPLIT_KV,
        kNumQStages, kNumKVStages, kNumSMs,
        kNumSpecializedThreads, kNumMathThreads,
        kSparseCount,
        qk_dtype_t, logits_dtype_t, reduce_dtype_t,
        decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits_buffer, sparse_indices,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

// =============================================================================
// Entry point: Paged KV with TopK
// =============================================================================
template <uint32_t kTokensPerRequest, uint32_t kNumHeads,
          uint32_t kHeadDim, uint32_t PAGE_KV,
          bool kIsMXSF, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t SPLIT_KV, uint32_t kSplitsPerChunk,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          uint32_t kSparseCount,
          typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(kNumSpecializedThreads + kNumMathThreads, 1)
void sm100_paged_mqa_logits_with_topk(
    const uint32_t num_q_tokens_total,
    const uint32_t logits_stride, const uint32_t block_table_stride,
    const uint32_t* context_lens, nv_bfloat16* __restrict__ logits_buffer,
    int32_t* __restrict__ sparse_indices,
    const uint32_t* block_table, const uint32_t* indices,
    const uint32_t* schedule_meta,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights
) {
    static constexpr uint32_t BLOCK_Q = 128 / kNumHeads;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    DG_STATIC_ASSERT(SPLIT_KV == PAGE_KV * kNumPagesPerSplit, "Invalid split/page size");

    const auto make_scheduler = [&](const uint32_t& sm_idx, uint32_t* seq_k_start, uint32_t* seq_k_end) {
        return sched::SM100PagedMQALogitsScheduler<kTokensPerRequest, kIsContextLens2D, kIsVarlen,
                                                   kNumHeads, SPLIT_KV, PAGE_KV, kSplitsPerChunk>(
            sm_idx, context_lens, schedule_meta, indices,
            block_table, block_table_stride, num_q_tokens_total);
    };

    sm100_mqa_logits_with_topk_core_impl<
        kNumHeads, kHeadDim, kIsMXSF, false, BLOCK_Q, SPLIT_KV,
        kNumQStages, kNumKVStages, 0,
        kNumSpecializedThreads, kNumMathThreads,
        kSparseCount,
        qk_dtype_t, logits_dtype_t, reduce_dtype_t,
        decltype(make_scheduler), kNumMathWarpGroups>(
        logits_stride, logits_buffer, sparse_indices,
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
        make_scheduler);
}

} // namespace deep_gemm

#pragma clang diagnostic pop
