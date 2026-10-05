#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_RESOURCE_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_RESOURCE_HPP
#pragma once

#include <cstdint>
#include "../ascendc/tiling.h"
#include "sage_tile.hpp"
#include "catlass/arch/arch.hpp"
#if SAGE_ATTN_DEVICE_COMPILE
#include "catlass/arch/cross_core_sync.hpp"
#endif

namespace Catlass::SageAttn {

using Arch = Catlass::Arch::Ascend950;

/**
 * 1C2V 硬件资源配额契约（Atlas 950PR 架构白皮书表 4-2）。
 * 仅保留有生产消费方的配额项（InspectTilingPlan 的 L0C 判据、Vector0/1 预算链的
 * UB 封顶判据）；L1/L0A/L0B 总量由 tiling.h 的 L1_TOTAL_SIZE 与官方
 * Catlass::Arch::Ascend950 常数承载（test_tiling_struct 直锁官方源，不再设别名层）。
 */
struct HardwareResourceQuota {
    // 物理硬件上限
    static constexpr uint32_t L0C_TOTAL_BYTES  = Arch::L0C_SIZE;                  // 256 KB
    // UB 封顶唯一定义点 ascendc/tiling.h（UB_TOTAL_SIZE_PER_V），此处仅 constexpr 别名
    static constexpr uint32_t UB_PER_V_BYTES   = sage_attn::UB_TOTAL_SIZE_PER_V;
    // 昇腾 950 硬件跨核同步队列深度上限（复用 CATLASS Arch::MAX_REVERSE_DEPTH）
#if SAGE_ATTN_DEVICE_COMPILE
    static constexpr uint32_t MAX_CROSS_CORE_SYNC_DEPTH = Catlass::Arch::MAX_REVERSE_DEPTH;
#else
    static constexpr uint32_t MAX_CROSS_CORE_SYNC_DEPTH = 15;
#endif
};

/**
 * Vector 核心 UB 严格解耦划分契约 (task/01_design.md 3.2 节)
 */
struct Vector0UbBudget {
    // V0（量化 prologue 核）：负责 GM 输入暂存、INT8/FP8 量化、ND2NZ 摆放
    static constexpr uint32_t GM_INPUT_PING_OFFSET     = 0;
    static constexpr uint32_t GM_INPUT_PING_BYTES      = 48 * 1024;  // 48 KB

    static constexpr uint32_t QUANT_WORKSPACE_OFFSET   = 96 * 1024;
    static constexpr uint32_t QUANT_WORKSPACE_BYTES    = 48 * 1024;  // 48 KB

    // 量化工作区子偏移（P6 具名化；Q scale 槽区扩容至 sage_attn::kQScaleSlotStride=4096 f32
    // = 16KB 后重排——tile: rowmax 512B | scales 16KB | km 512B | strip 512B | vmin 512B | vsum 512B；
    // vmin/vsum 为 smooth_v 的通道统计预扫槽，A7 smooth_v 设备核专用）：
    static constexpr uint32_t QUANT_ROWMAX_OFFSET      = QUANT_WORKSPACE_OFFSET;
    static constexpr uint32_t QUANT_SCALES_OFFSET      = QUANT_ROWMAX_OFFSET + 512;
    static constexpr uint32_t QUANT_KM_OFFSET          = QUANT_SCALES_OFFSET + sage_attn::kQScaleSlotStride * 4;
    static constexpr uint32_t QUANT_STAGE_STRIP_OFFSET = QUANT_KM_OFFSET + 512;
    static constexpr uint32_t QUANT_STAGE_STRIP_END    = QUANT_STAGE_STRIP_OFFSET + 512;
    static constexpr uint32_t QUANT_VMIN_OFFSET        = QUANT_STAGE_STRIP_END;
    static constexpr uint32_t QUANT_VSUM_OFFSET        = QUANT_VMIN_OFFSET + 512;
    static_assert(QUANT_ROWMAX_OFFSET < QUANT_SCALES_OFFSET &&
                  QUANT_SCALES_OFFSET < QUANT_KM_OFFSET &&
                  QUANT_KM_OFFSET < QUANT_STAGE_STRIP_OFFSET &&
                  QUANT_STAGE_STRIP_END <= QUANT_VMIN_OFFSET &&
                  QUANT_VMIN_OFFSET < QUANT_VSUM_OFFSET &&
                  QUANT_VSUM_OFFSET + 512 <= QUANT_WORKSPACE_OFFSET + QUANT_WORKSPACE_BYTES,
                  "Vector0 quant workspace sub-offsets overlap or exceed the region");

    static constexpr uint32_t NZ_OUTPUT_OFFSET         = 144 * 1024;

    static constexpr uint32_t RESERVED_OFFSET          = 240 * 1024;
    static constexpr uint32_t RESERVED_BYTES           = 16 * 1024;  // 16 KB 保留

    static_assert(RESERVED_OFFSET + RESERVED_BYTES <= HardwareResourceQuota::UB_PER_V_BYTES,
                  "Vector0 UB allocation exceeds 256KB physical limit");
};

struct Vector1UbBudget {
    // V1（softmax 核）：负责 Online Softmax、O 主累加器、两级累加归约。
    // R17-2 重排（2026-10-03）：A7 设备态交接区（dot32/pv_partial）并入本记账结构——
    // 原 dot32@80KB/pv_partial@144KB 与 S pong 槽、TWO_LEVEL_REDUCE 记账区三方重叠
    //（Cube FixPipe 写 dot32 会踩 V1 的 S pong 槽），现全部区 512B 对齐、两两不相交：
    //   O_ACCUM [0,64K) | S 双槽 [64K, 64K+2·rows·block_n)（≤96K，FIXPIPE_RECV 记账区）
    //   A7 dot32 [96K,160K) | A7 pv_partial [160K,224K) | LSE_WORKSPACE [224K,240K)
    //   TWO_LEVEL_REDUCE [240K,256K)
    // 互斥与封顶由下方 static_assert 链编译期锁定（链式单调 ⇒ 两两不相交）。
    static constexpr uint32_t O_ACCUM_OFFSET           = 0;
    static constexpr uint32_t O_ACCUM_BYTES            = 64 * 1024;  // 64 KB (FP32 常驻主累加器)

    static constexpr uint32_t FIXPIPE_RECV_OFFSET      = 64 * 1024;
    static constexpr uint32_t FIXPIPE_RECV_BYTES       = 64 * 1024;  // 64 KB (S(P) 奇偶双槽记账区；
                                                                      //  实际占用 2·rows·block_n ≤ 32K)

    // A7 设备态交接区（task/12 D1/D2：Cube FixPipe 裸搬落点，V1 消费；单缓冲读保护
    // 握手 DOT_FREE / PV_PARTIAL_FREE，attention_sync.h）
    static constexpr uint32_t A7_DOT32_OFFSET          = 96 * 1024;
    static constexpr uint32_t A7_DOT32_BYTES           = 64 * 1024;  // rows·block_n·4 ≤ 128·128·4
    static constexpr uint32_t A7_PV_PARTIAL_OFFSET     = 160 * 1024;
    static constexpr uint32_t A7_PV_PARTIAL_BYTES      = 64 * 1024;  // rows·d_pad·4 ≤ 128·128·4

    static constexpr uint32_t LSE_WORKSPACE_OFFSET     = 224 * 1024;
    static constexpr uint32_t LSE_WORKSPACE_BYTES      = 16 * 1024;  // 行 stats + logits 行暂存 + scale 条带副本
    // LSE 工作区行距（512B = 128 float；消费侧子偏移 = OFFSET + n·LSE_ROW_STRIDE）
    static constexpr uint32_t LSE_ROW_STRIDE           = 512;
    // LSE 工作区行级暂存子槽唯一定义点（每槽 512B = 128 float）：
    static constexpr uint32_t V1_ROW_MAX_OFFSET        = LSE_WORKSPACE_OFFSET;
    static constexpr uint32_t V1_ROW_SUM_OFFSET        = LSE_WORKSPACE_OFFSET + LSE_ROW_STRIDE;
    static constexpr uint32_t V1_KS_ROW_OFFSET         = LSE_WORKSPACE_OFFSET + 2 * LSE_ROW_STRIDE;
    static constexpr uint32_t V1_QS_ROW_OFFSET         = LSE_WORKSPACE_OFFSET + 3 * LSE_ROW_STRIDE;
    static constexpr uint32_t V1_VS_CHAN_OFFSET        = LSE_WORKSPACE_OFFSET + 4 * LSE_ROW_STRIDE;
    static constexpr uint32_t V1_VM_CHAN_OFFSET        = LSE_WORKSPACE_OFFSET + 5 * LSE_ROW_STRIDE;
    static constexpr uint32_t V1_CORR_ROW_OFFSET       = LSE_WORKSPACE_OFFSET + 6 * LSE_ROW_STRIDE;
    static constexpr uint32_t V1_LSE_ROW_OFFSET        = LSE_WORKSPACE_OFFSET + 7 * LSE_ROW_STRIDE;
    static constexpr uint32_t V1_M_LN_ROW_OFFSET       = LSE_WORKSPACE_OFFSET + 8 * LSE_ROW_STRIDE;
    static constexpr uint32_t V1_RS_FACTOR_ROW_OFFSET  = LSE_WORKSPACE_OFFSET + 9 * LSE_ROW_STRIDE;
    static_assert(V1_RS_FACTOR_ROW_OFFSET + 512u <= LSE_WORKSPACE_OFFSET + LSE_WORKSPACE_BYTES,
                  "Vector1 LSE workspace sub-offsets overlap or exceed the region");

    static constexpr uint32_t TWO_LEVEL_REDUCE_OFFSET  = 240 * 1024;
    static constexpr uint32_t TWO_LEVEL_REDUCE_BYTES   = 16 * 1024;  // 两级累加归约缓冲（标准 fp32+fp16 周期归约）

    // 行界常数（与 SageResourceManager::MAX_GQA_PACK_ROWS / host 分块域 block_n ∈ {64,128} /
    // d_pad ≤ 128 验收域同步；唯一 行上限定义点 sage_attn::MAX_PACK_ROWS（tiling.h），
    // 本值经下方 static_assert 锁定相等，改动须双侧联动 + 台账回归）
    static constexpr uint32_t A7_REGION_ROW_MAX        = 128;
    static constexpr uint32_t A7_REGION_BLOCK_N_MAX    = 128;
    static constexpr uint32_t A7_REGION_HEAD_DIM_MAX   = 128;
    static_assert(A7_REGION_ROW_MAX == sage_attn::MAX_PACK_ROWS,
                  "A7_REGION_ROW_MAX must equal tiling.h MAX_PACK_ROWS (single row-cap definition point)");

    static_assert(O_ACCUM_OFFSET + O_ACCUM_BYTES <= FIXPIPE_RECV_OFFSET,
                  "A7 O_ACCUM region overlaps S slot region");
    static_assert(FIXPIPE_RECV_OFFSET +
                          2u * A7_REGION_ROW_MAX * A7_REGION_BLOCK_N_MAX <= A7_DOT32_OFFSET,
                  "A7 S pong slot region overlaps dot32");
    static_assert(A7_DOT32_OFFSET +
                          A7_REGION_ROW_MAX * A7_REGION_BLOCK_N_MAX * sizeof(int32_t)
                      <= A7_PV_PARTIAL_OFFSET,
                  "A7 dot32 region overlaps pv_partial");
    static_assert(A7_PV_PARTIAL_OFFSET +
                          A7_REGION_ROW_MAX * A7_REGION_HEAD_DIM_MAX * sizeof(float)
                      <= LSE_WORKSPACE_OFFSET,
                  "A7 pv_partial region overlaps LSE workspace");
    static_assert(LSE_WORKSPACE_OFFSET + LSE_WORKSPACE_BYTES <= TWO_LEVEL_REDUCE_OFFSET,
                  "A7 LSE workspace overlaps TWO_LEVEL_REDUCE");
    static_assert(TWO_LEVEL_REDUCE_OFFSET + TWO_LEVEL_REDUCE_BYTES
                      <= HardwareResourceQuota::UB_PER_V_BYTES,
                  "Vector1 UB allocation exceeds 256KB physical limit");
};

/**
 * 运行时资源台账违规码（结构化 fail-fast，RES-01）：
 * 片上容量超限在 NPU 上表现为无诊断信息的硬件异常（案例研究 ascend-rs §6 的
 * C++ 16384 元素静默踩 UB 事故），故随 shape 变化的足迹必须在 host 发射前拒绝，
 * 并给出可归档的区域级违规原因，而非依赖 device 侧 static_assert 单点防线。
 */
enum class FootprintViolation : uint32_t {
    kNone = 0,
    kInvalidShape,     // 行/列/头维为 0，分块计划本身非法
    kGqaRowOverflow,   // GQA 聚合 M 打包行数超出核内行状态上限
    kL1Bank0Q,         // L1 Bank 0：Q 数据区越界（Data+Scale 相邻契约被破坏）
    kL1Bank1Kv,        // L1 Bank 1：KV Ping-Pong 双槽越界
    kL0CAccum,         // L0C：QK^T INT32 累加器越界
    kUbV1OAccum,       // V1 UB：O 主累加器（fp32 常驻）越界
    kUbV1SoftmaxSlot,  // V1 UB：S(P) 奇偶双槽越出 FIXPIPE_RECV 区
    kUbV1A7Handoff,    // V1 UB：A7 dot32/pv_partial 交接区越界（Cube FixPipe 落点，R17-2 建模）
    kUbV0Input,        // V0 UB：GM 输入 ping/pong 暂存槽越界
};

/**
 * 资源分配校验器
 */
class SageResourceManager {
public:
    // 与 sage_attn::MAX_PACK_ROWS（tiling.h 唯一定义点，kernel 行窗口直读）契约同步
    // （相等由 static_assert 编译期锁定，改动须双侧联动 + test_tiling_struct 台账用例
    // 回归）；Q_MAX_BYTES 的 128x128 规格隐含同一上限。
    static constexpr uint32_t MAX_GQA_PACK_ROWS = 128;
    static_assert(MAX_GQA_PACK_ROWS == sage_attn::MAX_PACK_ROWS,
                  "MAX_GQA_PACK_ROWS must equal tiling.h MAX_PACK_ROWS (single row-cap definition point)");

    /**
     * 运行时资源台账检查（RES-01，host 发射前 fail-fast 的唯一判据）：
     * total_rows 为 GQA 聚合后的 M 打包宽（gqa_ratio * m_per_core，即 kernel 侧
     * TotalRows），不是 m_per_core 本身——O 累加器、S 双槽、L0C 均按聚合宽计。
     * 纯 constexpr 整数算术，host 热路径零堆分配，微秒级发射。
     * 各区域容量唯一定义点：L1DoubleBankContract / Vector0UbBudget /
     * Vector1UbBudget / HardwareResourceQuota（本文件与 sage_tile.hpp）。
     */
    [[nodiscard]] static constexpr FootprintViolation InspectTilingPlan(
        uint32_t total_rows, uint32_t block_n, uint32_t head_dim)
    {
        if (total_rows == 0u || block_n == 0u || head_dim == 0u) {
            return FootprintViolation::kInvalidShape;
        }
        if (total_rows > MAX_GQA_PACK_ROWS) {
            return FootprintViolation::kGqaRowOverflow;
        }

        // L1 Bank 0：Q 数据区按 [total_rows][head_dim] INT8 摆放（kernel
        // LoadAndQuantizeQ 的 L1 写出契约），越界即破坏后邻 Q Scale 区
        const uint64_t q_bytes = static_cast<uint64_t>(total_rows) * head_dim;
        if (q_bytes > L1DoubleBankContract::Q_MAX_BYTES) {
            return FootprintViolation::kL1Bank0Q;
        }

        // L1 Bank 1：KV 双槽（K INT8 + V FP8 各半槽，含槽内 Scale 区）
        const uint64_t kv_bytes = static_cast<uint64_t>(L1DoubleBankContract::KV_SLOT_BYTES) * 2u;
        if (kv_bytes > L1DoubleBankContract::BANK1_SIZE) {
            return FootprintViolation::kL1Bank1Kv;
        }

        // L0C：QK^T INT32 累加器 [total_rows][block_n]
        const uint64_t l0c_bytes = static_cast<uint64_t>(total_rows) * block_n * 4u;
        if (l0c_bytes > HardwareResourceQuota::L0C_TOTAL_BYTES) {
            return FootprintViolation::kL0CAccum;
        }

        // V1 UB：O 主累加器 [total_rows][head_dim] FP32（O_ACCUM_OFFSET = 0 起）
        const uint64_t o_accum_bytes = static_cast<uint64_t>(total_rows) * head_dim * 4u;
        if (o_accum_bytes > Vector1UbBudget::O_ACCUM_BYTES) {
            return FootprintViolation::kUbV1OAccum;
        }

        // V1 UB：S(P) 奇偶双槽驻留 FIXPIPE_RECV 区（kernel SoftmaxSlotOffset 以
        // ub_softmax_offset = FIXPIPE_RECV_OFFSET 为基址，槽步长 rows * block_n 字节）
        const uint64_t softmax_slot_bytes = static_cast<uint64_t>(total_rows) * block_n * 2u;
        if (softmax_slot_bytes > Vector1UbBudget::FIXPIPE_RECV_BYTES) {
            return FootprintViolation::kUbV1SoftmaxSlot;
        }

        // V1 UB：A7 dot32 裸搬区 [total_rows][block_n] INT32（R17-2 全区建模：Cube
        // EmitL0CToUbRaw 落点、V1 DequantMaskLogitsRowsVF 原地反量化消费（R4：float logits 元素等宽覆写），FLAG_DOT_FREE 读保护）
        const uint64_t dot32_bytes = static_cast<uint64_t>(total_rows) * block_n * sizeof(int32_t);
        if (dot32_bytes > Vector1UbBudget::A7_DOT32_BYTES) {
            return FootprintViolation::kUbV1A7Handoff;
        }

        // V1 UB：A7 PV 部分和交接槽 [total_rows][head_dim] FP32（ExecuteEpilogue 落点、
        // V1 MergePvPartialRowsVF 消费，FLAG_PV_PARTIAL_FREE 读保护）
        const uint64_t pv_partial_bytes = static_cast<uint64_t>(total_rows) * head_dim * 4u;
        if (pv_partial_bytes > Vector1UbBudget::A7_PV_PARTIAL_BYTES) {
            return FootprintViolation::kUbV1A7Handoff;
        }

        // V0 UB：GM 输入 ping 槽按 max(total_rows, block_n) × head_dim FP16 暂存（Q 聚合行或 KV 分块行，
        // 偏移唯一定义 GM_INPUT_PING_OFFSET，校验槽容量即覆盖该消费契约）
        const uint64_t max_v0_rows = (total_rows > block_n) ? total_rows : block_n;
        const uint64_t v0_input_bytes = max_v0_rows * head_dim * 2u;
        if (v0_input_bytes > Vector0UbBudget::GM_INPUT_PING_BYTES) {
            return FootprintViolation::kUbV0Input;
        }

        return FootprintViolation::kNone;
    }
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_RESOURCE_HPP
