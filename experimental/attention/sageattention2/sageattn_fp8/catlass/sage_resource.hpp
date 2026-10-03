#ifndef SAGE_ATTN_FP8_CATLASS_SAGE_RESOURCE_HPP
#define SAGE_ATTN_FP8_CATLASS_SAGE_RESOURCE_HPP

#include <cstdint>
#include <cstddef>
#include "../ascendc/tiling.h"
#include "sage_tile.hpp"

namespace Catlass::SageAttn {

/**
 * 1C2V 硬件资源配额契约（Atlas 950PR 架构白皮书表 4-2）
 */
struct HardwareResourceQuota {
    // 物理硬件上限
    static constexpr uint32_t L1_TOTAL_BYTES   = sage_attn::L1_TOTAL_SIZE;        // 512 KB
    static constexpr uint32_t L0A_TOTAL_BYTES  = sage_attn::L0A_SIZE;             // 64 KB
    static constexpr uint32_t L0B_TOTAL_BYTES  = sage_attn::L0B_SIZE;             // 64 KB
    static constexpr uint32_t L0C_TOTAL_BYTES  = sage_attn::L0C_SIZE;             // 256 KB
    static constexpr uint32_t UB_PER_V_BYTES   = sage_attn::UB_TOTAL_SIZE_PER_V;  // 256 KB
    static constexpr uint32_t UB_TOTAL_BYTES   = sage_attn::UB_TOTAL_SIZE;        // 512 KB
};

/**
 * Vector 核心 UB 严格解耦划分契约 (task/01_design.md 3.2 节)
 */
struct Vector0UbBudget {
    // V0（量化 prologue 核）：负责 GM 输入暂存、INT8/FP8 量化、ND2NZ 摆放
    static constexpr uint32_t GM_INPUT_PING_OFFSET     = 0;
    static constexpr uint32_t GM_INPUT_PING_BYTES      = 48 * 1024;  // 48 KB

    static constexpr uint32_t GM_INPUT_PONG_OFFSET     = 48 * 1024;
    static constexpr uint32_t GM_INPUT_PONG_BYTES      = 48 * 1024;  // 48 KB

    static constexpr uint32_t QUANT_WORKSPACE_OFFSET   = 96 * 1024;
    static constexpr uint32_t QUANT_WORKSPACE_BYTES    = 48 * 1024;  // 48 KB

    static constexpr uint32_t NZ_OUTPUT_OFFSET         = 144 * 1024;
    static constexpr uint32_t NZ_OUTPUT_BYTES          = 96 * 1024;  // 96 KB (含每行 32B padding 防 bank conflict)

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
    // 握手 FLAG_DOT_FREE=13 / FLAG_PV_PARTIAL_FREE=12，sync_guard.h）
    static constexpr uint32_t A7_DOT32_OFFSET          = 96 * 1024;
    static constexpr uint32_t A7_DOT32_BYTES           = 64 * 1024;  // rows·block_n·4 ≤ 128·128·4
    static constexpr uint32_t A7_PV_PARTIAL_OFFSET     = 160 * 1024;
    static constexpr uint32_t A7_PV_PARTIAL_BYTES      = 64 * 1024;  // rows·d_pad·4 ≤ 128·128·4

    static constexpr uint32_t LSE_WORKSPACE_OFFSET     = 224 * 1024;
    static constexpr uint32_t LSE_WORKSPACE_BYTES      = 16 * 1024;  // 行 stats + logits 行暂存 + scale 条带副本

    static constexpr uint32_t TWO_LEVEL_REDUCE_OFFSET  = 240 * 1024;
    static constexpr uint32_t TWO_LEVEL_REDUCE_BYTES   = 16 * 1024;  // 两级累加归约缓冲（skip-scale 变体预留）

    // 行界常数（与 SageResourceManager::MAX_GQA_PACK_ROWS / host 分块域 block_n ∈ {64,128} /
    // d_pad ≤ 128 验收域同步；op_kernel kKernelMaxRows 同源，改动须双侧联动 + 台账回归）
    static constexpr uint32_t A7_REGION_ROW_MAX        = 128;
    static constexpr uint32_t A7_REGION_BLOCK_N_MAX    = 128;
    static constexpr uint32_t A7_REGION_HEAD_DIM_MAX   = 128;

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
    // 与 op_kernel/sageattention2.cpp 的 kKernelMaxRows 契约同步（唯二出现点，
    // 改动须双侧联动 + test_tiling_struct 台账用例回归）；Q_MAX_BYTES 的
    // 128x128 规格隐含同一上限。
    static constexpr uint32_t MAX_GQA_PACK_ROWS = 128;

    /**
     * 静态校验分块配置是否满足片上容量硬约束
     */
    static constexpr bool ValidateBudget(uint32_t m_per_core, uint32_t block_n, uint32_t head_dim) {
        // Q 矩阵大小 (INT8)
        uint32_t q_bytes = m_per_core * head_dim * sizeof(int8_t);
        if (q_bytes > L1DoubleBankContract::BANK0_SIZE) {
            return false;
        }

        // KV 单块大小 (INT8 或 FP8)
        uint32_t kv_single_block_bytes = block_n * head_dim * sizeof(int8_t);
        // 双缓冲 Ping-Pong 需要 2 个 KV 块
        if (kv_single_block_bytes * 2U > L1DoubleBankContract::BANK1_SIZE) {
            return false;
        }

        // L0C 矩阵乘中间累加大小 (INT32 或 FP32)
        uint32_t l0c_bytes = m_per_core * block_n * sizeof(float);
        if (l0c_bytes > HardwareResourceQuota::L0C_TOTAL_BYTES) {
            return false;
        }

        return true;
    }

    /**
     * 运行时资源台账检查（RES-01，host 发射前 fail-fast 的唯一判据）：
     * total_rows 为 GQA 聚合后的 M 打包宽（gqa_ratio * m_per_core，即 kernel 侧
     * TotalRows），不是 m_per_core 本身——O 累加器、S 双槽、L0C 均按聚合宽计。
     * 纯 constexpr 整数算术，host 热路径零堆分配，微秒级发射。
     * 各区域容量唯一定义点：L1DoubleBankContract / Vector0UbBudget /
     * Vector1UbBudget / HardwareResourceQuota（本文件与 sage_tile.hpp）。
     */
    static constexpr FootprintViolation InspectTilingPlan(
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
        // EmitL0CToUbRaw 落点、V1 DequantMaskLogitsRowVF 消费，FLAG_DOT_FREE 读保护）
        const uint64_t dot32_bytes = static_cast<uint64_t>(total_rows) * block_n * sizeof(int32_t);
        if (dot32_bytes > Vector1UbBudget::A7_DOT32_BYTES) {
            return FootprintViolation::kUbV1A7Handoff;
        }

        // V1 UB：A7 PV 部分和交接槽 [total_rows][head_dim] FP32（ExecuteEpilogue 落点、
        // V1 MergePvPartialRowVF 消费，FLAG_PV_PARTIAL_FREE 读保护）
        const uint64_t pv_partial_bytes = static_cast<uint64_t>(total_rows) * head_dim * 4u;
        if (pv_partial_bytes > Vector1UbBudget::A7_PV_PARTIAL_BYTES) {
            return FootprintViolation::kUbV1A7Handoff;
        }

        // V0 UB：GM 输入 ping 槽按 [block_n][head_dim] FP16 暂存（pong 槽等宽，
        // 偏移由 Vector0UbBudget 常量锁定，校验单槽容量即可覆盖双槽契约）
        const uint64_t v0_input_bytes = static_cast<uint64_t>(block_n) * head_dim * 2u;
        if (v0_input_bytes > Vector0UbBudget::GM_INPUT_PING_BYTES) {
            return FootprintViolation::kUbV0Input;
        }

        return FootprintViolation::kNone;
    }

    /**
     * 布尔包装：host tiling / 单测的直接判据
     */
    static constexpr bool ValidateTilingPlan(
        uint32_t total_rows, uint32_t block_n, uint32_t head_dim)
    {
        return InspectTilingPlan(total_rows, block_n, head_dim) == FootprintViolation::kNone;
    }
};

} // namespace Catlass::SageAttn

#endif // SAGE_ATTN_FP8_CATLASS_SAGE_RESOURCE_HPP
