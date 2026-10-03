#ifndef SAGE_ATTN_FP8_ASCENDC_TILING_H
#define SAGE_ATTN_FP8_ASCENDC_TILING_H

#include <cstdint>
#include <cstddef>

/**
 * Host/Device 切换（本头文件被所有 ascendc/catlass 头包含，是 __aicore__ 的首个定义点）：
 * - bisheng -xasc 设备编译预定义 __aicore__ = __attribute__((cce_aicore))，下方 #ifndef 直接保留内建；
 * - KT 设备/主机单测定义 __CCE_KT_TEST__（或历史 __DAV_C220_*），属性展开为 [[]]；
 * - 纯宿主语法门（g++/clang++ 无任何宏）展开为空。
 * 此前此处为无条件空展开，会抢在 online_softmax.h 之前把 __aicore__ 污染为空，
 * 使 KT 单测路径丢失属性——已改为与其他头一致的完整切换范式。
 */
#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

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
constexpr uint32_t ATLAS_950_DIE_NUM = 2;              // Atlas 950PR 标配 2 个计算 Die
constexpr uint32_t ATLAS_950_GROUPS_PER_DIE = 4;       // Atlas 950PR 每 Die 包含 4 个 Core Group (全片 8-Group)
constexpr uint32_t ATLAS_950_CORES_PER_DIE = 16;       // Atlas 950PR 满配单 Die 16 个 AI Core
constexpr float S_FP8_OFFSET = 8.807354922057604f;      // log2(448.0f) P 矩阵 e4m3 偏置常数 (task/01_design 2.4)
constexpr float LN_448 = 6.104793232414985f;            // ln(448.0f) = S_FP8_OFFSET * INV_LOG2E（N7/R11-1 裁定后 448· 放大恒以乘法在指数之后，本常量保留为域关系注记）

/**
 * 共享数学与量化常量（全仓唯一定义点；quant_host.h / online_softmax.h /
 * quant_common.h 经 include + using 引用，禁止再复制）
 */
constexpr float LOG2E = 1.4426950408889634f;            // log2(e)
constexpr float INV_LOG2E = 0.6931471805599453f;        // 1.0f / log2(e) = ln(2)
constexpr float INT8_QUANT_MAX = 127.0f;                // INT8 对称量化上限
constexpr float SCALE_EPSILON = 1e-7f;                  // INT8 量化 scale 兜底微量 (amax/127 + 1e-7)
constexpr float FP8_E4M3_MAX = 448.0f;                  // FP8 e4m3fn 最大有限值 (448.0 字面量唯一定义点)
constexpr float SCALE_MAX_FP16_ACCUM = 2.25f;           // fp32+fp16 档 V 量化 scale_max (对齐开源)
constexpr float SCALE_MAX_FP32_ACCUM = FP8_E4M3_MAX;    // 其余档位 V 量化 scale_max (饱和至 e4m3 满量程)
constexpr float SCALE_ONE_TOLERANCE = 1e-6f;            // 缩放系数 (alpha/rescale) 与 1.0 比较的浮点容差
constexpr float SKIP_SCALE_TAU = 6.0f;                  // FlashMLA / Online Softmax 跳步缩放阈值 tau = 6.0 (唯一定义点)

/**
 * 两级累加归约周期缺省步长（tiling.k_step ∈ {8, 16, 32} 的缺省值；
 * AccumReduceController 兜底、孪生参考 fp16 子块累加均引用此处）
 */
constexpr uint32_t DEFAULT_K_STEP = 16;

/**
 * GM 暂存区布局（V0→V1 的 scale 桥接，2026-10-03 设备数据通路 A7 落地）：
 * Vector0 量化核产出的 scale 供 Vector1 softmax 核消费（两 Vector UB 互不可见，
 * L1 侧仅供 Cube 读），经 GetUserWorkspace 用户区按核分条带交接。host 侧
 * （op_host/aclnn host tiling）按 STAGE_CORE_STRIDE × core_num_used 预留，
 * kernel 侧以 GetBlockIdx（AIV 折半后 core_idx）× STAGE_CORE_STRIDE 寻址——
 * 布局唯一定义点即本组常量，禁止两侧各自推导。
 * 每条带 4 个区域，各 512B（128 float 行解析 scale）：
 *   [0,   512)   行解析 q_scale（V0 按 QScaleIndex 分组定标后逐行展开，gqa 聚合行序）
 *   [512, 1024)  块解析 k_scale ping 槽（V0 按 KScaleIndex 分组定标后逐 token 展开）
 *   [1024,1536)  块解析 k_scale pong 槽
 *   [1536,2048)  V per-channel scale（V0 全序列 amax 预扫产出，任务级一次）
 */
constexpr uint32_t STAGE_CORE_STRIDE = 2048;
constexpr uint32_t STAGE_Q_SCALE_OFFSET = 0;
constexpr uint32_t STAGE_K_SCALE_PING_OFFSET = 512;
constexpr uint32_t STAGE_K_SCALE_PONG_OFFSET = 1024;
constexpr uint32_t STAGE_V_SCALE_OFFSET = 1536;
constexpr uint32_t STAGE_SLOT_FLOATS = 128;               // 单区域容量（float 数）
constexpr uint32_t STAGE_SLOT_BYTES = STAGE_SLOT_FLOATS * sizeof(float);

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

    // 3. 数值字段 (12 字节)
    float sm_scale_log2e;     // Host 端折叠后的缩放因子: sm_scale * log2(e) (约 1.44269504)
    float v_scale_max;        // V 量化饱和上限 (fp32+fp16 为 2.25f, 其余为 448.0f)
    // 【I1 kernel-lane 契约增量，请勿回退删除】task/01_design.md 3.5 节冻结字段：
    // P 矩阵 e4m3 放大偏置 log2(448)≈8.807（语义见 2.4 节）。取值恒等于上方 S_FP8_OFFSET
    // 常量，入结构体仅为单测参数化；op_host 侧按常量填充即可。device 侧
    // op_kernel/sageattention2.cpp 的 ComputeSoftmaxProbability / ConvertLseToHost 出口依赖此字段。
    float s_fp8_offset;

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

    // 7. Die 调度与硬件亲和性字段 (16 字节)
    uint32_t enable_die_affinity;  // 是否启用 Die 亲和性多核调度 (0=关闭, 1=启用)
    uint32_t die_num;              // 芯片计算 Die 数量 (Atlas 950PR 双 Die 架构，值为 2)
    uint32_t cores_per_die;        // 每个计算 Die 包含的核心数 (core_num_used / die_num, 标配 16 或 14)
    uint32_t tasks_per_die;        // 每个计算 Die 分配的基础任务数 ((total_tasks + 1) / die_num)

    // 8. dtype 与补齐保留字段：输入位型分派（任务书 §3 支持 FP16/BF16）。
    // 编码：0 视作 fp16 缺省（消费方 (==2)?2:1 归一），1=float16, 2=bfloat16
    uint8_t input_dtype;  // 1=fp16（缺省）, 2=bf16
    uint8_t reserved[3];

    // 便捷访问辅助函数
    bool IsCausal() const noexcept {
        return is_causal != 0;
    }

    bool IsSmoothK() const noexcept {
        return smooth_k != 0;
    }

    bool IsSmoothV() const noexcept {
        return smooth_v != 0;
    }

    bool IsHND() const noexcept {
        return tensor_layout == LAYOUT_HND;
    }

    bool IsPerThread() const noexcept {
        return qk_quant_gran == QUANT_GRAN_PER_THREAD;
    }

    bool IsPerWarp() const noexcept {
        return qk_quant_gran == QUANT_GRAN_PER_WARP;
    }

};

// 静态断言验证结构体大小和对齐属性
static_assert(sizeof(SageAttention2TilingData) == 160, "SageAttention2TilingData size must be exactly 160 bytes");
static_assert(sizeof(SageAttention2TilingData) % 32 == 0, "SageAttention2TilingData size must be a multiple of 32 bytes");
static_assert(alignof(SageAttention2TilingData) == 32, "SageAttention2TilingData must be 32-byte aligned");

using TilingData = SageAttention2TilingData;

/**
 * DieTask 算法：将连续的虚拟任务映射到 Atlas 950PR 双 Die / 8-Group 拓扑
 * Die 0 (核 0 ~ cores/2-1) 亲和前半段任务 [0, total/2)
 * Die 1 (核 cores/2 ~ cores-1) 亲和后半段任务 [total/2, total)
 * 纯简洁整数运算与移位，保证微秒级发射时延与纳秒级内核解码
 *
 * @param task 虚拟任务索引 (wave * cores + core)
 * @param total 任务总数 (b * hkv * q_blks)
 * @param cores 参与调度的核心总数 (必须为偶数，如 32 或 28)
 * @return 物理亲和后的实际任务索引
 */
__aicore__ inline uint64_t DieTask(uint64_t task, uint64_t total, uint32_t cores) noexcept {
    if (cores < 2 || total < 2) {
        return task;
    }
    const uint32_t half = cores / 2;
    const uint64_t half_total = total / 2;
    return (task % cores / half) * half_total + (task / cores) * half + (task % half);
}

/**
 * DieTaskEnd 算法：计算当前核心在虚拟循环任务空间中的终止边界 (支持尾波次 partial wave)
 *
 * @param total 任务总数
 * @param core 当前核心物理索引 (GetBlockIdx)
 * @param cores 参与调度的核心总数
 * @param enable 是否启用 Die 亲和性调度
 * @return 当前核心的任务循环终止上界
 */
__aicore__ inline uint64_t DieTaskEnd(uint64_t total, uint32_t core, uint32_t cores, bool enable = true) noexcept {
    if (!enable || cores < 2 || total < 2 || (cores % 2 != 0)) {
        return total;
    }
    const uint32_t half = cores / 2;
    const uint32_t rank = core % half;
    const uint64_t local = total / 2;
    const uint64_t count = (local > rank) ? ((local - rank + half - 1) / half) : 0;
    return core + count * cores;
}

} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_TILING_H
