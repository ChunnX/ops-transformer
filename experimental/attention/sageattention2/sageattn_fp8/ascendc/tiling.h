#ifndef SAGE_ATTN_FP8_ASCENDC_TILING_H
#define SAGE_ATTN_FP8_ASCENDC_TILING_H
#pragma once

#include <cstddef>
#include <cstdint>

// 官方 CATLASS 容量常量一致性锁：本头文件保持零依赖基座（被 quant 裸编译等无
// CATLASS -I 场景包含），容量取白皮书字面量；与官方 Catlass::Arch::Ascend950 的
// 一致性由 test_tiling_struct.cpp（含 catlass 的 TU）直锁官方源（L1 字面量断言 +
// HardwareResourceQuota::L0C_TOTAL_BYTES == Arch::Ascend950::L0C_SIZE 唯一映射，
// L0A/L0B 无生产判据不设 canary，见 sage_resource.hpp 配额注记）。

/**
 * Host/Device 切换（本头文件被所有 ascendc/catlass 头包含，是 __aicore__ 的首个定义点）：
 * - bisheng -xasc 设备编译预定义 __aicore__ = __attribute__((cce_aicore))，下方 #ifndef 直接保留内建；
 * - 纯宿主门禁（BiSheng 宿主模式）展开为空。
 */
#ifndef __aicore__
#define __aicore__
#endif

#ifndef __host__
#define __host__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif

/**
 * 编译域判定宏（全仓唯一定义点，其余头文件禁止再定义或重复守卫）：
 * SAGE_ATTN_DEVICE_COMPILE=1 当且仅当 bisheng -xasc 设备编译（编译器预定义 __CCE__），
 * 此时 kernel_operator.h 与 CATLASS 设备组件可用、跨核同步为真实硬件原语；
 * =0 为 BiSheng 宿主模式（宿主单测/孪生执行器），kernel 侧 API 落下方桩，
 * 执行路径为宿主串行仿真（V0 → Cube → V1 顺序直跑）。
 * （历史 SAGE_ATTN_KERNEL_API 与 __CCE_KT_TEST__ 单测桩域已随 2026-10-04 死代码
 * 清扫合并删除：构建已收敛到毕昇容器双路径，两宏取值与 DEVICE_COMPILE 恒等。）
 */
#ifndef SAGE_ATTN_DEVICE_COMPILE
#if defined(__CCE__)
#define SAGE_ATTN_DEVICE_COMPILE 1
#else
#define SAGE_ATTN_DEVICE_COMPILE 0
#endif
#endif



namespace sage_attn {

/**
 * 片上存储容量与硬件拓扑常量（白皮书表 4-2 与第 3 节硬件规格）。
 * 本头文件为全仓零依赖基座（quant 裸编译等无 CATLASS -I 场景亦包含），容量取
 * 白皮书字面量；与官方 Catlass::Arch::Ascend950（L1 512KB / L0C 256KB）的一致性由
 * test_tiling_struct.cpp 直锁官方源（L1 字面量断言 + HardwareResourceQuota 的 L0C
 * 唯一映射 static_assert），官方演进漂移在测试编译期显式暴露；L0A/L0B 无生产判据，
 * 不设配额别名与官方源 canary（sage_resource.hpp 同口径）。
 * UB 不取官方 Arch::Ascend950::UB_SIZE（248KB，官方样例工程口径）：本算子按任务书
 * 验收域（Atlas 950PR 白皮书表 4-2 单 Vector 核 256KB）冻结 256KB——两值为不同口径
 * 而非换算关系（sage_resource.hpp HardwareResourceQuota 同口径）。
 */
constexpr uint32_t L1_TOTAL_SIZE = 512 * 1024;         // L1 总容量 512KB
constexpr uint32_t L1_BANK_SIZE = L1_TOTAL_SIZE / 2;   // 单 Bank 容量 256KB（L1 对半分 Bank 的拓扑事实）
constexpr uint32_t L1_BANK0_OFFSET = 0;                // Bank 0 起始偏移 (0~256KB, Q 与 Q scale)
constexpr uint32_t L1_BANK1_OFFSET = L1_BANK_SIZE;     // Bank 1 起始偏移 (256~512KB, KV 与 KV scale)

constexpr uint32_t UB_TOTAL_SIZE_PER_V = 256 * 1024;   // 单 Vector 核心 UB 容量 256KB (V0 256KB, V1 256KB)

// L0A/L0B/L0C 容量不在本头重复映射：官方 Arch::Ascend950 → 本算子配额的唯一映射点
// 为 sage_resource.hpp HardwareResourceQuota（L0A/L0B 64KB、L0C 256KB），消费方直引。
// 原 ATLAS_950_AIV_NUM 拓扑常量零调用方，已随 2026-10-03 死代码清扫删除（核数由
// tiling.core_num_used 运行时携带，host 推导见 op_host/aclnn host tiling）；本文件
// STARS_* Group 拓扑常量为 M6 Die 亲和契约的编译期消费载体（下方 static_assert 与
// tests/test_tiling_struct.cpp Test 6 直锁白皮书字面量）。
constexpr uint32_t ATLAS_950_AIC_NUM = 32;             // Atlas 950PR 满配 32 个 AI Core
constexpr uint32_t ATLAS_950_DIE_NUM = 2;              // Atlas 950PR 标配 2 个计算 Die

/**
 * STARS 2.0 硬件调度拓扑（白皮书第 4.4 节：AI Core 分为最多 8 个 Group，支持按 Die
 * 进行亲和性调度；注意白皮书「Group 内核数可软件配置」为系统级资源池语境，非算子
 * tiling 编程面）：Atlas 950PR 满配网格 2 Die × 每 Die 4 Group × 每 Group 4 核为
 * 本仓推断拆分（白皮书原文仅约束上限 8 Group 与按 Die 亲和，未规定均分），仅被
 * 下方 static_assert 与契约测试消费，无运行时行为。
 * 接口考证结论（CANN 9.2.0-beta.1，容器内 ${ASCEND_HOME_PATH}/include，grep 全树）：
 * host tiling 层无 STARS Group 直控公开 API——
 * - acl/acl_rt.h:978 aclrtHacType 仅暴露 ACL_RT_HAC_TYPE_STARS 硬件队列类型枚举；
 * - pto/comm/async/sdma/sdma_types.hpp:23 仅 STARS SQE 类型常量（kRtStarsSqeTypeSdma=11），
 *   sdma_workspace_manager.hpp:273 仅 rtGetDeviceInfo(kInfoTypePhyDieId=19) 物理 Die 号
 *   查询面（host 流级，非 per-op tiling 面）；
 * - 全 include 树无 HSCB 关键词（HSCB 为纯硬件总线，无软件编程面）；gert 侧任务下发
 *   仅 SetBlockDim/SetScheduleMode（exe_graph/runtime/tiling_context.h:227/254）；
 * - ./git/catlass 无 Die 亲和/Group 调度封装；STARS 仅现于 tools/tuner/src/profiler.cpp:17,20,190-192,225
 *   的 prof_drv_start 性能采集通道（STARS_ENABLE_FLAG/CHANNEL_STARS_SOC_LOG_BUFFER），非 per-op 调度面。
 * 故本算子 Die 亲和分组按仓内既有惯例落地：TilingData 第 7 段四字段
 * （enable_die_affinity/die_num/cores_per_die/tasks_per_die，布局冻结 160 字节，不新增
 * 字段）显式编码 2 个物理 Die Group 配置（enable_die_affinity=1 即下发 Die 亲和调度），
 * 核内按 GetBlockIdx 经 DieTask/DieTaskEnd（下方唯一定义点）解码 Batch/Head 分块到
 * Die 的闭式映射，运行时 STARS 调度器按 blockDim 波次把任务下沉至各 Die 硬件队列
 * 纳秒级发射。
 */
constexpr uint32_t STARS_MAX_GROUP_NUM = 8;   // 全片 AI Core 硬件 Group 数上限（白皮书 4.4）
constexpr uint32_t STARS_GROUPS_PER_DIE = STARS_MAX_GROUP_NUM / ATLAS_950_DIE_NUM;   // 每 Die 4 个 Group
constexpr uint32_t STARS_CORES_PER_GROUP = ATLAS_950_AIC_NUM / STARS_MAX_GROUP_NUM;  // 每 Group 4 核
static_assert(STARS_GROUPS_PER_DIE * ATLAS_950_DIE_NUM == STARS_MAX_GROUP_NUM,
              "STARS group topology must tile exactly into the physical dies");
static_assert(STARS_CORES_PER_GROUP * STARS_MAX_GROUP_NUM == ATLAS_950_AIC_NUM,
              "STARS group topology must tile exactly into the full AI core grid");

// 单任务 GQA 行打包上限（kernel 行窗口直读本常量 / host total_rows 校验 / InspectTilingPlan
// 三方同源唯一点，sage_resource.hpp 的 MAX_GQA_PACK_ROWS / A7_REGION_ROW_MAX 以
// static_assert 锁定与本值相等）：单任务打包 min(gqa_ratio, MAX_PACK_ROWS / m_per_core) 个 Q 头，
// gqa_ratio 超出时按 chunk 拆多任务（如 8:1 × m_per_core=32 → 2 chunk × 4 头 × 32 行）
constexpr uint32_t MAX_PACK_ROWS = 128;

// GQA 头数上限（gqa_ratio ≤ 8；官方 500 用例套件最大 8:1）：
// SimMemory/设备 UB 的 Q scale 槽按 8 头配额冻结，host 校验（aclnn host tiling
// gqa_ratio > MAX_GQA_RATIO 拒绝）与本常量同源
constexpr uint32_t MAX_GQA_RATIO = 8;
constexpr float S_FP8_OFFSET = 8.807354922057604f;      // log2(448.0f) P 矩阵 e4m3 偏置常数 (task/01_design 2.4)

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

/**
 * 两级累加归约周期缺省步长（tiling.k_step ∈ {8, 16, 32} 的缺省值；
 * 消费方：aclnn host tiling 缺省 k_step、kernel 0 归一兜底（防死循环）、
 * 孪生参考 twin::AccumulatePvRow 的 fp16 子块累加。原 AccumReduceController
 * 兜底消费已随 2026-10-03 死代码清扫删除，见 tests/test_tiling_struct.cpp 清理记录）
 */
constexpr uint32_t DEFAULT_K_STEP = 16;

/**
 * Q/K INT8 scale 槽区每头步长（SimMemory 与设备 UB 暂存的槽布局常量；
 * 槽寻址 = 全局头序 × 步长 + QScaleIndex/KScaleIndex(token)，索引唯一定义点
 * 在 quant/quant_common.h）：
 * - Q：4096 = QScaleCount(16384, per_thread)——P-07 域 sq ≤ 16384 不越界；
 *   此前步长 2048 在 per_thread sq > 8192 时越头污染 scale（P0 级精度隐患修复）
 * - K：2048 = KScaleCount(32768, per_thread)——sk ≤ 32768 域
 * 容量域经 quant_common.h 的 static_assert 与 QScaleCount/KScaleCount 锁定。
 */
constexpr uint32_t kQScaleSlotStride = 4096;
constexpr uint32_t kKScaleSlotStride = 2048;

/**
 * GM 暂存区布局（V0→V1 的 scale 桥接，2026-10-03 设备数据通路 A7 落地，DEFECT-3 修复）：
 * Vector0 量化核产出的 scale 供 Vector1 softmax 核消费（两 Vector UB 互不可见，
 * L1 侧仅供 Cube 读），经 GetUserWorkspace 用户区按核分条带交接。host 侧
 * （op_host/aclnn host tiling）按 STAGE_CORE_STRIDE × core_num_used 预留，
 * kernel 侧以 core_idx × STAGE_CORE_STRIDE 寻址——布局唯一定义点即本组常量。
 * STAGE_CORE_STRIDE 设为 64KB (65536 字节)，每个核预留充裕空间；
 * 为每个 K block 提供独立的 scale 槽位 (k % 126u)，彻底避免覆写竞争，
 * 无需硬件不支持的 V1→V0 同步信号：
 *   [0,    512)   行解析 q_scale（V0 按 QScaleIndex 分组定标后逐行展开，gqa 聚合行序）
 *   [512,  1024)  V per-channel scale（V0 全序列 amax 预扫产出，任务级一次）
 *   [1024, 65536) 块解析 k_scale 槽区（126 个独立 slot，各 512B，偏移 STAGE_K_SCALE_BASE_OFFSET + (k % 126u) * 512）
 */
constexpr uint32_t STAGE_CORE_STRIDE = 65536;             // 单核 GM scale 交接条带容量 64KB
constexpr uint32_t STAGE_Q_SCALE_OFFSET = 0;              // Q scale 起始字节偏移 [0, 512)
constexpr uint32_t STAGE_V_SCALE_OFFSET = 512;            // V scale 起始字节偏移 [512, 1024)
constexpr uint32_t STAGE_K_SCALE_BASE_OFFSET = 1024;      // K scale 独立 slot 基址偏移 [1024, 65536)
constexpr uint32_t STAGE_K_SCALE_NUM_SLOTS = 126;         // 126 个独立 K scale slot，避免轮转覆盖
constexpr uint32_t STAGE_SLOT_FLOATS = 128;               // 单区域容量（float 数，512B）
constexpr uint32_t STAGE_SLOT_BYTES = STAGE_SLOT_FLOATS * sizeof(float);  // 单区域字节容量（512B，K scale 环形槽步进）

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
    // （原 s_fp8_offset 布局冻结字段已删除：448· 放大恒经乘法启用（twin::SoftmaxProbCode），
    // LSE 出口恒以 S_FP8_OFFSET 常量补扣（唯一定义本文件），该字段从不参与运行时数值，
    // 其 4 字节已并入尾部保留域，160 字节 TilingData 布局契约不变。）

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
    // 尾部保留域 (7 字节)：吸收原 causal_early_exit（kernel 从未读取，host 写值无消费者）
    // 与 s_fp8_offset（kernel 直读 S_FP8_OFFSET 常量）删除后的冗余位，
    // 160 字节 TilingData 布局契约（ctypes 镜像断言）不变。
    uint8_t reserved[7];

    // 便捷访问辅助函数（仅 host 侧使用；设备核内因 bisheng 对 const __gm__ this
    // 的成员函数限制，一律直接读字段——见 op_kernel/sageattention2.cpp Init 注释）
    [[nodiscard]] bool IsSmoothV() const noexcept {
        return smooth_v != 0;
    }
};

// 静态断言验证结构体大小和对齐属性
static_assert(sizeof(SageAttention2TilingData) == 160, "SageAttention2TilingData size must be exactly 160 bytes");
static_assert(sizeof(SageAttention2TilingData) % 32 == 0, "SageAttention2TilingData size must be a multiple of 32 bytes");
static_assert(alignof(SageAttention2TilingData) == 32, "SageAttention2TilingData must be 32-byte aligned");

/**
 * DieTask 闭式映射 φ：将连续的虚拟任务索引映射到 Atlas 950PR 双 Die / 8-Group 拓扑
 * （STARS 2.0 按 Die 亲和，拓扑常量见上方 STARS_MAX_GROUP_NUM 组）：
 *   Die 0 (核 0 ~ cores/2-1) 亲和前半段任务 [0, total/2)
 *   Die 1 (核 cores/2 ~ cores-1) 亲和后半段任务 [total/2, total)
 * 波次展开 φ(v) = (⌊(v mod C)/H⌋)·(T/2) + ⌊v/C⌋·H + (v mod H)，C=cores，H=C/2；
 * T 为偶数时 φ 是活跃核域到 [0, T) 的双射（编译期采样锁值见下方 static_assert，
 * 生产域运行时穷举见 tests/test_tiling_struct.cpp Test 6）；T 为奇数时尾任务 T-1
 * 无核执行，属契约外危险域（host enable 判据以偶性硬门禁排除，唯一定义点
 * aclnn/sage_attn_host_tiling.h 第 10 步）。constexpr 纯整数运算与移位，host 判据
 * 与核内纳秒级解码同式同源。
 *
 * @param task 虚拟任务索引 (wave * cores + core)
 * @param total 任务总数 (b * hkv * gqa_chunk * q_blks，4D 口径与 DecodeTask 解码一致；
 *         唯一生产消费方 op_kernel Process 的 total_tasks 恒含 GQA chunk 因子)
 * @param cores 参与调度的核心总数 (必须为偶数，如 32 或 28)
 * @return 物理亲和后的实际任务索引
 */
[[nodiscard]] __aicore__ __forceinline__ constexpr uint64_t DieTask(uint64_t task, uint64_t total, uint32_t cores) noexcept {
    if (cores < 2 || total < 2) {
        return task;
    }
    const uint32_t half = cores / 2;
    const uint64_t half_total = total / 2;
    return (task % cores / half) * half_total + (task / cores) * half + (task % half);
}

/**
 * DieTaskEnd 闭式映射：计算当前核心在虚拟循环任务空间中的终止边界 (支持尾波次 partial wave)
 * 波次计数 ⌈(T/2 − rank)/H⌉（rank = core mod H 为本 Die 内名次），保证两 Die 各核
 * 任务数相差不超过 1 且全体活跃核对 [0, T) 双射覆盖；enable=false / 单核 / 奇数核 /
 * 极小任务域恒退化为 total（等价单组全量循环，单 Die 环境与老用例零行为差异）。
 *
 * @param total 任务总数
 * @param core 当前核心物理索引 (GetBlockIdx)
 * @param cores 参与调度的核心总数
 * @param enable 是否启用 Die 亲和性调度 (tiling.enable_die_affinity 单一控制位)
 * @return 当前核心的任务循环终止上界
 */
[[nodiscard]] __aicore__ __forceinline__ constexpr uint64_t DieTaskEnd(uint64_t total, uint32_t core, uint32_t cores, bool enable = true) noexcept {
    if (!enable || cores < 2 || total < 2 || (cores % 2 != 0)) {
        return total;
    }
    const uint32_t half = cores / 2;
    const uint32_t rank = core % half;
    const uint64_t local = total / 2;
    const uint64_t count = (local > rank) ? ((local - rank + half - 1) / half) : 0;
    return core + count * cores;
}

// Die 亲和闭式映射编译期契约（constexpr 自锁；生产域运行时穷举与 (b,h) 块整对齐
// 定理实证见 tests/test_tiling_struct.cpp Test 6）：满配 32 核 / T=64 双 Die 半分
// 双射采样 + partial wave 边界 + 退化恒等域
static_assert(DieTask(0, 64, 32) == 0 && DieTask(15, 64, 32) == 15,
              "Die 0 wave 0 must map identity into the first half [0, 32)");
static_assert(DieTask(16, 64, 32) == 32 && DieTask(31, 64, 32) == 47,
              "Die 1 wave 0 must map into the second half [32, 64)");
static_assert(DieTask(32, 64, 32) == 16 && DieTask(47, 64, 32) == 31,
              "Die 0 wave 1 must stay inside the first half [0, 32)");
static_assert(DieTask(48, 64, 32) == 48 && DieTask(63, 64, 32) == 63,
              "Die 1 wave 1 must stay inside the second half [32, 64)");
static_assert(DieTaskEnd(64, 0, 32, true) == 64 && DieTaskEnd(64, 16, 32, true) == 80,
              "full wave domain must schedule exactly 2 waves per core");
static_assert(DieTaskEnd(16, 8, 32, true) == 8,
              "partial wave must retire Die 0 cores beyond the local task count");
static_assert(DieTaskEnd(16, 16, 32, true) == 48,
              "partial wave must keep the Die 1 first-rank core active");
static_assert(DieTask(7, 0, 32) == 7 && DieTask(7, 1, 32) == 7 && DieTask(7, 16, 1) == 7,
              "single-die/single-task domain must degenerate to identity");
static_assert(DieTaskEnd(16, 3, 32, false) == 16 && DieTaskEnd(16, 3, 31, true) == 16 &&
              DieTaskEnd(1, 0, 32, true) == 1,
              "disabled/odd-core/tiny-total domains must degenerate to the full loop bound");

} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_TILING_H
