// 2026-10-03 自 sageattn_fp8/ascendc/sage_attn_kernel.cpp 物理迁移至本路径
// （CANN opc autogen 打包约定：优先解析 <op>/op_kernel/<opFile>.cpp，布局 A；
// sageattn_fp8/ascendc/ 仍保留 tiling/softmax/accum/sync 等头文件与宿主孪生实现）。

// 注意：本文件位于 op_kernel/，包含 ascendc/ 头必须用相对路径——
// 裸 "tiling.h" 会按 -I 顺序误解析到 CANN 的 adv_api/matmul/tiling.h
//（bisheng 编译门的 -W#pragma-message 曾捕获该错位）。
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/ascendc/online_softmax.h"
#include "../sageattn_fp8/ascendc/accum_reduce.h"
#include "../sageattn_fp8/ascendc/sync_guard.h"

#include "../sageattn_fp8/catlass/sage_tile.hpp"
#include "../sageattn_fp8/catlass/sage_resource.hpp"
#include "../sageattn_fp8/catlass/sage_qk_mmad.hpp"
#include "../sageattn_fp8/catlass/sage_pv_mmad.hpp"
#include "sage_device_vf.h"

// 量化/转换原语唯一定义点（quant_common.h 自 2026-10-03 起为 device-safe 实现，
// 不依赖 std::floor/std::nearbyint，设备分支可安全 include）：
// RoundHalfAwayFromZero / RoundNearestEven / FloatToE4m3fn / E4m3fnToFloat /
// Fp16ToFp32 / Fp32ToFp16 / QScaleIndex / KScaleIndex / V_PERM_* / 分组常量。
#include "../quant/quant_common.h"

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <memory>

// KT/宿主标量基线（Step-C）唯一真值路径：QK 反量化 logit、块内 exp2 和、P 量化与
// PV 行累加均经 twin_reference.h 单点 include 复用（设计决策 5：语义公式禁止在内核
// 内复制第二份）。twin_reference.h 携带宿主 STL 载体（std::vector），设备分支
// （bisheng -xasc，SAGE_ATTN_DEVICE_COMPILE=1）不引入——七函数的数值实现位于宿主/KT
// 分支，设备侧维持流水结构桩，向量化实现留给后续 pass。
#if !SAGE_ATTN_DEVICE_COMPILE
#include "twin_reference.h"
#endif
/**
 * 设备与宿主分支切换实现：
 * 以 __CCE__（bisheng 设备编译恒定义，宿主 g++/clang++ 恒不定义）判定设备分支。
 */
#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#include "kernel_operator.h"
#if defined(__CCE__) && !defined(__CCE_KT_TEST__)
// canonical 注册三件套：正式出包由 ops-transformer 构建体系注入同义宏；
// 独立 bisheng 编译门（ci/kernel_compile.sh）经此处 #ifndef 定义补齐，
// 语义对齐 cmake/scripts/utest/gen_tiling_data_stub.py 的构建体系形态。
#ifndef REGISTER_TILING_DEFAULT
#define REGISTER_TILING_DEFAULT(tiling_struct)
#endif
#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg)                       \
    tiling_struct tiling_data;                                                                    \
    do {                                                                                          \
        for (uint32_t sage_tiling_i = 0; sage_tiling_i < sizeof(tiling_struct); ++sage_tiling_i) {\
            reinterpret_cast<uint8_t *>(&tiling_data)[sage_tiling_i] = tiling_arg[sage_tiling_i]; \
        }                                                                                         \
    } while (0)
#endif
#ifndef GetUserWorkspace
#define GetUserWorkspace(workspace) (workspace)
#endif
#endif
using namespace AscendC;
#else
#ifndef __global__
#define __global__
#endif
#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif
#ifndef KERNEL_TASK_TYPE_DEFAULT
#define KERNEL_TASK_TYPE_DEFAULT(x)
#endif
#ifndef KERNEL_TYPE_MIX_AIC_1_2
#define KERNEL_TYPE_MIX_AIC_1_2 0
#endif
#ifndef ASCEND_IS_AIC
#define ASCEND_IS_AIC true
#endif
#ifndef ASCEND_IS_AIV
#define ASCEND_IS_AIV false
#endif

// 宿主机或本地单测桩函数定义。
// 回退桩带 __aicore__ 属性，确保在 [[aicore]] 属性域内安全调用。
__aicore__ inline uint32_t GetBlockIdx() { return 0; }
__aicore__ inline uint32_t GetSubBlockIdx() { return 0; }
#endif

/**
 * 核内域判定宏：bisheng 设备分支中 ASCEND_IS_AIC 展开为编译期谓词 constexpr(...)，
 * 官方调用范式为无括号 `if ASCEND_IS_AIC {`（内置算子 geglu_3510_impl.h 同款）；
 * 宿主/KT 分支展开为普通布尔，按 C 风格加括号。统一经 SAGE_ATTN_IS_AIC/AIV 书写。
 */
#if defined(__CCE__) && !defined(__CCE_KT_TEST__)
#define SAGE_ATTN_IS_AIC ASCEND_IS_AIC
#define SAGE_ATTN_IS_AIV ASCEND_IS_AIV
#else
#define SAGE_ATTN_IS_AIC (ASCEND_IS_AIC)
#define SAGE_ATTN_IS_AIV (ASCEND_IS_AIV)
#endif

namespace sage_attn {

// MTE/V 管线序助手（收敛轮 I/I-2/I-3 单点定义；事件 ID 0 与文件内既有
// V_S(0)/FIX_S(0) 风格一致——即时自配对、无在途交叠）：
// - SyncMte2ToV：GM→UB DataCopy（MTE2 写）后、VF/标量读（V 管）前调用；
// - SyncVToMte3：VF/标量写（V 管）后、UB→L1/GM DataCopy（MTE3 读）前调用；
// - SyncMte3ToV：UB→GM DataCopy（MTE3 读）后、同槽被下一轮 VF 写复用前调用
//   （DEFECT-I-1：PipeBarrier<PIPE_V> 不排 MTE3 在途读）；
// - SyncVToMte2（R22 回边）：VF 读毕（V 管）后、同槽被下一轮 GM→UB DataCopy
//   （MTE2 写）复用前调用——MTE2 重写不 await V 管在途读，须显式事件对。
#if SAGE_ATTN_DEVICE_COMPILE
__aicore__ inline void SyncMte2ToV() {
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
}
__aicore__ inline void SyncVToMte3() {
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
}
__aicore__ inline void SyncMte3ToV() {
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
}
__aicore__ inline void SyncVToMte2() {
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
}
#else
inline void SyncMte2ToV() {}
inline void SyncVToMte3() {}
inline void SyncMte3ToV() {}
inline void SyncVToMte2() {}
#endif


// L1 数值追踪打点（task/07_profile.md 三级调试法 L1）：仅调试构建定义
// SAGE_ATTN_TRACE=1 时启用，性能构建零开销。
#if defined(SAGE_ATTN_TRACE) && !defined(__CCE__)
#include <cstdio>
#define SAGE_TRACE(...) std::printf(__VA_ARGS__)
#else
#define SAGE_TRACE(...)
#endif

// GQA 聚合后的窗口行上限（m_per_core 为 64 的倍数、gqa_ratio ≤ 8 的验收域内恒满足；
// 常量不随 SimMemory 分支门控，row_states_ 成员在设备分支亦需该界）
constexpr uint32_t kKernelMaxRows = 128;

__aicore__ inline uint32_t SageFastMin(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

// A7 设备态 UB 布局（V1 256KB 地址空间，task/12_device_data_path.md；R17-2 重排定案）：
// 区界/互斥/封顶的唯一定义点为 sage_resource.hpp Vector1UbBudget（static_assert 链
// 编译期锁定两两不相交且 ≤ 256KB，512B 对齐），本文件仅保留调用点别名与 V1 行级
// 暂存子槽——重排后 S pong 槽（≤96KB）不再被 Cube dot32 裸搬（96KB 起）踩踏：
//   O_ACCUM [0,64K) | S 双槽 [64K, ≤96K) | dot32 [96K,160K) | pv_partial [160K,224K)
//   | LSE 区 [224K,240K)：行 stats(+0)/logits 行(+512)/ks 行(+1K)/qs 行(+1.5K)/vs(+2K)
//   | TWO_LEVEL_REDUCE [240K,256K)
constexpr uint32_t SAGE_UB_DOT32_OFFSET = Catlass::SageAttn::Vector1UbBudget::A7_DOT32_OFFSET;
constexpr uint32_t SAGE_UB_PV_PARTIAL_OFFSET = Catlass::SageAttn::Vector1UbBudget::A7_PV_PARTIAL_OFFSET;
constexpr uint32_t SAGE_UB_V1_STATS_OFFSET = Catlass::SageAttn::Vector1UbBudget::LSE_WORKSPACE_OFFSET;
constexpr uint32_t SAGE_UB_V1_LOGITS_ROW_OFFSET = SAGE_UB_V1_STATS_OFFSET + 512;
constexpr uint32_t SAGE_UB_V1_KS_ROW_OFFSET = SAGE_UB_V1_STATS_OFFSET + 1024;
constexpr uint32_t SAGE_UB_V1_QS_ROW_OFFSET = SAGE_UB_V1_STATS_OFFSET + 1536;
constexpr uint32_t SAGE_UB_V1_VS_CHAN_OFFSET = SAGE_UB_V1_STATS_OFFSET + 2048;
static_assert(SAGE_UB_V1_VS_CHAN_OFFSET + 512u <= Catlass::SageAttn::Vector1UbBudget::LSE_WORKSPACE_OFFSET +
                  Catlass::SageAttn::Vector1UbBudget::LSE_WORKSPACE_BYTES,
              "A7 V1 row scratch exceeds LSE workspace region");

#if !SAGE_ATTN_DEVICE_COMPILE
/**
 * KT/宿主标量基线暂存区（SimMemory）：以固定大小数组成员承载真实硬件的 L1/UB/L0 分域，
 * 偏移约定与 tiling/sage_resource.hpp 契约一致：
 * - l1_mem：Q（tiling.l1_q_offset 起，[GQA 行][d] INT8）、KV ping/pong 槽
 *   （L1DoubleBankContract::KV_PING/PONG_OFFSET，槽首 K [token][d] INT8 + Data/Scale
 *   相邻的块 scale，槽内 KV_V_SUB_OFFSET 起 V [d][block_n] e4m3 16-token permute）；
 * - ub_mem：S(P) 双缓冲槽（SoftmaxSlotOffset = ub_softmax_offset + 奇偶·tile_bytes，
 *   tile_bytes = GQA 聚合行数 × block_n，与 Cube 侧 SagePvMmad::LoadL0A 的
 *   ub_softmax_offset 读取约定一致）；
 * - o_accum：V1 UB O 主累加器（[GQA 行][d_pad] fp32，Vector1UbBudget::O_ACCUM 契约归属 V1）；
 * - logits：当前拍反量化 logit（[GQA 行][block_n]，exp2 域，含 causal 掩码哨兵 -1e30）；
 * - q/k/v_scales：本任务 (b, hkv) 的量化 scale（q 按 GQA 头分段 [g*2048+组号]）；
 * - lse_corr：smooth_k 的行级 lse_correction（q_i · km，[GQA 行]）。
 * GQA 行 = g * cur_q_len + i（g 为 KV 头内 Q 头序号），上限 kKernelMaxRows。
 */
struct SimMemory {
    uint8_t l1_mem[512 * 1024]{0};
    uint8_t ub_mem[512 * 1024]{0};
    int32_t l0c_qk[128 * 128]{0};
    float o_accum[128 * 128]{0.0f};
    float logits[128 * 128]{0.0f};
    float q_scales[8 * 2048]{0.0f};
    float k_scales[2048]{0.0f};
    float v_scales[256]{0.0f};
    float lse_corr[128]{0.0f};
};
#endif

/**
 * SageAttention2 1C2V 设备核执行器
 * 核心技术规范：
     * 1. 1C2V 硬件编排：1 个 Cube Core 搭配 2 个 Vector Core (V0 量化 prologue 核，V1 Softmax 与 LSE 核)
     * 2. 延迟一拍流水：Cube 与 Vector 双域交错掩盖计算与搬运延时
     * 3. GQA 头聚合：当 Hq > Hkv 时沿 M 维聚合至 64 的倍数，一份 L1 KV 被多个 Q head 连续复用
     * 4. 64 位 GM 寻址：所有显存指针与 stride 计算强转 int64_t，杜绝长序列整数溢出
     * 5. 跨核 Flag 同步保序：严格遵循生产者在前、消费者在后，且每个任务域内 set/wait 1:1 配对
     *    （进域预置"消费者已归还"许可、退域回收残余许可，防跨任务波次残留死锁，task/02_code.md B-2）
     *
     * 数值架构契约（Step-C 标量基线，与 twin_reference.h RunSequential 唯一真值逐位一致，
     * 数值验证入口为 tests/test_kernel_twin.py（宿主串行分支 + 孪生执行器对 golden 25 桶
     * L-A/L-B 交叉验证，即本文件宿主分支的 golden 级回归）、tests/test_golden_precision.py
     * （GPU golden 对标）与 tests/test_function.py（功能用例））：
 * a. 逐行 Softmax：OnlineSoftmaxState 按逻辑行（r = g*cur_q_len + i）维护，行 max/行分母/
 *    行 rescale/行 LSE/行 lse_correction 全部独立——softmax 是行级语义；
 * b. P 逐块锚定：每块以 is_force_rescale=true 强制重缩放（m_scale 恒等于 m_run），P 在
 *    本块锚点量化——twin/golden 的逐块锚定轨迹（e4m3 量化网格依赖锚点，锚点滞后会偏离
 *    golden 网格）；skip-scale（tau=6）机制保留在 online_softmax.h 供向量化 pass 复用；
 * c. 448· 放大域经乘法实现（twin::SoftmaxProbCode 单点，行 max 恰为 448 填满 e4m3 动态
 *    范围）；s_fp8_offset 指数偏置域为非默认可选（与 skip-scale 存在 2^δ·448 溢出冲突，
 *    启用须 tau=0），基线恒为 0；
 * d. 两级累加：fp32 档逐 token 直累；fp32+fp16 档按 k_step=16 子块 FP16 域累加
 *    （twin::QuantizePAndAccumulatePvRow 单点），O 主累加器驻留 V1 UB（fp32）。
 */
class SageAttnKernel {
public:


    /**
     * 核心初始化逻辑
     */
    __aicore__ void Init(
        __gm__ uint8_t *query,
        __gm__ uint8_t *key,
        __gm__ uint8_t *value,
        __gm__ uint8_t *keyMean,
        __gm__ uint8_t *lseCorrection,
        __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *softmaxLse,
        __gm__ uint8_t *workspace,
        const SageAttention2TilingData *__restrict tilingData)
    {
        tiling_ = tilingData;
        query_gm_ = query;
        key_gm_ = key;
        value_gm_ = value;
        key_mean_gm_ = keyMean;
        lse_correction_gm_ = lseCorrection;
        out_gm_ = attentionOut;
        lse_gm_ = softmaxLse;
        workspace_gm_ = workspace; // 签名契约占位：用户区经 GetUserWorkspace 旁路直达 kernel 入口，
                                   // 成员保留以对齐设备入口形参。

        // GetBlockIdx 设备端返回 int64_t，显式收窄避免隐式截断
        core_idx_ = static_cast<uint32_t>(GetBlockIdx());
        if SAGE_ATTN_IS_AIV {
            sub_core_idx_ = core_idx_ % 2;
            core_idx_ /= 2;
        } else {
            sub_core_idx_ = 0;
        }

        // 提取核心 tiling 常量
        b_ = tiling_->b;
        hq_ = tiling_->hq;
        hkv_ = tiling_->hkv;
        sq_ = tiling_->sq;
        sk_ = tiling_->sk;
        d_ = tiling_->d_pad;
        gqa_ratio_ = (hkv_ > 0) ? (hq_ / hkv_) : 1;

        m_per_core_ = tiling_->m_per_core;
        block_n_ = tiling_->block_n;
        k_step_ = (tiling_->k_step != 0u) ? tiling_->k_step : DEFAULT_K_STEP; // 0 归一防死循环

        // 计算当前核的任务归属与 DieTask 解耦入口
        // 沿 (Batch, Hkv, Sq_block) 空间切分，优先将同一 KV 头的多个 Q 头聚合在同一核内执行
        // 注意：tiling_ 为 const __gm__ 限定，bisheng dav-3510 禁止在设备域调用其 const 成员函数
        //（报 reference to __host__ function / this 未带 __gm__ 限定），故核内一律直接读字段。
        uint32_t q_blks_per_head = (sq_ + m_per_core_ - 1) / m_per_core_;
        uint32_t total_tasks = b_ * hkv_ * q_blks_per_head;
        uint32_t cores = tiling_->core_num_used;
        // DieTask 分发：在奇 cores 下防跨 die 错配
        bool use_die = (tiling_->enable_die_affinity != 0) && (cores % 2u == 0u);

        // 依据 DieTaskEnd 判定当前物理核的任务终止上界 (支持局部尾波次退出)
        const uint64_t task_end = DieTaskEnd(total_tasks, core_idx_, cores, use_die);
        if (core_idx_ >= task_end) {
            is_active_core_ = false;
            return;
        }
        is_active_core_ = true;

        // 解码初始任务 (第 0 波次)
        uint64_t initial_task = use_die ? DieTask(core_idx_, total_tasks, cores) : core_idx_;
        DecodeTask(initial_task, q_blks_per_head);

        for (uint32_t r = 0; r < kKernelMaxRows; ++r) {
            lse_corr_[r] = 0.0f;
        }

#if !SAGE_ATTN_DEVICE_COMPILE
        if (sim_mem_ == nullptr) {
            // 宿主仿真缓冲随核对象生命周期自动释放
            sim_mem_ = std::make_unique<SimMemory>();
        }
#endif
    }

    /**
     * 算子主流水执行入口
     */
    __aicore__ void Process()
    {
        if (!is_active_core_) {
            return;
        }

        uint32_t q_blks_per_head = (sq_ + m_per_core_ - 1) / m_per_core_;
        uint32_t total_tasks = b_ * hkv_ * q_blks_per_head;
        uint32_t cores = tiling_->core_num_used;
        // DieTask 分发：在奇 cores 下防跨 die 错配
        bool use_die = (tiling_->enable_die_affinity != 0) && (cores % 2u == 0u);
        const uint64_t task_end = DieTaskEnd(total_tasks, core_idx_, cores, use_die);

        // 循环执行分配给当前物理核的所有波次任务 (Die 内部严格亲和)
        // 跨任务波次 flag 纪律（task/02_code.md B-2）：每个任务域内 Cube 进域预置
        // KV/S 槽"已归还"许可、V0/V1 退域回收残余许可，保证任意任务边界上 flag 余量为零。
        for (uint64_t virtual_task = core_idx_; virtual_task < task_end; virtual_task += cores) {
            // 当虚拟任务推进至后续波次时重新解码任务
            if (virtual_task > core_idx_) {
                uint64_t mapped_task = use_die ? DieTask(virtual_task, total_tasks, cores) : virtual_task;
                DecodeTask(mapped_task, q_blks_per_head);
            }

#if defined(__CCE_KT_TEST__) || defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
            if SAGE_ATTN_IS_AIC {
                ProcessCube();
            } else {
                if (sub_core_idx_ == 0) {
                    ProcessVec0();
                } else {
                    ProcessVec1();
                }
            }
#else
            // 宿主机模拟时按逻辑顺序串行演进
            ProcessVec0();
            ProcessCube();
            ProcessVec1();
#endif
        }
    }

private:
    /**
     * 任务解码辅助函数：将一维任务索引解耦为 (Batch, Hkv, Q_blk) 并更新当前 Token 范围与累加控制器
     */
    __aicore__ void DecodeTask(uint64_t task_idx, uint32_t q_blks_per_head)
    {
        int64_t t = static_cast<int64_t>(task_idx);
        int64_t q_blk_id = t % q_blks_per_head;
        int64_t rem = t / q_blks_per_head;
        int64_t hkv_id = rem % hkv_;
        int64_t b_id = rem / hkv_;

        cur_b_ = static_cast<uint32_t>(b_id);
        cur_hkv_ = static_cast<uint32_t>(hkv_id);
        cur_q_blk_ = static_cast<uint32_t>(q_blk_id);

        // 计算当前 Q 分块的实际 token 范围
        cur_q_start_ = cur_q_blk_ * m_per_core_;
        cur_q_len_ = (cur_q_start_ + m_per_core_ <= sq_) ? m_per_core_ : (sq_ - cur_q_start_);

        // 初始化两级累加控制器与因果掩码活跃分块数 (Causal Mask Early Exit)
        uint32_t total_kv_blocks = (sk_ + block_n_ - 1) / block_n_;
        uint32_t active_kv_blocks = (tiling_->is_causal != 0)
            ? SageFastMin(total_kv_blocks, (cur_q_start_ + cur_q_len_ + block_n_ - 1) / block_n_)
            : total_kv_blocks;
        active_kv_blocks_ = active_kv_blocks;
        accum_ctrl_.Init(tiling_->pv_accum_dtype, k_step_, active_kv_blocks);
    }

    /**
     * S(P) 矩阵 UB 双缓冲槽地址（双缓冲避免延迟一拍流水下的读写竞争）。
     * e4m3 每元素 1 字节，PONG 槽紧随 PING 槽之后，要求 host 侧 ub_softmax_offset
     * 预留 >= 2 * 窗口行数(GQA 聚合行数) * block_n 字节。槽步长按实际写出的行数
     * （gqa_ratio * cur_q_len，上限 kKernelMaxRows）计，保证奇偶槽互不交叠。
     */
    __aicore__ uint32_t SoftmaxSlotOffset(bool is_ping) const
    {
        uint32_t rows = cur_q_len_ * gqa_ratio_;
        if (rows > kKernelMaxRows) {
            rows = kKernelMaxRows;
        }
        const uint32_t tile_bytes = rows * block_n_;
        return tiling_->ub_softmax_offset + (is_ping ? 0u : tile_bytes);
    }

    // GQA 聚合后的窗口逻辑行数与当前块的实际 K 列数
    __aicore__ uint32_t TotalRows() const
    {
        return cur_q_len_ * gqa_ratio_;
    }

    __aicore__ uint32_t CurBlockK() const
    {
        const int64_t sk_start = static_cast<int64_t>(cur_k_) * block_n_;
        return (sk_start + block_n_ <= sk_) ? block_n_ : static_cast<uint32_t>(sk_ - sk_start);
    }

    __aicore__ uint32_t CurKvSlotBase() const
    {
        return ((cur_k_ % 2u) == 0u) ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                     : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET;
    }

    __aicore__ __gm__ float *StageStrip() const
    {
        return reinterpret_cast<__gm__ float *>(workspace_gm_) +
               static_cast<uint64_t>(core_idx_) * (STAGE_CORE_STRIDE / sizeof(float));
    }

    /**
     * 1. Cube 核心处理逻辑（MMAD 计算与宽流水延迟退休主流水）
     * 拍 k：INT8 QK^T MMAD，L0C 累加
     * 拍 k+1：FP8 PV MMAD，L0C 部分和经 FixPipe 裸搬至 UB 交接槽
     *
     * 缓冲布局与读取契约说明：
     * 宿主串行仿真时序为 ProcessVec0 → ProcessCube → ProcessVec1 整段执行，拍 k 的 PV
     * 依赖 V1 在拍 k-1 写出的 S 槽（ub_softmax_offset 奇偶双缓冲），整段时序下 Cube 的
     * PV MMAD 无从消费；且 L0C 单缓冲无法为 V1 保留逐拍 QK^T。故基线的 QK^T 反量化与
     * PV 累加由 V1 按 twin 5.1/5.2/5.4 唯一定义标量执行（见 ComputeBlockMaxAndSum /
     * AccumulatePvBlockTwin），逐位对齐孪生轨迹；S 槽写出布局维持 SoftmaxSlotOffset
     * 契约不变。Cube 侧 INT8 QK^T / FP8 PV MMAD + FixPipe + 3 槽退休队列整体保留在
     * 设备分支（SAGE_ATTN_DEVICE_COMPILE）。
     */
    __aicore__ void ProcessCube()
    {
#if SAGE_ATTN_DEVICE_COMPILE
        Catlass::SageAttn::SageQkMmad<> qk_mmad;
        Catlass::SageAttn::SagePvMmad<> pv_mmad;

        // A7（task/12 D3）：热路径不使用 OutputRetireQueue——golden 轨迹为逐块锚定
        // 强制 rescale，队列的 tau/skip-scale 双锚点与 V1 独占 O 主累加器冲突；
        // 每块 PV L0C 部分和经 FixPipe 裸搬至 UB 交接槽，由 V1 按档位合并。

        // 进域预置（task/02_code.md B-2 init 纪律）：Cube 是 KV 槽与 S 槽的消费者，
        // 预置"首次使用前已归还"许可，V0/V1 首轮写槽前的 wait 立即放行。
        // A7 新增：PARTIAL_FREE / DOT_FREE 预置（交接槽与 dot32 单缓冲首拍放行）。
        // 余量核算（单任务域，N = total_kv_blocks；R22 逐 flag 对账，与实现一致）：
        //   KV_PING/PONG_FREE：set 2+N（预置 2 合计 + 逐块 N）/ wait N+2（V0 逐块 N + 退域 2）
        //   SOFTMAX_S_PING/PONG_FREE：set 2+N（预置 2 合计 + 逐前块 N 合计）/
        //     wait N+2（V1 逐块 N + 退域 2）
        //   PV_PARTIAL_FREE：set 1+N（预置 1 + V1 合并 N：拍顶 N-1 + 收尾 1）/
        //     wait N+1（逐块发射前 N-1 + drain 1 + 退域回收 1）
        //   DOT_FREE：set 1+N（预置 1 + V1 逐块 N）/ wait N+1（逐块 N + 退域回收 1）
        // N=0/1 边界：PARTIAL/DOT 各 set 1=wait 1 / 2=2（退域回收无条件执行预置回收，
        // 故置于分块条件之外）；任务边界余量严格归零。
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC0, CrossCoreFlags::FLAG_KV_PING_FREE);
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC0, CrossCoreFlags::FLAG_KV_PONG_FREE);
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_SOFTMAX_S_PING_FREE);
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_SOFTMAX_S_PONG_FREE);
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_PARTIAL_FREE);
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_DOT_FREE);

        // 1. 等待 V0 完成 Q 矩阵量化并常驻 L1 Bank 0
        CrossCoreWait(CrossCoreDirection::VEC0_TO_CUBE, CrossCoreFlags::FLAG_Q_READY);

        // 加载 Q 分块至 L0A（GQA 头聚合沿 M 维堆叠）
        qk_mmad.LoadL0A(tiling_->l1_q_offset, 0, cur_q_len_ * gqa_ratio_, d_);

        uint32_t total_kv_blocks = (sk_ + block_n_ - 1) / block_n_;
        uint32_t active_kv_blocks = (tiling_->is_causal != 0)
            ? SageFastMin(total_kv_blocks, (cur_q_start_ + cur_q_len_ + block_n_ - 1) / block_n_)
            : total_kv_blocks;
        const uint32_t rows = cur_q_len_ * gqa_ratio_;

        // 延迟一拍宽流水主循环 (Wide Stream Double Buffering Ping-Pong)
        for (uint32_t k = 0; k < active_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            uint32_t kv_flag = is_ping ? CrossCoreFlags::FLAG_KV_PING_READY : CrossCoreFlags::FLAG_KV_PONG_READY;
            uint32_t kv_free = is_ping ? CrossCoreFlags::FLAG_KV_PING_FREE : CrossCoreFlags::FLAG_KV_PONG_FREE;
            uint32_t l1_kv_addr = is_ping ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                          : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET;

            // 1. 等待 V0 当前 KV 分块量化就绪 (N256 Cube / N512 Vector 解耦)
            CrossCoreWait(CrossCoreDirection::VEC0_TO_CUBE, kv_flag);

            // 2. 拍 k（Cube）：INT8 QK^T MMAD（L0C 单缓冲：dot(k) 必须先于 PV(k-1) Mmad 搬出）
            // 加载 K 分块至 L0B
            qk_mmad.LoadL0B(l1_kv_addr, 0, block_n_, d_);
            qk_mmad.Compute(true); // 逐块独立部分和（golden 逐块锚定轨迹，task/12 归约粒度定案）

            // K 块（L1 Bank 1 槽）消费完成：归还缓冲许可，V0 在重写同奇偶槽（k+2）前会 wait。
            CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC0, kv_free);

            // 3. A7：QK dot32 裸搬 UB（读保护：V1 完成上一块反量化读取后经 DOT_FREE 放行；
            //    FixPipe 走 FIX 管，CrossCoreSet 在 S 管——先自配对排空 FIX 再 raise flag 保 GM/UB 可见序）
            CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_DOT_FREE);
            qk_mmad.EmitL0CToUbRaw(SAGE_UB_DOT32_OFFSET, SAGE_UB_DOT32_OFFSET + 64 * block_n_ * sizeof(int32_t),
                                   rows, block_n_);
            Catlass::SageAttn::Sync<HardEvent::FIX_S>();
            CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_QK_P_READY);

            // 4. 拍 k（Cube 并发）：计算上一拍的 FP8 PV MMAD (当 k > 0 时)
            if (k > 0) {
                uint32_t prev_k = k - 1;
                bool prev_is_ping = ((prev_k % 2) == 0);
                uint32_t prev_l1_v_addr = (prev_is_ping ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                                        : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET)
                                          + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET;
                uint32_t s_flag = prev_is_ping ? CrossCoreFlags::FLAG_SOFTMAX_S_PING_READY
                                               : CrossCoreFlags::FLAG_SOFTMAX_S_PONG_READY;
                uint32_t s_free = prev_is_ping ? CrossCoreFlags::FLAG_SOFTMAX_S_PING_FREE
                                               : CrossCoreFlags::FLAG_SOFTMAX_S_PONG_FREE;

                // 等待 V1 上一拍 Softmax 概率矩阵 S 就绪（按 prev_k 奇偶双缓冲）
                CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, s_flag);

                // 加载 S 矩阵至 L0A（对应奇偶槽），加载 V 矩阵至 L0B。
                // 形状记账（DEFECT-M-2）：PV MMAD 契约为 C[rows×d] = P[rows×block_n] ×
                // V[block_n×d]——LoadL0B 的 (n,k) 须为 (d, block_n)（n_=输出列 d、
                // k_=收缩维 block_n）。原参 (block_n_, d_) 会把 LoadL0A 设定的 k_=block_n
                // 覆写为 d_：收缩维读越 S 的 block_n 宽行（L0A 越读）、L0C 产 rows×block_n
                // 与 pv_partial 区/MergePvPartialRows 的 rows×d 契约错位。nZ 分形转置
                // 语义（16×16 ifTranspose）为真机 L-A 复核项，与 A7 族登记一致。
                pv_mmad.LoadL0A(SoftmaxSlotOffset(prev_is_ping), 0, rows, block_n_);
                pv_mmad.LoadL0B(prev_l1_v_addr, 0, d_, block_n_);

                // 每块独立部分和（init=true；两级累加的二级归并由 V1 按 pv_accum_dtype 档位执行）
                pv_mmad.Compute(true);

                // S[prev_k]（UB 槽）消费完成：归还缓冲许可，V1 在重写同奇偶槽前会 wait
                CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, s_free);

                // A7：PV 部分和裸搬 UB 交接槽（读保护：V1 合并完上一块后经 PARTIAL_FREE 放行）
                CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_PV_PARTIAL_FREE);
                FixpipeConfig fix_cfg = accum_ctrl_.GetFixpipeConfig();
                pv_mmad.ExecuteEpilogue(0, SAGE_UB_PV_PARTIAL_OFFSET, fix_cfg);
                Catlass::SageAttn::Sync<HardEvent::FIX_S>();
                CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_PARTIAL_READY);
            }
        }

        // Drain 拍：完成最后一个 KV 分块的 PV MMAD 与部分和搬出
        if (active_kv_blocks > 0) {
            uint32_t last_k = active_kv_blocks - 1;
            bool last_is_ping = ((last_k % 2) == 0);
            uint32_t last_l1_v_addr = (last_is_ping ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                                    : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET)
                                      + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET;
            uint32_t s_flag = last_is_ping ? CrossCoreFlags::FLAG_SOFTMAX_S_PING_READY
                                           : CrossCoreFlags::FLAG_SOFTMAX_S_PONG_READY;
            uint32_t s_free = last_is_ping ? CrossCoreFlags::FLAG_SOFTMAX_S_PING_FREE
                                           : CrossCoreFlags::FLAG_SOFTMAX_S_PONG_FREE;

            CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, s_flag);

            pv_mmad.LoadL0A(SoftmaxSlotOffset(last_is_ping), 0, cur_q_len_ * gqa_ratio_, block_n_);
            // (n,k)=(d, block_n) 同上：PV 收缩维 block_n、输出列 d（DEFECT-M-2）
            pv_mmad.LoadL0B(last_l1_v_addr, 0, d_, block_n_);
            pv_mmad.Compute(true);

            // 归还最后一个 S 槽许可（V1 退域回收依赖此配平）
            CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, s_free);

            // 最后一块部分和搬出
            CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_PV_PARTIAL_FREE);
            FixpipeConfig fix_cfg = accum_ctrl_.GetFixpipeConfig();
            pv_mmad.ExecuteEpilogue(0, SAGE_UB_PV_PARTIAL_OFFSET, fix_cfg);
            Catlass::SageAttn::Sync<HardEvent::FIX_S>();
            CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_PARTIAL_READY);
        }

        // 退域回收（B-2 destroy 纪律；R22 修正：移出分块条件——N=0 时亦须回收预置许可，
        // N≥1 时消费 V1 最后一块的归还许可，保证任务边界 DOT/PARTIAL_FREE 余量严格归零）
        CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_DOT_FREE);
        CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_PV_PARTIAL_FREE);
#else
        // KT/宿主标量基线：本函数无 Cube 侧数值职责。
        // 跨核 flag 的进域预置/退域回收在宿主分支为空操作，V1 退域回收依赖的
        // S 槽许可配平由设备分支版本定义，基线不重复维护。
#endif

        // 通知 V1：整个 Attention 的 PV 累加已全部完成
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_O_DONE);
    }

    /**
     * 2. Vector 0 核心处理逻辑（量化 prologue 核）
     * 负责 GM 流式预取、K 减均值、INT8 量化、V 转置与 FP8 量化、ND2NZ 摆放
     */
    __aicore__ void ProcessVec0()
    {
        // 64 位 GM 基地址与步长计算
        int64_t b_idx = static_cast<int64_t>(cur_b_);
        int64_t hkv_idx = static_cast<int64_t>(cur_hkv_);
        int64_t sq_start = static_cast<int64_t>(cur_q_start_);

        // GQA 聚合：当前 KV 头所对应的第一个 Q 头
        int64_t hq_first = hkv_idx * static_cast<int64_t>(gqa_ratio_);

        // 计算 64 位全局地址偏移
        int64_t q_stride_b, q_stride_h, q_stride_s;
        if ((tiling_->tensor_layout == LAYOUT_HND)) {
            q_stride_b = static_cast<int64_t>(hq_) * sq_ * d_;
            q_stride_h = static_cast<int64_t>(sq_) * d_;
            q_stride_s = d_;
        } else {
            q_stride_b = static_cast<int64_t>(sq_) * hq_ * d_;
            q_stride_s = static_cast<int64_t>(hq_) * d_;
            q_stride_h = d_;
        }
        int64_t q_gm_offset = b_idx * q_stride_b + hq_first * q_stride_h + sq_start * q_stride_s;

        // 1. Prologue：加载当前核负责的 Q，执行 INT8 量化并写入 L1 Bank 0
        LoadAndQuantizeQ(query_gm_ + q_gm_offset * sizeof(uint16_t), cur_q_len_, d_);

        // 通知 Cube：Q 量化常驻已就位
        CrossCoreSet(CrossCoreDirection::VEC0_TO_CUBE, CrossCoreFlags::FLAG_Q_READY);

        uint32_t total_kv_blocks = (sk_ + block_n_ - 1) / block_n_;
        uint32_t active_kv_blocks = (tiling_->is_causal != 0)
            ? SageFastMin(total_kv_blocks, (cur_q_start_ + cur_q_len_ + block_n_ - 1) / block_n_)
            : total_kv_blocks;

        // 2. 流水循环：流式预取第 k 块 KV 并执行 INT8/FP8 量化
        for (uint32_t k = 0; k < active_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            uint32_t kv_flag = is_ping ? CrossCoreFlags::FLAG_KV_PING_READY : CrossCoreFlags::FLAG_KV_PONG_READY;
            uint32_t kv_free = is_ping ? CrossCoreFlags::FLAG_KV_PING_FREE : CrossCoreFlags::FLAG_KV_PONG_FREE;
            uint32_t l1_dst_offset = is_ping ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                             : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET;

            // 缓冲回收握手：同奇偶槽须待 Cube 消费完上一轮（k-2）后方可覆写。
            // 首两个槽（k=0/1）的许可由 Cube 进域预置，此处 wait 立即放行。
            CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC0, kv_free);

            cur_k_ = k;
            int64_t sk_start = static_cast<int64_t>(k) * block_n_;
            cur_k_start_ = static_cast<uint32_t>(sk_start);
            uint32_t cur_block_k = (sk_start + block_n_ <= sk_) ? block_n_ : static_cast<uint32_t>(sk_ - sk_start);

            // 计算 K/V 全局内存偏移
            int64_t kv_stride_b, kv_stride_h, kv_stride_s;
            if ((tiling_->tensor_layout == LAYOUT_HND)) {
                kv_stride_b = static_cast<int64_t>(hkv_) * sk_ * d_;
                kv_stride_h = static_cast<int64_t>(sk_) * d_;
                kv_stride_s = d_;
            } else {
                kv_stride_b = static_cast<int64_t>(sk_) * hkv_ * d_;
                kv_stride_s = static_cast<int64_t>(hkv_) * d_;
                kv_stride_h = d_;
            }

            int64_t k_gm_offset = b_idx * kv_stride_b + hkv_idx * kv_stride_h + sk_start * kv_stride_s;
            int64_t v_gm_offset = k_gm_offset;

            // 获取 K 均值指针 (若启用 smooth_k)。
            // __gm__ 地址空间限定显式保留，统一保持 `const __gm__ float *` 贯穿传递。
            const __gm__ float *km_ptr = nullptr;
            if ((tiling_->smooth_k != 0) && key_mean_gm_ != nullptr) {
                int64_t km_offset = (b_idx * static_cast<int64_t>(hkv_) + hkv_idx) * d_;
                km_ptr = reinterpret_cast<const __gm__ float*>(key_mean_gm_ + km_offset * sizeof(float));
            }

            // 执行 K 减均值 + INT8 量化 + 摆放，写入 L1 Bank 1
            QuantizeAndStageKey(key_gm_ + k_gm_offset * sizeof(uint16_t), km_ptr, l1_dst_offset, cur_block_k, d_);

            // 执行 V 转置 + 16-token permute + FP8 量化，写入 L1 Bank 1 槽内 V 子区
            QuantizeAndStageValue(value_gm_ + v_gm_offset * sizeof(uint16_t),
                                  l1_dst_offset + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET,
                                  cur_block_k, d_);

            // 严格保证生产者在前：写入完成后触发跨核事件通知 Cube
            // 生产管排空（DEFECT-H-2）：MTE3 写 KV 槽 → Cube MTE 读，CrossCoreSet
            // （S 管）不排空 MTE3，先自配对事件等待写毕再举旗（设备态；宿主
            // CrossCoreSet 为空桩且无 HardEvent，见 sync_guard.h）
#if SAGE_ATTN_DEVICE_COMPILE
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
#endif
            CrossCoreSet(CrossCoreDirection::VEC0_TO_CUBE, kv_flag);
        }

        // 退域回收（task/02_code.md B-2 destroy 纪律）：消费 Cube 进域预置的最后两个
        // KV 槽许可，保证本任务结束时 FLAG_KV_*_FREE 余量为零，下一波次进域预置才重新立券，
        // 防跨任务残留许可导致首两槽未消费即被覆写。
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC0, CrossCoreFlags::FLAG_KV_PING_FREE);
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC0, CrossCoreFlags::FLAG_KV_PONG_FREE);
    }

    /**
     * 3. Vector 1 核心处理逻辑（Softmax 核）
     * 负责 exp2 域 Online Softmax、skip-scale 判定、O 主累加器更新、LSE 累积与写回。
     * softmax 状态为行级（GQA 逻辑行 r = g * cur_q_len + i 各自独立维护 max/denominator，
     * 与 twin RunSequential 的行级状态一致）；S 槽写出布局维持 SoftmaxSlotOffset 奇偶
     * 双缓冲契约不变（V1 写出、Cube 读取）。
     * R17-1（A7 设备消费侧）：设备态补齐 dot32/pv_partial 交接槽的消费握手与数据通路——
     *   每拍 k：拍顶合并第 k-1 块 PV 部分和（wait PARTIAL_READY → MergePvPartialRows →
     *   set PARTIAL_FREE；O 锚点仍为 m_scale(k-1)，早于本拍 Step 重缩放，与 twin 5.4 的
     *   "块内 rescale 后并入"轨迹等价，task/12 第三节时序）→ wait S_FREE → wait QK_P →
     *   逐行 dot32 反量化/掩码 → 行 max/exp2 和 → Step → O 行重缩放 → P 码逐行写 S 槽 →
     *   V_S 排空 → set DOT_FREE（dot32 单缓冲读保护归还）→ set S_READY；
     *   收尾：wait PARTIAL_READY → 合并最后一块 → set PARTIAL_FREE → wait PV_O_DONE。
     */
    __aicore__ void ProcessVec1()
    {
#if !SAGE_ATTN_DEVICE_COMPILE
        if (sim_mem_ != nullptr) {
            for (uint32_t r = 0; r < TotalRows() && r < kKernelMaxRows; ++r) {
                row_states_[r].Reset();
                // O 主累加器按任务复位（真实硬件由下方设备分支的 UB 复位承担）
                float *o_row = &sim_mem_->o_accum[static_cast<size_t>(r) * d_];
                for (uint32_t c = 0; c < d_; ++c) {
                    o_row[c] = 0.0f;
                }
            }
        }
#else
        for (uint32_t r = 0; r < TotalRows() && r < kKernelMaxRows; ++r) {
            row_states_[r].Reset();
        }
        // O 主累加器按任务复位（V1 UB 常驻区；多波次任务域防跨任务残留，与宿主 sim 复位同语义）
        {
            const uint32_t rows = TotalRows();
            LocalTensor<float> ub_o(TPosition::VECCALC,
                                    Catlass::SageAttn::Vector1UbBudget::O_ACCUM_OFFSET, rows * d_);
            Duplicate(ub_o, 0.0f, rows * d_);
        }
#endif
        uint32_t total_kv_blocks = (sk_ + block_n_ - 1) / block_n_;
        uint32_t active_kv_blocks = (tiling_->is_causal != 0)
            ? SageFastMin(total_kv_blocks, (cur_q_start_ + cur_q_len_ + block_n_ - 1) / block_n_)
            : total_kv_blocks;

        for (uint32_t k = 0; k < active_kv_blocks; ++k) {
            cur_k_ = k;
            cur_k_start_ = k * block_n_;
            bool is_ping = ((k % 2) == 0);
            uint32_t s_flag = is_ping ? CrossCoreFlags::FLAG_SOFTMAX_S_PING_READY
                                      : CrossCoreFlags::FLAG_SOFTMAX_S_PONG_READY;
            uint32_t s_free = is_ping ? CrossCoreFlags::FLAG_SOFTMAX_S_PING_FREE
                                      : CrossCoreFlags::FLAG_SOFTMAX_S_PONG_FREE;

#if SAGE_ATTN_DEVICE_COMPILE
            // A7 拍顶部：合并上一块（第 k-1 块）PV 部分和并归还交接槽许可（时序见函数注释）
            if (k > 0) {
                CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_PARTIAL_READY);
                MergePvPartialRows();
                CrossCoreSet(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_PV_PARTIAL_FREE);
            }
#endif
            // 缓冲回收握手：同奇偶 S 槽须待 Cube 消费完上一轮（k-2）后方可覆写（宿主为空操作）。
            // R22 前移至行循环前：设备态 P 码在行循环内逐行写槽，许可须先于首行写获取。
            CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, s_free);

            // 1. 等待 Cube 拍 k 的 P 矩阵块就绪
            CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_QK_P_READY);

#if !SAGE_ATTN_DEVICE_COMPILE
            // 1.5 宿主串行仿真重摆：恢复设备分支"V0 产出→消费"的逐拍序（幂等，逐位一致）。
            RestageKvBlockHost(k);
#else
            // 1.6 A7 scale 条带桥（task/12 第四节）：q 行解析 / v per-channel 任务级一次，
            //     k 块解析按奇偶槽逐拍接入（GM 暂存条带 → UB 副本，MTE2 排空后进 VF/S）
            if (k == 0) {
                StageV1ScaleStrips();
            }
            StageKScaleStrip(is_ping);
#endif

#if SAGE_ATTN_DEVICE_COMPILE
            // 2/3. A7 融合行轨迹：dot32 反量化/掩码 → 行 max/448· 域 exp2 和 → Step →
            //      O 行重缩放 → P 码逐行写 S 槽（掩码行跳过 Step，P 码经 exp2 下溢自然为 0，
            //      与宿主整幅 ComputeSoftmaxProbability 写出同值）
            for (uint32_t r = 0; r < TotalRows() && r < kKernelMaxRows; ++r) {
                float cur_max = MIN_FLOAT_VAL;
                float cur_sum = 0.0f;
                ComputeBlockMaxAndSum(r, cur_max, cur_sum, k);
                if (cur_max > -1.0e30f) {
                    const float rescale_factor =
                        OnlineSoftmaxTransformer::Step(row_states_[r], cur_max, cur_sum, true);
                    if (row_states_[r].need_rescale &&
                        SageFastAbs(rescale_factor - 1.0f) > SCALE_ONE_TOLERANCE) {
                        RescaleAccumulator(rescale_factor, r);
                    }
                }
                SoftmaxProbRow(r, is_ping);
            }
#else
            // 2. 逐行块内 logit 反量化/掩码、行 max 与 exp2 和（twin 5.1/5.2 单点），
            //    随后按 twin 5.3 步进行级状态并重缩放 O 行。
            for (uint32_t r = 0; r < TotalRows() && r < kKernelMaxRows; ++r) {
                float cur_max = MIN_FLOAT_VAL;
                float cur_sum = 0.0f;
                ComputeBlockMaxAndSum(r, cur_max, cur_sum, k);
                if (cur_max <= -1.0e30f) {
                    continue; // twin 5.1：整块被因果掩码（前缀行），无贡献不步进
                }
                const float rescale_factor =
                    OnlineSoftmaxTransformer::Step(row_states_[r], cur_max, cur_sum, true);
                if (row_states_[r].need_rescale &&
                    SageFastAbs(rescale_factor - 1.0f) > SCALE_ONE_TOLERANCE) {
                    RescaleAccumulator(rescale_factor, r);
                }
            }

            // 3. 计算 exp2 域 Softmax 概率矩阵 P[k] 写入 (k%2) 槽（448· 放大域经乘法实现，
            //    行 max 恰为 448）
            ComputeSoftmaxProbability(is_ping);

            // 4. KT 标量基线：PV 累加经 twin 5.4 单点执行（两级累加 fp32 / fp32+fp16 同轨迹；
            //    真实硬件为 Cube PV MMAD + FixPipe → V1 拍顶/收尾合并，见函数注释）
            AccumulatePvBlockTwin(CurBlockK());
#endif

            // 生产管排空（DEFECT-H-2）：VF 写 P 槽（V 管）→ Cube LoadL0A（MTE）读，
            // CrossCoreSet（S 管）不排空 V 管，先自配对事件等待写毕再举旗
#if SAGE_ATTN_DEVICE_COMPILE
            Catlass::SageAttn::Sync<HardEvent::V_S>();
            // dot32 单缓冲读保护归还（V 管读毕，Cube 方可在 beat k+1 覆写 dot32 区）
            CrossCoreSet(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_DOT_FREE);
#endif
            CrossCoreSet(CrossCoreDirection::VEC1_TO_CUBE, s_flag);
        }

#if SAGE_ATTN_DEVICE_COMPILE
        // 收尾：合并最后一块（第 N-1 块）PV 部分和——O 锚点 m_scale(N-1) 上直接并入，
        // 与 twin 5.5 归一化入口的 O 状态逐位同域（时序无死锁：Cube drain 发射 READY 后
        // 经退域回收等本侧 FREE，本侧合并后举 FREE 再等 PV_O_DONE，无环）
        if (active_kv_blocks > 0) {
            CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_PARTIAL_READY);
            MergePvPartialRows();
            CrossCoreSet(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_PV_PARTIAL_FREE);
        }
#endif

        // 等待 Cube 全部 PV 累加收尾完成
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_O_DONE);

        // 5. 最终输出归一化与 LSE 换算写回 GM
        FinalizeAndStoreOutput();

        // 退域回收（task/02_code.md B-2 destroy 纪律）：与 V0 的 KV 槽同理，
        // 消费 Cube 进域预置的最后两个 S 槽许可，保证跨任务波次 FLAG_SOFTMAX_S_*_FREE 余量为零
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_SOFTMAX_S_PING_FREE);
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_SOFTMAX_S_PONG_FREE);
    }

#if SAGE_ATTN_DEVICE_COMPILE
    /**
     * A7 scale 条带桥（task/12 第四节；GM 暂存条带常量以 tiling.h 为唯一定义点）：
     * V0 产出 scale → GetUserWorkspace 用户区按核条带（core_idx 折半后 × STAGE_CORE_STRIDE）
     * → V1 以 MTE2 接入 UB 副本。GM 可见性序：V0 条带写以 S_MTE3/MTE3_S 双自配对排空后
     * 方随 Q_READY/KV_*_READY raise（LoadAndQuantizeQ/QuantizeAndStageKey/Value 设备分支），
     * 本侧一律在 wait QK_P_READY 之后读取（跨核 flag 链传递保序）。
     * 已知限制（真机验证项）：k 块解析 ping/pong 条带为单槽双缓冲，V0 在拍 k+2 经
     * kv_free 许可重写同奇偶槽，早于 V1 拍 k 读取完成的协议序未闭合（需条带消费
     * ACK 扩展 flag 协议，task/12 第四节演进项）——L-A/真机门前须复核。
     */
    __aicore__ void StageV1ScaleStrips()
    {
        if (workspace_gm_ == nullptr) {
            return;
        }
        __gm__ uint8_t *strip =
            workspace_gm_ + static_cast<int64_t>(core_idx_) * STAGE_CORE_STRIDE;
        GlobalTensor<float> gm_strip;
        LocalTensor<float> ub_qs(TPosition::VECCALC, SAGE_UB_V1_QS_ROW_OFFSET, STAGE_SLOT_FLOATS);
        LocalTensor<float> ub_vs(TPosition::VECCALC, SAGE_UB_V1_VS_CHAN_OFFSET, STAGE_SLOT_FLOATS);
        gm_strip.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(strip + STAGE_Q_SCALE_OFFSET));
        DataCopy(ub_qs, gm_strip, STAGE_SLOT_FLOATS);
        gm_strip.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(strip + STAGE_V_SCALE_OFFSET));
        DataCopy(ub_vs, gm_strip, STAGE_SLOT_FLOATS);
        SyncMte2ToV();
        // qs 由 S 管逐行 GetValue 消费：MTE2 写 → S 管读须显式排空（vs 由 V 管消费，MTE2_V 已排）
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
    }

    __aicore__ void StageKScaleStrip(bool is_ping)
    {
        if (workspace_gm_ == nullptr) {
            return;
        }
        __gm__ uint8_t *strip =
            workspace_gm_ + static_cast<int64_t>(core_idx_) * STAGE_CORE_STRIDE;
        GlobalTensor<float> gm_strip;
        LocalTensor<float> ub_ks(TPosition::VECCALC, SAGE_UB_V1_KS_ROW_OFFSET, STAGE_SLOT_FLOATS);
        gm_strip.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
            strip + (is_ping ? STAGE_K_SCALE_PING_OFFSET : STAGE_K_SCALE_PONG_OFFSET)));
        DataCopy(ub_ks, gm_strip, block_n_);
        SyncMte2ToV();
    }

    /**
     * A7：pv_partial 交接槽全行合并（MergePvPartialRowVF，sage_pv_mmad.hpp 单点）。
     * 轨迹 = 宿主 AccumulatePvBlockTwin 的设备段实例：O 行 += partial·v_scale[c]，
     * fp32+fp16 档按 D2 以 enable_inline_quant 为档位判据在合并处做 FP16 RNE 舍回
     * （QuantizeToFp16 唯一口径）；块内求和序由 PV MMAD 承担（设备段粒度，
     * L-A 容差域登记 task/12）。
     */
    __aicore__ void MergePvPartialRows()
    {
        const uint32_t rows = TotalRows();
        const uint32_t fp16_round =
            accum_ctrl_.GetFixpipeConfig().enable_inline_quant ? 1u : 0u;
        LocalTensor<float> ub_vs(TPosition::VECCALC, SAGE_UB_V1_VS_CHAN_OFFSET, d_);
        for (uint32_t r = 0; r < rows && r < kKernelMaxRows; ++r) {
            LocalTensor<float> ub_partial(TPosition::VECCALC,
                                          SAGE_UB_PV_PARTIAL_OFFSET + r * d_ * sizeof(float), d_);
            LocalTensor<float> ub_o(TPosition::VECCALC,
                                    Catlass::SageAttn::Vector1UbBudget::O_ACCUM_OFFSET +
                                        r * d_ * sizeof(float),
                                    d_);
            Catlass::SageAttn::MergePvPartialRowVF(
                reinterpret_cast<__ubuf__ float*>(ub_partial.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(ub_o.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(ub_vs.GetPhyAddr()),
                d_, fp16_round);
        }
    }

    /**
     * A7：单行 P 码写 S 槽（SoftmaxProbRowVF；行锚 = row_states_[r].m_scale，
     * 与宿主 ComputeSoftmaxProbability 同式——掩码哨兵行经 exp2 下溢自然落到 e4m3 码 0）。
     */
    __aicore__ void SoftmaxProbRow(uint32_t r, bool is_ping)
    {
        const float m_prime = row_states_[r].m_scale;
        LocalTensor<float> ub_logits(TPosition::VECCALC, SAGE_UB_V1_LOGITS_ROW_OFFSET, block_n_);
        LocalTensor<float8_e4m3_t> ub_out_p(TPosition::VECCALC,
                                            SoftmaxSlotOffset(is_ping) + r * block_n_, block_n_);
        dev_vf::SoftmaxProbRowVF(reinterpret_cast<__ubuf__ float*>(ub_logits.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_out_p.GetPhyAddr()),
                                 m_prime, block_n_);
    }
#endif

    // -------------------------------------------------------------------------
    // 内部实现辅助函数（Step-C 标量基线：数值语义以 twin_reference.h RunSequential 为
    // 唯一真值，量化原语/分组索引/P 量化/PV 行累加经其单点 include 复用，禁止复制公式；
    // 设备分支维持流水结构桩，向量化实现留给后续 pass）
    // -------------------------------------------------------------------------

    /**
     * 1. V0：GM→L1 Q 搬入与 INT8 量化（twin 量化轨迹逐步一致）：
     * - per_thread：组内 amax，scale = amax/INT8_QUANT_MAX + SCALE_EPSILON，
     *   RoundHalfAwayFromZero 舍入；per_warp：amax 先做 SCALE_EPSILON 兜底，
     *   scale = amax/INT8_QUANT_MAX，RoundNearestEven 舍入；
     * - 保持除法运算（val / scale）以对齐 GPU/孪生 ULP 舍入轨迹（禁止换乘倒数）；
     * - GQA 聚合沿 M 维堆叠行：逻辑行 r = g * cur_q_len + i（g 为 KV 头内 Q 头序号），
     *   Q scale 按头分段存放 sim q_scales[g*2048 + 组号]（KT 基线的 scale 暂存；
     *   L1 内 scale 摆放随 tiling.l1_q_scale_offset 于向量化 pass 落地），L1 内 [行][d] 摆放；
     * - smooth_k：行级 lse_correction = q_i · km（q 按输入 dtype 读取、fp32 积和，
     *   km 由 host 按 twin 同口径预计算传入），供 FinalizeAndStoreOutput 的 LSE 出口；
     *   设备态经 GM 标量读按宿主同式补算（R22，lse_corr_[r] 全分支共享状态）。
     */
#if !SAGE_ATTN_DEVICE_COMPILE
    /**
     * 输入位型分派：Q/K/V 读取与输出写出按 tiling.input_dtype
     * 分派，位级转换唯一来源 twin::detail::ReadInput/WriteOutput。
     */
    inline float ReadInputBits(const uint16_t *base, int64_t idx) const
    {
        return twin::detail::ReadInput(base, static_cast<size_t>(idx),
                                       (tiling_->input_dtype == 2u) ? 2 : 1);
    }

    inline void WriteOutputBits(uint16_t *base, int64_t idx, float val) const
    {
        twin::detail::WriteOutput(base, static_cast<size_t>(idx),
                                  (tiling_->input_dtype == 2u) ? 2 : 1, val);
    }
#endif

    __aicore__ void LoadAndQuantizeQ(__gm__ uint8_t *q_src, uint32_t m, uint32_t d)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        if (q_src == nullptr) return;
        // DEFECT-F-1 重构：组粒度两阶段——行 abs-max VF 进 UB rowmax 数组，
        // 组归并/定档经 quant::QScaleIndex 单点（与宿主串行路径逐步一致），
        // 逐行量化 VF 走除法轨迹 + per_thread half-away / per_warp RINT。
        const bool per_thread_q = (tiling_->qk_quant_gran == QUANT_GRAN_PER_THREAD);
        const uint32_t total_rows_q = TotalRows();
        const int64_t q_stride_h = (tiling_->tensor_layout == LAYOUT_HND)
                                       ? static_cast<int64_t>(sq_) * d_ : d_;
        const int64_t q_stride_s = (tiling_->tensor_layout == LAYOUT_HND)
                                       ? d_ : static_cast<int64_t>(hq_) * d_;
        LocalTensor<float> ub_rowmax(TPosition::VECCALC,
                                     Catlass::SageAttn::Vector0UbBudget::QUANT_WORKSPACE_OFFSET,
                                     kKernelMaxRows);
        LocalTensor<float> ub_scales(TPosition::VECCALC,
                                     Catlass::SageAttn::Vector0UbBudget::QUANT_WORKSPACE_OFFSET + 512,
                                     2048);
        LocalTensor<half> ub_in(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::GM_INPUT_PING_OFFSET, d);
        LocalTensor<int8_t> ub_out(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::NZ_OUTPUT_OFFSET, d);
        LocalTensor<int8_t> l1_q(TPosition::A1, tiling_->l1_q_offset, total_rows_q * d);
        const uint32_t dtype_mode = (tiling_->input_dtype == 2u) ? 2u : 1u;

        // 窗口 scale 组清零（防跨任务波次陈旧 amax 混入）
        const uint32_t grp_first_q = quant::QScaleIndex(cur_q_start_, per_thread_q);
        const uint32_t grp_last_q = quant::QScaleIndex(cur_q_start_ + cur_q_len_ - 1u, per_thread_q);
        for (uint32_t g = 0; g < gqa_ratio_ && g < 8u; ++g) {
            for (uint32_t s = grp_first_q; s <= grp_last_q && s < 2048u; ++s) {
                ub_scales.SetValue(g * 2048u + s, 0.0f);
            }
        }

        // A. 行 abs-max（每行一次 MTE2 + VF）
        for (uint32_t r = 0; r < total_rows_q && r < kKernelMaxRows; ++r) {
            // V→MTE2 回边（R22）：ub_in 逐行复用，上一行 VF 读毕（V 管）方可 MTE2 覆写
            SyncVToMte2();
            const uint32_t g = r / cur_q_len_;
            const uint32_t i = r % cur_q_len_;
            const int64_t row_gm_offset = (g * q_stride_h + static_cast<int64_t>(i) * q_stride_s) * static_cast<int64_t>(sizeof(half));
            GlobalTensor<half> gm_row;
            gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q_src + row_gm_offset));
            DataCopy(ub_in, gm_row, d);
            SyncMte2ToV();
            dev_vf::RowAbsMaxVF(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float*>(ub_rowmax.GetPhyAddr()) + r,
                                nullptr, d, 0u, dtype_mode);
        }
        // V 管写 → S 管读同步（DEFECT-G-1，与 V1 统计读取同款事件对）
        Catlass::SageAttn::Sync<HardEvent::V_S>();

        // B. 组归并 + scale 定档（宿主同式；per_thread=amax/127+1e-7，
        //    per_warp=clamp(amax,1e-7)/127）
        for (uint32_t r = 0; r < total_rows_q && r < kKernelMaxRows; ++r) {
            const uint32_t g = r / cur_q_len_;
            const uint32_t token = cur_q_start_ + (r % cur_q_len_);
            const float amax = ub_rowmax.GetValue(r);
            const uint32_t slot = g * 2048u + quant::QScaleIndex(token, per_thread_q);
            if (amax > ub_scales.GetValue(slot)) ub_scales.SetValue(slot, amax);
        }
        for (uint32_t g = 0; g < gqa_ratio_ && g < 8u; ++g) {
            for (uint32_t s = grp_first_q; s <= grp_last_q && s < 2048u; ++s) {
                const float amax = ub_scales.GetValue(g * 2048u + s);
                ub_scales.SetValue(g * 2048u + s, per_thread_q
                    ? (amax / INT8_QUANT_MAX + SCALE_EPSILON)
                    : ((amax < SCALE_EPSILON ? SCALE_EPSILON : amax) / INT8_QUANT_MAX));
            }
        }

        // C. 逐行量化写 L1（除法轨迹 + 舍入分档）
        for (uint32_t r = 0; r < total_rows_q && r < kKernelMaxRows; ++r) {
            // V→MTE2 回边（R22）：ub_in 逐行复用，上一行 VF 读毕方可 MTE2 覆写
            SyncVToMte2();
            const uint32_t g = r / cur_q_len_;
            const uint32_t i = r % cur_q_len_;
            const uint32_t token = cur_q_start_ + i;
            const float scale = ub_scales.GetValue(g * 2048u + quant::QScaleIndex(token, per_thread_q));
            const int64_t row_gm_offset = (g * q_stride_h + static_cast<int64_t>(i) * q_stride_s) * static_cast<int64_t>(sizeof(half));
            GlobalTensor<half> gm_row;
            gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q_src + row_gm_offset));
            DataCopy(ub_in, gm_row, d);
            SyncMte2ToV();
            dev_vf::QuantRowInt8VF(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                   reinterpret_cast<__ubuf__ int8_t*>(ub_out.GetPhyAddr()),
                                   scale, d, per_thread_q ? 1u : 0u, 0u, nullptr, dtype_mode);
            SyncVToMte3();
            DataCopyParams cp;
            cp.blockCount = 1;
            cp.blockLen = d / 32;
            cp.srcStride = 0;
            cp.dstStride = 0;
            DataCopy(l1_q[r * d], ub_out, cp);
            // MTE3→V 回边（DEFECT-I-1 同款）：ub_out 逐行复用，在途 MTE3 读排空后方可下一行 VF 写
            SyncMte3ToV();
        }

        // A7 V0（task/12 第五节第 3 条）：行解析 q_scale 暂存写——组定标后的逐行
        // scale 按 GQA 聚合行序展开写 GM 条带（STAGE_Q_SCALE_OFFSET，零填充至 128 槽），
        // V1 反量化按行读取。S→MTE3 与 MTE3→S 双自配对：标量 SetValue 对 MTE3 可见、
        // MTE3 写对后续 S 管 raise FLAG_Q_READY 可见（GM 可见性，task/12 第四节）。
        LocalTensor<float> ub_stage_q(TPosition::VECCALC,
                                      Catlass::SageAttn::Vector0UbBudget::QUANT_WORKSPACE_OFFSET + 10240,
                                      STAGE_SLOT_FLOATS);
        for (uint32_t s = 0; s < STAGE_SLOT_FLOATS; ++s) {
            ub_stage_q.SetValue(s, 0.0f);
        }
        for (uint32_t r = 0; r < total_rows_q && r < kKernelMaxRows; ++r) {
            const uint32_t g = r / cur_q_len_;
            const uint32_t token = cur_q_start_ + (r % cur_q_len_);
            ub_stage_q.SetValue(r, ub_scales.GetValue(g * 2048u + quant::QScaleIndex(token, per_thread_q)));
        }
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        GlobalTensor<float> gm_stage_q;
        gm_stage_q.SetGlobalBuffer(StageStrip() + STAGE_Q_SCALE_OFFSET);
        DataCopy(gm_stage_q, ub_stage_q, STAGE_SLOT_FLOATS);
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);

        // 4. smooth_k 行级 lse_correction（设备态，R22）：q·km 逐行点积——宿主步 4 同式：
        //    输入按位型解码（bf16/fp16 → fp32，quant 层唯一口径）× km（fp32，host 预计算
        //    传入），顺序 fp32 积和逐位同式；GM 标量读范式与 V per-channel 预扫一致。
        if ((tiling_->smooth_k != 0) && key_mean_gm_ != nullptr) {
            const int64_t km_offset =
                (static_cast<int64_t>(cur_b_) * hkv_ + cur_hkv_) * d;
            GlobalTensor<float> gm_km;
            gm_km.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(key_mean_gm_ + km_offset * sizeof(float)));
            GlobalTensor<uint16_t> gm_q_bits;
            gm_q_bits.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(q_src));
            for (uint32_t r = 0; r < total_rows_q && r < kKernelMaxRows; ++r) {
                const uint32_t g = r / cur_q_len_;
                const uint32_t i = r % cur_q_len_;
                const int64_t row_elem = g * q_stride_h + static_cast<int64_t>(i) * q_stride_s;
                float corr = 0.0f;
                for (uint32_t c = 0; c < d; ++c) {
                    const uint16_t bits = gm_q_bits(row_elem + c);
                    const float val = (dtype_mode == 2u) ? quant::Bf16ToFp32(bits)
                                                         : quant::Fp16ToFp32(bits);
                    corr += val * gm_km(c);
                }
                lse_corr_[r] = corr;
            }
        }
#else
        if (sim_mem_ == nullptr || q_src == nullptr) {
            return;
        }
        const bool per_thread = (tiling_->qk_quant_gran == QUANT_GRAN_PER_THREAD);
        const int64_t q_stride_h = (tiling_->tensor_layout == LAYOUT_HND)
                                       ? static_cast<int64_t>(sq_) * d_ : d_;
        const int64_t q_stride_s = (tiling_->tensor_layout == LAYOUT_HND)
                                       ? d_ : static_cast<int64_t>(hq_) * d_;
        const uint16_t *q_u16 = reinterpret_cast<const uint16_t *>(q_src);
        int8_t *l1_q = reinterpret_cast<int8_t *>(sim_mem_->l1_mem + tiling_->l1_q_offset);

        // 窗口覆盖的 scale 组清零（跨任务波次复用 SimMemory，防陈旧 amax 混入）
        const uint32_t grp_first = quant::QScaleIndex(cur_q_start_, per_thread);
        const uint32_t grp_last = quant::QScaleIndex(cur_q_start_ + m - 1u, per_thread);
        for (uint32_t g = 0; g < gqa_ratio_ && g < 8u; ++g) {
            for (uint32_t s = grp_first; s <= grp_last && s < 2048u; ++s) {
                sim_mem_->q_scales[g * 2048u + s] = 0.0f;
            }
        }

        // 1. 逐 (GQA 头, scale 组) 的 amax（组边界 = quant::QScaleIndex 唯一定义；
        //    m_per_core 为 64 的倍数保证窗口不切割 32-token 子块）
        for (uint32_t r = 0; r < TotalRows(); ++r) {
            const uint32_t g = r / cur_q_len_;
            if (r >= kKernelMaxRows || g >= 8u) break;
            const uint32_t i = r % cur_q_len_;
            const uint32_t token = cur_q_start_ + i;
            float *scale_slot = &sim_mem_->q_scales[g * 2048u + quant::QScaleIndex(token, per_thread)];
            const uint16_t *row_ptr = q_u16 + g * q_stride_h + static_cast<int64_t>(i) * q_stride_s;
            for (uint32_t c = 0; c < d; ++c) {
                const float abs_v = SageFastAbs(ReadInputBits(row_ptr, c));
                if (abs_v > *scale_slot) *scale_slot = abs_v;
            }
        }

        // 2. scale 定档（quant 层唯一口径）
        for (uint32_t g = 0; g < gqa_ratio_ && g < 8u; ++g) {
            for (uint32_t s = grp_first; s <= grp_last && s < 2048u; ++s) {
                const float amax = sim_mem_->q_scales[g * 2048u + s];
                sim_mem_->q_scales[g * 2048u + s] = per_thread
                    ? (amax / INT8_QUANT_MAX + SCALE_EPSILON)
                    : ((amax < SCALE_EPSILON ? SCALE_EPSILON : amax) / INT8_QUANT_MAX);
            }
        }

        // 3. 量化写 L1（除法轨迹）
        for (uint32_t r = 0; r < TotalRows(); ++r) {
            const uint32_t g = r / cur_q_len_;
            if (r >= kKernelMaxRows || g >= 8u) break;
            const uint32_t i = r % cur_q_len_;
            const uint32_t token = cur_q_start_ + i;
            const float scale = sim_mem_->q_scales[g * 2048u + quant::QScaleIndex(token, per_thread)];
            const uint16_t *row_ptr = q_u16 + g * q_stride_h + static_cast<int64_t>(i) * q_stride_s;
            int8_t *row_out = l1_q + static_cast<size_t>(r) * d;
            for (uint32_t c = 0; c < d; ++c) {
                const float val = ReadInputBits(row_ptr, c);
                row_out[c] = per_thread ? quant::RoundHalfAwayFromZero(val / scale)
                                        : quant::RoundNearestEven(val / scale);
            }
        }

        // 4. smooth_k 行级 lse_correction
        if ((tiling_->smooth_k != 0) && key_mean_gm_ != nullptr) {
            const int64_t km_offset =
                (static_cast<int64_t>(cur_b_) * hkv_ + cur_hkv_) * d;
            const float *km = reinterpret_cast<const float *>(
                key_mean_gm_ + km_offset * sizeof(float));
            for (uint32_t r = 0; r < TotalRows(); ++r) {
                if (r >= kKernelMaxRows) break;
                const uint32_t g = r / cur_q_len_;
                const uint32_t i = r % cur_q_len_;
                const uint16_t *row_ptr = q_u16 + g * q_stride_h + static_cast<int64_t>(i) * q_stride_s;
                float corr = 0.0f;
                for (uint32_t c = 0; c < d; ++c) {
                    corr += ReadInputBits(row_ptr, c) * km[c];
                }
                sim_mem_->lse_corr[r] = corr;
                lse_corr_[r] = corr;
            }
        }
#endif
    }

    /**
     * 2. V0：K 减 kMean + INT8 量化 + 摆放写 L1 槽首（twin 量化轨迹逐步一致）：
     * - kMean 减法按 HalfReader::Sub 口径（差值舍入回 fp16 后再取 amax/量化）；
     * - per_thread：64-token 块 4 组相邻对交织组，half-away 舍入；
     *   per_warp：64-token 块 1 组，RNE 舍入；组边界 = quant::KScaleIndex 唯一定义；
     * - L1 槽布局（Data + Scale 相邻）：[block_n 行 K INT8][FP32 scale 组]，
     *   尾块 pad 行写 0（对应 token ≥ sk，V1 侧按掩码哨兵排除）。
     */
    __aicore__ void QuantizeAndStageKey(
        __gm__ uint8_t *k_src,
        const __gm__ float *km_ptr,
        uint32_t l1_dst_addr,
        uint32_t block_k,
        uint32_t d)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        if (k_src == nullptr) return;
        // DEFECT-F-1 重构：组粒度两阶段（组边界 quant::KScaleIndex 单点，
        // 块内本地序号写 l1_k_scale——Data+Scale 相邻契约）
        const bool per_thread_k = (tiling_->qk_quant_gran == QUANT_GRAN_PER_THREAD);
        const int64_t k_stride_s = (tiling_->tensor_layout == LAYOUT_HND)
                                       ? d_ : static_cast<int64_t>(hkv_) * d_;
        const uint32_t dtype_mode = (tiling_->input_dtype == 2u) ? 2u : 1u;
        const uint32_t grp_per_blk = per_thread_k ? 4u : 1u;
        const uint32_t blk_first = cur_k_start_ / quant::K_BLK;
        const uint32_t grp_lo = blk_first * grp_per_blk;
        const uint32_t grp_hi = ((cur_k_start_ + block_n_ - 1u) / quant::K_BLK + 1u) * grp_per_blk;
        LocalTensor<float> ub_rowmax(TPosition::VECCALC,
                                     Catlass::SageAttn::Vector0UbBudget::QUANT_WORKSPACE_OFFSET,
                                     kKernelMaxRows);
        LocalTensor<float> ub_k_scales(TPosition::VECCALC,
                                       Catlass::SageAttn::Vector0UbBudget::QUANT_WORKSPACE_OFFSET + 512,
                                       2048);
        LocalTensor<float> ub_km(TPosition::VECCALC,
                                 Catlass::SageAttn::Vector0UbBudget::QUANT_WORKSPACE_OFFSET + 1024 + 2048 * 4,
                                 d);
        LocalTensor<half> ub_in(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::GM_INPUT_PING_OFFSET, d);
        LocalTensor<int8_t> ub_out(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::NZ_OUTPUT_OFFSET, d);
        LocalTensor<int8_t> l1_k(TPosition::B1, l1_dst_addr, block_n_ * d);
        LocalTensor<float> l1_k_scale(TPosition::B1, l1_dst_addr + static_cast<uint32_t>(block_n_ * d), 2048);
        const uint32_t has_km = (km_ptr != nullptr) ? 1u : 0u;

        if (has_km) {
            GlobalTensor<float> gm_km;
            gm_km.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(const_cast<__gm__ float*>(km_ptr)));
            DataCopy(ub_km, gm_km, d);
            SyncMte2ToV();
        }

        // 窗口 scale 组清零
        for (uint32_t s = grp_lo; s < grp_hi && s < 2048u; ++s) {
            ub_k_scales.SetValue(s, 0.0f);
        }

        // A. 行 abs-max（仅真实 token；pad 行零贡献且 pad 组写 0 由 C 段 pad 分支覆盖）
        for (uint32_t j = 0; j < block_k; ++j) {
            // V→MTE2 回边（R22）：ub_in 逐行复用，上一行 VF 读毕（V 管）方可 MTE2 覆写
            SyncVToMte2();
            const int64_t row_gm_offset = static_cast<int64_t>(j) * k_stride_s * static_cast<int64_t>(sizeof(half));
            GlobalTensor<half> gm_row;
            gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k_src + row_gm_offset));
            DataCopy(ub_in, gm_row, d);
            SyncMte2ToV();
            dev_vf::RowAbsMaxVF(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                reinterpret_cast<__ubuf__ float*>(ub_rowmax.GetPhyAddr()) + j,
                                reinterpret_cast<__ubuf__ float*>(ub_km.GetPhyAddr()),
                                d, has_km, dtype_mode);
        }
        // V 管写 → S 管读同步（DEFECT-G-1）
        Catlass::SageAttn::Sync<HardEvent::V_S>();

        // B. 组归并（pair/块粒度，KScaleIndex 单点）+ 定档 + l1_k_scale 本地写出
        for (uint32_t j = 0; j < block_k; ++j) {
            const uint32_t slot = quant::KScaleIndex(cur_k_start_ + j, per_thread_k);
            if (ub_rowmax.GetValue(j) > ub_k_scales.GetValue(slot)) ub_k_scales.SetValue(slot, ub_rowmax.GetValue(j));
        }
        for (uint32_t s = grp_lo; s < grp_hi && s < 2048u; ++s) {
            const float amax = ub_k_scales.GetValue(s);
            const float scale = per_thread_k
                ? (amax / INT8_QUANT_MAX + SCALE_EPSILON)
                : ((amax < SCALE_EPSILON ? SCALE_EPSILON : amax) / INT8_QUANT_MAX);
            ub_k_scales.SetValue(s, scale);
            l1_k_scale.SetValue(s - grp_lo, scale);
        }

        // A7 V0（task/12 第五节第 3 条）：块解析 k_scale 暂存写——逐 token 展开写
        // GM 条带 ping/pong 槽（按 cur_k_ 奇偶，零填充至 128 槽），V1 反量化按当前块
        // 奇偶读取。S→MTE3 / MTE3→S 双自配对排空后方由调用方 raise KV_*_READY
        //（GM 可见性，task/12 第四节；QK_P 传递性保证 V1 读取时序）。
        LocalTensor<float> ub_stage_k(TPosition::VECCALC,
                                      Catlass::SageAttn::Vector0UbBudget::QUANT_WORKSPACE_OFFSET + 10240,
                                      STAGE_SLOT_FLOATS);
        for (uint32_t s = 0; s < STAGE_SLOT_FLOATS; ++s) {
            ub_stage_k.SetValue(s, 0.0f);
        }
        for (uint32_t j = 0; j < block_n_; ++j) {
            if (j < block_k) {
                const uint32_t token = cur_k_start_ + j;
                ub_stage_k.SetValue(j, ub_k_scales.GetValue(quant::KScaleIndex(token, per_thread_k)));
            } // 尾块 pad 列保持 0：V1 侧前缀掩码哨兵不消费该列 scale
        }
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        GlobalTensor<float> gm_stage_k;
        gm_stage_k.SetGlobalBuffer(StageStrip() +
                                   ((cur_k_ % 2u) == 0u ? STAGE_K_SCALE_PING_OFFSET : STAGE_K_SCALE_PONG_OFFSET));
        DataCopy(gm_stage_k, ub_stage_k, STAGE_SLOT_FLOATS);
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);

        // C. 逐行量化写 L1 + 尾块 pad 置零
        for (uint32_t j = 0; j < block_n_; ++j) {
            // V→MTE2 回边（R22）：ub_in 逐行复用，上一行 VF 读毕方可 MTE2 覆写
            SyncVToMte2();
            if (j < block_k) {
                const int64_t row_gm_offset = static_cast<int64_t>(j) * k_stride_s * static_cast<int64_t>(sizeof(half));
                GlobalTensor<half> gm_row;
                gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k_src + row_gm_offset));
                DataCopy(ub_in, gm_row, d);
                SyncMte2ToV();
                const uint32_t token = cur_k_start_ + j;
                const float scale = ub_k_scales.GetValue(quant::KScaleIndex(token, per_thread_k));
                dev_vf::QuantRowInt8VF(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ int8_t*>(ub_out.GetPhyAddr()),
                                       scale, d, per_thread_k ? 1u : 0u, has_km,
                                       reinterpret_cast<__ubuf__ float*>(ub_km.GetPhyAddr()), dtype_mode);
            } else {
                Duplicate(ub_out, int8_t(0), d);
            }
            SyncVToMte3();
            DataCopyParams cp;
            cp.blockCount = 1;
            cp.blockLen = d / 32;
            cp.srcStride = 0;
            cp.dstStride = 0;
            DataCopy(l1_k[j * d], ub_out, cp);
            // MTE3→V 回边（DEFECT-I-1 同款）：ub_out 逐行复用，在途 MTE3 读排空后方可下一行 VF/Duplicate 写
            SyncMte3ToV();
        }
#else
        if (sim_mem_ == nullptr || k_src == nullptr) {
            return;
        }
        const bool per_thread = (tiling_->qk_quant_gran == QUANT_GRAN_PER_THREAD);
        const int64_t k_stride_s = (tiling_->tensor_layout == LAYOUT_HND)
                                       ? d_ : static_cast<int64_t>(hkv_) * d_;
        const uint16_t *k_u16 = reinterpret_cast<const uint16_t *>(k_src);
        int8_t *l1_k = reinterpret_cast<int8_t *>(sim_mem_->l1_mem + l1_dst_addr);
        float *l1_k_scale = reinterpret_cast<float *>(
            sim_mem_->l1_mem + l1_dst_addr + static_cast<size_t>(block_n_) * d);

        // 本核块覆盖的 scale 组清零（防跨块/跨任务陈旧 amax 混入）。
        // block_n 可为 128（host 侧 sk>=2048 档），一个核块横跨两个 64-token K 量化块，
        // 组区间须按 ceil(block_n/K_BLK) 个量化块覆盖。
        const uint32_t grp_per_blk = per_thread ? 4u : 1u;
        const uint32_t blk_first = cur_k_start_ / quant::K_BLK;
        const uint32_t blk_last = (cur_k_start_ + block_n_ - 1u) / quant::K_BLK;
        const uint32_t grp_lo = blk_first * grp_per_blk;
        const uint32_t grp_hi = (blk_last + 1u) * grp_per_blk;
        for (uint32_t s = grp_lo; s < grp_hi && s < 2048u; ++s) {
            sim_mem_->k_scales[s] = 0.0f;
        }

        // 1. 逐组 amax（kMean 减法带 fp16 舍入）
        for (uint32_t j = 0; j < block_k; ++j) {
            const uint32_t token = cur_k_start_ + j;
            float *scale_slot = &sim_mem_->k_scales[quant::KScaleIndex(token, per_thread)];
            const uint16_t *row_ptr = k_u16 + static_cast<int64_t>(j) * k_stride_s;
            for (uint32_t c = 0; c < d; ++c) {
                float val = ReadInputBits(row_ptr, c);
                if (km_ptr != nullptr) {
                    // 差值位型舍回（对齐孪生 Half/Bf16Reader::Sub 口径）
                    val = (tiling_->input_dtype == 2u)
                              ? quant::Bf16ToFp32(quant::Fp32ToBf16(val - km_ptr[c]))
                              : quant::Fp16ToFp32(quant::Fp32ToFp16(val - km_ptr[c]));
                }
                const float abs_v = SageFastAbs(val);
                if (abs_v > *scale_slot) *scale_slot = abs_v;
            }
        }

        // 2. scale 定档（l1_k_scale 写槽内本地序号：Data+Scale 相邻布局按块内组序
        // 排布，防全局组号越界）
        for (uint32_t s = grp_lo; s < grp_hi && s < 2048u; ++s) {
            const float amax = sim_mem_->k_scales[s];
            const float scale = per_thread
                ? (amax / INT8_QUANT_MAX + SCALE_EPSILON)
                : ((amax < SCALE_EPSILON ? SCALE_EPSILON : amax) / INT8_QUANT_MAX);
            sim_mem_->k_scales[s] = scale;
            l1_k_scale[s - grp_lo] = scale;
        }

        // 3. 量化写 L1 + 尾块 pad 置零
        for (uint32_t j = 0; j < block_n_; ++j) {
            int8_t *row_out = l1_k + static_cast<size_t>(j) * d;
            if (j < block_k) {
                const uint32_t token = cur_k_start_ + j;
                const float scale = sim_mem_->k_scales[quant::KScaleIndex(token, per_thread)];
                const uint16_t *row_ptr = k_u16 + static_cast<int64_t>(j) * k_stride_s;
                for (uint32_t c = 0; c < d; ++c) {
                    float val = ReadInputBits(row_ptr, c);
                    if (km_ptr != nullptr) {
                        // 差值位型舍回（对齐孪生 Half/Bf16Reader::Sub 口径）
                        val = (tiling_->input_dtype == 2u)
                                  ? quant::Bf16ToFp32(quant::Fp32ToBf16(val - km_ptr[c]))
                                  : quant::Fp16ToFp32(quant::Fp32ToFp16(val - km_ptr[c]));
                    }
                    row_out[c] = per_thread ? quant::RoundHalfAwayFromZero(val / scale)
                                            : quant::RoundNearestEven(val / scale);
                }
            } else {
                for (uint32_t c = 0; c < d; ++c) {
                    row_out[c] = 0;
                }
            }
        }
#endif
    }

    /**
     * 3. V0：V 转置 + 16-token permute + per-channel FP8 量化写 L1 槽内 V 子区
     * （twin/quant_fp8 量化轨迹逐步一致）：
     * - per-channel scale 为全序列静态量：首块（cur_k_==0）对整个 KV 头做 GM amax 预扫
     *   （pad token 计 0 不影响 amax），scale = amax / v_scale_max（fp32+fp16 档 2.25、
     *   其余 448，由 tiling 携带）；全零通道 scale=0 经 NaN → e4m3 0x7f 容错（同 quant_fp8）；
     * - 量化保持除法轨迹：scaled = val / scale，饱和 ±scale_max 后 RNE 转 e4m3；
     * - L1 槽内 V 布局 [d][block_n]，16-token 段内物理槽 p 存 token V_PERM_PATTERN[p]
     *   （token jl 的物理槽 = V_PERM_INV_PATTERN[jl%16]，与 golden v_fp8 同构、与
     *   twin v_deq 读取式一致）；尾块 pad 槽写 0。
     */
    __aicore__ void QuantizeAndStageValue(
        __gm__ uint8_t *v_src,
        uint32_t l1_dst_addr,
        uint32_t block_k,
        uint32_t d)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        if (v_src == nullptr) return;
        // DEFECT-F-1 重构：per-channel 全序列 amax 预扫（首块一次，标量语义优先、
        // 向量化为后续 pass）→ UB scale 张量；逐行量化 VF 走除法轨迹 + ±scale_max
        // 截断 + 全零通道 0/0=NaN → e4m3 0x7f。
        const uint32_t dtype_mode = (tiling_->input_dtype == 2u) ? 2u : 1u;
        const int64_t v_stride_s = (tiling_->tensor_layout == LAYOUT_HND)
                                       ? d_ : static_cast<int64_t>(hkv_) * d_;
        LocalTensor<float> ub_v_scale(TPosition::VECCALC,
                                      Catlass::SageAttn::Vector0UbBudget::QUANT_WORKSPACE_OFFSET,
                                      d);
        LocalTensor<half> ub_in(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::GM_INPUT_PING_OFFSET, d);
        LocalTensor<float8_e4m3_t> ub_out(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::NZ_OUTPUT_OFFSET, d);
        LocalTensor<float8_e4m3_t> l1_v(TPosition::B1, l1_dst_addr, block_n_ * d);
        const float scale_max = tiling_->v_scale_max;

        if (cur_k_ == 0) {
            // A7 V0（task/12 第五节第 3 条）：per-channel 全序列 amax 预扫向量化——
            // ub_v_scale 充零作 amax 累加器，逐 token 行 ChannelAmaxAccumVF 累加
            //（v_src 即 KV 头首行基址；pad token 计 0 不影响），FinalizeVScaleVF 按
            // v_scale_max 除法定标（quant_fp8/孪生唯一口径）。
            Duplicate(ub_v_scale, 0.0f, d);
            for (uint32_t t = 0; t < sk_; ++t) {
                const int64_t row_gm_offset = static_cast<int64_t>(t) * v_stride_s * static_cast<int64_t>(sizeof(half));
                GlobalTensor<half> gm_row;
                gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v_src + row_gm_offset));
                DataCopy(ub_in, gm_row, d);
                SyncMte2ToV();
                dev_vf::ChannelAmaxAccumVF(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                           d, dtype_mode);
            }
            dev_vf::FinalizeVScaleVF(reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                     scale_max, d);

            // A7：v_scale 暂存写（STAGE_V_SCALE_OFFSET，任务级一次）。V→MTE3 排空
            // 保证 scale 向量写毕才发起 GM 搬出，MTE3→S 排空后方由调用方在首块
            // KV_READY raise 前 GM 可见（V1 合并侧经 PV_PARTIAL→QK_P 传递性读取）。
            SyncVToMte3();
            GlobalTensor<float> gm_stage_v;
            gm_stage_v.SetGlobalBuffer(StageStrip() + STAGE_V_SCALE_OFFSET);
            DataCopy(gm_stage_v, ub_v_scale, d);
            SetFlag<HardEvent::MTE3_S>(0);
            WaitFlag<HardEvent::MTE3_S>(0);
        }

        for (uint32_t j = 0; j < block_n_; ++j) {
            // V→MTE2 回边（R22）：ub_in 逐行复用，上一行 VF 读毕（V 管）方可 MTE2 覆写
            SyncVToMte2();
            if (j < block_k) {
                const int64_t row_gm_offset = static_cast<int64_t>(j) * v_stride_s * static_cast<int64_t>(sizeof(half));
                GlobalTensor<half> gm_row;
                gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v_src + row_gm_offset));
                DataCopy(ub_in, gm_row, d);
                SyncMte2ToV();
                dev_vf::QuantVRowFp8VF(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_out.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                       d, scale_max, dtype_mode);
            } else {
                Duplicate(ub_out.ReinterpretCast<int8_t>(), int8_t(0), d);
            }
            SyncVToMte3();
            DataCopyParams cp;
            cp.blockCount = 1;
            cp.blockLen = d / 32;
            cp.srcStride = 0;
            cp.dstStride = 0;
            DataCopy(l1_v[j * d], ub_out, cp);
            // MTE3→V 回边（DEFECT-I-1 同款）：ub_out 逐行复用，在途 MTE3 读排空后方可下一行 VF/Duplicate 写
            SyncMte3ToV();
        }
#else
        if (sim_mem_ == nullptr || v_src == nullptr) {
            return;
        }
        const int64_t v_stride_s = (tiling_->tensor_layout == LAYOUT_HND)
                                       ? d_ : static_cast<int64_t>(hkv_) * d_;
        const uint16_t *v_u16 = reinterpret_cast<const uint16_t *>(v_src);
        uint8_t *l1_v = sim_mem_->l1_mem + l1_dst_addr;
        const float scale_max = tiling_->v_scale_max;

        if (cur_k_ == 0) {
            const uint16_t *v_head = v_u16 - static_cast<int64_t>(cur_k_start_) * v_stride_s;
            for (uint32_t c = 0; c < d && c < 256u; ++c) {
                float amax = 0.0f;
                for (uint32_t t = 0; t < sk_; ++t) {
                    const float abs_v = SageFastAbs(
                        ReadInputBits(v_head, static_cast<int64_t>(t) * v_stride_s + c));
                    if (abs_v > amax) amax = abs_v;
                }
                sim_mem_->v_scales[c] = amax / scale_max;
            }
        }

        for (uint32_t c = 0; c < d; ++c) {
            const float scale = (c < 256u) ? sim_mem_->v_scales[c] : 1.0f;
            uint8_t *chan_out = l1_v + static_cast<size_t>(c) * block_n_;
            for (uint32_t jl = 0; jl < block_n_; ++jl) {
                uint8_t code = 0;
                if (jl < block_k) {
                    const uint16_t *row_ptr = v_u16 + static_cast<int64_t>(jl) * v_stride_s;
                    float scaled = ReadInputBits(row_ptr, c) / scale;
                    if (scaled > scale_max) scaled = scale_max;
                    if (scaled < -scale_max) scaled = -scale_max;
                    code = quant::FloatToE4m3fn(scaled);
                }
                chan_out[(jl / 16u) * 16u + quant::V_PERM_INV_PATTERN[jl % 16u]] = code;
            }
        }
#endif
    }

    /**
     * 4. V1：单逻辑行的块内反量化 logit、行 max 与 exp2 和（twin 5.1/5.2 单点轨迹）：
     * - 设备态（A7，R17-1）：Cube 经 EmitL0CToUbRaw 落 dot32 区 [row][block_n] INT32，
     *   本函数 DequantMaskLogitsRowVF 反量化+掩码写 logits 行暂存（twin::DequantQkLogit
     *   左结合乘序，|dot| ≤ 2^21 cast 精确，01_design 2.9；causal/尾块掩码 -1e30 哨兵），
     *   随后 BlockMaxAndSumVF 行 max 与 448· 域 exp2 和——组合即宿主同构语义；
     * - 宿主/KT 标量基线：INT32 精确点积 + twin::DequantQkLogit 单点 scale 乘
     *   （q_scale·k_scale·sm_scale_log2e 左结合乘序同孪生）——串行整段仿真时序
     *   （Vec0→Cube→V1）中 L0C 单缓冲无法保留逐拍 QK^T，详见 ProcessCube；
     * - 掩码（causal j > token_q、尾块 token_k ≥ sk_）写 -1e30 哨兵，不参与 max/sum；
     * - cur_sum 为 448· 放大域（twin::OnlineBlockExp2Sum 同式，与 P 量化同域相消；
     *   设备 exp 域换算经 ln2 缩放，真机 L-A 容差域，A7 族登记）。
     * 整块被掩码时 cur_max = -1e30、cur_sum = 0（调用方跳过 Step，twin 5.1 同款）。
     */
    __aicore__ void ComputeBlockMaxAndSum(uint32_t row, float &cur_max, float &cur_sum, uint32_t k_idx)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        (void)k_idx;
        const uint32_t token_q = cur_q_start_ + (row % cur_q_len_);
        LocalTensor<int32_t> ub_dot(TPosition::VECCALC,
                                    SAGE_UB_DOT32_OFFSET + row * block_n_ * sizeof(int32_t), block_n_);
        LocalTensor<float> ub_logits(TPosition::VECCALC, SAGE_UB_V1_LOGITS_ROW_OFFSET, block_n_);
        LocalTensor<float> ub_ks(TPosition::VECCALC, SAGE_UB_V1_KS_ROW_OFFSET, block_n_);
        LocalTensor<float> ub_qs(TPosition::VECCALC, SAGE_UB_V1_QS_ROW_OFFSET, STAGE_SLOT_FLOATS);
        LocalTensor<float> ub_stats(TPosition::VECCALC, SAGE_UB_V1_STATS_OFFSET, 128);
        // 行解析 q_scale：StageV1ScaleStrips 已以 MTE2_S 排空 S 管读取序
        const float q_scale = ub_qs.GetValue(row);
        dev_vf::DequantMaskLogitsRowVF(reinterpret_cast<__ubuf__ int32_t*>(ub_dot.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ float*>(ub_logits.GetPhyAddr()),
                                       q_scale,
                                       reinterpret_cast<__ubuf__ float*>(ub_ks.GetPhyAddr()),
                                       tiling_->sm_scale_log2e, token_q, cur_k_start_, sk_,
                                       tiling_->is_causal, block_n_);
        dev_vf::BlockMaxAndSumVF(reinterpret_cast<__ubuf__ float*>(ub_logits.GetPhyAddr()),
                                 block_n_,
                                 reinterpret_cast<__ubuf__ float*>(ub_stats.GetPhyAddr()));
        Catlass::SageAttn::Sync<HardEvent::V_S>();
        cur_max = ub_stats.GetValue(0);
        cur_sum = ub_stats.GetValue(64);
#else
        if (sim_mem_ == nullptr) {
            cur_max = 0.0f;
            cur_sum = 1.0f;
            return;
        }
        (void)k_idx;
        const bool per_thread = (tiling_->qk_quant_gran == QUANT_GRAN_PER_THREAD);
        const uint32_t g = row / cur_q_len_;
        const uint32_t token_q = cur_q_start_ + (row % cur_q_len_);
        const float q_scale =
            sim_mem_->q_scales[g * 2048u + quant::QScaleIndex(token_q, per_thread)];
        float *logits_row = &sim_mem_->logits[static_cast<size_t>(row) * block_n_];
        const int8_t *l1_q = reinterpret_cast<const int8_t *>(
            sim_mem_->l1_mem + tiling_->l1_q_offset);
        const int8_t *l1_k = reinterpret_cast<const int8_t *>(
            sim_mem_->l1_mem + CurKvSlotBase());

        for (uint32_t j = 0; j < block_n_; ++j) {
            const uint32_t token_k = cur_k_start_ + j;
            if (token_k >= sk_ || (tiling_->is_causal != 0 &&
                                   static_cast<int64_t>(token_k) > static_cast<int64_t>(token_q))) {
                logits_row[j] = -1.0e30f;
                continue;
            }
            const int8_t *qr = l1_q + static_cast<size_t>(row) * d_;
            const int8_t *kr = l1_k + static_cast<size_t>(j) * d_;
            int32_t dot = 0;
            for (uint32_t c = 0; c < d_; ++c) {
                dot += qr[c] * kr[c];
            }
            logits_row[j] = twin::DequantQkLogit(
                dot, q_scale,
                sim_mem_->k_scales[quant::KScaleIndex(token_k, per_thread)],
                tiling_->sm_scale_log2e);
        }

        float row_max = -1.0e30f;
        for (uint32_t j = 0; j < block_n_; ++j) {
            if (logits_row[j] > row_max) row_max = logits_row[j];
        }
        if (row_max <= -1.0e30f) {
            cur_max = -1.0e30f;
            cur_sum = 0.0f;
            return;
        }
        cur_max = row_max;
        cur_sum = twin::OnlineBlockExp2Sum(logits_row, cur_k_start_,
                                           cur_k_start_, cur_k_start_ + block_n_,
                                           tiling_->is_causal != 0, token_q, row_max);
#endif
    }

    /**
     * 5. V1：O 主累加器单行原地重缩放。twin 5.3 口径为纯 fp32 乘——即使 fp32+fp16 档
     * 也不经 FP16 中间舍入（FP16 域累加仅发生在 PV 归约路径，见
     * twin::QuantizePAndAccumulatePvRow），与 RunSequential 逐位一致。
     */
    __aicore__ void RescaleAccumulator(float rescale_factor, uint32_t row)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        LocalTensor<float> ub_accum(TPosition::VECCALC,
                                    Catlass::SageAttn::Vector1UbBudget::O_ACCUM_OFFSET + row * d_ * sizeof(float),
                                    d_);
        dev_vf::RescaleAccumRowVF(reinterpret_cast<__ubuf__ float*>(ub_accum.GetPhyAddr()),
                                  rescale_factor, d_);
#else
        if (sim_mem_ == nullptr) {
            return;
        }
        float *o_row = &sim_mem_->o_accum[static_cast<size_t>(row) * d_];
        for (uint32_t c = 0; c < d_; ++c) {
            o_row[c] *= rescale_factor;
        }
#endif
    }

#if !SAGE_ATTN_DEVICE_COMPILE
    /**
     * 6. V1（宿主/KT 标量基线）：P 矩阵整幅量化写 S(P) 双缓冲槽（SoftmaxSlotOffset(is_ping)，
     * 与 Cube 侧 SagePvMmad::LoadL0A 的 ub_softmax_offset 读取约定一致）。数值为
     * twin::SoftmaxProbCode 单点：P = FastExp2(logit - m_scale) · 448，
     * 饱和 + RNE 转 e4m3。448· 放大经乘法实现（行 max 恰为 448 填满 e4m3 动态范围）。
     * 行锚点取 row_states_[r].m_scale（S 与 O 同锚）；掩码哨兵 -1e30 经 exp2 下溢自然落到
     * e4m3 码 0。设备态对应原语为上行循环内的 SoftmaxProbRow（逐行同式，掩码行同值码 0）。
     */
    __aicore__ void ComputeSoftmaxProbability(
        bool is_ping)
    {
        if (sim_mem_ == nullptr) {
            return;
        }
        uint8_t *slot_ptr = sim_mem_->ub_mem + SoftmaxSlotOffset(is_ping);
        const uint32_t rows = TotalRows();
        for (uint32_t r = 0; r < rows && r < kKernelMaxRows; ++r) {
            const float m_prime = row_states_[r].m_scale;
            const float *logits_row = &sim_mem_->logits[static_cast<size_t>(r) * block_n_];
            uint8_t *slot_row = slot_ptr + static_cast<size_t>(r) * block_n_;
            for (uint32_t j = 0; j < block_n_; ++j) {
                slot_row[j] = twin::SoftmaxProbCode(logits_row[j], m_prime);
            }
        }
    }
#endif

#if !SAGE_ATTN_DEVICE_COMPILE
    /**
     * 7pre. V1（宿主串行仿真专属）：第 k 块 KV 的 L1 槽重摆（幂等）。
     * 宿主串行时序说明：宿主按 ProcessVec0 → ProcessCube → ProcessVec1 整段执行，
     * V0 会先把全部 KV 块写入 ping/pong 双槽，V1 消费前重摆当前块：同 GM 输入走同一量化
     * 轨迹（K 组清零→amax→定档→量化、V 首块全序列 amax 预扫→per-channel 量化均幂等），
     * 逐位一致，仅宿主分支生效。
     */
    __aicore__ void RestageKvBlockHost(uint32_t k)
    {
        if (sim_mem_ == nullptr) {
            return;
        }
        if (k >= active_kv_blocks_) {
            return;
        }
        cur_k_ = k;
        cur_k_start_ = k * block_n_;
        const uint32_t cur_block_k = CurBlockK();
        const bool is_ping = ((k % 2) == 0);
        const uint32_t l1_dst_offset = is_ping
                                           ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                           : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET;

        const int64_t b_idx = static_cast<int64_t>(cur_b_);
        const int64_t hkv_idx = static_cast<int64_t>(cur_hkv_);
        const int64_t sk_start = static_cast<int64_t>(cur_k_start_);

        int64_t kv_stride_b, kv_stride_h, kv_stride_s;
        if ((tiling_->tensor_layout == LAYOUT_HND)) {
            kv_stride_b = static_cast<int64_t>(hkv_) * sk_ * d_;
            kv_stride_h = static_cast<int64_t>(sk_) * d_;
            kv_stride_s = d_;
        } else {
            kv_stride_b = static_cast<int64_t>(sk_) * hkv_ * d_;
            kv_stride_s = static_cast<int64_t>(hkv_) * d_;
            kv_stride_h = d_;
        }

        const int64_t kv_gm_offset = b_idx * kv_stride_b + hkv_idx * kv_stride_h + sk_start * kv_stride_s;
        const __gm__ float *km_ptr = nullptr;
        if ((tiling_->smooth_k != 0) && key_mean_gm_ != nullptr) {
            const int64_t km_offset = (b_idx * static_cast<int64_t>(hkv_) + hkv_idx) * d_;
            km_ptr = reinterpret_cast<const __gm__ float *>(key_mean_gm_ + km_offset * sizeof(float));
        }

        QuantizeAndStageKey(key_gm_ + kv_gm_offset * sizeof(uint16_t), km_ptr, l1_dst_offset,
                            cur_block_k, d_);
        QuantizeAndStageValue(value_gm_ + kv_gm_offset * sizeof(uint16_t),
                              l1_dst_offset + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET,
                              cur_block_k, d_);
    }

    /**
     * 7a. V1（KT 标量基线专属）：P 槽写出后的 PV 累加，经 twin::QuantizePAndAccumulatePvRow
     * 单点执行（5.4 唯一定义）：fp32 档逐 token 直累；fp32+fp16 档按 k_step=16 子块在
     * FP16 域累加后并入主累加器。V 反量化按 twin v_deq 同式（e4m3 解码 · 通道 scale，
     * 16-token permute 槽寻址）。真实硬件该步骤由 Cube PV MMAD + FixPipe 承担（详见 ProcessCube）。
     */
    __aicore__ void AccumulatePvBlockTwin(uint32_t block_k)
    {
        if (sim_mem_ == nullptr) {
            return;
        }
        const bool fp16_accum = (tiling_->pv_accum_dtype == PV_ACCUM_FP32_FP16);
        const uint8_t *l1_v = sim_mem_->l1_mem + CurKvSlotBase() +
                              Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET;
        const uint32_t rows = TotalRows();
        for (uint32_t r = 0; r < rows && r < kKernelMaxRows; ++r) {
            const uint32_t token_q = cur_q_start_ + (r % cur_q_len_);
            const float m_prime = row_states_[r].m_scale;
            twin::QuantizePAndAccumulatePvRow(
                &sim_mem_->o_accum[static_cast<size_t>(r) * d_], d_,
                &sim_mem_->logits[static_cast<size_t>(r) * block_n_], cur_k_start_,
                cur_k_start_, cur_k_start_ + block_k,
                tiling_->is_causal != 0, token_q, m_prime,
                fp16_accum, k_step_,
                [&](uint32_t j, uint32_t c) -> float {
                    const uint32_t jl = j - cur_k_start_;
                    const uint32_t slot = (jl / 16u) * 16u + quant::V_PERM_INV_PATTERN[jl % 16u];
                    return quant::E4m3fnToFloat(l1_v[static_cast<size_t>(c) * block_n_ + slot]) *
                           sim_mem_->v_scales[c];
                });
        }
    }
#endif

    /**
     * 7. V1：最终归一化与 LSE 换算写回 GM（twin 5.5/5.6 单点轨迹）：
     * - 行级 inv_d = GetFinalInvDenominator(row_states_[r])（O 与分母同处 448· 放大域，
     *   除法处放大精确相消）；
     * - LSE 出口 = ConvertLseToHost(lse, sm_scale, lse_correction, smooth_k,
     *   S_FP8_OFFSET)：核内 exp2 域 LSE 含 448· 放大（乘法实现）所致的
     *   +log2(448) 偏置，出口以 S_FP8_OFFSET 常量补扣；smooth_k 的行级
     *   lse_correction · sm_scale 项在此并入。
     */
    __aicore__ void FinalizeAndStoreOutput()
    {
        int64_t b_idx = static_cast<int64_t>(cur_b_);
        int64_t hkv_idx = static_cast<int64_t>(cur_hkv_);
        int64_t sq_start = static_cast<int64_t>(cur_q_start_);

        int64_t out_stride_b, out_stride_h, out_stride_s;
        if ((tiling_->tensor_layout == LAYOUT_HND)) {
            out_stride_b = static_cast<int64_t>(hq_) * sq_ * d_;
            out_stride_h = static_cast<int64_t>(sq_) * d_;
            out_stride_s = d_;
        } else {
            out_stride_b = static_cast<int64_t>(sq_) * hq_ * d_;
            out_stride_s = static_cast<int64_t>(hq_) * d_;
            out_stride_h = d_;
        }

#if SAGE_ATTN_DEVICE_COMPILE
        if (out_gm_ == nullptr) return;
        const bool is_smooth_k = (tiling_->smooth_k != 0);
        LocalTensor<half> ub_half(TPosition::VECCALC, Catlass::SageAttn::Vector1UbBudget::FIXPIPE_RECV_OFFSET, d_);

        for (uint32_t g = 0; g < gqa_ratio_; ++g) {
            const int64_t hq_curr = hkv_idx * static_cast<int64_t>(gqa_ratio_) + g;
            if (hq_curr >= hq_) break;

            for (uint32_t i = 0; i < cur_q_len_; ++i) {
                const uint32_t r = g * cur_q_len_ + i;
                if (r >= kKernelMaxRows) break;
                const float inv_row = OnlineSoftmaxTransformer::GetFinalInvDenominator(row_states_[r]);
                const float lse_corr_val = (r < kKernelMaxRows) ? lse_corr_[r] : 0.0f;
                // LSE 出口：核内 exp2 域 LSE 含 448· 放大（乘法实现）所致的 +log2(448) 偏置，
                // 以 S_FP8_OFFSET 常量补扣（唯一定义，见 ConvertLseToHost 注释）
                const float lse_host = OnlineSoftmaxTransformer::ConvertLseToHost(
                    row_states_[r].lse, tiling_->sm_scale_log2e * INV_LOG2E,
                    lse_corr_val, is_smooth_k, S_FP8_OFFSET);

                LocalTensor<float> ub_o_row(TPosition::VECCALC,
                                            Catlass::SageAttn::Vector1UbBudget::O_ACCUM_OFFSET + r * d_ * sizeof(float),
                                            d_);
                dev_vf::FinalizeRowVF(reinterpret_cast<__ubuf__ float*>(ub_o_row.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ uint16_t*>(ub_half.GetPhyAddr()),
                                     inv_row, tiling_->d_og, tiling_->input_dtype);
                // V 管写 → MTE3 读（DEFECT-I-1）：DataCopy 前置 V→MTE3 事件对，
                // 保证本行 VF 写毕才发起 GM 搬出（支持 half 与 bfloat16 双路径，DEFECT-F-3）
                SyncVToMte3();

                const int64_t out_gm_offset =
                    (b_idx * out_stride_b + hq_curr * out_stride_h +
                     (sq_start + i) * out_stride_s) * sizeof(half);
                if (tiling_->input_dtype == 2u) {
                    GlobalTensor<bfloat16_t> gm_out;
                    gm_out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out_gm_ + out_gm_offset));
                    LocalTensor<bfloat16_t> ub_bf16 = ub_half.ReinterpretCast<bfloat16_t>();
                    DataCopy(gm_out, ub_bf16, tiling_->d_og);
                } else {
                    GlobalTensor<half> gm_out;
                    gm_out.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out_gm_ + out_gm_offset));
                    DataCopy(gm_out, ub_half, tiling_->d_og);
                }
                // ub_half 逐行复用：上一行在途 MTE3 读须先于下一行 VF 写排空
                //（DEFECT-H-1/I-1：PipeBarrier<PIPE_V> 不排 MTE3，改 MTE3→V 事件对）
                SyncMte3ToV();

                if (lse_gm_ != nullptr) {
                    const int64_t lse_offset =
                        ((b_idx * static_cast<int64_t>(hq_) + hq_curr) * sq_ + (sq_start + i)) * sizeof(float);
                    *reinterpret_cast<__gm__ float*>(lse_gm_ + lse_offset) = lse_host;
                }
            }
        }
#else
        if (sim_mem_ == nullptr || out_gm_ == nullptr) {
            return;
        }
        const bool is_smooth_k = (tiling_->smooth_k != 0);
        for (uint32_t g = 0; g < gqa_ratio_; ++g) {
            const int64_t hq_curr = hkv_idx * static_cast<int64_t>(gqa_ratio_) + g;
            if (hq_curr >= hq_) break;

            for (uint32_t i = 0; i < cur_q_len_; ++i) {
                const uint32_t r = g * cur_q_len_ + i;
                if (r >= kKernelMaxRows) break;
                const float inv_row =
                    OnlineSoftmaxTransformer::GetFinalInvDenominator(row_states_[r]);
                const float lse_host = OnlineSoftmaxTransformer::ConvertLseToHost(
                    row_states_[r].lse, tiling_->sm_scale_log2e * INV_LOG2E,
                    sim_mem_->lse_corr[r], is_smooth_k, S_FP8_OFFSET);

                for (uint32_t c = 0; c < tiling_->d_og; ++c) {
                    const float val = sim_mem_->o_accum[static_cast<size_t>(r) * d_ + c] * inv_row;
                    const int64_t out_gm_offset =
                        (b_idx * out_stride_b + hq_curr * out_stride_h +
                         (sq_start + i) * out_stride_s + c) * sizeof(uint16_t);
                    WriteOutputBits(reinterpret_cast<uint16_t *>(out_gm_) + out_gm_offset / sizeof(uint16_t), 0, val);
                }

                if (lse_gm_ != nullptr) {
                    const int64_t lse_offset =
                        ((b_idx * static_cast<int64_t>(hq_) + hq_curr) * sq_ + (sq_start + i)) *
                        sizeof(float);
                    *reinterpret_cast<float *>(lse_gm_ + lse_offset) = lse_host;
                }
            }
        }
#endif
    }

    // 内部状态变量
#if !SAGE_ATTN_DEVICE_COMPILE
    // 宿主仿真缓冲（unique_ptr 生命周期管理）
    std::unique_ptr<SimMemory> sim_mem_{nullptr};
#endif
    const SageAttention2TilingData *tiling_{nullptr};
    __gm__ uint8_t *query_gm_{nullptr};
    __gm__ uint8_t *key_gm_{nullptr};
    __gm__ uint8_t *value_gm_{nullptr};
    __gm__ uint8_t *key_mean_gm_{nullptr};
    // smooth_k 的 GM 侧 lse_correction 直读通道（注册入参 4）：基线数值由核内
    // Q·km 单点计算（twin 契约，LoadAndQuantizeQ 步 4），该通道仅完成指针承接，
    // 消费留给向量化 pass（届时以非空判优，避免与核内计算双份叠加）。
    __gm__ uint8_t *lse_correction_gm_{nullptr};
    __gm__ uint8_t *out_gm_{nullptr};
    __gm__ uint8_t *lse_gm_{nullptr};
    __gm__ uint8_t *workspace_gm_{nullptr};

    uint32_t core_idx_{0};
    uint32_t sub_core_idx_{0};
    bool is_active_core_{false};

    uint32_t b_{0};
    uint32_t hq_{0};
    uint32_t hkv_{0};
    uint32_t sq_{0};
    uint32_t sk_{0};
    uint32_t d_{0};
    uint32_t gqa_ratio_{1};

    uint32_t m_per_core_{64};
    uint32_t block_n_{64};
    uint32_t k_step_{DEFAULT_K_STEP};

    uint32_t cur_b_{0};
    uint32_t cur_hkv_{0};
    uint32_t cur_q_blk_{0};
    uint32_t cur_q_start_{0};
    uint32_t cur_q_len_{0};
    uint32_t cur_k_{0};
    uint32_t cur_k_start_{0};
    uint32_t active_kv_blocks_{0};

    // 行级 online softmax 状态（GQA 逻辑行 r = g * cur_q_len + i 各自独立维护，
    // 与 twin RunSequential 的行级状态一致；真实硬件驻留 V1 UB LSE_WORKSPACE 区）
    OnlineSoftmaxState row_states_[kKernelMaxRows];

    // smooth_k 行级 lse_correction 统一状态（全分支共享，杜绝设备态硬编码 0.0f）
    float lse_corr_[kKernelMaxRows]{0.0f};

    AccumReduceController accum_ctrl_;
};

} // namespace sage_attn

/**
 * AscendC 全局设备核函数入口 (C 链接)，函数名须与 op 注册名一致（小写下划线）——
 * CANN opc autogen 以 origin_func_name="sage_attention2" 在内核源文件中检索入口，
 * 名称不符报 "kernel entry not implement"。
 * 声明 KERNEL_TYPE_MIX_AIC_1_2 硬件拓扑
 */
extern "C" __global__ __aicore__ void sage_attention2(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *keyMean,
    __gm__ uint8_t *lseCorrection,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    // 入参顺序契约：INPUT(query,key,value) → OPTIONAL_INPUT(key_mean,lse_correction) →
    // OUTPUT(output,lse) → workspace → tiling（严格对齐 sage_attention2_def.cpp 注册顺序，
    // opc autogen 按此顺序生成 launch 调用，缺参/错序报 "requires 8 arguments, but 9"）。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // canonical 注册三件套（对齐仓内规范；宿主/KT 分支为等价桩）：
    // REGISTER_TILING_DEFAULT 登记 TilingData 类型；GET_TILING_DATA_WITH_STRUCT 把
    // GM tiling 一次性拷入核内结构体（免 __gm__ 逐字段远端读）；GetUserWorkspace
    // 取得跳过 CANN 系统 workspace 前缀后的用户区指针。
#if defined(__CCE__) && !defined(__CCE_KT_TEST__)
    REGISTER_TILING_DEFAULT(sage_attn::SageAttention2TilingData);
    GET_TILING_DATA_WITH_STRUCT(sage_attn::SageAttention2TilingData, t, tiling);
    __gm__ uint8_t *user_workspace = GetUserWorkspace(workspace);
    sage_attn::SageAttnKernel kernel;
    kernel.Init(query, key, value, keyMean, lseCorrection, attentionOut, softmaxLse, user_workspace, &t);
#else
    (void)workspace;
    // 基线 lse_correction 由核内 Q·km 单点计算（twin 契约），GM 直读通道留给向量化 pass
    const sage_attn::SageAttention2TilingData *tiling_data =
        reinterpret_cast<const sage_attn::SageAttention2TilingData *>(tiling);
    sage_attn::SageAttnKernel kernel;
    kernel.Init(query, key, value, keyMean, lseCorrection, attentionOut, softmaxLse, nullptr, tiling_data);
#endif
    kernel.Process();
}
