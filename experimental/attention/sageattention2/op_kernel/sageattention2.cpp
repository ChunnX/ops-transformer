// 2026-10-03 自 sageattn_fp8/ascendc/sage_attn_kernel.cpp 物理迁移至本路径
// （CANN opc autogen 打包约定：优先解析 <op>/op_kernel/<opFile>.cpp，布局 A；
// sageattn_fp8/ascendc/ 仍保留 tiling/softmax/accum/sync 等头文件与宿主孪生实现）。

// 注意：本文件位于 op_kernel/，包含 ascendc/ 头必须用相对路径——
// 裸 "tiling.h" 会按 -I 顺序误解析到 CANN 的 adv_api/matmul/tiling.h
//（bisheng 编译门的 -W#pragma-message 曾捕获该错位）。
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/ascendc/online_softmax.h"
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
#include <algorithm>
#include <memory>

// 宿主/KT 孪生执行分支数值真值路径：QK 反量化 logit、块内 exp2 和、P 量化与
// PV 行累加均经 twin_reference.h 单点 include 复用（设计决策 5：语义公式禁止在内核
// 内复制第二份）。twin_reference.h 携带宿主 STL 载体（std::vector），设备分支
// （bisheng -xasc，SAGE_ATTN_DEVICE_COMPILE=1）完全由 AscendC 硬件原语与
// dev_vf 向量化流水算子承载，不引入宿主 STL 依赖。
#if !SAGE_ATTN_DEVICE_COMPILE
#include "twin_reference.h"
#endif
/**
 * 设备与宿主分支切换实现：
 * 以 __CCE__（BiSheng 设备编译恒定义，BiSheng 宿主模式恒不定义）判定设备分支。
 */
#if SAGE_ATTN_KERNEL_API
#include "kernel_operator.h"
#if SAGE_ATTN_DEVICE_COMPILE
// canonical 注册三件套：正式出包由 ops-transformer 构建体系注入同义宏；
// 独立 bisheng 编译门（run.sh kernel）经此处 #ifndef 定义补齐，
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
#endif

// M5 编译期位型特化唯一定义点：输入位型档位常量 / NormDtypeMode 归一 /
// tiling key 编码 / canonical ASCENDC_TPL_ARGS_DECL 声明（设备编译仅 OPP
// autogen 态纳入 canonical 头，裸 bisheng 门禁自动回落 RunByDtype 形态）。
// 置于 using namespace AscendC 之后：canonical 头 KERNEL 态的 TypeFromId
// DT_* 特化按无限定名引用 AscendC 枚举，须域注入先于 include。
#include "sage_attention2_template_tiling_key.h"

/**
 * 核内域判定宏：bisheng 设备分支中 ASCEND_IS_AIC 展开为编译期谓词 constexpr(...)，
 * 官方调用范式为无括号 `if ASCEND_IS_AIC {`（内置算子 geglu_3510_impl.h 同款）；
 * 宿主/KT 分支展开为普通布尔，按 C 风格加括号。统一经 ASCEND_IS_AIC/AIV 书写。
 */
namespace sage_attn {

// MTE/V/S/FIX 管线序事件对统一走官方 CATLASS 侧单点 Catlass::SageAttn::Sync<E>
//（sage_tile.hpp；SetFlag/WaitFlag 同 ID 0 一比一配对，宿主分支编译期剔除）：
// - Sync<HardEvent::MTE2_V>：GM→UB DataCopy（MTE2 写）毕、VF 读前；
// - Sync<HardEvent::V_MTE2>（R22 回边）：VF 读毕（V 管）、同槽被下一轮 MTE2 写复用前
//  （MTE2 重写不 await V 管在途读，须显式事件对）；
// - Sync<HardEvent::V_MTE3>：VF 写毕、MTE3 读前；
// - Sync<HardEvent::MTE3_V>（DEFECT-I-1）：UB→GM DataCopy（MTE3 读）毕、同槽被
//   下一轮 VF 写复用前（PipeBarrier<PIPE_V> 不排 MTE3 在途读）；
// - Sync<HardEvent::S_MTE3 / MTE3_S>：标量 SetValue 与 MTE3 搬出互排；
// - Sync<HardEvent::MTE2_S>：MTE2 写毕、S 管读前；
// - Sync<HardEvent::MTE3_MTE2>：V0 的 MTE3 写 KV 槽毕、Cube MTE 读回边。
// 原四个 no-op 宿主包装（SyncMte2ToV 等）与文件内散装 SetFlag/WaitFlag 直配对
// 已删——事件对单点化后本文件不再持有任何裸 SetFlag/WaitFlag。

// GQA 聚合后的窗口行上限（m_per_core 为 64 的倍数、gqa_ratio ≤ 8 的验收域内恒满足；
// 常量不随 SimMemory 分支门控，row_states_ 成员在设备分支亦需该界）
// 行打包上限唯一定义点：tiling.h MAX_PACK_ROWS（host/InspectTilingPlan 同源）
// 标量 min/max 复用官方 CATLASS 对齐组件（catlass/detail/alignment.hpp 全局
// Min/Max，CATLASS_HOST_DEVICE 强圈内联，同 ternary 语义零开销），不再自造。

// A7 设备态 UB 布局（V1 256KB 地址空间，task/12_device_data_path.md；R17-2 重排定案）：
// 区界/互斥/封顶的唯一定义点为 sage_resource.hpp Vector1UbBudget（static_assert 链
// 编译期锁定两两不相交且 ≤ 256KB，512B 对齐），本文件仅保留调用点别名与 V1 行级
// 暂存子槽——重排后 S pong 槽（≤96KB）不再被 Cube dot32 裸搬（96KB 起）踩踏：
//   O_ACCUM [0,64K) | S 双槽 [64K, ≤96K) | dot32 [96K,160K) | pv_partial [160K,224K)
//   | LSE 区 [224K,240K)：行 max(+0)/行 exp2 和(+512)/ks 行(+1K)/qs 行(+1.5K)/vs(+2K)
//     /vm(+2.5K)/corr 行(+3K)/lse 行(+3.5K)/行 P 锚 m'·ln2(+4K)/行 rescale 系数(+4.5K)
//   | TWO_LEVEL_REDUCE [240K,256K)
// R4 行批化注：原 logits 行暂存槽(+512)在反量化改驻 dot32 区（元素等宽原地覆写，
// 读保护窗不变）后释放为行 exp2 和槽；行 max/和/锚/系数四槽均为 1 float/行
//（VL1 lane0 写、GetValue/DIST_BRC_B32 读），行距无槽间越界。
using Catlass::SageAttn::Vector1UbBudget;
[[maybe_unused]] constexpr uint32_t SAGE_UB_DOT32_OFFSET = Vector1UbBudget::A7_DOT32_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_PV_PARTIAL_OFFSET = Vector1UbBudget::A7_PV_PARTIAL_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_V1_STATS_OFFSET = Vector1UbBudget::LSE_WORKSPACE_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_V1_ROW_MAX_OFFSET = Vector1UbBudget::V1_ROW_MAX_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_V1_ROW_SUM_OFFSET = Vector1UbBudget::V1_ROW_SUM_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_V1_KS_ROW_OFFSET = Vector1UbBudget::V1_KS_ROW_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_V1_QS_ROW_OFFSET = Vector1UbBudget::V1_QS_ROW_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_V1_VS_CHAN_OFFSET = Vector1UbBudget::V1_VS_CHAN_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_V1_VM_CHAN_OFFSET = Vector1UbBudget::V1_VM_CHAN_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_V1_CORR_ROW_OFFSET = Vector1UbBudget::V1_CORR_ROW_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_V1_LSE_ROW_OFFSET = Vector1UbBudget::V1_LSE_ROW_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_V1_M_LN_ROW_OFFSET = Vector1UbBudget::V1_M_LN_ROW_OFFSET;
[[maybe_unused]] constexpr uint32_t SAGE_UB_V1_RS_FACTOR_ROW_OFFSET = Vector1UbBudget::V1_RS_FACTOR_ROW_OFFSET;

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
 * - v_means：smooth_v 的 per-channel 均值（V0 预扫产出，PV 反量化时加回，[通道]）。
 * GQA 行 = g * cur_q_len + i（g 为 KV 头内 Q 头序号），上限 MAX_PACK_ROWS。
 */
struct SimMemory {
    uint8_t l1_mem[512 * 1024]{0};
    uint8_t ub_mem[512 * 1024]{0};
    float o_accum[128 * 128]{0.0f};
    float logits[128 * 128]{0.0f};
    float q_scales[MAX_GQA_RATIO * kQScaleSlotStride]{0.0f};
    float k_scales[kKScaleSlotStride]{0.0f};
    float v_scales[256]{0.0f};
    float v_means[256]{0.0f};
};
#endif

/**
 * SageAttention2 1C2V 设备核执行器
 * 核心技术规范：
     * 1. 1C2V 硬件编排：1 个 Cube Core 搭配 2 个 Vector Core (V0 量化 prologue 核，V1 Softmax 与 LSE 核)
     * 2. 延迟一拍流水：Cube 与 Vector 双域交错掩盖计算与搬运延时
     * 3. GQA 头聚合：当 Hq > Hkv 时沿 M 维聚合至 64 的倍数，一份 L1 KV 被多个 Q head 连续复用
     * 4. 64 位 GM 寻址：所有显存指针与 stride 计算强转 int64_t，杜绝长序列整数溢出
     * 5. 跨核 Flag 同步保序：严格遵循生产者在前、消费者在后；set/wait 配对按 flag 族分治——
     *    仅 S 槽 FREE 对由 Cube 进域预置"消费者已归还"许可、V1 退域补等配平（含预置/回收 1:1），
     *    KV/DOT/PARTIAL 族无预置/回收，靠首拍缓冲区天然为空跳过 wait（域内残余许可供
     *    下一波次首槽直写，不跨任务死锁，task/02_code.md B-2）
     *
     * 数值架构契约（Step-C 标量基线，与 twin_reference.h RunSequential 唯一真值逐位一致，
     * 数值验证入口为 tests/test_kernel_twin.py（宿主串行分支 + 孪生执行器对 golden 25 桶
     * L-A/L-B 交叉验证，即本文件宿主分支的 golden 级回归）、tests/test_golden_precision.py
     * （GPU golden 对标）与 tests/test_function.py（功能用例））：
 * a. 逐行 Softmax：OnlineSoftmaxState 按逻辑行（r = g*cur_q_len + i）维护，行 max/行分母/
 *    行 rescale/行 LSE/行 lse_correction 全部独立——softmax 是行级语义；
 * b. P 逐块锚定（恒逐块强锚定重缩放）：m_scale 恒等于 m_run，P 在本块锚点量化——
 *    twin/golden 的逐块锚定轨迹（e4m3 量化网格依赖锚点，锚点滞后会偏离 golden 网格）；
 * c. 448· 放大域经乘法实现（twin::SoftmaxProbCode 单点，行 max 恰为 448 填满 e4m3
 *    动态范围），LSE 出口恒以 S_FP8_OFFSET 常量补扣；
 * d. 两级累加：fp32 档逐 token 直累；fp32+fp16 档按 k_step=16 子块 FP16 域累加
 *    （twin::AccumulatePvRow 单点），O 主累加器驻留 V1 UB（fp32）。
 *
 * M5 编译期位型特化：类模板化于 DTYPE_MODE（归一化输入位型档位，1=fp16 / 2=bf16，
 * 归一唯一定义 sage_attn::NormDtypeMode）；原运行时 DtypeMode() 消费点全部改经
 * 编译期常量 kDtype（if constexpr / VF 模板实参），设备热路径零位型分支，各档位
 * 实例化执行的算术指令与原运行时分支逐一相同（逐位不变特化）。
 */
template <uint32_t DTYPE_MODE>
class SageAttnKernel {
public:
    // 编译期输入位型档位（原 DtypeMode() 的编译期形态）
    static constexpr uint32_t kDtype = NormDtypeMode(DTYPE_MODE);


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
        workspace_gm_ = workspace; // 遵循 CANN 算子签名契约：用户 Workspace 指针经 GetUserWorkspace 直达入口；
                                   // 非签名占位——A7 scale 交接条带（StageStrip，core_idx × STAGE_CORE_STRIDE
                                   // 定址）与 smooth_v 均值区（workspace_vm_offset）均自本指针实址消费。

        // 1C2V (KERNEL_TYPE_MIX_AIC_1_2) 模式下：
        // AIC 核心数 N（GetBlockIdx: 0..N-1），AIV 核心数 2N（GetBlockIdx: 0..2N-1）。
        // AIV 侧必须 core_idx_ /= 2，使其与同物理核的 AIC core_idx 对齐；
        // AIV 侧区分 Vector 0 与 Vector 1 使用 AscendC::GetSubBlockIdx() (返回 0 或 1)。
        core_idx_ = static_cast<uint32_t>(GetBlockIdx());
        if ASCEND_IS_AIV {
            sub_core_idx_ = static_cast<uint32_t>(AscendC::GetSubBlockIdx());
            core_idx_ /= 2;
        } else {
            sub_core_idx_ = 0u;
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

        // GQA 分块打包（官方 500 用例含 32:4 即 8:1）：单任务打包 pack_heads_ 个 Q 头，
        // gqa_ratio 超出行打包上限时按 gqa_chunks_ 拆多任务；chunk 内量化窗口恒为
        // m_per_core（32/64）token，不切割 32-token 量化子块（Q 组语义保持）。
        const uint32_t pack_cap = (m_per_core_ > 0u) ? (MAX_PACK_ROWS / m_per_core_) : 1u;
        pack_heads_ = (gqa_ratio_ < pack_cap) ? gqa_ratio_ : pack_cap;
        gqa_chunks_ = (pack_heads_ > 0u) ? ::CeilDiv(gqa_ratio_, pack_heads_) : 1u;

        // 计算当前核的任务归属与 DieTask 解耦入口
        // 沿 (Batch, Hkv, GQA_chunk, Sq_block) 空间切分，同一 KV 头的多个 Q 头优先聚合在同一核内执行
        // 注意：tiling_ 为 const __gm__ 限定，bisheng dav-3510 禁止在设备域调用其 const 成员函数
        //（报 reference to __host__ function / this 未带 __gm__ 限定），故核内一律直接读字段。
        // 向上取整统一走官方 CATLASS 对齐组件（::CeilDiv，与手写 (a+b-1)/b 位元一致）
        uint32_t q_blks_per_head = ::CeilDiv(sq_, m_per_core_);
        uint32_t total_tasks = b_ * hkv_ * gqa_chunks_ * q_blks_per_head;
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

        uint32_t q_blks_per_head = ::CeilDiv(sq_, m_per_core_);
        uint32_t total_tasks = b_ * hkv_ * gqa_chunks_ * q_blks_per_head;
        uint32_t cores = tiling_->core_num_used;
        // DieTask 分发：在奇 cores 下防跨 die 错配
        bool use_die = (tiling_->enable_die_affinity != 0) && (cores % 2u == 0u);
        const uint64_t task_end = DieTaskEnd(total_tasks, core_idx_, cores, use_die);

        // 循环执行分配给当前物理核的所有波次任务 (Die 内部严格亲和)
        // 跨任务波次 flag 纪律（task/02_code.md B-2，按族分治）：仅 S 槽 FREE 对由 Cube
        // 进域预置"已归还"许可、V1 退域补等配平；KV/DOT/PARTIAL 族无预置/回收，依赖
        // 首拍缓冲区天然为空跳过 wait（波次边界残余许可不影响下一任务首两拍直写）。
        for (uint64_t virtual_task = core_idx_; virtual_task < task_end; virtual_task += cores) {
            // 当虚拟任务推进至后续波次时重新解码任务
            if (virtual_task > core_idx_) {
                uint64_t mapped_task = use_die ? DieTask(virtual_task, total_tasks, cores) : virtual_task;
                DecodeTask(mapped_task, q_blks_per_head);
            }

#if SAGE_ATTN_KERNEL_API
            if ASCEND_IS_AIC {
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
     * 任务解码辅助函数：将一维任务索引解耦为 (Batch, Hkv, GQA_chunk, Q_blk) 并更新当前 Token 范围与累加控制器
     */
    __aicore__ void DecodeTask(uint64_t task_idx, uint32_t q_blks_per_head)
    {
        int64_t t = static_cast<int64_t>(task_idx);
        int64_t q_blk_id = t % q_blks_per_head;
        int64_t rem = t / q_blks_per_head;
        int64_t g_chunk = rem % gqa_chunks_;
        rem /= gqa_chunks_;
        int64_t hkv_id = rem % hkv_;
        int64_t b_id = rem / hkv_;

        cur_b_ = static_cast<uint32_t>(b_id);
        cur_hkv_ = static_cast<uint32_t>(hkv_id);
        cur_q_blk_ = static_cast<uint32_t>(q_blk_id);

        // 计算当前 Q 分块的实际 token 范围
        cur_q_start_ = cur_q_blk_ * m_per_core_;
        cur_q_len_ = (cur_q_start_ + m_per_core_ <= sq_) ? m_per_core_ : (sq_ - cur_q_start_);
        cur_g0_ = static_cast<uint32_t>(g_chunk) * pack_heads_;
        cur_heads_ = (cur_g0_ + pack_heads_ <= gqa_ratio_) ? pack_heads_ : (gqa_ratio_ - cur_g0_);

        // 初始化两级累加控制器与因果掩码活跃分块数 (Causal Mask Early Exit)
        active_kv_blocks_ = ActiveKvBlocks();
    }

    /**
     * S(P) 矩阵 UB 双缓冲槽地址（双缓冲避免延迟一拍流水下的读写竞争）。
     * e4m3 每元素 1 字节，PONG 槽紧随 PING 槽之后，要求 host 侧 ub_softmax_offset
     * 预留 >= 2 * 窗口行数(GQA 聚合行数) * block_n 字节。槽步长按实际写出的行数
     * （gqa_ratio * cur_q_len，上限 MAX_PACK_ROWS）计，保证奇偶槽互不交叠。
     */
    __aicore__ uint32_t SoftmaxSlotOffset(bool is_ping) const
    {
        const uint32_t tile_bytes = TotalRows() * block_n_;
        return tiling_->ub_softmax_offset + (is_ping ? 0u : tile_bytes);
    }

    // GQA 聚合后的窗口逻辑行数与当前块的实际 K 列数
    __aicore__ uint32_t TotalRows() const
    {
        return cur_heads_ * cur_q_len_;
    }

    __aicore__ uint32_t CurBlockK() const
    {
        const int64_t sk_start = static_cast<int64_t>(cur_k_) * block_n_;
        return (sk_start + block_n_ <= sk_) ? block_n_ : static_cast<uint32_t>(sk_ - sk_start);
    }

    // 输入位型归一唯一定义点已迁移 sage_attention2_template_tiling_key.h 的
    // sage_attn::NormDtypeMode；核内消费一律编译期常量 kDtype（M5）。
    // 当前任务的因果活跃 KV 分块数（全尾块公式唯一定义点；DecodeTask 缓存至
    // active_kv_blocks_，Cube/Vec0/Vec1 三处消费同源；块计数取整走官方
    // CATLASS ::CeilDiv，::Min 为同头文件官方极值组件）
    __aicore__ uint32_t ActiveKvBlocks() const
    {
        const uint32_t total_kv_blocks = ::CeilDiv(sk_, block_n_);
        return (tiling_->is_causal != 0)
            ? ::Min(total_kv_blocks, ::CeilDiv(cur_q_start_ + cur_q_len_, block_n_))
            : total_kv_blocks;
    }

    __aicore__ uint32_t CurKvSlotBase() const
    {
        return ((cur_k_ % 2u) == 0u) ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                     : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET;
    }

    __aicore__ inline __gm__ uint8_t *StageStrip() const
    {
        return workspace_gm_ + static_cast<uint64_t>(core_idx_) * STAGE_CORE_STRIDE;
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

        // A7（task/12 D3）：热路径不使用 OutputRetireQueue——golden 轨迹为恒逐块
        // 强锚定重缩放，队列的双锚点策略与 V1 独占 O 主累加器冲突；
        // 每块 PV L0C 部分和经 FixPipe 裸搬至 UB 交接槽，由 V1 按档位合并。

        // S 槽双缓冲进域预置：Cube 是 S 槽消费者，向 V1 发送 S 槽就绪前许可，V1 首两拍写槽前的 wait 立即放行。
        // KV_FREE / DOT_FREE / PARTIAL_FREE 首拍缓冲区天然为空，无需预置（避免跨核伪预置导致首拍死锁）。
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_SOFTMAX_S_PING_FREE);
        CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_SOFTMAX_S_PONG_FREE);

        // 1. 等待 V0 完成 Q 矩阵量化并常驻 L1 Bank 0
        CrossCoreWait(CrossCoreDirection::VEC0_TO_CUBE, CrossCoreFlags::FLAG_Q_READY);

        // 加载 Q 分块至 L0A（GQA 头聚合沿 M 维堆叠）
        qk_mmad.LoadL0A(tiling_->l1_q_offset, TotalRows(), d_);

        const uint32_t active_kv_blocks = ActiveKvBlocks();
        const uint32_t rows = TotalRows();

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
            qk_mmad.LoadL0B(l1_kv_addr, block_n_, d_);
            qk_mmad.Compute(true); // 逐块独立部分和（golden 逐块锚定轨迹，task/12 归约粒度定案）

            // K 块（L1 Bank 1 槽）消费完成：归还缓冲许可，V0 在重写同奇偶槽（k+2）前会 wait。
            CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC0, kv_free);

            // 3. A7：QK dot32 裸搬 UB（首拍 k=0 缓冲区天然为空无需等待；k > 0 时等待 V1 读毕上一拍 dot32；
            //    FixPipe 走 FIX 管，CrossCoreSet 在 S 管——先自配对排空 FIX 再 raise flag 保 GM/UB 可见序）
            if (k > 0) {
                CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_DOT_FREE);
            }
            qk_mmad.EmitL0CToUbRaw(SAGE_UB_DOT32_OFFSET, rows, block_n_);
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
                pv_mmad.LoadL0A(SoftmaxSlotOffset(prev_is_ping), rows, block_n_);
                pv_mmad.LoadL0B(prev_l1_v_addr, d_, block_n_);

                // 每块独立部分和（init=true；两级累加的二级归并由 V1 按 pv_accum_dtype 档位执行）
                pv_mmad.Compute(true);

                // S[prev_k]（UB 槽）消费完成：归还缓冲许可，V1 在重写同奇偶槽前会 wait
                CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, s_free);

                // A7：PV 部分和裸搬 UB 交接槽（首个 partial k=1 槽天然为空；k > 1 时等待 V1 合并完上一块交接槽）
                if (k > 1) {
                    CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_PV_PARTIAL_FREE);
                }
                pv_mmad.ExecuteEpilogue(SAGE_UB_PV_PARTIAL_OFFSET);
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

            pv_mmad.LoadL0A(SoftmaxSlotOffset(last_is_ping), TotalRows(), block_n_);
            // (n,k)=(d, block_n) 同上：PV 收缩维 block_n、输出列 d（DEFECT-M-2）
            pv_mmad.LoadL0B(last_l1_v_addr, d_, block_n_);
            pv_mmad.Compute(true);

            // 归还最后一个 S 槽许可（V1 退域回收依赖此配平）
            CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, s_free);

            // 最后一块部分和搬出（当 active_kv_blocks > 1 时需等待 V1 释放上一块 partial 交接槽）
            if (active_kv_blocks > 1) {
                CrossCoreWait(CrossCoreDirection::VEC1_TO_CUBE, CrossCoreFlags::FLAG_PV_PARTIAL_FREE);
            }
            pv_mmad.ExecuteEpilogue(SAGE_UB_PV_PARTIAL_OFFSET);
            Catlass::SageAttn::Sync<HardEvent::FIX_S>();
            CrossCoreSet(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_PV_PARTIAL_READY);
        }
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
        int64_t hq_first = hkv_idx * static_cast<int64_t>(gqa_ratio_) + static_cast<int64_t>(cur_g0_);

        // GM Q 布局 tla 单点（Catlass::SageAttn::MakeGmBhsdLayout）：基址 = 布局折叠
        // (b, hq_first, sq_start) 坐标，与手写 b·sb + h·sh + s·ss 位元一致（64 位寻址）；
        // 布局对象随行寻址传入 LoadAndQuantizeQ，消除核内第二份步距推导
        const Catlass::SageAttn::GmBhsdLayout q_layout =
            Catlass::SageAttn::MakeGmBhsdLayout(b_, hq_, sq_, d_, tiling_->tensor_layout == LAYOUT_HND);
        const int64_t q_gm_offset = q_layout(tla::MakeCoord(b_idx, hq_first, sq_start, 0u));

        // 1. Prologue：加载当前核负责的 Q，执行 INT8 量化并写入 L1 Bank 0
        LoadAndQuantizeQ(query_gm_ + q_gm_offset * sizeof(uint16_t), d_, q_layout);

        // 通知 Cube：Q 量化常驻已就位
        CrossCoreSet(CrossCoreDirection::VEC0_TO_CUBE, CrossCoreFlags::FLAG_Q_READY);

        const uint32_t active_kv_blocks = ActiveKvBlocks();

        // 2. 流水循环：流式预取第 k 块 KV 并执行 INT8/FP8 量化
        for (uint32_t k = 0; k < active_kv_blocks; ++k) {
            bool is_ping = ((k % 2) == 0);
            uint32_t kv_flag = is_ping ? CrossCoreFlags::FLAG_KV_PING_READY : CrossCoreFlags::FLAG_KV_PONG_READY;
            uint32_t kv_free = is_ping ? CrossCoreFlags::FLAG_KV_PING_FREE : CrossCoreFlags::FLAG_KV_PONG_FREE;
            uint32_t l1_dst_offset = is_ping ? Catlass::SageAttn::L1DoubleBankContract::KV_PING_OFFSET
                                             : Catlass::SageAttn::L1DoubleBankContract::KV_PONG_OFFSET;

            // 缓冲回收握手：同奇偶槽须待 Cube 消费完上一轮（k-2）后方可覆写。
            // 首两槽（k=0/1）直接写入，从 k >= 2 开始等待 Cube 释放该槽。
            if (k >= 2) {
                CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC0, kv_free);
            }


            cur_k_ = k;
            int64_t sk_start = static_cast<int64_t>(k) * block_n_;
            cur_k_start_ = static_cast<uint32_t>(sk_start);
            // 尾块公式唯一定义点 CurBlockK()（与 Cube/V1 消费侧同源，删行内重写式）
            uint32_t cur_block_k = CurBlockK();

            // GM K/V 布局 tla 单点（与 Q 同一 MakeGmBhsdLayout 表达式，坐标不同）；
            // 布局对象随行寻址传入 QuantizeAndStageKey/Value，消除核内第二份步距推导
            const Catlass::SageAttn::GmBhsdLayout kv_layout =
                Catlass::SageAttn::MakeGmBhsdLayout(b_, hkv_, sk_, d_, tiling_->tensor_layout == LAYOUT_HND);
            const int64_t kv_gm_offset = kv_layout(tla::MakeCoord(b_idx, hkv_idx, sk_start, 0u));

            // 获取 K 均值指针 (若启用 smooth_k)。
            // __gm__ 地址空间限定显式保留，统一保持 `const __gm__ float *` 贯穿传递。
            const __gm__ float *km_ptr = nullptr;
            if (tiling_->smooth_k != 0) {
                int64_t km_offset = (b_idx * static_cast<int64_t>(hkv_) + hkv_idx) * d_;
                km_ptr = reinterpret_cast<const __gm__ float*>(key_mean_gm_ + km_offset * sizeof(float));
            }

            // 执行 K 减均值 + INT8 量化 + 摆放，写入 L1 Bank 1
            QuantizeAndStageKey(key_gm_ + kv_gm_offset * sizeof(uint16_t), km_ptr, l1_dst_offset, cur_block_k, d_, kv_layout);

            // 执行 V 转置 + 16-token permute + FP8 量化，写入 L1 Bank 1 槽内 V 子区
            QuantizeAndStageValue(value_gm_ + kv_gm_offset * sizeof(uint16_t),
                                  l1_dst_offset + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET,
                                  cur_block_k, d_, kv_layout);

            // 严格保证生产者在前：写入完成后触发跨核事件通知 Cube
            // 生产管排空（DEFECT-H-2）：MTE3 写 KV 槽 → Cube MTE 读，CrossCoreSet
            // （S 管）不排空 MTE3，先自配对事件等待写毕再举旗（设备态；宿主
            // 环境无硬件跨核事件，CrossCoreSet 模拟为无操作且无 HardEvent，见 sync_guard.h）
#if SAGE_ATTN_DEVICE_COMPILE
            Catlass::SageAttn::Sync<HardEvent::MTE3_MTE2>();
#endif
            CrossCoreSet(CrossCoreDirection::VEC0_TO_CUBE, kv_flag);
        }

    }

    /**
     * 3. Vector 1 核心处理逻辑（Softmax 核）
     * 负责 exp2 域 Online Softmax、逐块 O 重缩放、O 主累加器更新、LSE 累积与写回。
     * softmax 状态为行级（GQA 逻辑行 r = g * cur_q_len + i 各自独立维护 max/denominator，
     * 与 twin RunSequential 的行级状态一致）；S 槽写出布局维持 SoftmaxSlotOffset 奇偶
     * 双缓冲契约不变（V1 写出、Cube 读取）。
     * R17-1（A7 设备消费侧）：设备态补齐 dot32/pv_partial 交接槽的消费握手与数据通路——
     *   每拍 k：拍顶合并第 k-1 块 PV 部分和（wait PARTIAL_READY → MergePvPartialRows →
     *   set PARTIAL_FREE；O 锚点仍为 m_scale(k-1)，早于本拍 Step 重缩放，与 twin 5.4 的
     *   "块内 rescale 后并入"轨迹等价，task/12 第三节时序）→ wait S_FREE → wait QK_P →
     *   dot32 反量化/掩码 → 行 max/exp2 和 → Step → O 行重缩放 → P 码写 S 槽 →
     *   V_S 排空 → set DOT_FREE（dot32 单缓冲读保护归还）→ set S_READY；
     *   收尾：wait PARTIAL_READY → 合并最后一块 → set PARTIAL_FREE → wait PV_O_DONE。
     * R4 行批化（设备态行循环整拍向量化）：行内 FP 轨迹（同操作数同顺序）不变，整窗
     *   行循环收敛为四个 VF pass——DequantMaskLogitsRowsVF（dot32 原地反量化/掩码）→
     *   BlockMaxAndSumRowsVF（行 max / 448· 域 exp2 和，VL1 写行槽）→ 标量 Step 段
     *   （每拍单次 V_S 后逐行 GetValue 读行统计、OnlineSoftmaxTransformer::Step 逐行
     *   不变，行锚 m'·ln2 与 rescale 系数 SetValue 入行槽）→ 单次 S_V 放行后
     *   RescaleAccumRowsVF（行位图许可）+ SoftmaxProbRowsVF（行锚 BRC 广播）。
     *   每拍管排空由"逐行 V_S + 2 次 GetValue"降为"单次 V_S + 单次 S_V"；跨核 flag
     *   序（PARTIAL merge → S_FREE wait → QK_P wait → 行段 → V_S → DOT_FREE → S_READY）
     *   逐 flag 保持不变（R22 对账不变，V_S/S_V 为核内自配对事件对，无跨核余量）。
     */
    __aicore__ void ProcessVec1()
    {
#if !SAGE_ATTN_DEVICE_COMPILE
        if (sim_mem_ != nullptr) {
            for (uint32_t r = 0; r < TotalRows() && r < MAX_PACK_ROWS; ++r) {
                row_states_[r].Reset();
                // O 主累加器按任务复位（真实硬件由下方设备分支的 UB 复位承担）
                std::fill_n(&sim_mem_->o_accum[static_cast<size_t>(r) * d_], d_, 0.0f);
            }
        }
#else
        for (uint32_t r = 0; r < TotalRows() && r < MAX_PACK_ROWS; ++r) {
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

        const uint32_t active_kv_blocks = ActiveKvBlocks();

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
            // R22 前移至行批 VF 段前：设备态 P 码在行批 VF 内写槽，许可须先于整段写获取。
            CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, s_free);

            // 1. 等待 Cube 拍 k 的 P 矩阵块就绪
            CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_QK_P_READY);

#if !SAGE_ATTN_DEVICE_COMPILE
            // 1.5 宿主串行仿真重摆：恢复设备分支"V0 产出→消费"的逐拍序（幂等，逐位一致）。
            RestageKvBlockHost(k);
#else
            // 1.6 A7 scale 条带桥（task/12 第四节）：q 行解析 / v per-channel 任务级一次，
            //     k 块解析按 126 独立 slot 逐拍接入（GM 暂存条带 → UB 副本，MTE2 排空后进 VF/S）
            if (k == 0) {
                StageV1ScaleStrips();
            }
            StageKScaleStrip(k);
#endif

#if SAGE_ATTN_DEVICE_COMPILE
            // 2/3. A7 行批化整拍 VF（R4）：每行 FP 轨迹（同操作数同顺序）不变——
            //      dequant→max→exp-sum→Step→rescale→P-code 保持行内同构，行间独立
            //      故跨行在途交织为轨迹等价（输出契约：e4m3 网格逐行落码不变）。
            //      整窗行循环收敛为四个 VF pass + 一个标量 Step 段：
            //      ① dot32 原地反量化/掩码（[row][block_n] float 覆写 int32 元素等宽，
            //        读保护窗内：QK_P 之后、V_S+DOT_FREE 之前，Cube 下拍覆写被挡）；
            //      ② 行 max / 448· 域 exp2 和 → VL1 写行槽；
            //      ③ 标量 Step 段：单次 V_S 后逐行 GetValue 读统计（twin 5.1 同款整块
            //        掩码跳步进）、Step 行级状态机逐行不变，行锚 m'·ln2 / rescale 系数
            //        SetValue 入行槽（标量域预乘，与原 Duplicate 实参同值）；
            //      ④ 单次 S_V 放行 → O 行位图许可重缩放 → 整幅 P 码写 S 槽（掩码行
            //        经 exp2 下溢自然为 0，与宿主整幅 ComputeSoftmaxProbability 同值）。
            const uint32_t rows = TotalRows();
            LocalTensor<float> ub_qs(TPosition::VECCALC, SAGE_UB_V1_QS_ROW_OFFSET, STAGE_SLOT_FLOATS);
            LocalTensor<float> ub_ks(TPosition::VECCALC, SAGE_UB_V1_KS_ROW_OFFSET, block_n_);
            LocalTensor<float> ub_row_max(TPosition::VECCALC, SAGE_UB_V1_ROW_MAX_OFFSET, MAX_PACK_ROWS);
            LocalTensor<float> ub_row_sum(TPosition::VECCALC, SAGE_UB_V1_ROW_SUM_OFFSET, MAX_PACK_ROWS);
            LocalTensor<float> ub_mln(TPosition::VECCALC, SAGE_UB_V1_M_LN_ROW_OFFSET, MAX_PACK_ROWS);
            LocalTensor<float> ub_factor(TPosition::VECCALC, SAGE_UB_V1_RS_FACTOR_ROW_OFFSET, MAX_PACK_ROWS);
            LocalTensor<float> ub_logits_dot(TPosition::VECCALC, SAGE_UB_DOT32_OFFSET, rows * block_n_);
            LocalTensor<float> ub_o(TPosition::VECCALC,
                                    Catlass::SageAttn::Vector1UbBudget::O_ACCUM_OFFSET, rows * d_);
            LocalTensor<float8_e4m3_t> ub_out_p(TPosition::VECCALC, SoftmaxSlotOffset(is_ping), rows * block_n_);
            dev_vf::DequantMaskLogitsRowsVF(reinterpret_cast<__ubuf__ float*>(ub_logits_dot.GetPhyAddr()), rows,
                                            reinterpret_cast<__ubuf__ float*>(ub_qs.GetPhyAddr()),
                                            reinterpret_cast<__ubuf__ float*>(ub_ks.GetPhyAddr()),
                                            tiling_->sm_scale_log2e, cur_q_start_, cur_q_len_,
                                            cur_k_start_, sk_, tiling_->is_causal, block_n_);
            dev_vf::BlockMaxAndSumRowsVF(reinterpret_cast<__ubuf__ float*>(ub_logits_dot.GetPhyAddr()), rows,
                                         block_n_,
                                         reinterpret_cast<__ubuf__ float*>(ub_row_max.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(ub_row_sum.GetPhyAddr()));
            // 每拍单次 V_S 排空：S 管读回行统计（取代原逐行 V_S + 逐行 2 次 GetValue）
            Catlass::SageAttn::Sync<HardEvent::V_S>();
            // 标量 Step 段（行级状态机与 twin 5.3 逐步一致，不批化）：rescale 位图 +
            // 行锚/系数行槽收集
            // rescale 位图：4×uint32_t 字 + `& 31` 掩码移位（禁 uint64_t 位图——
            // 64 位变量移位触发 BiSheng SelectionDAG 编译崩溃，见 sage_device_vf.h 头注）
            uint32_t bm[4] = {0u, 0u, 0u, 0u};
            for (uint32_t r = 0; r < rows && r < MAX_PACK_ROWS; ++r) {
                const float cur_max = ub_row_max.GetValue(r);
                const float cur_sum = ub_row_sum.GetValue(r);
                if (cur_max > SAGE_MASK_SENTINEL) {
                    const float rescale_factor =
                        OnlineSoftmaxTransformer::Step(row_states_[r], cur_max, cur_sum);
                    if (row_states_[r].need_rescale &&
                        SageFastAbs(rescale_factor - 1.0f) > SCALE_ONE_TOLERANCE) {
                        bm[r >> 5] |= (uint32_t{1} << (r & 31u));
                        ub_factor.SetValue(r, rescale_factor);
                    }
                }
                // 行 P 锚 = Step 后 row_states_[r].m_scale（与原 SoftmaxProbRow 同取）；
                // 标量域预乘 ln2（与原 Duplicate(m_prime*INV_LOG2E) 同一标量乘式同值），
                // VF 侧 DIST_BRC_B32 广播 Load 回读，行内算术链不变
                ub_mln.SetValue(r, row_states_[r].m_scale * INV_LOG2E);
            }
            // S 管 SetValue → V 管读回放行（行锚/系数）；核内自配对事件对，无跨核余量
            Catlass::SageAttn::Sync<HardEvent::S_V>();
            // O 行重缩放：位图许可逐行执行（同行同系数同乘轨迹；未置位行零开销跳过，
            // 与原 need_rescale && |factor-1|>tol 判定同集）
            dev_vf::RescaleAccumRowsVF(reinterpret_cast<__ubuf__ float*>(ub_o.GetPhyAddr()),
                                       reinterpret_cast<__ubuf__ float*>(ub_factor.GetPhyAddr()),
                                       rows, d_, bm[0], bm[1], bm[2], bm[3]);
            // 整幅 P 码写 S 槽（行锚 BRC 广播；全行无条件写出，与宿主整幅
            // ComputeSoftmaxProbability 的行覆盖域一致）
            dev_vf::SoftmaxProbRowsVF(reinterpret_cast<__ubuf__ float*>(ub_logits_dot.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_out_p.GetPhyAddr()),
                                      reinterpret_cast<__ubuf__ float*>(ub_mln.GetPhyAddr()),
                                      rows, block_n_);
#else
            // 2. 逐行块内 logit 反量化/掩码、行 max 与 exp2 和（twin 5.1/5.2 单点），
            //    随后按 twin 5.3 步进行级状态并重缩放 O 行。
            for (uint32_t r = 0; r < TotalRows() && r < MAX_PACK_ROWS; ++r) {
                float cur_max = MIN_FLOAT_VAL;
                float cur_sum = 0.0f;
                ComputeBlockMaxAndSum(r, cur_max, cur_sum, k);
                if (cur_max <= SAGE_MASK_SENTINEL) {
                    continue; // twin 5.1：整块被因果掩码（前缀行），无贡献不步进
                }
                const float rescale_factor =
                    OnlineSoftmaxTransformer::Step(row_states_[r], cur_max, cur_sum);
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
        // 与 twin 5.5 归一化入口的 O 状态逐位同域（时序无死锁：PARTIAL 族无预置/回收，
        // Cube drain 先等本侧合并上一块举出的 FREE 再发射最后一块 READY，本侧收尾
        // wait READY→合并→举 FREE→wait PV_O_DONE，链式无环）
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

        // 退域补等（task/02_code.md B-2 destroy 纪律，唯一预置/回收族）：消费 Cube 进域
        // 预置的最后两个 S 槽许可，保证跨任务波次 FLAG_SOFTMAX_S_*_FREE 余量为零
        //（KV/DOT/PARTIAL 族无预置/回收，不在此列）
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_SOFTMAX_S_PING_FREE);
        CrossCoreWait(CrossCoreDirection::CUBE_TO_VEC1, CrossCoreFlags::FLAG_SOFTMAX_S_PONG_FREE);
    }

#if SAGE_ATTN_DEVICE_COMPILE
    /**
     * A7 scale 条带桥（task/12 第四节；GM 暂存条带常量以 tiling.h 为唯一定义点）：
     * V0 产出 scale → GetUserWorkspace 用户区按核条带（core_idx × STAGE_CORE_STRIDE）
     * → V1 以 MTE2 接入 UB 副本。GM 可见性序：V0 条带写以 S_MTE3/MTE3_S 双自配对排空后
     * 方随 Q_READY/KV_*_READY raise（LoadAndQuantizeQ/QuantizeAndStageKey/Value 设备分支），
     * 本侧一律在 wait QK_P_READY 之后读取（跨核 flag 链传递保序）。
     * 槽位隔离闭环：k 块解析 scale 经 STAGE_K_SCALE_BASE_OFFSET + (k % 126u) * 512 独立槽存放，
     * 彻底消除覆写竞争隐患，无需硬件不支持的跨 Vector 同步。
     * R4 注：qs/ks 消费端已由 S 管 GetValue 改为 V 管 VF LoadAlign / DIST_BRC_B32 广播
     * Load（行批化 DequantMaskLogitsRowsVF），MTE2 写 → V 管读由本函数尾部 MTE2_V 单点
     * 排空承担（原 MTE2_S 对随其 S 管消费端一并删除，事件对不残留）。
     */
    __aicore__ void StageV1ScaleStrips()
    {
        __gm__ uint8_t *strip = StageStrip();
        GlobalTensor<float> gm_strip;
        LocalTensor<float> ub_qs(TPosition::VECCALC, SAGE_UB_V1_QS_ROW_OFFSET, STAGE_SLOT_FLOATS);
        LocalTensor<float> ub_vs(TPosition::VECCALC, SAGE_UB_V1_VS_CHAN_OFFSET, STAGE_SLOT_FLOATS);
        gm_strip.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(strip + STAGE_Q_SCALE_OFFSET));
        DataCopy(ub_qs, gm_strip, STAGE_SLOT_FLOATS);
        gm_strip.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(strip + STAGE_V_SCALE_OFFSET));
        DataCopy(ub_vs, gm_strip, STAGE_SLOT_FLOATS);
        Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
    }

    __aicore__ void StageKScaleStrip(uint32_t k)
    {
        __gm__ uint8_t *strip = StageStrip();
        GlobalTensor<float> gm_strip;
        LocalTensor<float> ub_ks(TPosition::VECCALC, SAGE_UB_V1_KS_ROW_OFFSET, STAGE_SLOT_FLOATS);
        uint32_t k_offset = STAGE_K_SCALE_BASE_OFFSET + (k % 126u) * 512u;
        gm_strip.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(strip + k_offset));
        DataCopy(ub_ks, gm_strip, block_n_);
        Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
    }

    /**
     * A7：pv_partial 交接槽全行合并（R4 行批化：MergePvPartialRowsVF 单次 VF 调用
     * 承载整窗行循环，dev_vf 单点；行内 Load→Mul(→FP16 RNE 舍回)→Add→Store 链与
     * 原逐行 MergePvPartialRowVF 逐位同式）。轨迹 = 宿主 AccumulatePvBlockTwin 的
     * 设备段实例：O 行 += partial·v_scale[c]，fp32+fp16 档按 D2 以 enable_inline_quant
     * 为档位判据在合并处做 FP16 RNE 舍回（QuantizeToFp16 唯一口径）；块内求和序由
     * PV MMAD 承担（设备段粒度，L-A 容差域登记 task/12）。
     */
    __aicore__ void MergePvPartialRows()
    {
        const uint32_t rows = TotalRows();
        // fp32+fp16 档在合并处做 FP16 RNE 舍回（D2 裁定；档位判据直读 tiling）
        const uint32_t fp16_round =
            (tiling_->pv_accum_dtype == PV_ACCUM_FP32_FP16) ? 1u : 0u;
        LocalTensor<float> ub_vs(TPosition::VECCALC, SAGE_UB_V1_VS_CHAN_OFFSET, d_);
        LocalTensor<float> ub_partial(TPosition::VECCALC, SAGE_UB_PV_PARTIAL_OFFSET, rows * d_);
        LocalTensor<float> ub_o(TPosition::VECCALC,
                                Catlass::SageAttn::Vector1UbBudget::O_ACCUM_OFFSET, rows * d_);
        dev_vf::MergePvPartialRowsVF(
            reinterpret_cast<__ubuf__ float*>(ub_partial.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ub_o.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(ub_vs.GetPhyAddr()),
            rows, d_, fp16_round);
    }
#endif

    // -------------------------------------------------------------------------
    // 内部实现辅助函数：
    // - 宿主分支：以 twin_reference.h RunSequential 为数值参考，量化原语/分组索引/
    //   P 量化/PV 累加单点复用 twin 实现；
    // - 设备分支：采用全向量化 AscendC 原语与 dev_vf 专用算子实现 1C2V 高性能流水。
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
     * - smooth_k 的行级 lse_correction 不在核内计算：桥接层按开源契约
     *   （core.py:593-597，fp32 matmul q·km_broadcast^T）算好经 lseCorrection GM
     *   入参传入，V1 FinalizeAndStoreOutput 按 (b, hq, token) 全局下标直读
     *   （V0/V1 跨 AIV 不共享私有状态，核内算 corr 本就无法送达 V1）。
     */
#if !SAGE_ATTN_DEVICE_COMPILE
    /**
     * 输入位型分派：Q/K/V 读取与输出写出按本实例化档位 kDtype（入口按
     * tiling.input_dtype 运行时选择实例）分派，位级转换唯一来源
     * twin::detail::ReadInput/WriteOutput。
     */
    inline float ReadInputBits(const uint16_t *base, int64_t idx) const
    {
        return twin::detail::ReadInput(base, static_cast<size_t>(idx),
                                       kDtype);
    }

    inline void WriteOutputBits(uint16_t *base, int64_t idx, float val) const
    {
        twin::detail::WriteOutput(base, static_cast<size_t>(idx),
                                  kDtype, val);
    }
#endif

    __aicore__ void LoadAndQuantizeQ(__gm__ uint8_t *q_src, uint32_t d,
                                     const Catlass::SageAttn::GmBhsdLayout &q_layout)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        // DEFECT-F-1 重构：组粒度两阶段——行 abs-max VF 进 UB rowmax 数组，
        // 组归并/定档经 quant::QScaleIndex 单点（与宿主串行路径逐步一致），
        // 逐行量化 VF 走除法轨迹 + per_thread half-away / per_warp RINT。
        const bool per_thread_q = (tiling_->qk_quant_gran == QUANT_GRAN_PER_THREAD);
        const uint32_t total_rows_q = TotalRows();
        LocalTensor<float> ub_rowmax(TPosition::VECCALC,
                                     Catlass::SageAttn::Vector0UbBudget::QUANT_ROWMAX_OFFSET,
                                     MAX_PACK_ROWS);
        LocalTensor<float> ub_scales(TPosition::VECCALC,
                                     Catlass::SageAttn::Vector0UbBudget::QUANT_SCALES_OFFSET,
                                     kQScaleSlotStride);
        LocalTensor<half> ub_in(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::GM_INPUT_PING_OFFSET, d);
        LocalTensor<int8_t> ub_out(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::NZ_OUTPUT_OFFSET, d);
        LocalTensor<int8_t> l1_q(TPosition::A1, tiling_->l1_q_offset, total_rows_q * d);
        // M5：输入位型档位为编译期常量 kDtype（VF 调用模板实参，原 dtype_mode 形参删除）

        // 窗口 scale 组清零（防跨任务波次陈旧 amax 混入；P2-8：连续组段以 FillF32VF
        // 广播填充向量化——写集与标量循环逐一重合，槽界钳制保持）
        const uint32_t grp_first_q = quant::QScaleIndex(cur_q_start_, per_thread_q);
        const uint32_t grp_last_q = quant::QScaleIndex(cur_q_start_ + cur_q_len_ - 1u, per_thread_q);
        const uint32_t grp_cnt_q = (grp_first_q < kQScaleSlotStride)
                                       ? (::Min(grp_last_q + 1u, kQScaleSlotStride) - grp_first_q)
                                       : 0u;
        __ubuf__ float *ub_scales_ptr = reinterpret_cast<__ubuf__ float*>(ub_scales.GetPhyAddr());
        for (uint32_t g = 0; g < cur_heads_; ++g) {
            dev_vf::FillF32VF(ub_scales_ptr + (cur_g0_ + g) * kQScaleSlotStride + grp_first_q, 0.0f, grp_cnt_q);
        }

        // A. 行 abs-max（每行一次 MTE2 + VF）
        for (uint32_t r = 0; r < total_rows_q && r < MAX_PACK_ROWS; ++r) {
            // V→MTE2 回边（R22）：ub_in 逐行复用，上一行 VF 读毕（V 管）方可 MTE2 覆写
            Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
            const uint32_t g = r / cur_q_len_;
            const uint32_t i = r % cur_q_len_;
            const int64_t row_gm_offset = q_layout(tla::MakeCoord(0u, g, i, 0u)) * static_cast<int64_t>(sizeof(half));
            GlobalTensor<half> gm_row;
            gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q_src + row_gm_offset));
            // Q 行 GM 流式读取：启用 L2 Cache Bypass (CACHE_MODE_DISABLE)，防止冲刷片上 64MB L2，锁定 K 矩阵长效驻留
            gm_row.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            DataCopy(ub_in, gm_row, d);
            Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
            dev_vf::RowAbsMaxVF<kDtype>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ float*>(ub_rowmax.GetPhyAddr()) + r,
                                        nullptr, d, 0u);
        }
        // V 管写 → S 管读同步（DEFECT-G-1，与 V1 统计读取同款事件对）
        Catlass::SageAttn::Sync<HardEvent::V_S>();

        // B. 组归并 + scale 定档（宿主同式；per_thread=amax/127+1e-7，
        //    per_warp=clamp(amax,1e-7)/127）。
        //    P2-8 向量化裁定（保留标量）：组归并为行→槽散布写——per_thread 组号含
        //    8 相位跨步 8 交织（quant::QScaleIndex 唯一定义），行段非连续且 dav-3510
        //    MicroAPI 无跨 lane 散布/收集原语，散布归并保持标量（2-3 S 拍/行，任务级
        //    一次）；定档循环为 quant::FinalizeQkScale 单点公式，向量化须把公式复制进
        //    VF 违反一处定义，保持标量（≤2·pack_heads 元素/任务）。
        for (uint32_t r = 0; r < total_rows_q && r < MAX_PACK_ROWS; ++r) {
            const uint32_t g = r / cur_q_len_;
            const uint32_t token = cur_q_start_ + (r % cur_q_len_);
            const float amax = ub_rowmax.GetValue(r);
            const uint32_t slot = (cur_g0_ + g) * kQScaleSlotStride + quant::QScaleIndex(token, per_thread_q);
            if (amax > ub_scales.GetValue(slot)) ub_scales.SetValue(slot, amax);
        }
        for (uint32_t g = 0; g < cur_heads_; ++g) {
            for (uint32_t s = grp_first_q; s <= grp_last_q && s < kQScaleSlotStride; ++s) {
                const float amax = ub_scales.GetValue((cur_g0_ + g) * kQScaleSlotStride + s);
                ub_scales.SetValue((cur_g0_ + g) * kQScaleSlotStride + s,
                                   quant::FinalizeQkScale(amax, per_thread_q));
            }
        }

        // C. 逐行量化写 L1（除法轨迹 + 舍入分档）
        for (uint32_t r = 0; r < total_rows_q && r < MAX_PACK_ROWS; ++r) {
            // V→MTE2 回边（R22）：ub_in 逐行复用，上一行 VF 读毕方可 MTE2 覆写
            Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
            const uint32_t g = r / cur_q_len_;
            const uint32_t i = r % cur_q_len_;
            const uint32_t token = cur_q_start_ + i;
            const float scale = ub_scales.GetValue((cur_g0_ + g) * kQScaleSlotStride + quant::QScaleIndex(token, per_thread_q));
            const int64_t row_gm_offset = q_layout(tla::MakeCoord(0u, g, i, 0u)) * static_cast<int64_t>(sizeof(half));
            GlobalTensor<half> gm_row;
            gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q_src + row_gm_offset));
            // Q 行 GM 流式读取：启用 L2 Cache Bypass，避免流式数据冲刷 L2 缓存
            gm_row.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            DataCopy(ub_in, gm_row, d);
            Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
            dev_vf::QuantRowInt8VF<kDtype>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ int8_t*>(ub_out.GetPhyAddr()),
                                           scale, d, per_thread_q ? 1u : 0u, 0u, nullptr);
            Catlass::SageAttn::Sync<HardEvent::V_MTE3>();
            Catlass::SageAttn::CopyUbRowToL1(l1_q[r * d], ub_out, d);
            // MTE3→V 回边（DEFECT-I-1 同款）：ub_out 逐行复用，在途 MTE3 读排空后方可下一行 VF 写
            Catlass::SageAttn::Sync<HardEvent::MTE3_V>();
        }

        // A7 V0（task/12 第五节第 3 条）：行解析 q_scale 暂存写——组定标后的逐行
        // scale 按 GQA 聚合行序展开写 GM 条带（STAGE_Q_SCALE_OFFSET，零填充至 128 槽），
        // V1 反量化按行读取。S→MTE3 与 MTE3→S 双自配对：标量 SetValue 对 MTE3 可见、
        // MTE3 写对后续 S 管 raise FLAG_Q_READY 可见（GM 可见性，task/12 第四节）。
        LocalTensor<float> ub_stage_q(TPosition::VECCALC,
                                      Catlass::SageAttn::Vector0UbBudget::QUANT_STAGE_STRIP_OFFSET,
                                      STAGE_SLOT_FLOATS);
        // 条带零填充向量化（FillF32VF 单点；128 float = 2 全 chunk）
        dev_vf::FillF32VF(reinterpret_cast<__ubuf__ float*>(ub_stage_q.GetPhyAddr()), 0.0f, STAGE_SLOT_FLOATS);
        // 逐行展开保持标量（P2-8）：行→槽为 8 相位交织寻址，无连续段亦无收集原语
        for (uint32_t r = 0; r < total_rows_q && r < MAX_PACK_ROWS; ++r) {
            const uint32_t g = r / cur_q_len_;
            const uint32_t token = cur_q_start_ + (r % cur_q_len_);
            ub_stage_q.SetValue(r, ub_scales.GetValue((cur_g0_ + g) * kQScaleSlotStride + quant::QScaleIndex(token, per_thread_q)));
        }
        Catlass::SageAttn::Sync<HardEvent::S_MTE3>();
        GlobalTensor<float> gm_stage_q;
        gm_stage_q.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(StageStrip() + STAGE_Q_SCALE_OFFSET));
        DataCopy(gm_stage_q, ub_stage_q, STAGE_SLOT_FLOATS);
        Catlass::SageAttn::Sync<HardEvent::MTE3_S>();
#else
        if (sim_mem_ == nullptr || q_src == nullptr) {
            return;
        }
        const bool per_thread = (tiling_->qk_quant_gran == QUANT_GRAN_PER_THREAD);
        const uint16_t *q_u16 = reinterpret_cast<const uint16_t *>(q_src);
        int8_t *l1_q = reinterpret_cast<int8_t *>(sim_mem_->l1_mem + tiling_->l1_q_offset);

        // 窗口覆盖的 scale 组清零（跨任务波次复用 SimMemory，防陈旧 amax 混入）
        const uint32_t grp_first = quant::QScaleIndex(cur_q_start_, per_thread);
        const uint32_t grp_last = quant::QScaleIndex(cur_q_start_ + cur_q_len_ - 1u, per_thread);
        for (uint32_t g = 0; g < cur_heads_; ++g) {
            for (uint32_t s = grp_first; s <= grp_last && s < kQScaleSlotStride; ++s) {
                sim_mem_->q_scales[(cur_g0_ + g) * kQScaleSlotStride + s] = 0.0f;
            }
        }

        // 1. 逐 (GQA 头, scale 组) 的 amax（组边界 = quant::QScaleIndex 唯一定义；
        //    m_per_core 为 64 的倍数保证窗口不切割 32-token 子块）
        for (uint32_t r = 0; r < TotalRows(); ++r) {
            const uint32_t g = r / cur_q_len_;
            if (r >= MAX_PACK_ROWS || g >= cur_heads_) break;
            const uint32_t i = r % cur_q_len_;
            const uint32_t token = cur_q_start_ + i;
            float *scale_slot = &sim_mem_->q_scales[(cur_g0_ + g) * kQScaleSlotStride + quant::QScaleIndex(token, per_thread)];
            const uint16_t *row_ptr = q_u16 + q_layout(tla::MakeCoord(0u, g, i, 0u));
            for (uint32_t c = 0; c < d; ++c) {
                const float abs_v = SageFastAbs(ReadInputBits(row_ptr, c));
                if (abs_v > *scale_slot) *scale_slot = abs_v;
            }
        }

        // 2. scale 定档（quant 层唯一口径）
        for (uint32_t g = 0; g < cur_heads_; ++g) {
            for (uint32_t s = grp_first; s <= grp_last && s < kQScaleSlotStride; ++s) {
                const float amax = sim_mem_->q_scales[(cur_g0_ + g) * kQScaleSlotStride + s];
                sim_mem_->q_scales[(cur_g0_ + g) * kQScaleSlotStride + s] =
                    quant::FinalizeQkScale(amax, per_thread);
            }
        }

        // 3. 量化写 L1（除法轨迹）
        for (uint32_t r = 0; r < TotalRows(); ++r) {
            const uint32_t g = r / cur_q_len_;
            if (r >= MAX_PACK_ROWS || g >= cur_heads_) break;
            const uint32_t i = r % cur_q_len_;
            const uint32_t token = cur_q_start_ + i;
            const float scale = sim_mem_->q_scales[(cur_g0_ + g) * kQScaleSlotStride + quant::QScaleIndex(token, per_thread)];
            const uint16_t *row_ptr = q_u16 + q_layout(tla::MakeCoord(0u, g, i, 0u));
            int8_t *row_out = l1_q + static_cast<size_t>(r) * d;
            for (uint32_t c = 0; c < d; ++c) {
                const float val = ReadInputBits(row_ptr, c);
                row_out[c] = per_thread ? quant::RoundHalfAwayFromZero(val / scale)
                                        : quant::RoundNearestEven(val / scale);
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
        uint32_t d,
        const Catlass::SageAttn::GmBhsdLayout &kv_layout)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        // DEFECT-F-1 重构：组粒度两阶段（组边界 quant::KScaleIndex 单点，
        // 块内本地序号写 l1_k_scale——Data+Scale 相邻契约）
        const bool per_thread_k = (tiling_->qk_quant_gran == QUANT_GRAN_PER_THREAD);
        // M5：输入位型档位为编译期常量 kDtype（VF 调用模板实参，原 dtype_mode 形参删除）
        const uint32_t grp_per_blk = per_thread_k ? 4u : 1u;
        const uint32_t blk_first = cur_k_start_ / quant::K_BLK;
        const uint32_t grp_lo = blk_first * grp_per_blk;
        const uint32_t grp_hi = ((cur_k_start_ + block_n_ - 1u) / quant::K_BLK + 1u) * grp_per_blk;
        LocalTensor<float> ub_rowmax(TPosition::VECCALC,
                                     Catlass::SageAttn::Vector0UbBudget::QUANT_ROWMAX_OFFSET,
                                     MAX_PACK_ROWS);
        LocalTensor<float> ub_k_scales(TPosition::VECCALC,
                                     Catlass::SageAttn::Vector0UbBudget::QUANT_SCALES_OFFSET,
                                     kKScaleSlotStride);
        LocalTensor<float> ub_km(TPosition::VECCALC,
                                 Catlass::SageAttn::Vector0UbBudget::QUANT_KM_OFFSET,
                                 d);
        LocalTensor<half> ub_in(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::GM_INPUT_PING_OFFSET, d);
        LocalTensor<int8_t> ub_out(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::NZ_OUTPUT_OFFSET, d);
        LocalTensor<int8_t> l1_k(TPosition::B1, l1_dst_addr, block_n_ * d);
        LocalTensor<float> l1_k_scale(TPosition::B1, l1_dst_addr + static_cast<uint32_t>(block_n_ * d), kKScaleSlotStride);
        const uint32_t has_km = (km_ptr != nullptr) ? 1u : 0u;

        if (has_km) {
            GlobalTensor<float> gm_km;
            gm_km.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(const_cast<__gm__ float*>(km_ptr)));
            DataCopy(ub_km, gm_km, d);
            Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
        }

        // 窗口 scale 组清零（P2-8：连续组段 FillF32VF 广播填充，槽界钳制保持）
        const uint32_t grp_fill_cnt = (grp_lo < kKScaleSlotStride)
                                          ? (::Min(grp_hi, kKScaleSlotStride) - grp_lo)
                                          : 0u;
        if (grp_fill_cnt != 0u) {
            dev_vf::FillF32VF(reinterpret_cast<__ubuf__ float*>(ub_k_scales.GetPhyAddr()) + grp_lo,
                              0.0f, grp_fill_cnt);
        }

        // A. 行 abs-max（仅真实 token；pad 行零贡献且 pad 组写 0 由 C 段 pad 分支覆盖）
        for (uint32_t j = 0; j < block_k; ++j) {
            // V→MTE2 回边（R22）：ub_in 逐行复用，上一行 VF 读毕（V 管）方可 MTE2 覆写
            Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
            const int64_t row_gm_offset = kv_layout(tla::MakeCoord(0u, 0u, j, 0u)) * static_cast<int64_t>(sizeof(half));
            GlobalTensor<half> gm_row;
            gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k_src + row_gm_offset));
            // K 矩阵 GM 读：不旁路 L2 Cache，使其在 Atlas 950PR 片上 64MB L2 中长效驻留，供后续 Q 块高效命中复用
            DataCopy(ub_in, gm_row, d);
            Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
            dev_vf::RowAbsMaxVF<kDtype>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                        reinterpret_cast<__ubuf__ float*>(ub_rowmax.GetPhyAddr()) + j,
                                        reinterpret_cast<__ubuf__ float*>(ub_km.GetPhyAddr()),
                                        d, has_km);
        }
        // V 管写 → S 管读同步（DEFECT-G-1）
        Catlass::SageAttn::Sync<HardEvent::V_S>();

        // B. 组归并（pair/块粒度，KScaleIndex 单点）+ 定档 + l1_k_scale 本地写出
        //（P2-8：归并/定档保留标量，理由同 LoadAndQuantizeQ B 段——交织组散布写 +
        // quant 单点公式；amax 单次读出替代原条件真分支的二次 GetValue）
        for (uint32_t j = 0; j < block_k; ++j) {
            const uint32_t slot = quant::KScaleIndex(cur_k_start_ + j, per_thread_k);
            const float amax = ub_rowmax.GetValue(j);
            if (amax > ub_k_scales.GetValue(slot)) ub_k_scales.SetValue(slot, amax);
        }
        for (uint32_t s = grp_lo; s < grp_hi && s < kKScaleSlotStride; ++s) {
            const float amax = ub_k_scales.GetValue(s);
            const float scale = quant::FinalizeQkScale(amax, per_thread_k);
            ub_k_scales.SetValue(s, scale);
            l1_k_scale.SetValue(s - grp_lo, scale);
        }

        // A7 V0（task/12 第五节第 3 条）：块解析 k_scale 暂存写——逐 token 展开写
        // GM 条带独立 k 槽（STAGE_K_SCALE_BASE_OFFSET + (cur_k_ % 126u) * 512，126 槽
        // 轮转隔离防覆写，槽内零填充至 128 float=512B），V1 反量化按同式 k % 126 槽号
        // 读取（StageKScaleStrip 同源）。S→MTE3 / MTE3→S 双自配对排空后方由调用方
        // raise KV_*_READY（GM 可见性，task/12 第四节；QK_P 传递性保证 V1 读取时序）。
        LocalTensor<float> ub_stage_k(TPosition::VECCALC,
                                      Catlass::SageAttn::Vector0UbBudget::QUANT_STAGE_STRIP_OFFSET,
                                      STAGE_SLOT_FLOATS);
        // 条带零填充向量化（FillF32VF 单点；128 float = 2 全 chunk）
        dev_vf::FillF32VF(reinterpret_cast<__ubuf__ float*>(ub_stage_k.GetPhyAddr()), 0.0f, STAGE_SLOT_FLOATS);
        // 逐 token 展开保持标量（P2-8：交织组寻址 + 尾块 pad 列零保持，同 Q 条带理由）
        for (uint32_t j = 0; j < block_n_; ++j) {
            if (j < block_k) {
                const uint32_t token = cur_k_start_ + j;
                ub_stage_k.SetValue(j, ub_k_scales.GetValue(quant::KScaleIndex(token, per_thread_k)));
            } // 尾块 pad 列保持 0：V1 侧前缀掩码哨兵不消费该列 scale
        }
        Catlass::SageAttn::Sync<HardEvent::S_MTE3>();
        GlobalTensor<float> gm_stage_k;
        uint32_t k_offset = STAGE_K_SCALE_BASE_OFFSET + (cur_k_ % 126u) * 512u;
        gm_stage_k.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(StageStrip() + k_offset));
        DataCopy(gm_stage_k, ub_stage_k, STAGE_SLOT_FLOATS);
        Catlass::SageAttn::Sync<HardEvent::MTE3_S>();

        // C. 逐行量化写 L1 + 尾块 pad 置零
        for (uint32_t j = 0; j < block_n_; ++j) {
            // V→MTE2 回边（R22）：ub_in 逐行复用，上一行 VF 读毕方可 MTE2 覆写
            Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
            if (j < block_k) {
                const int64_t row_gm_offset = kv_layout(tla::MakeCoord(0u, 0u, j, 0u)) * static_cast<int64_t>(sizeof(half));
                GlobalTensor<half> gm_row;
                gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k_src + row_gm_offset));
                // K 矩阵 GM 读：保持 L2 Cache 驻留
                DataCopy(ub_in, gm_row, d);
                Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
                const uint32_t token = cur_k_start_ + j;
                const float scale = ub_k_scales.GetValue(quant::KScaleIndex(token, per_thread_k));
                dev_vf::QuantRowInt8VF<kDtype>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ int8_t*>(ub_out.GetPhyAddr()),
                                               scale, d, per_thread_k ? 1u : 0u, has_km,
                                               reinterpret_cast<__ubuf__ float*>(ub_km.GetPhyAddr()));
            } else {
                Duplicate(ub_out, int8_t(0), d);
            }
            Catlass::SageAttn::Sync<HardEvent::V_MTE3>();
            Catlass::SageAttn::CopyUbRowToL1(l1_k[j * d], ub_out, d);
            // MTE3→V 回边（DEFECT-I-1 同款）：ub_out 逐行复用，在途 MTE3 读排空后方可下一行 VF/Duplicate 写
            Catlass::SageAttn::Sync<HardEvent::MTE3_V>();
        }
#else
        if (sim_mem_ == nullptr || k_src == nullptr) {
            return;
        }
        const bool per_thread = (tiling_->qk_quant_gran == QUANT_GRAN_PER_THREAD);
        const uint16_t *k_u16 = reinterpret_cast<const uint16_t *>(k_src);
        int8_t *l1_k = reinterpret_cast<int8_t *>(sim_mem_->l1_mem + l1_dst_addr);
        float *l1_k_scale = reinterpret_cast<float *>(
            sim_mem_->l1_mem + l1_dst_addr + static_cast<size_t>(block_n_) * d);

        // 本核块覆盖的 scale 组清零（防跨块/跨任务陈旧 amax 混入）。
        // block_n 可为 128（host 侧 sk>=2048 档），一个核块横跨两个 64-token K 量化块，
        // 组区间须按 ceil(block_n/K_BLK) 个量化块覆盖。
        const uint32_t grp_per_blk = per_thread ? 4u : 1u;
        const uint32_t blk_first = cur_k_start_ / quant::K_BLK;
        // blk_last 为尾元素所在量化块号（floor 语义），非 ceildiv——禁改官方对齐组件
        const uint32_t blk_last = (cur_k_start_ + block_n_ - 1u) / quant::K_BLK;
        const uint32_t grp_lo = blk_first * grp_per_blk;
        const uint32_t grp_hi = (blk_last + 1u) * grp_per_blk;
        for (uint32_t s = grp_lo; s < grp_hi && s < kKScaleSlotStride; ++s) {
            sim_mem_->k_scales[s] = 0.0f;
        }

        // 1. 逐组 amax（kMean 减法带 fp16 舍入）
        for (uint32_t j = 0; j < block_k; ++j) {
            const uint32_t token = cur_k_start_ + j;
            float *scale_slot = &sim_mem_->k_scales[quant::KScaleIndex(token, per_thread)];
            const uint16_t *row_ptr = k_u16 + kv_layout(tla::MakeCoord(0u, 0u, j, 0u));
            for (uint32_t c = 0; c < d; ++c) {
                float val = ReadInputBits(row_ptr, c);
                val = quant::SubKm(val, km_ptr, c, kDtype);
                const float abs_v = SageFastAbs(val);
                if (abs_v > *scale_slot) *scale_slot = abs_v;
            }
        }

        // 2. scale 定档（l1_k_scale 写槽内本地序号：Data+Scale 相邻布局按块内组序
        // 排布，防全局组号越界）
        for (uint32_t s = grp_lo; s < grp_hi && s < kKScaleSlotStride; ++s) {
            const float amax = sim_mem_->k_scales[s];
            const float scale = quant::FinalizeQkScale(amax, per_thread);
            sim_mem_->k_scales[s] = scale;
            l1_k_scale[s - grp_lo] = scale;
        }

        // 3. 量化写 L1 + 尾块 pad 置零
        for (uint32_t j = 0; j < block_n_; ++j) {
            int8_t *row_out = l1_k + static_cast<size_t>(j) * d;
            if (j < block_k) {
                const uint32_t token = cur_k_start_ + j;
                const float scale = sim_mem_->k_scales[quant::KScaleIndex(token, per_thread)];
                const uint16_t *row_ptr = k_u16 + kv_layout(tla::MakeCoord(0u, 0u, j, 0u));
                for (uint32_t c = 0; c < d; ++c) {
                    float val = ReadInputBits(row_ptr, c);
                    val = quant::SubKm(val, km_ptr, c, kDtype);
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
     * - per-channel scale 为全序列静态量：首块（cur_k_==0）对整个 KV 头做 GM 预扫
     *   （pad token 计 0 不影响 amax），scale = amax / v_scale_max（fp32+fp16 档 2.25、
     *   其余 448，由 tiling 携带）；smooth_v（host 门控仅 fp32 档置位）时预扫累计
     *   max/min/sum 三统计量，vm = sum/fp8_padded、amax = max(|max-vm|,|min-vm|,
     *   pad |0-vm|)（dispatch._quant_v_fp8 / quant_fp8 ProcessV 同式）；全零通道
     *   scale=0 经 NaN → e4m3 0x7f 容错（同 quant_fp8）；
     * - 量化保持除法轨迹：scaled = (val - vm) / scale，饱和 ±scale_max 后 RNE 转 e4m3；
     * - vm 经 workspace GM 区（tiling.workspace_vm_offset，host 按 [b,hkv,d_pad] fp32
     *   预留）传 V1，Finalize 归一化后逐通道加回；v_scale 仍走 STAGE_V_SCALE 条带；
     * - L1 槽内 V 布局 [d][block_n]，16-token 段内物理槽 p 存 token V_PERM_PATTERN[p]
     *   （token jl 的物理槽 = V_PERM_INV_PATTERN[jl%16]，与 golden v_fp8 同构、与
     *   twin v_deq 读取式一致）；尾块 pad 槽写 0。
     */
    __aicore__ void QuantizeAndStageValue(
        __gm__ uint8_t *v_src,
        uint32_t l1_dst_addr,
        uint32_t block_k,
        uint32_t d,
        const Catlass::SageAttn::GmBhsdLayout &kv_layout)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        // DEFECT-F-1 重构：per-channel 全序列预扫（首块一次）→ UB scale 张量；逐行量化
        // VF 走除法轨迹 + ±scale_max 截断 + 全零通道 0/0=NaN → e4m3 0x7f。
        // M5：输入位型档位为编译期常量 kDtype（VF 调用模板实参，原 dtype_mode 形参删除）
        LocalTensor<float> ub_v_scale(TPosition::VECCALC,
                                      Catlass::SageAttn::Vector0UbBudget::QUANT_ROWMAX_OFFSET,
                                      d);
        LocalTensor<float> ub_v_min(TPosition::VECCALC,
                                    Catlass::SageAttn::Vector0UbBudget::QUANT_VMIN_OFFSET,
                                    d);
        LocalTensor<float> ub_v_sum(TPosition::VECCALC,
                                    Catlass::SageAttn::Vector0UbBudget::QUANT_VSUM_OFFSET,
                                    d);
        LocalTensor<half> ub_in(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::GM_INPUT_PING_OFFSET, d);
        LocalTensor<float8_e4m3_t> ub_out(TPosition::VECCALC, Catlass::SageAttn::Vector0UbBudget::NZ_OUTPUT_OFFSET, d);
        LocalTensor<float8_e4m3_t> l1_v(TPosition::B1, l1_dst_addr, block_n_ * d);
        const float scale_max = tiling_->v_scale_max;
        const uint32_t smooth_v = (tiling_->smooth_v != 0) ? 1u : 0u;

        if (cur_k_ == 0) {
            // fp8_padded：序列维 16-token 段对齐（官方 ::RoundUp，与手写位元一致）
            const uint32_t fp8_padded = ::RoundUp<16u>(sk_);
            if (smooth_v != 0u) {
                // smooth_v：单趟 max/min/sum 统计预扫 → vm/scale 定标（FinalizeSmoothVScaleVF
                // 单点；vm 留 UB 待 SM 级搬出），统计量经 V→S 排空后由 VF 消费
                Duplicate(ub_v_scale, MIN_FLOAT_VAL, d);
                Duplicate(ub_v_min, -MIN_FLOAT_VAL, d);
                Duplicate(ub_v_sum, 0.0f, d);
                for (uint32_t t = 0; t < sk_; ++t) {
                    const int64_t row_gm_offset = kv_layout(tla::MakeCoord(0u, 0u, t, 0u)) * static_cast<int64_t>(sizeof(half));
                    GlobalTensor<half> gm_row;
                    gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v_src + row_gm_offset));
                    // V 全序列预扫（任务级一次的流式读）：启用 L2 直通旁路，避免冲刷片上 64MB L2
                    gm_row.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                    Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
                    DataCopy(ub_in, gm_row, d);
                    Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
                    dev_vf::ChannelStatsAccumVF<kDtype>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                                        reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                                        reinterpret_cast<__ubuf__ float*>(ub_v_min.GetPhyAddr()),
                                                        reinterpret_cast<__ubuf__ float*>(ub_v_sum.GetPhyAddr()),
                                                        d);
                }
                Catlass::SageAttn::Sync<HardEvent::V_S>();
                dev_vf::FinalizeSmoothVScaleVF(reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float*>(ub_v_min.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float*>(ub_v_sum.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                               fp8_padded, (fp8_padded > sk_) ? 1u : 0u, scale_max, d);
                Catlass::SageAttn::Sync<HardEvent::V_S>();
            } else {
                // A7 V0（task/12 第五节第 3 条）：per-channel 全序列 amax 预扫向量化——
                // ub_v_scale 充零作 amax 累加器，逐 token 行 ChannelAmaxAccumVF 累加
                //（v_src 即 KV 头首行基址；pad token 计 0 不影响），FinalizeVScaleVF 按
                // v_scale_max 除法定标（quant_fp8/孪生唯一口径）。
                Duplicate(ub_v_scale, 0.0f, d);
                for (uint32_t t = 0; t < sk_; ++t) {
                    const int64_t row_gm_offset = kv_layout(tla::MakeCoord(0u, 0u, t, 0u)) * static_cast<int64_t>(sizeof(half));
                    GlobalTensor<half> gm_row;
                    gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v_src + row_gm_offset));
                    // V 全序列预扫（任务级一次的流式读）：启用 L2 直通旁路，避免冲刷片上 64MB L2
                    gm_row.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                    Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
                    DataCopy(ub_in, gm_row, d);
                    Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
                    dev_vf::ChannelAmaxAccumVF<kDtype>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                                        reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                                        d);
                }
                dev_vf::FinalizeVScaleVF(reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                         scale_max, d);
            }

            // A7：v_scale 暂存写（STAGE_V_SCALE_OFFSET，任务级一次）+ smooth_v 的 vm
            // 写 workspace GM 区（workspace_vm_offset，[b,hkv,d_pad] fp32）。V→MTE3 排空
            // 保证向量写毕才发起 GM 搬出，MTE3→S 排空后方由调用方在首块 KV_READY raise
            // 前 GM 可见（V1 Finalize 侧经 PV_O_DONE 前的跨核 flag 链传递读取时序）。
            Catlass::SageAttn::Sync<HardEvent::V_MTE3>();
            GlobalTensor<float> gm_stage_v;
            gm_stage_v.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(StageStrip() + STAGE_V_SCALE_OFFSET));
            DataCopy(gm_stage_v, ub_v_scale, d);
            if (smooth_v != 0u) {
                GlobalTensor<float> gm_vm;
                gm_vm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                    workspace_gm_ + tiling_->workspace_vm_offset +
                    static_cast<int64_t>(cur_b_) * hkv_ * d_ * sizeof(float) +
                    static_cast<int64_t>(cur_hkv_) * d_ * sizeof(float)));
                DataCopy(gm_vm, ub_v_sum, d);
            }
            Catlass::SageAttn::Sync<HardEvent::MTE3_S>();
        }

        for (uint32_t j = 0; j < block_n_; ++j) {
            // V→MTE2 回边（R22）：ub_in 逐行复用，上一行 VF 读毕（V 管）方可 MTE2 覆写
            Catlass::SageAttn::Sync<HardEvent::V_MTE2>();
            if (j < block_k) {
                const int64_t row_gm_offset = kv_layout(tla::MakeCoord(0u, 0u, j, 0u)) * static_cast<int64_t>(sizeof(half));
                GlobalTensor<half> gm_row;
                gm_row.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v_src + row_gm_offset));
                // V 行 GM 流式读（单任务一次性消费）：L2 直通旁路，避免污染 Die 本地 L2
                gm_row.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                DataCopy(ub_in, gm_row, d);
                Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
                dev_vf::QuantVRowFp8VF<kDtype>(reinterpret_cast<__ubuf__ half*>(ub_in.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_out.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float*>(ub_v_scale.GetPhyAddr()),
                                               reinterpret_cast<__ubuf__ float*>(ub_v_sum.GetPhyAddr()),
                                               smooth_v, d, scale_max);
            } else {
                Duplicate(ub_out.ReinterpretCast<int8_t>(), int8_t(0), d);
            }
            Catlass::SageAttn::Sync<HardEvent::V_MTE3>();
            Catlass::SageAttn::CopyUbRowToL1(l1_v[j * d], ub_out, d);
            // MTE3→V 回边（DEFECT-I-1 同款）：ub_out 逐行复用，在途 MTE3 读排空后方可下一行 VF/Duplicate 写
            Catlass::SageAttn::Sync<HardEvent::MTE3_V>();
        }
#else
        if (sim_mem_ == nullptr || v_src == nullptr) {
            return;
        }
        const uint16_t *v_u16 = reinterpret_cast<const uint16_t *>(v_src);
        uint8_t *l1_v = sim_mem_->l1_mem + l1_dst_addr;
        const float scale_max = tiling_->v_scale_max;

        if (cur_k_ == 0) {
            const uint16_t *v_head = v_u16 - kv_layout(tla::MakeCoord(0u, 0u, cur_k_start_, 0u));
            // fp8_padded：序列维 16-token 段对齐（官方 ::RoundUp，与设备分支同式）
            const uint32_t fp8_padded = ::RoundUp<16u>(sk_);
            const bool smooth_v = (tiling_->smooth_v != 0);
            for (uint32_t c = 0; c < d && c < 256u; ++c) {
                if (smooth_v) {
                    // quant_fp8 ProcessV smooth_v 支路同式：vm = sum/fp8_padded，
                    // amax = max(pad |0-vm|, max_t |v_t - vm|)（pad token 计 0 贡献 |vm|），
                    // scale = amax / scale_max（全零通道 scale=0 → NaN 容错同非 smooth 路径）
                    float sum_v = 0.0f;
                    for (uint32_t t = 0; t < sk_; ++t) {
                        sum_v += ReadInputBits(v_head, kv_layout(tla::MakeCoord(0u, 0u, t, 0u)) + c);
                    }
                    const float vm = sum_v / static_cast<float>(fp8_padded);
                    float amax = (fp8_padded > sk_) ? SageFastAbs(vm) : 0.0f;
                    for (uint32_t t = 0; t < sk_; ++t) {
                        const float diff =
                            ReadInputBits(v_head, kv_layout(tla::MakeCoord(0u, 0u, t, 0u)) + c) - vm;
                        const float abs_v = SageFastAbs(diff);
                        if (abs_v > amax) amax = abs_v;
                    }
                    sim_mem_->v_means[c] = vm;
                    sim_mem_->v_scales[c] = amax / scale_max;
                } else {
                    float amax = 0.0f;
                    for (uint32_t t = 0; t < sk_; ++t) {
                        const float abs_v = SageFastAbs(
                            ReadInputBits(v_head, kv_layout(tla::MakeCoord(0u, 0u, t, 0u)) + c));
                        if (abs_v > amax) amax = abs_v;
                    }
                    sim_mem_->v_scales[c] = amax / scale_max;
                }
            }
        }

        const bool smooth_v = (tiling_->smooth_v != 0);
        for (uint32_t c = 0; c < d; ++c) {
            const float scale = (c < 256u) ? sim_mem_->v_scales[c] : 1.0f;
            const float vm = (smooth_v && c < 256u) ? sim_mem_->v_means[c] : 0.0f;
            uint8_t *chan_out = l1_v + static_cast<size_t>(c) * block_n_;
            for (uint32_t jl = 0; jl < block_n_; ++jl) {
                uint8_t code = 0;
                if (jl < block_k) {
                    const uint16_t *row_ptr = v_u16 + kv_layout(tla::MakeCoord(0u, 0u, jl, 0u));
                    float scaled = (ReadInputBits(row_ptr, c) - vm) / scale;
                    if (scaled > scale_max) scaled = scale_max;
                    if (scaled < -scale_max) scaled = -scale_max;
                    code = quant::FloatToE4m3fn(scaled);
                }
                chan_out[quant::VSlotOfToken(jl)] = code;
            }
        }
#endif
    }

#if !SAGE_ATTN_DEVICE_COMPILE
    /**
     * 4. V1：单逻辑行的块内反量化 logit、行 max 与 exp2 和（twin 5.1/5.2 单点轨迹）。
     * R4 注：本函数收敛为宿主/KT 标量基线专属（twin 真值路径，设备分支不再保留第二份
     * 数值实现）——设备态由 ProcessVec1 行批 VF 段（DequantMaskLogitsRowsVF +
     * BlockMaxAndSumRowsVF + 标量 Step 段）承担，逐位同构见对应注释：
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
            sim_mem_->q_scales[(cur_g0_ + g) * kQScaleSlotStride + quant::QScaleIndex(token_q, per_thread)];
        float *logits_row = &sim_mem_->logits[static_cast<size_t>(row) * block_n_];
        const int8_t *l1_q = reinterpret_cast<const int8_t *>(
            sim_mem_->l1_mem + tiling_->l1_q_offset);
        const int8_t *l1_k = reinterpret_cast<const int8_t *>(
            sim_mem_->l1_mem + CurKvSlotBase());

        for (uint32_t j = 0; j < block_n_; ++j) {
            const uint32_t token_k = cur_k_start_ + j;
            if (token_k >= sk_ || (tiling_->is_causal != 0 &&
                                   static_cast<int64_t>(token_k) > static_cast<int64_t>(token_q))) {
                logits_row[j] = SAGE_MASK_SENTINEL;
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

        float row_max = SAGE_MASK_SENTINEL;
        for (uint32_t j = 0; j < block_n_; ++j) {
            if (logits_row[j] > row_max) row_max = logits_row[j];
        }
        if (row_max <= SAGE_MASK_SENTINEL) {
            cur_max = SAGE_MASK_SENTINEL;
            cur_sum = 0.0f;
            return;
        }
        cur_max = row_max;
        cur_sum = twin::OnlineBlockExp2Sum(logits_row, cur_k_start_,
                                           cur_k_start_, cur_k_start_ + block_n_,
                                           tiling_->is_causal != 0, token_q, row_max);
    }

    /**
     * 5. V1：O 主累加器单行原地重缩放（宿主/KT 标量基线专属；设备态同行重缩放收敛至
     * RescaleAccumRowsVF 行位图许可 pass）。twin 5.3 口径为纯 fp32 乘——即使 fp32+fp16 档
     * 也不经 FP16 中间舍入（FP16 域累加仅发生在 PV 归约路径，见
     * twin::AccumulatePvRow），与 RunSequential 逐位一致。
     */
    __aicore__ void RescaleAccumulator(float rescale_factor, uint32_t row)
    {
        if (sim_mem_ == nullptr) {
            return;
        }
        float *o_row = &sim_mem_->o_accum[static_cast<size_t>(row) * d_];
        for (uint32_t c = 0; c < d_; ++c) {
            o_row[c] *= rescale_factor;
        }
    }
#endif

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
        for (uint32_t r = 0; r < rows && r < MAX_PACK_ROWS; ++r) {
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

        // GM K/V 布局 tla 单点（与 ProcessVec0 流水路径同一 MakeGmBhsdLayout 表达式）
        const Catlass::SageAttn::GmBhsdLayout kv_layout =
            Catlass::SageAttn::MakeGmBhsdLayout(b_, hkv_, sk_, d_, tiling_->tensor_layout == LAYOUT_HND);
        const int64_t kv_gm_offset = kv_layout(tla::MakeCoord(b_idx, hkv_idx, sk_start, 0u));
        const __gm__ float *km_ptr = nullptr;
        if (tiling_->smooth_k != 0) {
            const int64_t km_offset = (b_idx * static_cast<int64_t>(hkv_) + hkv_idx) * d_;
            km_ptr = reinterpret_cast<const __gm__ float *>(key_mean_gm_ + km_offset * sizeof(float));
        }

        QuantizeAndStageKey(key_gm_ + kv_gm_offset * sizeof(uint16_t), km_ptr, l1_dst_offset,
                            cur_block_k, d_, kv_layout);
        QuantizeAndStageValue(value_gm_ + kv_gm_offset * sizeof(uint16_t),
                              l1_dst_offset + Catlass::SageAttn::L1DoubleBankContract::KV_V_SUB_OFFSET,
                              cur_block_k, d_, kv_layout);
    }

    /**
     * 7a. V1（KT 标量基线专属）：P 槽写出后的 PV 累加，经 twin::AccumulatePvRow
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
        for (uint32_t r = 0; r < rows && r < MAX_PACK_ROWS; ++r) {
            const uint32_t token_q = cur_q_start_ + (r % cur_q_len_);
            const float m_prime = row_states_[r].m_scale;
            twin::AccumulatePvRow(
                &sim_mem_->o_accum[static_cast<size_t>(r) * d_], d_,
                &sim_mem_->logits[static_cast<size_t>(r) * block_n_], cur_k_start_,
                cur_k_start_, cur_k_start_ + block_k,
                tiling_->is_causal != 0, token_q, m_prime,
                fp16_accum ? PV_ACCUM_FP32_FP16 : PV_ACCUM_FP32, k_step_,
                [&](uint32_t j, uint32_t c) -> float {
                    const uint32_t jl = j - cur_k_start_;
                    const uint32_t slot = quant::VSlotOfToken(jl);
                    // smooth_v：v_deq = e4m3 解码 · 通道 scale + vm（dispatch._quant_v_fp8
                    // smooth_v 支路同式；vm 项随 PV 累加进主累加器，归一化后恰还原 +vm）
                    const float vm = (tiling_->smooth_v != 0) ? sim_mem_->v_means[c] : 0.0f;
                    return quant::E4m3fnToFloat(l1_v[static_cast<size_t>(c) * block_n_ + slot]) *
                               sim_mem_->v_scales[c] + vm;
                });
        }
    }
#endif

    /**
     * 7. V1：最终归一化与 LSE 换算写回 GM（twin 5.5/5.6 单点轨迹）：
     * - 行级 inv_d = GetFinalInvDenominator(row_states_[r])（O 与分母同处 448· 放大域，
     *   除法处放大精确相消）；
     * - smooth_k 的行级 lse_correction 由桥接层算好（fp32 matmul q·km_broadcast^T，
     *   开源契约 core.py:593-597）经 lseCorrection GM 入参传入，对齐域逐头整行
     *   MTE2 接入 UB（未对齐域逐 token 直读）——校正项启用与否以该张量存在
     *   （非 nullptr）为唯一定义，V0/V1 跨 AIV 不共享私有状态，核内不得自算；
     * - smooth_v（host 门控仅 fp32 档）：V 减均值量化下 PV 分子分母的 vm·Σp 项
     *   归一化后恰还原为 vm，Finalize 经 workspace GM 区（workspace_vm_offset）
     *   读回 vm 后逐通道加回（o·inv + vm）；
     * - LSE 出口 = ConvertLseToHost(lse, sm_scale, lse_correction, has_corr,
     *   S_FP8_OFFSET)：核内 exp2 域 LSE 含 448· 放大（乘法实现）所致的
     *   +log2(448) 偏置，出口以 S_FP8_OFFSET 常量补扣。
     */
    __aicore__ void FinalizeAndStoreOutput()
    {
        int64_t b_idx = static_cast<int64_t>(cur_b_);
        int64_t hkv_idx = static_cast<int64_t>(cur_hkv_);
        int64_t sq_start = static_cast<int64_t>(cur_q_start_);

        // GM O 布局 tla 单点（与输入同一 MakeGmBhsdLayout；D 连续维由 DataCopy 承担）
        const Catlass::SageAttn::GmBhsdLayout out_layout =
            Catlass::SageAttn::MakeGmBhsdLayout(b_, hq_, sq_, d_, tiling_->tensor_layout == LAYOUT_HND);
        // LSE / lseCorrection 的 (b,hq,sq) 稠密布局 tla 单点（与 tensor_layout 无关）
        const Catlass::SageAttn::GmBhsLayout lse_layout =
            Catlass::SageAttn::MakeGmBhsLayout(b_, hq_, sq_);

        // smooth_k 校正项启用唯一定义：lseCorrection GM 张量存在（桥接层仅在
        // smooth_k && return_lse 时产出该张量）
        const bool has_lse_corr = (lse_correction_gm_ != nullptr);
#if SAGE_ATTN_DEVICE_COMPILE
        // smooth_v 的 vm 通道向量：V0 首块产出并写 workspace GM 区，本侧 Finalize 前
        // 经 MTE2 接入 UB（读取时序在 PV_O_DONE 之后，由跨核 flag 链传递保序——
        // vm 写毕方 raise 首块 KV_READY，QK_P 链直达本函数）
        const bool has_vm = (tiling_->smooth_v != 0);
        LocalTensor<float> ub_vm(TPosition::VECCALC, SAGE_UB_V1_VM_CHAN_OFFSET, d_);
        if (has_vm) {
            GlobalTensor<float> gm_vm;
            gm_vm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                workspace_gm_ + tiling_->workspace_vm_offset +
                (b_idx * static_cast<int64_t>(hkv_) + hkv_idx) * d_ * sizeof(float)));
            DataCopy(ub_vm, gm_vm, d_);
            Catlass::SageAttn::Sync<HardEvent::MTE2_V>();
        }
        LocalTensor<half> ub_half(TPosition::VECCALC, Catlass::SageAttn::Vector1UbBudget::FIXPIPE_RECV_OFFSET, d_);

        for (uint32_t g = 0; g < cur_heads_; ++g) {
            const int64_t hq_curr = hkv_idx * static_cast<int64_t>(gqa_ratio_) + static_cast<int64_t>(cur_g0_) + g;
            if (hq_curr >= hq_) break;

            // P2-8 LSE 条带突发化：corr 行单次 MTE2 接入 UB、lse 行单次 MTE3 搬出——
            // 取代逐 token 标量 GM 读/写（S 管逐元素 GM 访问串行化）。突发约束为
            // DataCopy 块搬运的 32B 对齐域：GM 元素基址与长度均须 8（fp32）对齐；
            // 不满足（sq 尾块/头基址未对齐）时逐 token 回落标量路径，位级同值同址。
            const int64_t lse_elem_base = lse_layout(tla::MakeCoord(b_idx, hq_curr, sq_start));
            const bool lse_burst_ok =
                (static_cast<uint64_t>(lse_elem_base) % 8u == 0u) && ((cur_q_len_ % 8u) == 0u);
            LocalTensor<float> ub_corr(TPosition::VECCALC, SAGE_UB_V1_CORR_ROW_OFFSET, cur_q_len_);
            LocalTensor<float> ub_lse_row(TPosition::VECCALC, SAGE_UB_V1_LSE_ROW_OFFSET, cur_q_len_);
            if (has_lse_corr && lse_burst_ok) {
                GlobalTensor<float> gm_corr;
                gm_corr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                    lse_correction_gm_ + lse_elem_base * static_cast<int64_t>(sizeof(float))));
                DataCopy(ub_corr, gm_corr, cur_q_len_);
                // 条带副本由 S 管逐行 GetValue 消费：MTE2 写 → S 管读显式排空
                Catlass::SageAttn::Sync<HardEvent::MTE2_S>();
            }

            for (uint32_t i = 0; i < cur_q_len_; ++i) {
                const uint32_t r = g * cur_q_len_ + i;
                if (r >= MAX_PACK_ROWS) break;
                const float inv_row = OnlineSoftmaxTransformer::GetFinalInvDenominator(row_states_[r]);
                // LSE 校正项：burst 域读 UB 条带副本，否则按 (b, hq, token) 全局下标直读
                float lse_corr_val = 0.0f;
                if (has_lse_corr) {
                    if (lse_burst_ok) {
                        lse_corr_val = ub_corr.GetValue(i);
                    } else {
                        const int64_t corr_offset = lse_layout(tla::MakeCoord(b_idx, hq_curr, sq_start + i)) *
                                                    static_cast<int64_t>(sizeof(float));
                        lse_corr_val = *reinterpret_cast<const __gm__ float*>(lse_correction_gm_ + corr_offset);
                    }
                }
                // LSE 出口：核内 exp2 域 LSE 含 448· 放大（乘法实现）所致的 +log2(448) 偏置，
                // 以 S_FP8_OFFSET 常量补扣（唯一定义，见 ConvertLseToHost 注释）
                const float lse_host = OnlineSoftmaxTransformer::ConvertLseToHost(
                    row_states_[r].lse, tiling_->sm_scale_log2e * INV_LOG2E,
                    lse_corr_val, has_lse_corr, S_FP8_OFFSET);

                LocalTensor<float> ub_o_row(TPosition::VECCALC,
                                            Catlass::SageAttn::Vector1UbBudget::O_ACCUM_OFFSET + r * d_ * sizeof(float),
                                            d_);
                dev_vf::FinalizeRowVF<kDtype>(reinterpret_cast<__ubuf__ float*>(ub_o_row.GetPhyAddr()),
                                     reinterpret_cast<__ubuf__ uint16_t*>(ub_half.GetPhyAddr()),
                                     inv_row,
                                     has_vm ? reinterpret_cast<__ubuf__ float*>(ub_vm.GetPhyAddr()) : nullptr,
                                     tiling_->d_og);
                // V 管写 → MTE3 读（DEFECT-I-1）：DataCopy 前置 V→MTE3 事件对，
                // 保证本行 VF 写毕才发起 GM 搬出（支持 half 与 bfloat16 双路径，DEFECT-F-3）
                Catlass::SageAttn::Sync<HardEvent::V_MTE3>();

                const int64_t out_gm_offset = out_layout(tla::MakeCoord(b_idx, hq_curr, sq_start + i, 0u)) *
                                              static_cast<int64_t>(sizeof(half));
                // M5：输出位型编译期分流（原运行时 DtypeMode()==2 分支的 if constexpr 形态，
                // 档位常量唯一定义 sage_attention2_template_tiling_key.h）
                if constexpr (kDtype == kTplDtypeBf16) {
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
                Catlass::SageAttn::Sync<HardEvent::MTE3_V>();

                if (lse_gm_ != nullptr) {
                    if (lse_burst_ok) {
                        ub_lse_row.SetValue(i, lse_host);
                    } else {
                        const int64_t lse_offset = lse_layout(tla::MakeCoord(b_idx, hq_curr, sq_start + i)) *
                                                   static_cast<int64_t>(sizeof(float));
                        *reinterpret_cast<__gm__ float*>(lse_gm_ + lse_offset) = lse_host;
                    }
                }
            }
            if (lse_gm_ != nullptr && lse_burst_ok) {
                // 行条带 S 管写毕 → MTE3 搬出；搬出排空后方可在下一头复用该行（MTE3→S）
                Catlass::SageAttn::Sync<HardEvent::S_MTE3>();
                GlobalTensor<float> gm_lse;
                gm_lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                    lse_gm_ + lse_elem_base * static_cast<int64_t>(sizeof(float))));
                DataCopy(gm_lse, ub_lse_row, cur_q_len_);
                Catlass::SageAttn::Sync<HardEvent::MTE3_S>();
            }
        }
#else
        if (sim_mem_ == nullptr || out_gm_ == nullptr) {
            return;
        }
        for (uint32_t g = 0; g < cur_heads_; ++g) {
            const int64_t hq_curr = hkv_idx * static_cast<int64_t>(gqa_ratio_) + static_cast<int64_t>(cur_g0_) + g;
            if (hq_curr >= hq_) break;

            for (uint32_t i = 0; i < cur_q_len_; ++i) {
                const uint32_t r = g * cur_q_len_ + i;
                if (r >= MAX_PACK_ROWS) break;
                const float inv_row =
                    OnlineSoftmaxTransformer::GetFinalInvDenominator(row_states_[r]);
                // LSE 校正项：lseCorrection GM 按 (b, hq, token) 全局下标直读（与设备分支同式）
                float lse_corr_val = 0.0f;
                if (has_lse_corr) {
                    const int64_t corr_offset = lse_layout(tla::MakeCoord(b_idx, hq_curr, sq_start + i)) *
                                                static_cast<int64_t>(sizeof(float));
                    lse_corr_val = *reinterpret_cast<const float*>(lse_correction_gm_ + corr_offset);
                }
                const float lse_host = OnlineSoftmaxTransformer::ConvertLseToHost(
                    row_states_[r].lse, tiling_->sm_scale_log2e * INV_LOG2E,
                    lse_corr_val, has_lse_corr, S_FP8_OFFSET);

                for (uint32_t c = 0; c < tiling_->d_og; ++c) {
                    const float val = sim_mem_->o_accum[static_cast<size_t>(r) * d_ + c] * inv_row;
                    const int64_t out_elem = out_layout(tla::MakeCoord(b_idx, hq_curr, sq_start + i, c));
                    WriteOutputBits(reinterpret_cast<uint16_t *>(out_gm_) + out_elem, 0, val);
                }

                if (lse_gm_ != nullptr) {
                    const int64_t lse_offset = lse_layout(tla::MakeCoord(b_idx, hq_curr, sq_start + i)) *
                                               static_cast<int64_t>(sizeof(float));
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
    // smooth_k 的行级 lse_correction（注册入参 4）：桥接层按开源契约算好（fp32
    // q·km_broadcast^T，仅 return_lse 时非空），V1 Finalize 按 (b, hq, token) 全局
    // 下标直读；校正项启用与否以本指针非空为唯一定义（V0/V1 跨 AIV 不共享私有
    // 状态，核内不得自算）。
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
    // GQA 分块打包状态（pack/gchunk 唯一推导在 Init，逐任务解码在 DecodeTask）
    uint32_t pack_heads_{1};
    uint32_t gqa_chunks_{1};
    uint32_t cur_g0_{0};    // 本任务窗口首个 Q 头在 KV 头内的全局序号
    uint32_t cur_heads_{1}; // 本任务窗口打包的 Q 头数

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
    OnlineSoftmaxState row_states_[MAX_PACK_ROWS];

};

/**
 * M5 编译期位型特化启动器：模板核执行器 Run<DTYPE_MODE> + RunByDtype 运行时一次分派。
 * 两者独立于模板 tiling key 宏存在（裸 bisheng 编译门无需 ASCENDC_TPL_ARGS_DECL）；
 * canonical 模板档位出包（SAGE_HAS_CANONICAL_TPL_ARGS）时框架按 tiling key 直连
 * Run<INPUT_DTYPE>，不经 RunByDtype。分派谓词与核内 kDtype 同出 NormDtypeMode
 * 唯一定义，两形态同档位实例执行的算术逐位一致。
 */
struct SageAttnLaunch {
    template <uint32_t DTYPE_MODE>
    static __aicore__ void Run(
        __gm__ uint8_t *query,
        __gm__ uint8_t *key,
        __gm__ uint8_t *value,
        __gm__ uint8_t *keyMean,
        __gm__ uint8_t *lseCorrection,
        __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *softmaxLse,
        __gm__ uint8_t *workspace,
        const SageAttention2TilingData *tiling_data)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        SageAttnKernel<DTYPE_MODE> kernel;
        kernel.Init(query, key, value, keyMean, lseCorrection, attentionOut, softmaxLse, workspace, tiling_data);
#else
        (void)workspace;
        SageAttnKernel<DTYPE_MODE> kernel;
        kernel.Init(query, key, value, keyMean, lseCorrection, attentionOut, softmaxLse, nullptr, tiling_data);
#endif
        kernel.Process();
    }

    static __aicore__ void RunByDtype(
        uint32_t dtype_mode,
        __gm__ uint8_t *query,
        __gm__ uint8_t *key,
        __gm__ uint8_t *value,
        __gm__ uint8_t *keyMean,
        __gm__ uint8_t *lseCorrection,
        __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *softmaxLse,
        __gm__ uint8_t *workspace,
        const SageAttention2TilingData *tiling_data)
    {
        if (NormDtypeMode(dtype_mode) == kTplDtypeBf16) {
            Run<kTplDtypeBf16>(query, key, value, keyMean, lseCorrection, attentionOut, softmaxLse, workspace, tiling_data);
        } else {
            Run<kTplDtypeFp16>(query, key, value, keyMean, lseCorrection, attentionOut, softmaxLse, workspace, tiling_data);
        }
    }
};

} // namespace sage_attn

/**
 * AscendC 全局设备核函数入口 (C 链接)，函数名须与 op 注册名一致（小写下划线）——
 * CANN opc autogen 以 origin_func_name="sage_attention2" 在内核源文件中检索入口，
 * 名称不符报 "kernel entry not implement"。
 * 声明 KERNEL_TYPE_MIX_AIC_1_2 硬件拓扑
 */
#if SAGE_ATTN_DEVICE_COMPILE && defined(SAGE_HAS_CANONICAL_TPL_ARGS)
/**
 * canonical 模板档位入口（OPP opc autogen 出包形态，competitor 同款）：框架依
 * ASCENDC_TPL_ARGS_DECL 声明与 host SetTilingKey 的 tiling key 直连模板实例，
 * 核内零运行时位型分派。裸 bisheng 门禁（无 template_argument.h）编入下方
 * extern "C" 运行时 RunByDtype 形态。
 */
template <uint32_t INPUT_DTYPE>
__global__ __aicore__ void sage_attention2(
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
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(sage_attn::SageAttention2TilingData);
    GET_TILING_DATA_WITH_STRUCT(sage_attn::SageAttention2TilingData, t, tiling);
    __gm__ uint8_t *user_workspace = GetUserWorkspace(workspace);
    sage_attn::SageAttnLaunch::Run<INPUT_DTYPE>(query, key, value, keyMean, lseCorrection,
                                                attentionOut, softmaxLse, user_workspace, &t);
}
#else
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

    // canonical 注册宏（对齐仓内规范；宿主与 KT 编译环境提供等价模拟宏）：
    // REGISTER_TILING_DEFAULT 登记 TilingData 类型；GET_TILING_DATA_WITH_STRUCT 把
    // GM tiling 一次性拷入核内结构体（免 __gm__ 逐字段远端读）；GetUserWorkspace
    // 取得跳过 CANN 系统 workspace 前缀后的用户区指针。
#if SAGE_ATTN_DEVICE_COMPILE
    REGISTER_TILING_DEFAULT(sage_attn::SageAttention2TilingData);
    GET_TILING_DATA_WITH_STRUCT(sage_attn::SageAttention2TilingData, t, tiling);
    __gm__ uint8_t *user_workspace = GetUserWorkspace(workspace);
    // M5：裸 bisheng 形态按 tiling.input_dtype 运行时一次分派到模板实例
    //（分派在任务 prologue，热路径零位型分支；canonical 模板档位出包时由
    // 上方模板入口直连，无此运行时归一）
    sage_attn::SageAttnLaunch::RunByDtype(t.input_dtype, query, key, value, keyMean, lseCorrection,
                                          attentionOut, softmaxLse, user_workspace, &t);
#else
    (void)workspace;
    // 基线 lse_correction 由桥接层算好传入（开源契约），本入口直承 GM 指针，
    // V1 FinalizeAndStoreOutput 消费（校正启用以指针非空为唯一定义）
    const sage_attn::SageAttention2TilingData *tiling_data =
        reinterpret_cast<const sage_attn::SageAttention2TilingData *>(tiling);
    // M5：宿主/KT 分支保持 tiling.input_dtype 运行时一次分派（测试契约不变）
    sage_attn::SageAttnLaunch::RunByDtype(tiling_data->input_dtype, query, key, value, keyMean, lseCorrection,
                                          attentionOut, softmaxLse, nullptr, tiling_data);
#endif
}
#endif
