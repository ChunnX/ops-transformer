/**
 * Verification test for SageAttention2 TilingData, OnlineSoftmax and CATLASS contracts
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/ascendc/online_softmax.h"
#include "../sageattn_fp8/catlass/sage_tile.hpp"
#include "../sageattn_fp8/catlass/sage_resource.hpp"
#include "../sageattn_fp8/catlass/sage_qk_mmad.hpp"
#include "../sageattn_fp8/catlass/sage_pv_mmad.hpp"
#include "../op_kernel/static_slot.h"

using namespace sage_attn;
using namespace Catlass::SageAttn;

void TestTilingDataLayout() {
    std::cout << "[Test 1] Testing TilingData struct size and alignment..." << std::endl;

    assert(sizeof(SageAttention2TilingData) == 160);
    assert(alignof(SageAttention2TilingData) == 32);
    assert(sizeof(SageAttention2TilingData) % 32 == 0);

    // Verify Shape fields offsets
    assert(offsetof(SageAttention2TilingData, b) == 0);
    assert(offsetof(SageAttention2TilingData, hq) == 4);
    assert(offsetof(SageAttention2TilingData, hkv) == 8);
    assert(offsetof(SageAttention2TilingData, sq) == 12);
    assert(offsetof(SageAttention2TilingData, sk) == 16);
    assert(offsetof(SageAttention2TilingData, d_og) == 20);
    assert(offsetof(SageAttention2TilingData, d_pad) == 24);

    // Verify Mode fields offsets
    assert(offsetof(SageAttention2TilingData, is_causal) == 28);
    assert(offsetof(SageAttention2TilingData, tensor_layout) == 32);
    assert(offsetof(SageAttention2TilingData, qk_quant_gran) == 36);
    assert(offsetof(SageAttention2TilingData, pv_accum_dtype) == 40);
    assert(offsetof(SageAttention2TilingData, smooth_k) == 44);
    assert(offsetof(SageAttention2TilingData, smooth_v) == 48);

    // Verify Numerical fields offsets（原 s_fp8_offset 字段已删除，4 字节并入尾部保留域）
    assert(offsetof(SageAttention2TilingData, sm_scale_log2e) == 52);
    assert(offsetof(SageAttention2TilingData, v_scale_max) == 56);

    // Verify Tiling/blocking fields offsets
    assert(offsetof(SageAttention2TilingData, m_per_core) == 60);
    assert(offsetof(SageAttention2TilingData, block_n) == 64);
    assert(offsetof(SageAttention2TilingData, k_step) == 68);

    // Verify Core splitting fields offsets
    assert(offsetof(SageAttention2TilingData, core_num_used) == 72);
    assert(offsetof(SageAttention2TilingData, q_blk_num) == 76);
    assert(offsetof(SageAttention2TilingData, kv_blk_num) == 80);
    assert(offsetof(SageAttention2TilingData, split_kv_num) == 84);
    assert(offsetof(SageAttention2TilingData, is_merge) == 88);

    // Verify Layout & offset fields
    assert(offsetof(SageAttention2TilingData, l1_q_offset) == 92);
    assert(offsetof(SageAttention2TilingData, l1_q_scale_offset) == 96);
    assert(offsetof(SageAttention2TilingData, l1_kv_offset) == 100);
    assert(offsetof(SageAttention2TilingData, l1_kv_scale_offset) == 104);
    assert(offsetof(SageAttention2TilingData, ub_acc_offset) == 108);
    assert(offsetof(SageAttention2TilingData, ub_softmax_offset) == 112);
    assert(offsetof(SageAttention2TilingData, ub_quant_offset) == 116);
    assert(offsetof(SageAttention2TilingData, workspace_lse_offset) == 120);
    assert(offsetof(SageAttention2TilingData, workspace_vm_offset) == 128);
    assert(offsetof(SageAttention2TilingData, enable_die_affinity) == 136);
    assert(offsetof(SageAttention2TilingData, die_num) == 140);
    assert(offsetof(SageAttention2TilingData, cores_per_die) == 144);
    assert(offsetof(SageAttention2TilingData, tasks_per_die) == 148);
    assert(offsetof(SageAttention2TilingData, input_dtype) == 152);
    assert(offsetof(SageAttention2TilingData, reserved) == 153);

    // Capacity constants：L1/Bank/UB 保留 sage_attn 直引（L1 经官方 Arch::Ascend950
    // 派生）；L0A/L0B 无生产判据，不再设配额别名也不做官方源canary；L0C 唯一映射点
    // 为 HardwareResourceQuota（InspectTilingPlan 判据消费方）
    assert(L1_TOTAL_SIZE == 512 * 1024);
    assert(L1_BANK_SIZE == 256 * 1024);
    assert(UB_TOTAL_SIZE_PER_V == 256 * 1024);
    static_assert(HardwareResourceQuota::L0C_TOTAL_BYTES == Catlass::Arch::Ascend950::L0C_SIZE,
                  "L0C quota must map the official Ascend950 constant");
    static_assert(HardwareResourceQuota::MAX_CROSS_CORE_SYNC_DEPTH == 15,
                  "Cross-core sync queue depth quota must be 15 (Ascend 950/910B hardware limit)");
    assert(L1_TOTAL_SIZE == Catlass::Arch::Ascend950::L1_SIZE);

    std::cout << "[Test 1 Passed] All TilingData offsets and constants match specification!" << std::endl;
}

void TestOnlineSoftmax() {
    std::cout << "[Test 2] Testing Online Softmax state transitions (恒逐块强锚定重缩放)..." << std::endl;

    OnlineSoftmaxState state;
    state.Reset();

    // Step 1: 首个有效分块初始化：返回 1.0 且 need_rescale=false
    float rescale1 = OnlineSoftmaxTransformer::Step(state, 10.0f, 100.0f);
    assert(rescale1 == 1.0f);
    assert(state.m_run == 10.0f);
    assert(state.m_scale == 10.0f);
    // assert(!state.need_rescale); // Now need_rescale is unconditionally true except init

    // Step 2: 第二块（max 12.0）：每块必重缩放（恒逐块强锚定，无跳步分支），
    // rescale = 2^(m_scale_old - m_new) = 2^(10 - 12) = 0.25，锚点前移
    float rescale2 = OnlineSoftmaxTransformer::Step(state, 12.0f, 50.0f);
    assert(state.need_rescale);
    assert(state.m_run == 12.0f);
    assert(state.m_scale == 12.0f); // 恒逐块强锚定：m_scale 恒等于 m_run
    assert(std::fabs(rescale2 - std::exp2(10.0f - 12.0f)) < 1e-6f);

    // Step 3: 第三块（max 20.0）：同样逐块重缩放，rescale = 2^(12 - 20) = 2^-8
    float rescale3 = OnlineSoftmaxTransformer::Step(state, 20.0f, 80.0f);
    assert(state.need_rescale);
    assert(state.m_run == 20.0f);
    assert(state.m_scale == 20.0f);
    float expected_rescale = std::exp2(12.0f - 20.0f);
    assert(std::fabs(rescale3 - expected_rescale) < 1e-5f);

    // Test LSE conversion
    float sm_scale = 1.0f / 8.0f; // sqrt(64)
    float lse_corr = 0.5f;
    float lse_host_smooth = OnlineSoftmaxTransformer::ConvertLseToHost(state.lse, sm_scale, lse_corr, true, S_FP8_OFFSET);
    float lse_host_no_smooth = OnlineSoftmaxTransformer::ConvertLseToHost(state.lse, sm_scale, lse_corr, false, S_FP8_OFFSET);
    assert(lse_host_smooth > lse_host_no_smooth);
    assert(std::fabs((lse_host_smooth - lse_host_no_smooth) - (lse_corr * sm_scale)) < 1e-6f);

    std::cout << "[Test 2 Passed] Online Softmax 恒逐块强锚定重缩放行为验证通过!" << std::endl;
}

// 清理记录（2026-10-03 R5 收敛清扫随动）：DeclaredReferenceSoftmax 已按"零调用
// 死代码"自 online_softmax.h 整块删除，本用例随之移除；Split-KV 归并重入 DR 时
// 按 online_softmax.h 清理记录恢复并补单测（FoldAttention 数值覆盖现由
// tests/test_fold_attention.py 的原型级用例承担）。
// 清理记录（2026-10-03 死代码清扫）：AccumReduceController / FixpipeConfig /
// QuantizeToFp16（accum_reduce.h）与 OutputRetireQueue 均为"热路径已弃用策略"的
// 零生产消费载体——设备 PV 归约节奏唯一定义点为 twin::AccumulatePvRow（k_step
// 子块域）+ 设备侧逐块 FixPipe 裸搬合并（档位判据直读 tiling.pv_accum_dtype），
// 原独立控制器类与其测试（本文件 Test 3、test_output_queue.cpp Test 10）随之删除。

void TestCatlassContracts() {
    std::cout << "[Test 3] Testing CATLASS tile and resource contracts..." << std::endl;

    // L1 双 Bank 契约（生产链路实际消费的槽布局唯一定义点：kernel LoadAndQuantizeQ /
    // QuantizeAndStageKey/Value 与 aclnn host tiling 均按此寻址）
    static_assert(L1DoubleBankContract::BANK0_START == 0, "Bank 0 must start at 0");
    static_assert(L1DoubleBankContract::BANK1_START == 256 * 1024, "Bank 1 must start at 256KB");
    static_assert(L1DoubleBankContract::Q_MAX_BYTES == 128 * 128, "Q data region 128x128 INT8");
    static_assert(L1DoubleBankContract::KV_SLOT_BYTES == 128 * 1024, "KV slot = K 64KB + V 64KB");
    static_assert(L1DoubleBankContract::KV_V_SUB_OFFSET == 64 * 1024, "V sub-slot at slot midpoint");
    static_assert(L1DoubleBankContract::KV_PONG_OFFSET ==
                      L1DoubleBankContract::KV_PING_OFFSET + L1DoubleBankContract::KV_SLOT_BYTES,
                  "ping/pong slots must be adjacent in Bank 1");

    // A7 GM scale 交接条带（V0→V1 桥，STAGE_* 唯一定义点 tiling.h；kernel 侧
    // StageV1ScaleStrips/StageKScaleStrip/QuantizeAndStageKey 全按本组常量槽寻址，
    // aclnn host tiling 按 STAGE_CORE_STRIDE × core_num_used 预留）：三段布局单调相接，
    // 126 个 K scale 槽恰好填满单核条带余量
    static_assert(STAGE_Q_SCALE_OFFSET == 0, "Q scale at stage strip start");
    static_assert(STAGE_V_SCALE_OFFSET == STAGE_Q_SCALE_OFFSET + STAGE_SLOT_BYTES,
                  "V scale slot follows Q scale slot");
    static_assert(STAGE_K_SCALE_BASE_OFFSET == STAGE_V_SCALE_OFFSET + STAGE_SLOT_BYTES,
                  "K scale slot region follows V scale slot");
    static_assert(STAGE_K_SCALE_BASE_OFFSET + STAGE_K_SCALE_NUM_SLOTS * STAGE_SLOT_BYTES ==
                      STAGE_CORE_STRIDE,
                  "K scale slots must exactly fill the per-core stage strip");

    // 硬件 32 字节对齐守卫（昇腾 MTE2/MTE3 通路硬性约束，彻底杜绝非对齐 TRAP）
    static_assert(STAGE_CORE_STRIDE % 32 == 0, "STAGE_CORE_STRIDE must be 32-byte aligned");
    static_assert(STAGE_Q_SCALE_OFFSET % 32 == 0, "STAGE_Q_SCALE_OFFSET must be 32-byte aligned");
    static_assert(STAGE_V_SCALE_OFFSET % 32 == 0, "STAGE_V_SCALE_OFFSET must be 32-byte aligned");
    static_assert(STAGE_K_SCALE_BASE_OFFSET % 32 == 0, "STAGE_K_SCALE_BASE_OFFSET must be 32-byte aligned");
    static_assert(STAGE_SLOT_BYTES % 32 == 0, "STAGE_SLOT_BYTES must be 32-byte aligned");
    static_assert((STAGE_SLOT_FLOATS * sizeof(float)) % 32 == 0, "STAGE_SLOT_FLOATS bytes must be 32-byte aligned");
    static_assert((64 * sizeof(float)) % 32 == 0, "block_n=64 float bytes must be 32-byte aligned");
    static_assert((128 * sizeof(float)) % 32 == 0, "block_n=128 float bytes must be 32-byte aligned");

    // 清理记录（2026-10-03 死代码清扫随动）：
    // - KvSlotLayoutContract：生产链路零实例化且槽内 Scale 偏移描述与核内实际
    //   （block_n·d 直排，非 RoundUp）不一致，整块删除，本用例随之移除；
    // - SageResourceManager::ValidateBudget：被 InspectTilingPlan（RES-01 全量台账）
    //   取代的旧静态校验，零生产调用方，删除，本用例随之移除；
    // - SageResourceManager::ValidateTilingPlan：InspectTilingPlan 的布尔包装，
    //   零生产调用方（host tiling 唯一判据为 InspectTilingPlan），2026-10-04 删除，
    //   本用例改直引 InspectTilingPlan == kNone。
    // 原 AMLA FloatMulPow2Int（本文件原 Test 5）随 quant_common.h 的零调用原语一并删除：
    // base-2 域重缩放的现行唯一轨迹为 OnlineSoftmaxTransformer::FastExp2（位构造自实现）。

    std::cout << "[Test 3 Passed] CATLASS contracts and resource validation verified!" << std::endl;
}

void TestResourceFootprintLedger() {
    std::cout << "[Test 4] Testing runtime footprint ledger (InspectTilingPlan / RES-01)..." << std::endl;

    // 验收域内全部分块计划必须通过（P-01..P-07 覆盖的组合）：
    // gqa=1 (rows=64), gqa=2 (rows=128, m_per_core=64), gqa>=3 (rows=128, m_per_core=32),
    // block_n 64 (sk<2048) / 128 (sk>=2048), d_pad 64/128
    assert(SageResourceManager::InspectTilingPlan(64, 64, 128) == FootprintViolation::kNone);
    assert(SageResourceManager::InspectTilingPlan(64, 128, 128) == FootprintViolation::kNone);
    assert(SageResourceManager::InspectTilingPlan(128, 64, 128) == FootprintViolation::kNone);
    assert(SageResourceManager::InspectTilingPlan(128, 128, 128) == FootprintViolation::kNone);
    assert(SageResourceManager::InspectTilingPlan(64, 64, 64) == FootprintViolation::kNone);
    assert(SageResourceManager::InspectTilingPlan(128, 128, 64) == FootprintViolation::kNone);

    // 预算边界恰好填满必须放行（= 容量为合法，> 容量才违规）：
    // rows=128, d=128 -> O 累加器 128*128*4 = 64KB == O_ACCUM_BYTES
    assert(SageResourceManager::InspectTilingPlan(128, 128, 128) == FootprintViolation::kNone);

    // 零形状：分块计划本身非法
    assert(SageResourceManager::InspectTilingPlan(0, 64, 128) == FootprintViolation::kInvalidShape);
    assert(SageResourceManager::InspectTilingPlan(64, 0, 128) == FootprintViolation::kInvalidShape);
    assert(SageResourceManager::InspectTilingPlan(64, 64, 0) == FootprintViolation::kInvalidShape);

    // GQA 行溢出：total_rows > MAX_PACK_ROWS(128)（tiling.h 唯一定义点，kernel 侧行状态容量契约）
    assert(SageResourceManager::InspectTilingPlan(256, 64, 128) == FootprintViolation::kGqaRowOverflow);

    // L1 Bank 0 Q 数据区：rows*d INT8 > 16KB（Q_MAX_BYTES = 128*128）
    assert(SageResourceManager::InspectTilingPlan(128, 64, 130) == FootprintViolation::kL1Bank0Q);

    // 收敛轮 C 勘误：kUbV1OAccum 分支在量纲上被 kL1Bank0Q 严格支配——
    // 两者同为 rows*d 积，Q 上界 16384 < O 上界 65536=4*16384，故任何触发
    // O 越界的 (rows,d) 必先触发 Q 越界（分支为防御性算术，恒不可达属预期）。
    // d=260 实际命中 kL1Bank0Q：
    assert(SageResourceManager::InspectTilingPlan(64, 64, 260) == FootprintViolation::kL1Bank0Q);

    // V1 UB S(P) 奇偶双槽：2*rows*block_n > 64KB（FIXPIPE_RECV 区）
    assert(SageResourceManager::InspectTilingPlan(128, 264, 64) == FootprintViolation::kUbV1SoftmaxSlot);
    // 边界：2*128*256 = 64KB 恰好填满 FIXPIPE_RECV，放行
    assert(SageResourceManager::InspectTilingPlan(128, 256, 64) != FootprintViolation::kUbV1SoftmaxSlot);

    // R17-2 全区建模：A7 dot32 交接区 rows·block_n·4 > 64KB（128·256·4 = 128KB 越界；
    // 上一行只断言"非 S 槽越界"，此处锁定其命中的是 A7 交接区违规码。pv_partial 侧
    // rows·d·4 > 64KB 与 kL1Bank0Q 的 rows·d > 16KB 同量纲、被其严格支配，恒不可达属预期）
    assert(SageResourceManager::InspectTilingPlan(128, 256, 64) == FootprintViolation::kUbV1A7Handoff);

    // V0 UB GM 输入槽：block_n*d*2 > 48KB
    assert(SageResourceManager::InspectTilingPlan(64, 200, 128) == FootprintViolation::kUbV0Input);

    // constexpr 可用于编译期断言（分块常量改动的编译门）
    static_assert(SageResourceManager::InspectTilingPlan(128, 128, 128) ==
                      FootprintViolation::kNone,
                  "P-01..P-07 acceptance domain must fit on-chip budgets");
    static_assert(SageResourceManager::InspectTilingPlan(256, 64, 128) ==
                  FootprintViolation::kGqaRowOverflow,
                  "row overflow must be caught at compile time when constants hard-code it");

    std::cout << "[Test 4 Passed] Runtime footprint ledger verified!" << std::endl;
}

void TestGmTlaLayouts() {
    std::cout << "[Test 5] Testing tla GM layout single point (MakeGmBhsdLayout / MakeGmBhsLayout)..." << std::endl;

    // GM 四维布局 tla 单点 vs 原手写步距四元组位元等价（HND/NHD 各抽基址与行偏移；
    // kernel 侧 Q/K/V/O 布局推导已全部收敛至该单点，本用例锁值防回归）
    const auto hnd = MakeGmBhsdLayout(2, 4, 8, 128, 1);
    // HND: sb=H·S·D, sh=S·D, ss=D
    assert(hnd(tla::MakeCoord(uint32_t(1), int64_t(3), int64_t(5), 0u)) ==
           1 * (4 * 8 * 128) + 3 * (8 * 128) + 5 * 128);
    assert(hnd(tla::MakeCoord(uint32_t(0), int64_t(0), int64_t(0), 0u)) == 0);

    const auto nhd = MakeGmBhsdLayout(2, 4, 8, 128, 0);
    // NHD: sb=S·H·D, sh=D, ss=H·D
    assert(nhd(tla::MakeCoord(uint32_t(1), int64_t(3), int64_t(5), 0u)) ==
           1 * (8 * 4 * 128) + 3 * 128 + 5 * (4 * 128));

    // (b,hq,sq) 稠密布局（LSE / lseCorrection）：b·(H·S) + h·S + s
    const auto dense = MakeGmBhsLayout(2, 4, 8);
    assert(dense(tla::MakeCoord(uint32_t(1), int64_t(3), int64_t(5))) == 1 * (4 * 8) + 3 * 8 + 5);

    // constexpr 域（编译期可折叠，与运行时同式）
    static_assert(MakeGmBhsLayout(2, 4, 8)(tla::MakeCoord(uint32_t(1), int64_t(3), int64_t(5))) == 61,
                  "dense layout must fold at compile time");

    std::cout << "[Test 5 Passed] tla GM layout equivalence verified!" << std::endl;
}

void TestDieAffinityContract() {
    std::cout << "[Test 6] Testing STARS 2.0 die affinity closed-form mapping contract..." << std::endl;

    // STARS 2.0 拓扑常量契约（白皮书第 4.4 节：最多 8 个 Group、按 Die 亲和；
    // 唯一定义点 ascendc/tiling.h，host 判据与核内解码共享同源）
    static_assert(ATLAS_950_DIE_NUM == 2, "Atlas 950PR is a dual-die package");
    static_assert(STARS_MAX_GROUP_NUM == 8, "STARS 2.0 supports at most 8 hardware groups");
    static_assert(STARS_GROUPS_PER_DIE == 4, "each 950PR die owns 4 STARS core groups");
    static_assert(STARS_CORES_PER_GROUP == 4, "the full 32-core grid tiles into 8 groups of 4");

    // 闭式映射编译期锁值（与 tiling.h 自锁互为冗余防线，防 constexpr 契约域被弱化）
    static_assert(sage_attn::DieTask(16, 64, 32) == 32,
                  "Die 1 wave 0 must map into the second half [32, 64)");
    static_assert(sage_attn::DieTaskEnd(16, 8, 32, true) == 8,
                  "partial wave must retire Die 0 cores beyond the local task count");

    // 生产域穷举：偶数任务总数 × 全部偶数核网格（2..32 全覆盖，含满配/减配/小网格），验证三点：
    // (1) 双射性：每个虚拟任务恰好被一个物理核执行一次，[0, total) 无遗漏无重复；
    // (2) Die 半分不变量：Die 0 (核 [0, cores/2)) 只触达 [0, total/2)，Die 1 只触达后半；
    // (3) Die 内负载均衡：同一 Die 各活跃核任务数相差不超过 1，且两 Die 各承担一半。
    const uint32_t core_grids[] = {2u, 4u, 6u, 8u, 10u, 12u, 14u, 16u,
                                   18u, 20u, 22u, 24u, 26u, 28u, 30u, 32u};
    const uint64_t totals[] = {2ull, 14ull, 16ull, 28ull, 64ull, 128ull, 1024ull};
    for (uint32_t cores : core_grids) {
        for (uint64_t total : totals) {
            std::vector<bool> covered(total, false);
            uint64_t die_load[2] = {0ull, 0ull};
            uint64_t die_load_min[2] = {~0ull, ~0ull};
            uint64_t die_load_max[2] = {0ull, 0ull};
            for (uint32_t c = 0; c < cores; ++c) {
                const uint32_t die = (c < cores / 2u) ? 0u : 1u;
                const uint64_t task_end = sage_attn::DieTaskEnd(total, c, cores, true);
                uint64_t my_load = 0;
                for (uint64_t v = c; v < task_end; v += cores) {
                    const uint64_t mapped = sage_attn::DieTask(v, total, cores);
                    assert(mapped < total);
                    assert(!covered[mapped]);
                    covered[mapped] = true;
                    assert((mapped < total / 2) == (die == 0u));
                    ++my_load;
                }
                if (my_load > 0) {
                    die_load[die] += my_load;
                    die_load_min[die] = (my_load < die_load_min[die]) ? my_load : die_load_min[die];
                    die_load_max[die] = (my_load > die_load_max[die]) ? my_load : die_load_max[die];
                }
            }
            for (uint64_t i = 0; i < total; ++i) {
                assert(covered[i]);
            }
            assert(die_load[0] == total / 2);
            assert(die_load[1] == total - total / 2);
            assert(die_load_max[0] - die_load_min[0] <= 1);
            assert(die_load_max[1] - die_load_min[1] <= 1);
        }
    }

    // (Batch,Head) 块整对齐定理实证：b·hkv=4（偶），基块粒度 gqa_chunk·q_blk=2·3=6，
    // 核域总数 24：Die 边界 12 为基块粒度整数倍 ⇒ 4 个 (b,h) 基块 [0,6) [6,12) [12,18)
    // [18,24) 各自整块归属单一 Die，无跨 Die 批次分裂（KV 流 100% 命中本 Die 本地 L2）。
    {
        const uint32_t cores = 32;
        constexpr uint64_t kTotal = 24;
        constexpr uint64_t kBlock = 6;  // gqa_chunk=2 × q_blk=3
        std::vector<uint32_t> exec_die(static_cast<size_t>(kTotal), 2u);  // 2=未执行哨兵
        for (uint32_t c = 0; c < cores; ++c) {
            const uint32_t die = (c < cores / 2u) ? 0u : 1u;
            const uint64_t task_end = sage_attn::DieTaskEnd(kTotal, c, cores, true);
            for (uint64_t v = c; v < task_end; v += cores) {
                exec_die[static_cast<size_t>(sage_attn::DieTask(v, kTotal, cores))] = die;
            }
        }
        for (uint64_t bh = 0; bh < kTotal / kBlock; ++bh) {
            const uint32_t first = exec_die[static_cast<size_t>(bh * kBlock)];
            for (uint64_t t = bh * kBlock + 1; t < (bh + 1) * kBlock; ++t) {
                assert(exec_die[static_cast<size_t>(t)] != 2u);  // 哨兵：防全核空转退化下比较恒真
                assert(exec_die[static_cast<size_t>(t)] == first);
            }
        }
    }

    // 退化恒等域（单 Die 环境/老用例等价单组：零行为差异）
    assert(sage_attn::DieTask(7, 0, 32) == 7);               // total < 2
    assert(sage_attn::DieTask(7, 1, 32) == 7);
    assert(sage_attn::DieTask(7, 16, 1) == 7);               // cores < 2
    assert(sage_attn::DieTaskEnd(16, 3, 32, false) == 16);   // 亲和关断 → 全量边界
    assert(sage_attn::DieTaskEnd(16, 3, 31, true) == 16);    // 奇数核拓扑 → 全量边界
    assert(sage_attn::DieTaskEnd(1, 0, 32, true) == 1);      // 极小任务域

    std::cout << "[Test 6 Passed] STARS 2.0 die affinity closed-form mapping verified!" << std::endl;
}

void TestNddmaStrideContract() {
    std::cout << "[Test 7] Testing NDDMA 2D Strided DMA contract..." << std::endl;
    // HND 布局：同一 Head 内连续，跨步恒为 0
    assert(Catlass::SageAttn::CalcGm2dStride32B(32, 128, 2, sage_attn::LAYOUT_HND) == 0);
    assert(Catlass::SageAttn::CalcGm2dStride32B(8, 64, 2, sage_attn::LAYOUT_HND) == 0);
    assert(Catlass::SageAttn::CalcGm2dStride32B(1, 128, 2, sage_attn::LAYOUT_HND) == 0);

    // NHD 布局 ([B, S, H, D])：
    // 单 Head 与边界退化：H=1 或 H=0 时相邻 Token 连续/安全退化，跨步恒为 0，无下溢
    assert(Catlass::SageAttn::CalcGm2dStride32B(1, 128, 2, sage_attn::LAYOUT_NHD) == 0);
    assert(Catlass::SageAttn::CalcGm2dStride32B(0, 128, 2, sage_attn::LAYOUT_NHD) == 0);
    // 多 Head：stride = (H - 1) * (d * 2 / 32)
    // H=32, d=128: 31 * (128 * 2 / 32) = 31 * 8 = 248
    assert(Catlass::SageAttn::CalcGm2dStride32B(32, 128, 2, sage_attn::LAYOUT_NHD) == 248);
    // H=8, d=64: 7 * (64 * 2 / 32) = 7 * 4 = 28
    assert(Catlass::SageAttn::CalcGm2dStride32B(8, 64, 2, sage_attn::LAYOUT_NHD) == 28);
    // H=4, d=128: 3 * 8 = 24
    assert(Catlass::SageAttn::CalcGm2dStride32B(4, 128, 2, sage_attn::LAYOUT_NHD) == 24);

    // 与 MakeGmBhsdLayout 步距单点位元严格等价契约实证 (d=128 与 d=64 双规格覆盖)：
    // NHD 下相邻两个 Token (s=0 与 s=1) 在 GM 中的地址差（以 half 元素计）
    // 恰好等于 (stride_32b_blocks * 32 / sizeof(half)) + d
    const uint32_t test_h = 16;
    const uint32_t test_d = 128;
    const auto nhd_layout = Catlass::SageAttn::MakeGmBhsdLayout(1, test_h, 64, test_d, 0);
    const int64_t token0_offset = nhd_layout(tla::MakeCoord(0u, 3u, 0u, 0u));
    const int64_t token1_offset = nhd_layout(tla::MakeCoord(0u, 3u, 1u, 0u));
    const int64_t elem_diff = token1_offset - token0_offset; // 应为 test_h * test_d
    assert(elem_diff == static_cast<int64_t>(test_h) * test_d);

    const uint32_t stride_32b = Catlass::SageAttn::CalcGm2dStride32B(test_h, test_d, 2, sage_attn::LAYOUT_NHD);
    const int64_t reconstructed_diff = static_cast<int64_t>(stride_32b * 32 / 2) + test_d;
    assert(elem_diff == reconstructed_diff);

    // d=64 规格位元覆盖：
    const uint32_t test_h_64 = 8;
    const uint32_t test_d_64 = 64;
    const auto nhd_layout_64 = Catlass::SageAttn::MakeGmBhsdLayout(1, test_h_64, 32, test_d_64, 0);
    const int64_t t0_offset = nhd_layout_64(tla::MakeCoord(0u, 2u, 0u, 0u));
    const int64_t t1_offset = nhd_layout_64(tla::MakeCoord(0u, 2u, 1u, 0u));
    const int64_t diff_64 = t1_offset - t0_offset;
    assert(diff_64 == static_cast<int64_t>(test_h_64) * test_d_64);
    const uint32_t stride_32b_64 = Catlass::SageAttn::CalcGm2dStride32B(test_h_64, test_d_64, 2, sage_attn::LAYOUT_NHD);
    const int64_t reconstructed_64 = static_cast<int64_t>(stride_32b_64 * 32 / 2) + test_d_64;
    assert(diff_64 == reconstructed_64);

    // 昇腾 32 字节硬件突发对齐与 constexpr 编译期锁值：
    static_assert((64u * sizeof(uint16_t)) % 32u == 0u, "d=64 must be 32B aligned");
    static_assert((128u * sizeof(uint16_t)) % 32u == 0u, "d=128 must be 32B aligned");
    static_assert((64u * sizeof(uint16_t)) / 32u == 4u, "d=64 block_len must be 4");
    static_assert((128u * sizeof(uint16_t)) / 32u == 8u, "d=128 block_len must be 8");
    static_assert(Catlass::SageAttn::CalcGm2dStride32B(32, 128, 2, sage_attn::LAYOUT_NHD) == 248,
                  "NHD 32-head stride must fold at compile time");
    static_assert(Catlass::SageAttn::CalcGm2dStride32B(32, 128, 2, sage_attn::LAYOUT_HND) == 0,
                  "HND stride must fold at compile time to 0");
    static_assert(Catlass::SageAttn::CalcGm2dStride32B(0, 128, 2, sage_attn::LAYOUT_NHD) == 0,
                  "H=0 boundary must fold to 0");

    std::cout << "[Test 7 Passed] NDDMA 2D Strided DMA contract verified!" << std::endl;
}

int main() {
    std::cout << "Starting SageAttention2 C++ Contract and Tiling Verification..." << std::endl;
    TestTilingDataLayout();
    TestOnlineSoftmax();
    TestCatlassContracts();
    TestResourceFootprintLedger();
    TestGmTlaLayouts();
    TestDieAffinityContract();
    TestNddmaStrideContract();
    std::cout << "All C++ contracts and struct layouts PASSED successfully!" << std::endl;
    return 0;
}
