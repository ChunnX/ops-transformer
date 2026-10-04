/**
 * Verification test for SageAttention2 TilingData, OnlineSoftmax and CATLASS contracts
 */

#include <iostream>
#include <cassert>
#include <cstddef>
#include <cmath>

#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/ascendc/online_softmax.h"
#include "../sageattn_fp8/catlass/sage_tile.hpp"
#include "../sageattn_fp8/catlass/sage_resource.hpp"
#include "../sageattn_fp8/catlass/sage_qk_mmad.hpp"
#include "../sageattn_fp8/catlass/sage_pv_mmad.hpp"

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
    // Capacity constants：L1/Bank/UB 保留 sage_attn 直引（L1 经官方 Arch::Ascend950
    // 派生）；L0A/L0B 无生产判据，不再设配额别名也不做官方源canary；L0C 唯一映射点
    // 为 HardwareResourceQuota（InspectTilingPlan 判据消费方）
    assert(L1_TOTAL_SIZE == 512 * 1024);
    assert(L1_BANK_SIZE == 256 * 1024);
    assert(UB_TOTAL_SIZE_PER_V == 256 * 1024);
    static_assert(HardwareResourceQuota::L0C_TOTAL_BYTES == Catlass::Arch::Ascend950::L0C_SIZE,
                  "L0C quota must map the official Ascend950 constant");
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

int main() {
    std::cout << "Starting SageAttention2 C++ Contract and Tiling Verification..." << std::endl;
    TestTilingDataLayout();
    TestOnlineSoftmax();
    TestCatlassContracts();
    TestResourceFootprintLedger();
    TestGmTlaLayouts();
    std::cout << "All C++ contracts and struct layouts PASSED successfully!" << std::endl;
    return 0;
}
