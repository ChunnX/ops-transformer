/**
 * Verification test for SageAttention2 TilingData, SyncGuard, OnlineSoftmax, AccumReduce and CATLASS contracts
 */

#include <iostream>
#include <cassert>
#include <cstddef>
#include <cmath>

#include "../sageattn_fp8/ascendc/tiling.h"
#include "../sageattn_fp8/ascendc/sync_guard.h"
#include "../sageattn_fp8/ascendc/online_softmax.h"
#include "../sageattn_fp8/ascendc/accum_reduce.h"
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

    // Verify Numerical fields offsets
    assert(offsetof(SageAttention2TilingData, sm_scale_log2e) == 52);
    assert(offsetof(SageAttention2TilingData, v_scale_max) == 56);
    assert(offsetof(SageAttention2TilingData, s_fp8_offset) == 60);

    // Verify Tiling/blocking fields offsets
    assert(offsetof(SageAttention2TilingData, m_per_core) == 64);
    assert(offsetof(SageAttention2TilingData, block_n) == 68);
    assert(offsetof(SageAttention2TilingData, k_step) == 72);

    // Verify Core splitting fields offsets
    assert(offsetof(SageAttention2TilingData, core_num_used) == 76);
    assert(offsetof(SageAttention2TilingData, q_blk_num) == 80);
    assert(offsetof(SageAttention2TilingData, kv_blk_num) == 84);
    assert(offsetof(SageAttention2TilingData, split_kv_num) == 88);
    assert(offsetof(SageAttention2TilingData, is_merge) == 92);

    // Verify Layout & offset fields
    assert(offsetof(SageAttention2TilingData, l1_q_offset) == 96);
    assert(offsetof(SageAttention2TilingData, l1_q_scale_offset) == 100);
    assert(offsetof(SageAttention2TilingData, l1_kv_offset) == 104);
    assert(offsetof(SageAttention2TilingData, l1_kv_scale_offset) == 108);
    assert(offsetof(SageAttention2TilingData, ub_acc_offset) == 112);
    assert(offsetof(SageAttention2TilingData, ub_softmax_offset) == 116);
    assert(offsetof(SageAttention2TilingData, ub_quant_offset) == 120);
    assert(offsetof(SageAttention2TilingData, workspace_lse_offset) == 124);
    assert(offsetof(SageAttention2TilingData, workspace_vm_offset) == 132);
    assert(offsetof(SageAttention2TilingData, enable_die_affinity) == 140);
    assert(offsetof(SageAttention2TilingData, die_num) == 144);
    assert(offsetof(SageAttention2TilingData, cores_per_die) == 148);
    assert(offsetof(SageAttention2TilingData, tasks_per_die) == 152);
    assert(offsetof(SageAttention2TilingData, reserved) == 157 /* input_dtype 占 156 */);

    // Capacity constants
    assert(L1_TOTAL_SIZE == 512 * 1024);
    assert(L1_BANK_SIZE == 256 * 1024);
    assert(UB_TOTAL_SIZE_PER_V == 256 * 1024);
    assert(L0A_SIZE == 64 * 1024);
    assert(L0B_SIZE == 64 * 1024);
    assert(L0C_SIZE == 256 * 1024);

    std::cout << "[Test 1 Passed] All TilingData offsets and constants match specification!" << std::endl;
}

void TestOnlineSoftmax() {
    std::cout << "[Test 2] Testing Online Softmax state transitions and skip-scale..." << std::endl;

    OnlineSoftmaxState state;
    state.Reset();

    // Step 1: Initial block with max 10.0, sum 100.0
    float rescale1 = OnlineSoftmaxTransformer::Step(state, 10.0f, 100.0f);
    assert(rescale1 == 1.0f);
    assert(state.m_run == 10.0f);
    assert(state.m_scale == 10.0f);
    assert(!state.need_rescale);

    // Step 2: Block with small increase: max 12.0 (delta = 2.0 < tau=6.0)
    // Should trigger skip-scale (need_rescale = false, rescale factor = 1.0)
    float rescale2 = OnlineSoftmaxTransformer::Step(state, 12.0f, 50.0f);
    assert(rescale2 == 1.0f);
    assert(state.m_run == 12.0f);
    assert(state.m_scale == 10.0f); // m_scale unchanged due to skip-scale!
    assert(!state.need_rescale);

    // Step 3: Block with large increase: max 20.0 (delta = 20.0 - 10.0 = 10.0 >= tau=6.0)
    // Should trigger full rescale!
    float rescale3 = OnlineSoftmaxTransformer::Step(state, 20.0f, 80.0f);
    assert(state.need_rescale);
    assert(state.m_run == 20.0f);
    assert(state.m_scale == 20.0f); // m_scale updated!
    // Expected rescale factor is 2^(m_scale_old - m_new) = 2^(10.0 - 20.0) = 2^(-10) = 1.0 / 1024.0
    float expected_rescale = std::exp2(10.0f - 20.0f);
    assert(std::fabs(rescale3 - expected_rescale) < 1e-5f);

    // Test LSE conversion
    float sm_scale = 1.0f / 8.0f; // sqrt(64)
    float lse_corr = 0.5f;
    float lse_host_smooth = OnlineSoftmaxTransformer::ConvertLseToHost(state.lse, sm_scale, lse_corr, true);
    float lse_host_no_smooth = OnlineSoftmaxTransformer::ConvertLseToHost(state.lse, sm_scale, lse_corr, false);
    assert(lse_host_smooth > lse_host_no_smooth);
    assert(std::fabs((lse_host_smooth - lse_host_no_smooth) - (lse_corr * sm_scale)) < 1e-6f);

    std::cout << "[Test 2 Passed] Online Softmax and skip-scale behavior verified!" << std::endl;
}

// 清理记录（2026-10-03 R5 收敛清扫随动）：DeclaredReferenceSoftmax 已按"零调用
// 死代码"自 online_softmax.h 整块删除，本用例随之移除；Split-KV 归并重入 DR 时
// 按 online_softmax.h 清理记录恢复并补单测（FoldAttention 数值覆盖现由
// tests/test_fold_attention.py 的原型级用例承担）。

void TestAccumReduce() {
    std::cout << "[Test 3] Testing AccumReduceController for 3 modes..." << std::endl;

    // Mode 0: FP32
    AccumReduceController ctrl_fp32;
    ctrl_fp32.Init(PV_ACCUM_FP32, 16, 32);
    assert(!ctrl_fp32.ShouldReduce(0));
    // 2026-10-03 R9 随动（对齐交付侧 20ab9175a"fp32+fp32 两级累加独立轨迹"）：
    // fp32 档为单一运行和（twin AccumulatePvRow fp32 轨迹单点，对齐开源 f8f8f32
    // 连续累加），仅末块归约；k_step 周期归约为 fp32+fp32/fp16 两档专属节奏。
    // 原"三档统一周期归约"（DEFECT-N5 首版口径）随真值层演化同步退役；N5 的
    // 设备侧 L0C 块间锚差缩放仍属 Phase-D 登记，与本节奏契约解耦。
    assert(!ctrl_fp32.ShouldReduce(15));
    assert(ctrl_fp32.ShouldReduce(31));   // final block forces reduction
    auto cfg_fp32 = ctrl_fp32.GetFixpipeConfig();
    assert(!cfg_fp32.enable_inline_quant);

    // Mode 1: FP32+FP32
    AccumReduceController ctrl_fp32_fp32;
    ctrl_fp32_fp32.Init(PV_ACCUM_FP32_FP32, 16, 32);
    assert(!ctrl_fp32_fp32.ShouldReduce(0));
    assert(ctrl_fp32_fp32.ShouldReduce(15)); // reduces at k_step
    assert(!ctrl_fp32_fp32.ShouldReduce(16));
    assert(ctrl_fp32_fp32.ShouldReduce(31));  // reduces at last step
    auto cfg_fp32_fp32 = ctrl_fp32_fp32.GetFixpipeConfig();
    assert(!cfg_fp32_fp32.enable_inline_quant);

    // Mode 2: FP32+FP16 (Default acceptance mode)
    AccumReduceController ctrl_fp32_fp16;
    ctrl_fp32_fp16.Init(PV_ACCUM_FP32_FP16, 16, 32);
    assert(!ctrl_fp32_fp16.ShouldReduce(0));
    assert(ctrl_fp32_fp16.ShouldReduce(15));
    assert(ctrl_fp32_fp16.ShouldReduce(31));
    auto cfg_fp32_fp16 = ctrl_fp32_fp16.GetFixpipeConfig();
    assert(cfg_fp32_fp16.enable_inline_quant); // inline quantization enabled
    assert(cfg_fp32_fp16.quant_dst_type == 1);  // FP16 destination

    std::cout << "[Test 3 Passed] AccumReduceController verified for all 3 modes!" << std::endl;
}

void TestCatlassContracts() {
    std::cout << "[Test 4] Testing CATLASS tile and resource contracts..." << std::endl;

    // Test KV slot layout
    using SlotContract = KvSlotLayoutContract<int8_t, 64, 128>;
    assert(SlotContract::DATA_ELEMENTS == 64 * 128);
    assert(SlotContract::DATA_BYTES == 8192);
    assert(SlotContract::SCALE_OFFSET == 8192);
    assert(SlotContract::PER_THREAD_SCALE_COUNT == 4);
    assert(SlotContract::PER_THREAD_SCALE_BYTES == 16);
    assert(SlotContract::PER_THREAD_SCALE_BYTES_ALIGNED == 32);
    assert(SlotContract::TOTAL_SLOT_BYTES_PER_THREAD == 8192 + 32);

    // Test Resource manager budget validation
    assert(SageResourceManager::ValidateBudget(64, 64, 128) == true);
    assert(SageResourceManager::ValidateBudget(128, 128, 128) == true);
    // Invalid budget (too large to fit in L1 Bank 0)
    assert(SageResourceManager::ValidateBudget(4096, 64, 128) == false);

    std::cout << "[Test 4 Passed] CATLASS contracts and resource validation verified!" << std::endl;
}

void TestResourceFootprintLedger() {
    std::cout << "[Test 5] Testing runtime footprint ledger (InspectTilingPlan / RES-01)..." << std::endl;

    // 验收域内全部分块计划必须通过（P-01..P-07 覆盖的组合）：
    // gqa=1 (rows=64), gqa=2 (rows=128, m_per_core=64), gqa>=3 (rows=128, m_per_core=32),
    // block_n 64 (sk<2048) / 128 (sk>=2048), d_pad 64/128
    assert(SageResourceManager::ValidateTilingPlan(64, 64, 128));
    assert(SageResourceManager::ValidateTilingPlan(64, 128, 128));
    assert(SageResourceManager::ValidateTilingPlan(128, 64, 128));
    assert(SageResourceManager::ValidateTilingPlan(128, 128, 128));
    assert(SageResourceManager::ValidateTilingPlan(64, 64, 64));
    assert(SageResourceManager::ValidateTilingPlan(128, 128, 64));

    // 预算边界恰好填满必须放行（= 容量为合法，> 容量才违规）：
    // rows=128, d=128 -> O 累加器 128*128*4 = 64KB == O_ACCUM_BYTES
    assert(SageResourceManager::InspectTilingPlan(128, 128, 128) == FootprintViolation::kNone);

    // 零形状：分块计划本身非法
    assert(SageResourceManager::InspectTilingPlan(0, 64, 128) == FootprintViolation::kInvalidShape);
    assert(SageResourceManager::InspectTilingPlan(64, 0, 128) == FootprintViolation::kInvalidShape);
    assert(SageResourceManager::InspectTilingPlan(64, 64, 0) == FootprintViolation::kInvalidShape);

    // GQA 行溢出：total_rows > kKernelMaxRows(128)（kernel 侧行状态容量契约）
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

    // V0 UB GM 输入槽：block_n*d*2 > 48KB
    assert(SageResourceManager::InspectTilingPlan(64, 200, 128) == FootprintViolation::kUbV0Input);

    // constexpr 可用于编译期断言（分块常量改动的编译门）
    static_assert(SageResourceManager::ValidateTilingPlan(128, 128, 128),
                  "P-01..P-07 acceptance domain must fit on-chip budgets");
    static_assert(SageResourceManager::InspectTilingPlan(256, 64, 128) ==
                  FootprintViolation::kGqaRowOverflow,
                  "row overflow must be caught at compile time when constants hard-code it");

    std::cout << "[Test 5 Passed] Runtime footprint ledger verified!" << std::endl;
}

void TestAmlaFloatMulPow2Int() {
    std::cout << "[Test 6] Testing AMLA FloatMulPow2Int integer addition equivalence (Lemma 1)..." << std::endl;
    // 1. 标准正规数乘 2^n
    float f = 1.5f;
    assert(quant::FloatMulPow2Int(f, 0) == 1.5f);
    assert(quant::FloatMulPow2Int(f, 1) == 3.0f);
    assert(quant::FloatMulPow2Int(f, 2) == 6.0f);
    assert(quant::FloatMulPow2Int(f, -1) == 0.75f);
    assert(quant::FloatMulPow2Int(f, -2) == 0.375f);

    // 2. 负数
    float fn = -2.25f;
    assert(quant::FloatMulPow2Int(fn, 1) == -4.5f);
    assert(quant::FloatMulPow2Int(fn, -1) == -1.125f);

    // 3. 边界值：0.0 恒为 0.0
    assert(quant::FloatMulPow2Int(0.0f, 5) == 0.0f);

    // 4. 下溢与上溢截断
    float tiny = 1e-35f;
    assert(quant::FloatMulPow2Int(tiny, -50) == 0.0f);

    std::cout << "[Test 6 Passed] AMLA FloatMulPow2Int mathematical equivalence verified!" << std::endl;
}

int main() {
    std::cout << "Starting SageAttention2 C++ Contract and Tiling Verification..." << std::endl;
    TestTilingDataLayout();
    TestOnlineSoftmax();
    TestAccumReduce();
    TestCatlassContracts();
    TestResourceFootprintLedger();
    TestAmlaFloatMulPow2Int();
    std::cout << "All C++ contracts and struct layouts PASSED successfully!" << std::endl;
    return 0;
}
