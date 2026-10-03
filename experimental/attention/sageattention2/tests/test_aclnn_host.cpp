#include "../aclnn/sage_attn_proto.h"
// ComputeHostTiling 的 inline 唯一定义在本头（proto.h 仅有声明）：
// 本 TU 直接调用它，缺此 include 会在链接期报 undefined reference（R7 修正）
#include "../aclnn/sage_attn_host_tiling.h"
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../quant/quant_host.h"

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <cstdlib>

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAILED] " << msg << " (" << #cond << ") at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

namespace {

/**
 * 辅助函数：构造测试用的 aclTensor
 */
aclTensor CreateMockTensor(int64_t d0, int64_t d1, int64_t d2, int64_t d3)
{
    aclTensor t;
    t.shape = {d0, d1, d2, d3};
    t.dtype = 0;
    t.data = reinterpret_cast<void*>(0x1000);
    return t;
}

// 1. 测试 TilingData 结构体大小与内存对齐
bool TestTilingDataLayout()
{
    std::cout << "[RUN] TestTilingDataLayout..." << std::endl;

    TEST_CHECK(sizeof(sage_attn::SageAttention2TilingData) == 160, "TilingData size must be exactly 160 bytes");
    TEST_CHECK(sizeof(sage_attn::SageAttention2TilingData) % 32 == 0, "TilingData size must be multiple of 32");
    TEST_CHECK(alignof(sage_attn::SageAttention2TilingData) == 32, "TilingData must be 32-byte aligned");

    sage_attn::SageAttention2TilingData data;
    auto base = reinterpret_cast<uintptr_t>(&data);

    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.b) - base == 0, "b offset == 0");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.hq) - base == 4, "hq offset == 4");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.hkv) - base == 8, "hkv offset == 8");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.sq) - base == 12, "sq offset == 12");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.sk) - base == 16, "sk offset == 16");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.d_og) - base == 20, "d_og offset == 20");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.d_pad) - base == 24, "d_pad offset == 24");

    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.is_causal) - base == 28, "is_causal offset == 28");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.tensor_layout) - base == 32, "tensor_layout offset == 32");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.qk_quant_gran) - base == 36, "qk_quant_gran offset == 36");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.pv_accum_dtype) - base == 40, "pv_accum_dtype offset == 40");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.smooth_k) - base == 44, "smooth_k offset == 44");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.smooth_v) - base == 48, "smooth_v offset == 48");

    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.sm_scale_log2e) - base == 52, "sm_scale_log2e offset == 52");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.v_scale_max) - base == 56, "v_scale_max offset == 56");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.s_fp8_offset) - base == 60, "s_fp8_offset offset == 60");

    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.m_per_core) - base == 64, "m_per_core offset == 64");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.block_n) - base == 68, "block_n offset == 68");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.k_step) - base == 72, "k_step offset == 72");

    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.core_num_used) - base == 76, "core_num_used offset == 76");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.q_blk_num) - base == 80, "q_blk_num offset == 80");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.kv_blk_num) - base == 84, "kv_blk_num offset == 84");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.split_kv_num) - base == 88, "split_kv_num offset == 88");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.is_merge) - base == 92, "is_merge offset == 92");

    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.l1_q_offset) - base == 96, "l1_q_offset offset == 96");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.l1_q_scale_offset) - base == 100, "l1_q_scale_offset offset == 100");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.l1_kv_offset) - base == 104, "l1_kv_offset offset == 104");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.l1_kv_scale_offset) - base == 108, "l1_kv_scale_offset offset == 108");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.ub_acc_offset) - base == 112, "ub_acc_offset offset == 112");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.ub_softmax_offset) - base == 116, "ub_softmax_offset offset == 116");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.ub_quant_offset) - base == 120, "ub_quant_offset offset == 120");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.workspace_lse_offset) - base == 124, "workspace_lse_offset offset == 124");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.workspace_vm_offset) - base == 132, "workspace_vm_offset offset == 132");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.enable_die_affinity) - base == 140, "enable_die_affinity offset == 140");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.die_num) - base == 144, "die_num offset == 144");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.cores_per_die) - base == 148, "cores_per_die offset == 148");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.tasks_per_die) - base == 152, "tasks_per_die offset == 152");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.input_dtype) - base == 156, "input_dtype offset == 156");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.reserved) - base == 157, "reserved offset == 157 (dtype 占 156)");

    std::cout << "[PASS] TestTilingDataLayout" << std::endl;
    return true;
}

// 2. 测试标准 GQA 4:1 场景下的 Host Tiling 计算与 Workspace 内存核算
bool TestHostTilingStandardGqa()
{
    std::cout << "[RUN] TestHostTilingStandardGqa..." << std::endl;

    // HND 布局: [B=2, Hq=32, Sq=1024, D=128], KV: [B=2, Hkv=8, Sk=1024, D=128]
    aclTensor q = CreateMockTensor(2, 32, 1024, 128);
    aclTensor k = CreateMockTensor(2, 8, 1024, 128);
    aclTensor v = CreateMockTensor(2, 8, 1024, 128);
    aclTensor out = CreateMockTensor(2, 32, 1024, 128);
    aclTensor lse = CreateMockTensor(2, 32, 1024, 1);

    uint64_t ws_size = 0;
    aclOpExecutor *executor = nullptr;

    double sm_scale = 1.0 / std::sqrt(128.0);
    int64_t qk_quant_gran = sage_attn::QUANT_GRAN_PER_THREAD;
    int64_t pv_accum_dtype = sage_attn::PV_ACCUM_FP32_FP16; // 默认验收档位
    bool is_causal = false;
    int64_t tensor_layout = sage_attn::LAYOUT_HND;
    bool smooth_k = false;
    bool smooth_v = false;

    aclnnStatus ret = aclnnSageAttention2GetWorkspaceSize(
        &q, &k, &v,
        nullptr, nullptr,
        sm_scale, qk_quant_gran, pv_accum_dtype,
        is_causal, tensor_layout, smooth_k, smooth_v,
        &out, &lse,
        &ws_size, &executor);

    TEST_CHECK(ret == ACLNN_SUCCESS, "GetWorkspaceSize returned error");
    TEST_CHECK(executor != nullptr, "executor is nullptr");
    TEST_CHECK((ws_size % 512) == 0, "workspaceSize must be 512-byte aligned");

    // 验证两段式第二阶段
    std::vector<uint8_t> dummy_ws(ws_size > 0 ? ws_size : 512, 0);
    ret = aclnnSageAttention2(dummy_ws.data(), ws_size, executor, nullptr);
    TEST_CHECK(ret == ACLNN_SUCCESS, "aclnnSageAttention2 failed");

    // 验证 Die 亲和性调度 Tiling 字段
    sage_attn::SageAttention2TilingData td;
    uint64_t calculated_ws = 0;
    aclnnStatus t_ret = sage_attn::ComputeHostTiling(
        2, 1024, 1024, 32, 8, /*dtype_mode*/ 1, 128,
        sm_scale, qk_quant_gran, pv_accum_dtype,
        is_causal, tensor_layout, smooth_k, smooth_v,
        false, true,
        td, calculated_ws);
    TEST_CHECK(t_ret == ACLNN_SUCCESS, "ComputeHostTiling failed");
    TEST_CHECK(td.enable_die_affinity == 1, "enable_die_affinity must be 1 for standard GQA");
    TEST_CHECK(td.die_num == 2, "die_num must be 2 for Atlas 950PR");
    TEST_CHECK(td.cores_per_die == 16, "cores_per_die must be 16 (32 / 2)");
    // 约束：gqa=4 → m_per_core=32（gqa·m_per_core<=128 行容量约束）
    // → q_blk = 1024/32 = 32 → tasks = b*hkv*q_blk = 512 → per die 256
    TEST_CHECK(td.m_per_core == 32, "m_per_core must be 32 (gqa=4, 4*32<=128 row capacity)");
    TEST_CHECK(td.tasks_per_die == 256, "tasks_per_die must be 256 (512 / 2)");

    // 显存过小报错验证
    if (ws_size > 0) {
        ret = aclnnSageAttention2(dummy_ws.data(), ws_size - 1, executor, nullptr);
        TEST_CHECK(ret == ACLNN_ERR_PARAM_INVALID, "aclnnSageAttention2 should reject small workspace");
    }

    delete executor;
    std::cout << "[PASS] TestHostTilingStandardGqa" << std::endl;
    return true;
}

// 3. 测试全精度 fp32 档位且开启 smooth_v 时的 Workspace 与 Tiling
bool TestHostTilingFp32SmoothV()
{
    std::cout << "[RUN] TestHostTilingFp32SmoothV..." << std::endl;

    aclTensor q = CreateMockTensor(2, 16, 512, 64);
    aclTensor k = CreateMockTensor(2, 16, 512, 64);
    aclTensor v = CreateMockTensor(2, 16, 512, 64);
    aclTensor out = CreateMockTensor(2, 16, 512, 64);

    uint64_t ws_size = 0;
    aclOpExecutor *executor = nullptr;

    double sm_scale = 1.0 / 8.0;
    int64_t qk_quant_gran = sage_attn::QUANT_GRAN_PER_WARP;
    int64_t pv_accum_dtype = sage_attn::PV_ACCUM_FP32; // 全精度档
    bool is_causal = true;
    int64_t tensor_layout = sage_attn::LAYOUT_HND;
    bool smooth_k = false;
    bool smooth_v = true; // 启用 smooth_v，触发 fuse_v_mean 空间分配

    aclnnStatus ret = aclnnSageAttention2GetWorkspaceSize(
        &q, &k, &v,
        nullptr, nullptr,
        sm_scale, qk_quant_gran, pv_accum_dtype,
        is_causal, tensor_layout, smooth_k, smooth_v,
        &out, nullptr,
        &ws_size, &executor);

    TEST_CHECK(ret == ACLNN_SUCCESS, "GetWorkspaceSize failed");
    TEST_CHECK(executor != nullptr, "executor is null");
    TEST_CHECK(ws_size > 0, "workspace should be allocated for smooth_v");
    TEST_CHECK((ws_size % 512) == 0, "workspace must be 512 aligned");

    delete executor;
    std::cout << "[PASS] TestHostTilingFp32SmoothV" << std::endl;
    return true;
}

// 4. 测试任务不足触发 Split-KV 的极端短序列场景
bool TestHostTilingSplitKvTrigger()
{
    std::cout << "[RUN] TestHostTilingSplitKvTrigger..." << std::endl;

    // 单头、极小批次：B=1, Hq=1, Hkv=1, Sq=64, Sk=1024, D=64
    aclTensor q = CreateMockTensor(1, 1, 64, 64);
    aclTensor k = CreateMockTensor(1, 1, 1024, 64);
    aclTensor v = CreateMockTensor(1, 1, 1024, 64);
    aclTensor out = CreateMockTensor(1, 1, 64, 64);

    uint64_t ws_size = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus ret = aclnnSageAttention2GetWorkspaceSize(
        &q, &k, &v,
        nullptr, nullptr,
        0.125, sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
        false, sage_attn::LAYOUT_HND, false, false,
        &out, nullptr,
        &ws_size, &executor);

    TEST_CHECK(ret == ACLNN_SUCCESS, "GetWorkspaceSize failed");
    TEST_CHECK(executor != nullptr, "executor is null");
    // Split-KV 使能后需要为 partial O 和 partial LSE 分配显存
    TEST_CHECK(ws_size > 0, "Split-KV should allocate workspace");

    delete executor;
    std::cout << "[PASS] TestHostTilingSplitKvTrigger" << std::endl;
    return true;
}

// 5. 测试 28 核动态支持能力 (设置 SAGEATTN_NUM_CORES=28)
bool TestCoreCompatibility28()
{
    std::cout << "[RUN] TestCoreCompatibility28..." << std::endl;

    setenv("SAGEATTN_NUM_CORES", "28", 1);

    aclTensor q = CreateMockTensor(4, 16, 512, 128);
    aclTensor k = CreateMockTensor(4, 16, 512, 128);
    aclTensor v = CreateMockTensor(4, 16, 512, 128);
    aclTensor out = CreateMockTensor(4, 16, 512, 128);

    uint64_t ws_size = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus ret = aclnnSageAttention2GetWorkspaceSize(
        &q, &k, &v,
        nullptr, nullptr,
        0.125, sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
        false, sage_attn::LAYOUT_HND, false, false,
        &out, nullptr,
        &ws_size, &executor);

    TEST_CHECK(ret == ACLNN_SUCCESS, "GetWorkspaceSize failed with 28 cores");
    delete executor;

    unsetenv("SAGEATTN_NUM_CORES");
    std::cout << "[PASS] TestCoreCompatibility28" << std::endl;
    return true;
}

// 6. 测试参数校验与非法入参拦截 (红线规则)
bool TestParameterValidation()
{
    std::cout << "[RUN] TestParameterValidation..." << std::endl;

    aclTensor q = CreateMockTensor(2, 32, 128, 64);
    aclTensor k = CreateMockTensor(2, 8, 128, 64);
    aclTensor v = CreateMockTensor(2, 8, 128, 64);
    aclTensor out = CreateMockTensor(2, 32, 128, 64);
    uint64_t ws_size = 0;
    aclOpExecutor *executor = nullptr;

    // 1. 空指针入参
    TEST_CHECK(aclnnSageAttention2GetWorkspaceSize(nullptr, &k, &v, nullptr, nullptr, 0.1, 3, 2, false, 1, false, false, &out, nullptr, &ws_size, &executor) == ACLNN_ERR_PARAM_NULLPTR, "Should reject null query");
    TEST_CHECK(aclnnSageAttention2GetWorkspaceSize(&q, &k, &v, nullptr, nullptr, 0.1, 3, 2, false, 1, false, false, &out, nullptr, nullptr, &executor) == ACLNN_ERR_PARAM_NULLPTR, "Should reject null ws_size");
    TEST_CHECK(aclnnSageAttention2GetWorkspaceSize(&q, &k, &v, nullptr, nullptr, 0.1, 3, 2, false, 1, false, false, &out, nullptr, &ws_size, nullptr) == ACLNN_ERR_PARAM_NULLPTR, "Should reject null executor");

    // 2. GQA 不可整除拦截 (Hq=31, Hkv=8)
    aclTensor q_bad_gqa = CreateMockTensor(2, 31, 128, 64);
    TEST_CHECK(aclnnSageAttention2GetWorkspaceSize(&q_bad_gqa, &k, &v, nullptr, nullptr, 0.1, 3, 2, false, 1, false, false, &out, nullptr, &ws_size, &executor) == ACLNN_ERR_PARAM_INVALID, "Should reject non-divisible GQA heads");

    // 3. HeadDim 超过 128 拦截 (D=256)
    aclTensor q_bad_d = CreateMockTensor(2, 32, 128, 256);
    aclTensor k_bad_d = CreateMockTensor(2, 8, 128, 256);
    aclTensor v_bad_d = CreateMockTensor(2, 8, 128, 256);
    TEST_CHECK(aclnnSageAttention2GetWorkspaceSize(&q_bad_d, &k_bad_d, &v_bad_d, nullptr, nullptr, 0.1, 3, 2, false, 1, false, false, &out, nullptr, &ws_size, &executor) == ACLNN_ERR_PARAM_INVALID, "Should reject D > 128");

    // 4. 非法量化粒度与累加精度
    TEST_CHECK(aclnnSageAttention2GetWorkspaceSize(&q, &k, &v, nullptr, nullptr, 0.1, 999, 2, false, 1, false, false, &out, nullptr, &ws_size, &executor) == ACLNN_ERR_PARAM_INVALID, "Should reject invalid qkQuantGran");
    TEST_CHECK(aclnnSageAttention2GetWorkspaceSize(&q, &k, &v, nullptr, nullptr, 0.1, 3, 999, false, 1, false, false, &out, nullptr, &ws_size, &executor) == ACLNN_ERR_PARAM_INVALID, "Should reject invalid pvAccumDtype");

    // 5. 非法张量布局编码拦截（仅 0=NHD/1=HND 合法，乱码/负值显式报错不静默归一）
    TEST_CHECK(aclnnSageAttention2GetWorkspaceSize(&q, &k, &v, nullptr, nullptr, 0.1, 3, 2, false, 7, false, false, &out, nullptr, &ws_size, &executor) == ACLNN_ERR_PARAM_INVALID, "Should reject invalid tensorLayout (7)");
    TEST_CHECK(aclnnSageAttention2GetWorkspaceSize(&q, &k, &v, nullptr, nullptr, 0.1, 3, 2, false, -1, false, false, &out, nullptr, &ws_size, &executor) == ACLNN_ERR_PARAM_INVALID, "Should reject invalid tensorLayout (-1)");

    // 6. 第二段空执行器拦截
    TEST_CHECK(aclnnSageAttention2(nullptr, 0, nullptr, nullptr) == ACLNN_ERR_PARAM_NULLPTR, "Should reject null executor in phase 2");

    std::cout << "[PASS] TestParameterValidation" << std::endl;
    return true;
}

// 7. 测试 Atlas 950PR 双 Die / 8-Group 亲和性多核调度 (DieTask 算法准确性)
bool TestDieAffinityScheduling()
{
    std::cout << "[RUN] TestDieAffinityScheduling..." << std::endl;

    // 1. 验证 32 核满配、总任务数 64 场景 (每 Die 16 核，总任务等分各 32)
    uint64_t total_tasks = 64;
    uint32_t cores = 32;
    std::vector<bool> task_covered(total_tasks, false);

    for (uint32_t c = 0; c < cores; ++c) {
        uint64_t task_end = sage_attn::DieTaskEnd(total_tasks, c, cores, true);
        TEST_CHECK(task_end == c + 64, "Wave count must be 2 for 64 tasks on 32 cores");

        for (uint64_t vtask = c; vtask < task_end; vtask += cores) {
            uint64_t mapped = sage_attn::DieTask(vtask, total_tasks, cores);
            TEST_CHECK(mapped < total_tasks, "Mapped task must be < total_tasks");
            TEST_CHECK(!task_covered[mapped], "Each mapped task must be visited exactly once");
            task_covered[mapped] = true;

            // Die 0 (核 0~15) 必须严格亲和前半部分任务 [0, 32)
            if (c < 16) {
                TEST_CHECK(mapped < 32, "Die 0 core must only execute tasks < 32");
            } else {
                // Die 1 (核 16~31) 必须严格亲和后半部分任务 [32, 64)
                TEST_CHECK(mapped >= 32, "Die 1 core must only execute tasks >= 32");
            }
        }
    }

    for (uint64_t i = 0; i < total_tasks; ++i) {
        TEST_CHECK(task_covered[i], "All tasks must be covered");
    }

    // 2. 验证 partial wave 场景 (总任务 16，核心 32，每 Die 需分 8 任务)
    uint64_t small_total = 16;
    std::vector<bool> small_covered(small_total, false);
    uint32_t active_core_count = 0;

    for (uint32_t c = 0; c < cores; ++c) {
        uint64_t task_end = sage_attn::DieTaskEnd(small_total, c, cores, true);
        if (c < task_end) {
            ++active_core_count;
            uint64_t mapped = sage_attn::DieTask(c, small_total, cores);
            TEST_CHECK(mapped < small_total, "Mapped task must be < 16");
            small_covered[mapped] = true;

            if (c < 16) {
                TEST_CHECK(mapped < 8, "Die 0 must only execute tasks [0, 8)");
            } else {
                TEST_CHECK(mapped >= 8, "Die 1 must only execute tasks [8, 16)");
            }
        }
    }

    TEST_CHECK(active_core_count == 16, "Exactly 16 cores (8 on Die 0, 8 on Die 1) must be active");
    for (uint64_t i = 0; i < small_total; ++i) {
        TEST_CHECK(small_covered[i], "All small tasks must be covered");
    }

    // 3. 验证 28 核动态支持能力 (Atlas 950PR 减配拓扑，每 Die 14 核)
    uint32_t cores_28 = 28;
    uint64_t total_28 = 28;
    for (uint32_t c = 0; c < cores_28; ++c) {
        uint64_t mapped = sage_attn::DieTask(c, total_28, cores_28);
        if (c < 14) {
            TEST_CHECK(mapped < 14, "Die 0 on 28-core must execute [0, 14)");
        } else {
            TEST_CHECK(mapped >= 14 && mapped < 28, "Die 1 on 28-core must execute [14, 28)");
        }
    }

    // 4. 验证奇数任务数与 Split-KV 场景下的平滑降级 (关闭 Die 亲和性，避免任务遗漏)
    sage_attn::SageAttention2TilingData td_odd;
    uint64_t ws_odd = 0;
    // B=1, Hq=35, Hkv=1, Sq=128 (total_tasks = 1 * 1 * 1 = 1 < 2 或其它奇数场景)
    // 构造 total_tasks = 35 (B=1, Hkv=35, Hq=35, Sq=64, M=64 -> q_blk=1, total_tasks=35)
    aclnnStatus t_ret_odd = sage_attn::ComputeHostTiling(
        1, 64, 128, 35, 35, /*dtype_mode*/ 1, 128,
        0.125, sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
        false, sage_attn::LAYOUT_HND, false, false,
        false, false,
        td_odd, ws_odd);
    TEST_CHECK(t_ret_odd == ACLNN_SUCCESS, "ComputeHostTiling for odd tasks succeeded");
    TEST_CHECK(td_odd.enable_die_affinity == 0, "Odd task count (35) must disable die affinity for safety");

    std::cout << "[PASS] TestDieAffinityScheduling" << std::endl;
    return true;
}

} // namespace

int main()
{
    std::cout << "==========================================================" << std::endl;
    std::cout << "Starting SageAttention2 ACLNN Host & Tiling Unit Tests..." << std::endl;
    std::cout << "==========================================================" << std::endl;

    bool all_passed = true;
    all_passed &= TestTilingDataLayout();
    all_passed &= TestHostTilingStandardGqa();
    all_passed &= TestHostTilingFp32SmoothV();
    all_passed &= TestHostTilingSplitKvTrigger();
    all_passed &= TestCoreCompatibility28();
    all_passed &= TestParameterValidation();
    all_passed &= TestDieAffinityScheduling();

    std::cout << "==========================================================" << std::endl;
    if (all_passed) {
        std::cout << "[SUCCESS] ALL 7 SAGEATTENTION2 ACLNN HOST TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cerr << "[FAILURE] SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}
