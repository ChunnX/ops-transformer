// ComputeHostTiling 的 inline 唯一定义与 aclnn 状态码词汇表均在本头链
// （host_tiling.h -> sage_attn_proto.h -> ascendc/tiling.h）：
#include "../aclnn/sage_attn_host_tiling.h"
#include "../sageattn_fp8/ascendc/tiling.h"
#include "../quant/quant_host.h"

#include <iostream>
#include <cmath>
#include <vector>

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAILED] " << msg << " (" << #cond << ") at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

namespace {

// 直测 ComputeHostTiling/TilingData 的公共入口：交付态两段式 C-ABI 唯一面在
// op_api/aclnn_sage_attention2（CANN 运行时承载，无法宿主桩执行）；
// tiling 数值与校验契约以闭式计算核心为唯一直测点。
bool CallHostTiling(uint32_t b, uint32_t sq, uint32_t sk, uint32_t hq, uint32_t hkv,
                    int dtype_mode, uint32_t d_og,
                    double sm_scale, int64_t qk_quant_gran, int64_t pv_accum_dtype,
                    bool is_causal, int64_t tensor_layout, bool smooth_k, bool smooth_v,
                    bool has_lse,
                    sage_attn::SageAttention2TilingData &td, uint64_t &ws)
{
    return sage_attn::ComputeHostTiling(b, sq, sk, hq, hkv, dtype_mode, d_og,
                                        sm_scale, qk_quant_gran, pv_accum_dtype,
                                        is_causal, tensor_layout, smooth_k, smooth_v,
                                        has_lse, td, ws) == ACLNN_SUCCESS;
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

    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.m_per_core) - base == 60, "m_per_core offset == 60");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.block_n) - base == 64, "block_n offset == 64");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.k_step) - base == 68, "k_step offset == 68");

    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.core_num_used) - base == 72, "core_num_used offset == 72");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.q_blk_num) - base == 76, "q_blk_num offset == 76");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.kv_blk_num) - base == 80, "kv_blk_num offset == 80");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.split_kv_num) - base == 84, "split_kv_num offset == 84");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.is_merge) - base == 88, "is_merge offset == 88");

    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.l1_q_offset) - base == 92, "l1_q_offset offset == 92");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.l1_q_scale_offset) - base == 96, "l1_q_scale_offset offset == 96");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.l1_kv_offset) - base == 100, "l1_kv_offset offset == 100");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.l1_kv_scale_offset) - base == 104, "l1_kv_scale_offset offset == 104");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.ub_acc_offset) - base == 108, "ub_acc_offset offset == 108");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.ub_softmax_offset) - base == 112, "ub_softmax_offset offset == 112");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.ub_quant_offset) - base == 116, "ub_quant_offset offset == 116");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.workspace_lse_offset) - base == 120, "workspace_lse_offset offset == 120");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.workspace_vm_offset) - base == 128, "workspace_vm_offset offset == 128");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.enable_die_affinity) - base == 136, "enable_die_affinity offset == 136");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.die_num) - base == 140, "die_num offset == 140");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.cores_per_die) - base == 144, "cores_per_die offset == 144");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.tasks_per_die) - base == 148, "tasks_per_die offset == 148");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.input_dtype) - base == 152, "input_dtype offset == 152");
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.reserved) - base == 153, "reserved offset == 153");

    std::cout << "[PASS] TestTilingDataLayout" << std::endl;
    return true;
}

// 2. 测试标准 GQA 4:1 场景下的 Host Tiling 计算与 Workspace 内存核算
bool TestHostTilingStandardGqa()
{
    std::cout << "[RUN] TestHostTilingStandardGqa..." << std::endl;

    // HND 布局: [B=2, Hq=32, Sq=1024, D=128], KV: [B=2, Hkv=8, Sk=1024, D=128]
    uint64_t ws_size = 0;
    sage_attn::SageAttention2TilingData td;

    TEST_CHECK(CallHostTiling(2, 1024, 1024, 32, 8, /*dtype_mode*/ 1, 128,
                              1.0 / std::sqrt(128.0),
                              sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                              /*is_causal*/ false, sage_attn::LAYOUT_HND,
                              /*smooth_k*/ false, /*smooth_v*/ false,
                              /*has_lse*/ true,
                              td, ws_size),
               "ComputeHostTiling returned error");
    TEST_CHECK(ws_size > 0, "workspaceSize must cover lse temporary");
    TEST_CHECK((ws_size % 512) == 0, "workspaceSize must be 512-byte aligned");

    // 验证 Die 亲和性调度 Tiling 字段
    TEST_CHECK(td.enable_die_affinity == 1, "enable_die_affinity must be 1 for standard GQA");
    TEST_CHECK(td.die_num == 2, "die_num must be 2 for Atlas 950PR");
    TEST_CHECK(td.cores_per_die == 16, "cores_per_die must be 16 (32 / 2)");
    // 约束：gqa=4 → m_per_core=32（gqa·m_per_core<=128 行容量约束）
    // → q_blk = 1024/32 = 32 → tasks = b*hkv*q_blk = 512 → per die 256
    TEST_CHECK(td.m_per_core == 32, "m_per_core must be 32 (gqa=4, 4*32<=128 row capacity)");
    TEST_CHECK(td.tasks_per_die == 256, "tasks_per_die must be 256 (512 / 2)");

    std::cout << "[PASS] TestHostTilingStandardGqa" << std::endl;
    return true;
}

// 3. 测试全精度 fp32 档位且开启 smooth_v 时的 Workspace 与 Tiling
bool TestHostTilingFp32SmoothV()
{
    std::cout << "[RUN] TestHostTilingFp32SmoothV..." << std::endl;

    uint64_t ws_size = 0;
    sage_attn::SageAttention2TilingData td;

    TEST_CHECK(CallHostTiling(2, 512, 512, 16, 16, /*dtype_mode*/ 1, 64,
                              1.0 / 8.0, sage_attn::QUANT_GRAN_PER_WARP, sage_attn::PV_ACCUM_FP32,
                              /*is_causal*/ true, sage_attn::LAYOUT_HND,
                              /*smooth_k*/ false, /*smooth_v*/ true,
                              /*has_lse*/ false,
                              td, ws_size),
               "ComputeHostTiling failed");
    // smooth_v 仅全精度 fp32 档生效：fuse_v_mean 均值空间必须入账
    TEST_CHECK(td.smooth_v == 1, "smooth_v must survive on fp32 accum tier");
    TEST_CHECK(ws_size > 0, "workspace should be allocated for smooth_v");
    TEST_CHECK((ws_size % 512) == 0, "workspace must be 512 aligned");

    std::cout << "[PASS] TestHostTilingFp32SmoothV" << std::endl;
    return true;
}

// 4. 测试任务不足触发 Split-KV 的极端短序列场景
bool TestHostTilingSplitKvTrigger()
{
    std::cout << "[RUN] TestHostTilingSplitKvTrigger..." << std::endl;

    // 单头、极小批次：B=1, Hq=1, Hkv=1, Sq=64, Sk=1024, D=64
    uint64_t ws_size = 0;
    sage_attn::SageAttention2TilingData td;

    TEST_CHECK(CallHostTiling(1, 64, 1024, 1, 1, /*dtype_mode*/ 1, 64,
                              0.125, sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                              /*is_causal*/ false, sage_attn::LAYOUT_HND,
                              /*smooth_k*/ false, /*smooth_v*/ false,
                              /*has_lse*/ false,
                              td, ws_size),
               "ComputeHostTiling failed");
    // Split-KV 使能后需要为 partial O 和 partial LSE 分配显存
    TEST_CHECK(td.split_kv_num > 1, "single short task must trigger Split-KV");
    TEST_CHECK(ws_size > 0, "Split-KV should allocate workspace");

    std::cout << "[PASS] TestHostTilingSplitKvTrigger" << std::endl;
    return true;
}

// 5. 测试 28 核动态支持能力 (设置 SAGEATTN_NUM_CORES=28)
bool TestCoreCompatibility28()
{
    std::cout << "[RUN] TestCoreCompatibility28..." << std::endl;

    setenv("SAGEATTN_NUM_CORES", "28", 1);

    uint64_t ws_size = 0;
    sage_attn::SageAttention2TilingData td;

    TEST_CHECK(CallHostTiling(4, 512, 512, 16, 16, /*dtype_mode*/ 1, 128,
                              0.125, sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                              /*is_causal*/ false, sage_attn::LAYOUT_HND,
                              /*smooth_k*/ false, /*smooth_v*/ false,
                              /*has_lse*/ false,
                              td, ws_size),
               "ComputeHostTiling failed with 28 cores");
    TEST_CHECK(td.core_num_used <= 28, "core_num_used must respect the 28-core topology");

    unsetenv("SAGEATTN_NUM_CORES");
    std::cout << "[PASS] TestCoreCompatibility28" << std::endl;
    return true;
}

// 6. 测试参数校验与非法入参拦截 (红线规则；校验唯一点在共享的 ComputeHostTiling，
//    Python 边界与交付 op_api 的白名单按同一域拒绝)
bool TestParameterValidation()
{
    std::cout << "[RUN] TestParameterValidation..." << std::endl;

    sage_attn::SageAttention2TilingData td;
    uint64_t ws = 0;

    // 1. GQA 不可整除拦截 (Hq=31, Hkv=8)
    TEST_CHECK(!CallHostTiling(2, 128, 128, 31, 8, 1, 64, 0.1,
                               sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                               false, sage_attn::LAYOUT_HND, false, false, false, td, ws),
               "Should reject non-divisible GQA heads");

    // 2. HeadDim 越界拦截 (D=256 > 128 与 D=0)
    TEST_CHECK(!CallHostTiling(2, 128, 128, 32, 8, 1, 256, 0.1,
                               sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                               false, sage_attn::LAYOUT_HND, false, false, false, td, ws),
               "Should reject D > 128");
    TEST_CHECK(!CallHostTiling(2, 128, 128, 32, 8, 1, 0, 0.1,
                               sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                               false, sage_attn::LAYOUT_HND, false, false, false, td, ws),
               "Should reject D == 0");

    // 3. 非法量化粒度与累加精度
    TEST_CHECK(!CallHostTiling(2, 128, 128, 32, 8, 1, 64, 0.1, 999,
                               sage_attn::PV_ACCUM_FP32_FP16,
                               false, sage_attn::LAYOUT_HND, false, false, false, td, ws),
               "Should reject invalid qkQuantGran");
    TEST_CHECK(!CallHostTiling(2, 128, 128, 32, 8, 1, 64, 0.1,
                               sage_attn::QUANT_GRAN_PER_THREAD, 999,
                               false, sage_attn::LAYOUT_HND, false, false, false, td, ws),
               "Should reject invalid pvAccumDtype");

    // 4. 非法张量布局编码拦截（仅 0=NHD/1=HND 合法，乱码/负值显式报错不静默归一）
    TEST_CHECK(!CallHostTiling(2, 128, 128, 32, 8, 1, 64, 0.1,
                               sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                               false, 7, false, false, false, td, ws),
               "Should reject invalid tensorLayout (7)");
    TEST_CHECK(!CallHostTiling(2, 128, 128, 32, 8, 1, 64, 0.1,
                               sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                               false, -1, false, false, false, td, ws),
               "Should reject invalid tensorLayout (-1)");

    // 5. causal 与 Sq!=Sk 组合拒绝（DEFECT-L-2 负向锁：任务书 §2.2 因果掩码仅当
    //    Sq==Sk 合法，共享点 ComputeHostTiling 一处拒绝——交付 op_api 透传
    //    ACLNN_ERR_PARAM_INVALID，gert TilingImpl 同点映射 GRAPH_FAILED）
    TEST_CHECK(!CallHostTiling(2, 128, 256, 32, 8, 1, 64, 0.1,
                               sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                               true, sage_attn::LAYOUT_HND, false, false, false, td, ws),
               "Should reject causal with Sq != Sk");
    // 非 causal 的 Sq!=Sk 为合法路径（Split-KV 域，参 TestHostTilingSplitKvTrigger），必须保留
    TEST_CHECK(CallHostTiling(2, 128, 256, 32, 8, 1, 64, 0.1,
                              sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                              false, sage_attn::LAYOUT_HND, false, false, false, td, ws),
               "Should retain non-causal with Sq != Sk");

    // 6. sm_scale 冻结语义（三层分治）：默认 1/sqrt(D) 由 Python 桥接（sm_scale=None）
    //    与 gert 路径 TilingImpl（0/缺省）折叠后传入；本函数（闭式 ComputeHostTiling）
    //    不折叠，对直连调用方拒绝 0/负值/非有限值（与 sage_attn_host_tiling.h 检查注释同口径）
    TEST_CHECK(!CallHostTiling(2, 128, 128, 32, 8, 1, 64, 0.0,
                               sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                               false, sage_attn::LAYOUT_HND, false, false, false, td, ws),
               "Should reject sm_scale == 0");
    TEST_CHECK(!CallHostTiling(2, 128, 128, 32, 8, 1, 64, -0.5,
                               sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
                               false, sage_attn::LAYOUT_HND, false, false, false, td, ws),
               "Should reject negative sm_scale");

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
        false,
        td_odd, ws_odd);
    TEST_CHECK(t_ret_odd == ACLNN_SUCCESS, "ComputeHostTiling for odd tasks succeeded");
    TEST_CHECK(td_odd.enable_die_affinity == 0, "Odd task count (35) must disable die affinity for safety");

    // 5. 验证 100% KV reuse 命中对应 Die 的 64MB 本地 L2 Cache (无跨 Die 颠簸)
    // 设定 32 核满配、B=2, Hkv=16 (b * hkv = 32 为偶数), Sq=256, m_per_core=64 (q_blks=4)
    // 总任务数 total_tasks = 2 * 16 * 4 = 128
    const uint64_t l2_total_tasks = 128;
    const uint32_t l2_cores = 32;
    const uint32_t q_blks_test = 4;
    for (uint32_t c = 0; c < l2_cores; ++c) {
        uint64_t task_end = sage_attn::DieTaskEnd(l2_total_tasks, c, l2_cores, true);
        // Die 半分映射（ComputeHostTiling enable_die 判据的核内 DieTask 契约）：
        // Die 0 = 前 cores/2 核，Die 1 = 后 cores/2 核
        const uint32_t core_die = (c < l2_cores / 2) ? 0u : 1u;
        for (uint64_t vtask = c; vtask < task_end; vtask += l2_cores) {
            uint64_t mapped = sage_attn::DieTask(vtask, l2_total_tasks, l2_cores);
            uint32_t kv_head_idx = static_cast<uint32_t>(mapped / q_blks_test);
            uint32_t expected_die = (kv_head_idx < 16) ? 0 : 1;
            TEST_CHECK(core_die == expected_die, "KV head must strictly reside on the core's local Die L2 cache!");
            if (core_die == 0) {
                TEST_CHECK(kv_head_idx < 16, "Die 0 core must only access first half of KV heads [0..15]");
            } else {
                TEST_CHECK(kv_head_idx >= 16 && kv_head_idx < 32, "Die 1 core must only access second half of KV heads [16..31]");
            }
        }
    }

    // 6. 验证 Auto-Tuned Tiling 经验调优查询与闭式推导平滑回退
    // (1) 查询 P-01 (B=1, Hq=32, Hkv=32, Sq=4096, Sk=4096, D=128, fp16, non-causal)
    sage_attn::SageAttention2TilingData td_p01;
    uint64_t ws_p01 = 0;
    aclnnStatus ret_p01 = sage_attn::ComputeHostTiling(
        1, 4096, 4096, 32, 32, /*dtype_mode*/ 1, 128,
        0.125, sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
        false, sage_attn::LAYOUT_HND, false, false,
        false, td_p01, ws_p01);
    TEST_CHECK(ret_p01 == ACLNN_SUCCESS, "P-01 ComputeHostTiling succeeded");
    TEST_CHECK(td_p01.m_per_core == 128, "P-01 tuned m_per_core must be 128");
    TEST_CHECK(td_p01.block_n == 128, "P-01 tuned block_n must be 128");
    TEST_CHECK(td_p01.k_step == 8 || td_p01.k_step == 16, "P-01 tuned k_step must be 8 or 16");
    TEST_CHECK(td_p01.enable_die_affinity == 1, "P-01 use_die_grouped must be 1");

    // (2) 查询 P-03 (B=1, Hq=32, Hkv=32, Sq=8192, Sk=8192, D=128, fp16, causal=true)
    sage_attn::SageAttention2TilingData td_p03;
    uint64_t ws_p03 = 0;
    aclnnStatus ret_p03 = sage_attn::ComputeHostTiling(
        1, 8192, 8192, 32, 32, /*dtype_mode*/ 1, 128,
        0.125, sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
        true, sage_attn::LAYOUT_HND, false, false,
        false, td_p03, ws_p03);
    TEST_CHECK(ret_p03 == ACLNN_SUCCESS, "P-03 ComputeHostTiling succeeded");
    TEST_CHECK(td_p03.m_per_core == 128, "P-03 tuned m_per_core must be 128");
    TEST_CHECK(td_p03.block_n == 128, "P-03 tuned block_n must be 128");
    TEST_CHECK(td_p03.k_step == 8 || td_p03.k_step == 16, "P-03 tuned k_step must be 8 or 16");
    TEST_CHECK(td_p03.enable_die_affinity == 1, "P-03 use_die_grouped must be 1");

    // (3) 验证动态非调优 shape 的闭式推导平滑回退
    sage_attn::SageAttention2TilingData td_dyn;
    uint64_t ws_dyn = 0;
    aclnnStatus ret_dyn = sage_attn::ComputeHostTiling(
        3, 1000, 1500, 16, 16, /*dtype_mode*/ 1, 128,
        0.125, sage_attn::QUANT_GRAN_PER_THREAD, sage_attn::PV_ACCUM_FP32_FP16,
        false, sage_attn::LAYOUT_HND, false, false,
        false, td_dyn, ws_dyn);
    TEST_CHECK(ret_dyn == ACLNN_SUCCESS, "Dynamic shape ComputeHostTiling succeeded");
    TEST_CHECK(td_dyn.m_per_core == 64, "Dynamic shape fallback m_per_core must be 64");
    TEST_CHECK(td_dyn.block_n == 64, "Dynamic shape (sk=1500 < 2048) fallback block_n must be 64");
    TEST_CHECK(td_dyn.k_step == 16, "Dynamic shape fallback k_step must be default 16");

    // (4) 验证辅助函数子块已随 2026-10-04 死代码清扫移除：UseDieGroups /
    // GetDieIndexForCore / GetKvHeadDieAffinity 为零生产调用方辅助（enable_die
    // 判据唯一定义在 ComputeHostTiling 第 10 步内联实现），删除，本用例随之移除。

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
