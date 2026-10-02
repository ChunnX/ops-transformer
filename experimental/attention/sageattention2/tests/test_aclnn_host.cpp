/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../aclnn/sage_attn_proto.h"
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
    TEST_CHECK(reinterpret_cast<uintptr_t>(&data.reserved) - base == 136, "reserved offset == 136");

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

// 5. 测试 28 核动态兼容性 (设置 SAGEATTN_NUM_CORES=28)
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

    // 5. 第二段空执行器拦截
    TEST_CHECK(aclnnSageAttention2(nullptr, 0, nullptr, nullptr) == ACLNN_ERR_PARAM_NULLPTR, "Should reject null executor in phase 2");

    std::cout << "[PASS] TestParameterValidation" << std::endl;
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

    std::cout << "==========================================================" << std::endl;
    if (all_passed) {
        std::cout << "[SUCCESS] ALL 6 SAGEATTENTION2 ACLNN HOST TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cerr << "[FAILURE] SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}
