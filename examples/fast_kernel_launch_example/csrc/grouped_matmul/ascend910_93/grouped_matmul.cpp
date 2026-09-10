/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul.cpp
 * \brief
 */
#include "grouped_matmul_tiling_common.h"
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "platform/platform_ascendc.h"
#include <type_traits>

#include "op_kernel/grouped_matmul_kernel.h"

namespace ascend_ops {
namespace GroupedMatmul {

namespace {
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_X = {at::kBFloat16};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_WEIGHT = {at::kBFloat16};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_BIAS = {at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_SCALE = {at::kLong};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_OFFSET = {at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_ANTIQUANTSCALE = {at::kHalf};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_ANTIQUANTOFFSET = {at::kHalf};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_GROUPLIST = {at::kLong};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_PERTOKENSCALE = {at::kFloat};
const std::vector<c10::ScalarType> SUPPORTED_DTYPE_OUTPUT = {at::kBFloat16};

const std::vector<TypeCombo> &getSupportedCombos()
{
    static const auto combos = TypeComboManager::createCombosFromLists(
        SUPPORTED_DTYPE_X, SUPPORTED_DTYPE_BIAS, SUPPORTED_DTYPE_SCALE, SUPPORTED_DTYPE_OFFSET,
        SUPPORTED_DTYPE_ANTIQUANTSCALE, SUPPORTED_DTYPE_ANTIQUANTOFFSET, SUPPORTED_DTYPE_GROUPLIST,
        SUPPORTED_DTYPE_PERTOKENSCALE, SUPPORTED_DTYPE_WEIGHT, SUPPORTED_DTYPE_OUTPUT);
    return combos;
}
} // namespace

// 持久 cycle buffer（固定 GM 地址，capture 时地址固化进 kernel 参数，replay 模式必需）
static torch::Tensor g_gmm_cycle_tensor;  // [64*3] int64, NPU device
static int64_t g_gmm_cycle_launch_count = 0;

// Register the operator's schema
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("grouped_matmul(Tensor[] x, Tensor[] weight, Tensor[]? bias, Tensor[]? scale, Tensor[]? offset, Tensor[]? "
          "antiquantScale, Tensor[]? antiquantOffset, Tensor? groupList, Tensor[]? perTokenScale, int splitItem, int "
          "groupType, int groupListType, int actType,int[]? tuningConfigOptional) -> Tensor");
    // gemm 头开销打点读取接口（与 add 的 read_cycle 分开命名，避免 schema 冲突）
    m.def("gmm_read_cycle(Tensor dummy) -> Tensor");
    m.def("gmm_read_cycle_launch_count(Tensor dummy) -> int");
    m.def("gmm_reset_cycle(Tensor dummy) -> ()");
}

__global__ __aicore__ __cube__ void groupedmatmul_kernel(__gm__ uint8_t *x, __gm__ uint8_t *weight, __gm__ uint8_t *bias,
                                                __gm__ uint8_t *scale, __gm__ uint8_t *offset,
                                                __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                                __gm__ uint8_t *groupList, __gm__ uint8_t *perTokenScale,
                                                __gm__ uint8_t *y, __gm__ uint8_t *workspace,
                                                GroupedMatmulTilingData tilingData, GM_ADDR cycleData)
{
    // T0: kernel 第一行
    int64_t T0 = AscendC::GetSystemCycle();
    GroupedMatmulKernelImpl<GMM_TPL_BF16, GMM_TPL_BF16, GMM_TPL_BF16, 0, 0, 1, 0, 0, 0, 0, 0>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, y, workspace,
        &tilingData, T0, cycleData);
}

void groupedmatmul_api(const TypeCombo &matchedCombo, int comboIndex, aclrtStream stream, const at::TensorList &x,
                       const at::TensorList &weight, const c10::optional<at::TensorList> &bias,
                       const c10::optional<at::TensorList> &scale, const c10::optional<at::TensorList> &offset,
                       const c10::optional<at::TensorList> &antiquantScale,
                       const c10::optional<at::TensorList> &antiquantOffset,
                       const c10::optional<torch::Tensor> &groupList,
                       const c10::optional<at::TensorList> &perTokenScale, const at::TensorList &y,
                       const int64_t splitItem, const int64_t groupType, const int64_t groupListType,
                       const int64_t actType, const vector<int64_t> *tuningConfigOptional)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizePlatFrom;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatFrom);
    GroupedMatmulTilingData tilingData;
    GroupedMatmulNs::GroupedMatmulTiling::GroupedMatmulCommonTiling<at::TensorList, c10::optional<at::TensorList>,
                                                                    c10::optional<torch::Tensor>>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, tilingData,
        ascendcPlatform->GetCoreNumAic(), ubSizePlatFrom);
    uint32_t blockDim = 20; tilingData.gmmBaseParams.coreNum = 20;
    auto x_ptr = get_first_tensor_address<at::TensorList>(matchedCombo.x, x, false);
    auto weight_ptr = get_first_tensor_address<at::TensorList>(matchedCombo.weight, weight, false);
    auto y_ptr = get_first_tensor_address<at::TensorList>(matchedCombo.output, y, false);
    auto bias_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.bias, bias, true);
    auto scale_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.scale, scale, true);
    auto offset_ptr = get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.offset, offset, true);
    auto antiquantScale_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.antiquantScale, antiquantScale, true);
    auto antiquantOffset_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.antiquantOffset, antiquantOffset, true);
    auto groupList_ptr =
        get_first_tensor_address<c10::optional<torch::Tensor>>(matchedCombo.groupList, groupList, true);
    auto perTokenScale_ptr =
        get_first_tensor_address<c10::optional<at::TensorList>>(matchedCombo.perTokenScale, perTokenScale, true);
    uint64_t workspaceSize = 16U * 1024U * 1024U;
    void *workspace_ptr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspace_ptr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d\n", ret);
    }
    // 持久 cycle buffer 懒初始化必须已在 RunOpApi 外部完成:
    // torch::zeros 走 NPU dispatch 会再次进入 OpCommand::RunOpApiV2,
    // 其互斥锁不可重入, 在 acl_thread 内调用会死锁 (A3/CANN 8.5.T107 实测)。
    // 初始化已上移至 grouped_matmul_npu 构造 acl_call 之前 (64 blocks × 3 slots: T0/T1/T2)。
    TORCH_CHECK(g_gmm_cycle_tensor.defined(), "gmm cycle buffer not initialized");
    ++g_gmm_cycle_launch_count;
    auto cycle_ptr = (GM_ADDR)g_gmm_cycle_tensor.data_ptr();
    groupedmatmul_kernel<<<blockDim, nullptr, stream>>>(
        (__gm__ uint8_t *)x_ptr, (__gm__ uint8_t *)weight_ptr, (__gm__ uint8_t *)bias_ptr, (__gm__ uint8_t *)scale_ptr,
        (__gm__ uint8_t *)offset_ptr, (__gm__ uint8_t *)antiquantScale_ptr, (__gm__ uint8_t *)antiquantOffset_ptr,
        (__gm__ uint8_t *)groupList_ptr, (__gm__ uint8_t *)perTokenScale_ptr, (__gm__ uint8_t *)y_ptr,
        (__gm__ uint8_t *)workspace_ptr, tilingData, cycle_ptr);

    // Free workspace memory
    if (workspaceSize > 0 && workspace_ptr != nullptr) {
        aclrtFree(workspace_ptr);
        workspace_ptr = nullptr;
    }
}

// Getter: 读取 gemm 持久 cycle buffer（capture/replay 外部调用）
// 返回 [64, 3] int64 CPU tensor，仅 t0>0 的行有效
torch::Tensor gmm_read_cycle(const torch::Tensor &dummy)
{
    TORCH_CHECK(g_gmm_cycle_tensor.defined(), "gmm cycle buffer not initialized, call grouped_matmul first");
    return g_gmm_cycle_tensor.to(torch::kCPU).clone();
}

// Getter: 累计 launch 次数
int64_t gmm_read_cycle_launch_count(const torch::Tensor &dummy)
{
    return g_gmm_cycle_launch_count;
}

// Reset: 清零 cycle buffer 并归零计数
void gmm_reset_cycle(const torch::Tensor &dummy)
{
    TORCH_CHECK(g_gmm_cycle_tensor.defined(), "gmm cycle buffer not initialized, call grouped_matmul first");
    g_gmm_cycle_tensor.zero_();
    g_gmm_cycle_launch_count = 0;
}


torch::Tensor grouped_matmul_npu(
    const torch::TensorList &x, const torch::TensorList &weight, const c10::optional<torch::TensorList> &bias,
    const c10::optional<torch::TensorList> &scale, const c10::optional<torch::TensorList> &offset,
    const c10::optional<torch::TensorList> &antiquantScale, const c10::optional<torch::TensorList> &antiquantOffset,
    const c10::optional<torch::Tensor> &groupList, const c10::optional<torch::TensorList> &perTokenScale,
    const int64_t splitItem, const int64_t groupType, const int64_t groupListType, const int64_t actType,
    const c10::optional<c10::IntArrayRef> &tuningConfigOptional)
{
    // OptionalDeviceGuard 确保后续操作在正确的设备上下文执行
    // 它会记录当前设备状态，执行完作用域代码后自动恢复
    const c10::OptionalDeviceGuard guard(x[0].device());
    checkTensorOnNPU(x, "x", false);
    checkTensorOnNPU(weight, "weight", false);
    checkTensorOnNPU(bias, "bias", true);
    checkTensorOnNPU(scale, "scale", true);
    checkTensorOnNPU(offset, "offset", true);
    checkTensorOnNPU(antiquantScale, "antiquantScale", true);
    checkTensorOnNPU(antiquantOffset, "antiquantOffset", true);
    checkTensorOnNPU(perTokenScale, "perTokenScale", true);
    checkTensorOnNPU(groupList, "groupList", true);

    const auto &SUPPORTED_COMBOS = getSupportedCombos();

    int matched_index = TypeComboManager::findMatchingCombo(SUPPORTED_COMBOS, x, weight, bias, scale, offset,
                                                            antiquantScale, antiquantOffset, groupList, perTokenScale);

    if (matched_index == -1) {
        TORCH_CHECK(false, "no match dtype combo");
    }

    const auto &matched_combo = SUPPORTED_COMBOS[matched_index];
    const c10::ScalarType output_type = matched_combo.output;
    auto shapeX = getTensorListFirstShape(x);
    int64_t m = shapeX[0];
    auto shapeWeight = getTensorListFirstShape(weight);
    int64_t n = shapeWeight[1];

    at::Tensor y = at::empty({m, n},                    // 指定目标shape [M, N]
                             at::dtype(output_type)     // 数据类型
                                 .device(x[0].device()) // 对齐x[0]的设备（CPU/GPU/Ascend）
                                 .layout(x[0].layout()) // 对齐x[0]的内存布局
    );

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    // 持久 cycle buffer 懒初始化（一次分配固定地址，capture 时固化进 kernel 参数）
    // 必须在 RunOpApi/acl_call 外部执行: torch::zeros 走 NPU dispatch 会再次进入
    // OpCommand::RunOpApiV2, 其互斥锁不可重入, 在 acl_thread 内调用会死锁
    if (!g_gmm_cycle_tensor.defined() || g_gmm_cycle_tensor.device() != x[0].device()) {
        auto cycle_options = torch::TensorOptions().dtype(torch::kInt64).device(x[0].device());
        g_gmm_cycle_tensor = torch::zeros({64 * 3}, cycle_options);  // 最多 64 blocks × 3 slots (T0/T1/T2)
    }

    // 实体化 ArrayRef 参数, 避免 acl_thread 异步执行时悬垂引用:
    // pybind 传入的 TensorList/IntArrayRef 底层指向临时数组, 本函数返回后即析构,
    // 而 acl_call 由 acl_thread 异步消费, 队列积压时会访问已释放内存 (SIGSEGV)
    std::vector<at::Tensor> x_vec(x.begin(), x.end());
    std::vector<at::Tensor> weight_vec(weight.begin(), weight.end());
    std::vector<at::Tensor> bias_vec, scale_vec, offset_vec, aq_scale_vec, aq_offset_vec, pts_vec;
    if (bias.has_value()) { bias_vec.assign(bias->begin(), bias->end()); }
    if (scale.has_value()) { scale_vec.assign(scale->begin(), scale->end()); }
    if (offset.has_value()) { offset_vec.assign(offset->begin(), offset->end()); }
    if (antiquantScale.has_value()) { aq_scale_vec.assign(antiquantScale->begin(), antiquantScale->end()); }
    if (antiquantOffset.has_value()) { aq_offset_vec.assign(antiquantOffset->begin(), antiquantOffset->end()); }
    if (perTokenScale.has_value()) { pts_vec.assign(perTokenScale->begin(), perTokenScale->end()); }
    std::vector<int64_t> tuning_vec;
    if (tuningConfigOptional.has_value()) {
        tuning_vec.assign(tuningConfigOptional->begin(), tuningConfigOptional->end());
    }

    auto acl_call = [=, &matched_combo]() -> int {
        c10::optional<at::TensorList> bias_arg, scale_arg, offset_arg, aq_scale_arg, aq_offset_arg, pts_arg;
        if (bias.has_value()) { bias_arg = at::TensorList(bias_vec); }
        if (scale.has_value()) { scale_arg = at::TensorList(scale_vec); }
        if (offset.has_value()) { offset_arg = at::TensorList(offset_vec); }
        if (antiquantScale.has_value()) { aq_scale_arg = at::TensorList(aq_scale_vec); }
        if (antiquantOffset.has_value()) { aq_offset_arg = at::TensorList(aq_offset_vec); }
        if (perTokenScale.has_value()) { pts_arg = at::TensorList(pts_vec); }
        const vector<int64_t> *tuning_config_ptr = tuningConfigOptional.has_value() ? &tuning_vec : nullptr;

        groupedmatmul_api(matched_combo, matched_index, stream, x_vec, weight_vec, bias_arg, scale_arg, offset_arg,
                          aq_scale_arg, aq_offset_arg, groupList, pts_arg, y, splitItem, groupType, groupListType,
                          actType, tuning_config_ptr);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("GroupedMatmul", acl_call);
    return y;
}

// Register the NPU implementation
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("grouped_matmul", grouped_matmul_npu);
    m.impl("gmm_read_cycle", &gmm_read_cycle);
    m.impl("gmm_read_cycle_launch_count", &gmm_read_cycle_launch_count);
    m.impl("gmm_reset_cycle", &gmm_reset_cycle);
}

} // namespace GroupedMatmul
} // namespace ascend_ops
