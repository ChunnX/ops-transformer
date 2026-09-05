/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#include "mte_runtime.h"
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "platform/platform_ascendc.h"
#include <limits>
#include <vector>

namespace OpsTransformer::CatccosMte {
namespace {
void CheckAnchor(const at::Tensor &anchor)
{
    TORCH_CHECK(anchor.defined() && anchor.device().type() == c10::DeviceType::PrivateUse1 &&
                anchor.scalar_type() == at::kByte, "device_anchor must be an NPU uint8 tensor");
}

void ConfigureMte(const at::Tensor &anchor, int64_t rank, int64_t worldSize,
                  const std::string &initMethod, int64_t heapSize)
{
    CheckAnchor(anchor);
    TORCH_CHECK(worldSize > 0 && worldSize <= 8 && rank >= 0 && rank < worldSize,
                "invalid MTE rank topology");
    TORCH_CHECK(heapSize > 0, "heap_size must be positive");
    const c10::OptionalDeviceGuard guard(anchor.device());
    MteRuntime::Instance().Initialize(static_cast<int32_t>(rank), static_cast<int32_t>(worldSize),
        initMethod, static_cast<uint64_t>(heapSize), c10_npu::getCurrentNPUStream().stream(false));
}

void FinalizeMte(const at::Tensor &anchor)
{
    CheckAnchor(anchor);
    const c10::OptionalDeviceGuard guard(anchor.device());
    MteRuntime::Instance().Finalize(c10_npu::getCurrentNPUStream().stream(false));
}

at::Tensor EmptySymmetric(const at::Tensor &anchor, at::IntArrayRef sizes, at::ScalarType dtype)
{
    CheckAnchor(anchor);
    const c10::OptionalDeviceGuard guard(anchor.device());
    TORCH_CHECK(dtype == at::kBFloat16, "MTE symmetric tensors support only bfloat16");
    int64_t elements = 1;
    for (int64_t size : sizes) {
        TORCH_CHECK(size > 0 && elements <= std::numeric_limits<int64_t>::max() / size,
                    "symmetric tensor must have positive dimensions without overflow");
        elements *= size;
    }
    TORCH_CHECK(static_cast<uint64_t>(elements) <= std::numeric_limits<size_t>::max() / sizeof(uint16_t),
                "symmetric tensor byte count overflow");
    MteRuntime &runtime = MteRuntime::Instance();
    void *pointer = runtime.AllocateSymmetric(static_cast<size_t>(elements) * sizeof(uint16_t));
    return at::from_blob(pointer, sizes, [&runtime](void *p) { runtime.FreeSymmetric(p); },
                        at::TensorOptions().dtype(dtype).device(anchor.device()));
}

void SetOverride(const at::Tensor &anchor, at::IntArrayRef values)
{
    CheckAnchor(anchor);
    TORCH_CHECK(values.size() == 4, "MTE tuning expects four uint32 values");
    for (auto value : values) {
        TORCH_CHECK(value >= 0 && value <= std::numeric_limits<uint32_t>::max(), "tuning value must fit uint32");
    }
    MteRuntime::Instance().SetTuningOverride({static_cast<uint32_t>(values[0]), static_cast<uint32_t>(values[1]),
                                            static_cast<uint32_t>(values[2]), static_cast<uint32_t>(values[3])});
}

void ClearOverride(const at::Tensor &anchor)
{
    CheckAnchor(anchor);
    MteRuntime::Instance().ClearTuningOverride();
}

std::vector<int64_t> GetConfig(const at::Tensor &anchor)
{
    CheckAnchor(anchor);
    auto config = MteRuntime::Instance().RequestedCommConfig();
    return {config.commAivCount, config.workersPerPeer, config.commChunkBytes, config.commUbStages};
}

std::vector<int64_t> GetPlatform(const at::Tensor &anchor)
{
    CheckAnchor(anchor);
    const c10::OptionalDeviceGuard guard(anchor.device());
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "Ascend platform query is unavailable");
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t blockDim = platform->GetCoreNumAic();
    TORCH_CHECK(blockDim > 0 && ubBytes > 0, "Ascend platform returned invalid AIC/UB capacity");
    return {blockDim, static_cast<int64_t>(ubBytes)};
}
} // namespace
} // namespace OpsTransformer::CatccosMte

TORCH_LIBRARY_FRAGMENT(ops_transformer_catccos_mte, m)
{
    m.def("_configure_mte(Tensor anchor, int rank, int world_size, str init_method, int heap_size) -> ()");
    m.def("_finalize_mte(Tensor anchor) -> ()");
    m.def("_empty_symmetric(Tensor anchor, int[] shape, ScalarType dtype) -> Tensor");
    m.def("_set_mte_tuning_override(Tensor anchor, int[] config) -> ()");
    m.def("_clear_mte_tuning_override(Tensor anchor) -> ()");
    m.def("_get_mte_comm_config(Tensor anchor) -> int[]");
    m.def("_get_mte_platform(Tensor anchor) -> int[]");
}

TORCH_LIBRARY_IMPL(ops_transformer_catccos_mte, PrivateUse1, m)
{
    using namespace OpsTransformer::CatccosMte;
    m.impl("_configure_mte", &ConfigureMte);
    m.impl("_finalize_mte", &FinalizeMte);
    m.impl("_empty_symmetric", &EmptySymmetric);
    m.impl("_set_mte_tuning_override", &SetOverride);
    m.impl("_clear_mte_tuning_override", &ClearOverride);
    m.impl("_get_mte_comm_config", &GetConfig);
    m.impl("_get_mte_platform", &GetPlatform);
}
