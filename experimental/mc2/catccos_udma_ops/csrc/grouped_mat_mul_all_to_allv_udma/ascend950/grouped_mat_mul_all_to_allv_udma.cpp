/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "grouped_mat_mul_all_to_allv_udma_device.h"
#include "../../all_to_allv_grouped_mat_mul_udma/ascend950/udma_runtime.h"

#include "catccos/dgemm/device/kernel_adapter.hpp"
#include "platform/platform_ascendc.h"

#include <cstdint>
#include <limits>

namespace OpsTransformer::CatccosUdma {
namespace {

using Config = GroupedMatMulAllToAllVUdmaDefaultConfig;
using DeviceOp = Config::Device;
using Kernel = Config::Kernel;

constexpr uint32_t kM0 = 128;
constexpr uint32_t kN0 = 256;
constexpr uint32_t kCommTileM = 64;
constexpr uint32_t kCommInterval = 10;
constexpr uint32_t kCommBlockM = 64;
constexpr uint32_t kCommDataSplit = 16;
constexpr uint32_t kCommNpuSplit = 1;
constexpr uint32_t kWorkspaceStages = 2;
constexpr uint32_t kMaxLocalExperts = 32;

void CheckNpuTensor(const at::Tensor &tensor, const char *name, at::ScalarType dtype)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

at::Tensor AllocateSymmetricTensor(
    const at::Tensor &deviceAnchor, at::IntArrayRef sizes, at::ScalarType dtype)
{
    TORCH_CHECK(deviceAnchor.defined() &&
                    deviceAnchor.device().type() == c10::DeviceType::PrivateUse1,
                "device_anchor must be an NPU tensor");
    const c10::OptionalDeviceGuard guard(deviceAnchor.device());
    TORCH_CHECK(dtype == at::kBFloat16,
                "CatCCOS UDMA symmetric tensors currently support only bfloat16");
    int64_t elements = 1;
    for (int64_t size : sizes) {
        TORCH_CHECK(size >= 0, "symmetric tensor sizes must be non-negative");
        TORCH_CHECK(size == 0 || elements <= std::numeric_limits<int64_t>::max() / size,
                    "symmetric tensor element count overflow");
        elements *= size;
    }
    TORCH_CHECK(elements > 0, "symmetric tensors must contain at least one element");
    TORCH_CHECK(static_cast<uint64_t>(elements) <=
                    std::numeric_limits<size_t>::max() / sizeof(uint16_t),
                "symmetric tensor byte count overflow");
    const size_t bytes = static_cast<size_t>(elements) * sizeof(uint16_t);
    UdmaRuntime &runtime = UdmaRuntime::Instance();
    void *pointer = runtime.AllocateSymmetric(bytes);
    auto deleter = [&runtime](void *allocation) { runtime.FreeSymmetric(allocation); };
    return at::from_blob(
        pointer, sizes, deleter,
        at::TensorOptions().dtype(dtype).device(deviceAnchor.device()));
}

at::Tensor EmptySymmetric(
    const at::Tensor &deviceAnchor, at::IntArrayRef sizes, at::ScalarType dtype)
{
    CheckNpuTensor(deviceAnchor, "device_anchor", at::kByte);
    return AllocateSymmetricTensor(deviceAnchor, sizes, dtype);
}

void LaunchGroupedMatMulAllToAllVUdma(
    const DeviceOp::Params &params, uint32_t blockDim,
    aclrtStream stream, uint64_t fftsAddress)
{
    if (fftsAddress == 0) {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(
            params, nullptr);
    } else {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(
            params, fftsAddress, nullptr);
    }
}

at::Tensor GroupedMatMulAllToAllVUdmaImpl(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &localTokensPerExpert,
    const at::Tensor &globalTokensPerLocalExpert,
    int64_t epSize,
    int64_t expertNum,
    const c10::optional<at::Tensor> &outputOptional)
{
    const c10::OptionalDeviceGuard guard(x.device());
    CheckNpuTensor(x, "x", at::kBFloat16);
    CheckNpuTensor(weight, "weight", at::kBFloat16);
    CheckNpuTensor(localTokensPerExpert, "local_tokens_per_expert", at::kInt);
    CheckNpuTensor(globalTokensPerLocalExpert, "global_tokens_per_local_expert", at::kInt);
    TORCH_CHECK(weight.device() == x.device(), "weight and x must be on the same NPU");
    TORCH_CHECK(localTokensPerExpert.device() == x.device() &&
                    globalTokensPerLocalExpert.device() == x.device(),
                "routing tables and x must be on the same NPU");
    TORCH_CHECK(x.dim() == 2, "x must have shape [m, k]");
    TORCH_CHECK(weight.dim() == 3,
                "weight must have shape [local_expert_num, k, n]");

    UdmaRuntime &runtime = UdmaRuntime::Instance();
    runtime.EnsureInitialized();
    const int64_t worldSize = runtime.WorldSize();
    TORCH_CHECK(epSize == worldSize,
                "TP=1 is required: ep_size must equal UDMA world_size");
    TORCH_CHECK(expertNum > 0 && expertNum % epSize == 0,
                "expert_num must be positive and divisible by ep_size");
    const int64_t localExpertNum = expertNum / epSize;
    TORCH_CHECK(runtime.Rank() < epSize, "UDMA rank must be smaller than ep_size");
    TORCH_CHECK(localExpertNum <= kMaxLocalExperts,
                "local expert count exceeds ", kMaxLocalExperts);

    const int64_t m64 = x.size(0);
    const int64_t k64 = x.size(1);
    const int64_t n64 = weight.size(2);
    TORCH_CHECK(m64 > 0 && k64 > 0 && n64 > 0, "m, k and n must be positive");
    TORCH_CHECK(m64 <= std::numeric_limits<uint32_t>::max() &&
                    k64 <= std::numeric_limits<uint32_t>::max() &&
                    n64 <= std::numeric_limits<uint32_t>::max(),
                "shape exceeds the kernel's uint32 range");
    TORCH_CHECK(weight.size(0) == localExpertNum && weight.size(1) == k64,
                "weight must have shape [expert_num / ep_size, k, n]");
    TORCH_CHECK(localTokensPerExpert.numel() == expertNum,
                "local_tokens_per_expert must contain expert_num elements");
    TORCH_CHECK(globalTokensPerLocalExpert.numel() == worldSize * localExpertNum,
                "global_tokens_per_local_expert must contain world_size * local_expert_num elements");

    const uint32_t blockDim =
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(blockDim > 0, "failed to query Ascend AI-core count");
    TORCH_CHECK((static_cast<uint64_t>(blockDim) * kCommInterval) % worldSize == 0,
                "AI-core count * comm_interval must be divisible by world_size");
    const uint64_t requiredWorkspaceBytes =
        static_cast<uint64_t>(kWorkspaceStages) * blockDim * kCommInterval *
        kM0 * kN0 * sizeof(uint16_t);
    TORCH_CHECK(requiredWorkspaceBytes <= Config::kSymmetricDataBytes,
                "kernel tiling requires ", requiredWorkspaceBytes,
                " symmetric workspace bytes, exceeding ", Config::kSymmetricDataBytes);

    at::Tensor output = outputOptional.has_value()
        ? outputOptional.value()
        : at::empty({m64, n64}, x.options());
    CheckNpuTensor(output, "out", at::kBFloat16);
    TORCH_CHECK(output.device() == x.device(), "out and x must be on the same NPU");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == m64 && output.size(1) == n64,
                "out must have shape [m, n]");
    TORCH_CHECK(output.data_ptr() != x.data_ptr() && output.data_ptr() != weight.data_ptr(),
                "out must not alias x or weight");

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    runtime.RecordStream(stream);
    Catlass::GemmCoord shape{
        static_cast<uint32_t>(m64), static_cast<uint32_t>(n64), static_cast<uint32_t>(k64)};
    Catlass::MatrixCoord commCoreSplit{kCommDataSplit, kCommNpuSplit};
    Catlass::MatrixCoord commBlockShape{kCommBlockM, kN0};
    Catlass::MatrixCoord commTileShape{kCommTileM / 2, kN0};
    DeviceOp::Arguments arguments{
        shape,
        static_cast<uint32_t>(runtime.Rank()),
        static_cast<uint32_t>(worldSize),
        kCommInterval,
        static_cast<uint32_t>(epSize),
        static_cast<uint32_t>(expertNum),
        reinterpret_cast<uint8_t *>(x.data_ptr()),
        reinterpret_cast<uint8_t *>(weight.data_ptr()),
        reinterpret_cast<uint8_t *>(output.data_ptr()),
        reinterpret_cast<uint8_t *>(localTokensPerExpert.data_ptr()),
        reinterpret_cast<uint8_t *>(globalTokensPerLocalExpert.data_ptr()),
        runtime.Workspace(),
        commCoreSplit,
        commBlockShape,
        commTileShape,
    };

    DeviceOp deviceOp;
    const Catlass::Status initializeStatus = deviceOp.Initialize(arguments);
    TORCH_CHECK(initializeStatus == Catlass::Status::kSuccess,
                "failed to initialize the CatCCOS device operator");
    const DeviceOp::Params params = deviceOp.params();
    const uint64_t fftsAddress = runtime.FftsAddress();
    auto launch = [=]() -> int {
        LaunchGroupedMatMulAllToAllVUdma(params, blockDim, stream, fftsAddress);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("GroupedMatMulAllToAllVUdma", launch);
    return output;
}

at::Tensor GroupedMatMulAllToAllVUdmaNpu(
    const at::Tensor &x, const at::Tensor &weight,
    const at::Tensor &localTokensPerExpert,
    const at::Tensor &globalTokensPerLocalExpert,
    int64_t epSize, int64_t expertNum)
{
    return GroupedMatMulAllToAllVUdmaImpl(
        x, weight, localTokensPerExpert, globalTokensPerLocalExpert,
        epSize, expertNum, c10::nullopt);
}

at::Tensor GroupedMatMulAllToAllVUdmaOutNpu(
    const at::Tensor &x, const at::Tensor &weight,
    const at::Tensor &localTokensPerExpert,
    const at::Tensor &globalTokensPerLocalExpert,
    int64_t epSize, int64_t expertNum, const at::Tensor &output)
{
    return GroupedMatMulAllToAllVUdmaImpl(
        x, weight, localTokensPerExpert, globalTokensPerLocalExpert,
        epSize, expertNum, output);
}

at::Tensor GroupedMatMulAllToAllVUdmaMeta(
    const at::Tensor &x, const at::Tensor &weight,
    const at::Tensor &, const at::Tensor &, int64_t epSize, int64_t expertNum)
{
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 3,
                "expected x[m,k] and weight[e,k,n]");
    TORCH_CHECK(epSize > 0 && expertNum > 0 && expertNum % epSize == 0,
                "invalid expert topology");
    return at::empty({x.size(0), weight.size(2)}, x.options());
}

at::Tensor GroupedMatMulAllToAllVUdmaOutMeta(
    const at::Tensor &x, const at::Tensor &weight,
    const at::Tensor &, const at::Tensor &,
    int64_t epSize, int64_t expertNum, const at::Tensor &output)
{
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 3,
                "expected x[m,k] and weight[e,k,n]");
    TORCH_CHECK(epSize > 0 && expertNum > 0 && expertNum % epSize == 0,
                "invalid expert topology");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == x.size(0) &&
                    output.size(1) == weight.size(2),
                "invalid out shape");
    return output;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("_empty_symmetric(Tensor device_anchor, int[] size, ScalarType dtype) -> Tensor");
    m.def("grouped_mat_mul_all_to_allv_udma(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
          "Tensor global_tokens_per_local_expert, int ep_size, int expert_num) -> Tensor");
    m.def("grouped_mat_mul_all_to_allv_udma_out(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
          "Tensor global_tokens_per_local_expert, int ep_size, int expert_num, "
          "Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("_empty_symmetric", EmptySymmetric);
    m.impl("grouped_mat_mul_all_to_allv_udma", GroupedMatMulAllToAllVUdmaNpu);
    m.impl("grouped_mat_mul_all_to_allv_udma_out", GroupedMatMulAllToAllVUdmaOutNpu);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("grouped_mat_mul_all_to_allv_udma", GroupedMatMulAllToAllVUdmaMeta);
    m.impl("grouped_mat_mul_all_to_allv_udma_out", GroupedMatMulAllToAllVUdmaOutMeta);
}

} // namespace OpsTransformer::CatccosUdma
