/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "all_to_allv_grouped_mat_mul_udma_device.h"
#include "udma_runtime.h"

#include "catccos/dgemm/device/kernel_adapter.hpp"
#include "catccos/dgemm/moe_route_metadata.hpp"
#include "platform/platform_ascendc.h"

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

namespace OpsTransformer::CatccosUdma {
namespace {

constexpr uint32_t kMaxLocalExperts = 64;
constexpr uint64_t kSymmetricDataBytes =
    200UL * 1024 * 1024 * sizeof(uint16_t);
constexpr uint64_t kMaxUdmaWriteNotifyBytes = 256UL * 1024 * 1024;
constexpr uint32_t kMaxK0 = 256;
static_assert(sizeof(Catccos::DGemm::MoeRouteSegment) == 24,
              "unexpected MoeRouteSegment ABI");
static_assert(sizeof(Catccos::DGemm::MoeRouteMetadata) == 12880,
              "unexpected MoeRouteMetadata ABI");

enum class KernelVariant {
    kM0_128,
    kM0_256,
    kM0_256N0_256,
};

template <class T>
struct TypeTag {
    using type = T;
};

struct TilingProfile {
    KernelVariant kernelVariant;
    uint32_t m0;
    uint32_t n0;
    uint32_t k0;
    uint32_t commInterval;
    uint32_t commTileM;
    uint32_t commBlockM;
    // These two values are retained from the tuning record. The migrated
    // CatCCOS UDMA kernel ABI does not consume either legacy host field.
    uint32_t commNpuSplit;
    uint32_t commDataSplit;
    // Physical pipeline controls introduced by CatCCOS 54522865. Keeping
    // these explicit decouples communication capacity from legacy tiling
    // fields and lets workspace depth/worker pools evolve independently.
    uint32_t stageRows;
    uint32_t workspaceStages;
    uint32_t localCopyTileRows;
    uint32_t mSplit;
    uint32_t exportCoreCount;
    uint32_t selfCopyCores;
};

constexpr TilingProfile kDefaultTiling{
    KernelVariant::kM0_128, 128, 256, 256, 10, 64, 64, 1, 16,
    1280, 2, 32, 1, 0, 1};
constexpr TilingProfile kM131072K2048N1024NonExportTiling{
    KernelVariant::kM0_128, 128, 256, 256, 12, 128, 128, 4, 4,
    1536, 2, 64, 1, 0, 1};
constexpr TilingProfile kM131072K2048N1024ExportTiling{
    KernelVariant::kM0_256N0_256, 256, 256, 256, 8, 32, 32, 1, 16,
    2048, 2, 16, 1, 16, 8};
constexpr TilingProfile kM131072K2048N512NonExportTiling{
    KernelVariant::kM0_256, 256, 128, 256, 6, 128, 128, 4, 4,
    1536, 2, 64, 1, 0, 1};
constexpr TilingProfile kM131072K2048N512ExportTiling{
    KernelVariant::kM0_256N0_256, 256, 256, 256, 12, 32, 32, 1, 16,
    3072, 3, 16, 2, 32, 4};

// Dynamic-tiling compatibility mapping for the two export records above:
//   * MMAD config id 2 is encoded by kM0_256N0_256 and (256, 256, 256).
//   * execution mode is not part of the migrated kernel ABI.
// commNpuSplit/commDataSplit remain recorded for traceability but are not
// consumed by this operator's current host/kernel ABI.

const TilingProfile &SelectTiling(
    uint32_t m, uint32_t k, uint32_t n, uint32_t worldSize, uint32_t expertNum,
    bool outputAllToAllV)
{
    // Every rank participating in one collective launch must select the same
    // communication geometry. In an imbalanced route, received_rows is local
    // to a rank and may differ, so it must not participate in tiling selection.
    // This operator requires the input M/K/N and the export mode to be
    // rank-invariant; those values are therefore safe selection keys.
    const bool isEightRank256ExpertCase =
        worldSize == 8 && expertNum == 256;
    if (isEightRank256ExpertCase && m == 131072 && k == 2048 && n == 1024) {
        return outputAllToAllV
            ? kM131072K2048N1024ExportTiling
            : kM131072K2048N1024NonExportTiling;
    }
    if (isEightRank256ExpertCase && m == 131072 && k == 2048 && n == 512) {
        return outputAllToAllV
            ? kM131072K2048N512ExportTiling
            : kM131072K2048N512NonExportTiling;
    }
    return kDefaultTiling;
}

uint32_t RoundUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

void CheckNpuTensor(const at::Tensor &tensor, const char *name, at::ScalarType dtype)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

bool TensorByteRangesOverlap(const at::Tensor &lhs, const at::Tensor &rhs)
{
    if (lhs.numel() == 0 || rhs.numel() == 0) {
        return false;
    }
    const uintptr_t lhsBegin = reinterpret_cast<uintptr_t>(lhs.data_ptr());
    const uintptr_t rhsBegin = reinterpret_cast<uintptr_t>(rhs.data_ptr());
    const uintptr_t lhsEnd = lhsBegin +
        static_cast<uintptr_t>(lhs.numel()) * lhs.element_size();
    const uintptr_t rhsEnd = rhsBegin +
        static_cast<uintptr_t>(rhs.numel()) * rhs.element_size();
    return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

void CheckNoTensorOverlap(const at::Tensor &written, const char *writtenName,
                          const at::Tensor &read, const char *readName)
{
    TORCH_CHECK(!TensorByteRangesOverlap(written, read), writtenName,
                " must not overlap the storage range of ", readName);
}

void ConfigureUdma(const at::Tensor &deviceAnchor, int64_t rank, int64_t worldSize,
                   const std::string &initMethod, int64_t heapSize)
{
    CheckNpuTensor(deviceAnchor, "device_anchor", at::kByte);
    TORCH_CHECK(worldSize > 0 && worldSize <= 8, "world_size must be in [1, 8]");
    TORCH_CHECK(rank >= 0 && rank < worldSize, "rank must be in [0, world_size)");
    TORCH_CHECK(heapSize > 0, "heap_size must be positive");
    TORCH_CHECK(worldSize <= std::numeric_limits<int32_t>::max() &&
                rank <= std::numeric_limits<int32_t>::max(), "rank topology exceeds int32 range");
    const c10::OptionalDeviceGuard guard(deviceAnchor.device());
    UdmaRuntime::Instance().Initialize(
        static_cast<int32_t>(rank), static_cast<int32_t>(worldSize), initMethod, static_cast<uint64_t>(heapSize));
}

void FinalizeUdma(const at::Tensor &deviceAnchor)
{
    CheckNpuTensor(deviceAnchor, "device_anchor", at::kByte);
    const c10::OptionalDeviceGuard guard(deviceAnchor.device());
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    UdmaRuntime::Instance().Finalize(stream);
}

void PoisonUdmaWorkspace(const at::Tensor &deviceAnchor, int64_t byteValue)
{
    CheckNpuTensor(deviceAnchor, "device_anchor", at::kByte);
    TORCH_CHECK(byteValue >= 0 && byteValue <= 255,
                "workspace poison byte_value must be in [0, 255]");
    const c10::OptionalDeviceGuard guard(deviceAnchor.device());
    UdmaRuntime::Instance().PoisonWorkspace(static_cast<uint8_t>(byteValue));
}

template <class Kernel, class Params>
void LaunchAllToAllVGroupedMatMulUdma(const Params &params, uint32_t blockDim,
                                      aclrtStream stream, uint64_t fftsAddress)
{
    // Match CatCCOS DeviceDGemm::Run: zero selects the adapter overload that
    // does not configure an explicit FFTS synchronization base address.
    if (fftsAddress == 0) {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(params, nullptr);
    } else {
        Catccos::DGemm::Device::KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(
            params, fftsAddress, nullptr);
    }
}

template <class Config>
void InitializeAndLaunch(
    const TilingProfile &tiling,
    const Catlass::GemmCoord &shape,
    uint32_t rank,
    uint32_t worldSize,
    uint32_t epSize,
    uint32_t expertNum,
    uint32_t outputRows,
    uint8_t *x,
    uint8_t *weight,
    uint8_t *output,
    uint8_t *localTokensPerExpert,
    uint8_t *globalTokensPerLocalExpert,
    uint8_t *workspace,
    uint8_t *allToAllVOutput,
    uint8_t *routeMetadata,
    uint32_t blockDim,
    aclrtStream stream,
    uint64_t fftsAddress)
{
    using DeviceOp = typename Config::Device;
    using Kernel = typename Config::Kernel;
    TORCH_CHECK(tiling.m0 == Config::kM0 && tiling.n0 == Config::kN0 &&
                    tiling.k0 == Config::kK0 &&
                    tiling.workspaceStages == Config::kWorkspaceStages &&
                    tiling.mSplit == Config::kMSplit,
                "tiling profile does not match the selected kernel instance");
    Catlass::MatrixCoord commBlockShape{tiling.commBlockM, RoundUp(shape.k(), tiling.k0)};
    Catlass::MatrixCoord commTileShape{tiling.localCopyTileRows, tiling.k0};
    typename DeviceOp::Arguments arguments{
        shape,
        rank,
        worldSize,
        tiling.commInterval,
        epSize,
        expertNum,
        outputRows,
        x,
        weight,
        output,
        localTokensPerExpert,
        globalTokensPerLocalExpert,
        workspace,
        commBlockShape,
        commTileShape,
        allToAllVOutput,
        // Host route inputs supply precomputed prefixes through this pointer.
        // Device-only routes keep it null and use the kernel's fallback,
        // avoiding a synchronous route-table D2H copy.
        routeMetadata,
        tiling.stageRows,
        tiling.exportCoreCount,
        tiling.selfCopyCores,
    };

    DeviceOp deviceOp;
    const Catlass::Status initializeStatus = deviceOp.Initialize(arguments);
    TORCH_CHECK(initializeStatus == Catlass::Status::kSuccess,
                "failed to initialize the CatCCOS device operator");
    const typename DeviceOp::Params params = deviceOp.params();
    auto launch = [=]() -> int {
        LaunchAllToAllVGroupedMatMulUdma<Kernel>(
            params, blockDim, stream, fftsAddress);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("AllToAllVGroupedMatMulUdma", launch);
}

at::Tensor AllToAllVGroupedMatMulUdmaImpl(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &localTokensPerExpert,
    const at::Tensor &globalTokensPerLocalExpert,
    int64_t epSize,
    int64_t expertNum,
    int64_t receivedRows64,
    const c10::optional<at::Tensor> &outputOptional,
    const c10::optional<at::Tensor> &allToAllVOutputOptional,
    const c10::optional<at::Tensor> &routeMetadataOptional)
{
    const c10::OptionalDeviceGuard guard(x.device());
    CheckNpuTensor(x, "x", at::kBFloat16);
    CheckNpuTensor(weight, "weight", at::kBFloat16);
    CheckNpuTensor(localTokensPerExpert, "local_tokens_per_expert", at::kLong);
    CheckNpuTensor(globalTokensPerLocalExpert, "global_tokens_per_local_expert", at::kLong);
    TORCH_CHECK(weight.device() == x.device(), "weight and x must be on the same NPU");
    TORCH_CHECK(localTokensPerExpert.device() == x.device() && globalTokensPerLocalExpert.device() == x.device(),
                "routing tables and x must be on the same NPU");
    TORCH_CHECK(x.dim() == 2, "x must have shape [m, k]");
    TORCH_CHECK(weight.dim() == 3, "weight must have shape [local_expert_num, k, n]");

    UdmaRuntime &runtime = UdmaRuntime::Instance();
    runtime.EnsureInitialized();
    const int64_t worldSize = runtime.WorldSize();
    TORCH_CHECK(epSize == worldSize, "TP=1 is required: ep_size must equal UDMA world_size");
    TORCH_CHECK(expertNum > 0 && expertNum % epSize == 0,
                "expert_num must be positive and divisible by ep_size");
    const int64_t localExpertNum = expertNum / epSize;
    TORCH_CHECK(runtime.Rank() < epSize, "UDMA rank must be smaller than ep_size");
    TORCH_CHECK(localExpertNum <= kMaxLocalExperts, "local expert count exceeds ", kMaxLocalExperts);

    const int64_t m64 = x.size(0);
    const int64_t k64 = x.size(1);
    const int64_t n64 = weight.size(2);
    TORCH_CHECK(m64 > 0 && k64 > 0 && n64 > 0, "m, k and n must be positive");
    TORCH_CHECK(m64 <= std::numeric_limits<uint32_t>::max() &&
                k64 <= std::numeric_limits<uint32_t>::max() - (kMaxK0 - 1) &&
                n64 <= std::numeric_limits<uint32_t>::max(), "shape exceeds the kernel's uint32 range");
    TORCH_CHECK(weight.size(0) == localExpertNum && weight.size(1) == k64,
                "weight must have shape [expert_num / ep_size, k, n]");
    TORCH_CHECK(localTokensPerExpert.numel() == expertNum,
                "local_tokens_per_expert must contain expert_num elements");
    TORCH_CHECK(globalTokensPerLocalExpert.numel() == worldSize * localExpertNum,
                "global_tokens_per_local_expert must contain world_size * local_expert_num elements");
    TORCH_CHECK(receivedRows64 >= 0 && receivedRows64 <= worldSize * m64,
                "received_rows must be in [0, world_size * m]");
    TORCH_CHECK(receivedRows64 <= std::numeric_limits<uint32_t>::max(),
                "received_rows exceeds the kernel's uint32 range");

    const uint32_t m = static_cast<uint32_t>(m64);
    const uint32_t k = static_cast<uint32_t>(k64);
    const uint32_t n = static_cast<uint32_t>(n64);
    const uint32_t receivedRows = static_cast<uint32_t>(receivedRows64);
    const TilingProfile &tiling = SelectTiling(
        m, k, n, static_cast<uint32_t>(worldSize), static_cast<uint32_t>(expertNum),
        allToAllVOutputOptional.has_value());
    TORCH_CHECK((tiling.workspaceStages == 2 || tiling.workspaceStages == 3) &&
                    tiling.stageRows > 0 && tiling.stageRows % tiling.m0 == 0,
                "invalid CatCCOS physical staging profile");
    const uint64_t requiredWorkspaceBytes =
        static_cast<uint64_t>(tiling.workspaceStages) * tiling.stageRows *
        static_cast<uint64_t>(worldSize) * k * sizeof(uint16_t);
    TORCH_CHECK(requiredWorkspaceBytes <= kSymmetricDataBytes,
                "kernel tiling requires ", requiredWorkspaceBytes,
                " symmetric workspace bytes, exceeding ", kSymmetricDataBytes);

    at::Tensor output = outputOptional.has_value()
        ? outputOptional.value()
        : at::empty({receivedRows64, n64}, x.options());
    CheckNpuTensor(output, "out", at::kBFloat16);
    TORCH_CHECK(output.device() == x.device(), "out and x must be on the same NPU");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == receivedRows64 && output.size(1) == n64,
                "out must have shape [received_rows, n]");
    CheckNoTensorOverlap(output, "out", x, "x");
    CheckNoTensorOverlap(output, "out", weight, "weight");
    CheckNoTensorOverlap(output, "out", localTokensPerExpert, "local_tokens_per_expert");
    CheckNoTensorOverlap(output, "out", globalTokensPerLocalExpert,
                         "global_tokens_per_local_expert");

    at::Tensor allToAllVOutput;
    if (allToAllVOutputOptional.has_value()) {
        allToAllVOutput = allToAllVOutputOptional.value();
        CheckNpuTensor(allToAllVOutput, "all_to_allv_out", at::kBFloat16);
        TORCH_CHECK(allToAllVOutput.device() == x.device(),
                    "all_to_allv_out and x must be on the same NPU");
        TORCH_CHECK(allToAllVOutput.dim() == 2 &&
                        allToAllVOutput.size(0) == receivedRows64 &&
                        allToAllVOutput.size(1) == k64,
                    "all_to_allv_out must have shape [received_rows, k]");
        CheckNoTensorOverlap(allToAllVOutput, "all_to_allv_out", x, "x");
        CheckNoTensorOverlap(allToAllVOutput, "all_to_allv_out", weight, "weight");
        CheckNoTensorOverlap(allToAllVOutput, "all_to_allv_out", output, "out");
        CheckNoTensorOverlap(allToAllVOutput, "all_to_allv_out", localTokensPerExpert,
                             "local_tokens_per_expert");
        CheckNoTensorOverlap(allToAllVOutput, "all_to_allv_out", globalTokensPerLocalExpert,
                             "global_tokens_per_local_expert");
    }
    if (routeMetadataOptional.has_value()) {
        const at::Tensor &routeMetadata = routeMetadataOptional.value();
        TORCH_CHECK(allToAllVOutputOptional.has_value(),
                    "route_metadata is only valid when exporting AllToAllV");
        CheckNpuTensor(routeMetadata, "route_metadata", at::kByte);
        TORCH_CHECK(routeMetadata.device() == x.device(),
                    "route_metadata and x must be on the same NPU");
        TORCH_CHECK(routeMetadata.numel() ==
                        static_cast<int64_t>(sizeof(Catccos::DGemm::MoeRouteMetadata)),
                    "route_metadata has an invalid packed size");
        CheckNoTensorOverlap(output, "out", routeMetadata, "route_metadata");
        CheckNoTensorOverlap(allToAllVOutput, "all_to_allv_out", routeMetadata,
                             "route_metadata");
    }
    const uint64_t stageRows = tiling.stageRows == 0
        ? static_cast<uint64_t>(tiling.commInterval) * tiling.m0
        : tiling.stageRows;
    const uint64_t maximumPeerSliceBytes =
        stageRows * static_cast<uint64_t>(k64) * sizeof(uint16_t);
    TORCH_CHECK(
        maximumPeerSliceBytes <= kMaxUdmaWriteNotifyBytes,
        "each staged peer slice must fit one UDMA write-with-notify request; got ",
        maximumPeerSliceBytes, " bytes, maximum ", kMaxUdmaWriteNotifyBytes);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    runtime.RecordStream(stream);

    Catlass::GemmCoord shape{m, n, k};
    const uint32_t blockDim =
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    TORCH_CHECK(blockDim > 0, "failed to query Ascend AI-core count");
    const uint32_t exportCoreCount =
        tiling.exportCoreCount == 0 ? blockDim : tiling.exportCoreCount;
    TORCH_CHECK(exportCoreCount > 0 && exportCoreCount <= blockDim,
                "export_core_count must be in [1, AI-core count]");
    TORCH_CHECK(tiling.selfCopyCores > 0 &&
                    static_cast<uint64_t>(worldSize) + tiling.selfCopyCores - 1 <= blockDim,
                "self_copy_cores exceeds the available AIV worker pool");
    const uint64_t fftsAddress = runtime.FftsAddress();
    auto launchWithConfig = [&](auto configTag) {
        using SelectedConfig = typename decltype(configTag)::type;
        InitializeAndLaunch<SelectedConfig>(
            tiling, shape,
            static_cast<uint32_t>(runtime.Rank()),
            static_cast<uint32_t>(worldSize),
            static_cast<uint32_t>(epSize),
            static_cast<uint32_t>(expertNum),
            receivedRows,
            // UDMA PUT reads its local source directly from ordinary NPU GM.
            reinterpret_cast<uint8_t *>(x.data_ptr()),
            reinterpret_cast<uint8_t *>(weight.data_ptr()),
            reinterpret_cast<uint8_t *>(output.data_ptr()),
            reinterpret_cast<uint8_t *>(localTokensPerExpert.data_ptr()),
            reinterpret_cast<uint8_t *>(globalTokensPerLocalExpert.data_ptr()),
            runtime.Workspace(),
            allToAllVOutputOptional.has_value()
                ? reinterpret_cast<uint8_t *>(allToAllVOutput.data_ptr())
                : nullptr,
            routeMetadataOptional.has_value()
                ? reinterpret_cast<uint8_t *>(routeMetadataOptional.value().data_ptr())
                : nullptr,
            blockDim, stream, fftsAddress);
    };
    auto launchByMSplit = [&](auto mSplit1, auto mSplit2, auto mSplit4) {
        if (tiling.mSplit == 2) {
            launchWithConfig(mSplit2);
        } else if (tiling.mSplit == 4) {
            launchWithConfig(mSplit4);
        } else {
            launchWithConfig(mSplit1);
        }
    };
    if (allToAllVOutputOptional.has_value()) {
        if (tiling.kernelVariant == KernelVariant::kM0_256N0_256) {
            if (tiling.workspaceStages == 3) {
                launchByMSplit(
                    TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 3, 1>>{},
                    TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 3, 2>>{},
                    TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 3, 4>>{});
            } else {
                launchByMSplit(
                    TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 2, 1>>{},
                    TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 2, 2>>{},
                    TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256, true, 2, 4>>{});
            }
        } else if (tiling.kernelVariant == KernelVariant::kM0_256) {
            if (tiling.workspaceStages == 3) {
                launchByMSplit(
                    TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 128, 256, true, 3, 1>>{},
                    TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 128, 256, true, 3, 2>>{},
                    TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 128, 256, true, 3, 4>>{});
            } else {
                launchByMSplit(
                    TypeTag<AllToAllVGroupedMatMulUdmaM0_256ExportConfig>{},
                    TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 128, 256, true, 2, 2>>{},
                    TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<256, 128, 256, true, 2, 4>>{});
            }
        } else if (tiling.workspaceStages == 3) {
            launchByMSplit(
                TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<128, 256, 256, true, 3, 1>>{},
                TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<128, 256, 256, true, 3, 2>>{},
                TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<128, 256, 256, true, 3, 4>>{});
        } else {
            launchByMSplit(
                TypeTag<AllToAllVGroupedMatMulUdmaDefaultExportConfig>{},
                TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<128, 256, 256, true, 2, 2>>{},
                TypeTag<Bf16AllToAllVGroupedMatMulUdmaConfig<128, 256, 256, true, 2, 4>>{});
        }
    } else {
        TORCH_CHECK(tiling.workspaceStages == 2 && tiling.mSplit == 1,
                    "non-export kernel currently supports workspace_stages=2 and m_split=1");
        if (tiling.kernelVariant == KernelVariant::kM0_256N0_256) {
            launchWithConfig(TypeTag<
                Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256>>{});
        } else if (tiling.kernelVariant == KernelVariant::kM0_256) {
            launchWithConfig(TypeTag<AllToAllVGroupedMatMulUdmaM0_256Config>{});
        } else {
            launchWithConfig(TypeTag<AllToAllVGroupedMatMulUdmaDefaultConfig>{});
        }
    }
    return output;
}

at::Tensor AllToAllVGroupedMatMulUdmaNpu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &localTokensPerExpert,
    const at::Tensor &globalTokensPerLocalExpert,
    int64_t epSize,
    int64_t expertNum,
    int64_t receivedRows)
{
    return AllToAllVGroupedMatMulUdmaImpl(
        x, weight, localTokensPerExpert, globalTokensPerLocalExpert,
        epSize, expertNum, receivedRows, c10::nullopt, c10::nullopt,
        c10::nullopt);
}

at::Tensor AllToAllVGroupedMatMulUdmaOutNpu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &localTokensPerExpert,
    const at::Tensor &globalTokensPerLocalExpert,
    int64_t epSize,
    int64_t expertNum,
    int64_t receivedRows,
    const at::Tensor &output)
{
    return AllToAllVGroupedMatMulUdmaImpl(
        x, weight, localTokensPerExpert, globalTokensPerLocalExpert,
        epSize, expertNum, receivedRows, output, c10::nullopt,
        c10::nullopt);
}

std::tuple<at::Tensor, at::Tensor> AllToAllVGroupedMatMulUdmaExportNpu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &localTokensPerExpert,
    const at::Tensor &globalTokensPerLocalExpert,
    int64_t epSize,
    int64_t expertNum,
    int64_t receivedRows,
    const c10::optional<at::Tensor> &routeMetadata,
    const at::Tensor &allToAllVOutput)
{
    at::Tensor output = AllToAllVGroupedMatMulUdmaImpl(
        x, weight, localTokensPerExpert, globalTokensPerLocalExpert,
        epSize, expertNum, receivedRows, c10::nullopt, allToAllVOutput,
        routeMetadata);
    return {output, allToAllVOutput};
}

std::tuple<at::Tensor, at::Tensor> AllToAllVGroupedMatMulUdmaOutExportNpu(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &localTokensPerExpert,
    const at::Tensor &globalTokensPerLocalExpert,
    int64_t epSize,
    int64_t expertNum,
    int64_t receivedRows,
    const c10::optional<at::Tensor> &routeMetadata,
    const at::Tensor &output,
    const at::Tensor &allToAllVOutput)
{
    AllToAllVGroupedMatMulUdmaImpl(
        x, weight, localTokensPerExpert, globalTokensPerLocalExpert,
        epSize, expertNum, receivedRows, output, allToAllVOutput,
        routeMetadata);
    return {output, allToAllVOutput};
}

at::Tensor AllToAllVGroupedMatMulUdmaMeta(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &,
    const at::Tensor &,
    int64_t epSize,
    int64_t expertNum,
    int64_t receivedRows)
{
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 3, "expected x[m,k] and weight[e,k,n]");
    TORCH_CHECK(epSize > 0 && expertNum > 0 && expertNum % epSize == 0, "invalid expert topology");
    TORCH_CHECK(receivedRows >= 0 && receivedRows <= epSize * x.size(0), "invalid received_rows");
    return at::empty({receivedRows, weight.size(2)}, x.options());
}

at::Tensor AllToAllVGroupedMatMulUdmaOutMeta(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &,
    const at::Tensor &,
    int64_t epSize,
    int64_t expertNum,
    int64_t receivedRows,
    const at::Tensor &output)
{
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 3, "expected x[m,k] and weight[e,k,n]");
    TORCH_CHECK(epSize > 0 && expertNum > 0 && expertNum % epSize == 0, "invalid expert topology");
    TORCH_CHECK(receivedRows >= 0 && receivedRows <= epSize * x.size(0), "invalid received_rows");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == receivedRows &&
                output.size(1) == weight.size(2), "invalid out shape");
    return output;
}

std::tuple<at::Tensor, at::Tensor> AllToAllVGroupedMatMulUdmaExportMeta(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &,
    const at::Tensor &,
    int64_t epSize,
    int64_t expertNum,
    int64_t receivedRows,
    const c10::optional<at::Tensor> &,
    const at::Tensor &allToAllVOutput)
{
    TORCH_CHECK(receivedRows >= 0 && receivedRows <= epSize * x.size(0), "invalid received_rows");
    TORCH_CHECK(allToAllVOutput.dim() == 2 && allToAllVOutput.size(0) == receivedRows &&
                allToAllVOutput.size(1) == x.size(1),
                "invalid all_to_allv_out shape");
    return {at::empty({receivedRows, weight.size(2)}, x.options()), allToAllVOutput};
}

std::tuple<at::Tensor, at::Tensor> AllToAllVGroupedMatMulUdmaOutExportMeta(
    const at::Tensor &x,
    const at::Tensor &weight,
    const at::Tensor &,
    const at::Tensor &,
    int64_t epSize,
    int64_t expertNum,
    int64_t receivedRows,
    const c10::optional<at::Tensor> &,
    const at::Tensor &output,
    const at::Tensor &allToAllVOutput)
{
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 3, "expected x[m,k] and weight[e,k,n]");
    TORCH_CHECK(epSize > 0 && expertNum > 0 && expertNum % epSize == 0, "invalid expert topology");
    TORCH_CHECK(receivedRows >= 0 && receivedRows <= epSize * x.size(0), "invalid received_rows");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == receivedRows &&
                output.size(1) == weight.size(2), "invalid out shape");
    TORCH_CHECK(allToAllVOutput.dim() == 2 && allToAllVOutput.size(0) == receivedRows &&
                allToAllVOutput.size(1) == x.size(1),
                "invalid all_to_allv_out shape");
    return {output, allToAllVOutput};
}

} // namespace

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("_configure_udma(Tensor device_anchor, int rank, int world_size, str init_method, int heap_size) -> ()");
    m.def("_finalize_udma(Tensor device_anchor) -> ()");
    m.def("_poison_udma_workspace(Tensor device_anchor, int byte_value) -> ()");
    m.def("all_to_allv_grouped_mat_mul_udma(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
          "Tensor global_tokens_per_local_expert, int ep_size, int expert_num, int received_rows) -> Tensor");
    m.def("all_to_allv_grouped_mat_mul_udma_out(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
          "Tensor global_tokens_per_local_expert, int ep_size, int expert_num, int received_rows, "
          "Tensor(a!) out) -> Tensor(a!)");
    m.def("all_to_allv_grouped_mat_mul_udma_export(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
          "Tensor global_tokens_per_local_expert, int ep_size, int expert_num, int received_rows, "
          "Tensor? route_metadata, Tensor(a!) all_to_allv_out) -> (Tensor, Tensor(a!))");
    m.def("all_to_allv_grouped_mat_mul_udma_out_export(Tensor x, Tensor weight, Tensor local_tokens_per_expert, "
          "Tensor global_tokens_per_local_expert, int ep_size, int expert_num, int received_rows, "
          "Tensor? route_metadata, Tensor(a!) out, Tensor(b!) all_to_allv_out) -> (Tensor(a!), Tensor(b!))");
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("_configure_udma", ConfigureUdma);
    m.impl("_finalize_udma", FinalizeUdma);
    m.impl("_poison_udma_workspace", PoisonUdmaWorkspace);
    m.impl("all_to_allv_grouped_mat_mul_udma", AllToAllVGroupedMatMulUdmaNpu);
    m.impl("all_to_allv_grouped_mat_mul_udma_out", AllToAllVGroupedMatMulUdmaOutNpu);
    m.impl("all_to_allv_grouped_mat_mul_udma_export", AllToAllVGroupedMatMulUdmaExportNpu);
    m.impl("all_to_allv_grouped_mat_mul_udma_out_export", AllToAllVGroupedMatMulUdmaOutExportNpu);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("all_to_allv_grouped_mat_mul_udma", AllToAllVGroupedMatMulUdmaMeta);
    m.impl("all_to_allv_grouped_mat_mul_udma_out", AllToAllVGroupedMatMulUdmaOutMeta);
    m.impl("all_to_allv_grouped_mat_mul_udma_export", AllToAllVGroupedMatMulUdmaExportMeta);
    m.impl("all_to_allv_grouped_mat_mul_udma_out_export", AllToAllVGroupedMatMulUdmaOutExportMeta);
}

} // namespace OpsTransformer::CatccosUdma
