/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstdint>
#include <limits>

#include "graph/utils/type_utils.h"
#include "mc2_log.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "../../../op_kernel/allto_allv_grouped_mat_mul_shmem_tiling.h"

namespace optiling {
namespace {
using AlltoAllvGmmShmem::AlltoAllvGroupedMatMulShmemTilingData;

constexpr size_t X_INDEX = 0;
constexpr size_t WEIGHT_INDEX = 1;
constexpr size_t CONTEXT_INDEX = 2;
constexpr size_t Y_INDEX = 0;
constexpr size_t EP_ATTR_INDEX = 0;
constexpr size_t SEND_ATTR_INDEX = 1;
constexpr size_t RECV_ATTR_INDEX = 2;
constexpr uint32_t SOURCE_M0_SMALL = 128;
constexpr uint32_t SOURCE_N0_LARGE = 256;
constexpr uint32_t SOURCE_M0_LARGE = 256;
constexpr uint32_t SOURCE_N0_SMALL = 128;
constexpr uint32_t SOURCE_K0 = 256;
constexpr uint32_t SOURCE_L0_K = 64;
constexpr uint32_t SOURCE_COMM_INTERVAL = 10;
constexpr uint32_t SOURCE_COMM_TILE_M = 32;
constexpr uint32_t SOURCE_COMM_BLOCK_M = 64;
constexpr uint64_t SHMEM_HEAP_LIMIT_BYTES = 1004ULL * 1024ULL * 1024ULL;

bool MulOverflow(uint64_t lhs, uint64_t rhs, uint64_t &out)
{
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return true;
    }
    out = lhs * rhs;
    return false;
}

ge::graphStatus BuildMatmulTiling(const gert::TilingContext *context, uint32_t m0, uint32_t n0, uint32_t k,
                                  uint32_t aicCoreNum, TCubeTiling &tiling)
{
    matmul_tiling::MultiCoreMatmulTiling mm;
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BFLOAT16, false);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BFLOAT16, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                matmul_tiling::DataType::DT_BFLOAT16);
    mm.SetBias(false);
    mm.SetDim(aicCoreNum);
    mm.SetShape(m0, n0, k);
    mm.SetOrgShape(m0, n0, k);
    mm.SetBufferSpace(512 * 1024, -1, -1);
    OP_TILING_CHECK(mm.SetFixSplit(m0, n0, SOURCE_L0_K) != 0,
                    OP_LOGE(context->GetNodeName(), "SetFixSplit(%u,%u,%u) failed.", m0, n0, SOURCE_L0_K),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mm.GetTiling(tiling) == -1,
                    OP_LOGE(context->GetNodeName(), "BF16 matmul tiling failed for [%u,%u,%u].", m0, n0, k),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AlltoAllvGroupedMatMulShmemTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr, OP_LOGE("AlltoAllvGroupedMatMulShmem", "context is null."),
                    return ge::GRAPH_FAILED);
    const gert::StorageShape *xShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *weightShape = context->GetInputShape(WEIGHT_INDEX);
    const gert::StorageShape *contextShape = context->GetInputShape(CONTEXT_INDEX);
    const gert::StorageShape *yShape = context->GetOutputShape(Y_INDEX);
    const gert::CompileTimeTensorDesc *xDesc = context->GetInputDesc(X_INDEX);
    const gert::CompileTimeTensorDesc *weightDesc = context->GetInputDesc(WEIGHT_INDEX);
    const gert::CompileTimeTensorDesc *contextDesc = context->GetInputDesc(CONTEXT_INDEX);
    const gert::CompileTimeTensorDesc *yDesc = context->GetOutputDesc(Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, contextShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    OP_TILING_CHECK(xDesc->GetDataType() != ge::DT_BF16 || weightDesc->GetDataType() != ge::DT_BF16 ||
                        yDesc->GetDataType() != ge::DT_BF16 || contextDesc->GetDataType() != ge::DT_UINT8,
                    OP_LOGE(context->GetNodeName(), "Only BF16 x/weight/y and UINT8 borrowed context are supported."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDesc->GetStorageFormat() != ge::FORMAT_ND ||
                        weightDesc->GetStorageFormat() != ge::FORMAT_ND ||
                        contextDesc->GetStorageFormat() != ge::FORMAT_ND ||
                        yDesc->GetStorageFormat() != ge::FORMAT_ND,
                    OP_LOGE(context->GetNodeName(), "All tensors must use ND storage format."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xShape->GetStorageShape().GetDimNum() != 2 ||
                        weightShape->GetStorageShape().GetDimNum() != 3 ||
                        contextShape->GetStorageShape().GetDimNum() != 1 ||
                        yShape->GetStorageShape().GetDimNum() != 2,
                    OP_LOGE(context->GetNodeName(), "Invalid tensor ranks."), return ge::GRAPH_FAILED);

    const int64_t m = xShape->GetStorageShape().GetDim(0);
    const int64_t k = xShape->GetStorageShape().GetDim(1);
    const int64_t experts = weightShape->GetStorageShape().GetDim(0);
    const int64_t weightK = weightShape->GetStorageShape().GetDim(1);
    const int64_t n = weightShape->GetStorageShape().GetDim(2);
    OP_TILING_CHECK(m < 0 || k <= 0 || n <= 0 || experts < 1 ||
                        experts > AlltoAllvGmmShmem::MAX_LOCAL_EXPERTS || weightK != k,
                    OP_LOGE(context->GetNodeName(), "Invalid M/K/N/local expert dimensions."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(m > std::numeric_limits<uint32_t>::max() || k > std::numeric_limits<uint32_t>::max() ||
                        n > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE(context->GetNodeName(), "M/K/N exceed uint32 source scheduler limits."),
                    return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *epWorldSize = attrs->GetAttrPointer<int64_t>(EP_ATTR_INDEX);
    const gert::ContinuousVector *sendCounts = attrs->GetAttrPointer<gert::ContinuousVector>(SEND_ATTR_INDEX);
    const gert::ContinuousVector *recvCounts = attrs->GetAttrPointer<gert::ContinuousVector>(RECV_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendCounts);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvCounts);
    OP_TILING_CHECK(*epWorldSize < 1 || *epWorldSize > AlltoAllvGmmShmem::MAX_EP_WORLD_SIZE,
                    OP_LOGE(context->GetNodeName(), "epWorldSize must be in [1,8]."), return ge::GRAPH_FAILED);
    const size_t countSize = static_cast<size_t>(*epWorldSize * experts);
    OP_TILING_CHECK(sendCounts->GetSize() != countSize || recvCounts->GetSize() != countSize,
                    OP_LOGE(context->GetNodeName(), "Counts length must equal epWorldSize * localExpertNum."),
                    return ge::GRAPH_FAILED);
    const int64_t *sendData = static_cast<const int64_t *>(sendCounts->GetData());
    const int64_t *recvData = static_cast<const int64_t *>(recvCounts->GetData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendData);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvData);
    uint64_t sendTotal = 0;
    uint64_t recvTotal = 0;
    for (size_t i = 0; i < countSize; ++i) {
        OP_TILING_CHECK(sendData[i] < 0 || recvData[i] < 0 ||
                            sendData[i] > std::numeric_limits<uint32_t>::max() ||
                            recvData[i] > std::numeric_limits<uint32_t>::max(),
                        OP_LOGE(context->GetNodeName(), "Counts must be non-negative uint32 values."),
                        return ge::GRAPH_FAILED);
        sendTotal += static_cast<uint64_t>(sendData[i]);
        recvTotal += static_cast<uint64_t>(recvData[i]);
    }
    OP_TILING_CHECK(sendTotal != static_cast<uint64_t>(m),
                    OP_LOGE(context->GetNodeName(), "sum(sendCounts)=%llu must equal M=%ld.",
                            static_cast<unsigned long long>(sendTotal), m),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvTotal > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                    OP_LOGE(context->GetNodeName(), "sum(recvCounts) exceeds int64 output shape limits."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(recvTotal) ||
                        yShape->GetStorageShape().GetDim(1) != n,
                    OP_LOGE(context->GetNodeName(), "gmmY must be [sum(recvCounts), N]."),
                    return ge::GRAPH_FAILED);

    const uint32_t m0 = (n >= SOURCE_N0_LARGE) ? SOURCE_M0_SMALL : SOURCE_M0_LARGE;
    const uint32_t n0 = (n >= SOURCE_N0_LARGE) ? SOURCE_N0_LARGE : SOURCE_N0_SMALL;
    const uint64_t kLoops = (static_cast<uint64_t>(k) + SOURCE_K0 - 1) / SOURCE_K0;
    uint64_t perStagePeerElements = static_cast<uint64_t>(SOURCE_COMM_INTERVAL) * m0 * SOURCE_K0 * kLoops;
    uint64_t requiredStagingBytes = 0;
    OP_TILING_CHECK(MulOverflow(perStagePeerElements, AlltoAllvGmmShmem::WORKSPACE_STAGES *
                                                        static_cast<uint64_t>(*epWorldSize) * sizeof(uint16_t),
                                requiredStagingBytes) ||
                        requiredStagingBytes > AlltoAllvGmmShmem::WORKSPACE_DATA_BYTES,
                    OP_LOGE(context->GetNodeName(), "Fixed 400 MiB symmetric staging window is insufficient."),
                    return ge::GRAPH_FAILED);
    uint64_t inputElements = 0;
    uint64_t inputBytes = 0;
    OP_TILING_CHECK(MulOverflow(static_cast<uint64_t>(m), static_cast<uint64_t>(k), inputElements) ||
                        MulOverflow(inputElements, sizeof(uint16_t), inputBytes) ||
                        inputBytes > SHMEM_HEAP_LIMIT_BYTES - AlltoAllvGmmShmem::WORKSPACE_BYTES,
                    OP_LOGE(context->GetNodeName(), "gmmX plus workspace exceeds the source 1004 MiB SHMEM heap."),
                    return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicCoreNum = platform.GetCoreNumAic();
    const uint32_t aivCoreNum = platform.GetCoreNumAiv();
    OP_TILING_CHECK(aicCoreNum == 0 || aivCoreNum == 0,
                    OP_LOGE(context->GetNodeName(), "AIC/AIV core count must be non-zero."),
                    return ge::GRAPH_FAILED);

    AlltoAllvGroupedMatMulShmemTilingData *tilingData =
        context->GetTilingData<AlltoAllvGroupedMatMulShmemTilingData>();
    OPS_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->m = static_cast<uint32_t>(m);
    tilingData->n = static_cast<uint32_t>(n);
    tilingData->k = static_cast<uint32_t>(k);
    tilingData->epWorldSize = static_cast<uint32_t>(*epWorldSize);
    tilingData->localExpertNum = static_cast<uint32_t>(experts);
    tilingData->aicCoreNum = aicCoreNum;
    tilingData->aivCoreNum = aivCoreNum;
    tilingData->countSize = static_cast<uint32_t>(countSize);
    tilingData->m0 = m0;
    tilingData->n0 = n0;
    tilingData->k0 = SOURCE_K0;
    tilingData->l0K = SOURCE_L0_K;
    tilingData->commInterval = SOURCE_COMM_INTERVAL;
    tilingData->commTileM = SOURCE_COMM_TILE_M;
    tilingData->commBlockM = SOURCE_COMM_BLOCK_M;
    const uint64_t commShapeM = static_cast<uint64_t>(SOURCE_COMM_INTERVAL) * m0;
    uint64_t commLoops = 0;
    for (int64_t rank = 0; rank < *epWorldSize; ++rank) {
        uint64_t sendToRank = 0;
        uint64_t recvFromRank = 0;
        for (int64_t expert = 0; expert < experts; ++expert) {
            const size_t index = static_cast<size_t>(rank * experts + expert);
            sendToRank += static_cast<uint64_t>(sendData[index]);
            recvFromRank += static_cast<uint64_t>(recvData[index]);
        }
        OP_TILING_CHECK(sendToRank > std::numeric_limits<uint32_t>::max() ||
                            recvFromRank > std::numeric_limits<uint32_t>::max(),
                        OP_LOGE(context->GetNodeName(),
                                "Per-peer send/recv totals must fit the source uint32 scheduler."),
                        return ge::GRAPH_FAILED);
        commLoops = std::max(commLoops, (sendToRank + commShapeM - 1) / commShapeM);
        commLoops = std::max(commLoops, (recvFromRank + commShapeM - 1) / commShapeM);
    }
    tilingData->commLoops = static_cast<uint32_t>(commLoops);
    tilingData->recvTotal = recvTotal;
    tilingData->workspaceDataBytes = AlltoAllvGmmShmem::WORKSPACE_DATA_BYTES;
    tilingData->syncMmadOffset = AlltoAllvGmmShmem::SYNC_MMAD_OFFSET;
    tilingData->syncCommOffset = AlltoAllvGmmShmem::SYNC_COMM_OFFSET;
    for (size_t i = 0; i < countSize; ++i) {
        tilingData->sendCounts[i] = static_cast<uint32_t>(sendData[i]);
        tilingData->recvCounts[i] = static_cast<uint32_t>(recvData[i]);
    }
    OP_TILING_CHECK(BuildMatmulTiling(context, m0, n0, static_cast<uint32_t>(k), aicCoreNum,
                                     tilingData->matmulTiling) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "BuildMatmulTiling failed."), return ge::GRAPH_FAILED);

    const uint32_t blockDim = platform.CalcTschBlockDim(aivCoreNum, aicCoreNum, aivCoreNum);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(1UL);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = AlltoAllvGmmShmem::WORKSPACE_BYTES;
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllvGroupedMatMulShmemCompileInfo {};
ge::graphStatus TilingParseForAlltoAllvGroupedMatMulShmem(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(AlltoAllvGroupedMatMulShmem)
    .Tiling(AlltoAllvGroupedMatMulShmemTilingFunc)
    .TilingParse<AlltoAllvGroupedMatMulShmemCompileInfo>(TilingParseForAlltoAllvGroupedMatMulShmem);
} // namespace optiling
