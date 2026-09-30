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
 * \file kv_cache_turbo_quant_tiling.cpp
 * \brief Tiling: split rows over the AI cores, tile the two dense fp32 GEMMs, place the workspace.
 */

#include <cstring>
#include <algorithm>
#include <limits>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "lib/tiling_api.h"
#include "kv_cache_turbo_quant_tiling.h"
#include "../op_kernel/kv_cache_turbo_quant_tiling_data.h"

namespace optiling {
using namespace matmul_tiling;

namespace {
constexpr size_t IN_KV_VECTORS = 0;
constexpr size_t IN_ROTATION = 1;
constexpr size_t IN_QJL = 2;
constexpr size_t OUT_QUANT_NORM = 2;
constexpr size_t OUT_QUANT_GAMMA = 3;
constexpr size_t ATTR_MSE_BITS = 0;
constexpr uint32_t ALIGN_ROWS = 16;   // the GEMM tile needs the row count to be 16 aligned
constexpr uint64_t WS_ALIGN = 512;

inline uint32_t CeilAlignU32(uint32_t value, uint32_t align)
{
    return (value + align - 1U) / align * align;
}

inline uint64_t CeilAlignU64(uint64_t value, uint64_t align)
{
    return (value + align - 1U) / align * align;
}

ge::graphStatus CheckMatrixInput(gert::TilingContext *context, size_t index, const char *name)
{
    const gert::StorageShape *shape = context->GetInputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    const gert::Shape &storage = shape->GetStorageShape();
    OP_CHECK_IF(storage.GetDimNum() != 2 || storage.GetDim(0) != KCTQ_HEAD_DIM ||
                    storage.GetDim(1) != KCTQ_HEAD_DIM,
                OP_LOGE(context, "%s must be [%u, %u], got rank %zu", name, KCTQ_HEAD_DIM, KCTQ_HEAD_DIM,
                        storage.GetDimNum()),
                return ge::GRAPH_FAILED);
    const auto *desc = context->GetInputDesc(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    // The task book allows "float32 / bfloat16" for both matrices; the bf16 case is an interface
    // extension that the kernel serves by upcasting once into the workspace.
    OP_CHECK_IF(desc->GetDataType() != ge::DT_FLOAT && desc->GetDataType() != ge::DT_BF16,
                OP_LOGE(context, "%s must be float32 or bfloat16", name), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus KvCacheTurboQuantTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);

    const gert::StorageShape *kvShape = context->GetInputShape(IN_KV_VECTORS);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvShape);
    const gert::Shape &kv = kvShape->GetStorageShape();
    OP_CHECK_IF(kv.GetDimNum() != 3, OP_LOGE(context, "kv_vectors rank must be 3"), return ge::GRAPH_FAILED);
    const int64_t numTokens = kv.GetDim(0);
    const int64_t numHeads = kv.GetDim(1);
    OP_CHECK_IF(kv.GetDim(2) != KCTQ_HEAD_DIM,
                OP_LOGE(context, "head_dim must be %u", KCTQ_HEAD_DIM), return ge::GRAPH_FAILED);
    OP_CHECK_IF(numTokens <= 0 || numHeads <= 0, OP_LOGE(context, "kv_vectors must not be empty"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numHeads < static_cast<int64_t>(KCTQ_MIN_HEADS) || numHeads > static_cast<int64_t>(KCTQ_MAX_HEADS),
                OP_LOGE(context, "num_kv_heads must be in [%u, %u], got %ld", KCTQ_MIN_HEADS, KCTQ_MAX_HEADS,
                        numHeads),
                return ge::GRAPH_FAILED);
    const auto *kvDesc = context->GetInputDesc(IN_KV_VECTORS);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvDesc);
    OP_CHECK_IF(kvDesc->GetDataType() != ge::DT_BF16, OP_LOGE(context, "kv_vectors must be bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckMatrixInput(context, IN_ROTATION, "rotation_matrix") != ge::GRAPH_SUCCESS ||
                    CheckMatrixInput(context, IN_QJL, "qjl_matrix") != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "matrix input check failed"), return ge::GRAPH_FAILED);

    // Interface extensions (task book §2.3 lists "float32 / bfloat16" for the matrices and
    // "float16 / bfloat16" for the metadata): a bf16 matrix is upcast to fp32 once inside the
    // kernel, an fp16 metadata only changes the store dtype. The formal cases use index 0 of the
    // OpDef dtype lists (float32 + bfloat16) and are not affected by either branch.
    const auto *rotDesc = context->GetInputDesc(IN_ROTATION);
    const auto *qjlDesc = context->GetInputDesc(IN_QJL);
    OP_CHECK_NULL_WITH_CONTEXT(context, rotDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjlDesc);
    OP_CHECK_IF(rotDesc->GetDataType() != qjlDesc->GetDataType(),
                OP_LOGE(context, "rotation_matrix and qjl_matrix must share one dtype"),
                return ge::GRAPH_FAILED);
    const bool matrixBf16 = (rotDesc->GetDataType() == ge::DT_BF16);
    const auto *normDesc = context->GetOutputDesc(OUT_QUANT_NORM);
    const auto *gammaDesc = context->GetOutputDesc(OUT_QUANT_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, normDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaDesc);
    // An unspecified (inference only) dtype keeps the formal default: bf16.
    OP_CHECK_IF(normDesc->GetDataType() != gammaDesc->GetDataType(),
                OP_LOGE(context, "quant_norm/quant_gamma must share one dtype"), return ge::GRAPH_FAILED);
    const ge::DataType metaDtype = normDesc->GetDataType();
    OP_CHECK_IF(metaDtype != ge::DT_BF16 && metaDtype != ge::DT_FLOAT16 && metaDtype != ge::DT_UNDEFINED,
                OP_LOGE(context, "quant_norm/quant_gamma must be bfloat16 or float16"),
                return ge::GRAPH_FAILED);
    const bool metadataFp16 = (metaDtype == ge::DT_FLOAT16);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *mseBitsPtr = attrs->GetAttrPointer<int64_t>(ATTR_MSE_BITS);
    const int64_t mseBits = (mseBitsPtr == nullptr) ? 3 : *mseBitsPtr;
    OP_CHECK_IF(mseBits != 2 && mseBits != 3 && mseBits != 4,
                OP_LOGE(context, "mse_bits must be 2, 3 or 4, got %ld", mseBits), return ge::GRAPH_FAILED);

    const uint64_t totalRows = static_cast<uint64_t>(numTokens) * static_cast<uint64_t>(numHeads);

    const int64_t aicNum = platform.GetCoreNumAic();
    OP_CHECK_IF(aicNum <= 0, OP_LOGE(context, "aic core number is zero"), return ge::GRAPH_FAILED);
    // Every row count derived below is a 16 aligned uint32: `usedCoreNum` (<= aicNum) blocks of
    // `rowsPerCore` rows have to fit, and neither the alignment itself nor
    // `rowsPerCore * usedCoreNum` may wrap. The four buffers are indexed with those counts, so a
    // wrap would leave the workspace (alignedRows / wsRows) inconsistent with the rows the kernel
    // walks and the device would write outside the buffer. Reject the request here instead.
    const uint64_t maxRows = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) -
                             static_cast<uint64_t>(ALIGN_ROWS) * static_cast<uint64_t>(aicNum);
    OP_CHECK_IF(totalRows > maxRows,
                OP_LOGE(context, "token*head count %lu exceeds the supported maximum %lu", totalRows, maxRows),
                return ge::GRAPH_FAILED);

    const uint32_t rowUnits = CeilAlignU32(static_cast<uint32_t>(totalRows), ALIGN_ROWS) / ALIGN_ROWS;
    uint32_t usedCoreNum = std::min<uint32_t>(static_cast<uint32_t>(aicNum), rowUnits);
    usedCoreNum = std::max<uint32_t>(usedCoreNum, 1U);
    const uint32_t rowsPerCore =
        CeilAlignU32(static_cast<uint32_t>((totalRows + usedCoreNum - 1U) / usedCoreNum), ALIGN_ROWS);
    const uint32_t alignedRows = rowsPerCore * usedCoreNum;

    // Fill the registered TilingData object (it is not a POD, so it must not be memset).
    KvCacheTurboQuantTilingDef tilingObj;
    KvCacheTurboQuantTilingDef *td = &tilingObj;

    MultiCoreMatmulTiling cubeTiling(platform);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT, true);
    cubeTiling.SetCType(TPosition::LCM, CubeFormat::ND, DataType::DT_FLOAT);
    cubeTiling.SetBias(false);
    // The single core GEMM of the tiling is one *vector* core (an AIV owns rowsPerCore / 2 rows and
    // is the KFC client of its own cube server instance), so the M shape is the AIV's row count:
    // that makes the framework end its M loop on the AIV's last row and hands the partial last tile
    // over as the tail (32 rows on the t2048 case) instead of a whole baseM block.
    const uint32_t rowsPerAiv = rowsPerCore / KCTQ_AIV_PER_AIC;
    cubeTiling.SetShape(static_cast<int32_t>(rowsPerAiv), static_cast<int32_t>(KCTQ_HEAD_DIM),
                        static_cast<int32_t>(KCTQ_HEAD_DIM));
    cubeTiling.SetOrgShape(static_cast<int32_t>(rowsPerAiv), static_cast<int32_t>(KCTQ_HEAD_DIM),
                           static_cast<int32_t>(KCTQ_HEAD_DIM));
    cubeTiling.SetBufferSpace(-1, -1, -1);
    cubeTiling.SetDim(1);
    OP_CHECK_IF(cubeTiling.GetTiling(td->cubeTiling) != 0, OP_LOGE(context, "GetTiling failed"),
                return ge::GRAPH_FAILED);

    td->set_mseBits(static_cast<uint32_t>(mseBits));
    td->set_usedCoreNum(usedCoreNum);
    td->set_totalRows(static_cast<uint32_t>(totalRows));
    td->set_rowsPerCore(rowsPerCore);
    td->set_alignedRows(alignedRows);
    td->set_headDim(KCTQ_HEAD_DIM);
    // The framework applies its M tail to the AIV's last tile, so an AIV slice only needs its own
    // rows: the workspace keeps the plain rowsPerCore / KCTQ_AIV_PER_AIC rows per AIV.
    const uint32_t aivRowStride = rowsPerAiv;
    td->set_aivRowStride(aivRowStride);
    const uint32_t wsRows = usedCoreNum * KCTQ_AIV_PER_AIC * aivRowStride;
    td->set_idxBytesPerRow(KCTQ_HEAD_DIM * static_cast<uint32_t>(mseBits) / 8U);
    td->set_qjlBytesPerRow(KCTQ_HEAD_DIM / 8U);

    // Only two of the four row matrices have to exist at any moment: the four buffers are used in
    // strictly disjoint phases of the pipeline and every phase boundary is a full GEMM drain.
    //   wsU : written by NormalizeRows, read as the GEMM1 A operand -> dead once GEMM1 drained,
    //         which is when QuantizeRows starts writing the GEMM2 A operand wsR;
    //   wsY : written as the GEMM1 C tile, read by QuantizeRows -> dead once QuantizeRows drained,
    //         which is when GEMM2 starts writing its C tile wsP (GEMM2 can only write C after the
    //         AIV copied the A tile, i.e. after QuantizeRows returned).
    // So wsR aliases wsU and wsP aliases wsY. That halves the user workspace, which also keeps the
    // per call aclrtMalloc request under the ~35MB pooled fast path threshold of the runtime (the
    // aclnn harness allocates the reported workspace every call and pays for the slow path above it).
    td->set_matrixBf16(matrixBf16 ? 1U : 0U);
    td->set_metadataFp16(metadataFp16 ? 1U : 0U);
    const uint64_t rowBytes = static_cast<uint64_t>(wsRows) * KCTQ_HEAD_DIM * sizeof(float);
    const uint64_t matBytes = static_cast<uint64_t>(KCTQ_HEAD_DIM) * KCTQ_HEAD_DIM * sizeof(float);
    uint64_t offset = 0;
    if (matrixBf16) {
        // fp32 copies of the bf16 matrices (the kernel upcasts them in place of the input read)
        td->set_wsRotMat(offset);
        offset += matBytes;
        td->set_wsQjlMat(offset);
        offset += matBytes;
    } else {
        td->set_wsRotMat(0);
        td->set_wsQjlMat(0);
    }
    td->set_wsU(offset);
    offset += rowBytes;
    td->set_wsY(offset);
    offset += rowBytes;
    td->set_wsR(td->get_wsU());
    td->set_wsP(td->get_wsY());
    const uint64_t wsNorm = CeilAlignU64(offset, WS_ALIGN);
    td->set_wsNorm(wsNorm);
    offset = wsNorm + static_cast<uint64_t>(wsRows) * sizeof(float);
    td->set_wsBytes(CeilAlignU64(offset, WS_ALIGN));

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(0U); // a single binary serves 2/3/4 bit; mseBits travels in the TilingData
    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize() + td->get_wsBytes());

    const size_t capacity = context->GetRawTilingData()->GetCapacity();
    OP_CHECK_IF(static_cast<size_t>(td->GetDataSize()) > capacity,
                OP_LOGE(context, "tiling data (%zu) exceeds the buffer capacity (%zu)",
                        static_cast<size_t>(td->GetDataSize()), capacity),
                return ge::GRAPH_FAILED);
    td->SaveToBuffer(context->GetRawTilingData()->GetData(), capacity);
    context->GetRawTilingData()->SetDataSize(td->GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus KvCacheTurboQuantTilingParse(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<KvCacheTurboQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aicNum = platform.GetCoreNumAic();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

IMPL_OP_OPTILING(KvCacheTurboQuant)
    .Tiling(optiling::KvCacheTurboQuantTilingFunc)
    .TilingParse<optiling::KvCacheTurboQuantCompileInfo>(optiling::KvCacheTurboQuantTilingParse);
