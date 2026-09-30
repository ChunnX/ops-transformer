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
 * \file kv_cache_turbo_quant_tiling.h
 * \brief Host side TilingData definition. The field list must stay in sync with the device side POD
 *        mirror in op_kernel/kv_cache_turbo_quant_tiling_data.h (the kernel copies the raw buffer).
 */

#ifndef OPS_OP_HOST_KV_CACHE_TURBO_QUANT_TILING_H_
#define OPS_OP_HOST_KV_CACHE_TURBO_QUANT_TILING_H_

#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(KvCacheTurboQuantTilingDef)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
TILING_DATA_FIELD_DEF(uint32_t, mseBits);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, totalRows);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, alignedRows);
TILING_DATA_FIELD_DEF(uint32_t, aivRowStride);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, idxBytesPerRow);
TILING_DATA_FIELD_DEF(uint32_t, qjlBytesPerRow);
TILING_DATA_FIELD_DEF(uint64_t, wsU);
TILING_DATA_FIELD_DEF(uint64_t, wsY);
TILING_DATA_FIELD_DEF(uint64_t, wsR);
TILING_DATA_FIELD_DEF(uint64_t, wsP);
TILING_DATA_FIELD_DEF(uint64_t, wsNorm);
TILING_DATA_FIELD_DEF(uint64_t, wsBytes);
TILING_DATA_FIELD_DEF(uint32_t, matrixBf16);
TILING_DATA_FIELD_DEF(uint32_t, metadataFp16);
TILING_DATA_FIELD_DEF(uint64_t, wsRotMat);
TILING_DATA_FIELD_DEF(uint64_t, wsQjlMat);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCacheTurboQuant, KvCacheTurboQuantTilingDef)

struct KvCacheTurboQuantCompileInfo {
    int64_t aicNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling

#endif // OPS_OP_HOST_KV_CACHE_TURBO_QUANT_TILING_H_
