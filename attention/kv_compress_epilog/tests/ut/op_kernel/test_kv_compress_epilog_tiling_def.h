/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TEST_KV_COMPRESS_EPILOG_TILING_DEF_H
#define TEST_KV_COMPRESS_EPILOG_TILING_DEF_H

#include <vector>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

using namespace ge;

namespace optiling {
BEGIN_TILING_DATA_DEF(KvCompressEpilogTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, kvCacheCol);
TILING_DATA_FIELD_DEF(int64_t, kvCacheRowStride);
TILING_DATA_FIELD_DEF(int64_t, kvCacheBlockSize);
TILING_DATA_FIELD_DEF(int64_t, kvCacheBlockStride);
TILING_DATA_FIELD_DEF(int64_t, scaleCol);
TILING_DATA_FIELD_DEF(int64_t, concatCol);
TILING_DATA_FIELD_DEF(int64_t, padCol);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, roundScale);
TILING_DATA_FIELD_DEF(int64_t, perGroupSize);
TILING_DATA_FIELD_DEF(int64_t, rowOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, rowLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, rowFactor);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, tailRowFactorOfTailBlock);
TILING_DATA_FIELD_DEF(float, scalesAttr);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCompressEpilog, KvCompressEpilogTilingData)
}  // namespace optiling

#endif  // TEST_KV_COMPRESS_EPILOG_TILING_DEF_H
