/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkGatedDeltaRuleComputeWyTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, numChunks);
TILING_DATA_FIELD_DEF(int64_t, groupSize);
TILING_DATA_FIELD_DEF(int64_t, totalTasks);
TILING_DATA_FIELD_DEF(uint32_t, localWorkspaceSize);
TILING_DATA_FIELD_DEF(uint32_t, perCoreWorkspaceBytes);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, workspaceOffset);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmAttn);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmSquare);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmApplyU);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmApplyW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkGatedDeltaRuleComputeWy, ChunkGatedDeltaRuleComputeWyTilingData)

struct ChunkGatedDeltaRuleComputeWyCompileInfo {};
} // namespace optiling
