/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "register/op_def_registry.h"
#include "../op_kernel/eye_decoupled_attention_tiling_data.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    EyeDecoupledAttentionTilingData* tiling = context->GetTilingData<EyeDecoupledAttentionTilingData>();
    const gert::StorageShape *x_shape = context->GetInputShape(0);
    uint32_t B = x_shape->GetStorageShape().GetDim(0);
    uint32_t N = x_shape->GetStorageShape().GetDim(1);
    uint32_t C = *(context->GetAttrs()->GetInt(0));
    uint32_t hb = *(context->GetAttrs()->GetInt(1));
    uint32_t wb = *(context->GetAttrs()->GetInt(2));
    uint32_t bn_h = *(context->GetAttrs()->GetInt(3));
    uint32_t bn_w = *(context->GetAttrs()->GetInt(4));
    tiling->B = B; tiling->N = N; tiling->C = C; tiling->hb = hb; tiling->wb = wb;
    tiling->bn_h = bn_h; tiling->bn_w = bn_w; tiling->tile_n = 1;
    tiling->totalLength = B * N * C;
    tiling->workspaceSize = 3 * B * N * C * 2;
    size_t* _ws = context->GetWorkspaceSizes(1);
    if (_ws) { _ws[0] = tiling->workspaceSize; }
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(EyeDecoupledAttention).Tiling(TilingFunc);
} // namespace optiling
