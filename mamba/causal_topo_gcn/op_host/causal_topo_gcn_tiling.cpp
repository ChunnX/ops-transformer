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
#include "../op_kernel/causal_topo_gcn_tiling_data.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    CausalTopoGcnTilingData* tiling = context->GetTilingData<CausalTopoGcnTilingData>();
    
    // 【TODO: 需要你修改】获取正确的维度和属性，并赋值给 tiling 结构体
    const gert::StorageShape *x_shape = context->GetInputShape(0);
    const gert::StorageShape *edge_shape = context->GetInputShape(1);
    
    uint32_t num_nodes = x_shape->GetStorageShape().GetDim(0);
    uint32_t in_channels = *(context->GetAttrs()->GetInt(0));
    uint32_t out_channels = *(context->GetAttrs()->GetInt(1));

    // CANN 8.0 标准赋值法
    tiling->num_nodes = num_nodes;
    tiling->in_channels = in_channels;
    tiling->out_channels = out_channels;
    tiling->totalLength = num_nodes * in_channels;
    
    // 设置 Workspace
    tiling->workspaceSize = num_nodes * in_channels * 2;
    size_t* _ws = context->GetWorkspaceSizes(1);
    if (_ws) { _ws[0] = tiling->workspaceSize; }
    
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(CausalTopoGCN).Tiling(TilingFunc);
} // namespace optiling
