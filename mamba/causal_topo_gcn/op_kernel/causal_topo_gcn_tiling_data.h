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
#ifndef CAUSAL_TOPO_GCN_TILING_DATA_H
#define CAUSAL_TOPO_GCN_TILING_DATA_H
#include <cstdint>
struct CausalTopoGcnTilingData {
    // 【TODO: 需要你修改】填入你的 Tiling 切分数据结构
    uint32_t num_nodes;
    uint32_t num_edges;
    uint32_t in_channels;
    uint32_t out_channels;
    uint32_t totalLength;
    uint32_t workspaceSize;
};
#endif
