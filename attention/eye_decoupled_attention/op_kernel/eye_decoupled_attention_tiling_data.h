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
#ifndef EYE_DECOUPLED_ATTENTION_TILING_DATA_H
#define EYE_DECOUPLED_ATTENTION_TILING_DATA_H
#include <cstdint>
struct EyeDecoupledAttentionTilingData {
    uint32_t B; uint32_t N; uint32_t C; uint32_t hb; uint32_t wb;
    uint32_t bn_h; uint32_t bn_w; uint32_t tile_n; uint32_t totalLength; uint32_t workspaceSize;
};
#endif
