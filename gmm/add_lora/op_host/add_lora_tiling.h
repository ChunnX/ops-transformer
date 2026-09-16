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
 * \file add_lora_tiling.h
 * \brief
 */

#ifndef ADD_LORA_TILING_H_
#define ADD_LORA_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

// 注意：TilingData 结构体 AddLoraTiling（19 个 int64_t 字段：B/W/L/R/H1/H2/H3、
// layer_idx/scale_bits/y_offset/s、nseg_cap、ws_xg_off/ws_t_off/ws_lora_off/
// ws_cnt_off/ws_segbase_off/ws_rowpos_off/ws_tbl_off）由 pypto_codegen 从
// op_kernel/add_lora.py 的 dataclass 生成的 AddLoraTiling_tiling.h 提供——
// cmake 对本目录 *_tiling*.cpp 自动 force-include 该生成头（见
// cmake/obj_func.cmake 的 PYPTO_GEN_DIR 处理，* _tilingkey.h 在前、*_tiling.h
// 在后），本头文件不重复定义（同 dense_lightning_indexer_softmax_lse_v2 样板
// tiling.cpp 直接 GetTilingData<DenseLILseV2Tiling>() 的模式）。
struct AddLoraCompileInfo {
    static ge::graphStatus ParamCheck(gert::TilingContext *context, bool hasWeightA, int64_t layerIdx, float scale,
                                      int64_t yOffset, int64_t ySliceSize, int64_t bSize, int64_t wSize,
                                      int64_t lSize, int64_t rSize, int64_t h1Size, int64_t h2Size, int64_t h3Size,
                                      int64_t xDim0, int64_t indicesDim0, int64_t waW, int64_t waL, int64_t waR,
                                      int64_t waH1);
    static ge::graphStatus CheckShapeDims(gert::TilingContext *context, const gert::StorageShape *yShape,
                                          const gert::StorageShape *xShape,
                                          const gert::StorageShape *weightBShape,
                                          const gert::StorageShape *indicesShape,
                                          const gert::StorageShape *weightAShape);
};

} // namespace optiling

#endif // ADD_LORA_TILING_H_
