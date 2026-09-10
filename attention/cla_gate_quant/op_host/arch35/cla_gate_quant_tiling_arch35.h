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
 * \file cla_gate_quant_tiling_arch35.h
 * \brief Host-side tiling declarations for ClaGateQuant
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_CLA_GATE_QUANT_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_CLA_GATE_QUANT_H

#include <cstdint>
#include <vector>
#include <string>
#include <set>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/cla_gate_quant_tilingdata.h"

namespace optiling {

struct ClaGateQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct ClaGateQuantTilingParam {
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t ubSize{0};
    int64_t roundMode{0};
    int64_t dstType{0};
    int64_t scaleAlg{0};
    int64_t blockW{256};
    int64_t splitBlockH{64};
    int64_t dimM{1};
    int64_t dimN{1};
    int64_t dimHeadNum{1};
    int64_t dimHeadSize{1};
    int64_t dimNBlockNum{0};
    int64_t dimNTail{0};
    int64_t workspaceSize{0};
    int64_t dtypeSize{2};
    int64_t dualAxisFlag{1}; // 1: dual-axis, 0: tail-axis single
    int64_t rowTileNum{0};
    int64_t colTileNum{0};
    int64_t rowBlockSize{0};
    int64_t rowTailSize{0};
    int64_t colBlockSize{0};
    int64_t colTailSize{0};
    int64_t batchCount{0};
    int64_t batchSize{0};
    ge::DataType yDtype{ge::DT_UNDEFINED};
};

enum class RoundModeList {
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

ge::graphStatus TilingForClaGateQuant(gert::TilingContext *context);
ge::graphStatus TilingPrepareForClaGateQuant(gert::TilingParseContext *context);

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_CLA_GATE_QUANT_H
