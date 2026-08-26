/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_CHECKER_H
#define QUANT_LIGHTNING_INDEXER_V2_CHECKER_H

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/tiling_api.h"

namespace optiling {
ge::graphStatus CheckQuantLightningIndexerV2(gert::TilingContext *context);
}
#endif
