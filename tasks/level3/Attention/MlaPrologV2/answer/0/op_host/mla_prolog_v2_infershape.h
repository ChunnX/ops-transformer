/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: MulanPSL-2.0
 */
#ifndef MLA_PROLOG_V2_INFERSHAPE_H
#define MLA_PROLOG_V2_INFERSHAPE_H

#include "mla_prolog_infershape.h"

using namespace ge;

namespace ops {
// INPUT（与开源仓 mla_prolog_v2_infershape.h 一致）
constexpr uint32_t DEQUANT_SCALE_X_INDEX = 12;
constexpr uint32_t QUANT_SCALE_CKV_INDEX = 16;
// OUTPUT
constexpr uint32_t DEQUANT_SCALE_Q_NOPE_INDEX = 4;

ge::graphStatus SetMlaPrologV2ShapeDim(const MlaPrologProtoShapeParam &shapeParam, gert::InferShapeContext *context);

ge::graphStatus InferShapeMlaPrologV2(gert::InferShapeContext *context);
ge::graphStatus InferDataTypeMlaPrologV2(gert::InferDataTypeContext *context);

} // namespace ops

#endif
