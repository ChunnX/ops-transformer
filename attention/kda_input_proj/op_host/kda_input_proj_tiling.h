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
 * \file kda_input_proj_tiling.h
 * \brief
 */

#ifndef KDA_INPUT_PROJ_TILING_H
#define KDA_INPUT_PROJ_TILING_H

#include "err/ops_err.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/kda_input_proj_tiling_data.h"
#include "kda_input_proj_host_common.h"

namespace optiling {
using kda_input_proj::X_INDEX;
using kda_input_proj::WEIGHT_QKV_INDEX;
using kda_input_proj::WEIGHT_BETA_INDEX;
using kda_input_proj::WEIGHT_GATE_INDEX;
using kda_input_proj::WEIGHT_G_INDEX;
using kda_input_proj::WEIGHT_QKV_SCALE_INDEX;
using kda_input_proj::QKV_INDEX;
using kda_input_proj::BETA_INDEX;
using kda_input_proj::GATE_INDEX;
using kda_input_proj::G_INDEX;
using kda_input_proj::ATTR_TRANS_WEIGHT_QKV_INDEX;
using kda_input_proj::ATTR_TRANS_WEIGHT_BETA_INDEX;
using kda_input_proj::ATTR_TRANS_WEIGHT_GATE_INDEX;
using kda_input_proj::ATTR_TRANS_WEIGHT_G_INDEX;

struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct KdaInputProjCompileInfo {};

struct KdaInputProjParaInfo {
    TilingRequiredParaInfo x = {nullptr, nullptr};
    TilingRequiredParaInfo weightQkv = {nullptr, nullptr};
    TilingRequiredParaInfo weightBeta = {nullptr, nullptr};
    TilingRequiredParaInfo weightGate = {nullptr, nullptr};
    TilingRequiredParaInfo weightG = {nullptr, nullptr};
    TilingRequiredParaInfo weightQkvScale = {nullptr, nullptr};
    TilingRequiredParaInfo qkvOut = {nullptr, nullptr};
    TilingRequiredParaInfo betaOut = {nullptr, nullptr};
    TilingRequiredParaInfo gateOut = {nullptr, nullptr};
    TilingRequiredParaInfo gOut = {nullptr, nullptr};

    const bool *transWeightQkv = nullptr;
    const bool *transWeightBeta = nullptr;
    const bool *transWeightGate = nullptr;
    const bool *transWeightG = nullptr;
};

class KdaInputProjTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    KdaInputProjParaInfo opParamInfo;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND950;
    uint32_t tSize = 0;
    bool transWeightQkv = true;
    bool transWeightBeta = true;
    bool transWeightGate = true;
    bool transWeightG = true;
    ge::DataType inputXType = ge::DT_BF16;
    ge::DataType inputWeightQkvType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType outputQkvType = ge::DT_BF16;
};

class KdaInputProjInfoParser {
public:
    explicit KdaInputProjInfoParser(gert::TilingContext *context) : context_(context)
    {
    }
    ~KdaInputProjInfoParser() = default;

    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAndCheckAttrParaInfo();
    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetBaseShapeInfo();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus ParseAndCheck(KdaInputProjTilingInfo &tilingInfo);
    void GenerateInfo(KdaInputProjTilingInfo &tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    KdaInputProjParaInfo opParamInfo_;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND950;
    uint32_t tSize_ = 0;
    ge::DataType inputXType_ = ge::DT_BF16;
    ge::DataType inputWeightQkvType_ = ge::DT_FLOAT8_E4M3FN;
    ge::DataType outputQkvType_ = ge::DT_BF16;
};

class KdaInputProjTiling {
public:
    explicit KdaInputProjTiling(gert::TilingContext *context) : context_(context)
    {
    }
    ge::graphStatus DoTiling(const KdaInputProjTilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    KdaInputProjTilingData tilingData_;
};
} // namespace optiling

#endif // KDA_INPUT_PROJ_TILING_H
