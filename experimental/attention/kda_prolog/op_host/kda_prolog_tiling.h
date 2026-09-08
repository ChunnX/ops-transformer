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
 * \file kda_prolog_tiling.h
 * \brief
 */

#ifndef KDA_PROLOG_TILING_H
#define KDA_PROLOG_TILING_H

#include <cstdint>
#include "err/ops_err.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/kda_prolog_tiling_data.h"
#include "kda_prolog_host_common.h"

namespace optiling {
using kda_prolog::X_INDEX;
using kda_prolog::WEIGHT_QKV_INDEX;
using kda_prolog::WEIGHT_BETA_INDEX;
using kda_prolog::WEIGHT_GATE_INDEX;
using kda_prolog::WEIGHT_G_INDEX;
using kda_prolog::SCALE_QKV_INDEX;
using kda_prolog::QKV_INDEX;
using kda_prolog::BETA_INDEX;
using kda_prolog::GATE_INDEX;
using kda_prolog::G_INDEX;
using kda_prolog::ATTR_TRANS_WEIGHT_QKV_INDEX;
using kda_prolog::ATTR_TRANS_WEIGHT_BETA_INDEX;
using kda_prolog::ATTR_TRANS_WEIGHT_GATE_INDEX;
using kda_prolog::ATTR_TRANS_WEIGHT_G_INDEX;

struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc = nullptr;
    const gert::StorageShape *shape = nullptr;
};

struct KdaPrologCompileInfo {};

struct KdaPrologParaInfo {
    TilingRequiredParaInfo x;
    TilingRequiredParaInfo weightQkv;
    TilingRequiredParaInfo weightBeta;
    TilingRequiredParaInfo weightGate;
    TilingRequiredParaInfo weightG;
    TilingRequiredParaInfo scaleQkv;
    const gert::CompileTimeTensorDesc *qkvOutDesc = nullptr;
    const gert::CompileTimeTensorDesc *betaOutDesc = nullptr;
    const gert::CompileTimeTensorDesc *gateOutDesc = nullptr;
    const gert::CompileTimeTensorDesc *gOutDesc = nullptr;

    const bool *transWeightQkv = nullptr;
    const bool *transWeightBeta = nullptr;
    const bool *transWeightGate = nullptr;
    const bool *transWeightG = nullptr;
};

class KdaPrologTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    KdaPrologBaseParams baseParams; // 与 tiling_data.baseParams 对齐；核数运行时经 platform 获取
    bool transWeightQkv = true;
    bool transWeightBeta = true;
    bool transWeightGate = true;
    bool transWeightG = true;
    uint32_t aicNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
};

class KdaPrologInfoParser {
public:
    explicit KdaPrologInfoParser(gert::TilingContext *context)
        : context_(context)
    {}
    ~KdaPrologInfoParser() = default;

    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAndCheckAttrParaInfo();
    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetBaseShapeInfo();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus ParseAndCheck(KdaPrologTilingInfo &tilingInfo);
    void GenerateInfo(KdaPrologTilingInfo &tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    KdaPrologParaInfo opParamInfo_;
    KdaPrologBaseParams baseParams_;
    uint32_t aicNum_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t l0cSize_ = 0;
};

class KdaPrologTiling {
public:
    explicit KdaPrologTiling(gert::TilingContext *context)
        : context_(context)
    {}
    ge::graphStatus DoTiling(const KdaPrologTilingInfo *tilingInfo);

private:
    ge::graphStatus FillBaseParams(const KdaPrologTilingInfo &tilingInfo);
    ge::graphStatus CalcModuleTilings(const KdaPrologTilingInfo &tilingInfo);
    ge::graphStatus CalcWorkspaceSize(const KdaPrologTilingInfo &tilingInfo);
    ge::graphStatus SetTilingKey(const KdaPrologTilingInfo &tilingInfo);
    ge::graphStatus WriteTilingResult(const KdaPrologTilingInfo &tilingInfo);

    gert::TilingContext *context_ = nullptr;
    KdaPrologTilingData tilingData_;
    uint64_t workspaceSize_ = 0;
};
} // namespace optiling

#endif // KDA_PROLOG_TILING_H
