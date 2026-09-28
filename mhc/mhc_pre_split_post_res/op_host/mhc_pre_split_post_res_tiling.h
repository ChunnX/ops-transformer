/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mhc_pre_split_post_res_tiling.h
 * \brief MhcPreSplitPostRes TilingData/CompileInfo 定义与 Tiling 解析类声明
 */

#ifndef MHC_PRE_SPLIT_POST_RES_TILING_H_
#define MHC_PRE_SPLIT_POST_RES_TILING_H_

#include "mhc_pre_split_post_res_const.h"

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"

namespace optiling {

// Tiling key（schMode 模板实参）定义（constexpr，支持 optiling:: 命名空间限定访问，UT 亦引用）。
// 与 op_kernel/arch35/mhc_pre_split_post_res_tiling_key.h 的 MHC_PRS_SCH_MODE_* 宏同值互锚
// （kernel 侧无法 include 本头文件，两侧改动必须同步）；Host 侧经 GET_TPL_TILING_KEY 编码进 TilingKey。
// 单Tile模式：dimLoop==1（D<=2560）；token 不足（T < coreNum/2）时每token使用2个核心，否则1个
constexpr uint64_t MHC_PRE_SPLIT_TILING_KEY_SINGLE_TILE = 0;
// 多Tile模式：dimLoop>1（D>2560）；token 不足（T < coreNum/2）时每token使用2个核心，否则1个
constexpr uint64_t MHC_PRE_SPLIT_TILING_KEY_MULTI_TILE = 1;

// -----------算子TilingData定义（13个字段）---------------
BEGIN_TILING_DATA_DEF(MhcPreSplitPostResTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalLength);     // B*S 或 T，合轴后的token总数
TILING_DATA_FIELD_DEF(int64_t, n);               // 超连接维度 (固定4)
TILING_DATA_FIELD_DEF(int64_t, d);               // Dim维度 (白名单 2560/5120/6144/7168)
TILING_DATA_FIELD_DEF(int64_t, phiRows);         // N^2+N，phi的第一维
TILING_DATA_FIELD_DEF(int64_t, dimTile);         // 核内D的分块大小，min(D, 2560)
TILING_DATA_FIELD_DEF(int64_t, dimLoop);         // D需要几个tile
TILING_DATA_FIELD_DEF(int64_t, coresPerToken);   // 每token使用的核数（1或2）
TILING_DATA_FIELD_DEF(int64_t, phiRowMid);       // 单核处理的phi行起始偏移（双核模式用）
TILING_DATA_FIELD_DEF(int64_t, baseT);           // 每（组）核处理的token数（前coreTAct-1组）
TILING_DATA_FIELD_DEF(int64_t, tailT);           // 最后一（组）核处理的token数
TILING_DATA_FIELD_DEF(int64_t, coreTAct);        // T轴实际分组数
TILING_DATA_FIELD_DEF(float, normEps);           // RmsNorm epsilon
TILING_DATA_FIELD_DEF(float, invK);              // 1.0f/(N*D)，RmsNorm均值系数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcPreSplitPostRes, MhcPreSplitPostResTilingData)

// -----------算子CompileInfo定义-------------------
struct MhcPreSplitPostResCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::RESERVED_VERSION;
};

// -----------算子Tiling入参信息解析及Check类---------------
class MhcPreSplitPostResInfoParser {
public:
    explicit MhcPreSplitPostResInfoParser(gert::TilingContext *context) : context_(context)
    {}
    ~MhcPreSplitPostResInfoParser() = default;

    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus CheckXValid();
    ge::graphStatus CheckPhiValid();
    ge::graphStatus CheckAlphaValid();
    ge::graphStatus CheckBiasValid();
    ge::graphStatus GenerateTiling();
    ge::graphStatus ParseAndCheck();

protected:
    gert::TilingContext  *context_     = nullptr;
    const char           *opName_      = nullptr;
    fe::PlatFormInfos    *platformInfo_ = nullptr;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND950;

    // 硬件参数
    uint64_t coreNum_ = 0;
    uint64_t ubSize_  = 0;

    // 算子参数
    int64_t  totalLength_   = 0;   // 合轴后 T = B*S 或直接 T
    int64_t  n_             = 0;
    int64_t  d_             = 0;
    int64_t  phiRows_       = 0;   // N^2 + N（完整形态）或 N（仅 post 形态，无 h_res）
    bool     hasRes_        = true; // phi 行数为 N^2+N 时为 true（h_res 必选）
    int64_t  dimTile_       = 0;
    int64_t  dimLoop_       = 0;
    int64_t  coresPerToken_ = 0;
    int64_t  phiRowMid_     = 0;
    int64_t  blockDim_      = 0;
    int64_t  baseT_         = 0;
    int64_t  tailT_         = 0;
    int64_t  coreTAct_      = 0;
    float    invK_            = 0.0f; // 1.0f/(N*D)

    float    normEps_     = MHC_DEFAULT_NORM_EPS;

    // tiling key
    uint64_t tilingKey_ = 0;

private:
    // GenerateTiling 子计算（超大函数拆分）
    ge::graphStatus CheckUbBudget();
};

// ---------------算子Tiling类---------------
class MhcPreSplitPostResTiling : public MhcPreSplitPostResInfoParser {
public:
    explicit MhcPreSplitPostResTiling(gert::TilingContext *context)
        : MhcPreSplitPostResInfoParser(context)
    {}
    ~MhcPreSplitPostResTiling() = default;
    ge::graphStatus DoTiling();

private:
    MhcPreSplitPostResTilingData tilingData_;
};

}  // namespace optiling
#endif  // MHC_PRE_SPLIT_POST_RES_TILING_H_
