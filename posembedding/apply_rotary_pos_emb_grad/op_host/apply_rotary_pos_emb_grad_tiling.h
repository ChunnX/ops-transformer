/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_grad_tiling.h
 * \brief apply_rotary_pos_emb_grad host 侧 tiling 类声明和 tiling data 定义
 * 双路反向：Q+K 同时计算
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_ROTARY_POS_EMB_GRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_ROTARY_POS_EMB_GRAD_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "../op_kernel/arch35/apply_rotary_pos_emb_grad_tiling_data.h"
#include "platform/platform_info.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "util/math_util.h"

namespace optiling {

// ============================================================================
// Host 侧 tiling data —— 用 TILING_DATA_FIELD_DEF 宏注册，支持序列化到 kernel 侧
// ============================================================================
BEGIN_TILING_DATA_DEF(ApplyRopeGradParams)
TILING_DATA_FIELD_DEF(uint64_t, layout);
TILING_DATA_FIELD_DEF(uint64_t, b);
TILING_DATA_FIELD_DEF(uint64_t, s);
TILING_DATA_FIELD_DEF(uint64_t, d);
TILING_DATA_FIELD_DEF(uint64_t, nQ);
TILING_DATA_FIELD_DEF(uint64_t, nK);
TILING_DATA_FIELD_DEF(uint64_t, cosb);
TILING_DATA_FIELD_DEF(uint64_t, blockNumB);
TILING_DATA_FIELD_DEF(uint64_t, blockFactorB);
TILING_DATA_FIELD_DEF(uint64_t, blockNumS);
TILING_DATA_FIELD_DEF(uint64_t, blockFactorS);
TILING_DATA_FIELD_DEF(uint64_t, ubFactorS);
TILING_DATA_FIELD_DEF(uint64_t, ubFactorB);
TILING_DATA_FIELD_DEF(uint64_t, ubLoopNumN);
TILING_DATA_FIELD_DEF(uint64_t, ubFactorN);
TILING_DATA_FIELD_DEF(uint64_t, ubTailFactorN);
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, coreNum);
TILING_DATA_FIELD_DEF(uint64_t, rotaryMode);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyRopeGradParamsOp, ApplyRopeGradParams)

BEGIN_TILING_DATA_DEF(ApplyRopeGradABParams)
TILING_DATA_FIELD_DEF(uint64_t, b);
TILING_DATA_FIELD_DEF(uint64_t, s);
TILING_DATA_FIELD_DEF(uint64_t, d);
TILING_DATA_FIELD_DEF(uint64_t, nQ);
TILING_DATA_FIELD_DEF(uint64_t, nK);
TILING_DATA_FIELD_DEF(uint64_t, dAlign);
TILING_DATA_FIELD_DEF(uint64_t, dSplitCoef);
TILING_DATA_FIELD_DEF(uint64_t, blockNumBS);
TILING_DATA_FIELD_DEF(uint64_t, blockFactorBS);
TILING_DATA_FIELD_DEF(uint64_t, blockTailBS);
TILING_DATA_FIELD_DEF(uint64_t, blockNumN);
TILING_DATA_FIELD_DEF(uint64_t, blockFactorN);
TILING_DATA_FIELD_DEF(uint64_t, blockTailN);
TILING_DATA_FIELD_DEF(uint64_t, ubFactorBS);
TILING_DATA_FIELD_DEF(uint64_t, ubFactorN);
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, rotaryMode);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyRopeGradABParamsOp, ApplyRopeGradABParams)

BEGIN_TILING_DATA_DEF(ApplyRotaryXParamsDef)
TILING_DATA_FIELD_DEF(uint64_t, b);
TILING_DATA_FIELD_DEF(uint64_t, s);
TILING_DATA_FIELD_DEF(uint64_t, d);
TILING_DATA_FIELD_DEF(uint64_t, nQ);
TILING_DATA_FIELD_DEF(uint64_t, nK);
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, blockFactorN);
TILING_DATA_FIELD_DEF(uint64_t, blockTailFactorN);
TILING_DATA_FIELD_DEF(uint64_t, ubFactorN);
TILING_DATA_FIELD_DEF(uint64_t, rotaryMode);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyRotaryXParamsDefOp, ApplyRotaryXParamsDef)

BEGIN_TILING_DATA_DEF(ApplyRopeGradTilingDataHost)
TILING_DATA_FIELD_DEF_STRUCT(ApplyRopeGradParams, ropeGradParams);
TILING_DATA_FIELD_DEF_STRUCT(ApplyRopeGradABParams, ropeGradABParams);
TILING_DATA_FIELD_DEF_STRUCT(ApplyRotaryXParamsDef, rotaryXParams);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmbGrad, ApplyRopeGradTilingDataHost)

// ============================================================================
// ApplyRopeLayout 枚举 — internal layout, 与 kernel 侧 Layout (0-3) 一致
// OpDef attr layout: {1=BSND, 2=SBND, 3=BNSD, 4=TND} → 内部 {0,1,2,0}, + NO_BROADCAST=3
// ============================================================================
enum class ApplyRopeLayout : int64_t
{
    BSND = 0,          // cos (1, S, 1, D) (含 TND 映射后的 1S1D) → BAB
    SBND = 1,          // cos (B, S, 1, D) B>1 → AB
    BNSD = 2,          // cos (B, 1, S, D) / (1, 1, S, D) — 参数预留，不实现
    NO_BROADCAST = 3   // cos == grad_query_embed (N=1) → A
};

// ============================================================================
// DxTilingKey 枚举
// ============================================================================
enum class ApplyDxTilingKey : uint32_t
{
    KEY_BAB = 203,  // BSND (含 TND) — BAB tiling
    KEY_AB  = 204,  // SBND — AB tiling
    KEY_A   = 205   // NO_BROADCAST (N=1, cos==grad) — A tiling
};

// ============================================================================
// ApplyRopeGradRegbaseTilingClass — 母板 tiling class
// ============================================================================
class ApplyRopeGradRegbaseTilingClass : public Ops::Transformer::OpTiling::TilingBaseClass
{
public:
    explicit ApplyRopeGradRegbaseTilingClass(gert::TilingContext* context) : TilingBaseClass(context) {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

private:
    ge::graphStatus CheckNullptr();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckShapeAllPositive() const;
    ge::graphStatus CheckInPutShapeAllPositive(const int64_t idx) const;
    ge::graphStatus CheckOutPutShapeAllPositive(const int64_t idx) const;
    ge::graphStatus CheckDtypeAndAttr();
    ge::graphStatus CheckParam();
    ge::graphStatus CheckShapeLimit();
    ge::graphStatus CheckOptionalInput() const;
    ge::graphStatus CheckShapeDim() const;
    ge::graphStatus JudgeLayoutByShape(
        const gert::Shape& dyQShape, const gert::Shape& dyKShape, const gert::Shape& cosShape, int64_t layoutVal);
    ge::graphStatus CheckRotaryModeShapeRelation(const int64_t d);
    ge::graphStatus GetInputParam(Ops::Base::ReduceOpInputParam& opInput, uint32_t inputIdx, uint32_t axesIdx);
    ge::graphStatus InitTilingData();
    ge::graphStatus TilingReduce();
    ge::graphStatus SetTilingKeyBlockDim(uint32_t dxTilingKey, uint32_t layout);
    ge::graphStatus SetRotaryXTilingData();
    ge::graphStatus GetReduceOpCompileInfo(Ops::Base::ReduceOpCompileInfo* compileInfo);

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetWorkspaceSize() override { return ge::GRAPH_SUCCESS; }
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }
    ge::graphStatus DoOpTiling() override { return ge::GRAPH_SUCCESS; }
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override { return ge::GRAPH_SUCCESS; }
    bool IsCapable() override { return true; }

protected:
    // 硬件参数
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    int64_t blockSize_{0};
    int64_t vLength_{0};
    int64_t dSplitCoef_{2};  // half mode: D/2

    // 形状参数
    int64_t b_{0}, s_{0}, nQ_{0}, nK_{0}, d_{0};
    int64_t cosb_{0};
    int64_t usedCoreNum_{0};
    ge::DataType dtype_;
    ApplyRopeLayout layout_ = ApplyRopeLayout::BSND;
    int64_t rotaryMode_{0};  // 仅 HALF=0

    // Dcos flag
    uint32_t dCosFlag_{0};

    // Tiling data
    ApplyRopeGradTilingData* tilingData_{nullptr};

    // Reduce tiling
    Ops::Base::ReduceTilingKey key_;
    uint64_t tilingKey_{0};

    // 标志
    bool isTndLayout_{false};
    bool is1snd_{false};
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_ROTARY_POS_EMB_GRAD_H
