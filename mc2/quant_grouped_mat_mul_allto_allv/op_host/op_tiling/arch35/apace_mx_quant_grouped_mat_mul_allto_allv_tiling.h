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
 * \file apace_mx_quant_grouped_mat_mul_allto_allv_tiling.h
 * \brief QuantGroupedMatMulAlltoAllv 的 apace hcomm 通路 tiling 类
 *
 * 复用 MxQuantGroupedMatmulAllToAllvTiling 的全量输入校验，产出 apace 自有
 * hcommQuantGmmAlltoAllvTilingData 并通过 TILINGKEY_COMM_MODE=TILINGKEY_TPL_APACE
 * 选择 apace kernel 实例。未使能（环境变量关闭/不满足 apace 约束）时返回
 * GRAPH_PARAM_INVALID，回退原通路 tiling 类。
 */

#ifndef APACE_MX_QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H
#define APACE_MX_QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H

#include "mx_quant_grouped_mat_mul_allto_allv_tiling.h"
#include "apace/kernel/fusions/quant_grouped_matmul_all_to_all_v/quant_grouped_matmul_all_to_all_v_tiling_data.h"
#include "apace/tiling/quant_matmul_tiling_swat.h"

namespace Mc2Tiling {
namespace Mc2GroupedMatmul {

class ApaceMxQuantGroupedMatmulAllToAllvTiling : public MxQuantGroupedMatmulAllToAllvTiling {
public:
    explicit ApaceMxQuantGroupedMatmulAllToAllvTiling(gert::TilingContext *context)
        : MxQuantGroupedMatmulAllToAllvTiling(context) {};
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }
    ~ApaceMxQuantGroupedMatmulAllToAllvTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

private:
    bool IsApaceEnvEnabled() const;
    ge::graphStatus SetApaceTilingData(Apace::hcommQuantGmmAlltoAllvTilingData *td);
    ge::graphStatus SetApaceHcclTiling(Apace::hcommQuantGmmAlltoAllvTilingData *td);
    ge::graphStatus RunSwatTiling(uint64_t m, uint64_t n, uint64_t k, bool transB, QuantMatmulTilingData &mmTile);

    uint32_t usedCoreNum_{0};
    uint64_t apaceWorkSpaceSize_{0};
};

} // namespace Mc2GroupedMatmul
} // namespace Mc2Tiling

#endif // APACE_MX_QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_H
