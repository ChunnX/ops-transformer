/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_matmul_tiling_v2.h
 * \brief
 */

#ifndef __ALL_GATHER_MATMUL_TILING_V2__
#define __ALL_GATHER_MATMUL_TILING_V2__

#pragma once
#include "../all_gather_matmul_tiling_base_v2.h"
#include "register/tilingdata_base.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "mc2_matmul_tiling_cfg.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "../../../op_kernel/all_gather_matmul_tiling_data.h"
// APACE 非量化分支（N4）：tiling 结构复用既有 hcommAllGatherMatmulTilingData（include 引用不改动，R4）
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_tiling_data.h"

namespace optiling {

using namespace mc2_matmul_v3_advanced;

class AllGatherMatmulTilingV2 : public AllGatherMatmulTilingBase {
public:
    explicit AllGatherMatmulTilingV2(gert::TilingContext *context);
    ~AllGatherMatmulTilingV2() override = default;
    ge::graphStatus DoMatmulV3Tiling(Mc2MatmulHelper::Mc2MatmulTilingCfg &tilingCfg, Mc2MMRegisterCfg &registerCfg,
                                     Mc2MatMulV3TilingData &tilingData);

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus PostTiling() override;

    // APACE 分支覆写：tiling key 扩 USING_APACE_IMPL 位（位序见 apt_tiling_key.h 冻结句）
    [[nodiscard]] uint64_t GetTilingKey() const override;

    // APACE 分支覆写：workspace = libApi + gatherLen（不消费 v1 的 biasLen 预留）
    ge::graphStatus GetWorkspaceSize() override;

    CutResult GetTilingResult() override;

    Mc2Tiling::RCSTiling &MutableRCSTilingData()
    {
        return allGatherMatmulTilingDataV2_->param;
    }

    Mc2MatMulV3TilingData &MutableMC2MatmulV3LocalTilingData()
    {
        return allGatherMatmulTilingDataV2_->mc2MmV3LocalTilingData;
    }
    Mc2MatMulV3TilingData &MutableMC2MatmulV3TileTilingData()
    {
        return allGatherMatmulTilingDataV2_->mc2MmV3TileTilingData;
    }
    Mc2MatMulV3TilingData &MutableMC2MatmulV3TailTilingData()
    {
        return allGatherMatmulTilingDataV2_->mc2MmV3TailTilingData;
    }

    void PrintAllTilingData();
    ge::graphStatus DoVersion2Tiling();
    ge::graphStatus SetMc2Hcomm(Mc2Tiling::RCSTiling &rcsCfg);
    ge::graphStatus SetRawTilingData();
    ge::graphStatus CheckInput() override;

private:
    // APACE_IMPL_MAX_RANK_DIM=64 为本算子自定义值（V2-B3：按 B7 rankSize 支持域 2~64 放宽，
    // 由 N2 的 FragmentTensor MaxFragments=64 + 数组扩容承接；a2a 先例同名常量为 8 属其业务上限）
    static constexpr uint64_t APACE_IMPL_MAX_RANK_DIM = 64;

    // APACE 使能开关（B2 拍板：初期默认 v1 路径，预留 env 覆盖，上板验证通过后切默认）
    void SetApaceImplSwitch();
    // APACE 分支 tiling 产出：commTile（fit-balance + 256 对齐封顶，O4）/ mmTile（v1 local tiling 映射）/
    // mc2InitTiling/mc2CcTiling（SetMc2Hcomm 产出共享）/ gatherLen（v1 GetStorageA 语义）
    ge::graphStatus DoApaceTiling();
    void FillApaceMmTile(const Mc2MatMulV3TilingData &localTiling);

    Mc2Tiling::AllGatherMatmulTilingDataV2 allGatherMatmulTilingDataV2Self_;
    Mc2Tiling::AllGatherMatmulTilingDataV2 *allGatherMatmulTilingDataV2_;
    uint64_t myWorkSpaceSize_{0U};
    Mc2MatmulV3CompileInfo compileInfo_;
    Mc2MatMulV3Args mmV3Args_;
    bool usingApaceImpl_{false};
    bool usingApaceImplEnabled_{false};
    Apace::hcommAllGatherMatmulTilingData hcommTilingDataSelf_{};
};
ge::graphStatus AllGatherMatmulTilingAIVModeFunc(gert::TilingContext *context);
} // namespace optiling
#endif
