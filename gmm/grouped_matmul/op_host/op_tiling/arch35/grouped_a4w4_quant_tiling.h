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
 * \file grouped_a4w4_quant_tiling.h
 * \brief A5 A4W4 (INT4xINT4) quant tiling.
 *        Inherits GroupedQmmBasicApiTiling, reusing inputParams_/basicTiling_/mList_/kList_/nList_/aicoreParams_.
 *        Plan: baseM=128, baseN=256, ubCalSize=4096. Tiling + dispatch only (no kernel prologue/epilogue).
 */
#ifndef GROUPED_A4W4_QUANT_TILING_H
#define GROUPED_A4W4_QUANT_TILING_H

#include "../grouped_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "grouped_quant_matmul_tiling.h"
#include "grouped_quant_basic_api_matmul_tiling.h"
#include "op_host/tiling_base.h"

namespace optiling {

constexpr uint32_t A4W4_QUANT_GROUP_SIZE = 128U;    // int8 baseK 对齐粒度（c0=32 倍数，L0B 约束）
constexpr uint32_t A4W4_BASE_M_950 = 128U;
constexpr uint32_t A4W4_BASE_N_950 = 256U;          // 950 L0C=256KB 支持 baseM·baseN=32768（dbL0C=2）
constexpr uint32_t A4W4_VEC_BASE_M_950 = 16U;
constexpr uint32_t A4W4_BASE_K_950 = 128U;          // int8 Mmad baseK（L0B 约束 baseN·baseK≤32768）
constexpr uint32_t A4W4_EPILOGUE_UB_COEFF = 24U;    // epilogue UB: 4(vecIn half DB2)+4(vecOut half DB2)+16(4×float)
constexpr uint32_t PROLOGUE_MIN_BYTES_950 = 48UL * 1024UL;
constexpr uint32_t A4W4_MMOUT_PIPELINE = 4U;

class GroupedA4W4QuantTiling : public GroupedQmmBasicApiTiling {
public:
    explicit GroupedA4W4QuantTiling(gert::TilingContext *context);
    ~GroupedA4W4QuantTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        GroupedQmmBasicApiTiling::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    virtual void Reset();

    // A4W4-specific (hide base's same-named; call base then sync a4w4Tiling_)
    bool SetMKN(const gert::Shape &xShape, const gert::Shape &wShape);
    bool SetGroupNum(uint32_t groupListIndex);
    bool SetMKNList();
    bool AnalyzeA4W4();
    bool CheckA4W4Params();
    void CalBasicBlockA4W4();
    ge::graphStatus CalUbDivideA4W4();
    ge::graphStatus CalWorkspaceA4W4();
    uint32_t FindBestSingleN() const;
    void InitCommonL1TilingFields();
    ge::graphStatus CalcLeftL1Size(uint64_t &leftL1Size) const;
    ge::graphStatus CalL1Tiling();
    ge::graphStatus CalL1Depth(uint64_t leftL1Size);

private:
    struct A4W4BasicTiling {
        uint64_t m = 0;
        uint64_t n = 0;
        uint64_t k = 0;
        uint32_t baseM = A4W4_BASE_M_950;
        uint32_t baseN = A4W4_BASE_N_950;
        uint32_t baseK = 0;
        uint32_t ubCalSize = 0;
        uint32_t ubRestBytes = 0;
        uint32_t quantGroupNum = 1;
        uint32_t singleN = 0;
        uint32_t usedCoreNum = 1;
        uint32_t isPerTokenQuant = 0;
        uint32_t isA4W4Optimize = 0;
        int8_t groupType = 0;
        uint8_t groupListType = 0;
        uint32_t groupNum = 0;
        uint64_t singleCoreM = 1;
        uint64_t singleCoreN = 1;
        uint64_t singleCoreK = 1;
        uint64_t depthA1 = 1;
        uint64_t depthB1 = 1;
        uint64_t stepM = 1;
        uint64_t stepN = 1;
        uint64_t stepKa = 1;
        uint64_t stepKb = 1;
        uint32_t iterateOrder = 0;
        uint32_t dbL0c = 1;
    };
    A4W4BasicTiling a4w4Tiling_;
    // Shadows base's private tilingData_ (GMMQuantBasicApiTilingData); own type
    GroupedMatmulTilingData::GMMA4W4QuantTilingData tilingData_;
    uint64_t int8WeightWs_ = 0;
    uint64_t int8XWs_ = 0;      // ★ x int4->int8 预转换 workspace (m·k·1B, m=totalM single x)
    uint64_t mmOutWs_ = 0;
    // mList_/kList_/nList_ shadow base's private same-named (base private, derived inaccessible)
    int32_t mList_[GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t kList_[GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t nList_[GroupedMatmul::MAX_TENSOR_CONT] = {0};
};

}  // namespace optiling

#endif  // GROUPED_A4W4_QUANT_TILING_H
