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
 * \file grouped_quant_basic_api_matmul_tiling.h
 * \brief
 */
#ifndef GROUPED_QUANT_BASIC_API_MATMUL_TILING_H
#define GROUPED_QUANT_BASIC_API_MATMUL_TILING_H

#include "grouped_quant_matmul_tiling.h"
namespace optiling {

class GroupedQmmBasicApiTiling : public GroupedQmmTiling {
public:
    explicit GroupedQmmBasicApiTiling(gert::TilingContext *context);
    ~GroupedQmmBasicApiTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Ops::Transformer::OpTiling::TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    virtual void Reset();
    ge::graphStatus CalL1Tiling();

private:
    ge::graphStatus CalL1Depth(uint64_t leftL1Size);
    uint64_t GetDepthWithHighBW(uint64_t mnL1) const;
    void ModifyDepthForUnalign(uint64_t leftL1Size, uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleABSize);
    ge::graphStatus CalScaleFactors();
    bool CanEnableThreeL1Buffer() const;
    GroupedMatmulTilingData::GMMQuantBasicApiTilingData tilingData_;
};

class GroupedS8S4BasicApiTiling : public GroupedQmmTiling {
public:
    explicit GroupedS8S4BasicApiTiling(gert::TilingContext *context);
    ~GroupedS8S4BasicApiTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Ops::Transformer::OpTiling::TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    bool AnalyzeDtype() override;
    bool AnalyzeAttrs() override;
    bool AnalyzeInputs() override;
    uint64_t GetTilingKey() const override;
    void Reset() override;

private:
    enum class DequantMode : uint8_t {
        SYMMETRIC_PER_GROUP = 0,
        ASYMMETRIC_PER_CHANNEL = 1,
    };

    bool CheckTensorLists() const;
    bool CheckCommonShapes(const gert::Shape &xShape, const gert::Shape &wShape) const;
    bool CheckBiasShape() const;
    bool CheckPerTokenScaleShape() const;
    bool CheckAntiquantInputsEmpty() const;
    bool CheckWeightStorageShape(const gert::StorageShape &weightShape);
    bool CheckScaleAndOffsetShapes();
    bool SetLogicalMkn(const gert::Shape &xShape, const gert::Shape &wShape);
    void SetBasicBlock();
    uint64_t GetRequiredL1Size(uint64_t kL1) const;
    ge::graphStatus ValidateFixedTileResources(uint64_t kL1) const;

    static constexpr uint32_t QUANT_GROUP_SIZE = 256U;
    static constexpr uint32_t S8S4_BASE_M = 256U;
    static constexpr uint32_t S8S4_ASYM_BASE_M = 256U;
    static constexpr uint32_t S8S4_BASE_N = 256U;
    static constexpr uint32_t S8S4_BASE_K = 128U;
    static constexpr uint32_t S8S4_PER_GROUP_K_L1 = 256U;
    static constexpr uint32_t S8S4_PER_CHANNEL_K_L1 = 256U;
    static constexpr uint64_t MAX_K_PER_GROUP = 18432UL;
    static constexpr uint64_t PACKED_INT32_RATIO = 8UL;
    static constexpr uint64_t NZ_BASE_K_ALIGN = 16UL;
    static constexpr uint64_t NZ_BASE_N_ALIGN = 64UL;
    static constexpr uint64_t NZ_SPECIAL_K_ALIGN = 64UL;
    static constexpr uint64_t NZ_SPECIAL_N_ALIGN = 16UL;
    static constexpr size_t TUNING_CONFIG_EXPECTED_TOKEN_INDEX = 0UL;
    static constexpr size_t TUNING_CONFIG_WEIGHT_FORMAT_INDEX = 1UL;
    static constexpr int64_t TUNING_CONFIG_SPECIAL_WEIGHT_FORMAT = 1L;
    static constexpr int64_t GROUP_LIST_CUMSUM = 0L;
    static constexpr int64_t GROUP_LIST_COUNT = 1L;
    static constexpr int64_t GROUP_LIST_SPARSE = 2L;
    static constexpr int8_t X_SEPARATED = 2;
    static constexpr int8_t NO_SEPARATED = 3;

    bool hasOffset_ = false;
    bool specialWeightFormat_ = false;
    bool weightPackedInt32_ = false;
    uint32_t expectedTokenNum_ = 0;
    uint64_t userWorkspaceSize_ = 0;
    DequantMode dequantMode_ = DequantMode::SYMMETRIC_PER_GROUP;
    GroupedMatmulTilingData::GMMS8S4BasicApiTilingData tilingData_;
};
} // namespace optiling

#endif // GROUPED_QUANT_BASIC_API_MATMUL_TILING_H
