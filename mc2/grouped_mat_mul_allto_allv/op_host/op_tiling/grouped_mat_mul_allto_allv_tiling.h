/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_mat_mul_allto_allv_tiling.h
 * \brief 声明 GroupedMatmulAllToAllvTiling 切分（tiling）类，继承自量化版本的公共切分类，
 *        通过重写各虚函数实现本算子特有的参数校验、本地参数设置与 tiling key 生成逻辑。
 */
#ifndef MC2_GROUPED_MATMUL_ALLTO_ALLV_TILING_STRUCT_H
#define MC2_GROUPED_MATMUL_ALLTO_ALLV_TILING_STRUCT_H

#include "../../quant_grouped_mat_mul_allto_allv/op_host/op_tiling/arch35/quant_grouped_mat_mul_allto_allv_tiling_common.h"

namespace optiling {
// 继承自量化公共切分类，重写虚函数实现非量化场景的切分逻辑
class GroupedMatmulAllToAllvTiling : public Mc2Tiling::Mc2GroupedMatmul::QuantGroupedMatmulAllToAllvTilingCommon {
public:
    explicit GroupedMatmulAllToAllvTiling(gert::TilingContext *context)
        : Mc2Tiling::Mc2GroupedMatmul::QuantGroupedMatmulAllToAllvTilingCommon(context) {};

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    // send/recv counts tensor 当前不支持
    ge::graphStatus CheckOpInputSingleParamsTensorNotSupport() override;
    ge::graphStatus CheckOpInputSingleParamsTensorSupport() override;
    // 共享专家 mm 相关的单参数 tensor 输入校验
    ge::graphStatus CheckOpInputSingleParamsTensorMM() override;
    ge::graphStatus CheckAndSetLocalParamsGmm() override;
    ge::graphStatus CheckAndSetLocalParamsMm() override;
    ge::graphStatus CheckAndSetLocalParamsAttr() override;
    ge::graphStatus CheckAndSetLocalParams() override;
    ge::graphStatus CheckParamsRelationGmm() override;
    ge::graphStatus CheckParamsRelationMm() override;
    ge::graphStatus CheckParamsAttrEpAndSetLocalParams() override;
    ge::graphStatus GetPlatformInfo() override;
    // 计算 tiling key，用于编译期选择对应的 kernel 模板
    uint64_t GetTilingKey() const override;
    uint32_t GetCommModeIndex() const override;
    ge::graphStatus GetAndConvertCommMode(gert::TilingContext *context, uint8_t &commMode) const;

    // 输入索引
    static constexpr uint32_t GMM_X_INDEX = 0;
    static constexpr uint32_t GMM_WEIGHT_INDEX = 1;
    static constexpr uint32_t SEND_COUNTS_TENSOR_OPTIONAL_INDEX = 2;
    static constexpr uint32_t RECV_COUNTS_TENSOR_OPTIONAL_INDEX = 3;
    static constexpr uint32_t MM_X_OPTIONAL_INDEX = 4;
    static constexpr uint32_t MM_WEIGHT_OPTIONAL_INDEX = 5;
    // 输出索引
    static constexpr uint32_t OUTPUT_Y_INDEX = 0;
    static constexpr uint32_t OUTPUT_MM_Y_OPTIONAL_INDEX = 1;
    // 属性索引
    static constexpr uint32_t ATTR_GROUP_INDEX = 0;
    static constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;
    static constexpr uint32_t ATTR_SEND_COUNTS_INDEX = 2;
    static constexpr uint32_t ATTR_RECV_COUNTS_INDEX = 3;
    static constexpr uint32_t ATTR_TRANS_GMM_WEIGHT_INDEX = 4;
    static constexpr uint32_t ATTR_TRANS_MM_WEIGHT_INDEX = 5;
    static constexpr uint32_t ATTR_COMM_MODE = 6;
    // 卡数阈值：卡数 <= 该值时 arch35 平台默认使用 CCU 通信模式
    static constexpr int64_t RANK_DIM_BOUNDARY = 8;

    static const std::vector<uint32_t> GMM_X_DTYPE_LIST;        // fp16/bf16
    static const std::vector<uint32_t> GMM_WEIGHT_DTYPE_LIST;   // fp16/bf16
    static const std::vector<uint32_t> GMM_Y_DTYPE_LIST;        // fp16/bf16
    static const std::set<int64_t> A5_SUPPORT_RANK_SIZE;        // Ascend 950 支持的卡数集合
    static const std::set<int64_t> A3_SUPPORT_RANK_SIZE;        // Atlas A3 支持的卡数集合
};
} // namespace optiling
#endif
