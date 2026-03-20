/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotate_dynamic_quant_tiling.h
 * \brief RotateDynamicQuant Tiling数据结构定义
 */
#ifndef __OP_HOST_ROTATE_DYNAMIC_QUANT_TILING_H__
#define __OP_HOST_ROTATE_DYNAMIC_QUANT_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "../../../common/include/tiling_base/tiling_base.h"
#include "../../../common/include/err/ops_err.h"

namespace optiling {

// Tiling 数据结构定义
BEGIN_TILING_DATA_DEF(RotateDynamicQuantTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);      // 使用的核数
  TILING_DATA_FIELD_DEF(uint64_t, totalLength);  // 总数据长度
  TILING_DATA_FIELD_DEF(uint64_t, M);            // 输入矩阵 M 维度
  TILING_DATA_FIELD_DEF(uint64_t, N);            // 输入矩阵 N 维度
  TILING_DATA_FIELD_DEF(uint32_t, dstType);      // 输出数据类型: 0=INT8, 1=INT32
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RotateDynamicQuant, RotateDynamicQuantTilingData);

// CompileInfo 结构体，用于存储编译时平台信息
struct RotateDynamicQuantCompileInfo {
    uint64_t aicNum{0UL};
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    uint64_t l0CSize{0UL};
    uint64_t l0ASize{0UL};
    uint64_t l0BSize{0UL};
};

/**
 * @brief RotateDynamicQuant Tiling 基类
 */
class RotateDynamicQuantBaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RotateDynamicQuantBaseTiling(gert::TilingContext* context) : Ops::Transformer::OpTiling::TilingBaseClass(context) {};

    ~RotateDynamicQuantBaseTiling() override = default;

protected:
    bool IsCapable() override {
        return true;
    }

    // 1、获取平台信息
    ge::graphStatus GetPlatformInfo() override {return ge::GRAPH_SUCCESS;};

    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override{return ge::GRAPH_SUCCESS;};

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;

    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override{return ge::GRAPH_SUCCESS;};

    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override{return ge::GRAPH_SUCCESS;};

    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    // 辅助方法
    void PrintTilingData();
    ge::graphStatus ParseInputAndAttr();
    void FillTilingData();
    ge::graphStatus TilingProcess();

private:
    RotateDynamicQuantTilingData tilingData_;
    uint32_t blockDim_;         // AIC 核数
    uint64_t totalLength_;      // 总数据长度
    uint64_t M_;                // M 维度
    uint64_t N_;                // N 维度
    uint32_t dstType_;          // 输出数据类型
    size_t workspaceSize_;      // workspace 大小
    uint64_t tilingKey_;        // tiling key
};

} // namespace optiling

#endif // __OP_HOST_ROTATE_DYNAMIC_QUANT_TILING_H__
