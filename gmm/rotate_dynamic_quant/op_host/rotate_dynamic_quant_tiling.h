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
 * \brief Tiling数据结构定义
 */
#ifndef __OP_HOST_ROTATE_DYNAMIC_QUANT_TILING_H__
#define __OP_HOST_ROTATE_DYNAMIC_QUANT_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "../../../common/include/tiling_base/tiling_base.h"
#include "../../../common/include/err/ops_err.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RotateDynamicQuantTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint64_t, totalLength);
  TILING_DATA_FIELD_DEF(uint64_t, mDim);
  TILING_DATA_FIELD_DEF(uint64_t, nDim);
  TILING_DATA_FIELD_DEF(int64_t, dstType);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RotateDynamicQuant, RotateDynamicQuantTilingData);

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

class RotateDynamicQuantBaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RotateDynamicQuantBaseTiling(gert::TilingContext* context) : Ops::Transformer::OpTiling::TilingBaseClass(context) {};

    ~RotateDynamicQuantBaseTiling() override = default;

protected:
    bool IsCapable() override {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override {return ge::GRAPH_SUCCESS;};
    ge::graphStatus GetShapeAttrsInfo() override{return ge::GRAPH_SUCCESS;};
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override{return ge::GRAPH_SUCCESS;};
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override{return ge::GRAPH_SUCCESS;};
    ge::graphStatus PostTiling() override;

    ge::graphStatus GetInputShape();
    void PrintTilingData();
    ge::graphStatus ParseInputAndAttr();
    void FillTilingData();
    ge::graphStatus TilingProcess();

private:
    RotateDynamicQuantTilingData tilingData_;
    uint32_t blockDim_;
    uint64_t totalLength_;
    uint64_t mDim_;
    uint64_t nDim_;
    int64_t dstType_;
    size_t workspaceSize_;
};

} // namespace optiling

#endif // __OP_HOST_ROTATE_DYNAMIC_QUANT_TILING_H__
