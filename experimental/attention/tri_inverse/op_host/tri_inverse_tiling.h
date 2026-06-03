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
 * \file tri_inverse_tiling.h
 * \brief
 */

#ifndef TRI_INVERSE_TILING_H
#define TRI_INVERSE_TILING_H

#include <string>
#include "err/ops_err.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "../op_kernel/tri_inverse_tiling_data.h"

namespace optiling {

struct TriInverseCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t aicNum{0UL};
    uint64_t ubSize{0UL};
};

class TriInverseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit TriInverseTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    }
    ~TriInverseTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    void InitCompileInfo();
    void PrintTilingData() const;
    ge::graphStatus CheckContext() const;
    ge::graphStatus AnalyzeDtype() const;
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus AnalyzeFormat() const;
    ge::graphStatus DoMatmulTiling();
    ge::graphStatus SetScheduleConfig();
    bool CheckFormat(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;

    TriInverseCompileInfo compileInfo_;
    TriInverse::TriInverseTilingData tilingData_{};
    uint64_t tilingKey_{0};
    uint64_t workspaceSize_{0};
};

} // namespace optiling

#endif // TRI_INVERSE_TILING_H
