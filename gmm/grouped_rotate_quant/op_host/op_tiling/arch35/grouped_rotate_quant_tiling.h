/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GROUPED_ROTATE_QUANT_TILING_H
#define GROUPED_ROTATE_QUANT_TILING_H

#include "op_host/tiling_base.h"
#include "grouped_rotate_quant_planner.h"

namespace optiling {
struct GroupedRotateQuantCompileInfo {
    grq::Hardware hardware;
    int32_t npuArch = 0;
};

class GroupedRotateQuantTiling950 : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GroupedRotateQuantTiling950(gert::TilingContext *context) : TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }
    uint64_t GetTilingKey() const override { return GroupedRotateQuant::TILING_KEY; }
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    GroupedRotateQuantCompileInfo compileInfo_{};
    grq::InputParams inputParams_;
    GroupedRotateQuant::TilingData tilingData_{};
};
} // namespace optiling
#endif
