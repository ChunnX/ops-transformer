/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_MXFP8_TILING_H_
#define QBSA_MXFP8_TILING_H_

#include "../../tiling_common/qbsa_common_check.h"
#include "../../tiling_common/qbsa_tiling_strategy.h"
#include "../../../op_kernel/arch35/qbsa_mxfp8/qbsa_mxfp8_tiling_data.h"

namespace optiling {
class QbsaMxfp8Arch35Tiling final : public QbsaTilingStrategy {
public:
    explicit QbsaMxfp8Arch35Tiling(gert::TilingContext *context);
    ~QbsaMxfp8Arch35Tiling() override = default;

private:
    ge::graphStatus CheckAttributes() const override;
    ge::graphStatus CheckTensorMetadata() const override;
    ge::graphStatus ParseShapeFacts(QbsaTilingContextParser &parser) override;
    ge::graphStatus CheckShapeRelations() const override;
    ge::graphStatus BuildTilingData() override;
    ge::graphStatus SaveTilingData() override;

    ge::graphStatus CheckMxAttrs() const;
    ge::graphStatus CheckMxScaleDtype() const;
    ge::graphStatus CheckMxSparseBlockSize() const;
    ge::graphStatus CheckMxPaBlockSize() const;
    ge::graphStatus CheckMxfp8FullQuantShape() const;

    QbsaCommonCheck commonCheck_;
    QuantBlockSparseAttnMxfp8TilingData tilingData_;
};
} // namespace optiling

#endif // QBSA_MXFP8_TILING_H_
