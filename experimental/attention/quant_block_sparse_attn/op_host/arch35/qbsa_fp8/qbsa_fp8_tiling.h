/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_FP8_TILING_H_
#define QBSA_FP8_TILING_H_

#include "../../tiling_common/qbsa_common_check.h"
#include "../../tiling_common/qbsa_tiling_strategy.h"

namespace optiling {
class QbsaFp8Arch35Tiling final : public QbsaTilingStrategy {
public:
    explicit QbsaFp8Arch35Tiling(gert::TilingContext *context);
    ~QbsaFp8Arch35Tiling() override = default;

private:
    ge::graphStatus CheckAttributes() const override;
    ge::graphStatus CheckTensorMetadata() const override;
    ge::graphStatus ParseShapeFacts(QbsaTilingContextParser &parser) override;
    ge::graphStatus CheckShapeRelations() const override;
    ge::graphStatus BuildTilingData() override;

    ge::graphStatus CheckFp8Attrs() const;
    ge::graphStatus CheckFp8Inputs() const;
    ge::graphStatus CheckFp8PaStride() const;
    ge::graphStatus CheckFp8ScaleDtype() const;
    ge::graphStatus CheckFp8BlockSize() const;
    ge::graphStatus CheckFp8QuantShape() const;

    QbsaCommonCheck commonCheck_;
};
} // namespace optiling

#endif // QBSA_FP8_TILING_H_
