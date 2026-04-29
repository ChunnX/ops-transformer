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
 * \file allto_all_fp_matmul_tiling_base.h
 * \brief
 */

#ifndef ALLTO_ALL_FP_MATMUL_TILING_BASE_H
#define ALLTO_ALL_FP_MATMUL_TILING_BASE_H

#include "securec.h"
#include "mc2_matmul_tiling_cfg.h"
#include "op_host/op_tiling/new_mc2_tiling_utils.h"
#include "../allto_all_matmul_tiling_base.h"
#include "../../../op_kernel/arch35/allto_all_matmul_tiling_data.h"
#include "../../../op_kernel/arch35/allto_all_matmul_tiling_key.h"
#include "mc2/matmul_allto_all/op_host/op_tiling/common/matmul_allto_all_util_tiling.h"

namespace MC2Tiling {

using namespace optiling;

class AllToAllFpMatmulTilingBase : public AllToAllMatmulTilingBase {
public:
    explicit AllToAllFpMatmulTilingBase(gert::TilingContext *context);
    ~AllToAllFpMatmulTilingBase() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    ge::graphStatus CheckOpInputInfo();
    ge::graphStatus InitTilingContextParameters();
    ge::graphStatus DoMMTiling();
    ge::graphStatus SetHcclTiling();
    void SetTilingInfo(AlltoAllMatmulTilingInfo &tilingInfo) const;

private:
    void PrintAlltoAllMatmulTilingData(AlltoAllMatmulTilingData &alltoAllMatmulTilingData);
    void PrintAlltoAllMatmulTilingInfo(const std::string &opName, AlltoAllMatmulTilingInfo &tilingInfo);

    AlltoAllMatmulTilingData localTilingData_;
};

} // namespace MC2Tiling
#endif
