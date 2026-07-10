/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv_tiling.h
 * \brief StemOamPrepPagedKV simd TilingData definition and Tiling class
 */
#ifndef STEM_OAM_PREP_PAGED_KV_TILING_SIMD_H
#define STEM_OAM_PREP_PAGED_KV_TILING_SIMD_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base.h"
#include "op_kernel/arch35/stem_oam_prep_paged_kv_tiling_data.h"

namespace optiling {

constexpr uint64_t MAX_KV_PADDED = 262144;

struct StemPrepKVCompileInfo {
    uint32_t coreNum = 0;
    uint64_t ubSize = 0;
};

class StemPrepKVTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit StemPrepKVTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
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
    void DumpTilingInfo() override;

private:
    uint64_t totalCoreNum_ = 0;

    uint32_t batchSize_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t kvLayout_ = 0;
    uint32_t kvBlockSize_ = 64;
    uint32_t stemBlockSize_ = 128;
    uint32_t stemStride_ = 16;
    float lambdaMag_ = 0.3f;
    uint32_t maxKvBlockSize_ = 0;
    uint32_t dimQk_ = 128;
    uint32_t maxKb_ = 0;
    uint32_t kflatDim_ = 0;
    uint32_t rVal_ = 0;

    gert::Stride kCacheStride_;
    gert::Stride vCacheStride_;
    gert::Stride kScaleCacheStride_;

    StemPrepKVTilingData* tilingData_;
};

} // namespace optiling

#endif // STEM_OAM_PREP_PAGED_KV_TILING_SIMD_H
