/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv_tiling.h
 * \brief StemOamPrepPagedKV arch35 TilingData definition and Tiling class (split_mode=2)
 */
#ifndef STEM_OAM_PREP_PAGED_KV_TILING_ARCH35_H
#define STEM_OAM_PREP_PAGED_KV_TILING_ARCH35_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(StemPrepKVTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch_size);
TILING_DATA_FIELD_DEF(uint32_t, num_kv_heads);
TILING_DATA_FIELD_DEF(uint32_t, dim_qk);
TILING_DATA_FIELD_DEF(uint32_t, dim_v);
TILING_DATA_FIELD_DEF(uint32_t, stem_block_size);
TILING_DATA_FIELD_DEF(uint32_t, stem_stride);
TILING_DATA_FIELD_DEF(uint32_t, R_val);
TILING_DATA_FIELD_DEF(uint32_t, kflat_dim);
TILING_DATA_FIELD_DEF(uint32_t, kv_block_size);
TILING_DATA_FIELD_DEF(uint32_t, kv_layout);
TILING_DATA_FIELD_DEF(uint32_t, total_blocks);
TILING_DATA_FIELD_DEF(uint32_t, max_k_padded);
TILING_DATA_FIELD_DEF(uint32_t, max_Kb);
TILING_DATA_FIELD_DEF(uint32_t, max_kv_blocks_size);
TILING_DATA_FIELD_DEF(uint32_t, vbias_stride);
TILING_DATA_FIELD_DEF(float, lambda_mag);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(uint32_t, total_tasks);
TILING_DATA_FIELD_DEF(uint32_t, used_core_num);
TILING_DATA_FIELD_DEF(uint32_t, split_mode);
TILING_DATA_FIELD_DEF(uint32_t, cores_per_bh);
TILING_DATA_FIELD_DEF(uint32_t, total_stem_blocks);
TILING_DATA_FIELD_DEF(uint32_t, stem_blocks_per_core);
TILING_DATA_FIELD_DEF(uint32_t, kv_blocks_per_stem_block);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(StemOamPrepPagedKV, StemPrepKVTilingData)

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

private:
    uint64_t totalCoreNum_ = 0;
    uint64_t ubSize_ = 0;

    uint32_t batch_size_ = 0;
    uint32_t num_kv_heads_ = 0;
    uint32_t kv_layout_ = 0;
    uint32_t kv_block_size_ = 64;
    uint32_t stem_block_size_ = 128;
    uint32_t stem_stride_ = 16;
    float lambda_mag_ = 0.3f;
    uint32_t max_kv_len_ = 0;
    uint32_t max_k_padded_ = 0;
    uint32_t max_Kb_ = 0;
    uint32_t vbias_stride_ = 0;
    uint32_t total_blocks_ = 0;
    uint32_t max_kv_block_size_ = 0;
    uint32_t total_stem_blocks_ = 0;
    uint32_t stem_blocks_per_core_ = 0;
    uint32_t kv_blocks_per_stem_block_ = 0;
    uint32_t used_core_num_ = 0;
    uint32_t split_mode_ = 2;

    StemPrepKVTilingData tilingData_;
};

} // namespace optiling

#endif // STEM_OAM_PREP_PAGED_KV_TILING_ARCH35_H