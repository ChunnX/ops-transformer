/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* !
 * \file new_mc2_matmul_tiling_cfg.h
 * \brief
 */

#ifndef __NEW_MC2_MATMUL_TILING_CFG_H__
#define __NEW_MC2_MATMUL_TILING_CFG_H__

#pragma once
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_data.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_cfg.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "tiling/mc2_tiling_struct.h"

namespace Mc2MatmulHelper {
class NewMc2MatmulTilingCfg : public optiling::Mc2MatMulTilingCfg
{
public:
    NewMc2MatmulTilingCfg(const void* compileInfoIn, const void* argsIn, uint32_t baseMLimit = 0, bool needUpdateIn = true)
        : Mc2MatMulTilingCfg(needUpdateIn, compileInfoIn, argsIn), baseMLimit_(baseMLimit)
    {
    }

    void SetMatMulV3TilingData(optiling::MC2MatmulV3TilingData& tilingData);
    void Update(const optiling::Mc2TilingResult& result) override;
    void SetRankDim(uint64_t rankDim);
    void SetCommCnt(uint64_t commCnt);

private:
    void SetMMTilingData() const;
    void SetTailCntAndType() const;
    void DealBaseBlock() const;

    int32_t baseMLimit_{0};
    uint64_t rankDim_{0};
    uint64_t commCnt_{0};
    optiling::MC2MatmulV3TilingData* mc2MmV3TilingData_{nullptr};
    Mc2MatMulV3TilingData* mmv3TilingData_{nullptr};
};

}  // namespace Mc2MatmulHelper

#endif