/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mhc_pre_sinkhorn_backward_arch35_tiling_deterministic.h
 * \brief
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_DETERMINISTIC_H_
#define MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_DETERMINISTIC_H_

#include "op_host/tiling_base.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mhc/mhc_pre_sinkhorn_backward/op_kernel/arch35/mhc_pre_sinkhorn_backward_data_arch35.h"
#include "mhc/mhc_pre_sinkhorn_backward/op_kernel/arch35/mhc_pre_sinkhorn_backward_key_arch35.h"
#include "mhc_pre_sinkhorn_backward_arch35_tiling_base.h"
#include "err/ops_err.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;

class MhcPreSinkhornBackwardArch35DeterminiticTiling : public MhcPreSinkhornBackwardArch35Tiling {
public:
    explicit MhcPreSinkhornBackwardArch35DeterminiticTiling(gert::TilingContext* context)
        : MhcPreSinkhornBackwardArch35Tiling(context)
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    void DumpTilingInfo() override;
    ge::graphStatus DoLibApiTiling() override;

private:
    ge::graphStatus SetTilingData();
    void DoUbTiling();

private:
    // BS 分核参数
    int64_t blockFactorBS_{0};   // 普通核处理 BS 数
    int64_t blockTailBS_{0};     // 尾核 BS 数
    int64_t usedCoreNum_{0};     // 实际使用 AIV 核数
    // UB 因子
    int64_t ubFactorBS_{0};      // UB 单次处理 BS 数
    int64_t ubFactorC_{0};       // UB 单次处理 C 数 (=c 不切, <c 切)
    int64_t ubBlockSize_{0};     // UB block 字节数
};

} // namespace optiling
#endif // MHC_PRE_SINKHORN_BACKWARD_ARCH35_TILING_DETERMINISTIC_H_
