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
 * \file kvquant_sparse_attn_sharedkv_tiling.h
 * \brief
 */

#ifndef kv_quant_sparse_flash_mla_TILING_H_
#define kv_quant_sparse_flash_mla_TILING_H_

#include "err/ops_err.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "platform/soc_spec.h"
#include "../../kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.h"
namespace optiling {
}  // namespace optiling
#endif  // kvquant_sparse_flash_mla_TILING_H_