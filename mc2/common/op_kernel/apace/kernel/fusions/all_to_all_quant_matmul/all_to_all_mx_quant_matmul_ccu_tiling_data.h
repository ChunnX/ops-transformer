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
 * \file all_to_all_mx_quant_matmul_ccu_tiling_data.h
 * \brief AllToAll MX Quant Matmul（CCU 版）——统一 tiling 聚合（通信 + 计算）
 *
 * 通信侧 CcuCommTilingData 来自 core/ccu_comm（通信服务契约）；
 * 计算侧 QuantMatmulTilingData 来自同目录共享件；经 <<<>>> 整体传参。
 */

#pragma once

#include "all_to_all_matmul_tiling_data.h"                        // QuantMatmulTilingData（共享件）
#include "apace/core/ccu_comm/all_to_all/ccu_all_to_all_comm.h"   // CcuCommTilingData（通信契约）

// 统一 tiling：通信切分 + 计算切分
#pragma pack(push, 8)
struct alignas(8) ccuAllToAllMatmulTilingData {
    CcuCommTilingData comm;
    QuantMatmulTilingData matmul;
};
#pragma pack(pop)
