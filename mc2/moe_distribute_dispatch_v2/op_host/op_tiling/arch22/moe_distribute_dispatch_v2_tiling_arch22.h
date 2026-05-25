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
 * \file moe_distribute_dispatch_v2_tiling_arch22.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_TILING_ARCH22_H
#define MOE_DISTRIBUTE_DISPATCH_V2_TILING_ARCH22_H

#include "op_host/op_tiling/moe_tiling_base.h"
#include "../moe_distribute_dispatch_tiling_helper.h"
#include "../moe_distribute_dispatch_tiling_v2.h"

namespace optiling {

// A2A3 类保留供 A2 和 V3 使用（A2 仍继承基类，V3 仍创建 A2A3 对象调用旧入口）
class MoeDistributeDispatchV2TilingFuncA2A3 : public MoeDistributeDispatchV2TilingFuncBase {
public:
    ge::graphStatus MoeDistributeDispatchV2TilingFunc(gert::TilingContext* context) override;
    ge::graphStatus MoeDistributeDispatchTilingFuncImpl(gert::TilingContext* context) override;
    bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName,
        DispatchV2TilingCtx &tc, DispatchV2Config &config) override;
    uint64_t CalTilingKey(const DispatchV2TilingCtx &tc) override;
    ge::graphStatus CheckQuantModeMatchScales(gert::TilingContext *context, const char *nodeName,
        DispatchV2TilingCtx &tc, DispatchV2Config &config) override;
    ge::graphStatus CheckCommAlgPtr(const char* commAlgPtr, const char *nodeName) override;
    ge::graphStatus CheckQuantModePtr(const int64_t* quantModePtr, const char *nodeName) override;
};

// A3 独立类：不继承基类（原则5：架构差异通过参数/回调传递）
class MoeDistributeDispatchV2TilingFuncA3 {
public:
    ge::graphStatus MoeDistributeDispatchV2TilingFunc(gert::TilingContext* context);
    bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName,
        DispatchV2TilingCtx &tc, DispatchV2Config &config);
    uint64_t CalTilingKey(const DispatchV2TilingCtx &tc);
    ge::graphStatus CheckQuantModeMatchScales(gert::TilingContext *context, const char *nodeName,
        DispatchV2TilingCtx &tc, DispatchV2Config &config);
    ge::graphStatus CheckCommAlgPtr(const char* commAlgPtr, const char *nodeName);
    ge::graphStatus CheckQuantModePtr(const int64_t* quantModePtr, const char *nodeName);
};
} // namespace optiling

#endif // MOE_DISTRIBUTE_DISPATCH_TILING_ARCH22_H