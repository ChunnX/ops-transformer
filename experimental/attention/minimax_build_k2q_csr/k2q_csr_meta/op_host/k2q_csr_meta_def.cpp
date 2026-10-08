/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * K2qCsrMeta op definition.
 * Inputs: cu_seqlens, cu_block_lens, scratch(inplace)
 * Attrs: order_method, total_rows, max_kv, num_heads, num_tokens, topk
 */
#include "register/op_def.h"
#include "register/op_def_registry.h"

namespace ops {
namespace {
// 910b / 910_93 / 950 共用同一组编译与校验开关，仅 opFile.value 不同；
// 抽为文件内 helper，避免同一段链式配置在每个算子定义里重复。
inline void ApplyK2qCsrAICoreFlags(OpAICoreConfig &config)
{
    config.DynamicCompileStaticFlag(true);
    config.DynamicFormatFlag(false);
    config.DynamicRankSupportFlag(true);
    config.DynamicShapeSupportFlag(true);
    config.NeedCheckSupportFlag(false);
    config.PrecisionReduceFlag(true);
}
}  // namespace
class K2qCsrMeta : public OpDef {
public:
    explicit K2qCsrMeta(const char *name) : OpDef(name)
    {
        this->Input("cu_seqlens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_block_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("scratch")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("scratch")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("order_method").AttrType(REQUIRED).Int(0);
        this->Attr("total_rows").AttrType(REQUIRED).Int(0);
        this->Attr("max_kv").AttrType(REQUIRED).Int(0);
        this->Attr("num_heads").AttrType(REQUIRED).Int(0);
        this->Attr("num_tokens").AttrType(REQUIRED).Int(0);
        this->Attr("topk").AttrType(REQUIRED).Int(0);

        OpAICoreConfig aicoreConfig;
        ApplyK2qCsrAICoreFlags(aicoreConfig);
        aicoreConfig.ExtendCfgInfo("opFile.value", "k2q_csr_meta");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);

        OpAICoreConfig a5Config;
        ApplyK2qCsrAICoreFlags(a5Config);
        a5Config.ExtendCfgInfo("opFile.value", "k2q_csr_meta_apt");
        this->AICore().AddConfig("ascend950", a5Config);
    }
};
OP_ADD(K2qCsrMeta);
} // namespace ops
