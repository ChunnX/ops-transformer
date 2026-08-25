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
 * \file gather_pa_kv_cache_with_dequant_def.cpp
 * \brief Operator definition for fused GatherPaKvCache + Dequant
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {

class GatherPaKvCacheWithDequant : public OpDef {
public:
    explicit GatherPaKvCacheWithDequant(const char *name) : OpDef(name)
    {
        GatherPaKvCacheWithDequantDefFor950();
    }

private:
    void GatherPaKvCacheWithDequantDefFor950()
    {
        OpAICoreConfig aicoreConfig;
        
        // key_cache: INT8
        aicoreConfig.Input("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        // key_scale_cache: FLOAT
        aicoreConfig.Input("key_scale_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        // value_cache: FP8 E4M3FN
        aicoreConfig.Input("value_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        // value_scale: FLOAT
        aicoreConfig.Input("value_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        // block_tables: INT32/INT64
        aicoreConfig.Input("block_tables")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        // seq_lens: INT32/INT64
        aicoreConfig.Input("seq_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        // key: BF16 (ref input)
        aicoreConfig.Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        // value: BF16 (ref input)
        aicoreConfig.Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        // seq_offset: INT32/INT64 (optional)
        aicoreConfig.Input("seq_offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        // key output: BF16
        aicoreConfig.Output("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        // value output: BF16
        aicoreConfig.Output("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(GatherPaKvCacheWithDequant);

} // namespace ops
