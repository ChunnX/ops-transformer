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
 * \file msa_index_score_def.cpp
 * \brief MsaIndexScore 算子原型注册。
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class MsaIndexScore : public OpDef {
public:
    explicit MsaIndexScore(const char *name) : OpDef(name)
    {
        // index query, TND: [totalQ, numQHeads, headDim]
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // paged index key cache, BSND: [numBlocks, blockSize, numKvHeads, headDim]
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // [batch, maxBlocksPerBatch], 逻辑 block -> 物理 page
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // per-token per-head 正缩放, [totalQ, numQHeads]
        this->Input("scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // query 长度前缀和 (cumsum), [batch + 1]
        this->Input("actual_seq_lengths_query")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // 每个请求可见的 kv 长度, [batch]
        this->Input("actual_seq_lengths_key")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // 每个请求首个 query token 在自身 kv 序列中的位置, [batch]
        this->Input("start_loc")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // [numQHeads, totalQ, scoreBlockStride]
        this->Output("score")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        this->Attr("block_size").AttrType(OPTIONAL).Int(128);       // 128: 一个 sparse block / page 的 token 数
        this->Attr("fill_invalid_tail").AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};
OP_ADD(MsaIndexScore);
} // namespace ops
