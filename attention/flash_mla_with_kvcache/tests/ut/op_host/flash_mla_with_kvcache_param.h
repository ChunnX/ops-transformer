/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_WITH_KVCACHE_PARAM_H
#define FLASH_MLA_WITH_KVCACHE_PARAM_H

#include <string>
#include <vector>
#include <sstream>
#include "op_host_csv_case_loader.h"
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"
#include "../../../op_host/flash_mla_with_kvcache_tiling_common.h"

namespace FlashMlaWithKvcacheUT {

using optiling::flash_mla_with_kvcache::FlashMlaWithKvcacheCompileInfo;

struct FlashMlaWithKvcacheHostUtParamBase : public HostUtParamBase {
    // m0070 attr set, order matches flash_mla_with_kvcache_def.cpp registration order
    int64_t head_dim_v = 512;
    float softmax_scale = 0.0f;
    int64_t mask_mode = 0;
    int64_t max_seqlen_q = -1;
    int64_t max_seqlen_kv = -1;
    std::string layout_q = "TND";
    std::string layout_kv = "NZ";
    std::string layout_out = "NTD";
    int64_t return_softmax_lse = 0;

    FlashMlaWithKvcacheHostUtParamBase(const csv_map &csvMap) : HostUtParamBase(csvMap)
    {
        head_dim_v = std::stoll(ReadMap(csvMap, "head_dim_v", "512"));
        softmax_scale = std::stof(ReadMap(csvMap, "softmax_scale", "0.0"));
        mask_mode = std::stoll(ReadMap(csvMap, "mask_mode", "0"));
        max_seqlen_q = std::stoll(ReadMap(csvMap, "max_seqlen_q", "-1"));
        max_seqlen_kv = std::stoll(ReadMap(csvMap, "max_seqlen_kv", "-1"));
        layout_q = ReadMap(csvMap, "layout_q", "TND");
        layout_kv = ReadMap(csvMap, "layout_kv", "NZ");
        layout_out = ReadMap(csvMap, "layout_out", "NTD");
        return_softmax_lse = std::stoll(ReadMap(csvMap, "return_softmax_lse", "0"));
    }

    // dedicated TEST() cases build params in code; an empty csv map yields all defaults
    FlashMlaWithKvcacheHostUtParamBase() : FlashMlaWithKvcacheHostUtParamBase(csv_map{})
    {
    }
};

struct FlashMlaWithKvcacheTilingUtParam : public FlashMlaWithKvcacheHostUtParamBase {
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;

    // m0070 input set (order matches flash_mla_with_kvcache_def.cpp / parser indexes):
    // q, k_cache, block_table, cache_seqlens, cu_seqlens_q, seqused_q, attn_mask, metadata
    gert::TilingContextPara::TensorDescription q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription k_cache = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription block_table = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cache_seqlens = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cu_seqlens_q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription seqused_q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription attn_mask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription metadata = TD_DEFAULT;

    gert::TilingContextPara::TensorDescription attn_out = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription softmax_lse = TD_DEFAULT;

    uint64_t expectTilingKey = 0;
    std::string expectTilingDataHash;
    std::vector<size_t> expectWorkspaces;

    FlashMlaWithKvcacheCompileInfo compileInfo = {
        64, 32, 65536, 1048576, 32768, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};

    FlashMlaWithKvcacheTilingUtParam() : FlashMlaWithKvcacheHostUtParamBase(csv_map{})
    {
    }

    FlashMlaWithKvcacheTilingUtParam(const csv_map &csvMap) : FlashMlaWithKvcacheHostUtParamBase(csvMap)
    {
        inputInstance.emplace_back(GetTensorGE(csvMap, "q_shape", "q_dtype", "q_format", this->q));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "k_cache_shape", "k_cache_dtype", "k_cache_format", this->k_cache));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "block_table_shape", "block_table_dtype", "block_table_format", this->block_table));
        inputInstance.emplace_back(GetTensorGE(csvMap, "cache_seqlens_shape", "cache_seqlens_dtype",
                                               "cache_seqlens_format", this->cache_seqlens));
        inputInstance.emplace_back(GetTensorGE(csvMap, "cu_seqlens_q_shape", "cu_seqlens_q_dtype",
                                               "cu_seqlens_q_format", this->cu_seqlens_q));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "seqused_q_shape", "seqused_q_dtype", "seqused_q_format", this->seqused_q));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_mask_shape", "attn_mask_dtype", "attn_mask_format", this->attn_mask));
        inputInstance.emplace_back(
            GetTensorGE(csvMap, "metadata_shape", "metadata_dtype", "metadata_format", this->metadata));

        outputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_out_shape", "attn_out_dtype", "attn_out_format", this->attn_out));
        outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", this->softmax_lse));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            expectTilingKey = std::stoull(ReadMap(csvMap, "expectTilingKey", "0"));
            expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash", "");
            std::string wsStr = ReadMap(csvMap, "expectWorkspaceSize", "");
            if (!wsStr.empty()) {
                expectWorkspaces.push_back(static_cast<size_t>(std::stoull(wsStr)));
            }
        }
    }
};

struct FlashMlaWithKvcacheInferShapeUtParam : public FlashMlaWithKvcacheHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription k_cache = ID_DEFAULT;

    gert::InfershapeContextPara::TensorDescription attn_out = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription softmax_lse = ID_DEFAULT;

    std::vector<std::vector<int64_t>> expectOutputShape;

    FlashMlaWithKvcacheInferShapeUtParam(const csv_map &csvMap) : FlashMlaWithKvcacheHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "q_shape", "q_dtype", "q_format", this->q));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "k_cache_shape", "k_cache_dtype", "k_cache_format", this->k_cache));

        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_out_shape", "attn_out_dtype", "attn_out_format", this->attn_out));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", this->softmax_lse));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            std::string shapeStr = ReadMap(csvMap, "expect_attn_out_shape", "");
            if (!shapeStr.empty()) {
                std::vector<int64_t> shape;
                std::stringstream ss(shapeStr);
                int64_t dim;
                while (ss >> dim) {
                    shape.push_back(dim);
                }
                expectOutputShape.push_back(shape);
            }
            shapeStr = ReadMap(csvMap, "expect_softmax_lse_shape", "");
            if (!shapeStr.empty()) {
                std::vector<int64_t> shape;
                std::stringstream ss(shapeStr);
                int64_t dim;
                while (ss >> dim) {
                    shape.push_back(dim);
                }
                expectOutputShape.push_back(shape);
            }
        }
    }
};

struct FlashMlaWithKvcacheInferDTypeUtParam : public FlashMlaWithKvcacheHostUtParamBase {
    ge::DataType q_dtype = ge::DT_UNDEFINED;
    ge::DataType k_cache_dtype = ge::DT_UNDEFINED;

    ge::DataType expect_attn_out_dtype = ge::DT_UNDEFINED;
    ge::DataType expect_softmax_lse_dtype = ge::DT_UNDEFINED;

    FlashMlaWithKvcacheInferDTypeUtParam(const csv_map &csvMap) : FlashMlaWithKvcacheHostUtParamBase(csvMap)
    {
        GetDataTypeGE(csvMap, "q_dtype", this->q_dtype);
        GetDataTypeGE(csvMap, "k_cache_dtype", this->k_cache_dtype);

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            GetDataTypeGE(csvMap, "expect_attn_out_dtype", this->expect_attn_out_dtype);
            GetDataTypeGE(csvMap, "expect_softmax_lse_dtype", this->expect_softmax_lse_dtype);
        }
    }
};

} // namespace FlashMlaWithKvcacheUT

#endif // FLASH_MLA_WITH_KVCACHE_PARAM_H