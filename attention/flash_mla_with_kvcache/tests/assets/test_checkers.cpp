/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cassert>
#include "attention/flash_mla_with_kvcache/op_host/checkers/softmax_lse_checker.h"
#include "attention/flash_mla_with_kvcache/op_host/checkers/flash_mla_with_kvcache_checker.h"
#include <iostream>
#include "attention/flash_mla_with_kvcache/op_host/checkers/paged_attention_checker.h"
#include "attention/flash_mla_with_kvcache/op_host/checkers/sinks_checker.h"
#include "attention/flash_mla_with_kvcache/op_host/flash_mla_with_kvcache_tiling_info_parser.h"
#include "attention/flash_mla_with_kvcache/op_host/checkers/common_checker.h"
#include "attention/flash_mla_with_kvcache/op_host/checkers/mask_checker.h"
#include "attention/flash_mla_with_kvcache_metadata/op_host/flash_mla_with_kvcache_metadata_check.h"
using namespace optiling::flash_mla_with_kvcache;

void TestCompositeWithValidInputs()
{
    using L = FlashMlaWithKvcacheLayout;
    FlashMlaWithKvcacheTilingInfo info{};
    info.opName = "CompositeCheckerRegression";
    info.qLayout = L::TND;
    info.kvLayout = L::PA_NZ;
    info.outLayout = L::NTD;
    info.bSize = 2;
    info.n1Size = 64;
    info.n2Size = 1;
    info.qTSize = info.s1Size = 8;
    info.s2Size = info.blockSize = 128;
    info.totalBlockNum = 1;
    info.qkHeadDim = info.vHeadDim = 512;
    info.pageAttentionFlag = true;
    info.maxSeqQ = info.maxSeqKv = -1;
    info.maskMode = 0;
    info.winLeft = info.winRight = -1;
    info.softmaxLseFlag = true;
    int64_t headDimV = 512;
    int64_t enabled = 1;
    info.opParamInfo.headDimV = &headDimV;
    info.opParamInfo.returnSoftMaxLse = &enabled;
    gert::StorageShape q({8, 64, 576}, {8, 64, 576});
    gert::StorageShape k({1, 1, 36, 128, 16}, {1, 1, 36, 128, 16});
    gert::StorageShape out({64, 8, 512}, {64, 8, 512});
    gert::StorageShape lse({64, 8}, {64, 8});
    gert::CompileTimeTensorDesc halfDesc, intDesc, floatDesc;
    for (auto desc : {&halfDesc, &intDesc, &floatDesc}) {
        desc->SetOriginFormat(ge::FORMAT_ND);
    }
    halfDesc.SetDataType(ge::DT_FLOAT16);
    intDesc.SetDataType(ge::DT_INT32);
    floatDesc.SetDataType(ge::DT_FLOAT);
    info.opParamInfo.query.shape = &q;
    info.opParamInfo.query.desc = &halfDesc;
    info.opParamInfo.kCache.shape = &k;
    info.opParamInfo.kCache.desc = &halfDesc;
    info.opParamInfo.attnOut.shape = &out;
    info.opParamInfo.attnOut.desc = &halfDesc;
    info.opParamInfo.lseOut.shape = &lse;
    info.opParamInfo.lseOut.desc = &floatDesc;
    gert::Tensor table, cache, cu, metadata;
    table.MutableStorageShape() = gert::Shape({2, 1});
    cache.MutableStorageShape() = gert::Shape({2});
    cu.MutableStorageShape() = gert::Shape({3});
    metadata.MutableStorageShape() = gert::Shape({4096});
    info.opParamInfo.blockTable.tensor = &table;
    info.opParamInfo.blockTable.desc = &intDesc;
    info.opParamInfo.cacheSeqlens.tensor = &cache;
    info.opParamInfo.cacheSeqlens.desc = &intDesc;
    info.opParamInfo.cuSeqlensQ.tensor = &cu;
    info.opParamInfo.cuSeqlensQ.desc = &intDesc;
    info.opParamInfo.metadata.tensor = &metadata;
    info.opParamInfo.metadata.desc = &intDesc;
    FlashMlaWithKvcacheChecker composite;
    composite.Init(info);
    assert(composite.Process(info) == ge::GRAPH_SUCCESS);
    lse.MutableStorageShape() = gert::Shape({1});
    assert(composite.Process(info) == ge::GRAPH_FAILED);
    lse.MutableStorageShape() = gert::Shape({64, 8});
    info.opParamInfo.lseOut.desc = nullptr;
    assert(composite.Process(info) == ge::GRAPH_FAILED);
    info.opParamInfo.lseOut.desc = &floatDesc;
    info.opParamInfo.query.desc = nullptr;
    assert(composite.Process(info) == ge::GRAPH_FAILED);
    std::cout << "Composite checker accepts valid input and rejects malformed LSE/missing descriptors.\n";
}

int main()
{
    TestCompositeWithValidInputs();
    using L = FlashMlaWithKvcacheLayout;
    CommonChecker common;
    MaskChecker mask;
    FlashMlaWithKvcacheTilingInfo info{};
    info.opName = "FlashMlaCheckerRegression";
    int accepted = 0, rejected = 0;
    for (auto q : {L::BSND, L::BNSD, L::TND, L::NTD}) {
        for (auto kv : {L::PA_BNBD, L::PA_BBND, L::PA_NZ, L::BNSD}) {
            for (auto out : {L::BSND, L::BNSD, L::TND, L::NTD}) {
                info.qLayout = q;
                info.kvLayout = kv;
                info.outLayout = out;
                const bool expected = q != L::NTD && kv != L::BNSD && (out == q || (q == L::TND && out == L::NTD));
                const bool actual = common.CheckSinglePara(info) == ge::GRAPH_SUCCESS &&
                                    common.CheckMultiParaLayout(info) == ge::GRAPH_SUCCESS;
                assert(actual == expected);
                expected ? ++accepted : ++rejected;
            }
        }
    }
    for (int mode : {-1, 0, 1, 2, 3, 4, 5}) {
        info.maskMode = mode;
        const bool expected = mode == 0 || mode == 3 || mode == 4;
        assert((mask.CheckSinglePara(info) == ge::GRAPH_SUCCESS) == expected);
        assert((FlashMlaWithKvcacheMetadataCheck::CheckMask(mode, -1, -1) == ACLNN_SUCCESS) == expected);
    }
    for (auto q : {"BSND", "BNSD", "TND", "NTD", "BAD"}) {
        const bool expected = std::string(q) != "NTD" && std::string(q) != "BAD";
        assert((FlashMlaWithKvcacheMetadataCheck::CheckBaseAttr(-1, -1, 64, 1, 576, 512, q) == ACLNN_SUCCESS) ==
               expected);
    }
    assert(FlashMlaWithKvcacheMetadataCheck::CheckMask(4, 0, 0) == ACLNN_SUCCESS);
    assert(FlashMlaWithKvcacheMetadataCheck::CheckMask(4, INT32_MAX, INT32_MAX) == ACLNN_SUCCESS);
    assert(FlashMlaWithKvcacheMetadataCheck::CheckMask(4, -2, 0) != ACLNN_SUCCESS);
    assert(FlashMlaWithKvcacheMetadataCheck::CheckMask(4, 0, int64_t(INT32_MAX) + 1) != ACLNN_SUCCESS);
    assert(FlashMlaWithKvcacheMetadataCheck::CheckMask(0, 0, 0) != ACLNN_SUCCESS);
    assert(FlashMlaWithKvcacheMetadataCheck::CheckMask(3, 0, 0) != ACLNN_SUCCESS);

    // Exhaust the new head interval and the adjacent rejected values.
    gert::StorageShape qShape({1, 1, 64, 576}, {1, 1, 64, 576});
    gert::StorageShape kShape({1, 128, 1, 576}, {1, 128, 1, 576});
    info.opParamInfo.query.shape = &qShape;
    info.opParamInfo.kCache.shape = &kShape;
    info.qLayout = L::BSND;
    info.bSize = info.n2Size = info.s1Size = info.s2Size = 1;
    for (int heads = -1; heads <= 129; ++heads) {
        info.n1Size = heads;
        const bool expected = heads >= 1 && heads <= 128;
        assert((common.CheckAxis(info) == ge::GRAPH_SUCCESS) == expected);
        assert((FlashMlaWithKvcacheMetadataCheck::CheckBaseAttr(-1, -1, heads, 1, 576, 512, "BSND") == ACLNN_SUCCESS) ==
               expected);
    }
    info.n1Size = 64;
    gert::Tensor table;
    table.MutableStorageShape() = gert::Shape({1, 8});
    gert::CompileTimeTensorDesc tableDesc;
    tableDesc.SetDataType(ge::DT_INT32);
    tableDesc.SetOriginFormat(ge::FORMAT_ND);
    info.opParamInfo.blockTable.tensor = &table;
    info.opParamInfo.blockTable.desc = &tableDesc;
    info.pageAttentionFlag = true;
    PagedAttentionChecker paged;
    int validBlocks = 0;
    for (int size = 0; size <= 1040; ++size) {
        info.blockSize = size;
        const bool expected = size >= 16 && size <= 1024 && size % 16 == 0;
        assert((paged.CheckSinglePara(info) == ge::GRAPH_SUCCESS) == expected);
        validBlocks += expected;
    }
    assert(validBlocks == 64);

    // Every Q/K axis, including the five-dimensional PA_NZ storage shape.
    FlashMlaWithKvcacheInfoParser parser(nullptr);
    parser.opName_ = "EmptyTensorRegression";
    parser.opParamInfo_.query.shape = &qShape;
    parser.opParamInfo_.kCache.shape = &kShape;
    for (auto dims : {gert::Shape({1, 64, 576}), gert::Shape({1, 1, 64, 576}), gert::Shape({1, 64, 1, 576})}) {
        qShape.MutableStorageShape() = dims;
        for (size_t axis = 0; axis < dims.GetDimNum(); ++axis) {
            qShape.MutableStorageShape().SetDim(axis, 0);
            parser.emptyTensorFlag_ = false;
            assert(parser.GetEmptyTensorFlag() == ge::GRAPH_FAILED);
            assert(parser.emptyTensorFlag_);
            qShape.MutableStorageShape() = dims;
        }
    }
    for (auto dims : {gert::Shape({1, 128, 1, 576}), gert::Shape({1, 1, 128, 576}), gert::Shape({1, 1, 36, 128, 16})}) {
        kShape.MutableStorageShape() = dims;
        for (size_t axis = 0; axis < dims.GetDimNum(); ++axis) {
            kShape.MutableStorageShape().SetDim(axis, 0);
            parser.emptyTensorFlag_ = false;
            assert(parser.GetEmptyTensorFlag() == ge::GRAPH_FAILED);
            assert(parser.emptyTensorFlag_);
            kShape.MutableStorageShape() = dims;
        }
    }
    parser.emptyTensorFlag_ = false;
    assert(parser.GetEmptyTensorFlag() == ge::GRAPH_SUCCESS);

    gert::Tensor maskTensor;
    maskTensor.MutableStorageShape() = gert::Shape({2048, 2048});
    gert::CompileTimeTensorDesc maskDesc;
    maskDesc.SetDataType(ge::DT_INT8);
    maskDesc.SetOriginFormat(ge::FORMAT_ND);
    info.opParamInfo.attnMask.desc = &maskDesc;
    info.winLeft = info.winRight = -1;
    for (int mode : {0, 3, 4}) {
        info.maskMode = mode;
        for (bool present : {false, true}) {
            info.opParamInfo.attnMask.tensor = present ? &maskTensor : nullptr;
            assert((mask.CheckParaExistence(info) == ge::GRAPH_SUCCESS) == (present == (mode != 0)));
        }
    }
    assert(mask.CheckSinglePara(info) == ge::GRAPH_SUCCESS);
    maskDesc.SetDataType(ge::DT_FLOAT);
    assert(mask.CheckSinglePara(info) == ge::GRAPH_FAILED);

    gert::Tensor sinkTensor;
    gert::CompileTimeTensorDesc sinkDesc;
    sinkDesc.SetOriginFormat(ge::FORMAT_ND);
    info.opParamInfo.sinks.tensor = &sinkTensor;
    info.opParamInfo.sinks.desc = &sinkDesc;
    SinksChecker sinks;
    for (int heads = 1; heads <= 128; ++heads) {
        info.n1Size = heads;
        sinkTensor.MutableStorageShape() = gert::Shape({heads});
        for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32}) {
            sinkDesc.SetDataType(dtype);
            assert((sinks.CheckSinglePara(info) == ge::GRAPH_SUCCESS) == (dtype == ge::DT_FLOAT));
        }
        sinkDesc.SetDataType(ge::DT_FLOAT);
        sinkTensor.MutableStorageShape() = gert::Shape({heads + 1});
        assert(sinks.CheckSinglePara(info) == ge::GRAPH_FAILED);
    }
    std::cout << "Head bounds 1..128, all 64 aligned block sizes, 24 zero-axis cases, mask presence and sink "
                 "dtype/length passed.\n";

    // Optional sink placeholders still disable the feature.
    sinkTensor.MutableStorageShape() = gert::Shape({0});
    assert(sinks.CheckSinglePara(info) == ge::GRAPH_SUCCESS);
    info.opParamInfo.sinks.tensor = nullptr;
    assert(sinks.CheckSinglePara(info) == ge::GRAPH_SUCCESS);

    // Malformed API input must return an error, never throw from std::string.
    assert(FlashMlaWithKvcacheMetadataCheck::ParamsCheck(nullptr, nullptr, nullptr, -1, -1, 64, 1, 576, 512, 0, -1, -1,
                                                         nullptr, nullptr, 1, 1) == ACLNN_ERR_PARAM_NULLPTR);

    // Zero KV heads must be rejected before the divisibility check.
    info.n1Size = 64;
    info.n2Size = 0;
    assert(common.CheckNonQuantHeadNum(info) == ge::GRAPH_FAILED);
    info.n2Size = 1;

    SoftmaxLSEChecker lse;
    gert::StorageShape lseShape;
    gert::CompileTimeTensorDesc lseDesc;
    lseDesc.SetOriginFormat(ge::FORMAT_ND);
    lseDesc.SetDataType(ge::DT_FLOAT);
    info.opParamInfo.lseOut.desc = &lseDesc;
    info.opParamInfo.lseOut.shape = &lseShape;
    int64_t enabled = 1;
    info.opParamInfo.returnSoftMaxLse = &enabled;
    info.softmaxLseFlag = true;
    info.bSize = 2;
    info.n1Size = 64;
    info.s1Size = 4;
    info.qTSize = 8;
    for (auto layout : {L::BSND, L::BNSD, L::TND}) {
        info.qLayout = layout;
        const gert::Shape expected = layout == L::TND ? gert::Shape({64, 8}) : gert::Shape({2, 64, 4});
        lseShape.MutableStorageShape() = expected;
        assert(lse.CheckParaExistence(info) == ge::GRAPH_SUCCESS);
        assert(lse.CheckSinglePara(info) == ge::GRAPH_SUCCESS);
        assert(lse.CheckMultiPara(info) == ge::GRAPH_SUCCESS);
        for (size_t axis = 0; axis < expected.GetDimNum(); ++axis) {
            lseShape.MutableStorageShape().SetDim(axis, 0);
            assert(lse.CheckMultiPara(info) == ge::GRAPH_FAILED);
            lseShape.MutableStorageShape() = expected;
            lseShape.MutableStorageShape().SetDim(axis, expected.GetDim(axis) + 1);
            assert(lse.CheckMultiPara(info) == ge::GRAPH_FAILED);
            lseShape.MutableStorageShape() = expected;
        }
        lseShape.MutableStorageShape() = gert::Shape({1});
        assert(lse.CheckMultiPara(info) == ge::GRAPH_FAILED);
    }
    lseDesc.SetDataType(ge::DT_FLOAT16);
    assert(lse.CheckSinglePara(info) == ge::GRAPH_FAILED);
    lseDesc.SetDataType(ge::DT_FLOAT);
    info.opParamInfo.lseOut.desc = nullptr;
    assert(lse.CheckParaExistence(info) == ge::GRAPH_FAILED);
    assert(lse.CheckSinglePara(info) == ge::GRAPH_FAILED);
    info.opParamInfo.lseOut.shape = nullptr;
    assert(lse.CheckMultiPara(info) == ge::GRAPH_FAILED);
    info.softmaxLseFlag = false;
    enabled = 0;
    assert(lse.CheckParaExistence(info) == ge::GRAPH_SUCCESS);
    assert(lse.CheckSinglePara(info) == ge::GRAPH_SUCCESS);
    assert(lse.CheckMultiPara(info) == ge::GRAPH_SUCCESS);

    // The composite checker must establish existence before touching tensor data.
    FlashMlaWithKvcacheTilingInfo absent{};
    absent.opName = "AbsentTensorRegression";
    FlashMlaWithKvcacheChecker composite;
    composite.Init(absent);
    assert(composite.Process(absent) == ge::GRAPH_FAILED);
    absent.emptyTensorFlag = true;
    assert(composite.Process(absent) == ge::GRAPH_FAILED);
    std::cout << "Null layout, zero KV heads, LSE shapes and composite existence checks passed.\n";
    std::cout << "Layout matrix: " << accepted << " accepted, " << rejected
              << " rejected; main/metadata mask modes and metadata Q/window boundaries passed.\n";
}
