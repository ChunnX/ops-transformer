/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_select_attention_infer_case.h
 * \brief NsaSelectAttentionInfer 测试用例.
 */

#pragma once
#include <vector>
#include <cstdint>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tests/utils/case.h"
#include "tests/utils/op_info.h"
#include "tests/utils/context.h"
#include "tests/utils/tensor.h"
#include "tests/utils/tensor_list.h"

namespace ops::adv::tests::NsaSelectAttentionInfer {
    class NsaSelectAttentionInferCase : public ops::adv::tests::utils::Case {
        using OpInfo = ops::adv::tests::utils::OpInfo;
        using Context = ops::adv::tests::utils::Context;
        using Tensor = ops::adv::tests::utils::Tensor;
        using TensorList = ops::adv::tests::utils::TensorList;
    public:
        class Param {
        public:
            int64_t batchSize = 1;
            int64_t qSeqSize = 1;
            int64_t headSize = 2;
            int64_t headDim = 192;
            int64_t maxBlockNumPerBatch = 8;
            int64_t blockSize = 128;
            int64_t seqSize = 1024;
            int64_t headDimV = 128;
            int64_t headSizeV = 1;
            int64_t selectedBlockSize = 272;
            int64_t selectedBlockCount = 4;
            int64_t sparseMode = 1;
            float scaleValue = 1.0f;
            std::string inputLayout = "BSND";
            ge::DataType optionalDataType = ge::DT_FLOAT16;
            std::vector<int64_t> actualSeqQLenList = {};
            std::vector<int64_t> actualSeqKVLenList = {};
            int64_t numtokens = 1;
            Param();
            Param(int64_t pBatchSize, int64_t pQSeqSize, int64_t pHeadSize, int64_t pHeadDim, int64_t pMaxBlockNumPerBatch,
                  int64_t pBlockSize, int64_t pSeqSize, int64_t pHeadDimV, int64_t pHeadSizeV, int64_t pSelectedBlockSize, int64_t pSelectedBlockCount, int64_t pSparseMode,
                  float pScaleValue, std::string pInputLayout, ge::DataType pOptionalDataType, std::vector<int64_t> pActualSeqQLenList, std::vector<int64_t> pActualSeqKVLenList, int64_t numtokens);
        };
        class DoTilingParam {
        public:
            gert::TilingContext *ctx = nullptr;
            ge::graphStatus ret = ge::GRAPH_SUCCESS;
            gert::Tensor *actSeqSelQLenTensor = nullptr;
            gert::Tensor *actSeqSelKVLenTensor = nullptr;
            gert::Tensor *blocktableTensor = nullptr;
            gert::Tensor *topkIndicesTensor = nullptr;
        };
        OpInfo mOpInfo;
        Context mCtx;
        Param mParam;
        Tensor query;
        Tensor key;
        Tensor value;
        Tensor topkIndices;
        Tensor attenMask;
        Tensor blocktable;
        Tensor actualQSeqLengths;
        Tensor actualKvSeqLengths;
        Tensor attentionOut;
        std::vector<int32_t> topkData;
        std::vector<int32_t> blocktableData;
        gert::OpImplRegisterV2::TilingKernelFunc nsaSelectAttentionInferTilingFunc = nullptr;
        NsaSelectAttentionInferCase();
        NsaSelectAttentionInferCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param);
        void TNDInitParam();
        void BSNDInitParam();
        void BSHInitParam();
        void TopkInitParam();
        void InitTopkDataTND();
        void InitTopkDataForHead();
        void InitTopkDataForSelectedBlockSize();
        void InitTopkDataOtherLayout();
        void InitTopkDataForS1();
        bool Run() override;
        bool InitParam() override;
        bool InitOpInfo() override;
        bool InitCurrentCasePtr() override;
        bool DoOpTiling(DoTilingParam &tilingParam);
        template <class T> static bool InitTensor(Tensor &tensor, std::vector<T> &hostData)
        {
            if (hostData.empty()) {
                return true;
            }
            int64_t expMinSize = hostData.size() * sizeof(T);
            if (tensor.AllocDevData(0, expMinSize) == nullptr) {
                printf("Tensor(%s, %ld) AllocDevData Failed.", tensor.Name().c_str(), expMinSize);
                return false;
            }
            return tensor.CopyHostToDevData(hostData);
        }
    };
} // namespace ops::adv::tests::NsaSelectAttentionInfer