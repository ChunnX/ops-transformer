// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t DIM_THREE = 3;
constexpr int64_t DIM_FOUR = 4;
}

std::tuple<at::Tensor, at::Tensor> npu_generic_block_sparse_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &sparseBlockIdx,
    const at::Tensor &sparseBlockCount, const c10::optional<at::Tensor> &metadata,
    const c10::optional<at::Tensor> &attenMask, const c10::optional<at::Tensor> &qDequantScale,
    const c10::optional<at::Tensor> &kDequantScale, const c10::optional<at::Tensor> &vDequantScale,
    const c10::optional<at::Tensor> &pQuantScale, const c10::optional<at::Tensor> &cuSeqLengthsQ,
    const c10::optional<at::Tensor> &cuSeqLengthsKv, const c10::optional<at::Tensor> &seqUsedQ,
    const c10::optional<at::Tensor> &seqUsedKv, const c10::optional<at::Tensor> &blockTable,
    at::IntArrayRef blockShape, int64_t isPackedGQA, c10::string_view layoutQ, c10::string_view layoutKv,
    double scaleValue, int64_t maskType, int64_t quantType, double dstTypeMax, int64_t softmaxPrecision,
    int64_t winLeft, int64_t winRight, int64_t returnSoftmaxLse)
{
    std::string layoutQStr(layoutQ);
    std::string layoutKvStr(layoutKv);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    at::Tensor attentionOut = at::empty_like(query);
    at::Tensor softmaxLse;
    if (returnSoftmaxLse != 0) {
        // Align current GBSA ST / tiling LSE layout: TND -> [T, N, 1], otherwise [B, N, S, 1].
        if (query.dim() == DIM_THREE) {
            softmaxLse = at::empty({query.size(0), query.size(1), 1}, query.options().dtype(at::kFloat));
        } else if (query.dim() == DIM_FOUR) {
            if (layoutQStr == "BNSD") {
                softmaxLse = at::empty({query.size(0), query.size(1), query.size(2), 1},
                    query.options().dtype(at::kFloat));
            } else {
                // BSND
                softmaxLse = at::empty({query.size(0), query.size(2), query.size(1), 1},
                    query.options().dtype(at::kFloat));
            }
        } else {
            TORCH_CHECK(false, "Unsupported query rank for softmax_lse allocation: ", query.dim());
        }
    } else {
        softmaxLse = at::empty({0}, query.options().dtype(at::kFloat));
    }

    ACLNN_CMD(aclnnGenericBlockSparseAttention, query, key, value, sparseBlockIdx, sparseBlockCount, metadata,
        attenMask, qDequantScale, kDequantScale, vDequantScale, pQuantScale, cuSeqLengthsQ, cuSeqLengthsKv, seqUsedQ,
        seqUsedKv, blockTable, blockShape, isPackedGQA, layoutQPtr, layoutKvPtr, scaleValue, maskType, quantType,
        dstTypeMax, softmaxPrecision, winLeft, winRight, returnSoftmaxLse, attentionOut, softmaxLse);
    return std::tuple<at::Tensor, at::Tensor>(attentionOut, softmaxLse);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_generic_block_sparse_attention", &npu_generic_block_sparse_attention,
        "npu_generic_block_sparse_attention");
}
} // namespace op_api
