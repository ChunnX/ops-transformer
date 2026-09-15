/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>
#include "aclnn_common.h"

namespace op_api {

// npu tensor max size
const int SIZE = 8;
const int64_t DIM_THREE = 3;
const int64_t DIM_FOUR = 4;
const int64_t DIM_FIVE = 5;
const int64_t FP32_BYTES = 4;
const int64_t QBSA_FP8_QUANT_MODE = 1;
const int64_t QBSA_MXFP8_FULL_QUANT_MODE = 2;
const int64_t QBSA_MXFP8_SCALE_GROUP_SIZE = 64;
const int64_t QBSA_MXFP8_SCALE_LAST_DIM = 2;

// 工具函数，推导输出 attention_out / softmax_lse 的 shape 与 dtype
//   - attention_out: BF16，TND 保持原布局，NTD 转为 TND
//   - softmax_lse:   FLOAT32，FP8 为 (N1,T)，MXFP8 为 (T,N1)
std::tuple<at::Tensor, at::Tensor> construct_bsa_output_tensors(const at::Tensor &query, const at::Tensor &value,
                                                                std::string layout_q, int64_t quant_mode,
                                                                bool return_softmax_lse)
{
    const std::string layout_q_str(layout_q);
    TORCH_CHECK(query.dim() == DIM_THREE, "query should be 3D for layout_q TND/NTD, but got ", query.dim(), "D.");
    TORCH_CHECK(layout_q_str == "TND" || layout_q_str == "NTD", "layout_q should be TND or NTD, but got ",
                layout_q_str);
    TORCH_CHECK(quant_mode != QBSA_MXFP8_FULL_QUANT_MODE || layout_q_str == "TND",
                "quant_mode=2 MXFP8 full-quant only supports layout_q TND, but got ", layout_q_str, ".");
    for (auto i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", query.size(i));
    }

    const bool is_ntd = layout_q_str == "NTD";
    int64_t t_size = is_ntd ? query.size(1) : query.size(0);
    int64_t n1_size = is_ntd ? query.size(0) : query.size(1);
    int64_t d_size = query.size(2);
    at::SmallVector<int64_t, SIZE> atten_out_size = {t_size, n1_size, d_size};
    at::SmallVector<int64_t, SIZE> softmax_lse_size;
    if (quant_mode == QBSA_MXFP8_FULL_QUANT_MODE) {
        softmax_lse_size = {t_size, n1_size};
    } else {
        softmax_lse_size = {n1_size, t_size};
    }
    if (!return_softmax_lse) {
        softmax_lse_size = {};
    }

    at::Tensor attention_out = at::empty(atten_out_size, query.options().dtype(at::kBFloat16));
    at::Tensor softmax_lse = at::empty(softmax_lse_size, query.options().dtype(at::kFloat));
    return std::tuple<at::Tensor, at::Tensor>(attention_out, softmax_lse);
}

// step2, 为NPU设备实现前向接口（函数形参顺序 = schema 顺序）
std::tuple<at::Tensor, at::Tensor> npu_quant_block_sparse_attn(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &q_descale,
    const at::Tensor &k_descale, const at::Tensor &v_descale, const at::Tensor &sparse_indices,
    const at::Tensor &sparse_seq_len, const c10::optional<at::Tensor> &p_scale,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &block_table, const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &metadata, double softmax_scale, int64_t sparse_block_size_q,
    int64_t sparse_block_size_kv, std::string layout_q, std::string layout_kv, std::string layout_out,
    std::string layout_sparse_indices, int64_t quant_mode, int64_t mask_mode, bool return_softmax_lse)
{
    const c10::OptionalDeviceGuard device_guard(query.device());
    TORCH_CHECK(layout_kv == "PA_BNBD" && layout_out == "TND", "QBSA supports layout_kv=PA_BNBD and layout_out=TND.");
    TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
    TORCH_CHECK(key.dim() == DIM_FOUR && value.dim() == DIM_FOUR,
                "key/value should be 4D PA_BNBD [blockNum, Nkv, blockSize, headDim], but got dims ", key.dim(), "/",
                value.dim());
    int64_t paBlockStride = key.stride(0);
    TORCH_CHECK(paBlockStride > 0, "key.stride(0) (paBlockStride) should be greater than 0, but got ", paBlockStride);
    TORCH_CHECK(value.stride(0) == paBlockStride,
                "value.stride(0) should equal key.stride(0), but got value.stride(0)=", value.stride(0),
                " and key.stride(0)=", paBlockStride);
    TORCH_CHECK(key.stride(1) == key.size(2) * key.size(3) && key.stride(2) == key.size(3) && key.stride(3) == 1,
                "key should use segmented PA_BNBD stride [paBlockStride, blockSize * headDim, headDim, 1], but got ",
                key.strides());
    TORCH_CHECK(
        value.stride(1) == value.size(2) * value.size(3) && value.stride(2) == value.size(3) && value.stride(3) == 1,
        "value should use segmented PA_BNBD stride [paBlockStride, blockSize * headDim, headDim, 1], but got ",
        value.strides());
    if (quant_mode == QBSA_MXFP8_FULL_QUANT_MODE) {
        const int64_t keyScaleDSize = (key.size(3) + QBSA_MXFP8_SCALE_GROUP_SIZE - 1) / QBSA_MXFP8_SCALE_GROUP_SIZE;
        const int64_t valueScaleBlockSize =
            (value.size(2) + QBSA_MXFP8_SCALE_GROUP_SIZE - 1) / QBSA_MXFP8_SCALE_GROUP_SIZE;
        TORCH_CHECK(k_descale.dim() == DIM_FIVE,
                    "k_descale should be 5D [blockNum, Nkv, blockSize, D / 64, 2] in quant_mode=2 MXFP8 "
                    "full-quant scenario, but got dim=",
                    k_descale.dim());
        TORCH_CHECK(k_descale.size(0) == key.size(0) && k_descale.size(1) == key.size(1) &&
                        k_descale.size(2) == key.size(2) && k_descale.size(3) == keyScaleDSize &&
                        k_descale.size(4) == QBSA_MXFP8_SCALE_LAST_DIM,
                    "k_descale should be [blockNum, Nkv, blockSize, D / 64, 2] and match key PA_BNBD "
                    "shape, but got k_descale sizes ",
                    k_descale.sizes(), " and key sizes ", key.sizes());
        TORCH_CHECK(k_descale.stride(4) == 1 && k_descale.stride(3) == k_descale.size(4) &&
                        k_descale.stride(2) == k_descale.size(3) * k_descale.size(4) &&
                        k_descale.stride(1) == k_descale.size(2) * k_descale.size(3) * k_descale.size(4) &&
                        k_descale.stride(0) == k_descale.size(1) * k_descale.stride(1),
                    "k_descale should use contiguous [blockNum, Nkv, blockSize, D / 64, 2] stride in "
                    "quant_mode=2 MXFP8 full-quant scenario, but got ",
                    k_descale.strides());
        TORCH_CHECK(v_descale.dim() == DIM_FIVE,
                    "v_descale should be 5D [blockNum, Nkv, blockSize / 64, DV, 2] in quant_mode=2 MXFP8 "
                    "scenario, but got dim=",
                    v_descale.dim());
        TORCH_CHECK(v_descale.size(0) == value.size(0) && v_descale.size(1) == value.size(1) &&
                        v_descale.size(2) == valueScaleBlockSize && v_descale.size(3) == value.size(3) &&
                        v_descale.size(4) == QBSA_MXFP8_SCALE_LAST_DIM,
                    "v_descale should be [blockNum, Nkv, blockSize / 64, DV, 2] and match value PA_BNBD "
                    "shape, but got v_descale sizes ",
                    v_descale.sizes(), " and value sizes ", value.sizes());
        TORCH_CHECK(v_descale.stride(4) == 1 && v_descale.stride(3) == v_descale.size(4) &&
                        v_descale.stride(2) == v_descale.size(3) * v_descale.size(4) &&
                        v_descale.stride(1) == v_descale.size(2) * v_descale.size(3) * v_descale.size(4) &&
                        v_descale.stride(0) == v_descale.size(1) * v_descale.stride(1),
                    "v_descale should use contiguous [blockNum, Nkv, blockSize / 64, DV, 2] stride in "
                    "quant_mode=2 MXFP8 full-quant scenario, but got ",
                    v_descale.strides());
    } else if (quant_mode == QBSA_FP8_QUANT_MODE) {
        TORCH_CHECK(k_descale.dim() == DIM_FOUR,
                    "k_descale should be 4D [blockNum, Nkv, blockSize, 1] in quant_mode=1 FP8 "
                    "full-quant scenario, but got dim=",
                    k_descale.dim());
        TORCH_CHECK(k_descale.size(0) == key.size(0) && k_descale.size(1) == key.size(1) &&
                        k_descale.size(2) == key.size(2) && k_descale.size(3) == 1,
                    "k_descale should be [blockNum, Nkv, blockSize, 1] and match key PA_BNBD shape, "
                    "but got k_descale sizes ",
                    k_descale.sizes(), " and key sizes ", key.sizes());
        const int64_t expectedPaBlockStride = key.size(1) * key.size(2) * key.size(3) +
                                              value.size(1) * value.size(2) * value.size(3) +
                                              k_descale.size(1) * k_descale.size(2) * k_descale.size(3) * FP32_BYTES;
        TORCH_CHECK(paBlockStride == expectedPaBlockStride,
                    "key.stride(0) should equal K/V/k_descale concatenated physical block size, but got ",
                    paBlockStride, " and expected ", expectedPaBlockStride);
        TORCH_CHECK(k_descale.stride(0) * FP32_BYTES == paBlockStride, "k_descale.stride(0) * ", FP32_BYTES,
                    " should equal key.stride(0), but got k_descale.stride(0)=", k_descale.stride(0),
                    " and key.stride(0)=", paBlockStride);
        TORCH_CHECK(k_descale.stride(1) == k_descale.size(2) && k_descale.stride(2) == 1 && k_descale.stride(3) == 1,
                    "k_descale should use segmented PA_BNBD stride [paBlockStride / 4, blockSize, 1, 1], "
                    "but got ",
                    k_descale.strides());
    } else {
        TORCH_CHECK(false, "quant_mode must be 1 (FP8) or 2 (MXFP8 full-quant), but got ", quant_mode);
    }

    // construct the output tensors
    std::tuple<at::Tensor, at::Tensor> outputs =
        construct_bsa_output_tensors(query, value, layout_q, quant_mode, return_softmax_lse);
    at::Tensor attention_out = std::get<0>(outputs);
    at::Tensor softmax_lse = std::get<1>(outputs);

    // convert str
    std::string layout_kv_str = std::string(layout_kv);
    std::string layout_q_str = std::string(layout_q);
    std::string layout_sparse_indices_str = std::string(layout_sparse_indices);
    std::string layout_out_str = std::string(layout_out);
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_sparse_indices_ptr = const_cast<char *>(layout_sparse_indices_str.c_str());
    char *layout_out_ptr = const_cast<char *>(layout_out_str.c_str());

    // aclnn 框架不接受 nullptr aclTensor*：将未提供的 optional p_scale 归一化为 numel=0 空 tensor。
    // dtype 须匹配 CheckScaleDtype：FP8->FLOAT, MXFP8->FLOAT8_E8M0
    at::Tensor p_scale_placeholder;
    const at::Tensor *p_scale_arg;
    if (p_scale.has_value() && p_scale.value().defined()) {
        p_scale_arg = &p_scale.value();
    } else {
        const at::ScalarType empty_dtype =
            (quant_mode == QBSA_MXFP8_FULL_QUANT_MODE) ? at::kFloat8_e8m0fnu : at::kFloat;
        p_scale_placeholder = at::empty({0}, query.options().dtype(empty_dtype));
        p_scale_arg = &p_scale_placeholder;
    }

    // ACLNN_CMD 输入和属性顺序与 Torch schema / CANN IR 一致，末尾追加输出。
    ACLNN_CMD(aclnnQuantBlockSparseAttn, query, key, value, q_descale, k_descale, v_descale, sparse_indices,
              sparse_seq_len, *p_scale_arg, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, block_table, atten_mask,
              metadata, softmax_scale, sparse_block_size_q, sparse_block_size_kv, layout_q_ptr, layout_kv_ptr,
              layout_out_ptr, layout_sparse_indices_ptr, quant_mode, mask_mode, return_softmax_lse, attention_out,
              softmax_lse);

    return std::tuple<at::Tensor, at::Tensor>(attention_out, softmax_lse);
}

constexpr uint64_t QBSA_HEAD_METADATA_SIZE = 8;
constexpr uint64_t QBSA_AIC_CORE_NUM = 36;
constexpr uint64_t QBSA_AIV_CORE_NUM = 72;
constexpr uint64_t QBSA_METADATA_SIZE = 8;
constexpr uint64_t QBSA_FD_METADATA_SIZE = 8;

uint64_t GetQbsaMetadataOutputSize(int64_t batchSize, int64_t numHeadsQ)
{
    return QBSA_HEAD_METADATA_SIZE +
           static_cast<uint64_t>(batchSize) * static_cast<uint64_t>(numHeadsQ) * QBSA_AIC_CORE_NUM *
               QBSA_METADATA_SIZE +
           QBSA_AIV_CORE_NUM * QBSA_FD_METADATA_SIZE;
}

c10::optional<at::Tensor> get_qbsa_valid_tensor(const c10::optional<at::Tensor> &tensorOpt, const at::Device &device)
{
    if (tensorOpt.has_value() && tensorOpt.value().defined()) {
        return tensorOpt;
    }
    return c10::optional<at::Tensor>(torch::empty({0}, torch::dtype(torch::kInt32).device(device)));
}

at::Tensor npu_quant_block_sparse_attn_metadata(const at::Tensor &sparseSeqLen, int64_t numHeadsQ, int64_t numHeadsKv,
                                                int64_t headDim, const c10::optional<at::Tensor> &cuSeqlensQ,
                                                const c10::optional<at::Tensor> &cuSeqlensKv,
                                                const c10::optional<at::Tensor> &sequsedQ,
                                                const c10::optional<at::Tensor> &sequsedKv, int64_t batchSize,
                                                int64_t sparseBlockSizeQ, int64_t sparseBlockSizeK, int64_t quantMode,
                                                int64_t maskMode, std::string layoutQ, std::string layoutKv,
                                                std::string layoutSparseIndices)
{
    TORCH_CHECK(batchSize > 0 && numHeadsQ > 0 && numHeadsKv > 0 && headDim > 0,
                "batch_size, num_heads_q, num_heads_kv and head_dim must be positive.");
    const c10::OptionalDeviceGuard device_guard(sparseSeqLen.device());
    at::Device outputDevice = sparseSeqLen.device();
    uint64_t metadata_size = GetQbsaMetadataOutputSize(batchSize, numHeadsQ);
    at::Tensor output =
        torch::empty({static_cast<int64_t>(metadata_size)}, torch::dtype(torch::kInt32).device(outputDevice));

    auto cuSeqlensQValue = get_qbsa_valid_tensor(cuSeqlensQ, outputDevice);
    auto cuSeqlensKvValue = get_qbsa_valid_tensor(cuSeqlensKv, outputDevice);
    auto sequsedQValue = get_qbsa_valid_tensor(sequsedQ, outputDevice);
    auto sequsedKvValue = get_qbsa_valid_tensor(sequsedKv, outputDevice);

    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    std::string layoutSparseIndicesStr = std::string(layoutSparseIndices);
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());
    char *layoutSparseIndicesPtr = const_cast<char *>(layoutSparseIndicesStr.c_str());

    ACLNN_CMD(aclnnQuantBlockSparseAttnMetadata, sparseSeqLen, cuSeqlensQValue, cuSeqlensKvValue, sequsedQValue,
              sequsedKvValue, batchSize, numHeadsQ, numHeadsKv, headDim, sparseBlockSizeQ, sparseBlockSizeK, quantMode,
              maskMode, layoutQPtr, layoutKvPtr, layoutSparseIndicesPtr, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_quant_block_sparse_attn", &npu_quant_block_sparse_attn);
    m.def("npu_quant_block_sparse_attn_metadata", &npu_quant_block_sparse_attn_metadata);
}
} // namespace op_api
