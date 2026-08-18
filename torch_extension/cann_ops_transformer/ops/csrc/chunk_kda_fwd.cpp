/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_kda_fwd.cpp
 * \brief Torch bridge for aclnnChunkKdaFwd.
 */

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
namespace {

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int64_t GetSeqNum(int64_t batch, at::OptionalIntArrayRef cuSeqlens)
{
    if (!cuSeqlens.has_value()) {
        return batch;
    }
    return static_cast<int64_t>(cuSeqlens->size()) - 1;
}

void CheckCuSeqlens(at::OptionalIntArrayRef cuSeqlens, int64_t totalTokens)
{
    if (!cuSeqlens.has_value()) {
        return;
    }
    const auto values = cuSeqlens.value();
    TORCH_CHECK(values.size() >= 2, "chunk_kda_fwd: cu_seqlens must contain at least [0, total_tokens].");
    TORCH_CHECK(values[0] == 0, "chunk_kda_fwd: cu_seqlens[0] must be 0.");
    TORCH_CHECK(
        values[values.size() - 1] == totalTokens,
        "chunk_kda_fwd: cu_seqlens[-1] must equal the sequence length.");
    for (size_t index = 0; index + 1 < values.size(); ++index) {
        TORCH_CHECK(
            values[index] <= values[index + 1],
            "chunk_kda_fwd: cu_seqlens must be nondecreasing.");
    }
}

void CheckChunkIndices(
    at::OptionalIntArrayRef chunkIndices,
    at::OptionalIntArrayRef cuSeqlens,
    int64_t chunkSize)
{
    if (!chunkIndices.has_value()) {
        return;
    }
    const auto indices = chunkIndices.value();
    TORCH_CHECK(
        indices.size() % 2 == 0,
        "chunk_kda_fwd: chunk_indices must contain flattened (seq_id, chunk_id) pairs.");
    TORCH_CHECK(cuSeqlens.has_value(), "chunk_kda_fwd: chunk_indices requires cu_seqlens.");

    const auto cuValues = cuSeqlens.value();
    int64_t expectedChunks = 0;
    for (size_t seq = 0; seq + 1 < cuValues.size(); ++seq) {
        expectedChunks += CeilDiv(cuValues[seq + 1] - cuValues[seq], chunkSize);
    }
    TORCH_CHECK(
        static_cast<int64_t>(indices.size() / 2) == expectedChunks,
        "chunk_kda_fwd: chunk_indices must contain exactly one pair per chunk.");

    size_t index = 0;
    for (size_t seq = 0; seq + 1 < cuValues.size(); ++seq) {
        const int64_t chunks = CeilDiv(cuValues[seq + 1] - cuValues[seq], chunkSize);
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            TORCH_CHECK(
                indices[index] == static_cast<int64_t>(seq) && indices[index + 1] == chunk,
                "chunk_kda_fwd: chunk_indices must use canonical sequence-major order.");
            index += 2;
        }
    }
}

std::vector<int64_t> BuildChunkIndices(at::IntArrayRef cuSeqlens, int64_t chunkSize)
{
    std::vector<int64_t> indices;
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens.size(); ++seq) {
        totalChunks += CeilDiv(cuSeqlens[seq + 1] - cuSeqlens[seq], chunkSize);
    }
    indices.reserve(static_cast<size_t>(totalChunks * 2));
    for (size_t seq = 0; seq + 1 < cuSeqlens.size(); ++seq) {
        const int64_t chunks = CeilDiv(cuSeqlens[seq + 1] - cuSeqlens[seq], chunkSize);
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            indices.push_back(static_cast<int64_t>(seq));
            indices.push_back(chunk);
        }
    }
    return indices;
}

int64_t GetTotalChunks(
    int64_t seqlen,
    int64_t chunkSize,
    at::OptionalIntArrayRef cuSeqlens,
    at::OptionalIntArrayRef chunkIndices)
{
    if (chunkIndices.has_value()) {
        return static_cast<int64_t>(chunkIndices->size()) / 2;
    }
    if (!cuSeqlens.has_value()) {
        return CeilDiv(seqlen, chunkSize);
    }

    int64_t totalChunks = 0;
    const auto values = cuSeqlens.value();
    for (size_t seq = 0; seq + 1 < values.size(); ++seq) {
        totalChunks += CeilDiv(values[seq + 1] - values[seq], chunkSize);
    }
    return totalChunks;
}

} // namespace

using ChunkKdaOutputs = std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>;

ChunkKdaOutputs ChunkKdaFwd(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &g,
    const at::Tensor &beta,
    double scale,
    int64_t chunkSize,
    const std::string &layout,
    const c10::optional<at::Tensor> &initialState,
    c10::optional<bool> outputFinalState,
    at::OptionalIntArrayRef cuSeqlens,
    at::OptionalIntArrayRef chunkIndices,
    c10::optional<bool> safeGate,
    c10::optional<double> lowerBound,
    c10::optional<bool> useQkL2normInKernel,
    c10::optional<bool> useGateInKernel,
    c10::optional<bool> useBetaSigmoidInKernel,
    c10::optional<bool> useExp2,
    const c10::optional<at::Tensor> &aLog,
    const c10::optional<at::Tensor> &dtBias,
    c10::optional<bool> disableRecompute,
    c10::optional<bool> returnIntermediateStates,
    c10::optional<bool> stateVFirst)
{
    TORCH_CHECK(
        layout == "BSND" || layout == "BNSD" || layout == "TND" || layout == "NTD",
        "chunk_kda_fwd: layout must be one of BSND, BNSD, TND and NTD.");
    TORCH_CHECK(chunkSize == 64 || chunkSize == 128, "chunk_kda_fwd: chunk_size must be 64 or 128.");

    const bool outputFinalStateValue = outputFinalState.value_or(false);
    const bool safeGateValue = safeGate.value_or(false);
    const double lowerBoundValue = lowerBound.value_or(-5.0);
    const bool useQkL2normValue = useQkL2normInKernel.value_or(false);
    const bool useGateValue = useGateInKernel.value_or(false);
    const bool useBetaSigmoidValue = useBetaSigmoidInKernel.value_or(false);
    const bool useExp2Value = useExp2.value_or(true);
    const bool disableRecomputeValue = disableRecompute.value_or(false);
    const bool returnIntermediateStatesValue = returnIntermediateStates.value_or(false);
    const bool stateVFirstValue = stateVFirst.value_or(false);
    TORCH_CHECK(
        !useQkL2normValue,
        "chunk_kda_fwd: use_qk_l2norm_in_kernel currently supports false only.");
    TORCH_CHECK(
        !useBetaSigmoidValue,
        "chunk_kda_fwd: use_beta_sigmoid_in_kernel currently supports false only.");
    (void)useExp2Value;

    const bool isTnd = layout == "TND";
    const bool isNtd = layout == "NTD";
    const bool isBsnd = layout == "BSND";
    const bool isBnsd = layout == "BNSD";
    const bool isRank3 = isTnd || isNtd;
    TORCH_CHECK(
        (isRank3 && q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && g.dim() == 3 && beta.dim() == 2) ||
            (!isRank3 && q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && g.dim() == 4 && beta.dim() == 3),
        "chunk_kda_fwd: input ranks do not match layout.");
    TORCH_CHECK(q.sizes() == k.sizes(), "chunk_kda_fwd: q and k must have identical shape.");

    const auto qSizes = q.sizes();
    const auto vSizes = v.sizes();
    const int64_t batch = isRank3 ? 1 : qSizes[0];
    const int64_t seqlen = isTnd ? qSizes[0] : (isNtd ? qSizes[1] : (isBnsd ? qSizes[2] : qSizes[1]));
    const int64_t h = isTnd ? qSizes[1] : (isNtd ? qSizes[0] : (isBnsd ? qSizes[1] : qSizes[2]));
    const int64_t kDim = isRank3 ? qSizes[2] : qSizes[3];
    const int64_t hv = isTnd ? vSizes[1] : (isNtd ? vSizes[0] : (isBnsd ? vSizes[1] : vSizes[2]));
    const int64_t vDim = isRank3 ? vSizes[2] : vSizes[3];
    TORCH_CHECK(
        h > 0 && hv >= h && hv % h == 0 && h <= 128 && hv <= 128,
        "chunk_kda_fwd: H/HV must satisfy 0 < H <= HV <= 128 and HV % H == 0.");
    TORCH_CHECK(
        kDim >= 16 && kDim <= 256 && kDim % 16 == 0 && vDim >= 16 && vDim <= 256 && vDim % 16 == 0,
        "chunk_kda_fwd: K/V must be multiples of 16 and no greater than 256.");
    TORCH_CHECK(
        q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
        "chunk_kda_fwd: q/k/v must use float16 or bfloat16.");
    TORCH_CHECK(
        k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(),
        "chunk_kda_fwd: q/k/v dtype must match.");
    TORCH_CHECK(
        (g.scalar_type() == at::kFloat || g.scalar_type() == at::kBFloat16) &&
            (beta.scalar_type() == at::kFloat || beta.scalar_type() == at::kBFloat16),
        "chunk_kda_fwd: g and beta must be float32 or bfloat16.");

    CheckCuSeqlens(cuSeqlens, seqlen);
    CheckChunkIndices(chunkIndices, cuSeqlens, chunkSize);
    TORCH_CHECK(
        !cuSeqlens.has_value() || isRank3 || batch == 1,
        "chunk_kda_fwd: rank4 varlen input requires B=1.");

    const auto gSizes = g.sizes();
    TORCH_CHECK(
        (isTnd && beta.sizes() == at::IntArrayRef({seqlen, hv})) ||
            (isNtd && beta.sizes() == at::IntArrayRef({hv, seqlen})) ||
            (isBsnd && beta.sizes() == at::IntArrayRef({batch, seqlen, hv})) ||
            (isBnsd && beta.sizes() == at::IntArrayRef({batch, hv, seqlen})),
        "chunk_kda_fwd: beta shape mismatch.");
    TORCH_CHECK(
        (isTnd && vSizes == at::IntArrayRef({seqlen, hv, vDim}) &&
         gSizes == at::IntArrayRef({seqlen, hv, kDim})) ||
            (isNtd && vSizes == at::IntArrayRef({hv, seqlen, vDim}) &&
             gSizes == at::IntArrayRef({hv, seqlen, kDim})) ||
            (isBsnd && vSizes == at::IntArrayRef({batch, seqlen, hv, vDim}) &&
             gSizes == at::IntArrayRef({batch, seqlen, hv, kDim})) ||
            (isBnsd && vSizes == at::IntArrayRef({batch, hv, seqlen, vDim}) &&
             gSizes == at::IntArrayRef({batch, hv, seqlen, kDim})),
        "chunk_kda_fwd: v/g shapes do not match layout.");

    const int64_t seqNum = GetSeqNum(batch, cuSeqlens);
    TORCH_CHECK(seqNum <= 1024, "chunk_kda_fwd: at most 1024 sequences are supported.");
    const std::vector<int64_t> stateShape =
        stateVFirstValue ? std::vector<int64_t>{seqNum, hv, vDim, kDim}
                         : std::vector<int64_t>{seqNum, hv, kDim, vDim};
    if (initialState.has_value() && initialState->defined()) {
        TORCH_CHECK(
            initialState->scalar_type() == at::kFloat,
            "chunk_kda_fwd: initial_state must be float32 when provided.");
        TORCH_CHECK(
            initialState->sizes() == at::IntArrayRef(stateShape),
            "chunk_kda_fwd: initial_state shape does not match state_v_first.");
    }

    if (useGateValue) {
        TORCH_CHECK(
            aLog.has_value() && aLog->defined() && aLog->scalar_type() == at::kFloat &&
                aLog->sizes() == at::IntArrayRef({hv}),
            "chunk_kda_fwd: A_log must be float32 [HV] when use_gate_in_kernel=True.");
        if (dtBias.has_value() && dtBias->defined()) {
            TORCH_CHECK(
                dtBias->scalar_type() == at::kFloat && dtBias->sizes() == at::IntArrayRef({hv * kDim}),
                "chunk_kda_fwd: dt_bias must be float32 [HV*K].");
        }
        if (safeGateValue) {
            TORCH_CHECK(
                lowerBoundValue >= -5.0 && lowerBoundValue < 0.0,
                "chunk_kda_fwd: lower_bound must be in [-5, 0).");
        }
    }

    std::vector<int64_t> generatedChunkIndices;
    c10::optional<at::IntArrayRef> chunkIndicesForCall;
    if (chunkIndices.has_value()) {
        chunkIndicesForCall = chunkIndices.value();
    } else if (cuSeqlens.has_value()) {
        generatedChunkIndices = BuildChunkIndices(cuSeqlens.value(), chunkSize);
        chunkIndicesForCall = at::IntArrayRef(generatedChunkIndices);
    } else {
        chunkIndicesForCall = c10::nullopt;
    }

    const int64_t totalChunks = GetTotalChunks(seqlen, chunkSize, cuSeqlens, chunkIndicesForCall);
    const std::vector<int64_t> attnShape =
        isRank3 ? std::vector<int64_t>{seqlen, hv, vDim}
                : std::vector<int64_t>{batch, seqlen, hv, vDim};
    const std::vector<int64_t> matrixShape =
        isRank3 ? std::vector<int64_t>{hv, seqlen, chunkSize}
                : std::vector<int64_t>{batch, hv, seqlen, chunkSize};
    const std::vector<int64_t> kShape =
        isRank3 ? std::vector<int64_t>{hv, seqlen, kDim}
                : std::vector<int64_t>{batch, hv, seqlen, kDim};
    const std::vector<int64_t> vShape =
        isRank3 ? std::vector<int64_t>{hv, seqlen, vDim}
                : std::vector<int64_t>{batch, hv, seqlen, vDim};
    const std::vector<int64_t> hShape =
        isRank3
            ? (stateVFirstValue ? std::vector<int64_t>{totalChunks, hv, vDim, kDim}
                                : std::vector<int64_t>{totalChunks, hv, kDim, vDim})
            : (stateVFirstValue ? std::vector<int64_t>{batch, totalChunks, hv, vDim, kDim}
                                : std::vector<int64_t>{batch, totalChunks, hv, kDim, vDim});

    at::Tensor attnOut;
    at::Tensor finalState;
    at::Tensor gkOut;
    at::Tensor aqk;
    at::Tensor akk;
    at::Tensor w;
    at::Tensor u;
    at::Tensor qg;
    at::Tensor kg;
    at::Tensor vNew;
    at::Tensor hOut;
    {
        const auto localDevice = c10::Device(q.device());
        const c10::OptionalDeviceGuard deviceGuard(localDevice);
        attnOut = at::empty(attnShape, v.options());
        finalState =
            outputFinalStateValue ? at::empty(stateShape, q.options().dtype(at::kFloat)) : at::Tensor();
        gkOut = !useGateValue ? at::empty(kShape, q.options().dtype(at::kFloat)) : at::Tensor();
        aqk = at::empty(matrixShape, q.options());
        akk = at::empty(matrixShape, q.options());
        w = disableRecomputeValue ? at::empty(kShape, q.options()) : at::Tensor();
        u = disableRecomputeValue ? at::empty(vShape, q.options()) : at::Tensor();
        qg = disableRecomputeValue ? at::empty(kShape, q.options()) : at::Tensor();
        kg = disableRecomputeValue ? at::empty(kShape, q.options()) : at::Tensor();
        vNew = disableRecomputeValue ? at::empty(vShape, q.options()) : at::Tensor();
        hOut = returnIntermediateStatesValue ? at::empty(hShape, q.options()) : at::Tensor();
    }

    at::Tensor initialStateValue = c10::value_or_else(initialState, [] { return at::Tensor(); });
    at::Tensor aLogValue = c10::value_or_else(aLog, [] { return at::Tensor(); });
    at::Tensor dtBiasValue = c10::value_or_else(dtBias, [] { return at::Tensor(); });
    const char *layoutValue = layout.c_str();

    ACLNN_CMD(
        aclnnChunkKdaFwd,
        q,
        k,
        v,
        g,
        beta,
        aLogValue,
        dtBiasValue,
        initialStateValue,
        cuSeqlens,
        chunkIndicesForCall,
        layoutValue,
        scale,
        chunkSize,
        safeGateValue,
        lowerBoundValue,
        useGateValue,
        stateVFirstValue,
        attnOut,
        finalState,
        gkOut,
        aqk,
        akk,
        w,
        u,
        qg,
        kg,
        vNew,
        hOut);

    return std::make_tuple(attnOut, finalState, gkOut, aqk, akk, w, u, qg, kg, vNew, hOut, initialStateValue);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("chunk_kda_fwd", &ChunkKdaFwd, "chunk KDA forward");
}

} // namespace op_api
