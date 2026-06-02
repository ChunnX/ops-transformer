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
 * \file moe_quant.cpp
 * \brief Per-channel activation quantization for MoE W8A8 inference.
 */

#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <cstdint>
#include <limits>
#include "acl/acl.h"
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace npu_ops_transformer_ext {
namespace MoeQuant {

#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BLOCK_DIM = 40;
constexpr int32_t BF16_32B_ALIGN = 16;

class MoeQuantKernel {
public:
    __aicore__ inline MoeQuantKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR localRows, GM_ADDR quantScale, GM_ADDR out, int32_t tokenRows,
                                int32_t expertNum, int32_t hidden, int32_t localRowsCount, int32_t blockCount)
    {
        coreId_ = GetBlockIdx();
        tokenRows_ = tokenRows;
        expertNum_ = expertNum;
        hidden_ = hidden;
        localRowsCount_ = localRowsCount;
        blockCount_ = blockCount;

        const int32_t baseSize = 32 / sizeof(bfloat16_t);
        const int32_t baseSizeTotal = 32 / sizeof(int64_t);
        alignHidden_ = DivCeil(hidden_, baseSize) * baseSize;
        alignLocalRowsCount_ = DivCeil(localRowsCount_, baseSizeTotal) * baseSizeTotal;

        int32_t loopCount = expertNum_ / blockCount_;
        const int32_t tailCount = expertNum_ - loopCount * blockCount_;
        if (coreId_ < tailCount) {
            loopCount += 1;
        }
        loopCount_ = loopCount;

        xGm_.SetGlobalBuffer((__gm__ bfloat16_t *)x, tokenRows_ * hidden_);
        outGm_.SetGlobalBuffer((__gm__ int8_t *)out, tokenRows_ * hidden_);
        quantScaleGm_.SetGlobalBuffer((__gm__ float *)quantScale, expertNum_ * hidden_);
        localRowsGm_.SetGlobalBuffer((__gm__ int64_t *)localRows, alignLocalRowsCount_);

        pipe_.InitBuffer(localRowsQueue_, 1, alignLocalRowsCount_ * sizeof(int64_t));
        pipe_.InitBuffer(xQueue_, BUFFER_NUM, alignHidden_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(xFloatQueue_, BUFFER_NUM, alignHidden_ * sizeof(float));
        pipe_.InitBuffer(quantScaleQueue_, 1, alignHidden_ * sizeof(float));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, alignHidden_ * sizeof(int8_t));
        pipe_.InitBuffer(workBuf_, alignHidden_ * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int64_t> localRowsLocal = localRowsQueue_.AllocTensor<int64_t>();
        DataCopy(localRowsLocal, localRowsGm_[0], alignLocalRowsCount_);
        localRowsQueue_.EnQue(localRowsLocal);
        localRowsLocal_ = localRowsQueue_.DeQue<int64_t>();

        for (int32_t i = 0; i < loopCount_; ++i) {
            ProcessExpert(i);
        }

        localRowsQueue_.FreeTensor(localRowsLocal_);
    }

private:
    __aicore__ inline void ProcessExpert(int32_t loopIndex)
    {
        const int32_t expertIndex = loopIndex * blockCount_ + coreId_;
        if (expertIndex >= expertNum_) {
            return;
        }

        int32_t startRow = 0;
        if (expertIndex > 0) {
            startRow = static_cast<int32_t>(localRowsLocal_.GetValue(expertIndex - 1));
        }
        const int32_t endRow = static_cast<int32_t>(localRowsLocal_.GetValue(expertIndex));
        const int32_t countRows = endRow - startRow;

        LocalTensor<bfloat16_t> xLocal = xQueue_.AllocTensor<bfloat16_t>();
        LocalTensor<float> xFloatLocal = xFloatQueue_.AllocTensor<float>();
        LocalTensor<int8_t> outLocal = outQueue_.AllocTensor<int8_t>();
        LocalTensor<half> tempLocal = workBuf_.Get<half>();
        LocalTensor<float> quantScaleLocal = quantScaleQueue_.AllocTensor<float>();

        if (countRows > 0) {
            DataCopy(quantScaleLocal, quantScaleGm_[expertIndex * hidden_], alignHidden_);
        }

        for (int32_t rowIndex = 0; rowIndex < countRows; ++rowIndex) {
            const int32_t inputOffset = startRow + rowIndex;

            int32_t eventId1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventId1);
            WaitFlag<HardEvent::MTE3_MTE2>(eventId1);

            DataCopy(xLocal, xGm_[inputOffset * hidden_], alignHidden_);

            int32_t eventId2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId2);
            WaitFlag<HardEvent::MTE2_V>(eventId2);

            Cast(xFloatLocal, xLocal, RoundMode::CAST_NONE, alignHidden_);
            Mul(xFloatLocal, xFloatLocal, quantScaleLocal, alignHidden_);
            Mins(xFloatLocal, xFloatLocal, static_cast<float>(127.0), alignHidden_);
            Maxs(xFloatLocal, xFloatLocal, static_cast<float>(-128.0), alignHidden_);
            Cast(tempLocal, xFloatLocal, RoundMode::CAST_NONE, alignHidden_);
            Cast(outLocal, tempLocal, RoundMode::CAST_RINT, alignHidden_);

            int32_t eventId3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventId3);
            WaitFlag<HardEvent::V_MTE3>(eventId3);

            DataCopy(outGm_[inputOffset * hidden_], outLocal, alignHidden_);
        }

        xQueue_.FreeTensor(xLocal);
        xFloatQueue_.FreeTensor(xFloatLocal);
        quantScaleQueue_.FreeTensor(quantScaleLocal);
        outQueue_.FreeTensor(outLocal);
        workBuf_.FreeTensor(tempLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xFloatQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> localRowsQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> quantScaleQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<QuePosition::VECCALC> workBuf_;

    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<int8_t> outGm_;
    GlobalTensor<float> quantScaleGm_;
    GlobalTensor<int64_t> localRowsGm_;
    LocalTensor<int64_t> localRowsLocal_;

    int32_t coreId_;
    int32_t tokenRows_;
    int32_t expertNum_;
    int32_t hidden_;
    int32_t localRowsCount_;
    int32_t blockCount_;
    int32_t alignHidden_;
    int32_t alignLocalRowsCount_;
    int32_t loopCount_;
};

extern "C" __global__ __aicore__ void moe_quant_kernel(GM_ADDR x, GM_ADDR localRows, GM_ADDR quantScale, GM_ADDR out,
                                                       int32_t tokenRows, int32_t expertNum, int32_t hidden,
                                                       int32_t localRowsCount, int32_t blockCount)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    MoeQuantKernel op;
    op.Init(x, localRows, quantScale, out, tokenRows, expertNum, hidden, localRowsCount, blockCount);
    op.Process();
}

void moe_quant_launch(uint8_t *x, uint8_t *localRows, uint8_t *quantScale, uint8_t *out, int32_t tokenRows,
                      int32_t expertNum, int32_t hidden, int32_t localRowsCount, int32_t blockDim,
                      aclrtStream stream)
{
    if (expertNum <= 0) {
        return;
    }
    const int32_t realBlockDim = expertNum < blockDim ? expertNum : blockDim;
    moe_quant_kernel<<<realBlockDim, nullptr, stream>>>(x, localRows, quantScale, out, tokenRows, expertNum, hidden,
                                                        localRowsCount, realBlockDim);
}

void check_moe_quant_common(const at::Tensor &x, const at::Tensor &localRows, const at::Tensor &quantScale)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2-D tensor, got dim=", x.dim());
    TORCH_CHECK(localRows.dim() == 1, "local_rows must be a 1-D tensor, got dim=", localRows.dim());
    TORCH_CHECK(quantScale.dim() == 2, "quant_scale must be a 2-D tensor, got dim=", quantScale.dim());
    TORCH_CHECK(localRows.numel() > 0, "local_rows must not be empty");

    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16, got ", x.scalar_type());
    TORCH_CHECK(localRows.scalar_type() == at::kLong, "local_rows must be int64, got ", localRows.scalar_type());
    TORCH_CHECK(quantScale.scalar_type() == at::kFloat, "quant_scale must be float32, got ",
                quantScale.scalar_type());

    TORCH_CHECK(quantScale.size(0) == localRows.numel(), "quant_scale first dim must equal local_rows numel");
    TORCH_CHECK(quantScale.size(1) == x.size(1), "quant_scale second dim must equal x hidden dim");

    TORCH_CHECK(x.size(0) <= std::numeric_limits<int32_t>::max(), "x rows exceed int32 range");
    TORCH_CHECK(x.size(1) <= std::numeric_limits<int32_t>::max(), "x hidden dim exceeds int32 range");
    TORCH_CHECK(localRows.numel() <= std::numeric_limits<int32_t>::max(), "local_rows numel exceeds int32 range");

    TORCH_CHECK(x.size(1) % BF16_32B_ALIGN == 0,
                "hidden must be aligned to 32 bytes for bfloat16. hidden=", x.size(1),
                ", required multiple=", BF16_32B_ALIGN);
}

} // namespace

at::Tensor moe_quant_meta(const at::Tensor &x, const at::Tensor &localRows, const at::Tensor &quantScale)
{
    check_moe_quant_common(x, localRows, quantScale);
    return at::empty({x.size(0), x.size(1)}, x.options().dtype(at::kChar));
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("moe_quant", moe_quant_meta);
}

at::Tensor moe_quant_npu(const at::Tensor &x, const at::Tensor &localRows, const at::Tensor &quantScale)
{
    check_moe_quant_common(x, localRows, quantScale);
    TORCH_CHECK(torch_npu::utils::is_npu(x), "x must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(localRows), "local_rows must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(quantScale), "quant_scale must be on NPU device");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(localRows.is_contiguous(), "local_rows must be contiguous");
    TORCH_CHECK(quantScale.is_contiguous(), "quant_scale must be contiguous");

    const c10::OptionalDeviceGuard guard(x.device());
    at::Tensor out = moe_quant_meta(x, localRows, quantScale);

    auto xPtr = static_cast<uint8_t *>(x.data_ptr());
    auto localRowsPtr = static_cast<uint8_t *>(localRows.data_ptr());
    auto quantScalePtr = static_cast<uint8_t *>(quantScale.data_ptr());
    auto outPtr = static_cast<uint8_t *>(out.data_ptr());
    const int32_t tokenRows = static_cast<int32_t>(x.size(0));
    const int32_t expertNum = static_cast<int32_t>(localRows.numel());
    const int32_t hidden = static_cast<int32_t>(x.size(1));
    const int32_t localRowsCount = static_cast<int32_t>(localRows.numel());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    auto aclCall = [=]() -> int {
        moe_quant_launch(xPtr, localRowsPtr, quantScalePtr, outPtr, tokenRows, expertNum, hidden, localRowsCount,
                         BLOCK_DIM, stream);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MoeQuant", aclCall);
    return out;
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("moe_quant", moe_quant_npu);
}

} // namespace MoeQuant
} // namespace npu_ops_transformer_ext
