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
 * \file moe_gate_silu.cpp
 * \brief Fast kernel launch implementation of gate_silu for MoE.
 */

#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <torch/all.h>
#include <torch/library.h>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {
namespace MoeGateSilu {

using namespace AscendC;

namespace {

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t UB_BUFFER_SIZE = 192512;
constexpr int32_t ACL_DTYPE_FLOAT16 = 1;
constexpr int32_t ACL_DTYPE_BFLOAT16 = 27;

#define MOE_GATE_SILU_TYPE_SWITCH(dtype, DType, ...) \
    switch (dtype) {                                 \
        case ACL_DTYPE_FLOAT16: {                    \
            using DType = half;                      \
            {                                        \
                __VA_ARGS__                          \
            }                                        \
            break;                                   \
        }                                            \
        case ACL_DTYPE_BFLOAT16: {                   \
            using DType = bfloat16_t;                \
            {                                        \
                __VA_ARGS__                          \
            }                                        \
            break;                                   \
        }                                            \
        default: {                                   \
            using DType = bfloat16_t;                \
            {                                        \
                __VA_ARGS__                          \
            }                                        \
            break;                                   \
        }                                            \
    }

int32_t get_acl_dtype(const at::ScalarType dtype)
{
    if (dtype == at::kHalf) {
        return ACL_DTYPE_FLOAT16;
    }
    if (dtype == at::kBFloat16) {
        return ACL_DTYPE_BFLOAT16;
    }
    TORCH_CHECK(false, "moe_gate_silu only supports float16 and bfloat16, got ", dtype);
}

template <typename T>
class MoeGateSiluKernel {
public:
    __aicore__ inline MoeGateSiluKernel() {}

    __aicore__ inline void InitTotal(int32_t perTotalCount, GM_ADDR totalRows, int32_t blockCount)
    {
        coreId_ = GetBlockIdx();
        realRows_ = -1;
        baseSizeTotal_ = 32 / sizeof(int64_t);
        tileLengthTotal_ = perTotalCount;
        perDimTotal_ = perTotalCount;
        alignTotalCount_ = DivCeil(perDimTotal_, baseSizeTotal_) * baseSizeTotal_;
        totalRowsGm_.SetGlobalBuffer((__gm__ int64_t *)totalRows, alignTotalCount_);
        pipe_.InitBuffer(inQueueTotalRows_, 2, alignTotalCount_ * sizeof(int64_t));
    }

    __aicore__ inline void ProcessTotal()
    {
        LocalTensor<int64_t> totalRowsLocal = inQueueTotalRows_.AllocTensor<int64_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(tileLengthTotal_ * sizeof(int64_t)), 0, 0};
        DataCopyPadParams padParams;
        DataCopyPad(totalRowsLocal, totalRowsGm_[0], copyParams, padParams);
        int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
        realRows_ = totalRowsLocal.GetValue(tileLengthTotal_ - 1);
        inQueueTotalRows_.FreeTensor(totalRowsLocal);
    }

    __aicore__ inline void Init(int32_t rows, int32_t perCount, GM_ADDR x, GM_ADDR z, int32_t blockCount)
    {
        coreId_ = GetBlockIdx();
        baseSize_ = 32 / sizeof(T);
        rows = realRows_ == -1 ? rows : realRows_;

        blockCount_ = rows < blockCount ? rows : blockCount;
        if (coreId_ < blockCount_) {
            tileLength_ = perCount;
            perDim_ = perCount / 2;

            int32_t rowsPerCore = static_cast<int32_t>(rows / blockCount_);
            rowsPerCorePerLoop_ = 1;
            alignCount_ = DivCeil(perDim_, baseSize_) * baseSize_;

            if (rowsPerCore >= 2) {
                int32_t limitRowsPerCore = UB_BUFFER_SIZE / (alignCount_ * BUFFER_NUM * 6 + alignCount_ * 8);
                rowsPerCorePerLoop_ = min(rowsPerCore, limitRowsPerCore);
            }
            tailRowsLastLoopPerCore_ = rowsPerCorePerLoop_;
            int32_t loopCount = rows / (rowsPerCorePerLoop_ * blockCount_);

            fullLoopCount_ = loopCount;
            fullRowsOffset_ = fullLoopCount_ * rowsPerCorePerLoop_ * blockCount_ * tileLength_;
            rowsOffset_ = 0;
            int32_t tailCount = rows - (fullLoopCount_ * blockCount_ * rowsPerCorePerLoop_);
            if (tailCount > 0) {
                loopCount += 1;
            }
            tailRowsLastLoopPerCore_ = tailCount / blockCount_;

            tailCount = tailCount % blockCount_;
            tailRowsOffset_ = tailCount * (tailRowsLastLoopPerCore_ + 1) * tileLength_;
            if (coreId_ < tailCount) {
                tailRowsLastLoopPerCore_ += 1;
            }

            loopCount_ = loopCount;
            tailCount_ = tailCount;

            xGm_.SetGlobalBuffer((__gm__ T *)x, rowsPerCorePerLoop_ * alignCount_);
            zGm_.SetGlobalBuffer((__gm__ T *)z, rowsPerCorePerLoop_ * alignCount_);

            pipe_.InitBuffer(inQueueX_, BUFFER_NUM, rowsPerCorePerLoop_ * alignCount_ * sizeof(T));
            pipe_.InitBuffer(inQueueY_, BUFFER_NUM, rowsPerCorePerLoop_ * alignCount_ * sizeof(T));
            pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, rowsPerCorePerLoop_ * alignCount_ * sizeof(T));
            pipe_.InitBuffer(workBuf_, rowsPerCorePerLoop_ * alignCount_ * sizeof(float));
            pipe_.InitBuffer(workBuf1_, rowsPerCorePerLoop_ * alignCount_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (coreId_ < blockCount_) {
            for (int32_t i = 0; i < loopCount_; ++i) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t loopCount)
    {
        LocalTensor<T> xLeftLocal = inQueueX_.AllocTensor<T>();
        LocalTensor<T> xRightLocal = inQueueY_.AllocTensor<T>();

        globalCoreIndex_ = loopCount * blockCount_ + coreId_;
        if (loopCount < fullLoopCount_) {
            rowsOffset_ = globalCoreIndex_ * rowsPerCorePerLoop_ * tileLength_;
        } else {
            rowsPerCorePerLoop_ = tailRowsLastLoopPerCore_;
            if (coreId_ < tailCount_) {
                rowsOffset_ = fullRowsOffset_ + coreId_ * rowsPerCorePerLoop_ * tileLength_;
            } else {
                rowsOffset_ = fullRowsOffset_ + tailRowsOffset_ +
                              (coreId_ - tailCount_) * rowsPerCorePerLoop_ * tileLength_;
            }
        }

        DataCopyParams copyParams{static_cast<uint16_t>(rowsPerCorePerLoop_),
                                  static_cast<uint16_t>(tileLength_ / 2 * sizeof(T)),
                                  static_cast<uint16_t>(tileLength_ / 2 * sizeof(T)), 0};
        DataCopyPadParams padParams;
        DataCopyPad(xLeftLocal, xGm_[rowsOffset_], copyParams, padParams);
        DataCopyPad(xRightLocal, xGm_[rowsOffset_ + tileLength_ / 2], copyParams, padParams);
        inQueueX_.EnQue(xLeftLocal);
        inQueueY_.EnQue(xRightLocal);
    }

    __aicore__ inline void Compute(int32_t loopCount)
    {
        LocalTensor<T> xLeftLocal = inQueueX_.DeQue<T>();
        LocalTensor<T> xRightLocal = inQueueY_.DeQue<T>();

        LocalTensor<T> zLocal = outQueueZ_.AllocTensor<T>();
        LocalTensor<float> xLeftFp32Local = workBuf_.Get<float>();
        LocalTensor<float> siluOutFp32Local = workBuf1_.Get<float>();
        Cast(xLeftFp32Local, xLeftLocal, RoundMode::CAST_NONE, rowsPerCorePerLoop_ * alignCount_);
        Silu(siluOutFp32Local, xLeftFp32Local, rowsPerCorePerLoop_ * alignCount_);
        Cast(xLeftFp32Local, xRightLocal, RoundMode::CAST_NONE, rowsPerCorePerLoop_ * alignCount_);
        Mul(siluOutFp32Local, siluOutFp32Local, xLeftFp32Local, rowsPerCorePerLoop_ * alignCount_);
        Cast(zLocal, siluOutFp32Local, RoundMode::CAST_ROUND, rowsPerCorePerLoop_ * alignCount_);

        outQueueZ_.EnQue(zLocal);

        inQueueX_.FreeTensor(xLeftLocal);
        inQueueY_.FreeTensor(xRightLocal);
        workBuf_.FreeTensor(xLeftFp32Local);
        workBuf1_.FreeTensor(siluOutFp32Local);
    }

    __aicore__ inline void CopyOut(int32_t loopCount)
    {
        LocalTensor<T> zLocal = outQueueZ_.DeQue<T>();
        DataCopyParams copyParams{static_cast<uint16_t>(1),
                                  static_cast<uint16_t>(rowsPerCorePerLoop_ * alignCount_ * sizeof(T)), 0, 0};
        DataCopyPad(zGm_[rowsOffset_ / 2], zLocal, copyParams);
        outQueueZ_.FreeTensor(zLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTotalRows_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;
    TBuf<QuePosition::VECCALC> workBuf1_;
    TBuf<QuePosition::VECCALC> workBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> zGm_;
    GlobalTensor<int64_t> totalRowsGm_;

    int32_t coreId_;
    int32_t tileLength_;
    int32_t tileLengthTotal_;
    int32_t perDim_;
    int32_t perDimTotal_;
    int32_t baseSize_;
    int32_t baseSizeTotal_;
    int32_t blockCount_;
    int32_t alignCount_;
    int32_t alignTotalCount_;
    int32_t tailCount_;
    int32_t fullRowsOffset_;
    int32_t tailRowsOffset_;
    int32_t realRows_;
    int32_t loopCount_;
    int32_t fullLoopCount_;
    int32_t rowsPerCorePerLoop_;
    int32_t tailRowsLastLoopPerCore_;
    int32_t globalCoreIndex_;
    int32_t rowsOffset_;
};

extern "C" __global__ __aicore__ void moe_gate_silu_kernel(GM_ADDR x, GM_ADDR totalRows, GM_ADDR z, int32_t rows,
                                                           int32_t perCount, int32_t perTotalCount,
                                                           int32_t blockCount, int32_t dtype)
{
    MOE_GATE_SILU_TYPE_SWITCH(dtype, T, {
        MoeGateSiluKernel<T> op;
        op.InitTotal(perTotalCount, totalRows, blockCount);
        op.ProcessTotal();
        op.Init(rows, perCount, x, z, blockCount);
        op.Process();
    });
}

void moe_gate_silu_launch(uint8_t *x, uint8_t *totalRows, uint8_t *out, int32_t rows, int32_t perCount,
                          int32_t perTotalCount, int32_t blockDim, aclrtStream stream, int32_t dtype)
{
    if (rows <= 0) {
        return;
    }
    int32_t realBlockDim = rows < blockDim ? rows : blockDim;
    moe_gate_silu_kernel<<<realBlockDim, nullptr, stream>>>(x, totalRows, out, rows, perCount, perTotalCount,
                                                            realBlockDim, dtype);
}

void check_moe_gate_silu_common(const at::Tensor &x, const at::Tensor &totalRows)
{
    TORCH_CHECK(x.dim() == 2, "x must be a 2-D tensor, got dim=", x.dim());
    TORCH_CHECK(x.size(0) >= 0, "x rows must be non-negative");
    TORCH_CHECK(x.size(1) > 0 && x.size(1) % 2 == 0,
                "x shape must be [rows, 2 * hidden], got second dim=", x.size(1));
    TORCH_CHECK(totalRows.dim() >= 1, "total_rows must have at least one dimension");
    TORCH_CHECK(totalRows.numel() > 0, "total_rows must not be empty");
    TORCH_CHECK(totalRows.scalar_type() == at::kLong, "total_rows must be int64");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x must be float16 or bfloat16, got ", x.scalar_type());

    const int64_t hidden = x.size(1) / 2;
    const int64_t elementsPer32Bytes = 32 / static_cast<int64_t>(x.element_size());
    TORCH_CHECK(hidden % elementsPer32Bytes == 0,
                "hidden must be aligned to 32 bytes for the current dtype. hidden=", hidden,
                ", required multiple=", elementsPer32Bytes);
    TORCH_CHECK(hidden <= 8192, "hidden=", hidden, " exceeds the validated fast-kernel limit 8192");
    TORCH_CHECK(x.size(0) <= std::numeric_limits<int32_t>::max(), "x rows exceed int32 range");
    TORCH_CHECK(x.size(1) <= std::numeric_limits<int32_t>::max(), "x second dim exceeds int32 range");
    TORCH_CHECK(totalRows.numel() <= std::numeric_limits<int32_t>::max(), "total_rows numel exceeds int32 range");
}

} // namespace

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("moe_gate_silu(Tensor x, Tensor total_rows) -> Tensor");
}

at::Tensor moe_gate_silu_meta(const at::Tensor &x, const at::Tensor &totalRows)
{
    check_moe_gate_silu_common(x, totalRows);
    return at::empty({x.size(0), x.size(1) / 2}, x.options());
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("moe_gate_silu", moe_gate_silu_meta);
}

at::Tensor moe_gate_silu_npu(const at::Tensor &x, const at::Tensor &totalRows)
{
    check_moe_gate_silu_common(x, totalRows);
    TORCH_CHECK(torch_npu::utils::is_npu(x), "x must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(totalRows), "total_rows must be on NPU device");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(totalRows.is_contiguous(), "total_rows must be contiguous");

    const c10::OptionalDeviceGuard guard(x.device());
    at::Tensor out = moe_gate_silu_meta(x, totalRows);
    if (x.size(0) == 0) {
        return out;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t blockDim = static_cast<int32_t>(ascendcPlatform->GetCoreNumAiv());
    TORCH_CHECK(blockDim > 0, "AIV core number must be positive");

    auto xPtr = static_cast<uint8_t *>(x.data_ptr());
    auto totalRowsPtr = static_cast<uint8_t *>(totalRows.data_ptr());
    auto outPtr = static_cast<uint8_t *>(out.data_ptr());
    int32_t rows = static_cast<int32_t>(x.size(0));
    int32_t perCount = static_cast<int32_t>(x.size(1));
    int32_t perTotalCount = static_cast<int32_t>(totalRows.numel());
    int32_t dtype = get_acl_dtype(x.scalar_type());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    auto aclCall = [=]() -> int {
        moe_gate_silu_launch(xPtr, totalRowsPtr, outPtr, rows, perCount, perTotalCount, blockDim, stream, dtype);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MoeGateSilu", aclCall);
    return out;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("moe_gate_silu", moe_gate_silu_npu);
}

} // namespace MoeGateSilu
} // namespace ascend_ops
