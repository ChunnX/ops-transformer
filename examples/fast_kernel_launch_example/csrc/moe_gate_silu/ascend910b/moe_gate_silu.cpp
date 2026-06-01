/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <tuple>

#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {
namespace MoeGateSiluOp {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t UB_BUFFER_SIZE = 192512;
constexpr int32_t ACL_DTYPE_FP16 = 1;
constexpr int32_t ACL_DTYPE_BF16 = 27;

#define TYPE_SWITCH(dtype, DType, ...) \
    switch (dtype) {                   \
        case ACL_DTYPE_FP16: {         \
            typedef half DType;        \
            {                          \
                __VA_ARGS__            \
            }                          \
        } break;                       \
        case ACL_DTYPE_BF16: {         \
            typedef bfloat16_t DType;  \
            {                          \
                __VA_ARGS__            \
            }                          \
        } break;                       \
        default: {                     \
            typedef bfloat16_t DType;  \
            {                          \
                __VA_ARGS__            \
            }                          \
        } break;                       \
    }

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("moe_gate_silu(Tensor x, Tensor total_rows) -> Tensor");
}

torch::Tensor moe_gate_silu_meta(const torch::Tensor &x, const torch::Tensor &total_rows)
{
    TORCH_CHECK(x.dim() == 2, "moe_gate_silu: x must be a 2-D tensor, got dim=", x.dim());
    TORCH_CHECK(x.size(0) > 0, "moe_gate_silu: x.size(0) must be positive");
    TORCH_CHECK(x.size(1) > 0, "moe_gate_silu: x.size(1) must be positive");
    TORCH_CHECK(x.size(1) % 2 == 0,
                "moe_gate_silu: x.size(1) must be even, got ", x.size(1));
    TORCH_CHECK(total_rows.dim() == 1,
                "moe_gate_silu: total_rows must be a 1-D int64 tensor, got dim=", total_rows.dim());
    TORCH_CHECK(total_rows.numel() > 0, "moe_gate_silu: total_rows must not be empty");
    TORCH_CHECK(total_rows.scalar_type() == at::kLong,
                "moe_gate_silu: total_rows dtype must be torch.int64");
    return torch::empty({x.size(0), x.size(1) / 2}, x.options());
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("moe_gate_silu", moe_gate_silu_meta);
}

template <typename T>
class MoeGateSilu {
public:
    __aicore__ inline MoeGateSilu() {}

    __aicore__ inline void InitTotal(int32_t perTotalCount, GM_ADDR totalRows, int32_t blockCnt)
    {
        coreId_ = GetBlockIdx();
        realRows_ = -1;
        baseSizeTotal_ = 32 / sizeof(int64_t);
        tileLengthTotal_ = perTotalCount;
        perDimTotal_ = perTotalCount;
        alignTotalCount_ = DivCeil(perDimTotal_, baseSizeTotal_) * baseSizeTotal_;
        totalRowsGm_.SetGlobalBuffer((__gm__ int64_t *)totalRows, alignTotalCount_);
        pipe_.InitBuffer(inQueueTotalRows_, BUFFER_NUM, alignTotalCount_ * sizeof(int64_t));
    }

    __aicore__ inline void ProcessTotal()
    {
        LocalTensor<int64_t> totalRowsLocal = inQueueTotalRows_.AllocTensor<int64_t>();
        DataCopyParams copyParams{1, (uint16_t)(tileLengthTotal_ * sizeof(int64_t)), 0, 0};
        DataCopyPadParams padParams;
        DataCopyPad(totalRowsLocal, totalRowsGm_[0], copyParams, padParams);
        int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
        realRows_ = totalRowsLocal.GetValue(tileLengthTotal_ - 1);
        inQueueTotalRows_.FreeTensor(totalRowsLocal);
    }

    __aicore__ inline void Init(int32_t rows, int32_t perCount, GM_ADDR x, GM_ADDR z, int32_t blockCnt)
    {
        coreId_ = GetBlockIdx();
        baseSize_ = 32 / sizeof(T);
        rows = realRows_ == -1 ? rows : realRows_;

        blockCnt_ = rows < blockCnt ? rows : blockCnt;
        if (coreId_ < blockCnt_) {
            tileLength_ = perCount;
            perDim_ = perCount / 2;
            int32_t rowsPerCore = int(rows / blockCnt_);
            rowsPerCorePerLoop_ = 1;

            alignCount_ = DivCeil(perDim_, baseSize_) * baseSize_;

            if (rowsPerCore >= 2) {
                int32_t limitRowsPerCore = UB_BUFFER_SIZE / (alignCount_ * BUFFER_NUM * 6 + alignCount_ * 8);
                rowsPerCorePerLoop_ = min(rowsPerCore, limitRowsPerCore);
            }
            tailRowsLastLoopPerCore_ = rowsPerCorePerLoop_;
            int32_t loopCount = rows / (rowsPerCorePerLoop_ * blockCnt_);

            fullLoopCount_ = loopCount;
            fullRowsOffset_ = fullLoopCount_ * rowsPerCorePerLoop_ * blockCnt_ * tileLength_;
            rowsOffset_ = 0;
            int32_t tailCount = rows - (fullLoopCount_ * blockCnt_ * rowsPerCorePerLoop_);
            if (tailCount > 0) {
                loopCount += 1;
            }
            tailRowsLastLoopPerCore_ = tailCount / blockCnt_;

            tailCount = tailCount % blockCnt_;
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
        if (coreId_ < blockCnt_) {
            for (int32_t i = 0; i < loopCount_; ++i) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t loopCnt)
    {
        LocalTensor<T> xLeftLocal = inQueueX_.AllocTensor<T>();
        LocalTensor<T> xRightLocal = inQueueY_.AllocTensor<T>();

        int32_t globalCoreIndex = loopCnt * blockCnt_ + coreId_;
        if (loopCnt < fullLoopCount_) {
            rowsOffset_ = globalCoreIndex * rowsPerCorePerLoop_ * tileLength_;
        } else {
            rowsPerCorePerLoop_ = tailRowsLastLoopPerCore_;
            if (coreId_ < tailCount_) {
                rowsOffset_ = fullRowsOffset_ + coreId_ * rowsPerCorePerLoop_ * tileLength_;
            } else {
                rowsOffset_ =
                    fullRowsOffset_ + tailRowsOffset_ + (coreId_ - tailCount_) * rowsPerCorePerLoop_ * tileLength_;
            }
        }

        DataCopyParams copyParams{(uint16_t)rowsPerCorePerLoop_, (uint16_t)(tileLength_ / 2 * sizeof(T)),
                                  (uint16_t)(tileLength_ / 2 * sizeof(T)), 0};
        DataCopyPadParams padParams;
        DataCopyPad(xLeftLocal, xGm_[rowsOffset_], copyParams, padParams);
        DataCopyPad(xRightLocal, xGm_[rowsOffset_ + tileLength_ / 2], copyParams, padParams);
        inQueueX_.EnQue(xLeftLocal);
        inQueueY_.EnQue(xRightLocal);
    }

    __aicore__ inline void Compute(int32_t loopCnt)
    {
        (void)loopCnt;
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

    __aicore__ inline void CopyOut(int32_t loopCnt)
    {
        (void)loopCnt;
        LocalTensor<T> zLocal = outQueueZ_.DeQue<T>();
        DataCopyParams copyParams{1, (uint16_t)(rowsPerCorePerLoop_ * alignCount_ * sizeof(T)), 0, 0};
        DataCopyPad(zGm_[rowsOffset_ / 2], zLocal, copyParams);
        outQueueZ_.FreeTensor(zLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTotalRows_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;
    TBuf<QuePosition::VECCALC> workBuf_;
    TBuf<QuePosition::VECCALC> workBuf1_;
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
    int32_t blockCnt_;
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
    int32_t rowsOffset_;
};

extern "C" __global__ __aicore__ void moe_gate_silu_kernel(GM_ADDR x, GM_ADDR totalRows, GM_ADDR z,
                                                            int32_t numRows, int32_t perCount,
                                                            int32_t perTotalCount, int32_t blockCnt,
                                                            int32_t dtype)
{
    TYPE_SWITCH(dtype, T, {
        MoeGateSilu<T> op;
        op.InitTotal(perTotalCount, totalRows, blockCnt);
        op.ProcessTotal();
        op.Init(numRows, perCount, x, z, blockCnt);
        op.Process();
    });
}

void moe_gate_silu_launch(uint8_t *x, uint8_t *totalRows, uint8_t *out, int32_t numRows, int32_t perCount,
                          int32_t perTotalCount, int32_t blockDim, void *stream, int32_t dtype)
{
    int32_t realBlockDim = numRows < blockDim ? numRows : blockDim;
    moe_gate_silu_kernel<<<realBlockDim, nullptr, stream>>>(x, totalRows, out, numRows, perCount, perTotalCount,
                                                            realBlockDim, dtype);
}

int32_t get_acl_dtype(const torch::Tensor &x)
{
    if (x.scalar_type() == at::kHalf) {
        return ACL_DTYPE_FP16;
    }
    if (x.scalar_type() == at::kBFloat16) {
        return ACL_DTYPE_BF16;
    }
    TORCH_CHECK(false, "moe_gate_silu: only torch.float16 and torch.bfloat16 are supported");
}

int32_t get_aiv_core_num()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    TORCH_CHECK(coreNum > 0 && coreNum <= std::numeric_limits<int32_t>::max(),
                "moe_gate_silu: invalid AIV core number ", coreNum);
    return static_cast<int32_t>(coreNum);
}

void check_tensor_limits(const torch::Tensor &x, const torch::Tensor &totalRows)
{
    TORCH_CHECK(x.size(0) <= std::numeric_limits<int32_t>::max(),
                "moe_gate_silu: x.size(0) exceeds int32 range");
    TORCH_CHECK(x.size(1) <= std::numeric_limits<int32_t>::max(),
                "moe_gate_silu: x.size(1) exceeds int32 range");
    TORCH_CHECK(totalRows.numel() <= std::numeric_limits<int32_t>::max(),
                "moe_gate_silu: total_rows.numel() exceeds int32 range");
}

torch::Tensor moe_gate_silu_npu(const torch::Tensor &x, const torch::Tensor &totalRows)
{
    TORCH_CHECK(torch_npu::utils::is_npu(x), "moe_gate_silu: x must be on NPU");
    TORCH_CHECK(torch_npu::utils::is_npu(totalRows), "moe_gate_silu: total_rows must be on NPU");
    TORCH_CHECK(x.device() == totalRows.device(), "moe_gate_silu: x and total_rows must be on the same device");
    TORCH_CHECK(x.is_contiguous(), "moe_gate_silu: x must be contiguous");
    TORCH_CHECK(totalRows.is_contiguous(), "moe_gate_silu: total_rows must be contiguous");

    const c10::OptionalDeviceGuard guard(x.device());
    auto out = moe_gate_silu_meta(x, totalRows);
    check_tensor_limits(x, totalRows);

    int32_t numRows = static_cast<int32_t>(x.size(0));
    int32_t perCount = static_cast<int32_t>(x.size(1));
    int32_t perTotalCount = static_cast<int32_t>(totalRows.numel());
    int32_t blockDim = get_aiv_core_num();
    int32_t dtype = get_acl_dtype(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    auto xPtr = reinterpret_cast<uint8_t *>(x.data_ptr());
    auto totalRowsPtr = reinterpret_cast<uint8_t *>(totalRows.data_ptr());
    auto outPtr = reinterpret_cast<uint8_t *>(out.data_ptr());

    auto aclCall = [=]() -> int {
        moe_gate_silu_launch(xPtr, totalRowsPtr, outPtr, numRows, perCount, perTotalCount, blockDim, stream, dtype);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MoeGateSilu", aclCall);
    return out;
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("moe_gate_silu", moe_gate_silu_npu);
}

#undef TYPE_SWITCH

} // namespace MoeGateSiluOp
} // namespace ascend_ops
