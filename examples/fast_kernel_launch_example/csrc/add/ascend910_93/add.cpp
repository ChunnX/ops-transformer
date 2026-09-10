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
 * \file add.cpp
 * \brief
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <type_traits>

namespace ascend_ops {
namespace Add {

// Register the operator's schema
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("add(Tensor x, Tensor y) -> Tensor");
    m.def("read_cycle(Tensor dummy) -> Tensor");
    m.def("read_cycle_launch_count(Tensor dummy) -> int");
    m.def("reset_cycle(Tensor dummy) -> ()");
}

// Meta function implementation of Add
torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
{
    TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
    auto z = torch::empty_like(x);
    return z;
}

// Register the Meta implementation
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("add", add_meta);
}

std::tuple<int64_t, int64_t, int64_t> calc_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    constexpr static int64_t PIPELINE_DEPTH = 2;
    constexpr static int64_t BUFFER_NUM = 3;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    TORCH_CHECK(coreNum > 0, "coreNum must be positive.");
    int64_t numBlocks = 40;
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    int64_t tileSize = ubSize / PIPELINE_DEPTH / BUFFER_NUM;
    return std::make_tuple(numBlocks, blockLength, tileSize);
}

template <typename T>
__global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength,
                                      uint32_t tileSize, GM_ADDR cycleData)
{
    // T0: kernel 第一行
    int64_t T0 = AscendC::GetSystemCycle();

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr static int64_t PIPELINE_DEPTH = 2;
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> xGm, yGm, zGm;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(inQueueY, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileSize);
    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ T *)y + blockLength * AscendC::GetBlockIdx());
    zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());

    int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) {
        currentBlockLength = blockLength;
    }
    int64_t elementNumPerTile = tileSize / sizeof(T);
    int64_t tileNum = currentBlockLength / elementNumPerTile;
    int64_t tailTileElementNum = currentBlockLength - tileNum * elementNumPerTile;

    // T1: prologue 之后
    int64_t T1 = AscendC::GetSystemCycle();
    __gm__ int64_t *cycleBuf = (__gm__ int64_t *)cycleData + AscendC::GetBlockIdx() * 3;
    cycleBuf[0] = T0;
    cycleBuf[1] = T1;

    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t offset = i * elementNumPerTile;
        // CopyIn
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = elementNumPerTile * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        // Compute
        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        AscendC::Add(zLocal, xLocal, yLocal, elementNumPerTile);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        // CopyOut
        zLocal = outQueueZ.DeQue<T>();
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

    if (tailTileElementNum > 0) {
        int64_t offset = tileNum * elementNumPerTile;
        // CopyIn
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = tailTileElementNum * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        // Compute
        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        AscendC::Add(zLocal, xLocal, yLocal, tailTileElementNum);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        // CopyOut
        zLocal = outQueueZ.DeQue<T>();
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

    // T2: kernel 末尾（所有计算完成后）
    // [对照实验] 等 MTE3 搬出排空后再打点: 验证 pmu_end-T2 硬件尾可被同步消除
    // 官方仓数据依赖护栏同款原语 (ops-transformer 中 133 处同款写法)
    {
        event_t mte3DoneEvent = static_cast<event_t>(pipe.AllocEventID<AscendC::HardEvent::MTE3_S>());
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(mte3DoneEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(mte3DoneEvent);
        pipe.ReleaseEventID<AscendC::HardEvent::MTE3_S>(mte3DoneEvent);
    }
    int64_t T2 = AscendC::GetSystemCycle();
    cycleBuf[2] = T2;
}

// 持久 cycle buffer：固定 GM 地址，跨 launch 复用（replay/capture 模式必需）
// add_npu 内不做任何 host 回读（回读与 NPUGraph capture 互斥），
// 由 read_cycle() getter 在 capture/replay 外部调用读取。
static torch::Tensor g_cycle_tensor;  // [64*3] int64, NPU device
static int64_t g_cycle_launch_count = 0;  // 累计 launch 次数

torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y)
{
    const c10::OptionalDeviceGuard guard(x.device());
    auto z = add_meta(x, y);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t totalLength, numBlocks, blockLength, tileSize;
    totalLength = x.numel();
    std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(totalLength);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto z_ptr = (GM_ADDR)z.data_ptr();
    // 持久 buffer 懒初始化（一次分配，固定地址；capture 时该地址被固化进 kernel 参数）
    if (!g_cycle_tensor.defined() || g_cycle_tensor.device() != x.device()) {
        auto cycle_options = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
        g_cycle_tensor = torch::zeros({64 * 3}, cycle_options);  // 最多 64 blocks × 3 slots (T0/T1/T2)
    }
    auto cycle_ptr = (GM_ADDR)g_cycle_tensor.data_ptr();
    ++g_cycle_launch_count;
    auto acl_call = [=]() -> int {
        AT_DISPATCH_SWITCH(x.scalar_type(), "add_npu", AT_DISPATCH_CASE(torch::kFloat32, [&] {
                               using scalar_t = float;
                               add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength,
                                                                                      blockLength, tileSize, cycle_ptr);
                           }) AT_DISPATCH_CASE(torch::kFloat16, [&] {
                               using scalar_t = half;
                               add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength,
                                                                                      blockLength, tileSize, cycle_ptr);
                           }) AT_DISPATCH_CASE(torch::kInt32, [&] {
                               using scalar_t = int32_t;
                               add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength,
                                                                                      blockLength, tileSize, cycle_ptr);
                           }));
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Add", acl_call);
    return z;
}

// Getter: 读取持久 cycle buffer（capture/replay 外部调用）
// 返回 [64, 3] int64 CPU tensor，仅 t0>0 的行有效
// 注意：capture 阶段不可调用（含 D2H 拷贝），replay 之后或 eager 模式下调用
torch::Tensor read_cycle(const torch::Tensor &dummy)
{
    TORCH_CHECK(g_cycle_tensor.defined(), "cycle buffer not initialized, call add first");
    return g_cycle_tensor.to(torch::kCPU).clone();
}

// Getter: 累计 launch 次数（capture 阶段 kernel 只录不跑，但 host 计数仍会增加）
int64_t read_cycle_launch_count(const torch::Tensor &dummy)
{
    return g_cycle_launch_count;
}

// Reset: 清零 cycle buffer 并归零计数（新测试会话开始时调用；不可在 capture 中调用）
void reset_cycle(const torch::Tensor &dummy)
{
    TORCH_CHECK(g_cycle_tensor.defined(), "cycle buffer not initialized, call add first");
    g_cycle_tensor.zero_();
    g_cycle_launch_count = 0;
}

// Register the NPU implementation
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("add", add_npu);
    m.impl("read_cycle", &read_cycle);
    m.impl("read_cycle_launch_count", &read_cycle_launch_count);
    m.impl("reset_cycle", &reset_cycle);
}

} // namespace Add
} // namespace ascend_ops
