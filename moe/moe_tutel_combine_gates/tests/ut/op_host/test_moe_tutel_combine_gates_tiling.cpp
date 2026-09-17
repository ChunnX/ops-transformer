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
 * \file test_moe_tutel_combine_gates_tiling.cpp
 * \brief Host tiling unit tests for MoeTutelCombineGates.
 *
 *        Every expected value is derived in this file from the tiling contract
 *        itself (taskNumPerBlock = CeilAlign(CeilDiv(S*B, aivCoreNum), 256), the two
 *        UB route tiers of CalcUbTiers, the dtype -> tilingKey route table), never
 *        from the tiling function under test.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../../../op_host/moe_tutel_combine_gates_tiling_arch35.h"
#include "../../../op_kernel/arch35/moe_tutel_combine_gates_tiling_struct.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {

constexpr int64_t kOffset = 256;     // per-core task-count alignment granularity
constexpr int64_t kInnerOffset = 16; // tasks in one inner transfer block
constexpr int64_t kBlockSize = 32;   // Ops::Base::GetUbBlockSize (platform constant)
constexpr int64_t kVectorSize = 256; // Ops::Base::GetVRegSize (platform constant)
constexpr uint64_t kAivCoreNum = 40; // AIV core count handed to the tiling context
constexpr uint64_t kUbSize = 262144; // UB bytes per core handed to the tiling context

constexpr int64_t kKeyBf16 = 0;
constexpr int64_t kKeyFp32 = 1;
constexpr int64_t kKeyFp16 = 2;

int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}
int64_t CeilAlign(int64_t x, int64_t align)
{
    return CeilDiv(x, align) * align;
}

// The two H-long UB route tiers the host hands to the kernel.
struct UbTiers {
    int64_t rowElems;
    int64_t fp32Elems;
};

UbTiers ExpectedUbTiers(int64_t hidden, bool isFp32)
{
    const int64_t elemBytes = isFp32 ? 4 : 2;
    const int64_t rowAlign = (kBlockSize > kVectorSize ? kBlockSize : kVectorSize) / elemBytes;
    const int64_t fp32Align = (kBlockSize > kVectorSize ? kBlockSize : kVectorSize) / 4;
    const int64_t h = (hidden > 1) ? hidden : 1;
    return {CeilAlign(h, rowAlign), CeilAlign(h, fp32Align)};
}

int64_t ExpectedUbNeedBytes(int64_t hidden, bool isFp32)
{
    const UbTiers tiers = ExpectedUbTiers(hidden, isFp32);
    const int64_t small32 = CeilAlign(kInnerOffset * 4, kBlockSize);
    const int64_t small16 = CeilAlign(kInnerOffset * 2, kBlockSize);
    const int64_t regBytes = (kBlockSize > kVectorSize) ? kBlockSize : kVectorSize;
    if (isFp32) {
        // idx/loc/tmp (3 x small32) + sum (regBytes) + out (16 x 4)
        return 3 * small32 + regBytes + kInnerOffset * 4 + 2 * tiers.rowElems * 4 + 2 * tiers.fp32Elems * 4;
    }
    // idx/loc/tmp (3 x small32) + sum (regBytes) + out (16 x 2) + zTmp (regBytes)
    return 3 * small32 + regBytes + kInnerOffset * 2 + regBytes + 2 * tiers.rowElems * 2 + 4 * tiers.fp32Elems * 4;
}

int64_t ExpectedTaskNumPerBlock(int64_t samples, int64_t batch)
{
    const int64_t taskNum = samples * batch;
    if (taskNum == 0) {
        return 0;
    }
    return CeilAlign(CeilDiv(taskNum, static_cast<int64_t>(kAivCoreNum)), kOffset);
}

std::vector<gert::TilingContextPara::TensorDescription> MakeInputs(int64_t s, int64_t h, int64_t b, int64_t c,
                                                                   ge::DataType fcDtype, ge::DataType idxDtype,
                                                                   ge::Format format = ge::FORMAT_ND)
{
    const std::initializer_list<int64_t> xShape = {s, h};
    const std::initializer_list<int64_t> yShape = {b, c, h};
    const std::initializer_list<int64_t> iShape = {b, s};
    return {
        {gert::StorageShape(xShape, xShape), fcDtype, format},
        {gert::StorageShape(yShape, yShape), fcDtype, format},
        {gert::StorageShape(iShape, iShape), idxDtype, ge::FORMAT_ND},
        {gert::StorageShape(iShape, iShape), idxDtype, ge::FORMAT_ND},
    };
}

std::vector<gert::TilingContextPara::TensorDescription> MakeOutputs(int64_t s, int64_t b, ge::DataType fcDtype)
{
    return {{gert::StorageShape({b, s}, {b, s}), fcDtype, ge::FORMAT_ND}};
}

// The tiling callback queries the platform live and never reads the compile info,
// but the context builder still needs a valid carrier object.
optiling::MoeTutelCombineGatesCompileInfo g_compileInfo = {kAivCoreNum, kUbSize};

gert::TilingContextPara MakePara(const std::vector<gert::TilingContextPara::TensorDescription> &inputs,
                                 const std::vector<gert::TilingContextPara::TensorDescription> &outputs)
{
    return gert::TilingContextPara("MoeTutelCombineGates", inputs, outputs, &g_compileInfo, "Ascend950", kAivCoreNum,
                                   kUbSize);
}

// Runs the tiling function and, on success, checks the seven TilingData fields,
// the block dim and the tiling key against their independently derived values.
void ExpectTilingOk(const gert::TilingContextPara &para, int64_t s, int64_t h, int64_t b, int64_t c,
                    int64_t expectedKey, bool isFp32)
{
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info)) << "tiling returned failure";
    ASSERT_EQ(info.tilingDataSize, sizeof(MoeTutelCombineGatesTilingData));
    const auto *td = reinterpret_cast<const MoeTutelCombineGatesTilingData *>(info.tilingData.get());
    const int64_t taskNum = s * b;

    EXPECT_EQ(info.tilingKey, static_cast<uint64_t>(expectedKey));
    EXPECT_EQ(td->samples, s);
    EXPECT_EQ(td->hidden, h);
    EXPECT_EQ(td->batchSize, b);
    EXPECT_EQ(td->capacity, c);
    EXPECT_EQ(td->taskNumPerBlock, ExpectedTaskNumPerBlock(s, b));

    const UbTiers tiers = ExpectedUbTiers(h, isFp32);
    EXPECT_EQ(td->ubRowElems, tiers.rowElems);
    EXPECT_EQ(td->ubFp32Elems, tiers.fp32Elems);

    // The empty task axis is the only path that reduces the launched core count.
    EXPECT_EQ(info.blockNum, taskNum == 0 ? 1u : kAivCoreNum);
    // The operator declares no workspace of its own.
    for (size_t ws : info.workspaceSizes) {
        EXPECT_EQ(ws, 0u);
    }
}

void ExpectTilingFail(const gert::TilingContextPara &para)
{
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(para, info));
}

class MoeTutelCombineGatesTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeTutelCombineGatesTiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "MoeTutelCombineGatesTiling TearDown" << std::endl;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Success cases: the three dtype combinations and the shape forms the tiling owns.
// ---------------------------------------------------------------------------
TEST_F(MoeTutelCombineGatesTiling, tiling_bf16_key0)
{
    const int64_t s = 1024, h = 1024, b = 8, c = 256;
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT32), MakeOutputs(s, b, ge::DT_BF16));
    ExpectTilingOk(para, s, h, b, c, kKeyBf16, false);
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fp32_key1)
{
    const int64_t s = 500, h = 64, b = 2, c = 500;
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_FLOAT, ge::DT_INT32), MakeOutputs(s, b, ge::DT_FLOAT));
    ExpectTilingOk(para, s, h, b, c, kKeyFp32, true);
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fp16_key2)
{
    const int64_t s = 128, h = 513, b = 2, c = 64;
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_FLOAT16, ge::DT_INT32), MakeOutputs(s, b, ge::DT_FLOAT16));
    ExpectTilingOk(para, s, h, b, c, kKeyFp16, false);
}

// H = 1 degenerates the reduction axis; every UB route still holds one element.
TEST_F(MoeTutelCombineGatesTiling, tiling_hidden_one)
{
    const int64_t s = 4, h = 1, b = 2, c = 3;
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_FLOAT, ge::DT_INT32), MakeOutputs(s, b, ge::DT_FLOAT));
    ExpectTilingOk(para, s, h, b, c, kKeyFp32, true);
}

// H = 0 with S, B > 0 is not an empty tensor: taskNum stays S*B and the kernel
// writes exactly 0 from its `hidden <= 0` short circuit.
TEST_F(MoeTutelCombineGatesTiling, tiling_hidden_zero_not_empty)
{
    const int64_t s = 4, h = 0, b = 2, c = 3;
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_FLOAT, ge::DT_INT32), MakeOutputs(s, b, ge::DT_FLOAT));
    ExpectTilingOk(para, s, h, b, c, kKeyFp32, true);
}

// The empty task axis short circuits to taskNumPerBlock = 0 / blockDim 1, but the
// shape fields keep the real shapes AND the tiling key still follows the dtype.
TEST_F(MoeTutelCombineGatesTiling, tiling_empty_tensor_keeps_dtype_key)
{
    const int64_t s = 0, h = 8, b = 2, c = 3;
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_FLOAT, ge::DT_INT32), MakeOutputs(s, b, ge::DT_FLOAT));
    ExpectTilingOk(para, s, h, b, c, kKeyFp32, true);
}

TEST_F(MoeTutelCombineGatesTiling, tiling_empty_batch)
{
    const int64_t s = 4, h = 8, b = 0, c = 3;
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT32), MakeOutputs(s, b, ge::DT_BF16));
    ExpectTilingOk(para, s, h, b, c, kKeyBf16, false);
}

// taskNum exactly on a core boundary, and one task below it (core tail).
TEST_F(MoeTutelCombineGatesTiling, tiling_core_quota_boundary)
{
    const int64_t s = 16, h = 128, b = 40, c = 4; // taskNum = 640 = 40 * 16
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT32), MakeOutputs(s, b, ge::DT_BF16));
    ExpectTilingOk(para, s, h, b, c, kKeyBf16, false);
}

// The largest H the UB budget of this platform admits on the 16b path, and the
// first one above it (checked in the failure case below).
TEST_F(MoeTutelCombineGatesTiling, tiling_hidden_at_ub_bound_16b)
{
    int64_t h = 1;
    while (ExpectedUbNeedBytes(h + 128, false) <= static_cast<int64_t>(kUbSize)) {
        h += 128;
    }
    const int64_t s = 8, b = 2, c = 4;
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT32), MakeOutputs(s, b, ge::DT_BF16));
    ExpectTilingOk(para, s, h, b, c, kKeyBf16, false);
}

TEST_F(MoeTutelCombineGatesTiling, tiling_hidden_at_ub_bound_fp32)
{
    int64_t h = 1;
    while (ExpectedUbNeedBytes(h + 64, true) <= static_cast<int64_t>(kUbSize)) {
        h += 64;
    }
    const int64_t s = 8, b = 2, c = 4;
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_FLOAT, ge::DT_INT32), MakeOutputs(s, b, ge::DT_FLOAT));
    ExpectTilingOk(para, s, h, b, c, kKeyFp32, true);
}

// ---------------------------------------------------------------------------
// Rejection cases: rank, format, dtype pairing, shape binding, numeric bounds
// and the UB capacity gate.
// ---------------------------------------------------------------------------
TEST_F(MoeTutelCombineGatesTiling, tiling_fail_rank_x)
{
    const int64_t s = 8, h = 64, b = 2, c = 4;
    auto inputs = MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT32);
    inputs[0] = {gert::StorageShape({b, s, h}, {b, s, h}), ge::DT_BF16, ge::FORMAT_ND}; // x must be rank 2
    ExpectTilingFail(MakePara(inputs, MakeOutputs(s, b, ge::DT_BF16)));
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fail_rank_ygrad)
{
    const int64_t s = 8, h = 64, b = 2, c = 4;
    auto inputs = MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT32);
    inputs[1] = {gert::StorageShape({c, h}, {c, h}), ge::DT_BF16, ge::FORMAT_ND}; // y_grad must be rank 3
    ExpectTilingFail(MakePara(inputs, MakeOutputs(s, b, ge::DT_BF16)));
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fail_format_not_nd)
{
    const int64_t s = 8, h = 64, b = 2, c = 4;
    auto inputs = MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT32, ge::FORMAT_FRACTAL_NZ);
    ExpectTilingFail(MakePara(inputs, MakeOutputs(s, b, ge::DT_BF16)));
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fail_dtype_pair_mismatch)
{
    const int64_t s = 8, h = 64, b = 2, c = 4;
    auto inputs = MakeInputs(s, h, b, c, ge::DT_FLOAT, ge::DT_INT32);
    inputs[1] = {gert::StorageShape({b, c, h}, {b, c, h}), ge::DT_BF16, ge::FORMAT_ND};
    ExpectTilingFail(MakePara(inputs, MakeOutputs(s, b, ge::DT_FLOAT)));
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fail_dtype_not_supported)
{
    const int64_t s = 8, h = 64, b = 2, c = 4;
    auto inputs = MakeInputs(s, h, b, c, ge::DT_DOUBLE, ge::DT_INT32);
    ExpectTilingFail(MakePara(inputs, MakeOutputs(s, b, ge::DT_DOUBLE)));
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fail_index_dtype_not_int32)
{
    const int64_t s = 8, h = 64, b = 2, c = 4;
    auto inputs = MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT64);
    ExpectTilingFail(MakePara(inputs, MakeOutputs(s, b, ge::DT_BF16)));
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fail_hidden_binding)
{
    const int64_t s = 8, b = 2, c = 4;
    auto inputs = MakeInputs(s, 64, b, c, ge::DT_BF16, ge::DT_INT32);
    // y_grad.shape[2] = 32 disagrees with x.shape[1] = 64
    inputs[1] = {gert::StorageShape({b, c, 32}, {b, c, 32}), ge::DT_BF16, ge::FORMAT_ND};
    ExpectTilingFail(MakePara(inputs, MakeOutputs(s, b, ge::DT_BF16)));
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fail_batch_binding)
{
    const int64_t s = 8, h = 64, c = 4;
    auto inputs = MakeInputs(s, h, 2, c, ge::DT_BF16, ge::DT_INT32);
    inputs[2] = {gert::StorageShape({3, s}, {3, s}), ge::DT_INT32, ge::FORMAT_ND}; // indices.shape[0] != B
    ExpectTilingFail(MakePara(inputs, MakeOutputs(s, 2, ge::DT_BF16)));
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fail_task_element_count)
{
    const int64_t s = 8, h = 64, b = 2, c = 4;
    auto inputs = MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT32);
    inputs[3] = {gert::StorageShape({b, s - 1}, {b, s - 1}), ge::DT_INT32, ge::FORMAT_ND};
    ExpectTilingFail(MakePara(inputs, MakeOutputs(s, b, ge::DT_BF16)));
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fail_output_shape)
{
    const int64_t s = 8, h = 64, b = 2, c = 4;
    auto outputs = MakeOutputs(s, b, ge::DT_BF16);
    outputs[0] = {gert::StorageShape({b, s - 1}, {b, s - 1}), ge::DT_BF16, ge::FORMAT_ND};
    ExpectTilingFail(MakePara(MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT32), outputs));
}

TEST_F(MoeTutelCombineGatesTiling, tiling_fail_hidden_above_ub_bound)
{
    int64_t h = 1;
    while (ExpectedUbNeedBytes(h + 128, false) <= static_cast<int64_t>(kUbSize)) {
        h += 128;
    }
    h += 128; // first H the budget cannot hold
    const int64_t s = 8, b = 2, c = 4;
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT32), MakeOutputs(s, b, ge::DT_BF16));
    ExpectTilingFail(para);
}

TEST_F(MoeTutelCombineGatesTiling, tiling_ok_capacity_not_in_ub_budget)
{
    // C only enters the y_grad element count, not the UB budget, so a large C with
    // a small H must still succeed; this case pins that C is NOT part of the gate.
    const int64_t s = 8, h = 64, b = 2, c = 1 << 20;
    auto para = MakePara(MakeInputs(s, h, b, c, ge::DT_BF16, ge::DT_INT32), MakeOutputs(s, b, ge::DT_BF16));
    ExpectTilingOk(para, s, h, b, c, kKeyBf16, false);
}
