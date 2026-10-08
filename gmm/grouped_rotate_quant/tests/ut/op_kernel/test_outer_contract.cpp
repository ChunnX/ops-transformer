// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include <cassert>
#include <cstring>
#include <iostream>
#include "../../../op_kernel/arch35/grouped_rotate_quant_tensor_api.h"
#include "../../../op_host/op_tiling/arch35/grouped_rotate_quant_planner.h"

namespace {
int calls = 0;
uint8_t addresses[6]{};
GroupedRotateQuant::TilingData expected{};
struct RecordingKernel {
    void operator()(const GroupedRotateQuant::KernelArguments &args)
    {
        ++calls;
        assert(AscendC::activePipes == 1);
        assert(args.xList == &addresses[0]);
        assert(args.rotationList == &addresses[1]);
        assert(args.alpha == (expected.hasAlpha ? &addresses[2] : nullptr));
        assert(args.groupList == (expected.hasGroupList ? &addresses[3] : nullptr));
        assert(args.yList == &addresses[4]);
        assert(args.scaleList == &addresses[5]);
        assert(std::memcmp(&args.tiling, &expected, sizeof(expected)) == 0);
    }
};
} // namespace

#ifndef GRQ_TEST_MISSING_BACKEND
namespace GroupedRotateQuant {
template <>
struct TensorApiBackend<DTYPE_X, DTYPE_Y> {
    static constexpr bool AVAILABLE = true;
    using Kernel = RecordingKernel;
};
} // namespace GroupedRotateQuant
#endif

// Compile and exercise the actual device entry with interface substitutes.
#include "../../../op_kernel/grouped_rotate_quant_apt.cpp"

namespace {
void Invoke()
{
    grouped_rotate_quant(&addresses[0], &addresses[1], &addresses[2], &addresses[3],
        &addresses[4], &addresses[5], nullptr, reinterpret_cast<GM_ADDR>(&expected));
    assert(AscendC::activePipes == 0);
    assert(testTaskType == KERNEL_TYPE_MIX_AIC_1_2);
}
void MakePlan(optiling::grq::InputParams p)
{
    using namespace optiling::grq;
    const Hardware hardware{24, 48, 512 * 1024, 64 * 1024, 64 * 1024, 256 * 1024, 248 * 1024};
    if constexpr (std::is_same_v<DTYPE_Y, AscendC::fp4x2_e2m1_t>) {
        p.attrs.scaleAlg = 2;
        p.attrs.dstTypeMax = 6.5f;
        p.attrs.roundMode = "floor";
    } else {
        p.attrs.outputType = std::is_same_v<DTYPE_Y, AscendC::fp8_e4m3fn_t> ?
            OutputType::FP8_E4M3 : OutputType::FP8_E5M2;
        p.attrs.scaleAlg = 1;
    }
    std::string error;
    assert(ValidateInputs(p, false, error));
    assert(BuildPlan(p, hardware, expected, error));
}
} // namespace

int main()
{
    using namespace optiling::grq;
    // Metadata comes from the real host model/planner, including list-mode
    // normalization and runtime-M tile counts; addresses remain opaque.
    for (int split = 0; split != 4; ++split) {
        for (int listType = 0; listType != 2; ++listType) {
            InputParams p;
            p.attrs.groupType = 0;
            p.attrs.splitItem = split;
            p.attrs.groupListType = listType;
            p.x = (split == 0 || split == 2) ? std::vector<Shape>{{0, 96}, {4, 96}, {9, 96}} :
                                               std::vector<Shape>{{13, 96}};
            p.rotation = {{32, 32}};
            p.groupList = {3};
            p.hasGroupList = true;
            p.yNum = p.scaleNum = split < 2 ? 3 : 1;
            p.hasAlpha = true;
            p.alpha = {1};
            MakePlan(p);
            const int before = calls;
            Invoke();
            assert(calls == before + 1);
            if (split == 1 || split == 3) {
                assert(expected.tileCount == 0 && expected.totalElements > 0);
            }
        }
    }
    InputParams p;
    p.x = {{96}, {2, 1, 2, 1, 1, 3, 128}};
    p.rotation = {{3, 32, 32}, {128, 128}};
    p.yNum = p.scaleNum = 2;
    MakePlan(p);
    Invoke(); // absent optional inputs become nullptr, although dummy addresses were supplied
    assert(calls == 9);

    p.x = {{0, 96}, {7, 0}};
    p.rotation = {{32, 32}, {0, 128, 128}};
    MakePlan(p);
    Invoke();
    assert(calls == 9); // all-empty work never calls the backend

    p = {};
    p.attrs.groupType = 0;
    p.attrs.splitItem = 1;
    p.x = {{13, 96}};
    p.rotation = {{32, 32}};
    p.hasGroupList = true;
    p.groupList = {1};
    p.yNum = p.scaleNum = 1;
    MakePlan(p);
    assert(expected.splitItem == 0 && expected.runtimeM == 1);
    Invoke();
    assert(calls == 10);

    expected.version++;
    bool rejected = false;
    try { Invoke(); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected && calls == 10 && AscendC::activePipes == 0);
    expected.version--;
    testTilingKey = 1;
    Invoke();
    assert(calls == 10);

    Hardware h{24, 48, 40960, 65536, 65536, 262144, 253952};
    uint32_t maxM = 0;
    assert(!ResourceLimit(h, 128, maxM));
    h.l1Size = 73727;
    assert(!ResourceLimit(h, 128, maxM));
    h.l1Size = 73728;
    assert(ResourceLimit(h, 128, maxM) && maxM == 16);
    h.l1Size = 524288;
    for (uint32_t r : {32U, 64U, 128U}) {
        assert(ResourceLimit(h, r, maxM));
        assert(maxM == 16384 / r);
        assert(4ULL * maxM * r + 4ULL * r * r <= h.l1Size);
    }
    std::cout << "outer contract passed\n";
}
