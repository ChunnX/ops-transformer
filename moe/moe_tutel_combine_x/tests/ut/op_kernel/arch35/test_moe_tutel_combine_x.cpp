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
 * \file test_moe_tutel_combine_x.cpp
 * \brief CSV 驱动的 MoeTutelCombineX kernel UT（arch35 / Ascend950，cpudebug 仿真）。
 *
 * 用例矩阵见 ../test_moe_tutel_combine_x.csv（20 例：三分支 tilingKey 0/1/2 ×
 * 三 dtype × 边界集），迁移自插件工作区的内核级白盒/黑盒测试资产。
 *
 * 内核执行方式：
 *   - float32 用例走真实 extern "C" 入口 moe_tutel_combine_x（apt.cpp 经
 *     DTYPE_Y_GRAD=float 编译，UT 框架按文件名强制包含本目录的
 *     moe_tutel_combine_x_tiling.h shim，见 arch35/CMakeLists.txt 注释）；
 *   - float16 / bfloat16 用例走本 TU 定义的模板包装器（直接实例化
 *     MoeTutelCombineXKernel / MoeTutelCombineXEmptyKernel 的 16b 模板参数），
 *     与入口的分支结构逐行同构。
 *
 * 输出校验：kernel 输出（GM 共享内存）回读后与 utils 内的 fp64 参考
 * （README.md 公开公式）逐点比对；输出缓冲先以 0xFF 毒化，未写入即失配。
 */

#ifdef __CCE_KT_TEST__
#include "../../../../op_kernel/moe_tutel_combine_x_apt.cpp"
#include "tikicpulib.h"
#endif
#include "../test_moe_tutel_combine_x_utils.h"

namespace {

using namespace MoeTutelCombineXUt;

#ifdef __CCE_KT_TEST__

// 16b dtype 的内核包装器：与 apt.cpp 入口同构，仅模板参数 T 由包装器绑定。
template <typename T>
__global__ __aicore__ void MoeTutelCombineXUtKernel(GM_ADDR yGrad, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations,
                                                    GM_ADDR xGrad, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    REGISTER_TILING_DEFAULT(MoetutelcombinexTilingData);
    GET_TILING_DATA_WITH_STRUCT(MoetutelcombinexTilingData, t, tiling);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_EMPTY)) {
        MoetutelcombinexTilingData tdMutable = t;
        MoeTutelCombineXEmptyKernel<T> op;
        op.Init(yGrad, gates, indices, locations, xGrad, &tdMutable, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_NORMAL_BATCH)) {
        GM_ADDR ins[kMaxInputSlots] = {yGrad, gates, indices, locations};
        GM_ADDR outs[kMaxOutputSlots] = {xGrad};
        MoeTutelCombineXKernel<T, false, true> op;
        op.Init(ins, outs, &t);
        op.Process();
    } else {
        GM_ADDR ins[kMaxInputSlots] = {yGrad, gates, indices, locations};
        GM_ADDR outs[kMaxOutputSlots] = {xGrad};
        MoeTutelCombineXKernel<T, false, false> op;
        op.Init(ins, outs, &t);
        op.Process();
    }
}

// ---------------------------------------------------------------------------
// 单用例执行：GM 分配（毒化输出）-> 确定性输入 -> tiling 序列化 -> ICPU 仿真
// -> 输出回读 -> 与 fp64 参考逐点比对。
// ---------------------------------------------------------------------------
void RunOneCase(const KernelTestParam &param)
{
    const CaseData data = BuildCase(param);
    const int64_t b = param.batchSize;
    const int64_t s = param.samples;
    const int64_t h = param.hidden;
    const int64_t ec = param.eC;

    const size_t yBytes = std::max<size_t>(1, static_cast<size_t>(b * ec * h)) * DtypeSize(param.dtype);
    const size_t gateBytes = std::max<size_t>(1, static_cast<size_t>(b * s)) * DtypeSize(param.dtype);
    const size_t idxBytes = std::max<size_t>(1, static_cast<size_t>(b * s)) * sizeof(int32_t);
    const size_t outBytes = std::max<size_t>(1, static_cast<size_t>(b * s * h)) * DtypeSize(param.dtype);

    uint8_t *yGrad = static_cast<uint8_t *>(AscendC::GmAlloc(yBytes));
    uint8_t *gates = static_cast<uint8_t *>(AscendC::GmAlloc(gateBytes));
    uint8_t *indices = static_cast<uint8_t *>(AscendC::GmAlloc(idxBytes));
    uint8_t *locations = static_cast<uint8_t *>(AscendC::GmAlloc(idxBytes));
    uint8_t *xGrad = static_cast<uint8_t *>(AscendC::GmAlloc(outBytes));
    uint8_t *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(16ULL * 1024 * 1024)); // 入口形参占位
    uint8_t *tiling = static_cast<uint8_t *>(AscendC::GmAlloc(sizeof(MoetutelcombinexTilingData)));
    ASSERT_NE(yGrad, nullptr);
    ASSERT_NE(gates, nullptr);
    ASSERT_NE(indices, nullptr);
    ASSERT_NE(locations, nullptr);
    ASSERT_NE(xGrad, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    WriteDtypeBuffer(yGrad, data.yGrad, param.dtype);
    WriteDtypeBuffer(gates, data.gates, param.dtype);
    ASSERT_EQ(std::memcpy(indices, data.indices.data(), data.indices.size() * sizeof(int32_t)), indices);
    ASSERT_EQ(std::memcpy(locations, data.locations.data(), data.locations.size() * sizeof(int32_t)), locations);
    ASSERT_EQ(std::memcpy(tiling, &data.tiling, sizeof(data.tiling)), tiling);
    std::memset(xGrad, 0xFF, outBytes); // 毒化：kernel 未写即失配

    ICPU_SET_TILING_KEY(static_cast<uint64_t>(param.tilingKey));
    if (param.dtype == Dtype::FP32) {
        // 真实入口（apt.cpp 以 -DDTYPE_Y_GRAD=float 编译）
        ICPU_RUN_KF(moe_tutel_combine_x, data.launchBlocks, yGrad, gates, indices, locations, xGrad, workspace, tiling);
    } else if (param.dtype == Dtype::FP16) {
        ICPU_RUN_KF((MoeTutelCombineXUtKernel<half>), data.launchBlocks, yGrad, gates, indices, locations, xGrad,
                    workspace, tiling);
    } else {
        ICPU_RUN_KF((MoeTutelCombineXUtKernel<bfloat16_t>), data.launchBlocks, yGrad, gates, indices, locations, xGrad,
                    workspace, tiling);
    }

    const size_t outCount = static_cast<size_t>(std::max<int64_t>(0, b * s * h));
    const std::vector<float> got = ReadDtypeBuffer(xGrad, outCount, param.dtype);
    ASSERT_EQ(got.size(), data.expected.size());
    const float tol = (param.dtype == Dtype::FP32) ? 1e-5F : ((param.dtype == Dtype::FP16) ? 1e-3F : 4e-3F);
    size_t mismatch = 0UL;
    size_t firstIdx = 0UL;
    for (size_t i = 0UL; i < got.size(); ++i) {
        // NaN 安全谓词:IEEE-754 下 NaN 与任何数的有序比较(> / <=)均为 false,
        // 若写成 fabs(diff) > tol,毒化缓冲(0xFF 字节在 fp32/bf16 下即 NaN)会被
        // 当作命中容差,kernel 未写输出也判 PASS。必须用否定形式 !(diff <= tol)
        // 并显式拒绝 NaN(参考值全部有限,输出出现 NaN 即为垃圾)。
        const float diff = std::fabs(got[i] - data.expected[i]);
        if (std::isnan(got[i]) || !(diff <= tol + tol * std::fabs(data.expected[i]))) {
            if (mismatch == 0UL) {
                firstIdx = i;
            }
            ++mismatch;
        }
    }
    EXPECT_EQ(mismatch, 0UL) << param.caseName << ": first mismatch at " << firstIdx << ", got " << got[firstIdx]
                             << ", expected " << data.expected[firstIdx];

    AscendC::GmFree(yGrad);
    AscendC::GmFree(gates);
    AscendC::GmFree(indices);
    AscendC::GmFree(locations);
    AscendC::GmFree(xGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

#endif // __CCE_KT_TEST__

const KernelCsvLoadResult &GetKernelCases()
{
    static const KernelCsvLoadResult result = LoadParams("Ascend950", "MOE_TUTEL_COMBINE_X_KERNEL");
    return result;
}

std::string MakeParamName(const testing::TestParamInfo<KernelTestParam> &info)
{
    return ops::ut::MakeSafeParamName(info.param.caseName);
}

class MoeTutelCombineXKernelTest : public testing::TestWithParam<KernelTestParam> {
protected:
    static void SetUpTestCase()
    {
#ifdef __CCE_KT_TEST__
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
#endif
    }
};

TEST(MoeTutelCombineXKernelCsv, LoadsAscend950Cases)
{
    const auto &result = GetKernelCases();
    for (const auto &error : result.errors) {
        ADD_FAILURE() << error;
    }
    EXPECT_FALSE(result.params.empty());
}

TEST_P(MoeTutelCombineXKernelTest, RunKernel)
{
#ifdef __CCE_KT_TEST__
    RunOneCase(GetParam());
#else
    GTEST_SKIP() << "Kernel CPU simulator is unavailable";
#endif
}

INSTANTIATE_TEST_SUITE_P(MOE_TUTEL_COMBINE_X_ASCEND950, MoeTutelCombineXKernelTest,
                         testing::ValuesIn(GetKernelCases().params), MakeParamName);

} // namespace
