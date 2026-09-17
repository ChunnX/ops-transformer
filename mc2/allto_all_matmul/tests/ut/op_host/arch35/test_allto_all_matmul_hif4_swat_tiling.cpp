/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.openhisilicon.com
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file test_allto_all_matmul_hif4_swat_tiling.cpp
 * \brief HiF4 全量化 Apace SWAT tiling 直调 UT
 *
 * 用例来源: mc2/allto_all_matmul/docs/hif4/HiF4_UT_Case_Design.md §4
 *   - UT-AP-01..06: HiF4 泛化常量与 scale 字节推导(apace_common_utils.h / apace_constant.h)
 *   - UT-SW-01..09: QuantMatmulTilingSwat<DT_HIFLOAT4, DT_HIFLOAT4, DT_FLOAT> tiling 决策
 *
 * 本文件为 host 侧纯 tiling 测试, 不依赖 DT_HIFLOAT4/DT_HIFLOAT4_SCALE GE 枚举
 * (mm::DataType 为仓内 apace_common_utils.h 定义的 tiling 侧枚举), A5 包可编译执行;
 * 断言值基于固定平台 mock 校准(AIC=32, L1=1MB, L0A/B=64KB, L0C=256KB, L2=32MB)。
 * 端到端 tiling/aclnn 用例见 test_allto_all_matmul_hif4_tiling.csv /
 * test_aclnn_allto_all_quant_matmul_hif4.csv(依赖 A6 devkit, 带运行时跳过门控)。
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <map>
#include <string>

#include "apace/tiling/quant_matmul_tiling_swat.h"
#include "apace/utils/apace_constant.h"
#include "platform/platform_infos_def.h"

namespace {
using Hif4SwatTiling =
    QuantMatmulTilingSwat<mm::DataType::DT_HIFLOAT4, mm::DataType::DT_HIFLOAT4, mm::BiasDataType::DT_FLOAT>;

// 平台信息 mock: 键与用法对齐 tests/ut/framework_normal/common/tiling_case_executor.cpp 的 DO_TILING
// (必须 Init() 创建 impl, 且按核类型刷新 core_num_, 否则平台参数全 0 会导致 tiling 内除零)
fe::PlatFormInfos MakeSwatPlatformInfos(uint32_t aiCoreNum)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::map<std::string, std::string> socInfos = {{"ai_core_cnt", std::to_string(aiCoreNum)},
                                                   {"l2_size", "33554432"},
                                                   {"core_type_list", "AICore"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    std::map<std::string, std::string> aicoreSpec = {{"ub_size", "262144"},
                                                     {"l1_size", "1048576"},
                                                     {"l0_a_size", "65536"},
                                                     {"l0_b_size", "65536"},
                                                     {"l0_c_size", "262144"},
                                                     {"bt_size", "0"},
                                                     {"load3d_constraints", "1"}};
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    std::map<std::string, std::string> intrinsics;
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    std::map<std::string, std::string> versions = {{"NpuArch", "9201"}, {"Short_SoC_version", "Ascend960DT"}};
    platformInfo.SetPlatformRes("version", versions);
    return platformInfo;
}

// adjustBasicBlock: true 对齐 host DoHcommMxQuantMMTiling 默认行为(核负载再均衡);
// false 关闭再均衡以获得确定性的基础块断言(隔离 CalcBasicBlock 基线行为)
QuantMatmulTilingData RunHif4SwatTiling(uint32_t aiCoreNum, uint64_t m, uint64_t n, uint64_t k, bool transB,
                                        bool hasBias = false, bool adjustBasicBlock = false)
{
    fe::PlatFormInfos platformInfo = MakeSwatPlatformInfos(aiCoreNum);
    Hif4SwatTiling tilingSwat;
    tilingSwat.SetPlatformInfoPtr(&platformInfo);
    tilingSwat.SetBiasInfo(hasBias);
    tilingSwat.SetAdjustBasicBlockEnable(adjustBasicBlock);
    QuantMatmulTilingData tilingData;
    tilingSwat.GetTilingData(m, n, k, /*transA=*/false, transB, tilingData);
    return tilingData;
}
} // namespace

// ================= UT-AP-01..06: HiF4 泛化常量与 scale 字节推导 =================

// 每 64 个数据元素共享 1 个 hif4_scale; MX 各 dtype 为 32 元素/组
TEST(Hif4ApaceUtilsTest, PerGroupSizeForHif4Is64AndMxIs32)
{
    EXPECT_EQ(GetPerGroupSize<mm::DataType::DT_HIFLOAT4>(), 64UL);
    EXPECT_EQ(GetPerGroupSize<mm::DataType::DT_FLOAT4_E2M1>(), 32UL);
    EXPECT_EQ(GetPerGroupSize<mm::DataType::DT_FLOAT8_E4M3FN>(), 32UL);
    EXPECT_EQ(GetPerGroupSize<mm::DataType::DT_FLOAT8_E5M2>(), 32UL);
}

// HiF4 每 group 携带 4 字节(1 个 4-byte hif4_scale); MX 每 group 1 字节(E8M0)
TEST(Hif4ApaceUtilsTest, ScaleUnitPerGroupForHif4Is4AndMxIs1)
{
    EXPECT_EQ(GetScaleUnitPerGroup<mm::DataType::DT_HIFLOAT4>(), 4UL);
    EXPECT_EQ(GetScaleUnitPerGroup<mm::DataType::DT_FLOAT4_E2M1>(), 1UL);
    EXPECT_EQ(GetScaleUnitPerGroup<mm::DataType::DT_FLOAT8_E4M3FN>(), 1UL);
    EXPECT_EQ(GetScaleUnitPerGroup<mm::DataType::DT_FLOAT8_E5M2>(), 1UL);
}

// HiF4 scale 单元无需成对对齐(1); MX 的 E8M0 需按 2 单元对齐
TEST(Hif4ApaceUtilsTest, AlignSizeForHif4Is1AndMxIs2)
{
    EXPECT_EQ(GetAlignSize<mm::DataType::DT_HIFLOAT4>(), 1UL);
    EXPECT_EQ(GetAlignSize<mm::DataType::DT_FLOAT4_E2M1>(), 2UL);
    EXPECT_EQ(GetAlignSize<mm::DataType::DT_FLOAT8_E4M3FN>(), 2UL);
    EXPECT_EQ(GetAlignSize<mm::DataType::DT_FLOAT8_E5M2>(), 2UL);
}

// 4-bit 打包: 元素数 = 字节数 x 2; 字节数 = ceil(元素数 / 2)
TEST(Hif4ApaceUtilsTest, PackedShapeAndSizeConversionForHif4)
{
    EXPECT_EQ(GetShapeWithDataType<mm::DataType::DT_HIFLOAT4>(32UL), 64UL);
    EXPECT_EQ(GetShapeWithDataType<mm::DataType::DT_HIFLOAT4>(1UL), 2UL);
    EXPECT_EQ(GetSizeWithDataType<mm::DataType::DT_HIFLOAT4>(64UL), 32UL);
    EXPECT_EQ(GetSizeWithDataType<mm::DataType::DT_HIFLOAT4>(65UL), 33UL);
    EXPECT_EQ(GetSizeWithDataType<mm::DataType::DT_HIFLOAT4>(1UL), 1UL);
    // MX FP4 打包行为保持一致
    EXPECT_EQ(GetShapeWithDataType<mm::DataType::DT_FLOAT4_E2M1>(32UL), 64UL);
    EXPECT_EQ(GetSizeWithDataType<mm::DataType::DT_FLOAT4_E2M1>(65UL), 33UL);
}

// 常量与 apace_constant.h 声明一致
TEST(Hif4ApaceUtilsTest, Hif4ConstantsMatchDesignDoc)
{
    EXPECT_EQ(HIF4_GROUP_SIZE, 64UL);
    EXPECT_EQ(HIF4_SCALE_UNIT, 4UL);
    EXPECT_EQ(TILING_HIF4_MULTI_BASE_SIZE, 1UL);
    EXPECT_EQ(MX_GROUP_SIZE, 32UL);
    EXPECT_EQ(TILING_MXFP_MULTI_BASE_SIZE, 2UL);
    EXPECT_EQ(TILING_MXFP_DIVISOR_SIZE, 64UL);
}

// scale 块字节推导(设计文档 §6.4):
//   HiF4: Align(CeilDiv(baseK,64), 1) x 4B -> baseK=128 每行 8B, baseK=256 每行 16B (4B/64元素)
//   MX:   Align(CeilDiv(baseK,32), 2) x 1B -> baseK=128 每行 4B, baseK=256 每行 8B (2B/64元素)
TEST(Hif4ApaceUtilsTest, ScaleBytesPerBaseBlockDerivation)
{
    constexpr uint64_t baseK128 = 128UL;
    constexpr uint64_t baseK256 = 256UL;
    auto hif4BytesPerRow = [](uint64_t baseK) {
        return Align((baseK + GetPerGroupSize<mm::DataType::DT_HIFLOAT4>() - 1UL) /
                         GetPerGroupSize<mm::DataType::DT_HIFLOAT4>(),
                     GetAlignSize<mm::DataType::DT_HIFLOAT4>()) *
               GetScaleUnitPerGroup<mm::DataType::DT_HIFLOAT4>();
    };
    auto mxFp4BytesPerRow = [](uint64_t baseK) {
        return Align((baseK + GetPerGroupSize<mm::DataType::DT_FLOAT4_E2M1>() - 1UL) /
                         GetPerGroupSize<mm::DataType::DT_FLOAT4_E2M1>(),
                     GetAlignSize<mm::DataType::DT_FLOAT4_E2M1>()) *
               GetScaleUnitPerGroup<mm::DataType::DT_FLOAT4_E2M1>();
    };
    EXPECT_EQ(hif4BytesPerRow(baseK128), 8UL);
    EXPECT_EQ(hif4BytesPerRow(baseK256), 16UL);
    EXPECT_EQ(mxFp4BytesPerRow(baseK128), 4UL);
    EXPECT_EQ(mxFp4BytesPerRow(baseK256), 8UL);
}

// ================= UT-SW-01..09: SWAT tiling 决策(断言值基于平台 mock 校准) =================

// UT-SW-01: 基础块决策(关闭再均衡): 4-bit 档 baseK=min(Ka,256) 对齐 64, DN 时 baseN 对齐 16
TEST(Hif4SwatTilingTest, BasicShapeAndBaseBlock)
{
    auto tilingData = RunHif4SwatTiling(32, 128, 256, 128, true);
    EXPECT_EQ(tilingData.m, 128U);
    EXPECT_EQ(tilingData.n, 256U);
    EXPECT_EQ(tilingData.k, 128U);
    EXPECT_EQ(tilingData.baseM, 128U); // min(m,256) 对齐 16
    EXPECT_EQ(tilingData.baseN, 256U); // transB: min(n,256) 对齐 16
    EXPECT_EQ(tilingData.baseK, 128U); // min(Ka,256) 对齐 64
    EXPECT_EQ(tilingData.stepK, 1U);
    EXPECT_EQ(tilingData.scaleKL1, 128U); // min(sfA*stepKa*baseK, sfB*stepKb*baseK)
    EXPECT_EQ(tilingData.usedCoreNum, 30U); // 尾块 5x6 分裂
    EXPECT_EQ(tilingData.dbL0c, 2U); // 128*256*4*2=256KB <= L0C 256KB
}

// UT-SW-02: Ka=512/1024 时 baseK 按 4-bit 档 clamp 到 256; stepK 随 K 增长且不超过 4
TEST(Hif4SwatTilingTest, BaseKClampedTo256AndStepKGrows)
{
    auto k512 = RunHif4SwatTiling(32, 128, 256, 512, true);
    EXPECT_EQ(k512.baseK, 256U);
    EXPECT_EQ(k512.stepK, 2U);
    EXPECT_EQ(k512.scaleKL1, 512U);

    auto k1024 = RunHif4SwatTiling(32, 128, 256, 1024, true);
    EXPECT_EQ(k1024.baseK, 256U);
    EXPECT_EQ(k1024.stepK, 4U); // STEPK_THERSHOLD=4 上限
    EXPECT_EQ(k1024.scaleKL1, 1024U);
}

// UT-SW-03: ND(transB=false) 时 baseN 按 64 对齐, DN(transB=true) 时按 16 对齐
TEST(Hif4SwatTilingTest, NdBaseNAlignedTo64AndDnTo16)
{
    auto ndTiling = RunHif4SwatTiling(32, 128, 80, 128, false);
    EXPECT_EQ(ndTiling.baseN, 128U); // Align(80, 64) = 128
    auto dnTiling = RunHif4SwatTiling(32, 128, 80, 128, true);
    EXPECT_EQ(dnTiling.baseN, 80U); // Align(80, 16) = 80
}

// UT-SW-04: M 尾块 (m=272 = 256+16) 触发尾块分裂, dbL0c 降为 1(256x256 块超 L0C/2)
TEST(Hif4SwatTilingTest, MTailSplit)
{
    auto tilingData = RunHif4SwatTiling(32, 272, 256, 128, true);
    EXPECT_EQ(tilingData.baseM, 256U);
    EXPECT_EQ(tilingData.mTailTile, 4U);
    EXPECT_EQ(tilingData.usedCoreNum, 32U);
    EXPECT_EQ(tilingData.dbL0c, 1U);
}

// UT-SW-05: N 尾块 (n=272 = 256+16, DN)
TEST(Hif4SwatTilingTest, NTailSplit)
{
    auto tilingData = RunHif4SwatTiling(32, 256, 272, 128, true);
    EXPECT_EQ(tilingData.baseN, 256U);
    EXPECT_EQ(tilingData.nTailTile, 4U);
    EXPECT_EQ(tilingData.usedCoreNum, 32U);
}

// UT-SW-07: hasBias 使能时 tiling 仍有效(bias L1 预算扣除后基础块不变)
TEST(Hif4SwatTilingTest, BiasEnabledTilingStillValid)
{
    auto tilingData = RunHif4SwatTiling(32, 128, 256, 128, true, true);
    EXPECT_EQ(tilingData.m, 128U);
    EXPECT_EQ(tilingData.n, 256U);
    EXPECT_EQ(tilingData.k, 128U);
    EXPECT_EQ(tilingData.baseK, 128U);
    EXPECT_EQ(tilingData.usedCoreNum, 30U);
}

// UT-SW-08: dbL0c 档位切换(256x256x4Bx2=512KB > L0C 256KB -> 1; 128x256 -> 2)
TEST(Hif4SwatTilingTest, DbL0cToggleByBaseShape)
{
    auto bigC = RunHif4SwatTiling(32, 256, 256, 128, true);
    EXPECT_EQ(bigC.dbL0c, 1U);
    auto smallC = RunHif4SwatTiling(32, 128, 256, 128, true);
    EXPECT_EQ(smallC.dbL0c, 2U);
}

// UT-SW-08b: 核负载再均衡(host 默认行为): 单块任务在 32 核下被再切分,
// baseK 按 4-bit L2 行对齐扩展到 256 元素(GetShape<HIF4>(L2_ALIGN_SIZE=128)=256)
TEST(Hif4SwatTilingTest, AdjustBasicBlockForCoreBalance)
{
    auto tilingData = RunHif4SwatTiling(32, 272, 256, 128, true, false, true);
    EXPECT_EQ(tilingData.baseM, 48U);
    EXPECT_EQ(tilingData.baseN, 64U);
    EXPECT_EQ(tilingData.baseK, 256U);
    EXPECT_EQ(tilingData.usedCoreNum, 32U);
    EXPECT_EQ(tilingData.dbL0c, 2U);
}

// UT-SW-09: 同一 tiling 对象连续两次 GetTilingData, 状态清零不串扰
TEST(Hif4SwatTilingTest, TilingObjectReuseAcrossShapes)
{
    fe::PlatFormInfos platformInfo = MakeSwatPlatformInfos(32);
    Hif4SwatTiling tilingSwat;
    tilingSwat.SetPlatformInfoPtr(&platformInfo);
    tilingSwat.SetAdjustBasicBlockEnable(false); // 与 RunHif4SwatTiling 基线一致, 保证确定性断言
    QuantMatmulTilingData first;
    tilingSwat.GetTilingData(128, 256, 128, false, true, first);
    EXPECT_EQ(first.baseK, 128U);

    QuantMatmulTilingData second;
    tilingSwat.GetTilingData(272, 80, 512, false, false, second);
    QuantMatmulTilingData secondFresh = RunHif4SwatTiling(32, 272, 80, 512, false);
    EXPECT_EQ(second.m, secondFresh.m);
    EXPECT_EQ(second.n, secondFresh.n);
    EXPECT_EQ(second.k, secondFresh.k);
    EXPECT_EQ(second.baseM, secondFresh.baseM);
    EXPECT_EQ(second.baseN, secondFresh.baseN);
    EXPECT_EQ(second.baseK, secondFresh.baseK);
    EXPECT_EQ(second.stepK, secondFresh.stepK);
    EXPECT_EQ(second.scaleKL1, secondFresh.scaleKL1);
    EXPECT_EQ(second.usedCoreNum, secondFresh.usedCoreNum);
}
