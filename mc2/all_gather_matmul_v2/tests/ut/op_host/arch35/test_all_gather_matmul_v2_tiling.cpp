/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../all_gather_matmul_v2_host_ut_param.h"
#include "mc2_tiling_case_executor.h"
// Ka==K 断言（T5-②）解析 tiling 原始输出所需（与 tiling 实现同源）
#include "../../../../op_kernel/all_gather_matmul_tiling_data.h"
#include <cstddef>
// APACE 分支 tiling 结构断言（2.1 v2 §5.4.3 HA1~HA8）解析原始输出所需——与 apace
// tiling 实现同源（mc2/common/op_kernel 经 OP_TILING_INCLUDE 解析）
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_tiling_data.h"

namespace all_gather_matmul_v2_ut {

class AllGatherMatmulV2Arch35TilingTest : public testing::TestWithParam<AllGatherMatmulV2TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulV2 Arch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulV2 Arch35TilingTest TearDown" << std::endl;
    }
};

struct AllGatherMatmulV2CompileInfo {};

// ---------------------------------------------------------------------------
// 2.1 v2 §5.4.3 HA1~HA8 / §7.2 T-A1/T-A2/T-A4：APACE 分支（USING_APACE_IMPL=1）
// 结构断言集（ut_apace_ 前缀触发）。
// tiling key 布局（H1 修正冻结口径）：既有 6 参占 bit0~8（位宽 1+1+2+2+2+1=9），
// USING_APACE_IMPL 为 DECL 末尾追加的第 7 参、占最高位 bit9（CANN
// FastEncodeTilingKeyDirect 按声明顺序正序编码、先声明占低位——
// utils/tiling/template_argument.h L291-322 实证）。
// ---------------------------------------------------------------------------
constexpr uint64_t UT_APACE_IMPL_KEY_BIT = 1ULL << 9; // 512：USING_APACE_IMPL 最高位
// T-A1 使能开关注入合同（对齐 3.1 W2 定稿实现：all_gather_matmul_tiling_v2.cpp
// APACE_IMPL_ENABLE_ENV）——env 名 ASCEND_MC2_AGMV2_APACE_IMPL，值严格 "1"=on；
// 未设/其他值=off（B2 冻结：初期默认 v1 路径）
constexpr char kAscendMc2UsingApaceImpl[] = "ASCEND_MC2_AGMV2_APACE_IMPL";

// HA2（静态断言）：DFX 头部约定——dumpInfo 必须位于 apace 结构 offset 0
static_assert(offsetof(Apace::hcommAllGatherMatmulTilingData, dumpInfo) == 0,
              "HA2: dumpInfo must reside at offset 0 of hcommAllGatherMatmulTilingData");

// 非量化链路 v1 tiling key 换算（all_gather_matmul_tiling_base.cpp GetTilingKey 实源）：
// GET_TPL_TILING_KEY(inputIsBf16Fp16_=1, isBTrans, outputType=0, TPL_DEFAULT_MODE=0,
// SCALE_TYPE_NOT_IS_MX=0, commMode)——bit0=INPUT_IS_BF16FP16(恒1)；bit1=TRANS_B；
// bit2~7 恒 0（OUTPUTDTYPE/QUANTMMMODE/SCALETYPE 均索引 0）；bit8=HCCL_COMM_MODE
// （ai_cpu=1 / ccu=0）。AT 系列（AT-150 off 侧）同式。
uint64_t ComputeV1TilingKeyNonQuant(bool isTransB, const std::string &commMode)
{
    return 1ULL | (isTransB ? (1ULL << 1) : 0ULL) | ((commMode == "ai_cpu") ? (1ULL << 8) : 0ULL);
}

// HA6：mc2InitTiling/mc2CcTiling 区域非空判定（GetTiling 序列化产物必含
// groupName/algConfig 等非零内容）。引擎参数与 comm_mode 的匹配属 §7.3 通信层
// 走查项（S4 拆分后 UT 侧可断言物 = 非空）。
bool UtHasAnyNonZeroByte(const uint8_t *begin, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        if (begin[i] != 0) {
            return true;
        }
    }
    return false;
}

gert::TilingContextPara BuildTilingContextParam(const AllGatherMatmulV2TilingUtParam &param,
                                                AllGatherMatmulV2CompileInfo *compileInfo)
{
    return gert::TilingContextPara(
        "AllGatherMatmulV2",
        {
            param.x1,
            param.x2,
            param.bias,
            param.x1_scale,
            param.x2_scale,
            param.quant_scale,
        },
        {
            param.y,
            param.gather_out,
            param.amax_out,
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
         {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.gather_index)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rank_size)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.block_size)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
         {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_gather_out)},
         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_amax_out)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.comm_mode)}},
        param.inputInstance, param.outputInstance, compileInfo, param.soc, param.coreNum, param.ubsize);
}

TEST_P(AllGatherMatmulV2Arch35TilingTest, param)
{
    auto param = GetParam();
    constexpr char kAscendMc2DebugMode[] = "ASCEND_MC2_DEBUG_MODE";
    bool debugModeEnvSet = false;
    constexpr const char kUtMc2Debug4Prefix[] = "ut_mc2_debug4_";
    if (param.case_name.compare(0, std::size(kUtMc2Debug4Prefix) - 1, kUtMc2Debug4Prefix) == 0) {
        (void)setenv(kAscendMc2DebugMode, "4", 1);
        debugModeEnvSet = true;
    }
    // MODE=1 = MC2_DEBUG_ONLY_CUBE（只算不通信，agm_v2_common.h L42）。
    // T5-⑤：ut_mc2_debug1_* 仅覆盖 tiling 输出；kernel 分支覆盖由
    // 仿真通道（2.1 §8 A9 回退分支 2，MODE=1 单卡仿真）承载。
    constexpr const char kUtMc2Debug1Prefix[] = "ut_mc2_debug1_";
    if (param.case_name.compare(0, std::size(kUtMc2Debug1Prefix) - 1, kUtMc2Debug1Prefix) == 0) {
        (void)setenv(kAscendMc2DebugMode, "1", 1);
        debugModeEnvSet = true;
    }
    AllGatherMatmulV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara = BuildTilingContextParam(param, &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rank_size}};
    // Ka==K 不变量硬门禁（2.1 §7 T5-② / 2.2 F4/R3）：gather 路径
    // MC2MatmulAswKernelDerive 无 K 轴循环、通信量按 tileM×Ka 计——
    // tile/tail tiling 触发 mmv3 切 K（Ka != rankK）即静默数据错误，违例 FAIL。
    constexpr const char kUtKakInvPrefix[] = "ut_kak_inv_";
    if (param.expectResult == ge::GRAPH_SUCCESS &&
        param.case_name.compare(0, std::size(kUtKakInvPrefix) - 1, kUtKakInvPrefix) == 0) {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        ASSERT_GE(tilingInfo.tilingDataSize, sizeof(Mc2Tiling::AllGatherMatmulTilingDataV2));
        const auto *agmTiling =
            reinterpret_cast<const Mc2Tiling::AllGatherMatmulTilingDataV2 *>(tilingInfo.tilingData.get());
        EXPECT_EQ(agmTiling->mc2MmV3TileTilingData.tCubeTiling.Ka,
                  static_cast<int32_t>(agmTiling->param.rankK))
            << "ut_kak_inv: tile tCubeTiling.Ka != rankK (split-K triggered on gather path)";
        if (agmTiling->param.tailM > 0) {
            EXPECT_EQ(agmTiling->mc2MmV3TailTilingData.tCubeTiling.Ka,
                      static_cast<int32_t>(agmTiling->param.rankK))
                << "ut_kak_inv: tail tCubeTiling.Ka != rankK (split-K triggered on gather path)";
        }
        if (debugModeEnvSet) {
            (void)unsetenv(kAscendMc2DebugMode);
        }
        return;
    }
    // 白盒结构断言（ut_struct_ 前缀，3.3）：读 tiling 原始输出，校验驱动
    // kernel 分支的决策变量与 csv 参数自洽，并打印观测值作为分支命中证据。
    constexpr const char kUtStructPrefix[] = "ut_struct_";
    if (param.expectResult == ge::GRAPH_SUCCESS &&
        param.case_name.compare(0, std::size(kUtStructPrefix) - 1, kUtStructPrefix) == 0) {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        ASSERT_GE(tilingInfo.tilingDataSize, sizeof(Mc2Tiling::AllGatherMatmulTilingDataV2));
        const auto *agmStruct =
            reinterpret_cast<const Mc2Tiling::AllGatherMatmulTilingDataV2 *>(tilingInfo.tilingData.get());
        const auto &rcs = agmStruct->param;
        const int64_t mVal = param.x1.shape_.GetStorageShape().GetDim(0);
        const int64_t kVal = param.x1.shape_.GetStorageShape().GetDim(1);
        const int64_t x2Dim0 = param.x2.shape_.GetStorageShape().GetDim(0);
        const int64_t x2Dim1 = param.x2.shape_.GetStorageShape().GetDim(1);
        // n 值判别与 SetTilingArgsDim 同口径：x2 非 transB 为 (k,n)、transB 为 (n,k)
        const int64_t nVal = (kVal == x2Dim0) ? x2Dim1 : x2Dim0;
        const bool biasPresent = (param.bias.dtype_ != ge::DT_UNDEFINED);
        const bool biasBf16 = biasPresent && (param.bias.dtype_ == ge::DT_BF16);
        uint64_t gatherOutNumel = 0;
        if (param.gather_out.dtype_ != ge::DT_UNDEFINED) {
            const auto &gatherShape = param.gather_out.shape_.GetStorageShape();
            gatherOutNumel = 1;
            for (size_t d = 0; d < gatherShape.GetDimNum(); d++) {
                gatherOutNumel *= static_cast<uint64_t>(gatherShape.GetDim(d));
            }
        }
        // 1) rank 维度传递（含 transB 的 (n,k) 判别支）
        EXPECT_EQ(rcs.rankM, static_cast<uint32_t>(mVal));
        EXPECT_EQ(rcs.rankK, static_cast<uint32_t>(kVal));
        EXPECT_EQ(rcs.rankN, static_cast<uint32_t>(nVal));
        // 2) gather 两态：is_gather_out=1 且 gatherOut 非空 → 通信直写输出
        //    （gatherLen==0）；否则 gather 落 workspace（gatherLen>0）
        if (param.is_gather_out && gatherOutNumel > 0) {
            EXPECT_EQ(rcs.gatherLen, 0u) << "is_gather_out=true: 通信直写输出, gatherLen 应为 0";
            EXPECT_EQ(rcs.storageGather, 1u);
        } else {
            EXPECT_GT(rcs.gatherLen, 0u) << "is_gather_out=false: gather 落 workspace, gatherLen 应 > 0";
            EXPECT_EQ(rcs.storageGather, 0u);
        }
        // 3) biasLen 两态：isBias && bType==BF16 才分配 fp32 cast workspace
        if (biasBf16) {
            EXPECT_GT(rcs.biasLen, 0u) << "bf16 x2 + bias: biasLen 应 > 0";
        } else {
            EXPECT_EQ(rcs.biasLen, 0u) << "fp16 x2 / 无 bias: biasLen 应为 0";
        }
        // 观测值打印（分支命中证据）
        std::cout << "  [ut_struct] " << param.case_name << " rankM/N/K=" << rcs.rankM << "/" << rcs.rankN << "/"
                  << rcs.rankK << " gatherLen=" << rcs.gatherLen << " storageGather=" << rcs.storageGather
                  << " biasLen=" << rcs.biasLen << " tileCnt=" << rcs.tileCnt << " tailCnt=" << rcs.tailCnt
                  << " tailM=" << rcs.tailM << std::endl;
        if (nVal > 0) {
            // 4) 切分一致性（n>0 时 tCubeTiling 有效；n==0 跳过 mmv3 tiling）
            const uint32_t structTileM = static_cast<uint32_t>(agmStruct->mc2MmV3TileTilingData.tCubeTiling.M) /
                (rcs.rankDim - 1);
            EXPECT_EQ(static_cast<uint64_t>(rcs.tileCnt) * structTileM +
                static_cast<uint64_t>(rcs.tailCnt) * rcs.tailM, static_cast<uint64_t>(mVal))
                << "切分一致性: tileCnt*tileM + tailCnt*tailM == m";
            EXPECT_LE(rcs.tailCnt, 1u);
            EXPECT_GE(rcs.tileCnt, 1u);
            if (param.comm_mode == "ccu") {
                // 5) CCU 后置条件：单轮通信量 ≤ 256MB（fp16/bf16 dtypeSize=2）
                EXPECT_LE(static_cast<uint64_t>(structTileM) * static_cast<uint64_t>(kVal) * 2ULL *
                    static_cast<uint64_t>(rcs.rankDim), 268435456ULL) << "CCU 单轮通信量 <= 256MB";
            }
            EXPECT_LE(rcs.tileCnt + rcs.tailCnt, 62u) << "总通信次数 <= HCCL_MAX_TOTAL_TILES";
        }
        if (debugModeEnvSet) {
            (void)unsetenv(kAscendMc2DebugMode);
        }
        return;
    }
    // 白盒补充（3.3 v2，ut_wbkey_ 前缀）：使能开关 env 取值语义 + 范围隔离验证。
    // using_apace_impl 列在本前缀下作 env 取值选择：1=setenv"1"（on）、2=setenv"0"
    //（set 但非 "1"——验证 N4 L83 strncmp(env,"1",2)==0 的严格相等语义）、0/unset=默认 off。
    // 覆盖：APACE key 515（ccu×transB，3.2 缺口）/ 量化 dtype+env on 隔离（key 264 不变，
    // N4 切换条件 INPUT_IS_BF16FP16 因子 false 侧）/ env="0" 严格 off（key 257）。
    constexpr const char kUtWbKeyPrefix[] = "ut_wbkey_";
    if (param.expectResult == ge::GRAPH_SUCCESS &&
        param.case_name.compare(0, std::size(kUtWbKeyPrefix) - 1, kUtWbKeyPrefix) == 0) {
        if (param.using_apace_impl == 1) {
            (void)setenv(kAscendMc2UsingApaceImpl, "1", 1);
        } else if (param.using_apace_impl == 2) {
            (void)setenv(kAscendMc2UsingApaceImpl, "0", 1);
        } else {
            (void)unsetenv(kAscendMc2UsingApaceImpl);
        }
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        TilingInfo wbKeyTilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, wbKeyTilingInfo)) << "ut_wbkey: tiling 应 SUCCESS";
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        EXPECT_EQ(static_cast<uint64_t>(wbKeyTilingInfo.tilingKey), param.expectTilingKey)
            << "ut_wbkey: key 与 csv 期望一致（envSel=" << param.using_apace_impl
            << "：1=on / 2=env\"0\"严格off / 0=unset默认off）";
        std::cout << "  [ut_wbkey] " << param.case_name << " envSel=" << param.using_apace_impl
                  << " key=" << wbKeyTilingInfo.tilingKey << "（期望 " << param.expectTilingKey << "）"
                  << std::endl;
        (void)unsetenv(kAscendMc2UsingApaceImpl);
        if (debugModeEnvSet) {
            (void)unsetenv(kAscendMc2DebugMode);
        }
        return;
    }
    // APACE 分支结构断言（ut_apace_ 前缀，2.1 v2 §5.4.3 HA1~HA8 / §7.2 T-A1/T-A2/T-A4）：
    // 按 using_apace_impl 两侧分别断言——on 侧解析 Apace::hcommAllGatherMatmulTilingData
    // 六字段（HA1/HA3~HA7）+ key 最高位（T-A4）；off 侧（AT-150，HA8）解析 v1
    // AllGatherMatmulTilingDataV2 + key 最高位=0。on 侧执行依赖 3.1 N4/N5 落地——
    // 未落地时 tiling 仍走 v1 分支（env 未被消费），在 T-A4 key 断言处以
    // 实得=旧值的形态失败，即「执行待 3.1」的判据（资产先行定稿）。
    constexpr const char kUtApacePrefix[] = "ut_apace_";
    if (param.expectResult == ge::GRAPH_SUCCESS &&
        param.case_name.compare(0, std::size(kUtApacePrefix) - 1, kUtApacePrefix) == 0) {
        if (param.using_apace_impl == 1) {
            (void)setenv(kAscendMc2UsingApaceImpl, "1", 1);
        } else {
            // off 侧 = 默认 v1 路径（B2 初期默认；unset 使判定不受环境残留影响）
            (void)unsetenv(kAscendMc2UsingApaceImpl);
        }
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        TilingInfo apaceTilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, apaceTilingInfo)) << "ut_apace: tiling 应 SUCCESS";
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        // m/k/n 推导与 ut_struct_ 同口径：x2 非 transB 为 (k,n)、transB 为 (n,k)
        const int64_t apaceM = param.x1.shape_.GetStorageShape().GetDim(0);
        const int64_t apaceK = param.x1.shape_.GetStorageShape().GetDim(1);
        const int64_t apaceX2Dim0 = param.x2.shape_.GetStorageShape().GetDim(0);
        const int64_t apaceX2Dim1 = param.x2.shape_.GetStorageShape().GetDim(1);
        const int64_t apaceN = (apaceK == apaceX2Dim0) ? apaceX2Dim1 : apaceX2Dim0;
        uint64_t apaceGatherNumel = 0;
        if (param.gather_out.dtype_ != ge::DT_UNDEFINED) {
            const auto &apaceGatherShape = param.gather_out.shape_.GetStorageShape();
            apaceGatherNumel = 1;
            for (size_t d = 0; d < apaceGatherShape.GetDimNum(); d++) {
                apaceGatherNumel *= static_cast<uint64_t>(apaceGatherShape.GetDim(d));
            }
        }
        const uint64_t utV1Key = ComputeV1TilingKeyNonQuant(param.is_trans_b, param.comm_mode);
        const uint64_t utApaceKey = utV1Key | UT_APACE_IMPL_KEY_BIT;
        if (param.using_apace_impl == 1) {
            // T-A4/HA7：APACE 分支 key = v1 key 置最高位（末尾追加、既有低位 6 维不变）
            ASSERT_EQ(static_cast<uint64_t>(apaceTilingInfo.tilingKey), utApaceKey)
                << "T-A4/HA7: apace key = v1|" << UT_APACE_IMPL_KEY_BIT << "（若实得 " << apaceTilingInfo.tilingKey
                << " 为 v1 旧值，即 3.1 N4 使能开关/tiling 分支未生效——执行待 3.1）";
            EXPECT_EQ(static_cast<uint64_t>(apaceTilingInfo.tilingKey) & ~UT_APACE_IMPL_KEY_BIT, utV1Key)
                << "HA7: 既有低位 6 维不变";
            EXPECT_EQ(param.expectTilingKey, utApaceKey) << "csv expectTilingKey 与换算式一致";
            // HA1：SetDataSize 取 max（两分支 tiling 结构取大者）——≥ 两结构 sizeof
            // （V2-B6 方向核对兜底：v1 ≥ apace 方向性 + UT 双断言）
            EXPECT_GE(apaceTilingInfo.tilingDataSize, sizeof(Apace::hcommAllGatherMatmulTilingData))
                << "HA1: tilingDataSize >= sizeof(apace 结构)";
            EXPECT_GE(apaceTilingInfo.tilingDataSize, sizeof(Mc2Tiling::AllGatherMatmulTilingDataV2))
                << "HA1: tilingDataSize >= sizeof(v1 结构)（max 口径）";
            const auto *apaceTd = reinterpret_cast<const Apace::hcommAllGatherMatmulTilingData *>(
                apaceTilingInfo.tilingData.get());
            // HA3：commTile 语义（切分轴=M_per_rank、非切分轴=K；fit-balance+256 对齐封顶）
            EXPECT_EQ(apaceTd->commTile.nonSplitAxisSize, static_cast<uint64_t>(apaceK))
                << "HA3: nonSplitAxisSize == k";
            EXPECT_EQ(apaceTd->commTile.splitAxisTileSize * apaceTd->commTile.splitAxisTileCnt +
                          apaceTd->commTile.splitAxisTailSize * apaceTd->commTile.splitAxisTailCnt,
                      static_cast<uint64_t>(apaceM))
                << "HA3: M 覆盖完整性 tileSize*tileCnt + tailSize*tailCnt == m";
            EXPECT_LE(apaceTd->commTile.splitAxisTailCnt, 1ULL) << "HA3: 尾块 tailCnt <= 1";
            EXPECT_GE(apaceTd->commTile.splitAxisTileCnt, 1ULL) << "HA3: 头块 tileCnt >= 1";
            EXPECT_TRUE(apaceTd->commTile.splitAxisTileSize % 256 == 0 ||
                        (apaceTd->commTile.splitAxisTileCnt == 1 &&
                         apaceTd->commTile.splitAxisTileSize == static_cast<uint64_t>(apaceM)))
                << "HA3: splitAxisTileSize 256 对齐（AlignCutResForKernel 同款）或收缩单块";
            // HA4：mmTile 语义（v1 mmv3 registry 输出 → QuantMatmulTilingData 字段映射）
            EXPECT_EQ(apaceTd->mmTile.m, static_cast<uint32_t>(apaceM)) << "HA4: problemShape m";
            EXPECT_EQ(apaceTd->mmTile.n, static_cast<uint32_t>(apaceN)) << "HA4: problemShape n";
            EXPECT_EQ(apaceTd->mmTile.k, static_cast<uint32_t>(apaceK))
                << "HA4: problemShape k == k（K extent 映射一致性门禁——Ka==K 断言的 APACE "
                << "对应物：mmTile.k 为 kernel 消费的 K 全长字段，2.2 §3.4 映射表）";
            EXPECT_GT(apaceTd->mmTile.baseM, 0U) << "HA4: baseM 非零";
            EXPECT_GT(apaceTd->mmTile.baseN, 0U) << "HA4: baseN 非零";
            // baseK 合理性（2.2 §5.2「baseK 合理性断言」+ §3.4 映射表：baseK ← tCube.baseK
            // 为 L0 级基础块、mmv3 runInfo_.baseK = min(k 对齐, kValueMax)，k>256 时
            // 物理上不等于 k——2.1 §5.4.3 HA4 字面「baseK==k」与本映射语义分叉，
            // K extent 门禁由上方 mmTile.k==k 承接，字面断言不落地并回传上报（上游裁定）
            EXPECT_GT(apaceTd->mmTile.baseK, 0U) << "HA4: baseK 非零";
            EXPECT_LE(apaceTd->mmTile.baseK, static_cast<uint32_t>(apaceK)) << "HA4: baseK <= k（L0 块不超 K 全长）";
            // HA4-S3 配平不等式（W2 核定值回填形态）：多通信块（tileCnt≥2）强制
            // baseM ≤ splitAxisTileSize（mm 单块不跨通信 tile 边界——真实约束）；
            // 单通信块收缩场景（m < 256 对齐块）无内部 tile 边界、约束空洞，退化为
            // mmv3 夹取域 baseM ≤ CeilAlign(m,16)（matmul_v3_asw_tiling.cpp L290/L306
            // min(CeilAlign(m,16),·) 实源；AT-120 m=1 实测 baseM=16）——跨块正确性由
            // Blaze scheduler 尾块拆分 + WaitTile 递进消费承接（2.2 §6 正确性不变量）。
            if (apaceTd->commTile.splitAxisTileCnt >= 2) {
                EXPECT_LE(apaceTd->mmTile.baseM, apaceTd->commTile.splitAxisTileSize)
                    << "HA4-S3: baseM <= splitAxisTileSize（多通信块：mm 单块不跨 tile 边界）";
            } else {
                const uint64_t mmv3MClamp =
                    (static_cast<uint64_t>(apaceM) + 15ULL) / 16ULL * 16ULL; // CeilAlign(m,16)
                EXPECT_LE(apaceTd->mmTile.baseM, mmv3MClamp)
                    << "HA4-S3(单块): baseM <= CeilAlign(m,16)（mmv3 夹取域，边界由尾块拆分承接）";
            }
            // HA4 dbL0c 域（W2 核定值回填形态）：mmv3 dbL0C = (baseM*baseN*4*2 <= l0CSize) ? 2 : 1
            // （matmul_v3_asw_tiling.cpp L348）——域 {1,2}，2=L0C pingpong；N2 内核按
            // `dbL0c > 1` 布尔消费（all_gather_bmm_kernel.h L302）；2.2 §3.4 注记「>1 使能」。
            // （2.1 v2 §5.4.3 字面「∈ {0,1}」为方案侧笔误，回传⑥上报修订）
            EXPECT_TRUE(apaceTd->mmTile.dbL0c == 1 || apaceTd->mmTile.dbL0c == 2)
                << "HA4: dbL0c ∈ {1,2}（mmv3 L0C 分区因子，2=pingpong）";
            // 3.3 v2 白盒标记断言（用例名内嵌 marker，决策变量级分支驱动证据）：
            if (param.case_name.find("_mtile_") != std::string::npos) {
                EXPECT_GE(apaceTd->commTile.splitAxisTileCnt, 2ULL)
                    << "WB-mtile: 多通信块 tileCnt>=2（驱动 MAIN 多轮 UpdateMainRoundAddrs，"
                    << "N2 L386-387；3.2 AT 矩阵实测 tileCnt 恒 1——本标记行补齐）";
            }
            if (param.case_name.find("_mtileexact_") != std::string::npos) {
                EXPECT_GE(apaceTd->commTile.splitAxisTileCnt, 2ULL);
                EXPECT_EQ(apaceTd->commTile.splitAxisTailCnt, 0ULL)
                    << "WB-mtileexact: 多块整除无尾（N4 L150-152 ternary false 侧：m%tile==0）";
            }
            if (param.case_name.find("_r64tail_") != std::string::npos) {
                EXPECT_EQ(static_cast<uint64_t>(param.rank_size), 64ULL);
                EXPECT_EQ(apaceTd->commTile.splitAxisTailCnt, 1ULL)
                    << "WB-r64tail: R=64 且含尾块（TAIL 区 64 fragment == MaxFragments=64 数组边界，"
                    << "N2 L90/L200-209）";
            }
            if (param.case_name.find("_tailunaln_") != std::string::npos) {
                EXPECT_GT(apaceTd->commTile.splitAxisTailSize, 0ULL) << "WB-tailunaln: 尾块存在";
                std::cout << "  [ut_apace-wb] tailM=" << apaceTd->commTile.splitAxisTailSize
                          << " tailM%16=" << (apaceTd->commTile.splitAxisTailSize % 16ULL)
                          << "（非 0 即驱动 N1 L155 paddedTailM CeilAlign 进位支）" << std::endl;
            }
            // workspace 覆盖断言（N4 GetWorkspaceSize apace 覆写支 L435-444：ws=libApi+gatherLen；
            // libApi 为平台常量，跨行常量性由观测记录核对，行内断言 ws>=gatherLen）
            if (!apaceTilingInfo.workspaceSizes.empty() && apaceTd->gatherLen > 0) {
                EXPECT_GE(apaceTilingInfo.workspaceSizes[0], static_cast<int64_t>(apaceTd->gatherLen))
                    << "WB-ws: apace workspace >= gatherLen（gather 落 workspace 时覆盖 gather 区）";
            }
            // HA5：gatherLen 两态（is_gather_out=true 且 gatherOut 非空 → 通信直写；否则 workspace）
            if (param.is_gather_out && apaceGatherNumel > 0) {
                EXPECT_EQ(apaceTd->gatherLen, 0ULL) << "HA5: gatherOut 直写, gatherLen == 0";
            } else {
                EXPECT_GT(apaceTd->gatherLen, 0ULL) << "HA5: gather 落 workspace, gatherLen > 0";
            }
            // HA6：mc2InitTiling/mc2CcTiling 非空（GetTiling 序列化产物）
            EXPECT_TRUE(UtHasAnyNonZeroByte(reinterpret_cast<const uint8_t *>(&apaceTd->mc2InitTiling),
                                            sizeof(apaceTd->mc2InitTiling)))
                << "HA6: mc2InitTiling 非空";
            EXPECT_TRUE(UtHasAnyNonZeroByte(reinterpret_cast<const uint8_t *>(&apaceTd->mc2CcTiling),
                                            sizeof(apaceTd->mc2CcTiling)))
                << "HA6: mc2CcTiling 非空";
            // 观测值打印（分支命中证据；stepK/nBufferNum 为 2.2 §3.4 W2 核定项——
            // 期望值不冻结，3.1 落地后按实际输出回填核对）
            std::cout << "  [ut_apace] " << param.case_name << " key=" << apaceTilingInfo.tilingKey
                      << " dataSize=" << apaceTilingInfo.tilingDataSize << " commTile(tile/cnt/tail/cnt/nonSplit)="
                      << apaceTd->commTile.splitAxisTileSize << "/" << apaceTd->commTile.splitAxisTileCnt << "/"
                      << apaceTd->commTile.splitAxisTailSize << "/" << apaceTd->commTile.splitAxisTailCnt << "/"
                      << apaceTd->commTile.nonSplitAxisSize << " mmTile(m/n/k/baseM/baseN/baseK/stepK/nBuf/dbL0c)="
                      << apaceTd->mmTile.m << "/" << apaceTd->mmTile.n << "/" << apaceTd->mmTile.k << "/"
                      << apaceTd->mmTile.baseM << "/" << apaceTd->mmTile.baseN << "/" << apaceTd->mmTile.baseK
                      << "/" << static_cast<uint32_t>(apaceTd->mmTile.stepK) << "/"
                      << static_cast<uint32_t>(apaceTd->mmTile.nBufferNum) << "/"
                      << static_cast<uint32_t>(apaceTd->mmTile.dbL0c) << " gatherLen=" << apaceTd->gatherLen
                      << " ws=" << (apaceTilingInfo.workspaceSizes.empty() ?
                                          -1 : apaceTilingInfo.workspaceSizes[0])
                      << std::endl;
        } else {
            // HA8 off 侧（AT-150）：使能开关 off（默认）→ v1 分支
            ASSERT_EQ(static_cast<uint64_t>(apaceTilingInfo.tilingKey), utV1Key)
                << "HA8-off: v1 分支 key 为旧值（T-A3 零回填口径）";
            EXPECT_EQ(static_cast<uint64_t>(apaceTilingInfo.tilingKey) & UT_APACE_IMPL_KEY_BIT, 0ULL)
                << "HA8-off: key 最高位 = 0";
            EXPECT_EQ(param.expectTilingKey, utV1Key) << "csv expectTilingKey = v1 旧值";
            EXPECT_GE(apaceTilingInfo.tilingDataSize, sizeof(Mc2Tiling::AllGatherMatmulTilingDataV2));
            const auto *utV1Td = reinterpret_cast<const Mc2Tiling::AllGatherMatmulTilingDataV2 *>(
                apaceTilingInfo.tilingData.get());
            EXPECT_EQ(utV1Td->param.rankM, static_cast<uint32_t>(apaceM)) << "HA8-off: v1 结构 rankM";
            EXPECT_EQ(utV1Td->param.rankK, static_cast<uint32_t>(apaceK)) << "HA8-off: v1 结构 rankK";
            EXPECT_EQ(utV1Td->param.rankN, static_cast<uint32_t>(apaceN)) << "HA8-off: v1 结构 rankN";
            std::cout << "  [ut_apace-off] " << param.case_name << " key=" << apaceTilingInfo.tilingKey
                      << "（v1 结构，bit9=0）" << std::endl;
        }
        (void)unsetenv(kAscendMc2UsingApaceImpl);
        if (debugModeEnvSet) {
            (void)unsetenv(kAscendMc2DebugMode);
        }
        return;
    }
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey,
                       param.expectTilingDataHash, {}, MC2_TILING_DATA_RESERVED_LEN, true);
    if (debugModeEnvSet) {
        (void)unsetenv(kAscendMc2DebugMode);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmulV2, AllGatherMatmulV2Arch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    GetCaseInfoString<AllGatherMatmulV2TilingUtParam>);

} // namespace all_gather_matmul_v2_ut
