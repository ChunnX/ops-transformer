/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../all_gather_matmul_v2_host_ut_param.h"
#include "mc2_tiling_case_executor.h"
// v1 tiling 结构（Mc2Tiling::AllGatherMatmulTilingDataV2）：HA2 布局契约
// static_assert 与 HA1 max 口径断言（apace/v1 两结构取大者）解析所需——
// 与 tiling 实现同源
#include "../../../../op_kernel/all_gather_matmul_tiling_data.h"
#include <cstddef>
// APACE 分支 tiling 结构断言（2.1 v2 §5.4.3 HA1~HA7 / §7.2 T-A1/T-A2/T-A4；
// 3.3 资产翻转后非量化默认恒 APACE）解析原始输出所需——与 apace tiling
// 实现同源（mc2/common/op_kernel 经 OP_TILING_INCLUDE 解析）
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
// 2.1 v2 §5.4.3 HA1~HA7 / §7.2 T-A1/T-A2/T-A4：APACE 分支结构断言集
// （ut_apace_ 前缀触发）。默认语义（R13 回落定稿）：非量化（fp16/bf16）恒走
// APACE 分支——USING_APACE_IMPL 使能开关及其 tiling key 维度已从产品代码
// 整体删除，UT 侧不注入/读取任何 env。
// tiling key 布局（R13 回落口径）：6 参占 bit0~8（位宽 1+1+2+2+2+1=9）——
// 非量化 key = 6 维编码（USING_APACE_IMPL 维度已删除，非量化恒 APACE），
// 量化 key 不变。
// ---------------------------------------------------------------------------

// HA2（静态断言，B9⑤ 公共结构布局修复后同步）：hcomm 路径布局契约——
// HCCL 框架从 tiling blob offset 0 解析 Mc2InitTiling.version（task-sink 契约），
// mc2InitTiling 必须位于 apace 结构 offset 0；dumpInfo-first 头部约定仅适用于
// URMA 结构。mc2CcTiling 紧随其后；dumpInfo 位于 mc2InitTiling/mc2CcTiling 之后，
// 前缀（mc2InitTiling→mc2CcTiling→dumpInfo）与 v1 结构
// Mc2Tiling::AllGatherMatmulTilingDataV2 同型同序对齐（异常 dump 回调按调用方
// 传入的 v1 offsetof(..., dumpInfo) 偏移解析 DfxDumpInfo，前缀对齐后
// v1/APACE 两分支均正确）。
static_assert(offsetof(Apace::hcommAllGatherMatmulTilingData, mc2InitTiling) == 0,
              "HA2: mc2InitTiling must reside at offset 0 of hcommAllGatherMatmulTilingData "
              "(HCCL parses Mc2InitTiling.version from tiling blob offset 0, task-sink contract)");
static_assert(offsetof(Apace::hcommAllGatherMatmulTilingData, mc2CcTiling) == sizeof(Mc2InitTiling),
              "HA2: mc2CcTiling must follow mc2InitTiling (prefix aligned with v1/a2a layout)");
static_assert(offsetof(Apace::hcommAllGatherMatmulTilingData, dumpInfo) ==
                  offsetof(Mc2Tiling::AllGatherMatmulTilingDataV2, dumpInfo),
              "HA2: dumpInfo offset must equal v1 AllGatherMatmulTilingDataV2's (dump callback "
              "parses DfxDumpInfo at caller-passed v1 offsetof)");

// 非量化链路 v1 tiling key 换算（all_gather_matmul_tiling_base.cpp GetTilingKey 实源）：
// GET_TPL_TILING_KEY(inputIsBf16Fp16_=1, isBTrans, outputType=0, TPL_DEFAULT_MODE=0,
// SCALE_TYPE_NOT_IS_MX=0, commMode)——bit0=INPUT_IS_BF16FP16(恒1)；bit1=TRANS_B；
// bit2~7 恒 0（OUTPUTDTYPE/QUANTMMMODE/SCALETYPE 均索引 0）；bit8=HCCL_COMM_MODE
// （ai_cpu=1 / ccu=0）。非量化默认恒 APACE：key 即本式 6 维编码值。
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
    AllGatherMatmulV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara = BuildTilingContextParam(param, &compileInfo);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rank_size}};
    // Ka==K 不变量硬门禁（2.1 §7 T5-② / 2.2 F4/R3；3.3 资产翻转后 apace 形态）：
    // v1 侧「tile/tail tCubeTiling.Ka == rankK」在 APACE 分支的对应物为
    // mmTile.k == K 全长（单一 QuantMatmulTilingData，无 tile/tail 双结构——
    // 2.2 §3.4 映射表）；K 轴被切（mmTile.k != K）即静默数据错误，违例 FAIL。
    constexpr const char kUtKakInvPrefix[] = "ut_kak_inv_";
    if (param.expectResult == ge::GRAPH_SUCCESS &&
        param.case_name.compare(0, std::size(kUtKakInvPrefix) - 1, kUtKakInvPrefix) == 0) {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        const int64_t kakK = param.x1.shape_.GetStorageShape().GetDim(1);
        const uint64_t kakV1Key = ComputeV1TilingKeyNonQuant(param.is_trans_b, param.comm_mode);
        ASSERT_EQ(static_cast<uint64_t>(tilingInfo.tilingKey), kakV1Key)
            << "ut_kak_inv: 非量化 key = 6 维编码（USING_APACE_IMPL 维度已删除，非量化恒 APACE）";
        ASSERT_GE(tilingInfo.tilingDataSize, sizeof(Apace::hcommAllGatherMatmulTilingData));
        const auto *kakTd =
            reinterpret_cast<const Apace::hcommAllGatherMatmulTilingData *>(tilingInfo.tilingData.get());
        EXPECT_EQ(kakTd->mmTile.k, static_cast<uint32_t>(kakK))
            << "ut_kak_inv: mmTile.k != K 全长（APACE 侧 K 轴被切，静默数据错误风险）";
        EXPECT_GT(kakTd->mmTile.baseK, 0U) << "ut_kak_inv: baseK 非零";
        EXPECT_LE(kakTd->mmTile.baseK, static_cast<uint32_t>(kakK)) << "ut_kak_inv: baseK <= k（L0 块不超 K 全长）";
        std::cout << "  [ut_kak_inv] " << param.case_name << " key=" << tilingInfo.tilingKey
                  << " mmTile(m/k/baseK)=" << kakTd->mmTile.m << "/" << kakTd->mmTile.k << "/" << kakTd->mmTile.baseK
                  << std::endl;
        return;
    }

    // 白盒结构断言（ut_struct_ 前缀，3.3；3.3 资产翻转后 apace 形态）：读 tiling
    // 原始输出，按 Apace::hcommAllGatherMatmulTilingData 字段（参考 ut_apace_wb_*
    // 系列断言写法）校验驱动 kernel 分支的决策变量与 csv 参数自洽，并打印观测值
    // 作为分支命中证据。v1 专属字段断言（rankM/rankN/biasLen/tileCnt/storageGather）
    // 随默认语义翻转废弃；同 shape/组合与 ut_apace_*/ut_apace_wb_* 重复的用例已删
    // （csv 账目见 3.3 回执），存留用例为独特组合/边界代表。
    constexpr const char kUtStructPrefix[] = "ut_struct_";
    if (param.expectResult == ge::GRAPH_SUCCESS &&
        param.case_name.compare(0, std::size(kUtStructPrefix) - 1, kUtStructPrefix) == 0) {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        TilingInfo tilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        const int64_t mVal = param.x1.shape_.GetStorageShape().GetDim(0);
        const int64_t kVal = param.x1.shape_.GetStorageShape().GetDim(1);
        const int64_t x2Dim0 = param.x2.shape_.GetStorageShape().GetDim(0);
        const int64_t x2Dim1 = param.x2.shape_.GetStorageShape().GetDim(1);
        // n 值判别与 SetTilingArgsDim 同口径：x2 非 transB 为 (k,n)、transB 为 (n,k)
        const int64_t nVal = (kVal == x2Dim0) ? x2Dim1 : x2Dim0;
        uint64_t gatherOutNumel = 0;
        if (param.gather_out.dtype_ != ge::DT_UNDEFINED) {
            const auto &gatherShape = param.gather_out.shape_.GetStorageShape();
            gatherOutNumel = 1;
            for (size_t d = 0; d < gatherShape.GetDimNum(); d++) {
                gatherOutNumel *= static_cast<uint64_t>(gatherShape.GetDim(d));
            }
        }
        // 0) 非量化 key = 6 维编码（USING_APACE_IMPL 维度已删除，非量化恒 APACE），
        //    tiling blob 为 apace 结构
        const uint64_t structV1Key = ComputeV1TilingKeyNonQuant(param.is_trans_b, param.comm_mode);
        ASSERT_EQ(static_cast<uint64_t>(tilingInfo.tilingKey), structV1Key)
            << "ut_struct: 非量化 key = 6 维编码（USING_APACE_IMPL 维度已删除）";
        EXPECT_EQ(param.expectTilingKey, structV1Key) << "csv expectTilingKey 与换算式一致";
        ASSERT_GE(tilingInfo.tilingDataSize, sizeof(Apace::hcommAllGatherMatmulTilingData));
        const auto *apaceTd =
            reinterpret_cast<const Apace::hcommAllGatherMatmulTilingData *>(tilingInfo.tilingData.get());
        // 1) commTile 语义（切分轴=M_per_rank、非切分轴=K）
        EXPECT_EQ(apaceTd->commTile.nonSplitAxisSize, static_cast<uint64_t>(kVal)) << "nonSplitAxisSize == k";
        EXPECT_EQ(apaceTd->commTile.splitAxisTileSize * apaceTd->commTile.splitAxisTileCnt +
                      apaceTd->commTile.splitAxisTailSize * apaceTd->commTile.splitAxisTailCnt,
                  static_cast<uint64_t>(mVal))
            << "M 覆盖完整性: tileSize*tileCnt + tailSize*tailCnt == m";
        EXPECT_LE(apaceTd->commTile.splitAxisTailCnt, 1ULL) << "尾块 tailCnt <= 1";
        EXPECT_GE(apaceTd->commTile.splitAxisTileCnt, 1ULL) << "头块 tileCnt >= 1";
        // 2) gatherLen 两态：is_gather_out=1 且 gatherOut 非空 → 通信直写输出
        //    （gatherLen==0）；否则 gather 落 workspace（gatherLen>0）
        if (param.is_gather_out && gatherOutNumel > 0) {
            EXPECT_EQ(apaceTd->gatherLen, 0ULL) << "is_gather_out=true: 通信直写输出, gatherLen 应为 0";
        } else {
            EXPECT_GT(apaceTd->gatherLen, 0ULL) << "is_gather_out=false: gather 落 workspace, gatherLen 应 > 0";
        }
        if (nVal > 0) {
            // 3) mmTile problemShape（n==0 跳过 mmv3 tiling——同 v1 分支口径）
            EXPECT_EQ(apaceTd->mmTile.m, static_cast<uint32_t>(mVal));
            EXPECT_EQ(apaceTd->mmTile.n, static_cast<uint32_t>(nVal));
            EXPECT_EQ(apaceTd->mmTile.k, static_cast<uint32_t>(kVal));
            EXPECT_GT(apaceTd->mmTile.baseM, 0U) << "baseM 非零";
            EXPECT_GT(apaceTd->mmTile.baseN, 0U) << "baseN 非零";
            EXPECT_GT(apaceTd->mmTile.baseK, 0U) << "baseK 非零";
            EXPECT_LE(apaceTd->mmTile.baseK, static_cast<uint32_t>(kVal)) << "baseK <= k（L0 块不超 K 全长）";
        }
        // 观测值打印（分支命中证据）
        std::cout << "  [ut_struct] " << param.case_name << " key=" << tilingInfo.tilingKey
                  << " commTile(tile/cnt/tail/cnt/nonSplit)=" << apaceTd->commTile.splitAxisTileSize << "/"
                  << apaceTd->commTile.splitAxisTileCnt << "/" << apaceTd->commTile.splitAxisTailSize << "/"
                  << apaceTd->commTile.splitAxisTailCnt << "/" << apaceTd->commTile.nonSplitAxisSize
                  << " mmTile(m/n/k/baseM/baseN/baseK)=" << apaceTd->mmTile.m << "/" << apaceTd->mmTile.n << "/"
                  << apaceTd->mmTile.k << "/" << apaceTd->mmTile.baseM << "/" << apaceTd->mmTile.baseN << "/"
                  << apaceTd->mmTile.baseK << " gatherLen=" << apaceTd->gatherLen << std::endl;
        return;
    }

    // 白盒补充（3.3 v2，ut_wbkey_ 前缀；R13 回落后 env 语义已死）：tiling key
    // 白盒验证——非量化 6 维 key（3 = ccu×transB 形态，3.2 缺口）/ 量化 key 不受
    // 默认语义回落影响（264——切换条件 INPUT_IS_BF16FP16 因子 false 侧）。
    constexpr const char kUtWbKeyPrefix[] = "ut_wbkey_";
    if (param.expectResult == ge::GRAPH_SUCCESS &&
        param.case_name.compare(0, std::size(kUtWbKeyPrefix) - 1, kUtWbKeyPrefix) == 0) {
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
        TilingInfo wbKeyTilingInfo;
        ASSERT_TRUE(ExecuteTiling(tilingContextPara, wbKeyTilingInfo)) << "ut_wbkey: tiling 应 SUCCESS";
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
        EXPECT_EQ(static_cast<uint64_t>(wbKeyTilingInfo.tilingKey), param.expectTilingKey)
            << "ut_wbkey: key 与 csv 期望一致（非量化=6 维 key / 量化=量化 key 不变）";
        std::cout << "  [ut_wbkey] " << param.case_name << " key=" << wbKeyTilingInfo.tilingKey << "（期望 "
                  << param.expectTilingKey << "）" << std::endl;
        return;
    }

    // APACE 分支结构断言（ut_apace_ 前缀，2.1 v2 §5.4.3 HA1~HA7 / §7.2 T-A1/T-A2/T-A4；
    // R13 回落后非量化默认恒 APACE，无 env 两侧）：解析
    // Apace::hcommAllGatherMatmulTilingData 六字段（HA1/HA3~HA7）+ key 6 维形态（T-A4）。
    constexpr const char kUtApacePrefix[] = "ut_apace_";
    if (param.expectResult == ge::GRAPH_SUCCESS &&
        param.case_name.compare(0, std::size(kUtApacePrefix) - 1, kUtApacePrefix) == 0) {
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
        // T-A4/HA7：非量化 key = 6 维编码（USING_APACE_IMPL 维度已删除，非量化恒 APACE）
        ASSERT_EQ(static_cast<uint64_t>(apaceTilingInfo.tilingKey), utV1Key)
            << "T-A4/HA7: 非量化 key = 6 维编码（USING_APACE_IMPL 维度已删除，非量化恒 APACE）";
        EXPECT_EQ(param.expectTilingKey, utV1Key) << "csv expectTilingKey 与换算式一致";
        // HA1：SetDataSize 取 max（两分支 tiling 结构取大者）——≥ 两结构 sizeof
        // （V2-B6 方向核对兜底：v1 ≥ apace 方向性 + UT 双断言）
        EXPECT_GE(apaceTilingInfo.tilingDataSize, sizeof(Apace::hcommAllGatherMatmulTilingData))
            << "HA1: tilingDataSize >= sizeof(apace 结构)";
        EXPECT_GE(apaceTilingInfo.tilingDataSize, sizeof(Mc2Tiling::AllGatherMatmulTilingDataV2))
            << "HA1: tilingDataSize >= sizeof(v1 结构)（max 口径）";
        const auto *apaceTd =
            reinterpret_cast<const Apace::hcommAllGatherMatmulTilingData *>(apaceTilingInfo.tilingData.get());
        // HA3：commTile 语义（切分轴=M_per_rank、非切分轴=K；fit-balance+256 对齐封顶）
        EXPECT_EQ(apaceTd->commTile.nonSplitAxisSize, static_cast<uint64_t>(apaceK)) << "HA3: nonSplitAxisSize == k";
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
            const uint64_t mmv3MClamp = (static_cast<uint64_t>(apaceM) + 15ULL) / 16ULL * 16ULL; // CeilAlign(m,16)
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
        // 3.3 v2.4 白盒标记断言（P10 修复分支决策变量，3.4 上卡补验 v1.7 §12.6 交办；
        // 跨界判据与 3.4 §12.2 实测规律一致：跨界 ⟺ headRows%baseM≠0 或 tileM%baseM≠0）：
        if (param.case_name.find("_xregion_") != std::string::npos) {
            // P10-①：区域边界跨界块 → Process 分段循环 seg>=2 多段路径
            // （all_gather_bmm_kernel.h L403-450，段边界=region/MAIN-round 边界）
            const uint64_t wbHeadRows = apaceTd->commTile.splitAxisTileSize * apaceTd->commTile.splitAxisTileCnt;
            const uint64_t wbBaseM = apaceTd->mmTile.baseM;
            EXPECT_TRUE(wbHeadRows % wbBaseM != 0ULL || apaceTd->commTile.splitAxisTileSize % wbBaseM != 0ULL)
                << "WB-xregion: 区域边界跨界块（headRows%baseM / tileM%baseM 至少一个非 0"
                << " → P10 分段循环多段路径决策变量成立）";
            std::cout << "  [ut_apace-wb] headRows=" << wbHeadRows << " baseM=" << wbBaseM
                      << " headRows%baseM=" << (wbHeadRows % wbBaseM)
                      << " tileM%baseM=" << (apaceTd->commTile.splitAxisTileSize % wbBaseM)
                      << "（非 0 即跨界块 → seg>=2 多段路径，上卡 P10 原 FAIL 集同形态）" << std::endl;
        }
        if (param.case_name.find("_noseg_") != std::string::npos) {
            // P10-①退化侧：无跨界块（对齐）→ 单段原路径（首段 segStop=blkEnd 即块尾，
            // 逐值等价修复前路径；N2 L396「无跨界块时退化为单段原路径」的驱动变量侧）
            const uint64_t wbHeadRows = apaceTd->commTile.splitAxisTileSize * apaceTd->commTile.splitAxisTileCnt;
            const uint64_t wbBaseM = apaceTd->mmTile.baseM;
            EXPECT_EQ(wbHeadRows % wbBaseM, 0ULL) << "WB-noseg: headRows 对齐 baseM（无跨界块）";
            EXPECT_EQ(apaceTd->commTile.splitAxisTileSize % wbBaseM, 0ULL) << "WB-noseg: tileM 对齐 baseM（无跨界块）";
        }
        if (param.case_name.find("_tailsplit_") != std::string::npos) {
            // P10-②：mmv3 尾块分裂协议参数产出侧保留（mTailTile 等非平凡输出——修复前
            // 该形态曾驱动 basic 调度器畸形块，L1-133 上卡 printf 实测）；kernel 消费侧
            // 已固定 1（Run L300-313 plain 网格化）——本断言钉住「产出侧保留、消费侧
            // 解耦」事实（tiling 输出与修复前逐字节不变，P10 修复不触 tiling 侧）
            EXPECT_TRUE(apaceTd->mmTile.mTailTile > 1U || apaceTd->mmTile.mBaseTailSplitCnt > 1U ||
                        apaceTd->mmTile.mTailMain > 0U)
                << "WB-tailsplit: mmv3 尾块分裂参数非平凡（tiling 产出侧保留；kernel 消费侧"
                << " P10 修复固定 1——断言钉住产出侧保留事实）";
            std::cout << "  [ut_apace-wb] mTailTile=" << apaceTd->mmTile.mTailTile
                      << " mBaseTailSplitCnt=" << apaceTd->mmTile.mBaseTailSplitCnt
                      << " mTailMain=" << apaceTd->mmTile.mTailMain
                      << "（kernel 消费侧固定 1，P10-② plain 网格；core_num=28 镜像真卡 950PR）" << std::endl;
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
        EXPECT_TRUE(
            UtHasAnyNonZeroByte(reinterpret_cast<const uint8_t *>(&apaceTd->mc2CcTiling), sizeof(apaceTd->mc2CcTiling)))
            << "HA6: mc2CcTiling 非空";
        // 观测值打印（分支命中证据；stepK/nBufferNum 为 2.2 §3.4 W2 核定项——
        // 期望值不冻结，3.1 落地后按实际输出回填核对）
        std::cout << "  [ut_apace] " << param.case_name << " key=" << apaceTilingInfo.tilingKey
                  << " dataSize=" << apaceTilingInfo.tilingDataSize
                  << " commTile(tile/cnt/tail/cnt/nonSplit)=" << apaceTd->commTile.splitAxisTileSize << "/"
                  << apaceTd->commTile.splitAxisTileCnt << "/" << apaceTd->commTile.splitAxisTailSize << "/"
                  << apaceTd->commTile.splitAxisTailCnt << "/" << apaceTd->commTile.nonSplitAxisSize
                  << " mmTile(m/n/k/baseM/baseN/baseK/stepK/nBuf/dbL0c)=" << apaceTd->mmTile.m << "/"
                  << apaceTd->mmTile.n << "/" << apaceTd->mmTile.k << "/" << apaceTd->mmTile.baseM << "/"
                  << apaceTd->mmTile.baseN << "/" << apaceTd->mmTile.baseK << "/"
                  << static_cast<uint32_t>(apaceTd->mmTile.stepK) << "/"
                  << static_cast<uint32_t>(apaceTd->mmTile.nBufferNum) << "/"
                  << static_cast<uint32_t>(apaceTd->mmTile.dbL0c) << " gatherLen=" << apaceTd->gatherLen
                  << " ws=" << (apaceTilingInfo.workspaceSizes.empty() ? -1 : apaceTilingInfo.workspaceSizes[0])
                  << std::endl;
        return;
    }
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey,
                       param.expectTilingDataHash, {}, MC2_TILING_DATA_RESERVED_LEN, true);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherMatmulV2, AllGatherMatmulV2Arch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<AllGatherMatmulV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    GetCaseInfoString<AllGatherMatmulV2TilingUtParam>);

} // namespace all_gather_matmul_v2_ut
