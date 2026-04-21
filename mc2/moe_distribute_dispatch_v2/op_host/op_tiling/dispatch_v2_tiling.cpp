/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - dispatch_v2 主流程（A3/A5 MTE）
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §七
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §2.2
 *
 * 重构要点：
 *   - 去掉框架继承（原 MoeTilingBase → helper 匿名命名空间），改为普通函数注册
 *   - 公共校验通过公共层组件完成
 *   - arch 差异（MX dtype、isUrma）通过 ArchType 参数区分
 */

#include "dispatch_v2_layout.h"
#include "dispatch_v2_win_size.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/utils.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/quant_checker_factory.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/elastic_info_checker.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/active_mask_checker.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/performance_info_checker.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/fullmesh_checker.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/comm_cfg.h"
// TODO: include tilingData header (MoeDistributeDispatchV2Info)
// TODO: include framework headers

using namespace moe_dispatch_utils;
using namespace moe_dispatch_quant;
using namespace moe_dispatch_defs;

// ================================================================
// v2 各阶段子函数
// ================================================================

// 阶段 B: 属性读取 + 校验 + 写入 tilingData
static graphStatus CheckAndGetAttrs(TilingContext* ctx,
                                    const MoeDispatchTilingLayout& layout,
                                    DispatchV2TilingCtx* tc) {
    // 公共属性范围校验（上界 v2: ep<=768，与 setup/teardown 的 384 不同）
    AttrCheckRule attrRules[] = {
        {layout.attrEpWorldSizeIndex,      "epWorldSize",      2,   768, true},
        {layout.attrEpRankIdIndex,         "epRankId",         0,   767, true},
        {layout.attrMoeExpertNumIndex,     "moeExpertNum",     1,   512, true},
        {layout.attrQuantModeIndex,        "quantMode",        0,     4, true},
        {layout.attrTpWorldSizeIndex,      "tpWorldSize",      1,   512, true},
        {layout.attrSharedExpertNumIndex,  "sharedExpertNum",  0,     4, true},
        // TODO: 补充其余属性（globalBs, commAlg, zeroExpert 等，共约 17 个）
    };
    CHECK_SUCCESS(CheckAttrRules(ctx, attrRules, COUNT_OF(attrRules)));

    // 读取属性值到 tc
    // TODO: tc->epWorldSize = GetAttr<uint32_t>(ctx, layout.attrEpWorldSizeIndex);
    // TODO: 读取其余属性...
    // TODO: tc->isLayered = (tc->commAlg == LAYERED_COMM_ALG);
    // TODO: tc->isFullMeshV2 = (tc->commAlg == FULLMESH_COMM_ALG);

    // 关联校验（跨属性）
    CHECK_SUCCESS(CheckSharedExpertRelation(tc->sharedExpertNum, tc->sharedExpertRankNum));
    // TODO: epRankId < epWorldSize 校验
    // TODO: sharedExpertRankNum 与 tpWorldSize 关系校验
    return ge::GRAPH_SUCCESS;
}

// 阶段 C: 可选特性互斥校验（产出 bool 标志）
static graphStatus CheckOptionalFeatures(TilingContext* ctx,
                                         const MoeDispatchTilingLayout& layout,
                                         DispatchV2TilingCtx* tc) {
    const char* nodeName = ctx->GetNodeName();

    // elastic_info
    tc->hasElastic = (layout.elasticInfoIndex != LAYOUT_INDEX_ABSENT &&
                      ctx->GetInputTensor(layout.elasticInfoIndex) != nullptr);
    if (tc->hasElastic) {
        CHECK_SUCCESS(elastic_info_check::CheckCompatibility(nodeName, tc->isLayered, tc->tpWorldSize > 1));
    }

    // activeMask
    tc->hasActiveMask = (layout.activeMaskIndex != LAYOUT_INDEX_ABSENT &&
                         ctx->GetInputTensor(layout.activeMaskIndex) != nullptr);
    if (tc->hasActiveMask) {
        // TODO: 获取 isExpertMask 标志
        bool isExpertMask = false;
        CHECK_SUCCESS(active_mask_check::CheckCompatibility(nodeName, tc->isLayered, isExpertMask, tc->globalBs));
    }

    // performanceInfo
    tc->hasPerformance = (layout.performanceInfoIndex != LAYOUT_INDEX_ABSENT &&
                          ctx->GetInputTensor(layout.performanceInfoIndex) != nullptr);
    if (tc->hasPerformance) {
        CHECK_SUCCESS(performance_info_check::CheckCompatibility(nodeName, tc->isLayered));
    }

    // fullmesh
    if (tc->isFullMeshV2) {
        uint32_t bs = 0, k = 0;  // TODO: 从属性获取
        CHECK_SUCCESS(fullmesh_check::CheckConstraints(nodeName, bs, k, tc->tpWorldSize));
    }

    return ge::GRAPH_SUCCESS;
}

// 阶段 D: 中间值计算
static void CalcIntermediateValues(DispatchV2TilingCtx* tc) {
    auto sharedResult = CalcLocalExpertNum(tc->epRankId, tc->sharedExpertRankNum,
                                           tc->moeExpertNum, tc->epWorldSize);
    tc->localExpertNum       = sharedResult.localExpertNum;
    tc->isSharedRank         = sharedResult.isSharedRank;
    tc->moeExpertPerRankNum  = sharedResult.moeExpertPerRankNum;

    // TODO: 计算 A 值（bs, k, localExpertNum, isSharedRank, hasElastic 等）
    tc->A = 0;  // TODO: CalcA(...)
}

// 阶段 E: tensor 基本校验
static graphStatus CheckTensors(TilingContext* ctx,
                                 const MoeDispatchTilingLayout& layout,
                                 const DispatchV2TilingCtx& tc) {
    // 必选 tensor（dim + dtype + format）
    TensorCheckRule tensorRules[] = {
        {layout.xIndex,         TensorType::INPUT,  2, {DT_FLOAT16, DT_BF16, DT_INT8}, 3, "x"},
        {layout.expertIdsIndex, TensorType::INPUT,  2, {DT_INT32},                     1, "expertIds"},
        // TODO: 输出 tensor（expandX、expandExpertIds、expertTokenNums 等）
    };
    CHECK_SUCCESS(CheckTensorRules(ctx, tensorRules, COUNT_OF(tensorRules)));

    // 条件性 tensor
    if (tc.hasElastic) {
        TensorCheckRule r = {(uint32_t)layout.elasticInfoIndex, TensorType::INPUT, 1, {DT_INT64}, 1, "elasticInfo"};
        CHECK_SUCCESS(CheckOneTensorRule(ctx, r));
    }
    if (tc.hasActiveMask) {
        // TODO: 1D/2D 均支持，dtype=UINT8 或 BOOL
        TensorCheckRule r = {(uint32_t)layout.activeMaskIndex, TensorType::INPUT, 0, {DT_UINT8, DT_BOOL}, 2, "activeMask"};
        CHECK_SUCCESS(CheckOneTensorRule(ctx, r));
    }
    if (tc.hasPerformance) {
        TensorCheckRule r = {(uint32_t)layout.performanceInfoIndex, TensorType::INPUT, 1, {DT_INT64}, 1, "performanceInfo"};
        CHECK_SUCCESS(CheckOneTensorRule(ctx, r));
    }

    return ge::GRAPH_SUCCESS;
}

// 阶段 F: 量化校验（arch 差异通过构造参数解耦）
static graphStatus CheckQuant(TilingContext* ctx,
                               const MoeDispatchTilingLayout& layout,
                               DispatchV2TilingCtx* tc) {
    // MX_QUANT 的 dynamicScales dtype 因 arch 而不同
    DataType mxDtype = (tc->archType == ArchType::ARCH_A5) ? DT_E8M0 : DT_FLOAT;
    auto quantChecker = CreateQuantChecker(tc->quantMode, mxDtype);
    if (quantChecker == nullptr) {
        // TODO: OPS_LOG_E("invalid quantMode: %u", tc->quantMode)
        return ge::GRAPH_FAILED;
    }

    // TODO: 获取 xDtype
    DataType xDtype = DT_FLOAT16;  // TODO: 从 tensor desc 获取
    tc->hasScales = quantChecker->ShouldHaveScales(xDtype);

    if (tc->hasScales) {
        // TODO: 获取 scales tensor desc
        uint32_t expectedDim0 = 0;  // TODO: 根据公式计算
        uint32_t h = 0;             // TODO: 从 x tensor 获取
        CHECK_SUCCESS(quantChecker->CheckScales(ctx, nullptr /*scalesDesc*/, expectedDim0, h));
    }

    // TODO: 获取 dynamicScales tensor desc
    uint32_t dim0 = GetExpectedDynamicScalesDim0(tc->A, tc->tpWorldSize);
    CHECK_SUCCESS(quantChecker->CheckDynamicScales(ctx, nullptr /*dynScalesDesc*/, dim0, 0 /*h*/));

    tc->tokenMsgSize = quantChecker->CalcTokenMsgSize(0 /*h*/);

    // expandX dtype 与 quantMode 关系校验
    DataType expandXDtype = DT_FLOAT16;  // TODO: 从 output tensor 获取
    CHECK_SUCCESS(CheckQuantModeMap(tc->quantMode, expandXDtype));

    return ge::GRAPH_SUCCESS;
}

// 阶段 G: shape 精确值校验（v2 独有逻辑）
static graphStatus CheckShapeValues(TilingContext* ctx,
                                    const MoeDispatchTilingLayout& layout,
                                    const DispatchV2TilingCtx& tc) {
    // TODO: 校验 expertTokenNums dim0 = moeExpertNum (或 zeroExpert 调整后)
    // TODO: 校验 expandX dim0 = A
    // TODO: 校验 epRecvCounters shape
    // TODO: 校验 assistInfo dim0 = ASSIST_INFO_FACTOR * epWorldSize
    // TODO: layered 场景的特殊 shape 公式
    // TODO: elastic 调整后的 A 值 shape

    if (tc.hasElastic) {
        CHECK_SUCCESS(elastic_info_check::CheckShape(ctx, layout.elasticInfoIndex, tc.epWorldSize));
    }
    if (tc.hasActiveMask) {
        // TODO: 获取 bs/k
        CHECK_SUCCESS(active_mask_check::CheckShape(ctx, layout.activeMaskIndex, 0 /*bs*/, 0 /*k*/));
    }
    if (tc.hasPerformance) {
        CHECK_SUCCESS(performance_info_check::CheckShape(ctx, layout.performanceInfoIndex, tc.epWorldSize));
    }
    return ge::GRAPH_SUCCESS;
}

// ================================================================
// v2 主流程（A3/A5 统一入口）
// ================================================================

static graphStatus DispatchV2TilingFunc(TilingContext* ctx,
                                         const MoeDispatchTilingLayout& layout,
                                         ArchType archType) {
    DispatchV2TilingCtx tc{};
    tc.archType = archType;

    /* A: 获取 tilingData */
    // TODO: auto* tilingData = GetTilingData<MoeDistributeDispatchV2Info>(ctx);

    /* B: 属性读取 + 校验 */
    CHECK_SUCCESS(CheckAndGetAttrs(ctx, layout, &tc));

    /* C: 可选特性互斥校验 */
    CHECK_SUCCESS(CheckOptionalFeatures(ctx, layout, &tc));

    /* D: 中间值计算 */
    CalcIntermediateValues(&tc);

    /* E: tensor 基本校验 */
    CHECK_SUCCESS(CheckTensors(ctx, layout, tc));

    /* F: 量化校验 */
    CHECK_SUCCESS(CheckQuant(ctx, layout, &tc));

    /* G: shape 精确值校验 */
    CHECK_SUCCESS(CheckShapeValues(ctx, layout, tc));

    /* H: HCCL buffer / WinSize 校验 */
    CHECK_SUCCESS(CheckV2WinSize(ctx, layout, tc));

    /* I: 设置 */
    auto platformInfo = GetPlatformInfo(ctx);
    // TODO: SetPlatformInfo(ctx, tilingData);
    // TODO: SetWorkspace(tilingData, SYSTEM_NEED_WORKSPACE + 512 * platformInfo.aivNum * platformInfo.aivNum);
    // TODO: SetTilingKey(tilingData, GET_TPL_TILING_KEY(...));
    // 通信配置 —— arch 差异通过参数传入，不依赖继承
    hccl::HcclAlgConfig algConfig{};
    CHECK_SUCCESS(comm_cfg::SetEpAllToAllCfg(ctx, algConfig, archType == ArchType::ARCH_A5));
    if (tc.tpWorldSize > 1) {
        CHECK_SUCCESS(comm_cfg::SetTpAllGatherCfg(ctx));
    }

    /* J: 日志 */
    // TODO: PrintTilingDataInfo(tilingData);

    return ge::GRAPH_SUCCESS;
}

// ================================================================
// 注册入口（替代原有框架继承）
// ================================================================

// A3 入口
graphStatus DispatchV2A3Entry(TilingContext* ctx) {
    return DispatchV2TilingFunc(ctx, V2_A3_LAYOUT, ArchType::ARCH_A3);
}

// A5 入口
graphStatus DispatchV2A5Entry(TilingContext* ctx) {
    return DispatchV2TilingFunc(ctx, V2_A5_LAYOUT, ArchType::ARCH_A5);
}

// TODO: REGISTER_TILING_FUNC 宏注册 A3/A5 入口
