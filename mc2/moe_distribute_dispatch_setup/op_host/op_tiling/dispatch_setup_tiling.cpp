/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - dispatch_setup 主流程
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §七 7.1
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §4
 *
 * 重构要点：
 *   - 去掉 MoeTilingBase → Base → arch32/arch35 继承体系（原 7 个文件）
 *   - 单文件实现，arch 差异通过 comm_cfg 参数化（isUrma）
 *   - 公共校验通过公共层组件完成
 *   - epWorldSize 上界：384（与 v2 的 768 不同）
 */

#include "../../../../common/op_host/op_tiling/moe_dispatch/utils.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/quant_checker_factory.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/active_mask_checker.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/comm_cfg.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/hccl_buffer_checker.h"
// TODO: include tilingData header (MoeDistributeDispatchSetupInfo)
// TODO: include framework headers

using namespace moe_dispatch_utils;
using namespace moe_dispatch_quant;
using namespace moe_dispatch_defs;

// setup 的局部 Layout（仅在本文件使用，不复用 v2 Layout）
struct SetupTilingLayout {
    uint32_t xIndex;
    uint32_t expertIdsIndex;
    int32_t  scalesIndex;       // 可选
    int32_t  activeMaskIndex;   // 可选（仅支持 1D）
    uint32_t yIndex;            // 输出
    uint32_t expandIdxIndex;    // 输出
    uint32_t commCmdInfoIndex;  // 输出
    // 属性索引（11 个，无 tp/layered/commAlg/zeroExpert）
    uint32_t attrEpWorldSizeIndex;
    uint32_t attrEpRankIdIndex;
    uint32_t attrMoeExpertNumIndex;
    uint32_t attrQuantModeIndex;
    uint32_t attrSharedExpertNumIndex;
    uint32_t attrSharedExpertRankNumIndex;
    // TODO: 补充其余属性索引
};

// setup 的阶段间中间值载体
struct SetupTilingCtx {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t quantMode;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    // 阶段 C
    bool hasActiveMask;
    // 阶段 D
    uint32_t localExpertNum;
    bool     isSharedRank;
    // 阶段 F
    uint32_t tokenMsgSize;
    bool     hasScales;
};

graphStatus DispatchSetupTilingFunc(TilingContext* ctx) {
    SetupTilingCtx tc{};

    /* A: 获取 tilingData */
    // TODO: auto* tilingData = GetTilingData<MoeDistributeDispatchSetupInfo>(ctx);

    /* B: 属性校验（上界与 v2 不同：epWorldSize<=384）*/
    // TODO: 构造 setup 的 AttrCheckRule 数组（11 个属性）
    AttrCheckRule attrRules[] = {
        // {layout.attrEpWorldSizeIndex, "epWorldSize", 2, 384, true},   // setup: 384, v2: 768
        // {layout.attrEpRankIdIndex,    "epRankId",    0, 383, true},
        // TODO: 其余 9 个属性
    };
    CHECK_SUCCESS(CheckAttrRules(ctx, attrRules, COUNT_OF(attrRules)));
    // TODO: 读取属性到 tc
    CHECK_SUCCESS(CheckSharedExpertRelation(tc.sharedExpertNum, tc.sharedExpertRankNum));

    /* C: 仅 activeMask（setup 仅支持 1D）*/
    tc.hasActiveMask = false;  // TODO: 检测 activeMask 是否存在
    if (tc.hasActiveMask) {
        bool isExpertMask = false;
        CHECK_SUCCESS(active_mask_check::CheckCompatibility(ctx->GetNodeName(), false, isExpertMask, 0));
    }

    /* D: 中间值（复用公共层）*/
    auto sharedResult = CalcLocalExpertNum(tc.epRankId, tc.sharedExpertRankNum,
                                           tc.moeExpertNum, tc.epWorldSize);
    tc.localExpertNum = sharedResult.localExpertNum;
    tc.isSharedRank   = sharedResult.isSharedRank;

    /* E: tensor 校验（setup tensor 列表与 v2 不同）*/
    TensorCheckRule tensorRules[] = {
        // {layout.xIndex,         TensorType::INPUT,  2, {DT_FLOAT16, DT_BF16, DT_INT8}, 3, "x"},
        // {layout.expertIdsIndex, TensorType::INPUT,  2, {DT_INT32}, 1, "expertIds"},
        // {layout.yIndex,         TensorType::OUTPUT, 2, {DT_FLOAT16, DT_BF16, DT_INT8}, 3, "y"},
        // {layout.expandIdxIndex, TensorType::OUTPUT, 1, {DT_INT32},  1, "expandIdx"},
        // {layout.commCmdInfoIndex, TensorType::OUTPUT, 1, {DT_INT64}, 1, "commCmdInfo"},
        // TODO: scales（可选）, activeMask（可选）
    };
    CHECK_SUCCESS(CheckTensorRules(ctx, tensorRules, COUNT_OF(tensorRules)));

    /* F: 量化（部分）—— setup 无 arch 差异（MX dtype 统一用 DT_FLOAT）*/
    auto quantChecker = CreateQuantChecker(tc.quantMode);
    if (quantChecker == nullptr) return ge::GRAPH_FAILED;
    tc.tokenMsgSize = quantChecker->CalcTokenMsgSize(0 /*h: TODO*/);
    // TODO: scales 存在性判断 + 校验（y dtype 和 scales 的关系）

    /* G: shape 精确值校验（setup 独有）*/
    // TODO: 校验 y dim0 = A (= localExpertNum * bs 相关)
    // TODO: 校验 expandIdx dim0 = A
    // TODO: 校验 commCmdInfo dim0 = COMM_CMD_INFO_FACTOR * epWorldSize
    if (tc.hasActiveMask) {
        // TODO: 仅 1D，dim0 = bs
        CHECK_SUCCESS(active_mask_check::CheckShape(ctx, 0 /*index: TODO*/, 0 /*bs*/, 0 /*k*/));
    }

    /* H: HCCL buffer 校验 */
    // TODO: 计算 minRequired
    CHECK_SUCCESS(hccl_buffer_check::CheckBuffSizeFromContext(ctx, 0 /*groupIndex: TODO*/, 0 /*minRequired*/));

    /* I: 设置（arch 差异通过 GetArchType 参数化，不用继承）*/
    // TODO: SetWorkspace(tilingData, 32 * 1024 * 1024);  // setup: 32MB
    // TODO: SetTilingKey(tilingData, 1000 + tc.quantMode + scales 标记);
    // 通信配置 —— arch 差异通过参数传入
    hccl::HcclAlgConfig algConfig{};
    // TODO: auto archType = GetArchType(ctx);
    // A3: SetEpAllToAllCfg(ctx, algConfig, /*isUrma=*/false) — BatchWrite/SDMA
    // A5: SetEpAllToAllCfg(ctx, algConfig, /*isUrma=*/true)  — AllToAll+URMA
    CHECK_SUCCESS(comm_cfg::SetEpAllToAllCfg(ctx, algConfig, false /*TODO: 按 arch 判断*/));

    /* J: 日志 */
    // TODO: PrintTilingDataInfo(tilingData);

    return ge::GRAPH_SUCCESS;
}

// TODO: REGISTER_TILING_FUNC 宏注册 setup 入口（对应 arch35 平台）
