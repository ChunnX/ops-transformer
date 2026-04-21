/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - dispatch_teardown 主流程
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §七 7.1
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §5
 *
 * 重构要点：
 *   - 去掉 MoeTilingBase → Base → arch32/arch35 继承体系（原 7 个文件）
 *   - 单文件实现，arch 差异通过 comm_cfg 参数化
 *   - 公共校验通过公共层组件完成
 *   - epWorldSize 上界：384（与 v2 的 768 不同）
 *   - MX_QUANT dynamicScales dtype 为 DT_FLOAT（与 v2-A5 的 DT_E8M0 不同）
 *
 * 注意（已知 bug，本次不修改，见 architecture_v0.md §1.2）：
 *   - teardown: CheckOptionalAttrValue 中 quantMode 校验使用 && 应为 ||
 *   - teardown: CheckRelationTensorDataType 已声明但未调用
 *   - teardown: 缺失 sharedExpert 组合校验
 */

#include "../../../../common/op_host/op_tiling/moe_dispatch/utils.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/quant_checker_factory.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/comm_cfg.h"
#include "../../../../common/op_host/op_tiling/moe_dispatch/hccl_buffer_checker.h"
// TODO: include tilingData header (MoeDistributeDispatchTeardownInfo)
// TODO: include framework headers

using namespace moe_dispatch_utils;
using namespace moe_dispatch_quant;
using namespace moe_dispatch_defs;

// teardown 的阶段间中间值载体（最小，无 elastic/tp/layered/fullmesh）
struct TeardownTilingCtx {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t quantMode;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    // 阶段 D
    uint32_t localExpertNum;
    bool     isSharedRank;
    // 阶段 F
    uint32_t tokenMsgSize;
};

graphStatus DispatchTeardownTilingFunc(TilingContext* ctx) {
    TeardownTilingCtx tc{};

    /* A: 获取 tilingData */
    // TODO: auto* tilingData = GetTilingData<MoeDistributeDispatchTeardownInfo>(ctx);

    /* B: 属性校验（12 个，无 elastic/tp/layered/fullmesh；epWorldSize<=384）*/
    AttrCheckRule attrRules[] = {
        // {layout.attrEpWorldSizeIndex, "epWorldSize", 2, 384, true},  // teardown: 384
        // TODO: 其余 11 个属性
    };
    CHECK_SUCCESS(CheckAttrRules(ctx, attrRules, COUNT_OF(attrRules)));
    // TODO: 读取属性到 tc
    CHECK_SUCCESS(CheckSharedExpertRelation(tc.sharedExpertNum, tc.sharedExpertRankNum));

    /* C: teardown 无可选特性互斥校验 */

    /* D: 中间值（复用公共层）*/
    auto sharedResult = CalcLocalExpertNum(tc.epRankId, tc.sharedExpertRankNum,
                                           tc.moeExpertNum, tc.epWorldSize);
    tc.localExpertNum = sharedResult.localExpertNum;
    tc.isSharedRank   = sharedResult.isSharedRank;

    /* E: tensor 校验（teardown tensor 列表与 v2/setup 不同）*/
    TensorCheckRule tensorRules[] = {
        // {xIndex,          TensorType::INPUT,  2, {DT_FLOAT16, DT_BF16, DT_INT8}, 3, "x"},
        // {expertIdsIndex,  TensorType::INPUT,  2, {DT_INT32},  1, "expertIds"},
        // {commCmdInfoIndex,TensorType::INPUT,  1, {DT_INT64},  1, "commCmdInfo"},
        // {expandXOutIndex, TensorType::OUTPUT, 2, {...},       ?, "expandXOut"},
        // {expertTokenNumsIndex, TensorType::OUTPUT, 1, {DT_INT64}, 1, "expertTokenNums"},
        // {assistInfoIndex, TensorType::OUTPUT, 1, {DT_INT64},  1, "assistInfo"},
        // {expandXDynScalesIndex, TensorType::OUTPUT, ?, {...}, ?, "expandXDynamicScales"},
        // TODO: 完整 teardown tensor 列表
    };
    CHECK_SUCCESS(CheckTensorRules(ctx, tensorRules, COUNT_OF(tensorRules)));

    /* F: 量化（部分）—— MX dtype 为 DT_FLOAT（teardown 与 v2-A5 不同）*/
    auto quantChecker = CreateQuantChecker(tc.quantMode, DT_FLOAT);  // teardown: MX 用 float
    if (quantChecker == nullptr) return ge::GRAPH_FAILED;
    tc.tokenMsgSize = quantChecker->CalcTokenMsgSize(0 /*h: TODO*/);
    // TODO: CheckDynamicScales（teardown 有 expandXDynamicScales 输出）

    /* G: shape 精确值校验（teardown 独有）*/
    // TODO: 校验 expandXOut dim0 = A
    // TODO: 校验 assistInfo dim0 = ASSIST_INFO_FACTOR * epWorldSize
    // TODO: 校验 expertTokenNums dim0 = moeExpertNum

    /* H: HCCL buffer 校验 */
    // TODO: 计算 minRequired
    CHECK_SUCCESS(hccl_buffer_check::CheckBuffSizeFromContext(ctx, 0 /*groupIndex: TODO*/, 0 /*minRequired*/));

    /* I: 设置 */
    // TODO: SetWorkspace(tilingData, SYSTEM_NEED_WORKSPACE);  // teardown: 16MB
    // TODO: SetTilingKey(tilingData, 10000 + tc.quantMode + shared 标记);
    // 通信配置（teardown 固定 AllToAll + URMA）
    hccl::HcclAlgConfig algConfig{};
    CHECK_SUCCESS(comm_cfg::SetEpAllToAllCfg(ctx, algConfig, /*isUrma=*/true));

    /* J: 日志 */
    // TODO: PrintTilingDataInfo(tilingData);

    return ge::GRAPH_SUCCESS;
}

// TODO: REGISTER_TILING_FUNC 宏注册 teardown 入口（对应 arch35 平台）
