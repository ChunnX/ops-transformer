/*!
 * \file sage_attention2_tiling.cpp
 * \brief SageAttention2 Host TilingImpl 注册：将闭式 Tiling 结果序列化进
 *        gert::TilingContext 的 RawTilingData（两段式 CommonOpExecutorRun
 *        下发 kernel 时由框架从该 buffer 取 tiling）。
 *
 * 序列化写法对齐仓库内其他实验算子的标准模式（如 turbo_quant_sparse_flash_attention
 * 的 TilingImpl::SetTilingData）：写入 RawTilingData->GetData() 并 SetDataSize。
 * Tiling 数值计算复用 aclnn/sage_attn_host_tiling.h 的 sage_attn::ComputeHostTiling
 * 闭式实现，保证 GetWorkspaceSize 阶段与 TilingImpl 阶段结果位元一致。
 */

#include "sage_attention2_def.h"

#include <cmath>
#include <cstring>

// 闭式 Tiling 计算、tiling 结构体、数值编码与字符串映射均经本头引入
// （sage_attn_host_tiling.h -> sage_attn_proto.h -> ascendc/tiling.h）
#include "../aclnn/sage_attn_host_tiling.h"

// M5 编译期位型特化：tiling key 单点编码（dtype nibble bits 20-23；canonical
// 模板档位出包时宏映射 GET_TPL_TILING_KEY，裸 bisheng 出包为本仓冻结编码）
#include "../op_kernel/sage_attention2_template_tiling_key.h"

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"

namespace ops {

namespace {

/**
 * 标准宿主 Tiling 入口：
 * 1. 读取 Q/K/V 形状与全部字符串/标量属性并映射为数值编码
 *    （唯一映射位于 sage_attn_proto.h 的 *CodeFromString）
 * 2. 复用闭式 ComputeHostTiling 计算并填充 SageAttention2TilingData
 * 3. 将 tiling 位元写入 gert::RawTilingData（框架经 CommonOpExecutorRun 下发至 kernel）
 * 4. 设置 BlockDim 与 Workspace 大小
 */
static ge::graphStatus TilingSageAttention2(gert::TilingContext *context)
{
    if (context == nullptr || context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *query_shape = nullptr;
    const gert::Shape *key_shape = nullptr;
    const gert::Shape *value_shape = nullptr;
    if (context->GetInputShape(sage_attn2_idx::IN_QUERY) != nullptr &&
        context->GetInputShape(sage_attn2_idx::IN_KEY) != nullptr &&
        context->GetInputShape(sage_attn2_idx::IN_VALUE) != nullptr) {
        // TilingContext 返回 StorageShape，取原始逻辑 shape 参与闭式计算
        query_shape = &context->GetInputShape(sage_attn2_idx::IN_QUERY)->GetOriginShape();
        key_shape = &context->GetInputShape(sage_attn2_idx::IN_KEY)->GetOriginShape();
        value_shape = &context->GetInputShape(sage_attn2_idx::IN_VALUE)->GetOriginShape();
    }
    if (query_shape == nullptr || key_shape == nullptr || value_shape == nullptr ||
        query_shape->GetDimNum() != 4 || key_shape->GetDimNum() != 4 || value_shape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const auto *attrs = context->GetAttrs();
    const char *layout_str = (attrs != nullptr) ? ReadStringAttr(attrs, sage_attn2_idx::ATTR_TENSOR_LAYOUT) : nullptr;
    const char *gran_str = (attrs != nullptr) ? ReadStringAttr(attrs, sage_attn2_idx::ATTR_QK_QUANT_GRAN) : nullptr;
    const char *accum_str = (attrs != nullptr) ? ReadStringAttr(attrs, sage_attn2_idx::ATTR_PV_ACCUM_DTYPE) : nullptr;
    const bool *is_causal = (attrs != nullptr) ? attrs->GetBool(sage_attn2_idx::ATTR_IS_CAUSAL) : nullptr;
    const float *sm_scale = (attrs != nullptr) ? attrs->GetFloat(sage_attn2_idx::ATTR_SM_SCALE) : nullptr;
    const bool *smooth_k = (attrs != nullptr) ? attrs->GetBool(sage_attn2_idx::ATTR_SMOOTH_K) : nullptr;
    const bool *smooth_v = (attrs != nullptr) ? attrs->GetBool(sage_attn2_idx::ATTR_SMOOTH_V) : nullptr;

    // 任务书硬契约：严格白名单校验（非法字符串属性必须报错返回 fail-closed）
    if (layout_str != nullptr && !sage_attn::IsValidTensorLayout(layout_str)) {
        return ge::GRAPH_FAILED;
    }
    if (gran_str != nullptr && !sage_attn::IsValidQuantGran(gran_str)) {
        return ge::GRAPH_FAILED;
    }
    if (accum_str != nullptr && !sage_attn::IsValidPvAccumDtype(accum_str)) {
        return ge::GRAPH_FAILED;
    }

    // 微秒级极速发射：tiling 热路径禁止 std::string 堆分配，
    // 缺省折叠与编码映射全部走 const char* 比较
    const char *layout = (layout_str != nullptr) ? layout_str : sage_attn::kDefaultLayout;
    const char *gran = (gran_str != nullptr) ? gran_str : sage_attn::kDefaultQkQuantGran;
    const char *accum = (accum_str != nullptr) ? accum_str : sage_attn::kDefaultPvAccumDtype;

    const int64_t layout_code = sage_attn::TensorLayoutCodeFromString(layout);
    const int64_t accum_code = sage_attn::PvAccumCodeFromString(accum);

    // smooth_v 档位钳制唯一定义：ComputeHostTiling（smoothV && pvAccum==PV_ACCUM_FP32，
    // 见 aclnn/sage_attn_host_tiling.h）——此处不再复制第二份抑制逻辑；
    // 用户告警由 Python 侧 handle_smooth_v 发出
    const bool smooth_v_val = (smooth_v != nullptr) ? *smooth_v : false;

    uint32_t b = 0, sq = 0, sk = 0, hq = 0, hkv = 0, d_og = 0;
    if (layout_code == sage_attn::LAYOUT_HND) {
        // HND 布局: [B, H, S, D]
        b = static_cast<uint32_t>(query_shape->GetDim(0));
        hq = static_cast<uint32_t>(query_shape->GetDim(1));
        sq = static_cast<uint32_t>(query_shape->GetDim(2));
        d_og = static_cast<uint32_t>(query_shape->GetDim(3));
        hkv = static_cast<uint32_t>(key_shape->GetDim(1));
        sk = static_cast<uint32_t>(key_shape->GetDim(2));
    } else {
        // NHD 布局: [B, S, H, D]
        b = static_cast<uint32_t>(query_shape->GetDim(0));
        sq = static_cast<uint32_t>(query_shape->GetDim(1));
        hq = static_cast<uint32_t>(query_shape->GetDim(2));
        d_og = static_cast<uint32_t>(query_shape->GetDim(3));
        sk = static_cast<uint32_t>(key_shape->GetDim(1));
        hkv = static_cast<uint32_t>(key_shape->GetDim(2));
    }

    // sm_scale：GE 图属性未显式提供时为默认 0，此时折叠为 1/sqrt(d)
    // （与 Python 桥接 None->默认 语义一致）；负值为非法显式入参，直接报错
    double scale = (sm_scale != nullptr) ? static_cast<double>(*sm_scale) : 0.0;
    if (scale < 0.0) {
        return ge::GRAPH_FAILED;
    }
    if (scale <= 0.0) {
        scale = 1.0 / std::sqrt(static_cast<double>(d_og));
    }

    sage_attn::SageAttention2TilingData tiling_data;
    uint64_t workspace_bytes = 0;
    // gert::TilingContext 无 GetInputDataType（CANN 9.2.0-beta.1 实测），dtype 经
    // GetInputDesc 的 CompileTimeTensorDesc 读取（仓内 turbo_quant_sparse_flash_attention
    // TilingImpl 同款惯例）；desc 缺失按非法图 fail-closed。
    const auto *query_desc = context->GetInputDesc(sage_attn2_idx::IN_QUERY);
    if (query_desc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto query_dtype = query_desc->GetDataType();
    // 档位常量唯一定义 op_kernel/sage_attention2_template_tiling_key.h（kTplDtypeFp16/Bf16）
    const int dtype_mode = (query_dtype == ge::DT_BF16) ? static_cast<int>(sage_attn::kTplDtypeBf16)
                                                        : static_cast<int>(sage_attn::kTplDtypeFp16);

    uint32_t hw_cores = 0;
    if (context->GetPlatformInfo() != nullptr) {
        hw_cores = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetCoreNumAic();
    }

    aclnnStatus status = sage_attn::ComputeHostTiling(
        b, sq, sk, hq, hkv, dtype_mode, d_og,
        scale,
        sage_attn::QuantGranCodeFromString(gran),
        accum_code,
        (is_causal != nullptr) ? *is_causal : false,
        layout_code,
        (smooth_k != nullptr) ? *smooth_k : true,
        smooth_v_val,
        context->GetOutputShape(sage_attn2_idx::OUT_LSE) != nullptr,   // has_lse 实况
        tiling_data,
        workspace_bytes,
        hw_cores);
    if (status != ACLNN_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // blockDim 以平台实测 AIC 数封顶（28 核减配拓扑防超订；
    // CompileInfo/platform 为权威源，env 覆盖仅供自动化测试）。
    // 封顶必须在序列化之前完成，保证 RawTilingData 与 blockDim 一致。
    if (context->GetPlatformInfo() != nullptr) {
        const uint32_t aic = platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
                                 .GetCoreNumAic();
        if (aic > 0 && aic < tiling_data.core_num_used) {
            tiling_data.core_num_used = aic;
            // cores_per_die 契约定义为 core_num_used / die_num（tiling.h 注释），
            // 随钳制联动，避免序列化出与 core_num_used 矛盾的亲和元数据
            tiling_data.cores_per_die =
                (aic >= sage_attn::ATLAS_950_DIE_NUM) ? (aic / sage_attn::ATLAS_950_DIE_NUM) : aic;
            if ((aic % sage_attn::ATLAS_950_DIE_NUM) != 0) {
                tiling_data.enable_die_affinity = 0;
            }
        }
    }
    context->SetBlockDim(tiling_data.core_num_used);

    // 标准 tiling 序列化：写入 RawTilingData 并回填数据长度（须在全部钳制之后，
    // 保证下发位元与 SetBlockDim 一致）
    auto *raw_tiling = context->GetRawTilingData();
    if (raw_tiling->GetData() == nullptr || raw_tiling->GetCapacity() < sizeof(tiling_data)) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw_tiling->GetData(), &tiling_data, sizeof(tiling_data));
    raw_tiling->SetDataSize(sizeof(tiling_data));

    // TilingKey（M5 编译期位型特化）：dtype 档位 nibble（bits 20-23，编码唯一定义
    // op_kernel/sage_attention2_template_tiling_key.h，标准 GET_TPL_TILING_KEY
    // 单点）。canonical 模板档位出包时框架依此直连模板入口实例；裸 bisheng
    // 出包核内按 tiling.input_dtype 经 RunByDtype 等效一次分派。
    context->SetTilingKey(SAGE_TPL_TILING_KEY(static_cast<uint32_t>(dtype_mode)));

    // 1C2V 架构使用私有 ModeId 4 跨核 flag，非活跃核提前退域无需全局 Barrier，
    // 禁止设置 SetScheduleMode(1)（否则 CANN 批调度器等待全核同步触发 507014 死锁超时）。

    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        constexpr uint64_t reserved = 16ULL * 1024 * 1024; // 16MB CANN 系统预留空间
        workspaces[0] = static_cast<size_t>(reserved + workspace_bytes);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace

/**
 * canonical 注册（对齐仓内规范）：
 * - Tiling：闭式计算 + RawTilingData 序列化 + TilingKey/ScheduleMode/Workspace 元信息；
 * - TilingParse<CompileInfo>：图模式标准交付件（docs/zh/develop/aicore_develop_guide.md
 *   交付件1：注册链与 CompileInfo 结构须保留以满足框架调用规范）。核数权威源为 Tiling
 *   入口的 PlatformAscendC 直查并经 tiling.core_num_used 下发，CompileInfo 无实际逻辑
 *   可置空（原 cores/vectors 写只读零消费字段，2026-10-04 死代码清扫移除）。
 */
struct SageAttention2CompileInfo {};

ge::graphStatus ParseSageAttention2CompileInfo(gert::TilingParseContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return (context->GetCompiledInfo<SageAttention2CompileInfo>() != nullptr) ? ge::GRAPH_SUCCESS
                                                                             : ge::GRAPH_FAILED;
}

IMPL_OP_OPTILING(SageAttention2)
    .Tiling(TilingSageAttention2)
    .TilingParse<SageAttention2CompileInfo>(ParseSageAttention2CompileInfo);

} // namespace ops
