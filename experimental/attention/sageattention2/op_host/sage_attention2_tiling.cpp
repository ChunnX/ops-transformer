/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
#include <string>

// 闭式 Tiling 计算、tiling 结构体、数值编码与字符串映射均经本头引入
// （sage_attn_host_tiling.h -> sage_attn_proto.h -> ascendc/tiling.h）
#include "../aclnn/sage_attn_host_tiling.h"

#if defined(SAGE_HAS_OP_IMPL_REGISTRY)
#if defined(__has_include)
#if __has_include("exe_graph/runtime/tiling_context.h")
#include "exe_graph/runtime/tiling_context.h"
#define SAGE_HAS_REAL_TILING_CONTEXT 1
#endif
#endif
#endif

#if defined(SAGE_HAS_REAL_TILING_CONTEXT)

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

    // 微秒级极速发射：tiling 热路径禁止 std::string 堆分配，
    // 缺省折叠与编码映射全部走 const char* 比较
    const char *layout = (layout_str != nullptr) ? layout_str : "BNSD";
    const char *gran = (gran_str != nullptr) ? gran_str : "per_thread";
    const char *accum = (accum_str != nullptr) ? accum_str : "fp32+fp16";

    const int64_t layout_code = sage_attn::TensorLayoutCodeFromString(layout);

    uint32_t b = 0, sq = 0, sk = 0, hq = 0, hkv = 0, d_og = 0;
    if (layout_code == sage_attn::LAYOUT_HND) {
        // HND/BNSD 布局: [B, H, S, D]
        b = static_cast<uint32_t>(query_shape->GetDim(0));
        hq = static_cast<uint32_t>(query_shape->GetDim(1));
        sq = static_cast<uint32_t>(query_shape->GetDim(2));
        d_og = static_cast<uint32_t>(query_shape->GetDim(3));
        hkv = static_cast<uint32_t>(key_shape->GetDim(1));
        sk = static_cast<uint32_t>(key_shape->GetDim(2));
    } else {
        // NHD/BSND 布局: [B, S, H, D]
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
    const auto query_dtype = context->GetInputDataType(sage_attn2_idx::IN_QUERY);
    const int dtype_mode = (query_dtype == ge::DT_BF16) ? 2 : 1;

    aclnnStatus status = sage_attn::ComputeHostTiling(
        b, sq, sk, hq, hkv, dtype_mode, d_og,
        scale,
        sage_attn::QuantGranCodeFromString(gran),
        sage_attn::PvAccumCodeFromString(accum),
        (is_causal != nullptr) ? *is_causal : false,
        layout_code,
        (smooth_k != nullptr) ? *smooth_k : true,
        (smooth_v != nullptr) ? *smooth_v : false,
        context->GetInputShape(sage_attn2_idx::IN_KEY_MEAN) != nullptr, // has_key_mean 实况
        false, // has_lse
        tiling_data,
        workspace_bytes);
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

    // TilingKey：运行时模式编码（gran×accum×layout×causal×pad），kernel 侧运行时字段
    // 消费；编译期模板特化（ASCENDC_TPL_ARGS_DECL）待 dtype 分裂时再引入（M5）。
    const uint32_t tiling_key = tiling_data.qk_quant_gran | (tiling_data.pv_accum_dtype << 4) |
                                (tiling_data.tensor_layout << 8) | (tiling_data.is_causal << 12) |
                                ((tiling_data.d_pad == 128u ? 1u : 0u) << 16);
    context->SetTilingKey(static_cast<uint64_t>(tiling_key));

    // SyncAll/三段式调度要求 MIX 1:2 全网格驻留
    context->SetScheduleMode(1);

    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        // dav-3510 GetUserWorkspace 会跳过 CANN 系统 workspace 前缀，host 侧须预留
        uint64_t reserved = 16ULL * 1024 * 1024;
        if (context->GetPlatformInfo() != nullptr) {
            reserved = platform_ascendc::PlatformAscendC(context->GetPlatformInfo())
                           .GetLibApiWorkSpaceSize();
        }
        workspaces[0] = static_cast<size_t>(reserved + workspace_bytes);
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace

/**
 * canonical 注册（对齐仓内规范）：
 * - Tiling：闭式计算 + RawTilingData 序列化 + TilingKey/ScheduleMode/Workspace 元信息；
 * - TilingParse<CompileInfo>：离线期固化 AIC/AIV 核数，运行时免 torch 初始化查询。
 */
struct SageAttention2CompileInfo {
    uint32_t cores = 0;
    uint32_t vectors = 0;
};

ge::graphStatus ParseSageAttention2CompileInfo(gert::TilingParseContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto *info = context->GetCompiledInfo<SageAttention2CompileInfo>();
    if (info == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    info->cores = platform.GetCoreNumAic();
    info->vectors = platform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SageAttention2)
    .Tiling(TilingSageAttention2)
    .TilingParse<SageAttention2CompileInfo>(ParseSageAttention2CompileInfo);

} // namespace ops

#else
// 本地无 CANN exe_graph 头文件环境：TilingImpl 由容器内构建启用，
// 本地路径的 tiling 数值验证由 aclnn/sage_attn_host.cpp 的 ComputeHostTiling 单测覆盖。
#endif // SAGE_HAS_REAL_TILING_CONTEXT
