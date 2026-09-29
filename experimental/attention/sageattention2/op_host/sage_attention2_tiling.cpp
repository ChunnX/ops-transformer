/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <limits>
#include "acl/acl_rt.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sage_attention2_tiling_data.h"
#include "../op_kernel/sage_attention2_template_tiling_key.h"

namespace optiling {
namespace {
ge::graphStatus Failure(int line)
{
    std::fprintf(stderr, "SageAttention2 tiling validation failed at line %d\n", line);
    return ge::GRAPH_FAILED;
}
struct CompileInfo {
    uint32_t cores = 0, vectors = 0;
};
// Core-count resolution chain: tiling platform info, then offline compile
// info, then runtime device queries. Standalone aclnn launches may carry an
// empty compile-info object; runtime queries resolve optional symbols via
// dlsym without imposing a dependency on offline tiling tools, and never
// guess core counts from a device model.
bool ResolveCoreCounts(const gert::TilingContext *c, uint32_t &cores, uint32_t &vectors)
{
    static uint32_t s_cachedCores = 0, s_cachedVectors = 0;
    if (s_cachedCores > 0 && s_cachedVectors >= 2 * s_cachedCores) {
        cores = s_cachedCores;
        vectors = s_cachedVectors;
        return true;
    }
    if (c->GetPlatformInfo()) {
        auto platform = platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
        cores = platform.GetCoreNumAic();
        vectors = platform.GetCoreNumAiv();
    }
    if (!cores && c->GetCompileInfo()) {
        const auto *info = static_cast<const CompileInfo *>(c->GetCompileInfo());
        cores = info->cores;
        vectors = info->vectors;
    }
    if (!cores || vectors < 2 * cores) {
        static auto getDevice = reinterpret_cast<aclError (*)(int32_t *)>(dlsym(RTLD_DEFAULT, "aclrtGetDevice"));
        static auto getInfo = reinterpret_cast<aclError (*)(uint32_t, aclrtDevAttr, int64_t *)>(
            dlsym(RTLD_DEFAULT, "aclrtGetDeviceInfo"));
        int32_t device = 0;
        int64_t aic = 0, aiv = 0;
        if (getDevice && getInfo && getDevice(&device) == ACL_SUCCESS &&
            getInfo(device, ACL_DEV_ATTR_CUBE_CORE_NUM, &aic) == ACL_SUCCESS &&
            getInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv) == ACL_SUCCESS && aic > 0 && aic < UINT32_MAX &&
            aiv >= 2 * aic && aiv < UINT32_MAX) {
            cores = static_cast<uint32_t>(aic);
            vectors = static_cast<uint32_t>(aiv);
        }
    }
    if (cores > 0 && vectors >= 2 * cores) {
        s_cachedCores = cores;
        s_cachedVectors = vectors;
        return true;
    }
    return false;
}
ge::graphStatus Tile(gert::TilingContext *c)
{
    if (!c || !c->GetAttrs()) {
        return Failure(__LINE__);
    }
    for (int i = 0; i < 3; ++i) {
        if (!c->GetInputShape(i) || !c->GetInputDesc(i)) {
            return Failure(__LINE__);
        }
        const auto &shape = c->GetInputShape(i)->GetStorageShape();
        if (shape.GetDimNum() != 4) {
            return Failure(__LINE__);
        }
        for (int j = 0; j < 4; ++j) {
            if (shape[j] <= 0 || shape[j] >= UINT32_MAX - 64) {
                return Failure(__LINE__);
            }
        }
    }
    const auto &q = c->GetInputShape(0)->GetStorageShape();
    const auto &k = c->GetInputShape(1)->GetStorageShape();
    const auto &v = c->GetInputShape(2)->GetStorageShape();
    auto dt = c->GetInputDesc(0)->GetDataType();
    if ((dt != ge::DT_FLOAT16 && dt != ge::DT_BF16) || c->GetInputDesc(1)->GetDataType() != dt ||
        c->GetInputDesc(2)->GetDataType() != dt || k != v || q[0] != k[0] || q[3] != k[3] || q[3] > 128 ||
        q[1] % k[1] || uint64_t(k[0]) * k[1] >= UINT32_MAX) {
        return Failure(__LINE__);
    }
    if (q[2] > UINT32_MAX - 127 || k[2] > UINT32_MAX - 511) {
        return Failure(__LINE__);
    }
    SageAttention2TilingData t{};
    auto &p = t.params;
    p.b = q[0];
    p.hq = q[1];
    p.hk = k[1];
    p.sq = q[2];
    p.sk = k[2];
    p.d = q[3];
    p.dp = p.d <= 64 ? 64 : 128;
    const auto *attrs = c->GetAttrs();
    auto sm = attrs->GetAttrPointer<float>(0);
    auto smooth = attrs->GetAttrPointer<bool>(1);
    auto causal = attrs->GetAttrPointer<bool>(2);
    auto lse = attrs->GetAttrPointer<bool>(3);
    if (!sm || !smooth || !causal || !lse || !std::isfinite(*sm) || (*causal && p.sq != p.sk)) {
        return Failure(__LINE__);
    }
    p.smScale = *sm;
    p.smoothK = *smooth;
    p.causal = *causal;
    p.returnLse = *lse;
    p.qs[2] = p.d;
    p.qs[1] = uint64_t(p.sq) * p.d;
    p.qs[0] = uint64_t(p.hq) * p.qs[1];
    p.ks[2] = p.vs[2] = p.d;
    p.ks[1] = p.vs[1] = uint64_t(p.sk) * p.d;
    p.ks[0] = p.vs[0] = uint64_t(p.hk) * p.ks[1];
    // Standalone aclnn may provide an empty compile-info object. Runtime
    // platform information is authoritative and does not require torch init.
    uint32_t vectors = 0;
    if (!ResolveCoreCounts(c, p.cores, vectors)) {
        return Failure(__LINE__);
    }
    c->SetBlockDim(p.cores);
    p.qp = (p.sq + 127U) & ~127U;
    p.kp = (p.sk + 511U) & ~511U;
    const bool wide = p.dp == 128 && p.sq % 128 == 0 && p.sk % 512 == 0 && p.sk >= 4096;
    c->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint32_t>(dt), wide));
    // SyncAll requires the entire MIX 1:2 grid to be resident together.
    c->SetScheduleMode(1);
    uint64_t bytes = 0;
    constexpr uint64_t WORKSPACE_ALIGN = 512;
    constexpr uint64_t DEFAULT_RESERVED_WS = 16ULL * 1024 * 1024;
    auto allocate = [&bytes](uint64_t &offset, std::initializer_list<uint64_t> factors) {
        uint64_t size = 1;
        for (auto factor : factors) {
            if (factor && size > std::numeric_limits<uint64_t>::max() / factor) {
                return false;
            }
            size *= factor;
        }
        if (size > std::numeric_limits<uint64_t>::max() - (WORKSPACE_ALIGN - 1)) {
            return false;
        }
        size = (size + (WORKSPACE_ALIGN - 1)) & ~(WORKSPACE_ALIGN - 1);
        if (bytes > std::numeric_limits<uint64_t>::max() - size) {
            return false;
        }
        offset = bytes;
        bytes += size;
        return true;
    };
    if (!allocate(t.mean, {p.b, p.hk, p.dp, sizeof(float)}) || !allocate(t.vMxScale, {p.b, p.hk, p.kp / 32, p.dp}) ||
        !allocate(t.q8, {p.b, p.hq, p.qp, p.dp}) || !allocate(t.k8, {p.b, p.hk, p.kp, p.dp}) ||
        !allocate(t.v8, {p.b, p.hk, p.kp, p.dp}) || !allocate(t.qScale, {p.b, p.hq, p.qp, sizeof(uint16_t)}) ||
        !allocate(t.kScale, {p.b, p.hk, p.kp, sizeof(uint16_t)})) {
        return Failure(__LINE__);
    }
    uint64_t reserved = DEFAULT_RESERVED_WS;
    if (c->GetPlatformInfo()) {
        reserved = platform_ascendc::PlatformAscendC(c->GetPlatformInfo()).GetLibApiWorkSpaceSize();
    }
    if (bytes > std::numeric_limits<size_t>::max() - reserved) {
        return Failure(__LINE__);
    }
    auto raw = c->GetRawTilingData();
    auto ws = c->GetWorkspaceSizes(1);
    if (!raw || raw->GetCapacity() < sizeof(t) || !ws) {
        return Failure(__LINE__);
    }
    std::memcpy(raw->GetData(), &t, sizeof(t));
    raw->SetDataSize(sizeof(t));
    ws[0] = reserved + bytes;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus Parse(gert::TilingParseContext *c)
{
    if (!c || !c->GetPlatformInfo()) {
        return Failure(__LINE__);
    }
    auto *info = c->GetCompiledInfo<CompileInfo>();
    if (!info) {
        return Failure(__LINE__);
    }
    auto platform = platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    info->cores = platform.GetCoreNumAic();
    info->vectors = platform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_OPTILING(SageAttention2).Tiling(Tile).TilingParse<CompileInfo>(Parse);
} // namespace optiling
