/*
 * Copyright (c) 2026 G_W_E. All rights reserved.
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
#include "acl/acl_rt.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/params.h"

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
ge::graphStatus Tile(gert::TilingContext *c, bool cube, bool splitStats = false)
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
    SageParams p{};
    p.b = q[0];
    p.hq = q[1];
    p.hk = k[1];
    p.sq = q[2];
    p.sk = k[2];
    p.d = q[3];
    p.dp = p.d <= 64 ? 64 : 128;
    p.qp = (p.sq + 127) / 128 * 128;
    p.kp = (p.sk + 511) / 512 * 512;
    const auto *attrs = c->GetAttrs();
    auto sm = attrs->GetAttrPointer<float>(0);
    auto vm = attrs->GetAttrPointer<float>(1);
    auto smooth = attrs->GetAttrPointer<bool>(2);
    auto causal = attrs->GetAttrPointer<bool>(3);
    auto lse = attrs->GetAttrPointer<bool>(4);
    auto smoothV = attrs->GetAttrPointer<bool>(5);
    if (!sm || !vm || !smooth || !causal || !lse || !smoothV || !std::isfinite(*sm) || *vm != 448.0f ||
        (*causal && p.sq != p.sk)) {
        return Failure(__LINE__);
    }
    p.smScale = *sm;
    p.vMax = *vm;
    p.smoothK = *smooth;
    p.smoothV = *smoothV;
    p.causal = *causal;
    p.returnLse = *lse;
    p.bf16 = dt == ge::DT_BF16;
    p.qs[2] = p.d;
    p.qs[1] = uint64_t(p.sq) * p.d;
    p.qs[0] = uint64_t(p.hq) * p.qs[1];
    p.ks[2] = p.vs[2] = p.d;
    p.ks[1] = p.vs[1] = uint64_t(p.sk) * p.d;
    p.ks[0] = p.vs[0] = uint64_t(p.hk) * p.ks[1];
    // Standalone aclnn may provide an empty compile-info object. Runtime
    // platform information is authoritative and does not require torch init.
    uint32_t vectors = 0;
    if (c->GetPlatformInfo()) {
        auto platform = platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
        p.cores = platform.GetCoreNumAic();
        vectors = platform.GetCoreNumAiv();
    }
    if (!p.cores && c->GetCompileInfo()) {
        const auto *info = static_cast<const CompileInfo *>(c->GetCompileInfo());
        p.cores = info->cores;
        vectors = info->vectors;
    }
    if (!p.cores || vectors < 2 * p.cores) {
        // Some direct aclnn launches carry no populated tiling platform data.
        // Resolve optional runtime queries without imposing a runtime dependency
        // on offline tiling tools; never guess core counts from a device model.
        auto getDevice = reinterpret_cast<aclError (*)(int32_t *)>(dlsym(RTLD_DEFAULT, "aclrtGetDevice"));
        auto getInfo = reinterpret_cast<aclError (*)(uint32_t, aclrtDevAttr, int64_t *)>(
            dlsym(RTLD_DEFAULT, "aclrtGetDeviceInfo"));
        int32_t device = 0;
        int64_t aic = 0, aiv = 0;
        if (getDevice && getInfo && getDevice(&device) == ACL_SUCCESS &&
            getInfo(device, ACL_DEV_ATTR_CUBE_CORE_NUM, &aic) == ACL_SUCCESS &&
            getInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv) == ACL_SUCCESS && aic > 0 && aic < UINT32_MAX &&
            aiv >= 2 * aic && aiv < UINT32_MAX) {
            p.cores = static_cast<uint32_t>(aic);
            vectors = static_cast<uint32_t>(aiv);
        }
    }
    if (!p.cores || vectors < 2 * p.cores) {
        return Failure(__LINE__);
    }
    c->SetBlockDim(cube ? p.cores : 2 * p.cores);
    c->SetTilingKey(0);
    // MODE4 CV transport requires the associated Cube and Vector group.
    if (cube) {
        c->SetScheduleMode(1);
    }
    auto raw = c->GetRawTilingData();
    if (!raw || raw->GetCapacity() < sizeof(p)) {
        return Failure(__LINE__);
    }
    std::memcpy(raw->GetData(), &p, sizeof(p));
    raw->SetDataSize(sizeof(p));
    auto ws = c->GetWorkspaceSizes(1);
    if (!ws) {
        return Failure(__LINE__);
    }
    ws[0] = 0;
    if (splitStats && (p.smoothV || (p.d == p.dp && p.sk >= 1024))) {
        // dav3510 GetUserWorkspace skips 16 MiB of system workspace.
        uint64_t reserved = 16ULL * 1024 * 1024;
        if (c->GetPlatformInfo()) {
            reserved = platform_ascendc::PlatformAscendC(c->GetPlatformInfo()).GetLibApiWorkSpaceSize();
        }
        ws[0] = reserved + uint64_t(p.b) * p.hk * SAGE_STATS_SPLITS * (p.smoothV ? 4 : 2) * p.dp * sizeof(float);
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus Stats(gert::TilingContext *c)
{
    return Tile(c, false, true);
}
ge::graphStatus Quant(gert::TilingContext *c)
{
    return Tile(c, false);
}
ge::graphStatus Core(gert::TilingContext *c)
{
    return Tile(c, true);
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
IMPL_OP_OPTILING(SageAttention2Stats).Tiling(Stats).TilingParse<CompileInfo>(Parse);
IMPL_OP_OPTILING(SageAttention2Quant).Tiling(Quant).TilingParse<CompileInfo>(Parse);
IMPL_OP_OPTILING(SageAttention2Core).Tiling(Core).TilingParse<CompileInfo>(Parse);
} // namespace optiling
