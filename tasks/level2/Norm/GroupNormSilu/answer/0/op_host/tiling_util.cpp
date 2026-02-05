/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */

#include "tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/soc_spec.h"

namespace OpTiling {
static const gert::Shape g_vec_1_shape = {1};

static bool IsRegbaseSocVersion(NpuArch npuArch)
{
    const static std::set<NpuArch> regbaseArch = {
        NpuArch::DAV_3510};

    return regbaseArch.find(npuArch) != regbaseArch.end();
}

bool IsRegbaseSocVersion(const gert::TilingParseContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto arch = ascendcPlatform.GetCurNpuArch();
    return IsRegbaseSocVersion(arch);
}

bool IsRegbaseSocVersion(const gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto arch = ascendcPlatform.GetCurNpuArch();
    return IsRegbaseSocVersion(arch);
}

const gert::Shape &EnsureNotScalar(const gert::Shape &inShape) {
  if (inShape.IsScalar()) {
    return g_vec_1_shape;
  }
  return inShape;
}
} // namespace OpTiling
