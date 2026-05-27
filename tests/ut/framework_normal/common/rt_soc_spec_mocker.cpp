/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GPL-2.0 only license.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include <securec.h>
#include <string>
#include <runtime/rt_external_base.h>

namespace {
std::string g_shortSocVersion = "Ascend910B";
std::string g_npuArch = "";
bool g_rtGetSocSpecFail = false;
}

extern "C" {
void SetRtSocSpecShortVersion(const char *version)
{
    g_shortSocVersion = version ? version : "";
}

void SetRtSocSpecNpuArch(const char *arch)
{
    g_npuArch = arch ? arch : "";
}

void SetRtSocSpecFail(bool fail)
{
    g_rtGetSocSpecFail = fail;
}

rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen)
{
    if (g_rtGetSocSpecFail) {
        return static_cast<rtError_t>(1);
    }
    if (label == nullptr || key == nullptr || val == nullptr) {
        return static_cast<rtError_t>(1);
    }
    if (std::string(label) == "version" && std::string(key) == "Short_SoC_version") {
        if (maxLen == 0) {
            return static_cast<rtError_t>(1);
        }
        const auto &v = g_shortSocVersion;
        auto copyLen = static_cast<uint32_t>(v.size());
        if (copyLen > maxLen - 1) {
            copyLen = maxLen - 1;
        }
        if (memcpy_s(val, maxLen, v.c_str(), copyLen) != EOK) {
            return static_cast<rtError_t>(1);
        }
        val[copyLen] = '\0';
        return RT_ERROR_NONE;
    }
    if (std::string(label) == "version" && std::string(key) == "NpuArch") {
        if (maxLen == 0) {
            return static_cast<rtError_t>(1);
        }
        const auto &v = g_npuArch;
        auto copyLen = static_cast<uint32_t>(v.size());
        if (copyLen > maxLen - 1) {
            copyLen = maxLen - 1;
        }
        if (memcpy_s(val, maxLen, v.c_str(), copyLen) != EOK) {
            return static_cast<rtError_t>(1);
        }
        val[copyLen] = '\0';
        return RT_ERROR_NONE;
    }
    return RT_ERROR_NONE;
}
}
