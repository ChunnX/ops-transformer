/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GPL-2.0 only license.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef RT_SOC_SPEC_MOCKER_H
#define RT_SOC_SPEC_MOCKER_H

extern "C" {
void SetRtSocSpecShortVersion(const char *version);
void SetRtSocSpecNpuArch(const char *arch);
void SetRtSocSpecFail(bool fail);
}

#endif // RT_SOC_SPEC_MOCKER_H
