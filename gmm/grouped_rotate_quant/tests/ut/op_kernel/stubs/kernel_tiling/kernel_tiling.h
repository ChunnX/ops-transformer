// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#pragma once
#include <cstring>
#define GET_TILING_DATA_WITH_STRUCT(type, name, address) type name{}; std::memcpy(&name, address, sizeof(type))
