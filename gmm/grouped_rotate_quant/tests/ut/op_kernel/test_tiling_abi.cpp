// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
#include <cstddef>
#include <cstdint>
#pragma pack(push, 1)
#include "../../../op_kernel/arch35/grouped_rotate_quant_tiling_data.h"
struct PackingAfterInclude { char c; uint64_t value; };
static_assert(sizeof(PackingAfterInclude) == 9, "header must restore the caller's packing");
#pragma pack(pop)
static_assert(sizeof(GroupedRotateQuant::TilingData) == 128);
static_assert(alignof(GroupedRotateQuant::TilingData) == 8);
static_assert(offsetof(GroupedRotateQuant::TilingData, tiles) == 96);
int main() {}
