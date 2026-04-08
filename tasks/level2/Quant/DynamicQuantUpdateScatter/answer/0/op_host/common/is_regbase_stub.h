/**
 * 替代缺失的 `tiling_base/tiling_util.h` 中对 `IsRegbaseSocVersion` 的依赖（Bench 包内路径不可用时使用）。
 * 若需准确 regbase 判定，可改为链接 CANN 自带 tiling_util 或拷贝完整实现。
 */
#pragma once

#include "register/op_impl_registry.h"

inline bool IsRegbaseSocVersion(const gert::TilingParseContext *context)
{
    (void)context;
    return false;
}

inline bool IsRegbaseSocVersion(const gert::TilingContext *context)
{
    (void)context;
    return false;
}
