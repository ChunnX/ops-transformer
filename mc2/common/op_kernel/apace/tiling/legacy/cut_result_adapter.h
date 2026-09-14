/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cut_result_adapter.h
 * \brief CutResult{long/short} 与 CommTilingData{head/tail} 的双向适配（L3 输出层）
 *
 * 语义映射（v1 apace-impl 分支的 AlignCutResForKernel 已验证该映射方向可行）:
 *   head(headSize=longTileLen, headCnt=numLongTile) + tail(tailSize=shortTileLen, tailCnt=numShortTile)
 *
 * 约束: 仅支持 shortTileAtBack == true 的切分（短块前置的通信 bound 切分无法用
 * head/tail 表达, 调用方应继续直接消费 CutResult）。返回 false 表示切分不可适配。
 */

#ifndef __APACE_LEGACY_CUT_RESULT_ADAPTER_H__
#define __APACE_LEGACY_CUT_RESULT_ADAPTER_H__

#pragma once

#include "../tiling_data_struct_define.h"
#include "legacy_tiling_types.h"

namespace apace {
namespace legacy {

// CutResult -> CommTilingData 的切分轴字段（nonSplitAxisSize 由算子侧按自身布局另填）
inline bool FillCommTilingDataFromCut(const CutResult &cut, CommTilingData &tiling)
{
    if (!cut.shortTileAtBack) {
        return false; // 短块前置无法用 head/tail 表达
    }
    tiling.splitAxisTileSize = cut.longTileLen;
    tiling.splitAxisTileCnt = cut.numLongTile;
    tiling.splitAxisTailSize = cut.shortTileLen;
    tiling.splitAxisTailCnt = cut.numShortTile;
    return true;
}

// CommTilingData 的切分轴字段 -> CutResult（短块恒后置）
inline CutResult CutResultFromCommTiling(const CommTilingData &tiling)
{
    CutResult cut;
    cut.shortTileAtBack = true;
    cut.longTileLen = tiling.splitAxisTileSize;
    cut.numLongTile = tiling.splitAxisTileCnt;
    cut.shortTileLen = tiling.splitAxisTailSize;
    cut.numShortTile = tiling.splitAxisTailCnt;
    cut.totalTileCnt = cut.numLongTile + cut.numShortTile;
    return cut;
}

} // namespace legacy
} // namespace apace

#endif // __APACE_LEGACY_CUT_RESULT_ADAPTER_H__
