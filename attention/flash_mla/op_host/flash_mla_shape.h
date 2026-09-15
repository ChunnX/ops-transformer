/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_SHAPE_H_
#define FLASH_MLA_SHAPE_H_
#include <cstdint>
#include <string>
#include <vector>
namespace FlashMla {
struct MlaAxes {
    int64_t batch = 0;
    int64_t seq = 0;
    int64_t heads = 0;
    int64_t dim = 0;
};
inline bool SplitMlaLayout(const char *input, std::string &query, std::string &out)
{
    if (input == nullptr) {
        return false;
    }
    std::string layout(input);
    auto split = layout.find('_');
    query = layout.substr(0, split);
    out = split == std::string::npos ? query : layout.substr(split + 1);
    if (query != "BSH" && query != "BSND" && query != "BNSD" && query != "TND") {
        return false;
    }
    if (split == std::string::npos) {
        return true;
    }
    return (query != "TND" && out == "NBSD") || (query == "TND" && out == "NTD");
}
template <typename Shape>
bool GetMlaAxes(const Shape &shape, const std::string &layout, int64_t heads, MlaAxes &axes)
{
    const size_t rank = shape.GetDimNum();
    if (heads <= 0) {
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (shape.GetDim(i) < 0) {
            return false;
        }
    }
    if (layout == "BSH" && rank == 3 && shape.GetDim(2) % heads == 0) {
        axes = {shape.GetDim(0), shape.GetDim(1), heads, shape.GetDim(2) / heads};
    } else if (layout == "BSND" && rank == 4) {
        axes = {shape.GetDim(0), shape.GetDim(1), shape.GetDim(2), shape.GetDim(3)};
    } else if (layout == "BNSD" && rank == 4) {
        axes = {shape.GetDim(0), shape.GetDim(2), shape.GetDim(1), shape.GetDim(3)};
    } else if (layout == "TND" && rank == 3) {
        axes = {0, shape.GetDim(0), shape.GetDim(1), shape.GetDim(2)};
    } else {
        return false;
    }
    return axes.heads == heads;
}
inline std::vector<int64_t> MlaOutputShape(const MlaAxes &q, const std::string &out)
{
    if (out == "BSH") {
        return {q.batch, q.seq, q.heads * 512};
    }
    if (out == "BSND") {
        return {q.batch, q.seq, q.heads, 512};
    }
    if (out == "BNSD") {
        return {q.batch, q.heads, q.seq, 512};
    }
    if (out == "NBSD") {
        return {q.heads, q.batch, q.seq, 512};
    }
    if (out == "NTD") {
        return {q.heads, q.seq, 512};
    }
    return {q.seq, q.heads, 512};
}
template <typename Shape>
bool MlaShapeEquals(const Shape &shape, const std::vector<int64_t> &expected)
{
    if (shape.GetDimNum() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (shape.GetDim(i) != expected[i]) {
            return false;
        }
    }
    return true;
}
} // namespace FlashMla
#endif
