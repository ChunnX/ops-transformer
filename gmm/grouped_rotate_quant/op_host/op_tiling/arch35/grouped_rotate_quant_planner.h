/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GROUPED_ROTATE_QUANT_PLANNER_H
#define GROUPED_ROTATE_QUANT_PLANNER_H

#include <array>
#include "../../grouped_rotate_quant_model.h"

namespace optiling::grq {
struct Hardware {
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    uint64_t l0CSize = 0;
    uint64_t ubSize = 0;
};
// Existing RotateQuant Tensor API epilogue: BF16[16384], output[16384],
// limit/max-exp/reciprocal[512] (2 bytes each), scale[512] (1 byte),
// then mTile * 32-byte scale rows. AIVs alternate WHOLE tiles; no M halving.
constexpr uint64_t EPILOGUE_FIXED_BYTES = 52736;
constexpr uint64_t EPILOGUE_MAX_ELEMENTS = 16384;
constexpr uint64_t SYSTEM_WORKSPACE_BYTES = 16 * 1024 * 1024;

inline size_t ProfileIndex(int64_t r) { return r == 32 ? 0 : (r == 64 ? 1 : 2); }

inline bool ResourceLimit(const Hardware &h, uint64_t r, uint32_t &maxM)
{
    const uint64_t rotationBytes = r * r * 2;
    // The rotation-changing path uses two L1 B slots. Reserve both for every
    // profile because the same R can serve shared and per-column rotations.
    const uint64_t rotationL1Bytes = rotationBytes * 2;
    if (h.l1Size <= rotationL1Bytes || h.l0BSize < rotationBytes * 2 || h.ubSize <= EPILOGUE_FIXED_BYTES) {
        return false;
    }
    const uint64_t limit = std::min({h.l0ASize / (r * 4), h.l0CSize / (r * 8),
        (h.l1Size - rotationL1Bytes) / (r * 4), EPILOGUE_MAX_ELEMENTS / r,
        (h.ubSize - EPILOGUE_FIXED_BYTES) / 32});
    maxM = static_cast<uint32_t>(limit / 16 * 16);
    return maxM >= 16;
}

// p has concrete, validated descriptor shapes. Runtime M hints influence tile
// selection only; they never limit loops, addresses, allocations or group sizes.
inline bool BuildPlan(const InputParams &p, const Hardware &h, GroupedRotateQuant::TilingData &data,
                      std::string &error)
{
    if (h.aicNum == 0 || static_cast<uint64_t>(h.aivNum) != 2ULL * h.aicNum) {
        return Fail(error, "Ascend950 RotateQuant requires nonzero AIC and AIV:AIC=2:1");
    }
    data = {};
    data.version = GroupedRotateQuant::TILING_VERSION;
    data.groupNum = static_cast<uint32_t>(p.groupNum);
    data.xNum = static_cast<uint32_t>(p.x.size());
    data.rotationNum = static_cast<uint32_t>(p.rotation.size());
    data.yNum = static_cast<uint32_t>(p.yNum);
    data.groupType = static_cast<int32_t>(p.attrs.groupType);
    data.splitItem = p.attrs.splitItem < 2 ? (p.x.size() == p.yNum ? 0 : 1) : (p.x.size() == 1 ? 3 : 2);
    data.groupListType = static_cast<uint32_t>(p.attrs.groupListType);
    data.hasGroupList = p.hasGroupList;
    data.hasAlpha = p.hasAlpha;
    data.runtimeM = p.runtimeM;
    data.axis = -1;
    data.roundMode = p.attrs.roundMode == "rint" ? 4 : (p.attrs.roundMode == "floor" ? 1 : 0);
    data.scaleAlg = static_cast<uint32_t>(p.attrs.scaleAlg);
    data.dstTypeMax = p.attrs.dstTypeMax;
    data.invDstTypeMax = data.dstTypeMax > 0 ? 1.0f / data.dstTypeMax : 0;
    data.totalM = static_cast<uint64_t>(p.totalM);
    data.totalElements = static_cast<uint64_t>(p.totalElements);
    std::array<int64_t, GroupedRotateQuant::ROTATION_PROFILE_NUM> work{};
    std::array<bool, GroupedRotateQuant::ROTATION_PROFILE_NUM> present{};
    for (size_t i = 0; i < p.groupNum; ++i) {
        const auto &x = p.x[p.x.size() == 1 ? 0 : i];
        const int64_t r = p.rotation[p.rotation.size() == 1 ? 0 : i].back();
        const auto index = ProfileIndex(r);
        present[index] = true;
        int64_t m;
        if (p.runtimeM) {
            m = p.attrs.expectedM > 0 ? p.attrs.expectedM : CeilDiv(p.totalM, static_cast<int64_t>(p.groupNum));
        } else if (!FlattenM(x, m)) {
            return Fail(error, "flattened M overflow");
        }
        int64_t groupWork;
        // Saturation is safe here: this is a performance estimate, never a shape.
        if (!Multiply(m, x.back() / r, groupWork) || !Add(work[index], groupWork, work[index])) {
            work[index] = std::numeric_limits<int64_t>::max();
        }
    }
    for (size_t i = 0; i < present.size(); ++i) {
        if (!present[i]) {
            continue;
        }
        const uint32_t r = 32U << i;
        uint32_t maxM;
        if (!ResourceLimit(h, r, maxM)) {
            return Fail(error, "L1/L0A/L0B/L0C/UB cannot hold a minimum 16xRxR tile");
        }
        const auto target = std::min<int64_t>(maxM, std::max<int64_t>(16, CeilDiv(work[i], h.aicNum)));
        data.tiles[i] = {r, static_cast<uint32_t>(CeilDiv(target, 16) * 16)};
    }
    // Descriptor-known groups are combined into one global tile stream. No per-group launches.
    int64_t tiles = 0;
    if (!p.runtimeM) {
        for (size_t i = 0; i < p.groupNum; ++i) {
            const auto &x = p.x[p.x.size() == 1 ? 0 : i];
            const auto &rot = p.rotation[p.rotation.size() == 1 ? 0 : i];
            const int64_t r = rot.back();
            const int64_t b = x.back() / r;
            const int64_t mTile = data.tiles[ProfileIndex(r)].mTile;
            int64_t m;
            int64_t groupTiles;
            if (!FlattenM(x, m)) {
                return Fail(error, "flattened M overflow");
            }
            if (rot.size() == 3 && b > 1) {
                if (!Multiply(CeilDiv(m, mTile), b, groupTiles)) {
                    return Fail(error, "tile count overflow");
                }
            } else {
                int64_t effectiveM;
                if (!Multiply(m, b, effectiveM)) {
                    return Fail(error, "effective M overflow");
                }
                groupTiles = CeilDiv(effectiveM, mTile);
            }
            if (!Add(tiles, groupTiles, tiles)) {
                return Fail(error, "total tile count overflow");
            }
        }
        data.tileCount = static_cast<uint64_t>(tiles);
    }
    data.usedCoreNum = p.totalElements == 0 ? 1 : (p.runtimeM ? h.aicNum :
        static_cast<uint32_t>(std::min<int64_t>(tiles, h.aicNum)));
    return true;
}
} // namespace optiling::grq
#endif
