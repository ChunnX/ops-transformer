/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GROUPED_ROTATE_QUANT_TILING_DATA_H
#define GROUPED_ROTATE_QUANT_TILING_DATA_H

#ifndef __CCE_AICORE__
#include <cstddef>
#include <cstdint>
#endif

namespace GroupedRotateQuant {
constexpr uint32_t TILING_VERSION = 1;
constexpr uint32_t ROTATION_PROFILE_NUM = 3;
constexpr uint32_t MAX_GROUP_NUM = 1024;
constexpr uint64_t TILING_KEY = 0;

// Profile index: R=32 -> 0, R=64 -> 1, R=128 -> 2. Unused profiles are zero.
// L1 and L0 both use [mTile, rotationSize, rotationSize]. Two A buffers in L1;
// two rotation buffers in L1 and two buffers each in L0A/L0B/L0C. Budget both
// rotation slots even when a shared rotation only needs one resident copy.
#pragma pack(push, 8)
struct alignas(8) RotationTile {
    uint32_t rotationSize;
    uint32_t mTile;
};

// Fixed 128-byte ABI. Per-group shapes/addresses come from GM TensorList descriptors.
// groupList values are NEVER copied into this structure by the host.
struct alignas(8) TilingData {
    uint32_t version;
    uint32_t groupNum;
    uint32_t xNum;
    uint32_t rotationNum;
    uint32_t yNum; // scaleNum == yNum; the two lists have separate GM descriptors
    uint32_t usedCoreNum;
    int32_t groupType;
    uint32_t splitItem; // normalized list mode; runtimeM, not splitItem, decides the M source
    uint32_t groupListType;
    uint32_t hasGroupList;
    uint32_t hasAlpha;
    uint32_t runtimeM; // single x + device groupList: resolve EVERY M_g on device, even E=1
    int32_t axis; // normalized to -1
    uint32_t roundMode; // round=0, floor=1, rint=4
    uint32_t scaleAlg;
    uint32_t reserved;
    float dstTypeMax;
    float invDstTypeMax;
    uint64_t totalM;
    uint64_t totalElements;
    uint64_t tileCount; // exact for descriptor-known M; 0 with runtimeM is NOT an empty-work marker
    RotationTile tiles[ROTATION_PROFILE_NUM];
    uint64_t reservedTail;
};
#pragma pack(pop)
static_assert(sizeof(RotationTile) == 8, "GroupedRotateQuant rotation profile ABI changed");
static_assert(alignof(TilingData) == 8, "GroupedRotateQuant tiling alignment changed");
static_assert(sizeof(TilingData) == 128, "GroupedRotateQuant host/device ABI changed");
#ifndef __CCE_AICORE__
static_assert(offsetof(TilingData, dstTypeMax) == 64, "GroupedRotateQuant attributes offset changed");
static_assert(offsetof(TilingData, totalM) == 72, "GroupedRotateQuant totals offset changed");
static_assert(offsetof(TilingData, tileCount) == 88, "GroupedRotateQuant task count offset changed");
static_assert(offsetof(TilingData, tiles) == 96, "GroupedRotateQuant profiles offset changed");
static_assert(offsetof(TilingData, reservedTail) == 120, "GroupedRotateQuant reserved offset changed");
#endif
} // namespace GroupedRotateQuant
#endif
