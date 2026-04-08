/*!
 * \file softmax_tiling_mocker.cpp
 * \brief Stub for GetSoftMaxMinTmpSize / GetSoftMaxFlashV2MinTmpSize.
 *
 * 与 mc2_hcom_topology_mocker.cpp 的打桩原理相同：
 * 当 UT 构建时，此文件编译进 softmax_tiling_ut_stub.so，并在链接时优先于
 * tiling_api.so 中的同名符号被解析，从而拦截所有来自算子 tiling 代码的调用。
 *
 * 与原始实现的唯一区别：将
 *   PlatformAscendCManager::GetInstance()          // 无参，依赖系统环境
 * 替换为
 *   PlatformAscendCManager::GetInstance(socVersion) // 有参，由 UT 指定 SoC
 *
 * 计算逻辑与 asc-devkit/impl/adv_api/tiling/activation/softmax_tiling.cpp 完全一致。
 */

#include <algorithm>
#include <vector>
#include <cstdint>
#include <iostream>

// 声明头：函数签名在 namespace AscendC 内
#include "activation/softmax_tiling.h"
// platform：PlatformAscendCManager::GetInstance(socVersion)
#include "tiling/platform/platform_ascendc.h"

#include "softmax_tiling_mocker.h"

namespace AscendC {

// ---- 与 softmax_tiling.cpp 相同的内部常量 --------------------------------
static constexpr uint32_t SOFTMAX_DEFAULT_BLK_SIZE     = 32U;
static constexpr uint32_t SOFTMAX_TMPBUFFER_COUNT      = 2U;
static constexpr uint32_t SOFTMAX_TMPFLASHUPDATE_COUNT = 4U;
static constexpr uint32_t SOFTMAX_HALF_SIZE            = 2U;
static constexpr uint32_t SOFTMAX_FLOAT_SIZE           = 4U;
static constexpr uint32_t BASIC_TILE_NUM               = SOFTMAX_DEFAULT_BLK_SIZE / SOFTMAX_FLOAT_SIZE;
static constexpr uint32_t SOFTMAX_BASICBLOCK_UNIT      = 64U;
static constexpr uint32_t SOFTMAX_SPECIAL_BASICBLOCK_LEN =
    256U * SOFTMAX_FLOAT_SIZE;  // SOFTMAX_BASICBLOCK_MIN_SIZE(256) * SOFTMAX_FLOAT_SIZE

// ---- 与 softmax_tiling.cpp 相同的内部辅助函数 ----------------------------
inline std::vector<uint32_t> GetLastAxisShapeND(const ge::Shape& srcShape)
{
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    uint32_t calculateSize = 1U;
    for (size_t i = 0U; i < shapeDims.size(); i++) {
        calculateSize *= static_cast<uint32_t>(shapeDims[i]);
    }
    const uint32_t srcK = static_cast<uint32_t>(shapeDims.back());
    const uint32_t srcM = calculateSize / srcK;
    return { srcM, srcK };
}

// ---- 打桩：GetSoftMaxMinTmpSize ------------------------------------------
// 逻辑与原版完全一致，仅将 GetInstance() 替换为 GetInstance(socVersion)
uint32_t GetSoftMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize,
    const bool /*isReuseSource*/)
{
    std::cout << "20260325 GetSoftMaxMinTmpSize_mocker" << std::endl;
    const uint32_t inputSize = static_cast<uint32_t>(srcShape.GetShapeSize());
    if (inputSize == 0U) { return 0U; }
    if (dataTypeSize == 0U || (dataTypeSize != SOFTMAX_HALF_SIZE && dataTypeSize != SOFTMAX_FLOAT_SIZE)) {
        return 0U;
    }

    const std::vector<uint32_t> retVec = GetLastAxisShapeND(srcShape);
    const uint32_t srcM = retVec[0];
    const uint32_t srcK = retVec[1];
    const uint32_t elementNumPerBlk = SOFTMAX_DEFAULT_BLK_SIZE / dataTypeSize;

    // 关键替换：使用有参 GetInstance，由 SoftmaxTilingMocker 提供 SoC 版本字符串
    platform_ascendc::PlatformAscendC* platform =
        platform_ascendc::PlatformAscendCManager::GetInstance(
            SoftmaxTilingMocker::GetInstance().GetSocVersion());
    if (platform == nullptr) { return 0U; }
    const auto npuArch = platform->GetCurNpuArch();

    uint32_t needSize;
    if (npuArch == NpuArch::DAV_3510 ||
        npuArch == NpuArch::DAV_3003 ||
        npuArch == NpuArch::DAV_5102) {
        const uint32_t needSize1 = srcM * (BASIC_TILE_NUM + srcK) +
            SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_TMPFLASHUPDATE_COUNT +
            (srcM + BASIC_TILE_NUM - 1U) / BASIC_TILE_NUM * BASIC_TILE_NUM;
        const uint32_t needSize2 = srcM * (elementNumPerBlk + srcK);
        needSize = std::max(needSize1, needSize2);
    } else {
        needSize = elementNumPerBlk + srcK + SOFTMAX_BASICBLOCK_UNIT;
    }
    return needSize * SOFTMAX_FLOAT_SIZE;
}

// ---- 打桩：GetSoftMaxFlashV2MinTmpSize -----------------------------------
// 逻辑与原版完全一致，仅将 GetInstance() 替换为 GetInstance(socVersion)
uint32_t GetSoftMaxFlashV2MinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1,
    const uint32_t dataTypeSize2, const bool isUpdate, const bool isBasicBlock,
    const bool isFlashOutputBrc)
{
    const uint32_t inputSize = static_cast<uint32_t>(srcShape.GetShapeSize());
    if (inputSize == 0U) { return 0U; }
    if (dataTypeSize1 == 0U || (dataTypeSize1 != SOFTMAX_HALF_SIZE && dataTypeSize1 != SOFTMAX_FLOAT_SIZE)) {
        return 0U;
    }
    if (dataTypeSize2 == 0U || (dataTypeSize2 != SOFTMAX_HALF_SIZE && dataTypeSize2 != SOFTMAX_FLOAT_SIZE)) {
        return 0U;
    }

    const std::vector<uint32_t> retVec = GetLastAxisShapeND(srcShape);
    const uint32_t srcM = retVec[0];
    const uint32_t srcK = retVec[1];
    const uint32_t elementNumPerBlk = SOFTMAX_DEFAULT_BLK_SIZE / dataTypeSize2;

    // 关键替换：使用有参 GetInstance
    platform_ascendc::PlatformAscendC* platform =
        platform_ascendc::PlatformAscendCManager::GetInstance(
            SoftmaxTilingMocker::GetInstance().GetSocVersion());
    if (platform == nullptr) { return 0U; }
    const auto npuArch = platform->GetCurNpuArch();

    uint32_t needMinSize = 0U;
    if (npuArch == NpuArch::DAV_3510 || npuArch == NpuArch::DAV_5102) {
        if (isUpdate) {
            if (dataTypeSize1 == SOFTMAX_HALF_SIZE) {
                const uint32_t size1 =
                    (srcM * SOFTMAX_TMPBUFFER_COUNT + SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_TMPBUFFER_COUNT - 1U) /
                    (SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_TMPBUFFER_COUNT) *
                    (SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_TMPBUFFER_COUNT);
                if (dataTypeSize2 == SOFTMAX_HALF_SIZE) {
                    needMinSize = size1 + srcM * (elementNumPerBlk * SOFTMAX_TMPBUFFER_COUNT + srcK);
                } else {
                    needMinSize = size1 + srcM * (elementNumPerBlk * SOFTMAX_TMPBUFFER_COUNT + srcK + elementNumPerBlk);
                }
            } else {
                needMinSize =
                    (srcM + SOFTMAX_BASICBLOCK_UNIT - 1U) / SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_BASICBLOCK_UNIT +
                    srcM * (elementNumPerBlk * SOFTMAX_TMPBUFFER_COUNT);
            }
        } else {
            const uint32_t size0 = isBasicBlock ? SOFTMAX_TMPFLASHUPDATE_COUNT : BASIC_TILE_NUM;
            const uint32_t needSize1 = srcM * (size0 + srcK) +
                SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_TMPFLASHUPDATE_COUNT +
                (srcM + BASIC_TILE_NUM - 1U) / BASIC_TILE_NUM * BASIC_TILE_NUM;
            const uint32_t needSize2 = srcM * (elementNumPerBlk + srcK);
            needMinSize = std::max(needSize1, needSize2);
        }
    } else {
        if (isBasicBlock && srcK % SOFTMAX_BASICBLOCK_UNIT == 0U) {
            if (dataTypeSize1 == SOFTMAX_HALF_SIZE) {
                needMinSize = (srcK == SOFTMAX_SPECIAL_BASICBLOCK_LEN) ?
                    BASIC_TILE_NUM * (elementNumPerBlk + srcK + SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_HALF_SIZE) :
                    BASIC_TILE_NUM * (elementNumPerBlk + srcK + SOFTMAX_BASICBLOCK_UNIT);
            } else {
                needMinSize = (srcK == SOFTMAX_SPECIAL_BASICBLOCK_LEN) ?
                    BASIC_TILE_NUM * (elementNumPerBlk + SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_HALF_SIZE) :
                    BASIC_TILE_NUM * (elementNumPerBlk + SOFTMAX_BASICBLOCK_UNIT);
            }
        } else {
            needMinSize = (dataTypeSize1 == SOFTMAX_HALF_SIZE) ?
                elementNumPerBlk * SOFTMAX_TMPBUFFER_COUNT + srcK + SOFTMAX_BASICBLOCK_UNIT :
                elementNumPerBlk + SOFTMAX_BASICBLOCK_UNIT;
        }

        if (isFlashOutputBrc) {
            const uint32_t minSrcM = std::min(SOFTMAX_DEFAULT_BLK_SIZE / dataTypeSize1, srcM);
            if (isBasicBlock && srcK % SOFTMAX_BASICBLOCK_UNIT == 0U) {
                needMinSize = (needMinSize / BASIC_TILE_NUM) * minSrcM;
            } else {
                needMinSize = needMinSize * minSrcM;
            }
        }
    }
    return needMinSize * SOFTMAX_FLOAT_SIZE;
}

// ---- 打桩：GetSoftMaxFlashMinTmpSize ------------------------------------
// 逻辑与 softmax_tiling.cpp 第259-298行一致，仅替换 GetInstance()
uint32_t GetSoftMaxFlashMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize,
    const bool isUpdate, const bool /*isReuseSource*/)
{
    const uint32_t inputSize = static_cast<uint32_t>(srcShape.GetShapeSize());
    if (inputSize == 0U) { return 0U; }
    if (dataTypeSize == 0U || (dataTypeSize != SOFTMAX_HALF_SIZE && dataTypeSize != SOFTMAX_FLOAT_SIZE)) {
        return 0U;
    }

    const std::vector<uint32_t> retVec = GetLastAxisShapeND(srcShape);
    const uint32_t elementNumPerBlk = SOFTMAX_DEFAULT_BLK_SIZE / dataTypeSize;
    const uint32_t srcM = retVec[0];
    const uint32_t srcK = retVec[1];

    platform_ascendc::PlatformAscendC* platform =
        platform_ascendc::PlatformAscendCManager::GetInstance(
            SoftmaxTilingMocker::GetInstance().GetSocVersion());
    if (platform == nullptr) { return 0U; }
    const auto npuArch = platform->GetCurNpuArch();

    uint32_t needSize;
    if (npuArch == NpuArch::DAV_3510 ||
        npuArch == NpuArch::DAV_5102) {
        const uint32_t needSize2 = srcM * (elementNumPerBlk + srcK);
        if (!isUpdate) {
            const uint32_t needSize1 = srcM * (BASIC_TILE_NUM + srcK) +
                SOFTMAX_BASICBLOCK_UNIT * SOFTMAX_FLOAT_SIZE +
                (srcM + BASIC_TILE_NUM - 1U) / BASIC_TILE_NUM * BASIC_TILE_NUM;
            needSize = std::max(needSize1, needSize2);
        } else {
            needSize = needSize2;
        }
    } else {
        needSize = !isUpdate ?
            elementNumPerBlk + srcK + SOFTMAX_BASICBLOCK_UNIT :
            elementNumPerBlk * SOFTMAX_TMPFLASHUPDATE_COUNT + srcK * SOFTMAX_TMPBUFFER_COUNT;
    }
    return needSize * SOFTMAX_FLOAT_SIZE;
}

}  // namespace AscendC
