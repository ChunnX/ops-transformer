/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
/**
 * 全部行无效(blockEffRows=0 / validGroups==0)专用模板 VF（合并版）：
 * 直通指令流 + 运行时限界循环（无运行时 if），把整块/整 tile 处理为：
 *   - pDest(P) 填 0
 *   - local_group_max 填 MIN_VALUE（使下游 computePscale 得到 scale=0/有限）
 *   - global_max 置极负有限值 -60000（对齐 golden 全 -inf block 的 max safe 处理，
 *     避免下游读到未初始化 buffer -> scale=255(+inf) -> P_eff=0*inf=NaN）
 * 调用侧(epilogue/wrapper, __aicore__ 层)按 chunk 有效行数分发到本模板。
 * 模板参数：T=half, QsBase=128/64, GROUPS=每 chunk 32 行组数(4 或 1), CHUNKS=tile 内 chunk 数。
 */
#ifndef VF_SOFTMAX_ALL_INVALID_H_
#define VF_SOFTMAX_ALL_INVALID_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace NpuArch::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

// 通用 chunk 全0 模板：pDest 清 0 + local_group_max 填 MIN_VALUE + global_max 置 -60000
// HAS_HIGH_OFF: qs128 布局有高位远端(16256/16384), qs64 无; local_group_max 每组固定 128 元素
// LGM_GROUPS: local_group_max 覆盖组数(0=默认 CHUNKS*GROUPS)。kvs32_single(尾<=32)场景原主 VF 会
//   额外写 padding 组(GROUP_COUNT*QsBase 偏移, "GroupMaxpadding到64 使 pscale=0"), 故传 2;
//   P 清零仍只按 GROUPS 组写, 避免越界。
template <typename T, bool HAS_HIGH_OFF = true, uint16_t GROUPS = 4, uint16_t CHUNKS = 2,
          uint16_t LGM_GROUPS = 0>
__simd_vf__ inline void softmax_all_invalid_chunk_vf(__ubuf__ uint8_t *p_dest, __ubuf__ T *local_group_max,
                                                     __ubuf__ T *global_max)
{
    constexpr uint16_t P_GROUP_STRIDE = 2048;  // 每 32 行组 P 字节数(qs128/qs64 均为 2048)
    constexpr uint16_t P_HIGH_OFF = 16256;     // qs128 高位 128 字节远端地址
    constexpr uint16_t P_HIGH_STEP = 16384;    // qs128 高位步进
    constexpr uint16_t LGM_GROUP = 128;        // 每 32 行组 local_group_max 元素数(128/64 布局均 128)

    MaskReg preg_all_16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all_8 = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_vl128 = CreateMask<uint8_t, MaskPattern::VL128>();
    MaskReg preg_vl128_not;
    MaskNot(preg_vl128_not, preg_vl128, preg_all_8);

    RegTensor<uint8_t> zero8;
    RegTensor<T> min_reg, neg_reg;
    Duplicate(zero8, static_cast<uint8_t>(0));
    Duplicate(min_reg, MIN_VALUE);
    Duplicate(neg_reg, static_cast<T>(-60000));

    // pDest 全 0（组循环 + 高低位写，与主 VF padding 分支一致）
    for (uint16_t c = 0; c < CHUNKS; ++c) {
        const uint16_t groupBase = c * GROUPS;
        for (uint16_t g = 0; g < GROUPS; ++g) {
            for (uint16_t j = 0; j < 8; j += 2) {
                StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * P_GROUP_STRIDE + j * 256, zero8,
                           preg_vl128);
                if constexpr (HAS_HIGH_OFF) {
                    StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * P_GROUP_STRIDE + j * 256 + P_HIGH_OFF,
                               zero8, preg_vl128_not);
                }
                StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * P_GROUP_STRIDE + j * 256 + 128, zero8,
                           preg_vl128);
                if constexpr (HAS_HIGH_OFF) {
                    StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * P_GROUP_STRIDE + j * 256 + P_HIGH_STEP,
                               zero8, preg_vl128_not);
                }
            }
        }
    }
    // local_group_max 填 MIN_VALUE（LGM_GROUPS 组 × LGM_GROUP; 默认 CHUNKS*GROUPS 组）
    constexpr uint16_t lgmGroups = (LGM_GROUPS == 0) ? CHUNKS * GROUPS : LGM_GROUPS;
    for (uint16_t i = 0; i < lgmGroups * LGM_GROUP; i += 128) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + i, min_reg, preg_all_16);
    }
    // global_max 置极负有限值（对齐 golden 全 -inf block 的 max safe 处理）
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(global_max, neg_reg, preg_all_16);
}

template <typename T, bool HAS_HIGH_OFF = true, uint16_t GROUPS = 4, uint16_t CHUNKS = 2,
          uint16_t LGM_GROUPS = 0>
__aicore__ inline void SoftmaxAllInvalidCallVF(const LocalTensor<uint8_t> &pDest,
                                               const LocalTensor<T> &localGroupMax,
                                               const LocalTensor<T> &localGlobalMax)
{
    softmax_all_invalid_chunk_vf<T, HAS_HIGH_OFF, GROUPS, CHUNKS, LGM_GROUPS>(
        (__ubuf__ uint8_t *)pDest.GetPhyAddr(), (__ubuf__ T *)localGroupMax.GetPhyAddr(),
        (__ubuf__ T *)localGlobalMax.GetPhyAddr());
}

} // namespace NpuArch::Epilogue::Block::Mxfp4VF
#endif // VF_SOFTMAX_ALL_INVALID_H_
