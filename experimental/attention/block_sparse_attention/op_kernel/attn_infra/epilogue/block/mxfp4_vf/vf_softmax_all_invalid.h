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

// 通用 chunk 全0 模板：pDest 清 0 + local_group_max 填 MIN_VALUE + global_max 置极负
// HAS_HIGH_OFF: qs128 布局有高位远端(16256/16384), qs64 无; local_group_max 每组固定 128 元素
// LGM_GROUPS: local_group_max 覆盖组数(0=默认 CHUNKS*GROUPS)。kvs32_single(尾<=32)场景原主 VF 会
//   额外写 padding 组(GROUP_COUNT*QsBase 偏移, "GroupMaxpadding到64 使 pscale=0"), 故传 2;
//   P 清零仍只按 GROUPS 组写, 避免越界。
// CHUNK_START: 起始 chunk 偏移(0=chunk0, 1=chunk1), 用于 kvs256 部分 chunk 无效时只清指定 chunk 的
//   P/LGM 区域(global_max 不参与, 见下方说明)。
// 【重要】本模板**不写 global_max**。
//   global_max 是跨 tile 累积的 running max, 只应由有效数据推进。全无效的 tile 没有任何有效数据,
//   对 running max 没有贡献, 所以必须"不动它"——一旦在这里把它覆盖成一个很小的占位值(如 -60000),
//   后续 compute_pscale 算 mxscale = SAT(lm_i - (gmax - 125)) 时, 先前有效 tile 的组(lm 约 100~150)
//   与该占位值相差 6 万 -> 饱和到 255(OCP E8M0 的 255 即 NaN) -> P_eff = P * NaN = NaN -> 输出全 NaN。
//   不写则: 有效组照常; 无效组 lm 被填成 MIN_VALUE(-65504), 相对正常 gmax 会饱和到 0
//   (scale = 2^-127) -> P_eff = 0, 正好是「无效块贡献 0」的语义。
//   global_max 的安全初值由 InitGlobalMaxCallVF 在「任务首 tile / TileGroup 首 tile」写一次。
template <typename T, bool HAS_HIGH_OFF = true, uint16_t GROUPS = 4, uint16_t CHUNKS = 2,
          uint16_t LGM_GROUPS = 0, uint16_t CHUNK_START = 0>
__simd_vf__ inline void softmax_all_invalid_chunk_vf(__ubuf__ uint8_t *p_dest, __ubuf__ T *local_group_max)
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
    RegTensor<T> min_reg;
    Duplicate(zero8, static_cast<uint8_t>(0));
    Duplicate(min_reg, MIN_VALUE);

    // pDest 全 0（组循环 + 高低位写，与主 VF padding 分支一致）
    for (uint16_t c = CHUNK_START; c < CHUNK_START + CHUNKS; ++c) {
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
    constexpr uint16_t lgmBase = CHUNK_START * GROUPS * LGM_GROUP;
    for (uint16_t i = lgmBase; i < lgmBase + lgmGroups * LGM_GROUP; i += 128) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + i, min_reg, preg_all_16);
    }
    // 注意: 这里**不写** global_max —— 见本函数上方注释
}

template <typename T, bool HAS_HIGH_OFF = true, uint16_t GROUPS = 4, uint16_t CHUNKS = 2,
          uint16_t LGM_GROUPS = 0, uint16_t CHUNK_START = 0>
__aicore__ inline void SoftmaxAllInvalidCallVF(const LocalTensor<uint8_t> &pDest,
                                               const LocalTensor<T> &localGroupMax)
{
    softmax_all_invalid_chunk_vf<T, HAS_HIGH_OFF, GROUPS, CHUNKS, LGM_GROUPS, CHUNK_START>(
        (__ubuf__ uint8_t *)pDest.GetPhyAddr(), (__ubuf__ T *)localGroupMax.GetPhyAddr());
}
// global_max 安全初值: 只在「任务首 tile」或「TileGroup 首 tile」写一次。
// 这两个时机 tileMaxIdx_ 会切到新的四缓冲槽位(kernel:704-708), 槽位里是 4 个 TileGroup 之前的
// 残留/未初始化值; 不落初值会读到垃圾(典型 MIN_VALUE=-65504) -> gmax-125 溢出 -> pscale 饱和 255 -> NaN。
// 必须是 -60000 而非 MIN_VALUE(-65504): -65504 - 125 = -65629 超 fp16 有限值下限变 -inf。
// -60000 留出 ~5379 余量, 使无效组(lm=-65504)的 mxscale 落到 0(scale=2^-127) -> P_eff = 0。
template <typename T>
__simd_vf__ inline void init_global_max_vf(__ubuf__ T *global_max)
{
    MaskReg preg_all_16 = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<T> neg_reg;
    Duplicate(neg_reg, static_cast<T>(-60000));
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(global_max, neg_reg, preg_all_16);
}

template <typename T>
__aicore__ inline void InitGlobalMaxCallVF(const LocalTensor<T> &localGlobalMax)
{
    init_global_max_vf<T>((__ubuf__ T *)localGlobalMax.GetPhyAddr());
}


} // namespace NpuArch::Epilogue::Block::Mxfp4VF
#endif // VF_SOFTMAX_ALL_INVALID_H_
