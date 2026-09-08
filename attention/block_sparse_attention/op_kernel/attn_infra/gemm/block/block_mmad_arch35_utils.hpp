/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_MMAD_ARCH35_UTILS_HPP
#define BLOCK_MMAD_ARCH35_UTILS_HPP

namespace NpuArch::Gemm::Block {

namespace MXFP4 {
static constexpr uint32_t KB_BYTE = 1024;

// L1 分配
// pL1Tensor: 16K, 128*256*sizeof(e2m1), BUFCNT=20
static constexpr uint32_t L1_P_BUF_CNT = 20;
static constexpr uint32_t L1_P_BUF_SIZE = 16 * KB_BYTE;
static constexpr uint32_t L1_P_BUF_OFFSET = 0;

// pDescaleL1Tensor: 1.25K, 128*2*(256/64+1)*sizeof(e8m0), BUFCNT=20
static constexpr uint32_t L1_P_SCALE_BUF_CNT = 20;
static constexpr uint32_t L1_P_SCALE_BUF_SIZE = 1280;
static constexpr uint32_t L1_P_SCALE_BUF_OFFSET = 320 * KB_BYTE;

// qL1Tensor: 8K, 128*D*sizeof(e2m1), BUFCNT=2
static constexpr uint32_t L1_Q_BUF_CNT = 2;
static constexpr uint32_t L1_Q_BUF_SIZE = 8 * KB_BYTE;
static constexpr uint32_t L1_Q_BUF_OFFSET = 345 * KB_BYTE;

// qDescaleL1Tensor: 0.5K, 128*D/32*sizeof(e8m0), BUFCNT=2
static constexpr uint32_t L1_Q_DESCALE_BUF_CNT = 2;
static constexpr uint32_t L1_Q_DESCALE_BUF_SIZE = 512;
static constexpr uint32_t L1_Q_DESCALE_BUF_OFFSET = 361 * KB_BYTE;

// kvL1Tensor: 32K, 512*D*sizeof(e2m1), BUFCNT=4
// todo 先用2块，避免qk、pv同步冲突
static constexpr uint32_t L1_KV_BUF_CNT = 4;
static constexpr uint32_t L1_KV_BUF_SIZE = 32 * KB_BYTE;
static constexpr uint32_t L1_KV_BUF_OFFSET = 362 * KB_BYTE;

// kvDescaleL1Tensor: 2K, 512*D/32*sizeof(e8m0), BUFCNT=4
// todo 先用2块，避免qk、pv同步冲突
static constexpr uint32_t L1_KV_DESCALE_BUF_CNT = 4;
static constexpr uint32_t L1_KV_DESCALE_BUF_SIZE = 2 * KB_BYTE;
static constexpr uint32_t L1_KV_DESCALE_BUF_OFFSET = 490 * KB_BYTE;

// localGlobalMaxL1: 0.5K, 256*sizeof(half), BUFCNT=4
static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_CNT = 4;
static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_SIZE = 512;
static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_OFFSET = 498 * KB_BYTE;

// [QK/PV 共享 KV 池] event ID：4 路 MTE1_MTE2（MTE2 gather 写 ↔ MTE1 L1→L0 LoadData 读）。
//   QK(K) 与 PV(V) 共用同一组 ID——角色无关门不问槽装 K/V/seed，只保证「MTE1 读完→MTE2 才能覆写」。
//   与 PV 的 PV_L0AB(M_MTE1, 4/5)/PV_L0C(M_FIX, 4) 数值复用，不同 HardEvent 类型独立 flag 存储(结论#7)。
static constexpr uint32_t KV_EVENT0 = 4;
static constexpr uint32_t KV_EVENT1 = 5;
static constexpr uint32_t KV_EVENT2 = 6;
static constexpr uint32_t KV_EVENT3 = 7;

// [QK/PV 共享调试] true=打印槽位地址别名(GetPhyAddr)+游标/门轨迹；生产前改 false(if constexpr 编译期消除)。
static constexpr bool KV_SHARE_DEBUG = true;
// 每个 block 前 N 个 tile 打详细轨迹(ctor 别名打印不受此限)，避免长用例刷屏。
static constexpr uint32_t KV_TRACE_TILES = 40;

// L0 分配
static constexpr uint32_t L0A_QK_BUF_CNT = 2;
static constexpr uint32_t L0A_QK_BUF_SIZE = 8 * KB_BYTE;
static constexpr uint32_t L0A_QK_BUF_OFFSET = 0;

static constexpr uint32_t L0A_PV_BUF_CNT = 2;
static constexpr uint32_t L0A_PV_BUF_SIZE = 18 * KB_BYTE;
static constexpr uint32_t L0A_PV_BUF_OFFSET = 16 * KB_BYTE;

static constexpr uint32_t L0B_QK_BUF_CNT = 2;
static constexpr uint32_t L0B_QK_BUF_SIZE = 8 * KB_BYTE;
static constexpr uint32_t L0B_QK_BUF_OFFSET = 0;

static constexpr uint32_t L0B_PV_BUF_CNT = 2;
static constexpr uint32_t L0B_PV_BUF_SIZE = 18 * KB_BYTE;
static constexpr uint32_t L0B_PV_BUF_OFFSET = 16 * KB_BYTE;

static constexpr uint32_t L0C_QK_BUF_CNT = 2;
static constexpr uint32_t L0C_QK_BUF_SIZE = 64 * KB_BYTE;
static constexpr uint32_t L0C_QK_BUF_OFFSET = 0;

static constexpr uint32_t L0C_PV_BUF_CNT = 1;
static constexpr uint32_t L0C_PV_BUF_SIZE = 72 * KB_BYTE;
static constexpr uint32_t L0C_PV_BUF_OFFSET = 128 * KB_BYTE;

static constexpr uint32_t BLOCK_SIZE = 32;

static constexpr uint32_t V_SCALE_L0A_SIZE = (256 / 64) * ((128 + 16) / 16) * 32;

static constexpr uint32_t mBaseSize = 128; // [#9] M 方向基准 tile（对应 QFA mBaseSize），用于 pad 满块判断

static constexpr uint32_t CONST_32 = 32;
static constexpr uint32_t CONST_64 = 64;
static constexpr uint32_t CONST_128 = 128;

static constexpr uint32_t ROW_SUM_PAD_NUM = 16;
static constexpr uint32_t ROW_SUM_NUM = 128;

// 硬件 NZ 分形块维度
static constexpr uint32_t NZ_C0_ELEMS = 16;  // NZ 分形 C0 维元素个数
static constexpr uint32_t FP4_C0_ELEMS = 64; // fp4 数据类型每个 C0 块元素个数

// PV mmad M 维（含 rowsum pad）
static constexpr uint32_t PV_MMAD_M_DIM = ROW_SUM_NUM + ROW_SUM_PAD_NUM; // = 144

// S2 方向 base tile 尺寸
static constexpr uint32_t S2_BASE_TILE_SIZE = 256;

// InitL0BufferForReduceSum seed fill 模式
// V data = fp4 4.0 打包，V scale = e8m0 0.25 打包 → 有效值 4.0 × 0.25 = 1.0
static constexpr uint16_t SEED_V_DATA_FILL = 0x6666;
static constexpr uint16_t SEED_V_SCALE_FILL = 0x7D7D;
static constexpr uint16_t ZERO_FILL_PATTERN = 0x0000;

// GM SetGlobalBuffer 最大长度哨兵值
static constexpr uint32_t GM_MAX_BUFFER_LEN = 0x7FFFFFFFu;

} // namespace MXFP4

namespace MXFP8 {
static constexpr uint32_t KB_BYTE = 1024;

// L1：方案 A G=16 / Dpv=20；P 环 = Dpv = 20；单槽 16KB = 128×128×1B
// 见 docs/superpowers/specs/2026-08-25-bsa-mxfp8-tile2048-design.md
static constexpr uint32_t L1_P_BUF_CNT = 20;
static constexpr uint32_t L1_P_BUF_SIZE = 16 * KB_BYTE;
static constexpr uint32_t L1_P_BUF_OFFSET = 0;

static constexpr uint32_t L1_P_SCALE_BUF_CNT = 20;
static constexpr uint32_t L1_P_SCALE_BUF_SIZE = 768; // 128×2×(128/64+1)
static constexpr uint32_t L1_P_SCALE_BUF_OFFSET = 320 * KB_BYTE;

static constexpr uint32_t L1_Q_BUF_CNT = 2;
static constexpr uint32_t L1_Q_BUF_SIZE = 16 * KB_BYTE;
static constexpr uint32_t L1_Q_BUF_OFFSET = 335 * KB_BYTE;

static constexpr uint32_t L1_Q_DESCALE_BUF_CNT = 2;
static constexpr uint32_t L1_Q_DESCALE_BUF_SIZE = 512;
static constexpr uint32_t L1_Q_DESCALE_BUF_OFFSET = 367 * KB_BYTE;

static constexpr uint32_t L1_KV_BUF_CNT = 4;
static constexpr uint32_t L1_KV_BUF_SIZE = 32 * KB_BYTE; // ≥18KB seed 镜像
static constexpr uint32_t L1_KV_BUF_OFFSET = 368 * KB_BYTE;

static constexpr uint32_t L1_KV_DESCALE_BUF_CNT = 4;
static constexpr uint32_t L1_KV_DESCALE_BUF_SIZE = 2 * KB_BYTE;
static constexpr uint32_t L1_KV_DESCALE_BUF_OFFSET = 496 * KB_BYTE;

static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_CNT = 4;
static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_SIZE = 512;
static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_OFFSET = 504 * KB_BYTE;

static constexpr uint32_t KV_EVENT0 = 4;
static constexpr uint32_t KV_EVENT1 = 5;
static constexpr uint32_t KV_EVENT2 = 6;
static constexpr uint32_t KV_EVENT3 = 7;

static constexpr bool KV_SHARE_DEBUG = false;
static constexpr uint32_t KV_TRACE_TILES = 40;

// L0：QK 一拍 128×128。K 单缓 16KB（2×16 会与 Vᵀ 2×18 合计 68>64）；
// L0C QK 2×64KB + PV 72KB = 200KB < 256KB。Q 仍单缓 16KB（不绑双槽）。
static constexpr uint32_t L0A_QK_BUF_CNT = 1;
static constexpr uint32_t L0A_QK_BUF_SIZE = 16 * KB_BYTE; // 128×128×1B K
static constexpr uint32_t L0A_QK_BUF_OFFSET = 0;

static constexpr uint32_t L0A_PV_BUF_CNT = 2;
static constexpr uint32_t L0A_PV_BUF_SIZE = 18 * KB_BYTE; // 144×128×1B Vᵀ
static constexpr uint32_t L0A_PV_BUF_OFFSET = 16 * KB_BYTE;

static constexpr uint32_t L0B_QK_BUF_CNT = 1;
static constexpr uint32_t L0B_QK_BUF_SIZE = 16 * KB_BYTE; // 128×128×1B Q（单缓冲）
static constexpr uint32_t L0B_QK_BUF_OFFSET = 0;

static constexpr uint32_t L0B_PV_BUF_CNT = 2;
static constexpr uint32_t L0B_PV_BUF_SIZE = 16 * KB_BYTE; // 128×128×1B Pᵀ
static constexpr uint32_t L0B_PV_BUF_OFFSET = 16 * KB_BYTE;

static constexpr uint32_t L0C_QK_BUF_CNT = 2;
static constexpr uint32_t L0C_QK_BUF_SIZE = 64 * KB_BYTE; // 128×128×4B
static constexpr uint32_t L0C_QK_BUF_OFFSET = 0;

static constexpr uint32_t L0C_PV_BUF_CNT = 1;
static constexpr uint32_t L0C_PV_BUF_SIZE = 72 * KB_BYTE;
static constexpr uint32_t L0C_PV_BUF_OFFSET = 128 * KB_BYTE;

static constexpr uint32_t BLOCK_SIZE = 32;

static constexpr uint32_t V_SCALE_L0A_SIZE = (128 / 64) * ((128 + 16) / 16) * 32;

static constexpr uint32_t mBaseSize = 128;

static constexpr uint32_t CONST_32 = 32;
static constexpr uint32_t CONST_64 = 64;
static constexpr uint32_t CONST_128 = 128;

static constexpr uint32_t ROW_SUM_PAD_NUM = 16;
static constexpr uint32_t ROW_SUM_NUM = 128;

static constexpr uint32_t NZ_C0_ELEMS = 16;
static constexpr uint32_t FP8_C0_ELEMS = 32;
static constexpr uint32_t MX_GROUP_ELEMS = 32;

static constexpr uint32_t PV_MMAD_M_DIM = ROW_SUM_NUM + ROW_SUM_PAD_NUM; // 144
static constexpr uint32_t S2_BASE_TILE_SIZE = 128;
static constexpr uint32_t QK_HALF_KVS = 128; // 一拍 QK 打满基块 kvs 128

static constexpr uint16_t SEED_V_DATA_FILL = 0x3838;
static constexpr uint16_t SEED_V_SCALE_FILL = 0x7F7F;
static constexpr uint16_t ZERO_FILL_PATTERN = 0x0000;

static constexpr uint32_t GM_MAX_BUFFER_LEN = 0x7FFFFFFFu;

// P1：偶 tile 一次搬 256 行填满 32KB L1 槽，奇 tile 复用。softmax / P 环仍 128。
#ifndef BSA_MXFP8_K_PAIR256
#define BSA_MXFP8_K_PAIR256 1
#endif

// 等长非连续两块：一条 Nd2Nz/Dn2Nz（ndNum/dnNum=2 + 恒定 stride）。
#ifndef BSA_MXFP8_NDNUM2
#define BSA_MXFP8_NDNUM2 1
#endif

// 索引去阻塞：叠在 P1 上 S=118800 scalar/aic 涨 ~0.9 ms，保持 0。
#ifndef BSA_MXFP8_IDX_PREFETCH
#define BSA_MXFP8_IDX_PREFETCH 0
#endif

// P1b：K/V scale 每 4 个 tile 搬一次。整包实测 S≥4096 死锁，保持 0。
#ifndef BSA_MXFP8_SCALE_BATCH4
#define BSA_MXFP8_SCALE_BATCH4 0
#endif

static constexpr uint32_t SCALE_BATCH_TILES = 4;

// 探针：K-scale copy 后再发一条（写同一 L1 槽）。测完必须改回 0。
#ifndef BSA_MXFP8_MTE2_INSTR_PROBE
#define BSA_MXFP8_MTE2_INSTR_PROBE 0
#endif

__aicore__ inline uint32_t Mxfp8Align16(uint32_t x)
{
    return (x + 15u) / 16u * 16u;
}

__aicore__ inline uint32_t Mxfp8Align64(uint32_t x)
{
    return (x + 63u) / 64u * 64u;
}

// 同 task 内下一 KV tile 的实际行数；已是最后一拍则 0。
__aicore__ inline uint32_t Mxfp8NextKvsTileRows(uint32_t gatheredKvSTileIdx, uint32_t kvSBaseTile,
                                                uint64_t gatheredKvSeqlen)
{
    uint64_t nextStart = (static_cast<uint64_t>(gatheredKvSTileIdx) + 1u) * static_cast<uint64_t>(kvSBaseTile);
    if (nextStart >= gatheredKvSeqlen) {
        return 0;
    }
    uint64_t rem = gatheredKvSeqlen - nextStart;
    return (rem < kvSBaseTile) ? static_cast<uint32_t>(rem) : kvSBaseTile;
}

// 下一次真正发 MTE2 的 tile 下标。P1 奇 tile 跳过 gather，故步进 2。
__aicore__ inline uint32_t Mxfp8NextGatherTileIdx(uint32_t gatheredKvSTileIdx)
{
#if BSA_MXFP8_K_PAIR256
    return gatheredKvSTileIdx + 2u;
#else
    return gatheredKvSTileIdx + 1u;
#endif
}

// 从 gatheredKvSTileIdx 起连续 nTiles 个基块在 gather 流上的行数（尾部截断）。
__aicore__ inline uint32_t Mxfp8GatheredTilesRows(uint32_t gatheredKvSTileIdx, uint32_t nTiles, uint32_t kvSBaseTile,
                                                  uint64_t gatheredKvSeqlen)
{
    uint64_t start = static_cast<uint64_t>(gatheredKvSTileIdx) * static_cast<uint64_t>(kvSBaseTile);
    if (start >= gatheredKvSeqlen) {
        return 0;
    }
    uint64_t rem = gatheredKvSeqlen - start;
    uint64_t cap = static_cast<uint64_t>(nTiles) * static_cast<uint64_t>(kvSBaseTile);
    uint64_t rows = (rem < cap) ? rem : cap;
    return static_cast<uint32_t>(rows);
}

__aicore__ inline bool Mxfp8U32Delta(int64_t delta, uint32_t &out)
{
    if (delta <= 0 || delta > static_cast<int64_t>(0xFFFFFFFFu)) {
        return false;
    }
    out = static_cast<uint32_t>(delta);
    return true;
}

static_assert(SCALE_BATCH_TILES * S2_BASE_TILE_SIZE * (mBaseSize / MX_GROUP_ELEMS) <= L1_KV_DESCALE_BUF_SIZE,
              "P1b 512-row scale must fit in 2KB KV descale slot (D=128)");

static_assert(L1_P_SCALE_BUF_OFFSET + L1_P_SCALE_BUF_CNT * L1_P_SCALE_BUF_SIZE == L1_Q_BUF_OFFSET,
              "MXFP8 L1 Q must start at P-scale end (320KB+20*768)");
static_assert(L1_LOCAL_GLOBAL_MAX_BUF_OFFSET + L1_LOCAL_GLOBAL_MAX_BUF_CNT * L1_LOCAL_GLOBAL_MAX_BUF_SIZE <=
                  512 * KB_BYTE,
              "MXFP8 L1 exceeds 512KB");
static_assert(L0A_QK_BUF_OFFSET + L0A_QK_BUF_CNT * L0A_QK_BUF_SIZE == L0A_PV_BUF_OFFSET,
              "MXFP8 L0A QK must end at Vᵀ offset");
static_assert(L0A_PV_BUF_OFFSET + L0A_PV_BUF_CNT * L0A_PV_BUF_SIZE <= 64 * KB_BYTE, "MXFP8 L0A exceeds 64KB");
static_assert(L0B_PV_BUF_OFFSET + L0B_PV_BUF_CNT * L0B_PV_BUF_SIZE <= 64 * KB_BYTE, "MXFP8 L0B exceeds 64KB");
static_assert(L0C_QK_BUF_OFFSET + L0C_QK_BUF_CNT * L0C_QK_BUF_SIZE == L0C_PV_BUF_OFFSET,
              "MXFP8 L0C QK must end at PV offset");
static_assert(L0C_PV_BUF_OFFSET + L0C_PV_BUF_CNT * L0C_PV_BUF_SIZE <= 256 * KB_BYTE, "MXFP8 L0C exceeds 256KB");
} // namespace MXFP8

} // namespace NpuArch::Gemm::Block

#endif // BLOCK_MMAD_ARCH35_UTILS_HPP
