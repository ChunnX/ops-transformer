/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_imp_base.h
 * \brief
 */

 #ifndef MEGA_MOE_IMP_BASE_H
 #define MEGA_MOE_IMP_BASE_H
 
 namespace MegaMoeImpl {
 constexpr uint32_t ALIGN32 = 32U;
 constexpr uint32_t L1_TILE_M_256 = 256;
 constexpr uint32_t L1_TILE_M_128 = 128;
 constexpr uint32_t L1_TILE_N = 256;
 constexpr uint32_t L1_TILE_K = 256;
 constexpr uint32_t L0_TILE_K = 128;
 constexpr uint32_t SCALE_K_L1_RATE = 2;
 constexpr uint32_t SWIGLU_N_HALF = 2;
 constexpr uint32_t MAX_SINGLE_MN_256_256 = 256 * 256;
 constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_256 = (MAX_SINGLE_MN_256_256 + 31U) / ALIGN32 * ALIGN32;
 constexpr uint32_t MAX_SINGLE_MN_128_256 = 128 * 256;
 constexpr uint32_t MAX_SINGLE_MN_ALIGN32_NUM_128 = (MAX_SINGLE_MN_128_256 + 31U) / ALIGN32 * ALIGN32;
 constexpr uint32_t META_INFO_TENSOR_ADDR = 200U * 1024U; // metaInfo tensor 在 UB 中的起始地址
 
 // Combine "data as flag"：AIV 侧 [数据|flag] 拼接 buffer 的 UB 起始地址。
 //   l0cOutUbGMM2 双 buffer 占 [0,128K)，META_INFO_TENSOR_ADDR 在 200K，故 [128K,200K) 可用。
 //   布局常量的唯一真值源在 mega_moe_workspace_info.h（host tiling 也要用）。
 //   注意：本文件被 mega_moe_workspace_info.h 提前 include，且会被 host tiling 编译，
 //         故这里只能放 constexpr，不能放带 __aicore__ 的函数，也不能引用 workspace_info.h 的常量。
 constexpr uint32_t COMBINE_SEND_BUF_ADDR = 128U * 1024U;
 // 拼接区可用字节数 = [COMBINE_SEND_BUF_ADDR, META_INFO_TENSOR_ADDR)，当前为 72KB。
 constexpr uint32_t COMBINE_SEND_BUF_BYTES = META_INFO_TENSOR_ADDR - COMBINE_SEND_BUF_ADDR;
 // CombineTokens 单批行数上限。必须分批的两个原因（缺一不可，改动前踩过）：
 //   1) GMM2 的 tileM 并非恒为 128：A8W4/A4W4 路径在 BuildProblemConfig 里无条件取 L1_TILE_M_256，
 //      故 lenTile 上界是 256；而拼接区只有 72KB，按 544B/行 只装得下 135 行 —— 不分批会踩穿
 //      META_INFO_TENSOR_ADDR 上的 metaInfoTensor（三元组被自己的数据覆盖 → 写错卡/非法地址）。
 //   2) Copy 的 repeatTimes 形参是 uint8_t，lenTile==256 会截断成 0 → 数据一个字节都不搬，
 //      但块尾 flag 仍合法发出 → 接收侧判"已到达"却拿到垃圾，静默算错。
 // 取 128 使 128*544=69632 ≤ 72KB 且 ≤255，两个约束同时满足；实际上界由 CombineTokens 的
 // static_assert 编译期兜底，改这里若超预算会直接编译报错而非上板才炸。
 constexpr uint32_t COMBINE_SEND_MAX_ROWS = L1_TILE_M_128;
 } // namespace MegaMoeImpl
 #endif // MEGA_MOE_IMP_BASE_H