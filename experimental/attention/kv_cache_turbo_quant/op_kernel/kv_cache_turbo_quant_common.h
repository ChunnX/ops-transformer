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
 * \file kv_cache_turbo_quant_common.h
 * \brief encode kernel 公共常量与 Lloyd-Max 质心表（与 golden.py 的 2/3/4bit 质心一致）。
 */
#ifndef KV_CACHE_TURBO_QUANT_COMMON_H
#define KV_CACHE_TURBO_QUANT_COMMON_H

namespace KvCacheTurboQuant {

constexpr int32_t HEAD_DIM = 128;
constexpr int32_t QJL_BYTES = 16; // qjl_dim=128, 1 bit/channel

// Lloyd-Max 质心（单位球面上逐维高斯的各 bit 档最优量化器，来源同任务方 golden.py）
template <int BITS>
struct CentTable;

template <>
struct CentTable<2> {
    static constexpr float CENT[4] = {-0.1335033178f, -0.04002048075f, 0.04002048075f, 0.1335033178f};
};

template <>
struct CentTable<3> {
    static constexpr float CENT[8] = {-0.19020693f,   -0.1187859178f, -0.06682205945f, -0.02166347019f,
                                      0.02166347019f, 0.06682205945f, 0.1187859178f,   0.19020693f};
};

template <>
struct CentTable<4> {
    static constexpr float CENT[16] = {-0.2414890379f,  -0.1828317791f,  -0.1429702938f,  -0.1109927073f,
                                       -0.08325428516f, -0.05802082643f, -0.03428063914f, -0.01134236995f,
                                       0.01134236995f,  0.03428063914f,  0.05802082643f,  0.08325428516f,
                                       0.1109927073f,   0.1429702938f,   0.1828317791f,   0.2414890379f};
};

} // namespace KvCacheTurboQuant

#endif // KV_CACHE_TURBO_QUANT_COMMON_H
