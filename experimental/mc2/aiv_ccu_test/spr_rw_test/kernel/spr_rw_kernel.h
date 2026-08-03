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
 * \file spr_rw_kernel.h
 * \brief AIV 访问 SPR (DATA_MAIN_BASE) 寄存器 Kernel 核心逻辑
 *
 * 实现 background.md 描述的 SPR→vaddr 映射访问：
 *   - 本核通过 cce 指令 get_data_main_base() 读取自身 SPR
 *   - 他核通过 st_dev 对 spr vaddr 写入
 *   - 本 kernel 由 AIV 执行，验证本核/他核 SPR 访问能力
 */

#ifndef SPR_RW_KERNEL_H
#define SPR_RW_KERNEL_H

#include "kernel_operator.h"

namespace SprRwImpl {

using namespace AscendC;

// SPR 寄存器偏移（background.md 2.3: DATA_MAIN_BASE 需要 +0x00D0）
constexpr uint64_t SPR_DATA_MAIN_BASE_OFFSET = 0x00D0;
// 解锁寄存器偏移（background.md 2.4: 读前需写 0x19101920 到 +0x4078）
constexpr uint64_t SPR_UNLOCK_OFFSET = 0x4078;
constexpr uint32_t SPR_UNLOCK_VAL = 0x19101920;
// 已知不可写的核 id（background.md 2.4）
// +0写：11, 17, 56, 68 起不来；aic: 11, 17, 20 写不了
// 本实验只在本核自访问，规避他核写限制

// 写入 SPR 的测试值
constexpr uint32_t SPR_TEST_WRITE_VAL = 0x12345678;

// 结果回写布局 (HBM, uint64_t 单位):
//   result[0] = sprVaddr (host 传入，回显)
//   result[1] = readValBeforeWrite  (写前读值)
//   result[2] = readValAfterWrite   (写后读值)
//   result[3] = verifyResult        (0=通过, 1=失败, 2=skip)
//   result[4] = aivCoreId
constexpr uint32_t RESULT_WORDS = 5;

template <typename T>
class SprRw {
public:
    __aicore__ inline SprRw(){};
    __aicore__ inline void Init(uint64_t sprVaddr, GM_ADDR sprVaddrTable, GM_ADDR resultAddr,
                                uint32_t targetCoreId, uint32_t blockDim);
    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteThenReadSelf();
    __aicore__ inline void WriteThenReadPeer();

    uint64_t sprVaddr_;            // 目标核 SPR vaddr (本核或他核)
    GM_ADDR sprVaddrTable_;        // host 传入的所有核 SPR vaddr 表 (GM)
    GlobalTensor<uint64_t> resultGm_;
    uint32_t targetCoreId_;        // 要访问的目标核 id
    uint32_t blockDim_;
    int64_t aivId_;                // 本核 coreId
};

template <typename T>
__aicore__ inline void SprRw<T>::Init(uint64_t sprVaddr, GM_ADDR sprVaddrTable, GM_ADDR resultAddr,
                                      uint32_t targetCoreId, uint32_t blockDim)
{
    sprVaddr_ = sprVaddr;
    sprVaddrTable_ = sprVaddrTable;
    targetCoreId_ = targetCoreId;
    blockDim_ = blockDim;
    aivId_ = GetBlockIdx();
    resultGm_.SetGlobalBuffer((__gm__ uint64_t *)resultAddr);
}

// 本核访问自身 SPR: 通过 vaddr 读写同一寄存器
template <typename T>
__aicore__ inline void SprRw<T>::WriteThenReadSelf()
{
    __gm__ uint8_t *selfSpr = reinterpret_cast<__gm__ uint8_t *>(
        reinterpret_cast<__gm__ uint64_t *>(sprVaddrTable_)[aivId_]);
    __gm__ uint32_t *sprReg = reinterpret_cast<__gm__ uint32_t *>(selfSpr + SPR_DATA_MAIN_BASE_OFFSET);

    // 1. 读取写前值 (通过 GM 读同一寄存器地址)
    uint32_t readBefore = *sprReg;

    // 2. 解锁寄存器 (background.md 2.4)
    st_dev(SPR_UNLOCK_VAL, (__gm__ uint32_t *)(selfSpr + SPR_UNLOCK_OFFSET), 0);

    // 3. CrossCore 同步 (background.md 2.4: 结合 st_dev + CrossCore 提高成功率)
    CrossCoreSetFlag<0, PIPE_ALL>(11);
    CrossCoreWaitFlag<0, PIPE_ALL>(11);

    // 4. 写 SPR (st_dev 写 +0x00D0)
    st_dev(SPR_TEST_WRITE_VAL, sprReg, 0);

    // 5. 读取写后值 (同一寄存器地址)
    uint32_t readAfter = *sprReg;

    // 6. 回写结果: 比较写后读值与写入值是否一致
    resultGm_.SetValue(0, sprVaddr_);
    resultGm_.SetValue(1, static_cast<uint64_t>(readBefore));
    resultGm_.SetValue(2, static_cast<uint64_t>(readAfter));
    resultGm_.SetValue(3, (readAfter == SPR_TEST_WRITE_VAL) ? 0 : 1);
    resultGm_.SetValue(4, static_cast<uint64_t>(aivId_));

    DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                             AscendC::DcciDst::CACHELINE_OUT>(resultGm_);
}

// 访问他核 SPR: 通过 st_dev 写他核 spr vaddr, 通过自身 get_data_main_base() 不可用
// 他核 SPR 值需要他核自己读并回写，本实验简化为只做写操作
template <typename T>
__aicore__ inline void SprRw<T>::WriteThenReadPeer()
{
    __gm__ uint64_t *vaddrTable = reinterpret_cast<__gm__ uint64_t *>(sprVaddrTable_);
    __gm__ uint8_t *peerSpr = reinterpret_cast<__gm__ uint8_t *>(vaddrTable[targetCoreId_]);

    printf("[spr_rw] aiv %ld: write peer core %u, sprVaddr=0x%lx\n", aivId_, targetCoreId_,
           reinterpret_cast<uint64_t>(peerSpr));

    // 解锁
    st_dev(SPR_UNLOCK_VAL, (__gm__ uint32_t *)(peerSpr + SPR_UNLOCK_OFFSET), 0);
    CrossCoreSetFlag<0, PIPE_ALL>(11);
    CrossCoreWaitFlag<0, PIPE_ALL>(11);

    // 写他核 SPR
    st_dev(SPR_TEST_WRITE_VAL, (__gm__ uint32_t *)(peerSpr + SPR_DATA_MAIN_BASE_OFFSET), 0);

    // 本核无法直接读他核 SPR 值（get_data_main_base 只读本核）
    // 回写状态：仅完成写操作
    resultGm_.SetValue(0, reinterpret_cast<uint64_t>(peerSpr));
    resultGm_.SetValue(1, 0);  // 写前读: N/A (他核)
    resultGm_.SetValue(2, 0);  // 写后读: N/A (他核)
    resultGm_.SetValue(3, 2);  // skip (需他核配合读)
    resultGm_.SetValue(4, static_cast<uint64_t>(aivId_));

    DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                             AscendC::DcciDst::CACHELINE_OUT>(resultGm_);
}

template <typename T>
__aicore__ inline void SprRw<T>::Process()
{
    // 仅 aivId_ == 0 的核执行，避免多核重复写
    if (aivId_ != 0) {
        return;
    }

    printf("[spr_rw] kernel start, aivId=%ld, targetCoreId=%u, sprVaddr=0x%lx\n",
           aivId_, targetCoreId_, sprVaddr_);

    // 判断目标是本核还是他核
    // background.md 中 coreId 与 blockId 不一定对应，这里用 aivId_ == targetCoreId_ 近似
    if (static_cast<uint32_t>(aivId_) == targetCoreId_) {
        printf("[spr_rw] access self SPR (core %ld)\n", aivId_);
        WriteThenReadSelf();
    } else {
        printf("[spr_rw] access peer SPR (target core %u)\n", targetCoreId_);
        WriteThenReadPeer();
    }

    printf("[spr_rw] kernel end, aivId=%ld\n", aivId_);
}

} // namespace SprRwImpl

#endif // SPR_RW_KERNEL_H
