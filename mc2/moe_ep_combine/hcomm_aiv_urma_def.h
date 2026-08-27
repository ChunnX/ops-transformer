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
 * \file hcomm_aiv_urma_def.h
 * \brief Hcomm AIV URMA definition for V310
 */

#if !defined(HCOMM_INCLUDE_INTERNAL_HEADERS)
#pragma message("This is an internal Hcomm header. Please include hcomm/hcomm.h instead.")
#define HCOMM_INCLUDE_INTERNAL_HEADERS
#define HCOMM_UNDEF_INCLUDE_INTERNAL_HEADERS_HCOMM_AIV_URMA_DEF_H
#endif

#ifndef IMPL_ADV_API_DETAIL_HCOMM_IMPL_PLATFORM_V310_HCOMM_AIV_URMA_DEF_H
#define IMPL_ADV_API_DETAIL_HCOMM_IMPL_PLATFORM_V310_HCOMM_AIV_URMA_DEF_H

#include "../../common/hcomm_inner_def.h"

namespace AscendC {

constexpr uint32_t HCOMM_URMA_MAX_RETRY_TIMES = 1000000;
constexpr uint32_t HCOMM_URMA_TMP_BUF_SIZE = 512;
constexpr uint32_t HCOMM_URMA_WQE_U32_NUM = 32;
constexpr uint32_t HCOMM_URMA_CQE_U32_NUM = 16;
constexpr uint32_t HCOMM_URMA_UDF_FLAG = 0x80U;
constexpr uint32_t POLL_CQ_THRESHOLD = 10;
constexpr uint32_t NUM_CQE_PER_POLL_CQ = 100;
#if defined(UT_TEST)
constexpr uint32_t HCOMM_URMA_MUTEX_ID = 27U;
#else
constexpr uint32_t HCOMM_URMA_MUTEX_ID = 29U;
#endif
constexpr uint32_t HCOMM_URMA_WQEBB_SIZE = 64U;
constexpr uint32_t HCOMM_URMA_WQEBB_U32_NUM = HCOMM_URMA_WQEBB_SIZE / sizeof(uint32_t);
constexpr uint32_t HCOMM_URMA_WRITE_WITH_NOTIFY_WQEBB_NUM = 2U;

constexpr uint32_t HCOMM_URMA_INVALID_REDUCE_DATA_TYPE = 0xFFFFFFFFU;
template <typename T>
constexpr uint32_t HCOMM_URMA_REDUCE_DATA_TYPE =
    std::is_same<T, int8_t>::value     ? 0x0U :
    std::is_same<T, int16_t>::value    ? 0x1U :
    std::is_same<T, int32_t>::value    ? 0x2U :
    std::is_same<T, uint32_t>::value   ? 0x5U :
    std::is_same<T, half>::value       ? 0x6U :
    std::is_same<T, float>::value      ? 0x7U :
    std::is_same<T, bfloat16_t>::value ? 0x8U :
                                         HCOMM_URMA_INVALID_REDUCE_DATA_TYPE;

template <typename T>
struct UdmaParams {
    T value;
    T cond;
    uint32_t reduceDataType;
    uint32_t reduceOpcode;
};

template <>
class HcommImpl<COMM_PROTOCOL_UBC_CTP> {
public:
    __aicore__ inline HcommImpl();
    __aicore__ inline ~HcommImpl();
    __aicore__ inline int32_t Init(__ubuf__ uint8_t *buff, uint32_t len);
    template <typename T>
    __aicore__ inline int32_t Init(const LocalTensor<T> &buff, uint32_t len);
    template <typename U>
    __aicore__ inline UbcCtpBatchHandle MakeBatchHandle(ChannelHandle channel, const LocalTensor<U> &buff,
                                                        uint32_t buffLen, GM_ADDR remoteAddr, GM_ADDR localAddr);
    template <bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len);
    template <auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteNbi(UbcCtpBatchHandle &batchHandle, GM_ADDR dst, GM_ADDR src, uint32_t len);
    template <typename T, HcommUrmaReduceOp reduceOp, bool commit = true, pipe_t commitPipe = PIPE_S,
              pipe_t reqPipe = PIPE_MTE3, auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteReduceNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t count);
    template <bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t ReadNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len);
    template <auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t ReadNbi(UbcCtpBatchHandle &batchHandle, GM_ADDR dst, GM_ADDR src, uint32_t len);
    template <typename T, bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_INLINE_CFG>
    __aicore__ inline int32_t WriteValueNbi(ChannelHandle channel, GM_ADDR dst, T value);
    template <bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteWithNotifyNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len,
                                                 GM_ADDR notifyAddr, uint64_t notifyVal);
    template <auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteWithNotifyNbi(UbcCtpBatchHandle &batchHandle, GM_ADDR dst, GM_ADDR src, uint32_t len,
                                                 GM_ADDR notifyAddr, uint64_t notifyVal);
    template <typename T, bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t AtomicFAA(ChannelHandle channel, GM_ADDR dst, GM_ADDR fetchAddr, T addVal);
    template <typename T, bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t AtomicCAS(ChannelHandle channel, GM_ADDR dst, GM_ADDR fetchAddr, T compareVal, T swapVal);
    template <pipe_t pipe = PIPE_S>
    __aicore__ inline int32_t Commit(ChannelHandle channel);
    __aicore__ inline int32_t BatchCommit(UbcCtpBatchHandle &batchHandle);
    template <pipe_t pipe = PIPE_MTE3>
    __aicore__ inline int32_t Drain(ChannelHandle channel);
    template <pipe_t pipe = PIPE_MTE3>
    __aicore__ inline int32_t Drain(UbcCtpBatchHandle &batchHandle);

private:
    template <bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              HcommUrmaOpCode opCode = HcommUrmaOpCode::WRITE, auto const &config = URMA_DEFAULT_CFG,
              typename T = uint64_t>
    __aicore__ inline int32_t PostSend(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len,
                                       GM_ADDR notifyAddr = nullptr, const UdmaParams<T> &params = UdmaParams<T>{});
    template <HcommUrmaOpCode opCode, auto const &config>
    __aicore__ inline int32_t BatchPostSend(UbcCtpBatchHandle &batchHandle, GM_ADDR remoteAddr, GM_ADDR localAddr,
                                            uint32_t len, GM_ADDR notifyAddr = nullptr, uint64_t notifyVal = 0);
    __aicore__ inline void CommitImpl(ChannelHandle channel, const SqContext &sqCtx, uint32_t sqHead, uint32_t cqeCnt);
    __aicore__ inline void PollCqWhenSqOverflow(ChannelHandle channel, const SqContext &sqCtx, const CqContext &cqCtx,
                                                uint32_t sqHead, uint32_t cqeCnt);
    __aicore__ inline uint32_t PollCqImpl(uint64_t cqBaseAddr, uint32_t cqeSize, uint32_t cqDepth, uint32_t expectTail,
                                          uint32_t &curTail, LocalTensor<uint32_t> cqeItem);
    __aicore__ inline uint32_t PollCq(ChannelHandle channel, uint32_t expectTail);
    __aicore__ inline uint32_t PollBatchCq(UbcCtpBatchHandle &batchHandle, uint32_t expectTail);

private:
    LocalTensor<uint32_t> wqeItem_;
    LocalTensor<uint32_t> cqeItem_;
};
} // namespace AscendC

#endif
#if defined(HCOMM_UNDEF_INCLUDE_INTERNAL_HEADERS_HCOMM_AIV_URMA_DEF_H)
#undef HCOMM_INCLUDE_INTERNAL_HEADERS
#undef HCOMM_UNDEF_INCLUDE_INTERNAL_HEADERS_HCOMM_AIV_URMA_DEF_H
#endif
