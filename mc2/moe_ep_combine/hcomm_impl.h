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
 * \file hcomm_impl.h
 * \brief Hcomm implementation
 */

#if !defined(HCOMM_INCLUDE_INTERNAL_HEADERS)
#pragma message("This is an internal Hcomm header. Please include public Hcomm headers instead.")
#define HCOMM_INCLUDE_INTERNAL_HEADERS
#define HCOMM_UNDEF_INCLUDE_INTERNAL_HEADERS_HCOMM_IMPL_H
#endif

#ifndef IMPL_ADV_API_DETAIL_HCOMM_IMPL_HCOMM_IMPL_H
#define IMPL_ADV_API_DETAIL_HCOMM_IMPL_HCOMM_IMPL_H

#include <type_traits>

#if __NPU_ARCH__ == 2201
#include "platform_v220/hcomm_aiv.h"
#elif __NPU_ARCH__ == 3510
#include "platform_v310/hcomm_aiv_roce.h"
#include "platform_v310/hcomm_aiv_urma.h"
#endif

namespace AscendC {

template <CommProtocol commProtocol>
__aicore__ inline int32_t Hcomm<commProtocol>::Init(__ubuf__ uint8_t *buff, uint32_t len)
{
    return impl_.Init(buff, len);
}

template <CommProtocol commProtocol>
template <typename T>
__aicore__ inline int32_t Hcomm<commProtocol>::Init(const LocalTensor<T> &buff, uint32_t len)
{
    return impl_.Init(buff, len);
}

template <CommProtocol commProtocol>
template <typename T, typename U>
__aicore__ inline BatchHandle<T> Hcomm<commProtocol>::MakeBatchHandle(T channel, const LocalTensor<U> &buff,
                                                                      uint32_t buffLen, GM_ADDR remoteAddr,
                                                                      GM_ADDR localAddr)
{
    PipeBarrier<PIPE_ALL>();
    static_assert(commProtocol == COMM_PROTOCOL_UBC_CTP, "BatchHandle only supports COMM_PROTOCOL_UBC_CTP");
    auto batchHandle = impl_.MakeBatchHandle(channel, buff, buffLen, remoteAddr, localAddr);
    using ExpectedType = BatchHandle<T>;
    static_assert(std::is_same<decltype(batchHandle), ExpectedType>::value,
                  "Channel type and BatchHandle type do not match");
    return batchHandle;
}

template <CommProtocol commProtocol>
template <auto const &config, typename T, typename BatchHandleTraits<T>::ChannelType *>
__aicore__ inline int32_t Hcomm<commProtocol>::WriteNbi(T &batchHandle, GM_ADDR dst, GM_ADDR src, uint32_t len)
{
    PipeBarrier<PIPE_ALL>();
    return impl_.template WriteNbi<config>(batchHandle, dst, src, len);
}

template <CommProtocol commProtocol>
template <bool commit, pipe_t commitPipe, pipe_t reqPipe, auto const &config>
__aicore__ inline int32_t Hcomm<commProtocol>::WriteNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    return impl_.template WriteNbi<commit, commitPipe, reqPipe, config>(channel, dst, src, len);
}

template <CommProtocol commProtocol>
template <typename T, bool commit, pipe_t commitPipe, pipe_t reqPipe, auto const &config>
__aicore__ inline int32_t Hcomm<commProtocol>::WriteValueNbi(ChannelHandle channel, GM_ADDR dst, T value)
{
    return impl_.template WriteValueNbi<T, commit, commitPipe, reqPipe, config>(channel, dst, value);
}

template <CommProtocol commProtocol>
template <typename T, HcommUrmaReduceOp reduceOp, bool commit, pipe_t commitPipe, pipe_t reqPipe, auto const &config>
__aicore__ inline int32_t Hcomm<commProtocol>::WriteReduceNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src,
                                                              uint64_t count)
{
    return impl_.template WriteReduceNbi<T, reduceOp, commit, commitPipe, reqPipe, config>(channel, dst, src, count);
}

template <CommProtocol commProtocol>
template <auto const &config, typename T, typename BatchHandleTraits<T>::ChannelType *>
__aicore__ inline int32_t Hcomm<commProtocol>::WriteWithNotifyNbi(T &batchHandle, GM_ADDR dst, GM_ADDR src,
                                                                  uint32_t len, GM_ADDR notifyAddr, uint64_t notifyVal)
{
    PipeBarrier<PIPE_ALL>();
    return impl_.template WriteWithNotifyNbi<config>(batchHandle, dst, src, len, notifyAddr, notifyVal);
}

template <CommProtocol commProtocol>
template <bool commit, pipe_t commitPipe, pipe_t reqPipe, auto const &config>
__aicore__ inline int32_t Hcomm<commProtocol>::WriteWithNotifyNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src,
                                                                  uint64_t len, GM_ADDR notifyAddr, uint64_t notifyVal)
{
    return impl_.template WriteWithNotifyNbi<commit, commitPipe, reqPipe, config>(channel, dst, src, len, notifyAddr,
                                                                                  notifyVal);
}

template <CommProtocol commProtocol>
template <bool commit, pipe_t commitPipe, pipe_t reqPipe, auto const &config>
__aicore__ inline int32_t Hcomm<commProtocol>::ReadNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    return impl_.template ReadNbi<commit, commitPipe, reqPipe, config>(channel, dst, src, len);
}

template <CommProtocol commProtocol>
template <auto const &config, typename T, typename BatchHandleTraits<T>::ChannelType *>
__aicore__ inline int32_t Hcomm<commProtocol>::ReadNbi(T &batchHandle, GM_ADDR dst, GM_ADDR src, uint32_t len)
{
    PipeBarrier<PIPE_ALL>();
    return impl_.template ReadNbi<config>(batchHandle, dst, src, len);
}

template <CommProtocol commProtocol>
template <typename T, bool commit, pipe_t commitPipe, pipe_t reqPipe, auto const &config>
__aicore__ inline int32_t Hcomm<commProtocol>::AtomicFAA(ChannelHandle channel, GM_ADDR dst, GM_ADDR fetchAddr,
                                                         T addVal)
{
    return impl_.template AtomicFAA<T, commit, commitPipe, reqPipe, config>(channel, dst, fetchAddr, addVal);
}

template <CommProtocol commProtocol>
template <typename T, bool commit, pipe_t commitPipe, pipe_t reqPipe, auto const &config>
__aicore__ inline int32_t Hcomm<commProtocol>::AtomicCAS(ChannelHandle channel, GM_ADDR dst, GM_ADDR fetchAddr,
                                                         T compareVal, T swapVal)
{
    return impl_.template AtomicCAS<T, commit, commitPipe, reqPipe, config>(channel, dst, fetchAddr, compareVal,
                                                                            swapVal);
}

template <CommProtocol commProtocol>
template <pipe_t pipe>
__aicore__ inline int32_t Hcomm<commProtocol>::Commit(ChannelHandle channel)
{
    return impl_.template Commit<pipe>(channel);
}

template <CommProtocol commProtocol>
template <typename T, typename BatchHandleTraits<T>::ChannelType *>
__aicore__ inline int32_t Hcomm<commProtocol>::BatchCommit(T &batchHandle)
{
    PipeBarrier<PIPE_ALL>();
    return impl_.BatchCommit(batchHandle);
}

template <CommProtocol commProtocol>
template <pipe_t pipe>
__aicore__ inline int32_t Hcomm<commProtocol>::Drain(ChannelHandle channel)
{
    return impl_.template Drain<pipe>(channel);
}

template <CommProtocol commProtocol>
template <pipe_t pipe, typename T, typename BatchHandleTraits<T>::ChannelType *>
__aicore__ inline int32_t Hcomm<commProtocol>::Drain(T &batchHandle)
{
    PipeBarrier<PIPE_ALL>();
    return impl_.template Drain<pipe>(batchHandle);
}
} // namespace AscendC

#endif // IMPL_ADV_API_DETAIL_HCOMM_IMPL_HCOMM_IMPL_H
#if defined(HCOMM_UNDEF_INCLUDE_INTERNAL_HEADERS_HCOMM_IMPL_H)
#undef HCOMM_INCLUDE_INTERNAL_HEADERS
#undef HCOMM_UNDEF_INCLUDE_INTERNAL_HEADERS_HCOMM_IMPL_H
#endif
