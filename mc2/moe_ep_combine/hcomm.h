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
 * \file hcomm.h
 * \brief Hcomm interface
 */
#ifndef INCLUDE_ADV_API_HCOMM_HCOMM_H
#define INCLUDE_ADV_API_HCOMM_HCOMM_H

#include "kernel_basic_intf.h"
#include "hcomm_common.h"

#if !defined(HCOMM_INCLUDE_INTERNAL_HEADERS)
#define HCOMM_INCLUDE_INTERNAL_HEADERS
#define HCOMM_UNDEF_INCLUDE_INTERNAL_HEADERS_HCOMM_H
#endif

#include "../../../impl/comm_api/aicore/hcomm/impl/hcomm_impl_def.h"

namespace AscendC {

/*!
 * @class Hcomm
 * @brief This class mainly provides a series of point-to-point communication primitive interfaces,
 *        benchmarking against Huawei's point-to-point communication C++ interface,
 *        including WriteNbi、ReadNbi and so on.
 *        The typical usage of this class is as follows:
 *          1) Create Hcomm object.
 *          2) Initialize communication channel.
 *          3) Launch comm tasks asynchronously through the corresponding interface,
 *             and the server starts assembling and launching comm tasks as soon as it listens.
 *          4) If commit is false when launching task, call the Commit interface to notify the execution of the
 *             corresponding comm task.
 *          5) Call the Drain interface (blocking) to wait for the server to complete the corresponding comm task.
 * @tparam commProtocol: The communication protocol to use. COMM_PROTOCOL_UBC_CTP (implemented through URMA) is used
 *                       by default.
 */
template <CommProtocol commProtocol = COMM_PROTOCOL_UBC_CTP>
class Hcomm {
public:
    /*!
     * @brief Initialize Hcomm workspace.
     * @param [in] buff: The UB buffer provided by caller.
     * @param [in] len: The buffer length in bytes.
     * @return 0 indicates success and -1 indicates failure.
     * @note URMA uses buff as its temporary workspace after 32-byte alignment.
     */
    __aicore__ inline int32_t Init(__ubuf__ uint8_t *buff, uint32_t len);

    /*!
     * @brief Initialize Hcomm workspace using LocalTensor.
     * @tparam T: The element type of the LocalTensor.
     * @param [in] buff: The LocalTensor buffer provided by caller. Start address must be 32-byte aligned.
     * @param [in] len: The buffer length in bytes.
     * @return 0 indicates success and -1 indicates failure.
     */
    template <typename T>
    __aicore__ inline int32_t Init(const LocalTensor<T> &buff, uint32_t len);

    /*!
     * @brief Create a protocol-specific batch handle using a LocalTensor buffer.
     * @tparam T: The communication channel handle type.
     * @tparam U: The element type of the LocalTensor.
     * @param [in] channel: The handle of the communication channel.
     * @param [in] buff: The LocalTensor buffer used to prepare batched WQEs.
     * @param [in] buffLen: The buffer length in bytes.
     * @param [in] remoteAddr: An address used for the one-time lookup of remote registered memory. If it is null,
     *                         the first remote registered buffer is selected. This is the only remote registration
     *                         lookup performed for the returned batch handle.
     * @param [in] localAddr: Reserved local memory address.
     * @return An initialized batch handle for commProtocol.
     * @note For UBC CTP, the tokenId and tokenValue selected by remoteAddr are cached in the returned handle.
     *       Subsequent batch operations neither read the remote registration table nor validate their remote
     *       address ranges. The caller must ensure that every remote range belongs to the registered memory
     *       represented by the cached tokenId/tokenValue. buffLen must be smaller than the SQ capacity in bytes.
     */
    template <typename T, typename U>
    __aicore__ inline BatchHandle<T> MakeBatchHandle(T channel, const LocalTensor<U> &buff, uint32_t buffLen,
                                                     GM_ADDR remoteAddr = nullptr, GM_ADDR localAddr = nullptr);

    /*!
     * @class Hcomm
     * @brief The task launching interface of the Write point-to-point communication operator.
     *        (task content: Write data of length len from src to dst through the specified channel.)
     * @tparam commit: true/false true: commit the task immediately; false: do not commit immediately.
     * @tparam commitPipe: The pipe type to use for commit, PIPE_S supported as default.
     * @tparam reqPipe: The pipe type to use for req, PIPE_MTE3 supported as default.
     * @tparam config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] channel: The handle of the communication channel.
     * @param [out] dst: The destination address of the data.
     * @param [in] src: The source address of the data.
     * @param [in] len: The length of the data to write, using byte as the basic unit.
     * @return 0 indicates success and -1 indicates failure.
     * @note Must be called after channel initialization.
     */
    template <bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len);

    /*!
     * @brief Prepare a Write WQE in a protocol-specific batch handle.
     * @tparam config: URMA WQE control config. Inline WQE is not supported.
     * @tparam T: The protocol-specific batch handle type.
     * @param [in,out] batchHandle: The batch handle to append the WQE to.
     * @param [out] dst: The remote destination address. The caller must ensure that its access range matches the
     *                   tokenId/tokenValue cached in batchHandle.
     * @param [in] src: The local source address.
     * @param [in] len: The length of the data to write in bytes.
     * @return 0 indicates success and -1 indicates failure.
     * @note The destination range is not validated against the registration selected by MakeBatchHandle and the
     *       remote registration table is not queried again.
     */
    template <auto const &config = URMA_DEFAULT_CFG, typename T, typename BatchHandleTraits<T>::ChannelType * = nullptr>
    __aicore__ inline int32_t WriteNbi(T &batchHandle, GM_ADDR dst, GM_ADDR src, uint32_t len);

    /*!
     * @class Hcomm
     * @brief The task launching interface of the inline Write point-to-point communication operator.
     *        The source data is provided by value and carried inline in the WQE.
     * @tparam T: The value type to write.
     * @tparam commit: true/false true: commit the task immediately; false: do not commit immediately.
     * @tparam commitPipe: The pipe type to use for commit, PIPE_S supported as default.
     * @tparam reqPipe: The pipe type to use for req, PIPE_MTE3 supported as default.
     * @tparam config: URMA WQE control config, only used by URMA.
     *         Default: strongly ordered + fence + CQE + inline enabled.
     * @param [in] channel: The handle of the communication channel.
     * @param [out] dst: The destination address of the data.
     * @param [in] value: The inline value to write.
     * @return 0 indicates success and -1 indicates failure.
     * @note Must be called after channel initialization.
     */
    template <typename T, bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_INLINE_CFG>
    __aicore__ inline int32_t WriteValueNbi(ChannelHandle channel, GM_ADDR dst, T value);

    /*!
     * @class Hcomm
     * @brief The task launching interface of the write-with-reduce point-to-point communication operator.
     *        The source data is reduced into the destination data on the remote side.
     * @tparam T: The element data type. int8_t, int16_t, int32_t, uint32_t, half, float and bfloat16_t are supported.
     * @tparam reduceOp: The reduction operation.
     * @tparam commit: true/false true: commit the task immediately; false: do not commit immediately.
     * @tparam commitPipe: The pipe type to use for commit, PIPE_S supported as default.
     * @tparam reqPipe: The pipe type to use for req, PIPE_MTE supported as default.
     * @tparam config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] channel: The handle of the communication channel.
     * @param [out] dst: The remote destination address of the reduction.
     * @param [in] src: The local source address of the reduction.
     * @param [in] count: The number of elements to reduce.
     * @return 0 indicates success and -1 indicates failure.
     * @note Only the UBC_CTP/URMA path supports this interface.
     */
    template <typename T, HcommUrmaReduceOp reduceOp, bool commit = true, pipe_t commitPipe = PIPE_S,
              pipe_t reqPipe = PIPE_MTE3, auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteReduceNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t count);

    /*!
     * @class Hcomm
     * @brief @brief The task launching interface of the Write-with-notify point-to-point communication operator.
     * @tparam commit: true/false true: commit the task immediately; false: do not commit immediately.
     * @tparam commitPipe: The pipe type to use for commit, PIPE_S supported as default.
     * @tparam reqPipe: The pipe type to use for req, PIPE_MTE3 supported as default.
     * @tparam config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] channel: The handle of the communication channel.
     * @param [out] dst: The destination address of the data.
     * @param [in] src: The source address of the data.
     * @param [in] len: The length of the data to write, using byte as the basic unit.
     * @param [in] notifyAddr: The remote notify address.
     * @param [in] notifyVal: The remote notify value.
     * @return 0 indicates success and -1 indicates failure.
     * @note Must be called after channel initialization.
     */
    template <bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t WriteWithNotifyNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len,
                                                 GM_ADDR notifyAddr, uint64_t notifyVal);

    /*!
     * @brief Prepare a Write-with-notify WQE in a protocol-specific batch handle.
     * @tparam config: URMA WQE control config. Inline WQE is not supported.
     * @tparam T: The protocol-specific batch handle type.
     * @param [in,out] batchHandle: The batch handle to append the WQE to.
     * @param [out] dst: The remote destination address. The caller must ensure that its access range matches the
     *                   tokenId/tokenValue cached in batchHandle.
     * @param [in] src: The local source address.
     * @param [in] len: The length of the data to write in bytes.
     * @param [in] notifyAddr: The remote notify address. The caller must ensure that its access range matches the
     *                         tokenId/tokenValue cached in batchHandle.
     * @param [in] notifyVal: The remote notify value.
     * @return 0 indicates success and -1 indicates failure.
     * @note The destination and notify ranges are not validated against the registration selected by
     *       MakeBatchHandle, and the remote registration table is not queried again. The notify context reuses the
     *       tokenId/tokenValue cached in batchHandle.
     */
    template <auto const &config = URMA_DEFAULT_CFG, typename T, typename BatchHandleTraits<T>::ChannelType * = nullptr>
    __aicore__ inline int32_t WriteWithNotifyNbi(T &batchHandle, GM_ADDR dst, GM_ADDR src, uint32_t len,
                                                 GM_ADDR notifyAddr, uint64_t notifyVal);

    /*!
     * @class Hcomm
     * @brief @brief The task launching interface of the Fetch-and-add point-to-point communication operator.
     * @tparam T: The data type of the atomic operation. Only int32_t, uint32_t, int64_t, uint64_t is supported.
     * @tparam commit: true/false true: commit the task immediately; false: do not commit immediately.
     * @tparam commitPipe: The pipe type to use for commit, PIPE_S supported as default.
     * @tparam reqPipe: The pipe type to use for req, PIPE_MTE3 supported as default.
     * @tparam config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] channel: The handle of the communication channel.
     * @param [out] dst: The destination address of the data.
     * @param [out] fetchAddr: The address to store the old value before atomic add.
     * @param [in] addVal: The remote add value.
     * @return 0 indicates success and -1 indicates failure.
     * @note Must be called after channel initialization.
     */
    template <typename T, bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t AtomicFAA(ChannelHandle channel, GM_ADDR dst, GM_ADDR fetchAddr, T addVal);

    /*!
     * @class Hcomm
     * @brief @brief The task launching interface of the Compare-and-swap point-to-point communication operator.
     * @tparam T: The data type of the atomic operation. Only int32_t, uint32_t, int64_t, uint64_t is supported.
     * @tparam commit: true/false true: commit the task immediately; false: do not commit immediately.
     * @tparam commitPipe: The pipe type to use for commit, PIPE_S supported as default.
     * @tparam reqPipe: The pipe type to use for req, PIPE_MTE3 supported as default.
     * @tparam config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] channel: The handle of the communication channel.
     * @param [out] dst: The destination address of the data.
     * @param [out] fetchAddr: The address to store the old value before atomic compare-and-swap.
     * @param [in] compareVal: The value to compare against the value at the destination address.
     * @param [in] swapVal: The value to swap into the destination address if comparison succeeds.
     * @return 0 indicates success and -1 indicates failure.
     * @note Must be called after channel initialization.
     */
    template <typename T, bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t AtomicCAS(ChannelHandle channel, GM_ADDR dst, GM_ADDR fetchAddr, T compareVal, T swapVal);

    /*!
     * @class Hcomm
     * @brief The task launching interface of the Read point-to-point communication operator.
     *        (task content: Read data of length len from src to dst through the specified channel.)
     * @tparam commit: true/false true: commit the task immediately; false: do not commit immediately.
     * @tparam commitPipe: The pipe type to use for commit, PIPE_S supported as default.
     * @tparam reqPipe: The pipe type to use for req, PIPE_MTE3 supported as default.
     * @tparam config: URMA WQE control config, only used by URMA. Default: strongly ordered + fence + CQE enabled.
     * @param [in] channel: The handle of the communication channel.
     * @param [out] dst: The destination address of the data.
     * @param [in] src: The source address of the data.
     * @param [in] len: The length of the data to read, using byte as the basic unit.
     * @return 0 indicates success and -1 indicates failure.
     * @note Must be called after channel initialization.
     */
    template <bool commit = true, pipe_t commitPipe = PIPE_S, pipe_t reqPipe = PIPE_MTE3,
              auto const &config = URMA_DEFAULT_CFG>
    __aicore__ inline int32_t ReadNbi(ChannelHandle channel, GM_ADDR dst, GM_ADDR src, uint64_t len);

    /*!
     * @brief Prepare a Read WQE in a protocol-specific batch handle.
     * @tparam config: URMA WQE control config. Inline WQE is not supported.
     * @tparam T: The protocol-specific batch handle type.
     * @param [in,out] batchHandle: The batch handle to append the WQE to.
     * @param [out] dst: The local destination address.
     * @param [in] src: The remote source address. The caller must ensure that its access range matches the
     *                  tokenId/tokenValue cached in batchHandle.
     * @param [in] len: The length of the data to read in bytes.
     * @return 0 indicates success and -1 indicates failure.
     * @note The source range is not validated against the registration selected by MakeBatchHandle and the remote
     *       registration table is not queried again.
     */
    template <auto const &config = URMA_DEFAULT_CFG, typename T, typename BatchHandleTraits<T>::ChannelType * = nullptr>
    __aicore__ inline int32_t ReadNbi(T &batchHandle, GM_ADDR dst, GM_ADDR src, uint32_t len);

    /*!
     * @class Hcomm
     * @brief Informed that tasks submitted on channel can be executed.
     * @tparam pipe: The pipe type to use for commit, PIPE_S supported as default.
     * @param [in] channel: The handle of the communication channel.
     * @return 0 indicates success and -1 indicates failure.
     */
    template <pipe_t pipe = PIPE_S>
    __aicore__ inline int32_t Commit(ChannelHandle channel);

    /*!
     * @brief Submit all WQEs prepared in a protocol-specific batch handle.
     * @tparam T: The protocol-specific batch handle type.
     * @param [in,out] batchHandle: The batch handle to submit. Its prepared WQEBB count is reset after success.
     * @return 0 indicates success and -1 indicates failure.
     */
    template <typename T, typename BatchHandleTraits<T>::ChannelType * = nullptr>
    __aicore__ inline int32_t BatchCommit(T &batchHandle);

    /*!
     * @class Hcomm
     * @brief Block Aicore and drain comm tasks submitted on channel until finish processing.
     * @tparam pipe: The pipe type to use for drain, PIPE_MTE3 supported as default.
     * @param [in] channel: The handle of the communication channel.
     * @return 0 indicates success. A non-zero value indicates failure. For COMM_PROTOCOL_UBC_CTP, the underlying
     *         CQ polling error code is returned directly.
     */
    template <pipe_t pipe = PIPE_MTE3>
    __aicore__ inline int32_t Drain(ChannelHandle channel);

    /*!
     * @brief Block Aicore until all communication tasks submitted through a batch handle are complete.
     * @tparam pipe: The pipe type to use for drain, PIPE_MTE3 supported as default.
     * @tparam T: The protocol-specific batch handle type.
     * @param [in,out] batchHandle: The batch handle whose WQE buffer is reused as CQE scratch space.
     * @return 0 indicates success. A non-zero value indicates failure.
     * @note This overload does not require Init. It must be called after BatchCommit and before preparing
     *       another batch on the same handle.
     */
    template <pipe_t pipe = PIPE_MTE3, typename T, typename BatchHandleTraits<T>::ChannelType * = nullptr>
    __aicore__ inline int32_t Drain(T &batchHandle);

private:
    HcommImpl<commProtocol> impl_;
};
} // namespace AscendC

#if defined(__NPU_ARCH__) && \
    (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 1001 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "../../../impl/comm_api/aicore/hcomm/impl/hcomm_impl.h"
#endif

#if defined(HCOMM_UNDEF_INCLUDE_INTERNAL_HEADERS_HCOMM_H)
#undef HCOMM_INCLUDE_INTERNAL_HEADERS
#undef HCOMM_UNDEF_INCLUDE_INTERNAL_HEADERS_HCOMM_H
#endif

#endif // INCLUDE_ADV_API_HCOMM_HCOMM_H
