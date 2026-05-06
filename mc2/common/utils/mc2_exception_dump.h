/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_WIN_DUMP_H
#define MC2_WIN_DUMP_H

#include "acl/acl.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "acl/acl_dump.h"
#include "../op_kernel/moe_distribute_comm_ctx.h"
#include "mc2_log.h"
#include "mc2_tiling_utils.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <vector>
#include "version/runtime_version.h"
#include "version/metadef_version.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_tiling.h"
#include "../../moe_distribute_combine_v2/op_kernel/moe_distribute_combine_tiling.h"
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_a2_constant.h"
#include "./mc2_hcom_topo_info.h"

#define EXCEPTION_DUMP_SUPPORT_VERSION 90000000

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
namespace Mc2Exception {

const std::string OP_NAME = "Mc2Exception";
const uint32_t WIN_SIZE = 1024U * 1024U;
const uint32_t MS_WIDTH = 3U;
const uint32_t MS_PER_S = 1000U;
const mode_t FILE_MODE = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

inline std::string GetTimestampWithMilliseconds()
{
    auto now = std::chrono::system_clock::now();
    time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm tm_utc;
    gmtime_r(&now_time_t, &tm_utc);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % MS_PER_S;

    std::stringstream ss;
    ss << std::put_time(&tm_utc, "%Y%m%d%H%M%S") << std::setfill('0') << std::setw(MS_WIDTH) << ms.count();
    return ss.str();
}

inline std::string GenDumpFileName(aclrtExceptionInfo *args, const char *op)
{
    std::stringstream ss;

    uint32_t streamId = aclrtGetStreamIdFromExceptionInfo(args);
    uint32_t taskId = aclrtGetTaskIdFromExceptionInfo(args);
    std::string ts = GetTimestampWithMilliseconds();

    ss << "mc2_exception_info_" << op << "." << streamId << "." << taskId << "." << ts;

    return ss.str();
}

inline bool IsStrEmpty(std::string str)
{
    // return if the string is null or empty or spaces
    return (str.empty() || std::all_of(str.begin(), str.end(), [](unsigned char c) { return std::isspace(c); }));
}

inline int DumpToFile(std::string dir, std::string name, uint32_t id, void *buf)
{
    // check if dir and buf valid
    if (IsStrEmpty(dir) || IsStrEmpty(name)) {
        OP_LOGE(OP_NAME, "Dump path or buf is null.");
        return -1;
    }

    std::string rankPath = dir + "/" + std::to_string(id) + "/";
    std::string path = rankPath + name;
    OP_LOGE(OP_NAME, "Start to dump file. The dump path is %s", path.c_str());

    // Open file
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, FILE_MODE);
    if (fd < 0) {
        int openErrno = errno;
        OP_LOGE(OP_NAME, "Failed to open a dump file. errno=%d(%s)", openErrno, strerror(openErrno));
        return -1;
    }

    // Write to file
    ssize_t ret = write(fd, buf, WIN_SIZE);
    if (ret < 0) {
        int writeErrno = errno;
        OP_LOGE(OP_NAME, "Failed to write a dump file. errno=%d(%s)", writeErrno, strerror(writeErrno));
        close(fd);
        return -1;
    }

    // Close file
    close(fd);
    OP_LOGE(OP_NAME, "Dump to file %s done.", path.c_str());
    return 0;
}

inline int ProcessArgsForA5(uint64_t argsAddr, std::vector<uint8_t> &winBuf)
{
    // Get hccl context from its addr
    std::vector<uint8_t> hcclArgs(sizeof(HcclCombinOpParam), 0);
    auto ret = aclrtMemcpy(hcclArgs.data(), sizeof(HcclCombinOpParam), (void *)argsAddr, sizeof(HcclCombinOpParam),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy HcclCombinOpParam from device to host failed. ret = %d", ret);
        return -1;
    }
    HcclCombinOpParam* winContext = reinterpret_cast<HcclCombinOpParam *>(hcclArgs.data());
    if (winContext == nullptr) {
        OP_LOGE(OP_NAME, "Cast to winContext failed. HcclCombinOpParam is null.");
        return -1;
    }
    OP_LOGD(OP_NAME, "Get winContext from args. rankId=%u, rankDim=%u", winContext->rankId, winContext->rankDim);

    void* winAddr = reinterpret_cast<void *>(winContext->windowsIn[winContext->rankId]);
    if (winAddr == nullptr) {
        OP_LOGE(OP_NAME, "Get winaddr failed.");
        return -1;
    }
    // Get windowsIn of each rank from hccl context
    ret = aclrtMemcpy(winBuf.data(), WIN_SIZE, winAddr, WIN_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy windows from device to host failed. ret = %d", ret);
        return -1;
    }
    return 0;
}

inline int ProcessArgsForA3(uint64_t argsAddr, std::vector<uint8_t> &winBuf)
{
    // Get hccl context from its addr
    std::vector<uint8_t> hcclArgs(sizeof(CommContextForDump), 0);
    auto ret = aclrtMemcpy(hcclArgs.data(), sizeof(CommContextForDump), (void *)argsAddr, sizeof(CommContextForDump),
                           ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy CommContextForDump from device to host failed. ret = %d", ret);
        return -1;
    }
    CommContextForDump* winContext = reinterpret_cast<CommContextForDump *>(hcclArgs.data());
    if (winContext == nullptr) {
        OP_LOGE(OP_NAME, "Cast to winContext failed. CommContextForDump is null.");
        return -1;
    }
    OP_LOGD(OP_NAME, "Get winContext from args. rankId=%u", winContext->epRankid);

    void* winAddr = reinterpret_cast<void *>(winContext->epHcclBufffer_[winContext->epRankid]);
    if (winAddr == nullptr) {
        OP_LOGE(OP_NAME, "Get win addr failed.");
        return -1;
    }

    // Get windowsIn of each rank from hccl context
    ret = aclrtMemcpy(winBuf.data(), WIN_SIZE, winAddr, WIN_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
        return -1;
    }
    return 0;
}

struct TestStruct {
    uint8_t res[16];
    char groupName[128];
    uint8_t res1[136];
};

template <typename TilingData>
int ProcessArgs910BImpl(void* devArgsPtr, std::vector<uint8_t> &winBuf, uint32_t devArgsLen, uint32_t tilingDataOffset, bool isDispatch)
{
    uint64_t argsAddr = 0;
    auto maxArgNum = devArgsLen / sizeof(uint64_t);
    std::vector<uint8_t> devArgsVector(devArgsLen, 0);
    auto ret = aclrtMemcpy(devArgsVector.data(), devArgsLen, devArgsPtr, devArgsLen, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<uint64_t> devArgsVector64(maxArgNum, 0);
    std::memcpy(devArgsVector64.data(), devArgsVector.data(), devArgsVector.size());
    if (!(ret == ACL_SUCCESS)) {
        OP_LOGE(OP_NAME, "aclrtMemcpy address of args failed. ret=%d", ret);
        return -1;
    }

    std::vector<uint8_t> rawTilingData(sizeof(TilingData), 0);
    argsAddr = devArgsVector64[tilingDataOffset]; // The loc of tiling data.
    ret = aclrtMemcpy(rawTilingData.data(), sizeof(TilingData), (void *)argsAddr, sizeof(TilingData), ACL_MEMCPY_DEVICE_TO_HOST);
    TilingData* tilingData = reinterpret_cast<TilingData*>(rawTilingData.data());
    Mc2CcTiling mc2CcTiling = tilingData->mc2CcTiling;
    TestStruct* test = reinterpret_cast<TestStruct*>(&mc2CcTiling);
    OP_LOGE(OP_NAME, "GroupName=%s", test->groupName);
    uint64_t size;
    void *buffer;
    Mc2Hcom::MC2HcomTopology::CommGetHcclBufferByGroup(test->groupName, &buffer, &size);
    OP_LOGE(OP_NAME, "HCCL BUFFER SIZE=%luMB. WindowInAddr=%p.", size / (1 * 1024 * 1024) / 2, buffer);
    void *aDumpBuffer;
    aDumpBuffer = buffer + size / 4 - Mc2A2Kernel::BUFFER_ADUMP_WIN_SIZE * 2U;
    OP_LOGE(OP_NAME, "ADumpBuffer=%p", aDumpBuffer);

    if (winBuf.size() < Mc2A2Kernel::BUFFER_ADUMP_WIN_ADDR_START + Mc2A2Kernel::BUFFER_ADUMP_WIN_SIZE * 2U) {
        OP_LOGE(OP_NAME, "winBuf size is too small.");
        return -1;
    }

    ret = aclrtMemcpy(winBuf.data() + Mc2A2Kernel::BUFFER_ADUMP_WIN_ADDR_START, Mc2A2Kernel::BUFFER_ADUMP_WIN_SIZE * 2U, (const void *)aDumpBuffer, Mc2A2Kernel::BUFFER_ADUMP_WIN_SIZE * 2U, ACL_MEMCPY_DEVICE_TO_HOST);
    if (!(ret == ACL_SUCCESS)) {
        OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
        return -1;
    }
    std::vector<uint32_t> singleAiv(Mc2A2Kernel::BUFFER_ADUMP_WIN_SIZE_PER_AIV / sizeof(uint32_t), 0);
    uint32_t offsetForValidDump = isDispatch ? 0U : Mc2A2Kernel::BUFFER_ADUMP_WIN_SIZE;
    std::memcpy(singleAiv.data(), winBuf.data() + Mc2A2Kernel::BUFFER_ADUMP_WIN_ADDR_START + offsetForValidDump, singleAiv.size());
    OP_LOGE(OP_NAME, "IsLayered=%d", singleAiv[Mc2A2Kernel::ISLAYERED_POS]);
    uint32_t bufferId = singleAiv[Mc2A2Kernel::BUFFERID_POS];
    uint32_t axisH = singleAiv[Mc2A2Kernel::H_POS];
    if (!isDispatch) {
        bufferId = 1 - bufferId;
    }
    void *statusBuffer = buffer;
    
    if (singleAiv[Mc2A2Kernel::ISLAYERED_POS] == 1) {
        constexpr static uint64_t STATE_OFFSET = 512UL;
        constexpr static uint32_t SERVER_RANK_SIZE = 8;
        // Dispatch
        uint64_t offset = 0;
        uint32_t serverNum = singleAiv[Mc2A2Kernel::TILING_WORLDSIZE_POS] / SERVER_RANK_SIZE;
        uint32_t statusBytes = (serverNum + 1) * 2U * STATE_OFFSET;
        // arrive status
        ret = aclrtMemcpy(winBuf.data() + offset, Mc2A2Kernel::BUFFER_ADUMP_WIN_DISPATCH_SIZE, (const void *)statusBuffer + size / 4 * bufferId, statusBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (!(ret == ACL_SUCCESS)) {
            OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
            return -1;
        }
        // Dispatch IpcSync
        offset = statusBytes;
        ret = aclrtMemcpy(winBuf.data() + offset, Mc2A2Kernel::BUFFER_ADUMP_WIN_DISPATCH_SIZE, (const void *)statusBuffer + size / 4 - 1 * 1024 * 1024, SERVER_RANK_SIZE * Mc2A2Kernel::UB_ALIGN, ACL_MEMCPY_DEVICE_TO_HOST);
        if (!(ret == ACL_SUCCESS)) {
            OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
            return -1;
        }
        // Combine
        statusBytes = (serverNum + 1) * STATE_OFFSET;
        // arrive status
        offset = Mc2A2Kernel::BUFFER_ADUMP_WIN_DISPATCH_SIZE;
        ret = aclrtMemcpy(winBuf.data() + offset, Mc2A2Kernel::BUFFER_ADUMP_WIN_COMBINE_SIZE, (const void *)statusBuffer + size / 4 * (1 - bufferId), statusBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (!(ret == ACL_SUCCESS)) {
            OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
            return -1;
        }
        // Combine IpcSync
        offset = Mc2A2Kernel::BUFFER_ADUMP_WIN_DISPATCH_SIZE + statusBytes;
        ret = aclrtMemcpy(winBuf.data() + offset, Mc2A2Kernel::BUFFER_ADUMP_WIN_COMBINE_SIZE, (const void *)statusBuffer + size / 4 - 2 * 1024 * 1024, SERVER_RANK_SIZE * Mc2A2Kernel::UB_ALIGN, ACL_MEMCPY_DEVICE_TO_HOST);
        if (!(ret == ACL_SUCCESS)) {
            OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
            return -1;
        }
    } else if (singleAiv[Mc2A2Kernel::ISLAYERED_POS] == 0) {
        uint32_t worldSize = singleAiv[Mc2A2Kernel::TILING_WORLDSIZE_POS];
        uint32_t moeExpertNum = singleAiv[Mc2A2Kernel::MOE_NUM_POS];
        OP_LOGE(OP_NAME, "worldSize=%d, moeExpertNum=%d, axisH=%d", worldSize, moeExpertNum, axisH);

        uint64_t dataSizePerRank_ = (size / 4 - 1 * 1024 * 1024) / worldSize / Mc2A2Kernel::UB_ALIGN * Mc2A2Kernel::UB_ALIGN;
        uint32_t localMoeExpertNum = moeExpertNum / worldSize;
        uint32_t statusEntryCount = ((localMoeExpertNum + 1) * sizeof(uint32_t) + Mc2A2Kernel::UB_ALIGN - 1) / Mc2A2Kernel::UB_ALIGN * Mc2A2Kernel::UB_ALIGN;
        // Dispatch
        for (uint32_t rank = 0; rank < worldSize; ++rank) {
            uint32_t srcOffset = rank * dataSizePerRank_ + size / 4 * bufferId;
            uint32_t dstOffset = rank * statusEntryCount;
            ret = aclrtMemcpy(winBuf.data() + dstOffset, Mc2A2Kernel::BUFFER_ADUMP_WIN_DISPATCH_SIZE, (const void *)statusBuffer + srcOffset, statusEntryCount, ACL_MEMCPY_DEVICE_TO_HOST);
            if (!(ret == ACL_SUCCESS)) {
                OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
                return -1;
            }
        }
        // Combine
        OP_LOGE(OP_NAME, "dataSizePerRank_=%d", dataSizePerRank_);
        for (uint32_t rank = 0; rank < worldSize; ++rank) {
            uint32_t srcOffset = rank * dataSizePerRank_ + size / 4 * (1 - bufferId);
            uint32_t dstOffset = rank * Mc2A2Kernel::UB_ALIGN * 2 + Mc2A2Kernel::BUFFER_ADUMP_WIN_DISPATCH_SIZE;
            ret = aclrtMemcpy(winBuf.data() + dstOffset, Mc2A2Kernel::BUFFER_ADUMP_WIN_COMBINE_SIZE, (const void *)statusBuffer + srcOffset, Mc2A2Kernel::UB_ALIGN, ACL_MEMCPY_DEVICE_TO_HOST);
            if (!(ret == ACL_SUCCESS)) {
                OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
                return -1;
            }
            uint32_t tokenNum = 0;
            std::memcpy(&tokenNum, winBuf.data() + dstOffset, sizeof(uint32_t));
            OP_LOGE(OP_NAME, "rank=%d, tokenNum=%d", rank, tokenNum);
            srcOffset += Mc2A2Kernel::UB_ALIGN * 2 + tokenNum * axisH * sizeof(uint16_t);
            dstOffset += Mc2A2Kernel::UB_ALIGN;
            ret = aclrtMemcpy(winBuf.data() + dstOffset, Mc2A2Kernel::BUFFER_ADUMP_WIN_COMBINE_SIZE, (const void *)statusBuffer + srcOffset, Mc2A2Kernel::UB_ALIGN, ACL_MEMCPY_DEVICE_TO_HOST);
            if (!(ret == ACL_SUCCESS)) {
                OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
                return -1;
            }
        }

    } else {
        OP_LOGE(OP_NAME, "Unsupported isLayered: %d", singleAiv[Mc2A2Kernel::ISLAYERED_POS]);
    }
    return 0;
}

inline int ProcessArgs910B(void* devArgsPtr, std::vector<uint8_t> &winBuf, uint32_t devArgsLen, const char *op)
{
    if (std::strcmp(op, "MoeDistributeDispatchV2") == 0) {
        return ProcessArgs910BImpl<MoeDistributeDispatchA2TilingData>(devArgsPtr, winBuf, devArgsLen, 18, true);
    } else if (std::strcmp(op, "MoeDistributeCombineV2") == 0) {
        return ProcessArgs910BImpl<MoeDistributeCombineA2TilingData>(devArgsPtr, winBuf, devArgsLen, 23, false);
    } else {
        OP_LOGE(OP_NAME, "Unsupported op: %s", op);
        return -1;
    }
}

inline void Mc2ExceptionImpl(aclrtExceptionInfo *args, void *userdata, const char *op)
{
    const char* socName = aclrtGetSocName();

    OP_LOGD(OP_NAME, "Start to handle mc2 exception and dump win info.");

    // Get addr of hccl context from ExceptionInfo
    void* devArgsPtr = nullptr;
    uint32_t devArgsLen = 0;
    auto ret = aclrtGetArgsFromExceptionInfo(args, &devArgsPtr, &devArgsLen);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtGetArgsFromExceptionInfo failed. ret=%d", ret);
        return;
    }
    uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtGetDeviceIdFromExceptionInfo failed. ret=%d", ret);
        return;
    }
    OP_LOGD(OP_NAME, "Get context from args. deviceId=%u, devArgsAddr=%p, devArgsLen=%u", deviceId, devArgsPtr,
            devArgsLen);

    uint64_t argsAddr = 0;
    uint64_t argsOffset = 0;
    // 由于dispatchv3中包含syncacll，推断为mix算子，其args的首地址是ffts，第二个地址才是hcclcontext
    if ((std::strstr(socName, "Ascend910_93") != nullptr) && (std::strstr(op, "MoeDistributeDispatchV3") != nullptr)) {
        argsOffset = sizeof(uint64_t);
    }
    ret = aclrtMemcpy(&argsAddr, sizeof(uint64_t), devArgsPtr + argsOffset, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy address of args failed. ret=%d", ret);
        return;
    }

    // Get win content
    std::vector<uint8_t> winContent(WIN_SIZE, 0);
    if (std::strstr(socName, "Ascend910_93") != nullptr) {
        if (ProcessArgsForA3(argsAddr, winContent) != 0) {
            OP_LOGE(OP_NAME, "Failed to get win content.");
            return;
        }
    } else if (std::strstr(socName, "Ascend950") != nullptr) {
        if (ProcessArgsForA5(argsAddr, winContent) != 0) {
            OP_LOGE(OP_NAME, "aclrtMemcpy address of args failed. ret=%d", ret);
            return;
        }
    } else if (std::strstr(socName, "Ascend910B") != nullptr) {
        if (ProcessArgs910B(devArgsPtr, winContent, devArgsLen, op) != 0) {
            OP_LOGE(OP_NAME, "Failed to get win content.");
            return;
        }
    }
    // Write to bin file
    if (DumpToFile(std::string(acldumpGetPath(acldumpType::AIC_ERR_BRIEF_DUMP)), GenDumpFileName(args, op), deviceId,
                   winContent.data()) != 0) {
        OP_LOGE(OP_NAME, "Failed to get win content.");
    }
}
} // namespace Mc2Exception
#endif

#endif