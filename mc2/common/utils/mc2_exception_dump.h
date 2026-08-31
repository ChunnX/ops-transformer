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
#include "../op_kernel/mc2_moe_context.h"
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
#include <map>
#include "version/runtime_version.h"
#include "version/metadef_version.h"
#include <dlfcn.h>
#include <cstring>
#include <mutex>
#include <algorithm>
#include <cctype>
#include <memory>
#include <new>
#include <limits>
#include <unistd.h>
#include "./mc2_hcom_topo_info.h"
#include "../op_kernel/mc2_exception_dump.h"

// 支持 9.0.0 版本（90000000）, 当版本带着 beta, alpha, rc 时, rts 接口获取到的版本号会在 90000000 基础上减去一个对应的
// weight 值, 其下限为 9.0.0-alpha, 对应版本号 90000000 - 300 = 89999700
#define EXCEPTION_DUMP_SUPPORT_VERSION 89999700

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
namespace Mc2Exception {

const std::string OP_NAME = "Mc2Exception";
const uint32_t WIN_SIZE = 1024U * 1024U;
const uint32_t MS_WIDTH = 3U;
const uint32_t MS_PER_S = 1000U;
const mode_t FILE_MODE = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
constexpr size_t MAX_GROUP_NAME_LENGTH = 128UL;

const std::map<std::string, std::string> MC2_OP_CONTEXT = {{"MoeDistributeDispatchV3", "Mc2MoeContext"},
                                                           {"MoeDistributeCombineV3", "Mc2MoeContext"},
                                                           {"MoeDistributeDispatchV2", "HcclCombinOpParam"},
                                                           {"MoeDistributeCombineV2", "HcclCombinOpParam"},
                                                           {"MegaMoe", "Mc2MoeContext"}};

class MC2GroupNameManager {
public:
    static MC2GroupNameManager &GetInstance()
    {
        static MC2GroupNameManager instance;
        return instance;
    }
    void SetGroupName(const char *groupName)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (groupName == nullptr || strnlen(groupName, MAX_GROUP_NAME_LENGTH) == 0 ||
            strnlen(groupName, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH) {
            OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "groupName", "invalid", "valid group name");
            return;
        }
        groupName_ = std::string(groupName);
    }
    const char *GetGroupName()
    {
        return groupName_.empty() ? nullptr : groupName_.c_str();
    }

private:
    MC2GroupNameManager() = default;
    std::string groupName_;
    std::mutex mutex_;
};

// 函数指针类型定义
typedef aclError (*aclrtGetArgsFromExceptionInfo_t)(aclrtExceptionInfo *info, void **args, uint32_t *argsSize);
typedef const char *(*acldumpGetPath_t)(acldumpType dumpType);

// 动态加载clrtGetArgsFromExceptionInfo函数
inline aclrtGetArgsFromExceptionInfo_t GetAclrtGetArgsFromExceptionInfoFunc()
{
    static aclrtGetArgsFromExceptionInfo_t func = nullptr;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (func == nullptr) {
        func = reinterpret_cast<aclrtGetArgsFromExceptionInfo_t>(dlsym(RTLD_DEFAULT, "aclrtGetArgsFromExceptionInfo"));
    }
    if (func == nullptr) {
        // 直接从 libacl_rt.so 加载符号
        void *handle = dlopen("libacl_rt.so", RTLD_NOLOAD | RTLD_LAZY);
        if (handle != nullptr) {
            func = (aclrtGetArgsFromExceptionInfo_t)dlsym(handle, "aclrtGetArgsFromExceptionInfo");
            if (func == nullptr) {
                OP_LOGE(OP_NAME, "dlsym aclrtGetArgsFromExceptionInfo failed: %s", dlerror());
                dlclose(handle);
            }
        } else {
            OP_LOGE(OP_NAME, "dlopen failed: %s", dlerror());
        }
    }
    return func;
}

// 动态加载acldumpGetPath函数
inline acldumpGetPath_t GetAcldumpGetPathFunc()
{
    static acldumpGetPath_t func = nullptr;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (func == nullptr) {
        func = reinterpret_cast<acldumpGetPath_t>(dlsym(RTLD_DEFAULT, "acldumpGetPath"));
    }
    // Do not cache a failed lookup: the dump library may be loaded after an earlier lookup.
    if (func == nullptr) {
        // 尝试加载ascend_dump.so
        void *handle = dlopen("libascend_dump.so", RTLD_NOLOAD | RTLD_LAZY);
        if (handle != nullptr) {
            func = (acldumpGetPath_t)dlsym(handle, "acldumpGetPath");
            if (func == nullptr) {
                OP_LOGE(OP_NAME, "dlsym acldumpGetPath failed: %s", dlerror());
                dlclose(handle);
            } else {
                OP_LOGD(OP_NAME, "Successfully loaded acldumpGetPath function");
            }
        } else {
            OP_LOGE(OP_NAME, "dlopen libascend_dump.so failed: %s", dlerror());
        }
    }
    return func;
}

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

inline int DumpToFile(std::string dir, std::string name, uint32_t id, void *buf, size_t bufSize = WIN_SIZE)
{
    // check if dir and buf valid
    if (IsStrEmpty(dir) || IsStrEmpty(name)) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "dir or name");
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
    ssize_t ret = write(fd, buf, bufSize);
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

const uint32_t XN_ADDR_DUMP_SIZE = 64U * 64U * 8U;
const uint32_t CKE_ADDR_DUMP_SIZE = 64U * 2U * 8U;

inline std::string GenDumpFileNameWithPrefix(aclrtExceptionInfo *args, const char *op, const char *prefix)
{
    std::stringstream ss;

    uint32_t streamId = aclrtGetStreamIdFromExceptionInfo(args);
    uint32_t taskId = aclrtGetTaskIdFromExceptionInfo(args);
    std::string ts = GetTimestampWithMilliseconds();

    ss << prefix << op << "." << streamId << "." << taskId << "." << ts;

    return ss.str();
}

inline bool EnsureDumpDirectory(const std::string &dir)
{
    // The exception callback can run before RTS creates the per-device directory.
    for (size_t end = 1; end <= dir.size(); ++end) {
        if (end != dir.size() && dir[end] != '/') {
            continue;
        }
        const std::string part = dir.substr(0, end);
        if (mkdir(part.c_str(), S_IRWXU | S_IRGRP | S_IXGRP) != 0) {
            const int err = errno;
            struct stat info {};
            if (err != EEXIST || stat(part.c_str(), &info) != 0 || !S_ISDIR(info.st_mode)) {
                OP_LOGE(OP_NAME, "Cannot create dump directory %s, errno=%d(%s).",
                        part.c_str(), err, strerror(err));
                return false;
            }
        }
    }
    return !dir.empty();
}

inline int OpenDumpFile(const std::string &dir, const std::string &name, uint32_t id)
{
    if (IsStrEmpty(dir) || IsStrEmpty(name)) {
        OP_LOGE(OP_NAME, "Dump directory or file name is empty.");
        return -1;
    }
    const std::string rankPath = dir + "/" + std::to_string(id);
    if (!EnsureDumpDirectory(rankPath)) {
        return -1;
    }
    const std::string path = rankPath + "/" + name;
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, FILE_MODE);
    if (fd < 0) {
        const int err = errno;
        OP_LOGE(OP_NAME, "Failed to open dump file %s, errno=%d(%s).", path.c_str(), err, strerror(err));
    } else {
        OP_LOGE(OP_NAME, "Start to dump file. The dump path is %s", path.c_str());
    }
    return fd;
}

inline bool WriteDumpBytes(int fd, const void *buf, size_t size)
{
    const auto *data = static_cast<const uint8_t *>(buf);
    size_t written = 0;
    while (written < size) {
        const size_t chunkSize = std::min<size_t>(size - written, WIN_SIZE);
        const ssize_t ret = write(fd, data + written, chunkSize);
        if (ret < 0 && errno == EINTR) {
            continue;
        }
        if (ret <= 0) {
            const int err = ret < 0 ? errno : EIO;
            OP_LOGE(OP_NAME, "Dump write failed after %zu/%zu bytes, errno=%d(%s).",
                    written, size, err, strerror(err));
            return false;
        }
        written += static_cast<size_t>(ret);
    }
    return true;
}

inline int DumpToFileWithSize(const std::string &dir, const std::string &name, uint32_t id,
                              const void *buf, size_t size)
{
    if (buf == nullptr && size != 0) {
        return -1;
    }
    const int fd = OpenDumpFile(dir, name, id);
    if (fd < 0) {
        return -1;
    }
    if (!WriteDumpBytes(fd, buf, size)) {
        close(fd);
        return -1;
    }
    if (close(fd) != 0) {
        OP_LOGE(OP_NAME, "Failed to close dump file %s, errno=%d.", name.c_str(), errno);
        return -1;
    }
    OP_LOGE(OP_NAME, "Dump to file %s done, size=%zu.", name.c_str(), size);
    return 0;
}

inline int DumpDeviceMemoryToFile(const std::string &dir, const std::string &name, uint32_t id,
                                  uint64_t srcAddr, uint64_t size)
{
    if (srcAddr == 0 || size == 0 || size > std::numeric_limits<uint64_t>::max() - srcAddr) {
        OP_LOGE(OP_NAME, "Invalid device dump range: addr=0x%llx, size=%llu.",
                static_cast<unsigned long long>(srcAddr), static_cast<unsigned long long>(size));
        return -1;
    }
    // A matrix segment can exceed 4 GiB. Bound host memory and keep the length 64-bit.
    const size_t bufferSize = static_cast<size_t>(std::min<uint64_t>(size, WIN_SIZE));
    std::unique_ptr<uint8_t[]> buffer(new (std::nothrow) uint8_t[bufferSize]);
    if (buffer == nullptr) {
        OP_LOGE(OP_NAME, "Cannot allocate %zu bytes for device dump.", bufferSize);
        return -1;
    }
    int fd = -1;
    for (uint64_t offset = 0; offset < size;) {
        const size_t chunkSize = static_cast<size_t>(std::min<uint64_t>(size - offset, bufferSize));
        const aclError ret = aclrtMemcpy(buffer.get(), bufferSize, reinterpret_cast<void *>(srcAddr + offset),
                                         chunkSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "D2H dump failed: file=%s, device=%u, addr=0x%llx, offset=%llu, size=%zu, ret=%d. "
                    "Any existing file contains only the preceding chunks.", name.c_str(), id,
                    static_cast<unsigned long long>(srcAddr + offset), static_cast<unsigned long long>(offset),
                    chunkSize, ret);
            if (fd >= 0) {
                close(fd);
            }
            return -1;
        }
        // Do not create an empty "successful" dump when the first D2H copy fails.
        if (fd < 0) {
            fd = OpenDumpFile(dir, name, id);
            if (fd < 0) {
                return -1;
            }
        }
        if (!WriteDumpBytes(fd, buffer.get(), chunkSize)) {
            close(fd);
            return -1;
        }
        offset += chunkSize;
    }
    if (close(fd) != 0) {
        OP_LOGE(OP_NAME, "Failed to close device dump %s, errno=%d.", name.c_str(), errno);
        return -1;
    }
    OP_LOGE(OP_NAME, "Dump to file %s done, size=%llu.", name.c_str(), static_cast<unsigned long long>(size));
    return 0;
}

inline int ProcessArgsForA5(uint64_t argsAddr, std::vector<uint8_t> &winBuf, const char *op)
{
    // Get hccl context from its addr
    if (op == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "op");
        return -1;
    }
    void *winAddr = nullptr;
    aclError ret = ACL_SUCCESS;
    auto is_support_op = MC2_OP_CONTEXT.find(op);
    if (is_support_op == MC2_OP_CONTEXT.end()) {
        // 不支持的算子
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "op", op, "The value of op must be within the supported range.");
        return -1;
    }
    const std::string &context_type = is_support_op->second;
    // V2分支
    if (context_type == "HcclCombinOpParam") {
        std::vector<uint8_t> hcclArgs(sizeof(HcclCombinOpParam), 0);
        ret = aclrtMemcpy(hcclArgs.data(), sizeof(HcclCombinOpParam), (void *)argsAddr, sizeof(HcclCombinOpParam),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy HcclCombinOpParam from device to host failed. ret = %d", ret);
            return -1;
        }
        HcclCombinOpParam *winContext = reinterpret_cast<HcclCombinOpParam *>(hcclArgs.data());
        if (winContext == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "winContext");
            return -1;
        }
        OP_LOGD(OP_NAME, "Get winContext from args. rankId=%u, rankDim=%u", winContext->rankId, winContext->rankDim);
        if (winContext->rankId >= HCCL_MTE_MAX_RANK_NUM) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "rankId", std::to_string(winContext->rankId).c_str(),
                                                  "The value of rankId must be less than HCCL_MTE_MAX_RANK_NUM");
            return -1;
        }
        winAddr = reinterpret_cast<void *>(winContext->windowsIn[winContext->rankId]);
    } else if (context_type == "Mc2MoeContext") {
        // V3分支
        std::vector<uint8_t> hcclArgs(sizeof(Mc2Aclnn::Mc2MoeContext), 0);
        ret = aclrtMemcpy(hcclArgs.data(), sizeof(Mc2Aclnn::Mc2MoeContext), (void *)argsAddr,
                          sizeof(Mc2Aclnn::Mc2MoeContext), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy Mc2MoeContext from device to host failed. ret = %d", ret);
            return -1;
        }
        Mc2Aclnn::Mc2MoeContext *winContext = reinterpret_cast<Mc2Aclnn::Mc2MoeContext *>(hcclArgs.data());
        if (winContext == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "winContext");
            return -1;
        }
        OP_LOGD(OP_NAME, "Get winContext from args. rankId=%u", winContext->epRankId);
        if (winContext->epRankId >= Mc2Aclnn::HCCL_MAX_RANK_SIZE) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "epRankId", std::to_string(winContext->epRankId).c_str(),
                                                  "The value of epRankId must be less than HCCL_MAX_RANK_SIZE");
            return -1;
        }
        winAddr = reinterpret_cast<void *>(winContext->epHcclBuffer_[winContext->epRankId]);
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "context_type", context_type.c_str(),
                                              "The value of context_type must be within the supported range.");
        return -1;
    }
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
    CommContextForDump *winContext = reinterpret_cast<CommContextForDump *>(hcclArgs.data());
    if (winContext == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(OP_NAME, "winContext");
        return -1;
    }
    OP_LOGD(OP_NAME, "Get winContext from args. rankId=%u", winContext->epRankId);

    void *winAddr = reinterpret_cast<void *>(winContext->epHcclBufffer_[winContext->epRankId]);
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

inline int ProcessArgsForA2(const char *groupName, std::vector<uint8_t> &winBuf)
{
    if (groupName == nullptr || strnlen(groupName, MAX_GROUP_NAME_LENGTH) == 0 ||
        strnlen(groupName, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "groupName", groupName,
                                              "The value of groupName must be within the supported range.");
        return -1;
    }
    uint64_t size;
    void *winAddr = nullptr;
    Mc2Hcom::MC2HcomTopology::CommGetHcclBufferByGroup(groupName, &winAddr, &size);
    if (winAddr == nullptr) {
        OP_LOGE(OP_NAME, "Get win addr failed.");
        return -1;
    }
    OP_LOGE(OP_NAME, "HCCL BUFFER SIZE=%luMB. WindowInAddr=%p.", size / (1 * 1024 * 1024) / 2, winAddr);
    auto ret = aclrtMemcpy(winBuf.data(), WIN_SIZE, winAddr, WIN_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy win from device to host failed. ret = %d", ret);
        return -1;
    }
    return 0;
}

// 获取 peermem 基址，用于 MegaMoe 结构化 dump
inline void *GetPeermemBase(const char *socName, uint64_t argsAddr)
{
    OP_LOGD(OP_NAME, "GetPeermemBase: argsAddr=0x%lx", argsAddr);
    std::vector<uint8_t> hcclArgs(sizeof(CommContextForDump), 0);
    if (aclrtMemcpy(hcclArgs.data(), sizeof(CommContextForDump), reinterpret_cast<void *>(argsAddr),
                    sizeof(CommContextForDump), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "GetPeermemBase: aclrtMemcpy CommContextForDump failed");
        return nullptr;
    }
    CommContextForDump *ctx = reinterpret_cast<CommContextForDump *>(hcclArgs.data());
    OP_LOGD(OP_NAME, "GetPeermemBase: epRankId=%u", ctx->epRankId);
    void *winAddr = reinterpret_cast<void *>(ctx->epHcclBufffer_[ctx->epRankId]);
    OP_LOGD(OP_NAME, "GetPeermemBase: peermemBase=%p", winAddr);
    return winAddr;
}

inline int ParseStructuredDump(uint64_t argsAddr, std::vector<uint8_t> &winBuf)
{
    // Phase 1: 从 device 拷贝 Header 到 host（读取固定区域大小的前 sizeof(Header) 字节）
    OP_LOGD(OP_NAME, "ParseStructuredDump Phase1: copy Header from device addr=0x%lx, regionSize=%zu", argsAddr,
            MC2ExceptionDump::DUMP_HEADER_REGION_SIZE);
    MC2ExceptionDump::Header hdr;
    aclError ret = aclrtMemcpy(&hdr, sizeof(MC2ExceptionDump::Header), reinterpret_cast<void *>(argsAddr),
                               sizeof(MC2ExceptionDump::Header), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy Header failed. ret=%d", ret);
        return -1;
    }
    OP_LOGD(OP_NAME, "Phase1: headerMagic=0x%x opType=%d stageNum=%d tilingDataSize=%u numBlocks=%u dumpCount=%lu",
            hdr.headerMagic, hdr.opType, hdr.stageNum, hdr.tilingDataSize, hdr.numBlocks, hdr.dumpCount);
    if (hdr.headerMagic != MC2ExceptionDump::HEADER_MAGIC) {
        OP_LOGE(OP_NAME, "headerMagic mismatch: 0x%x != 0x%x, ExceptionDump not enabled", hdr.headerMagic,
                MC2ExceptionDump::HEADER_MAGIC);
        return -1;
    }

    uint64_t dumpCount = hdr.dumpCount;
    if (dumpCount > MC2ExceptionDump::MAX_DUMP_ENTRIES) {
        OP_LOGE(OP_NAME, "dumpCount=%lu > MAX_DUMP_ENTRIES=%d, truncating", dumpCount,
                MC2ExceptionDump::MAX_DUMP_ENTRIES);
        dumpCount = MC2ExceptionDump::MAX_DUMP_ENTRIES;
    }
    uint32_t tilingDataSize = hdr.tilingDataSize;
    if (tilingDataSize > MC2ExceptionDump::MAX_DUMP_TILING_SIZE) {
        OP_LOGE(OP_NAME, "tilingDataSize=%u > MAX=%zu, truncating", tilingDataSize,
                MC2ExceptionDump::MAX_DUMP_TILING_SIZE);
        tilingDataSize = MC2ExceptionDump::MAX_DUMP_TILING_SIZE;
    }
    uint32_t numBlocks = hdr.numBlocks;
    if (numBlocks == 0) {
        OP_LOGE(OP_NAME, "numBlocks=0, Header not initialized");
        return -1;
    }

    size_t headerSize = sizeof(MC2ExceptionDump::Header);
    size_t blockStageSize = sizeof(uint64_t) + hdr.stageNum * sizeof(uint64_t);
    size_t blockStageTotalSize = static_cast<size_t>(numBlocks) * blockStageSize;
    size_t dumpParamsSize = static_cast<size_t>(dumpCount) * sizeof(MC2ExceptionDump::DumpParams);
    OP_LOGD(OP_NAME, "Phase1: headerSize=%zu blockStageSize=%zu blockStageTotalSize=%zu dumpParamsSize=%zu", headerSize,
            blockStageSize, blockStageTotalSize, dumpParamsSize);

    // Device 内存偏移计算（使用固定区域大小）
    uint64_t tilingOffset = argsAddr + MC2ExceptionDump::DUMP_HEADER_REGION_SIZE;
    uint64_t blockStageOffset = tilingOffset + MC2ExceptionDump::DUMP_TILING_REGION_SIZE;
    uint64_t dumpParamsOffset = blockStageOffset + MC2ExceptionDump::DUMP_BLOCK_STAGE_REGION_SIZE;
    OP_LOGD(OP_NAME, "Phase1: tilingOffset=0x%lx blockStageOffset=0x%lx dumpParamsOffset=0x%lx", tilingOffset,
            blockStageOffset, dumpParamsOffset);

    // Phase 2: 从 device 拷贝 DumpParams 到临时 host buffer，计算 dump 数据总大小
    OP_LOGD(OP_NAME, "Phase2: copy DumpParams from device addr=0x%lx, size=%zu", dumpParamsOffset, dumpParamsSize);
    std::vector<uint8_t> dumpParamsBuf(dumpParamsSize, 0);
    ret = aclrtMemcpy(dumpParamsBuf.data(), dumpParamsSize, reinterpret_cast<void *>(dumpParamsOffset), dumpParamsSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy DumpParams failed. ret=%d", ret);
        return -1;
    }

    size_t totalDumpDataSize = 0;
    for (uint64_t i = 0; i < dumpCount; i++) {
        const MC2ExceptionDump::DumpParams *dp = reinterpret_cast<const MC2ExceptionDump::DumpParams *>(
            dumpParamsBuf.data() + i * sizeof(MC2ExceptionDump::DumpParams));
        if (dp->dumpAddr != 0) {
            OP_LOGD(OP_NAME, "Phase2: dumpEntry[%lu] addr=0x%lx blockCount=%zu blockLen=%zu srcStride=%zu", i,
                    dp->dumpAddr, dp->blockCount, dp->blockLen, dp->srcStride);
            totalDumpDataSize += dp->blockCount * dp->blockLen;
        } else {
            OP_LOGD(OP_NAME, "Phase2: dumpEntry[%lu] addr=0, skipped", i);
        }
    }
    OP_LOGD(OP_NAME, "Phase2: totalDumpDataSize=%zu", totalDumpDataSize);

    // Phase 3: 一次性 resize winBuf，将所有数据连续拷入
    // 布局: [Header][Tiling][BlockStage[]][DumpParams[]][dump data...]
    size_t totalSize = headerSize + tilingDataSize + blockStageTotalSize + dumpParamsSize + totalDumpDataSize;
    OP_LOGD(OP_NAME, "Phase3: totalSize=%zu, resizing winBuf", totalSize);
    winBuf.resize(totalSize, 0);
    size_t offset = 0;

    // 3a. Header
    OP_LOGD(OP_NAME, "Phase3a: copy Header to winBuf offset=%zu, size=%zu", offset, headerSize);
    memcpy(winBuf.data() + offset, &hdr, headerSize);
    offset += headerSize;

    // 3b. Tiling（从固定区域偏移读取，按实际大小拷贝）
    OP_LOGD(OP_NAME, "Phase3b: copy Tiling from device addr=0x%lx, size=%zu, offset=%zu", tilingOffset, tilingDataSize,
            offset);
    if (tilingDataSize > 0) {
        ret = aclrtMemcpy(winBuf.data() + offset, tilingDataSize, reinterpret_cast<void *>(tilingOffset),
                          tilingDataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy Tiling failed. ret=%d", ret);
        }
    }
    offset += tilingDataSize;

    // 3c. BlockStage（从固定区域偏移读取，按实际大小拷贝）
    OP_LOGD(OP_NAME, "Phase3c: copy BlockStage from device addr=0x%lx, size=%zu, offset=%zu", blockStageOffset,
            blockStageTotalSize, offset);
    if (blockStageTotalSize > 0) {
        ret = aclrtMemcpy(winBuf.data() + offset, blockStageTotalSize, reinterpret_cast<void *>(blockStageOffset),
                          blockStageTotalSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy BlockStage failed. ret=%d", ret);
        }
    }
    offset += blockStageTotalSize;

    // 3d. DumpParams（已在 host）
    OP_LOGD(OP_NAME, "Phase3d: copy DumpParams to winBuf offset=%zu, size=%zu", offset, dumpParamsSize);
    memcpy(winBuf.data() + offset, dumpParamsBuf.data(), dumpParamsSize);
    offset += dumpParamsSize;

    // 3e. 逐条搬运 dump 数据（从 device 各 dumpAddr 搬到 winBuf 末尾连续区域）
    for (uint64_t i = 0; i < dumpCount; i++) {
        const MC2ExceptionDump::DumpParams *dp = reinterpret_cast<const MC2ExceptionDump::DumpParams *>(
            dumpParamsBuf.data() + i * sizeof(MC2ExceptionDump::DumpParams));
        if (dp->dumpAddr == 0) {
            continue;
        }
        OP_LOGD(OP_NAME, "Phase3e: dumpEntry[%lu] addr=0x%lx blockCount=%zu blockLen=%zu srcStride=%zu offset=%zu", i,
                dp->dumpAddr, dp->blockCount, dp->blockLen, dp->srcStride, offset);
        for (size_t b = 0; b < dp->blockCount; b++) {
            void *srcAddr = reinterpret_cast<void *>(dp->dumpAddr + b * dp->srcStride);
            ret = aclrtMemcpy(winBuf.data() + offset, dp->blockLen, srcAddr, dp->blockLen, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                OP_LOGE(OP_NAME, "aclrtMemcpy dump entry %lu block %zu failed. ret=%d", i, b, ret);
                break;
            }
            offset += dp->blockLen;
        }
    }

    OP_LOGE(OP_NAME, "Structured dump completed: totalSize=%zu dumpCount=%lu winBufSize=%zu", totalSize, dumpCount,
            winBuf.size());
    return 0;
}

inline void Mc2ExceptionImpl(aclrtExceptionInfo *args, void *userdata, const char *op)
{
    const char *socName = aclrtGetSocName();
    if ((std::strstr(socName, "Ascend950") == nullptr) && (std::strstr(socName, "Ascend910_93") == nullptr) &&
        (std::strstr(socName, "Ascend910B") == nullptr)) {
        OP_LOGE(OP_NAME, "The soc version is %s, skip dump process", socName);
        return;
    }

    OP_LOGD(OP_NAME, "Start to handle mc2 exception: soc=%s op=%s", socName, op);

    // Get addr of hccl context from ExceptionInfo
    void *devArgsPtr = nullptr;
    uint32_t devArgsLen = 0;

    auto aclrtGetArgsFromExceptionInfoFunc = GetAclrtGetArgsFromExceptionInfoFunc();
    if (aclrtGetArgsFromExceptionInfoFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load aclrtGetArgsFromExceptionInfo function.");
        return;
    }

    auto ret = aclrtGetArgsFromExceptionInfoFunc(args, &devArgsPtr, &devArgsLen);
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
    if (std::strstr(op, "MegaMoe") != nullptr) {
        argsOffset = sizeof(uint64_t);
    }
    ret =
        aclrtMemcpy(&argsAddr, sizeof(uint64_t), devArgsPtr + argsOffset, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy address of args failed. ret=%d", ret);
        return;
    }
    OP_LOGD(OP_NAME, "argsAddr=0x%lx (argsOffset=%lu)", argsAddr, argsOffset);

    // Get win content
    std::vector<uint8_t> winContent;
    bool isMegaMoe = (std::strstr(op, "MegaMoe") != nullptr);
    bool isA2orA3 = (std::strstr(socName, "Ascend910_93") != nullptr || std::strstr(socName, "Ascend910B") != nullptr);
    OP_LOGD(OP_NAME, "isMegaMoe=%d isA2orA3=%d", isMegaMoe, isA2orA3);

    if (isMegaMoe && isA2orA3) {
        // MegaMoe on A2/A3: 结构化 dump（从 device 按 3 阶段搬运到 host）
        OP_LOGD(OP_NAME, "MegaMoe structured dump: getting peermem base");
        void *peermemBase = GetPeermemBase(socName, argsAddr);
        if (peermemBase == nullptr) {
            OP_LOGE(OP_NAME, "Failed to get peermem base.");
            return;
        }
        if (ParseStructuredDump(reinterpret_cast<uint64_t>(peermemBase), winContent) != 0) {
            OP_LOGE(OP_NAME, "Failed to parse structured dump.");
            return;
        }
    } else {
        // 其他场景: 原始 1MB raw dump
        OP_LOGD(OP_NAME, "Raw dump: winContent resize to WIN_SIZE=%u", WIN_SIZE);
        winContent.resize(WIN_SIZE, 0);
        if (std::strstr(socName, "Ascend910_93") != nullptr) {
            if (ProcessArgsForA3(argsAddr, winContent) != 0) {
                OP_LOGE(OP_NAME, "Failed to get win content.");
                return;
            }
        } else if (std::strstr(socName, "Ascend950") != nullptr) {
            if (ProcessArgsForA5(argsAddr, winContent, op) != 0) {
                OP_LOGE(OP_NAME, "Failed to get win content.");
                return;
            }
        } else if (std::strstr(socName, "Ascend910B") != nullptr) {
            const char *groupName = MC2GroupNameManager::GetInstance().GetGroupName();
            if (ProcessArgsForA2(groupName, winContent) != 0) {
                OP_LOGE(OP_NAME, "Failed to get win content.");
                return;
            }
        }
    }

    // 获取acldumpGetPath函数指针
    auto acldumpGetPathFunc = GetAcldumpGetPathFunc();
    if (acldumpGetPathFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load acldumpGetPath function, skip dump");
        return;
    }

    // 获取dump路径
    const char *dumpPath = acldumpGetPathFunc(acldumpType::AIC_ERR_BRIEF_DUMP);
    if (dumpPath == nullptr) {
        OP_LOGE(OP_NAME, "acldumpGetPath returned NULL");
        return;
    }
    OP_LOGD(OP_NAME, "dumpPath=%s, winContent.size()=%zu", dumpPath, winContent.size());

    // Write to bin file
    std::string fileName = GenDumpFileName(args, op);
    OP_LOGD(OP_NAME, "Writing dump file: deviceId=%u fileName=%s bufSize=%zu", deviceId, fileName.c_str(),
            winContent.size());
    if (DumpToFile(std::string(dumpPath), fileName, deviceId, winContent.data(), winContent.size()) != 0) {
        OP_LOGE(OP_NAME, "Failed to dump win content.");
    }
}

inline void Mc2ExceptionImplTmp(aclrtExceptionInfo *args, void *userdata, const char *op)
{
    const char* socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        OP_LOGE(OP_NAME, "The soc version is %s, skip dump process", socName);
        return;
    }

    OP_LOGD(OP_NAME, "Start to handle mc2 exception and dump xn/cke info.");

    // 1. 动态加载 aclrtGetArgsFromExceptionInfo 函数
    auto aclrtGetArgsFromExceptionInfoFunc = GetAclrtGetArgsFromExceptionInfoFunc();
    if (aclrtGetArgsFromExceptionInfoFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load aclrtGetArgsFromExceptionInfo function.");
        return;
    }

    // 2. 调用获取函数，拿到 devArgsPtr 和 devArgsLen
    void* devArgsPtr = nullptr;
    uint32_t devArgsLen = 0;
    auto ret = aclrtGetArgsFromExceptionInfoFunc(args, &devArgsPtr, &devArgsLen);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtGetArgsFromExceptionInfo failed. ret=%d", ret);
        return;
    }

    // 3. 获取报错的 deviceId
    uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    OP_LOGD(OP_NAME, "Get context from args. deviceId=%u, devArgsAddr=%p, devArgsLen=%u", deviceId, devArgsPtr,
            devArgsLen);

    // 4. args 首地址中的内容即为 hccl context 的 device 地址，做一次 D2H copy（无 argsOffset）
    uint64_t argsAddr = 0;
    ret = aclrtMemcpy(&argsAddr, sizeof(uint64_t), devArgsPtr, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy address of args failed. ret=%d", ret);
        return;
    }

    // 5. 根据 hccl context 的 device 地址，做一次 D2H copy，大小为 HcclCombinOpParam
    std::vector<uint8_t> hcclArgs(sizeof(HcclCombinOpParam), 0);
    ret = aclrtMemcpy(hcclArgs.data(), sizeof(HcclCombinOpParam), (void *)argsAddr, sizeof(HcclCombinOpParam),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy HcclCombinOpParam from device to host failed. ret = %d", ret);
        return;
    }
    HcclCombinOpParam* winContext = reinterpret_cast<HcclCombinOpParam *>(hcclArgs.data());
    if (winContext == nullptr) {
        OP_LOGE(OP_NAME, "Cast to winContext failed. HcclCombinOpParam is null.");
        return;
    }
    OP_LOGD(OP_NAME, "Get winContext from args. rankId=%u, rankDim=%u, xnAddr=0x%lx, ckeAddr=0x%lx",
            winContext->rankId, winContext->rankDim, winContext->xnAddr, winContext->ckeAddr);

    // 6. 解析 HcclCombinOpParam，拿到 xnAddr 和 ckeAddr，分别做 D2H copy 并落盘
    auto acldumpGetPathFunc = GetAcldumpGetPathFunc();
    if (acldumpGetPathFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load acldumpGetPath function, skip dump");
        return;
    }
    const char* dumpPath = acldumpGetPathFunc(acldumpType::AIC_ERR_BRIEF_DUMP);
    if (dumpPath == nullptr) {
        OP_LOGE(OP_NAME, "acldumpGetPath returned NULL");
        return;
    }

    // xnAddr: 64*64*8 bytes
    if (winContext->xnAddr == 0) {
        OP_LOGE(OP_NAME, "xnAddr is 0, skip xn dump.");
    } else {
        std::vector<uint8_t> xnBuf(XN_ADDR_DUMP_SIZE, 0);
        ret = aclrtMemcpy(xnBuf.data(), XN_ADDR_DUMP_SIZE, (void *)winContext->xnAddr, XN_ADDR_DUMP_SIZE,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy xnAddr from device to host failed. ret = %d", ret);
        } else if (DumpToFileWithSize(std::string(dumpPath), GenDumpFileNameWithPrefix(args, op, "xn_addr_info_"),
                                      deviceId, xnBuf.data(), XN_ADDR_DUMP_SIZE) != 0) {
            OP_LOGE(OP_NAME, "Failed to dump xn addr info.");
        }
    }

    // ckeAddr: 64*2*8 bytes
    if (winContext->ckeAddr == 0) {
        OP_LOGE(OP_NAME, "ckeAddr is 0, skip cke dump.");
    } else {
        std::vector<uint8_t> ckeBuf(CKE_ADDR_DUMP_SIZE, 0);
        ret = aclrtMemcpy(ckeBuf.data(), CKE_ADDR_DUMP_SIZE, (void *)winContext->ckeAddr, CKE_ADDR_DUMP_SIZE,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "aclrtMemcpy ckeAddr from device to host failed. ret = %d", ret);
        } else if (DumpToFileWithSize(std::string(dumpPath), GenDumpFileNameWithPrefix(args, op, "cke_addr_info_"),
                                      deviceId, ckeBuf.data(), CKE_ADDR_DUMP_SIZE) != 0) {
            OP_LOGE(OP_NAME, "Failed to dump cke addr info.");
        }
    }
}

inline uint32_t Mc2DumpWorkspaceSegments(aclrtExceptionInfo *args, const char *op, const std::string &dumpDir,
                                         uint32_t deviceId, uint64_t workspaceGM,
                                         const Mc2Tiling::WorkspaceLayoutInfo &wsLayout)
{
    if (wsLayout.totalSize == 0 || wsLayout.segCount == 0 ||
        wsLayout.segCount > Mc2Tiling::MAX_WORKSPACE_SEGMENTS) {
        OP_LOGE(OP_NAME, "wsLayout invalid: totalSize=%llu, segCount=%u; rebuild matching host/kernel DFX code.",
                static_cast<unsigned long long>(wsLayout.totalSize), wsLayout.segCount);
        return 0;
    }
    // Both the kernel and this callback receive the user base, after LIB_API.
    uint64_t libApiSize = 0;
    bool hasLibApi = false;
    for (uint32_t i = 0; i < wsLayout.segCount; ++i) {
        const auto &seg = wsLayout.segments[i];
        if (seg.type == Mc2Tiling::WS_SEG_LIB_API) {
            if (hasLibApi || seg.offset != 0 || seg.size > wsLayout.totalSize) {
                OP_LOGE(OP_NAME, "Invalid or duplicate LIB_API segment at index %u.", i);
                return 0;
            }
            hasLibApi = true;
            libApiSize = seg.size;
        }
    }
    OP_LOGE(OP_NAME, "workspace user base=0x%llx, libApiSize=%llu, userSize=%llu, segCount=%u",
            static_cast<unsigned long long>(workspaceGM), static_cast<unsigned long long>(libApiSize),
            static_cast<unsigned long long>(wsLayout.totalSize - libApiSize), wsLayout.segCount);

    uint32_t dumpedSegments = 0;
    bool runtimeFound = false;
    // Save the 64KB fault record before potentially multi-GB matrix buffers.
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t i = 0; i < wsLayout.segCount; ++i) {
            const auto &seg = wsLayout.segments[i];
            const bool isRuntime = seg.type == Mc2Tiling::WS_SEG_RUNTIME_INFO;
            if ((pass == 0) != isRuntime || seg.type == Mc2Tiling::WS_SEG_LIB_API ||
                seg.type == Mc2Tiling::WS_SEG_RESERVED || seg.size == 0) {
                continue;
            }
            runtimeFound = runtimeFound || isRuntime;
            if (seg.offset < libApiSize || seg.offset > wsLayout.totalSize ||
                seg.size > wsLayout.totalSize - seg.offset ||
                (isRuntime && seg.size != Mc2Tiling::RUNTIME_INFO_TOTAL_SIZE)) {
                OP_LOGE(OP_NAME, "Invalid workspace seg[%u]: type=%u, offset=%llu, size=%llu.",
                        i, seg.type, static_cast<unsigned long long>(seg.offset),
                        static_cast<unsigned long long>(seg.size));
                continue;
            }
            const uint64_t userOffset = seg.offset - libApiSize;
            if (userOffset > std::numeric_limits<uint64_t>::max() - workspaceGM) {
                OP_LOGE(OP_NAME, "Workspace address overflows for seg[%u].", i);
                continue;
            }
            const uint64_t srcAddr = workspaceGM + userOffset;
            OP_LOGE(OP_NAME, "seg[%u]: type=%u, offset=%llu, userOffset=%llu, size=%llu, srcAddr=0x%llx",
                    i, seg.type, static_cast<unsigned long long>(seg.offset),
                    static_cast<unsigned long long>(userOffset), static_cast<unsigned long long>(seg.size),
                    static_cast<unsigned long long>(srcAddr));
            const std::string prefix = "workspace_seg" + std::to_string(i) + "_type" +
                                       std::to_string(seg.type) + "_";
            const std::string name = GenDumpFileNameWithPrefix(args, op, prefix.c_str());
            if (DumpDeviceMemoryToFile(dumpDir, name, deviceId, srcAddr, seg.size) == 0) {
                ++dumpedSegments;
            } else {
                OP_LOGE(OP_NAME, "Failed to dump workspace seg[%u], type=%u; continuing other segments.",
                        i, seg.type);
            }
        }
    }
    if (!runtimeFound) {
        OP_LOGE(OP_NAME, "No RUNTIME_INFO segment: the running host tiling has not populated the DFX layout.");
    }
    OP_LOGE(OP_NAME, "Workspace dump done: op=%s, deviceId=%u, dumpedSegments=%u.",
            op, deviceId, dumpedSegments);
    return dumpedSegments;
}

inline void Mc2DumpTilingData(aclrtExceptionInfo *args, const char *op, uint32_t tilingGmArgIdx,
                              uint32_t argsOffset = 0U, uint32_t workspaceGmArgIdx = 0U,
                              uint32_t quantModeOffset = 0U,
                              uint32_t wsLayoutOffsetNonQuant = 0U,
                              uint32_t wsLayoutOffsetQuant = 0U,
                              uint32_t tilingSizeNonQuant = 2048U,
                              uint32_t tilingSizeQuant = 2048U)
{
    if (args == nullptr || op == nullptr) {
        return;
    }
    auto getArgsFunc = GetAclrtGetArgsFromExceptionInfoFunc();
    if (getArgsFunc == nullptr) {
        OP_LOGE(OP_NAME, "Failed to load aclrtGetArgsFromExceptionInfo function.");
        return;
    }
    void *devArgsPtr = nullptr;
    uint32_t devArgsLen = 0;
    const aclError argsRet = getArgsFunc(args, &devArgsPtr, &devArgsLen);
    if (argsRet != ACL_SUCCESS || devArgsPtr == nullptr || devArgsLen == 0) {
        OP_LOGE(OP_NAME, "aclrtGetArgsFromExceptionInfo failed: ret=%d, ptr=%p, size=%u.",
                argsRet, devArgsPtr, devArgsLen);
        return;
    }
    const uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    OP_LOGE(OP_NAME, "Tiling dump start: op=%s, devArgsPtr=%p, devArgsLen=%u, tilingGmArgIdx=%u, argsOffset=%u",
            op, devArgsPtr, devArgsLen, tilingGmArgIdx, argsOffset);
    constexpr uint32_t argsDumpSize = 256U;
    const uint32_t actualArgsDumpSize = std::min(devArgsLen, argsDumpSize);
    std::vector<uint8_t> argsBuf(actualArgsDumpSize, 0);
    const aclError copyRet = aclrtMemcpy(argsBuf.data(), actualArgsDumpSize, devArgsPtr,
                                         actualArgsDumpSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (copyRet != ACL_SUCCESS) {
        OP_LOGE(OP_NAME, "aclrtMemcpy args buffer failed: ret=%d.", copyRet);
        return;
    }
    // Read both pointers from the same args snapshot, before any tiling/layout failure.
    auto readArg = [&](uint32_t index, uint64_t &address) {
        const uint64_t offset = static_cast<uint64_t>(argsOffset) + index * sizeof(uint64_t);
        if (offset + sizeof(uint64_t) > argsBuf.size()) {
            OP_LOGE(OP_NAME, "args[%u] at offset=%llu exceeds args snapshot size=%zu.",
                    index, static_cast<unsigned long long>(offset), argsBuf.size());
            return false;
        }
        memcpy(&address, argsBuf.data() + offset, sizeof(address));
        return true;
    };
    uint64_t tilingGM = 0;
    uint64_t workspaceGM = 0;
    const bool hasTiling = readArg(tilingGmArgIdx, tilingGM) && tilingGM != 0;
    if (workspaceGmArgIdx != 0) {
        readArg(workspaceGmArgIdx, workspaceGM);
    }
    OP_LOGE(OP_NAME, "tilingGM=0x%llx (args[%u]), workspaceGM=0x%llx (args[%u])",
            static_cast<unsigned long long>(tilingGM), tilingGmArgIdx,
            static_cast<unsigned long long>(workspaceGM), workspaceGmArgIdx);

    auto dumpPathFunc = GetAcldumpGetPathFunc();
    const char *dumpPath = dumpPathFunc == nullptr ? nullptr : dumpPathFunc(acldumpType::AIC_ERR_BRIEF_DUMP);
    if (dumpPath == nullptr || IsStrEmpty(dumpPath)) {
        OP_LOGE(OP_NAME, "No exception dump path for op=%s; check exception-dump configuration and dump library.", op);
        return;
    }
    const std::string dumpDir(dumpPath);
    OP_LOGE(OP_NAME, "DFX dump directory=%s, deviceId=%u.", dumpDir.c_str(), deviceId);
    if (DumpToFileWithSize(dumpDir, GenDumpFileNameWithPrefix(args, op, "args_info_"), deviceId,
                           argsBuf.data(), argsBuf.size()) != 0) {
        OP_LOGE(OP_NAME, "Failed to dump args info; continuing tiling/workspace.");
    }
    if (!hasTiling) {
        OP_LOGE(OP_NAME, "No tiling pointer; cannot locate workspace layout.");
        return;
    }

    uint32_t wsLayoutOffset = wsLayoutOffsetNonQuant;
    uint32_t tilingDumpSize = tilingSizeNonQuant;
    if (wsLayoutOffsetNonQuant != wsLayoutOffsetQuant || tilingSizeNonQuant != tilingSizeQuant) {
        uint32_t quantMode = 0;
        if (static_cast<uint64_t>(quantModeOffset) + sizeof(quantMode) >
            std::min(tilingSizeNonQuant, tilingSizeQuant)) {
            OP_LOGE(OP_NAME, "quantModeOffset=%u is outside the tiling header.", quantModeOffset);
            return;
        }
        const aclError ret = aclrtMemcpy(&quantMode, sizeof(quantMode),
                                         reinterpret_cast<void *>(tilingGM + quantModeOffset),
                                         sizeof(quantMode), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_NAME, "Cannot read quantMode at offset=%u, ret=%d; cannot select workspace layout.",
                    quantModeOffset, ret);
            return;
        }
        if (quantMode != 0) {
            wsLayoutOffset = wsLayoutOffsetQuant;
            tilingDumpSize = tilingSizeQuant;
        }
        OP_LOGE(OP_NAME, "Quant mode detect: quantMode=%u, wsLayoutOffset=%u, tilingSize=%u.",
                quantMode, wsLayoutOffset, tilingDumpSize);
    }

    // Read the segment table independently. A failed full tiling copy/file must not suppress workspace.
    Mc2Tiling::WorkspaceLayoutInfo wsLayout {};
    bool hasLayout = false;
    if (workspaceGmArgIdx != 0 && workspaceGM != 0) {
        OP_LOGE(OP_NAME, "Workspace dump start: op=%s, wsGmArgIdx=%u, wsLayoutOffset=%u.",
                op, workspaceGmArgIdx, wsLayoutOffset);
        if (wsLayoutOffset != 0 &&
            static_cast<uint64_t>(wsLayoutOffset) + sizeof(wsLayout) <= tilingDumpSize) {
            const aclError ret = aclrtMemcpy(&wsLayout, sizeof(wsLayout),
                                             reinterpret_cast<void *>(tilingGM + wsLayoutOffset),
                                             sizeof(wsLayout), ACL_MEMCPY_DEVICE_TO_HOST);
            hasLayout = ret == ACL_SUCCESS;
            if (!hasLayout) {
                OP_LOGE(OP_NAME, "Cannot read workspace layout: offset=%u, size=%zu, ret=%d.",
                        wsLayoutOffset, sizeof(wsLayout), ret);
            }
        } else {
            OP_LOGE(OP_NAME, "wsLayoutOffset=%u is outside tilingSize=%u.", wsLayoutOffset, tilingDumpSize);
        }
    }

    if (DumpDeviceMemoryToFile(dumpDir, GenDumpFileNameWithPrefix(args, op, "tiling_data_"),
                               deviceId, tilingGM, tilingDumpSize) != 0) {
        OP_LOGE(OP_NAME, "Failed to dump tiling data; continuing independently-read workspace layout.");
    }
    if (workspaceGmArgIdx != 0) {
        if (workspaceGM == 0 || !hasLayout) {
            OP_LOGE(OP_NAME, "Workspace dump unavailable: workspaceGM=0x%llx, layoutReadable=%d.",
                    static_cast<unsigned long long>(workspaceGM), static_cast<int>(hasLayout));
        } else {
            Mc2DumpWorkspaceSegments(args, op, dumpDir, deviceId, workspaceGM, wsLayout);
        }
    }
}

} // namespace Mc2Exception
#endif

#endif
