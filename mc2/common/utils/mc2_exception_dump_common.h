#ifndef MC2_EXCEPTION_DUMP_COMMON_H
#define MC2_EXCEPTION_DUMP_COMMON_H

#include "acl/acl.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "acl/acl_dump.h"
#include "mc2_log.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include "version/runtime_version.h"
#include "version/metadef_version.h"

#define EXCEPTION_DUMP_SUPPORT_VERSION 89999700

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION

namespace Mc2Exception {

const std::string COMMON_NAME = "Mc2ExceptionCommon";
const uint32_t MS_WIDTH = 3U;
const uint32_t MS_PER_S = 1000U;
const mode_t FILE_MODE = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

typedef aclError (*aclrtGetArgsFromExceptionInfo_t)(aclrtExceptionInfo *info, void **args, uint32_t *argsSize);
typedef const char *(*acldumpGetPath_t)(acldumpType dumpType);

inline aclrtGetArgsFromExceptionInfo_t GetAclrtGetArgsFromExceptionInfoFunc()
{
    static aclrtGetArgsFromExceptionInfo_t func = nullptr;
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        void *handle = dlopen("libacl_rt.so", RTLD_NOLOAD | RTLD_LAZY);
        if (handle != nullptr) {
            func = (aclrtGetArgsFromExceptionInfo_t)dlsym(handle, "aclrtGetArgsFromExceptionInfo");
            if (func == nullptr) {
                OP_LOGE(COMMON_NAME, "dlsym aclrtGetArgsFromExceptionInfo failed: %s", dlerror());
            }
        } else {
            OP_LOGE(COMMON_NAME, "dlopen failed: %s", dlerror());
        }
    }
    return func;
}

inline acldumpGetPath_t GetAcldumpGetPathFunc()
{
    static acldumpGetPath_t func = nullptr;
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        void *handle = dlopen("libascend_dump.so", RTLD_NOLOAD | RTLD_LAZY);
        if (handle != nullptr) {
            func = (acldumpGetPath_t)dlsym(handle, "acldumpGetPath");
            if (func == nullptr) {
                OP_LOGE(COMMON_NAME, "dlsym acldumpGetPath failed: %s", dlerror());
            }
        } else {
            OP_LOGE(COMMON_NAME, "dlopen libascend_dump.so failed: %s", dlerror());
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

inline std::string GenMoeEpDumpFileName(aclrtExceptionInfo *args, const char *op, uint32_t rank,
                                        const std::string &regionName)
{
    uint32_t streamId = aclrtGetStreamIdFromExceptionInfo(args);
    uint32_t taskId = aclrtGetTaskIdFromExceptionInfo(args);
    std::string ts = GetTimestampWithMilliseconds();
    std::stringstream ss;
    ss << "mc2_exception_info_" << op << "." << streamId << "." << taskId << "." << ts << "."
       << rank << "." << regionName << ".bin";
    return ss.str();
}

inline bool IsStrEmpty(const std::string &str)
{
    return (str.empty() || std::all_of(str.begin(), str.end(), [](unsigned char c) { return std::isspace(c); }));
}

inline int DumpRegionToFile(const std::string &dir, const std::string &name, uint32_t deviceId,
                            void *buf, size_t bufSize)
{
    if (IsStrEmpty(dir) || IsStrEmpty(name) || buf == nullptr || bufSize == 0) {
        OP_LOGE(COMMON_NAME, "Invalid dump params: dir=%s name=%s buf=%p size=%zu",
                dir.c_str(), name.c_str(), buf, bufSize);
        return -1;
    }
    std::string rankPath = dir + "/" + std::to_string(deviceId) + "/";
    std::string path = rankPath + name;
    OP_LOGE(COMMON_NAME, "Dump file: %s (%zu bytes)", path.c_str(), bufSize);
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, FILE_MODE);
    if (fd < 0) {
        OP_LOGE(COMMON_NAME, "Failed to open dump file. errno=%d(%s)", errno, strerror(errno));
        return -1;
    }
    ssize_t ret = write(fd, buf, bufSize);
    if (ret < 0) {
        OP_LOGE(COMMON_NAME, "Failed to write dump file. errno=%d(%s)", errno, strerror(errno));
        close(fd);
        return -1;
    }
    close(fd);
    OP_LOGE(COMMON_NAME, "Dump done: %s", path.c_str());
    return 0;
}

} // namespace Mc2Exception
#endif
#endif
