#ifndef OPS_TRANSFORMER_GMM_COMMON_FORMAT_STRING_UTIL_H
#define OPS_TRANSFORMER_GMM_COMMON_FORMAT_STRING_UTIL_H

#include <cstdarg>
#include <string>
#include "securec.h"

constexpr size_t LIMIT_MESSAGE_SIZE = 1024U;
inline std::string FormatString(const char* fmt, ...)
{
    if (fmt == nullptr) {
        return "";
    }
    char buf[LIMIT_MESSAGE_SIZE] = {0};
    va_list args;
    va_start(args, fmt);
    int ret = vsnprintf_s(buf, LIMIT_MESSAGE_SIZE, LIMIT_MESSAGE_SIZE - 1, fmt, args);
    va_end(args);
    if (ret < 0) {
        return "";
    }
    return std::string(buf, static_cast<size_t>(ret));
}

#endif