#pragma once
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>

// Log level definitions
#define DLOG_DEBUG 3
#define DLOG_INFO 4
#define DLOG_WARN 5
#define DLOG_ERROR 6

// Log module definitions
#define OP 0

// Two-parameter form: OP_LOGD(nodeName, fmt, ...) - nodeName is ignored
#define OP_LOGD(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGI(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGW(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE_WITHOUT_REPORT(nodeName, fmt, ...) do { std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

// CheckLogLevel - always return 0 to disable debug logs
inline int32_t CheckLogLevel(int32_t module, int32_t level) {
    (void)module;
    (void)level;
    return 0;
}

namespace Ops {
namespace Base {
    template<typename T>
    inline std::string ToString(const T& val) {
        std::ostringstream oss;
        oss << val;
        return oss.str();
    }
    
    template<typename T>
    inline std::string ToString(const std::vector<T>& vec) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << vec[i];
        }
        oss << "]";
        return oss.str();
    }
}
}

namespace gert {
class Shape;
}

namespace Ops {
namespace Base {
    template<>
    inline std::string ToString<gert::Shape>(const gert::Shape& shape);
}
}

#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do {                                         \
        if (unlikely(condition)) {               \
            log;                                 \
            return_expr;                         \
        }                                        \
    } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)         \
    if ((ptr) == nullptr)                                 \
    {                                                     \
        std::printf("nullptr error!");                    \
        return ge::GRAPH_SUCCESS;                         \
    }
