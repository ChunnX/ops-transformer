#ifndef MC2_GET_HCCL_BUFFER_SIZE_H
#define MC2_GET_HCCL_BUFFER_SIZE_H
#include <cstdint>
#include <cstdlib>
#include <string>
#include "log/log.h"
namespace mc2tiling {

uint64_t GetHcclBufferMaxSize()
{
    uint16_t defaultWindowSize = 200;
    if (getenv(HCCL_BUFFSIZE) == nullptr) {
        OP_LOGD("", "Env HCCL_BUFFSIZE don't set");
    } else {
        try {
        std::string envStr(getenv(HCCL_BUFFSIZE));
        defaultWindowSize = std::stoi(envStr);
        } catch (...) {
        OP_LOGE("",
                "Unknown Exception encountered when parser env HCCL_BUFFERSIZE");
        }
    }
    const uint64_t maxWindowSize =
        static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
    OP_LOGI("", "Get maxWindowSize is %lu", maxWindowSize);
    return maxWindowSize;
}

}  // namespace mc2tiling

#endif  // MC2_GET_HCCL_BUFFER_SIZE_H