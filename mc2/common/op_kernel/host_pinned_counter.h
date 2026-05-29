#ifndef HOST_PINNED_COUNTER_H
#define HOST_PINNED_COUNTER_H

#include <chrono>
#include <stdexcept>

#include "acl/acl.h"

class HostPinnedCounter {
public:
    HostPinnedCounter() {
        ACL_CHECK(aclrtMallocHost(&hostPtr_, sizeof(int32_t),
                                  ACL_HOST_MEM_MAP_DEVICE));
        ACL_CHECK(aclrtGetDevicePointerFromHostPtr(hostPtr_, &devPtr_));
        Reset();
    }

    ~HostPinnedCounter() {
        if (hostPtr_) {
            aclrtFreeHost(hostPtr_);
            hostPtr_ = nullptr;
            devPtr_ = nullptr;
        }
    }

    void Reset() {
        *reinterpret_cast<volatile int32_t*>(hostPtr_) = -1;
    }

    int32_t SpinWait(int timeoutSecs = 100) {
        auto start = std::chrono::high_resolution_clock::now();
        while (true) {
            int32_t v = *reinterpret_cast<volatile int32_t*>(hostPtr_);
            if (v >= 0) return v;
            auto el = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::high_resolution_clock::now() - start).count();
            if (el > timeoutSecs)
                throw std::runtime_error("MoeEpDispatch CPU recv timeout");
        }
    }

    void* DevicePtr() const { return devPtr_; }
    void* HostPtr() const { return hostPtr_; }

private:
    void* hostPtr_ = nullptr;
    void* devPtr_  = nullptr;
};

#define ACL_CHECK(expr) \
    do { \
        aclnnStatus _s = (expr); \
        if (_s != ACLNN_SUCCESS) { \
            throw std::runtime_error("ACL error: " + std::string(__FILE__) + ":" + \
                                     std::to_string(__LINE__) + " code=" + std::to_string(_s)); \
        } \
    } while(0)

#endif  // HOST_PINNED_COUNTER_H