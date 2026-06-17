// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <chrono>
#include <stdexcept>
#include <torch/extension.h>
#include "acl/acl.h"

#define ACL_CHECK(expr) \
    do { \
        aclError _s = (expr); \
        if (_s != ACL_SUCCESS) { \
            throw std::runtime_error("ACL error: " + std::string(__FILE__) + ":" + \
                                     std::to_string(__LINE__) + " code=" + std::to_string(_s)); \
        } \
    } while(0)

class HostPinnedCounter {
public:
    HostPinnedCounter() {
        ACL_CHECK(aclrtMallocHost(&hostPtr_, 4 * sizeof(int64_t)));
        ACL_CHECK(aclrtHostRegisterV2(hostPtr_, 4 * sizeof(int64_t), ACL_HOST_REG_MAPPED));
        ACL_CHECK(aclrtHostGetDevicePointer(hostPtr_, &devPtr_, 0));
        Reset();
    }

    ~HostPinnedCounter() {
        if (hostPtr_) {
            aclrtHostUnregister(hostPtr_);
            aclrtFreeHost(hostPtr_);
            hostPtr_ = nullptr;
            devPtr_ = nullptr;
        }
    }

    void Reset() {
        *reinterpret_cast<volatile int64_t*>(hostPtr_) = -1;
    }

    int64_t SpinWait(int timeoutSecs = 100) {
        auto start = std::chrono::high_resolution_clock::now();
        while (true) {
            int64_t v = *reinterpret_cast<volatile int64_t*>(hostPtr_);
            if (v >= 0) { return v; }
            auto el = std::chrono::duration_cast<std::chrono::seconds>(
                          std::chrono::high_resolution_clock::now() - start).count();
            if (el > timeoutSecs) {
                throw std::runtime_error("MoeEpDispatch CPU recv timeout");
            }
        }
    }

    uintptr_t DevicePtr() const { return reinterpret_cast<uintptr_t>(devPtr_); }
    uintptr_t HostPtr() const { return reinterpret_cast<uintptr_t>(hostPtr_); }

private:
    void* hostPtr_ = nullptr;
    void* devPtr_  = nullptr;
};

namespace op_api {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<HostPinnedCounter>(m, "HostPinnedCounter")
        .def(py::init<>())
        .def("spin_wait", &HostPinnedCounter::SpinWait, py::arg("timeout_secs") = 100)
        .def("reset", &HostPinnedCounter::Reset)
        .def("device_ptr", &HostPinnedCounter::DevicePtr)
        .def("host_ptr", &HostPinnedCounter::HostPtr);
}

} // op_api