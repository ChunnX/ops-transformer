// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Adapted from DeepEP; original copyright and license notices follow.

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <pybind11/pybind11.h>
#include <memory>
#include <dlfcn.h>
#include "host/init/shmem_host_init.h"
#include "runtime/session.hpp"

namespace py = pybind11;
namespace deepep {
void bind_low_latency_dispatch(py::module_&);
}
#ifdef DEEPEP_BUILD_LOW_LATENCY_COMBINE
namespace deepep {
void bind_low_latency_combine(py::module_&);
}
#endif
PYBIND11_MODULE(_C, module)
{
    module.attr("runtime_protocol") = 1;
    module.def("shmem_library_path", []() {
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&aclshmemx_init_attr), &info) == 0 || info.dli_fname == nullptr) {
            throw std::runtime_error("Cannot identify the loaded SHMEM library");
        }
        return std::string(info.dli_fname);
    });
    module.def("runtime_busy", &deepep::Session::busy, py::call_guard<py::gil_scoped_release>());
    py::class_<deepep::Session>(module, "Runtime")
        .def(py::init<int, int, uint64_t, const std::string&, unsigned, bool>(), py::arg("rank"), py::arg("size"),
             py::arg("bytes"), py::arg("endpoint"), py::arg("timeout"), py::arg("udma") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("barrier", &deepep::Session::barrier, py::call_guard<py::gil_scoped_release>())
        .def("close", &deepep::Session::close, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("closed", &deepep::Session::closed)
        .def_property_readonly("capacity", &deepep::Session::capacity)
        .def_property_readonly("peer_table", &deepep::Session::peer_table);
#ifdef DEEPEP_BUILD_MEGAMOE
    module.def(
        "create_megamoe_runtime",
        [](int rank, int size, uint64_t bytes, const std::string& endpoint, unsigned timeout) {
            return std::make_unique<deepep::Session>(rank, size, bytes, endpoint, timeout, false, true);
        },
        py::call_guard<py::gil_scoped_release>());
    module.def("megamoe_workspace_address", &deepep::Session::workspace_address);
#endif

    deepep::bind_low_latency_dispatch(module);
#ifdef DEEPEP_BUILD_LOW_LATENCY_COMBINE
    deepep::bind_low_latency_combine(module);
#endif
}
