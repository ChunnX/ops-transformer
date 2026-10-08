// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "low_latency_bindings.hpp"
#include "../op_kernel/low_latency/combine_layout.h"
#include "../op_kernel/low_latency/ep_memory_server_dedup_combine_tiling.h"
#include "../op_kernel/low_latency/launch.hpp"

namespace deepep {
void bind_low_latency_combine(py::module_& module)
{
    module.def("low_latency_combine_tiling", [](int p, int b, int h, int k, int e, const std::string& dtype) {
        CombineDedup::Layout plan{};
        if ((dtype != "bfloat16" && dtype != "float16") || !CombineDedup::BuildLayout(p, b, h, k, e, plan)) {
            throw std::invalid_argument("LL combine exceeds dtype, topology, UB or workspace limits");
        }
        CombineDedup::EpServerDedupCombineTilingData data{};
        data.bs = b;
        data.h = h;
        data.topK = k;
        data.moeExpertNum = e;
        data.dtype = dtype_code(dtype);
        return bytes(data);
    });
    module.def("combine_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e,
                                         const std::string& dtype, uintptr_t stream_address) {
        CombineDedup::Layout plan{};
        if (!stream_address || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes ||
            (dtype != "bfloat16" && dtype != "float16") ||
            !CombineDedup::BuildLayout(session.world(), b, h, k, e, plan)) {
            throw std::invalid_argument("Invalid LL combine dtype, shape, workspace or stream");
        }
        const int64_t c = plan.capacityRows;
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        deepep::EpServerDedupCombineKernelArgs args{};
        args.commArgs = get("comm_args", {sizeof(DispatchDedup::CommArgs)}, "uint8");
        // Python validates rows against the dispatch handle. The device selects
        // rows below the actual receive count; C_B sizes metadata, not expert storage.
        const auto input_shape = tensors["input"].attr("shape").cast<Shape>();
        if (input_shape.size() != 2 || input_shape[0] < 0 || input_shape[1] != h) {
            throw std::invalid_argument("LL combine input must have shape [rows, H]");
        }
        args.expertOut = get("input", input_shape, dtype.c_str());
        args.destinationIndex = get("destination_index", {c, 3}, "int32");
        args.sendCounts = get("send_counts", {e}, "int32");
        args.expertScales = get("source_weights", {b, k}, "float32");
        args.tokenType = get("token_type", {c}, "int32");
        args.relayReadIndex = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMask = get("source_mask", {b}, "int32");
        args.yOut = get("output", {b, h}, dtype.c_str());
        args.tilingData = get("tiling", {sizeof(CombineDedup::EpServerDedupCombineTilingData)}, "uint8");
        session.workspace();
        py::gil_scoped_release release;
        session.invalidate_dispatch_controls();
        combine_server_dedup_kernel_do(reinterpret_cast<void*>(stream_address), &args);
    });
}
} // namespace deepep
