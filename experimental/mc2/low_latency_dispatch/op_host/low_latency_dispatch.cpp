// Copyright (c) 2026, edconeone
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "low_latency_bindings.hpp"

namespace deepep {
void bind_low_latency_dispatch(py::module_& module)
{
    module.def("low_latency_layout", [](int p, int b, int h, int k, int e, const std::string& dtype) {
        // A rejected shape is a Host-only capability result, not a launch failure.
        dtype_code(dtype);
        try {
            auto plan = layout(p, b, h, k, e, dtype == "float8_e4m3fn");
            py::dict result;
            result["capacity_rows"] = plan.capacityRows;
            result["scale_packs"] = (h + 127) / 128;
            return py::object(result);
        } catch (const std::invalid_argument&) {
            return py::object(py::none());
        }
    });
    module.def("low_latency_comm_args", [](Session& session) {
        if (session.capacity() != DispatchDedup::kEpServerDedupPeerBytes || session.world() > 128) {
            throw std::invalid_argument("LL requires exactly 8 GiB of symmetric workspace");
        }
        DispatchDedup::CommArgs args{};
        args.rank = session.rank();
        args.rankSize = session.world();
        args.localRank = session.rank() % 8;
        args.localRankSize = 8;
        args.peerMemBytes = session.capacity();
        args.fftsVal = shmemx_get_ffts_config();
        for (int peer = 0; peer < session.world(); ++peer) {
            args.peerMems[peer] = static_cast<uint8_t*>(aclshmem_ptr(session.workspace(), peer));
            if (args.peerMems[peer] == nullptr || reinterpret_cast<uintptr_t>(args.peerMems[peer]) % 32) {
                throw std::invalid_argument("LL requires aligned MTE mappings for every peer");
            }
        }
        return bytes(args);
    });
    module.def("low_latency_dispatch_tiling", [](int p, int b, int h, int k, int e, const std::string& dtype) {
        const bool fp8 = dtype == "float8_e4m3fn";
        layout(p, b, h, k, e, fp8);
        DispatchDedup::EpServerDedupTilingData data{};
        data.bs = b;
        data.h = h;
        data.topK = k;
        data.moeExpertNum = e;
        data.expertTokenNumsType = 1;
        data.dtype = dtype_code(dtype);
        data.quantMode = fp8 ? 4 : 0;
        data.inputMode = static_cast<uint8_t>(fp8 ? DispatchDedup::EpServerDedupInputMode::PrequantizedFp8Packs :
                                                    DispatchDedup::EpServerDedupInputMode::Plain16);
        data.expandXOutDtype = data.dtype;
        return bytes(data);
    });
    module.def("dispatch_low_latency", [](Session& session, py::dict tensors, int b, int h, int k, int e, int rows,
                                          const std::string& dtype, uintptr_t stream_address) {
        dtype_code(dtype);
        const bool fp8 = dtype == "float8_e4m3fn";
        const auto plan = layout(session.world(), b, h, k, e, fp8);
        const int64_t c = plan.capacityRows;
        if (!stream_address || rows < c || session.capacity() != DispatchDedup::kEpServerDedupPeerBytes) {
            throw std::invalid_argument("Invalid LL output capacity, workspace or stream");
        }
        auto get = [&](const char* name, Shape shape, const char* type) { return tensor(tensors[name], shape, type); };
        deepep::EpServerDedupKernelArgs args{};
        args.commArgs = get("comm_args", {sizeof(DispatchDedup::CommArgs)}, "uint8");
        args.x = get("input", {b, h}, dtype.c_str());
        args.expertIds = get("indices", {b, k}, "int32");
        args.expertScales = get("weights", {b, k}, "float32");
        args.expandXOut = get("output", {rows, h}, dtype.c_str());
        // Both formats occupy four bytes per 128 hidden elements. The kernel
        // transports opaque bytes: FP32 per-128 or packed E8M0 per-32.
        const char* scale_dtype = "int32";
        if (fp8) {
            auto torch = py::module_::import("torch");
            if (tensors["scales"].attr("dtype").is(torch.attr("float32"))) {
                scale_dtype = "float32";
            }
        }
        args.dynamicScalesOut = fp8 ? get("output_scales", {rows, (h + 127) / 128}, scale_dtype) : nullptr;
        args.expertTokenNumsOut = get("expert_recv_counts", {e / session.world()}, "int64");
        args.sendCountsOut = get("send_counts", {e}, "int32");
        args.tokenTypeOut = get("token_type", {c}, "int32");
        args.destinationIndexOut = get("destination_index", {c, 3}, "int32");
        args.relayReadIndexOut = get("relay_read_index", {c, k, 2}, "int32");
        args.sourceMaskOut = get("source_mask", {b}, "int32");
        args.expertScalesOut = get("source_weights", {b, k}, "float32");
        args.tilingData = get("tiling", {sizeof(DispatchDedup::EpServerDedupTilingData)}, "uint8");
        args.inputScales = fp8 ? get("scales", {b, (h + 127) / 128}, scale_dtype) : nullptr;
        session.workspace(); // Validate the owning context before launch.
        py::gil_scoped_release release;
        session.invalidate_dispatch_controls();
        dispatch_server_dedup_kernel_do(reinterpret_cast<void*>(stream_address), &args);
    });
}
} // namespace deepep
