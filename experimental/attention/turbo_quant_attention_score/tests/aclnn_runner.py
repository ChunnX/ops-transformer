# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call the installed aclnn operator directly using torch-owned NPU storage."""

import ctypes as C
import os
import time

import torch
import torch_npu  # noqa: F401


class AclnnRunner:
    def __init__(self):
        self.acl = C.CDLL("libascendcl.so", mode=C.RTLD_GLOBAL)
        self.base = C.CDLL("libopapi.so", mode=C.RTLD_GLOBAL)
        path = os.environ.get(
            "TURBOQUANT_OPAPI",
            "/root/turboquant-install/vendors/turboquant_transformer/op_api/lib/libcust_opapi.so",
        )
        self.op = C.CDLL(path, mode=C.RTLD_GLOBAL)
        ptr = C.c_void_p
        self.base.aclCreateTensor.argtypes = [
            C.POINTER(C.c_int64),
            C.c_uint64,
            C.c_int,
            C.POINTER(C.c_int64),
            C.c_int64,
            C.c_int,
            C.POINTER(C.c_int64),
            C.c_uint64,
            ptr,
        ]
        self.base.aclCreateTensor.restype = ptr
        self.base.aclDestroyTensor.argtypes = [ptr]
        self.plan = self.op.aclnnTurboQuantAttentionScoreTensorGetWorkspaceSize
        self.plan.argtypes = [ptr] * 8 + [
            C.c_int64,
            ptr,
            C.POINTER(C.c_uint64),
            C.POINTER(ptr),
        ]
        self.plan.restype = C.c_int
        self.plan_host = self.op.aclnnTurboQuantAttentionScoreGetWorkspaceSize
        self.plan_host.argtypes = self.plan.argtypes
        self.plan_host.restype = C.c_int
        self.base.aclCreateIntArray.argtypes = [C.POINTER(C.c_int64), C.c_uint64]
        self.base.aclCreateIntArray.restype = ptr
        self.base.aclDestroyIntArray.argtypes = [ptr]
        self.execute = self.op.aclnnTurboQuantAttentionScore
        self.execute.argtypes = [ptr, C.c_uint64, ptr, ptr]
        self.execute.restype = C.c_int

    def tensor(self, tensor):
        types = {
            torch.float32: 0,
            torch.float16: 1,
            torch.int32: 3,
            torch.uint8: 4,
            torch.int64: 9,
            torch.bfloat16: 27,
        }
        dims = (C.c_int64 * tensor.ndim)(*tensor.shape)
        strides = (C.c_int64 * tensor.ndim)(*tensor.stride())
        handle = self.base.aclCreateTensor(
            dims,
            tensor.ndim,
            types[tensor.dtype],
            strides,
            0,
            2,
            dims,
            tensor.ndim,
            tensor.data_ptr(),
        )
        if not handle:
            raise RuntimeError("aclCreateTensor failed")
        return handle

    def __call__(self, inputs, bits, max_len, host_lengths=False, output_fill=None):
        output = torch.empty(
            (*inputs[0].shape[:2], max_len), dtype=torch.float32, device="npu"
        )
        if output_fill is not None:
            output.fill_(output_fill)
        handles = [self.tensor(t) for t in [*inputs, output]]
        length_array = None
        if host_lengths:
            lengths = inputs[7].cpu().tolist()
            values = (C.c_int64 * len(lengths))(*lengths)
            length_array = self.base.aclCreateIntArray(values, len(lengths))
        size, executor = C.c_uint64(), C.c_void_p()
        start = time.perf_counter()
        try:
            call_inputs = [*handles[:7], length_array] if host_lengths else handles[:8]
            plan = self.plan_host if host_lengths else self.plan
            status = plan(
                *call_inputs, bits, handles[8], C.byref(size), C.byref(executor)
            )
            if status:
                raise RuntimeError(f"GetWorkspaceSize failed: {status}")
            workspace = (
                torch.empty(size.value, dtype=torch.uint8, device="npu")
                if size.value
                else None
            )
            begin, end = (
                torch.npu.Event(enable_timing=True),
                torch.npu.Event(enable_timing=True),
            )
            begin.record()
            stream = torch.npu.current_stream().npu_stream
            status = self.execute(
                workspace.data_ptr() if workspace is not None else None,
                size.value,
                executor,
                stream,
            )
            if status:
                raise RuntimeError(f"aclnn execution failed: {status}")
            end.record()
            end.synchronize()
            return output, {
                "device_us": begin.elapsed_time(end) * 1000,
                "wall_us": (time.perf_counter() - start) * 1e6,
                "workspace_bytes": size.value,
            }
        finally:
            for handle in handles:
                self.base.aclDestroyTensor(handle)
            if length_array:
                self.base.aclDestroyIntArray(length_array)
