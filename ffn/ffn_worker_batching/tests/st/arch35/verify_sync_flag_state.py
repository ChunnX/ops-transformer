#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""ACLNN device protocol tests; run with the newly built custom OPP environment.

Checks persistent context across calls and delayed flag publication. No TTK internals
or TEST_MAGIC are used: schedule_context contains real device addresses.
"""
import ctypes as C
import json
import struct
import time
import numpy as np
P, I, Q = (C.c_void_p, C.c_int64, C.c_uint64)

def bind(lib, name, args, result=C.c_int):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = (args, result)
    return fn

def check(code):
    if code:
        raise RuntimeError(f'ACL status {code}')

class Device:

    def __init__(self):
        self.rt = C.CDLL('libascendcl.so', mode=C.RTLD_GLOBAL)
        self.base = C.CDLL('libnnopbase.so', mode=C.RTLD_GLOBAL)
        self.op = C.CDLL('libcust_opapi.so')
        self.malloc = bind(self.rt, 'aclrtMalloc', [C.POINTER(P), Q, C.c_int])
        self.free = bind(self.rt, 'aclrtFree', [P])
        self.copy = bind(self.rt, 'aclrtMemcpy', [P, Q, P, Q, C.c_int])
        self.sync = bind(self.rt, 'aclrtSynchronizeStreamWithTimeout', [P, C.c_int])
        self.query = bind(self.rt, 'aclrtStreamQuery', [P, C.POINTER(C.c_int)])
        self.tensor = bind(self.base, 'aclCreateTensor', [P, Q, C.c_int, P, I, C.c_int, P, Q, P], P)
        self.array = bind(self.base, 'aclCreateIntArray', [P, Q], P)
        self.destroy_tensor = bind(self.base, 'aclDestroyTensor', [P])
        self.destroy_array = bind(self.base, 'aclDestroyIntArray', [P])
        self.getws = bind(self.op, 'aclnnFfnWorkerBatchingGetWorkspaceSize', [P, I, P, I, I, I, I] + [P] * 8 + [C.POINTER(Q), C.POINTER(P)])
        self.run = bind(self.op, 'aclnnFfnWorkerBatching', [P, Q, P, P])
        check(bind(self.rt, 'aclInit', [P])(None))
        check(bind(self.rt, 'aclrtSetDevice', [C.c_int])(0))
        self.stream = P()
        check(bind(self.rt, 'aclrtCreateStream', [C.POINTER(P)])(C.byref(self.stream)))
        self.allocs, self.tensors = ([], [])

    def alloc(self, a):
        p = P()
        check(self.malloc(C.byref(p), a.nbytes, 0))
        self.allocs.append(p)
        self.upload(p, a)
        return p

    def upload(self, p, a):
        check(self.copy(p, a.nbytes, P(a.ctypes.data), a.nbytes, 1))

    def read(self, p, a):
        out = np.empty_like(a)
        check(self.copy(P(out.ctypes.data), out.nbytes, p, out.nbytes, 2))
        return out

    def make_tensor(self, p, a, dtype):
        shape = np.array(a.shape, dtype=np.int64)
        strides = np.array([x // a.itemsize for x in a.strides], dtype=np.int64)
        t = self.tensor(P(shape.ctypes.data), shape.size, dtype, P(strides.ctypes.data), 0, 2, P(shape.ctypes.data), shape.size, p)
        assert t
        self.tensors.append(t)
        return t

    def close(self):
        for t in self.tensors:
            check(self.destroy_tensor(t))
        for p in reversed(self.allocs):
            check(self.free(p))
        check(bind(self.rt, 'aclrtDestroyStream', [P])(self.stream))
        check(bind(self.rt, 'aclrtResetDevice', [C.c_int])(0))
        check(bind(self.rt, 'aclFinalize', [])())

def main():
    dev = Device()
    try:
        A, M, BS, K, H, E = (4, 3, 3, 3, 66, 8)
        Y = A * BS * K
        tokens = (np.arange(A * M * BS * K * H) % 127).astype(np.float16).reshape(A, M, BS, K, H)
        desc = np.zeros((A, M, 2 + BS * K), dtype=np.int32)
        # layer_id 使用非零值，完整描述符比较可以检出清 flag 时误写相邻字段。
        desc[:, :, 1] = 7
        ids = np.arange(A * M * BS * K, dtype=np.int32).reshape(A, M, BS * K) % E
        desc[:, :, 2:] = ids
        td = dev.alloc(tokens)
        ti = dev.alloc(desc)
        ctx = np.zeros(1024, dtype=np.int8)
        for off, val in [(0, A), (4, M), (8, BS), (12, K), (16, E), (20, H * 2)]:
            struct.pack_into('<I', ctx, off, val)
        # 写入真实 aclrtMalloc 设备地址，不设置 TEST_MAGIC；验证生产环境的二级指针寻址。
        for off, val in [(384, ti.value), (392, desc.nbytes), (400, td.value), (408, tokens.nbytes)]:
            struct.pack_into('<Q', ctx, off, val)
        sc = dev.alloc(ctx)
        sct = dev.make_tensor(sc, ctx, 2)
        outs = [np.zeros((Y, H), np.float16), np.zeros((E, 2), np.int64)] + [np.zeros(Y, np.int32) for _ in range(4)] + [np.zeros(Y, np.float32), np.zeros(1, np.int64)]
        ptrs = [dev.alloc(a) for a in outs]
        ts = [dev.make_tensor(p, a, t) for p, a, t in zip(ptrs, outs, [1, 9, 3, 3, 3, 3, 0, 9])]
        shape = np.array([A, BS, K, H], np.int64)
        attr = dev.array(P(shape.ctypes.data), 4)
        reports = []

        def invoke(name, selected, mb, cursor, sync_flag=1, publish=None):
            nonlocal desc, ctx
            before = desc.copy()
            ws, exe = (Q(), P())
            check(dev.getws(sct, E, attr, 0, 1, 1, sync_flag, *ts, C.byref(ws), C.byref(exe)))
            workspace = dev.alloc(np.zeros(max(ws.value, 1), np.uint8))
            check(dev.run(workspace, ws, exe, dev.stream))
            # 先下发消费者，再由 host 延迟发布 flag。只改 flag，token/ids 已提前写好，
            # 对应生产者“先写完整数据、最后发布 ready”的顺序。
            if publish is not None:
                time.sleep(0.2)
                status = C.c_int()
                check(dev.query(dev.stream, C.byref(status)))
                # ACL_STREAM_STATUS_NOT_READY=1：确保无数据时没有提前返回空结果。
                assert status.value == 1, f'{name}: returned before a ready flag was published'
                for a, m in publish:
                    flag = np.array([1], np.int32)
                    offset = (a * M + m) * desc.shape[-1] * 4
                    dev.upload(P(ti.value + offset), flag)
                    before[a, m, 0] = 1
            check(dev.sync(dev.stream, 10000))
            result = [dev.read(p, a) for p, a in zip(ptrs, outs)]
            rows = []
            for a in selected:
                for t, e in enumerate(before[a, mb, 2:]):
                    if e < 1000000:
                        rows.append((int(e), a, t // K, t % K))
            rows.sort(key=lambda r: r[0])
            assert result[7][0] == len(rows), (name, result[7], len(rows))
            for i, (e, a, b, k) in enumerate(rows):
                np.testing.assert_array_equal(result[0][i], tokens[a, mb, b, k])
                assert tuple((int(result[j][i]) for j in [2, 3, 4, 5])) == (a, mb, b, k)
            groups = []
            for e, _, _, _ in rows:
                if not groups or groups[-1][0] != e:
                    groups.append([e, 0])
                groups[-1][1] += 1
            if groups:
                np.testing.assert_array_equal(result[1][:len(groups)], groups)
            else:
                np.testing.assert_array_equal(result[1][0], [0, 0])
            # 比较整个描述符表，而非仅检查输出：未选中 session、其它 micro batch 和
            # 所有 layer_id 必须保留；选中行只能清 flag 并把 ids 置为无效值。
            expected = before.copy()
            for a in selected:
                expected[a, mb, 0] = 0
                expected[a, mb, 2:] = np.iinfo(np.int32).max
            desc = dev.read(ti, desc)
            np.testing.assert_array_equal(desc, expected)
            ctx = dev.read(sc, ctx)
            assert struct.unpack_from('<Q', ctx, 416)[0] == cursor
            np.testing.assert_array_equal(dev.read(td, tokens), tokens)
            reports.append({'case': name, 'status': 'PASS', 'actual_token_num': len(rows)})
        # 以下调用共用同一份设备上下文，串联检查游标推进、回绕以及槽位再次使用。
        invoke('empty_wait_then_one_ready', [3], 2, 0, publish=[(3, 2)])
        desc[1, 1, 0] = 1
        dev.upload(ti, desc)
        invoke('skip_empty_mb_and_preserve_others', [1], 1, 2)
        desc[0, 0, 0] = 1
        desc[2, 0, 0] = 1
        dev.upload(ti, desc)
        invoke('cursor_wrap_and_partial_ready', [0, 2], 0, 1)
        desc[3, 2, 0] = 1
        desc[3, 2, 2:] = ids[3, 2]
        dev.upload(ti, desc)
        invoke('reuse_consumed_slot', [3], 2, 0)
        desc[0, 0, 0] = 1
        desc[0, 0, 2:] = 1000000
        dev.upload(ti, desc)
        invoke('ready_with_no_valid_tokens', [0], 0, 1)
        desc[:, 1, 2:] = ids[:, 1]
        desc[0, 1, 0] = 1
        dev.upload(ti, desc)
        # 对照组：已有一个 session ready 时仍应等待其它三个，防止默认同步语义退化。
        invoke('sync_waits_for_all', list(range(A)), 1, 2, sync_flag=0, publish=[(1, 1), (2, 1), (3, 1)])
        check(dev.destroy_array(attr))
        print(json.dumps(reports, indent=2))
    finally:
        dev.close()
if __name__ == '__main__':
    main()
