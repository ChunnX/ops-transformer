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

"""Golden and customize_inputs for aclnnFfnWorkerBatching.

This module combines:
  - inputs: real-device-address scheduleContext (HBM buffers + test data)
  - golden: CPU golden reference using cached test data
  - compare: custom comparison (kernel writes only actual_token_num elements)

Auxiliary buffers (token_data / token_info / flat ids) live in dedicated
aclrtMalloc allocations and the context carries absolute device addresses,
matching the production double-dereference path (no TEST_MAGIC relative
offsets). RECV consumes descriptors on device, so its TTK execution count is
pinned to 1 per case (TTK's default 3 would hang on the second run).
"""

import atexit
import ctypes
import json
import os
import struct
import numpy as np
import torch

__spec__ = {
    "aclnnFfnWorkerBatching": "AclnnFfnWorkerBatchingLegacyTestSpec",
    "aclnnFfnWorkerBatchingV2": "AclnnFfnWorkerBatchingTestSpec",
    "ffn_worker_batching": "FfnWorkerBatchingKernelTestSpec",
    "torch.ops.cann_ops_transformer.ffn_worker_batching": "TorchFfnWorkerBatchingTestSpec",
}

# ==================== Shared cache ====================

_DATA_CACHE = {}
_HBM_ALLOCS = []
_HBM_VIEWS = {}  # actual suballocation address -> payload size, for replay validation
_ACL_LIB = None
_MASK_VALUE = 1000000
# ScheduleContext ABI v1, defined by common/include/op_kernel/attention_ffn_schedule.h.
_SCHEDULE_CONTEXT_BYTES = 1024
_CONTEXT_ARCHIVE_VERSION = 1
_TOKEN_STRIDE_ALIGNMENT = 512
_HBM_ALLOCATION_ALIGNMENT = (
    512  # Allocation alignment and row stride are separate contracts.
)
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2
_NUM_OUTPUTS = 8
_OUT_Y = 0
_OUT_GROUP_LIST = 1
_OUT_SESSION_IDS = 2
_OUT_MICRO_BATCH_IDS = 3
_OUT_TOKEN_IDS = 4
_OUT_EXPERT_OFFSETS = 5
_OUT_SCALE = 6
_OUT_TOKEN_COUNT = 7
_TOKEN_PREFIX_OUTPUTS = (
    _OUT_Y,
    _OUT_SESSION_IDS,
    _OUT_MICRO_BATCH_IDS,
    _OUT_TOKEN_IDS,
    _OUT_EXPERT_OFFSETS,
    _OUT_SCALE,
)

# ---- ScheduleContext byte offsets ----
_OFF_SESSION_NUM = 0
_OFF_MICRO_BATCH_NUM = 4
_OFF_MICRO_BATCH_SIZE = 8
_OFF_SELECTED_EXPERT_NUM = 12
_OFF_EXPERT_NUM = 16
_OFF_ATTN_TO_FFN_TOKEN_SIZE = 20
_OFF_FFN_TO_ATTN_TOKEN_SIZE = 24
_OFF_SCHEDULE_MODE = 28
_OFF_RUN_FLAG = 128
_OFF_FFN_TOKEN_INFO_BUF = 384
_OFF_FFN_TOKEN_INFO_BUF_SIZE = 392
_OFF_FFN_TOKEN_DATA_BUF = 400
_OFF_FFN_TOKEN_DATA_BUF_SIZE = 408
_OFF_POLLING_INDEX = 416
_OFF_SESSION_IDS_BUF = 528
_OFF_SESSION_IDS_BUF_SIZE = 536
_OFF_MICRO_BATCH_IDS_BUF = 544
_OFF_MICRO_BATCH_IDS_BUF_SIZE = 552
_OFF_EXPERT_IDS_BUF = 560
_OFF_EXPERT_IDS_BUF_SIZE = 568
_OFF_OUT_NUM = 576

# ==================== ACL / HBM helpers ====================


def _load_acl():
    global _ACL_LIB
    if _ACL_LIB is not None:
        return _ACL_LIB
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtMalloc.restype = ctypes.c_int32
    lib.aclrtMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_int32,
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int32
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int32,
    ]
    lib.aclrtFree.restype = ctypes.c_int32
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    lib.aclrtSetDevice.restype = ctypes.c_int32
    lib.aclrtSetDevice.argtypes = [ctypes.c_int32]
    lib.aclrtGetDevice.restype = ctypes.c_int32
    lib.aclrtGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    lib.aclrtSynchronizeDevice.restype = ctypes.c_int32
    lib.aclrtSynchronizeDevice.argtypes = []
    lib.aclrtMemFlush.restype = ctypes.c_int32
    lib.aclrtMemFlush.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.aclrtMemInvalidate.restype = ctypes.c_int32
    lib.aclrtMemInvalidate.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _ACL_LIB = lib
    return lib


_HBM_BASE = None
_HBM_OFFSET = 0
_HBM_CAP = 0
_HBM_BLOCK_SIZE = 4 * 1024 * 1024


def _ensure_device(lib):
    # 沿用进程内已设置的当前设备（TTK/torch_npu 可能已初始化），否则落到 0 卡。
    dev = ctypes.c_int32(-1)
    if lib.aclrtGetDevice(ctypes.byref(dev)) == 0 and dev.value >= 0:
        return
    lib.aclrtSetDevice(0)


def _hbm_alloc(size):
    # context 中保存的是可供 kernel 二次解引用的设备绝对地址，不是 host 数组地址。
    # 大块分配后按 512B 切分，既满足生产者的缓冲起点约束，又减少小对象 aclrtMalloc。
    # _HBM_VIEWS 记录每个子分配的逻辑长度，而非整块容量；归档校验必须匹配这段
    # payload 的大小，不能因为某地址落在已分配的大块内就允许读取任意长度。
    # 分配保留到进程退出，避免 Python 临时数组析构后 context 中的设备地址失效。
    global _HBM_BASE, _HBM_OFFSET, _HBM_CAP
    lib = _load_acl()
    _ensure_device(lib)
    aligned_size = (size + (_HBM_ALLOCATION_ALIGNMENT - 1)) & ~(
        _HBM_ALLOCATION_ALIGNMENT - 1
    )
    if _HBM_BASE is None or _HBM_OFFSET + aligned_size > _HBM_CAP:
        # 超过默认块大小的缓冲（如 a4/h8192 的 token_data）整块独立分配
        cap = max(_HBM_BLOCK_SIZE, aligned_size)
        ptr = ctypes.c_void_p(0)
        ret = lib.aclrtMalloc(ctypes.byref(ptr), cap, 0)
        if ret != 0 or ptr.value is None:
            raise RuntimeError(f"aclrtMalloc failed: ret={ret}, size={cap}")
        _HBM_BASE = ptr.value
        _HBM_OFFSET = 0
        _HBM_CAP = cap
        _HBM_ALLOCS.append(_HBM_BASE)
    addr = _HBM_BASE + _HBM_OFFSET
    _HBM_OFFSET += aligned_size
    _HBM_VIEWS[addr] = size
    return addr


def _hbm_copy(host_data, hbm_ptr):
    lib = _load_acl()
    if isinstance(host_data, np.ndarray):
        buf = host_data.tobytes()
    elif isinstance(host_data, (bytes, bytearray)):
        buf = bytes(host_data)
    else:
        buf = host_data
    ret = lib.aclrtMemcpy(
        hbm_ptr, len(buf), ctypes.c_char_p(buf), len(buf), _ACL_MEMCPY_HOST_TO_DEVICE
    )
    if ret != 0:
        raise RuntimeError(f"aclrtMemcpy H2D failed: ret={ret}")
    lib.aclrtMemInvalidate(ctypes.c_void_p(hbm_ptr), len(buf))


@atexit.register
def _cleanup_hbm():
    global _HBM_BASE, _HBM_OFFSET, _HBM_CAP
    lib = _ACL_LIB
    if lib is None:
        return
    for ptr in _HBM_ALLOCS:
        try:
            lib.aclrtFree(ctypes.c_void_p(ptr))
        except Exception:
            pass
    _HBM_ALLOCS.clear()
    _HBM_VIEWS.clear()
    _HBM_BASE = None
    _HBM_OFFSET = 0
    _HBM_CAP = 0


def _context_buffer_offsets(need_schedule):
    return (
        (_OFF_FFN_TOKEN_DATA_BUF, _OFF_FFN_TOKEN_INFO_BUF)
        if need_schedule
        else (
            _OFF_FFN_TOKEN_DATA_BUF,
            _OFF_SESSION_IDS_BUF,
            _OFF_MICRO_BATCH_IDS_BUF,
            _OFF_EXPERT_IDS_BUF,
        )
    )


def validate_live_context(header, need_schedule=0):
    """Reject a plain dump from another process before dereferencing its pointers."""
    # 1024B 头本身不是完整输入：token/描述符/路由数组各自位于独立 HBM。
    # 当前进程登记表只用于确认测试夹具拥有这批缓冲，不能验证全部生产者协议，
    # 例如 flag 发布顺序、实际 A/BS/K 与静态容量的关系仍不由这里检查。
    header = np.asarray(header, dtype=np.int8).reshape(-1)
    if header.size < _SCHEDULE_CONTEXT_BYTES:
        raise ValueError("schedule_context needs at least 1024 bytes")
    for offset in _context_buffer_offsets(need_schedule):
        pointer, size = struct.unpack_from("<QQ", header, offset)
        if not size or _HBM_VIEWS.get(pointer) != size:
            raise ValueError(
                "raw context replay has no live secondary buffers; "
                "save with FFN_WB_SAVE_ARCHIVE and restore with FFN_WB_REPLAY_ARCHIVE"
            )


def save_context_archive(path, header, attributes):
    """Save payloads as well as metadata; archived addresses are deliberately zero.

    Only buffers owned by this TestSpec can be saved. This prevents accidentally
    reading stale addresses from an ordinary input dump. Synchronize before D2H
    so the archive represents completed input preparation, not queued writes.
    """
    # 必须复制头再将地址清零：不能修改仍将交给本次 kernel 的 live context。
    # 属性也随包保存，因为相同字节在 FP4/FP8、NORM/RECV 下具有不同解释。
    # 同步只保证已提交任务完成；归档不会自动恢复 RECV 消费前的 flag/IDs，
    # 因此要重放初始输入，调用方必须在消费前保存，而不能归档已被清理的描述符。
    header = (
        np.asarray(header, dtype=np.int8).reshape(-1)[:_SCHEDULE_CONTEXT_BYTES].copy()
    )
    mode = attributes["need_schedule"]
    validate_live_context(header, mode)
    lib = _load_acl()
    ret = lib.aclrtSynchronizeDevice()
    if ret:
        raise RuntimeError(f"archive synchronize failed: {ret}")
    records = {
        "version": np.array([_CONTEXT_ARCHIVE_VERSION], dtype=np.int64),
        "attributes": np.frombuffer(
            json.dumps(attributes, sort_keys=True).encode(), dtype=np.uint8
        ),
    }
    for offset in _context_buffer_offsets(mode):
        pointer, size = struct.unpack_from("<QQ", header, offset)
        payload = np.empty(size, dtype=np.uint8)
        ret = lib.aclrtMemcpy(
            payload.ctypes.data, size, pointer, size, _ACL_MEMCPY_DEVICE_TO_HOST
        )
        if ret:
            raise RuntimeError(f"archive D2H failed: {ret}")
        records[f"buffer_{offset}"] = payload
        struct.pack_into("<Q", header, offset, 0)
    records["header"] = header
    with open(path, "wb") as stream:
        np.savez(stream, **records)


def restore_context_archive(path, expected_attributes=None):
    """Allocate fresh HBM and relocate the header; never reuse saved addresses."""
    # 文件只提供字节与元数据，旧进程虚拟地址不可继承。先检查所有 payload，
    # 再逐段分配/复制并重定位头部指针，使新进程的 kernel 访问新设备内存。
    # allow_pickle=False 使格式限于数组；这里验证格式不等于验证调度并发行为。
    with np.load(path, allow_pickle=False) as archive:
        if archive["version"].tolist() != [_CONTEXT_ARCHIVE_VERSION]:
            raise ValueError("unsupported FFN context archive version")
        attributes = json.loads(archive["attributes"].tobytes())
        if expected_attributes is not None and attributes != expected_attributes:
            raise ValueError("FFN context archive attributes mismatch")
        header = archive["header"].copy()
        if header.dtype != np.int8 or header.shape != (_SCHEDULE_CONTEXT_BYTES,):
            raise ValueError("invalid FFN context archive header")
        # Validate every payload before starting any allocation/copy.
        offsets = _context_buffer_offsets(attributes["need_schedule"])
        for offset in offsets:
            pointer, size = struct.unpack_from("<QQ", header, offset)
            payload = archive[f"buffer_{offset}"]
            if (
                pointer != 0
                or size <= 0
                or payload.dtype != np.uint8
                or payload.shape != (size,)
            ):
                raise ValueError("invalid FFN context archive buffer")
        for offset in offsets:
            payload = archive[f"buffer_{offset}"]
            pointer = _hbm_alloc(payload.size)
            _hbm_copy(payload, pointer)
            struct.pack_into("<Q", header, offset, pointer)
    return header, attributes


_ORIG_RUN_TIME = None


def _pin_run_time(need_schedule):
    """RECV 在设备侧消费描述符（清 flag / ids 置 sentinel），真实地址缓冲不会随
    TTK 的重复执行刷新：第二次执行面对空表会永久等待。TTK 默认 run_time=3，
    这里把 RECV 用例钉到单次执行，NORM 恢复原值。依赖 TTK 逐用例
    “输入生成先于执行器读取 run_time”的时序（aclnn / kernel 两模式均如此）。"""
    global _ORIG_RUN_TIME
    try:
        from ttk.utilities.container_utils import get_global_storage

        storage = get_global_storage()
        if _ORIG_RUN_TIME is None:
            _ORIG_RUN_TIME = storage.run_time
        storage.run_time = 1 if need_schedule == 1 else _ORIG_RUN_TIME
    except Exception:
        pass


# ==================== Data generation ====================


def _seed_from_name(name):
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return h


def _pack_ctx(ctx_bytes, offset, fmt, value):
    struct.pack_into(fmt, ctx_bytes, offset, value)


def _generate_test_data(
    testcase_name, A, M, BS, K, H, expertNum, tokenDtype, needSchedule
):
    rng = np.random.RandomState(_seed_from_name(testcase_name))
    # 当前夹具使用满容量 NORM：out_num=A，并在下方使用顺序 session、microbatch=0。
    # 因而本生成器尚不覆盖 out_num=0/<A、session 重排或 NORM 非零 microbatch；
    # 不能把这些静态用例通过当作对应间接寻址/容量边界已经得到验证。
    out_num = A
    has_mask = "masked" in testcase_name
    single_expert = "single_expert" in testcase_name

    if needSchedule == 0:
        expert_ids = rng.randint(0, expertNum, size=(out_num, BS, K)).astype(np.int32)
    else:
        expert_ids = rng.randint(0, expertNum, size=(A, M, BS, K)).astype(np.int32)

    if "all_masked" in testcase_name:
        expert_ids[...] = _MASK_VALUE
    elif single_expert:
        expert_ids[...] = 0
    elif has_mask:
        mask = rng.random(expert_ids.shape) < 0.15
        expert_ids[mask] = _MASK_VALUE

    if tokenDtype == 0:
        token_data = (rng.randn(A, M, BS, K, H) * 0.1).astype(np.float16)
        token_scales = None
    elif tokenDtype == 1:
        token_data = (rng.randn(A, M, BS, K, H) * 0.1).astype(np.float32)
        token_scales = None
    elif tokenDtype >= 3:
        from ttk.utilities.dtypes import (
            numpy_float8_e5m2,
            numpy_float8_e4m3fn,
            numpy_float4_e2m1,
            numpy_float8_e8m0,
        )

        dtype = {3: numpy_float8_e5m2, 4: numpy_float8_e4m3fn, 5: numpy_float4_e2m1}[
            tokenDtype
        ]()
        # 原始位模式包含正负零及 FP8 非有限值，搬运不得改变位模式。
        token_data = (
            rng.randint(0, 16 if tokenDtype == 5 else 256, size=(A, M, BS, K, H))
            .astype(np.uint8)
            .view(dtype)
        )
        token_scales = (
            rng.randint(0, 256, size=(A, M, BS, K, (H + 31) // 32))
            .astype(np.uint8)
            .view(numpy_float8_e8m0())
        )
    else:
        token_data = rng.randint(-128, 127, size=(A, M, BS, K, H)).astype(np.int8)
        token_scales = (rng.rand(A, M, BS, K) * 2 + 0.1).astype(np.float32)

    session_ids = np.arange(A, dtype=np.int32)
    micro_batch_ids = np.zeros(A, dtype=np.int32)

    return {
        "A": A,
        "M": M,
        "BS": BS,
        "K": K,
        "H": H,
        "expertNum": expertNum,
        "tokenDtype": tokenDtype,
        "needSchedule": needSchedule,
        "expert_ids": expert_ids,
        "token_data": token_data,
        "token_scales": token_scales,
        "session_ids_buf": session_ids,
        "micro_batch_ids_buf": micro_batch_ids,
        "out_num": out_num,
        "cur_micro_batch_id": 0,
    }


def _build_token_data_bytes(token_data, token_scales, tokenDtype, A, M, BS, K, H):
    if tokenDtype >= 3:
        from ttk.utilities.dtypes import pack_4bits

        raw_tokens = (
            pack_4bits(token_data).reshape(A, M, BS, K, H // 2)
            if tokenDtype == 5
            else token_data.view(np.uint8)
        )
        return np.concatenate((raw_tokens, token_scales.view(np.uint8)), axis=-1)
    if tokenDtype == 1:
        t = torch.from_numpy(np.ascontiguousarray(token_data, dtype=np.float32))
        t_bf16 = t.to(torch.bfloat16)
        return t_bf16.contiguous().view(torch.uint8).numpy()
    if tokenDtype != 2:
        return np.ascontiguousarray(token_data)
    per_token = H + 4
    buf = np.zeros((A, M, BS, K, per_token), dtype=np.uint8)
    buf[..., :H] = token_data.view(np.uint8)
    scale_bytes = (
        np.ascontiguousarray(token_scales, dtype=np.float32)
        .view(np.uint8)
        .reshape(A, M, BS, K, 4)
    )
    buf[..., H : H + 4] = scale_bytes
    return buf


def _build_token_info_buf(expert_ids, A, M, BS, K):
    F = 2 + BS * K
    token_info = np.zeros((A, M, F), dtype=np.int32)
    for a in range(A):
        for m in range(M):
            token_info[a, m, 0] = 1
            token_info[a, m, 1] = 0
            token_info[a, m, 2 : 2 + BS * K] = expert_ids[a, m].flatten()
    return token_info


# ==================== Golden helpers ====================

EXPERT_MASK_VALUE = 1000000
# ScheduleContext ABI v1, defined by common/include/op_kernel/attention_ffn_schedule.h.
_SCHEDULE_CONTEXT_BYTES = 1024
_CONTEXT_ARCHIVE_VERSION = 1
_TOKEN_STRIDE_ALIGNMENT = 512
_HBM_ALLOCATION_ALIGNMENT = (
    512  # Allocation alignment and row stride are separate contracts.
)
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2
_NUM_OUTPUTS = 8
_OUT_Y = 0
_OUT_GROUP_LIST = 1
_OUT_SESSION_IDS = 2
_OUT_MICRO_BATCH_IDS = 3
_OUT_TOKEN_IDS = 4
_OUT_EXPERT_OFFSETS = 5
_OUT_SCALE = 6
_OUT_TOKEN_COUNT = 7
_TOKEN_PREFIX_OUTPUTS = (
    _OUT_Y,
    _OUT_SESSION_IDS,
    _OUT_MICRO_BATCH_IDS,
    _OUT_TOKEN_IDS,
    _OUT_EXPERT_OFFSETS,
    _OUT_SCALE,
)


def _get_cached_data(testcase_name, **gen_kwargs):
    if testcase_name not in _DATA_CACHE:
        if gen_kwargs:
            try:
                data = _generate_test_data(
                    testcase_name,
                    gen_kwargs.get("A", 1),
                    gen_kwargs.get("M", 1),
                    gen_kwargs.get("BS", 1),
                    gen_kwargs.get("K", 1),
                    gen_kwargs.get("H", 1),
                    gen_kwargs.get("expertNum", 1),
                    gen_kwargs.get("tokenDtype", 0),
                    gen_kwargs.get("needSchedule", 0),
                )
                data["HS"] = gen_kwargs.get("HS", 1)
                _DATA_CACHE[testcase_name] = data
            except Exception:
                pass
    if testcase_name not in _DATA_CACHE:
        raise KeyError(
            f"No cached data for testcase '{testcase_name}'. "
            f"Available: {list(_DATA_CACHE.keys())}"
        )
    return _DATA_CACHE[testcase_name]


def _sort_expert_ids(expert_ids_flat):
    # 稳定排序保留同一专家内的原始 token 顺序。参考实现直接用整数排序，
    # 不复制设备 proposal 的浮点 key/uint32 index 表示，从而可检查设备排序的
    # 键与原始位置是否被混淆。当前普通路径输入由生成器保证非负；负 ID 的
    # 过滤由异步层编码参考另行处理，此函数本身只实现上界 mask 判据。
    sort_key = np.where(
        expert_ids_flat < EXPERT_MASK_VALUE, expert_ids_flat, np.iinfo(np.int32).max
    ).astype(np.int64)
    sorted_order = np.argsort(sort_key, kind="stable")
    valid_count = int(np.sum(expert_ids_flat < EXPERT_MASK_VALUE))
    sorted_valid_ids = expert_ids_flat[sorted_order[:valid_count]]
    return sorted_order, valid_count, sorted_valid_ids


def _generate_group_list(sorted_expert_ids, valid_count, expert_num):
    group_list = np.zeros((expert_num, 2), dtype=np.int64)
    if valid_count == 0:
        return group_list
    offset = 0
    cur_expert = int(sorted_expert_ids[0])
    token_count = 0
    for i in range(valid_count):
        eid = int(sorted_expert_ids[i])
        if eid != cur_expert:
            if offset < expert_num:
                group_list[offset, 0] = cur_expert
                group_list[offset, 1] = token_count
                offset += 1
            cur_expert = eid
            token_count = 1
        else:
            token_count += 1
    if offset < expert_num:
        group_list[offset, 0] = cur_expert
        group_list[offset, 1] = token_count
        offset += 1
    if offset < expert_num:
        group_list[offset, 0] = 0
        group_list[offset, 1] = 0
    return group_list


def _to_torch(arr, dtype):
    if "float8" in arr.dtype.name or "float4" in arr.dtype.name:
        from ttk.utilities.dtypes import numpy_to_torch_tensor

        tensor = numpy_to_torch_tensor(np.ascontiguousarray(arr))
        return (
            tensor.reshape(*arr.shape[:-1], arr.shape[-1] // 2)
            if "float4" in arr.dtype.name
            else tensor
        )
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if dtype is not None:
        t = t.to(dtype)
    return t


# ==================== TestSpec class ====================


class AclnnFfnWorkerBatchingTestSpec:
    @staticmethod
    def customize_inputs(
        scheduleContext,
        expertNum,
        maxOutShape,
        tokenDtype,
        needSchedule,
        layerNum,
        syncFlag: bool,
        y,
        groupList,
        sessionIds,
        microBatchIds,
        tokenIds,
        expertOffsets,
        dynamicScale,
        actualTokenNum,
        *args,
        **kwargs,
    ):
        testcase_name = kwargs.get("testcase_name", "default")
        _pin_run_time(needSchedule)
        A = int(maxOutShape[0])
        BS = int(maxOutShape[1])
        K = int(maxOutShape[2])
        H = int(maxOutShape[3])
        M = 3 if "sync_skip_mb" in testcase_name else 1

        data = _generate_test_data(
            testcase_name, A, M, BS, K, H, expertNum, tokenDtype, needSchedule
        )

        token_data = data["token_data"]
        token_scales = data["token_scales"]
        expert_ids = data["expert_ids"]
        session_ids = data["session_ids_buf"]
        micro_batch_ids = data["micro_batch_ids_buf"]
        out_num = data["out_num"]

        if tokenDtype in (0, 1):
            dtype_size = 2
        else:
            dtype_size = 1
        HS = H * dtype_size if tokenDtype != 2 else H + 4
        if tokenDtype >= 3:
            HS = (H // 2 if tokenDtype == 5 else H) + (H + 31) // 32

        td_arr = _build_token_data_bytes(
            token_data, token_scales, tokenDtype, A, M, BS, K, H
        )
        # Physical token stride follows the producer ABI. Keep scale immediately
        # after the payload and zero only the remaining 512-byte stride padding.
        payload = np.ascontiguousarray(td_arr).view(np.uint8).reshape(A, M, BS, K, HS)
        HS = (
            (HS + (_TOKEN_STRIDE_ALIGNMENT - 1))
            // _TOKEN_STRIDE_ALIGNMENT
            * _TOKEN_STRIDE_ALIGNMENT
        )
        padded = np.zeros((A, M, BS, K, HS), dtype=np.uint8)
        padded[..., : payload.shape[-1]] = payload
        td_bytes = padded.tobytes()

        # 真实地址协议：辅助数据放独立 HBM 缓冲，context 仅保留 1024B 头部并写入
        # 绝对设备地址（生产二级指针寻址路径）；不写 TEST_MAGIC，偏移 640 保持 0。
        td_ptr = _hbm_alloc(len(td_bytes))
        _hbm_copy(td_bytes, td_ptr)

        if needSchedule == 0:
            ei_bytes = np.ascontiguousarray(expert_ids).tobytes()
            si_bytes = np.ascontiguousarray(session_ids).tobytes()
            mi_bytes = np.ascontiguousarray(micro_batch_ids).tobytes()
            ei_ptr = _hbm_alloc(len(ei_bytes))
            _hbm_copy(ei_bytes, ei_ptr)
            si_ptr = _hbm_alloc(len(si_bytes))
            _hbm_copy(si_bytes, si_ptr)
            mi_ptr = _hbm_alloc(len(mi_bytes))
            _hbm_copy(mi_bytes, mi_ptr)
        else:
            token_info = _build_token_info_buf(expert_ids, A, M, BS, K)
            layer_ids = np.zeros((A, M), dtype=np.int32)
            if syncFlag and "async_layers" in testcase_name:
                per_layer = expertNum // layerNum
                valid = (expert_ids >= 0) & (expert_ids < _MASK_VALUE)
                expert_ids[valid] %= per_layer
                token_info[:, :, 2:] = expert_ids.reshape(A, M, BS * K)
                layer_ids[:] = (np.arange(A, dtype=np.int32)[::-1] % layerNum)[:, None]
                token_info[:, :, 1] = layer_ids
            data["layer_ids"] = layer_ids
            # 独立构造“已发布描述符”的集合，不把未就绪行的 expert_ids 预先 mask。
            # 这样错误读取未就绪行的 kernel 会产生额外 token，能够被 golden 检出。
            ready = np.ones((A, M), dtype=bool)
            if syncFlag and "sync_" in testcase_name:
                if "sync_partial" in testcase_name:
                    ready[1::2, :] = False
                elif "sync_last" in testcase_name or "sync_skip_mb" in testcase_name:
                    ready[:] = False
                    ready[-1, -1] = True
                elif "sync_masked" in testcase_name:
                    ready[1:, :] = False
                    token_info[0, :, 2:] = _MASK_VALUE
                    expert_ids[0, :, :, :] = _MASK_VALUE
            token_info[:, :, 0] = ready.astype(np.int32)
            data["ready"] = ready
            # 静态 ST 的入口 cursor 为 0，至少预置一个 ready；期望选中首个非空 micro batch。
            # 完全无 ready 的等待/延迟发布不能用静态 CSV 表达，需要独立状态测试。
            # 当前仓库未随此夹具提供该状态测试源码，不能据此宣称已覆盖并发发布。
            # RECV 单次执行（run_time 已钉到 1），描述符消费后不复位。
            data["cur_micro_batch_id"] = next(m for m in range(M) if ready[:, m].any())
            ti_bytes = token_info.tobytes()
            ti_ptr = _hbm_alloc(len(ti_bytes))
            _hbm_copy(ti_bytes, ti_ptr)

        total_size = 1024
        ctx = bytearray(total_size)

        _pack_ctx(ctx, _OFF_SESSION_NUM, "<I", A)
        _pack_ctx(ctx, _OFF_MICRO_BATCH_NUM, "<I", M)
        _pack_ctx(ctx, _OFF_MICRO_BATCH_SIZE, "<I", BS)
        _pack_ctx(ctx, _OFF_SELECTED_EXPERT_NUM, "<I", K)
        _pack_ctx(ctx, _OFF_EXPERT_NUM, "<I", expertNum)
        _pack_ctx(ctx, _OFF_ATTN_TO_FFN_TOKEN_SIZE, "<I", HS)
        _pack_ctx(ctx, _OFF_FFN_TO_ATTN_TOKEN_SIZE, "<I", HS)
        _pack_ctx(ctx, _OFF_SCHEDULE_MODE, "<i", 0)
        _pack_ctx(ctx, _OFF_RUN_FLAG, "<i", 1)
        _pack_ctx(ctx, _OFF_FFN_TOKEN_DATA_BUF, "<Q", td_ptr)
        _pack_ctx(ctx, _OFF_FFN_TOKEN_DATA_BUF_SIZE, "<Q", len(td_bytes))

        if needSchedule == 0:
            _pack_ctx(ctx, _OFF_SESSION_IDS_BUF, "<Q", si_ptr)
            _pack_ctx(ctx, _OFF_SESSION_IDS_BUF_SIZE, "<Q", len(si_bytes))
            _pack_ctx(ctx, _OFF_MICRO_BATCH_IDS_BUF, "<Q", mi_ptr)
            _pack_ctx(ctx, _OFF_MICRO_BATCH_IDS_BUF_SIZE, "<Q", len(mi_bytes))
            _pack_ctx(ctx, _OFF_EXPERT_IDS_BUF, "<Q", ei_ptr)
            _pack_ctx(ctx, _OFF_EXPERT_IDS_BUF_SIZE, "<Q", len(ei_bytes))
            _pack_ctx(ctx, _OFF_OUT_NUM, "<I", out_num)
        else:
            _pack_ctx(ctx, _OFF_FFN_TOKEN_INFO_BUF, "<Q", ti_ptr)
            _pack_ctx(ctx, _OFF_FFN_TOKEN_INFO_BUF_SIZE, "<Q", len(ti_bytes))
            _pack_ctx(ctx, _OFF_POLLING_INDEX, "<Q", 0)

        ctx_arr = np.array(ctx, dtype=np.int8)
        if isinstance(scheduleContext, np.ndarray):
            if scheduleContext.nbytes < total_size:
                raise ValueError(
                    f"scheduleContext requires {total_size} bytes, got {scheduleContext.nbytes}"
                )
            scheduleContext.reshape(-1)[:total_size] = ctx_arr
        else:
            cur_storage = scheduleContext.untyped_storage()
            if cur_storage.nbytes() < total_size:
                new_storage = torch.UntypedStorage(total_size)
                scheduleContext.set_(
                    new_storage,
                    scheduleContext.storage_offset(),
                    scheduleContext.shape,
                    scheduleContext.stride(),
                )
            scheduleContext.untyped_storage()[:total_size].copy_(
                torch.from_numpy(ctx_arr).untyped_storage()
            )

        data["HS"] = HS
        _DATA_CACHE[testcase_name] = data

    @staticmethod
    def golden(
        scheduleContext,
        expertNum,
        maxOutShape,
        tokenDtype,
        needSchedule,
        layerNum,
        syncFlag: bool,
        y,
        groupList,
        sessionIds,
        microBatchIds,
        tokenIds,
        expertOffsets,
        dynamicScale,
        actualTokenNum,
        *args,
        **kwargs,
    ):
        testcase_name = kwargs.get("testcase_name", "default")
        A = int(maxOutShape[0])
        BS = int(maxOutShape[1])
        K = int(maxOutShape[2])
        H = int(maxOutShape[3])
        M = 1
        if tokenDtype in (0, 1):
            HS = H * 2
        elif tokenDtype >= 3:
            HS = (H // 2 if tokenDtype == 5 else H) + (H + 31) // 32
        else:
            HS = H + 4
        HS = (
            (HS + (_TOKEN_STRIDE_ALIGNMENT - 1))
            // _TOKEN_STRIDE_ALIGNMENT
            * _TOKEN_STRIDE_ALIGNMENT
        )
        data = _get_cached_data(
            testcase_name,
            A=A,
            M=M,
            BS=BS,
            K=K,
            H=H,
            expertNum=expertNum,
            tokenDtype=tokenDtype,
            needSchedule=needSchedule,
            HS=HS,
        )

        A = data["A"]
        M = data["M"]
        BS = data["BS"]
        K = data["K"]
        H = data["H"]
        token_data = data["token_data"]
        token_scales = data.get("token_scales")
        session_ids_buf = data["session_ids_buf"]
        micro_batch_ids_buf = data["micro_batch_ids_buf"]
        expert_ids = data["expert_ids"]

        Y = A * BS * K

        if tokenDtype == 0:
            out_torch_dtype = torch.float16
            out_np_dtype = np.float16
        elif tokenDtype == 1:
            out_torch_dtype = torch.bfloat16
            out_np_dtype = np.float32
        elif tokenDtype >= 3:
            out_torch_dtype = None
            out_np_dtype = token_data.dtype
        else:
            out_torch_dtype = torch.int8
            out_np_dtype = np.int8

        if needSchedule == 1:
            cur_mb = data.get("cur_micro_batch_id", 0)
            # 只在 CPU 参考副本上屏蔽未选中 session；设备输入仍保留它们的合法 ids。
            ids = expert_ids[:, cur_mb].copy()
            if syncFlag:
                per_layer = expertNum // layerNum
                layer_ids = data.get("layer_ids", np.zeros((A, M), np.int32))[:, cur_mb]
                valid = (
                    (ids >= 0)
                    & (ids < per_layer)
                    & (layer_ids[:, None, None] >= 0)
                    & (layer_ids[:, None, None] < layerNum)
                )
                ids = np.where(
                    valid,
                    ids.astype(np.int64) + layer_ids[:, None, None] * per_layer,
                    _MASK_VALUE,
                ).astype(np.int32)
                ids[~data.get("ready", np.ones((A, M), dtype=bool))[:, cur_mb]] = (
                    _MASK_VALUE
                )
            expert_ids_flat = ids.reshape(-1).astype(np.int32)
        else:
            expert_ids_flat = expert_ids.reshape(-1).astype(np.int32)
        # CPU 参考不插入 RECV 的 32B 行 padding：只枚举业务 token，再以 BS*K
        # 解码原位置。设备实现必须自行排除 padding，并用其物理行跨度解码，
        # 两种路径最终的 session/token/expert-offset 应一致。
        sorted_order, valid_count, sorted_valid_ids = _sort_expert_ids(expert_ids_flat)

        y_out = np.zeros((Y, H), dtype=out_np_dtype)
        session_ids_out = np.zeros(Y, dtype=np.int32)
        micro_batch_ids_out = np.zeros(Y, dtype=np.int32)
        token_ids_out = np.zeros(Y, dtype=np.int32)
        expert_offsets_out = np.zeros(Y, dtype=np.int32)
        dynamic_scale_out = (
            np.zeros((Y, (H + 31) // 32), dtype=token_scales.dtype)
            if tokenDtype >= 3
            else np.ones(Y, dtype=np.float32)
        )

        bsk = BS * K
        cur_mb = data.get("cur_micro_batch_id", 0)

        for i in range(valid_count):
            gidx = int(sorted_order[i])
            a_idx = gidx // bsk
            rem = gidx % bsk
            bs_idx = rem // K
            k_idx = rem % K

            if needSchedule == 0:
                s_idx = int(session_ids_buf[a_idx])
                mb_idx = int(micro_batch_ids_buf[a_idx])
            else:
                s_idx = a_idx
                mb_idx = cur_mb

            session_ids_out[i] = s_idx
            micro_batch_ids_out[i] = mb_idx
            token_ids_out[i] = bs_idx
            expert_offsets_out[i] = k_idx
            y_out[i] = token_data[s_idx, mb_idx, bs_idx, k_idx, :]

            if tokenDtype >= 2 and token_scales is not None:
                dynamic_scale_out[i] = token_scales[s_idx, mb_idx, bs_idx, k_idx]

        group_list_out = _generate_group_list(sorted_valid_ids, valid_count, expertNum)
        actual_token_num_out = np.array([valid_count], dtype=np.int64)

        if kwargs.get("use_numpy", False):
            return [
                y_out,
                group_list_out,
                session_ids_out,
                micro_batch_ids_out,
                token_ids_out,
                expert_offsets_out,
                dynamic_scale_out,
                actual_token_num_out,
            ]

        return [
            _to_torch(y_out, out_torch_dtype),
            _to_torch(group_list_out, None),
            _to_torch(session_ids_out, None),
            _to_torch(micro_batch_ids_out, None),
            _to_torch(token_ids_out, None),
            _to_torch(expert_offsets_out, None),
            _to_torch(dynamic_scale_out, None),
            _to_torch(actual_token_num_out, None),
        ]

    tolerance = {
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int8": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }

    @staticmethod
    def compare(*outputs, **kwargs):
        # 本比较器的输入仅包含八个输出张量及其参考值，不读取消费后的 context
        # 或二级 descriptor 缓冲。因此即使输出全对，也不证明 polling_index
        # 推进正确、已选行 flag/IDs 已清理、未选行保持不变，或下次调用不会重复消费。
        # 这些状态生命周期性质必须通过额外设备状态读回/多次调用测试确认。
        # 接口固定返回八个张量。先验证数量和元数据，不能通过截短到 min(size)
        # 来比较，否则空输出/缺行输出会因“没有发现不同元素”而被误判为通过。
        def failure(message):
            return {"pass": False, "precision": "0%", "error_info": message}

        if len(outputs) != 2 * _NUM_OUTPUTS:
            return [
                failure("expected 8 NPU outputs and 8 golden outputs")
                for _ in range(_NUM_OUTPUTS)
            ]

        def to_np(t):
            if isinstance(t, torch.Tensor):
                from ttk.utilities.dtypes import torch_to_numpy_tensor

                array = torch_to_numpy_tensor(t)
                if "float4" in str(t.dtype):
                    array = array.reshape(*t.shape[:-1], t.shape[-1] * 2)
                return array
            return np.asarray(t)

        npu_outputs = [to_np(t) for t in outputs[:_NUM_OUTPUTS]]
        golden_outputs = [to_np(t) for t in outputs[_NUM_OUTPUTS:]]
        actual_token_num = int(golden_outputs[_OUT_TOKEN_COUNT].flat[0])
        has_scale = golden_outputs[_OUT_Y].dtype.name not in ("float16", "bfloat16")
        results = []
        for idx, (npu, gold) in enumerate(zip(npu_outputs, golden_outputs)):
            if npu.shape != gold.shape or npu.dtype != gold.dtype:
                results.append(
                    failure(
                        f"metadata mismatch [idx={idx}]: {npu.shape}/{npu.dtype} "
                        f"!= {gold.shape}/{gold.dtype}"
                    )
                )
                continue
            # FP16/BF16 没有 scale 载荷，只有数值内容未定义；张量元数据仍须正确。
            if idx == _OUT_SCALE and not has_scale:
                results.append({"pass": True, "precision": "100%", "error_info": None})
                continue
            # y/索引/scale 只保证有效 token 前缀，容量尾部允许未写入。
            # group_list 则要求所有未使用行清零，尤其全 mask 场景不能直接 PASS。
            if idx in _TOKEN_PREFIX_OUTPUTS:
                if not 0 <= actual_token_num <= gold.shape[0]:
                    results.append(failure(f"invalid golden token count [idx={idx}]"))
                    continue
                npu = npu[:actual_token_num]
                gold = gold[:actual_token_num]
            if any(
                kind in gold.dtype.name for kind in ("float8", "float4", "bfloat16")
            ):
                npu = npu.view(np.uint8)
                gold = gold.view(np.uint8)
            diff = int(np.count_nonzero(npu != gold))
            total = gold.size
            passed = diff == 0
            results.append(
                {
                    "pass": passed,
                    "precision": f"{(total - diff) / total * 100}%"
                    if total
                    else "100%",
                    "error_info": None if passed else f"{diff} mismatches [idx={idx}]",
                }
            )
        return results


class TorchFfnWorkerBatchingTestSpec:
    """TTK eager/aclgraph NORM cases with real-device-address context buffers.

    RECV consumes descriptors and cannot safely use TTK's default repeated runs.
    Real-pointer publication/replay needs separate state tests; those scripts
    are not included in the current repository test sources.
    """

    @staticmethod
    def npu_preprocess(
        schedule_context,
        expert_num,
        max_out_shape,
        *,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag: bool = False,
        mx_output_uint8=False,
        **kwargs,
    ):
        # TTK manual input files contain only the context tensor. A fresh worker
        # must restore an explicit archive before invoking the pointer-bearing
        # operator; accepting the old process's raw addresses risks invalid DMA.
        attributes = dict(
            expert_num=expert_num,
            max_out_shape=list(max_out_shape),
            token_dtype=token_dtype,
            need_schedule=need_schedule,
            layer_num=layer_num,
            sync_flag=sync_flag,
            mx_output_uint8=mx_output_uint8,
        )
        archive = os.getenv("FFN_WB_REPLAY_ARCHIVE")
        if archive:
            header, _ = restore_context_archive(archive, attributes)
            schedule_context.reshape(-1)[:_SCHEDULE_CONTEXT_BYTES].copy_(
                torch.from_numpy(header)
            )
        else:
            header = (
                schedule_context.reshape(-1)[:_SCHEDULE_CONTEXT_BYTES]
                .detach()
                .cpu()
                .numpy()
            )
            validate_live_context(header, need_schedule)
        archive = os.getenv("FFN_WB_SAVE_ARCHIVE")
        if archive:
            save_context_archive(archive, header, attributes)

    @staticmethod
    def customize_inputs(
        schedule_context,
        expert_num,
        max_out_shape,
        *,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag: bool = False,
        mx_output_uint8=False,
        **kwargs,
    ):
        # 当前 Torch CSV 限于可重复执行的 NORM。下方错误消息保留历史脚本名，
        # 但仓库没有随附 verify_torch_*.py；RECV 状态测试需另行提供入口。
        if need_schedule != 0:
            raise ValueError(
                "TTK Torch cases require need_schedule=0; use verify_torch_*.py for RECV"
            )
        # Only the 1024-byte header is part of this tensor; secondary buffers
        # have their own HBM allocations and are archived explicitly for replay.
        if schedule_context.numel() < _SCHEDULE_CONTEXT_BYTES:
            raise ValueError("schedule_context must include the 1024-byte header")
        AclnnFfnWorkerBatchingTestSpec.customize_inputs(
            schedule_context.numpy(),
            expert_num,
            max_out_shape,
            token_dtype,
            need_schedule,
            layer_num,
            sync_flag,
            *([None] * 8),
            **kwargs,
        )
        # --no-prof only prepares inputs and never calls npu_preprocess. Save
        # here as well so TTK's documented prepare/replay workflow is complete.
        archive = os.getenv("FFN_WB_SAVE_ARCHIVE")
        if archive:
            attributes = dict(
                expert_num=expert_num,
                max_out_shape=list(max_out_shape),
                token_dtype=token_dtype,
                need_schedule=need_schedule,
                layer_num=layer_num,
                sync_flag=sync_flag,
                mx_output_uint8=mx_output_uint8,
            )
            save_context_archive(archive, schedule_context.numpy(), attributes)

    @staticmethod
    def golden(
        schedule_context,
        expert_num,
        max_out_shape,
        *,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag: bool = False,
        mx_output_uint8=False,
        **kwargs,
    ):
        outputs = AclnnFfnWorkerBatchingTestSpec.golden(
            schedule_context,
            expert_num,
            max_out_shape,
            token_dtype,
            need_schedule,
            layer_num,
            sync_flag,
            *([None] * 8),
            **kwargs,
        )
        # uint8 是图模式传输视图，不是量化/数值转换。E8M0 的指数位模式必须保留；
        # FP4 Torch 最后一维计打包字节，每字节含两个逻辑值，view 不应再乘除 H。
        if mx_output_uint8 and token_dtype >= 3:
            outputs[6] = outputs[6].view(torch.uint8)
            if token_dtype == 5:
                outputs[0] = outputs[0].view(torch.uint8)
        return outputs

    compare = staticmethod(AclnnFfnWorkerBatchingTestSpec.compare)
    tolerance = AclnnFfnWorkerBatchingTestSpec.tolerance


class FfnWorkerBatchingKernelTestSpec:
    """Kernel 接口使用 snake_case 属性及 numpy 数据，复用 ACLNN 的输入协议和 golden。"""

    @staticmethod
    def customize_inputs(
        schedule_context,
        expert_num,
        max_out_shape,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag: bool = False,
        **kwargs,
    ):
        tensor = torch.from_numpy(schedule_context.copy())
        AclnnFfnWorkerBatchingTestSpec.customize_inputs(
            tensor,
            expert_num,
            max_out_shape,
            token_dtype,
            need_schedule,
            layer_num,
            sync_flag,
            *([None] * 8),
            **kwargs,
        )
        # 上下文只需 1024B 头部；CSV 声明的更大 shape 其尾部不再使用
        # （辅助数据走独立 HBM 真实地址缓冲）。
        raw = torch.empty(0, dtype=torch.int8).set_(tensor.untyped_storage())
        return [raw.numpy()]

    @staticmethod
    def golden(
        schedule_context,
        expert_num,
        max_out_shape,
        token_dtype=0,
        need_schedule=0,
        layer_num=0,
        sync_flag: bool = False,
        **kwargs,
    ):
        from ttk.utilities.dtypes import torch_to_numpy_tensor

        outputs = AclnnFfnWorkerBatchingTestSpec.golden(
            schedule_context,
            expert_num,
            max_out_shape,
            token_dtype,
            need_schedule,
            layer_num,
            sync_flag,
            *([None] * 8),
            **kwargs,
        )
        arrays = [torch_to_numpy_tensor(t) for t in outputs]
        if token_dtype == 5:
            arrays[0] = arrays[0].reshape(-1, max_out_shape[3])
        return arrays

    compare = staticmethod(AclnnFfnWorkerBatchingTestSpec.compare)
    tolerance = AclnnFfnWorkerBatchingTestSpec.tolerance


class AclnnFfnWorkerBatchingLegacyTestSpec(AclnnFfnWorkerBatchingTestSpec):
    """V1 has no syncFlag argument; keep its positional contract separate from V2."""

    @staticmethod
    def customize_inputs(
        scheduleContext,
        expertNum,
        maxOutShape,
        tokenDtype,
        needSchedule,
        layerNum,
        y,
        groupList,
        sessionIds,
        microBatchIds,
        tokenIds,
        expertOffsets,
        dynamicScale,
        actualTokenNum,
        *args,
        **kwargs,
    ):
        return AclnnFfnWorkerBatchingTestSpec.customize_inputs(
            scheduleContext,
            expertNum,
            maxOutShape,
            tokenDtype,
            needSchedule,
            layerNum,
            False,
            y,
            groupList,
            sessionIds,
            microBatchIds,
            tokenIds,
            expertOffsets,
            dynamicScale,
            actualTokenNum,
            *args,
            **kwargs,
        )

    @staticmethod
    def golden(
        scheduleContext,
        expertNum,
        maxOutShape,
        tokenDtype,
        needSchedule,
        layerNum,
        y,
        groupList,
        sessionIds,
        microBatchIds,
        tokenIds,
        expertOffsets,
        dynamicScale,
        actualTokenNum,
        *args,
        **kwargs,
    ):
        return AclnnFfnWorkerBatchingTestSpec.golden(
            scheduleContext,
            expertNum,
            maxOutShape,
            tokenDtype,
            needSchedule,
            layerNum,
            False,
            y,
            groupList,
            sessionIds,
            microBatchIds,
            tokenIds,
            expertOffsets,
            dynamicScale,
            actualTokenNum,
            *args,
            **kwargs,
        )
