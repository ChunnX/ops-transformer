#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Customize inputs for aclnnFfnWorkerBatching.

The scheduleContext is a 1024-byte packed struct containing device memory
pointers to token data buffers.  This module:
  1. Generates deterministic test data (expert_ids, token_data, etc.)
  2. Allocates HBM via aclrtMalloc and copies data via aclrtMemcpy
  3. Packs the ScheduleContext struct with HBM pointers
  4. Modifies the scheduleContext tensor in-place
  5. Caches CPU copies for the golden function
"""

import atexit
import ctypes
import struct
import numpy as np
import torch

_DATA_CACHE = {}
_HBM_ALLOCS = []
_ACL_INITIALIZED = False
_ACL_LIB = None

_MASK_VALUE = 1000000


# ---- ScheduleContext byte offsets (verified against kernel source) ----
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

_OFF_TEST_MAGIC = 640
_TEST_MAGIC = 0x54455354


def _load_acl():
    global _ACL_LIB, _ACL_INITIALIZED
    if _ACL_LIB is not None:
        return _ACL_LIB
    lib = ctypes.CDLL('libascendcl.so')
    lib.aclrtMalloc.restype = ctypes.c_int32
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                ctypes.c_size_t, ctypes.c_int32]
    lib.aclrtMemcpy.restype = ctypes.c_int32
    lib.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32]
    lib.aclrtFree.restype = ctypes.c_int32
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    lib.aclrtSetDevice.restype = ctypes.c_int32
    lib.aclrtSetDevice.argtypes = [ctypes.c_int32]
    lib.aclrtSynchronizeDevice.restype = ctypes.c_int32
    lib.aclrtSynchronizeDevice.argtypes = []
    lib.aclrtMemFlush.restype = ctypes.c_int32
    lib.aclrtMemFlush.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.aclrtMemInvalidate.restype = ctypes.c_int32
    lib.aclrtMemInvalidate.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _ACL_LIB = lib
    _ACL_INITIALIZED = True
    return lib


_HBM_BASE = None
_HBM_OFFSET = 0
_HBM_BLOCK_SIZE = 4 * 1024 * 1024  # 4MB block to avoid overlapping with TTK allocations


def _hbm_alloc(size):
    global _HBM_BASE, _HBM_OFFSET
    lib = _load_acl()
    lib.aclrtSetDevice(0)
    if _HBM_BASE is None or _HBM_OFFSET + size > _HBM_BLOCK_SIZE:
        ptr = ctypes.c_void_p(0)
        ret = lib.aclrtMalloc(ctypes.byref(ptr), _HBM_BLOCK_SIZE, 0)
        if ret != 0 or ptr.value is None:
            raise RuntimeError(f"aclrtMalloc failed: ret={ret}, size={_HBM_BLOCK_SIZE}")
        _HBM_BASE = ptr.value
        _HBM_OFFSET = 0
        _HBM_ALLOCS.append(_HBM_BASE)
    aligned_size = (size + 511) & ~511  # align to 512 bytes
    addr = _HBM_BASE + _HBM_OFFSET
    _HBM_OFFSET += aligned_size
    return addr


def _hbm_copy(host_data, hbm_ptr):
    lib = _load_acl()
    if isinstance(host_data, np.ndarray):
        buf = host_data.tobytes()
    elif isinstance(host_data, (bytes, bytearray)):
        buf = bytes(host_data)
    else:
        buf = host_data
    ret = lib.aclrtMemcpy(hbm_ptr, len(buf), ctypes.c_char_p(buf),
                          len(buf), 1)
    if ret != 0:
        raise RuntimeError(f"aclrtMemcpy H2D failed: ret={ret}, dst={hex(hbm_ptr)}, size={len(buf)}")
    lib.aclrtMemInvalidate(ctypes.c_void_p(hbm_ptr), len(buf))


@atexit.register
def _cleanup_hbm():
    global _HBM_BASE, _HBM_OFFSET
    lib = _ACL_LIB
    if lib is None:
        return
    for ptr in _HBM_ALLOCS:
        try:
            lib.aclrtFree(ctypes.c_void_p(ptr))
        except Exception:
            pass
    _HBM_ALLOCS.clear()
    _HBM_BASE = None
    _HBM_OFFSET = 0


def _seed_from_name(name):
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return h


def _pack_ctx(ctx_bytes, offset, fmt, value):
    struct.pack_into(fmt, ctx_bytes, offset, value)


def _generate_test_data(testcase_name, A, M, BS, K, H, expertNum,
                        tokenDtype, needSchedule):
    rng = np.random.RandomState(_seed_from_name(testcase_name))

    out_num = A
    has_mask = 'masked' in testcase_name
    single_expert = 'single_expert' in testcase_name

    if needSchedule == 0:
        expert_ids = rng.randint(0, expertNum,
                                 size=(out_num, BS, K)).astype(np.int32)
    else:
        expert_ids = rng.randint(0, expertNum,
                                 size=(A, M, BS, K)).astype(np.int32)

    if single_expert:
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
    else:
        token_data = rng.randint(-128, 127,
                                 size=(A, M, BS, K, H)).astype(np.int8)
        token_scales = (rng.rand(A, M, BS, K) * 2 + 0.1).astype(np.float32)

    session_ids = np.arange(A, dtype=np.int32)
    micro_batch_ids = np.zeros(A, dtype=np.int32)

    return {
        'A': A, 'M': M, 'BS': BS, 'K': K, 'H': H,
        'expertNum': expertNum, 'tokenDtype': tokenDtype,
        'needSchedule': needSchedule,
        'expert_ids': expert_ids,
        'token_data': token_data,
        'token_scales': token_scales,
        'session_ids_buf': session_ids,
        'micro_batch_ids_buf': micro_batch_ids,
        'out_num': out_num,
        'cur_micro_batch_id': 0,
    }


def _build_token_data_bytes(token_data, token_scales, tokenDtype, A, M, BS, K, H):
    if tokenDtype == 1:
        t = torch.from_numpy(np.ascontiguousarray(token_data, dtype=np.float32))
        t_bf16 = t.to(torch.bfloat16)
        return t_bf16.contiguous().view(torch.uint8).numpy()
    if tokenDtype != 2:
        return np.ascontiguousarray(token_data)

    per_token = H + 4
    buf = np.zeros((A, M, BS, K, per_token), dtype=np.uint8)
    buf[..., :H] = token_data.view(np.uint8)
    scale_bytes = np.ascontiguousarray(token_scales, dtype=np.float32).view(np.uint8).reshape(A, M, BS, K, 4)
    buf[..., H:H + 4] = scale_bytes
    return buf


def _build_token_info_buf(expert_ids, A, M, BS, K):
    F = 2 + BS * K
    token_info = np.zeros((A, M, F), dtype=np.int32)
    for a in range(A):
        for m in range(M):
            token_info[a, m, 0] = 1
            token_info[a, m, 1] = 0
            token_info[a, m, 2:2 + BS * K] = expert_ids[a, m].flatten()
    return token_info


def ffn_worker_batching_inputs(scheduleContext, expertNum, maxOutShape,
                               tokenDtype, needSchedule, layerNum,
                               y, groupList, sessionIds, microBatchIds,
                               tokenIds, expertOffsets, dynamicScale,
                               actualTokenNum, **kwargs):
    """Construct the scheduleContext with HBM buffers.

    Modifies ``scheduleContext`` in-place.  All 14 positional args match the
    aclnnFfnWorkerBatchingGetWorkspaceSize signature (without workspaceSize
    and executor).  Output tensors are pre-allocated placeholders and are
    not modified here.
    """
    testcase_name = kwargs.get('testcase_name', 'default')

    A = int(maxOutShape[0])
    BS = int(maxOutShape[1])
    K = int(maxOutShape[2])
    H = int(maxOutShape[3])
    M = 1

    data = _generate_test_data(
        testcase_name, A, M, BS, K, H, expertNum, tokenDtype, needSchedule)

    token_data = data['token_data']
    token_scales = data['token_scales']
    expert_ids = data['expert_ids']
    session_ids = data['session_ids_buf']
    micro_batch_ids = data['micro_batch_ids_buf']
    out_num = data['out_num']

    if tokenDtype in (0, 1):
        dtype_size = 2
    else:
        dtype_size = 1
    HS = H * dtype_size if tokenDtype != 2 else H + 4

    # --- Build data arrays ---
    td_arr = _build_token_data_bytes(
        token_data, token_scales, tokenDtype, A, M, BS, K, H)
    td_bytes = td_arr.tobytes()

    if needSchedule == 0:
        ei_bytes = np.ascontiguousarray(expert_ids).tobytes()
        si_bytes = np.ascontiguousarray(session_ids).tobytes()
        mi_bytes = np.ascontiguousarray(micro_batch_ids).tobytes()
        aux_data = td_bytes + ei_bytes + si_bytes + mi_bytes
        td_offset = 1024
        ei_offset = td_offset + len(td_bytes)
        si_offset = ei_offset + len(ei_bytes)
        mi_offset = si_offset + len(si_bytes)
    else:
        ti_bytes = _build_token_info_buf(expert_ids, A, M, BS, K).tobytes()
        aux_data = td_bytes + ti_bytes
        td_offset = 1024
        ti_offset = td_offset + len(td_bytes)

    # --- Pack scheduleContext + auxiliary data into a single buffer ---
    total_size = 1024 + len(aux_data)
    ctx = bytearray(total_size)

    _pack_ctx(ctx, _OFF_SESSION_NUM, '<I', A)
    _pack_ctx(ctx, _OFF_MICRO_BATCH_NUM, '<I', M)
    _pack_ctx(ctx, _OFF_MICRO_BATCH_SIZE, '<I', BS)
    _pack_ctx(ctx, _OFF_SELECTED_EXPERT_NUM, '<I', K)
    _pack_ctx(ctx, _OFF_EXPERT_NUM, '<I', expertNum)
    _pack_ctx(ctx, _OFF_ATTN_TO_FFN_TOKEN_SIZE, '<I', HS)
    _pack_ctx(ctx, _OFF_FFN_TO_ATTN_TOKEN_SIZE, '<I', HS)
    _pack_ctx(ctx, _OFF_SCHEDULE_MODE, '<i', 0)

    _pack_ctx(ctx, _OFF_RUN_FLAG, '<i', 1)
    _pack_ctx(ctx, _OFF_TEST_MAGIC, '<I', _TEST_MAGIC)

    _pack_ctx(ctx, _OFF_FFN_TOKEN_DATA_BUF, '<Q', td_offset)
    _pack_ctx(ctx, _OFF_FFN_TOKEN_DATA_BUF_SIZE, '<Q', len(td_bytes))

    if needSchedule == 0:
        _pack_ctx(ctx, _OFF_SESSION_IDS_BUF, '<Q', si_offset)
        _pack_ctx(ctx, _OFF_SESSION_IDS_BUF_SIZE, '<Q', len(si_bytes))
        _pack_ctx(ctx, _OFF_MICRO_BATCH_IDS_BUF, '<Q', mi_offset)
        _pack_ctx(ctx, _OFF_MICRO_BATCH_IDS_BUF_SIZE, '<Q', len(mi_bytes))
        _pack_ctx(ctx, _OFF_EXPERT_IDS_BUF, '<Q', ei_offset)
        _pack_ctx(ctx, _OFF_EXPERT_IDS_BUF_SIZE, '<Q', len(ei_bytes))
        _pack_ctx(ctx, _OFF_OUT_NUM, '<I', out_num)
    else:
        _pack_ctx(ctx, _OFF_FFN_TOKEN_INFO_BUF, '<Q', ti_offset)
        _pack_ctx(ctx, _OFF_FFN_TOKEN_INFO_BUF_SIZE, '<Q', len(ti_bytes))
        _pack_ctx(ctx, _OFF_POLLING_INDEX, '<Q', 0)

    ctx[1024:1024 + len(aux_data)] = aux_data

    # --- Write scheduleContext into tensor ---
    ctx_arr = np.array(ctx, dtype=np.int8)
    cur_storage = scheduleContext.untyped_storage()
    if cur_storage.nbytes() < total_size:
        new_storage = torch.UntypedStorage(total_size)
        scheduleContext.set_(new_storage, scheduleContext.storage_offset(),
                             scheduleContext.shape, scheduleContext.stride())
    scheduleContext.untyped_storage()[:total_size].copy_(
        torch.from_numpy(ctx_arr).untyped_storage())

    # --- Cache CPU data for golden ---
    data['HS'] = HS
    _DATA_CACHE[testcase_name] = data
