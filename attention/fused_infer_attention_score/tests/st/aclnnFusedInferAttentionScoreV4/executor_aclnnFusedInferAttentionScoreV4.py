#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import torch
import logging
import math
import numpy as np
import copy
import time
import random
import ctypes
import copy
import tensorflow as tf
from functools import wraps
from enum import Enum
from collections import defaultdict
from dataclasses import dataclass
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray,pointer

# fia
def get_np_dtype(type_str):
    type_dict = {
        'fp64': np.float64, 'fp32': np.float32, 'fp16': np.float16,
        'int64': np.int64, 'int32': np.int32, 'int16': np.int16, 'int8': np.int8,
        'uint64': np.uint64, 'uint32': np.uint32, 'uint16': np.uint16, 'uint8': np.uint8,
        'bool': np.bool_, 'complex64': np.complex64, 'complex128': np.complex128,
        'complex32': np.float16,
        'bf16': tf.bfloat16.as_numpy_dtype,
        'bfloat16': tf.bfloat16.as_numpy_dtype,
        'float4_e2m1': np.uint8,
        'float4_e1m2': np.uint8,
        'float8_e8m0': np.uint8,
        'hifloat8': np.uint8,
        'fp4_e1m2': np.uint8,
        "fp4_e2m1": np.uint8,
        "fp8_e8m0": np.uint8,
        "float32": np.float32,
        "float16": np.float16,
        "qint8": np.int8,
        "qint32": np.int32,
        "quint8": np.uint8,
        "qint16": np.int16,
        "uint1": np.uint8,
        "quint16": np.uint16,
        "fp4_e2m1_as_fp32": np.float32
    }
    if type_str == "int4":
        from ml_dtypes import int4
        return int4
    elif type_str == "float8_e5m2":
        from ml_dtypes import float8_e5m2
        return float8_e5m2
    elif type_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    else:
        return type_dict[type_str]

def get_pt_dtype(type_str):
    type_dict = {
        'fp32': torch.float32, 'fp16': torch.float16, 'fp64': torch.float64,
        'int8': torch.int8, 'int16': torch.int16, 'int32': torch.int32, 'int64': torch.int64,
        'uint8': torch.uint8, 'bool': torch.bool, 'complex64': torch.complex64,
        'complex128': torch.complex128, 'bf16': torch.bfloat16, 'uint1': torch.uint8
    }
    if type_str == 'hifloat8':
        import torch_npu
        return torch_npu.hifloat8
    return type_dict[type_str]

def cvt_hifuint8_to_float(x, over_mode=True):
    x = int(x)
    if x == 0:
        return float(0)
    elif x == 128:
        if over_mode:
            return np.nan
        else:
            return float(0)
    elif x == 239:
        if over_mode:
            return -np.inf
        else:
            return -32768
    elif x == 111:
        if over_mode:
            return np.inf
        else:
            return 32768
    else:
        if x >= 128:
            sign = -1.0
        else:
            sign = 1.0
        dot_4_bits = x & 120 #b01111000 = 120
        dot_4_value = dot_4_bits >> 3
        if dot_4_value >= 12:
            #b1100 =12 D4
            exponet = x & 30 #b00011110 = 30
            exponet_int = exponet >> 1
            if exponet_int >= 8:
                #b1000 = 8
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 8

            fra_int = x & 1 #b00000001
            m_value = 1.0 + fra_int * 0.5
        elif dot_4_value >= 8:
            #b1000 =8 D3
            exponet = x & 28 #b00011100 = 28
            exponet_int = exponet >> 2
            if exponet_int >= 4:
                #b100 = 4
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 4
            fra_int = x & 3 #b00000011
            m_value = 1.0 + fra_int * 0.25
        elif dot_4_value >= 4:
            #b0100 =8 D2
            exponet = x & 24  # b00011000 = 24
            exponet_int = exponet >> 3
            if exponet_int >= 2:
                # b10 = 2
                exponet_value = -exponet_int
            else:
                exponet_value = exponet_int + 2
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value >= 2:
            #b0010 =2 D1
            exponet = x & 8 # b00001000 = 8
            exponet_sign = exponet >> 3
            if exponet_sign >= 1:
                # b10 = 2
                exponet_value = -1
            else:
                exponet_value = 1
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == 1:
            #d0
            exponet_value = 0
            fra_int = x & 7  # b00000111
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == 0:
            #dml
            m_value = 1
            exponet_value = (x & 7) - 23  # b00000111 = 7
        else:
            print("error,dot error")
            m_value = 0.0
            exponet_value = 0
        return sign*pow(2.0, exponet_value)*m_value

def cvt_fp4_e1m2_to_bfloat16(x):
    Fp4e1m2ToBf16 = {'0': 0.0, '1': 0.25, '2': 0.5, '3': 0.75,
                     '4': 1.0, '5': 1.25, '6': 1.5, '7': 1.75,
                     '8': -0.0, '9': -0.25, '10': -0.5, '11': -0.75,
                     '12': -1.0, '13': -1.25, '14': -1.5, '15': -1.75
                     }
    x = int(x)
    first_fp4val = x & 0x0f
    first_fp4str = str(first_fp4val)
    return Fp4e1m2ToBf16[first_fp4str]

def new_trans_np_fp4_e2m1_tensor_to_bfloat16(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)
    bfloat16_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        bfloat16_tensor[i] = cvt_fp4_e2m1_to_bfloat16(in_tensor[i])
    return bfloat16_tensor.reshape(shape_tensor)

def new_trans_np_fp4_e1m2_tensor_to_bfloat16(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)
    bfloat16_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        bfloat16_tensor[i] = cvt_fp4_e1m2_to_bfloat16(in_tensor[i])
    return bfloat16_tensor.reshape(shape_tensor)

def trans_np_hifuint8_tensor_to_float32(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    out_tensor = np.zeros(multi_shape).astype(np.float32)
    in_tensor = in_tensor.reshape(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = cvt_hifuint8_to_float(in_tensor[i])
    out_tensor = out_tensor.reshape(shape_tensor).astype(np.float32)
    return out_tensor

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

FIA_ENABLE_DEBUG = False
FIA_ENABLE_INFO = True
FIA_USE_TORCH_CPU = True

FIA_ENABLE_DEBUG_DATA = False
FIA_ENABLE_DEBUG_FIA_TENSOR = False
FIA_ENABLE_FUNC_BEGIN_PRINT = False

FIA_ALL_ONE_DEBUG = False

_performance_data = defaultdict(lambda: {'total_time': 0.0, 'calls': 0})


def timeit_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        # 更新性能数据
        func_name = func.__qualname__  # 这会包含类名和函数名，例如 MyClass.method
        _performance_data[func_name]['total_time'] += duration
        _performance_data[func_name]['calls'] += 1
        return result

    return wrapper


def print_performance_report(n=10, sort_by='total_time'):
    # 计算平均耗时
    data = []
    for func_name, stats in _performance_data.items():
        total_time = stats['total_time']
        calls = stats['calls']
        avg_time = total_time / calls if calls > 0 else 0
        data.append((func_name, total_time, avg_time, calls))
    # 排序
    if sort_by == 'total_time':
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    elif sort_by == 'avg_time':
        sorted_data = sorted(data, key=lambda x: x[2], reverse=True)
    else:
        raise ValueError("sort_by must be 'total_time' or 'avg_time'")
    # 打印报告
    print(f"{'Function':<40} {'Total Time (s)':<15} {'Avg Time (s)':<15} {'Calls':<10}")
    print('-' * 80)
    for item in sorted_data[:n]:
        func_name, total_time, avg_time, calls = item
        print(f"{func_name:<40} {total_time:15.6f} {avg_time:15.6f} {calls:10}")


def fia_debug_fia_tensor(string):
    if FIA_ENABLE_DEBUG_FIA_TENSOR:
        print(f"[DEBUG] {string}")


def fia_debug_data(string, data):
    if FIA_ENABLE_DEBUG_DATA:
        print(f"[DEBUG] {string} {data.shape} | {data.flatten().round(4)[:10]}")


def fia_debug_func_begin(string):
    if FIA_ENABLE_FUNC_BEGIN_PRINT:
        print(f"[DEBUG] {string}")


def fia_debug(string):
    if FIA_ENABLE_DEBUG:
        print(f"[DEBUG] {string}")


def fia_info(string):
    if FIA_ENABLE_INFO:
        print(f"[INFO] {string}")


def fia_warn(string):
    print(f"[WARNING] {string}")


class StorageMode(Enum):
    CONTIGUOES = 0
    PAGE_ATTENTION = 1
    TENSOR_LIST = 2


class FiaTensor():
    def __init__(self, data, shape, dtype, layout, head_nums=None, actual_seq_lens=None, name=None):
        debug_str = f"FiaTensor {name} init, shape: {shape}, layout: {layout}, dtype: {dtype}, head_nums: {head_nums}"
        if actual_seq_lens is not None:
            debug_str += f", actual_seq_lens: {actual_seq_lens}"
        fia_debug_fia_tensor(debug_str)
        if FIA_ALL_ONE_DEBUG:
            self._data = self.init_ones(data, data.shape)
        else:
            self._data = data
        self._shape = shape
        self._layout = layout
        self._bnsd_shape = None
        self._bnsd_data = None
        self._bsnd_shape = None
        self._bsnd_data = None
        self._dtype = self._get_dtype(dtype)
        self._np_dtype = self.get_np_dtype(dtype)
        self._head_nums = head_nums
        self._actual_seq_lens = actual_seq_lens
        self._name = name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._shape = list(value.shape)

    @property
    def shape(self):
        return self._shape

    @property
    def layout(self):
        return self._layout

    @property
    def dtype(self):
        return self._dtype

    @property
    def np_dtype(self):
        return self._np_dtype

    @property
    def bnsd_data(self):
        if self._bnsd_data is None:
            self._bnsd_data = self.to_bnsd()
        return self._bnsd_data

    @property
    def bnsd_shape(self):
        if self._bnsd_shape is None:
            self._bnsd_data = self.to_bnsd()
        return list(self._bnsd_data.shape)

    @property
    def bsnd_data(self):
        if self._bsnd_data is None:
            self._bsnd_data = self.to_bsnd()
        return self._bsnd_data

    @property
    def bsnd_shape(self):
        if self._bsnd_shape is None:
            self._bsnd_data = self.to_bsnd()
        return list(self._bsnd_data.shape)

    def _get_axis(self, axis_name):
        if self.layout != 'BNDSD0' and len(self.layout) != len(self.shape):
            raise RuntimeError(
                f"{self._name} the length of layout {self.layout} and shape {self.shape} should be equal")

        if self.layout == 'BNDSD0' and len(self.shape) != 5:
            raise RuntimeError(f"{self._name} layout is BNDSD0, len(self.shape) = {len(self.shape)} should be 5")

        if axis_name not in self.layout:
            raise RuntimeError(f"{self._name} layout {self.layout} do not have axis {axis_name}")

        return self.shape[self.layout.index(axis_name)]

    @property
    def B(self):
        if self.is_tnd_like_layout():
            if not self._actual_seq_lens:
                raise RuntimeError(f"{self._name} layout is {self.layout}, actual_seq_lens should be exists")
            return len(self._actual_seq_lens)
        return self._get_axis('B')

    @property
    def N(self):
        if 'N' in self.layout:
            return self._get_axis('N')
        return self._head_nums

    @property
    def S(self):
        if self.is_tnd_like_layout():
            if not self._actual_seq_lens:
                raise RuntimeError(f"{self._name} layout is {self.layout}, actual_seq_lens should be exists")
            return max(self._actual_seq_lens)
        return self._get_axis('S')

    @property
    def D(self):
        if 'H' in self.layout:
            if self._head_nums is None:
                raise ValueError(f"{self._name} layout is {self.layout}, head_nums should not be None")
            if self.H % self._head_nums != 0:
                raise RuntimeError(f"H({self.H}) % head_nums({self._head_nums}) should be 0")
            return int(self.H // self._head_nums)
        return self._get_axis('D')

    @property
    def T(self):
        return self._get_axis('T')

    @property
    def H(self):
        return self._get_axis('H')

    def empty(self):
        return 0 in self.shape

    def is_tnd_like_layout(self):
        return self.layout in ["TND", "NTD"]

    @staticmethod
    def _get_dtype(input_dtype):
        if input_dtype == 'fp16':
            return 'float16'
        elif input_dtype == 'int8':
            return 'int8'
        elif input_dtype == 'uint8':
            return 'uint8'
        elif input_dtype == 'bf16':
            return 'bfloat16'
        elif input_dtype == 'bool':
            return 'bool'
        elif input_dtype == 'int32':
            return 'int32'
        elif input_dtype == 'fp32':
            return 'float32'
        elif input_dtype == 'int4':
            return 'int4'
        else:
            return input_dtype

    @classmethod
    def transpose(cls, data, dims):
        if isinstance(data, np.ndarray):
            return data.transpose(dims)
        elif isinstance(data, torch.Tensor):
            return data.permute(dims).contiguous()
        else:
            raise RuntimeError(f"Unsupported dtype {type(data)}")

    @classmethod
    def init_zeros(cls, data, shape):
        if isinstance(data, np.ndarray):
            return np.zeros(shape, dtype=data.dtype)
        elif isinstance(data, torch.Tensor):
            return torch.zeros(size=shape, dtype=data.dtype)
        else:
            raise RuntimeError(f"Unsupported dtype {type(data)}")

    @classmethod
    def init_ones(cls, data, shape):
        if isinstance(data, np.ndarray):
            return np.ones(shape, dtype=data.dtype)
        elif isinstance(data, torch.Tensor):
            return torch.ones(size=shape, dtype=data.dtype)
        else:
            raise RuntimeError(f"Unsupported dtype {type(data)}")
    
    def get_np_dtype(self, type_str):
        type_dict = {
            'fp64': np.float64, 'fp32': np.float32, 'fp16': np.float16,
            'int64': np.int64, 'int32': np.int32, 'int16': np.int16, 'int8': np.int8,
            'uint64': np.uint64, 'uint32': np.uint32, 'uint16': np.uint16, 'uint8': np.uint8,
            'bool': np.bool_, 'complex64': np.complex64, 'complex128': np.complex128,
            'complex32': np.float16,
            'bf16': tf.bfloat16.as_numpy_dtype,
            'bfloat16': tf.bfloat16.as_numpy_dtype,
            'float4_e2m1': np.uint8,
            'float4_e1m2': np.uint8,
            'float8_e8m0': np.uint8,
            'hifloat8': np.uint8,
            'fp4_e1m2': np.uint8,
            "fp4_e2m1": np.uint8,
            "fp8_e8m0": np.uint8,
            "float32": np.float32,
            "float16": np.float16,
            "qint8": np.int8,
            "qint32": np.int32,
            "quint8": np.uint8,
            "qint16": np.int16,
            "uint1": np.uint8,
            "quint16": np.uint16,
            "fp4_e2m1_as_fp32": np.float32
        }
        if type_str == "int4":
            from ml_dtypes import int4
            return int4
        elif type_str == "float8_e5m2":
            from ml_dtypes import float8_e5m2
            return float8_e5m2
        elif type_str == "float8_e4m3fn":
            from ml_dtypes import float8_e4m3fn
            return float8_e4m3fn
        else:
            return type_dict[type_str]

    def to_bnsd(self):
        if self.layout == "BNSD":
            return self.data
        elif self.layout == "BSH":
            return self.transpose(self.data.reshape(self.B, -1, self._head_nums, self.D), (0, 2, 1, 3))
        elif self.layout == "BSND":
            return self.transpose(self.data, (0, 2, 1, 3))
        elif self.layout == "NBSD":
            return self.transpose(self.data, (1, 0, 2, 3))
        elif self.layout == "TND" or self.layout == "NTD":
            if self._actual_seq_lens is None:
                raise RuntimeError(f"layout is {self.layout}, actual_seq_lens should not be None")
            output_data = self.init_zeros(self.data, (self.B, self.N, self.S, self.D))
            data = self.transpose(self.data, (1, 0, 2)) if self.layout == "NTD" else self.data
            t_start = 0
            for b_idx in range(self.B):
                act_s = self._actual_seq_lens[b_idx]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_idx in range(self.N):
                    output_data[b_idx, n_idx, 0:act_s, :] = data[t_start:t_end, n_idx, :]
                t_start += act_s
            return output_data
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def to_bsnd(self):
        return self.transpose(self.bnsd_data, (0, 2, 1, 3))

    def to_layout(self, dst_layout, actual_seq_lens=None):
        if self.layout != "BNSD":
            raise ValueError(f"Unsupported layout {self.layout}")

        B, N, S, D = self.shape
        if dst_layout == "BSH":
            H = N * D
            return self.transpose(self.data, [0, 2, 1, 3]).reshape(B, S, H)
        elif dst_layout == "BSND":
            return self.transpose(self.data, [0, 2, 1, 3])
        elif dst_layout == "NBSD":
            return self.transpose(self.data, [1, 0, 2, 3])
        elif dst_layout == "TND":
            if actual_seq_lens is None:
                raise ValueError("actual_seq_lens must be provided for TND layout.")

            T = sum(actual_seq_lens)
            output_data = self.init_zeros(self.data, (T, N, D))
            t_start = 0

            for b_idx in range(B):
                act_s = actual_seq_lens[b_idx]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                # 将批次b_idx的数据填充到output[t_start:t_end]
                output_data[t_start:t_end, :, :] = self.transpose(self.data[b_idx, :, :act_s, :], [1, 0, 2])
                t_start += act_s

            return output_data
        elif dst_layout == "NTD":
            if actual_seq_lens is None:
                raise ValueError("actual_seq_lens must be provided for NTD layout.")
            # 先转换为TND，再转置为NTD
            tnd_tensor = self.to_layout("TND", actual_seq_lens)
            return self.transpose(tnd_tensor, [1, 0, 2])
        elif dst_layout == "BNSD":
            return self.data
        else:
            raise RuntimeError(f"Unsupported dst layout: {dst_layout}")

    def trans_to_1n1d(self):
        shape_len = len(self.data.shape)
        if shape_len == 1:
            d = self.data.shape[0] // self._head_nums
            _1n1d = self.data.reshape(((1, self._head_nums, 1, d)))
            self.data = _1n1d
        elif shape_len == 2:
            d = self.data.shape[1]
            _1n1d = self.data.reshape(((1, self._head_nums, 1, d)))
            self.data = _1n1d
        elif shape_len == 3:
            if self.data.shape[0] == 1 and self.data.shape[1] == 1:
                d = self.data.shape[-1] // self._head_nums
                _1n1d = self.data.reshape(((1, self._head_nums, 1, d)))
                self.data = _1n1d
        elif shape_len == 4:
            if self.data.shape[0] == 1 and self.data.shape[1] == 1:
                d = self.data.shape[-1]
                _1n1d = self.data.reshape(((1, self._head_nums, 1, d)))
                self.data = _1n1d
        else:
            print("layout do not support")
        # 根据input将data变成1n1d
        pass


class FiaTensorList(FiaTensor):
    def __init__(self, data_list, shape_list, dtype, layout, head_nums=None, actual_seq_lens=None, name=None):
        print()
        debug_str = f"FiaTensorList {name} init, shape_list: {shape_list}, layout: {layout}, dtype: {dtype}, head_nums: {head_nums}"
        if actual_seq_lens is not None:
            debug_str += f", actual_seq_lens: {actual_seq_lens}"
        fia_debug_fia_tensor(debug_str)

        if len(data_list) != len(shape_list):
            raise ValueError(
                f"FiaTensorList {name} len(data_list) = {len(data_list)} should be equal to len(shape_list) = {len(shape_list)}")
        if len(data_list) == 0:
            raise ValueError(f"FiaTensorList {name} len(data_list) should not be 0")

        super().__init__(data_list[0], shape_list[0], dtype, layout, head_nums, actual_seq_lens, name)
        self._tensor_list = [FiaTensor(d, s, dtype, layout, head_nums, actual_seq_lens, name) for d, s in
                             zip(data_list, shape_list)]
        self._data_list = data_list
        self._shape_list = shape_list
        self._bnsd_data_list = None
        self._bnsd_shape_list = None
        self._len = len(self._tensor_list)
        self._name = name

    @property
    def S(self):
        S = 0
        for idx in range(self._len):
            S = max(self._tensor_list[idx].S, S)
        return S

    @property
    def kv_s_list(self):
        return [tensor.S for tensor in self._tensor_list]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self._tensor_list[index]

    @property
    def tensor_list(self):
        return self._tensor_list

    @property
    def data_list(self):
        return self._data_list

    @property
    def shape_list(self):
        return self._shape_list

    @property
    def tensor(self):
        return self._tensor_list[0]

    def trans_to_bnsd_list(self):
        data_list = [tensor.bnsd_data for tensor in self._tensor_list]
        shape_list = [tensor.bnsd_shape for tensor in self._tensor_list]
        return data_list, shape_list

    @property
    def bnsd_data_list(self):
        if self._bnsd_data_list is None:
            self._bnsd_data_list, self._bnsd_shape_list = self.trans_to_bnsd_list()
        return self._bnsd_data_list

    @property
    def bnsd_shape_list(self):
        if self._bnsd_shape_list is None:
            self._bnsd_data_list, self._bnsd_shape_list = self.trans_to_bnsd_list()
        return self._bnsd_shape_list


def need_gen_input(action_type):
    if "output" in action_type:
        return False
    return True


def get_slopes(n_heads):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m


def concat_common(tensors, axis):
    if isinstance(tensors[0], np.ndarray):
        return np.concatenate(tensors, axis=axis)
    elif isinstance(tensors[0], torch.Tensor):
        return torch.cat(tensors, dim=axis)
    else:
        raise RuntimeError(f"Unsupported data type {type(tensors[0])}")


def concat_tensor(tensor1, tensor2):
    # Check if the number of dimensions is the same
    if len(tensor1.shape) != len(tensor2.shape):
        raise ValueError(
            f"Number of dimensions mismatch: {len(tensor1.shape)} vs {len(tensor2.shape)}"
        )

    # Check if the layouts are the same
    if tensor1.layout != tensor2.layout:
        raise ValueError(
            f"Layout mismatch: {tensor1.layout} vs {tensor2.layout}"
        )

    # Check if the dtypes are the same
    if tensor1.dtype != tensor2.dtype:
        raise ValueError(
            f"Dtype mismatch: {tensor1.dtype} vs {tensor2.dtype}"
        )

    # Check if the head numbers are the same
    if tensor1._head_nums != tensor2._head_nums:
        raise ValueError(
            f"Head numbers mismatch: {tensor1._head_nums} vs {tensor2._head_nums}"
        )

    # Get tensor properties
    layout = tensor1.layout
    dtype = tensor1.dtype
    head_nums = tensor1._head_nums
    actual_seq_lens = tensor1._actual_seq_lens

    # Concatenate based on the layout
    if 'D' in layout:
        # Concatenate along the 'D' axis
        d_axis = layout.index('D')
        concat_data = concat_common((tensor1.data, tensor2.data), d_axis)
        concat_shape = list(concat_data.shape)
    elif layout == 'BSH':
        # For 'BSH' layout, reshape to 4D and concatenate along the 4th dimension
        concat_shape = [
            tensor1.shape[0],
            tensor1.shape[1],
            tensor1.shape[2] + tensor2.shape[2]
        ]
        tensor1_data_tmp = tensor1.data.reshape(tensor1.B, tensor1.S, tensor1.N, tensor1.D)
        tensor2_data_tmp = tensor2.data.reshape(tensor2.B, tensor2.S, tensor2.N, tensor2.D)
        concat_data = concat_common((tensor1_data_tmp, tensor2_data_tmp), 3).reshape(concat_shape)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    # Create and return the concatenated tensor
    return FiaTensor(concat_data, concat_shape, dtype, layout, head_nums, actual_seq_lens)


def concat_tensor_list(tensor_list1, tensor_list2):
    tensor1 = None
    tensor2 = None
    if isinstance(tensor_list1, FiaTensorList):
        if len(tensor_list1) != 1:
            raise ValueError(f"tensor_list1 len = {len(tensor_list1)} must be 1")
        tensor1 = tensor_list1[0]
    else:
        tensor1 = tensor_list1

    if isinstance(tensor_list2, FiaTensorList):
        if len(tensor_list2) != 1:
            raise ValueError(f"tensor_list2 len = {len(tensor_list2)} must be 1")
        tensor2 = tensor_list2[0]
    else:
        tensor2 = tensor_list2

    tensor_concat = concat_tensor(tensor1, tensor2)
    tensor_concat_list = FiaTensorList([tensor_concat.data], [tensor_concat.shape], tensor_concat.dtype,
                                       tensor_concat.layout, tensor_concat._head_nums, tensor_concat._actual_seq_lens)
    return tensor_concat_list


def get_attention_mask_batch_num(npu_m_shape, is_bs):
    batch, numhead = None, None
    if len(npu_m_shape) == 2:
        if is_bs:
            batch = npu_m_shape[0]
            s1 = 1
            s2 = npu_m_shape[1]
        else:
            s1 = npu_m_shape[0]
            s2 = npu_m_shape[1]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 3:
        batch = npu_m_shape[0]
        s1 = npu_m_shape[1]
        s2 = npu_m_shape[2]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 4:
        batch = npu_m_shape[0]
        numhead = npu_m_shape[1]
        s1 = npu_m_shape[2]
        s2 = npu_m_shape[3]
        return batch, numhead, s1, s2


def _np_broadcast_mask_n(m_tensor, m_shape, cpu_m_shape, numheads, q_batch):
    print(f"broadcast_mask_n:mask shape:{m_shape} with numheads:{numheads} q_batch:{q_batch}")
    mask_cur_shape = cpu_m_shape
    if len(m_shape) == 4:
        # b1ss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[2], m_shape[3]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i:i + 1, :, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 3:
        # bss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[1], m_shape[2]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i:i + 1, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 2:
        # ss
        B = q_batch
        m_res = []
        for i in range(B):
            m_res.append(m_tensor)
        return m_res
    else:
        return m_tensor


def quant(x, qscale, qoffset):
    """
    优化版本：使用矩阵和向量运算替代嵌套循环

    参数说明：
    x: 输入张量，形状为 [N, C, H, W]
    qscale: 量化尺度（标量）
    qoffset: 量化偏移（标量）
    """
    # 将输入转换为half精度
    x_half = np.half(x)

    # 将量化参数转换为half精度
    qscale_half = np.half(qscale)
    qoffset_half = np.half(qoffset)

    # 向量化计算：x * qscale + qoffset
    intermediate = x_half * qscale_half + qoffset_half

    # 应用向量化的饱和函数
    s9_result = s9_saturation_vectorized(intermediate)
    rounded = np.round(s9_result)
    s8_res_cal = s8_saturation_vectorized(rounded)

    return s8_res_cal


def quant_pc(x, qscale, qoffset, n1):
    """
    优化版本：使用矩阵和向量运算替代嵌套循环

    参数说明：
    x: 输入张量，形状为 [N, C, H, W]
    qscale: 量化尺度，形状为 [1, n1, 1, W]
    qoffset: 量化偏移，形状为 [1, n1, 1, W]
    n1: 索引参数
    """
    # 将输入转换为half精度
    x_half = np.half(x)

    # 提取对应的量化参数（广播到与x相同的形状）
    qscale_broadcast = np.half(qscale[0, n1, 0, :])  # 形状: [W]
    qoffset_broadcast = np.half(qoffset[0, n1, 0, :])  # 形状: [W]

    # 重塑量化参数以便进行广播运算
    # 将 [W] 重塑为 [1, 1, 1, W] 以便与 [N, C, H, W] 进行广播
    qscale_reshaped = qscale_broadcast.reshape(1, 1, 1, -1)
    qoffset_reshaped = qoffset_broadcast.reshape(1, 1, 1, -1)

    # 向量化计算：x * qscale + qoffset
    intermediate = x_half * qscale_reshaped + qoffset_reshaped

    # 应用向量化的饱和函数
    s9_result = s9_saturation_vectorized(intermediate)
    rounded = np.round(s9_result)
    s8_res_cal = s8_saturation_vectorized(rounded)

    return s8_res_cal


def s8_saturation_vectorized(inputdata):
    """向量化的s8饱和函数"""
    # 使用np.where替代if-else条件判断
    saturated = np.where(inputdata > 127, 127,
                         np.where(inputdata < -128, -128, inputdata))
    return saturated.astype(np.int8)


def s9_saturation_vectorized(inputdata):
    """向量化的s9饱和函数"""
    # 使用np.where替代if-else条件判断
    return np.where(inputdata > 255, 255,
                    np.where(inputdata < -256, -256, inputdata))


class FiaSoftmax():
    @staticmethod
    def softmaxv1(x):
        x = x.astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y / x_sum
        return ans, x_max, x_sum

    @staticmethod
    def softmax(x, sinks=None):
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        x_max = np.max(x, axis=-1, keepdims=True)
        x -= x_max
        np.exp(x, out=x)
        x_sum = np.sum(x, axis=-1, keepdims=True, dtype=np.float32)
        if sinks is not None:
            sinks_reshaped = sinks.reshape(1, -1, 1, 1)
            sinks_sub = sinks_reshaped - x_max
            sink_exp = np.exp(sinks_sub)
            x_sum += sink_exp.sum(axis=-1, keepdims=True)
        return x, x_sum, x_max

    @staticmethod
    def _t_softmax(x):
        x_max = torch.max(x, dim=-1, keepdims=True)[0]
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True)
        ans = y.div(x_sum)
        return ans, x_max, x_sum


class MaskGenerator():
    @staticmethod
    def _random_fill_tensor(tensor, shape, random_number, value=0):
        for i in range(0, random_number):
            point = []
            for k in range(0, len(shape)):
                point.append(random.randint(1, shape[k]) - 1)
            tensor[point[0], point[1]] = value
        return tensor

    @classmethod
    def _create_mask_right_down(cls, m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV,
                                actualprefixKV,
                                prefix_kvs, batch, numheads, kv_s_list, m_dtype):
        mask_s_q = m_shape[0]
        mask_s_kv = m_shape[1]

        next_tokens_list = []
        re_mask_batch = []
        for i in range(batch):
            if len(actualSeqLengths) == 0:
                S1 = mask_s_q
            else:
                S1 = actualSeqLengths[i]
            if len(actualSeqLengthsKV) != 0:
                S2 = actualSeqLengthsKV[i] + actualprefixKV
            elif len(kv_s_list) > 1:
                S2 = kv_s_list[i] + actualprefixKV
            else:
                S2 = mask_s_kv - prefix_kvs + actualprefixKV
            next_tokens = S2 - S1
            next_tokens_list.append(next_tokens)
            atten_masks = cls._create_mask(m_shape, pre_tokens, next_tokens)
            re_mask_batch.append(atten_masks)
        return re_mask_batch, next_tokens_list

    @staticmethod
    def _create_mask(m_shape, pre_tokens, next_tokens):
        next_masks = np.triu(np.ones(m_shape, dtype='uint8'), k=1 + int(next_tokens))  # 生成下三角全是0的矩阵
        pre_mask = np.tril(np.ones(m_shape, dtype='uint8'), k=-1 - int(pre_tokens))  # 生成上三角全是0的矩阵
        atten_masks = pre_mask + next_masks

        return atten_masks

    @classmethod
    def _create_mask_no_sparse(cls, m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype,
                               random_ones=0):
        re_mask_batch = []
        re_mask_npu_batch = []

        pad_flag = False
        npu_mask = None
        if m_shape[0] != npu_m_shape[0] or m_shape[1] != npu_m_shape[1]:
            pad_flag = True
            npu_mask = np.ones(npu_m_shape, dtype='uint8')
        cpu_mask = cls._create_mask(m_shape, pre_tokens, next_tokens)
        if pad_flag:
            if batch == None:
                cpu_mask = cls._random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
                npu_mask[:cpu_mask.shape[0], :cpu_mask.shape[1]] = cpu_mask
                return cpu_mask, npu_mask
            for i in range(batch):
                re_mask_num = []
                re_mask_npu_num = []
                re_mask = cls._random_fill_tensor(cpu_mask, m_shape, random_ones, 1)

                npu_mask[:re_mask.shape[0], :re_mask.shape[1]] = re_mask
                if numheads:
                    # cpu
                    re_mask_num.append(re_mask)
                    re_mask_batch.append(re_mask_num)
                    # npu

                    re_mask_npu_num.append(npu_mask)
                    re_mask_npu_batch.append(re_mask_npu_num)
                else:
                    re_mask_batch.append(re_mask)
                    re_mask_npu_batch.append(npu_mask)
            cpu_mask = np.array(re_mask_batch).astype(m_dtype)
            npu_mask = np.array(re_mask_npu_batch).astype(m_dtype)
            return cpu_mask, npu_mask
        else:
            if batch == None:
                cpu_mask = cls._random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
                return cpu_mask, cpu_mask
            for i in range(batch):
                re_mask_num = []
                re_mask = cls._random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
                if numheads:
                    re_mask_num.append(re_mask)
                    re_mask_batch.append(re_mask_num)
                else:
                    re_mask_batch.append(re_mask)

            cpu_mask = np.array(re_mask_batch).astype(m_dtype)
            return cpu_mask, cpu_mask

    @classmethod
    def _create_mask_left_up(cls, m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype, random_ones=0):
        re_mask_batch = []
        attentionmask = cls._create_mask(m_shape, pre_tokens, next_tokens)
        for i in range(batch):
            re_mask_batch.append(attentionmask)
        return re_mask_batch

    @classmethod
    def _create_mask_band(cls, m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, actualprefixKV,
                          prefix_kvs, batch, numheads, kv_s_list, m_dtype):
        mask_s_q = m_shape[0]
        mask_s_kv = m_shape[1]

        pre_tokens_list = []
        next_tokens_list = []
        re_mask_batch = []
        for i in range(batch):
            if len(actualSeqLengths) == 0:
                S1 = mask_s_q
            else:
                S1 = actualSeqLengths[i]
            if len(actualSeqLengthsKV) != 0:
                S2 = actualSeqLengthsKV[i] + actualprefixKV
            elif len(kv_s_list) > 1:
                S2 = kv_s_list[i] + actualprefixKV
            else:
                S2 = mask_s_kv - prefix_kvs + actualprefixKV
            pre_tokens_new = S1 - S2 + pre_tokens
            pre_tokens_list.append(pre_tokens_new)

            next_tokens_new = S2 - S1 + next_tokens
            next_tokens_list.append(next_tokens_new)
            atten_masks = cls._create_mask(m_shape, pre_tokens_new, next_tokens_new)
            re_mask_batch.append(atten_masks)
        return re_mask_batch, pre_tokens_list, next_tokens_list

    @classmethod
    def _create_random_mask_by_spars(cls, cpu_m_shape, npu_m_shape, m_dtype, pre_tokens, next_tokens, actualSeqLengths,
                                     actualSeqLengthsKV, actualprefixKV, prefix_kvs, kv_s_list, batch=1, numheads=1,
                                     sp_mode=0, random_ones=0):
        # mask shape [sq,skv]  #mshape  npu  fshape cpu
        print(
            f"[_create_random_mask_by_spars] full_m_shape:{cpu_m_shape} m_shape:{npu_m_shape} datype:{m_dtype} pret:{pre_tokens} nextt:{next_tokens} sp_mode:{sp_mode}")
        if sp_mode == 0:
            cpu_mask, npu_mask = cls._create_mask_no_sparse(cpu_m_shape, npu_m_shape, pre_tokens, next_tokens, batch,
                                                            numheads,
                                                            m_dtype, random_ones)
            return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
        if sp_mode == 1:
            print(f"[_create_random_mask_by_spars] sp_mode is 1 return all zero mask")
            pre_tokens = 214748647
            next_tokens = 214748647
            cpu_mask, npu_mask = cls._create_mask_no_sparse(cpu_m_shape, npu_m_shape, pre_tokens, next_tokens, batch,
                                                            numheads,
                                                            m_dtype, random_ones)
            return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens

        if sp_mode == 2:
            pre_tokens = 214748647
            next_tokens = 0
            print(f"[_create_random_mask_by_spars] sp_mode is 2 npu mask shape:{npu_m_shape}")
            npu_mask = np.triu(np.ones(npu_m_shape), k=1)
            cpu_mask = cls._create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype)
            return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
        if sp_mode == 3:  # rightdown
            pre_tokens = 214748647
            print(f"[_create_random_mask_by_spars] sp_mode is 3 npu mask shape:{npu_m_shape}")
            npu_mask = np.triu(np.ones(npu_m_shape), k=1)
            cpu_mask, next_tokens_new = cls._create_mask_right_down(cpu_m_shape, pre_tokens, next_tokens,
                                                                    actualSeqLengths,
                                                                    actualSeqLengthsKV, actualprefixKV, prefix_kvs,
                                                                    batch,
                                                                    numheads, kv_s_list, m_dtype)
            return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens_new
        if sp_mode == 4:
            npu_mask = np.triu(np.ones(npu_m_shape), k=1)
            cpu_mask, pre_tokens_new, next_tokens_new = cls._create_mask_band(cpu_m_shape, pre_tokens, next_tokens,
                                                                              actualSeqLengths, actualSeqLengthsKV,
                                                                              actualprefixKV, prefix_kvs, batch,
                                                                              numheads, kv_s_list, m_dtype)
            return np.array(cpu_mask, dtype=m_dtype), npu_mask.astype(m_dtype), pre_tokens_new, next_tokens_new


class FiaBoradCastTool():
    @classmethod
    def broadcast_kv_n2_to_n1(cls, num_heads, num_kv_heads, kv_tensor, input_dtype):
        factor = num_heads // num_kv_heads
        kv_shape = kv_tensor.shape
        B = kv_shape[0]
        S = kv_shape[2]
        D = kv_shape[3]
        kv_res = np.zeros([B, num_heads, S, D], dtype=input_dtype)
        for i in range(num_heads):
            j = i // factor
            kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
        return kv_res, kv_res.shape


class FiaLayoutTool():
    @classmethod
    def transpose(cls, data, dims):
        if isinstance(data, np.ndarray):
            return data.transpose(dims)
        elif isinstance(data, torch.Tensor):
            return data.permute(dims).contiguous()
        else:
            raise RuntimeError(f"Unsupported data type {type(data)}")

    @classmethod
    def trans_bnsd_to_target(cls, tensor, shape, target_layout, actual_seq_lens=None):
        B, N, S, D = tensor.shape

        if target_layout == "BSH":
            H = N * D
            return cls.transpose(tensor, [0, 2, 1, 3]).reshape(B, S, H)
        elif target_layout == "BSND":
            return cls.transpose(tensor, [0, 2, 1, 3])
        elif target_layout == "NBSD":
            return cls.transpose(tensor, [1, 0, 2, 3])
        elif target_layout == "TND":
            if actual_seq_lens is None:
                raise ValueError("actual_seq_lens must be provided for TND layout.")

            T = sum(actual_seq_lens)
            output = torch.zeros(size=(T, N, D), dtype=tensor.dtype)
            t_start = 0

            for b_idx in range(B):
                act_s = actual_seq_lens[b_idx]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                # 将批次b_idx的数据填充到output[t_start:t_end]
                output[t_start:t_end, :, :] = cls.transpose(tensor[b_idx, :, :act_s, :], [1, 0, 2])
                t_start += act_s

            return output
        elif target_layout == "NTD":
            if actual_seq_lens is None:
                raise ValueError("actual_seq_lens must be provided for NTD layout.")
            # 先转换为TND，再转置为NTD
            tnd_tensor = cls.trans_bnsd_to_target(tensor, shape, "TND", actual_seq_lens)
            return cls.transpose(tnd_tensor, [1, 0, 2])
        elif target_layout == "BNSD":
            return tensor
        else:
            raise RuntimeError(f"trans_bnsd_to_target does not support target_layout: {target_layout}")

    @classmethod
    def trans_bnsd_to_bsh(cls, tensor, shape):
        return cls.trans_bnsd_to_target(tensor, shape, "BSH")


class FiaOpParam():
    _flag_list_index = {
        "query": 0,
        "key": 1,
        "value": 2,
        "pse_shift": 3,
        "atten_mask": 4,
        "actual_seq_lens_q": 5,
        "actual_seq_lens_kv": 6,
        "dequant_scale1": 7,
        "quant_scale1": 8,
        "dequant_scale2": 9,
        "quant_scale2": 10,
        "quant_offset2": 11,
        "antiquant_scale": 12,
        "antiquant_offset": 13,
        "block_table": 14,
        "q_padding_size": 15,
        "kv_padding_size": 16,
        "k_antiquant_scale": 17,
        "k_antiquant_offset": 18,
        "v_antiquant_scale": 19,
        "v_antiquant_offset": 20,
        "k_shared_prefix": 21,
        "v_shared_prefix": 22,
        "actual_shared_prefix_len": 23,
        "output": 24,
        "input_layout": 25,
        "q_rope": 26,
        "k_rope": 27,
        "key_rope_scale": 28,
        "sinks": 30,
    }

    _param_index = {
        "pse_shift": 1,
        "atten_mask": 2,
        "dequant_scale1": 3,
        "quant_scale1": 4,
        "dequant_scale2": 5,
        "quant_scale2": 6,
        "quant_offset2": 7,
        "antiquant_scale": 8,
        "antiquant_offset": 9,
        "block_table": 10,
        "q_padding_size": 11,
        "kv_padding_size": 12,
        "k_antiquant_scale": 13,
        "k_antiquant_offset": 14,
        "v_antiquant_scale": 15,
        "v_antiquant_offset": 16,
        "k_shared_prefix": 17,
        "v_shared_prefix": 18,
        "k_cache": 19,
        "v_cache": 20,
        "q_rope": 21,
        "k_rope": 22,
        "k_rope_cache": 23,
        "dequant_scale_query": 24,
        "sinks": 26,
    }

    def _get_kv_num(self):
        if self.tnd_flag:
            return 1
        else:
            q_shape = self.data_list[0].shape
            kv_shape = self.data_list[1].shape
            q_b = q_shape[0]
            k_b = kv_shape[0]
            if q_b == k_b:
                return 1
            else:
                return q_b

    @classmethod
    def _get_param_index(cls, data_name, kv_num=1):
        k_start_index = 1
        k_end_index = kv_num
        v_start_index = kv_num + 1
        v_end_index = kv_num + kv_num
        if data_name == 'query':
            return 0
        elif data_name == 'key':
            return k_start_index, k_end_index + 1
        elif data_name == 'value':
            return v_start_index, v_end_index + 1
        else:
            if data_name not in cls._param_index:
                raise ValueError(f"data_name {data_name} is invalid")
            data_index = cls._param_index[data_name]
            return v_end_index + data_index

    @classmethod
    def get_param_index(cls, data_name, kv_num=1):
        index = cls._get_param_index(data_name, kv_num)
        if isinstance(index, tuple):
            return index[0]
        else:
            return index

    def _get_data(self, data_name):
        index = self._get_param_index(data_name, self.kv_num)
        if isinstance(index, tuple):
            return self.data_list[index[0]: index[1]]
        else:
            return self.data_list[index]

    def _get_range(self, range_name):
        index = self._get_param_index(range_name, self.kv_num)
        if isinstance(index, tuple):
            return self.params['range_input'][index[0]]
        else:
            return self.params['range_input'][index]

    def _get_shape(self, shape_name):
        index = self._get_param_index(shape_name, self.kv_num)
        if isinstance(index, tuple):
            return self.params['shape_input'][index[0]: index[1]]
        else:
            return self.params['shape_input'][index]

    def _get_dtype(self, dtype_name):
        index = self._get_param_index(dtype_name, self.kv_num)
        if isinstance(index, tuple):
            return self.params['dtype_input'][index[0]]
        else:
            return self.params['dtype_input'][index]

    def _debug_info(self):
        fia_debug(f"action_type: {self.action_type}")
        fia_debug(f"rope_flag: {self.rope_flag}")
        fia_debug(f"storage_mode: {self.storage_mode}")
        fia_debug(f"kv_num: {self.kv_num}")
        fia_debug(f"pa_flag: {self.pa_flag}")
        fia_debug(f"tnd_flag: {self.tnd_flag}")
        fia_debug(f"num_heads: {self.num_heads}")
        fia_debug(f"num_kv_heads: {self.num_kv_heads}")
        fia_debug(f"input_layout: {self.input_layout}")
        fia_debug(f"q_layout: {self.q_layout}")
        fia_debug(f"out_layout: {self.out_layout}")
        fia_debug(f"kv_layout: {self.kv_layout}")
        fia_debug(f"actual_seq_lens_q_raw: {self.actual_seq_lens_q_raw}")
        fia_debug(f"actual_seq_lens_q: {self.actual_seq_lens_q}")
        fia_debug(f"actual_seq_lens_kv_raw: {self.actual_seq_lens_kv_raw}")
        fia_debug(f"actual_seq_lens_kv: {self.actual_seq_lens_kv}")

        fia_debug(f"is_deepseek_mla: {self.is_deepseek_mla}")
        fia_debug(f"key   shape: {self.key.shape}")
        fia_debug(f"value shape: {self.value.shape}")
        fia_debug(f"key   bnsd shape: {self.key.bnsd_shape}")
        fia_debug(f"value bnsd shape: {self.value.bnsd_shape}")

        fia_debug(f"scale_value: {self.scale_value}")
        fia_debug(f"block_size: {self.block_size}")
        fia_debug(f"inner_precise: {self.inner_precise}")

        fia_debug(f"pre_tokens: {self.pre_tokens}")
        fia_debug(f"next_tokens: {self.next_tokens}")
        fia_debug(f"sparse_mode: {self.sparse_mode}")
        fia_debug(f"softmax_lse_flag: {self.softmax_lse_flag}")
        fia_debug(f"out_quant_flag: {self.out_quant_flag}")
        fia_debug(f"pse_shift_flag: {self.pse_shift_flag}")
        fia_debug(f"atten_mask_flag: {self.atten_mask_flag}")
        fia_debug(f"q_padding_size_flag: {self.q_padding_size_flag}")
        fia_debug(f"kv_padding_size_flag: {self.kv_padding_size_flag}")
        fia_debug(f"shared_prefix_flag: {self.shared_prefix_flag}")
        fia_debug(f"prefix_act_flag: {self.prefix_act_flag}")
        fia_debug(f"sink_flag: {self.sink_flag}")

    def __init__(self, data_list, params):
        self.data_list = data_list
        self.params = params

        self.flag_list = self.str_to_bool_list(self.params['flaglist'])

        self.parse_basic_info()
        self.parse_flag_list()
        self._debug_info()

    def _expand_actual_seq_lens(self, actual_seq_lens):
        if len(actual_seq_lens) == 1 and self.batch > 1:
            return actual_seq_lens * self.batch
        elif len(actual_seq_lens) > self.batch:
            return actual_seq_lens[:self.batch]
        return actual_seq_lens

    @staticmethod
    def _trans_tnd_actseq(actual_seq_lens):
        normal_seq_lens = [actual_seq_lens[0]]
        for i in range(len(actual_seq_lens) - 1):
            seq_len = actual_seq_lens[i + 1] - actual_seq_lens[i]
            if seq_len < 0:
                raise RuntimeError(f"_trans_tnd_actseq: actual_seq_lens[{i}] = {seq_len}, it should >= 0")
            normal_seq_lens.append(seq_len)
        return normal_seq_lens

    def _get_actual_seq_lens_q(self):
        actual_seq_lens_q = self._expand_actual_seq_lens(copy.deepcopy(self.params['actualseqlengths']))
        if self.tnd_flag:
            # 将tnd格式下的act seq转成普通的act seq
            fia_debug("_trans_tnd_actseq actual_seq_lens_q")
            actual_seq_lens_q = self._trans_tnd_actseq(actual_seq_lens_q)
        return actual_seq_lens_q

    def _get_actual_seq_lens_kv(self):
        actual_seq_lens_kv = self._expand_actual_seq_lens(copy.deepcopy(self.params['actualseqlengthskv']))
        if self.tnd_flag and (not self.pa_flag):
            fia_debug("_trans_tnd_actseq actual_seq_lens_kv")
            # 将tnd格式下的act seq转成普通的act seq
            actual_seq_lens_kv = self._trans_tnd_actseq(actual_seq_lens_kv)
        return actual_seq_lens_kv

    def parse_basic_info(self):
        self.tnd_flag = self._get_tnd_flag()
        self.action_type = self._get_action_type()
        self.input_layout = self._get_input_layout()
        self.kv_num = self._get_kv_num()
        self.num_heads = self._get_num_heads()
        self.num_kv_heads = self._get_num_kv_heads()
        self.rope_flag = self._get_flag("q_rope")
        self.storage_mode = self._get_storage_mode()
        self.pa_flag = (self.storage_mode == StorageMode.PAGE_ATTENTION)

        self._parse_layout()
        self.actual_seq_lens_q_raw = self._get_actual_seq_lens_q_raw()
        self.actual_seq_lens_kv_raw = self._get_actual_seq_lens_kv_raw()
        if not self.tnd_flag:
            self.batch = FiaTensor(self._get_data("query"), self._get_shape("query"),
                                   self._get_dtype("query"), self.q_layout, name="dummy").B
        else:
            self.batch = len(self.actual_seq_lens_q_raw)
        self.actual_seq_lens_q = self._get_actual_seq_lens_q()
        self.actual_seq_lens_kv = self._get_actual_seq_lens_kv()

        self.scale_value = self._get_scale_value()
        self.block_size = self._get_block_size()
        self.inner_precise = self._get_inner_precise()

        self.antiquant_mode = self._get_antiquant_mode()
        self.k_antiquant_mode = self._get_k_antiquant_mode()
        self.v_antiquant_mode = self._get_v_antiquant_mode()

        self.pre_tokens = self._get_pretokens()
        self.next_tokens = self._get_nexttokens()
        self.sparse_mode = self._get_sparse_mode()
        self.softmax_lse_flag = self._get_softmax_lse_flag()

        self._parse_input_tensor()
        self._parse_output_tensor()
        self._parse_optional_tensor()
        self._parse_quant_info()

    def _get_flag(self, flag_name):
        if flag_name not in self._flag_list_index:
            raise ValueError(f"param flag_name {flag_name} should in _flag_list_index")
        index = self._flag_list_index[flag_name]
        return self.flag_list[index]

    def _get_action_type(self):
        return self.params["action_type"]

    def _get_input_layout(self):
        return self.params['inputlayout']

    def _get_actual_seq_lens_q_raw(self):
        return self.params['actualseqlengths']

    def _get_actual_seq_lens_kv_raw(self):
        return self.params['actualseqlengthskv']

    def _get_num_heads(self):
        return self.params['numheads']

    def _get_scale_value(self):
        return self.params['scalevalue']

    def _get_block_size(self):
        return self.params['blocksize']

    def _get_inner_precise(self):
        return self.params['innerprecise']

    def _get_antiquant_mode(self):
        return str(self.params['antiquant_mode'])

    def _get_k_antiquant_mode(self):
        return str(self.params['k_antiquant_mode'])

    def _get_v_antiquant_mode(self):
        return str(self.params['v_antiquant_mode'])

    def _get_pretokens(self):
        return self.params['pretokens']

    def _get_nexttokens(self):
        return self.params['nexttokens']

    def _get_sparse_mode(self):
        return self.params['sparsemode']

    def _get_softmax_lse_flag(self):
        return self.params['softmax_lse_flag']

    def _get_num_kv_heads(self):
        # 当numKeyValueHeads传入0时，处理为与numHeads相等
        return self.params['numkeyvalueheads'] if self.params['numkeyvalueheads'] != 0 else self.params['numheads']

    def _get_tnd_flag(self):
        return (self.params['inputlayout'] in ["TND", "TND_NTD", "NTD", "NTD_TND"])

    def _get_storage_mode(self):
        if self._get_flag("block_table"):
            return StorageMode.PAGE_ATTENTION

        if self.kv_num > 1:
            return StorageMode.TENSOR_LIST

        return StorageMode.CONTIGUOES

    def _parse_layout(self):
        self.q_layout = self.params['inputlayout'].split("_")[0]
        self.out_layout = self.params['inputlayout'].split("_")[-1]
        self.kv_layout = "BNSD" if (
                self.tnd_flag and (self.storage_mode == StorageMode.PAGE_ATTENTION)) else self.q_layout
        self.lse_layout = "TND" if self.tnd_flag else "BNSD"

    def _parse_input_tensor(self):
        self.query = FiaTensor(self._get_data("query"), self._get_shape("query"),
                               self._get_dtype("query"), self.q_layout, self.num_heads, self.actual_seq_lens_q,
                               name="query")
        self.key = FiaTensorList(self._get_data("key"), self._get_shape("key"),
                                 self._get_dtype("key"), self.kv_layout, self.num_kv_heads, self.actual_seq_lens_kv,
                                 name="key")
        self.value = FiaTensorList(self._get_data("value"), self._get_shape("value"),
                                   self._get_dtype("value"), self.kv_layout, self.num_kv_heads, self.actual_seq_lens_kv,
                                   name="value")
        self.is_deepseek_mla = (self.query.D == 512) and (self.value.D == 512) and self.rope_flag
        if self.is_deepseek_mla:
            self.value = FiaTensorList(self._get_data("key"), self._get_shape("key"),
                                       self._get_dtype("key"), self.kv_layout, self.num_kv_heads,
                                       self.actual_seq_lens_kv, name="value")
        else:
            self.value = FiaTensorList(self._get_data("value"), self._get_shape("value"),
                                       self._get_dtype("value"), self.kv_layout, self.num_kv_heads,
                                       self.actual_seq_lens_kv, name="value")

        self.q_s = self.query.S
        self.kv_s = self.key.S
        self.q_d = self.query.D
        self.v_d = self.value.D
        self.kv_s_list = self.key.kv_s_list
        self.q_t = self.query.T if self.query.is_tnd_like_layout() else 0

        kv_num = self.batch if (self.storage_mode == StorageMode.TENSOR_LIST) else 1
        if kv_num != self.kv_num:
            raise RuntimeError(
                f"kv_num({kv_num}) calculate by batch is not equal to kv_num({self.kv_num}) calculate by data_list num")
        if self.storage_mode == StorageMode.CONTIGUOES:
            if (len(self.key) != 1 or len(self.value) != 1):
                raise RuntimeError(
                    f"In CONTIGUOES situation, len(key) = {len(self.key)} \
                    and len(value) = {len(self.value)} should be equal to 1")
        if self.storage_mode == StorageMode.TENSOR_LIST:
            if self.key[0].B != 1 or self.value[0].B != 1:
                raise RuntimeError(
                    f"In TENSOR_LIST situation, key[0].B = {self.key[0].B} \
                    and value[0].B = {self.value[0].B} should be 1")
        self._parse_page_attention_input()

    def _get_kv_cache_layout(self, k_cache_shape):
        kv_cache_layout = 'BSH'
        if len(k_cache_shape) == 1:
            fia_info("kv_cache shape is 1")
        elif len(k_cache_shape) == 3:
            kv_cache_layout = "BSH"
        elif len(k_cache_shape) == 4:
            kv_cache_layout = "BNSD"
        elif len(k_cache_shape) == 5:
            kv_cache_layout = "BNDSD0"
        else:
            raise ValueError(f"len(k_cache_shape) should be in 3/4/5, got {len(k_cache_shape)}")
        return kv_cache_layout

    def _parse_page_attention_input(self):
        self.block_table = FiaTensor(self._get_data("block_table"),
                                     self._get_shape("block_table"), self._get_dtype("block_table"), "ND",
                                     name="block_table")
        k_cache_shape = self._get_shape("k_cache")
        v_cache_shape = self._get_shape("v_cache")
        if len(k_cache_shape) != len(v_cache_shape):
            raise RuntimeError(f"len(k_cache_shape) = {len(k_cache_shape)} should be \
                equal to len(v_cahce_shape) = {len(v_cache_shape)}")
        self.kv_cache_layout = self._get_kv_cache_layout(k_cache_shape)
        self.k_cache = FiaTensor(self._get_data("k_cache"),
                                 k_cache_shape, self._get_dtype("k_cache"), self.kv_cache_layout, name="k_cache")
        self.v_cache = FiaTensor(self._get_data("v_cache"),
                                 v_cache_shape, self._get_dtype("v_cache"), self.kv_cache_layout, name="v_cache")
        self.k_rope_cache = FiaTensor(
            self._get_data("k_rope_cache"), self._get_shape("k_rope_cache"),
            self._get_dtype("k_rope_cache"), self.kv_cache_layout, name="k_rope_cache")

    def _get_output_shape(self):
        if self.out_layout == "BSND":
            return [self.batch, self.q_s, self.num_heads, self.v_d]
        elif self.out_layout == "BNSD":
            return [self.batch, self.num_heads, self.q_s, self.v_d]
        elif self.out_layout == "NBSD":
            return [self.num_heads, self.batch, self.q_s, self.v_d]
        elif self.out_layout == "BSH":
            return [self.batch, self.q_s, self.num_heads * self.v_d]
        elif self.out_layout == "TND":
            return [self.q_t, self.num_heads, self.v_d]
        elif self.out_layout == "NTD":
            return [self.num_heads, self.q_t, self.v_d]
        else:
            raise ValueError(f"unsupported out_layout {self.out_layout}")

    def _get_lse_shape(self):
        if self.lse_layout == "TND":
            return [self.q_t, self.num_heads, 1]
        elif self.lse_layout == "BNSD":
            return [self.batch, self.num_heads, self.q_s, 1]
        else:
            raise ValueError(f"unsupported lse_layout {self.lse_layout}")

    def _parse_output_tensor(self):
        output_shape = self._get_output_shape()
        self.output = FiaTensor(np.zeros(output_shape), output_shape, self.params['dtype_output'][0], self.out_layout,
                                self.num_heads, self.actual_seq_lens_q, name="output")
        lse_shape = self._get_lse_shape()
        self.lse = FiaTensor(np.full(lse_shape, np.inf), lse_shape, "fp32", self.lse_layout,
                             self.num_heads, self.actual_seq_lens_q, name="lse")

    def _parse_optional_tensor(self):
        self.q_rope = FiaTensor(
            self._get_data("q_rope"), self._get_shape("q_rope"), self._get_dtype("q_rope"),
            self.q_layout, self.num_heads, self.actual_seq_lens_q, name="q_rope")
        self.k_rope = FiaTensor(
            self._get_data("k_rope"), self._get_shape("k_rope"), self._get_dtype("k_rope"),
            self.kv_layout, self.num_kv_heads, self.actual_seq_lens_kv, name="k_rope")
        self.pse_shift = FiaTensor(
            self._get_data("pse_shift"), self._get_shape("pse_shift"),
            self._get_dtype("pse_shift"), "ND", name="pse_shift")
        self.atten_mask = FiaTensor(
            self._get_data("atten_mask"), self._get_shape("atten_mask"),
            self._get_dtype("atten_mask"), "ND", name="atten_mask")
        self.k_shared_prefix = FiaTensor(
            self._get_data("k_shared_prefix"), self._get_shape("k_shared_prefix"),
            self._get_dtype("k_shared_prefix"), self.kv_layout, self.num_kv_heads, name="k_shared_prefix")
        self.v_shared_prefix = FiaTensor(
            self._get_data("v_shared_prefix"), self._get_shape("v_shared_prefix"),
            self._get_dtype("v_shared_prefix"), self.kv_layout, self.num_kv_heads, name="v_shared_prefix")
        self.sinks = FiaTensor(
            self._get_data("sinks"), self._get_shape("sinks"),
            self._get_dtype("sinks"), "ND", name = "sinks")

    def _get_quant_scale_offset_layout(self, shape):
        dim2layout = {1: 'H', 2: 'ND'}
        return dim2layout.get(len(shape), '1N1D')

    def _parse_quant_info(self):
        self.quant_scale2 = FiaTensor(
            self._get_data("quant_scale2"), self._get_shape("quant_scale2"),
            self._get_dtype("quant_scale2"),
            self._get_quant_scale_offset_layout(self._get_shape("quant_scale2")),
            self.num_heads, name="quant_scale2")
        self.quant_offset2 = FiaTensor(
            self._get_data("quant_offset2"), self._get_shape("quant_offset2"),
            self._get_dtype("quant_offset2"),
            self._get_quant_scale_offset_layout(self._get_shape("quant_offset2")),
            self.num_heads, name="quant_offset2")

        self.out_quant_flag = self._get_flag("quant_scale2")
        self.out_quant_pc_flag = False
        if self.out_quant_flag:
            if not self._get_flag("quant_offset2"):
                # 当不传入quantOffset2时，也需要生成一个和scale一样大小的全0 Offset，用于后续计算
                quant_offset2_shape = self._get_shape("quant_scale2")
                quant_offset2_data = np.zeros(quant_offset2_shape, np.float32)
                self.quant_offset2 = FiaTensor(quant_offset2_data, quant_offset2_shape, "fp32",
                                               self._get_quant_scale_offset_layout(quant_offset2_shape),
                                               self.num_heads, name="quant_offset2")
            if self._get_shape("quant_scale2") != [1]:
                self.out_quant_pc_flag = True

    def _get_normal_flag(self):
        if not self.flag_list[0] or not self.flag_list[1] or not self.flag_list[2] or not self.flag_list[24]:
            fia_warn("q/k/v/out flag is false, return zero output")
            return False
        if self.query.empty():
            fia_warn("query is empty, return zero output")
            return False
        if self.key.empty():
            fia_warn("key is empty, return zero output")
            return False
        if self.value.empty():
            fia_warn("value is empty, return zero output")
            return False
        if not ((self.flag_list[21] and self.flag_list[22]) or (not self.flag_list[21] and not self.flag_list[22])):
            fia_warn("prefix未成对出现, 返回全0输出")
            return False
        if self.block_table.empty():
            fia_warn("block_table is empty, return zero output")
            return False
        return True

    def _get_prefix_act_lens(self):
        prefix_act_lens = 0
        if self.shared_prefix_flag:
            prefix_act_lens = self.k_shared_prefix.bnsd_shape[2]
            if self.prefix_act_flag:
                prefix_act_lens = self.params['prefix_act_lens'][0]
        return prefix_act_lens

    def _parse_feature_flag(self):
        self.pse_shift_flag = self._get_flag("pse_shift")
        self.atten_mask_flag = self._get_flag("atten_mask")
        self.actual_seq_lens_q_flag = self._get_flag("actual_seq_lens_q")
        self.actual_seq_lens_kv_flag = self._get_flag("actual_seq_lens_kv")
        self.q_padding_size_flag = self._get_flag("q_padding_size")
        if self.q_padding_size_flag:
            self.q_padding_size = max(self._get_range("q_padding_size")[0], 0)
        self.kv_padding_size_flag = self._get_flag("kv_padding_size")
        if self.kv_padding_size_flag:
            self.kv_padding_size = max(self._get_range("kv_padding_size")[0], 0)
        self.shared_prefix_flag = self._get_flag("k_shared_prefix") or self._get_flag("v_shared_prefix")
        self.prefix_act_flag = self._get_flag("actual_shared_prefix_len")
        self.prefix_act_lens = self._get_prefix_act_lens()
        self.prefix_kvs = 0
        self.sink_flag = self._get_flag("sinks")

    def parse_flag_list(self):
        self._parse_feature_flag()
        self.normal_flag = self._get_normal_flag()  # TODO: 处理normal_flag，返回全0

    def str_to_bool_list(self, lst):
        return [True if l else False for l in lst]


class FiaOpPreprocess():
    def __init__(self, data_list, params, op_params):
        self.params = params
        self.data_list = data_list
        self.op_params = op_params
        self.query = op_params.query
        self.key = op_params.key
        self.value = op_params.value
        self.q_rope = op_params.q_rope
        self.k_rope = op_params.k_rope
        self.quant_scale2 = op_params.quant_scale2
        self.quant_offset2 = op_params.quant_offset2

    def preprocess(self):
        if self.op_params.rope_flag:
            self.concat_rope_tensor()

        self.preprocess_kv()
        self.preprocess_shared_prefix()
        self.preprocess_pse_shift()
        self.preprocess_atten_mask()
        self.preprocess_block_table()
        self.preprocess_kv_cache()
        self.preprocess_post_quant()

    def preprocess_post_quant(self):
        # 如果是perchannel模式，要将后量化参数统一转换成1n1d格式
        if self.op_params.out_quant_pc_flag:
            fia_debug_func_begin("begin FiaOpPreprocess.preprocess_post_quant")
            self.quant_scale2.trans_to_1n1d()
            self.quant_offset2.trans_to_1n1d()

    def concat_rope_tensor(self):
        # 非全量化场景1.将Q与QROPE拼接2.将K与KROPE拼接3.伪量化场景，k_antiscale与k_rope_antiscale拼接
        self.op_params.query = concat_tensor(self.query, self.q_rope)

        if self.op_params.kv_num == 1:
            self.op_params.key = concat_tensor_list(self.key, self.k_rope)
        else:
            raise ValueError("k tensor 长度不为1, deepseek预处理异常，输出空tensor！")

    def preprocess_kv(self):
        # >> kv预处理：1、将kv list 转换为bnsd 2、GQA场景，将kvn扩展为qn
        return

    def preprocess_block_table(self):
        # block_table: [B, ceil(max_s/block_size)]
        # cache: BnBsH/BnDBsD/NZ
        # 1、生成随机的block_table，并覆写原有bin文件
        if not self.op_params.pa_flag:
            return

        if not need_gen_input(self.op_params.action_type):
            return

        fia_debug_func_begin("begin FiaOpPreprocess.preprocess_block_table")
        # 生成blocktable
        block_num = self.op_params.k_cache.shape[0]
        block_size = self.op_params.block_size
        block_table_shape = self.op_params.block_table.shape

        block_num_each_batch = []
        block_num_expect_min = 0
        for actual_seq in self.op_params.actual_seq_lens_kv:
            block_num_cur_batch = math.ceil(actual_seq / block_size)
            block_num_each_batch.append(block_num_cur_batch)
            block_num_expect_min += block_num_cur_batch

        if block_num_expect_min > block_num:
            raise RuntimeError(
                f"[ERROR]Wrong input k_cache_shape: get block_num = {block_num}, but expect block_num > {block_num_expect_min}")

        block_idx_list = np.arange(0, block_num, 1)
        block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
        block_table = [-1] * block_table_shape[1]
        block_table = np.tile(block_table, (block_table_shape[0], 1)).astype(np.int32)

        block_idx = 0
        for batch_idx, block_num_cur_batch in enumerate(block_num_each_batch):
            for block_num_cur_batch_idx in range(block_num_cur_batch):
                block_table[batch_idx][block_num_cur_batch_idx] = (block_idx_list[block_idx])
                block_idx += 1
        self.op_params.block_table.data = block_table

        ids = FiaOpParam.get_param_index("block_table")
        tools.modify_alcnn_input_file(ids=ids, origin_index=[ids], type='tensor', mode='rewrite',
                                      tensors=torch.tensor(block_table, dtype=torch.int32),
                                      params=self.params)

    def _generate_cache(self, cache_shape, tensor_bnsd, shape_bnsd, src_dtype, dst_dtype):
        block_table_dim1 = self.op_params.block_table.shape[1]
        block_size = self.op_params.block_size
        block_table = self.op_params.block_table.data
        max_s_batch = block_table_dim1 * block_size
        B, N, S, D = shape_bnsd
        cache = np.zeros(cache_shape)
        if len(cache_shape) == 3:  # BSH
            # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
            tensor_bsh_raw = FiaLayoutTool.trans_bnsd_to_bsh(tensor_bnsd, shape_bnsd)
            H = N * D
            if src_dtype == "int32":
                H = int(H / 8)
            tensor_bsh = np.zeros((B, max_s_batch, H))
            tensor_bsh[:, :tensor_bsh_raw.shape[1], :] = tensor_bsh_raw[:, :, :]
            for batch_idx in range(B):
                for block_idx, cache_block_id in enumerate(block_table[batch_idx]):
                    block_offset = block_idx * block_size
                    if cache_block_id == -1:
                        continue
                    else:
                        cache[cache_block_id, 0:block_size, :] = tensor_bsh[batch_idx,
                                                                 block_offset:(block_offset + block_size), :]
        elif len(cache_shape) == 4:  # BNSD
            k_tensor_bnsd = np.zeros((B, N, max_s_batch, D))
            k_tensor_bnsd[:, :, :S, :] = tensor_bnsd[:, :, :, :]

            for batch_idx in range(B):
                for block_idx, cache_block_id in enumerate(block_table[batch_idx]):
                    block_offset = block_idx * block_size
                    if cache_block_id == -1:
                        continue
                    else:
                        cache[cache_block_id, :, 0:block_size, :] = \
                            k_tensor_bnsd[batch_idx, :, block_offset:(block_offset + block_size), :]
        elif len(cache_shape) == 5:  # NZ
            k_cache_tensor_bnbd = self._generate_cache([cache_shape[0], N, block_size, D],
                                                       tensor_bnsd, shape_bnsd, src_dtype, dst_dtype)
            D0 = 32 if src_dtype == "int8" else 16
            cache = k_cache_tensor_bnbd.reshape(k_cache_tensor_bnbd.shape[0], k_cache_tensor_bnbd.shape[1],
                                                k_cache_tensor_bnbd.shape[2],
                                                k_cache_tensor_bnbd.shape[3] // D0, D0).transpose(0, 1, 3, 2, 4)
        else:
            raise ValueError(f"cache shape dim should be 3/4/5, but got {len(cache_shape)}")
        return cache.astype(dst_dtype)

    def _preprocess_kv_cache_rope(self):
        fia_debug_func_begin("begin FiaOpPreprocess._preprocess_kv_cache_rope")
        k_cache = self._generate_cache(self.op_params.k_cache.shape,
                                       self.key.bnsd_data,
                                       self.key.bnsd_shape,
                                       self.key.dtype,
                                       self.key.np_dtype)

        v_cache = None
        if not self.op_params.is_deepseek_mla:
            v_cache = self._generate_cache(self.op_params.v_cache.shape,
                                           self.value.bnsd_data,
                                           self.value.bnsd_shape,
                                           self.value.dtype,
                                           self.value.np_dtype)
        else:
            v_cache = k_cache
        k_rope_cache = self._generate_cache(self.op_params.k_rope_cache.shape,
                                            self.k_rope.bnsd_data,
                                            self.k_rope.bnsd_shape,
                                            self.key.dtype,
                                            self.key.np_dtype)
        # return torch.zeros(out_shape)
        # 将kv cache 生成新的bin文件
        k_cache_index = FiaOpParam.get_param_index("k_cache")
        v_cache_index = FiaOpParam.get_param_index("v_cache")
        k_rope_cache_index = FiaOpParam.get_param_index("k_rope_cache")
        tools.modify_alcnn_input_file(ids=k_cache_index,
                                      origin_index=[k_cache_index],
                                      type='tensor_list',
                                      mode='rewrite',
                                      tensors=[k_cache],
                                      params=self.params,
                                      data_dtype=self.params['dtype_input'][FiaOpParam.get_param_index("key")])
        tools.modify_alcnn_input_file(ids=v_cache_index,
                                      origin_index=[v_cache_index],
                                      type='tensor_list',
                                      mode='rewrite',
                                      tensors=[v_cache],
                                      params=self.params,
                                      data_dtype=self.params['dtype_input'][FiaOpParam.get_param_index("value")])
        tools.modify_alcnn_input_file(ids=k_rope_cache_index,
                                      origin_index=[k_rope_cache_index],
                                      type='tensor',
                                      mode='rewrite',
                                      tensors=k_rope_cache,
                                      params=self.params,
                                      data_dtype=self.params['dtype_input'][FiaOpParam.get_param_index("key")])

    def _preprocess_kv_cache_no_rope(self):
        k_cache = self._generate_cache(self.op_params.k_cache.shape,
                                       self.key.bnsd_data_list[0],
                                       self.key.bnsd_shape_list[0],
                                       self.key.dtype,
                                       self.key.np_dtype)
        v_cache = self._generate_cache(self.op_params.v_cache.shape,
                                       self.value.bnsd_data_list[0],
                                       self.value.bnsd_shape_list[0],
                                       self.value.dtype,
                                       self.value.np_dtype)

        # 将kv cache 生成新的bin文件
        k_cache_index = FiaOpParam.get_param_index("k_cache")
        v_cache_index = FiaOpParam.get_param_index("v_cache")
        tools.modify_alcnn_input_file(ids=k_cache_index, origin_index=[k_cache_index], type='tensor_list',
                                      mode='rewrite',
                                      tensors=[k_cache],
                                      params=self.params,
                                      data_dtype=self.params['dtype_input'][1])
        tools.modify_alcnn_input_file(ids=v_cache_index, origin_index=[v_cache_index], type='tensor_list',
                                      mode='rewrite',
                                      tensors=[v_cache],
                                      params=self.params,
                                      data_dtype=self.params['dtype_input'][2])

    def preprocess_kv_cache(self):
        # 2、将kv shape 统一转换成bsh
        # 3、生成kv cache
        # 4、将kv cache dump成新的bin文件，供aclnn接口调用
        if not self.op_params.pa_flag:
            return

        if not need_gen_input(self.op_params.action_type):
            return

        fia_info(f"[PageAtten]Input Kdtype:{self.params['dtype_input'][1]} Vdtype:{self.params['dtype_input'][2]}")
        if not self.op_params.rope_flag:
            self._preprocess_kv_cache_no_rope()
        else:
            self._preprocess_kv_cache_rope()

    def preprocess_pse_shift(self):
        # >> pse 处理
        if not self.op_params.pse_shift_flag:
            return

        if self.op_params.action_type in ["bm_output_gold", "bm_output"]:
            return

        fia_debug_func_begin("begin FiaOpPreprocess.preprocess_pse_shift")
        maya = get_slopes(self.op_params.pse_shift.shape[1])
        maya = maya.numpy()

        pse_shift = np.zeros(self.op_params.pse_shift.shape)
        for n in range(self.op_params.pse_shift.shape[1]):
            alibi_biases = np.zeros([1, self.op_params.pse_shift.shape[-1]])
            for x in range(0, self.op_params.pse_shift.shape[-1]):
                alibi_biases[0, x] = -1 * x
            pse_shift[:, n:n + 1, :, :] = alibi_biases * maya[n]
        pse_shift = pse_shift.astype(np.float32)
        # 覆写
        if self.op_params.pse_shift.dtype == "float16":
            pse_shift = pse_shift.astype(np.float16)
            pse_shift[pse_shift == -math.inf] = -65504
            self.op_params.pse_shift.data = pse_shift
            p_tensor = torch.tensor(pse_shift, dtype=torch.float16)
        elif self.op_params.pse_shift.dtype == "bfloat16":
            pse_shift = pse_shift.astype(tf.bfloat16.as_numpy_dtype)
            pse_shift[pse_shift == -math.inf] = -65504
            self.op_params.pse_shift.data = pse_shift
            p_tensor = torch.tensor(pse_shift.astype(np.float32), dtype=torch.bfloat16)
        else:
            raise ValueError(f"wrong p_tensor dtype!")

        ids = FiaOpParam.get_param_index("pse_shift")
        tools.modify_alcnn_input_file(ids=ids, origin_index=[ids], type='tensor', mode='rewrite', tensors=p_tensor,
                                      params=self.params)

    def preprocess_atten_mask(self):
        # >> m预处理：1、将m扩展为BN1S  2、padding场景下，偏移部分设置为1 3、针对FP16格式，将tensor转成0/1
        sparse_mode = self.op_params.sparse_mode
        q_shape_bnsd = self.op_params.query.bnsd_shape

        randoms = 0
        mrandom_type = "NORMAL"
        if 'mrandomtype' in self.params:
            mrandom_type = self.params['mrandomtype']
            if mrandom_type == 'ones':
                randoms = int(self.params['mrandom'])

        if (not self.op_params.atten_mask_flag) or self.op_params.atten_mask.empty():
            self.op_params.pre_tokens = 214748647
            self.op_params.next_tokens = 214748647
        else:
            fia_debug_func_begin("begin FiaOpPreprocess.preprocess_atten_mask")
            batch = q_shape_bnsd[0]
            num_heads = q_shape_bnsd[1]
            npu_m_shape_s = self.op_params.atten_mask.shape
            self.op_params.is_mask_bs = False
            if sparse_mode == 0 or sparse_mode == 1:
                self.op_params.is_mask_bs = (self.op_params.atten_mask.shape[0] == self.op_params.batch) and (
                        len(self.op_params.atten_mask.shape) == 2)
                batch, num_heads, ns1, ns2 = get_attention_mask_batch_num(self.op_params.atten_mask.shape,
                                                                          self.op_params.is_mask_bs)  # 获取输入attentionmask的batch 和numhead
                npu_m_shape_s = [ns1, ns2]

            q_s = self.op_params.query.bnsd_shape[2]
            kvs = self.op_params.kv_s  # TODO: 考虑prefix
            if self.op_params.shared_prefix_flag:
                kvs += self.op_params.prefix_act_lens
            cpu_m_shape = [q_s, kvs]  # cpu
            cpu_m_tensor, npu_m_tensor, self.op_params.pre_tokens, self.op_params.next_tokens = \
                MaskGenerator._create_random_mask_by_spars(cpu_m_shape, npu_m_shape_s,
                                                           self.op_params.atten_mask.dtype, self.op_params.pre_tokens,
                                                           self.op_params.next_tokens,
                                                           self.op_params.actual_seq_lens_q,
                                                           self.op_params.actual_seq_lens_kv,
                                                           self.op_params.prefix_act_lens,
                                                           self.op_params.prefix_act_lens,
                                                           self.op_params.kv_s_list,
                                                           batch,
                                                           num_heads, sparse_mode,
                                                           random_ones=randoms)
            if mrandom_type == 'invalid' or mrandom_type == 'invaild':
                randoms = int(self.params['mrandom'])
                cpu_m_tensor[..., :randoms] = 1
                npu_m_tensor[..., :randoms] = 1
            if self.op_params.is_mask_bs:
                npu_m_tensor = npu_m_tensor.reshape(self.op_params.atten_mask.shape)

            atten_mask_index = FiaOpParam.get_param_index("atten_mask")
            # tools.modify_alcnn_input_file(ids=atten_mask_index, origin_index=[atten_mask_index],
            #                               type='tensor', mode='rewrite', tensors=torch.from_numpy(npu_m_tensor),
            #                               params=self.params)

            if (sparse_mode == 0 or sparse_mode == 1) and (not self.op_params.is_mask_bs):
                m_tensor = _np_broadcast_mask_n(cpu_m_tensor, self.op_params.atten_mask.shape, cpu_m_shape,
                                                self.op_params.num_heads, q_shape_bnsd[0])
            else:
                m_tensor = cpu_m_tensor
            self.op_params.atten_mask.data = np.array(m_tensor)

    def preprocess_shared_prefix(self):
        # psefix 预处理：1.转成1nsd 2.GQA场景扩展N; 3.按act_prefix裁剪 4.获取perfix_act_lens
        if not self.op_params.shared_prefix_flag:
            return

        if self.op_params.prefix_act_flag:
            self.op_params.k_shared_prefix.data = \
                self.op_params.k_shared_prefix.bnsd_data[:, :, :self.op_params.prefix_act_lens, :]
            self.op_params.v_shared_prefix.data = \
                self.op_params.v_shared_prefix.bnsd_data[:, :, :self.op_params.prefix_act_lens, :]
        else:
            self.op_params.k_shared_prefix.data = \
                self.op_params.k_shared_prefix.bnsd_data
            self.op_params.v_shared_prefix.data = \
                self.op_params.v_shared_prefix.bnsd_data
        fia_info(f"prefix_act_lens:{str(self.op_params.prefix_act_lens)}")


class FiaOpPreprocessTorch(FiaOpPreprocess):
    def __init__(self, data_list, params, op_params, device=None):
        super().__init__(data_list, params, op_params)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    def preprocess(self):
        if self.op_params.rope_flag:
            self.concat_rope_tensor()
        else:
            raise ValueError(f"G only supported rope")
        self.preprocess_block_table()

    def preprocess_block_table(self):
        import triton
        import torch.nn.functional as F

        key = self.op_params.key
        fia_debug("key_shape: key.shape")
        d = key.D
        kv_head_nums = key.N
        # pa场景生成blocktable
        block_size = 64
        batch = self.op_params.batch
        self.op_params.cache_seqlens = torch.tensor(
            self.op_params.actual_seq_lens_kv_raw, dtype=torch.int32).to(self.device)
        max_seqlen = self.op_params.cache_seqlens.max().item()
        max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
        padding_length = max_seqlen_pad - max_seqlen
        # 裁剪ks
        k_new_tensor = key.bsnd_data[:, :key.S, :, :]
        print(f"k_cache_before_padding:{k_new_tensor.shape}")
        k_new_tensor = F.pad(k_new_tensor, (0, 0, 0, 0, 0, padding_length), "constant", 0)
        print(f"k_cache_after_padding:{k_new_tensor.shape}")
        block_table = torch.arange(
            batch * max_seqlen_pad // block_size, dtype=torch.int32
        ).view(batch, max_seqlen_pad // block_size)
        fia_debug("block_table shape: block_table.shape")
        self.op_params.block_table = block_table.to(self.device)
        k_new_tensor = k_new_tensor.transpose(1, 2)
        blocked_k = k_new_tensor.reshape(block_table.numel(), block_size, kv_head_nums, d)
        fia_debug("k_cache shape: blocked_k.shape")
        for i in range(batch):
            if blocked_k.dtype == torch.int8:
                blocked_k.view(batch, max_seqlen_pad, kv_head_nums, d)[i, self.op_params.cache_seqlens[i].item():] = (0)
            else:
                blocked_k.view(batch, max_seqlen_pad, kv_head_nums, d)[i, self.op_params.cache_seqlens[i].item():] = (
                    float("nan")
                )
        self.op_params.blocked_k = blocked_k.to(self.device)


class FiaOpPreprocessNumpy(FiaOpPreprocess):
    pass


class FiaOpForward():
    def __init__(self, data_list, params, mode='numpy', device=None):

        self.params = params
        self.data_list = data_list
        self.op_params = FiaOpParam(data_list, params)
        self.mode = mode
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        if mode == 'numpy':
            self.fia_op_preprocess = FiaOpPreprocessNumpy(data_list, params, self.op_params)
        else:
            self.fia_op_preprocess = FiaOpPreprocessTorch(data_list, params, self.op_params, device)
        self.act_seq_kv = 0
        self.act_seq_q = 0
        self.lse_default_value = np.inf
        self.n_factor = self.op_params.num_heads // self.op_params.num_kv_heads

    def _post_quant(self, y, n1):
        if self.op_params.out_quant_flag:
            if self.op_params.out_quant_pc_flag:
                fia_debug_func_begin("begin FiaOpForward._post_quant quant_pc")
                y = quant_pc(y, self.op_params.quant_scale2.data, self.op_params.quant_offset2.data, n1)
            else:
                fia_debug_func_begin("begin FiaOpForward._post_quant quant")
                y = quant(y, self.op_params.quant_scale2.data[0], self.op_params.quant_offset2.data[0])
        return y

    def _calculate_q_s_begin_end(self):
        q_s_begin = 0
        q_s_end = self.query.S
        if self.op_params.q_padding_size_flag:
            q_s_begin = int(self.query.S - self.act_seq_q - self.op_params.q_padding_size)
            q_s_end = int(self.query.S - self.op_params.q_padding_size)
            fia_debug(f"query_left padding--- s_begin:{q_s_begin}, s_end:{q_s_end}")
        else:
            if self.act_seq_q is not None:
                q_s_end = self.act_seq_q
        self.q_s_begin = q_s_begin
        self.q_s_end = q_s_end

    def _calculate_kv_s_begin_end(self, bidx):
        kv_s_begin = 0

        if self.op_params.storage_mode == StorageMode.TENSOR_LIST:
            kv_s_end = self.key._bnsd_shape_list[bidx][2]
        else:
            kv_s_end = self.key.S
        if self.op_params.kv_padding_size_flag:
            kv_s_begin = int(kv_s_end - self.act_seq_kv - self.op_params.kv_padding_size)
            kv_s_end = int(kv_s_end - self.op_params.kv_padding_size)
            fia_debug(f"kv_left padding--- s_begin:{kv_s_begin}, s_end:{kv_s_end}")
        else:
            if self.op_params.actual_seq_lens_kv_flag:
                if self.act_seq_kv is not None:
                    kv_s_end = self.act_seq_kv

        self.kv_s_begin = kv_s_begin
        self.kv_s_end = kv_s_end

    def _calculate_s_begin_end(self, bidx):
        self._calculate_q_s_begin_end()
        self._calculate_kv_s_begin_end(bidx)

    def _get_matmul_dtype(self):
        matmul_dtype = np.float32
        return matmul_dtype

    @timeit_decorator
    def _calculte_bmm1(self, q, k, matmul_dtype):
        qkBmmRes = np.matmul(q, k.transpose(0, 1, 3, 2), dtype=matmul_dtype)
        fia_debug_data(f"mm1 output", qkBmmRes)
        return qkBmmRes

    @timeit_decorator
    def _calculte_scale(self, qkBmmRes):
        scale_value = self.op_params.scale_value
        qkEleRes = qkBmmRes * scale_value
        fia_debug_data(f"mm1*scale output", qkEleRes)
        return qkEleRes

    @timeit_decorator
    def _calculate_pse(self, qkEleRes, b_idx, n_idx):
        if not self.op_params.pse_shift_flag:
            return qkEleRes

        pse_cur = self.pse_cur[:, :, self.q_s_begin:self.q_s_end, self.kv_s_begin:self.kv_s_end]
        np.add(qkEleRes, pse_cur, out=qkEleRes)

        fia_debug_data("calculate pse output", qkEleRes)
        return qkEleRes

    @timeit_decorator
    def _calculate_atten_mask(self, qkEleRes):
        if not self.op_params.atten_mask_flag:
            return qkEleRes, None

        fia_debug_func_begin("begin FiaOpForward._calculate_atten_mask")
        current_mask = self.mask_cur
        fia_debug_data(f"_calculate_atten_mask mask", current_mask)

        # Adjust mask dimensions based on sparse mode
        if self.op_params.sparse_mode in [2, 3, 4]:
            current_mask = current_mask[:, :, :(self.q_s_end - self.q_s_begin), :(self.kv_s_end - self.kv_s_begin)]
        else:
            current_mask = current_mask[:, :, self.q_s_begin:self.q_s_end, self.kv_s_begin:self.kv_s_end]
        # Apply mask to attention scores
        if self.op_params.atten_mask.dtype == 'float16':
            qkEleRes = np.where(current_mask, qkEleRes - 10000, qkEleRes)
        else:
            qkEleRes[current_mask.astype(np.bool_)] = -1.7e38

        fia_debug_data(f"_calculate_atten_mask output", qkEleRes)
        return qkEleRes, current_mask

    @timeit_decorator
    def _calculate_softmax(self, qkEleRes, sinks=None):
        return FiaSoftmax.softmax(qkEleRes, sinks)

    @timeit_decorator
    def _calculate_bmm2(self, softmax_res, softmax_sum, v_cur, matmul_dtype):
        assert isinstance(softmax_res, np.ndarray), "softmax_res must be a numpy array"
        assert isinstance(v_cur, np.ndarray), "v_cur must be a numpy array"
        assert isinstance(softmax_sum, (np.ndarray, float)), "softmax_sum must be a numpy array or float"

        if self.query.dtype == "float16":
            softmax_res = softmax_res.astype(np.float16)
        elif self.query.dtype == "bfloat16":
            softmax_res = softmax_res.astype(tf.bfloat16.as_numpy_dtype)

        bmm2Res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype)

        if isinstance(softmax_sum, np.ndarray):
            bmm2Res /= softmax_sum
        else:
            bmm2Res = bmm2Res / softmax_sum
        fia_debug_data(f"bmm2 output", bmm2Res)

        return bmm2Res

    @timeit_decorator
    def _calculate_bmm2_mask(self, bmm2Res, mask_cur, n1):
        if self.op_params.out_quant_flag:
            bmm2Res = self._post_quant(bmm2Res, n1)
        if mask_cur is not None:
            reshaped_bmm2 = bmm2Res.reshape(-1, mask_cur.shape[2], bmm2Res.shape[3])
            mask_to_zero = mask_cur.all(axis=(0, 1, 3))
            reshaped_bmm2[:, mask_to_zero, :] = 0
            fia_debug_data("bmm2 mask output", bmm2Res)
        return bmm2Res

    @timeit_decorator
    def _calculate_lse(self, softmax_sum, softmax_max, mask_cur):
        lse_flag = self.op_params.softmax_lse_flag
        if not lse_flag:
            return None

        fia_debug_func_begin("begin FiaOpForward._calculate_lse")
        lse = np.log(softmax_sum) + softmax_max
        if mask_cur is not None:
            mask_to_default = mask_cur.all(axis=(0, 1, 3))
            reshaped_lse = lse.reshape(-1, mask_cur.shape[2], lse.shape[3])
            reshaped_lse[:, mask_to_default, :] = self.lse_default_value
        fia_debug_data("lse output", lse)

        return lse
    
    @staticmethod
    def _get_torch_dtype(dtype):
        if dtype == 'bfloat16':
            return torch.bfloat16
        else:
            return torch.float32

    def compute_once_bnsd(self, q, k, v, b_idx, n_idx, sinks=None):
        matmul_dtype = self._get_matmul_dtype()
        fia_debug_data(f"q", q)
        fia_debug_data(f"k", k)
        fia_debug_data(f"v", v)

        qkBmmRes = self._calculte_bmm1(q, k, matmul_dtype)
        qkEleRes = self._calculte_scale(qkBmmRes)
        qkEleRes = self._calculate_pse(qkEleRes, b_idx, n_idx)
        qkEleRes, mask_cur = self._calculate_atten_mask(qkEleRes)
        if self.op_params.sink_flag:
            fia_debug_data(f"_t_ifaattention_act sink input", sinks)
        softmax_res, softmax_sum, softmax_max = self._calculate_softmax(qkEleRes, sinks)
        bmm2Res = self._calculate_bmm2(softmax_res, softmax_sum, v, matmul_dtype)
        bmm2Res = self._calculate_bmm2_mask(bmm2Res, mask_cur, n_idx)
        lse = self._calculate_lse(softmax_sum, softmax_max, mask_cur)
        return bmm2Res, lse
    
    @timeit_decorator
    def _get_atc_seq_qkv(self, b_idx, q_s):
        debug_parts = [f"b_idx:{b_idx}"]

        if self.op_params.actual_seq_lens_q_flag:
            self.act_seq_q = self.op_params.actual_seq_lens_q[b_idx]
            debug_parts.append(f"act_seq_q:{self.act_seq_q}")
        else:
            self.act_seq_q = q_s
            debug_parts.append(f"q_s:{self.act_seq_q}")

        prefix_len = self.op_params.prefix_act_lens if self.op_params.shared_prefix_flag else 0

        if self.op_params.actual_seq_lens_kv_flag:
            self.act_seq_kv = self.op_params.actual_seq_lens_kv[b_idx]
            debug_parts.append(f"act_seq_kv:{self.act_seq_kv}")
        else:
            self.act_seq_kv = self.op_params.kv_s
            debug_parts.append(f"k_s:{self.act_seq_kv}")

        fia_debug(" ".join(debug_parts))

        return self.act_seq_kv, self.act_seq_q

    @timeit_decorator
    def _process_shared_prefix(self, k, v, bidx, nidx):
        if nidx is not None:
            nidx = nidx // self.n_factor
        if self.op_params.shared_prefix_flag:
            k_shared_prefix = self.op_params.k_shared_prefix.data
            v_shared_prefix = self.op_params.v_shared_prefix.data
            k_shared_prefix = k_shared_prefix[0:1, nidx:nidx + 1, :, :]
            v_shared_prefix = v_shared_prefix[0:1, nidx:nidx + 1, :, :]
            if k_shared_prefix is None or v_shared_prefix is None:
                raise ValueError("Shared prefix data is not available.")
            k = np.concatenate((k_shared_prefix, k), axis=2)
            v = np.concatenate((v_shared_prefix, v), axis=2)
        return k, v

    def _get_atten_mask_cur(self, b_idx):
        # 判断attenmask是否为空
        mask_cur = None
        if self.op_params.atten_mask_flag:
            if self.op_params.prefix_act_flag or self.op_params.prefix_act_lens > 0:
                mask_cur = np.zeros([1, 1, self.op_params.q_s, self.op_params.kv_s + self.op_params.prefix_act_lens],
                                    dtype='uint8')
            else:
                mask_cur = np.zeros([1, 1, self.op_params.q_s, self.op_params.kv_s], dtype='uint8')
            mask_cur[0, 0, :, :] = self.op_params.atten_mask.data[b_idx]
        return mask_cur

    def _get_pse_cur(self, b_idx, n_idx=None):
        # 判断pse是否为空,如果非空,检查pse第一维是否为1：如果格式为1n1s,则直接传入下层计算;如果格式为bn1s,则按B拆分后进入下层。
        if not self.op_params.pse_shift_flag:
            pse_cur = None
        elif self.op_params.pse_shift.shape[0] == 1:
            if n_idx is None:
                pse_cur = self.op_params.pse_shift.data[:, :, :, :]
            else:
                pse_cur = self.op_params.pse_shift.data[:, n_idx:(n_idx + 1), :, :]
        else:
            if n_idx is None:
                pse_cur = self.op_params.pse_shift.data[b_idx:(b_idx + 1), :, :, :]
            else:
                pse_cur = self.op_params.pse_shift.data[b_idx:(b_idx + 1), n_idx:(n_idx + 1), :, :]
        return pse_cur

    def _get_sink_cur(self, n_idx):
        # 判断sink是否为空
        sink_cur = None
        if self.op_params.sink_flag:
            sink_cur = self.op_params.sinks.data[n_idx:(n_idx + 1)]
        return sink_cur

    def _get_k_shape_by_idx(self, b_idx):
        if self.op_params.storage_mode != StorageMode.TENSOR_LIST:
            return self.key.bnsd_shape
        return self.key.bnsd_shape_list[b_idx]

    def _get_v_shape_by_idx(self, b_idx):
        if self.op_params.storage_mode != StorageMode.TENSOR_LIST:
            return self.key.bnsd_shape
        return self.key.bnsd_shape_list[b_idx]

    def _get_q_by_idx(self, b_idx, q_s_begin, q_s_end, n_idx=None):
        if n_idx is None:
            return self.op_params.query.bnsd_data[b_idx:(b_idx + 1), :, q_s_begin:q_s_end, :]
        else:
            return self.op_params.query.bnsd_data[b_idx:(b_idx + 1), n_idx:(n_idx + 1), q_s_begin:q_s_end, :]

    def _get_from_list(self, data_list, b_idx, kv_s_begin, kv_s_end, n_idx=None):
        if n_idx is None:
            return data_list[b_idx][:, :, kv_s_begin:kv_s_end, :]
        else:
            return data_list[b_idx][:, n_idx:(n_idx + 1), kv_s_begin:kv_s_end, :]

    def _get_from_tensor(self, tensor, b_idx, kv_s_begin, kv_s_end, n_idx=None):
        if n_idx is None:
            return tensor[b_idx:(b_idx + 1), :, kv_s_begin:kv_s_end, :]
        else:
            return tensor[b_idx:(b_idx + 1), n_idx:(n_idx + 1), kv_s_begin:kv_s_end, :]

    def _get_kv_by_idx(self, fia_tensor, b_idx, kv_s_begin, kv_s_end, n_idx=None):
        storage_mode = self.op_params.storage_mode
        if n_idx is not None:
            n_idx = n_idx // self.n_factor

        if storage_mode != StorageMode.TENSOR_LIST:
            return self._get_from_tensor(fia_tensor.bnsd_data, b_idx, kv_s_begin, kv_s_end, n_idx)
        else:
            return self._get_from_list(fia_tensor.bnsd_data_list, b_idx, kv_s_begin, kv_s_end, n_idx)

    def _get_k_by_idx(self, b_idx, kv_s_begin, kv_s_end, n_idx=None):
        return self._get_kv_by_idx(self.key, b_idx, kv_s_begin, kv_s_end, n_idx)

    def _get_v_by_idx(self, b_idx, kv_s_begin, kv_s_end, n_idx=None):
        return self._get_kv_by_idx(self.value, b_idx, kv_s_begin, kv_s_end, n_idx)

    def compute_bnsd(self):
        y = self.attention_out_bnsd.data
        lse = self.lse_bnsd.data
        if (
                self.op_params.q_padding_size_flag or self.op_params.kv_padding_size_flag) and self.op_params.sparse_mode == 0:
            self.op_params.pre_tokens = 2147483647
            self.op_params.next_tokens = 2147483647
        for b_idx in range(self.op_params.batch):
            act_seq_kv, act_seq_q = self._get_atc_seq_qkv(b_idx, self.op_params.q_s)
            if act_seq_kv == 0 and self.op_params.q_s == 1:
                for n_idx in range(self.op_params.num_heads):
                    if self.op_params.out_quant_flag:
                        y[b_idx:(b_idx + 1), n_idx:(n_idx + 1), :, :] = self._post_quant(
                            y[b_idx:(b_idx + 1), n_idx:(n_idx + 1), :, :], n_idx)
                continue
            if act_seq_kv == 0 or act_seq_q == 0 or 0 in self._get_k_shape_by_idx(
                    b_idx) or 0 in self._get_v_shape_by_idx(b_idx):
                fia_debug("skip calc for actual seq 0 or kv shape has 0")
                continue
            for n_idx in range(self.op_params.num_heads):
                self._calculate_s_begin_end(b_idx)
                q = self._get_q_by_idx(b_idx, self.q_s_begin, self.q_s_end, n_idx)
                k = self._get_k_by_idx(b_idx, self.kv_s_begin, self.kv_s_end, n_idx)
                v = self._get_v_by_idx(b_idx, self.kv_s_begin, self.kv_s_end, n_idx)
                k, v = self._process_shared_prefix(k, v, b_idx, n_idx)
                if self.op_params.shared_prefix_flag:
                    if self.op_params.prefix_act_lens == 0:
                        self.kv_s_end += self.op_params.k_shared_prefix.shape.S
                    else:
                        self.kv_s_end += self.op_params.prefix_act_lens
                self.mask_cur = self._get_atten_mask_cur(b_idx)
                self.pse_cur = self._get_pse_cur(b_idx, n_idx)
                sinks = self._get_sink_cur(n_idx)
                y[b_idx:(b_idx + 1), n_idx:(n_idx + 1), self.q_s_begin:self.q_s_end, :], \
                    lse[b_idx:(b_idx + 1), n_idx:(n_idx + 1), self.q_s_begin:self.q_s_end, :] = \
                    self.compute_once_bnsd(q, k, v, b_idx, n_idx, sinks)
        self.attention_out_bnsd.data = y
        return y, lse

    def padding_size_overflow(self):
        if self.op_params.kv_padding_size_flag:
            max_act_seq = max(self.op_params.actual_seq_lens_kv_raw)
            kv_s = self.op_params.kv_s
            if kv_s - self.op_params.kv_padding_size - max_act_seq < 0:
                fia_warn('paddingsize 溢出，输出空tensor！')
                return True
        return False

    def padding_size_overflow(self):
        if not self.op_params.kv_padding_size_flag:
            return False

        max_act_seq = max(self.op_params.actual_seq_lens_kv_raw)
        kv_s = self.op_params.kv_s
        kv_padding_size = self.op_params.kv_padding_size

        if kv_s - kv_padding_size - max_act_seq < 0:
            fia_warn(
                f'kv_padding_size overflow！kv_s={kv_s}，kv_padding_size={kv_padding_size}，max_act_seq={max_act_seq}')
            return True

        return False

    def route_to_old(self):
        if (self.op_params.input_layout in ['SH', 'NSD']) or \
                (self.op_params.query.dtype not in ['float16', 'bfloat16', 'float32']) or \
                (self.op_params.key.dtype not in ['float16', 'bfloat16', 'float32']) or \
                (self.op_params.query.dtype != self.op_params.key.dtype):
            return True
        # if self.op_params.query.D not in [64, 128, 192, 512]:
        #     return True
        fia_debug("*********************************ROUTE TO FIA success")
        return False

    def route_to_old_ifa(self):
        if self.op_params.sparse_mode == 4:
            fia_debug("*********************************ROUTE TO PFA")
            return False
        if (self.op_params.query.S == 1) or (self.op_params.rope_flag and self.op_params.query.D == 512) or \
                (self.op_params.query.dtype != self.op_params.key.dtype):
            fia_debug("*********************************ROUTE TO IFA")
            return True
        fia_debug("*********************************ROUTE TO PFA")
        return False

    def forward_numpy_old(self):
        
        if self.route_to_old_ifa():
            return ifa.aclnn_op_func_ifa_cpu(self.data_list, self.params)
        else:
            return pfa.aclnnPromptFlashAttention_unification(self.data_list, self.params)

    def forward_numpy(self):
        if self.route_to_old():
            return self.forward_numpy_old()

        if (not self.op_params.normal_flag) or self.padding_size_overflow():
            return torch.zeros(self.op_params.output.shape)

        self.fia_op_preprocess.preprocess()

        self.query = self.op_params.query
        self.key = self.op_params.key
        self.value = self.op_params.value

        self.attention_out_bnsd = FiaTensor(
            np.zeros(self.op_params.output.bnsd_shape, dtype=np.float32),
            self.op_params.output.bnsd_shape,
            "fp32", "BNSD", name="attention_out_bnsd"
        )
        self.lse_bnsd = FiaTensor(
            np.full(self.op_params.lse.bnsd_shape, self.lse_default_value),
            self.op_params.lse.bnsd_shape,
            "fp32", "BNSD", name="lse_bnsd"
        )

        self.compute_bnsd()

        y_all = self.attention_out_bnsd.to_layout(self.op_params.out_layout, self.op_params.actual_seq_lens_q)
        fia_debug_data(f"final output", y_all)

        # print_performance_report()
        if self.op_params.softmax_lse_flag:
            lse = self.lse_bnsd.to_layout(self.op_params.lse_layout, self.op_params.actual_seq_lens_q)
            fia_debug_data(f"final lse output", lse)
            return torch.from_numpy(y_all), torch.from_numpy(lse)
        else:
            return torch.from_numpy(y_all)

    def forward_torch(self):
        
        print("flash_mla_with_kvcache...")
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        self.query = self.op_params.query
        self.key = self.op_params.key
        self.value = self.op_params.value

        self.fia_op_preprocess.preprocess()

        causal = True if self.op_params.sparse_mode == 3 else False

        q_s = self.query.S
        q_head_nums = self.query.N
        head_dim_v = self.value.D
        kv_head_nums = self.key.N

        q = self.op_params.query.bsnd_data
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            self.op_params.cache_seqlens, q_s * q_head_nums // kv_head_nums, kv_head_nums
        )

        def flash_mla():
            fia_debug("*************************************start flash_mla_with_kvcache*************************")
            return flash_mla_with_kvcache(
                q,
                self.op_params.blocked_k,
                self.op_params.block_table,
                self.op_params.cache_seqlens,
                head_dim_v,
                tile_scheduler_metadata,
                num_splits,
                causal=causal,
            )

        out_flash, lse_flash = flash_mla()
        return out_flash.cpu()

    def forward(self):
        if self.mode == 'numpy':
            return self.forward_numpy()
        elif self.mode == 'torch':
            return self.forward_torch()
        else:
            raise ValueError(f"Unsupported mode {self.mode}")

# ATK 处理逻辑
dtype_map = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
    torch.int8: "int8",
    torch.bool: "bool",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uin8",
    torch.float64: "fp64"
}

def aclnn_op_func_fia_cpu(input_data : InputDataset, case_id, is_benchmark_task):
    tensor_list = [None] * 29
    shape_input = [[1]] * 29
    range_input = [['null', 'null']] * 29
    dtype_input = ['fp16'] * 29
    format_input = ['ND'] * 29
    type_input = ['tensor'] * 29

    params = {
        'dtype_output': ['fp16'], 
        'attr_1': 'actualseqlengths', 'actualseqlengths': [], 'required_actualseqlengths': 1, 
        'attr_2': 'actualseqlengthskv', 'actualseqlengthskv': [], 'required_actualseqlengthskv': 1, 
        'attr_3': 'prefix_act_lens', 'prefix_act_lens': [], 'required_prefix_act_lens': 1, 
        'attr_4': 'numheads', 'numheads': 8, 'required_numheads': 1, 
        'attr_5': 'scalevalue', 'scalevalue': 0.08838834764831843, 'required_scalevalue': 1, 
        'attr_6': 'pretokens', 'pretokens': 2147483647, 'required_pretokens': 1, 
        'attr_7': 'nexttokens', 'nexttokens': 2147483647, 'required_nexttokens': 1, 
        'attr_8': 'inputlayout', 'inputlayout': 'BNSD', 'required_inputlayout': 1, 
        'attr_9': 'numkeyvalueheads', 'numkeyvalueheads': 8, 'required_numkeyvalueheads': 1, 
        'attr_10': 'sparsemode', 'sparsemode': 0, 'required_sparsemode': 1, 
        'attr_11': 'innerprecise', 'innerprecise': 0, 'required_innerprecise': 1, 
        'attr_12': 'blocksize', 'blocksize': 0, 'required_blocksize': 1, 
        'attr_13': 'antiquant_mode', 'antiquant_mode': 0, 'required_antiquant_mode': 1, 
        'attr_14': 'softmax_lse_flag', 'softmax_lse_flag': False, 'required_softmax_lse_flag': 1, 
        'attr_15': 'k_antiquant_mode', 'k_antiquant_mode': 0, 'required_k_antiquant_mode': 1, 
        'attr_16': 'v_antiquant_mode', 'v_antiquant_mode': 0, 'required_v_antiquant_mode': 1, 
        'attr_17': 'query_quant_mode', 'query_quant_mode': 0, 'required_query_quant_mode': 1, 
        'attr_18': 'fused_flag', 'fused_flag': 'yes', 'required_fused_flag': 1, 
        'attr_19': 'mrandomtype', 'mrandomtype': 'Normal', 'required_mrandomtype': 1, 
        'attr_20': 'mrandom', 'mrandom': 0, 'required_mrandom': 1, 
        'attr_21': 'prandom', 'prandom': 0, 'required_prandom': 1, 
        'attr_22': 'enablegpu', 'enablegpu': 'True', 'required_enablegpu': 1, 
        'attr_23': 'flaglist', 'flaglist': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 'required_flaglist': 1}
    
    params["is_benchmark_task"] = is_benchmark_task
    params["case_id"] = case_id

    if input_data.kwargs["query"].dtype == torch.int8:
        tensor_list[0] = query = input_data.kwargs["query"].to(dtype=torch.int8).numpy()
        tensor_list[1] = key = input_data.kwargs["key"][0].to(dtype=torch.int8).numpy()
        tensor_list[2] = value = input_data.kwargs["value"][0].to(dtype=torch.int8).numpy()
    elif input_data.kwargs["antiquantScaleOptional"] != None:
        tensor_list[0] = query = input_data.kwargs["query"].to(dtype=torch.float32).numpy()
        tensor_list[1] = key = input_data.kwargs["key"][0].to(dtype=torch.int8).numpy()
        tensor_list[2] = value = input_data.kwargs["value"][0].to(dtype=torch.int8).numpy()
    else:
        tensor_list[0] = query = input_data.kwargs["query"].to(dtype=torch.float16).numpy()
        tensor_list[1] = key = input_data.kwargs["key"][0].to(dtype=torch.float16).numpy()
        tensor_list[2] = value = input_data.kwargs["value"][0].to(dtype=torch.float16).numpy()
        
    tensor_list[3] = pse = input_data.kwargs["pseShiftOptional"] if input_data.kwargs["pseShiftOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[4] = attenmask = input_data.kwargs["attenMaskOptional"] if input_data.kwargs["attenMaskOptional"] != None else np.array([], dtype=np.float32)

    tensor_list[5] = dequantscale1 = input_data.kwargs["deqScale1Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["deqScale1Optional"] != None else np.array([], dtype=np.float32)
    tensor_list[6] = quantscale1 = input_data.kwargs["quantScale1Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["quantScale1Optional"] != None else np.array([], dtype=np.float32)
    tensor_list[7] = dequantscale2 = input_data.kwargs["deqScale2Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["deqScale2Optional"] != None else np.array([], dtype=np.float32)

    tensor_list[8] = quantscale2 = input_data.kwargs["quantScale2Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["quantScale2Optional"] != None else np.array([], dtype=np.float32)
    tensor_list[9] = quantoffset2 = input_data.kwargs["quantOffset2Optional"].to(dtype=torch.float32).numpy() if input_data.kwargs["quantOffset2Optional"] != None else np.array([], dtype=np.float32)
    tensor_list[10] = antiquantscale = input_data.kwargs["antiquantScaleOptional"].to(dtype=torch.float32).numpy() if input_data.kwargs["antiquantScaleOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[11] = antiquantoffset = input_data.kwargs["antiquantOffsetOptional"].to(dtype=torch.float32).numpy() if input_data.kwargs["antiquantOffsetOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[12] = blocktable = input_data.kwargs["blockTableOptional"] if input_data.kwargs["blockTableOptional"] != None else np.array([], dtype=np.int32)

    tensor_list[13] = q_padding_size = input_data.kwargs["queryPaddingSizeOptional"] if input_data.kwargs["queryPaddingSizeOptional"] != None else np.array([], dtype=np.uint64)
    tensor_list[14] = padding_size = input_data.kwargs["kvPaddingSizeOptional"] if input_data.kwargs["kvPaddingSizeOptional"] != None else np.array([], dtype=np.uint64)
    tensor_list[15] = k_antiquantscale = input_data.kwargs["keyAntiquantScaleOptional"] if input_data.kwargs["keyAntiquantScaleOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[16] = k_antiquantoffset = input_data.kwargs["keyAntiquantOffsetOptional"] if input_data.kwargs["keyAntiquantOffsetOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[17] = v_antiquantscale = input_data.kwargs["valueAntiquantScaleOptional"] if input_data.kwargs["valueAntiquantScaleOptional"] != None else np.array([], dtype=np.float32)

    tensor_list[18] = v_antiquantoffset = input_data.kwargs["valueAntiquantOffsetOptional"] if input_data.kwargs["valueAntiquantOffsetOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[19] = k_prefix = input_data.kwargs["keySharedPrefixOptional"] if input_data.kwargs["keySharedPrefixOptional"] != None else np.array([], dtype=np.int8)
    tensor_list[20] = v_prefix = input_data.kwargs["valueSharedPrefixOptional"] if input_data.kwargs["valueSharedPrefixOptional"] != None else np.array([], dtype=np.int8)
    
    tensor_list[23] = q_rope = input_data.kwargs["queryRopeOptional"] if input_data.kwargs["queryRopeOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[24] = k_rope = input_data.kwargs["keyRopeOptional"] if input_data.kwargs["keyRopeOptional"] != None else np.array([], dtype=np.float32)

    tensor_list[26] = k_rope_antiquantScale = input_data.kwargs["keyRopeAntiquantScaleOptional"] if input_data.kwargs["keyRopeAntiquantScaleOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[27] = dequantScale_query = input_data.kwargs["dequantScaleQueryOptional"] if input_data.kwargs["dequantScaleQueryOptional"] != None else np.array([], dtype=np.float32)
    tensor_list[28] = sinks = input_data.kwargs["learnableSinkOptional"] if input_data.kwargs["learnableSinkOptional"] != None else np.array([], dtype=np.float32)

    shape_input[0] = list(input_data.kwargs["query"].shape)
    dtype_input[0] = dtype_map[input_data.kwargs["query"].dtype]
    params['dtype_output'][0] = dtype_map[input_data.kwargs["query"].dtype]

    shape_input[1] = list(input_data.kwargs["key"][0].shape)
    dtype_input[1] = dtype_map[input_data.kwargs["key"][0].dtype]
    shape_input[2] = list(input_data.kwargs["value"][0].shape)
    dtype_input[2] = dtype_map[input_data.kwargs["value"][0].dtype]

    if input_data.kwargs["pseShiftOptional"] != None:
        params['flaglist'][3] = 1
        dtype_input[3] = dtype_map[input_data.kwargs["pseShiftOptional"].dtype]
        shape_input[3] = list(input_data.kwargs["pseShiftOptional"].shape)
    
    if input_data.kwargs["attenMaskOptional"] != None:
        params['flaglist'][4] = 1
        dtype_input[4] = dtype_map[input_data.kwargs["attenMaskOptional"].dtype]
        shape_input[4] = list(input_data.kwargs["attenMaskOptional"].shape)
    
    if input_data.kwargs["actualSeqLengthsOptional"] != None:
        params['flaglist'][5] = 1

    if input_data.kwargs["actualSeqLengthsKvOptional"] != None:
        params['flaglist'][6] = 1

    if input_data.kwargs["deqScale1Optional"] != None:
        params['flaglist'][7] = 1
        dtype_input[5] = dtype_map[input_data.kwargs["deqScale1Optional"].dtype]
        shape_input[5] = list(input_data.kwargs["deqScale1Optional"].shape)

    if input_data.kwargs["quantScale1Optional"] != None:
        params['flaglist'][8] = 1
        dtype_input[6] = dtype_map[input_data.kwargs["quantScale1Optional"].dtype]
        shape_input[6] = list(input_data.kwargs["quantScale1Optional"].shape)

    if input_data.kwargs["deqScale2Optional"] != None:
        params['flaglist'][9] = 1
        dtype_input[7] = dtype_map[input_data.kwargs["deqScale2Optional"].dtype]
        shape_input[7] = list(input_data.kwargs["deqScale2Optional"].shape)

    if input_data.kwargs["quantScale2Optional"] != None:
        params['flaglist'][10] = 1
        dtype_input[8] = dtype_map[input_data.kwargs["quantScale2Optional"].dtype]
        shape_input[8] = list(input_data.kwargs["quantScale2Optional"].shape)
    
    if input_data.kwargs["quantOffset2Optional"] != None:
        params['flaglist'][11] = 1
        dtype_input[9] = dtype_map[input_data.kwargs["quantOffset2Optional"].dtype]
        shape_input[9] = list(input_data.kwargs["quantOffset2Optional"].shape)

    if input_data.kwargs["antiquantScaleOptional"] != None:
        params['flaglist'][12] = 1
        dtype_input[10] = dtype_map[input_data.kwargs["antiquantScaleOptional"].dtype]
        shape_input[10] = list(input_data.kwargs["antiquantScaleOptional"].shape)
    
    if input_data.kwargs["antiquantOffsetOptional"] != None:
        params['flaglist'][13] = 1
        dtype_input[11] = dtype_map[input_data.kwargs["antiquantOffsetOptional"].dtype]
        shape_input[11] = list(input_data.kwargs["antiquantOffsetOptional"].shape)
    
    if input_data.kwargs["blockTableOptional"] != None:
        params['flaglist'][14] = 1
        dtype_input[12] = dtype_map[input_data.kwargs["blockTableOptional"].dtype]
        shape_input[12] = list(input_data.kwargs["blockTableOptional"].shape)
        actual_seq = key.shape[1] if input_data.kwargs["inputLayout"] in ["BSH", "BSND"] else key.shape[2]
        headdim = key.shape[3] if input_data.kwargs["inputLayout"] in ["BSND", "BNSD"] else key.shape[2] / input_data.kwargs["numKeyValueHeads"]
        block_num = math.ceil(actual_seq / input_data.kwargs["blockSize"])
        cache_shape = [block_num, input_data.kwargs["blockSize"], input_data.kwargs["numKeyValueHeads"] * headdim]
        shape_input[21] = cache_shape
        shape_input[22] = cache_shape

    params["actualseqlengths"] = list(input_data.kwargs["actualSeqLengthsOptional"])
    params["actualseqlengthskv"] = list(input_data.kwargs["actualSeqLengthsKvOptional"])
    params["prefix_act_lens"] = list(input_data.kwargs["actualSharedPrefixLenOptional"])
    params["numheads"] = input_data.kwargs["numHeads"]
    params["scalevalue"] = input_data.kwargs["scaleValue"]
    params["pretokens"] = input_data.kwargs["preTokens"]
    params["nexttokens"] = input_data.kwargs["nextTokens"]
    params["inputlayout"] = input_data.kwargs["inputLayout"]
    params["numkeyvalueheads"] = input_data.kwargs["numKeyValueHeads"]
    params["sparsemode"] = input_data.kwargs["sparseMode"]
    params["innerprecise"] = input_data.kwargs["innerPrecise"]
    params["blocksize"] = input_data.kwargs["blockSize"]
    params["antiquant_mode"] = input_data.kwargs["antiquantMode"]
    params["softmax_lse_flag"] = input_data.kwargs["softmaxLseFlag"]
    params["k_antiquant_mode"] = input_data.kwargs["keyAntiquantMode"]
    params["v_antiquant_mode"] = input_data.kwargs["valueAntiquantMode"]
    params["query_quant_mode"] = input_data.kwargs["queryQuantMode"]

    params['shape_input'] = shape_input
    params['dtype_input'] = dtype_input
    params['range_input'] = range_input
    params['format_input'] = format_input
    params['type_input'] = type_input
    params['action_type'] = 'bm'
    output = FiaOpForward(tensor_list, params).forward()
    
    return output.to(dtype=input_data.kwargs["query"].dtype)

@register("executor_fused_infer_attention_score_v4")
class fusedInferAttentionScoreApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(fusedInferAttentionScoreApi, self).__init__(task_result)
    
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if input_data.kwargs["softmaxLseFlag"] == True:
            output, output_lse = aclnn_op_func_fia_cpu(input_data, case_id = self.task_result.case_config.id, is_benchmark_task = True)
            return output, output_lse
        else:
            output = aclnn_op_func_fia_cpu(input_data, case_id = self.task_result.case_config.id, is_benchmark_task = True)
            return output

@register("executor_aclnn_fused_infer_attention_score_v4")
class aclnnFusedInferAttentionScoreApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnFusedInferAttentionScoreApi, self).__init__(task_result, backend)
    
    def init_by_input_data(self, input_data: InputDataset):
        input_args = []  # 算子的入参列表
        if input_data.kwargs["blockTableOptional"] != None: 
            load_kv_cache(input_data, self.task_result.case_config.id)
        input_args, output_packages = super().init_by_input_data(input_data)
        output_packages = []  # 算子的出参数据包列表
        input_args.pop()
        if input_data.kwargs["softmaxLseFlag"] == True:
            input_args.pop()
            output_packages.append(input_args[-2])
            output_packages.append(input_args[-1])
        else:
            output_packages.append(input_args[-2])
        return input_args, output_packages

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp_output_pack = self.acl_tensor_to_torch(output_pack)
            output.append(temp_output_pack)
        return output