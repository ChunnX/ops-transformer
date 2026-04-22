import numpy as np
import tensorflow as tf
import os
import json
import torch
import re
from ml_dtypes import float8_e4m3fn


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



def convert_tensor_data(data_pool, data_path, type_str, params=None):
    KNOW_NP_DTYPES = [
        'fp64', 'fp32', 'fp16', 'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8', 'bool',
        'complex64', 'complex128', 'complex32', 'bf16', 'bfloat16',
        'fp8_e8m0', 'float32', 'float16', 'qint8', 'qint32', 'quint8', 'qint16',
        "uint1", "quint16", 'fp4_e2m1_as_fp32',
    ]
    dump_shape = None
    is_tensor = False

    np_type = data_pool.dtype
    dump_np_data = None
    if len(np_type) > 1:
        dump_np_data = data_pool
    elif type_str.lower() in KNOW_NP_DTYPES:
        np_dtype = get_np_dtype(type_str.lower())
        dump_np_data = np.array(data_pool).astype(np_dtype)
    elif type_str.lower() in ['int4', 'float4_e2m1', 'float4_e1m2', 'fp4_e1m2', 'fp4_e2m1']:
        data_pool_np = np.array(data_pool)
        if not is_tensor:
            data_pool_np = pack_int4(data_pool_np)
        dump_np_data = np.array(data_pool_np)
    elif type_str.lower() in ['float8_e5m2', 'float8_e4m3fn', 'float8_e8m0', 'hifloat8']:
        dump_np_data = np.array(data_pool)
    else:
        dump_np_data = np.array(data_pool)
    if dump_np_data is not None:
        dump_shape = list(dump_np_data.shape)
        file_name = os.path.basename(data_path)
        if "cpu_output" in file_name and isinstance(params, type({})):
            if "shape_output_tmp" not in params.keys():
                params["shape_output_tmp"] = []
            params["shape_output_tmp"].append(dump_shape)
            shape_info = {"shape_output": params["shape_output_tmp"]}
            modify_cs_info(shape_info, params)
    return dump_np_data


def get_str_dtype(type_str):
    type_dict = {
        torch.float32: 'fp32', torch.float16: 'fp16', torch.float64: 'fp64',
        torch.int8: 'int8', torch.int16: 'int16', torch.int32: 'int32', torch.int64: 'int64',
        torch.uint8: 'uint8', torch.bool: 'bool', torch.complex64: 'complex64',
        torch.complex128: 'complex128', torch.bfloat16: 'bf16'
    }
    return type_dict[type_str]


def get_tensor_dtype(tensor, input_dtype):
    if input_dtype:
        tensor_dtype = input_dtype
    else:
        if torch.is_tensor(tensor):
            tensor_dtype = get_str_dtype(tensor.dtype)
        else:
            tensor_dtype = str(tensor.dtype)
    return tensor_dtype


def modify_alcnn_input_file(name: str,
                            tensors,
                            params: dict,
                            data_range=None,
                            data_dtype=None):
    out_tensor = tensors
    if tensors is not None:
        data_range = data_range or [-10.0, 10.0]
        tensor_dtype = get_tensor_dtype(tensors, data_dtype)
        if torch.is_tensor(tensors):
            if tensor_dtype == "bf16":
                out_tensor = np.array(tensors.float().detach().cpu().numpy().astype(tf.bfloat16.as_numpy_dtype))
            else:
                out_tensor = tensors.detach().cpu().numpy()
        else:
            out_tensor = convert_tensor_data(tensors, name, tensor_dtype, params)

        params["shape_input"][name] = list(tensors.shape)
        params["dtype_input"][name] = tensor_dtype
        params["range_input"][name] = data_range

    return out_tensor
