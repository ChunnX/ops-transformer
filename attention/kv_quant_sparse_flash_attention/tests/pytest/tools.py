import numpy as np
import tensorflow as tf
import datetime
import os
import sys
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

def print_log(data=None, level='INFO'):
    print("[%s] [%s]-%s:%s - %s" % (datetime.datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S"), level, os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                    str(sys._getframe().f_back.f_lineno).zfill(4), data))


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


def modify_alcnn_input_file(ids: int,
                            origin_index: list,
                            type: str,
                            mode: str,
                            tensors,
                            params: dict,
                            json_data: dict,
                            keepcs=None,
                            data_range=None,
                            data_dtype=None):
    cs_data = params
    input_dic = dict()
    # 公共信息
    input_dic["name"] = "input" + str(ids)
    input_dic["type"] = type
    input_dic["acl_format"] = "nd"
    input_dic["data_distribution"] = "uniform"

    if tensors is not None:
        for i in json_data["input"]:
            if i["name"] == "input" + str(origin_index[0]):
                # 在这里请不要修改，会影响库上已有用例
                input_dic["acl_format"] = i["acl_format"]
                json_data["input"].remove(i)

        input_dic["data_range"] = data_range or [-10.0, 10.0]
        input_dic["shape"] = tensors.shape
        input_dic["dtype"] = get_tensor_dtype(tensors, data_dtype)

        json_data["input"].insert(ids, input_dic)
    else:
        for i in json_data["input"]:
            if i["name"] == "input" + str(origin_index[0]):
                i["name"] == "input" + str(ids)

    cs_dtype_input = []
    cs_format_input = []
    cs_range_input = []
    cs_shape_input = []
    cs_type_input = []

    for i in json_data["input"]:
        if re.search(r'input\d+', i['name']):
            if i["type"] == 'tensor_list':
                for j in range(len(i["shapes"])):
                    cs_dtype_input.append(i["dtype"])
                    cs_format_input.append(i["acl_format"])
                    cs_range_input.append(i["data_range"][j])
                    cs_shape_input.append(i["shapes"][j])
                    cs_type_input.append('tensor')
            elif i["type"] == 'tensor':
                cs_dtype_input.append(i["dtype"])
                cs_format_input.append(i["acl_format"])
                cs_range_input.append(i["data_range"])
                cs_shape_input.append(i["shape"])
                cs_type_input.append(i["type"])

    cs_data["dtype_input"] = cs_dtype_input
    cs_data["format_input"] = cs_format_input
    cs_data["range_input"] = cs_range_input
    cs_data["shape_input"] = cs_shape_input
    cs_data["type_input"] = cs_type_input
    return tensors
