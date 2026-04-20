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



def dump_file(data_pool, data_path, type_str, params=None):
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
        print_log(f"[dump_file] input dtype {np_type} is stuct ndarray, will directly save to file {data_path}")
        dump_np_data = data_pool
    elif type_str.lower() in KNOW_NP_DTYPES:
        np_dtype = get_np_dtype(type_str.lower())
        dump_np_data = np.array(data_pool).astype(np_dtype)
    elif type_str.lower() in ['int4', 'float4_e2m1', 'float4_e1m2', 'fp4_e1m2', 'fp4_e2m1']:
        data_pool_np = np.array(data_pool)
        if not is_tensor:
            print_log(f"[dump_file]numpy for {type_str} packed to uint8")  # tensor 输出的场景已经是打包的，numpy输出场景需要打包
            data_pool_np = pack_int4(data_pool_np)
        dump_np_data = np.array(data_pool_np)
    elif type_str.lower() in ['float8_e5m2', 'float8_e4m3fn', 'float8_e8m0', 'hifloat8']:
        print_log(f"[dump_file]for {type_str} will directly save to file")
        dump_np_data = np.array(data_pool)
    else:
        print_log(f"[dump_file]WARN UNKNOW TYPE {type_str},will directly save to file")
        dump_np_data = np.array(data_pool)
    if dump_np_data is not None:
        dump_shape = list(dump_np_data.shape)
        dump_np_data.tofile(data_path)
        file_name = os.path.basename(data_path)
        if "cpu_output" in file_name and isinstance(params, type({})):
            if "shape_output_tmp" not in params.keys():
                params["shape_output_tmp"] = []
            params["shape_output_tmp"].append(dump_shape)
            shape_info = {"shape_output": params["shape_output_tmp"]}
            modify_cs_info(shape_info, params)
        print_log(f"[dump_file]save data to {data_path} with shape:{dump_shape} dtype:{type_str}")

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
                            json_data: dict
                            keepcs=None,
                            data_range=None,
                            data_dtype=None):
    # json_path = os.path.join(params['case_path'], params['case_name'] + ".json")
    # with open(json_path, "r") as f:
    #     json_data = json.load(f)

    # cs_path = os.path.join(params['case_path'], params['case_name'] + ".cs")
    # with open(cs_path, "r") as f:
    #     cs_data = json.load(f)
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
                input_dic["bin_path"] = i["bin_path"]
                # 在这里请不要修改，会影响库上已有用例
                input_dic["acl_format"] = i["acl_format"]
                json_data["input"].remove(i)

        input_dic["data_range"] = data_range or [-10.0, 10.0]
        input_dic["shape"] = tensors.shape
        input_dic["dtype"] = get_tensor_dtype(tensors, data_dtype)
        # tensors.detach().numpy().tofile(input_dic["bin_path"])
        input_file_i = os.path.join(params['case_path'], input_dic["bin_path"])
        if torch.is_tensor(tensors):
            if input_dic["dtype"] == "bf16":
                np.array(tensors.float().detach().cpu().numpy().astype(tf.bfloat16.as_numpy_dtype)).tofile(
                    input_file_i)
            elif input_dic["dtype"] == "float8_e4m3fn":
                safe_write_fp8(tensors, input_file_i)
            else:
                tensors.detach().cpu().numpy().tofile(input_file_i)
        else:
            dump_file(tensors, input_file_i, input_dic["dtype"], params)

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
    cs_bin_path_input = []
    n = 0

    for i in json_data["input"]:
        if re.search(r'input\d+', i['name']):
            exist = True
            if i["type"] == 'tensor_list':
                for j in range(len(i["shapes"])):
                    cs_dtype_input.append(i["dtype"])
                    cs_format_input.append(i["acl_format"])
                    cs_range_input.append(i["data_range"][j])
                    cs_shape_input.append(i["shapes"][j])
                    cs_type_input.append('tensor')
                    cs_bin_path_input.append(i['bin_paths'][j])
            elif i["type"] == 'tensor':
                cs_dtype_input.append(i["dtype"])
                cs_format_input.append(i["acl_format"])
                cs_range_input.append(i["data_range"])
                cs_shape_input.append(i["shape"])
                cs_type_input.append(i["type"])
                cs_bin_path_input.append(i['bin_path'])

    cs_data["dtype_input"] = cs_dtype_input
    cs_data["format_input"] = cs_format_input
    cs_data["range_input"] = cs_range_input
    cs_data["shape_input"] = cs_shape_input
    cs_data["type_input"] = cs_type_input
    cs_data["bin_path_input"] = cs_bin_path_input
    return tensors

