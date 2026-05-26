import sys
import os
import csv
import math
import time
import random
import ctypes
import copy

import numpy as np
import pandas
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
import torch_npu
from openpyxl import load_workbook
import subprocess
from datetime import datetime
from .generate_hifloat8_data import *
from .generate_float_data import *
from ml_dtypes import float8_e4m3fn, float8_e5m2, bfloat16, int4
from en_dtypes import float8_e8m0, float4_e2m1, float4_e1m2, hifloat8
from typing import Union
import psutil

TYPE_DICT = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.int8: np.int8,
    torch.bfloat16: np.float32,
    torch.int64: np.int64,
    torch.int32: np.int32,
    'hif8': np.float32,
    torch.float8_e4m3fn: float8_e4m3fn,
    torch.float8_e5m2: float8_e5m2
}

UNSUPPORTED_DTYPES = [
    "fp8_e4m3fn",
    "fp8_e5m2",
    "fp8_e8m0",
    "hif8",
    "fp4_e2m1",
    "fp4_e1m2",
]

DTYPE_BYTE_MAP = {
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "fp32": 4,
    "int32": 4,
    "int64": 8,
    "fp8_e4m3fn": 1,
    "fp8_e5m2": 1,
    "fp8_e8m0": 1,
    "hif8": 1
}

TORCH_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "int8": torch.int8,
    "int32": torch.int32,
    "int64": torch.int64,
    "int4": torch.uint8,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_e8m0": torch.uint8,
    "hif8": torch.uint8,
    "fp4_e2m1": torch.uint8,
    "fp4_e1m2": torch.uint8,
}

NUMPY_DTYPE_MAP = {
    "fp16": np.float16,
    "bf16": bfloat16,
    "fp32": np.float32,
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
    "int4": int4,
    "fp8_e4m3fn": float8_e4m3fn,
    "fp8_e5m2": float8_e5m2,
    "fp8_e8m0": float8_e8m0,
    "fp4_e2m1": float4_e2m1,
    "fp4_e1m2": float4_e1m2,
    "hif8": hifloat8,
}

DTYPE_ENUM = {
    "fp16": None,
    "bf16": None,
    "fp32": None,
    "int8": None,
    "int32": None,
    "int64": None,
    "int4": None,
    "fp8_e4m3fn": None,
    "fp8_e5m2": None,
    "fp8_e8m0": 293,  # torch_npu.float8_e8m0
    "hif8": 290,  # torch_npu.hifloat8
    "fp4_e2m1": 296,  # torch_npu.float4_e2m1
    "fp4_e1m2": 297,  # torch_npu.float4_e1m2
}

OUT_DTYPE_ENUM = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

def save_file(folder_path, tensor, dtype, tensor_name, bin_name, rank):
    try:
        if not os.path.exists(folder_path):
            print("Make dir:", folder_path)
            os.makedirs(folder_path)
    except OSError:
        pass
    if bin_name == "None":
        if dtype == torch.bfloat16:
            from ml_dtypes import bfloat16
            np.array(tensor.float().cpu()).astype(bfloat16).tofile(
                '{}/{}.bin'.format(folder_path, tensor_name))
        elif dtype == "int4":
            tensor.tofile('{}/{}.bin'.format(folder_path, tensor_name))
        else:
            np.array(tensor.cpu()).tofile('{}/{}.bin'.format(folder_path, tensor_name))
    else:
        if dtype == 'bf16':
            np.array(tensor.float().cpu()).tofile('{}/{}_{}_{}.bin'.format(folder_path, bin_name, tensor_name, rank))
        else:
            np.array(tensor.cpu()).tofile('{}/{}_{}_{}.bin'.format(folder_path, bin_name, tensor_name, rank))


def pack_int4(src):
    """
    Pack int4 numpy array (each int4 number stored in one byte actually) to be continuous bytes stored in uint8
    """
    if src.dtype.name not in ["int4", "float4_e2m1", "float4_e1m2", "uint8"]:
        raise RuntimeError(f"Dtype of source tensor only support [int4, float4_e2m1, float4_e1m2, uint8], but got {src.dtype.name}")
    pack_size = 2
    shift = np.array([0, 4], dtype=np.uint8)
    array = src
    if src.size % pack_size != 0:
        array = np.pad(src.flatten(), (0, pack_size - src.size % pack_size), mode='constant')
    reshaped = array.reshape([-1, 2])
    # bitwise_and is for arm
    out = np.sum(np.bitwise_and(reshaped.view(np.uint8), 0b00001111) << shift, axis=1, dtype=np.uint8)
    return out


def get_dtype(dtype):
    return TORCH_DTYPE_MAP[dtype]


def write_csv(csv_name, write_list):
    with open(csv_name, 'a', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(write_list)


def get_avg_perf(all_list, run_time=10):
    # 剔除前一半性能不稳定的数据
    valid_run_time = math.ceil(run_time / 2)
    neg_start_idx = len(all_list) // run_time * valid_run_time
    avg_perf = sum(all_list[-neg_start_idx:]) / valid_run_time
    return avg_perf

def get_memory_performance(case, type, cal_mask=False, perf_type='npu',soc='910B'):
    case_name = case['case_name']
    case_path = os.path.join(os.getcwd(), 'Profiling_Result_' + type + case_name)
    performance = 0.0
    golden_hccl_max = 0.0
    golden_mm_max = 0.0
    result_list = []
    memory = 0
    for dirpath, _, filenames in os.walk(case_path):
        for filename in filenames:
            if filename.startswith("memory_record") and "ASCEND_PROFILER_OUTPUT" in dirpath:
                memory_file = filename
                with open(os.path.join(dirpath, memory_file)) as f1:
                    memory_list = []
                    csvreader = csv.reader(f1)
                    for row in list(csvreader)[1:]:
                        if row[2]:
                            memory_list.append(float(row[2]))
                    memory_use = max(memory_list) - min(memory_list)
                if memory < memory_use:
                    memory = memory_use
            if filename.startswith("op_summary"):
                summary_file = filename
                with open(os.path.join(dirpath, summary_file)) as f2:
                    csvreader = csv.reader(f2)
                    headers = next(csvreader)
                    t = headers.index('Task Duration(us)')
                    task_type = headers.index('Task Type')
                    op_type = headers.index('OP Type')
                    hccl_list, matmul_list, mc2_list, aicpu_list = [], [], [], []
                    for row in list(csvreader)[:]:
                        if row[op_type] == "Exp":  # 跳过插入的算子
                            continue
                        if cal_mask:
                            if perf_type == 'golden':
                                if row[task_type] in ["HCCL", "COMMUNICATION"]:
                                    hccl_list.append(float(row[t]))
                                # 910D 大数据量下，需要统计hccl通信算子的AICPU编排耗时
                                elif row[task_type] == "AI_CPU" and soc == '910D':
                                    aicpu_list.append(float(row[t]))
                                elif row[task_type] in ['AI_VECTOR_CORE', 'AI_CORE', 'MIX_AIC', "MIX_AIV"]:
                                    matmul_list.append(float(row[t]))
                            else:
                                if row[task_type] != 'AI_CPU':
                                    mc2_list.append(float(row[t]))
                        else:
                            if row[task_type] != 'AI_CPU':
                                mc2_list.append(float(row[t]))
                if all([hccl_list, aicpu_list]):
                    # 合并aicpu + hccl,910D大数据量场景下
                    hccl_list = [i + j for i, j in zip(hccl_list, aicpu_list)]
                if type == "hccl_mm_" and cal_mask:
                    # 写入hccl耗时/计算耗时
                    avg_hccl = get_avg_perf(hccl_list)
                    avg_matmul = get_avg_perf(matmul_list)
                    if avg_hccl + avg_matmul > performance:
                        performance = avg_hccl + avg_matmul
                        golden_hccl_max = avg_hccl
                        golden_mm_max = avg_matmul
                else:
                    avg_mc2 = get_avg_perf(mc2_list)
                    if avg_mc2 > performance:
                        performance = avg_mc2
    result_list.extend([[golden_hccl_max, golden_mm_max], memory, performance])
    return result_list


def get_case(flags):
    wb = load_workbook(flags.xlsx)
    ws = wb[flags.sheet]
    if len(flags.num) == 2:
        if 'one_hccl' not in flags.scene:
            print("[ERROR] --num should only contains one number, eg. --num 1")
            sys.exit(1)
        start, end = flags.num[0], min(flags.num[1], ws.max_row - 1)
    else:
        if 'one_hccl' in flags.scene:
            print("[ERROR] --num should contains start and end, eg. --num 1 10")
            sys.exit(1)
        start = end = flags.num[0]
    # 获取数据
    datas = []
    for i in range(start, end + 1):
        data = {}
        for column in ws.iter_cols(1, ws.max_column):
            title = column[0].value
            try:
                data[title] = eval(column[i].value)
            except:
                data[title] = column[i].value
        data['sheet_name'] = flags.sheet
        data['op_name'] = flags.op_name
        data['scene'] = flags.scene
        data['soc'] = flags.soc
        data['master_ip'] = flags.master_ip
        data['master_port'] = flags.master_port
        data['server_num'] = flags.server_num
        data['server_index'] = flags.server_index
        data['dev_num'] = flags.dev_num
        data['weight_same'] = flags.weight_same
        data['save_rank'] = data['server_index'] * data['dev_num']
        for i in range(data['server_num']):
            data[f'rank_list{i}'] = list(range(i * data['dev_num'], (i + 1) * data['dev_num']))
        try:
            bin_name = '_'.join(data['input_tensor_dtype']) + '-' + '_'.join(
                [str(j) for i in data['input_tensor_shape'] for j in i])
        except:
            bin_name = data['case_name']
        data.update({'bin_name': bin_name})
        if len(flags.num) == 2:
            datas.append(data)
        else:
            return data
    return datas


def get_hcomm_info(world_size, rank, ep_size=0, tp_size=0, ep_ranks_list=None, tp_ranks_list=None,
                   master_ip='127.0.0.1', master_port='50001'):
    hcomm_info_dist = {
        'default_pg_info': None,
        'ep_hcomm_info': None,
        'group_ep': None,
        'tp_hcomm_info': None,
        'group_tp': None
    }
    init_method = 'tcp://' + master_ip + ':' + master_port
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=init_method
    )
    print(f"rank:{rank} init_process_group success")
    if ep_size and tp_size:
        group_ep, group_tp = setup_ep_tp(rank, tp_size, ep_size, "hccl", ep_ranks_list, tp_ranks_list)
        hcomm_info_dist['ep_hcomm_info'] = group_ep._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        hcomm_info_dist['tp_hcomm_info'] = group_tp._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        hcomm_info_dist['group_ep'] = group_ep
        hcomm_info_dist['group_tp'] = group_tp
    else:
        hcomm_info_dist['default_pg_info'] = _get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    print(f"rank{rank} hcomm info is : {hcomm_info_dist}")
    return dist, hcomm_info_dist


def setup_ep_tp(rank, tp_size, ep_size, backend_type, ep_ranks_list=None, tp_ranks_list=None):
    print(f"rank:{rank} init ep_group", flush=True)
    for i in range(tp_size):  # 初始化EP域, 如果指定ep list,按照指定ep list划分ep域
        if ep_ranks_list:
            ep_ranks = ep_ranks_list[i]
        else:  # [[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
            ep_ranks = [x * tp_size + i for x in range(ep_size)]
        ep_group = dist.new_group(backend=backend_type, ranks=ep_ranks)
        if rank in ep_ranks:
            ep_group_tmp = ep_group
    print(f"rank:{rank} init tp_group", flush=True)
    for i in range(ep_size):  # 初始化TP域, 如果指定tp list,按照指定tp list划分tp域
        if tp_ranks_list:
            tp_ranks = tp_ranks_list[i]
        else:  # [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
            tp_ranks = [x + tp_size * i for x in range(tp_size)]
        tp_group = dist.new_group(backend=backend_type, ranks=tp_ranks)
        if rank in tp_ranks:
            tp_group_tmp = tp_group
    return ep_group_tmp, tp_group_tmp


def get_tensor_list(case, rank, name):
    seed_num = int(case['seed']) if 'seed' in case else 1
    tensors_dtype = get_dtype(case['{}_dtype'.format(name)][0])  # tensor_list所有dtype相同
    tensors_shape = case['{}_shape'.format(name)]
    tensors_range = case['{}_range'.format(name)]
    if name == "x1":
        seed_num += 1 * rank
    elif name == "x2":
        seed_num += 2 * rank
    elif name == "bias":
        seed_num += 3 * rank
    np.random.seed(seed_num)
    tensor_list = []
    for i, shape in enumerate(tensors_shape):
        if tensors_range[i][0] in ["nan", "inf", "-inf"]:
            np_tmp = np.random.uniform(-1, 1, size=shape).astype(TYPE_DICT[tensors_dtype])
            np_tmp = random_inf_nan(np_tmp, shape, 10, tensors_range[i][0])
        else:
            np_tmp = np.random.uniform(float(tensors_range[i][0]), float(tensors_range[i][1]), size=shape).astype(
                TYPE_DICT[tensors_dtype])
        tensor_tmp = torch.tensor(np_tmp)
        if tensors_dtype == torch.bfloat16:
            tensor_tmp = tensor_tmp.to(torch.bfloat16)
        tensor_list.append(tensor_tmp)
    return tensor_list


# 将x按dim维度，进行k/device切分，并生成长度为world_size大小的tensor_list
def get_x_split(x, dim, world_size):
    res = []
    for i in range(world_size):
        tmp = []
        for tensor in x:
            tmp.append(torch.chunk(tensor, world_size, dim=dim)[i].npu())
        res.append(tmp)
    return res


def change_to_one_dim(tensor_list):
    res = []
    for tensor in tensor_list:
        res.append(tensor.reshape(-1))
    return torch.cat(res, dim=0)


def get_tensor_bais(case, rank, name='bias'):
    case_dtype = list(map(get_dtype, case['input_tensor_dtype']))
    seed_num = int(case['seed']) if 'seed' in case else 1
    is_trans_b = int(case['is_trans_b']) if 'is_trans_b' in case else 0
    if name == 'bias':
        bias_dtype = case_dtype[2]
        bias_shape = case['input_tensor_shape'][2]
        bias_range = case['input_tensor_range'][2]
    else:
        bias_dtype = get_dtype(case['{}_dtype'.format(name)])
        bias_shape = case['{}_shape'.format(name)]
        bias_range = case['{}_range'.format(name)]
    # allreduce伪量化场景，weight转置，antiquant_scale/offset需转置，excel未转脚本做处理
    if case['input_tensor_shape'][0][-1] != case['input_tensor_shape'][1][0]:
        if name in ['antiquant_scale', 'antiquant_offset'] and len(bias_shape) == 2 and case[
            'op_name'] == "aclnnWeightQuantMatmulAllReduce" \
                and case.get("quant_mode", "") == 'a16w8':
            bias_shape.reverse()
    if name == "bias":
        seed_num += 3 * rank
    elif name == "antiquant_scale":
        seed_num += 4 * rank
    elif name == "antiquant_offset":
        seed_num += 5 * rank
    elif name == "x3":
        seed_num += 6 * rank
    elif name == "dequant_scale":
        seed_num += 7 * rank
    elif name == "residual":
        seed_num += 8 * rank
    elif name == "gamma":
        seed_num += 9 * rank
    elif name == "pertoken_scale":
        seed_num += 10 * rank
    elif name == "scaleInv1":
        seed_num += 11 * rank
    elif name == "scaleInv2":
        seed_num += 11 * rank

    scaleInv2_trans_flag = True if is_trans_b and name == "scaleInv2" and len(bias_shape) == 2 else False
    if scaleInv2_trans_flag:
        bias_shape = [bias_shape[1], bias_shape[0]]

    np.random.seed(seed_num)
    if bias_range[0] in ["nan", "inf", "-inf"]:
        np_bias = np.random.uniform(-1, 1, size=bias_shape).astype(TYPE_DICT[bias_dtype])
        np_bias = random_inf_nan(np_bias, bias_shape, 10, bias_range[0])
    else:
        np_bias = np.random.uniform(float(bias_range[0]), float(bias_range[1]), size=bias_shape).astype(
            TYPE_DICT[bias_dtype])
    if case["op_name"] == "aclnnWeightQuantMatmulAllReduce" and case.get("quant_mode", "") == 'a16w4':
        # a16w4--cpu:用numpy计算, get_hccl_mm:c++计算, get_npu:c++计算
        tensor_bias = torch.tensor(np_bias).to(bias_dtype)
        save_file(f"bin_path/{case['case_name']}/input/{rank}", tensor_bias, tensor_bias.dtype, name, "None", rank)
        if name == 'bias' and tensor_bias.dtype == torch.bfloat16:
            print("A16W4场景，bias dtype 为bf16时，额外保存fp32的bias_fp32.bin 供小算子级联使用")
            bias_golden = tensor_bias.to(torch.float32)
            name_tmp = f"{name}_fp32"
            save_file(f"bin_path/{case['case_name']}/input/{rank}", bias_golden, bias_golden, name_tmp, "None", rank)
        return np_bias
    else:
        if case['soc'] == '910D':  # david场景，仅aclnn_golden场景需要将tensor转为npu tensor
            bias = torch.tensor(np_bias).npu() if "aclnn_golden" in case["scene"] else torch.tensor(np_bias)
            bias = bias.to(torch.bfloat16) if bias_dtype == torch.bfloat16 else bias
            if "aclnn" in case["scene"]:  # david场景，aclnn场景保存bin文件
                save_file(f"bin_path/{case['case_name']}/input/{rank}", bias, bias.dtype, name, "None", rank)
            if case["scene"] == 'cpu' and scaleInv2_trans_flag:
                bias = bias.t()
        else:
            bias = torch.tensor(np_bias) if "cpu" in case["scene"] else torch.tensor(np_bias).npu()
            bias = bias.to(torch.bfloat16) if bias_dtype == torch.bfloat16 else bias
            if "aclnn_golden" in case["scene"]:
                save_file(f"bin_path/{case['case_name']}/input/{rank}", bias, bias.dtype, name, "None", rank)

        return bias


def remove_even_lines(text):
    # 将字符串按行分割成列表
    lines = text.split('\n')
    # 获取奇数行
    odd_lines = [line for i, line in enumerate(lines) if i % 2 == 1]
    # 将列表重新组合成字符串
    result = '\n'.join(odd_lines)
    return result


def random_inf_nan(array, array_shape, iterations, overflow_data_type, server_num=1):
    inf_nan_map = {
        "nan": np.nan,
        "inf": np.inf,
        "-inf": -np.inf,
    }
    total = int(np.prod(array_shape))
    if total == 0:
        return array
    for _ in range(iterations):
        # 存放随机点的索引值
        point = []
        for shape in array_shape:
            if server_num != 1:
                random.seed(1)
            index = random.randint(0, shape - 1)
            point.append(index)
        if len(point) == 1:
            array[point[0]] = inf_nan_map[overflow_data_type]
        if len(point) == 2:
            array[point[0], point[1]] = inf_nan_map[overflow_data_type]
        if len(point) == 3:
            array[point[0], point[1], point[2]] = inf_nan_map[overflow_data_type]
        if len(point) == 4:
            array[point[0], point[1], point[2], point[3]] = inf_nan_map[overflow_data_type]
    return array


def numpy_to_torch_tensor(np_array: np.ndarray, is_complex32: bool = False):
    if np_array is None:
        return None
    np_dtype = np_array.dtype.name
    if "bfloat16" in np_dtype:
        np_int16 = np_array.view(dtype=np.int16)
        t_int16 = torch.from_numpy(np_int16)
        return t_int16.view(torch.bfloat16)
    elif "int4" in np_dtype or "float4" in np_dtype:
        raise RuntimeError(f"Can only transfer numpy.ndarray to torch.Tensor with dtype [{np_dtype}]")
    elif "float8" in np_dtype:
        if np_dtype not in ("float8_e4m3fn", "float8_e5m2"):
            raise RuntimeError(f"Dtype [{np_dtype}] is not supported to convert to torch.Tensor yet.")
        if not hasattr(torch, np_dtype):
            raise RuntimeError(f"Current pytorch version [{torch.__version__}] is too old. "
                               f"{np_dtype} is not supported.")
        return torch.from_numpy(np_array.view(dtype=np.uint8)).view(getattr(torch, np_dtype))
    elif is_complex32:
        if np_dtype != "float16":
            raise RuntimeError(f"Can only transfer numpy.float16 to torch.complex32 "
                               f"rather than {np_dtype}")
        if not hasattr(torch, "complex32"):
            raise RuntimeError(f"Current pytorch version [{torch.__version__}] is too old. "
                               f"Please update to at least v1.13.1")
        ret = torch.from_numpy(np_array)
        return ret.view(torch.complex32)
    else:
        return torch.from_numpy(np_array)


def torch_to_numpy_tensor(torch_tensor) -> np.ndarray:
    if not isinstance(torch_tensor, torch.Tensor):
        raise RuntimeError(f"Only support torch.Tensor. "
                           f"But got {type(torch_tensor)}")
    torch_dtype = torch_tensor.dtype
    torch_dtype_str = str(torch_dtype)
    if torch_dtype == torch.bfloat16:
        t_int16 = torch_tensor.view(torch.int16)
        np_int16 = t_int16.numpy()
        return np_int16.view(dtype=bfloat16)
    elif "complex32" in torch_dtype_str:
        t_fp16 = torch_tensor.view(torch.float16)
        return t_fp16.numpy()
    elif "float8" in torch_dtype_str:
        np_dtype = eval(f"{torch_dtype_str.split('.')[-1]}")
        np_uint8 = torch_tensor.view(torch.uint8).numpy()
        return np_uint8.view(np_dtype)
    else:
        return torch_tensor.numpy()


def random_input(dtype: str, data_range: list, shape: Union[tuple, list], seed=0, data_type="array"):
    low, high = data_range[0], data_range[1]

    def _random(_dtype, _shape):
        np.random.seed(seed)
        if low in ["nan", "inf", "-inf"]:
            np_input = np.random.uniform(-1, 1, size=shape).astype(dtype)
            array = random_inf_nan(np_input, shape, 10, low)
        elif low == high:
            array = np.full(_shape, fill_value=low, dtype=_dtype)
        else:
            array = np.random.uniform(low, high, _shape).astype(_dtype, copy=False)
        return array

    input_array = _random(dtype, shape)
    if data_type == "tensor":
        return numpy_to_torch_tensor(input_array)
    return input_array


def save_file_new(data, file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if torch.is_tensor(data):
        data = torch_to_numpy_tensor(data)
    elif not isinstance(data, np.ndarray):
        print(f"[ERROR] save_file fail, data type only support torch.tensor or np.ndarray. actuality data is {data}")
    data.tofile(file_path)


def get_memory_diff(case):
    if case['op_name'] not in ["aclnnAlltoAllAllGatherBatchMatMul", "aclnnBatchMatMulReduceScatterAlltoAll"]:
        if 'input_tensor_shape' not in case:
            return 0
        weight_shape = case['input_tensor_shape'][1]
        isbias = case['is_bias'] if 'is_bias' in case else None
        if isinstance(case['input_tensor_dtype'], str):
            input_dtype = case['input_tensor_dtype']
        else:
            case_dtype = list(map(get_dtype, case['input_tensor_dtype']))
            input_dtype = case_dtype[0]
        k = weight_shape[0]
        n = weight_shape[1]
        size = math.ceil((k + 16) * (n + 16) / 512) * 512 * 2
        sync = 2048  # 软同步所需额外workspace
        if isbias and input_dtype == 'torch.float16':
            bias_len = math.ceil(n * 4 / 256) * 256
            size = size + bias_len
        diff = (size + sync) / 1024 / 1024 + 16
    else:
        diff = 0
    return diff


def get_profiling_config(case_name, type):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False, data_simplification=False
    )
    profiling_result = './Profiling_Result_' + type + case_name
    return experimental_config, profiling_result


def save_performance_result_csv(case):
    if 'input_tensor_shape' not in case:
        origin_list = [case["op_name"], case["case_name"]]
        title = ['op_name', 'case_name', 'mc2_memory', 'golden_memory', 'memory_rate', 'mc2_prof', 'golden_prof',
                 'prof_rate', 'recordtime']
    else:
        origin_list = [case["op_name"], case["case_name"], str(case["input_tensor_shape"]), str(case["input_tensor_range"]),
                       str(case["input_tensor_dtype"])]
        title = ['op_name', 'case_name', 'input_tensor_shape', 'input_tensor_range', 'input_tensor_dtype', 'mc2_memory',
                 'golden_memory', 'memory_rate', 'mc2_prof', 'golden_prof', 'prof_rate', 'recordtime']
        cal_mask = True if case['op_name'] in ["aclnnAlltoAllAllGatherBatchMatMul", "aclnnBatchMatMulReduceScatterAlltoAll"] or case['soc'] == '910D' else False
        if cal_mask:
            title = title + ['avg_hccl', 'avg_matmul', 'mask_rate', 'result']
    file = "mc2_{}_pro_result.csv".format(case["op_name"])
    data = pandas.read_csv(file, names=["用例名", "rank", "memory", "performance"])
    list_1 = data.values.tolist()
    mc2_memory = list_1[-1][-2]
    mc2_prof = list_1[-1][-1]
    file = "hccl_mm_{}_pro_result.csv".format(case["op_name"])
    data = pandas.read_csv(file, names=["用例名", "rank", "memory", "performance"])
    list_2 = data.values.tolist()
    golden_memory = list_2[-1][-2]
    golden_prof = list_2[-1][-1]
    memory_rate = float(mc2_memory / golden_memory) if golden_memory else None
    prof_rate = float(golden_prof / mc2_prof) if golden_prof else None
    show_list = origin_list + [mc2_memory, golden_memory, memory_rate, mc2_prof, golden_prof, prof_rate,
                               time.strftime("%Y-%m-%d %H:%M:%S")]
    if cal_mask:
        avg_hccl = list_2[-1][-4]
        avg_matmul = list_2[-1][-3]
        result = "FAIL"
        if case['op_name'] in ["aclnnQuantReduceScatter", "aclnnQuantAllReduce"]:
            avg_hccl, avg_matmul, mask_rate, result = None, None, None, None
        else:
            # 掩盖率计算:(小算子-mc2)/min(小算子计算，小算子通信)
            if avg_matmul and avg_hccl:
                mask_rate = (golden_prof - mc2_prof) / min(avg_hccl, avg_matmul)
            else:
                mask_rate = 0 # 计算/通信时间为0时，不统计掩盖率
            if case['op_name'] in ["aclnnAlltoAllAllGatherBatchMatMul", "aclnnBatchMatMulReduceScatterAlltoAll"]:
                # pass标准：(golden_prof<2000)小shape>10%，(golden_prof>2000)大shape>20%
                if (golden_prof <= 2000 and mask_rate > 0.1) or (golden_prof > 2000 and mask_rate > 0.2):
                    result = "PASS"
            elif mask_rate > 0.7:
                ## 910D上其他算子，切分数量>8块时，掩盖率70%，小于8块时，(n-1)/n * 0.8，当前仅考虑>8块场景，切分数计算公式待给出
                result = "PASS"
        show_list = show_list + [avg_hccl, avg_matmul, mask_rate, result]
    csv_name = f'{case["op_name"]}_performance_result.csv'
    if os.path.exists(csv_name):
        write_csv(csv_name, show_list)
    else:
        write_csv(csv_name, title)
        write_csv(csv_name, show_list)


def remove_profiling_result():
    profiling_result = './Profiling_Result_'
    os.system(f'rm -rf {profiling_result}*')


def run_errcode():
    # 运行Linux命令，获取输出
    if "aarch64" in os.uname().machine:
        smi = os.path.join(os.getcwd(), "npu_smi")
    else:
        smi = os.path.join(os.getcwd(), "npu_smi_x86")
    cmd = "{} | tee  smi.log".format(smi)
    output = subprocess.check_output(cmd, shell=True).decode()
    print(output)
    # 按照tab分割输出，并获取第n个字符串
    n = 8
    lines = remove_even_lines(output)
    lines = lines.split('\n')[1:9]  # 获取第2行到第9行

    device_err = []
    content_err = []
    for line in lines:
        fields = line.split()
        if fields[n] == '0':
            device_err.append("pass")
        else:
            device_err.append("fail")
            content_err.append(fields[9])

    print(device_err)
    casename = ' '.join(sys.argv[0:])
    if "fail" in device_err:
        os.system("msnpureport -f")
        print("错误码捕捉到")
        status_content = "ERROR perrorcode"
    else:
        print("没有报错日志")
        status_content = "PASS"
    print(casename)
    write_csv("errcode.csv", [str(casename), status_content, content_err])


def hcom_quant(mm_output, comm_quant_scale_1, comm_quant_scale_2, diff_tensor, all_to_all_out, all_gather_out,
               world_size, is_alltoall, rank=0):
    # 获取原始shape, 若mm_output是三维则reshape成2维进行后续操作
    original_shape = mm_output.shape
    N = mm_output.shape[-1]
    if len(mm_output.shape) == 3:
        M = mm_output.shape[0] * mm_output.shape[1]
        mm_output = mm_output.reshape(-1, mm_output.shape[-1])
    else:
        M = mm_output.shape[0]

    # step1:quant操作 mm_output / comm_quant_scale_1
    output = (mm_output / comm_quant_scale_1)

    # step2：拼接脏数据使得tensor可被all_to_all整除
    new_output = torch.concat([output, diff_tensor], dim=0)

    # step3: 拼接完的tensor reshape成3维：[world_size, -1, N]
    quant_1_output = new_output.reshape(world_size, -1, N)
    # 转int8
    quant_1_int8_output = torch.clamp(torch.round(quant_1_output), -128, 127).to(torch.int8)

    if is_alltoall:
        # step4: 调用all_to_all接口, all_to_all结果转fp32
        # print(f"befor alltoall ,rank:{rank},int8 shape:{output.shape},int8 output :{output}")
        # ------------ 规避A5上 alltoallsingle输入多维场景动态图失败问题 ---------------
        all_to_all_out_shape = all_to_all_out.shape
        all_to_all_out_tmp = all_to_all_out.reshape(all_to_all_out.numel())
        quant_1_int8_output_shape = quant_1_int8_output.shape
        quant_1_int8_output_tmp = quant_1_int8_output.reshape(quant_1_int8_output.numel())
        # ------------ 规避A5上 alltoallsingle输入多维场景动态图失败问题 ---------------
        dist.all_to_all_single(all_to_all_out_tmp, quant_1_int8_output_tmp)
        all_to_all_out = all_to_all_out_tmp.reshape(all_to_all_out_shape)
        # ------------ 规避A5上 alltoallsingle输入多维场景动态图失败问题 ---------------
        # print(f"rank:{rank},all to all output:{all_to_all_out}")

        # step5: all_to_all结果做reduce sum,shape：[1,-1,N]
        sum_out = torch.sum(all_to_all_out, dim=0, keepdim=True).to(torch.float32)
    else:
        sum_out = torch.sum(quant_1_int8_output, dim=0, keepdim=True)
    # step6: sum结果做comm_quant_1反量化
    sum_dequant_1_out = sum_out * comm_quant_scale_1
    # step7: sum后对1反量化的结果,做comm_quant_2量化，并转int8送进allgather
    quant_2_out = sum_dequant_1_out / comm_quant_scale_2
    quant_2_int8_out = torch.clamp(torch.round(quant_2_out), -128, 127).to(torch.int8)
    dist._all_gather_base(all_gather_out, quant_2_int8_out)
    # step 8:all_gather结果做comm_quant_2反量化
    dequant_2_out = (all_gather_out * comm_quant_scale_2)
    # step 9:最终结果删除脏数据，并转回3维
    output = dequant_2_out.reshape(-1, dequant_2_out.shape[-1])[0:M, 0:N]
    final_output = output.reshape(original_shape)
    # print(f"rank:{rank},final_output:{final_output}")

    return final_output


def hcom_quant_cpu(case, mm_output, comm_quant_scale_1, comm_quant_scale_2, world_size, is_alltoall, save_path=None, ev=None,
                   comm_quant_list=[], rank=0, out_dtype='fp16'):
    output_dtype = torch.float16 if out_dtype == 'fp16' else torch.bfloat16
    # 获取原始shape, 若mm_output是三维则reshape成2维进行后续操作
    original_shape = mm_output.shape
    N = mm_output.shape[-1]
    if len(mm_output.shape) == 3:
        M = mm_output.shape[0] * mm_output.shape[1]
        mm_output = mm_output.reshape(-1, mm_output.shape[-1])
    else:
        M = mm_output.shape[0]

    # ------------------------------------------#
    if comm_quant_scale_1 is None or comm_quant_scale_2 is None:
        # 获取comm_quant_scale_1
        abs_tensor = torch.abs(mm_output)
        comm_quant_scale_1_tmp = torch.div(torch.max(abs_tensor, 0)[0], 127).to(torch.float32)
        # 若输入为全0,除0操作置1
        comm_quant_scale_1_rank = torch.ones_like(comm_quant_scale_1_tmp) if torch.all(
            comm_quant_scale_1_tmp == 0) else comm_quant_scale_1_tmp
        print(f"rank:{rank},当前卡计算出系数1:{comm_quant_scale_1_rank}")
        comm_quant_list.append(comm_quant_scale_1_rank)
        comm_quant_scale_1 = torch.zeros([world_size, N], dtype=torch.float32)

        if len(comm_quant_list) != world_size:
            print(f"rank:{rank},未获取所有comm_quant, 开始等待")
            ev.wait()

        if len(comm_quant_list) == world_size:
            #       print(f"rank:{rank},event set ")
            for i in range(0, world_size):
                comm_quant_scale_1[i, :] = comm_quant_list[i]
            comm_quant_scale_1 = comm_quant_scale_1.max(axis=0)[0]
            print(f"rank:{rank},final_tensor:{comm_quant_scale_1}")
            print(f"rank:{rank},已获取所有rank的量化系数，阻塞解除")
            print(f"comm_quant_scale 1 计算完成, 存bin到本地")
            ev.set()
        save_file(save_path, comm_quant_scale_1, 'bf16', "comm_quant_scale_1", "fp32", 0)
        save_file(save_path, comm_quant_scale_1.to(output_dtype), output_dtype, "comm_quant_scale_1", "None", 0)
        ev.clear()  # set标志位重新置false
    # -----------------------------------------#

    # step1:quant操作 mm_output / comm_quant_scale_1
    output = (mm_output / comm_quant_scale_1)

    # step2：计算需padding的tensor，并concat拼接成可被all_to_all整除的tensor
    diff_tensor_shape = [math.ceil(M / world_size) * world_size - M, N]
    diff_tensor = torch.zeros(diff_tensor_shape)
    if output.device.type != "cpu":
        diff_tensor = diff_tensor.npu()
    new_output = torch.concat([output, diff_tensor], dim=0)

    # step3: 拼接完的tensor reshape成3维：[world_size, -1, N]
    quant_1_output = new_output.reshape(world_size, -1, N)

    # 转int8
    quant_1_int8_output = torch.clamp(torch.round(quant_1_output), -128, 127).to(torch.int8)

    if is_alltoall:
        # step4: 调用all_to_all接口, all_to_all结果转fp32
        # print(f"befor alltoall ,rank:{rank},int8 shape:{output.shape},int8 output :{output}")
        all_to_all_out = torch.zeros_like(quant_1_int8_output)
        dist.all_to_all_single(all_to_all_out, quant_1_int8_output)
        # print(f"rank:{rank},all to all output:{all_to_all_out}")

        # step5: all_to_all结果做reduce sum,shape：[1,-1,N]
        sum_out = torch.sum(all_to_all_out, dim=0, keepdim=True).to(torch.float32)
    else:
        sum_out = torch.sum(quant_1_int8_output, dim=0, keepdim=True)
    # step6: sum结果做comm_quant_1反量化
    sum_dequant_1_out = sum_out * comm_quant_scale_1

    # -----------------------------------------#
    # 获取comm_quant_scale_2
    if comm_quant_scale_1 is None or comm_quant_scale_2 is None:
        abs_mul_out = torch.abs(sum_dequant_1_out.reshape(-1, sum_dequant_1_out.shape[-1]))
        comm_quant_scale_2_tmp = torch.div(torch.max(abs_mul_out, 0)[0], 127).to(torch.float32)
        # 若输入为全0,除0操作置1
        comm_quant_scale_2_rank = torch.ones_like(comm_quant_scale_2_tmp) if torch.all(
            comm_quant_scale_2_tmp == 0) else comm_quant_scale_2_tmp

        comm_quant_list.append(comm_quant_scale_2_rank)
        comm_quant_scale_2 = torch.zeros([world_size, N], dtype=torch.float32)

        if len(comm_quant_list) != 2 * world_size:
            # print(f"rank:{rank},未获取所有comm_quant_2, 开始等待")
            ev.wait()

        if len(comm_quant_list) == 2 * world_size:
            print(f"rank:{rank},event 2 set ")
            for i in range(world_size, world_size * 2):
                comm_quant_scale_2[i - world_size, :] = comm_quant_list[i]
            comm_quant_scale_2 = comm_quant_scale_2.max(axis=0)[0]
            # print(f"rank:{rank},final_tensor 2:{comm_quant_scale_2}")
            # print(f"rank:{rank},2 已获取所有rank的量化系数，阻塞解除")
            print(f"comm_quant_scale 2 计算完成, 存bin到本地")
            ev.set()
        save_file(save_path, comm_quant_scale_2, 'bf16', "comm_quant_scale_2", "fp32", 0)
        if case['soc'] == "910D":
            save_file(f"bin_path/{case['case_name']}/input/{rank}", comm_quant_scale_1.to(output_dtype), output_dtype, "comm_quant_scale_1",
                      "None", 0)
        else:
            save_file(save_path, comm_quant_scale_2.to(output_dtype), output_dtype, "comm_quant_scale_2", "None", 0)
    #  -----------------------------------------#

    # step7: sum后对1反量化的结果,做comm_quant_2量化，并转int8送进allgather
    quant_2_out = sum_dequant_1_out / comm_quant_scale_2
    quant_2_int8_out = torch.clamp(torch.round(quant_2_out), -128, 127).to(torch.int8)

    all_gather_list = [torch.zeros_like(quant_2_int8_out) for _ in range(world_size)]
    dist.all_gather(all_gather_list, quant_2_int8_out)
    all_gather_out = torch.cat(all_gather_list, dim=0)
    # step 8:all_gather结果做comm_quant_2反量化，并转fp32
    dequant_2_out = (all_gather_out * comm_quant_scale_2).to(torch.float32)
    # step 9:最终结果删除脏数据，并转回3维
    output = dequant_2_out.reshape(-1, dequant_2_out.shape[-1])[0:M, 0:N]
    final_output = output.reshape(original_shape)
    # print(f"rank:{rank},final_output:{final_output}")

    return final_output


def permute_func(tokens, _e, _world_size, _rank, _device, _expTokenNums):
        indices = torch.zeros(_e, _world_size).long()
        for j in range(_e):
            for i in range(_world_size):
                indices[j][i] = _expTokenNums[i][j + (_e * _rank)]
        trans = indices.permute(1, 0)
        flaten = trans.reshape(-1).to(_device)
        sum_list = torch.cumsum(flaten, dim=0)
        tmp = []
        for i in range(len(sum_list)):
            if i == 0:
                tmp.append(range(0, sum_list[i]))
            else:
                tmp.append(range(sum_list[i - 1], sum_list[i]))
        out = []
        for e in range(_e):
            exp_token = []
            for r in range(_world_size):
                exp_token += list(tmp[e + r * _e])
            combined = torch.tensor(exp_token, dtype=torch.int64).to(_device)
            out.append(tokens.to(_device).index_select(0, combined))
        if _device.type == "npu":
            return torch.cat(out, dim=0).to(_device)
        else:
            return out


def unpermute_func(tokens, ranks, rank, world_size, expTokenNums, exp_per_card, device="cpu"):
    emptyArr = np.zeros((ranks, exp_per_card),dtype=np.int64)
    for i in range(world_size):
        tmp = expTokenNums[i][rank * exp_per_card:(rank + 1) * exp_per_card]
        emptyArr[i] = tmp
    tmp1 = emptyArr.transpose()
    sum_list1 = np.sum(tmp1, axis=1)
    sum_list2= np.cumsum(sum_list1,axis=0)
    offsets = sum_list2[0:-1].tolist()
    offsets = [0] + offsets
    sum_list = np.cumsum(tmp1, axis=1)
    indices_list = []
    for i in range(exp_per_card):
        tmp = []
        for j in range(ranks):
            if j == 0:
                tmp.append(list(map(lambda x: x + offsets[i],  list(range(0, sum_list[i][j])))))
            else:
                tmp.append(list(map(lambda x: x + offsets[i], list(range(sum_list[i][j - 1], sum_list[i][j])))))
        indices_list.append(tmp)
    selected = []
    for i in range(ranks):
        for j in range(exp_per_card):
            indices = torch.tensor(indices_list[j][i], dtype=torch.long).to(device)
            selected_rows = tokens.to(device).index_select(dim=0, index=indices)
            selected.append(selected_rows)
    return torch.cat(selected, dim=0).to(tokens.dtype)


def cal_inf_nan(rank, input, type='npu'):
    # inf/nan计数
    value_nan_num = np.isnan(input).sum()
    value_inf_num = (np.isinf(input) & (input > 0)).sum()
    value_ninf_num = (np.isinf(input) & (input < 0)).sum()
    print(f"[INFO] device_{rank} {type} nan:{value_nan_num} inf:{value_inf_num} -inf:{value_ninf_num}")


def get_act_type(act_type):
    if not act_type:
        act_int = 0
    elif act_type.upper() == "GELU":
        act_int = 1
    elif act_type.upper() == 'SILU':
        act_int = 2
    elif act_type.upper() == 'RELU':
        act_int = 3
    elif act_type.upper() == 'FASTGELU':
        act_int = 4
    else:
        print("act_type_error")
    return act_int


def get_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H:%M:%S.%f")[:-3]
    return formatted_time


def scale_generate(fp32_deq_scale):
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32)
    # 与高19位运算，模拟硬件
    uint32_deq_scale &= 0XFFFFE000
    fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)

    return fp32_deq_scale


def fetch_batch_broadcast(batch_x1, batch_x2):
    batch_out = copy.deepcopy(batch_x1) if len(batch_x1) > len(batch_x2) else copy.deepcopy(batch_x2)

    # 当x1和x2 batch维度不相同时且均不为0时，需要根据x1和x2中batch情况进行做broadcast 处理
    # 前面的batch_out只是获取到最大维度的batch，但存在列外，如[2,2,2,2,1,1]和[4,4,4,1,1], 实际batchout应为[2,4,4,4]
    min_len, max_len = 0, 0
    if batch_x2 != batch_x1 and batch_x1 and batch_x2:
        min_len = min(len(batch_x1), len(batch_x2))
        max_len = max(len(batch_x1), len(batch_x2))

    # 更新 batch_out 的前 min_len 个元素
    for idx in range(min_len):
        batch_out[-(idx + 1)] = max(batch_x1[-(idx + 1)], batch_x2[-(idx + 1)])

    # 如果 batch_x1 更长，更新剩余部分
    if len(batch_x1) > len(batch_x2):
        for idx in range(min_len, max_len):
            batch_out[-(idx + 1)] = batch_x1[-(idx + 1)]
    # 如果 batch_x2 更长，更新剩余部分
    else:
        for idx in range(min_len, max_len):
            batch_out[-(idx + 1)] = batch_x2[-(idx + 1)]
    return batch_out


def unpack_group_size(group_size):
    if group_size == -1:
        return 0, 0, 0
    group_size_m = (group_size >> 32) & 0xFFFF
    group_size_n = (group_size >> 16) & 0xFFFF
    group_size_k = group_size & 0xFFFF
    return group_size_m, group_size_n, group_size_k


def value_batch_broadcast(x, batch):
    new_x_shape = batch + list(x.shape[-2:])
    x = torch.broadcast_to(x, new_x_shape)
    return x


def per_block_cpu_compute(group_size, x1, x2, x1_scale, x2_scale):
    if x1.dim() != x1_scale.dim():
        print(f"[ERROR] x1.dim() != x1_scale.dim(),x1.dim()={x1.dim()}, x1_scale.dim()={x1_scale.dim()}")
    if x2.dim() != x2_scale.dim():
        print(f"[ERROR] x2.dim() != x2_scale.dim(),x2.dim()={x2.dim()}, x2_scale.dim()={x2_scale.dim()}")
    batch_x1 = np.array(x1.shape[:-2]).astype(int).tolist()
    batch_x2 = np.array(x2.shape[:-2]).astype(int).tolist()
    # 获取x1和x2broadcast的batch维度
    batch_out = fetch_batch_broadcast(batch_x1, batch_x2)
    # 基于batch_out的对x1和x1_scale 进行broadcast，如果维度不相等
    if batch_x1 != batch_out:
        x1 = value_batch_broadcast(x1, batch_out)
        x1_scale = value_batch_broadcast(x1_scale, batch_out)
    # 基于batch_out的对x2和x2_scale 进行broadcast，如果维度不相等
    if batch_x2 != batch_out:
        x2 = value_batch_broadcast(x2, batch_out)
        x2_scale = value_batch_broadcast(x2_scale, batch_out)

    batch_all = 1
    is2dim = True
    if batch_out != []:
        is2dim = False
        batch_all = np.prod(batch_out)
        x1 = x1.reshape([batch_all] + list(x1.shape[-2:]))
        x2 = x2.reshape([batch_all] + list(x2.shape[-2:]))
        x1_scale = x1_scale.reshape([batch_all] + list(x1_scale.shape[-2:]))
        x2_scale = x2_scale.reshape([batch_all] + list(x2_scale.shape[-2:]))
    m = x1.shape[-2]  # 非转置情况下，m是倒数第二维
    k = x1.shape[-1]  # 非转置情况下，k是倒数第一维
    n = x2.shape[-1]  # 非转置情况下，n是倒数第一维
    out = torch.zeros(m, n)
    if x2_scale.dim() > 2 or x1_scale.dim() > 2:
        out = torch.zeros(batch_all, m, n)
    group_size_m, group_size_n, group_size_k = unpack_group_size(group_size)
    for i in range(batch_all):
        for m_idx in range((m + group_size_m - 1) // group_size_m):
            m_start = m_idx * group_size_m
            m_end = min((m_idx + 1) * group_size_m, m)
            for n_idx in range((n + group_size_n - 1) // group_size_n):
                n_start = n_idx * group_size_n
                n_end = min((n_idx + 1) * group_size_n, n)
                for k_idx in range((k + group_size_k - 1) // group_size_k):
                    k_start = k_idx * group_size_k
                    k_end = min((k_idx + 1) * group_size_k, k)
                    if batch_all == 1 and is2dim:
                        block_output = torch.matmul(x1[m_start:m_end, k_start:k_end],
                                                    x2[k_start:k_end, n_start:n_end]) * x1_scale[m_idx, k_idx] * x2_scale[k_idx, n_idx]
                        out[m_start:m_end, n_start:n_end] += block_output
                    else:
                        out[i, m_start:m_end, n_start:n_end] += torch.matmul(x1[i, m_start:m_end, k_start:k_end],
                                                                             x2[i, k_start:k_end, n_start:n_end]) * x1_scale[
                                                                    i, m_idx, k_idx] * x2_scale[i, k_idx, n_idx]
    if x2_scale.dim() > 2 or x1_scale.dim() > 2:
        out_shape = batch_out
        out_shape.append(m)
        out_shape.append(n)
        out = torch.reshape(out, out_shape)
    else:
        out = torch.reshape(out, [m, n])
    return out


def save_file_v2(folder_path, np_array, tensor_name, rank):
    try:
        if not os.path.exists(folder_path):
            print("Make dir:", folder_path)
            os.makedirs(folder_path)
    except OSError:
        pass
    np_array.tofile('{}/{}.bin'.format(folder_path, tensor_name))


def generate_matrix(m, n, sum_row, sum_col, seed=1, max_iter=10000):
    # 验证行和与列和是否兼容
    if m * sum_row != n * sum_col:
        raise ValueError("m * sum_row 必须等于 n * sum_col")
    np.random.seed(seed)

    # 生成初始矩阵（每行和为sum_row，元素≥1）
    matrix = np.random.multinomial(sum_row - n, [1 / n] * n, size=m) + 1

    # 计算当前列和
    current_col_sums = matrix.sum(axis=0)
    target_col_sums = np.full(n, sum_col)

    # 迭代调整列和
    for _ in range(max_iter):
        if np.array_equal(current_col_sums, target_col_sums):
            return matrix

        # 找到超出和不足最多的列
        j = np.argmax(current_col_sums - target_col_sums)  # 超出最多的列
        k = np.argmin(current_col_sums - target_col_sums)  # 不足最多的列

        # 随机遍历行寻找可调整的项
        for i in np.random.permutation(m):  # 受种子控制
            if matrix[i, j] > 1:
                # 调整元素：j列减1，k列加1
                matrix[i, j] -= 1
                matrix[i, k] += 1
                current_col_sums[j] -= 1
                current_col_sums[k] += 1
                break
        else:
            continue  # 无法调整则跳过

    raise RuntimeError(f"未能在{max_iter}次迭代内找到解")

def generate_expTokenNums(exp_num, ep_world_size, bsk, seed, recv_count_zero_rank_list=[]):
    """
    生成 token 分配矩阵，支持指定某些 rank 的接收总数为 0。

    参数:
        exp_num: 专家总数
        ep_world_size: 卡数
        bsk: 每张卡发送的 token 数
        seed: 随机种子
        recv_count_zero_rank_list: list of int，接收总数为 0 的 rank 列表

    返回:
        (list of lists, torch.Tensor) 形状均为 (ep_world_size, exp_num)
    """
    np.random.seed(seed)
    total_tokens = ep_world_size * bsk

    if not recv_count_zero_rank_list:
        # 情况一：没有指定接收为0的rank
        if total_tokens % exp_num == 0:
            sum_row = total_tokens // exp_num  # 确保为整数
            # 原方法要求每个专家至少1个token，且每行和等于sum_row，每列和等于bsk
            # 若sum_row < ep_world_size，则无法保证每行每个元素≥1，此时应改用随机分配
            if sum_row >= ep_world_size:
                try:
                    ret = generate_matrix(exp_num, ep_world_size, sum_row, bsk, seed)
                    return [list(col) for col in zip(*ret)], torch.tensor(ret).permute(1, 0).to(torch.int64)
                except Exception as e:
                    # 若生成失败（如无解），回退到随机分配
                    pass
        # 不整除或无法满足原条件时，采用随机分配
        M = np.zeros((ep_world_size, exp_num), dtype=int)
        prob = np.ones(exp_num) / exp_num  # 均匀概率
        for i in range(ep_world_size):
            M[i, :] = np.random.multinomial(bsk, prob)
        return [list(row) for row in M], torch.tensor(M).to(torch.int64)
    else:
        # 情况二：有接收为0的rank
        if exp_num % ep_world_size != 0:
            raise ValueError("exp_num must be divisible by ep_world_size")
        exp_per_card = exp_num // ep_world_size

        zero_experts = set()
        for rank in recv_count_zero_rank_list:
            start = rank * exp_per_card
            end = (rank + 1) * exp_per_card
            zero_experts.update(range(start, end))

        other_experts = [j for j in range(exp_num) if j not in zero_experts]

        if not other_experts:
            if bsk == 0:
                M = np.zeros((ep_world_size, exp_num), dtype=int)
            else:
                raise ValueError("No experts left to receive tokens, but bsk > 0")
        else:
            M = np.zeros((ep_world_size, exp_num), dtype=int)
            prob = [1.0 / len(other_experts)] * len(other_experts)
            for i in range(ep_world_size):
                counts = np.random.multinomial(bsk, prob)
                M[i, other_experts] = counts

        return [list(row) for row in M], torch.tensor(M).to(torch.int64)

def generate_positive_integers(n, sum_total, seed):
    """
    生成n个能被n整除的正整数，且总和为指定值
    参数：
    n : 生成数字的数量（必须为正整数）
    sum_total : 目标总和（必须为正整数）
    seed : 控制随机性的字符串种子
    返回：
    np.ndarray: 生成的n个正整数数组
    约束条件：
    1. sum_total必须能被n整除
    2. sum_total必须 >= n²
    """
    # 输入验证
    if not isinstance(n, int) or n <= 0:
        raise ValueError("参数n必须是正整数")
    if not isinstance(sum_total, int) or sum_total <= 0:
        raise ValueError("参数sum_total必须是正整数")
    # 验证总和能被n整除
    if sum_total % n != 0:
        raise ValueError(f"总和{sum_total}必须能被n={n}整除")
    # 计算商并验证最小值
    quotient = sum_total // n
    if quotient < n:
        raise ValueError(f"总和必须满足sum_total >= n²，当前sum_total={sum_total}，n²={n ** 2}")
    # 初始化随机生成器（处理字符串种子）
    rng = np.random.default_rng(seed)

    # 生成n个非负整数，总和为quotient - n
    remain = quotient - n
    if remain == 0:
        return np.full(n, n, dtype=int)
    # 使用多项式分布生成均匀分布的随机数
    base_numbers = rng.multinomial(remain, [1 / n] * n)

    # 生成最终结果（每个数加1后乘n）
    return (base_numbers + 1) * n


def generate_gmm_alltoallv_matrix_normal(k_values, m, n, rng):
    blocks = []
    for k in k_values:
        # 创建 m x n 的块
        block = np.zeros((m, n), dtype=int)
        # 为每列生成正整数分布
        for col in range(n):
            # 生成m个非负整数，和为k - m，使用均匀分布
            counts = rng.multinomial(k - m, [1 / m] * m)
            block[:, col] = counts + 1  # 转换为正整数
        blocks.append(block)
    # 垂直堆叠所有块
    tmp = np.vstack(blocks)
    return [list(col) for col in zip(*tmp)], torch.tensor(tmp).permute(1, 0).to(torch.int64)


def generate_gmm_alltoallv_matrix_special(k_values, m, n, rng):
    blocks = []
    for k in k_values:
        # 创建 m x n 的块
        block = np.zeros((m, n), dtype=int)
        # 为每列生成正整数分布
        for col in range(n):
            # 生成m个非负整数，和为k，使用均匀分布
            counts = rng.multinomial(k, [1 / m] * m)
            block[:, col] = counts  # 转换为正整数
        blocks.append(block)
    # 垂直堆叠所有块
    tmp = np.vstack(blocks)
    return [list(col) for col in zip(*tmp)], torch.tensor(tmp).permute(1, 0).to(torch.int64)


def generate_gmm_alltoallv_matrix(A_array, m, recv_count_zero_rank_list=[], seed=1):
    """
    参数：
    A_array (list)  : 正整数数组，每个元素 ≥ m*len(A_array)
    m       (int)   : 每个元素对应的行块数
    seed    (str/int) : 控制随机性的种子（可选）
    返回：
    np.ndarray : 生成的矩阵，形状为 (m*len(A_array), len(A_array))
    """
    """
    生成满足条件的二维矩阵

    参数：
    A_array : list 或 np.ndarray，长度为n的正整数数组
    m : int，每组行数
    seed : int，控制随机生成的种子

    返回：
    np.ndarray: 生成的二维矩阵

    异常：
    ValueError: 当输入不满足约束条件时抛出
    """
    total = sum(A_array)
    k_values = []
    n = len(A_array)
    exp_num = n * m
    rng = np.random.default_rng(seed)

    flag = 0
    for idx, a in enumerate(A_array):
        if a % n != 0:
            flag = 2
            # raise ValueError(f"元素 A_array[{idx}]={a} 不能被 n={n} 整除")
        k = a // n
        if k < m:
            flag = 3
            # raise ValueError(f"A_array[{idx}]/n={k} 必须 >= m={m}")
        k_values.append(k)
        
    if recv_count_zero_rank_list:
        # 存在接收为0的rank
        if exp_num % n != 0:
            raise ValueError(f"exp_num must be divisible by ep_world_size")

        zero_experts = set()
        for rank in recv_count_zero_rank_list:
            start = rank * m
            end = (rank + 1) * m
            zero_experts.update(range(start, end))

        other_experts = [j for j in range(exp_num) if j not in zero_experts]

        if not other_experts:
            if total == 0:
                M = np.zeros((n, exp_num), dtype=int)
            else:
                raise ValueError("No experts left to receive tokens, but bsk > 0")
        else:
            M = np.zeros((n, exp_num), dtype=int)
            prob = [1.0 / len(other_experts)] * len(other_experts)
            for k in range(n):
                counts = rng.multinomial(A_array[k], prob)
                index = 0
                for i in range(n):
                    if i not in recv_count_zero_rank_list:
                        print(f"counts[index:(index+1)*m] is {counts[index*m:(index+1)*m]}")
                        M[i, k*m:(k+1)*m] = counts[index*m:(index+1)*m]
                        index = index + 1

        return [list(row) for row in M], torch.tensor(M).to(torch.int64)

    if total % n != 0:
        flag = 1
        # raise ValueError(f"sum(A_array)={total} 必须能被 n={n} 整除")

    # 验证每个元素能被n整除且满足最小值约束
    if flag == 0:
        return generate_gmm_alltoallv_matrix_normal(k_values, m, n, rng)
    else:
        return generate_gmm_alltoallv_matrix_special(k_values, m, n, rng)

def get_soc_version():
    rtsdll = ctypes.CDLL("libruntime.so")
    c_char_t = ctypes.create_string_buffer(b'\xff' * 256, 256)
    rtsdll.rtGetSocVersion(c_char_t, ctypes.c_uint32(256))
    soc_version = c_char_t.value.decode("utf-8")
    return soc_version
    
def init_csv(file):
    item=['testcase_name','bin_name', 'output_dtypes', 'stc_outputs', 'input_data_ranges']
    if os.path.exists(file):
        os.remove(file)
    with open(file, mode="a", encoding='utf-8-sig') as f:
        write = csv.writer(f)
        write.writerow(item)
def get_case_info(case_name,word_sie,bin_name,input_tensor_range):
    tmp=[0,0,0,0,0]
    tmp[0]=case_name
    tmp[1]=bin_name
    if "fp16" in case_name:
        tmp_1="'float16',"*int(word_sie)
    else:
        tmp_1 = (("'bfloat16'," * int(word_sie)))
    tmp_2=("(),"*int(word_sie))
    # print(tmp)
    tmp[2]="({})".format(tmp_1)
    tmp[3] = "({})".format(tmp_2)
    tmp[4]="({})".format(input_tensor_range)
    return tmp

def get_pids(proc_name):
    """通过进程名aaa获取所有匹配的PID"""
    pids = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'].lower() == proc_name.lower():
                pids.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids
