import argparse
import ast
import csv
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import os
import random
import shutil
import socket
import sys
import traceback
from queue import Empty

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process, Manager

from cann_ops_transformer import ElasticBuffer

try:
    import psutil
except ImportError:
    psutil = None

# fetch_mode 常量: 0=sliced(切片跨NPU通信), 1=local(全量本NPU)
FETCH_MODE_SLICED = 0
FETCH_MODE_LOCAL = 1
_FETCH_MODE_MAP = {
    "0": FETCH_MODE_SLICED, "sliced": FETCH_MODE_SLICED,
    "1": FETCH_MODE_LOCAL, "local": FETCH_MODE_LOCAL,
}

# ===================== 原始默认参数 =====================
rank_per_dev = 8  # 单机几卡
server_num = 1    # 机器数量

num_entries_total = 200
hidden_size = 7166
indices_type = np.int32        # indices数据类型
storage_type = torch.bfloat16  # storage数据类型
world_size = rank_per_dev * server_num  # 初始化
fetch_token_num = [[2, 3, 4], [1], [4, 2], [4]]  # 各卡fetch次数以及每次fetch的token数

seed = 1             # 随机数种子
is_aclgraph = False  # 是否开启aclgraph
is_profiling = False # 是否开启profiling
use_fp8 = False      # 是否使用FP8量化（带SF表），默认按表格use_fp8列控制
sf_dtype = torch.float32  # SF表数据类型，支持float32和float8_e8m0fnu

# 初始化为int类型0
num_cpu_bytes = 0

is_cloud = 0  # 是否在云上环境进行测试，0为物理机，1为云上环境
if not is_cloud:
    server_index = 0 if server_num > 1 else 0  # 多机每个脚本保持一致并递增index值
    MASTER_IP = '141.61.94.223' if server_num > 1 else '127.0.0.1'  # 多机时设置为master节点的IP
else:
    server_index = int(os.environ.get("VC_TASK_INDEX"))
    host_url_list = os.environ.get("VC_WORKER_HOSTS").split(",")
    MASTER_IP = socket.gethostbyname(host_url_list[0])


def safe_int(val):
    try:
        return int(float(val))
    except Exception:
        raise argparse.ArgumentTypeError("必须传入0或1的整数，不支持其他格式")


# ===================== 命令行参数解析函数 =====================
def parse_cli_args(num_entries_total_global, hidden_size_global,
                   storage_type_global, fetch_token_num_global, world_size_global,
                   num_cpu_bytes_global, is_aclgraph_global, is_profiling_global, use_fp8_global, seed_global):
    parser = argparse.ArgumentParser(description="Egram ElasticBuffer Test")
    parser.add_argument("--case_name", type=str, help="当前测试用例名称")
    parser.add_argument("--input_tensor_shape", type=str, help="输入张量shape，字符串形式如'[[256,128]]'")
    parser.add_argument("--shape_semantics", type=str, default="total", choices=["total", "per_rank"],
                        help="input_tensor_shape[0] 的语义: total=全局vocab总行数(默认); "
                             "per_rank=每rank行数, 全局总行数=n*world_size(旧行为, 兼容旧表)")
    parser.add_argument("--input_tensor_dtype", type=str, help="数据类型，字符串形式如'[\"bf16\"]'")
    parser.add_argument("--input_tensor_range", type=str, help="数值范围，字符串形式如'[(-5,5)]'")
    parser.add_argument("--fetch_indicies", type=str, help="各rank每次fetch的token列表，字符串形式")
    parser.add_argument("--world_size", type=int, help="分布式总进程数")
    parser.add_argument("--num_cpu_bytes", type=int, help="CPU侧预留字节数，非负整数，0则自动计算")
    parser.add_argument("--is_graph", type=safe_int, help="是否开启aclgraph，仅支持0或1")
    parser.add_argument("--is_profiling", type=safe_int, help="是否开启性能profiling，仅支持0或1")
    parser.add_argument("--use_fp8", type=safe_int, help="是否使用FP8量化（带SF表），仅支持0或1")
    parser.add_argument("--sf_dtype", type=str, help="SF表数据类型：float32或e8m0")
    parser.add_argument("--seed", type=int, help="随机数种子，非负整数")
    parser.add_argument("--fetch_mode", type=str, default="sliced",
                        help="engram fetch模式: sliced(0)=切片跨NPU通信(默认), local(1)=全量本NPU")
    parser.add_argument("--prep_only", action="store_true", help="仅准备数据并保存到文件，不执行NPU测试")
    parser.add_argument("--run_only", action="store_true", help="从文件加载预准备数据，仅执行NPU测试")
    parser.add_argument("--start_method", type=str, default="fork", choices=["fork", "forkserver"],
                        help="子进程启动方式，默认fork(要求父进程不初始化NPU)")
    parser.add_argument("--master_port", type=int, help="HCCL rendezvous端口，默认取环境变量MASTER_PORT或50008")
    parser.add_argument("--loop_time", type=int, help="性能采集轮数，默认50")
    parser.add_argument("--repeat_time", type=int, help="性能采集分批数，默认1")
    parser.add_argument("--warmup_time", type=int, help="每批次热身轮数，默认0")
    parser.add_argument("--cpu_map", type=str, default="",
                        help='CPU亲和性映射，格式"0-3:288-383;4-7:96-191"，默认空=关闭')
    parser.add_argument("--verify_write", action="store_true",
                        help="engram_write后读取host窗口指针与write golden比对(默认关闭)")

    args = parser.parse_args()

    num_entries_total = num_entries_total_global
    hidden_size = hidden_size_global
    storage_type = storage_type_global
    val_min, val_max, case_name = None, None, None
    fetch_token_num = fetch_token_num_global
    world_size = world_size_global
    num_cpu_bytes = num_cpu_bytes_global
    is_aclgraph = is_aclgraph_global
    is_profiling = is_profiling_global
    use_fp8 = False
    sf_dtype_out = torch.float32
    seed = seed_global

    # 0. world_size 先解析（per_rank 语义的 total = n * world_size 依赖它）
    if args.world_size is not None:
        if args.world_size <= 0:
            raise ValueError("world_size 必须是正整数")
        world_size = args.world_size

    # 1. 解析 input_tensor_shape 并按 shape_semantics 解释
    if args.input_tensor_shape is not None:
        shape_list = ast.literal_eval(args.input_tensor_shape)
        n, h = shape_list[0][0], shape_list[0][1]
        if args.shape_semantics == "per_rank":
            # 旧行为（兼容旧表）: shape[0] 为每 rank 行数, 全局总行数 = n * world_size
            num_entries_total = n * world_size
        else:
            # 默认语义: shape[0] 即全局 vocab 总行数
            num_entries_total = n
        hidden_size = h

    # 2. dtype 映射
    if args.input_tensor_dtype is not None:
        dtype_str = ast.literal_eval(args.input_tensor_dtype)[0]
        dtype_map = {
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "int8": torch.int8,
            "int32": torch.int32,
            "int64": torch.int64,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
            "float8_e5m2": torch.float8_e5m2,
        }
        storage_type = dtype_map[dtype_str]

    # 3. 数值范围转 float
    if args.input_tensor_range is not None:
        range_list = ast.literal_eval(args.input_tensor_range)
        raw_min, raw_max = range_list[0][0], range_list[0][1]
        val_min = float(raw_min)
        val_max = float(raw_max)

    # 4. fetch_indicies 所有元素强制转为 int
    if args.fetch_indicies is not None:
        raw_fetch = ast.literal_eval(args.fetch_indicies)
        def to_int_list(lst):
            res = []
            for item in lst:
                if isinstance(item, list):
                    res.append(to_int_list(item))
                elif isinstance(item, int):
                    res.append(item)
                else:
                    raise ValueError("fetch_token_num 内所有元素必须为整数")
            return res
        fetch_token_num = to_int_list(raw_fetch)

    # 4.1 fetch_token_num 广播: 当提供的 rank 数少于 world_size 时，循环填充
    if args.world_size is not None and len(fetch_token_num) < args.world_size:
        ws = args.world_size
        orig_len = len(fetch_token_num)
        fetch_token_num = [fetch_token_num[i % orig_len] for i in range(ws)]
        print(f"[INFO] fetch_token_num 广播: 原始 {orig_len} 个 rank 的 fetch 配置已循环填充到 {ws} 个 rank")

    # 5. num_cpu_bytes 非负int校验
    if args.num_cpu_bytes is not None:
        num_cpu_bytes = args.num_cpu_bytes
        if num_cpu_bytes < 0:
            raise ValueError("num_cpu_bytes 必须为大于等于0的非负整数")

    # 6. case_name 赋值
    if args.case_name is not None:
        case_name = args.case_name

    # is_aclgraph：仅允许 0/1，转为布尔值
    if args.is_graph is not None:
        if args.is_graph not in (0, 1):
            raise ValueError("is_graph 只能传入 0 或 1")
        is_aclgraph = bool(args.is_graph)

    # is_profiling：仅允许 0/1，转为布尔值
    if args.is_profiling is not None:
        if args.is_profiling not in (0, 1):
            raise ValueError("is_profiling 只能传入 0 或 1")
        is_profiling = bool(args.is_profiling)

    # use_fp8：仅允许 0/1，转为布尔值
    if args.use_fp8 is not None:
        if args.use_fp8 not in (0, 1):
            raise ValueError("use_fp8 只能传入 0 或 1")
        use_fp8 = bool(args.use_fp8)

    # sf_dtype：SF表数据类型
    if args.sf_dtype is not None:
        sf_dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "e8m0": torch.float8_e8m0fnu,
            "float8_e8m0fnu": torch.float8_e8m0fnu,
        }
        if args.sf_dtype not in sf_dtype_map:
            raise ValueError(f"sf_dtype 只支持 {list(sf_dtype_map.keys())}")
        sf_dtype_out = sf_dtype_map[args.sf_dtype]

    # seed：非负整数校验
    if args.seed is not None:
        if not isinstance(args.seed, int):
            raise TypeError("seed 必须传入整数类型")
        if args.seed < 0:
            raise ValueError("seed 必须为大于等于0的非负整数")
        seed = args.seed

    # fetch_mode 解析
    fetch_mode = FETCH_MODE_SLICED
    if args.fetch_mode is not None:
        key = str(args.fetch_mode).strip().lower()
        if key not in _FETCH_MODE_MAP:
            raise ValueError(f"fetch_mode 只支持 {list(_FETCH_MODE_MAP.keys())}, got: {args.fetch_mode}")
        fetch_mode = _FETCH_MODE_MAP[key]

    # start_method：子进程启动方式
    start_method = args.start_method

    # master_port：CLI 优先，其次环境变量 MASTER_PORT，最后默认 50008
    if args.master_port is not None:
        if args.master_port <= 0 or args.master_port > 65535:
            raise ValueError("master_port 必须在 1~65535 范围内")
        master_port = args.master_port
    else:
        master_port = int(os.environ.get("MASTER_PORT", "50008"))

    # loop_time/repeat_time/warmup_time
    loop_time = 50 if args.loop_time is None else args.loop_time
    repeat_time = 1 if args.repeat_time is None else args.repeat_time
    warmup_time = 0 if args.warmup_time is None else args.warmup_time
    if loop_time < 1:
        raise ValueError("loop_time 必须是正整数")
    if repeat_time < 1:
        raise ValueError("repeat_time 必须是正整数")
    if warmup_time < 0:
        raise ValueError("warmup_time 必须是非负整数")

    # cpu_map / verify_write
    cpu_map = (args.cpu_map or "").strip()
    verify_write = bool(args.verify_write)

    return (num_entries_total, hidden_size, storage_type, val_min, val_max, fetch_token_num,
            world_size, num_cpu_bytes, case_name, is_aclgraph, is_profiling, use_fp8, sf_dtype_out,
            seed, fetch_mode, args.prep_only, args.run_only, args.shape_semantics, start_method,
            master_port, loop_time, repeat_time, warmup_time, cpu_map, verify_write)


def set_global_seed(seed):
    print(f"[INFO] seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fill_randn_worker(args):
    """worker: write randn directly into shared memory"""
    shm_name, offset, count, hidden, seed = args
    import numpy as _np
    _np.random.seed(seed)
    existing_shm = shm.SharedMemory(name=shm_name)
    try:
        arr = _np.ndarray((count, hidden), dtype=_np.float32, buffer=existing_shm.buf, offset=offset)
        arr[:] = _np.random.standard_normal((count, hidden)).astype(_np.float32)
    finally:
        existing_shm.close()


def _fill_randint_worker(args):
    """worker: write randint directly into shared memory"""
    shm_name, offset, count, hidden, low, high, itemsize, np_dtype_str, seed = args
    import numpy as _np
    _np.random.seed(seed)
    existing_shm = shm.SharedMemory(name=shm_name)
    try:
        arr = _np.ndarray((count, hidden), dtype=np_dtype_str, buffer=existing_shm.buf, offset=offset)
        arr[:] = _np.random.randint(low, high, (count, hidden), dtype=np_dtype_str)
    finally:
        existing_shm.close()


def _fill_rand_worker(args):
    """worker: write rand (uniform) directly into shared memory"""
    shm_name, offset, count, hidden, seed = args
    import numpy as _np
    _np.random.seed(seed)
    existing_shm = shm.SharedMemory(name=shm_name)
    try:
        arr = _np.ndarray((count, hidden), dtype=_np.float32, buffer=existing_shm.buf, offset=offset)
        arr[:] = _np.random.random((count, hidden)).astype(_np.float32)
    finally:
        existing_shm.close()


def _parallel_randn(total, hidden, dtype=torch.float32, num_workers=64):
    """Parallel randn via shared memory - no IPC copy"""
    if total <= 0:
        return torch.randn(0, hidden, dtype=dtype)
    # Small data: skip parallel overhead
    if total * hidden < 10_000_000:
        return torch.randn(total, hidden, dtype=dtype)
    num_workers = min(num_workers, max(1, total // 1000))
    size_bytes = total * hidden * 4
    shared = shm.SharedMemory(create=True, size=size_bytes)
    try:
        chunk_size = (total + num_workers - 1) // num_workers
        args = []
        for i in range(num_workers):
            start = i * chunk_size
            if start >= total:
                break
            end = min((i + 1) * chunk_size, total)
            count = end - start
            offset = start * hidden * 4
            args.append((shared.name, offset, count, hidden, i + 1))
        with mp.Pool(len(args)) as pool:
            pool.map(_fill_randn_worker, args)
        result = torch.frombuffer(shared.buf, dtype=torch.float32, count=total*hidden).reshape(total, hidden).clone()
    finally:
        shared.close()
        shared.unlink()
    return result.to(dtype) if dtype != torch.float32 else result


def _parallel_rand(total, hidden, dtype=torch.float32, num_workers=64):
    """Parallel rand (uniform) via shared memory - no IPC copy"""
    if total <= 0:
        return torch.rand(0, hidden, dtype=dtype)
    # Small data: skip parallel overhead
    if total * hidden < 10_000_000:
        return torch.rand(total, hidden, dtype=dtype)
    num_workers = min(num_workers, max(1, total // 1000))
    size_bytes = total * hidden * 4
    shared = shm.SharedMemory(create=True, size=size_bytes)
    try:
        chunk_size = (total + num_workers - 1) // num_workers
        args = []
        for i in range(num_workers):
            start = i * chunk_size
            if start >= total:
                break
            end = min((i + 1) * chunk_size, total)
            count = end - start
            offset = start * hidden * 4
            args.append((shared.name, offset, count, hidden, i + 1))
        with mp.Pool(len(args)) as pool:
            pool.map(_fill_rand_worker, args)
        result = torch.frombuffer(shared.buf, dtype=torch.float32, count=total*hidden).reshape(total, hidden).clone()
    finally:
        shared.close()
        shared.unlink()
    return result.to(dtype) if dtype != torch.float32 else result


def _parallel_randint(total, hidden, low, high, dtype=torch.int32, num_workers=64):
    """Parallel randint via shared memory - no IPC copy"""
    if total <= 0:
        return torch.randint(low, high, (0, hidden), dtype=dtype)
    # Small data: skip parallel overhead
    if total * hidden < 10_000_000:
        return torch.randint(low, high, (total, hidden), dtype=dtype)
    np_dtype_map = {
        torch.int8: 'int8', torch.int16: 'int16', torch.int32: 'int32',
        torch.int64: 'int64', torch.uint8: 'uint8',
    }
    np_dtype_str = np_dtype_map.get(dtype, 'int32')
    itemsize = torch.tensor([], dtype=dtype).element_size()
    num_workers = min(num_workers, max(1, total // 1000))
    size_bytes = total * hidden * itemsize
    shared = shm.SharedMemory(create=True, size=size_bytes)
    try:
        chunk_size = (total + num_workers - 1) // num_workers
        args = []
        for i in range(num_workers):
            start = i * chunk_size
            if start >= total:
                break
            end = min((i + 1) * chunk_size, total)
            count = end - start
            offset = start * hidden * itemsize
            args.append((shared.name, offset, count, hidden, low, high, itemsize, np_dtype_str, i + 1))
        with mp.Pool(len(args)) as pool:
            pool.map(_fill_randint_worker, args)
        result = torch.frombuffer(shared.buf, dtype=dtype, count=total*hidden).reshape(total, hidden).clone()
    finally:
        shared.close()
        shared.unlink()
    return result


def gen_storage_write_golden(num_entries_total, hidden, world_size, storage_type, val_min=None, val_max=None, use_fp8=False, sf_dtype=torch.float32):
    import time as _time
    total = num_entries_total
    num_workers = min((os.cpu_count() or 1) // 2, 40)
    t0 = _time.time()
    print(f"[INFO] 并行数据生成(共享内存): {num_workers} workers, shape=[{total}, {hidden}]")
    if use_fp8:
        storage = _parallel_randn(total, hidden, dtype=torch.float32, num_workers=num_workers).to(storage_type)
        num_sf_packs = (hidden + 31) // 32
        sf_table_fp32 = _parallel_rand(total, num_sf_packs, dtype=torch.float32, num_workers=num_workers)
        if sf_dtype == torch.float8_e8m0fnu:
            bits = sf_table_fp32.view(torch.int32)
            biased_exp = ((bits >> 23) & 0xFF).to(torch.uint8)
            sf_table = biased_exp.view(torch.float8_e8m0fnu).contiguous()
        else:
            sf_table = sf_table_fp32.contiguous()
    elif val_min is not None and val_max is not None:
        if storage_type in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            storage = _parallel_randint(total, hidden, int(val_min), int(val_max) + 1, dtype=storage_type, num_workers=num_workers)
        elif storage_type in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu):
            storage = ((_parallel_rand(total, hidden, dtype=torch.float32, num_workers=num_workers) * (val_max - val_min)) + val_min).to(storage_type)
        else:
            storage = (_parallel_rand(total, hidden, dtype=storage_type, num_workers=num_workers) * (val_max - val_min)) + val_min
        sf_table = None
    else:
        if storage_type in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            storage = _parallel_randint(total, hidden, -128, 128, dtype=storage_type, num_workers=num_workers)
        elif storage_type in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu):
            storage = _parallel_randn(total, hidden, dtype=torch.float32, num_workers=num_workers).to(storage_type)
        else:
            storage = _parallel_randn(total, hidden, dtype=storage_type, num_workers=num_workers)
        sf_table = None

    write_golden = list(storage.chunk(world_size, dim=0))
    t1 = _time.time()
    print(f"[INFO] 数据生成完成，耗时: {t1-t0:.1f}s")
    return storage, write_golden, sf_table


def gen_balanced_indices(num_token, all_num_token, indices_type, world_size, rng):
    """按 world_size 分段均衡采样全局索引, 返回 np.ndarray(dtype=int32)。

    - world_size 与 rng 均为显式参数, 不读任何全局变量;
    - num_token > all_num_token 时不失败: 有放回重复采样(repeated sampling,
      prefill 长上下文/大 batch 场景), 由调用方打印 WARNING。
    """
    segment_size = all_num_token // world_size
    base_count = num_token // world_size
    remainder = num_token % world_size

    all_indices = []
    for i in range(world_size):
        start = i * segment_size
        end = start + segment_size if i < world_size - 1 else all_num_token
        count = base_count + (1 if i < remainder else 0)
        chunk = rng.integers(start, end, size=count)
        all_indices.append(chunk)

    indices = np.concatenate(all_indices)
    rng.shuffle(indices)
    return indices.astype(indices_type)


def gen_fetch_golden(indices_arr, storage, hidden_size):
    idx_tensor = torch.from_numpy(indices_arr.astype(np.int64))
    return torch.index_select(storage, 0, idx_tensor)


def gen_fetch_sf_golden(indices_arr, sf_table):
    if sf_table is None:
        return None
    idx_tensor = torch.from_numpy(indices_arr.astype(np.int64))
    is_e8m0 = sf_table.dtype == torch.float8_e8m0fnu
    sf_view = sf_table.view(torch.uint8) if is_e8m0 else sf_table
    result = sf_view[idx_tensor]
    return result.view(torch.float8_e8m0fnu) if is_e8m0 else result


def compare_fetch_sf(fetch_sf_result, fetch_sf_golden, rank, fetch_idx, max_print_rows=8):
    if fetch_sf_golden is None:
        return True
    if fetch_sf_result is None:
        print(f"[ERROR] rank {rank} fetch[{fetch_idx}] fetched_sf is None but golden exists")
        return False
    is_e8m0 = fetch_sf_result.dtype == torch.float8_e8m0fnu
    fetch_sf_result_cpu = fetch_sf_result.view(torch.uint8).cpu() if is_e8m0 else fetch_sf_result.cpu()
    golden_cmp = fetch_sf_golden.view(torch.uint8) if fetch_sf_golden.dtype == torch.float8_e8m0fnu else fetch_sf_golden
    print(f"[INFO] rank {rank} fetch[{fetch_idx}] fetched_sf shape: {fetch_sf_result_cpu.shape}, dtype: {fetch_sf_result.dtype}")
    if fetch_sf_result_cpu.shape != golden_cmp.shape:
        print(f"[ERROR] rank {rank} fetch[{fetch_idx}] fetched_sf shape mismatch: {fetch_sf_result_cpu.shape} vs {golden_cmp.shape}")
        return False
    if torch.equal(fetch_sf_result_cpu, golden_cmp):
        print(f"[SUCCESS] rank {rank} fetch[{fetch_idx}] fetched_sf: shape={fetch_sf_result.shape}, exactly equal to golden")
        return True
    mismatch_mask = fetch_sf_result_cpu != golden_cmp
    print(f"[ERROR] rank {rank} fetch[{fetch_idx}] fetched_sf mismatch, total mismatched elements: {mismatch_mask.sum().item()}")
    if fetch_sf_result_cpu.dim() > 1:
        mismatch_rows = torch.nonzero(mismatch_mask.any(dim=-1)).flatten()
    else:
        mismatch_rows = torch.nonzero(mismatch_mask).flatten()
    print(f"        Mismatched rows: {mismatch_rows.numel()}, 最多打印前 {max_print_rows} 行:")
    for i in mismatch_rows[:max_print_rows].tolist():
        print(f"        row[{i}] result={fetch_sf_result_cpu[i].tolist()}, golden={golden_cmp[i].tolist()}")
    return False


def compare_write(write_input, write_golden: torch.Tensor):
    import ctypes
    golden = write_golden.contiguous()
    num_elem = golden.numel()
    elem_bytes = golden.element_size()
    total_bytes = num_elem * elem_bytes

    if isinstance(write_input, torch.Tensor):
        write_tensor = write_input.contiguous().flatten()
    elif isinstance(write_input, int):
        buf = (ctypes.c_ubyte * total_bytes).from_address(write_input)
        write_tensor = torch.frombuffer(buf, dtype=golden.dtype, count=num_elem)
    else:
        raise TypeError("write_input only support tensor or host address int")

    golden_flat = golden.flatten()
    if write_tensor.numel() != golden_flat.numel():
        print(f"[ERROR] write count mismatch: {write_tensor.numel()} vs {golden_flat.numel()}")
        return False

    all_pass = True
    if not torch.equal(write_tensor, golden_flat):
        all_pass = False
        diff_mask = write_tensor != golden_flat
        err_idx = torch.nonzero(diff_mask).squeeze()
        if err_idx.dim() == 0:
            err_idx = err_idx.unsqueeze(0)
        for idx in err_idx:
            i = int(idx)
            print(f"[ERROR] write[{i}] mismatch, result={write_tensor[i].item()}, golden={golden_flat[i].item()}")

    if all_pass:
        print("[SUCCESS] compare_write: all write results match golden")
    return all_pass


def compare_fetch(fetch_result, fetch_golden, rank, fetch_idx, max_print_rows=8):
    fetch_result_cpu = fetch_result.cpu()
    print(f"[INFO] rank {rank} fetch[{fetch_idx}] fetch_result shape: {fetch_result_cpu.shape}, "
          f"dtype: {fetch_result.dtype}, golden dtype: {fetch_golden.dtype}")
    if fetch_result_cpu.shape != fetch_golden.shape:
        print(f"[ERROR] rank {rank} fetch[{fetch_idx}] shape mismatch: {fetch_result_cpu.shape} vs {fetch_golden.shape}")
        return False
    if torch.equal(fetch_result_cpu, fetch_golden):
        print(f"[SUCCESS] rank {rank} fetch[{fetch_idx}]: shape={fetch_result.shape}, exactly equal to golden")
        return True
    diff = (fetch_result_cpu.float() - fetch_golden.float()).abs()
    print(f"[ERROR] rank {rank} fetch[{fetch_idx}] mismatch, max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}")
    mismatch_mask = fetch_result_cpu != fetch_golden
    print(f"        Total mismatched elements: {mismatch_mask.sum().item()}")
    if fetch_result_cpu.dim() > 1:
        mismatch_rows = torch.nonzero(mismatch_mask.any(dim=-1)).flatten()
    else:
        mismatch_rows = torch.nonzero(mismatch_mask).flatten()
    print(f"        Mismatched rows: {mismatch_rows.numel()}, 最多打印前 {max_print_rows} 行:")
    for i in mismatch_rows[:max_print_rows].tolist():
        print(f"        row[{i}] result={fetch_result_cpu[i].float().tolist()}, golden={fetch_golden[i].float().tolist()}")
    return False


def _parse_cpu_range(seg):
    seg = seg.strip()
    if "-" in seg:
        lo, hi = seg.split("-", 1)
        lo, hi = int(lo), int(hi)
        if hi < lo:
            raise ValueError(f"CPU 区间非法: {seg}")
        return list(range(lo, hi + 1))
    return [int(seg)]


def set_cpu_affinity(rank, cpu_map):
    """按 --cpu_map 配置绑定子进程 CPU 亲和性, 格式: "0-3:288-383;4-7:96-191"。"""
    if not cpu_map:
        return
    if psutil is None:
        print("[WARNING] 未安装 psutil, 跳过 CPU 亲和性设置(pip install psutil 后生效)")
        return
    try:
        for pair in cpu_map.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            rank_seg, sep, cpu_seg = pair.partition(":")
            if not sep:
                raise ValueError(f"cpu_map 段缺少 ':': {pair}")
            if "-" in rank_seg:
                lo, hi = rank_seg.split("-", 1)
                rank_hit = int(lo) <= rank <= int(hi)
            else:
                rank_hit = (rank == int(rank_seg))
            if rank_hit:
                cpu_ids = _parse_cpu_range(cpu_seg)
                psutil.Process().cpu_affinity(cpu_ids)
                print(f"[INFO] rank {rank} 已绑定 CPU 亲和性: {cpu_seg} ({len(cpu_ids)} CPUs)")
                return
        print(f"[WARNING] rank {rank} 未在 cpu_map({cpu_map}) 中命中任何 rank 区间, 不做绑定")
    except Exception as e:
        print(f"[WARNING] rank {rank} CPU 亲和性设置失败, 跳过: {e}")


def set_device(rank, rank_per_dev):
    real_device = rank % rank_per_dev
    torch.npu.set_device(real_device)
    print(f"[INFO] rank: {rank}, device: {real_device}")


def init_hccl_comm(rank, world_size, master_ip, master_port):
    print(f"[INFO] device_{rank} 创建HCCL通信链路 (master={master_ip}:{master_port})")
    dist.init_process_group(
        backend='hccl',
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://{master_ip}:{master_port}'
    )
    print(f"[INFO] device_{rank} HCCL通信链路创建成功")
    global_group = dist.new_group(backend='hccl', ranks=list(range(world_size)))
    print(f"[INFO] device_{rank} 全局通信组创建成功")
    return global_group


def get_profiling(loop, repeat, warmup, path):
    active = loop // repeat - warmup

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=repeat, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    )

    return prof


class EngramFetchGraphModel(torch.nn.Module):
    def __init__(self, buffer, use_fp8=False):
        super().__init__()
        self.buffer = buffer
        self.use_fp8 = use_fp8

    def forward(self, indices):
        with torch.profiler.record_function("Engram"):
            a = torch.mul(indices, 1)
            wait_func = self.buffer.engram_fetch(a)
            result = wait_func()
            if self.use_fp8:
                fetched, fetched_sf = result
                return fetched, fetched_sf
            return result, None


def egram_cpu(rank, size_idx, num_token, indices_type, storage, sf_table, world_size, seed, warned_sizes=None):
    """生成一组确定性 indices 与对应的 CPU golden。

    - 索引种子: SeedSequence(entropy=seed, spawn_key=(rank, size_idx)), 同 seed 下
      local/sliced 配对用例生成完全相同的 indices(性能可直接对比), 且跨 run 可复现;
    - num_token > vocab 时不失败, 打印一次 WARNING 后继续(重复采样, prefill 场景)。
    """
    all_num_token, hidden_size = storage.shape
    if num_token > all_num_token and warned_sizes is not None and num_token not in warned_sizes:
        warned_sizes.add(num_token)
        print(f"[WARNING] rank {rank} fetch[{size_idx}]: num_token={num_token} > vocab={all_num_token}, "
              f"repeated sampling, prefill long-context/large-batch scenario")
    rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(rank, size_idx)))
    indices_arr = gen_balanced_indices(num_token, all_num_token, indices_type, world_size, rng)
    fetch_golden = gen_fetch_golden(indices_arr, storage, hidden_size)
    fetch_sf_golden = gen_fetch_sf_golden(indices_arr, sf_table)
    print(f"[INFO] rank: {rank} fetch[{size_idx}] num_token={num_token}, fetch_golden shape: {fetch_golden.shape}")
    return indices_arr, fetch_golden, fetch_sf_golden


def egram_npu(queue, rank, write_golden, storage, fetch_num, num_token, storage_type, indices_type,
              num_cpu_bytes, world_size, rank_per_dev, master_ip, master_port, is_aclgraph, is_profiling,
              use_fp8, sf_dtype, sf_table, profiling_path, fetch_mode, loop_time, repeat_time, warmup_time,
              seed, cpu_map, verify_write):
    """子进程入口: 一切依赖(含 world_size/端口/轮数/seed/cpu_map 等)均由显式参数传入, 不读模块级全局变量。

    结果消息协议: [rank, phase, detail], phase ∈ {"write", "fetch", "error"};
    每个子进程保证恰好 put 2 条消息(异常路径在 finally 中用哨兵补齐)。
    """
    msg_state = {"puts": 0, "error_sent": False}

    def put_result(phase, detail):
        msg_state["puts"] += 1
        queue.put([rank, phase, detail])

    warned_sizes = set()
    try:
        set_cpu_affinity(rank, cpu_map)
        set_device(rank, rank_per_dev)
        group = init_hccl_comm(rank, world_size, master_ip, master_port)

        device = f"npu:{(rank % rank_per_dev)}"
        local_rank = rank

        # 按 fetch_mode 决定本 rank 写入的表:
        #   sliced(0): 本 rank 分片 (write_golden[local_rank])
        #   local(1):  全量表 (storage 本身, 各 rank 一致)
        if fetch_mode == FETCH_MODE_LOCAL:
            write_inputs = storage
        else:
            write_inputs = write_golden[local_rank]
        num_entries, hidden = write_inputs.shape
        write_cpu = write_inputs
        mode_name = 'local' if fetch_mode == FETCH_MODE_LOCAL else 'sliced'
        print(f"[INFO] rank {rank}: fetch_mode={mode_name}, write_inputs shape={write_cpu.shape}")

        if num_cpu_bytes == 0:
            num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
                num_entries=num_entries, hidden=hidden, dtype=storage_type)
            print(f"[INFO] rank {rank}: num_cpu_bytes=0,自动计算存储大小: {num_cpu_bytes}")
        else:
            print(f"[INFO] rank {rank}: 使用用户指定num_cpu_bytes: {num_cpu_bytes}")

        buffer = ElasticBuffer(group=group, num_cpu_bytes=num_cpu_bytes, fetch_mode=fetch_mode)

        print(f"\n[INFO] rank: {rank} ElasticBuffer created, num_entries={num_entries}, hidden={hidden}")

        # write
        write_pass = True
        if use_fp8 and sf_table is not None:
            sf_device = sf_table.to(device)
            buffer.engram_write(write_cpu, sf_device)
        else:
            buffer.engram_write(write_cpu)
        print(f"[INFO] rank: {rank} engram_write completed, shape={write_cpu.shape}")

        # --verify_write: 读取 host 窗口指针与 write golden 比对(私有接口, 异常时打 WARNING 跳过)
        if verify_write:
            try:
                host_buf_ptr = buffer._runtime.get_host_buf_ptr()
                if not compare_write(host_buf_ptr, write_cpu):
                    write_pass = False
            except Exception as e:
                print(f"[WARNING] rank {rank} --verify_write 跳过: host 窗口指针读取失败({e})")

        # fetch
        fetch_pass = True
        if is_aclgraph:
            model = EngramFetchGraphModel(buffer, use_fp8).npu()
            import torch._dynamo
            torch._dynamo.config.recompile_limit = 256
            opt_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=False)

            if is_profiling:
                tensor_pre = torch.ones(10000, 10000, dtype=torch.float16).npu()
                torch.npu.synchronize()
                del tensor_pre
                torch.npu.empty_cache()
                rank_prof_path = os.path.join(profiling_path, f"rank_{rank}")
                if os.path.exists(rank_prof_path):
                        shutil.rmtree(rank_prof_path)
                os.makedirs(rank_prof_path, exist_ok=True)
                with get_profiling(fetch_num * loop_time, repeat_time, warmup_time, rank_prof_path) as prof:
                    for i in range(fetch_num):
                        indices_arr, fetch_golden, fetch_sf_golden = egram_cpu(
                            rank, i, num_token[i], indices_type, storage, sf_table, world_size, seed, warned_sizes)
                        print(f"[INFO] rank {rank} indices shape: {indices_arr.shape}, dtype: {indices_arr.dtype}")
                        print(f"[INFO] rank {rank} fetch_golden dtype: {fetch_golden.dtype}")
                        indices_npu = torch.from_numpy(indices_arr).to(device)

                        for loop_idx in range(loop_time):
                            with torch.profiler.record_function(f"Engram_rank_{rank}_fetch_{i}_loop_{loop_idx}"):
                                fetched, fetched_sf = opt_model(indices_npu)
                                prof.step()
                        print(f"[INFO] rank {rank} fetch[{i}]: fetched shape={fetched.shape}, dtype={fetched.dtype}")
                        if not compare_fetch(fetched, fetch_golden, rank, i):
                            fetch_pass = False
                        if use_fp8 and not compare_fetch_sf(fetched_sf, fetch_sf_golden, rank, i):
                            fetch_pass = False
            else:
                for i in range(fetch_num):
                    indices_arr, fetch_golden, fetch_sf_golden = egram_cpu(
                        rank, i, num_token[i], indices_type, storage, sf_table, world_size, seed, warned_sizes)
                    print(f"[INFO] rank {rank} indices shape: {indices_arr.shape}, dtype: {indices_arr.dtype}")
                    print(f"[INFO] rank {rank} fetch_golden dtype: {fetch_golden.dtype}")
                    indices_npu = torch.from_numpy(indices_arr).to(device)
                    fetched, fetched_sf = opt_model(indices_npu)

                    print(f"[INFO] rank {rank} fetch[{i}]: fetched shape={fetched.shape}, dtype={fetched.dtype}")
                    if not compare_fetch(fetched, fetch_golden, rank, i):
                        fetch_pass = False
                    if use_fp8 and not compare_fetch_sf(fetched_sf, fetch_sf_golden, rank, i):
                        fetch_pass = False
        else:
            if is_profiling:
                tensor_pre = torch.ones(10000, 10000, dtype=torch.float16).npu()
                torch.npu.synchronize()
                del tensor_pre
                torch.npu.empty_cache()
                rank_prof_path = os.path.join(profiling_path, f"rank_{rank}")
                if os.path.exists(rank_prof_path):
                        shutil.rmtree(rank_prof_path)
                os.makedirs(rank_prof_path, exist_ok=True)
                with get_profiling(fetch_num * loop_time, repeat_time, warmup_time, rank_prof_path) as prof:
                    for i in range(fetch_num):
                        indices_arr, fetch_golden, fetch_sf_golden = egram_cpu(
                            rank, i, num_token[i], indices_type, storage, sf_table, world_size, seed, warned_sizes)
                        print(f"[INFO] rank {rank} indices shape: {indices_arr.shape}, dtype: {indices_arr.dtype}")
                        print(f"[INFO] rank {rank} fetch_golden dtype: {fetch_golden.dtype}")
                        indices_npu = torch.from_numpy(indices_arr).to(device)
                        for loop_idx in range(loop_time):
                            with torch.profiler.record_function(f"Engram_rank_{rank}_fetch_{i}_loop_{loop_idx}"):
                                wait_func = buffer.engram_fetch(indices_npu)
                                result = wait_func()
                                if use_fp8:
                                    fetched, fetched_sf = result
                                else:
                                    fetched = result
                                prof.step()

                        print(f"[INFO] rank {rank} fetch[{i}]: fetched shape={fetched.shape}, dtype={fetched.dtype}")
                        if not compare_fetch(fetched, fetch_golden, rank, i):
                            fetch_pass = False
                        if use_fp8 and not compare_fetch_sf(fetched_sf, fetch_sf_golden, rank, i):
                            fetch_pass = False
            else:
                for i in range(fetch_num):
                    indices_arr, fetch_golden, fetch_sf_golden = egram_cpu(
                        rank, i, num_token[i], indices_type, storage, sf_table, world_size, seed, warned_sizes)
                    print(f"[INFO] rank {rank} indices shape: {indices_arr.shape}, dtype: {indices_arr.dtype}")
                    print(f"[INFO] rank {rank} fetch_golden dtype: {fetch_golden.dtype}")
                    indices_npu = torch.from_numpy(indices_arr).to(device)
                    wait_func = buffer.engram_fetch(indices_npu)
                    result = wait_func()
                    if use_fp8:
                        fetched, fetched_sf = result
                    else:
                        fetched = result

                    print(f"[INFO] rank {rank} fetch[{i}]: fetched shape={fetched.shape}, dtype={fetched.dtype}")
                    if not compare_fetch(fetched, fetch_golden, rank, i):
                        fetch_pass = False
                    if use_fp8 and not compare_fetch_sf(fetched_sf, fetch_sf_golden, rank, i):
                        fetch_pass = False

        buffer.destroy()
        dist.destroy_process_group()

        if write_pass:
            print(f"[INFO] rank {rank} WRITE SUCCESS")
            put_result("write", "success")
        else:
            print(f"[INFO] rank {rank} WRITE FAILED")
            put_result("write", "fail")

        if fetch_pass:
            print(f"[INFO] rank {rank} ALL  FETCH SUCCESS")
            put_result("fetch", "success")
        else:
            print(f"[INFO] rank {rank} SOME FETCH FAILED")
            put_result("fetch", "fail")

    except Exception as e:
        print(f"[ERROR] rank {rank} 执行失败: {str(e)}", flush=True)
        traceback.print_exc()
        if not msg_state["error_sent"] and msg_state["puts"] < 2:
            msg_state["error_sent"] = True
            put_result("error", f"{type(e).__name__}: {str(e)}")
    finally:
        # 保证恰好 put 2 条消息: 正常/异常路径缺失的消息用哨兵补齐(error_sent 标记保证不重复)
        try:
            while msg_state["puts"] < 2:
                put_result("error", "crashed: missing result (sentinel)")
        except Exception:
            pass
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
        print(f"[INFO] rank {rank} 资源清理完成", flush=True)


if __name__ == "__main__":
    # 获取入参，如没有则赋值全局变量
    (num_entries_total, hidden_size, storage_type, val_min, val_max, fetch_token_num,
     world_size, num_cpu_bytes, case_name, is_aclgraph, is_profiling, use_fp8, sf_dtype,
     seed, fetch_mode, prep_only, run_only, shape_semantics, start_method, master_port,
     loop_time, repeat_time, warmup_time, cpu_map, verify_write) = parse_cli_args(
                                                        num_entries_total, hidden_size, storage_type, fetch_token_num, world_size, num_cpu_bytes, is_aclgraph, is_profiling, use_fp8, seed)

    # sliced 模式校验: 全局 vocab 总行数必须能被 world_size 整除(全局->owner 均匀映射的单一属性)
    if fetch_mode == FETCH_MODE_SLICED and num_entries_total % world_size != 0:
        remainder = num_entries_total % world_size
        print(f"[ERROR] sliced 模式要求 input_tensor_shape[0](总vocab行数)能被 world_size 整除: "
              f"num_entries_total={num_entries_total}, world_size={world_size}, "
              f"建议将 input_tensor_shape[0] 调整为 {num_entries_total - remainder} 或 "
              f"{num_entries_total + (world_size - remainder)}")
        sys.exit(1)

    set_global_seed(seed)
    print(f"[INFO] 开始 engram 测试")
    print(f"[INFO] 当前参数: shape_semantics={shape_semantics}, num_entries_total={num_entries_total}, hidden_size={hidden_size}, dtype={storage_type}, world_size={world_size}, range=[{val_min},{val_max}], num_cpu_bytes={num_cpu_bytes} (type:{type(num_cpu_bytes)})\n")

    prep_data_file = f"./{case_name}_prep_data.pt" if case_name else "./prep_data.pt"

    if run_only:
        print(f"[INFO] [run_only] 从文件加载预准备数据: {prep_data_file}")
        prep_data = torch.load(prep_data_file, weights_only=False)
        storage = prep_data['storage']
        write_golden = list(storage.chunk(world_size, dim=0))
        sf_table = prep_data.get('sf_table', None)
        print(f"[INFO] [run_only] 数据加载完成")
    else:
        storage, write_golden, sf_table = gen_storage_write_golden(
            num_entries_total, hidden_size, world_size, storage_type, val_min, val_max, use_fp8, sf_dtype)

        if prep_only:
            print(f"[INFO] [prep_only] 保存预准备数据到: {prep_data_file}")
            torch.save({'storage': storage, 'sf_table': sf_table}, prep_data_file)
            print(f"[INFO] [prep_only] 数据准备完成，退出")
            sys.exit(0)

    print(f"[INFO] storage shape: {storage.shape}, dtype: {storage.dtype}")
    for i, w in enumerate(write_golden):
        print(f"[INFO] write_golden[{i}] shape: {w.shape}")
    mode_name = 'local' if fetch_mode == FETCH_MODE_LOCAL else 'sliced'
    print(f"[INFO] acl_graph: {is_aclgraph}, profiling: {is_profiling}, use_fp8: {use_fp8}, storage_dtype: {storage_type}, sf_dtype: {sf_dtype}, fetch_mode: {mode_name}")
    print(f"[INFO] fetch_mode={mode_name}: " + ("全量表冗余到每个NPU, 纯本地取数" if fetch_mode == FETCH_MODE_LOCAL else "全局表按rank切片分存, 跨NPU通信取数"))
    if use_fp8 and sf_table is not None:
        print(f"[INFO] sf_table shape: {sf_table.shape}, dtype: {sf_table.dtype}")

    # profiling 输出根目录: 环境变量 PROF_TMP_DIR 优先(runner 为每条用例建独立 mktemp 目录),
    # 默认 ./profiling_data/{case_name}_profiling, 一律绝对路径化(消除 qrun cwd 依赖)
    prof_root_env = os.environ.get("PROF_TMP_DIR")
    profiling_path = os.path.abspath(
        prof_root_env if prof_root_env else f"./profiling_data/{case_name if case_name else 'engram_test'}_profiling")
    if is_profiling:
        print(f"[INFO] profiling 输出根目录: {profiling_path}" + (" (来自 PROF_TMP_DIR)" if prof_root_env else ""))

    # 多进程启动: 默认 fork(父进程必须不初始化 NPU), 可选 forkserver 兜底
    mp.set_start_method(start_method, force=True)
    if start_method == "fork" and torch.npu.is_initialized():
        print("[ERROR] start_method=fork 要求父进程不初始化 NPU(检测到 torch.npu 已初始化), 拒绝 fork 子进程")
        sys.exit(1)

    proc_list = []
    manager = Manager()
    result_queue = manager.Queue()

    # 跨机场景: 每台机器只启动本机的 rank_per_dev 个进程; 单机场景: 启动全部 world_size 个进程
    if server_num > 1:
        print(f"multi_server scene!!!!! server_num={server_num}, world_size={world_size}, server_index={server_index}. Only support running case one by one!!!!!!!!!!!!!")
        rank_list = list(range(server_index * rank_per_dev, (server_index + 1) * rank_per_dev))
    else:
        print("single_server scene!!!!!")
        rank_list = list(range(world_size))
    print(f"rank list is: {rank_list}")

    for rank in rank_list:
        fetch_num = len(fetch_token_num[rank])
        num_token = fetch_token_num[rank]
        print(f"[INFO] rank: {rank} 的fetch次数:{fetch_num}")
        if len(num_token) != fetch_num:
            print(f"[ERROR] rank: {rank} input num_token length:{len(num_token)} != fetch_num:{fetch_num}")
        print(f"[INFO] rank: {rank} 每次fetch的token数为:{num_token}")
        proc = Process(
            target=egram_npu,
            args=(result_queue, rank, write_golden, storage, fetch_num, num_token,
                  storage_type, indices_type, num_cpu_bytes, world_size, rank_per_dev,
                  MASTER_IP, master_port, is_aclgraph, is_profiling, use_fp8, sf_dtype,
                  sf_table, profiling_path, fetch_mode, loop_time, repeat_time, warmup_time,
                  seed, cpu_map, verify_write)
        )
        proc.start()
        proc_list.append(proc)

    local_rank_num = len(rank_list)
    rank_write_outputs = [None] * local_rank_num
    rank_fetch_outputs = [None] * local_rank_num
    rank_error_outputs = [None] * local_rank_num

    # 收集子进程结果: 每个 rank 恰好 2 条消息 [rank, phase, detail], phase ∈ {"write","fetch","error"}
    expected = 2 * local_rank_num
    received = 0
    while received < expected:
        try:
            rank_id, phase, detail = result_queue.get(timeout=600)
        except Empty:
            print(f"[ERROR] 等待子进程结果超时(600s): 已收到 {received}/{expected} 条消息, "
                  f"缺失的 rank 按 crashed 计入失败")
            break
        received += 1
        print(f"[INFO] rank {rank_id} [{phase}] {detail}")
        local_rank_id = rank_id - server_index * rank_per_dev
        if not (0 <= local_rank_id < local_rank_num):
            print(f"[WARNING] 忽略非本机 rank 的消息: rank={rank_id}, phase={phase}, detail={detail}")
            continue
        if phase == "write":
            rank_write_outputs[local_rank_id] = detail
        elif phase == "fetch":
            rank_fetch_outputs[local_rank_id] = detail
        elif phase == "error":
            rank_error_outputs[local_rank_id] = detail
        else:
            print(f"[WARNING] rank {rank_id} 未知 phase: {phase!r}(detail: {detail}), 不静默丢弃, 按失败处理")
            rank_error_outputs[local_rank_id] = f"unknown phase {phase!r}: {detail}"

    for proc in proc_list:
        proc.join()

    for idx, global_rank_id in enumerate(rank_list):
        exitcode = proc_list[idx].exitcode
        if exitcode != 0:
            print(f"[ERROR] rank {global_rank_id} 子进程异常退出: exitcode={exitcode}")
            if rank_error_outputs[idx] is None:
                rank_error_outputs[idx] = f"crashed (exitcode={exitcode})"

    all_write_pass = all(res == "success" for res in rank_write_outputs)
    all_fetch_pass = all(res == "success" for res in rank_fetch_outputs)
    all_error_free = all(res is None for res in rank_error_outputs)
    all_success = all_write_pass and all_fetch_pass and all_error_free

    if all_success:
        print(f"\n{'='*60}")
        print("[SUCCESS] 所有 rank 测试通过!")
        print(f"{'='*60}")
        final_result = "SUCCESS"
    else:
        print(f"\n{'='*60}")
        print("[FAIL] 存在失败的 rank:")
        for i in range(local_rank_num):
            global_rank_id = server_index * rank_per_dev + i
            fail_msg = []
            if rank_write_outputs[i] != "success":
                fail_msg.append(f"write: {rank_write_outputs[i] if rank_write_outputs[i] is not None else 'no result (process crashed)'}")
            if rank_fetch_outputs[i] != "success":
                fail_msg.append(f"fetch: {rank_fetch_outputs[i] if rank_fetch_outputs[i] is not None else 'no result (process crashed)'}")
            if rank_error_outputs[i] is not None:
                fail_msg.append(f"error: {rank_error_outputs[i]}")
            if fail_msg:
                print(f"  rank {global_rank_id}: {', '.join(fail_msg)}")
        print(f"{'='*60}")
        final_result = "FAIL"

    csv_path = os.environ.get("RESULT_CSV", "result.csv")
    # 判断文件是否存在，不存在需要写表头
    file_exist = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exist:
            writer.writerow(["case_name", "result"])
        writer.writerow([case_name, final_result])

    print(f"\n[INFO] 测试结果已写入 {csv_path}: {case_name} -> {final_result}")

    sys.exit(0 if all_success else 1)
