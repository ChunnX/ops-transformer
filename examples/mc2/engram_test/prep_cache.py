"""prep_cache: engram_test 数据生成的 mmap 分块直写缓存 (设计文档 M2+M3)。

- 只依赖标准库 + numpy + torch, 无 torch_npu/cann_ops_transformer/psutil 依赖,
  无 NPU 机器可运行; 模块导入零副作用;
- 数据按最终 dtype 字节宽度直接落盘 (np.memmap), 分块生成、逐块转换, 无整表 fp32 中间、
  无 clone、不依赖 /dev/shm;
- 确定性: 每块 rng 由 SeedSequence(entropy=seed, spawn_key=(kind_id, block_idx)) 派生,
  数据只依赖 (seed, kind, block_idx), 与 worker 数无关 → 任意机器任意并行度逐字节一致;
- 缓存键不含 fetch_mode: local/sliced 配对用例自动复用同一份缓存 (M3);
- GEN_VERSION 递增即可使全部旧缓存自动失效。
"""

import argparse
import ctypes
import glob
import hashlib
import json
import multiprocessing as mp
import os
import platform
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np
import torch

GEN_VERSION = 1
KIND_STORAGE = 0
KIND_SF = 1
DEFAULT_BLOCK_ROWS = 8192
MADV_DONTNEED = 4

# dtype 别名 -> 规范名 (规范名参与缓存键/meta)
_DTYPE_ALIASES = {
    "fp8_e4m3fn": "fp8_e4m3fn",
    "float8_e4m3fn": "fp8_e4m3fn",
    "fp8_e5m2": "fp8_e5m2",
    "float8_e5m2": "fp8_e5m2",
    "e8m0": "fp8_e8m0",
    "fp8_e8m0": "fp8_e8m0",
    "float8_e8m0fnu": "fp8_e8m0",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "fp32": "fp32",
    "float32": "fp32",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
}

# 规范名 -> torch dtype
_TORCH_DTYPE = {
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
    "fp8_e8m0": torch.float8_e8m0fnu,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}

# 规范名 -> 落盘字节宽度的 numpy dtype (fp8 系 1 字节, bf16/fp16 2 字节, fp32 4 字节)
_NP_WIDTH = {
    "fp8_e4m3fn": "|u1",
    "fp8_e5m2": "|u1",
    "fp8_e8m0": "|u1",
    "bf16": "|u2",
    "fp16": "|u2",
    "fp32": "|u4",
    "int8": "i1",
    "int16": "i2",
    "int32": "i4",
    "int64": "i8",
    "uint8": "|u1",
}

_DTYPE_IS_INT = {name: name.startswith(("int", "uint")) for name in _TORCH_DTYPE}

# 浮点 dtype -> 等宽且可 numpy 化的 torch dtype (numpy 无 fp8/bf16, 统一按位宽 view 后写盘)
_BYTES_VIEW_TORCH = {
    "fp8_e4m3fn": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "fp8_e8m0": torch.uint8,
    "bf16": torch.uint16,
    "fp16": torch.uint16,
    "fp32": torch.uint32,
}

_TORCH_DTYPE_TO_NAME = {v: k for k, v in _TORCH_DTYPE.items()}

# load() 注册的 memmap 区间 [(addr, nbytes, np.memmap)], drop_page_cache 只处理注册区间,
# 防止误对普通内存(窗口/堆)调 MADV_DONTNEED 清零
_MEMMAP_REGISTRY = []


def normalize_dtype(dtype):
    """torch dtype 或别名串 -> 规范名。"""
    if isinstance(dtype, str):
        key = dtype.strip().lower()
        if key not in _DTYPE_ALIASES:
            raise ValueError(f"不支持的 dtype: {dtype!r}, 可选: {sorted(set(_DTYPE_ALIASES.values()))}")
        return _DTYPE_ALIASES[key]
    name = _TORCH_DTYPE_TO_NAME.get(dtype)
    if name is None:
        raise ValueError(f"不支持的 torch dtype: {dtype}")
    return name


def cache_key(total, hidden, dtype_name, use_fp8, sf_dtype_name, seed):
    """缓存键: sha1(json{total, hidden, dtype, use_fp8, sf_dtype, seed, GEN_VERSION})。

    注意 fetch_mode 不参与 (local/sliced 配对复用); pattern 不参与键、但参与 meta 命中校验。
    """
    payload = json.dumps({
        "total": int(total),
        "hidden": int(hidden),
        "dtype": normalize_dtype(dtype_name),
        "use_fp8": int(bool(use_fp8)),
        "sf_dtype": normalize_dtype(sf_dtype_name),
        "seed": int(seed),
        "gen_version": GEN_VERSION,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _paths(cache_dir, key):
    base = os.path.join(cache_dir, key)
    return base + ".storage.bin", base + ".sf.bin", base + ".meta.json"


def _expected_meta(total, hidden, dtype_name, use_fp8, sf_dtype_name, seed, pattern):
    return {
        "total": int(total),
        "hidden": int(hidden),
        "dtype": dtype_name,
        "use_fp8": int(bool(use_fp8)),
        "sf_dtype": sf_dtype_name,
        "seed": int(seed),
        "gen_version": GEN_VERSION,
        "pattern": pattern,
    }


def _cache_hit(cache_dir, total, hidden, dtype_name, use_fp8, sf_dtype_name, seed, pattern):
    """命中判定: 三文件齐全(sf 按需)且 meta 全字段匹配且文件字节数一致。"""
    key = cache_key(total, hidden, dtype_name, use_fp8, sf_dtype_name, seed)
    storage_path, sf_path, meta_path = _paths(cache_dir, key)
    if not (os.path.isfile(storage_path) and os.path.isfile(meta_path)):
        return False
    if use_fp8 and not os.path.isfile(sf_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    expected = _expected_meta(total, hidden, dtype_name, use_fp8, sf_dtype_name, seed, pattern)
    if any(meta.get(k) != v for k, v in expected.items()):
        return False
    if os.path.getsize(storage_path) != total * hidden * np.dtype(_NP_WIDTH[dtype_name]).itemsize:
        return False
    if use_fp8:
        sf_cols = (hidden + 31) // 32
        if os.path.getsize(sf_path) != total * sf_cols * np.dtype(_NP_WIDTH[sf_dtype_name]).itemsize:
            return False
    return True


def num_blocks(total, block_rows=DEFAULT_BLOCK_ROWS):
    return (int(total) + int(block_rows) - 1) // int(block_rows)


def _block_rng(seed, kind_id, block_idx):
    """每块独立 rng: 只依赖 (seed, kind, block_idx), 与 worker 数无关。"""
    return np.random.default_rng(
        np.random.SeedSequence(entropy=int(seed), spawn_key=(int(kind_id), int(block_idx))))


def _fp32_to_width(blk, dtype_name):
    """fp32 小块 -> 目标 dtype -> 等宽字节 view, 返回落盘宽度 dtype 的 ndarray。"""
    t = torch.from_numpy(blk).to(_TORCH_DTYPE[dtype_name])
    return t.view(_BYTES_VIEW_TORCH[dtype_name]).numpy()


def _int_range(dtype_name, val_min, val_max):
    if val_min is not None and val_max is not None:
        low, high = int(val_min), int(val_max) + 1
    elif dtype_name == "uint8":
        low, high = 0, 256
    else:
        low, high = -128, 128
    return low, high - low


def _arange_block(r0, r1, cols, dtype_name, val_min=None, val_max=None):
    """pattern=arange: (row*2654435761 + col) 混合后截断的行号派生模式, 不用 rng, 近零成本。

    浮点 dtype 映射到小整数锯齿波再按 dtype 转换, 保证无 NaN/Inf (golden 位级比对可用);
    按 ~4M 元素子块计算, 避免 int64 大临时。
    """
    width = np.dtype(_NP_WIDTH[dtype_name])
    out = np.empty((r1 - r0, cols), dtype=width)
    is_int = _DTYPE_IS_INT[dtype_name]
    sub = max(1, (4 * 1024 * 1024) // max(1, cols))
    for rs in range(r0, r1, sub):
        re = min(rs + sub, r1)
        rows = np.arange(rs, re, dtype=np.int64)[:, None]
        cols_v = np.arange(cols, dtype=np.int64)[None, :]
        mixed = (rows * np.int64(2654435761) + cols_v) & np.int64(0x7FFFFFFF)
        if is_int:
            low, span = _int_range(dtype_name, val_min, val_max)
            out[rs - r0:re - r0] = (np.int64(low) + mixed % np.int64(span)).astype(width)
        else:
            base = (mixed % np.int64(197)).astype(np.float32) - np.float32(98.0)
            if val_min is not None and val_max is not None:
                base = (base + np.float32(98.0)) * (np.float32(val_max - val_min) / np.float32(196.0)) \
                    + np.float32(val_min)
            out[rs - r0:re - r0] = _fp32_to_width(base, dtype_name)
    return out


def _arange_sf_block(r0, r1, sf_cols, sf_dtype_name):
    """sf 表的 arange 模式: e8m0 取 [1,254] 的合法指数字节(避开 0x00/0xFF), fp32 走锯齿波。"""
    if sf_dtype_name != "fp8_e8m0":
        return _arange_block(r0, r1, sf_cols, "fp32")
    out = np.empty((r1 - r0, sf_cols), dtype=np.uint8)
    sub = max(1, (4 * 1024 * 1024) // max(1, sf_cols))
    for rs in range(r0, r1, sub):
        re = min(rs + sub, r1)
        rows = np.arange(rs, re, dtype=np.int64)[:, None]
        cols_v = np.arange(sf_cols, dtype=np.int64)[None, :]
        mixed = (rows * np.int64(2654435761) + cols_v) & np.int64(0x7FFFFFFF)
        out[rs - r0:re - r0] = (1 + mixed % np.int64(254)).astype(np.uint8)
    return out


def _compute_storage_block(r0, r1, hidden, dtype_name, seed, block_idx, pattern, val_min, val_max):
    """纯函数: 重算 storage 块 [r0, r1) 的最终字节 (落盘宽度 dtype)。"""
    if pattern == "arange":
        return _arange_block(r0, r1, hidden, dtype_name, val_min, val_max)
    rng = _block_rng(seed, KIND_STORAGE, block_idx)
    if _DTYPE_IS_INT[dtype_name]:
        low, high = _int_range(dtype_name, val_min, val_max)
        return rng.integers(low, high, size=(r1 - r0, hidden), dtype=np.dtype(_NP_WIDTH[dtype_name]))
    if val_min is not None and val_max is not None:
        blk = rng.random((r1 - r0, hidden), dtype=np.float32)
        blk *= np.float32(val_max - val_min)
        blk += np.float32(val_min)
    else:
        blk = rng.standard_normal((r1 - r0, hidden), dtype=np.float32)
    return _fp32_to_width(blk, dtype_name)


def _compute_sf_block(r0, r1, sf_cols, sf_dtype_name, seed, block_idx, pattern):
    """纯函数: 重算 sf 块 [r0, r1)。e8m0: rng.random fp32 位运算 (bits>>23)&0xFF; fp32: 直写。"""
    if pattern == "arange":
        return _arange_sf_block(r0, r1, sf_cols, sf_dtype_name)
    rng = _block_rng(seed, KIND_SF, block_idx)
    if sf_dtype_name == "fp8_e8m0":
        r = rng.random((r1 - r0, sf_cols), dtype=np.float32)
        return ((r.view(np.uint32) >> np.uint32(23)) & np.uint32(0xFF)).astype(np.uint8)
    return _fp32_to_width(rng.random((r1 - r0, sf_cols), dtype=np.float32), "fp32")


def expected_storage_block(block_idx, total, hidden, dtype, seed, pattern="random",
                           block_rows=DEFAULT_BLOCK_ROWS, val_min=None, val_max=None):
    """独立重算 storage 块 (自测/基准抽样校验用), 返回落盘宽度 dtype 的 ndarray。"""
    r0 = int(block_idx) * int(block_rows)
    if r0 >= int(total):
        raise ValueError(f"block_idx {block_idx} 超出 total {total}")
    r1 = min(r0 + int(block_rows), int(total))
    return _compute_storage_block(r0, r1, int(hidden), normalize_dtype(dtype), int(seed),
                                  int(block_idx), pattern, val_min, val_max)


def expected_sf_block(block_idx, total, hidden, sf_dtype, seed, pattern="random",
                      block_rows=DEFAULT_BLOCK_ROWS):
    """独立重算 sf 块 (自测/基准抽样校验用), 返回落盘宽度 dtype 的 ndarray。"""
    r0 = int(block_idx) * int(block_rows)
    if r0 >= int(total):
        raise ValueError(f"block_idx {block_idx} 超出 total {total}")
    r1 = min(r0 + int(block_rows), int(total))
    return _compute_sf_block(r0, r1, (int(hidden) + 31) // 32, normalize_dtype(sf_dtype),
                             int(seed), int(block_idx), pattern)


def _write_blocks(task):
    """worker: 打开自己的 np.memmap(mode='r+'), 逐块生成小块临时并立即转换写入。

    入口先 torch.set_num_threads(1): fork 出的 worker 若继承父进程已拉起的 OpenMP
    线程组状态, 首个并行区会挂死 (libgomp fork 已知问题), 单线程化后免疫。
    """
    torch.set_num_threads(1)
    (kind, path, total, rows_dim, width_str, blocks, block_rows,
     dtype_name, seed, pattern, val_min, val_max) = task
    mm = np.memmap(path, mode="r+", shape=(total, rows_dim), dtype=np.dtype(width_str))
    try:
        for b in blocks:
            r0 = b * block_rows
            r1 = min(r0 + block_rows, total)
            if kind == KIND_STORAGE:
                blk = _compute_storage_block(r0, r1, rows_dim, dtype_name, seed, b, pattern,
                                             val_min, val_max)
            else:
                blk = _compute_sf_block(r0, r1, rows_dim, dtype_name, seed, b, pattern)
            if blk.dtype != mm.dtype:
                blk = blk.view(mm.dtype)
            mm[r0:r1] = blk
            del blk
        mm.flush()
    finally:
        del mm
    return len(blocks)


def _round_robin_groups(num_blk, num_workers):
    return [[b for b in range(num_blk) if b % num_workers == w] for w in range(num_workers)]


def generate(cache_dir, total, hidden, dtype, use_fp8, sf_dtype, seed, pattern="random",
             num_workers=None, block_rows=DEFAULT_BLOCK_ROWS, val_min=None, val_max=None):
    """生成(或命中)prep 缓存: {cache_dir}/{key}.storage.bin / {key}.sf.bin / {key}.meta.json。

    - 命中: 三文件齐全(sf 按需)且 meta 全字段匹配, 不重生成;
    - 未命中: np.memmap(mode='w+') 按最终 dtype 字节宽度落盘, 行空间切成固定 block_rows 的块,
      块按 round-robin 分给 fork Pool 的 worker, 每块 rng 由 (seed, kind, block_idx) 派生;
    - 返回 {"cache_hit": bool, "key", "path", "sf_path", "elapsed_s", "storage_bytes", "sf_bytes"}。
    """
    t_start = time.time()
    total, hidden, seed = int(total), int(hidden), int(seed)
    dtype_name = normalize_dtype(dtype)
    sf_dtype_name = normalize_dtype(sf_dtype)
    use_fp8 = bool(use_fp8)
    if total <= 0 or hidden <= 0:
        raise ValueError(f"total/hidden 必须为正整数: total={total}, hidden={hidden}")
    if pattern not in ("random", "arange"):
        raise ValueError(f"pattern 只支持 random/arange: {pattern!r}")
    block_rows = max(1, int(block_rows))
    if num_workers is None:
        num_workers = min(32, (os.cpu_count() or 1) // 2)
    num_workers = max(1, int(num_workers))

    key = cache_key(total, hidden, dtype_name, use_fp8, sf_dtype_name, seed)
    storage_path, sf_path, meta_path = _paths(cache_dir, key)
    sf_cols = (hidden + 31) // 32

    if _cache_hit(cache_dir, total, hidden, dtype_name, use_fp8, sf_dtype_name, seed, pattern):
        return {
            "cache_hit": True,
            "key": key,
            "path": storage_path,
            "sf_path": sf_path if use_fp8 else None,
            "elapsed_s": time.time() - t_start,
            "storage_bytes": os.path.getsize(storage_path),
            "sf_bytes": os.path.getsize(sf_path) if use_fp8 else 0,
        }

    os.makedirs(cache_dir, exist_ok=True)
    n_blk = num_blocks(total, block_rows)
    workers = min(num_workers, n_blk)

    mm = np.memmap(storage_path, mode="w+", shape=(total, hidden),
                   dtype=np.dtype(_NP_WIDTH[dtype_name]))
    sf_mm = None
    if use_fp8:
        sf_mm = np.memmap(sf_path, mode="w+", shape=(total, sf_cols),
                          dtype=np.dtype(_NP_WIDTH[sf_dtype_name]))
    groups = [g for g in _round_robin_groups(n_blk, workers) if g]
    tasks = [
        (KIND_STORAGE, storage_path, total, hidden, _NP_WIDTH[dtype_name], g, block_rows,
         dtype_name, seed, pattern, val_min, val_max)
        for g in groups
    ]
    if use_fp8:
        tasks += [
            (KIND_SF, sf_path, total, sf_cols, _NP_WIDTH[sf_dtype_name], g, block_rows,
             sf_dtype_name, seed, pattern, None, None)
            for g in groups
        ]
    try:
        if workers <= 1:
            # 单 worker 内联路径: 主进程单线程化, 避免拉起 OpenMP 线程组,
            # 保证本进程后续 fork (Pool/测试子进程) 不受 libgomp fork 问题影响
            prev_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                for task in tasks:
                    _write_blocks(task)
            finally:
                torch.set_num_threads(prev_threads)
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=workers) as pool:
                pool.map(_write_blocks, tasks, chunksize=1)
        mm.flush()
        if sf_mm is not None:
            sf_mm.flush()
    finally:
        del mm
        del sf_mm

    elapsed = time.time() - t_start
    meta = _expected_meta(total, hidden, dtype_name, use_fp8, sf_dtype_name, seed, pattern)
    meta.update({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "host": platform.node(),
        "elapsed_s": round(elapsed, 3),
        "workers": workers,
    })
    tmp_meta = meta_path + ".tmp"
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(tmp_meta, meta_path)
    return {
        "cache_hit": False,
        "key": key,
        "path": storage_path,
        "sf_path": sf_path if use_fp8 else None,
        "elapsed_s": elapsed,
        "storage_bytes": total * hidden * np.dtype(_NP_WIDTH[dtype_name]).itemsize,
        "sf_bytes": total * sf_cols * np.dtype(_NP_WIDTH[sf_dtype_name]).itemsize if use_fp8 else 0,
    }


def _register_memmap(mm):
    _MEMMAP_REGISTRY.append((mm.ctypes.data, mm.nbytes, mm))


def load(cache_dir, total, hidden, dtype, use_fp8, sf_dtype, seed, pattern="random"):
    """加载缓存为 memmap 视图张量 (零拷贝, 不 clone)。

    返回 (storage_tensor, sf_tensor|None); fp8 系为 .view(torch dtype) 后的视图。
    meta 不匹配/缺失/文件缺失时抛 RuntimeError (含 key 与期望 meta)。
    """
    total, hidden, seed = int(total), int(hidden), int(seed)
    dtype_name = normalize_dtype(dtype)
    sf_dtype_name = normalize_dtype(sf_dtype)
    use_fp8 = bool(use_fp8)
    key = cache_key(total, hidden, dtype_name, use_fp8, sf_dtype_name, seed)
    storage_path, sf_path, meta_path = _paths(cache_dir, key)
    if not _cache_hit(cache_dir, total, hidden, dtype_name, use_fp8, sf_dtype_name, seed, pattern):
        expected = _expected_meta(total, hidden, dtype_name, use_fp8, sf_dtype_name, seed, pattern)
        raise RuntimeError(
            f"prep_cache 未命中: cache_dir={cache_dir}, key={key}, 期望 meta={expected}, "
            f"storage 存在={os.path.exists(storage_path)}, sf 存在={os.path.exists(sf_path)}, "
            f"meta 存在={os.path.exists(meta_path)}; 请先执行数据准备(generate/prep_only)")

    storage_mm = np.memmap(storage_path, mode="r", shape=(total, hidden),
                           dtype=np.dtype(_NP_WIDTH[dtype_name]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        storage_tensor = torch.from_numpy(storage_mm).view(_TORCH_DTYPE[dtype_name])
    _register_memmap(storage_mm)

    sf_tensor = None
    if use_fp8:
        sf_cols = (hidden + 31) // 32
        sf_mm = np.memmap(sf_path, mode="r", shape=(total, sf_cols),
                          dtype=np.dtype(_NP_WIDTH[sf_dtype_name]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf_tensor = torch.from_numpy(sf_mm).view(_TORCH_DTYPE[sf_dtype_name])
        _register_memmap(sf_mm)
    return storage_tensor, sf_tensor


def drop_page_cache(tensor):
    """对 load() 返回的 memmap 视图张量调 madvise(MADV_DONTNEED) 释放页缓存 (尽力而为)。

    只处理 prep_cache 注册的 memmap 区间, 防止误伤普通内存; 任何异常打 WARNING 后忽略。
    """
    try:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return
        ptr = tensor.data_ptr()
        nbytes = tensor.numel() * tensor.element_size()
        if ptr == 0 or nbytes == 0:
            return
        if not any(addr <= ptr and ptr + nbytes <= addr + size for addr, size, _ in _MEMMAP_REGISTRY):
            raise RuntimeError(
                f"tensor @0x{ptr:x} 不在 prep_cache 注册的 memmap 区间内, 跳过以防误伤普通内存")
        page = os.sysconf("SC_PAGE_SIZE")
        start = ptr & ~(page - 1)
        end = (ptr + nbytes + page - 1) & ~(page - 1)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int
        ret = libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start),
                           ctypes.c_int(MADV_DONTNEED))
        if ret != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
    except Exception as e:
        print(f"[WARNING] drop_page_cache 失败, 已忽略: {e}")


def purge(cache_dir, keys=None):
    """删除全部(keys=None)或指定 key 的缓存文件, 返回已删除路径列表。"""
    removed = []
    if not os.path.isdir(cache_dir):
        return removed
    if keys is None:
        for pat in ("*.storage.bin", "*.sf.bin", "*.meta.json", "*.meta.json.tmp"):
            for p in glob.glob(os.path.join(cache_dir, pat)):
                try:
                    os.remove(p)
                    removed.append(p)
                except FileNotFoundError:
                    pass
    else:
        for key in keys:
            for p in _paths(cache_dir, key):
                if os.path.exists(p):
                    os.remove(p)
                    removed.append(p)
    return removed


def _selftest():
    cache_dir = tempfile.mkdtemp(prefix="prep_cache_selftest_")
    try:
        total, hidden, seed = 4096, 576, 12345
        r = generate(cache_dir, total, hidden, "fp8_e4m3fn", True, "e8m0", seed,
                     pattern="random", num_workers=2, block_rows=1024)
        assert not r["cache_hit"], "首次 generate 应为未命中"
        r2 = generate(cache_dir, total, hidden, "fp8_e4m3fn", True, "e8m0", seed,
                      pattern="random", num_workers=2, block_rows=1024)
        assert r2["cache_hit"], "二次 generate 应命中缓存"

        storage, sf = load(cache_dir, total, hidden, "fp8_e4m3fn", True, "e8m0", seed)
        assert tuple(storage.shape) == (total, hidden), f"storage shape: {storage.shape}"
        assert storage.dtype == torch.float8_e4m3fn, f"storage dtype: {storage.dtype}"
        sf_cols = (hidden + 31) // 32
        assert tuple(sf.shape) == (total, sf_cols), f"sf shape: {sf.shape}"
        assert sf.dtype == torch.float8_e8m0fnu, f"sf dtype: {sf.dtype}"

        s_raw = storage.view(torch.uint8).numpy()
        sf_raw = sf.view(torch.uint8).numpy()
        for b in (0, 2, 3):  # 首块 / 中部 / 末块, 独立 rng 重算逐字节比对
            r0 = b * 1024
            r1 = min(r0 + 1024, total)
            exp_s = expected_storage_block(b, total, hidden, "fp8_e4m3fn", seed, block_rows=1024)
            exp_f = expected_sf_block(b, total, hidden, "e8m0", seed, block_rows=1024)
            assert np.array_equal(s_raw[r0:r1], exp_s), f"storage 块 {b} 与独立重算不一致"
            assert np.array_equal(sf_raw[r0:r1], exp_f), f"sf 块 {b} 与独立重算不一致"

        removed = purge(cache_dir)
        assert removed, "purge 应删除缓存文件"
        assert not os.path.exists(r["path"]), "purge 后 storage 文件应不存在"
        assert not os.path.exists(r["sf_path"]), "purge 后 sf 文件应不存在"
        print("SELFTEST PASS")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def _main():
    parser = argparse.ArgumentParser(description="prep_cache 自测 (设计文档 M2+M3)")
    parser.add_argument("--selftest", action="store_true", help="运行自测: 生成/命中/加载/抽样比对/清理")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
        return
    parser.error("仅支持 --selftest")


if __name__ == "__main__":
    _main()
