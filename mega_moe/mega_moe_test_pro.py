import argparse
import builtins
import functools
import gc
import json
import os
import random
import sys
import threading
import time

import numpy as np
import torch

_print = builtins.print
builtins.print = functools.partial(_print, flush=True)

from gen_input_and_golden import GenInputAndGolden


def get_current_rss_bytes():
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            return None

    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            get_process_memory_info = getattr(
                ctypes.windll.kernel32,
                "K32GetProcessMemoryInfo",
                ctypes.windll.psapi.GetProcessMemoryInfo,
            )
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            ok = get_process_memory_info(handle, ctypes.byref(counters), counters.cb)
            if ok:
                return int(counters.WorkingSetSize)
        except Exception:
            return None

    return None


def get_resource_peak_rss_bytes():
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(peak)
        return int(peak) * 1024
    except Exception:
        return None


class PeakMemoryMonitor:
    def __init__(self, interval_seconds):
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.sample()
        self._thread.start()

    def sample(self):
        rss_bytes = get_current_rss_bytes()
        if rss_bytes is not None:
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self.sample()
        resource_peak = get_resource_peak_rss_bytes()
        if resource_peak is not None:
            self.peak_rss_bytes = max(self.peak_rss_bytes, resource_peak)
        return self.peak_rss_bytes

    def _run(self):
        while not self._stop_event.wait(self.interval_seconds):
            self.sample()


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


GOLDEN_FFN_CHUNK_SIZE = 16
GOLDEN_LOG_INTERVAL = 512
GOLDEN_LOG_STATS = False
GOLDEN_BATCH_MATMUL = False
GOLDEN_BATCH_MATMUL_MIN_M = 128
GOLDEN_TORCH_NUM_INTEROP_THREADS = 1


def get_total_cpu_count():
    return max(1, os.cpu_count() or 1)


def get_golden_parallel_config(chunk_size=GOLDEN_FFN_CHUNK_SIZE):
    cpu_count = get_total_cpu_count()
    expert_workers = min(chunk_size, max(1, cpu_count // 4))
    torch_num_threads = max(1, (cpu_count + expert_workers - 1) // expert_workers)
    return cpu_count, expert_workers, torch_num_threads


def get_torch_dtype(name):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map[name.lower()]


def format_bytes(num_bytes):
    if num_bytes is None:
        return "unknown"
    return f"{num_bytes / 1024 ** 3:.3f} GiB ({num_bytes / 1024 ** 2:.1f} MiB)"


def build_args():
    _, default_expert_workers, default_torch_num_threads = get_golden_parallel_config()
    parser = argparse.ArgumentParser(description="Run Mega MoE golden only and print elapsed time plus peak RSS.")
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--h", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--ep-world-size", type=int, default=2)
    parser.add_argument("--local-moe-expert-num", type=int, default=1024)
    parser.add_argument("--quant-mode", type=int, default=4)
    parser.add_argument("--comm-quant-mode", type=int, default=0)
    parser.add_argument("--transpose-weight", type=int, default=1)
    parser.add_argument("--quant-dtype", default="float8_e4m3fn")
    parser.add_argument("--weight-dtype", default="float8_e4m3fn")
    parser.add_argument("--swiglu-quant-dtype", default="float8_e4m3fn")
    parser.add_argument("--out-dtype", default="bfloat16", choices=("bfloat16", "bf16", "float16", "fp16", "float32", "fp32"))
    parser.add_argument("--expert-scales-dtype", default="float32", choices=("float32", "fp32"))
    parser.add_argument("--clamp-limit", type=float, default=-1)
    parser.add_argument("--golden-ffn-chunk-size", type=int, default=GOLDEN_FFN_CHUNK_SIZE)
    parser.add_argument("--golden-log-interval", type=int, default=GOLDEN_LOG_INTERVAL)
    parser.add_argument("--golden-log-stats", type=parse_bool, default=GOLDEN_LOG_STATS)
    parser.add_argument("--golden-batch-matmul", type=parse_bool, default=GOLDEN_BATCH_MATMUL)
    parser.add_argument("--golden-batch-matmul-min-m", type=int, default=GOLDEN_BATCH_MATMUL_MIN_M)
    parser.add_argument("--golden-expert-workers", type=int, default=default_expert_workers)
    parser.add_argument("--fast-random-weight", type=parse_bool, default=True)
    parser.add_argument("--is-topk-balance", type=parse_bool, default=True)
    parser.add_argument("--is-x-random", type=parse_bool, default=True)
    parser.add_argument("--is-mask", type=int, default=0)
    parser.add_argument("--is-variable-bs", type=int, default=0)
    parser.add_argument("--is-unfixed-real-token", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=831)
    parser.add_argument("--monitor-interval", type=float, default=0.1)
    parser.add_argument("--torch-num-threads", type=int, default=default_torch_num_threads)
    parser.add_argument("--torch-num-interop-threads", type=int, default=GOLDEN_TORCH_NUM_INTEROP_THREADS)
    return parser.parse_args()


def main():
    args = build_args()
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_num_interop_threads)
    global_bs = args.bs * args.ep_world_size
    moe_expert_num = args.ep_world_size * args.local_moe_expert_num

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    input_kwargs = {
        "bs": args.bs,
        "h": args.h,
        "n": args.n,
        "quant_dtype": args.quant_dtype,
        "is_x_random": args.is_x_random,
        "global_bs": global_bs,
        "out_dtype": get_torch_dtype(args.out_dtype),
        "moe_expert_num": moe_expert_num,
        "is_topk_balance": args.is_topk_balance,
        "k": args.k,
        "is_mask": args.is_mask,
        "is_unfixed_real_token": args.is_unfixed_real_token,
        "ep_world_size": args.ep_world_size,
        "is_variable_bs": args.is_variable_bs,
        "expert_scales_dtype": get_torch_dtype(args.expert_scales_dtype),
        "local_moe_expert_num": args.local_moe_expert_num,
        "quant_mode": args.quant_mode,
        "transpose_weight": args.transpose_weight,
        "comm_quant_mode": args.comm_quant_mode,
        "weight_dtype": args.weight_dtype,
        "swiglu_quant_dtype": args.swiglu_quant_dtype,
        "clamp_limit": args.clamp_limit,
        "golden_ffn_chunk_size": args.golden_ffn_chunk_size,
        "golden_log_interval": args.golden_log_interval,
        "golden_log_stats": args.golden_log_stats,
        "golden_batch_matmul": args.golden_batch_matmul,
        "golden_batch_matmul_min_m": args.golden_batch_matmul_min_m,
        "golden_expert_workers": args.golden_expert_workers,
        "fast_random_weight": args.fast_random_weight,
        "retain_npu_weights": False,
        "random_seed": args.random_seed,
    }

    config = {
        "bs": args.bs,
        "cpu_count": get_total_cpu_count(),
        "global_bs": global_bs,
        "ep_world_size": args.ep_world_size,
        "local_moe_expert_num": args.local_moe_expert_num,
        "moe_expert_num": moe_expert_num,
        "h": args.h,
        "n": args.n,
        "k": args.k,
        "quant_mode": args.quant_mode,
        "comm_quant_mode": args.comm_quant_mode,
        "transpose_weight": args.transpose_weight,
        "quant_dtype": args.quant_dtype,
        "weight_dtype": args.weight_dtype,
        "swiglu_quant_dtype": args.swiglu_quant_dtype,
        "clamp_limit": args.clamp_limit,
        "golden_ffn_chunk_size": args.golden_ffn_chunk_size,
        "golden_log_interval": args.golden_log_interval,
        "golden_log_stats": args.golden_log_stats,
        "golden_batch_matmul": args.golden_batch_matmul,
        "golden_batch_matmul_min_m": args.golden_batch_matmul_min_m,
        "golden_expert_workers": args.golden_expert_workers,
        "fast_random_weight": args.fast_random_weight,
        "retain_npu_weights": False,
        "random_seed": args.random_seed,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "pid": os.getpid(),
    }

    print("[INFO] Mega MoE golden-only test config:")
    print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"[INFO] Initial RSS: {format_bytes(get_current_rss_bytes())}")

    monitor = PeakMemoryMonitor(args.monitor_interval)
    input_data = None
    start_time = time.perf_counter()
    monitor.start()
    try:
        input_data = GenInputAndGolden(**input_kwargs)
    except BaseException:
        elapsed = time.perf_counter() - start_time
        peak_rss_bytes = monitor.stop()
        print(f"[ERROR] Golden failed after {elapsed:.3f}s")
        print(f"[RESULT] peak_rss_bytes={peak_rss_bytes}")
        print(f"[RESULT] peak_rss={format_bytes(peak_rss_bytes)}")
        raise
    else:
        elapsed = time.perf_counter() - start_time
        peak_rss_bytes = monitor.stop()
        summary = {
            "golden_seconds": elapsed,
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_gib": None if peak_rss_bytes is None else peak_rss_bytes / 1024 ** 3,
            "current_rss_bytes": get_current_rss_bytes(),
            "x_final_shapes": [list(x.shape) for x in input_data.x_final],
            "expert_token_nums_shapes": [list(x.shape) for x in input_data.expert_token_nums],
        }
        print("[RESULT] Mega MoE golden-only test summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"[RESULT] Golden time: {elapsed:.3f}s")
        print(f"[RESULT] Peak RSS: {format_bytes(peak_rss_bytes)}")
    finally:
        del input_data
        gc.collect()


if __name__ == "__main__":
    main()
