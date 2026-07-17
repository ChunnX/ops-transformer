import csv
import gc
import logging
import time
import traceback
from pathlib import Path

import torch

from compare import Comparator, DefaultComparator
from core.gen_data import GenDataBackend
from core.cpu import CpuBackend
from core.npu import NpuBackend
from csv_parser import CSV_HEADERS
from precision_analyze import analyze_failure

logger = logging.getLogger("orchestrator")


# ============ Reporter ============
class Reporter:
    def __init__(self):
        self._results = {}
        self._dtypes = {}
        self._csv_file = None
        self._csv_writer = None

    def set_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        self._csv_file = open(path, "a", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, CSV_HEADERS)
        if not exists:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def record(self, name: str, result: dict, error: str = None, dtype: str = ""):
        if error:
            result["error"] = error
        self._results[name] = result
        self._dtypes[name] = dtype
        if self._csv_writer:
            for key, m in result.items():
                if not isinstance(m, dict):
                    continue
                self._csv_writer.writerow({
                    "case": name,
                    "dtype": dtype,
                    "key": key,
                    "pass": m.get("passed"),
                    "max_abs": m.get("max_abs", ""),
                    "mean_abs": m.get("mean_abs", ""),
                    "fail_ratio": m.get("fail_ratio", ""),
                    "error": error or "",
                })
            self._csv_file.flush()

    def print_interim(self, total: int):
        self._print_table(self._results, self._dtypes, total, is_final=False)

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def print_final(self) -> int:
        self.close()
        return self._print_table(self._results, self._dtypes, len(self._results), is_final=True)

    @staticmethod
    def _print_table(results: dict, dtypes: dict, total: int, is_final: bool) -> int:
        max_len = max((len(n) for n in results), default=8)
        width = max_len + 52
        SEP = "─" * width
        title = "  汇总结果" if is_final else f"  中间统计  ({len(results)}/{total})"
        print(f"\n┌{SEP}┐")
        print(f"│{title}")
        print(f"├{SEP}┤")
        print(f"│  {'Case':<{max_len}}  {'Dtype':>6}  {'Keys':<28}  {'Status':>6}")
        print(f"├{SEP}┤")
        pass_cnt = fail_cnt = 0
        for name, res in results.items():
            tags = []
            all_pass = True
            for k, v in res.items():
                if not isinstance(v, dict):
                    continue
                p = v.get("passed")
                tags.append(f"{k}:{'P' if p else 'F'}")
                if not p:
                    all_pass = False
            status = "PASS" if all_pass else "FAIL"
            print(f"│  {name:<{max_len}}  {dtypes.get(name, ''):>6}  {' '.join(tags):<28}  {status:>6}")
            if all_pass:
                pass_cnt += 1
            else:
                fail_cnt += 1
        print(f"├{SEP}┤")
        print(f"│  通过: {pass_cnt}   失败: {fail_cnt}   共: {len(results)}")
        print(f"└{SEP}┘")
        return fail_cnt

    @staticmethod
    def check_results(result: dict) -> None:
        import pytest
        for key, m in result.items():
            if not isinstance(m, dict):
                continue
            if not m.get("passed"):
                total = m.get("total", 0)
                fail_cnt = m.get("fail_cnt", 0)
                ratio = m.get("fail_ratio", float("nan"))
                pass_rate = (total - fail_cnt) / total * 100 if total > 0 else 0.0
                pytest.fail(
                    f"精度校验失败 key={key}, pass率={pass_rate:.2f}%, fail_ratio={ratio}")


def make_failure_result(error: str) -> dict:
    return {"out": {"passed": False, "max_abs": float("nan"), "mean_abs": float("nan"),
                    "fail_cnt": 0, "total": 0, "fail_ratio": float("nan")}, "error": error}


# ============ Helpers ============
def _pt_path(case_set: str, case_id: str, stage: str) -> Path:
    return Path("data") / case_set / case_id / f"{stage}.pt"


def _save_pt(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    torch.save(save, path)


def _load_pt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _to_device(data: dict, device: str) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def _run_prof_stage(stage_name, func, case_id, result_dir):
    try:
        func(case_id, [], str(Path(result_dir) / stage_name))
    except NotImplementedError:
        pass


def _params_summary(params: dict) -> str:
    keys = ["B", "N1", "N2", "S1", "S2", "D", "DV", "dtype",
            "layout_q", "layout_kv", "mask_mode"]
    return " ".join(f"{k}={params[k]}" for k in keys if k in params)


def _shapes_summary(data: dict) -> str:
    parts = []
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            parts.append(f"{k}={tuple(v.shape)}")
    return " ".join(parts) if parts else "(no tensors)"


def _exec_or_load(stages: set, stage_name: str, compute_fn, case_set: str,
                  case_id: str, extra: str = ""):
    """执行阶段(若在stages中)并保存pt, 否则从pt加载, pt不存在则返回None。
    内聚 start/done/skip 计时日志, extra 追加到 start 行 (如 npu_mode)。"""
    if stage_name in stages:
        if extra:
            logger.info("[%s]  start | %s", stage_name, extra)
        else:
            logger.info("[%s]  start", stage_name)
        t0 = time.perf_counter()
        output = compute_fn()
        logger.info("[%s]  done  | %.2fs | %s", stage_name, time.perf_counter() - t0,
                    _shapes_summary(output) if output else "(none)")
        _save_pt(output, _pt_path(case_set, case_id, stage_name))
        return output
    pt = _pt_path(case_set, case_id, stage_name)
    if pt.exists():
        output = _load_pt(pt)
        logger.info("[%s]  skip  | %s", stage_name,
                    _shapes_summary(output) if output else "(none)")
        return output
    logger.info("[%s]  skip  | pt not found", stage_name)
    return None


# ============ Stages ============
def _stage_gen(stages: set, params: dict, case_set: str, case_id: str):
    """生成输入。start 日志含参数摘要, 与统一格式不同, 故内联。"""
    if "gen" in stages:
        gen = GenDataBackend()
        logger.info("[gen]  start | %s", _params_summary(params))
        t0 = time.perf_counter()
        inputs = gen.gen_inputs(gen.normalize_params(params))
        logger.info("[gen]  done  | %.2fs | %s", time.perf_counter() - t0, _shapes_summary(inputs))
        _save_pt(inputs, _pt_path(case_set, case_id, "input"))
        return inputs
    logger.info("[gen]  skip  | loading from pt")
    return _load_pt(_pt_path(case_set, case_id, "input"))


def _stage_cpu(stages: set, inputs: dict, case_set: str, case_id: str):
    """CPU 计算。"""
    cpu = CpuBackend()
    return _exec_or_load(stages, "cpu", lambda: cpu.cpu_compute(inputs),
                         case_set, case_id)


def _stage_npu(stages: set, inputs: dict, npu_mode: str, case_set: str, case_id: str):
    """NPU 计算 (extra=npu_mode 显示在 start 行)。"""
    npu = NpuBackend()
    return _exec_or_load(
        stages, "npu",
        lambda: npu.npu_compute(_to_device(inputs, "npu"), npu_mode),
        case_set, case_id, extra=npu_mode)


def _stage_gpu(stages: set, inputs: dict, case_set: str, case_id: str):
    """GPU 计算 (仅 stages 时执行+save, 不加载; 否则返回 None)。"""
    if "gpu" not in stages:
        return None
    from core.gpu import GpuBackend
    gpu = GpuBackend()
    logger.info("[gpu]  start")
    t0 = time.perf_counter()
    gpu_out = gpu.gpu_compute(_to_device(inputs, "cuda"))
    logger.info("[gpu]  done  | %.2fs | %s", time.perf_counter() - t0, _shapes_summary(gpu_out))
    _save_pt(gpu_out, _pt_path(case_set, case_id, "gpu"))
    return gpu_out


def _stage_compare(stages: set, cpu_out, npu_out, gpu_out, params: dict,
                   reporter: "Reporter", case_id: str, result_dir: str,
                   verbose_diff, chunk_size: int, comparator: Comparator) -> dict:
    """精度比对 + 失败分析 + 记录。"""
    result = {"passed": True, "fail_msg": ""}
    if "compare" not in stages:
        return result
    if cpu_out is None or npu_out is None:
        raise RuntimeError(
            "compare 阶段缺少 cpu/npu 输出 (上游未执行且无 pt 缓存)")
    logger.info("[cmp]  start")
    t0 = time.perf_counter()
    if gpu_out is not None:
        result = comparator.compare_three_way(
            cpu_out, npu_out, gpu_out, chunk_size=chunk_size)
    else:
        result = comparator.compare_two_way(
            cpu_out, npu_out, chunk_size=chunk_size)
    logger.info("[cmp]  done  | %.2fs", time.perf_counter() - t0)
    for key, m in result.items():
        if isinstance(m, dict):
            status = "PASS" if m.get("passed") else "FAIL"
            logger.info("[cmp]  %s | %s: max_abs=%.6f fail_ratio=%.4f%%",
                        status, key, m.get("max_abs", 0), m.get("fail_ratio", 0) * 100)
    result["fail_msg"] = "" if result.get("passed", False) else "compare failed"
    if not result.get("passed", False):
        layout = params.get("layout_out", "")
        print(analyze_failure(cpu_out, npu_out, result,
                              case_dir=result_dir, verbose=verbose_diff,
                              layout=layout))
    reporter.record(case_id, result, dtype=params.get("dtype", ""))
    return result


def _run_profs(stages: set, case_id: str, result_dir: str):
    """prof 阶段 (接口未定, 暂为 NotImplementedError 占位)。"""
    if "npuprof" in stages:
        from prof.npuprof import run_msprof
        _run_prof_stage("npuprof", run_msprof, case_id, result_dir)
    if "npuprofop" in stages:
        from prof.npuprofop import run_msprof_op
        _run_prof_stage("npuprofop", run_msprof_op, case_id, result_dir)
    if "gpuprof" in stages:
        from prof.gpuprof import run_gpuprof
        _run_prof_stage("gpuprof", run_gpuprof, case_id, result_dir)
    if "l2_flush" in stages:
        from prof.l2_flush import flush_l2_cache
        try:
            flush_l2_cache()
        except NotImplementedError:
            pass


# ============ Engine ============
def run_case(stages: set, params: dict, reporter: Reporter, *,
             npu_mode: str = "eager",
             chunk_size: int = 0, verbose_diff: str = None,
             comparator: Comparator = None) -> dict:
    case_set = params.get("case_set", "")
    case_id = params.get("case_id", params.get("name", ""))
    result_dir = str(Path("result") / case_id)
    dtype = params.get("dtype", "")
    if comparator is None:
        comparator = DefaultComparator()
    result = {"passed": True, "fail_msg": ""}
    inputs = cpu_out = npu_out = gpu_out = None
    try:
        inputs = _stage_gen(stages, params, case_set, case_id)
        cpu_out = _stage_cpu(stages, inputs, case_set, case_id)
        npu_out = _stage_npu(stages, inputs, npu_mode, case_set, case_id)
        gpu_out = _stage_gpu(stages, inputs, case_set, case_id)
        result = _stage_compare(stages, cpu_out, npu_out, gpu_out, params,
                                reporter, case_id, result_dir,
                                verbose_diff, chunk_size, comparator)
        _run_profs(stages, case_id, result_dir)
        return result
    except Exception as e:
        logger.error("[err]  %s: %s", type(e).__name__, e)
        traceback.print_exc()
        fail = make_failure_result(str(e))
        fail["passed"] = False
        fail["fail_msg"] = str(e)
        reporter.record(case_id, fail, error=str(e), dtype=dtype)
        return fail
    finally:
        if "npu" in stages:
            torch.npu.empty_cache()
        if "gpu" in stages:
            torch.cuda.empty_cache()
        for t in (inputs, cpu_out, npu_out, gpu_out):
            if t is not None:
                del t
        gc.collect()
