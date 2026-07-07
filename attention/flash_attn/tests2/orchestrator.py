import csv
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from compare import compare_outputs
from core.gen_data import GenDataBackend
from core.cpu import CpuBackend
from core.npu import NpuBackend
from csv_parser import CSV_HEADERS
from precision_analyze import analyze_failure


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

    def print_final(self) -> int:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
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


# ============ Engine ============
def run_case(stages: set, params: dict, reporter: Reporter, *,
             max_retries: int = 0, load_pt: str = None, graph_path: int = 0,
             chunk_size: int = 0, num_workers: int = 0, verbose_diff: bool = False,
             fail_analysis: bool = True, keep_pt: bool = True) -> dict:
    case_set = params.get("case_set", "")
    case_id = params.get("case_id", params.get("name", ""))
    result_dir = str(Path("result") / case_id)
    dtype = params.get("dtype", "")
    result = {"passed": True, "fail_msg": ""}
    try:
        gen = GenDataBackend()
        cpu = CpuBackend()
        npu = NpuBackend()

        if "gen" in stages:
            if load_pt:
                inputs = gen.load_inputs(load_pt)
            else:
                inputs = gen.gen_inputs(gen.normalize_params(params))
            if keep_pt:
                _save_pt(inputs, _pt_path(case_set, case_id, "input"))
        else:
            inputs = _load_pt(_pt_path(case_set, case_id, "input"))

        if "cpu" in stages:
            cpu_out = cpu.cpu_compute(inputs)
            if keep_pt:
                _save_pt(cpu_out, _pt_path(case_set, case_id, "cpu"))
        else:
            cpu_pt = _pt_path(case_set, case_id, "cpu")
            cpu_out = _load_pt(cpu_pt) if cpu_pt.exists() else None

        if "npu" in stages:
            dev_inputs = _to_device(inputs, "npu")
            with ThreadPoolExecutor(max_workers=1) as executor:
                for attempt in range(max_retries + 1):
                    try:
                        future = executor.submit(npu.npu_compute, dev_inputs, graph_path)
                        npu_out = future.result()
                        break
                    except Exception:
                        if attempt < max_retries:
                            continue
                        raise
            if keep_pt:
                _save_pt(npu_out, _pt_path(case_set, case_id, "npu"))
            del dev_inputs
        else:
            npu_pt = _pt_path(case_set, case_id, "npu")
            npu_out = _load_pt(npu_pt) if npu_pt.exists() else None

        if "gpu" in stages:
            from core.gpu import GpuBackend
            gpu = GpuBackend()
            dev_inputs = _to_device(inputs, "cuda")
            gpu_out = gpu.gpu_compute(dev_inputs)
            if keep_pt:
                _save_pt(gpu_out, _pt_path(case_set, case_id, "gpu"))
            del dev_inputs

        if "compare" in stages and cpu_out is not None and npu_out is not None:
            result = compare_outputs(cpu_out, npu_out,
                                     chunk_size=chunk_size, num_workers=num_workers)
            result["fail_msg"] = "" if result.get("passed", False) else "compare failed"
            if not result.get("passed", False) and fail_analysis:
                print(analyze_failure(cpu_out, npu_out, result,
                                      case_dir=result_dir, verbose=verbose_diff))
            reporter.record(case_id, result, dtype=dtype)

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

        if "npu" in stages:
            torch.npu.empty_cache()
        if "gpu" in stages:
            torch.cuda.empty_cache()
        del inputs, cpu_out, npu_out
        gc.collect()
        return result
    except Exception as e:
        traceback.print_exc()
        fail = make_failure_result(str(e))
        fail["passed"] = False
        fail["fail_msg"] = str(e)
        reporter.record(case_id, fail, error=str(e), dtype=dtype)
        return fail
