import gc
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from core.gen_data import load as load_gen
from core.cpu import load as load_cpu
from core.npu import load as load_npu
from core.gpu import load as load_gpu
from core.compare import compare_outputs
from core.reporter import Reporter, make_failure_result
from precision_analyze import analyze_failure


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


def run_case(stages: set, op_name: str, params: dict, reporter: Reporter, *,
             max_retries: int = 0, load_pt: str = None, graph_path: int = 0,
             chunk_size: int = 0, num_workers: int = 0, verbose_diff: bool = False,
             fail_analysis: bool = True, keep_pt: bool = True) -> dict:
    case_set = params.get("case_set", op_name)
    case_id = params.get("case_id", params.get("name", op_name))
    result_dir = str(Path("result") / case_id)
    dtype = params.get("dtype", "")
    try:
        gen = load_gen(op_name)
        cpu = load_cpu(op_name)
        npu = load_npu(op_name)

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
            cpu_out = _load_pt(_pt_path(case_set, case_id, "cpu"))

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
            npu_out = _load_pt(_pt_path(case_set, case_id, "npu"))

        if "gpu" in stages:
            gpu = load_gpu(op_name)
            dev_inputs = _to_device(inputs, "cuda")
            gpu_out = gpu.gpu_compute(dev_inputs)
            if keep_pt:
                _save_pt(gpu_out, _pt_path(case_set, case_id, "gpu"))
            del dev_inputs

        result = {"passed": True, "fail_msg": ""}
        if "compare" in stages:
            result = compare_outputs(cpu_out, npu_out,
                                     chunk_size=chunk_size, num_workers=num_workers)
            result["fail_msg"] = "" if result.get("passed", False) else "compare failed"
            if not result.get("passed", False) and fail_analysis:
                print(analyze_failure(cpu_out, npu_out, result,
                                      case_dir=result_dir, verbose=verbose_diff))
            reporter.record(case_id, result, dtype=dtype)

        if "npuprof" in stages:
            from prof.npuprof import run_msprof
            run_msprof(case_id, [], str(Path(result_dir) / "npuprof"))
        if "npuprofop" in stages:
            from prof.npuprofop import run_msprof_op
            run_msprof_op(case_id, [], str(Path(result_dir) / "npuprofop"))
        if "gpuprof" in stages:
            from prof.gpuprof import run_gpuprof
            run_gpuprof(case_id, [], str(Path(result_dir) / "gpuprof"))

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
