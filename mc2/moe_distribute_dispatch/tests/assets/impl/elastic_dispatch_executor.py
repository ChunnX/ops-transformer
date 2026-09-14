#!/usr/bin/env python3

import base64
import json
import logging
import os
import fcntl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

import importlib.util

_SHARED_STATE = None
_PREPARE_LOCK = threading.Lock()
_MOE_CACHE_DIR = None


def run_elastic_dispatch_warmup(
    thread_contexts, device_ids, main_device, stream_value, plugin_path
):
    global _SHARED_STATE
    if (
        _SHARED_STATE is not None
        and _SHARED_STATE.get("payloads") is not None
        and tuple(_SHARED_STATE.get("device_ids", ())) == tuple(device_ids)
    ):
        return
    module_path = Path(__file__).parents[4] / "common" / "tests" / "assets" / "impl" / "moe_elastic_prelude.py"
    spec = importlib.util.spec_from_file_location(
        "mc2_moe_elastic_dispatch_prelude", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_moe_elastic_warmup(
        thread_contexts, device_ids, main_device, stream_value, plugin_path
    )
    _SHARED_STATE = {
        "contexts": thread_contexts,
        "device_ids": tuple(device_ids),
        "ready": set(),
        "completed": threading.Event(),
        "payloads": None,
        "run_dir": None,
    }


def prepare_elastic_dispatch(thread_contexts, device_ids):
    global _SHARED_STATE
    first_context = next(iter(thread_contexts.values()))
    if int(first_context.attributes.get("_moeDeepep", 0) or 0) != 1:
        return
    lock_path = Path(tempfile.gettempdir()) / "ttk_moe_elastic_dispatch.prepare.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if _SHARED_STATE is not None and _SHARED_STATE.get("payloads") is not None:
            return _SHARED_STATE["payloads"]
    global _MOE_CACHE_DIR
    if _MOE_CACHE_DIR is None:
        cache_dir_env = os.environ.get("TTK_MOE_CACHE_DIR")
        if cache_dir_env:
            _MOE_CACHE_DIR = Path(cache_dir_env).expanduser()
        else:
            from ttk.config.loader import get_moe_config

            _MOE_CACHE_DIR = Path(get_moe_config().cache_dir).expanduser()
        _MOE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    moe_cache_dir = _MOE_CACHE_DIR
    run_dir = tempfile.mkdtemp(prefix="ttk_moe_elastic_", dir=str(moe_cache_dir))
    try:
        payloads = _run_external_worker(thread_contexts, tuple(device_ids), run_dir)
    except BaseException:
        failure_log = Path(run_dir) / "launcher_failure.log"
        failure_log.write_text(
            f"Worker run directory retained at: {run_dir}\n", encoding="utf-8"
        )
        raise
    _SHARED_STATE = {
        "contexts": thread_contexts,
        "device_ids": tuple(device_ids),
        "ready": set(),
        "completed": threading.Event(),
        "payloads": payloads,
        "run_dir": run_dir,
    }
    return payloads


def _is_elastic(context):
    return int(context.attributes.get("_moeDeepep", 0) or 0) == 1


def _release_main_resources(thread_contexts, device_ids):
    from ttk.core_modules.npu.op_api.profiling import (
        release_retained_multi_device_resources,
    )

    first_context = next(iter(thread_contexts.values()))
    release_retained_multi_device_resources(first_context, list(device_ids))


def _load_profiling_result_class():
    from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult

    return ApiProfilingResult


def _tensor_bytes(tensor):
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return value.numpy().tobytes()


def _hccl_buffer_size(context):
    extension_root = (
        Path(__file__).resolve().parents[5]
        / "torch_extension"
        / "build"
        / "lib"
    )
    sys.path.insert(0, str(extension_root))
    from cann_ops_transformer_mc2moe.ops import MoeDistributeBuffer

    attributes = context.attributes
    expert_ids = context.flatten_tensors[1]
    world_size = int(attributes["epWorldSize"])
    local_batch_size = int(attributes.get("globalBs", 1)) // world_size
    top_k = int(
        attributes.get(
            "_moeDeepepOriginalTopK",
            expert_ids.shape[-1] if expert_ids.dim() > 1 else 1,
        )
    )
    return MoeDistributeBuffer.get_low_latency_ccl_buffer_size(
        world_size,
        local_batch_size,
        int(context.flatten_tensors[0].shape[-1]),
        int(attributes["moeExpertNum"]),
        top_k,
        int(attributes.get("sharedExpertNum", 1)),
        int(attributes.get("sharedExpertRankNum", 0)),
        str(attributes.get("commAlg", "")),
    )


def _run_external_worker(thread_contexts, device_ids, run_dir):
    global _MOE_CACHE_DIR
    if _MOE_CACHE_DIR is None:
        cache_dir_env = os.environ.get("TTK_MOE_CACHE_DIR")
        if cache_dir_env:
            _MOE_CACHE_DIR = Path(cache_dir_env).expanduser()
        else:
            from ttk.config.loader import get_moe_config

            _MOE_CACHE_DIR = Path(get_moe_config().cache_dir).expanduser()
        _MOE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    configured_ranks = next(iter(thread_contexts.values())).attributes.get(
        "_moeElasticAvailableRanks"
    )
    elastic_info = next(iter(thread_contexts.values())).attributes.get(
        "_moeDeepepElasticInfo"
    )
    available_ranks = sorted(
        range(len(device_ids)) if configured_ranks is None else map(int, configured_ranks)
    )
    hccl_buffer_size = max(
        int(os.environ.get("HCCL_BUFFSIZE", "0") or 0),
        _hccl_buffer_size(thread_contexts[device_ids[0]]),
    )
    jobs = []
    endpoint = f"tcp://127.0.0.1:{_allocate_rendezvous_port()}"
    for rank_idx, device_id in enumerate(device_ids):
        context = thread_contexts[device_id]
        plan_path = Path(run_dir) / f"plan_rank_{rank_idx}.json"
        output_path = Path(run_dir) / f"output_rank_{rank_idx}.npz"
        plan = {
            "outputPath": str(output_path),
            "availableRanks": available_ranks,
            "x": base64.b64encode(_tensor_bytes(context.flatten_tensors[0])).decode("ascii"),
            "inputDtype": str(context.flatten_tensors[0].dtype).replace("torch.", ""),
            "expertIds": base64.b64encode(
                _tensor_bytes(context.flatten_tensors[1].view(torch.int32))
            ).decode("ascii"),
            "xActiveMask": base64.b64encode(
                _tensor_bytes(context.flatten_tensors[3])
            ).decode("ascii"),
            "xShape": list(context.flatten_tensors[0].shape),
            "expertIdsShape": list(context.flatten_tensors[1].shape),
            "scalesShape": (
                list(context.flatten_tensors[2].shape)
                if context.flatten_tensors[2] is not None
                else []
            ),
            "xActiveMaskShape": list(context.flatten_tensors[3].shape),
            "expertIdsExpand": base64.b64encode(
                _tensor_bytes(context.attributes["_moeDeepepExpertIdsExpand"])
            ).decode("ascii"),
            "expertIdsExpandShape": list(
                context.attributes["_moeDeepepExpertIdsExpand"].shape
            ),
            "scales": (
                base64.b64encode(
                    _tensor_bytes(context.flatten_tensors[2])
                ).decode("ascii")
                if context.flatten_tensors[2] is not None
                else ""
            ),
            "elasticInfo": (
                elastic_info.detach().cpu().numpy().astype(np.int32).tolist()
                if elastic_info is not None
                else []
            ),
            "attributes": {
                key: value
                for key, value in context.attributes.items()
                if key in {
                    "epWorldSize", "tpWorldSize", "moeExpertNum", "expertShardType", "isExpand",
                    "sharedExpertNum", "sharedExpertRankNum", "globalBs", "quantMode",
                    "zeroExpertNum", "copyExpertNum", "constExpertNum", "isElastic",
                    "expertTokenNumsType", "commAlg", "quantMode",
                }
            },
        }
        with open(plan_path, "w", encoding="utf-8") as plan_file:
            json.dump(plan, plan_file)
        jobs.append((rank_idx, device_id, output_path))

    env = os.environ.copy()
    env.pop("HCCL_NPU_SOCKET_PORT_RANGE", None)
    env["HCCL_BUFFSIZE"] = str(hccl_buffer_size)
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["MAX_JOBS"] = "1"
    env["TORCH_EXTENSIONS_DIR"] = str(_MOE_CACHE_DIR / "torch_extensions")
    extension_root = str(
        Path(__file__).resolve().parents[5]
        / "torch_extension"
        / "build"
        / "lib"
    )
    env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (extension_root, env.get("PYTHONPATH", ""))
        if path
    )
    launcher = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("dispatch_elastic_launcher.py")),
            run_dir,
            endpoint,
            str(Path(__file__).with_name("dispatch_elastic_worker.py")),
            str(len(device_ids)),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        output, _ = launcher.communicate(timeout=900)
    except subprocess.TimeoutExpired:
        launcher.kill()
        output, _ = launcher.communicate()
        raise RuntimeError(
            "Moe elastic Dispatch launcher timed out after 900s; "
            "generated worker stdout/stderr:\n"
            f"{output}"
        ) from None

    if output:
        print(output, flush=True)
    if launcher.returncode != 0:
        raise RuntimeError(
            f"Moe elastic Dispatch forkserver launcher failed rc={launcher.returncode}"
        ) from RuntimeError(_worker_failure_details(run_dir, jobs))

    payloads = {}
    for rank_idx, device_id, output_path in jobs:
        if not output_path.is_file() or output_path.stat().st_size == 0:
            payloads[device_id] = None
            continue
        with np.load(output_path, allow_pickle=False) as payload:
            flat_output_shapes = payload["output_shapes"].tolist()
            output_dims = payload["output_dims"].tolist()
            output_shapes = []
            shape_offset = 0
            for output_dim in output_dims:
                output_shapes.append(
                    tuple(flat_output_shapes[shape_offset : shape_offset + output_dim])
                )
                shape_offset += output_dim
            payloads[device_id] = (
                [payload[f"output_{index}"].tobytes() for index in range(7)],
                output_shapes,
                int(payload["rank"]),
            )
    return payloads


def _worker_failure_details(run_dir, jobs):
    details = []
    for rank_idx, _, _ in jobs:
        log_path = Path(run_dir) / f"worker_rank_{rank_idx}.log"
        if not log_path.is_file():
            continue
        details.append(f"rank={rank_idx}\n{log_path.read_text(encoding='utf-8')}")
    return "\n".join(details)


def _allocate_rendezvous_port():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def execute_elastic_dispatch(thread_context, device, stream_value):
    if not _is_elastic(thread_context):
        return None

    state = _SHARED_STATE
    if state is None:
        raise RuntimeError("Moe elastic Dispatch shared execution state is missing")
    available_ranks = set(
        int(rank)
        for rank in state["contexts"][state["device_ids"][0]].attributes.get(
            "_moeElasticAvailableRanks", state["device_ids"]
        )
    )
    expected_ready = len(available_ranks)
    coordinator_rank = min(available_ranks)
    state["ready"].add(thread_context.device_ids.index(thread_context.my_rank))
    rank_idx = thread_context.device_ids.index(thread_context.my_rank)
    if rank_idx == coordinator_rank:
        try:
            deadline = time.monotonic() + 1800
            while len(state["ready"]) < expected_ready and time.monotonic() < deadline:
                time.sleep(0.01)
            if len(state["ready"]) < expected_ready:
                raise TimeoutError("Moe elastic Dispatch ranks were not ready")
            if state["run_dir"] is None:
                state["run_dir"] = tempfile.TemporaryDirectory(
                    prefix="ttk_moe_elastic_", dir=str(moe_cache_dir)
                )
            state["payloads"] = _run_external_worker(
        state["contexts"], state["device_ids"], state["run_dir"]
            )
        except BaseException:
            state["payloads"] = None
            raise
        finally:
            state["completed"].set()
    else:
        if not state["completed"].wait(1800):
            raise TimeoutError("Moe elastic Dispatch coordinator did not finish")

    payload = state["payloads"][thread_context.device_ids.index(thread_context.my_rank)]
    if payload is None:
        from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult

        return ApiProfilingResult.fail("SUPPRESSED")
    ApiProfilingResult = _load_profiling_result_class()
    output_bytes, output_shapes, payload_rank = payload
    if payload_rank != rank_idx:
        raise RuntimeError(f"Moe elastic Dispatch payload rank mismatch: {payload_rank}")
    expected_output_shapes = tuple(thread_context.flat_output_storage_shapes)
    effective_output_shapes = []
    for output_index, output_shape in enumerate(expected_output_shapes):
        expected_element_count = max(1, int(np.prod(output_shape)))
        dtype_name = str(thread_context.flat_output_dtypes[output_index])
        dtype_width = {
            "bool": 1,
            "int8": 1,
            "uint8": 1,
            "float16": 2,
            "bfloat16": 2,
            "int16": 2,
            "uint16": 2,
            "float32": 4,
            "int32": 4,
            "uint32": 4,
            "float64": 8,
            "int64": 8,
            "uint64": 8,
        }.get(dtype_name, 1)
        if len(output_bytes[output_index]) == 0 and expected_element_count > 1:
            output_bytes[output_index] = bytes(expected_element_count * dtype_width)
            effective_output_shapes.append(output_shape)
        else:
            effective_output_shapes.append(output_shapes[output_index])
    effective_output_shapes = tuple(effective_output_shapes)
    thread_context._output_storage_shapes = effective_output_shapes
    thread_context._output_view_shapes = effective_output_shapes
    logging.info(
        "Moe elastic Dispatch output contract rank=%s details=%s",
        rank_idx,
        [
            (
                output_index,
                effective_output_shape,
                str(thread_context.flat_output_dtypes[output_index]),
                len(output_bytes[output_index]),
            )
            for output_index, effective_output_shape in enumerate(effective_output_shapes)
        ],
    )
    return ApiProfilingResult(
        True,
        "E2E_EXECUTED",
        "E2E_EXECUTED",
        list(output_bytes),
        list(effective_output_shapes),
    )
