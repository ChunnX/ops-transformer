from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from golden import (
    CombineGolden,
    DispatchGolden,
    Feature,
    GoldenHandle,
    feature_main,
    feature_scales,
    is_tuple_feature,
)


def dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "fp8_e5m2": getattr(torch, "float8_e5m2", None),
        "float8_e5m2": getattr(torch, "float8_e5m2", None),
        "fp8_e8m0": torch.uint8,
        "float8_e8m0": torch.uint8,
        "float8_e8m0fnu": torch.uint8,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }
    dtype = mapping.get(dtype_name)
    if dtype is None:
        raise ValueError(f"不支持的 dtype: {dtype_name}")
    return dtype


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def tensor_storage_dtype(dtype: torch.dtype) -> np.dtype:
    if dtype == torch.bfloat16:
        return np.dtype("int16")
    if "float8" in dtype_name(dtype):
        return np.dtype("uint8")
    if dtype == torch.float16:
        return np.dtype("float16")
    if dtype == torch.float32:
        return np.dtype("float32")
    if dtype == torch.int32:
        return np.dtype("int32")
    if dtype == torch.int64:
        return np.dtype("int64")
    if dtype == torch.uint8:
        return np.dtype("uint8")
    raise ValueError(f"暂不支持缓存 dtype: {dtype}")


def tensor_to_storage_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy()
    if "float8" in dtype_name(tensor.dtype):
        return tensor.view(torch.uint8).numpy()
    return tensor.numpy()


def tensor_from_storage_numpy(array: np.ndarray, item: Dict[str, Any]) -> torch.Tensor:
    dtype = dtype_from_name(item["dtype"])
    shape = item["shape"]
    if dtype == torch.bfloat16:
        return torch.from_numpy(array.astype(np.int16, copy=False)).view(torch.bfloat16).reshape(shape)
    if "float8" in item["dtype"]:
        return torch.from_numpy(array.astype(np.uint8, copy=False)).view(dtype).reshape(shape)
    return torch.from_numpy(array).to(dtype).reshape(shape)


def file_md5(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def script_md5s(paths: List[Path]) -> Dict[str, str]:
    return {str(path): file_md5(path) for path in paths}


class GoldenCacheWriter:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.tensors: Dict[str, Dict[str, Any]] = {}

    def add_tensor(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        safe_name = name.replace("/", "_")
        file_name = f"{safe_name}.bin"
        tensor_to_storage_numpy(tensor).tofile(self.cache_dir / file_name)
        item = {
            "file": file_name,
            "shape": list(tensor.shape),
            "dtype": dtype_name(tensor.dtype),
            "storage_dtype": str(tensor_storage_dtype(tensor.dtype)),
        }
        self.tensors[name] = item
        return {"tensor_ref": name}


def feature_to_state(writer: GoldenCacheWriter, feature: Feature, name: str) -> Dict[str, Any]:
    if is_tuple_feature(feature):
        return {
            "kind": "tuple",
            "x": writer.add_tensor(feature_main(feature), f"{name}/x"),
            "scales": writer.add_tensor(feature_scales(feature), f"{name}/scales"),
        }
    return {"kind": "tensor", "x": writer.add_tensor(feature_main(feature), f"{name}/x")}


def handle_to_state(writer: GoldenCacheWriter, handle: GoldenHandle, name: str) -> Dict[str, Any]:
    return {
        "dst_buffer_slot_idx": writer.add_tensor(handle.dst_buffer_slot_idx, f"{name}/dst_buffer_slot_idx"),
        "recv_src_metadata": writer.add_tensor(handle.recv_src_metadata, f"{name}/recv_src_metadata"),
        "num_recv_payloads_per_rank": writer.add_tensor(
            handle.num_recv_payloads_per_rank, f"{name}/num_recv_payloads_per_rank"
        ),
        "num_recv_tokens_per_expert": writer.add_tensor(
            handle.num_recv_tokens_per_expert, f"{name}/num_recv_tokens_per_expert"
        ),
        "topk_idx": writer.add_tensor(handle.topk_idx, f"{name}/topk_idx"),
        "num_experts": handle.num_experts,
        "expert_alignment": handle.expert_alignment,
        "num_max_tokens_per_rank": handle.num_max_tokens_per_rank,
        "num_expanded_tokens": handle.num_expanded_tokens,
    }


def dispatch_to_state(writer: GoldenCacheWriter, dispatch: DispatchGolden, name: str) -> Dict[str, Any]:
    return {
        "recv_x": feature_to_state(writer, dispatch.recv_x, f"{name}/recv_x"),
        "recv_topk_idx": None,
        "recv_topk_weights": (
            None if dispatch.recv_topk_weights is None
            else writer.add_tensor(dispatch.recv_topk_weights, f"{name}/recv_topk_weights")
        ),
        "handle": handle_to_state(writer, dispatch.handle, f"{name}/handle"),
    }


def combine_to_state(writer: GoldenCacheWriter, combine: CombineGolden, name: str) -> Dict[str, Any]:
    return {
        "combined_x": feature_to_state(writer, combine.combined_x, f"{name}/combined_x"),
        "combined_topk_weights": (
            None if combine.combined_topk_weights is None
            else writer.add_tensor(combine.combined_topk_weights, f"{name}/combined_topk_weights")
        ),
    }


def load_tensor(cache_dir: Path, tensors: Dict[str, Dict[str, Any]], ref: Dict[str, str]) -> torch.Tensor:
    item = tensors[ref["tensor_ref"]]
    array = np.fromfile(cache_dir / item["file"], dtype=np.dtype(item["storage_dtype"]))
    return tensor_from_storage_numpy(array, item)


def feature_from_state(cache_dir: Path, tensors: Dict[str, Dict[str, Any]], state: Dict[str, Any]) -> Feature:
    if state["kind"] == "tuple":
        return (
            load_tensor(cache_dir, tensors, state["x"]),
            load_tensor(cache_dir, tensors, state["scales"]),
        )
    return load_tensor(cache_dir, tensors, state["x"])


def handle_from_state(cache_dir: Path, tensors: Dict[str, Dict[str, Any]], state: Dict[str, Any]) -> GoldenHandle:
    return GoldenHandle(
        dst_buffer_slot_idx=load_tensor(cache_dir, tensors, state["dst_buffer_slot_idx"]),
        recv_src_metadata=load_tensor(cache_dir, tensors, state["recv_src_metadata"]),
        num_recv_payloads_per_rank=load_tensor(cache_dir, tensors, state["num_recv_payloads_per_rank"]),
        num_recv_tokens_per_expert=load_tensor(cache_dir, tensors, state["num_recv_tokens_per_expert"]),
        topk_idx=load_tensor(cache_dir, tensors, state["topk_idx"]),
        num_experts=int(state["num_experts"]),
        expert_alignment=int(state["expert_alignment"]),
        num_max_tokens_per_rank=int(state["num_max_tokens_per_rank"]),
        num_expanded_tokens=int(state["num_expanded_tokens"]),
    )


def dispatch_from_state(cache_dir: Path, tensors: Dict[str, Dict[str, Any]], state: Dict[str, Any]) -> DispatchGolden:
    return DispatchGolden(
        recv_x=feature_from_state(cache_dir, tensors, state["recv_x"]),
        recv_topk_idx=None,
        recv_topk_weights=(
            None if state["recv_topk_weights"] is None
            else load_tensor(cache_dir, tensors, state["recv_topk_weights"])
        ),
        handle=handle_from_state(cache_dir, tensors, state["handle"]),
    )


def combine_from_state(cache_dir: Path, tensors: Dict[str, Dict[str, Any]], state: Dict[str, Any]) -> CombineGolden:
    return CombineGolden(
        combined_x=feature_from_state(cache_dir, tensors, state["combined_x"]),
        combined_topk_weights=(
            None if state["combined_topk_weights"] is None
            else load_tensor(cache_dir, tensors, state["combined_topk_weights"])
        ),
    )


def save_case_cache(
    cache_root: Path,
    case_name: str,
    case_data: Dict[str, Any],
    params: Dict[str, Any],
    script_paths: List[Path],
) -> None:
    cache_dir = cache_root / case_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    writer = GoldenCacheWriter(cache_dir)
    state = {
        "params": params,
        "script_md5s": script_md5s(script_paths),
        "x_per_rank": [feature_to_state(writer, x, f"inputs/x_rank{rank}") for rank, x in enumerate(case_data["x_per_rank"])],
        "topk_idx_per_rank": [
            writer.add_tensor(t, f"inputs/topk_idx_rank{rank}") for rank, t in enumerate(case_data["topk_idx_per_rank"])
        ],
        "topk_weights_per_rank": [
            writer.add_tensor(t, f"inputs/topk_weights_rank{rank}")
            for rank, t in enumerate(case_data["topk_weights_per_rank"])
        ],
        "dispatch_goldens": [
            dispatch_to_state(writer, d, f"dispatch/rank{rank}") for rank, d in enumerate(case_data["dispatch_goldens"])
        ],
        "combine_goldens": [
            combine_to_state(writer, c, f"combine/rank{rank}") for rank, c in enumerate(case_data["combine_goldens"])
        ],
        "tensors": writer.tensors,
    }
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_case_cache(
    cache_root: Path,
    case_name: str,
    params: Dict[str, Any],
    script_paths: List[Path],
    *,
    enabled: bool = True,
    force_generate: bool = False,
) -> Optional[Dict[str, Any]]:
    if force_generate or not enabled:
        return None
    cache_dir = cache_root / case_name
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    if state.get("params") != params:
        print("[INFO] golden cache 参数变化，不复用")
        return None
    if state.get("script_md5s") != script_md5s(script_paths):
        print("[INFO] golden cache 脚本 md5 变化，不复用")
        return None
    tensors = state["tensors"]
    return {
        "x_per_rank": [feature_from_state(cache_dir, tensors, s) for s in state["x_per_rank"]],
        "topk_idx_per_rank": [load_tensor(cache_dir, tensors, s) for s in state["topk_idx_per_rank"]],
        "topk_weights_per_rank": [load_tensor(cache_dir, tensors, s) for s in state["topk_weights_per_rank"]],
        "dispatch_goldens": [dispatch_from_state(cache_dir, tensors, s) for s in state["dispatch_goldens"]],
        "combine_goldens": [combine_from_state(cache_dir, tensors, s) for s in state["combine_goldens"]],
    }
