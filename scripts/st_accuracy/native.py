from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .adapters import Adapter
from .st_cases import CaseResult, StCase


ACL_DTYPE_ATTR_TO_NAME = {
    0: "fp32",
    1: "fp16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    9: "int64",
    11: "fp64",
    12: "bool",
    27: "bf16",
}


class NativeRunner:
    def __init__(self, repo_root: Path, out_dir: Path, vendor: str, soc: str, device: int):
        self.repo_root = repo_root
        self.out_dir = out_dir
        self.vendor = vendor
        self.soc = soc
        self.device = device
        self.source = repo_root / "scripts" / "st_accuracy" / "native" / "st_accuracy_aclnn_runner.cpp"
        self.binary = out_dir / "native" / "st_accuracy_aclnn_runner"

    def run(self, case: StCase, adapter: Adapter) -> CaseResult:
        return self.run_many([(case, adapter)])[0]

    def run_many(self, items: list[tuple[StCase, Adapter]]) -> list[CaseResult]:
        if not items:
            return []

        start = time.monotonic()
        try:
            self._ensure_built()
            batch_file = self._write_batch_file(items)
            completed = subprocess.run(
                [
                    str(self.binary),
                    "--batch-file",
                    str(batch_file),
                    "--device",
                    str(self.device),
                    "--opapi-lib",
                    str(self._vendor_opapi_lib()),
                ],
                cwd=self.repo_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 - runner errors must become report rows.
            return [case.to_result(status="error", message=str(exc)) for case, _adapter in items]

        fallback_duration_ms = (time.monotonic() - start) * 1000.0
        try:
            payloads = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        except json.JSONDecodeError:
            message = completed.stderr.strip() or f"invalid native output: {completed.stdout!r}"
            return [case.to_result(status="error", message=message, duration_ms=fallback_duration_ms) for case, _adapter in items]

        if not payloads:
            message = completed.stderr.strip() or "native runner produced no output"
            status = "error" if completed.returncode != 0 else "failed"
            return [case.to_result(status=status, message=message, duration_ms=fallback_duration_ms) for case, _adapter in items]

        by_index = {int(payload.get("case_index", -1)): payload for payload in payloads}
        results: list[CaseResult] = []
        for index, (case, _adapter) in enumerate(items):
            payload = by_index.get(index)
            if payload is None:
                results.append(case.to_result(status="error", message="missing native batch result", duration_ms=fallback_duration_ms))
                continue
            status = str(payload.pop("status", "passed"))
            payload.pop("case_index", None)
            duration_ms = float(payload.pop("duration_ms", fallback_duration_ms))
            message = str(payload.pop("message", ""))
            results.append(case.to_result(status=status, message=message, duration_ms=duration_ms, **payload))
        return results

    def _ensure_built(self) -> None:
        if self.binary.is_file() and self.binary.stat().st_mtime >= self.source.stat().st_mtime:
            return
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        vendor_root = Path(ascend_home) / "opp" / "vendors" / f"{self.vendor}_transformer" / "op_api"
        include_path = Path(ascend_home) / "include"
        lib_path = Path(ascend_home) / "lib64"
        vendor_include = vendor_root / "include"
        vendor_lib = vendor_root / "lib"
        self.binary.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "g++",
            str(self.source),
            "-std=c++17",
            "-O2",
            "-Wno-psabi",
            "-I",
            str(include_path),
            "-I",
            str(include_path / "aclnnop"),
            "-I",
            str(vendor_include),
            "-L",
            str(vendor_lib),
            "-L",
            str(lib_path),
            "-lascendcl",
            "-lnnopbase",
            "-ldl",
            "-o",
            str(self.binary),
            f"-Wl,-rpath={vendor_lib}",
        ]
        subprocess.run(cmd, cwd=self.repo_root, check=True)

    def _write_batch_file(self, items: list[tuple[StCase, Adapter]]) -> Path:
        batch_file = self.out_dir / "native" / "batch_cases.tsv"
        batch_file.parent.mkdir(parents=True, exist_ok=True)
        with batch_file.open("w", encoding="utf-8") as handle:
            for index, (case, adapter) in enumerate(items):
                handle.write(self._case_batch_line(index, case, adapter))
                handle.write("\n")
        return batch_file

    def _case_batch_line(self, index: int, case: StCase, adapter: Adapter) -> str:
        inputs = case.flat_inputs
        if adapter.name in {"Cat", "Stack", "ChunkCat"}:
            tensor_list = [item for item in inputs if item.get("type") == "tensors"]
            fields = {
                "case_index": str(index),
                "api": adapter.name,
                "dtype": str(tensor_list[0]["dtype"]),
                "shapes": _tensor_list_shapes_arg(tensor_list),
                "ranges": _tensor_list_ranges_arg(tensor_list),
                "dim": _int_attr_arg(inputs, "dim"),
            }
            if adapter.name == "ChunkCat":
                fields["num_chunks"] = _int_attr_arg(inputs, "num_chunks")
            return "\t".join(f"{key}={value}" for key, value in fields.items())

        tensors = [item for item in inputs if item.get("type") == "tensor"]
        scalars = [item for item in inputs if item.get("type") == "scalar"]
        if adapter.name == "LightningIndexer":
            return _lightning_indexer_batch_line(index, inputs, tensors, adapter.name)
        if adapter.name == "RecurrentGatedDeltaRule":
            return _recurrent_gated_delta_rule_batch_line(index, inputs, tensors, adapter.name)
        self_tensor = _item_by_name(tensors, "self") or tensors[0]
        fields = {
            "case_index": str(index),
            "api": adapter.name,
            "dtype": str(self_tensor["dtype"]),
            "shape_a": _shape_arg(self_tensor.get("shape")),
            "range_a": _range_arg(self_tensor.get("range_values")),
        }
        if adapter.name in {"Add", "Sub"}:
            fields["shape_b"] = _shape_arg(tensors[1].get("shape"))
            fields["range_b"] = _range_arg(tensors[1].get("range_values"))
            fields["alpha"] = _scalar_arg(scalars[0].get("range_values"))
        elif adapter.name in {"Div", "Mul", "Maximum", "Minimum", "Ger"}:
            fields["shape_b"] = _shape_arg(tensors[1].get("shape"))
            fields["range_b"] = _range_arg(tensors[1].get("range_values"))
        elif adapter.name in {"Adds", "Subs"}:
            fields["other"] = _scalar_arg(scalars[0].get("range_values"))
            fields["alpha"] = _scalar_arg(scalars[1].get("range_values"))
        elif adapter.name in {"Muls", "Divs"}:
            fields["other"] = _scalar_arg(scalars[0].get("range_values"))
        elif adapter.name == "MaskedSelect":
            mask_tensor = _item_by_name(tensors, "mask") or tensors[1]
            fields["mask_shape"] = _shape_arg(mask_tensor.get("shape"))
        elif adapter.name in {"Hardtanh", "Clamp"}:
            fields["min"] = _named_scalar_arg(inputs, "clipValueMin")
            fields["max"] = _named_scalar_arg(inputs, "clipValueMax")
        elif adapter.name == "ClampMin":
            fields["min"] = _named_scalar_arg(inputs, "clipValueMin")
        elif adapter.name == "ClampMax":
            fields["max"] = _named_scalar_arg(inputs, "clipValueMax")
        elif adapter.name == "ClampTensor":
            min_tensor = _item_by_name(tensors, "clipValueMinTensor") or tensors[1]
            max_tensor = _item_by_name(tensors, "clipValueMaxTensor") or tensors[2]
            fields["dtype_b"] = str(min_tensor["dtype"])
            fields["shape_b"] = _shape_arg(min_tensor.get("shape"))
            fields["range_b"] = _range_arg(min_tensor.get("range_values"))
            fields["dtype_c"] = str(max_tensor["dtype"])
            fields["shape_c"] = _shape_arg(max_tensor.get("shape"))
            fields["range_c"] = _range_arg(max_tensor.get("range_values"))
        elif adapter.name == "ClampMinTensor":
            min_tensor = _item_by_name(tensors, "clipValueMinTensor") or tensors[1]
            fields["dtype_b"] = str(min_tensor["dtype"])
            fields["shape_b"] = _shape_arg(min_tensor.get("shape"))
            fields["range_b"] = _range_arg(min_tensor.get("range_values"))
        elif adapter.name == "ClampMaxTensor":
            max_tensor = _item_by_name(tensors, "clipValueMaxTensor") or tensors[1]
            fields["dtype_b"] = str(max_tensor["dtype"])
            fields["shape_b"] = _shape_arg(max_tensor.get("shape"))
            fields["range_b"] = _range_arg(max_tensor.get("range_values"))
        elif adapter.name == "Cast":
            fields["out_dtype"] = _acl_dtype_attr_arg(inputs)
        elif adapter.name in {"Amax", "Amin", "ReduceSum"}:
            dim_name = "dims" if adapter.name == "ReduceSum" else "dim"
            keepdim_name = "keepDims" if adapter.name == "ReduceSum" else "keepdim"
            fields["dims"] = _dims_arg(inputs, dim_name)
            fields["keep_dims"] = _bool_attr_arg(inputs, keepdim_name, default=False)
            if adapter.name == "ReduceSum":
                fields["dtype_attr"] = _dtype_attr_arg(str(tensors[0]["dtype"]), inputs)
                fields["noop_with_empty_axes"] = _bool_attr_arg(inputs, "noop_with_empty_axes", default=False)
        elif adapter.name == "Permute":
            fields["dims"] = _dims_arg(inputs, "dims")
        elif adapter.name == "aclnnChannelShuffle":
            fields["groups"] = _int_attr_arg(inputs, "groups")
        elif adapter.name == "Cumsum":
            fields["dims"] = _int_attr_arg(inputs, "dim")
            fields["dtype_attr"] = _acl_dtype_attr_arg(inputs)
        elif adapter.name == "CumsumV2":
            fields["dims"] = _int_attr_arg(inputs, "axis")
            fields["exclusive"] = _bool_attr_arg(inputs, "exclusive", default=False)
            fields["reverse"] = _bool_attr_arg(inputs, "reverse", default=False)
        elif adapter.name == "Prod":
            fields["dtype_attr"] = _acl_dtype_attr_arg(inputs)
        elif adapter.name == "ProdDim":
            fields["dims"] = _int_attr_arg(inputs, "dim")
            fields["keep_dims"] = _bool_attr_arg(inputs, "keepDim", default=False)
            fields["dtype_attr"] = _acl_dtype_attr_arg(inputs)
        elif adapter.name in {"Tril", "Triu"}:
            fields["diagonal"] = _int_attr_arg(inputs, "diagnoal")
        elif adapter.name == "Std":
            fields["dims"] = _dims_arg(inputs, "dim", allow_range_list=True)
            fields["correction"] = _int_attr_arg(inputs, "correction", allow_range_list=True)
            fields["keep_dims"] = _bool_attr_arg(inputs, "keepdim", default=False)
        elif adapter.name in {"Var", "VarCorrection", "VarMean"}:
            fields["dims"] = _dims_arg(inputs, "dim")
            fields["keep_dims"] = _bool_attr_arg(inputs, "keepdim", default=False)
            if adapter.name == "Var":
                fields["correction"] = _bool_attr_arg(inputs, "unbiased", default=False)
            else:
                fields["correction"] = _int_attr_arg(inputs, "correction")
        elif adapter.name == "Slice":
            fields["dims"] = _int_attr_arg(inputs, "dim")
            fields["start"] = _int_attr_arg(inputs, "start")
            fields["end"] = _int_attr_arg(inputs, "end")
            fields["step"] = _int_attr_arg(inputs, "step")
        elif adapter.name == "SplitWithSize":
            fields["dim"] = _int_attr_arg(inputs, "dim", allow_range_list=True)
            fields["split_sizes"] = _dims_arg(inputs, "splitSize")
        elif adapter.name == "StridedSlice":
            fields["begin"] = _dims_arg(inputs, "begin")
            fields["end"] = _dims_arg(inputs, "end")
            fields["strides"] = _dims_arg(inputs, "strides")
            fields["begin_mask"] = _int_attr_arg(inputs, "begin_mask")
            fields["end_mask"] = _int_attr_arg(inputs, "end_mask")
            fields["ellipsis_mask"] = _int_attr_arg(inputs, "ellipsis_mask")
            fields["new_axis_mask"] = _int_attr_arg(inputs, "new_axis_mask")
            fields["shrink_axis_mask"] = _int_attr_arg(inputs, "shrink_axis_mask")
        elif adapter.name == "UnfoldGrad":
            fields["input_sizes"] = _dims_arg(inputs, "input_sizes")
            fields["dim"] = _int_attr_arg(inputs, "dim")
            fields["size"] = _int_attr_arg(inputs, "size")
            fields["step"] = _int_attr_arg(inputs, "step")
        return "\t".join(f"{key}={value}" for key, value in fields.items())

    def _vendor_opapi_lib(self) -> Path:
        build_lib = self.repo_root / "build" / "libcust_opapi.so"
        if build_lib.is_file():
            return build_lib
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        return Path(ascend_home) / "opp" / "vendors" / f"{self.vendor}_transformer" / "op_api" / "lib" / "libcust_opapi.so"


def _tensor_list_shapes_arg(values: list[dict[str, Any]]) -> str:
    return ";".join(_shape_arg(item.get("shape")) for item in values)


def _tensor_list_ranges_arg(values: list[dict[str, Any]]) -> str:
    return ";".join(_range_arg(item.get("range_values")) for item in values)


def _lightning_indexer_batch_line(index: int, inputs: list[dict[str, Any]], tensors: list[dict[str, Any]], api: str) -> str:
    query = _required_tensor(tensors, "query")
    key = _required_tensor(tensors, "key")
    weights = _required_tensor(tensors, "weights")
    fields = {
        "case_index": str(index),
        "api": api,
        "dtype": str(query["dtype"]),
        "query_shape": _shape_arg(query.get("shape")),
        "key_shape": _shape_arg(key.get("shape")),
        "weights_shape": _shape_arg(weights.get("shape")),
        "query_range": _range_arg(query.get("range_values")),
        "key_range": _range_arg(key.get("range_values")),
        "weights_dtype": str(weights["dtype"]),
        "weights_range": _range_arg(weights.get("range_values")),
        "layout_query": _string_attr_arg(inputs, "layoutQuery", default="BSND"),
        "layout_key": _string_attr_arg(inputs, "layoutKey", default="BSND"),
        "sparse_count": _int_attr_arg(inputs, "sparseCount"),
        "sparse_mode": _int_attr_arg(inputs, "sparseMode"),
        "pre_tokens": _optional_int_attr_arg(inputs, "preTokens", default=9223372036854775807),
        "next_tokens": _optional_int_attr_arg(inputs, "nextTokens", default=9223372036854775807),
        "return_values": _bool_attr_arg(inputs, "returnValues", default=False),
        "actual_seq_lengths_query": _optional_tensor_values_arg(tensors, "actualSeqLengthsQuery"),
        "actual_seq_lengths_key": _optional_tensor_values_arg(tensors, "actualSeqLengthsKey"),
        "block_table_shape": _optional_tensor_shape_arg(tensors, "blockTable"),
        "block_table_values": _optional_tensor_values_arg(tensors, "blockTable"),
    }
    return "\t".join(f"{key}={value}" for key, value in fields.items())


def _recurrent_gated_delta_rule_batch_line(
    index: int, inputs: list[dict[str, Any]], tensors: list[dict[str, Any]], api: str
) -> str:
    fields = {
        "case_index": str(index),
        "api": api,
        "dtype": "bf16",
        "query_shape": _shape_arg(_required_tensor(tensors, "query").get("shape")),
        "key_shape": _shape_arg(_required_tensor(tensors, "key").get("shape")),
        "value_shape": _shape_arg(_required_tensor(tensors, "value").get("shape")),
        "beta_shape": _shape_arg(_required_tensor(tensors, "beta").get("shape")),
        "state_shape": _shape_arg(_required_tensor(tensors, "stateRef").get("shape")),
        "query_range": _range_arg(_required_tensor(tensors, "query").get("range_values")),
        "key_range": _range_arg(_required_tensor(tensors, "key").get("range_values")),
        "value_range": _range_arg(_required_tensor(tensors, "value").get("range_values")),
        "beta_range": _range_arg(_required_tensor(tensors, "beta").get("range_values")),
        "state_range": _range_arg(_required_tensor(tensors, "stateRef").get("range_values")),
        "actual_seq_lengths": _required_tensor_values_arg(tensors, "actualSeqLengths"),
        "ssm_state_indices": _required_tensor_values_arg(tensors, "ssmStateIndices"),
        "num_accepted_tokens": _required_tensor_values_arg(tensors, "numAcceptedTokens"),
        "g_present": "1" if _item_by_name(tensors, "g") is not None else "0",
        "g_shape": _optional_tensor_shape_arg(tensors, "g"),
        "g_range": _optional_tensor_range_arg(tensors, "g"),
        "gk_present": "1" if _item_by_name(tensors, "gk") is not None else "0",
        "gk_shape": _optional_tensor_shape_arg(tensors, "gk"),
        "gk_range": _optional_tensor_range_arg(tensors, "gk"),
        "scale_value": _float_attr_arg(inputs, "scaleValue"),
    }
    return "\t".join(f"{key}={value}" for key, value in fields.items())


def _required_tensor(tensors: list[dict[str, Any]], name: str) -> dict[str, Any]:
    tensor = _item_by_name(tensors, name)
    if tensor is None:
        raise ValueError(f"missing tensor {name}")
    return tensor


def _optional_tensor_shape_arg(tensors: list[dict[str, Any]], name: str) -> str:
    tensor = _item_by_name(tensors, name)
    return "" if tensor is None else _shape_arg(tensor.get("shape"))


def _optional_tensor_range_arg(tensors: list[dict[str, Any]], name: str) -> str:
    tensor = _item_by_name(tensors, name)
    return "" if tensor is None else _range_arg(tensor.get("range_values"))


def _required_tensor_values_arg(tensors: list[dict[str, Any]], name: str) -> str:
    tensor = _required_tensor(tensors, name)
    return _values_arg(tensor.get("value"))


def _optional_tensor_values_arg(tensors: list[dict[str, Any]], name: str) -> str:
    tensor = _item_by_name(tensors, name)
    if tensor is None:
        return ""
    return _values_arg(tensor.get("value"))


def _values_arg(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ",".join(str(int(item)) for item in value)
    return str(int(value))


def _item_by_name(items: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for item in items:
        if item.get("name") == name:
            return item
    return None


def _shape_arg(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, list):
        raise ValueError(f"shape must be a list, got {value!r}")
    return ",".join(str(int(dim)) for dim in value)


def _range_arg(value: Any) -> str:
    if isinstance(value, list) and len(value) >= 2 and _is_number(value[0]) and _is_number(value[1]):
        return f"{float(value[0])},{float(value[1])}"
    if isinstance(value, (int, float)):
        return f"{float(value)},{float(value)}"
    if isinstance(value, dict):
        mean = value.get("mean")
        if isinstance(mean, list) and len(mean) >= 2:
            return f"{float(mean[0])},{float(mean[1])}"
    return "-1,1"


def _scalar_arg(value: Any) -> str:
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        mean = value.get("mean")
        if isinstance(mean, list) and mean:
            value = mean[0]
        else:
            value = 1
    if not _is_number(value):
        value = 1
    return str(float(value))


def _named_scalar_arg(inputs: list[dict[str, Any]], name: str) -> str:
    for item in inputs:
        if item.get("type") == "scalar" and item.get("name") == name:
            return _scalar_arg(item.get("range_values"))
    raise ValueError(f"missing scalar {name}")


def _string_attr_arg(inputs: list[dict[str, Any]], name: str, default: str) -> str:
    for item in inputs:
        if item.get("type") == "attr" and item.get("name") == name:
            return str(item.get("range_values"))
    return default


def _float_attr_arg(inputs: list[dict[str, Any]], name: str) -> str:
    for item in inputs:
        if item.get("type") == "attr" and item.get("name") == name:
            value = item.get("range_values")
            if not _is_number(value):
                raise ValueError(f"attr {name} must be numeric")
            return str(float(value))
    raise ValueError(f"missing float attr {name}")


def _dims_arg(inputs: list[dict[str, Any]], name: str = "dims", allow_range_list: bool = False) -> str:
    values = []
    for item in inputs:
        if item.get("type") in {"attrs", "attr_tuple"} and item.get("name") == name:
            values.append(str(_int_attr_value(item.get("range_values", 0), allow_range_list=allow_range_list)))
    return ",".join(values)


def _bool_attr_arg(inputs: list[dict[str, Any]], name: str, default: bool) -> str:
    value = default
    for item in inputs:
        if item.get("type") == "attr" and item.get("name") == name:
            value = bool(item.get("range_values"))
            break
    return "1" if value else "0"


def _int_attr_arg(inputs: list[dict[str, Any]], name: str, allow_range_list: bool = False) -> str:
    for item in inputs:
        if item.get("type") == "attr" and item.get("name") == name:
            return str(_int_attr_value(item.get("range_values"), allow_range_list=allow_range_list))
    raise ValueError(f"missing int attr {name}")


def _optional_int_attr_arg(inputs: list[dict[str, Any]], name: str, default: int) -> str:
    for item in inputs:
        if item.get("type") == "attr" and item.get("name") == name:
            return str(_int_attr_value(item.get("range_values")))
    return str(default)


def _dtype_attr_arg(tensor_dtype: str, inputs: list[dict[str, Any]]) -> str:
    for item in inputs:
        if item.get("type") == "attr" and item.get("name") == "dtype":
            value = item.get("range_values")
            if value in (None, 0, "0"):
                return tensor_dtype
            return str(value)
    return tensor_dtype


def _acl_dtype_attr_arg(inputs: list[dict[str, Any]]) -> str:
    value = int(_int_attr_arg(inputs, "dtype"))
    dtype = ACL_DTYPE_ATTR_TO_NAME.get(value)
    if dtype is None:
        raise ValueError(f"unsupported ACL dtype attr {value}")
    return dtype


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _int_attr_value(value: Any, allow_range_list: bool = False) -> int:
    if isinstance(value, list):
        if len(value) == 1 or allow_range_list:
            value = value[0] if value else None
    return int(value)
