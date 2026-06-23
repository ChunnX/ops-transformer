from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

from .st_cases import StCase


SUPPORTED_DTYPES = {"fp32", "fp64", "fp16", "bf16", "int32", "int64", "int16", "int8", "uint8", "bool"}
FLOAT_DTYPES = {"fp32", "fp64", "fp16", "bf16"}
INTEGER_DTYPES = {"int32", "int64", "int16", "int8", "uint8"}
SIGNED_INTEGER_DTYPES = {"int32", "int64", "int16", "int8"}
FLOAT_OR_INTEGER_DTYPES = FLOAT_DTYPES | INTEGER_DTYPES
FLOAT_OR_SIGNED_INTEGER_DTYPES = FLOAT_DTYPES | SIGNED_INTEGER_DTYPES
ROUND_DTYPES = FLOAT_DTYPES | {"int32", "int64"}
TRUNC_DTYPES = {"fp32", "fp16", "bf16"}
BOOL_UNARY_DTYPES = {"fp32", "fp16", "bf16"}
SCALAR_CLIP_DTYPES = FLOAT_OR_INTEGER_DTYPES
GLOBAL_MAX_POOL_NUMEL_LIMIT = 10_000_000
DIVS_NUMEL_LIMIT = 10_000_000
CAST_NUMEL_LIMIT = 10_000_000
TENSOR_LIST_NUMEL_LIMIT = 10_000_000
MASKED_SELECT_NUMEL_LIMIT = 10_000_000
SPLIT_WITH_SIZE_NUMEL_LIMIT = 10_000_000
CHUNK_CAT_DTYPES = {"fp32", "fp16", "bf16"}
SPLIT_WITH_SIZE_DTYPES = {"fp32", "fp16", "int64", "int32", "int16", "int8", "bool"}
REDUCE_VAR_DTYPES = {"fp32", "fp16", "bf16"}
REDUCE_VAR_MEAN_DTYPES = {"fp32", "fp16"}
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
ACL_DTYPE_NAME_TO_ATTR = {name: code for code, name in ACL_DTYPE_ATTR_TO_NAME.items()}
INTEGER_OUTPUT_RANGES = {
    "int8": (-(2**7), 2**7 - 1),
    "uint8": (0, 2**8 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
}


@dataclass(frozen=True)
class Adapter:
    name: str
    tensor_inputs: int
    scalar_inputs: int
    supported_dtypes: frozenset[str] = frozenset(SUPPORTED_DTYPES)

    def supports(self, case: StCase) -> tuple[bool, str]:
        inputs = case.flat_inputs
        tensors = [item for item in inputs if item.get("type") == "tensor"]
        scalars = [item for item in inputs if item.get("type") == "scalar"]
        if len(tensors) != self.tensor_inputs:
            return False, f"{self.name} expects {self.tensor_inputs} tensor inputs"
        if len(scalars) != self.scalar_inputs:
            return False, f"{self.name} expects {self.scalar_inputs} scalar inputs"
        for item in tensors + scalars:
            dtype = str(item.get("dtype"))
            if dtype not in self.supported_dtypes:
                return False, f"dtype {dtype} is not supported by V1 native runner"
            shape = item.get("shape")
            if shape is not None and numel(shape) <= 0:
                return False, "zero-sized tensors are not supported by V1 native runner"
        return True, ""


def numel(shape: Any) -> int:
    if not isinstance(shape, list):
        return 1
    if not shape:
        return 1
    try:
        return int(reduce(mul, (int(dim) for dim in shape), 1))
    except (TypeError, ValueError):
        return 0


def tensor_items(case: StCase) -> list[dict[str, Any]]:
    return case.tensor_items


def tensor_list_items(case: StCase) -> list[dict[str, Any]]:
    return case.tensor_list_items


def scalar_items(case: StCase) -> list[dict[str, Any]]:
    return case.scalar_items


def scalar_by_name(scalars: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for scalar in scalars:
        if scalar.get("name") == name:
            return scalar
    return None


def item_by_name(items: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for item in items:
        if item.get("name") == name:
            return item
    return None


def support_binary_tensors(
    name: str, tensors: list[dict[str, Any]], supported_dtypes: set[str] | frozenset[str] = SUPPORTED_DTYPES
) -> tuple[bool, str]:
    for tensor in tensors:
        supported, reason = support_unary_tensor(name, tensor, supported_dtypes)
        if not supported:
            return supported, reason
    dtypes = {str(tensor.get("dtype")) for tensor in tensors}
    if len(dtypes) != 1:
        return False, f"{name} mixed tensor dtype promotion is not supported by native runner"
    if not broadcastable(tensors[0].get("shape"), tensors[1].get("shape")):
        return False, f"{name} input shapes are not broadcastable"
    return True, ""


def support_unary_tensor(
    name: str,
    tensor: dict[str, Any],
    supported_dtypes: set[str] | frozenset[str],
    allow_zero_sized: bool = False,
    allow_range_fallback: bool = False,
) -> tuple[bool, str]:
    dtype = str(tensor.get("dtype"))
    if dtype not in supported_dtypes:
        return False, f"dtype {dtype} is not supported by {name} native runner"
    shape = tensor.get("shape")
    if shape is not None and numel(shape) <= 0 and not allow_zero_sized:
        return False, "zero-sized tensors are not supported by V2 elementwise native runner"
    range_values = tensor.get("range_values")
    if not is_numeric_range(range_values) and not (allow_range_fallback and is_fallback_range_metadata(range_values)):
        return False, f"{name} non-numeric range is not supported by native runner"
    return True, ""


def shape_values(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    shape: list[int] = []
    for dim in value:
        if isinstance(dim, bool) or not isinstance(dim, int):
            return None
        shape.append(dim)
    return shape


def support_scalars(name: str, scalars: list[dict[str, Any]]) -> tuple[bool, str]:
    for scalar in scalars:
        dtype = str(scalar.get("dtype"))
        if dtype not in SUPPORTED_DTYPES:
            return False, f"scalar dtype {dtype} is not supported by {name} native runner"
        if not is_numeric_range(scalar.get("range_values")):
            return False, f"{name} scalar non-numeric range is not supported by native runner"
    return True, ""


def broadcastable(a: Any, b: Any) -> bool:
    return broadcast_shape(a, b) is not None


def broadcast_shape(a: Any, b: Any) -> list[int] | None:
    shape_a = shape_values(a)
    shape_b = shape_values(b)
    if shape_a is None or shape_b is None:
        return None
    rank = max(len(shape_a), len(shape_b))
    result: list[int] = []
    for index in range(rank):
        a_dim = 1 if index < rank - len(shape_a) else shape_a[index - (rank - len(shape_a))]
        b_dim = 1 if index < rank - len(shape_b) else shape_b[index - (rank - len(shape_b))]
        if a_dim != b_dim and a_dim != 1 and b_dim != 1:
            return None
        result.append(max(a_dim, b_dim))
    return result


def broadcast_shape_many(shapes: list[Any]) -> list[int] | None:
    result: Any = []
    for shape in shapes:
        result = broadcast_shape(result, shape)
        if result is None:
            return None
    return result


def is_numeric_range(value: Any) -> bool:
    if is_number(value):
        return True
    if isinstance(value, list) and value:
        return all(is_number(item) for item in value[:2])
    if isinstance(value, dict):
        mean = value.get("mean")
        return isinstance(mean, list) and bool(mean) and all(is_number(item) for item in mean[:2])
    return False


def is_fallback_range_metadata(value: Any) -> bool:
    if is_nonfinite_sentinel(value):
        return True
    if isinstance(value, list) and value:
        prefix = value[:2]
        return any(is_nonfinite_sentinel(item) for item in prefix) and all(
            is_number(item) or is_nonfinite_sentinel(item) for item in prefix
        )
    if isinstance(value, dict):
        mean = value.get("mean")
        if isinstance(mean, list) and mean:
            prefix = mean[:2]
            return any(is_nonfinite_sentinel(item) for item in prefix) and all(
                is_number(item) or is_nonfinite_sentinel(item) for item in prefix
            )
    return False


def is_nonfinite_sentinel(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan", "+nan", "-nan"}


def numeric_range_bounds(value: Any) -> tuple[float, float] | None:
    if is_number(value):
        numeric = float(value)
        return numeric, numeric
    if isinstance(value, list) and value and all(is_number(item) for item in value[:2]):
        low = float(value[0])
        high = float(value[1] if len(value) > 1 else value[0])
        return min(low, high), max(low, high)
    if isinstance(value, dict):
        mean = value.get("mean")
        if isinstance(mean, list) and mean and all(is_number(item) for item in mean[:2]):
            low = float(mean[0])
            high = float(mean[1] if len(mean) > 1 else mean[0])
            return min(low, high), max(low, high)
    return None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numeric_scalar_value(value: Any) -> float | None:
    if is_number(value):
        return float(value)
    if isinstance(value, list) and value and is_number(value[0]):
        return float(value[0])
    if isinstance(value, dict):
        mean = value.get("mean")
        if isinstance(mean, list) and mean and is_number(mean[0]):
            return float(mean[0])
    return None


def attr_value(inputs: list[dict[str, Any]], name: str) -> Any:
    for item in inputs:
        if item.get("type") == "attr" and item.get("name") == name:
            return item.get("range_values")
    return None


def bool_attr_value(inputs: list[dict[str, Any]], name: str, default: bool | None = None) -> bool | None:
    value = attr_value(inputs, name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return None


def coerce_int_attr_value(value: Any, allow_range_list: bool = False) -> int | None:
    if isinstance(value, list):
        if len(value) == 1 or allow_range_list:
            value = value[0] if value else None
        else:
            return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def int_attr_values(
    inputs: list[dict[str, Any]],
    name: str,
    accepted_types: set[str] | None = None,
    allow_range_list: bool = False,
) -> list[int] | None:
    if accepted_types is None:
        accepted_types = {"attrs"}
    values: list[int] = []
    for item in inputs:
        if item.get("type") not in accepted_types or item.get("name") != name:
            continue
        value = coerce_int_attr_value(item.get("range_values"), allow_range_list)
        if value is None:
            return None
        values.append(value)
    return values


def int_attr_value(inputs: list[dict[str, Any]], name: str, allow_range_list: bool = False) -> int | None:
    value = attr_value(inputs, name)
    return coerce_int_attr_value(value, allow_range_list)


def support_dtype_attr_matches_tensor(name: str, inputs: list[dict[str, Any]], tensor_dtype: str) -> tuple[bool, str]:
    dtype_attr = int_attr_value(inputs, "dtype")
    if dtype_attr is None:
        return False, f"{name} dtype attr is invalid"
    if ACL_DTYPE_ATTR_TO_NAME.get(dtype_attr) != tensor_dtype:
        return False, f"{name} dtype attr different from tensor dtype is not supported"
    return True, ""
