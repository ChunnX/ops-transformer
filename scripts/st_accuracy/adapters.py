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
            if shape is not None and _numel(shape) <= 0:
                return False, "zero-sized tensors are not supported by V1 native runner"
        return True, ""


@dataclass(frozen=True)
class ReduceSumAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        inputs = case.flat_inputs
        tensors = [item for item in inputs if item.get("type") == "tensor"]
        if len(tensors) != 1:
            return False, "ReduceSum expects 1 tensor input"

        tensor = tensors[0]
        dtype = str(tensor.get("dtype"))
        if dtype not in self.supported_dtypes:
            return False, f"dtype {dtype} is not supported by V1 native runner"

        shape = tensor.get("shape")

        dims = _int_attr_values(inputs, "dims")
        if dims is None:
            return False, "ReduceSum dims attrs are invalid"
        rank = len(shape) if isinstance(shape, list) else 0
        if rank == 0 and dims:
            return False, "ReduceSum dims are invalid for scalar input"
        normalized: set[int] = set()
        for dim in dims:
            if dim < -rank or dim >= rank:
                return False, "ReduceSum dims are out of range"
            axis = dim + rank if dim < 0 else dim
            if axis in normalized:
                return False, "ReduceSum duplicate dims are not supported"
            normalized.add(axis)

        keep_dims = _bool_attr_value(inputs, "keepDims")
        if keep_dims is None:
            return False, "ReduceSum keepDims attr is invalid"
        noop_with_empty_axes = _bool_attr_value(inputs, "noop_with_empty_axes", default=False)
        if noop_with_empty_axes is None:
            return False, "ReduceSum noop_with_empty_axes attr is invalid"
        if noop_with_empty_axes and dims:
            return False, "ReduceSum noop_with_empty_axes=true is not supported by native runner"

        dtype_attr = _attr_value(inputs, "dtype")
        if dtype_attr not in (None, 0, "0"):
            return False, "ReduceSum explicit dtype attr is not supported by native runner"
        return True, ""


@dataclass(frozen=True)
class ReductionAxesAdapter(Adapter):
    dim_attr_name: str = "dim"
    keepdim_attr_name: str = "keepdim"
    allow_range_fallback: bool = False

    def supports(self, case: StCase) -> tuple[bool, str]:
        inputs = case.flat_inputs
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"

        supported, reason = self._support_tensor(tensors[0])
        if not supported:
            return supported, reason
        return self._support_reduction_attrs(inputs, tensors[0].get("shape"))

    def _support_tensor(self, tensor: dict[str, Any]) -> tuple[bool, str]:
        return _support_unary_tensor(self.name, tensor, self.supported_dtypes, allow_range_fallback=self.allow_range_fallback)

    def _support_reduction_attrs(self, inputs: list[dict[str, Any]], shape: Any) -> tuple[bool, str]:
        dims = _int_attr_values(inputs, self.dim_attr_name, accepted_types={"attr_tuple", "attrs"})
        if not dims:
            return False, f"{self.name} dim attrs are invalid"
        rank = len(shape) if isinstance(shape, list) else 0
        if rank == 0 and dims:
            return False, f"{self.name} dims are invalid for scalar input"
        normalized: set[int] = set()
        for dim in dims:
            if dim < -rank or dim >= rank:
                return False, f"{self.name} dims are out of range"
            axis = dim + rank if dim < 0 else dim
            if axis in normalized:
                return False, f"{self.name} duplicate dims are not supported"
            normalized.add(axis)

        keepdim = _bool_attr_value(inputs, self.keepdim_attr_name)
        if keepdim is None:
            return False, f"{self.name} keepdim attr is invalid"
        return True, ""


@dataclass(frozen=True)
class MeanReductionAdapter(ReductionAxesAdapter):
    dim_attr_name: str = "dims"

    def _support_tensor(self, tensor: dict[str, Any]) -> tuple[bool, str]:
        dtype = str(tensor.get("dtype"))
        if dtype not in self.supported_dtypes:
            return False, f"dtype {dtype} is not supported by {self.name} native runner"
        shape = tensor.get("shape")
        if shape is not None and _numel(shape) <= 0:
            return False, "zero-sized tensors are not supported by V2 elementwise native runner"
        if not _is_numeric_range(tensor.get("range_values")):
            return False, "Mean non-finite range is not supported by native runner"
        return True, ""


@dataclass(frozen=True)
class AllReduceAdapter(Adapter):
    allow_range_fallback: bool = False

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"
        return _support_unary_tensor(self.name, tensors[0], self.supported_dtypes, allow_range_fallback=self.allow_range_fallback)


@dataclass(frozen=True)
class ProdAdapter(Adapter):
    has_dim: bool = False

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"

        supported, reason = _support_unary_tensor(self.name, tensors[0], self.supported_dtypes)
        if not supported:
            return supported, reason

        if self.has_dim:
            supported, reason = self._support_dim(case.flat_inputs, tensors[0].get("shape"))
            if not supported:
                return supported, reason

            keep_dim = _bool_attr_value(case.flat_inputs, "keepDim")
            if keep_dim is None:
                return False, "ProdDim keepDim attr is invalid"

        dtype_attr = _int_attr_value(case.flat_inputs, "dtype")
        if dtype_attr is None:
            return False, f"{self.name} dtype attr is invalid"
        output_dtype = ACL_DTYPE_ATTR_TO_NAME.get(dtype_attr)
        if output_dtype is None or output_dtype not in SUPPORTED_DTYPES:
            return False, f"{self.name} dtype attr {dtype_attr} is not supported by native runner"
        input_dtype = str(tensors[0].get("dtype"))
        if output_dtype != input_dtype:
            supported, reason = self._support_mixed_output_dtype(case.flat_inputs, tensors[0].get("shape"))
            if not supported:
                return supported, reason
        return True, ""

    def _support_dim(self, inputs: list[dict[str, Any]], shape: Any) -> tuple[bool, str]:
        dim = _int_attr_value(inputs, "dim")
        if dim is None:
            return False, "ProdDim dim attr is invalid"
        rank = len(shape) if isinstance(shape, list) else 0
        if rank == 0 or dim < -rank or dim >= rank:
            return False, "ProdDim dim is out of range"
        return True, ""

    def _support_mixed_output_dtype(self, inputs: list[dict[str, Any]], shape: Any) -> tuple[bool, str]:
        if not isinstance(shape, list):
            return False, f"{self.name} mixed output dtype is only supported for known single-element reductions"
        if not self.has_dim:
            if _numel(shape) != 1:
                return False, "Prod mixed output dtype is only supported for single-element reductions by native runner"
            return True, ""

        dim = _int_attr_value(inputs, "dim")
        if dim is None:
            return False, "ProdDim dim attr is invalid"
        rank = len(shape)
        normalized_dim = dim + rank if dim < 0 else dim
        if normalized_dim < 0 or normalized_dim >= rank:
            return False, "ProdDim dim is out of range"
        try:
            reduced_extent = int(shape[normalized_dim])
        except (TypeError, ValueError):
            return False, "ProdDim mixed output dtype is only supported for known single-element reductions"
        if reduced_extent != 1:
            return False, "ProdDim mixed output dtype is only supported when reduced dimension size is 1 by native runner"
        return True, ""


@dataclass(frozen=True)
class CumsumAdapter(Adapter):
    dim_attr_name: str = "dim"
    has_dtype_attr: bool = True
    allow_range_fallback: bool = False

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"

        supported, reason = _support_unary_tensor(
            self.name, tensors[0], self.supported_dtypes, allow_range_fallback=self.allow_range_fallback
        )
        if not supported:
            return supported, reason

        dim = _int_attr_value(case.flat_inputs, self.dim_attr_name)
        if dim is None:
            return False, f"{self.name} dim attr is invalid"
        shape = tensors[0].get("shape")
        rank = len(shape) if isinstance(shape, list) else 0
        if rank == 0 or dim < -rank or dim >= rank:
            return False, f"{self.name} dim is out of range"

        if self.has_dtype_attr:
            return _support_dtype_attr_matches_tensor(self.name, case.flat_inputs, str(tensors[0].get("dtype")))

        exclusive = _bool_attr_value(case.flat_inputs, "exclusive")
        if exclusive is None:
            return False, "CumsumV2 exclusive attr is invalid"
        reverse = _bool_attr_value(case.flat_inputs, "reverse")
        if reverse is None:
            return False, "CumsumV2 reverse attr is invalid"
        return True, ""


@dataclass(frozen=True)
class TriangularAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"

        supported, reason = _support_unary_tensor(self.name, tensors[0], self.supported_dtypes)
        if not supported:
            return supported, reason

        shape = tensors[0].get("shape")
        if not isinstance(shape, list) or len(shape) < 2:
            return False, f"{self.name} rank must be at least 2"

        diagonal = _int_attr_value(case.flat_inputs, "diagnoal")
        if diagonal is None:
            return False, f"{self.name} diagonal attr is invalid"
        return True, ""


@dataclass(frozen=True)
class StdAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, "Std expects 1 tensor input"
        if scalars:
            return False, "Std expects 0 scalar inputs"

        supported, reason = _support_unary_tensor(self.name, tensors[0], self.supported_dtypes)
        if not supported:
            return supported, reason

        shape = tensors[0].get("shape")
        rank = len(shape) if isinstance(shape, list) else 0
        if rank == 0:
            return False, "Std dims are invalid for scalar input"

        dims = _int_attr_values(case.flat_inputs, "dim", accepted_types={"attrs"}, allow_range_list=True)
        if not dims:
            return False, "Std dim attrs are invalid"
        for dim in dims:
            if dim < -rank or dim >= rank:
                return False, "Std dims are out of range"

        correction = _int_attr_value(case.flat_inputs, "correction", allow_range_list=True)
        if correction is None:
            return False, "Std correction attr is invalid"
        keepdim = _bool_attr_value(case.flat_inputs, "keepdim")
        if keepdim is None:
            return False, "Std keepdim attr is invalid"
        return True, ""


@dataclass(frozen=True)
class ReduceVarAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"

        supported, reason = _support_unary_tensor(self.name, tensors[0], self.supported_dtypes)
        if not supported:
            return supported, reason

        shape = _shape_values(tensors[0].get("shape"))
        if shape is None or not 1 <= len(shape) <= 8:
            return False, f"{self.name} rank must be in [1, 8]"

        dims = _int_attr_values(case.flat_inputs, "dim", accepted_types={"attrs"})
        if not dims:
            return False, f"{self.name} dim attrs are invalid"
        normalized: set[int] = set()
        rank = len(shape)
        for dim in dims:
            if dim < -rank or dim >= rank:
                return False, f"{self.name} dims are out of range"
            axis = dim + rank if dim < 0 else dim
            if axis in normalized:
                return False, f"{self.name} duplicate dims are not supported"
            normalized.add(axis)

        keepdim = _bool_attr_value(case.flat_inputs, "keepdim")
        if keepdim is None:
            return False, f"{self.name} keepdim attr is invalid"

        if self.name == "Var":
            unbiased = _bool_attr_value(case.flat_inputs, "unbiased")
            if unbiased is None:
                return False, "Var unbiased attr is invalid"
        else:
            correction = _int_attr_value(case.flat_inputs, "correction")
            if correction is None or correction < 0:
                return False, f"{self.name} correction attr is invalid"
        return True, ""


@dataclass(frozen=True)
class GlobalPoolAdapter(Adapter):
    max_numel: int | None = None
    allow_range_fallback: bool = False

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"

        supported, reason = _support_unary_tensor(
            self.name, tensors[0], self.supported_dtypes, allow_range_fallback=self.allow_range_fallback
        )
        if not supported:
            return supported, reason

        shape = tensors[0].get("shape")
        if not isinstance(shape, list) or len(shape) not in {4, 5}:
            return False, f"{self.name} rank must be 4 or 5"
        if self.max_numel is not None and _numel(shape) > self.max_numel:
            return False, f"{self.name} shape numel exceeds native runner limit"
        return True, ""


@dataclass(frozen=True)
class SliceAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, "Slice expects 1 tensor input"
        if scalars:
            return False, "Slice expects 0 scalar inputs"

        supported, reason = self._support_slice_tensor(tensors[0])
        if not supported:
            return supported, reason

        shape = tensors[0].get("shape")
        rank = len(shape) if isinstance(shape, list) else 0
        dim = _int_attr_value(case.flat_inputs, "dim")
        if dim is None:
            return False, "Slice dim attr is invalid"
        if rank == 0 or dim < -rank or dim >= rank:
            return False, "Slice dim is out of range"
        if _int_attr_value(case.flat_inputs, "start") is None:
            return False, "Slice start attr is invalid"
        if _int_attr_value(case.flat_inputs, "end") is None:
            return False, "Slice end attr is invalid"
        step = _int_attr_value(case.flat_inputs, "step")
        if step is None or step <= 0:
            return False, "Slice step attr is invalid"
        return True, ""

    def _support_slice_tensor(self, tensor: dict[str, Any]) -> tuple[bool, str]:
        dtype = str(tensor.get("dtype"))
        if dtype not in self.supported_dtypes:
            return False, f"dtype {dtype} is not supported by {self.name} native runner"
        shape = tensor.get("shape")
        if not _is_numeric_range(tensor.get("range_values")):
            if dtype == "bool" and shape is not None and _numel(shape) == 0:
                return True, ""
            return False, f"{self.name} non-numeric range is not supported by native runner"
        return True, ""


@dataclass(frozen=True)
class StridedSliceAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, "StridedSlice expects 1 tensor input"
        if scalars:
            return False, "StridedSlice expects 0 scalar inputs"

        supported, reason = _support_unary_tensor(self.name, tensors[0], self.supported_dtypes, allow_zero_sized=True)
        if not supported:
            return supported, reason

        begin = _int_attr_values(case.flat_inputs, "begin")
        end = _int_attr_values(case.flat_inputs, "end")
        strides = _int_attr_values(case.flat_inputs, "strides")
        if begin is None or end is None or strides is None or len(begin) != len(end) or len(begin) != len(strides):
            return False, "StridedSlice begin/end/strides attrs are invalid"
        if any(stride == 0 for stride in strides):
            return False, "StridedSlice strides attr is invalid"

        masks = {
            name: _int_attr_value(case.flat_inputs, name)
            for name in ("begin_mask", "end_mask", "ellipsis_mask", "new_axis_mask", "shrink_axis_mask")
        }
        if any(value is None or value < 0 for value in masks.values()):
            return False, "StridedSlice mask attrs are invalid"

        shape = tensors[0].get("shape")
        rank = len(shape) if isinstance(shape, list) else 0
        max_spec_axes = rank + int(masks["new_axis_mask"]).bit_count() + int(masks["ellipsis_mask"] != 0)
        if len(begin) > max_spec_axes:
            return False, "StridedSlice begin/end/strides attrs are invalid"
        return True, ""


@dataclass(frozen=True)
class UnsupportedAdapter(Adapter):
    reason: str = ""

    def supports(self, case: StCase) -> tuple[bool, str]:
        return False, self.reason


@dataclass(frozen=True)
class BinaryTensorAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 2:
            return False, f"{self.name} expects 2 tensor inputs"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"
        return _support_binary_tensors(self.name, tensors, self.supported_dtypes)


@dataclass(frozen=True)
class GerAdapter(Adapter):
    supported_dtypes: frozenset[str] = frozenset({"fp16", "fp32"})

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 2:
            return False, "Ger expects 2 tensor inputs"
        if scalars:
            return False, "Ger expects 0 scalar inputs"
        dtypes = {str(tensor.get("dtype")) for tensor in tensors}
        if len(dtypes) != 1:
            return False, "Ger mixed tensor dtype promotion is not supported by native runner"
        dtype = next(iter(dtypes))
        if dtype not in self.supported_dtypes:
            return False, f"dtype {dtype} is not supported by Ger native runner"
        for tensor in tensors:
            shape = tensor.get("shape")
            if not isinstance(shape, list) or len(shape) != 1:
                return False, "Ger expects 1D tensor inputs"
            supported, reason = _support_unary_tensor("Ger", tensor, self.supported_dtypes)
            if not supported:
                return supported, reason
        return True, ""


@dataclass(frozen=True)
class BinaryFloatingTensorAdapter(Adapter):
    supported_dtypes: frozenset[str] = frozenset(FLOAT_DTYPES)

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 2:
            return False, f"{self.name} expects 2 tensor inputs"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"
        for tensor in tensors:
            supported, reason = _support_unary_tensor(self.name, tensor, self.supported_dtypes)
            if not supported:
                return supported, reason
        dtypes = {str(tensor.get("dtype")) for tensor in tensors}
        if len(dtypes) != 1:
            return False, f"{self.name} mixed tensor dtype promotion is not supported by native runner"
        if not _broadcastable(tensors[0].get("shape"), tensors[1].get("shape")):
            return False, f"{self.name} input shapes are not broadcastable"
        return True, ""


@dataclass(frozen=True)
class BinaryTensorAlphaAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 2:
            return False, f"{self.name} expects 2 tensor inputs"
        if len(scalars) != 1:
            return False, f"{self.name} expects 1 scalar input"
        supported, reason = _support_binary_tensors(self.name, tensors)
        if not supported:
            return supported, reason
        return _support_scalars(self.name, scalars)


@dataclass(frozen=True)
class TensorScalarAdapter(Adapter):
    allow_zero_sized: bool = False

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if len(scalars) != 1:
            return False, f"{self.name} expects 1 scalar input"
        supported, reason = _support_unary_tensor(self.name, tensors[0], SUPPORTED_DTYPES, self.allow_zero_sized)
        if not supported:
            return supported, reason
        return _support_scalars(self.name, scalars)


@dataclass(frozen=True)
class DivsAdapter(Adapter):
    max_numel: int = DIVS_NUMEL_LIMIT

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, "Divs expects 1 tensor input"
        if len(scalars) != 1:
            return False, "Divs expects 1 scalar input"

        tensor = tensors[0]
        tensor_dtype = str(tensor.get("dtype"))
        if tensor_dtype not in self.supported_dtypes:
            return False, f"dtype {tensor_dtype} is not supported by Divs native runner"

        scalar = scalars[0]
        scalar_dtype = str(scalar.get("dtype"))
        if scalar_dtype != tensor_dtype:
            return False, f"Divs scalar dtype {scalar_dtype} does not match tensor dtype {tensor_dtype}"

        shape = tensor.get("shape")
        if shape is not None and _numel(shape) <= 0:
            return False, "zero-sized tensors are not supported by V2 elementwise native runner"
        if not _is_numeric_range(tensor.get("range_values")):
            return False, "Divs non-finite range is not supported by native runner"
        if not _is_numeric_range(scalar.get("range_values")):
            return False, "Divs scalar non-finite range is not supported by native runner"
        if shape is not None and _numel(shape) > self.max_numel:
            return False, "Divs shape numel exceeds native runner limit"
        return True, ""


@dataclass(frozen=True)
class ScalarClipAdapter(Adapter):
    required_scalar_names: tuple[str, ...] = ()

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if len(scalars) != len(self.required_scalar_names):
            return False, f"{self.name} expects {len(self.required_scalar_names)} scalar inputs"

        tensor = tensors[0]
        supported, reason = _support_unary_tensor(self.name, tensor, self.supported_dtypes)
        if not supported:
            return supported, reason

        tensor_dtype = str(tensor.get("dtype"))
        values: dict[str, float] = {}
        for scalar_name in self.required_scalar_names:
            scalar = _scalar_by_name(scalars, scalar_name)
            if scalar is None:
                return False, f"{self.name} missing scalar {scalar_name}"
            scalar_dtype = str(scalar.get("dtype"))
            if scalar_dtype != tensor_dtype:
                return False, f"{self.name} scalar dtype {scalar_dtype} does not match tensor dtype {tensor_dtype}"
            if not _is_numeric_range(scalar.get("range_values")):
                return False, f"{self.name} scalar non-numeric range is not supported by native runner"
            value = _numeric_scalar_value(scalar.get("range_values"))
            if value is None:
                return False, f"{self.name} scalar non-numeric range is not supported by native runner"
            values[scalar_name] = value

        if "clipValueMin" in values and "clipValueMax" in values and values["clipValueMin"] > values["clipValueMax"]:
            return False, f"{self.name} clipValueMin must be <= clipValueMax"
        return True, ""


@dataclass(frozen=True)
class TensorClipAdapter(Adapter):
    required_bound_names: tuple[str, ...] = ()

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        expected_tensor_count = 1 + len(self.required_bound_names)
        if len(tensors) != expected_tensor_count:
            return False, f"{self.name} expects {expected_tensor_count} tensor inputs"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"

        self_tensor = _item_by_name(tensors, "self") or tensors[0]
        supported, reason = _support_unary_tensor(self.name, self_tensor, self.supported_dtypes)
        if not supported:
            return supported, reason

        bound_tensors: list[dict[str, Any]] = []
        for bound_name in self.required_bound_names:
            tensor = _item_by_name(tensors, bound_name)
            if tensor is None:
                return False, f"{self.name} missing tensor {bound_name}"
            supported, reason = _support_unary_tensor(self.name, tensor, self.supported_dtypes)
            if not supported:
                return supported, reason
            bound_tensors.append(tensor)

        shapes = [self_tensor.get("shape"), *(tensor.get("shape") for tensor in bound_tensors)]
        if _broadcast_shape_many(shapes) is None:
            return False, f"{self.name} input shapes are not broadcastable"

        if "clipValueMinTensor" in self.required_bound_names and "clipValueMaxTensor" in self.required_bound_names:
            min_tensor = _item_by_name(bound_tensors, "clipValueMinTensor")
            max_tensor = _item_by_name(bound_tensors, "clipValueMaxTensor")
            min_bounds = _numeric_range_bounds(min_tensor.get("range_values") if min_tensor is not None else None)
            max_bounds = _numeric_range_bounds(max_tensor.get("range_values") if max_tensor is not None else None)
            if min_bounds is None or max_bounds is None:
                return False, f"{self.name} tensor non-numeric range is not supported by native runner"
            if min_bounds[1] > max_bounds[0]:
                return False, f"{self.name} clipValueMinTensor range may exceed clipValueMaxTensor range"
        return True, ""


@dataclass(frozen=True)
class CastAdapter(Adapter):
    max_numel: int = CAST_NUMEL_LIMIT

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, "Cast expects 1 tensor input"
        if scalars:
            return False, "Cast expects 0 scalar inputs"

        tensor = tensors[0]
        supported, reason = _support_unary_tensor(self.name, tensor, self.supported_dtypes)
        if not supported:
            return supported, reason

        shape = tensor.get("shape")
        if shape is not None and _numel(shape) > self.max_numel:
            return False, "Cast shape numel exceeds native runner limit"

        dtype_attr = _int_attr_value(case.flat_inputs, "dtype")
        if dtype_attr is None:
            return False, "Cast dtype attr is invalid"
        output_dtype = ACL_DTYPE_ATTR_TO_NAME.get(dtype_attr)
        if output_dtype is None or output_dtype not in self.supported_dtypes:
            return False, f"Cast dtype attr {dtype_attr} is not supported by native runner"
        if output_dtype in INTEGER_OUTPUT_RANGES:
            bounds = _numeric_range_bounds(tensor.get("range_values"))
            if bounds is None:
                return False, "Cast non-numeric range is not supported by native runner"
            low, high = bounds
            min_value, max_value = INTEGER_OUTPUT_RANGES[output_dtype]
            if low < min_value or high > max_value:
                return False, f"Cast input range exceeds {output_dtype} output range"
        return True, ""


@dataclass(frozen=True)
class TensorListAdapter(Adapter):
    mode: str = "cat"
    max_numel: int = TENSOR_LIST_NUMEL_LIMIT

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_list_items(case)
        if not tensors:
            return False, f"{self.name} expects a non-empty TensorList input"
        if _tensor_items(case):
            return False, f"{self.name} expects TensorList inputs only"
        if _scalar_items(case):
            return False, f"{self.name} expects 0 scalar inputs"

        dtypes: set[str] = set()
        shapes: list[list[int]] = []
        for tensor in tensors:
            supported, reason = _support_unary_tensor(self.name, tensor, self.supported_dtypes)
            if not supported:
                return supported, reason
            dtype = str(tensor.get("dtype"))
            dtypes.add(dtype)
            shape = _shape_values(tensor.get("shape"))
            if shape is None:
                return False, f"{self.name} TensorList shape is invalid"
            shapes.append(shape)

        if len(dtypes) != 1:
            return False, f"{self.name} mixed TensorList dtype promotion is not supported by native runner"

        dim = _int_attr_value(case.flat_inputs, "dim")
        if dim is None:
            return False, f"{self.name} dim attr is invalid"

        if self.mode == "cat":
            supported, reason, output_shape = self._cat_output_shape(shapes, dim)
        elif self.mode == "stack":
            supported, reason, output_shape = self._stack_output_shape(shapes, dim, len(tensors))
        else:
            return False, f"{self.name} TensorList mode is invalid"

        if not supported:
            return supported, reason
        if _numel(output_shape) > self.max_numel:
            return False, f"{self.name} output numel exceeds native runner limit"
        return True, ""

    def _cat_output_shape(self, shapes: list[list[int]], dim: int) -> tuple[bool, str, list[int]]:
        rank = len(shapes[0])
        if rank == 0:
            return False, "Cat scalar TensorList is not supported by native runner", []
        if dim < -rank or dim >= rank:
            return False, "Cat dim is out of range", []
        axis = dim + rank if dim < 0 else dim
        output_shape = list(shapes[0])
        output_shape[axis] = 0
        for shape in shapes:
            if len(shape) != rank:
                return False, "Cat TensorList ranks must match", []
            for index, (actual, expected) in enumerate(zip(shape, shapes[0])):
                if index != axis and actual != expected:
                    return False, "Cat TensorList shapes must match except concat dim", []
            output_shape[axis] += shape[axis]
        return True, "", output_shape

    def _stack_output_shape(self, shapes: list[list[int]], dim: int, count: int) -> tuple[bool, str, list[int]]:
        rank = len(shapes[0])
        if dim < -(rank + 1) or dim > rank:
            return False, "Stack dim is out of range", []
        for shape in shapes:
            if shape != shapes[0]:
                return False, "Stack TensorList shapes must match", []
        axis = dim + rank + 1 if dim < 0 else dim
        output_shape = list(shapes[0])
        output_shape.insert(axis, count)
        return True, "", output_shape


@dataclass(frozen=True)
class ChunkCatAdapter(Adapter):
    max_numel: int = TENSOR_LIST_NUMEL_LIMIT

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_list_items(case)
        if not tensors:
            return False, "ChunkCat expects a non-empty TensorList input"
        if _tensor_items(case):
            return False, "ChunkCat expects TensorList inputs only"
        if _scalar_items(case):
            return False, "ChunkCat expects 0 scalar inputs"

        dim = _int_attr_value(case.flat_inputs, "dim")
        if dim is None:
            return False, "ChunkCat dim attr is invalid"
        if dim != 0:
            return False, "ChunkCat dim only supports 0 on ascend910b native runner"

        num_chunks = _int_attr_value(case.flat_inputs, "num_chunks")
        if num_chunks is None:
            return False, "ChunkCat num_chunks attr is invalid"
        if num_chunks <= 0:
            return False, "ChunkCat num_chunks must be positive"

        dtypes: set[str] = set()
        shapes: list[list[int]] = []
        for tensor in tensors:
            dtype = str(tensor.get("dtype"))
            if dtype not in CHUNK_CAT_DTYPES:
                return False, f"dtype {dtype} is not supported by ChunkCat native runner"
            dtypes.add(dtype)
            if not _is_numeric_range(tensor.get("range_values")):
                return False, "ChunkCat non-numeric range is not supported by native runner"
            shape = _shape_values(tensor.get("shape"))
            if shape is None:
                return False, "ChunkCat TensorList shape is invalid"
            if not 1 <= len(shape) <= 8:
                return False, "ChunkCat TensorList rank must be between 1 and 8"
            if any(dim_value <= 0 for dim_value in shape):
                return False, "zero-sized tensors are not supported by V2 elementwise native runner"
            shapes.append(shape)

        if len(dtypes) != 1:
            return False, "ChunkCat mixed TensorList dtype promotion is not supported by native runner"

        output_shape = self._output_shape(shapes, num_chunks)
        if _numel(output_shape) > self.max_numel:
            return False, "ChunkCat output numel exceeds native runner limit"
        return True, ""

    def _output_shape(self, shapes: list[list[int]], num_chunks: int) -> list[int]:
        dim1 = 0
        for shape in shapes:
            chunk_rows = (shape[0] + num_chunks - 1) // num_chunks
            trailing = _numel(shape[1:])
            dim1 += chunk_rows * trailing
        return [num_chunks, dim1]


@dataclass(frozen=True)
class SplitWithSizeAdapter(Adapter):
    max_numel: int = SPLIT_WITH_SIZE_NUMEL_LIMIT

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, "SplitWithSize expects 1 tensor input"
        if scalars:
            return False, "SplitWithSize expects 0 scalar inputs"

        tensor = _item_by_name(tensors, "self") or tensors[0]
        dtype = str(tensor.get("dtype"))
        if dtype not in self.supported_dtypes:
            return False, f"dtype {dtype} is not supported by SplitWithSize native runner"
        if not _is_numeric_range(tensor.get("range_values")):
            return False, "SplitWithSize non-numeric range is not supported by native runner"

        shape = _shape_values(tensor.get("shape"))
        if shape is None or not 1 <= len(shape) <= 8:
            return False, "SplitWithSize rank must be in [1, 8]"
        if _numel(shape) <= 0:
            return False, "zero-sized tensors are not supported by SplitWithSize native runner"
        if _numel(shape) > self.max_numel:
            return False, "SplitWithSize input numel exceeds native runner limit"

        dim = _int_attr_value(case.flat_inputs, "dim", allow_range_list=True)
        if dim is None:
            return False, "SplitWithSize dim attr is invalid"
        rank = len(shape)
        if dim < -rank or dim >= rank:
            return False, "SplitWithSize dim is out of range"
        axis = dim + rank if dim < 0 else dim

        split_sizes = _int_attr_values(case.flat_inputs, "splitSize")
        if not split_sizes:
            return False, "SplitWithSize splitSize attrs are invalid"
        if any(value < 0 for value in split_sizes):
            return False, "SplitWithSize splitSize values must be non-negative"
        if sum(split_sizes) != shape[axis]:
            return False, "SplitWithSize splitSize sum must match split dimension"
        for split_size in split_sizes:
            out_shape = list(shape)
            out_shape[axis] = split_size
            if _numel(out_shape) > self.max_numel:
                return False, "SplitWithSize output numel exceeds native runner limit"
        return True, ""


@dataclass(frozen=True)
class MaskedSelectAdapter(Adapter):
    max_numel: int = MASKED_SELECT_NUMEL_LIMIT

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 2:
            return False, "MaskedSelect expects 2 tensor inputs"
        if scalars:
            return False, "MaskedSelect expects 0 scalar inputs"

        self_tensor = _item_by_name(tensors, "self") or tensors[0]
        mask_tensor = _item_by_name(tensors, "mask") or tensors[1]
        self_dtype = str(self_tensor.get("dtype"))
        if self_dtype not in self.supported_dtypes:
            return False, f"dtype {self_dtype} is not supported by MaskedSelect native runner"

        mask_dtype = str(mask_tensor.get("dtype"))
        if mask_dtype != "bool":
            return False, f"MaskedSelect mask dtype {mask_dtype} is not supported by native runner"

        self_shape = _shape_values(self_tensor.get("shape"))
        mask_shape = _shape_values(mask_tensor.get("shape"))
        if self_shape is None or mask_shape is None:
            return False, "MaskedSelect input shape is invalid"
        if _numel(self_shape) <= 0 or _numel(mask_shape) <= 0:
            return False, "zero-sized tensors are not supported by V2 elementwise native runner"

        output_shape = _broadcast_shape(self_shape, mask_shape)
        if output_shape is None:
            return False, "MaskedSelect input shapes are not broadcastable"
        if _numel(output_shape) > self.max_numel:
            return False, "MaskedSelect broadcast numel exceeds native runner limit"
        if not _is_numeric_range(self_tensor.get("range_values")):
            return False, "MaskedSelect non-numeric range is not supported by native runner"
        return True, ""


@dataclass(frozen=True)
class UnfoldGradAdapter(Adapter):
    supported_dtypes: frozenset[str] = frozenset({"fp32", "fp16", "bf16"})

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, "UnfoldGrad expects 1 tensor input"
        if scalars:
            return False, "UnfoldGrad expects 0 scalar inputs"

        tensor = tensors[0]
        supported, reason = _support_unary_tensor(self.name, tensor, self.supported_dtypes)
        if not supported:
            return supported, reason

        grad_out_shape = _shape_values(tensor.get("shape"))
        if grad_out_shape is None:
            return False, "UnfoldGrad grad_out shape is invalid"

        input_sizes = _int_attr_values(case.flat_inputs, "input_sizes")
        if not input_sizes:
            return False, "UnfoldGrad input_sizes attrs are invalid"
        if any(size_value < 0 for size_value in input_sizes):
            return False, "UnfoldGrad input_sizes attrs are invalid"
        if _numel(input_sizes) <= 0:
            return False, "zero-sized tensors are not supported by V2 elementwise native runner"

        dim = _int_attr_value(case.flat_inputs, "dim")
        size = _int_attr_value(case.flat_inputs, "size")
        step = _int_attr_value(case.flat_inputs, "step")
        if dim is None:
            return False, "UnfoldGrad dim attr is invalid"
        if size is None or size <= 0:
            return False, "UnfoldGrad size attr is invalid"
        if step is None or step <= 0:
            return False, "UnfoldGrad step attr is invalid"

        rank = len(input_sizes)
        if rank == 0 or dim < -rank or dim >= rank:
            return False, "UnfoldGrad dim is out of range"
        axis = dim + rank if dim < 0 else dim
        if axis not in {rank - 1, rank - 2}:
            return False, "UnfoldGrad dim must be last or second-last input axis"
        if size > input_sizes[axis]:
            return False, "UnfoldGrad size attr is invalid"

        window_count = (input_sizes[axis] - size) // step + 1
        expected_shape = list(input_sizes)
        expected_shape[axis] = window_count
        expected_shape.append(size)
        if grad_out_shape != expected_shape:
            return False, "UnfoldGrad grad_out shape does not match input_sizes/dim/size/step"
        return True, ""


@dataclass(frozen=True)
class TensorScalarAlphaAdapter(Adapter):
    allow_zero_sized: bool = False

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if len(scalars) != 2:
            return False, f"{self.name} expects 2 scalar inputs"
        supported, reason = _support_unary_tensor(self.name, tensors[0], SUPPORTED_DTYPES, self.allow_zero_sized)
        if not supported:
            return supported, reason
        return _support_scalars(self.name, scalars)


@dataclass(frozen=True)
class UnaryAdapter(Adapter):
    allow_zero_sized: bool = False

    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"
        return _support_unary_tensor(self.name, tensors[0], self.supported_dtypes, self.allow_zero_sized)


@dataclass(frozen=True)
class UnaryFloatingAdapter(UnaryAdapter):
    supported_dtypes: frozenset[str] = frozenset(FLOAT_DTYPES)


@dataclass(frozen=True)
class PermuteAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"

        supported, reason = _support_unary_tensor(self.name, tensors[0], self.supported_dtypes)
        if not supported:
            return supported, reason

        shape = tensors[0].get("shape")
        rank = len(shape) if isinstance(shape, list) else 0
        dims = _int_attr_values(case.flat_inputs, "dims")
        if dims is None or len(dims) != rank:
            return False, "Permute dims must be a rank-length permutation"

        normalized: list[int] = []
        for dim in dims:
            if dim < -rank or dim >= rank:
                return False, "Permute dims must be a rank-length permutation"
            normalized.append(dim + rank if dim < 0 else dim)
        if sorted(normalized) != list(range(rank)):
            return False, "Permute dims must be a rank-length permutation"
        return True, ""


@dataclass(frozen=True)
class ChannelShuffleAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        tensors = _tensor_items(case)
        scalars = _scalar_items(case)
        if len(tensors) != 1:
            return False, f"{self.name} expects 1 tensor input"
        if scalars:
            return False, f"{self.name} expects 0 scalar inputs"

        supported, reason = _support_unary_tensor(self.name, tensors[0], self.supported_dtypes)
        if not supported:
            return supported, reason

        shape = tensors[0].get("shape")
        if not isinstance(shape, list) or not 3 <= len(shape) <= 7:
            return False, "aclnnChannelShuffle rank must be in [3, 7]"

        groups = _int_attr_value(case.flat_inputs, "groups")
        if groups is None or groups <= 0:
            return False, "aclnnChannelShuffle groups attr is invalid"
        try:
            channels = int(shape[1])
        except (TypeError, ValueError):
            return False, "aclnnChannelShuffle channels attr is invalid"
        if channels % groups != 0:
            return False, "aclnnChannelShuffle channels must be divisible by groups"
        return True, ""


def _numel(shape: Any) -> int:
    if not isinstance(shape, list):
        return 1
    if not shape:
        return 1
    try:
        return int(reduce(mul, (int(dim) for dim in shape), 1))
    except (TypeError, ValueError):
        return 0


def _tensor_items(case: StCase) -> list[dict[str, Any]]:
    return case.tensor_items


def _tensor_list_items(case: StCase) -> list[dict[str, Any]]:
    return case.tensor_list_items


def _scalar_items(case: StCase) -> list[dict[str, Any]]:
    return case.scalar_items


def _scalar_by_name(scalars: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for scalar in scalars:
        if scalar.get("name") == name:
            return scalar
    return None


def _item_by_name(items: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for item in items:
        if item.get("name") == name:
            return item
    return None


def _support_binary_tensors(
    name: str, tensors: list[dict[str, Any]], supported_dtypes: set[str] | frozenset[str] = SUPPORTED_DTYPES
) -> tuple[bool, str]:
    for tensor in tensors:
        supported, reason = _support_unary_tensor(name, tensor, supported_dtypes)
        if not supported:
            return supported, reason
    dtypes = {str(tensor.get("dtype")) for tensor in tensors}
    if len(dtypes) != 1:
        return False, f"{name} mixed tensor dtype promotion is not supported by native runner"
    if not _broadcastable(tensors[0].get("shape"), tensors[1].get("shape")):
        return False, f"{name} input shapes are not broadcastable"
    return True, ""


def _support_unary_tensor(
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
    if shape is not None and _numel(shape) <= 0 and not allow_zero_sized:
        return False, "zero-sized tensors are not supported by V2 elementwise native runner"
    range_values = tensor.get("range_values")
    if not _is_numeric_range(range_values) and not (allow_range_fallback and _is_fallback_range_metadata(range_values)):
        return False, f"{name} non-numeric range is not supported by native runner"
    return True, ""


def _shape_values(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    shape: list[int] = []
    for dim in value:
        if isinstance(dim, bool) or not isinstance(dim, int):
            return None
        shape.append(dim)
    return shape


def _support_scalars(name: str, scalars: list[dict[str, Any]]) -> tuple[bool, str]:
    for scalar in scalars:
        dtype = str(scalar.get("dtype"))
        if dtype not in SUPPORTED_DTYPES:
            return False, f"scalar dtype {dtype} is not supported by {name} native runner"
        if not _is_numeric_range(scalar.get("range_values")):
            return False, f"{name} scalar non-numeric range is not supported by native runner"
    return True, ""


def _broadcastable(a: Any, b: Any) -> bool:
    return _broadcast_shape(a, b) is not None


def _broadcast_shape(a: Any, b: Any) -> list[int] | None:
    shape_a = _shape_values(a)
    shape_b = _shape_values(b)
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


def _broadcast_shape_many(shapes: list[Any]) -> list[int] | None:
    result: Any = []
    for shape in shapes:
        result = _broadcast_shape(result, shape)
        if result is None:
            return None
    return result


def _is_numeric_range(value: Any) -> bool:
    if _is_number(value):
        return True
    if isinstance(value, list) and value:
        return all(_is_number(item) for item in value[:2])
    if isinstance(value, dict):
        mean = value.get("mean")
        return isinstance(mean, list) and bool(mean) and all(_is_number(item) for item in mean[:2])
    return False


def _is_fallback_range_metadata(value: Any) -> bool:
    if _is_nonfinite_sentinel(value):
        return True
    if isinstance(value, list) and value:
        prefix = value[:2]
        return any(_is_nonfinite_sentinel(item) for item in prefix) and all(
            _is_number(item) or _is_nonfinite_sentinel(item) for item in prefix
        )
    if isinstance(value, dict):
        mean = value.get("mean")
        if isinstance(mean, list) and mean:
            prefix = mean[:2]
            return any(_is_nonfinite_sentinel(item) for item in prefix) and all(
                _is_number(item) or _is_nonfinite_sentinel(item) for item in prefix
            )
    return False


def _is_nonfinite_sentinel(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan", "+nan", "-nan"}


def _numeric_range_bounds(value: Any) -> tuple[float, float] | None:
    if _is_number(value):
        numeric = float(value)
        return numeric, numeric
    if isinstance(value, list) and value and all(_is_number(item) for item in value[:2]):
        low = float(value[0])
        high = float(value[1] if len(value) > 1 else value[0])
        return min(low, high), max(low, high)
    if isinstance(value, dict):
        mean = value.get("mean")
        if isinstance(mean, list) and mean and all(_is_number(item) for item in mean[:2]):
            low = float(mean[0])
            high = float(mean[1] if len(mean) > 1 else mean[0])
            return min(low, high), max(low, high)
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_scalar_value(value: Any) -> float | None:
    if _is_number(value):
        return float(value)
    if isinstance(value, list) and value and _is_number(value[0]):
        return float(value[0])
    if isinstance(value, dict):
        mean = value.get("mean")
        if isinstance(mean, list) and mean and _is_number(mean[0]):
            return float(mean[0])
    return None


def _attr_value(inputs: list[dict[str, Any]], name: str) -> Any:
    for item in inputs:
        if item.get("type") == "attr" and item.get("name") == name:
            return item.get("range_values")
    return None


def _bool_attr_value(inputs: list[dict[str, Any]], name: str, default: bool | None = None) -> bool | None:
    value = _attr_value(inputs, name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return None


def _coerce_int_attr_value(value: Any, allow_range_list: bool = False) -> int | None:
    if isinstance(value, list):
        if len(value) == 1 or allow_range_list:
            value = value[0] if value else None
        else:
            return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _int_attr_values(
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
        value = _coerce_int_attr_value(item.get("range_values"), allow_range_list)
        if value is None:
            return None
        values.append(value)
    return values


def _int_attr_value(inputs: list[dict[str, Any]], name: str, allow_range_list: bool = False) -> int | None:
    value = _attr_value(inputs, name)
    return _coerce_int_attr_value(value, allow_range_list)


def _support_dtype_attr_matches_tensor(name: str, inputs: list[dict[str, Any]], tensor_dtype: str) -> tuple[bool, str]:
    dtype_attr = _int_attr_value(inputs, "dtype")
    if dtype_attr is None:
        return False, f"{name} dtype attr is invalid"
    if ACL_DTYPE_ATTR_TO_NAME.get(dtype_attr) != tensor_dtype:
        return False, f"{name} dtype attr different from tensor dtype is not supported"
    return True, ""


def _string_attr_value(inputs: list[dict[str, Any]], name: str, default: str | None = None) -> str | None:
    value = _attr_value(inputs, name)
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return None


def _named_tensor(case: StCase, name: str) -> dict[str, Any] | None:
    return _item_by_name(_tensor_items(case), name)


def _support_attention_tensor(
    api_name: str,
    tensor: dict[str, Any] | None,
    name: str,
    supported_dtypes: set[str],
    require_numeric_range: bool = True,
) -> tuple[bool, str]:
    if tensor is None:
        return False, f"{api_name} missing tensor {name}"
    dtype = str(tensor.get("dtype"))
    if dtype not in supported_dtypes:
        return False, f"{api_name} tensor {name} dtype {dtype} is not supported"
    shape = tensor.get("shape")
    if not isinstance(shape, list) or not shape or _numel(shape) <= 0:
        return False, f"{api_name} tensor {name} shape is invalid"
    if require_numeric_range and "value" not in tensor and not _is_numeric_range(tensor.get("range_values")):
        return False, f"{api_name} tensor {name} range is not numeric"
    return True, ""


@dataclass(frozen=True)
class LightningIndexerAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        inputs = case.flat_inputs
        query = _named_tensor(case, "query")
        key = _named_tensor(case, "key")
        weights = _named_tensor(case, "weights")
        for tensor, name, dtypes in (
            (query, "query", {"fp16", "bf16"}),
            (key, "key", {"fp16", "bf16"}),
            (weights, "weights", {"fp16", "bf16", "fp32"}),
        ):
            supported, reason = _support_attention_tensor(self.name, tensor, name, dtypes)
            if not supported:
                return supported, reason

        layout_query = _string_attr_value(inputs, "layoutQuery", default="BSND")
        layout_key = _string_attr_value(inputs, "layoutKey", default="BSND")
        if layout_query not in {"BSND", "TND"}:
            return False, "LightningIndexer layoutQuery must be BSND or TND"
        if layout_key not in {"BSND", "TND", "PA_BSND"}:
            return False, "LightningIndexer layoutKey must be BSND, TND, or PA_BSND"
        if layout_key != "PA_BSND" and layout_key != layout_query:
            return False, "LightningIndexer non-PA layoutQuery and layoutKey must match"

        query_shape = query.get("shape")
        key_shape = key.get("shape")
        weights_shape = weights.get("shape")
        if layout_query == "BSND":
            if len(query_shape) != 4 or len(weights_shape) != 3:
                return False, "LightningIndexer BSND query/weights shapes are invalid"
            n1 = int(query_shape[2])
            dim = int(query_shape[3])
            if weights_shape[:3] != query_shape[:3]:
                return False, "LightningIndexer weights shape must match query B/S/N1"
        else:
            if len(query_shape) != 3 or len(weights_shape) != 2:
                return False, "LightningIndexer TND query/weights shapes are invalid"
            n1 = int(query_shape[1])
            dim = int(query_shape[2])
            if weights_shape != query_shape[:2]:
                return False, "LightningIndexer TND weights shape must match query T/N1"
            if _named_tensor(case, "actualSeqLengthsQuery") is None:
                return False, "LightningIndexer TND requires actualSeqLengthsQuery"

        if n1 < 1 or n1 > 64:
            return False, "LightningIndexer query N1 must be in [1, 64]"
        if dim != 128:
            return False, "LightningIndexer head dim must be 128"

        if layout_key == "PA_BSND":
            if len(key_shape) != 4:
                return False, "LightningIndexer PA_BSND key shape must be 4D"
            block_size = int(key_shape[1])
            n2 = int(key_shape[2])
            key_dim = int(key_shape[3])
            if block_size <= 0 or block_size > 1024 or block_size % 16 != 0:
                return False, "LightningIndexer PA_BSND block size must be a 16 multiple in (0, 1024]"
            if _named_tensor(case, "actualSeqLengthsKey") is None or _named_tensor(case, "blockTable") is None:
                return False, "LightningIndexer PA_BSND requires actualSeqLengthsKey and blockTable"
        elif layout_key == "BSND":
            if len(key_shape) != 4:
                return False, "LightningIndexer BSND key shape must be 4D"
            n2 = int(key_shape[2])
            key_dim = int(key_shape[3])
        else:
            if len(key_shape) != 3:
                return False, "LightningIndexer TND key shape must be 3D"
            n2 = int(key_shape[1])
            key_dim = int(key_shape[2])
            if _named_tensor(case, "actualSeqLengthsKey") is None:
                return False, "LightningIndexer TND requires actualSeqLengthsKey"
        if n2 != 1:
            return False, "LightningIndexer key N2 must be 1"
        if key_dim != 128:
            return False, "LightningIndexer key head dim must be 128"

        sparse_count = _int_attr_value(inputs, "sparseCount")
        if sparse_count is None:
            return False, "LightningIndexer sparseCount attr is invalid"
        if not ((1 <= sparse_count <= 2048) or sparse_count in {3072, 4096, 5120, 6144, 7168, 8192}):
            return False, "LightningIndexer sparseCount is out of supported range"
        sparse_mode = _int_attr_value(inputs, "sparseMode")
        if sparse_mode not in {0, 3}:
            return False, "LightningIndexer sparseMode must be 0 or 3"
        return_values = _bool_attr_value(inputs, "returnValues", default=False)
        if return_values is None:
            return False, "LightningIndexer returnValues attr is invalid"
        return True, ""


@dataclass(frozen=True)
class RecurrentGatedDeltaRuleAdapter(Adapter):
    def supports(self, case: StCase) -> tuple[bool, str]:
        required = {
            "query": {"bf16"},
            "key": {"bf16"},
            "value": {"bf16"},
            "beta": {"bf16"},
            "stateRef": {"bf16"},
            "actualSeqLengths": {"int32"},
            "ssmStateIndices": {"int32"},
            "numAcceptedTokens": {"int32"},
        }
        for name, dtypes in required.items():
            supported, reason = _support_attention_tensor(
                self.name,
                _named_tensor(case, name),
                name,
                dtypes,
                require_numeric_range=name not in {"actualSeqLengths", "ssmStateIndices", "numAcceptedTokens"},
            )
            if not supported:
                return supported, reason

        for name in ("g", "gk"):
            tensor = _named_tensor(case, name)
            if tensor is not None:
                supported, reason = _support_attention_tensor(self.name, tensor, name, {"fp32"})
                if not supported:
                    return supported, reason

        query_shape = _named_tensor(case, "query").get("shape")
        key_shape = _named_tensor(case, "key").get("shape")
        value_shape = _named_tensor(case, "value").get("shape")
        beta_shape = _named_tensor(case, "beta").get("shape")
        state_shape = _named_tensor(case, "stateRef").get("shape")
        if len(query_shape) != 3 or len(key_shape) != 3 or len(value_shape) != 3:
            return False, "RecurrentGatedDeltaRule query/key/value must be 3D"
        if len(beta_shape) != 2 or len(state_shape) != 4:
            return False, "RecurrentGatedDeltaRule beta/stateRef shapes are invalid"
        if query_shape != key_shape:
            return False, "RecurrentGatedDeltaRule query and key shapes must match"

        total_t, nk, dk = [int(v) for v in query_shape]
        value_t, nv, dv = [int(v) for v in value_shape]
        block_num, state_nv, state_dv, state_dk = [int(v) for v in state_shape]
        if total_t != value_t or beta_shape != [total_t, nv]:
            return False, "RecurrentGatedDeltaRule value/beta T/Nv must match query/value"
        if [state_nv, state_dv, state_dk] != [nv, dv, dk]:
            return False, "RecurrentGatedDeltaRule stateRef Nv/Dv/Dk must match value/query"
        if block_num <= 0:
            return False, "RecurrentGatedDeltaRule BlockNum must be positive"
        if nk <= 0 or nv <= 0 or dk <= 0 or dv <= 0:
            return False, "RecurrentGatedDeltaRule dimensions must be positive"
        if nk > 256 or nv > 256 or dk > 256 or dv > 256:
            return False, "RecurrentGatedDeltaRule dimensions exceed documented limits"
        if nv % nk != 0:
            return False, "RecurrentGatedDeltaRule native golden requires Nv to be divisible by Nk"

        g = _named_tensor(case, "g")
        if g is not None and g.get("shape") != [total_t, nv]:
            return False, "RecurrentGatedDeltaRule g shape must be [T, Nv]"
        gk = _named_tensor(case, "gk")
        if gk is not None and gk.get("shape") != [total_t, nv, dk]:
            return False, "RecurrentGatedDeltaRule gk shape must be [T, Nv, Dk]"
        scale = _attr_value(case.flat_inputs, "scaleValue")
        if not _is_number(scale):
            return False, "RecurrentGatedDeltaRule scaleValue attr is invalid"
        return True, ""


ADAPTERS = {
    "Abs": Adapter(name="Abs", tensor_inputs=1, scalar_inputs=0),
    "Add": Adapter(name="Add", tensor_inputs=2, scalar_inputs=1),
    "Adds": Adapter(name="Adds", tensor_inputs=1, scalar_inputs=2),
    "Amax": ReductionAxesAdapter(name="Amax", tensor_inputs=1, scalar_inputs=0, allow_range_fallback=True),
    "Amin": ReductionAxesAdapter(name="Amin", tensor_inputs=1, scalar_inputs=0, allow_range_fallback=True),
    "Cast": CastAdapter(name="Cast", tensor_inputs=1, scalar_inputs=0),
    "Cat": TensorListAdapter(name="Cat", tensor_inputs=0, scalar_inputs=0, mode="cat"),
    "Ceil": UnaryAdapter(
        name="Ceil", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(FLOAT_DTYPES), allow_zero_sized=True
    ),
    "ChunkCat": ChunkCatAdapter(
        name="ChunkCat", tensor_inputs=0, scalar_inputs=0, supported_dtypes=frozenset(CHUNK_CAT_DTYPES)
    ),
    "Cos": UnaryFloatingAdapter(name="Cos", tensor_inputs=1, scalar_inputs=0),
    "Cumsum": CumsumAdapter(name="Cumsum", tensor_inputs=1, scalar_inputs=0),
    "CumsumV2": CumsumAdapter(
        name="CumsumV2",
        tensor_inputs=1,
        scalar_inputs=0,
        dim_attr_name="axis",
        has_dtype_attr=False,
        allow_range_fallback=True,
    ),
    "Div": BinaryFloatingTensorAdapter(name="Div", tensor_inputs=2, scalar_inputs=0),
    "Divs": DivsAdapter(
        name="Divs",
        tensor_inputs=1,
        scalar_inputs=1,
        supported_dtypes=frozenset({"fp16", "fp32", "bf16"}),
    ),
    "Var": ReduceVarAdapter(
        name="Var", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(REDUCE_VAR_DTYPES)
    ),
    "VarCorrection": ReduceVarAdapter(
        name="VarCorrection", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(REDUCE_VAR_DTYPES)
    ),
    "VarMean": ReduceVarAdapter(
        name="VarMean", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(REDUCE_VAR_MEAN_DTYPES)
    ),
    "Exp": UnaryAdapter(
        name="Exp", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(FLOAT_DTYPES), allow_zero_sized=True
    ),
    "Floor": UnaryAdapter(
        name="Floor", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(FLOAT_DTYPES), allow_zero_sized=True
    ),
    "GlobalAveragePool": GlobalPoolAdapter(
        name="GlobalAveragePool",
        tensor_inputs=1,
        scalar_inputs=0,
        supported_dtypes=frozenset(FLOAT_DTYPES),
        allow_range_fallback=True,
    ),
    "GlobalMaxPool": GlobalPoolAdapter(
        name="GlobalMaxPool",
        tensor_inputs=1,
        scalar_inputs=0,
        supported_dtypes=frozenset({"fp16", "fp32"}),
        max_numel=GLOBAL_MAX_POOL_NUMEL_LIMIT,
        allow_range_fallback=True,
    ),
    "Ger": GerAdapter(name="Ger", tensor_inputs=2, scalar_inputs=0),
    "LightningIndexer": LightningIndexerAdapter(name="LightningIndexer", tensor_inputs=3, scalar_inputs=0),
    "IsFinite": UnaryAdapter(
        name="IsFinite", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(BOOL_UNARY_DTYPES)
    ),
    "IsInf": UnaryAdapter(
        name="IsInf", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(BOOL_UNARY_DTYPES)
    ),
    "Slice": SliceAdapter(
        name="Slice",
        tensor_inputs=1,
        scalar_inputs=0,
        supported_dtypes=frozenset({"fp32", "fp16", "bf16", "int32", "int64", "int8", "uint8", "bool"}),
    ),
    "SplitWithSize": SplitWithSizeAdapter(
        name="SplitWithSize",
        tensor_inputs=1,
        scalar_inputs=0,
        supported_dtypes=frozenset(SPLIT_WITH_SIZE_DTYPES),
    ),
    "StridedSlice": UnsupportedAdapter(
        name="StridedSlice",
        tensor_inputs=1,
        scalar_inputs=0,
        reason=(
            "StridedSlice is not supported on ascend910b custom package; "
            "aclnnStridedSliceGetWorkspaceSize returns 161002 and L2 CheckDtypeValid logs only support ASCEND950"
        ),
    ),
    "aclnnChannelShuffle": ChannelShuffleAdapter(name="aclnnChannelShuffle", tensor_inputs=1, scalar_inputs=0),
    "Log": UnaryAdapter(
        name="Log", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(FLOAT_DTYPES), allow_zero_sized=True
    ),
    "Hardtanh": ScalarClipAdapter(
        name="Hardtanh",
        tensor_inputs=1,
        scalar_inputs=2,
        supported_dtypes=frozenset(SCALAR_CLIP_DTYPES),
        required_scalar_names=("clipValueMin", "clipValueMax"),
    ),
    "Clamp": ScalarClipAdapter(
        name="Clamp",
        tensor_inputs=1,
        scalar_inputs=2,
        supported_dtypes=frozenset(SCALAR_CLIP_DTYPES),
        required_scalar_names=("clipValueMin", "clipValueMax"),
    ),
    "ClampMin": ScalarClipAdapter(
        name="ClampMin",
        tensor_inputs=1,
        scalar_inputs=1,
        supported_dtypes=frozenset(SCALAR_CLIP_DTYPES),
        required_scalar_names=("clipValueMin",),
    ),
    "ClampMax": ScalarClipAdapter(
        name="ClampMax",
        tensor_inputs=1,
        scalar_inputs=1,
        supported_dtypes=frozenset(SCALAR_CLIP_DTYPES),
        required_scalar_names=("clipValueMax",),
    ),
    "ClampTensor": TensorClipAdapter(
        name="ClampTensor",
        tensor_inputs=3,
        scalar_inputs=0,
        supported_dtypes=frozenset(SCALAR_CLIP_DTYPES),
        required_bound_names=("clipValueMinTensor", "clipValueMaxTensor"),
    ),
    "ClampMinTensor": TensorClipAdapter(
        name="ClampMinTensor",
        tensor_inputs=2,
        scalar_inputs=0,
        supported_dtypes=frozenset(SCALAR_CLIP_DTYPES),
        required_bound_names=("clipValueMinTensor",),
    ),
    "ClampMaxTensor": TensorClipAdapter(
        name="ClampMaxTensor",
        tensor_inputs=2,
        scalar_inputs=0,
        supported_dtypes=frozenset(SCALAR_CLIP_DTYPES),
        required_bound_names=("clipValueMaxTensor",),
    ),
    "Max": AllReduceAdapter(name="Max", tensor_inputs=1, scalar_inputs=0, allow_range_fallback=True),
    "Mean": MeanReductionAdapter(
        name="Mean", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(FLOAT_DTYPES)
    ),
    "MaskedSelect": MaskedSelectAdapter(name="MaskedSelect", tensor_inputs=2, scalar_inputs=0),
    "Maximum": BinaryTensorAdapter(
        name="Maximum", tensor_inputs=2, scalar_inputs=0, supported_dtypes=frozenset({"fp16", "fp32"})
    ),
    "Minimum": BinaryTensorAdapter(
        name="Minimum", tensor_inputs=2, scalar_inputs=0, supported_dtypes=frozenset({"fp32"})
    ),
    "Min": AllReduceAdapter(name="Min", tensor_inputs=1, scalar_inputs=0, allow_range_fallback=True),
    "Mul": BinaryTensorAdapter(name="Mul", tensor_inputs=2, scalar_inputs=0),
    "Muls": TensorScalarAdapter(name="Muls", tensor_inputs=1, scalar_inputs=1, allow_zero_sized=True),
    "Neg": UnaryAdapter(
        name="Neg",
        tensor_inputs=1,
        scalar_inputs=0,
        supported_dtypes=frozenset(FLOAT_OR_SIGNED_INTEGER_DTYPES),
        allow_zero_sized=True,
    ),
    "Permute": PermuteAdapter(name="Permute", tensor_inputs=1, scalar_inputs=0),
    "Prod": ProdAdapter(name="Prod", tensor_inputs=1, scalar_inputs=0),
    "ProdDim": ProdAdapter(name="ProdDim", tensor_inputs=1, scalar_inputs=0, has_dim=True),
    "ReduceSum": ReduceSumAdapter(name="ReduceSum", tensor_inputs=1, scalar_inputs=0),
    "RecurrentGatedDeltaRule": RecurrentGatedDeltaRuleAdapter(
        name="RecurrentGatedDeltaRule", tensor_inputs=10, scalar_inputs=0
    ),
    "Reciprocal": UnaryAdapter(
        name="Reciprocal",
        tensor_inputs=1,
        scalar_inputs=0,
        supported_dtypes=frozenset(FLOAT_DTYPES),
        allow_zero_sized=True,
    ),
    "Round": UnaryAdapter(
        name="Round",
        tensor_inputs=1,
        scalar_inputs=0,
        supported_dtypes=frozenset(ROUND_DTYPES),
        allow_zero_sized=True,
    ),
    "Rsqrt": UnaryAdapter(
        name="Rsqrt", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(FLOAT_DTYPES), allow_zero_sized=True
    ),
    "Sin": UnaryAdapter(
        name="Sin", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(FLOAT_DTYPES), allow_zero_sized=True
    ),
    "Sign": UnaryAdapter(
        name="Sign", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset({"fp32"}), allow_zero_sized=True
    ),
    "Sqrt": UnaryAdapter(
        name="Sqrt", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(FLOAT_DTYPES), allow_zero_sized=True
    ),
    "Square": UnsupportedAdapter(
        name="Square",
        tensor_inputs=1,
        scalar_inputs=0,
        reason="Square is not supported on ascend910b custom package; eager example fails with aclnnSquareGetWorkspaceSize 161002",
    ),
    "Std": StdAdapter(
        name="Std", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset({"fp16", "fp32", "bf16"})
    ),
    "Sub": BinaryTensorAlphaAdapter(name="Sub", tensor_inputs=2, scalar_inputs=1),
    "Subs": TensorScalarAlphaAdapter(name="Subs", tensor_inputs=1, scalar_inputs=2, allow_zero_sized=True),
    "Stack": TensorListAdapter(name="Stack", tensor_inputs=0, scalar_inputs=0, mode="stack"),
    "Tanh": UnaryAdapter(
        name="Tanh", tensor_inputs=1, scalar_inputs=0, supported_dtypes=frozenset(FLOAT_DTYPES), allow_zero_sized=True
    ),
    "Tril": TriangularAdapter(name="Tril", tensor_inputs=1, scalar_inputs=0),
    "Trunc": UnaryAdapter(
        name="Trunc",
        tensor_inputs=1,
        scalar_inputs=0,
        supported_dtypes=frozenset(TRUNC_DTYPES),
        allow_zero_sized=True,
    ),
    "Triu": TriangularAdapter(name="Triu", tensor_inputs=1, scalar_inputs=0),
    "UnfoldGrad": UnfoldGradAdapter(name="UnfoldGrad", tensor_inputs=1, scalar_inputs=0),
}


def get_adapter(aclnn_name: str) -> Adapter | None:
    return ADAPTERS.get(aclnn_name) or ADAPTERS.get(aclnn_name.removeprefix("aclnn"))
