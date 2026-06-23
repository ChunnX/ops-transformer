from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapter_utils import ACL_DTYPE_ATTR_TO_NAME
from .adapter_utils import ACL_DTYPE_NAME_TO_ATTR
from .adapter_utils import BOOL_UNARY_DTYPES
from .adapter_utils import CAST_NUMEL_LIMIT
from .adapter_utils import CHUNK_CAT_DTYPES
from .adapter_utils import DIVS_NUMEL_LIMIT
from .adapter_utils import FLOAT_DTYPES
from .adapter_utils import FLOAT_OR_INTEGER_DTYPES
from .adapter_utils import FLOAT_OR_SIGNED_INTEGER_DTYPES
from .adapter_utils import GLOBAL_MAX_POOL_NUMEL_LIMIT
from .adapter_utils import INTEGER_OUTPUT_RANGES
from .adapter_utils import MASKED_SELECT_NUMEL_LIMIT
from .adapter_utils import REDUCE_VAR_DTYPES
from .adapter_utils import REDUCE_VAR_MEAN_DTYPES
from .adapter_utils import ROUND_DTYPES
from .adapter_utils import SCALAR_CLIP_DTYPES
from .adapter_utils import SPLIT_WITH_SIZE_DTYPES
from .adapter_utils import SPLIT_WITH_SIZE_NUMEL_LIMIT
from .adapter_utils import SUPPORTED_DTYPES
from .adapter_utils import TENSOR_LIST_NUMEL_LIMIT
from .adapter_utils import TRUNC_DTYPES
from .adapter_utils import Adapter
from .adapter_utils import attr_value as _attr_value
from .adapter_utils import bool_attr_value as _bool_attr_value
from .adapter_utils import broadcast_shape as _broadcast_shape
from .adapter_utils import broadcast_shape_many as _broadcast_shape_many
from .adapter_utils import broadcastable as _broadcastable
from .adapter_utils import int_attr_value as _int_attr_value
from .adapter_utils import int_attr_values as _int_attr_values
from .adapter_utils import is_numeric_range as _is_numeric_range
from .adapter_utils import item_by_name as _item_by_name
from .adapter_utils import numel as _numel
from .adapter_utils import numeric_range_bounds as _numeric_range_bounds
from .adapter_utils import numeric_scalar_value as _numeric_scalar_value
from .adapter_utils import scalar_by_name as _scalar_by_name
from .adapter_utils import scalar_items as _scalar_items
from .adapter_utils import shape_values as _shape_values
from .adapter_utils import support_binary_tensors as _support_binary_tensors
from .adapter_utils import support_dtype_attr_matches_tensor as _support_dtype_attr_matches_tensor
from .adapter_utils import support_scalars as _support_scalars
from .adapter_utils import support_unary_tensor as _support_unary_tensor
from .adapter_utils import tensor_items as _tensor_items
from .adapter_utils import tensor_list_items as _tensor_list_items
from .attention_adapters import LightningIndexerAdapter
from .attention_adapters import RecurrentGatedDeltaRuleAdapter
from .st_cases import StCase


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
