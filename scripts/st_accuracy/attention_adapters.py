from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapter_utils import Adapter
from .adapter_utils import attr_value
from .adapter_utils import bool_attr_value
from .adapter_utils import int_attr_value
from .adapter_utils import is_number
from .adapter_utils import is_numeric_range
from .adapter_utils import item_by_name
from .adapter_utils import numel
from .adapter_utils import tensor_items
from .st_cases import StCase


def _string_attr_value(inputs: list[dict[str, Any]], name: str, default: str | None = None) -> str | None:
    value = attr_value(inputs, name)
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return None


def _named_tensor(case: StCase, name: str) -> dict[str, Any] | None:
    return item_by_name(tensor_items(case), name)


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
    if not isinstance(shape, list) or not shape or numel(shape) <= 0:
        return False, f"{api_name} tensor {name} shape is invalid"
    if require_numeric_range and "value" not in tensor and not is_numeric_range(tensor.get("range_values")):
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

        sparse_count = int_attr_value(inputs, "sparseCount")
        if sparse_count is None:
            return False, "LightningIndexer sparseCount attr is invalid"
        if not ((1 <= sparse_count <= 2048) or sparse_count in {3072, 4096, 5120, 6144, 7168, 8192}):
            return False, "LightningIndexer sparseCount is out of supported range"
        sparse_mode = int_attr_value(inputs, "sparseMode")
        if sparse_mode not in {0, 3}:
            return False, "LightningIndexer sparseMode must be 0 or 3"
        return_values = bool_attr_value(inputs, "returnValues", default=False)
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
        scale = attr_value(case.flat_inputs, "scaleValue")
        if not is_number(scale):
            return False, "RecurrentGatedDeltaRule scaleValue attr is invalid"
        return True, ""
