#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Compare model-level fused UDMA dumps with unfused reference dumps.

The analyzer is intentionally CPU-only.  It understands ordinary ``torch.save``
ZIP files and byte-split files named ``*.pt.part.N`` without importing torch, so
large model dumps can be inspected on a development machine that only has NumPy.

The useful boundary is an operator invocation: all inputs must be identical
before an output difference can be attributed to that invocation.  Reports keep
NaN propagation separate from newly introduced NaN/Inf values.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import heapq
import io
import json
import math
import os
import pickle
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on the analysis host
    raise SystemExit(
        "NumPy is required to analyze model dumps. Install numpy in the analysis "
        "environment; torch and torch_npu are not required."
    ) from exc


PART_RE = re.compile(
    r"^(?P<base>.+\.(?:pt|pth|bin))\.part\.(?P<part>[0-9]+)$", re.IGNORECASE
)
RANK_RE = re.compile(r"(?:^|[^a-z0-9])rank[_-]?(?P<rank>[0-9]+)(?:[^a-z0-9]|$)")
SEQUENCE_RE = re.compile(
    r"(?:call|invoke|invocation|sequence|seq|order|dump|step|launch|iter(?:ation)?)"
    r"[_-]?(?P<sequence>[0-9]+)"
)
ORDINAL_CALL_RE = re.compile(
    r"(?:^|[/_-])(?P<ordinal>first|second|third|fourth|fifth)[_-]call"
    r"(?:[/_-]|$)"
)
ORDINAL_CALL_INDEX = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
}
DEFAULT_STRIP_TOKENS = (
    "udma",
    "fused",
    "reference",
    "baseline",
    "original",
    "legacy",
    "golden",
    "expected",
    "actual",
)
DEFAULT_EXTENSIONS = (".pt", ".pth", ".bin")
TOP_ROW_LIMIT = 128
OP_PIPELINE_ORDER = {
    "all_to_allv_grouped_mat_mul": 0,
    "grouped_mat_mul_all_to_allv": 1,
    "unknown": 2,
}
PHASE_ORDER = {
    "numbered_call": 0,
    "forward": 1,
    "backward": 2,
    "unspecified": 3,
}

OP_ALIASES = (
    (
        re.compile(
            r"all[_-]?(?:to|2)[_-]?allv?[_-]?grouped[_-]?mat[_-]?mul|"
            r"alltoallvgroupedmatmul|a2av[_-]?gmm|a2a[_-]?gemm",
            re.IGNORECASE,
        ),
        "all_to_allv_grouped_mat_mul",
    ),
    (
        re.compile(
            r"grouped[_-]?mat[_-]?mul[_-]?all[_-]?(?:to|2)[_-]?allv?|"
            r"groupedmatmulalltoallv|gmm[_-]?a2av|gemm[_-]?a2a",
            re.IGNORECASE,
        ),
        "grouped_mat_mul_all_to_allv",
    ),
)


@dataclasses.dataclass(frozen=True)
class StorageType:
    name: str


@dataclasses.dataclass(frozen=True)
class StorageRef:
    storage_type: StorageType
    key: str
    location: str
    numel: int


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    storage: StorageRef
    storage_offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


@dataclasses.dataclass
class LoadedTensor:
    path: Path
    component: str
    array: Any
    serialized_dtype: str
    storage_numel: int
    storage_offset: int
    stride: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Artifact:
    side: str
    key: str
    relative_path: str
    path: Path
    rank: Optional[int]
    sequence: Optional[int]
    phase: str
    op: str
    io_kind: str
    tensor_name: str
    invocation: str
    order_confidence: str


def _rebuild_tensor(
    storage: StorageRef,
    storage_offset: int,
    size: Iterable[int],
    stride: Iterable[int],
    *unused: Any,
) -> TensorSpec:
    return TensorSpec(
        storage=storage,
        storage_offset=int(storage_offset),
        size=tuple(int(value) for value in size),
        stride=tuple(int(value) for value in stride),
    )


def _rebuild_parameter(tensor: Any, *unused: Any) -> Any:
    return tensor


class TorchMetadataUnpickler(pickle.Unpickler):
    """Restricted unpickler for tensor metadata in a trusted torch dump."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name in {
            "_rebuild_tensor",
            "_rebuild_tensor_v2",
            "_rebuild_tensor_v3",
        }:
            return _rebuild_tensor
        if module == "torch._utils" and name in {
            "_rebuild_parameter",
            "_rebuild_parameter_with_state",
        }:
            return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(
            "unsupported global in torch metadata: {}.{}".format(module, name)
        )

    def persistent_load(self, saved_id: Any) -> StorageRef:
        if not isinstance(saved_id, tuple) or not saved_id or saved_id[0] != "storage":
            raise pickle.UnpicklingError(
                "unsupported persistent id: {!r}".format(saved_id)
            )
        if len(saved_id) < 5:
            raise pickle.UnpicklingError("invalid storage id: {!r}".format(saved_id))
        storage_type, key, location, numel = saved_id[1:5]
        if not isinstance(storage_type, StorageType):
            raise pickle.UnpicklingError(
                "unsupported storage type: {!r}".format(storage_type)
            )
        return StorageRef(storage_type, str(key), str(location), int(numel))


DTYPE_MAP: dict[str, tuple[str, str]] = {
    "DoubleStorage": ("<f8", "float64"),
    "FloatStorage": ("<f4", "float32"),
    "HalfStorage": ("<f2", "float16"),
    "BFloat16Storage": ("<u2", "bfloat16"),
    "LongStorage": ("<i8", "int64"),
    "IntStorage": ("<i4", "int32"),
    "ShortStorage": ("<i2", "int16"),
    "CharStorage": ("i1", "int8"),
    "ByteStorage": ("u1", "uint8"),
    "BoolStorage": ("?", "bool"),
}

PRECISION_STANDARDS: dict[str, dict[str, Any]] = {
    "float16": {
        "rtol": 2**-9,
        "atol": 2**-9,
        "required_ratio": 0.99,
        "hard_fixed": 1e-1,
        "mantissa_bits": 10,
        "min_exponent": -14,
    },
    "bfloat16": {
        "rtol": 2**-6,
        "atol": 2**-6,
        "required_ratio": 0.99,
        "hard_fixed": 1.0,
        "mantissa_bits": 7,
        "min_exponent": -126,
    },
    "float32": {
        "rtol": 2**-10,
        "atol": 2**-16,
        "required_ratio": 0.99,
        "hard_fixed": 1e-2,
        "mantissa_bits": 23,
        "min_exponent": -126,
    },
    "float64": {
        "rtol": 1e-12,
        "atol": 1e-12,
        "required_ratio": 1.0,
        "hard_fixed": 1e-10,
        "mantissa_bits": 52,
        "min_exponent": -1022,
    },
}


def _find_tensor_specs(obj: Any, prefix: str = "root") -> list[tuple[str, TensorSpec]]:
    found: list[tuple[str, TensorSpec]] = []
    if isinstance(obj, TensorSpec):
        found.append((prefix, obj))
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            found.extend(_find_tensor_specs(value, "{}[{}]".format(prefix, repr(key))))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            found.extend(_find_tensor_specs(value, "{}[{}]".format(prefix, index)))
    return found


def _zip_member(names: Sequence[str], suffix: str) -> str:
    matches = [name for name in names if name == suffix or name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            "expected one ZIP member ending in {!r}, got {!r}".format(suffix, matches)
        )
    return matches[0]


def load_tensor_bundle(path: Path) -> dict[str, LoadedTensor]:
    """Load tensor leaves from one torch.save ZIP without importing torch."""
    if not zipfile.is_zipfile(path):
        raise RuntimeError(
            "{} is not the ZIP torch.save format; re-dump with a current PyTorch "
            "version or analyze it in an environment with a format-specific loader".format(path)
        )
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        metadata_member = _zip_member(names, "data.pkl")
        metadata = TorchMetadataUnpickler(
            io.BytesIO(archive.read(metadata_member))
        ).load()
        specs = _find_tensor_specs(metadata)
        if not specs:
            raise RuntimeError("{} contains no tensor leaves".format(path))
        raw_by_storage: dict[tuple[str, str], bytes] = {}
        result: dict[str, LoadedTensor] = {}
        for component, spec in specs:
            storage_name = spec.storage.storage_type.name
            if storage_name not in DTYPE_MAP:
                raise RuntimeError(
                    "{}: unsupported storage dtype {}".format(path, storage_name)
                )
            numpy_dtype, serialized_dtype = DTYPE_MAP[storage_name]
            cache_key = (spec.storage.key, storage_name)
            if cache_key not in raw_by_storage:
                storage_member = _zip_member(names, "data/{}".format(spec.storage.key))
                raw_by_storage[cache_key] = archive.read(storage_member)
            raw = raw_by_storage[cache_key]
            itemsize = np.dtype(numpy_dtype).itemsize
            expected_bytes = spec.storage.numel * itemsize
            if len(raw) < expected_bytes:
                raise RuntimeError(
                    "{}: storage {} is truncated: {} < {} bytes".format(
                        path, spec.storage.key, len(raw), expected_bytes
                    )
                )
            base = np.frombuffer(
                raw, dtype=np.dtype(numpy_dtype), count=spec.storage.numel
            )
            if serialized_dtype == "bfloat16":
                base = (base.astype(np.uint32) << np.uint32(16)).view(np.float32)
            if len(spec.size) != len(spec.stride):
                raise RuntimeError("{}: size/stride rank mismatch".format(path))
            element_count = math.prod(spec.size)
            if element_count == 0:
                array = np.empty(spec.size, dtype=base.dtype)
            else:
                minimum = spec.storage_offset
                maximum = spec.storage_offset
                for size, stride in zip(spec.size, spec.stride):
                    delta = (size - 1) * stride
                    minimum += min(delta, 0)
                    maximum += max(delta, 0)
                if minimum < 0 or maximum >= len(base):
                    raise RuntimeError(
                        "{}: tensor view [{}, {}] exceeds storage with {} elements".format(
                            path, minimum, maximum, len(base)
                        )
                    )
                byte_strides = tuple(
                    stride * base.dtype.itemsize for stride in spec.stride
                )
                array = np.lib.stride_tricks.as_strided(
                    base[spec.storage_offset :],
                    shape=spec.size,
                    strides=byte_strides,
                )
                if not array.flags.c_contiguous:
                    array = np.ascontiguousarray(array)
            component_key = component
            suffix = 1
            while component_key in result:
                suffix += 1
                component_key = "{}#{}".format(component, suffix)
            result[component_key] = LoadedTensor(
                path=path,
                component=component_key,
                array=array,
                serialized_dtype=serialized_dtype,
                storage_numel=spec.storage.numel,
                storage_offset=spec.storage_offset,
                stride=spec.stride,
            )
    return result


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def prepare_dump_files(
    root: Path,
    reconstruction_root: Path,
    extensions: Sequence[str],
    exclude_root: Path,
) -> dict[str, Path]:
    """Discover dumps and reconstruct byte-level ``.part.N`` groups."""
    root = root.resolve()
    extension_set = {value.lower() for value in extensions}
    regular: dict[str, Path] = {}
    parts: dict[str, list[tuple[int, Path]]] = collections.defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if _is_relative_to(resolved, exclude_root.resolve()):
            continue
        relative = path.relative_to(root).as_posix()
        match = PART_RE.match(relative)
        if match:
            base = match.group("base")
            if Path(base).suffix.lower() in extension_set:
                parts[base].append((int(match.group("part")), path))
            continue
        if path.suffix.lower() in extension_set:
            regular[relative] = path

    reconstruction_root.mkdir(parents=True, exist_ok=True)
    for relative, numbered in parts.items():
        if relative in regular:
            continue
        numbered.sort(key=lambda item: item[0])
        actual_numbers = [number for number, _ in numbered]
        expected_numbers = list(range(len(numbered)))
        if actual_numbers != expected_numbers:
            raise RuntimeError(
                "{}: non-contiguous split files {}, expected {}".format(
                    relative, actual_numbers, expected_numbers
                )
            )
        destination = reconstruction_root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_size = sum(path.stat().st_size for _, path in numbered)
        if not destination.exists() or destination.stat().st_size != expected_size:
            temporary = destination.with_name(destination.name + ".tmp")
            with temporary.open("wb") as output:
                for _, part in numbered:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            if temporary.stat().st_size != expected_size:
                raise RuntimeError("{}: reconstructed size mismatch".format(relative))
            os.replace(temporary, destination)
        regular[relative] = destination
    return regular


def _normalize_op(text: str) -> str:
    for pattern, name in OP_ALIASES:
        if pattern.search(text):
            return name
    return "unknown"


def _infer_op(text: str) -> str:
    normalized = text.lower().replace("\\", "/")
    # An operator name embedded in a call-directory is stronger evidence than
    # a tensor filename. This matters for paths such as
    # ``first_call_gmma2av_backward/hidden_states_before_a2a_gemm.pt``.
    for component in normalized.split("/"):
        if re.search(r"(?:^|[_-])call(?:[_-]|$)", component):
            op = _normalize_op(component)
            if op != "unknown":
                return op
    return _normalize_op(normalized)


def _infer_io_kind(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    tokens = normalized.split("_")
    if any(token in {"output", "outputs", "out", "result", "results"} for token in tokens):
        return "output"
    if any(token in {"input", "inputs", "in"} for token in tokens):
        return "input"
    if "after" in tokens:
        return "output"
    if "before" in tokens:
        return "input"
    if any(
        marker in normalized
        for marker in (
            "local_tokens",
            "global_tokens",
            "weight",
            "route",
            "group_list",
        )
    ) or any(token in {"x", "a", "b"} for token in tokens):
        return "input"
    return "unknown"


def _canonical_key(relative_path: str, strip_tokens: Sequence[str]) -> str:
    value = relative_path.lower().replace("\\", "/")
    value = PART_RE.sub(lambda match: match.group("base"), value)
    for suffix in DEFAULT_EXTENSIONS:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    for pattern, replacement in OP_ALIASES:
        value = pattern.sub(replacement, value)
    tokens = [token for token in re.split(r"[^a-z0-9]+", value) if token]
    discarded = {token.lower() for token in strip_tokens}
    return "_".join(token for token in tokens if token not in discarded)


def _infer_sequence(text: str) -> tuple[Optional[int], str]:
    lowered = text.lower()
    match = SEQUENCE_RE.search(lowered)
    if match:
        return int(match.group("sequence")), "explicit"
    ordinal_match = ORDINAL_CALL_RE.search(lowered)
    if ordinal_match:
        return ORDINAL_CALL_INDEX[ordinal_match.group("ordinal")], "ordinal_call"
    number_matches = list(re.finditer(r"(?<![a-z])([0-9]+)(?![a-z])", lowered))
    rank_match = RANK_RE.search(lowered)
    for number_match in reversed(number_matches):
        if (
            rank_match
            and rank_match.start("rank")
            <= number_match.start(1)
            < rank_match.end("rank")
        ):
            continue
        return int(number_match.group(1)), "numeric_filename"
    return None, "lexical_only"


def _infer_phase(text: str) -> str:
    lowered = text.lower().replace("\\", "/")
    # Inspect only directory components that describe a call or an operator.
    # Tensor filenames may contain unrelated phase words and must not change
    # invocation grouping. Descriptive tokens are allowed between ``call`` and
    # the phase, and ``call`` itself is optional when the operator is explicit,
    # for example ``first_all2all_grouped_matmul_backward_000000``.
    components = [component for component in lowered.split("/") if component]
    if components and re.search(
        r"\.(?:pt|pth|bin)(?:\.part\.[0-9]+)?$", components[-1]
    ):
        components = components[:-1]
    phase_components = [
        component
        for component in components
        if re.search(r"(?:^|[_-])call(?:[_-]|$)", component)
        or _normalize_op(component) != "unknown"
    ]
    for component in phase_components:
        tokens = {
            token for token in re.split(r"[^a-z0-9]+", component) if token
        }
        if tokens.intersection({"backward", "backwards", "back", "bwd"}):
            return "backward"
        if tokens.intersection({"forward", "forwards", "fwd"}):
            return "forward"
    if SEQUENCE_RE.search(lowered):
        return "numbered_call"
    return "unspecified"


def _infer_tensor_name(text: str, io_kind: str) -> str:
    stem = Path(text).stem.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if any(
        marker in normalized
        for marker in (
            "all_to_allv_output",
            "alltoallv_output",
            "a2av_output",
            "export_output",
            "output_export",
        )
    ):
        return "all_to_allv_output"
    for marker in (
        "global_tokens",
        "local_tokens",
        "group_list",
        "weight",
        "x",
        "input",
        "output",
        "result",
        "out",
    ):
        if re.search(r"(?:^|_){}(?:_|$)".format(re.escape(marker)), normalized):
            return marker
    return io_kind if io_kind != "unknown" else normalized


def _build_artifact(
    side: str,
    relative_path: str,
    path: Path,
    strip_tokens: Sequence[str],
    manifest_row: Optional[Mapping[str, Any]] = None,
) -> Artifact:
    text = relative_path.replace("\\", "/")
    canonical = _canonical_key(text, strip_tokens)
    row = manifest_row or {}
    rank_match = RANK_RE.search(text.lower())
    inferred_rank = int(rank_match.group("rank")) if rank_match else None
    inferred_sequence, confidence = _infer_sequence(text)
    rank = int(row["rank"]) if row.get("rank") is not None else inferred_rank
    sequence = (
        int(row["sequence"])
        if row.get("sequence") is not None
        else inferred_sequence
    )
    if row.get("sequence") is not None:
        confidence = "manifest"
    phase = str(row.get("phase") or _infer_phase(text))
    op = str(row.get("op") or _infer_op(text))
    io_kind = str(row.get("io") or row.get("io_kind") or _infer_io_kind(text))
    tensor_name = str(row.get("tensor") or _infer_tensor_name(text, io_kind))
    invocation = str(
        row.get("invocation")
        or "rank={}|sequence={}|phase={}|op={}".format(
            rank, sequence, phase, op
        )
    )
    if row.get("pair_key"):
        key = str(row["pair_key"])
    elif manifest_row is not None:
        key = "{}|io={}|tensor={}".format(invocation, io_kind, tensor_name)
    else:
        key = canonical
    return Artifact(
        side=side,
        key=key,
        relative_path=text,
        path=path,
        rank=rank,
        sequence=sequence,
        phase=phase,
        op=op,
        io_kind=io_kind,
        tensor_name=tensor_name,
        invocation=invocation,
        order_confidence=confidence,
    )


def _read_manifest(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    payload = json.loads(text)
    if isinstance(payload, Mapping):
        payload = payload.get("tensors", payload.get("records"))
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON list or JSONL records")
    return payload


def build_artifacts(
    side: str,
    root: Path,
    files: Mapping[str, Path],
    strip_tokens: Sequence[str],
    include_pattern: Optional[str],
    manifest: Optional[Path],
) -> list[Artifact]:
    include = re.compile(include_pattern) if include_pattern else None
    artifacts: list[Artifact] = []
    if manifest is not None:
        for row in _read_manifest(manifest):
            relative = str(row.get("file", "")).replace("\\", "/")
            if not relative:
                raise ValueError("manifest row is missing 'file': {!r}".format(row))
            if relative not in files:
                raise FileNotFoundError(
                    "manifest file {!r} is not present under {}".format(relative, root)
                )
            artifacts.append(
                _build_artifact(
                    side, relative, files[relative], strip_tokens, manifest_row=row
                )
            )
        return artifacts
    for relative, path in sorted(files.items()):
        if include and not include.search(relative):
            continue
        artifacts.append(_build_artifact(side, relative, path, strip_tokens))
    return artifacts


def pair_artifacts(
    fused: Sequence[Artifact], reference: Sequence[Artifact]
) -> tuple[list[tuple[Artifact, Artifact]], list[Artifact], list[Artifact]]:
    def index(values: Sequence[Artifact]) -> dict[str, Artifact]:
        result: dict[str, Artifact] = {}
        collisions: dict[str, list[str]] = collections.defaultdict(list)
        for artifact in values:
            if artifact.key in result:
                collisions[artifact.key].append(result[artifact.key].relative_path)
                collisions[artifact.key].append(artifact.relative_path)
            else:
                result[artifact.key] = artifact
        if collisions:
            details = "; ".join(
                "{} -> {}".format(key, sorted(set(paths)))
                for key, paths in sorted(collisions.items())
            )
            raise RuntimeError(
                "ambiguous canonical dump keys: {}. Use separate roots, --*-include, "
                "additional --strip-token values, or manifests with pair_key.".format(details)
            )
        return result

    fused_index = index(fused)
    reference_index = index(reference)
    common = sorted(set(fused_index) & set(reference_index))
    pairs = [(fused_index[key], reference_index[key]) for key in common]
    fused_only = [fused_index[key] for key in sorted(set(fused_index) - set(reference_index))]
    reference_only = [
        reference_index[key] for key in sorted(set(reference_index) - set(fused_index))
    ]
    return pairs, fused_only, reference_only


def _logical_position(flat_index: int, shape: tuple[int, ...]) -> dict[str, Any]:
    if not shape:
        return {"flat_index": flat_index, "index": []}
    index = [int(value) for value in np.unravel_index(flat_index, shape)]
    result: dict[str, Any] = {"flat_index": flat_index, "index": index}
    if len(shape) >= 2:
        row_width = math.prod(shape[1:])
        result["row"] = flat_index // row_width
        result["column_flat"] = flat_index % row_width
    return result


def _display_value(value: Any) -> Any:
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    return number


def _update_top_heap(
    heap: list[tuple[float, int, float, float]],
    top_k: int,
    errors: Any,
    indices: Any,
    expected: Any,
    actual: Any,
    chunk_start: int,
) -> None:
    if top_k <= 0 or len(indices) == 0:
        return
    take = min(top_k, len(indices))
    candidate_errors = errors[indices]
    selected = (
        np.arange(len(indices))
        if take == len(indices)
        else np.argpartition(candidate_errors, -take)[-take:]
    )
    for position in selected:
        local = int(indices[int(position)])
        item = (
            float(errors[local]),
            chunk_start + local,
            float(expected[local]),
            float(actual[local]),
        )
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)


def _standard_for(
    dtype: str, atol_override: Optional[float], rtol_override: Optional[float]
) -> dict[str, Any]:
    if dtype in PRECISION_STANDARDS:
        result = dict(PRECISION_STANDARDS[dtype])
        result["name"] = "opbase_experimental"
    else:
        result = {
            "name": "exact",
            "rtol": 0.0,
            "atol": 0.0,
            "required_ratio": 1.0,
            "hard_fixed": None,
            "mantissa_bits": None,
            "min_exponent": None,
        }
    if atol_override is not None:
        result["atol"] = atol_override
        result["name"] = "user_override"
    if rtol_override is not None:
        result["rtol"] = rtol_override
        result["name"] = "user_override"
    return result


def compare_tensors(
    fused: LoadedTensor,
    reference: LoadedTensor,
    *,
    chunk_elements: int,
    top_k: int,
    nonfinite_samples: int,
    atol: Optional[float],
    rtol: Optional[float],
) -> dict[str, Any]:
    actual_array = fused.array
    expected_array = reference.array
    actual_shape = tuple(int(value) for value in actual_array.shape)
    expected_shape = tuple(int(value) for value in expected_array.shape)
    actual = actual_array.reshape(-1)
    expected = expected_array.reshape(-1)
    common_numel = min(actual.size, expected.size)
    standard = _standard_for(reference.serialized_dtype, atol, rtol)
    counters: collections.Counter[str] = collections.Counter()
    top_heap: list[tuple[float, int, float, float]] = []
    nonfinite_locations: list[dict[str, Any]] = []
    first_issue: Optional[dict[str, Any]] = None
    max_abs_error = 0.0
    sum_abs_error = 0.0
    sum_square_error = 0.0
    expected_square_sum = 0.0
    actual_square_sum = 0.0
    cross_sum = 0.0
    finite_count = 0
    row_width = math.prod(expected_shape[1:]) if len(expected_shape) >= 2 else None
    row_count = expected_shape[0] if len(expected_shape) >= 2 else None
    row_issues = np.zeros(row_count, dtype=np.int64) if row_count else None
    row_new_nan = np.zeros(row_count, dtype=np.int64) if row_count else None
    column_new_nan = (
        np.zeros(row_width, dtype=np.int64)
        if row_width is not None and row_width <= 2_000_000
        else None
    )
    started = time.perf_counter()

    for start in range(0, common_numel, chunk_elements):
        end = min(start + chunk_elements, common_numel)
        e = expected[start:end]
        a = actual[start:end]
        floating = np.issubdtype(e.dtype, np.floating) or np.issubdtype(
            a.dtype, np.floating
        )
        if floating:
            e_nan = np.isnan(e)
            a_nan = np.isnan(a)
            e_posinf = np.isposinf(e)
            a_posinf = np.isposinf(a)
            e_neginf = np.isneginf(e)
            a_neginf = np.isneginf(a)
            e_finite = np.isfinite(e)
            a_finite = np.isfinite(a)
        else:
            e_nan = np.zeros(len(e), dtype=bool)
            a_nan = np.zeros(len(a), dtype=bool)
            e_posinf = np.zeros(len(e), dtype=bool)
            a_posinf = np.zeros(len(a), dtype=bool)
            e_neginf = np.zeros(len(e), dtype=bool)
            a_neginf = np.zeros(len(a), dtype=bool)
            e_finite = np.ones(len(e), dtype=bool)
            a_finite = np.ones(len(a), dtype=bool)
        both_finite = e_finite & a_finite
        same_nonfinite = (
            (e_nan & a_nan) | (e_posinf & a_posinf) | (e_neginf & a_neginf)
        )
        finite_exact = both_finite & (e == a)
        semantic_equal = finite_exact | same_nonfinite
        semantic_mismatch = ~semantic_equal
        new_nan = e_finite & a_nan
        new_posinf = e_finite & a_posinf
        new_neginf = e_finite & a_neginf
        expected_nonfinite_actual_finite = (~e_finite) & a_finite

        counters["expected_nan"] += int(np.count_nonzero(e_nan))
        counters["actual_nan"] += int(np.count_nonzero(a_nan))
        counters["expected_posinf"] += int(np.count_nonzero(e_posinf))
        counters["actual_posinf"] += int(np.count_nonzero(a_posinf))
        counters["expected_neginf"] += int(np.count_nonzero(e_neginf))
        counters["actual_neginf"] += int(np.count_nonzero(a_neginf))
        counters["introduced_nan"] += int(np.count_nonzero(new_nan))
        counters["introduced_posinf"] += int(np.count_nonzero(new_posinf))
        counters["introduced_neginf"] += int(np.count_nonzero(new_neginf))
        counters["expected_nonfinite_actual_finite"] += int(
            np.count_nonzero(expected_nonfinite_actual_finite)
        )
        counters["semantic_mismatch"] += int(np.count_nonzero(semantic_mismatch))
        counters["nonfinite_mismatch"] += int(
            np.count_nonzero(semantic_mismatch & ~both_finite)
        )

        errors = np.zeros(len(e), dtype=np.float64)
        tolerance_issue = np.zeros(len(e), dtype=bool)
        tolerance_issue |= semantic_mismatch & ~both_finite
        if np.any(both_finite):
            finite_indices = np.flatnonzero(both_finite)
            e64 = e[both_finite].astype(np.float64)
            a64 = a[both_finite].astype(np.float64)
            finite_errors = np.abs(a64 - e64)
            errors[finite_indices] = finite_errors
            threshold = float(standard["atol"]) + float(standard["rtol"]) * np.abs(e64)
            finite_tolerance_fail = finite_errors > threshold
            tolerance_issue[finite_indices] = finite_tolerance_fail
            counters["tolerance_fail"] += int(np.count_nonzero(finite_tolerance_fail))
            if standard["hard_fixed"] is not None:
                golden_abs = np.abs(e64)
                safe_abs = np.maximum(golden_abs, np.finfo(np.float64).tiny)
                exponents = np.floor(np.log2(safe_abs))
                exponents = np.maximum(exponents, standard["min_exponent"])
                ulp = np.exp2(exponents - standard["mantissa_bits"])
                hard_limit = np.maximum(32.0 * ulp, standard["hard_fixed"])
                counters["hard_limit_fail"] += int(
                    np.count_nonzero(finite_errors > hard_limit)
                )
            finite_count += len(finite_indices)
            sum_abs_error += float(np.sum(finite_errors))
            sum_square_error += float(np.dot(finite_errors, finite_errors))
            max_abs_error = max(
                max_abs_error,
                float(np.max(finite_errors)) if len(finite_errors) else 0.0,
            )
            expected_square_sum += float(np.dot(e64, e64))
            actual_square_sum += float(np.dot(a64, a64))
            cross_sum += float(np.dot(e64, a64))
            differing = np.flatnonzero(both_finite & (errors > 0))
            _update_top_heap(
                top_heap, top_k, errors, differing, e, a, start
            )
        issue = tolerance_issue | new_nan | new_posinf | new_neginf
        if first_issue is None and np.any(issue):
            local = int(np.flatnonzero(issue)[0])
            first_issue = {
                **_logical_position(start + local, expected_shape),
                "reference": _display_value(e[local]),
                "fused": _display_value(a[local]),
                "reference_finite": bool(e_finite[local]),
                "fused_finite": bool(a_finite[local]),
                "abs_error": float(errors[local]) if both_finite[local] else None,
            }
        for local_value in np.flatnonzero(new_nan | new_posinf | new_neginf):
            if len(nonfinite_locations) >= nonfinite_samples:
                break
            local = int(local_value)
            nonfinite_locations.append(
                {
                    **_logical_position(start + local, expected_shape),
                    "reference": _display_value(e[local]),
                    "fused": _display_value(a[local]),
                }
            )
        if row_width is not None:
            absolute_indices = np.flatnonzero(issue) + start
            if len(absolute_indices) and row_issues is not None:
                row_issues += np.bincount(
                    absolute_indices // row_width, minlength=len(row_issues)
                )[: len(row_issues)]
            new_nan_indices = np.flatnonzero(new_nan) + start
            if len(new_nan_indices) and row_new_nan is not None:
                row_new_nan += np.bincount(
                    new_nan_indices // row_width, minlength=len(row_new_nan)
                )[: len(row_new_nan)]
            if len(new_nan_indices) and column_new_nan is not None:
                column_new_nan += np.bincount(
                    new_nan_indices % row_width, minlength=len(column_new_nan)
                )[: len(column_new_nan)]

    missing_or_extra = abs(int(actual.size) - int(expected.size))
    counters["missing_or_extra_elements"] = missing_or_extra
    compared = max(1, common_numel)
    # Finite tolerance failures plus any differing nonfinite category.  The
    # latter includes finite->NaN/Inf, nonfinite->finite and NaN<->Inf changes.
    mismatched_nonfinite = counters["nonfinite_mismatch"]
    tolerance_fail_total = counters["tolerance_fail"] + mismatched_nonfinite
    matched_ratio = max(0.0, (common_numel - tolerance_fail_total) / compared)
    exact_equal = (
        actual_shape == expected_shape
        and fused.serialized_dtype == reference.serialized_dtype
        and counters["semantic_mismatch"] == 0
    )
    precision_passed = (
        actual_shape == expected_shape
        and fused.serialized_dtype == reference.serialized_dtype
        and matched_ratio >= float(standard["required_ratio"])
        and counters["hard_limit_fail"] == 0
        and mismatched_nonfinite == 0
    )
    actual_nonfinite = (
        counters["actual_nan"]
        + counters["actual_posinf"]
        + counters["actual_neginf"]
    )
    reference_nonfinite = (
        counters["expected_nan"]
        + counters["expected_posinf"]
        + counters["expected_neginf"]
    )

    top_mismatches = []
    for error, flat_index, expected_value, actual_value in sorted(
        top_heap, reverse=True
    ):
        top_mismatches.append(
            {
                **_logical_position(flat_index, expected_shape),
                "reference": expected_value,
                "fused": actual_value,
                "abs_error": error,
                "rel_error": (
                    error / abs(expected_value)
                    if expected_value != 0
                    else (None if error else 0.0)
                ),
            }
        )

    row_distribution: list[dict[str, Any]] = []
    row_runs: list[dict[str, Any]] = []
    periodicity: list[dict[str, Any]] = []
    if row_issues is not None:
        bad_rows = np.flatnonzero(row_issues > 0)
        ordered_rows = sorted(
            (int(row) for row in bad_rows),
            key=lambda row: (int(row_issues[row]), int(row_new_nan[row])),
            reverse=True,
        )[:TOP_ROW_LIMIT]
        row_distribution = [
            {
                "row": row,
                "issues": int(row_issues[row]),
                "introduced_nan": int(row_new_nan[row]),
            }
            for row in ordered_rows
        ]
        if len(bad_rows):
            run_start = int(bad_rows[0])
            previous = run_start
            for value in bad_rows[1:]:
                row = int(value)
                if row != previous + 1:
                    row_runs.append(
                        {
                            "row_start": run_start,
                            "row_end": previous + 1,
                            "rows": previous - run_start + 1,
                        }
                    )
                    run_start = row
                previous = row
            row_runs.append(
                {"row_start": run_start, "row_end": previous + 1, "rows": previous - run_start + 1}
            )
        for period in (16, 32, 64, 128, 256):
            if period > len(row_issues):
                continue
            issue_counts = [int(np.sum(row_issues[offset::period])) for offset in range(period)]
            top_offsets = sorted(
                range(period),
                key=lambda offset: issue_counts[offset],
                reverse=True,
            )[:8]
            periodicity.append(
                {
                    "period": period,
                    "top_residues": [
                        {"residue": offset, "issues": issue_counts[offset]}
                        for offset in top_offsets
                        if issue_counts[offset]
                    ],
                }
            )
    column_distribution: list[dict[str, Any]] = []
    if column_new_nan is not None:
        columns = np.flatnonzero(column_new_nan > 0)
        ordered = sorted(
            (int(column) for column in columns),
            key=lambda column: int(column_new_nan[column]),
            reverse=True,
        )[:TOP_ROW_LIMIT]
        column_distribution = [
            {"column_flat": column, "introduced_nan": int(column_new_nan[column])}
            for column in ordered
        ]

    cosine = (
        cross_sum / math.sqrt(expected_square_sum * actual_square_sum)
        if expected_square_sum > 0 and actual_square_sum > 0
        else None
    )
    relative_l2 = (
        math.sqrt(sum_square_error / expected_square_sum)
        if expected_square_sum > 0
        else None
    )
    return {
        "fused_file": str(fused.path),
        "reference_file": str(reference.path),
        "fused_component": fused.component,
        "reference_component": reference.component,
        "fused_shape": actual_shape,
        "reference_shape": expected_shape,
        "shape_equal": actual_shape == expected_shape,
        "fused_dtype": fused.serialized_dtype,
        "reference_dtype": reference.serialized_dtype,
        "dtype_equal": fused.serialized_dtype == reference.serialized_dtype,
        "fused_stride": fused.stride,
        "reference_stride": reference.stride,
        "fused_storage_offset": fused.storage_offset,
        "reference_storage_offset": reference.storage_offset,
        "fused_numel": int(actual.size),
        "reference_numel": int(expected.size),
        "compared_numel": int(common_numel),
        "exact_equal": exact_equal,
        "precision_passed": precision_passed,
        "tensor_valid": actual_nonfinite == 0,
        "reference_tensor_valid": reference_nonfinite == 0,
        "matched_ratio": matched_ratio,
        "precision_standard": standard,
        "counters": dict(counters),
        "introduced_nonfinite": int(
            counters["introduced_nan"]
            + counters["introduced_posinf"]
            + counters["introduced_neginf"]
        ),
        "actual_nonfinite": int(actual_nonfinite),
        "reference_nonfinite": int(reference_nonfinite),
        "max_abs_error_both_finite": max_abs_error,
        "mean_abs_error_both_finite": (
            sum_abs_error / finite_count if finite_count else 0.0
        ),
        "rmse_both_finite": (
            math.sqrt(sum_square_error / finite_count) if finite_count else 0.0
        ),
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "first_issue": first_issue,
        "top_mismatches": top_mismatches,
        "introduced_nonfinite_locations": nonfinite_locations,
        "row_distribution": row_distribution,
        "row_runs": row_runs[:TOP_ROW_LIMIT],
        "row_runs_total": len(row_runs),
        "column_new_nan_distribution": column_distribution,
        "row_periodicity": periodicity,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _component_pairs(
    fused: Mapping[str, LoadedTensor], reference: Mapping[str, LoadedTensor]
) -> tuple[list[tuple[LoadedTensor, LoadedTensor]], list[str], list[str]]:
    fused_keys = set(fused)
    reference_keys = set(reference)
    if len(fused) == 1 and len(reference) == 1:
        return [(next(iter(fused.values())), next(iter(reference.values())))], [], []
    common = sorted(fused_keys & reference_keys)
    return (
        [(fused[key], reference[key]) for key in common],
        sorted(fused_keys - reference_keys),
        sorted(reference_keys - fused_keys),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"([0-9]+)", value.lower())
    )


def build_timeline(comparisons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for comparison in comparisons:
        grouped[str(comparison["invocation"])].append(comparison)
    timeline: list[dict[str, Any]] = []
    for invocation, records in grouped.items():
        inputs = [record for record in records if record["io_kind"] == "input"]
        outputs = [record for record in records if record["io_kind"] == "output"]
        inputs_exact = bool(inputs) and all(record.get("exact_equal", False) for record in inputs)
        input_diverged = any(not record.get("exact_equal", False) for record in inputs)
        output_diverged = any(
            not record.get(
                "comparison_passed", record.get("precision_passed", False)
            )
            for record in outputs
        )
        unknowns = [record for record in records if record["io_kind"] == "unknown"]
        unknown_diverged = any(
            not record.get(
                "comparison_passed", record.get("precision_passed", False)
            )
            for record in unknowns
        )
        introduced_nonfinite = sum(
            int(record.get("introduced_nonfinite", 0)) for record in outputs
        )
        inherited_nonfinite = sum(
            int(record.get("reference_nonfinite", 0)) for record in inputs
        )
        if input_diverged:
            classification = "input_already_diverged"
        elif outputs and inputs_exact and introduced_nonfinite:
            classification = "operator_introduced_nonfinite"
        elif outputs and inputs_exact and output_diverged:
            classification = "operator_output_diverged"
        elif outputs and inputs_exact:
            classification = "matched_boundary"
        elif outputs and output_diverged:
            classification = "output_diverged_inputs_unclassified"
        elif unknown_diverged:
            classification = "unclassified_tensor_diverged"
        else:
            classification = "insufficient_boundary_metadata"
        first = records[0]
        timeline.append(
            {
                "invocation": invocation,
                "rank": first.get("rank"),
                "sequence": first.get("sequence"),
                "phase": first.get("phase"),
                "op": first.get("op"),
                "order_confidence": first.get("order_confidence"),
                "input_tensor_count": len(inputs),
                "output_tensor_count": len(outputs),
                "inputs_exact": inputs_exact,
                "input_diverged": input_diverged,
                "output_diverged": output_diverged,
                "introduced_nonfinite": introduced_nonfinite,
                "inherited_input_nonfinite": inherited_nonfinite,
                "classification": classification,
                "bad_tensors": [
                    str(record["tensor_name"])
                    for record in records
                    if not (
                        record.get("exact_equal", False)
                        if record["io_kind"] == "input"
                        else record.get(
                            "comparison_passed",
                            record.get("precision_passed", False),
                        )
                    )
                ],
            }
        )
    timeline.sort(
        key=lambda row: (
            row["sequence"] is None,
            row["sequence"] if row["sequence"] is not None else sys.maxsize,
            PHASE_ORDER.get(str(row.get("phase")), 3),
            OP_PIPELINE_ORDER.get(str(row["op"]), 2),
            row["rank"] if row["rank"] is not None else sys.maxsize,
            _natural_key(str(row["invocation"])),
        )
    )
    return timeline


def derive_diagnosis(
    timeline: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]]
) -> tuple[Optional[Mapping[str, Any]], list[str]]:
    issue_classes = {
        "input_already_diverged",
        "operator_introduced_nonfinite",
        "operator_output_diverged",
        "output_diverged_inputs_unclassified",
        "unclassified_tensor_diverged",
    }
    first = next(
        (row for row in timeline if row.get("classification") in issue_classes), None
    )
    hints: list[str] = []
    if first is None:
        hints.append(
            "No fused/reference divergence was found at a paired tensor boundary. "
            "If the model still becomes NaN, dump the immediately following consumer input."
        )
        return None, hints
    classification = str(first["classification"])
    if first.get("order_confidence") == "lexical_only":
        hints.append(
            "The earliest boundary is based on lexical path order because no call sequence "
            "was encoded. Use manifests before treating it as the true first runtime event."
        )
    if classification == "input_already_diverged":
        hints.append(
            "The first bad invocation already receives a different input in the fused run. "
            "The current operator is not yet isolated: inspect the previous producer or verify "
            "that both runs use identical seeds, routing, micro-batch order and weights."
        )
    elif classification == "operator_introduced_nonfinite":
        hints.append(
            "All paired inputs at the first bad invocation are bitwise identical, while a "
            "reference-finite output becomes NaN/Inf. This localizes creation of the nonfinite "
            "value to this fused invocation (compute, UDMA/staging synchronization, layout, or "
            "output write), rather than to upstream model data."
        )
    elif classification == "operator_output_diverged":
        hints.append(
            "All paired inputs at the first bad invocation are bitwise identical, but the "
            "output fails the dtype precision standard. This localizes the first numerical "
            "divergence to this fused invocation."
        )
    elif classification == "output_diverged_inputs_unclassified":
        hints.append(
            "An output first diverges, but its input files were not recognized. Add io/tensor "
            "fields through manifests before attributing the fault to the operator."
        )
    else:
        hints.append(
            "A tensor first diverges, but its input/output role was not recognized from the "
            "path. Add io/tensor fields through manifests to turn this into an operator-boundary "
            "causal conclusion."
        )

    invocation_records = [
        row for row in comparisons if row.get("invocation") == first.get("invocation")
    ]
    export_records = [
        row
        for row in invocation_records
        if row.get("io_kind") == "output"
        and "all_to_allv" in str(row.get("tensor_name", ""))
    ]
    final_records = [
        row
        for row in invocation_records
        if row.get("io_kind") == "output" and row not in export_records
    ]
    if first.get("op") == "all_to_allv_grouped_mat_mul" and export_records:
        if any(not row.get("comparison_passed", False) for row in export_records):
            hints.append(
                "The exported AllToAllV activation is already bad. Prioritize route offsets, "
                "remote payload visibility, stage lifetime and communication completion before "
                "the grouped matmul arithmetic."
            )
        elif any(not row.get("comparison_passed", False) for row in final_records):
            hints.append(
                "The exported AllToAllV activation matches but the final GMM output does not. "
                "Prioritize the GMM read/layout/tail path after communication."
            )
    if (
        first.get("op") == "grouped_mat_mul_all_to_allv"
        and classification
        in {"operator_introduced_nonfinite", "operator_output_diverged"}
    ):
        hints.append(
            "For GroupedMatMul-AllToAllV, input/final-output dumps isolate the fused call but "
            "cannot alone separate local GMM staging production from remote UDMA GET/layout. "
            "If a finer split is needed, dump the local producer staging (before GET) for only "
            "this first-bad sequence."
        )
    bad_ranks = sorted(
        {
            int(row["rank"])
            for row in timeline
            if row.get("sequence") == first.get("sequence")
            and row.get("phase") == first.get("phase")
            and row.get("op") == first.get("op")
            and row.get("classification") in issue_classes
            and row.get("rank") is not None
        }
    )
    if bad_ranks:
        hints.append(
            "At the first bad sequence, affected ranks are {}. A strict subset of ranks is "
            "more consistent with routing/communication/synchronization than with deterministic "
            "local GEMM arithmetic.".format(bad_ranks)
        )
    hints.append(
        "One fused/reference dump pair cannot prove a race. Repeat only the isolated first-bad "
        "invocation if you need to distinguish deterministic offset/layout errors from changing "
        "cross-rank mismatch locations."
    )
    return first, hints


def _format_markdown(
    summary: Mapping[str, Any], timeline: Sequence[Mapping[str, Any]]
) -> str:
    first = summary.get("first_divergence")
    lines = [
        "# Model dump comparison",
        "",
        "## Outcome",
        "",
    ]
    if first:
        lines.append(
            "First reported divergence: sequence `{}`, phase `{}`, rank `{}`, operator `{}`, "
            "classification `{}`.".format(
                first.get("sequence"),
                first.get("phase"),
                first.get("rank"),
                first.get("op"),
                first.get("classification"),
            )
        )
    else:
        lines.append("No divergence was found among paired tensor boundaries.")
    lines.extend(["", "## Diagnosis", ""])
    lines.extend("- {}".format(value) for value in summary["diagnosis"])
    lines.extend(
        [
            "",
            "## Inventory",
            "",
            "- Paired files: {}".format(summary["paired_files"]),
            "- Tensor comparisons: {}".format(summary["tensor_comparisons"]),
            "- Unmatched fused files: {}".format(summary["unmatched_fused_files"]),
            "- Unmatched reference files: {}".format(summary["unmatched_reference_files"]),
            "- Load/comparison errors: {}".format(summary["comparison_errors"]),
            "",
            "## Timeline (first 200 invocations)",
            "",
            "| sequence | phase | rank | op | classification | inputs exact | "
            "new nonfinite | bad tensors |",
            "|---:|---|---:|---|---|---|---:|---|",
        ]
    )
    for row in timeline[:200]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row.get("sequence"),
                row.get("phase"),
                row.get("rank"),
                row.get("op"),
                row.get("classification"),
                row.get("inputs_exact"),
                row.get("introduced_nonfinite"),
                ", ".join(row.get("bad_tensors", [])),
            )
        )
    lines.extend(
        [
            "",
            "Detailed evidence is in `comparisons.csv`, `timeline.csv`, "
            "`top_mismatches.csv`, `introduced_nonfinite.csv`, "
            "`row_distribution.csv`, `row_runs.csv`, and `summary.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fused UDMA and unfused model tensor dumps, then locate the "
            "first divergent operator boundary across ranks."
        )
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        help=(
            "single model dump root containing rank_*/mc2 and rank_*/mc2_udma; "
            "mc2 is the reference and mc2_udma is the fused result"
        ),
    )
    parser.add_argument("--fused-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="report directory (default: <fused-root>/_model_dump_analysis)",
    )
    parser.add_argument(
        "--fused-include", help="optional regular expression applied to fused relative paths"
    )
    parser.add_argument(
        "--reference-include",
        help="optional regular expression applied to reference relative paths",
    )
    parser.add_argument("--fused-manifest", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument(
        "--strip-token",
        action="append",
        default=[],
        help="additional filename token ignored while auto-pairing; repeat as needed",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="comma-separated torch-save extensions",
    )
    parser.add_argument("--chunk-elements", type=int, default=1_048_576)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--nonfinite-samples", type=int, default=200)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument(
        "--fail-on-divergence",
        action="store_true",
        help="return exit code 2 when a paired boundary diverges",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.chunk_elements <= 0 or args.top_k < 0 or args.nonfinite_samples < 0:
        raise ValueError("chunk-elements must be positive; sample counts must be non-negative")
    if bool(args.fused_manifest) != bool(args.reference_manifest):
        raise ValueError("provide both --fused-manifest and --reference-manifest")
    if args.dump_dir is not None:
        if args.fused_root is not None or args.reference_root is not None:
            raise ValueError(
                "--dump-dir cannot be combined with --fused-root/--reference-root"
            )
        fused_root = args.dump_dir.resolve()
        reference_root = fused_root
        fused_include = args.fused_include or r"(?:^|/)mc2_udma(?:/|$)"
        reference_include = args.reference_include or r"(?:^|/)mc2(?:/|$)"
    else:
        if args.fused_root is None or args.reference_root is None:
            raise ValueError(
                "provide --dump-dir, or provide both --fused-root and --reference-root"
            )
        fused_root = args.fused_root.resolve()
        reference_root = args.reference_root.resolve()
        fused_include = args.fused_include
        reference_include = args.reference_include
    if not fused_root.is_dir() or not reference_root.is_dir():
        raise FileNotFoundError("both dump roots must exist and be directories")
    if fused_root == reference_root and not (fused_include and reference_include):
        raise ValueError(
            "when both sides use the same root, provide --fused-include and "
            "--reference-include so files are not compared with themselves"
        )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (fused_root / "_model_dump_analysis").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    extensions = tuple(
        value.strip().lower()
        if value.strip().startswith(".")
        else "." + value.strip().lower()
        for value in args.extensions.split(",")
        if value.strip()
    )
    strip_tokens = tuple(DEFAULT_STRIP_TOKENS) + tuple(args.strip_token)
    print("Discovering fused dumps under {}".format(fused_root), flush=True)
    fused_files = prepare_dump_files(
        fused_root, output_dir / "_reconstructed" / "fused", extensions, output_dir
    )
    print("Discovering reference dumps under {}".format(reference_root), flush=True)
    reference_files = prepare_dump_files(
        reference_root,
        output_dir / "_reconstructed" / "reference",
        extensions,
        output_dir,
    )
    fused_artifacts = build_artifacts(
        "fused",
        fused_root,
        fused_files,
        strip_tokens,
        fused_include,
        args.fused_manifest,
    )
    reference_artifacts = build_artifacts(
        "reference",
        reference_root,
        reference_files,
        strip_tokens,
        reference_include,
        args.reference_manifest,
    )
    pairs, fused_only, reference_only = pair_artifacts(
        fused_artifacts, reference_artifacts
    )
    if not pairs:
        raise RuntimeError(
            "no dump pairs found; inspect unmatched_files.csv or use manifests/pair_key"
        )
    print(
        "Paired {} files ({} fused-only, {} reference-only)".format(
            len(pairs), len(fused_only), len(reference_only)
        ),
        flush=True,
    )

    comparisons: list[dict[str, Any]] = []
    comparison_errors: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    nonfinite_rows: list[dict[str, Any]] = []
    row_distribution_rows: list[dict[str, Any]] = []
    row_run_rows: list[dict[str, Any]] = []
    column_distribution_rows: list[dict[str, Any]] = []
    for pair_index, (fused_artifact, reference_artifact) in enumerate(pairs):
        print(
            "[{}/{}] {}".format(pair_index + 1, len(pairs), fused_artifact.key),
            flush=True,
        )
        try:
            fused_bundle = load_tensor_bundle(fused_artifact.path)
            reference_bundle = load_tensor_bundle(reference_artifact.path)
            component_pairs, fused_components, reference_components = _component_pairs(
                fused_bundle, reference_bundle
            )
            if fused_components or reference_components:
                comparison_errors.append(
                    {
                        "key": fused_artifact.key,
                        "error": "unmatched tensor components",
                        "fused_only_components": fused_components,
                        "reference_only_components": reference_components,
                    }
                )
            for fused_tensor, reference_tensor in component_pairs:
                result = compare_tensors(
                    fused_tensor,
                    reference_tensor,
                    chunk_elements=args.chunk_elements,
                    top_k=args.top_k,
                    nonfinite_samples=args.nonfinite_samples,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                exact_policy = (
                    fused_artifact.io_kind == "input"
                    or (
                        fused_artifact.op == "all_to_allv_grouped_mat_mul"
                        and fused_artifact.tensor_name == "all_to_allv_output"
                    )
                    or result["reference_dtype"] not in PRECISION_STANDARDS
                )
                result["comparison_policy"] = (
                    "bitwise_exact" if exact_policy else "opbase_precision"
                )
                result["comparison_passed"] = (
                    result["exact_equal"]
                    if exact_policy
                    else result["precision_passed"]
                )
                comparison_key = "{}::{}".format(
                    fused_artifact.key, fused_tensor.component
                )
                record = {
                    "comparison_key": comparison_key,
                    "pair_key": fused_artifact.key,
                    "rank": fused_artifact.rank,
                    "sequence": fused_artifact.sequence,
                    "phase": fused_artifact.phase,
                    "op": fused_artifact.op,
                    "io_kind": fused_artifact.io_kind,
                    "tensor_name": fused_artifact.tensor_name,
                    "invocation": fused_artifact.invocation,
                    "order_confidence": fused_artifact.order_confidence,
                    "fused_relative_path": fused_artifact.relative_path,
                    "reference_relative_path": reference_artifact.relative_path,
                    **result,
                }
                comparisons.append(record)
                for mismatch in result["top_mismatches"]:
                    top_rows.append(
                        {"comparison_key": comparison_key, **mismatch}
                    )
                for location in result["introduced_nonfinite_locations"]:
                    nonfinite_rows.append(
                        {"comparison_key": comparison_key, **location}
                    )
                for row_item in result["row_distribution"]:
                    row_distribution_rows.append(
                        {"comparison_key": comparison_key, **row_item}
                    )
                for run in result["row_runs"]:
                    row_run_rows.append(
                        {"comparison_key": comparison_key, **run}
                    )
                for column in result["column_new_nan_distribution"]:
                    column_distribution_rows.append(
                        {"comparison_key": comparison_key, **column}
                    )
            del fused_bundle, reference_bundle
        except Exception as error:  # keep other ranks/calls analyzable
            comparison_errors.append(
                {
                    "key": fused_artifact.key,
                    "fused_file": str(fused_artifact.path),
                    "reference_file": str(reference_artifact.path),
                    "error": "{}: {}".format(type(error).__name__, error),
                }
            )

    timeline = build_timeline(comparisons)
    first_divergence, diagnosis = derive_diagnosis(timeline, comparisons)
    unmatched_rows = [
        {"side": "fused", "key": item.key, "file": item.relative_path}
        for item in fused_only
    ] + [
        {"side": "reference", "key": item.key, "file": item.relative_path}
        for item in reference_only
    ]
    comparison_table = [
        {
            "comparison_key": row["comparison_key"],
            "rank": row["rank"],
            "sequence": row["sequence"],
            "phase": row["phase"],
            "op": row["op"],
            "io_kind": row["io_kind"],
            "tensor_name": row["tensor_name"],
            "fused_shape": row["fused_shape"],
            "reference_shape": row["reference_shape"],
            "fused_dtype": row["fused_dtype"],
            "reference_dtype": row["reference_dtype"],
            "exact_equal": row["exact_equal"],
            "precision_passed": row["precision_passed"],
            "comparison_policy": row["comparison_policy"],
            "comparison_passed": row["comparison_passed"],
            "tensor_valid": row["tensor_valid"],
            "reference_tensor_valid": row["reference_tensor_valid"],
            "introduced_nonfinite": row["introduced_nonfinite"],
            "actual_nonfinite": row["actual_nonfinite"],
            "reference_nonfinite": row["reference_nonfinite"],
            "matched_ratio": row["matched_ratio"],
            "max_abs_error_both_finite": row["max_abs_error_both_finite"],
            "mean_abs_error_both_finite": row["mean_abs_error_both_finite"],
            "relative_l2_error": row["relative_l2_error"],
            "cosine_similarity": row["cosine_similarity"],
            "first_issue": row["first_issue"],
            "fused_file": row["fused_relative_path"],
            "reference_file": row["reference_relative_path"],
        }
        for row in comparisons
    ]
    summary = {
        "fused_root": str(fused_root),
        "reference_root": str(reference_root),
        "paired_files": len(pairs),
        "tensor_comparisons": len(comparisons),
        "unmatched_fused_files": len(fused_only),
        "unmatched_reference_files": len(reference_only),
        "comparison_errors": len(comparison_errors),
        "first_divergence": first_divergence,
        "diagnosis": diagnosis,
        "timeline": timeline,
        "comparisons": comparisons,
        "unmatched_files": unmatched_rows,
        "errors": comparison_errors,
    }
    safe_summary = _json_safe(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(safe_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "comparisons.csv", comparison_table)
    _write_csv(output_dir / "timeline.csv", timeline)
    _write_csv(output_dir / "top_mismatches.csv", top_rows)
    _write_csv(output_dir / "introduced_nonfinite.csv", nonfinite_rows)
    _write_csv(output_dir / "row_distribution.csv", row_distribution_rows)
    _write_csv(output_dir / "row_runs.csv", row_run_rows)
    _write_csv(
        output_dir / "column_new_nan_distribution.csv", column_distribution_rows
    )
    _write_csv(output_dir / "unmatched_files.csv", unmatched_rows)
    _write_csv(output_dir / "comparison_errors.csv", comparison_errors)
    report = _format_markdown(safe_summary, timeline)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n{}".format(report), flush=True)
    print("Reports: {}".format(output_dir), flush=True)
    has_divergence = any(
        row["classification"]
        not in {"matched_boundary", "insufficient_boundary_metadata"}
        for row in timeline
    )
    if args.fail_on_divergence and has_divergence:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
