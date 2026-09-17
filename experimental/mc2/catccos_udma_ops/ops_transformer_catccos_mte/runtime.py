# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Process-local MTE lifecycle; importing this module is CPU-only."""

import ipaddress
import os
import socket
import threading
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from typing import Optional

from . import _build_versions
from ._loader import load_extension


@dataclass(frozen=True)
class MteConfig:
    rank: int
    world_size: int
    init_method: str
    heap_size: int
    device: str


_config: Optional[MteConfig] = None
_anchor = None
_pending_override = None
_state_lock = threading.RLock()


def _synchronized(function):
    @wraps(function)
    def call(*args, **kwargs):
        with _state_lock:
            return function(*args, **kwargs)
    return call


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _resolve_master_addr(host):
    try:
        return str(ipaddress.IPv4Address(host))
    except ipaddress.AddressValueError:
        try:
            return socket.gethostbyname(host)
        except OSError as error:
            raise RuntimeError("Set MTE_MASTER_ADDR to a reachable numeric IPv4 address") from error


@_synchronized
def init_mte_group(*, rank=None, world_size=None, local_rank=None, init_method=None,
                   heap_size=1 << 30):
    global _config, _anchor
    rank = _env_int("RANK", 0) if rank is None else rank
    world_size = _env_int("WORLD_SIZE", 1) if world_size is None else world_size
    local_rank = _env_int("LOCAL_RANK", rank) if local_rank is None else local_rank
    for name, value in (("rank", rank), ("world_size", world_size), ("local_rank", local_rank),
                        ("heap_size", heap_size)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(name + " must be an integer")
    if not 1 <= world_size <= 8 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("MTE requires 1..8 ranks, a valid rank and nonnegative local_rank")
    if heap_size <= (400 << 20) + (128 << 10):
        raise ValueError("MTE heap must exceed the 400 MiB data + 128 KiB control workspace")
    if init_method is None:
        host = _resolve_master_addr(os.environ.get("MTE_MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1")))
        port = _env_int("MTE_MASTER_PORT", _env_int("MASTER_PORT", 29500) + 1)
        if not 1 <= port <= 65535:
            raise ValueError("MTE_MASTER_PORT must be in [1, 65535]")
        init_method = "tcp://{}:{}".format(host, port)
    requested = MteConfig(rank, world_size, init_method, heap_size, "npu:{}".format(local_rank))
    if _config is not None:
        if requested != _config:
            raise RuntimeError("MTE is already initialized with a different configuration")
        return _config
    load_extension()
    import torch
    import torch_npu
    torch_npu.npu.set_device(local_rank)
    anchor = torch.empty(0, dtype=torch.uint8, device=requested.device)
    torch.ops.ops_transformer_catccos_mte._configure_mte(anchor, rank, world_size, init_method, heap_size)
    try:
        if _pending_override is not None:
            torch.ops.ops_transformer_catccos_mte._set_mte_tuning_override(anchor, list(_pending_override))
    except Exception:
        torch.ops.ops_transformer_catccos_mte._finalize_mte(anchor)
        raise
    _anchor, _config = anchor, requested
    return requested


@_synchronized
def destroy_mte_group():
    global _anchor, _config, _pending_override
    if _anchor is not None:
        import torch
        torch.ops.ops_transformer_catccos_mte._finalize_mte(_anchor)
    _anchor = _config = _pending_override = None


@_synchronized
def get_mte_config():
    if _config is None:
        raise RuntimeError("MTE is not initialized; call init_mte_group() first")
    return _config


@_synchronized
def empty_symmetric(shape, *, dtype=None):
    get_mte_config()
    import torch
    return torch.ops.ops_transformer_catccos_mte._empty_symmetric(
        _anchor, list(shape), torch.bfloat16 if dtype is None else dtype)


@contextmanager
def mte_group(**kwargs):
    config = init_mte_group(**kwargs)
    try:
        yield config
    finally:
        destroy_mte_group()


def _require_tuning():
    if not _build_versions.WITH_TUNING:
        raise RuntimeError("MTE tuning is disabled; rebuild with --backend mte --with-tuning")


@_synchronized
def _set_tuning_override(config):
    """Private tuning bridge; queue before init or apply to the active runtime."""
    global _pending_override
    _require_tuning()
    values = tuple(config)
    if len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
        raise ValueError("MTE tuning requires four uint32 integers")
    if any(v < 0 or v > 0xFFFFFFFF for v in values):
        raise ValueError("MTE tuning values must fit uint32")
    aivs, workers, chunk, stages = values
    if workers < 1 or chunk < 32 or chunk % 32 or stages not in (1, 2):
        raise ValueError("invalid MTE worker/chunk/stage configuration")
    if chunk * stages + 2048 + 8192 > 256 * 1024:
        raise ValueError("MTE buffers exceed UB after scratch and system reserve")
    if _anchor is not None:
        import torch
        torch.ops.ops_transformer_catccos_mte._set_mte_tuning_override(_anchor, list(values))
    _pending_override = values


@_synchronized
def _clear_tuning_override():
    global _pending_override
    _require_tuning()
    if _anchor is not None:
        import torch
        torch.ops.ops_transformer_catccos_mte._clear_mte_tuning_override(_anchor)
    _pending_override = None


@_synchronized
def _get_requested_comm_config():
    if _anchor is not None:
        import torch
        return tuple(torch.ops.ops_transformer_catccos_mte._get_mte_comm_config(_anchor))
    if _pending_override is not None:
        return _pending_override
    config = get_mte_config()
    return (config.world_size - 1, 1, 65536, 2)


@_synchronized
def _get_tuning_platform():
    get_mte_config()
    import torch
    block_dim, ub_bytes = torch.ops.ops_transformer_catccos_mte._get_mte_platform(_anchor)
    return {"block_dim": int(block_dim), "ub_bytes": int(ub_bytes)}
