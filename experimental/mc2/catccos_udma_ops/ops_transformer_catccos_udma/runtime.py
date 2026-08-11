# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Process-local lifecycle for the CatCCOS UDMA communication domain."""

import ipaddress
import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass(frozen=True)
class UdmaConfig:
    rank: int
    world_size: int
    init_method: str
    heap_size: int
    device: str


_config: Optional[UdmaConfig] = None
_anchor: Optional[torch.Tensor] = None


def _env_int(name: str, default: Optional[int] = None) -> int:
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise RuntimeError(f"{name} is required")
        return default
    return int(value)


def _resolve_master_addr(host: str) -> str:
    """Return the numeric IPv4 address required by the SHMEM bootstrap."""
    try:
        return str(ipaddress.IPv4Address(host))
    except ipaddress.AddressValueError:
        try:
            return socket.gethostbyname(host)
        except OSError as error:
            raise RuntimeError(
                f"Unable to resolve UDMA bootstrap host {host!r} to an IPv4 address. "
                "Set UDMA_MASTER_ADDR to a reachable numeric IPv4 address."
            ) from error


def init_udma_group(
    *,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
    init_method: Optional[str] = None,
    heap_size: int = 1 << 30,
) -> UdmaConfig:
    """Initialize one UDMA PE; call once in each ``torchrun`` worker."""
    global _anchor, _config

    rank = _env_int("RANK", 0) if rank is None else rank
    world_size = _env_int("WORLD_SIZE", 1) if world_size is None else world_size
    local_rank = _env_int("LOCAL_RANK", rank) if local_rank is None else local_rank
    if init_method is None:
        host = os.environ.get(
            "UDMA_MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1")
        )
        host = _resolve_master_addr(host)
        default_port = str(int(os.environ.get("MASTER_PORT", "29500")) + 1)
        port = os.environ.get("UDMA_MASTER_PORT", default_port)
        init_method = f"tcp://{host}:{port}"
    requested = UdmaConfig(rank, world_size, init_method, heap_size, f"npu:{local_rank}")

    if _config is not None:
        if requested != _config:
            raise RuntimeError(f"UDMA is already initialized as {_config}, requested {requested}")
        return _config

    import torch_npu

    torch_npu.npu.set_device(local_rank)
    anchor = torch.empty(0, dtype=torch.uint8, device=requested.device)
    torch.ops.ops_transformer_catccos_udma._configure_udma(
        anchor, rank, world_size, init_method, heap_size
    )
    _anchor = anchor
    _config = requested
    return requested


def destroy_udma_group() -> None:
    """Synchronize outstanding UDMA work and release the symmetric heap."""
    global _anchor, _config
    if _anchor is None:
        return
    torch.ops.ops_transformer_catccos_udma._finalize_udma(_anchor)
    _anchor = None
    _config = None


def get_udma_config() -> UdmaConfig:
    if _config is None:
        raise RuntimeError("UDMA is not initialized; call init_udma_group() first")
    return _config


@contextmanager
def udma_group(**kwargs) -> Iterator[UdmaConfig]:
    config = init_udma_group(**kwargs)
    try:
        yield config
    finally:
        destroy_udma_group()
