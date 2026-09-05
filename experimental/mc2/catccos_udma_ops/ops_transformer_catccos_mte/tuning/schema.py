# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Strict configuration schema; no accelerator imports or silent clamping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Mapping

FIELDS = ("commAivCount", "workersPerPeer", "commChunkBytes", "commUbStages")
OPERATORS = ("all_to_allv_grouped_mat_mul", "grouped_mat_mul_all_to_allv")
DEFAULT_SPACE = {
    "commAivCount": [1, 2, 4, 7, 8, 14, 16, 21, 24, 28, 31],
    "workersPerPeer": [1, 2, 3, 4],
    "commChunkBytes": [4096, 8192, 16384, 32768, 65536, 122880],
    "commUbStages": [1, 2],
}


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Platform:
    block_dim: int = 32
    ub_bytes: int = 256 * 1024
    system_reserved_bytes: int = 8192
    payload_offset_bytes: int = 2048

    def __post_init__(self):
        integer(self.block_dim, "block_dim")
        integer(self.ub_bytes, "ub_bytes")
        integer(self.system_reserved_bytes, "system_reserved_bytes", 8192)
        if self.payload_offset_bytes != 2048:
            raise ValueError("payload_offset_bytes is fixed at 2048")
        if self.payload_budget_bytes <= 0:
            raise ValueError("UB budget has no payload capacity")

    @property
    def payload_budget_bytes(self):
        # CANN may report physical 256 KiB or the usable 248 KiB. Never
        # subtract the architecture's 8 KiB reservation twice.
        return min(self.ub_bytes, 256 * 1024 - self.system_reserved_bytes) - self.payload_offset_bytes

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CommConfig:
    commAivCount: int
    workersPerPeer: int
    commChunkBytes: int
    commUbStages: int

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != set(FIELDS):
            raise ValueError(f"configuration must contain exactly {FIELDS}; legacy splits are unsupported")
        return cls(**value)

    def to_dict(self):
        return asdict(self)

    def as_tuple(self):
        return tuple(getattr(self, name) for name in FIELDS)

    def validate(self, case, platform):
        integer(self.commAivCount, "commAivCount", 0)
        integer(self.workersPerPeer, "workersPerPeer")
        if self.workersPerPeer > 64:
            raise ValueError("workersPerPeer must not exceed 64")
        integer(self.commChunkBytes, "commChunkBytes", 32)
        integer(self.commUbStages, "commUbStages")
        if self.commUbStages not in (1, 2):
            raise ValueError("commUbStages must be 1 or 2")
        if self.commChunkBytes % 32:
            raise ValueError("commChunkBytes must be 32-byte aligned")
        if self.commChunkBytes * self.commUbStages > platform.payload_budget_bytes:
            raise ValueError("payload stages exceed conservative UB budget")
        world = integer(case["world_size"], "world_size")
        if world > 8:
            raise ValueError("world_size exceeds the frozen kernel limit of 8")
        budget = remote_worker_budget(case, platform)
        if world == 1:
            if self.commAivCount != 0 or self.workersPerPeer != 1:
                raise ValueError("EP1 requires commAivCount=0 and workersPerPeer=1")
        else:
            if not 1 <= self.commAivCount <= budget:
                raise ValueError(f"commAivCount must be in [1, {budget}]")
            if self.commAivCount % self.workersPerPeer:
                raise ValueError("commAivCount must be divisible by workersPerPeer")
            if self.commAivCount // self.workersPerPeer > world - 1:
                raise ValueError("peer groups exceed remote peers and would leave whole groups idle")
        return self


def frozen_geometry(case, platform=None):
    """The delivery profiles are metadata, never tuning dimensions."""
    op = case["operator"]
    if op not in OPERATORS:
        raise ValueError(f"unsupported operator: {op}")
    if op == OPERATORS[1]:
        device = platform or Platform()
        n, k, world = case["shape"]["n"], case["shape"]["k"], case["world_size"]
        business = world == 8 and case["expert_num"] == 256
        m0, n0, k0, interval, tile_m, block_m = 128, 256, 256, 10, 64, 64
        if business and k == 1024 and n == 2048:
            interval = 14
        elif business and k == 512 and n == 2048:
            m0, n0, interval, tile_m, block_m = 256, 128, 12, 16, 16
        n_loops = (n + n0 - 1) // n0
        tiles_per_rank = device.block_dim * interval // world
        if tiles_per_rank < n_loops:
            raise ValueError("frozen GMM staging requires block_dim * commInterval / rankSize >= ceil(N / N0)")
        stage_rows = (tiles_per_rank // n_loops) * m0
        if 2 * stage_rows * world * n * 2 > 400 * 1024 * 1024:
            raise ValueError("frozen GMM workspace exceeds the 400 MiB data region")
        aligned_n = (n + 15) // 16 * 16
        local_rows = min(tile_m // 2, (256 * 1024 - 256) // (2 * aligned_n * 2))
        if local_rows < 1:
            raise ValueError("N is too large for frozen GMM local copy geometry")
        return {"m0": m0, "n0": n0, "k0": k0, "commInterval": interval,
                "commTileM": tile_m, "commBlockM": block_m, "workspaceStages": 2,
                "stageRows": stage_rows, "localCopyTileRows": local_rows,
                "localCopyTileColumns": aligned_n, "selfCopyCores": 1}
    shape = case["shape"]
    business = (case["world_size"] == 8 and case["expert_num"] == 256
                and shape["m"] == 131072 and shape["k"] == 2048)
    export = case["output_all_to_allv"]
    # m0,n0,k0,commInterval,commBlockM,stageRows,workspaceStages,
    # localCopyTileRows,mSplit,exportCoreCount,selfCopyCores
    row = (128, 256, 256, 10, 64, 1280, 2, 32, 1, 0, 1)
    if business and shape["n"] == 1024:
        row = ((256, 256, 256, 8, 32, 2048, 2, 16, 1, 16, 8) if export
               else (128, 256, 256, 12, 128, 1536, 2, 64, 1, 0, 1))
    elif business and shape["n"] == 512:
        row = ((256, 256, 256, 12, 32, 3072, 3, 16, 2, 32, 4) if export
               else (256, 128, 256, 6, 128, 1536, 2, 64, 1, 0, 1))
    names = ("m0", "n0", "k0", "commInterval", "commBlockM", "stageRows",
             "workspaceStages", "localCopyTileRows", "mSplit", "exportCoreCount", "selfCopyCores")
    return dict(zip(names, row))


def remote_worker_budget(case, platform):
    geometry = frozen_geometry(case, platform)
    if geometry.get("exportCoreCount", 0) > platform.block_dim:
        raise ValueError("frozen exportCoreCount exceeds platform block_dim")
    if case["world_size"] + geometry["selfCopyCores"] - 1 > platform.block_dim:
        raise ValueError("rankSize + selfCopyCores - 1 exceeds platform block_dim")
    return platform.block_dim - geometry["selfCopyCores"]


def enumerate_candidates(case, platform, space=None):
    is_default = space is None
    space = dict(DEFAULT_SPACE if is_default else space)
    if case["world_size"] == 1 and is_default:
        space["commAivCount"] = [0]
        space["workersPerPeer"] = [1]
    if set(space) != set(FIELDS):
        raise ValueError(f"search space must contain exactly {FIELDS}")
    for name, values in space.items():
        if not isinstance(values, (tuple, list)) or not values:
            raise ValueError(f"{name} candidates must be a nonempty list")
        for value in values:
            integer(value, name, 0 if name == "commAivCount" else 1)
    accepted, rejected, seen = [], [], set()
    for values in product(*(space[name] for name in FIELDS)):
        if values in seen:
            continue
        seen.add(values)
        candidate = CommConfig(*values)
        try:
            candidate.validate(case, platform)
        except ValueError as error:
            rejected.append({"config": candidate.to_dict(), "reason": str(error)})
        else:
            accepted.append(candidate)
    return accepted, rejected


def worker_tasks(config, remote_peers, chunks_per_peer):
    """CPU oracle of peer-group/lane assignment used by both MTE kernels."""
    if config.commAivCount == 0:
        return {}
    groups = config.commAivCount // config.workersPerPeer
    result = {}
    for worker in range(config.commAivCount):
        group, lane = divmod(worker, config.workersPerPeer)
        result[worker] = [
            (remote_peers[ordinal], chunk)
            for ordinal in range(group, len(remote_peers), groups)
            for chunk in range(lane, chunks_per_peer[remote_peers[ordinal]], config.workersPerPeer)
        ]
    return result
