# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Create a real multi-chip MTE case for the debug simulator's -c interface."""

from pathlib import Path
import struct
import subprocess

import numpy as np

from build import check_build
from common import ROOT, read, sha, source_hash, validate_config, write
from elf import images
from tensors import expert_tensors, inputs, routing

CHIP_STRIDE = 0x10000000000  # 1 TiB address stride, not a physical HBM capacity.
WINDOW = 0x8000000000
ARENA_START, ARENA_END = 0x9100000000, 0xA000000000
GUARD = bytes([0xD3]) * 512
SLOTS = [
    "context",
    "x",
    "topk_ids",
    "topk_weights",
    "moe_w1",
    "moe_w2",
    "moe_s1",
    "moe_s2",
    "bias1",
    "bias2",
    "x_active_mask",
    "x_scales",
    "shared_w1",
    "shared_w2",
    "shared_s1",
    "shared_s2",
    "shared_bias1",
    "shared_bias2",
    "mask_buffer",
    "y",
    "expert_counts",
    "workspace",
    "tiling",
]


def align(value, alignment=65536):
    return (value + alignment - 1) // alignment * alignment


class AddressMap:
    def __init__(self, rank):
        self.base = rank * CHIP_STRIDE
        self.cursor = self.base + ARENA_START
        self.regions = []
        for name, offset, size in (
            ("aic_code", 0x40000000, 0x1000000),
            ("aiv_code", 0x41000000, 0x1000000),
            ("hbm_parameters", 0x8100000000, 0x1000000),
            ("stack", 0x8200000000, 0x100000000),
            ("sq", 0x9010000000, 0x8000000),
            ("cq", 0x9018000000, 0x10000000),
            ("sq_swap", 0x9028000000, 0x10000000),
        ):
            self.reserve(name, self.base + offset, size)

    def reserve(self, name, addr, size):
        if size <= 0 or addr < self.base or addr + size > self.base + CHIP_STRIDE:
            raise ValueError(f"Invalid rank-local region {name}")
        for old in self.regions:
            if addr < old["addr"] + old["size"] and old["addr"] < addr + size:
                raise ValueError(f"Address overlap: {name} and {old['name']}")
        self.regions.append({"name": name, "addr": addr, "size": size})
        return addr

    def allocate(self, name, size):
        addr = self.cursor
        extent = align(size)
        if addr + extent > self.base + ARENA_END:
            raise ValueError("Tensor/workspace arena exhausted")
        self.reserve(name, addr, extent)
        self.cursor += extent
        return addr


def layout(cfg, build_dir, dest):
    validate_config(cfg)
    m = cfg["model"]
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    command = [str(Path(build_dir) / "tiling_helper"), str(dest / "tiling.bin")]
    command += [
        str(m[k])
        for k in (
            "tokens_per_rank",
            "hidden",
            "intermediate",
            "ep",
            "moe_experts_per_rank",
            "shared_experts_per_rank",
            "top_k",
            "aic_per_rank",
            "max_tokens_per_rank",
            "max_recv_tokens",
            "ub_bytes",
        )
    ]
    command.append(cfg.get("topk_weights_dtype", "bfloat16"))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Layout helper failed: {result.stdout}\n{result.stderr}")
    import json

    info = json.loads(
        next(
            line
            for line in result.stdout.splitlines()
            if line.startswith('{"tiling_key"')
        )
    )
    if info["tiling_key"] != m["tiling_key"]:
        raise ValueError("Host key mismatch")
    write(dest / "layout.json", info)
    return info


def generate(cfg, build_dir, dest):
    validate_config(cfg)
    build_dir, dest = Path(build_dir).resolve(), Path(dest).resolve()
    manifest = check_build(cfg, build_dir)
    if dest.exists():
        raise ValueError(
            f"Case directory already exists: {dest}; choose a fresh --case directory"
        )
    bins, symbols = images(build_dir / manifest["elf"], cfg["model"]["tiling_key"])
    dest.mkdir(parents=True)
    info = layout(cfg, build_dir, dest)
    write_case(cfg, dest, info, bins)
    write(dest / "build.json", {**manifest, "elf_symbols": symbols})
    # Keep the linked ELF for disassembly and resolving simulator PCs.
    import shutil

    shutil.copy2(build_dir / manifest["elf"], dest / "kernel.o")
    from validate import validate_case

    validate_case(dest)
    print(f"Case ready: {dest}\nnpusim -c {dest}")


def generate_inputs(cfg, build_dir, dest):
    validate_config(cfg)
    build_dir, dest = Path(build_dir).resolve(), Path(dest).resolve()
    provenance = read(build_dir / "layout_build.json")
    if provenance["source_sha256"] != source_hash() or provenance[
        "helper_sha256"
    ] != sha(build_dir / "tiling_helper"):
        raise ValueError("Host helper is stale; run ./mega-sim build --host-only")
    if dest.exists():
        raise ValueError(f"Case directory already exists: {dest}")
    dest.mkdir(parents=True)
    info = layout(cfg, build_dir, dest)
    write_case(cfg, dest, info, None)
    write(dest / "input_build.json", provenance)
    from validate import validate_case

    validate_case(dest, inputs_only=True)
    print(
        f"Inputs prepared and checked: {dest}\nBuild on target, then ./mega-sim attach --case {dest}"
    )


def attach(build_dir, dest):
    import shutil
    import tempfile
    from validate import validate_case

    dest, build_dir = Path(dest).resolve(), Path(build_dir).resolve()
    if (dest / "top.json").exists():
        raise ValueError(
            "This case already has a kernel attached; use a fresh input case"
        )
    cfg = read(dest / "config.json")
    manifest = check_build(cfg, build_dir)
    # Recalculate with the actual server's headers/compiler before using prepared bytes.
    with tempfile.TemporaryDirectory(prefix="mega-layout-") as tmp:
        info = layout(cfg, build_dir, Path(tmp))
        if (
            info != read(dest / "layout.json")
            or (Path(tmp) / "tiling.bin").read_bytes()
            != (dest / "tiling.bin").read_bytes()
        ):
            raise ValueError(
                "Server tiling/layout differs from prepared inputs; regenerate with ./mega-sim case"
            )
    top = read(dest / "top.pending.json")
    for rank in range(cfg["model"]["ep"]):
        top[f"chip{rank}"]["die0"]["case_path"] = str(dest / f"chip_{rank}") + "/"
    write(dest / "top.pending.json", top)
    validate_case(dest, inputs_only=True)
    bins, symbols = images(build_dir / manifest["elf"], cfg["model"]["tiling_key"])
    files = read(dest / "files.json")
    for rank in range(cfg["model"]["ep"]):
        for kind, data in bins.items():
            if not 64 < len(data) <= 0x1000000:
                raise ValueError("Kernel does not fit code reservation")
            path = dest / f"chip_{rank}" / f"{kind}.bin"
            path.write_bytes(data)
            files[str(path.relative_to(dest))] = {
                "size": len(data),
                "sha256": sha(path),
            }
    write(dest / "files.json", files)
    write(dest / "build.json", {**manifest, "elf_symbols": symbols})
    shutil.copy2(build_dir / manifest["elf"], dest / "kernel.o")
    write(dest / "top.json", top)
    try:
        validate_case(dest)
    except BaseException:
        (dest / "top.json").unlink(missing_ok=True)
        raise
    print(f"Case ready: npusim -c {dest}")


def write_case(cfg, dest, info, bins):
    """Internal pure generator, also used by CPU contract tests with labelled dummy code."""
    m = cfg["model"]
    ep, b, h = m["ep"], m["tokens_per_rank"], m["hidden"]
    if bins is not None:
        for kind in ("aic", "aiv"):
            if not 64 < len(bins[kind]) <= 0x1000000:
                raise ValueError(f"Unexpected {kind} image size")
    top, maps, files, counts = {}, {}, {}, []
    for rank in range(ep):
        directory = dest / f"chip_{rank}"
        directory.mkdir()
        memory = AddressMap(rank)
        base = memory.base
        task = {
            "sqe_type": "AIC",
            "mix": 1,
            "ratio": 2,
            "block_dim": m["aic_per_rank"],
            "group_dim": 1,
            "group_slice": m["aic_per_rank"],
            "latency": 1,
            "schem": 1,
            "task_type": 6,
            "title": "mega",
            "hbm_para_addr": hex(base + 0x8100000000),
            "sub_hbm_para_addr": hex(base + 0x8100000000),
            "stack_phy_base": hex(base + 0x8200000000),
            "aic_icache_prefetch_cnt": 0,
            "aiv_icache_prefetch_cnt": 0,
            "aic_dcache_prefetch_cnt": 64,
            "aiv_dcache_prefetch_cnt": 64,
            "input_para_array": [],
            "output_para_array": [],
            "ast_output_array": [],
        }
        tensors = {}

        def emit(slot, name, data, tensor_list=False, addr=None, shape=None):
            data = memoryview(data).cast("B")
            payload_size = len(data) + (512 if tensor_list else 0)
            storage_size = align(payload_size, 512)
            total = storage_size + len(GUARD)
            if addr is None:
                addr = memory.allocate(name, total)
            else:
                memory.reserve(name, addr, align(total))
            path = directory / (name + ".bin")
            with path.open("wb") as f:
                if tensor_list:
                    # Kernel GetTensorAddr(): first uint64 is pointer-table byte offset.
                    f.write(struct.pack("<QQ", 8, addr + 512) + bytes(496))
                f.write(data)
                f.write(bytes(storage_size - payload_size))
                f.write(GUARD)
            init_slot = slot
            task["input_para_array"].append(
                {
                    "name": path.name,
                    "size": hex(total),
                    "addr": hex(addr),
                    "para_offset": init_slot,
                    "hbm_para_offset_en": 1,
                    "para_type": 0,
                }
            )
            guard_name = f"guard_{name}.bin"
            task["ast_output_array"].append(
                {
                    "name": guard_name,
                    "addr": hex(addr + storage_size),
                    "size": hex(len(GUARD)),
                }
            )
            tensors[name] = {
                "slot": slot,
                "init_slot": init_slot,
                "addr": addr,
                "payload_bytes": payload_size,
                "data_bytes": len(data),
                "file_bytes": total,
                "tensor_list": tensor_list,
                "shape": shape,
                "guard": guard_name,
                "guard_offset": storage_size,
            }
            files[str(path.relative_to(dest))] = {"size": total, "sha256": sha(path)}
            return addr

        context = bytearray(info["context_bytes"])
        if info["context_windows_offset"] != 16 or len(context) < 16 + 8 * ep:
            raise ValueError("Mc2MoeContext ABI changed; update context serializer")
        struct.pack_into("<IIQ", context, 0, rank, ep, 0)
        for peer in range(ep):
            struct.pack_into("<Q", context, 16 + 8 * peer, peer * CHIP_STRIDE + WINDOW)
        emit(0, "context", context)
        for slot, (name, arr) in inputs(cfg, rank).items():
            emit(slot, name, arr, shape=list(arr.shape))
        for shared in (False, True):
            for slot, name, arr in expert_tensors(cfg, rank, shared):
                emit(slot, name, arr, tensor_list=True, shape=list(arr.shape))
        # Every optional ABI argument is explicitly null, never uninitialized parameter RAM.
        for slot in (8, 9, 10, 16, 17, 18):
            task["input_para_array"].append(
                {
                    "size": "0x0",
                    "addr": "0x0",
                    "para_offset": slot,
                    "hbm_para_offset_en": 1,
                    "para_type": 0,
                }
            )
        # A5 debug JSON loads all 23 ABI slots via input_para_array, including output
        # storage. There is no separate output allocation/AST task in its default schema.
        y_addr = emit(
            19, "y_init", b"\xc1\x7f" * (b * h), shape=[b, h]
        )  # BF16 NaN poison
        count_addr = emit(20, "counts_init", b"\xff" * (m["moe_experts_per_rank"] * 4))
        task["ast_output_array"] += [
            {"name": "actual_y.bin", "addr": hex(y_addr), "size": hex(b * h * 2)},
            {
                "name": "actual_counts.bin",
                "addr": hex(count_addr),
                "size": hex(m["moe_experts_per_rank"] * 4),
            },
        ]
        emit(21, "workspace", bytes(info["workspace_bytes"]))
        emit(22, "tiling", (dest / "tiling.bin").read_bytes())
        # Extra simulator-only pointer after the 23 kernel arguments initializes peer memory.
        # No kernel code reads argument 23. This avoids relying on implicit zero-filled HBM.
        emit(
            23, "communication_window", bytes(info["window_bytes"]), addr=base + WINDOW
        )
        task["input_para_array"].sort(key=lambda p: p["para_offset"])
        if not cfg["simulation"].get("dump_outputs", False):
            task.pop("ast_output_array")
        task["BIN"] = {
            "name": "aic.bin",
            "addr": hex(base + 0x40000000),
            "sub_name": "aiv.bin",
            "sub_addr": hex(base + 0x41000000),
        }
        for kind, data in (bins or {}).items():
            path = directory / f"{kind}.bin"
            path.write_bytes(data)
            files[str(path.relative_to(dest))] = {
                "size": len(data),
                "sha256": sha(path),
            }
        write(directory / "mega_task.json", task)
        write(
            directory / "mega_sq.json",
            {"name": "mega_sq", "tasks": [{"sqe_type": "AIC", "name": "mega"}]},
        )
        write(
            directory / "sq_top.json",
            {
                "name": "sq_top",
                "sq_base_addr": hex(base + 0x9010000000),
                "cq_base_addr": hex(base + 0x9018000000),
                "sq_ns_swap_buffer_addr": hex(base + 0x9028000000),
                "sq": [
                    {
                        "name": "mega",
                        "id": 0,
                        "stack_phy_base": hex(base + 0x8200000000),
                    }
                ],
            },
        )
        top[f"chip{rank}"] = {
            "die0": {
                "case_path": str(directory.resolve()) + "/",
                "sq_file": "sq_top.json",
            }
        }
        maps[str(rank)] = {"regions": memory.regions, "tensors": tensors}
        counts.append(
            np.bincount(
                routing(cfg, rank).ravel(), minlength=ep * m["moe_experts_per_rank"]
            ).tolist()
        )
    write(dest / "config.json", cfg)
    write(dest / "memory_map.json", maps)
    write(dest / "files.json", files)
    total = np.sum(counts, axis=0).tolist()
    write(
        dest / "routing.json",
        {
            "source_to_expert": counts,
            "tokens_per_expert": total,
            "expected_local_counts": [
                total[
                    r * m["moe_experts_per_rank"] : (r + 1) * m["moe_experts_per_rank"]
                ]
                for r in range(ep)
            ],
        },
    )
    # top.json is written last: incomplete generation cannot look like a ready case.
    write(dest / ("top.json" if bins is not None else "top.pending.json"), top)
