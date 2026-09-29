# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate serialized bytes and task addresses independently before launch."""

from pathlib import Path
import re
import struct
import numpy as np

from case import CHIP_STRIDE, WINDOW, GUARD
from common import ROOT, read, sha, validate_config, write


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_source_abi():
    source = (ROOT / "mc2/mega_moe/op_kernel/arch35/mega_moe_apt.cpp").read_text()
    match = re.search(r"void mega_moe\((.*?)\)\s*\{", source, re.S)
    require(match is not None, "Kernel entry signature changed")
    names = re.findall(r"GM_ADDR\s+(\w+)", match.group(1))
    expected = [
        "context",
        "x",
        "topkIds",
        "topkWeights",
        "weight1",
        "weight2",
        "weightScales1",
        "weightScales2",
        "bias1",
        "bias2",
        "xActiveMask",
        "scales",
        "sharedWeight1",
        "sharedWeight2",
        "sharedWeightScales1",
        "sharedWeightScales2",
        "sharedBias1",
        "sharedBias2",
        "maskBuffer",
        "yOut",
        "expertTokenNumsOut",
        "workspaceGM",
        "tilingGM",
    ]
    require(names == expected, f"Kernel ABI changed: {names}; update case slot mapping")
    require(
        "KERNEL_TYPE_MIX_AIC_1_2" in source, "Kernel no longer declares MIX_AIC_1_2"
    )


def validate_case(directory, hashes=True, inputs_only=False):
    directory = Path(directory).resolve()
    cfg, info = read(directory / "config.json"), read(directory / "layout.json")
    validate_config(cfg)
    m = cfg["model"]
    ep, b, h, n, local = (
        m[k]
        for k in (
            "ep",
            "tokens_per_rank",
            "hidden",
            "intermediate",
            "moe_experts_per_rank",
        )
    )
    top, maps = (
        read(directory / ("top.pending.json" if inputs_only else "top.json")),
        read(directory / "memory_map.json"),
    )
    require(
        set(top) == {f"chip{r}" for r in range(ep)}, "Missing or extra chip in top.json"
    )
    for name, metadata in read(directory / "files.json").items():
        path = directory / name
        require(
            path.is_file() and path.stat().st_size == metadata["size"],
            f"File size mismatch: {path}",
        )
        if hashes:
            require(
                sha(path) == metadata["sha256"],
                f"Input/bin was modified: {path}; regenerate case",
            )
    total_counts = np.zeros(ep * local, np.int64)
    for rank in range(ep):
        d = directory / f"chip_{rank}"
        base = rank * CHIP_STRIDE
        require(
            Path(top[f"chip{rank}"]["die0"]["case_path"]).resolve() == d,
            "Case moved: run ./mega-sim relocate",
        )
        task = read(d / "mega_task.json")
        for key, value in {
            "sqe_type": "AIC",
            "mix": 1,
            "ratio": 2,
            "block_dim": m["aic_per_rank"],
            "group_dim": 1,
            "group_slice": m["aic_per_rank"],
            "schem": 1,
        }.items():
            require(
                task.get(key) == value,
                f"rank {rank}: {key} must be {value} for this arch35 MIX profile",
            )
        require(
            task["hbm_para_addr"] == task["sub_hbm_para_addr"],
            "AIC/AIV parameter tables differ",
        )
        require(
            int(task["hbm_para_addr"], 0) == base + 0x8100000000,
            "Parameter address mismatch",
        )
        require(
            int(task["stack_phy_base"], 0) == base + 0x8200000000,
            "Stack address mismatch",
        )
        require(
            read(d / "mega_sq.json")["tasks"] == [{"sqe_type": "AIC", "name": "mega"}],
            "SQ must dispatch AIC MIX task",
        )
        sq = read(d / "sq_top.json")
        for key, offset in (
            ("sq_base_addr", 0x9010000000),
            ("cq_base_addr", 0x9018000000),
            ("sq_ns_swap_buffer_addr", 0x9028000000),
        ):
            require(int(sq[key], 0) == base + offset, f"Bad {key}")
        regions = sorted(maps[str(rank)]["regions"], key=lambda r: r["addr"])
        for index, region in enumerate(regions):
            require(
                base
                <= region["addr"]
                < region["addr"] + region["size"]
                <= base + CHIP_STRIDE,
                "Region crosses rank",
            )
            if index:
                require(
                    regions[index - 1]["addr"] + regions[index - 1]["size"]
                    <= region["addr"],
                    "Memory regions overlap",
                )
        entries = task["input_para_array"] + task["output_para_array"]
        require(
            len(entries) == 24
            and sorted(e["para_offset"] for e in entries) == list(range(24)),
            "All ABI slots plus window initializer must appear exactly once",
        )
        slots = {e["para_offset"]: e for e in entries}
        dump_enabled = cfg["simulation"].get("dump_outputs", False)
        dumps = {e["name"]: e for e in task.get("ast_output_array", [])}
        require(
            bool(dumps) == dump_enabled,
            "Task dump mode differs from simulation.dump_outputs",
        )
        for slot in (8, 9, 10, 16, 17, 18):
            require(
                int(slots[slot]["size"], 0) == int(slots[slot]["addr"], 0) == 0,
                f"Optional slot {slot} must be null",
            )
        for name, tensor in maps[str(rank)]["tensors"].items():
            entry = slots[tensor["init_slot"]]
            addr, size = int(entry["addr"], 0), int(entry["size"], 0)
            require(
                entry["name"] == name + ".bin"
                and addr == tensor["addr"]
                and size == tensor["file_bytes"],
                f"Bad entry {name}",
            )
            require(
                any(
                    r["addr"] <= addr and addr + size <= r["addr"] + r["size"]
                    for r in regions
                ),
                f"Unreserved allocation {name}",
            )
            with (d / entry["name"]).open("rb") as f:
                if tensor["tensor_list"]:
                    offset, pointer = struct.unpack("<QQ", f.read(16))
                    require(
                        offset == 8 and pointer == addr + 512,
                        f"Invalid TensorList pointer {name}",
                    )
                    require(
                        pointer + tensor["data_bytes"] <= addr + tensor["guard_offset"],
                        "TensorList points outside payload",
                    )
                f.seek(tensor["guard_offset"])
                require(f.read() == GUARD, f"Initial guard corrupted: {name}")
            if dump_enabled:
                guard = dumps[tensor["guard"]]
                require(
                    int(guard["addr"], 0) == addr + tensor["guard_offset"]
                    and int(guard["size"], 0) == len(GUARD),
                    "Guard dump mismatch",
                )
        tensors = maps[str(rank)]["tensors"]
        require(
            task["output_para_array"] == [],
            "A5 debug profile uses input_para_array for every argument",
        )
        for slot, name in ((19, "y_init"), (20, "counts_init")):
            require(
                int(slots[slot]["addr"], 0) == tensors[name]["addr"],
                "Output pointer differs from initializer",
            )
        expected_sizes = {
            "x": b * h,
            "topk_ids": b * m["top_k"] * 4,
            "topk_weights": b
            * m["top_k"]
            * (4 if cfg.get("topk_weights_dtype") == "float32" else 2),
            "x_scales": b * h // 32,
            "workspace": info["workspace_bytes"],
            "communication_window": info["window_bytes"],
            "context": info["context_bytes"],
            "tiling": info["tiling_bytes"],
            "y_init": b * h * 2,
            "counts_init": local * 4,
        }
        for prefix, experts, divisor in (
            ("moe", local, 2),
            ("shared", m["shared_experts_per_rank"], 1),
        ):
            expected_sizes.update(
                {
                    f"{prefix}_w1": experts * 2 * n * h // divisor,
                    f"{prefix}_w2": experts * h * n // divisor,
                    f"{prefix}_s1": experts * 2 * n * h // 32,
                    f"{prefix}_s2": experts * h * n // 32,
                }
            )
        for name, size in expected_sizes.items():
            require(
                tensors[name]["data_bytes"] == size,
                f"Shape/dtype byte size mismatch: {name}",
            )
        context = (d / "context.bin").read_bytes()
        require(
            struct.unpack_from("<IIQ", context) == (rank, ep, 0),
            "Bad Mc2MoeContext header",
        )
        windows = struct.unpack_from(f"<{ep}Q", context, info["context_windows_offset"])
        require(
            windows == tuple(r * CHIP_STRIDE + WINDOW for r in range(ep)),
            "Peer windows are not real cross-rank addresses",
        )
        require(
            tensors["communication_window"]["addr"] == windows[rank],
            "Local window initialization address differs from context",
        )
        ids = np.fromfile(
            d / "topk_ids.bin", dtype="<i4", count=b * m["top_k"]
        ).reshape(b, m["top_k"])
        require(
            np.all((ids >= 0) & (ids < ep * local)),
            "Shared/out-of-range expert in topkIds",
        )
        require(
            all(len(set(row)) == len(row) for row in ids.tolist()),
            "Duplicate expert in one token's routing",
        )
        total_counts += np.bincount(ids.ravel(), minlength=ep * local)
        x = np.fromfile(d / "x.bin", dtype=np.uint8, count=b * h)
        require(np.all((x & 0x7F) != 0x7F), "FP8 input contains NaN")
        if cfg.get("topk_weights_dtype") == "float32":
            gates = np.fromfile(
                d / "topk_weights.bin", dtype="<f4", count=b * m["top_k"]
            )
        else:
            gates = (
                np.fromfile(
                    d / "topk_weights.bin", dtype="<u2", count=b * m["top_k"]
                ).astype("<u4")
                << 16
            ).view("<f4")
        require(
            np.all(np.isfinite(gates)) and np.all(gates >= 0), "Invalid topk weights"
        )
        require(
            np.allclose(gates.reshape(b, m["top_k"]).sum(axis=1), 1, atol=0.01),
            "Topk weights do not sum approximately to one",
        )
        for name in ("shared_w1", "shared_w2"):
            data = np.fromfile(
                d / (name + ".bin"),
                dtype=np.uint8,
                count=tensors[name]["data_bytes"],
                offset=512,
            )
            require(
                np.all((data & 0x7F) != 0x7F), f"FP8 shared weight contains NaN: {name}"
            )
        for name in ("x_scales", "moe_s1", "moe_s2", "shared_s1", "shared_s2"):
            t = tensors[name]
            data = np.fromfile(
                d / (name + ".bin"),
                dtype=np.uint8,
                count=t["data_bytes"],
                offset=512 if t["tensor_list"] else 0,
            )
            require(np.all(data < 255), f"E8M0 NaN in {name}")
        require(
            (d / "tiling.bin").read_bytes()[: info["tiling_bytes"]]
            == (directory / "tiling.bin").read_bytes(),
            "Tiling differs by rank",
        )
        for kind, name_key, addr_key, offset in (
            ("aic", "name", "addr", 0x40000000),
            ("aiv", "sub_name", "sub_addr", 0x41000000),
        ):
            require(task["BIN"][name_key] == kind + ".bin", f"Missing {kind} image")
            require(
                int(task["BIN"][addr_key], 0) == base + offset,
                f"Bad {kind} load address",
            )
            if not inputs_only:
                require(
                    64 < (d / (kind + ".bin")).stat().st_size <= 0x1000000,
                    "Kernel exceeds reserved region",
                )
        for name, addr, size in (
            ("actual_y.bin", tensors["y_init"]["addr"], b * h * 2),
            ("actual_counts.bin", tensors["counts_init"]["addr"], local * 4),
        ):
            if dump_enabled:
                require(
                    int(dumps[name]["addr"], 0) == addr
                    and int(dumps[name]["size"], 0) == size,
                    "Output dump mismatch",
                )
    expected = ep * b * m["top_k"] // (ep * local)
    require(
        np.all(total_counts == expected), f"Unbalanced routing: {total_counts.tolist()}"
    )
    require(
        total_counts.tolist() == read(directory / "routing.json")["tokens_per_expert"],
        "Routing manifest mismatch",
    )
    result = {
        "ranks": ep,
        "tokens_per_expert": expected,
        "aic_per_rank": m["aic_per_rank"],
        "aiv_per_rank": m["aiv_per_rank"],
        "tiling_key": m["tiling_key"],
        "address_check": "passed",
        "kernel_attached": not inputs_only,
    }
    write(directory / "validation.json", result)
    return result


def check_outputs(directory):
    directory = Path(directory).resolve()
    cfg, maps, routing = (
        read(directory / "config.json"),
        read(directory / "memory_map.json"),
        read(directory / "routing.json"),
    )
    require(
        cfg["simulation"].get("dump_outputs", False),
        "Output dumps are disabled in the A5 debug profile; enable dump_outputs only if your simulator supports ast_output_array",
    )
    results = {}
    for rank in range(cfg["model"]["ep"]):
        d = directory / f"chip_{rank}"
        for name, tensor in maps[str(rank)]["tensors"].items():
            path = d / tensor["guard"]
            require(
                path.exists(),
                f"Missing guard dump: {path}; simulation/dump did not complete",
            )
            require(
                path.read_bytes() == GUARD,
                f"Buffer boundary overwritten: rank {rank} {name}",
            )
        counts_path = d / "actual_counts.bin"
        require(
            counts_path.stat().st_size == cfg["model"]["moe_experts_per_rank"] * 4,
            "Bad count dump size",
        )
        counts = np.fromfile(counts_path, dtype="<i4")
        require(
            counts.tolist() == routing["expected_local_counts"][rank],
            f"rank {rank}: unexpected expert counts {counts}",
        )
        y_path = d / "actual_y.bin"
        require(
            y_path.stat().st_size
            == cfg["model"]["tokens_per_rank"] * cfg["model"]["hidden"] * 2,
            "Bad y dump size",
        )
        y = (np.fromfile(y_path, dtype="<u2").astype("<u4") << 16).view("<f4")
        require(
            np.all(np.isfinite(y)),
            f"rank {rank}: output has NaN/Inf or unwritten poison",
        )
        require(
            np.any(y != 0),
            f"rank {rank}: output is entirely zero for a nonzero fixture",
        )
        results[str(rank)] = {
            "expert_counts": counts.tolist(),
            "guards": "passed",
            "finite_output": True,
        }
    result = {
        "structural_output_checks": "passed",
        "numerical_accuracy": "not_checked",
        "ranks": results,
    }
    write(directory / "output_validation.json", result)
    return result
