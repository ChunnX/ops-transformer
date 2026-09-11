#  Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
#

"""Topology sweep coordination and portable reports; no Torch/NPU imports."""

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


def peer_counts(world_size):
    """Powers of two followed by every remote rank, without duplicates."""
    if world_size < 2:
        raise ValueError("Topology sweeps require world_size > 1")
    counts = []
    count = 1
    while count < world_size - 1:
        counts.append(count)
        count *= 2
    return counts + [world_size - 1]


def sweep_cases(world_size):
    cases = [("jetty_scan", 3, jettys, world_size - 1) for jettys in (1, 2, 4, 8, 16, 32)]
    cases += [("peer_scan", 3, 15, peers) for peers in peer_counts(world_size)]
    cases.append(("mode4_baseline", 4, world_size - 1, 1))
    return cases


def save_case_results(config, results, server_index, world_size, aiv_num):
    case = json.loads(os.environ["BANDWIDTH_TEST_CASE"])
    rows = []
    for result in results:
        batch = result["shared_jetty_batch_tokens"]
        row = {
            "experiment": case["experiment"], "mode": config.model.mode,
            "world_size": world_size, "server_index": server_index,
            "sampled_ranks": ",".join(map(str, result["sampled_ranks"])),
            "comm_alg": config.model.comm_alg, "comm_level": config.model.shared_jetty_comm_level,
            "graph_type": config.model.graph_type, "aiv_num": aiv_num,
            "jettys": case["jettys"], "peers_per_jetty": case["peers"],
            "peer_selection": "cyclic_spaced" if config.model.mode == 3 else "one_jetty_per_peer",
            "config_index": result["config_index"], "bs": result["bs"], "h": result["data_size"],
            "dtype": config.shape.x_dtype,
            "requested_batch_tokens": config.model.batch_tokens, "batch_tokens": batch,
            "batch_bytes": batch * result["data_size"] * 2,
            "rank_data_bytes": round(result["single_rank_data_kb"] * 1024),
            "drain_threshold": config.model.drain_threshold, "subtracted_overhead_us": 2.0,
        }
        for key in ("mean_time_us", "min_time_us", "max_time_us", "mean_bandwidth_gbps", "max_bandwidth_gbps"):
            value = float(result[key])
            row[key] = value if math.isfinite(value) else None
        row["status"] = "ok" if row["min_time_us"] is not None and row["min_time_us"] > 0 else "invalid_corrected_time"
        rows.append(row)
    Path(case["result_path"]).write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")


def write_reports(output_dir, rows):
    """Rewrite accumulated reports after each completed topology; preserve earlier results on failure."""
    if not rows:
        return
    with (output_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = ["experiment", "mode", "jettys", "peers_per_jetty", "bs", "h", "rank_data_bytes",
               "batch_tokens", "batch_bytes", "mean_time_us", "min_time_us", "max_time_us",
               "mean_bandwidth_gbps", "max_bandwidth_gbps", "status"]
    first = rows[0]
    lines = ["# Bandwidth topology sweep", "",
             f"Server: {first['server_index']}; sampled ranks: {first['sampled_ranks']}; world size: {first['world_size']}.",
             f"URMA comm level: {first['comm_level']} (0=mixed, 1=all UBC_CTP, 2=all UBC_RTP); "
             f"graph type: {first['graph_type']}; AIVs: {first['aiv_num']}; dtype: {first['dtype']}.",
             f"Requested batch: {first['requested_batch_tokens']}; Drain threshold: {first['drain_threshold']}.",
             "", "All reported times subtract 2 us per operator. Bandwidth columns retain the script's GB/s label "
             "but use bytes / 2^30 / seconds (GiB/s). Traffic counts payload only; flags still consume queue budget.",
             "Statistics cover local ranks only, not a cross-server aggregate. Invalid corrected times produce N/A bandwidth.",
             "Mode 3 peer k for Jetty j: (rank + 1 + (j + floor(k*(W-1)/P)) % (W-1)) % W.", ""]
    for experiment in dict.fromkeys(row["experiment"] for row in rows):
        lines += [f"## {experiment}", "", "| " + " | ".join(columns) + " |",
                  "| " + " | ".join("---" for _ in columns) + " |"]
        for row in rows:
            if row["experiment"] != experiment:
                continue
            values = ["N/A" if row[key] is None else f"{row[key]:.6f}" if isinstance(row[key], float)
                      else str(row[key]) for key in columns]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    (output_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def run_sweep(config):
    world_size = config.parallel.server_num * config.parallel.rank_per_dev
    cases = sweep_cases(world_size)
    aiv_num = config.parallel.aiv_num
    if aiv_num is None:
        aiv_num = 64 if config.parallel.is_full_core else 2 * config.parallel.partial_core_num
    if any(jettys > min(aiv_num, 56) for _, _, jettys, _ in cases):
        raise ValueError("Sweep Jetty counts (including mode 4 baseline) must fit selected AIVs and the 56-Jetty limit")
    if not config.shape.test_configs:
        raise ValueError("Sweep requires shape.test_configs")
    if not 1 <= config.parallel.master_port <= 65535 - len(cases) + 1:
        raise ValueError("Sweep needs one consecutive master port per topology")
    server_index = int(os.environ["VC_TASK_INDEX"]) if config.parallel.is_cloud else config.parallel.server_index
    if not 0 <= server_index < config.parallel.server_num:
        raise ValueError("Invalid server index")
    output = Path(config.run.sweep_output_path).resolve() / f"server_{server_index}" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "world_size": world_size, "server_index": server_index,
                "planned_cases": len(cases), "shapes_per_case": len(config.shape.test_configs), "completed_cases": 0}
    manifest_path = output / "manifest.json"
    rows = []
    print(f"Sweep: {len(cases)} topologies x {len(config.shape.test_configs)} shapes; output: {output}", flush=True)
    try:
        for index, (experiment, mode, jettys, peers) in enumerate(cases):
            name = f"{index + 1:02d}_{experiment}_j{jettys}_p{peers}"
            case_dir = output / name
            case_dir.mkdir()
            case = {"experiment": experiment, "mode": mode, "jettys": jettys, "peers": peers,
                    "peers_attr": peers if mode == 3 else 0,
                    "master_port": config.parallel.master_port + index,
                    "profiling_path": str(case_dir / "Profiling_Result"),
                    "result_path": str(case_dir / "results.json")}
            manifest["current_case"] = case
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"[{index + 1}/{len(cases)}] {name}; master port {case['master_port']}", flush=True)
            env = os.environ.copy()
            env["BANDWIDTH_TEST_CASE"] = json.dumps(case)
            subprocess.run([sys.executable, str(Path(__file__).with_name("bandwidth_test.py"))], env=env, check=True)
            case_rows = json.loads(Path(case["result_path"]).read_text(encoding="utf-8"))
            if len(case_rows) != len(config.shape.test_configs):
                raise RuntimeError(f"Incomplete results for {name}")
            rows.extend(case_rows)
            write_reports(output, rows)
            manifest["completed_cases"] += 1
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Sweep {manifest['status']}; reports: {output}", flush=True)
