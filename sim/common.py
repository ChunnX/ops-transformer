# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Utilities for the standalone MegaMoE simulation workflow."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash():
    # Include untracked kernel headers too: changing code must invalidate old binaries.
    files = set()
    for name in ("mc2/mega_moe", "mc2/common", "mc2/3rd", "common", "cmake", "sim"):
        files.update(
            p
            for p in (ROOT / name).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in (".pyc",)
        )
    files.add(ROOT / "CMakeLists.txt")
    h = hashlib.sha256()
    for p in sorted(files):
        # Case values may change without recompiling the kernel or layout helper.
        if "configs" in p.parts:
            continue
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(bytes.fromhex(sha(p)))
    return h.hexdigest()


def run(cmd, cwd, log, env=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = list(map(str, cmd))
    print(f"[{log.stem}] {cmd[0]}  (log: {log})", flush=True)
    write(log.with_suffix(".command.json"), {"argv": cmd, "cwd": str(cwd)})
    with log.open("w") as f:
        result = subprocess.run(
            cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT
        )
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-35:])
        raise RuntimeError(f"Command exited {result.returncode}; {log}\n{tail}")


def validate_config(cfg):
    m = cfg["model"]
    if cfg.get("topk_weights_dtype", "bfloat16") not in ("bfloat16", "float32"):
        raise ValueError("topk_weights_dtype must be bfloat16 or float32")
    for key, val in m.items():
        if type(val) is not int or val < 0:
            raise ValueError(f"model.{key} must be a nonnegative integer")
    if m["ep"] not in (2, 4, 8):
        raise ValueError("This MTE case generator supports 2/4/8 ranks")
    if m["aiv_per_rank"] != 2 * m["aic_per_rank"] or not 1 <= m["aic_per_rank"] <= 32:
        raise ValueError("MIX_AIC_1_2 requires AIV=2*AIC, AIC in 1..32")
    if not 1 <= m["tokens_per_rank"] <= m["max_tokens_per_rank"]:
        raise ValueError("tokens_per_rank must be in 1..max_tokens_per_rank")
    if not (1024 <= m["hidden"] <= 8192 and m["hidden"] % 64 == 0):
        raise ValueError("A8W4 H must be 64-aligned and in 1024..8192")
    if not (256 <= m["intermediate"] <= 4096 and m["intermediate"] % 128 == 0):
        raise ValueError("SwiGLU N must be 128-aligned and in 256..4096")
    if (
        not 1 <= m["moe_experts_per_rank"] <= 1024
        or not 1 <= m["shared_experts_per_rank"] <= 4
    ):
        raise ValueError("Invalid expert counts")
    experts = m["ep"] * m["moe_experts_per_rank"]
    if experts > 2048 or not 1 <= m["top_k"] <= min(32, experts):
        raise ValueError("Invalid routed expert count/top_k")
    if m["ep"] * m["tokens_per_rank"] * m["top_k"] % experts:
        raise ValueError("Balanced routing requires an integer token count per expert")
    if m["max_tokens_per_rank"] * m["top_k"] > 32768 or m["tiling_key"] != 730152:
        raise ValueError(
            "This dtype/dispatch profile requires key 730152 and int16 internal route indices"
        )
    received = (
        m["ep"]
        * m["tokens_per_rank"]
        * m["top_k"]
        // experts
        * m["moe_experts_per_rank"]
    )
    if m["max_recv_tokens"] and m["max_recv_tokens"] < received:
        raise ValueError(
            "max_recv_tokens is smaller than actual per-rank received tokens"
        )
    if not 197 * 1024 <= m["ub_bytes"] <= 248 * 1024:
        raise ValueError(
            "Set ub_bytes to the target platform UB capacity (supported 197..248 KiB)"
        )
    if cfg["fixture"]["kind"] not in ("sparse", "random"):
        raise ValueError("fixture.kind must be sparse or random")
    if not -8 <= cfg["fixture"]["scale_exponent"] <= 0:
        raise ValueError("fixture.scale_exponent must be in -8..0")


def cann_env():
    env = os.environ.copy()
    if not env.get("ASCEND_HOME_PATH"):
        raise RuntimeError("Source the CANN set_env.sh before building")
    env.update(CCACHE_DISABLE="1", PYTHONDONTWRITEBYTECODE="1")
    return env
