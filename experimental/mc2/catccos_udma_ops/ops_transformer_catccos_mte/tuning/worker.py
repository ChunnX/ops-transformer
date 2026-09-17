# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Torchrun-only bridge to production validation workers and tuning-only API."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

from .controller import read_json, write_json
from .schema import CommConfig, Platform


def install_override(package, runtime, payload, output):
    """Apply after device/group initialization; public math signatures stay intact."""
    config = CommConfig.from_dict(payload["config"])
    device = Platform(**payload["platform"])
    config.validate(payload["case"], device)
    original_init = package.init_mte_group

    def initialize(*args, **kwargs):
        result = original_init(*args, **kwargs)
        actual = runtime._get_tuning_platform()
        actual_device = Platform(**dict(device.to_dict(), **actual))
        if actual_device.block_dim != device.block_dim:
            raise ValueError(f"platform mismatch: block_dim plan={device.block_dim}, actual={actual_device.block_dim}")
        if actual_device.ub_bytes < 248 * 1024 or actual_device.payload_budget_bytes != device.payload_budget_bytes:
            raise ValueError(f"platform mismatch: payload UB plan={device.payload_budget_bytes}, actual={actual_device.payload_budget_bytes}")
        config.validate(payload["case"], actual_device)
        runtime._set_tuning_override(config.as_tuple())
        effective = tuple(runtime._get_requested_comm_config())
        if effective != config.as_tuple():
            raise ValueError(f"requested configuration did not apply: requested={config.as_tuple()}, readback={effective}")
        rank = int(os.environ["RANK"])
        write_json(output / f"effective_rank_{rank}.json", {
            "rank": rank, "config": config.to_dict(), "platform": actual,
            "platform_verified": True, "validation": "host strict no-clamp checker plus readback",
        })
        return result

    package.init_mte_group = initialize
    return original_init


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("accuracy", "performance"), required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    payload = read_json(args.payload)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # These imports are intentionally confined to worker execution.
    import ops_transformer_catccos_mte as package
    from ops_transformer_catccos_mte import runtime
    from ops_transformer_catccos_mte._build_versions import WITH_TUNING
    from ops_transformer_catccos_mte.validation.case_schema import write_resolved_cases

    if not WITH_TUNING:
        raise RuntimeError("search/replay requires a --with-tuning wheel; production wheel rejects overrides")
    original = install_override(package, runtime, payload, output)
    # Each rank reads its own file: concurrent JSON truncate/write operations
    # on a shared resolved_case.json can corrupt another rank's startup read.
    resolved = output / f"resolved_case_rank_{int(os.environ['RANK'])}.json"
    write_resolved_cases(resolved, [payload["case"]])
    module = importlib.import_module(f"ops_transformer_catccos_mte.validation.{args.stage}_worker")
    previous = sys.argv
    sys.argv = [module.__file__, "--resolved-cases" if args.stage == "accuracy" else "--resolved-case",
                str(resolved), "--output-dir", str(output)]
    try:
        return module.main()
    finally:
        sys.argv = previous
        package.init_mte_group = original


if __name__ == "__main__":
    raise SystemExit(main())
