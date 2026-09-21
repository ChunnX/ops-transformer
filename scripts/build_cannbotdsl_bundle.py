#!/usr/bin/env python3
"""Build one conventional CANNBotDSL Entry and stage its built-in ACLNN export."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


VENDOR = "builtin"
VENDOR_ROOT = f"packages/vendors/{VENDOR}"


def _load_source(source: Path):
    module_name = f"_cannbotdsl_bundle_{source.stem}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CANNBotDSL source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rewrite_for_builtin(stage: Path, processor: str) -> None:
    """Map a framework vendor stage onto the parent OPP built-in layout."""

    manifest_path = stage / "stage.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    operator = manifest.get("operator")
    if not isinstance(operator, str) or not operator:
        raise RuntimeError("framework stage has no operator name")
    expected_artifact = f"{VENDOR_ROOT}/op_impl/cannbotdsl/{operator}"
    expected_headers = f"{VENDOR_ROOT}/op_api/include"
    if manifest.get("artifact_install_destination") != expected_artifact:
        raise RuntimeError("unexpected framework artifact destination")
    if manifest.get("api_header_install_destination") != expected_headers:
        raise RuntimeError("unexpected framework API header destination")
    manifest["artifact_install_destination"] = (
        f"opp/built-in/op_impl/cannbotdsl/{operator}"
    )
    manifest["api_header_install_destination"] = (
        f"{processor}-linux/include/aclnnop"
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent-opp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--npu-arch", required=True)
    parser.add_argument("--aic-count", type=int, default=0)
    parser.add_argument("--aiv-count", type=int, default=0)
    parser.add_argument("--barrier-limit", type=int, default=0)
    parser.add_argument("--ub-bytes", type=int, default=0)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    parent_opp = args.parent_opp.resolve(strict=True)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    import cannbotdsl
    import cannbot_framework

    module = _load_source(source)
    builder_name = f"build_{source.stem}"
    builder = getattr(module, builder_name, None)
    if not callable(builder):
        raise AttributeError(
            f"{source} must expose the conventional builder {builder_name}(target)"
        )
    bundle = builder(
        cannbotdsl.Target(
            args.npu_arch,
            aic_count=args.aic_count,
            aiv_count=args.aiv_count,
            barrier_limit=args.barrier_limit,
            ub_bytes=args.ub_bytes,
        )
    )
    operator = bundle.public_schema.get("name")
    if not isinstance(operator, str) or not operator:
        raise RuntimeError("OperatorBundle public schema has no name")
    staged = cannbot_framework.stage(
        bundle=bundle,
        export=cannbot_framework.AclnnExportSpec(
            operator=operator,
            public_name=operator,
            l0_name=operator,
            op_type=operator,
            vendor=VENDOR,
        ),
        parent_opp=parent_opp,
        output=output,
    )
    if Path(staged) != output:
        raise RuntimeError(f"framework staged an unexpected path: {staged}")
    _rewrite_for_builtin(output, args.processor)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
