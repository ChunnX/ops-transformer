#!/usr/bin/env python3
"""Build and stage the arena FlashKDA Entry for the built-in ACLNN DSO."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ENTRY = "FlashKda"
VENDOR = "builtin"
VENDOR_ROOT = f"packages/vendors/{VENDOR}"
BUILTIN_ARTIFACT = f"opp/built-in/op_impl/cannbotdsl/{ENTRY}"


def _load_source(source: Path):
    spec = importlib.util.spec_from_file_location("flash_kda", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load FlashKDA source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rewrite_for_builtin(stage: Path, processor: str) -> None:
    """Map the framework vendor layout onto the CANN built-in layout.

    The generated ACLNN DSO is installed at ``opp/built-in/op_api/lib``. Its
    framework-owned Provider lookup can therefore retain the canonical
    ``../../op_impl/cannbotdsl`` relative layout.
    """

    manifest_path = stage / "stage.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_artifact = f"{VENDOR_ROOT}/op_impl/cannbotdsl/{ENTRY}"
    expected_headers = f"{VENDOR_ROOT}/op_api/include"
    if manifest.get("artifact_install_destination") != expected_artifact:
        raise RuntimeError("unexpected framework artifact destination")
    if manifest.get("api_header_install_destination") != expected_headers:
        raise RuntimeError("unexpected framework API header destination")
    manifest["artifact_install_destination"] = BUILTIN_ARTIFACT
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
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    parent_opp = args.parent_opp.resolve(strict=True)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    import cannbotdsl
    import cannbot_framework

    module = _load_source(source)
    bundle = module.build_flash_kda(
        cannbotdsl.Target(
            "dav-3510",
            aic_count=24,
            aiv_count=48,
            barrier_limit=16,
            ub_bytes=256 * 1024,
        )
    )
    staged = cannbot_framework.stage(
        bundle=bundle,
        export=cannbot_framework.AclnnExportSpec(
            operator=ENTRY,
            public_name=ENTRY,
            l0_name=ENTRY,
            op_type=ENTRY,
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
