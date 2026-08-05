#!/usr/bin/env python3
"""Generate the clearly labelled host-only parser/HTML contract fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run_critical_path_trace import decode_artifacts, write_index, write_json
from test_critical_path_trace import make_synthetic_capture


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate a synthetic (not hardware) MegaMoe trace parser fixture."
    )
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts/synthetic")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    make_synthetic_capture(output_dir)
    results = decode_artifacts(
        artifact_dir=output_dir,
        spec_path=here / "trace-spec.json",
        audit_path=here / "audit-implemented/trace_point_audit.json",
        trace_tool=here / "trace_tool.py",
        python=sys.executable,
        allow_truncated=False,
        expected_source_sha="synthetic-fixture",
    )
    normalized_path = output_dir / "rank0.normalized.json"
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    normalized["metadata"]["capture_kind"] = (
        "synthetic host-only contract fixture; NOT a device execution result"
    )
    normalized["metadata"]["raw_artifact"] = "rank0.raw"
    normalized_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Re-render after portable metadata labelling.
    from run_critical_path_trace import run_checked

    run_checked(
        [
            sys.executable,
            str(here / "trace_tool.py"),
            "render",
            "--spec",
            str(here / "trace-spec.json"),
            "--trace",
            str(normalized_path),
            "--audit",
            str(here / "audit-implemented/trace_point_audit.json"),
            "--output",
            str(output_dir / "rank0.html"),
        ]
    )
    manifest = {
        "status": "passed",
        "capture_kind": "synthetic host-only contract fixture; NOT hardware validation",
        "purpose": "exercise raw decode, header/config checks, wrap-safe order/bounds checks, and HTML rendering",
        "captures": results,
    }
    write_json(output_dir / "run-manifest.json", manifest)
    write_index(output_dir, results, manifest)
    print(f"[OK] synthetic fixture generated at {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
