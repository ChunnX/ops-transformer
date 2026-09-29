"""Archive a completed matrix as reviewable text, without binaries or local path dependencies."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="archive a stopped run as counterevidence, not a successful matrix")
    args = parser.parse_args()
    source = args.results.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        parser.error("choose a new evidence directory; existing evidence is never overwritten")
    summary = json.loads((source / "summary.json").read_text())
    rows = summary["results"]
    if not rows or (not args.allow_partial and (not summary.get("completed") or not all(row["ok"] for row in rows))):
        parser.error("matrix is incomplete or has unexpected outcomes")
    work = source.parent.parent
    snapshots = {}
    for row in rows:
        if digest(source / row["log"]) != row["log_sha256"]:
            raise RuntimeError(f"log changed: {row['log']}")
        manifest = json.loads((work / row["case"] / "manifest.json").read_text())
        if manifest["audit"]["sha256"] != row["binary_sha256"]:
            raise RuntimeError("build changed since matrix execution")
        snapshots[row["case"]] = manifest
    destination.mkdir(parents=True)

    def sanitized(text):
        return text.replace(str(ROOT), "<REPO>").replace(sys.prefix, "<PYTHON_ENV>")

    for row in rows:
        target = destination / row["log"]
        target.write_text(sanitized((source / row["log"]).read_text()))
        row["raw_log_sha256"] = row["log_sha256"]
        row["log_sha256"] = digest(target)
    summary["archive_note"] = "Local checkout/Python paths normalized; raw log hashes preserved; no binary included."
    summary["partial_archive"] = args.allow_partial
    summary["source_run"] = source.name
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, manifest in snapshots.items():
        (destination / f"{name}.manifest.json").write_text(sanitized(json.dumps(manifest, indent=2)) + "\n")
        (destination / f"{name}.disasm").write_text(sanitized((work / name / "kernel.disasm").read_text()))
    environment = dict(date=source.name.split("_")[0], machine=platform.machine(),
                       python=platform.python_version(), torch=importlib.metadata.version("torch"),
                       torch_npu=importlib.metadata.version("torch-npu"),
                       cann=os.environ.get("SLIG_CANN_PATH", "see manifests"),
                       runtime_test_dtype="bfloat16", device=summary["device"])
    (destination / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(f"Archived {len(rows)} runs / {len(snapshots)} configurations: {destination}")


if __name__ == "__main__":
    main()
