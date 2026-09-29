"""Offline source build using the installed CANN SDK custom-op template.

No prebuilt OPP, old workspace, downloaded CMake/opbase or binary patch is used.
The copied SDK project and all generated binaries live under SLIG_WORK_DIR.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from audit_binary import audit

ROOT = Path(__file__).resolve().parents[2]
OP = "mlp_sparse_lightning_indexer_grad_kl_loss"
SOURCE = ROOT / "attention" / OP
OWNER = "klloss-source-reproducer-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, **kwargs):
    print("[BUILD] " + " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def build(variant, rounds, work, cann):
    name = "pipeline" if variant == "pipeline" else f"{variant}_r{rounds}"
    case = work / name
    marker = case / "manifest.json"
    if case.exists() and (not marker.is_file() or json.loads(marker.read_text()).get("owner") != OWNER):
        raise RuntimeError(f"refuse to overwrite an unowned directory: {case}")
    case.mkdir(parents=True, exist_ok=True)
    manifest = dict(owner=OWNER, variant=variant, rounds=None if variant == "pipeline" else rounds,
                    compiled=False, source_build=True, cann=str(cann), machine=platform.machine(),
                    vc_flag=0, vv_flag=15 if variant == "split" else 0)
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    template = cann / "tools/op_project_templates/ascendc/customize"
    if not template.is_dir():
        raise RuntimeError(f"CANN 9.1 SDK custom-op template missing: {template}")
    project = case / "sdk_project"
    shutil.copytree(template, project, dirs_exist_ok=True)  # Dereference SDK utility symlinks.
    for src in (SOURCE / "op_host").iterdir():
        if src.suffix in (".h", ".cpp"):
            shutil.copy2(src, project / "op_host" / src.name)
    for src in (SOURCE / "op_kernel").iterdir():
        if src.suffix in (".h", ".cpp"):
            shutil.copy2(src, project / "op_kernel" / src.name)
    if variant != "pipeline":
        for src in (SOURCE / "experiments/cv_vv_minimal").iterdir():
            if src.suffix in (".h", ".cpp"):
                shutil.copy2(src, project / "op_kernel" / src.name)
        (project / "op_kernel/slig_cv_vv_build_config.h").write_text(
            "// Generated build configuration.\n#pragma once\n"
            f"#define SLIG_CV_VV_ROUNDS {rounds}\n#define SLIG_CV_VV_VC_FLAG 0\n"
            f"#define SLIG_CV_VV_VV_FLAG {manifest['vv_flag']}\n")
    (project / "op_kernel/CMakeLists.txt").write_text(
        f"add_ops_compile_options(MlpSparseLightningIndexerGradKLLoss OPTIONS "
        "--cce-auto-sync=off -Wno-deprecated-declarations)\nadd_kernels_compile()\n")
    build_dir = case / "build"
    install = case / "install"
    run(["cmake", "-S", project, "-B", build_dir,
         f"-DASCEND_CANN_PACKAGE_PATH={cann}", f"-DASCEND_PYTHON_EXECUTABLE={sys.executable}",
         "-DASCEND_COMPUTE_UNIT=ascend910b", "-Dvendor_name=klloss_repro",
         "-DENABLE_BINARY_PACKAGE=ON", "-DENABLE_SOURCE_PACKAGE=ON",
         "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={install}"])
    run(["cmake", "--build", build_dir, "-j", os.environ.get("SLIG_JOBS", "8")])
    run(["cmake", "--build", build_dir, "--target", "binary", "-j", os.environ.get("SLIG_JOBS", "8")])
    run(["cmake", "--install", build_dir])
    vendor = install / "packages/vendors/klloss_repro"
    manifest["vendor"] = str(vendor)
    artifacts = {}
    for path in sorted(vendor.rglob("*")):
        if path.is_file() and path.suffix in (".so", ".o", ".json"):
            artifacts[str(path.relative_to(vendor))] = sha(path)
    if not any(name.endswith(".o") for name in artifacts):
        raise RuntimeError("no compiled device binaries in generated OPP")
    manifest["artifacts"] = artifacts
    manifest["source_sha256"] = {
        str(p.relative_to(ROOT)): sha(p) for parent in (SOURCE, ROOT / "scripts/klloss")
        for p in sorted(parent.rglob("*")) if p.is_file() and p.suffix in (".py", ".h", ".cpp", ".sh")
    }
    audit(case, manifest)
    manifest["compiled"] = True
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[BUILD] READY {marker}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("pipeline", "same", "split", "all"))
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 65536:
        parser.error("rounds must be 1..65536")
    work = Path(os.environ.get("SLIG_WORK_DIR", ROOT / ".klloss_repro")).resolve()
    cann = Path(os.environ["SLIG_CANN_PATH"]).resolve()
    for variant in ("pipeline", "same", "split") if args.variant == "all" else (args.variant,):
        build(variant, args.rounds, work, cann)


if __name__ == "__main__":
    main()
