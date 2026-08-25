#!/usr/bin/env python3
"""Verify that MoeTokenUnpermuteWithEp Ascend950 changes are platform-isolated."""

import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path


KERNEL_GUARD = "defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)"
HOST_GUARD = "defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)"
TARGETS = (
    ("moe/moe_token_unpermute_with_ep/op_kernel/moe_token_unpermute_with_ep.h", KERNEL_GUARD),
    ("moe/moe_token_unpermute_with_ep/op_host/moe_token_unpermute_with_ep_tiling.cpp", HOST_GUARD),
    ("moe/moe_token_unpermute_with_ep/op_host/moe_token_unpermute_with_ep_tiling.h", HOST_GUARD),
)


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"cannot find repository root from {start}")


def git_show(repo: Path, ref: str, relative_path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{ref}:{relative_path}"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def select_guard(text: str, guard: str, take_true: bool) -> str:
    """Select one branch of an exact guard while preserving unrelated directives."""
    lines = text.splitlines(keepends=True)
    selected = []
    index = 0
    marker = f"#if {guard}"
    while index < len(lines):
        if lines[index].strip() != marker:
            selected.append(lines[index])
            index += 1
            continue

        depth = 1
        cursor = index + 1
        else_index = None
        while cursor < len(lines):
            directive = lines[cursor].strip()
            if directive.startswith("#if"):
                depth += 1
            elif directive == "#endif":
                depth -= 1
                if depth == 0:
                    break
            elif directive == "#else" and depth == 1:
                else_index = cursor
            cursor += 1
        if depth != 0:
            raise AssertionError(f"unterminated platform guard: {marker}")

        true_end = else_index if else_index is not None else cursor
        false_start = (else_index + 1) if else_index is not None else cursor
        branch = lines[index + 1 : true_end] if take_true else lines[false_start:cursor]
        selected.append(select_guard("".join(branch), guard, take_true))
        index = cursor + 1
    return "".join(selected)


def normalized_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_non950(repo: Path, baseline_ref: str, relative_path: str, guard: str) -> dict:
    current = (repo / relative_path).read_text(encoding="utf-8")
    baseline = git_show(repo, baseline_ref, relative_path)
    non950 = select_guard(current, guard, take_true=False)
    current_lines = normalized_lines(non950)
    baseline_lines = normalized_lines(baseline)
    diff = list(
        difflib.unified_diff(
            baseline_lines,
            current_lines,
            fromfile=f"{baseline_ref}:{relative_path}",
            tofile=f"non950:{relative_path}",
            lineterm="",
        )
    )
    return {
        "path": relative_path,
        "guard": guard,
        "guard_count": current.count(f"#if {guard}"),
        "baseline_nonempty_lines": len(baseline_lines),
        "non950_nonempty_lines": len(current_lines),
        "non950_exact_baseline": not diff,
        "diff": diff[:80],
    }


def check_cmake(repo: Path, baseline_ref: str) -> dict:
    relative_path = "moe/moe_token_unpermute_with_ep/op_host/CMakeLists.txt"
    text = (repo / relative_path).read_text(encoding="utf-8")
    baseline = git_show(repo, baseline_ref, relative_path)
    compact = " ".join(text.split())
    baseline_compact = " ".join(baseline.split())
    legacy_host_target = (
        "target_sources(op_host_aclnn PRIVATE "
        "moe_token_unpermute_with_ep_def.cpp )"
    )
    legacy_compile_options = (
        "add_ops_compile_options( OP_NAME MoeTokenUnpermuteWithEp "
        "OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Werror )"
    )
    legacy_aclnn_selection = (
        "add_modules_sources(OPTYPE moe_token_unpermute_with_ep ACLNNTYPE aclnn)"
    )
    checks = {
        "ascend950_condition": 'if("${CONDITION_UNIT}" STREQUAL "ascend950")' in text,
        "ascend950_aclnn_exclude": "ACLNNTYPE aclnn_exclude" in text,
        "ascend950_manual_api_dir": "OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op_api/ascend950" in text,
        "ascend950_host_macro": "MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950=1" in text,
        "non950_auto_aclnn": "ACLNNTYPE aclnn)" in text,
        "non950_host_target": "target_sources(op_host_aclnn PRIVATE" in text,
        "ascend950_host_target": "target_sources(op_host_aclnnExc PRIVATE" in text,
        "baseline_host_target_preserved_once": (
            legacy_host_target in baseline_compact and compact.count(legacy_host_target) == 1
        ),
        "baseline_compile_options_preserved_once": (
            legacy_compile_options in baseline_compact and compact.count(legacy_compile_options) == 1
        ),
        "baseline_aclnn_selection_preserved_once": (
            legacy_aclnn_selection in baseline_compact and compact.count(legacy_aclnn_selection) == 1
        ),
    }
    api_root = repo / "moe/moe_token_unpermute_with_ep/op_host/op_api"
    top_level_sources = sorted(path.name for path in api_root.glob("*.cpp"))
    ascend950_sources = sorted(path.name for path in (api_root / "ascend950").glob("*.cpp"))
    checks["no_shared_manual_api_sources"] = not top_level_sources
    checks["ascend950_manual_api_sources"] = ascend950_sources == [
        "aclnn_moe_token_unpermute_with_ep.cpp",
        "moe_token_unpermute_with_ep.cpp",
    ]
    return {
        "path": relative_path,
        "checks": checks,
        "top_level_cpp": top_level_sources,
        "ascend950_cpp": ascend950_sources,
        "passed": all(checks.values()),
    }


def check_ascend910b_evidence(evidence: Path) -> dict:
    compile_commands_path = evidence / "cmake/compile_commands.json"
    cache_path = evidence / "cmake/CMakeCache.txt"
    opbuild_path = evidence / "cmake/opbuild_gen_default.build.make"
    tiling_flags_path = evidence / "cmake/ophost_transformer_tiling.flags.make"
    generated_cpp = evidence / "autogen/aclnn_moe_token_unpermute_with_ep.cpp"
    generated_header = evidence / "autogen/aclnn_moe_token_unpermute_with_ep.h"
    package = evidence / "cann-ops-transformer-custom_linux-x86_64.run"

    required = (
        compile_commands_path,
        cache_path,
        opbuild_path,
        tiling_flags_path,
        generated_cpp,
        generated_header,
        package,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"path": str(evidence), "missing": missing, "checks": {}, "passed": False}

    compile_entries = json.loads(compile_commands_path.read_text(encoding="utf-8"))
    commands = "\n".join(entry.get("command", "") for entry in compile_entries)
    files = [entry.get("file", "") for entry in compile_entries]
    cache = cache_path.read_text(encoding="utf-8")
    opbuild = opbuild_path.read_text(encoding="utf-8")
    tiling_flags = tiling_flags_path.read_text(encoding="utf-8")
    kernel_objects = sorted((evidence / "kernel_objects").glob("MoeTokenUnpermuteWithEp_*.o"))
    checks = {
        "compute_unit_is_ascend910b": "ASCEND_COMPUTE_UNIT:STRING=ascend910b" in cache,
        "automatic_aclnn_generation_enabled": "OPS_ACLNN_GEN=1" in opbuild,
        "automatic_aclnn_product_is_ascend910b": 'OPS_PRODUCT_NAME="ascend910b"' in opbuild,
        "legacy_op_host_target_selected": "libop_host_aclnn.so" in opbuild,
        "generated_aclnn_source_compiled": any(
            path.endswith("/build/autogen/aclnn_moe_token_unpermute_with_ep.cpp") for path in files
        ),
        "legacy_opdef_target_compiled": any(
            "CMakeFiles/op_host_aclnn.dir" in entry.get("output", "")
            and entry.get("file", "").endswith("moe_token_unpermute_with_ep_def.cpp")
            for entry in compile_entries
        ),
        "ascend950_manual_api_not_compiled": "op_api/ascend950" not in commands
        and all("op_api/ascend950" not in path for path in files),
        "ascend950_host_definition_absent": (
            "MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950" not in commands
            and "MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950" not in tiling_flags
        ),
        "generated_aclnn_files_nonempty": generated_cpp.stat().st_size > 0
        and generated_header.stat().st_size > 0,
        "three_legacy_kernel_objects_built": len(kernel_objects) == 3
        and all(path.stat().st_size > 0 for path in kernel_objects),
        "package_nonempty": package.stat().st_size > 0,
    }
    artifacts = (*required, *kernel_objects)
    return {
        "path": str(evidence),
        "missing": [],
        "checks": checks,
        "kernel_object_count": len(kernel_objects),
        "artifact_sha256": {str(path.relative_to(evidence)): sha256(path) for path in artifacts},
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--baseline-ref", default="origin/master")
    parser.add_argument("--ascend910b-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo = (args.repo_root or find_repo_root(Path(__file__).resolve())).resolve()
    source_results = [compare_non950(repo, args.baseline_ref, path, guard) for path, guard in TARGETS]
    host_sources = "\n".join(
        (repo / path).read_text(encoding="utf-8") for path, _ in TARGETS if "/op_host/" in path
    )
    kernel_source = (repo / TARGETS[0][0]).read_text(encoding="utf-8")
    marker_checks = {
        "host_does_not_use_npu_arch": "__NPU_ARCH__" not in host_sources,
        "kernel_uses_exact_3510_guard": f"#if {KERNEL_GUARD}" in kernel_source,
        "stable_reduction_is_present_for_3510": "REDUCTION_LANES" in kernel_source,
        "index_validation_is_present_for_3510": "ValidateIndices" in kernel_source,
    }
    cmake = check_cmake(repo, args.baseline_ref)
    ascend910b = (
        check_ascend910b_evidence(args.ascend910b_evidence.resolve())
        if args.ascend910b_evidence
        else None
    )
    passed = (
        all(item["non950_exact_baseline"] and item["guard_count"] > 0 for item in source_results)
        and all(marker_checks.values())
        and cmake["passed"]
        and (ascend910b is None or ascend910b["passed"])
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "baseline_ref": args.baseline_ref,
        "source_sha256": {
            path: sha256(repo / path)
            for path in (*[item[0] for item in TARGETS], cmake["path"])
        },
        "source_branch_comparisons": source_results,
        "marker_checks": marker_checks,
        "cmake_routing": cmake,
        "ascend910b_build_routing": ascend910b,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else repo / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
