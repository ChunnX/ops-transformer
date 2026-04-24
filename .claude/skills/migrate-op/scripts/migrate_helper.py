#!/usr/bin/env python3
"""Helper script for AscendC operator migration.

Usage:
    python scripts/migrate_helper.py find <operator_name>
    python scripts/migrate_helper.py create <operator_name> <level> <category>
    python scripts/migrate_helper.py extract <src_dir> <dst_dir>
    python scripts/migrate_helper.py list_existing <opname>
    python scripts/migrate_helper.py is_complex <src_dir>
    python scripts/migrate_helper.py discover <src_dir>
    python scripts/migrate_helper.py extract_complex <src_dir> <dst_dir>
"""

import argparse
import json
import os
import pathlib
import shutil
import sys

import re

TEMPLATES_DIR = pathlib.Path(__file__).resolve().parent.parent / "templates"

BENCH_ROOT = pathlib.Path("/home/repo/NPUKernelBench")
TASKS_ROOT = BENCH_ROOT / "tasks"

OPS_DIRS = [
    pathlib.Path("/home/repo/ops-nn"),
    pathlib.Path("/home/repo/ops-cv"),
    pathlib.Path("/home/repo/ops-math"),
    pathlib.Path("/home/repo/ops-transformer")
]


def pascal_to_snake(name):
    """Convert PascalCase to snake_case (e.g., AdaptiveAvgPool3d -> adaptive_avg_pool3d)."""
    # Insert underscore before uppercase letters, except at word boundaries
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
    return s.lower()


def snake_to_pascal(name):
    """Convert snake_case to PascalCase (e.g., adaptive_avg_pool3d -> AdaptiveAvgPool3d).

    If the name does not contain underscores, it is assumed to already be PascalCase
    and is returned as-is.
    """
    if '_' not in name:
        return name
    return ''.join(part.capitalize() for part in name.split('_'))

CATEGORY_MAP = {
    # ops-nn
    "activation": ("level2", "Activation"),
    "conv": ("level2", "Activation"),
    "control": ("level1", "Condition"),
    "foreach": ("level2", "Foreach"),
    "hash": ("level1", "Index"),
    "index": ("level1", "Index"),
    "loss": ("level2", "Loss"),
    "matmul": ("level3", "GMM"),
    "norm": ("level2", "Norm"),
    "optim": ("level2", "Optim"),
    "pooling": ("level2", "Pooling"),
    "quant": ("level2", "Conversion"),
    "rnn": ("level2", "Activation"),
    # ops-cv
    "image": ("level2", "Image"),
    "objdetect": ("level2", "Image"),
    # ops-math
    "math": ("level1", "Math"),
}


def find_operator(op_name):
    """Find operator source directory in ops-nn/ops-cv/ops-math.

    Accepts both PascalCase (e.g. AdaptiveAvgPool3d) and snake_case (e.g. adaptive_avg_pool3d).
    NPUKernelBench uses PascalCase; ops repos use snake_case.
    Both forms are tried when searching.
    """
    # Generate candidate names to try: original + converted
    snake_name = pascal_to_snake(op_name)
    pascal_name = snake_to_pascal(op_name)
    candidates = list(dict.fromkeys([op_name, snake_name, pascal_name]))

    for ops_dir in OPS_DIRS:
        if not ops_dir.exists():
            continue
        for category_dir in sorted(ops_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for candidate in candidates:
                op_dir = category_dir / candidate
                if op_dir.is_dir() and (op_dir / "op_host").is_dir():
                    src_category = category_dir.name
                    repo = ops_dir.name
                    level, task_category = CATEGORY_MAP.get(src_category, ("level1", "Math"))
                    return {
                        "src_dir": str(op_dir),
                        "repo": repo,
                        "src_category": src_category,
                        "level": level,
                        "task_category": task_category,
                        "ops_dir_name": candidate,
                        "bench_dir_name": pascal_name,
                    }
    return None


def create_project(op_name, level, category):
    """Create project directory structure under tasks/.

    The directory name is always PascalCase regardless of input format.
    """
    bench_name = snake_to_pascal(op_name)
    task_dir = TASKS_ROOT / level / category / bench_name
    dirs = [
        task_dir / "answer" / "0" / "op_host",
        task_dir / "answer" / "0" / "op_kernel",
        task_dir / "question" / "op_host",
        task_dir / "question" / "op_kernel",
        task_dir / "validation",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return str(task_dir)


def extract_files(src_dir, dst_dir):
    """Extract files from CANN ops source directory.

    Files matching _tiling.cpp, _tiling.h, _def.cpp are listed for LLM transformation.
    Other files (.cpp, .h, etc.) are copied directly.
    After extraction, a CMakeLists.txt is generated automatically.
    """
    src = pathlib.Path(src_dir)
    dst = pathlib.Path(dst_dir)
    transform_suffixes = ["_tiling.cpp", "_tiling.h", "_def.cpp"]
    skip_suffixes = ["_infershape.cpp"]
    target_dirs = ["op_host", "op_kernel"]
    result = {"to_transform": [], "copied": []}

    for sub_dir_name in target_dirs:
        sub_dir = src / sub_dir_name
        if not sub_dir.exists():
            continue

        out_sub_dir = dst / sub_dir_name
        out_sub_dir.mkdir(parents=True, exist_ok=True)

        for f in sorted(sub_dir.iterdir()):
            if not f.is_file():
                continue
            if f.name.endswith(".txt"):
                continue

            rel_path = f"{sub_dir_name}/{f.name}"

            if any(f.name.endswith(s) for s in skip_suffixes):
                result["copied"].append(f"SKIPPED: {rel_path}")
                continue

            if any(f.name.endswith(s) for s in transform_suffixes):
                content = f.read_text(encoding="utf-8", errors="ignore")
                result["to_transform"].append({
                    "rel_path": rel_path,
                    "abs_path": str(f),
                    "content": content,
                    "lines": content.count("\n") + 1,
                })
            else:
                # Copy directly (kernel .cpp, helper .h, etc.)
                target = out_sub_dir / f.name
                shutil.copy2(f, target)
                result["copied"].append(rel_path)

    # Auto-generate CMakeLists.txt
    _generate_cmakelists(dst)
    result["copied"].append("CMakeLists.txt")

    return result


def _load_template(name):
    """Load a template file from the templates directory."""
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


def _generate_cmakelists(dst_dir):
    """Generate CMakeLists.txt from template file."""
    (pathlib.Path(dst_dir) / "CMakeLists.txt").write_text(
        _load_template("root_CMakeLists.txt"), encoding="utf-8"
    )
    print("GENERATED: CMakeLists.txt")


def check_existing(op_name):
    """Check if operator already exists in tasks/.

    Accepts both PascalCase and snake_case. NPUKernelBench uses PascalCase for directories.
    """
    pascal_name = snake_to_pascal(op_name)
    candidates = list(dict.fromkeys([op_name, pascal_name]))

    results = []
    for level_dir in TASKS_ROOT.iterdir():
        if not level_dir.is_dir():
            continue
        for cat_dir in level_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            for candidate in candidates:
                op_dir = cat_dir / candidate
                if op_dir.is_dir():
                    results.append(str(op_dir))
    return results


# ---------------------------------------------------------------------------
# Complexity detection
# ---------------------------------------------------------------------------

# Framework-internal headers not available in msopgen standalone build
FRAMEWORK_INTERNAL_HEADERS = {
    "op_cache_tiling.h",
    "runtime_kb_api.h",
    "op_cache_def_tiling.h",
}


def _read_source_files(src_dir, sub_dirs=("op_host",)):
    """Read all .cpp/.h files under src_dir/<sub_dirs>/."""
    src = pathlib.Path(src_dir)
    files = {}
    for sub in sub_dirs:
        sub_dir = src / sub
        if not sub_dir.exists():
            continue
        for f in sub_dir.rglob("*"):
            if f.is_file() and f.suffix in (".cpp", ".h") and "arch35" not in f.parts:
                files[str(f)] = f.read_text(encoding="utf-8", errors="ignore")
    return files


def check_complexity(src_dir):
    """Check if an operator is 'complex' based on detection criteria.

    Returns dict with:
        is_complex: bool
        reasons: list[str] -- matched criteria descriptions
    """
    reasons = []
    files = _read_source_files(src_dir)

    # 1. op_host/ include points to sibling operator directories
    for path, content in files.items():
        if "op_host" not in path:
            continue
        # Pattern: #include "../<other_op>/..." (cross-operator reference)
        if re.search(r'#include\s*[<"]\.\./[^.][^/>"]+/', content):
            reasons.append("op_host include points to sibling operator directory")
            break
        # Pattern: #include "matmul/..." (cross-category reference)
        if re.search(r'#include\s*"matmul/(common|mat_mul_v3|batch_mat_mul_v3)/', content):
            if "op_host include references cross-operator directory" not in reasons:
                reasons.append("op_host include references cross-operator directory")
            break

    # 2. common/ subdirectory with tiling headers
    src = pathlib.Path(src_dir)
    common_dir = src / "op_host" / "common"
    if common_dir.exists():
        tiling_headers = list(common_dir.glob("tiling*.h"))
        if tiling_headers:
            reasons.append("common/ subdirectory contains tiling headers")

    # 3. Tiling class inheritance chain > 1 level
    for path, content in files.items():
        # Look for class declarations inheriting from something other than OpDef
        # Pattern: class XxxTiling : public YyyBaseTiling where Yyy != OpDef
        if re.search(r'class\s+\w+\s*:\s*public\s+\w+Base\w*|class\s+\w+\s*:\s*public\s+\w+Tiling', content):
            if re.search(r':\s*public\s+(?!OpDef\b)\w+', content):
                reasons.append("Tiling class inheritance chain > 1 level")
                break

    # 4. CompileInfo contains cube hardware params or platform feature detection
    #    (simple TilingParse with empty/basic CompileInfo does NOT make it complex)
    cube_params = ['l0ASize', 'l0BSize', 'l0CSize', 'l2Size', 'btSize', 'cubeFreq',
                   'npuArch', 'supportL0c2out', 'supportL12BtBf16', 'supportMmad']
    for path, content in files.items():
        if re.search(r'TilingParse<\w+CompileInfo>', content):
            # Check if the CompileInfo struct contains cube hardware params
            # Find CompileInfo struct definition in nearby files
            for p2, c2 in files.items():
                if re.search(r'struct\s+\w+CompileInfo', c2):
                    for param in cube_params:
                        if param in c2:
                            reasons.append(f"CompileInfo contains cube hardware param ({param})")
                            break
                    else:
                        continue
                    break
            break

    # 5. ASCENDC_TPL_* or GenSimplifiedKey
    for path, content in files.items():
        if 'ASCENDC_TPL_' in content or 'GenSimplifiedKey' in content:
            reasons.append("contains ASCENDC_TPL_* or GenSimplifiedKey")
            break

    # 6. Framework-internal header references
    for path, content in files.items():
        for header in FRAMEWORK_INTERNAL_HEADERS:
            if header in content:
                if "references framework-internal header" not in reasons:
                    reasons.append(f"references framework-internal header ({header})")
                break

    return {"is_complex": len(reasons) > 0, "reasons": reasons}


# ---------------------------------------------------------------------------
# Dependency discovery
# ---------------------------------------------------------------------------

INCLUDE_RE = re.compile(r'#include\s*[<\"]([^\">]+)[\">]')


def discover_dependencies(src_dir, max_depth=3):
    """Discover cross-operator dependencies and framework-internal deps.

    Returns dict with:
        direct_deps: dict[str, list[str]] -- {dir_path: [file_list]}
        framework_internal: list[str] -- files needing stubs
        all_includes: list[tuple[str, str]] -- (source_file, include_path)
    """
    src = pathlib.Path(src_dir).resolve()
    ops_root = None
    for ops_dir in OPS_DIRS:
        if ops_dir.exists():
            try:
                src.relative_to(ops_dir)
                ops_root = ops_dir
                break
            except ValueError:
                continue

    direct_deps = {}
    framework_internal = []
    all_includes = []

    files = _read_source_files(src_dir, sub_dirs=("op_host", "op_kernel"))

    for fpath, content in files.items():
        for inc_match in INCLUDE_RE.finditer(content):
            inc_path = inc_match.group(1)
            all_includes.append((fpath, inc_path))

            # Check framework-internal
            inc_basename = os.path.basename(inc_path)
            if inc_basename in FRAMEWORK_INTERNAL_HEADERS:
                if inc_basename not in framework_internal:
                    framework_internal.append(inc_basename)

            # Check cross-operator reference
            if ops_root is None:
                continue
            # Pattern: "matmul/mat_mul_v3/op_host/op_tiling/xxx.h"
            # Pattern: "matmul/common/op_host/xxx.h"
            dep_match = re.match(
                r'(\w+)/(\w+)/op_host/(.+)', inc_path
            )
            if dep_match:
                dep_dir = str(ops_root / dep_match.group(1) / dep_match.group(2))
                dep_file = dep_match.group(3)
                if dep_dir not in direct_deps:
                    direct_deps[dep_dir] = []
                if dep_file not in direct_deps[dep_dir]:
                    direct_deps[dep_dir].append(dep_file)

            # Pattern: "../../op_kernel/xxx.h" -> resolve relative
            if inc_path.startswith("../"):
                try:
                    resolved = (pathlib.Path(fpath).parent / inc_path).resolve()
                    if ops_root in resolved.parents:
                        rel = resolved.relative_to(ops_root)
                        parts = rel.parts
                        if len(parts) >= 3:
                            dep_dir = str(ops_root / parts[0] / parts[1])
                            dep_file = str(pathlib.Path(*parts[2:]))
                            if dep_dir != str(src):
                                if dep_dir not in direct_deps:
                                    direct_deps[dep_dir] = []
                                if dep_file not in direct_deps[dep_dir]:
                                    direct_deps[dep_dir].append(dep_file)
                except (ValueError, RuntimeError):
                    pass

    return {
        "src_dir": str(src),
        "direct_deps": direct_deps,
        "framework_internal": framework_internal,
        "include_count": len(all_includes),
    }


# ---------------------------------------------------------------------------
# Include path rewriting
# ---------------------------------------------------------------------------

INCLUDE_REWRITE_RULES = [
    # "matmul/mat_mul_v3/op_host/op_tiling/xxx.h" -> "matmul_v3/xxx.h"
    (r'"matmul/mat_mul_v3/op_host/op_tiling/([^"]+)"', r'"matmul_v3/\1"'),
    # "matmul/mat_mul_v3/op_kernel/xxx.h" -> "matmul_v3_kernel/xxx.h"
    (r'"matmul/mat_mul_v3/op_kernel/([^"]+)"', r'"matmul_v3_kernel/\1"'),
    (r'\.\./\.\./op_kernel/([^"]+)"', r'"matmul_v3_kernel/\1"'),
    # "matmul/common/op_host/xxx.h" -> "common/xxx.h"
    (r'"matmul/common/op_host/([^"]+)"', r'"common/\1"'),
    # "matmul/common/op_kernel/xxx.h" -> "common/xxx.h"
    (r'"matmul/common/op_kernel/([^"]+)"', r'"common/\1"'),
    # "matmul/common/inc/op_host/xxx.h" -> "common/xxx.h"
    (r'"matmul/common/inc/op_host/([^"]+)"', r'"common/\1"'),
]


def rewrite_includes(content):
    """Rewrite cross-operator include paths to local paths."""
    for pattern, replacement in INCLUDE_REWRITE_RULES:
        content = re.sub(pattern, replacement, content)
    # Remove framework-internal includes
    for header in FRAMEWORK_INTERNAL_HEADERS:
        content = re.sub(
            rf'^\s*#include\s*[<"]({re.escape(header)})[>"].*\n',
            f'// REMOVED: #include "\\1" (framework-internal, stubbed)\n',
            content,
            flags=re.MULTILINE,
        )
    return content


# ---------------------------------------------------------------------------
# Complex extraction
# ---------------------------------------------------------------------------

def extract_complex(src_dir, dst_dir):
    """Extract files for complex operators with cross-operator dependencies.

    Steps:
    1. Copy local op_host/op_kernel files (excluding arch35, config, tests, docs)
    2. Discover and copy dependency files
    3. Rewrite include paths in all copied files
    4. Generate CMakeLists.txt with GLOB_RECURSE
    """
    src = pathlib.Path(src_dir).resolve()
    dst = pathlib.Path(dst_dir).resolve()
    result = {"copied": [], "rewritten": [], "skipped": []}

    # --- Step 1: Copy local op_host files ---
    skip_subdirs = {"arch35", "config", "op_api"}
    skip_prefixes = ("test_",)
    host_dst = dst / "op_host"
    host_dst.mkdir(parents=True, exist_ok=True)

    op_host_dir = src / "op_host"
    if op_host_dir.exists():
        for f in sorted(op_host_dir.rglob("*")):
            if not f.is_file():
                continue
            # Skip unwanted subdirs and files
            if any(s in f.parts for s in skip_subdirs):
                result["skipped"].append(f.relative_to(src))
                continue
            rel = f.relative_to(op_host_dir)
            if any(rel.name.startswith(p) for p in skip_prefixes):
                result["skipped"].append(f.relative_to(src))
                continue

            target = host_dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            result["copied"].append(f"op_host/{rel}")

    # --- Step 1b: Copy local op_kernel files ---
    kernel_dst = dst / "op_kernel"
    kernel_dst.mkdir(parents=True, exist_ok=True)

    op_kernel_dir = src / "op_kernel"
    if op_kernel_dir.exists():
        for f in sorted(op_kernel_dir.rglob("*")):
            if not f.is_file():
                continue
            if any(s in f.parts for s in skip_subdirs):
                result["skipped"].append(f.relative_to(src))
                continue
            rel = f.relative_to(op_kernel_dir)
            if any(rel.name.startswith(p) for p in skip_prefixes):
                result["skipped"].append(f.relative_to(src))
                continue

            target = kernel_dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            result["copied"].append(f"op_kernel/{rel}")

    # --- Step 2: Discover and copy dependency files ---
    deps = discover_dependencies(str(src))
    for dep_dir_str, dep_files in deps["direct_deps"].items():
        dep_dir = pathlib.Path(dep_dir_str)
        if not dep_dir.exists():
            result["skipped"].append(f"DEP_NOT_FOUND: {dep_dir_str}")
            continue
        for dep_file in dep_files:
            src_file = dep_dir / "op_host" / dep_file
            if not src_file.exists():
                src_file = dep_dir / "op_kernel" / dep_file
            if not src_file.exists():
                # Try to find in common/inc
                for ops_dir in OPS_DIRS:
                    alt = ops_dir / "common" / "inc" / "op_host" / dep_file
                    if alt.exists():
                        src_file = alt
                        break
            if not src_file.exists():
                result["skipped"].append(f"DEP_FILE_NOT_FOUND: {dep_file}")
                continue

            # Determine target path based on dep_dir structure
            dep_parts = pathlib.Path(dep_file).parts
            if len(dep_parts) >= 2:
                target_sub = host_dst / pathlib.Path(*dep_parts[:-1])
            else:
                target_sub = host_dst
            target_sub.mkdir(parents=True, exist_ok=True)
            target = target_sub / dep_parts[-1]

            if not target.exists():
                shutil.copy2(src_file, target)
                result["copied"].append(f"DEP: {dep_file}")

    # --- Step 3: Rewrite include paths in ALL copied files ---
    for cpp_h in sorted(host_dst.rglob("*")):
        if cpp_h.is_file() and cpp_h.suffix in (".cpp", ".h"):
            content = cpp_h.read_text(encoding="utf-8", errors="ignore")
            new_content = rewrite_includes(content)
            if new_content != content:
                cpp_h.write_text(new_content, encoding="utf-8")
                result["rewritten"].append(str(cpp_h.relative_to(dst)))

    # Also rewrite kernel includes
    for cpp_h in sorted(kernel_dst.rglob("*")):
        if cpp_h.is_file() and cpp_h.suffix in (".cpp", ".h"):
            content = cpp_h.read_text(encoding="utf-8", errors="ignore")
            new_content = rewrite_includes(content)
            if new_content != content:
                cpp_h.write_text(new_content, encoding="utf-8")
                result["rewritten"].append(str(cpp_h.relative_to(dst)))

    # --- Step 4: Generate CMakeLists.txt ---
    (dst / "CMakeLists.txt").write_text(_load_template("root_CMakeLists.txt"), encoding="utf-8")
    result["copied"].append("CMakeLists.txt")

    (host_dst / "CMakeLists.txt").write_text(_load_template("op_host_CMakeLists.txt"), encoding="utf-8")
    result["copied"].append("op_host/CMakeLists.txt")

    return result


def main():
    parser = argparse.ArgumentParser(description="AscendC operator migration helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_find = subparsers.add_parser("find", help="Find operator source")
    p_find.add_argument("op_name")

    p_create = subparsers.add_parser("create", help="Create project structure")
    p_create.add_argument("op_name")
    p_create.add_argument("level")
    p_create.add_argument("category")

    p_extract = subparsers.add_parser("extract", help="Extract files from source")
    p_extract.add_argument("src_dir")
    p_extract.add_argument("dst_dir")

    p_check = subparsers.add_parser("list_existing", help="Check if operator exists in tasks")
    p_check.add_argument("op_name")

    p_complex = subparsers.add_parser("is_complex", help="Check operator complexity")
    p_complex.add_argument("src_dir")

    p_discover = subparsers.add_parser("discover", help="Discover cross-operator dependencies")
    p_discover.add_argument("src_dir")

    p_extract_complex = subparsers.add_parser("extract_complex", help="Extract with dependency resolution")
    p_extract_complex.add_argument("src_dir")
    p_extract_complex.add_argument("dst_dir")

    args = parser.parse_args()

    if args.command == "find":
        info = find_operator(args.op_name)
        if info:
            # Add complexity info
            complexity = check_complexity(info["src_dir"])
            info["is_complex"] = complexity["is_complex"]
            info["complexity_reasons"] = complexity["reasons"]
            print(json.dumps(info, indent=2))
            # Also list files
            src = pathlib.Path(info["src_dir"])
            for sub in ["op_host", "op_kernel"]:
                sub_dir = src / sub
                if sub_dir.exists():
                    for f in sorted(sub_dir.iterdir()):
                        if f.is_file() and not f.name.endswith(".txt"):
                            print(f"FILE: {sub}/{f.name}")
        else:
            print("NOT_FOUND")
            sys.exit(1)

    elif args.command == "create":
        task_dir = create_project(args.op_name, args.level, args.category)
        print(f"CREATED: {task_dir}")

    elif args.command == "extract":
        result = extract_files(args.src_dir, args.dst_dir)
        for item in result["to_transform"]:
            print(f"TO_TRANSFORM: {item['rel_path']} ({item['lines']} lines)")
        for item in result["copied"]:
            print(f"COPIED: {item}")

    elif args.command == "list_existing":
        existing = check_existing(args.op_name)
        if existing:
            for p in existing:
                print(f"EXISTS: {p}")
        else:
            print("NOT_EXISTS")

    elif args.command == "is_complex":
        result = check_complexity(args.src_dir)
        print(json.dumps(result, indent=2))

    elif args.command == "discover":
        result = discover_dependencies(args.src_dir)
        print(json.dumps(result, indent=2))

    elif args.command == "extract_complex":
        result = extract_complex(args.src_dir, args.dst_dir)
        for item in result["copied"]:
            print(f"COPIED: {item}")
        for item in result["rewritten"]:
            print(f"REWRITTEN: {item}")
        for item in result["skipped"]:
            print(f"SKIPPED: {item}")


if __name__ == "__main__":
    main()
