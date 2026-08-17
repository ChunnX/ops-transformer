#!/usr/bin/env python3
"""Generalization runner for block sparse attention grad.

The runner deliberately reuses /home/dhn/test_bsa.py as the golden source so
its CPU formula and mask convention stay identical to the established test.
Each case runs in a fresh process to isolate the NPU runtime context.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OPP_ROOT = REPO_ROOT / "build_out/bsag_test_opp/vendors/custom_transformer"


QUICK_CASES = (
    ("block16_multiblock", 64, (16, 16), "BNSD", 1, 1),
    ("block32_multiblock", 128, (32, 32), "BNSD", 1, 1),
    ("block48_multiblock", 144, (48, 48), "BNSD", 1, 1),
    ("block64_multiblock", 128, (64, 64), "BNSD", 1, 1),
    ("block16_tail", 70, (16, 16), "BNSD", 1, 1),
    ("block16x32_tail", 70, (16, 32), "BNSD", 1, 1),
    ("block32x16_tail", 70, (32, 16), "BNSD", 1, 1),
    ("block128_regression", 404, (128, 128), "BNSD", 1, 1),
)

FULL_CASES = QUICK_CASES + (
    ("block16_bsnd", 64, (16, 16), "BSND", 1, 1),
    ("block32_bsnd", 96, (32, 32), "BSND", 1, 1),
    ("block16_gqa", 64, (16, 16), "BNSD", 4, 2),
    ("block32_gqa", 96, (32, 32), "BNSD", 4, 2),
)


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected one occurrence of {old!r}, got {source.count(old)}")
    return source.replace(old, new, 1)


def materialize_case(
    source: str,
    case: tuple[str, int, tuple[int, int], str, int, int],
    device: int,
) -> str:
    name, seq_len, block_shape, layout, q_heads, kv_heads = case
    transformed = replace_once(
        source,
        "    block_shape = [128, 128]",
        f"    block_shape = [{block_shape[0]}, {block_shape[1]}]",
    )
    transformed = replace_once(
        transformed,
        "    dst = np.exp(src - lse)",
        "    dst = torch.exp(src - lse)",
    )
    transformed = replace_once(
        transformed, "    torch.npu.set_device(2)", f"    torch.npu.set_device({device})"
    )
    transformed = replace_once(
        transformed, '        layout="BNSD",', f'        layout="{layout}",'
    )
    transformed = replace_once(
        transformed, "        q_head_num = 1,", f"        q_head_num = {q_heads},"
    )
    transformed = replace_once(
        transformed, "        kv_head_num = 1,", f"        kv_head_num = {kv_heads},"
    )
    transformed = replace_once(
        transformed, "        max_seqlen_q = 404,", f"        max_seqlen_q = {seq_len},"
    )
    transformed = replace_once(
        transformed, "        max_seqlen_k = 404,", f"        max_seqlen_k = {seq_len},"
    )
    transformed = replace_once(
        transformed, '        case_name = f"bsag_test",', f'        case_name = "{name}",'
    )
    return transformed


def prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    """Prepend a path once while preserving an explicitly configured environment."""
    path_text = str(path.resolve())
    entries = [entry for entry in env.get(key, "").split(os.pathsep) if entry]
    env[key] = os.pathsep.join([path_text] + [entry for entry in entries if entry != path_text])


def configure_opp_env(env: dict[str, str], opp_root: Path) -> None:
    """Expose the locally installed custom OPP to each fresh test subprocess."""
    prepend_env_path(env, "ASCEND_CUSTOM_OPP_PATH", opp_root)
    prepend_env_path(env, "LD_LIBRARY_PATH", opp_root / "op_api/lib")


def run_case(
    reference: Path,
    case: tuple[str, int, tuple[int, int], str, int, int],
    device: int,
    timeout: int,
    opp_root: Path,
) -> bool:
    name = case[0]
    source = reference.read_text(encoding="utf-8")
    script = materialize_case(source, case, device)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{name}.py", encoding="utf-8", delete=False
    ) as handle:
        handle.write(script)
        script_path = Path(handle.name)

    env = os.environ.copy()
    configure_opp_env(env, opp_root)
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(reference.parent),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        print(f"[FAIL] {name}: timed out after {timeout}s")
        if error.stdout:
            print(error.stdout[-4000:])
        return False
    finally:
        script_path.unlink(missing_ok=True)

    success_count = result.stdout.count("success")
    no_nan = "nan exist: dq=False, dk=False, dv=False" in result.stdout
    passed = result.returncode == 0 and success_count == 3 and no_nan
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: gradients={success_count}/3, returncode={result.returncode}")
    if not passed:
        print(result.stdout[-8000:])
    return passed


def validate_cases(
    reference: Path,
    cases: tuple[tuple[str, int, tuple[int, int], str, int, int], ...],
    device: int,
) -> bool:
    """Validate every source transformation without importing torch or using an NPU."""
    source = reference.read_text(encoding="utf-8")
    for case in cases:
        name = case[0]
        try:
            script = materialize_case(source, case, device)
            compile(script, f"<{name}>", "exec")
        except (SyntaxError, ValueError) as error:
            print(f"[FAIL] {name}: {error}")
            return False
        print(f"[PASS] {name}: generated source is valid")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("/home/dhn/test_bsa.py"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--opp-root",
        type=Path,
        default=DEFAULT_OPP_ROOT,
        help="custom vendor root to prepend to ASCEND_CUSTOM_OPP_PATH",
    )
    parser.add_argument("--full", action="store_true", help="include BSND and GQA cases")
    parser.add_argument("--case", action="append", help="run only the named case; repeatable")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate generated scripts without importing torch or using an NPU",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.reference.is_file():
        print(f"reference script not found: {args.reference}", file=sys.stderr)
        return 2
    if not args.dry_run and not args.opp_root.is_dir():
        print(
            f"custom OPP not found: {args.opp_root}; build/install the package or pass --opp-root",
            file=sys.stderr,
        )
        return 2

    cases = FULL_CASES if args.full else QUICK_CASES
    if args.case:
        selected = set(args.case)
        cases = tuple(case for case in cases if case[0] in selected)
        missing = selected - {case[0] for case in cases}
        if missing:
            print(f"unknown cases: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    if args.dry_run:
        passed = validate_cases(args.reference, cases, args.device)
        print(f"summary: {len(cases) if passed else 0}/{len(cases)} cases generated")
        return 0 if passed else 1

    print(f"using custom OPP: {args.opp_root.resolve()}")
    passed = sum(
        run_case(args.reference, case, args.device, args.timeout, args.opp_root)
        for case in cases
    )
    print(f"summary: {passed}/{len(cases)} cases passed")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
