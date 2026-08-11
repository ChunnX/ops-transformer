#!/usr/bin/env python3
"""从Excel读取测试用例，通过AST源码替换运行 attention2ffn_server.py。

用法:
    python3 run_excel_cases.py                # 运行全部用例
    python3 run_excel_cases.py --list         # 列出用例
    python3 run_excel_cases.py --dry-run      # 仅验证源码生成
    python3 run_excel_cases.py --case-index 0,2
"""
import ast, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_SCRIPT = SCRIPT_DIR / "attention2ffn_server.py"
EXCEL_PATH = SCRIPT_DIR.parent / "test_cases.xlsx"

# Excel列名 → attention2ffn_server.py 顶层变量名
COL_MAP = {
    'case_name': 'current_case_name', 'x': 'X', 'bs': 'bs', 'h': 'h', 'k': 'k',
    'x_dtype': 'out_dtype', 'graph_type': 'graph_type', 'quant_mode': 'quant_mode',
    'is_mask': 'is_mask', 'sync_flag': 'sync_flag', 'is_scale': 'is_attn2ffn_scales',
    'moeExpertPerRank': 'moe_expert_per_rank', 'moeRankFFNNum': 'moe_rank_ffn_num',
    'shared_expert_num': 'shared_expert_num', 'shareExpertRankNum': 'rank_num_per_shared_expert',
    'is_performance': 'is_performance', 'is_fullcore': 'is_full_core', 'corenum': 'core_num',
}

COMMON = {"server_num": 1, "is_check_setting": True, "is_topk_balance": True}


def read_excel(path):
    """读取Excel，返回 [(case_name, overrides), ...]。"""
    df = pd.read_excel(path)
    cases = []
    for idx, row in df.iterrows():
        ov = dict(COMMON)
        for col, var in COL_MAP.items():
            if col in df.columns and pd.notna(row[col]):
                ov[var] = row[col] if isinstance(row[col], str) else int(row[col])
        # FFNWorkerNum + attentionWorkerNum → rank_per_dev (world_size)
        ffn = int(row['FFNWorkerNum']) if 'FFNWorkerNum' in df.columns and pd.notna(row.get('FFNWorkerNum')) else 0
        attn = int(row['attentionWorkerNum']) if 'attentionWorkerNum' in df.columns and pd.notna(row.get('attentionWorkerNum')) else 0
        if ffn and attn:
            ov['rank_per_dev'] = ffn + attn
        cases.append((ov.get('current_case_name', f'case_{idx}'), ov))
    return cases


def make_source(source, overrides):
    """用AST替换 attention2ffn_server.py 顶层赋值，返回修改后的源码字符串。"""
    tree = ast.parse(source)
    assigns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assigns[node.targets[0].id] = node

    missing = set(overrides) - set(assigns)
    if missing:
        raise ValueError(f"未找到顶层变量: {missing}")

    lines = source.splitlines(keepends=True)
    reps = []
    for name, val in overrides.items():
        node = assigns[name]
        end = getattr(node, "end_lineno", node.lineno)
        reps.append((node.lineno - 1, end, f"{name} = {val!r}\n"))
    for start, end, rep in sorted(reps, reverse=True):
        lines[start:end] = [rep]
    return "".join(lines)


def run_case(source, name, overrides, log_dir, dry_run=False, timeout=900):
    """生成临时脚本并以子进程运行单个用例。"""
    case_src = make_source(source, overrides)
    if dry_run:
        print(f"[DRY-RUN] {name}: OK")
        return "DRY-RUN"

    log_path = log_dir / f"{name}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(SCRIPT_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    with tempfile.TemporaryDirectory(prefix=f"attn2ffn_{name}_") as td:
        script = Path(td) / f"{name}.py"
        script.write_text(case_src, encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as log:
            try:
                proc = subprocess.Popen([sys.executable, "-u", str(script)], cwd=SCRIPT_DIR, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=(os.name != "nt"))
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = -1

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    markers = re.findall(r"attn2ffn test (Success|Failed)\.", log_text)
    passed = rc == 0 and markers and markers[-1] == "Success"
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}, log: {log_path}")
    if not passed:
        print("\n".join(log_text.splitlines()[-30:]))
    return status


def main():
    import argparse
    p = argparse.ArgumentParser(description="从Excel读取用例运行 attention2ffn_server.py")
    p.add_argument("--excel", type=Path, default=EXCEL_PATH)
    p.add_argument("--case-index", default="", help="用例索引(逗号分隔, 从0开始)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()

    cases = read_excel(args.excel)
    print(f"读取到 {len(cases)} 条用例")

    if args.case_index:
        idxs = [int(i) for i in args.case_index.split(",")]
        cases = [cases[i] for i in idxs]

    if args.list:
        for i, (name, ov) in enumerate(cases):
            print(f"[{i}] {name}: bs={ov.get('bs')}, h={ov.get('h')}, k={ov.get('k')}, rank_per_dev={ov.get('rank_per_dev')}")
        return

    source = TEST_SCRIPT.read_text(encoding="utf-8")
    log_dir = SCRIPT_DIR / "excel_logs" / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (name, ov) in enumerate(cases, 1):
        print(f"\n===== [{i}/{len(cases)}] {name} =====")
        try:
            status = run_case(source, name, ov, log_dir, args.dry_run, args.timeout)
        except Exception as e:
            status = "FAIL"
            print(f"[FAIL] {name}: {e}")
        results.append({"case": name, "status": status})
        if args.stop_on_failure and status not in ("PASS", "DRY-RUN"):
            break

    (log_dir / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(r["status"] == "PASS" for r in results)
    print(f"\nSummary: {passed}/{len(results)} passed, log dir: {log_dir}")


if __name__ == "__main__":
    main()
