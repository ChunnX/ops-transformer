import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

from core.case_loader import load_cases, resolve_case_ids
from core.orchestrator import run_case
from core.reporter import Reporter, make_failure_result

_VALID_STAGES = {"all", "gen", "cpu", "npu", "gpu", "compare", "npuprof", "npuprofop", "gpuprof"}


def _parse_stages(raw):
    parts = {s.strip() for s in raw.split(",") if s.strip()}
    invalid = parts - _VALID_STAGES
    if invalid:
        print(f"[ERROR] Invalid stages: {invalid}. Valid: {_VALID_STAGES}")
        sys.exit(1)
    if "all" in parts:
        return {"gen", "cpu", "npu", "compare"}
    return parts


def main():
    parser = argparse.ArgumentParser(description="精度测试 CLI 入口")
    parser.add_argument("--op", default="flash_attn", help="算子名称")
    parser.add_argument("--stages", default="all",
                        help="执行阶段: all/gen/cpu/npu/gpu/compare/npuprof/npuprofop/gpuprof")
    parser.add_argument("--case-id", default="all", help="用例ID过滤")
    parser.add_argument("--case-set", default="ci", help="用例集marker: ci/debug/func_rdv/perf_rdv")
    parser.add_argument("--graph-path", default=0, type=int, help="图模式路径")
    parser.add_argument("--verbose-diff", action="store_true", help="打印全部失败元素")
    parser.add_argument("--fail-analysis", action="store_true", default=True, help="失败分析报告")
    parser.add_argument("--keep-pt", action="store_true", default=True, help="保留中间pt产物")
    parser.add_argument("--report-interval", default=20, type=int, help="中间报告间隔")
    parser.add_argument("--load-pt", default=None, help="从pt加载inputs")
    parser.add_argument("--chunk-size", default=0, type=int, help="比对分块大小")
    parser.add_argument("--num-workers", default=0, type=int, help="比对并行worker数")
    parser.add_argument("--max-retries", default=0, type=int, help="NPU最大重试次数")
    args = parser.parse_args()

    stages = _parse_stages(args.stages)
    cases, case_ids = load_cases(args.op, args.case_set)

    if args.case_id != "all":
        filtered = resolve_case_ids(args.case_id, case_ids)
        idx_map = {cid: i for i, cid in enumerate(case_ids)}
        selected = [idx_map[cid] for cid in filtered if cid in idx_map]
        cases = [cases[i] for i in selected]
        case_ids = filtered

    if not cases:
        print("[ERROR] 没有可运行的 case")
        sys.exit(1)

    reporter = Reporter()
    csv_dir = Path("reports")
    csv_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reporter.set_csv(csv_dir / f"result_{ts}.csv")

    for i, (params, case_id) in enumerate(zip(cases, case_ids)):
        print(f"\n{'='*66}")
        print(f"  Case: {case_id}")
        print(f"{'='*66}")
        try:
            run_case(stages, args.op, params, reporter,
                     max_retries=args.max_retries,
                     load_pt=args.load_pt,
                     graph_path=args.graph_path,
                     chunk_size=args.chunk_size,
                     num_workers=args.num_workers,
                     verbose_diff=args.verbose_diff,
                     fail_analysis=args.fail_analysis,
                     keep_pt=args.keep_pt)
        except Exception as e:
            err = f"{e}\n{traceback.format_exc()}"
            print(f"[ERROR] {case_id} 运行异常: {e}")
            fail = make_failure_result(str(e))
            fail["passed"] = False
            fail["fail_msg"] = str(e)
            reporter.record(case_id, fail, error=err, dtype=params.get("dtype", ""))

        if args.report_interval > 0 and (i + 1) % args.report_interval == 0 \
                and (i + 1) < len(cases):
            reporter.print_interim(len(cases))

    fail_cnt = reporter.print_final()
    sys.exit(0 if fail_cnt == 0 else 1)


if __name__ == "__main__":
    main()
