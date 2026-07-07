import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pytest

from core.case_loader import load_cases, resolve_case_ids
from core.orchestrator import run_case
from core.reporter import Reporter, make_failure_result

_VALID_STAGES = {"all", "gen", "cpu", "npu", "gpu", "compare",
                 "npuprof", "npuprofop", "gpuprof", "l2_flush"}


def _parse_stages(raw):
    parts = {s.strip() for s in raw.split(",") if s.strip()}
    invalid = parts - _VALID_STAGES
    if invalid:
        raise ValueError(f"Invalid stages: {invalid}. Valid: {_VALID_STAGES}")
    if "all" in parts:
        return {"gen", "cpu", "npu", "compare"}
    return parts


def _collect_cases(op_name, case_set, case_id_opt):
    cases, case_ids = load_cases(op_name, case_set)
    if case_id_opt != "all":
        filtered = resolve_case_ids(case_id_opt, case_ids)
        idx_map = {cid: i for i, cid in enumerate(case_ids)}
        selected = [idx_map[cid] for cid in filtered if cid in idx_map]
        cases = [cases[i] for i in selected]
        case_ids = [cid for cid in filtered if cid in idx_map]
    return cases, case_ids


def _make_reporter():
    rep = Reporter()
    result_dir = Path("result")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep.set_csv(result_dir / f"result_{ts}.csv")
    return rep


def _setup_hookwrapper(_pytest_terminal):
    _original_get_line = _pytest_terminal._get_line_with_reprcrash_message
    _fail_msgs = {}

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(item, call):
        outcome = yield
        rep = outcome.get_result()
        if rep.when == "call" and rep.failed:
            msg = None
            try:
                if isinstance(rep.longrepr, str):
                    msg = rep.longrepr
                else:
                    msg = rep.longrepr.reprcrash.message
            except AttributeError:
                pass
            if msg and msg.startswith("Failed: "):
                msg = msg[len("Failed: "):]
            _fail_msgs[rep.nodeid] = msg

    def _custom_get_line(config, rep, tw, word_markup):
        if hasattr(rep, 'when') and rep.when == "call" and rep.failed:
            fail_msg = _fail_msgs.get(rep.nodeid)
            if fail_msg:
                verbose_word, verbose_markup = rep._get_verbose_word_with_markup(
                    config, word_markup
                )
                word = tw.markup(verbose_word, **verbose_markup)
                return f"{word} {fail_msg}"
        return _original_get_line(config, rep, tw, word_markup)

    _pytest_terminal._get_line_with_reprcrash_message = _custom_get_line


def test_op(params, stages, reporter, op_name, request):
    result = run_case(stages, op_name, params, reporter,
                      max_retries=request.config.getoption("--max-retries"),
                      load_pt=request.config.getoption("--load-pt"),
                      graph_path=request.config.getoption("--graph-path"),
                      chunk_size=request.config.getoption("--chunk-size"),
                      num_workers=request.config.getoption("--num-workers"),
                      verbose_diff=request.config.getoption("--verbose-diff"),
                      fail_analysis=request.config.getoption("--fail-analysis"),
                      keep_pt=request.config.getoption("--keep-pt"))
    if not result.get("passed", False):
        pytest.fail(result.get("fail_msg", "精度校验失败"))


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
    cases, case_ids = _collect_cases(args.op, args.case_set, args.case_id)

    if not cases:
        print("[ERROR] 没有可运行的 case")
        sys.exit(1)

    reporter = _make_reporter()

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
