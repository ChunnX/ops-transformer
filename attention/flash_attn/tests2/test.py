"""精度测试入口: 同时承载 pytest 用例 (test_op) 与裸 CLI (main) 两个入口。

两个入口共享同一套: CLI 参数定义 (_register_cli_params)、用例收集 (_collect_cases)、
报告器构造 (_make_reporter)、阶段解析 (_parse_stages)。conftest.py 复用这些实现,
保证 pytest 入口与 CLI 入口行为一致。test_op 把 run_case 成败翻译成 pytest 用例成败;
main 自行遍历 case 并以退出码反映失败数。
"""
import argparse
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pytest

from core.case_loader import load_cases, resolve_case_ids
from orchestrator import run_case, Reporter, make_failure_result

_VALID_STAGES = {"all", "gen", "cpu", "npu", "gpu", "compare",
                 "npuprof", "npuprofop", "gpuprof", "l2_flush"}


def _register_cli_params(register):
    """CLI参数唯一定义。register = parser.addoption(conftest) 或 parser.add_argument(argparse)"""

    # ── 执行控制 ──
    register("--stages", default="all",
             help="执行阶段, 逗号组合: all/gen/cpu/npu/gpu/compare/npuprof/npuprofop/gpuprof")

    # ── 用例选择 ──
    register("--case_id", default="all", help="用例ID过滤")
    register("--case_set", default="ci", help="用例集marker: ci/debug/func_rdv/perf_rdv")

    # ── NPU 配置 ──
    register("--npu_mode", default="eager", help="NPU执行模式: eager(直调,默认)/graph(aclgraph)")

    # ── 输出控制 ──
    register("--verbose_diff", nargs="?", const="fold", default=None,
             help="精度失败分析: 不传=≤10项, fold(全量片段), all(全量点位)")
    register("--report_interval", default=20, type=int, help="中间报告间隔")
    register("--chunk_size", default=0, type=int, help="比对分块大小(0=不分块)")
    register("--perf_runs", default=5, type=int, help="性能采集采样轮数")


def _parse_stages(raw):
    parts = {s.strip() for s in raw.split(",") if s.strip()}
    invalid = parts - _VALID_STAGES
    if invalid:
        raise ValueError(f"Invalid stages: {invalid}. Valid: {_VALID_STAGES}")
    if "all" in parts:
        # all = 精度主流程 4 阶段 (不含 gpu/prof, 那些需显式指定)
        return {"gen", "cpu", "npu", "compare"}
    return parts


def _collect_cases(case_set, case_id_opt):
    """加载 case_set 全部用例, 按 --case_id 过滤 (支持前缀/后缀/全名匹配)。"""
    cases, case_ids = load_cases(case_set)
    if case_id_opt != "all":
        filtered = resolve_case_ids(case_id_opt, case_ids)
        idx_map = {cid: i for i, cid in enumerate(case_ids)}
        selected = [idx_map[cid] for cid in filtered if cid in idx_map]
        cases = [cases[i] for i in selected]
        case_ids = [cid for cid in filtered if cid in idx_map]
    return cases, case_ids


def _make_reporter():
    """构造 Reporter 并绑定带时间戳的结果 CSV (每次运行独立文件)。"""
    rep = Reporter()
    result_dir = Path("result")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep.set_csv(result_dir / f"result_{ts}.csv")
    rep.set_perf_csv(result_dir / f"perf_summary_{ts}.csv")
    return rep


def _setup_hookwrapper(_pytest_terminal):
    """monkeypatch pytest 终端行渲染: 让失败用例名后追加 fail_msg (默认只显示 'FAILED')。
    两步: ① hookwrapper 捕获每个失败用例的 fail_msg 存入 _fail_msgs[nodeid];
          ② 替换 _get_line_with_reprcrash_message, 渲染时取出 fail_msg 拼到行尾。"""
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


def test_op(params, stages, reporter, request):
    """pytest 测试入口: 跑一个 case 的精度流程, 失败则 pytest.fail。

    设计: 用例遍历/参数化/退出码交给 pytest; 这里只把 run_case 的成败翻译成
    pytest 用例成败。params/stages/reporter 由 conftest 注入。"""
    result = run_case(stages, params, reporter,
                      npu_mode=request.config.getoption("--npu_mode"),
                      chunk_size=request.config.getoption("--chunk_size"),
                      verbose_diff=request.config.getoption("--verbose_diff"),
                      perf_runs=request.config.getoption("--perf_runs"))
    if not result.get("passed", False):
        pytest.fail(result.get("fail_msg", "精度校验失败"))


def main():
    """裸 CLI 入口 (python test.py): 不依赖 pytest, 自行遍历用例并跑完整流程。

    设计: 与 test_op 互为两个入口, 共享 run_case; 这里自行管用例遍历、
    异常捕获与退出码, 退出码反映失败数。"""
    parser = argparse.ArgumentParser(description="精度测试 CLI 入口")
    _register_cli_params(parser.add_argument)
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="框架日志级别: DEBUG(预留) | INFO(阶段进度/耗时/shape, 默认) | WARNING(key缺失) | ERROR(异常)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(message)s")

    stages = _parse_stages(args.stages)
    cases, case_ids = _collect_cases(args.case_set, args.case_id)

    if not cases:
        print("[ERROR] 没有可运行的 case")
        sys.exit(1)

    reporter = _make_reporter()

    for i, (params, case_id) in enumerate(zip(cases, case_ids)):
        print(f"\n{'='*66}")
        print(f"  Case: {case_id}")
        print(f"{'='*66}")
        try:
            run_case(stages, params, reporter,
                     npu_mode=args.npu_mode,
                     chunk_size=args.chunk_size,
                     verbose_diff=args.verbose_diff,
                     perf_runs=args.perf_runs)
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
