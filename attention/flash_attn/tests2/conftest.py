import pytest
import _pytest.terminal
from pathlib import Path

from test import _parse_stages, _collect_cases, _make_reporter


def pytest_addoption(parser):
    parser.addoption("--op", default="flash_attn", help="算子名称")
    parser.addoption("--stages", default="all",
                     help="执行阶段, 逗号组合: all/gen/cpu/npu/gpu/compare/npuprof/npuprofop/gpuprof")
    parser.addoption("--case-id", default="all", help="用例ID过滤, 逗号组合")
    parser.addoption("--case-set", default="ci", help="用例集marker: ci/debug/func_rdv/perf_rdv")
    parser.addoption("--graph-path", default=0, type=int, help="图模式路径(0=非图模式)")
    parser.addoption("--verbose-diff", action="store_true", help="精度不达标时打印全部失败元素")
    parser.addoption("--fail-analysis", action="store_true", default=True,
                     help="精度不达标时输出失败分析报告")
    parser.addoption("--keep-pt", action="store_true", default=True, help="保留中间pt产物")
    parser.addoption("--report-interval", default=20, type=int, help="中间报告间隔")
    parser.addoption("--load-pt", default=None, help="从pt加载inputs替代gen_inputs")
    parser.addoption("--chunk-size", default=0, type=int, help="比对分块大小(0=不分块)")
    parser.addoption("--num-workers", default=0, type=int, help="比对并行worker数(0=单进程)")
    parser.addoption("--max-retries", default=0, type=int, help="NPU计算最大重试次数")


def pytest_generate_tests(metafunc):
    if "params" not in metafunc.fixturenames:
        return
    op_name = metafunc.config.getoption("--op")
    case_set = metafunc.config.getoption("--case-set")
    case_id_opt = metafunc.config.getoption("--case-id")
    cases, case_ids = _collect_cases(op_name, case_set, case_id_opt)
    metafunc.parametrize("params", cases, ids=case_ids)


@pytest.fixture(scope="session")
def op_name(request):
    return request.config.getoption("--op")


@pytest.fixture(scope="session")
def stages(request):
    return _parse_stages(request.config.getoption("--stages"))


@pytest.fixture(scope="session")
def reporter(request):
    rep = _make_reporter()
    yield rep
    rep.print_final()


_original_get_line = _pytest.terminal._get_line_with_reprcrash_message
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


_pytest.terminal._get_line_with_reprcrash_message = _custom_get_line
