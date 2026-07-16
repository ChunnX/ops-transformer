import logging

import pytest
import _pytest.terminal

from test import _parse_stages, _collect_cases, _make_reporter, _setup_hookwrapper

_setup_hookwrapper(_pytest.terminal)


def pytest_addoption(parser):
    parser.addoption("--stages", default="all",
                     help="执行阶段, 逗号组合: all/gen/cpu/npu/gpu/compare/npuprof/npuprofop/gpuprof")
    parser.addoption("--case_id", default="all", help="用例ID过滤, 逗号组合")
    parser.addoption("--case_set", default="ci", help="用例集marker: ci/debug/func_rdv/perf_rdv")
    parser.addoption("--npu_mode", default="eager", help="NPU执行模式: eager(直调,默认)/graph(aclgraph)")
    parser.addoption("--verbose_diff", action="store_true", help="精度不达标时打印全部失败元素")
    parser.addoption("--report_interval", default=20, type=int, help="中间报告间隔")
    parser.addoption("--chunk_size", default=0, type=int, help="比对分块大小(0=不分块)")


def pytest_generate_tests(metafunc):
    if "params" not in metafunc.fixturenames:
        return
    case_set = metafunc.config.getoption("--case_set")
    case_id_opt = metafunc.config.getoption("--case_id")
    cases, case_ids = _collect_cases(case_set, case_id_opt)
    metafunc.parametrize("params", cases, ids=case_ids)


@pytest.fixture(scope="session", autouse=True)
def _configure_logging(request):
    level = request.config.getoption("--log-cli-level", default="INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


@pytest.fixture(scope="session")
def stages(request):
    return _parse_stages(request.config.getoption("--stages"))


@pytest.fixture(scope="session")
def reporter(request):
    rep = _make_reporter()
    yield rep
    rep.print_final()
