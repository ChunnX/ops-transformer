"""pytest 装配层: 注册 CLI 选项、参数化用例、提供 session 级 fixture。

本身不含业务逻辑, 全部委托 test.py 中的共享实现 (_register_cli_params /
_collect_cases 等), 保证 pytest 入口与 test.py CLI 入口行为一致。
"""
import logging

import pytest
import _pytest.terminal

from test import _parse_stages, _collect_cases, _make_reporter, _setup_hookwrapper, _register_cli_params

# 改写 pytest 终端行渲染: 失败用例名后追加 fail_msg (见 test._setup_hookwrapper)
_setup_hookwrapper(_pytest.terminal)


def pytest_addoption(parser):
    """注册 CLI 选项 (与 test.py argparse 共享同一份定义)。"""
    _register_cli_params(parser.addoption)


def pytest_generate_tests(metafunc):
    """按 --case_set/--case_id 展开用例, 注入 params fixture (每 case 一份参数)。"""
    if "params" not in metafunc.fixturenames:
        return
    case_set = metafunc.config.getoption("--case_set")
    case_id_opt = metafunc.config.getoption("--case_id")
    cases, case_ids = _collect_cases(case_set, case_id_opt)
    metafunc.parametrize("params", cases, ids=case_ids)


@pytest.fixture(scope="session", autouse=True)
def _configure_logging(request):
    """session 级日志配置 (autouse, 全局生效一次)。"""
    level = request.config.getoption("--log-cli-level", default="INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


@pytest.fixture(scope="session")
def stages(request):
    """解析 --stages 为阶段集合 (全 session 共享, 不随用例变)。"""
    return _parse_stages(request.config.getoption("--stages"))


@pytest.fixture(scope="session")
def reporter(request):
    """CSV 报告器: 贯穿整个 session 收集结果, 结束时打印汇总表。"""
    rep = _make_reporter()
    yield rep
    rep.print_final()
