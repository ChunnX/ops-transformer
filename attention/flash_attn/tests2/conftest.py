import logging

import pytest
import _pytest.terminal

from test import _parse_stages, _collect_cases, _make_reporter, _setup_hookwrapper, _register_cli_params

_setup_hookwrapper(_pytest.terminal)


def pytest_addoption(parser):
    _register_cli_params(parser.addoption)


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
