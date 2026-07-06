import pytest

from core.orchestrator import run_case


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
