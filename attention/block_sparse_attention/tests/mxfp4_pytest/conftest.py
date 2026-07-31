def pytest_addoption(parser):
    parser.addoption("--b", default=None, type=int, help="batch size")
    parser.addoption("--n1", default=None, type=int, help="query heads")
    parser.addoption("--n2", default=None, type=int, help="kv heads")
    parser.addoption("--s1", default=None, type=int, help="q seq len")
    parser.addoption("--s2", default=None, type=int, help="kv seq len")
    parser.addoption("--qk_d", default=None, type=int, help="head dim")
    # 注意: type=bool 对字符串 "False" 返回 True(Python 坑), 改用 str + 显式解析
    parser.addoption("--dump", default=None, type=str, help="dump on (True/False)")

def _parse_bool(val):
    if val is None:
        return None
    return str(val).strip().lower() in ("true", "1", "yes", "y")

def pytest_configure(config):
    """pytest 启动时把 CLI 参数写入模块级全局, 供 _run 读取。
    注: pytest_generate_tests 仅对参数化测试触发, 非参数化测试不触发, 故用 pytest_configure。"""
    import test_bsa_mxfp4_golden as mod
    overrides = {}
    for key in ("b", "n1", "n2", "s1", "s2", "qk_d"):
        val = config.getoption(f"--{key}")
        if val is not None:
            overrides[key] = val
    dump_val = _parse_bool(config.getoption("--dump"))
    if dump_val is not None:
        overrides["dump"] = dump_val
    mod._CLI_OVERRIDES = overrides
