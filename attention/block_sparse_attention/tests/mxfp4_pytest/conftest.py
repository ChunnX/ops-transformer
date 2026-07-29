def pytest_addoption(parser):
    parser.addoption("--b", default=None, type=int, help="batch size")
    parser.addoption("--n1", default=None, type=int, help="query heads")
    parser.addoption("--n2", default=None, type=int, help="kv heads")
    parser.addoption("--s1", default=None, type=int, help="q seq len")
    parser.addoption("--s2", default=None, type=int, help="kv seq len")
    parser.addoption("--qk_d", default=None, type=int, help="head dim")


def pytest_generate_tests(metafunc):
    from test_bsa_mxfp4_golden import TestBSAMxfp4Golden
    cls = metafunc.cls
    if cls is not TestBSAMxfp4Golden:
        return
    # 如果传了 CLI 参数，就覆盖对应测试方法的 cfg
    overrides = {}
    for key in ("b", "n1", "n2", "s1", "s2", "qk_d"):
        val = metafunc.config.getoption(f"--{key}")
        if val is not None:
            overrides[key] = val
    if overrides:
        # 给测试类注入一个类变量，_run 里读取
        metafunc.cls._cfg_overrides = overrides
