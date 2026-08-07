def pytest_addoption(parser):
    parser.addoption("--b", default=None, type=int, help="batch size")
    parser.addoption("--n1", default=None, type=int, help="query heads")
    parser.addoption("--n2", default=None, type=int, help="kv heads")
    parser.addoption("--s1", default=None, type=int, help="q seq len")
    parser.addoption("--s2", default=None, type=int, help="kv seq len")
    parser.addoption("--qk_d", default=None, type=int, help="head dim")
    # 注意: type=bool 对字符串 "False" 返回 True(Python 坑), 改用 str + 显式解析
    parser.addoption("--dump", default=None, type=str, help="dump on (True/False)")
    # 量化模式: 2=OCP, 3=CX(对齐 host quantMode); None=用测试用例自带值
    parser.addoption("--quant_mode", default=None, type=int, help="quant mode: 2=OCP, 3=CX")
    # 输出 dtype: bfloat16|float16; None=用测试用例自带值
    parser.addoption("--out_dtype", default=None, type=str, help="output dtype: bfloat16|float16")
    # CX dstTypeMax: 0=默认6.0(log2(6)); 仅 quant_mode=3 生效; None=用测试用例自带值
    parser.addoption("--dst_type_max", default=None, type=float, help="CX dstTypeMax (0=default 6.0)")

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
    # 量化模式(2=OCP/3=CX): 覆盖测试用例的 quant_mode
    quant_mode_val = config.getoption("--quant_mode")
    if quant_mode_val is not None:
        overrides["quant_mode"] = quant_mode_val
    # 输出 dtype(bfloat16|float16): 覆盖测试用例的 out_dtype
    out_dtype_val = config.getoption("--out_dtype")
    if out_dtype_val is not None:
        overrides["out_dtype"] = out_dtype_val
    # CX dstTypeMax(0=默认6.0): 覆盖测试用例的 dst_type_max
    dst_type_max_val = config.getoption("--dst_type_max")
    if dst_type_max_val is not None:
        overrides["dst_type_max"] = dst_type_max_val
    mod._CLI_OVERRIDES = overrides

def pytest_runtest_logreport(report):
    """每个测试的 report 阶段触发; 在 passed 的 call 阶段后打印分隔线,
    使其出现在 pytest 打印的 'PASSED' 之后。
    用 sys.stdout.write + flush 显式补换行, 避免 pytest 进度行 \\r 导致 PASSED 与分隔线黏连。"""
    import sys
    if report.when == "call" and report.passed:
        sys.stdout.write("\n=========================================================\n")
        sys.stdout.flush()
