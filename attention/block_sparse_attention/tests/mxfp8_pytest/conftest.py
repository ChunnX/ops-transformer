def pytest_addoption(parser):
    parser.addoption("--b", default=None, type=int, help="batch size")
    parser.addoption("--n1", default=None, type=int, help="query heads")
    parser.addoption("--n2", default=None, type=int, help="kv heads")
    parser.addoption("--s1", default=None, type=int, help="q seq len")
    parser.addoption("--s2", default=None, type=int, help="kv seq len")
    parser.addoption("--qk_d", default=None, type=int, help="head dim")
    parser.addoption("--dump", default=None, type=str, help="dump on (True/False)")
    parser.addoption(
        "--quant_mode", default=None, type=int, help="quant mode: 4=MXFP8 OCP"
    )
    parser.addoption(
        "--out_dtype", default=None, type=str, help="output dtype: bfloat16|float16"
    )
    parser.addoption(
        "--p_quant", default=None, type=str, help="P quant: snap_local|none"
    )


def _parse_bool(val):
    if val is None:
        return None
    return str(val).strip().lower() in ("true", "1", "yes", "y")


_CUR_NODEID = {"v": ""}


def pytest_runtest_setup(item):
    _CUR_NODEID["v"] = item.nodeid


def pytest_configure(config):
    import json
    import os
    import test_bsa_mxfp8_golden as mod

    fulfill_log = os.environ.get("BSA_FULFILL_JSONL", "").strip()
    if fulfill_log:
        orig = mod.check_result

        def _check_result_logged(*args, **kwargs):
            pct, res = orig(*args, **kwargs)
            rec = {"nodeid": _CUR_NODEID["v"], "fulfill": float(pct), "result": res}
            with open(fulfill_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return pct, res

        mod.check_result = _check_result_logged
    overrides = {}
    for key in ("b", "n1", "n2", "s1", "s2", "qk_d"):
        val = config.getoption(f"--{key}")
        if val is not None:
            overrides[key] = val
    dump_val = _parse_bool(config.getoption("--dump"))
    if dump_val is not None:
        overrides["dump"] = dump_val
    quant_mode_val = config.getoption("--quant_mode")
    if quant_mode_val is not None:
        overrides["quant_mode"] = quant_mode_val
    out_dtype_val = config.getoption("--out_dtype")
    if out_dtype_val is not None:
        overrides["out_dtype"] = out_dtype_val
    p_quant_val = config.getoption("--p_quant")
    if p_quant_val is not None:
        overrides["p_quant"] = p_quant_val
    mod._CLI_OVERRIDES = overrides


def pytest_runtest_logreport(report):
    import sys

    if report.when == "call" and report.passed:
        sys.stdout.write(
            "\n=========================================================\n"
        )
        sys.stdout.flush()
