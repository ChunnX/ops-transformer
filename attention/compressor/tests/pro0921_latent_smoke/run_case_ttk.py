# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# One isolated CAMODEL case for the formal Pro0921 operators. Deliberately NOT
# a copy of the historical v5 run.py: no sha256 provenance gate, and both the
# TTK process/run timeouts are bounded (the caller also wraps it in
# `timeout 180`, so the hard per-case wall clock stays 180 s).
#
# The process working directory is forced to the case directory: TTK resolves
# several relative paths (kernel_meta, dumps, msprof) against the cwd, and the
# merged kernel_meta must live inside the case directory rather than the shared
# run root.
#
# Required environment:
#   PRO0921_CASE_DIR     case directory (case.csv + .npy fixtures + golden.json)
#   PRO0921_CASE_NAME    the single enabled testcase name
#   PRO0921_PLUGIN_FILE  absolute path to plugin.py
# Optional:
#   PRO0921_CASE_TIMEOUT_S   hard per-case budget in seconds (default 170)
#   PRO0921_READBACK_XCACHE  when "1", copy the inputs listed in
#                            <case>/readback_spec.json back from device memory
#                            into <case>/readback/*.bin.  This is a local
#                            harness hook (it only patches the TTK class method
#                            in this process); it cannot observe a mutated input
#                            through the standard result CSV, which only carries
#                            outputs.

import json
import logging
import os
from pathlib import Path


os.environ["TTK_PARENT_PID"] = str(os.getpid())

from ttk._env import setup_env  # noqa: E402

setup_env()

from ttk.cli.bridge import run_with_switches  # noqa: E402
from ttk.core_modules.tbe_multiprocessing import SimpleCommandProcess  # noqa: E402
from ttk.utilities.classes import MODE, SWITCHES  # noqa: E402


HARD_TIMEOUT_S = int(os.environ.get("PRO0921_CASE_TIMEOUT_S", "170"))

# An explicit compiler resource limit keeps smoke tiling small. The model still
# has 32/64 physical cores; kernels use the legal subset selected by host tiling.
if os.environ.get("PRO0921_CORE_NUM"):
    from ttk.core_modules.operator.tbe_interface import TbeOpc, AscOpc
    _smoke_core_num = int(os.environ["PRO0921_CORE_NUM"])
    if not 1 <= _smoke_core_num <= 32:
        raise ValueError("PRO0921_CORE_NUM must be in [1, 32]")

    def _set_smoke_resources(platform, soc_version, core_type):
        platform.set_current_compile_soc_info(soc_version, core_type, aicore_num=_smoke_core_num)
        platform.set_platform_info_res(0, {
            "cube_core_cnt": str(_smoke_core_num),
            "ai_core_cnt": str(_smoke_core_num),
            "vector_core_cnt": str(2 * _smoke_core_num),
        })
        platform.set_core_num_by_core_type(core_type)

    def _tbe_smoke_soc(self, soc_version, core_type="AiCore"):
        _set_smoke_resources(self._tbe.common.platform.platform_info, soc_version, core_type)

    def _asc_smoke_soc(self, soc_version, core_type="AiCore"):
        _set_smoke_resources(self._asc.common.platform.platform_info, soc_version, core_type)

    TbeOpc.set_compile_soc_info = _tbe_smoke_soc
    AscOpc.set_compile_soc_info = _asc_smoke_soc


# Use the compiler's explicit template-key filter for bounded local smoke builds.
if os.environ.get("PRO0921_TILING_KEYS"):
    from ttk.core_modules.operator.op_interface import OperatorInterface
    _original_compile_context = OperatorInterface.set_dynamic_compile_context
    _selected_keys = os.environ["PRO0921_TILING_KEYS"].split(",")

    def _smoke_compile_context(self, cxt, testcase, operator_func, kernel_name, dyn_params):
        _original_compile_context(self, cxt, testcase, operator_func, kernel_name, dyn_params)
        cxt.add_addition("tiling_key", _selected_keys)

    OperatorInterface.set_dynamic_compile_context = _smoke_compile_context



def main():
    case_dir = Path(os.environ["PRO0921_CASE_DIR"]).resolve()
    os.chdir(case_dir)
    dump_dir = case_dir / "dump"
    dump_dir.mkdir(exist_ok=True)
    os.environ["NPU_DUMP_PATH"] = str(dump_dir)
    (case_dir / "native_execution.json").unlink(missing_ok=True)
    readback_spec = case_dir / "readback_spec.json"
    if readback_spec.exists():
        for entry in json.loads(readback_spec.read_text()):
            for suffix in (".bin", ".meta.json"):
                (case_dir / "readback" / (entry["name"] + suffix)).unlink(missing_ok=True)

    switches = SWITCHES()
    switches.kernel_meta = str(case_dir / "kernel_meta")
    print("PRO0921_KERNEL_META=" + switches.kernel_meta, flush=True)
    print("PRO0921_CWD=" + os.getcwd(), flush=True)
    switches.logging_to_file = True
    switches.input_files = [str(case_dir / "case.csv")]
    switches.output_file_name = str(case_dir / "result.csv")
    switches.selected_testcases = [os.environ["PRO0921_CASE_NAME"]]
    switches.plugin_path = (Path(os.environ["PRO0921_PLUGIN_FILE"]),)
    if os.environ.get("PRO0921_REUSE_COMPILE") == "1":
        switches.dyn_switches.realtime = False
    if os.environ.get("PRO0921_COMPILE_ONLY") == "1":
        switches.dyn_switches.prof = False
    switches.mode = MODE.ASCEND_CAMODEL
    switches.test_mode = "op"
    switches.dev_plat = "Ascend950PR_9589"
    switches.device_count = 1
    switches.process_per_device = 1
    switches.no_memory_check = True
    switches.run_time = 1
    switches.warmup = False          # the tiny cases are single-shot
    switches.proc_timeout = HARD_TIMEOUT_S
    switches.run_timeout = HARD_TIMEOUT_S
    switches.random_seed = 20260927
    switches.golden_mode = "Enable"
    switches.dump_config.dump_on_fail = True
    switches.dump_config.enable_all()
    switches.dump_config.file_format = "npy"
    switches.single_testcase_log_mode = True
    try:
        run_with_switches(switches)
    finally:
        for process in SimpleCommandProcess.all_processes:
            process.close()


if __name__ == "__main__":
    main()
