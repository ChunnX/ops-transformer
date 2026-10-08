"""Exercise CMake selection without compiling a Native binary."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("selected,builtin,soc,expected,has_dsl,error", [
    ("flash_kda", True, "ascend950", "", True, False),
    ("flash_kda,other", True, "ascend950", "other", True, False),
    ("other", True, "ascend950", "other", False, False),
    ("ALL", False, "ascend950", "ALL", False, False),
    ("ALL", True, "ascend910b", "ALL", False, False),
    ("flash_kda", False, "ascend950", "", False, True),
    ("flash_kda", True, "ascend910b", "", False, True),
])
def test_selection(tmp_path, selected, builtin, soc, expected, has_dsl, error):
    repo = Path(__file__).parents[3]
    (tmp_path / "cmake").mkdir()
    shutil.copy2(repo / "cmake/cannbotdsl.cmake", tmp_path / "cmake")
    (tmp_path / "cmake/cannbotdsl_aclnn.cmake").write_text("# Generation is checked by the real built-in integration test.\n")
    source = tmp_path / "attention/flash_kda/op_kernel"
    source.mkdir(parents=True)
    (source / "flash_kda.py").write_text('@aclnn("FlashKDA")\ndef entry(): pass\n')
    (tmp_path / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.19)
project(selection NONE)
set(BUILD_OPEN_PROJECT ON)
set(ENABLE_BUILT_IN {'ON' if builtin else 'OFF'})
set(ENABLE_OPS_KERNEL ON)
set(ASCEND_OP_NAME "{selected}")
set(ASCEND_COMPUTE_UNIT "{soc}")
function(should_add_module category result)
  set(${{result}} TRUE PARENT_SCOPE)
endfunction()
include(cmake/cannbotdsl.cmake)
if(NOT "${{ASCENDC_BUILD_OPS}}" STREQUAL "{expected}")
  message(FATAL_ERROR "unexpected AscendC selection: ${{ASCENDC_BUILD_OPS}}")
endif()
if({'NOT ' if has_dsl else ''}TARGET ops_transformer_cannbotdsl)
  message(FATAL_ERROR "unexpected DSL target state")
endif()
""")
    result = subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")],
                            capture_output=True, text=True,
                            env={k: v for k, v in os.environ.items() if k not in {
                                "LD_PRELOAD", "ASCEND_SLOG_PRINT_TO_STDOUT", "ASCEND_GLOBAL_LOG_LEVEL"}})
    if error:
        assert result.returncode and "require --pkg --soc=ascend950" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
