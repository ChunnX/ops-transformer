# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _configure(tmp_path, sanitizer, producer_barrier, backend="udma"):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required to evaluate the production build options")
    source = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    # Execute the actual build-option section without requiring CANN or a C++
    # compiler. This exercises CMake's option and conditional evaluation.
    begin = source.index("option(CATCCOS_UDMA_MSSANITIZER")
    end = source.index("if(NOT NPU_SOC_VERSION", begin)
    script = tmp_path / "options.cmake"
    script.write_text(
        source[begin:end]
        + '\nfile(WRITE "${CMAKE_CURRENT_LIST_DIR}/flags.txt" "${CATCCOS_UDMA_ASCEND_FLAGS}")\n',
        encoding="utf-8",
    )
    return subprocess.run(
        [cmake, "-DCATCCOS_UDMA_MSSANITIZER=" + sanitizer,
         "-DCATCCOS_BACKEND=" + backend,
         "-DCATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER=" + producer_barrier,
         "-P", str(script)],
        capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("sanitizer", ["0", "1"])
def test_production_build_options(tmp_path, sanitizer):
    result = _configure(tmp_path, sanitizer, "0")
    assert result.returncode == 0, result.stdout + result.stderr
    flags = (tmp_path / "flags.txt").read_text().split()
    assert set(flags) == {
        "-DCATCCOS_UDMA_MSSANITIZER=" + sanitizer,
        "-DASCENDC_DUMP=" + sanitizer,
    } | ({"-g"} if sanitizer == "1" else set())


@pytest.mark.parametrize("sanitizer", ["0", "1"])
def test_removed_producer_barrier_cannot_be_reenabled(tmp_path, sanitizer):
    result = _configure(tmp_path, sanitizer, "1")
    assert result.returncode != 0
    assert "Only CATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER=0 is supported" in result.stderr


def test_mte_build_does_not_apply_legacy_udma_producer_option(tmp_path):
    result = _configure(tmp_path, "0", "1", backend="mte")
    assert result.returncode == 0, result.stdout + result.stderr
