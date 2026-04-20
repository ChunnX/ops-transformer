"""Template rendering and source file preparation — zero torch dependency."""

from __future__ import annotations

import os
import shutil
from typing import Optional

from npukernelbench.config.models import FullConfig
from npukernelbench.models.task import BuildHandle
from npukernelbench.utils.io import safe_read_file, safe_write_file


class TemplateRenderer:
    """Centralized string replacement for all build templates."""

    @staticmethod
    def render_pybind(content: str, op_name: str, kernel_name: str) -> str:
        return content.replace("aclnnCustomOp", f"aclnn{op_name}") \
                      .replace("custom_pybind_api", kernel_name)

    @staticmethod
    def render_cmake_ops(content: str, op_name: str, kernel_name: str) -> str:
        return content.replace("CustomOp", op_name) \
                      .replace("custom_op", kernel_name)

    @staticmethod
    def render_cmake_presets(content: str, ascend_home: str) -> str:
        return content.replace("/usr/local/Ascend/ascend-toolkit/latest", ascend_home)

    @staticmethod
    def patch_kernel_signature(kernel: str) -> str:
        return kernel.replace("GM_ADDR workspace", "GM_ADDR workspace, GM_ADDR tiling")


def prepare_source_files(build: BuildHandle, config: FullConfig, logger=None) -> None:
    """Copy templates + generated/reference code into src/ directory."""
    src = build.paths.src_dir
    renderer = TemplateRenderer()
    os.makedirs(src, exist_ok=True)

    # 1. Copy common build scaffolding
    common_src = os.path.join(config.compile.common_dir, config.compile.mode)
    shutil.copytree(common_src, src, dirs_exist_ok=True)

    # 2. Copy validation files
    val_src = os.path.join(build.metadata.problem_def_path, "validation")
    val_dst = os.path.join(src, "validation")
    shutil.copytree(val_src, val_dst, dirs_exist_ok=True)

    # 3. Render pybind.cpp
    pybind_file = os.path.join(src, "validation", "pybind.cpp")
    if os.path.exists(pybind_file):
        content = safe_read_file(pybind_file) or ""
        rendered = renderer.render_pybind(content, build.metadata.op_name, build.metadata.kernel_name)
        safe_write_file(pybind_file, rendered)

    # 4. Write kernel source files
    if build.gen_content is not None:
        # LLM-generated code path
        _write_generated_sources(build, src, config, renderer)
    else:
        # Reference answer path
        _write_reference_sources(build, src, config, renderer)

    # 5. Patch CMakePresets.json with actual ASCEND_HOME_PATH
    cmake_file = os.path.join(src, "CMakePresets.json")
    if os.path.exists(cmake_file):
        content = safe_read_file(cmake_file) or ""
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        rendered = renderer.render_cmake_presets(content, ascend_home)
        safe_write_file(cmake_file, rendered)

    # 6. Optional: inject tiling parameter prints
    if config.tiling_dump:
        from npukernelbench.stages.compile.tiling_injector import inject_tiling_prints
        inject_tiling_prints(src, logger)

    if logger:
        logger.info(f"Source files prepared in {src}")


def _write_generated_sources(build: BuildHandle, src: str, config: FullConfig, renderer: TemplateRenderer) -> None:
    """Write LLM-generated tiling and kernel files."""
    kernel_name = build.metadata.kernel_name

    if not config.kernel_only_mode:
        files = [
            (f"op_host/{kernel_name}_tiling.h", build.tiling_def),
            (f"op_host/{kernel_name}.cpp", build.tiling_content),
            (f"op_kernel/{kernel_name}.cpp", build.kernel_content),
        ]
    else:
        kernel = renderer.patch_kernel_signature(build.kernel_content or "")
        files = [
            (f"op_host/{kernel_name}.cpp", build.tiling_content),
            (f"op_kernel/{kernel_name}.cpp", kernel),
        ]

    for rel_path, content in files:
        if not content:
            raise RuntimeError(f"Empty content for {rel_path}")
        safe_write_file(os.path.join(src, rel_path), content)


def _write_reference_sources(build: BuildHandle, src: str, config: FullConfig, renderer: TemplateRenderer) -> None:
    """Copy reference answer source files."""
    answer_path = build.answer_path
    kernel_name = build.metadata.kernel_name

    if not config.kernel_only_mode:
        ops_dst = os.path.join(src, "src", "ops", kernel_name)
        shutil.copytree(answer_path, ops_dst, dirs_exist_ok=True)

        # Patch CMakeLists.txt
        cmake_file = os.path.join(ops_dst, "CMakeLists.txt")
        if os.path.exists(cmake_file):
            content = safe_read_file(cmake_file) or ""
            rendered = renderer.render_cmake_ops(content, build.metadata.op_name, kernel_name)
            safe_write_file(cmake_file, rendered)
    else:
        question_src = answer_path
        shutil.copytree(question_src, src, dirs_exist_ok=True)
