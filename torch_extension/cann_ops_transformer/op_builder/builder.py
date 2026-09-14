# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import logging
import os
import shutil
import socket
from abc import ABC, abstractmethod
from typing import List, Union
import torch
from torch.utils.cpp_extension import load
from torch.library import Library

ASCEND_HOME_PATH = "ASCEND_HOME_PATH"
_as_library = None

logger = logging.getLogger(__name__)


def get_as_library():
    global _as_library
    if _as_library is None:
        try:
            _as_library = Library("cann_ops_transformer", "DEF")
        except RuntimeError:
            _as_library = Library("cann_ops_transformer", "FRAGMENT")
    return _as_library


class OpBuilder(ABC):
    _loaded_ops = {}

    def __init__(self, name, category=None):
        self.name = name
        self.category = category
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        import torch_npu

        self._torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
        self._package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._cann_path = self.get_cann_path()
        if not hasattr(torch.ops.cann_ops_transformer, self.name):
            self.register_schema(self.schema())
            self.register_meta()
        self._initialized = True

    def ensure_initialized(self):
        self._ensure_initialized()

    def get_cann_path(self):
        if ASCEND_HOME_PATH in os.environ and os.path.exists(
            os.environ[ASCEND_HOME_PATH]
        ):
            return os.environ[ASCEND_HOME_PATH]
        return os.path.dirname(os.path.dirname(self._torch_npu_path))

    @property
    def cann_path(self):
        self._ensure_initialized()
        return self._cann_path

    @property
    def torch_npu_path(self):
        self._ensure_initialized()
        return self._torch_npu_path

    def get_absolute_paths(self, paths):
        self._ensure_initialized()
        return [os.path.join(self._package_path, path) for path in paths]

    def resolve_source(self, cpp_filename):
        self._ensure_initialized()
        if self.category is None:
            return cpp_filename
        return f"csrc/{self.category}/{cpp_filename}"

    def register_schema(self, op_schema: Union[str, List[str], None]):
        if op_schema is None:
            return
        if isinstance(op_schema, str):
            op_schema = [op_schema]
        for schema in op_schema:
            get_as_library().define(schema)

    @abstractmethod
    def sources(self): ...

    @abstractmethod
    def schema(self): ...

    @abstractmethod
    def register_meta(self): ...

    def _custom_opp_paths(self):
        custom_opp_paths = []
        custom_opp_env = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
        for custom_opp_path in custom_opp_env.split(os.pathsep):
            custom_opp_path = custom_opp_path.strip()
            if custom_opp_path and os.path.isdir(custom_opp_path):
                custom_opp_paths.append(custom_opp_path)
        vendors_dir = os.path.join(self._cann_path, "opp", "vendors")
        if os.path.isdir(vendors_dir):
            for vendor_name in sorted(os.listdir(vendors_dir)):
                vendor_dir = os.path.join(vendors_dir, vendor_name)
                if os.path.isdir(vendor_dir):
                    custom_opp_paths.append(vendor_dir)
        return custom_opp_paths

    def include_paths(self):
        self._ensure_initialized()
        paths = []
        for vendor_dir in self._custom_opp_paths():
            inc = os.path.join(vendor_dir, "op_api", "include")
            if os.path.isdir(inc):
                paths.append(inc)
        paths.extend(
            [
                os.path.join(self._torch_npu_path, "include"),
                os.path.join(self._torch_npu_path, "include/third_party/hccl/inc"),
                os.path.join(self._torch_npu_path, "include/third_party/acl/inc"),
                os.path.join(self._torch_npu_path, "include/third_party/op-plugin"),
                os.path.join(self._cann_path, "include"),
                os.path.join(self._cann_path, "include/aclnnop"),
                os.path.join(self._package_path, "common"),
            ]
        )
        return paths

    def cxx_args(self):
        args = [
            "-O3",
            "-w",
            "-fPIC",
            "-fstack-protector-all",
            "-Wl,-z,relro,-z,now,-z,noexecstack",
            "-pie",
            "-s",
            "-fvisibility=hidden",
            "-D_FORTIFY_SOURCE=2",
        ]
        if torch._C._GLIBCXX_USE_CXX11_ABI:
            args.append("-D_GLIBCXX_USE_CXX11_ABI=1")
        else:
            args.append("-D_GLIBCXX_USE_CXX11_ABI=0")
        return args

    def extra_ldflags(self):
        self._ensure_initialized()
        flags = []
        for vendor_dir in self._custom_opp_paths():
            lib = os.path.join(vendor_dir, "op_api", "lib")
            if os.path.isdir(lib):
                flags.append("-L" + lib)
        flags.extend(
            [
                "-L" + os.path.join(self._cann_path, "lib64"),
                "-lascendcl",
                "-L" + os.path.join(self._torch_npu_path, "lib"),
                "-ltorch_npu",
            ]
        )
        return flags

    def load(self, verbose=True):
        self._ensure_initialized()
        if self.name in OpBuilder._loaded_ops:
            return OpBuilder._loaded_ops[self.name]

        self._clear_stale_build_lock()

        # Lazy import avoids pulling arg_check into every builder import path.
        from cann_ops_transformer.utils.arg_check import wrap_op_module

        op_module = load(
            name=self.name,
            sources=self.get_absolute_paths(self.sources()),
            extra_include_paths=self.get_absolute_paths(self.include_paths()),
            extra_cflags=self.cxx_args(),
            extra_ldflags=self.extra_ldflags(),
            verbose=verbose,
        )
        # Arg type-check proxy; ops can call through load() unchanged.
        op_module = wrap_op_module(op_module)
        OpBuilder._loaded_ops[self.name] = op_module

        return op_module

    def _build_directory(self):
        """Return torch's JIT build directory for this op, or None."""
        if os.environ.get("CANN_OPS_LOCK_CLEAN", "1") == "0":
            return None
        try:
            from torch.utils.cpp_extension import _get_build_directory

            build_dir = _get_build_directory(self.name, False)
        except Exception as e:
            logger.debug("torch _get_build_directory unavailable: %s", e)
            build_dir = None
        if build_dir:
            return build_dir
        root = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.join(
            os.path.expanduser("~"), ".cache", "torch_extensions"
        )
        try:
            from torch.utils.cpp_extension import _get_impl_name

            prefix = _get_impl_name()
        except Exception as e:
            logger.debug("torch _get_impl_name unavailable: %s", e)
            return None
        return os.path.join(root, prefix, self.name)

    @staticmethod
    def _pid_alive(pid):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        return pid > 0 and os.path.exists("/proc/%d" % pid)

    def _clear_stale_build_lock(self):
        """Remove the build lock left by a killed JIT compile process.

        torch guards each JIT build dir with a FileBaton lock. If the owning
        process is killed, the lock stays and blocks later builds until
        torch's own deadlock detection (minutes) kicks in. Only remove the
        lock when we are sure the owner is gone; otherwise leave it untouched
        and let torch take over.
        """
        build_dir = self._build_directory()
        if not build_dir or not os.path.isdir(build_dir):
            return
        lock_path = os.path.join(build_dir, "lock")
        if os.path.isdir(lock_path):
            self._clear_stale_dir_lock(lock_path)
        elif os.path.isfile(lock_path):
            self._clear_stale_file_lock(lock_path)

    def _clear_stale_dir_lock(self, lock_path):
        """FileBaton directory lock (older torch)."""
        pid_file = lock_path + ".pid"
        if os.path.isfile(pid_file):
            try:
                with open(pid_file, "r") as f:
                    pid = f.read().strip()
            except OSError:
                return
            if self._pid_alive(pid):
                return
        shutil.rmtree(lock_path, ignore_errors=True)
        if os.path.isfile(pid_file):
            try:
                os.remove(pid_file)
            except OSError:
                pass
        logger.info("Removed stale build lock: %s", lock_path)

    def _clear_stale_file_lock(self, lock_path):
        """FileBaton file lock (newer torch) or empty lock file (torch_npu fork)."""
        try:
            with open(lock_path, "r") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict) and data.get("pid") is not None:
            if data.get("host") != socket.gethostname():
                return
            if self._pid_alive(data.get("pid")):
                return
        elif os.path.getsize(lock_path) != 0:
            return
        try:
            os.remove(lock_path)
            logger.info("Removed stale build lock: %s", lock_path)
        except OSError:
            pass
