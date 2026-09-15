"""TTK plugin for the metadata-first FlashMlaWithKvcache API."""

import importlib.util
from pathlib import Path
import sys


_PACKAGE = "_flash_mla_with_kvcache_ttk_impl"
_IMPL = Path(__file__).with_name("impl")
if _PACKAGE not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _PACKAGE, _IMPL / "__init__.py", submodule_search_locations=[str(_IMPL)]
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_PACKAGE] = _module
    _spec.loader.exec_module(_module)


def _load(stem):
    return importlib.import_module(f"{_PACKAGE}.{stem}")


class FlashMlaWithKvcacheSpec:
    @staticmethod
    def golden(*args, **kwargs):
        return _load("golden").cpu_flash_mla_with_kvcache(*args, **kwargs)

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return _load("inputs").customize_inputs(*args, **kwargs)

    @staticmethod
    def compare(*outputs, **kwargs):
        return _load("compare").compare(*outputs)

    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
    torch_graph = _load("graph").FlashMlaWithKvcacheAclGraph
    npu_preprocess = staticmethod(_load("npu_preprocess").run)


__spec__ = {
    "flash_mla_with_kvcache_ttk_ops.flash_mla_with_kvcache_ttk": "FlashMlaWithKvcacheSpec",
}
