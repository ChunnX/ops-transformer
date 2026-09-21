"""Load checkout interfaces and record the DSL sources used by QSLI tests."""
import sys
from pathlib import Path
import pytest

# The checkout's Torch interface must win over an installed older copy.
import importlib
import importlib.abc
import importlib.util
import hashlib
import os


class _CheckoutTorchInterface(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        folder = Path(__file__).resolve().parent.parent.name
        prefix = "cann_ops_transformer.ops.attention." + folder
        directory = Path(__file__).resolve().parent.parent / "torch_extension"
        if fullname == prefix:
            return importlib.util.spec_from_file_location(
                fullname, directory / "__init__.py",
                submodule_search_locations=[str(directory)],
            )
        if fullname == prefix + "." + folder.removesuffix("_dsl"):
            return importlib.util.spec_from_file_location(
                fullname, directory / (folder.removesuffix("_dsl") + ".py"),
            )


sys.meta_path.insert(0, _CheckoutTorchInterface())


@pytest.fixture(scope="session", autouse=True)
def report_indexer_sources():
    # Source assertions are opt-in so CPU-only utility tests stay independent
    # of the installed DSL runtime.
    if not os.environ.get("CANNBOTDSL_TEST_OPS_DIR"):
        return
    folder = Path(__file__).resolve().parent.parent.name
    interface = importlib.import_module(
        "cann_ops_transformer.ops.attention." + folder + "." + folder.removesuffix("_dsl")
    )
    expected_interface = Path(__file__).resolve().parent.parent / "torch_extension"
    assert Path(interface.__file__).resolve().parent == expected_interface
    print("TORCH_INTERFACE", interface.__file__, flush=True)
    expected_ops = os.environ.get("CANNBOTDSL_TEST_OPS_DIR")
    operator = folder.removesuffix("_dsl")
    for name in (operator + "_dsl", operator + "_metadata_dsl"):
        module = importlib.import_module("ops." + name)
        source = Path(module.__file__).resolve()
        if expected_ops:
            assert source.parent == Path(expected_ops).resolve(), str(source)
        print("DSL_SOURCE", str(source), hashlib.sha256(source.read_bytes()).hexdigest(), flush=True)
