"""Static contracts for the arena FlashKDA built-in ACLNN integration."""

from __future__ import annotations

import ast
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "attention/flash_kda/op_kernel/flash_kda.py"


def _call_decorator(node: ast.FunctionDef, owner: str, name: str):
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        function = decorator.func
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == owner
            and function.attr == name
        ):
            return decorator
    return None


def test_flash_kda_uses_the_arena_frozen_entry_contract():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    entry = _call_decorator(functions["flash_kda"], "cannbotdsl", "entry")
    assert entry is not None
    keywords = {keyword.arg: keyword.value for keyword in entry.keywords}
    assert ast.literal_eval(keywords["name"]) == "FlashKda"
    assert "build_flash_kda" in functions

    # The retired Transformer publication API and eager driver cache must not
    # creep back into the single arena source.
    text = SOURCE.read_text(encoding="utf-8")
    assert "@register(" not in text
    assert "_get_compiled_kernel" not in text


def test_flash_kda_is_loaded_by_the_builtin_opapi():
    selection = (ROOT / "cmake/cannbotdsl.cmake").read_text(encoding="utf-8")
    integration = (ROOT / "cmake/cannbotdsl_aclnn.cmake").read_text(
        encoding="utf-8"
    )
    symbols = (ROOT / "cmake/symbol.cmake").read_text(encoding="utf-8")
    builder = (ROOT / "scripts/build_cannbotdsl_aclnn.py").read_text(
        encoding="utf-8"
    )
    assert "@cannbotdsl\\\\.entry" in selection
    assert "cannbotdsl_aclnn.cmake" in selection
    assert "cannbotdsl_attach_aclnn" in integration
    assert "cannbotdsl_attach_aclnn(${OPAPI_NAME})" in symbols
    assert "opapi_cannbot_transformer" in integration
    assert '"-Wl,--no-as-needed"' in integration
    assert "$ORIGIN/../../opp/built-in/op_api/lib" in integration
    assert "NO_SONAME TRUE" in integration
    assert "opp/built-in/op_impl/cannbotdsl" in builder
    assert "aclnnFlashKdaGetWorkspaceSize" in integration


def test_old_torch_dispatcher_was_removed():
    dispatcher = ROOT / "attention/flash_kda/torch_extension"
    assert not dispatcher.exists() or not any(dispatcher.iterdir())


def test_builtin_manifest_delivers_the_aclnn_dso_tree():
    module = ET.parse(ROOT / "scripts/package/module/ascend/OpsTransformer.xml")
    entries = {node.get("value"): node for node in module.findall("file_info")}
    aclnn = entries["cannbotdsl_aclnn"]
    assert aclnn.get("dst_path") == "opp/built-in"
    assert aclnn.find("file").get("value") == "op_api"
