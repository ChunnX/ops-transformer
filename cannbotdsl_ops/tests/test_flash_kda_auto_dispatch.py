from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib

from cannbotdsl.package.dispatch import synthesize_aot_dispatch


_OPERATOR_DIR = Path(__file__).resolve().parents[1] / "attention" / "flash_kda"


def _load_operator():
    source = _OPERATOR_DIR / "flash_kda.py"
    spec = importlib.util.spec_from_file_location("ops_flash_kda_auto", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flash_kda_source_uses_entry_only_auto_dispatch():
    module = _load_operator()
    synthesis = synthesize_aot_dispatch(
        module.flash_kda_l0_entry, target_soc="ascend950pr"
    )

    assert not hasattr(module, "flash_kda_variants")
    assert synthesis.entry["schema_version"] == 4
    assert "internal_parameters" not in synthesis.entry
    assert synthesis.entry["generated_internal_parameters"] == [
        "dv_base",
        "group",
    ]
    assert len(synthesis.variants) == 6
    assert {
        (
            item["static_values"]["group"],
            item["static_values"]["dv_base"],
        )
        for item in synthesis.dispatch_plan["specializations"]
    } == {
        (128, 16),
        (256, 16),
        (64, 64),
        (8, 128),
        (4, 128),
        (2, 128),
    }


def test_package_spec_has_no_manual_function_or_variant_provider():
    payload = tomllib.loads(
        (_OPERATOR_DIR / "package.toml").read_text(encoding="utf-8")
    )
