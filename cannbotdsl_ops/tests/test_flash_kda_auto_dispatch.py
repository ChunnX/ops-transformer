from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib

from cannbotdsl.package.dispatch import synthesize_aot_dispatch
from cannbotdsl.package.stage_source import direct_package_spec


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


def test_direct_source_entry_has_no_manual_function_or_variant_provider():
    source = _OPERATOR_DIR / "flash_kda.py"
    path, text = direct_package_spec(
        source=source,
        entry="FlashKda",
        operator="ops-transformer::attention/flash_kda",
        package_name="ops-transformer-flash-kda",
        version="0.1.0",
        vendor="ops-transformer",
        target_soc="ascend950pr",
        npu_arch="dav-3510",
    )
    payload = tomllib.loads(text)

    assert path == source
    assert payload["operators"] == [
        {
            "operator": "ops-transformer::attention/flash_kda",
            "entry": "flash_kda:FlashKda",
            "entry_public_name": True,
        }
    ]
    assert "function" not in payload["operators"][0]
    assert "provider" not in payload["operators"][0]
