from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


_POLICY_DIR = Path(__file__).resolve().parents[1] / "attention" / "flash_kda"
sys.path.insert(0, str(_POLICY_DIR))

import flash_kda_variant_policy as policy  # noqa: E402
import generate_variant_policy as generator  # noqa: E402


def _sidecar(item: policy.FlashKdaVariantPolicy) -> dict:
    constexpr = {
        "head_dim": ("int", policy.HEAD_DIM),
        "group": ("int", item.group),
        "layout_qkv": ("str", "BNSD"),
        "dv_base": ("int", item.dv_base),
        "block_num": ("int", item.block_num),
    }
    return {
        "call_contract": {
            "dimensions": [
                {
                    "name": "S",
                    "min": item.sequence_min,
                    "max": item.sequence_max,
                    "multiple_of": policy.CHUNK_SIZE,
                }
            ],
            "parameters": [
                {
                    "name": "K",
                    "role": "runtime",
                    "argument": {
                        "shape": [
                            {"kind": "constant", "value": item.batch},
                            {"kind": "constant", "value": item.heads},
                            {"kind": "dimension", "name": "S"},
                            {"kind": "constant", "value": policy.HEAD_DIM},
                        ]
                    },
                },
                *[
                    {
                        "name": name,
                        "role": "constexpr",
                        "static_value": {"kind": kind, "value": str(value)},
                    }
                    for name, (kind, value) in constexpr.items()
                ],
            ],
        }
    }


def _manifest(tmp_path: Path) -> Path:
    artifacts = []
    for item in policy.FLASH_KDA_VARIANT_POLICIES:
        sidecar = tmp_path / f"{item.variant_id}.json"
        sidecar.write_text(json.dumps(_sidecar(item)), encoding="utf-8")
        artifacts.append({"variant": item.variant_id, "sidecar": sidecar.name})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifacts": artifacts}), encoding="utf-8")
    return manifest


def test_manifest_and_generated_matcher_share_the_delivery_policy(tmp_path):
    manifest = _manifest(tmp_path)

    generator.verify_manifest(manifest)
    header = generator.render_header()

    for item in policy.FLASH_KDA_VARIANT_POLICIES:
        assert f'"{item.variant_id}"' in header
    assert "MatchFlashKdaVariantPolicy" in header


def test_manifest_specialization_drift_is_rejected(tmp_path):
    manifest = _manifest(tmp_path)
    item = policy.FLASH_KDA_VARIANT_POLICIES[0]
    sidecar_path = tmp_path / f"{item.variant_id}.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    for parameter in sidecar["call_contract"]["parameters"]:
        if parameter["name"] == "group":
            parameter["static_value"]["value"] = "1"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="specialization drift"):
        generator.verify_manifest(manifest)


def test_overlapping_delivery_domains_are_rejected(monkeypatch):
    first = policy.FLASH_KDA_VARIANT_POLICIES[0]
    overlap = policy.FlashKdaVariantPolicy(
        "overlap",
        first.batch,
        first.heads,
        first.sequence_min,
        first.sequence_max,
        first.group,
        first.dv_base,
    )
    monkeypatch.setattr(
        policy,
        "FLASH_KDA_VARIANT_POLICIES",
        (*policy.FLASH_KDA_VARIANT_POLICIES, overlap),
    )

    with pytest.raises(ValueError, match="matches"):
        policy.validate_variant_policies()
