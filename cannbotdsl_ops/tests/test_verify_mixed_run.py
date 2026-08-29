from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

from cannbotdsl.package.errors import PackageIntegrityError


_VERIFIER_PATH = Path(__file__).with_name("verify_mixed_run.py")
_SPEC = importlib.util.spec_from_file_location("mixed_run_verifier", _VERIFIER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load verifier: {_VERIFIER_PATH}")
verifier = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verifier
_SPEC.loader.exec_module(verifier)


def test_required_symbols_cover_l0_standalone_and_custom_l2_delivery():
    required = set(verifier.REQUIRED_SYMBOLS)
    assert {
        "l0op::FlashKda(",
        "aclnnFlashKda",
        "aclnnFlashKdaGetWorkspaceSize",
        "aclnnCustomFlashKda",
        "aclnnCustomFlashKdaGetWorkspaceSize",
    } <= required
    assert "aclnn_custom_flash_kda.h" in verifier.REQUIRED_PUBLIC_HEADERS


def test_symbol_verifier_rejects_missing_custom_l2_export():
    symbols = "\n".join(
        symbol
        for symbol in verifier.REQUIRED_SYMBOLS
        if symbol != "aclnnCustomFlashKdaGetWorkspaceSize"
    )
    with pytest.raises(AssertionError, match="aclnnCustomFlashKdaGetWorkspaceSize"):
        verifier._verify_required_symbols(symbols)


def test_get_workspace_prefix_does_not_mask_missing_execute_export():
    symbols = "\n".join(
        symbol
        for symbol in verifier.REQUIRED_SYMBOLS
        if symbol != "aclnnCustomFlashKda"
    )
    with pytest.raises(AssertionError, match="'aclnnCustomFlashKda'"):
        verifier._verify_required_symbols(symbols)


def test_symbol_verifier_accepts_complete_delivery_surface():
    verifier._verify_required_symbols("\n".join(verifier.REQUIRED_SYMBOLS))


_SUPPORT_DIGEST = "a" * 64
_SOC = "Ascend950PR_9599"
_FLASH_KDA_OPERATOR = "ops-transformer::attention/flash_kda"
_MISSING = object()


def _predicate(*, batch: int, sequence_min: int, sequence_max: int):
    return {
        "tensors": {
            "V": {
                "dtype": "float16",
                "shape": [
                    {"equal": batch},
                    {"equal": 32},
                    {
                        "min": sequence_min,
                        "max": sequence_max,
                        "multiple_of": 16,
                    },
                    {"equal": 128},
                ],
            }
        },
        "attrs": {},
    }


def _entry():
    return {
        "schema_version": 5,
        "name": "flash_kda_l0_entry",
        "support_digest": _SUPPORT_DIGEST,
        "public_parameters": list(verifier.PUBLIC_PARAMETERS),
        "generated_internal_parameters": list(
            verifier.GENERATED_INTERNAL_PARAMETERS
        ),
        "output_parameters": list(verifier.OUTPUT_PARAMETERS),
        "variant_manifest": {
            "schema_version": 1,
            "kind": "cannbotdsl-variant-manifest",
            "operator_identity": "flash_kda_l0_entry",
            "public_contract_digest": _SUPPORT_DIGEST,
            "target_soc": _SOC,
            "proof": {
                "coverage_status": "proven",
                "overlap_status": "proven-disjoint",
                "method": "finite-domain-variant-key-v1",
            },
            "artifacts": [
                {
                    "variant_id": "auto_a",
                    "physical_key": "1" * 64,
                    "variant_key": {"dv_base": 128, "group": 4},
                    "content_key": "3" * 64,
                    "cases": [
                        {
                            "case_id": "case_a0",
                            "predicate": _predicate(
                                batch=1, sequence_min=16, sequence_max=64
                            ),
                        },
                        {
                            "case_id": "case_a1",
                            "predicate": _predicate(
                                batch=2, sequence_min=16, sequence_max=64
                            ),
                        },
                    ],
                },
                {
                    "variant_id": "auto_b",
                    "physical_key": "2" * 64,
                    "variant_key": {"dv_base": 64, "group": 8},
                    "content_key": "4" * 64,
                    "cases": [
                        {
                            "case_id": "case_b0",
                            "predicate": _predicate(
                                batch=1, sequence_min=80, sequence_max=128
                            ),
                        }
                    ],
                },
            ],
        },
    }


def _package_manifest():
    entry = _entry()
    return {
        "compatibility": {"soc": _SOC},
        "artifacts": [
            {
                "operator": _FLASH_KDA_OPERATOR,
                "variant": "auto_a",
                "content_key": "3" * 64,
                "entry": deepcopy(entry),
            },
            {
                "operator": _FLASH_KDA_OPERATOR,
                "variant": "auto_b",
                "content_key": "4" * 64,
                "entry": deepcopy(entry),
            },
        ],
    }


def _rules(payload):
    return verifier._validated_variant_rules(payload, expected_provider_count=2)


def _mutate_all_entries(package, mutate):
    for artifact in package["artifacts"]:
        mutate(artifact["entry"])


def _rename_manifest_variant(package):
    _mutate_all_entries(
        package,
        lambda entry: entry["variant_manifest"]["artifacts"][0].__setitem__(
            "variant_id", "auto_missing"
        ),
    )


def _rename_entry(package):
    _mutate_all_entries(
        package, lambda entry: entry.__setitem__("name", "other_entry")
    )


def _drift_support_digest(package):
    _mutate_all_entries(
        package, lambda entry: entry.__setitem__("support_digest", "b" * 64)
    )


def _set_all_entry_fields(package, field, value):
    for artifact in package["artifacts"]:
        if value is _MISSING:
            artifact["entry"].pop(field, None)
        else:
            artifact["entry"][field] = value


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", _MISSING, "entry.name"),
        ("name", None, "entry.name"),
        ("name", 7, "entry.name"),
        ("name", "", "entry.name"),
        ("support_digest", _MISSING, "entry.support_digest"),
        ("support_digest", None, "entry.support_digest"),
        ("support_digest", 7, "entry.support_digest"),
        ("support_digest", "a" * 63, "entry.support_digest"),
        ("support_digest", "A" * 64, "entry.support_digest"),
        ("variant_manifest", _MISSING, "entry.variant_manifest"),
        ("variant_manifest", None, "entry.variant_manifest"),
        ("variant_manifest", [], "entry.variant_manifest"),
    ],
)
def test_entry_identity_fields_are_validated_before_shared_flattener(
    field, value, message, monkeypatch
):
    package = _package_manifest()
    _set_all_entry_fields(package, field, value)

    def forbidden_flatten(*args, **kwargs):
        raise AssertionError("shared flattener called before Entry identity validation")

    monkeypatch.setattr(verifier, "flatten_variant_manifest", forbidden_flatten)

    with pytest.raises(AssertionError, match=message):
        _rules(package)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda package: package["artifacts"][0].pop("operator"),
            "artifact 0.*operator",
        ),
        (
            lambda package: package["artifacts"][0].__setitem__(
                "operator", "other::attention/flash_kda"
            ),
            "artifact 0.*operator",
        ),
        (
            lambda package: [
                artifact.__setitem__("operator", "other::attention/flash_kda")
                for artifact in package["artifacts"]
            ],
            "operator",
        ),
    ],
)
def test_outer_artifacts_are_fixed_to_the_flash_kda_operator(mutate, message):
    package = _package_manifest()
    mutate(package)

    with pytest.raises(AssertionError, match=message):
        _rules(package)


def test_entry_identity_cannot_drift_as_one_self_consistent_table():
    package = _package_manifest()
    for artifact in package["artifacts"]:
        artifact["entry"]["name"] = "other_entry"
        artifact["entry"]["variant_manifest"][
            "operator_identity"
        ] = "other_entry"

    with pytest.raises(AssertionError, match="flash_kda_l0_entry"):
        _rules(package)


def test_repeated_canonical_entry_and_outer_operator_are_accepted():
    package = _package_manifest()

    rules = _rules(package)

    assert {artifact["operator"] for artifact in package["artifacts"]} == {
        _FLASH_KDA_OPERATOR
    }
    assert {rule["variant_id"] for rule in rules} == {"auto_a", "auto_b"}


def test_v5_manifest_is_flattened_once_and_grouped_cases_remain_one_artifact(
    monkeypatch,
):
    package = _package_manifest()
    shared_flatten = verifier.flatten_variant_manifest
    calls = []

    def counted_flatten(*args, **kwargs):
        calls.append((args, kwargs))
        return shared_flatten(*args, **kwargs)

    monkeypatch.setattr(verifier, "flatten_variant_manifest", counted_flatten)

    rules = _rules(package)

    assert len(package["artifacts"]) == 2
    assert [rule["case_id"] for rule in rules] == [
        "case_a0",
        "case_a1",
        "case_b0",
    ]
    assert [rule["variant_id"] for rule in rules] == [
        "auto_a",
        "auto_a",
        "auto_b",
    ]
    assert len(calls) == 1
    _, keyword = calls[0]
    assert keyword == {
        "require_content_keys": True,
        "public_contract_digest": _SUPPORT_DIGEST,
        "operator_identity": "flash_kda_l0_entry",
        "target_soc": _SOC,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda package: package["artifacts"][1].__setitem__(
                "variant", "auto_a"
            ),
            "duplicate.*outer|missing=.*auto_b|extra=",
        ),
        (
            lambda package: package["artifacts"][0].__setitem__(
                "content_key", "4" * 64
            ),
            "variant/content|missing=.*auto_a|extra=.*auto_a",
        ),
        (
            _rename_manifest_variant,
            "variant/content|missing=.*auto_missing|extra=.*auto_a",
        ),
    ],
)
def test_outer_manifest_identity_join_is_exact_and_duplicate_sensitive(
    mutate, message
):
    package = _package_manifest()
    mutate(package)

    with pytest.raises((AssertionError, PackageIntegrityError), match=message):
        _rules(package)


def test_shared_flattener_rejects_corrupted_predicate_before_identity_join():
    package = _package_manifest()
    manifest = package["artifacts"][0]["entry"]["variant_manifest"]
    manifest["artifacts"][1]["cases"][0]["predicate"] = deepcopy(
        manifest["artifacts"][0]["cases"][0]["predicate"]
    )
    package["artifacts"][1]["entry"] = deepcopy(package["artifacts"][0]["entry"])

    with pytest.raises(PackageIntegrityError, match="predicate overlap"):
        _rules(package)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda package: package["compatibility"].__setitem__(
                "soc", "Ascend950PR_9999"
            ),
            "target SoC",
        ),
        (
            _rename_entry,
            "flash_kda_l0_entry|operator identity",
        ),
        (
            _drift_support_digest,
            "public contract digest",
        ),
    ],
)
def test_manifest_is_bound_to_trusted_package_and_entry_identities(mutate, message):
    package = _package_manifest()
    mutate(package)

    with pytest.raises((AssertionError, PackageIntegrityError), match=message):
        _rules(package)


def test_final_flash_kda_verifier_rejects_v4_and_parallel_dispatch_plan():
    package = _package_manifest()
    for artifact in package["artifacts"]:
        artifact["entry"]["schema_version"] = 4
    with pytest.raises(AssertionError, match="schema v5"):
        _rules(package)

    package = _package_manifest()
    for artifact in package["artifacts"]:
        artifact["entry"]["dispatch_plan"] = {"specializations": []}
    with pytest.raises(AssertionError, match="dispatch_plan"):
        _rules(package)


def test_generated_static_values_must_match_declared_internal_parameters():
    package = _package_manifest()
    entry = package["artifacts"][0]["entry"]
    entry["variant_manifest"]["artifacts"][0]["variant_key"] = {
        "dv_base": 128,
        "unexpected": 4,
    }
    package["artifacts"][1]["entry"] = deepcopy(entry)

    with pytest.raises(AssertionError, match="generated internal"):
        _rules(package)
