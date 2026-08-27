"""Validate the staged FlashKDA package and emit its L0 variant matcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flash_kda_variant_policy import (
    CHUNK_SIZE,
    FLASH_KDA_VARIANT_POLICIES,
    HEAD_DIM,
    validate_variant_policies,
)


def _constant(expr: dict, expected: int, *, context: str) -> None:
    if expr != {"kind": "constant", "value": expected}:
        raise ValueError(f"{context} expected constant {expected}, got {expr!r}")


def _static_values(contract: dict) -> dict[str, object]:
    values = {}
    for parameter in contract["parameters"]:
        if parameter["role"] != "constexpr":
            continue
        encoded = parameter["static_value"]
        kind = encoded["kind"]
        value = encoded["value"]
        values[parameter["name"]] = int(value) if kind == "int" else value
    return values


def _verify_artifact(package_root: Path, artifact: dict, policy) -> None:
    sidecar = json.loads((package_root / artifact["sidecar"]).read_text())
    contract = sidecar["call_contract"]
    dimensions = contract["dimensions"]
    expected_dimension = {
        "name": "S",
        "min": policy.sequence_min,
        "max": policy.sequence_max,
        "multiple_of": CHUNK_SIZE,
    }
    if dimensions != [expected_dimension]:
        raise ValueError(
            f"{policy.variant_id} dimension contract drift: {dimensions!r}"
        )
    parameters = {item["name"]: item for item in contract["parameters"]}
    k_shape = parameters["K"]["argument"]["shape"]
    if len(k_shape) != 4 or k_shape[2] != {"kind": "dimension", "name": "S"}:
        raise ValueError(f"{policy.variant_id} K shape no longer uses dimension S")
    _constant(k_shape[0], policy.batch, context=f"{policy.variant_id} batch")
    _constant(k_shape[1], policy.heads, context=f"{policy.variant_id} heads")
    _constant(k_shape[3], HEAD_DIM, context=f"{policy.variant_id} head dim")
    expected_static = {
        "head_dim": HEAD_DIM,
        "group": policy.group,
        "layout_qkv": "BNSD",
        "dv_base": policy.dv_base,
        "block_num": policy.block_num,
    }
    actual_static = _static_values(contract)
    if actual_static != expected_static:
        raise ValueError(
            f"{policy.variant_id} specialization drift: expected "
            f"{expected_static!r}, got {actual_static!r}"
        )


def verify_manifest(manifest_path: Path) -> None:
    validate_variant_policies()
    package_root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    artifacts = {item["variant"]: item for item in manifest["artifacts"]}
    expected_ids = {policy.variant_id for policy in FLASH_KDA_VARIANT_POLICIES}
    if set(artifacts) != expected_ids or len(artifacts) != len(manifest["artifacts"]):
        raise ValueError(
            "FlashKDA manifest variants differ from the delivery policy: "
            f"expected {sorted(expected_ids)}, got {sorted(artifacts)}"
        )
    for policy in FLASH_KDA_VARIANT_POLICIES:
        _verify_artifact(package_root, artifacts[policy.variant_id], policy)


def render_header() -> str:
    rows = "\n".join(
        "  {"
        f'"{item.variant_id}", {item.batch}, {item.heads}, '
        f"{item.sequence_min}, {item.sequence_max}, {CHUNK_SIZE}, "
        f"{item.group}, {item.dv_base}, {item.block_num}"
        "},"
        for item in FLASH_KDA_VARIANT_POLICIES
    )
    return f"""// Generated from flash_kda_variant_policy.py. Do not edit.
#pragma once
#include <cstddef>
#include <cstdint>

namespace cannbotdsl {{
namespace generated {{

struct FlashKdaVariantPolicy {{
  const char* variantId;
  int64_t batch;
  int64_t heads;
  int64_t sequenceMin;
  int64_t sequenceMax;
  int64_t sequenceMultiple;
  int64_t group;
  int64_t dvBase;
  int64_t blockNum;
}};

inline constexpr FlashKdaVariantPolicy kFlashKdaVariantPolicies[] = {{
{rows}
}};

inline const FlashKdaVariantPolicy* MatchFlashKdaVariantPolicy(
    int64_t batch, int64_t heads, int64_t sequence) {{
  const FlashKdaVariantPolicy* result = nullptr;
  for (const auto& candidate : kFlashKdaVariantPolicies) {{
    if (batch != candidate.batch || heads != candidate.heads ||
        sequence < candidate.sequenceMin || sequence > candidate.sequenceMax ||
        sequence % candidate.sequenceMultiple != 0) {{
      continue;
    }}
    if (result != nullptr) return nullptr;
    result = &candidate;
  }}
  return result;
}}

}}  // namespace generated
}}  // namespace cannbotdsl
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_header())
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
