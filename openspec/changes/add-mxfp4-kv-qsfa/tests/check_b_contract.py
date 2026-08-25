#!/usr/bin/env python3
"""Reject accidental drift from the owner-approved B-I raw-record ABI."""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "operator-evidence.json").read_text(encoding="utf-8"))
inputs = {item["name"]: item for item in manifest["interface"]["inputs"]}
key = inputs["key"]
assert key["dtypes"] == ["DT_INT8"], key
assert any("400" in shape for shape in key["shapes"]), key
assert any("256B packed FP4 E2M1" in case for case in manifest["supported_cases"]), manifest["supported_cases"]
assert any("key_dequant_scale" in item and "not numerical" in item for item in manifest["interface"]["state_mutation"])

kernel = (ROOT.parents[2] / "attention/kv_quant_sparse_flash_attention/op_kernel/arch35/"
          "kv_quant_sparse_flash_attention_service_vector_mla_arch35.h").read_text(encoding="utf-8")
assert "StageBIMxfp4Scales" in kernel, "B-I must stage inline E8M0 scales before FP4 dequantization"
assert "rowStrideBytes = 416" in kernel, "B-I must use the 400B->416B aligned input row stride"
assert "MicroAPI::PostLiteral::POST_MODE_UPDATE" in kernel, "B-I aligned VF loads/stores must use update forms"
assert "g * (groupDim / 2)" not in kernel, "B-I must not LoadAlign odd 16B FP4 groups"
print("B-I interface contract check passed")
