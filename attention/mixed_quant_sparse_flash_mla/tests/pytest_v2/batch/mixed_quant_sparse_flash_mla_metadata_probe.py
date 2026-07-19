#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Metadata-only accelerator for validating the PR8737 MQ interface and layout."""

import argparse
from dataclasses import replace
from typing import Any, Dict, Iterable, Tuple

import cann_ops_transformer  # noqa: F401
import torch
import torch_npu

from .consistency_adapter import CaseAdapter
from .metadata_coverage import CoverageContext, MetadataCoverageAnalyzer
from .metadata_layout import MetadataLayout, ParsedMetadata


class MixedQuantMetadataProbe:
    """Run MQ metadata only; never report a main-op consistency PASS."""

    def __init__(self, layout: MetadataLayout = None):
        self.layout = layout or MetadataLayout()

    @staticmethod
    def npu_optional(value: Any, empty_if_none: bool = False) -> Any:
        if value is None:
            return torch.tensor([], dtype=torch.int32).npu() if empty_if_none else None
        return value.npu()

    def run(
        self, input_data: Dict[str, Any], s2_values: Iterable[int] = ()
    ) -> Tuple[ParsedMetadata, Dict[str, Any]]:
        adapter = CaseAdapter(input_data)
        if adapter.name != "mqsmla":
            raise ValueError("MixedQuantMetadataProbe requires an MQSMLA op_input case")
        metadata_input = adapter.metadata
        tensors = adapter.tensors
        metadata = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
            num_heads_q=metadata_input["num_heads_q"],
            num_heads_kv=metadata_input["num_heads_kv"],
            head_dim=metadata_input["head_dim"],
            cu_seqlens_q=self.npu_optional(tensors.get("cu_seqlens_q"), empty_if_none=True),
            cu_seqlens_ori_kv=self.npu_optional(
                tensors.get("cu_seqlens_ori_kv"), empty_if_none=True
            ),
            cu_seqlens_cmp_kv=self.npu_optional(
                tensors.get("cu_seqlens_cmp_kv"), empty_if_none=True
            ),
            seqused_q=self.npu_optional(tensors.get("seqused_q"), empty_if_none=True),
            seqused_ori_kv=self.npu_optional(tensors.get("seqused_ori_kv")),
            seqused_cmp_kv=self.npu_optional(tensors.get("seqused_cmp_kv")),
            cmp_residual_kv=self.npu_optional(tensors.get("cmp_residual_kv")),
            ori_topk_length=self.npu_optional(tensors.get("ori_topk_length")),
            cmp_topk_length=self.npu_optional(tensors.get("cmp_topk_length")),
            quant_mode=tensors["quant_mode"],
            batch_size=metadata_input["batch_size"],
            max_seqlen_q=metadata_input["max_seqlen_q"],
            max_seqlen_ori_kv=metadata_input["max_seqlen_ori_kv"],
            max_seqlen_cmp_kv=metadata_input["max_seqlen_cmp_kv"],
            ori_topk=0,
            cmp_topk=metadata_input["topk"],
            rope_head_dim=tensors["rope_head_dim"],
            cmp_ratio=metadata_input["cmp_ratio"],
            ori_mask_mode=metadata_input["ori_mask_mode"],
            cmp_mask_mode=metadata_input["cmp_mask_mode"],
            ori_win_left=metadata_input["ori_win_left"],
            ori_win_right=metadata_input["ori_win_right"],
            layout_q=metadata_input["layout_q"],
            layout_kv=metadata_input["layout_kv"],
            has_ori_kv=metadata_input["has_ori_kv"],
            has_cmp_kv=metadata_input["has_cmp_kv"],
        )
        torch_npu.npu.synchronize()
        parsed = self.layout.parse(metadata.cpu())
        context = CoverageContext.from_input_data(input_data)
        getter = getattr(torch_npu.npu, "_get_deterministic_level", None)
        if getter is not None:
            context = replace(context, batch_consistency=int(getter()) == 3)
        report = MetadataCoverageAnalyzer.analyze(
            parsed,
            s2_values=s2_values,
            context=context,
        )
        report["source"] = "mixed_quant_sparse_flash_mla_metadata"
        report["scope"] = "metadata-only; main operator batch consistency not asserted"
        return parsed, report

    def run_at_level(
        self,
        input_data: Dict[str, Any],
        deterministic_level: int,
        s2_values: Iterable[int] = (),
    ) -> Tuple[ParsedMetadata, Dict[str, Any]]:
        getter = getattr(torch_npu.npu, "_get_deterministic_level", None)
        if getter is None:
            raise RuntimeError("torch_npu.npu._get_deterministic_level is unavailable")
        previous_level = int(getter())
        try:
            torch_npu.npu.set_deterministic_level(deterministic_level)
            parsed, report = self.run(input_data, s2_values=s2_values)
            report["deterministic_level"] = deterministic_level
            return parsed, report
        finally:
            if int(getter()) != previous_level:
                torch_npu.npu.set_deterministic_level(previous_level)


class MetadataProbeCli:
    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Probe installed MQSMLA metadata only")
        parser.add_argument("--pt", required=True, help="MQSMLA .pt case")
        parser.add_argument("--device", type=int, default=0, help="NPU device id")
        parser.add_argument("--level", type=int, default=None, help="Optional deterministic level")
        parser.add_argument("--s2", default="", help="Optional comma-separated S2 values")
        return parser

    @classmethod
    def main(cls) -> None:
        args = cls.build_parser().parse_args()
        torch_npu.npu.set_device(args.device)
        input_data = torch.load(args.pt, map_location="cpu")
        s2_values = [int(value) for value in args.s2.split(",") if value.strip()]
        probe = MixedQuantMetadataProbe()
        if args.level is None:
            _, report = probe.run(input_data, s2_values=s2_values)
        else:
            _, report = probe.run_at_level(input_data, args.level, s2_values=s2_values)
        print(MetadataCoverageAnalyzer.dumps(report))


if __name__ == "__main__":
    MetadataProbeCli.main()
