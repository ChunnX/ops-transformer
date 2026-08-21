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

"""Layout-aware black-box transforms for SMLA and MQSMLA cases."""

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from . import layouts, schema
from .adapter import CaseAdapter
from .model import ConsistencyCase, TokenOrigin


class UnsupportedTransformError(Exception):
    """Raised when a requested transform has not been implemented."""


class InvalidTransformError(ValueError):
    """Raised when a transform would change the target token's attention semantics."""


class TransformSemanticGuard:
    """Checks position-sensitive mask invariants before consistency execution."""

    @staticmethod
    def validate_token_selection(input_data: Dict[str, Any], selections: Sequence[Sequence[int]]) -> None:
        adapter = CaseAdapter(input_data)
        tensors = adapter.tensors
        ori_mode = tensors.get("ori_mask_mode", 0)
        cmp_mode = tensors.get("cmp_mask_mode", 0)
        has_cmp = tensors.get("cmp_kv") is not None
        position_sensitive = ori_mode in (3, 4) or (has_cmp and cmp_mode in (3, 4))
        if not position_sensitive:
            return

        old_lengths = schema.get_q_lengths(input_data, valid_only=True)
        for batch, selected in enumerate(selections):
            new_length = len(selected)
            for new_position, old_position in enumerate(selected):
                old_distance_from_right = old_lengths[batch] - old_position
                new_distance_from_right = new_length - new_position
                if old_distance_from_right != new_distance_from_right:
                    raise InvalidTransformError(
                        "Mode 3 changes a position-sensitive mask: "
                        f"batch={batch}, old_token={old_position}, new_token={new_position}, "
                        f"old_length={old_lengths[batch]}, new_length={new_length}"
                    )


class ActualInputSemanticOracle:
    """Proves that mapped tokens consume identical Q/KV inputs and mask windows."""

    @staticmethod
    def _tensor_bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
        """Compare exact storage bits so identical FP8 NaN payloads remain equal."""
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        left_bytes = left.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
        right_bytes = right.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
        return torch.equal(left_bytes, right_bytes)

    @staticmethod
    def _lengths(input_data: Dict[str, Any], prefix: str) -> List[int]:
        adapter = CaseAdapter(input_data)
        tensors = adapter.tensors
        seqused = tensors.get(f"seqused_{prefix}_kv")
        if seqused is not None:
            return [int(value) for value in seqused.to("cpu", torch.int64).tolist()]
        if adapter.get_layout_kv() == "TND":
            cu = tensors.get(f"cu_seqlens_{prefix}_kv")
            if cu is None:
                return [0] * adapter.get_batch_size()
            return layouts.lengths_from_cu_seqlens(cu)
        kv = tensors.get(f"{prefix}_kv")
        if kv is None:
            return [0] * adapter.get_batch_size()
        if adapter.get_layout_kv() == "BSND":
            return [int(kv.shape[1])] * adapter.get_batch_size()
        raise InvalidTransformError(f"PA_BBND {prefix}_kv requires seqused_{prefix}_kv")

    @staticmethod
    def _q_token(input_data: Dict[str, Any], batch: int, token: int) -> torch.Tensor:
        adapter = CaseAdapter(input_data)
        tensors = adapter.tensors
        if adapter.get_layout_q() == "BSND":
            return tensors["q"][batch, token]
        cu = tensors["cu_seqlens_q"].to("cpu", torch.int64).tolist()
        return tensors["q"][cu[batch] + token]

    @staticmethod
    def _q_aligned_token(
        input_data: Dict[str, Any], name: str, batch: int, token: int
    ) -> Optional[torch.Tensor]:
        adapter = CaseAdapter(input_data)
        tensor = adapter.tensors.get(name)
        if tensor is None:
            return None
        if adapter.get_layout_q() == "TND":
            cu = adapter.tensors["cu_seqlens_q"].to("cpu", torch.int64).tolist()
            return tensor[cu[batch] + token]
        if name in ("ori_sparse_indices", "cmp_sparse_indices"):
            return tensor[batch, :, token]
        return tensor[batch, token]

    @classmethod
    def _logical_kv(cls, input_data: Dict[str, Any], prefix: str, batch: int) -> Optional[torch.Tensor]:
        adapter = CaseAdapter(input_data)
        tensors = adapter.tensors
        kv = tensors.get(f"{prefix}_kv")
        if kv is None:
            return None
        length = cls._lengths(input_data, prefix)[batch]
        layout = adapter.get_layout_kv()
        if layout == "BSND":
            return kv[batch, :length]
        if layout == "TND":
            cu = tensors[f"cu_seqlens_{prefix}_kv"].to("cpu", torch.int64).tolist()
            return kv[cu[batch]:cu[batch] + length]

        table = tensors.get(f"{prefix}_block_table")
        if table is None:
            raise InvalidTransformError(f"PA_BBND {prefix}_kv requires a block table")
        block_size = int(kv.shape[1])
        block_count = (length + block_size - 1) // block_size
        block_ids = table[batch, :block_count].to("cpu", torch.int64).tolist()
        if any(block_id < 0 or block_id >= kv.shape[0] for block_id in block_ids):
            raise InvalidTransformError(
                f"PA_BBND {prefix}_block_table cannot address {length} logical tokens"
            )
        pieces = [kv[block_id] for block_id in block_ids]
        if not pieces:
            return kv.new_empty((0,) + tuple(kv.shape[2:]))
        return torch.cat(pieces, dim=0)[:length]

    @classmethod
    def _windows(cls, input_data: Dict[str, Any], batch: int, token: int) -> Dict[str, Tuple[int, int]]:
        adapter = CaseAdapter(input_data)
        tensors = adapter.tensors
        q_length = schema.get_q_lengths(input_data, valid_only=True)[batch]
        ori_length = cls._lengths(input_data, "ori")[batch]
        cmp_lengths = cls._lengths(input_data, "cmp")
        cmp_length = cmp_lengths[batch]
        ori_mode = int(tensors.get("ori_mask_mode", 0))
        cmp_mode = int(tensors.get("cmp_mask_mode", 0))
        win_left = int(tensors.get("ori_win_left", -1))
        win_right = int(tensors.get("ori_win_right", 0))

        if ori_mode == 0:
            ori_window = (0, ori_length)
        elif ori_mode in (3, 4):
            threshold = ori_length - q_length + token + 1
            start = 0 if ori_mode == 3 or win_left == -1 else max(threshold - win_left - 1, 0)
            ori_window = (max(start, 0), min(max(threshold + win_right, 0), ori_length))
        else:
            raise InvalidTransformError(f"Unsupported ori_mask_mode={ori_mode} in semantic oracle")

        if tensors.get("cmp_kv") is None:
            cmp_window = (0, 0)
        else:
            ratio = int(tensors.get("cmp_ratio") or adapter.metadata.get("cmp_ratio") or 1)
            if cmp_mode == 0:
                end = min(cmp_length, ori_length // ratio)
            elif cmp_mode == 3:
                end = min(cmp_length, (ori_length - q_length + token + 1) // ratio)
            else:
                raise InvalidTransformError(f"Unsupported cmp_mask_mode={cmp_mode} in semantic oracle")
            cmp_window = (0, max(end, 0))
        return {"ori": ori_window, "cmp": cmp_window}

    @classmethod
    def validate_mapped_tokens(
        cls,
        baseline: Dict[str, Any],
        derived: Dict[str, Any],
        mapping: Sequence[Tuple[int, int, int, int]],
    ) -> Dict[str, Any]:
        baseline_adapter = CaseAdapter(baseline)
        derived_adapter = CaseAdapter(derived)
        if baseline_adapter.get_layout_q() != derived_adapter.get_layout_q():
            raise InvalidTransformError("Q layout changed during transform")
        if baseline_adapter.get_layout_kv() != derived_adapter.get_layout_kv():
            raise InvalidTransformError("KV layout changed during transform")
        if not cls._tensor_bits_equal(
            baseline_adapter.tensors["sinks"], derived_adapter.tensors["sinks"]
        ):
            raise InvalidTransformError("sinks changed during transform")

        checked = []
        for old_batch, old_token, new_batch, new_token in mapping:
            if not cls._tensor_bits_equal(
                cls._q_token(baseline, old_batch, old_token),
                cls._q_token(derived, new_batch, new_token),
            ):
                raise InvalidTransformError(
                    f"Q changed for source ({old_batch}, {old_token})"
                )
            old_windows = cls._windows(baseline, old_batch, old_token)
            new_windows = cls._windows(derived, new_batch, new_token)
            if old_windows != new_windows:
                raise InvalidTransformError(
                    "Mapped token sees a different mask window: "
                    f"source=({old_batch},{old_token}) {old_windows}, "
                    f"derived=({new_batch},{new_token}) {new_windows}"
                )

            for prefix in ("ori", "cmp"):
                start, end = old_windows[prefix]
                old_kv = cls._logical_kv(baseline, prefix, old_batch)
                new_kv = cls._logical_kv(derived, prefix, new_batch)
                if old_kv is None and new_kv is None:
                    continue
                if (
                    old_kv is None
                    or new_kv is None
                    or not cls._tensor_bits_equal(old_kv[start:end], new_kv[start:end])
                ):
                    raise InvalidTransformError(
                        f"Visible {prefix}_kv changed for source ({old_batch}, {old_token})"
                    )

            for name in (
                "ori_sparse_indices",
                "cmp_sparse_indices",
                "ori_topk_length",
                "cmp_topk_length",
            ):
                old_value = cls._q_aligned_token(baseline, name, old_batch, old_token)
                new_value = cls._q_aligned_token(derived, name, new_batch, new_token)
                if old_value is None and new_value is None:
                    continue
                if (
                    old_value is None
                    or new_value is None
                    or not cls._tensor_bits_equal(old_value, new_value)
                ):
                    raise InvalidTransformError(
                        f"{name} changed for source ({old_batch}, {old_token})"
                    )
            checked.append(
                {
                    "source": [old_batch, old_token],
                    "derived": [new_batch, new_token],
                    "windows": old_windows,
                }
            )
        return {"pass": True, "checked_token_count": len(checked), "tokens": checked}


class BatchCaseTransformer:
    """Owns coupled tensor/metadata updates for consistency modes 1-4."""

    def __init__(self, input_data: Dict[str, Any]):
        schema.validate_schema(input_data)
        self.input_data = input_data
        self.adapter = CaseAdapter(input_data)
        self.tensors = self.adapter.tensors
        self.layout_q = self.adapter.get_layout_q()
        self.layout_kv = self.adapter.get_layout_kv()

    @staticmethod
    def _clone_input_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
        cloned = {}
        for key, value in input_data.items():
            if isinstance(value, dict):
                cloned[key] = dict(value)
            elif isinstance(value, list):
                cloned[key] = list(value)
            else:
                cloned[key] = value
        adapter = CaseAdapter(input_data)
        cloned[adapter.tensor_key] = dict(adapter.tensors)
        cloned["metadata_input"] = dict(adapter.metadata)
        cloned["params"] = dict(adapter.params)
        return cloned

    @staticmethod
    def _cu_tensor(values: List[int], reference: Optional[torch.Tensor]) -> torch.Tensor:
        if reference is None:
            return torch.tensor(values, dtype=torch.int32)
        return torch.tensor(values, dtype=reference.dtype, device=reference.device)

    @staticmethod
    def _select_kv(
        kv: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        layout_kv: str,
        batch_ids: List[int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[int]]]:
        if layout_kv == "PA_BBND":
            return kv, layouts.select_pa_block_table(block_table, batch_ids), None
        if layout_kv == "TND":
            if kv is None:
                return None, None, None
            if cu_seqlens is None:
                raise schema.SchemaError("TND KV tensor requires matching cu_seqlens")
            selected, new_cu = layouts.select_tnd(kv, cu_seqlens, batch_ids)
            return selected, None, new_cu
        return layouts.select_bsnd(kv, batch_ids), None, None

    @staticmethod
    def _max_kv_length(
        kv: Optional[torch.Tensor],
        seqused: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        layout_kv: str,
    ) -> int:
        if seqused is not None and seqused.numel():
            return int(seqused.max().item())
        if kv is None:
            return 0
        if layout_kv == "BSND":
            return int(kv.shape[1])
        if layout_kv == "TND":
            if cu_seqlens is None:
                raise schema.SchemaError("Cannot derive TND KV max length without cu_seqlens")
            return max(layouts.lengths_from_cu_seqlens(cu_seqlens), default=0)
        raise schema.SchemaError("PA_BBND requires seqused KV lengths")

    @classmethod
    def _logical_token_count(
        cls, kv: Optional[torch.Tensor], seqused: Optional[torch.Tensor], layout_kv: str
    ) -> Optional[int]:
        if kv is None:
            return None
        if layout_kv == "TND":
            return int(kv.shape[0])
        if layout_kv == "BSND":
            return int(kv.shape[0] * kv.shape[1])
        if seqused is None:
            raise schema.SchemaError("PA_BBND requires seqused KV lengths")
        return int(seqused.to(torch.int64).sum().item())

    @staticmethod
    def _build_origins(batch_ids: Sequence[int], q_lengths: Sequence[int]) -> List[TokenOrigin]:
        origins = []
        for new_batch, old_batch in enumerate(batch_ids):
            origins.extend(
                TokenOrigin(source_batch=old_batch, source_token=token)
                for token in range(q_lengths[new_batch])
            )
        return origins

    def _select_q(self, batch_ids: List[int]) -> Tuple[torch.Tensor, Optional[List[int]]]:
        if self.layout_q == "BSND":
            return layouts.select_bsnd(self.tensors["q"], batch_ids), None
        cu_q = self.tensors.get("cu_seqlens_q")
        if cu_q is None:
            raise schema.SchemaError("TND Q requires cu_seqlens_q")
        return layouts.select_tnd(self.tensors["q"], cu_q, batch_ids)

    def _select_q_aligned_tensor(
        self, tensor: Optional[torch.Tensor], batch_ids: List[int]
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if self.layout_q == "BSND":
            return layouts.select_bsnd(tensor, batch_ids)
        selected, _ = layouts.select_tnd(tensor, self.tensors["cu_seqlens_q"], batch_ids)
        return selected

    def select_batches(self, batch_ids: List[int]) -> Dict[str, Any]:
        new_data = self._clone_input_data(self.input_data)
        new_adapter = CaseAdapter(new_data)
        tensors = new_adapter.tensors
        metadata = new_adapter.metadata
        params = new_adapter.params
        new_batch_size = len(batch_ids)
        original_cu_q = self.tensors.get("cu_seqlens_q")

        new_q, new_cu_q = self._select_q(batch_ids)
        tensors["q"] = new_q

        for prefix in ("ori", "cmp"):
            kv_name = f"{prefix}_kv"
            table_name = f"{prefix}_block_table"
            cu_name = f"cu_seqlens_{prefix}_kv"
            new_kv, new_table, new_cu = self._select_kv(
                self.tensors.get(kv_name),
                self.tensors.get(table_name),
                self.tensors.get(cu_name),
                self.layout_kv,
                batch_ids,
            )
            if kv_name in tensors or new_kv is not None:
                tensors[kv_name] = new_kv
            if new_table is not None:
                tensors[table_name] = new_table
            if new_cu is not None:
                cu_tensor = self._cu_tensor(new_cu, self.tensors.get(cu_name))
                tensors[cu_name] = cu_tensor
                metadata[cu_name] = cu_tensor
                params[cu_name] = new_cu

        for sparse_name in ("ori_sparse_indices", "cmp_sparse_indices"):
            if self.tensors.get(sparse_name) is not None:
                tensors[sparse_name] = self._select_q_aligned_tensor(self.tensors[sparse_name], batch_ids)

        if self.layout_q == "TND" and new_cu_q is not None:
            cu_tensor = self._cu_tensor(new_cu_q, original_cu_q)
            tensors["cu_seqlens_q"] = cu_tensor
            metadata["cu_seqlens_q"] = cu_tensor
            params["cu_seqlens_q"] = new_cu_q
        elif self.layout_q == "BSND" and original_cu_q is not None:
            physical_s1 = int(new_q.shape[1])
            cu_values = [index * physical_s1 for index in range(new_batch_size + 1)]
            cu_tensor = self._cu_tensor(cu_values, original_cu_q)
            tensors["cu_seqlens_q"] = cu_tensor
            metadata["cu_seqlens_q"] = cu_tensor
            params["cu_seqlens_q"] = cu_values

        for field_name in ("seqused_q", "seqused_ori_kv", "seqused_cmp_kv", "cmp_residual_kv"):
            value = self.tensors.get(field_name)
            if value is not None:
                selected = layouts.select_per_batch_vec(value, batch_ids)
                tensors[field_name] = selected
                metadata[field_name] = selected
                params[field_name] = selected.to("cpu", torch.int64).tolist()

        for field_name in ("ori_topk_length", "cmp_topk_length"):
            value = self.tensors.get(field_name)
            if value is not None:
                selected = self._select_q_aligned_tensor(value, batch_ids)
                tensors[field_name] = selected
                metadata[field_name] = selected
                params[field_name] = selected.detach().cpu().tolist()

        new_adapter.set_batch_size(new_batch_size)
        self._sync_derived_fields(new_adapter)
        new_data["cpu_output"] = None
        if "softmax_lse" in new_data:
            new_data["softmax_lse"] = None
        schema.check_invariants(new_data)
        return new_data

    def _sync_derived_fields(self, adapter: CaseAdapter) -> None:
        tensors = adapter.tensors
        q = tensors["q"]
        q_lengths = schema.get_q_lengths(adapter.input_data)
        new_t1 = int(q.shape[0]) if self.layout_q == "TND" else int(q.shape[0] * q.shape[1])
        new_t2 = self._logical_token_count(tensors.get("ori_kv"), tensors.get("seqused_ori_kv"), self.layout_kv)
        new_t3 = self._logical_token_count(tensors.get("cmp_kv"), tensors.get("seqused_cmp_kv"), self.layout_kv)
        for name, value in (("T1", new_t1), ("T2", new_t2), ("T3", new_t3)):
            if value is not None:
                adapter.set_reporting_field(name, value)

        adapter.set_reporting_field("S1", max(q_lengths, default=1))
        max_ori = self._max_kv_length(
            tensors.get("ori_kv"), tensors.get("seqused_ori_kv"), tensors.get("cu_seqlens_ori_kv"), self.layout_kv
        )
        max_cmp = self._max_kv_length(
            tensors.get("cmp_kv"), tensors.get("seqused_cmp_kv"), tensors.get("cu_seqlens_cmp_kv"), self.layout_kv
        )
        adapter.set_reporting_field("S2", max_ori)
        adapter.metadata["max_seqlen_q"] = max(q_lengths, default=1)
        adapter.metadata["max_seqlen_ori_kv"] = max_ori
        adapter.metadata["max_seqlen_cmp_kv"] = max_cmp

    def reorder(self, order: Optional[List[int]], seed: Optional[int]) -> ConsistencyCase:
        batch_size = self.adapter.get_batch_size()
        if batch_size < 2:
            raise ValueError(f"Mode 1 requires B >= 2, got B={batch_size}")
        if order is None:
            order = generate_non_identity_permutation(batch_size, seed)
        elif sorted(order) != list(range(batch_size)):
            raise ValueError(f"order must be a permutation of [0..{batch_size - 1}], got {order}")

        new_data = self.select_batches(order)
        physical_q_lengths = schema.get_q_lengths(new_data)
        valid_q_lengths = schema.get_q_lengths(new_data, valid_only=True)
        return ConsistencyCase(
            name=f"mode1_reorder_B{batch_size}",
            input_data=new_data,
            output_origins=self._build_origins(order, valid_q_lengths),
            transform_meta={
                "mode": 1,
                "mode_name": "reorder",
                "order": order,
                "seed": seed,
                "layout_q": self.layout_q,
                "layout_kv": self.layout_kv,
                "operator": self.adapter.name,
                "physical_q_lengths": physical_q_lengths,
                "valid_q_lengths": valid_q_lengths,
            },
        )

    def split(
        self,
        groups: Optional[List[List[int]]],
        seed: Optional[int],
        randomize: bool,
        num_groups: Optional[int],
    ) -> List[ConsistencyCase]:
        batch_size = self.adapter.get_batch_size()
        if batch_size < 2:
            raise ValueError(f"Mode 2 requires B >= 2, got B={batch_size}")
        if groups is None:
            groups = generate_split_groups(batch_size, num_groups, seed, randomize=randomize)
        self._validate_groups(groups, batch_size)
        flattened = [batch for group in groups for batch in group]
        complete_partition = sorted(flattened) == list(range(batch_size))

        cases = []
        for group_index, group in enumerate(groups):
            new_data = self.select_batches(group)
            physical_q_lengths = schema.get_q_lengths(new_data)
            valid_q_lengths = schema.get_q_lengths(new_data, valid_only=True)
            cases.append(
                ConsistencyCase(
                    name=f"mode2_split_g{group_index}_B{len(group)}",
                    input_data=new_data,
                    output_origins=self._build_origins(group, valid_q_lengths),
                    transform_meta={
                        "mode": 2,
                        "mode_name": "split",
                        "group_index": group_index,
                        "batch_ids": group,
                        "seed": seed,
                        "randomized": randomize,
                        "complete_partition": complete_partition,
                        "layout_q": self.layout_q,
                        "layout_kv": self.layout_kv,
                        "operator": self.adapter.name,
                        "physical_q_lengths": physical_q_lengths,
                        "valid_q_lengths": valid_q_lengths,
                    },
                )
            )
        return cases

    @staticmethod
    def _validate_groups(groups: List[List[int]], batch_size: int) -> None:
        if not groups:
            raise ValueError("groups must contain at least one non-empty batch subset")
        seen = set()
        for index, group in enumerate(groups):
            if not group:
                raise ValueError(f"groups[{index}] is empty; empty groups are not allowed")
            if len(set(group)) != len(group):
                raise ValueError(f"groups[{index}] contains duplicate batch ids: {group}")
            for batch in group:
                if batch < 0 or batch >= batch_size:
                    raise ValueError(f"batch id {batch} is outside [0, {batch_size})")
                if batch in seen:
                    raise ValueError(f"batch id {batch} appears in more than one group")
                seen.add(batch)

    def token_select(
        self,
        token_ids_by_batch: Optional[List[List[int]]],
        batch_ids: Optional[List[int]],
        validate_semantics: bool,
    ) -> ConsistencyCase:
        if batch_ids is None:
            batch_ids = list(range(self.adapter.get_batch_size()))
        batch_size = self.adapter.get_batch_size()
        if not batch_ids:
            raise ValueError("batch_ids must contain at least one batch")
        if len(set(batch_ids)) != len(batch_ids):
            raise ValueError(f"batch_ids contains duplicates: {batch_ids}")
        if any(batch < 0 or batch >= batch_size for batch in batch_ids):
            raise ValueError(f"batch_ids must be within [0, {batch_size}), got {batch_ids}")
        original_lengths = schema.get_q_lengths(self.input_data, valid_only=True)
        if token_ids_by_batch is None:
            empty_batches = [batch for batch in batch_ids if original_lengths[batch] == 0]
            if empty_batches:
                raise InvalidTransformError(
                    f"Mode 3 has no valid Q token in batches {empty_batches}"
                )
            token_ids_by_batch = [[0] for _ in batch_ids]
        if len(token_ids_by_batch) != len(batch_ids):
            raise ValueError("token_ids_by_batch length must match batch_ids length")

        selections = []
        for old_batch, token_ids in zip(batch_ids, token_ids_by_batch):
            if not token_ids:
                raise ValueError(f"batch {old_batch} has an empty token selection")
            if len(set(token_ids)) != len(token_ids):
                raise ValueError(f"batch {old_batch} token selection contains duplicates")
            if any(token < 0 or token >= original_lengths[old_batch] for token in token_ids):
                raise ValueError(f"batch {old_batch} token selection {token_ids} is out of range")
            selections.append(list(token_ids))

        if validate_semantics:
            compact_input = self.select_batches(batch_ids)
            TransformSemanticGuard.validate_token_selection(compact_input, selections)

        new_data = self.select_batches(batch_ids)
        new_adapter = CaseAdapter(new_data)
        tensors = new_adapter.tensors
        self._select_query_tokens(tensors, batch_ids, selections)
        self._sync_token_lengths(new_adapter, selections)
        self._sync_derived_fields(new_adapter)
        new_data["cpu_output"] = None
        if "softmax_lse" in new_data:
            new_data["softmax_lse"] = None
        schema.check_invariants(new_data)

        origins = [
            TokenOrigin(source_batch=old_batch, source_token=token)
            for old_batch, selected in zip(batch_ids, selections)
            for token in selected
        ]
        physical_q_lengths = schema.get_q_lengths(new_data)
        valid_q_lengths = schema.get_q_lengths(new_data, valid_only=True)
        return ConsistencyCase(
            name=f"mode3_token_select_B{len(batch_ids)}",
            input_data=new_data,
            output_origins=origins,
            transform_meta={
                "mode": 3,
                "mode_name": "token-split",
                "batch_ids": batch_ids,
                "token_ids_by_batch": selections,
                "layout_q": self.layout_q,
                "layout_kv": self.layout_kv,
                "operator": self.adapter.name,
                "physical_q_lengths": physical_q_lengths,
                "valid_q_lengths": valid_q_lengths,
            },
        )

    @staticmethod
    def _resize_tensor_axis(
        tensor: torch.Tensor, axis: int, target: int, fill_value: int = 0
    ) -> torch.Tensor:
        current = int(tensor.shape[axis])
        if target <= current:
            slices = [slice(None)] * tensor.dim()
            slices[axis] = slice(0, target)
            return tensor[tuple(slices)].contiguous()
        pad_shape = list(tensor.shape)
        pad_shape[axis] = target - current
        padding = torch.full(
            pad_shape,
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, padding], dim=axis).contiguous()

    def _resize_query_prefix(self, input_data: Dict[str, Any], target_length: int) -> None:
        if target_length < 1:
            raise InvalidTransformError("A derived query must contain at least one token")
        adapter = CaseAdapter(input_data)
        tensors = adapter.tensors
        if adapter.get_batch_size() != 1:
            raise InvalidTransformError("Query prefix resizing currently requires one selected batch")
        if self.layout_q == "BSND":
            tensors["q"] = self._resize_tensor_axis(tensors["q"], 1, target_length)
            for name in ("ori_sparse_indices", "cmp_sparse_indices"):
                if tensors.get(name) is not None:
                    tensors[name] = self._resize_tensor_axis(tensors[name], 2, target_length, -1)
            for name in ("ori_topk_length", "cmp_topk_length"):
                if tensors.get(name) is not None:
                    tensors[name] = self._resize_tensor_axis(tensors[name], 1, target_length)
        else:
            for name in (
                "q",
                "ori_sparse_indices",
                "cmp_sparse_indices",
                "ori_topk_length",
                "cmp_topk_length",
            ):
                tensor = tensors.get(name)
                if tensor is not None:
                    fill_value = -1 if "sparse_indices" in name else 0
                    tensors[name] = self._resize_tensor_axis(tensor, 0, target_length, fill_value)
        self._sync_token_lengths(adapter, [list(range(target_length))])
        self._sync_derived_fields(adapter)

    @staticmethod
    def _set_vector(
        adapter: CaseAdapter, name: str, values: List[int], reference: Optional[torch.Tensor]
    ) -> torch.Tensor:
        tensor = torch.tensor(
            values,
            dtype=reference.dtype if reference is not None else torch.int32,
            device=reference.device if reference is not None else None,
        )
        adapter.tensors[name] = tensor
        adapter.metadata[name] = tensor
        adapter.params[name] = values
        return tensor

    def _set_kv_lengths(
        self, input_data: Dict[str, Any], prefix: str, lengths: List[int]
    ) -> None:
        adapter = CaseAdapter(input_data)
        tensors = adapter.tensors
        kv_name = f"{prefix}_kv"
        kv = tensors.get(kv_name)
        if kv is None:
            if any(lengths):
                raise InvalidTransformError(f"Cannot assign non-zero lengths without {kv_name}")
            return
        if len(lengths) != adapter.get_batch_size() or any(length < 0 for length in lengths):
            raise InvalidTransformError(f"Invalid {prefix}_kv lengths: {lengths}")

        layout = adapter.get_layout_kv()
        if layout == "BSND":
            target = max(lengths, default=0)
            tensors[kv_name] = self._resize_tensor_axis(kv, 1, max(target, int(kv.shape[1])))
        elif layout == "TND":
            cu_name = f"cu_seqlens_{prefix}_kv"
            cu = tensors.get(cu_name)
            if cu is None:
                raise InvalidTransformError(f"TND {kv_name} requires {cu_name}")
            segments = layouts.split_tnd(kv, cu)
            resized = [self._resize_tensor_axis(segment, 0, length) for segment, length in zip(segments, lengths)]
            joined, new_cu = layouts.concat_tnd_segments(resized)
            tensors[kv_name] = joined
            cu_tensor = self._cu_tensor(new_cu, cu)
            tensors[cu_name] = cu_tensor
            adapter.metadata[cu_name] = cu_tensor
            adapter.params[cu_name] = new_cu
        else:
            table = tensors.get(f"{prefix}_block_table")
            if table is None:
                raise InvalidTransformError(f"PA_BBND {kv_name} requires a block table")
            block_size = int(kv.shape[1])
            for batch, length in enumerate(lengths):
                needed = (length + block_size - 1) // block_size
                available = int((table[batch] >= 0).sum().item())
                if needed > available:
                    raise InvalidTransformError(
                        f"PA_BBND {prefix} batch {batch} needs {needed} blocks, only {available} mapped"
                    )

        reference = tensors.get(f"seqused_{prefix}_kv")
        self._set_vector(adapter, f"seqused_{prefix}_kv", lengths, reference)

    def _align_position_sensitive_context(
        self, input_data: Dict[str, Any], source_batch: int, new_ori_length: int
    ) -> None:
        adapter = CaseAdapter(input_data)
        tensors = adapter.tensors
        ori_mode = int(tensors.get("ori_mask_mode", 0))
        cmp_mode = int(tensors.get("cmp_mask_mode", 0))
        has_cmp = tensors.get("cmp_kv") is not None
        if ori_mode not in (3, 4) and not (has_cmp and cmp_mode in (3, 4)):
            return

        self._set_kv_lengths(input_data, "ori", [new_ori_length])
        if has_cmp:
            ratio = int(tensors.get("cmp_ratio") or adapter.metadata.get("cmp_ratio") or 1)
            cmp_length, residual = divmod(new_ori_length, ratio)
            self._set_kv_lengths(input_data, "cmp", [cmp_length])
            reference = tensors.get("cmp_residual_kv")
            self._set_vector(adapter, "cmp_residual_kv", [residual], reference)
        self._sync_derived_fields(adapter)
        schema.check_invariants(input_data)

    def _has_position_sensitive_context(self) -> bool:
        ori_mode = int(self.tensors.get("ori_mask_mode", 0))
        cmp_mode = int(self.tensors.get("cmp_mask_mode", 0))
        return ori_mode in (3, 4) or (
            self.tensors.get("cmp_kv") is not None and cmp_mode in (3, 4)
        )

    def token_partition(self, batch_id: int, split_sizes: Sequence[int]) -> List[ConsistencyCase]:
        """Mode 3: split one batch into explicit, complete token-range calls."""
        valid_lengths = schema.get_q_lengths(self.input_data, valid_only=True)
        if batch_id < 0 or batch_id >= len(valid_lengths):
            raise ValueError(f"batch_id {batch_id} is outside [0, {len(valid_lengths)})")
        if len(split_sizes) < 2 or any(size <= 0 for size in split_sizes):
            raise ValueError("Mode 3 split_sizes must contain at least two positive values")
        old_q_length = valid_lengths[batch_id]
        if sum(split_sizes) != old_q_length:
            raise ValueError(
                f"Mode 3 split sizes {list(split_sizes)} sum to {sum(split_sizes)}, expected {old_q_length}"
            )
        old_ori_length = ActualInputSemanticOracle._lengths(self.input_data, "ori")[batch_id]

        cases = []
        start = 0
        for part, size in enumerate(split_sizes):
            end = start + int(size)
            selected = list(range(start, end))
            case = self.token_select([selected], [batch_id], validate_semantics=False)
            new_ori_length = (
                old_ori_length - old_q_length + end
                if self._has_position_sensitive_context()
                else old_ori_length
            )
            if self._has_position_sensitive_context() and new_ori_length < 0:
                raise InvalidTransformError(
                    f"Mode 3 part [{start},{end}) requires a negative aligned KV length"
                )
            self._align_position_sensitive_context(case.input_data, batch_id, new_ori_length)
            mapping = [(batch_id, token, 0, token - start) for token in selected]
            oracle = ActualInputSemanticOracle.validate_mapped_tokens(
                self.input_data, case.input_data, mapping
            )
            case.name = f"mode3_token_split_p{part}_{start}_{end}"
            case.transform_meta.update(
                {
                    "part_index": part,
                    "token_range": [start, end],
                    "split_sizes": list(split_sizes),
                    "complete_partition": True,
                    "semantic_oracle": oracle,
                }
            )
            cases.append(case)
            start = end
        return cases

    def shape_change_common_prefix(
        self,
        batch_id: int,
        common_tokens: int,
        derived_extra_tokens: int,
    ) -> ConsistencyCase:
        """Mode 4: change Q shape while comparing an unchanged common prefix."""
        valid_lengths = schema.get_q_lengths(self.input_data, valid_only=True)
        if batch_id < 0 or batch_id >= len(valid_lengths):
            raise ValueError(f"batch_id {batch_id} is outside [0, {len(valid_lengths)})")
        old_q_length = valid_lengths[batch_id]
        if common_tokens < 1 or common_tokens > old_q_length:
            raise ValueError(f"common_tokens must be within [1, {old_q_length}]")
        if derived_extra_tokens < 0:
            raise ValueError("derived_extra_tokens must be >= 0")
        target_length = common_tokens + derived_extra_tokens

        new_data = self.select_batches([batch_id])
        self._resize_query_prefix(new_data, target_length)
        old_ori_length = ActualInputSemanticOracle._lengths(self.input_data, "ori")[batch_id]
        new_ori_length = (
            old_ori_length + target_length - old_q_length
            if self._has_position_sensitive_context()
            else old_ori_length
        )
        if self._has_position_sensitive_context() and new_ori_length < 0:
            raise InvalidTransformError("Mode 4 requires a negative aligned KV length")
        self._align_position_sensitive_context(new_data, batch_id, new_ori_length)

        mapping = [(batch_id, token, 0, token) for token in range(common_tokens)]
        oracle = ActualInputSemanticOracle.validate_mapped_tokens(self.input_data, new_data, mapping)
        baseline_extra = old_q_length - common_tokens
        return ConsistencyCase(
            name=(
                f"mode4_shape_change_common_{common_tokens}_"
                f"extra_{baseline_extra}_to_{derived_extra_tokens}"
            ),
            input_data=new_data,
            output_origins=[TokenOrigin(batch_id, token) for token in range(common_tokens)],
            transform_meta={
                "mode": 4,
                "mode_name": "shape-change",
                "batch_id": batch_id,
                "common_tokens": common_tokens,
                "baseline_extra_tokens": baseline_extra,
                "derived_extra_tokens": derived_extra_tokens,
                "derived_compare_token_ids": list(range(common_tokens)),
                "layout_q": self.layout_q,
                "layout_kv": self.layout_kv,
                "operator": self.adapter.name,
                "semantic_oracle": oracle,
            },
        )

    def _select_query_tokens(
        self, tensors: Dict[str, Any], batch_ids: List[int], selections: List[List[int]]
    ) -> None:
        if self.layout_q == "BSND":
            tensors["q"] = self._select_bsnd_token_tensor(self.tensors["q"], batch_ids, selections, 1, 0)
            for sparse_name in ("ori_sparse_indices", "cmp_sparse_indices"):
                if self.tensors.get(sparse_name) is not None:
                    tensors[sparse_name] = self._select_bsnd_token_tensor(
                        self.tensors[sparse_name], batch_ids, selections, 2, -1
                    )
            for topk_name in ("ori_topk_length", "cmp_topk_length"):
                if self.tensors.get(topk_name) is not None:
                    tensors[topk_name] = self._select_bsnd_token_tensor(
                        self.tensors[topk_name], batch_ids, selections, 1, 0
                    )
            return

        cu_q = self.tensors["cu_seqlens_q"].to("cpu", torch.int64).tolist()
        for field_name in ("q", "ori_sparse_indices", "cmp_sparse_indices", "ori_topk_length", "cmp_topk_length"):
            source = self.tensors.get(field_name)
            if source is None:
                continue
            pieces = []
            for old_batch, selected in zip(batch_ids, selections):
                offset = cu_q[old_batch]
                index = torch.tensor([offset + token for token in selected], dtype=torch.long, device=source.device)
                pieces.append(source.index_select(0, index))
            tensors[field_name] = torch.cat(pieces, dim=0).contiguous()

    @staticmethod
    def _select_bsnd_token_tensor(
        tensor: torch.Tensor,
        batch_ids: List[int],
        selections: List[List[int]],
        token_dim: int,
        fill_value: int,
    ) -> torch.Tensor:
        max_tokens = max(len(selected) for selected in selections)
        shape = list(tensor.shape)
        shape[0] = len(batch_ids)
        shape[token_dim] = max_tokens
        result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
        for new_batch, (old_batch, selected) in enumerate(zip(batch_ids, selections)):
            source_index = torch.tensor(selected, dtype=torch.long, device=tensor.device)
            source = tensor[old_batch].index_select(token_dim - 1, source_index)
            target_slices = [new_batch] + [slice(None)] * (tensor.dim() - 1)
            target_slices[token_dim] = slice(0, len(selected))
            result[tuple(target_slices)] = source
        return result.contiguous()

    def _sync_token_lengths(self, adapter: CaseAdapter, selections: List[List[int]]) -> None:
        tensors = adapter.tensors
        metadata = adapter.metadata
        params = adapter.params
        lengths = [len(selected) for selected in selections]
        reference = self.tensors.get("seqused_q")
        if reference is None:
            reference = self.tensors.get("cu_seqlens_q")
        seqused = torch.tensor(
            lengths,
            dtype=reference.dtype if reference is not None else torch.int32,
            device=reference.device if reference is not None else None,
        )
        tensors["seqused_q"] = seqused
        metadata["seqused_q"] = seqused
        params["seqused_q"] = lengths

        if self.layout_q == "TND":
            cu_values = layouts.recompute_cu_seqlens(lengths)
        else:
            physical_s1 = int(tensors["q"].shape[1])
            cu_values = [index * physical_s1 for index in range(len(lengths) + 1)]
        if self.tensors.get("cu_seqlens_q") is not None or self.layout_q == "TND":
            cu_tensor = self._cu_tensor(cu_values, self.tensors.get("cu_seqlens_q"))
            tensors["cu_seqlens_q"] = cu_tensor
            metadata["cu_seqlens_q"] = cu_tensor
            params["cu_seqlens_q"] = cu_values


def generate_non_identity_permutation(batch_size: int, seed: Optional[int] = None) -> List[int]:
    if batch_size < 2:
        raise ValueError(f"Mode 1 requires B >= 2, got B={batch_size}")
    rng = random.Random(seed)
    while True:
        order = list(range(batch_size))
        rng.shuffle(order)
        if order != list(range(batch_size)):
            return order


def generate_split_groups(
    batch_size: int,
    num_groups: Optional[int] = None,
    seed: Optional[int] = None,
    randomize: bool = False,
) -> List[List[int]]:
    """Build a deterministic acceptance split, or an explicitly randomized fuzz split."""
    if batch_size < 2:
        raise ValueError(f"Mode 2 requires B >= 2, got B={batch_size}")
    if not randomize and num_groups is None and batch_size == 8:
        return groups_from_split_sizes(batch_size, [3, 5])
    if num_groups is None:
        num_groups = random.Random(seed).randint(2, batch_size) if randomize else 2
    if num_groups < 2 or num_groups > batch_size:
        raise ValueError(f"num_groups must satisfy 2 <= num_groups <= B ({batch_size}), got {num_groups}")

    indices = list(range(batch_size))
    if randomize:
        random.Random(seed).shuffle(indices)
    base_size, remainder = divmod(batch_size, num_groups)
    groups = []
    offset = 0
    for group_index in range(num_groups):
        size = base_size + (1 if group_index < remainder else 0)
        groups.append(indices[offset:offset + size])
        offset += size
    return groups


def groups_from_split_sizes(batch_size: int, split_sizes: Sequence[int]) -> List[List[int]]:
    """Convert explicit acceptance sizes such as B=8 -> 3+5 into batch ids."""
    if len(split_sizes) < 2 or any(size <= 0 for size in split_sizes):
        raise ValueError("split_sizes must contain at least two positive values")
    if sum(split_sizes) != batch_size:
        raise ValueError(
            f"split_sizes {list(split_sizes)} sum to {sum(split_sizes)}, expected B={batch_size}"
        )
    groups = []
    offset = 0
    for size in split_sizes:
        groups.append(list(range(offset, offset + int(size))))
        offset += int(size)
    return groups


def transform_mode1_reorder(
    input_data: Dict[str, Any], order: Optional[List[int]] = None, seed: Optional[int] = None
) -> ConsistencyCase:
    return BatchCaseTransformer(input_data).reorder(order, seed)


def transform_mode2_split(
    input_data: Dict[str, Any],
    groups: Optional[List[List[int]]] = None,
    seed: Optional[int] = None,
    randomize: bool = False,
    num_groups: Optional[int] = None,
) -> List[ConsistencyCase]:
    return BatchCaseTransformer(input_data).split(groups, seed, randomize, num_groups)


def transform_mode3_token_split(
    input_data: Dict[str, Any],
    token_ids_by_batch: Optional[List[List[int]]] = None,
    batch_ids: Optional[List[int]] = None,
    validate_semantics: bool = True,
) -> ConsistencyCase:
    return BatchCaseTransformer(input_data).token_select(token_ids_by_batch, batch_ids, validate_semantics)


def transform_mode3_token_partition(
    input_data: Dict[str, Any], batch_id: int, split_sizes: Sequence[int]
) -> List[ConsistencyCase]:
    return BatchCaseTransformer(input_data).token_partition(batch_id, split_sizes)


def transform_mode4_shape_change(
    input_data: Dict[str, Any],
    batch_id: int,
    common_tokens: int,
    derived_extra_tokens: int,
) -> ConsistencyCase:
    return BatchCaseTransformer(input_data).shape_change_common_prefix(
        batch_id,
        common_tokens,
        derived_extra_tokens,
    )


def transform_mode4_prepend_delete(*args: Any, **kwargs: Any) -> ConsistencyCase:
    """Deprecated compatibility alias for the primary common-prefix requirement."""
    return transform_mode4_shape_change(*args, **kwargs)
