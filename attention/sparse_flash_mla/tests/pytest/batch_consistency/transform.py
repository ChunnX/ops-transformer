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
    """Checks position-sensitive mask invariants before mode-3 execution."""

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


class BatchCaseTransformer:
    """Owns the coupled tensor/metadata updates for modes 1, 2 and 3."""

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


def transform_mode4_prepend_delete(*args: Any, **kwargs: Any) -> None:
    raise UnsupportedTransformError(
        "Mode 4 (shape-change) is not implemented. The STC says append tokens while the framework/meeting "
        "describes prepending right-aligned tokens; an actual-input golden oracle must resolve this conflict."
    )
