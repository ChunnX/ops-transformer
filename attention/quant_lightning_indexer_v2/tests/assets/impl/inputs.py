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

"""Input customization for QuantLightningIndexer V2 TTK cases."""

import importlib.util
import inspect
import sys
from pathlib import Path

import torch


QUANT_MODE_MXFP8 = 3
QUANT_MODE_MXFP4 = 5
MX_SCALE_SHAPE_ALIGN = 64
MX_SCALE_PACK_NUM = 2
FP4_PACK_NUM = 2


class QuantLightningIndexerV2InputAdapter:
    """Translate a TTK case and reuse the pytest input/golden generator."""

    @staticmethod
    def module_load_error(stage, path, exc):
        return RuntimeError(
            "Failed to load QuantLightningIndexerV2 module; "
            f"stage={stage}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    def __init__(self):
        self.pytest_golden = None
        self.pytest_normalizer = None

    @staticmethod
    def load_golden_store():
        name = "qli_v2_ttk_golden"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("golden.py")
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(name, None)
            raise QuantLightningIndexerV2InputAdapter.module_load_error(
                "assets Golden store", path, exc
            ) from exc
        return module

    def load_pytest_golden(self):
        if self.pytest_golden is not None:
            return self.pytest_golden
        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        path = pytest_dir / "quant_lightning_indexer_v2_golden.py"
        name = "qli_v2_pytest_golden"
        inserted = str(pytest_dir) not in sys.path
        if inserted:
            sys.path.insert(0, str(pytest_dir))
        try:
            if name in sys.modules:
                module = sys.modules[name]
            else:
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            self.pytest_golden = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error("pytest Golden", path, exc) from exc
        finally:
            if inserted:
                sys.path.remove(str(pytest_dir))

    def load_pytest_normalizer(self):
        if self.pytest_normalizer is not None:
            return self.pytest_normalizer
        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        path = pytest_dir / "qliv2_parameter_normalization.py"
        name = "qli_v2_pytest_normalizer"
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            self.pytest_normalizer = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error("pytest parameter normalizer", path, exc) from exc

    @staticmethod
    def list_value(kwargs, name):
        value = kwargs.get(f"{name}_values")
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in value]

    @staticmethod
    def prefix_lengths(value):
        if not value:
            return []
        return [int(value[index + 1]) - int(value[index])
                for index in range(len(value) - 1)]

    @staticmethod
    def data_range(input_ranges, index):
        if input_ranges and index < len(input_ranges) and input_ranges[index] is not None:
            return repr(list(input_ranges[index]))
        return None

    @staticmethod
    def qk_dtype_name(tensor, quant_mode):
        dtype_name = str(tensor.dtype)
        if quant_mode == QUANT_MODE_MXFP8:
            if "float8_e4m3fn" in dtype_name or tensor.dtype == torch.uint8:
                return "FLOAT8_E4M3FN"
            raise ValueError(
                "QLI_V2 MXFP8 q/k must use float8_e4m3fn or packed uint8, "
                f"got {tensor.dtype}"
            )
        if quant_mode == QUANT_MODE_MXFP4:
            if "float4_e2m1" in dtype_name or tensor.dtype == torch.uint8:
                return "FLOAT4_E2M1FN_X2"
            raise ValueError(
                "QLI_V2 MXFP4 q/k must use float4_e2m1fn_x2 or packed uint8, "
                f"got {tensor.dtype}"
            )
        if dtype_name == "torch.int8":
            return "INT8"
        if "float8_e4m3fn" in dtype_name:
            return "FLOAT8_E4M3FN"
        if dtype_name == "torch.uint8":
            return "HIFLOAT8"
        if "float4_e2m1" in dtype_name:
            raise ValueError("QLI_V2 float4 q/k requires quant_mode=5")
        raise ValueError(f"unsupported QLI_V2 q/k dtype: {tensor.dtype}")

    @staticmethod
    def dequant_dtype_name(tensor, quant_mode):
        dtype_name = str(tensor.dtype)
        if "float8_e8m0" in dtype_name:
            return "FLOAT8_E8M0FNU"
        if quant_mode in (QUANT_MODE_MXFP8, QUANT_MODE_MXFP4):
            if tensor.dtype == torch.uint8:
                return "FLOAT8_E8M0FNU"
            raise ValueError(
                "QLI_V2 MXFP8/MXFP4 dequant scale must use float8_e8m0fnu "
                f"or packed uint8, got {tensor.dtype}"
            )
        if tensor.dtype == torch.float16:
            return "FP16"
        if tensor.dtype == torch.float32:
            return "FP32"
        raise ValueError(f"unsupported QLI_V2 dequant dtype: {tensor.dtype}")

    @staticmethod
    def logical_head_dim(physical_head_dim, quant_mode):
        if quant_mode == QUANT_MODE_MXFP4:
            return physical_head_dim * FP4_PACK_NUM
        return physical_head_dim

    @staticmethod
    def uses_byte_storage(dtype):
        dtype_name = str(dtype)
        return "float4" in dtype_name or "float8" in dtype_name

    @staticmethod
    def semantic_mx_tensor(tensor, quant_mode, is_scale=False):
        if tensor is None:
            return None
        result = tensor.detach().cpu()
        if result.dtype != torch.uint8:
            return result
        if is_scale and quant_mode in (QUANT_MODE_MXFP8, QUANT_MODE_MXFP4):
            e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
            if e8m0_dtype is None:
                raise RuntimeError("current PyTorch does not provide float8_e8m0fnu")
            return result.contiguous().view(e8m0_dtype)
        if not is_scale and quant_mode == QUANT_MODE_MXFP8:
            mxfp8_dtype = getattr(torch, "float8_e4m3fn", None)
            if mxfp8_dtype is None:
                raise RuntimeError("current PyTorch does not provide float8_e4m3fn")
            return result.contiguous().view(mxfp8_dtype)
        if not is_scale and quant_mode == QUANT_MODE_MXFP4:
            fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
            if fp4_dtype is None:
                raise RuntimeError("current PyTorch does not provide float4_e2m1fn_x2")
            return result.contiguous().view(fp4_dtype)
        return result

    @classmethod
    def validate_mx_scale(cls, scale, data, quant_mode, name):
        if quant_mode not in (QUANT_MODE_MXFP8, QUANT_MODE_MXFP4):
            return
        head_dim = cls.logical_head_dim(int(data.shape[-1]), quant_mode)
        if head_dim % MX_SCALE_SHAPE_ALIGN != 0:
            raise ValueError(
                f"{name} logical head_dim must be divisible by "
                f"{MX_SCALE_SHAPE_ALIGN}, got {head_dim}"
            )
        expected_shape = tuple(data.shape[:-1]) + (
            head_dim // MX_SCALE_SHAPE_ALIGN,
            MX_SCALE_PACK_NUM,
        )
        if tuple(scale.shape) != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, "
                f"got {tuple(scale.shape)}"
            )
        scale_dtype_name = str(scale.dtype)
        if "float8_e8m0" not in scale_dtype_name and scale.dtype != torch.uint8:
            raise ValueError(
                f"{name} must use float8_e8m0fnu or packed uint8, got {scale.dtype}"
            )

    @staticmethod
    def weight_dtype_name(tensor):
        mapping = {
            torch.int8: "INT8",
            torch.uint8: "UINT8",
            torch.float16: "FP16",
            torch.float32: "FP32",
            torch.bfloat16: "BF16",
        }
        if tensor.dtype in mapping:
            return mapping[tensor.dtype]
        if "float8_e4m3fn" in str(tensor.dtype):
            return "FLOAT8_E4M3FN"
        raise ValueError(f"unsupported QLI_V2 weight dtype: {tensor.dtype}")

    @staticmethod
    def pytest_uses_weight_dtype(pytest_golden):
        parameters = inspect.signature(
            pytest_golden.GeneralizedQLIV2.__init__
        ).parameters
        return "weight_dtype" in parameters

    def geometry(self, query, key, layout_query, layout_key,
                 cu_q, cu_k, seq_q, seq_k, quant_mode):
        q_lengths = self.prefix_lengths(cu_q) or (seq_q or [])
        k_lengths = self.prefix_lengths(cu_k) or (seq_k or [])
        if layout_query == "BSND":
            batch_size, q_seq, q_head_num, physical_head_dim = [int(item) for item in query.shape]
            q_t_size = 0
        elif layout_query == "TND":
            q_t_size, q_head_num, physical_head_dim = [int(item) for item in query.shape]
            batch_size = len(q_lengths)
            q_seq = max(q_lengths, default=q_t_size)
        else:
            raise ValueError(f"unsupported QLI_V2 query layout: {layout_query}")

        head_dim = self.logical_head_dim(physical_head_dim, quant_mode)

        if layout_key == "BSND":
            _, k_seq, k_head_num, key_physical_head_dim = [int(item) for item in key.shape]
            k_t_size = 0
            block_size = 0
            block_num = 0
        elif layout_key == "TND":
            k_t_size, k_head_num, key_physical_head_dim = [int(item) for item in key.shape]
            k_seq = max(k_lengths, default=k_t_size)
            block_size = 0
            block_num = 0
        elif layout_key == "PA_BBND":
            block_num, block_size, k_head_num, key_physical_head_dim = [int(item) for item in key.shape]
            capacity = block_num * block_size
            per_batch_capacity = capacity // batch_size if batch_size > 0 else block_size
            k_seq = max(k_lengths, default=per_batch_capacity)
            k_t_size = 0
        else:
            raise ValueError(f"unsupported QLI_V2 key layout: {layout_key}")
        if key_physical_head_dim != physical_head_dim:
            raise ValueError(
                "QLI_V2 query/key physical head_dim mismatch: "
                f"query={physical_head_dim}, key={key_physical_head_dim}"
            )
        return (
            batch_size, q_seq, k_seq, q_t_size, k_t_size,
            q_head_num, k_head_num, head_dim, block_size, block_num,
        )

    def build_case_params(self, query, key, weights, query_dequant_scale,
                          layout_query, layout_key, kwargs,
                          include_weight_dtype=False):
        cu_q = self.list_value(kwargs, "cu_seqlens_q")
        cu_k = self.list_value(kwargs, "cu_seqlens_k")
        seq_q = self.list_value(kwargs, "seqused_q")
        seq_k = self.list_value(kwargs, "seqused_k")
        residual = self.list_value(kwargs, "cmp_residual_k")
        quant_mode = kwargs.get("quant_mode")
        quant_mode = 1 if quant_mode is None else int(quant_mode)
        query_dtype_name = self.qk_dtype_name(query, quant_mode)
        key_dtype_name = self.qk_dtype_name(key, quant_mode)
        if query_dtype_name != key_dtype_name:
            raise ValueError(
                f"QLI_V2 query/key dtype mismatch: query={query.dtype}, key={key.dtype}"
            )
        self.validate_mx_scale(
            query_dequant_scale, query, quant_mode, "query_dequant_scale"
        )
        geometry = self.geometry(
            query, key, layout_query, layout_key, cu_q, cu_k, seq_q, seq_k,
            quant_mode,
        )
        input_ranges = kwargs.get("qli_input_ranges") or kwargs.get("input_ranges") or ()
        output_range = kwargs.get("output_idx_offset_range")
        output_range = None if output_range is None else repr(list(output_range))
        max_seqlen_q = kwargs.get("max_seqlen_q")
        max_seqlen_q = None if max_seqlen_q is None else int(max_seqlen_q)
        dtype_params = (query_dtype_name,)
        if include_weight_dtype:
            dtype_params += (self.weight_dtype_name(weights),)
        dtype_params += (
            self.dequant_dtype_name(query_dequant_scale, quant_mode),
            "INT32",
        )
        params = geometry + dtype_params + (
            cu_q,
            cu_k,
            seq_q,
            seq_k,
            residual,
            max_seqlen_q,
            quant_mode,
            layout_query,
            layout_key,
            kwargs.get("sparse_count"),
            kwargs.get("sparse_mode"),
            self.data_range(input_ranges, 0),
            self.data_range(input_ranges, 1),
            self.data_range(input_ranges, 2),
            self.data_range(input_ranges, 3),
            self.data_range(input_ranges, 4),
            kwargs.get("cmp_ratio"),
            kwargs.get("return_value"),
            output_range,
        )
        return self.load_pytest_normalizer().normalize_qliv2_params(params)

    @staticmethod
    def copy_tensor(dst, src, name, packed_dtype=None):
        if dst is None:
            if src is not None:
                raise ValueError(
                    f"{name} is absent from CSV but pytest generator produced a tensor"
                )
            return
        if src is None:
            raise ValueError(f"{name} is present in CSV but pytest generator returned None")
        src_cpu = src.detach().cpu() if torch.is_tensor(src) else torch.as_tensor(src)
        src_cpu = src_cpu.contiguous()
        if packed_dtype is not None and dst.dtype == torch.uint8:
            src_cpu = src_cpu.to(packed_dtype).contiguous().view(torch.uint8)
        if tuple(dst.shape) != tuple(src_cpu.shape):
            raise ValueError(
                f"{name} shape mismatch: TTK={tuple(dst.shape)} "
                f"pytest={tuple(src_cpu.shape)}"
            )
        raw_byte_copy = (
            src_cpu.element_size() == 1
            and dst.element_size() == 1
            and (
                QuantLightningIndexerV2InputAdapter.uses_byte_storage(src_cpu.dtype)
                or QuantLightningIndexerV2InputAdapter.uses_byte_storage(dst.dtype)
            )
        )
        if raw_byte_copy:
            src_storage = src_cpu.view(torch.uint8)
            dst_storage = dst.view(torch.uint8)
            if src_storage.numel() != dst_storage.numel():
                raise ValueError(
                    f"{name} storage size mismatch: TTK={dst_storage.numel()} "
                    f"pytest={src_storage.numel()}"
                )
            dst_storage.copy_(src_storage.to(device=dst.device))
            return
        dst.copy_(src_cpu.to(dtype=dst.dtype, device=dst.device))

    @staticmethod
    def golden_data(data, layout_query, cu_seqlens_q):
        cu_q = None
        if cu_seqlens_q is not None:
            cu_q = cu_seqlens_q.detach().cpu().clone()
        return {
            "cpu_result": data["cpu_result"],
            "topk_value": data["topk_value"],
            "cpu_topk_value": data["cpu_topk_value"],
            "params": data["params"],
            "output_idx_offset": data.get("output_idx_offset"),
            "score_layout": layout_query,
            "cu_seqlens_q": cu_q,
        }

    @staticmethod
    def tensor_values(tensor):
        if tensor is None:
            return None
        return [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]

    @classmethod
    def restore_paged_tensor(cls, tensor, block_table, batch_size, sequence_length):
        """Restore a paged key or scale tensor to its logical BNS(D) layout."""
        physical = tensor.detach().cpu()
        table = block_table.detach().cpu().to(torch.int64)
        if physical.ndim < 3:
            raise ValueError(
                f"paged tensor must have at least 3 dimensions, got {tuple(physical.shape)}"
            )
        block_size, head_num = int(physical.shape[1]), int(physical.shape[2])
        trailing = tuple(int(dim) for dim in physical.shape[3:])
        use_storage_view = cls.uses_byte_storage(physical.dtype)
        physical_storage = (
            physical.contiguous().view(torch.uint8) if use_storage_view else physical
        )
        logical_storage = torch.zeros(
            (batch_size, head_num, sequence_length, *trailing),
            dtype=physical_storage.dtype,
        )
        for batch_idx in range(batch_size):
            for logical_block, block_id_value in enumerate(table[batch_idx].tolist()):
                if block_id_value < 0:
                    continue
                if block_id_value >= physical_storage.shape[0]:
                    raise ValueError(f"block id {block_id_value} exceeds paged block count")
                start = logical_block * block_size
                if start >= sequence_length:
                    break
                count = min(block_size, sequence_length - start)
                block = physical_storage[block_id_value, :count]
                permutation = (1, 0, *range(2, block.ndim))
                logical_storage[batch_idx, :, start:start + count] = block.permute(*permutation)
        if use_storage_view:
            return logical_storage.view(physical.dtype)
        return logical_storage

    @staticmethod
    def normalize_compare_attributes(compare_context):
        attrs = dict(compare_context.attributes)
        aliases = {
            "topk": "sparse_count",
            "quantMode": "quant_mode",
            "maxSeqlenQ": "max_seqlen_q",
            "layoutQOptional": "layout_query",
            "layoutKOptional": "layout_key",
            "maskMode": "sparse_mode",
            "cmpRatio": "cmp_ratio",
            "returnValue": "return_value",
        }
        for source, target in aliases.items():
            if target not in attrs and source in attrs:
                attrs[target] = attrs[source]
        return attrs

    def rebuild_compare_data(self, compare_context):
        """Rebuild replay-safe TopK score context from restored API inputs."""
        tensors = tuple(compare_context.input_tensors or ())
        if len(tensors) < 12:
            raise ValueError("QuantLightningIndexerV2 compare context requires twelve input slots")
        (
            query, key, weights, query_scale, key_scale,
            cu_q, cu_k, seq_q, seq_k, residual, block_table, offset,
        ) = tensors[:12]
        attrs = self.normalize_compare_attributes(compare_context)
        for name, tensor in (
            ("cu_seqlens_q", cu_q),
            ("cu_seqlens_k", cu_k),
            ("seqused_q", seq_q),
            ("seqused_k", seq_k),
            ("cmp_residual_k", residual),
        ):
            values = self.tensor_values(tensor)
            if values is not None:
                attrs[f"{name}_values"] = values

        layout_q = attrs.get("layout_query")
        layout_k = attrs.get("layout_key")
        if layout_q is None or layout_k is None:
            raise ValueError(
                "QLI_V2 replay compare requires layout_query and layout_key from the CSV or API attributes"
            )
        pytest_golden = self.load_pytest_golden()
        uses_weight_dtype = self.pytest_uses_weight_dtype(pytest_golden)
        params = self.build_case_params(
            query,
            key,
            weights,
            query_scale,
            layout_q,
            layout_k,
            attrs,
            include_weight_dtype=uses_weight_dtype,
        )
        if uses_weight_dtype:
            (
                batch_size, q_seq, k_seq, q_t_size, k_t_size,
                q_head_num, k_head_num, head_dim, block_size, block_num,
                _, _, _, _, cu_q_values, cu_k_values, seq_q_values,
                seq_k_values, residual_values, max_seqlen_q, quant_mode,
                _, _, sparse_count, sparse_mode, _, _, _, _, _, cmp_ratio,
                return_value, _,
            ) = params
        else:
            (
                batch_size, q_seq, k_seq, q_t_size, k_t_size,
                q_head_num, k_head_num, head_dim, block_size, block_num,
                _, _, _, cu_q_values, cu_k_values, seq_q_values, seq_k_values,
                residual_values, max_seqlen_q, quant_mode, _, _, sparse_count,
                sparse_mode, _, _, _, _, _, cmp_ratio, return_value, _,
            ) = params

        q_lengths = (
            self.prefix_lengths(cu_q_values)
            if layout_q == "TND"
            else (seq_q_values or [q_seq] * batch_size)
        )
        k_lengths = (
            self.prefix_lengths(cu_k_values)
            if layout_k == "TND"
            else (seq_k_values or [k_seq] * batch_size)
        )
        residual_for_cpu = (
            [0] * batch_size
            if cmp_ratio == 1 or sparse_mode == 0
            else list(residual_values)
        )

        def as_int_tensor(value):
            return None if value is None else torch.tensor(value, dtype=torch.int32)

        cu_q_cpu = as_int_tensor(cu_q_values)
        cu_k_cpu = as_int_tensor(cu_k_values)
        seq_q_cpu = as_int_tensor(seq_q_values)
        seq_k_cpu = as_int_tensor(seq_k_values)
        self.validate_mx_scale(key_scale, key, quant_mode, "key_dequant_scale")
        if (self.dequant_dtype_name(key_scale, quant_mode)
                != self.dequant_dtype_name(query_scale, quant_mode)):
            raise ValueError(
                "QLI_V2 query/key dequant scale dtype mismatch: "
                f"query={query_scale.dtype}, key={key_scale.dtype}"
            )

        query_for_cpu = self.semantic_mx_tensor(query, quant_mode)
        key_for_cpu = self.semantic_mx_tensor(key, quant_mode)
        weights_for_cpu = weights.detach().cpu()
        query_scale_for_cpu = self.semantic_mx_tensor(
            query_scale, quant_mode, is_scale=True
        )
        key_scale_for_cpu = self.semantic_mx_tensor(
            key_scale, quant_mode, is_scale=True
        )
        block_table_for_cpu = (
            None if block_table is None else block_table.detach().cpu()
        )
        offset_for_cpu = None if offset is None else offset.detach().cpu()
        model_args = [
            batch_size, q_seq, k_seq, q_t_size, k_t_size,
            q_head_num, k_head_num, head_dim, block_size, block_num,
            query_for_cpu.dtype,
        ]
        if uses_weight_dtype:
            model_args.append(weights_for_cpu.dtype)
        model_args.extend([
            query_scale_for_cpu.dtype, torch.int32,
            cu_q_cpu, cu_k_cpu, q_lengths, k_lengths, residual_for_cpu,
            max_seqlen_q, quant_mode, layout_q, layout_k, sparse_count,
            sparse_mode, None, None, None, None, None,
            cmp_ratio, return_value,
        ])
        model = pytest_golden.GeneralizedQLIV2(*model_args)

        if layout_k == "PA_BBND":
            if block_table_for_cpu is None or not k_lengths:
                raise ValueError("PA_BBND compare context requires block_table and seqused_k")
            sequence_length = max(k_lengths)
            key_for_cpu = self.restore_paged_tensor(
                key_for_cpu, block_table_for_cpu, batch_size, sequence_length
            )
            if quant_mode != 4:
                key_scale_for_cpu = self.restore_paged_tensor(
                    key_scale_for_cpu, block_table_for_cpu,
                    batch_size, sequence_length,
                )

        if quant_mode == 4:
            query_scale_for_cpu = torch.full(
                tuple(query_for_cpu.shape[:-1]),
                query_scale_for_cpu.reshape(-1)[0].item(),
                dtype=query_scale_for_cpu.dtype,
            )
            if layout_k == "PA_BBND":
                key_scale_shape = (batch_size, k_head_num, max(k_lengths))
            else:
                key_scale_shape = tuple(key.shape[:-1])
            key_scale_for_cpu = torch.full(
                key_scale_shape,
                key_scale_for_cpu.reshape(-1)[0].item(),
                dtype=key_scale_for_cpu.dtype,
            )

        _, scores, _ = model.forward(
            query_for_cpu, key_for_cpu, weights_for_cpu,
            query_scale_for_cpu, key_scale_for_cpu,
            cu_q_cpu, cu_k_cpu, seq_q_cpu, seq_k_cpu,
            block_table_for_cpu, offset_for_cpu,
        )
        return {
            "params": params,
            "scores": scores,
            "topk_value": scores,
            "output_idx_offset": offset_for_cpu,
            "score_layout": layout_q,
            "cu_seqlens_q": cu_q_cpu,
        }

    def customize(self, query, key, weights, query_dequant_scale,
                  key_dequant_scale, cu_seqlens_q, cu_seqlens_k,
                  seqused_q, seqused_k, cmp_residual_k, block_table,
                  output_idx_offset, layout_query, layout_key, kwargs):
        pytest_golden = self.load_pytest_golden()
        uses_weight_dtype = self.pytest_uses_weight_dtype(pytest_golden)
        params = self.build_case_params(
            query,
            key,
            weights,
            query_dequant_scale,
            layout_query,
            layout_key,
            kwargs,
            include_weight_dtype=uses_weight_dtype,
        )
        quant_mode = params[20 if uses_weight_dtype else 19]
        self.validate_mx_scale(
            key_dequant_scale, key, quant_mode, "key_dequant_scale"
        )
        if (self.dequant_dtype_name(key_dequant_scale, quant_mode)
                != self.dequant_dtype_name(query_dequant_scale, quant_mode)):
            raise ValueError(
                "QLI_V2 query/key dequant scale dtype mismatch: "
                f"query={query_dequant_scale.dtype}, key={key_dequant_scale.dtype}"
            )
        golden_store = self.load_golden_store().CASE_DATA
        golden_store.clear()
        data = pytest_golden.generate_qliv2_test_data(params)
        for name, dst, src_name in (
            ("query", query, "query"),
            ("key", key, "key"),
            ("weights", weights, "weights"),
            ("query_dequant_scale", query_dequant_scale, "query_dequant_scale"),
            ("key_dequant_scale", key_dequant_scale, "key_dequant_scale"),
            ("cu_seqlens_q", cu_seqlens_q, "cu_seqlens_query"),
            ("cu_seqlens_k", cu_seqlens_k, "cu_seqlens_key"),
            ("seqused_q", seqused_q, "seqused_q"),
            ("seqused_k", seqused_k, "seqused_k"),
            ("cmp_residual_k", cmp_residual_k, "cmp_residual_k_for_npu"),
            ("block_table", block_table, "block_table"),
            ("output_idx_offset", output_idx_offset, "output_idx_offset"),
        ):
            packed_dtype = None
            if (
                quant_mode == QUANT_MODE_MXFP8
                and name in ("query", "key")
                and dst is not None
                and dst.dtype == torch.uint8
            ):
                packed_dtype = getattr(torch, "float8_e4m3fn", None)
                if packed_dtype is None:
                    raise RuntimeError("current PyTorch does not provide float8_e4m3fn")
            self.copy_tensor(dst, data.get(src_name), name, packed_dtype)
        testcase_name = kwargs.get("testcase_name")
        golden_store.put(
            testcase_name,
            self.golden_data(data, layout_query, cu_seqlens_q),
        )
        return data


INPUT_ADAPTER = QuantLightningIndexerV2InputAdapter()


def rebuild_qli_v2_compare_data(compare_context):
    return INPUT_ADAPTER.rebuild_compare_data(compare_context)


def generate_qli_v2_inputs(query, key, weights, query_dequant_scale,
                           key_dequant_scale, *, cu_seqlens_q=None,
                           cu_seqlens_k=None, seqused_q=None, seqused_k=None,
                           cmp_residual_k=None, block_table=None,
                           output_idx_offset=None, layout_query=None,
                           layout_key=None, **kwargs):
    """Generate the exact pytest inputs and CPU golden for a TTK case."""
    INPUT_ADAPTER.customize(
        query,
        key,
        weights,
        query_dequant_scale,
        key_dequant_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        layout_query,
        layout_key,
        kwargs,
    )
