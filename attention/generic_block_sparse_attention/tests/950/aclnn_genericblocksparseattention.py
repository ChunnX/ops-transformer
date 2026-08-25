import ctypes
import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

from fd_case_specs import fd_case_spec_from_shapes
from fd_metadata_v8 import (
    ARCH35_AIC_NUM,
    METADATA_TOTAL_SIZE,
    FdPlan,
    print_fd_plan,
    simulate_aicpu_metadata_v8,
)


_PRINTED_FD_PLAN_KEYS = set()


def print_fd_plan_once(plan):
    key = (
        plan.config_signature,
        plan.total_split_tasks,
        plan.block_prefix,
        plan.enabled,
    )
    if key in _PRINTED_FD_PLAN_KEYS:
        return
    _PRINTED_FD_PLAN_KEYS.add(key)
    print_fd_plan(plan, show_cores=True)


def safe_to_tensor(arr):
    if arr is None:
        return None
    if isinstance(arr, list):
        arr = np.array(arr)
    if isinstance(arr, torch.Tensor):
        return arr
    if hasattr(arr, "dtype") and arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(arr)


# Real AICPU metadata inline (same path as pytest / SMLA): max_* from cu storage.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TORCH_EXT = _REPO_ROOT / "torch_extension"
_META_OP = None


def _max_seq_len_from_cu(cu, fallback):
    if cu is None:
        return int(max(fallback))
    vals = cu.detach().cpu().tolist() if hasattr(cu, "detach") else list(cu)
    if len(vals) < 2:
        return int(max(fallback))
    return max(int(vals[i + 1]) - int(vals[i]) for i in range(len(vals) - 1))


def _get_gbsa_metadata_op():
    global _META_OP
    if _META_OP is not None:
        return _META_OP
    if str(_TORCH_EXT) not in sys.path:
        sys.path.insert(0, str(_TORCH_EXT))
    spec = importlib.util.spec_from_file_location(
        "generic_block_sparse_attention_metadata",
        _TORCH_EXT
        / "cann_ops_transformer"
        / "ops"
        / "generic_block_sparse_attention_metadata.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _META_OP = mod.npu_generic_block_sparse_attention_metadata
    return _META_OP


def prefix_sum_from_seqlens(seqlens):
    out = [0]
    for s in seqlens:
        out.append(out[-1] + int(s))
    return out


def _optional_int_vec(t, batch):
    if t is None:
        return None
    t = safe_to_tensor(t)
    if t is None:
        return None
    t = t.cpu().to(torch.int64).view(-1)
    if t.numel() != batch:
        return None
    return [int(x) for x in t.tolist()]


def decode_cu_and_seqused(cu_q, cu_kv, seqused_q=None, seqused_kv=None, total_q=None):
    """Decode storage (cu) and actual (seqused if present) per-batch lengths."""
    if cu_q is None or cu_kv is None:
        return None
    cu_q = safe_to_tensor(cu_q).cpu().to(torch.int64).view(-1)
    cu_kv = safe_to_tensor(cu_kv).cpu().to(torch.int64).view(-1)
    if cu_q.numel() < 2 or cu_kv.numel() != cu_q.numel():
        return None
    if int(cu_q[0].item()) != 0 or int(cu_kv[0].item()) != 0:
        return None
    batch = cu_q.numel() - 1
    storage_q = [(cu_q[i + 1] - cu_q[i]).item() for i in range(batch)]
    storage_kv = [(cu_kv[i + 1] - cu_kv[i]).item() for i in range(batch)]
    if any(s < 0 for s in storage_q) or any(s < 0 for s in storage_kv):
        return None
    if total_q is not None and sum(storage_q) != int(total_q):
        return None
    used_q = _optional_int_vec(seqused_q, batch)
    used_kv = _optional_int_vec(seqused_kv, batch)
    actual_q = used_q if used_q is not None else storage_q
    actual_kv = used_kv if used_kv is not None else storage_kv
    if any(a < 0 or a > s for a, s in zip(actual_q, storage_q)):
        return None
    if any(a < 0 or a > s for a, s in zip(actual_kv, storage_kv)):
        return None
    # actual==0 allowed (padding / empty request); only reject when both >0 and kv < q.
    if any(q > 0 and kv > 0 and kv < q for q, kv in zip(actual_q, actual_kv)):
        return None
    return actual_q, actual_kv, storage_q, storage_kv


def seqlens_from_cu_seq(cu_q, cu_kv, total_q=None, seqused_q=None, seqused_kv=None):
    """Actual seqlens (seqused if present, else cu). None if invalid."""
    decoded = decode_cu_and_seqused(
        cu_q, cu_kv, seqused_q=seqused_q, seqused_kv=seqused_kv, total_q=total_q
    )
    if decoded is None:
        return None
    return decoded[0], decoded[1]


def recover_batch_seqlens(query, sparse_block_idx, block_table, metadata=None):
    """Equal-batch seqlens: q=T/B, kv=maxBlocks*128 (metadata is INT32[1024], not kv len)."""
    del sparse_block_idx, metadata
    batch = int(block_table.shape[0])
    total_q = int(query.shape[0])
    max_blocks = int(block_table.shape[1])
    if batch <= 0 or total_q % batch != 0:
        raise ValueError(
            f"Cannot recover equal-batch q seqlens: T={total_q}, B={batch}"
        )
    q_seqlen = total_q // batch
    kv_seqlen = max(max_blocks * 128, q_seqlen)
    return [q_seqlen] * batch, [kv_seqlen] * batch


def generate_block_table(
    batch, max_blocks_per_batch, num_physical, seed=10, identity=False
):
    """logical->physical map. Smoke cases use identity; otherwise shuffle like test_bf16."""
    need = batch * max_blocks_per_batch
    if need > num_physical:
        raise ValueError(f"need {need} physical ids but key only has {num_physical}")
    block_table = torch.full((batch, max_blocks_per_batch), -1, dtype=torch.int32)
    if identity or need == num_physical == 1:
        for b in range(batch):
            for i in range(max_blocks_per_batch):
                block_table[b, i] = b * max_blocks_per_batch + i
        return block_table
    rng = random.Random(seed)
    all_physical = list(range(num_physical))
    rng.shuffle(all_physical)
    chosen = all_physical[:need]
    for b in range(batch):
        for i in range(max_blocks_per_batch):
            block_table[b, i] = chosen[b * max_blocks_per_batch + i]
    return block_table


def build_sparse_and_block_table(
    q_seqlens,
    kv_seqlens,
    kv_heads,
    top_k,
    num_physical,
    block_shape_y=128,
    seed=10,
    q_storage_seqlens=None,
    sparse_count_pattern=None,
):
    """Build TND packed sparseBlockIdx/Count [Nkv, totalQBlocks, ...] and blockTable.

    totalQBlocks follows storage (cu). Only actual q tokens (seqused) get filled rows.
    Smoke (max_blocks==1, topK==1): always select logical block 0, identity blockTable.
    """
    np.random.seed(seed)
    random.seed(seed)
    batch = len(q_seqlens)
    if q_storage_seqlens is None:
        q_storage_seqlens = q_seqlens
    total_q_blocks = sum(int(s) for s in q_storage_seqlens)
    max_blocks = max((kv + block_shape_y - 1) // block_shape_y for kv in kv_seqlens)
    top_k = min(top_k, max_blocks)
    smoke = max_blocks == 1 and top_k == 1

    sparse_idx = torch.full((kv_heads, total_q_blocks, top_k), -1, dtype=torch.int32)
    sparse_count = torch.zeros((kv_heads, total_q_blocks), dtype=torch.int32)
    block_table = generate_block_table(
        batch, max_blocks, num_physical, seed=seed, identity=smoke
    )

    expected_pattern_size = sum(int(s) for s in q_seqlens) * int(kv_heads)
    if sparse_count_pattern is not None:
        sparse_count_pattern = [int(value) for value in sparse_count_pattern]
        if len(sparse_count_pattern) != expected_pattern_size:
            raise ValueError(
                "sparse_count_pattern must provide one count per actual Q/KV-head task"
            )
    pattern_idx = 0
    q_storage = 0
    for b in range(batch):
        if int(q_seqlens[b]) == 0 or int(kv_seqlens[b]) == 0:
            q_storage += int(q_storage_seqlens[b])
            continue
        history_len = kv_seqlens[b] - q_seqlens[b]
        for q_block in range(q_seqlens[b]):
            global_q = q_storage + q_block
            last_logical = (history_len + q_block) // block_shape_y
            candidates = list(range(last_logical + 1))
            if not candidates:
                pattern_idx += kv_heads
                continue
            for h in range(kv_heads):
                if sparse_count_pattern is not None:
                    num_select = sparse_count_pattern[pattern_idx]
                    if num_select < 0 or num_select > min(len(candidates), top_k):
                        raise ValueError(
                            f"deterministic sparse count {num_select} is outside "
                            f"[0, {min(len(candidates), top_k)}]"
                        )
                    if num_select == 0:
                        selected = []
                    else:
                        must = last_logical
                        others = [c for c in candidates if c != must]
                        selected = sorted(others[:num_select - 1] + [must])
                elif smoke:
                    selected = [0]
                else:
                    must = last_logical
                    others = [c for c in candidates if c != must]
                    num_select = random.randint(1, min(len(candidates), top_k))
                    selected = [must]
                    remain = num_select - 1
                    if remain > 0 and others:
                        selected.extend(random.sample(others, min(remain, len(others))))
                    selected = sorted(set(selected))[:top_k]
                pattern_idx += 1
                sparse_count[h, global_q] = len(selected)
                for j, lid in enumerate(selected):
                    sparse_idx[h, global_q, j] = lid
        q_storage += int(q_storage_seqlens[b])

    return sparse_idx, sparse_count, block_table


def apply_init_tensors(input_data, device="cpu"):
    """Build legal sparse inputs and the complete simulated Metadata-v8 plan."""
    query = safe_to_tensor(input_data.kwargs["query"])
    key = safe_to_tensor(input_data.kwargs["key"])
    sparse_block_idx = safe_to_tensor(input_data.kwargs["sparseBlockIdx"])
    block_table = safe_to_tensor(input_data.kwargs["blockTableOptional"])
    storage_q, storage_kv = recover_batch_seqlens(
        query,
        sparse_block_idx,
        block_table,
    )
    kv_heads = int(sparse_block_idx.shape[0])
    top_k = int(sparse_block_idx.shape[2])
    num_physical = int(key.shape[0])
    # Vary sparse pattern by case shape (still deterministic).
    seed = (
        10
        + int(query.shape[0]) * 17
        + int(query.shape[1]) * 31
        + num_physical * 13
        + top_k * 7
        + int(block_table.shape[0]) * 3
        + int(block_table.shape[1]) * 5
    )
    test_mode = os.environ.get("TEST_MODE", "normal").lower()
    sparse_count_pattern = None
    if test_mode == "fd":
        spec = fd_case_spec_from_shapes(
            batch=len(storage_q),
            max_blocks=int(block_table.shape[1]),
            num_physical=num_physical,
        )
        if len(storage_q) != 1 or int(storage_q[0]) != int(spec["q_storage"]):
            raise ValueError(
                f"FD Case {spec['name']} storage Q shape does not match its template"
            )
        actual_q = [int(spec["q_actual"])]
        sparse_count_pattern = list(spec["counts"])
        use_seqused = actual_q != [int(value) for value in storage_q]
    else:
        # Keep the original randomized normal-path padding coverage.
        use_seqused = all(int(s) >= 2 for s in storage_q) and (seed % 10 == 0)
        actual_q = (
            [int(s) - 1 for s in storage_q]
            if use_seqused
            else [int(s) for s in storage_q]
        )
    actual_kv = [int(s) for s in storage_kv]

    sparse_idx, sparse_count, block_table_new = build_sparse_and_block_table(
        actual_q,
        actual_kv,
        kv_heads,
        top_k,
        num_physical=num_physical,
        block_shape_y=128,
        seed=seed,
        q_storage_seqlens=storage_q,
        sparse_count_pattern=sparse_count_pattern,
    )
    cu_q = torch.tensor(prefix_sum_from_seqlens(storage_q), dtype=torch.int64)
    cu_kv = torch.tensor(prefix_sum_from_seqlens(storage_kv), dtype=torch.int64)
    seqused_q = torch.tensor(actual_q, dtype=torch.int32) if use_seqused else None
    seqused_kv = torch.tensor(actual_kv, dtype=torch.int32) if use_seqused else None

    num_heads = int(query.shape[1])
    head_dim = int(query.shape[-1])
    is_packed_gqa = int(input_data.kwargs.get("isPackedGQA", 1))
    mask_type = int(input_data.kwargs.get("maskType", 1))
    quant_type = int(input_data.kwargs.get("quantType", 0))
    softmax_precision = int(input_data.kwargs.get("softmaxPrecision", 1))
    block_shape = input_data.kwargs.get("blockShape", [1, 128])
    if hasattr(block_shape, "tolist"):
        block_shape = block_shape.tolist()
    block_shape = [int(block_shape[0]), int(block_shape[1])]

    metadata, plan = simulate_aicpu_metadata_v8(
        sparse_block_idx=sparse_idx,
        sparse_block_count=sparse_count,
        cu_seq_lengths_q=cu_q,
        cu_seq_lengths_kv=cu_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        max_q_seq_len=_max_seq_len_from_cu(cu_q, actual_q),
        max_kv_seq_len=_max_seq_len_from_cu(cu_kv, actual_kv),
        num_q_heads=num_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        block_shape=block_shape,
        is_packed_gqa=is_packed_gqa,
        layout_q=input_data.kwargs.get("layoutQ", "TND"),
        layout_kv=input_data.kwargs.get("layoutKv", "PAGED_BBND"),
        softmax_precision=softmax_precision,
        return_softmax_lse=int(input_data.kwargs.get("returnSoftmaxlse", 0)),
        physical_aic_num=ARCH35_AIC_NUM,
    )
    if metadata.numel() != METADATA_TOTAL_SIZE:
        raise RuntimeError(f"simulated metadata size must be {METADATA_TOTAL_SIZE}")

    if device in ("pyaclnn", "npu"):
        # Exercise the real AICPU metadata operator and require it to match the
        # simulator used by the FD golden pipeline. Keep Case tensors on CPU;
        # the ATK backend owns the single host-to-device copy.
        sparse_idx_npu = sparse_idx.npu()
        sparse_count_npu = sparse_count.npu()
        cu_q_npu = cu_q.npu()
        cu_kv_npu = cu_kv.npu()
        seqused_q_npu = seqused_q.npu() if use_seqused else None
        seqused_kv_npu = seqused_kv.npu() if use_seqused else None
        meta_op = _get_gbsa_metadata_op()
        real_metadata = meta_op(
            sparse_idx_npu,
            sparse_count_npu,
            _max_seq_len_from_cu(cu_q, actual_q),
            _max_seq_len_from_cu(cu_kv, actual_kv),
            int(num_heads),
            int(kv_heads),
            int(head_dim),
            block_shape,
            cu_seq_lengths=cu_q_npu,
            cu_seq_lengths_kv=cu_kv_npu,
            seq_used_q=seqused_q_npu,
            seq_used_kv=seqused_kv_npu,
            is_packed_gqa=int(is_packed_gqa),
            q_input_layout="TND",
            kv_input_layout="PAGED_BBND",
            mask_type=int(mask_type),
            quant_type=int(quant_type),
            softmax_precision=int(softmax_precision),
            window_size_left=-1,
            window_size_right=-1,
        )
        torch.npu.synchronize()
        meta_cpu = real_metadata.detach().cpu().to(torch.int32)
        if not torch.equal(meta_cpu, metadata.cpu().to(torch.int32)):
            raise RuntimeError("real AICPU Metadata-v8 output differs from the simulator")
        metadata = meta_cpu
        print(
            f"[GBSA META] path=REAL device={device} "
            f"header={meta_cpu[:5].tolist()} nonzero={int((meta_cpu != 0).sum())}",
            flush=True,
        )
    else:
        print(f"[GBSA META] path=SIMULATED device={device}", flush=True)

    input_data.kwargs["sparseBlockIdx"] = sparse_idx
    input_data.kwargs["sparseBlockCount"] = sparse_count
    input_data.kwargs["blockTableOptional"] = block_table_new
    input_data.kwargs["cuSeqLengthsQOptional"] = cu_q
    input_data.kwargs["cuSeqLengthsKvOptional"] = cu_kv
    input_data.kwargs["sequsedQOptional"] = seqused_q
    input_data.kwargs["sequsedKvOptional"] = seqused_kv
    input_data.kwargs["metadataOptional"] = metadata
    input_data.kwargs["winLeft"] = -1
    input_data.kwargs["winRight"] = -1
    input_data.kwargs["dstTypeMax"] = 0.0
    # softmaxPrecision legality is owned by tests/{st,950} generators + host tiling:
    #   st(910B): bf16→0, fp16→0|1;  950(A5): always 1.

    if test_mode == "fd":
        if not plan.enabled:
            raise RuntimeError(
                f"deterministic FD Case unexpectedly selected normal pipeline: {plan.reason}"
            )
        print_fd_plan_once(plan)

    return actual_q, actual_kv, plan


@register("aclnn_genericblocksparseattentioninputprocess")
class GenericBlockSparseAttentionInputProcess(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(GenericBlockSparseAttentionInputProcess, self).__init__(
            task_result, backend
        )
        self.return_lse = 0

    def init_by_input_data(self, input_data):
        np.random.seed(10)
        apply_init_tensors(input_data, device="cpu")

        input_args = []
        name_to_idx = {}
        for i, arg in enumerate(input_data.args):
            data = self.backend.convert_input_data(arg, index=i)
            input_args.extend(data)
        for name, kwarg in input_data.kwargs.items():
            start = len(input_args)
            data = self.backend.convert_input_data(kwarg, name=name)
            input_args.extend(data)
            # Most ATK converts yield one ctypes object per named input.
            if data:
                name_to_idx[name] = start

        output_packages = []
        for index, output_data in enumerate(self.task_result.output_info_list):
            output = self.backend.convert_output_data(output_data, index)
            output_packages.extend(output)
        input_args.extend(output_packages)

        AclTensorPtr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)

        # Null unused optionals by name (do NOT touch blockTableOptional / metadata).
        for name in (
            "attenMaskOptional",
            "qDequantScaleOptional",
            "kDequantScaleOptional",
            "vDequantScaleOptional",
            "pQuantScaleOptional",
        ):
            if name in name_to_idx:
                input_args[name_to_idx[name]] = null_tensor_ptr
        for name in ("sequsedQOptional", "sequsedKvOptional"):
            if name in name_to_idx and input_data.kwargs.get(name) is None:
                input_args[name_to_idx[name]] = null_tensor_ptr

        self.return_lse = int(input_data.kwargs.get("returnSoftmaxlse", 0))
        output_packages = []
        if self.return_lse == 1:
            input_args.pop()
            input_args.pop()
            output_packages.append(input_args[-2])
            output_packages.append(input_args[-1])
        else:
            input_args.pop()
            output_packages.append(input_args[-2])
        return input_args, output_packages

    def __call__(self):
        self.backend.aclnn_x_get_workspace_size()
        self.backend.aclnn_x()

    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            temp = self.acl_tensor_to_torch(output_pack).to(dtype=torch.float32)
            output.append(temp)
        if len(output) == 2:
            return output[0], output[1]
        return output[0]

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnGenericBlockSparseAttentionGetWorkspaceSize("
            "const aclTensor *query, const aclTensor *key, const aclTensor *value, "
            "const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount, "
            "const aclTensor *metadataOptional, const aclTensor *attenMaskOptional, "
            "const aclTensor *qDequantScaleOptional, const aclTensor *kDequantScaleOptional, "
            "const aclTensor *vDequantScaleOptional, const aclTensor *pQuantScaleOptional, "
            "const aclTensor *cuSeqLengthsQOptional, const aclTensor *cuSeqLengthsKvOptional, "
            "const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, "
            "const aclTensor *blockTableOptional, const aclIntArray *blockShape, "
            "int64_t isPackedGQA, char *layoutQ, char *layoutKv, double scaleValue, "
            "int64_t maskType, int64_t quantType, double dstTypeMax, int64_t softmaxPrecision, "
            "int64_t winLeft, int64_t winRight, int64_t returnSoftmaxlse, "
            "const aclTensor *attentionOut, const aclTensor *softmaxLseOptional, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )


class TestGenericBlockSparseAttentionTorch:
    @staticmethod
    def _torch_dtype_to_np(dtype):
        if dtype == torch.float16:
            return np.float16
        if dtype == torch.bfloat16:
            try:
                from ml_dtypes import bfloat16 as np_bf16

                return np_bf16
            except ImportError:
                return np.float32
        return np.float32

    @staticmethod
    def _tensor_to_np(t, np_dtype=None):
        t = safe_to_tensor(t)
        if t.dtype == torch.bfloat16:
            arr = t.detach().float().cpu().numpy()
            return arr.astype(np_dtype) if np_dtype is not None else arr
        arr = t.detach().cpu().numpy()
        if np_dtype is not None and arr.dtype != np_dtype:
            return arr.astype(np_dtype)
        return arr

    @classmethod
    def base_tile_mm(cls, left, right, mm_k_tile=128):
        """Match BSA950: K-split matmul, accumulate in fp32 (batched, fast)."""
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        k_dim = left.shape[1]
        if k_dim <= mm_k_tile:
            return np.matmul(left, right)
        res = None
        mm_k_loop = (k_dim + mm_k_tile - 1) // mm_k_tile
        for idx in range(mm_k_loop):
            sub_k = min(mm_k_tile, k_dim - idx * mm_k_tile)
            left_slice = left[:, idx * mm_k_tile : idx * mm_k_tile + sub_k]
            right_slice = right[idx * mm_k_tile : idx * mm_k_tile + sub_k, :]
            res_slice = np.matmul(left_slice, right_slice)
            res = res_slice if res is None else (res + res_slice)
        return res

    @classmethod
    def base_tile_mm_rows(cls, left, right, mm_k_tile=128):
        """Row-stable matmul for QK when query packs a GQA group (see scenario BM)."""
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        m = left.shape[0]
        if m == 1:
            return cls.base_tile_mm(left, right, mm_k_tile)
        out = np.empty((m, right.shape[1]), dtype=np.float32)
        for i in range(m):
            out[i] = cls.base_tile_mm(left[i : i + 1], right, mm_k_tile)[0]
        return out

    @classmethod
    def _rowsum_half_tree(cls, p):
        """Match AscendC WholeReduceSum<half>: pairwise add with half rounding each step.

        numpy np.sum(fp16) accumulates in fp32 then casts once; device half tree-reduce
        can differ by 1 ulp and shows up in LSE via log(gl).
        """
        x = p.astype(np.float32)
        while x.shape[1] > 1:
            if x.shape[1] % 2 == 1:
                tail = x[:, -1:]
                x = x[:, :-1]
            else:
                tail = None
            x = (x[:, 0::2] + x[:, 1::2]).astype(np.float16).astype(np.float32)
            if tail is not None:
                if x.shape[1] == 0:
                    x = tail
                else:
                    x = np.concatenate(
                        [
                            x[:, :-1],
                            (x[:, -1:] + tail).astype(np.float16).astype(np.float32),
                        ],
                        axis=1,
                    )
        return x[:, 0].astype(np.float16).reshape(-1, 1)

    @classmethod
    def ref_flash(
        cls,
        query,
        key,
        value,
        scale,
        np_dtype,
        use_low_sm,
        round_output=True,
    ):
        """BSA950-style tiled flash. query [S1,D], key [D,S2], value [S2,D].

        softmaxPrecision=1: SM in query.dtype, scale rounded to SM (arch35 ToBfloat16/half).
        softmaxPrecision=0: fp32 SM + DownCastP before PV.
        Rescale stays fp32 (arch35 mixed path).
        """
        cur_kv_s = key.shape[1]
        kv_s_base_tile = 128
        interm_dtype_re = np.float32
        if use_low_sm:
            interm_dtype_sm = np_dtype
            scale_f = np.float32(np.asarray(scale, dtype=np_dtype))
        else:
            interm_dtype_sm = np.float32
            scale_f = np.float32(scale)

        gm = gl = go = None
        for kv_s_start in range(0, cur_kv_s, kv_s_base_tile):
            cur_kv_s_tile = min(kv_s_base_tile, cur_kv_s - kv_s_start)
            key_tile = key[:, kv_s_start : kv_s_start + cur_kv_s_tile]
            value_tile = value[kv_s_start : kv_s_start + cur_kv_s_tile, :]
            qk = cls.base_tile_mm_rows(query, key_tile, 128)
            if use_low_sm:
                qk = qk.astype(interm_dtype_sm)
                qk = (np.asarray(qk, dtype=np.float32) * scale_f).astype(
                    interm_dtype_sm
                )
            else:
                qk = qk * scale_f
            sim = qk.astype(interm_dtype_sm)
            lm = np.max(sim, axis=-1, keepdims=True).astype(interm_dtype_sm)
            is_first = kv_s_start == 0
            if is_first:
                hm = lm
                gm = lm.astype(interm_dtype_re)
                dm = np.ones_like(gm, dtype=interm_dtype_re)
            else:
                lm_f = lm.astype(interm_dtype_re)
                hm_f = np.maximum(gm, lm_f)
                dm = np.exp(gm - hm_f).astype(interm_dtype_re)
                gm = hm_f
                hm = hm_f.astype(interm_dtype_sm)
            p = np.exp((sim - hm).astype(interm_dtype_sm))
            if interm_dtype_sm == np.float16:
                ll = cls._rowsum_half_tree(p)
            else:
                ll = np.sum(p, axis=-1, keepdims=True).astype(interm_dtype_sm)
            p = p.astype(np_dtype)
            lo = cls.base_tile_mm(p, value_tile, 128).astype(interm_dtype_re)
            if is_first:
                gl = ll.astype(interm_dtype_re)
                go = lo
            else:
                gl = gl * dm + ll.astype(interm_dtype_re)
                go = go * dm + lo

        go = go / gl
        if round_output:
            go = go.astype(np_dtype)
        lse = np.squeeze((np.log(gl) + gm), axis=-1).astype(np.float32)
        return np.asarray(go, dtype=np.float32), np.asarray(lse, dtype=np.float32)

    @staticmethod
    def ref_attention(query, key, value, scale):
        """BSA950 fp32 真值: one-shot matmul + softmax."""
        s = np.matmul(
            np.asarray(query, dtype=np.float32),
            np.asarray(key, dtype=np.float32),
        ) * np.float32(scale)
        row_max = np.max(s, axis=-1, keepdims=True)
        p = np.exp(s - row_max)
        row_sum = np.sum(p, axis=-1, keepdims=True)
        p = p / row_sum
        o = np.matmul(p, np.asarray(value, dtype=np.float32))
        lse = np.squeeze((np.log(row_sum) + row_max), axis=-1).astype(np.float32)
        return o.astype(np.float32), lse

    @classmethod
    def _compute_range(
        cls,
        *,
        query_np,
        key_np,
        value_np,
        sparse_idx,
        sparse_cnt,
        block_table_np,
        batch_idx,
        global_q,
        local_q,
        q_seqlen,
        kv_seqlen,
        kv_head,
        group_size,
        block_shape_y,
        raw_begin,
        raw_end,
        scale,
        np_dtype,
        use_low_sm,
        low_precision,
        round_output=True,
    ):
        """Compute one raw sparse-block interval shared by normal and FD."""

        top_k = int(sparse_idx.shape[2])
        valid_topk = min(max(int(sparse_cnt[kv_head, global_q]), 0), top_k)
        raw_begin = min(max(int(raw_begin), 0), valid_topk)
        raw_end = min(max(int(raw_end), 0), valid_topk)
        q_start = kv_head * group_size
        q_group = query_np[global_q, q_start:q_start + group_size, :]
        if raw_begin >= raw_end or q_seqlen <= 0 or kv_seqlen <= 0:
            return (
                np.zeros((group_size, query_np.shape[2]), dtype=np.float32),
                np.full((group_size,), np.finfo(np.float32).min, dtype=np.float32),
            )

        history_len = kv_seqlen - q_seqlen
        causal_bound = history_len + local_q
        k_parts = []
        v_parts = []
        for raw_idx in range(raw_begin, raw_end):
            logical_id = int(sparse_idx[kv_head, global_q, raw_idx])
            if logical_id < 0:
                continue
            if logical_id >= block_table_np.shape[1]:
                raise IndexError(f"logical block ID {logical_id} is out of blockTable")
            block_begin = logical_id * block_shape_y
            block_end = min(block_begin + block_shape_y, kv_seqlen)
            effective_end = min(block_end, causal_bound + 1)
            if effective_end <= block_begin:
                continue
            physical_id = int(block_table_np[batch_idx, logical_id])
            if physical_id < 0 or physical_id >= key_np.shape[0]:
                raise IndexError(f"physical block ID {physical_id} is out of KV storage")
            tile_len = effective_end - block_begin
            k_parts.append(key_np[physical_id, :tile_len, kv_head, :].T)
            v_parts.append(value_np[physical_id, :tile_len, kv_head, :])

        if not k_parts:
            return (
                np.zeros((group_size, query_np.shape[2]), dtype=np.float32),
                np.full((group_size,), np.finfo(np.float32).min, dtype=np.float32),
            )
        key_g = np.concatenate(k_parts, axis=1)
        value_g = np.concatenate(v_parts, axis=0)
        if low_precision:
            return cls.ref_flash(
                q_group,
                key_g,
                value_g,
                scale,
                np_dtype,
                use_low_sm,
                round_output=round_output,
            )
        return cls.ref_attention(q_group, key_g, value_g, scale)

    @classmethod
    def _fd_pipeline(
        cls,
        *,
        plan,
        query_np,
        key_np,
        value_np,
        sparse_idx,
        sparse_cnt,
        block_table_np,
        block_shape_y,
        q_seqlens,
        kv_seqlens,
        q_storage_seqlens,
        scale,
        np_dtype,
        use_low_sm,
    ):
        """Mirror FD partial O/LSE production and stable combine order."""

        if not isinstance(plan, FdPlan) or not plan.enabled:
            raise ValueError("FD pipeline requires an enabled Metadata-v8 plan")
        total_q, num_heads, head_dim = query_np.shape
        kv_heads = int(sparse_idx.shape[0])
        group_size = num_heads // kv_heads
        if (
            plan.total_base_tasks != sum(int(value) for value in q_seqlens) * kv_heads
            or plan.group_size != group_size
            or plan.top_k != int(sparse_idx.shape[2])
        ):
            raise ValueError("Metadata-v8 plan does not match the ATK Case")

        lowest = np.float32(np.finfo(np.float32).min)
        output = np.zeros((total_q, num_heads, head_dim), dtype=np.float32)
        lse_output = np.full((total_q, num_heads, 1), lowest, dtype=np.float32)
        partial_o = np.zeros(
            (plan.partial_task_num, group_size, head_dim), dtype=np.float32
        )
        partial_lse = np.full(
            (plan.partial_task_num, 2, plan.fd_lse_sub_stride),
            lowest,
            dtype=np.float32,
        )

        base_locations = []
        q_storage_start = 0
        for batch_idx, q_seqlen in enumerate(q_seqlens):
            for local_q in range(int(q_seqlen)):
                global_q = q_storage_start + local_q
                base_locations.extend(
                    (batch_idx, global_q, local_q, kv_head)
                    for kv_head in range(kv_heads)
                )
            q_storage_start += int(q_storage_seqlens[batch_idx])
        if len(base_locations) != plan.total_base_tasks:
            raise RuntimeError("base-task location count does not match Metadata-v8 plan")

        row_split = min(group_size, ((group_size + 7) // 8 * 8) // 2)
        for core in plan.core_ranges:
            for task_slice in core.task_slices:
                batch_idx, global_q, local_q, kv_head = base_locations[
                    task_slice.base_task
                ]
                is_partial = task_slice.partial_task_id is not None
                partial_out, partial_lse_rows = cls._compute_range(
                    query_np=query_np,
                    key_np=key_np,
                    value_np=value_np,
                    sparse_idx=sparse_idx,
                    sparse_cnt=sparse_cnt,
                    block_table_np=block_table_np,
                    batch_idx=batch_idx,
                    global_q=global_q,
                    local_q=local_q,
                    q_seqlen=int(q_seqlens[batch_idx]),
                    kv_seqlen=int(kv_seqlens[batch_idx]),
                    kv_head=kv_head,
                    group_size=group_size,
                    block_shape_y=block_shape_y,
                    raw_begin=task_slice.raw_begin,
                    raw_end=task_slice.raw_end,
                    scale=scale,
                    np_dtype=np_dtype,
                    use_low_sm=use_low_sm,
                    low_precision=True,
                    round_output=not is_partial,
                )
                q_start = kv_head * group_size
                if not is_partial:
                    output[global_q, q_start:q_start + group_size, :] = partial_out
                    lse_output[global_q, q_start:q_start + group_size, 0] = (
                        partial_lse_rows
                    )
                    continue
                partial_id = task_slice.partial_task_id
                partial_o[partial_id] = partial_out
                partial_lse[partial_id, 0, :row_split] = partial_lse_rows[:row_split]
                partial_lse[partial_id, 1, :group_size - row_split] = (
                    partial_lse_rows[row_split:]
                )

        for base_task in plan.combine_base_tasks:
            partial_start = plan.partial_start_by_base[base_task]
            split_count = plan.partial_count_by_base[base_task]
            lse_rows = np.empty((split_count, group_size), dtype=np.float32)
            lse_rows[:, :row_split] = partial_lse[
                partial_start:partial_start + split_count, 0, :row_split
            ]
            lse_rows[:, row_split:] = partial_lse[
                partial_start:partial_start + split_count,
                1,
                :group_size - row_split,
            ]
            split_count_align = (split_count + 7) // 8 * 8
            lse_aligned = np.full(
                (split_count_align, group_size), lowest, dtype=np.float32
            )
            lse_aligned[:split_count] = lse_rows
            lse_max = np.max(lse_aligned, axis=0)
            lse_sum = np.sum(
                np.exp(lse_aligned - lse_max[None, :]).astype(np.float32),
                axis=0,
                dtype=np.float32,
            )
            global_lse = (np.log(lse_sum) + lse_max).astype(np.float32)
            weights = np.exp(lse_rows - global_lse[None, :]).astype(np.float32)
            combined = np.zeros((group_size, head_dim), dtype=np.float32)
            for split in range(split_count):
                combined = (
                    combined
                    + partial_o[partial_start + split]
                    * weights[split, :, None]
                ).astype(np.float32)

            _, global_q, _, kv_head = base_locations[base_task]
            q_start = kv_head * group_size
            output[global_q, q_start:q_start + group_size, :] = (
                combined.astype(np_dtype).astype(np.float32)
            )
            lse_output[global_q, q_start:q_start + group_size, 0] = global_lse
        return output, lse_output

    def calc_data(
        self,
        query,
        key,
        value,
        sparse_block_idx,
        sparse_block_count,
        block_table,
        block_shape,
        q_seqlens,
        kv_seqlens,
        scale_value,
        softmax_precision,
        is_benchmark=False,
        q_storage_seqlens=None,
        plan=None,
    ):
        """Return (out_bm, out_gt, lse_bm, lse_gt). Only the requested path is computed.

        is_benchmark=False -> kernel-aligned bm; True -> fp32 gt.
        """
        query_t = safe_to_tensor(query)
        np_dtype = self._torch_dtype_to_np(query_t.dtype)
        query_np = self._tensor_to_np(query_t, np_dtype)
        key_np = self._tensor_to_np(key, np_dtype)
        value_np = self._tensor_to_np(value, np_dtype)
        sparse_idx = self._tensor_to_np(sparse_block_idx).astype(np.int32)
        sparse_cnt = self._tensor_to_np(sparse_block_count).astype(np.int32)
        block_table_np = self._tensor_to_np(block_table).astype(np.int32)

        block_shape_x = int(block_shape[0])
        block_shape_y = int(block_shape[1])
        if block_shape_x != 1:
            raise ValueError(
                f"calc_data currently requires blockShapeX=1, got {block_shape_x}"
            )
        if q_storage_seqlens is None:
            q_storage_seqlens = q_seqlens
        total_q, num_heads, head_dim = query_np.shape
        kv_heads = int(sparse_idx.shape[0])
        group_size = num_heads // kv_heads
        top_k = int(sparse_idx.shape[2])
        use_low_sm = int(softmax_precision) == 1
        scale = float(scale_value)
        compute_bm = not is_benchmark
        compute_gt = is_benchmark

        lowest = np.float32(np.finfo(np.float32).min)
        out_bm = np.zeros((total_q, num_heads, head_dim), dtype=np.float32)
        out_gt = np.zeros((total_q, num_heads, head_dim), dtype=np.float32)
        lse_bm = np.full((total_q, num_heads, 1), lowest, dtype=np.float32)
        lse_gt = np.full((total_q, num_heads, 1), lowest, dtype=np.float32)

        if compute_bm and plan is not None and plan.enabled:
            out_bm, lse_bm = self._fd_pipeline(
                plan=plan,
                query_np=query_np,
                key_np=key_np,
                value_np=value_np,
                sparse_idx=sparse_idx,
                sparse_cnt=sparse_cnt,
                block_table_np=block_table_np,
                block_shape_y=block_shape_y,
                q_seqlens=q_seqlens,
                kv_seqlens=kv_seqlens,
                q_storage_seqlens=q_storage_seqlens,
                scale=scale,
                np_dtype=np_dtype,
                use_low_sm=use_low_sm,
            )
            return (
                torch.from_numpy(out_bm),
                torch.from_numpy(out_gt),
                torch.from_numpy(lse_bm),
                torch.from_numpy(lse_gt),
            )

        q_offset = 0
        for b in range(len(q_seqlens)):
            q_seqlen = int(q_seqlens[b])
            kv_seqlen = int(kv_seqlens[b])
            if q_seqlen == 0 or kv_seqlen == 0:
                q_offset += int(q_storage_seqlens[b])
                continue
            for q_token in range(q_seqlen):
                global_q = q_offset + q_token
                for kv_h in range(kv_heads):
                    q_start = kv_h * group_size
                    go, lse = self._compute_range(
                        query_np=query_np,
                        key_np=key_np,
                        value_np=value_np,
                        sparse_idx=sparse_idx,
                        sparse_cnt=sparse_cnt,
                        block_table_np=block_table_np,
                        batch_idx=b,
                        global_q=global_q,
                        local_q=q_token,
                        q_seqlen=q_seqlen,
                        kv_seqlen=kv_seqlen,
                        kv_head=kv_h,
                        group_size=group_size,
                        block_shape_y=block_shape_y,
                        raw_begin=0,
                        raw_end=top_k,
                        scale=scale,
                        np_dtype=np_dtype,
                        use_low_sm=use_low_sm,
                        low_precision=compute_bm,
                        round_output=True,
                    )
                    if compute_bm:
                        out_bm[global_q, q_start:q_start + group_size, :] = go
                        lse_bm[global_q, q_start:q_start + group_size, 0] = lse
                    else:
                        out_gt[global_q, q_start:q_start + group_size, :] = go
                        lse_gt[global_q, q_start:q_start + group_size, 0] = lse
            q_offset += int(q_storage_seqlens[b])

        return (
            torch.from_numpy(out_bm),
            torch.from_numpy(out_gt),
            torch.from_numpy(lse_bm),
            torch.from_numpy(lse_gt),
        )


@register("aclnn_genericblocksparseattention")
class GenericBlockSparseAttentionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(GenericBlockSparseAttentionApi, self).__init__(task_result)
        self.fd_plan = None

    def init_by_input_data(self, input_data: InputDataset):
        np.random.seed(10)
        _, _, self.fd_plan = apply_init_tensors(input_data, device=self.device)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        query = input_data.kwargs["query"]
        key = input_data.kwargs["key"]
        value = input_data.kwargs["value"]
        sparse_block_idx = input_data.kwargs["sparseBlockIdx"]
        sparse_block_count = input_data.kwargs["sparseBlockCount"]
        block_table = input_data.kwargs["blockTableOptional"]
        block_shape = input_data.kwargs["blockShape"]
        scale_value = input_data.kwargs["scaleValue"]
        softmax_precision = input_data.kwargs["softmaxPrecision"]
        return_lse = int(input_data.kwargs.get("returnSoftmaxlse", 0))

        # Prefer rewritten cu + seqused; else equal-batch recovery from shapes.
        query_t = safe_to_tensor(query)
        decoded = decode_cu_and_seqused(
            input_data.kwargs.get("cuSeqLengthsQOptional"),
            input_data.kwargs.get("cuSeqLengthsKvOptional"),
            seqused_q=input_data.kwargs.get("sequsedQOptional"),
            seqused_kv=input_data.kwargs.get("sequsedKvOptional"),
            total_q=int(query_t.shape[0]) if query_t is not None else None,
        )
        if decoded is not None:
            q_seqlens, kv_seqlens, q_storage_seqlens, _kv_storage = decoded
        else:
            q_seqlens, kv_seqlens = recover_batch_seqlens(
                query_t,
                safe_to_tensor(sparse_block_idx),
                safe_to_tensor(block_table),
            )
            q_storage_seqlens = q_seqlens

        test_obj = TestGenericBlockSparseAttentionTorch()
        atten_out_bm, atten_out_gt, lse_bm, lse_gt = test_obj.calc_data(
            query,
            key,
            value,
            sparse_block_idx,
            sparse_block_count,
            block_table,
            block_shape,
            q_seqlens,
            kv_seqlens,
            scale_value,
            softmax_precision,
            is_benchmark=bool(self.task_result.is_benchmark_task),
            q_storage_seqlens=q_storage_seqlens,
            plan=self.fd_plan,
        )

        # BSA tests/950: False -> kernel-aligned bm (标杆); True -> fp32 gt (真值)
        if not self.task_result.is_benchmark_task:
            atten_out = atten_out_bm
            lse_out = lse_bm
        else:
            atten_out = atten_out_gt
            lse_out = lse_gt

        if return_lse == 1:
            return atten_out, lse_out
        return atten_out
