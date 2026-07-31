"""Sparse Lightning Indexer Grad KL Loss 的主机侧诊断工具。"""

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


@dataclass(frozen=True)
class DebugOptions:
    """由环境变量解析出的诊断选项。"""

    requested: bool
    task: Optional[Tuple[int, int]]
    bmm1_coord: Optional[Tuple[int, int]]
    relu_coord: Optional[Tuple[int, int]]


_NO_DEBUG_OPTIONS = DebugOptions(
    requested=False,
    task=None,
    bmm1_coord=None,
    relu_coord=None,
)


@dataclass(frozen=True)
class DebugContext:
    """一次 BSND kernel 调用所需的诊断上下文。"""

    options: DebugOptions
    full_valid: bool
    source_s2: int
    d_rope_orig: int
    scale_value: float
    loss_partial_stride: int
    gather_sentinel: float
    gather_chunk_size: int
    k_split: int
    n_split: int
    preprocess_gather: Callable
    query: torch.Tensor
    key: torch.Tensor
    query_index: torch.Tensor
    key_index: torch.Tensor
    weights: torch.Tensor
    sparse_indices: torch.Tensor
    softmax_max: torch.Tensor
    softmax_sum: torch.Tensor
    query_rope: torch.Tensor
    key_rope: torch.Tensor
    k_real_size: torch.Tensor
    gather_key_ws: torch.Tensor
    gather_key_rope_ws: torch.Tensor
    gather_key_index_ws: torch.Tensor
    bmm1_res_ws: torch.Tensor
    bmm2_res_ws: torch.Tensor
    p_reduced_ws: torch.Tensor
    v1_stats_ws: torch.Tensor
    dweight_partial_ws: torch.Tensor
    relu_grad_out: torch.Tensor
    dk_matmul_partial: torch.Tensor
    d_query_index_accum: torch.Tensor
    d_key_index_accum: torch.Tensor
    loss_partial_ws: torch.Tensor
    d_weight: torch.Tensor
    loss_out: torch.Tensor


def _parse_coordinate(
    text: str,
    name: str,
    coordinate_format: str,
    example: str,
) -> Tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"{name} must use the format '{coordinate_format}', "
            f"for example '{example}'"
        )
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError as exc:
        raise ValueError(
            f"{name} must contain two integer values: '{coordinate_format}'"
        ) from exc


def parse_debug_options(
    batch_size: int,
    query_length: int,
    index_head_count: int,
    execution_k: int,
) -> DebugOptions:
    """解析并校验主机诊断环境变量。"""
    requested = os.environ.get("SLIG_DEBUG", "0") == "1"
    task_text = os.environ.get("SLIG_DEBUG_TASK", "").strip()
    relu_coord_text = os.environ.get("SLIG_DEBUG_RELU_COORD", "").strip()
    task = None
    relu_coord = None

    if not requested and not task_text and not relu_coord_text:
        return _NO_DEBUG_OPTIONS

    if task_text.lower() == "auto":
        raise ValueError(
            "SLIG_DEBUG_TASK=auto is disabled because scanning the full BMM1 "
            "workspace can allocate several GiB; use SLIG_SELECT_DEBUG_TASK=1 "
            "in test_0721_general.py, then rerun with SLIG_DEBUG_TASK=batch,query"
        )
    if task_text:
        task = _parse_coordinate(
            task_text,
            "SLIG_DEBUG_TASK",
            "batch,query",
            "0,25",
        )
        if not (0 <= task[0] < batch_size and 0 <= task[1] < query_length):
            raise ValueError(
                f"SLIG_DEBUG_TASK={task_text!r} is outside BSND shape "
                f"B={batch_size}, S1={query_length}"
            )
    if relu_coord_text:
        relu_coord = _parse_coordinate(
            relu_coord_text,
            "SLIG_DEBUG_RELU_COORD",
            "head,k",
            "2,2153",
        )
        if not (
            0 <= relu_coord[0] < index_head_count
            and 0 <= relu_coord[1] < execution_k
        ):
            raise ValueError(
                f"SLIG_DEBUG_RELU_COORD={relu_coord_text!r} is outside "
                f"reluGrad shape Nidx1={index_head_count}, K={execution_k}"
            )

    return DebugOptions(
        requested=requested,
        task=task,
        bmm1_coord=None,
        relu_coord=relu_coord,
    )


def run_debug_diagnostics(context: DebugContext) -> bool:
    """按选项运行诊断，返回是否完成了一次 dump。"""
    if context.options.task is not None:
        _run_task_diagnostics(context)
        return True
    if context.options.requested:
        _run_full_diagnostics(context)
        return True
    return False


def _debug_task_tensor(name: str, tensor: torch.Tensor) -> None:
    tensor_fp32 = tensor.float()
    finite = torch.isfinite(tensor_fp32)
    finite_count = int(finite.sum().item())
    nonfinite_count = tensor_fp32.numel() - finite_count
    finite_abs_max = (
        tensor_fp32[finite].abs().max().item()
        if finite_count > 0
        else float("nan")
    )
    print(
        f"[DEBUG_TASK] {name}: shape={tuple(tensor.shape)}, "
        f"nonfinite={nonfinite_count}, finite_abs_max={finite_abs_max:.6e}"
    )


def _debug_task_compare(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    finite = torch.isfinite(actual_fp32) & torch.isfinite(expected_fp32)
    nonfinite_count = actual_fp32.numel() - int(finite.sum().item())
    if finite.any().item():
        diff = (actual_fp32[finite] - expected_fp32[finite]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        norm_rel = torch.linalg.vector_norm(diff).item() / (
            torch.linalg.vector_norm(expected_fp32[finite]).item() + 1e-6
        )
    else:
        max_diff = float("nan")
        mean_diff = float("nan")
        norm_rel = float("nan")
    print(
        f"[DEBUG_TASK] {name}: max_diff={max_diff:.6e}, "
        f"mean_diff={mean_diff:.6e}, norm_rel={norm_rel:.6e}, "
        f"nonfinite={nonfinite_count}"
    )


def _debug_gather_bad_rows(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    debug_k_real: int,
    debug_sparse_indices: torch.Tensor,
    gather_chunk_size: int,
) -> set:
    actual_fp32 = actual[:debug_k_real].float()
    expected_fp32 = expected[:debug_k_real].float()
    element_diff = (actual_fp32 - expected_fp32).abs()
    row_bad = (~torch.isfinite(actual_fp32)).any(dim=1)
    row_bad = row_bad | (~torch.isfinite(expected_fp32)).any(dim=1)
    row_bad = row_bad | (element_diff.amax(dim=1) != 0)
    bad_rows_tensor = torch.nonzero(row_bad, as_tuple=False).flatten()
    bad_row_count = int(bad_rows_tensor.numel())
    first_rows = [int(value) for value in bad_rows_tensor[:16].cpu().tolist()]
    first_row_details = [
        (
            row,
            row // gather_chunk_size,
            row % gather_chunk_size,
            int(debug_sparse_indices[row].item()),
        )
        for row in first_rows
    ]
    print(
        f"[DEBUG_TASK] {name} bad rows: count={bad_row_count} "
        f"first(row,chunk,lane,sparse_idx)={first_row_details}"
    )
    if bad_row_count > 0:
        first_row = first_rows[0]
        first_row_diff = element_diff[first_row]
        first_col = int(
            torch.nonzero(first_row_diff != 0, as_tuple=False)[0, 0].item()
        )
        candidate_row_diff = (
            expected_fp32 - actual_fp32[first_row].unsqueeze(0)
        ).abs().amax(dim=1)
        nearest_row = int(candidate_row_diff.argmin().item())
        print(
            f"[DEBUG_TASK] {name} first error: "
            f"coord=({first_row},{first_col}) "
            f"actual={actual_fp32[first_row, first_col].item():.6e} "
            f"expected={expected_fp32[first_row, first_col].item():.6e} "
            f"actual_row_abs_max={actual_fp32[first_row].abs().max().item():.6e} "
            f"nearest_expected_row={nearest_row} "
            f"nearest_sparse_idx={int(debug_sparse_indices[nearest_row].item())} "
            f"nearest_row_max_diff={candidate_row_diff[nearest_row].item():.6e}"
        )
    return set(int(value) for value in bad_rows_tensor.cpu().tolist())


def _run_task_diagnostics(context: DebugContext) -> None:
    debug_b, debug_q = context.options.task
    query = context.query
    key = context.key
    query_index = context.query_index
    key_index = context.key_index
    weights = context.weights
    sparse_indices = context.sparse_indices
    softmax_max = context.softmax_max
    softmax_sum = context.softmax_sum
    query_rope = context.query_rope
    key_rope = context.key_rope
    gather_key_ws = context.gather_key_ws
    gather_key_rope_ws = context.gather_key_rope_ws
    gather_key_index_ws = context.gather_key_index_ws
    bmm1_res_ws = context.bmm1_res_ws
    bmm2_res_ws = context.bmm2_res_ws
    p_reduced_ws = context.p_reduced_ws
    v1_stats_ws = context.v1_stats_ws
    dweight_partial_ws = context.dweight_partial_ws
    relu_grad_out = context.relu_grad_out
    dk_matmul_partial = context.dk_matmul_partial
    d_query_index_accum = context.d_query_index_accum
    d_weight = context.d_weight
    source_s2 = context.source_s2
    full_valid = context.full_valid
    scale_value = context.scale_value
    d_rope = key_rope.shape[-1]
    K = gather_key_ws.shape[2]
    Nidx1 = query_index.shape[2]
    DQuery = query.shape[-1]

    debug_sparse_indices = sparse_indices[debug_b, debug_q, 0, :K].to(torch.long)
    debug_index_valid = (
        (debug_sparse_indices >= 0) & (debug_sparse_indices < source_s2)
    )
    debug_safe_indices = debug_sparse_indices.clamp(0, source_s2 - 1)
    debug_valid_mask = (
        torch.ones(K, dtype=torch.float32, device=query.device)
        if full_valid
        else (
            torch.arange(K, dtype=torch.int32, device=query.device)
            < context.k_real_size[debug_b, debug_q]
        ).float()
    )
    debug_k_real = (
        K
        if full_valid
        else int(context.k_real_size[debug_b, debug_q].item())
    )
    debug_expected_gki = torch.index_select(
        key_index[debug_b, :, 0, :], 0, debug_safe_indices
    )
    debug_expected_gki = debug_expected_gki * debug_index_valid.unsqueeze(-1).to(
        debug_expected_gki.dtype
    )
    debug_expected_gk = torch.index_select(
        key[debug_b, :, 0, :], 0, debug_safe_indices
    )
    debug_expected_gk = debug_expected_gk * debug_index_valid.unsqueeze(-1).to(
        debug_expected_gk.dtype
    )
    if d_rope > 0:
        debug_expected_grope = torch.index_select(
            key_rope[debug_b, :, 0, :], 0, debug_safe_indices
        )
        debug_expected_grope = debug_expected_grope * debug_index_valid.unsqueeze(-1).to(
            debug_expected_grope.dtype
        )
    else:
        debug_expected_grope = None

    debug_gk = gather_key_ws[debug_b, debug_q]
    debug_grope = gather_key_rope_ws[debug_b, debug_q]
    debug_gki = gather_key_index_ws[debug_b, debug_q]
    debug_bmm2 = bmm2_res_ws[debug_b, debug_q]
    debug_bmm1 = bmm1_res_ws[debug_b, debug_q]
    debug_query = query[debug_b, debug_q].float()
    debug_bmm1_from_gather = torch.matmul(
        debug_query, debug_gk.float().transpose(0, 1)
    )
    debug_bmm1_from_host_gather = torch.matmul(
        debug_query, debug_expected_gk.float().transpose(0, 1)
    )
    if d_rope > 0:
        debug_query_rope = query_rope[debug_b, debug_q].float()
        debug_bmm1_from_gather = debug_bmm1_from_gather + torch.matmul(
            debug_query_rope, debug_grope.float().transpose(0, 1)
        )
        debug_bmm1_from_host_gather = debug_bmm1_from_host_gather + torch.matmul(
            debug_query_rope, debug_expected_grope.float().transpose(0, 1)
        )

    debug_query_index = query_index[debug_b, debug_q].float()
    debug_bmm2_from_gather = torch.relu(
        torch.matmul(debug_query_index, debug_gki.float().transpose(0, 1))
    )
    debug_bmm2_from_host_gather = torch.relu(
        torch.matmul(
            debug_query_index,
            debug_expected_gki.float().transpose(0, 1),
        )
    )
    debug_softmax_max = softmax_max[debug_b, 0, debug_q].float()
    debug_softmax_sum = softmax_sum[debug_b, 0, debug_q].float().abs().clamp_min(1e-8)
    debug_expected_p = (
        torch.exp(
            debug_bmm1.float() * scale_value - debug_softmax_max.unsqueeze(-1)
        )
        / debug_softmax_sum.unsqueeze(-1)
    ).mean(dim=0) * debug_valid_mask
    debug_reduce_sum_y = (
        debug_bmm2.float() * weights[debug_b, debug_q].float().unsqueeze(-1)
    ).sum(dim=0)
    debug_reduce_sum_y = debug_reduce_sum_y.masked_fill(
        debug_valid_mask == 0, -1e30
    )
    debug_expected_softmax_i = (
        torch.softmax(debug_reduce_sum_y, dim=-1) * debug_valid_mask
    )
    debug_diff = debug_expected_softmax_i - debug_expected_p
    debug_half_k = K // 2
    debug_expected_stats = torch.zeros(
        4, dtype=torch.float32, device=query.device
    )
    debug_local_max = []
    for debug_owner in range(2):
        debug_owner_start = debug_owner * debug_half_k
        debug_owner_end = debug_owner_start + debug_half_k
        debug_owner_score = debug_reduce_sum_y[
            debug_owner_start:debug_owner_end
        ]
        debug_owner_max = debug_owner_score.max().clamp_min(-1e20)
        debug_local_max.append(debug_owner_max)
        debug_expected_stats[debug_owner] = debug_owner_max
    debug_global_max = torch.maximum(debug_local_max[0], debug_local_max[1])
    for debug_owner in range(2):
        debug_owner_start = debug_owner * debug_half_k
        debug_owner_end = debug_owner_start + debug_half_k
        debug_expected_stats[2 + debug_owner] = torch.exp(
            debug_reduce_sum_y[debug_owner_start:debug_owner_end]
            - debug_global_max
        ).sum()
    debug_expected_dweight = (
        debug_bmm2.float() * debug_diff.unsqueeze(0)
    ).sum(dim=-1)
    debug_expected_dweight_partial = torch.stack(
        (
            (
                debug_bmm2[:, :debug_half_k].float()
                * debug_diff[:debug_half_k].unsqueeze(0)
            ).sum(dim=-1),
            (
                debug_bmm2[:, debug_half_k:].float()
                * debug_diff[debug_half_k:].unsqueeze(0)
            ).sum(dim=-1),
        )
    )
    debug_expected_relu = (
        weights[debug_b, debug_q].float().unsqueeze(-1)
        * debug_diff.unsqueeze(0)
        * (debug_bmm2.float() > 0).float()
    ).to(relu_grad_out.dtype).float()
    debug_relu = relu_grad_out[debug_b, debug_q].float()

    if context.options.relu_coord is not None:
        debug_relu_head, debug_relu_k = context.options.relu_coord
        debug_relu_unmasked = (
            weights[debug_b, debug_q, debug_relu_head].float()
            * debug_diff[debug_relu_k]
        )
        print(
            f"[DEBUG_TASK] relu coord=(h={debug_relu_head},k={debug_relu_k}) "
            f"actual={debug_relu[debug_relu_head, debug_relu_k].item():.6e} "
            f"expected={debug_expected_relu[debug_relu_head, debug_relu_k].item():.6e} "
            f"unmasked={debug_relu_unmasked.item():.6e} "
            f"bmm2={debug_bmm2[debug_relu_head, debug_relu_k].item():.6e} "
            f"diff={debug_diff[debug_relu_k].item():.6e} "
            f"weight={weights[debug_b, debug_q, debug_relu_head].float().item():.6e}"
        )

    debug_expected_dq = torch.matmul(debug_relu, debug_gki.float())
    debug_expected_dk = torch.matmul(
        debug_relu.transpose(0, 1), debug_query_index
    )
    print(
        f"[DEBUG_TASK] selected b={debug_b} q={debug_q} "
        f"kReal={debug_k_real} full_valid={full_valid}"
    )
    _debug_task_tensor("GatherKey_ws", debug_gk)
    if d_rope > 0:
        _debug_task_tensor("GatherKeyRope_ws", debug_grope)
    _debug_task_tensor("GatherKeyIndex_ws", debug_gki)
    _debug_task_tensor("bmm1Res", debug_bmm1)
    _debug_task_tensor("bmm2Res", debug_bmm2)
    _debug_task_tensor("PReduced", p_reduced_ws[debug_b, debug_q])
    _debug_task_tensor("V1Stats", v1_stats_ws[debug_b, debug_q])
    _debug_task_tensor(
        "DWeightPartial", dweight_partial_ws[debug_b, debug_q]
    )
    _debug_task_tensor("dWeightOut", d_weight[debug_b, debug_q])
    _debug_task_tensor("reluGradRes", relu_grad_out[debug_b, debug_q])
    _debug_task_compare(
        "GatherKey_ws vs host gather (valid K)",
        debug_gk[:debug_k_real],
        debug_expected_gk[:debug_k_real],
    )
    if d_rope > 0:
        _debug_task_compare(
            "GatherKeyRope_ws vs host gather (valid K)",
            debug_grope[:debug_k_real],
            debug_expected_grope[:debug_k_real],
        )
    _debug_task_compare(
        "BMM1 vs Query @ GatherKey.T plus Rope (valid K)",
        debug_bmm1[:, :debug_k_real],
        debug_bmm1_from_gather[:, :debug_k_real],
    )
    _debug_task_compare(
        "BMM1 host-gather cross-check (valid K)",
        debug_bmm1_from_gather[:, :debug_k_real],
        debug_bmm1_from_host_gather[:, :debug_k_real],
    )

    if context.options.bmm1_coord is not None:
        debug_head, debug_k = context.options.bmm1_coord
        if debug_k < debug_k_real:
            debug_terms = []
            for debug_chunk in range(DQuery // context.k_split):
                chunk_start = debug_chunk * context.k_split
                chunk_end = (debug_chunk + 1) * context.k_split
                debug_terms.append(
                    float(
                        torch.dot(
                            debug_query[debug_head, chunk_start:chunk_end],
                            debug_gk[debug_k, chunk_start:chunk_end].float(),
                        ).item()
                    )
                )
            debug_rope_term = 0.0
            if d_rope > 0:
                debug_rope_term = float(
                    torch.dot(
                        debug_query_rope[debug_head],
                        debug_grope[debug_k].float(),
                    ).item()
                )
            print(
                f"[DEBUG_TASK] BMM1 first-error coord=(h={debug_head}, k={debug_k}) "
                f"sparse_idx={int(debug_sparse_indices[debug_k].item())} "
                f"index_valid={bool(debug_index_valid[debug_k].item())} "
                f"tilelang={debug_bmm1[debug_head, debug_k].float().item():.6e} "
                f"host={debug_bmm1_from_host_gather[debug_head, debug_k].item():.6e} "
                f"q128={debug_terms} rope={debug_rope_term:.6e}"
            )

    _debug_task_compare(
        "GatherKeyIndex_ws vs host gather (valid K)",
        debug_gki[:debug_k_real],
        debug_expected_gki[:debug_k_real],
    )
    debug_gk_bad_rows = _debug_gather_bad_rows(
        "GatherKey_ws",
        debug_gk,
        debug_expected_gk,
        debug_k_real,
        debug_sparse_indices,
        context.gather_chunk_size,
    )
    if d_rope > 0:
        debug_grope_bad_rows = _debug_gather_bad_rows(
            "GatherKeyRope_ws",
            debug_grope,
            debug_expected_grope,
            debug_k_real,
            debug_sparse_indices,
            context.gather_chunk_size,
        )
    else:
        debug_grope_bad_rows = set()
    debug_gki_bad_rows = _debug_gather_bad_rows(
        "GatherKeyIndex_ws",
        debug_gki,
        debug_expected_gki,
        debug_k_real,
        debug_sparse_indices,
        context.gather_chunk_size,
    )
    if d_rope > 0:
        debug_all_bad_rows = (
            debug_gk_bad_rows & debug_grope_bad_rows & debug_gki_bad_rows
        )
        debug_bad_rows_union = (
            debug_gk_bad_rows | debug_grope_bad_rows | debug_gki_bad_rows
        )
    else:
        debug_all_bad_rows = debug_gk_bad_rows & debug_gki_bad_rows
        debug_bad_rows_union = debug_gk_bad_rows | debug_gki_bad_rows
    print(
        f"[DEBUG_TASK] Gather bad-row overlap: "
        f"all={sorted(debug_all_bad_rows)[:32]} "
        f"key_only={sorted(debug_gk_bad_rows - debug_grope_bad_rows - debug_gki_bad_rows)[:32]} "
        f"rope_only={sorted(debug_grope_bad_rows - debug_gk_bad_rows - debug_gki_bad_rows)[:32]} "
        f"key_index_only={sorted(debug_gki_bad_rows - debug_gk_bad_rows - debug_grope_bad_rows)[:32]} "
        f"union_count={len(debug_bad_rows_union)}"
    )
    _debug_task_compare(
        "bmm2Res vs QueryIndex @ GatherKeyIndex_ws.T (valid K)",
        debug_bmm2[:, :debug_k_real],
        debug_bmm2_from_gather[:, :debug_k_real],
    )
    _debug_task_compare(
        "bmm2 host-gather cross-check (valid K)",
        debug_bmm2_from_gather[:, :debug_k_real],
        debug_bmm2_from_host_gather[:, :debug_k_real],
    )
    _debug_task_compare(
        "PReduced vs BMM1 formula",
        p_reduced_ws[debug_b, debug_q],
        debug_expected_p,
    )
    _debug_task_compare(
        "V1Stats scalars vs owned-K formula",
        v1_stats_ws[debug_b, debug_q, :, 0],
        debug_expected_stats,
    )
    _debug_task_compare(
        "DWeightPartial vs owned-K formula",
        dweight_partial_ws[debug_b, debug_q],
        debug_expected_dweight_partial,
    )
    _debug_task_compare(
        "dWeightOut vs BMM2/P/SY formula",
        d_weight[debug_b, debug_q],
        debug_expected_dweight,
    )
    _debug_task_compare(
        "reluGradRes vs BF16 formula",
        relu_grad_out[debug_b, debug_q],
        debug_expected_relu,
    )

    debug_relu_abs_diff = (debug_relu - debug_expected_relu).abs()
    debug_relu_bad = torch.nonzero(debug_relu_abs_diff != 0, as_tuple=False)
    debug_relu_false_zero = (debug_relu == 0) & (debug_expected_relu != 0)
    debug_relu_false_nonzero = (debug_relu != 0) & (debug_expected_relu == 0)
    debug_relu_wrong_nonzero = (
        (debug_relu != 0)
        & (debug_expected_relu != 0)
        & (debug_relu != debug_expected_relu)
    )
    debug_relu_significant = debug_relu_abs_diff > 1e-3
    debug_significant_false_zero = (
        debug_relu_false_zero & debug_relu_significant
    ).sum().item()
    debug_significant_false_nonzero = (
        debug_relu_false_nonzero & debug_relu_significant
    ).sum().item()
    debug_significant_wrong_nonzero = (
        debug_relu_wrong_nonzero & debug_relu_significant
    ).sum().item()
    print(
        f"[DEBUG_TASK] reluGrad mismatch summary: "
        f"total={debug_relu_bad.shape[0]} "
        f"false_zero={debug_relu_false_zero.sum().item()} "
        f"false_nonzero={debug_relu_false_nonzero.sum().item()} "
        f"wrong_nonzero={debug_relu_wrong_nonzero.sum().item()} "
        f"significant_false_zero={debug_significant_false_zero} "
        f"significant_false_nonzero={debug_significant_false_nonzero} "
        f"significant_wrong_nonzero={debug_significant_wrong_nonzero} "
        f"diff_gt_1e-3={debug_relu_significant.sum().item()} "
        f"diff_gt_1e-2={(debug_relu_abs_diff > 1e-2).sum().item()}"
    )
    if debug_relu_bad.shape[0] > 0:
        debug_relu_focus = torch.nonzero(debug_relu_significant, as_tuple=False)
        if debug_relu_focus.shape[0] == 0:
            debug_relu_focus = debug_relu_bad
        debug_relu_focus_diff = debug_relu_abs_diff[
            debug_relu_focus[:, 0], debug_relu_focus[:, 1]
        ]
        debug_relu_focus_order = torch.argsort(
            debug_relu_focus_diff, descending=True
        )
        debug_relu_focus = debug_relu_focus[debug_relu_focus_order]
        debug_rows_per_aiv = Nidx1 // 2
        debug_n_block = context.n_split
        debug_bw_col_block = min(debug_n_block, 64)
        debug_bw_col_count = debug_n_block // debug_bw_col_block
        debug_n_num = K // debug_n_block
        debug_relu_tile_ids = (
            (debug_relu_focus[:, 0] // debug_rows_per_aiv)
            * (debug_n_num * debug_bw_col_count)
            + debug_relu_focus[:, 1] // debug_bw_col_block
        )
        debug_relu_unique_tiles, debug_relu_tile_counts = torch.unique(
            debug_relu_tile_ids, return_counts=True
        )
        debug_relu_tile_order = torch.argsort(
            debug_relu_tile_counts, descending=True
        )
        debug_relu_tile_summary = []
        for debug_tile_order_id in debug_relu_tile_order[:16].cpu().tolist():
            debug_tile_id = int(
                debug_relu_unique_tiles[debug_tile_order_id].item()
            )
            debug_tile_segment = debug_tile_id % (
                debug_n_num * debug_bw_col_count
            )
            debug_relu_tile_summary.append(
                (
                    debug_tile_id // (debug_n_num * debug_bw_col_count),
                    debug_tile_segment // debug_bw_col_count,
                    debug_tile_segment % debug_bw_col_count,
                    int(debug_relu_tile_counts[debug_tile_order_id].item()),
                )
            )
        print(
            "[DEBUG_TASK] reluGrad bad tiles "
            "(aiv,n_idx,bw_col,count)="
            f"{debug_relu_tile_summary}"
        )
        for debug_bad_id in range(min(16, debug_relu_focus.shape[0])):
            debug_head = int(debug_relu_focus[debug_bad_id, 0].item())
            debug_k = int(debug_relu_focus[debug_bad_id, 1].item())
            debug_aiv = debug_head // debug_rows_per_aiv
            debug_local_head = debug_head % debug_rows_per_aiv
            debug_n_idx = debug_k // debug_n_block
            debug_bw_col = (
                debug_k % debug_n_block
            ) // debug_bw_col_block
            debug_local_col = debug_k % debug_bw_col_block
            debug_unmasked = (
                weights[debug_b, debug_q, debug_head].float()
                * debug_diff[debug_k]
            )
            print(
                f"[DEBUG_TASK] reluGrad error[{debug_bad_id}]: "
                f"coord=(h={debug_head},k={debug_k}) "
                f"tile=(aiv={debug_aiv},n_idx={debug_n_idx},"
                f"bw_col={debug_bw_col}) "
                f"local=(h={debug_local_head},group={debug_local_head // 4},"
                f"col={debug_local_col}) "
                f"actual={debug_relu[debug_head, debug_k].item():.6e} "
                f"expected={debug_expected_relu[debug_head, debug_k].item():.6e} "
                f"unmasked={debug_unmasked.item():.6e} "
                f"bmm2={debug_bmm2[debug_head, debug_k].item():.6e} "
                f"diff={debug_diff[debug_k].item():.6e} "
                f"weight={weights[debug_b, debug_q, debug_head].float().item():.6e} "
                f"valid={debug_valid_mask[debug_k].item():.0f}"
            )

    _debug_task_compare(
        "C2 dQueryAccum vs stored reluGrad",
        d_query_index_accum[debug_b, debug_q],
        debug_expected_dq,
    )
    _debug_task_compare(
        "C2 dk_partial vs stored reluGrad",
        dk_matmul_partial[debug_b, debug_q],
        debug_expected_dk,
    )


def _debug_tensor(name: str, tensor: torch.Tensor) -> None:
    tensor_fp32 = tensor.float()
    nan_count = torch.isnan(tensor_fp32).sum().item()
    inf_count = torch.isinf(tensor_fp32).sum().item()
    finite_tensor = torch.nan_to_num(
        tensor_fp32, nan=0.0, posinf=0.0, neginf=0.0
    )
    print(
        f"[DEBUG] {name}: shape={tuple(tensor.shape)}, "
        f"nan={nan_count}, inf={inf_count}, "
        f"finite_abs_max={finite_tensor.abs().max().item():.6e}"
    )


def _run_full_diagnostics(context: DebugContext) -> None:
    """运行一次完整工作区诊断。"""
    query = context.query
    key = context.key
    query_index = context.query_index
    key_index = context.key_index
    weights = context.weights
    sparse_indices = context.sparse_indices
    softmax_max = context.softmax_max
    softmax_sum = context.softmax_sum
    key_rope = context.key_rope
    k_real_size = context.k_real_size
    gather_key_ws = context.gather_key_ws
    gather_key_rope_ws = context.gather_key_rope_ws
    gather_key_index_ws = context.gather_key_index_ws
    bmm1_res_ws = context.bmm1_res_ws
    bmm2_res_ws = context.bmm2_res_ws
    p_reduced_ws = context.p_reduced_ws
    v1_stats_ws = context.v1_stats_ws
    dweight_partial_ws = context.dweight_partial_ws
    relu_grad_out = context.relu_grad_out
    dk_matmul_partial = context.dk_matmul_partial
    d_query_index_accum = context.d_query_index_accum
    d_key_index_accum = context.d_key_index_accum
    loss_partial_ws = context.loss_partial_ws
    d_weight = context.d_weight
    loss_out = context.loss_out
    full_valid = context.full_valid
    B, S1 = query.shape[:2]
    N2 = key.shape[2]
    Nidx2 = key_index.shape[2]
    K = gather_key_ws.shape[2]
    DQuery = query.shape[-1]
    DQueryIndex = query_index.shape[-1]
    d_rope = key_rope.shape[-1]

    expected_gk, expected_gki, expected_gkr = context.preprocess_gather(
        key,
        key_index,
        key_rope,
        sparse_indices[..., :K],
        B,
        S1,
        context.source_s2,
        N2,
        Nidx2,
        K,
        DQuery,
        DQueryIndex,
        context.d_rope_orig,
        query.dtype,
    )
    if context.d_rope_orig > 0 and d_rope > context.d_rope_orig:
        expected_gkr = torch.nn.functional.pad(
            expected_gkr,
            (0, d_rope - context.d_rope_orig),
        )

    _debug_tensor("Key", key)
    _debug_tensor("Query", query)
    _debug_tensor("QueryIndex", query_index)
    _debug_tensor("KeyIndex", key_index)
    _debug_tensor("Weights", weights)
    if full_valid:
        print("[DEBUG] FullValidStaticPath: enabled")
    else:
        _debug_tensor("KRealSize", k_real_size)
    _debug_tensor("SoftmaxMax", softmax_max)
    _debug_tensor("SoftmaxSum", softmax_sum)
    _debug_tensor("GatherKey_ws", gather_key_ws)
    _debug_tensor("GatherKeyRope_ws", gather_key_rope_ws)
    _debug_tensor("GatherKeyIndex_ws", gather_key_index_ws)
    _debug_tensor("bmm1Res", bmm1_res_ws)
    _debug_tensor("bmm2Res", bmm2_res_ws)
    _debug_tensor("PReduced", p_reduced_ws)
    _debug_tensor("V1Stats", v1_stats_ws)
    _debug_tensor("DWeightPartial", dweight_partial_ws)
    _debug_tensor("reluGradRes", relu_grad_out)
    _debug_tensor("dk_matmul_partial", dk_matmul_partial)
    _debug_tensor("dQueryIndexAccum", d_query_index_accum)
    _debug_tensor("dKeyIndexAccum", d_key_index_accum)
    _debug_tensor("dWeightOut", d_weight)
    _debug_tensor("LossPartial", loss_partial_ws)
    _debug_tensor("LossOut", loss_out)

    print(
        f"[DEBUG] Gather sentinel remaining: "
        f"Key={(gather_key_ws == context.gather_sentinel).sum().item()}, "
        f"Rope={(gather_key_rope_ws == context.gather_sentinel).sum().item()}, "
        f"KeyIndex={(gather_key_index_ws == context.gather_sentinel).sum().item()}"
    )

    gk_diff = (gather_key_ws.float() - expected_gk.float()).abs()
    gki_diff = (gather_key_index_ws.float() - expected_gki.float()).abs()
    gkr_diff = (gather_key_rope_ws.float() - expected_gkr.float()).abs()
    debug_softmax_max = softmax_max[:, 0, :, :].float()
    debug_softmax_sum = softmax_sum[:, 0, :, :].float().abs().clamp_min(1e-8)
    debug_scaled_bmm1 = bmm1_res_ws.float() * context.scale_value
    debug_valid_mask = (
        torch.ones((B, S1, K), dtype=torch.float32, device=query.device)
        if full_valid
        else (
            torch.arange(K, dtype=torch.int32, device=query.device).view(1, 1, K)
            < k_real_size.unsqueeze(-1)
        ).float()
    )
    debug_p_reduced = (
        torch.exp(debug_scaled_bmm1 - debug_softmax_max.unsqueeze(-1))
        / debug_softmax_sum.unsqueeze(-1)
    ).mean(dim=2) * debug_valid_mask
    debug_reduce_sum_y = (
        bmm2_res_ws.float() * weights.float().unsqueeze(-1)
    ).sum(dim=2)
    debug_reduce_sum_y = debug_reduce_sum_y.masked_fill(
        debug_valid_mask == 0,
        -1e30,
    )
    debug_softmax_i = (
        torch.softmax(debug_reduce_sum_y, dim=-1) * debug_valid_mask
    )
    debug_diff = debug_softmax_i - debug_p_reduced
    debug_relu_grad = (
        weights.float().unsqueeze(-1)
        * debug_diff.unsqueeze(2)
        * (bmm2_res_ws.float() > 0).float()
    )
    debug_d_weight = (
        bmm2_res_ws.float() * debug_diff.unsqueeze(2)
    ).sum(dim=-1)
    debug_p_safe = debug_p_reduced.clamp_min(1e-8)
    debug_softmax_i_safe = debug_softmax_i.clamp_min(1e-8)
    debug_loss = (
        debug_p_reduced
        * (torch.log(debug_p_safe) - torch.log(debug_softmax_i_safe))
    ).sum()
    debug_loss_partials = loss_partial_ws[
        :B * S1 * 2 * context.loss_partial_stride:context.loss_partial_stride
    ]
    p_reduced_diff = (p_reduced_ws.float() - debug_p_reduced).abs()
    relu_grad_diff = (relu_grad_out.float() - debug_relu_grad).abs()
    debug_relu_grad_bf16 = debug_relu_grad.to(relu_grad_out.dtype).float()
    relu_grad_bf16_diff = (
        relu_grad_out.float() - debug_relu_grad_bf16
    ).abs()
    debug_unmasked_rg = weights.float().unsqueeze(-1) * debug_diff.unsqueeze(2)
    debug_expected_mask = bmm2_res_ws.float() > 0
    debug_actual_rg = relu_grad_out.float()
    debug_strong_candidate = debug_unmasked_rg.abs() > 1e-3
    debug_false_zero = (
        debug_expected_mask
        & debug_strong_candidate
        & (debug_actual_rg == 0)
    )
    debug_false_nonzero = (
        (~debug_expected_mask) & (debug_actual_rg.abs() > 1e-3)
    )

    # 根据保存的 BF16 reluGrad 重建 C2，用于区分 V1 和 C2 的误差来源。
    debug_dq_from_stored_rg = torch.matmul(
        debug_actual_rg,
        gather_key_index_ws.float(),
    )
    debug_dk_from_stored_rg = torch.matmul(
        debug_actual_rg.transpose(-2, -1),
        query_index.float(),
    )
    debug_c2_dq_diff = (
        d_query_index_accum.float() - debug_dq_from_stored_rg
    ).abs()
    debug_c2_dk_diff = (
        dk_matmul_partial.float() - debug_dk_from_stored_rg
    ).abs()
    d_weight_diff = (d_weight.float() - debug_d_weight).abs()
    loss_diff = (loss_out.float().reshape(-1)[0] - debug_loss).abs()
    p_max_diff = (
        debug_scaled_bmm1.amax(dim=-1) - debug_softmax_max
    ).abs()

    print(
        f"[DEBUG] GatherKey_ws vs host_gather: "
        f"max_diff={gk_diff.max().item():.6e}, "
        f"mean_diff={gk_diff.mean().item():.6e}"
    )
    print(
        f"[DEBUG] GatherKeyIndex_ws vs host_gather: "
        f"max_diff={gki_diff.max().item():.6e}, "
        f"mean_diff={gki_diff.mean().item():.6e}"
    )
    print(
        f"[DEBUG] GatherKeyRope_ws vs host_gather: "
        f"max_diff={gkr_diff.max().item():.6e}, "
        f"mean_diff={gkr_diff.mean().item():.6e}"
    )
    print(
        f"[DEBUG] scaled bmm1 row max vs SoftmaxMax: "
        f"max_diff={p_max_diff.max().item():.6e}, "
        f"mean_diff={p_max_diff.mean().item():.6e}"
    )
    print(
        f"[DEBUG] PReduced vs host formula: "
        f"max_diff={p_reduced_diff.max().item():.6e}, "
        f"mean_diff={p_reduced_diff.mean().item():.6e}"
    )
    print(
        f"[DEBUG] bmm2Res ReLU check: "
        f"negative={(bmm2_res_ws < 0).sum().item()}, "
        f"min={bmm2_res_ws.min().item():.6e}"
    )
    print(
        f"[DEBUG] reluGradRes vs host formula: "
        f"max_diff={relu_grad_diff.max().item():.6e}, "
        f"mean_diff={relu_grad_diff.mean().item():.6e}"
    )

    relu_max_flat = int(relu_grad_bf16_diff.reshape(-1).argmax().item())
    relu_max_coords = [0] * relu_grad_bf16_diff.ndim
    relu_remaining = relu_max_flat
    for dim_idx in range(relu_grad_bf16_diff.ndim - 1, -1, -1):
        dim_size = relu_grad_bf16_diff.shape[dim_idx]
        relu_max_coords[dim_idx] = relu_remaining % dim_size
        relu_remaining //= dim_size
    relu_max_coords = tuple(relu_max_coords)
    relu_b, relu_s1, relu_g, relu_k = relu_max_coords
    print(
        f"[DEBUG] reluGradRes vs BF16-rounded host formula: "
        f"max_diff={relu_grad_bf16_diff.max().item():.6e}, "
        f"mean_diff={relu_grad_bf16_diff.mean().item():.6e}, "
        f"diff_gt_1e-3={(relu_grad_bf16_diff > 1e-3).sum().item()}, "
        f"diff_gt_1e-2={(relu_grad_bf16_diff > 1e-2).sum().item()}"
    )
    print(
        f"[DEBUG] reluGrad max-error coord={relu_max_coords}: "
        f"actual={debug_actual_rg[relu_max_coords].item():.6e}, "
        f"expected_bf16={debug_relu_grad_bf16[relu_max_coords].item():.6e}, "
        f"unmasked={debug_unmasked_rg[relu_max_coords].item():.6e}, "
        f"bmm2={bmm2_res_ws[relu_max_coords].float().item():.6e}, "
        f"P={debug_p_reduced[relu_b, relu_s1, relu_k].item():.6e}, "
        f"SY={debug_softmax_i[relu_b, relu_s1, relu_k].item():.6e}, "
        f"weight={weights[relu_b, relu_s1, relu_g].float().item():.6e}"
    )
    print(
        f"[DEBUG] reluGrad inferred mask disagreement: "
        f"false_zero={debug_false_zero.sum().item()}, "
        f"false_nonzero={debug_false_nonzero.sum().item()}"
    )
    print(
        f"[DEBUG] C2 dQueryAccum vs host(stored reluGrad): "
        f"max_diff={debug_c2_dq_diff.max().item():.6e}, "
        f"mean_diff={debug_c2_dq_diff.mean().item():.6e}"
    )
    print(
        f"[DEBUG] C2 dk_partial vs host(stored reluGrad): "
        f"max_diff={debug_c2_dk_diff.max().item():.6e}, "
        f"mean_diff={debug_c2_dk_diff.mean().item():.6e}"
    )
    print(
        f"[DEBUG] dWeightOut vs host formula: "
        f"max_diff={d_weight_diff.max().item():.6e}, "
        f"mean_diff={d_weight_diff.mean().item():.6e}"
    )
    print(
        f"[DEBUG] aligned LossPartial sum vs host formula: "
        f"partial_sum={debug_loss_partials.sum().item():.6e}, "
        f"abs_diff={(debug_loss_partials.sum() - debug_loss).abs().item():.6e}"
    )
    print(
        f"[DEBUG] LossOut vs host formula: "
        f"abs_diff={loss_diff.item():.6e}"
    )
