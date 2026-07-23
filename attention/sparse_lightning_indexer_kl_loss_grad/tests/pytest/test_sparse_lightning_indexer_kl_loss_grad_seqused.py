"""
Regression tests for arch22 seqused_q/seqused_k support.

The result produced with physical allocation lengths plus seqused_q/seqused_k
is compared batch by batch with the existing no-seqused path using compact
inputs. This verifies both effective-length selection and physical addressing.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


def find_repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "build.sh").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()
TORCH_EXTENSION_PATH = REPO_ROOT / "torch_extension"
if TORCH_EXTENSION_PATH.exists():
    sys.path.insert(0, str(TORCH_EXTENSION_PATH))


N1 = 8
N2 = 1
HEAD_DIM = 128
TOPK = 512
METADATA_TOTAL_SIZE_INDEX = 1


@dataclass(frozen=True)
class SeqUsedCase:
    name: str
    layout: str
    alloc_q: Tuple[int, ...]
    alloc_k: Tuple[int, ...]
    used_q: Optional[Tuple[int, ...]]
    used_k: Optional[Tuple[int, ...]]
    dtype: torch.dtype = torch.float16
    mask_mode: int = 0
    cmp_ratio: int = 1
    cmp_residual_k: Optional[Tuple[int, ...]] = None

    @property
    def batch_size(self) -> int:
        return len(self.alloc_q)

    @property
    def effective_q(self) -> Tuple[int, ...]:
        return self.alloc_q if self.used_q is None else self.used_q

    @property
    def effective_k(self) -> Tuple[int, ...]:
        return self.alloc_k if self.used_k is None else self.used_k

    @property
    def residual_k(self) -> Tuple[int, ...]:
        if self.cmp_residual_k is None:
            return (0,) * self.batch_size
        return self.cmp_residual_k

    def validate(self) -> None:
        if len(self.alloc_k) != self.batch_size:
            raise ValueError("alloc_q and alloc_k must have the same batch size")
        if self.used_q is not None and len(self.used_q) != self.batch_size:
            raise ValueError("used_q must have one element per batch")
        if self.used_k is not None and len(self.used_k) != self.batch_size:
            raise ValueError("used_k must have one element per batch")
        if len(self.residual_k) != self.batch_size:
            raise ValueError("cmp_residual_k must have one element per batch")
        if self.layout == "BSND":
            if len(set(self.alloc_q)) != 1 or len(set(self.alloc_k)) != 1:
                raise ValueError("BSND allocation lengths must be uniform")
        elif self.layout != "TND":
            raise ValueError(f"unsupported layout: {self.layout}")
        for used, allocated in zip(self.effective_q, self.alloc_q):
            if used < 0 or used > allocated:
                raise ValueError(f"used_q {used} is outside [0, {allocated}]")
        for used, allocated in zip(self.effective_k, self.alloc_k):
            if used < 0 or used > allocated:
                raise ValueError(f"used_k {used} is outside [0, {allocated}]")


CASES = (
    SeqUsedCase(
        name="tnd_q_only",
        layout="TND",
        alloc_q=(12, 9),
        alloc_k=(5, 4),
        used_q=(7, 5),
        used_k=None,
    ),
    SeqUsedCase(
        name="tnd_k_only_including_zero",
        layout="TND",
        alloc_q=(8, 7),
        alloc_k=(5, 6),
        used_q=None,
        used_k=(0, 3),
    ),
    SeqUsedCase(
        name="tnd_q_and_k_causal_bf16",
        layout="TND",
        alloc_q=(16, 15),
        alloc_k=(6, 6),
        used_q=(10, 13),
        used_k=(3, 4),
        dtype=torch.bfloat16,
        mask_mode=3,
        cmp_ratio=4,
        cmp_residual_k=(1, 2),
    ),
    SeqUsedCase(
        name="bsnd_q_and_k",
        layout="BSND",
        alloc_q=(10, 10),
        alloc_k=(6, 6),
        used_q=(6, 8),
        used_k=(4, 3),
    ),
)


@pytest.fixture(scope="module")
def npu_device() -> torch.device:
    if not torch.npu.is_available():
        pytest.skip("NPU device is unavailable")
    torch_npu.npu.set_device(0)
    pytest.importorskip("cann_ops_transformer")
    return torch.device("npu:0")


def cumulative_lengths(lengths: Tuple[int, ...]) -> Tuple[int, ...]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return tuple(result)


def make_cu_seqlens(lengths: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(cumulative_lengths(lengths), dtype=torch.int32, device=device)


def trunc_div(numerator: int, denominator: int) -> int:
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


def valid_k_count(
    case: SeqUsedCase,
    batch_index: int,
    local_q_index: int,
) -> int:
    used_q = case.effective_q[batch_index]
    used_k = case.effective_k[batch_index]
    if case.mask_mode == 0:
        sparse_len = used_k
    else:
        pre_compress_k = used_k * case.cmp_ratio + case.residual_k[batch_index]
        sparse_len = trunc_div(
            pre_compress_k - used_q + local_q_index + 1,
            case.cmp_ratio,
        )
    return max(0, min(sparse_len, used_k, TOPK))


def make_batch_inputs(
    case: SeqUsedCase,
) -> List[Dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(20260723)
    batches = []
    for batch_index, (alloc_q, alloc_k) in enumerate(zip(case.alloc_q, case.alloc_k)):
        q = torch.rand(
            alloc_q,
            N1,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        ).mul_(0.25).to(case.dtype)
        k = torch.rand(
            alloc_k,
            N2,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        ).mul_(0.25).to(case.dtype)
        w = torch.rand(
            alloc_q,
            N1,
            generator=generator,
            dtype=torch.float32,
        ).mul_(0.02).add_(0.01)

        # Unused Q rows deliberately contain nonzero inputs. Zero output there
        # must therefore come from seqused handling, not from the test data.
        sparse_indices = torch.zeros(alloc_q, N2, TOPK, dtype=torch.int32)
        softmax_l1 = torch.rand(
            alloc_q,
            N2,
            TOPK,
            generator=generator,
            dtype=torch.float32,
        )
        softmax_l1 /= softmax_l1.sum(dim=-1, keepdim=True)

        used_q = case.effective_q[batch_index]
        for local_q_index in range(used_q):
            count = valid_k_count(case, batch_index, local_q_index)
            sparse_indices[local_q_index].fill_(-1)
            softmax_l1[local_q_index].zero_()
            if count == 0:
                continue
            sparse_indices[local_q_index, 0, :count] = torch.arange(
                count,
                dtype=torch.int32,
            )
            values = torch.rand(count, generator=generator, dtype=torch.float32)
            softmax_l1[local_q_index, 0, :count] = values / values.sum()

        batches.append(
            {
                "q": q.contiguous(),
                "k": k.contiguous(),
                "w": w.contiguous(),
                "sparse_indices": sparse_indices.contiguous(),
                "softmax_l1": softmax_l1.contiguous(),
            }
        )
    return batches


def pack_batches(
    batches: List[Dict[str, torch.Tensor]],
    layout: str,
) -> Dict[str, torch.Tensor]:
    packed = {}
    for name in batches[0]:
        tensors = [batch[name] for batch in batches]
        packed[name] = (
            torch.cat(tensors, dim=0).contiguous()
            if layout == "TND"
            else torch.stack(tensors, dim=0).contiguous()
        )
    return packed


def unpack_output(
    tensor: torch.Tensor,
    layout: str,
    alloc_lengths: Tuple[int, ...],
) -> List[torch.Tensor]:
    if layout == "BSND":
        return [tensor[index] for index in range(tensor.shape[0])]

    offsets = cumulative_lengths(alloc_lengths)
    return [
        tensor[offsets[index] : offsets[index + 1]]
        for index in range(len(alloc_lengths))
    ]


def run_operator(
    inputs: Dict[str, torch.Tensor],
    layout: str,
    alloc_q: Tuple[int, ...],
    alloc_k: Tuple[int, ...],
    device: torch.device,
    *,
    used_q: Optional[Tuple[int, ...]] = None,
    used_k: Optional[Tuple[int, ...]] = None,
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    cmp_residual_k: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    batch_size = len(alloc_q)
    optional_inputs = {}
    if layout == "TND":
        optional_inputs["cu_seqlens_q"] = make_cu_seqlens(alloc_q, device)
        optional_inputs["cu_seqlens_k"] = make_cu_seqlens(alloc_k, device)
    if used_q is not None:
        optional_inputs["seqused_q"] = torch.tensor(
            used_q,
            dtype=torch.int32,
            device=device,
        )
    if used_k is not None:
        optional_inputs["seqused_k"] = torch.tensor(
            used_k,
            dtype=torch.int32,
            device=device,
        )
    if cmp_residual_k is not None:
        optional_inputs["cmp_residual_k"] = torch.tensor(
            cmp_residual_k,
            dtype=torch.int32,
            device=device,
        )

    metadata = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
        N1,
        N2,
        HEAD_DIM,
        batch_size=batch_size,
        max_seqlen_q=max(alloc_q),
        max_seqlen_k=max(alloc_k),
        topk=TOPK,
        layout_q=layout,
        layout_k=layout,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        **optional_inputs,
    )

    npu_inputs = {
        name: tensor.to(device)
        for name, tensor in inputs.items()
    }
    outputs = torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
        npu_inputs["q"],
        npu_inputs["k"],
        npu_inputs["w"],
        npu_inputs["sparse_indices"],
        npu_inputs["softmax_l1"],
        metadata=metadata,
        layout_q=layout,
        layout_k=layout,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        **optional_inputs,
    )
    torch_npu.npu.synchronize()
    return metadata.cpu(), tuple(output.cpu() for output in outputs)


def compact_batch(
    batch: Dict[str, torch.Tensor],
    used_q: int,
    used_k: int,
    layout: str,
) -> Dict[str, torch.Tensor]:
    compact = {
        "q": batch["q"][:used_q],
        "k": batch["k"][:used_k],
        "w": batch["w"][:used_q],
        "sparse_indices": batch["sparse_indices"][:used_q],
        "softmax_l1": batch["softmax_l1"][:used_q],
    }
    if layout == "BSND":
        compact = {
            name: tensor.unsqueeze(0).contiguous()
            for name, tensor in compact.items()
        }
    return compact


def assert_output_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=0.08,
        atol=0.01,
        msg=lambda message: f"{name} mismatch: {message}",
    )


def assert_exact_zero(name: str, tensor: torch.Tensor) -> None:
    nonzero = int(torch.count_nonzero(tensor).item())
    assert nonzero == 0, f"{name} contains {nonzero} nonzero elements"


@pytest.mark.ci
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_sparse_lightning_indexer_kl_loss_grad_seqused(
    case: SeqUsedCase,
    npu_device: torch.device,
) -> None:
    case.validate()

    batches = make_batch_inputs(case)
    packed_inputs = pack_batches(batches, case.layout)
    metadata, actual_outputs = run_operator(
        packed_inputs,
        case.layout,
        case.alloc_q,
        case.alloc_k,
        npu_device,
        used_q=case.used_q,
        used_k=case.used_k,
        mask_mode=case.mask_mode,
        cmp_ratio=case.cmp_ratio,
        cmp_residual_k=case.cmp_residual_k,
    )

    expected_total_tasks = (
        sum(case.used_q)
        if case.used_q is not None
        else sum(case.alloc_q)
    )
    assert int(metadata[METADATA_TOTAL_SIZE_INDEX]) == expected_total_tasks

    dq_batches = unpack_output(actual_outputs[0], case.layout, case.alloc_q)
    dk_batches = unpack_output(actual_outputs[1], case.layout, case.alloc_k)
    dw_batches = unpack_output(actual_outputs[2], case.layout, case.alloc_q)
    softmax_batches = unpack_output(actual_outputs[3], case.layout, case.alloc_q)

    for batch_index, (used_q, used_k) in enumerate(
        zip(case.effective_q, case.effective_k)
    ):
        if used_q > 0 and used_k > 0:
            baseline_inputs = compact_batch(
                batches[batch_index],
                used_q,
                used_k,
                case.layout,
            )
            residual = (
                (case.residual_k[batch_index],)
                if case.cmp_residual_k is not None
                else None
            )
            _, baseline_outputs = run_operator(
                baseline_inputs,
                case.layout,
                (used_q,),
                (used_k,),
                npu_device,
                mask_mode=case.mask_mode,
                cmp_ratio=case.cmp_ratio,
                cmp_residual_k=residual,
            )
            baseline_dq = baseline_outputs[0][0] if case.layout == "BSND" else baseline_outputs[0]
            baseline_dk = baseline_outputs[1][0] if case.layout == "BSND" else baseline_outputs[1]
            baseline_dw = baseline_outputs[2][0] if case.layout == "BSND" else baseline_outputs[2]
            baseline_softmax = (
                baseline_outputs[3][0]
                if case.layout == "BSND"
                else baseline_outputs[3]
            )
            assert_output_close(
                f"batch {batch_index} dq",
                dq_batches[batch_index][:used_q],
                baseline_dq,
            )
            assert_output_close(
                f"batch {batch_index} dk",
                dk_batches[batch_index][:used_k],
                baseline_dk,
            )
            assert_output_close(
                f"batch {batch_index} dw",
                dw_batches[batch_index][:used_q],
                baseline_dw,
            )
            assert_output_close(
                f"batch {batch_index} softmax_out",
                softmax_batches[batch_index][:used_q],
                baseline_softmax,
            )
        else:
            assert_exact_zero(
                f"batch {batch_index} dq with empty effective sequence",
                dq_batches[batch_index][:used_q],
            )
            assert_exact_zero(
                f"batch {batch_index} dk with empty effective sequence",
                dk_batches[batch_index][:used_k],
            )
            assert_exact_zero(
                f"batch {batch_index} dw with empty effective sequence",
                dw_batches[batch_index][:used_q],
            )
            assert_exact_zero(
                f"batch {batch_index} softmax_out with empty effective sequence",
                softmax_batches[batch_index][:used_q],
            )

        assert_exact_zero(
            f"batch {batch_index} dq unused Q tail",
            dq_batches[batch_index][used_q:],
        )
        assert_exact_zero(
            f"batch {batch_index} dw unused Q tail",
            dw_batches[batch_index][used_q:],
        )
        assert_exact_zero(
            f"batch {batch_index} softmax_out unused Q tail",
            softmax_batches[batch_index][used_q:],
        )
        assert_exact_zero(
            f"batch {batch_index} dk unused K tail",
            dk_batches[batch_index][used_k:],
        )
