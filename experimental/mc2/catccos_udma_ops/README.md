# CatCCOS UDMA operators for PyTorch

This standalone project delivers the Ascend950 BF16
`alltoallvgroupedmatmul` and `groupedmatmulalltoallv` UDMA BF16 kernels as an
installable Wheel. The Wheel contains the C++/AscendC extension (`_C.abi3.so`)
and the SHMEM shared libraries required at runtime. The actual device launch
remains a Fast Kernel Launch expression:
`KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(...)`.

The CatCCOS dependency comes from the official
`https://gitcode.com/cann/catccos.git` repository and is pinned to
`878c2e0a504be8c49c03b5e1157a7c714a628e29`; its own CATLASS and SHMEM gitlinks
are used, so this project does not change the repository-wide CATLASS version.
The operator-local `csrc/catccos_overlay` directory contains only the
UDMA all-to-all-v headers that have not yet landed in that official revision.
The GroupedMatMul-AllToAllV configuration is migrated from CatCCOS commit
`ee01c5181c50ce2aeb5feb05d603b568eb31a61a`; the fork is a source reference,
not a build dependency.

## Build and install

On an Ascend950 build host, source the CANN and torch_npu environments first.
The build requires CMake 3.18.4 or newer from `PATH`; a system or Conda
installation is supported and the PyPI `cmake` package is not required.

```bash
cd experimental/mc2/catccos_udma_ops
bash scripts/prepare_dependencies.sh
export CATCCOS_SOURCE_PATH="$PWD/../../../3rdparty/catccos"
export SHMEM_HOME_PATH="$CATCCOS_SOURCE_PATH/3rdparty/shmem/install"
bash scripts/build_wheel.sh
python3 -m pip install --force-reinstall --no-deps dist/*.whl
```

`prepare_dependencies.sh` refuses an existing CatCCOS checkout at another
revision. SHMEM is built with `-soc_type Ascend950` and the same C++ ABI setting
as the active PyTorch installation.

## PyTorch API

Run one process per NPU. `UDMA_MASTER_ADDR` must resolve to a numeric IPv4
address reachable by every rank, and `UDMA_MASTER_PORT` must differ from
torchrun's rendezvous port. The package resolves hostnames before passing them
to SHMEM and constructs the required `tcp://<IPv4>:<port>` bootstrap URI. If
omitted, the address follows `MASTER_ADDR` and the port uses
`MASTER_PORT + 1`. The bundled single-node validation launchers explicitly use
`127.0.0.1` for both torchrun rendezvous and UDMA bootstrap, so an automatically
selected hostname cannot break SHMEM bootstrap.

```python
import os
import torch
import torch_npu
import ops_transformer_catccos_udma as udma

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch_npu.npu.set_device(local_rank)
udma.init_udma_group(rank=rank, world_size=world_size, local_rank=local_rank)

m, n, k, expert_num = 128, 256, 256, 8
device = f"npu:{local_rank}"
x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
weight = torch.randn(expert_num // world_size, k, n,
                     dtype=torch.bfloat16, device=device)
local_tokens, global_tokens = udma.build_rank_route_tensors(
    m, rank, world_size, expert_num, device
)

output = udma.all_to_allv_grouped_mat_mul(
    x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num,
)
torch_npu.npu.synchronize()
udma.destroy_udma_group()
```

`x` must be packed by global expert in the order described by
`local_tokens_per_expert`. `weight` is `[local_expert_num, k, n]`. The returned
capacity is `[world_size * m, n]`; valid rows are packed first and their count is
`global_tokens_per_local_expert.sum()`.

`x` is an ordinary contiguous BF16 NPU tensor. The kernel uses it directly as
the local source of UDMA PUT operations; only the internal remote-receive
workspace and synchronization state use symmetric SHMEM memory. There is no
NPU-to-SHMEM input staging copy. For repeated launches, reuse the output to
avoid output allocation in the timed region:

```python
out = torch.zeros(world_size * m, n, dtype=torch.bfloat16, device=device)
output = udma.all_to_allv_grouped_mat_mul(
    x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num, out=out,
)
```

`grouped_mat_mul_all_to_allv` performs the inverse order: local grouped
matmuls followed by UDMA AllToAllV. Its `x` and `weight` inputs remain ordinary
contiguous BF16 NPU tensors. UDMA GET writes the result directly, so the result
is backed by the registered SHMEM heap:

```python
import gc

result = udma.grouped_mat_mul_all_to_allv(
    x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num,
)

# Reuse registered output so allocation stays outside repeated launches.
reusable = udma.empty_symmetric((m, n))
udma.grouped_mat_mul_all_to_allv(
    x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num, out=reusable,
)
consume(result, reusable)

# Every registered tensor must be released before destroy_udma_group().
del result, reusable
gc.collect()
```

For this direction, `x` is grouped by the local experts according to the sums
of `global_tokens_per_local_expert`; the output shape is `[m, n]`. Every Rank
must allocate and release registered tensors in the same order. The runtime
rejects finalization while such tensors are alive instead of leaving dangling
PyTorch storage.

## Accuracy and performance validation

Install the validation extra together with the Wheel. The tests deliberately
import the installed package; they do not add the source directory to
`PYTHONPATH`.

```bash
python3 -m pip install --force-reinstall --no-deps dist/*.whl
python3 -m pip install 'PyYAML>=6.0' 'matplotlib>=3.5'
```

Cases use a versioned YAML manifest at
`validation/cases/alltoallv_grouped_mat_mul.yaml`. Each case defines `m/n/k`,
world size, expert count, input distribution, tags, accuracy repeats and
performance sampling. Run all accuracy cases, one case, or a tag subset with:

```bash
bash scripts/run_accuracy.sh
bash scripts/run_accuracy.sh --case-id aligned_m128_n256_k256_ep2_uniform
bash scripts/run_accuracy.sh --include-tag smoke --output-dir /tmp/udma_accuracy
```

The GroupedMatMul-AllToAllV BF16 cases use a separate manifest so their
baselines and operator names cannot be mixed with the opposite direction:

```bash
bash scripts/run_accuracy.sh \
  --cases validation/cases/grouped_mat_mul_all_to_allv.yaml
bash scripts/run_performance.sh \
  --cases validation/cases/grouped_mat_mul_all_to_allv.yaml \
  --case-id gmm_a2av_m128_n256_k256_ep2_uniform
```

The installed-Wheel equivalent is `ops-transformer-udma-accuracy`; the
performance command below is likewise available as
`ops-transformer-udma-performance`.

The validation control plane uses CPU Gloo, keeping HCCL off the device path so
it cannot contend with ACLSHMEM for UDMA/HDC resources or pollute device-side
profiling. The independent Golden gathers each source rank's packed input as an
FP32 CPU tensor, reconstructs destination-expert slices from the route matrix,
and performs CPU FP32 matmul followed by BF16 conversion. Both automatic-output
and reusable-output calls use the same ordinary input tensor and are checked.

The SHMEM heap always contains the fixed communication workspace and
synchronization state. GroupedMatMul-AllToAllV additionally allocates its
`[m, n]` UDMA destination there; its ordinary `x` and `weight` inputs do not
consume SHMEM. Validation defaults to 1 GiB. Set `UDMA_HEAP_SIZE` to an integer
byte count when a larger output requires more capacity.

Pass/fail follows the
[opbase experimental precision standard](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
for BF16 exactly: mixed
tolerance `rtol=atol=2^-6` must match at least 99% of meaningful output
elements. Every element must also satisfy the BF16 hard limit
`abs_error <= max(1.0, 32 * ULP_at_abs_golden)`; the ULP is evaluated at that
element's Golden magnitude. Padded capacity is reported separately and is not
counted as valid output. Every path/rank writes `report.json`, a Top-K error CSV
including the per-element ULP/hard limit, a downsampled error-ratio heatmap,
expert and source-rank breakdowns, and a bounded `failure_dump.pt` on failure.
The batch root contains machine-readable `summary.json` and `summary.csv`.

Pass/fail counts and maximum error always use every meaningful output element.
For tensors larger than 1,048,576 elements, diagnostic median/p90/p99 values use
a deterministic uniform sample capped at that size; `report.json` records the
total, finite and sampled element counts plus `quantiles_approximate`. Mean and
maximum remain exact full-tensor reductions.

Performance is collected from exported `msprof` `op_summary` rows, following
the official [`--task-time` collection definition](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/Profiling/atlasprofiling_16_0011.html),
rather than Python or NPU Event wall time:

```bash
bash scripts/run_performance.sh
bash scripts/run_performance.sh --case-id aligned_m128_n256_k256_ep2_uniform \
  --output-dir /tmp/udma_perf_current
```

Preprocessing, allocation and Gloo host barriers are outside the selected
operator rows. Before each measured repeat, all ranks enter a host barrier and
issue primer launches. For every measured iteration the parser aligns the
occurrence index across devices, records all device durations, and defines the
authoritative latency as the p50 of the iteration-wise maximum device duration.
This prevents the fastest card from hiding time spent by an early card waiting
inside the fused operator. If p90 cross-rank skew exceeds the case's
`max_rank_skew_ratio`, the capture is marked invalid and cannot pass regression
gating; raw per-iteration and per-device data remain in `report.json`.

For an apples-to-apples original-example baseline, run the original binary with
the same YAML case and the exact warmup/primer/measured launch layout, collect
CSV `op_summary` with `msprof`, then import it:

```bash
ops-transformer-udma-import-baseline \
  --profile-dir /path/to/original/msprof \
  --case-id aligned_m128_n256_k256_ep2_uniform \
  --output /tmp/original_baseline.json

bash scripts/run_performance.sh --baseline /tmp/original_baseline.json
```

The baseline and current performance-workload digest, SoC, CANN, PyTorch and torch_npu
fingerprints must match. A mismatch is rejected unless explicitly reviewed and
overridden. The default case threshold permits no positive latency regression.
The same launch layout is essential: the importer rejects missing or extra
target-kernel occurrences instead of silently selecting convenient samples.

Current kernel constraints are BF16, TP=1 (`ep_size == world_size`), at most 8
ranks, at most 64 local experts, contiguous row-major tensors, and the original
CatCCOS peer-memory/tiling limit.
