# CatCCOS UDMA operators for PyTorch

This standalone project delivers the Ascend950 BF16
`alltoallvgroupedmatmul` and `groupedmatmulalltoallv` UDMA BF16 kernels as an
installable Wheel. The Wheel contains the C++/AscendC extension (`_C.abi3.so`)
and the SHMEM shared libraries required at runtime. The actual device launch
remains a Fast Kernel Launch expression:
`KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(...)`.

The CatCCOS dependency comes from the official
`https://gitcode.com/cann/catccos.git` repository and is pinned to
`878c2e0a504be8c49c03b5e1157a7c714a628e29`; its CATLASS gitlink is used, so
this project does not change the repository-wide CATLASS version. SHMEM is
independently pinned to the official `cann/shmem` master snapshot
`2a59c8fdcce9b717114fe445be09c96a13a55c39`, checked on 2026-09-04. Its upstream
`VERSION` is `1.7.0`; this is an unreleased master commit. The exact commit is
used for both device headers and bundled SHMEM libraries, and is recorded in
the package's `_build_versions.py` together with `SHMEM_REF = "master"`.
The operator-local `csrc/catccos_overlay` directory contains only the
UDMA all-to-all-v headers that have not yet landed in that official revision.
The GroupedMatMul-AllToAllV configuration is migrated from CatCCOS commit
`ee01c5181c50ce2aeb5feb05d603b568eb31a61a`; the fork is a source reference,
not a build dependency. The optional AllToAllV output and its pipelined export
follow CatCCOS commits `1d96ae736f1054a28b0ddf3cbb96cd0c64c81a21` and
`2444c7ee5acee145c8a296e3462b4bcf487571b8` respectively.

## Build and install

On an Ascend950 build host, source the CANN and torch_npu environments first.
The build requires CMake 3.18.4 or newer from `PATH`; a system or Conda
installation is supported and the PyPI `cmake` package is not required.

```bash
cd experimental/mc2/catccos_udma_ops
bash scripts/prepare_dependencies.sh
export CATCCOS_SOURCE_PATH="$PWD/../../../3rdparty/catccos"
export SHMEM_SOURCE_PATH="$PWD/../../../3rdparty/shmem"
export SHMEM_HOME_PATH="$SHMEM_SOURCE_PATH/install"
bash scripts/build_wheel.sh
python3 -m pip install --force-reinstall --no-deps dist/*.whl
```

`build_wheel.sh` removes only this subproject's previous `build`, `dist`,
`egg-info`, staged extension, and staged runtime libraries before rebuilding.
It then rejects a Wheel unless its filename and sole `dist-info/METADATA` entry
match the package version in `ops_transformer_catccos_udma/_version.py`.

`prepare_dependencies.sh` refuses an existing CatCCOS checkout at another
revision. It moves a clean SHMEM checkout to the pinned commit, fetching the
commit when needed; local changes and untracked source files are rejected.
SHMEM is built with `-soc_type Ascend950` and the same C++ ABI setting as the
active PyTorch installation. A successful build records the source commit;
the Wheel build rejects mismatched source, modified headers, or libraries
without a matching build marker. Rerun dependency preparation when upgrading
an existing installation. This SHMEM snapshot also requires the CANN-provided
`libhcomm.so` on both build and target hosts.
The target host only installs the Wheel; it does not need either source checkout,
`SHMEM_HOME_PATH`, or a manually generated rootinfo file.

The custom MTE/UDMA communication blocks use peer-scoped completion before
publishing a software completion signal. SHMEM master makes the generic
`aclshmem_fence()` and `aclshmem_quiet()` poll all reachable peers, which would
cross the fused kernels' AIV queue ownership boundaries. These workers instead
drain their own pending UDMA peer through `aclshmemx_udma_quiet(peer)` and use
`aclshmemx_mte_quiet()` for local memory completion. Mid-loop completion keeps
the local MTE events alive for subsequent tiles. The existing UDMA tile calls
retain the default QP 0 and inherit upstream's CQ error/timeout trap behavior.

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
local_tokens, global_tokens, received_rows = udma.build_rank_route(
    m, rank, world_size, expert_num, device
)

output = udma.all_to_allv_grouped_mat_mul(
    x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num,
    received_rows=received_rows,
)
torch_npu.npu.synchronize()
udma.destroy_udma_group()
```

`x` must be packed by global expert in the order described by
`local_tokens_per_expert`. `weight` is `[local_expert_num, k, n]`. The returned
tensor has exact shape `[received_rows, n]`, where `received_rows` equals
`global_tokens_per_local_expert.sum()`.
Route inputs may be host integer sequences or contiguous `torch.int64` tensors.
The facade converts host routes to the same NPU as `x`; NPU routes must already
reside there. Values must be non-negative and fit in the kernel's `uint32` shape
range. Pass a host-known `received_rows` with NPU routes to avoid a device-to-host
synchronization during output allocation. Host sequences are convenient for
one-off calls. When AllToAllV export is enabled, a host `global_tokens` also
builds a 12,880-byte CatCCOS prefix descriptor and performs one additional H2D
so the kernel can use its precomputed-metadata path. An NPU `global_tokens`
never triggers D2H and instead selects the compatible kernel fallback. Repeated
launches should reuse device route tensors and pass the same host-known row
count to avoid per-call route allocation and H2D copies.

A directly runnable multi-process example is available at
`examples/run_all_to_allv_grouped_mat_mul.py`. It constructs balanced test
routing and accepts `--m`, `--k`, `--n`, and `--expert-num`; model integration
should replace the generated route tables with the router's real count tensors.

```bash
UDMA_MASTER_ADDR=127.0.0.1 UDMA_MASTER_PORT=29601 \
python3 -m torch.distributed.run \
  --master-addr=127.0.0.1 --master-port=29500 --nproc-per-node=8 \
  examples/run_all_to_allv_grouped_mat_mul.py \
  --m 131072 --k 2048 --n 1024 --expert-num 256
```

To test coexistence with an already initialized HCCL communicator, use
`examples/run_hccl_then_all_to_allv_grouped_mat_mul.py`. It validates one HCCL
all-gather first, keeps that process group alive, and then initializes UDMA and
calls AllToAllV-GroupedMatMul. Its shape can also be supplied through
`A2AV_GMM_M`, `A2AV_GMM_K`, `A2AV_GMM_N`, and `A2AV_GMM_EXPERT_NUM`; this avoids
the ambiguous `--m` abbreviation in some `torch.distributed.run` versions.

`x` is an ordinary contiguous BF16 NPU tensor. The kernel uses it directly as
the local source of UDMA PUT operations; only the internal remote-receive
workspace and synchronization state use symmetric SHMEM memory. There is no
NPU-to-SHMEM input staging copy. For repeated launches, reuse the output to
avoid output allocation in the timed region:

```python
out = torch.empty(received_rows, n, dtype=torch.bfloat16, device=device)
output = udma.all_to_allv_grouped_mat_mul(
    x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num,
    received_rows=received_rows, out=out,
)
```

The device-side `local_tokens` and `global_tokens` are contiguous one-dimensional
`torch.int64` tensors with shape `[expert_num]`. The global table is flattened
in `[source_rank][local_expert]` order, matching the kernel's linear read ABI.
The public facade also accepts CPU tensors and host integer sequences, and
normalizes tensor inputs with `reshape(-1).contiguous()` before dispatch.

To additionally obtain the AllToAllV activation consumed by grouped matmul,
enable the optional export path:

```python
output, all_to_allv_output = udma.all_to_allv_grouped_mat_mul(
    x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num,
    received_rows=received_rows,
    output_all_to_allv=True,
)
```

`all_to_allv_output` is an ordinary BF16 NPU tensor with shape
`[global_tokens.sum(), k]`. Its rows are ordered by local expert first and
source rank second. When `output_all_to_allv` is omitted or false, the original
compile-time kernel variant is used and no export allocation or copy is issued.
For repeated calls, pass preallocated `out=` and `all_to_allv_out=` tensors.

The bundled accuracy and performance manifests enable this export path by
default for AllToAllV-GroupedMatMul. Set `output_all_to_allv: false` on a case
to benchmark or validate the original non-export kernel explicitly. Performance
preallocates both outputs before warm-up, so allocation and route-count
preprocessing are excluded from the profiled launch loop.

`grouped_mat_mul_all_to_allv` performs the inverse order: local grouped
matmuls followed by UDMA AllToAllV. Its `x` and `weight` inputs remain ordinary
contiguous BF16 NPU tensors. UDMA GET writes the result directly, so the result
is an ordinary NPU tensor; only the internal remotely readable workspace uses
the registered SHMEM heap:

```python
result = udma.grouped_mat_mul_all_to_allv(
    x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num,
)

# Reuse an ordinary NPU output so allocation stays outside repeated launches.
output_rows = int(local_tokens.sum().item())
reusable = torch.empty((output_rows, n), dtype=torch.bfloat16, device=x.device)
udma.grouped_mat_mul_all_to_allv(
    x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num, out=reusable,
)
consume(result, reusable)
```

For this direction, `x` has `sum(global_tokens_per_local_expert)` rows grouped
by local expert, while the output independently has
`sum(local_tokens_per_expert)` rows. This supports imbalanced routing without
requiring the input and output row totals to match. The kernel stages complete
N-width row ranges in destination order, so each active source rank uses one
contiguous UDMA GET per communication round instead of one request per output
row. Every Rank must allocate and release registered tensors in the same order.
The runtime rejects finalization while such tensors are alive instead of
leaving dangling PyTorch storage.

Like the forward direction, both route inputs accept an NPU `torch.int64`
tensor, a CPU `torch.int64` tensor, or a host integer sequence. Host routes are
converted to NPU int64 tensors on every call. If `local_tokens` is host-resident,
the facade uses its sum to allocate the exact output and dispatches the reusable
out variant, avoiding the C++ route-count D2H synchronization. Reuse NPU route
tensors and an explicit `out=` in performance-critical loops.

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

The SHMEM heap contains only the fixed communication workspace and
synchronization state. Inputs and outputs of both public operators use ordinary
NPU memory, so larger model tensors do not increase the SHMEM heap requirement.
Validation defaults to a 1 GiB heap; `UDMA_HEAP_SIZE` can override that runtime
configuration when the internal workspace layout changes.

Pass/fail follows the
[opbase experimental precision standard](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)
for BF16 exactly: mixed
tolerance `rtol=atol=2^-6` must match at least 99% of meaningful output
elements. Every element must also satisfy the BF16 hard limit
`abs_error <= max(1.0, 32 * ULP_at_abs_golden)`; the ULP is evaluated at that
element's Golden magnitude. Operator outputs are exact-size tensors, so every
returned row participates in the comparison. Every path/rank writes
`report.json`, a Top-K error CSV
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

For `grouped_mat_mul_all_to_allv`, the input and output row counts are distinct
under unbalanced routing: `x.shape[0] == sum(global_tokens)` and
`out.shape[0] == sum(local_tokens)`. Supplying a correctly sized reusable
`out=` avoids an additional device-to-host route-count reduction in the launch
path.

## Fixed business-shape tiling

`all_to_allv_grouped_mat_mul` selects the following tuned profiles only when
`world_size=8`, `expert_num=256`, and `(M, K, N)` matches exactly. Export and
non-export calls select separate records. All other shapes and topologies
retain the default profile.

| AllToAllV output | M | K | N | MMAD id | M split | M0 | N0 | K0 | interval | comm tile M | comm block M | NPU/data split | stage rows | WS stages | local-copy rows | export cores | self-copy cores |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| disabled | 131072 | 2048 | 1024 | 0 | 1 | 128 | 256 | 256 | 12 | 128 | 128 | 4/4 | 1536 | 2 | 64 | 0 | 1 |
| enabled | 131072 | 2048 | 1024 | 2 | 1 | 256 | 256 | 256 | 8 | 32 | 32 | 1/16 | 2048 | 2 | 16 | 16 | 8 |
| disabled | 131072 | 2048 | 512 | 1 | 1 | 256 | 128 | 256 | 6 | 128 | 128 | 4/4 | 1536 | 2 | 64 | 0 | 1 |
| enabled | 131072 | 2048 | 512 | 2 | 2 | 256 | 256 | 256 | 12 | 32 | 32 | 1/16 | 3072 | 3 | 16 | 32 | 4 |

For AllToAllV-GroupedMatMul, `commNpuSplit`, `commDataSplit`, and the explicit
`commTileM` value are compatibility fields: the migrated kernel consumes
`commBlockM` and `localCopyTileRows` instead. MMAD id 2 maps to the compiled
`256x256x256` instance. The search result's execution mode is not part of this
kernel ABI. The new search was performed through the training/export entry, so
the non-export profiles remain unchanged.

`grouped_mat_mul_all_to_allv` independently selects these profiles for the
same `world_size=8`, `expert_num=256` topology. Because imbalanced routing can
give every Rank a different `x.shape[0]`, profile selection uses only K and N;
all peers therefore keep the same M0 and communication schedule. The M column
below records the balanced workload used for tuning and is not a runtime
selection predicate. Other K/N shapes and topologies retain the default profile.

| Tuning M | K | N | M0 | N0 | K0 | commInterval | commTileM | commBlockM | commNpuSplit | commDataSplit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 131072 | 1024 | 2048 | 128 | 256 | 256 | 14 | 64 | 64 | 4 | 4 |
| 131072 | 512 | 2048 | 256 | 128 | 256 | 12 | 16 | 16 | 1 | 20 |

The grouped-matmul host forwards both split values through `commCoreSplit` for
ABI parity with the tuning record. The target-order scheduler currently does
not use them to partition transfers; M0, `commInterval`, `commTileM`, and
`commBlockM` are effective in the selected kernel path.
