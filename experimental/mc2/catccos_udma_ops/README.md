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
independently pinned to the official `cann/shmem`
[`release/v1.6.0`](https://gitcode.com/cann/shmem/tree/release/v1.6.0) branch tip
`0132a6f504c2035d9c1a04100e0d7254fd2968e4`, verified on 2026-09-08. Its upstream
`VERSION` is `1.6.0`; the snapshot includes fixes after the `v1.6.0` tag.
The exact commit is used for both device headers and bundled SHMEM libraries,
and is recorded in the package's `_build_versions.py` together with
`SHMEM_REF = "release/v1.6.0"`.
Builds stay pinned to this commit when the upstream branch advances.
The operator-local `csrc/catccos_overlay` directory contains only the
UDMA all-to-all-v headers that have not yet landed in that official revision.
The GroupedMatMul-AllToAllV configuration is migrated from CatCCOS commit
`ee01c5181c50ce2aeb5feb05d603b568eb31a61a`; the fork is a source reference,
not a build dependency. The optional AllToAllV output and its pipelined export
follow CatCCOS commits `1d96ae736f1054a28b0ddf3cbb96cd0c64c81a21` and
`2444c7ee5acee145c8a296e3462b4bcf487571b8` respectively.

## Build and install

On an Ascend950 build host, source the CANN and torch_npu environments first.
Building SHMEM requires CMake 3.19 or newer from `PATH`; a system or Conda
installation is supported and the PyPI `cmake` package is not required.
The pinned SHMEM branch documents CANN 9.1.0+ and Ascend HDK 25.5.6+ for
Ascend950 UDMA; see its
[environment requirements](https://gitcode.com/cann/shmem/blob/0132a6f504c2035d9c1a04100e0d7254fd2968e4/docs/quickstart.md).

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
without a matching build marker. When switching from the previous 1.7.0
snapshot, rerun both preparation and Wheel building using the commands above;
the old SHMEM libraries and Wheel must be rebuilt. This SHMEM snapshot also
requires the CANN-provided `libhcomm.so` on both build and target hosts.
The target host only installs the Wheel; it does not need either source checkout,
`SHMEM_HOME_PATH`, or a manually generated rootinfo file.

The custom MTE/UDMA communication blocks use peer-scoped completion before
publishing a software completion signal. In SHMEM 1.6.0 the generic
`aclshmem_fence()` and `aclshmem_quiet()` only drain local pipelines and flush
the data cache; they do not wait for UDMA transfers. These workers explicitly
drain their own pending UDMA peer through `aclshmemx_udma_quiet(peer)` and use
`aclshmemx_mte_quiet()` for local memory completion. Mid-loop completion keeps
the local MTE events alive for subsequent tiles. The UDMA tile calls use the
release branch's single QP 0 per peer and its `PIPE_S` / `PIPE_MTE3` overloads;
no master-only UDMA QP configuration, batching, or operation-config APIs are
required. CQ error handling follows the release implementation.
Host initialization preserves the release header's optional-attribute version
and timeout defaults, and selects UDMA explicitly.

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

## Search tiling on the deployment machines

Both operators accept an optional `tiling=` mapping. Without it, the built-in
profiles below remain the default. Search follows the finite compiled-instance
enumeration and parameter filtering in
[CatCCOS tests/dynamic_tiling](https://gitcode.com/cann/catccos/tree/master/tests/dynamic_tiling)
(reference revision `4aef53a9d28e1eb62b2749945945440b9aabf76c`). It uses the
UDMA operators and validation cases in this Wheel, including the export path.
Rebuild and reinstall this Wheel on every participating host first; an older
extension does not expose the tiling override or hardware query.

Install the search dependency in the target Python environment:

```bash
python3 -m pip install PyYAML
```

From this subproject directory, first inspect a plan without loading Torch or
using an NPU. `--aic-cores` is an optional hardware estimate for a dry run; in a
real run it must match the queried AI-core count.

```bash
bash scripts/search_tiling.sh --dry-run --aic-cores 32 \
  --max-candidates 8 --output-dir /tmp/udma_tiling_plan
```

Run the search on the customer's machines with their actual rank placement and
network configuration. The default manifests cover both operators with small
two-rank cases and eight-rank model cases. The following selects one bundled model case:

```bash
bash scripts/search_tiling.sh \
  --cases validation/cases/tiling_business.yaml \
  --case-id a2av_gmm_m131072_k2048_n1024_export \
  --strategy random --max-candidates 64 --seed 2026 \
  --topology-label customer-cluster-a \
  --timeout 1200 --output-dir /tmp/udma_tiling_customer
```

The installed equivalent is `ops-transformer-udma-search-tiling`, or
`python3 -m validation.search_tiling`. `--cases` is repeatable. Copy and edit a
manifest for the customer's M/K/N, expert count, world size, export mode and input
distribution. Search sampling is controlled separately by the CLI options
below; the manifest's performance sampling remains for `run_performance.sh`.
The current validation generator
supports **balanced BF16 routes, TP=1 and at most eight ranks**. Its measured
winner is specific to that workload; it does not characterize an unbalanced
production route. Search runs in **performance-only mode**: it does not build
CPU Golden results, compare outputs or generate accuracy reports for any
candidate, the built-in fallback or the selected winner. Manifest `accuracy`
settings are used by the separate accuracy workflow and do not run during search.

`--space validation/cases/tiling_space_example.json` customizes the Cartesian
space. Unspecified dimensions keep their defaults. `mmad` selects an existing
`[M0, N0, K0]` instance; changing runtime parameters does not rebuild a kernel.
`--strategy random` samples without replacement using `--seed`;
`--strategy exhaustive` visits the grid in order. `--max-candidates` is a
per-case cap **including the valid built-in fallback**; its default is 64.
Use `--strategy exhaustive --max-candidates 0` to visit the complete legal
default space without a candidate cap. The tool reports the best **successfully measured** configuration,
not a guarantee beyond the tested space. Hardware-equivalent candidates are
deduplicated before applying the budget.

To search the complete default space for all four model cases on one eight-NPU
host, run the following from this subproject directory. The cases execute in
sequence and each reuses its own inputs and runtime:

```bash
bash scripts/search_tiling.sh \
  --cases validation/cases/alltoallv_grouped_mat_mul.yaml \
  --cases validation/cases/grouped_mat_mul_all_to_allv.yaml \
  --case-id a2av_gmm_model_case_1 \
  --case-id a2av_gmm_model_case_2 \
  --case-id gmm_a2av_model_case_1 \
  --case-id gmm_a2av_model_case_2 \
  --strategy exhaustive --max-candidates 0 \
  --nnodes 1 --nproc-per-node 8 --batch-size 64 \
  --timeout 1200 --output-dir /tmp/udma_tiling_full
```

| Operator | Search dimensions |
|---|---|
| AllToAllV → GroupedMatMul | `mmad`: 128×256×256, 256×128×256 or 256×256×256; `comm_block_m`, `stage_rows`, `local_copy_tile_rows`, `self_copy_cores` |
| AllToAllV → GroupedMatMul with export | All the above, plus `workspace_stages` 2/3, `m_split` 1/2/4 and `export_core_count` (0 means all AI cores) |
| GroupedMatMul → AllToAllV | `mmad`: 128×256×256 or 256×128×256; `comm_interval`, `local_copy_tile_rows` |

Non-export AllToAllV uses two workspace stages, `m_split=1` and
`export_core_count=0`. The current inverse operator always uses two workspace
stages. Historical NPU/data split fields, the inverse operator's comm block M,
and AllToAllV's legacy interval are not search dimensions because the selected
kernel paths do not consume them independently.

The launcher probes actual AI-core count and UB capacity before generating
candidates. Both Python and native code check compiled instances, staging,
worker pools, UB and the 400 MiB symmetric workspace. All ranks must run the
same operator build and have the same AI-core count; the inverse operator's
communication geometry depends on that count. Each case normally starts one
persistent `torchrun` process group. Gloo, SHMEM, the input, weight, route,
output and communication workspace are reused while changing tilings. All
ranks check the same session plan before launching any candidate. Successful
measurement requires complete, finite timing samples from every rank.

The default sampling uses `--warmup 10` once per session, then
`--align-iterations 3 --iterations 3 --repeats 3` for each candidate: 18
operator calls per candidate after the shared warmup. Increase the measured
iterations or repeats when closer candidates need more stable measurements.
Up to `--batch-size 64` candidates are submitted before waiting for the
device and gathering rank timing samples. Host alignment occurs before each
candidate batch; primer calls precede each measured repeat to reduce rank
start skew. The timeout is a no-progress limit for startup, a candidate batch,
or teardown, so `--timeout 1200` allows a search to run longer than 1200 seconds
as long as it keeps committing batches.

Preflight rejection, including a changed software or hardware environment,
stops immediately. On a single node, an execution failure preserves committed batches and retries the
first uncommitted batch in fresh processes one candidate at a time, then
continues the remaining candidates in a new persistent session. This isolates
bad candidates without paying process startup for every successful candidate.
A multi-node session failure stops the search on the participating nodes;
automatic recovery and resume are currently single-node only.

Timing uses NPU events around each candidate's repeated preallocated-output calls.
For each measured repeat it takes the **maximum per-call time across ranks**,
then reports the median across repeats for that candidate. Host rendezvous and input creation are outside the
timed interval. This is a stream/batch measurement and is
different from the kernel-only `msprof` metric in `run_performance.sh`; do not
compare their absolute values as the same metric.
The built-in fallback is measured through the same explicit-tiling API as
other candidates when reporting `speedup_vs_default`.

The output directory contains `search_manifest.json`, session logs and
immutable `batch_*.json` checkpoints under each case's session directories.
Each checkpoint includes the completed candidates' rank timing samples.
`results.jsonl` and `summary.csv` are appended as candidates finish;
`best_tilings.json` and a compact progress file are refreshed after each
batch. The full `summary.json` is written when the search ends, avoiding
rewriting all historical results for each candidate.
Session results include `phase_timings_s` for preparing inputs and warming up;
batch checkpoints include each rank's enqueue and device-wait times. These
separate one-time setup and host overhead from the per-candidate NPU latency.
Hardware/software versions, device/rank placement, build
hashes and `--topology-label` are recorded. Set that label to identify cabling,
network settings or topology changes that device APIs cannot discover.
Single-node searches can resume with the same arguments plus `--resume`;
`--retry-failed` retries previously failed candidates. Resume re-probes the
machines and validates the immutable checkpoints before skipping completed
candidates. It rejects a changed workload, environment, sampling, execution
mode or search plan. Old per-candidate search output cannot be resumed with
the persistent search implementation; choose a new output directory.
Results record `search_mode="performance_only"`,
`execution_mode="persistent_batches_v1"` and `accuracy_checked=false`.
The `passed`/`valid` fields in `best_tilings.json` mean measurement succeeded;
they do not claim numerical correctness. The existing accuracy scripts remain
available independently of search when output validation is needed.

For two hosts with four NPUs each, run this on **both hosts**, setting
`NODE_RANK=0` on the first and `NODE_RANK=1` on the second:

```bash
bash scripts/search_tiling.sh \
  --cases validation/cases/tiling_business.yaml \
  --case-id gmm_a2av_m131072_k1024_n2048 \
  --nnodes 2 --node-rank "$NODE_RANK" --nproc-per-node 4 \
  --master-addr 10.0.0.10 --run-id customer-search-001 \
  --max-candidates 64 --timeout 1200 \
  --topology-label customer-cluster-a --output-dir /tmp/udma_tiling_customer
```

Replace the address with the first host's reachable IPv4 address. Keep all
other search arguments identical. Torchrun and SHMEM use separate ports that
advance per probe/session, so the number of tiling candidates does not exhaust
the port range. Results are aggregated through Gloo and written under each
host's `node_0`/`node_1` directory; a shared filesystem is not required.
Multi-node resume is currently unsupported.
If changing the base ports, `--master-port` and `--udma-master-port` must have
opposite parity, ensuring their two advancing port sequences never overlap.

Load the chosen configuration once and explicitly pass it on every rank:

```python
tiling = udma.load_tiling_config(
    "/tmp/udma_tiling_customer/best_tilings.json",
    "a2av_gmm_m131072_k2048_n1024_export",
)
output, dispatched = udma.all_to_allv_grouped_mat_mul(
    x, weight, local_route, global_route,
    ep_size=world_size, expert_num=expert_num, received_rows=received_rows,
    out=output, output_all_to_allv=True, all_to_allv_out=dispatched,
    tiling=tiling,
)
```

`grouped_mat_mul_all_to_allv(..., tiling=tiling)` uses the same loading pattern
with its own case ID. Use the same configuration on every rank and re-search
when the deployment workload or topology changes. There is no automatic
shape-only cache that could silently select a profile measured elsewhere.

## Built-in business-shape profiles

The four model-case defaults use the winners from the customer's
`tiling_full_20260909_012623` search on 2026-09-09, with
Ascend950DT_9582, 32 AI cores and 248 KiB of available UB per device. The
eight-rank balanced BF16 search used three primer calls, three timed calls
per repeat and three repeats. Latency is the median of the per-repeat
maximum across ranks. These are performance-only results
(`accuracy_checked=false`); the independent accuracy workflow remains
available to validate outputs after rebuilding and reinstalling the Wheel.

| Case | Winning candidate | Search latency (us) |
|---|---|---:|
| `a2av_gmm_model_case_1` (export enabled) | `986c5d67aef6c9a2` | 1671.049 |
| `a2av_gmm_model_case_2` (export enabled) | `55f6ea82e55c35f2` | 1615.745 |
| `gmm_a2av_model_case_1` | `f1d10672a756a4a9` | 1991.315 |
| `gmm_a2av_model_case_2` | `fdfb758ae9b102f9` | 1637.861 |

`all_to_allv_grouped_mat_mul` selects the following tuned profiles only when
`world_size=8`, `expert_num=256`, and `(M, K, N)` matches exactly. Export and
non-export calls select separate records. All other shapes and topologies
retain the default profile.

| AllToAllV output | M | K | N | M0 | N0 | K0 | comm block M | stage rows | WS stages | local-copy rows | M split | export cores | self-copy cores |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| disabled | 131072 | 2048 | 1024 | 128 | 256 | 256 | 128 | 1536 | 2 | 64 | 1 | 0 | 1 |
| enabled | 131072 | 2048 | 1024 | 256 | 256 | 256 | 32 | 1280 | 3 | 16 | 1 | 24 | 8 |
| disabled | 131072 | 2048 | 512 | 256 | 128 | 256 | 128 | 1536 | 2 | 64 | 1 | 0 | 1 |
| enabled | 131072 | 2048 | 512 | 256 | 128 | 256 | 64 | 1280 | 2 | 8 | 1 | 0 | 8 |

For AllToAllV-GroupedMatMul, `commNpuSplit`, `commDataSplit`, and the explicit
`commTileM` value are compatibility fields: the migrated kernel consumes
`commBlockM` and `localCopyTileRows` instead. The table lists the active
parameters; `export cores=0` means all AI cores. The search result's execution
mode is not part of this kernel ABI. Both model-case winners were measured
with export enabled. The non-export records retain the earlier non-export tuning.

`grouped_mat_mul_all_to_allv` independently selects these profiles for the
same `world_size=8`, `expert_num=256` topology. Because imbalanced routing can
give every Rank a different `x.shape[0]`, profile selection uses only K and N;
all peers therefore keep the same M0 and communication schedule. The M column
below records the balanced workload used for tuning and is not a runtime
selection predicate. Other K/N shapes and topologies retain the default profile.

| Tuning M | K | N | M0 | N0 | K0 | comm interval | local-copy rows |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 131072 | 1024 | 2048 | 128 | 256 | 256 | 10 | 4 |
| 131072 | 512 | 2048 | 128 | 256 | 256 | 16 | 16 |

The grouped-matmul host forwards both split values through `commCoreSplit` for
ABI compatibility. The target-order scheduler currently does
not use them or `commBlockM` to partition transfers. The active parameters are
the compiled MMAD shape, `commInterval`, and the local copy rows derived from
`commTileM / 2` (capped by UB capacity for the built-in fallback). The two
new records therefore store `commTileM=8` and `commTileM=32`, respectively;
their local-copy rows both fit the search machine's 248 KiB available UB.
The Python search baselines use the same effective parameters as these
native defaults, so subsequent searches include and measure them as default
candidates when the deployment hardware passes validation.
