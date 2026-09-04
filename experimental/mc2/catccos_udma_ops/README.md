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
independently pinned to the official `cann/shmem` `release/v1.6.0` commit
`7fa19662da0093587d9a332e740b91d6c929cdcf`, verified on 2026-09-04.
Upstream `VERSION` is `1.6.0`; the pin includes maintenance commits after the
original v1.6.0 tag, including the UDMA SQ wrap fix (!726), runtime-sized
software barrier pool (!729), and transport metadata/channel fixes. It does
not follow the unreleased 1.7.0 master branch. The earlier rootinfo fixes for
the 2048-byte capacity limit and POD/Server EID ports remain included.
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

The two EP8/256-expert GroupedMatMul-AllToAllV delivery profiles use independent
symmetric staging slots and do not recycle a slot within one launch. The
`K=1024, N=2048` profile uses 19 slots; the `K=512, N=2048` profile uses 22.
Choose the producer synchronization protocol when building the Wheel:

```bash
# Recommended/default: no reuse plus an all-PE producer barrier.
CATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER=1 \
  bash scripts/build_wheel.sh

# Experimental lower-overhead version: PIPE_FIX + local SyncAll + signal.
CATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER=0 \
  bash scripts/build_wheel.sh
```

The choice is compiled into the device kernel; there is no runtime branch or
public API difference. The second variant must pass repeated accuracy tests on
the target topology before delivery. Both variants compile out root-cause
records, producer-trace copies, diagnostic delays and device printf unless the
corresponding diagnostic build switches are explicitly enabled. Other shapes
retain the general two-stage kernel. The no-reuse profiles support at most 19
and 22 communication rounds respectively; their current delivery target is the
balanced `M=131072` business case.

`build_wheel.sh` removes only this subproject's previous `build`, `dist`,
`egg-info`, staged extension, and staged runtime libraries before rebuilding.
It then rejects a Wheel unless its filename and sole `dist-info/METADATA` entry
match the package version in `ops_transformer_catccos_udma/_version.py`.

Production builds compile out every device-side `AscendC::printf` call and
disable the AscendC Dump buffer. To inspect GET completion and ready-wait context,
build and install a separate diagnostic Wheel:

```bash
CATCCOS_UDMA_DEVICE_DIAGNOSTICS=1 bash scripts/build_wheel.sh
python3 -m pip install --force-reinstall --no-deps dist/*.whl
```

Both UDMA tiles complete through public `aclshmemx_udma_quiet(peer)`, leaving
QP selection and direct/relay handling to SHMEM. There is no local QP table
access or CQ polling implementation. Tiles expose void `Finalize(peer)`;
communication blocks use void `FinalizeBlockLoop()` before completion
publication or buffer reuse. There is no CQ status return or status-based
completion guard. The diagnostic Wheel enables ready-wait diagnostics. Device
printf reserves additional HBM and affects timing, so do not use that Wheel
for performance measurement or model delivery.

`prepare_dependencies.sh` keeps the CatCCOS revision fixed and switches a clean
SHMEM checkout to the pinned maintenance commit, including a checkout currently
on master. It refuses local SHMEM changes instead of
overwriting them. SHMEM is rebuilt with `-soc_type Ascend950` and the same C++
ABI setting as the active PyTorch installation. The build verifies that the
extension's device headers and bundled user-space SHMEM libraries come from
the same pinned revision. Run dependency preparation again when upgrading an
existing checkout, then rebuild and reinstall the Wheel; replacing only the
SHMEM shared libraries does not update the device code compiled into `_C`.
The target host only installs the Wheel; it does not need either source checkout,
`SHMEM_HOME_PATH`, or a manually generated rootinfo file.

The maintenance branch's generic `aclshmem_fence`/`aclshmem_quiet` still flush
local pipelines and cache; they do not drain UDMA peer queues. The fused
kernels retain their existing peer completion points and scheduling. The
contiguous strong-ready path still publishes payload and generation in one
WRITE_WITH_NOTIFY request. This update does not enable UDMA multi-QP or import
master's operation-config/CQE-counter changes.

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

The export kernel uses the production `strong_ready` completion protocol by
default. Each remote peer slice and its 64-bit per-stage/source generation are
published by one UDMA WRITE_WITH_NOTIFY request. An entry SHMEM barrier runs
immediately before the reusable MMAD/communication gates are cleared, followed
by the existing post-initialization barrier. Staging keeps the default L2
policy; cache bypass is not part of the production path. The
`CATCCOS_UDMA_SYNC_DIAGNOSTIC_MODE` environment variable is reserved for A/B
diagnosis (`baseline`, `barrier_only`, `generation_after_quiet`,
`strong_ready`, or `strong_ready_cache_safe`). Calls that do not request the
AllToAllV output retain their independently validated non-export completion
path.

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
and performs CPU FP32 matmul followed by BF16 conversion.

Ordinary accuracy defaults to two automatically allocated calls: `ordinary`
plus `auto_repeat_0`. Before each call every Rank enters the same Gloo barrier;
after the operator completes, each returned tensor is cloned to an immutable
NPU snapshot. Consequently no Rank can begin call N+1 until every Rank has
finished and preserved call N, and every per-call report describes that exact
global invocation. Unaligned consecutive launches are intentionally confined
to `scripts/run_sync_diagnosis.sh`, which passes `--automatic-only` to diagnose
cross-Rank launch skew and cross-invocation synchronization.

Reusable caller-provided outputs are not exercised by ordinary accuracy. Add
`--check-reusable-out` when that overload is required; `--memory-safety`
enables it automatically. Only those reusable outputs are prefilled with NaN
to detect unwritten regions. Full free-HBM NaN poisoning, guard allocations and
mutation checks are likewise exclusive to `--memory-safety`, so they do not
inflate an ordinary accuracy run:

```bash
bash scripts/run_accuracy.sh --case-id model_case_1
bash scripts/run_accuracy.sh --case-id model_case_1 --check-reusable-out
```

The SHMEM heap contains only the fixed communication workspace and
synchronization state. Inputs and outputs of both public operators use ordinary
NPU memory, so larger model tensors do not increase the SHMEM heap requirement.
The runtime reserves a 640 MiB data plane, leaves a 60 MiB guard gap, and puts
a disjoint 4 KiB control plane at offset 700 MiB. The two fused directions own
separate 2 KiB control slices, so their reusable signals cannot alias either
communication payloads or each other's signals. Validation defaults to a 1 GiB
heap and requires more than 764 MiB including allocator headroom;
`UDMA_HEAP_SIZE` can override that runtime configuration.

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
The same spatial breakdowns are also emitted as `mismatch_row_runs.csv`,
`by_local_expert.csv`, and `by_source_rank.csv`. Large failures retain exact selected output rows in
`failure_dump_compact.pt` instead of dropping the dump. `mismatch_row_runs` in
each report identifies contiguous failing row intervals and whether they align
with route-segment boundaries. Every Rank also writes `launch_diagnostics.json`
with launch order, allocation mode, output addresses and shapes, plus
`input_diagnostics.json` with route values, tensor metadata, sampled mutation
fingerprints and exact post-failure NaN/Inf scans. The per-Rank `summary.json`
contains an evidence-based failure-scope hint, while the batch root contains
machine-readable `summary.json` and `summary.csv`. The launcher also tees the
complete combined torchrun stdout/stderr into `torchrun_ep<N>.log` and records
its exact command, ports and heap selection in `launch_ep<N>.json`, so a copied
accuracy directory retains failures that occur before a Rank can write its
report. Each Rank writes `timings.json`, splitting input preparation, Golden
generation, pre-launch barriers, operator launches, output snapshots, accuracy
analysis and artifact writes. The batch root writes
`launch_ep<N>_timings.json` for torchrun and runtime-log collection, making a
slow run attributable without enabling a profiler.

Pass/fail counts and maximum error always use every meaningful output element.
For tensors larger than 1,048,576 elements, diagnostic median/p90/p99 values use
a deterministic uniform sample capped at that size; `report.json` records the
total, finite and sampled element counts plus `quantiles_approximate`. Mean and
maximum remain exact full-tensor reductions.

### Model-level fused/reference dump comparison

To locate the first NaN or numerical divergence after model integration, dump
the inputs and outputs at the same semantic boundaries in the fused UDMA run
and in the unfused AllToAllV + grouped-matmul run. The model dump layout below
is recognized directly:

```text
dump_dir/
└── rank_00000/
    ├── mc2/
    │   └── first_call_00000/
    │       ├── hidden_states_before_a2a_gemm.pt
    │       ├── hidden_states_after_a2a_gemm.pt
    │       ├── hidden_states_before_gemm_a2a.pt
    │       └── hidden_states_after_gemm_a2a.pt
    └── mc2_udma/
        └── first_call_00000/
            └── <the same four files>
```

`mc2` is treated as the reference and `mc2_udma` as the fused result. The
`a2a_gemm` before/after files form the AllToAllV-GroupedMatMul input/output
boundary; `gemm_a2a` forms the GroupedMatMul-AllToAllV boundary. Analyze all
ranks with one argument:

The call directory may also carry the operator direction and execution phase,
for example `first_call_a2avgmm_backward` or
`first_call_gmma2av_backward`. Common separator and abbreviation variants such
as `first-call-gmma2av-bwd` are accepted. These directories are assigned
sequence zero and the `backward` phase, while the operator name is inferred
independently. `first_call_00000` remains the separate `numbered_call` phase, so
the three directories never merge into one invocation even when their tensor
filenames are identical. The simpler `first_call_forward` and
`first_call_backward` forms remain supported.

Operator-explicit directory names do not require the `call` token. For example,
`first_all2all_grouped_matmul_backward_000000` and
`first_grouped_matmul_all2all_backward_000000` are recognized as sequence zero,
backward invocations of the corresponding operator.

```bash
bash scripts/analyze_model_dumps.sh \
  --dump-dir /data/dumps/dump_dir \
  --output-dir /data/dumps/analysis
```

Both ordinary `torch.save` files and byte-split files named `*.pt.part.0`,
`*.pt.part.1`, ... are accepted. Separate fused/reference trees remain
supported as an alternative:

```bash
bash scripts/analyze_model_dumps.sh \
  --fused-root /data/dumps/udma \
  --reference-root /data/dumps/unfused \
  --output-dir /data/dumps/analysis
```

After installing the validation extra, the equivalent command is
`ops-transformer-udma-analyze-dumps`.

Automatic pairing ignores markers such as `udma`, `fused`, `reference`, and
`baseline`, and normalizes both operator spellings. When both kinds of files
share one directory, select disjoint sets explicitly:

```bash
bash scripts/analyze_model_dumps.sh \
  --fused-root /data/dumps/all \
  --reference-root /data/dumps/all \
  --fused-include 'udma' \
  --reference-include '(^|/|_)reference(_|/|$)'
```

For an unambiguous runtime timeline, pass matching JSON or JSONL manifests with
one record per file. `pair_key` must be identical on the two sides; the other
fields drive causal grouping and reporting:

```json
{"file":"rank_0/000017_a2av_gmm/input_x.pt","pair_key":"r0-c17-a2av-x","rank":0,"sequence":17,"op":"all_to_allv_grouped_mat_mul","invocation":"rank0-call17-a2av-gmm","io":"input","tensor":"x"}
{"file":"rank_0/000017_a2av_gmm/output.pt","pair_key":"r0-c17-a2av-out","rank":0,"sequence":17,"op":"all_to_allv_grouped_mat_mul","invocation":"rank0-call17-a2av-gmm","io":"output","tensor":"output"}
```

Use `--fused-manifest fused.jsonl --reference-manifest reference.jsonl` to load
them. The analyzer does not import torch or torch_npu. It reassembles split
files on disk and performs bounded-memory NumPy reductions in chunks. Inputs
are required to be bitwise identical; floating outputs use the corresponding
OpBase experimental dtype threshold. Route tensors and the exported
AllToAllV activation remain bitwise-exact comparisons because they are data
movement results rather than arithmetic outputs. It reports inherited
nonfinite values separately from finite-reference values newly changed to
NaN/Inf.

The main artifacts are:

- `report.md`: first divergent invocation and evidence-based diagnosis;
- `timeline.csv`: every rank/invocation classified as matched, input already
  divergent, operator output divergent, or operator-introduced nonfinite;
- `comparisons.csv`: shape, dtype, exact/precision status, NaN/Inf counts,
  cosine similarity, relative L2 error and first failing coordinate;
- `top_mismatches.csv` and `introduced_nonfinite.csv`: concrete values and
  multidimensional indices for debugging layout/tile boundaries;
- `row_distribution.csv`, `row_runs.csv`, and
  `column_new_nan_distribution.csv`: spatial concentration and contiguous
  failing regions;
- `summary.json`: the complete machine-readable report, including row runs,
  row periodicity, unmatched files and load errors.

If paired inputs are identical and an output first introduces NaN/Inf, the
fault is localized to that fused invocation. For AllToAllV-GroupedMatMul, a bad
exported AllToAllV tensor points before GMM (routing, communication, staging or
completion); a correct export with a bad final output points to the GMM side.
If an input is already different, inspect the preceding producer or run
reproducibility before attributing the issue to the current operator.

For the balanced GroupedMatMul-AllToAllV synchronization failure, run the
single-process-lifetime no-reuse/delay matrix. It generates each Rank's input,
final-output Golden and local-GMM producer Golden once, then reuses the same
resident tensors for every mode and launch:

```bash
bash scripts/run_gmm_a2av_root_cause.sh \
  --cases validation/cases/grouped_mat_mul_all_to_allv.yaml \
  --case-id model_case_1 \
  --launches-per-mode 3 \
  --build-install
```

`--build-install` rebuilds and force-installs the Wheel before the one
`torchrun`; it automatically sets
`CATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE=1`. When building manually, set the same
environment variable before `scripts/build_wheel.sh`; the validation-only raw
operator is compiled out of ordinary delivery Wheels. The default matrix runs:

- `no_reuse_trace`: a dedicated profile-specific 19/22-stage kernel gives every
  `commIdx` an independent symmetric stage, disables per-round recycle, waits
  once at the end, and copies the retained producer staging into a regular NPU
  tensor;
- `pre_get_delay_medium`: 20,000 cycles after producer-ready wait and before
  GET;
- `post_quiet_delay_{short,medium,long}`: 2,000/20,000/200,000 cycles after
  public UDMA quiet and before completion publication;
- `baseline`: the original two-stage local-`SyncAll` diagnostic protocol.

Use repeated `--mode <name>` options to select a subset. Barrier modes require
`route: balanced` because every Rank must execute identical communication-loop
counts. Delays are diagnostics, not delivery fixes.

Every launch allocates a 1 MiB regular NPU diagnostic tensor. Device AIVs write
per-round/source records containing actual rows, output offset, GET bytes,
completion publication, delay placement, stage address/range and trace copy
completion. Diagnostic format version 3 records `quiet_completed` and
`producer_trace_copy_completed` as progress flags, with no CQ error codes.
The matching validation scripts reject older diagnostic tensor versions. After
stream synchronization Host writes these records to
`rank_<N>/modes/<mode>/launch_<I>/diagnostic.json`. Device printf is not used,
so the experiment does not reserve the AscendC printf buffer. For no-reuse,
`producer_trace/report.json`, breakdown CSVs, optional failure rows, and
`producer_trace/layout.json` distinguish a bad local GMM/staging payload from a
bad GET/final output. `summary.json` classifies producer visibility,
quiet-to-recycle timing, cyclic stage reuse, or CQ/transport failure. Input and
Golden reuse identities are recorded in `reuse_manifest.json`.

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
