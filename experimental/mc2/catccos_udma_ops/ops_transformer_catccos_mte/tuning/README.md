# MTE communication search

This package supports **`mte_comm` only**. The production kernels and their
frozen GMM, macro-stage, export and self-copy geometry are shared with tuning.
Joint GMM/communication search and legacy NPU/Data split fields are rejected.
Public mathematical operator signatures have no tuning parameters.
Current timing reuses the balanced-route production performance worker;
unbalanced routes are rejected by planning and belong in accuracy/stress
validation rather than being silently timed with a different route.

`plan` works without PyTorch, an NPU, or the extension DSO. Accelerator execution
requires the explicitly built `--with-tuning` wheel; a production wheel rejects
the private override API. A plan's `block_dim` and `ub_bytes` are declarations
that every worker verifies against the actual device before setting its override.
Install the wheel's validation extras (`pip install 'ops-transformer-catccos-mte[validation]'`)
or install PyYAML, NumPy and Matplotlib in the same environment. PyYAML is needed
for YAML planning; precision artifacts also use NumPy/Matplotlib. For NPU
execution, source the CANN environment and use a compatible torch/torch_npu.

From the repository's `experimental/mc2/catccos_udma_ops` directory, the following
plans all six frozen business modes (the two A2AV-GMM shapes with and without
export, and the two GMM-A2AV shapes):

```bash
python -m ops_transformer_catccos_mte.tuning plan \
  --cases ops_transformer_catccos_mte/validation/cases/mte_frozen_profiles.yaml \
  --output plan.json
python -m ops_transformer_catccos_mte.tuning search --plan plan.json \
  --output results --budget 32 --timeout 900 --topk 3 --stress-repeats 20
python -m ops_transformer_catccos_mte.tuning search --plan plan.json \
  --output results --budget 32 --timeout 900 --topk 3 --stress-repeats 20 --resume
python -m ops_transformer_catccos_mte.tuning replay --best results/best.json \
  --output replay_results --stress-repeats 20
```

For a single mode, add an actual case ID such as
`--case-id a2avgmm_m131072_k2048_n1024_export` or
`--case-id gmma2av_m131072_k512_n2048`. The manifest is included in the installed
wheel under `ops_transformer_catccos_mte/validation/cases/`; pass its absolute
installed path when running outside the repository.

Manifest case IDs are operator-specific; use the IDs actually present in each
manifest. The saved key includes backend, operator, export mode and the resolved
case digest, so identical business case names do not collide. `--platform`
accepts a JSON object with `block_dim` (default 32), `ub_bytes` (262144),
`system_reserved_bytes` (8192), and fixed `payload_offset_bytes` (2048).

`--space` accepts a JSON object containing exactly these four arrays:

```json
{
  "commAivCount": [7, 14, 21, 28],
  "workersPerPeer": [1, 2, 3, 4],
  "commChunkBytes": [4096, 8192, 16384, 32768, 65536],
  "commUbStages": [1, 2]
}
```

`commAivCount` counts remote workers on the assigned AIV subcore, excluding self
and export workers. It must divide into `workersPerPeer`-sized groups. A group
handles peer ordinals `group, group + groups, ...`; its lanes handle chunks
`lane, lane + workersPerPeer, ...`. A2AV-GMM's remote budget is
`block_dim - selfCopyCores`; GMM-A2AV's is `block_dim - 1`. Entirely idle peer
groups are pruned. `workersPerPeer` must not exceed 64, and the declared core
count must also accommodate `rankSize + selfCopyCores - 1`. EP1 fixes count 0
and workers-per-peer 1 while continuing to search chunk size and buffer stages.
Payload chunks are 32-byte aligned and obey
`chunkBytes * ubStages <= min(rawUbBytes, 256 KiB - reservedBytes) - 2 KiB`.
With the default 8 KiB architecture reserve, both CANN's physical 256 KiB report
and its usable 248 KiB report give the same 246 KiB payload budget. The worker
checks this effective budget, so the architecture reserve is never deducted
twice. The 2 KiB signal scratch remains separate from all payload slots.
Invalid combinations are reported in `plan.json`; none are silently clamped.

Every candidate first runs the existing per-rank precision/golden validation.
Only a passing candidate is profiled using the existing msprof worker/parser.
Accuracy and profiling use different process groups; subsequent candidates
always start fresh groups. The controller logs each command and externally
terminates the entire msprof/torchrun/rank tree on timeout or interruption.
It requires a matching configuration/platform record from every rank.

The ranking metric is **p50 of iteration-wise maximum rank kernel duration**,
after removing the manifest's warmup and alignment launches. The parser rejects
missing/extra launches and excessive rank skew. Top-K configurations receive
independent correctness and longer profiling reruns; optional final stress
repeats existing correctness checks for the winner. All failures remain in
`results.csv`/`results.json`. `best.json` records whether retesting/stress ran and
whether enumeration was exhausted: a budget-limited winner is not a global
performance-optimality claim.

`--budget` limits new search candidates per invocation, interleaved across cases.
Top-K retests and final stress are additional explicitly selected phases.
`--resume` requires identical cases, candidates, declared platform, installed
code/DSO bytes, package/CANN versions and measurement settings. Use
`--retry-failed` to retry failures in fresh process groups; successful candidates
are not rerun. Budget can increase on resume. A new build requires a new output
directory. Replay validates the saved case/operator/export key and remeasures
the saved configuration; it does not copy old timing results.
Each attempt reserves a fresh directory, including after a controller crash
before its result was saved. Interrupted artifacts stay available for diagnosis
and are never mixed into a resumed profile. Only MTE operator kernel names are
counted; UDMA kernels and the MTE initialization probe are excluded.
