---
name: instrument-operator-critical-path
description: Instrument evolving accelerator/operator code, especially MegaMoe Ascend C kernels, with low-perturbation `*_S` cycle timestamps; export a versioned self-checking trace; decode it; render an HTML timeline; and emit a mandatory audit explaining exactly where and why every point is placed. Use for 关键路径打点, xxx_S/GetSystemCycle tracing, AIC/AIV pipeline timing, output-buffer trace protocols, parser/HTML generation, or porting and revalidating a critical-path trace after kernel or test-script changes.
---

# Instrument Operator Critical Paths

Produce a trustworthy, reviewable critical-path trace rather than a collection of nearby timestamps. Treat timestamp placement, device-pipeline causality, wire layout, decoder, visualization, and human audit as one versioned contract.

## Load the right guidance

- Read [references/instrumentation-contract.md](references/instrumentation-contract.md) completely before changing device or runner code.
- Also read [references/mega-moe-case-study.md](references/mega-moe-case-study.md) when working on MegaMoe, Ascend AIC/AIV code, or a descendant of `codex/0717-critical-path-trace`.
- Run `scripts/trace_tool.py --help` for deterministic spec initialization, anchor auditing, and HTML rendering. Do not reimplement those pieces unless the target repository must be standalone.

## Resolve the runner input

Do not require the user to write a new script. Find an existing runnable test entry or command in the target repository first. If it lives outside the repository and cannot be discovered, ask only for its path or the exact run command. Create a thin repository-owned runner only when no reusable entry exists or the user requests a standalone workflow.

Treat runner adaptation as mandatory for an end-to-end trace that reuses `y`:

- keep the original test script unchanged when possible; prefer a separate wrapper, stable import, or exact failure-checked in-memory adaptation;
- capture `y` as unchanged raw bytes immediately after the operator and before dtype interpretation;
- skip normal `y` accuracy comparison for that diagnostic run, because trace records intentionally overwrite it;
- retain the ordinary accuracy path with tracing disabled;
- pass the raw bytes through runtime-header validation, decode, JSON emission, and HTML rendering.

Instrumentation/spec/audit planning can proceed without a runnable script, but do not claim end-to-end capture until a test entry actually executes the instrumented operator.

## Follow this workflow

### 1. Rediscover the current implementation

Work from the target checkout, not historical line numbers or patches.

1. Record `git status --short`, revision, target operator path, current build command, current test entry point, output/workspace ownership, core roles, and clock API.
2. Trace the current call graph and producer/consumer dependencies with symbol and statement searches. Identify actual work issue, dependency completion, async drains, final output writes, and all participating core classes.
3. Separate diagnostic changes from algorithm, tiling, scheduling, and performance changes already present on any reference branch. Never port a whole historical diff merely to obtain tracing.
4. Preserve unrelated dirty changes. Stop and report an ambiguous semantic anchor instead of guessing.

### 2. Define and audit the trace before editing

Create a repository-owned trace spec, normally beside the operator runner:

```bash
python3 /path/to/skill/scripts/trace_tool.py init \
  --name operator-critical-path \
  --output /path/to/operator/tools/critical_path_trace/trace-spec.json
```

Replace every `TODO` in the copied spec. Name every timestamp `<semantic>_<begin|end>_S`; use a separate `*_cycles` name for accumulated waits. For every timestamp, define:

- current file, enclosing symbol, semantic target anchor, before/after placement, and the eventual `TRACE_POINT: <id>` source marker;
- role and cardinality, such as first active task per physical core per wave;
- why the point is needed and the exact boundary it claims;
- upstream dependency, producer pipeline, timestamp pipeline, bridge or proof that no bridge is needed;
- expected diagnostic perturbation;
- whether an end is actual completion, issue-only, or inferred.

Declare the actual header and record word layout in the spec. Run `trace_tool.py validate`; if it reports a derived schema-id mismatch, put the reported id into the spec and the device header constant, then validate again. Never choose parser offsets independently from this layout.

When a trace already exists, inventory its device fields, output offsets, parser constants, and claimed boundaries first. Reconstruct the spec from current semantics, add the required `TRACE_POINT` marker comments beside existing samples, and preserve honest `issue`/`inferred` labels instead of duplicating or silently upgrading old points.

Run the planning gate:

```bash
python3 /path/to/skill/scripts/trace_tool.py audit \
  --phase plan --spec /path/to/trace-spec.json --repo-root /path/to/repo \
  --output-dir /path/to/trace-artifacts/audit-plan
```

Do not instrument until every target anchor resolves uniquely. Treat a missing or ambiguous anchor as evidence that the operator evolved: reread the current control/data flow and update the point semantics, not just the search string.

### 3. Instrument the device path

Implement the approved contract and keep `// TRACE_POINT: <id>` immediately beside each timestamp sample.

- Gate all diagnostic state, clock reads, bridges, and writes with one compile-time switch. Default it off unless the user explicitly wants a diagnostic-only branch.
- Keep hot-path samples in named scalar state. Use fixed named slots plus explicit truncation metadata for repeated waves when dynamic local arrays may spill.
- Sample a begin only after its real predecessor dependency reaches the timestamp pipeline. Add the narrow producer-to-Scalar bridge only when required.
- Label an end `actual` only after the relevant async engine has completed. Preserve issue boundaries as `issue`, never silently present them as completion.
- Write records only after useful work and pending writes to the trace carrier have completed. Use a rendezvous covering every writer and the header writer.
- Check carrier capacity before writing. Prefer dedicated diagnostic storage; if reusing an output tensor, declare that accuracy comparison is invalid for that run.
- Emit a self-checking header with magic, protocol version, schema id, header/record sizes, record/role counts, clock width, shape/config, capture capacity, truncation, rank, and trace flags.
- Compile disabled tracing out completely. Passing a non-null trace object into helpers can retain diagnostic bridges even when record methods are no-ops.

### 4. Build the decoder and runner against the current scripts

Keep source discovery and execution configurable. Accept explicit `--repo-root` and `--test-script` inputs, avoid baked absolute paths, and fail closed when an in-memory script rewrite anchor changes.

Make the runtime header authoritative: reject stale installed kernels, wrong layouts, unexpected shape/config, insufficient output capacity, and unsupported versions. Have the target runner save the trace-bearing output bytes unchanged. Use the generic decoder when the repository adopts the spec's role-major layout:

```bash
python3 /path/to/skill/scripts/trace_tool.py decode \
  --spec /path/to/trace-spec.json --raw /path/to/raw-trace.bin \
  --reference-role AIC --reference-core 0 \
  --reference-point process_begin_S \
  --expect-header rank=0 --expect-role-count AIC=36 \
  --require-flags-mask 0x1f --require-flags-value 0x1f \
  --output /path/to/trace-artifacts/normalized-trace.json
```

Add `--expect-header` checks for the target shape/config hash and any other path-defining header fields. The decoder rejects truncated capture by default; use `--allow-truncated` only for an explicitly reviewed partial trace. For a legacy/custom record order, implement a thin target decoder that performs the same header checks and emits the normalized JSON format documented in `instrumentation-contract.md`; do not falsely describe a custom layout as `role-major`. Decode unsigned clock arithmetic wrap-safely and preserve raw values.

Render the normalized data and include the implemented audit so the HTML itself exposes placement rationale:

```bash
python3 /path/to/skill/scripts/trace_tool.py audit \
  --phase implemented --spec /path/to/trace-spec.json --repo-root /path/to/repo \
  --output-dir /path/to/trace-artifacts/audit-implemented

python3 /path/to/skill/scripts/trace_tool.py render \
  --spec /path/to/trace-spec.json --trace /path/to/normalized-trace.json \
  --audit /path/to/trace-artifacts/audit-implemented/trace_point_audit.json \
  --output /path/to/trace-artifacts/critical_path.html
```

If a standalone repository runner is required, copy or adapt the renderer into the repository and record its provenance. Do not make a committed runner depend silently on a personal absolute skill path.

### 5. Verify both truth and containment

Require all applicable gates before claiming completion:

1. `trace_tool.py validate` passes with no placeholders.
2. Planning and implemented audits resolve all anchors; marker placement is unique, correctly ordered, and within the declared distance.
3. Trace-enabled and trace-disabled builds pass. Normal correctness tests run with tracing disabled.
4. Disabled code contains no live clock reads, trace writes, diagnostic barriers, or non-null helper trace pointers; inspect generated code when the toolchain supports it.
5. The runtime parser rejects bad magic, version/schema mismatch, record-size mismatch, stale config, truncation surprises, and undersized carriers.
6. Every exact span has a causal begin and actual-completion end. Zero/missing slots do not become fake work.
7. All compared rows share a proven clock domain and one reference. Otherwise render local timelines and explicitly prohibit cross-core gap conclusions.
8. Capture raw/normalized JSON, HTML, both audit reports, build/run commands, revision/dirty state, and observed diagnostic overhead. State clearly when hardware execution was unavailable.

## Mandatory completion report

Return all of the following; do not replace them with a generic “打点完成” summary:

- changed device, protocol, runner, and spec files;
- trace switch and enabled/disabled behavior;
- audit report paths and a compact table of point id, resolved file:line, placement, boundary claim, bridge, and reason;
- protocol version/schema id, carrier/capacity, write/rendezvous point, capture limit, and truncation behavior;
- exact versus inferred spans and known perturbations;
- commands actually run and results, including what could not be run;
- HTML and normalized/raw artifact paths;
- unresolved or manually reviewed anchors.

Do not claim the trace is trustworthy if the implemented audit fails, any exact boundary lacks dependency/completion proof, or the parser cannot prove it decoded the instrumented binary it expected.
