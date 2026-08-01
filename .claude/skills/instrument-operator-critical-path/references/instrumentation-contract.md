# Critical-path instrumentation contract

## Contents

1. Trace truth model
2. Point specification
3. Device implementation
4. Wire protocol and carrier
5. Runner and decoder
6. Normalized trace format
7. Verification gates
8. Evolution rules

## 1. Trace truth model

A timestamp is valid only when its clock instruction is causally ordered after the event named by the point. A visually plausible bar is not proof.

Classify every boundary explicitly:

| Boundary quality | Meaning | Allowed visualization |
|---|---|---|
| `causal` begin | The predecessor dependency has reached the timestamp pipeline before sampling | Exact stage begin |
| `actual` end | The responsible async engine has completed the claimed work before sampling | Exact stage end |
| `issue` end | Scalar control has issued/drained bookkeeping, but hardware completion is not proven | Issue marker or translucent span only |
| `inferred` | The interval lies between sparse samples and does not isolate the named work | Dashed/translucent span |

Do not infer pipeline ordering from source order. On an accelerator, a Scalar `GetSystemCycle` can execute while FIX, V, MTE2, or MTE3 work is pending. Bridge only the dependency needed for the claim; a blanket `PIPE_ALL` changes the schedule too much and hides the critical path.

For each stage, answer:

1. What exact instruction or condition makes the stage eligible to begin?
2. Which engine owns that predecessor?
3. On which engine/pipeline does the clock execute?
4. What event orders the predecessor into the clock pipeline?
5. What exact event proves the stage's last asynchronous work completed?
6. Is the point first/last per core, per wave, per expert, or per tile?
7. What cores have no work and must keep the slot missing?

If clocks are not comparable across roles/cores, do not place them on a shared cross-core axis. Either calibrate a shared epoch or render per-record local time and prohibit handoff-gap conclusions.

## 2. Point specification

Keep `trace-spec.json` in the target repository. It is the review contract shared by instrumentation, decoder, renderer, and humans.

Use timestamp ids ending in `_S`, normally `<stage>_begin_S` and `<stage>_end_S`. Use marker comments exactly as declared, for example:

```cpp
// TRACE_POINT: gmm1_begin_S
trace.gmm1Begin = CriticalPathTrace::Now();
```

The target anchor describes current semantic code near the point, not the inserted trace call and not a line number. Prefer a complete, stable single-line statement such as a dependency wait, first work issue, final drain, or publication call. The audit searches one source line at a time. Use regex anchors only when literals cannot express the current code; keep them narrow.

Planning audit resolves target anchors. Implemented audit additionally resolves marker comments, enforces before/after order and maximum distance, captures current line numbers/context, and records revision/dirty state. Line numbers are generated evidence, never reusable patch coordinates.

For repeated work, decide the capture policy before coding:

- Capture all repetitions only when bounded storage and device indexing are safe.
- Prefer fixed named slots on toolchains where dynamic local arrays may spill or trigger hot-path memory traffic.
- Emit actual count, capture capacity, and a truncation bit. Never silently omit later waves.
- Preserve zero/missing slots for inactive physical cores; do not synthesize work in the renderer.

## 3. Device implementation

### Compile-time containment

Use one compile-time switch around all of the following:

- trace fields and initialization;
- clock reads and accumulated wait accounting;
- diagnostic event bridges;
- helper pointer propagation;
- carrier capacity logic;
- terminal rendezvous and record writes.

Make the disabled helper pointer null or remove the argument at compile time. Merely turning record methods into no-ops is insufficient when helper control flow tests pointer presence and retains bridges.

Build both modes. Prefer a production default of off; a dedicated diagnostic branch may default on when stated in the spec and completion report.

### Hot-path behavior

Store samples in local named Scalar fields. Avoid a GM store at every point. Emit records after useful work so diagnostic memory traffic does not contend with the path under investigation.

Use first/last semantics deliberately:

- A begin usually records the first active unit after its dependency.
- An end usually overwrites until the last active unit reaches actual completion.
- A core with no assigned work leaves both fields missing.
- Accumulated polling waits use wrap-safe unsigned subtraction at each wait, not one coarse outer span.

Document every added bridge as measured perturbation. Reuse an event id only after proving it is fully set/wait recycled and does not overlap internal use.

### Begin boundaries

Place the sample after the readiness wait and any pipeline bridge, immediately before the first real work issue. Sampling before an asynchronous wait has reached Scalar paints dependency time as execution.

Examples:

- FIX producer to Scalar clock: use a narrow FIX-to-S bridge when no existing one proves completion.
- Vector wait issued on V with Scalar clock: bridge V-to-S before sampling.
- Scalar GM polling that returns only after visibility: no extra bridge may be needed, but document the visibility contract and proof.

### End boundaries

Distinguish issue from completion:

- A function return is insufficient if it only queued DMA or remote send work.
- A Scalar drain of bookkeeping is insufficient if completion waits live on another engine.
- Use MTE3-to-S for pending output stores when that is the owning engine.
- If an MTE3 completion dependency was transferred to MTE2, bridge MTE2-to-S before claiming completion.
- A publication store can be ordered after local FIX completion without proving another core already observes it. Keep publication-to-observation latency as a separate gap.

## 4. Wire protocol and carrier

Prefer a dedicated diagnostic output or workspace. Reuse a normal output only when all conditions hold:

1. capacity is checked on device before any record write;
2. useful output writes have completed;
3. every possible output writer joins a rendezvous with the header writer;
4. the run is declared diagnostic and excluded from accuracy comparison;
5. invalid capacity leaves a detectable missing/invalid trace rather than a partial record.

Use a fixed-width, endian-defined protocol. Include at least:

- magic, protocol version, schema id;
- header words/bytes and record words/bytes;
- record count and per-role counts/order;
- clock width/unit/scope;
- rank/device and runtime shape/config needed to identify the executed path;
- actual repetition count, capture capacity, and truncation flag;
- feature flags describing causal bridges/completion semantics;
- enough capacity metadata for the decoder to recompute required bytes.

Describe the concrete word layout in `protocol.header_fields` and `protocol.record_fields`; do not leave kernel/parser offsets only in duplicated source constants. Prefer `record_order: role-major` and store one role count word for each role in spec order. Mark inherited layouts `custom`, describe them precisely, and use a thin target decoder because the generic decoder will refuse them. A timestamp record field declares its point id, roles, first word, width, repetition count, and stride. The validator checks bounds, role coverage, and role-local overlap. Counter fields may use `kind: counter` with `count: 1` and are still included in the deterministic schema id.

The validator derives the expected uint32 schema id from the trace name, roles, clock contract, concrete wire layout, point cardinality/boundary qualities, and rendered spans. Set `protocol.schema_id` to the value reported by validation. This makes layout or semantic drift fail until kernel, spec, decoder, fixtures, and docs are updated together. Bump the human protocol version for incompatible decoder behavior as well.

The header write must not race late carrier writes. A rendezvous limited to AIV cores cannot protect a header written by AIC0. Prove that the selected primitive includes every core class involved.

## 5. Runner and decoder

Treat source preflight and runtime validation as separate defenses:

- Source preflight checks the current checkout's compile switch and expected macros.
- Runtime header validation proves which installed kernel actually executed.

Do not require a newly authored user script. Discover and adapt an existing test entry first; if it is external or ambiguous, request only its path or command. When `y` is the carrier, runner adaptation is mandatory: save its bytes before numerical interpretation, disable `y` accuracy comparison for the trace run, and keep the original correctness path for trace-off builds. Prefer a separate wrapper over editing the user's test file.

Never accept source preflight as proof of the installed binary. Reject bad magic, unsupported version/schema, wrong sizes/counts, unexpected role layout, wrong rank/shape/config, impossible flags, missing capacity, and unexpected truncation.

Avoid absolute repository/test paths. Support explicit arguments, environment overrides, and bounded upward discovery. When wrapping an evolving test script:

- prefer importing a stable callable API;
- if patching source in memory, require each exact anchor once and fail closed;
- do not edit the user's test script solely to run tracing;
- preserve the real script directory for multiprocessing/import resolution;
- keep target-case overrides explicit and echo them at runtime.

For a PyTorch-based runner, preserve the carrier bytes before any dtype conversion or numerical interpretation:

```python
raw = y_cpu.contiguous().view(torch.uint8).numpy().tobytes()
raw_path.write_bytes(raw)
```

Record the raw artifact hash and byte offset. Feed the unchanged bytes to the generic decoder when the spec uses role-major records.

Decode raw unsigned timestamps with modular arithmetic. Preserve raw values in JSON and use one declared reference. For a `bits`-wide clock:

```text
delta = (value - reference) mod 2**bits
if delta >= 2**(bits - 1): delta -= 2**bits
span = (end - begin) mod 2**bits
```

## 6. Normalized trace format

`trace_tool.py decode` reads a raw binary carrier for the validated role-major word layout and produces the normalized JSON below. Supply the chosen reference role/core/point, expected rank/role counts/config header fields, and required flag mask on the command line. It rejects truncated capture unless `--allow-truncated` is explicit. A custom legacy record order requires a thin repository decoder but must emit the same structure. `trace_tool.py render` accepts that JSON shaped as follows:

```json
{
  "schema_version": 1,
  "trace_name": "example-critical-path",
  "protocol_version": 1,
  "schema_id": "0x1234abcd",
  "metadata": {
    "rank": 0,
    "shape": "bs=512,h=4096",
    "raw_artifact": "rank0.bin"
  },
  "reference": {
    "role": "AIC",
    "core": 0,
    "point": "process_begin_S",
    "occurrence": 0
  },
  "records": [
    {
      "role": "AIC",
      "core": 0,
      "label": "AIC-00",
      "timestamps": {
        "process_begin_S": 4294967200,
        "gmm1_begin_S": [4294967250, 40],
        "gmm1_end_S": [20, 90]
      },
      "counters": {
        "gmm1_ready_cycles": 50
      }
    }
  ]
}
```

A timestamp value may be one integer/null or an array of integers/nulls for waves/repetitions. Paired span arrays use the same index. Use the configured missing sentinel only for missing work. Keep role/core pairs unique. The trace name, protocol version, and schema id must match the spec.

The renderer produces one shared-axis self-contained HTML timeline, tooltips with boundary claims, and a point-audit table. It marks `exact` spans solid and `inferred`/non-exact spans translucent and dashed. It fails on malformed input or a mismatched contract rather than drawing a misleading chart.

## 7. Verification gates

Verify in proportion to available hardware, and report unavailable gates explicitly.

### Static

- Validate the spec and both audits.
- Check point ids, marker uniqueness, anchor proximity, roles, exact-span boundary qualities, and placeholders.
- Review output capacity arithmetic for overflow and all supported shapes.
- Review record ownership for collisions and holes.
- Review event ids and every core-class rendezvous.
- Confirm parser constants derive from or match the protocol contract.

### Build and correctness

- Build trace enabled.
- Build trace disabled.
- Run compile/unit tests in both modes when practical.
- Run numerical correctness with tracing disabled.
- Inspect generated code/disassembly for complete removal of clock reads, diagnostic events, and stores when disabled when supported.

### Runtime trace

- Run the target case and at least one shape/core-count variation if supported.
- Exercise insufficient carrier capacity and bad/stale header rejection.
- Exercise a repeated-count overflow to prove the truncation signal.
- Check no-work cores remain missing.
- Check every exact end follows its begin modulo clock width.
- Check derived handoff gaps are physically plausible; investigate negative gaps rather than normalizing them away.
- Compare traced and untraced duration to quantify perturbation. Do not use trace-build operator total time as production performance.

## 8. Evolution rules

On any kernel, tiling, workspace, output, test-script, core-count, or scheduler change:

1. Rerun both audits before trusting old artifacts.
2. Reconstruct the semantic dependency chain for every failed anchor.
3. Update point reasoning and boundary semantics when control flow changed.
4. Recompute carrier capacity, record ownership, and capture count.
5. Bump protocol version/schema id for layout or semantic changes.
6. Rebuild both modes and rerun parser negative tests.
7. Regenerate normalized JSON, HTML, and audits; never compare an old parser against a new kernel silently.

Do not “fix” evolution failures by broadening regexes until they match. An audit failure is the intended signal to perform a fresh semantic review.
