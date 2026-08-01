# MegaMoe reference case: `codex/0717-critical-path-trace`

## Contents

1. Reference scope
2. Reusable design
3. Boundary lessons
4. Emission and parsing lessons
5. Porting checklist

## 1. Reference scope

The reference implementation is branch `codex/0717-critical-path-trace`, head `6a8e1f223`, in the ops-transformer MegaMoe repository. Its trace work spans commits `19d61e4c3` through `6a8e1f223` on top of an already evolving optimization branch.

Use it as a case study, not a patch source. The branch also contains algorithm/tiling and pipeline-documentation changes, including GMM1 wave-tiling behavior. Do not transplant those functional changes when the task is only instrumentation.

Key reference files at that head:

- `mc2/mega_moe/op_kernel/arch35/mega_moe_impl_base.h`: trace state, constants, named wave slots, clock helpers.
- `mc2/mega_moe/op_kernel/arch35/mega_moe.h`: point placement, capacity check, final emission, rendezvous.
- `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h`: GMM/SwiGLU dependency-aware begin points and helper pointer contract.
- `mc2/mega_moe/tools/run_critical_path_trace.py`: source preflight, runtime validation, decode, JSON, role/triplet HTML.
- `mc2/mega_moe/docs/critical_path_trace.md`: v4 semantics and operating constraints.

## 2. Reusable design

The v4 layout records per-physical-core AIC/AIV0/AIV1 data. Fixed named wave slots avoid device-side dynamic indexing and possible spills. The runtime header reports both actual wave count and capture capacity, so later waves are explicitly marked truncated.

Hot-path instrumentation only updates named Scalar state. All records are copied to `y` after useful work. The header carries magic/version/layout, role counts, experts per wave, runtime shape/config, flags describing semantic guarantees, and capture capacity.

The runner checks two different truths:

- current source macro defaults, when source is available;
- the executed installed kernel's header, always after a run.

It emits raw structured JSON plus two HTML orderings on one cycle axis: role-grouped and logical AIC/AIV0/AIV1 triplets.

## 3. Boundary lessons

### Dispatch

- Begin: first expert enters Dispatch control for the wave.
- End: last expert returns after the existing per-expert MTE3-to-S drain proves token/triple GM writes complete.
- Lesson: reuse an existing completion dependency when it already reaches Scalar; do not add a blanket drain.

### GMM1

- Begin: dispatch-ready polling and any ping-pong FIX wait have passed; a diagnostic FIX-to-S bridge then orders the first MMAD boundary into Scalar.
- End: a narrow FIX-to-S bridge after the wave proves actual local FIX completion.
- Lesson: sampling Scalar immediately after a wait issued on another pipeline can run ahead and include dependency time in the stage.

### SwiGLU

- Begin: AIC-to-AIV0 V-pipeline wait has passed, followed by V-to-S before the clock.
- End: existing per-tile MTE3-to-S completion makes the outer final return actual.
- Lesson: source-code return and issue order alone are not clock causality.

### GMM2 publication

- Begin: Scalar SwiGLU-ready dependency has passed, before first MMAD.
- End: ready publication contains FIX-to-S and occurs after direct-GM completion.
- Publication end does not mean another core has observed the GM value. Preserve publication-to-ready-observation as a separate gap.

### Combine

- Begin: ready fan-in and triple MTE2-to-S dependencies have passed, before the first row send.
- The legacy Scalar end only described issue/drain bookkeeping.
- v4 actual end bridges the six-slot drain's MTE2-held completion dependency back to Scalar.
- Lesson: never label an issue boundary as completed remote send.

## 4. Emission and parsing lessons

The branch history is particularly useful because later commits expose common trace bugs:

| Commit | Failure corrected | General rule |
|---|---|---|
| `3b636dd71` | Runner depended on one source location | Accept explicit paths and discover bounded roots; runtime header remains authoritative |
| `46478859d` | Late Unpermute MTE3 writes could overwrite the trace header in `y` | Drain every carrier writer before repurposing output |
| `73df0d6b8` | Stage starts were not causal across pipelines | Bridge the predecessor pipeline to Scalar before the clock |
| `f7dcec3db` | AIV-only synchronization did not protect an AIC0 header writer | Rendezvous must include all writer/header core classes |
| `6a8e1f223` | Disabled trace methods were no-ops but non-null helper pointers retained diagnostic bridges | Compile out state/control flow and return null when disabled |

Specific MegaMoe details to revalidate rather than assume:

- `SyncAll<true>()` means AIV-only in this codebase; v4 needs the mixed-core form after Unpermute drain.
- Static diagnostic events used `EVENT_ID0` only after confirming full set/wait recycling; other ids could conflict with internal implementation.
- Reusing `y` invalidates accuracy comparison and requires a per-shape byte-capacity check.
- 32-bit `GetSystemCycle` values require wrap-safe subtraction.
- Operator total time includes trace rendezvous and record emission; internal `work_end` is the better diagnostic tail boundary.

## 5. Porting checklist

For a new MegaMoe branch or later code version:

1. Locate current `Process`, rolling schedule, GMM helpers, Dispatch, SwiGLU, ready publication/fan-in, Combine drain, Unpermute, and sync primitives by symbol.
2. Draw the current role-specific dependency chain. Do not assume the same wave schedule, role ownership, or event transfer.
3. Inventory every output/workspace write and confirm whether `y` is still a safe diagnostic carrier.
4. Recompute physical AIC/AIV counts and record ordering from runtime data.
5. Recompute wave count/capacity for current tiling. Do not copy the reference branch's four-wave assumption.
6. Recheck existing completion bridges and event-id lifetimes.
7. Keep every trace-only change under the compile switch; do not pull in the reference branch's tiling or performance changes.
8. Update the current runner through stable APIs where possible. If using in-memory replacement, make every anchor exact and failure-checked.
9. Include current runtime macros/shape fields in the header and reject an installed binary that does not match the requested path.
10. Generate the mandatory point audit and inspect its resolved source context before accepting the HTML.
