# MegaMoe A8W4 critical-path trace

This diagnostic trace is source-switched and intentionally does not use build-system options or `-D` flags.
It is only valid for the FP8-activation/FP4-weight `MegaMoeA8W4Wave` path.

## Enable

Before rebuilding, manually change all three switches:

Search for `MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE` and change all three definitions:

1. In `op_kernel/arch35/common/mega_moe_a8w4_trace_config.h`, set it to `1`. This switch is shared by the kernel,
   host infer-shape and tiling code.
2. In `torch_extension/csrc/mega_moe.cpp`, set it to `1`.
3. In `torch_extension/mega_moe.py`, set it to `True`.

The second and third switches are only needed for the torch extension. A direct ACLNN caller must instead allocate
`y` with shape `[max(bs, ceil(2 MiB / (h * sizeof(y_dtype)))), h]` itself.

When disabled, all three switches must be restored to `0`/`False`. The formal kernel path then has no trace member,
trace workspace offset, timestamp read or trace-only synchronization.

## Capture raw

The trace build expands total `y` storage to at least 2 MiB. At kernel completion the trace overwrites the beginning
of `y`; the result is not suitable for accuracy checking. Save exactly the first 2 MiB for each EP rank, for example:

```python
from pathlib import Path

raw = y.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
Path(f"rank{rank}.bin").write_bytes(raw[: 2 * 1024 * 1024])
```

Use identical trace sources and switches for good and bad builds. Do not compare traced latency with an untraced
build as an absolute performance number.

## Decode and compare

```bash
python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py decode \
  raw/good/rank0.bin raw/good/rank1.bin \
  --events-csv good_events.csv \
  --summary-csv good_summary.csv \
  --timeline good_timeline.html \
  --json good_summary.json

python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py compare \
  --good raw/good/rank0.bin raw/good/rank1.bin \
  --bad raw/bad/rank0.bin raw/bad/rank1.bin \
  --csv good_bad.csv --json good_bad.json

python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py synthetic-test
```

The decoder validates magic/version, fixed record boundaries, rank/role/core identity, written counts and timestamp
ordering. It only reads `written_count` events. A nonzero `dropped_count` is reported as `truncated`. Stage summaries
contain `total_cycles`, explicit child `wait_cycles`, and `active_cycles = total - union(child waits)`.

## Completion and perturbation

- `GetSystemCycle()` is only a Scalar timestamp.
- Dispatch completes at its existing `MTE3_S` followed by the row-ready AtomicAdd.
- GMM1 AIC tiles complete at their existing `FIX_S`; activation tiles complete at the epilogue's existing
  `MTE3_S` before the GMM2-ready AtomicAdd.
- AIV0 GMM1/GMM2 prologue tiles lack a Scalar completion boundary. Trace mode adds one `MTE3_S` per recorded tile.
- A8W4 GMM2 tiles lack a per-tile Scalar FIXPIPE boundary. Trace mode adds one `FIX_S` per recorded tile.
- Combine work events end at submission and carry `END_SUBMIT`; the per-wave drain event adds one `MTE3_S` and is
  the true send-completion boundary.
- Unpermute adds one tail `MTE3_S`.
- The two full-core synchronizations in trace publication occur after normal computation and are not used as stage
  completion boundaries.

The per-tile `MTE3_S`/`FIX_S` points are the largest perturbation. They make completion times precise but reduce
queueing and overlap. Good/bad conclusions must therefore be based on relative changes from the same instrumented
build, and should be cross-checked with hardware counters when attributing engine-only MTE2/MTE1/MAC/FIXPIPE time.
