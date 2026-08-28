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
`y` with shape `[max(bs, ceil(8 MiB / (h * sizeof(y_dtype)))), h]` itself.

When disabled, all three switches must be restored to `0`/`False`. The formal kernel path then has no trace member,
trace workspace offset, timestamp read or trace-only synchronization.

## Capture raw

The trace build expands total `y` storage to at least 8 MiB. At kernel completion the trace overwrites the beginning
of `y`; the result is not suitable for accuracy checking. Save exactly the first 8 MiB for each EP rank, for example:

```python
from pathlib import Path

raw = y.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
Path(f"rank{rank}.bin").write_bytes(raw[: 8 * 1024 * 1024])
```

Use identical trace sources and switches for good and bad builds. Do not compare traced latency with an untraced
build as an absolute performance number.

## Generate reports

```bash
# A case directory: automatically discovers rank*.raw and rank*.bin numerically.
python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py raw/good

# One or more explicit rank captures.
python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py \
  raw/good/rank0.bin raw/good/rank1.bin \
  --output-dir good-trace-report

# This platform defaults to 1000 cycles/μs; override only for another platform.
python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py raw/good \
  --cycles-per-us 800
```

For one capture, the default output is `<capture-stem>-trace/`. For a case directory or multiple captures, the
default output is `trace-report/`. Every rank gets `trace.json`, `events.csv`, `timeline.svg` and an interactive
`index.html`; the multi-rank root gets `run-summary.json` and an index linking all rank reports. Rank clocks are
shown on separate axes and are never treated as globally synchronized.

The interactive report can enable or disable whole stage groups or individual events. Its timeline domain is
automatically fitted to the visible events so a dense full trace can be narrowed to one dependency chain. The
selection statistics follow the statistics Block, Wave, wait and event filters and report count, summed
total/active/wait time, mean, P50, P95, maximum and wall-clock coverage span in both cycles and μs. The default
conversion for this platform is `1000 cycles/μs`. Summed intervals can overlap when a
parent stage and its child wait are both selected, so the sum is not an operator wall-clock duration.

The display Block selector controls the timeline, lane overview, Wave summary and event details. A separate
statistics Block selector controls only the selection statistics, making it possible to display one Block while
aggregating another Block or all Blocks. Display Top-N limits do not truncate the statistics selection.

The report retains every event and event-detail row in the selected display scope. Its embedded payload uses a
versioned compact row protocol instead of repeated JSON event objects. The timeline uses one Canvas for the whole
view and draws only lanes intersecting the vertical viewport; the event-detail and Wave tables materialize only their
visible scroll windows. Exact selection statistics and Wave aggregation run in a zero-dependency Web Worker, with a
main-thread fallback when Worker creation is unavailable. Hovered events are raised with an outline and shadow;
clicking keeps a persistent selected marker while showing the event details. These are lossless interaction
optimizations, not event limits: scrolling, hit-testing, filtering and statistics still use the full selected data.
Statistics-Block changes refresh only the statistics panel; display-only Block, sorting, Top-N and pin changes reuse
the filtered model; and expensive summary/detail updates are split across animation frames.

## Optional Parquet / DuckDB analysis

The interactive HTML remains the only timeline implementation. For batch SQL, commit bisection and good/bad
aggregation, the same canonical `events.csv` can additionally be converted to Zstd-compressed Parquet with DuckDB:

```bash
python3 -m pip install duckdb
python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py raw/good \
  --parquet --output-dir good-trace-report

python3 - <<'PY'
import duckdb

duckdb.sql("""
  SELECT role, core, event, count(*) AS count,
         sum(duration_cycles) AS total_cycles,
         quantile_cont(duration_cycles, 0.95) AS p95_cycles
  FROM read_parquet('good-trace-report/rank0/events.parquet')
  GROUP BY role, core, event
  ORDER BY total_cycles DESC
""").show()
PY
```

`--parquet` is optional and does not affect HTML generation. Without the `duckdb` package the normal JSON, CSV,
SVG and HTML artifacts continue to work; requesting `--parquet` reports the missing dependency explicitly.

High-level presets provide one-click views for `PRE_MOE`, the MoE main pipeline (event IDs 2 through 13),
`COUNT_EXPORT`, `UNPERMUTE`, `POST_MOE`, shared experts and cross-rank synchronization. Selecting individual events
switches the preset indicator to `custom`.

`compare` is an optional aggregate difference report. It groups each side by rank/role/event and writes good,
bad and `bad - good` count/total/active/wait/mean/max values. It does not replace the per-rank timeline:

```bash
python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py compare \
  --good raw/good --bad raw/bad --output-dir good-bad-compare

python3 mc2/mega_moe/tools/critical_path_trace/trace_tool.py synthetic-test
```

The parser validates magic/version, fixed record boundaries, rank/role/core identity, written counts and timestamp
ordering. It only reads `written_count` events. A nonzero `dropped_count` is reported as `truncated`; an event without
an end timestamp is retained as an unpaired zero-duration event and listed in `warnings`. Parent stages contain
explicit child `wait_cycles`, with `active_cycles = total - child waits`. Hover details include wave, expert, tile,
token range, completion kind and source locations.

## Completion and perturbation

- `GetSystemCycle()` is only a Scalar timestamp.
- Dispatch completes at its existing `MTE3_S` followed by the row-ready AtomicAdd.
- GMM1 AIC tiles complete at their existing `FIX_S`; activation tiles complete at the epilogue's existing
  `MTE3_S` before the GMM2-ready AtomicAdd.
- AIV0 GMM1/GMM2 prologue tiles lack a Scalar completion boundary. Trace mode adds one `MTE3_S` per recorded tile.
- A8W4 GMM2 publishes each Combine group counter only after its existing `FIX_S`; the trace records after that
  publication and does not add another GMM2 completion wait.
- Combine work events end at submission and carry `END_SUBMIT`. Current mainline has no per-schedule Scalar-visible
  MTE3 completion, so event 13 is not emitted and no trace-only drain is added.
- Count export normally ends with an `MTE3_MTE2` dependency, which is not Scalar-visible completion. Trace mode adds
  one tail `MTE3_S` on the sole exporting AIV1 block.
- Unpermute adds one tail `MTE3_S`.
- The two full-core synchronizations in trace publication occur after normal computation and are not used as stage
  completion boundaries.

The per-tile prologue `MTE3_S` points are the largest main-pipeline perturbation. They make prologue completion
times precise but reduce queueing and overlap. Combine is intentionally left submit-only to preserve the mainline
overlap. Good/bad conclusions must therefore be based on relative changes from semantically matching events, and
should be cross-checked with hardware counters when attributing engine-only MTE2/MTE1/MAC/FIXPIPE time.
