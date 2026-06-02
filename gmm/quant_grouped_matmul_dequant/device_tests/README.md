# device_tests/ — on-device validation for qgmmd

On-device correctness + profiling sweep for
`torch_npu.npu_quant_grouped_matmul_dequant`. Each test runs inside a
`msprof op` wrap; the OPPROF_* dir is parsed into one STATS.txt (compact
aggregate) and one `<tag>.log` (full visualize_data.bin + CSV dump) per
config.

## Prerequisites

- CANN + torch_npu already sourced in the shell.
- `msprof` on `PATH` (part of the CANN toolkit).
- qgmmd operator built + installed:
  ```bash
  bash talk/qgmmd/test/build_and_install.sh
  ```

## Config matrix

Shapes pull straight from `talk/qgmmd/test/generate_data.py` — both LIGHT
(simulator-friendly) and HEAVY (device-only) tiers.

| LIGHT shape  | E | M | K | N | purpose |
|---|---|---|---|---|---|
| `tiny`        | 2  | 32  | 128  | 128  | sanity |
| `tiny_moe`    | 4  | 16  | 256  | 256  | sparse-MoE sanity |
| `decode_like` | 1  | 4   | 2048 | 1536 | single-token decode |
| `decode_moe`  | 2  | 8   | 2048 | 1536 | MoE decode |
| `swift_low/small/mid/high` | 1 | 24/32/128/192 | 2048 | 1536 | SWIFT range |
| `k_heavy`     | 1  | 16  | 8192  | 128  | K-heavy; SmallM path |
| `k_long`      | 1  | 16  | 15360 | 128  | phaseXChunk streaming |
| `n_heavy`     | 1  | 32  | 512   | 4096 | large-N, many cbs |
| `tail_m/k/n`  | 1  | various, misaligned dims — tail-cb handling |
| `ffn_tall_sm` | 1  | 32  | 5120  | 1536 | FFN down-proj (LIGHT scale) |
| `ffn_fat_sm`  | 1  | 32  | 1536  | 5120 | FFN up-proj (LIGHT scale) |

| HEAVY shape   | E | M | K | N | purpose |
|---|---|---|---|---|---|
| `decode_moe_full` | 8 | 8 | 2048 | 1536 | dense MoE decode |
| `ffn_tall_md` / `ffn_fat_md` | 1 | 64 | 7168/2048 | 2048/7168 | device-scale FFN |
| `small` / `moe_sparse` / `half` / `real` | MoE at production size |
| `prefill_med` / `prefill_big` | dense long-sequence prefill |
| `ffn_tall` / `ffn_fat` | plan §App. G reference shapes |

| QWEN3 shape (priority — commercial release) | E | M | K | N | purpose |
|---|---|---|---|---|---|
| `qw30b_tp{2,4}_g{1,2}_dec_bs{1,8}` | 128 | 8 / 64 | 192–2048 | 384–2048 | Qwen3-30B-A3B decode (BS=1 vs BS=8) |
| `qw30b_tp{2,4}_g{1,2}_pre`         | 128 | 16384  | 192–2048 | 384–2048 | Qwen3-30B-A3B prefill T=2048 |
| `qw35b_tp{2,4}_g{1,2}_dec_bs{1,8}` | 256 | 8 / 64 | 128–2048 | 256–2048 | Qwen3.5-35B-A3B decode |
| `qw35b_tp{2,4}_g{1,2}_pre`         | 256 | 16384  | 128–2048 | 256–2048 | Qwen3.5-35B-A3B prefill T=2048 |

`QWEN3` tier = 24 shapes total (16 decode + 8 prefill). Naming: `qw{30b,35b}_tp{2,4}_g{1,2}_{dec_bs1,dec_bs8,pre}` where `g1` = GMM1 (gate&up, K=2048, N=2I), `g2` = GMM2 (down, K=I, N=2048). Exact (E, M, K, N) per shape: see `talk/qgmmd/test/generate_data.py:QWEN3`.

**Variants (3 per shape):**

| variant | effect |
|---|---|
| `dynamic`         | default path — dynamic per-token quant, fp32 weight_scale |
| `dynamic_u64`     | uint64-encoded weight_scale (VDEQ16 direct path) |
| `dynamic_smooth`  | `dynamic` + smooth_scale input (fp16 per-K) |

Count / list configs:

```bash
python3 gmm/quant_grouped_matmul_dequant/device_tests/configs.py --list
python3 gmm/quant_grouped_matmul_dequant/device_tests/configs.py --list --shape prefill_med
python3 gmm/quant_grouped_matmul_dequant/device_tests/configs.py --list --tier HEAVY
```

## Running

Each run always lands inside `<out>/<shape_names>_YYYY-MM-DD_HH-MM/` so
re-runs against the same `--out` never overwrite history.

### All LIGHT + HEAVY

```bash
bash gmm/quant_grouped_matmul_dequant/device_tests/run_all.sh device_tests/out
# or restrict to one tier:
TIER=LIGHT bash gmm/quant_grouped_matmul_dequant/device_tests/run_all.sh device_tests/out
TIER=HEAVY bash gmm/quant_grouped_matmul_dequant/device_tests/run_all.sh device_tests/out
TIER=QWEN3 bash gmm/quant_grouped_matmul_dequant/device_tests/run_all.sh device_tests/out
# or a shape subset:
SHAPES="decode_like k_heavy" bash gmm/quant_grouped_matmul_dequant/device_tests/run_all.sh device_tests/out
```

### One shape / one variant

```bash
python3 gmm/quant_grouped_matmul_dequant/device_tests/run_device.py --shape decode_like
python3 gmm/quant_grouped_matmul_dequant/device_tests/run_device.py --shape k_heavy --variant dynamic_u64
python3 gmm/quant_grouped_matmul_dequant/device_tests/run_device.py --shape k_heavy --shape k_long  # multiple shapes
```

### Re-extract STATS from an existing run

```bash
python3 gmm/quant_grouped_matmul_dequant/device_tests/extract_msprof_stats.py \
    --root device_tests/out/decode_like_<stamp>
```

## What happens per test

1. Inputs + golden are staged under `<dataroot>/<shape>/` — by default
   `talk/qgmmd/test/data/<shape>/` so simulator + device share one data
   cache. Variants share data; the variant only selects which file the
   runner loads (uint64 vs fp32 scale, smooth on/off).
2. Subprocess runs as
   `msprof op --kernel-name=QuantGroupedMatmulDequant python3 run_one.py …`
   with `cwd = <tag>/`, so msprof writes its OPPROF_* there.
3. `run_one.py` does a 1-warmup + 1-measure call to
   `torch_npu.npu_quant_grouped_matmul_dequant` and writes
   `result.json` with cos / pass / max/mean abs error / wall_ms.
4. SUMMARY.txt gets one line per test.

After all tests finish, `extract_msprof_stats.py` parses every OPPROF_*
`visualize_data.bin` + CSVs and emits a compact aggregate STATS.txt plus
one full-detail `<tag>.log` per test.

## Output layout

```
<out>/
  SUMMARY.txt                # one line per test
  STATS.txt                  # primary pipe table + bandwidth table
  <tag>.log                  # full visualize_data + CSV dump per test
  <tag>/
    OPPROF_<stamp>_<id>/
      visualize_data.bin
      OpBasicInfo.csv
      PipeUtilization.csv
      Memory.csv / MemoryL0.csv / MemoryUB.csv
      L2Cache.csv / ResourceConflictRatio.csv
      ArithmeticUtilization.csv
    result.json              # cos / pass / wall_ms
    msprof_stdout.log        # stdout from msprof + python
```

## STATS.txt layout

**Primary table** — one row per test:

- `tag`, `E`, `M`, `K`, `N`, `cos%`, `PASS`
- `Task µs` (kernel duration from msprof)
- `BlkDim`
- Per-pipe `ratio% | wait%` for Cube, Vector, MTE1, MTE2, MTE3
- `Scalar%`
- `L2 hit%`, `iCache%`

**Secondary table** — bandwidth + instruction counts:

- `MTE2 instr`, `MTE3 instr`, `MTE1 instr` (from visualize_data.bin Pipe table)
- `GM read GB/s`, `GM write GB/s` (from `core_memory_map.memory_unit`)
- `est W·MTE2 B` — ideal single-pass weight bytes (`E·K·N` / block_dim)

Both tables also include a geomean `Task Duration` per shape family.

## Decisions this enables

| Question | Look at |
|---|---|
| Does S15 runtime wiring pay off? | `MTE2 instr` + `MTE2%` and `GM read GB/s` pre/post |
| Is cb-outer swizzle (S16) worth landing? | `GM read GB/s` vs DDR peak; `L2 hit%` — high L2 hit = weight-reuse benefits |
| Phase X DMA burst p95 ≥ 16 KB target? | per-test `<tag>.log` `memory_unit` bandwidth + request pairs |
| MTE2 util ≥ 97 % on HEAVY? | `MTE2%` on prefill_med / ffn_tall / ffn_fat |
| MTE3 overhead ≤ +10 %? | `MTE3%` and `MTE3 wait%` |
| Still 0 `MySyncAll` calls? | grep compiled kernel binary for `MySyncAll` |
| Cos ≥ 0.9999 (§15 S13 lower bar)? | `cos%` + `PASS` column |

## Notes

- Pass threshold: `cos > 0.9999` (9999 in percent). Matches qbmv3's
  threshold — the fp16 precision plateau we documented on simulator
  applies on device too.
- One OPPROF_* per test (msprof writes a fresh one per subprocess).
- Large intermediate CSVs + visualize_data.bin stay on disk for
  re-extraction; only STATS.txt is the "stable audit trail".
