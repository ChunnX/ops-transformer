# How to use the benchmarking/correctness program benchmark.py

# Installation

To install the sparse kernel go in the `ops-transformer` project home folder and run:

```shell
bash build.sh --make_clean --experimental -j96 --pkg --soc=ascend910b --ops=blitz_sparse_attention
./build/cann-ops-transformer-custom_linux-"$(uname -i)".run
(cd experimental/attention/blitz_sparse_attention/torch_interface && bash build.sh custom)
```
Fast recompilation only of bf16 kernel variant
```shell
bash build.sh --make_clean --experimental -j96 --pkg --soc=ascend910b --ops=blitz_sparse_attention --op-variant=2
./build/cann-ops-transformer-custom_linux-"$(uname -i)".run
```
Oneliner smoketest and benchmark
```shell
(cd experimental/attention/blitz_sparse_attention/benchmark && python test_attn.py && python benchmark.py)
```

Benchmark + plot pipeline (writes `benchmark.png` summarising all
`(BLOCK_SIZE_Q, BLOCK_SIZE_KV, sparsity)` combinations):
```shell
(cd experimental/attention/blitz_sparse_attention/benchmark && python benchmark.py)
```

## Parameters

Using the CAPITAL_LETTERS variables at the top of the file you can ru all the things you need. Wherever there is a list, it means you can set multiple values, and the file will run correctness/time evaluations for **all the possible combinations**.

* `B_VALS`: batch sizes to test. For block-sparsity only `B_VALS = [1]` works for now.
* `H_VALS`: number of heads
* `S_VALS`: sequence lengths
* `D_VALS`: head dimensions
* `N_REPEATS`: how many runs to do to estimate time
* `N_WARMUP`: how many warmup runs before running `N_REPEATS` calls
* `SPARSITY_VALS`: how many blocks to activate: randomly chosen, but each Q-block row will have the same number of blocks. The block-sparse pattern is the only kernel mode the benchmark exercises.
* `BLOCK_SHAPES`: List of `(BLOCK_SIZE_Q, BLOCK_SIZE_KV)` pairs to sweep. The kernel accepts every combination of `BLOCK_SIZE_Q ∈ {128, 256, 512, 1024}` and `BLOCK_SIZE_KV ∈ {128, 256, 512, 1024}` at runtime via the `block_shape` op-attr — no rebuild needed to switch granularities. The default sweep is the full 4 × 4 = 16-combination grid. The first column of the benchmark output labels every row with its active block shape. `BLOCK_SIZE_Q` and `BLOCK_SIZE_KV` aliases (set to `BLOCK_SHAPES[0]` for backward compatibility) are still readable from the top of `benchmark.py`. Set `BSA_SHARD="i/N"` in the environment to keep only the i-th of N contiguous slices of `BLOCK_SHAPES` — used by the parallel-NPU sharding recipe at the bottom of this file.
* `BLOCK_MASK_SEED`: for reproducibility of random sampling. Can set to any value to change random sampling
* `FRAMES_BY_BLOCK_SHAPE`: 2-D dict keyed by `(BLOCK_SIZE_Q, BLOCK_SIZE_KV)` returning a `FrameWidths(left_cols, right_cols, top_rows, bottom_rows)` namedtuple — extra blocks forced active on top of the sparsity budget. Models a typical sink/streaming pattern. Per-pair entries scale to keep the forced-token footprint comparable across granularities; see Example 2 for the derivation.
* `PRINT_OUTPUTS`: For manual visualization of output tensors
* `PRINT_MASK`: To visually check if the mask is as expected
* `PRINT_BLOCK_EQUALITY`: if the correctness test fails, you can check which blocks in the outputs match with the reference and which not. The block granularity to visualize is given by `PRINT_HEIGHT` and `PRINT_WIDTH`
* `RUN_REFERENCE`: Besides measuring time, also test correctness. This only is possible for small enough masks, otherwise you will get OOM in the reference run. For this reason, you can disable this for very long contexts, and enable it for short enough sequences.
* `TORCH_REFERENCE`: If `True`, correctness will be tested against the torch attention implementation. If `False`, for block-sparse attention the reference is the torch_npu dense with the corresponding attention mask (like in `sparse_block` mode). Please only use `H=1`, as torch_npu doesn't support different masks for different heads.

## Examples

### 1 - Test correctness and speed with multiple values (and shorter sequence lengths)

#### Input

```python
B_VALS = [1]
H_VALS = [16, 14]
S_VALS = [4096, 10_000]  # S_q = S_kv
D_VALS = [128]   # head dimension

N_REPEATS = 20
N_WARMUP = 2

SPARSITY_VALS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 512
BLOCK_MASK_SEED = 1234
USE_FRAME = False   # For short inputs, better to disable the frame or the mask will be dense

# Print tensors for manual comparisons
PRINT_OUTPUTS = False
PRINT_MASK = False
# For printing tensor differneces in blocks
PRINT_BLOCK_EQUALITY = False
PRINT_HEIGHT = 128
PRINT_WIDTH = 8

RUN_REFERENCE = True
TORCH_REFERENCE = True  # If False, will instead run the torch_npu reference
```

#### Output:

```
==========================================================================================
  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'
==========================================================================================
  H   B    S_q   S_kv    D  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]
------------------------------------------------------------------------------------------
 16   1   4096   4096  128      0.10             yes           17638.60            1328.46
 16   1   4096   4096  128      0.20             yes           17645.56            1166.37
 16   1   4096   4096  128      0.30             yes           17642.52            1162.32
 16   1   4096   4096  128      0.40             yes           17622.96            1060.28
 16   1   4096   4096  128      0.50             yes           17628.49            1025.70
 16   1   4096   4096  128      0.60             yes           17626.99             980.80
 16   1   4096   4096  128      0.70             yes           17646.98             936.72
 16   1   4096   4096  128      0.80             yes           17669.57             955.56
 16   1   4096   4096  128      0.90             yes           17663.78             910.45
 16   1  10000  10000  128      0.10             yes          110451.61            7973.21
 16   1  10000  10000  128      0.20             yes          110456.47            7199.73
 16   1  10000  10000  128      0.30             yes          110440.53            6303.86
 16   1  10000  10000  128      0.40             yes          110435.28            5396.22
 16   1  10000  10000  128      0.50             yes          110441.05            4525.04
 16   1  10000  10000  128      0.60             yes          110446.19            3694.06
 16   1  10000  10000  128      0.70             yes          110438.79            2830.48
 16   1  10000  10000  128      0.80             yes          110458.57            1968.18
 16   1  10000  10000  128      0.90             yes          110501.12            1105.10
 24   1   4096   4096  128      0.10             yes           26154.51            1991.36
 24   1   4096   4096  128      0.20             yes           26161.87            1757.80
 24   1   4096   4096  128      0.30             yes           26115.79            1740.01
 24   1   4096   4096  128      0.40             yes           26215.35            1492.14
 24   1   4096   4096  128      0.50             yes           26202.39            1243.03
 24   1   4096   4096  128      0.60             yes           26141.22            1091.14
 24   1   4096   4096  128      0.70             yes           26161.22            1028.43
 24   1   4096   4096  128      0.80             yes           26180.69            1025.53
 24   1   4096   4096  128      0.90             yes           26216.21             971.07
```
Reference goes OOM for 10_000 sequence length and 24 heads (not shown).

On the left you see the inputs, then the "yes" line checks correctness, after that you see the reference runtime and our runtime. After that you see the memory bandwidth usage, which is much better in our kernel.

Note that for short sequence lengths, the speedups are negligible because overheads are higher than actual computation.


### 2 - block_shape granularity sweep (S=118806, H=3, D=128, BF16)

The runtime `block_shape` attr lets the same kernel run at every combination of
`(BLOCK_SIZE_Q, BLOCK_SIZE_KV)` with each dimension in `{128, 256, 512, 1024}`
— **16 granularities** sharing one kernel binary. Finer sabi (smaller block
sizes) buys more sparsity-pattern resolution per Q-row at the cost of more
sabi-tensor HBM traffic + more sub-block plumbing per cube tile; coarser sabi
buys cheaper per-row metadata at the cost of resolution. The sweep below
measures the resulting latency/throughput trade-off at S = 118 806.

#### Why `(left_cols, right_cols, top_rows, bottom_rows)` scales as `(29,15,8,4)` / `(29,15,8,4)`

The benchmark forces a frame in the attention matrix that models a typical
generative-model sink/streaming pattern: **the first ~3600 attention rows and
columns plus the last ~6 rows and columns are always selected**. 6 tokens is
less than a single block in every dimension, so it always rounds up to one
full block-row / block-column at the bottom/right.

`FrameWidths.left_cols` / `right_cols` are in `BLOCK_SIZE_KV`-token units, and
`top_rows` / `bottom_rows` are in `BLOCK_SIZE_Q`-token units, so the per-pair
values fall out of rounding the constant 3600-token target up to whole blocks
in each dimension independently:

| size  | `ceil(3600 / size)` |
|------:|:-------------------:|
|  128  | 29                  |
|  256  | 15                  |
|  512  |  8                  |
| 1024  |  4                  |

That gives a 2-D `FRAMES_BY_BLOCK_SHAPE[(BLOCK_SIZE_Q, BLOCK_SIZE_KV)]` table
where `left_cols ∈ {29, 15, 8, 4}` (indexed by `BLOCK_SIZE_KV`), `top_rows ∈
{29, 15, 8, 4}` (indexed by `BLOCK_SIZE_Q`), and `right_cols = bottom_rows = 1`
everywhere. e.g. `(BLOCK_SIZE_Q=1024, BLOCK_SIZE_KV=1024)` ⇒
`FrameWidths(left_cols=4, right_cols=1, top_rows=4, bottom_rows=1)`.

The test files apply the same scaling rule with a smaller target (~256 tokens)
for their `SPARSE_FRAME`.

#### Input
```python
B_VALS = [1]; H_VALS = [3]; S_VALS = [118_806]; D_VALS = [128]
N_REPEATS = 10; N_WARMUP = 2
SPARSITY_VALS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BLOCK_SHAPES = [(bsq, bskv)
                for bsq  in (128, 256, 512, 1024)
                for bskv in (128, 256, 512, 1024)]                 # 4 × 4 = 16 pairs
# Per-pair frame: left_cols/top_rows are scaled by 128 / size in each dim so
# the forced-token footprint stays comparable across block shapes.
FRAMES_BY_BLOCK_SHAPE = {
    (bsq, bskv): FrameWidths(
        left_cols   = {128: 29, 256: 15, 512: 8, 1024: 4}[bskv],
        right_cols  = 1,
        top_rows    = {128: 29, 256: 15, 512: 8, 1024: 4}[bsq],
        bottom_rows = 1,
    )
    for bsq  in (128, 256, 512, 1024)
    for bskv in (128, 256, 512, 1024)
}
TORCH_REFERENCE = "npu_fusion_attention"
```

#### Output

Run `python benchmark.py | tee >(python plot.py)` to also generate
a `speedup-vs-sparsity` figure with one curve per `(sequence_length, block_shape)`
pair. The full raw table below was captured at S=118 806, H=3, D=128, BF16:

```
================================================================================================
  DTYPE=torch.bfloat16  INPUT_LAYOUT='BNSD'
  BLOCK_SHAPES=[(128,128),(128,256),(128,512),(128,1024),
                (256,128),(256,256),(256,512),(256,1024),
                (512,128),(512,256),(512,512),(512,1024),
                (1024,128),(1024,256),(1024,512),(1024,1024)]
  TORCH_REFERENCE='npu_fusion_attention'
================================================================================================
  block_shape   H   B    s_q   s_kv    D  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]
------------------------------------------------------------------------------------------------
      128x128   3   1 118806 118806  128      0.00             yes          160834.68          222439.65
      128x128   3   1 118806 118806  128      0.10             N/A                N/A          167611.15
      128x128   3   1 118806 118806  128      0.20             N/A                N/A          164763.44
      128x128   3   1 118806 118806  128      0.30             N/A                N/A          157588.59
      128x128   3   1 118806 118806  128      0.40             N/A                N/A          147927.00
      128x128   3   1 118806 118806  128      0.50             N/A                N/A          135397.29
      128x128   3   1 118806 118806  128      0.60             N/A                N/A          119691.65
      128x128   3   1 118806 118806  128      0.70             N/A                N/A           96235.68
      128x128   3   1 118806 118806  128      0.80             N/A                N/A           66596.97
      128x128   3   1 118806 118806  128      0.90             N/A                N/A           28013.06
      128x256   3   1 118806 118806  128      0.00             yes          162830.13          197855.22
      128x256   3   1 118806 118806  128      0.50             N/A                N/A           98464.03
      128x256   3   1 118806 118806  128      0.90             N/A                N/A           20407.69
      128x512   3   1 118806 118806  128      0.00             yes          164261.63          222498.10
      128x512   3   1 118806 118806  128      0.50             N/A                N/A           87831.53
      128x512   3   1 118806 118806  128      0.90             N/A                N/A           19868.71
     128x1024   3   1 118806 118806  128      0.00             yes          165972.40          194220.53
     128x1024   3   1 118806 118806  128      0.50             N/A                N/A           86817.52
     128x1024   3   1 118806 118806  128      0.90             N/A                N/A           17723.82
      256x128   3   1 118806 118806  128      0.00             yes          166371.09          221960.40
      256x128   3   1 118806 118806  128      0.50             N/A                N/A          135575.98
      256x128   3   1 118806 118806  128      0.90             N/A                N/A           27825.63
      256x256   3   1 118806 118806  128      0.00             yes          166028.88          197476.27
      256x256   3   1 118806 118806  128      0.50             N/A                N/A           98570.60
      256x256   3   1 118806 118806  128      0.90             N/A                N/A           20037.58
      256x512   3   1 118806 118806  128      0.00             yes          166818.91          224199.10
      256x512   3   1 118806 118806  128      0.50             N/A                N/A           89593.69
      256x512   3   1 118806 118806  128      0.90             N/A                N/A           18887.64
     256x1024   3   1 118806 118806  128      0.00             yes          167224.23          194548.27
     256x1024   3   1 118806 118806  128      0.50             N/A                N/A           87931.34
     256x1024   3   1 118806 118806  128      0.90             N/A                N/A           16409.03
      512x128   3   1 118806 118806  128      0.00             yes          167523.89          223262.23
      512x128   3   1 118806 118806  128      0.50             N/A                N/A          136341.86
      512x128   3   1 118806 118806  128      0.90             N/A                N/A           28692.35
      512x256   3   1 118806 118806  128      0.00             yes          167157.95          197503.83
      512x256   3   1 118806 118806  128      0.50             N/A                N/A           99276.25
      512x256   3   1 118806 118806  128      0.90             N/A                N/A           21158.86
      512x512   3   1 118806 118806  128      0.00             yes          167537.83          224312.11
      512x512   3   1 118806 118806  128      0.50             N/A                N/A           89056.08
      512x512   3   1 118806 118806  128      0.90             N/A                N/A           20053.99
     512x1024   3   1 118806 118806  128      0.00             yes          167761.99          194517.43
     512x1024   3   1 118806 118806  128      0.50             N/A                N/A           91402.81
     512x1024   3   1 118806 118806  128      0.90             N/A                N/A           17879.50
     1024x128   3   1 118806 118806  128      0.00             yes          167996.48          222323.02
     1024x128   3   1 118806 118806  128      0.50             N/A                N/A          135781.29
     1024x128   3   1 118806 118806  128      0.90             N/A                N/A           26745.03
     1024x256   3   1 118806 118806  128      0.00             yes          167954.50          197222.90
     1024x256   3   1 118806 118806  128      0.50             N/A                N/A           98631.72
     1024x256   3   1 118806 118806  128      0.90             N/A                N/A           20271.77
     1024x512   3   1 118806 118806  128      0.00             yes          167974.17          224656.81
     1024x512   3   1 118806 118806  128      0.50             N/A                N/A           88902.53
     1024x512   3   1 118806 118806  128      0.90             N/A                N/A           19302.96
    1024x1024   3   1 118806 118806  128      0.00             yes          168305.00          195211.47
    1024x1024   3   1 118806 118806  128      0.50             N/A                N/A           92811.27
    1024x1024   3   1 118806 118806  128      0.90             N/A                N/A           16633.18
================================================================================================
```

(only sparsities `0.0 / 0.5 / 0.9` are kept above for brevity; the live run
emits every sparsity in `SPARSITY_VALS`.)

#### Speedup vs `npu_fusion_attention` (dense) — full 4 × 4 grid

Each cell is `Ref_Latency(sparsity=0) / Our_Latency(sparsity=p)`. Values
**< 1.0** mean BSA is slower than PFA dense; **> 1.0** mean BSA is faster.

| Q\KV | sparsity | 128   | 256   | 512   | 1024  |
|------|----------|------:|------:|------:|------:|
| 128  | 0.0      | 0.77× | 0.78× | 0.79× | 0.85× |
|      | 0.5      | 1.30× | 1.72× | 1.82× | 1.90× |
|      | 0.8      | 3.00× | 4.00× | 4.55× | 4.76× |
|      | 0.9      | 6.57× | 7.97× | 8.61× | **9.13×** |
| 256  | 0.0      | 0.79× | 0.79× | 0.79× | 0.86× |
|      | 0.5      | 1.34× | 1.73× | 1.77× | 1.89× |
|      | 0.8      | 3.07× | 4.04× | 4.62× | 4.75× |
|      | 0.9      | 6.74× | 8.32× | 9.23× | **9.75×** |
| 512  | 0.0      | 0.79× | 0.79× | 0.79× | 0.86× |
|      | 0.5      | 1.35× | 1.73× | 1.67× | 1.77× |
|      | 0.8      | 3.10× | 4.05× | 4.42× | 4.55× |
|      | 0.9      | 6.71× | 8.06× | 8.84× | **9.12×** |
| 1024 | 0.0      | 0.80× | 0.79× | 0.80× | 0.85× |
|      | 0.5      | 1.36× | 1.75× | 1.68× | 1.79× |
|      | 0.8      | 3.13× | 4.05× | 4.52× | 4.66× |
|      | 0.9      | 7.18× | 8.52× | 9.02× | **9.89×** |

* Speedup grows monotonically with sparsity in every cell — BSA never regresses
  *worse* than ~0.77× even at sparsity 0, and reaches **6.6×–9.9× at sparsity
  0.9** across the grid.
* **BLOCK_SIZE_KV is the dominant lever.** At sparsity 0.9, moving KV from
  128 → 1024 buys 1.4×–1.5× on top of the existing speedup at every Q; the
  KV=1024 column outperforms every smaller KV at every Q. The same trend
  holds for sparsity 0.8 (≈ 1.5× from KV=128 to KV=1024). This is because
  a wider sabi entry means less sabi-tensor HBM traffic per kept block.
* **BLOCK_SIZE_Q has a much smaller effect.** Q-tile re-parameterisation does
  not change the cube tile width (`basicSInnerSize = 512` for D=128 BF16) or
  the per-Q-row workload; the only effect is collapsing several 128-token Q
  groups onto a single sabi row, which is mostly free. At sparsity 0.9 the
  spread between Q=128 and Q=1024 (same KV) is ≤ 10 %.
* **The dense-mode regression (sparsity 0) shrinks with KV.** At KV=128 the
  overhead vs PFA is ≈ 23 % (0.77×); at KV=1024 it is ≈ 14 % (0.85–0.86×).
  This is the win from the wide-KV packer: each cube tile now reads its K/V
  bytes contiguously without sabi indirection.
* **128×128 remains the most expensive baseline.** Finest sabi resolution but
  the most metadata-per-kept-block.
* **1024×1024 is the fastest at high sparsity** — about 9.9× over PFA dense
  at sparsity 0.9 — though it has the coarsest sparsity-pattern resolution.
  Picking a granularity is therefore an application-level trade-off between
  pattern-fidelity and end-to-end latency; the kernel does not force one.

The PFA reference (`npu_fusion_attention` dense) is shape-invariant — the
small differences in the `Ref_Latency_[usec]` column across block shapes are
timing noise from re-running the same workload 16 times.


### 3 - softmax_lse correctness tests (test_lse.py)

`test_lse.py` verifies the `softmax_lse` (log-sum-exp) second output of the kernel.

```shell
pytest test_lse.py -v
```

Every test case is exercised at all **16 default block shapes**
(`{128, 256, 512, 1024} × {128, 256, 512, 1024}`) via a parametrized
`block_shape` fixture, so the live test count is ~16× the per-shape count
below (~1600 instances total). A handful of cases at small S or high sparsity
are auto-skipped because the test parameterisation becomes inherently
unsatisfiable (forced-frame density exceeds `1 − sparsity`, or the row's
expected kept-KV-block count drops below 1) — see the `pytest.skip` rules in
the file.

#### What is tested per block shape (B=1 only)

| Test | Reference | Shapes | Dtypes | Notes |
|---|---|---|---|---|
| `test_bsa_lse_vs_fias` | FIAS dense | 14 shapes, S up to 24 000 | bfloat16, float16 | Dense mode (no sabi) |
| `test_bsa_lse_vs_fias_sparse` | FIAS sparse_mode=1 | 8 shapes, S up to 24 000 | bfloat16 | H=1, no border frame |
| `test_bsa_lse_vs_fias_sparse_framed` | FIAS sparse_mode=1 | 2 shapes, S ≥ 16 384 | bfloat16 | H=1, FrameWidths(2,1,3,1) |
| `test_bsa_lse_sparse_multihead` | Python float32 reference | 4 shapes, S ≤ 1 024 | bfloat16 | H=2,4; Python loops over sabi |
| `test_bsa_lse_zero_input` | Analytical: log(S) | 5 shapes | bfloat16, float16 | Q=K=V=0 ⟹ uniform softmax |
| `test_bsa_lse_shape_and_dtype` | — | 3 shapes | bfloat16 | Checks shape=[B,H,S], dtype=float32 |
| `test_bsa_lse_disabled_returns_empty` | — | 3 shapes | bfloat16 | flag=False ⟹ numel=0 |

#### Why FIAS is the LSE reference

FIAS (`npu_fused_infer_attention_score`) already exposes a `softmax_lse` output via its `softmaxLseFlag` bool attr (output index 1, shape `[B, N, S, 1]`, float32).  BSA computes the identical quantity — `log(Σ exp(q·kᵀ/√d))` over attended tokens — so FIAS with a matching token-level attention mask serves as a drop-in reference for H=1 shapes.

For H>1 shapes FIAS cannot be used directly (its `sparse_mode=1` requires a `[B,1,S,S]` broadcast mask; passing `[B,H,S,S]` hangs the NPU), so a Python float32 reference that loops over the sabi indices is used instead.

#### Tolerances

| Comparison | atol |
|---|---|
| BSA vs FIAS (bfloat16 input) | 0.01 |
| BSA vs FIAS (float16 input) | 0.01 |
| BSA vs Python float32 reference | 0.01 |
| Zero-input invariant (bf16/fp16) | 0.01 |

float16 is excluded from sparse tests: FIAS processes `-inf`-masked tiles through its flash-attention tiling while BSA skips them entirely; the resulting LSE divergence exceeds any reasonable tolerance at high sparsity.

## Test setup

```
Host CPU: aarch64
Device: Ascend 910B2
Device Driver: 25.3.rc1   
Docker image: `docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops`
OS: ubuntu: 22.04
CANN: 8.5.0-beta.1
Python: 3.11.10
torch: 2.8.0+cpu
torch_npu: 2.8.0
```
