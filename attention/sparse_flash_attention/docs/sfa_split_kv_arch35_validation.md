# A5 SparseFlashAttention Split-KV validation

## Scope and baseline

This change targets Ascend 950 (A5) SparseFlashAttention: matching FP16/BF16
inputs/output, TND/PA_BSND, one KV head, G=1–128, 512-dimensional values,
RoPE absent or dimension 64, sparse mode 0/3, TopK 256–2048, LSE on/off and
no sinks. Other valid shapes retain the original tiling. Trailing sparse
padding is handled by the corrected unsplit row computation described in
[the algorithm](sfa_split_kv_arch35.md).

The baseline is master's native SFA. Both implementations use the same
toolchain and separate vendor prefixes.

Environment: Ascend950DT_9582 (36 AICs), CANN 9.1, Python 3.11.10,
PyTorch 2.10.0 and torch_npu 2.10.0.post4.

## Single-operator measurement

Each graph contains 32 SFA calls. Each round records 101 event-timed replays;
baseline and candidate alternate for three rounds. Reported latency is the
median of the three per-round medians, divided by 32, in microseconds.
Speedup is baseline/candidate; reduction is `(baseline-candidate)/baseline`.
Both arms use identical inputs and the same repository ACLNN V2 adapter.

Default configuration: B=1, one query per batch, G=8, BF16, TopK=2048,
physical KV length 4096, page size 128, RoPE dimension 64, mode 0, LSE off.
The B/Q grid varies B and queries per batch independently. Generalization
sweeps hold B/Q and every other parameter fixed and vary exactly one field.
Because KV has one head, query-head count and G are the same variable.

The [complete CSV](sfa_split_kv_arch35_results.csv) contains 168 sweep points
(160 unique configurations, with the default repeated in each group):

- 16 B/Q combinations: B=1/2/4/8 and Q per batch=1/2/4/8.
- Every integer G from 1 through 128.
- FP16/BF16, mode 0/3, LSE off/on and RoPE 64/absent.
- TopK 128/256/512/768/1000/1024/1536/2048; 128 exercises the non-FD path.
- Page sizes 16/64/128/256/512 and KV lengths 4096/16384/65536.

### B/Q results

Defaults are as above; Q is queries per batch.

| B | Q / batch | Baseline (µs) | With partitioning (µs) | 倍率 | 时延下降 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 69.232 | 17.609 | 3.932x | 74.6% |
| 1 | 2 | 69.191 | 17.993 | 3.845x | 74.0% |
| 1 | 4 | 69.242 | 19.917 | 3.477x | 71.2% |
| 1 | 8 | 69.509 | 27.530 | 2.525x | 60.4% |
| 2 | 1 | 72.090 | 18.197 | 3.962x | 74.8% |
| 2 | 2 | 72.385 | 20.197 | 3.584x | 72.1% |
| 2 | 4 | 72.624 | 28.172 | 2.578x | 61.2% |
| 2 | 8 | 72.961 | 44.125 | 1.653x | 39.5% |
| 4 | 1 | 72.614 | 20.105 | 3.612x | 72.3% |
| 4 | 2 | 72.902 | 28.047 | 2.599x | 61.5% |
| 4 | 4 | 73.045 | 44.134 | 1.655x | 39.6% |
| 4 | 8 | 73.335 | 73.665 | 0.996x | -0.4% |
| 8 | 1 | 72.882 | 28.277 | 2.577x | 61.2% |
| 8 | 2 | 73.229 | 44.136 | 1.659x | 39.7% |
| 8 | 4 | 73.490 | 73.894 | 0.995x | -0.5% |
| 8 | 8 | 136.977 | 137.587 | 0.996x | -0.4% |

### Head generalization

B=1 and Q=1; only G changes. All G=1–128 were measured; the table shows
common values and head-split boundaries. See the CSV for all head counts.

| Query heads / G | Baseline (µs) | With partitioning (µs) | 倍率 | 时延下降 |
| --- | --- | --- | --- | --- |
| 1 | 68.917 | 16.916 | 4.074x | 75.5% |
| 2 | 69.063 | 16.958 | 4.073x | 75.4% |
| 3 | 69.136 | 17.081 | 4.048x | 75.3% |
| 4 | 68.962 | 17.079 | 4.038x | 75.2% |
| 7 | 69.173 | 17.669 | 3.915x | 74.5% |
| 8 | 69.184 | 17.630 | 3.924x | 74.5% |
| 9 | 69.263 | 18.236 | 3.798x | 73.7% |
| 15 | 68.896 | 19.459 | 3.541x | 71.8% |
| 16 | 68.944 | 19.271 | 3.578x | 72.0% |
| 17 | 69.412 | 23.891 | 2.905x | 65.6% |
| 31 | 69.473 | 26.618 | 2.610x | 61.7% |
| 32 | 69.468 | 26.757 | 2.596x | 61.5% |
| 33 | 69.920 | 30.825 | 2.268x | 55.9% |
| 63 | 71.050 | 40.540 | 1.753x | 42.9% |
| 64 | 70.993 | 40.488 | 1.753x | 43.0% |
| 65 | 49.175 | 32.085 | 1.533x | 34.8% |
| 66 | 49.605 | 32.416 | 1.530x | 34.7% |
| 95 | 49.768 | 34.825 | 1.429x | 30.0% |
| 96 | 49.740 | 35.091 | 1.417x | 29.5% |
| 97 | 52.448 | 39.243 | 1.337x | 25.2% |
| 127 | 53.275 | 41.936 | 1.270x | 21.3% |
| 128 | 53.331 | 42.016 | 1.269x | 21.2% |

### Other one-factor-at-a-time sweeps

B=1, Q=1, G=8. Each group changes only the specified field from the default.

| 变量 | 取值 | Baseline (µs) | With partitioning (µs) | 倍率 | 时延下降 |
| --- | --- | --- | --- | --- | --- |
| 数据类型 | BF16 | 69.258 | 17.536 | 3.949x | 74.7% |
| 数据类型 | FP16 | 69.313 | 17.416 | 3.980x | 74.9% |
| Sparse mode | 0 | 69.223 | 17.532 | 3.948x | 74.7% |
| Sparse mode | 3 | 69.205 | 17.587 | 3.935x | 74.6% |
| 返回 LSE | 关 | 69.215 | 17.745 | 3.900x | 74.4% |
| 返回 LSE | 开 | 69.028 | 17.620 | 3.918x | 74.5% |
| RoPE dim | 64 | 69.101 | 17.725 | 3.898x | 74.3% |
| RoPE dim | 不传 RoPE | 61.244 | 16.500 | 3.712x | 73.1% |
| TopK | 128 | 12.603 | 12.588 | 1.001x | 0.1% |
| TopK | 256 | 16.294 | 12.008 | 1.357x | 26.3% |
| TopK | 512 | 23.986 | 12.769 | 1.878x | 46.8% |
| TopK | 768 | 31.541 | 16.456 | 1.917x | 47.8% |
| TopK | 1000 | 38.280 | 14.395 | 2.659x | 62.4% |
| TopK | 1024 | 39.113 | 14.321 | 2.731x | 63.4% |
| TopK | 1536 | 54.253 | 18.140 | 2.991x | 66.6% |
| TopK | 2048 | 69.106 | 17.585 | 3.930x | 74.6% |
| Page size | 16 | 70.100 | 18.886 | 3.712x | 73.1% |
| Page size | 64 | 69.308 | 17.692 | 3.918x | 74.5% |
| Page size | 128 | 69.171 | 17.486 | 3.956x | 74.7% |
| Page size | 256 | 69.078 | 17.312 | 3.990x | 74.9% |
| Page size | 512 | 69.073 | 17.429 | 3.963x | 74.8% |
| KV 长度 | 4096 | 69.149 | 17.683 | 3.910x | 74.4% |
| KV 长度 | 16384 | 69.449 | 18.188 | 3.818x | 73.8% |
| KV 长度 | 65536 | 71.161 | 19.764 | 3.601x | 72.2% |


## Test coverage

- Host UT covers RoPE absent/64, head splits, available
  AIC counts, FD template keys/workspace sizes, and common-validation rejection
  of invalid sparse block sizes and RoPE dimensions.
- Template probes check FD registration for HAS_ROPE=0/1.
- Device coverage includes every head count 1–128, FP16/BF16, modes 0/3,
  RoPE absent/64, LSE off/on, single/multiple batches and eager/graph replay.
- Padding tests cover G=1/3/8/64/65/97/128 and valid-prefix
  lengths 0/1/63/64/65/127/128/129/255/256/257/2047, mixed full/short rows and
  mixed full/empty rows, plus non-FD padding with 40 queries.
- Non-FD coverage includes TopK 128/4096 and
  workloads with too many queries to enable FD.
- All output rows are included in independent FP32 softmax-reference checks:
  relative L2 <0.02 for BF16 and <0.005 for FP16. Returned FP32 max/sum
  are checked separately. Empty rows must have zero output/max/sum.
- Each timing configuration also undergoes independent numerical checks.
- Existing OpDef validation rejects both mixed Q/KV dtype combinations in
  modes 0 and 3. Common tiling validation continues to reject unsupported
  sparse block sizes and RoPE dimensions.

The trailing-padding fix follows the documented valid-prefix/invalid-suffix
contract and the existing golden reference. It addresses the gather/compute
length mismatch reported in [Issue #5208](https://gitcode.com/cann/ops-transformer/issues/5208);
it does not add support for interleaved invalid indices or change sparse-mode semantics.

## Reproduction

Build and install baseline and candidate separately:

```bash
bash build.sh --ophost_test --soc=ascend950 --ops=sparse_flash_attention -j8
bash build.sh --pkg --soc=ascend950 --ops=sparse_flash_attention \
  --vendor_name=sfa_a5_validation -j8
```

Select one private vendor prefix per fresh process, then run:

```bash
python3 -m pytest -q attention/sparse_flash_attention/tests/pytest/test_sfa_split_kv_arch35.py
python3 attention/sparse_flash_attention/tests/pytest/bench_sfa_split_kv_sweeps_arch35.py \
  --calls 32 --repetitions 101
```

For older TorchNPU wrappers that allocate PA LSE outputs incorrectly, use the
repository's unchanged ACLNN V2 extension for both arms:

```bash
mkdir -p /tmp/sfa_arch35_test_adapter
cp attention/sparse_flash_attention/torch_extension/csrc/sparse_flash_attention.cpp \
  torch_extension/cann_ops_transformer/common/aclnn_common.h /tmp/sfa_arch35_test_adapter/
export SFA_TEST_EXTENSION_DIR=/tmp/sfa_arch35_test_adapter
export MAX_JOBS=2
```

The same environment variable is honored by correctness and timing scripts.
This opt-in adapter is not used by the packaged operator or production kernel.
