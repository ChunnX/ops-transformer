# SlidingTileAttention perf_a100 Loop Counts

This note records the loop counts and rough compute volume for:

```text
perf_a100_b1_h8_d64_img_30x48x80_text_128_mixed_windows
```

## Case Shape

```text
B = 1
H = 8
D = 64
Image shape = 30 x 48 x 80
Image tokens = 115200
Text tokens = 128
SeqLen = 115328

Image tile = 6 x 8 x 8 = 384 tokens
Image tile grid = 5 x 6 x 10 = 300 tiles
```

## Query Groups

The GM-workspace pipeline schedules natural query groups instead of synthetic query blocks:

```text
image query groups per head = 300
text query groups per head = 1
query groups per head = 301
total query-group tasks = 1 * 8 * 301 = 2408
```

Each image query group contains one full 384-token image tile. The text query group contains the whole text span.

## Per-Head Visible Ranges

For image queries, each image query group processes visible image tiles plus one full text range.

| Head | Window | Image ranges per image query group | Total KV ranges per image query group |
|---:|---:|---:|---:|
| 0 | `1x1x1` | 1 | 2 |
| 1 | `3x3x3` | 27 | 28 |
| 2 | `3x5x5` | 75 | 76 |
| 3 | `5x5x5` | 125 | 126 |
| 4 | `1x3x3` | 9 | 10 |
| 5 | `3x3x5` | 45 | 46 |
| 6 | `5x5x7` | 175 | 176 |
| 7 | `7x7x7` | 210 | 211 |

Head 7 is clamped by the tile grid `5x6x10`, so `7x7x7` becomes `5x6x7 = 210`.

For text queries, the text query group attends to all image tiles plus one full text range:

```text
image tile ranges = 300
text range = 1
total KV ranges for the text query group = 301
```

## Total QK/PV Calls

Per head:

| Head | Image query-group calls | Text query-group calls | Total calls |
|---:|---:|---:|---:|
| 0 | `300 * 2 = 600` | 301 | 901 |
| 1 | `300 * 28 = 8400` | 301 | 8701 |
| 2 | `300 * 76 = 22800` | 301 | 23101 |
| 3 | `300 * 126 = 37800` | 301 | 38101 |
| 4 | `300 * 10 = 3000` | 301 | 3301 |
| 5 | `300 * 46 = 13800` | 301 | 14101 |
| 6 | `300 * 176 = 52800` | 301 | 53101 |
| 7 | `300 * 211 = 63300` | 301 | 63601 |

Totals:

```text
total visible KV ranges = 204908
QK matmul calls = 204908
PV matmul calls = 204908
```

## Per-Range Compute

For one image-tile range with image queries:

```text
QK: [384, 64] x [384, 64]^T
PV: [384, 384] x [384, 64]

QK MAC = 384 * 384 * 64 = 9437184
PV MAC = 384 * 384 * 64 = 9437184
Total = 18874368 MAC
```

For one text range with image queries:

```text
QK: [384, 64] x [128, 64]^T
PV: [384, 128] x [128, 64]

QK MAC = 384 * 128 * 64 = 3145728
PV MAC = 384 * 128 * 64 = 3145728
Total = 6291456 MAC
```

For one image-tile range with text queries:

```text
QK: [128, 64] x [384, 64]^T
PV: [128, 384] x [384, 64]

QK MAC = 128 * 384 * 64 = 3145728
PV MAC = 128 * 384 * 64 = 3145728
Total = 6291456 MAC
```

The arithmetic volume stays close to the prior smaller-row schedule, but the scheduler now launches far fewer AIC/AIV handoff rounds.

## Current Main Observation

The natural query-group schedule removes the synthetic query and text-KV block loops. Intermediate score, probability, and output ranges are handed off through GM workspace with pingpong slots, so the remaining fragmentation mostly follows real image tiles and the single text span.
