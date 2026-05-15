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

qBlock = 16
image kv block = 384
text kv block = 128
```

## Query Blocks

```text
qBlockNum per head = 115328 / 16 = 7208
image qBlocks per head = 115200 / 16 = 7200
text qBlocks per head = 128 / 16 = 8
total qBlock tasks = 1 * 8 * 7208 = 57664
```

## Per-Head Image Window Blocks

For image queries, each qBlock processes visible image tiles plus one text kv block.

| Head | Window | Image tiles per image qBlock | Total kv blocks per image qBlock |
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

## Text Query Blocks

For text queries, each qBlock attends to all image tiles plus one text kv block:

```text
image tile blocks = 300
text kv blocks = 1
total kv blocks per text qBlock = 301
```

## Total QK/PV Tile Calls

Per head:

| Head | Image qBlock calls | Text qBlock calls | Total calls |
|---:|---:|---:|---:|
| 0 | `7200 * 2 = 14400` | `8 * 301 = 2408` | 16808 |
| 1 | `7200 * 28 = 201600` | 2408 | 204008 |
| 2 | `7200 * 76 = 547200` | 2408 | 549608 |
| 3 | `7200 * 126 = 907200` | 2408 | 909608 |
| 4 | `7200 * 10 = 72000` | 2408 | 74408 |
| 5 | `7200 * 46 = 331200` | 2408 | 333608 |
| 6 | `7200 * 176 = 1267200` | 2408 | 1269608 |
| 7 | `7200 * 211 = 1519200` | 2408 | 1521608 |

Totals:

```text
image-tile 384-token blocks = 4802400 + 19200 = 4821600
text 128-token blocks = 57600 + 64 = 57664
total kv blocks = 4879264
```

So AIC runs approximately:

```text
QK matmul calls = 4879264
PV matmul calls = 4879264
```

AIV runs one softmax stage per kv block:

```text
softmax row-block count = 4879264 * 16 = 78068224 rows
```

With task ratio 2, each AIV subblock handles about 8 rows, but the aggregate is still 16 rows per qBlock.

## Per-Block Compute

For one image 384-token block:

```text
QK: [16, 64] x [384, 64]^T
PV: [16, 384] x [384, 64]

QK MAC = 16 * 384 * 64 = 393216
PV MAC = 16 * 384 * 64 = 393216
Total = 786432 MAC
```

For one text 128-token block:

```text
QK MAC = 16 * 128 * 64 = 131072
PV MAC = 16 * 128 * 64 = 131072
Total = 262144 MAC
```

## Total Rough MAC Count

```text
image blocks MAC = 4821600 * 786432 = 3794486476800
text blocks MAC = 57664 * 262144 = 15116271616
total MAC = 3809602748416
```

That is about:

```text
3.81e12 MAC
```

## Current Main Observation

The 384-token image block reduces the previous three 128-token image blocks into one block. This keeps the arithmetic count similar, but reduces:

```text
AIC/AIV sync count
AIV softmax block count
prob write transactions
out read transactions
per-block scalar scheduling overhead
```

This is why the profile improved by roughly 2.8x on the large image cases.
