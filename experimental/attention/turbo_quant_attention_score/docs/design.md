# TurboQuantAttentionScore 设计说明

## 1. 数学契约

与 golden 一致：

```text
score[i] = x_norm[i] * <Hq, dequant(idx[i])>
         + (sqrt(pi/2) / head_dim) * gamma[i] * <S(Hq), qjl_signs[i]>
```

### 1.1 MSE 解包

- `mse_bits=3`：packed last dim = 48；reshape 为 `(..., 16, 3)`，
  `word = b0|(b1<<8)|(b2<<16)`，再取 `(word>>s)&7`，`s=0,3,...,21`，得 128 个 index。
- `mse_bits=2/4`：按字节内从低位起每 `bits` 位取一个 index，reshape 到 `head_dim`。
- index 查表映射到对应 centroids（2/3/4 bit 各一套常量表）。

### 1.2 QJL 解包

每个 byte 的 8 bit 从低到高：`0 -> -1`，`1 -> +1`，共 `head_dim` 个符号。

### 1.3 Paged 寻址

```text
batchIdx = (batch == 1) ? 0 : token
logical = kvPos / 128
offset  = kvPos % 128
phys    = block_table[batchIdx, logical]
kvHead  = qHead / groups
```

`kvPos >= seq_lens[batchIdx]` 的位置写 `-inf`。

## 2. Kernel 数据流

Vector Core 融合基线，无用户 workspace：

1. 按 `(token, q_head)` 循环分核（`blockDim = min(T*H, aivCoreNum)`）。
2. 每个 query 行搬入 UB，Cast 为 fp32 常驻。
3. KV 按 `kvTile` 分块；每个有效位置：
   - 查 `block_table` 得 physical block；
   - 搬入 packed idx/qjl 与 bf16 norm/gamma；
   - 标量解包 + centroid 查表得到 dequant / signs；
   - UB 内 `Mul + ReduceSum` 算两个点积并合成 score。
4. centroids 在 Init 时写入 UB 常驻；`mse_bits` 由 tiling key 区分。
5. `PipeBarrier<PIPE_V>` / `HardEvent`（`MTE2_S`、`V_S`、`S_V`、`S_MTE3`）与
   scaled_cosine 同风格，保证标量/向量/搬运依赖正确。

## 3. Tiling

Host 侧通过 `PlatformAscendC` 读取 AIV core 数与 UB 容量：

| 字段 | 含义 |
|---|---|
| numTokens / numQHeads / numKvHeads / headDim | 运行时维度 |
| blockSize / numBlocks | 固定 128 / cache 块数 |
| idxPackedBytes / qjlPackedBytes | packed 末维 |
| maxBatch / maxBlocksPerSeq / maxKvLen | paged 与输出宽 |
| kvTile | 按 UB 预算反推，上限 64 |
| usedCoreNum | `min(T*H, aivCoreNum)` |
| mseBits / groups / qjlCoeff | 属性与派生量 |

Tiling key：`0/1/2` 对应 `mse_bits=2/3/4`。

## 4. 校验与边界

Host 拒绝：

- rank/dtype 不符合接口；
- `block_size != 128` 或 packed bytes 与 `mse_bits` 不一致；
- 非 GQA（`H % KvH != 0`）；
- `batch > 1` 且 `T != batch`；
- `mse_bits` 非 `{2,3,4}`；
- UB 放不下最小 tile。

## 5. 后续性能路线

当前点积走 Vector reduction，aclnn kv4k 约 **6.8×** Full-QK。
msprof 与预解量化 GEMM（约 **1.13×**）表明：达标关键是 **Cube**
`Hq @ Y_hat^T` / `query_qjl @ Signs^T`，并把 bit-unpack 留在 Vector 与 Cube 流水重叠。

`S(Hq) @ Signs^T` tiling key。
