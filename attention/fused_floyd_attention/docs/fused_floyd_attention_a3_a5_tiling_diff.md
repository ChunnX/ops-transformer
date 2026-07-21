# FusedFloydAttention A3 vs A5 Tiling 对比

> 测试 shape: B=1, H=1, N=16, M=128, K=128, D=32 (FP16 + atten_mask)
> 平台参数:
> - A3: Ascend910_93, 24 cube / 48 vector, UB=192KB, L1=512KB, L0C=128KB
> - A5: Ascend950PR, 28 cube / 56 vector, UB=248KB, L1=512KB, L0C=256KB

## 对比结果

tiling 结果几乎完全一致，仅有 5 处差异：

| # | 位置 | 字段 | A3 | A5 | 说明 |
|---|---|---|---|---|---|
| 1 | Global | workspace[0] | 142606336 (136 MB) | 163577856 (156 MB) | A5 L0C 扩容导致单核 workspace 增大 |
| 2 | [114] Bmm1Tiling | shareL0CSize | 131072 (128KB) | 262144 (256KB) | A5 L0C 翻倍 |
| 3 | [139] Bmm1k1Tiling | shareL0CSize | 131072 (128KB) | 262144 (256KB) | 同上 |
| 4 | [164] Bmm2Tiling | shareL0CSize | 131072 (128KB) | 262144 (256KB) | 同上 |
| 5 | [189] Bmm2v2Tiling | shareL0CSize | 131072 (128KB) | 262144 (256KB) | 同上 |

## 一致项

- tilingKey: `-8446744072687520673`（无符号 `10000000001022030943`）—— A3/A5 完全一致
- blockNum: 2
- tilingDataSize: 1856
- InputParams / MultiCoreParams / CoreParams / TensorSizeParams: 完全一致
- SoftmaxFlashTiling: 完全一致

## 结论

tiling 逻辑在 A5 上工作正常。差异仅来自 L0C 容量不同（128KB→256KB），体现为 4 个 Bmm tiling 的 `shareL0CSize` 翻倍和 workspace 相应增大，符合预期。

> 注：A3 的 `coreNum` 在 tiling 中为 2（totalSize 较小），与 A5 一致；A3 的 48 vector core 和 A5 的 56 vector core 在本 shape 下未造成差异（数据量不足以占满更多核）。
