# 2026-04-29 S=64 Matmul MAC 分析

## 结论

`S=64,Dk=Dv=64,Hq=Nv=16,Hkv=Nk=4` 下，当前 Neumann 新算子的 **理论 MAC 总量并没有变多**，按 MNK 逻辑乘法次数估算约为普通算子的 `0.561x`。

profiling 里 `aic_mac_time_us` 变成约 `1.724x`，不是因为数学 MAC 总量变多，而是因为：

1. Neumann 求逆子项本身比原 32×32 block inverse 多 `1.5x` MAC。
2. 总体上虽然 MAC 少，但从少量大矩阵变成大量 `16×16` tiny matmul，Cube 利用率下降，MAC time 不再正比于理论 MAC。
3. `S=64` 时 normal 只有 1 个 C64 chunk，Neumann 有 4 个 C16 chunk，matmul 调用次数从每 head 8 次变为每 head 48 次，固定调度开销和低利用率被 profiler 计入 AIC/MAC 相关时间。

## 变量定义

| 符号 | 含义 | normal | neumann |
|---|---|---:|---:|
| `S` | sequence length | 64 | 64 |
| `C` | chunk size | 64 | 16 |
| `P=S/C` | chunk 数 | 1 | 4 |
| `Dk` | key/query dim | 64 | 64 |
| `Dv` | value dim | 64 | 64 |
| `Nv` | value/query head 数 | 16 | 16 |
| `B` | batch | 8/16/32/64 | 8/16/32/64 |
| `I` | normal inverse block | 32 | - |
| `R` | Neumann round 数 | - | 3 |

普通算子 tiling 为 `C=64`，Neumann 当前 tiling 为 `C=16`。

## Stage1 MAC 展开

### normal C64

每个 `B,Nv` 下只有 `P=1` 个 chunk。

Stage1 AIC matmul：

1. `K @ K^T`: `C*C*Dk`
2. `Q @ K^T`: `C*C*Dk`
3. block inverse 合并左下角两次 matmul，`I=32`：
   - `leftDown @ leftTop`: `I*I*I`
   - `rightDown @ leftDown`: `I*I*I`
   - 合计 `2*I^3`
4. `attn @ gBK`: `C*C*Dk`
5. `attn @ vBeta`: `C*C*Dv`

公式：

```text
normal_stage1_per_BNv
= P * (3*C^2*Dk + C^2*Dv + 2*I^3)
= 1 * (3*64^2*64 + 64^2*64 + 2*32^3)
= 1,114,112
```

### Neumann C16

每个 `B,Nv` 下有 `P=4` 个 chunk。

每个 C16 chunk 的 Stage1 AIC matmul：

1. `K @ K^T`: `C*C*Dk`
2. `Q @ K^T`: `C*C*Dk`
3. Neumann 3 轮，每轮两次 `16×16×16`：
   - `L_pow = L_pow @ L_pow`
   - `INV += INV @ L_pow`
   - 合计 `2*R*C^3`
4. `INV @ gBK`: `C*C*Dk`
5. `INV @ vBeta`: `C*C*Dv`

公式：

```text
neumann_stage1_per_BNv
= P * (3*C^2*Dk + C^2*Dv + 2*R*C^3)
= 4 * (3*16^2*64 + 16^2*64 + 2*3*16^3)
= 360,448
```

Stage1 理论 MAC ratio：

```text
360,448 / 1,114,112 = 0.324x
```

所以如果只看 Stage1 总理论 MAC，Neumann 应该更少。

## 求逆子项单独比较

求逆这一个子项确实是 Neumann 更多：

```text
normal_inverse_per_BNv = 2 * 32^3 = 65,536
neumann_inverse_per_BNv = 4 chunks * 3 rounds * 2 matmul * 16^3
                       = 98,304
ratio = 1.5x
```

这解释了“求逆轮次乘上以后，新算子求逆 MAC 反而多”的现象。

但求逆只占 Stage1 的一部分；`K@K^T/Q@K^T/INV@gBK/INV@vBeta` 随 `C` 从 64 降到 16 后，跨 4 个 chunk 合计仍下降到 `1/4`：

```text
4 * 16^2 * D / (64^2 * D) = 1/4
```

因此 Stage1 总理论 MAC 仍下降。

## Stage2 / Stage3 MAC 展开

Stage2 每个 token 仍做三次状态相关 matmul：

```text
stage2_per_BNv = 3 * S * Dk * Dv
               = 3 * 64 * 64 * 64
               = 786,432
```

Stage2 理论 MAC 与 chunkSize 基本无关。

Stage3：

normal C64：

```text
normal_stage3_per_BNv = S * C * Dv
                      = 64 * 64 * 64
                      = 262,144
```

Neumann C16：

```text
neumann_stage3_per_BNv = S * C * Dv
                       = 64 * 16 * 64
                       = 65,536
```

Stage3 理论 MAC ratio 为 `0.25x`。

## 全算子理论 MAC

每个 `B,Nv`：

```text
normal_total = 1,114,112 + 786,432 + 262,144
             = 2,162,688

neumann_total = 360,448 + 786,432 + 65,536
              = 1,212,416

ratio = 1,212,416 / 2,162,688 = 0.561x
```

乘上 `B*Nv` 后：

| B | normal MAC | neumann MAC | ratio |
|---:|---:|---:|---:|
| 8 | 276,824,064 | 155,189,248 | 0.561x |
| 16 | 553,648,128 | 310,378,496 | 0.561x |
| 32 | 1,107,296,256 | 620,756,992 | 0.561x |
| 64 | 2,214,592,512 | 1,241,513,984 | 0.561x |

## profiling 对照

`S=64,D=64,Hq=16,Hkv=4` 的实测：

| B | normal aic_mac_time(us) | neumann aic_mac_time(us) | ratio |
|---:|---:|---:|---:|
| 8 | 10.636 | 18.337 | 1.724x |
| 16 | 21.271 | 36.679 | 1.724x |
| 32 | 42.543 | 73.439 | 1.726x |
| 64 | 85.085 | 146.829 | 1.726x |

这个 ratio 随 B 非常稳定，说明不是某个 batch case 异常，而是 C16 tiny matmul 的系统性低效率。

## 为什么理论 MAC 少，MAC time 反而多

### 1. Matmul 调用次数放大

每个 `B,Nv` 下：

normal C64：

```text
Stage1: 2 + 2 + 2 = 6 次
Stage2: 3 次
Stage3: 1 次
合计: 10 次 matmul
```

Neumann C16：

```text
每个 C16 chunk:
  Stage1: 2 + 6 + 2 = 10 次
  Stage2: 3 次
  Stage3: 1 次
  合计: 14 次

S=64 有 4 个 chunk:
  合计: 56 次 matmul
```

如果只看 AIC matmul 调用次数，Neumann 是 normal 的 `5.6x`。

### 2. Neumann 新增的是极小矩阵

Neumann 新增 24 次 `16×16×16` matmul：

```text
4 chunks * 3 rounds * 2 matmul = 24
```

单次只有 `4096` 理论 MAC。该尺寸下 Cube pipeline、MTE、FixPipe、SetTensor/IterateAll 调度成本占比很高，实际 `aic_mac_time_us` 不会按 `4096 MAC` 线性缩小。

### 3. normal 的大矩阵更容易吃满 Cube

normal 的 `64×64×64`、`64×64×Dv` 等矩阵单次 MAC 更大，固定开销摊薄；Neumann 虽然总理论 MAC 少，但被拆成更多小任务，Cube 利用率下降。

profiling 中 `aic_mac_ratio` 也验证了这一点：

- normal `S=64`：约 `0.016~0.017`
- neumann `S=64`：约 `0.013~0.014`

Neumann 的 MAC 时间更长，但 MAC 占比更低，说明更多时间花在低效调度/流水开销上。

## 结论修正

严格说法应该是：

- **求逆部分理论 MAC 增加了 `1.5x`**。
- **全 Stage1 理论 MAC 降到 `0.324x`**。
- **全算子理论 MAC 降到 `0.561x`**。
- profiling 里 `aic_mac_time_us` 增加到 `1.724x`，反映的是 `16×16` tiny matmul 的低 Cube 效率和 matmul 调用次数放大，不是数学 MAC 总量增加。

