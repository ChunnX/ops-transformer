# expandedRowIdx 参数语义分析

## 算子公式

```
expertid = expertIdx[i, k]

out(i,j) = x1_{i,j} + x2_{i,j} + sum_{k=0}^{K-1}(scales_{i,k} * (expandedX[ expandedRowIdx_{i+k*num_rows}, j ] + bias[expertid, j]))
```

## 核心发现

**`expandedRowIdx` 是一个间接索引表（indirection table）**，每个值代表 `expandedX` 中的行号。

在 kernel 中的核心用法（`moe_finalize_routing_v2_row_k_h_full_load.h:265`）：

```cpp
expandedXLocal[expandedRowIdxGmValue * tilingData->h]
```

### 两种内存布局

由 `dropPadMode` 控制：

| dropPadMode | 布局 | expandedRowIdx 偏移公式 |
|---|---|---|
| 0 (drop less, col) | 列优先 | `expandedRowIdx[i + k * row]` |
| 2 (drop less, row) | 行优先 | `expandedRowIdx[i * k + k]` |

### INVALID_IDX

值为 **-1** 时表示该 (token, expert) 位置被丢弃，kernel 直接 `continue` 跳过不参与累加。

### 与 expertIdx 的区别

- **`expertIdx[i, k]`**：表示 token i 第 k 个位置选了**哪个专家**（专家 ID）
- **`expandedRowIdx[i, k]`**：表示 token i 第 k 个位置的 FFN 输出数据在 `expandedX` 的**哪一行**

两者正交但配套使用。

---

## 构造示例

参数设置：**token=8, K=2, E=4, H=16, dropPadMode=2**（行优先布局）

### 1. expandedX — shape (16, 16)

每行是某个 token 的某个专家 FFN 输出。每行填充自身的行号值便于追踪：

```
expandedX = [
  [ 0.0,  0.0, ...]  // row 0:  token0 的第1个专家输出
  [ 1.0,  1.0, ...]  // row 1:  token0 的第2个专家输出
  [ 2.0,  2.0, ...]  // row 2:  token1 的第1个专家输出
  [ 3.0,  3.0, ...]  // row 3:  token1 的第2个专家输出
  [ 4.0,  4.0, ...]  // row 4:  token2 的第1个专家输出
  [ 5.0,  5.0, ...]  // row 5:  token2 的第2个专家输出
  [ 6.0,  6.0, ...]  // row 6:  token3 的第1个专家输出
  [ 7.0,  7.0, ...]  // row 7:  token3 的第2个专家输出
  [ 8.0,  8.0, ...]  // row 8:  token4 的第1个专家输出
  [ 9.0,  9.0, ...]  // row 9:  token4 的第2个专家输出
  [10.0, 10.0, ...]  // row 10: token5 的第1个专家输出
  [11.0, 11.0, ...]  // row 11: token5 的第2个专家输出
  [12.0, 12.0, ...]  // row 12: token6 的第1个专家输出
  [13.0, 13.0, ...]  // row 13: token6 的第2个专家输出
  [14.0, 14.0, ...]  // row 14: token7 的第1个专家输出
  [15.0, 15.0, ...]  // row 15: token7 的第2个专家输出
]
```

每行 H=16 列，均填该行的行号值。

### 2. expandedRowIdx — shape (16, ) 概念上 reshape 为 (8, 2)

Row 布局 (`dropPadMode=2`)，展开顺序：`[i=0,k=0], [i=0,k=1], [i=1,k=0], ...`

**故意打乱 `expandedRowIdx` 的值（不按 0,1,2,... 顺序），体现其"间接索引"的语义**，并让 token3 的第 2 个专家位置被丢弃：

```
expandedRowIdx 概念视图 (8, 2):          展平为 1D (16,):
┌──────────┬───────────────────┐        expandedRowIdx[ 0] =  4   ← token0, k=0 用 expandedX 第4行
│ token    │ expandedRowIdx 值  │        expandedRowIdx[ 1] = 12   ← token0, k=1 用 expandedX 第12行
├──────────┼───────┬───────────┤        expandedRowIdx[ 2] =  0   ← token1, k=0 用 expandedX 第0行
│ token 0  │  k=0  │     4     │        expandedRowIdx[ 3] =  8   ← token1, k=1 用 expandedX 第8行
│          │  k=1  │    12     │        expandedRowIdx[ 4] =  6   ← token2, k=0 用 expandedX 第6行
├──────────┼───────┼───────────┤        expandedRowIdx[ 5] = 14   ← token2, k=1 用 expandedX 第14行
│ token 1  │  k=0  │     0     │        expandedRowIdx[ 6] =  1   ← token3, k=0 用 expandedX 第1行
│          │  k=1  │     8     │        expandedRowIdx[ 7] = -1   ← token3, k=1 丢弃!
├──────────┼───────┼───────────┤        expandedRowIdx[ 8] = 10   ← token4, k=0 用 expandedX 第10行
│ token 2  │  k=0  │     6     │        expandedRowIdx[ 9] =  5   ← token4, k=1 用 expandedX 第5行
│          │  k=1  │    14     │        expandedRowIdx[10] =  9   ← token5, k=0 用 expandedX 第9行
├──────────┼───────┼───────────┤        expandedRowIdx[11] =  3   ← token5, k=1 用 expandedX 第3行
│ token 3  │  k=0  │     1     │        expandedRowIdx[12] =  2   ← token6, k=0 用 expandedX 第2行
│          │  k=1  │    -1     │        expandedRowIdx[13] = 13   ← token6, k=1 用 expandedX 第13行
├──────────┼───────┼───────────┤        expandedRowIdx[14] = 11   ← token7, k=0 用 expandedX 第11行
│ token 4  │  k=0  │    10     │        expandedRowIdx[15] =  7   ← token7, k=1 用 expandedX 第7行
│          │  k=1  │     5     │
├──────────┼───────┼───────────┤
│ token 5  │  k=0  │     9     │
│          │  k=1  │     3     │
├──────────┼───────┼───────────┤
│ token 6  │  k=0  │     2     │
│          │  k=1  │    13     │
├──────────┼───────┼───────────┤
│ token 7  │  k=0  │    11     │
│          │  k=1  │     7     │
└──────────┴───────┴───────────┘
```

### 3. 其他参数

```
scales      (8, 2)    = 全 1.0
expertIdx   (8, 2)    = [[0,2],[1,3],[0,1],[2,3],[0,3],[1,2],[0,1],[2,3]]
bias        (4, 16)   = 全 0.0   (不使用)
x1, x2                = 无       (不使用)
```

### 4. 逐 token 计算过程

| token i | k=0: expandedX[row] | k=1: expandedX[row] | 求和 = out[i] (每个元素) |
|---------|---------------------|---------------------|-------------------------|
| 0       | `expandedX[4]` = **4.0** | `expandedX[12]` = **12.0** | **16.0** |
| 1       | `expandedX[0]` = **0.0** | `expandedX[8]` = **8.0** | **8.0** |
| 2       | `expandedX[6]` = **6.0** | `expandedX[14]` = **14.0** | **20.0** |
| 3       | `expandedX[1]` = **1.0** | `skip (INVALID)` | **1.0** |
| 4       | `expandedX[10]` = **10.0** | `expandedX[5]` = **5.0** | **15.0** |
| 5       | `expandedX[9]` = **9.0** | `expandedX[3]` = **3.0** | **12.0** |
| 6       | `expandedX[2]` = **2.0** | `expandedX[13]` = **13.0** | **15.0** |
| 7       | `expandedX[11]` = **11.0** | `expandedX[7]` = **7.0** | **18.0** |

### 5. 最终输出 `out` — shape (8, 16)

```
out[0] = [16.0, 16.0, ... 16.0]
out[1] = [ 8.0,  8.0, ...  8.0]
out[2] = [20.0, 20.0, ... 20.0]
out[3] = [ 1.0,  1.0, ...  1.0]    ← 只累加了1个专家 (k=1 被丢)
out[4] = [15.0, 15.0, ... 15.0]
out[5] = [12.0, 12.0, ... 12.0]
out[6] = [15.0, 15.0, ... 15.0]
out[7] = [18.0, 18.0, ... 18.0]
```

## 总结

1. **形状**：1D 张量 `(token × K)`，即每个 (token, 专家序号) 对应一个值
2. **核心语义**：它是**间接行索引表**——值指向 `expandedX` 的行号，告诉 operator "这个 token 的第 k 个专家对应的 FFN 输出在 `expandedX` 的哪一行"
3. **INVALID_IDX = -1**：表示该位置被丢弃，对应专家输出不被累加
4. **`expandedRowIdx` 不需要按 0,1,2... 顺序排列**——它允许重排，因为 `expandedX` 的行顺序是由前序 `moe_init_routing` 的 dispatch 决定的，`expandedRowIdx` 承担了"将 dispatch 后的物理行号映射回逻辑 (token, k) 位置"的角色
5. **`expertIdx` 说"是哪个专家"，`expandedRowIdx` 说"数据在哪儿"**——两者正交但配套使用
