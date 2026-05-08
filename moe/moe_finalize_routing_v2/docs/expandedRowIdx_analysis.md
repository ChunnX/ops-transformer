# expandedRowIdx 参数语义分析

## 算子公式

```
expertid = expertIdx[i, k]

out(i,j) = x1_{i,j} + x2_{i,j} + sum_{k=0}^{K-1}(scales_{i,k} * (expandedX[ expandedRowIdx_{i+k*num_rows}, j ] + bias[expertid, j]))
```

## 核心发现

**`expandedRowIdx` 是一个间接索引表（indirection table）**，每个值代表 `expandedX` 中的行号。

在 kernel 中的核心用法：

```cpp
// moe_finalize_routing_v2_row_k_h_full_load.h:265
expandedXLocal[expandedRowIdxGmValue * tilingData->h]
```

即 `expandedRowIdxValue` 直接作为 `expandedX` 的行索引，取出该 token 该专家的 FFN 输出。

## 两种内存布局

由 `dropPadMode` 控制。同一份"token i 专家 k → expandedX 行号"的映射关系，在内存中有两种排布方式：

| dropPadMode | 场景 | 布局 | expandedRowIdx 偏移公式 | 取值范围 |
|---|---|---|---|---|
| **0** | drop less, col | **列优先** | `expandedRowIdx[i + k * row]` | `[0, row*K - 1]` |
| 1 | drop pad, col | 列优先 | `expandedRowIdx[i + k * row]` | `[-1, E*C - 1]` |
| 2 | drop less, row | 行优先 | `expandedRowIdx[i * K + k]` | `[0, row*K - 1]` |
| 3 | drop pad, row | 行优先 | `expandedRowIdx[i * K + k]` | `[-1, E*C - 1]` |

**取值范围的差异**：
- **drop less (模式 0/2)**：`expandedX` shape 为 `(row*K, H)`，所以 `expandedRowIdx` 取值范围是 `[0, row*K - 1]`
- **drop pad (模式 1/3)**：`expandedX` shape 为 `(E, C, H)`，存在 token 丢弃机制（capacity 限制），所以取值范围是 `[-1, E*C - 1]`，其中 `-1` 表示丢弃

### 列优先 (dropPadMode=0) vs 行优先 (dropPadMode=2) 内存排布对比

同一份 8 个 token、K=2 的索引映射，两种布局下 expandedRowIdx 的 1D 排列顺序不同：

```
模式 0 (列优先):  按"专家 k"分块，先排所有 token 的 k=0，再排所有 token 的 k=1
  offset = i + k * row
  [token0_k0, token1_k0, token2_k0, token3_k0, token4_k0, token5_k0, token6_k0, token7_k0,
   token0_k1, token1_k1, token2_k1, token3_k1, token4_k1, token5_k1, token6_k1, token7_k1]

模式 2 (行优先):  按"token i"分块，每个 token 自己的 K 个专家连续排
  offset = i * K + k
  [token0_k0, token0_k1, token1_k0, token1_k1, token2_k0, token2_k1, token3_k0, token3_k1,
   token4_k0, token4_k1, token5_k0, token5_k1, token6_k0, token6_k1, token7_k0, token7_k1]
```

模式 0 对应 `moe_init_routing` V1 的输出格式；模式 2 对应 `moe_init_routing_v2` 的输出格式。

### INVALID_IDX

值为 **-1** 时表示该 (token, expert) 位置被丢弃，kernel 直接 `continue` 跳过，不参与累加。仅在 drop pad 模式（1/3）下出现；drop less 模式（0/2）下所有值均 ≥0。

### 与 expertIdx 的区别

- **`expertIdx[i, k]`**：表示 token i 第 k 个位置选了**哪个专家**（专家 ID，范围 `[0, E-1]`）
- **`expandedRowIdx[i, k]`**：表示 token i 第 k 个位置的 FFN 输出数据在 `expandedX` 的**哪一行**（物理行号）

两者正交但配套使用：`expertIdx` 用来查 `bias[expertId]`，`expandedRowIdx` 用来查 `expandedX[rowIdx]`。

---

## 构造示例 — dropPadMode=0（列优先，drop less）

参数：**token=8, K=2, E=4, H=16, dropPadMode=0**

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

每行 H=16 列，均填该行的行号值（如 row 4 全是 4.0）。

### 2. expandedRowIdx — shape (16, )

**列优先布局**，偏移公式：`expandedRowIdx[i + k * row]`（i=token索引, k=专家序号, row=8）。

故意打乱行号（不按自然顺序 0,1,2...），体现"间接索引"语义：

```
expandedRowIdx 概念视图 (row=8, K=2):      1D 展平 (16,) = 实际内存布局:
                                          ┌─────────────────────────────────┐
token0, k=0 → expandedRowIdx[ 0] =  4     │ idx 0~7:  所有 token 的第1个专家 │
token1, k=0 → expandedRowIdx[ 1] =  0     │                                    │
token2, k=0 → expandedRowIdx[ 2] =  6     │  [ 0] =  4  ← token0 用第4行     │
token3, k=0 → expandedRowIdx[ 3] =  1     │  [ 1] =  0  ← token1 用第0行     │
token4, k=0 → expandedRowIdx[ 4] = 10     │  [ 2] =  6  ← token2 用第6行     │
token5, k=0 → expandedRowIdx[ 5] =  9     │  [ 3] =  1  ← token3 用第1行     │
token6, k=0 → expandedRowIdx[ 6] =  2     │  [ 4] = 10  ← token4 用第10行    │
token7, k=0 → expandedRowIdx[ 7] = 11     │  [ 5] =  9  ← token5 用第9行     │
──────────────────────────────────────     │  [ 6] =  2  ← token6 用第2行     │
token0, k=1 → expandedRowIdx[ 8] = 12     │  [ 7] = 11  ← token7 用第11行    │
token1, k=1 → expandedRowIdx[ 9] =  8     ├─────────────────────────────────┤
token2, k=1 → expandedRowIdx[10] = 14     │ idx 8~15: 所有 token 的第2个专家 │
token3, k=1 → expandedRowIdx[11] = -1     │                                    │
token4, k=1 → expandedRowIdx[12] =  5     │  [ 8] = 12  ← token0 用第12行    │
token5, k=1 → expandedRowIdx[13] =  3     │  [ 9] =  8  ← token1 用第8行     │
token6, k=1 → expandedRowIdx[14] = 13     │  [10] = 14  ← token2 用第14行    │
token7, k=1 → expandedRowIdx[15] =  7     │  [11] = -1  ← token3 丢弃!       │
                                          │  [12] =  5  ← token4 用第5行     │
                                          │  [13] =  3  ← token5 用第3行     │
                                          │  [14] = 13  ← token6 用第13行    │
                                          │  [15] =  7  ← token7 用第7行     │
                                          └─────────────────────────────────┘
```

**关键观察**：同一个 1D 数组在模式 0 下访问方式不同于模式 2：
- 要取 token3 的 k=1 数据：模式 0 读 `expandedRowIdx[3 + 1*8]` = `expandedRowIdx[11]` = **-1**（丢弃）
- 要取 token5 的 k=0 数据：模式 0 读 `expandedRowIdx[5 + 0*8]` = `expandedRowIdx[5]` = **9**（用第9行）

### 3. 其他参数

```
scales      (8, 2)    = 全 1.0
expertIdx   (8, 2)    = [[0,2],[1,3],[0,1],[2,3],[0,3],[1,2],[0,1],[2,3]]
bias        (4, 16)   = 全 0.0   (不使用)
x1, x2                = 无       (不使用)
```

### 4. 逐 token 计算过程

| token i | k=0 偏移 `i+0*8` | 值 | expandedX[row] | k=1 偏移 `i+1*8` | 值 | expandedX[row] | out[i] |
|---------|-----------------|---|----------------|-----------------|---|----------------|--------|
| 0 | `idx[0]` | 4 | **4.0** | `idx[8]` | 12 | **12.0** | **16.0** |
| 1 | `idx[1]` | 0 | **0.0** | `idx[9]` | 8 | **8.0** | **8.0** |
| 2 | `idx[2]` | 6 | **6.0** | `idx[10]` | 14 | **14.0** | **20.0** |
| 3 | `idx[3]` | 1 | **1.0** | `idx[11]` | -1 | skip | **1.0** |
| 4 | `idx[4]` | 10 | **10.0** | `idx[12]` | 5 | **5.0** | **15.0** |
| 5 | `idx[5]` | 9 | **9.0** | `idx[13]` | 3 | **3.0** | **12.0** |
| 6 | `idx[6]` | 2 | **2.0** | `idx[14]` | 13 | **13.0** | **15.0** |
| 7 | `idx[7]` | 11 | **11.0** | `idx[15]` | 7 | **7.0** | **18.0** |

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

---

## 对比 — 同一映射关系在 dropPadMode=2 下的布局

同样的语义（token0→{4,12}, token1→{0,8}, ...），`dropPadMode=2` 下 expandedRowIdx 布局为：

```
模式 2 (行优先), offset = i * K + k:

expandedRowIdx[ 0] =  4   ← token0, k=0  (0*2+0=0)
expandedRowIdx[ 1] = 12   ← token0, k=1  (0*2+1=1)
expandedRowIdx[ 2] =  0   ← token1, k=0  (1*2+0=2)
expandedRowIdx[ 3] =  8   ← token1, k=1  (1*2+1=3)
expandedRowIdx[ 4] =  6   ← token2, k=0
expandedRowIdx[ 5] = 14   ← token2, k=1
expandedRowIdx[ 6] =  1   ← token3, k=0
expandedRowIdx[ 7] = -1   ← token3, k=1
expandedRowIdx[ 8] = 10   ← token4, k=0
expandedRowIdx[ 9] =  5   ← token4, k=1
expandedRowIdx[10] =  9   ← token5, k=0
expandedRowIdx[11] =  3   ← token5, k=1
expandedRowIdx[12] =  2   ← token6, k=0
expandedRowIdx[13] = 13   ← token6, k=1
expandedRowIdx[14] = 11   ← token7, k=0
expandedRowIdx[15] =  7   ← token7, k=1
```

**同一份语义映射，两种模式下的 1D 排布**：

```
语义映射 (token_i, k_j → expandedX 行号):
  t0→{4,12}, t1→{0,8}, t2→{6,14}, t3→{1,-1}, t4→{10,5}, t5→{9,3}, t6→{2,13}, t7→{11,7}

模式 0 (列优先) 1D:
  [4, 0, 6, 1, 10, 9, 2, 11, 12, 8, 14, -1, 5, 3, 13, 7]
   └─── k=0 所有token ───┘  └─── k=1 所有token ──────────┘

模式 2 (行优先) 1D:
  [4, 12, 0, 8, 6, 14, 1, -1, 10, 5, 9, 3, 2, 13, 11, 7]
   └─t0─┘ └─t1─┘ └─t2─┘ └─t3─┘ └─t4──┘ └─t5─┘ └─t6──┘ └─t7──┘
```

**计算结果是完全一致的**——因为无论哪种布局，每个 (token, k) 最终查到的 expandedRowIdx 值相同，取的都是同一行 expandedX。

## 总结

1. **形状**：1D 张量 `(row × K)`，每个 (token, 专家序号) 对应一个值
2. **核心语义**：**间接行索引表**——值指向 `expandedX` 的行号，告诉算子"该 token 第 k 个专家的 FFN 输出在 expandedX 的哪一行"
3. **INVALID_IDX = -1**：表示该位置被丢弃（仅在 drop pad 模式 1/3），不参与累加
4. **取值与 dropPadMode 的关系**：
   - 模式 0/2（drop less）：`[0, row*K - 1]`，所有值有效
   - 模式 1/3（drop pad）：`[-1, E*C - 1]`，-1 表示专家容量满导致的 token 丢弃
5. **`expandedRowIdx` 不需要按自然顺序排列**——允许重排，因为 expandedX 行顺序由前序 dispatch 决定，expandedRowIdx 承担了"物理行号 → 逻辑 (token, k) 位置"的映射角色
6. **`expertIdx` 说"是哪个专家"，`expandedRowIdx` 说"数据在哪儿"**——两者正交但配套使用
