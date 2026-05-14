# AlltoAllvQuantGMM 小shape模式通信逻辑（修订版）

## 核心思想

**以接收卡视角组织轮次**：每轮只有一个接收卡，接收来自一个发送卡的专家数据。

## 轮次编号

按接收卡视角，轮次顺序：**先按专家，再按发送卡**

```
roundIdx = expertIdx × rankDim + sourceRankIdx
总轮次 = e × rankDim 次（每个接收卡）
```

每轮含义：
- 接收方：按轮次分配（比如轮次0-11给rank0，轮次12-23给rank1...）
- 接收内容：来自 `sourceRankIdx` 的 `expertIdx` 数据

---

## 举例：4卡3专家，接收方Rank0视角

### Rank0 接收顺序（共12轮次）

**接收顺序**：expert0来自rank0 → expert0来自rank1 → expert0来自rank2 → expert0来自rank3 → expert1来自rank0 → expert1来自rank1...

| roundIdx | expertIdx | sourceRankIdx | 接收内容 | recvCnt[sourceRankIdx] |
|----------|-----------|---------------|----------|------------------------|
| 0        | 0         | 0             | expert0来自rank0 | D |
| 1        | 0         | 1             | expert0来自rank1 | D |
| 2        | 0         | 2             | expert0来自rank2 | D |
| 3        | 0         | 3             | expert0来自rank3 | D |
| 4        | 1         | 0             | expert1来自rank0 | D |
| 5        | 1         | 1             | expert1来自rank1 | D |
| 6        | 1         | 2             | expert1来自rank2 | D |
| 7        | 1         | 3             | expert1来自rank3 | D |
| 8        | 2         | 0             | expert2来自rank0 | D |
| 9        | 2         | 1             | expert2来自rank1 | D |
| 10       | 2         | 2             | expert2来自rank2 | D |
| 11       | 2         | 3             | expert2来自rank3 | D |

**Rank0的recvCnt数组变化**：
- 轮次0：`recvCnt = [D, 0, 0, 0]`（接收rank0的expert0）
- 轮次1：`recvCnt = [0, D, 0, 0]`（接收rank1的expert0）
- 轮次2：`recvCnt = [0, 0, D, 0]`（接收rank2的expert0）
- ...

---

## 发送方视角：每张卡如何判断是否发送

**判断逻辑**：
每张卡在每轮次检查：
1. 本轮接收方是谁？（targetRankIdx）
2. 本轮接收方要接收的(sourceRankIdx, expertIdx)是什么？
3. 如果 `myRankIdx == sourceRankIdx`，则发送 `expertIdx` 数据给 `targetRankIdx`

### 举例：轮次1（expertIdx=0, sourceRankIdx=1）

假设接收方是 Rank0，本轮接收 expert0 来自 rank1。

| 卡号 | myRankIdx | sourceRankIdx | 是否发送 | sendCnt[targetRankIdx] |
|------|-----------|---------------|----------|------------------------|
| rank0 | 0 | 1 | ❌ 不发送 | 0 |
| rank1 | 1 | 1 | ✅ 发送 | D（发给rank0） |
| rank2 | 2 | 1 | ❌ 不发送 | 0 |
| rank3 | 3 | 1 | ❌ 不发送 | 0 |

**发送方Rank1的sendCnt数组**：
- `sendCnt = [D, 0, 0, 0]`（只发给rank0）

---

## 接收方分配策略

为简化，假设按轮次顺序轮流分配接收方：
```
targetRankIdx = roundIdx % rankDim
```

但更合理的分配是：
- 每个卡作为接收方的轮次集中在一起
- Rank0 的12轮 → Rank1 的12轮 → Rank2 的12轮 → Rank3 的12轮

**推荐轮次分配**：
```
totalRounds = e × rankDim × rankDim  // 每个卡接收 e×rankDim 次
for targetRankIdx in [0, rankDim-1]:
    for expertIdx in [0, e-1]:
        for sourceRankIdx in [0, rankDim-1]:
            # 轮次：(targetRankIdx, expertIdx, sourceRankIdx)
```

---

## 修正后的轮次编号

**方案A：按接收方集中**
```
roundIdx = targetRankIdx × e × rankDim + expertIdx × rankDim + sourceRankIdx
总轮次 = rankDim × e × rankDim 次（每个卡接收 e×rankDim 次）
```

**方案B：按专家集中（用户描述）**
```
roundIdx = expertIdx × rankDim × rankDim + targetRankIdx × rankDim + sourceRankIdx
总轮次 = e × rankDim × rankDim 次
```

---

## 4卡3专家完整轮次表（方案A）

总轮次：48次（每个卡接收12次）

### Rank0 作为接收方（轮次0-11）

| roundIdx | targetRank | expertIdx | sourceRank | 接收内容 | recvCnt[sourceRank] |
|----------|------------|-----------|------------|----------|---------------------|
| 0        | 0          | 0         | 0          | expert0@rank0 | D |
| 1        | 0          | 0         | 1          | expert0@rank1 | D |
| 2        | 0          | 0         | 2          | expert0@rank2 | D |
| 3        | 0          | 0         | 3          | expert0@rank3 | D |
| 4        | 0          | 1         | 0          | expert1@rank0 | D |
| 5        | 0          | 1         | 1          | expert1@rank1 | D |
| 6        | 0          | 1         | 2          | expert1@rank2 | D |
| 7        | 0          | 1         | 3          | expert1@rank3 | D |
| 8        | 0          | 2         | 0          | expert2@rank0 | D |
| 9        | 0          | 2         | 1          | expert2@rank1 | D |
| 10       | 0          | 2         | 2          | expert2@rank2 | D |
| 11       | 0          | 2         | 3          | expert2@rank3 | D |

### Rank1 作为接收方（轮次12-23）

| roundIdx | targetRank | expertIdx | sourceRank | 接收内容 | recvCnt[sourceRank] |
|----------|------------|-----------|------------|----------|---------------------|
| 12       | 1          | 0         | 0          | expert0@rank0 | D |
| 13       | 1          | 0         | 1          | expert0@rank1 | D |
| 14       | 1          | 0         | 2          | expert0@rank2 | D |
| 15       | 1          | 0         | 3          | expert0@rank3 | D |
| 16       | 1          | 1         | 0          | expert1@rank0 | D |
| 17       | 1          | 1         | 1          | expert1@rank1 | D |
| 18       | 1          | 1         | 2          | expert1@rank2 | D |
| 19       | 1          | 1         | 3          | expert1@rank3 | D |
| 20       | 1          | 2         | 0          | expert2@rank0 | D |
| 21       | 1          | 2         | 1          | expert2@rank1 | D |
| 22       | 1          | 2         | 2          | expert2@rank2 | D |
| 23       | 1          | 2         | 3          | expert2@rank3 | D |

### Rank2 作为接收方（轮次24-35）

| roundIdx | targetRank | expertIdx | sourceRank | 接收内容 | recvCnt[sourceRank] |
|----------|------------|-----------|------------|----------|---------------------|
| 24       | 2          | 0         | 0          | expert0@rank0 | D |
| 25       | 2          | 0         | 1          | expert0@rank1 | D |
| 26       | 2          | 0         | 2          | expert0@rank2 | D |
| 27       | 2          | 0         | 3          | expert0@rank3 | D |
| ...      | ...        | ...       | ...        | ...      | ... |

### Rank3 作为接收方（轮次36-47）

| roundIdx | targetRank | expertIdx | sourceRank | 接收内容 | recvCnt[sourceRank] |
|----------|------------|-----------|------------|----------|---------------------|
| 36       | 3          | 0         | 0          | expert0@rank0 | D |
| 37       | 3          | 0         | 1          | expert0@rank1 | D |
| ...      | ...        | ...       | ...        | ...      | ... |

---

## 每张卡的sendCnt/recvCnt设置逻辑

### 接收方（targetRankIdx == myRankIdx）

```cpp
// 清空recvCnt数组
for (i = 0; i < rankDim; i++) {
    recvCnt[i] = 0;
}
// 设置接收来源
recvCnt[sourceRankIdx] = sendCnt[expertIdx + sourceRankIdx * e] * axis;
```

### 发送方（sourceRankIdx == myRankIdx && targetRankIdx != myRankIdx）

```cpp
// 清空sendCnt数组
for (i = 0; i < rankDim; i++) {
    sendCnt[i] = 0;
}
// 设置发送目标
sendCnt[targetRankIdx] = sendCnt[expertIdx + myRankIdx * e] * axis;
```

### 其他卡（既不接收也不发送）

```cpp
// 所有count设为0
for (i = 0; i < rankDim; i++) {
    sendCnt[i] = 0;
    recvCnt[i] = 0;
}
```

---

## sendCnt数据来源说明

`TaskTilingInfo.sendCnt[expertIdx + rankIdx * e]` 含义：
- **接收方视角**：rankIdx 卡发送 expertIdx 给当前卡的数据量
- **发送方视角**：当前卡发送 expertIdx 给 rankIdx 卡的数据量

这两个值相等（发送方发给接收方的量 = 接收方接收发送方的量）。

---

## 实现关键点

1. **轮次参数获取**：
   ```cpp
   uint32_t targetRankIdx = roundIdx / (e * rankDim);
   uint32_t localRoundIdx = roundIdx % (e * rankDim);
   uint32_t expertIdx = localRoundIdx / rankDim;
   uint32_t sourceRankIdx = localRoundIdx % rankDim;
   ```

2. **角色判断**：
   ```cpp
   bool isReceiver = (myRankIdx == targetRankIdx);
   bool isSender = (myRankIdx == sourceRankIdx && myRankIdx != targetRankIdx);
   ```

3. **AlltoAllv调用**：
   - 所有卡都必须调用（集合通信语义）
   - 通过设置 count=0 实现"不真正发送/接收"

---

## 流水收益分析

### 小shape场景特点
- 计算时间短（bsk < 阈值）
- 通信时间相对较长

### 卡级流水优势
- 细粒度轮次：每轮只处理一个(sourceRank, expert)
- 接收后立即计算：减少等待时间
- 更好的overlap：计算与通信交替进行

### 调用开销
- 调用次数：从 e 次增加到 e×rankDim×rankDim 次
- HCCL AlltoAllv 开销：每次调用有固定开销
- 需要测试验证：细粒度流水能否抵消调用开销