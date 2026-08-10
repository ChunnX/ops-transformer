# SplitFD 函数逐行详解

## 函数位置

[split_core_v2.cpp:625-674](./split_core_v2.cpp#L625-L674)

## 函数签名

```cpp
void SplitFD(FAMetaData &result)
```

## 功能概述

`SplitFD` 在 `SplitCore` 的最后阶段被调用（仅当 `fdRes.fdNum > 0` 时），负责将 FD 归约任务**按负载均衡**地分配到所有可用的 Vec 核上。每个 FD 归约任务对应一行被 S2 跨核切分的 Q 行，需要把多个核产生的中间结果（accumOut + LSE）做 LSE 加权合并。

## 调用时机

```
SplitCore()
├─ 1. CalcSplitInfo() + CalcCostInfo()
├─ 2. CalcSplitPlan()  →  产生 fdRes.fdNum 个 FD 归约任务
├─ 3. CheckChooseWithFd()  →  决策是否最终采用 FD 方案
└─ 4. if (fdRes.fdNum > 0) → SplitFD(result)  ← 本函数
```

## 输入输出

### 输入（`result.fdRes` 中已有）

| 字段 | 含义 |
|------|------|
| `fdNum` | FD 归约任务数量 |
| `fdS2SplitNum[i]` | 第 i 个归约任务的 S2 切分份数（参与合并的中间结果份数） |
| `mSize[i]` | 第 i 个归约任务的 M 轴大小（该归约任务覆盖的 Q 行数） |

### 输出（`result.fdRes` 中填充）

| 字段 | 含义 |
|------|------|
| `taskIdx[vecIdx]` | 第 vecIdx 个 Vec 核处理的归约任务 ID |
| `mStart[vecIdx]` | 该 Vec 核处理的 M 轴起点 |
| `mLen[vecIdx]` | 该 Vec 核处理的 M 轴行数 |
| `fdUsedVecNum` | FD 阶段实际使用的 Vec 核总数 |

## 逐行解析

### 第1段：计算 FD 总负载

```cpp
FlashDecodeResult &fdRes = result.fdRes;
// 计算FD的总数据量
uint64_t totalFDLoad = 0;
for (uint32_t i = 0; i < result.fdRes.fdNum; i++) {
    totalFDLoad += fdRes.fdS2SplitNum[i] * fdRes.mSize[i];
}
```

**逻辑**：遍历所有 FD 归约任务，累加每个任务的"切分份数 × M 行数"作为总负载。

**物理含义**：`fdS2SplitNum[i] × mSize[i]` 表示第 i 个归约任务需要读取的数据块数。每个 Vec 核在规约时按 (份数, M行) 循环读取 GM 上的中间结果，因此这个乘积反映了规约的计算量。

**举例**：
- 任务0：3 份 × 128 行 = 384
- 任务1：2 份 × 64 行 = 128
- totalFDLoad = 512

---

### 第2段：计算可用 Vec 核数与平均负载

```cpp
uint32_t maxVectorNum = result.usedCoreNum * result.vecCubeRatio;
uint32_t emptyVectorNum = maxVectorNum - result.fdRes.fdNum;
uint64_t averageLoad = (totalFDLoad + maxVectorNum - 1U) / maxVectorNum; // 向上取整，避免核负载为0
```

**逐行**：

1. **`maxVectorNum`**：总 Vec 核数 = 使用的 Cube 核数 × Vec/Cube ratio。每个 Cube 核绑定 `vecCubeRatio` 个 Vec 核。

2. **`emptyVectorNum`**：空闲 Vec 核数。每个 FD 归约任务**自带 1 个 Vec 核**（Cube 核完成 FA 后自身的 Vec 核可直接做规约），所以 `emptyVectorNum = maxVectorNum - fdNum` 表示额外可用的空闲核数。

3. **`averageLoad`**：每个 Vec 核的平均负载，向上取整（`+maxVectorNum-1` 再整除）保证非零。

**举例**：若 `usedCoreNum=4`，`vecCubeRatio=2`，`fdNum=2`：
- `maxVectorNum = 8`
- `emptyVectorNum = 8 - 2 = 6`
- `averageLoad = (512 + 7) / 8 = 64`

---

### 第3段：初始化核索引

```cpp
uint32_t curCoreIndex = 0;
```

`curCoreIndex` 是全局 Vec 核计数器，记录当前已分配到第几个 Vec 核。

---

### 第4段：遍历每个 FD 归约任务进行分配

```cpp
for (uint32_t i = 0; i < result.fdRes.fdNum; i++) {
```

对每个 FD 归约任务 i 独立计算其所需的 Vec 核数和每个核的 M 行范围。

---

#### 4.1 快速路径：无空闲核

```cpp
    if (emptyVectorNum == 0U) {
        fdRes.taskIdx[curCoreIndex] = i;
        fdRes.mStart[curCoreIndex] = 0U;
        fdRes.mLen[curCoreIndex] = fdRes.mSize[i];
        curCoreIndex++;
        continue;
    }
```

**逻辑**：当没有空闲 Vec 核时，该 FD 任务只用自带的 1 个 Vec 核处理全部 M 行。

**字段填充**：
- `taskIdx = i`：处理第 i 个 FD 任务
- `mStart = 0`：从 M 轴起点开始
- `mLen = mSize[i]`：处理全部行

---

#### 4.2 计算当前 FD 任务所需的 Vec 核数

```cpp
    uint32_t curFDVectorNum =
        std::max(fdRes.fdS2SplitNum[i] * fdRes.mSize[i] / averageLoad,
                 1UL); // 计算当前归约任务所用核数，向下取整，避免使用核数超出总核数且最少分一个核
```

**逻辑**：按负载比例计算该 FD 任务应该分配的 Vec 核数。

**公式**：`curFDVectorNum = max(任务负载 / 平均负载, 1)`

- `fdS2SplitNum[i] * mSize[i]`：该任务的负载
- `/averageLoad`：按比例分配（向下取整）
- `max(..., 1)`：保底至少 1 个核

**举例**：任务0 负载=384，averageLoad=64 → `384/64 = 6` 个核

---

#### 4.3 计算每个 Vec 核的 M 行数

```cpp
    uint32_t curAvgMSize = (fdRes.mSize[i] + curFDVectorNum - 1U) /
                           curFDVectorNum; // 计算当前归约任务每个核的行数，向上取整，避免行数为0
```

**逻辑**：将任务的 M 行均分到 `curFDVectorNum` 个核，向上取整。

**公式**：`curAvgMSize = ceil(mSize[i] / curFDVectorNum)`

**举例**：mSize[0]=128，curFDVectorNum=6 → `ceil(128/6) = 22` 行/核

---

#### 4.4 修正 Vec 核数

```cpp
    curFDVectorNum =
        (fdRes.mSize[i] + curAvgMSize - 1U) / curAvgMSize; // 重新计算正确的核数，避免因为向上取整导致有核为空
```

**逻辑**：用向上取整后的 `curAvgMSize` 反算实际需要的核数。

**为什么需要修正**：上一步向上取整可能导致核数减少。例如 mSize=128, curFDVectorNum=6 → curAvgMSize=22 → 实际需要 `ceil(128/22) = 6` 核，不变。但若 mSize=10, curFDVectorNum=3 → curAvgMSize=4 → 实际只需 `ceil(10/4) = 3` 核，也没变。但某些边界情况会减少：

- mSize=10, curFDVectorNum=4 → curAvgMSize=ceil(10/4)=3 → 实际 ceil(10/3)=4，不变
- mSize=100, curFDVectorNum=6 → curAvgMSize=17 → 实际 ceil(100/17)=6，不变

修正主要防止 `curAvgMSize` 向上取整后多出的核被浪费（分配了但无行可处理）。

---

#### 4.5 限制不超过空闲核数

```cpp
    curFDVectorNum = std::min(curFDVectorNum, emptyVectorNum + 1U); // 1: Fd任务自身带一个核
```

**逻辑**：实际分配的 Vec 核数不能超过"空闲核数 + 1"。

**`+1` 的原因**：每个 FD 任务自带 1 个 Vec 核（来自产生该任务的 Cube 核对应的 Vec 核），不占用 `emptyVectorNum`。所以上限是 `emptyVectorNum + 1`。

**举例**：curFDVectorNum=6，emptyVectorNum=6 → `min(6, 6+1)=6`，OK

---

#### 4.6 填充每个 Vec 核的任务信息

```cpp
    for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
        fdRes.taskIdx[curCoreIndex] = i;
        fdRes.mStart[curCoreIndex] = vid * curAvgMSize;
        fdRes.mLen[curCoreIndex] = (vid < curFDVectorNum - 1) ? curAvgMSize : (fdRes.mSize[i] - vid * curAvgMSize);
        curCoreIndex++;
    }
```

**逐行**：

1. **`taskIdx = i`**：当前 Vec 核处理第 i 个 FD 归约任务
2. **`mStart = vid * curAvgMSize`**：M 轴起点 = 核内序号 × 每核行数
3. **`mLen`**：
   - 非末核：`curAvgMSize`（固定行数）
   - 末核：`mSize[i] - vid * curAvgMSize`（剩余行数，可能小于 curAvgMSize）

**举例**：mSize=128，curFDVectorNum=6，curAvgMSize=22

| vid | mStart | mLen | 覆盖行 |
|-----|--------|------|--------|
| 0 | 0 | 22 | [0, 22) |
| 1 | 22 | 22 | [22, 44) |
| 2 | 44 | 22 | [44, 66) |
| 3 | 66 | 22 | [66, 88) |
| 4 | 88 | 22 | [88, 110) |
| 5 | 110 | 18 | [110, 128) ← 末核取剩余 |

---

#### 4.7 更新空闲核数

```cpp
    emptyVectorNum -= (curFDVectorNum - 1U); // 1: 空余核不包含FD自身的核，要-1
```

**逻辑**：扣减本次任务消耗的空闲核数。

**`-1` 的原因**：`curFDVectorNum` 中有 1 个是 FD 任务自带的核（不占用空闲额度），所以实际消耗的空闲核数是 `curFDVectorNum - 1`。

**举例**：curFDVectorNum=6 → 消耗 5 个空闲核 → emptyVectorNum: 6 → 1

---

### 第5段：记录总使用核数

```cpp
fdRes.fdUsedVecNum = curCoreIndex;
```

**逻辑**：所有 FD 任务分配完毕后，`curCoreIndex` 就是 FD 阶段实际使用的 Vec 核总数。

---

## 完整示例

### 输入

```
usedCoreNum = 4, vecCubeRatio = 2  → maxVectorNum = 8
fdNum = 2

任务0: fdS2SplitNum=3, mSize=128  → 负载 384
任务1: fdS2SplitNum=2, mSize=64   → 负载 128
totalFDLoad = 512
averageLoad = ceil(512/8) = 64
emptyVectorNum = 8 - 2 = 6
```

### 任务0 分配过程

```
1. curFDVectorNum = max(384/64, 1) = 6
2. curAvgMSize = ceil(128/6) = 22
3. curFDVectorNum = ceil(128/22) = 6   (不变)
4. curFDVectorNum = min(6, 6+1) = 6
5. 填充 6 个 Vec 核:
   vec0:  taskIdx=0, mStart=0,   mLen=22
   vec1:  taskIdx=0, mStart=22,  mLen=22
   vec2:  taskIdx=0, mStart=44,  mLen=22
   vec3:  taskIdx=0, mStart=66,  mLen=22
   vec4:  taskIdx=0, mStart=88,  mLen=22
   vec5:  taskIdx=0, mStart=110, mLen=18
6. emptyVectorNum = 6 - (6-1) = 1
```

### 任务1 分配过程

```
1. curFDVectorNum = max(128/64, 1) = 2
2. curAvgMSize = ceil(64/2) = 32
3. curFDVectorNum = ceil(64/32) = 2   (不变)
4. curFDVectorNum = min(2, 1+1) = 2
5. 填充 2 个 Vec 核:
   vec6:  taskIdx=1, mStart=0,  mLen=32
   vec7:  taskIdx=1, mStart=32, mLen=32
6. emptyVectorNum = 1 - (2-1) = 0
```

### 最终输出

```
fdUsedVecNum = 8

taskIdx = [0, 0, 0, 0, 0, 0, 1, 1]
mStart  = [0, 22, 44, 66, 88, 110, 0, 32]
mLen    = [22, 22, 22, 22, 22, 18, 32, 32]
```

8 个 Vec 核全部被使用，每个核知道要规约哪个 FD 任务的哪些 M 行。

---

## 关键设计要点

### 1. 每个 FD 任务自带 1 个 Vec 核

FD 归约任务由 Cube 核的 FA 阶段产生，该 Cube 核绑定的 Vec 核自然成为归约执行者之一。因此：
- `emptyVectorNum = maxVectorNum - fdNum`（扣除自带的）
- 分配上限 `emptyVectorNum + 1`（自带 + 空闲）
- 消耗 `curFDVectorNum - 1` 个空闲核

### 2. 按负载比例分配

`curFDVectorNum = 任务负载 / 平均负载`，保证大任务多分核、小任务少分核，整体负载均衡。

### 3. 三次修正保证合理性

| 修正 | 目的 |
|------|------|
| `max(..., 1)` | 至少 1 个核 |
| `ceil(mSize/curFDVectorNum)` 反算 | 避免向上取整导致空核 |
| `min(..., emptyVectorNum+1)` | 不超过空闲核上限 |

### 4. 末核取剩余

`mLen = (vid < curFDVectorNum-1) ? curAvgMSize : (mSize - vid*curAvgMSize)` 保证 M 行不重叠不遗漏，末核承担除法余数。
