# IFA Preload vs Preload DD 对比分析

> **对应PDF**: 02_IFA_Preload_vs_Preload_DD_Comparison.pdf
> **Token**: VTI0bayckotnKRxYKWScJKTrnoe

---

## 一、两种模式概述

| 特性 | Preload模式 | Preload DD模式 |
|------|-------------|----------------|
| 预加载深度 | PRE_LOAD_NUM=2 | PRE_LOAD_NUM_DD=4 |
| Workspace槽数 | 2 | 4 |
| 双缓冲 | 无 | 有（DD=Double Buffer） |
| 适用场景 | 简单Decode | GQA+伪量化复杂场景 |

---

## 二、核心差异分析

### 2.1 预加载机制差异

**Preload模式**:
```cpp
#define PRE_LOAD_NUM 2
// 2路轮转，简单流水
```

**Preload DD模式**:
```cpp
#define PRE_LOAD_NUM_DD 4
// 4路双缓冲轮转，深度流水
```

### 2.2 TASK_NUM对比

**【重要修正】**: 原始文档混淆了TASK_NUM和PRE_LOAD_NUM_DD

| 参数 | Preload | Preload DD |
|------|---------|------------|
| TASK_NUM | 1 | 2 |
| PRE_LOAD_NUM | 2 | 4 (PRE_LOAD_NUM_DD) |
| EXTRA_NUM | 4 | 6 |

### 2.3 流水深度对比

| 流水级数 | Preload | Preload DD |
|----------|---------|------------|
| 预加载槽数 | 2 | 4 |
| 任务并行度 | 1 | 2 |
| 流水覆盖 | 2轮 | 4轮 |

---

## 三、性能对比

### 3.1 Cube/Vector负载均衡

| 场景 | Preload问题 | Preload DD解决 |
|------|-------------|----------------|
| GQA场景 | Vector计算重，Cube等待 | 双缓冲掩盖延迟 |
| 伪量化 | 反量化开销大 | 预处理提前完成 |
| 长序列 | 流水空泡多 | 4路轮转填充 |

### 3.2 Workspace对比

| 组分 | Preload | Preload DD |
|------|---------|------------|
| mm1ResGm | 2×24×size | 4×24×size |
| vec1ResGm | 2×24×size | 4×24×size |
| 总大小 | ~100MB | ~200MB |

**代价**: DD模式workspace翻倍，换取流水效率

---

## 四、代码实现差异

### 4.1 Buffer轮转索引

**Preload**:
```cpp
uint64_t offset = (info.loop % PRE_LOAD_NUM) * size;
```

**Preload DD**:
```cpp
uint64_t offset = (info.loop % PRE_LOAD_NUM_DD) * size;
```

### 4.2 Flag同步差异

**Preload**: 简单2阶段握手
**Preload DD**: 4阶段深度握手

---

## 五、选型建议

| 场景 | 推荐 |
|------|------|
| 非GQA+非伪量化 | Preload（workspace小） |
| GQA+伪量化 | Preload DD（流水效率） |
| 长序列Decode | Preload DD |
| 短序列 | Preload |

---

## 六、错误修正汇总

| 原错误 | 修正 |
|--------|------|
| dbWorkspaceRatio=6 | PRE_LOAD_NUM_DD=4 |
| TASK_NUM混淆 | TASK_NUM=2, PRE_LOAD_NUM_DD=4 |

---

*本文档对比两种模式的核心差异，修正了参数定义错误。*