# IFA Preload DD 跨平台面试分析

> **对应PDF**: 04_IFA_Preload_DD_CrossPlatform_Interview.pdf
> **Token**: M6JSbPCGHopaPpxCdpJccBBcnqb

---

## 一、架构迁移能力考点

### Q1: 从910B迁移到910C需要哪些改动？

**回答要点**:
- 910B和910C架构基本相同（24核AICore，1MB L1）
- 主要差异在FP8原生支持
- 可直接复用现有代码

### Q2: 从910B迁移到950需要哪些适配？

**关键适配点**:

| 特性变化 | 适配方案 |
|----------|----------|
| L1容量变化 | 重新规划Buffer分配 |
| MAC阵列尺寸 | 调整NZ布局Block大小 |
| 新同步原语 | 替换CrossCore Flag |
| FP8原生支持 | 可替代MSD双段 |

**工程方案**:
```cpp
if constexpr (NpuArch == ARCH_910B) {
    // MSD双段反量化
} else if constexpr (NpuArch == ARCH_950) {
    // FP8原生MAC
}
```

---

## 二、跨架构通用优化原则

### 2.1 保持不变的优化

| 优化措施 | 跨架构通用性 |
|----------|--------------|
| Vector+Cube异构 | 通用 |
| CrossCore流水握手 | 通用（可能新原语） |
| Online Safe Softmax | 通用 |
| Tiling分档策略 | 需重新调优 |
| NZ布局优化 | 需适配Block尺寸 |

### 2.2 需要重新调优的参数

| 参数 | 调优依据 |
|------|----------|
| sInnerSize | L1容量上限 |
| PRE_LOAD_NUM | L1/L2带宽 |
| TASK_NUM | 流水深度vs workspace |

---

## 三、不同场景的架构选型

### 3.1 推理场景 vs 训练场景

| 场景 | 关注点 | 架构需求 |
|------|--------|----------|
| 推理Decode | 极低时延(batch=1) | 高流水效率 |
| 训练Prefill | 高吞吐(大batch) | 高算力效率 |

### 3.2 GQA vs MQA vs MHA

| Attention类型 | 特点 | 优化重点 |
|---------------|------|----------|
| MHA | 所有头独立 | 标准流水 |
| GQA | Query分组共享KV | Vector负载重 |
| MQA | 所有Query共享KV | KV Cache优化 |

---

## 四、未来架构演进趋势

### 4.1 FP8原生支持的影响

| 对比 | MSD双段 | FP8原生 |
|------|---------|---------|
| 精度 | ~14bit | ~7-8bit |
| 延迟 | 2轮Fixpipe | 1轮 |
| 适用 | 无FP8时 | 有FP8时 |

### 4.2 MLA架构适配

**MLA特点**: KV压缩到低维潜在空间
**优化点**: KV Cache带宽大幅降低

---

## 五、错误修正汇总

| 原错误 | 修正 |
|--------|------|
| PRE_LOAD_NUM混淆 | PRE_LOAD_NUM_DD=4, TASK_NUM=2 |
| workspace计算 | dbWorkspaceRatio=4 |

---

## 六、面试回答模板

**"如何适配新架构"模板**:
1. 存储层次重新分析（L1/L0/UB容量）
2. MAC阵列尺寸适配（NZ Block大小）
3. 新指令利用（FP8 MAC替代MSD）
4. 工程方案：if constexpr保持一套代码
5. 性能验证：msprof + Roofline

---

*跨平台面试分析，修正参数定义错误。*