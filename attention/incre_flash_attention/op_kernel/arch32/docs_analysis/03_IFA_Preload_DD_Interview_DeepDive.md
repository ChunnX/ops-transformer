# IFA Preload DD 面试深度分析

> **对应PDF**: 03_IFA_Preload_DD_Interview_DeepDive.pdf
> **Token**: RUEebevzdo3fjTxN13wc69P7nqb

---

## 一、架构设计深度考点

### Q1: Vector+Cube异构流水线的必要性

**回答要点**:
1. Attention包含两类算子：矩阵乘(90%计算量)+逐元素操作
2. Cube: 16×16脉动阵列，256 MAC/cycle，专门优化矩阵乘
3. Vector: 逐元素操作效率高，但矩阵乘吞吐仅~16 MAC/cycle
4. 异构流水：四级握手(V0→C1→V1→C2→V2)，隐藏延迟

---

### Q2: CrossCore Flag vs Barrier

**核心差异**:
| 同步方式 | 延迟 | 适用场景 |
|----------|------|----------|
| Barrier | 高 | 全核同步(SyncAll) |
| CrossCore Flag | 低(~几十cycle) | 流水线握手 |

**SYNC_MODE2语义**:
- pipe_flush: 保证数据写回
- 跨Core通知: 点对点信号

---

### Q3: TASK_NUM与PRE_LOAD_NUM_DD区别【必考】

**【重要修正】**: 原面试题混淆了两个参数

| 参数 | 值 | 含义 |
|------|-----|------|
| TASK_NUM | 2 | 任务并行深度 |
| PRE_LOAD_NUM_DD | 4 | Buffer预加载槽数 |
| EXTRA_NUM | 6 | 状态数组大小 |

**为什么TASK_NUM=2最优**:
- 更深: workspace膨胀
- 更浅: 流水空泡

---

## 二、存储层次考点

### Q4: L1 Buffer规划原理

**分配策略**:
- Q: 64KB×1 (固定，单次加载)
- K/P: 64KB×3 (轮转)
- V: 64KB×4 (与PRE_LOAD_NUM_DD对应)

**约束**: 总计512KB < 1MB L1容量

### Q5: UB节省技术 - Brcb指令

**【修正】**: 节省空间约18KB（非21KB）

```cpp
Brcb(tmp, src, (mSizeVector+7)/8, {1, 8});
// 从寄存器广播，无需复制
```

---

## 三、精度与量化考点

### Q6: MSD双段反量化

**精度分析【修正】**:
| 方式 | 量化级别 | 理论精度 | 实际精度 |
|------|----------|----------|----------|
| INT8直接 | 256 | 8bit | 8bit |
| MSD双段 | 65024 | 16bit | ~14bit |

**为什么254而非256**: 避免第二段溢出

**硬件配合**: Fixpipe原子加，零额外开销

---

### Q7: CAST_RINT vs CAST_ROUND

| 舍入模式 | 偏差 | 适用 |
|----------|------|------|
| CAST_ROUND | +0.5 LSB正偏差 | 单次转换 |
| CAST_RINT | 无偏 | 多轮累加 |

---

## 四、Tiling策略考点

### Q8: sInnerSize分档逻辑

| 分档 | 场景 | 原因 |
|------|------|------|
| 8192 | 非GQA+非伪量化 | Vector负载轻 |
| 2048 | GQA+伪量化 | 平衡 |
| 1024 | GQA+伪量化+DD | Vector计算重 |

### Q9: Workspace大小【修正】

**原错误**: dbWorkspaceRatio=6
**修正**: PRE_LOAD_NUM_DD=4

---

## 五、性能工程考点

### Q10: 如何验证优化效果

1. CANN msprof + PMU计数器
2. Cube利用率 vs Vector利用率
3. Roofline模型判断bound
4. A/B消融实验

### Q11: 新架构适配思路

1. L1/L0容量变化 → Buffer重规划
2. MAC阵列尺寸 → NZ布局调整
3. 新指令（FP8 MAC） → 替代MSD
4. if constexpr保持一套代码

---

## 六、错误修正汇总表

| 原错误 | 修正内容 | 代码位置 |
|--------|----------|----------|
| TASK_NUM混淆 | TASK_NUM=2, PRE_LOAD_NUM_DD=4 | :35-39 |
| dbWorkspaceRatio=6 | PRE_LOAD_NUM_DD=4 | :328 |
| Brcb节省21KB | 约18KB | :49-54 |
| MSD精度7→14bit | 理论16bit，实际14bit | :256-259 |

---

*本面试深度分析修正了原始文档的技术错误。*