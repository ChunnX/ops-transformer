# IFA Preload DD 算子项目 — 面试考点与回答（修正版）

> **对应PDF**: IFA Preload DD 算子项目 — 面试考点与回答.pdf
> **Token**: UGPPb3PzkoTeskxBN0icPLZ1nFb
> **原始文档**: MxKUdaBGYoLq2Ax4TDOcejqtn4d (飞书DOCX)

---

## 一、架构设计类（Q1-Q3）

### Q1: 为什么要采用 Vector + Cube 异构流水线架构？

**考点**：对 DaVinci 架构双引擎的理解。

**回答要点**：

1. **计算特性异构**：矩阵乘适合Cube的16×16脉动阵列；Softmax等逐元素操作适合Vector

2. **硬件基础**：910B/C Cube和Vector共享L1/UB/GM，通过CrossCore Flag同步（延迟~几十cycle）

3. **纯Cube问题**：Softmax需要反复写回GM再读入L1，效率极低

4. **纯Vector问题**：矩阵乘效率远低于Cube（吞吐约1/16）

5. **流水收益**：四级流水隐藏延迟，总耗时≈max(ΣVector, ΣCube)

---

### Q2: CrossCore Flag 同步机制的原理是什么？

**考点**：硬件同步原语选型。

**回答要点**：

1. CrossCore Flag是硬件支持的轻量级点对点同步原语

2. `CrossCoreSetFlag<SYNC_MODE2>`双重语义：pipe_flush + 跨Core通知

3. 四个Flag握手链：**V0→C1→V1→C2→V2**

4. vs Barrier：Flag延迟更低，适合流水线

---

### Q3: TASK_NUM 与 PRE_LOAD_NUM_DD 有什么区别？

**考点**：流水深度与资源占用权衡【已修正】。

**【重要修正】**：原始文档混淆了两个参数！

| 参数 | 值 | 含义 |
|------|-----|------|
| TASK_NUM | 2 | 任务并行深度 |
| PRE_LOAD_NUM_DD | 4 | Buffer预加载槽数 |
| EXTRA_NUM | 6 | 状态数组大小 |

**代码证据**：
```cpp
// incre_flash_attention_preload_dd.h:35-39
#define PRE_LOAD_NUM_DD 4
#define EXTRA_NUM 6
#define TASK_NUM 2
```

---

## 二、Tiling策略类（Q4-Q6）

### Q4: sInnerSize分档逻辑？

| 分档 | 场景 |
|------|------|
| 8192 | 非GQA+非伪量化 |
| 2048 | GQA+伪量化+CUBE_VIEW_MM |
| 1024 | GQA+伪量化+DD模式 |

### Q5: SplitBN负载均衡？

- 均衡分配：seqLen相同时均分任务
- 稀疏分配：按estimatedLoad贪心分配

### Q6: Workspace为什么~200MB？【修正】

**原错误**：计算用dbWorkspaceRatio=6
**修正**：PRE_LOAD_NUM_DD=4

| 组分 | 公式 |
|------|------|
| mm1ResGm | 4×24×mmResUbSize×2B |
| vec1ResGm | 4×24×mmResUbSize×1B |

---

## 三、访存优化类（Q7-Q9）

### Q7: ND vs NZ布局？

- ND：行主序，通用但Cube不亲和
- NZ：16×16 Block列主序，与MAC阵列匹配
- 转换代价：一次DataCopy，净收益30-50%

### Q8: Page Attention三段式？

- head：首个block剩余
- mid：完整block逐个
- tail：最后不完整block

### Q9: K/V L2 Cache关闭？

- Decode场景复用率为零
- LRU会驱逐高频复用数据
- `SetL2CacheHint(CACHE_MODE_DISABLE)`绕过L2

---

## 四、精度与量化类（Q10-Q12）

### Q10: MSD双段反量化精度？【修正】

**原表述**："从7bit提升到14bit"

**修正**：
| 方式 | 理论精度 | 实际精度 |
|------|----------|----------|
| INT8直接 | 8bit | 8bit |
| MSD双段 | 16bit | ~14bit |

为什么254而非256：避免溢出

### Q11: BF16用CAST_RINT？

- CAST_ROUND：+0.5 LSB偏差
- CAST_RINT：IEEE无偏舍入
- 多轮累加必须用RINT

### Q12: Q预量化只做一次？

- Q矩阵在S方向循环中固定不变
- 量化一次后重复使用

---

## 五、Softmax优化类（Q13-Q14）

### Q13: Online Safe Softmax稳定性？

- `exp(m_old - m_new) ≤ 1`不会溢出
- 流式计算，无需全量数据

### Q14: Brcb节省UB空间？【修正】

**原表述**："节省~21KB"

**修正**：约18KB（SOFTMAX_MAX_BUF_BYTES=512B）

---

## 六、性能与工程类（Q15-Q18）

### Q15: 如何验证优化效果？

- CANN msprof + PMU计数器
- Roofline模型判断bound
- A/B消融实验

### Q16: 新架构适配？

- L1/L0容量变化 → Buffer重规划
- MAC阵列尺寸 → NZ调整
- FP8原生 → 替代MSD
- if constexpr保持一套代码

### Q17: 最难技术点？

- Cube/Vector负载不均衡 → sInnerSize+DD预加载
- Page Attention NZ适配 → 三段式搬运

### Q18: 未来发展方向？

- FlashAttention V3 + FP8
- MLA压缩KV
- 稀疏Attention

---

## 附录：错误修正汇总

| 错误点 | 原表述 | 修正 |
|--------|--------|------|
| TASK_NUM混淆 | 混淆两者 | TASK_NUM=2, PRE_LOAD_NUM_DD=4, EXTRA_NUM=6 |
| workspace计算 | dbWorkspaceRatio=6 | PRE_LOAD_NUM_DD=4 |
| Brcb节省空间 | ~21KB | ~18KB |
| MSD精度 | 7→14bit | 理论16bit，实际14bit |

---

*面试考点修正版，确保技术分析准确。*