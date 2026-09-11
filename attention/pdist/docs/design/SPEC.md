# 算子需求规范

## 机器合同

> 下列 JSON 是公式、公开接口、验收配置的唯一机器事实源。正文只解释语义与证据，不复制字段值。

```json machine-contract
{
  "schema_version": 1,
  "op_name": "pdist",
  "formula": "(1) For every row pair (i, j) with 0 <= i < j <= N-1, the output index is idx = i*(2*N-i-1)//2 + (j-i-1).\n(2) If 0 < p < +inf: y[idx] = ( sum_{k=0}^{M-1} abs(x[i,k] - x[j,k])**p )**(1/p).\n(3) If p == 0: y[idx] = sum_{k=0}^{M-1} (1 if x[i,k] != x[j,k] else 0).\n(4) If p == +inf: y[idx] = max_{k=0}^{M-1} abs(x[i,k] - x[j,k]).\n(5) y has length N*(N-1)//2, laid out in row-major upper-triangular order with the diagonal excluded.",
  "supported_dtypes": ["float32"],
  "inputs": [
    {
      "name": "x",
      "shape": ["N", "M"],
      "dtype": "float32",
      "value_range": [-65504.0, 65504.0]
    }
  ],
  "outputs": [
    {
      "name": "y",
      "shape": ["N*(N-1)//2"],
      "dtype": "float32",
      "value_range": [0.0, 65504.0]
    }
  ],
  "default_params": {
    "p": 2.0
  },
  "tolerance": {
    "atol": 0.001,
    "rtol": 0.001
  },
  "dynamic_axes_ranges": {
    "N": [2, 65535],
    "M": [1, 65535]
  },
  "shape_constraints": [
    "x is a 2-D tensor of shape [N, M] with N >= 2 (at least one row pair) and M >= 1",
    "p is a scalar attribute; p >= 0 or p == +inf is required and p < 0 is rejected as a parameter error",
    "y has length N*(N-1)//2; every row pair (i, j) with i < j maps to exactly one output index and vice versa",
    "N is bounded to 2 <= N <= 65535; at N = 65535 the output holds about 2.1e9 elements, near the fp32 8 GB memory bound",
    "float64 inputs are out of scope (P3): the NPU ecosystem has no fp64 support for this operator",
    "The params value \"inf\" in p0_cases denotes p = +infinity"
  ],
  "p0_cases": [
    {
      "name": "perf_p0_euclidean",
      "params": {"p": 2.0},
      "input_shapes": {"x": [4096, 256]},
      "output_shapes": {"y": [8386560]}
    },
    {
      "name": "perf_p0_fp16_large",
      "params": {"p": 2.0},
      "input_shapes": {"x": [8192, 512]},
      "output_shapes": {"y": [33550336]}
    },
    {
      "name": "func_p0_hamming_p0",
      "params": {"p": 0.0},
      "input_shapes": {"x": [1000, 200]},
      "output_shapes": {"y": [499500]}
    },
    {
      "name": "func_p0_manhattan_p1",
      "params": {"p": 1.0},
      "input_shapes": {"x": [1000, 200]},
      "output_shapes": {"y": [499500]}
    },
    {
      "name": "func_p0_euclidean_p2",
      "params": {"p": 2.0},
      "input_shapes": {"x": [1000, 200]},
      "output_shapes": {"y": [499500]}
    },
    {
      "name": "func_p0_fractional_p05",
      "params": {"p": 0.5},
      "input_shapes": {"x": [1000, 200]},
      "output_shapes": {"y": [499500]}
    },
    {
      "name": "func_p0_chebyshev_pinf",
      "params": {"p": "inf"},
      "input_shapes": {"x": [1000, 200]},
      "output_shapes": {"y": [499500]}
    },
    {
      "name": "func_p0_min_shape",
      "params": {"p": 2.0},
      "input_shapes": {"x": [2, 1]},
      "output_shapes": {"y": [1]}
    },
    {
      "name": "func_p0_large_m",
      "params": {"p": 2.0},
      "input_shapes": {"x": [64, 8192]},
      "output_shapes": {"y": [2016]}
    },
    {
      "name": "func_p0_partial_block",
      "params": {"p": 1.0},
      "input_shapes": {"x": [777, 300]},
      "output_shapes": {"y": [301476]}
    }
  ],
  "perf_target": null
}
```

## 语义说明

### 1. 功能与分类

`pdist` 计算二维输入张量 `x (N, M)` 中**每对行向量**之间的 p-范数距离，按行优先上三角顺序（`i < j`，不含对角线）输出为一维张量 `y (N*(N-1)/2,)`。典型用途：对比学习（Triplet Loss 的 pairwise distance 前置）、聚类、近邻检索。

- 算子分类：**reduction（pair-wise 距离规约，非 element-wise）**
- 复杂度判定：**复杂算子**——O(N²) 对枚举 + p 三分支语义 + 跨行规约（来源：分析报告 §1，✓高）
- 框架参考：`torch.nn.functional.pdist(input, p=2)`；等价于 `torch.cdist(x, x, p=p)` 的上三角（不含对角线）展平（来源：分析报告 §1，✓高）
- 本 SPEC 只覆盖**前向**；反向梯度（PdistGrad）明确不在范围内（P3）

### 2. 公式符号与依据

- 三分支公式（p-范数语义）见机器合同 `formula` 第 (2)–(4) 条：
  - `0 < p < +inf`（闵可夫斯基）：先逐维取 `|x_ik - x_jk|^p`，沿 M 求和，再取 `1/p` 次幂；
  - `p == 0`（汉明）：逐维不等元素计数（结果为非负整数语义的距离值）；
  - `p == +inf`（切比雪夫）：逐维最大绝对差。
- 输出索引映射：`idx = i*(2*N-i-1)/2 + (j-i-1)`，即上三角行优先、排除对角线；该式是 PyTorch 源码 `GetIJFromIndex` 的逆运算（来源：分析报告 §1.2，✓高）。
- 语义验证（PyTorch 2.10 本地实测，`x=[[1,2,3],[4,5,6],[1,1,1]]`，来源：分析报告 §1.2，✓高）：

| p | 输出 | 语义确认 |
|---|------|---------|
| 2 | [5.196, 2.236, 7.071] | 欧氏距离 ✓ |
| 0 | [3.0, 2.0, 3.0] | 汉明（不等元素计数）✓ |
| inf | [3.0, 2.0, 5.0] | 切比雪夫（逐维最大差）✓ |
| 1 | [9.0, 3.0, 12.0] | 曼哈顿 ✓ |
| 0.5 | [27.0, 5.828, 35.618] | 分数 p 有效（pow→sum→inv-pow）✓ |

### 3. 算法描述

数学过程（不规定 tile 或硬件实现；分块/并行属 Stage 3 设计）：

```
Algorithm: pdist forward
输入: x ∈ R^{N×M}, 标量属性 p ∈ [0, +inf]
输出: y ∈ R^{N(N-1)/2}

1. 对每个行对 (i, j), 0 <= i < j <= N-1（共 N(N-1)/2 对，枚举顺序即输出顺序）：
   1.1 d_k = |x[i,k] - x[j,k]|, k = 0..M-1        // 逐维绝对差
   1.2 分支（由 p 决定，与逐对无关，属全局常量分支）：
       - 0 < p < +inf: s = Σ_k d_k^p;  y_pair = s^(1/p)
       - p == 0:       s = Σ_k 𝟙(d_k ≠ 0);  y_pair = s
       - p == +inf:    y_pair = max_k d_k
   1.3 y[idx] = y_pair, 其中 idx = i*(2N-i-1)/2 + (j-i-1)
```

要点：M 维在数学上是一个整体规约（分块累加只改变求和顺序的浮点舍入，不改变语义）；各行对彼此独立，无跨对依赖。

### 4. 数据流说明

```
            输入 x [N, M]                          输出 y [N(N-1)/2]
      FLOAT32 / FLOAT16                      与 x 同 dtype
              │                                     ▲
              ▼                                     │
    ┌────────────────────┐   ┌──────────────┐   ┌─────────────┐
    │ 逐对取行 x_i, x_j  │ → │ d = |x_i-x_j│ → │ 分支规约      │
    │ (i<j 对枚举)       │   │  (逐维)      │   │ (沿 M)      │
    └────────────────────┘   └──────────────┘   └──────┬──────┘
                                                     ▼
                                              0<p<inf: (·)^(1/p)
                                              p=0/p=inf: 直写
```

- 动态轴：N（行数）与 M（维度）均为动态；输出长度由 N 派生。
- 输出排列：`[dist(0,1), dist(0,2), …, dist(0,N-1), dist(1,2), …]`（上三角行优先）。

### 5. 接口语义

- `x`：rank-2 张量 `[N, M]`，公开输入。支持的 IO dtype：**float32 与 float16**（同 dtype 进、同 dtype 出）。机器合同以 float32 为规范组合（精度基准 dtype）；float16 是同一接口签名、同一公式下的第二 IO dtype，其表达见"自动决策"第 1 条。float64 不支持（P3 排除，NPU 生态无 fp64）。
- `p`：标量属性（非 tensor），默认 `2.0`；取值 `p >= 0` 或 `p == +inf`，`p < 0` 为参数错误（对应 Ascend 生态的 `ACLNN_ERR_PARAM_INVALID` 行为；来源：分析报告 §6，✓高）。
- `y`：rank-1 张量，长度 `N*(N-1)/2`，dtype 与 `x` 一致；每个元素对应唯一行对，写入位置由上三角索引映射唯一确定（覆盖写，无累加语义）。

### 6. 功能与可选参数依据

特性拆解（来源：分析报告 §4，✓高，除非另行标注）：

| 特性 | 是否需要 | 置信度 | 优先级 | 说明 |
|------|----------|--------|--------|------|
| p 三分支（0 / (0,∞) / ∞） | ✓ | ✓高 | **P0** | 三分支语义即公式本体，全部进入首个版本 |
| 上三角索引映射 | ✓ | ✓高 | **P0** | 线性 idx ↔ (i,j) 互推，不物化 N×N 矩阵 |
| 多核切分（pair 域并行） | ✓ | ✓高 | **P0** | O(N²) 对独立，天然可并行（实现细节由 Stage 3 设计） |
| M 维分块规约 | ✓ | ✓高 | **P0** | M 超片上内存时部分和累加，语义不变 |
| 确定性实现 | ✓ | ✓高 | **P0** | 逐对独立规约、静态切分天然确定 |
| fp16 IO（中间 fp32 累加） | ✓ | ✓高 | **P1** | 语义见 §7；性能 P0 case `perf_p0_fp16_large` 以 fp16 运行 |
| 数值稳定性（大 p 上溢保护） | — | ⚠中 | P2 | 与 PyTorch 行为对齐（PyTorch 亦不做 log-sum-exp 保护），可暂缓 |
| fp64 | ✗ | ⚠中 | P3 | NPU 生态不支持，成本高收益低 |
| 反向梯度（PdistGrad） | ✗ | ⚠中 | P3 | 仅前向；训练场景才需要 |

可选参数 `p` 的四项语义分析：

- 公式中的位置：分支选择子 + 幂指数（`d^p` 与 `s^(1/p)`）；
- 计算作用：决定距离度量（闵可夫斯基/汉明/切比雪夫），p=2 为欧氏距离；
- 依赖/互斥：无其他参数，无互斥关系；`p >= 0` 或 `+inf`；
- 省略时默认语义：`p = 2.0`（欧氏距离，最常用）。

### 7. 精度语义

- **中间规约一律以 fp32 累加**：fp16 输入时仅 IO 为 fp16，`diff/pow/累加/后处理` 在 kernel 内以 fp32 进行（来源：分析报告 §6 与 §4 fp16 精度行，✓高）。
- NaN/Inf 传播：输入含 NaN 时距离逐对传播 NaN（与 PyTorch 一致，正常计算即可）；`p=+inf` 且逐维差含 inf 时输出 inf（来源：分析报告 §6，✓高）。
- 大 p 上溢：`|diff|^p` 在大 p 时可能上溢；与 PyTorch（CUDA/CPU）行为一致不做保护，P2 不阻塞验收（来源：分析报告 §4，⚠中）。
- 验收容差：atol/rtol 见机器合同 `tolerance`，fp32 与 fp16 case 同标准（fp16 的误差已被 fp32 中间累加控制；来源：分析报告 §6 建议，✓高）。

### 8. 动态 Shape 与约束依据

- `N ∈ [2, 65535]`：上限 65535 来自已有 Ascend 实现的显式校验（`MAX_ROWS=65535`，输出元素数以 64 位计数防溢出）；N=65535 时输出 ≈ 2.1×10⁹ 元素，已接近 fp32 8 GB 内存上限，故该上界是内存意义上的合理约束（来源：分析报告 §7，✓高）。N=2 为最小值（恰好一个行对）。
- `M ∈ [1, 65535]`：数学上 M 无硬上限（分块规约天然支持任意大 M，来源：分析报告 §7，✓高）；机器合同需要有限闭区间，取 65535 作为工程上界（覆盖全部 P0 case 的最大 M=8192），见"自动决策"第 3 条。
- 每对输出元素由两行共同决定（跨行规约），输出长度与输入元素数无 elementwise 对应关系。

### 9. 边界条件处理

| 边界 | 行为 | 依据 |
|------|------|------|
| `p = 0` 且两行完全相同 | 输出 0（𝟙 求和为 0） | 分析报告 §6，✓高 |
| `p < 0` | 参数错误，不进入计算 | 分析报告 §6，✓高 |
| `M = 1` | 距离退化为单维 `|x_i0 - x_j0|`（p 分支同理） | 公式自然推论 |
| `N = 2` | 输出长度 1，仅 `dist(0,1)` | 公式自然推论 |
| 输入含 NaN | 含 NaN 的行对其距离为 NaN，逐对传播 | 分析报告 §6，✓高 |
| `p = +inf` 且 diff 含 inf | 输出 inf | 分析报告 §6，✓高 |
| 大 p / 大幅值上溢 | 不做保护，按 IEEE 上溢为 inf，与 PyTorch 对齐（P2） | 分析报告 §4，⚠中 |
| fp16 输出超 65504 | 按 IEEE 上溢为 inf（P2 不保护） | IEEE-754 语义 |

### 10. P0 与性能目标依据

P0 典型配置（来源：分析报告 §9，✓高；dtype 列注明各 case 的运行 IO dtype）：

| 配置名称（对应 p0_cases 名） | 类型 | p | Shape | 输出 Shape | dtype | 说明 |
|------------------------------|------|---|-------|-----------|-------|------|
| `perf_p0_euclidean` | 性能 | 2 | [4096, 256] | [8386560,] | float32 | 核心性能场景（欧氏距离，最常用） |
| `perf_p0_fp16_large` | 性能 | 2 | [8192, 512] | [33550336,] | **float16** | fp16 大规模（对比学习典型） |
| `func_p0_hamming_p0` … `func_p0_chebyshev_pinf` | 功能 | 0 / 1 / 2 / 0.5 / inf | [1000, 200] | [499500,] | float32 | p 全分支语义验证（5 个 case） |
| `func_p0_min_shape` | 功能 | 2 | [2, 1] | [1,] | float32 | 最小 shape（N=2, M=1） |
| `func_p0_large_m` | 功能 | 2 | [64, 8192] | [2016,] | float32 | M 超片上内存，强制分块规约 |
| `func_p0_partial_block` | 功能 | 1 | [777, 300] | [301476,] | float32 | 非整块尾部 + 多核不均 |

> 注：分析报告 §9 表中 `性能_P0_fp16` 与 `功能_P0_部分块` 的输出长度为笔误（33,558,528 / 301,086）；按输出长度公式 `N*(N-1)/2` 修正为 33,550,336 / 301,476（`validate_spec.py` 机器校验一致），SPEC 以公式为准。

性能目标：用户未给出可复算数值目标，`perf_target` 为 `null`；Stage 5 将使用逐 P0 case 的默认目标 `golden_reference_ratio >= 1.0`（见"自动决策"第 5 条）。

### 11. 参考与来源

| 项 | 位置 | 置信度 |
|----|------|--------|
| 需求主输入（本 SPEC 的主要事实源） | `/data/z00954450/zzz/pdist_analysis_report.md` | ✓高（PyTorch 2.10 本地实测 + 源码实读） |
| PyTorch 参考实现 | `torch.nn.functional.pdist`（本机 2.10.0 实测）；CUDA kernel `aten/src/ATen/native/cuda/Pdist.cu`（每 pair 一 block） | ✓高 |
| Ascend 同语义参考实现（C++ 算子包，非本任务实现路径） | `/data/z00954450/zzz/code_repo/ops-math/experimental/math/pdist/`（op_host/op_kernel/op_api/tests 全套，ST 97/97；**仅注册 910b/910_93，未适配 950**） | ✓高 |
| 相似算子 | `cdist`（两输入泛化）、`PairwiseDistance`（nn.Module 封装） | — |

生态缺口备注（背景信息，非本任务阻断项）：上述 C++ 算子包未适配 950（无 `AddConfig("ascend950")`）且未进 CANN 正式包导致 op-plugin 无 pdist 映射。本任务的开发路径是 **PyPTO-Pro DSL**（`pypto_pro.language`），在 950PR（A5）上直接实现，不依赖该 C++ 包的可运行性；其语义、tiling 思路与 ST 用例集可作参考。

### 12. 自动决策

无人值守模式下，以下非阻塞字段按可追溯默认裁定（来源与影响均已披露）：

1. **machine-contract 的 dtype 组合**：合同 `inputs/outputs` 的 dtype 字段按校验器约束只能冻结一组 canonical dtype 组合，本合同取 **float32**（PyTorch CPU 原生 dtype、精度基准、功能 P0 全部 case 的运行 dtype）。**float16 是同一接口签名、同一公式下的第二 IO dtype**（输入 fp16 → 中间 fp32 → 输出 fp16），完整表达于：正文 §5/§7、kernel 契约补充的 cast 边界链、KB_SELECTION 的 `properties.dtypes`，以及 P0 case `perf_p0_fp16_large`（dtype=float16，见 §10 表）。不拆分独立 class/SPEC 的理由：pdist 输入签名单一（x + 标量 p），fp32/fp16 两组合同之间不存在 index/mask/量化辅助 tensor 等 dtype 会被静默丢失的对象（拆分规则的保护目标不存在）；且下游 Stage 2–4（`{op}_golden.py` / `DESIGN.md` / `test_{op}.py`）为单文件链，无多 class 消费协议。影响：Stage 2 golden 需按 §10 表的 dtype 列为 fp16 case 单独覆盖（fp16 golden 可由 fp32 参考 cast 派生，见 kernel 契约补充）。
2. **`x.value_range = [-65504, 65504]`**：取 fp16 最大可表示有限值 ±65504 为合同元素值域——同时覆盖 fp16 dtype 的全有限值域与 fp32 case 的典型测试分布（golden 输入为小幅度随机分布）；限制幅值亦缓解大 p 时 `|diff|^p` 的上溢面。来源：IEEE-754 half 精度范围 + 分析报告 §4/§6；影响：数值安全分析（Stage 3）以该区间为强制输入域。
3. **`M` 上限取 65535**：数学上 M 无硬上限（分析报告 §7，✓高），机器合同要求有限闭区间，取 65535 为工程上界（覆盖最大 P0 case 的 M=8192 约八倍余量，与 N 上限对称）。影响：Stage 3 的分块设计须在 M∈[1,65535] 全区间正确。
4. **`p=+inf` 在 p0_cases 中的表示**：JSON 不允许非有限数值（校验器拒绝 `Infinity`），case `func_p0_chebyshev_pinf` 的 params 使用字符串 `"inf"` 表示 `p = +infinity`（shape_constraints 已声明该约定）。影响：Stage 2 golden 构造输入时须将 `"inf"` 解释为 `float('inf')`。
5. **`perf_target = null`**：用户未给出数值性能目标（分析报告仅给出典型配置，无 speedup/时长指标）。Stage 5 使用逐 P0 case 默认目标 `golden_reference_ratio >= 1.0`（系统默认，非用户要求）。
6. **`y.value_range = [0, 65504]`**：距离非负；上界取 fp16 输出 dtype 的最大有限值。全部 P0 case 的期望输出均落在该区间（最大量级 case：M=8192、N(0,1) 输入的 p=1 距离 ~10⁴）。超出该区间的行为（fp16 上溢为 inf）按 §9 与 PyTorch 对齐，不属验收范围。

## kernel 契约补充

| 字段 | 裁定（Stage 1） |
|------|-----------------|
| 辅助张量语义 | 无辅助 tensor。`p` 是标量属性（host 侧编译期常量，非 tensor 输入）；无 workspace/中间公开量。实现位置（是否需要内部临时 UB buffer）由 Stage 3 设计 |
| cast 边界链 | 输入 IO dtype fp32 或 fp16 →（fp16 输入时在 kernel 入口 cast 到 fp32）→ diff / abs / pow / 沿 M 规约累加 / 1/p 幂**全部 fp32** → 输出段（fp16 case 在出口单点 cast 回 fp16；fp32 case 直写）。golden 对照：fp16 case 可由 fp32 参考（同输入 cast 上取）在 host 侧 cast 派生期望值，容差按机器合同 tolerance |
| 累加/写回语义 | 每个行对的规约**彼此独立、一次性覆盖写**：输出位置 `idx = i*(2N-i-1)/2 + (j-i-1)` 唯一确定，无跨块累加到 GM、无原子累加、无读改写。M 维分块的部分和只存在于 kernel 内部累加变量（数学上仅改变求和顺序） |
| 目标设备 | **A5（Ascend 950PR）已确认**（运行环境：Ascend 950PR，CANN 9.2.0，8 卡，`TILE_FWK_DEVICE_ID=0`）——非默认假设，是运行时实测环境；KB 路由按 A5 触发 `constraints/arch-a5.md` |
| topology / tile / 同步 | **由 Stage 3 设计**，Stage 1 不猜测。Stage 1 仅记录：逐对独立（无跨对依赖）→ 无 kernel 间同步语义要求；多核切分与 M 分块的实现选择（含尾块处理）交 Stage 3 |

---
*生成时间: 2026-09-01*
*确认状态: 无人值守模式——基于用户提供的高置信度分析报告（✓高，PyTorch 2.10 本地实测 + 源码实读）裁定；非阻塞默认值已全部披露于 §12 自动决策*
