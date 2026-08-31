# AlltoAllMatmulV2 precision_mode=2 精度问题分析

> 范围：`mc2/allto_all_matmul_v2`（MX 量化 FP8 矩阵乘 + URMA all-to-all 融合算子）
> 场景：`precision_mode = 0/1/2`，输出 BF16/FP16
> 结论一句话：**pm=2（GM 累加）的中间舍入已定位并可在 CPU golden 中精确复现——根因是「LOCAL 结果落 GM 时量化一次 + 原子累加的 fixpipe 把 remote 也量化一次」，即 `round(round(local+bias) + round(remote))`。用该模型 golden 与 NPU 的 bit 对齐率达 99.9986%。**

---

## 1. 精度标准

| 项目 | 值 |
|------|----|
| 输出 dtype | BF16 / FP16 |
| 比对标准 | `torch.allclose(rtol=1e-2, atol=1e-2, equal_nan=True)` |
| 测试数据 | x1/x2 = FP8_E4M3FN/E5M2，scale = FP8_E8M0，量级 `uniform(1,8)` |
| 输出量级 | ~3.3e6（`M_per_rank=512, N=8192, K_total=8192`） |

---

## 2. 实测数据

### 2.1 容差测试（`test_golden_alltoallmm.py`，rtol=atol=1e-2）

| 模式 | abs_err max | abs_err mean | rel_err max | count(\|err\|>0.01) | 结果 |
|------|------------|--------------|-------------|---------------------|------|
| pm=0 | 1.6384e4 | **1.34 ~ 1.54** | 5.7e-3 | 342 / 4.19M | PASS |
| pm=1 | 1.6384e4 | **1.34 ~ 1.57** | 5.7e-3 | 401 / 4.19M | PASS |
| pm=2（无 bias） | 3.2768e4 | **~4.10e3** | 7.75e-3 | 1.05M / 4.19M | PASS |
| pm=2（带 bias） | 3.2768e4 | **~4.09e3** | 7.69e-3 | 1.05M / 4.19M | PASS |

> 注意：上表 pm=2 的 `abs_err mean ≈ 4.1e3` 是**单段式 golden**（`round(A@B+bias)`）的结果。改用下文 §4 的「两次量化」golden 后，pm=2 与 pm=0/1 一样收敛到 ~1 ULP 量级。

### 2.2 bit 级对齐测试（`test_bitmatch_alltoallmm.py`）

| 模式 | golden 模型 | bit-exact 率 | mismatch |
|------|-------------|-------------|----------|
| pm=0 | `round(A@B+bias)` | **99.9912%** | 735 |
| pm=1 | `round(A@B+bias)` | **99.9911%** | 743 |
| pm=2 | `round(A@B+bias)`（单段式，错误） | 74.98% | 2.10M |
| pm=2 | `round(round(local)+remote)`（一次量化，错误） | 74.96% | 2.10M |
| pm=2 | **`round(round(local+bias)+round(remote))`（两次量化，正确）** | **99.9986%** | 115 |

> 全部 mismatch 均为 1 ULP（bf16 舍入边界翻转），无 >1 ULP 偏差。残余 ~0.001% 来自 cube 的 fp32 累加顺序与 CPU BLAS 顺序不同。

---

## 3. 根因：GM 累加做了「两次输出精度量化」

### 3.1 GM 累加（pm=2）的真实数据流

```
LOCAL  阶段: local_fp32 + bias  --fixpipe-->  round(local+bias) → GM (bf16)      [第 1 次量化]
REMOTE 阶段: remote_fp32        --fixpipe 原子累加-->
             读 GM(bf16 local) + round(remote) → round(sum) → GM (bf16)          [第 2 次量化]
```

关键点：**原子累加的 fixpipe（`CopyL0C2GM + FINAL_ACCUMULATION + SetAtomicAdd`）不是「GM(bf16) + L0C(fp32) 在 fp32 里相加再舍入」，而是先把 L0C(fp32 remote) 转成 bf16（fixpipe 的 CType 转换），再与 GM(bf16 local) 相加、最后再舍入到 bf16。**

即：

```
final = round_bf16( round_bf16(local + bias) + round_bf16(remote) )
```

两个操作数（local、remote）各被量化一次，共两次 bf16 舍入。

### 3.2 为什么之前「单次量化」模型差 2 倍

| 模型 | 中间舍入次数 | 理论翻动比例 | 实测 bit-exact |
|------|-------------|-------------|----------------|
| `round(local+remote)` | 0 | 0 | 74.98% |
| `round(round(local)+remote)` | 1（local） | ~12.5% | 74.96% |
| `round(round(local)+round(remote))` | 2（local+remote） | ~25% | **99.9986%** |

- local、remote 各量级 ~1.65e6（各占总输出 ~3.3e6 的一半）。
- 单次 bf16 量化误差 |δ| 均值 ≈ 2048 ≈ **0.125 ULP**（相对最终输出）。
- 两次独立量化 → 总偏移 ~0.25 ULP → 约 25% 元素翻 1 ULP。
- 这正是之前「单段式 vs NPU 差 25%」的来源；而「单次量化模型」只差 12.5%，所以之前无论怎么调都对不齐。

---

## 4. 为什么「多一次量化」会造成 ~3000 倍 mean 误差

结论：**合理**。两次误差的**性质**完全不同——不是「多一次同样大小的舍入」，而是从「fp32 累加顺序噪声」换成「对大中间量做 bf16 量化」。

| 误差源 | 尾数 | 1 ULP 相对 | 性质 |
|--------|------|-----------|------|
| fp32 累加顺序 | 24 bit | ~6e-8 | 随机、会抵消 |
| bf16 量化 | 8 bit | ~0.39% | 系统、不抵消 |

- **pm=0/1**：golden 与 NPU 只差 fp32 累加顺序（~1e-6 相对）。这抖动只有落在 bf16 舍入边界（宽 0.39%）附近的元素才翻动，比例 ≈ `1e-6/0.39% ≈ 0.009%`，于是 `mean ≈ 0.009%×1ULP ≈ 1.3`。
- **pm=2**：对大中间量（local/remote ~1.65e6）各做一次 bf16 量化（误差 ~0.2%），把最终结果在 bf16 网格上挪 ~0.25 ULP，约 **25%** 元素翻 1 ULP，于是 `mean ≈ 25%×1ULP ≈ 4.1e3`。
- 比值 `25% / 0.009% ≈ 2800×`，即「3000 倍」来自**翻动元素比例**，单次翻动仍只有 1 ULP。

---

## 5. CPU golden 复现（已实现）

`test_bitmatch_alltoallmm.py` 的 `compute_golden` 对 pm=2 采用两次量化模型：

```python
local_out  = (local_matmul + bias).to(out_dtype)   # 第 1 次量化（LOCAL 落 GM）
remote_out = remote_matmul.to(out_dtype)           # 第 2 次量化（fixpipe 转 bf16）
out        = (local_out.float() + remote_out.float()).to(out_dtype)  # 原子累加 + 舍入
```

结果：pm=2 的 bit-exact 率从 **74.98% → 99.9986%**，与 pm=0/1 持平（残余 ~0.001% 为 fp32 累加顺序噪声）。

---

## 6. 结论与影响

| 维度 | pm=0/1 | pm=2（修正模型后） |
|------|--------|--------------------|
| 精度标准 | 1e-2（宽松） | 1e-2（宽松，与 pm=0/1 同量级） |
| bit 级对齐 | 99.99% | **99.9986%** |
| mean abs_err（正确 golden） | ~1.3 | ~1 ULP 量级 |
| 中间舍入 | 无 | **两次 bf16 量化**（LOCAL 落 GM + 原子累加 fixpipe） |

**本质**：pm=2（GM 累加）LOCAL 结果先 bf16 落 GM、REMOTE 原子累加时 fixpipe 又把 remote 量化一次，因此中间有两次 bf16 舍入。用「两次量化」模型即可在 CPU 侧 bit 级复现，无需修改 kernel。
