# Cumprod 算子需求规格（SPEC.md）

> 来源：`ops-registry-invoke-glacier/docs/cumprod/REQUIREMENTS.md` 与 `spec.yaml`（Stage 1 输入材料，已验证存在）。
> 目标平台：Ascend 950PR / Ascend 950DT。实现方式：pypto-pro（PyPTO Pro tile 级 DSL）kernel + AscendC 自定义算子工程（ops-transformer 仓）。

## 1. 算子定义

| 项 | 内容 |
|---|---|
| 算子名 | Cumprod（op_file: `cumprod`） |
| 语义 | 沿 `dim` 轴的 inclusive / forward 累积乘积：`y[i0..ik..] = prod(x[..., 0..k, ...], axis=dim)` |
| 公式 | `y = np.cumprod(x, axis=dim)` |
| 范式 | Recurrence（沿扫描轴 inter-element 顺序依赖，禁止乱序合并） |
| exclusive / reverse | 固定 false / false（对齐官方 aclnnCumprod 接口面，不暴露） |

## 2. 接口

- 输入 `x`：ND 布局，rank 1~8。
- 属性 `dim`：Int（REQUIRED）。支持负轴：`dim += rank(x)`；取值 `[-rank, rank)`，越界报错。
- 输出 `y`：shape 与 dtype 均与 `x` 完全一致。
- dtype 支持范围（spec.yaml 全集 12 种）：本版本收敛为 **float16 / bfloat16 / float32**（对齐 pypto 内置 cumprod 的 dtype 集与关键 shape 场景；整型为后续扩展项，见 §6 已知范围）。

## 3. 数值口径（spec.yaml AP-007）

- fp16/bf16：链乘累加器拓宽到 **fp32**（cast 加载 → fp32 累乘 → 逐元素写回时 cast 回原 dtype），与 golden（torch CPU acc_type=fp32 逐元素写回）语义一致。
- fp32：直乘。
- 溢出/下溢：浮点链乘上溢 Inf / 下溢 0 属数学语义本身，kernel 不做稳定化。
- 确定性：沿扫描轴严格顺序累乘，同 dtype + fp32 拓宽 → 同设备多次执行 bitwise 一致。

## 4. 精度容差（来源 spec.yaml numerical_tolerance）

| dtype | rtol | atol | metric |
|---|---|---|---|
| float32 | 1.0e-4 | 1.0e-4 | max_relative |
| float16 | 2.0e-3 | 2.0e-3 | max_relative |
| bfloat16 | 4.0e-3 | 4.0e-3 | max_relative |

（并行分段扫描改变乘法结合顺序 → 非 bitwise，用相对容差。）

## 5. 关键 shape 场景（REQUIREMENTS §7.4）

| # | shape | dtype | axis | 场景 |
|---|---|---|---|---|
| 1 | [1000] | fp32 | 0 | DDPM 调度系数 |
| 2 | [256, 365] | fp32 | 1（末轴） | 生存分析 |
| 3 | [64, 2048] | fp32 | 1（末轴） | RL 折扣链 |
| 4 | [32, 1000, 64] | bf16 | 1（中间轴） | 批 HMM 概率链 |
| 5 | [4096] | fp16 | 0 | 1D 长链 |
| 6 | [8, 65536] | fp32 | 1（末轴） | 长序列极限 |

边界：dim 轴长度 1（恒等）、负 dim、非末轴扫描、空 Tensor（host 层拦截）、dim 越界报错、全 1 / 全 0 / 含 NaN / 含 +inf 极值。

## 6. 交付范围与已知约束

- 交付：op_host（def / tiling / infershape / CMakeLists）、op_kernel（pypto-pro `cumprod.py`）、TTK 用例（GE 通路 + aclnn 通路）、文档。
- 本版本 dtype 收敛为 fp16/bf16/fp32；整型（int8~uint64）留待后续版本（spec.yaml 全集为对齐目标）。
- 单 line 单核串行 chunk 扫描（1D 超长链）为 v1 已知性能边界，正确性优先。
