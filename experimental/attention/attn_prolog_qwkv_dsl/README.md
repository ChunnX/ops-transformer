# AttnPrologQwkv

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

`attn_prolog_qwkv` 是融合 attention 前处理算子。一次调用同时完成：

- **Q 路**：MXFP8 GEMM（`qr @ wqb^T`）→ reshape 成 `(T, N, D)` → 对尾部 `Dr` 维做 inplace RoPE → 沿 `D` 做 MXFP4 量化（数据 E2M1，scale E8M0）。
- **W 路**：BF16 GEMM（`x @ ww^T`）再乘标量 `softmax_scale`，输出 FP32。

当前实现只覆盖冻结几何 `dim=5120, q_lora=1280, N=32, D=128, Dr=64`；可变轴是 token 数 `T`。`T < 4096` 时按 head 轴再切，把 32 个 AIC 填满；`T ≥ 4096` 只切 `T`。W 路由每个 T-tile 的 `h_block==0` 核作为 leader 计算，其余核只跑 Q。

Q 路公式：

```
A[t,k] = dequant(qr)[t,k]
B[n,k] = dequant(wqb)[n,k]
Y[t,n] = Σ_k A[t,k] * B[n,k]          # n 跑遍 N*D，FP32 累加
Q0     = reshape(Y, (T, N, D))
Q_fp32 = rope_inplace(Q0, sin, cos)   # 只改最后 Dr 列
q, descale_q = mxfp4_quant(Q_fp32)    # group=32，沿 D
```

W 路公式：

```
w[t,n] = softmax_scale * Σ_k float(x[t,k]) * float(ww[n,k])
```

## 参数说明

```python
attn_prolog_qwkv(
    x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos,
    *,
    softmax_scale,
    q=None,
    descale_q=None,
    w=None,
) -> tuple[Tensor, Tensor, Tensor]
```

前 8 个参数为位置 Tensor；`softmax_scale` 为必选 keyword。可选 `q` / `descale_q` / `w` 传入时原地写入，否则内部分配。

### 维度说明

| 命名 | 含义 | 当前范围 |
| :--- | :--- | :--- |
| `T` | token 数（`b*s` 展平） | 动态，且大于 0 |
| `dim` | `x` / `ww` 的隐层宽 | 冻结 5120 |
| `q_lora` | Q LoRA 宽 | 冻结 1280 |
| `N` | head 数 | 冻结 32 |
| `D` | head dim | 冻结 128 |
| `Dr` | RoPE 长度，作用在 `D` 的尾部 | 冻结 64 |

### 输入和属性

| 参数名 | 可选/必选 | 描述 | 数据类型 | 维度 |
| :--- | :--- | :--- | :--- | :--- |
| `x` | 必选 | W 路激活 | `bfloat16` | `(T, dim)` |
| `qr` | 必选 | Q 路激活，MXFP8 E4M3 的 uint8 字节视图 | `uint8` | `(T, q_lora)` |
| `wqb` | 必选 | Q 路权重，存储布局为 `(N*D, q_lora)` | `uint8` | `(N*D, q_lora)` |
| `ww` | 必选 | W 路权重 | `bfloat16` | `(N, dim)` |
| `descale_qr` | 必选 | `qr` 的 paired E8M0 scale | `uint8` | `(T, ceil(q_lora/64), 2)` |
| `descale_wqb` | 必选 | `wqb` 的 paired E8M0 scale | `uint8` | `(N*D, ceil(q_lora/64), 2)` |
| `rope_sin` | 必选 | RoPE sine，HF / DeepSeek rotate-half | `float32` | `(T, Dr)` |
| `rope_cos` | 必选 | RoPE cosine | `float32` | `(T, Dr)` |
| `softmax_scale` | 必选 | W 路 GEMM 之后的标量 | `float` | - |
| `q` | 可选 | 预分配的量化 Q 输出 | `uint8` | `(T, N, D/2)` |
| `descale_q` | 可选 | 预分配的 Q scale 输出 | `uint8` | `(T, N, ceil(D/64), 2)` |
| `w` | 可选 | 预分配的 W 输出 | `float32` | `(T, N)` |

MX scale 每 32 个元素共享一个 E8M0；公开存储把相邻两个 scale 绑成 pair，故最后一维为 2。`q` 为 OCP E2M1 打包：低 nibble = 偶数下标 `D[2i]`，高 nibble = 奇数下标 `D[2i+1]`。

### 输出

| 参数名 | 描述 | 数据类型 | 形状 |
| :--- | :--- | :--- | :--- |
| `q` | 量化后的 Q，逻辑 FP4 `(T, N, D)` | `uint8` | `(T, N, D/2)` |
| `descale_q` | Q 的 E8M0 scale | `uint8` | `(T, N, ceil(D/64), 2)` |
| `w` | W 路结果，已乘 `softmax_scale` | `float32` | `(T, N)` |

## 约束说明

- 当前仅支持冻结几何 `dim=5120, q_lora=1280, N=32, D=128, Dr=64`。
- `D`、`Dr` 必须为偶数；`Dr ≤ D`。NPU kernel 还要求 `D` 是 64 的倍数、`q_lora` MX 对齐。
- `x` 与 `qr` 的 `T` 必须一致；`wqb.shape[0]` 必须等于 `N * D`。
- `softmax_scale` 必须显式传入，无默认值。
- CPU 张量走 golden；NPU 张量走融合 kernel。`Plan` 不支持的几何回退 golden。
- 当前未提供 GE 图转换实现。

## 调用说明

依赖已配置的 CANN、PyTorch、torch_npu 和 CANNBotDSL 环境。通过仓库 torch_extension 构建流程安装 `cann_ops_transformer` 后使用：

| 调用方式 | 入口 | 说明 |
| :--- | :--- | :--- |
| PyTorch API | `cann_ops_transformer.ops.ds41` | 与 compressor_v2 / MQSMLA 共用 `ds41` 模块 |
| torch.ops | `torch.ops.cann_ops_transformer.ds41.attn_prolog_qwkv` | 导入扩展完成注册后调用 |
| DSL 直调 | 本目录 `torch_extension/op.py` | 同序参数；可选 keyword `q` / `descale_q` / `w` |

```python
import torch
import torch_npu
from cann_ops_transformer.ops.ds41 import attn_prolog_qwkv


q, descale_q, w = attn_prolog_qwkv(
    x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos,
    softmax_scale=1.0 / (128 ** 0.5),
)
```

本算子首次合入时把 CANNBotDSL host / kernel / golden 一并放在 `torch_extension/`，随 `cann_ops_transformer` wheel 打包。后续若迁到 net wheel，torch_extension 可改回只做注册和接口适配。

## 测试说明

测试资产全部位于 [tests](tests/)。从 ops-transformer 根目录执行：

```bash
# CPU golden，不需要 NPU
python3 -m pytest -q experimental/attention/attn_prolog_qwkv_dsl/tests/test_golden.py

# Plan / LUT / AOT 编译（需要 cannbotdsl）
python3 -m pytest -q experimental/attention/attn_prolog_qwkv_dsl/tests/test_kernel_compile.py

# raw-VF 量化数学逐位对齐
python3 -m pytest -q experimental/attention/attn_prolog_qwkv_dsl/tests/test_vf_math.py

# 真机精度（需要 CANN + NPU）
source /usr/local/Ascend/cann/set_env.sh   # 或本机等价路径
python3 -m pytest -q experimental/attention/attn_prolog_qwkv_dsl/tests/test_npu_precision.py

# 全部
bash experimental/attention/attn_prolog_qwkv_dsl/tests/test.sh
```

安装 wheel 后也可通过 `torch.ops.cann_ops_transformer.ds41.attn_prolog_qwkv` 调用同一入口：

```bash
bash build.sh --torch_extension
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
```

| 测试 | 覆盖 |
| :--- | :--- |
| `test_golden.py` | CPU golden 的接口 / 数值契约 |
| `test_kernel_compile.py` | `Plan` 预算、head 切分、冻结用例 AOT 编译 |
| `test_vf_math.py` | raw-VF 整数量化算法逐位对齐 golden |
| `test_npu_precision.py` | 真机 `q` / `descale_q` 逐位、`w` 容差对比 |

`q` 与 `descale_q` 必须逐位对齐；`w` 为 BF16 GEMM 的 FP32 归约，使用 `rtol=atol=2e-2`。冻结用例 T=4096 上 chip MFU 约 75%。
