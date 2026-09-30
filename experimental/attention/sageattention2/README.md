# SageAttention2 融合注意力算子

SageAttention2 是一款面向 Atlas 950PR 硬件平台，基于 **AscendC + CATLASS** 联合开发的 INT8 $QK^\top$ + FP8 $PV$ 高吞吐融合注意力算子。算子严格对齐开源算法规范（[thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) `sageattn_qk_int8_pv_fp8_cuda`）与任务书参数契约，提供 PyTorch 与 aclnn 双路径交付，并在各场景全面对标官方性能基线 FIA（`FusedInferAttentionScore`）。

---

## 1. 产品支持情况

- **硬件平台支持**：
  - **Atlas 950PR**：原生支持（适用芯片架构为 Ascend 950 系列，算子执行单元为 1 AIC Cube + 2 AIV Vector 异构协同）。
  - **其他昇腾产品**：未适配。

---

## 2. 架构设计与微架构亮点

本算子充分释放 Atlas 950PR 芯片的硬件微架构潜力，在设备侧实现了多项原创优化：

1. **AscendC 与 CATLASS 深度融合分工**：
   - **CATLASS**：封装 Cube GEMM-1（INT8 $\times$ INT8 $\to$ INT32）与 GEMM-2（FP8 $\times$ FP8 $\to$ FP32/FP16）的 Mmad 循环、Tile 迭代器与 $16 \times 16$ NZ 分形排布映射；
   - **AscendC**：负责 Vector 向量化动态对称量化、Key/Value 均值平滑、Online Softmax 归约更新、FastReciprocal 零除法指令逼近及 Tri-Queue 核心异步流水线。
2. **片上三级内存规划与随路转换**：
   - 严格管控 **L0C (256 KB)**、**UB (512 KB，静态预算 248 KB 高水位线)** 与 **L1 (512 KB)** 内存层次；
   - 依托硬件 Fixpipe 单元实现 L0C $\to$ UB 随路反量化与 $16 \times 16$ NZ 分形排布到连续 ND 排布的在线转换，消灭片内格式重排开销。
3. **片上流式即时量化（On-Chip Streaming Quantization，仅宽路径）**：
   - 宽路径（N=512）下 Query 矩阵采用块级片上在线流式量化，消除 Query 中间低比特激活的 GM 写回往返；标准路径仍为两阶段量化写 GM workspace（收敛计划见 todo.md 优化项 a）。
4. **同源 Normalizer（Homologous Normalizer）**：
   - Softmax 分母累加项严格采用量化后并反量化的离散码字进行行求和，使归一化因子与 PV 矩阵乘法消费的有效概率完全同源，消除长序列下的概率质量损失。
5. **蛇形波前折叠调度（Snake-Folded Wavefront Balancing）**：
   - 针对因果掩码下三角拓扑，偶数 Head 正向推进，奇数 Head 反向回扫，使物理核心任务复合负载呈首尾互补均衡分布（数值测试断言口径：方差降低 ≥10×、CV ≤ 0.05；真机实测待填）。

---

## 3. 接口规范与参数说明

### 3.1 统一自动分发入口 `sageattn`

```python
from typing import Any, Optional, Union, Tuple
import torch

def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

#### 参数说明
- `q`（`torch.Tensor`，必填）：Query 张量。`HND` 布局为 `[B, Hq, Sq, D]`；`NHD` 布局为 `[B, Sq, Hq, D]`。
- `k`（`torch.Tensor`，必填）：Key 张量。`HND` 布局为 `[B, Hk, Sk, D]`；`NHD` 布局为 `[B, Sk, Hk, D]`。
- `v`（`torch.Tensor`，必填）：Value 张量。shape 与 `k` 相同约定。
- `tensor_layout`（`str`，默认 `"HND"`）：张量排布格式，支持 `"HND"` 或 `"NHD"`。
- `is_causal`（`bool`，默认 `False`）：是否应用因果掩码；仅当 `Sq == Sk` 时合法。
- `sm_scale`（`Optional[float]`，默认 `None`）：Softmax 缩放系数；默认取 `1.0 / sqrt(head_dim_og)`。
- `return_lse`（`bool`，默认 `False`）：是否返回 LogSumExp 归一化对数因子（用于 Ring Attention 等场景）。
- `**kwargs`（`Any`）：透传参数。支持程序化覆盖 `pv_accum_dtype` 与 `qk_quant_gran`。

#### 自动分发与覆盖行为（任务书 §2.4）
- **默认分发目标**：在 Atlas 950PR 上固定路由至 `sageattn_qk_int8_pv_fp8_asc` 原生实现（默认 `pv_accum_dtype="fp32+fp16"`, `qk_quant_gran="per_thread"`）；
- **配置覆盖规则**：遵循「**显式 kwargs > 环境变量 > 默认值**」优先级：
  - `SAGEATTN_PV_ACCUM_DTYPE`：可选覆盖为 `"fp32+fp16"`、`"fp32+fp32"` 或 `"fp32"`；
  - `SAGEATTN_QK_QUANT_GRAN`：可选覆盖为 `"per_thread"` 或 `"per_warp"`；
  - 传入非法参数或非法环境变量值时，均严格拦截并抛出 `ValueError`。

---

### 3.2 显式原生接口 `sageattn_qk_int8_pv_fp8_asc`

对标开源 `sageattn_qk_int8_pv_fp8_cuda`，参数列表、默认值与返回值语义逐项对齐：

```python
def sageattn_qk_int8_pv_fp8_asc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
```

#### 参数说明
- `q` / `k` / `v`（`torch.Tensor`，必填）：输入张量，dtype 须同为 `float16` 或 `bfloat16`，同设备，末维连续。
- `tensor_layout`（`str`，默认 `"HND"`）：`"HND"` 或 `"NHD"`。
- `is_causal`（`bool`，默认 `False`）：因果掩码；仅当 `Sq == Sk` 时合法。
- `qk_quant_gran`（`str`，默认 `"per_thread"`）：Q/K 量化算法粒度，支持 `"per_thread"` 或 `"per_warp"`。
- `sm_scale`（`Optional[float]`，默认 `None`）：Softmax 缩放系数；`None` 时取 `1.0 / sqrt(head_dim_og)`。
- `pv_accum_dtype`（`str`，默认 `"fp32+fp16"`）：PV 累加模式：`"fp32+fp16"`（默认验收配置）、`"fp32+fp32"` 或 `"fp32"`。
- `smooth_k`（`bool`，默认 `True`）：开启 Key 序列均值平滑；GQA 下自动执行广播校正。
- `smooth_v`（`bool`，默认 `False`）：Value 平滑；仅在 `pv_accum_dtype="fp32"` 时生效，其他累加模式下发出 Warning 并安全忽略。
- `return_lse`（`bool`，默认 `False`）：是否返回 LSE；返回 shape 为 `[B, Hq, Sq]`，自然对数域。
- `**kwargs`（`Any`）：仅允许 `attn_mask=None` 与 `dropout_p=0.0`，传入其他值抛出异常。

#### 返回值契约
- 当 `return_lse=False`：返回注意力输出张量 `out`，shape 与 `q` 相同，dtype 与 `q` 相同；
- 当 `return_lse=True`：返回元组 `(out, lse)`。`lse` 为自然对数域的 $\log \sum_j \exp(\cdot)$，shape `[B, Hq, Sq]`，dtype `float32`。内核已产出自然对数域值，框架层只补回 smooth-K 居中项：
  $$\text{lse}_{\text{final}} = \text{lse}_{\text{kernel}} + \left(\text{lse\_correction} \cdot \sigma\right) \quad (\text{当 } \text{smooth\_k}=\text{True})$$
  任务书把内核值写作 log2 域并要求框架侧 `/1.44269504`；本实现把该换底折叠进内核，**对外返回值不变**（仍为自然对数域真值），只是不再存在中间的 log2 表示。
  **接口差异**：`aclnn` 入口返回内核值本身（不含 `lse_correction`）。该项需要一次 $Q\bar K^{\top}$ 跨头归约，且 Q 分片在 LSE 退休时已不在 UB 中，放入内核会引入 $O(B\cdot H_q\cdot S_q)$ 的 GM 中转区；`smooth_k=True` 且需要真值 LSE 的调用方应使用 PyTorch 入口（`sageattn` / `sageattn_qk_int8_pv_fp8_asc`），精度门禁 TC-04 亦按任务书口径在 PyTorch 层判定。

---

### 3.3 ACLNN 底层两段式 C 接口

符合 CANN 9.2.0-beta.1 标准两段式调用规范（定义位于 `op_api/aclnn_sage_attention2.h`）：

```c
#include "aclnn/acl_meta.h"

// 第一阶段：查询 Workspace 字节数并构造执行器
ACL_FUNC_VISIBILITY aclnnStatus aclnnSageAttention2GetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const char *tensorLayout,
    bool isCausal,
    const char *qkQuantGran,
    double smScale,
    const char *pvAccumDtype,
    bool smoothK,
    bool smoothV,
    bool returnLse,
    const aclTensor *out,
    const aclTensor *lse,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 第二阶段：流式发射执行算子
ACL_FUNC_VISIBILITY aclnnStatus aclnnSageAttention2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

---

## 4. 约束条件与边界拦截

- **输入数据类型**：Q/K/V 必须为相同的数据类型，仅支持 `torch.float16` 与 `torch.bfloat16`；
- **设备与布局**：张量必须位于同一昇腾 NPU 设备；末维物理连续（`stride(-1) == 1`）；
- **特征维度（Head Dim）**：支持原始特征维度 $D_{\text{orig}} \in (0, 128]$：
  - $D_{\text{orig}} \le 64$：内部自动填充至 64；
  - $64 < D_{\text{orig}} \le 128$：内部自动填充至 128；
  - $D_{\text{orig}} > 128$：算子立即报错拦截；
  - 输出端自动将填充维度裁剪剥离，无缝还原为 $D_{\text{orig}}$；
- **GQA 支持**：强制要求 $H_Q$ 能被 $H_{KV}$ 整除（$H_Q \% H_{KV} == 0$）；
- **序列长度**：非因果模式支持 $S_Q \ne S_K$；因果模式（`is_causal=True`）强制要求 $S_Q == S_K$；
- **不支持特性**：不支持任意显式注意力掩码（`attn_mask` 须为 `None`）、不支持 Dropout、不支持 Varlen 打包序列、不支持反向求导（纯前向算子）。

---

## 5. 编译与安装指南

### 5.1 依赖环境
- CANN Toolkit: **9.2.0-beta.1**
- PyTorch: **2.1.0** 及以上
- Ascend Extension for PyTorch: **torch_npu**
- CATLASS: **master 分支**（适配 Ascend 950 硬件架构）

### 5.2 构建与部署步骤

在仓库根目录执行顶层构建：

```bash
# 1. 导入 CANN 环境变量
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
export CATLASS_ROOT=/path/to/catlass

# 2. 编译 SageAttention2 自定义算子 OPP 包
bash build.sh --pkg --soc=ascend950 --ops=sageattention2 --build-type=Release -O3

# 3. 安装生成的自定义算子包
source build/opp/vendors/custom_transformer/bin/set_env.bash

# 4. 导入 Python 适配层路径
export PYTHONPATH="$PWD/experimental/attention/sageattention2/python${PYTHONPATH:+:$PYTHONPATH}"
```

---

## 6. 测试与运行指导

测试套件位于 `tests/` 目录，详见 [tests/README.md](tests/README.md)：

```bash
# 1. 运行 TC-01 至 TC-10 全量功能验收矩阵
python3 -m unittest experimental/attention/sageattention2/tests/test_functional_matrix.py -v

# 2. 运行 L-A / L-B 双层精度门禁测试
python3 -m unittest experimental/attention/sageattention2/tests/test_precision_dual_level.py -v

# 3. 运行宿主机可移植 C++ 内核数值规格测试 (17 大类 43 项断言)
g++ -O2 -std=c++17 experimental/attention/sageattention2/tests/numeric_error_test.cpp -o /tmp/numeric_error_test
/tmp/numeric_error_test

# 4. 运行 P-01 至 P-07 性能对比测试 (Atlas 950PR 真机)
# (1) aclnn 路径
g++ -O2 -std=c++17 \
    -I$ASCEND_HOME_PATH/include \
    -I$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/include \
    experimental/attention/sageattention2/tests/benchmark_aclnn.cpp \
    -o /tmp/benchmark_aclnn -L$ASCEND_HOME_PATH/lib64 -lascendcl -lopapi
/tmp/benchmark_aclnn --warmup 50 --repeats 100 --output aclnn_perf.json

# (2) PyTorch 路径
python3 experimental/attention/sageattention2/tests/benchmark_pytorch_runner.py pytorch_perf.json
```

---

## 7. 工程目录结构说明

```text
experimental/attention/sageattention2/
├── op_kernel/                              # 设备侧 Kernel 实现
│   ├── sageattention2.cpp                  # 算子入口模板特化
│   ├── platform_traits.h                   # 950PR 硬件常量与内存水线 (UB 248KB 等)
│   ├── sage_attention2_tiling_data.h       # Tiling 数据结构与参数定义
│   ├── dynamic_symmetric_quant.h           # INT8 对称动态量化与平滑统计 (AscendC)
│   ├── online_reduction_softmax.h          # 在线 Softmax、同源 Normalizer 与 LSE 导出
│   ├── delayed_fp16_gemm.h                 # CATLASS Cube GEMM-1 与 GEMM-2 模板
│   ├── catlass_fractal_packer.h            # CATLASS NZ 分形排布装配原语
│   ├── single_pass_engine.h                # 标准路径单趟流式引擎
│   ├── decoupled_tri_queue.h               # 宽路径 Tri-Queue 空间解耦流水引擎
│   ├── snake_wavefront_balancer.h          # 蛇形波前折叠调度器 (因果负载均衡)
│   ├── credit_handshake.h                  # 异构核心轻量级信号量握手
│   └── math_approximations.h               # FastReciprocal 零除法指令逼近工具
├── op_host/                                # Host 侧参数校验、属性注册与 Tiling
│   ├── sage_attention2_def.cpp             # 原型定义与参数属性
│   └── sage_attention2_tiling.cpp          # 硬件拓扑感知、动态分块决策
├── op_api/                                 # C API 导出层 (aclnn 规范)
│   ├── aclnn_sage_attention2.h             # aclnn 接口声明
│   └── aclnn_sage_attention2.cpp           # 两段式调用封装与图构建
├── python/                                 # PyTorch 前端与适配层
│   └── sageattention2/
│       ├── __init__.py                     # 顶层符号导出
│       ├── core.py                         # 对齐开源 core.py 接口
│       ├── _ops.py                         # Dispatcher 注册、计划缓存与环境变量覆写
│       └── reference.py                    # 算法参考基准实现
├── tests/                                  # 自动化测试与性能基准套件
│   ├── README.md                           # 测试执行与 GPU Golden 导出复现文档
│   ├── test_functional_matrix.py           # TC-01 ~ TC-10 功能矩阵测试
│   ├── test_precision_dual_level.py        # L-A / L-B 双层精度门禁测试
│   ├── numeric_error_test.cpp              # C++17 内核数值规格一致性测试
│   ├── benchmark_aclnn.cpp                 # P-01 ~ P-07 aclnn C++ 性能压测
│   └── benchmark_pytorch_runner.py         # P-01 ~ P-07 PyTorch 路径性能压测（真机环境运行）
└── CMakeLists.txt                          # 算子编译脚本
```

---

## 8. 交付合规与社区验收材料

本算子严格遵循社区开源规范交付至 `cann/ops-transformer` 仓库：
1. **交付件 1（算子设计文档）**：按 cann-competitions 模板撰写并以 PR 形式提交评审合入（仓库内不留存设计文档副本）；
2. **交付件 2（自测用例与测试代码）**：配套 `tests/` 目录完整测试矩阵，复现步骤见 [tests/README.md](tests/README.md)；
3. **交付件 3（自测报告）**：按验收模板填写用例参数、精度对比与性能实测数据（随验收材料提交，仓库内不留存）；
4. **交付件 4（待验收代码）**：个人代码仓分支按标准规范组织，并邀请社区账号 `Ascend-CANN` 作为开发者协同评审。
