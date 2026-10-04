# PyTorch 接入说明

本文面向在 Atlas 950PR 真机上把 SageAttention2（INT8 QK^T + FP8 PV，仅前向）接入 PyTorch 应用的开发者，给出安装顺序、最小调用示例、SDPA 热替换与图模式注意事项。所有事实以本目录 `./README.md`、`./python/sageattention2/` 源码与 `./op_api/aclnn_sage_attention2.h` 为准；CUDA 上游语义差异见 [与 CUDA 默认接口的差异](./CUDA接口差异说明.md)。

## 环境准备与安装顺序

- 硬件与基础软件：Atlas 950PR（`ascend950`）；CANN Toolkit（已验证 `9.2.0-beta.1`）；torch 与 torch_npu（Ascend Extension for PyTorch）
- 注入 CANN 环境变量（路径以实际安装位置为准）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

- 构建并安装算子 OPP 包（二选一）：

```bash
# 方式一：仓库整包出 run 包（在 ops-transformer 仓库根执行），安装 run 包后含 ACLNN 算子库 libsageattention2.so
cd ./ops-transformer
bash build.sh --pkg --experimental --soc=ascend950 --vendor_name=customize --ops=sageattention2 -j16

# 方式二：容器门禁出包并本地部署（在算子目录执行；build_opp.sh 在 BiSheng 容器内编译，install_opp.sh 只做部署）
bash run.sh opp
```

- 导出算子运行环境变量（`install_opp.sh` 默认部署到算子本地 `./build/opp_installed`，`--export-env` 输出可直接 eval 的注入命令，含 `ASCEND_CUSTOM_OPP_PATH` 与 `LD_LIBRARY_PATH` 追加项）：

```bash
eval "$(./install_opp.sh --export-env)"
```

- 接入 Python 包（按源码路径接入；算子根目录另含可选 `setup.py`，默认同样走 JIT 编译，`BUILD_NPU_EXT=1` 时改为 AOT 编译同一桥接扩展）：将 `./python` 加入 `PYTHONPATH`；首次 NPU 调用会经 `torch.utils.cpp_extension` JIT 编译 C++ 桥接扩展（依赖 `ASCEND_HOME_PATH` 指向的 CANN 与 torch_npu 头库，编译缓存由 torch 扩展缓存托管；`SAGEATTN_BUILD_VERBOSE=1` 可输出编译日志）
- 自检（应能正常导入并列出两个入口函数）：

```bash
PYTHONPATH=./python python3 -c "import torch, torch_npu; import sageattention2 as s; print(s.sageattn, s.sageattn_qk_int8_pv_fp8_asc)"
```

- 设备侧功能验收入口（TC-01..TC-10 用例矩阵，宿主机无 NPU 时自动跳过）：

```bash
SAGEATTN_TEST_NPU=1 python3 -m pytest tests/test_npu_standalone.py -v
```

## 接口速览

- 入口一（自动分发，NPU 上固定走 FP8 PV）：

```python
sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None,
         return_lse=False, **kwargs)
```

- 入口二（显式全参，对标开源 `sageattn_qk_int8_pv_fp8_cuda`，函数名改为 `*_asc`）：

```python
sageattn_qk_int8_pv_fp8_asc(q, k, v, tensor_layout="HND", is_causal=False,
                            qk_quant_gran="per_thread", sm_scale=None,
                            pv_accum_dtype=None, smooth_k=True, smooth_v=False,
                            return_lse=False, **kwargs)
```

- 返回值契约：`return_lse=False`（默认）单张量返回 `o`；`return_lse=True` 返回 `(o, lse)` 二元组
- `o`：shape/布局与 `q` 相同，dtype 与 `q` 相同（pad 的 head_dim 已裁回原值）
- `lse`：FP32，shape `[B, Hq, Sq]`，为 `QK^T * sm_scale` 各行的自然域 logsumexp（smooth_k 校正语义对齐开源 `core.py` 返回路径）
- `tensor_layout`：`"HND"` 为 `[B, H, S, D]`，`"NHD"` 为 `[B, S, H, D]`
- `sm_scale=None` 时自动取 `1.0 / sqrt(head_dim)`（pad 前的原始 head_dim）
- `pv_accum_dtype` 合法取值与覆盖优先级：kwargs 显式传参 高于 环境变量 `SAGEATTN_PV_ACCUM_DTYPE` 高于 默认 `"fp32+fp16"`（默认验收档）；非法取值直接抛 `ValueError`，无静默回退

## 调用示例

- 基础调用（HND，默认参数即 `fp32+fp16` 交付档）：

```python
import torch
import torch_npu
import sageattention2

q = torch.randn(1, 32, 4096, 128, dtype=torch.float16, device="npu")
k = torch.randn(1, 32, 4096, 128, dtype=torch.float16, device="npu")
v = torch.randn(1, 32, 4096, 128, dtype=torch.float16, device="npu")

out = sageattention2.sageattn(q, k, v)                 # [1, 32, 4096, 128] fp16
```

- NHD 布局与 LSE：

```python
q = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="npu")
k = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="npu")
v = torch.randn(1, 1024, 8, 128, dtype=torch.bfloat16, device="npu")

out, lse = sageattention2.sageattn(q, k, v, tensor_layout="NHD", return_lse=True)
# out: [1, 1024, 8, 128] bf16；lse: [1, 8, 1024] fp32
```

- GQA 与因果（显式入口，全参数）：

```python
q = torch.randn(1, 32, 4096, 128, dtype=torch.float16, device="npu")   # Hq = 32
k = torch.randn(1,  8, 4096, 128, dtype=torch.float16, device="npu")   # Hkv = 8
v = torch.randn(1,  8, 4096, 128, dtype=torch.float16, device="npu")

out = sageattention2.sageattn_qk_int8_pv_fp8_asc(
    q, k, v,
    tensor_layout="HND",
    is_causal=True,                 # 仅 Sq == Sk 时合法
    qk_quant_gran="per_thread",     # 或 "per_warp"
    pv_accum_dtype="fp32+fp16",     # 默认验收档，显式写出
    smooth_k=True,
    return_lse=False,
)
```

- 经环境变量覆盖默认累加档（对全部未显式传参的调用生效）：

```bash
export SAGEATTN_PV_ACCUM_DTYPE="fp32"    # 或 "fp32+fp32" / "fp32+fp16"
```

- 非因果跨长度（Sq 不等于 Sk）合法；因果与跨长度组合会被 `ValueError` 拒绝

## SDPA 热替换（plug-and-play）

工具位于 `./python/sageattention2/monkey_patch.py`，随包导入后即可使用：

```python
import torch
import sageattention2

sageattention2.replace_sdpa()          # 全局热替换 F.scaled_dot_product_attention
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
sageattention2.restore_sdpa()          # 恢复还原

# 或作用域式（离开 with 自动还原，支持嵌套）：
with sageattention2.sdpa_patch(pv_accum_dtype="fp32+fp16"):
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

- `replace_sdpa()` 支持参数：`pv_accum_dtype`、`qk_quant_gran`、`smooth_k`、`smooth_v`、`fallback_to_original`（默认 `True`）
- `fallback_to_original=True` 时，不满足算子约束的调用（非空 `attn_mask`、`dropout_p > 0`、非 4 维、dtype 非 FP16/BF16、`requires_grad=True`、末维非连续、GQA 不整除、因果且 Sq 不等于 Sk 等）自动回退原始 PyTorch SDPA；置 `False` 则直接抛 `ValueError`
- `is_sdpa_patched()` 查询当前是否处于热替换态；`restore_sdpa(prev)` 可精确还原到 `replace_sdpa()` 的返回值

## 约束与错误行为

- 输入：4 维 q/k/v，同 device（同一 NPU）、同 dtype（FP16 或 BF16）、末维连续（`stride(-1) == 1`）
- head_dim：原始 `D ∈ [1, 128]`；`D < 64` 内部 pad 到 64，`64 < D < 128` pad 到 128，输出裁回原值；`D > 128` 抛 `ValueError`
- GQA：`Hq` 能被 `Hkv` 整除，且 `Hq / Hkv <= 8`（交付验收域上限；上限值定义于 `./sageattn_fp8/ascendc/tiling.h` 的 `MAX_GQA_RATIO`，超出时 Host Tiling 显式拒绝并报错）
- 因果：`is_causal=True` 仅当 `Sq == Sk`
- `attn_mask`：不支持；经 kwargs 传入非 None 值（含 SDPA 形态位置实参误传张量）抛 `ValueError`；`dropout_p` 非 0 同样拒绝
- 未知 kwargs：抛 `TypeError`
- `smooth_v`：仅在 `pv_accum_dtype="fp32"` 档真实生效（V 减均值 + 输出补偿正路径）；在 `"fp32+fp32"` / `"fp32+fp16"` 档发 `UserWarning` 后忽略
- 仅前向：不支持反向与 autograd
- 传非法参数一律显式报错，不存在静默降级

## 环境变量汇总

- `SAGEATTN_PV_ACCUM_DTYPE`：`sageattn` 入口 `pv_accum_dtype` 默认值覆盖
- `SAGEATTN_ALLOW_CPU_STUB`：CPU 张量恒走高保真 CPU 参考实现（torch.library CPU 后端；输出为 CPU 参考结果，禁止作为 NPU 精度与性能验收证据），置 `1`/`true`/`yes` 时该路径附带 `UserWarning`（默认不告警）。NPU 张量始终走 PrivateUse1 真机算子，不存在 CPU 回退
- `SAGEATTN_NUM_CORES`：Host Tiling 参与调度的 AI Core 数覆盖（如 28 核减配拓扑）
- `SAGEATTN_BUILD_VERBOSE`：置 `1` 输出 C++ 扩展 JIT 编译日志
- `SAGE_OPAPI_LIBRARY` / `FIA_OPAPI_LIBRARY`：仅 `./benchmarks/` 基准工具 dlsym 外部算子库时的路径

## 图模式注意事项

- 算子以 `torch.library` 自定义算子注册（`torch.ops.sageattention2.forward`），含 Meta 实现，支持 FakeTensor 推理 shape（`./tests/test_torch_extension.py` 验证 `torch.compile(backend="eager", fullgraph=True)` 整图捕获无断图）
- 建议先以 eager 模式跑若干次 warmup：首次 NPU 调用触发 C++ 桥接 JIT 编译与 aclnn 算子初始化，宜在进图前完成
- 仅前向推理；不要在需要梯度的图里使用
- 其他图模式（torchair 等下沉编译）未在本交付中验证，不作承诺

## 与 aclnn 接口的关系

- PyTorch 入口经 `torch.ops.sageattention2.forward`（PrivateUse1 后端）→ C++ 桥接 `./python/sageattention2/csrc/attention.cpp` → 两段式 `aclnnSageAttention2`（第一段 `GetWorkspaceSize`，第二段执行）
- C/C++ 调用方直接使用交付态 `./op_api/aclnn_sage_attention2.h`（字符串属性 ABI）；完整两段式调用示例见 `./benchmarks/benchmark_aclnn.cpp`
- 单算子 Profiling（PyTorch 路径，torch_npu.profiler/kineto 采集 + chrome trace，附 msprof 附加方式说明）：`python3 benchmarks/profile_single_op.py --case P-01`；aclnn/C++ 路径性能对比用 `./benchmarks/benchmark_aclnn.cpp`，PyTorch 路径对 FIA 的性能门禁用 `./benchmarks/run_validation_performance.py`
