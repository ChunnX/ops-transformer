# SageAttention2

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| Ascend 950PR | √ |
| 其他昇腾产品 | 未适配 |

本算子使用 Ascend 950 的 Cube/Vector 数据通路和寄存器向量 API，基于 AscendC 与 CATLASS 实现 INT8 QK、FP8 PV 的前向 Attention。

## 功能说明

计算 `softmax(Q @ Kᵀ * sm_scale) @ V` 的量化近似，提供 PyTorch `sageattn`、`sageattn_qk_int8_pv_fp8_asc` 和两阶段 `aclnnSageAttention2` 接口。

执行分为三个内部算子：

1. **Stats**：计算 K 均值和 V 通道统计；开启 smooth V 时同时计算 V 均值。
2. **Quant**：按 token 将 Q/K 量化为 INT8，将 V 量化为 E4M3 FP8，生成吸收 `sm_scale` 的固定平衡 FP16 scale。
3. **Core**：INT8 QK、online softmax、FP8 PV 与 FP32 O 更新流水执行，最终恢复 V scale 和可选 V 均值。

QK 在 Cube 中累加为 INT32，经固定 `2^-11` DEQF16 转为 FP16，再乘预处理的 Q/K scale；score 与 max 使用 FP16，ExpSub 输出 FP32。P/V 的 FP8 缩放统一为 448，PV、持续累加的 O 和 softmax 分母全部使用 FP32，除法采用默认 AscendC Div。

通用路径采用 M128/N256 分块、四个 score 槽和两个 P/PV 共享槽。D padding 为 128、Sq 为 128 的倍数且 Sk 为 512 的倍数并不少于 4096 时，使用 N512 路径：两个 N256 Cube 片段共享一次 softmax/O 更新，流水跨 query 连续执行。O 常驻 UB，P 经 UB→L1 交给 Cube，QK/PV 经 L0C→UB 交给 Vector。

## 参数说明

输入 Q/K/V 均为四维、同设备、同 dtype 的 FP16 或 BF16 NPU Tensor，最后一维连续。输出 shape/dtype 与 Q 相同。

| 参数 | 类型/默认值 | 说明 |
|---|---|---|
| `q` | Tensor | HND 为 `[B,Hq,Sq,D]`，NHD 为 `[B,Sq,Hq,D]` |
| `k`、`v` | Tensor | HND 为 `[B,Hkv,Sk,D]`，NHD 为 `[B,Sk,Hkv,D]` |
| `tensor_layout` | str，`HND` | 支持 `HND`、`NHD` |
| `is_causal` | bool，`False` | 开启时要求 Sq=Sk |
| `qk_quant_gran` | str，`per_token` | 仅支持 per-token Q/K 量化 |
| `sm_scale` | float/None，`None` | 默认 `1/sqrt(D)`，须可表示为有限 FP32 |
| `pv_accum_dtype` | 兼容参数，`fp32` | 值被忽略，始终采用 FP32 累加 |
| `smooth_k` | bool，`True` | 量化前减去 K 沿序列维的均值 |
| `smooth_v` | bool，`False` | 量化前减去 V 均值，输出时加回 |
| `return_lse` | bool，`False` | 返回 `(out, lse)`；LSE 为 FP32、shape `[B,Hq,Sq]` |

支持 GQA（Hq 必须能被 Hkv 整除）、1≤D≤128、D padding、序列尾块、末维连续的 stride 输入。非因果模式支持 Sq≠Sk。LSE 包含 K smoothing 的行常数补偿。

不支持任意 attention mask、dropout、varlen 和 autograd。

```python
import torch
import torch_npu
from sageattention2 import sageattn

q = torch.randn(1, 4, 4096, 128, dtype=torch.float16, device="npu")
k = torch.randn(1, 2, 4096, 128, dtype=q.dtype, device=q.device)
v = torch.randn_like(k)
out, lse = sageattn(q, k, v, smooth_v=True, return_lse=True)
```

ACLNN 声明见 [aclnn_sage_attention2.h](op_api/aclnn_sage_attention2.h)。先调用 `aclnnSageAttention2GetWorkspaceSize` 获得 workspace 大小和 executor，再调用 `aclnnSageAttention2`。可运行的调用示例见 [benchmark_aclnn.cpp](benchmarks/benchmark_aclnn.cpp)。

## 编译与安装

已验证环境：CANN Toolkit/950 ops **9.2.0-beta.1**、PyTorch **2.9.0**、torch_npu **2.9.0.post8**，CATLASS commit `337cc89254d46f5d6041d117aca64edf7197027a`。Python 依赖见 [requirements-dev.txt](requirements-dev.txt)。

本目录由 ops-transformer 顶层工程构建。`sage_attention2_stats`、`sage_attention2_quant`、`sage_attention2_core` 三个同级目录提供内部算子的注册入口，构建 `sageattention2` 时会一并编译。

在本目录执行；CANN 和 CATLASS 路径按实际环境设置：

```bash
source /usr/local/Ascend/cann/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
export CATLASS_ROOT=/path/to/catlass
export ASCEND_RT_VISIBLE_DEVICES=0

bash install_opp.sh
source build/opp/vendors/custom_transformer/bin/set_env.bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

`install_opp.sh` 调用仓库完整构建流程并安装到本目录的 `build/opp`。PyTorch 默认使用 ACLNN 后端，`SAGEATTN_ACLNN_LIBRARY` 可指定自定义 API 库路径。

原语和中间态测试另需 native 测试库，它与 OPP 使用同一套 kernel 头文件：

```bash
bash build_native.sh
bash benchmarks/build_aclnn_benchmark.sh
SAGEATTN_TEST_ACLNN=1 SAGEATTN_TEST_STANDALONE=1 \
    python -m pytest -q tests -o junit_family=xunit1 --junitxml=build/pytest-results.xml
python tools/run_clang_tidy.py
ruff check .
ruff format --check .
```

`SAGEATTN_BACKEND=native` 可切换到 native 执行路径，`SAGEATTN_LIBRARY` 可指定该测试库。修改 kernel 后须重新完整构建、安装，再执行 NPU 测试。

## 测试说明

清理后的分支已完成 native/OPP 完整构建安装，回归 **675 项通过**（673 项常规通过、2 项下述最大误差例外），clang-tidy 与 Python 检查通过。

测试覆盖 Q/K/V 量化、QK/PV 传输、softmax 中间态、槽位复用、长序列、dtype/layout/stride/GQA/causal、smooth K/V、LSE、公开接口和两阶段 ACLNN。

[原始任务用例](tests/data/task_cases_500.json) 包含 486 条功能/精度/参数用例及 14 条性能用例。执行时将原 per-thread/per-warp 正例映射为 per-token；`pv_accum_dtype` 按兼容参数处理。

数值参考是本目录的 per-token CPU 算法及独立 FP32 SDPA。CPU 同算法检查采用 FP16 rtol/atol=2^-9、BF16=2^-6、至少 99% 匹配比例；最大绝对误差通常分别限于 0.1/1。SDPA 的 max/mean/RMSE 误差比例门限为 2/1.2/1.2。

`functional_4d705280dd` 和 `functional_242c35a5d6` 使用相同的 FP16 causal GQA 输入：Q `[1,16,128,128]`、K/V `[1,8,128,128]`、seed=42。默认 Div 的微小商误差会在 V 的 FP8 舍入中点产生相邻编码差异。两项实测匹配比例为 99.9035%、最大绝对误差为 0.123046875；采用设备量化结果重放 CPU 参考后，最大误差为 0.001953125。仅这两项最大误差上限采用 0.125，范围由 [记录](tests/data/accuracy_waivers.json) 中的完整用例 SHA256 限定，其他检查保留，JUnit 同时记录原始门限和实测值。

上述结果不等同于原始 GPU per-thread/per-warp L-A/L-B 或正式 AscendOpTest 验收；这些对照尚未执行。

## 性能测试

使用同版本 CANN 从 ops-transformer 源码构建 FIA vendor，加载它的环境后运行：

```bash
python benchmarks/run_validation_performance.py \
    --output build/performance \
    --fia-opapi /path/to/fia_vendor/op_api/lib/libcust_opapi.so \
    --fia-source-commit SOURCE_COMMIT
```

默认覆盖必测 P-01～P-07 和 32K/64K/128K；`--cases` 选择用例，`--smooth-v` 开启 V smoothing。每个实现预热 20 次、采样 50 次，先完成全部 Sage，再完成全部 FIA，两接口串行执行。

普通 `msprof` 不加 `op`、不指定 aimetrics。分析工具逐调用相加 Stats+Quant+Core 的 device task duration，再取中位数，并核对 `op_summary` 与 `task_time`。NPU Event 独立采集、单独报告。完整算子的有效 TFLOPS 按两次矩阵乘法的 `4BHqS²D / (time_us × 10⁶)` 计算。

本分支完整重建后复测的普通 msprof 中位数（B=1、Hq=Hkv=32、D=128、FP16、非因果）：

| 接口 | 序列长度 | 完整 Sage ms | FIA ms | 加速比 |
|---|---:|---:|---:|---:|
| PyTorch | 4096 | 0.684 | 0.939 | 1.374× |
| ACLNN | 4096 | 0.665 | 0.974 | 1.465× |
| PyTorch | 16384 | 10.487 | 28.046 | 2.674× |
| ACLNN | 16384 | 10.643 | 27.479 | 2.582× |

FIA 来源为 ops-transformer commit `32d64c27fc4055f1a4fa702ca6d94ef2b1343c96`，使用同一 CANN 构建并显式选择其 API 库。两种计时均已执行，表格统一采用 CSV 口径。

完整交付版本的必测性能仍有场景未达到逐例 1.6× 门禁；长上下文结果不能替代短序列验收。[独立交付仓库](https://gitcode.com/G_W_E/sage-attention2-950pr) 保留 128K 图表、逐例性能与原始测试证据。

## 目录说明

| 目录 | 内容 |
|---|---|
| `op_kernel` | Stats、Quant、Cube/Vector 分块、online softmax 和 O 更新 |
| `op_host`、`op_api` | 参数校验、注册、tiling 和两阶段 ACLNN |
| `python` | PyTorch 适配层、native 测试桥接和 CPU 参考 |
| `native`、`tests` | 原语测试入口及功能/精度回归 |
| `benchmarks`、`tools` | 双接口计时、CSV 审计和 clang-tidy |
