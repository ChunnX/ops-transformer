# SageAttention2

## 产品支持情况

| 产品 | 是否支持 |
|---|:---:|
| Ascend 950PR | √ |
| 其他昇腾产品 | 未适配 |

本算子使用 Ascend 950 的 Cube/Vector 数据通路和寄存器向量 API，基于 AscendC 与 CATLASS 实现 INT8 QK、FP8 PV 的前向 Attention。

## 功能说明

计算 `softmax(Q @ Kᵀ * sm_scale) @ V` 的量化近似，提供 PyTorch `sageattn`、`sageattn_qk_int8_pv_fp8_asc` 和两阶段 `aclnnSageAttention2` 接口。

一次 `SageAttention2` MIX AIC:AIV=1:2 kernel launch 内依次完成三个阶段：

1. **Stats**：计算 K 均值和 V 通道统计；开启 smooth V 时同时计算 V 均值。
2. **Quant**：按 token 将 Q/K 量化为 INT8，将 V 量化为 E4M3 FP8，生成吸收 `sm_scale` 的固定平衡 FP16 scale。
3. **Core**：INT8 QK、online softmax、FP8 PV 与 FP32 O 更新流水执行，最终恢复 V scale 和可选 V 均值。

连续长序列的 Stats 使用两个 128 行 K/V 输入槽，预取下一批输入与当前批计算重叠。V 的绝对值最大值直接在 FP16/BF16 中归约，只将最终通道最大值转成 FP32；K 均值保留原有分片和每通道 FP32 累加顺序。

Stats 内部分片归约和 Stats→Quant 使用 `SyncAll<true>()` 同步所有 AIV；Quant→Core 使用 `SyncAll<false>()` 同步所有 AIC/AIV，确保量化数据写入 GM 后才开始 Attention。预处理按 `2 * cores` 分配 AIV 任务，空闲核也参与屏障。Host 使用 `SetScheduleMode(1)` 的 batch 调度，保证全核同步所需的同时驻留。

dtype 和 wide 通过 `ASCENDC_TPL_ARGS_DECL` 声明为模板参数，Host 用 `GET_TPL_TILING_KEY(input_dtype, wide)` 选择入口。FP16/BF16 与普通/wide 共四种模板组合，tiling key 分别为 1/27/257/283（dtype 占低 8 位，wide 占第 8 位）。每次调用仍只启动一个融合 kernel。`WIDE` 同时传给 Quant 的 V 排列和 Attention 的 N256/N512 路径，融合设备代码不再按 dtype 或 shape 执行这两处分发；公共 `attention.h` 只提供 `Attention<T, WIDE>`。

中间结果仅保存在算子 workspace：K/V 均值、V scale、Q8/K8/V8、FP16 Q/K scale 和 Stats 分片归约缓冲。各区域按 512 字节对齐，tiling 检查大小溢出并加上 CANN 系统 workspace 前缀；每次调用独立分配，无跨调用全局状态。ACLNN 只注册和启动一个融合算子，接口签名不变。

QK 在 Cube 中累加为 INT32，经固定 `2^-11` DEQF16 转为 FP16，再乘预处理的 Q/K scale；score 与 max 使用 FP16，ExpSub 输出 FP32。每个块的 max 在 FP16 中减去 `FP16(ln(448))`，与旧的平移后 max 取最大值；概率、online alpha 和 LSE 共用这一状态，因此省去概率逐元素乘 448，LSE 使用 `shifted_max + log(denom)`。V 的 FP8 缩放为 448，PV、持续累加的 O 和 softmax 分母全部使用 FP32，除法采用默认 AscendC Div。

连续输入的批量量化路径按 64 行处理：Q/K 预先计算每个 token 的 FP32 scale 倒数并保存在独立 UB 区域，V 预先计算每个通道的 FP32 scale 倒数，随后通过乘法完成归一化。供 Attention 使用的原始 scale 保持不变；倒数乘法与直接除法在量化舍入边界处可能存在差异。

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

已验证环境：CANN Toolkit/950 ops **9.2.0-beta.1**、PyTorch **2.9.0**、torch_npu **2.9.0.post8**，CATLASS commit `337cc89254d46f5d6041d117aca64edf7197027a`。构建和测试依赖见 [requirements.txt](requirements.txt)。

本目录 `experimental/attention/sage_attention2` 由 ops-transformer 顶层工程构建（`--ops=sage_attention2`），包含唯一的 `SageAttention2` 注册入口，无额外的同级阶段算子目录。

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

`install_opp.sh` 调用仓库完整构建流程并安装到本目录的 `build/opp`。PyTorch 使用 ACLNN 后端，`SAGEATTN_ACLNN_LIBRARY` 可指定自定义 API 库路径。

测试使用已安装的 ACLNN 融合算子：

```bash
bash benchmarks/build_aclnn_benchmark.sh
SAGEATTN_TEST_STANDALONE=1 \
    python -m pytest -q tests -o junit_family=xunit1 --junitxml=build/pytest-results.xml
```

修改 kernel 后须重新完整构建、安装，再执行 NPU 测试。

## 测试说明

完整回归结果：**641 passed**（包含独立 C++ ACLNN 测试）。CPU 量化参考使用上述 FP16 平移 max 定义及 FP8 饱和转换；独立 SDPA 标准未改变。

测试全部通过公开 ACLNN 路径运行，并对照独立 per-token CPU 算法、FP32 SDPA 或可解析的常数输入结果。覆盖 dtype/layout/stride/GQA/causal、smooth K/V、LSE、长序列、模板选择边界、重复调用、并发流和独立 C++ ACLNN 入口。

[原始任务用例](tests/data/task_cases_500.json) 包含 486 条功能/精度/参数用例及 14 条性能用例。执行时将原 per-thread/per-warp 正例映射为 per-token；`pv_accum_dtype` 按兼容参数处理。

PV 和 smooth V 用例采用 CPU 独立量化的端到端对照，使用下述统一标准，并检查重复调用逐位一致性或独立 SDPA 误差。wide 的 LSE 对照包含 FP16 scale/score 舍入的量化参考。

数值参考是本目录的 per-token CPU 算法及独立 FP32 SDPA。CPU 同算法检查采用 FP16 rtol/atol=2^-9、BF16=2^-6、至少 99% 匹配比例；最大绝对误差通常分别限于 0.1/1。SDPA 的 max/mean/RMSE 误差比例门限为 2/1.2/1.2。

旧版（ExpSub 后乘 448）的 `functional_4d705280dd` 和 `functional_242c35a5d6` 使用相同的 FP16 causal GQA 输入：Q `[1,16,128,128]`、K/V `[1,8,128,128]`、seed=42。默认 Div 的微小商误差会在 V 的 FP8 舍入中点产生相邻编码差异。两项实测匹配比例为 99.9035%、最大绝对误差为 0.123046875。仅这两项最大误差上限采用 0.125，范围由 [记录](tests/data/accuracy_waivers.json) 中的完整用例 SHA256 限定，其他检查保留，JUnit 同时记录原始门限和实测值。

FP16 平移的舍入会改变 P 的实际缩放，因此不保证与原先 `448 * exp(score - max)` 逐位或按旧量化参考一致。零 score、V 全 1 时，FP16 输出为约 0.999512；较大且相等的 score 还可能使 P 超过 448 后饱和。独立 SDPA 检查保留原有标准。12 个随机诊断配置中，RMSE 相对旧版变化约 −0.08%～+0.14%；这不代表所有模型或输入分布的误差上界。

上述结果不等同于原始 GPU per-thread/per-warp L-A/L-B 或正式 AscendOpTest 验收；这些对照尚未执行。

## 性能测试

使用 msprof 统计独立 C++ ACLNN 入口的 kernel duration。SageAttention2 在单个融合 kernel 内完成 Stats、Quant 和 Core，计时包含三个阶段及其内部同步。每个实现预热 **2 次**、采样 **5 次**，报告耗时中位数；加速比为 FIA kernel 耗时除以 SageAttention2 kernel 耗时。

以下 FIA 对照于 2026-09-08 使用已启用 FP16 ln(448) 优化的版本重新采集；优化自身的旧版/新版 A/B 结果见本节末尾。测试环境为 Ascend 950PR、CANN 9.2.0-beta.1，FIA 使用同版本 CANN 构建的 `fia_latest_transformer` vendor。所有用例 Hq=32、D=128，smooth K 开启、smooth V 关闭。Sage 和 FIA 串行执行，每例各采集 7 个 task，丢弃前 2 个；`op_summary` 与 `task_time` 的耗时已交叉核对。

| 用例 | 配置（Hq=32、D=128） | Sage kernel ms | FIA kernel ms | 加速比 |
|---|---|---:|---:|---:|
| P-01 | 4K, B=1, Hkv=32, FP16 | 0.637 | 0.986 | 1.548× |
| P-02 | 8K, B=1, Hkv=32, FP16 | 2.160 | 3.509 | 1.625× |
| P-03 | 8K, B=1, Hkv=32, FP16, causal | 1.321 | 1.869 | 1.414× |
| P-04 | 8K, B=2, Hkv=32, FP16 | 4.390 | 7.845 | 1.787× |
| P-05 | 4K, B=1, Hkv=8, FP16 | 0.514 | 0.953 | 1.852× |
| P-06 | 4K, B=1, Hkv=32, BF16 | 0.661 | 0.899 | 1.359× |
| P-07 | 16K, B=1, Hkv=32, FP16 | 8.354 | 20.057 | 2.401× |
| P-08 | 32K, B=1, Hkv=32, FP16 | 39.517 | 113.981 | 2.884× |
| P-09 | 64K, B=1, Hkv=32, FP16 | 155.217 | 461.081 | 2.971× |
| P-10 | 128K, B=1, Hkv=32, FP16 | 622.917 | 1888.583 | 3.032× |

本轮原始 CSV、应用元数据与交叉核对结果保存在 `build/performance_msprof_ln448_20260908/`，汇总见 `summary.json`。

在本目录加载 CANN、Sage OPP 和待比较的 FIA vendor 后，可按以下方式采集 P-01：

```bash
bash benchmarks/build_aclnn_benchmark.sh
export ASCEND_RT_VISIBLE_DEVICES=0
# 自定义 FIA vendor 需指定其 API 库路径：
# export FIA_OPAPI_LIBRARY=/path/to/fia_vendor/op_api/lib/libcust_opapi.so
mkdir -p build/performance_msprof/P-01
SAGE_PROFILE_ONLY=1 msprof --output=build/performance_msprof/P-01 \
    --application="$(pwd)/build/benchmarks/benchmark_aclnn 4096 1 32 0 0 2 5 $(pwd)/build/performance_msprof/P-01/application.json"
```

`benchmark_aclnn` 参数依次为 `SEQ BATCH KV_HEADS BF16 CAUSAL WARMUP REPEATS OUTPUT_JSON`。按表中配置调整参数即可采集其他用例。统计时分别按执行顺序提取 SageAttention2 和 FIA task，排除各自前 2 次预热后，对剩余 5 次 kernel duration 取中位数。

### Stats 预处理优化后的接口复测

2026-09-09，在 scale 倒数复用基础上加入 Stats 的 16 位最大值归约和双缓冲后，复测 P-01～P-07。所有用例 smooth K 开启、smooth V 关闭。PyTorch 使用公开 API 外的 NPU event；独立 ACLNN 使用 ACL device event，输出/workspace 预分配、GetWorkspaceSize 在计时外。每例、每实现预热 20 次、采样 50 次，取中位数。以下是接口计时，与前表的历史 msprof kernel 计时分别列示。

| 用例 | PyTorch Sage ms | PyTorch FIA ms | 加速比 | ACLNN Sage ms | ACLNN FIA ms | 加速比 |
|---|---:|---:|---:|---:|---:|---:|
| P-01 | 0.6766 | 1.0050 | 1.485× | 0.6493 | 1.0241 | 1.577× |
| P-02 | 2.4217 | 5.8448 | 2.414× | 2.3880 | 6.0154 | 2.519× |
| P-03 | 1.4423 | 2.1258 | 1.474× | 1.3981 | 2.1098 | 1.509× |
| P-04 | 5.8196 | 11.4924 | 1.975× | 5.8012 | 11.8270 | 2.039× |
| P-05 | 0.5827 | 0.9960 | 1.709× | 0.5256 | 0.9487 | 1.805× |
| P-06 | 0.6914 | 0.9388 | 1.358× | 0.6517 | 0.9866 | 1.514× |
| P-07 | 10.8357 | 28.8634 | 2.664× | 10.7714 | 28.4442 | 2.641× |

P-01、P-03、P-06 仍未满足两条接口均 ≥1.6× 的性能门禁。正式 OPP 的两轮 msprof op A/B 对照中，这三项 kernel 耗时分别下降约 2.30%、0.67%、1.16%；独立 Stats 的提速不能直接作为完整算子加速比。全量回归 641 项通过，额外 48 组零、次正规数、有限数值、padding/stride 输入的 K mean 和 V scale 与优化前逐位一致。

## 目录说明

| 目录 | 内容 |
|---|---|
| `op_kernel` | Stats、Quant、Cube/Vector 分块、online softmax 和 O 更新 |
| `op_host`、`op_api` | 参数校验、注册、tiling 和两阶段 ACLNN |
| `python` | PyTorch ACLNN 适配层和 CPU 参考 |
| `tests` | 公开接口功能、精度和多流回归 |
| `benchmarks` | 性能测试脚本和独立 ACLNN 调用示例 |
